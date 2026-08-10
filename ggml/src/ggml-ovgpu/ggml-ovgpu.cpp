#include "ggml-ovgpu.h"
#include "ggml-ovgpu.hpp"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <intel_gpu/runtime/device_query.hpp>
#include <intel_gpu/runtime/engine.hpp>
#include <intel_gpu/runtime/memory.hpp>
#include <ocl/ocl_engine.hpp>
#include <ocl/ocl_memory.hpp>

void ggml_ovgpu_smoke_test(cldnn::engine & engine);

// one cldnn engine shared by the whole backend (PoC: single device)
struct ggml_backend_ovgpu_reg_context {
    std::shared_ptr<cldnn::engine> engine;
    cldnn::stream_ptr             stream; // blocking service stream for host<->device copies
    std::vector<ggml_backend_dev_t> devices;
};

struct ggml_backend_ovgpu_device_context {
    int         index;
    std::string name;
    std::string description;
    ggml_backend_ovgpu_reg_context * reg_ctx;
};

struct ggml_backend_ovgpu_buffer_type_context {
    ggml_backend_ovgpu_reg_context * reg_ctx;
    std::string                    name;
};

struct ggml_backend_ovgpu_buffer_context {
    cldnn::memory_ptr mem;
    void *            base; // USM device pointer (real pointer, arithmetic-safe)

    ~ggml_backend_ovgpu_buffer_context() {
        mem.reset();
    }
};

struct ggml_backend_ovgpu_context {
    ggml_backend_ovgpu_reg_context * reg_ctx;
    ggml::ovgpu::op_cache            op_cache;
};

static std::shared_ptr<cldnn::engine> ggml_ovgpu_create_engine() {
    cldnn::device_query query(cldnn::engine_types::ocl, cldnn::runtime_types::ocl);
    auto devices = query.get_available_devices();
    GGML_ASSERT(!devices.empty() && "OVGPU: no OpenCL GPU device found");
    return cldnn::engine::create(cldnn::engine_types::ocl, cldnn::runtime_types::ocl, devices.begin()->second);
}

//
// buffer
//

static void ggml_backend_ovgpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    delete (ggml_backend_ovgpu_buffer_context *) buffer->context;
}

static void * ggml_backend_ovgpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ((ggml_backend_ovgpu_buffer_context *) buffer->context)->base;
}

static void ggml_backend_ovgpu_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_ovgpu_buffer_context * ctx = (ggml_backend_ovgpu_buffer_context *) buffer->context;
    ggml_backend_ovgpu_reg_context * reg_ctx = ((ggml_backend_ovgpu_buffer_type_context *) buffer->buft->context)->reg_ctx;
    size_t dev_off = (char *) tensor->data - (char *) ctx->base + offset;
    // Direct device write of ONLY this tensor's bytes. copy_from(stream, src, src_off,
    // dst_off, size) does a clEnqueueMemcpyINTEL of `size` bytes at dev_off - O(size),
    // not O(buffer). (buffer_clear uses the same path safely.)
    // Previously this used mem_lock<read_write> over the ENTIRE model buffer per
    // tensor -> full device->host->device round-trip of ~2GB per weight tensor ->
    // 100+ s model load. copy_from touches only the tensor's region.
    ctx->mem->copy_from(*reg_ctx->stream, data, 0, dev_off, size, true /*blocking*/);
}

static void ggml_backend_ovgpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_ovgpu_buffer_context * ctx = (ggml_backend_ovgpu_buffer_context *) buffer->context;
    ggml_backend_ovgpu_reg_context * reg_ctx = ((ggml_backend_ovgpu_buffer_type_context *) buffer->buft->context)->reg_ctx;
    size_t dev_off = (char *) tensor->data - (char *) ctx->base + offset;
    // Direct device read of only this tensor's bytes (copy_to = copy_from into the
    // host ptr). Avoids locking/mapping the whole buffer.
    ctx->mem->copy_to(*reg_ctx->stream, data, dev_off, 0, size, true /*blocking*/);
}

static void ggml_backend_ovgpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    // PoC: clear by copying a zero-filled host buffer. (USM fill not wired yet.)
    ggml_backend_ovgpu_buffer_context * ctx = (ggml_backend_ovgpu_buffer_context *) buffer->context;
    ggml_backend_ovgpu_reg_context * reg_ctx = ((ggml_backend_ovgpu_buffer_type_context *) buffer->buft->context)->reg_ctx;
    std::vector<uint8_t> zeros(buffer->size, value);
    ctx->mem->copy_from(*reg_ctx->stream, zeros.data(), 0, 0, buffer->size, true);
}

static const struct ggml_backend_buffer_i ggml_backend_ovgpu_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_ovgpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_ovgpu_buffer_get_base,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_ovgpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_ovgpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_ovgpu_buffer_clear,
    /* .reset           = */ NULL,
};

//
// buffer type
//

static const char * ggml_backend_ovgpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return ((ggml_backend_ovgpu_buffer_type_context *) buft->context)->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_ovgpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_ovgpu_buffer_type_context * buft_ctx = (ggml_backend_ovgpu_buffer_type_context *) buft->context;
    cldnn::engine & engine = *buft_ctx->reg_ctx->engine;

    // USM device memory: real pointer (arithmetic-safe) for get_base; this iGPU
    // only supports usm_device (not host-accessible usm_shared), so host access
    // goes through copy_from/copy_to (GPU memcpy) in set/get_tensor.
    cldnn::layout layout{ov::element::u8, cldnn::format::bfyx, cldnn::tensor(1, 1, (cldnn::tensor::value_type) size, 1)};
    cldnn::memory_ptr mem = engine.allocate_memory(layout, cldnn::allocation_type::usm_device, false);

    void * base = mem->buffer_ptr();
    GGML_ASSERT(base != nullptr && "OVGPU: USM device allocation returned null (is USM supported?)");

    if (getenv("GGML_OVGPU_DEBUG")) {
        fprintf(stderr, "[ovgpu] alloc_buffer size=%zu base=%p bytes_count=%zu\n", size, base, mem->size());
    }

    ggml_backend_ovgpu_buffer_context * ctx = new ggml_backend_ovgpu_buffer_context{mem, base};
    return ggml_backend_buffer_init(buft, ggml_backend_ovgpu_buffer_interface, ctx, size);
}

static size_t ggml_backend_ovgpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;
    GGML_UNUSED(buft);
}

static const struct ggml_backend_buffer_type_i ggml_backend_ovgpu_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_ovgpu_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_ovgpu_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_ovgpu_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL,
    /* .get_alloc_size   = */ NULL,
    /* .is_host          = */ NULL,
};

//
// backend (stream)
//

static const char * ggml_backend_ovgpu_get_name(ggml_backend_t backend) {
    return ((ggml_backend_ovgpu_device_context *) backend->device->context)->name.c_str();
}

static void ggml_backend_ovgpu_free(ggml_backend_t backend) {
    delete (ggml_backend_ovgpu_context *) backend->context;
    delete backend;
}

static enum ggml_status ggml_backend_ovgpu_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_ovgpu_context * ctx = (ggml_backend_ovgpu_context *) backend->context;
    cldnn::engine & engine = *ctx->reg_ctx->engine;
    ctx->op_cache.engine = &engine;
    // All op networks share reg_ctx->stream (one in-order queue) - set_tensor
    // writes, every execute, and get_tensor reads all land on it in order, so no
    // per-op finish() is needed. (set_tensor/get_tensor use blocking copy_from/to,
    // so they're already committed on the stream before/after executes.)
    ctx->op_cache.stream = ctx->reg_ctx->stream;
    const bool debug       = getenv("GGML_OVGPU_DEBUG") != nullptr;
    const bool fuse_disable = getenv("GGML_OVGPU_FUSE_DISABLE") != nullptr;
    if (debug) fprintf(stderr, "[ovgpu] graph_compute n_nodes=%d\n", cgraph->n_nodes);

    // Bind a ggml tensor's storage (zero-copy view over the backend USM buffer at the
    // tensor's byte offset) into a cldnn network input/output slot. Shared by the
    // per-op and chain paths.
    auto bind_input = [&engine](cldnn::network & net, const std::string & id, const ggml_tensor * t,
                                const cldnn::layout & lay) {
        GGML_ASSERT(t->buffer && "OVGPU: input not in backend buffer");
        auto * buf_ctx = (ggml_backend_ovgpu_buffer_context *) t->buffer->context;
        size_t off = (char *) t->data - (char *) buf_ctx->base;
        net.set_input_data(id, ggml::ovgpu::wrap_tensor(engine, *buf_ctx->mem, lay, off));
    };
    auto bind_output = [&engine](cldnn::network & net, const std::string & id, const ggml_tensor * t,
                                 const cldnn::layout & lay) {
        auto * buf_ctx = (ggml_backend_ovgpu_buffer_context *) t->buffer->context;
        size_t off = (char *) t->data - (char *) buf_ctx->base;
        net.set_output_memory(id, ggml::ovgpu::wrap_tensor(engine, *buf_ctx->mem, lay, off));
    };

    // --- Fusion chain detection (Tier B-lite). ---
    // Find maximal linear chains [head, child1, ..., childN] where head is MUL_MAT
    // (FC) or RMS_NORM (rms) and each non-final node's output has exactly ONE
    // consumer (the next node) - so the intermediate is absorbable (cldnn fuses it
    // away, the ggml dst is never materialized). Children are ADD/MUL (eltwise) in
    // phase 1. build_chain is the authority on shape/dtype eligibility; if it rejects,
    // chain_fail_cache prevents retry and we fall back to per-op at execute time.
    struct chain_info {
        std::vector<int> node_idx;  // [head, child1, ...] indices into cgraph->nodes
    };
    std::vector<chain_info> chain_at(cgraph->n_nodes);  // [i] populated iff i is a head
    std::vector<bool>       is_chain_head(cgraph->n_nodes, false);
    std::vector<bool>       absorbed(cgraph->n_nodes, false);

    if (!fuse_disable) {
        // node_index[t] = cgraph index of compute node t (leafs/weights are absent).
        // consumers[t]  = node indices that list t in their src[].
        std::unordered_map<const ggml_tensor *, int>              node_index;
        std::unordered_map<const ggml_tensor *, std::vector<int>> consumers;
        for (int i = 0; i < cgraph->n_nodes; i++) {
            node_index[cgraph->nodes[i]] = i;
            for (int k = 0; k < GGML_MAX_SRC; k++) {
                const ggml_tensor * s = cgraph->nodes[i]->src[k];
                if (s) {
                    consumers[s].push_back(i);
                }
            }
        }
        for (int i = 0; i < cgraph->n_nodes; i++) {
            const ggml_tensor * node = cgraph->nodes[i];
            if (node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_RMS_NORM) {
                continue;
            }
            // skip metadata / zero-element heads (mirror the main-loop skips)
            if (node->op == GGML_OP_NONE || ggml_nelements(node) == 0) {
                continue;
            }
            std::vector<int>    chain = { i };
            const ggml_tensor * cur = node;
            while (true) {
                auto it = consumers.find(cur);
                if (it == consumers.end() || it->second.size() != 1) {
                    break;  // 0 or >1 consumers -> cur is materialized (chain ends)
                }
                int                 next_idx = it->second[0];
                const ggml_tensor * next = cgraph->nodes[next_idx];
                if (absorbed[next_idx]) {
                    break;  // already claimed by another chain (DAG -> shouldn't happen)
                }
                const bool next_eltwise = (next->op == GGML_OP_ADD || next->op == GGML_OP_MUL);
                const bool next_silu =
                    (next->op == GGML_OP_UNARY && ggml_get_op_params_i32(next, 0) == GGML_UNARY_OP_SILU);
                if (!next_eltwise && !next_silu) {
                    break;  // only eltwise (ADD/MUL) and SILU children fuse
                }
                // AVAILABILITY: the child's external operand (the non-chain src) must be
                // already materialized when the chain runs at the head's position i. A
                // leaf (weight/input) always is; a compute node is only if its cgraph
                // index < i (so graph_compute executed it before reaching the head).
                // This rejects diamond patterns like ((mm1+mm2)+mm3) where a child's
                // external operand is a sibling matmul computed AFTER the head.
                const ggml_tensor * ext = nullptr;
                for (int k = 0; k < GGML_MAX_SRC; k++) {
                    if (next->src[k] && next->src[k] != cur) {
                        ext = next->src[k];
                        break;
                    }
                }
                if (ext) {
                    auto eit = node_index.find(ext);
                    if (eit != node_index.end() && eit->second >= i) {
                        break;  // ext is a compute node not yet materialized at i
                    }
                    // else: leaf (always available) or compute node before i (available)
                }
                chain.push_back(next_idx);
                cur = next;
            }
            if (chain.size() >= 2) {
                chain_at[i] = { chain };
                is_chain_head[i] = true;
                for (size_t c = 1; c < chain.size(); c++) {
                    absorbed[chain[c]] = true;
                }
            }
        }
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (debug) fprintf(stderr, "[ovgpu]   loop i=%d\n", i);
        ggml_tensor * node = cgraph->nodes[i];

        if (node->op == GGML_OP_NONE) {
            continue; // leaf / no-op; data already lives in the buffer
        }
        switch (node->op) {
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                continue;  // pure metadata (ne/nb only); no kernel to launch
            default:
                break;
        }

        // Zero-element output. A multi-ubatch decode's non-final ubatch has
        // n_outputs==0, so the last-layer ggml_get_rows(cur, inp_out_ids) gets an
        // ids tensor with ne[0]==0 (models/llama.cpp last-layer output gather),
        // and every op downstream of it (FFN mul_mat, logits mul_mat, ...) also
        // produces a 0-element dst. cldnn's reshape / oneDNN FC reject zero-count
        // buffers ("output layout count ... not equal to input layout count(=0)")
        // and throw, which surfaces as llama_decode -3. A 0-element op writes
        // nothing, so skip it entirely - the same semantics as the CPU backend,
        // whose per-element threads simply do no work on an empty tensor. Only the
        // real outputs of the ubatch (K/V into the cache via SET_ROWS, which is
        // non-zero) are needed and still run.
        if (ggml_nelements(node) == 0) {
            continue;
        }

        // Absorbed into a fusion chain -> computed by the chain head's execute.
        if (absorbed[i]) {
            if (debug) fprintf(stderr, "[ovgpu]   node %d op=%d (absorbed)\n", i, node->op);
            continue;
        }

        if (debug) fprintf(stderr, "[ovgpu]   node %d op=%d\n", i, node->op);

        // --- Fusion chain path. ---
        bool ran_chain = false;
        if (is_chain_head[i]) {
            std::vector<const ggml_tensor *> cnodes;
            cnodes.reserve(chain_at[i].node_idx.size());
            for (int idx : chain_at[i].node_idx) {
                cnodes.push_back(cgraph->nodes[idx]);
            }
            auto co = ctx->op_cache.get_or_build_chain(cnodes);
            if (co) {
                // Bind the chain's external inputs (head operands + each child's
                // non-chain operand, e.g. the residual / gamma), in the builder's
                // declared order. input_src_map[k] = (chain-node-idx, src-idx).
                for (size_t k = 0; k < co->input_ids.size(); k++) {
                    int               ni = co->input_src_map[k].first;
                    int               si = co->input_src_map[k].second;
                    const ggml_tensor * src = cnodes[ni]->src[si];
                    bind_input(*co->net, co->input_ids[k], src, co->input_layouts[k]);
                }
                // Bind the chain's single output (the final node's dst).
                const ggml_tensor * final_node = cnodes.back();
                bind_output(*co->net, co->output_id, final_node, co->layout_for(final_node));
                co->net->execute();
                ran_chain = true;
            } else {
                // build_chain rejected this chain (or it threw) -> fall back to per-op
                // for the head AND un-absorb the members so the loop runs them per-op.
                for (size_t c = 1; c < chain_at[i].node_idx.size(); c++) {
                    absorbed[chain_at[i].node_idx[c]] = false;
                }
            }
        }
        if (ran_chain) {
            continue;
        }

        // --- Per-op (Tier A) path. ---
        auto co = ctx->op_cache.get_or_build(node);
        if (!co) {
            GGML_LOG_ERROR("OVGPU: no translator for op %d (%s)\n", node->op, ggml_op_name(node->op));
            return GGML_STATUS_FAILED;
        }
        if (debug) fprintf(stderr, "[ovgpu]   built\n");

        // Bind inputs in the translator's declared order, SKIPPING absent (NULL)
        // optional srcs. Translators list input_ids only for the srcs that exist
        // (e.g. SOFT_MAX with sinks but no mask: src[0]=scores, src[1]=NULL(mask),
        // src[2]=sinks -> input_ids={"in_0","in_2"}); binding positionally would
        // hand the NULL mask to the sinks slot. So walk src[0..], skip NULL, and
        // bind each present src to the next input_id in turn.
        {
            size_t ii = 0;
            for (int k = 0; k < GGML_MAX_SRC && ii < co->input_ids.size(); k++) {
                const ggml_tensor * src = node->src[k];
                if (!src) {
                    continue;  // optional src absent (e.g. SOFT_MAX mask / ROPE ff)
                }
                if (debug) fprintf(stderr, "[ovgpu]   bind in %s (src[%d])\n", co->input_ids[ii].c_str(), k);
                bind_input(*co->net, co->input_ids[ii], src, co->layout_for(src));
                ii++;
            }
        }
        if (debug) fprintf(stderr, "[ovgpu]   inputs bound\n");

        // bind output in-place into the node's tensor
        bind_output(*co->net, co->output_id, node, co->layout_for(node));

        co->net->execute();
        // NO per-op finish(): all networks share reg_ctx->stream (in-order), so op
        // N's output is committed before op N+1's execute reads it. The caller's
        // get_tensor (blocking copy_to on the same stream) is ordered after the last
        // execute, so it sees committed data without an explicit finish here.
    }

    return GGML_STATUS_SUCCESS;
}

static const struct ggml_backend_i ggml_backend_ovgpu_interface = {
    /* .get_name             = */ ggml_backend_ovgpu_get_name,
    /* .free                 = */ ggml_backend_ovgpu_free,
    /* .set_tensor_async     = */ NULL,
    /* .get_tensor_async     = */ NULL,
    /* .set_tensor_2d_async  = */ NULL,
    /* .get_tensor_2d_async  = */ NULL,
    /* .cpy_tensor_async     = */ NULL,
    /* .synchronize          = */ NULL,
    /* .graph_plan_create    = */ NULL,
    /* .graph_plan_free      = */ NULL,
    /* .graph_plan_update    = */ NULL,
    /* .graph_plan_compute   = */ NULL,
    /* .graph_compute        = */ ggml_backend_ovgpu_graph_compute,
    /* .event_record         = */ NULL,
    /* .event_wait           = */ NULL,
    /* .graph_optimize       = */ NULL,
};

static ggml_guid_t ggml_backend_ovgpu_guid() {
    static ggml_guid guid = {0x4f, 0x56, 0x47, 0x50, 0x55, 0x00, 0xa1, 0xb2, 0xc3, 0xd4, 0xe5, 0xf6, 0x07, 0x18, 0x29, 0x3a};
    return &guid;
}

static ggml_backend_t ggml_backend_ovgpu_device_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_ovgpu_device_context * dev_ctx = (ggml_backend_ovgpu_device_context *) dev->context;

    ggml_backend_t backend = new ggml_backend{
        /* .guid    = */ ggml_backend_ovgpu_guid(),
        /* .iface   = */ ggml_backend_ovgpu_interface,
        /* .device  = */ dev,
        /* .context = */ new ggml_backend_ovgpu_context{dev_ctx->reg_ctx},
    };
    return backend;
}

//
// device
//

static const char * ggml_backend_ovgpu_device_get_name(ggml_backend_dev_t dev) {
    return ((ggml_backend_ovgpu_device_context *) dev->context)->name.c_str();
}

static const char * ggml_backend_ovgpu_device_get_description(ggml_backend_dev_t dev) {
    return ((ggml_backend_ovgpu_device_context *) dev->context)->description.c_str();
}

static void ggml_backend_ovgpu_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_ovgpu_device_context * dev_ctx = (ggml_backend_ovgpu_device_context *) dev->context;
    auto * ocl_engine = dynamic_cast<cldnn::ocl::ocl_engine *>(dev_ctx->reg_ctx->engine.get());
    cl_ulong mem = 0;
    clGetDeviceInfo(ocl_engine->get_cl_device()(), CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, nullptr);
    *total = (size_t) mem;
    *free  = (size_t) mem; // PoC: no free-memory query yet
    GGML_UNUSED(dev_ctx);
}

static enum ggml_backend_dev_type ggml_backend_ovgpu_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;
    GGML_UNUSED(dev);
}

static void ggml_backend_ovgpu_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_ovgpu_device_get_name(dev);
    props->description = ggml_backend_ovgpu_device_get_description(dev);
    props->type        = ggml_backend_ovgpu_device_get_type(dev);
    ggml_backend_ovgpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
    };
}

static ggml_backend_buffer_type_t ggml_backend_ovgpu_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_ovgpu_device_context * dev_ctx = (ggml_backend_ovgpu_device_context *) dev->context;
    static ggml_backend_buffer_type buft;
    static std::once_flag flag;
    std::call_once(flag, [&]() {
        buft = ggml_backend_buffer_type{
            /* .iface   = */ ggml_backend_ovgpu_buffer_type_interface,
            /* .device  = */ dev,
            /* .context = */ new ggml_backend_ovgpu_buffer_type_context{dev_ctx->reg_ctx, GGML_OVGPU_NAME},
        };
    });
    return &buft;
}

static bool ggml_backend_ovgpu_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    // Mirror what the translators can actually express. Anything not handled here
    // returns false -> the scheduler splits the graph to CPU.
    auto ok_t = [](ggml_type t) { return t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_BF16; };

    switch (op->op) {
        case GGML_OP_NONE:  // leaf tensor (weight / input) - supported if its type is
            // I32/I64 leaves are indices (e.g. GET_ROWS / SET_ROWS), not a compute
            // dtype - allow them through as plain data, distinct from ok_t.
            return ok_t(op->type) || op->type == GGML_TYPE_I32 || op->type == GGML_TYPE_I64;
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            // Pure metadata (no compute): data already lives in the buffer, only
            // ne/nb change. test-backend-ops queries supports_op on every tensor
            // in the graph (not just compute-op nodes), so these must be allowed
            // through or an otherwise-supported op reading such a view/reshape
            // (e.g. RMS_NORM's non-contiguous test case) would be rejected here
            // first. graph_compute skips them the same way as GGML_OP_NONE.
            return true;
        case GGML_OP_MUL_MAT: {
            const ggml_tensor * w = op->src[0];
            const ggml_tensor * a = op->src[1];
            if (!w || !a) return false;
            if (!ok_t(w->type) || !ok_t(a->type)) return false;
            // weight must be 2D: a group dim (ne[2]/ne[3] > 1) is the GQA-style
            // weight broadcast, which a single 2D-weight FC cannot express -> CPU.
            if (ggml_n_dims(w) != 2) return false;
            // operands must be contiguous, a "flat padded view" (k_v sub-box), OR a
            // "strided view" (genuine permute/transpose - e.g. the decomposed
            // attention's permuted KV-cache q/k/v). build_mul_mat compacts strided
            // operands (permute+reorder+reshape) before FC.
            auto ok_mm_operand = [](const ggml_tensor * t) {
                return ggml_is_contiguous(t) || ggml::ovgpu::ggml_is_flat_padded_view(t) ||
                       ggml::ovgpu::ggml_is_strided_view(t);
            };
            if (!ok_mm_operand(w) || !ok_mm_operand(a)) {
                return false;
            }
            if (w->ne[0] != a->ne[0]) return false; // K must match
            // mixed dtypes allowed: f32/f16/bf16 x f32/f16/bf16 -> f32 dst.
            return true;
        }
        case GGML_OP_ADD:
        case GGML_OP_MUL:
            {
                // dst = src[0] (+|*) src[1], dst type == src[0]. Both contiguous runtime
                // activations. ggml allows ggml_can_repeat (b TILES into a - a.ne[i] %
                // b.ne[i]==0), but cldnn eltwise uses NUMPY broadcast (one operand's dim
                // must be 1). So accept only the subset that reduces to NUMPY broadcast:
                // for each dim, a->ne[i]==b->ne[i] (match) or b->ne[i]==1 (b broadcasts).
                // The tiling case (b->ne[i]>1 but <a->ne[i]) is NOT supported -> CPU.
                const ggml_tensor * a = op->src[0];
                const ggml_tensor * b = op->src[1];
                if (!a || !b) {
                    return false;
                }
                if (!ok_t(a->type) || !ok_t(b->type)) {
                    return false;
                }
                // Contiguous, or a "strided view" - a sub-box view and/or a permute/
                // transpose (e.g. test_bin_bcast's perm1) - see ggml_is_strided_view.
                if (!ggml_is_contiguous(a) && !ggml::ovgpu::ggml_is_strided_view(a)) {
                    return false;
                }
                if (!ggml_is_contiguous(b) && !ggml::ovgpu::ggml_is_strided_view(b)) {
                    return false;
                }
                if (!ggml_can_repeat(b, a)) {
                    return false;
                }
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    if (a->ne[i] != b->ne[i] && b->ne[i] != 1) {
                        return false;
                    }
                }
                return true;
            }
        case GGML_OP_RMS_NORM:
            {
                // dst = src0 / sqrt(mean(src0^2, ne[0]) + eps). No src[1] (gamma is a
                // separate ggml MUL). dst type == src0. Contiguous, or a "padded view"
                // (strided sub-box, e.g. ggml_view_4d) - see ggml_is_padded_view.
                const ggml_tensor * a = op->src[0];
                if (!a) {
                    return false;
                }
                if (!ok_t(a->type)) {
                    return false;
                }
                if (!ggml_is_contiguous(a) && !ggml::ovgpu::ggml_is_padded_view(a)) {
                    return false;
                }
                return true;
            }
        case GGML_OP_SOFT_MAX:
            {
                // dst = softmax(src0*scale + slope*mask) over ne[0], with optional
                // ALiBi (max_bias>0 -> per-head slope) and sinks (src[2], per-head
                // max-prior). Two paths: plain (no mask/sinks/ALiBi/scale) -> cldnn
                // softmax; otherwise the OVGPU_SOFT_MAX_KERNEL custom_gpu port of
                // ggml-cpu softmax (handles ggml's modular nr23 mask broadcast
                // i02%ne12/i03%ne13, per-head ALiBi slope, and sinks). a contiguous;
                // mask contiguous f16/f32 with ne[0]==a.ne[0] && ne[1]==a.ne[1] (cpu
                // indexes mask by i01 directly); sinks f32 [a.ne[2]].
                const ggml_tensor * a = op->src[0];
                if (!a) {
                    return false;
                }
                if (!ok_t(a->type)) {
                    return false;
                }
                if (!ggml_is_contiguous(a)) {
                    return false;
                }
                const ggml_tensor * mask = op->src[1];
                if (mask) {
                    if (mask->type != GGML_TYPE_F32 && mask->type != GGML_TYPE_F16) {
                        return false;
                    }
                    if (!ggml_is_contiguous(mask)) {
                        return false;
                    }
                    if (mask->ne[0] != a->ne[0] || mask->ne[1] != a->ne[1]) {
                        return false;  // mask dim0/dim1 must match a (modular broadcast
                                      // is only over ne[2]/ne[3])
                    }
                }
                const ggml_tensor * sinks = op->src[2];
                if (sinks) {
                    if (sinks->type != GGML_TYPE_F32 || !ggml_is_contiguous(sinks)) {
                        return false;
                    }
                    if (sinks->ne[0] != a->ne[2]) {
                        return false;  // one sink per head (a.ne[2])
                    }
                }
                return true;
            }
        case GGML_OP_GET_ROWS:
            {
                // dst = table[indices] (row gather). dst always f32, indices always
                // I32 (ggml_get_rows asserts). Scope: 2D table only - a grouped/
                // batched table (ne[2]>1, the GQA-style case) is deferred to CPU,
                // same as MUL_MAT's weight-broadcast deferral. bf16 excluded: the
                // clDNN gather kernel selector has no bf16 kernel and aborts
                // (ov::AssertFailure) instead of failing gracefully - verified via
                // test-backend-ops crash investigation, see WORKLOG.
                const ggml_tensor * table = op->src[0];
                const ggml_tensor * ids   = op->src[1];
                if (!table || !ids) {
                    return false;
                }
                if (!ok_t(table->type)) {
                    return false;
                }
                if (table->type == GGML_TYPE_BF16) {
                    return false;
                }
                if (ids->type != GGML_TYPE_I32) {
                    return false;
                }
                if (table->ne[2] != 1 || table->ne[3] != 1) {
                    return false;
                }
                if (!ggml_is_contiguous(table) || !ggml_is_contiguous(ids)) {
                    return false;
                }
                return true;
            }
        case GGML_OP_CPY:
            {
                // dst = copy of src[0] into src[1] (node is a view of src[1]=b).
                // Types may differ (cast f32/f16/bf16/i32). Scope: same ne[] (a
                // reshape across axis decompositions deferred), contiguous a and b
                // (permuted src / strided-dst views deferred to CPU).
                const ggml_tensor * a = op->src[0];
                const ggml_tensor * b = op->src[1];
                if (!a || !b) {
                    return false;
                }
                auto cpy_t = [](ggml_type t) {
                    return t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_BF16 || t == GGML_TYPE_I32;
                };
                if (!cpy_t(a->type) || !cpy_t(b->type)) {
                    return false;
                }
                // Cast-DOWN to bf16 (f32/f16 -> bf16) diverges from ggml's
                // round-to-nearest-even (cldnn's bf16 cast rounds differently,
                // NMSE ~1e-5 > 1e-6) -> offload to CPU. Same-type bf16 copy and
                // cast FROM bf16 (exact) are kept.
                if (b->type == GGML_TYPE_BF16 && a->type != GGML_TYPE_BF16) {
                    return false;
                }
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    if (a->ne[i] != b->ne[i]) {
                        return false;
                    }
                }
                if (!ggml_is_contiguous(a) && !ggml::ovgpu::ggml_is_strided_view(a)) {
                    return false;
                }
                // Strided/permuted bf16 src produces wrong results (cldnn's
                // permute/reorder kernel path mis-handles bf16 + padding - f16/f32
                // strided work, bf16 strided does not). Offload bf16 strided to CPU.
                if (!ggml_is_contiguous(a) && a->type == GGML_TYPE_BF16) {
                    return false;
                }
                if (!ggml_is_contiguous(b)) {
                    return false;  // strided dst (dst_alloc / permute_dst) deferred
                }
                return true;
            }
        case GGML_OP_CONT:
            {
                // dst = contiguous copy of src[0] (ggml_cont: fresh contiguous dst,
                // same type as src). Reuses build_cpy (src[1]=NULL). src contiguous
                // (flat copy) or strided/padded view (permute+reorder); strided bf16
                // -> CPU (same cldnn bf16+padding quirk as CPY). dst is always
                // contiguous. This is the most frequent op in the llama graph (ggml
                // inserts it to ensure contiguity before matmuls) - supporting it
                // removes a large number of CPU splits.
                const ggml_tensor * a = op->src[0];
                if (!a) return false;
                auto cpy_t = [](ggml_type t) {
                    return t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_BF16 || t == GGML_TYPE_I32;
                };
                if (!cpy_t(a->type)) return false;
                if (!ggml_is_contiguous(a) && !ggml::ovgpu::ggml_is_strided_view(a)) return false;
                if (!ggml_is_contiguous(a) && a->type == GGML_TYPE_BF16) return false;
                return true;
            }
        case GGML_OP_SET_ROWS:
            {
                // node = view of src[2]=a (dst table); src[0]=b (updates),
                // src[1]=c (indices). Custom OpenCL kernel (scatter_update can't
                // express per-group batched indices). Cross-type f32<->f16 writes
                // supported (f32 updates -> f16 KV cache): the kernel casts on
                // write (SRCFLOAT -> DSTFLOAT, round-to-nearest-even, matching
                // ggml). Quantized dst (Q4_0/Q8_0/...) needs block quantization
                // the scalar kernel can't do -> CPU. bf16 excluded (cldnn bf16
                // padding path mis-handles, same as CPY). i32/i64 indices.
                // Batched (ne2/ne3 > 1) + broadcast (ne11|ne2, ne12|ne3) handled
                // natively by the kernel.
                const ggml_tensor * b = op->src[0];
                const ggml_tensor * c = op->src[1];
                const ggml_tensor * a = op->src[2];
                if (!b || !c || !a) {
                    return false;
                }
                if (b->type != GGML_TYPE_F32 && b->type != GGML_TYPE_F16) {
                    return false;  // updates must be f32/f16
                }
                if (a->type != GGML_TYPE_F32 && a->type != GGML_TYPE_F16) {
                    return false;  // dst table must be f32/f16 (quantized/bf16 -> CPU)
                }
                if (c->type != GGML_TYPE_I32 && c->type != GGML_TYPE_I64) {
                    return false;
                }
                // b/c may be contiguous OR a "padded view" (test_set_rows v=true:
                // a sub-box view of [ne0,r,...] sliced to r/2 rows). build_set_rows
                // pre-compacts padded views (bind per-dim padded -> reorder ->
                // reshape to flat) before the flat-indexing kernel. A genuinely
                // permuted operand (not a padded sub-box) -> CPU.
                auto ok_layout = [](const ggml_tensor * t) {
                    return ggml_is_contiguous(t) || ggml::ovgpu::ggml_is_padded_view(t);
                };
                if (!ok_layout(b) || !ok_layout(c) || !ggml_is_contiguous(a)) {
                    return false;  // a (dst table) is the in-place output - must be
                                   // contiguous (the kernel writes flat into it)
                }
                return true;
            }
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            {
                // build_rope (custom_gpu kernel): all modes (NORMAL/NEOX/MROPE/
                // IMROPE/VISION), YaRN (ext_factor/attn_factor), forward + BACK,
                // contiguous src0. Strided/padded src0 (v=1/v=2 views) deferred -
                // the kernel indexes flat raw memory (nb-stride indexing is a
                // follow-up). freq_factors (src[2]) optional, F32, contiguous.
                const ggml_tensor * a   = op->src[0];
                const ggml_tensor * pos = op->src[1];
                const ggml_tensor * ff  = op->src[2];
                if (!a || !pos) {
                    return false;
                }
                if (a->type != GGML_TYPE_F32 && a->type != GGML_TYPE_F16) {
                    return false;
                }
                if (pos->type != GGML_TYPE_I32) {
                    return false;
                }
                const int32_t * op_params = (const int32_t *) op->op_params;
                const int mode = op_params[2];
                if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX &&
                    mode != GGML_ROPE_TYPE_MROPE  && mode != GGML_ROPE_TYPE_IMROPE &&
                    mode != GGML_ROPE_TYPE_VISION) {
                    return false;
                }
                if (!ggml_is_contiguous(a)) {
                    return false;  // strided/padded src0 view -> CPU (flat-indexing kernel)
                }
                if (ff) {
                    if (ff->type != GGML_TYPE_F32 || !ggml_is_contiguous(ff)) {
                        return false;
                    }
                }
                // The kernel computes cos/sin inline; OpenCL cos/sin can differ
                // from host cosf/sinf by ~1 ULP. Float constants (theta_scale,
                // freq_scale, mscale, ...) are serialized with %.9g so the GPU
                // theta progression matches the host EXACTLY (a 6-digit
                // std::to_string drifted ~4e-7/step -> ~2.6e-5 over n_dims/2
                // pairs, intermittently flipping f16 rounding). With exact theta
                // the residual 1-ULP cos/sin difference stays well inside the
                // comparison threshold for f16, so no precision gate is needed.
                return true;
            }
        case GGML_OP_UNARY:
            {
                // SILU subtype only (out = silu(x) = x/(1+exp(-x))). Other unary
                // ops (GELU/RELU/...) -> CPU. Custom kernel; f32/f16, contiguous
                // or padded view (v=1). dst = new contiguous tensor.
                if (ggml_get_op_params_i32(op, 0) != GGML_UNARY_OP_SILU) return false;
                const ggml_tensor * a = op->src[0];
                if (!a) return false;
                if (a->type != GGML_TYPE_F32 && a->type != GGML_TYPE_F16) return false;
                if (!ggml_is_contiguous(a) && !ggml::ovgpu::ggml_is_padded_view(a)) return false;
                return true;
            }
        case GGML_OP_GLU:
            {
                // SwiGLu only (out = silu(a)*b). 2-input (glu_split, what Llama
                // uses) or 1-input (split halves + swapped). Other GLU ops
                // (GEGLU/REGLU/...) -> CPU. f32/f16, contiguous or padded view.
                const ggml_tensor * a = op->src[0];
                const ggml_tensor * b = op->src[1];
                if (!a) return false;
                if (a->type != GGML_TYPE_F32 && a->type != GGML_TYPE_F16) return false;
                if (ggml_get_op_params_i32(op, 0) != GGML_GLU_OP_SWIGLU) return false;
                if (!ggml_is_contiguous(a) && !ggml::ovgpu::ggml_is_padded_view(a)) return false;
                if (b) {
                    if (b->type != a->type) return false;
                    if (!ggml_is_contiguous(b) && !ggml::ovgpu::ggml_is_padded_view(b)) return false;
                }
                return true;
            }
        case GGML_OP_CONCAT:
            {
                // dst = [src0, src1] along dim (op_params[0]). Custom stride-aware
                // kernel (one work-item/element, byte copy). blck_size==1 types
                // (f32/f16/i32/i8...); quantized block types -> CPU. Contiguous or
                // padded view inputs.
                const ggml_tensor * s0 = op->src[0];
                const ggml_tensor * s1 = op->src[1];
                if (!s0 || !s1) return false;
                if (s0->type != s1->type) return false;
                if (ggml_blck_size(s0->type) != 1) return false;
                if (s0->type == GGML_TYPE_BF16) return false;  // bf16 concat gives wrong
                    // results (same cldnn bf16+raw-bind quirk as CPY strided); KV cache is f16
                if (!ggml_is_contiguous(s0) && !ggml::ovgpu::ggml_is_padded_view(s0)) return false;
                if (!ggml_is_contiguous(s1) && !ggml::ovgpu::ggml_is_padded_view(s1)) return false;
                return true;
            }
        case GGML_OP_FLASH_ATTN_EXT:
            {
                // SDPA primitive mapping (build_flash_attn_ext). Gates to CPU:
                // ALiBi (max_bias>0), logit_softcap>0, sinks (src[4]).
                const ggml_tensor * q = op->src[0];
                const ggml_tensor * k = op->src[1];
                const ggml_tensor * v = op->src[2];
                const ggml_tensor * m = op->src[3];
                if (!q || !k || !v) return false;
                if (q->type != GGML_TYPE_F32 && q->type != GGML_TYPE_F16) return false;
                if (k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_F32) return false;
                if (v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_F32) return false;
                float max_bias, logit_softcap;
                memcpy(&max_bias,      op->op_params + 1, sizeof(float));
                memcpy(&logit_softcap, op->op_params + 2, sizeof(float));
                if (max_bias != 0.0f || logit_softcap != 0.0f) return false;
                if (op->src[4]) return false;
                if (m) {
                    if (m->type != GGML_TYPE_F16 && m->type != GGML_TYPE_F32) return false;
                    if (!ggml_is_contiguous(m)) return false;
                }
                // Mismatched K/V head sizes (hsk != hsv, MLA-style) with large heads ->
                // CPU. The cldnn SDPA kernel (sdpa_opt multi-token path; micro is disabled
                // for K_head!=V_head) segfaults on large-prefill builds for the MLA configs
                // DeepSeek 576/512 and Mistral4 320/256 (deterministic, not catchable - a
                // signal, not an exception). Smaller mismatched heads (192/128) pass the full
                // matrix, so only gate when max(hsk,hsv) > 256. Standard hsk==hsv attention
                // (Llama/Qwen/etc.) is unaffected.
                if (q->ne[0] != v->ne[0] && (q->ne[0] > 256 || v->ne[0] > 256)) return false;
                return true;
            }
        default:
            return false;
    }
    GGML_UNUSED(dev);
}

static bool ggml_backend_ovgpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_ovgpu_buffer_type_get_name;
    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_ovgpu_device_interface = {
    /* .get_name             = */ ggml_backend_ovgpu_device_get_name,
    /* .get_description      = */ ggml_backend_ovgpu_device_get_description,
    /* .get_memory           = */ ggml_backend_ovgpu_device_get_memory,
    /* .get_type             = */ ggml_backend_ovgpu_device_get_type,
    /* .get_props            = */ ggml_backend_ovgpu_device_get_props,
    /* .init_backend         = */ ggml_backend_ovgpu_device_init,
    /* .get_buffer_type      = */ ggml_backend_ovgpu_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_ovgpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_ovgpu_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

//
// registry
//

static const char * ggml_backend_ovgpu_reg_get_name(ggml_backend_reg_t reg) {
    return GGML_OVGPU_NAME;
    GGML_UNUSED(reg);
}

static size_t ggml_backend_ovgpu_reg_get_device_count(ggml_backend_reg_t reg) {
    return ((ggml_backend_ovgpu_reg_context *) reg->context)->devices.size();
}

static ggml_backend_dev_t ggml_backend_ovgpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    return ((ggml_backend_ovgpu_reg_context *) reg->context)->devices[index];
}

static const struct ggml_backend_reg_i ggml_backend_ovgpu_reg_interface = {
    /* .get_name         = */ ggml_backend_ovgpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_ovgpu_reg_get_device_count,
    /* .get_device       = */ ggml_backend_ovgpu_reg_get_device,
    /* .get_proc_address = */ NULL,
};

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_ovgpu_reg(void) {
    static ggml_backend_reg reg;
    static std::once_flag flag;
    std::call_once(flag, []() {
        ggml_backend_ovgpu_reg_context * ctx = new ggml_backend_ovgpu_reg_context;
        ctx->engine = ggml_ovgpu_create_engine();

        auto * ocl_engine = dynamic_cast<cldnn::ocl::ocl_engine *>(ctx->engine.get());
        GGML_ASSERT(ocl_engine != nullptr);
        // IN-ORDER stream: shared by ALL op networks (see make_network) so kernel
        // execution is ordered by the queue itself - op N's output is committed
        // before op N+1's kernels run, with NO per-op finish(). (cldnn's default
        // queue is out-of-order, which would need explicit event/finish sync
        // between networks.) Intra-network primitives are likewise queue-ordered.
        cldnn::ExecutionConfig stream_cfg;
        stream_cfg.set_property(ov::intel_gpu::queue_type(cldnn::QueueTypes::in_order));
        ctx->stream = ocl_engine->create_stream(stream_cfg);

        if (getenv("GGML_OVGPU_SMOKE_TEST")) {
            ggml_ovgpu_smoke_test(*ctx->engine);
        }

        const auto & info = ctx->engine->get_device_info();
        GGML_LOG_INFO("OVGPU: device=%s supports_immad=%d\n", info.dev_name.c_str(), (int) info.supports_immad);
        ggml_backend_ovgpu_device_context * dev_ctx = new ggml_backend_ovgpu_device_context{
            /* .index       = */ 0,
            /* .name        = */ std::string(GGML_OVGPU_NAME) + "0",
            /* .description = */ info.dev_name,
            /* .reg_ctx     = */ ctx,
        };

        ggml_backend_dev_t dev = new ggml_backend_device{
            /* .iface   = */ ggml_backend_ovgpu_device_interface,
            /* .reg     = */ &reg,
            /* .context = */ dev_ctx,
        };
        ctx->devices.push_back(dev);

        reg = ggml_backend_reg{
            /* .api_version = */ GGML_BACKEND_API_VERSION,
            /* .iface       = */ ggml_backend_ovgpu_reg_interface,
            /* .context     = */ ctx,
        };

        GGML_LOG_INFO("OVGPU: initialized device %s\n", info.dev_name.c_str());
    });
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_ovgpu_reg)
