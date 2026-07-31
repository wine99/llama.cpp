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
    // mem_lock read_write: read current buffer (preserve other tensors), patch our region, write back.
    cldnn::mem_lock<uint8_t, cldnn::mem_lock_type::read_write> lock(ctx->mem, *reg_ctx->stream);
    memcpy((uint8_t *) lock.data() + dev_off, data, size);
}

static void ggml_backend_ovgpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_ovgpu_buffer_context * ctx = (ggml_backend_ovgpu_buffer_context *) buffer->context;
    ggml_backend_ovgpu_reg_context * reg_ctx = ((ggml_backend_ovgpu_buffer_type_context *) buffer->buft->context)->reg_ctx;
    size_t dev_off = (char *) tensor->data - (char *) ctx->base + offset;
    cldnn::mem_lock<uint8_t, cldnn::mem_lock_type::read> lock(ctx->mem, *reg_ctx->stream);
    memcpy(data, (uint8_t *) lock.data() + dev_off, size);
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
    // commit all prior set_tensor writes (host->device on reg_ctx->stream) before compute
    ctx->reg_ctx->stream->finish();
    if (getenv("GGML_OVGPU_DEBUG")) fprintf(stderr, "[ovgpu] graph_compute n_nodes=%d\n", cgraph->n_nodes);

    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (getenv("GGML_OVGPU_DEBUG")) fprintf(stderr, "[ovgpu]   loop i=%d\n", i);
        ggml_tensor * node = cgraph->nodes[i];

        if (node->op == GGML_OP_NONE) {
            continue; // leaf / no-op; data already lives in the buffer
        }

        if (getenv("GGML_OVGPU_DEBUG")) fprintf(stderr, "[ovgpu]   node %d op=%d\n", i, node->op);
        auto co = ctx->op_cache.get_or_build(node);
        if (!co) {
            GGML_LOG_ERROR("OVGPU: no translator for op %d (%s)\n", node->op, ggml_op_name(node->op));
            return GGML_STATUS_FAILED;
        }
        if (getenv("GGML_OVGPU_DEBUG")) fprintf(stderr, "[ovgpu]   built\n");

        // bind inputs in the translator's declared order (build_mul_mat: {weight=src[0], activation=src[1]})
        for (size_t k = 0; k < co->input_ids.size(); k++) {
            const ggml_tensor * src = node->src[k];
            GGML_ASSERT(src && src->buffer && "OVGPU: input not in backend buffer");
            auto * buf_ctx = (ggml_backend_ovgpu_buffer_context *) src->buffer->context;
            size_t off = (char *) src->data - (char *) buf_ctx->base;
            if (getenv("GGML_OVGPU_DEBUG")) fprintf(stderr, "[ovgpu]   bind in %s off=%zu\n", co->input_ids[k].c_str(), off);
            auto mem = ggml::ovgpu::wrap_tensor(engine, *buf_ctx->mem, src, off);
            co->net->set_input_data(co->input_ids[k], mem);
        }
        if (getenv("GGML_OVGPU_DEBUG")) fprintf(stderr, "[ovgpu]   inputs bound\n");

        // bind output in-place into the node's tensor
        {
            ggml_tensor * dst = node;
            auto * buf_ctx = (ggml_backend_ovgpu_buffer_context *) dst->buffer->context;
            size_t off = (char *) dst->data - (char *) buf_ctx->base;
            auto mem = ggml::ovgpu::wrap_tensor(engine, *buf_ctx->mem, dst, off);
            co->net->set_output_memory(co->output_id, mem);
        }

        co->net->execute();
        co->net->get_stream().finish(); // commit output before get_tensor (cross-stream)
    }

    ctx->reg_ctx->stream->finish();
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
            return ok_t(op->type);
        case GGML_OP_MUL_MAT: {
            const ggml_tensor * w = op->src[0];
            const ggml_tensor * a = op->src[1];
            if (!w || !a) return false;
            if (!ok_t(w->type) || !ok_t(a->type)) return false;
            // weight must be 2D: a group dim (ne[2]/ne[3] > 1) is the GQA-style
            // weight broadcast, which a single 2D-weight FC cannot express -> CPU.
            if (ggml_n_dims(w) != 2) return false;
            // operands must be contiguous: the flat-batch mapping assumes tight
            // strides, so permuted views (ggml_permute) and strided K views break it.
            if (!ggml_is_contiguous(w) || !ggml_is_contiguous(a)) return false;
            if (w->ne[0] != a->ne[0]) return false; // K must match
            // mixed dtypes allowed: f32/f16/bf16 x f32/f16/bf16 -> f32 dst.
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
        ctx->stream = ocl_engine->create_stream(cldnn::ExecutionConfig{});

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
