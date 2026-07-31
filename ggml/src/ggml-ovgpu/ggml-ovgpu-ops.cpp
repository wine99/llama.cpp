#include "ggml-ovgpu.hpp"

#include "ggml-impl.h"

#include <intel_gpu/graph/topology.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/fully_connected.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/reorder.hpp>

#include <cmath>
#include <sstream>

namespace ggml::ovgpu {

// ---------------------------------------------------------------------------
// type / layout helpers
// ---------------------------------------------------------------------------

ov::element::Type ggml_type_to_cldnn(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: return ov::element::f32;
        case GGML_TYPE_F16: return ov::element::f16;
        case GGML_TYPE_BF16: return ov::element::bf16;
        default:            return ov::element::f32; // caller checks ggml_type_supports()
    }
}

static bool type_supported(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
}

// ggml ne[4] (fastest-first) -> cldnn bfyx tensor(batch, feature, 1, 1).
// Flatten ggml dims 1,2,3 into the cldnn batch dim (dim1 fastest), feature = dim0.
// For a contiguous tensor this matches ggml's physical memory exactly (element
// (i0,i1,i2,i3) at ((i3*ne2+i2)*ne1+i1)*ne0+i0 = flat_batch*ne0+i0), so the SAME
// flat ordering works for a 2D weight, a batched activation, and the batched dst.
static cldnn::tensor ggml_ne_to_tensor(const ggml_tensor * t) {
    auto v = [](int64_t n) { return (cldnn::tensor::value_type) std::max<int64_t>(n, 1); };
    int64_t batch = t->ne[1] * t->ne[2] * t->ne[3]; // dims 1,2,3 flattened
    return cldnn::tensor(v(batch), v(t->ne[0]), 1, 1); // batch, feature
}

cldnn::layout ggml_layout_for(const ggml_tensor * t) {
    return cldnn::layout{ggml_type_to_cldnn(t->type), cldnn::format::bfyx, ggml_ne_to_tensor(t)};
}

cldnn::memory::ptr wrap_tensor(cldnn::engine & engine, cldnn::memory & base_mem,
                               const ggml_tensor * t, size_t byte_offset) {
    cldnn::layout l = ggml_layout_for(t);
    return engine.create_subbuffer(base_mem, l, byte_offset);
}

// ---------------------------------------------------------------------------
// op signature (cache key)
// ---------------------------------------------------------------------------

std::string op_cache::key_for(const ggml_tensor * node) {
    std::ostringstream s;
    s << "op" << (int) node->op;
    s << "|out" << (int) node->type;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        const ggml_tensor * src = node->src[i];
        if (!src) break;
        s << "|s" << i << ":" << (int) src->type << ":";
        for (int d = 0; d < GGML_MAX_DIMS; d++) s << src->ne[d] << ",";
        // op params that affect the kernel (transpose flags etc.)
    }
    // a few op-specific params
    switch (node->op) {
        case GGML_OP_MUL_MAT:
            s << "|transposed=" << (node->op_params ? ggml_get_op_params_i32(node, 0) : 0);
            break;
        default: break;
    }
    return s.str();
}

std::unique_ptr<compiled_op> op_cache::get_or_build(const ggml_tensor * node) {
    std::string key = key_for(node);
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_cache.find(key);
    if (it != m_cache.end()) {
        auto co = std::make_unique<compiled_op>();
        co->net        = it->second->net;
        co->input_ids  = it->second->input_ids;
        co->output_id  = it->second->output_id;
        co->weight_ids = it->second->weight_ids;
        return co;
    }
    auto co = build_op(node, *engine);
    if (co) {
        auto stored = std::make_shared<compiled_op>();
        stored->net        = co->net;
        stored->input_ids  = co->input_ids;
        stored->output_id  = co->output_id;
        stored->weight_ids = co->weight_ids;
        m_cache[key] = std::move(stored);
    }
    return co;
}

// ---------------------------------------------------------------------------
// MUL_MAT -> fully_connected
// ---------------------------------------------------------------------------

// ggml MUL_MAT: dst = src[0]^T @ src[1]   (dst is ALWAYS f32, see ggml_mul_mat).
//   src[0] = weight   ne=[K, N]              (2D; group broadcast deferred)
//   src[1] = activation ne=[K, M, ne2, ne3]  (1-4D; dims 1..3 flattened to batch)
//   dst    = f32 [N, M, ne2, ne3]
// cldnn fully_connected(input, weights, bias): out = input @ weights^T.
//   input=[batch,K], weights=[N,K] -> out=[batch,N]. With the flat-batch mapping the
//   result bytes line up with ggml's contiguous dst.
//
// Precision: ggml widens f16/bf16 to f32 and accumulates in f32 (ggml_vec_dot_f16 ->
// float). cldnn's f32-input FC also accumulates in f32, so to match ggml semantics we
// feed FC f32 operands. The activation upcast is cheap (small operand).
//
// M1 PERF CAVEAT: f16/bf16 weights bound as a plain-bfyx input_layout hit cldnn's ocl
// kernel path (UHD 770 has supports_immad=0 -> no oneDNN), which mis-reads f16 weights
// for some batch sizes (wrong output / no kernel). So we upcast the weight to f32 per
// compute too. This is a full-matrix reorder every token and is NOT the intended fast
// path; the M2 fix is to bake weights natively (data nodes / blocked layout) so the
// selected kernel reads them in place. Correctness first for M1.

static std::unique_ptr<compiled_op> build_mul_mat(const ggml_tensor * node, cldnn::engine & engine) {
    const ggml_tensor * weight = node->src[0];
    const ggml_tensor * act    = node->src[1];
    if (!weight || !act) return nullptr;
    if (!type_supported(weight->type) || !type_supported(act->type)) return nullptr;
    if (ggml_n_dims(weight) != 2) return nullptr; // weight must be 2D (matches supports_op)

    auto co = std::make_unique<compiled_op>();
    // input_ids follow ggml src order so graph_compute binds src[k] -> input_ids[k].
    //   src[0] = weight, src[1] = activation.
    const std::string w_id = "in_0"; // weight     (src[0])
    const std::string a_id = "in_1"; // activation (src[1])
    co->input_ids  = {w_id, a_id};
    co->weight_ids = {w_id};
    co->output_id  = "out";

    // Both operands are bound per-compute as input_layouts -> network is shape-keyed,
    // so layers with identical dims share one compiled network.
    cldnn::topology topo;
    topo.add(cldnn::input_layout(w_id, ggml_layout_for(weight)));
    topo.add(cldnn::input_layout(a_id, ggml_layout_for(act)));

    // Upcast operands to f32 (see precision note above). f32 operands skip the reorder.
    auto upcast = [&](const std::string & in_id, const std::string & out_id, const ggml_tensor * t) -> std::string {
        if (t->type == GGML_TYPE_F32) return in_id;
        cldnn::layout l{ov::element::f32, cldnn::format::bfyx, ggml_ne_to_tensor(t)};
        topo.add(cldnn::reorder(out_id, cldnn::input_info(in_id), l));
        return out_id;
    };
    std::string fc_weight = upcast(w_id, "w_f32", weight);
    std::string fc_input  = upcast(a_id, "act_f32", act);

    topo.add(cldnn::fully_connected("fc_out", cldnn::input_info(fc_input), fc_weight, "",
                                    2 /*input_size*/, 2 /*weights_rank*/, true /*weights_transposed*/));

    // Trailing reorder forces bfyx f32 output: cldnn's layout optimizer can pick a
    // different output format for some shapes (e.g. yxfb for batch=8, a transpose of
    // bfyx for 2D), which would garble the ggml tensor.
    topo.add(cldnn::reorder(co->output_id, cldnn::input_info("fc_out"), ggml_layout_for(node)));

    co->net = std::make_shared<cldnn::network>(engine, topo, cldnn::ExecutionConfig{});
    return co;
}

// ---------------------------------------------------------------------------
// dispatch
// ---------------------------------------------------------------------------

std::unique_ptr<compiled_op> build_op(const ggml_tensor * node, cldnn::engine & engine) {
    switch (node->op) {
        case GGML_OP_MUL_MAT: return build_mul_mat(node, engine);
        default:              return nullptr;
    }
}

}  // namespace ggml::ovgpu
