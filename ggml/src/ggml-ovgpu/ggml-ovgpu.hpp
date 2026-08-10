#pragma once

// OVGPU backend internals - shared between the backend skeleton (ggml-ovgpu.cpp)
// and the op translators (ggml-ovgpu-ops.cpp).

#include "ggml-impl.h"
#include "ggml.h"

#include <functional>
#include <intel_gpu/graph/network.hpp>
#include <intel_gpu/runtime/engine.hpp>
#include <intel_gpu/runtime/memory.hpp>
#include <intel_gpu/runtime/stream.hpp>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

struct ggml_tensor;

namespace ggml::ovgpu {

// cldnn data type for a ggml type; returns data_types::i8 as a fallback (caller
// checks the bool for support).
ov::element::Type                  ggml_type_to_cldnn(ggml_type type);
// cldnn layout (bfyx tensor) for a ggml tensor's logical shape + type.
// ggml ne = [n0, n1, n2, n3] (fastest-first) -> cldnn tensor(batch = n1*n2*n3, feature = n0, 1, 1).
cldnn::layout                      ggml_layout_for(const ggml_tensor * t);
// Broadcast-preserving layout (ne[0]->x innermost, ne[1..3]->y,f,b unflattened);
// for eltwise ops where numpy broadcast must see each ggml dim separately.
cldnn::layout                      ggml_layout_for_eltwise(const ggml_tensor * t);
// Reduction-over-ne[0] layout (ne[0]->x as sole non-trivial spatial, ne[1..3]->batch);
// for rms/normalize ops where cldnn reduces over all spatials but ggml over ne[0] only.
cldnn::layout                      ggml_layout_for_reduce_ne0(const ggml_tensor * t);
// True for a strided tensor describing a "sub-box" of a larger contiguous
// tensor (e.g. a ggml_view_4d taking a sub-range of every dim) - the shape
// ggml_layout_for_reduce_ne0_maybe_padded/ggml_layout_for_padded can bind
// zero-copy via cldnn per-dim padding.
bool                               ggml_is_padded_view(const ggml_tensor * t);
// Like ggml_is_padded_view, but also requires ne[1..3] to be densely nested (no
// padding outside ne[0]) - the stronger condition ggml_layout_for_maybe_padded
// demands (it can only attach padding to the feature/ne[0] axis). A padded view
// with padding on ne[1]/ne[2]/ne[3] is NOT flat-padded-view-able and must be
// offloaded to CPU (else ggml_layout_for_maybe_padded throws -> hard-fail).
bool                               ggml_is_flat_padded_view(const ggml_tensor * t);
// Like ggml_layout_for_reduce_ne0, but also accepts a strided "padded view"
// tensor (see ggml_is_padded_view in ggml-ovgpu-ops.cpp) via a per-dim padded
// layout; falls back to ggml_layout_for_reduce_ne0 for contiguous tensors.
cldnn::layout                      ggml_layout_for_reduce_ne0_maybe_padded(const ggml_tensor * t);
// Like ggml_layout_for, but also accepts a "padded view" tensor (see
// ggml_is_padded_view) via ggml_layout_for's OWN flat-batch axis convention
// (ne[1..3] collapsed into one cldnn batch axis, ne[0] -> feature - NOT a
// per-dim padded layout like ggml_layout_for_reduce_ne0_maybe_padded), with the
// padding attached to the feature axis; falls back to ggml_layout_for for
// contiguous tensors. Throws if ne[1..3] have padding of their own (unsupported).
// Used by build_mul_mat (e.g. a strided-K view, ggml_view_4d shrinking ne[0]
// from a larger k_v allocation).
cldnn::layout                      ggml_layout_for_maybe_padded(const ggml_tensor * t);
// True for a tensor whose strides describe a (possibly PERMUTED) "sub-box" of a
// larger contiguous tensor: unlike ggml_is_padded_view (which requires ggml's
// own dim order 0,1,2,3 to already be fastest-to-slowest), this also accepts a
// view that has been permuted (ggml_permute/ggml_transpose composed with a
// view) - sorting the tensor's dims by ascending stride must still recover a
// nested layout. This is the general case; ggml_is_padded_view is the special
// case where that sort order is already the identity.
bool                               ggml_is_strided_view(const ggml_tensor * t);
// Like ggml_layout_for_eltwise, but also accepts a ggml_is_strided_view tensor;
// falls back to ggml_layout_for_eltwise for contiguous tensors. Only usable as
// the RAW bind layout for the first node in a strided-input chain (see
// build_eltwise) - the caller must still permute+reorder into the logical
// ne0->x/ne1->y/ne2->f/ne3->b convention before feeding a compute primitive.
cldnn::layout                      ggml_layout_for_eltwise_maybe_strided(const ggml_tensor * t);
// wrap a ggml tensor's storage into a non-owning cldnn::memory view over the
// backend's USM base memory, at byte_offset, using the given layout. create_subbuffer
// returns a view (no ownership) so it does not free the underlying allocation.
cldnn::memory::ptr                 wrap_tensor(cldnn::engine &       engine,
                                               cldnn::memory &       base_mem,
                                               const cldnn::layout & l,
                                               size_t                byte_offset);

// A compiled single-op network + the ids of its external inputs/outputs.
struct compiled_op {
    std::shared_ptr<cldnn::network> net;
    std::vector<std::string>        input_ids;  // input_layout primitive ids, in ggml src order
    std::string                     output_id;  // the op's output primitive id
    std::vector<std::string>        weight_ids; // data() primitive ids (weights), not re-bound per compute
    // Layout strategy for binding a ggml tensor into cldnn memory. Defaults to the
    // flat-batch mapping (ggml_layout_for); eltwise ops override to a broadcast-
    // preserving mapping (ggml_layout_for_eltwise). graph_compute calls this per input/output.
    std::function<cldnn::layout(const ggml_tensor *)> layout_for = ggml_layout_for;

    // --- Fusion (multi-op chain) extensions. ---
    // When is_chain, this compiled_op drives a LINEAR chain of ggml ops
    // [head, child1, ..., childN] fused into ONE cldnn topology. The network's
    // single output is the chain's final node dst; intermediate (head + non-final
    // child) outputs are NOT materialized (cldnn fuses them away or holds them as
    // internal temps). graph_compute binds the chain's external inputs positionally
    // (input_ids[i] <- the ggml src named by input_src_map[i]) and executes once at
    // the head, skipping the absorbed children.
    bool                                     is_chain = false;
    // Per-external-input bind layout (chain child external operands use the HEAD's
    // output-layout convention so cldnn's fuser shape rule is satisfied). Empty for
    // single-op -> graph_compute uses layout_for(src) instead.
    std::vector<cldnn::layout>               input_layouts;
    // (chain-node-index, src-index) per external input: input i binds
    // chain_nodes[node_idx]->src[src_idx]. Self-describing so graph_compute and the
    // builder cannot drift on input ordering.
    std::vector<std::pair<int, int>>         input_src_map;
    // Number of nodes in the chain (head + children). graph_compute skips
    // chain_len-1 nodes after the head.
    int                                      chain_len = 0;
};

// Build a single-op cldnn network for the given ggml node. Returns nullptr if
// the op/shape/dtype combo is not (yet) supported - caller reports supports_op=false.
std::unique_ptr<compiled_op>       build_op(const ggml_tensor * node, cldnn::engine & engine, cldnn::stream::ptr stream);

// Build a FUSED multi-op cldnn network for a linear chain of ggml nodes
// [head, child1, ..., childN]. head is MUL_MAT (FC) or RMS_NORM (rms); each child
// is ADD/MUL (eltwise) or UNARY-SILU (activation) consuming the previous node's
// output. cldnn's program fuser merges the children into the head at build time
// (gated on optimize_data(true), set in make_config). Returns nullptr if the chain
// is not fusible (caller falls back to per-op). The nodes vector is [head, ...children].
std::unique_ptr<compiled_op>       build_chain(const std::vector<const ggml_tensor *> & nodes,
                                               cldnn::engine &                          engine,
                                               cldnn::stream::ptr                       stream);

// Per-op-network cache. Keyed by a structural signature of the ggml node
// (op + dtypes + ne[] + flags) so an identical shape bucket reuses the compiled
// network across layers and across compute calls.
struct op_cache {
    cldnn::engine *    engine;
    cldnn::stream::ptr stream;  // shared in-order stream for ALL op networks (so
                                // executes are ordered -> no per-op finish needed)
    std::unordered_map<std::string, std::shared_ptr<compiled_op>> m_cache;
    // Separate cache for fused chains (keyed by a structural signature of the whole
    // chain, not just one node). Kept distinct from m_cache so the two paths never
    // collide on a key.
    std::unordered_map<std::string, std::shared_ptr<compiled_op>> chain_cache;
    // Negative cache: chains whose build threw / build_chain rejected. Prevents
    // retrying (and re-logging) a non-fusible chain on every graph_compute call. The
    // detector can therefore be loose; build_chain is the authority on eligibility.
    std::unordered_set<std::string>                                  chain_fail_cache;
    std::mutex      m_mutex; // test-backend-ops runs cases in parallel threads

    static std::string key_for(const ggml_tensor * node);

    std::unique_ptr<compiled_op> get_or_build(const ggml_tensor * node);

    // Build (or fetch from chain_cache) a fused chain for `nodes` ([head, ...children]).
    // Returns nullptr if the chain is not fusible / build threw.
    std::unique_ptr<compiled_op> get_or_build_chain(const std::vector<const ggml_tensor *> & nodes);
};

}  // namespace ggml::ovgpu
