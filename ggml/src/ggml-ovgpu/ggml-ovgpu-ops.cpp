#include "ggml-impl.h"
#include "ggml-ovgpu.hpp"

#include <algorithm>
#include <cmath>
#include <intel_gpu/graph/network.hpp>
#include <intel_gpu/graph/program.hpp>
#include <intel_gpu/graph/topology.hpp>
#include "program_node.h" // FUSE_TRACE: has_fused_primitives / get_fused_primitives / is_type
#include <intel_gpu/primitives/custom_gpu_primitive.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/fully_connected.hpp>
#include <intel_gpu/primitives/gather.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/permute.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/reshape.hpp>
#include <intel_gpu/primitives/rms.hpp>
#include <intel_gpu/primitives/scaled_dot_product_attention.hpp>
#include <intel_gpu/primitives/softmax.hpp>
#include <intel_gpu/runtime/internal_properties.hpp>
#include <sstream>
#include <stdexcept>

// SET_ROWS custom OpenCL kernel (run via cldnn::custom_gpu_primitive).
// ggml SET_ROWS: for each (j, i2, i3), dst[ idx[j,i2%NE11,i3%NE12], i2, i3 ] =
// src0[ j, i2, i3 ]. STRIDE-AWARE: src0 (updates) and src1 (indices) are read via
// their ggml nb[] BYTE strides (NB0..NB3 / NB0I..NB2I), so a contiguous OR a
// padded/strided view (test_set_rows v=true) is handled WITHOUT pre-compaction -
// the kernel navigates the raw allocation directly. dst (the table, src[2]=a) is
// always contiguous (supports_op gates it) so it is indexed flat:
//   addr = ((i3*NE2+i2)*NE1 + idx)*NE0 + c.
// Scalars (R, NE0..NE3, NE11, NE12, NB*, NB*I) are baked as -D build_options.
// SRCFLOAT/DSTFLOAT support cross-type writes (f32 updates -> f16 KV cache): the
// read uses SRCFLOAT, the write casts to DSTFLOAT (round-to-nearest-even). DIDX in
// {int, long}. One work-item per output element. The kernel writes ONLY the
// indexed rows; unmodified rows are preserved in dst's existing memory (node is a
// view of src[2]=a, bound in-place - cldnn does not zero a bound output buffer).
static const char * OVGPU_SET_ROWS_KERNEL = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void ovgpu_set_rows(
    __global const char    * src0,   // updates [NE0,R,NE2,NE3], byte-strided
    __global const char    * src1,   // indices [R,NE11,NE12],  byte-strided
    __global       DSTFLOAT * dst)   // table   [NE0,NE1,NE2,NE3] (flat in-place view)
{
    const int c  = get_global_id(0);   // [0, NE0)
    const int fr = get_global_id(1);   // [0, R*NE2*NE3)  (flat row of src0)
    if (c >= NE0) return;
    const int rows_per_slice = R * NE2;
    if (fr >= rows_per_slice * NE3) return;
    const int j  = fr % R;
    const int t  = fr / R;
    const int i2 = t % NE2;
    const int i3 = t / NE2;
    const int i11 = i2 % NE11;        // broadcast: indices repeat over ne2/ne3
    const int i12 = i3 % NE12;
    const long idx = (long)*((__global const DIDX *)(src1 + (long)j*NB0I + (long)i11*NB1I + (long)i12*NB2I));
    const long src_off = (long)c*NB0 + (long)j*NB1 + (long)i2*NB2 + (long)i3*NB3;
    const long dst_off = (((long)i3*NE2 + i2)*NE1 + idx)*NE0 + c;
    dst[dst_off] = (DSTFLOAT)*((__global const SRCFLOAT *)(src0 + src_off));
}
)CLC";

// ROPE custom OpenCL kernel (all modes: NORMAL/NEOX/MROPE/IMROPE/VISION, YaRN,
// forward + ROPE_BACK). Faithful port of ggml-cpu's ggml_compute_forward_rope_flt
// (the test-backend-ops reference) - NOT the ggml-opencl kernel (which uses pow;
// cpu uses incremental theta *= theta_scale, so we do too for bit-closeness).
//
// src0 = [NE0=head_dim, NE1=heads, NE2=seq, NE3=batch]; pos = src1 (I32);
// for mrope/vision pos has 4 position ids per token (pos[i2 + NE2*k], k=0..3).
// ff = src2 (F32 [>=N_DIMS/2]) only if HAS_FF. dst = same shape as src0.
//
// Per row (i1,i2,i3): theta starts at the position value(s) and multiplies by
// THETA_SCALE each pair (= pow(freq_base,-2/n_dims)); cos/sin via rope_yarn
// (YaRN ramp blend + mscale when EXT_FACTOR!=0). Mode selects pair layout:
//   NORMAL : ic=i0,   n_offset=1        (interleaved (i0,i0+1))
//   NEOX   : ic=i0/2, n_offset=N_DIMS/2 (split-half)
//   MROPE  : ic=i0/2, n_offset=N_DIMS/2 (NEOX layout, 4 per-dim thetas)
//   IMROPE : ic=i0/2, n_offset=N_DIMS/2 (MROPE, interleaved sector selection)
//   VISION : ic=i0/2, n_offset=N_DIMS   (rotates ALL ne0; n_dims==ne0/2; per-dim
//                                        thetas reset at section boundaries)
// Channels [N_DIMS,NE0) are tail-copied (non-vision only). SIN_SIGN=+1 forward,
// -1 ROPE_BACK (inverse rotation). One work-item per row (loops pairs).
static const char * OVGPU_ROPE_KERNEL = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

float ovgpu_rope_yarn_ramp(float low, float high, int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

__kernel void ovgpu_rope(
    __global const DFLOAT * src0,
    __global const int    * pos
#ifdef HAS_FF
    , __global const float * ff
#endif
    , __global       DFLOAT * dst)
{
    const int row = get_global_id(0);   // [0, NE1*NE2*NE3)
    const int t1 = row / NE1;
    const int i1 = row - t1 * NE1;
    const int i2 = t1 % NE2;
    const int i3 = t1 / NE2;
    if (i3 >= NE3) return;
    const long base = ((long)(i3 * NE2 + i2) * NE1 + i1) * (long)NE0;
    __global const DFLOAT * src = src0 + base;
    __global       DFLOAT * d   = dst   + base;

    // theta state - matches ggml-cpu: non-mrope tracks one theta; mrope/vision
    // track four (t/h/w/e), each *= THETA_SCALE every pair. VISION (INDEP_SECTS)
    // resets the active theta when re-entering its section.
    float theta_t = (float)pos[i2];
#if defined(ROPE_MROPE) || defined(ROPE_VISION)
    float theta_h = (float)pos[i2 + NE2];
    float theta_w = (float)pos[i2 + NE2 * 2];
    float theta_e = (float)pos[i2 + NE2 * 3];
    const int sect_dims = SEC0 + SEC1 + SEC2 + SEC3;
    const int sec_w     = SEC0 + SEC1;
#endif

    for (int i0 = 0; i0 < NE0; i0 += 2) {
        if (i0 < ROT_N) {
            float theta;
#if defined(ROPE_MROPE) || defined(ROPE_VISION)
            const int sector = (i0 / 2) % sect_dims;
  #ifdef ROPE_INDEP_SECTS   // VISION: reset theta at section re-entry
            if (sector == 0)                  theta_t = (float)pos[i2];
            else if (sector == SEC0)          theta_h = (float)pos[i2 + NE2];
            else if (sector == sec_w)         theta_w = (float)pos[i2 + NE2 * 2];
            else if (sector == sec_w + SEC2)  theta_e = (float)pos[i2 + NE2 * 3];
  #endif
  #ifdef ROPE_IMROPE   // qwen3vl interleaved sector->dim mapping
            if (sector % 3 == 1 && sector < 3 * SEC1)       theta = theta_h;
            else if (sector % 3 == 2 && sector < 3 * SEC2)  theta = theta_w;
            else if (sector % 3 == 0 && sector < 3 * SEC0)  theta = theta_t;
            else                                             theta = theta_e;
  #else
            if (sector < SEC0)               theta = theta_t;
            else if (sector < sec_w)         theta = theta_h;
            else if (sector < sec_w + SEC2)  theta = theta_w;
            else                             theta = theta_e;
  #endif
#else
            theta = theta_t;
#endif
            const int ic = i0 / IC_SCALE;
#ifdef HAS_FF
            const float ff_i = ff[i0 / 2];   // freq_factors ALWAYS indexed by i0/2
                                              // (ggml-cpu), independent of IC_SCALE
                                              // (NORMAL uses ic=i0 for src but i0/2 for ff)
#else
            const float ff_i = 1.0f;
#endif
            // rope_yarn (ggml-cpu): theta_interp = freq_scale*theta/ff; YaRN blends
            // interp/extrap by a per-i0 ramp when EXT_FACTOR!=0; mscale scales both.
            const float theta_extrap = theta / ff_i;
            const float theta_interp = FREQ_SCALE * theta_extrap;
            float th = theta_interp;
#ifdef ROPE_YARN
            const float ramp = ovgpu_rope_yarn_ramp(CORR_LOW, CORR_HIGH, i0) * EXT_FACTOR;
            th = theta_interp * (1.0f - ramp) + theta_extrap * ramp;
#endif
            const float c = cos(th) * MSCALE;
            const float s = sin(th) * MSCALE * SIN_SIGN;
            const float x0 = src[ic];
            const float x1 = src[ic + N_OFFSET];
            d[ic]            = (DFLOAT)(x0 * c - x1 * s);
            d[ic + N_OFFSET] = (DFLOAT)(x0 * s + x1 * c);
        } else {
            // tail copy: channels [N_DIMS, NE0) pass through unchanged (non-vision).
            d[i0]     = src[i0];
            d[i0 + 1] = src[i0 + 1];
        }
        theta_t *= THETA_SCALE;
#if defined(ROPE_MROPE) || defined(ROPE_VISION)
        theta_h *= THETA_SCALE;
        theta_w *= THETA_SCALE;
        theta_e *= THETA_SCALE;
#endif
    }
}
)CLC";

namespace ggml::ovgpu {

// ---------------------------------------------------------------------------
// type / layout helpers
// ---------------------------------------------------------------------------

// Serialize a float with enough precision to round-trip exactly when re-parsed
// by the OpenCL compiler (e.g. as a -D build_option). std::to_string(float) uses
// only 6 digits - e.g. theta_scale 0.865964353 -> "0.865964" - and the truncated
// constant drifts from the host's full-precision value over repeated multiplies
// (theta *= THETA_SCALE across n_dims/2 pairs), intermittently flipping f16
// output rounding across the comparison threshold. %.9g gives 9 sig digits, the
// minimum that round-trips every IEEE-754 single exactly.
static std::string fstr(float f) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.9g", (double) f);
    return std::string(buf);
}

// Build a cldnn::network for `topo` on the SHARED `stream` (not a fresh per-network
// stream). All op networks share one in-order stream -> executes are ordered, so
// op N's output is committed before op N+1 reads it -> NO per-op finish() needed
// (the old path made each network its own stream via engine.create_stream, forcing
// a finish after every execute). Builds the program with `config` (kernel/layout
// selection is config-driven, stream-independent), then allocates the network on
// `stream`.
static std::shared_ptr<cldnn::network> make_network(cldnn::engine &       engine,
                                                     cldnn::stream::ptr    stream,
                                                     const cldnn::topology & topo,
                                                     const cldnn::ExecutionConfig & config) {
    auto prog = cldnn::program::build_program(engine, topo, config);
    return cldnn::network::allocate_network(stream, prog, false /*is_internal*/, true /*is_primary_stream*/);
}

// Forward decl: build_mul_mat (the first translator) uses the shared config helper that is
// defined further down (before the other translators). optimize_data(true) + use_onednn(true)
// and NOT allow_new_shape_infer (oneDNN FC rank-2 reshape breaks under new shape infer).
static cldnn::ExecutionConfig make_config();

ov::element::Type ggml_type_to_cldnn(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: return ov::element::f32;
        case GGML_TYPE_F16: return ov::element::f16;
        case GGML_TYPE_BF16: return ov::element::bf16;
        case GGML_TYPE_I32:
            return ov::element::i32;                 // GET_ROWS indices, not a compute dtype
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

// Broadcast-preserving layout for eltwise ops. Unlike ggml_ne_to_tensor (which
// flattens ne[1..3] into one batch dim - fine for FC's tight memory match but
// destroys per-dim broadcast structure), this maps each ggml dim to its own cldnn
// dim WITHOUT collapsing: ne[0]->x (innermost, contiguous), ne[1]->y, ne[2]->f,
// ne[3]->b. NUMPY eltwise broadcast aligns from the trailing (innermost) dim, and
// ggml broadcasts along ne[0] (the contiguous dim), so putting ne[0] on x makes a
// ggml broadcast (b smaller in ne[0]) become a NUMPY broadcast (b.x=1) correctly.
// Memory still matches ggml's contiguous layout (element (i0,i1,i2,i3) at
// ((i3*ne2+i2)*ne1+i1)*ne0+i0 = bfyx(b,f,y,x) address), so the bind is zero-copy.
cldnn::layout ggml_layout_for_eltwise(const ggml_tensor * t) {
    auto v = [](int64_t n) {
        return (cldnn::tensor::value_type) std::max<int64_t>(n, 1);
    };
    // tensor ctor is (batch, feature, x, y); bfyx memory order is b,f,y,x.
    cldnn::tensor ten(v(t->ne[3]), v(t->ne[2]), v(t->ne[0]), v(t->ne[1]));
    return cldnn::layout{ ggml_type_to_cldnn(t->type), cldnn::format::bfyx, ten };
}

// Reduction-over-ne[0] layout for rms/normalize ops. cldnn rms reduces over ALL
// spatial dims (x,y,z - see rms_gpu_ref.cl), while ggml normalizes over ne[0] ONLY.
// So put ne[0] on x (the only non-trivial spatial), y=z=1, and flatten ne[1..3] into
// batch (b), feature=1. The reduction is then over x=ne0 (and 1,1) = ne0 exactly.
// Zero-copy: ne[0] on stride-1 x, ne[1..3] flattened into batch (matches ggml flat
// memory). Used by build_rms_norm.
cldnn::layout ggml_layout_for_reduce_ne0(const ggml_tensor * t) {
    auto v = [](int64_t n) {
        return (cldnn::tensor::value_type) std::max<int64_t>(n, 1);
    };
    int64_t       batch = t->ne[1] * t->ne[2] * t->ne[3];
    cldnn::tensor ten(v(batch), 1, v(t->ne[0]), 1);  // b=flat, f=1, x=ne0, y=1
    return cldnn::layout{ ggml_type_to_cldnn(t->type), cldnn::format::bfyx, ten };
}

// True for a tensor whose strides describe a "sub-box" of a larger contiguous
// tensor - i.e. exactly what ggml_view_4d(a, ne0/2, ne1/2, ne2/2, ne3/2,
// a->nb[1], a->nb[2], a->nb[3], 0) produces: dim0 packed, and each dim i's
// stride is a whole multiple of dim i-1's (used-extent * elemsize), covering
// at least the used extent. This is the shape ggml_layout_for_padded can
// express zero-copy via cldnn per-dim padding.
bool ggml_is_padded_view(const ggml_tensor * t) {
    if (t->nb[0] != ggml_type_size(t->type)) {
        return false;  // dim0 must be packed
    }
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        if (t->nb[i - 1] == 0 || t->nb[i] % t->nb[i - 1] != 0) {
            return false;
        }
        int64_t full = (int64_t) (t->nb[i] / t->nb[i - 1]);
        if (full < t->ne[i - 1]) {
            return false;  // full extent must contain the used one
        }
    }
    return true;
}

// Like ggml_is_padded_view, but also requires ne[1..3] densely nested (no padding
// outside ne[0]) - the condition ggml_layout_for_maybe_padded demands (it can
// only attach padding to the feature/ne[0] axis of a flat-batch layout). A padded
// view with padding on ne[1]/ne[2]/ne[3] fails this -> must offload to CPU, or
// ggml_layout_for_maybe_padded would throw (-> graph_compute hard-fail).
bool ggml_is_flat_padded_view(const ggml_tensor * t) {
    if (!ggml_is_padded_view(t)) {
        return false;
    }
    for (int i = 2; i < GGML_MAX_DIMS; i++) {
        if (t->nb[i] != t->nb[i - 1] * t->ne[i - 1]) {
            return false;  // padding outside ne[0] unsupported by the flat layout
        }
    }
    return true;
}

// Per-dim (non-flattened) layout for a possibly-strided "padded view" tensor
// (see ggml_is_padded_view): ne0->x, ne1->y, ne2->f, ne3->b - same mapping as
// ggml_layout_for_eltwise - with cldnn::padding filling the gap between each
// dim's used extent (ne[i]) and its implied full/padded extent recovered from
// the next dim's stride (nb[i+1]/nb[i]). The outermost dim (ne3) has no larger
// container so it is never padded. For a contiguous tensor all gaps are zero,
// reducing to a plain (unpadded) per-dim layout. Zero-copy either way.
static cldnn::layout ggml_layout_for_padded(const ggml_tensor * t) {
    auto v = [](int64_t n) {
        return (cldnn::tensor::value_type) std::max<int64_t>(n, 1);
    };
    cldnn::tensor used(v(t->ne[3]), v(t->ne[2]), v(t->ne[0]), v(t->ne[1]));  // ctor order (b,f,x,y)

    auto full_of = [&](int i) -> int64_t {
        if (i == GGML_MAX_DIMS - 1) {
            return t->ne[i];  // outermost: no larger container
        }
        return (int64_t) (t->nb[i + 1] / t->nb[i]);
    };
    int64_t        up_x = full_of(0) - t->ne[0];
    int64_t        up_y = full_of(1) - t->ne[1];
    int64_t        up_f = full_of(2) - t->ne[2];
    // padding's vectors are in FORMAT order (b,f,y,x for bfyx) - NOT the tensor
    // simple-ctor order (b,f,x,y) used just above for `used`.
    cldnn::padding pad({ 0, 0, 0, 0 }, { 0, up_f, up_y, up_x });
    return cldnn::layout{ ggml_type_to_cldnn(t->type), cldnn::format::bfyx, used, pad };
}

// Dispatcher used as compiled_op::layout_for when a translator supports both
// contiguous inputs (flat reduce-ne0 layout) and strided "padded view" inputs
// (per-dim padded layout) - e.g. build_rms_norm. Always matches the layout
// used to build the corresponding cldnn::input_layout/reorder in the topology.
cldnn::layout ggml_layout_for_reduce_ne0_maybe_padded(const ggml_tensor * t) {
    if (ggml_is_contiguous(t)) {
        return ggml_layout_for_reduce_ne0(t);
    }
    return ggml_layout_for_padded(t);
}

// Like ggml_layout_for_padded, but in ggml_layout_for's FLAT-BATCH axis
// convention (ne[1..3] collapsed into one cldnn batch axis, ne[0] -> feature)
// instead of ggml_layout_for_padded's per-dim one (ne0->x,ne1->y,ne2->f,ne3->b).
// A cldnn::reorder only changes format/strips padding for a FIXED (b,f,x,y) shape
// - it does not reshape between different axis decompositions of the same total
// size - so the padded raw-bind layout used by build_mul_mat (which needs to land
// in ggml_layout_for's shape for fully_connected) must be expressed in that same
// flat-batch shape from the start, with the padding (if any) attached to the
// feature axis (ne[0], where MUL_MAT's k_v test pads: a wider per-row allocation
// sliced down via ggml_view_4d). This requires ne[1..3] themselves to be densely
// nested with no padding of their own (checked below) - a stronger condition than
// ggml_is_padded_view alone guarantees, but the one MUL_MAT's k_v view satisfies.
cldnn::layout ggml_layout_for_maybe_padded(const ggml_tensor * t) {
    if (ggml_is_contiguous(t)) {
        return ggml_layout_for(t);
    }
    if (!ggml_is_padded_view(t)) {
        throw std::runtime_error("OVGPU: ggml_layout_for_maybe_padded: not a padded view");
    }
    for (int i = 2; i < GGML_MAX_DIMS; i++) {
        if (t->nb[i] != t->nb[i - 1] * t->ne[i - 1]) {
            throw std::runtime_error("OVGPU: ggml_layout_for_maybe_padded: padding outside ne[0] unsupported");
        }
    }
    auto v = [](int64_t n) {
        return (cldnn::tensor::value_type) std::max<int64_t>(n, 1);
    };
    int64_t        batch  = t->ne[1] * t->ne[2] * t->ne[3];
    int64_t        f_used = t->ne[0];
    int64_t        f_full = (int64_t) (t->nb[1] / t->nb[0]);  // elements per row in the backing alloc
    cldnn::tensor  used(v(batch), v(f_used), 1, 1);
    // padding's vectors are in FORMAT order (b,f,y,x) - pad the feature axis only.
    cldnn::padding pad({ 0, 0, 0, 0 }, { 0, f_full - f_used, 0, 0 });
    return cldnn::layout{ ggml_type_to_cldnn(t->type), cldnn::format::bfyx, used, pad };
}

// Sorts t's dims (excluding any with ne[i]==1 - their stride is never read, same
// skip ggml_is_contiguous applies) by ascending stride into `order`/`n_out`, then
// checks the ggml_is_padded_view nesting condition (packed fastest dim; each next
// dim's stride a whole multiple of the previous dim's used extent) IN THAT SORTED
// ORDER rather than ggml's fixed dim order 0,1,2,3. This is what lets a permuted
// view (where ggml's dim0 is no longer the fastest-varying one) still qualify.
static bool ggml_get_strided_order(const ggml_tensor * t, int * order, int * n_out) {
    int axes[GGML_MAX_DIMS];
    int n = 0;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (t->ne[i] > 1) {
            axes[n++] = i;
        }
    }
    std::sort(axes, axes + n, [&](int a, int b) { return t->nb[a] < t->nb[b]; });
    if (n > 0 && t->nb[axes[0]] != ggml_type_size(t->type)) {
        return false;  // fastest dim must be packed
    }
    for (int k = 1; k < n; k++) {
        int64_t nb_prev = t->nb[axes[k - 1]];
        int64_t nb_cur  = t->nb[axes[k]];
        if (nb_prev == 0 || nb_cur % nb_prev != 0) {
            return false;
        }
        if (nb_cur / nb_prev < t->ne[axes[k - 1]]) {
            return false;  // full extent must contain the used one
        }
    }
    for (int k = 0; k < n; k++) {
        order[k] = axes[k];
    }
    *n_out = n;
    return true;
}

bool ggml_is_strided_view(const ggml_tensor * t) {
    int order[GGML_MAX_DIMS], n;
    return ggml_get_strided_order(t, order, &n);
}

// Zero-copy bind layout for a ggml_is_strided_view tensor: places the
// smallest-stride ggml dim on cldnn x (innermost), next on y, next on f, next on
// b - i.e. matches the tensor's ACTUAL memory nesting, unlike ggml_layout_for_padded
// which assumes ne[0] is already fastest. Must be followed by a cldnn::permute
// (ggml_strided_permute_order) to reach the fixed ne0->x/ne1->y/ne2->f/ne3->b
// convention every op translator expects.
static cldnn::layout ggml_layout_for_strided(const ggml_tensor * t) {
    int  order[GGML_MAX_DIMS], n;
    bool ok = ggml_get_strided_order(t, order, &n);
    GGML_ASSERT(ok);
    auto v = [](int64_t x) {
        return (cldnn::tensor::value_type) std::max<int64_t>(x, 1);
    };
    int64_t ne_slot[GGML_MAX_DIMS] = { 1, 1, 1, 1 };  // [0]=x [1]=y [2]=f [3]=b
    int64_t up_slot[GGML_MAX_DIMS] = { 0, 0, 0, 0 };
    for (int k = 0; k < n; k++) {
        int axis     = order[k];
        ne_slot[k]   = t->ne[axis];
        int64_t full = (k + 1 < n) ? (int64_t) (t->nb[order[k + 1]] / t->nb[axis]) : t->ne[axis];
        up_slot[k]   = full - t->ne[axis];
    }
    cldnn::tensor  used(v(ne_slot[3]), v(ne_slot[2]), v(ne_slot[0]), v(ne_slot[1]));  // ctor order (b,f,x,y)
    cldnn::padding pad({ 0, 0, 0, 0 }, { 0, up_slot[2], up_slot[1], up_slot[0] });    // format order (b,f,y,x)
    return cldnn::layout{ ggml_type_to_cldnn(t->type), cldnn::format::bfyx, used, pad };
}

// cldnn::permute order (OV FORMAT-order slot numbering: 0=b,1=f,2=y,3=x - i.e.
// ov::Shape/PartialShape order, matching permute.cpp's output_shape[i] =
// input_shape[permute_order[i]] - NOT the tensor simple-ctor order (b,f,x,y)
// used for `used`/`ne_slot` above; same swap gotcha as cldnn::padding's vectors)
// that moves a ggml_layout_for_strided(t)-bound tensor's dims from their ACTUAL
// physical slots into the fixed ne0->x/ne1->y/ne2->f/ne3->b convention.
static std::vector<uint16_t> ggml_strided_permute_order(const ggml_tensor * t) {
    int  order[GGML_MAX_DIMS], n;
    bool ok = ggml_get_strided_order(t, order, &n);
    GGML_ASSERT(ok);
    // format-order slot (0=b,1=f,2=y,3=x) each sorted position k lands in when
    // bound via ggml_layout_for_strided: k=0 (fastest) -> x, k=1 -> y, k=2 -> f, k=3 -> b.
    static const int k_to_slot[GGML_MAX_DIMS] = { 3, 2, 1, 0 };
    int              axis_slot[GGML_MAX_DIMS];
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        axis_slot[i] = -1;
    }
    for (int k = 0; k < n; k++) {
        axis_slot[order[k]] = k_to_slot[k];
    }
    // dims with ne[axis]==1 didn't appear in `order` at all - assign them whichever
    // slot is still free (size 1 either way, so the choice doesn't matter).
    bool slot_used[GGML_MAX_DIMS] = { false, false, false, false };
    for (int k = 0; k < n; k++) {
        slot_used[k_to_slot[k]] = true;
    }
    int next_free = 0;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (axis_slot[i] == -1) {
            while (next_free < GGML_MAX_DIMS && slot_used[next_free]) {
                next_free++;
            }
            axis_slot[i]         = next_free;
            slot_used[next_free] = true;
        }
    }
    // target (logical) slot for ggml dim i: dim0->x,dim1->y,dim2->f,dim3->b
    static const int      axis_to_target_slot[GGML_MAX_DIMS] = { 3, 2, 1, 0 };
    std::vector<uint16_t> perm(GGML_MAX_DIMS);
    for (int axis = 0; axis < GGML_MAX_DIMS; axis++) {
        perm[axis_to_target_slot[axis]] = (uint16_t) axis_slot[axis];
    }
    return perm;
}

// SDPA (FLASH_ATTN_EXT) input binding info for a ggml q/k/v operand:
//  - src_layout:     physical-nesting layout (zero-copy external-memory bind). MUST match
//                    co->layout_for (contiguous -> ggml_layout_for_eltwise, else
//                    ggml_layout_for_strided) so cldnn's input-memory layout check passes.
//  - order:          per-input SDPA input_*_transpose_order = which physical slot
//                    (0=b,1=f,2=y,3=x) holds each logical dim (B,H,S,D = ggml axes 3,2,1,0).
//                    Identity {0,1,2,3} for contiguous (physical BHSD); {0,2,1,3} for a
//                    permute(0,2,1,3) view (physical BSHD) - the OV unit-test config
//                    (sdpa_gpu_test.cpp:94-95).
//  - compact_layout: compact f32 buffer in the SAME slot order (cast+strip-padding target
//                    for the reorder when native dtype != f32).
struct sdpa_input_bind {
    cldnn::layout            src_layout;     // physical nesting (external-memory bind)
    std::vector<int64_t>     order;          // SDPA input_*_transpose_order
    cldnn::layout            compact_layout; // compact f32, same slot order (cast target)
};

static sdpa_input_bind ggml_sdpa_input_bind(const ggml_tensor * t) {
    ov::element::Type native_dt = ggml_type_to_cldnn(t->type);
    auto v = [](int64_t x) {
        return (cldnn::tensor::value_type) std::max<int64_t>(x, 1);
    };

    if (ggml_is_contiguous(t)) {
        // Contiguous: physical nesting is BHSD (eltwise: ne0->x, ne1->y, ne2->f, ne3->b),
        // so identity transpose order maps logical BHSD -> physical BHSD. Matches
        // ggml_layout_for_eltwise exactly (what co->layout_for returns for contiguous).
        cldnn::layout src = ggml_layout_for_eltwise(t);
        cldnn::layout compact{ ov::element::f32, cldnn::format::bfyx, src.get_tensor() };
        return { src, {0, 1, 2, 3}, compact };
    }

    // Strided/permuted view: bind physical nesting (ggml_layout_for_strided) and compute
    // the per-input transpose order from the stride sort.
    int  order_arr[GGML_MAX_DIMS], n;
    bool ok = ggml_get_strided_order(t, order_arr, &n);
    GGML_ASSERT(ok);

    // ne_slot[k] = ggml ne of the axis at sorted position k (k=0 fastest -> x, ... -> b).
    int64_t ne_slot[GGML_MAX_DIMS] = { 1, 1, 1, 1 };  // [0]=x [1]=y [2]=f [3]=b
    for (int k = 0; k < n; k++) {
        ne_slot[k] = t->ne[order_arr[k]];
    }

    // axis_slot[axis] = format-order slot (0=b,1=f,2=y,3=x) holding ggml `axis`.
    // k=0 (fastest)->x(3), k=1->y(2), k=2->f(1), k=3->b(0); size-1 axes (absent from the
    // sorted order) take any free slot (extent 1, so the choice is immaterial - the slot's
    // physical extent is also 1, so the logical<->physical extent always matches).
    static const int k_to_slot[GGML_MAX_DIMS] = { 3, 2, 1, 0 };
    int              axis_slot[GGML_MAX_DIMS]  = { -1, -1, -1, -1 };
    bool             slot_used[GGML_MAX_DIMS] = { false, false, false, false };
    for (int k = 0; k < n; k++) {
        axis_slot[order_arr[k]] = k_to_slot[k];
        slot_used[k_to_slot[k]] = true;
    }
    int next_free = 0;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (axis_slot[i] == -1) {
            while (next_free < GGML_MAX_DIMS && slot_used[next_free]) {
                next_free++;
            }
            axis_slot[i]         = next_free;
            slot_used[next_free] = true;
        }
    }

    // SDPA transpose_order[i] = physical slot of logical dim i; canonical logical order
    // is (B, H, S, D) = ggml axes (3, 2, 1, 0).
    std::vector<int64_t> order = {
        (int64_t) axis_slot[3], (int64_t) axis_slot[2],
        (int64_t) axis_slot[1], (int64_t) axis_slot[0]
    };

    // src_layout == ggml_layout_for_strided(t) (guaranteed to match co->layout_for).
    // compact = same used dims, f32, no padding (cast target strips the view's padding).
    cldnn::tensor used(v(ne_slot[3]), v(ne_slot[2]), v(ne_slot[0]), v(ne_slot[1]));
    cldnn::layout src = ggml_layout_for_strided(t);
    cldnn::layout compact{ ov::element::f32, cldnn::format::bfyx, used };
    return { src, std::move(order), compact };
}

cldnn::layout ggml_layout_for_eltwise_maybe_strided(const ggml_tensor * t) {
    if (ggml_is_contiguous(t)) {
        return ggml_layout_for_eltwise(t);
    }
    return ggml_layout_for_strided(t);
}

// Dispatcher for CPY: flat-batch (ggml_layout_for) for a contiguous tensor, strided
// (ggml_layout_for_strided) for a permuted/strided one. Used as compiled_op::layout_for
// so graph_compute binds a contiguous a / the node as flat, and a strided a via its
// physical memory nesting (matching the input_layout built in build_cpy's strided
// branch). The cache key (which includes nb[]) keeps contiguous-a and strided-a CPYs
// as separate compiled networks.
cldnn::layout ggml_layout_for_maybe_strided(const ggml_tensor * t) {
    if (ggml_is_contiguous(t)) {
        return ggml_layout_for(t);
    }
    return ggml_layout_for_strided(t);
}

cldnn::memory::ptr wrap_tensor(cldnn::engine &       engine,
                               cldnn::memory &       base_mem,
                               const cldnn::layout & l,
                               size_t                byte_offset) {
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
        if (!src) {
            continue;  // optional src absent (e.g. SOFT_MAX mask=NULL with
                       // sinks=src[2]) - keep scanning so later srcs ARE in the
                       // key (else two ops differing only in a later src would
                       // collide and share a wrong compiled network).
        }
        s << "|s" << i << ":" << (int) src->type << ":";
        for (int d = 0; d < GGML_MAX_DIMS; d++) s << src->ne[d] << ",";
        // nb[] too: a translator's topology can differ (contiguous fast path vs.
        // a strided/permuted-view path with extra permute/reorder nodes) for two
        // tensors sharing the same ne[] but different strides - without this the
        // cache would wrongly share one compiled network between them.
        s << ":";
        for (int d = 0; d < GGML_MAX_DIMS; d++) {
            s << src->nb[d] << ",";
        }
        // op params that affect the kernel (transpose flags etc.)
    }
    // a few op-specific params
    switch (node->op) {
        case GGML_OP_MUL_MAT:
            s << "|transposed=" << (node->op_params ? ggml_get_op_params_i32(node, 0) : 0);
            break;
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            // n_dims, mode, n_ctx_orig, freq_base, freq_scale, ext_factor,
            // attn_factor, beta_fast, beta_slow and the 4 mrope sections are ALL
            // baked as -D build_options (custom_gpu has no scalar-arg mechanism),
            // so they MUST be part of the cache key - two ROPE ops with the same
            // tensor shapes but different n_dims/mode/sections/YaRN-params would
            // otherwise reuse a kernel compiled with the wrong constants. Dump all
            // 15 op_params ints (the cheap, fully-correct option).
            if (node->op_params) {
                const int32_t * p = (const int32_t *) node->op_params;
                s << "|rp=";
                for (int i = 1; i < 15; i++) {  // [0]=n_past unused, skip
                    s << p[i] << ",";
                }
            }
            break;
        case GGML_OP_RMS_NORM: {
            // eps is baked into the cldnn::rms primitive at build time. Two same-
            // shape RMS_NORMs with different eps would otherwise share a network
            // with the wrong eps (the cache persists across ops - same class of
            // bug as ROPE, confirmed manifesting; see WORKLOG).
            float eps;
            memcpy(&eps, node->op_params, sizeof(float));
            s << "|eps=" << eps;
            break;
        }
        case GGML_OP_SOFT_MAX: {
            // scale/max_bias affect the kernel; include for correctness (supported
            // cases currently have scale==1, max_bias==0 - defense-in-depth).
            float scale, max_bias;
            memcpy(&scale,    node->op_params + 0, sizeof(float));
            memcpy(&max_bias, node->op_params + 1, sizeof(float));
            s << "|scale=" << scale << "|maxbias=" << max_bias;
            break;
        }
        case GGML_OP_UNARY:
            // unary subtype (SILU/...) selects the kernel - include it.
            s << "|un=" << ggml_get_op_params_i32(node, 0);
            break;
        case GGML_OP_GLU:
            // op (SWIGLU/GEGLU/...) + swapped are baked as -D.
            s << "|glu=" << ggml_get_op_params_i32(node, 0)
              << "|sw=" << ggml_get_op_params_i32(node, 1);
            break;
        case GGML_OP_CONCAT:
            // dim is baked as -D.
            s << "|dim=" << ggml_get_op_params_i32(node, 0);
            break;
        case GGML_OP_FLASH_ATTN_EXT: {
            // scale/max_bias/softcap are baked into the SDPA primitive (scale_val field,
            // and the ALiBi/softcap cases gate to CPU). Include for correctness.
            float scale, max_bias, logit_softcap;
            memcpy(&scale,        node->op_params + 0, sizeof(float));
            memcpy(&max_bias,     node->op_params + 1, sizeof(float));
            memcpy(&logit_softcap, node->op_params + 2, sizeof(float));
            s << "|fa=" << scale << "," << max_bias << "," << logit_softcap;
            break;
        }
        default: break;
    }
    return s.str();
}

std::unique_ptr<compiled_op> op_cache::get_or_build(const ggml_tensor * node) {
    std::string key = key_for(node);
    std::lock_guard<std::mutex> lock(m_mutex);
    // Diagnostic knob: GGML_OVGPU_NO_CACHE=1 bypasses the cache entirely (every
    // node rebuilds) - used to isolate cache-key bugs from genuine build failures.
    static const bool no_cache = getenv("GGML_OVGPU_NO_CACHE") != nullptr;
    auto it = no_cache ? m_cache.end() : m_cache.find(key);
    if (it != m_cache.end()) {
        auto co = std::make_unique<compiled_op>();
        co->net        = it->second->net;
        co->input_ids  = it->second->input_ids;
        co->output_id  = it->second->output_id;
        co->weight_ids = it->second->weight_ids;
        co->layout_for = it->second->layout_for;
        return co;
    }
    // A translator's cldnn::network construction can throw (e.g. oneDNN rejects a
    // shape/dtype combo supports_op couldn't fully predict) - catch it here so an
    // unexpected rejection degrades to "no translator" (graph_compute fails that
    // compute cleanly) instead of an uncaught exception crashing the process.
    std::unique_ptr<compiled_op> co;
    try {
        co = build_op(node, *engine, stream);
    } catch (const std::exception & e) {
        // fprintf, not GGML_LOG: tools like llama-bench install a null log
        // callback that would swallow the one clue identifying the failure.
        fprintf(stderr, "OVGPU: build_op threw for op %d (%s): %s\nkey=%s\n",
                node->op, ggml_op_name(node->op), e.what(), key.c_str());
        return nullptr;
    }
    if (co && !no_cache) {
        auto stored = std::make_shared<compiled_op>();
        stored->net        = co->net;
        stored->input_ids  = co->input_ids;
        stored->output_id  = co->output_id;
        stored->weight_ids = co->weight_ids;
        stored->layout_for = co->layout_for;
        m_cache[key] = std::move(stored);
    }
    if (!co) {
        // The translator rejected this shape silently (its early-return guards) -
        // dump the full cache key (op + every src's type/ne/nb) so the rejecting
        // guard can be identified from the log. fprintf, not GGML_LOG: tools like
        // llama-bench install a null log callback that would swallow it.
        fprintf(stderr, "OVGPU: build_op returned null for op %d (%s), key=%s\n",
                node->op, ggml_op_name(node->op), key.c_str());
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

// Append the fully_connected compute primitive (NO trailing reorder, NO make_network)
// to `topo`. `prefix` namespaces the primitive ids so a chain head + its children can
// share one topology without id collisions. Returns the FC compute-output id
// (prefix+"fc_out"). Caller must have validated the node (2D weight, ok operands).
static std::string mul_mat_core(cldnn::topology & topo, const ggml_tensor * node,
                                cldnn::engine & engine, const std::string & prefix) {
    const ggml_tensor * weight = node->src[0];
    const ggml_tensor * act    = node->src[1];
    const std::string   w_id   = prefix + "in_0"; // weight     (src[0])
    const std::string   a_id   = prefix + "in_1"; // activation (src[1])

    // Bind a possibly-strided/permuted/padded operand and compact it into the flat
    // layout FC expects. Three cases:
    //  (a) contiguous: single input_layout (flat). No-op.
    //  (b) flat-padded view (k_v sub-box): bind flat-padded -> reorder strips padding.
    //  (c) strided/permuted view (e.g. permuted KV cache): bind strided (physical
    //      nesting) -> permute to ne0->x/ne1->y/ne2->f/ne3->b -> reorder (strip pad
    //      + cast to eltwise dense) -> reshape to flat-batch (free reinterpret).
    auto v = [](int64_t n) {
        return (cldnn::tensor::value_type) std::max<int64_t>(n, 1);
    };
    auto bind_operand = [&](const std::string & id, const ggml_tensor * t) -> std::string {
        if (ggml_is_contiguous(t)) {
            topo.add(cldnn::input_layout(id, ggml_layout_for(t)));
            return id;
        }
        if (ggml_is_flat_padded_view(t)) {
            topo.add(cldnn::input_layout(id, ggml_layout_for_maybe_padded(t)));
            const std::string compact_id = id + "_compact";
            topo.add(cldnn::reorder(compact_id, cldnn::input_info(id), ggml_layout_for(t)));
            return compact_id;
        }
        // strided/permuted view: bind physical, permute to logical, compact, reshape to flat.
        topo.add(cldnn::input_layout(id, ggml_layout_for_strided(t)));
        const std::string perm_id    = id + "_perm";
        topo.add(cldnn::permute(perm_id, cldnn::input_info(id), ggml_strided_permute_order(t)));
        const std::string compact_id = id + "_compact";
        cldnn::tensor    eltwise_ten(v(t->ne[3]), v(t->ne[2]), v(t->ne[0]), v(t->ne[1]));
        cldnn::layout    eltwise_lay{ ggml_type_to_cldnn(t->type), cldnn::format::bfyx, eltwise_ten };
        topo.add(cldnn::reorder(compact_id, cldnn::input_info(perm_id), eltwise_lay));
        const std::string flat_id = id + "_flat";
        int64_t batch = t->ne[1] * t->ne[2] * t->ne[3];
        topo.add(cldnn::reshape(flat_id, cldnn::input_info(compact_id),
                                cldnn::tensor(v(batch), v(t->ne[0]), 1, 1)));
        return flat_id;
    };
    const std::string w_bound = bind_operand(w_id, weight);
    const std::string a_bound = bind_operand(a_id, act);

    // Operand-dtype strategy. oneDNN FC (the path taken on supports_immad devices)
    // accepts only MATCHING-DTYPE dense pairs: f16f16, bf16bf16, f32f32
    // (see fully_connected_onednn.hpp validate_impl). ggml accumulates in f32 always
    // and reads f16/bf16 weights per-element into f32; oneDNN's f16/bf16 matmul on
    // Intel GPU accumulates in f32 too (DPAS), so f16f16->f32 out matches ggml
    // semantics - the only difference is the activation gets f16/bf16-rounded,
    // which every f16-weight backend does anyway. On non-immad devices the ocl path
    // mis-reads f16 weights for some batch sizes, so both operands upcast to f32.
    const bool        supports_immad = engine.get_device_info().supports_immad;
    ov::element::Type compute_dt     = ov::element::f32;  // default: both -> f32
    if (supports_immad) {
        if (weight->type == GGML_TYPE_F16) {
            compute_dt = ov::element::f16;
        } else if (weight->type == GGML_TYPE_BF16) {
            compute_dt = ov::element::bf16;
        }
        // f32 weight -> f32 (already matching)
    }

    auto convert = [&](const std::string & in_id, const std::string & out_id, const ggml_tensor * t) -> std::string {
        ov::element::Type t_dt = ggml_type_to_cldnn(t->type);
        if (t_dt == compute_dt) {
            return in_id;  // already the compute dtype, no reorder
        }
        cldnn::layout l{ compute_dt, cldnn::format::bfyx, ggml_ne_to_tensor(t) };
        topo.add(cldnn::reorder(out_id, cldnn::input_info(in_id), l));
        return out_id;
    };
    std::string fc_weight = convert(w_bound, prefix + "w_conv", weight);
    std::string fc_input  = convert(a_bound, prefix + "act_conv", act);

    const std::string fc_out_id = prefix + "fc_out";
    topo.add(cldnn::fully_connected(fc_out_id, cldnn::input_info(fc_input), fc_weight, "",
                                    2 /*input_size*/, 2 /*weights_rank*/, true /*weights_transposed*/));
    return fc_out_id;
}

static std::unique_ptr<compiled_op> build_mul_mat(const ggml_tensor * node, cldnn::engine & engine, cldnn::stream::ptr stream) {
    const ggml_tensor * weight = node->src[0];
    const ggml_tensor * act    = node->src[1];
    if (!weight || !act) return nullptr;
    if (!type_supported(weight->type) || !type_supported(act->type)) return nullptr;
    if (ggml_n_dims(weight) != 2) return nullptr; // weight must be 2D (matches supports_op)
    const bool w_contig = ggml_is_contiguous(weight);
    const bool a_contig = ggml_is_contiguous(act);
    // Accept contiguous, flat-padded-view (k_v sub-box), OR strided-view (permuted/
    // transposed - e.g. the decomposed attention's permuted KV-cache q/k/v views).
    auto ok_operand = [](const ggml_tensor * t) {
        return ggml_is_contiguous(t) || ggml_is_flat_padded_view(t) || ggml_is_strided_view(t);
    };
    if (!ok_operand(weight) || !ok_operand(act)) {
        return nullptr;
    }
    (void) w_contig; (void) a_contig;

    auto co = std::make_unique<compiled_op>();
    // input_ids follow ggml src order so graph_compute binds src[k] -> input_ids[k].
    //   src[0] = weight, src[1] = activation.
    const std::string w_id = "in_0"; // weight     (src[0])
    const std::string a_id = "in_1"; // activation (src[1])
    co->input_ids  = {w_id, a_id};
    co->weight_ids = {w_id};
    co->output_id  = "out";
    // layout_for dispatches to match the input_layout each operand is bound with:
    // flat for contiguous, flat-padded for k_v sub-box, strided for permuted views.
    co->layout_for = [](const ggml_tensor * t) -> cldnn::layout {
        if (ggml_is_contiguous(t)) {
            return ggml_layout_for(t);
        }
        if (ggml_is_flat_padded_view(t)) {
            return ggml_layout_for_maybe_padded(t);
        }
        return ggml_layout_for_strided(t);
    };

    cldnn::topology topo;
    std::string    fc_out = mul_mat_core(topo, node, engine, "" /*prefix*/);

    // Trailing reorder forces bfyx f32 output: cldnn's layout optimizer can pick a
    // different output format for some shapes (e.g. yxfb for batch=8, a transpose of
    // bfyx for 2D), which would garble the ggml tensor.
    topo.add(cldnn::reorder(co->output_id, cldnn::input_info(fc_out), ggml_layout_for(node)));

    cldnn::ExecutionConfig config = make_config();
    // NOTE: do NOT set allow_new_shape_infer(true) - oneDNN FC rank-2 reshape breaks
    // under new shape infer (see make_config note + WORKLOG).
    co->net = make_network(engine, stream, topo, config);
    return co;
}

// Build a network config with the oneDNN-engagement settings shared by all op
// translators: optimize_data (keeps plain bfyx layouts) + use_onednn (selects the
// oneDNN impl where available). Do NOT set allow_new_shape_infer - see MUL_MAT note.
static cldnn::ExecutionConfig make_config() {
    cldnn::ExecutionConfig config;
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::use_onednn(true));
    return config;
}

// ---------------------------------------------------------------------------
// ADD/MUL -> eltwise (sum/prod)
// ---------------------------------------------------------------------------

// ggml ADD/MUL: dst = a (+|*) b, with ggml_can_repeat(b, a) broadcast semantics
// (b is "repeatable" to a's shape - numpy-style broadcast along b's trailing dims).
// dst type == a->type (NOT widened to f32 like MUL_MAT). Both inputs are runtime
// activations (e.g. attention output + residual), so input_layout x2, shape-keyed.
// Either operand may also be a "strided view" (ggml_is_strided_view: a sub-box
// view and/or a permute/transpose, e.g. test_bin_bcast's perm1) - bound via its
// actual memory nesting, then permuted+reordered into the layout below.
static std::unique_ptr<compiled_op> build_eltwise(const ggml_tensor * node, cldnn::engine & engine, cldnn::stream::ptr stream) {
    const ggml_tensor * a = node->src[0];
    const ggml_tensor * b = node->src[1];
    if (!a || !b) {
        return nullptr;
    }
    if (!type_supported(a->type) || !type_supported(b->type)) {
        return nullptr;
    }
    const bool a_contig = ggml_is_contiguous(a);
    const bool b_contig = ggml_is_contiguous(b);
    if (!a_contig && !ggml_is_strided_view(a)) {
        return nullptr;
    }
    if (!b_contig && !ggml_is_strided_view(b)) {
        return nullptr;
    }

    auto              co   = std::make_unique<compiled_op>();
    const std::string a_id = "in_0";
    const std::string b_id = "in_1";
    co->input_ids          = { a_id, b_id };
    co->output_id          = "out";
    // Use the broadcast-preserving layout so numpy eltwise broadcast sees each ggml
    // dim independently (flatten would merge a broadcast dim into a non-1 dim -> crash).
    co->layout_for         = ggml_layout_for_eltwise_maybe_strided;

    cldnn::eltwise_mode mode = (node->op == GGML_OP_ADD) ? cldnn::eltwise_mode::sum : cldnn::eltwise_mode::prod;

    cldnn::topology topo;
    // Bind a possibly-strided operand: raw bind via its actual memory nesting,
    // then permute+reorder into the fixed ne0->x/ne1->y/ne2->f/ne3->b layout
    // eltwise expects. No-op (zero extra nodes) for an already-contiguous operand.
    auto            bind = [&](const std::string & id, const ggml_tensor * t, bool contig) -> std::string {
        if (contig) {
            topo.add(cldnn::input_layout(id, ggml_layout_for_eltwise(t)));
            return id;
        }
        topo.add(cldnn::input_layout(id, ggml_layout_for_strided(t)));
        const std::string perm_id = id + "_perm";
        topo.add(cldnn::permute(perm_id, cldnn::input_info(id), ggml_strided_permute_order(t)));
        const std::string compact_id = id + "_compact";
        topo.add(cldnn::reorder(compact_id, cldnn::input_info(perm_id), ggml_layout_for_eltwise(t)));
        return compact_id;
    };
    const std::string a_bound = bind(a_id, a, a_contig);
    const std::string b_bound = bind(b_id, b, b_contig);
    topo.add(cldnn::eltwise("ew", cldnn::input_info(a_bound), cldnn::input_info(b_bound), mode));
    // Trailing reorder -> ggml dst layout (a's dtype, bfyx, broadcast-preserving).
    // ADD/MUL dst type == a->type (NOT f32 like MUL_MAT). cldnn may pick a different
    // output format otherwise, garbling the ggml tensor.
    topo.add(cldnn::reorder(co->output_id, cldnn::input_info("ew"), ggml_layout_for_eltwise(node)));

    co->net = make_network(engine, stream, topo, make_config());
    return co;
}

// ---------------------------------------------------------------------------
// RMS_NORM -> rms (no gamma; gamma is a separate ggml MUL op)
// ---------------------------------------------------------------------------

// ggml RMS_NORM: dst = src0 / sqrt(mean(src0^2, over ne[0]) + eps). Normalizes over
// ne[0] (the contiguous/feature dim). NO gamma - ggml applies the weight via a separate
// ggml_mul (or a fused RMS_NORM+MUL). dst type == src0. epsilon in op_params[0].
// cldnn rms(id, input, epsilon) = no-gamma variant (elementwise_affine=false).
// Append the rms compute primitive (NO trailing reorder, NO make_network) to `topo`.
// `prefix` namespaces the primitive ids so a chain head + its children can share one
// topology without id collisions. Returns the rms compute-output primitive id
// (prefix+"rms"). Caller must have validated the node (contiguous or padded view).
static std::string rms_core(cldnn::topology & topo, const ggml_tensor * node, const std::string & prefix) {
    const ggml_tensor * a = node->src[0];
    float               eps;
    memcpy(&eps, node->op_params, sizeof(float));
    const std::string a_id   = prefix + "in_0";
    const std::string rms_id = prefix + "rms";
    if (ggml_is_contiguous(a)) {
        topo.add(cldnn::input_layout(a_id, ggml_layout_for_reduce_ne0(a)));
        topo.add(cldnn::rms(rms_id, cldnn::input_info(a_id), eps));
    } else {
        // Non-contiguous "padded view" (e.g. ggml_view_4d of a sub-box). Bind via
        // the per-dim padded layout, then: (1) reorder strips the padding, yielding
        // a DENSE per-dim tensor - note cldnn::reorder does NOT reshape: it keeps the
        // SOURCE's used tensor and only adopts the target's dtype/padding/format
        // (reorder.cpp:172-174); (2) a reshape flattens that to the reduce-ne0 layout
        // (b=ne1*ne2*ne3, f=1, x=ne0, y=1) so rms sees y=1 and reduces over ne0 only -
        // correct for BOTH rms_gpu_bfyx_opt (reduces over X) AND rms_gpu_ref (over
        // X*Y*Z). Without the reshape, ref would reduce over ne0*ne1 -> wrong scale.
        // The reshape is a free reinterpret: dense per-dim and flat share ggml byte
        // order (((i3*ne2+i2)*ne1+i1)*ne0+i0).
        topo.add(cldnn::input_layout(a_id, ggml_layout_for_padded(a)));
        const std::string compact_id = prefix + "compact";
        topo.add(cldnn::reorder(compact_id, cldnn::input_info(a_id), ggml_layout_for_reduce_ne0(a)));
        auto               v = [](int64_t n) {
            return (cldnn::tensor::value_type) std::max<int64_t>(n, 1);
        };
        int64_t            batch = a->ne[1] * a->ne[2] * a->ne[3];
        const std::string  flat_id = prefix + "flat";
        topo.add(cldnn::reshape(flat_id, cldnn::input_info(compact_id), cldnn::tensor(v(batch), 1, v(a->ne[0]), 1)));
        topo.add(cldnn::rms(rms_id, cldnn::input_info(flat_id), eps));
    }
    return rms_id;
}

static std::unique_ptr<compiled_op> build_rms_norm(const ggml_tensor * node, cldnn::engine & engine, cldnn::stream::ptr stream) {
    const ggml_tensor * a = node->src[0];
    if (!a) {
        return nullptr;
    }
    if (!type_supported(a->type)) {
        return nullptr;
    }
    const bool contig = ggml_is_contiguous(a);
    if (!contig && !ggml_is_padded_view(a)) {
        return nullptr;
    }

    auto              co   = std::make_unique<compiled_op>();
    const std::string a_id = "in_0";
    co->input_ids          = { a_id };
    co->output_id          = "out";
    // cldnn rms reduces over the SPATIAL dims (x,y,z - see rms_gpu_ref.cl), while
    // ggml RMS_NORM normalizes over ne[0]. So put ne[0] on x (the eltwise/broadcast
    // layout: ne[0]->x innermost, ne[1]->y, ne[2]->f, ne[3]->b) so the spatial
    // reduction = ne[0]. b/f/y carry the per-row grouping. Zero-copy (ne[0] on x).
    co->layout_for         = ggml_layout_for_reduce_ne0_maybe_padded;

    cldnn::topology   topo;
    std::string       rms_id = rms_core(topo, node, "" /*prefix*/);
    // Trailing reorder -> ggml dst layout (a's dtype, reduce-ne0). dst type == a->type.
    topo.add(cldnn::reorder(co->output_id, cldnn::input_info(rms_id), ggml_layout_for_reduce_ne0(node)));

    co->net = make_network(engine, stream, topo, make_config());
    return co;
}

// SOFT_MAX custom OpenCL kernel (mask + ALiBi + sinks + scale). Faithful port
// of ggml-cpu's ggml_compute_forward_soft_max_f32 (the test reference). Handles
// the cases cldnn softmax can't: an additive mask with ggml's modular nr23
// broadcast (i02%ne12, i03%ne13), ALiBi (per-head slope*mask), and sinks (a per-
// head max-prior: max=max(max,sk[h]); sum+=exp(sk[h]-max)). Plus scale!=1 with no
// mask. The plain no-mask/scale==1 case stays on cldnn softmax (faster, handles
// huge ne0).
//
// src0 = scores [NE0,NE1,NE2,NE3] (contiguous); src1 = mask [NE0,NE1,NE12,NE13]
// (optional, MFLOAT f16/f32, broadcasts i11=i01, i12=i02%NE12, i13=i03%NE13);
// src2 = sinks [NE2] (optional, F32 per head); dst = src0 shape/type (contiguous).
// Per row (i01,i02,i03): wp = src0*scale + slope*mask; max=max(wp) (or max(max,sk));
// sum=Σexp(wp-max) (or +exp(sk-max)); dst=exp(wp-max)/sum. ALiBi slope from
// max_bias via M0/M1/N_HEAD_LOG2 (host-computed). One work-item per row (loops NE0).
static const char * OVGPU_SOFT_MAX_KERNEL = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void ovgpu_soft_max(
    __global const DFLOAT * src0
#ifdef HAS_MASK
    , __global const MFLOAT * src1
#endif
#ifdef HAS_SINKS
    , __global const float * src2
#endif
    , __global       DFLOAT * dst)
{
    const int row = get_global_id(0);   // [0, NE1*NE2*NE3)
    if (row >= NE1*NE2*NE3) return;
    const int i01 = row % NE1;
    const int t1  = row / NE1;
    const int i02 = t1 % NE2;
    const int i03 = t1 / NE2;
    const long row_base = ((long)i03*NE2 + i02)*NE1 + i01;
    __global const DFLOAT * sp = src0 + row_base*(long)NE0;
    __global       DFLOAT * dp = dst   + row_base*(long)NE0;
#ifdef HAS_MASK
    const int i12 = i02 % NE12;
    const int i13 = i03 % NE13;
    __global const MFLOAT * mp = src1 + (((long)i13*NE12 + i12)*NE1 + i01)*(long)NE0;
#endif
    float slope = 1.0f;
#ifdef HAS_ALIBI
    {   // ALiBi: slope = h<n_head_log2 ? m0^(h+1) : m1^(2*(h-n_head_log2)+1).
        // (float) casts on M0/M1 disambiguate pow(): the bare -D decimal literals
        // otherwise match both pow(float,float) and pow(double,double) on the Intel
        // compiler (half-precision headers loaded) -> "call to 'pow' is ambiguous".
        const uint h = (uint)i02;
        slope = (h < N_HEAD_LOG2) ? pow((float)M0, (float)(h + 1))
                                 : pow((float)M1, (float)(2*(h - N_HEAD_LOG2) + 1));
    }
#endif
    // 1. parallel-ish max over the row (single work-item loop)
    float maxv = -INFINITY;
    for (int i = 0; i < NE0; i++) {
        float v = (float)sp[i] * SCALE;
#ifdef HAS_MASK
        v += slope * (float)mp[i];
#endif
        if (v > maxv) maxv = v;
    }
#ifdef HAS_SINKS
    const float sk = src2[i02];
    if (sk > maxv) maxv = sk;
#endif
    // 2. sum + write exp(dst) (exp is expensive; compute once)
    float sum = 0.0f;
    for (int i = 0; i < NE0; i++) {
        float v = (float)sp[i] * SCALE;
#ifdef HAS_MASK
        v += slope * (float)mp[i];
#endif
        float e = exp(v - maxv);
        dp[i] = (DFLOAT)e;
        sum += e;
    }
#ifdef HAS_SINKS
    sum += exp(sk - maxv);
#endif
    // 3. normalize
    const float inv = 1.0f / sum;
    for (int i = 0; i < NE0; i++) {
        dp[i] = (DFLOAT)((float)dp[i] * inv);
    }
}
)CLC";

// SILU custom OpenCL kernel: out = silu(x) = x / (1 + exp(-x)) (matches ggml-cpu
// ggml_vec_silu_f32, accurate exp not native_exp). Stride-aware (handles v=1
// padded views via src nb[]); dst is the new contiguous output. f32/f16.
static const char * OVGPU_SILU_KERNEL = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void ovgpu_silu(
    __global const char    * src,
    __global       DFLOAT  * dst)
{
    const long idx = get_global_id(0);   // [0, NE0*NE1*NE2*NE3) flat OUTPUT index
    if (idx >= NELS) return;
    long r = idx;
    const long i0 = r % NE0; r /= NE0;
    const long i1 = r % NE1; r /= NE1;
    const long i2 = r % NE2; r /= NE2;
    const long i3 = r;
    const long soff = i0*SNB0 + i1*SNB1 + i2*SNB2 + i3*SNB3;
    const float x = (float)*((__global const DFLOAT *)(src + soff));
    dst[idx] = (DFLOAT)(x / (1.0f + exp(-x)));
}
)CLC";

// GLU custom OpenCL kernel: SwiGLu only (out = silu(a) * b). Two variants:
//  - 2-input (ggml_glu_split / ggml_swiglu_split): a=src0 (gate), b=src1 (up),
//    both [NC, NE1, NE2, NE3] -> out [NC, NE1, NE2, NE3]. This is what Llama uses.
//  - 1-input (ggml_glu): input [2*NC, ...] split into a,b halves (swapped flag
//    selects which half is the gate). out [NC, ...].
// Other GLU ops (GEGLU/REGLU/...) -> CPU. Stride-aware (v=1 padded views). f32/f16.
static const char * OVGPU_GLU_KERNEL = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void ovgpu_glu(
    __global const char    * src0
#ifdef GLU_TWO_INPUT
    , __global const char * src1
#endif
    , __global       DFLOAT * dst)
{
    const long idx = get_global_id(0);   // [0, NC*NROWS)
    if (idx >= NC*NROWS) return;
    const long i0 = idx % NC;
    long r = idx / NC;
    const long i1 = r % NE1; r /= NE1;
    const long i2 = r % NE2; r /= NE2;
    const long i3 = r;
    const long row0 = i1*S0NB1 + i2*S0NB2 + i3*S0NB3;
#ifdef GLU_TWO_INPUT
    const float a = (float)*((__global const DFLOAT *)(src0 + i0*S0NB0 + row0));
    const float b = (float)*((__global const DFLOAT *)(src1 + i0*S1NB0 + i1*S1NB1 + i2*S1NB2 + i3*S1NB3));
#else
    // 1-input: gate/up are the two halves of src0 along dim0 (size 2*NC).
    // swapped: a = second half (i0+NC), b = first half (i0); else a=first, b=second.
    const long a0 = SWAPPED ? (i0 + NC) : i0;
    const long b0 = SWAPPED ? i0 : (i0 + NC);
    const float a = (float)*((__global const DFLOAT *)(src0 + a0*S0NB0 + row0));
    const float b = (float)*((__global const DFLOAT *)(src0 + b0*S0NB0 + row0));
#endif
    dst[idx] = (DFLOAT)((a / (1.0f + exp(-a))) * b);   // silu(a) * b
}
)CLC";

// CONCAT custom OpenCL kernel: dst = [src0, src1] along DIM (op_params[0]). For
// each output element, copy from src0 (if index < src0 extent along DIM) else
// src1 (offset back by src0's extent). Stride-aware (handles v=1 padded views);
// one work-item per output element copying ELEMSIZE bytes. f32/f16/i32/i8 etc
// (ELEMSIZE + raw byte copy, dtype-agnostic).
static const char * OVGPU_CONCAT_KERNEL = R"CLC(
__kernel void ovgpu_concat(
    __global const char * src0,
    __global const char * src1,
    __global       char * dst)
{
    const long idx = get_global_id(0);   // [0, NELS) flat OUTPUT index
    if (idx >= NELS) return;
    long r = idx;
    const long i0 = r % DNE0; r /= DNE0;
    const long i1 = r % DNE1; r /= DNE1;
    const long i2 = r % DNE2; r /= DNE2;
    const long i3 = r;
    const long doff = i0*DNB0 + i1*DNB1 + i2*DNB2 + i3*DNB3;
    __global const char * src;
    long off;
#if DIM == 0
    if (i0 < S0NE0) { src = src0; off = i0*S0NB0 + i1*S0NB1 + i2*S0NB2 + i3*S0NB3; }
    else            { src = src1; off = (i0 - S0NE0)*S1NB0 + i1*S1NB1 + i2*S1NB2 + i3*S1NB3; }
#elif DIM == 1
    if (i1 < S0NE1) { src = src0; off = i0*S0NB0 + i1*S0NB1 + i2*S0NB2 + i3*S0NB3; }
    else            { src = src1; off = i0*S1NB0 + (i1 - S0NE1)*S1NB1 + i2*S1NB2 + i3*S1NB3; }
#elif DIM == 2
    if (i2 < S0NE2) { src = src0; off = i0*S0NB0 + i1*S0NB1 + i2*S0NB2 + i3*S0NB3; }
    else            { src = src1; off = i0*S1NB0 + i1*S1NB1 + (i2 - S0NE2)*S1NB2 + i3*S1NB3; }
#else /* DIM == 3 */
    if (i3 < S0NE3) { src = src0; off = i0*S0NB0 + i1*S0NB1 + i2*S0NB2 + i3*S0NB3; }
    else            { src = src1; off = i0*S1NB0 + i1*S1NB1 + i2*S1NB2 + (i3 - S0NE3)*S1NB3; }
#endif
    for (int b = 0; b < ELEMSIZE; b++) {
        dst[doff + b] = src[off + b];
    }
}
)CLC";
// ALiBi (max_bias>0 -> per-head slope) and sinks (src[2], per-head max-prior).
// op_params[0]=scale, op_params[1]=max_bias. Two paths:
//  (1) plain (no mask, no sinks, max_bias==0, scale==1) -> cldnn softmax (fast,
//      handles huge ne0). reduce-ne0 layout (ne[0] on x).
//  (2) mask / ALiBi / sinks / scale!=1 -> custom OVGPU_SOFT_MAX_KERNEL (port of
//      ggml-cpu softmax): handles ggml's modular nr23 mask broadcast, per-head
//      ALiBi slope, and sinks natively. flat layout (ne[1..3]->batch). dst==src0.
static std::unique_ptr<compiled_op> build_soft_max(const ggml_tensor * node, cldnn::engine & engine, cldnn::stream::ptr stream) {
    const ggml_tensor * a     = node->src[0];
    const ggml_tensor * mask  = node->src[1];
    const ggml_tensor * sinks = node->src[2];
    if (!a) {
        return nullptr;
    }
    if (!type_supported(a->type)) {
        return nullptr;
    }
    if (!ggml_is_contiguous(a)) {
        return nullptr;
    }
    float scale, max_bias;
    memcpy(&scale,    node->op_params + 0, sizeof(float));
    memcpy(&max_bias, node->op_params + 1, sizeof(float));

    auto co   = std::make_unique<compiled_op>();
    co->output_id = "out";

    const bool use_custom = mask || sinks || max_bias != 0.0f || scale != 1.0f;
    if (!use_custom) {
        // (1) plain softmax(src0) over ne[0].
        const std::string a_id = "in_0";
        co->input_ids  = { a_id };
        co->layout_for = ggml_layout_for_reduce_ne0;
        cldnn::topology topo;
        topo.add(cldnn::input_layout(a_id, ggml_layout_for_reduce_ne0(a)));
        topo.add(cldnn::softmax("sm", cldnn::input_info(a_id), 3 /*dimension=x=ne0*/));
        topo.add(cldnn::reorder(co->output_id, cldnn::input_info("sm"), ggml_layout_for_reduce_ne0(node)));
        co->net = make_network(engine, stream, topo, make_config());
        return co;
    }

    // (2) custom kernel: mask / ALiBi / sinks / scale.
    if (mask) {
        if (mask->type != GGML_TYPE_F32 && mask->type != GGML_TYPE_F16) {
            return nullptr;
        }
        if (!ggml_is_contiguous(mask)) {
            return nullptr;
        }
    }
    if (sinks) {
        if (sinks->type != GGML_TYPE_F32 || !ggml_is_contiguous(sinks)) {
            return nullptr;
        }
    }

    const std::string a_id = "in_0";
    co->input_ids  = { a_id };
    co->layout_for = ggml_layout_for;  // flat (ne[1..3]->batch, ne[0]->feature)

    // cldnn input names: in_0=scores, in_1=mask (if present), in_2=sinks (if
    // present). cg_in_ids lists only the present ones, in kernel-param order
    // (scores, mask, sinks). The custom_gpu arg index must be each input's
    // POSITION in cg_in_ids (a running counter), NOT its name number - else a
    // sinks-only op (cg_in_ids={in_0,in_2}) would hand the wrong slot to sinks.
    std::vector<std::string> cg_in_ids = { a_id };
    if (mask) {
        co->input_ids.push_back("in_1");
        cg_in_ids.push_back("in_1");
    }
    if (sinks) {
        co->input_ids.push_back("in_2");
        cg_in_ids.push_back("in_2");
    }

    // ALiBi constants (host-computed, matching ggml-cpu):
    //   n_head = ne02; n_head_log2 = 2^floor(log2(n_head));
    //   m0 = 2^(-max_bias/n_head_log2);  m1 = 2^(-(max_bias/2)/n_head_log2).
    const uint32_t n_head      = (uint32_t) std::max<int64_t>(a->ne[2], 1);
    const uint32_t n_head_log2 = n_head > 0 ? (1u << (uint32_t) floor(log2((double) n_head))) : 1u;
    const float    m0 = powf(2.0f, -(max_bias)         / (float) n_head_log2);
    const float    m1 = powf(2.0f, -(max_bias / 2.0f) / (float) n_head_log2);

    const char * dfloat = (a->type == GGML_TYPE_F16) ? "half" : "float";
    std::string  build_opts =
        "-DDFLOAT=" + std::string(dfloat) +
        " -DNE0=" + std::to_string(a->ne[0]) + " -DNE1=" + std::to_string(a->ne[1]) +
        " -DNE2=" + std::to_string(a->ne[2]) + " -DNE3=" + std::to_string(a->ne[3]) +
        " -DSCALE=" + fstr(scale);
    if (mask) {
        const char * mfloat = (mask->type == GGML_TYPE_F16) ? "half" : "float";
        const int64_t ne12 = mask->ne[2];
        const int64_t ne13 = mask->ne[3];
        build_opts += " -DHAS_MASK -DMFLOAT=" + std::string(mfloat) +
                      " -DNE12=" + std::to_string(ne12) + " -DNE13=" + std::to_string(ne13);
    }
    if (sinks) {
        build_opts += " -DHAS_SINKS";
    }
    if (max_bias != 0.0f) {
        build_opts += " -DHAS_ALIBI -DN_HEAD_LOG2=" + std::to_string(n_head_log2) +
                      " -DM0=" + fstr(m0) + " -DM1=" + fstr(m1);
    }

    cldnn::topology topo;
    topo.add(cldnn::input_layout(a_id, ggml_layout_for(a)));
    if (mask) {
        topo.add(cldnn::input_layout("in_1", ggml_layout_for(mask)));
    }
    if (sinks) {
        topo.add(cldnn::input_layout("in_2", ggml_layout_for(sinks)));
    }
    std::vector<cldnn::input_info> cg_inputs;
    for (auto & id : cg_in_ids) {
        cg_inputs.emplace_back(id);
    }
    // arg index = each input's POSITION in cg_in_ids (a running counter), NOT a
    // fixed number: sinks-only has cg_in_ids={in_0,in_2} so sinks is at index 1,
    // while mask+sinks has it at index 2. A fixed index 2 would be out of range
    // for sinks-only (cg_inputs.size()==2) -> "input memory necessary" assert.
    std::vector<cldnn::custom_gpu_primitive::arg_desc> args;
    uint32_t arg_idx = 0;
    args.push_back({ cldnn::custom_gpu_primitive::arg_input, arg_idx, "" }); arg_idx++;  // scores
    if (mask)  { args.push_back({ cldnn::custom_gpu_primitive::arg_input, arg_idx, "" }); arg_idx++; }
    if (sinks) { args.push_back({ cldnn::custom_gpu_primitive::arg_input, arg_idx, "" }); arg_idx++; }
    args.push_back({ cldnn::custom_gpu_primitive::arg_output, 0, "" });

    const size_t n_rows = (size_t) (a->ne[1] * a->ne[2] * a->ne[3]);
    topo.add(cldnn::custom_gpu_primitive(
        co->output_id, cg_inputs, { std::string(OVGPU_SOFT_MAX_KERNEL) }, "ovgpu_soft_max",
        args, build_opts, { ggml_layout_for(node) }, { n_rows }, {}));

    cldnn::ExecutionConfig config;
    config.set_property(ov::intel_gpu::optimize_data(false));
    co->net = make_network(engine, stream, topo, config);
    return co;
}

// ---------------------------------------------------------------------------
// GET_ROWS -> gather (axis=0 row lookup; 2D table + 1D indices)
// ---------------------------------------------------------------------------

// ggml GET_ROWS: dst[:, i] = table[:, indices[i]] (each row is an ne0-length
// vector). dst is ALWAYS f32 (ggml_get_rows), indices are ALWAYS I32. Scope: 2D
// table (ne[2]==ne[3]==1; a grouped/batched table - ne[2]>1 - is deferred to CPU,
// matching MUL_MAT's GQA-weight deferral). ggml_get_rows' asserts (a->ne2==b->ne1,
// a->ne3==b->ne2) then force indices to be effectively 1D (b->ne1==b->ne2==1).
//
// cldnn gather(axis=0) on the existing flat mapping (ggml_layout_for: table ->
// tensor(n_vocab, n_embd,1,1), indices -> tensor(1, n_tokens,1,1) - i32 indices
// bind NATIVELY, no conversion) produces tensor(1, n_tokens, n_embd, 1). This is
// physically IDENTICAL bytes to ggml's target (token-major, embd-contiguous:
// addr = token*n_embd+embd in both formulas), just a different (b,f,x,y)
// labeling - confirmed against gather_gpu_int32.d14_axisB in the OV unit tests
// (dict{2,2,1,1}, idx{1,4,1,1}, output_shape{1,4,2,1} gathering rows 0,1,1,0).
// A cldnn::reshape (pure reinterpret, same dtype) relabels it to
// ggml_ne_to_tensor(node); a trailing reorder then converts dtype to f32 if the
// table wasn't already f32 (dst is always f32 per ggml_get_rows).
static std::unique_ptr<compiled_op> build_get_rows(const ggml_tensor * node, cldnn::engine & engine, cldnn::stream::ptr stream) {
    const ggml_tensor * table = node->src[0];
    const ggml_tensor * ids   = node->src[1];
    if (!table || !ids) {
        return nullptr;
    }
    if (!type_supported(table->type)) {
        return nullptr;
    }
    if (table->type == GGML_TYPE_BF16) {
        return nullptr;  // no bf16 gather kernel, see supports_op
    }
    if (ids->type != GGML_TYPE_I32) {
        return nullptr;
    }
    if (table->ne[2] != 1 || table->ne[3] != 1) {
        return nullptr;  // 2D table only
    }
    if (!ggml_is_contiguous(table) || !ggml_is_contiguous(ids)) {
        return nullptr;
    }

    auto              co   = std::make_unique<compiled_op>();
    const std::string t_id = "in_0";  // table   (src[0])
    const std::string i_id = "in_1";  // indices (src[1])
    co->input_ids          = { t_id, i_id };
    co->output_id          = "out";

    const int64_t n_embd   = table->ne[0];
    const int64_t n_tokens = ids->ne[0];
    auto          v        = [](int64_t n) {
        return (cldnn::tensor::value_type) std::max<int64_t>(n, 1);
    };

    cldnn::topology topo;
    topo.add(cldnn::input_layout(t_id, ggml_layout_for(table)));
    topo.add(cldnn::input_layout(i_id, ggml_layout_for(ids)));
    topo.add(cldnn::gather("gather", cldnn::input_info(t_id), cldnn::input_info(i_id), 0 /*axis*/, 4 /*input_rank*/,
                           ov::Shape{ 1, (size_t) n_tokens, (size_t) n_embd, 1 }, 0 /*batch_dim*/));
    // Relabel (1, n_tokens, n_embd, 1) -> ggml's (n_tokens, n_embd, 1, 1) - same
    // bytes (both token-major/embd-contiguous), so this is a pure reinterpret.
    topo.add(cldnn::reshape("reshaped", cldnn::input_info("gather"), cldnn::tensor(v(n_tokens), v(n_embd), 1, 1)));
    // Trailing reorder -> ggml dst layout (always f32; converts dtype only if the
    // table wasn't already f32 - shape is unchanged from the reshape above).
    topo.add(cldnn::reorder(co->output_id, cldnn::input_info("reshaped"), ggml_layout_for(node)));

    co->net = make_network(engine, stream, topo, make_config());
    return co;
}

// ---------------------------------------------------------------------------
// CPY -> reorder (copy + dtype cast, in-place into the dst view)
// ---------------------------------------------------------------------------

// ggml CPY: dst = copy of src[0] into src[1] (the node is a VIEW of src[1]=b).
//   src[0] = a (source), src[1] = b (destination; node = view of b).
//   a and b share ne[] (ggml asserts equal nelements; we require equal ne[] -
//   a reshape across axis decompositions is deferred). Types may differ (cast).
// cldnn::reorder(input, output_layout) copies + casts dtype/format, so a single
// reorder from a's layout to the node's (= b's) layout implements copy+cast. The
// output is bound in-place into node (= b's memory); b is NOT a separate input.
// Same shape + same bfyx format => the reorder is a pure dtype cast (or a no-op
// copy when types match). Scope: contiguous a and b (permuted src / strided dst
// views deferred to CPU).
static std::unique_ptr<compiled_op> build_cpy(const ggml_tensor * node, cldnn::engine & engine, cldnn::stream::ptr stream) {
    const ggml_tensor * a = node->src[0];
    const ggml_tensor * b = node->src[1];  // CPY: node is a view of b; CONT: b==NULL (node is a fresh contiguous tensor)
    if (!a) {
        return nullptr;
    }
    const bool a_contig = ggml_is_contiguous(a);
    if (!a_contig && !ggml_is_strided_view(a)) {
        return nullptr;
    }

    auto              co   = std::make_unique<compiled_op>();
    const std::string a_id = "in_0";
    co->input_ids          = { a_id };
    co->output_id          = "out";
    // flat for a contiguous a / the node; strided (physical nesting) for a permuted a.
    co->layout_for         = ggml_layout_for_maybe_strided;

    auto             v    = [](int64_t n) {
        return (cldnn::tensor::value_type) std::max<int64_t>(n, 1);
    };
    // dst dtype: CPY -> b's type (node is a view of b); CONT -> node's type (== a's type).
    ov::element::Type b_dt = ggml_type_to_cldnn(b ? b->type : node->type);

    cldnn::topology topo;
    if (a_contig) {
        // Contiguous a: a single reorder (copy + cast) into the node's layout.
        topo.add(cldnn::input_layout(a_id, ggml_layout_for(a)));
        topo.add(cldnn::reorder(co->output_id, cldnn::input_info(a_id), ggml_layout_for(node)));
    } else {
        // Permuted/strided a: bind in physical memory order, permute into the fixed
        // ne0->x/ne1->y/ne2->f/ne3->b convention, reorder (strip padding + cast to
        // b's dtype in one node), then reshape the dense eltwise result to the flat
        // layout the node is bound as. Dense eltwise and flat are the same ggml byte
        // order, so the reshape is a free reinterpret.
        topo.add(cldnn::input_layout(a_id, ggml_layout_for_strided(a)));
        const std::string perm_id    = a_id + "_perm";
        topo.add(cldnn::permute(perm_id, cldnn::input_info(a_id), ggml_strided_permute_order(a)));
        const std::string compact_id = a_id + "_compact";
        // eltwise used shape (b=ne3,f=ne2,x=ne0,y=ne1 ctor order), b's dtype, no pad.
        cldnn::tensor    eltwise_ten(v(a->ne[3]), v(a->ne[2]), v(a->ne[0]), v(a->ne[1]));
        cldnn::layout    eltwise_b{ b_dt, cldnn::format::bfyx, eltwise_ten };
        topo.add(cldnn::reorder(compact_id, cldnn::input_info(perm_id), eltwise_b));
        int64_t batch = node->ne[1] * node->ne[2] * node->ne[3];
        topo.add(cldnn::reshape(co->output_id, cldnn::input_info(compact_id),
                                cldnn::tensor(v(batch), v(node->ne[0]), 1, 1)));
    }

    co->net = make_network(engine, stream, topo, make_config());
    return co;
}

// ---------------------------------------------------------------------------
// SET_ROWS -> custom_gpu (in-place batched row scatter)
// ---------------------------------------------------------------------------

// ggml SET_ROWS: node = view of src[2]=a (the dst table [NE0, NE1, NE2, NE3]);
//   src[0]=b (updates [NE0, R, NE2, NE3]), src[1]=c (indices [R, NE11, NE12]).
//   For each (j, i2, i3): dst[ idx[j,i2%NE11,i3%NE12], i2, i03 ] = b[ j, i2, i03 ].
// cldnn scatter_update can't express the per-group (per ne2/ne3) indices (it applies
// uniform indices across non-axis dims), so use a custom OpenCL kernel via
// custom_gpu_primitive. The kernel handles batched + broadcast (i2%NE11) natively.
// All tensors bound flat-contiguous (ggml_layout_for); shape constants baked as -D.
// Output bound in-place into node (= a's memory); kernel writes only indexed rows.
static std::unique_ptr<compiled_op> build_set_rows(const ggml_tensor * node, cldnn::engine & engine, cldnn::stream::ptr stream) {
    const ggml_tensor * b = node->src[0];  // updates  (src[0])
    const ggml_tensor * c = node->src[1];  // indices  (src[1])
    if (!b || !c) {
        return nullptr;
    }

    auto              co   = std::make_unique<compiled_op>();
    const std::string b_id = "in_0";  // updates
    const std::string c_id = "in_1";  // indices
    co->input_ids          = { b_id, c_id };
    co->output_id          = "out";   // view of src[2]=a (the dst table)
    // layout_for must match the input_layout each operand is bound with: flat
    // (ggml_layout_for) for a contiguous operand / the always-contiguous output,
    // per-dim padded (ggml_layout_for_padded) for a padded-view operand - the
    // raw bind covers the full padded allocation the stride-aware kernel may
    // touch. graph_compute applies this to src[0], src[1] and the node (dst).
    co->layout_for         = [](const ggml_tensor * t) -> cldnn::layout {
        if (ggml_is_contiguous(t)) {
            return ggml_layout_for(t);
        }
        return ggml_layout_for_padded(t);
    };

    const int64_t ne0  = node->ne[0];
    const int64_t ne1  = node->ne[1];
    const int64_t ne2  = node->ne[2];
    const int64_t ne3  = node->ne[3];
    const int64_t r    = b->ne[1];     // rows to set
    const int64_t ne11 = c->ne[1];     // indices ne[1] (broadcast divisor of ne2)
    const int64_t ne12 = c->ne[2];     // indices ne[2] (broadcast divisor of ne3)

    const char * srcfloat = (b->type == GGML_TYPE_F16) ? "half" : "float";
    const char * dstfloat = (node->type == GGML_TYPE_F16) ? "half" : "float";  // node == dst table type
    const char * didx   = (c->type == GGML_TYPE_I64) ? "long" : "int";
    // Bake the source byte strides (nb[]) as -D: the kernel navigates a contiguous
    // OR padded/strided view's raw allocation directly (no pre-compaction). dst is
    // always contiguous (flat) so it needs no stride constants.
    std::string  build_opts =
        "-DSRCFLOAT=" + std::string(srcfloat) + " -DDSTFLOAT=" + std::string(dstfloat) +
        " -DDIDX=" + std::string(didx) +
        " -DR="    + std::to_string(r)    + " -DNE0="  + std::to_string(ne0) +
        " -DNE1="  + std::to_string(ne1)  + " -DNE2="  + std::to_string(ne2) +
        " -DNE3="  + std::to_string(ne3)  + " -DNE11=" + std::to_string(ne11) +
        " -DNE12=" + std::to_string(ne12) +
        " -DNB0="  + std::to_string(b->nb[0]) + " -DNB1=" + std::to_string(b->nb[1]) +
        " -DNB2="  + std::to_string(b->nb[2]) + " -DNB3=" + std::to_string(b->nb[3]) +
        " -DNB0I=" + std::to_string(c->nb[0]) + " -DNB1I=" + std::to_string(c->nb[1]) +
        " -DNB2I=" + std::to_string(c->nb[2]);

    cldnn::topology topo;
    // Bind each operand via the layout matching its memory: flat (ggml_layout_for)
    // for a contiguous tensor, per-dim padded (ggml_layout_for_padded) for a padded
    // view - the latter covers the full padded allocation the stride-aware kernel
    // may touch. No reorder/reshape: the kernel reads src0/src1 with their nb[]
    // strides directly. dst (node) is always contiguous -> flat.
    auto bind = [&](const std::string & id, const ggml_tensor * t) {
        if (ggml_is_contiguous(t)) {
            topo.add(cldnn::input_layout(id, ggml_layout_for(t)));
        } else {
            topo.add(cldnn::input_layout(id, ggml_layout_for_padded(t)));
        }
    };
    bind(b_id, b);
    bind(c_id, c);
    // kernel args: (src0=updates, src1=indices, dst=output) - memory pointers only.
    std::vector<cldnn::custom_gpu_primitive::arg_desc> args = {
        { cldnn::custom_gpu_primitive::arg_input,  0, "" },
        { cldnn::custom_gpu_primitive::arg_input,  1, "" },
        { cldnn::custom_gpu_primitive::arg_output, 0, "" },
    };
    const size_t total_rows = (size_t) (r * ne2 * ne3);
    topo.add(cldnn::custom_gpu_primitive(
        co->output_id,
        { cldnn::input_info(b_id), cldnn::input_info(c_id) },
        { std::string(OVGPU_SET_ROWS_KERNEL) },
        "ovgpu_set_rows",
        args,
        build_opts,
        { ggml_layout_for(node) },
        { (size_t) ne0, total_rows },  // gws: (NE0, R*NE2*NE3) - one work-item/element
        {}));                           // lws: let cldnn pick

    // optimize_data OFF: keep the declared flat bfyx layouts (the kernel indexes raw
    // flat memory; a layout optimizer inserting blocked formats / reorders would break
    // the indexing). use_onednn is irrelevant for a custom kernel.
    cldnn::ExecutionConfig config;
    config.set_property(ov::intel_gpu::optimize_data(false));
    co->net = make_network(engine, stream, topo, config);
    return co;
}

// ---------------------------------------------------------------------------
// ROPE -> custom_gpu (base case: NORMAL/NEOX, no YaRN)
// ---------------------------------------------------------------------------

// ggml ROPE / ROPE_BACK: rotates src[0] ([NE0, NE1, NE2, NE3]) by cos/sin derived
//   from src[1]=pos (I32, per-seq positions; 4 ids/token for mrope/vision) and
//   freq_base. src[2]=freq_factors (F32, optional). op_params: n_dims, mode,
//   n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast,
//   beta_slow, sections[4]. cldnn `rope` needs precomputed cos/sin tables
//   (positions are runtime -> can't be a static data() node), so use a custom
//   kernel that computes cos/sin inline, ported from ggml-cpu (the test
//   reference). Handles ALL modes (NORMAL/NEOX/MROPE/IMROPE/VISION), YaRN
//   (ext_factor!=0 and/or attn_factor!=1) and ROPE_BACK (sin_sign=-1).
//   Scope limit: contiguous src0 (strided/padded views deferred to CPU - the
//   kernel indexes flat raw memory; adding nb-stride indexing is a follow-up).
static std::unique_ptr<compiled_op> build_rope(const ggml_tensor * node, cldnn::engine & engine, cldnn::stream::ptr stream) {
    const ggml_tensor * a   = node->src[0];
    const ggml_tensor * pos = node->src[1];
    const ggml_tensor * ff  = node->src[2];
    if (!a || !pos) {
        return nullptr;
    }

    const int32_t * op_params = (const int32_t *) node->op_params;
    const int     n_dims      = op_params[1];
    const int     mode        = op_params[2];
    const int     n_ctx_orig  = op_params[4];
    const int *   sections    = op_params + 11;  // sections[4]
    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base,   op_params + 5,  sizeof(float));
    memcpy(&freq_scale,  op_params + 6,  sizeof(float));
    memcpy(&ext_factor,  op_params + 7,  sizeof(float));
    memcpy(&attn_factor, op_params + 8,  sizeof(float));
    memcpy(&beta_fast,   op_params + 9,  sizeof(float));
    memcpy(&beta_slow,   op_params + 10, sizeof(float));

    const bool is_normal = (mode == GGML_ROPE_TYPE_NORMAL);
    const bool is_neox   = (mode == GGML_ROPE_TYPE_NEOX);
    const bool is_mrope  = (mode == GGML_ROPE_TYPE_MROPE);
    const bool is_imrope = (mode == GGML_ROPE_TYPE_IMROPE);
    const bool is_vision = (mode == GGML_ROPE_TYPE_VISION);
    const bool has_ff    = (ff != nullptr);
    const bool is_yarn   = (ext_factor != 0.0f);
    const bool backward  = (node->op == GGML_OP_ROPE_BACK);

    // YaRN corr_dims + mscale, computed on the host (they're constants). Matches
    // ggml-cpu ggml_rope_yarn_corr_dims: with n_ctx_orig==0 (the common/test case)
    // the corr factor is -inf -> corr_low=0, corr_high clamps to -inf; the ramp
    // then evaluates identically to corr_low=corr_high=0 (max(0.001,...) clamps
    // the denominator), so we substitute 0 to avoid -inf in a -D constant.
    float corr_low = 0.0f, corr_high = 0.0f;
    float mscale   = attn_factor;  // ext_factor==0: pure attn_factor magnitude scale
    if (is_yarn) {
        if (n_ctx_orig > 0) {
            auto corr_factor = [&](float n_rot) {
                return (float) n_dims * logf((float) n_ctx_orig / (n_rot * 2.0f * (float) M_PI)) / (2.0f * logf(freq_base));
            };
            corr_low  = fmaxf(0.0f, floorf(corr_factor(beta_fast)));
            corr_high = fminf((float) n_dims - 1.0f, ceilf(corr_factor(beta_slow)));
        }
        mscale = attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));
    }

    auto              co   = std::make_unique<compiled_op>();
    const std::string a_id = "in_0";   // src0 (tensor)
    const std::string p_id = "in_1";   // pos
    std::string       f_id;
    co->input_ids = { a_id, p_id };
    if (has_ff) {
        f_id = "in_2";
        co->input_ids.push_back(f_id);
    }
    co->output_id  = "out";
    co->layout_for = ggml_layout_for;  // flat contiguous

    const char * dfloat = (a->type == GGML_TYPE_F16) ? "half" : "float";
    const float  theta_scale = powf(freq_base, -2.0f / (float) n_dims);
    // pair layout per mode (matches ggml-cpu rotate_pairs args):
    //   NORMAL: ic=i0/1,        n_offset=1
    //   NEOX/MROPE/IMROPE: ic=i0/2, n_offset=n_dims/2
    //   VISION: ic=i0/2,        n_offset=n_dims   (rotates ALL ne0; n_dims==ne0/2)
    const int ic_scale = is_normal ? 1 : 2;
    const int n_offset = is_normal ? 1 : (is_vision ? n_dims : n_dims / 2);
    const int rot_n    = is_vision ? (int) a->ne[0] : n_dims;  // vision rotates all ne0
    std::string  build_opts =
        "-DDFLOAT=" + std::string(dfloat) +
        " -DNE0="    + std::to_string(a->ne[0]) +
        " -DNE1="    + std::to_string(a->ne[1]) +
        " -DNE2="    + std::to_string(a->ne[2]) +
        " -DNE3="    + std::to_string(a->ne[3]) +
        " -DN_DIMS=" + std::to_string(n_dims) +
        " -DIC_SCALE=" + std::to_string(ic_scale) +
        " -DN_OFFSET=" + std::to_string(n_offset) +
        " -DROT_N="   + std::to_string(rot_n) +
        " -DFREQ_SCALE=" + fstr(freq_scale) +
        " -DTHETA_SCALE=" + fstr(theta_scale) +
        " -DMSCALE="  + fstr(mscale) +
        " -DSIN_SIGN=" + (backward ? fstr(-1.0f) : fstr(1.0f)) +
        " -DSEC0=" + std::to_string(sections[0]) +
        " -DSEC1=" + std::to_string(sections[1]) +
        " -DSEC2=" + std::to_string(sections[2]) +
        " -DSEC3=" + std::to_string(sections[3]) +
        (is_normal  ? " -DROPE_NORMAL"  : "") +
        (is_neox    ? " -DROPE_NEOX"    : "") +
        (is_mrope   ? " -DROPE_MROPE"   : "") +
        (is_imrope  ? " -DROPE_MROPE -DROPE_IMROPE" : "") +
        (is_vision  ? " -DROPE_VISION -DROPE_INDEP_SECTS" : "") +
        (is_yarn    ? " -DROPE_YARN -DEXT_FACTOR=" + fstr(ext_factor) +
                      " -DCORR_LOW=" + fstr(corr_low) +
                      " -DCORR_HIGH=" + fstr(corr_high) : "") +
        (has_ff     ? " -DHAS_FF" : "");

    cldnn::topology topo;
    topo.add(cldnn::input_layout(a_id, ggml_layout_for(a)));
    topo.add(cldnn::input_layout(p_id, ggml_layout_for(pos)));
    if (has_ff) {
        topo.add(cldnn::input_layout(f_id, ggml_layout_for(ff)));
    }
    std::vector<cldnn::input_info> cg_inputs = { cldnn::input_info(a_id), cldnn::input_info(p_id) };
    if (has_ff) {
        cg_inputs.push_back(cldnn::input_info(f_id));
    }
    std::vector<cldnn::custom_gpu_primitive::arg_desc> args = {
        { cldnn::custom_gpu_primitive::arg_input,  0, "" },
        { cldnn::custom_gpu_primitive::arg_input,  1, "" },
    };
    if (has_ff) {
        args.push_back({ cldnn::custom_gpu_primitive::arg_input, 2, "" });
    }
    args.push_back({ cldnn::custom_gpu_primitive::arg_output, 0, "" });

    const size_t n_rows = (size_t) (a->ne[1] * a->ne[2] * a->ne[3]);
    topo.add(cldnn::custom_gpu_primitive(
        co->output_id, cg_inputs, { std::string(OVGPU_ROPE_KERNEL) }, "ovgpu_rope",
        args, build_opts, { ggml_layout_for(node) }, { n_rows }, {}));

    cldnn::ExecutionConfig config;
    config.set_property(ov::intel_gpu::optimize_data(false));
    co->net = make_network(engine, stream, topo, config);
    return co;
}

// ---------------------------------------------------------------------------
// SILU / GLU / CONCAT -> custom_gpu
// ---------------------------------------------------------------------------

// Common helper: build a 1-or-2-input stride-aware custom kernel whose inputs may
// be contiguous or padded views, output is a new contiguous tensor. layout_for
// dispatches flat (contig) / per-dim padded (view) to match the input_layout bind.
static std::unique_ptr<compiled_op> build_custom_elementwise(
    const ggml_tensor * node, cldnn::engine & engine, cldnn::stream::ptr stream,
    const char * kernel_src, const char * entry_point,
    const std::string & build_opts, size_t n_rows_gws,
    std::vector<std::string> in_ids /* cldnn input names, in src order */) {
    auto co = std::make_unique<compiled_op>();
    co->input_ids = in_ids;
    co->output_id = "out";
    co->layout_for = [](const ggml_tensor * t) -> cldnn::layout {
        if (ggml_is_contiguous(t)) {
            return ggml_layout_for(t);
        }
        return ggml_layout_for_padded(t);
    };

    cldnn::topology topo;
    std::vector<cldnn::input_info> cg_inputs;
    for (size_t k = 0; k < in_ids.size(); k++) {
        const std::string & id = in_ids[k];
        // input name "in_<n>" -> binds node->src[n] (callers pass names matching
        // the src index: in_0=src[0], in_1=src[1], ...).
        const int src_idx = std::stoi(id.substr(3));
        topo.add(cldnn::input_layout(id, co->layout_for(node->src[src_idx])));
        cg_inputs.emplace_back(id);
    }
    std::vector<cldnn::custom_gpu_primitive::arg_desc> args;
    uint32_t arg_idx = 0;
    for (size_t k = 0; k < in_ids.size(); k++) {
        args.push_back({ cldnn::custom_gpu_primitive::arg_input, arg_idx, "" }); arg_idx++;
    }
    args.push_back({ cldnn::custom_gpu_primitive::arg_output, 0, "" });

    topo.add(cldnn::custom_gpu_primitive(
        co->output_id, cg_inputs, { std::string(kernel_src) }, entry_point,
        args, build_opts, { ggml_layout_for(node) }, { n_rows_gws }, {}));

    cldnn::ExecutionConfig config;
    config.set_property(ov::intel_gpu::optimize_data(false));
    co->net = make_network(engine, stream, topo, config);
    return co;
}

// ggml UNARY (SILU subtype only): out = silu(src0) = src0 / (1 + exp(-src0)).
// Other unary ops (GELU/RELU/...) -> CPU. dst = new contiguous.
static std::unique_ptr<compiled_op> build_unary(const ggml_tensor * node, cldnn::engine & engine, cldnn::stream::ptr stream) {
    const ggml_tensor * a = node->src[0];
    if (!a) return nullptr;
    if (a->type != GGML_TYPE_F32 && a->type != GGML_TYPE_F16) return nullptr;
    if (ggml_get_op_params_i32(node, 0) != GGML_UNARY_OP_SILU) return nullptr;  // only SILU

    const char * dfloat = (a->type == GGML_TYPE_F16) ? "half" : "float";
    std::string build_opts =
        "-DDFLOAT=" + std::string(dfloat) +
        " -DNE0=" + std::to_string(a->ne[0]) + " -DNE1=" + std::to_string(a->ne[1]) +
        " -DNE2=" + std::to_string(a->ne[2]) + " -DNE3=" + std::to_string(a->ne[3]) +
        " -DNELS=" + std::to_string(ggml_nelements(a)) +
        " -DSNB0=" + std::to_string(a->nb[0]) + " -DSNB1=" + std::to_string(a->nb[1]) +
        " -DSNB2=" + std::to_string(a->nb[2]) + " -DSNB3=" + std::to_string(a->nb[3]);

    return build_custom_elementwise(node, engine, stream, OVGPU_SILU_KERNEL, "ovgpu_silu",
                                    build_opts, (size_t) ggml_nelements(a), { "in_0" });
}

// ggml GLU: SwiGLu only (out = silu(a) * b). 2-input (glu_split) or 1-input
// (split halves, swapped flag). Other GLU ops -> CPU (returns nullptr).
static std::unique_ptr<compiled_op> build_glu(const ggml_tensor * node, cldnn::engine & engine, cldnn::stream::ptr stream) {
    const ggml_tensor * a = node->src[0];
    const ggml_tensor * b = node->src[1];
    if (!a) return nullptr;
    if (a->type != GGML_TYPE_F32 && a->type != GGML_TYPE_F16) return nullptr;
    const int32_t op = ggml_get_op_params_i32(node, 0);
    if (op != GGML_GLU_OP_SWIGLU) return nullptr;  // GEGLU/REGLU/... -> CPU

    const char * dfloat = (a->type == GGML_TYPE_F16) ? "half" : "float";
    const bool two_input = (b != nullptr);
    const int64_t nc = node->ne[0];          // output ne[0] (halved for 1-input)
    const int64_t nrows = node->ne[1] * node->ne[2] * node->ne[3];

    std::string build_opts =
        "-DDFLOAT=" + std::string(dfloat) +
        " -DNC=" + std::to_string(nc) +
        " -DNROWS=" + std::to_string(nrows) +
        " -DNE1=" + std::to_string(node->ne[1]) + " -DNE2=" + std::to_string(node->ne[2]) +
        " -DNE3=" + std::to_string(node->ne[3]) +
        " -DS0NB0=" + std::to_string(a->nb[0]) + " -DS0NB1=" + std::to_string(a->nb[1]) +
        " -DS0NB2=" + std::to_string(a->nb[2]) + " -DS0NB3=" + std::to_string(a->nb[3]);
    std::vector<std::string> in_ids = { "in_0" };
    if (two_input) {
        build_opts += " -DGLU_TWO_INPUT"
                      " -DS1NB0=" + std::to_string(b->nb[0]) + " -DS1NB1=" + std::to_string(b->nb[1]) +
                      " -DS1NB2=" + std::to_string(b->nb[2]) + " -DS1NB3=" + std::to_string(b->nb[3]);
        in_ids.push_back("in_1");
    } else {
        const int32_t swapped = ggml_get_op_params_i32(node, 1);
        build_opts += " -DSWAPPED=" + std::to_string(swapped);
    }

    return build_custom_elementwise(node, engine, stream, OVGPU_GLU_KERNEL, "ovgpu_glu",
                                    build_opts, (size_t) (nc * nrows), in_ids);
}

// ggml CONCAT: dst = [src0, src1] along dim (op_params[0]). Stride-aware; one
// work-item per output element (byte copy of ELEMSIZE). f32/f16/i32/i8.
static std::unique_ptr<compiled_op> build_concat(const ggml_tensor * node, cldnn::engine & engine, cldnn::stream::ptr stream) {
    const ggml_tensor * s0 = node->src[0];
    const ggml_tensor * s1 = node->src[1];
    if (!s0 || !s1) return nullptr;
    if (s0->type != s1->type) return nullptr;
    // any element type with a stable elemsize (q-block types with blck_size>1 not
    // supported by the element-wise copy - gate to CPU).
    const size_t es = ggml_type_size(s0->type);
    if (ggml_blck_size(s0->type) != 1) return nullptr;

    const int dim = ggml_get_op_params_i32(node, 0);
    const char * dfloat_unused = nullptr; GGML_UNUSED(dfloat_unused);
    std::string build_opts =
        " -DDIM=" + std::to_string(dim) +
        " -DELEMSIZE=" + std::to_string(es) +
        " -DNELS=" + std::to_string(ggml_nelements(node)) +
        " -DDNE0=" + std::to_string(node->ne[0]) + " -DDNE1=" + std::to_string(node->ne[1]) +
        " -DDNE2=" + std::to_string(node->ne[2]) + " -DDNE3=" + std::to_string(node->ne[3]) +
        " -DDNB0=" + std::to_string(node->nb[0]) + " -DDNB1=" + std::to_string(node->nb[1]) +
        " -DDNB2=" + std::to_string(node->nb[2]) + " -DDNB3=" + std::to_string(node->nb[3]) +
        " -DS0NE0=" + std::to_string(s0->ne[0]) + " -DS0NE1=" + std::to_string(s0->ne[1]) +
        " -DS0NE2=" + std::to_string(s0->ne[2]) + " -DS0NE3=" + std::to_string(s0->ne[3]) +
        " -DS0NB0=" + std::to_string(s0->nb[0]) + " -DS0NB1=" + std::to_string(s0->nb[1]) +
        " -DS0NB2=" + std::to_string(s0->nb[2]) + " -DS0NB3=" + std::to_string(s0->nb[3]) +
        " -DS1NB0=" + std::to_string(s1->nb[0]) + " -DS1NB1=" + std::to_string(s1->nb[1]) +
        " -DS1NB2=" + std::to_string(s1->nb[2]) + " -DS1NB3=" + std::to_string(s1->nb[3]);

    return build_custom_elementwise(node, engine, stream, OVGPU_CONCAT_KERNEL, "ovgpu_concat",
                                    build_opts, (size_t) ggml_nelements(node), { "in_0", "in_1" });
}

// ---------------------------------------------------------------------------
// FLASH_ATTN_EXT -> cldnn scaled_dot_product_attention (SDPA)
// ---------------------------------------------------------------------------

// ggml FLASH_ATTN_EXT: dst = attention(q, k, v, mask, scale, max_bias, logit_softcap).
//   q/k/v are permute(0,2,1,3) views: logical ne=[d, n_seq, n_head, batch] but physical
//   memory [d, n_head, n_seq, batch] = BSHD. SDPA reads via per-dim pitches (no contiguity
//   check) so strided/padded KV-cache views bind zero-copy. The permute(0,2,1,3) is handled
//   by SDPA's transpose_order {0,2,1,3} (physical BSHD -> logical BHSD). is_causal=false
//   (ggml passes an explicit additive mask). scale is a scalar field. Output needs
//   output_transpose_order={0,2,1,3} to land in ggml's contiguous BSHD output [d_v,H,S,B].
//   GQA (n_head != n_kv_head) works free (SDPA broadcasts). v_trans is always false in
//   the FA path. Gates to CPU: max_bias>0 (ALiBi), logit_softcap>0 (Gemma2/Qwen2.5),
//   sinks (src[4]) - SDPA supports sinks but needs extra input plumbing.
//   See memory ovgpu-flash-attn-sdpa-mapping for the verified contract.
static std::unique_ptr<compiled_op> build_flash_attn_ext(const ggml_tensor * node, cldnn::engine & engine,
                                                         cldnn::stream::ptr stream) {
    const ggml_tensor * q = node->src[0];
    const ggml_tensor * k = node->src[1];
    const ggml_tensor * v = node->src[2];
    const ggml_tensor * m = node->src[3];  // mask (optional)
    if (!q || !k || !v) {
        return nullptr;
    }

    float scale, max_bias, logit_softcap;
    memcpy(&scale,        node->op_params + 0, sizeof(float));
    memcpy(&max_bias,     node->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, node->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return nullptr;  // ALiBi / softcap -> CPU
    }
    if (node->src[4]) {
        return nullptr;  // sinks -> CPU (SDPA supports but needs reshape+scale-input plumbing)
    }

    // ggml FA in this version only has prec DEFAULT/F32 (both = f32 compute): it reads
    // f16 k/v, upcasts to f32, and accumulates QK^T / softmax / PV in f32. The cldnn SDPA
    // kernel requires MATCHING q/k/v dtypes (mixed q(F32)/k(F16) fails CL_BUILD), so cast
    // all three to f32 - matching ggml's f32 accumulation exactly. SDPA still accumulates
    // in f32 internally regardless of input dtype.
    ov::element::Type compute_dt = ov::element::f32;

    auto              co   = std::make_unique<compiled_op>();
    const std::string q_id = "in_0";
    const std::string k_id = "in_1";
    const std::string v_id = "in_2";
    co->input_ids  = { q_id, k_id, v_id };
    co->output_id  = "out";
    // External-memory bind layout per input: contiguous -> eltwise (physical nesting, no
    // pad); strided/permuted view -> ggml_layout_for_strided (physical nesting, padded).
    // Must match the input_layout each tensor is bound with below; ggml_sdpa_input_bind
    // derives its src_layout by the same stride sort, so the two agree.
    co->layout_for = [](const ggml_tensor * t) -> cldnn::layout {
        if (ggml_is_contiguous(t)) {
            return ggml_layout_for_eltwise(t);
        }
        return ggml_layout_for_strided(t);
    };
    if (m) {
        co->input_ids.push_back("in_3");
    }

    cldnn::topology topo;

    // Bind q/k/v in their PHYSICAL memory nesting (zero-copy) with a per-input SDPA
    // transpose order derived from the stride sort - no permute, no compaction copy.
    //  - contiguous / default-permute operand (physical BHSD) -> order {0,1,2,3}
    //  - permute(0,2,1,3) view (physical BSHD, the K/V-cache case) -> order {0,2,1,3}
    //    (= the OV unit test's own config, sdpa_gpu_test.cpp:94-95)
    // If native dtype != f32, one reorder casts to a compact f32 buffer in the same slot
    // order (reorder reads the padded source via pitches, writes compact). The SDPA kernel
    // reads q/k/v via per-dim pitches, so strided/padded KV-cache views bind zero-copy;
    // only head_size must be innermost+contiguous (always true for ggml FA). GQA
    // broadcast_axis = input_k_transpose_order[1] = the physical slot holding H_kv, which
    // is correct for either binding.
    // NOTE: the non-identity {0,2,1,3} SDPA input path crashes the IGC JIT for some head
    // sizes (e.g. hsk=80) - those are gated to unsupported in supports_op (fall back to
    // decomposed MUL_MAT attention).
    auto add_input = [&](const std::string & id, const sdpa_input_bind & b,
                         ov::element::Type native_dt) -> std::string {
        topo.add(cldnn::input_layout(id, b.src_layout));
        if (native_dt == compute_dt) {
            return id;  // zero-copy, already f32
        }
        const std::string cast_id = id + "_cast";
        topo.add(cldnn::reorder(cast_id, cldnn::input_info(id), b.compact_layout));
        return cast_id;
    };

    sdpa_input_bind q_b = ggml_sdpa_input_bind(q);
    sdpa_input_bind k_b = ggml_sdpa_input_bind(k);
    sdpa_input_bind v_b = ggml_sdpa_input_bind(v);

    const std::string q_bound = add_input(q_id, q_b, ggml_type_to_cldnn(q->type));
    const std::string k_bound = add_input(k_id, k_b, ggml_type_to_cldnn(k->type));
    const std::string v_bound = add_input(v_id, v_b, ggml_type_to_cldnn(v->type));

    // mask: contiguous [S_kv, S_q, 1, B] -> eltwise (b=B, f=1, y=S_q, x=S_kv) which
    // broadcasts to scores [B, H, S_q, S_kv] via the kernel's per-dim modulo mask index.
    // Additive (f16/f32), no transpose applied. Routed through a reorder into a fresh
    // cldnn-owned buffer - SDPA's mask handling can want padding room the raw ggml-buffer
    // subbuffer (sized to the exact tensor, no slack) doesn't have.
    std::string m_bound;
    if (m) {
        cldnn::layout m_lay = ggml_layout_for_eltwise(m);
        topo.add(cldnn::input_layout("in_3", m_lay));
        topo.add(cldnn::reorder("in_3_compact", cldnn::input_info("in_3"), m_lay));
        m_bound = "in_3_compact";
    }

    std::vector<cldnn::input_info> sdpa_inputs = {
        cldnn::input_info(q_bound), cldnn::input_info(k_bound), cldnn::input_info(v_bound)
    };
    if (m) {
        sdpa_inputs.push_back(cldnn::input_info(m_bound));
    }

    // SDPA: is_causal=false (ggml passes an explicit additive mask). Per-input transpose
    // orders map logical BHSD -> each operand's physical nesting. IDENTITY output order:
    // the sdpa_opt/sdpa_ref kernels ALWAYS write OUTPUT_GET_INDEX(b, head, seq, d) = canonical
    // BHSD coords straight into the physical output buffer - there is no output dims-order
    // remap (no OUTPUT_DIMS_ORDER anywhere in src/graph). So the output buffer MUST be
    // canonical BHSD; a non-identity output_transpose_order only changes the allocated shape
    // (shape inference honors it) while the kernel still writes BHSD coords -> transposed
    // garbage (H==S_q) or OOB writes (H!=S_q). Real OV flows keep output order identity
    // (TransposeSDPAFusion fuses only q/k/v input transposes) and emit a separate permute.
    auto sdpa_prim = cldnn::scaled_dot_product_attention(
        "sdpa",
        sdpa_inputs,
        false,                              // is_causal
        -1,                                 // indirect_axis
        q_b.order,                          // q transpose (physical -> logical BHSD)
        k_b.order,                          // k transpose
        v_b.order,                          // v transpose
        std::vector<int64_t>{0, 1, 2, 3}    // output transpose (IDENTITY - kernel writes BHSD)
    );
    sdpa_prim.scale_val = scale;
    topo.add(sdpa_prim);

    // SDPA output is canonical physical BHSD [B, H, S_q, D_v]. Permute to ggml's BSHD
    // [B, S_q, H, D_v] (= dst eltwise (b=B, f=S_q, y=H, x=D_v)), then compact into the f32
    // dst layout. ggml FA dst ne = {v->ne[0], q->ne[2], q->ne[1], q->ne[3]} (ggml.c:5432).
    topo.add(cldnn::permute("sdpa_perm", cldnn::input_info("sdpa"),
                            std::vector<uint16_t>{0, 2, 1, 3}));
    topo.add(cldnn::reorder(co->output_id, cldnn::input_info("sdpa_perm"),
                            ggml_layout_for_eltwise(node)));

    // SDPA needs allow_new_shape_infer(true) - the OV unit tests (sdpa_gpu_test.cpp)
    // always set it; the legacy calc_output_layout path doesn't thread the transpose orders
    // / mask broadcast through SDPA's shape logic correctly. (Opposite of MUL_MAT, where
    // the flag breaks the oneDNN FC rank-2 reshape - the two ops need opposite settings.)
    // Do NOT force an impl: SDPAOpt/SDPARef are both impl_types::ocl and the
    // force_implementations kernel-NAME is ignored for ocl_v2 impls (only impl_types +
    // format are honored; the name only reaches the legacy kernel_selector). Default
    // selection picks SDPAOpt everywhere; sdpa_micro is a stage inside SDPAOptImpl,
    // auto-gated by supports_micro_sdpa on immad devices.
    cldnn::ExecutionConfig config;
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    co->net = make_network(engine, stream, topo, config);
    return co;
}

// ---------------------------------------------------------------------------
// dispatch
// ---------------------------------------------------------------------------

std::unique_ptr<compiled_op> build_op(const ggml_tensor * node, cldnn::engine & engine, cldnn::stream::ptr stream) {
    switch (node->op) {
        case GGML_OP_MUL_MAT:
            return build_mul_mat(node, engine, stream);
        case GGML_OP_ADD:
        case GGML_OP_MUL:
            return build_eltwise(node, engine, stream);
        case GGML_OP_RMS_NORM:
            return build_rms_norm(node, engine, stream);
        case GGML_OP_SOFT_MAX:
            return build_soft_max(node, engine, stream);
        case GGML_OP_GET_ROWS:
            return build_get_rows(node, engine, stream);
        case GGML_OP_CPY:
        case GGML_OP_CONT:
            return build_cpy(node, engine, stream);
        case GGML_OP_SET_ROWS:
            return build_set_rows(node, engine, stream);
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            return build_rope(node, engine, stream);
        case GGML_OP_UNARY:
            return build_unary(node, engine, stream);
        case GGML_OP_GLU:
            return build_glu(node, engine, stream);
        case GGML_OP_CONCAT:
            return build_concat(node, engine, stream);
        case GGML_OP_FLASH_ATTN_EXT:
            return build_flash_attn_ext(node, engine, stream);
        default:
            return nullptr;
    }
}

// ---------------------------------------------------------------------------
// Fusion: multi-op cldnn topologies for fusible chains
// ---------------------------------------------------------------------------
//
// ovgpu is otherwise Tier A (one single-primitive cldnn network per ggml op), so
// cldnn's program-level fuser (prepare_primitive_fusing.cpp, gated on
// optimize_data(true) which make_config sets) has nothing to merge. build_chain
// instead builds ONE topology for a linear chain [head, child1, ..., childN] and
// lets program::fuse_nodes merge the children into the head at build time.
//
// Fusible on Lunar Lake (supports_immad=1), verified against the fuser predicates:
//   head = MUL_MAT (FC)  -> children ADD/MUL fuse as oneDNN binary/sum post-ops
//                          (fc_supports_fusings true on the oneDNN path)
//   head = RMS_NORM (rms)-> children ADD/MUL fuse as the rms ELTWISE fused op
//                          (ungated; rms_kernel_bfyx_opt declares ELTWISE)
// The single overriding constraint (enforced by the detector in graph_compute, not
// here): the head's output must have exactly ONE user (the chain). build_chain only
// validates shape/dtype/contiguity eligibility and returns nullptr if not fusible
// (caller falls back to per-op).

// One external (ggml-side) input recorded while building a chain: the cldnn
// input_layout id, the layout graph_compute must bind it with, and which
// (chain-node-index, src-index) it corresponds to.
struct chain_ext_input {
    std::string   id;
    cldnn::layout layout;
    int           node_idx;  // index into the chain's nodes vector
    int           src_idx;   // nodes[node_idx]->src[src_idx]
};

// Append an eltwise (ADD/MUL) CHILD consuming `upstream_id` (the previous compute-
// out) plus one external operand, to `topo`. The external operand is bound in the
// HEAD's output-layout convention (`ext_layout_fn`) and cast to `out_dt` so cldnn's
// fuser shape/dtype rule is satisfied (the eltwise adopts the head's output layout,
// NOT its own native eltwise layout). Records the external input in `ext`. Returns
// the eltwise compute-output id, or "" if the child has no external operand.
static std::string eltwise_child_core(cldnn::topology & topo, const ggml_tensor * node,
                                      const std::string & prefix, const std::string & upstream_id,
                                      const ggml_tensor * upstream_tensor, ov::element::Type out_dt,
                                      std::function<cldnn::layout(const ggml_tensor *)> ext_layout_fn,
                                      int node_idx, chain_ext_input & ext) {
    int ext_idx = -1;
    for (int k = 0; k < GGML_MAX_SRC; k++) {
        if (!node->src[k]) {
            continue;
        }
        if (node->src[k] == upstream_tensor) {
            continue;  // chain-flowing input
        }
        ext_idx = k;
        break;
    }
    if (ext_idx < 0) {
        return "";  // ADD/MUL always has an external operand
    }
    const ggml_tensor * e   = node->src[ext_idx];
    cldnn::layout       lay = ext_layout_fn(e);
    const std::string   in_id = prefix + "in";
    std::string         bound;
    if (ggml_type_to_cldnn(e->type) == out_dt) {
        topo.add(cldnn::input_layout(in_id, lay));
        bound = in_id;
    } else {
        // cast external operand to the head's compute-output dtype
        topo.add(cldnn::input_layout(in_id, lay));
        const std::string cast_id = prefix + "cast";
        cldnn::layout     cast_lay{ out_dt, cldnn::format::bfyx, lay.get_tensor() };
        topo.add(cldnn::reorder(cast_id, cldnn::input_info(in_id), cast_lay));
        bound = cast_id;
    }
    cldnn::eltwise_mode mode = (node->op == GGML_OP_ADD) ? cldnn::eltwise_mode::sum : cldnn::eltwise_mode::prod;
    const std::string   ew_id = prefix + "ew";
    topo.add(cldnn::eltwise(ew_id, cldnn::input_info(upstream_id), cldnn::input_info(bound), mode));
    ext = { in_id, lay, node_idx, ext_idx };
    return ew_id;
}

// Append an activation(SILU = swish) CHILD consuming `upstream_id` (the previous
// compute-out) to `topo`. SILU is unary (one src = the upstream), so it records NO
// external input. SILU = x*sigmoid(x) = cldnn activation_func::swish (alpha=1); oneDNN
// FC accepts it as an eltwise_swish post-op (convert_activation_func swish->eltwise_swish).
// Returns the activation compute-output id.
static std::string activation_silu_core(cldnn::topology & topo, const std::string & prefix,
                                        const std::string & upstream_id) {
    const std::string act_id = prefix + "act";
    topo.add(cldnn::activation(act_id, cldnn::input_info(upstream_id), cldnn::activation_func::swish));
    return act_id;
}

// Structural signature of a chain (head + each child's full key + which src of each
// child is the chain-flowing input). Distinct chains -> distinct keys so the
// chain_cache never shares a network between structurally different chains.
static std::string chain_key_for(const std::vector<const ggml_tensor *> & nodes) {
    std::ostringstream s;
    s << "CHAIN|n=" << nodes.size();
    s << "|h=" << op_cache::key_for(nodes[0]);
    const ggml_tensor * prev = nodes[0];
    for (size_t i = 1; i < nodes.size(); i++) {
        const ggml_tensor * c = nodes[i];
        int up_idx = -1;
        for (int k = 0; k < GGML_MAX_SRC; k++) {
            if (c->src[k] == prev) {
                up_idx = k;
                break;
            }
        }
        s << "|c" << i << "=" << op_cache::key_for(c) << "|up" << i << "=" << up_idx;
        prev = c;
    }
    return s.str();
}

std::unique_ptr<compiled_op> build_chain(const std::vector<const ggml_tensor *> & nodes,
                                         cldnn::engine &                          engine,
                                         cldnn::stream::ptr                       stream) {
    if (nodes.size() < 2) {
        return nullptr;
    }
    const ggml_tensor * head = nodes[0];
    const bool          head_is_fc  = (head->op == GGML_OP_MUL_MAT);
    const bool          head_is_rms = (head->op == GGML_OP_RMS_NORM);
    if (!head_is_fc && !head_is_rms) {
        return nullptr;
    }

    // Head eligibility (mirrors the standalone build_* / supports_op guards) + the
    // per-head layout conventions. head_layout_for binds the head's OWN operands
    // (matches the standalone layout_for); head_out_layout_fn is the head's OUTPUT
    // layout, in which child external operands are bound so the fuser's shape rule
    // holds. head_out_dt = head dst dtype (FC->f32, rms preserves src dtype).
    std::function<cldnn::layout(const ggml_tensor *)> head_layout_for;
    std::function<cldnn::layout(const ggml_tensor *)> head_out_layout_fn;
    auto flat_batch = [](const ggml_tensor * t) { return t->ne[1] * t->ne[2] * t->ne[3]; };

    if (head_is_fc) {
        const ggml_tensor * w = head->src[0];
        const ggml_tensor * a = head->src[1];
        if (!w || !a) {
            return nullptr;
        }
        if (!type_supported(w->type) || !type_supported(a->type)) {
            return nullptr;
        }
        if (ggml_n_dims(w) != 2) {
            return nullptr;
        }
        // Phase 1: only fuse FC chains with CONTIGUOUS operands. With strided/
        // flat-padded FC inputs the FC output stays in cldnn's preferred (possibly
        // blocked) format with no trailing reorder before the eltwise, and the
        // oneDNN binary post-op can misalign the plain peer -> wrong output. Real
        // weights/activations at residual-ADD points are contiguous, so this gates
        // out only artificial cases (e.g. test-backend-ops k_v!=0 diamonds); they
        // fall back to per-op (correct, just unfused).
        if (!ggml_is_contiguous(w) || !ggml_is_contiguous(a)) {
            return nullptr;
        }
        if (w->ne[0] != a->ne[0]) {
            return nullptr;
        }
        head_layout_for = [](const ggml_tensor * t) -> cldnn::layout {
            if (ggml_is_contiguous(t)) {
                return ggml_layout_for(t);
            }
            if (ggml_is_flat_padded_view(t)) {
                return ggml_layout_for_maybe_padded(t);
            }
            return ggml_layout_for_strided(t);
        };
        head_out_layout_fn = ggml_layout_for;  // flat bfyx (FC output layout)
    } else {  // rms
        const ggml_tensor * a = head->src[0];
        if (!a || !type_supported(a->type)) {
            return nullptr;
        }
        if (!ggml_is_contiguous(a) && !ggml_is_padded_view(a)) {
            return nullptr;
        }
        head_layout_for     = ggml_layout_for_reduce_ne0_maybe_padded;
        head_out_layout_fn  = ggml_layout_for_reduce_ne0;
    }
    const ov::element::Type head_out_dt = ggml_type_to_cldnn(head->type);

    cldnn::topology topo;
    std::vector<std::string>                input_ids;
    std::vector<cldnn::layout>              input_layouts;
    std::vector<std::pair<int, int>>        input_src_map;

    // Head core.
    std::string compute_out;
    if (head_is_fc) {
        compute_out = mul_mat_core(topo, head, engine, "h_");
        // head external inputs: src[0] (weight), src[1] (activation)
        input_ids.push_back("h_in_0");
        input_layouts.push_back(head_layout_for(head->src[0]));
        input_src_map.push_back({ 0, 0 });
        input_ids.push_back("h_in_1");
        input_layouts.push_back(head_layout_for(head->src[1]));
        input_src_map.push_back({ 0, 1 });
    } else {
        compute_out = rms_core(topo, head, "h_");
        input_ids.push_back("h_in_0");
        input_layouts.push_back(head_layout_for(head->src[0]));
        input_src_map.push_back({ 0, 0 });
    }

    // Children: ADD/MUL (eltwise) or UNARY-SILU (activation). SILU is unary (no
    // external operand); ADD/MUL have an external operand (residual/gamma/gate).
    const ggml_tensor * upstream = head;
    for (size_t i = 1; i < nodes.size(); i++) {
        const ggml_tensor * child = nodes[i];
        const bool          is_eltwise = (child->op == GGML_OP_ADD || child->op == GGML_OP_MUL);
        const bool          is_silu =
            (child->op == GGML_OP_UNARY && ggml_get_op_params_i32(child, 0) == GGML_UNARY_OP_SILU);
        if (!is_eltwise && !is_silu) {
            return nullptr;
        }
        // child must consume the previous node's output as one src.
        int up_idx = -1, ext_idx = -1;
        for (int k = 0; k < GGML_MAX_SRC; k++) {
            if (!child->src[k]) {
                continue;
            }
            if (child->src[k] == upstream) {
                up_idx = k;
            } else {
                ext_idx = k;
            }
        }
        if (up_idx < 0) {
            return nullptr;
        }

        const std::string prefix = "c" + std::to_string(i - 1) + "_";
        if (is_silu) {
            compute_out = activation_silu_core(topo, prefix, compute_out);
        } else {
            // ADD/MUL: the external operand must be contiguous and broadcast cleanly
            // in the head's output layout (flat-batch==1 or==head; ne[0]==1 or==head).
            const ggml_tensor * e = child->src[ext_idx];
            if (!e || !type_supported(e->type) || !ggml_is_contiguous(e)) {
                return nullptr;
            }
            const int64_t hb = flat_batch(head), eb = flat_batch(e);
            const bool   ok_b = (eb == 1 || eb == hb);
            const bool   ok_f = (e->ne[0] == 1 || e->ne[0] == head->ne[0]);
            if (!ok_b || !ok_f) {
                return nullptr;
            }
            chain_ext_input ext;
            std::string     ew = eltwise_child_core(topo, child, prefix, compute_out, upstream,
                                                    head_out_dt, head_out_layout_fn, (int) i, ext);
            if (ew.empty()) {
                return nullptr;
            }
            compute_out = ew;
            input_ids.push_back(ext.id);
            input_layouts.push_back(ext.layout);
            input_src_map.push_back({ ext.node_idx, ext.src_idx });
        }
        upstream = child;
    }

    // One trailing reorder -> the chain's final-node dst layout (forces bfyx so the
    // ggml tensor bytes line up; cldnn's layout optimizer could otherwise pick a
    // transposed format for some shapes).
    auto                    co = std::make_unique<compiled_op>();
    const ggml_tensor *     final_node = nodes.back();
    co->output_id  = "chain_out";
    cldnn::layout  out_lay = head_is_fc ? ggml_layout_for(final_node) : ggml_layout_for_reduce_ne0(final_node);
    topo.add(cldnn::reorder(co->output_id, cldnn::input_info(compute_out), out_lay));

    co->net          = make_network(engine, stream, topo, make_config());
    co->input_ids    = std::move(input_ids);
    co->input_layouts = std::move(input_layouts);
    co->input_src_map = std::move(input_src_map);
    co->is_chain     = true;
    co->chain_len    = (int) nodes.size();
    co->layout_for   = head_is_fc ? ggml_layout_for : ggml_layout_for_reduce_ne0;

    // GGML_OVGPU_FUSE_TRACE: confirm the children actually fused into the head (vs
    // running as separate kernels). Walks the compiled program's processing order
    // and prints each FC/rms node's fused-primitive list.
    if (getenv("GGML_OVGPU_FUSE_TRACE")) {
        fprintf(stderr, "[ovgpu][fuse] chain len=%d head=%s\n", (int) nodes.size(), ggml_op_name(head->op));
        auto prog = co->net->get_program();
        for (cldnn::program_node * pn : prog->get_processing_order()) {
            if (pn->is_type<cldnn::fully_connected>() || pn->is_type<cldnn::rms>()) {
                fprintf(stderr, "[ovgpu][fuse]   %-12s fused=%d", pn->id().c_str(), (int) pn->has_fused_primitives());
                for (auto & fp : pn->get_fused_primitives()) {
                    fprintf(stderr, " [%s]", fp.desc->id.c_str());
                }
                fprintf(stderr, "\n");
            }
        }
    }
    return co;
}

std::unique_ptr<compiled_op> op_cache::get_or_build_chain(const std::vector<const ggml_tensor *> & nodes) {
    std::string       key = chain_key_for(nodes);
    std::lock_guard<std::mutex> lock(m_mutex);
    static const bool no_cache = getenv("GGML_OVGPU_NO_CACHE") != nullptr;
    auto              it = no_cache ? chain_cache.end() : chain_cache.find(key);
    if (it != chain_cache.end()) {
        return std::make_unique<compiled_op>(*it->second);
    }
    if (!no_cache && chain_fail_cache.count(key)) {
        return nullptr;  // previously failed to build - don't retry every token
    }
    std::unique_ptr<compiled_op> co;
    try {
        co = build_chain(nodes, *engine, stream);
    } catch (const std::exception & e) {
        fprintf(stderr, "OVGPU: build_chain threw: %s\nkey=%s\n", e.what(), key.c_str());
        if (!no_cache) {
            chain_fail_cache.insert(key);
        }
        return nullptr;
    }
    if (!co) {
        // build_chain rejected this chain (shape/dtype/contiguity not fusible) - cache
        // the rejection so the detector's loose check doesn't re-form it every call.
        if (!no_cache) {
            chain_fail_cache.insert(key);
        }
        return nullptr;
    }
    if (!no_cache) {
        chain_cache[key] = std::make_shared<compiled_op>(*co);
    }
    return co;
}

}  // namespace ggml::ovgpu
