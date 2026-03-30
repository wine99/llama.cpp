// ggml-inpu-translate.cpp — translate ggml cgraph to OpenVINO IR for iNPU
//
// Handles MUL_MAT (with Q4_0/Q8_0/F16 weights, F32/F16 activations)
// plus optional fused ADD, GLU, RMS_NORM, ROPE, and RESHAPE ops.

#include "ggml-inpu-translate.h"
#include "ggml-inpu-impl.h"
#include "ggml-impl.h"

#include <openvino/op/add.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/clamp.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/cos.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/maximum.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/power.hpp>
#include <openvino/op/reduce_mean.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/sin.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/split.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/sqrt.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <openvino/op/gelu.hpp>
#include <openvino/op/swish.hpp>
#include <openvino/op/relu.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#define INPU_LOG_DEBUG(...) GGML_LOG_DEBUG("INPU-translate: " __VA_ARGS__)
#define INPU_LOG_ERROR(...) GGML_LOG_ERROR("INPU-translate: " __VA_ARGS__)

// ============================================================================
// Graph I/O analysis
// ============================================================================

inpu_graph_io inpu_analyze_graph_io(const struct ggml_cgraph * cgraph) {
    inpu_graph_io io;

    auto is_passthrough_view_like = [](enum ggml_op op) {
        switch (op) {
            case GGML_OP_VIEW:
            case GGML_OP_RESHAPE:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                return true;
            default:
                return false;
        }
    };

    // Build set of all node dst pointers (for fast lookup of intermediates)
    std::unordered_set<const struct ggml_tensor *> node_outputs;
    std::unordered_map<const struct ggml_tensor *, int> node_indices;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        node_outputs.insert(cgraph->nodes[i]);
        node_indices[cgraph->nodes[i]] = i;
    }

    // Count how many times each tensor is consumed by nodes within this subgraph.
    // We compare this against the full-graph use_counts (inherited from the
    // parent cgraph via ggml_graph_view) to determine which nodes have consumers
    // outside the subgraph and thus need to be OV outputs.
    std::unordered_map<const struct ggml_tensor *, int> local_compute_uses;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (cgraph->nodes[i]->op == GGML_OP_NONE) {
            continue;
        }
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (cgraph->nodes[i]->src[s]) {
                local_compute_uses[cgraph->nodes[i]->src[s]]++;
            }
        }
    }

    // Identify outputs using the full-graph use_counts.
    // A compute node is an output of this subgraph if:
    //   - total_uses == 0 : terminal node, nothing in the full graph uses it
    //   - total_uses > local_compute_uses : some consumer lives outside this
    //     subgraph (or is a VIEW that aliases into the buffer)
    // Trailing VIEW/RESHAPE/PERMUTE/TRANSPOSE chains are folded away so only
    // the backing compute tensor becomes an OV output.
    std::unordered_set<const struct ggml_tensor *> seen_outputs;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if (node->op == GGML_OP_NONE) {
            continue;
        }

        const int total_uses = ggml_node_get_use_count(cgraph, i);
        const int local_uses = local_compute_uses.count(node) ? local_compute_uses[node] : 0;

        if (total_uses == 0 || total_uses > local_uses) {
            const struct ggml_tensor * out_root = node;
            while (is_passthrough_view_like(out_root->op) && out_root->src[0] && node_outputs.count(out_root->src[0])) {
                out_root = out_root->src[0];
            }

            if (is_passthrough_view_like(out_root->op) && (!out_root->src[0] || !node_outputs.count(out_root->src[0]))) {
                continue;
            }

            if (seen_outputs.count(out_root)) {
                continue;
            }
            seen_outputs.insert(out_root);

            inpu_graph_io::output_info out_info;
            out_info.tensor   = const_cast<struct ggml_tensor *>(out_root);
            out_info.node_idx = node_indices.at(out_root);
            io.outputs.push_back(out_info);
        }
    }

    // Mark nodes that contribute to exported outputs.
    std::unordered_set<const struct ggml_tensor *> live_nodes;
    std::vector<const struct ggml_tensor *> stack;
    for (const auto & out : io.outputs) {
        stack.push_back(out.tensor);
    }
    while (!stack.empty()) {
        const struct ggml_tensor * node = stack.back();
        stack.pop_back();

        if (!node_outputs.count(node) || live_nodes.count(node)) {
            continue;
        }
        live_nodes.insert(node);

        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (node->src[s] && node_outputs.count(node->src[s])) {
                stack.push_back(node->src[s]);
            }
        }
    }

    // Track which inputs we've already added (by tensor pointer)
    std::unordered_set<const struct ggml_tensor *> seen_inputs;

    // Identify inputs: src tensors that are NOT produced by any live node in this graph
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if (!live_nodes.count(node) || node->op == GGML_OP_NONE) {
            continue;
        }

        for (int s = 0; s < GGML_MAX_SRC; s++) {
            const struct ggml_tensor * src = node->src[s];
            if (!src) {
                continue;
            }

            if (node_outputs.count(src)) {
                continue;
            }

            if (seen_inputs.count(src)) {
                continue;
            }

            seen_inputs.insert(src);

            inpu_graph_io::input_info info;
            info.tensor    = src;
            info.node_idx  = i;
            info.src_idx   = s;
            info.is_weight = ggml_inpu_tensor_in_inpu_buffer(src);
            io.inputs.push_back(info);
        }
    }

    return io;
}

// ============================================================================
// Helper: make a unique parameter name
// ============================================================================

static std::string make_param_name(const char * prefix, int node_idx, int src_idx, const char * suffix = "") {
    std::ostringstream ss;
    ss << prefix << "_n" << node_idx << "_s" << src_idx << suffix;
    return ss.str();
}

// ============================================================================
// Helper: ggml type → OV element type
// ============================================================================

static ov::element::Type ggml_type_to_ov(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: return ov::element::f32;
        case GGML_TYPE_F16: return ov::element::f16;
        case GGML_TYPE_I32: return ov::element::i32;
        default:
            GGML_ABORT("unsupported ggml type for OV conversion: %d", type);
    }
}

// ============================================================================
// Helper: create an OV shape from ggml tensor
// For ggml, dimensions are [ne[0], ne[1], ne[2], ne[3]] where ne[0] is innermost.
// OV uses row-major, so we need to reverse: [ne[3], ne[2], ne[1], ne[0]].
// But for MUL_MAT, ggml convention:
//   src0 (weight): [K, N, batch...]   where K=ne[0], N=ne[1]
//   src1 (activation): [K, M, batch...] where K=ne[0], M=ne[1]
//   dst: [N, M, batch...]
// ============================================================================

static ov::Shape ggml_tensor_to_ov_shape(const struct ggml_tensor * t) {
    // Find the effective number of dimensions (trim trailing 1s from the end)
    int ndims = GGML_MAX_DIMS;
    while (ndims > 2 && t->ne[ndims - 1] == 1) {
        ndims--;
    }

    // Build shape in OV order: reverse of ggml order
    ov::Shape shape(ndims);
    for (int i = 0; i < ndims; i++) {
        shape[i] = t->ne[ndims - 1 - i];
    }
    return shape;
}

// ============================================================================
// Helper: translate a quantized weight to dequantized OV subgraph
//
// Creates two Parameters (quants + scales) and returns a dequantized f32 output.
// During weight loading, u4/i8 quants are already converted to signed (i4/i8)
// and permuted to groups-first layout:
//   quants: [n_groups, N, group_size]  (i4 or i8)
//   scales: [n_groups, N, 1]           (f16)
//
// NPU-optimal dequant pattern:
//   [n_groups, N, gs] (Param, i4/i8) → Convert → f16 → Multiply → Transpose(1,0,2) → [N, n_groups, gs] → Reshape → [N, K]
//   [n_groups, N, 1]  (Param, f16)   -------------------------->
//
// The formula is: Convert(i4/i8 → f16) * scale, transpose, reshape, then convert to f32.
// ============================================================================

struct dequant_result {
    ov::Output<ov::Node> output;        // dequantized weight in f32
    std::shared_ptr<ov::op::v0::Parameter> quant_param;
    std::shared_ptr<ov::op::v0::Parameter> scale_param;
};

static dequant_result create_dequant_subgraph(
    const struct ggml_tensor * weight_tensor,
    const std::string & quant_name,
    const std::string & scale_name)
{
    auto * extra = static_cast<ggml_inpu_tensor_extra *>(weight_tensor->extra);
    GGML_ASSERT(extra && extra->is_quantized);

    dequant_result result;

    const size_t N          = (size_t) extra->n_rows;
    const size_t K          = (size_t) extra->n_cols;
    const size_t group_size = (size_t) extra->group_size;
    const size_t n_groups   = K / group_size;

    GGML_ASSERT(K % group_size == 0);

    // Quant parameter: groups-first 3D layout [n_groups, N, group_size]
    ov::Shape quant_shape = {n_groups, N, group_size};
    result.quant_param = std::make_shared<ov::op::v0::Parameter>(extra->quant_ov_type, quant_shape);
    result.quant_param->set_friendly_name(quant_name);
    result.quant_param->output(0).set_names({quant_name});

    // Scale parameter: groups-first 3D layout [n_groups, N, 1] — no Unsqueeze needed
    ov::Shape scale_shape = {n_groups, N, 1};
    result.scale_param = std::make_shared<ov::op::v0::Parameter>(extra->scale_ov_type, scale_shape);
    result.scale_param->set_friendly_name(scale_name);
    result.scale_param->output(0).set_names({scale_name});

    // Convert quants to f16: [n_groups, N, gs]
    auto quant_f16 = std::make_shared<ov::op::v0::Convert>(result.quant_param, ov::element::f16);

    // Multiply: [n_groups, N, gs] * [n_groups, N, 1] → [n_groups, N, gs] (broadcast on last dim)
    auto dequant = std::make_shared<ov::op::v1::Multiply>(quant_f16, result.scale_param);

    // Transpose: [n_groups, N, gs] → [N, n_groups, gs] via perm {1, 0, 2}
    auto transpose_order = ov::op::v0::Constant::create(
        ov::element::i64, {3}, std::vector<int64_t>{1, 0, 2});
    auto dequant_transposed = std::make_shared<ov::op::v1::Transpose>(dequant, transpose_order);

    // Reshape: [N, n_groups, gs] → [N, K]
    auto flat_shape = ov::op::v0::Constant::create(
        ov::element::i64, {2}, std::vector<int64_t>{(int64_t) N, (int64_t) K});
    auto dequant_flat = std::make_shared<ov::op::v1::Reshape>(dequant_transposed, flat_shape, false);

    // Convert to f32 for computation
    auto dequant_f32 = std::make_shared<ov::op::v0::Convert>(dequant_flat, ov::element::f32);

    result.output = dequant_f32;
    return result;
}

// ============================================================================
// Translate MUL_MAT
// ============================================================================

static ov::Output<ov::Node> translate_mul_mat(
    const struct ggml_tensor * node,
    const std::unordered_map<const struct ggml_tensor *, ov::Output<ov::Node>> & tensor_map)
{
    const struct ggml_tensor * src0 = node->src[0]; // weight: [K, N, batch...]
    const struct ggml_tensor * src1 = node->src[1]; // activation: [K, M, batch...]
    // dst: [N, M, batch...]

    auto it_B = tensor_map.find(src0);
    auto it_A = tensor_map.find(src1);
    GGML_ASSERT(it_B != tensor_map.end() && it_A != tensor_map.end());

    ov::Output<ov::Node> B = it_B->second; // weight
    ov::Output<ov::Node> A = it_A->second; // activation

    // Ensure both are f16 for NPU (or at least same type)
    // Convert activation to match weight type if needed
    if (A.get_element_type() != B.get_element_type()) {
        // Convert B to A's type (A is the activation, typically f32)
        // Actually, we want to compute in f16 for NPU efficiency.
        // But per D2, we accept f32 and let NPU convert.
        // So convert B to A's type.
        B = std::make_shared<ov::op::v0::Convert>(B, A.get_element_type());
    }

    // ggml MUL_MAT: dst = src1 @ src0^T
    // In OV (row-major) terms:
    //   src0 (weight):     ggml [K, N] → OV [N, K]
    //   src1 (activation): ggml [K, M] → OV [M, K]
    //   dst:               ggml [N, M] → OV [M, N]
    //
    // dst_ov = src1_ov @ src0_ov^T = [M, K] @ [K, N] = [M, N]
    // MatMul(A, B, false, true) where A=activation, B=weight

    // Handle batch dimensions (GQA broadcasting)
    auto A_shape = A.get_shape();
    auto B_shape = B.get_shape();

    // Ensure both are at least 2D (they should be)
    GGML_ASSERT(A_shape.size() >= 2 && B_shape.size() >= 2);

    // Handle GQA batch broadcasting
    // For 4D tensors: ggml [K, N, n_heads, batch] → OV [batch, n_heads, N, K]
    // GQA: n_heads_q > n_heads_kv, so we need to broadcast
    if (A_shape.size() > 2 && B_shape.size() > 2) {
        // A_shape in OV: [..., M, K], B_shape: [..., N, K]
        // Check if batch dims need broadcasting
        size_t A_batch = A_shape[A_shape.size() - 3];
        size_t B_batch = B_shape[B_shape.size() - 3];
        int64_t factor = (int64_t)(std::max(A_batch, B_batch) / std::min(A_batch, B_batch));

        if (factor > 1 && std::min(A_batch, B_batch) > 1) {
            // Need to broadcast the smaller batch dimension
            bool A_larger = A_batch > B_batch;
            ov::Output<ov::Node> & Z = A_larger ? B : A;
            auto Z_shape = Z.get_shape();

            auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {(int64_t)(Z_shape.size() - 2)});
            auto Z_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(Z, unsqueeze_axes);

            // Build broadcast shape: insert factor dimension
            std::vector<int64_t> bcast_shape(Z_shape.size() + 1, 1);
            for (size_t i = 0; i < Z_shape.size(); i++) {
                if (i < Z_shape.size() - 2) {
                    bcast_shape[i] = (int64_t)Z_shape[i];
                } else if (i == Z_shape.size() - 2) {
                    bcast_shape[i] = 1; // original batch dim
                    bcast_shape[i+1] = factor; // broadcast factor
                } else {
                    bcast_shape[i+1] = (int64_t)Z_shape[i];
                }
            }
            auto bcast_shape_const = ov::op::v0::Constant::create(ov::element::i64, {bcast_shape.size()}, bcast_shape);
            auto Z_broadcasted = std::make_shared<ov::op::v3::Broadcast>(Z_unsqueezed, bcast_shape_const, ov::op::BroadcastType::BIDIRECTIONAL);

            // Reshape back to original rank with expanded batch
            std::vector<int64_t> new_shape;
            for (size_t i = 0; i < Z_shape.size(); i++) {
                if (i == Z_shape.size() - 3) {
                    new_shape.push_back((int64_t)(std::min(A_batch, B_batch) * factor));
                } else {
                    new_shape.push_back((int64_t)Z_shape[i]);
                }
            }
            auto new_shape_const = ov::op::v0::Constant::create(ov::element::i64, {new_shape.size()}, new_shape);
            Z = std::make_shared<ov::op::v1::Reshape>(Z_broadcasted, new_shape_const, true);
        }
    }

    auto result = std::make_shared<ov::op::v0::MatMul>(A, B, false, true);

    // Convert output type if needed
    ov::element::Type dst_type = ggml_type_to_ov(node->type);
    ov::Output<ov::Node> output = result;
    if (result->get_output_element_type(0) != dst_type) {
        output = std::make_shared<ov::op::v0::Convert>(result, dst_type);
    }

    return output;
}

// ============================================================================
// Translate ADD
// ============================================================================

static ov::Output<ov::Node> translate_add(
    const struct ggml_tensor * node,
    const std::unordered_map<const struct ggml_tensor *, ov::Output<ov::Node>> & tensor_map)
{
    auto it0 = tensor_map.find(node->src[0]);
    auto it1 = tensor_map.find(node->src[1]);
    GGML_ASSERT(it0 != tensor_map.end() && it1 != tensor_map.end());

    ov::element::Type dst_type = ggml_type_to_ov(node->type);
    ov::Output<ov::Node> lhs = it0->second;
    ov::Output<ov::Node> rhs = it1->second;

    if (lhs.get_element_type() != dst_type) {
        lhs = std::make_shared<ov::op::v0::Convert>(lhs, dst_type);
    }
    if (rhs.get_element_type() != dst_type) {
        rhs = std::make_shared<ov::op::v0::Convert>(rhs, dst_type);
    }

    return std::make_shared<ov::op::v1::Add>(lhs, rhs);
}

// ============================================================================
// Translate MUL (element-wise multiply)
// ============================================================================

static ov::Output<ov::Node> translate_mul(
    const struct ggml_tensor * node,
    const std::unordered_map<const struct ggml_tensor *, ov::Output<ov::Node>> & tensor_map)
{
    auto it0 = tensor_map.find(node->src[0]);
    auto it1 = tensor_map.find(node->src[1]);
    GGML_ASSERT(it0 != tensor_map.end() && it1 != tensor_map.end());

    ov::element::Type dst_type = ggml_type_to_ov(node->type);
    ov::Output<ov::Node> lhs = it0->second;
    ov::Output<ov::Node> rhs = it1->second;

    if (lhs.get_element_type() != dst_type) {
        lhs = std::make_shared<ov::op::v0::Convert>(lhs, dst_type);
    }
    if (rhs.get_element_type() != dst_type) {
        rhs = std::make_shared<ov::op::v0::Convert>(rhs, dst_type);
    }

    return std::make_shared<ov::op::v1::Multiply>(lhs, rhs);
}

// ============================================================================
// Translate GLU (SWIGLU, GEGLU, REGLU, etc.)
//
// GLU splits the input along the last axis into two halves (gate, value),
// applies an activation to the gate, then multiplies gate × value.
// ============================================================================

static ov::Output<ov::Node> translate_glu(
    const struct ggml_tensor * node,
    const std::unordered_map<const struct ggml_tensor *, ov::Output<ov::Node>> & tensor_map)
{
    enum ggml_glu_op glu_op = ggml_get_glu_op(node);
    const bool swapped = ggml_get_op_params_i32(node, 1) != 0;

    // GLU has one or two inputs
    // If src[1] != nullptr, gate = src[0], value = src[1] (no split needed)
    // If src[1] == nullptr, split src[0] into two halves
    auto it0 = tensor_map.find(node->src[0]);
    GGML_ASSERT(it0 != tensor_map.end());

    ov::Output<ov::Node> gate_input;
    ov::Output<ov::Node> up_input;

    if (node->src[1]) {
        auto it1 = tensor_map.find(node->src[1]);
        GGML_ASSERT(it1 != tensor_map.end());
        gate_input = it0->second;
        up_input = it1->second;
    } else {
        // Split along last axis
        auto input = it0->second;
        auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
        auto split = std::make_shared<ov::op::v1::Split>(input, axis, 2);
        gate_input = split->output(swapped ? 1 : 0);
        up_input = split->output(swapped ? 0 : 1);
    }

    // Apply activation to the gate
    ov::Output<ov::Node> gate_activated;
    switch (glu_op) {
        case GGML_GLU_OP_SWIGLU:
        case GGML_GLU_OP_SWIGLU_OAI:
            gate_activated = std::make_shared<ov::op::v4::Swish>(gate_input);
            break;
        case GGML_GLU_OP_GEGLU:
        case GGML_GLU_OP_GEGLU_ERF:
            gate_activated = std::make_shared<ov::op::v7::Gelu>(gate_input, ov::op::GeluApproximationMode::ERF);
            break;
        case GGML_GLU_OP_GEGLU_QUICK:
            gate_activated = std::make_shared<ov::op::v7::Gelu>(gate_input, ov::op::GeluApproximationMode::TANH);
            break;
        case GGML_GLU_OP_REGLU:
            gate_activated = std::make_shared<ov::op::v0::Relu>(gate_input);
            break;
        default:
            GGML_ABORT("unsupported GLU op: %d", glu_op);
    }

    // gate × value
    auto result = std::make_shared<ov::op::v1::Multiply>(gate_activated, up_input);

    ov::element::Type dst_type = ggml_type_to_ov(node->type);
    ov::Output<ov::Node> output = result;
    if (result->get_output_element_type(0) != dst_type) {
        output = std::make_shared<ov::op::v0::Convert>(result, dst_type);
    }
    return output;
}

// ============================================================================
// Translate RMS_NORM
//
// RMS_NORM(x) = x / sqrt(mean(x²) + eps)
// The epsilon is stored in the first 4 bytes of op_params.
// ============================================================================

static ov::Output<ov::Node> translate_rms_norm(
    const struct ggml_tensor * node,
    const std::unordered_map<const struct ggml_tensor *, ov::Output<ov::Node>> & tensor_map)
{
    auto it0 = tensor_map.find(node->src[0]);
    GGML_ASSERT(it0 != tensor_map.end());

    ov::Output<ov::Node> input = it0->second;

    // Ensure f32 for the computation
    if (input.get_element_type() != ov::element::f32) {
        input = std::make_shared<ov::op::v0::Convert>(input, ov::element::f32);
    }

    // Extract epsilon from op_params
    float eps;
    memcpy(&eps, node->op_params, sizeof(float));

    // x² → ReduceMean(axis=-1, keepdims=true) → +eps → sqrt → 1/rms → x * reciprocal
    auto square = std::make_shared<ov::op::v1::Power>(
        input, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {2.0f}));

    auto mean = std::make_shared<ov::op::v1::ReduceMean>(
        square, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), true);

    auto rms = std::make_shared<ov::op::v0::Sqrt>(
        std::make_shared<ov::op::v1::Add>(
            mean, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {eps})));

    auto reciprocal = std::make_shared<ov::op::v1::Divide>(
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1.0f}), rms);

    auto result = std::make_shared<ov::op::v1::Multiply>(input, reciprocal);

    ov::element::Type dst_type = ggml_type_to_ov(node->type);
    ov::Output<ov::Node> output = result;
    if (result->get_output_element_type(0) != dst_type) {
        output = std::make_shared<ov::op::v0::Convert>(result, dst_type);
    }
    return output;
}

// ============================================================================
// Translate ROPE
//
// Rotary Position Embedding. Supports ROPE_TYPE_NORMAL (0) and NEOX (2).
//
// ggml ROPE tensor layout (4D):
//   src0 (data):      ne = [head_dim, n_heads, seq_len, batch]
//   src1 (positions):  ne = [seq_len] (i32)
//   src2 (freq_factors): ne = [head_dim/2] (f32, optional)
//   dst:              same shape as src0
//
// OV shape (reversed): [batch, seq_len, n_heads, head_dim]
//
// op_params layout (int32_t[15]):
//   [0] = n_past (unused)
//   [1] = n_dims
//   [2] = mode
//   [3] = n_ctx (unused)
//   [4] = n_ctx_orig
//   [5] = freq_base (float)
//   [6] = freq_scale (float)
//   [7] = ext_factor (float)
//   [8] = attn_factor (float)
//   [9] = beta_fast (float)
//   [10] = beta_slow (float)
// ============================================================================

// YaRN correction dim computation (same as in ggml.c)
static float inpu_rope_yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif
    return n_dims * logf(n_ctx_orig / (n_rot * 2 * (float) M_PI)) / (2 * logf(base));
}

static void inpu_rope_yarn_corr_dims(
    int n_dims, int n_ctx_orig, float freq_base,
    float beta_fast, float beta_slow, float dims[2])
{
    float start = floorf(inpu_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    float end   = ceilf(inpu_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = std::max(0.0f, start);
    dims[1] = std::min(static_cast<float>(n_dims - 1), end);
}

// Build sin/cos OV constants from rope parameters.
// inp_pos is the OV parameter for positions (i32), will be converted/reshaped.
// Returns (sin_theta, cos_theta) as OV outputs shaped [seq_len, 1, n_dims/2]
// for broadcasting with 3D data in OV layout [seq_len, n_heads, head_dim].
// (ne[3] is always 1 for ROPE, so ggml_tensor_to_ov_shape produces 3D.)
static std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> inpu_make_sin_cos(
    const int32_t * op_params,
    std::shared_ptr<ov::Node> inp_pos,
    std::shared_ptr<ov::Node> rope_freqs_weight)
{
    // Convert positions from i32 to f32 and reshape to [seq_len, 1, 1]
    auto pos_f32 = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
    auto pos_shape = ov::op::v0::Constant::create(
        ov::element::i64, ov::Shape{3}, std::vector<int64_t>{-1, 1, 1});
    auto pos_reshaped = std::make_shared<ov::op::v1::Reshape>(pos_f32, pos_shape, false);

    // Extract parameters
    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    const int n_dims     = op_params[1];
    const int n_ctx_orig = op_params[4];
    memcpy(&freq_base,    op_params + 5,  sizeof(float));
    memcpy(&freq_scale,   op_params + 6,  sizeof(float));
    memcpy(&ext_factor,   op_params + 7,  sizeof(float));
    memcpy(&attn_factor,  op_params + 8,  sizeof(float));
    memcpy(&beta_fast,    op_params + 9,  sizeof(float));
    memcpy(&beta_slow,    op_params + 10, sizeof(float));

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    float corr_dims[2];
    inpu_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    // Build frequency factor vector: theta_scale^0, theta_scale^1, ...
    // Shape: [1, 1, n_dims/2]
    const int half_n_dims = n_dims / 2;
    std::vector<float> factor(half_n_dims);
    factor[0] = 1.0f;
    for (int i = 1; i < half_n_dims; i++) {
        factor[i] = theta_scale * factor[i - 1];
    }

    auto freq_factors_const = ov::op::v0::Constant::create(
        ov::element::f32, ov::Shape{1, 1, (size_t) half_n_dims}, factor);

    ov::Output<ov::Node> freq_node = freq_factors_const;

    // If rope_freqs_weight (src[2]) is provided, divide by it
    if (rope_freqs_weight) {
        // freq_factors from src[2], shape [n_dims/2] → reshape to [1, 1, n_dims/2]
        auto ff_shape = ov::op::v0::Constant::create(
            ov::element::i64, ov::Shape{3}, std::vector<int64_t>{1, 1, -1});
        auto ff_reshaped = std::make_shared<ov::op::v1::Reshape>(rope_freqs_weight, ff_shape, false);
        freq_node = std::make_shared<ov::op::v1::Divide>(freq_node, ff_reshaped);
    }

    // theta_extrap = freq_factors * pos  → [seq_len, 1, n_dims/2]
    auto theta_extrap = std::make_shared<ov::op::v1::Multiply>(freq_node, pos_reshaped);

    // theta_interp = theta_extrap * freq_scale
    auto theta_interp = std::make_shared<ov::op::v1::Multiply>(
        theta_extrap, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {freq_scale}));

    ov::Output<ov::Node> theta;
    float mscale = attn_factor;

    if (ext_factor == 0.0f) {
        theta = theta_interp;
    } else {
        // YaRN ramp mixing
        std::vector<float> dim_ids_vec(half_n_dims);
        std::iota(dim_ids_vec.begin(), dim_ids_vec.end(), 0.0f);
        auto dim_ids = ov::op::v0::Constant::create(
            ov::element::f32, ov::Shape{1, 1, (size_t) half_n_dims}, dim_ids_vec);
        auto corr_low  = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1}, {corr_dims[0]});
        auto corr_high = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1}, {corr_dims[1]});
        auto denom = std::make_shared<ov::op::v1::Maximum>(
            std::make_shared<ov::op::v1::Subtract>(corr_high, corr_low),
            ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1}, {0.001f}));
        auto ramp_y = std::make_shared<ov::op::v1::Divide>(
            std::make_shared<ov::op::v1::Subtract>(dim_ids, corr_low), denom);
        auto ramp_clamped = std::make_shared<ov::op::v0::Clamp>(ramp_y, 0.0f, 1.0f);
        // CPU rope_yarn_ramp returns (1 - clamp(y)), so invert before scaling
        auto one_c = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1}, {1.0f});
        auto ramp_inverted = std::make_shared<ov::op::v1::Subtract>(one_c, ramp_clamped);
        auto ramp_mix = std::make_shared<ov::op::v1::Multiply>(
            ramp_inverted, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {ext_factor}));

        auto one = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1}, {1.0f});
        auto one_minus_ramp = std::make_shared<ov::op::v1::Subtract>(one, ramp_mix);

        theta = std::make_shared<ov::op::v1::Add>(
            std::make_shared<ov::op::v1::Multiply>(theta_interp, one_minus_ramp),
            std::make_shared<ov::op::v1::Multiply>(theta_extrap, ramp_mix));

        mscale *= (1.0f + 0.1f * std::log(1.0f / freq_scale));
    }

    // sin/cos → [seq_len, 1, n_dims/2]
    ov::Output<ov::Node> cos_theta = std::make_shared<ov::op::v0::Cos>(theta);
    ov::Output<ov::Node> sin_theta = std::make_shared<ov::op::v0::Sin>(theta);

    auto mscale_node = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {mscale});
    cos_theta = std::make_shared<ov::op::v1::Multiply>(cos_theta, mscale_node);
    sin_theta = std::make_shared<ov::op::v1::Multiply>(sin_theta, mscale_node);

    return {sin_theta, cos_theta};
}

static ov::Output<ov::Node> translate_rope(
    const struct ggml_tensor * node,
    const std::unordered_map<const struct ggml_tensor *, ov::Output<ov::Node>> & tensor_map)
{
    // src[0] = data, src[1] = positions (i32), src[2] = freq_factors (optional)
    auto it_data = tensor_map.find(node->src[0]);
    auto it_pos  = tensor_map.find(node->src[1]);
    GGML_ASSERT(it_data != tensor_map.end() && it_pos != tensor_map.end());

    ov::Output<ov::Node> data_node = it_data->second;
    auto inp_pos = it_pos->second.get_node_shared_ptr();

    std::shared_ptr<ov::Node> rope_freqs_weight;
    if (node->src[2]) {
        auto it_ff = tensor_map.find(node->src[2]);
        GGML_ASSERT(it_ff != tensor_map.end());
        rope_freqs_weight = it_ff->second.get_node_shared_ptr();
    }

    // Ensure data is f32 for computation
    if (data_node.get_element_type() != ov::element::f32) {
        data_node = std::make_shared<ov::op::v0::Convert>(data_node, ov::element::f32);
    }

    const int32_t * op_params = (const int32_t *) node->op_params;
    const int n_dims = op_params[1];
    const int mode   = op_params[2];

    // Build sin/cos from op_params and position input
    // sin/cos are 3D: [seq_len, 1, n_dims/2] — broadcasts with data [seq_len, n_heads, head_dim]
    auto [sin_theta, cos_theta] = inpu_make_sin_cos(op_params, inp_pos, rope_freqs_weight);

    // ggml shape: [ne0, ne1, ne2, ne3] = [head_dim, n_heads, seq_len, 1]
    // OV shape (reversed, trailing-1-trimmed): [seq_len, n_heads, head_dim] (3D)

    const int64_t head_dim = node->ne[0];

    ov::Output<ov::Node> res;

    if (mode == GGML_ROPE_TYPE_NORMAL) {
        // NORMAL mode: pairs at (i, i+1) are rotated together
        // Slice even and odd indices along last dimension
        auto neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        auto zero    = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto one     = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto two     = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});
        auto end     = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_dims});

        // even indices: 0, 2, 4, ... (step=2 along last dim)
        auto even_slice = std::make_shared<ov::op::v8::Slice>(data_node, zero, end, two, neg_one);
        // odd indices: 1, 3, 5, ...
        auto odd_slice  = std::make_shared<ov::op::v8::Slice>(data_node, one, end, two, neg_one);

        // rotated_even = even * cos - odd * sin
        auto rotated_even = std::make_shared<ov::op::v1::Subtract>(
            std::make_shared<ov::op::v1::Multiply>(even_slice, cos_theta),
            std::make_shared<ov::op::v1::Multiply>(odd_slice, sin_theta));

        // rotated_odd = even * sin + odd * cos
        auto rotated_odd = std::make_shared<ov::op::v1::Add>(
            std::make_shared<ov::op::v1::Multiply>(even_slice, sin_theta),
            std::make_shared<ov::op::v1::Multiply>(odd_slice, cos_theta));

        // Interleave back: unsqueeze on dim 3, concat, reshape to 3D
        auto unsqueeze_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
        auto even_unsq = std::make_shared<ov::op::v0::Unsqueeze>(rotated_even, unsqueeze_dim);
        auto odd_unsq  = std::make_shared<ov::op::v0::Unsqueeze>(rotated_odd, unsqueeze_dim);
        auto stacked = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{even_unsq, odd_unsq}, 3);

        // Reshape back to 3D: [seq_len, n_heads, n_dims]
        auto target_shape = ov::op::v0::Constant::create(
            ov::element::i64, {3}, std::vector<int64_t>{
                (int64_t) node->ne[2], (int64_t) node->ne[1], (int64_t) n_dims});
        auto rotated = std::make_shared<ov::op::v1::Reshape>(stacked, target_shape, false);

        if (n_dims < head_dim) {
            // Concat the unrotated tail
            auto tail_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_dims});
            auto tail_end   = ov::op::v0::Constant::create(ov::element::i64, {1}, {head_dim});
            auto tail_step  = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            auto unrotated_tail = std::make_shared<ov::op::v8::Slice>(data_node, tail_start, tail_end, tail_step, neg_one);
            res = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{rotated, unrotated_tail}, -1);
        } else {
            res = rotated;
        }
    } else if (mode == GGML_ROPE_TYPE_NEOX) {
        // NEOX mode: split last dim in half, each half rotated
        auto split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});

        // Only split n_dims portion if n_dims < head_dim
        ov::Output<ov::Node> data_to_rotate = data_node;
        ov::Output<ov::Node> unrotated_tail;
        bool has_tail = (n_dims < head_dim);

        if (has_tail) {
            auto neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
            auto zero    = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            auto one_s   = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            auto ndims_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_dims});
            auto hd_c    = ov::op::v0::Constant::create(ov::element::i64, {1}, {head_dim});
            data_to_rotate = std::make_shared<ov::op::v8::Slice>(data_node, zero, ndims_c, one_s, neg_one);
            unrotated_tail = std::make_shared<ov::op::v8::Slice>(data_node, ndims_c, hd_c, one_s, neg_one);
        }

        auto data_split = std::make_shared<ov::op::v1::Split>(data_to_rotate, split_axis, 2);
        auto first_half  = data_split->output(0);  // [..., n_dims/2]
        auto second_half = data_split->output(1);   // [..., n_dims/2]

        // rotated_first = first * cos - second * sin
        auto rotated_first = std::make_shared<ov::op::v1::Subtract>(
            std::make_shared<ov::op::v1::Multiply>(first_half, cos_theta),
            std::make_shared<ov::op::v1::Multiply>(second_half, sin_theta));

        // rotated_second = first * sin + second * cos
        auto rotated_second = std::make_shared<ov::op::v1::Add>(
            std::make_shared<ov::op::v1::Multiply>(first_half, sin_theta),
            std::make_shared<ov::op::v1::Multiply>(second_half, cos_theta));

        auto rotated = std::make_shared<ov::op::v0::Concat>(
            ov::OutputVector{rotated_first, rotated_second}, -1);

        if (has_tail) {
            res = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{rotated, unrotated_tail}, -1);
        } else {
            res = rotated;
        }
    } else {
        GGML_ABORT("unsupported ROPE mode: %d", mode);
    }

    // Convert output type if needed
    ov::element::Type dst_type = ggml_type_to_ov(node->type);
    if (res.get_element_type() != dst_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, dst_type);
    }

    return res;
}

// ============================================================================
// Translate RESHAPE
//
// Applies an explicit OV Reshape to change the tensor shape.
// ============================================================================

static ov::Output<ov::Node> translate_reshape(
    const struct ggml_tensor * node,
    const std::unordered_map<const struct ggml_tensor *, ov::Output<ov::Node>> & tensor_map)
{
    auto it0 = tensor_map.find(node->src[0]);
    GGML_ASSERT(it0 != tensor_map.end());

    ov::Output<ov::Node> input = it0->second;

    // Target shape from node's ne[], in OV order (reversed)
    ov::Shape target_ov_shape = ggml_tensor_to_ov_shape(node);
    std::vector<int64_t> shape_vec(target_ov_shape.size());
    for (size_t i = 0; i < target_ov_shape.size(); i++) {
        shape_vec[i] = (int64_t) target_ov_shape[i];
    }
    auto shape_const = ov::op::v0::Constant::create(
        ov::element::i64, {shape_vec.size()}, shape_vec);

    return std::make_shared<ov::op::v1::Reshape>(input, shape_const, false);
}

// ============================================================================
// Translate PERMUTE
//
// ggml PERMUTE reorders dimensions: result->ne[axis_i] = src->ne[i].
// The axes (axis0..axis3) are stored in op_params as int32[4].
//
// ggml dims are reversed relative to OV dims.  If ggml permutes with
// axes [a0, a1, a2, a3], the equivalent OV Transpose order is:
//   ov_order[ndims-1 - a_i] = ndims-1 - i   (for i = 0..ndims-1)
// ============================================================================

static ov::Output<ov::Node> translate_permute(
    const struct ggml_tensor * node,
    const std::unordered_map<const struct ggml_tensor *, ov::Output<ov::Node>> & tensor_map)
{
    auto it0 = tensor_map.find(node->src[0]);
    GGML_ASSERT(it0 != tensor_map.end());

    ov::Output<ov::Node> input = it0->second;
    const auto & ov_shape = input.get_shape();
    const int ndims = (int) ov_shape.size();

    // Read ggml permute axes from op_params
    int32_t ggml_axes[4];
    memcpy(ggml_axes, node->op_params, sizeof(ggml_axes));

    // Convert ggml axes → OV transpose order
    // ggml dim i maps to OV dim (ndims-1-i) (assuming only ndims active dims)
    std::vector<int64_t> ov_order(ndims);
    for (int i = 0; i < ndims; i++) {
        int ggml_src_dim = ggml_axes[i]; // ggml source dim that goes to position i
        int ov_dst = ndims - 1 - i;
        int ov_src = ndims - 1 - ggml_src_dim;
        ov_order[ov_dst] = ov_src;
    }

    auto order_const = ov::op::v0::Constant::create(
        ov::element::i64, {(size_t) ndims}, ov_order);

    return std::make_shared<ov::op::v1::Transpose>(input, order_const);
}

// ============================================================================
// Translate TRANSPOSE
//
// TRANSPOSE is ggml_permute(a, 1, 0, 2, 3) — swaps dims 0 and 1.
// ============================================================================

static ov::Output<ov::Node> translate_transpose(
    const struct ggml_tensor * node,
    const std::unordered_map<const struct ggml_tensor *, ov::Output<ov::Node>> & tensor_map)
{
    auto it0 = tensor_map.find(node->src[0]);
    GGML_ASSERT(it0 != tensor_map.end());

    ov::Output<ov::Node> input = it0->second;
    const auto & ov_shape = input.get_shape();
    const int ndims = (int) ov_shape.size();

    // TRANSPOSE swaps ggml dims 0 and 1, which are OV dims (ndims-1) and (ndims-2).
    std::vector<int64_t> ov_order(ndims);
    std::iota(ov_order.begin(), ov_order.end(), 0);
    std::swap(ov_order[ndims - 1], ov_order[ndims - 2]);

    auto order_const = ov::op::v0::Constant::create(
        ov::element::i64, {(size_t) ndims}, ov_order);

    return std::make_shared<ov::op::v1::Transpose>(input, order_const);
}

// ============================================================================
// Translate VIEW
//
// VIEW creates a sub-range view of the source tensor at a byte offset.
// In OV, this is implemented as dimension-wise Slice ops.
// If the view has the same number of elements as the source, it degenerates
// into a Reshape.
// ============================================================================

static ov::Output<ov::Node> translate_view(
    const struct ggml_tensor * node,
    const std::unordered_map<const struct ggml_tensor *, ov::Output<ov::Node>> & tensor_map)
{
    auto it0 = tensor_map.find(node->src[0]);
    GGML_ASSERT(it0 != tensor_map.end());

    ov::Output<ov::Node> input = it0->second;
    const struct ggml_tensor * src = node->src[0];

    // If same number of elements, treat as a reshape
    if (ggml_nelements(node) == ggml_nelements(src)) {
        ov::Shape target = ggml_tensor_to_ov_shape(node);
        std::vector<int64_t> shape_vec(target.begin(), target.end());
        auto shape_const = ov::op::v0::Constant::create(
            ov::element::i64, {shape_vec.size()}, shape_vec);
        return std::make_shared<ov::op::v1::Reshape>(input, shape_const, false);
    }

    // Compute byte offset relative to src[0].
    // node->view_offs is relative to view_src; subtract src[0]'s own view_offs.
    const size_t src_view_offs = src->view_src ? src->view_offs : 0;
    const size_t byte_offset   = node->view_offs - src_view_offs;
    const size_t elem_size     = ggml_type_size(node->type);

    // Effective dimensionality (same logic as ggml_tensor_to_ov_shape)
    int ndims = GGML_MAX_DIMS;
    while (ndims > 2 && src->ne[ndims - 1] == 1) {
        ndims--;
    }

    // Decompose element offset into per-dimension start indices
    // using the source tensor's contiguous layout
    const size_t element_offset = byte_offset / elem_size;
    std::vector<int64_t> starts(ndims, 0);
    {
        size_t remaining = element_offset;
        for (int i = ndims - 1; i >= 1; i--) {
            size_t stride = 1;
            for (int j = 0; j < i; j++) {
                stride *= (size_t) src->ne[j];
            }
            starts[i] = (int64_t)(remaining / stride);
            remaining  = remaining % stride;
        }
        starts[0] = (int64_t) remaining;
    }

    // Slice each dimension that is sub-ranged
    ov::Output<ov::Node> result = input;
    for (int ggml_dim = 0; ggml_dim < ndims; ggml_dim++) {
        if (node->ne[ggml_dim] < src->ne[ggml_dim] || starts[ggml_dim] > 0) {
            const int ov_dim = ndims - 1 - ggml_dim;  // OV dims are reversed

            auto start_c = ov::op::v0::Constant::create(
                ov::element::i64, {1}, {starts[ggml_dim]});
            auto stop_c  = ov::op::v0::Constant::create(
                ov::element::i64, {1}, {starts[ggml_dim] + (int64_t) node->ne[ggml_dim]});
            auto step_c  = ov::op::v0::Constant::create(
                ov::element::i64, {1}, {(int64_t) 1});
            auto axes_c  = ov::op::v0::Constant::create(
                ov::element::i64, {1}, {(int64_t) ov_dim});

            result = std::make_shared<ov::op::v8::Slice>(result, start_c, stop_c, step_c, axes_c);
        }
    }

    return result;
}

static bool inpu_is_passthrough_view_like_op(enum ggml_op op) {
    switch (op) {
        case GGML_OP_VIEW:
        case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            return false;
    }
}

static const struct ggml_tensor * inpu_get_input_materialization_tensor(const struct ggml_tensor * t) {
    while (t && inpu_is_passthrough_view_like_op(t->op) && t->src[0]) {
        t = t->src[0];
    }

    return t;
}

static ov::Output<ov::Node> inpu_replay_input_view_chain(
    const struct ggml_tensor * leaf,
    const struct ggml_tensor * base,
    std::unordered_map<const struct ggml_tensor *, ov::Output<ov::Node>> & tensor_map)
{
    if (leaf == base) {
        return tensor_map.at(base);
    }

    std::vector<const struct ggml_tensor *> chain;
    for (auto * cur = leaf; cur && cur != base; cur = cur->src[0]) {
        chain.push_back(cur);
    }

    GGML_ASSERT(!chain.empty() && chain.back()->src[0] == base);

    for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
        const struct ggml_tensor * node = *it;
        ov::Output<ov::Node> output;

        switch (node->op) {
            case GGML_OP_VIEW:
                output = translate_view(node, tensor_map);
                break;
            case GGML_OP_RESHAPE:
                output = translate_reshape(node, tensor_map);
                break;
            case GGML_OP_PERMUTE:
                output = translate_permute(node, tensor_map);
                break;
            case GGML_OP_TRANSPOSE:
                output = translate_transpose(node, tensor_map);
                break;
            default:
                GGML_ABORT("unexpected external input op in chain: %s", ggml_op_name(node->op));
        }

        tensor_map[node] = output;
    }

    return tensor_map.at(leaf);
}

static const struct ggml_tensor * inpu_get_output_materialization_tensor(const struct ggml_tensor * t) {
    while (t && inpu_is_passthrough_view_like_op(t->op)) {
        t = t->src[0];
    }

    return t;
}

static int inpu_find_node_index(const struct ggml_cgraph * cgraph, const struct ggml_tensor * tensor) {
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        if (cgraph->nodes[i] == tensor) {
            return i;
        }
    }

    return -1;
}

// ============================================================================
// Main translation entry point
// ============================================================================

inpu_translate_result inpu_translate_graph(const struct ggml_cgraph * cgraph, const inpu_graph_io & io) {
    inpu_translate_result result;

    ov::ParameterVector parameters;
    ov::ResultVector results;

    // Map from ggml tensor pointer → OV Output
    std::unordered_map<const struct ggml_tensor *, ov::Output<ov::Node>> tensor_map;

    std::unordered_set<const struct ggml_tensor *> node_outputs;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        node_outputs.insert(cgraph->nodes[i]);
    }

    std::unordered_set<const struct ggml_tensor *> live_nodes;
    std::vector<const struct ggml_tensor *> live_stack;
    for (const auto & out : io.outputs) {
        live_stack.push_back(out.tensor);
    }
    while (!live_stack.empty()) {
        const struct ggml_tensor * node = live_stack.back();
        live_stack.pop_back();

        if (!node_outputs.count(node) || live_nodes.count(node)) {
            continue;
        }
        live_nodes.insert(node);

        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (node->src[s] && node_outputs.count(node->src[s])) {
                live_stack.push_back(node->src[s]);
            }
        }
    }

    // I/O maps for the compiled graph
    std::vector<inpu_compiled_graph::io_entry> input_map;
    std::vector<inpu_compiled_graph::io_entry> output_map;

    // -----------------------------------------------------------------------
    // Create OV Parameters for each graph input
    // -----------------------------------------------------------------------
    for (const auto & inp : io.inputs) {
        const struct ggml_tensor * t = inp.tensor;

        if (inp.is_weight && t->extra) {
            auto * extra = static_cast<ggml_inpu_tensor_extra *>(t->extra);

            if (extra->is_quantized) {
                // Create two parameters: quants + scales
                std::string qname = make_param_name("w_quant", inp.node_idx, inp.src_idx);
                std::string sname = make_param_name("w_scale", inp.node_idx, inp.src_idx);

                auto dq = create_dequant_subgraph(t, qname, sname);
                parameters.push_back(dq.quant_param);
                parameters.push_back(dq.scale_param);

                tensor_map[t] = dq.output;

                // Add to input map
                inpu_compiled_graph::io_entry qe;
                qe.ov_name   = qname;
                qe.node_idx  = inp.node_idx;
                qe.bind_node_idx = inp.node_idx;
                qe.src_idx   = inp.src_idx;
                qe.is_quant  = true;
                qe.is_scale  = false;
                input_map.push_back(qe);

                inpu_compiled_graph::io_entry se;
                se.ov_name   = sname;
                se.node_idx  = inp.node_idx;
                se.bind_node_idx = inp.node_idx;
                se.src_idx   = inp.src_idx;
                se.is_quant  = false;
                se.is_scale  = true;
                input_map.push_back(se);

                continue;
            }
        }

        if (t->type == GGML_TYPE_Q4_0 || t->type == GGML_TYPE_Q8_0) {
            const char * buffer_name = t->buffer ? ggml_backend_buffer_name(t->buffer) : "<none>";
            INPU_LOG_ERROR(
                "quantized input '%s' is not backed by repacked iNPU weights (buffer=%s, extra=%p)\n",
                ggml_get_name(t), buffer_name, t->extra);
            return result;
        }

        // Non-quantized input (F16/F32 activation or F16 weight)
        std::string name = make_param_name("input", inp.node_idx, inp.src_idx);
        const struct ggml_tensor * bind_tensor = inpu_get_input_materialization_tensor(t);
        ov::element::Type ov_type = ggml_type_to_ov(bind_tensor->type);
        ov::Shape shape = ggml_tensor_to_ov_shape(bind_tensor);

        auto param = std::make_shared<ov::op::v0::Parameter>(ov_type, shape);
        param->set_friendly_name(name);
        param->output(0).set_names({name});
        parameters.push_back(param);

        tensor_map[bind_tensor] = param->output(0);
        if (t != bind_tensor) {
            tensor_map[t] = inpu_replay_input_view_chain(t, bind_tensor, tensor_map);
        }

        inpu_compiled_graph::io_entry ie;
        ie.ov_name   = name;
        ie.node_idx  = inp.node_idx;
        ie.bind_node_idx = inp.node_idx;
        ie.src_idx   = inp.src_idx;
        ie.is_quant  = false;
        ie.is_scale  = false;
        input_map.push_back(ie);
    }

    // -----------------------------------------------------------------------
    // Translate each compute node
    // -----------------------------------------------------------------------
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if (!live_nodes.count(node)) {
            continue;
        }

        // Skip NONE ops
        if (node->op == GGML_OP_NONE) {
            if (node->src[0] && tensor_map.count(node->src[0])) {
                tensor_map[node] = tensor_map[node->src[0]];
            }
            continue;
        }

        ov::Output<ov::Node> output;

        try {
            switch (node->op) {
                case GGML_OP_MUL_MAT:
                    output = translate_mul_mat(node, tensor_map);
                    break;

                case GGML_OP_ADD:
                    output = translate_add(node, tensor_map);
                    break;

                case GGML_OP_MUL:
                    output = translate_mul(node, tensor_map);
                    break;

                case GGML_OP_GLU:
                    output = translate_glu(node, tensor_map);
                    break;

                case GGML_OP_RMS_NORM:
                    output = translate_rms_norm(node, tensor_map);
                    break;

                case GGML_OP_ROPE:
                    output = translate_rope(node, tensor_map);
                    break;

                case GGML_OP_RESHAPE:
                    output = translate_reshape(node, tensor_map);
                    break;

                case GGML_OP_PERMUTE:
                    output = translate_permute(node, tensor_map);
                    break;

                case GGML_OP_TRANSPOSE:
                    output = translate_transpose(node, tensor_map);
                    break;

                case GGML_OP_VIEW:
                    output = translate_view(node, tensor_map);
                    break;

                default:
                    INPU_LOG_ERROR("unsupported op in translation: %s\n", ggml_op_name(node->op));
                    return result; // return empty (model == nullptr)
            }
        } catch (const std::exception & e) {
            INPU_LOG_ERROR("failed to translate op %s at node %d: %s\n", ggml_op_name(node->op), i, e.what());
            return result;
        }

        tensor_map[node] = output;
    }

    // -----------------------------------------------------------------------
    // Create OV Results for each graph output
    //
    // View-like outputs alias another tensor's storage. Writing densely into
    // the view pointer is incorrect for strided aliases, so materialize the
    // backing tensor instead and let downstream ggml views interpret it.
    // -----------------------------------------------------------------------
    std::unordered_set<const struct ggml_tensor *> seen_output_tensors;
    for (const auto & out : io.outputs) {
        const struct ggml_tensor * bind_tensor = inpu_get_output_materialization_tensor(out.tensor);
        if (!bind_tensor) {
            INPU_LOG_ERROR("failed to resolve backing tensor for output node %d\n", out.node_idx);
            return result;
        }

        int bind_node_idx = inpu_find_node_index(cgraph, bind_tensor);

        // If the backing tensor lives outside this compiled subgraph, keep the
        // original output tensor. This happens for external view/slice aliases
        // such as KV-cache writes, where there is no internal contiguous tensor
        // we can materialize instead.
        if (bind_node_idx < 0) {
            bind_tensor = out.tensor;
            bind_node_idx = out.node_idx;
        }

        if (seen_output_tensors.count(bind_tensor)) {
            continue;
        }
        seen_output_tensors.insert(bind_tensor);

        auto it = tensor_map.find(bind_tensor);
        if (it == tensor_map.end()) {
            INPU_LOG_ERROR("output tensor not found in tensor_map (node %d)\n", out.node_idx);
            return result;
        }

        std::string name = "output_n" + std::to_string(bind_node_idx);
        auto res = std::make_shared<ov::op::v0::Result>(it->second);
        res->set_friendly_name(name);
        // Set tensor name on the Result's input so get_tensor(name) works
        res->input(0).get_source_output().add_names({name});
        results.push_back(res);

        inpu_compiled_graph::io_entry oe;
        oe.ov_name   = name;
        oe.node_idx  = out.node_idx;
        oe.bind_node_idx = bind_node_idx;
        oe.src_idx   = -1;
        oe.is_quant  = false;
        oe.is_scale  = false;
        output_map.push_back(oe);
    }

    // -----------------------------------------------------------------------
    // Build the OV Model
    // -----------------------------------------------------------------------
    try {
        result.model = std::make_shared<ov::Model>(results, parameters, "inpu_graph");
        result.input_map = std::move(input_map);
        result.output_map = std::move(output_map);
    } catch (const std::exception & e) {
        INPU_LOG_ERROR("failed to create OV Model: %s\n", e.what());
        result.model = nullptr;
    }

    return result;
}
