#include "utils.hpp"

#include <cstddef>
#include <ctime>
#include <memory>
#include <openvino/op/add.hpp>
#include <openvino/op/clamp.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/cos.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/maximum.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/sin.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/transpose.hpp>
#include <string>

#include "ggml-impl.h"

namespace ov {
namespace frontend {
namespace ggml {

std::string getCurrentTime() {
    std::time_t now = std::time(nullptr);
    char buf[100];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return buf;
}

void num_inputs_check(const NodeContext& context, size_t min_inputs, size_t max_inputs) {
    auto input_size = context.get_input_size();
    FRONT_END_OP_CONVERSION_CHECK(input_size >= min_inputs, "Got less inputs than expected");
    FRONT_END_OP_CONVERSION_CHECK(input_size <= max_inputs, "Got more inputs than expected");
}

int non_cont_dim(std::vector<size_t> ne, std::vector<size_t> nb) {
    int dim = nb.size() - 1;
    size_t bytes = nb[dim];
    for (int i = dim; i > 0; i--) {
        bytes *= ne[i];
        if (bytes != nb[i - 1]) {
            return i;
        }
    }
    return 0;
}

std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::op::v3::ShapeOf>& shape,
                                         const std::vector<int>& dims) {
    using namespace ov::op;
    const auto zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    const auto dims_const = v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
    return std::make_shared<v8::Gather>(shape, dims_const, zero);
}

std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::Node>& node, const std::vector<int>& dims) {
    return get_dimensions(std::make_shared<ov::op::v3::ShapeOf>(node), dims);
}

OutputVector rename_outputs_with_suffix(const OutputVector& outputs, const std::string& suffix) {
    for (const auto& output : outputs) {
        auto node = output.get_node_shared_ptr();
        std::string name = node->get_friendly_name();
        name += "_";
        name += suffix;
        node->set_friendly_name(name);
        // std::cout << name << "  " << output.get_partial_shape() << std::endl;
    }
    return outputs;
}

namespace {
ov::Output<ov::Node> rope_yarn_ramp_mix(int n_dims, const float corr_dims[2], float ext_factor) {
    int half_n_dims = n_dims / 2;
    std::vector<float> dim_ids_vec(half_n_dims);
    std::iota(dim_ids_vec.begin(), dim_ids_vec.end(), 0);
    auto dim_ids = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, (size_t) half_n_dims}, dim_ids_vec);
    auto corr_low = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1}, {corr_dims[0]});
    auto corr_high = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1}, {corr_dims[1]});
    auto denom =
        std::make_shared<ov::op::v1::Maximum>(std::make_shared<ov::op::v1::Subtract>(corr_high, corr_low),
                                              ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1}, {0.001f}));
    auto ramp_y =
        std::make_shared<ov::op::v1::Divide>(std::make_shared<ov::op::v1::Subtract>(dim_ids, corr_low), denom);
    auto ramp_clamped = std::make_shared<ov::op::v0::Clamp>(ramp_y, 0.0f, 1.0f);
    auto ext_factor_node = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {ext_factor});
    auto ramp_mix = std::make_shared<ov::op::v1::Multiply>(ramp_clamped, ext_factor_node);
    return ramp_mix;
}

float ggml_rope_yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif
    return n_dims * logf(n_ctx_orig / (n_rot * 2 * (float) M_PI)) / (2 * logf(base));
}

void ggml_rope_yarn_corr_dims(int n_dims,
                              int n_ctx_orig,
                              float freq_base,
                              float beta_fast,
                              float beta_slow,
                              float dims[2]) {
    float start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    float end = ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = std::max(0.0f, start);
    dims[1] = std::min(static_cast<float>(n_dims - 1), end);
}
}  // namespace

std::pair<ov::Output<Node>, ov::Output<Node>> make_sin_cos(int32_t* rope_params,
                                                           std::shared_ptr<ov::Node> inp_pos,
                                                           std::shared_ptr<ov::Node> rope_freqs_weight) {
    GGML_LOG_WARN("%s: Starting make_sin_cos function\n", __func__);
    
    // Check for null pointers first
    if (!rope_params) {
        GGML_LOG_WARN("%s: rope_params is null!\n", __func__);
        return {};
    }
    if (!inp_pos) {
        GGML_LOG_WARN("%s: inp_pos is null!\n", __func__);
        return {};
    }
    
    GGML_LOG_WARN("%s: Parameters validated, proceeding with conversion\n", __func__);
    
    try {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        GGML_LOG_WARN("%s: Input position converted to f32\n", __func__);
        
        auto pos_perm = std::make_shared<ov::op::v0::Constant>(
            ov::element::i64, ov::Shape{3}, std::vector<int64_t>{2, 1, 0});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
        GGML_LOG_WARN("%s: Position transposed\n", __func__);

        // Read parameters with bounds checking
        GGML_LOG_WARN("%s: Reading rope parameters...\n", __func__);
        const int n_dims = rope_params[1];
        const int n_ctx_orig = rope_params[4];
        GGML_LOG_WARN("%s: n_dims: %d, n_ctx_orig: %d\n", __func__, n_dims, n_ctx_orig);
        
        if (n_dims <= 0 || n_dims > 8192) {
            GGML_LOG_WARN("%s: Invalid n_dims: %d\n", __func__, n_dims);
            return {};
        }
        
        float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
        
        GGML_LOG_WARN("%s: Copying float parameters from rope_params...\n", __func__);
        memcpy(&freq_base, rope_params + 5, sizeof(float));
        GGML_LOG_WARN("%s: freq_base: %f\n", __func__, freq_base);
        
        memcpy(&freq_scale, rope_params + 6, sizeof(float));
        GGML_LOG_WARN("%s: freq_scale: %f\n", __func__, freq_scale);
        
        memcpy(&ext_factor, rope_params + 7, sizeof(float));
        GGML_LOG_WARN("%s: ext_factor: %f\n", __func__, ext_factor);
        
        memcpy(&attn_factor, rope_params + 8, sizeof(float));
        GGML_LOG_WARN("%s: attn_factor: %f\n", __func__, attn_factor);
        
        memcpy(&beta_fast, rope_params + 9, sizeof(float));
        GGML_LOG_WARN("%s: beta_fast: %f\n", __func__, beta_fast);
        
        memcpy(&beta_slow, rope_params + 10, sizeof(float));
        GGML_LOG_WARN("%s: beta_slow: %f\n", __func__, beta_slow);

        const float theta_scale = powf(freq_base, -2.0f / n_dims);
        GGML_LOG_WARN("%s: theta_scale calculated: %f\n", __func__, theta_scale);

        float corr_dims[2];
        GGML_LOG_WARN("%s: Calculating correlation dimensions...\n", __func__);
        ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
        GGML_LOG_WARN("%s: corr_dims: [%f, %f]\n", __func__, corr_dims[0], corr_dims[1]);

        GGML_LOG_WARN("%s: Creating factor vector with size: %d\n", __func__, (n_dims / 2));
        std::vector<float> factor(n_dims / 2);
        factor[0] = 1.0f;
        for (size_t i = 1; i < factor.size(); i++) {
            factor[i] = theta_scale * factor[i - 1];
        }
        GGML_LOG_WARN("%s: Factor vector created successfully\n", __func__);

        GGML_LOG_WARN("%s: Creating frequency factors constant...\n", __func__);
        Output<Node> freq_factors = std::make_shared<ov::op::v0::Constant>(
            ov::element::f32, ov::Shape{1, 1, factor.size()}, factor);
        GGML_LOG_WARN("%s: Frequency factors constant created\n", __func__);
        
        if (rope_freqs_weight) {
            GGML_LOG_WARN("%s: Applying rope_freqs_weight division\n", __func__);
            freq_factors = std::make_shared<ov::op::v1::Divide>(freq_factors, rope_freqs_weight);
        }

        GGML_LOG_WARN("%s: Creating theta calculations...\n", __func__);
        auto theta_extrap = std::make_shared<ov::op::v1::Multiply>(freq_factors, inp_pos);
        auto theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));
        GGML_LOG_WARN("%s: Theta calculations created\n", __func__);

        Output<Node> theta;
        float mscale = attn_factor;
        
        GGML_LOG_WARN("%s: Processing extension factor: %f\n", __func__, ext_factor);
        if (ext_factor == 0.0f) {
            GGML_LOG_WARN("%s: Using interpolated theta (ext_factor == 0)\n", __func__);
            theta = theta_interp;
        } else {
            GGML_LOG_WARN("%s: Processing yarn ramp mix...\n", __func__);
            auto ramp_mix = rope_yarn_ramp_mix(n_dims, corr_dims, ext_factor);
            auto one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1}, {1.0f});
            auto one_minus_ramp = std::make_shared<ov::op::v1::Subtract>(one, ramp_mix);

            theta = std::make_shared<ov::op::v1::Add>(
                std::make_shared<ov::op::v1::Multiply>(theta_interp, one_minus_ramp),
                std::make_shared<ov::op::v1::Multiply>(theta_extrap, ramp_mix));
            mscale *= (1.0f + 0.1f * std::log(1.0f / freq_scale));
            GGML_LOG_WARN("%s: Yarn processing completed, mscale: %f\n", __func__, mscale);
        }

        GGML_LOG_WARN("%s: Creating trigonometric functions...\n", __func__);
        Output<Node> cos_theta = std::make_shared<ov::op::v0::Cos>(theta);
        Output<Node> sin_theta = std::make_shared<ov::op::v0::Sin>(theta);
        GGML_LOG_WARN("%s: Trigonometric functions created\n", __func__);

        auto mscale_node = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {mscale});
        
        GGML_LOG_WARN("%s: Applying mscale...\n", __func__);
        cos_theta = std::make_shared<ov::op::v1::Multiply>(cos_theta, mscale_node);
        sin_theta = std::make_shared<ov::op::v1::Multiply>(sin_theta, mscale_node);
        
        GGML_LOG_WARN("%s: make_sin_cos completed successfully\n", __func__);
        return std::make_pair(sin_theta, cos_theta);
        
    } catch (const std::exception& e) {
        GGML_LOG_WARN("%s: Exception in make_sin_cos: %s\n", __func__, e.what());
        return {};
    } catch (...) {
        GGML_LOG_WARN("%s: Unknown exception in make_sin_cos\n", __func__);
        return {};
    }
}
ov::Output<ov::Node> process_view_input(const NodeContext& context, int input_index, int slice_len) {
    // Only works for VIEW operations that slice at the lowest dimension
    // If the VIEW also reshape the result, `slice_len` should be provided
    auto input = context.get_input(input_index);
    int32_t* op_params = context.get_input_op_params(input_index);
    auto src1_stride = context.get_input_stride(input_index);

    int64_t split_addr = op_params[0] / src1_stride[2];
    if (slice_len == 0) {
        slice_len = context.get_input_shape(input_index)[2].get_length();
    }
    int64_t slice_end = split_addr + slice_len;

    auto begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {split_addr});
    auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_end});
    auto stride = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    auto axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});
    auto sliced = std::make_shared<ov::op::v8::Slice>(input, begin, end, stride, axes);
    return sliced;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
