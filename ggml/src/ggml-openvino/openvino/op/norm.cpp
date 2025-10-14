#include <memory>
#include <openvino/op/constant.hpp>
#include <openvino/op/mvn.hpp>

#include "../node_context.hpp"
#include "../op_table.hpp"
#include "../utils.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_norm(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    auto input_node = context.get_input(0);
    
    // Get epsilon from op_params (same pattern as rms_norm)
    float eps = 1e-5f; // default
    if (context.get_output_op_params(0)) {
        memcpy(&eps, context.get_output_op_params(0), sizeof(float));
    }
    
    // Create axes tensor for the last dimension (features)
    // GGML NORM normalizes across the last dimension (ncols)
    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    
    // Use MVN-6 operation with correct constructor signature
    auto res = std::make_shared<ov::op::v6::MVN>(
        input_node,                          // data input
        axes,                               // reduction_axes input
        true,                               // normalize_variance = true
        eps,                                // eps (float)
        ov::op::MVNEpsMode::INSIDE_SQRT     // eps_mode
    );

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov