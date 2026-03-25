I want to implement a backend for Intel NPUs. OpenVINO is the only entrance to use the NPU. I have an in-progress implementation of the OpenVINO backend, but I don't think that project will achieve long-term success. I have a summary of the OV backend in `llama.cpp-ov.typ`. The codebase of the OV backend is the folder `~/llama.cpp` and the branch `dev_backend_openvino`, the PR is `https://github.com/ggml-org/llama.cpp/pull/15307`. You should read the implementation in that codebase to help you implement the iNPU backend. You can either read that folder directly or pull the branch as a submodule in this repo.

Consider the architecture mismatch between OV and llama.cpp. This iNPU backend will not be a backend that runs the entire graph but will only accelerate compute ops.

To help you understand the NPU:
- NPU only supports fixed-shape OpenVINO graphs (OV IR).
- The NPUW plugin is a wrapper around the NPU plugin in OpenVINO to support LLMs. What it does is find the repeating layers and split the graph into subgraphs. We do not need NPUW for our implementation.
- NPU's best supported quantization scheme is int4 symmetric with channel-wise or group-wise of size 128. So the iNPU backend will only support Q4_0 and Q8_0 since they are symmetric. Q4_0 and Q8_0 group size is 32. The OV backend does a requantization to group size 128, but let's not do this. We should not sacrifice correctness for performance.
- NPU only supports int4 inputs/constants. Q4_0 and Q8_0 store uint4 data. The OV backend does an extra step of subtracting 8 from the uint4 data after the weights are loaded. Let's do this subtraction earlier, during model loading, so weights will be ready after loading is finished. The OV backend creates constant nodes for weights, but let's treat everything as inputs.

Before implementing the iNPU backend, a few decisions need to be made:
- Should the iNPU backend type be ACCEL since we are only accelerating compute ops? Does an ACCEL backend support defining its own backend buffer or extra_buffer_type? We need this because the subtraction of quantized weights must happen during tensor loading. The Hexagon backend (Qualcomm's NPU) defines its extra_buffer_type to repack quant weights, as does the CPU repack buffer type.
- We need an efficient caching mechanism for the compiled subgraphs. Since we only accelerate compute ops, the backend will see lots of small cgraphs.
- We need a way to determine which tensors are graph inputs and outputs in the converted OpenVINO IR. For example, if the cgraph contains two compute ops and the output of the second op has the same data address as the output of the first op, then the first op's output is only intermediate and not a graph output.
- The NPU's compute engines can only do arithmetic in fp16 (and fp8/int8, but that's not relevant here). However, the cgraph uses fp32 for all intermediate results. If we convert an IR with fp32 activation inputs and fp32 outputs, the NPU will do conversion on the fly, which will impact performance. Is there a way to make the cgraph mostly use fp16? What does SYCL_F16 do in the SYCL backend?
- The OV backend uses tensor names as keys in a lot of places, but it's not robust.

As a first step:
- Only accelerate MUL_MAT (and ADD and GLU, since it's common that an ADD or GLU follows a MUL_MAT; an OV IR with this pattern executes fast on the NPU because the ADD/GLU will be fused into the matmul in hardware). The supported MUL_MAT should be: weights in Q4_0/Q8_0/F16, activation in F16, output in F16.
- OV does not support strided tensors, so a compute op with a view tensor input is not supported. There are four view-type ops in ggml: VIEW, RESHAPE, PERMUTE, TRANSPOSE. But if the input is a RESHAPE, PERMUTE, or TRANSPOSE whose src is not itself a view-type op, then we can still support it since the translation of that RESHAPE/PERMUTE/TRANSPOSE is trivial.

During implementation, we need to make sure the following pass or run:
- test-backend-op, test-thread-safety
- llama-simple, llama-cli, llama-bench, llama-perplexity (and llama-server, if possible to test)

Throughout the implementation, please record all key design decisions, trade-offs, and rationale in a `docs/inpu-decisions.md` document. Update it as decisions are made or revised.

