#include "ggml-impl.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include <intel_gpu/graph/network.hpp>
#include <intel_gpu/graph/topology.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/fully_connected.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/runtime/engine.hpp>
#include <intel_gpu/runtime/internal_properties.hpp>
#include <intel_gpu/runtime/memory.hpp>
#include <intel_gpu/runtime/stream.hpp>

// M0 smoke test: run one fully_connected through a hand-built cldnn network,
// verify against a CPU reference, print engine memory statistics.
// Triggered by GGML_OVGPU_SMOKE_TEST=1 at backend registration.

void ggml_ovgpu_smoke_test(cldnn::engine & engine) {
    using namespace cldnn;

    const int B = 4, K = 8, N = 16;

    // Replicate OV's working oneDNN FC test (bf16_onednn_ops_gpu_test.cpp:227):
    // matrix on [b,f], bfyx{B,K,1,1} input + {N,K,1,1} weight, default ctor
    // (input_size=2, weights_transposed=true).
    auto input   = engine.allocate_memory(layout{ov::element::f32, format::bfyx, tensor(B, K, 1, 1)});
    auto weights = engine.allocate_memory(layout{ov::element::f32, format::bfyx, tensor(N, K, 1, 1)});
    auto bias    = engine.allocate_memory(layout{ov::element::f32, format::bfyx, tensor(1, N, 1, 1)});

    topology topo(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("bias", bias),
        fully_connected("fc", input_info("input"), "weights", "bias"),
        reorder("out", input_info("fc"), format::bfyx, data_types::f32));

    ExecutionConfig config;
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::use_onednn(true));
    // NOTE: do NOT set allow_new_shape_infer(true) - the new-shape-infer path
    // (calc_output_layouts, plural) feeds raw rank-4 shapes to MatMul shape_infer
    // without the reshape_to_2d collapse, failing batch broadcast. The legacy
    // calc_output_layout (singular) does reshape_to_2d for supports_immad devices.
    auto stream = engine.create_stream(config);

    std::vector<float> h_input(B * K), h_weights(N * K), h_bias(N);
    for (int b = 0; b < B; b++) {
        for (int k = 0; k < K; k++) {
            h_input[b * K + k] = 0.1f * b + 0.01f * k;
        }
    }
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            h_weights[n * K + k] = 0.02f * n - 0.03f * k;
        }
        h_bias[n] = 0.001f * n;
    }

    {
        mem_lock<float, mem_lock_type::write> l_input(input, *stream);
        mem_lock<float, mem_lock_type::write> l_weights(weights, *stream);
        mem_lock<float, mem_lock_type::write> l_bias(bias, *stream);
        memcpy(l_input.data(), h_input.data(), h_input.size() * sizeof(float));
        memcpy(l_weights.data(), h_weights.data(), h_weights.size() * sizeof(float));
        memcpy(l_bias.data(), h_bias.data(), h_bias.size() * sizeof(float));
    }

    network net(engine, topo, config);
    net.set_input_data("input", input);

    auto outputs = net.execute();
    auto output = outputs.at("out").get_memory();

    std::vector<float> h_out(B * N);
    {
        mem_lock<float, mem_lock_type::read> l_out(output, *stream);
        memcpy(h_out.data(), l_out.data(), h_out.size() * sizeof(float));
    }

    double max_err = 0.0;
    for (int b = 0; b < B; b++) {
        for (int n = 0; n < N; n++) {
            float ref = h_bias[n];
            for (int k = 0; k < K; k++) {
                ref += h_input[b * K + k] * h_weights[n * K + k];
            }
            max_err = std::max(max_err, (double) std::fabs(ref - h_out[b * N + n]));
        }
    }

    bool pass = max_err < 1e-5;
    GGML_LOG_INFO("OVGPU smoke test (fully_connected %dx%dx%d): %s (max_err=%g)\n", B, N, K, pass ? "PASS" : "FAIL", max_err);

    for (const auto & kv : engine.get_memory_statistics()) {
        GGML_LOG_INFO("OVGPU mem stat: %-20s %.2f MB\n", kv.first.c_str(), kv.second / 1048576.0);
    }
}
