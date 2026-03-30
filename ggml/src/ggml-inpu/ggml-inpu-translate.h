#pragma once

// ggml-inpu translate: convert ggml cgraph to OpenVINO IR

#include "ggml-inpu-impl.h"
#include "ggml-inpu-cache.h"

#include <openvino/openvino.hpp>

#include <memory>

// ---------------------------------------------------------------------------
// Graph I/O analysis
// ---------------------------------------------------------------------------

struct inpu_graph_io {
    // An input to the OV model
    struct input_info {
        const struct ggml_tensor * tensor;
        int node_idx;     // -1 if the tensor is not a node output
        int src_idx;      // which src[] of the consuming node
        bool is_weight;   // in iNPU buffer (quantized or F16)
    };

    // An output from the OV model
    struct output_info {
        const struct ggml_tensor * tensor;
        int node_idx;     // which node produces this output
    };

    std::vector<input_info>  inputs;
    std::vector<output_info> outputs;
};

// Analyze a cgraph to determine its I/O
inpu_graph_io inpu_analyze_graph_io(const struct ggml_cgraph * cgraph);

// ---------------------------------------------------------------------------
// OV model translation
// ---------------------------------------------------------------------------

struct inpu_translate_result {
    std::shared_ptr<ov::Model> model;
    std::vector<inpu_compiled_graph::io_entry> input_map;
    std::vector<inpu_compiled_graph::io_entry> output_map;
};

// Translate a ggml cgraph into an OV Model
inpu_translate_result inpu_translate_graph(const struct ggml_cgraph * cgraph, const inpu_graph_io & io);
