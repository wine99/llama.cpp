#pragma once

// ggml-inpu debug utilities — controlled by environment variables
//
// GGML_INPU_DUMP_CGRAPH=1        — log detailed cgraph info on every graph_compute
// GGML_INPU_DUMP_IR=<dir>        — serialize OV IR (xml+bin) to <dir>/inpu_model_N.{xml,bin}
//                                   use "1" to write to current working directory
// GGML_INPU_PROFILE=1            — print iNPU profiling summary at process exit
// GGML_INPU_PROFILE_VERBOSE=1    — log per-graph profiling details for every call
// GGML_INPU_PROFILE_TOP=<n>      — number of hot graphs to include in the summary (default: 8)

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <openvino/openvino.hpp>
#include <string>

// Check whether cgraph dumping is enabled (cached after first call)
bool inpu_debug_dump_cgraph_enabled();

// Return the IR dump directory, or nullptr if disabled (cached after first call)
const char * inpu_debug_dump_ir_dir();

// Check whether runtime profiling is enabled (cached after first call)
bool inpu_debug_profile_enabled();

// Check whether per-call profiling logs are enabled (cached after first call)
bool inpu_debug_profile_verbose_enabled();

// Print a detailed text dump of the ggml cgraph to GGML_LOG_INFO
void inpu_debug_dump_cgraph(const struct ggml_cgraph * cgraph);

// Serialize the OV model to XML/BIN files in the configured directory
void inpu_debug_dump_ir(const std::shared_ptr<ov::Model> & model);

// Build a short, human-readable graph label for profiling and logging
std::string inpu_debug_graph_label(const struct ggml_cgraph * cgraph);

struct inpu_profile_graph_call {
    size_t      graph_hash = 0;
    std::string graph_label;
    int         n_nodes           = 0;
    bool        cache_hit         = false;
    bool        request_from_pool = false;
    bool        success           = false;

    uint64_t key_us             = 0;
    uint64_t cache_lookup_us    = 0;
    uint64_t analyze_io_us      = 0;
    uint64_t translate_us       = 0;
    uint64_t compile_us         = 0;
    uint64_t acquire_request_us = 0;
    uint64_t bind_inputs_us     = 0;
    uint64_t bind_outputs_us    = 0;
    uint64_t infer_us           = 0;
    uint64_t release_request_us = 0;
    uint64_t total_us           = 0;

    size_t input_bytes    = 0;
    size_t output_bytes   = 0;
    size_t quant_bytes    = 0;
    size_t scale_bytes    = 0;
    size_t input_tensors  = 0;
    size_t output_tensors = 0;
};

void inpu_profile_record_graph_call(const inpu_profile_graph_call & call);
void inpu_profile_record_buffer_set(enum ggml_type type, size_t bytes, uint64_t elapsed_us);
void inpu_profile_record_buffer_get(enum ggml_type type, size_t bytes, uint64_t elapsed_us);
