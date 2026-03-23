#pragma once

// ggml-inpu debug utilities — controlled by environment variables
//
// GGML_INPU_DUMP_CGRAPH=1        — log detailed cgraph info on every graph_compute
// GGML_INPU_DUMP_IR=<dir>        — serialize OV IR (xml+bin) to <dir>/inpu_model_N.{xml,bin}
//                                   use "1" to write to current working directory

#include "ggml.h"

#include <openvino/openvino.hpp>

#include <memory>

// Check whether cgraph dumping is enabled (cached after first call)
bool inpu_debug_dump_cgraph_enabled();

// Return the IR dump directory, or nullptr if disabled (cached after first call)
const char * inpu_debug_dump_ir_dir();

// Print a detailed text dump of the ggml cgraph to GGML_LOG_INFO
void inpu_debug_dump_cgraph(const struct ggml_cgraph * cgraph);

// Serialize the OV model to XML/BIN files in the configured directory
void inpu_debug_dump_ir(const std::shared_ptr<ov::Model> & model);
