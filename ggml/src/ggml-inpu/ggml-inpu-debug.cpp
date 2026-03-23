// ggml-inpu-debug.cpp — debug dump utilities for the iNPU backend
//
// Controlled by environment variables:
//   GGML_INPU_DUMP_CGRAPH=1        — log ggml cgraph details
//   GGML_INPU_DUMP_IR=<dir>        — write OV IR to files (use "1" for cwd)

#include "ggml-inpu-debug.h"
#include "ggml-inpu-impl.h"
#include "ggml-impl.h"

#include <openvino/pass/serialize.hpp>

#include <atomic>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>

// ============================================================================
// Logging (match the convention used in other iNPU files)
// ============================================================================

#define INPU_LOG_DEBUG(...) GGML_LOG_DEBUG("INPU-debug: " __VA_ARGS__)
#define INPU_LOG_INFO(...)  GGML_LOG_INFO("INPU-debug: " __VA_ARGS__)
#define INPU_LOG_WARN(...)  GGML_LOG_WARN("INPU-debug: " __VA_ARGS__)
#define INPU_LOG_ERROR(...) GGML_LOG_ERROR("INPU-debug: " __VA_ARGS__)

// ============================================================================
// Environment variable helpers (cached after first call)
// ============================================================================

bool inpu_debug_dump_cgraph_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * val = std::getenv("GGML_INPU_DUMP_CGRAPH");
        cached = (val && val[0] != '0' && val[0] != '\0') ? 1 : 0;
        if (cached) {
            INPU_LOG_INFO("cgraph dump enabled (GGML_INPU_DUMP_CGRAPH=%s)\n", val);
        }
    }
    return cached != 0;
}

const char * inpu_debug_dump_ir_dir() {
    static const char * cached = nullptr;
    static bool checked = false;
    if (!checked) {
        checked = true;
        const char * val = std::getenv("GGML_INPU_DUMP_IR");
        if (val && val[0] != '0' && val[0] != '\0') {
            // "1" or "true" → use current directory; otherwise treat as directory path
            if (std::strcmp(val, "1") == 0 || std::strcmp(val, "true") == 0) {
                cached = ".";
            } else {
                cached = val;
            }
            INPU_LOG_INFO("IR dump enabled → directory: %s (GGML_INPU_DUMP_IR=%s)\n", cached, val);
        }
    }
    return cached;
}

// ============================================================================
// Cgraph dump
// ============================================================================

void inpu_debug_dump_cgraph(const struct ggml_cgraph * cgraph) {
    std::ostringstream os;
    os << "=== CGRAPH ===\n";
    os << "n_nodes = " << cgraph->n_nodes << ", n_leafs = " << cgraph->n_leafs << "\n";

    // clang-format off
    os << " " << std::setw(3) << "#"
       << std::setw(15)  << "shape"
       << std::setw(20)  << "op"
       << std::setw(48)  << "name"
       << std::setw(35)  << "stride"
       << std::setw(20)  << "buffer"
       << "\n";
    // clang-format on

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        const char * buf_name = "none";
        ggml_backend_buffer_t buf = node->view_src ? node->view_src->buffer : node->buffer;
        if (buf) {
            buf_name = ggml_backend_buffer_name(buf);
        }

        // clang-format off
        os << " - " << std::setw(3) << i << ": [ "
           << std::setw(5) << node->ne[0] << ", "
           << std::setw(5) << node->ne[1] << ", "
           << std::setw(5) << node->ne[2] << ", "
           << std::setw(5) << node->ne[3] << "] "
           << std::left << std::setw(20) << ggml_op_name(node->op) << std::right << " "
           << std::left << std::setw(45) << ggml_get_name(node)    << std::right
           << "[ "
           << std::setw(0) << node->nb[0] << ", "
           << std::setw(5) << node->nb[1] << ", "
           << std::setw(5) << node->nb[2] << ", "
           << std::setw(5) << node->nb[3] << "] "
           << std::setw(15) << buf_name
           << "\n";
        // clang-format on

        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (auto * src = node->src[s]) {
                const char * src_buf_name = "none";
                ggml_backend_buffer_t src_buf = src->view_src ? src->view_src->buffer : src->buffer;
                if (src_buf) {
                    src_buf_name = ggml_backend_buffer_name(src_buf);
                }

                // clang-format off
                os << std::setw(10) << " [ "
                   << std::setw(5) << src->ne[0] << ", "
                   << std::setw(5) << src->ne[1] << ", "
                   << std::setw(5) << src->ne[2] << ", "
                   << std::setw(5) << src->ne[3] << "] "
                   << std::setw(12)
                   << s << ": " << std::left << std::setw(12) << ggml_op_name(src->op) << std::right
                   << std::left << std::setw(30) << ggml_get_name(src) << std::right
                   << "[ "
                   << std::setw(0) << src->nb[0] << ", "
                   << std::setw(5) << src->nb[1] << ", "
                   << std::setw(5) << src->nb[2] << ", "
                   << std::setw(5) << src->nb[3] << "] "
                   << std::setw(15) << src_buf_name
                   << "\n";
                // clang-format on
            }
        }
    }

    if (cgraph->n_leafs > 0) {
        os << "n_leafs = " << cgraph->n_leafs << "\n";
        for (int i = 0; i < cgraph->n_leafs; i++) {
            struct ggml_tensor * leaf = cgraph->leafs[i];

            const char * leaf_buf_name = "none";
            ggml_backend_buffer_t leaf_buf = leaf->view_src ? leaf->view_src->buffer : leaf->buffer;
            if (leaf_buf) {
                leaf_buf_name = ggml_backend_buffer_name(leaf_buf);
            }

            // clang-format off
            os << " - " << std::setw(3) << i << ": [ "
               << std::setw(5) << leaf->ne[0] << ", "
               << std::setw(5) << leaf->ne[1] << "] "
               << std::setw(8) << ggml_op_name(leaf->op) << " "
               << std::setw(16) << ggml_get_name(leaf)
               << std::setw(20) << leaf_buf_name << "\n";
            // clang-format on
        }
    }

    os << "========================================\n";

    INPU_LOG_INFO("%s", os.str().c_str());
}

// ============================================================================
// OV IR dump
// ============================================================================

void inpu_debug_dump_ir(const std::shared_ptr<ov::Model> & model) {
    const char * dir = inpu_debug_dump_ir_dir();
    if (!dir || !model) {
        return;
    }

    static std::atomic<int> counter{0};
    int n = counter.fetch_add(1);

    std::string xml_path = std::string(dir) + "/inpu_model_" + std::to_string(n) + ".xml";
    std::string bin_path = std::string(dir) + "/inpu_model_" + std::to_string(n) + ".bin";

    try {
        ov::serialize(model, xml_path, bin_path);
        INPU_LOG_INFO("IR dumped → %s  (%zu params, %zu results)\n",
                      xml_path.c_str(), model->get_parameters().size(), model->get_results().size());
    } catch (const std::exception & e) {
        INPU_LOG_ERROR("failed to dump IR to %s: %s\n", xml_path.c_str(), e.what());
    }

    // Also log a brief model summary
    INPU_LOG_INFO("  model name: %s\n", model->get_friendly_name().c_str());
    for (const auto & param : model->get_parameters()) {
        INPU_LOG_INFO("  param: %-30s  shape=%s  type=%s\n",
                      param->get_friendly_name().c_str(),
                      param->get_output_shape(0).to_string().c_str(),
                      param->get_output_element_type(0).get_type_name().c_str());
    }
    for (const auto & result : model->get_results()) {
        INPU_LOG_INFO("  result: %-29s  shape=%s  type=%s\n",
                      result->get_friendly_name().c_str(),
                      result->get_output_shape(0).to_string().c_str(),
                      result->get_output_element_type(0).get_type_name().c_str());
    }
}
