// ggml-inpu-debug.cpp — debug dump utilities for the iNPU backend
//
// Controlled by environment variables:
//   GGML_INPU_DUMP_CGRAPH=1        — log ggml cgraph details
//   GGML_INPU_DUMP_IR=<dir>        — write OV IR to files (use "1" for cwd)

#include "ggml-inpu-debug.h"

#include "ggml-backend.h"
#include "ggml-impl.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <mutex>
#include <openvino/pass/serialize.hpp>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

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

bool inpu_debug_profile_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * val = std::getenv("GGML_INPU_PROFILE");
        cached           = (val && val[0] != '0' && val[0] != '\0') ? 1 : 0;
        if (cached) {
            INPU_LOG_INFO("profiling enabled (GGML_INPU_PROFILE=%s)\n", val);
        }
    }
    return cached != 0;
}

bool inpu_debug_profile_verbose_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char * val = std::getenv("GGML_INPU_PROFILE_VERBOSE");
        cached           = (val && val[0] != '0' && val[0] != '\0') ? 1 : 0;
        if (cached) {
            INPU_LOG_INFO("verbose profiling enabled (GGML_INPU_PROFILE_VERBOSE=%s)\n", val);
        }
    }
    return cached != 0;
}

static int inpu_debug_profile_top_n() {
    static int cached = -1;
    if (cached < 0) {
        cached           = 8;
        const char * val = std::getenv("GGML_INPU_PROFILE_TOP");
        if (val && val[0] != '\0') {
            char * end    = nullptr;
            long   parsed = std::strtol(val, &end, 10);
            if (end != val && parsed > 0) {
                cached = (int) parsed;
            }
        }
    }
    return cached;
}

static double inpu_profile_us_to_ms(uint64_t us) {
    return (double) us / 1000.0;
}

static double inpu_profile_bytes_to_mib(size_t bytes) {
    return (double) bytes / (1024.0 * 1024.0);
}

static std::string inpu_profile_shape_string(const struct ggml_tensor * t) {
    if (!t) {
        return "[]";
    }

    std::ostringstream os;
    os << '[' << t->ne[0] << 'x' << t->ne[1] << 'x' << t->ne[2] << 'x' << t->ne[3] << ']';
    return os.str();
}

std::string inpu_debug_graph_label(const struct ggml_cgraph * cgraph) {
    if (!cgraph) {
        return "<null-cgraph>";
    }

    std::array<int, GGML_OP_COUNT> op_counts{};
    int                            n_compute = 0;

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const struct ggml_tensor * node = cgraph->nodes[i];
        if (!node) {
            continue;
        }
        if (node->op != GGML_OP_NONE) {
            ++n_compute;
        }
        if ((int) node->op >= 0 && (int) node->op < GGML_OP_COUNT) {
            op_counts[(int) node->op]++;
        }
    }

    std::vector<std::pair<int, std::string>> top_ops;
    top_ops.reserve(GGML_OP_COUNT);
    for (int op = 0; op < GGML_OP_COUNT; ++op) {
        if (op_counts[op] > 0) {
            top_ops.emplace_back(op_counts[op], ggml_op_name((enum ggml_op) op));
        }
    }

    std::sort(top_ops.begin(), top_ops.end(), [](const auto & a, const auto & b) {
        if (a.first != b.first) {
            return a.first > b.first;
        }
        return a.second < b.second;
    });

    std::ostringstream os;
    os << "nodes=" << cgraph->n_nodes << "/" << n_compute;

    if (cgraph->n_nodes > 0 && cgraph->nodes[cgraph->n_nodes - 1]) {
        const struct ggml_tensor * out      = cgraph->nodes[cgraph->n_nodes - 1];
        const char *               out_name = ggml_get_name(out);
        os << " out=" << inpu_profile_shape_string(out);
        if (out_name && out_name[0] != '\0') {
            os << ' ' << out_name;
        }
    }

    os << " ops=";
    const size_t n_show = std::min<size_t>(5, top_ops.size());
    for (size_t i = 0; i < n_show; ++i) {
        if (i > 0) {
            os << ',';
        }
        os << top_ops[i].second << 'x' << top_ops[i].first;
    }

    return os.str();
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

namespace {

struct inpu_profile_copy_stats {
    uint64_t calls       = 0;
    uint64_t total_us    = 0;
    size_t   total_bytes = 0;
};

struct inpu_profile_graph_stats {
    size_t      graph_hash = 0;
    std::string graph_label;
    int         n_nodes = 0;

    uint64_t calls             = 0;
    uint64_t success_calls     = 0;
    uint64_t cache_hits        = 0;
    uint64_t cache_misses      = 0;
    uint64_t request_pool_hits = 0;

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

    size_t   input_bytes    = 0;
    size_t   output_bytes   = 0;
    size_t   quant_bytes    = 0;
    size_t   scale_bytes    = 0;
    uint64_t input_tensors  = 0;
    uint64_t output_tensors = 0;
};

class inpu_profiler {
  public:
    static inpu_profiler & instance() {
        static inpu_profiler profiler;
        return profiler;
    }

    void record_graph_call(const inpu_profile_graph_call & call) {
        if (!inpu_debug_profile_enabled()) {
            return;
        }

        ensure_atexit_registered();

        std::lock_guard<std::mutex> lock(mutex_);
        auto &                      dst = graph_stats_[graph_key(call)];

        if (dst.graph_label.empty()) {
            dst.graph_hash  = call.graph_hash;
            dst.graph_label = call.graph_label;
            dst.n_nodes     = call.n_nodes;
        }

        dst.calls++;
        dst.success_calls += call.success ? 1 : 0;
        dst.cache_hits += call.cache_hit ? 1 : 0;
        dst.cache_misses += call.cache_hit ? 0 : 1;
        dst.request_pool_hits += call.request_from_pool ? 1 : 0;

        dst.key_us += call.key_us;
        dst.cache_lookup_us += call.cache_lookup_us;
        dst.analyze_io_us += call.analyze_io_us;
        dst.translate_us += call.translate_us;
        dst.compile_us += call.compile_us;
        dst.acquire_request_us += call.acquire_request_us;
        dst.bind_inputs_us += call.bind_inputs_us;
        dst.bind_outputs_us += call.bind_outputs_us;
        dst.infer_us += call.infer_us;
        dst.release_request_us += call.release_request_us;
        dst.total_us += call.total_us;

        dst.input_bytes += call.input_bytes;
        dst.output_bytes += call.output_bytes;
        dst.quant_bytes += call.quant_bytes;
        dst.scale_bytes += call.scale_bytes;
        dst.input_tensors += call.input_tensors;
        dst.output_tensors += call.output_tensors;

        if (inpu_debug_profile_verbose_enabled()) {
            INPU_LOG_INFO(
                "profile graph hash=%zx total=%.3f ms infer=%.3f ms bind_in=%.3f ms bind_out=%.3f ms compile=%.3f ms "
                "cache=%s req=%s inputs=%.2f MiB outputs=%.2f MiB %s\n",
                call.graph_hash, inpu_profile_us_to_ms(call.total_us), inpu_profile_us_to_ms(call.infer_us),
                inpu_profile_us_to_ms(call.bind_inputs_us), inpu_profile_us_to_ms(call.bind_outputs_us),
                inpu_profile_us_to_ms(call.compile_us), call.cache_hit ? "hit" : "miss",
                call.request_from_pool ? "pool" : "new", inpu_profile_bytes_to_mib(call.input_bytes),
                inpu_profile_bytes_to_mib(call.output_bytes), call.graph_label.c_str());
        }
    }

    void record_buffer_set(enum ggml_type type, size_t bytes, uint64_t elapsed_us) {
        record_copy(set_stats_, type, bytes, elapsed_us);
    }

    void record_buffer_get(enum ggml_type type, size_t bytes, uint64_t elapsed_us) {
        record_copy(get_stats_, type, bytes, elapsed_us);
    }

    void print_summary() {
        if (!inpu_debug_profile_enabled()) {
            return;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        if (graph_stats_.empty() && set_stats_.empty() && get_stats_.empty()) {
            return;
        }

        uint64_t total_calls              = 0;
        uint64_t total_success            = 0;
        uint64_t total_cache_hits         = 0;
        uint64_t total_cache_misses       = 0;
        uint64_t total_request_pool_hits  = 0;
        uint64_t total_key_us             = 0;
        uint64_t total_cache_lookup_us    = 0;
        uint64_t total_analyze_io_us      = 0;
        uint64_t total_translate_us       = 0;
        uint64_t total_compile_us         = 0;
        uint64_t total_acquire_request_us = 0;
        uint64_t total_bind_inputs_us     = 0;
        uint64_t total_bind_outputs_us    = 0;
        uint64_t total_infer_us           = 0;
        uint64_t total_release_request_us = 0;
        uint64_t total_graph_us           = 0;
        size_t   total_input_bytes        = 0;
        size_t   total_output_bytes       = 0;
        size_t   total_quant_bytes        = 0;
        size_t   total_scale_bytes        = 0;

        std::vector<const inpu_profile_graph_stats *> graphs;
        graphs.reserve(graph_stats_.size());

        for (const auto & kv : graph_stats_) {
            const auto & s = kv.second;
            graphs.push_back(&s);
            total_calls += s.calls;
            total_success += s.success_calls;
            total_cache_hits += s.cache_hits;
            total_cache_misses += s.cache_misses;
            total_request_pool_hits += s.request_pool_hits;
            total_key_us += s.key_us;
            total_cache_lookup_us += s.cache_lookup_us;
            total_analyze_io_us += s.analyze_io_us;
            total_translate_us += s.translate_us;
            total_compile_us += s.compile_us;
            total_acquire_request_us += s.acquire_request_us;
            total_bind_inputs_us += s.bind_inputs_us;
            total_bind_outputs_us += s.bind_outputs_us;
            total_infer_us += s.infer_us;
            total_release_request_us += s.release_request_us;
            total_graph_us += s.total_us;
            total_input_bytes += s.input_bytes;
            total_output_bytes += s.output_bytes;
            total_quant_bytes += s.quant_bytes;
            total_scale_bytes += s.scale_bytes;
        }

        std::sort(graphs.begin(), graphs.end(), [](const auto * a, const auto * b) {
            if (a->total_us != b->total_us) {
                return a->total_us > b->total_us;
            }
            return a->calls > b->calls;
        });

        auto append_copy_section = [](std::ostringstream & os, const char * title,
                                      const std::unordered_map<int, inpu_profile_copy_stats> & stats) {
            if (stats.empty()) {
                return;
            }

            uint64_t total_calls = 0;
            uint64_t total_us    = 0;
            size_t   total_bytes = 0;
            for (const auto & kv : stats) {
                total_calls += kv.second.calls;
                total_us += kv.second.total_us;
                total_bytes += kv.second.total_bytes;
            }

            os << title << ": calls=" << total_calls << ", bytes=" << std::fixed << std::setprecision(2)
               << inpu_profile_bytes_to_mib(total_bytes) << " MiB"
               << ", time=" << std::fixed << std::setprecision(3) << inpu_profile_us_to_ms(total_us) << " ms\n";

            std::vector<std::pair<int, inpu_profile_copy_stats>> ordered(stats.begin(), stats.end());
            std::sort(ordered.begin(), ordered.end(),
                      [](const auto & a, const auto & b) { return a.second.total_us > b.second.total_us; });

            for (const auto & kv : ordered) {
                os << "  - " << ggml_type_name((enum ggml_type) kv.first) << ": calls=" << kv.second.calls
                   << ", bytes=" << std::fixed << std::setprecision(2)
                   << inpu_profile_bytes_to_mib(kv.second.total_bytes) << " MiB"
                   << ", avg=" << std::fixed << std::setprecision(3)
                   << inpu_profile_us_to_ms(kv.second.calls ? kv.second.total_us / kv.second.calls : 0) << " ms"
                   << ", total=" << inpu_profile_us_to_ms(kv.second.total_us) << " ms\n";
            }
        };

        std::ostringstream os;
        os << "=== iNPU profile summary ===\n";
        os << "graph calls=" << total_calls << ", success=" << total_success << ", cache hit/miss=" << total_cache_hits
           << '/' << total_cache_misses << ", request-pool reuse=" << total_request_pool_hits << '/' << total_calls
           << "\n";
        os << "graph wall=" << std::fixed << std::setprecision(3) << inpu_profile_us_to_ms(total_graph_us) << " ms"
           << " | infer=" << inpu_profile_us_to_ms(total_infer_us)
           << " ms | bind_in=" << inpu_profile_us_to_ms(total_bind_inputs_us)
           << " ms | bind_out=" << inpu_profile_us_to_ms(total_bind_outputs_us)
           << " ms | compile=" << inpu_profile_us_to_ms(total_compile_us)
           << " ms | translate=" << inpu_profile_us_to_ms(total_translate_us)
           << " ms | analyze_io=" << inpu_profile_us_to_ms(total_analyze_io_us) << " ms\n";
        os << "graph setup: key=" << inpu_profile_us_to_ms(total_key_us)
           << " ms, cache_lookup=" << inpu_profile_us_to_ms(total_cache_lookup_us)
           << " ms, acquire_req=" << inpu_profile_us_to_ms(total_acquire_request_us)
           << " ms, release_req=" << inpu_profile_us_to_ms(total_release_request_us) << " ms\n";
        os << "bound bytes: inputs=" << std::fixed << std::setprecision(2)
           << inpu_profile_bytes_to_mib(total_input_bytes)
           << " MiB, outputs=" << inpu_profile_bytes_to_mib(total_output_bytes)
           << " MiB, quant=" << inpu_profile_bytes_to_mib(total_quant_bytes)
           << " MiB, scales=" << inpu_profile_bytes_to_mib(total_scale_bytes) << " MiB\n";

        append_copy_section(os, "buffer set_tensor", set_stats_);
        append_copy_section(os, "buffer get_tensor", get_stats_);

        const int top_n = std::min<int>(inpu_debug_profile_top_n(), (int) graphs.size());
        if (top_n > 0) {
            os << "top graphs by total wall time:\n";
            for (int i = 0; i < top_n; ++i) {
                const auto & s = *graphs[i];
                os << "  " << (i + 1) << ") hash=" << std::hex << s.graph_hash << std::dec << ", calls=" << s.calls
                   << ", avg_total=" << std::fixed << std::setprecision(3)
                   << inpu_profile_us_to_ms(s.calls ? s.total_us / s.calls : 0) << " ms"
                   << ", avg_infer=" << inpu_profile_us_to_ms(s.calls ? s.infer_us / s.calls : 0) << " ms"
                   << ", avg_bind_in=" << inpu_profile_us_to_ms(s.calls ? s.bind_inputs_us / s.calls : 0) << " ms"
                   << ", avg_compile=" << inpu_profile_us_to_ms(s.calls ? s.compile_us / s.calls : 0) << " ms"
                   << ", cache=" << s.cache_hits << '/' << s.calls << ", req_pool=" << s.request_pool_hits << '/'
                   << s.calls << ", input=" << std::fixed << std::setprecision(2)
                   << inpu_profile_bytes_to_mib(s.input_bytes) << " MiB"
                   << ", output=" << inpu_profile_bytes_to_mib(s.output_bytes) << " MiB"
                   << "\n     " << s.graph_label << "\n";
            }
        }

        os << "============================\n";
        INPU_LOG_INFO("%s", os.str().c_str());
    }

  private:
    void ensure_atexit_registered() {
        if (atexit_registered_) {
            return;
        }
        atexit_registered_ = true;
        std::atexit(&inpu_profiler::print_summary_atexit);
    }

    static void print_summary_atexit() { instance().print_summary(); }

    static std::string graph_key(const inpu_profile_graph_call & call) {
        std::ostringstream os;
        os << std::hex << call.graph_hash;
        return os.str();
    }

    void record_copy(std::unordered_map<int, inpu_profile_copy_stats> & dst,
                     enum ggml_type                                     type,
                     size_t                                             bytes,
                     uint64_t                                           elapsed_us) {
        if (!inpu_debug_profile_enabled()) {
            return;
        }

        ensure_atexit_registered();

        std::lock_guard<std::mutex> lock(mutex_);
        auto &                      stats = dst[(int) type];
        stats.calls++;
        stats.total_bytes += bytes;
        stats.total_us += elapsed_us;
    }

    std::mutex                                                mutex_;
    bool                                                      atexit_registered_ = false;
    std::unordered_map<std::string, inpu_profile_graph_stats> graph_stats_;
    std::unordered_map<int, inpu_profile_copy_stats>          set_stats_;
    std::unordered_map<int, inpu_profile_copy_stats>          get_stats_;
};

}  // namespace

void inpu_profile_record_graph_call(const inpu_profile_graph_call & call) {
    inpu_profiler::instance().record_graph_call(call);
}

void inpu_profile_record_buffer_set(enum ggml_type type, size_t bytes, uint64_t elapsed_us) {
    inpu_profiler::instance().record_buffer_set(type, bytes, elapsed_us);
}

void inpu_profile_record_buffer_get(enum ggml_type type, size_t bytes, uint64_t elapsed_us) {
    inpu_profiler::instance().record_buffer_get(type, bytes, elapsed_us);
}
