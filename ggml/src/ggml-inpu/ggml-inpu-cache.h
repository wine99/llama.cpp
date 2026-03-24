#pragma once

// ggml-inpu cache: compiled OV model caching

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <openvino/openvino.hpp>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Cache key: identifies a unique OV model by the shapes and types of all ops
// ---------------------------------------------------------------------------

struct inpu_graph_key {
    struct node_desc {
        int         op;           // ggml_op
        int         op_param;     // sub-op (e.g. glu_op) or 0
        int         dst_type;     // ggml_type of dst
        int64_t     dst_ne[4];
        int32_t     op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
        struct src_desc {
            int     type;         // ggml_type
            int64_t ne[4];
            bool    present;      // false if src is nullptr
        } src[GGML_MAX_SRC];
    };

    std::vector<node_desc> nodes;

    bool operator==(const inpu_graph_key & other) const;
};

struct inpu_graph_key_hash {
    size_t operator()(const inpu_graph_key & key) const;
};

// Build a cache key from a cgraph
inpu_graph_key inpu_make_graph_key(const struct ggml_cgraph * cgraph);

// ---------------------------------------------------------------------------
// Cache value: compiled model + inference request pool
// ---------------------------------------------------------------------------

struct inpu_compiled_graph {
    std::shared_ptr<ov::CompiledModel> compiled_model;

    // Pool of InferRequests for thread safety
    std::vector<ov::InferRequest> request_pool;
    std::mutex pool_mutex;

    // I/O metadata: mapping from position → OV tensor name
    // Inputs: ordered list of (ggml tensor role description, ov parameter name)
    struct io_entry {
        std::string ov_name;       // name in the OV model
        int         node_idx;      // cgraph node index this relates to
        int         bind_node_idx; // cgraph node index whose buffer is bound to OV I/O
        int         src_idx;       // src index within the node (-1 for dst/output)
        bool        is_quant;      // true if this is the quant part of a quantized weight
        bool        is_scale;      // true if this is the scale part of a quantized weight
    };
    std::vector<io_entry> input_map;
    std::vector<io_entry> output_map;

    // Acquire an InferRequest (creates one if pool is empty)
    ov::InferRequest acquire_request(bool * from_pool = nullptr);
    // Return an InferRequest to the pool
    void release_request(ov::InferRequest && req);
};

// ---------------------------------------------------------------------------
// Cache store
// ---------------------------------------------------------------------------

class inpu_cache {
public:
    // Look up or return nullptr
    std::shared_ptr<inpu_compiled_graph> find(const inpu_graph_key & key);

    // Insert a new entry
    void insert(const inpu_graph_key & key, std::shared_ptr<inpu_compiled_graph> entry);

private:
    std::unordered_map<inpu_graph_key, std::shared_ptr<inpu_compiled_graph>, inpu_graph_key_hash> cache_;
    std::mutex mutex_;
};
