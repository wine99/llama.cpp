// ggml-inpu-cache.cpp — compiled OV model caching for iNPU backend

#include "ggml-inpu-cache.h"
#include "ggml-impl.h"

#include <cstring>
#include <functional>

// ============================================================================
// Graph key
// ============================================================================

bool inpu_graph_key::operator==(const inpu_graph_key & other) const {
    if (nodes.size() != other.nodes.size()) return false;

    for (size_t i = 0; i < nodes.size(); i++) {
        const auto & a = nodes[i];
        const auto & b = other.nodes[i];

        if (a.op != b.op) return false;
        if (a.op_param != b.op_param) return false;
        if (a.dst_type != b.dst_type) return false;

        for (int d = 0; d < 4; d++) {
            if (a.dst_ne[d] != b.dst_ne[d]) return false;
        }

        if (std::memcmp(a.op_params, b.op_params, sizeof(a.op_params)) != 0) return false;

        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (a.src[s].present != b.src[s].present) return false;
            if (!a.src[s].present) continue;
            if (a.src[s].type != b.src[s].type) return false;
            for (int d = 0; d < 4; d++) {
                if (a.src[s].ne[d] != b.src[s].ne[d]) return false;
            }
        }
    }
    return true;
}

size_t inpu_graph_key_hash::operator()(const inpu_graph_key & key) const {
    // Boost-style hash_combine
    auto hash_combine = [](size_t & seed, size_t v) {
        seed ^= v + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    };

    std::hash<int> hi;
    std::hash<int64_t> hi64;

    size_t h = 0;
    hash_combine(h, key.nodes.size());

    for (const auto & node : key.nodes) {
        hash_combine(h, hi(node.op));
        hash_combine(h, hi(node.op_param));
        hash_combine(h, hi(node.dst_type));
        for (int d = 0; d < 4; d++) {
            hash_combine(h, hi64(node.dst_ne[d]));
        }
        for (int p = 0; p < (int)(sizeof(node.op_params) / sizeof(int32_t)); p++) {
            hash_combine(h, hi(node.op_params[p]));
        }
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            hash_combine(h, (size_t)node.src[s].present);
            if (node.src[s].present) {
                hash_combine(h, hi(node.src[s].type));
                for (int d = 0; d < 4; d++) {
                    hash_combine(h, hi64(node.src[s].ne[d]));
                }
            }
        }
    }
    return h;
}

inpu_graph_key inpu_make_graph_key(const struct ggml_cgraph * cgraph) {
    inpu_graph_key key;
    key.nodes.resize(cgraph->n_nodes);

    for (int i = 0; i < cgraph->n_nodes; i++) {
        const struct ggml_tensor * node = cgraph->nodes[i];
        auto & desc = key.nodes[i];

        desc.op = node->op;

        // For GLU, include the sub-op
        if (node->op == GGML_OP_GLU) {
            desc.op_param = ggml_get_glu_op(node);
        } else {
            desc.op_param = 0;
        }

        desc.dst_type = node->type;
        for (int d = 0; d < 4; d++) {
            desc.dst_ne[d] = node->ne[d];
        }

        static_assert(sizeof(desc.op_params) == sizeof(node->op_params), "op_params size mismatch");
        std::memcpy(desc.op_params, node->op_params, sizeof(desc.op_params));

        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (node->src[s]) {
                desc.src[s].present = true;
                desc.src[s].type    = node->src[s]->type;
                for (int d = 0; d < 4; d++) {
                    desc.src[s].ne[d] = node->src[s]->ne[d];
                }
            } else {
                desc.src[s].present = false;
                desc.src[s].type    = 0;
                for (int d = 0; d < 4; d++) {
                    desc.src[s].ne[d] = 0;
                }
            }
        }
    }
    return key;
}

// ============================================================================
// Compiled graph — InferRequest pool
// ============================================================================

ov::InferRequest inpu_compiled_graph::acquire_request() {
    std::lock_guard<std::mutex> lock(pool_mutex);
    if (!request_pool.empty()) {
        auto req = std::move(request_pool.back());
        request_pool.pop_back();
        return req;
    }
    // Create new
    return compiled_model->create_infer_request();
}

void inpu_compiled_graph::release_request(ov::InferRequest && req) {
    std::lock_guard<std::mutex> lock(pool_mutex);
    request_pool.push_back(std::move(req));
}

// ============================================================================
// Cache store
// ============================================================================

std::shared_ptr<inpu_compiled_graph> inpu_cache::find(const inpu_graph_key & key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        return it->second;
    }
    return nullptr;
}

void inpu_cache::insert(const inpu_graph_key & key, std::shared_ptr<inpu_compiled_graph> entry) {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_[key] = std::move(entry);
}
