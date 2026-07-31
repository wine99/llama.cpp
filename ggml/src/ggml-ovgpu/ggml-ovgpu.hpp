#pragma once

// OVGPU backend internals - shared between the backend skeleton (ggml-ovgpu.cpp)
// and the op translators (ggml-ovgpu-ops.cpp).

#include "ggml.h"
#include "ggml-impl.h"

#include <intel_gpu/graph/network.hpp>
#include <intel_gpu/runtime/engine.hpp>
#include <intel_gpu/runtime/memory.hpp>
#include <intel_gpu/runtime/stream.hpp>

#include <map>
#include <memory>
#include <string>
#include <unordered_map>

struct ggml_tensor;

namespace ggml::ovgpu {

// cldnn data type for a ggml type; returns data_types::i8 as a fallback (caller
// checks the bool for support).
ov::element::Type                  ggml_type_to_cldnn(ggml_type type);
// cldnn layout (bfyx tensor) for a ggml tensor's logical shape + type.
// ggml ne = [n0, n1, n2, n3] (fastest-first) -> cldnn tensor(batch = n1*n2*n3, feature = n0, 1, 1).
cldnn::layout                      ggml_layout_for(const ggml_tensor * t);
// wrap a ggml tensor's storage into a non-owning cldnn::memory view over the
// backend's USM base memory, at byte_offset. create_subbuffer returns a view
// (no ownership) so it does not free the underlying allocation on destruction.
cldnn::memory::ptr                 wrap_tensor(cldnn::engine & engine, cldnn::memory & base_mem,
                                               const ggml_tensor * t, size_t byte_offset);

// A compiled single-op network + the ids of its external inputs/outputs.
struct compiled_op {
    std::shared_ptr<cldnn::network> net;
    std::vector<std::string>        input_ids;  // input_layout primitive ids, in ggml src order
    std::string                     output_id;  // the op's output primitive id
    std::vector<std::string>        weight_ids; // data() primitive ids (weights), not re-bound per compute
};

// Build a single-op cldnn network for the given ggml node. Returns nullptr if
// the op/shape/dtype combo is not (yet) supported - caller reports supports_op=false.
std::unique_ptr<compiled_op>       build_op(const ggml_tensor * node, cldnn::engine & engine);

// Per-op-network cache. Keyed by a structural signature of the ggml node
// (op + dtypes + ne[] + flags) so an identical shape bucket reuses the compiled
// network across layers and across compute calls.
struct op_cache {
    cldnn::engine * engine;
    std::unordered_map<std::string, std::shared_ptr<compiled_op>> m_cache;
    std::mutex      m_mutex; // test-backend-ops runs cases in parallel threads

    static std::string key_for(const ggml_tensor * node);

    std::unique_ptr<compiled_op> get_or_build(const ggml_tensor * node);
};

}  // namespace ggml::ovgpu
