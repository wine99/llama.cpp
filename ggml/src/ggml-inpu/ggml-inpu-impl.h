#pragma once

// ggml-inpu internal header — shared types and declarations

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <openvino/openvino.hpp>

#include <cstdint>

// ---------------------------------------------------------------------------
// Tensor extra: metadata attached to weight tensors in the iNPU buffer
// ---------------------------------------------------------------------------

struct ggml_inpu_tensor_extra {
    // Pointers into tensor->data for the separated layout
    void * quants;       // packed quant data (i4 for Q4_0, i8 for Q8_0, or nullptr for F16/F32)
    void * scales;       // fp16 scale data (or nullptr for F16/F32)
    size_t quants_bytes; // byte count of quants region
    size_t scales_bytes; // byte count of scales region

    ov::element::Type quant_ov_type; // i4 for Q4_0, i8 for Q8_0
    ov::element::Type scale_ov_type; // f16

    int64_t n_rows;      // weight matrix rows
    int64_t n_cols;      // weight matrix cols
    int     group_size;  // quantization group size (32 for Q4_0/Q8_0)
    bool    groups_first_layout;  // true for [n_groups, N, gs], false for [N, n_groups, gs]

    bool is_quantized;   // true for Q4_0/Q8_0, false for F16/F32
};

// ---------------------------------------------------------------------------
// Backend context: held by the ggml_backend instance
// ---------------------------------------------------------------------------

struct ggml_inpu_context; // forward-declared, defined in ggml-inpu.cpp

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Check if a buffer type is the iNPU buffer type
bool ggml_backend_is_inpu_buffer_type(ggml_backend_buffer_type_t buft);

// Check if a tensor is in an iNPU buffer
static inline bool ggml_inpu_tensor_in_inpu_buffer(const struct ggml_tensor * t) {
    return t && t->buffer && ggml_backend_is_inpu_buffer_type(t->buffer->buft);
}


