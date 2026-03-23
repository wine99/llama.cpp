// ggml-inpu.cpp — Intel NPU backend for llama.cpp
//
// Accelerates MUL_MAT (+ optional fused ADD/GLU) via OpenVINO on Intel NPUs.
// Registers as an ACCEL backend with a custom buffer type for weight preparation.

#include "ggml-inpu.h"
#include "ggml-inpu-impl.h"
#include "ggml-inpu-cache.h"
#include "ggml-inpu-debug.h"
#include "ggml-inpu-translate.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml.h"

#include <openvino/openvino.hpp>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// ============================================================================
// Logging
// ============================================================================

#define INPU_LOG_DEBUG(...) GGML_LOG_DEBUG("INPU: " __VA_ARGS__)
#define INPU_LOG_INFO(...)  GGML_LOG_INFO("INPU: " __VA_ARGS__)
#define INPU_LOG_WARN(...)  GGML_LOG_WARN("INPU: " __VA_ARGS__)
#define INPU_LOG_ERROR(...) GGML_LOG_ERROR("INPU: " __VA_ARGS__)

// ============================================================================
// Backend context
// ============================================================================

struct ggml_inpu_context {
    ov::Core core;
    inpu_cache cache;
    std::string device_name; // "NPU"
};

// ============================================================================
// Buffer type implementation
// ============================================================================

// Tag used to identify the iNPU buffer type
static const char * INPU_BUFT_NAME = "iNPU";

static const char * ggml_backend_inpu_buft_get_name(ggml_backend_buffer_type_t buft) {
    return INPU_BUFT_NAME;
    GGML_UNUSED(buft);
}

// We use host memory, but the data layout for quantized tensors differs from
// the standard ggml block format (separated quants+scales in planar layout).
// Return false so that tensor copies go through get_tensor/set_tensor which
// properly convert between the separated and block layouts.
static bool ggml_backend_inpu_buft_is_host(ggml_backend_buffer_type_t buft) {
    return false;
    GGML_UNUSED(buft);
}

static size_t ggml_backend_inpu_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    return 64; // cache-line aligned
    GGML_UNUSED(buft);
}

static size_t ggml_backend_inpu_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    // Separated layout has the same byte count as block format.
    return ggml_nbytes(tensor);
    GGML_UNUSED(buft);
}

// ---------------------------------------------------------------------------
// Buffer context: manages a host-memory allocation
// ---------------------------------------------------------------------------

struct ggml_backend_inpu_buffer_context {
    void * data;
    size_t size;
    // Tensor extras stored here to manage lifetime
    std::unordered_map<const struct ggml_tensor *, std::unique_ptr<ggml_inpu_tensor_extra>> extras;
    std::mutex extras_mutex;
};

static void ggml_backend_inpu_buffer_free(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<ggml_backend_inpu_buffer_context *>(buffer->context);
    free(ctx->data);
    delete ctx;
}

static void * ggml_backend_inpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<ggml_backend_inpu_buffer_context *>(buffer->context);
    return ctx->data;
}

static enum ggml_status ggml_backend_inpu_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    // No pre-allocation of extras here; they are created in set_tensor
    // when the actual weight data arrives.
    GGML_UNUSED(buffer);
    GGML_UNUSED(tensor);
    return GGML_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Q4_0 weight reformatting: block → OV separated layout (i4 quants + fp16 scales)
//
// block_q4_0 layout per block (18 bytes):
//   ggml_half d;           // scale (2 bytes)
//   uint8_t   qs[16];      // 32 nibbles packed into 16 bytes (low nib = elem 2i, high nib = elem 2i+1)
//
// OV expects a different nibble ordering (from unpack_32_4 in reference).
// After repacking to OV layout, we XOR each byte with 0x88 to convert u4→i4
// (subtracts 8 from each nibble), so the dequant formula is just: i4 * scale.
//
// Separated planar layout (groups-first / permuted for NPU-optimal dequant):
//   i4 quants: n_blocks_per_row * n_rows * 16 bytes  [n_groups, N, gs/2]
//   fp16 scales: n_blocks_per_row * n_rows * 2 bytes  [n_groups, N]
//
// The NPU's best dequant pattern expects the group dimension first:
//   [n_groups, N, gs] (i4) → Convert → f16 → Multiply → Transpose(1,0,2) → Reshape → [N, K]
//   [n_groups, N, 1]  (f16 scales) --------------------------->
// ---------------------------------------------------------------------------

// Repack 16 bytes (32 u4 nibbles) from ggml layout to OV u4 layout
static void unpack_32_4(const uint8_t * data, uint8_t * dst) {
    memset(dst, 0, 16);
    for (int j = 0; j < 16; ++j) {
        uint8_t x = (data[j] & 0x0F);       // low nibble
        uint8_t y = (data[j] >> 4);          // high nibble
        if (j % 2 != 0) {
            x <<= 4;
            y <<= 4;
        }
        dst[j / 2]     |= x;
        dst[8 + j / 2] |= y;
    }
}

// Reverse: repack 16 bytes from OV u4 layout back to ggml layout
static void pack_32_4(const uint8_t * src, uint8_t * dst) {
    memset(dst, 0, 16);
    for (int j = 0; j < 16; ++j) {
        uint8_t x, y;
        if (j % 2 == 0) {
            x = src[j / 2] & 0x0F;
            y = src[8 + j / 2] & 0x0F;
        } else {
            x = src[j / 2] >> 4;
            y = src[8 + j / 2] >> 4;
        }
        dst[j] = x | (y << 4);
    }
}

static void inpu_reformat_q4_0_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0);

    const int64_t n_rows = ggml_nrows(tensor);
    const int64_t n_cols = tensor->ne[0];
    const int64_t n_blocks_per_row = n_cols / 32;

    // Planar layout sizes
    const size_t total_quants_bytes = (size_t)n_rows * ((size_t)n_cols / 2);

    // Block layout sizes
    const size_t block_size = sizeof(uint16_t) + 16; // 18 bytes per block
    const size_t row_bytes_blk = (size_t)n_blocks_per_row * block_size;

    const int64_t max_rows = (int64_t)(size / row_bytes_blk);
    const int64_t rows_to_convert = std::min(n_rows, max_rows);

    uint8_t * dst_quants = static_cast<uint8_t *>(tensor->data);
    uint8_t * dst_scales = dst_quants + total_quants_bytes;

    // Groups-first layout: quants stored as [n_groups, n_rows, 16]
    //                        scales stored as [n_groups, n_rows]
    const size_t quant_group_bytes = 16; // 32 i4 nibbles = 16 bytes per group

    for (int64_t row = 0; row < rows_to_convert; row++) {
        const uint8_t * src_row = static_cast<const uint8_t *>(data) + row * row_bytes_blk;

        for (int64_t b = 0; b < n_blocks_per_row; b++) {
            const uint8_t * block = src_row + b * block_size;
            // Extract fp16 scale → dst[b * n_rows + row] (groups-first)
            memcpy(dst_scales + (b * n_rows + row) * sizeof(uint16_t), block, sizeof(uint16_t));
            // Repack u4 nibbles from ggml → OV layout, then XOR 0x88 to convert u4→i4
            uint8_t tmp[16];
            unpack_32_4(block + sizeof(uint16_t), tmp);
            uint8_t * dst_q = dst_quants + (b * n_rows + row) * quant_group_bytes;
            for (int i = 0; i < 16; i++) {
                dst_q[i] = tmp[i] ^ 0x88;
            }
        }
    }
}

static void inpu_reformat_q4_0_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0);

    const int64_t n_rows = ggml_nrows(tensor);
    const int64_t n_cols = tensor->ne[0];
    const int64_t n_blocks_per_row = n_cols / 32;

    const size_t total_quants_bytes = (size_t)n_rows * ((size_t)n_cols / 2);

    const size_t quant_group_bytes = 16; // 32 i4 nibbles = 16 bytes per group

    const size_t block_size = sizeof(uint16_t) + 16;
    const size_t row_bytes_blk = (size_t)n_blocks_per_row * block_size;

    const int64_t max_rows = (int64_t)(size / row_bytes_blk);
    const int64_t rows_to_convert = std::min(n_rows, max_rows);

    const uint8_t * src_quants = static_cast<const uint8_t *>(tensor->data);
    const uint8_t * src_scales = src_quants + total_quants_bytes;

    for (int64_t row = 0; row < rows_to_convert; row++) {
        uint8_t * dst_row = static_cast<uint8_t *>(data) + row * row_bytes_blk;

        for (int64_t b = 0; b < n_blocks_per_row; b++) {
            uint8_t * block = dst_row + b * block_size;
            // Restore scale from groups-first layout: src[b * n_rows + row]
            memcpy(block, src_scales + (b * n_rows + row) * sizeof(uint16_t), sizeof(uint16_t));
            // Reverse: XOR 0x88 to convert i4→u4, then repack OV → ggml nibble layout
            uint8_t tmp[16];
            const uint8_t * src_q = src_quants + (b * n_rows + row) * quant_group_bytes;
            for (int i = 0; i < 16; i++) {
                tmp[i] = src_q[i] ^ 0x88;
            }
            pack_32_4(tmp, block + sizeof(uint16_t));
        }
    }
}

// ---------------------------------------------------------------------------
// Q8_0 weight reformatting: block → OV separated layout (i8 quants + fp16 scales)
//
// block_q8_0 layout per block (34 bytes):
//   ggml_half d;           // scale (2 bytes)
//   int8_t    qs[32];      // 32 int8 quants
//
// ggml Q8_0 already stores signed int8, so no conversion needed.
// Just separate into planar layout: all i8 quants first, then all fp16 scales.
//
// Separated planar layout (groups-first / permuted for NPU-optimal dequant):
//   i8 quants: n_blocks_per_row * n_rows * 32 bytes  [n_groups, N, gs]
//   fp16 scales: n_blocks_per_row * n_rows * 2 bytes  [n_groups, N]
// ---------------------------------------------------------------------------

static void inpu_reformat_q8_0_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0);

    const int64_t n_rows = ggml_nrows(tensor);
    const int64_t n_cols = tensor->ne[0];
    const int64_t n_blocks_per_row = n_cols / 32;

    const size_t total_quants_bytes = (size_t)n_rows * (size_t)n_cols;

    const size_t block_size = sizeof(uint16_t) + 32; // 34 bytes per block
    const size_t row_bytes_blk = (size_t)n_blocks_per_row * block_size;

    const int64_t max_rows = (int64_t)(size / row_bytes_blk);
    const int64_t rows_to_convert = std::min(n_rows, max_rows);

    uint8_t * dst_quants = static_cast<uint8_t *>(tensor->data);
    uint8_t * dst_scales = dst_quants + total_quants_bytes;

    // Groups-first layout: quants stored as [n_groups, n_rows, 32]
    //                        scales stored as [n_groups, n_rows]
    for (int64_t row = 0; row < rows_to_convert; row++) {
        const uint8_t * src_row = static_cast<const uint8_t *>(data) + row * row_bytes_blk;

        for (int64_t b = 0; b < n_blocks_per_row; b++) {
            const uint8_t * block = src_row + b * block_size;
            // Extract fp16 scale → dst[b * n_rows + row] (groups-first)
            memcpy(dst_scales + (b * n_rows + row) * sizeof(uint16_t), block, sizeof(uint16_t));
            // Copy int8 quants directly (already signed) → groups-first
            const uint8_t * src_qs = block + sizeof(uint16_t);
            uint8_t * dst_q = dst_quants + (b * n_rows + row) * 32;
            memcpy(dst_q, src_qs, 32);
        }
    }
}

static void inpu_reformat_q8_0_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0);

    const int64_t n_rows = ggml_nrows(tensor);
    const int64_t n_cols = tensor->ne[0];
    const int64_t n_blocks_per_row = n_cols / 32;

    const size_t total_quants_bytes = (size_t)n_rows * (size_t)n_cols;

    const size_t block_size = sizeof(uint16_t) + 32;
    const size_t row_bytes_blk = (size_t)n_blocks_per_row * block_size;

    const int64_t max_rows = (int64_t)(size / row_bytes_blk);
    const int64_t rows_to_convert = std::min(n_rows, max_rows);

    const uint8_t * src_quants = static_cast<const uint8_t *>(tensor->data);
    const uint8_t * src_scales = src_quants + total_quants_bytes;

    for (int64_t row = 0; row < rows_to_convert; row++) {
        uint8_t * dst_row = static_cast<uint8_t *>(data) + row * row_bytes_blk;

        for (int64_t b = 0; b < n_blocks_per_row; b++) {
            uint8_t * block = dst_row + b * block_size;
            // Restore scale from groups-first layout: src[b * n_rows + row]
            memcpy(block, src_scales + (b * n_rows + row) * sizeof(uint16_t), sizeof(uint16_t));
            // Copy int8 quants directly from groups-first layout
            memcpy(block + sizeof(uint16_t), src_quants + (b * n_rows + row) * 32, 32);
        }
    }
}

// ---------------------------------------------------------------------------
// Buffer iface: set_tensor with reformatting
// ---------------------------------------------------------------------------

static void ggml_backend_inpu_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                struct ggml_tensor * tensor,
                                                const void * data,
                                                size_t offset,
                                                size_t size) {
    auto * buf_ctx = static_cast<ggml_backend_inpu_buffer_context *>(buffer->context);

    switch (tensor->type) {
        case GGML_TYPE_Q4_0: {
            inpu_reformat_q4_0_set(tensor, data, offset, size);

            const int64_t n_rows = ggml_nrows(tensor);
            const int64_t n_cols = tensor->ne[0];
            const int64_t n_blocks_per_row = n_cols / 32;
            const size_t quants_bytes = (size_t)n_rows * ((size_t)n_cols / 2);
            const size_t scales_bytes = (size_t)n_rows * (size_t)n_blocks_per_row * sizeof(uint16_t);

            auto extra = std::make_unique<ggml_inpu_tensor_extra>();
            extra->quants       = tensor->data;
            extra->scales       = static_cast<uint8_t *>(tensor->data) + quants_bytes;
            extra->quants_bytes = quants_bytes;
            extra->scales_bytes = scales_bytes;
            extra->quant_ov_type = ov::element::i4;
            extra->scale_ov_type = ov::element::f16;
            extra->n_rows       = n_rows;
            extra->n_cols       = n_cols;
            extra->group_size   = 32;
            extra->is_quantized = true;

            tensor->extra = extra.get();
            {
                std::lock_guard<std::mutex> lock(buf_ctx->extras_mutex);
                buf_ctx->extras[tensor] = std::move(extra);
            }
            break;
        }

        case GGML_TYPE_Q8_0: {
            inpu_reformat_q8_0_set(tensor, data, offset, size);

            const int64_t n_rows = ggml_nrows(tensor);
            const int64_t n_cols = tensor->ne[0];
            const int64_t n_blocks_per_row = n_cols / 32;
            const size_t quants_bytes = (size_t)n_rows * (size_t)n_cols;
            const size_t scales_bytes = (size_t)n_rows * (size_t)n_blocks_per_row * sizeof(uint16_t);

            auto extra = std::make_unique<ggml_inpu_tensor_extra>();
            extra->quants       = tensor->data;
            extra->scales       = static_cast<uint8_t *>(tensor->data) + quants_bytes;
            extra->quants_bytes = quants_bytes;
            extra->scales_bytes = scales_bytes;
            extra->quant_ov_type = ov::element::i8;
            extra->scale_ov_type = ov::element::f16;
            extra->n_rows       = n_rows;
            extra->n_cols       = n_cols;
            extra->group_size   = 32;
            extra->is_quantized = true;

            tensor->extra = extra.get();
            {
                std::lock_guard<std::mutex> lock(buf_ctx->extras_mutex);
                buf_ctx->extras[tensor] = std::move(extra);
            }
            break;
        }

        default:
            // F16, F32, etc.: plain memcpy
            memcpy(static_cast<char *>(tensor->data) + offset, data, size);
            break;
    }
}

static void ggml_backend_inpu_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                const struct ggml_tensor * tensor,
                                                void * data,
                                                size_t offset,
                                                size_t size) {
    GGML_UNUSED(buffer);

    switch (tensor->type) {
        case GGML_TYPE_Q4_0:
            inpu_reformat_q4_0_get(tensor, data, offset, size);
            break;

        case GGML_TYPE_Q8_0:
            inpu_reformat_q8_0_get(tensor, data, offset, size);
            break;

        default:
            memcpy(data, static_cast<const char *>(tensor->data) + offset, size);
            break;
    }
}

static void ggml_backend_inpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = static_cast<ggml_backend_inpu_buffer_context *>(buffer->context);
    memset(ctx->data, value, ctx->size);
}

static void ggml_backend_inpu_buffer_reset(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<ggml_backend_inpu_buffer_context *>(buffer->context);
    std::lock_guard<std::mutex> lock(ctx->extras_mutex);
    ctx->extras.clear();
}

// Buffer interface
static struct ggml_backend_buffer_i ggml_backend_inpu_buffer_i = {
    /* .free_buffer   = */ ggml_backend_inpu_buffer_free,
    /* .get_base      = */ ggml_backend_inpu_buffer_get_base,
    /* .init_tensor   = */ ggml_backend_inpu_buffer_init_tensor,
    /* .memset_tensor = */ nullptr,
    /* .set_tensor    = */ ggml_backend_inpu_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_inpu_buffer_get_tensor,
    /* .cpy_tensor    = */ nullptr,
    /* .clear         = */ ggml_backend_inpu_buffer_clear,
    /* .reset         = */ ggml_backend_inpu_buffer_reset,
};

// ---------------------------------------------------------------------------
// Buffer type: alloc_buffer
// ---------------------------------------------------------------------------

static ggml_backend_buffer_t ggml_backend_inpu_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = nullptr;
    if (size > 0) {
        const size_t alignment = 64;
        size = (size + alignment - 1) & ~(alignment - 1);
        data = aligned_alloc(alignment, size);
        if (!data) {
            INPU_LOG_ERROR("failed to allocate %zu bytes for iNPU buffer\n", size);
            return nullptr;
        }
    }

    auto * buf_ctx = new ggml_backend_inpu_buffer_context{data, size, {}, {}};

    return ggml_backend_buffer_init(buft, ggml_backend_inpu_buffer_i, buf_ctx, size);
}

// Buffer type interface
static struct ggml_backend_buffer_type_i ggml_backend_inpu_buft_i = {
    /* .get_name      = */ ggml_backend_inpu_buft_get_name,
    /* .alloc_buffer  = */ ggml_backend_inpu_buft_alloc_buffer,
    /* .get_alignment = */ ggml_backend_inpu_buft_get_alignment,
    /* .get_max_size  = */ nullptr,
    /* .get_alloc_size= */ ggml_backend_inpu_buft_get_alloc_size,
    /* .is_host       = */ ggml_backend_inpu_buft_is_host,
};

static ggml_backend_buffer_type_t ggml_backend_inpu_buffer_type(void) {
    static struct ggml_backend_buffer_type buft = {
        /* .iface   = */ ggml_backend_inpu_buft_i,
        /* .device  = */ nullptr, // set in device init
        /* .context = */ nullptr,
    };
    return &buft;
}

bool ggml_backend_is_inpu_buffer_type(ggml_backend_buffer_type_t buft) {
    return buft == ggml_backend_inpu_buffer_type();
}

// ============================================================================
// Backend instance (stream)
// ============================================================================

static const char * ggml_backend_inpu_get_name(ggml_backend_t backend) {
    return "iNPU";
    GGML_UNUSED(backend);
}

static void ggml_backend_inpu_free(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_inpu_context *>(backend->context);
    delete ctx;
}

static enum ggml_status ggml_backend_inpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto * ctx = static_cast<ggml_inpu_context *>(backend->context);

    auto resolve_input_binding_tensor = [](const struct ggml_tensor * t) -> const struct ggml_tensor * {
        while (t && t->src[0]) {
            switch (t->op) {
                case GGML_OP_VIEW:
                case GGML_OP_RESHAPE:
                case GGML_OP_PERMUTE:
                case GGML_OP_TRANSPOSE:
                    t = t->src[0];
                    break;
                default:
                    return t;
            }
        }

        return t;
    };

    if (cgraph->n_nodes == 0) {
        return GGML_STATUS_SUCCESS;
    }

    // Check whether there is at least one real compute node (skip NONE)
    bool has_compute = false;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (cgraph->nodes[i]->op != GGML_OP_NONE) {
            has_compute = true;
            break;
        }
    }

    if (!has_compute) {
        return GGML_STATUS_SUCCESS;
    }

    // 1. Make cache key
    inpu_graph_key key = inpu_make_graph_key(cgraph);

    // 2. Look up cache
    auto compiled = ctx->cache.find(key);

    if (!compiled) {
        // Debug: dump cgraph if requested via GGML_INPU_DUMP_CGRAPH=1
        if (inpu_debug_dump_cgraph_enabled()) {
            inpu_debug_dump_cgraph(cgraph);
        }

        // 3. Analyze I/O
        inpu_graph_io io = inpu_analyze_graph_io(cgraph);

        // 4. Translate to OV model
        inpu_translate_result result = inpu_translate_graph(cgraph, io);

        if (!result.model) {
            INPU_LOG_ERROR("failed to translate cgraph to OV model\n");
            return GGML_STATUS_FAILED;
        }

        // Debug: dump OV IR if requested via GGML_INPU_DUMP_IR=<dir>
        inpu_debug_dump_ir(result.model);

        // 5. Compile for NPU
        try {
            static ov::AnyMap config = {
                {"NPU_COMPILER_DYNAMIC_QUANTIZATION", "YES" },
                {"NPU_USE_NPUW",                      "YES" },
                {"NPUW_ONLINE_PIPELINE",              "NONE"},
            };
            auto cm = ctx->core.compile_model(result.model, ctx->device_name, config);

            compiled = std::make_shared<inpu_compiled_graph>();
            compiled->compiled_model = std::make_shared<ov::CompiledModel>(std::move(cm));
            compiled->input_map = std::move(result.input_map);
            compiled->output_map = std::move(result.output_map);

            ctx->cache.insert(key, compiled);

            INPU_LOG_DEBUG("compiled new OV model for NPU (%d nodes)\n", cgraph->n_nodes);
        } catch (const std::exception & e) {
            INPU_LOG_ERROR("OV compile failed: %s\n", e.what());
            return GGML_STATUS_FAILED;
        }
    }

    // 6. Acquire an InferRequest
    ov::InferRequest infer_req = compiled->acquire_request();

    auto ov_shape_nelements = [](const ov::Shape & shape) -> size_t {
        size_t n = 1;
        for (size_t dim : shape) {
            n *= dim;
        }
        return n;
    };

    try {
        // 7. Bind inputs
        for (const auto & entry : compiled->input_map) {
            const struct ggml_tensor * node = cgraph->nodes[entry.node_idx >= 0 ? entry.node_idx : 0];
            const struct ggml_tensor * src_tensor = nullptr;

            if (entry.src_idx >= 0 && entry.src_idx < GGML_MAX_SRC) {
                src_tensor = node->src[entry.src_idx];
            }

            if (!src_tensor) {
                INPU_LOG_ERROR("null src tensor for input '%s'\n", entry.ov_name.c_str());
                compiled->release_request(std::move(infer_req));
                return GGML_STATUS_FAILED;
            }

            const struct ggml_tensor * bind_tensor = resolve_input_binding_tensor(src_tensor);
            const void * data_ptr = bind_tensor->data;

            if (entry.is_quant || entry.is_scale) {
                // Quantized weight: use the separated layout from tensor->extra
                auto * extra = static_cast<ggml_inpu_tensor_extra *>(src_tensor->extra);
                if (!extra) {
                    INPU_LOG_ERROR("no tensor extra for quantized input '%s'\n", entry.ov_name.c_str());
                    compiled->release_request(std::move(infer_req));
                    return GGML_STATUS_FAILED;
                }

                if (entry.is_quant) {
                    ov::Shape shape = infer_req.get_tensor(entry.ov_name).get_shape();
                    auto t = ov::Tensor(extra->quant_ov_type, shape, extra->quants);
                    infer_req.set_tensor(entry.ov_name, t);
                } else {
                    ov::Shape shape = infer_req.get_tensor(entry.ov_name).get_shape();
                    auto t = ov::Tensor(extra->scale_ov_type, shape, extra->scales);
                    infer_req.set_tensor(entry.ov_name, t);
                }
            } else {
                // Non-quantized input: wrap the data pointer directly
                ov::Shape shape = infer_req.get_tensor(entry.ov_name).get_shape();
                ov::element::Type ov_type;
                switch (bind_tensor->type) {
                    case GGML_TYPE_F32: ov_type = ov::element::f32; break;
                    case GGML_TYPE_F16: ov_type = ov::element::f16; break;
                    case GGML_TYPE_I32: ov_type = ov::element::i32; break;
                    default:
                        INPU_LOG_ERROR("unsupported input type %d for '%s'\n", bind_tensor->type, entry.ov_name.c_str());
                        compiled->release_request(std::move(infer_req));
                        return GGML_STATUS_FAILED;
                }
                auto t = ov::Tensor(ov_type, shape, const_cast<void *>(data_ptr));
                infer_req.set_tensor(entry.ov_name, t);
            }
        }

        // 8. Bind outputs
        for (const auto & entry : compiled->output_map) {
            struct ggml_tensor * node = cgraph->nodes[entry.bind_node_idx];

            ov::Shape shape = infer_req.get_tensor(entry.ov_name).get_shape();
            const size_t ov_nelements = ov_shape_nelements(shape);
            const size_t ggml_nelems = (size_t) ggml_nelements(node);
            if (ov_nelements != ggml_nelems) {
                INPU_LOG_ERROR("output shape mismatch for '%s': OV has %zu elements, ggml tensor has %zu\n",
                               entry.ov_name.c_str(), ov_nelements, ggml_nelems);
                compiled->release_request(std::move(infer_req));
                return GGML_STATUS_FAILED;
            }

            ov::element::Type ov_type;
            switch (node->type) {
                case GGML_TYPE_F32: ov_type = ov::element::f32; break;
                case GGML_TYPE_F16: ov_type = ov::element::f16; break;
                default:
                    INPU_LOG_ERROR("unsupported output type %d for '%s'\n", node->type, entry.ov_name.c_str());
                    compiled->release_request(std::move(infer_req));
                    return GGML_STATUS_FAILED;
            }
            auto t = ov::Tensor(ov_type, shape, node->data);
            infer_req.set_tensor(entry.ov_name, t);
        }

        // 9. Run inference
        infer_req.infer();

    } catch (const std::exception & e) {
        INPU_LOG_ERROR("OV inference failed: %s\n", e.what());
        compiled->release_request(std::move(infer_req));
        return GGML_STATUS_FAILED;
    }

    // 10. Return the InferRequest to the pool
    compiled->release_request(std::move(infer_req));

    return GGML_STATUS_SUCCESS;
}

// Backend interface
static struct ggml_backend_i ggml_backend_inpu_i = {
    /* .get_name                = */ ggml_backend_inpu_get_name,
    /* .free                    = */ ggml_backend_inpu_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ nullptr,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_inpu_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

static ggml_guid_t ggml_backend_inpu_guid(void) {
    static ggml_guid guid = {
        0x69, 0x4e, 0x50, 0x55, // "iNPU"
        0xde, 0xad, 0xbe, 0xef,
        0x01, 0x02, 0x03, 0x04,
        0x05, 0x06, 0x07, 0x08
    };
    return &guid;
}

static ggml_backend_t ggml_backend_inpu_init(void) {
    auto * ctx = new ggml_inpu_context();
    ctx->device_name = "NPU";

    // Verify NPU is available
    try {
        auto devices = ctx->core.get_available_devices();
        bool found_npu = false;
        for (const auto & dev : devices) {
            if (dev.find("NPU") != std::string::npos) {
                found_npu = true;
                ctx->device_name = dev;
                break;
            }
        }
        if (!found_npu) {
            INPU_LOG_WARN("no NPU device found in OpenVINO, available devices:");
            for (const auto & dev : devices) {
                INPU_LOG_WARN("  %s", dev.c_str());
            }
            // Still create the backend — it will fail on compile_model
        }
    } catch (const std::exception & e) {
        INPU_LOG_ERROR("failed to enumerate OV devices: %s\n", e.what());
    }

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_inpu_guid(),
        /* .iface   = */ ggml_backend_inpu_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_inpu_reg(), 0),
        /* .context = */ ctx,
    };

    INPU_LOG_INFO("initialized iNPU backend (device: %s)\n", ctx->device_name.c_str());
    return backend;
}

// ============================================================================
// Device interface
// ============================================================================

static const char * ggml_backend_inpu_device_get_name(ggml_backend_dev_t dev) {
    return "iNPU";
    GGML_UNUSED(dev);
}

static const char * ggml_backend_inpu_device_get_description(ggml_backend_dev_t dev) {
    return "Intel NPU via OpenVINO";
    GGML_UNUSED(dev);
}

static void ggml_backend_inpu_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    *free = 0;
    *total = 0;
    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_inpu_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
    GGML_UNUSED(dev);
}

static void ggml_backend_inpu_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_inpu_device_get_name(dev);
    props->description = ggml_backend_inpu_device_get_description(dev);
    props->type        = ggml_backend_inpu_device_get_type(dev);
    ggml_backend_inpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_inpu_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_inpu_init();
    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_inpu_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_buffer_type_t buft = ggml_backend_inpu_buffer_type();
    buft->device = dev;
    return buft;
    GGML_UNUSED(dev);
}

static bool ggml_inpu_has_npu_broadcast_shape(const struct ggml_tensor * dst, const struct ggml_tensor * src) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (src->ne[i] != dst->ne[i] && src->ne[i] != 1) {
            return false;
        }
    }

    return true;
}

static bool ggml_inpu_has_same_shape(const struct ggml_tensor * a, const struct ggml_tensor * b) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (a->ne[i] != b->ne[i]) {
            return false;
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// supports_op: determine which ops the iNPU backend can handle
// ---------------------------------------------------------------------------

static bool ggml_backend_inpu_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);

    switch (op->op) {
        case GGML_OP_NONE:
            return true;

        case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_VIEW:
            return op->src[0] && ggml_inpu_tensor_in_inpu_buffer(op->src[0]);

        case GGML_OP_RMS_NORM: {
            // TODO correct but no perf gain
            return false;
            const struct ggml_tensor * src0 = op->src[0];
            // Only support F32/F16 inputs, and not in-place
            if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) return false;
            if (ggml_impl_is_view(op)) return false;
            return true;
        }

        case GGML_OP_ROPE: {
            // TODO correct but no perf gain
            return false;
            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];
            // Only F32/F16 data, I32 positions
            if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) return false;
            if (src1->type != GGML_TYPE_I32) return false;
            // Only support NORMAL and NEOX modes
            const int mode = ((const int32_t *) op->op_params)[2];
            if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) return false;
            // Reject in-place
            if (ggml_impl_is_view(op)) return false;
            return true;
        }

        case GGML_OP_MUL_MAT: {
            const struct ggml_tensor * src0 = op->src[0]; // weights
            // const struct ggml_tensor * src1 = op->src[1]; // activations

            // Weight must be a supported quantized/float type
            if (src0->type != GGML_TYPE_Q4_0 &&
                src0->type != GGML_TYPE_Q8_0 &&
                src0->type != GGML_TYPE_F16) return false;

            const bool quantized_weight =
                src0->type == GGML_TYPE_Q4_0 ||
                src0->type == GGML_TYPE_Q8_0;

            // During model loading, `select_weight_buft()` probes `supports_op()`
            // with a dummy zero-sized buffer before `set_tensor()` has run, so
            // `tensor->extra` is not available yet. Requiring the iNPU buffer
            // type here is enough for buffer selection; the runtime offload path
            // checks `tensor->extra` before execution.
            if (quantized_weight && !ggml_inpu_tensor_in_inpu_buffer(src0)) {
                return false;
            }

            if (op->src[0]->ne[2] != op->src[1]->ne[2]) return false;
            if (op->src[0]->ne[3] != op->src[1]->ne[3]) return false;
            if (op->src[0]->ne[2] > 128) return false;
            if (op->src[0]->ne[3] > 8) return false;

            return true;
        }

        case GGML_OP_MUL: {
            // TODO correct but no perf gain
            return false;
            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];
            if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) return false;
            if (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) return false;
            if (ggml_impl_is_view(op)) return false;
            if (!ggml_inpu_has_npu_broadcast_shape(src0, src1)) return false;
            return true;
        }

        case GGML_OP_ADD: {
            // TODO correct but no perf gain
            return false;
            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];

            // In-place ADD aliases src0/output memory. The current OV path binds
            // inputs and outputs separately, so reject these cases for now.
            if (ggml_impl_is_view(op)) return false;

            // Repeated F32 accumulation chains drift beyond the very strict backend-op
            // tolerance because the NPU executes the elementwise kernel in FP16.
            // Keep leaf/bias-style adds enabled, but fall back for chained adds.
            if (op->type == GGML_TYPE_F32) {
                if (src0->op == GGML_OP_NONE && src1->op == GGML_OP_NONE) return false;
                if ((src0->op == GGML_OP_ADD || src1->op == GGML_OP_ADD) && src0->op != GGML_OP_MUL_MAT) return false;
            } 

            // OpenVINO NPU elementwise ADD supports NumPy-style broadcasting.
            // ggml also allows repeating non-unit dimensions (for example 10 -> 20),
            // which the current translator does not lower explicitly.
            if (!ggml_inpu_has_npu_broadcast_shape(src0, src1)) return false;

            return true;
        }

        case GGML_OP_GLU: {
            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];
            const enum ggml_glu_op glu_op = ggml_get_glu_op(op);
            const int32_t swapped = ggml_get_op_params_i32(op, 1);

            // SWIGLU_OAI needs extra alpha/limit handling that is not translated yet.
            if (glu_op == GGML_GLU_OP_SWIGLU_OAI) return false;

            if (swapped) return false;

            if (!src1 && src0->ne[0] % 2 != 0) return false;

            return true;
        }

        default:
            return false;
    }
}

static bool ggml_backend_inpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return ggml_backend_buft_is_host(buft) || ggml_backend_is_inpu_buffer_type(buft);
}

static const struct ggml_backend_device_i ggml_backend_inpu_device_i = {
    /* .get_name             = */ ggml_backend_inpu_device_get_name,
    /* .get_description      = */ ggml_backend_inpu_device_get_description,
    /* .get_memory           = */ ggml_backend_inpu_device_get_memory,
    /* .get_type             = */ ggml_backend_inpu_device_get_type,
    /* .get_props            = */ ggml_backend_inpu_device_get_props,
    /* .init_backend         = */ ggml_backend_inpu_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_inpu_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_inpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_inpu_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

// ============================================================================
// Registry interface
// ============================================================================

static const char * ggml_backend_inpu_reg_get_name(ggml_backend_reg_t reg) {
    return "iNPU";
    GGML_UNUSED(reg);
}

static size_t ggml_backend_inpu_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;
    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_inpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_inpu_device = {
        /* .iface   = */ ggml_backend_inpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &ggml_backend_inpu_device;

    GGML_UNUSED(index);
}

// Extra buffer types for weight loading
static ggml_backend_buffer_type_t * ggml_backend_inpu_device_get_extra_buffers_type(ggml_backend_dev_t dev) {
    static ggml_backend_buffer_type_t bufts[2];
    bufts[0] = ggml_backend_inpu_buffer_type();
    bufts[1] = nullptr;
    return bufts;
    GGML_UNUSED(dev);
}

static void * ggml_backend_inpu_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        return (void *) ggml_backend_inpu_device_get_extra_buffers_type;
    }
    return nullptr;
    GGML_UNUSED(reg);
}

static const struct ggml_backend_reg_i ggml_backend_inpu_reg_i = {
    /* .get_name         = */ ggml_backend_inpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_inpu_reg_get_device_count,
    /* .get_device       = */ ggml_backend_inpu_reg_get_device,
    /* .get_proc_address = */ ggml_backend_inpu_get_proc_address,
};

ggml_backend_reg_t ggml_backend_inpu_reg(void) {
    static struct ggml_backend_reg ggml_backend_inpu_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_inpu_reg_i,
        /* .context     = */ nullptr,
    };
    return &ggml_backend_inpu_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_inpu_reg)
