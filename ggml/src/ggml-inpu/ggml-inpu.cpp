// ggml-inpu.cpp — Intel NPU backend for llama.cpp
//
// Accelerates MUL_MAT (+ optional fused ADD/GLU) via OpenVINO on Intel NPUs.
// Registers as an ACCEL backend with a custom buffer type for weight preparation.

#include "ggml-inpu.h"

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-inpu-cache.h"
#include "ggml-inpu-debug.h"
#include "ggml-inpu-impl.h"
#include "ggml-inpu-translate.h"
#include "ggml.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_npu/level_zero/level_zero.hpp>
#include <optional>
#include <string>
#include <vector>

#ifdef GGML_INPU_GPU_ENABLED
#    include <CL/cl.h>

#    include <openvino/runtime/intel_gpu/ocl/ocl.hpp>

// Intel USM extension function types
typedef cl_int(CL_API_CALL * clEnqueueMemFillINTEL_fn)(cl_command_queue queue,
                                                       void *           dst_ptr,
                                                       const void *     pattern,
                                                       size_t           pattern_size,
                                                       size_t           size,
                                                       cl_uint          num_events_in_wait_list,
                                                       const cl_event * event_wait_list,
                                                       cl_event *       event);
typedef cl_int(CL_API_CALL * clEnqueueMemcpyINTEL_fn)(cl_command_queue queue,
                                                      cl_bool          blocking,
                                                      void *           dst_ptr,
                                                      const void *     src_ptr,
                                                      size_t           size,
                                                      cl_uint          num_events_in_wait_list,
                                                      const cl_event * event_wait_list,
                                                      cl_event *       event);
#endif  // GGML_INPU_GPU_ENABLED

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

static const char * inpu_get_target_device_kind(void) {
    static const char * cached = []() -> const char * {
        const char * val = std::getenv("GGML_INPU_DEVICE");
        if (!val || *val == '\0') {
            return "NPU";
        }
        if (std::strcmp(val, "CPU") == 0) {
            return "CPU";
        }
        if (std::strcmp(val, "NPU") == 0) {
            return "NPU";
        }
#ifdef GGML_INPU_GPU_ENABLED
        if (std::strcmp(val, "GPU") == 0)
            return "GPU";
#endif
        INPU_LOG_WARN("unknown GGML_INPU_DEVICE=%s, defaulting to NPU\n", val);
        return "NPU";
    }();
    return cached;
}

static bool inpu_target_is_npu(void) {
    return std::strcmp(inpu_get_target_device_kind(), "NPU") == 0;
}

#ifdef GGML_INPU_GPU_ENABLED
static bool inpu_target_is_gpu(void) {
    return std::strcmp(inpu_get_target_device_kind(), "GPU") == 0;
}
#endif

static bool inpu_use_groups_first_quant_layout(void) {
    // NPU path uses groups-first [n_groups, N, gs] for best performance.
    // CPU/GPU paths keep row-first [N, n_groups, gs] to avoid extra permutation.
    return inpu_target_is_npu();
}

struct ggml_inpu_shared_runtime {
    std::mutex                                              mutex;
    std::shared_ptr<ov::Core>                               core;
    std::string                                             device_name = "NPU";
    std::shared_ptr<ov::intel_npu::level_zero::ZeroContext> l0_context;

#ifdef GGML_INPU_GPU_ENABLED
    // GPU (OpenCL) state
    cl_command_queue         cl_queue = nullptr;
    ov::RemoteContext        gpu_remote_context;
    bool                     has_gpu_context = false;
    clEnqueueMemFillINTEL_fn cl_mem_fill_fn  = nullptr;
    clEnqueueMemcpyINTEL_fn  cl_mem_cpy_fn   = nullptr;

    ~ggml_inpu_shared_runtime() {
        if (cl_queue) {
            clReleaseCommandQueue(cl_queue);
            cl_queue = nullptr;
        }
    }
#endif
};

struct ggml_inpu_context {
    ggml_inpu_shared_runtime * runtime;  // shared singleton, not owned
    inpu_cache                 cache;
};

static ggml_inpu_shared_runtime & ggml_inpu_get_shared_runtime() {
    static ggml_inpu_shared_runtime runtime;
    return runtime;
}

static void ggml_inpu_init_shared_runtime() {
    auto &                      runtime = ggml_inpu_get_shared_runtime();
    std::lock_guard<std::mutex> lock(runtime.mutex);

    if (runtime.core) {
        return;
    }

    const char * target_kind = inpu_get_target_device_kind();
    runtime.core        = std::make_shared<ov::Core>();
    runtime.device_name      = target_kind;

    try {
        auto devices      = runtime.core->get_available_devices();
        bool found_target = false;
        for (const auto & dev : devices) {
            if (dev.find(target_kind) != std::string::npos) {
                found_target        = true;
                runtime.device_name = dev;
                break;
            }
        }
        if (!found_target) {
            INPU_LOG_WARN("no %s device found in OpenVINO, available devices:", target_kind);
            for (const auto & dev : devices) {
                INPU_LOG_WARN("  %s", dev.c_str());
            }
        }
    } catch (const std::exception & e) {
        INPU_LOG_ERROR("failed to enumerate OV devices: %s\n", e.what());
    }

    // Only create L0 context for NPU device
    if (std::strcmp(target_kind, "NPU") == 0) {
        try {
            auto l0 =
                runtime.core->get_default_context(runtime.device_name).as<ov::intel_npu::level_zero::ZeroContext>();
            runtime.l0_context = std::make_shared<ov::intel_npu::level_zero::ZeroContext>(std::move(l0));
        } catch (const std::exception & e) {
            INPU_LOG_WARN("failed to create shared NPU Level Zero context: %s\n", e.what());
        }
    }

#ifdef GGML_INPU_GPU_ENABLED
    // Create OpenCL context + queue + ClContext for GPU
    if (std::strcmp(target_kind, "GPU") == 0) {
        cl_int         err;
        cl_platform_id platform;
        err = clGetPlatformIDs(1, &platform, nullptr);
        if (err != CL_SUCCESS) {
            INPU_LOG_ERROR("failed to get OpenCL platform: %d\n", err);
            return;
        }

        cl_device_id cl_device;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &cl_device, nullptr);
        if (err != CL_SUCCESS) {
            INPU_LOG_ERROR("failed to get OpenCL GPU device: %d\n", err);
            return;
        }

        cl_context cl_ctx = clCreateContext(nullptr, 1, &cl_device, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) {
            INPU_LOG_ERROR("failed to create OpenCL context: %d\n", err);
            return;
        }

        runtime.cl_queue = clCreateCommandQueueWithProperties(cl_ctx, cl_device, nullptr, &err);
        if (err != CL_SUCCESS) {
            INPU_LOG_ERROR("failed to create OpenCL command queue: %d\n", err);
            clReleaseContext(cl_ctx);
            return;
        }

        // Wrap queue in OpenVINO GPU remote context
        runtime.gpu_remote_context = ov::intel_gpu::ocl::ClContext(*runtime.core, runtime.cl_queue);
        runtime.has_gpu_context    = true;
        clReleaseContext(cl_ctx);  // queue keeps a reference

        // Lazy-load Intel USM extension function pointers
        runtime.cl_mem_fill_fn =
            (clEnqueueMemFillINTEL_fn) clGetExtensionFunctionAddressForPlatform(platform, "clEnqueueMemFillINTEL");
        runtime.cl_mem_cpy_fn =
            (clEnqueueMemcpyINTEL_fn) clGetExtensionFunctionAddressForPlatform(platform, "clEnqueueMemcpyINTEL");

        if (!runtime.cl_mem_cpy_fn) {
            INPU_LOG_WARN("clEnqueueMemcpyINTEL not available\n");
        }

        INPU_LOG_INFO("GPU OpenCL context initialized\n");
    }
#endif  // GGML_INPU_GPU_ENABLED
}

static ggml_inpu_shared_runtime & ggml_inpu_shared_runtime_ref() {
    ggml_inpu_init_shared_runtime();
    return ggml_inpu_get_shared_runtime();
}

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
    void *                                                       data             = nullptr;
    size_t                                                       size             = 0;
    bool                                                         is_l0_allocation = false;
    std::unique_ptr<ov::intel_npu::level_zero::ZeroBufferTensor> l0_tensor;  // keeps L0 allocation alive
    std::unordered_map<const struct ggml_tensor *, std::unique_ptr<ggml_inpu_tensor_extra>> extras;
    std::mutex extras_mutex;

#ifdef GGML_INPU_GPU_ENABLED
    bool                        is_gpu_remote = false;
    std::shared_ptr<ov::Tensor> gpu_usm_buffer;  // keeps USM allocation alive
#endif
};

static void ggml_backend_inpu_buffer_free(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<ggml_backend_inpu_buffer_context *>(buffer->context);
#ifdef GGML_INPU_GPU_ENABLED
    if (ctx->is_gpu_remote) {
        // USM memory freed by gpu_usm_buffer destructor
        delete ctx;
        return;
    }
#endif
    if (!ctx->is_l0_allocation) {
        free(ctx->data);
    }
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

static void inpu_reformat_q4_0_set(struct ggml_tensor * tensor,
                                   const void *         data,
                                   size_t               offset,
                                   size_t               size,
                                   bool                 groups_first_layout) {
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
            const size_t    idx =
                groups_first_layout ? (size_t) (b * n_rows + row) : (size_t) (row * n_blocks_per_row + b);
            // Extract fp16 scale
            memcpy(dst_scales + idx * sizeof(uint16_t), block, sizeof(uint16_t));
            // Repack u4 nibbles from ggml → OV layout, then XOR 0x88 to convert u4→i4
            uint8_t tmp[16];
            unpack_32_4(block + sizeof(uint16_t), tmp);
            uint8_t * dst_q = dst_quants + idx * quant_group_bytes;
            for (int i = 0; i < 16; i++) {
                dst_q[i] = tmp[i] ^ 0x88;
            }
        }
    }
}

static void inpu_reformat_q4_0_get(const struct ggml_tensor * tensor,
                                   void *                     data,
                                   size_t                     offset,
                                   size_t                     size,
                                   bool                       groups_first_layout) {
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
            const size_t idx =
                groups_first_layout ? (size_t) (b * n_rows + row) : (size_t) (row * n_blocks_per_row + b);
            // Restore scale
            memcpy(block, src_scales + idx * sizeof(uint16_t), sizeof(uint16_t));
            // Reverse: XOR 0x88 to convert i4→u4, then repack OV → ggml nibble layout
            uint8_t tmp[16];
            const uint8_t * src_q = src_quants + idx * quant_group_bytes;
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

static void inpu_reformat_q8_0_set(struct ggml_tensor * tensor,
                                   const void *         data,
                                   size_t               offset,
                                   size_t               size,
                                   bool                 groups_first_layout) {
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
            const size_t    idx =
                groups_first_layout ? (size_t) (b * n_rows + row) : (size_t) (row * n_blocks_per_row + b);
            // Extract fp16 scale
            memcpy(dst_scales + idx * sizeof(uint16_t), block, sizeof(uint16_t));
            // Copy int8 quants directly (already signed)
            const uint8_t * src_qs = block + sizeof(uint16_t);
            uint8_t *       dst_q  = dst_quants + idx * 32;
            memcpy(dst_q, src_qs, 32);
        }
    }
}

static void inpu_reformat_q8_0_get(const struct ggml_tensor * tensor,
                                   void *                     data,
                                   size_t                     offset,
                                   size_t                     size,
                                   bool                       groups_first_layout) {
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
            const size_t idx =
                groups_first_layout ? (size_t) (b * n_rows + row) : (size_t) (row * n_blocks_per_row + b);
            // Restore scale
            memcpy(block, src_scales + idx * sizeof(uint16_t), sizeof(uint16_t));
            // Copy int8 quants directly
            memcpy(block + sizeof(uint16_t), src_quants + idx * 32, 32);
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
    const bool    profile_enabled = inpu_debug_profile_enabled();
    const int64_t t_start_us      = profile_enabled ? ggml_time_us() : 0;
    const bool    groups_first_layout = inpu_use_groups_first_quant_layout();

#ifdef GGML_INPU_GPU_ENABLED
    // GPU path: for non-quantized data, use clEnqueueMemcpyINTEL (host→device).
    // For quantized weights, reformat into a host staging buffer, then copy to device.
    if (buf_ctx->is_gpu_remote) {
        auto & rt = ggml_inpu_get_shared_runtime();
        switch (tensor->type) {
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q8_0:
                {
                    // Allocate a host staging buffer, reformat there, then copy to device
                    std::vector<uint8_t> staging(ggml_nbytes(tensor));
                    void *               saved_data = tensor->data;
                    tensor->data                    = staging.data();

                    if (tensor->type == GGML_TYPE_Q4_0) {
                        inpu_reformat_q4_0_set(tensor, data, offset, size, groups_first_layout);
                    } else {
                        inpu_reformat_q8_0_set(tensor, data, offset, size, groups_first_layout);
                    }

                    // Copy reformatted data from host staging to device
                    tensor->data = saved_data;
                    cl_int err = rt.cl_mem_cpy_fn(rt.cl_queue, CL_TRUE, tensor->data, staging.data(), staging.size(), 0,
                                                  nullptr, nullptr);
                    if (err != CL_SUCCESS) {
                        INPU_LOG_ERROR("clEnqueueMemcpyINTEL (quant set) failed: %d\n", err);
                    }

                    // Create extra metadata (pointers are device pointers)
                    const int64_t n_rows           = ggml_nrows(tensor);
                    const int64_t n_cols           = tensor->ne[0];
                    const int64_t n_blocks_per_row = n_cols / 32;

                    auto extra                 = std::make_unique<ggml_inpu_tensor_extra>();
                    extra->quants              = tensor->data;
                    extra->n_rows              = n_rows;
                    extra->n_cols              = n_cols;
                    extra->group_size          = 32;
                    extra->groups_first_layout = groups_first_layout;
                    extra->is_quantized        = true;

                    if (tensor->type == GGML_TYPE_Q4_0) {
                        const size_t quants_bytes = (size_t) n_rows * ((size_t) n_cols / 2);
                        const size_t scales_bytes = (size_t) n_rows * (size_t) n_blocks_per_row * sizeof(uint16_t);
                        extra->scales             = static_cast<uint8_t *>(tensor->data) + quants_bytes;
                        extra->quants_bytes       = quants_bytes;
                        extra->scales_bytes       = scales_bytes;
                        extra->quant_ov_type      = ov::element::i4;
                        extra->scale_ov_type      = ov::element::f16;
                    } else {
                        const size_t quants_bytes = (size_t) n_rows * (size_t) n_cols;
                        const size_t scales_bytes = (size_t) n_rows * (size_t) n_blocks_per_row * sizeof(uint16_t);
                        extra->scales             = static_cast<uint8_t *>(tensor->data) + quants_bytes;
                        extra->quants_bytes       = quants_bytes;
                        extra->scales_bytes       = scales_bytes;
                        extra->quant_ov_type      = ov::element::i8;
                        extra->scale_ov_type      = ov::element::f16;
                    }

                    tensor->extra = extra.get();
                    {
                        std::lock_guard<std::mutex> lock(buf_ctx->extras_mutex);
                        buf_ctx->extras[tensor] = std::move(extra);
                    }
                    break;
                }
            default:
                {
                    cl_int err = rt.cl_mem_cpy_fn(rt.cl_queue, CL_TRUE, static_cast<char *>(tensor->data) + offset,
                                                  data, size, 0, nullptr, nullptr);
                    if (err != CL_SUCCESS) {
                        INPU_LOG_ERROR("clEnqueueMemcpyINTEL (set) failed: %d\n", err);
                    }
                    break;
                }
        }
        if (profile_enabled) {
            inpu_profile_record_buffer_set(tensor->type, size, (uint64_t) (ggml_time_us() - t_start_us));
        }
        return;
    }
#endif  // GGML_INPU_GPU_ENABLED

    switch (tensor->type) {
        case GGML_TYPE_Q4_0: {
                inpu_reformat_q4_0_set(tensor, data, offset, size, groups_first_layout);

                const int64_t n_rows           = ggml_nrows(tensor);
                const int64_t n_cols           = tensor->ne[0];
                const int64_t n_blocks_per_row = n_cols / 32;
                const size_t  quants_bytes     = (size_t) n_rows * ((size_t) n_cols / 2);
                const size_t  scales_bytes     = (size_t) n_rows * (size_t) n_blocks_per_row * sizeof(uint16_t);

                auto extra                 = std::make_unique<ggml_inpu_tensor_extra>();
                extra->quants              = tensor->data;
                extra->scales              = static_cast<uint8_t *>(tensor->data) + quants_bytes;
                extra->quants_bytes        = quants_bytes;
                extra->scales_bytes        = scales_bytes;
                extra->quant_ov_type       = ov::element::i4;
                extra->scale_ov_type       = ov::element::f16;
                extra->n_rows              = n_rows;
                extra->n_cols              = n_cols;
                extra->group_size          = 32;
                extra->groups_first_layout = groups_first_layout;
                extra->is_quantized        = true;

                tensor->extra = extra.get();
                {
                    std::lock_guard<std::mutex> lock(buf_ctx->extras_mutex);
                    buf_ctx->extras[tensor] = std::move(extra);
                }
            break;
        }

        case GGML_TYPE_Q8_0: {
                inpu_reformat_q8_0_set(tensor, data, offset, size, groups_first_layout);

                const int64_t n_rows           = ggml_nrows(tensor);
                const int64_t n_cols           = tensor->ne[0];
                const int64_t n_blocks_per_row = n_cols / 32;
                const size_t  quants_bytes     = (size_t) n_rows * (size_t) n_cols;
                const size_t  scales_bytes     = (size_t) n_rows * (size_t) n_blocks_per_row * sizeof(uint16_t);

                auto extra                 = std::make_unique<ggml_inpu_tensor_extra>();
                extra->quants              = tensor->data;
                extra->scales              = static_cast<uint8_t *>(tensor->data) + quants_bytes;
                extra->quants_bytes        = quants_bytes;
                extra->scales_bytes        = scales_bytes;
                extra->quant_ov_type       = ov::element::i8;
                extra->scale_ov_type       = ov::element::f16;
                extra->n_rows              = n_rows;
                extra->n_cols              = n_cols;
                extra->group_size          = 32;
                extra->groups_first_layout = groups_first_layout;
                extra->is_quantized        = true;

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

    if (profile_enabled) {
        inpu_profile_record_buffer_set(tensor->type, size, (uint64_t) (ggml_time_us() - t_start_us));
    }
}

static void ggml_backend_inpu_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                const struct ggml_tensor * tensor,
                                                void * data,
                                                size_t offset,
                                                size_t size) {
    GGML_UNUSED(buffer);
    const bool    profile_enabled = inpu_debug_profile_enabled();
    const int64_t t_start_us      = profile_enabled ? ggml_time_us() : 0;
    const bool    groups_first_layout = inpu_use_groups_first_quant_layout();

#ifdef GGML_INPU_GPU_ENABLED
    auto * buf_ctx = static_cast<ggml_backend_inpu_buffer_context *>(buffer->context);
    if (buf_ctx->is_gpu_remote) {
        auto & rt = ggml_inpu_get_shared_runtime();
        switch (tensor->type) {
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q8_0:
                {
                    // Copy device → host staging, then reverse-reformat
                    std::vector<uint8_t> staging(ggml_nbytes(tensor));
                    cl_int err = rt.cl_mem_cpy_fn(rt.cl_queue, CL_TRUE, staging.data(), tensor->data, staging.size(), 0,
                                                  nullptr, nullptr);
                    if (err != CL_SUCCESS) {
                        INPU_LOG_ERROR("clEnqueueMemcpyINTEL (quant get) failed: %d\n", err);
                    }
                    // Create a temporary tensor pointing to staging for the reformat functions
                    struct ggml_tensor tmp = *tensor;
                    tmp.data               = staging.data();
                    if (tensor->type == GGML_TYPE_Q4_0) {
                        inpu_reformat_q4_0_get(&tmp, data, offset, size, groups_first_layout);
                    } else {
                        inpu_reformat_q8_0_get(&tmp, data, offset, size, groups_first_layout);
                    }
                    break;
                }
            default:
                {
                    cl_int err =
                        rt.cl_mem_cpy_fn(rt.cl_queue, CL_TRUE, data, static_cast<const char *>(tensor->data) + offset,
                                         size, 0, nullptr, nullptr);
                    if (err != CL_SUCCESS) {
                        INPU_LOG_ERROR("clEnqueueMemcpyINTEL (get) failed: %d\n", err);
                    }
                    break;
                }
        }
        if (profile_enabled) {
            inpu_profile_record_buffer_get(tensor->type, size, (uint64_t) (ggml_time_us() - t_start_us));
        }
        return;
    }
#endif  // GGML_INPU_GPU_ENABLED

    switch (tensor->type) {
        case GGML_TYPE_Q4_0:
            inpu_reformat_q4_0_get(tensor, data, offset, size, groups_first_layout);
            break;

        case GGML_TYPE_Q8_0:
            inpu_reformat_q8_0_get(tensor, data, offset, size, groups_first_layout);
            break;

        default:
            memcpy(data, static_cast<const char *>(tensor->data) + offset, size);
            break;
    }

    if (profile_enabled) {
        inpu_profile_record_buffer_get(tensor->type, size, (uint64_t) (ggml_time_us() - t_start_us));
    }
}

static void ggml_backend_inpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = static_cast<ggml_backend_inpu_buffer_context *>(buffer->context);
#ifdef GGML_INPU_GPU_ENABLED
    if (ctx->is_gpu_remote) {
        auto & rt = ggml_inpu_get_shared_runtime();
        if (rt.cl_mem_fill_fn && rt.cl_queue) {
            uint8_t pattern = value;
            cl_int  err =
                rt.cl_mem_fill_fn(rt.cl_queue, ctx->data, &pattern, sizeof(pattern), ctx->size, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                INPU_LOG_ERROR("clEnqueueMemFillINTEL (clear) failed: %d\n", err);
            }
            clFinish(rt.cl_queue);
        }
        return;
    }
#endif
    memset(ctx->data, value, ctx->size);
}

#ifdef GGML_INPU_GPU_ENABLED
static void ggml_backend_inpu_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                   struct ggml_tensor *  tensor,
                                                   uint8_t               value,
                                                   size_t                offset,
                                                   size_t                size) {
    auto * ctx = static_cast<ggml_backend_inpu_buffer_context *>(buffer->context);
    if (ctx->is_gpu_remote) {
        auto & rt = ggml_inpu_get_shared_runtime();
        if (rt.cl_mem_fill_fn && rt.cl_queue) {
            uint8_t pattern = value;
            cl_int  err     = rt.cl_mem_fill_fn(rt.cl_queue, static_cast<char *>(tensor->data) + offset, &pattern,
                                                sizeof(pattern), size, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                INPU_LOG_ERROR("clEnqueueMemFillINTEL (memset) failed: %d\n", err);
            }
            clFinish(rt.cl_queue);
        }
    } else {
        memset(static_cast<char *>(tensor->data) + offset, value, size);
    }
}
#endif  // GGML_INPU_GPU_ENABLED

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
#ifdef GGML_INPU_GPU_ENABLED
    /* .memset_tensor = */ ggml_backend_inpu_buffer_memset_tensor,
#else
    /* .memset_tensor = */ nullptr,
#endif
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
    bool                                                         is_l0_allocation = false;
    std::unique_ptr<ov::intel_npu::level_zero::ZeroBufferTensor> l0_tensor;
#ifdef GGML_INPU_GPU_ENABLED
    bool                        is_gpu_remote = false;
    std::shared_ptr<ov::Tensor> gpu_usm_buffer;
#endif

    auto & rt = ggml_inpu_shared_runtime_ref();
    const bool is_npu = inpu_target_is_npu();

    if (size > 0) {
        const size_t alignment = 64;
        size = (size + alignment - 1) & ~(alignment - 1);

#ifdef GGML_INPU_GPU_ENABLED
        if (inpu_target_is_gpu() && rt.has_gpu_context) {
            try {
                auto gpu_context = rt.gpu_remote_context.as<ov::intel_gpu::ocl::ClContext>();
                auto usm_tensor  = gpu_context.create_usm_device_tensor(ov::element::u8, ov::Shape{ size });
                data             = usm_tensor.get();
                if (data) {
                    is_gpu_remote  = true;
                    gpu_usm_buffer = std::make_shared<ov::intel_gpu::ocl::USMTensor>(std::move(usm_tensor));
                    INPU_LOG_DEBUG("allocated GPU USM device buffer (%zu bytes)\n", size);
                }
            } catch (const std::exception & e) {
                INPU_LOG_WARN("GPU USM allocation failed, falling back to aligned_alloc: %s\n", e.what());
            }
        }
#endif

        if (!data && is_npu && rt.l0_context) {
            try {
                auto   remote_tensor  = rt.l0_context->create_l0_host_tensor(ov::element::u8, { size });
                void * level_zero_ptr = remote_tensor.get();

                if (level_zero_ptr != nullptr) {
                    data             = level_zero_ptr;
                    is_l0_allocation = true;
                    l0_tensor = std::make_unique<ov::intel_npu::level_zero::ZeroBufferTensor>(std::move(remote_tensor));
                    INPU_LOG_DEBUG("allocated iNPU buffer with shared L0 host tensor (%zu bytes)\n", size);
                }
            } catch (const std::exception & e) {
                INPU_LOG_WARN("L0 host tensor allocation failed, falling back to aligned_alloc: %s\n", e.what());
            }
        }

        if (!data) {
            data = aligned_alloc(alignment, size);
            if (data) {
                INPU_LOG_DEBUG("allocated iNPU buffer with aligned_alloc (%zu bytes)\n", size);
            }
        }
        if (!data) {
            INPU_LOG_ERROR("failed to allocate %zu bytes for iNPU buffer\n", size);
            return nullptr;
        }
    }

    auto * buf_ctx            = new ggml_backend_inpu_buffer_context();
    buf_ctx->data             = data;
    buf_ctx->size             = size;
    buf_ctx->is_l0_allocation = is_l0_allocation;
    buf_ctx->l0_tensor        = std::move(l0_tensor);
#ifdef GGML_INPU_GPU_ENABLED
    buf_ctx->is_gpu_remote  = is_gpu_remote;
    buf_ctx->gpu_usm_buffer = std::move(gpu_usm_buffer);
#endif

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
    const bool              profile_enabled  = inpu_debug_profile_enabled();
    const int64_t           t_total_start_us = profile_enabled ? ggml_time_us() : 0;
    inpu_profile_graph_call profile_call;

    if (profile_enabled) {
        profile_call.graph_label = inpu_debug_graph_label(cgraph);
        profile_call.n_nodes     = cgraph ? cgraph->n_nodes : 0;
    }

    auto finish = [&](enum ggml_status status) {
        if (profile_enabled) {
            profile_call.success  = status == GGML_STATUS_SUCCESS;
            profile_call.total_us = (uint64_t) (ggml_time_us() - t_total_start_us);
            inpu_profile_record_graph_call(profile_call);
        }
        return status;
    };

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
        return finish(GGML_STATUS_SUCCESS);
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
        return finish(GGML_STATUS_SUCCESS);
    }

    // 1. Make cache key
    const int64_t  t_key_start_us = profile_enabled ? ggml_time_us() : 0;
    inpu_graph_key key = inpu_make_graph_key(cgraph);
    if (profile_enabled) {
        profile_call.key_us     = (uint64_t) (ggml_time_us() - t_key_start_us);
        profile_call.graph_hash = inpu_graph_key_hash{}(key);
    }

    // 2. Look up cache
    const int64_t t_cache_lookup_start_us = profile_enabled ? ggml_time_us() : 0;
    auto compiled = ctx->cache.find(key);
    if (profile_enabled) {
        profile_call.cache_lookup_us = (uint64_t) (ggml_time_us() - t_cache_lookup_start_us);
        profile_call.cache_hit       = compiled != nullptr;
    }

    if (!compiled) {
        // Debug: dump cgraph if requested via GGML_INPU_DUMP_CGRAPH=1
        if (inpu_debug_dump_cgraph_enabled()) {
            inpu_debug_dump_cgraph(cgraph);
        }

        // 3. Analyze I/O
        const int64_t t_analyze_io_start_us = profile_enabled ? ggml_time_us() : 0;
        inpu_graph_io io = inpu_analyze_graph_io(cgraph);
        if (profile_enabled) {
            profile_call.analyze_io_us = (uint64_t) (ggml_time_us() - t_analyze_io_start_us);
        }

        // 4. Translate to OV model
        const int64_t         t_translate_start_us = profile_enabled ? ggml_time_us() : 0;
        inpu_translate_result result = inpu_translate_graph(cgraph, io);
        if (profile_enabled) {
            profile_call.translate_us = (uint64_t) (ggml_time_us() - t_translate_start_us);
        }

        if (!result.model) {
            INPU_LOG_ERROR("failed to translate cgraph to OV model\n");
            return finish(GGML_STATUS_FAILED);
        }

        // Debug: dump OV IR if requested via GGML_INPU_DUMP_IR=<dir>
        inpu_debug_dump_ir(result.model);

        // 5. Compile for target device
        try {
            const int64_t t_compile_start_us = profile_enabled ? ggml_time_us() : 0;
            auto * rt = ctx->runtime;
            if (!rt || !rt->core) {
                INPU_LOG_ERROR("shared OV runtime is not initialized\n");
                return finish(GGML_STATUS_FAILED);
            }

            ov::AnyMap config;
            const bool is_npu = rt->device_name.find("NPU") != std::string::npos;
            if (is_npu) {
                config = {
                    { "NPU_COMPILER_DYNAMIC_QUANTIZATION", "YES" },
                    { "NPU_USE_NPUW",                      "YES" },
                    { "NPUW_DEVICES",                      "NPU" },
                    { "NPUW_FOLD",                         "YES" },
                    { "NPUW_FUNCALL_FOR_ALL",              "YES" },
                    { "NPUW_FUNCALL_ASYNC",                "YES" },
                };
            }

            ov::CompiledModel cm = [&]() {
                if (is_npu && rt->l0_context) {
                    return rt->core->compile_model(result.model, *rt->l0_context, config);
                }
#ifdef GGML_INPU_GPU_ENABLED
                if (inpu_target_is_gpu() && rt->has_gpu_context) {
                    return rt->core->compile_model(result.model, rt->gpu_remote_context, config);
                }
#endif
                return rt->core->compile_model(result.model, rt->device_name, config);
            }();

            compiled = std::make_shared<inpu_compiled_graph>();
            compiled->compiled_model = std::make_shared<ov::CompiledModel>(std::move(cm));
            compiled->input_map = std::move(result.input_map);
            compiled->output_map = std::move(result.output_map);

            ctx->cache.insert(key, compiled);

            if (profile_enabled) {
                profile_call.compile_us = (uint64_t) (ggml_time_us() - t_compile_start_us);
            }

            INPU_LOG_DEBUG("compiled new OV model for %s (%d nodes)\n", rt->device_name.c_str(), cgraph->n_nodes);
        } catch (const std::exception & e) {
            INPU_LOG_ERROR("OV compile failed: %s\n", e.what());
            return finish(GGML_STATUS_FAILED);
        }
    }

    // 6. Acquire an InferRequest
    bool             request_from_pool          = false;
    const int64_t    t_acquire_request_start_us = profile_enabled ? ggml_time_us() : 0;
    ov::InferRequest infer_req = compiled->acquire_request(profile_enabled ? &request_from_pool : nullptr);
    if (profile_enabled) {
        profile_call.request_from_pool  = request_from_pool;
        profile_call.acquire_request_us = (uint64_t) (ggml_time_us() - t_acquire_request_start_us);
    }

    auto ov_shape_nelements = [](const ov::Shape & shape) -> size_t {
        size_t n = 1;
        for (size_t dim : shape) {
            n *= dim;
        }
        return n;
    };

    try {
        // Helper: create an ov::Tensor wrapping a (possibly device) pointer.
        // For GPU, we use the remote context to wrap USM pointers.
#ifdef GGML_INPU_GPU_ENABLED
        auto make_tensor = [&](ov::element::Type et, const ov::Shape & shape, void * ptr) -> ov::Tensor {
            if (inpu_target_is_gpu() && ctx->runtime->has_gpu_context) {
                auto gpu_context = ctx->runtime->gpu_remote_context.as<ov::intel_gpu::ocl::ClContext>();
                return gpu_context.create_tensor(et, shape, ptr);
            }
            return ov::Tensor(et, shape, ptr);
        };
#else
        auto make_tensor = [](ov::element::Type et, const ov::Shape & shape, void * ptr) -> ov::Tensor {
            return ov::Tensor(et, shape, ptr);
        };
#endif

        // 7. Bind inputs
        const int64_t t_bind_inputs_start_us = profile_enabled ? ggml_time_us() : 0;
        for (const auto & entry : compiled->input_map) {
            const struct ggml_tensor * node = cgraph->nodes[entry.node_idx >= 0 ? entry.node_idx : 0];
            const struct ggml_tensor * src_tensor = nullptr;

            if (entry.src_idx >= 0 && entry.src_idx < GGML_MAX_SRC) {
                src_tensor = node->src[entry.src_idx];
            }

            if (!src_tensor) {
                INPU_LOG_ERROR("null src tensor for input '%s'\n", entry.ov_name.c_str());
                compiled->release_request(std::move(infer_req));
                return finish(GGML_STATUS_FAILED);
            }

            const struct ggml_tensor * bind_tensor = resolve_input_binding_tensor(src_tensor);
            const void * data_ptr = bind_tensor->data;

            if (entry.is_quant || entry.is_scale) {
                // Quantized weight: use the separated layout from tensor->extra
                auto * extra = static_cast<ggml_inpu_tensor_extra *>(src_tensor->extra);
                if (!extra) {
                    INPU_LOG_ERROR("no tensor extra for quantized input '%s'\n", entry.ov_name.c_str());
                    compiled->release_request(std::move(infer_req));
                    return finish(GGML_STATUS_FAILED);
                }

                if (entry.is_quant) {
                    ov::Shape shape = infer_req.get_tensor(entry.ov_name).get_shape();
                    auto      t     = make_tensor(extra->quant_ov_type, shape, extra->quants);
                    infer_req.set_tensor(entry.ov_name, t);
                    if (profile_enabled) {
                        profile_call.input_tensors++;
                        profile_call.quant_bytes += extra->quants_bytes;
                        profile_call.input_bytes += extra->quants_bytes;
                    }
                } else {
                    ov::Shape shape = infer_req.get_tensor(entry.ov_name).get_shape();
                    auto      t     = make_tensor(extra->scale_ov_type, shape, extra->scales);
                    infer_req.set_tensor(entry.ov_name, t);
                    if (profile_enabled) {
                        profile_call.input_tensors++;
                        profile_call.scale_bytes += extra->scales_bytes;
                        profile_call.input_bytes += extra->scales_bytes;
                    }
                }
            } else {
                // Non-quantized input: wrap the data pointer directly
                ov::Shape shape = infer_req.get_tensor(entry.ov_name).get_shape();
                ov::element::Type ov_type;
                switch (bind_tensor->type) {
                    case GGML_TYPE_F32: ov_type = ov::element::f32; break;
                    case GGML_TYPE_F16: ov_type = ov::element::f16; break;
                    case GGML_TYPE_I32: ov_type = ov::element::i32; break;
                    case GGML_TYPE_I64:
                        ov_type = ov::element::i64;
                        break;
                    default:
                        INPU_LOG_ERROR("unsupported input type %d for '%s'\n", bind_tensor->type, entry.ov_name.c_str());
                        compiled->release_request(std::move(infer_req));
                        return finish(GGML_STATUS_FAILED);
                }
                auto t = make_tensor(ov_type, shape, const_cast<void *>(data_ptr));
                infer_req.set_tensor(entry.ov_name, t);
                if (profile_enabled) {
                    profile_call.input_tensors++;
                    profile_call.input_bytes += ggml_nbytes(bind_tensor);
                }
            }
        }
        if (profile_enabled) {
            profile_call.bind_inputs_us = (uint64_t) (ggml_time_us() - t_bind_inputs_start_us);
        }

        // 8. Bind outputs
        const int64_t t_bind_outputs_start_us = profile_enabled ? ggml_time_us() : 0;
        for (const auto & entry : compiled->output_map) {
            struct ggml_tensor * node = cgraph->nodes[entry.bind_node_idx];

            ov::Shape shape = infer_req.get_tensor(entry.ov_name).get_shape();
            const size_t ov_nelements = ov_shape_nelements(shape);
            const size_t ggml_nelems = (size_t) ggml_nelements(node);
            if (ov_nelements != ggml_nelems) {
                INPU_LOG_ERROR("output shape mismatch for '%s': OV has %zu elements, ggml tensor has %zu\n",
                               entry.ov_name.c_str(), ov_nelements, ggml_nelems);
                compiled->release_request(std::move(infer_req));
                return finish(GGML_STATUS_FAILED);
            }

            ov::element::Type ov_type;
            switch (node->type) {
                case GGML_TYPE_F32: ov_type = ov::element::f32; break;
                case GGML_TYPE_F16: ov_type = ov::element::f16; break;
                default:
                    INPU_LOG_ERROR("unsupported output type %d for '%s'\n", node->type, entry.ov_name.c_str());
                    compiled->release_request(std::move(infer_req));
                    return finish(GGML_STATUS_FAILED);
            }
            auto t = make_tensor(ov_type, shape, node->data);
            infer_req.set_tensor(entry.ov_name, t);
            if (profile_enabled) {
                profile_call.output_tensors++;
                profile_call.output_bytes += ggml_nbytes(node);
            }
        }
        if (profile_enabled) {
            profile_call.bind_outputs_us = (uint64_t) (ggml_time_us() - t_bind_outputs_start_us);
        }

        // 9. Run inference
        const int64_t t_infer_start_us = profile_enabled ? ggml_time_us() : 0;
        infer_req.infer();
        if (profile_enabled) {
            profile_call.infer_us = (uint64_t) (ggml_time_us() - t_infer_start_us);
        }

    } catch (const std::exception & e) {
        INPU_LOG_ERROR("OV inference failed: %s\n", e.what());
        compiled->release_request(std::move(infer_req));
        return finish(GGML_STATUS_FAILED);
    }

    // 10. Return the InferRequest to the pool
    const int64_t t_release_request_start_us = profile_enabled ? ggml_time_us() : 0;
    compiled->release_request(std::move(infer_req));
    if (profile_enabled) {
        profile_call.release_request_us = (uint64_t) (ggml_time_us() - t_release_request_start_us);
    }

    return finish(GGML_STATUS_SUCCESS);
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
    ctx->runtime = &ggml_inpu_shared_runtime_ref();

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_inpu_guid(),
        /* .iface   = */ ggml_backend_inpu_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_inpu_reg(), 0),
        /* .context = */ ctx,
    };

    INPU_LOG_INFO("initialized iNPU backend (GGML_INPU_DEVICE=%s, OV device: %s, quant_layout=%s)\n",
                  inpu_get_target_device_kind(), ctx->runtime->device_name.c_str(),
                  inpu_use_groups_first_quant_layout() ? "groups-first" : "row-first");
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
    static std::string desc;
    if (desc.empty()) {
        const char * kind = inpu_get_target_device_kind();
        desc              = std::string("Intel ") + kind + " via OpenVINO";
    }
    return desc.c_str();
    GGML_UNUSED(dev);
}

static void ggml_backend_inpu_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    *free = 0;
    *total = 0;
    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_inpu_device_get_type(ggml_backend_dev_t dev) {
#ifdef GGML_INPU_GPU_ENABLED
    if (inpu_target_is_gpu()) {
        return GGML_BACKEND_DEVICE_TYPE_GPU;
    }
#endif
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
        case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_VIEW:
            return true;

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

        case GGML_OP_ADD: {
            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];

            // In-place ADD aliases src0/output memory. The current OV path binds
            // inputs and outputs separately, so reject these cases for now.
            if (ggml_impl_is_view(op)) return false;

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

        case GGML_OP_RMS_NORM: {
            const struct ggml_tensor * src0 = op->src[0];
            // Only support F32/F16 inputs, and not in-place
            if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) return false;
            if (ggml_impl_is_view(op)) return false;
            return true;
        }

        case GGML_OP_MUL: {
            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];
            if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) return false;
            if (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) return false;
            if (ggml_impl_is_view(op)) return false;
            if (!ggml_inpu_has_npu_broadcast_shape(src0, src1)) return false;
            return true;
        }

        case GGML_OP_ROPE: {
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

        case GGML_OP_GET_ROWS:
            {
                const struct ggml_tensor * src0 = op->src[0];  // data table
                // Only support non-quantized data (F16/F32). Quantized embedding
                // lookups stay on CPU — llama.cpp does not offload them to backends.
                if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
                    return false;
                }
                return true;
            }

        case GGML_OP_SET_ROWS:
            {
#ifdef GGML_INPU_GPU_ENABLED
                if (!inpu_target_is_gpu()) {
                    return false;
                }
#else
                return false;
#endif
                const struct ggml_tensor * src0 = op->src[0];  // source data
                const struct ggml_tensor * src1 = op->src[1];  // indices
                const struct ggml_tensor * src2 = op->src[2];  // destination
                if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
                    return false;
                }
                if (src2->type != GGML_TYPE_F32 && src2->type != GGML_TYPE_F16) {
                    return false;
                }
                if (src1->type != GGML_TYPE_I64 && src1->type != GGML_TYPE_I32) {
                    return false;
                }

                // Match ggml_set_rows() constraints, including batched index
                // broadcast over dims 2/3.
                if (src0->ne[0] != src2->ne[0]) {
                    return false;
                }
                if (src0->ne[2] != src2->ne[2] || src0->ne[3] != src2->ne[3]) {
                    return false;
                }
                if (src0->ne[1] != src1->ne[0]) {
                    return false;
                }
                // ScatterUpdate path supports only a shared row index vector.
                // Per-batch index tensors are currently unsupported.
                // Zero-sized SET_ROWS also fail during NPU compile.
                if (src1->ne[0] == 0 || src1->ne[1] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
                    return false;
                }
                return true;
            }

        case GGML_OP_CPY:
            {
                const struct ggml_tensor * src0 = op->src[0];
                // Only support float-to-float copies
                if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) {
                    return false;
                }
                if (op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16) {
                    return false;
                }
                return true;
            }

        case GGML_OP_FLASH_ATTN_EXT:
            {
                // Only support standard attention (no ALiBi, no logit softcap)
                float max_bias, logit_softcap;
                memcpy(&max_bias, (const char *) op->op_params + sizeof(float), sizeof(float));
                memcpy(&logit_softcap, (const char *) op->op_params + 2 * sizeof(float), sizeof(float));
                if (max_bias != 0.0f) {
                    return false;
                }
                if (logit_softcap != 0.0f) {
                    return false;
                }
                return true;
            }

        default:
            return false;
    }
}

static bool ggml_backend_inpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
#ifdef GGML_INPU_GPU_ENABLED
    if (inpu_target_is_gpu()) {
        // GPU backend cannot directly use host buffer pointers (they are not USM allocations).
        // The scheduler will copy host tensors into iNPU (USM device) buffers as needed.
        return ggml_backend_is_inpu_buffer_type(buft);
    }
#endif
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
