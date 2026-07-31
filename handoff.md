# Handoff: ovgpu backend PoC → continue on Lunar Lake

Read this first, then `WORKLOG.md` for detail and `poc-ovgpu-plan.md` for the plan.

## What this is

A dedicated **OpenVINO GPU (clDNN) backend for ggml** that drives the OV GPU
plugin's clDNN layer **directly, in eager mode** — no `ov::Core` / `ov::Model` /
`compile_model` / `TransformationsPipeline` — reusing the plugin's high-performance
kernels (oneDNN matmul, kernel_selector OCL kernels). Branch `poc-ov-gpu-backend`,
remote `zijun` (github.com/wine99/llama.cpp).

## Why move to Lunar Lake

The backend's whole purpose is **best-in-class Intel-GPU perf via OV kernels**. On
the current dev box (UHD 770, `supports_immad=0`) oneDNN is unavailable, so the FC
falls back to the ocl `bfyx_ref` ref kernel — which can't beat Vulkan, and Vulkan
already runs llama.cpp on Intel GPU. **Lunar Lake (Core Ultra 200V, Xe2) has
`supports_immad=1` → the oneDNN FC path is selected**, where f16 weights are consumed
in-place (no repack). That is the target the backend is built for.

Keep the backend **correct everywhere**, but only **target oneDNN platforms for perf**.

## Current state (as of this commit)

- **M0 DONE**: backend skeleton (reg/device/buffer_type/buffer/stream ifaces),
  clDNN standalone engine init, FC smoke test. `supports_op` gates unsupported ops
  to CPU (scheduler split).
- **M1 MUL_MAT**: **65/65 supported cases pass** in `test-backend-ops`
  (f32/f16/bf16 weights × f32/f16 act; weight 2D; activation batched over ne[2]/ne[3]).
  0 failures, clean exit.
  - Deferred (`supports_op=false` → CPU, **not** failures): grouped/GQA weights
    (ne[2]/ne[3] > 1), permuted (`ggml_permute`) operands, strided-K views,
    quantized types.
- **M1 perf debt (UHD 770 only)**: f16/bf16 weights are upcasted to f32 per compute
  because the ocl path mis-reads f16 weights for some batch sizes (batch=8 wrong,
  batch=9 no kernel). This upcast should be **unnecessary on Lunar Lake** (oneDNN).

## FIRST TASK on Lunar Lake: native f16 weights via oneDNN

1. Build OV and llama.cpp-ovgpu (see "Build setup").
2. `./test-backend-ops -b OVGPU0 -o MUL_MAT` → expect **65/65**; confirm the init
   line prints **`supports_immad=1`**. With `OVGPU_FC_TRACE=1`, confirm a oneDNN FC
   kernel is selected (not `bfyx_ref`).
3. In `ggml/src/ggml-ovgpu/ggml-ovgpu-ops.cpp` `build_mul_mat`: make the weight
   upcast conditional — only upcast when oneDNN is NOT available. The cleanest gate
   is `!engine.get_device_info().supports_immad` (the activation upcast to f32 stays
   for f32-accumulation correctness regardless). Re-run the suite; if still 65/65
   and `llama-bench` is faster, keep it.
4. If native f16 hits any new issue on Lunar Lake, the `OVGPU_FC_TRACE` FC
   instrumentation in OV source (env-gated) is still in place to debug — see WORKLOG.

## Build setup (Lunar Lake)

- **OpenVINO**: clone to `/home/zijun/dev/openvino` (same path → presets work as-is;
  otherwise edit `OpenVINO_DIR` / `GGML_OVGPU_OV_SOURCE_DIR` in `CMakePresets.json`,
  currently `/home/zijun/dev/openvino`). Build the **Release GPU plugin** so the
  static libs + `onednn_gpu_install` exist under `build/Release`. (The instrumented
  `src/plugins/intel_gpu/src/graph/impls/ocl/fully_connected.cpp` is optional —
  env-gated `OVGPU_FC_TRACE`, only for debugging.)
- **llama.cpp-ovgpu**:
  ```
  cmake --preset x64-linux-ovgpu-release
  cmake --build build-x64-linux-ovgpu-release -j
  cmake --build build-x64-linux-ovgpu-release --target test-backend-ops llama-simple llama-bench
  ```
- The OV static-lib link surface is fiddly; the full recipe (libs, include dirs,
  compile defines) is in `WORKLOG.md` → "link surface discovery". It links
  `openvino_intel_gpu_{graph,kernels,runtime}` + `libopenvino_onednn_gpu.a` etc.

## Where things live

- `ggml/src/ggml-ovgpu/ggml-ovgpu.cpp` — backend ifaces, `graph_compute`,
  `supports_op`.
- `ggml/src/ggml-ovgpu/ggml-ovgpu-ops.cpp` — op translators (`build_mul_mat`) +
  layout/cache helpers (`ggml_ne_to_tensor`, `ggml_layout_for`, `wrap_tensor`).
- `ggml/src/ggml-ovgpu/ggml-ovgpu.hpp` — internals (`compiled_op`, `op_cache`).
- `ggml/src/ggml-ovgpu/CMakeLists.txt` — OV static-lib link surface.
- `WORKLOG.md` — **detailed log** (read it): link-surface discovery, USM/mem_lock
  decisions, n=8 yxfb root cause, dtype/accumulation semantics.

## Architecture quick-ref

- clDNN driven standalone: `engine::create(engine_types::ocl, runtime_types::ocl)`
  via `device_query`, no `ov::Core`.
- GPU buffer = **USM device memory** (real pointer for ggml offset arithmetic; this
  iGPU family has no host-accessible `usm_shared`). Host access via `mem_lock`
  (read_write for set, read for get) on `reg_ctx->stream`. Cross-stream sync:
  `reg_ctx->stream->finish()` before compute, `net->get_stream().finish()` after.
- **Layout**: flatten ggml dims 1,2,3 → cldnn batch (feature = ne[0]). Matches
  ggml contiguous memory, so one flat ordering serves a 2D weight, a batched
  activation, and the batched dst.
- **Every op translator ends in a trailing `reorder`** to force bfyx f32 output
  (cldnn's layout optimizer can pick yxfb etc. for some shapes = a transpose).
- **MUL_MAT precision**: ggml dst is always f32 and accumulates f32 (widened);
  cldnn `bfyx_ref` accumulates in the activation dtype, so FC is fed f32 operands.
  Weights stay native only on oneDNN platforms.
- Op networks are **shape-keyed + cached** (`op_cache`), shared across layers/inputs
  with identical shape. Mutex-guarded (test-backend-ops is multi-threaded).

## After MUL_MAT perf settles — next M1 ops

Implement + validate each in `test-backend-ops` (apply the trailing-reorder pattern):
eltwise **ADD/MUL**, **RMS_NORM** (rms primitive), **ROPE** (rope primitive),
**SOFT_MAX**, **GET_ROWS**, **CPY/SET_ROWS** (KV cache), views/reshape. Target:
`llama-simple` on a small dense f16 model (e.g. Qwen3-0.6B).

**Attention is a deferred sync point** — llama.cpp uses `FLASH_ATTN_EXT` or a
`MUL_MAT+SOFTMAX+MUL_MAT` decomposition (NOT paged attention / OV KVCache op).
Decide native cldnn SDPA vs the decomposition when reached.

## Claude memories (auto-load in a session here)

`ovgpu-targets-onednn-platforms` (scope + Lunar Lake rationale),
`ovgpu-mulmat-f32-accumulation` (why FC gets f32 operands).
