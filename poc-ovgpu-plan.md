# PoC: dedicated OpenVINO GPU backend for ggml (eager, clDNN-level)

Status: M1 in progress. M0 complete (skeleton + build + FC smoke test passing).
MUL_MAT done: 65/65 supported cases pass (f32/f16/bf16 weights x f32/f16 act,
weight 2D, batched activation). Grouped-weight (GQA) / quantized deferred.
PERF CAVEAT: f16/bf16 weights are upcast to f32 per compute on UHD 770
(supports_immad=0, no oneDNN); native-weight baking is M2. Worktree: llama.cpp-ovgpu,
branch poc-ov-gpu-backend (base: dev_backend_openvino). Detailed log: WORKLOG.md.

## Goal

GPU-only OpenVINO backend for ggml that drives the clDNN layer of the OV GPU
plugin directly (no ov::Core / ov::Model / compile_model / TransformationsPipeline),
reusing the plugin's high-performance kernels (oneDNN matmul, kernel_selector OCL
kernels) and kernel selection logic inside ggml's eager execution model.

## Locked decisions

1. **No dynamism.** Static shapes per bucket; network cache keyed by full static
   shape signature (op + shapes + dtypes + params). A new shape bucket = a rebuild.
   Slow cold start (JIT, network builds) is accepted.
2. **Weights baked as cldnn `data()` nodes.** Verified in OV source that the
   oneDNN FC/gemm path consumes weights in-place (fixed plain format tags,
   zero-copy `get_onednn_memory` wrap; `fully_connected_onednn.cpp:112-131`,
   `ocl_memory.cpp:207-216`), so N networks sharing one weight tensor keep exactly
   1 weight copy. The per-primitive_inst `_reordered_weights_cache` repack cost
   exists only on the ocl kernel-selector path and is bounded by bucket count.
3. **Fallback** via `supports_op=false` -> ggml scheduler splits graph to CPU.
4. **Memory**: backend `alloc_buffer` allocates through the cldnn engine (USM
   device); `tensor->data` wraps zero-copy into `cldnn::memory` via
   share_usm/attach_memory; outputs bound via `network::set_output_memory`.
5. **Skeleton**: copy ggml-cann (complete GPU iface + external-lib CMake pattern),
   NOT the legacy ggml-openvino backend.

## Architecture tiers (A is bring-up, B/C are end-state)

- **Tier A - per-op eager**: one single-primitive cldnn network per ggml op,
  cached. Exercises buffers, binding, layout mapping, op translators. No fusion.
- **Tier B - split-graph capture**: translate each scheduler split (maximal run of
  supported ops) 1:1 into one cldnn topology; weights as data() nodes; split I/O
  bound explicitly; intermediates owned by cldnn memory pool. Gets fusion, layout
  opt, single JIT session. Rebuild per shape bucket.
- **Tier C - hybrid**: Tier B + per-op fallback for unmappable views/aliasing +
  `mutable_data` pinning for KV cache writeback.

Pin only: split inputs/outputs and cross-graph-lifetime tensors (KV cache).
Do NOT pin every intermediate (kills fusion and pool reuse).

Known friction points:
- ggml arbitrary strides/permutations vs cldnn padded/blocked layouts; contiguous
  views are trivial, non-contiguous need explicit reorder/slice primitives.
- Rebuild-per-bucket cost: optimizer pass time per split + JIT (amortized by
  kernels_cache / cl_cache_dir). Measure early.
- Shape-dependent failure inside a captured split fails the whole split;
  supports_op must mirror translator capability.

### Attention: SYNC POINT (deferred to when we reach SDPA/KV)

Do NOT assume mapping onto cldnn scaled_dot_product_attention yet. llama.cpp does
not use paged attention; it uses view-based KV-cache slicing. Two paths, decide later:
- (a) report GGML_OP_FLASH_ATTN_EXT unsupported -> scheduler decomposes to
  MUL_MAT + SOFTMAX + MUL_MAT. Attention "just works" with zero attention-specific
  code once those three ops exist. Good first working path.
- (b) implement FLASH_ATTN_EXT natively onto cldnn scaled_dot_product_attention
  primitive - better perf, but must reconcile paged-attention semantics with
  llama.cpp's view-based KV. Separate decision when we get there.
The OV KV-cache primitive is designed around paged attention and likely does not
fit llama.cpp's model directly; do not assume it.

## Milestones (user-defined sequence)

test-backend-ops is verified continuously along the way: every op is validated
there as soon as it is implemented, in every milestone, before relying on it in
model-level tools.

1. **M1 - dense fp16/bf16**: llama-simple, llama-bench (varying context depth),
   llama-perplexity. Op set: MUL_MAT (fully_connected), ADD/MUL, RMS_NORM (rms),
   ROPE (rope), SDPA (scaled_dot_product_attention), SOFT_MAX, GET_ROWS,
   CPY/SET_ROWS (KV cache), views/reshape as no-ops or layout ops.
2. **M2 - dense quantized**: same three tools; quantization scheme TBD (ggml block
   quants do not map to OV u4/i4; options: dequant at load, custom dequant op,
   or ggml extra-buft-style load-time conversion - separate discussion).
3. **M3 - MoE + hybrid**: MoE (e.g. gemma4) via gather_matmul; hybrid (qwen35)
   incl. gated_delta_net primitive (exists in GPU plugin).

## Cache behavior, performance, and memory: expectations + verification

### Artifact/kernel cache

Expected:
- Artifact cache key: (op, dtypes, full static shape signature, params). Shared
  across layers with identical dims - artifact count scales with distinct shape
  signatures, not model size.
- Decode reuses artifacts across 256-aligned KV windows; prefill uses actual
  prompt length (chunked at n_ubatch only when prompt > n_ubatch).
- First build per unique kernel pays OpenCL JIT (cold start, accepted); in-memory
  kernels_cache is per-program, on-disk cl_cache_dir dedups compile time across
  process runs.
- Steady-state decode: zero network builds, zero JIT, only bind+enqueue. Any
  rebuild during decode indicates a bucketing failure.

Verification:
- Instrument artifact cache with counters (lookups/hits/misses/builds/JIT) behind
  a debug env var (GGML_OVGPU_CACHE_STATS); assert zero builds per decode step
  after warmup.
- llama-bench at increasing context depths to observe KV-bucket growth
  (~ctx/256 artifacts) and confirm bounded artifact count.

### Performance

Expected:
- Slow first prompt/token (JIT + network builds) - accepted by design.
- Warm decode: per-op host overhead is bind+enqueue only; should be comparable
  to other ggml GPU backends.

Verification:
- llama-bench pp/tg at several context depths; compare against CPU backend and
  (reference only) the legacy OV backend.
- Per-op timing instrumentation (GGML_OVGPU_PROFILING) to isolate dispatch
  overhead vs kernel time; OV GPU plugin profiling env vars for kernel-level view.

### Memory (host and GPU)

Expected:
- GPU: ~1x weights (baked data() nodes wrap the ggml weight buffer in-place;
  no hidden second copy like the legacy backend) + KV cache + ggml compute
  buffers + artifact binaries (tens of MB typical session).
- Repacked weight copies only on the ocl kernel-selector path; zero on the
  oneDNN FC/gemm path.
- Host: artifact CPU-side structures (KBs each; hundreds of artifacts -> MBs).
- Risk: artifact proliferation at very long context (128k -> ~512 KV buckets);
  mitigated by an LRU cap on the artifact cache, sized from measurements.

Verification:
- Runtime measurement via engine::get_used_device_memory(allocation_type) and
  get_memory_statistics(), exposed behind GGML_OVGPU_MEM_STATS; compare against
  the estimates above from the M0 smoke test onward.
- Confirm no per-network weight duplication (oneDNN path) by tracking GPU memory
  as artifact count grows.

## Link surface (verified)

- Static libs in /home/zijun/dev/openvino/bin/intel64/{Release,Debug}:
  openvino_intel_gpu_graph + _runtime + _kernels (+ OpenCL, openvino::runtime,
  openvino::shape_inference, openvino::itt).
- Plus ~4-8 small src/plugin/*.cpp compiled in for stray symbols (list documented
  in intel_gpu/tests/unit/CMakeLists.txt:26-36: variable_state.cpp,
  multi_tensor_variable_state.cpp, common_utils.cpp, simple_math.cpp,
  remote_context.cpp, remote_tensor.cpp, usm_host_tensor.cpp).
- Do NOT enable legacy ggml-openvino in the same binary (symbol collisions with
  plugin .so).

## Key OV-side APIs (verified)

- engine: cldnn::engine::create(engine_types, runtime_types) (engine.hpp:181)
- memory: attach_memory / share_usm / share_buffer / allocate_memory /
  reinterpret_buffer / create_subbuffer (engine.hpp:53-96)
- network: set_input_data / set_output_memory / execute
  (include/intel_gpu/graph/network.hpp:117-164)
- primitives of interest: fully_connected (compressed + dyn-quant variants),
  gemm (indirect/beam-table), scaled_dot_product_attention (KV compression),
  gather_matmul, rms, rope, custom_gpu_primitive
- standalone usage precedent: intel_gpu unit tests, e.g.
  tests/unit/test_cases/fully_connected_gpu_test.cpp:101-127

## Open questions

- Quantization approach for M2 (deferred by design).
- oneDNN vs ocl FC impl selection per shape (force vs default heuristics).
- Runtime engine: OCL vs Level Zero default per build (leave default initially).
- Backend name: ggml-ovgpu?
