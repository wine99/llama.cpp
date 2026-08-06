# PoC: dedicated OpenVINO GPU backend for ggml (eager, clDNN-level)

Status: M1 in progress on Lunar Lake - MUL_MAT, ADD/MUL, RMS_NORM, SOFT_MAX (basic),
GET_ROWS all landed and passing (see WORKLOG for per-op pass counts). M0 + build
wired. Detailed log: WORKLOG.md, short status + next steps: STATUS.md. Worktree:
llama.cpp-ovgpu, branch poc-ov-gpu-backend (base: dev_backend_openvino).

MUL_MAT: 116/116 supported cases pass (f32/f16/bf16 weights x f32/f16/bf16 act, weight
2D, batched activation, incl. non-contiguous `k_v` sub-box views). The clDNN **oneDNN**
FC path is engaged and correct on Lunar Lake (Xe2, supports_immad=1): f16/bf16 weights
run NATIVE via downcast-act (activation reordered to the weight's dtype; oneDNN
f16/bf16 matmul accumulates in f32 = matches ggml). 0 failures, clean exit, oneDNN FC
selected (OVGPU_FC_TRACE). Mixed *storage-class* dtypes fall back to both-to-f32
(oneDNN f32f32); non-immad devices keep both-to-f32 (ocl bfyx_ref). Grouped-weight
(GQA) / quantized deferred (M2). ADD/MUL/RMS_NORM/GET_ROWS now also support their
non-contiguous test variants (permuted operands for ADD/MUL, padded sub-box views
for RMS_NORM/MUL_MAT/GET_ROWS) - every `v=0`/`v=1`-equivalent pair behaves
identically; see WORKLOG's 2026-08-03 entry for the general "strided view"
mechanism. Builds: OV (minimal GPU+onednn-gpu+core, Ninja+ccache) + llama.cpp (ovgpu preset).

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

## Device-class / DPAS target matrix

The backend's perf path is gated on `supports_immad` - the cldnn device flag meaning
"this GPU has a hardware matrix engine (XMX/**DPAS**) AND oneDNN has a code generator
for it." Set from OpenCL `CL_DEVICE_FEATURE_FLAG_DPAS_INTEL` (OCL rt) or the L0 DPAS
module flag, **and** ngen recognizing the arch family. DPAS does f16×f16->f32 (and
i8×i8->i32) accumulate in hardware - this is why the native f16 path accumulates in f32
matching ggml. The cldnn device table (device.cpp:97-109) per family:

```
gfx_ver          arch        DPAS(f16/i8)   device_id                supports_immad
12.0-12.9        XeLP        -              TGL/RKL/ADL iGPU         NO  (old client iGPU)
12.55-12.57      XeHPG(DG2)  128/256        Alchemist (Arc A)        YES
12.70-12.71      XeHP        -              MTL/ARL-S iGPU           NO  (old client iGPU)
12.74            XeHPG       128/256        ARL-H                    YES
20.1-20.2        Xe2 (BMG)   128/256        Arc B580/B570 (Battlemage) YES
20.4             Xe2 (LNL)   128/256        Lunar Lake (current dev)  YES
30.0-30.1        Xe3/PTL     128/256        Panther Lake             YES
```

**Target**: functional + performant on all DPAS-capable client AND server GPUs:
- Client: Arc A (Alchemist), Arc B (Battlemage, e.g. B580), Lunar Lake, Panther Lake.
- Server: Data Center GPU Flex (Arctic Sound), Data Center GPU Max (Ponte Vecchio).
All of these are DPAS-capable -> `supports_immad=1` -> identical oneDNN FC path
(downcast-act native f16). The backend needs no per-device special-casing.

**Perf path**: supports_immad -> oneDNN FC, native f16/bf16 via downcast-act.
**Correctness safety net**: !supports_immad (old XeLP/MTL iGPU) -> both-to-f32,
ocl bfyx_ref scalar kernel. Correct, slower. Acceptable: no server GPU we target is
non-DPAS; this only affects legacy client iGPUs.

**Caveat - XeHPC/Ponte Vecchio**: the OV/clDNN build in this worktree THROWS on
`ngen::HW::XeHPC` (`ocl_device.cpp:63`: "XeHPC is not supported") at arch detection,
even though the oneDNN engine lists PVC as supported. An OV-version issue, NOT the
backend's - a PVC-aware ngen/OV build would handle it. Separate investigation if
datacenter PVC is a hard requirement; Arc/Flex/Max (non-XeHPC) are unaffected.

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
   llama-perplexity. Op set: MUL_MAT (fully_connected) [done], ADD/MUL [done],
   RMS_NORM (rms) [done], SOFT_MAX (basic case) [done], GET_ROWS (gather) [done],
   ROPE (rope) [deferred - see WORKLOG], SDPA (scaled_dot_product_attention)
   [sync point, see below], CPY/SET_ROWS (KV cache), views/reshape as no-ops
   [done - VIEW/RESHAPE/PERMUTE/TRANSPOSE all return true/no-op in supports_op
   and graph_compute].
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

## Custom-kernel strategy: custom_gpu + ggml-opencl kernel reuse

For ops that don't map cleanly to a cldnn primitive, inject a custom OpenCL kernel
via `cldnn::custom_gpu_primitive` (same cldnn engine + USM memory - no runtime mix).

**API** (`custom_gpu_primitive.hpp`): `(id, inputs, kernels_code[], entry_point,
kernel_arguments[], build_options, output_layouts[], gws, lws)`. `kernels_code` is a
vector of source strings; cldnn JIT-compiles via `clBuildProgram`. `kernel_arguments`
binds `arg_input`/`arg_output` (memory pointers) / `arg_internal` (scratch buffer,
sized by `size_expr`). **There is NO scalar-argument mechanism** - verified in
`custom_primitive.cpp:206-220` (`arg_desc` is only input/output/internal-buffer).

**The binding gap + bridge**: the ggml-opencl kernels (`kernels/*.cl`) are *generic* -
they take ~18 scalar args (dims + byte strides `nb*`). cldnn's binding passes only
memory pointers. Bridge:
1. **Bake every scalar as a `-D` build option** (`-DR=.. -DNE0=.. -DNE1=.. ...`). OK
   because the backend's op_cache is shape-keyed (one compiled network per shape) and
   cldnn's `kernels_cache` caches the compiled binary by (source + build_options).
2. **Force a contiguous layout**: the kernel receives raw pointers, so it can't know
   padding/strides. Bind via `ggml_layout_for` (flat) and **pre-compact** any
   non-contiguous input with a `cldnn::reorder` before the custom node (same pattern
   MUL_MAT/RMS_NORM/eltwise/CPY already use). Contiguous inputs (the common case) are
   zero-copy (raw pointer into the existing buffer) - no reorder, no overhead.

**Cost of pre-compaction**: transient device buffer + O(N) bandwidth, but ONLY for
non-contiguous views; contiguous inputs pay nothing. Compact-then-dense is usually
*faster* than a strided kernel (coalesced reads). `optimize_data(true)` may fuse/merge
reorders across the topology.

**Reuse, not rewrite**: the ggml-opencl kernels are OpenCL = cldnn's runtime, so the
kernel *logic* is ported verbatim; only the argument plumbing changes (scalars -> `-D`,
assume contiguous). Confirmed ready kernels: `set_rows.cl` (f32/f16 + i32/i64 + q4_0/
q8_0), `rope.cl`, `flash_attn_f16.cl` (which **does** handle ALiBi `max_bias` +
`logit_softcap` + mask - so ALiBi/softcap attention can stay on GPU via custom_gpu
instead of ggml-cpu). Embed the (adapted) source as a C++ string literal; cldnn JITs it.

**Ops this applies to**:
- **SET_ROWS**: batched per-group row scatter - `scatter_update` can't express per-group
  indices (uniform indices across non-axis dims). custom_gpu + a ~20-line contiguous
  kernel handles all batched/broadcast cases natively (the kernel does `i2%NE11`).
- **ROPE** with YaRN/ext-factors: cldnn `rope` needs precomputed cos/sin (positions are
  runtime, so can't be a static `data()` table). custom_gpu + `rope.cl` computes cos/sin
  inline from freq_base+positions, matching ggml exactly (all modes + YaRN). Base
  NEOX/NORMAL could also use cldnn `rope` + a cos/sin precompute graph.
- **FLASH_ATTN_EXT** ALiBi/softcap: cldnn SDPA has no ALiBi/softcap. custom_gpu +
  `flash_attn_f16.cl` (which has `get_alibi_slope` + `logit_softcap`). Base case uses
  native cldnn SDPA (is_causal or ATTN_MASK input; sinks supported natively).

## Caching layers: cldnn kernels_cache vs backend op_cache

Two complementary caches at different layers (do NOT collapse them):

- **cldnn `kernels_cache`** (`impls/ocl/kernels_cache.hpp`): per-engine cache of
  *compiled OpenCL kernels* (`_cached_kernels`, `_cached_binaries`, disk cache). The
  expensive `clBuildProgram` JIT is memoized globally per engine - re-creating a network
  that uses the same kernel is a lookup, not a recompile.
- **backend `op_cache`** (`ggml-ovgpu-ops.cpp`): caches the *constructed `cldnn::network`*
  + backend metadata (`layout_for`, input/output ids), keyed by op+types+ne+nb. cldnn
  does NOT cache network construction (program build + shape inference `calc_output_layout`
  + impl selection `create()` + `layout_optimizer` + `primitive_inst` alloc) - that runs
  every `cldnn::network(engine, topo, config)`. The backend cache amortizes it across
  repeated shapes (transformer blocks) + tokens; on a hit only bind+execute remain.

Keep the backend cache (complementary, not redundant); its only cost is memory (held
networks) -> add an LRU cap (byte/entry budget) to bound growth (later task). A future
bigger step: build one cldnn network per ggml *subgraph* (let cldnn fuse across ops),
which would shift caching to the graph level.
