# STATUS - OVGPU backend

> Short snapshot of current state + immediate next steps. Detailed how/why: WORKLOG.md.
> Stable design: poc-ovgpu-plan.md. Updated as work progresses.

Last updated: 2026-08-07
Worktree: llama.cpp-ovgpu  |  Branch: poc-ov-gpu-backend (base: dev_backend_openvino)
Dev box: Lunar Lake (Core Ultra 5 236V, Arc Graphics / Xe2, 56 CUs, supports_immad=1)

**OP FUSION landed (Tier B-lite).** ovgpu was one single-primitive cldnn network per ggml
op, so cldnn's program-level fuser had nothing to merge. Now `build_chain` builds ONE cldnn
topology per fusible linear chain `[head, child1, ...]` and `program::fuse_nodes` merges the
children into the head at build time. Verified fusible on Lunar Lake (oneDNN FC DOES accept
eltwise/activation post-ops; rms fuses eltwise ungated): **`rms->MUL` (gamma)** and
**`FC->ADD` (residual)** - both form + fuse in real models. Result: **decode (tg) +10% mean /
+23% best** (24 vs 21.75 t/s, d=0); pp512 unchanged (compute-bound). Output **bit-identical**
fusion vs `GGML_OVGPU_FUSE_DISABLE=1`. `test-backend-ops` still 1869/1869. Diagnostics:
`GGML_OVGPU_FUSE_TRACE=1` (confirms fusion per chain), `GGML_OVGPU_FUSE_DISABLE=1` (per-op A/B).
FC chains gated to contiguous operands (strided-FC + post-op has a peer-format alignment bug
to investigate; falls back to per-op, correct). `FC->SILU` implemented (oneDNN eltwise_swish
post-op) but not triggered - llama's FFN uses `ggml_swiglu_split`, not decomposed silu+mul.
See WORKLOG pm15 + memory `ovgpu-op-fusion-multi-op-cldnn`.

**FLASH_ATTN_EXT works** - `test-backend-ops -o FLASH_ATTN_EXT` passes (714/714
supported cases, 0 fail, 0 crashes, clean exit). Root cause of the prior garbage +
hangs/crashes: the cldnn SDPA kernels (`sdpa_opt.cl`/`sdpa_ref.cl`) **ignore
`output_transpose_order` when writing** - they always write `OUTPUT_GET_INDEX(b, head,
seq, d)` (canonical BHSD coords) into the physical output buffer, with no output
dims-order remap. The translator passed `output_transpose_order={0,2,1,3}`, so shape
inference allocated a `(B,S,H,D)` buffer that the kernel filled with `(B,H,S,D)` coords
-> transposed garbage (H==S_q) / OOB writes (H!=S_q, e.g. single-token S_q=1 with H
heads overflowed ~H× -> the sentinel overflow, corruption, hangs). Fix: identity output
order + explicit `cldnn::permute({0,2,1,3})` + reorder into ggml's BSHD dst (real OV
flows keep output order identity too - `TransposeSDPAFusion` fuses only q/k/v input
transposes). Inputs use the strided-binding path (physical BSHD + per-input `{0,2,1,3}`
transpose order, zero-copy via per-dim pitches; cast-only reorder for f16->f32). No impl
forcing (the `force_implementations` kernel-name is ignored for ocl_v2 impls; sdpa_micro
is a stage inside SDPAOptImpl, auto-gated). One residual kernel bug gated to CPU:
mismatched K/V head sizes (hsk!=hsv, MLA: DeepSeek 576/512, Mistral4 320/256) segfault
the sdpa_opt multi-token kernel on large prefill (192/128 is safe, kept on GPU). See
WORKLOG + memory `ovgpu-flash-attn-sdpa-mapping`.

Full-suite regression (no `-o` filter, **including FLASH_ATTN_EXT**): **1869/1869
pass, 0 fail, 0 crashes** (single-threaded; `-j` is unsafe for this backend - races on
the shared stream). Every OTHER supported case passes (100%). All ROPE modes
(NORMAL/NEOX/MROPE/IMROPE/VISION), YaRN, ROPE_BACK; SET_ROWS cross-type (f32<->f16
KV-cache writes) + padded-view v=true; SOFT_MAX mask/ALiBi/sinks/scale; CONCAT
(KV-cache append) + GLU SwiGLu + UNARY SILU (custom kernels) - added to cut graph
splits.

## Real-model perf (llama-simple, Llama-3.2-1B-Instruct-F16, Lunar Lake)

| metric | OVGPU | CPU | ratio |
|--------|-------|-----|-------|
| model load | 0.7 s | 2.4 s | OVGPU 3.4x faster |
| prompt eval (batch) | 80-94 tok/s | 9.4 tok/s | **OVGPU 8.5-10x faster** |
| single-token eval | ~39 ms/tok | ~40 ms/tok | parity (OVGPU slightly ahead) |

- **Weight load fixed**: was 113 s (mem_lock<read_write> over the whole ~2GB model
  buffer per tensor, incl. per-token logits readback). Now `copy_from`/`copy_to`
  (O(tensor size), not O(buffer)) -> 0.7 s. This also cut eval 112->47 ms/tok
  (the per-token logits get_tensor was the hidden cost).
- **Per-op `finish()` removed**: all op networks now share ONE in-order stream
  (`allocate_network(shared_stream, program)` + `queue_type=in_order`). cldnn's
  default queue is out-of-order (the trap: no-finish on out-of-order gave wrong
  results in fused multi-op graphs). In-order queue orders kernels across
  networks for free. Cut single-token 47->~39 ms/tok.
- **Single-token at parity = the remaining win**: the graph still splits 66/582
  nodes - **FLASH_ATTN_EXT** (attention) runs on CPU with GPU<->CPU copies per
  layer. Prompt eval is 8.5x faster (batch amortizes the matmul), so the GPU
  matmul path is fine; attention is what caps single-token at parity. **Next
  lever: FLASH_ATTN_EXT on GPU** (native cldnn SDPA, or decomposed
  MUL_MAT+SOFT_MAX+MUL_MAT - parts already supported). See `ovgpu-debug` skill +
  WORKLOG "Perf".

## Builds (green)

- **OpenVINO**: `openvino/` built minimal with Ninja+ccache. GPU plugin + oneDNN-GPU
  + core only (`ENABLE_INTEL_CPU/NPU=OFF`, frontends/tests/samples off,
  `ENABLE_DEBUG_CAPS=ON`, `GPU_RT_TYPE=OCL`, `ENABLE_ONEDNN_FOR_GPU=ON`). Archives
  in `bin/intel64/Release/` (matches ggml-ovgpu CMakeLists defaults; no OV install
  step; `OpenVINO_DIR` -> build-tree config).
  - Build fix: `-DCMAKE_DISABLE_FIND_PACKAGE_RapidJSON=ON` (use OV's bundled rapidjson;
    system 1.1.0 has a GCC-14 const-assignment bug in auto_tuner.cpp).
- **llama.cpp**: `cmake --preset x64-linux-ovgpu-release` (Ninja+ccache). Targets
  test-backend-ops / llama-simple / llama-bench built.
  - Preset fix: repointed `OpenVINO_DIR`/`GGML_OVGPU_OV_SOURCE_DIR` to real paths.
- Runtime: `LD_LIBRARY_PATH` needs OV (`bin/intel64/Release`) + TBB
  (`temp/Linux_x86_64/tbb/lib`).

## Done

- **M0**: skeleton (reg/device/buffer_type/buffer/stream ifaces), clDNN standalone
  engine init (no ov::Core), FC smoke test.
- **M1 MUL_MAT -> fully_connected**: **116/116 supported cases pass**, 0 fail, clean exit.
  - oneDNN FC engaged on Lunar Lake (config: `optimize_data(true)` + `use_onednn(true)`,
    NO `allow_new_shape_infer` - that flag breaks the rank-2 reshape path).
  - **Native f16/bf16 weights** via downcast-act: activation reordered to weight dtype,
    oneDNN f16/bf16 matmul accumulates f32 (matches ggml). Native weight read, no
    per-token repack. Verified (OVGPU_FC_TRACE: `in0=f16 wei=f16 out=f32`, 208x
    f16-w+f32-a, 149x bf16-w+f32-a).
  - Mixed storage-class dtypes -> both-to-f32 (oneDNN f32f32, exact). Non-immad devices
    -> both-to-f32 (ocl bfyx_ref, correctness fallback).
  - **Non-contiguous "padded view" operands now supported** (`k_v` test knob: a wider
    per-row `k` allocation sliced down via `ggml_view_4d`) via `ggml_layout_for_maybe_padded`
    (flat-batch shape with padding on the feature axis, then a compacting `reorder` -
    see WORKLOG "generalized non-contiguous support" entry). v=0/v=1-equivalent pairs
    now behave identically.
  - Deferred (supports_op=false -> CPU, NOT failures): grouped/GQA weights (ne[2]/ne[3]>1),
    permuted operands (unreachable anyway - every `per` test case is also grouped/GQA),
    quantized types (mxfp4/nvfp4 -> M2 compressed).
  - Shape mapping: ggml ne[0]->cldnn feature, ne[1..3] flattened->batch, spatials=1.
    Rationale documented in WORKLOG ("Design note: ... mapping rationale").
- **M1 ADD/MUL -> eltwise (sum/prod)**: **ADD 64/64, MUL 63/63 pass**, 0 fail. Broadcast-preserving
  layout (`ggml_layout_for_eltwise`, ne[0]->x, ne[1..3]->y/f/b unflattened) so NUMPY
  eltwise broadcast sees each ggml dim. Per-op `compiled_op::layout_for` strategy
  (flat for MUL_MAT, broadcast for eltwise). ggml repeat != NUMPY broadcast: gates to
  the NUMPY-broadcastable subset (tiling cases -> CPU). MUL_MAT still 116/116.
  **Non-contiguous permuted operands now supported** (`perm1` test knob: a genuine
  3-way axis rotation, `ggml_permute(ctx, b, 1, 2, 0, 3)`) via a new general
  "strided view" mechanism (`ggml_is_strided_view`/`ggml_layout_for_strided`/
  `ggml_strided_permute_order`): sorts a tensor's axes by actual stride, binds via
  that physical order, then a `cldnn::permute` relabels into the fixed logical
  convention before a compacting `reorder` (see WORKLOG entry - this is the general
  case of MUL_MAT/RMS_NORM's padded-view mechanism, which only handles pure
  sub-box views, not permutations).
- **M1 RMS_NORM -> rms (no gamma)**: **20/20 pass** (all eps incl. 0.1/10.0). cldnn rms
  reduces over ALL spatial dims, so a dedicated `ggml_layout_for_reduce_ne0` (ne[0]->x
  sole spatial, ne[1..3]->batch) makes the reduction = ne[0]. Non-contiguous "padded
  view" inputs (v=1) supported: `ggml_layout_for_padded` + compacting `reorder` +
  `reshape` (see WORKLOG) before `rms` (the reshape is essential - cldnn::reorder does
  NOT reshape, so without it rms_gpu_ref would reduce over ne0*ne1; see adversarial
  review finding). The earlier "eps>=0.1 fast-math divergence" was a misdiagnosis - the
  real cause was the op_cache key missing `eps` (two same-shape RMS_NORMs with different
  eps shared a wrong-eps network); fixed in key_for. v=0/v=1 behave identically.
- **M1 SOFT_MAX -> softmax / custom_gpu (212/212 pass)**. Two paths: (1) plain
  (no mask/sinks/ALiBi/scale==1) -> `cldnn::softmax` (reduce-ne0, fast, handles
  huge ne0); (2) mask/ALiBi/sinks/scale!=1 -> custom `ovgpu_soft_max` OpenCL
  kernel (faithful port of ggml-cpu softmax): `wp = src0*scale + slope*mask`;
  `max=max(wp)` (or `max(max,sk)`); `sum=Σexp(wp-max)` (or `+exp(sk-max)`);
  `dst=exp/sum`. Handles ggml's modular nr23 mask broadcast (`i02%ne12`,
  `i03%ne13`), per-head ALiBi slope (m0/m1/n_head_log2 host-computed), per-head
  sinks (src[2]), F16/F32 mask. Mask dim0/dim1 must match a (cpu indexes mask by
  i01 directly); sinks must be F32 [a.ne[2]]. **212/212 pass** (mask/ALiBi/sinks/
  scale, all combos). Two bugs found+fixed: `pow(M0,...)` ambiguous (bare -D
  decimal literal matches both float & double overloads on Intel -> cast
  `(float)M0`); and `key_for` broke at the first NULL src (sinks-only src[1]=NULL
  dropped src[2]=sinks from the key -> collided with the no-sink cldnn-softmax
  path -> sinks silently ignored). Fixed by `continue` past NULL srcs.
- **M1 GET_ROWS -> gather**: 2D-table row lookup (`table[indices]`), scope-gated to
  `ne[2]==ne[3]==1` (grouped/GQA table -> CPU, like MUL_MAT's weight deferral).
  `gather` -> `reshape` -> `reorder` pipeline. bf16 EXCLUDED (clDNN gather kernel
  selector has no bf16 kernel and aborts (SIGABRT) rather than failing gracefully -
  hardcoded exclusion, no way to probe ahead of time). **5/5 supported cases pass**
  (f32/f16, incl. a non-contiguous-indices `v=1` case).
- **M1 CPY -> reorder (copy + cast)**: `cldnn::reorder` from src[0]=a to the node (=
  view of src[1]=b); output bound in-place into b's memory. Same shape + bfyx =>
  pure dtype cast (or no-op copy). Types f32/f16/bf16/i32. Cast-DOWN to bf16
  excluded (cldnn bf16 cast rounds differently than ggml's RNE, NMSE ~1.5e-5 > 1e-6);
  same-type bf16 + cast-FROM-bf16 kept. **Strided/permuted src supported** (permute_src,
  _src_transpose): strided bind -> permute -> reorder (strip pad + cast) -> reshape to
  flat (reuses the ADD/MUL strided machinery); f16/f32 strided pass. Strided bf16 src
  gated out (cldnn permute/reorder mis-handles bf16+padding -> wrong results; offload
  to CPU). Strided dst (dst_alloc / permute_dst) still deferred. **CPY 49/49, 0 fail.**
- **M1 SET_ROWS -> custom_gpu (stride-aware, 86/86 pass)**. `scatter_update` couldn't
  express the batched per-group (per ne2/ne3) row indices, so a custom OpenCL kernel
  (`ovgpu_set_rows`) via `custom_gpu_primitive`. **Stride-aware**: src0 (updates) and
  src1 (indices) read via their ggml `nb[]` byte strides baked as `-D`, so a contiguous
  OR a padded/strided view (test `v=true`) is handled with NO pre-compaction - the
  kernel navigates the raw allocation directly. **Cross-type f32<->f16** writes
  (SRCFLOAT/DSTFLOAT, cast on write, RNE) - enables f32 updates -> f16 KV cache.
  Quantized dst (Q4_0/Q8_0/...; needs block quantization) + bf16 -> CPU. dst table
  always contiguous (flat-indexed in-place write). Batched + broadcast + i32/i64
  indices native. **86/86 pass** (v=0 + v=true, f32/f16 src/dst cross-type). Also
  fixed `GGML_OP_NONE` leaf to allow I64 (was only I32).
- **Fixed a latent bug affecting ALL ops**: `supports_op` had no case for
  `GGML_OP_RESHAPE`/`VIEW`/`PERMUTE`/`TRANSPOSE` (pure-metadata, no-compute nodes
  that test-backend-ops still queries supports_op on), so ANY op fed a
  view/permuted operand was wrongly rejected wholesale. Fixed by returning true for
  those four ops (matches every other ggml backend) + skipping them in
  graph_compute. Pass counts jumped 65->91 (MUL_MAT), 43->58 (ADD), ->57 (MUL),
  3->5 (GET_ROWS) once fixed.
- **M1 ROPE -> custom_gpu (ALL MODES, 134/134 pass)**. `build_rope` = a custom
  OpenCL kernel (`ovgpu_rope`) computing cos/sin inline from pos+freq_base (cldnn
  `rope` needs precomputed cos/sin; positions are runtime). Faithful port of
  ggml-cpu `ggml_compute_forward_rope_flt` (the test reference) - incremental
  `theta *= theta_scale`, NOT pow. **ALL modes**: NORMAL (interleaved) + NEOX
  (split-half) + MROPE + IMROPE (4 per-dim thetas t/h/w/e, interleaved sector
  mapping) + VISION (per-dim thetas with section-reset `indep_sects`). **YaRN**
  (ext_factor!=0: ramp blend + mscale) and **ROPE_BACK** (`SIN_SIGN=-1`, inverse
  rotation). op_params (n_dims, mode, n_ctx_orig, freq_base, freq_scale,
  ext_factor, attn_factor, beta_fast, beta_slow, sections[4]) all baked as -D
  and dumped into `key_for` (15 ints). **134/134 pass** (5 modes x {f32,f16} x
  {ff,no-ff} x {fs,ef,af combos} x {forward,back} x {v=0,1,2}). Root-cause bugs
  fixed (see WORKLOG): (1) `key_for` missing op_params -> wrong-constant kernel
  reuse; (2) freq_factors read as DFLOAT* (half for F16) -> NaN, fixed to F32*;
  (3) float constants serialized via `std::to_string` (6 digits) -> theta_scale
  drifted ~2.6e-5 over n_dims/2 multiplies, intermittently flipping F16 rounding
  -> fixed `fstr()` (%.9g, exact round-trip), which ALSO removed the F16+ff+fs!=1
  precision gate (now exact, no gate needed). Scope: contiguous src0 (strided/
  padded views -> CPU; a stride-aware port is the follow-up, done for SET_ROWS).
- Instrumentation: `OVGPU_FC_TRACE=1` env var (gated, in OV source: oneDNN FC
  validate/create dims + find_impl). Off by default.

## Immediate next steps

1. **OP FUSION landed (Tier B-lite)** - `rms->MUL` + `FC->ADD` chains fuse via
   `program::fuse_nodes`; decode +10-23%. See top of file + WORKLOG pm15. Next decode
   lever: **decompose `ggml_swiglu_split`** (the FFN GLU) into `SILU`+`MUL` at translation
   so `FC->SILU` (already implemented) + `FC->SILU->MUL` chains form - that hits the GLU
   decode bottleneck directly (the FFN gate/up matmuls + the swiglu custom kernel are a
   known split). Also: investigate the strided-FC + post-op peer-format alignment so FC
   chains can drop the contiguous-operand gate.
2. **FLASH_ATTN_EXT FIXED** (was the active blocker - garbage llama-simple output, test
   failures). Root cause: cldnn SDPA kernels ignore `output_transpose_order` on write
   (see top of file). Fix = identity output order + explicit permute/reorder; strided
   input binding; no impl forcing. Residual: hsk!=hsv (MLA, large heads) gated to CPU
   (sdpa_opt segfaults). `test-backend-ops -o FLASH_ATTN_EXT` = 714/714 pass; full
   suite = 1869/1869. `llama-simple` (Llama-3.2-1B-Instruct-F16) verified: coherent
   output, 25 t/s decode, 30 graphs reused.
3. All other M1 ops (MUL_MAT/ADD/MUL/RMS_NORM/SOFT_MAX/GET_ROWS/CPY/SET_ROWS/ROPE/
   CONCAT/GLU/UNARY) are done and passing 100%; not blocking.
4. **Other llama tools VERIFIED** (was the next step):
   - **llama-bench** at depths: `pp512/tg128` at d=0 (3875 / 30.8 t/s), d=1024
     (1768 / 27.2), d=4096 (762 / 16.8) - all pass.
   - **llama-perplexity** (wikitext-2, `-b 512 -c 2048` -> 4 ubatches/chunk, exercises
     the multi-ubatch path): **Final PPL = 12.77 ± 0.30**, no crash.
   - **llama-server `-np 4`**: 4 slots, single + 2 parallel `/v1/chat/completions`
     both coherent ("Red, Blue, Yellow"; "1. 2. 3. 4. 5."), clean exit. `-np 4` is safe
     (parallel decode is batched into one `llama_decode` -> one `graph_compute`, not
     concurrent backend calls - so the `-j` shared-stream race does not apply).
   - **Full test-backend-ops regression**: 1869/1869, 0 fail (unchanged).
   - **Multi-ubatch crash fixed** (see WORKLOG pm14): decodes split into >1 ubatch
     (depth>512 in llama-bench, or any `-b` < chunk ctx) failed with `llama_decode -3`.
     Root cause: non-final ubatches have `n_outputs==0` -> the last-layer
     `ggml_get_rows(cur, inp_out_ids)` gets an `ids` with `ne[0]==0`; cldnn's `reshape`
     rejects zero-count buffers ("output layout count ... not equal to input count(=0)")
     and throws -> `GGML_STATUS_FAILED`. Fix: skip zero-element-output nodes in
     `graph_compute` (writes nothing; matches the CPU backend). Plus error-visibility:
     build exceptions now `fprintf` to stderr (tools like llama-bench install a null
     ggml log callback that swallowed `GGML_LOG_ERROR`), and `GGML_OVGPU_NO_CACHE=1`
     bypasses the op cache for diagnosing cache-key bugs.

## Later

- **M2 quantized**: ggml block quants -> oneDNN FC `compressed_weights` (u4 + scales
  folded into matmul). Needs `data()` weights (per-layer cache). + the cache-size
  LRU cap (memory budget differs B580 12-16GB vs Flex 16GB vs Max 128GB).
- **PVC/XeHPC datacenter**: OV build throws on XeHPC (ngen), OV-version issue.
- **Verify on B580** when accessible (same Xe2 arch, should Just Work; smoke test
  is the canary).

## Key facts (don't re-derive)

- `supports_immad` = DPAS matrix engine present + oneDNN codegen exists. True on
  Arc A/B, Lunar Lake, Panther Lake, Flex, Max. False on old XeLP/MTL iGPUs.
  See plan "Device-class / DPAS target matrix".
- oneDNN FC dense path accepts ONLY matching-dtype pairs (f16f16/bf16bf16/f32f32).
  Mixed -> downcast act (preferred) or both-to-f32 (fallback). Quantized is a
  separate `compressed_weights` path.
- `data()` weights NOT needed for dense f16 correctness/perf (input_layout works).
  Needed for compressed/q4 + optimal weight layout.
- Do NOT set `allow_new_shape_infer(true)` - breaks oneDNN FC shape inference
  (feeds raw rank-4 to MatMul, fails batch broadcast).
- Layout strategies via `compiled_op::layout_for`: flat (MUL_MAT), eltwise/broadcast
  (ADD/MUL), reduce-ne0 (RMS_NORM/SOFT_MAX), reduce-ne0-maybe-padded (RMS_NORM's
  non-contiguous "padded view" path - see WORKLOG). Pick per op based on how cldnn
  reduces/broadcasts vs ggml.
- `cldnn::tensor`'s two constructors have DIFFERENT dim-order semantics, and so does
  `cldnn::padding`'s lower/upper vectors: simple ctor `tensor(b,f,x,y)` vs
  format-order `tensor(format,sizes)`/`padding(lower,upper)` which for bfyx is
  (b,f,y,x) - an X/Y swap relative to the simple ctor. Bit us on two different ops
  (GET_ROWS' `output_shape`, RMS_NORM's padded-view padding). Verify against an OV
  unit test, don't assume.
- `supports_op` must explicitly allow `GGML_OP_RESHAPE`/`VIEW`/`PERMUTE`/`TRANSPOSE`
  (return true, no compute) - test-backend-ops queries supports_op on every tensor in
  the graph, not just compute nodes, so missing this silently rejects any op fed a
  view/permuted operand.
