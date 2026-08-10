# WORKLOG - OVGPU backend (eager clDNN PoC)

Branch: poc-ov-gpu-backend (worktree llama.cpp-ovgpu, base dev_backend_openvino)
Plan: poc-ovgpu-plan.md (high-level reference). Status snapshot: STATUS.md.

This log records **decisions, things tried, and lessons learned** — not line-by-line
diffs (those live in git). Bugs are kept because the *root cause + why it was
misdiagnosed* is the reusable part.

## What this backend is

A ggml backend that drives Intel clDNN (the OpenVino GPU plugin's impl layer)
**directly** — `cldnn::engine`/`topology`/`network`, eager single-op networks, **no
`ov::Core`/`ov::Model`**. Each ggml op node -> a hand-built cldnn topology ->
`cldnn::network(engine, topo, config)` -> bind caller memory -> execute. A
shape-keyed `op_cache` stores the built `network` (+ `layout_for` metadata) so
repeat shapes skip topology construction. Device: Lunar Lake (Xe2, 56 CU,
`supports_immad=1`); earlier work on UHD 770 (`supports_immad=0`).

## M0 — skeleton proved clDNN is drivable standalone

- `engine::create(engine_types::ocl, runtime_types::ocl)` via `device_query`,
  no `ov::Core`. Hand-built `topology{input_layout, data, data, fully_connected}`
  -> `network` -> `set_input_data` -> `execute` -> `get_output_memory` matched a
  CPU reference. Backend registers, llama.cpp assigns layers to `OVGPU0`.
- **GPU buffer model MUST use USM device memory** (not `cl_mem`): `cl_mem` is a
  handle, not a pointer, so `get_base` can't return it for ggml's offset
  arithmetic. `usm_device` gives a real pointer. This iGPU has **no host-accessible
  `usm_shared`**, so host access goes through the runtime (no direct memcpy to the
  USM ptr). **This is the root of the slow weight-load bug** (see Perf section).
- Cross-stream sync: `set/get_tensor` use `reg_ctx->stream`; `network::execute`
  uses the network's own stream. `reg_ctx->stream->finish()` before compute +
  `net->get_stream().finish()` after each execute (coherent across streams on one
  context). **The per-op `finish()` is a perf limiter** (see Perf section).

### Link surface (reference, in case of rebuild)

Consuming clDNN standalone needs ~13 SYSTEM include dirs (intel_gpu public +
`src` for ocl_engine, onednn_gpu_install for generated `dnnl_config.h`, several
transitive OV dirs, bundled clhpp/cl_headers to avoid system cl2.hpp clash) +
compile defines matching the graph lib's ABI (`ENABLE_DEBUG_CAPS`,
`GPU_DEBUG_CONFIG=1`, `OV_GPU_WITH_OCL_RT=1`, `OV_THREAD=OV_THREAD_TBB_ADAPTIVE`,
...) + stray glue sources (plugin `variable_state`/`remote_context`/... + op
definitions for vtable/shape_infer symbols) + static libs
(`openvino_intel_gpu_{graph,kernels,runtime}`, `libopenvino_onednn_gpu.a`,
`openvino_{shape_inference,itt,reference,util,shutdown}`, pugixml) in
`-Wl,--start-group/--end-group`. OV built minimal (`ENABLE_INTEL_CPU/NPU=OFF`,
frontends/tests off, `ENABLE_DEBUG_CAPS=ON`, `GPU_RT_TYPE=OCL`,
`ENABLE_ONEDNN_FOR_GPU=ON`). Two build fixes needed: repoint preset
`OpenVINO_DIR`/`OV_SOURCE_DIR`; `-DCMAKE_DISABLE_FIND_PACKAGE_RapidJSON=ON`
(system 1.1.0 has a GCC-14 const-assignment bug; use OV's bundled copy).

## Design note: ggml ne[] -> cldnn bfyx mapping (reference)

`ggml_ne_to_tensor`: `batch = ne[1]*ne[2]*ne[3]` (dims 1..3 flattened),
`feature = ne[0]`, `y = x = 1` -> `tensor(batch, feature, 1, 1)` (bfyx). Why:

1. **Contiguous contracted dim on `feature`.** ggml `ne[0]` is the stride-1
   *contracted* dim (K in MUL_MAT). cldnn FC / oneDNN matmul reason about
   `feature` as the named contracted axis (input `[B,K]`, weight `[N,K]`).
   Putting it on a spatial (y/x) would make it a conv-style image plane FC's
   weight logic isn't built around.
2. **Spatials forced to 1 to suppress the format optimizer.** Non-unit spatials
   get rewritten to BLOCKED formats (`b_fs_yx_fsv4`), which oneDNN FC
   `validate_impl` REJECTS (only plain bfyx/bfzyx/bfwzyx/any). With y=x=1 nothing
   to block-pack; layout stays plain bfyx. This is also why `optimize_data(true)`
   is needed (keeps bfyx instead of picking blocked output for some shapes).
3. **Outer dims fill batch.** For a contiguous tensor the ggml flat address
   `((i3*ne2+i2)*ne1+i1)*ne0+i0` = `b*f + f` under this mapping = bfyx row-major.
   **ZERO-COPY** bind (non-owning view over the ggml buffer, no repack).

Per-op **layout strategy** (stored in `compiled_op::layout_for`, a
`std::function` — graph_compute binds each src/output via it; the topology's
input_layout MUST match the bind layout):
- `ggml_layout_for` (flat-batch): MUL_MAT (matrix on b,f; matches FC
  `reshape_to_2d`), CPY contiguous, SET_ROWS/ROPE/SOFT_MAX-custom (flat-indexed
  kernels).
- `ggml_layout_for_eltwise` (ne0->x, ne1->y, ne2->f, ne3->b, per-dim): ADD/MUL
  (NUMPY eltwise broadcast aligns from the trailing dim; ggml broadcasts along
  ne[0]=contiguous, so ne[0]->x makes a ggml broadcast become a NUMPY b.x=1).
  Flattening would merge a broadcast dim into a non-1 dim -> "shapes
  inconsistent" crash.
- `ggml_layout_for_reduce_ne0` (ne0->x sole spatial, y=z=1, ne[1..3]->batch,
  feature=1): RMS_NORM/SOFT_MAX-plain. cldnn `rms`/`softmax` reduce over ALL
  spatial dims (x,y,z), so ne[0] must be the *only* non-trivial spatial.
- `ggml_layout_for_padded` / `_strided` (per-dim + `cldnn::padding` for the gap
  between used and full extent): non-contiguous view operands (see Views below).

## Key decisions

- **Custom OpenCL kernels via `custom_gpu_primitive` for ops cldnn primitives
  can't express** (SET_ROWS per-group scatter, ROPE inline cos/sin, SOFT_MAX
  mask/ALiBi/sinks). Pattern (validated, generalizes): bake scalars as `-D`
  build_options (custom_gpu has NO scalar-arg mechanism — only memory pointers +
  scratch buffers), force contiguous OR stride-aware flat layouts, bind raw
  memory pointers. Port the **ggml-cpu** op (the test-backend-ops reference),
  not the ggml-opencl kernel (cpu uses incremental theta, etc. — bit-closer).
- **oneDNN FC engagement** (Lunar Lake): config = `optimize_data(true)` +
  `use_onednn(true)`, and **do NOT set `allow_new_shape_infer`** (it routes FC
  output shape through `calc_output_layouts` plural, feeding raw rank-4
  `[B,K,1,1]` to `MatMul::shape_infer` without collapsing trailing-1s ->
  "dst:0 inconsistent with src:0"). `calc_output_layout` singular does
  `reshape_to_2d` for immad devices -> rank-2 matmul happy.
- **Native matching-dtype weights** on `supports_immad`: feed weight at its
  native dtype (f16/bf16/f32), downcast the cheap activation to match (one reorder);
  oneDNN f16/bf16 matmul accumulates f32 = ggml semantics. Mixed storage-class
  (f16 weight + bf16 act) -> both-to-f32 (oneDNN f32f32, exact but repacks weight
  per token). `data()` constant weights NOT required for correctness or native-f16
  perf (input_layout works); `data()`/per-layer cache is M2 (compressed/q4 path).
- **Views**: a "padded view" (sub-box of a larger contiguous tensor, dim0 packed,
  each dim's stride a whole multiple of the prior's used extent) maps to
  `cldnn::padding` per-dim (zero-copy). A "strided view" (genuine
  permute/transpose) binds in its actual physical stride order + `permute` into
  the fixed convention. **`cldnn::reorder` does NOT reshape** — it keeps the
  source's used tensor, adopts only the target's dtype/format/padding. So a
  reorder between two *different axis decompositions* of the same size silently
  produces wrong results (not an error) — bind in the SAME axis convention you
  reorder into. (Bite: first MUL_MAT k_v attempt bound per-dim then reordered to
  flat-batch; compiled clean, wrong results.)
- **ggml repeat != NUMPY broadcast**: ggml allows TILING (`a.ne[i] %
  b.ne[i]==0`); NUMPY needs a dim==1. eltwise/SOFT_MAX gate to the
  NUMPY-broadcastable subset; tiling -> CPU. (Custom SOFT_MAX kernel now handles
  ggml's modular nr23 mask broadcast directly, so that gate relaxed.)
- **op_cache `key_for`** must include (a) each src's `nb[]` (same ne[] but
  different strides = different topology) AND (b) **every op-param baked into the
  compiled network** (MUL_MAT transposed, ROPE all 15 params, RMS_NORM eps,
  SOFT_MAX scale/max_bias). This is the #1 bug class — see below.
- **`get_or_build` wraps `build_op` in try/catch**: a translator's network
  construction can throw (oneDNN/cldnn rejecting a shape `supports_op` couldn't
  predict) — previously an uncaught exception crashed the whole
  test-backend-ops process. Now degrades to a single-test FAIL.
- **`supports_op` must return true for `GGML_OP_NONE` (allow I32+I64 leaves),
  `RESHAPE`/`VIEW`/`PERMUTE`/`TRANSPOSE`** (pure metadata; test-backend-ops
  queries supports_op on every tensor, so rejecting these silently rejected any op
  fed a view). Skip them in graph_compute. This was a latent bug affecting EVERY
  op (pass counts jumped 65->91 MUL_MAT etc. when fixed).

## Bugs & root causes (the reusable lessons)

Grouped by the *class* of mistake — each recurred:

### Cache-key collisions (op_params / nb[] not in key)
- **ROPE**: 17/19 failures. Two same-shape ROPEs with different n_dims/mode/
  freq_scale reused one wrong-constant kernel. Presented as "structural" ERR
  (0.4-1.6) on tail/ff/fs/large-seq — looked like a kernel-logic bug; was a cache
  key bug. **Misdiagnosed for a full session** as a kernel formula error (a
  standalone element-by-element test passed, which pointed at state/caching).
- **RMS_NORM eps**: the long-standing "large-eps fast-math divergence" (eps=0.1/10
  failing, ERR scaling with eps) was **NOT fast-math** — it was eps missing from
  the key, so a large-eps case reused a small-eps network. I'd attributed it to
  `native_powr`/`native_sqrt` in the bfyx rms kernel for an entire session. The
  adversarial review flagged it "latent, doesn't manifest" — it DID manifest once
  the ROPE fix forced the realization. **Lesson: the op_cache persists across
  ops/cases; ANY op-param baked into the network MUST be in `key_for`.**
- **nb[]**: contiguous vs padded-view operands with same ne[] collided. Add nb[].
- **`key_for` broke at first NULL src**: `if (!src) break;` dropped later optional
  srcs. Sinks-only SOFT_MAX (src[1]=NULL, src[2]=sinks) dropped sinks from the key
  -> collided with the no-sink cldnn-softmax path -> sinks silently ignored (ERR
  ~0.003-0.008 = the sinks-term fraction). Fix: `continue`, scan all GGML_MAX_SRC.

### Float-constant precision loss (custom kernels)
- `std::to_string(float)` gives 6 digits -> `theta_scale` `0.865964353` ->
  "0.865964" (4e-7 rel error/step). Over n_dims/2=64 multiplies drifted ~2.6e-5,
  intermittently flipping F16 output rounding across the 1e-3 threshold (1-in-N
  flake on f16+fs!=1). **Fix: `fstr()` = snprintf("%.9g") = exact IEEE-single
  round-trip.** This also removed a conservative F16+ff+fs!=1 precision gate
  (theta now exact; only cos/sin ~1-ULP remains, inside threshold). Apply to ALL
  baked floats. **Lesson: when a custom-kernel result is *intermittently* off by a
  hair on F16, suspect the constant serialization, not the math.**

### cldnn API gotchas (each bit once)
- **`cldnn::tensor` simple-ctor `(batch, feature, x, y)` vs `cldnn::padding` /
  `gather output_shape` / `permute` order use DIFFERENT axis orderings.** The
  simple ctor is `(b,f,x,y)`; padding vectors and permute order are FORMAT order
  `(b,f,y,x)` (= ov::Shape). Easy to swap and get silently-wrong results or a
  broadcast-merge crash. 2-axis cases pass by coincidence (a 2-element perm is its
  own reverse); 3-axis cases crash. Verify against an OV unit test or a Python
  simulation before trusting.
- **`pow(M0, ...)` ambiguous** with bare `-D` decimal literals: matches both
  `pow(float,float)` and `pow(double,double)` on the Intel compiler (half-precision
  headers loaded) -> `CL_BUILD_PROGRAM_FAILURE` for ALL ALiBi SOFT_MAX. Fix:
  `pow((float)M0, ...)`. **cldnn's build log (`GPU_DEBUG_INFO`) is compiled out in
  Release OV** — get the real error with a standalone `clBuildProgram` test.
- **`ggml_type_to_cldnn(I64)` falls through to f32** — a compaction-via-reorder
  misreads 8-byte I64 as 4-byte f32. Added I64 case; or use a stride-aware kernel
  (SET_ROWS) that reads `char*` + casts, sidestepping reorder entirely.
- **bf16 + padding** is mis-handled by cldnn permute/reorder (f16/f32 strided work,
  bf16 strided gives wrong results, ERR ~1-2). Gate bf16 strided to CPU.
- **oneDNN FC requires MATCHING dtypes** (f16f16/bf16bf16/f32f32) — mixed storage
  class can't use the native path, falls to both-to-f32. (memory
  ovgpu-onednn-fc-requires-matching-dtypes.)
- **clDNN gather has no bf16 kernel** and aborts (SIGABRT) instead of failing
  gracefully — hardcode-exclude bf16 for GET_ROWS (no way to probe ahead).
- **cast-DOWN to bf16**: cldnn rounds differently than ggml's RNE (NMSE ~1.5e-5 >
  1e-6) — gate f32/f16->bf16 CPY to CPU; same-type bf16 + cast-FROM-bf16 kept.

### Binding / arg-index bugs (optional srcs)
- **Positional input binding**: `graph_compute` bound `src[k]` -> `input_ids[k]`,
  but for sinks-only SOFT_MAX `input_ids={"in_0","in_2"}` (mask skipped) so
  src[1]=NULL was handed to the sinks slot. Fix: walk src[0..], skip NULL, bind
  each present src to the next input_id. General — handles any op with optional
  middle srcs.
- **custom_gpu arg index hardcoded**: sinks arg was `{arg_input, 2}` always, but
  sinks-only has `cg_inputs={in_0,in_2}` (2 entries) -> index 2 OOR -> "input
  memory necessary" assert at execute. Fix: running counter (arg index = position
  in cg_inputs).
- **cache copy omitted `layout_for`**: cache hits bound with the flat default ->
  "Unexpected layout of input memory". Copy `layout_for` into the stored entry.

### Misdiagnosis from terminal buffering
- stdout (result lines) + stderr (debug/gdb) interleave unpredictably when piped,
  so a crash's line number in the raw log doesn't identify the triggering case.
  Re-run with `stdbuf -oL -eL` (line-buffered) before trusting which case crashed.
  (Bite: attributed a GET_ROWS bf16 crash to the adjacent `be1=7,v=1` case.)

## Things tried that didn't work (don't repeat)

- **cldnn `rope` primitive**: tightly coupled to OV's internal RoPE opset
  (precomputed cos/sin cache tables, `RoPE::Config`). ggml computes rotation from
  freq_base+positions at runtime -> can't be a static `data()` table. Use a custom
  inline-cos/sin kernel.
- **cldnn `scatter_update` for SET_ROWS**: applies UNIFORM indices across non-axis
  dims; ggml SET_ROWS is a BATCHED per-group scatter (per ne2/ne3 group, r rows with
  group-specific indices). Can't express it. Custom kernel instead.
- **SOFT_MAX via add+softmax (eltwise `sum` coeff [scale,1.0])**: only handles
  NUMPY-broadcast masks; ggml modular nr23 mask broadcast + ALiBi + sinks can't.
  Replaced with the custom `ovgpu_soft_max` kernel (handles all natively).
- **SET_ROWS compaction-via-reorder** for v=true views: broke on I64 (see above).
  Stride-aware kernel (read via nb[]) handles contiguous + padded views uniformly,
  no compaction.
- **`allow_new_shape_infer(true)`** for oneDNN FC: breaks the rank-2 reshape path
  (see Key decisions). Don't set it.
- **`copy_from`/`copy_to` (clEnqueueMemcpyINTEL) for set/get_tensor**: caused
  heap corruption in M0, switched to `mem_lock`. **This is now the weight-load
  perf bug** (mem_lock read_write round-trips the whole buffer per tensor) —
  revisit: `copy_from(stream, data, 0, offset, size)` writes only the region and
  is used safely by `buffer_clear`. See Perf section.

## Current state

Full-suite regression: **1025/1025 pass, 0 fail, 0 crashes** (test-backend-ops
`-b OVGPU0`, no `-o` filter). Implemented: MUL_MAT (oneDNN FC, native f16/bf16
weights), ADD/MUL (eltwise, strided+permuted views), RMS_NORM (padded views),
SOFT_MAX (custom kernel: mask/ALiBi/sinks/scale), GET_ROWS (gather), CPY
(strided/permuted src), SET_ROWS (custom, stride-aware, cross-type f32<->f16 +
v=true views), ROPE (custom, all modes + YaRN + BACK). Deferred to CPU (correct,
not failures): quantized types, MUL_MAT GQA/grouped weights, CPY strided bf16 +
strided dst, SET_ROWS quantized/bf16 dst.

## Perf (profiled on Llama-3.2-1B-Instruct-F16, Lunar Lake)

First real-model run: `llama-simple -m Llama-3.2-1B-Instruct-F16.gguf`. Two issues:

**1. Weight loading super slow (113 s).** Root cause: `set_tensor`/
`get_tensor` used `mem_lock<read_write>` over the ENTIRE model buffer per tensor
(full ~2GB device->host->device round-trip per weight tensor, O(num_tensors *
bufsize)). Worse: per-token logits readback (`get_tensor` on the output node) ALSO
mem_locked the whole 2GB for a ~128KB read - the hidden per-token cost. **Fix:
`memory->copy_from(stream, data, 0, dev_off, size, true)` / `copy_to(...)`** -
writes/reads only the tensor's region (used safely by `buffer_clear`). The M0
"copy_from caused heap corruption" was a different misuse, not inherent. Result:
**load 113 s -> 0.7 s**, and **eval 112 -> 47 ms/tok** (the logits readback was
the big hidden win). The old mem_lock comment ("preserve other tensors") was
wrong - copy_from at an offset doesn't touch other tensors.

**2. tps.** After the load fix, vs CPU:
- prompt eval (batch): OVGPU **80 tok/s** vs CPU 9.4 (OVGPU 8.5x faster) -
  oneDNN FC + native f16 + batched matmul shines.
- single-token eval: OVGPU 47 vs CPU 46 ms/tok (parity). The graph SPLITS
  (98/582 nodes) because FLASH_ATTN_EXT, CONCAT (KV cache), GLU (fused SiLU-gate)
  are unsupported -> attention runs on CPU with GPU<->CPU copies around it per
  layer + per-op `finish()` serialization. Batch amortizes the matmul (prompt
  eval 8.5x faster); single-token is sync-bound. **Next lever: FLASH_ATTN_EXT on
  GPU** (native cldnn SDPA, or decomposed MUL_MAT+SOFT_MAX+MUL_MAT - the parts
  are already supported). CONCAT + GLU/SILU are easy incremental split-reducers.
  Instrumented via a gated `supports_op` default-branch op-name print
  (`OVGPU_UNSUP=1`) to confirm the split-causing ops.

## pm9 - CONCAT + GLU/SILU custom kernels; perf measured

Added to cut graph splits (the easy split-reducers): custom stride-aware kernels
for GGML_OP_CONCAT (KV-cache append, any dim, one work-item/element byte copy;
bf16 gated - same raw-bind quirk as CPY strided), GGML_OP_GLU SwiGLu only (1-input
split-halves + 2-input glu_split = what Llama's FFN uses; out=silu(a)*b; other GLU
ops -> CPU), GGML_OP_UNARY SILU (out=x/(1+exp(-x)), accurate exp). Full suite
1025->1125. Bugs: cldnn `__global` address-space on a `const char*` local (CONCAT);
`NROWS` -D missing (GLU); SILU is GGML_OP_UNARY not GGML_OP_SILU; -o filter uses
ggml_op_desc (SILU/SWIGLU, not UNARY/GLU).

**Measurement (Llama-3.2-1B-F16):** graph splits 98->66; single-token eval
47->45 ms/tok (modest); prompt eval unchanged (~69 tok/s). Gain is small because
the remaining splits are dominated by FLASH_ATTN_EXT (attention on CPU), and every
GPU op pays a per-op `net->get_stream().finish()` tax (~240 finishes/token,
est. ~24ms of the 45ms) - so adding GPU ops has diminishing returns until
finish() is removed. **Two next levers, in order: (1) remove per-op finish()**
(each op is a separate cldnn::network w/ its own stream -> needs sync before the
next op reads its output; investigate shared stream / events); (2) FLASH_ATTN_EXT
on GPU. Cold first-run after a rebuild shows ~3x slower prompt eval = new-kernel
JIT, ignore (re-runs stable).

## pm10 - remove per-op finish() via shared in-order stream

Each op was a separate cldnn::network with its OWN stream + a `finish()` after
every execute (~240 finishes/token). Fix: all networks share ONE stream via
`network::allocate_network(shared_stream, program)` (build program with
`program::build_program(engine, topo, config)`, alloc on reg_ctx->stream), and
remove the per-op finish.

**The trap**: cldnn's default queue is **out-of-order** (qtype=1, not in-order
despite execution_config default). Out-of-order queues do NOT order kernels, so
removing finish() on the shared (out-of-order) stream gave WRONG RESULTS in fused
multi-op graphs (RMS_NORM_MUL_ROPE etc.) - isolated single-op tests passed, only
chained ops failed. Diagnosed by printing `get_queue_type()` per op. Fix: create
reg_ctx->stream **in-order** (`config.set_property(queue_type, in_order)`) so the
queue orders kernels across networks for free. Intra-network primitives are
likewise queue-ordered (cldnn's event-based intra-network deps are null on
in-order, but the queue order suffices).

Result: single-token 45->~39 ms/tok (~13%), now at/slightly above CPU parity
(was slightly below). **Modest because finish() on a drained queue is cheap** -
the dominant single-token cost remains FLASH_ATTN_EXT on CPU (attention split).
Prompt eval also improved (up to 94 tok/s). Full suite 1125/1125, 0 regressions.
Builders now take a `stream` param threaded from graph_compute -> op_cache ->
build_op. See [[ovgpu-debug]] skill "Perf" + memory ovgpu-perf-baseline.

## pm11 - CONT support; attention split root cause found (was wrong about FA)

Added GGML_OP_CONT (ggml_cont = contiguous copy; reuses build_cpy with src[1]=NULL,
dst=fresh contiguous tensor). Full suite 1125->1155. **No tps effect** - the CONT
supports_op calls were scheduler probes, not real splits.

**Attention split root cause (GGML_SCHED_DEBUG=2 split dump):** I was WRONG that
"a fused FLASH_ATTN_EXT node runs on CPU." `flash_attn=auto` resolves to DISABLED
(our supports_op returns false for FA -> the auto-probe builds a graph, sees the FA
node land on CPU != layer device OVGPU -> device_mismatch -> `flash_attn=false`).
So the graph uses the DECOMPOSED attention: `kq=mul_mat(k,q); soft_max; mul_mat`.
BUT that decomposed attention is ALSO on CPU - because q/k/v are
`ggml_permute(0,2,1,3)` views of the KV cache, and MUL_MAT only accepts
`ggml_is_flat_padded_view` (ne[0] fastest, ne[1..3] densely nested). A genuine
axis-permuted view is NOT flat-padded -> rejected -> CPU. The strided+permute
machinery (ggml_is_strided_view) exists for ADD/MUL/CPY but MUL_MAT doesn't use it.
Also corrected: the frequent op was CONT (op 35), not CONCAT (op 505) - I misread
the enum earlier; CONCAT support helps test coverage only.

So the real lever is attention on GPU. Two paths: (1) native cldnn SDPA for
FLASH_ATTN_EXT (primitive takes q/k/v + transpose orders + mask + scale + sink,
handles the permutes internally - subagent mapping the exact contract); (2) make
MUL_MAT accept permuted/strided operands (wire existing strided+permute into
build_mul_mat). Debug method that cracked it: temporarily flip the `#if 0` debug
block in ggml-backend.cpp:867 to `#if 1`, rebuild, run with `GGML_SCHED_DEBUG=2` -
dumps per-node backend assignment + cause (1.inp/1.wgt0/3.best/4.cpy). RESTORED
to `#if 0` after.

Path (1) was taken first (`build_flash_attn_ext`, see memory ovgpu-flash-attn-sdpa-mapping)
and got FLASH_ATTN_EXT's `supports_op` returning true. **Since then, build_mul_mat's
`ok_operand` has ALSO grown a `ggml_is_strided_view(t)` branch** (in addition to
`ggml_is_contiguous`/`ggml_is_flat_padded_view`) - i.e. path (2) may now be partially
available too, UNVERIFIED because FA currently reports itself supported so decomposed
attention never gets exercised. Worth checking first if path (1) is abandoned.

## pm12 - FLASH_ATTN_EXT is numerically wrong + corrupts memory; root cause NOT native-SDPA-shaped

`test-backend-ops -o FLASH_ATTN_EXT` fails broadly (this was the state entering the
session - `supports_op` already returned true for the base case, `build_flash_attn_ext`
already existed per pm-prior/memory). Root-caused two distinct bugs; NEITHER is fixed.

**Bug A - wrong results, every case, independent of head size.** With the ORIGINAL
config (`make_config()`: `optimize_data(true)` + `use_onednn(true)`, no
`allow_new_shape_infer`), every FA case that actually builds (mask=1,sinks=0 cases)
returns numerically uncorrelated output: NMSE ~0.98-1.7, occasionally `inf`. Confirmed
on BOTH hsk=40 (the odd/unaligned size) AND **hsk=64 = hsv=64 (Llama-3.2's actual head
size, DPAS-aligned)** - same magnitude, same failure signature. **This rules out "just
gate supported head sizes to 64/96/128/256"** - the bug is not head-size-specific.

Lead: `sdpa_gpu_test.cpp` (the OV unit test that originally verified the SDPA contract)
ALWAYS sets `ov::intel_gpu::allow_new_shape_infer(true)` before building its SDPA
topology - our `build_flash_attn_ext` never did (it reuses the shared `make_config()`,
which MUST NOT set that flag for MUL_MAT's oneDNN FC rank-2 reshape - see "Key
decisions"). Tried a **dedicated config for FA only** with
`allow_new_shape_infer(true)` (MUL_MAT unaffected, separate `cldnn::program` per op) -
results stayed wrong (NMSE ~0.98) AND introduced Bug B (below) on every run, so this
flag is necessary-but-not-sufficient at best, and its own side effect (Bug B) needs
addressing first to even evaluate whether it helps mathematically.

**Bug B - sentinel/memory corruption with `allow_new_shape_infer(true)`.**
test-backend-ops's sentinel tensors (placed immediately after every `ggml_new_tensor*`
call to catch backend overflows) get corrupted: `sentinel mismatch: sent_N` where
`sent_N` is always the LAST tensor allocated immediately before the FA op's inputs -
the mask tensor's sentinel when a mask is present (`sent_5`), the V-cache view's
sentinel when it isn't (`sent_4`, one slot earlier since mask + its sentinel don't
exist). This points to an out-of-bounds WRITE by the SDPA topology into memory
adjacent to (immediately preceding, in ggml's context arena) whatever it was fed,
not a specific operand's layout being wrong - i.e. something in cldnn's internal
buffer sizing decides on a larger extent than the raw ggml-buffer subbuffer actually
has room for. Tried: forcing the impl to `sdpa_opt` (skip `sdpa_micro`, which earlier
notes already flagged as broken for unaligned head sizes on this HW - see
"Things tried that didn't work") via `force_implementations`, and independently
routing the mask through a compacting `reorder` into a fresh cldnn-owned buffer
(matching how q/k/v already are, via `bind_bhsd`) instead of feeding SDPA the raw
external `input_layout` directly. **Neither changed the corruption or the NMSE** -
same `sent_5`/`sent_4` mismatch, same ~1.0 NMSE, with EITHER change alone or both
together. So the corruption isn't tied to the specific implementation kernel or to
the mask input specifically.

**Bug C - resource exhaustion across repeated builds (separate from A/B).** Running
more than ~4 distinct FLASH_ATTN_EXT shapes in one process (even ones that don't
crash individually) eventually throws an UNCAUGHT `ov::Exception` from
`cldnn::ocl_stream::finish()` inside `cldnn::network::~network()` during
`ggml_backend_ovgpu_free` at process exit - `CL_OUT_OF_RESOURCES` (`ocl_stream.cpp:395`
`clFinish`). An exception from a destructor during unwind is `std::terminate`d
(SIGABRT), not a graceful FAIL. Separately, in a full (many-shape) `-o
FLASH_ATTN_EXT` run, EVERY build after some point throws `CL_INVALID_EVENT` from
`ocl_memory.cpp` (`clWaitForEvents`) inside `gpu_usm::fill` - root-caused (via an
added debug print, see below) to `enqueue_fill_mem`'s wrapper (`ocl_ext.hpp`)
silently discarding the real `cl_int` error code from `clEnqueueMemFillINTEL`
and unconditionally waiting on a possibly-never-populated event; but a SINGLE
isolated repro of the exact same shape (own process) never reproduces this - it's
a function of how many prior networks were built/destroyed in-process, i.e. a
genuine resource leak (event pool exhaustion, or similar) building up over
multiple FA network builds. **This is almost certainly the source of the "GPU hang
risk" observed when running the un-filtered full test suite** (confirmed: a 90s
timeout was hit and killed the process mid-run on `-o FLASH_ATTN_EXT` alone, no
`-b`/other-op filtering).

**Debug aid added (kept, gated):** `OVGPU_FILL_DEBUG=1` env var in
`openvino/src/plugins/intel_gpu/src/runtime/ocl/ocl_memory.cpp`'s `gpu_usm::fill`
prints `bytes`/`ptr`/`dep_events`/`enqueue_err` for every USM fill - this is how
Bug C's `enqueue_fill_mem` return-code-swallowing was found. Requires rebuilding
`openvino_intel_gpu_graph` (`ninja -C openvino/build/Release
openvino_intel_gpu_graph`) then relinking llama.cpp targets.

**Net conclusion**: this is not a small translator bug (wrong transpose order, wrong
axis mapping, missing cast) - hand-verified the q/k/v/mask layout math (padding,
strided order, permute order) against `ggml_get_strided_order`/`ggml_layout_for_strided`
by hand for the K-cache padded-view case and it checks out. The wrongness + corruption
survives across: two head sizes (40, 64), with/without mask, with/without the
`allow_new_shape_infer` flag, with/without forcing `sdpa_opt` over `sdpa_micro`, and
with/without a mask pre-compaction reorder. That breadth of invariance suggests either
(a) a more fundamental mismatch in how a bare per-op `cldnn::program`/`network`
(no `ov::Model`, no real predecessor/successor graph context) drives THIS SPECIFIC
primitive - SDPA's shape/buffer-size inference may depend on graph context our Tier-A
eager model doesn't provide, unlike every other primitive we've mapped so far - or
(b) an OV/driver bug on this exact build+device combination independent of our
translator. Distinguishing (a) from (b) would need a standalone (non-ggml) minimal
repro built directly against cldnn mirroring our exact construction, compared line-by-
line against `sdpa_gpu_test.cpp`'s working recipe - not yet done.

**Two paths forward, undecided (session paused here per user request):**
1. Keep pushing on native SDPA - next step would be the standalone minimal repro above.
2. Abandon native SDPA for now; report `GGML_OP_FLASH_ATTN_EXT` unsupported so the
   scheduler decomposes to `MUL_MAT`+`SOFT_MAX`+`MUL_MAT` (plan doc's accepted "path
   a"). Worth checking first: `build_mul_mat`'s `ok_operand` already accepts
   `ggml_is_strided_view` (see pm11 addendum above) - unverified whether that's enough
   for decomposed attention's permuted KV views to land on GPU now, since pm11 found
   MUL_MAT rejecting them was the reason decomposed attention ALSO fell to CPU.

**Do NOT run `test-backend-ops -o FLASH_ATTN_EXT` (or any run that reaches it)
without a hard `timeout` wrapper** - per Bug C, it can hang/crash the process.

## pm13 - FLASH_ATTN_EXT FIXED: kernels ignore output_transpose_order

`test-backend-ops -o FLASH_ATTN_EXT` = **714/714 pass**; full `test-backend-ops` =
**1869/1869 pass, 0 fail, 0 crash**; `llama-simple` (Llama-3.2-1B-Instruct-F16) emits
coherent text (25 t/s decode, 30 graphs reused). pm12's "net conclusion" (not a small
translator bug, possibly a fundamental bare-network/ov::Model mismatch) was **wrong** -
it was a one-line translator mistake that hand-verification of the *input* layout math
couldn't catch because the bug was on the *output* side.

**Root cause (the real Bug A+B):** the cldnn SDPA kernels (`sdpa_opt.cl:925,2610`,
`sdpa_ref.cl:314`) **always write `OUTPUT_GET_INDEX(b, head, seq, d)`** - canonical BHSD
coordinates straight into the physical output buffer. There is **no output dims-order
remap** (no `OUTPUT_DIMS_ORDER` macro anywhere in `src/graph/`; inputs get
`INPUTn_DIMS_ORDER` via `sdpa_base.cpp:34 get_dims_order`, outputs don't).
`output_transpose_order` only feeds **shape inference** (`calc_output_layouts` permutes
the allocated buffer shape via `shape_infer`), NOT the kernel write. The translator
passed `output_transpose_order={0,2,1,3}`, so shape inference allocated a `(B,S,H,D)`
buffer that the kernel filled with `(B,H,S,D)` coords -> head/seq placement swapped:
- **H==S_q** (e.g. square prefill): writes stay in-bounds but transposed -> NMSE ~1.0
  "numerically uncorrelated" on every case (pm12 Bug A).
- **H!=S_q**: out-of-bounds writes (non-safe `GET_INDEX`, no clamping). Single-token
  decode S_q=1 with H heads: head idx up to H-1 into the f-dim of size 1 -> offsets up
  to ~(H-1)·H·D in an H·D buffer -> the sentinel overflow / memory corruption (pm12 Bug B,
  "sent_5/sent_4 mismatch" = the OOB write clobbering the adjacent ggml arena tensor).

This is why pm12 saw the wrongness survive across head sizes, mask on/off,
`allow_new_shape_infer`, and impl forcing - none of those touch the output write path.
Real OV flows never hit it: `TransposeSDPAFusion` (`transpose_fusion.cpp:245-248,304`)
fuses ONLY q/k/v input transposes into the op and always leaves `order_output` at
`default_order` (identity) - any output transpose stays a separate `permute` node. The
OV unit test (`sdpa_gpu_test.cpp:95`) also uses identity output order. The non-identity
output path was completely untested and broken in the kernels.

**Fix (`build_flash_attn_ext`, ggml-ovgpu-ops.cpp):**
1. `output_transpose_order = {0,1,2,3}` (identity) -> SDPA output allocated canonical
   physical BHSD, matching the kernel write. Then explicit `cldnn::permute({0,2,1,3})`
   + compacting `reorder` into ggml's BSHD dst (same permute+reorder idiom as inputs).
2. **Strided input binding** (zero-copy): new `ggml_sdpa_input_bind` helper binds q/k/v
   in their physical nesting (contiguous -> eltwise BHSD identity order; strided/
   permuted view -> `ggml_layout_for_strided` BSHD, per-input `{0,2,1,3}` order computed
   from the stride sort) - no permute/compaction; cast-only reorder for f16->f32. SDPA
   reads via per-dim pitches, so padded KV-cache views bind zero-copy. (An earlier
   attempt used compact-BHSD+identity inputs - worked but copied; strided is the
   intended design and passes all head sizes.)
3. **Removed `force_implementations`** - it was a no-op: for ocl_v2 impls the forcing
   map's kernel-name string is ignored (only `impl_types` enum + format are honored;
   the name only reaches the legacy kernel_selector path). SDPAOpt/SDPARef are both
   `impl_types::ocl`, registry picks SDPAOpt; `sdpa_micro` is a *stage inside*
   SDPAOptImpl, auto-gated by `supports_micro_sdpa`. The "force sdpa_opt to disable
   micro" never disabled micro.
4. `build_mul_mat` consolidated through `make_config()` (forward-declared; identical
   settings).

**Residual gate (supports_op, ggml-ovgpu.cpp):** `hsk != hsv && max(hsk,hsv) > 256` ->
CPU. The `sdpa_opt` multi-token kernel segfaults (signal, not catchable - so
`get_or_build`'s try/catch can't rescue it) on large-prefill builds for MLA configs
DeepSeek 576/512 and Mistral4 320/256. `supports_micro_sdpa` requires K_head==V_head so
micro is already off for these (sdpa_opt path). 192/128 (hsk!=hsv, smaller) passes the
full matrix - kept on GPU. Standard hsk==hsv (Llama/Qwen/etc.) unaffected.

**pm12 Bug C ("resource exhaustion across builds") was a misdiagnosis too:** the
crash always landed on the hsk=320/hsv=256/nb=75 case (deterministic for that shape
family), not from accumulation - it just *appeared* after N prior cases because that
case is late in the test ordering. Isolated, hsk=320/hsv=256/nb=75 alone passes, but
after building the same-family nb=1/3/32 it crashes (local engine state from the
related kernels). The hsk!=hsv gate sidesteps it entirely. The `CL_OUT_OF_RESOURCES`/
`CL_INVALID_EVENT` symptoms pm12 saw were downstream noise from the OOB corruption,
not an independent leak. (The `OVGPU_FILL_DEBUG` probe in ocl_memory.cpp is no longer
needed for this but is left gated/harmless.)

**`-j` is unsafe for this backend:** parallel test-backend-ops workers race on the
shared in-order stream/op_cache and segfault (repro: hsk=192 nb=3 mask=0 under `-j`;
same case passes single-threaded). Run `test-backend-ops` WITHOUT `-j`.

**Perf note:** strided binding avoids the input permute/compaction, but with the current
f32-cast policy the f16->f32 reorder copy remains (not fully zero-copy). Dropping the
cast to feed native f16 k/v (mixed-dtype SDPA - pm12 claimed CL_BUILD failure, likely
confounded by the output bug) is the next zero-copy lever; untested.

## pm14 - multi-ubatch decode crash (llama_decode -3); zero-element-op skip

**Symptom:** `llama-bench -d 4096` (and any `-d` > 512) failed with
`test_prompt: failed to decode prompt batch, res = -3`. `-d 0` and `-d 512` worked.
`llama-perplexity`/`llama-server` would hit the same path.

**Bisect:** d=512 works (max S_kv 1024), d=640/1024/2048/4096 all fail. The threshold
is NOT sequence length - it's **ubatch count**: any decode split into >1 ubatch fails.
Confirmed: `-d 1024 -ub 1024` (single-ubatch priming) works; default `-ub 512` fails.
`--flash-attn 0` still fails (independent of FA); `GGML_OVGPU_NO_CACHE=1` still fails
(not a cache-key bug - tested per user's hypothesis, ruled out).

**Root cause:** A multi-ubatch decode's NON-final ubatches have `n_outputs == 0`
(llama-context.cpp: per-ubatch `n_outputs_new` counts output flags; only the last
ubatch of a decode carries the requested output tokens). With `n_outputs==0`,
`build_inp_out_ids` builds an `out_ids` tensor with `ne[0]==0`, and the last-layer
output gather `ggml_get_rows(cur, inp_out_ids)` (models/llama.cpp:175) feeds a 0-index
gather. The cldnn `reshape("reshaped", gather_out, [0, n_embd, 1, 1])` AFTER the gather
**rejects a zero-count buffer**:

```
OVGPU: build_op threw for op 40 (GET_ROWS): Error has occured for: reshaped
Output layout count(=2048) is not equal to: input layout count(=0)
Output layout of reshape primitive changes size of input buffer
key=op40|out0|s0:0:2048,512,1,1,:4,8192,4194304,4194304,|s1:26:0,1,1,1,:4,0,0,0,
```

(`s1` = out_ids, type 26 = I32, ne=[0,1,1,1] - zero indices.) The throw propagates:
`build_op` throws -> `get_or_build` catches -> returns nullptr -> `graph_compute` returns
`GGML_STATUS_FAILED` -> `llama_decode` returns -3. And it's not just GET_ROWS: every op
DOWNSTREAM of the zero-ids gather (FFN `mul_mat` with M=0, logits `mul_mat` with M=0)
would also throw on zero-batch oneDNN FC. d<=512 worked only because every decode was a
single ubatch (= the last ubatch, n_outputs>=1).

**Why it was invisible:** `build_op`'s catch used `GGML_LOG_ERROR`, but llama-bench (and
other tools) install `llama_null_log_callback` via `llama_log_set`, which swallows ggml
log output. The throw printed nothing - only `res = -3`. `GGML_OVGPU_DEBUG=1` showed the
failing node (`node 488 op=40`, no "built" after it) but not why. gdb raw-offset inspection
of `build_get_rows` args gave garbage (struct offset/inline issues) - dropped.

**Fix (ggml-ovgpu.cpp, graph_compute node loop):** skip any node whose output has zero
elements (`ggml_nelements(node) == 0 -> continue`), placed after the metadata-op skip. A
0-element op writes nothing - the same semantics as the CPU backend, whose per-element
threads simply do no work on an empty tensor. Only the ubatch's real outputs (K/V into the
cache via SET_ROWS, non-zero) are needed and still run; the post-gather FFN/logits on 0
gathered rows are genuinely no-ops. This handles GET_ROWS AND all downstream zero-batch
ops uniformly, without per-translator zero-size special-casing.

**Error visibility (ggml-ovgpu-ops.cpp, get_or_build):** the `build_op` catch and the
null-return path now `fprintf(stderr, ...)` the op name + exception `what()` + full cache
key (op + every src's type/ne/nb) instead of `GGML_LOG_ERROR` - fprintf can't be silenced
by a host log callback. `GGML_OVGPU_NO_CACHE=1` env knob bypasses the op cache (rebuilds
every node) for isolating cache-key vs genuine-build-failure bugs.

**Verification:**
- `test-backend-ops` full suite: 1869/1869, 0 fail (skip is a no-op there - no 0-element
  test shapes).
- `llama-bench -d {0,1024,4096}`: pp512 3875/1768/762 t/s, tg128 30.8/27.2/16.8 t/s.
- `llama-perplexity` (wikitext-2, `-b 512 -c 2048` = 4 ubatches/chunk): PPL 12.77 ± 0.30.
- `llama-server -np 4`: single + 2 parallel completions coherent, clean exit. `-np 4` is
  safe - parallel decode is batched into one `llama_decode` (one `graph_compute`), not
  concurrent backend calls, so the `-j` shared-stream race does not apply.

**Lesson:** the cache key was the user's first suspect (graph 1's gather built, graph 2's
"didn't"); the real tell was that the GGML_OVGPU_DEBUG log showed no "built" line for the
failing node -> `get_or_build` returned null/threw, not a cache-hit-with-wrong-network.
The no-cache test (still failed) confirmed it. The actual differentiator between the two
graphs was `n_outputs` (0 vs >=1) -> `out_ids` ne[0] (0 vs >=1) -> a NEW key that genuinely
fails to build, not a collision.

## pm15 - op fusion via multi-op cldnn topologies (Tier B-lite)

**Goal:** ovgpu was Tier A - one single-primitive cldnn network per ggml op - so cldnn's
program-level fuser (`prepare_primitive_fusing.cpp`, gated on `optimize_data(true)`, which
`make_config` already sets) had nothing to merge: each topology had exactly one compute
primitive. The per-op eager overhead (one bind+enqueue per node, plus materializing every
intermediate) capped decode at ~parity with CPU. This milestone builds ONE cldnn topology
per fusible *chain* of ops and lets `program::fuse_nodes` merge the children into the head
at build time.

**Fusibility verified** (read `prepare_primitive_fusing.cpp` + the oneDNN FC impl
`fully_connected_onednn.{hpp,cpp}` + `program_node.cpp`'s `create_onednn_primitive_attributes`,
via a research subagent). Single overriding constraint on Lunar Lake (`supports_immad=1`):
the fused-into node must have exactly ONE user. Plus `optimize_data(true)`+`use_onednn(true)`
already set. Findings:
- **oneDNN FC on Lunar Lake DOES accept eltwise(sum/prod) + activation(swish=SILU) post-ops**
  (applied as oneDNN matmul post-ops, single fused kernel) - confirmed via
  `create_onednn_primitive_attributes` (`program_node.cpp:1528`, `fully_connected_onednn.cpp:376`).
  This unblocks the whole FC-fusion family. (Prior exploration had stale-framed oneDNN FC
  engagement as an open blocker - it isn't; the downcast-act path engages oneDNN FC for the
  common f16-weight case.)
- `rms` + eltwise(MUL gamma) fuses UNGATED (rms in the parent list at `:1074`;
  `rms_kernel_bfyx_opt` declares `ELTWISE`).
- `fuse_swiglu` SKIPS FC on immad (`:229`) -> SwiGLu must be built as FC->act(swish)->eltwise(prod),
  NOT via the swiglu primitive.

**Mechanism (ggml-ovgpu.{hpp,cpp}, ggml-ovgpu-ops.cpp):**
- Extracted `rms_core` / `mul_mat_core` from `build_rms_norm` / `build_mul_mat`: append the
  compute primitive to a SHARED `cldnn::topology` (no trailing reorder, no `make_network`),
  namespaced by a `prefix` so head + children coexist without id collisions. `build_*` became
  core + trailing reorder + make_network (behavior identical - the refactor was
  regression-checked: 1869/1869 before any fusion wiring).
- `build_chain(nodes)`: one topology, head core -> each child core consumes the previous
  compute-out in the HEAD's output-layout convention (flat bfyx for FC, reduce-ne0 for rms)
  so the fuser's shape-broadcast rule holds; the child's external operand (residual/gamma)
  bound in that same layout, cast to the head's compute-output dtype; one trailing reorder +
  one `make_network`. `compiled_op` extended with `is_chain`/`input_layouts`/`input_src_map`/
  `chain_len`; `op_cache` got a `chain_cache` + negative `chain_fail_cache`.
- `graph_compute` chain detection: a consumer-count pre-pass finds maximal linear chains
  [head, child1, ...] (head MUL_MAT/RMS_NORM, children ADD/MUL/UNARY-SILU) where each non-final
  node's output has exactly one consumer. **Availability check**: a child's external operand
  must be a leaf or a cgraph node already materialized BEFORE the head (index < head) - this
  rejects diamond patterns like `((mm1+mm2)+mm3)` where a sibling matmul is computed after
  the head (the test-backend-ops `MUL_MAT o>1` case). Absorbed children are skipped; the head
  executes the whole chain. Fallback: if `build_chain` rejects/throws, un-absorb members and
  run per-op (correctness preserved). `bind_input`/`bind_output` helpers factored out, shared
  by both paths.

**Diagnostics:** `GGML_OVGPU_FUSE_TRACE=1` walks `net->get_program()->get_processing_order()`
and prints each FC/rms node's `has_fused_primitives()` + fused `desc->id` list - confirms the
chain fused (vs separate kernels). Needs `#include "program_node.h"` + a CMake include dir
(`${OVGPU_INTEL_GPU}/src/graph`, for `registry/implementation_manager.hpp`). `GGML_OVGPU_FUSE_DISABLE=1`
skips detection entirely (pure per-op A/B + safety).

**Bugs hit + fixed:**
- A `MUL_MAT o=3` test case (`((mm1+mm2)+mm3)` diamond) regressed: the detector formed
  `mm1->ADD->ADD` and ran it at mm1's position, reading mm2/mm3 before computed -> wrong output.
  Fix: the availability check above (a child's external operand must predate the head).
- After that, a strided-FC + ADD chain (`k_v=64` test) still mis-computed: with strided/
  flat-padded FC inputs the FC output stays in cldnn's preferred (blocked) format with no
  trailing reorder before the eltwise, and the oneDNN binary post-op mis-aligns the plain peer.
  Fix (phase-1 gate): FC chains require CONTIGUOUS operands; strided FC falls back to per-op
  (correct, unfused). Real weights/activations at residual-ADD points are contiguous, so this
  only gates artificial test shapes.

**Verification:**
- `test-backend-ops` full suite: 1869/1869, 0 fail (chains don't form in single-op tests;
  the strided-FC diamond is gated out and falls back to per-op, which is correct).
- `llama-simple` (Llama-3.2-1B-Instruct-F16): output **bit-identical** fusion vs
  `GGML_OVGPU_FUSE_DISABLE=1` ("The capital of France is Paris. The Eiffel Tower...").
- FUSE_TRACE: both `rms+MUL` and `FC+ADD` chains form with `fused=1` in the real model.
- Perf (d=0): tg128 fusion ~24 t/s vs disabled ~21.75 (**+10% mean, +23% best**; high
  variance = cold-start chain build, steady-state is the higher number; disabled rock-stable
  ~21.9). pp512 unchanged (compute-bound, as expected).

**Phase 2 (FC+SILU): implemented but not triggered.** `activation_silu_core` emits
`cldnn::activation(swish)`; oneDNN FC accepts it as an `eltwise_swish` post-op. But modern
llama `build_ffn` (`llama-graph.cpp:~1670`) uses `ggml_swiglu_split` (a single fused GLU op),
NOT decomposed `ggml_silu`+`ggml_mul` - so FC+SILU chains don't form for llama. Fusing the
GLU itself needs graph decomposition (replace `GGML_OP_GLU` SwiGLu with SILU+MUL at
translation) - deferred; that would let `FC->SILU` (done) + `FC->SILU->MUL` chains form and
hit the FFN decode bottleneck directly.

**Moved the `ovgpu-debug` skill** out of the openvino repo (where it was untracked) into
`llama.cpp/.claude/skills/ovgpu-debug/` (this repo) - it documents THIS backend, not OV.
Committed the OV-side debug logging (`OVGPU_FC_TRACE`/`OVGPU_FILL_DEBUG`, env-gated, no
behavior change) on a branch in the openvino repo -> pushed to the `zijun` (wine99) fork.

