---
name: ovgpu-debug
description: Debug the ggml-ovgpu backend (the eager clDNN translator in llama.cpp/ggml/src/ggml-ovgpu/) - runtime crashes, ov::AssertFailure, CL_* OpenCL errors, wrong/ERR results, accuracy/precision flakes, "not supported" offloads, perf. Lists the common bug sources for THIS backend and the debug approaches to try in order. Read this BEFORE spelunking cldnn/OpenCL source.
---

# Debug the OVGPU backend

The OVGPU backend (`llama.cpp/ggml/src/ggml-ovgpu/`) drives Intel clDNN directly
(eager single-op `cldnn::network`s, no `ov::Core`). Most bugs recur in a small
number of classes. **Try the approaches in order** - the cheap structural checks
first, value-level last. Detailed root causes + misdiagnoses are in
`llama.cpp/WORKLOG.md` ("Bugs & root causes"); this skill is the fast path.

Reference impl = **ggml-cpu** (`ggml/src/ggml-cpu/ops.cpp`) - that's what
test-backend-ops compares against, so port *that*, not the ggml-opencl kernels.

## Step 0 - Classify the symptom

| Symptom | Most likely class | Jump to |
|---------|-------------------|---------|
| `not supported [OVGPU0]` (case skipped, goes to CPU) | supports_op gate | Step 1 |
| Wrong result, **large** ERR (0.1 - 10), deterministic | cache-key collision OR layout/reorder trap | Step 2, then 3 |
| Wrong result, **small** ERR just over threshold, **intermittent** | float-constant precision / cos-sin ULP | Step 4 |
| `ov::AssertFailure` / `ov::Exception` at execute | binding / arg-index / shape | Step 5 |
| `CL_BUILD_PROGRAM_FAILURE` (-11) | OpenCL compile error (custom kernel) | Step 6 |
| `CL_OUT_OF_RESOURCES` (-5) | dtype/size mismatch in a bind, or kernel too big | Step 5, 6 |
| Crash / SIGABRT (no ov message) | uncaught throw from build_op (shouldn't happen - try/catch) or a clDNN kernel selector abort | Step 7 |
| Slow (weight load / tps) | perf: mem_lock whole-buffer / per-op finish() | Step 8 |

## Step 1 - "not supported" => check supports_op, then build_op

`supports_op` (ggml-ovgpu.cpp) is the gate; if it returns false the scheduler
offloads to CPU (correct, not a bug). If a case you *expect* on GPU is skipped:
- `GGML_OP_NONE` must allow I32+I64 leaves; `RESHAPE`/`VIEW`/`PERMUTE`/
  `TRANSPOSE` must return true (test-backend-ops queries supports_op on EVERY
  tensor, including pure-metadata view nodes - rejecting these silently rejects
  any op fed a view).
- Each op's `supports_op` must match its `build_op` scope exactly (the n=8
  MUL_MAT history: supports_op said yes, build guard disagreed -> wrong output).
- bf16 is excluded from GET_ROWS (no clDNN gather kernel, aborts) and from
  strided CPY/SET_ROWS (permute/reorder mis-handles bf16+padding) - hardcoded.
- Custom-kernel ops gate on the kernel's capabilities (e.g. SET_ROWS rejects
  quantized dst - block quantization the scalar kernel can't do).

## Step 2 - Deterministic large ERR => cache key FIRST

**This is the #1 bug class.** `op_cache::key_for` (ggml-ovgpu-ops.cpp) must
include EVERYTHING baked into the compiled network:
- each src's `nb[]` (not just ne[] - same shape, different strides = different
  topology),
- every op-param baked as a `-D` or into a cldnn primitive (MUL_MAT transposed,
  ROPE all 15 op_params, RMS_NORM eps, SOFT_MAX scale/max_bias).
- **`key_for` must `continue` past NULL srcs, not `break`** - else a later
  optional src (e.g. SOFT_MAX sinks with no mask: src[1]=NULL, src[2]=sinks) is
  dropped from the key and collides with the no-sink variant -> silent wrong
  results (ERR ~the dropped term's magnitude).

**How to confirm a cache collision**: dump the key for two cases that share shape
but differ in params - if identical, that's it. The ROPE bug was misdiagnosed for a
full session as a kernel-logic error because a standalone element-by-element test
passed (no cache) while test-backend-ops failed (cache reuse). **If a standalone
test passes but the suite fails with large ERR on the same config -> cache key.**

## Step 3 - Layout / reorder traps (silent wrong results)

`cldnn::reorder` does **NOT** reshape - it keeps the source's used tensor,
adopting only the target's dtype/format/padding. So a reorder between two
*different axis decompositions* of the same size (e.g. per-dim `ne0->x` bound,
then reordered to flat-batch `ne[1..3]->batch`) **compiles clean, runs, produces
wrong results**. Bind in the SAME axis convention you reorder into.

Other cldnn axis-ordering gotchas (each has bitten - verify, don't assume):
- `cldnn::tensor` simple-ctor is `(batch, feature, x, y)`; `cldnn::padding`
  vectors, `cldnn::gather` `output_shape`, and `cldnn::permute` order are
  **FORMAT order `(b,f,y,x)`** (= ov::Shape). Mixing them silently transposes.
- `permute` order: 2-axis cases pass by coincidence (self-reverse); 3-axis crash
  with a broadcast-merge AssertFailure. Simulate in Python before trusting.
- `rms`/`softmax` reduce over ALL spatial dims (x,y,z) - put ne[0] on the *only*
  non-trivial spatial (`ggml_layout_for_reduce_ne0`), else it reduces over too
  much. `cldnn::reorder` compacts padding but does NOT flatten - add a
  `cldnn::reshape` if you need a different reduction shape.

## Step 4 - Small/intermittent ERR => float-constant precision

Custom kernels bake scalars as `-D` build_options. `std::to_string(float)` gives
**6 digits** - `theta_scale` 0.865964353 -> "0.865964" drifts ~2.6e-5 over 64
multiplies, intermittently flipping F16 rounding across the threshold (a 1-in-N
flake). **Use `fstr()` = `snprintf("%.9g")`** (exact IEEE-single round-trip) for
ALL baked floats. If still flaky on F16: GPU cos/sin differ ~1 ULP from CPU
cosf/sinf, amplified by F16 quantization - widen the tolerance gate in
supports_op for that specific combo rather than chasing the math.

`pow(M0, ...)` with a bare `-D` decimal literal is **ambiguous** (matches both
`pow(float,float)` and `pow(double,double)` on the Intel compiler with half-precision
headers) -> `CL_BUILD_PROGRAM_FAILURE`. Cast: `pow((float)M0, ...)`.

## Step 5 - ov::AssertFailure / CL_OUT_OF_RESOURCES at execute

Read the assert message string - it names the check:
- "Unexpected layout of input memory for <id>" => the bind layout
  (`co->layout_for(src)`) doesn't match the topology's `input_layout` for that id.
  layout_for must dispatch (flat for contiguous, padded/strided for views) to
  match what build_op bound.
- "input memory necessary to set kernel arguments" (ocl_stream.cpp:92) =>
  `custom_gpu` arg index out of range. For optional inputs the arg index must be a
  **running counter** (position in cg_inputs), NOT a fixed number - sinks-only has
  cg_inputs={in_0,in_2} so sinks is index 1, not 2.
- "Argument shapes are inconsistent" / eltwise broadcast-merge => a permute
  order or axis-decomposition is wrong (Step 3), OR a non-NUMPY broadcast
  (ggml tiling != numpy; gate in supports_op).
- "Could not find a suitable kernel for <op>" => clDNN has no kernel for that
  dtype (e.g. gather bf16). Hardcode-exclude in supports_op.
- CL_OUT_OF_RESOURCES (-5) => often a dtype/size mismatch in a bind (e.g.
  `ggml_type_to_cldnn(I64)` falls through to f32 -> 8-byte I64 read as 4-byte
  f32 -> OOB), or a kernel exceeding device limits. Check type handling first.
- Binding: `graph_compute` binds src[k] positionally - it must **skip NULL srcs**
  (walk src[0..], bind each present src to the next input_id) for ops with
  optional middle srcs (SOFT_MAX mask/sinks).

## Step 6 - CL_BUILD_PROGRAM_FAILURE (custom kernel won't compile)

**cldnn's build log (`GPU_DEBUG_INFO`) is compiled out in Release OV** - the
exception has no source line. Get the real error with a **standalone
`clBuildProgram` test**: write the exact kernel string + exact `-D` options to a
.cl file, compile with a tiny C program (`clCreateProgramWithSource` +
`clBuildProgram` + `clGetProgramBuildInfo`). This gives the clang-style error with
line numbers. (The `pow` ambiguity and several others were found this way.) If
unsure of the exact options cldnn passes, add a gated `fprintf` of `build_opts`
in build_op and set `GGML_OVGPU_DEBUG=1`.

## Step 7 - Crash with no ov message

- If it's an uncaught C++ exception from `build_op` -> `get_or_build`'s
  try/catch is missing/broken (it should degrade to a single-test FAIL, not a
  process abort). Re-run under `stdbuf -oL -eL` (line-buffered stdout+stderr):
  piped output interleaves unpredictably, so the crash's line in the raw log
  often attributes it to the *adjacent* (wrong) test case.
- clDNN kernel-selector aborts (SIGABRT, "Could not find a suitable kernel") have
  no graceful path - probe/exclude in supports_op ahead of time.

## Step 8 - Perf

- **Slow weight load**: `set_tensor` does `mem_lock<read_write>` over the ENTIRE
  model buffer per tensor (full device->host->device round-trip per tensor) ->
  O(num_tensors * buffer_size). Use `memory->copy_from(stream, data, 0, dev_off,
  size, blocking)` (writes only the region; used safely by `buffer_clear`).
- **Poor tps**: per-op `net->get_stream().finish()` serializes every op (no
  overlap); MUL_MAT per-token activation downcast reorder; cache-lookup+bind
  overhead per op; no graph-level batching. Profile with `GGML_OVGPU_DEBUG=1`
  (prints alloc/bind per op) and `OVGPU_FC_TRACE=1` (oneDNN FC kernel selection,
  in OV source, gated).

## Tools & techniques (cheat sheet)

- **`test-backend-ops -b OVGPU0 -o <OP>`**: isolate one op. Strip ANSI for
  grepping: `| sed 's/\x1b\[[0-9;]*m//g'`. Count: `grep -c OK` (note: OK has
  ANSI codes - strip first, or grep `: OK$` won't match).
- **Run N times** to detect flakes (precision/cache-timing): a flake on F16 with
  specific param combos => Step 4.
- **`GGML_OVGPU_DEBUG=1`**: prints alloc_buffer, per-op node, bind offsets.
- **`OVGPU_FC_TRACE=1`**: oneDNN FC validate/create + find_impl traces (in OV
  source `fully_connected_onednn.{hpp,cpp}`, `primitive_inst.cpp`; rebuild OV:
  `ninja -C openvino/build/Release openvino_intel_gpu_graph` then relink llama).
- **Standalone clBuildProgram**: the only way to see a custom-kernel compile
  error (cldnn's log is compiled out in Release). Template in WORKLOG pm8.
- **Standalone element-by-element test**: build the op directly via ggml on OVGPU
  vs CPU, compare elements. If it passes standalone but fails in the suite =>
  **cache key** (Step 2). This disambiguates "kernel math wrong" from "state bug".
- **`ONEDNN_VERBOSE=all`**: oneDNN's own error text (e.g. "dst:0 inconsistent
  with src:0" -> shape-mapping failure).
- **OV source reads**: cldnn is opaque from outside; the source
  (`openvino/src/plugins/intel_gpu/`) is the spec. Reduction dims
  (`rms_gpu_ref.cl`), kernel selectors (no bf16 gather), format optimizer
  (blocked vs plain), `reorder.cpp:172` (doesn't reshape) - all confirmed by
  reading source. Add env-gated trace prints where needed (left in place, useful).
- Build: `ninja -C build-x64-linux-ovgpu-release ggml-ovgpu test-backend-ops`.
  Run: `LD_LIBRARY_PATH=build.../bin:openvino/temp/Linux_x86_64/tbb/lib`.

## Don't repeat these (tried, failed)

- cldnn `rope` primitive (needs precomputed cos/sin; ggml computes at runtime) ->
  custom inline kernel.
- cldnn `scatter_update` for SET_ROWS (uniform indices; ggml is per-group) ->
  custom kernel.
- SOFT_MAX add+softmax (only NUMPY masks; can't do ALiBi/sinks/nr23) -> custom kernel.
- SET_ROWS compaction-via-reorder for v=true (breaks on I64) -> stride-aware kernel.
- `allow_new_shape_infer(true)` for oneDNN FC (breaks rank-2 reshape) -> don't set.
- mem_lock read_write for set_tensor (correct but O(N*bufsize) slow) -> copy_from.
