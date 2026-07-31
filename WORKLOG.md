# WORKLOG - OVGPU backend (eager clDNN PoC)

Branch: poc-ov-gpu-backend (worktree llama.cpp-ovgpu, base dev_backend_openvino)
Plan: poc-ovgpu-plan.md (high-level reference, this file is the detailed log)

## M0 - skeleton + build wiring + FC smoke test  [DONE]

Goal: prove clDNN is drivable from inside a ggml backend - engine init, a
hand-built single-op network, eager execute, caller memory - before touching
any real op translator.

### 2026-07-30 - scaffolding

Created the backend in ggml/src/ggml-ovgpu/:
- ggml-ovgpu.cpp        : backend skeleton (reg/device/buffer_type/buffer/iface)
- ggml-ovgpu-smoke.cpp  : standalone fully_connected run + CPU reference check
- ggml/include/ggml-ovgpu.h : public reg symbol
- CMakeLists.txt        : links OV GPU static libs directly (no plugin .so)

Wired into the ggml build:
- ggml/CMakeLists.txt        : option(GGML_OVGPU ...)
- ggml/src/CMakeLists.txt    : ggml_add_backend(OVGPU)
- ggml/src/ggml-backend-reg.cpp : include + register_backend + load_best("ovgpu")
- CMakePresets.json          : x64-linux-ovgpu-release / -debug presets

Skeleton modeled on ggml-cann (complete GPU iface, external-lib CMake),
NOT the legacy ggml-openvino backend (which goes through ov::Core/ov::Model).
supports_op returns false for every op -> scheduler falls back to CPU.
graph_compute currently asserts (should never be reached while supports_op=false).

### 2026-07-30 - link surface discovery

The "consume clDNN standalone" claim from the investigation held up, but the
exact link surface took several iterations to pin down. Every missing piece
was "the graph lib transitively includes a header from an OV internal dir":

include dirs finally needed (all SYSTEM to keep ggml's -Werror clean):
  intel_gpu/include                      (public cldnn API)
  intel_gpu/src                          (ocl_engine.hpp, ocl_memory.hpp)
  intel_gpu/src/runtime                  (transitive includes)
  intel_gpu/src/graph/include            (transitive)
  onednn_gpu_install/include             (GENERATED dnnl_config.h lives here,
                                          not in onednn_gpu source tree)
  inference/dev_api                      (openvino/runtime/make_tensor.hpp ...)
  common/util/include                     (openvino/util/common_util.hpp)
  core/dev_api                            (openvino/runtime/itensor.hpp,
                                          openvino/core/log_util.hpp)
  common/itt/include                      (openvino/itt.hpp)
  common/transformations/include          (transformations/convert_precision.hpp)
  common/shape_inference/include          (gather_shape_inference.hpp ...)
  core/shape_inference/include            (matmul_shape_inference.hpp ...)
  thirdparty/ocl/clhpp_headers/include   (CL/opencl.hpp - bundled, avoids
                                          system cl2.hpp param_traits clash)
  thirdparty/ocl/cl_headers               (CL/cl.h)

compile defines must match the graph lib's build (ABI + struct layouts):
  ENABLE_ONEDNN_FOR_GPU, ENABLE_DEBUG_CAPS, GPU_DEBUG_CONFIG=1,
  OV_GPU_WITH_OCL_RT=1, OV_GPU_USE_OPENCL_HPP,
  OV_GPU_OPENCL_HPP_HAS_BUS_INFO, OV_GPU_OPENCL_HPP_HAS_UUID
  (last two fixed a cl::detail::param_traits redefinition in ocl_ext.hpp)
  CL_TARGET_OPENCL_VERSION=300, ENABLE_PROFILING_ITT,
  OV_THREAD=OV_THREAD_TBB_ADAPTIVE, DNNL_*, NGEN_CONFIG

glue sources compiled into our .so (stray symbols referenced by the graph lib):
  src/plugin/{variable_state,multi_tensor_variable_state,remote_context,
              remote_tensor,usm_host_tensor,common_utils,simple_math}.cpp
  + src/plugin/transformations/op/*.cpp  (op definitions: KVCache, Gemm,
    SDPA, SwiGlu, ... vtable/shape_infer symbols) - needed the two
    shape_inference include dirs above.

static libs linked (Release):
  openvino_intel_gpu_{graph,kernels,runtime}
  libopenvino_onednn_gpu.a  (from build/Release/.../onednn_gpu_install/lib)
  openvino_shape_inference, openvino_itt, pugixml
  openvino_reference  (SliceRange ctor + other reference symbols)
  openvino_util       (load_mmap_object, file_handle...)
  openvino_shutdown   (register_shutdown_callback)
  + openvino::runtime, openvino::threading, OpenCL::OpenCL
all wrapped in -Wl,--start-group / --end-group (heavy circular deps).

### 2026-07-30 - first successful run

configure: cmake --preset x64-linux-ovgpu-release   (clean)
build:     ninja llama-simple                         (links clean)
run:       GGML_OVGPU_SMOKE_TEST=1 ./llama-simple -m <dense.gguf> -p x -n 1

output:
  OVGPU smoke test (fully_connected 4x16x8): PASS (max_err=2.98023e-08)
  OVGPU: initialized device Intel(R) UHD Graphics 770
  llama_prepare_model_devices: using device OVGPU0 (...) - 59771 MiB free
  load_tensors: layer 0 assigned to device OVGPU0   (weights land in our buft)
  (then dies on tokenizer - synthetic test model has none; unrelated to backend)

What this proves:
- clDNN engine::create(engine_types::ocl, runtime_types::ocl) works standalone
  via device_query, no ov::Core.
- hand-built topology{input_layout, data, data, fully_connected} -> network ->
  set_input_data -> execute -> get_output_memory works, matching CPU ref.
- the backend registers, is detected by llama.cpp, and the device/buft path
  works end to end (layers assigned to OVGPU0).
- since supports_op=false everywhere, any actual compute would split to CPU;
  that's expected for M0.

Next (M1): start implementing op translators, validating each in
test-backend-ops. First op: MUL_MAT -> fully_connected.

### Open follow-ups (not blocking M0)
- get_memory returns free=total (PoC placeholder); needs a real free query.
- get_memory_statistics printed 0.00 MB right after smoke allocations - need to
  check whether stats track the smoke test's allocation type / timing; will
  verify once we have persistent weight buffers in M1.
- buffer set/get_tensor use a blocking cl::CommandQueue created at reg time;
  fine for M0, revisit when we move to async / a cldnn stream for compute.

## M1 - op translators (in progress)

### 2026-07-31 - MUL_MAT -> fully_connected

Infrastructure built:
- ggml-ovgpu.hpp: op_cache (shape-keyed compiled-network cache, thread-safe via
  mutex), compiled_op struct, ggml_layout_for / wrap_tensor helpers.
- ggml-ovgpu-ops.cpp: build_op dispatch + build_mul_mat.
- graph_compute: per-node loop, wraps tensors into cldnn::memory, binds via
  set_input_data/set_output_memory, executes.

Key discoveries during bring-up:
- GPU buffer model MUST use USM device memory (not cl_mem): cl_mem is a handle,
  not a pointer, so get_base can't return it for ggml's offset arithmetic.
  usm_device gives a real pointer (0xffff... range, GPU device memory).
  This iGPU (UHD 770) does NOT support host-accessible usm_shared, so host
  access must go through the runtime (not direct memcpy to the USM ptr).
- set/get_tensor: copy_from/copy_to (clEnqueueMemcpyINTEL) caused heap
  corruption; switched to mem_lock (read_write for set to preserve other
  tensors in the same buffer, read for get). This is the smoke-test-proven path.
- Cross-stream sync: set/get_tensor use reg_ctx->stream; network::execute uses
  the network's own stream. Added reg_ctx->stream->finish() before compute and
  net->get_stream().finish() after each execute (device mem is coherent across
  streams on the same context).
- Layout mapping: ggml ne[k,m] -> cldnn tensor(ne1=m, ne0=k, 1, 1) for 2D
  (batch=ne1, feature=ne0) matches the smoke test and the legacy translator's
  MatMul(act, weight, transpose_b=true) semantics.
- MUL_MAT: weight=src[0] -> FC weights (transposed), activation=src[1] -> FC input.
  input_ids follow ggml src order so graph_compute binds src[k]->input_ids[k].

Result: 9/17 f32/f32 MUL_MAT cases PASS in test-backend-ops. The 2D matmul path
works for most shapes.

OPEN ISSUE: specific shapes fail with LARGE error (not rounding):
  m=16, n=1, k=256  (ERR ~61-75)  <- GEMV path (M=1)
  m=16, n=8, k=256  (ERR ~2)
  m=16, n=1, k=4    (ERR ~1.7)
Deterministic per-shape (not flaky) -> kernel-selection / output-layout mismatch
for certain M values, not the core mapping. Needs investigation: which cldnn FC
kernel is selected for M=1 (GEMV) vs M=8, and whether the output memory layout
matches what those kernels produce (bfyx vs blocked). Likely set_output_memory
layout mismatch for specific impl selections.

### 2026-07-31 - instrumenting FC kernel selection (OV source)

User asked to instrument the OV kernel-selection code. Added a gated trace
(OVGPU_FC_TRACE=1) to ocl FC impl ctor in OV source:
  src/plugins/intel_gpu/src/graph/impls/ocl/fully_connected.cpp
prints kernel name + weight-reorder. Rebuild: ninja -C openvino/build/Release
openvino_intel_gpu_graph -> updates bin/intel64/Release/libopenvino_intel_gpu_graph.a
-> relink llama. (Left in place, env-gated, useful for future debugging.)

Findings:
- supports_immad=0 on UHD 770 -> FC uses the ocl kernel_selector path (NOT oneDNN).
- ALL 2D f32 cases (passing AND failing) select fully_connected_gpu_bfyx_ref,
  weight_reorder=0. Kernel selection is NOT the differentiator. n=7 picks a
  different kernel (fb_oi_b8_fp32_ref) but passes.
- mem_lock read_write on usm_device does proper read-modify-write (copies
  device->host staging on lock, host->device on unlock) - preserves other
  tensors. NOT the bug.
- set_output_memory is not the bug (output-copy diagnostic gave same pass/fail).

REAL BUG FOUND: build_mul_mat guard `ggml_n_dims(act) != 2` rejected the n=1
(1D activation) case, but supports_op accepted it (`n_dims > 2`). supports_op
said yes -> scheduler assigned to OVGPU -> graph_compute hit "no translator"
-> wrong output. Fixed by aligning build guard to `ggml_n_dims(act) > 2`.
Result: 9/17 -> 12/17.

REMAINING: n=8,k=256 fails deterministically (ERR ~2.2) even in isolation, same
bfyx_ref kernel, plausible output values. Genuine per-shape edge case, cause
unknown - candidate: output layout interpretation for batch=8, or ref-kernel
numerical edge. Needs exec-graph dump (OV_GPU_DUMP_GRAPHS_PATH) or deeper
output-layout comparison. Not blocking; 12/17 is solid for the core path.

### 2026-07-31 - n=8 root cause + fix -> 17/17

Root cause of the n=8 (batch=8) failure, found via element-by-element CPU ref
check + network output-layout dump:
- For batch=8, cldnn's layout optimizer produces FC output in format yxfb, while
  for batch=16 (and most others) it's bfyx. yxfb and bfyx are TRANSPOSES of each
  other for a 2D tensor (yxfb: elt(b,f) at f*B+b; bfyx: at b*F+f) - NOT the same
  physical layout. I bound/read the output as bfyx (ggml's layout), so the batch=8
  output came out transposed/garbled (126/128 elements wrong, maxerr ~20).
- The bfyx_ref kernel itself is scalar and correct; the bug was the format mismatch
  between what cldnn emits and what ggml expects.

Fix: append a trailing cldnn::reorder primitive after FC, forcing bfyx output
(= ggml_layout_for(node)). The reorder canonicalizes whatever format cldnn picks
back to ggml's bfyx. Topology: input_layout x2 -> fully_connected("fc_out") ->
reorder("out", bfyx).

Result: 17/17 f32/f16/bf16 2D MUL_MAT cases PASS in test-backend-ops (all cases
supports_op currently accepts). 0 failures, clean exit.

M1 MUL_MAT is correct for the supported shape set (2D weight, 1-2D activation,
same dtype f32/f16/bf16). Not yet supported: mixed dtypes (f16 weight + f32 act,
the common LLM case), ne[2]/ne[3] batching, quantized weights - those gate real
model inference and are the next M1 work. The trailing-reorder pattern should be
applied to every op translator (cldnn format != ggml layout in general).

### 2026-07-31 - mixed dtype + batched activation -> 65/65

Extended MUL_MAT to the dtype/shape set real dense layers need.

Key facts established from the OV source + ggml:
- ggml_mul_mat ALWAYS creates an f32 dst (ggml.c:3286), and broadcasts the weight
  over the activation batch dims (constraint: act ne[2]%w ne[2]==0, ne[3]%w ne[3]==0).
- ggml widens f16/bf16 to f32 and accumulates in f32 (ggml_vec_dot_f16 -> float,
  widened svfloat32_t). So the reference is f32-accumulated for every dtype combo.
- cldnn's bfyx_ref FC accumulates in ACCUMULATOR_TYPE = the activation (input0)
  dtype (GetAccumulatorType in fully_connected_kernel_base.cpp; bfyx_ref.cpp:69-80).
  So an f32 activation -> f32 accumulation = ggml semantics, with native f16/bf16
  weights read+converted per element (FILTER_TYPE -> ACCUMULATOR_TYPE in the JIT).

Changes:
- Layout mapping generalized to FLATTEN ggml dims 1,2,3 into the cldnn batch dim:
  ggml_ne_to_tensor now returns tensor(ne1*ne2*ne3, ne0, 1, 1). For contiguous
  tensors this matches ggml's physical memory exactly (one flat ordering serves a
  2D weight, a batched activation, and the batched dst). Backward-compatible: 2D
  operands (ne2=ne3=1) map to the same tensor as before.
- supports_op (MUL_MAT): dropped the same-dtype and act-2D limits. Now allows weight
  in {f16,bf16,f32} x activation in {f16,bf16,f32}, weight 2D only (group broadcast
  = GQA still deferred), operands must be contiguous (rejects ggml_permute views and
  strided k_v views), K must match.
- build_mul_mat: inserts an activation->f32 reorder (for f32 accumulation) and now
  ALSO a weight->f32 reorder (see perf caveat). Topology built with topo.add() so
  the reorders are conditional. f32 operands skip their reorder.

PERF CAVEAT (important): f16/bf16 weights bound as a plain-bfyx input_layout hit
cldnn's ocl kernel path (UHD 770: supports_immad=0 -> no oneDNN) and are mis-read
for some batch sizes - same bfyx_ref kernel that works for f32 weights gives garbage
for f16 at batch=8, and at batch=9 NO kernel is selected (crash). Confirmed via
OVGPU_FC_TRACE: f16-weight n=8 selects bfyx_ref weight_reorder=0 yet outputs ERR~1;
f16-weight n=9 -> kernels.empty(). So weights are upcast to f32 per compute (a full
matrix reorder every token). This is correct but NOT the intended fast path. M2 fix:
bake weights natively (data nodes / the kernel's expected blocked layout) so the
selected kernel reads them in place - needs per-layer (not shape-keyed) weight data,
i.e. a cache keyed by (shape, layer) or per-layer networks. On a supports_immad=1
device the oneDNN path should make native f16 Just Work; revisit there.

Deferred to later (reported NOT_SUPPORTED by supports_op, fall back to CPU - not
failures): grouped weights (ne[2]/ne[3] > 1 = GQA broadcast; the attention-score
matmul), permuted operands, strided-K views, quantized types.

Result: 65/65 supported MUL_MAT cases PASS in test-backend-ops (0 fail, clean exit).
Covers f32/f16/bf16 weights x f32/f16 act, weight 2D, activation batched over
ne[2]/ne[3] (nr=[2,1],[1,2] -> prompt-processing / multi-token). This is every
non-attention MUL_MAT a dense f16 model needs.
