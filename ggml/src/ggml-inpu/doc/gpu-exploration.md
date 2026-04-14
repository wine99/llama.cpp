# GPU Backend Exploration — Findings

## Problem

Running with `GGML_INPU_DEVICE=GPU` is very slow. VTune profiling shows a long
sequence of permute/reorder kernels before GEMM kernel begins.

## Root Cause

The OpenVINO GPU plugin requires weights in a hardware-specific blocked memory
layout (e.g. `oiyx`, `os_is_yx_isv16_osv16`) for efficient SIMD access in its
FullyConnected/MatMul OpenCL kernels. When the weight layout doesn't match, the
plugin inserts **reorder primitives** into the execution graph.

The iNPU backend passes all weights as `ov::op::v0::Parameter` nodes (dynamic
inputs bound at inference time). The GPU plugin's optimization pipeline handles
this as follows:

1. **`post_optimize_weights`** — inserts a reorder node between every weight
   Parameter and its consumer (MatMul/FC), regardless of whether the input is a
   Constant or Parameter.

2. **`propagate_constants`** (pass 26) — executes reorder nodes at compile time
   and caches the result, but **only for nodes where `is_constant() == true`**.
   Parameters fail this check, so they are skipped.

3. **Result** — the reorder nodes remain in the runtime execution graph and
   **re-execute on every `infer()` call**, even though the weight data never
   changes between calls.

The GPU plugin pre-allocates persistent output memory for these reorder
primitives at build time (`network::allocate_primitives`), so the GPU-side
buffer exists regardless. The per-inference cost is purely the reorder kernel
execution, not memory allocation.

## Alternatives Considered

### Use `ov::op::v0::Constant` for weights

OpenVINO provides a zero-copy Constant constructor:
```cpp
Constant(const element::Type& type, const Shape& shape,
         const void* data, std::shared_ptr<void> so);
```
This would wrap the ggml buffer data without copying, and `propagate_constants`
would pre-reorder the weights at compile time, eliminating runtime overhead.

**Problem:** breaks the iNPU compilation cache. The backend currently splits the
model into small cgraphs (one per accelerated subgraph). Many layers produce
identical cache keys because they share the same op sequence and shapes. With
Constant nodes, the weight data becomes part of the compiled blob, making each
layer's compilation unique and defeating cross-layer cache reuse.

### Pre-reorder weights in the ggml buffer

Not feasible. The target layout is an OpenVINO GPU-plugin internal detail (tied
to kernel selection, SIMD width, device capabilities). There is no public API to
query the required layout ahead of time, and the format may differ across GPU
generations.

## Conclusion

There is no straightforward way to avoid the per-inference weight reorder cost
on GPU while preserving the current caching strategy. The NPU path does not have
this issue because we know the exact layout that NPUW expects: F16 weights need
no reorder, and quantized weights use a groups-first layout (`[n_groups, N, gs]`)
that the iNPU backend prepacks during `set_tensor`. No runtime reorder is needed.
