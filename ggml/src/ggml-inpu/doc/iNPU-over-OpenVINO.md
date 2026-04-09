# Why iNPU Is a Better Backend Than OpenVINO for Intel NPU Support in llama.cpp

**Author:** Zijun Yu  
**Date:** March 2026

---

## Executive Summary

The OpenVINO (OV) backend was developed to bring a unified backend to llama.cpp that can leverage CPU, GPU, and NPU together through OpenVINO. The iNPU backend was developed specifically to bring Intel NPU support to llama.cpp. After building and testing both, the iNPU backend is a fundamentally better approach for NPU acceleration. It aligns with llama.cpp's backend contract, is dramatically simpler to maintain, passes tests with the same code path used in production, and can run every model llama.cpp supports — not just the ones hand-tuned for OpenVINO's graph translator. The only trade-off is raw single-stream performance, which is addressable incrementally.

This document presents a systematic comparison across architecture, correctness, maintainability, model coverage, and performance.

---

## 1. Architectural Alignment with llama.cpp

### How llama.cpp backends work

llama.cpp backends are **kernel-centric and step-dynamic**:

1. Each decoding step, the runtime builds a `ggml_cgraph` from scratch.
2. The backend scheduler splits the graph into subgraphs if the specified backend cannot run the full cgraph, and assigns each subgraph to a backend.
3. Each backend executes the subgraph it is given — typically by iterating over ops and dispatching kernels.
4. Buffers are managed by ggml; backends provide allocation and data-transfer implementations.

This contract assumes backends can handle small, rapidly-changing subgraphs and that ops fall back gracefully when a backend does not support them.

### OpenVINO backend: whole-graph translation

The OpenVINO (OV) backend translates the **entire** `ggml_cgraph` into an OpenVINO IR, compiles it, and runs it as a monolithic unit. This creates a fundamental mismatch:

- **It must translate every op in the graph.** If even one op is unsupported, the backend cannot execute the graph. The fallback mechanism (letting unsupported ops execute elsewhere) is effectively broken because the OV backend's translation pipeline assumes it will convert the entire graph — it cannot partially translate a cgraph and hand the rest back to another backend.
- **It must infer semantic roles of tensors by pattern matching** — identifying `inp_pos`, `inp_tokens`, `self_kq_mask`, KV cache parameters, etc. by inspecting the graph structure and op neighborhoods (e.g., `is_inp_tok` checks that `op == GET_ROWS && tensor == src[1] && src[0]->op == NONE`). These heuristics are fragile and break when graph structure changes.
- **Dynamic dimensions require special handling.** The OV backend must figure out which tensor dimensions vary across decoding steps (sequence length, KV cache size) and create symbolic shapes or separate static/dynamic IR variants. This is a substantial amount of inference logic that duplicates what the llama.cpp runtime already manages.
- **NPU requires separate prefill/decode IR compilation** with chunked prefill, stateful KV management, and padding logic — all implemented manually in the backend's decoder layer.

Total: **~7,850 lines across 45 source files**, including a full OpenVINO frontend implementation (`frontend.cpp`, `input_model.cpp`, `translate_session.cpp`), a graph decoder (`ggml-decoder.cpp` at 1,241 lines), 16 individual op translators, graph optimization passes (`fuse_to_sdpa`, `squeeze_matmul`), and quantization handling (`ggml-quants.cpp` at 956 lines).

### iNPU backend: op-level acceleration

The iNPU backend registers as an **ACCEL** backend — it only claims ops it can efficiently accelerate, and everything else falls back to CPU naturally via the backend scheduler.

- **No semantic inference needed.** The backend does not need to know what `inp_pos` or `self_kq_mask` is. It sees a small subgraph (e.g., `MUL_MAT` or `MUL_MAT → ADD`) and translates just those ops.
- **No dynamic shape handling.** Each small subgraph has concrete shapes at execution time. The backend translates exactly what it sees, compiles it, caches it, and runs it.
- **Fallback is automatic.** If a model introduces an op the iNPU backend cannot handle, it simply does not claim it, and the CPU backend runs it. No code changes needed.

Total: **~3,615 lines across 10 source files.** The entire translation layer is a single file (`ggml-inpu-translate.cpp`, 1,404 lines) instead of 16 separate op files plus a decoder, frontend, and translation session.

### Quantitative comparison

| Metric | OpenVINO Backend | iNPU Backend |
|--------|-----------------|-------------|
| Total source lines | ~7,850 | ~3,615 |
| Source files | 45 | 10 |
| Op translator files | 16 separate files | 1 file with inline functions |
| Graph optimization passes | 2 custom OV passes | 0 |
| Frontend/decoder layers | Full OV frontend impl | None |
| Semantic pattern matching | Extensive (tensor name/role inference) | None |
| Dynamic shape handling | Complex (symbolic dims, static/dynamic split) | None (shapes are concrete) |

---

## 2. Correctness and Testability

### Same code path for testing and inference

The iNPU backend uses **exactly the same execution path** for `test-backend-ops` and real model inference. A single `MUL_MAT` test exercises the same translation, compilation, and execution code that runs during model inference. This means test coverage directly validates production behavior.

The OpenVINO backend has **divergent code paths**:

- `test-backend-ops` and small graphs go through `naive_compute()`, which creates a fresh OV model per call, compiles it, runs it, and copies outputs back.
- Real model inference goes through `ov_graph_compute_dynamic()` (CPU/GPU) or `ov_graph_compute_static()` (NPU), which use complex caching, the `GgmlOvDecoder`, and pattern-based tensor identification.
- The quantization path differs between testing and inference: `test-backend-ops` uses a bias-based computation path, while normal inference extracts zero-points and uses `zp`-based computation (as noted in the OV summary slides).

This divergence means **passing `test-backend-ops` does not guarantee correct model inference**, and vice versa.

### Quantization consistency

The OV backend performs requantization of Q4_0/Q8_0 weights from group size 32 to group size 128 for the NPU path, sacrificing numerical precision for performance. It also uses a zero-point-based dequantization scheme that differs from ggml's bias-based formula.

The iNPU backend **preserves the original group size of 32** and performs a straightforward u4→i4 conversion (XOR with 0x88 to subtract 8 from each nibble). The dequantization formula is simply `Convert(i4 → f16) × scale`, matching ggml's semantics exactly. This simplicity is a deliberate design choice: the iNPU backend only supports Q4_0 and Q8_0, which are symmetric quantization formats that the NPU handles natively. By narrowing the scope to formats that align with hardware capabilities, the backend avoids the need for requantization or zero-point reinterpretation entirely.

### CI stability

The iNPU backend passes tests with minimal effort. The OpenVINO backend required significant work to stabilize CI, and `test-backend-ops` on NPU is still failing in the OV implementation.

---

## 3. Model Coverage

### iNPU: runs any model llama.cpp supports

Because the iNPU backend only accelerates specific ops and falls back gracefully, it can run **any model** that llama.cpp supports. Tested models include:

- Llama 3.2
- Phi-3
- LFM2
- Qwen 3.5

Adding support for new models requires **zero changes** to the backend — as long as the model uses `MUL_MAT` (which every model does), the backend accelerates it.

### OpenVINO: limited to hand-tuned models

The OV backend must translate every op in the graph. When a new model introduces ops or patterns the backend has not been tuned for, it fails. Enabling new models is difficult because:

1. New ops require new translator files and registration in the op table.
2. The graph decoder's tensor identification heuristics may not match the new model's graph structure.
3. Dynamic dimension handling may need to be extended for different axis patterns.
4. The NPU path (static IR) needs separate prefill/decode handling for each new pattern.

Supporting Qwen 3.5 — which includes multimodal capabilities and an SSM component — is a concrete example where the OV backend struggles and the iNPU backend works out of the box.

---

## 4. Maintainability and Upstream Compatibility

### Resilience to upstream changes

llama.cpp's graph generation evolves frequently — new models, new op arrangements, graph optimizations. The iNPU backend is resilient to these changes because:

- It does not depend on specific graph structures or tensor naming conventions.
- It only needs to translate the ops it claims to support.
- New ops simply fall back to CPU.

The OV backend is brittle against upstream changes because:

- The `GgmlOvDecoder` infers tensor roles from graph patterns (`is_inp_tok`, `is_inp_pos`, `is_inp_mask`, etc.), which are tightly coupled to how the runtime constructs graphs.
- The translation session preprocesses the graph with assumptions about KV cache layout, mask slicing, and RoPE frequency factors.
- The `compute_llm_params` function reverse-engineers model parameters from the graph structure.

Any change in how llama.cpp builds its compute graph can break these heuristics.

### Tensor identification: pointers vs. names

The OV backend uses **tensor names as keys** in many places (weight lookups, input/output mapping, KV cache identification). As noted in the design documents, this is not robust — tensor names can change, may not be unique, and are not part of the backend API contract.

The iNPU backend uses **tensor pointers** as keys throughout. The `ggml_tensor *` pointer is the authoritative identity of a tensor within a graph computation. Parameter names in the OV model are generated from structural indices (`input_n{node}_s{src}`, `output_n{node}`) and only serve as handles for the OV runtime — they are never used to infer semantic meaning.

### Code review surface

When reviewing a PR, a smaller and more self-contained codebase is easier to evaluate. The iNPU backend is less than half the size of the OV backend, has no custom OV frontend implementation, no graph optimization passes, and no pattern-matching heuristics. A reviewer can understand the entire backend by reading four files:

1. `ggml-inpu.cpp` — buffer management, backend registration, graph dispatch
2. `ggml-inpu-translate.cpp` — OV IR generation
3. `ggml-inpu-cache.cpp` — compiled model caching
4. `ggml-inpu-impl.h` — shared data structures

---

## 5. Performance

### Current state

The iNPU backend achieves performance **on par with the CPU backend** for F16 weights:

| Tool | CPU (ms/token) | iNPU (ms/token) |
|------|---------------|----------------|
| llama-simple (Llama-3.2-1B-F16) | ~40 (25 tok/s) | ~41 (24 tok/s) |
| llama-simple (Llama-3.2-1B-Q4_0) | ~15 (67 tok/s) | ~45 (22 tok/s) |

The F16 path is essentially at CPU parity. The Q4_0 path is significantly slower than CPU — this is expected because the CPU backend has highly optimized GEMM kernels for Q4_0 dequantization at group size 32, while the Lunar Lake NPU's best-supported 4-bit configuration uses group size 128 or channel-wise quantization. The iNPU Q4_0 performance (~22 tok/s) is comparable to native OpenVINO GenAI inference with group-size-32 quantized models on the same NPU, indicating the backend is extracting close to what the hardware can deliver for this quantization format.

### Performance advantage in specific scenarios

However, the iNPU backend has a **structural advantage over the OV backend** for larger context sizes. When context grows (e.g., `llama-cli -c 4096`), the OV backend's performance degrades dramatically — down to ~1 token/second. This is because the OV backend must build a **static IR** for the NPU, which means the attention subgraphs in the decode IR attend to the full KV cache capacity regardless of how many tokens are actually valid. As context size increases, the NPU wastes compute on padding in every attention operation.

The iNPU backend maintains ~15 tokens/second in the same scenario because it does not offload attention at all. The attention mechanism runs on CPU, where it naturally handles only the valid portion of the KV cache with no wasted compute on padding.

---

## 6. Design Decisions That Enable This Architecture

### ACCEL backend type

By registering as `GGML_BACKEND_DEVICE_TYPE_ACCEL`, the iNPU backend lets the scheduler split the compute graph and offload only supported subgraphs. This is the same approach used by other accelerator backends in llama.cpp (e.g., Hexagon/QNN for Qualcomm NPU).

### Extra buffer type for weight preparation

The backend uses `ggml_backend_dev_get_extra_bufts` to register a custom buffer type. This allows the weight loading pipeline to detect iNPU-supported buffers and route quantized weights through the reformatting path (`set_tensor` with u4→i4 conversion and planar layout separation) at load time. Weights are ready to use as soon as loading completes — no per-inference transformation needed.

### Subgraph caching

The `inpu_cache` stores compiled OV models keyed by `inpu_graph_key` (a structural fingerprint of the subgraph). Since the same matmul shapes appear repeatedly across layers and decoding steps, the cache achieves a very high hit rate after the first few tokens. The cache also pools `InferRequest` objects to avoid repeated allocation.

### Graph I/O analysis using use counts

The backend uses `ggml_node_get_use_count` to determine which tensors are internal intermediates vs. external outputs. This is a clean, robust mechanism — it compares in-subgraph consumption against total consumption — rather than relying on tensor names or pattern matching.

---

## 7. Conclusion

| Criterion | OpenVINO Backend | iNPU Backend | Winner |
|-----------|-----------------|-------------|--------|
| Architectural alignment | Whole-graph translation, semantic inference | Op-level acceleration, natural fallback | **iNPU** |
| Codebase complexity | ~7,850 LOC, 45 files | ~3,615 LOC, 10 files | **iNPU** |
| Test/production consistency | Divergent paths | Identical path | **iNPU** |
| Model coverage | Requires per-model tuning | Works with any model | **iNPU** |
| Upstream resilience | Fragile (pattern-dependent) | Robust (op-level only) | **iNPU** |
| Single-stream perf (1B-F16) | ~25 tok/s (OV NPU) | ~24 tok/s | Tie |
| Large-context perf | ~1 tok/s (degrades) | ~15 tok/s (stable) | **iNPU** |
| Quantization correctness | Requantizes; divergent test/inference paths | Preserves group size; single path | **iNPU** |

The iNPU backend matches the CPU backend on single-stream throughput while offering dramatically better architecture, correctness guarantees, model coverage, and maintainability compared to the OV backend. Given that llama.cpp already has strong CPU, SYCL, and Vulkan backends for Intel hardware, and the primary contribution of adding an NPU backend is expanding device coverage rather than replacing existing paths, **correctness and sound architecture should take precedence over raw performance**. Once we have a stable and maintainable backend, performance can be incrementally improved.
