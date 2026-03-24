#import "@preview/touying:0.6.1": *
#import themes.simple: *

#show: simple-theme.with(aspect-ratio: "16-9")
#set text(size: 20pt)

= OpenVINO Backend for llama.cpp

Zijun Yu
#v(0em)
March 4, 2026

== Introduction
- Goal: enable llama.cpp inference on Intel ecosystem via OpenVINO (CPU/GPU/NPU).
- Immediate value: leverage OpenVINO runtime optimizations and device portability.
- *Current status: backend is functional for llama-arch models, with meaningful performance on GPU.*
#v(0.8em)
#pause
- *Maintainer is open to merging the PR and marking OpenVINO as an in-progress backend.*
- Long-term success depends on a generic design that fits llama.cpp, whisper.cpp, and stable-diffusion.cpp style runtimes.

== Current status
- All tools (default config) are working on CPU + GPU:
  - `llama-simple`, `llama-cli`, `llama-server`, `llama-bench`, `llama-perplexity`
- CI is passing for CPU and GPU locally.
- GPU `llama-bench` is close to Vulkan on 8B Llama models.
#v(0.8em)
*NPU constraints*
- No `llama-server -np > 1` (parallel sequences unsupported).
- `llama-perplexity` only with `-b 512`.
- NPU CI is still in progress; `test-backend-ops` is currently failing.

== How llama.cpp runtime expects backends to work
- GGUF model files contain only weights and metadata, no graph topology/ops.
- For each decoding step, llama.cpp builds a ggml `cgraph`.
- Backends are expected to execute *ops/kernels* against provided buffers.
#pause
- Buffer management contract in practice:
  - ggml defines and manages buffer roles/lifetimes (weights, KV cache, compute tensors).
  - backend provides the buffer allocation implementation for those ggml buffers.
#pause
- Typical backend model:
  - op-by-op execution.
  - direct read/write into ggml memory.
- *Implication:* backend contract is kernel-centric and step-dynamic.

== OpenVINO execution model (integration challenge)
- OpenVINO is graph-centric and AOT-leaning:
  - build IR graph.
  - compile graph.
  - execute compiled artifact.
#pause
- Compilation is expensive relative to per-token decode.
- Recompiling every step is not feasible.
- *Core challenge:* llama.cpp treats each step as dynamic op execution, while OpenVINO wants a stable compiled graph artifact.

== What the OpenVINO backend does today
- Converts ggml `cgraph` to OpenVINO IR.
- CPU/GPU path: use dynamic IR (symbolic dimensions + extra runtime scalar inputs).
- NPU path: use static IR variants.
#pause
- Cache compiled models to avoid recompiling on each token.
- Examples of extracted dynamic values:
  - token length and related shape axes.
  - KV cache slice/view sizes (e.g., attention window / past length dependent).

== Why this needs careful scaling
- Backend infers semantics from `cgraph` patterns (e.g., locate input token tensors, mark selected dims dynamic).
- Depends on assumptions about graph shape and op arrangement.
#pause
- llama.cpp graph details can evolve (new models, new op arrangements, etc).
- Pattern-based graph translation creates maintenance burden and regression risk.
- *Risk:* functionality can break as upstream graph generation changes.

== Buffer management issue
- llama.cpp still allocates compute buffers for intermediates.
- OpenVINO backend does not consume those compute buffers in the normal kernel-style way.
#pause
- Weights are loaded into backend buffers and mapped into `ov::Constant` nodes.
- Compiled model / infer request may internally duplicate weight storage.

== Quantization compatibility gap
- OpenVINO does not natively support 5-bit / 6-bit quant.
- Current mapping uses int8 OV tensors.
#pause
- ggml quant uses floating-point bias-style correction for quantized weights.
- OpenVINO best-supported path is quantized zero-point (`zp`) based computation.
#pause
- Execution-path inconsistency:
  - `test-backend-ops` (single-op utility) uses a separate compute path where bias is used.
  - normal model execution path extracts/derives `zp` and uses `zp` in computation.
  - single-op validation behavior differs from end-to-end model execution behavior.

== What this means for long-term maintainability
- Short-term: backend is useful and delivers value on supported models/devices.
- Medium/long-term risk drivers:
  - contract mismatch (kernel API vs graph compiler backend).
  - dynamic graph handling complexity.
  - backend-specific assumptions on `cgraph` internals.
  - memory model mismatch.
#pause
- Possible next steps:
  - keep current backend for immediate value.
  - collaborate upstream on minimal API hooks reducing heuristic assumptions.

== Conclusion
- OpenVINO backend is already delivering practical value for llama.cpp on CPU/GPU.
- Main challenge is not correctness of individual ops; it is *architectural compatibility* between runtime contracts.
- Near-term path: keep it in-progress, scoped, and tested.
- Long-term success likely requires incremental runtime/backend interface evolution.

// == Appendix A) 30-second elevator pitch
// "We made OpenVINO work in llama.cpp today, with good GPU results and real tool coverage. The hard part is that llama.cpp backends are designed for per-op execution, while OpenVINO is designed for compiling and running graphs. Our current dynamic-IR translation works, but it relies on graph assumptions that are costly to maintain as upstream evolves. The right path is to keep this backend as in-progress with explicit scope now, while we reduce integration risk through better contracts and targeted upstream hooks."