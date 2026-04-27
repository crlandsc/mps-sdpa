# Design notes

Background and rationale for `mps-sdpa`. These notes explain why the package
exists, how it performs, and what design choices were made along the way —
useful for contributors who want context beyond the source code.

| Doc | Summary |
|---|---|
| [dispatch-rationale.md](dispatch-rationale.md) | Why this package exists. PyTorch's MPS backend doesn't dispatch `scaled_dot_product_attention` to Apple's fused `MPSGraph.scaledDotProductAttention` op — it builds a naive matmul → softmax → matmul graph instead. This package wraps the fused op directly. |
| [benchmarks.md](benchmarks.md) | Performance numbers. Per-shape inference, training, and dropout speedups vs stock; driver memory reduction; correctness tolerances. M4 / macOS 26 / torch 2.13 nightly, bf16. |
| [custom-kernel-experiment.md](custom-kernel-experiment.md) | Why we don't ship a hand-written Metal kernel. A naive Metal implementation came in ~60× slower than the zero-copy bridge to Apple's already-tuned op; a full FlashAttention-2 kernel was deemed not worth the effort. The probe is preserved as the `metal_proto` backend (registered, not auto-selected). |

For user-facing API docs, see the top-level [README.md](../../README.md).
For internal architecture, see [ARCHITECTURE.md](../../ARCHITECTURE.md).
For compatibility matrix, see [COMPAT.md](../../COMPAT.md).
