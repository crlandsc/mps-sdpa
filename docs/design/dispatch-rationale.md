# Why this package exists

PyTorch's MPS backend has scaled-dot-product-attention support, but it
doesn't take advantage of Apple's dedicated SDPA op. This document captures
the dispatch survey that established that gap and motivated `mps-sdpa`.

## The finding

PyTorch ships three SDPA kernels for the MPS backend, all C++ and all
selected by shape:

- **`sdpa_general_mps`** — the general path. Builds and caches an
  `MPSGraphExecutable` per shape/dtype signature. This is the path most
  realistic transformer attention shapes route through.
- **`sdpa_vector_fast_mps`** — fast path for very small vector-style
  attention (Lq = 1).
- **`sdpa_vector_2pass_mps`** — variant of the above for two-pass softmax.

The `sdpa_general_mps` path constructs a **naive** MPSGraph: matmul →
softmax → matmul. It does **not** call `MPSGraph.scaledDotProductAttention`,
the dedicated fused op Apple ships from macOS 15 onward.

Empirical confirmation: a forward pass at `(B=1, H=8, Lq=Lkv=4096, D=64)`
in bf16 takes ~74 ms under the default backend and ~74 ms under a forced
`SDPBackend.MATH` path — the ratio is 0.999, indicating both paths are
doing essentially the same work.

## What that means for performance

When you call `F.scaled_dot_product_attention` on an MPS tensor:

- Short sequences (L ≤ 256 or so): fast, well-tuned. No problem.
- Long sequences (L ≥ 1024): the naive matmul path materializes the
  full Lq × Lkv attention matrix, which is bandwidth-bound and slow,
  and it allocates that full matrix in driver memory.

This is exactly the regime where Apple's dedicated fused op (which
doesn't materialize the attention matrix) wins big — and exactly where
PyTorch on MPS is leaving performance on the table.

## What `mps-sdpa` does

The package wraps `MPSGraph.scaledDotProductAttentionWithQueryTensor:...`
through two complementary paths:

- **`mpsgraph_zc`** (default) — a C++/Objective-C++ torch extension.
  Uses ATen's `getMTLBufferStorage` to pass torch tensors directly to
  `MPSGraph` with no CPU copies. Forward + backward both run as compiled
  MPSGraph executables; backward is a manually-built graph (recompute +
  4 matmuls + manual softmax-bwd) since the fused op doesn't expose
  attention weights for autograd.

- **`mpsgraph`** — a pyobjc-based fallback for environments where the
  C++ extension can't build. Same graph structure, but tensor data is
  routed through CPU memcpy (slower, but no compile-time dependency).

A short-sequence threshold dispatches to stock for cases where the
copy/setup overhead would exceed the compute savings. The threshold is
auto-calibrated per `(chip, os, torch)` on first import.

The result is a drop-in replacement for `F.scaled_dot_product_attention`
that closes most of the CUDA-vs-MPS attention performance gap on Apple
silicon, with no model-side code changes required other than swapping
the function call.

## Out-of-scope packages

For completeness, two related projects exist that this package does not
wrap or depend on:

- [`philipturner/metal-flash-attention`](https://github.com/philipturner/metal-flash-attention) —
  Swift FlashAttention port; last updated 2023.
- [`mps-flash-attn`](https://pypi.org/project/mps-flash-attn/) — a small
  PyTorch bridge for the above.

Neither is evaluated, wrapped, or used here. `mps-sdpa` is an independent
implementation that wraps Apple's officially-supported
`MPSGraph.scaledDotProductAttention` op directly.
