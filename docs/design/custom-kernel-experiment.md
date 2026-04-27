# Custom Metal kernel experiment

The `metal_proto` backend is an intentionally-naive custom Metal SDPA kernel
written via `torch.mps.compile_shader()`. It is **correct but not fast** —
significantly slower than the `mpsgraph_zc` zero-copy bridge to Apple's fused
op. It ships as a reference / starting-point for anyone investigating a
fully-custom Metal attention kernel.

## TL;DR — why the default isn't a custom kernel

A naive Metal kernel (one thread per output row, online softmax in registers,
single pass over `Lkv`) comes in **~60× slower** than `mpsgraph_zc` on M4 /
bf16 / typical transformer shapes. A FlashAttention-2-style kernel that
could compete would need real specialized work — proper threadgroup-shared
memory tiling, simdgroup matrix ops, parallel softmax reduction — and even
then it isn't obvious it would beat Apple's already-tuned fused op.

For a drop-in replacement with broad shape coverage, wrapping the fused op
(what `mpsgraph_zc` does) is the higher-value path. `metal_proto` stays in
the codebase as a pinned correctness probe and a starting point for
experimentation — it is registered as available but is **not** auto-selected.
To exercise it directly:

```python
from mps_sdpa import sdpa_opt
out = sdpa_opt(q, k, v, backend="metal_proto")
```

## What the kernel does

A single-pass online-softmax SDPA. One thread per `(batch * head, q_idx)`
output row:

- Loads `Q[q_idx, :]` into per-thread registers (D ≤ 256 enforced for
  register budget).
- Streams over `Lkv`: for each `k_idx`, computes the dot-product score,
  updates a running `(max, denom)` pair using the standard online-softmax
  recurrence, accumulates the running output `Σ exp(s − max) · V[k_idx]`.
- Final divide by the running denom and write to output.

No tiling, no threadgroup-shared memory, no simdgroup matrix ops, no
collaborative loads. Correctness verified against stock SDPA: max diff
9.77e-4 in bf16 (well within the 5e-3 tolerance band for that dtype).

## Measured performance (M4, bf16, B=1, H=8, D=64)

| L    | stock     | mpsgraph_zc | metal_proto (naive) | naive / stock | naive / zc |
|------|-----------|-------------|---------------------|---------------|------------|
| 512  | 1.57 ms   | 0.39 ms     | 25.62 ms            | 0.06×         | 0.02×      |
| 1024 | 4.94 ms   | 0.93 ms     | 96.86 ms            | 0.05×         | 0.01×      |
| 2048 | 18.10 ms  | 3.34 ms     | 381.11 ms           | 0.05×         | 0.01×      |
| 4096 | 68.65 ms  | 11.18 ms    | 1516.83 ms          | 0.05×         | 0.01×      |

The naive kernel is uniformly 15–20× slower than stock PyTorch, and
60–120× slower than `mpsgraph_zc`. That is expected — single-thread-per-row
SDPA is compute-bound at quadratic work per thread and is not memory-efficient.

## What a real FA-2 implementation would need

For reference, a Metal FlashAttention-2 implementation would need
roughly:

1. **Tiled softmax** along `Lkv` in blocks of size `Bc` (e.g. 64). Each
   threadgroup handles a `(Br, Bc)` tile.
2. **Threadgroup shared memory** to stage K/V tiles so they are reused
   across all Q rows in the block.
3. **Simdgroup matrix ops** (`simdgroup_multiply_accumulate`) for the
   dot products — required to make use of Apple's matrix engine.
4. **Collaborative loads**: threads in a simdgroup share strided loads
   into shared memory.
5. **Parallel reduction** for the online softmax `(max, sumexp)` state
   across a simdgroup.
6. **Output tiling** over `Lq` with optional recompute on backward.

Each of these takes careful implementation and profiling. Even with a
textbook FA-2 implementation, beating Apple's
`scaledDotProductAttention` op (which has been tuned by Apple's MPS team
across multiple macOS releases) is uncertain.

## When to revisit

- If a future macOS removes or regresses Apple's fused SDPA op.
- If this package's workload shifts to short sequences (`Lq < 1024`)
  where `mpsgraph_zc`'s win is smaller and a custom kernel could
  specialize tighter.
- If a strong third-party Metal FA-2 implementation appears with a
  permissive license + comprehensive test coverage.

## Artifacts

- Probe kernel source: `src/mps_sdpa/backends/metal_proto.py`
- Tests pinning correctness + auto-pick exclusion: `tests/test_metal_proto.py`
