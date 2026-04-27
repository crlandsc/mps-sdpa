# Performance benchmarks

All numbers measured on **Apple M4 / macOS 26.4.1 / PyTorch 2.13 nightly**, bfloat16,
unless noted. Benchmarks use the `realistic` shape suite shipped with the package
(see `src/mps_sdpa/suites/realistic_shapes.py`) — a weighted set of generic
transformer attention shapes (short, mid, long self-attention; cross-attention
with asymmetric `Lq`/`Lkv`; one bool-mask case; one fp32 case).

Reproduce with:

```bash
mps-sdpa benchmark --backend mpsgraph_zc --baseline stock \
    --device mps --suite realistic --n-pairs 3
```

## Inference (forward only)

`B = 1`, `H = 8`, `D = 64`, bf16, paired-sample geomean over 5 hot iterations
per case.

| L     | stock   | mpsgraph_zc | speedup |
|-------|---------|-------------|---------|
| 1024  | 5.78 ms | 0.90 ms     | **6.42×** |
| 2048  | 19.0 ms | 3.82 ms     | **4.97×** |
| 4096  | 76.1 ms | 11.79 ms    | **6.45×** |
| 8192  | 317 ms  | 44.3 ms     | **7.17×** |

**Weighted geomean across the realistic shape suite: 4.88× over stock.**

## Training (forward + backward, full step)

Same shapes, same precision, full backward pass.

| L     | stock   | mpsgraph_zc | speedup |
|-------|---------|-------------|---------|
| 1024  | 9.93 ms | 5.06 ms     | **1.96×** |
| 2048  | 38.6 ms | 17.1 ms     | **2.25×** |
| 4096  | 154 ms  | 64.8 ms     | **2.38×** |
| 8192  | 608 ms  | 247 ms      | **2.46×** |

## Training with dropout (`dropout_p = 0.1`)

The fused MPSGraph op has no dropout variant, so when `dropout_p > 0` the
backend builds an unfused graph (scores → softmax → ⊙ dropout_mask → matmul V).
Same shapes, bf16:

| L     | stock     | mpsgraph_zc | speedup |
|-------|-----------|-------------|---------|
| 1024  | 14.19 ms  | 7.63 ms     | **1.86×** |
| 2048  | 55.83 ms  | 28.49 ms    | **1.96×** |
| 4096  | 228 ms    | 101 ms      | **2.26×** |

## Driver memory per call

Apple's fused op doesn't materialize the `[Lq, Lkv]` attention matrix, and the
zero-copy bridge avoids any CPU-side intermediate buffer. Measured via
`torch.mps.driver_allocated_memory()` deltas before / after a single call.

| L     | stock      | mpsgraph_zc | reduction |
|-------|------------|-------------|-----------|
| 2048  | 1024 MB    | <1 MB       | **≫128×** |
| 4096  | 1024 MB    | <1 MB       | **≫64×**  |
| 8192  | 1024 MB    | 32 MB       | **32×**   |

## Correctness

Tolerances per dtype, all measured against a math reference forward pass:

| dtype  | atol | rtol | max observed forward error |
|--------|------|------|----------------------------|
| fp32   | 5e-6 | 5e-5 | 7.15e-7                    |
| bf16   | 5e-3 | 5e-2 | 9.77e-4                    |
| fp16   | 5e-3 | 5e-2 | 6.10e-5                    |

Backward gradients verified to within 2× the forward tolerance vs. stock
PyTorch's autograd path on the same dtype, across all shapes in the
correctness suite.

## Method notes

- All measurements are paired (same input fed to stock and to `mpsgraph_zc`)
  and reported as geomean ratios so per-call jitter cancels.
- Each case includes warmup iterations to flush graph compilation; the
  numbers above are hot-call latencies.
- Memory deltas exclude the one-time graph cache cost (~10–50 MB per
  unique cached shape).
- For full per-shape correctness + memory data, run
  `mps-sdpa benchmark --backend mpsgraph_zc --baseline stock --measure-cold --out bench.csv`.
