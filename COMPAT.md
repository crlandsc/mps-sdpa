# mps-sdpa — Compatibility matrix

Tested configurations for the backends shipped in this package.

## Backends (pick one explicitly, or use `backend="auto"`)

| Name | Impl | Speedup vs stock (M4, bf16) | Autograd | Dropout | Notes |
|---|---|---|---|---|---|
| `mpsgraph_zc` | Obj-C++ torch ext, getMTLBufferStorage zero-copy | **4.97–7.17×** inference, **1.96–2.46×** training | ✅ | fallback | **Default.** Requires xcode-select'ed CLT + compile ~6s on first import. |
| `mpsgraph` | pyobjc + CPU-copy bridge | 2.0–5.9× inference, 1.2–2.2× training | ✅ | ✅ | Fallback when zc unavailable. Also handles dropout. |
| `stock` | `torch.nn.functional.scaled_dot_product_attention` | 1× | ✅ | ✅ | Always available. Final fallback. |
| `metal_proto` | torch.mps.compile_shader() naive kernel | 0.05× (regression) | ✗ | ✗ | Reference / experimentation. Correct but ~60× slower than zc. Not auto-selected. |

## Supported

| Category | Minimum | Empirically tested | Notes |
|---|---|---|---|
| **macOS** | 15.0 (Sequoia) | **26.x** | Maintainer only tests macOS 26+. macOS 15 should work (all API surfaces we use were introduced in 15.0) but is not on the test path. macOS 14 is unsupported: the `_check_mpsgraph_sdpa_available()` probe returns `(False, reason)` and the backend registers unavailable; no runtime crash, just clear fallback. |
| **Apple silicon** | (no minimum enforced — chip-agnostic code) | **M4 mini, M3 Max** | M1/M2 not on the maintainer test path; logic is not chip-specific and thresholds auto-calibrate per-machine on first import. Should theoretically work — community reports welcome. |
| **PyTorch** | 2.11.0 (stable) | 2.11.0 + 2.13.0.dev20260420 | Both exhaustively validated. No API drift between versions. |
| **Python** | 3.10 | 3.11.14 | No 3.10-specific features used. |
| **pyobjc-core** | 10.0 | 12.1 | Required frameworks: Metal, MetalPerformanceShaders, MetalPerformanceShadersGraph. |

### Hardware / OS coverage commitments

- **Maintainer-tested:** Apple M4 mini and M3 Max, macOS 26.4.1, torch 2.11 stable + 2.13 nightly. Any claim in this doc is backed by a test here unless noted.
- **Should work, not on the test path:** M1, M2; macOS 15.x. The code is chip-agnostic and auto-calibrates thresholds per-machine, and all MPSGraph methods we use are present from macOS 15.0. Bug reports from these configs are welcome, but the maintainer does not plan to test them directly.
- **Not supported:** macOS 14.x. The `MPSGraph.scaledDotProductAttention` op doesn't exist there; the backend registers as unavailable with a clear reason and `sdpa_opt` falls back to stock cleanly.

## Per-dtype threshold (auto-calibrated per machine)

The `(chip, os, torch)` fingerprint determines stock-vs-`mpsgraph_zc` crossover thresholds at first import; cached to `~/.cache/mps_sdpa/thresholds.json`. Reference values from the maintainer's machines:

| dtype | M4 mini | M3 Max | notes |
|---|---|---|---|
| bf16 | 2 MB (~1024²) | 8 MB (~2048²) | M3 Max's higher memory bandwidth keeps stock competitive at smaller shapes; crossover shifts to longer sequences |
| fp16 | 2 MB (~1024²) | 8 MB (~2048²) | same |
| fp32 | 16 MB (~2048²) | **effectively unbounded** | M3 Max stock fp32 wins universally at the calibrator's probe shapes; `mpsgraph_zc` is not auto-selected for fp32 on M3 Max |

Force re-calibration with `MPS_SDPA_FORCE_CALIBRATE=1`. Skip entirely (use conservative defaults) with `MPS_SDPA_SKIP_CALIBRATION=1`.

## Test tolerance policy

The package's correctness contract — `bf16/fp16 atol=5e-3, fp32 atol=5e-6` — is the standard for **math-reference comparisons** (`mps_sdpa.sdpa_opt(...)` vs a manually-computed reference at the same dtype). The full test suite validates these bounds.

A subset of tests additionally compare `mps_sdpa.sdpa_opt` against `torch.nn.functional.scaled_dot_product_attention` directly. Both implementations are correct within the math-reference contract, but they don't agree bit-exactly with each other — each Apple silicon chip's matmul accumulation kernel rounds slightly differently, so the implementation-vs-implementation drift can exceed `5e-3` even though both implementations are within spec.

For these tests, the suite uses [`tests/_tolerances.py::cross_impl_atol`](tests/_tolerances.py), which returns:

- **Strict (`5e-3` bf16/fp16)** by default — the math-reference contract, used on every chip family unless empirically observed to fail.
- **Loose (`5e-2` bf16/fp16)** for chips listed in `_LOOSE_CHIPS_BF16` after observed cross-impl drift > `5e-3`. Currently: **Apple M1 family** only.
- **`non_contig=True` knob** bumps to at least `1e-2` for tests where Q/K/V are non-contiguous views (independent of chip; the contig-copy path drift was documented since v0.1.0).

Adding a new chip to the loose list requires an observed test failure that bisects to bf16 ULP drift (failure on a `2^-N` boundary, no NaN/Inf, magnitude-correlated). The chip family entry must include a comment naming the failing test and the observed delta. This keeps the suite as strict as possible by default and only relaxes where empirically required.

## Shape / dtype / mask coverage

| Feature | Status | Notes |
|---|---|---|
| Head dim D ∈ {32, 64, 96, 128, 192, 256} | ✓ | All verified against math reference |
| Heads H ∈ {1, 2, 4, 8, 16, 32} | ✓ | |
| Batch B ∈ {1, 2, 4, 8, 32} | ✓ | |
| Non-power-of-2 Lq/Lkv | ✓ | 777, 1345, 3141 tested |
| Causal masking | ✓ | Via built causal mask tensor |
| Bool mask [1,1,Lq,Lkv] | ✓ | |
| Bool mask [B,H,Lq,Lkv] (per-head) | ✓ | |
| Float additive mask, any broadcast shape | ✓ | Auto-coerces dtype to Q's dtype |
| GQA (Hq ≠ Hkv) | fallback | Expands K/V via `repeat_interleave`, routes to stock, one-time warning |
| Dropout | fallback/narrow | Unfused graph wins only in 16-64 MB attn-matrix window |
| Second-order grads (create_graph=True) | raises | `@once_differentiable` — clear error rather than silent wrong values |
| AMP autocast | ✓ | Verified bf16 and fp16 paths |
| torch.utils.checkpoint | ✓ | Graph cache survives re-entry |
| retain_graph / grad accumulation | ✓ | |
| Mid-run backend toggle | ✓ | Flag on each `Attend` module; no weight corruption |
| `torch.compile` (Inductor) | ✓ (v0.2.0+) | Via `torch.library.custom_op` + `register_autograd` + `register_fake`. `dynamic=False` tested; output matches eager within `cross_impl_atol(dtype)`. Six equivalence tests in `tests/test_torch_compile.py`. |

## Known limitations

- **macOS 14**: `MPSGraph.scaledDotProductAttention` op is absent. Backend registers as unavailable with reason `"MPSGraph missing method ... — macOS 15.0+ (Sequoia) required"`.
- **GQA is not accelerated** (falls back to stock with K/V expansion).
- **Second-order gradients are not supported** (raises clear error).
- **Dropout** only accelerates within a narrow shape window.
- **fp64** is not supported by PyTorch MPS itself (not our limitation).

## Graceful-fallback list

All fallbacks emit a unified log message format `[mps_sdpa.mpsgraph] falling back to stock: <reason>`. Silent by default; enable with `MPS_SDPA_LOG_FALLBACKS=1` (INFO) or `=warn` (WARNING).

- Non-MPS device
- Unsupported dtype (int, fp64, etc.)
- Short sequence below calibrated threshold
- `is_causal=True` combined with explicit `attn_mask`
- Mask shape not expressible as [B_m, H_m, Lq, Lkv] broadcast
- Dropout outside the [dropout_min, dropout_max] byte window
- MPS OOM at execution time (retries on stock)
- GQA (Hq ≠ Hkv) — via api.py, routes around mpsgraph entirely

## Envvars

| Name | Values | Effect |
|---|---|---|
| `MPS_SDPA_SKIP_CALIBRATION` | `1` | Use conservative M4-tuned defaults, no micro-bench. Used by test suite. |
| `MPS_SDPA_FORCE_CALIBRATE` | `1` | Ignore cache, re-run calibration and overwrite. |
| `MPS_SDPA_LOG_FALLBACKS` | `1` / `info` / `warn` | Emit unified fallback log messages at INFO or WARNING level. |
| `WANDB_MODE`, `WANDB_DISABLED`, `WANDB_SILENT` | forced on import | Hard rule per spec — no W&B ever. |

## Verification commands

```bash
# Quick self-test (~1s)
python -m mps_sdpa.cli self-test --device mps

# Full pytest suite
python -m pytest tests/ -q

# Realistic-shape correctness suite (10 weighted cases)
python -m mps_sdpa.cli correctness --backend mpsgraph_zc --device mps --suite realistic
```
