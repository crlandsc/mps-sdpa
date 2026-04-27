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
| **macOS** | 15.0 (Sequoia) | **26.4.1 only** | 14.x is documented-unsupported: backend registers `available=False` with a clear reason string (probe via `hasattr` at import). We have not run on a real macOS 14 machine — the unavailable-reason path is validated via monkeypatch only. macOS 15 should work but is not empirically verified. |
| **Apple silicon** | (no minimum enforced — chip-agnostic code) | **M4** | **Planned:** M3. **Not planned:** M1/M2 (no hardware access; should theoretically work — logic is not chip-specific and thresholds auto-calibrate per-machine on first import). |
| **PyTorch** | 2.11.0 (stable) | 2.11.0 + 2.13.0.dev20260420 | Both exhaustively validated. No API drift between versions. |
| **Python** | 3.10 | 3.11.14 | No 3.10-specific features used. |
| **pyobjc-core** | 10.0 | 12.1 | Required frameworks: Metal, MetalPerformanceShaders, MetalPerformanceShadersGraph. |

### Hardware / OS coverage commitments

- **Tested:** M4 chip, macOS 26.4.1. This is the reference configuration; any claim in this doc is backed by a test here unless noted.
- **Planned:** M3 testing is on the roadmap. Will update this table with results.
- **Not planned:** M1 and M2 chips — no hardware access. The code is chip-agnostic and auto-calibrates thresholds per-machine; the headline behavior should work, but we make no empirical guarantees. Report issues if encountered and we'll add a compat shim.
- **macOS 15:** should work (all API surfaces we use were introduced in 15.0), not yet empirically tested.
- **macOS 14:** documented-unsupported. The `_check_mpsgraph_sdpa_available()` probe returns `(False, reason)` and the backend is registered unavailable. No runtime crash, just clear fallback.

## Per-dtype threshold (auto-calibrated, indicative M4 + torch 2.13 nightly values)

| dtype | fused_min_bytes | crossover shape |
|---|---|---|
| bf16 | 2 MB | ~1024² |
| fp16 | 2 MB | ~1024² |
| fp32 | 16 MB | ~2048² |

Thresholds are calibrated once per `(chip, os, torch)` fingerprint and cached to `~/.cache/mps_sdpa/thresholds.json`. Force a re-calibration with `MPS_SDPA_FORCE_CALIBRATE=1`. Skip entirely (use conservative defaults) with `MPS_SDPA_SKIP_CALIBRATION=1`.

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
