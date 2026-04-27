# Changelog

All notable changes to mps-sdpa.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-04-27

### Added

- Initial public release.
- **`mpsgraph_zc` zero-copy backend** — C++ / Objective-C++ torch extension
  wrapping Apple's native `MPSGraph.scaledDotProductAttention` op. Uses ATen's
  `getMTLBufferStorage` to pass torch tensors to MPSGraph with no CPU memcpy.
  Forward + backward via MPSGraph-native graphs. JIT-compiles on first import
  (~6 s, then cached under `~/.cache/torch_extensions/`).
- **`mpsgraph` pyobjc fallback backend** — same fused op via the pyobjc bridge.
  Used when the C++ extension can't build.
- **Drop-in `sdpa_opt(...)` API** — signature matches
  `torch.nn.functional.scaled_dot_product_attention`. Accepts the same keyword
  arguments (`attn_mask`, `dropout_p`, `is_causal`, `scale`).
- **Autograd integration** — custom `torch.autograd.Function` for both backends;
  `@once_differentiable` guard prevents silent wrong-value second-order grads.
- **Auto-calibrated thresholds** — per-`(chip, os, torch)` micro-benchmark on
  first import to find the stock-vs-opt crossover; cached at
  `~/.cache/mps_sdpa/thresholds.json`. Override with
  `MPS_SDPA_SKIP_CALIBRATION=1` (use defaults) or
  `MPS_SDPA_FORCE_CALIBRATE=1` (recalibrate).
- **Mask coverage** — bool + additive float masks; per-head `[B_m, H_m, Lq, Lkv]`
  broadcast (B_m ∈ {1, B}, H_m ∈ {1, H}); `is_causal=True`; auto-coerce mask
  dtype to query dtype.
- **Dropout (training-mode)** — unfused MPSGraph builds when `dropout_p > 0`;
  zero-copy bridge keeps it fast within a sensible attention-matrix size window.
  Stock fallback outside that window.
- **GQA fallback** — `Hq != Hkv` routes to stock with `repeat_interleave` and
  a one-time warning. Apple's fused op is MHA-only.
- **Thread-safe graph cache** with `threading.Lock`-protected miss → build → store.
- **Mid-run backend toggle** — flip `use_mps_sdpa_opt` on `Attend` modules
  during training; weights are not affected.
- **CLI** — `mps-sdpa self-test`, `correctness`, `benchmark`, `list-backends`,
  `run`. The `self-test` subcommand prints a clear `STATUS: ACTIVE [OK]`
  banner and benchmark numbers for the auto-picked backend.
- **Startup banner + dispatch counters** — `mps_sdpa.print_backend_banner()`
  for one-glance preflight; `mps_sdpa.api.get_call_stats()` /
  `print_call_stats()` for per-call dispatch breakdown.
- **213 unit tests** covering: shape/dtype/mask matrix; AMP autocast;
  `torch.utils.checkpoint`; `retain_graph` + grad accumulation; mid-run
  toggle; long-horizon (1000-step) convergence; numerical extremes
  (Q/K std=10 and std=1e-4, very large grad_out); fp16 AMP long-horizon;
  edge cases (degenerate shapes, non-contiguous inputs, strided views,
  cache thrash, repeat-call leak probe).
- **Documentation** — README, ARCHITECTURE, COMPAT, CONTRIBUTING, plus design
  notes under `docs/design/` (dispatch rationale, performance benchmarks,
  custom-kernel experiment).

### Performance (M4, macOS 26.4.1, torch 2.13 nightly, bf16)

- **Inference forward** — 5–7× over stock at L = 1024–8192 (B=1, H=8, D=64).
- **Training (forward + backward)** — 1.96–2.46× over stock at the same shapes.
- **Training with dropout** — 1.86–2.26× over stock at L = 1024–4096.
- **Driver memory** — ≫128× reduction at typical attention shapes (Apple's
  fused op doesn't materialize the [Lq, Lkv] attention matrix; the zero-copy
  bridge removes the CPU-side intermediate buffer too).

### Compatibility

- macOS 15.0+ (Sequoia or newer). On macOS 14, the backend registers as
  unavailable with a clear reason and `sdpa_opt` falls back to stock cleanly.
- Apple silicon M1–M4. Empirically tested on M4; auto-calibration handles
  per-chip threshold differences. M3 testing planned; M1/M2 untested but
  should work.
- PyTorch 2.11 (stable) and 2.13 (nightly). Both verified.

### Notes

- A custom Metal FlashAttention-2 kernel was probed and declined — a naive
  Metal implementation came in ~60× slower than `mpsgraph_zc`, and a
  full FA-2 kernel would need significant specialized work with uncertain
  prospects of beating Apple's already-tuned fused op. The probe is
  preserved as the `metal_proto` backend (registered, not auto-selected).
  See `docs/design/custom-kernel-experiment.md`.

[0.1.0]: https://github.com/crlandsc/mps-sdpa/releases/tag/v0.1.0
