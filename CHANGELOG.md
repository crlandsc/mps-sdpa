# Changelog

All notable changes to mps-sdpa.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] — UNRELEASED

### Added

- **M3 Max validation** — full test suite, correctness suite, and benchmark
  on Apple M3 Max / macOS 26.4.1 / torch 2.11 stable. 235/235 tests pass
  under the strict cross-implementation tolerance bound; no chip-specific
  loosening required. M3 Max behaves like M4. Calibrated thresholds:
  bf16/fp16 at 8 MB (vs M4's 2 MB — bandwidth shifts crossover to longer
  sequences); fp32 effectively unbounded (stock fp32 wins universally
  on M3 Max, `mpsgraph_zc` not auto-selected for fp32).
- **Standalone CI test workflow** (`.github/workflows/tests.yml`) running
  on `macos-latest` Apple silicon. Triggers on push to main, pull requests,
  manual dispatch, and as a reusable `workflow_call` from `pypi.yml`.
  Publish workflow now delegates testing to this single source of truth
  and gates publish on it (a tag push runs the test job before build/upload).
- **Chip-aware cross-implementation tolerance helper** (`tests/_tolerances.py`).
  Defaults to the strict math-reference atol (5e-3 bf16/fp16, 5e-6 fp32);
  loosens to 5e-2 only for chips on the empirically-maintained
  `_LOOSE_CHIPS_BF16` list. Currently lists Apple M1 family — surfaced
  by the first Apple-silicon CI run at exactly bf16 ULP boundaries
  (0.03125 = 2⁻⁵, 0.015625 = 2⁻⁶) on 4 sparse-reduction or non-contig
  tests. M2/M3/M4 default to strict; M3 Max validated empirically.
- **22 smoke tests** for the tolerance helper (`tests/test_chip_tolerances.py`)
  monkeypatch the chip probe to pin the policy: default-strict, M1 family
  loose, fp32 always strict, regex anchored against a hypothetical future
  "Apple M11" being silently misclassified as M1.
- **`scripts/verify_machine.sh`** — comprehensive new-chip capture script.
  Writes env metadata, available backends, self-test, forced calibration
  thresholds, full pytest log, realistic correctness output,
  `--measure-cold` benchmark CSV, and a memory probe at L=2048/4096/8192
  to a timestamped output directory. `SKIP_BENCH=1` escape hatch for
  runs under GPU contention. Preflight check fails fast with setup
  instructions if `mps_sdpa` isn't importable in the active env.
- **`docs/ROADMAP.md`** — aspirational forward-looking notes (v0.2.0 /
  v0.3.0 / research / out-of-scope buckets). Explicitly not a commitment
  to external users.

### Changed

- Documentation cleanup tightening scope to "macOS 26+ M-series only" —
  README, COMPAT, ARCHITECTURE, CONTRIBUTING aligned with the test count
  (213 → 235 with the new chip-tolerance suite), the actual workflow
  filename (`pypi.yml`), and the maintainer's actual test commitments
  (M4 mini + M3 Max). Removed the TestPyPI step from the release recipe
  in CONTRIBUTING; the workflow only targets prod PyPI.
- Test tolerance policy documented in COMPAT.md "Test tolerance policy"
  section — strict-by-default for math-reference comparisons, per-chip
  relaxation only with empirical evidence and a comment naming the
  failing test.

### Performance (M3 Max, macOS 26.4.1, torch 2.11 stable, bf16)

- **Inference single-shape (B=1, H=8, L=2048, D=64)** — 4.39× over stock
  (M4 baseline at the same shape: 4.97×). M3 Max's higher memory
  bandwidth keeps stock SDPA more competitive in absolute terms, so the
  speedup ratio is smaller while absolute throughput is much higher.
- **Realistic-shape weighted geomean** — 3.6× over stock across the
  10-case suite (M4 baseline: ~4.88×).
- **Driver memory reduction** — 41× / 65× / 161× at L=2048 / 4096 / 8192
  respectively. M3 Max's 36 GB lets stock allocate the full quadratic
  attention matrix, so the reduction looks bigger in absolute numbers
  than M4's. Algorithmic advantage is identical — Apple's fused op
  doesn't materialize the `[Lq, Lkv]` matrix.

### Notes

- M3 Max passes all cross-implementation tolerance bounds at the strict
  5e-3 default — confirms the chip-aware tolerance helper's loose list
  remains M1-only. Adding chips requires empirical drift > 5e-3, with
  the failing test named in the comment.

[0.1.1]: https://github.com/crlandsc/mps-sdpa/releases/tag/v0.1.1

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
