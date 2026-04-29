# mps-sdpa — Internal Architecture

This document maps the code so contributors can find where to make changes.
For the user-facing API, see [README.md](README.md). For what's tested and
what's deferred, see [COMPAT.md](COMPAT.md).

## Layout

```
mps_sdpa/
├── src/mps_sdpa/
│   ├── __init__.py              # public API re-exports
│   ├── api.py                   # sdpa_opt() — dispatch, GQA, mask coercion
│   ├── cli.py                   # `mps-sdpa` CLI subcommands
│   ├── backends/
│   │   ├── __init__.py          # backend registry (+ reason strings)
│   │   ├── stock.py             # thin wrapper over torch stock SDPA
│   │   ├── mpsgraph.py          # pyobjc + CPU-copy bridge (fallback)
│   │   ├── mpsgraph_zc.py       # C++ ext wrapper + autograd.Function
│   │   ├── metal_proto.py       # naive Metal kernel reference (not auto-selected)
│   │   └── _calibrate.py        # shape-threshold auto-calibration
│   ├── _cpp/
│   │   ├── __init__.py          # lazy JIT loader
│   │   └── mpsgraph_zc.mm       # Obj-C++ torch extension source
│   ├── harness/                 # benchmark + correctness + reference tools
│   ├── suites/                  # shape suites (correctness, realistic, general)
│   ├── training_check/          # synthetic trainer + loss compare + checkpoint interchange
│   └── utils/                   # env mgmt, thermal helpers
├── tests/                       # pytest suite (213 tests)
├── pyproject.toml               # packaging
├── docs/design/                 # design notes (rationale, benchmarks, kernel experiment)
└── (top-level docs: README.md, ARCHITECTURE.md, COMPAT.md, CONTRIBUTING.md, CHANGELOG.md)
```

## Dispatch flow

Every call to `sdpa_opt(q, k, v, ...)` goes through `api.py`:

```
sdpa_opt
  ├─ coerce additive float mask dtype to q.dtype (if needed)
  ├─ if Hq != Hkv (GQA): repeat_interleave K/V → call stock backend
  ├─ _pick_auto(q)  →  "mpsgraph_zc" | "mpsgraph" | "stock" (device-aware)
  └─ backends[name](q, k, v, ...)
       ├─ (mpsgraph_zc) _ZCSDPAFunction.apply or direct ext.sdpa_forward
       │    └─ fallbacks to mpsgraph backend on: dropout, OOM, non-MPS
       ├─ (mpsgraph)    _MpsGraphSDPAFunction.apply or _mpsgraph_forward_inner
       │    └─ fallbacks to stock on: short seq, unsupported dtype, mask shape,
       │                              dropout outside window, OOM
       └─ (stock)       F.scaled_dot_product_attention
```

`api._pick_auto()` preference order (first available wins):
```python
["mpsgraph_zc", "mpsgraph", "metal_op", "metal_proto", "stock"]
```

## Backend registry

`backends/__init__.py` is a tiny global registry:

```python
register_backend(name, fn, available=bool, reason=None)
get_backend(name)     # raises if unavailable (message includes reason)
available_backends()  # list of names
backend_reason(name)  # why a backend is unavailable
```

Each backend module calls `register_backend` at module import time. Registered
backends with `available=False` are known but won't be auto-picked.

## `mpsgraph_zc` — the default backend

Two layers:

1. **C++ / Objective-C++ extension** (`_cpp/mpsgraph_zc.mm`, ~350 LOC).
2. **Python wrapper** (`backends/mpsgraph_zc.py`, autograd.Function + dispatch logic).

### C++ layer

The extension compiles on first import via `torch.utils.cpp_extension.load()`.
Source is `mpsgraph_zc.mm` (Obj-C++ needed to bridge MPSGraph objects).

Key functions:
- `getMTLBufferStorage(tensor)` — ATen's `static inline` function that
  reinterprets `tensor.storage().data()` as an `id<MTLBuffer>`. This is the
  zero-copy trick. Available only on MPS-backed tensors.
- `runMPSGraph(stream, graph, feeds, results)` — ATen's own helper to dispatch
  an `MPSGraph` on a command queue.
- `sdpa_forward(q, k, v, mask=None, dropout_mask=None) -> out`
- `sdpa_backward(q, k, v, grad_out, mask=None, dropout_mask=None) -> (dQ, dK, dV)`

Graph caches:
- `g_graph_cache` — forward graphs keyed by `(dtype, B, H, Lq, Lkv, D, mask_kind,
  mask_B, mask_H, dropout)`. Thread-safe via `std::mutex`.
- `g_bwd_graph_cache` — same key, separate cache for backward graphs.

Two distinct graph constructions:
1. **Fused** (`dropout=false`): calls Apple's
   `scaledDotProductAttentionWithQueryTensor:keyTensor:valueTensor:[maskTensor:]scale:name:`.
2. **Unfused with dropout** (`dropout=true`): builds scores → softmax →
   `* dropout_mask` → matmul V explicitly, since Apple's fused op has no
   dropout variant.

The backward graph is always unfused (Apple's op doesn't expose attention
weights pre/post softmax). Uses the standard manual softmax backward:
`d_scores = attn * (d_attn - sum(attn * d_attn, axis=-1, keepdim=True))`.

### Python wrapper

`backends/mpsgraph_zc.py`:

- `_ZCSDPAFunction(torch.autograd.Function)` — wraps ext forward/backward.
  `@once_differentiable` on backward (MPSGraph is opaque to torch autograd).
- `mpsgraph_zc_sdpa(q, k, v, ...)` — dispatch entry:
  - ext not built → pyobjc fallback
  - non-MPS / unsupported dtype → pyobjc fallback
  - short seq (below calibrated threshold / 4) → pyobjc fallback
  - mask shape not broadcast-compatible → pyobjc fallback
  - causal + explicit mask → pyobjc fallback
  - otherwise: build mask tensor + dropout mask → ext.sdpa_forward or
    `_ZCSDPAFunction.apply`

## `mpsgraph` — pyobjc-based bridge

The pyobjc-based fallback backend. Same graph structure as `mpsgraph_zc`,
but uses pyobjc's Python→Objective-C bridge and routes tensor data via CPU
memcpy (which makes it slower). Stays in the codebase because:

1. Zero-build-dependency path — works if the C++ ext can't compile.
2. Has its own battle-tested dispatch logic that `mpsgraph_zc` delegates
   to for cases zc doesn't handle cleanly (e.g., dropout outside the
   fused-graph window).

## Shape-threshold auto-calibration

`backends/_calibrate.py` runs at first import (cached to
`~/.cache/mps_sdpa/thresholds.json` keyed by chip/os/torch). Measures the
crossover between stock and mpsgraph paths per dtype via 3 micro-benchmarks
(256², 1024², 2048² bf16/fp16/fp32) and picks the smallest shape where
mpsgraph wins by ≥ 5%.

Env-var controls:
- `MPS_SDPA_SKIP_CALIBRATION=1` — use M4-tuned defaults, no bench.
  Used by `tests/conftest.py` to keep test startup fast.
- `MPS_SDPA_FORCE_CALIBRATE=1` — ignore cache, recalibrate.

`mpsgraph_zc` uses `fused_min // 4` as its threshold — zero-copy has
significantly lower per-call overhead so the crossover shifts toward shorter
sequences.

## Fallback logging

All fallback paths go through `_log_fallback(reason)` with unified format:

```
[mps_sdpa.mpsgraph] falling back to stock: <reason>
```

Silent by default (DEBUG level). Opt in:
- `MPS_SDPA_LOG_FALLBACKS=1` → INFO level
- `MPS_SDPA_LOG_FALLBACKS=warn` → WARNING level

Reasons are specific (e.g., `"short-seq (2097152 < 4194304 bytes)"`) so users
can diagnose why a call fell back.

## Tests

`tests/` is organized by concern:

| File | Covers |
|---|---|
| `test_api.py` | Public entry-point behavior, GQA fallback |
| `test_dispatch.py` | Backend registry, availability probe |
| `test_calibration.py` | Threshold auto-calibration |
| `test_thread_safety.py` | Graph-cache race protection |
| `test_extended_shapes.py` | Extended D/H/B/non-POT matrix |
| `test_per_head_masks.py` | [B_m, H_m, Lq, Lkv] mask variants |
| `test_mask_dtype_coercion.py` | Cross-dtype mask acceptance |
| `test_partial_requires_grad.py` | 7 subsets of requires_grad |
| `test_amp_autocast.py` | torch.amp.autocast compat |
| `test_checkpoint_compat.py` | torch.utils.checkpoint re-entry |
| `test_second_order_grads.py` | `@once_differentiable` guard |
| `test_generic_transformer.py` | End-to-end smoke on a vanilla transformer block |
| `test_retain_graph.py` | Grad accumulation, retain_graph=True |
| `test_long_horizon.py` | 1000-step convergence |
| `test_numerical_extremes.py` | Extreme magnitudes, fp16 long-horizon |
| `test_mpsgraph_zc.py` | zero-copy backend wiring |
| `test_dropout_zc.py` | zero-copy dropout paths |
| `test_metal_proto.py` | Naive Metal kernel correctness + auto-pick exclusion |
| `test_edge_cases.py` | Discovery-style battle tests |
| `test_cli.py` | CLI subcommands |
| `test_fallback_logging.py` | Unified fallback log format |

`conftest.py` sets `MPS_SDPA_SKIP_CALIBRATION=1` to keep startup fast and
deterministic — individual tests that need real calibration override.

## Extending

**Adding a new backend:**
1. Create `backends/<name>.py`.
2. Implement `fn(q, k, v, *, attn_mask, dropout_p, is_causal, scale)`.
3. Call `register_backend("<name>", fn, available=<bool>, reason=<str or None>)`.
4. Import in `backends/__init__.py`.
5. Optionally add to `_pick_auto()` preference list in `api.py`.

**Modifying the C++ extension:**
1. Edit `_cpp/mpsgraph_zc.mm`.
2. Delete `~/.cache/torch_extensions/py*/mps_sdpa_zc_ext` to force rebuild.
3. Next `from mps_sdpa import sdpa_opt` recompiles (~6s).

**Adding a shape to auto-calibration:**
1. Edit `_calibrate._calibrate_dtype()` — add the probe shape.
2. Bump `_CACHE_SCHEMA_VERSION` so old caches are invalidated.

## Reproducibility checks

Every PR should:
1. `pytest tests/ -q` — full suite, 213 tests.
2. `mps-sdpa self-test --device mps` — quick correctness + speedup check.
3. For benchmark-affecting changes: `mps-sdpa benchmark --backend mpsgraph_zc
   --baseline stock --suite realistic` and attach the geomean delta.

CI workflows:
- `.github/workflows/tests.yml` — runs `pytest tests/ -q` on `macos-latest`
  (Apple silicon, M1 / current macOS) on every push to main, every PR, and
  on manual dispatch. The C++ extension JIT-compiles in CI; MPS-gated tests
  actually exercise the backend, not skip.
- `.github/workflows/pypi.yml` — fires only on tag push (`v*`) or manual
  dispatch. Calls `tests.yml` as a reusable workflow, then builds wheel +
  sdist, runs `twine check`, and publishes to PyPI via OIDC. Tests gate
  publish: a failure in `tests.yml` blocks the build and publish jobs.

Note: CI runs on M1 / macOS 15. Maintainer pre-tag testing on M-series /
macOS 26 remains the source of truth for the documented perf numbers; CI
is necessary but not sufficient for that claim.
