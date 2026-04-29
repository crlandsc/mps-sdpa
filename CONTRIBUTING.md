# Contributing to mps-sdpa

Bug reports and PRs welcome. The backend has been tested on M4 / macOS 26.4.1;
we'd especially love issues that uncover bugs on M1/M2/M3 or on macOS 15.x
since we haven't been able to exercise those configs ourselves.

## Reporting a bug

Please include:
- macOS version (`sw_vers`)
- chip (`sysctl -n machdep.cpu.brand_string`)
- Python version + `pip show torch` output
- output of `mps-sdpa self-test --device mps`
- a minimal reproducer — the most useful shape is `(B, H, Lq, Lkv, D, dtype,
  mask kind)` plus whether autograd is involved

If you hit a wrong-answer bug, please include the max diff vs stock
(`(out - F.scaled_dot_product_attention(q, k, v)).abs().max()`) — if it's
outside our documented tolerance bands (5e-3 bf16/fp16, 5e-6 fp32) that's a
correctness regression and top priority.

## Prerequisites

- macOS 15.0+ (Sequoia or newer) on Apple silicon. The C++ extension uses an
  MPSGraph method that doesn't exist on macOS 14; the backend will register as
  unavailable there and fall back to stock.
- Python ≥ 3.10.
- PyTorch ≥ 2.11 (stable) or ≥ 2.13 (nightly recommended).
- Xcode Command Line Tools (`xcode-select --install`) — required to JIT-compile
  the Obj-C++ extension.

## Development setup

```bash
git clone https://github.com/crlandsc/mps-sdpa.git
cd mps-sdpa
python -m venv .venv && source .venv/bin/activate
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
pip install -e ".[dev]"
pytest tests/ -q                 # 235 tests; takes ~50s on M4
mps-sdpa self-test --device mps  # quick correctness + speedup check
```

## Running the full test suite

```bash
pytest tests/ -q
```

Full suite takes ~50s on M4. Individual test files are self-contained; pick
by name to iterate on a specific concern:

```bash
pytest tests/test_edge_cases.py -v
pytest tests/test_dropout_zc.py -v
```

Tests skip gracefully if MPS is unavailable (e.g., CI on Linux).

## Benchmarking a change

Before/after numbers on the realistic shape suite:

```bash
# Baseline (before your change)
mps-sdpa benchmark --backend mpsgraph_zc --baseline stock --device mps \
    --suite realistic --n-pairs 3 --out bench-before.csv

# Apply your change, then
mps-sdpa benchmark --backend mpsgraph_zc --baseline stock --device mps \
    --suite realistic --n-pairs 3 --out bench-after.csv
```

The CLI prints a weighted geomean; include both in the PR description.

## Adding tests

If your change touches a backend, add a regression test in the matching file
(e.g., zero-copy changes → `test_mpsgraph_zc.py` or `test_dropout_zc.py`). For
new shape/dtype/mask combinations, prefer extending the existing
parametrizations in `test_extended_shapes.py` or `test_edge_cases.py` over
adding a new file.

Discovery-style tests (probes aimed at finding new bugs) go in
`test_edge_cases.py`.

## Modifying the C++ extension

After editing `src/mps_sdpa/_cpp/mpsgraph_zc.mm`:

```bash
rm -rf ~/.cache/torch_extensions/py*/mps_sdpa_zc_ext
python -c "from mps_sdpa._cpp import get_ext; get_ext()"
```

That triggers a clean rebuild. The first invocation after an edit compiles
(~6s); cached afterwards.

Inside the `.mm` file, remember:
- ARC is off (`-fno-objc-arc`). Manage Obj-C lifetimes with `[retain]`/`[release]`.
- Wrap Obj-C autoreleased objects (`MPSGraphTensorData`, `NSDictionary`,
  `NSArray`) inside an explicit `@autoreleasepool { ... }` block — without one,
  Python-driven training loops accumulate Obj-C objects that leak Metal
  resources into "other allocations" and trigger OOMs.
- `MPSDataType` constants live in `MPSCoreTypes.h` under the macOS SDK.
- `getMTLBufferStorage(tensor)` is `static inline` in ATen's
  `OperationUtils.h` — it's the zero-copy trick.

## Adding a new backend

1. Create `src/mps_sdpa/backends/my_backend.py` with a function matching:
   ```python
   def my_backend_sdpa(q, k, v, *, attn_mask, dropout_p, is_causal, scale):
       ...
   ```
2. Register: `register_backend("my_backend", my_backend_sdpa, available=...)`.
3. Import in `backends/__init__.py`.
4. Optionally wire into `api._pick_auto()` preference list.
5. Add tests in `tests/test_<name>.py`.

See `backends/mpsgraph_zc.py` for a full example.

## Style

- Code: ruff-default rules, ~100-char lines, type hints on public API.
- Comments: explain *why*, not *what*. The `.mm` file documents the ATen
  internal API surface; keep those comments accurate.
- Commit messages: imperative mood, concise body. Include benchmark deltas
  for performance-affecting changes.

## Releases

Releases use **PyPI Trusted Publishers** (OIDC) — no API tokens. The
`.github/workflows/pypi.yml` workflow handles test → build → upload, gated
by tag push only (regular pushes to `main` do **not** trigger a release).

Workflow:
1. Run the full local pre-flight on M4 / macOS 26:
   ```bash
   pytest tests/ -q
   mps-sdpa self-test --device mps
   ```
2. Bump `version` in `pyproject.toml` and add a new `## [x.y.z]` entry to
   `CHANGELOG.md`. Commit + push to `main`.
3. Tag and push — this fires the publish workflow:
   ```bash
   git tag v<x.y.z>
   git push origin v<x.y.z>
   ```
   Or cut a GitHub release (which creates the tag too):
   ```bash
   gh release create v<x.y.z> --title "v<x.y.z>" --notes-file CHANGELOG.md --target main
   ```

The workflow runs the test suite first; if anything fails, the publish step
does not run and nothing is uploaded to PyPI.

Local sanity build (no upload):
```bash
python -m build              # produces dist/mps_sdpa-<x.y.z>-{tar.gz,whl}
python -m twine check dist/* # validates PyPI metadata
```

## License

By contributing, you agree your changes are MIT-licensed (see LICENSE).
