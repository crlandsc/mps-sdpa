# Roadmap

> **Status: aspirational.** The items below are thoughts about where this
> project might go. Plans shift, priorities change, and any item here may
> never ship. External users should not infer commitments from this file —
> the only guarantee is what's in the current release on PyPI.

This document complements [CHANGELOG.md](../CHANGELOG.md), which records
what has shipped. The roadmap below lists what might ship next, grouped by
how concrete the plan is. Items closer to the top are more likely to land;
items further down are conditional or speculative.

## v0.2.0 — Coverage & polish

Near-term work that extends what v0.1.0 already does, without changing the
public API:

- **M3 hardware validation.** Run the full test suite, correctness suite,
  and benchmark on M3 silicon. Capture per-dtype calibrated thresholds and
  geomean speedups; update [COMPAT.md](../COMPAT.md) with verified numbers.
- **CI test gate on Apple silicon.** GitHub Actions `test` job on
  `macos-latest` that actually exercises the MPS backends, so a regression
  in the extension build, autograd, or correctness blocks publish.
- **`torch.compile` feasibility probe.** Timeboxed investigation of how the
  C++ extension behaves under `torch.compile` (eager mode is currently the
  only verified path). Goal is a clear yes/no on whether full Inductor
  support is reachable without major rework. If cheap, ship it here;
  otherwise defer.
- **Documentation hygiene.** Keep README / ARCHITECTURE / COMPAT consistent
  as the surface area grows.

## v0.3.0 — Capability gaps

Items currently flagged in the README's "What doesn't work (yet)" section
with a known shape:

- **Wider dropout fast-window.** The unfused dropout graph only beats stock
  in a narrow attention-matrix byte range today. Profile and widen.
- **GQA acceleration.** Today `Hq != Hkv` falls back to stock with
  `repeat_interleave`. Either wait for an MPSGraph-side GQA op or build a
  separate fused path.
- **`torch.compile` integration**, if the v0.2.0 probe shows it's tractable.

## Research / conditional

Larger investments that depend on external preconditions:

- **Second-order gradients (`create_graph=True`).** Would require rewriting
  the manual MPSGraph backward as a fully differentiable graph rather than
  a `@once_differentiable` block. Cost is high, demand is unclear.
- **Custom Metal FlashAttention-2 kernel.** Explicitly declined for v0.1.0
  (see [docs/design/custom-kernel-experiment.md](design/custom-kernel-experiment.md)
  for the rationale). Only worth revisiting if Apple regresses the fused
  op, the workload shifts to short sequences, or a permissively-licensed
  third-party FA-2 implementation appears.

## Out of scope

Things that are not on the roadmap and unlikely to ever be:

- **macOS 14 support.** The `MPSGraph.scaledDotProductAttention` op doesn't
  exist there. The backend already registers as unavailable with a clear
  reason and `sdpa_opt` falls back to stock cleanly.
- **fp64.** PyTorch MPS does not support fp64; not a limitation we can fix.
- **Hardware outside the maintainer's lab.** Bug reports from M1/M2 / older
  macOS are welcome and the code is chip-agnostic, but no commitment to
  test those configs directly.
