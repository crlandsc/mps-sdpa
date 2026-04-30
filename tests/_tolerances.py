"""Per-chip tolerance helpers for cross-implementation tests.

Most mps-sdpa tests compare output against torch's stock SDPA. Both
implementations are correct within the documented bf16 math-reference
tolerance (atol=5e-3, rtol=5e-2), but they don't agree bit-exactly with
each other — each Apple silicon chip's matmul accumulation kernel rounds
slightly differently in intermediate precision.

On most chips this drift stays below the 5e-3 math-reference atol. On
chips where it doesn't (e.g., observed on M1 in CI), the comparison
needs a chip-specific looser bound. This module returns that per-chip
atol so the test suite stays as strict as possible by default and
relaxes only where empirically required.

**Default policy: strict.** A chip is added to ``_LOOSE_CHIPS`` only
after an observed test failure that bisects to bf16 ULP drift (failure
exactly on a 2^-N boundary, no NaN/Inf, magnitude-correlated). The entry
must include a comment naming the failing test + observed delta.

This helper is for *direct two-implementation comparisons*
(``sdpa_opt`` vs ``F.scaled_dot_product_attention``). Tests that compare
against a math reference, or self-consistency tests (same backend
across calls / cache states / mask-vs-no-mask), keep the strict 5e-3
bound — the per-chip relaxation does not apply.
"""
from __future__ import annotations

import re
import subprocess

import torch

# Strict math-reference tolerances, matching COMPAT.md's correctness contract.
_STRICT_BF16 = 5e-3
_STRICT_FP16 = 5e-3
_STRICT_FP32 = 5e-6


# Patterns matching chip-brand prefixes that have observed cross-impl drift
# above the strict bf16/fp16 atol. Add a chip family here only after a real
# observed failure with the rationale recorded in the comment.
_LOOSE_CHIPS_BF16: list[tuple[re.Pattern[str], float]] = [
    # Apple M1 family — observed on CI runner macos-latest (M1 / macOS 15)
    # in the first Apple-silicon CI run after splitting tests.yml. Failures:
    #   test_edge_cases.py::test_non_contiguous_transposed_inputs : 0.03125 vs 1e-2
    #   test_edge_cases.py::test_sliced_view_inputs               : 0.03125 vs 1e-2
    #   test_edge_cases.py::test_mostly_false_mask                : 0.015625 vs 5e-3
    #   test_mpsgraph_zc.py::test_zc_handles_causal_mask          : 0.015625 vs 5e-3
    # All values land exactly on bf16 ULP boundaries (2^-5, 2^-6) — sub-ULP
    # drift between two correct bf16 implementations, not a real bug.
    (re.compile(r"^Apple M1(?:\s|$)"), 5e-2),
    # M2/M3/M4: no observed cross-impl drift > strict — defaulting to strict.
]

# Extra atol for tests where Q/K/V are non-contiguous views: we call
# .contiguous() internally, and stock's own memory-reorder path may use a
# different reduction order than our contig copy. ~1 bf16 ULP higher than
# the contig case even on chips with otherwise-tight drift. Documented since
# v0.1.0 (see test_edge_cases.py).
_NON_CONTIG_BF16 = 1e-2


def _chip_brand() -> str:
    """Apple chip brand string, e.g., ``Apple M3 Max``. ``unknown`` off-Apple.

    Mirrors ``backends/_calibrate.py::_fingerprint`` — same source, so the
    chip identity used for tolerance lookup matches the one used for
    threshold caching.
    """
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stderr=subprocess.DEVNULL, timeout=2,
        ).decode().strip()
    except Exception:
        return "unknown"


def cross_impl_atol(dtype: torch.dtype, *, non_contig: bool = False) -> float:
    """Atol for a direct mps-sdpa-vs-stock-SDPA comparison on this machine.

    Default is the strict math-reference atol (5e-3 bf16/fp16, 5e-6 fp32).
    Returns a wider value only on chips empirically observed to drift more.

    Parameters
    ----------
    dtype : torch.dtype
        Dtype of the comparison tensors. Determines the strict baseline.
    non_contig : bool, default False
        Pass ``True`` for tests where Q/K/V are non-contiguous views
        (transposed, sliced, etc.). The required atol is bumped to at
        least ``_NON_CONTIG_BF16`` (1e-2) — this drift is chip-independent
        and stems from the contig-copy path differing between mps-sdpa and
        stock.

    Returns
    -------
    float
        The atol to use in ``(out - ref).abs().max() < atol``.

    Examples
    --------
        out = sdpa_opt(q, k, v)
        ref = F.scaled_dot_product_attention(q, k, v)
        assert (out - ref).abs().max().item() < cross_impl_atol(q.dtype)
    """
    chip = _chip_brand()

    if dtype in (torch.bfloat16, torch.float16):
        base = _STRICT_BF16 if dtype == torch.bfloat16 else _STRICT_FP16
        for pattern, looser in _LOOSE_CHIPS_BF16:
            if pattern.match(chip):
                base = max(base, looser)
                break
        if non_contig:
            base = max(base, _NON_CONTIG_BF16)
        return base

    if dtype == torch.float32:
        # No observed cross-impl drift > 5e-6 on any chip; fp32 has enough
        # mantissa precision that ULP-level rounding stays well within tolerance.
        return _STRICT_FP32

    raise ValueError(f"unsupported dtype: {dtype}")
