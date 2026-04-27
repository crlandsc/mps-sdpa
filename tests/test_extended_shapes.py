"""Extended shape coverage: D/H/B sweeps + non-power-of-2 Lq/Lkv.

Surface any hard failures in realistic
shape/dtype/mask combinations so we can fix them or explicitly fall back.
"""
from __future__ import annotations

import pytest
import torch

from mps_sdpa.backends import mpsgraph as _mg
from mps_sdpa.harness.correctness import check_case
from mps_sdpa.suites.correctness_shapes import Case, iter_extended_cases


def _can_run() -> bool:
    try:
        return _mg._AVAILABLE and torch.backends.mps.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _can_run(), reason="requires MPSGraph backend + MPS device")
@pytest.mark.parametrize("case", list(iter_extended_cases()), ids=lambda c: c.case_id)
def test_extended_shape_correctness_mpsgraph(case: Case):
    """Every extended-matrix case must pass against the math reference."""
    r = check_case(backend_name="mpsgraph", case=case, device="mps")
    if r.get("environmental_skip"):
        pytest.skip(f"OOM or env-skip: {r.get('error')}")
    assert r["passed"], (
        f"{case.case_id} failed ({r.get('failure_class')}): err={r.get('error')} "
        f"max_abs={r.get('max_abs_err')} max_rel={r.get('max_rel_err')}"
    )


@pytest.mark.skipif(not _can_run(), reason="requires MPSGraph backend + MPS device")
def test_extended_matrix_no_hard_failures():
    """Aggregate: no hard failures across the whole matrix."""
    hard = []
    soft = []
    for c in iter_extended_cases():
        r = check_case(backend_name="mpsgraph", case=c, device="mps")
        if r.get("environmental_skip"):
            continue
        if r.get("failure_class") == "hard":
            hard.append((c.case_id, r.get("error")))
        elif r.get("failure_class") == "soft":
            soft.append((c.case_id, r.get("max_abs_err"), r.get("max_rel_err")))
    assert not hard, f"hard failures: {hard}"
    assert not soft, f"soft failures: {soft}"
