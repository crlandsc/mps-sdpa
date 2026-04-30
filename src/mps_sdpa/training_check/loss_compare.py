"""Compare stock vs opt loss curves for equivalence."""
from __future__ import annotations

import math
import statistics
from typing import Sequence


def compare(a: Sequence[float], b: Sequence[float], *, per_step_tol: float) -> dict:
    assert len(a) == len(b), f"length mismatch {len(a)} vs {len(b)}"
    diffs = [abs(x - y) for x, y in zip(a, b)]
    within = sum(1 for d in diffs if d <= per_step_tol) / len(diffs)
    mean_a = statistics.mean(a)
    mean_b = statistics.mean(b)
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    den_a = math.sqrt(sum((x - mean_a) ** 2 for x in a))
    den_b = math.sqrt(sum((y - mean_b) ** 2 for y in b))
    pearson = num / (den_a * den_b) if den_a > 0 and den_b > 0 else 1.0
    return {
        "n": len(a),
        "max_abs_diff": max(diffs) if diffs else 0.0,
        "mean_abs_diff": sum(diffs) / len(diffs) if diffs else 0.0,
        "step_diffs_within_tol_frac": within,
        "pearson": pearson,
        "final_a": a[-1], "final_b": b[-1],
    }
