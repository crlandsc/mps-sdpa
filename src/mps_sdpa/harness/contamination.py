"""Contamination detection for benchmark sessions (thermal + load + RAM rejection)."""
from __future__ import annotations

from typing import Iterable


def judge(
    samples: Iterable[dict], *,
    max_load1: float = 3.0, min_free_ram_gb: float = 2.0,
) -> dict:
    reasons: list[str] = []
    samples = list(samples)
    for s in samples:
        if s.get("thermal") in {"serious", "critical"}:
            reasons.append(f"thermal={s['thermal']}")
            break
        if s.get("load1", 0.0) > max_load1:
            reasons.append(f"load1={s['load1']} > {max_load1}")
            break
        if s.get("free_ram_gb", float("inf")) < min_free_ram_gb:
            reasons.append(f"free_ram_gb={s['free_ram_gb']} < {min_free_ram_gb}")
            break
    return {"accepted": len(reasons) == 0, "reasons": reasons}


def judge_distribution(stats: Iterable[dict], *, max_p90_over_p10: float = 1.25) -> dict:
    reasons: list[str] = []
    for s in stats:
        p10, p90 = s.get("p10"), s.get("p90")
        if p10 and p90 and (p90 / p10) > max_p90_over_p10:
            reasons.append(f"p90/p10={p90/p10:.2f} > {max_p90_over_p10}")
    return {"accepted": len(reasons) == 0, "reasons": reasons}
