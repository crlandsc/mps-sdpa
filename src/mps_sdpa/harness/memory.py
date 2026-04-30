"""MPS memory measurement helpers."""
from __future__ import annotations

import gc
from typing import Callable

import torch


def mps_snapshot() -> dict:
    if not torch.backends.mps.is_available():
        return {"current": 0, "driver": 0, "recommended_max": 0}
    return {
        "current": int(torch.mps.current_allocated_memory()),
        "driver": int(torch.mps.driver_allocated_memory()),
        "recommended_max": int(torch.mps.recommended_max_memory()),
    }


def measure_region(fn: Callable[[], torch.Tensor], *, device: str = "mps") -> dict:
    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()
    before = mps_snapshot()
    fn()
    after = mps_snapshot()
    return {
        "before": before,
        "after": after,
        "delta_current": after["current"] - before["current"],
        "delta_driver": after["driver"] - before["driver"],
    }
