"""Tolerance bands per dtype."""
from __future__ import annotations

_FWD = {
    "fp32": (5e-6, 5e-5),
    "fp16": (5e-3, 5e-2),
    "bf16": (5e-3, 5e-2),
}


def forward_tol(dtype: str) -> tuple[float, float]:
    if dtype not in _FWD:
        raise KeyError(f"unknown dtype {dtype!r}")
    return _FWD[dtype]


def backward_tol(dtype: str) -> tuple[float, float]:
    atol, rtol = forward_tol(dtype)
    return 2 * atol, 2 * rtol
