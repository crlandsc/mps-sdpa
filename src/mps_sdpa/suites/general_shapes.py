"""General secondary suite (Tier-2). Uniform weight, includes causal."""
from __future__ import annotations

from itertools import product
from typing import Iterator

from .realistic_shapes import WeightedCase


def iter_cases() -> Iterator[WeightedCase]:
    for B, H, L, D, dtype, mask in product(
        [1, 2, 4], [8, 16], [128, 512, 1024, 2048, 4096], [64, 128],
        ["fp16", "bf16", "fp32"], ["none", "bool_b1lk", "causal"],
    ):
        yield WeightedCase(
            case_id=f"g_B{B}H{H}L{L}D{D}_{dtype}_{mask}",
            origin="general",
            B=B, H=H, Lq=L, Lkv=L, D=D,
            dtype=dtype, mask=mask, weight=1,
        )
