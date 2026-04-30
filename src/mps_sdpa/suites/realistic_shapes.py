"""Realistic transformer attention shapes (Tier-1 ranking suite).

A weighted shape set representative of attention workloads in real
transformer architectures: short, mid, and long self-attention; cross-attention
with asymmetric Lq/Lkv; one bool-mask case; one fp32 case. Long-sequence
weights dominate — that's where the speedup matters most.

Used by the CLI's `benchmark` subcommand and by the auto-calibration probe.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class WeightedCase:
    case_id: str
    origin: str
    B: int
    H: int
    Lq: int
    Lkv: int
    D: int
    dtype: str
    mask: str
    weight: int
    contiguous: bool = True
    dropout_p: float = 0.0


_CASES: list[WeightedCase] = [
    WeightedCase("S1", "self-attn, short",        1, 8,  256,  256, 64, "bf16", "none", 1),
    WeightedCase("S2", "self-attn, mid",          1, 8, 1024, 1024, 64, "bf16", "none", 2),
    WeightedCase("S3", "self-attn, long",         1, 8, 4096, 4096, 64, "bf16", "none", 4),
    WeightedCase("T1", "self-attn, 512",          1, 8,  512,  512, 64, "bf16", "none", 1),
    WeightedCase("T2", "self-attn, 2k",           1, 8, 2048, 2048, 64, "bf16", "none", 2),
    WeightedCase("T3", "self-attn, 8k",           1, 8, 8192, 8192, 64, "bf16", "none", 4),
    WeightedCase("X1", "cross-attn, Lq<Lkv",      1, 8, 1024, 2048, 64, "bf16", "none", 2),
    WeightedCase("X2", "cross-attn, Lq>Lkv",      1, 8, 2048, 1024, 64, "bf16", "none", 2),
    WeightedCase("M1", "bool-mask self-attn",     1, 8, 2048, 2048, 64, "bf16", "bool_b1lk", 2),
    WeightedCase("F1", "fp32 self-attn, 1k",      1, 8, 1024, 1024, 64, "fp32", "none", 1),
]


def iter_cases() -> Iterator[WeightedCase]:
    yield from _CASES


def total_weight() -> int:
    return sum(c.weight for c in _CASES)
