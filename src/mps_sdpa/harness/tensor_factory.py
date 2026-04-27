"""Build q/k/v/mask tensors from a suite Case."""
from __future__ import annotations
from typing import Optional
import torch

from ..suites.correctness_shapes import Case
from ..suites.realistic_shapes import WeightedCase

_DTYPE = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def build(
    case: Case | WeightedCase, *, device: str = "cpu", seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    dtype = _DTYPE[case.dtype]
    shape_q = (case.B, case.H, case.Lq, case.D)
    shape_k = (case.B, case.H, case.Lkv, case.D)

    q = torch.randn(shape_q, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    k = torch.randn(shape_k, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    v = torch.randn(shape_k, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)

    contiguous = getattr(case, "contiguous", True)
    if not contiguous:
        q = q.transpose(-1, -2).transpose(-1, -2)
        k = k.transpose(-1, -2).transpose(-1, -2)
        v = v.transpose(-1, -2).transpose(-1, -2)

    mask = _build_mask(case, device=device, seed=seed)
    return q, k, v, mask


def _build_mask(case, *, device: str, seed: int) -> Optional[torch.Tensor]:
    B, H, Lq, Lkv = case.B, case.H, case.Lq, case.Lkv
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    if case.mask == "none":
        return None
    if case.mask == "causal":
        return None
    if case.mask == "bool_b1lk":
        m = torch.rand(B, 1, Lq, Lkv, generator=g) > 0.2
        return m.to(device=device)
    if case.mask == "bool_bhlk":
        m = torch.rand(B, H, Lq, Lkv, generator=g) > 0.2
        return m.to(device=device)
    if case.mask == "additive_float":
        m = (torch.rand(B, 1, Lq, Lkv, generator=g) * -2.0).to(dtype=_DTYPE[case.dtype])
        return m.to(device=device)
    if case.mask == "empty_row":
        m = torch.ones(B, 1, Lq, Lkv, dtype=torch.bool)
        m[:, :, 0, :] = False
        return m.to(device=device)
    raise ValueError(f"unknown mask kind: {case.mask}")
