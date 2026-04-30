"""Reference implementations for correctness validation."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def math_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    *, attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0, is_causal: bool = False, scale: Optional[float] = None,
) -> torch.Tensor:
    """Run SDPA under the MATH backend — the numerically stable reference."""
    with sdpa_kernel([SDPBackend.MATH]):
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask, dropout_p=dropout_p,
            is_causal=is_causal, scale=scale,
        )


def cpu_fp64_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    *, attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False, scale: Optional[float] = None,
) -> torch.Tensor:
    """High-precision CPU fp64 reference — for spot-checking the fp32 math path."""
    qd = q.to(device="cpu", dtype=torch.float64)
    kd = k.to(device="cpu", dtype=torch.float64)
    vd = v.to(device="cpu", dtype=torch.float64)
    with sdpa_kernel([SDPBackend.MATH]):
        return F.scaled_dot_product_attention(
            qd, kd, vd,
            attn_mask=attn_mask.to(device="cpu") if attn_mask is not None else None,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=scale,
        )
