"""Mask dtype coercion: accept mismatched mask dtype, auto-cast to Q's dtype.

Users routinely pass fp32 masks into bf16 models
(the mask comes from data, not from the model). Previously this worked by
accident through `.to(dtype=q.dtype)`; we add explicit tests so it stays
working.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from mps_sdpa import sdpa_opt
from mps_sdpa.backends import mpsgraph as _mg


def _can_run() -> bool:
    try:
        return _mg._AVAILABLE and torch.backends.mps.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _can_run(), reason="requires MPSGraph backend + MPS device")
@pytest.mark.parametrize("qkv_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("mask_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_additive_float_mask_dtype_mismatch_coerces(qkv_dtype, mask_dtype):
    """Mask dtype != qkv dtype must be auto-cast, not error out."""
    torch.manual_seed(0)
    B, H, Lq, Lkv, D = 1, 4, 1024, 1024, 64
    q = torch.randn(B, H, Lq, D, dtype=qkv_dtype, device="mps")
    k = torch.randn(B, H, Lkv, D, dtype=qkv_dtype, device="mps")
    v = torch.randn(B, H, Lkv, D, dtype=qkv_dtype, device="mps")
    # Mask in a potentially-different dtype
    m = (torch.randn(1, 1, Lq, Lkv) * -2.0).to(dtype=mask_dtype, device="mps")
    ref_m = m.to(dtype=qkv_dtype)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=ref_m)
    out = sdpa_opt(q, k, v, attn_mask=m, backend="mpsgraph")
    atol = {torch.float32: 1e-4, torch.float16: 5e-3, torch.bfloat16: 5e-3}[qkv_dtype]
    diff = (out - ref).abs().max().item()
    assert diff < atol, (
        f"qkv={qkv_dtype}, mask={mask_dtype}: diff={diff}, tol={atol}"
    )


def test_fp32_cpu_mask_on_bf16_mps_inputs():
    """User commonly passes fp32 masks from data pipeline into bf16 models."""
    torch.manual_seed(1)
    B, H, Lq, Lkv, D = 1, 4, 1024, 1024, 64
    q = torch.randn(B, H, Lq, D, dtype=torch.bfloat16, device="cpu")
    k = torch.randn(B, H, Lkv, D, dtype=torch.bfloat16, device="cpu")
    v = torch.randn(B, H, Lkv, D, dtype=torch.bfloat16, device="cpu")
    m = (torch.randn(1, 1, Lq, Lkv, dtype=torch.float32) * -2.0)
    # Stock dispatch on CPU — also benefits from the up-front coercion.
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=m.to(torch.bfloat16))
    out = sdpa_opt(q, k, v, attn_mask=m, backend="stock")
    diff = (out - ref).abs().max().item()
    assert diff < 5e-3, f"diff={diff}"
