"""Per-head / per-batch mask support tests.

cover all broadcast shapes
[B_m, H_m, Lq, Lkv] where B_m in {1,B} and H_m in {1,H}, for both
bool and additive-float masks. Previous behavior: fell back to stock on
anything other than (1, 1, Lq, Lkv).
"""
from __future__ import annotations
import itertools

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


def _mask_shapes(B: int, H: int, Lq: int, Lkv: int):
    """All valid broadcast shapes (B_m, H_m, Lq, Lkv)."""
    for B_m in ({1, B}):
        for H_m in ({1, H}):
            yield (B_m, H_m, Lq, Lkv)


@pytest.mark.skipif(not _can_run(), reason="requires MPSGraph backend + MPS device")
@pytest.mark.parametrize("B,H", [(1, 4), (1, 8), (2, 4), (2, 8)])
def test_per_head_bool_mask(B: int, H: int):
    """Bool mask in all broadcast shapes must match stock within bf16 tol."""
    torch.manual_seed(0)
    Lq, Lkv, D = 1024, 1024, 64
    q = torch.randn(B, H, Lq, D, dtype=torch.bfloat16, device="mps")
    k = torch.randn(B, H, Lkv, D, dtype=torch.bfloat16, device="mps")
    v = torch.randn(B, H, Lkv, D, dtype=torch.bfloat16, device="mps")
    for shape in _mask_shapes(B, H, Lq, Lkv):
        bm = (torch.rand(shape) > 0.2).to("mps")
        ref = F.scaled_dot_product_attention(q, k, v, attn_mask=bm)
        out = sdpa_opt(q, k, v, attn_mask=bm, backend="mpsgraph")
        diff = (out - ref).abs().max().item()
        assert diff < 5e-3, f"(B={B},H={H},mshape={shape}): diff={diff}"


@pytest.mark.skipif(not _can_run(), reason="requires MPSGraph backend + MPS device")
@pytest.mark.parametrize("B,H", [(1, 4), (1, 8), (2, 4), (2, 8)])
def test_per_head_additive_float_mask(B: int, H: int):
    """Float additive mask in all broadcast shapes must match stock."""
    torch.manual_seed(1)
    Lq, Lkv, D = 1024, 1024, 64
    q = torch.randn(B, H, Lq, D, dtype=torch.bfloat16, device="mps")
    k = torch.randn(B, H, Lkv, D, dtype=torch.bfloat16, device="mps")
    v = torch.randn(B, H, Lkv, D, dtype=torch.bfloat16, device="mps")
    for shape in _mask_shapes(B, H, Lq, Lkv):
        m = (torch.randn(shape) * -2.0).to(dtype=torch.bfloat16, device="mps")
        ref = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        out = sdpa_opt(q, k, v, attn_mask=m, backend="mpsgraph")
        diff = (out - ref).abs().max().item()
        assert diff < 5e-3, f"(B={B},H={H},mshape={shape}): diff={diff}"


@pytest.mark.skipif(not _can_run(), reason="requires MPSGraph backend + MPS device")
def test_per_head_mask_backward():
    """Backward must handle per-head masks too (graph cache key includes mask_shape)."""
    torch.manual_seed(2)
    B, H, Lq, Lkv, D = 1, 4, 1024, 1024, 64
    q = torch.randn(B, H, Lq, D, dtype=torch.bfloat16, device="mps", requires_grad=True)
    k = torch.randn(B, H, Lkv, D, dtype=torch.bfloat16, device="mps", requires_grad=True)
    v = torch.randn(B, H, Lkv, D, dtype=torch.bfloat16, device="mps", requires_grad=True)
    bm = (torch.rand(B, H, Lq, Lkv) > 0.2).to("mps")
    out = sdpa_opt(q, k, v, attn_mask=bm, backend="mpsgraph")
    out.sum().backward()
    assert not torch.isnan(q.grad).any()
    assert not torch.isnan(k.grad).any()
    assert not torch.isnan(v.grad).any()
