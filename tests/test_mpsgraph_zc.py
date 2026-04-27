"""Zero-copy MPSGraph backend tests.

the mpsgraph_zc backend uses the C++ extension for
forward; falls back to the pyobjc mpsgraph backend for autograd + dropout.
Build is triggered at import (~6s one-time, then cached).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from mps_sdpa import sdpa_opt, available_backends
from mps_sdpa.backends import mpsgraph_zc as _zc
from mps_sdpa.backends import mpsgraph as _mg


def _can_run() -> bool:
    try:
        return (_mg._AVAILABLE and _zc._AVAILABLE
                and torch.backends.mps.is_available())
    except Exception:
        return False


@pytest.mark.skipif(not _can_run(), reason="requires zc extension + MPS device")
def test_zc_registered_and_available():
    assert "mpsgraph_zc" in available_backends()


@pytest.mark.skipif(not _can_run(), reason="requires zc extension + MPS device")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_zc_forward_matches_stock(dtype):
    torch.manual_seed(0)
    B, H, L, D = 1, 8, 2048, 64
    q = torch.randn(B, H, L, D, dtype=dtype, device="mps")
    k = torch.randn(B, H, L, D, dtype=dtype, device="mps")
    v = torch.randn(B, H, L, D, dtype=dtype, device="mps")
    out = sdpa_opt(q, k, v, backend="mpsgraph_zc")
    ref = F.scaled_dot_product_attention(q, k, v)
    tol = {torch.float32: 5e-4, torch.float16: 5e-3, torch.bfloat16: 5e-3}[dtype]
    diff = (out - ref).abs().max().item()
    assert diff < tol, f"dtype={dtype}: diff={diff}, tol={tol}"


@pytest.mark.skipif(not _can_run(), reason="requires zc extension + MPS device")
def test_zc_falls_back_for_grad():
    """When requires_grad=True, zc backend delegates to pyobjc autograd path."""
    torch.manual_seed(0)
    B, H, L, D = 1, 8, 2048, 64
    q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps", requires_grad=True)
    k = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps", requires_grad=True)
    v = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps", requires_grad=True)
    out = sdpa_opt(q, k, v, backend="mpsgraph_zc")
    out.sum().backward()
    for g, name in [(q.grad, "q"), (k.grad, "k"), (v.grad, "v")]:
        assert g is not None and not torch.isnan(g).any(), f"{name}.grad bad"


@pytest.mark.skipif(not _can_run(), reason="requires zc extension + MPS device")
def test_zc_handles_per_head_mask():
    torch.manual_seed(0)
    B, H, L, D = 1, 4, 2048, 64
    q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
    k = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
    v = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
    bm = (torch.rand(B, H, L, L) > 0.2).to("mps")
    out = sdpa_opt(q, k, v, attn_mask=bm, backend="mpsgraph_zc")
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=bm)
    diff = (out - ref).abs().max().item()
    assert diff < 5e-3


@pytest.mark.skipif(not _can_run(), reason="requires zc extension + MPS device")
def test_zc_handles_causal_mask():
    torch.manual_seed(0)
    B, H, L, D = 1, 4, 2048, 64
    q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
    k = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
    v = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
    out = sdpa_opt(q, k, v, is_causal=True, backend="mpsgraph_zc")
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    diff = (out - ref).abs().max().item()
    assert diff < 5e-3


@pytest.mark.skipif(not _can_run(), reason="requires zc extension + MPS device")
def test_zc_falls_back_for_dropout():
    """Dropout routes to pyobjc backend (has unfused dropout graph)."""
    torch.manual_seed(0)
    B, H, L, D = 1, 8, 4096, 64
    q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps", requires_grad=True)
    k = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps", requires_grad=True)
    v = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps", requires_grad=True)
    out = sdpa_opt(q, k, v, dropout_p=0.1, backend="mpsgraph_zc")
    out.sum().backward()
    assert not torch.isnan(out).any()


@pytest.mark.skipif(not _can_run(), reason="requires zc extension + MPS device")
def test_auto_picks_zc_when_available():
    """sdpa_opt(backend='auto') must pick mpsgraph_zc when it's available."""
    from mps_sdpa import api as _api
    q = torch.randn(1, 4, 2048, 64, dtype=torch.bfloat16, device="mps")
    assert _api._pick_auto(q) == "mpsgraph_zc"
