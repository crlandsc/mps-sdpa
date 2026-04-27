"""AMP / autocast correctness.

Models commonly wrap forward in
`torch.amp.autocast("mps", dtype=bfloat16)` with fp32 weights. SDPA sees
bf16 inputs but the graph is built on the outside. Verify our backend
behaves identically to stock under autocast.
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
@pytest.mark.parametrize("cast_dtype", [torch.bfloat16, torch.float16])
def test_amp_autocast_forward_matches_stock(cast_dtype):
    """Under autocast, output must match stock within cast-dtype tolerance."""
    torch.manual_seed(0)
    B, H, Lq, Lkv, D = 1, 4, 1024, 1024, 64
    # fp32 inputs — autocast is expected to down-cast them.
    q = torch.randn(B, H, Lq, D, dtype=torch.float32, device="mps")
    k = torch.randn(B, H, Lkv, D, dtype=torch.float32, device="mps")
    v = torch.randn(B, H, Lkv, D, dtype=torch.float32, device="mps")
    with torch.amp.autocast("mps", dtype=cast_dtype):
        out = sdpa_opt(q, k, v, backend="mpsgraph")
        ref = F.scaled_dot_product_attention(q, k, v)
    # Both outputs will be in cast_dtype (autocast down-casts).
    assert out.dtype == ref.dtype, (
        f"dtype mismatch: out={out.dtype}, ref={ref.dtype}"
    )
    diff = (out - ref).abs().max().item()
    assert diff < 5e-3, f"cast_dtype={cast_dtype}: diff={diff}"


@pytest.mark.skipif(not _can_run(), reason="requires MPSGraph backend + MPS device")
def test_amp_autocast_backward_matches_stock():
    """Under autocast, gradients must also match stock within tolerance."""
    torch.manual_seed(0)
    B, H, Lq, Lkv, D = 1, 4, 1024, 1024, 64

    def run(fn):
        torch.manual_seed(0)
        q = torch.randn(B, H, Lq, D, dtype=torch.float32, device="mps", requires_grad=True)
        k = torch.randn(B, H, Lkv, D, dtype=torch.float32, device="mps", requires_grad=True)
        v = torch.randn(B, H, Lkv, D, dtype=torch.float32, device="mps", requires_grad=True)
        with torch.amp.autocast("mps", dtype=torch.bfloat16):
            out = fn(q, k, v)
        out.sum().backward()
        return q.grad, k.grad, v.grad

    dQ_m, dK_m, dV_m = run(lambda q, k, v: sdpa_opt(q, k, v, backend="mpsgraph"))
    dQ_s, dK_s, dV_s = run(lambda q, k, v: F.scaled_dot_product_attention(q, k, v))

    for name, g_m, g_s in zip("QKV", (dQ_m, dK_m, dV_m), (dQ_s, dK_s, dV_s)):
        diff = (g_m - g_s).abs().max().item()
        assert diff < 1e-2, f"d{name} diff={diff}"
