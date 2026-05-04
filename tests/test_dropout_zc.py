"""Dropout support in the zero-copy backend.

Semantics:
- dropout_p=0 (inference): uses Apple's fused SDPA op (fast path).
- dropout_p>0 (training): uses an unfused graph (scores→softmax→*mask→matmul V)
  with a pre-scaled dropout mask. Backward correctly chains through the
  softmax bwd.
- Caller is responsible for setting dropout_p=0 during inference (standard
  PyTorch pattern: nn.Dropout is disabled when .training=False).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from mps_sdpa import sdpa_opt
from mps_sdpa.backends import mpsgraph as _mg
from mps_sdpa.backends import mpsgraph_zc as _zc


def _can_run() -> bool:
    try:
        return (_zc._AVAILABLE and _mg._AVAILABLE
                and torch.backends.mps.is_available())
    except Exception:
        return False


@pytest.mark.skipif(not _can_run(), reason="requires zc extension + MPS device")
@pytest.mark.parametrize("L", [1024, 2048, 4096])
def test_dropout_forward_distributional(L: int):
    """Output under dropout should match stock in mean + variance."""
    torch.manual_seed(0)
    B, H, D = 1, 4, 64
    q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
    k = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
    v = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")

    opt_samples, stock_samples = [], []
    for s in range(8):
        torch.manual_seed(s)
        opt_samples.append(sdpa_opt(q, k, v, dropout_p=0.1, backend="mpsgraph_zc").float())
        torch.manual_seed(s + 1000)
        stock_samples.append(F.scaled_dot_product_attention(q, k, v, dropout_p=0.1).float())

    opt_t = torch.stack(opt_samples)
    stock_t = torch.stack(stock_samples)
    assert abs(opt_t.mean() - stock_t.mean()) < 0.02, (
        f"mean drift: opt={opt_t.mean()}, stock={stock_t.mean()}"
    )
    assert abs(opt_t.std() - stock_t.std()) / stock_t.std() < 0.1, (
        f"std drift: opt={opt_t.std()}, stock={stock_t.std()}"
    )


@pytest.mark.skipif(not _can_run(), reason="requires zc extension + MPS device")
def test_dropout_backward_smooth():
    """Dropout training: forward + backward produces finite grads."""
    torch.manual_seed(0)
    B, H, L, D = 1, 4, 2048, 64
    q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps", requires_grad=True)
    k = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps", requires_grad=True)
    v = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps", requires_grad=True)
    out = sdpa_opt(q, k, v, dropout_p=0.1, backend="mpsgraph_zc")
    out.sum().backward()
    for g, name in [(q.grad, "q"), (k.grad, "k"), (v.grad, "v")]:
        assert g is not None
        assert not torch.isnan(g).any(), f"{name}.grad has NaN"
        assert not torch.isinf(g).any(), f"{name}.grad has Inf"
        assert g.abs().max().item() > 0, f"{name}.grad all zero"


@pytest.mark.skipif(not _can_run(), reason="requires zc extension + MPS device")
def test_dropout_p_zero_uses_fused_path():
    """dropout_p=0 must use Apple's fused op; dropout_p>0 uses unfused graph.

    Inspect the graph cache: fused and unfused have different keys so after
    one of each we should see exactly two graphs cached.
    """
    from mps_sdpa._cpp import get_ext
    ext = get_ext()
    ext.clear_graph_cache()

    B, H, L, D = 1, 4, 2048, 64
    q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
    k = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
    v = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")

    _ = sdpa_opt(q, k, v, dropout_p=0.0, backend="mpsgraph_zc")
    fused_cache = ext.graph_cache_size()

    _ = sdpa_opt(q, k, v, dropout_p=0.1, backend="mpsgraph_zc")
    after_dropout = ext.graph_cache_size()

    assert fused_cache == 1, f"expected 1 graph after dropout=0, got {fused_cache}"
    assert after_dropout == 2, f"expected 2 graphs total, got {after_dropout}"


@pytest.mark.skipif(not _can_run(), reason="requires zc extension + MPS device")
def test_dropout_with_mask():
    """Dropout + attn_mask compose correctly in the unfused graph."""
    torch.manual_seed(0)
    B, H, L, D = 1, 4, 2048, 64
    q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps", requires_grad=True)
    k = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps", requires_grad=True)
    v = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps", requires_grad=True)
    bm = (torch.rand(B, 1, L, L) > 0.2).to("mps")
    out = sdpa_opt(q, k, v, attn_mask=bm, dropout_p=0.1, backend="mpsgraph_zc")
    out.sum().backward()
    assert not torch.isnan(out).any()
    assert q.grad is not None and not torch.isnan(q.grad).any()


@pytest.mark.skipif(not _can_run(), reason="requires zc extension + MPS device")
def test_inference_mode_zero_dropout_unchanged():
    """sdpa_opt inside torch.inference_mode() with dropout_p=0 matches stock."""
    torch.manual_seed(0)
    B, H, L, D = 1, 4, 2048, 64
    q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
    k = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
    v = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
    with torch.inference_mode():
        out = sdpa_opt(q, k, v, dropout_p=0.0, backend="mpsgraph_zc")
    ref = F.scaled_dot_product_attention(q, k, v)
    assert (out - ref).abs().max().item() < 5e-3


@pytest.mark.skipif(not _can_run(), reason="requires zc extension + MPS device")
def test_training_vs_inference_dropout_pattern():
    """Standard PyTorch pattern: nn.Module sets dropout_p=0 when .training=False.

    This test verifies the pattern works with our backend — in inference mode
    the call path becomes the fast fused op; in training it uses unfused+dropout.
    """
    import torch.nn as nn

    class _Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout_p = 0.1

        def forward(self, q, k, v):
            p = self.dropout_p if self.training else 0.0
            return sdpa_opt(q, k, v, dropout_p=p, backend="mpsgraph_zc")

    m = _Attn().to("mps")
    q = torch.randn(1, 4, 1024, 64, dtype=torch.bfloat16, device="mps")
    k = torch.randn(1, 4, 1024, 64, dtype=torch.bfloat16, device="mps")
    v = torch.randn(1, 4, 1024, 64, dtype=torch.bfloat16, device="mps")

    # Inference mode — dropout disabled — must match stock exactly.
    m.train(False)
    out_inf = m(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(out_inf, ref, atol=5e-3, rtol=5e-2)

    # Training mode — dropout active — must produce non-NaN output.
    m.train(True)
    out_train = m(q, k, v)
    assert not torch.isnan(out_train).any()
