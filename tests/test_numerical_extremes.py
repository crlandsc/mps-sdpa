"""Numerical extreme-value tests.

- Extreme-magnitude Q/K (large: std=10; small: std=1e-4) softmax stability
- Extreme-magnitude grad_out (large upstream grads) — no NaN/Inf propagation
- fp16 AMP long-horizon convergence (smaller dyn range than bf16)

These catch softmax overflow, Adam epsilon issues, and fp16 gradient
clipping edge cases.
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
@pytest.mark.parametrize("std", [10.0, 1e-4])
def test_extreme_magnitude_qk_forward(std: float):
    """Softmax must stay stable for Q/K with std=10 (overflow risk) and
    std=1e-4 (precision loss risk)."""
    torch.manual_seed(0)
    B, H, L, D = 1, 4, 2048, 64
    q = (torch.randn(B, H, L, D, device="mps") * std).to(torch.bfloat16)
    k = (torch.randn(B, H, L, D, device="mps") * std).to(torch.bfloat16)
    v = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")

    out = sdpa_opt(q, k, v, backend="mpsgraph")
    ref = F.scaled_dot_product_attention(q, k, v)

    assert not torch.isnan(out).any(), f"std={std}: NaN in mpsgraph output"
    assert not torch.isinf(out).any(), f"std={std}: Inf in mpsgraph output"
    # Tolerance is much looser at extreme magnitudes — softmax loses resolution.
    # Key check: no NaN/Inf, and output is in roughly the same ballpark as stock.
    diff = (out - ref).abs().max().item()
    ref_mag = ref.abs().max().item()
    assert diff < 0.5 * max(ref_mag, 1.0) + 1e-2, (
        f"std={std}: diff={diff}, ref_mag={ref_mag}"
    )


@pytest.mark.skipif(not _can_run(), reason="requires MPSGraph backend + MPS device")
@pytest.mark.parametrize("grad_scale", [1e3, 1e4])
def test_extreme_grad_out_backward(grad_scale: float):
    """Large upstream gradients must not produce NaN/Inf in dQ/dK/dV."""
    torch.manual_seed(0)
    B, H, L, D = 1, 4, 2048, 64
    q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps", requires_grad=True)
    k = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps", requires_grad=True)
    v = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps", requires_grad=True)
    out = sdpa_opt(q, k, v, backend="mpsgraph")
    # Apply a large scalar upstream gradient
    (out.sum() * grad_scale).backward()
    for name, g in zip("QKV", (q.grad, k.grad, v.grad)):
        assert not torch.isnan(g).any(), f"d{name} has NaN at grad_scale={grad_scale}"
        assert not torch.isinf(g).any(), f"d{name} has Inf at grad_scale={grad_scale}"


@pytest.mark.skipif(not _can_run(), reason="requires MPSGraph backend + MPS device")
def test_fp16_long_horizon_amp_convergence():
    """fp16 AMP over 500 steps — smaller dynamic range than bf16 so this is a
    stricter test of softmax numerical stability."""
    import torch.nn as nn
    torch.manual_seed(0)
    B, L, D, H = 1, 1024, 128, 4
    x = torch.randn(B, L, D, device="mps")
    target = torch.randn(B, L, D, device="mps")

    class _M(nn.Module):
        def __init__(self, use_opt):
            super().__init__()
            self.qkv = nn.Linear(D, 3 * D, bias=False)
            self.proj = nn.Linear(D, D, bias=False)
            self.use_opt = use_opt

        def forward(self, x):
            qkv = self.qkv(x).reshape(B, L, 3, H, D // H).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            if self.use_opt:
                y = sdpa_opt(q, k, v)
            else:
                y = F.scaled_dot_product_attention(q, k, v)
            return self.proj(y.transpose(1, 2).reshape(B, L, D))

    def run(use_opt):
        torch.manual_seed(42)
        model = _M(use_opt).to("mps")
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        losses = []
        for _ in range(500):
            opt.zero_grad()
            with torch.amp.autocast("mps", dtype=torch.float16):
                y = model(x)
                loss = ((y - target) ** 2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        return losses

    losses_s = run(False)
    losses_o = run(True)

    # No NaN
    assert all(loss == loss for loss in losses_s), "stock fp16 had NaN"
    assert all(loss == loss for loss in losses_o), "opt fp16 had NaN"
    # Both converged
    assert losses_s[-1] < losses_s[0], "stock fp16 didn't train"
    assert losses_o[-1] < losses_o[0], "opt fp16 didn't train"
    # Envelope
    rel = abs(losses_s[-1] - losses_o[-1]) / (0.5 * (losses_s[-1] + losses_o[-1]))
    assert rel < 0.3, f"fp16 final drift > 30%: {losses_s[-1]:.4f} vs {losses_o[-1]:.4f}"
