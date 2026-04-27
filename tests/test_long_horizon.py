"""Long-horizon convergence: 1000+ step stock vs opt comparison.

Short-horizon tests can miss slow drift — bf16 accumulates small differences
over thousands of steps. This test runs a minimal transformer for 1000 steps
under both stock and opt, and checks the final loss envelope.

Runs in ~20s. Sequence length is moderate (kept short to bound test runtime),
but long enough to detect cumulative drift between the two paths.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from mps_sdpa import sdpa_opt
from mps_sdpa.backends import mpsgraph as _mg


def _can_run() -> bool:
    try:
        return _mg._AVAILABLE and torch.backends.mps.is_available()
    except Exception:
        return False


class _TinyAttn(nn.Module):
    def __init__(self, D: int, H: int, use_opt: bool):
        super().__init__()
        self.D = D
        self.H = H
        self.Dh = D // H
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.proj = nn.Linear(D, D, bias=False)
        self.use_opt = use_opt

    def forward(self, x):
        B, L, _ = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.H, self.Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.use_opt:
            y = sdpa_opt(q, k, v)
        else:
            y = F.scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).reshape(B, L, self.D)
        return self.proj(y)


@pytest.mark.skipif(not _can_run(), reason="requires MPSGraph backend + MPS device")
def test_1000_step_convergence_opt_vs_stock():
    """Loss curves must stay within a tight envelope over 1000 SGD steps."""
    torch.manual_seed(0)
    B, L, D, H = 1, 2048, 128, 4
    x = torch.randn(B, L, D, dtype=torch.bfloat16, device="mps")
    target = torch.randn(B, L, D, dtype=torch.bfloat16, device="mps")
    N = 1000

    def run(use_opt: bool):
        torch.manual_seed(7)
        model = _TinyAttn(D, H, use_opt=use_opt).to(torch.bfloat16).to("mps")
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        losses = []
        for _ in range(N):
            opt.zero_grad()
            y = model(x)
            loss = ((y - target) ** 2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        return losses

    losses_s = run(False)
    losses_o = run(True)

    # Sanity: both curves decrease at least 30% over 1000 steps.
    assert losses_s[-1] < losses_s[0] * 0.7, (
        f"stock didn't train: {losses_s[0]:.4f} -> {losses_s[-1]:.4f}"
    )
    assert losses_o[-1] < losses_o[0] * 0.7, (
        f"opt didn't train: {losses_o[0]:.4f} -> {losses_o[-1]:.4f}"
    )

    # Final-loss envelope: 25% relative (bf16 + Adam momentum can drift over
    # 1000 steps, especially with single-batch memorization).
    rel = abs(losses_s[-1] - losses_o[-1]) / (0.5 * (losses_s[-1] + losses_o[-1]))
    assert rel < 0.25, (
        f"final loss drift > 25%: stock={losses_s[-1]:.4f}, opt={losses_o[-1]:.4f}"
    )

    # No NaN throughout either curve.
    assert all(x == x for x in losses_s), "NaN in stock curve"
    assert all(x == x for x in losses_o), "NaN in opt curve"
