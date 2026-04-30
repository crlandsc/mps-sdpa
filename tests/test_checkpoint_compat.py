"""Gradient-checkpointing compat.

`torch.utils.checkpoint.checkpoint` re-runs
forward during backward to save memory. Our autograd.Function + graph cache
must survive re-entry. A naive implementation could break if state gets
mutated during the first forward and then re-entered.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from mps_sdpa import sdpa_opt
from mps_sdpa.backends import mpsgraph as _mg


def _can_run() -> bool:
    try:
        return _mg._AVAILABLE and torch.backends.mps.is_available()
    except Exception:
        return False


class _DepthTwoAttention(nn.Module):
    """Minimal model with two SDPA layers; wraps both in checkpoint during train."""

    def __init__(self, use_opt: bool, use_ckpt: bool):
        super().__init__()
        self.use_opt = use_opt
        self.use_ckpt = use_ckpt
        self.proj1 = nn.Linear(64, 64, bias=False).to(torch.bfloat16)
        self.proj2 = nn.Linear(64, 64, bias=False).to(torch.bfloat16)

    def _attn(self, x):
        q = k = v = x
        if self.use_opt:
            return sdpa_opt(q, k, v, backend="mpsgraph")
        return F.scaled_dot_product_attention(q, k, v)

    def _block(self, x, proj):
        x = self._attn(x)
        b, h, seq, d = x.shape
        x_flat = x.reshape(b * h * seq, d)
        x_flat = proj(x_flat)
        return x_flat.reshape(b, h, seq, d)

    def forward(self, x):
        if self.use_ckpt:
            x = checkpoint(self._block, x, self.proj1, use_reentrant=False)
            x = checkpoint(self._block, x, self.proj2, use_reentrant=False)
        else:
            x = self._block(x, self.proj1)
            x = self._block(x, self.proj2)
        return x


@pytest.mark.skipif(not _can_run(), reason="requires MPSGraph backend + MPS device")
def test_gradient_checkpointing_opt_vs_stock():
    """checkpoint(opt) loss/grads must match checkpoint(stock) within bf16 tol."""
    torch.manual_seed(0)
    B, H, L, D = 1, 4, 1024, 64
    x_init = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")

    def run(use_opt, use_ckpt):
        torch.manual_seed(42)
        model = _DepthTwoAttention(use_opt=use_opt, use_ckpt=use_ckpt).to("mps")
        x = x_init.clone().requires_grad_(True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        return loss.item(), x.grad.clone()

    # All four combinations must agree within tolerance.
    loss_ss, grad_ss = run(False, False)  # stock, no ckpt
    loss_sc, grad_sc = run(False, True)   # stock + ckpt
    loss_os, grad_os = run(True, False)   # opt, no ckpt
    loss_oc, grad_oc = run(True, True)    # opt + ckpt

    for name, loss, grad in [
        ("stock+ckpt", loss_sc, grad_sc),
        ("opt", loss_os, grad_os),
        ("opt+ckpt", loss_oc, grad_oc),
    ]:
        # compare against stock-no-ckpt baseline
        diff_loss = abs(loss - loss_ss)
        diff_grad = (grad - grad_ss).abs().max().item()
        assert diff_loss < 1e-2 * abs(loss_ss) + 1, (
            f"{name} loss diff {diff_loss} (baseline {loss_ss})"
        )
        assert diff_grad < 5e-2, (
            f"{name} grad diff {diff_grad}"
        )
