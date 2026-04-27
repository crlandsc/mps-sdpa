"""Partial requires_grad support.

Verify that gradients are computed correctly
when only a subset of (Q, K, V) have requires_grad=True. The remaining
inputs must have no .grad attribute, not a zero tensor.
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


def _make_qkv(grad_pattern: tuple[bool, bool, bool], *, L: int = 1024, D: int = 64):
    B, H = 1, 4
    torch.manual_seed(0)
    q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps",
                    requires_grad=grad_pattern[0])
    k = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps",
                    requires_grad=grad_pattern[1])
    v = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps",
                    requires_grad=grad_pattern[2])
    return q, k, v


@pytest.mark.skipif(not _can_run(), reason="requires MPSGraph backend + MPS device")
@pytest.mark.parametrize("pattern", [
    (True, False, False),   # Q only
    (False, True, False),   # K only
    (False, False, True),   # V only
    (True, True, False),    # Q + K
    (True, False, True),    # Q + V
    (False, True, True),    # K + V
    (True, True, True),     # all three
])
def test_partial_requires_grad(pattern):
    """Only the tensors with requires_grad=True should receive gradients."""
    q, k, v = _make_qkv(pattern)
    out = sdpa_opt(q, k, v, backend="mpsgraph")
    out.sum().backward()
    for name, t, req in zip("qkv", (q, k, v), pattern):
        if req:
            assert t.grad is not None, f"{name}.grad should be set"
            assert not torch.isnan(t.grad).any(), f"{name}.grad has NaN"
        else:
            assert t.grad is None, f"{name}.grad should be None (no requires_grad)"


@pytest.mark.skipif(not _can_run(), reason="requires MPSGraph backend + MPS device")
def test_partial_grad_matches_stock():
    """With only Q.requires_grad=True, dQ must match what stock produces."""
    L, D = 1024, 64
    # mpsgraph path
    torch.manual_seed(0)
    q_m, k_m, v_m = _make_qkv((True, False, False), L=L, D=D)
    out_m = sdpa_opt(q_m, k_m, v_m, backend="mpsgraph")
    out_m.sum().backward()

    # stock path — same seed & params
    torch.manual_seed(0)
    q_s, k_s, v_s = _make_qkv((True, False, False), L=L, D=D)
    out_s = F.scaled_dot_product_attention(q_s, k_s, v_s)
    out_s.sum().backward()

    diff = (q_m.grad - q_s.grad).abs().max().item()
    assert diff < 5e-3, f"dQ diff={diff}"
    assert k_m.grad is None
    assert v_m.grad is None


@pytest.mark.skipif(not _can_run(), reason="requires MPSGraph backend + MPS device")
def test_no_requires_grad_no_autograd_path():
    """With no inputs requiring grad, autograd.Function is not invoked."""
    q, k, v = _make_qkv((False, False, False))
    out = sdpa_opt(q, k, v, backend="mpsgraph")
    # No grad_fn means no autograd machinery was invoked.
    assert out.grad_fn is None
    assert out.requires_grad is False
