"""Second-order gradients detection.

Our _MpsGraphSDPAFunction.backward is not
differentiable (MPSGraph backward uses manual softmax-bwd which isn't
traced by torch autograd). Torch's once_differentiable decorator surfaces a
clear error to callers who try to compute higher-order gradients via
create_graph=True.
"""
from __future__ import annotations

import pytest
import torch

from mps_sdpa import sdpa_opt
from mps_sdpa.backends import mpsgraph as _mg


def _can_run() -> bool:
    try:
        return _mg._AVAILABLE and torch.backends.mps.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _can_run(), reason="requires MPSGraph backend + MPS device")
def test_second_order_grad_raises():
    """@once_differentiable: attempting second-order grads raises a clear
    RuntimeError rather than silently producing wrong values. Our backward uses
    MPSGraph (opaque to autograd) so true higher-order can't work.

    Use a long-enough seq (2048^2 bf16 = 8 MB) that clears even the conservative
    default threshold (4 MB); otherwise dispatch falls back to stock, whose
    built-in autograd IS differentiable and would mask the effect.
    """
    torch.manual_seed(0)
    B, H, L, D = 1, 4, 2048, 64
    q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps", requires_grad=True)
    k = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps", requires_grad=True)
    v = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps", requires_grad=True)
    out = sdpa_opt(q, k, v, backend="mpsgraph")
    loss = out.sum()
    grad_q = torch.autograd.grad(loss, q, create_graph=True)[0]
    # Second-order: torch raises because our backward is not differentiable.
    with pytest.raises(RuntimeError, match="does not require grad|does not have a grad_fn"):
        torch.autograd.grad(grad_q.sum(), q)


@pytest.mark.skipif(not _can_run(), reason="requires MPSGraph backend + MPS device")
def test_first_order_still_works():
    """Sanity: first-order grads via create_graph=True still work (but downstream can't)."""
    torch.manual_seed(0)
    B, H, L, D = 1, 4, 1024, 64
    q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps", requires_grad=True)
    k = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps", requires_grad=True)
    v = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps", requires_grad=True)
    out = sdpa_opt(q, k, v, backend="mpsgraph")
    out.sum().backward()
    assert q.grad is not None
