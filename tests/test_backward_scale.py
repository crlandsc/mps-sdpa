"""Backward pass with explicit non-default scale.

The mpsgraph_zc dispatch wrapper pre-scales Q at the call site
(q_eff = q_user * s_ratio when scale != D^-0.5) and passes q_eff into
the registered custom op (mps_sdpa::sdpa_forward). Autograd's chain
rule through the q*s_ratio op handles the s_ratio adjustment on dQ
automatically — no manual scaling in the backward.

This test pins that path against stock SDPA's autograd. (An earlier
version of the legacy _ZCSDPAFunction.backward incorrectly applied
s_ratio TWICE; this test surfaced the bug, which was fixed in v0.2.0.
The class was removed in the same release; equivalent design is now
in src/mps_sdpa/backends/torch_compile_op.py::_backward.)

Closes Gap 2 from the v0.2.0 test-coverage audit.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from mps_sdpa import sdpa_opt
from mps_sdpa.backends import mpsgraph as _mg
from mps_sdpa.backends import mpsgraph_zc as _zc
from tests._tolerances import cross_impl_atol


def _can_run() -> bool:
    try:
        return (_zc._AVAILABLE and _mg._AVAILABLE
                and torch.backends.mps.is_available())
    except Exception:
        return False


@pytest.mark.skipif(not _can_run(), reason="requires zc + MPS")
@pytest.mark.parametrize("scale", [0.25, 0.5, 1.0, 2.0])
def test_non_default_scale_backward_matches_stock(scale: float):
    """sdpa_opt with explicit scale must produce gradients matching stock
    F.scaled_dot_product_attention with the same scale."""
    torch.manual_seed(0)
    B, H, L, D = 1, 4, 2048, 64

    q_opt = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps", requires_grad=True)
    k_opt = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps", requires_grad=True)
    v_opt = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps", requires_grad=True)
    q_ref = q_opt.detach().clone().requires_grad_(True)
    k_ref = k_opt.detach().clone().requires_grad_(True)
    v_ref = v_opt.detach().clone().requires_grad_(True)

    out_opt = sdpa_opt(q_opt, k_opt, v_opt, scale=scale)
    out_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, scale=scale)

    grad_out = torch.randn_like(out_opt)
    out_opt.backward(grad_out)
    out_ref.backward(grad_out)

    # Backward bands are 2× forward (standard rule). Gradient magnitudes
    # scale with the user-passed `scale` arg, so we compare max-abs-diff
    # against a max-magnitude-scaled threshold. (Element-wise torch.allclose
    # over-penalizes near-zero gradient elements that suffer disproportionate
    # bf16 ULP drift even when absolute drift is tiny — same effect used to
    # break the forward tests too; see tests/_tolerances.py.)
    # Per COMPAT.md: bf16 forward atol=5e-3, rtol=5e-2 → backward 1e-2, 1e-1.
    atol = 2 * cross_impl_atol(q_opt.dtype)
    rtol = 1e-1

    def _check(grad_opt, grad_ref, name):
        diff = (grad_opt - grad_ref).abs().max().item()
        threshold = atol + rtol * grad_ref.abs().max().item()
        assert diff < threshold, (
            f"d{name} mismatch at scale={scale}: max diff={diff:.4f}, "
            f"threshold={threshold:.4f} (atol={atol}, rtol={rtol}, "
            f"max ref={grad_ref.abs().max().item():.4f})"
        )

    _check(q_opt.grad, q_ref.grad, "Q")
    _check(k_opt.grad, k_ref.grad, "K")
    _check(v_opt.grad, v_ref.grad, "V")
