"""Verify torch.compile produces the same numerics as eager for sdpa_opt.

The custom-op + register_autograd path in torch_compile_op.py is what
makes this work. These tests fail if Dynamo graph-breaks or if the
fake-tensor kernel mis-propagates shape/dtype.
"""
from __future__ import annotations

import pytest
import torch

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
@pytest.mark.parametrize("L", [1024, 2048])
def test_compile_forward_matches_eager(L: int):
    """torch.compile(sdpa_opt) must produce numerically identical output
    to eager-mode sdpa_opt within cross-impl tolerance."""
    torch.manual_seed(0)
    q = torch.randn(1, 4, L, 64, dtype=torch.bfloat16, device="mps")
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    out_eager = sdpa_opt(q, k, v)

    compiled = torch.compile(sdpa_opt, dynamic=False)
    out_compiled = compiled(q, k, v)

    diff = (out_eager - out_compiled).abs().max().item()
    assert diff < cross_impl_atol(q.dtype), \
        f"compile output diverges from eager: max_diff={diff}"


@pytest.mark.skipif(not _can_run(), reason="requires zc + MPS")
def test_compile_backward_matches_eager():
    """Gradients via torch.compile must match eager."""
    torch.manual_seed(0)
    B, H, L, D = 1, 4, 2048, 64

    q_e = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps", requires_grad=True)
    k_e = torch.randn_like(q_e).detach().requires_grad_(True)
    v_e = torch.randn_like(q_e).detach().requires_grad_(True)

    q_c = q_e.detach().clone().requires_grad_(True)
    k_c = k_e.detach().clone().requires_grad_(True)
    v_c = v_e.detach().clone().requires_grad_(True)

    out_e = sdpa_opt(q_e, k_e, v_e)
    out_c = torch.compile(sdpa_opt, dynamic=False)(q_c, k_c, v_c)

    grad = torch.randn_like(out_e)
    out_e.backward(grad)
    out_c.backward(grad)

    atol = 2 * cross_impl_atol(q_e.dtype)
    assert (q_e.grad - q_c.grad).abs().max().item() < atol
    assert (k_e.grad - k_c.grad).abs().max().item() < atol
    assert (v_e.grad - v_c.grad).abs().max().item() < atol


@pytest.mark.skipif(not _can_run(), reason="requires zc + MPS")
def test_compile_with_mask():
    """Compile path must handle mask kwargs."""
    torch.manual_seed(0)
    B, H, L, D = 1, 4, 1024, 64
    q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    m = torch.ones(1, 1, L, L, dtype=torch.bool, device="mps")

    out_e = sdpa_opt(q, k, v, attn_mask=m)
    out_c = torch.compile(sdpa_opt, dynamic=False)(q, k, v, attn_mask=m)

    assert (out_e - out_c).abs().max().item() < cross_impl_atol(q.dtype)


@pytest.mark.skipif(not _can_run(), reason="requires zc + MPS")
def test_opcheck_smoke():
    """torch.library.opcheck on the registered op — basic sanity."""
    q = torch.randn(1, 4, 1024, 64, dtype=torch.bfloat16, device="mps")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    # opcheck verifies the fake kernel matches the real kernel's output shape/dtype
    torch.library.opcheck(
        torch.ops.mps_sdpa.sdpa_forward,
        (q, k, v, None, None),
    )


@pytest.mark.skipif(not _can_run(), reason="requires zc + MPS")
def test_compile_with_dropout_produces_fresh_entropy():
    """Under torch.compile, torch.rand inside sdpa_opt's dispatch must
    produce fresh entropy each call — not constant-fold to a fixed mask."""
    torch.manual_seed(0)
    B, H, L, D = 1, 8, 4096, 64  # within zc's dropout fast-window
    q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    compiled = torch.compile(sdpa_opt, dynamic=False)
    out_a = compiled(q, k, v, dropout_p=0.5)
    out_b = compiled(q, k, v, dropout_p=0.5)

    # Two calls with the same inputs but dropout_p > 0 should produce
    # different outputs (different dropout masks). If they're identical,
    # Dynamo cached the rand() result and we have a real correctness bug.
    assert (out_a - out_b).abs().max().item() > 1e-3, \
        "compile + dropout produces identical outputs across calls — entropy not fresh"


@pytest.mark.skipif(not _can_run(), reason="requires zc + MPS")
def test_second_order_grads_via_zc_path_raise_clearly():
    """Pin: removing _ZCSDPAFunction (replaced by register_autograd) means
    second-order grads on the zc backend should raise a clear error rather
    than silently produce wrong values. Mirrors test_second_order_grads.py
    but for the zc path (the existing test uses backend='mpsgraph')."""
    torch.manual_seed(0)
    B, H, L, D = 1, 4, 2048, 64
    q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps", requires_grad=True)
    k = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps", requires_grad=True)
    v = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps", requires_grad=True)
    # Force zc backend explicitly.
    out = sdpa_opt(q, k, v, backend="mpsgraph_zc")
    grad_q = torch.autograd.grad(out.sum(), q, create_graph=True)[0]
    # Second-order: torch should raise because sdpa_backward_op has no
    # register_autograd. Accept any RuntimeError — the exact message is an
    # internal torch implementation detail.
    with pytest.raises(RuntimeError):
        torch.autograd.grad(grad_q.sum(), q)
