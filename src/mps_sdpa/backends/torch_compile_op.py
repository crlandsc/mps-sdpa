"""torch.library.custom_op registration for the zero-copy SDPA path.

This is the torch.compile-compatible surface. The same C++ extension
(_cpp.mpsgraph_zc) provides the runtime; this module wraps the forward
and backward as a registered custom op so Dynamo can trace through
sdpa_opt without graph breaks.

Layered design:
- Forward: torch.library.custom_op("mps_sdpa::sdpa_forward", ...)
- Fake kernel: shape/dtype propagation for symbolic tracing
- Autograd: torch.library.register_autograd dispatches to a backward
  custom_op which itself wraps the C++ sdpa_backward

Phase 8 Task 1 (this commit) only registers the ops. Phase 8 Task 2
switches the dispatch path in mpsgraph_zc.py to call these ops instead
of the autograd.Function-based _ZCSDPAFunction.

Note on scale handling: scale is NOT a parameter of these ops. The
dispatch wrapper mpsgraph_zc_sdpa() pre-scales q at the call site
(q_eff = q_user * s_ratio when scale != D^-0.5) before invoking this
op. Autograd's chain rule through that pre-scaling op handles the
s_ratio adjustment on dQ automatically — _backward just returns
dL/dq_eff and lets autograd do the rest. (Phase 3 Task 1 fixed a
double-scaling bug in the legacy _ZCSDPAFunction.backward that did
this manually; this register_autograd path inherits the correct
design by NOT handling scale.)
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .._cpp import get_ext

_EXT = get_ext()
_AVAILABLE = _EXT is not None


# Type alias used by the op signature.
_OptionalTensor = Optional[Tensor]


@torch.library.custom_op("mps_sdpa::sdpa_forward", mutates_args=())
def sdpa_forward_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: _OptionalTensor,
    dropout_mask: _OptionalTensor,
) -> Tensor:
    """Forward pass via the C++ extension. Inputs are already pre-scaled
    and contiguous in the caller (mpsgraph_zc.py:mpsgraph_zc_sdpa). Mask
    is the additive form (or None); dropout_mask is pre-scaled by 1/(1-p)
    (or None).

    This op is opaque to torch.compile by design — Dynamo will not peek
    inside. Shape propagation is handled by the @sdpa_forward_op.register_fake
    kernel below.
    """
    if _EXT is None:
        raise RuntimeError("mps_sdpa C++ extension unavailable")
    return _EXT.sdpa_forward(q, k, v, mask, dropout_mask)


@sdpa_forward_op.register_fake
def _sdpa_forward_fake(q, k, v, mask, dropout_mask):
    """Shape/dtype propagation under FakeTensor / torch.compile.

    Output is shape [B, H, Lq, D] = q.shape, dtype = q.dtype, on q.device.
    Mask and dropout_mask don't influence output shape (only values).
    """
    return q.new_empty(q.shape)


@torch.library.custom_op("mps_sdpa::sdpa_backward", mutates_args=())
def sdpa_backward_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    grad_out: Tensor,
    mask: _OptionalTensor,
    dropout_mask: _OptionalTensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Backward pass via the C++ extension. Returns (dQ, dK, dV)."""
    if _EXT is None:
        raise RuntimeError("mps_sdpa C++ extension unavailable")
    return _EXT.sdpa_backward(q, k, v, grad_out, mask, dropout_mask)


@sdpa_backward_op.register_fake
def _sdpa_backward_fake(q, k, v, grad_out, mask, dropout_mask):
    """dQ shape = q, dK and dV shape = k (which equals v)."""
    return (
        q.new_empty(q.shape),
        k.new_empty(k.shape),
        v.new_empty(v.shape),
    )


def _setup_context(ctx, inputs, output):
    """Save the tensors needed for the backward."""
    q, k, v, mask, dropout_mask = inputs
    ctx.save_for_backward(q, k, v)
    ctx.mask = mask
    ctx.dropout_mask = dropout_mask


def _backward(ctx, grad_out):
    """Custom-op backward. Returns gradients for each forward input.

    Forward inputs are (q, k, v, mask, dropout_mask). mask and dropout_mask
    are non-differentiable. Returned tuple aligns with that order.

    Note on scale handling: scale is NOT a parameter of this op. The
    dispatch wrapper mpsgraph_zc_sdpa() pre-scales q at the call site
    (q_eff = q_user * s_ratio when scale != D^-0.5) before invoking this
    op. Autograd's chain rule through that pre-scaling op handles the
    s_ratio adjustment on dQ automatically — this backward just returns
    dL/dq_eff and lets autograd do the rest. (Phase 3 Task 1 fixed a
    double-scaling bug in the legacy _ZCSDPAFunction.backward; this
    register_autograd path inherits the correct design.)
    """
    q, k, v = ctx.saved_tensors
    mask = ctx.mask
    dropout_mask = ctx.dropout_mask
    dQ, dK, dV = sdpa_backward_op(q, k, v, grad_out, mask, dropout_mask)
    return dQ, dK, dV, None, None


torch.library.register_autograd(
    "mps_sdpa::sdpa_forward",
    _backward,
    setup_context=_setup_context,
)
