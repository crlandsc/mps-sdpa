"""retain_graph=True / gradient accumulation compat.

A common training pattern is:
  for mini_batch in batches:
      loss = model(mini_batch)
      loss.backward(retain_graph=True)  # accumulate into .grad
  optimizer.step()

Our custom Function must survive multi-step backward with retain_graph.
The concern: MTLBuffer / MPSGraphTensorData objects created inside
backward() must not be GC'd prematurely if autograd re-enters.
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
def test_retain_graph_accumulates_grads_correctly():
    """Two backward passes with retain_graph: grads accumulate, no crash."""
    torch.manual_seed(0)
    B, H, L, D = 1, 4, 2048, 64
    q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps", requires_grad=True)
    k = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps", requires_grad=True)
    v = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps", requires_grad=True)
    out = sdpa_opt(q, k, v, backend="mpsgraph")
    loss = out.sum()

    loss.backward(retain_graph=True)
    dQ1 = q.grad.clone()

    loss.backward(retain_graph=True)
    dQ2 = q.grad.clone()

    # After second backward, q.grad should be 2x the first backward.
    diff = (dQ2 - 2 * dQ1).abs().max().item()
    assert diff < 5e-2, (
        f"accumulation drift: dQ2 != 2*dQ1, max diff={diff}"
    )


@pytest.mark.skipif(not _can_run(), reason="requires MPSGraph backend + MPS device")
def test_grad_accumulation_across_microbatches():
    """Simulated grad-accumulation pattern: zero grads, N backwards, one step."""
    torch.manual_seed(1)
    B, H, L, D = 1, 4, 2048, 64
    n_micro = 4
    # "Weights" — shared across microbatches
    linear = torch.nn.Linear(D, D, bias=False).to(torch.bfloat16).to("mps")

    # Accumulated grad over n_micro microbatches, one synchronous step.
    linear.zero_grad()
    for i in range(n_micro):
        torch.manual_seed(10 + i)
        q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
        k = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
        v = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
        q = linear(q)
        out = sdpa_opt(q, k, v, backend="mpsgraph")
        loss = out.sum() / n_micro
        loss.backward()

    assert linear.weight.grad is not None
    assert not torch.isnan(linear.weight.grad).any()
    assert linear.weight.grad.abs().max().item() > 0  # non-trivial grad
