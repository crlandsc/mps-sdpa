"""Run torch.autograd.gradcheck against a backend on a tiny fp64 case."""
from __future__ import annotations
import torch

from ..backends import get_backend


def run_gradcheck(*, backend_name: str, device: str = "cpu") -> bool:
    fn = get_backend(backend_name)
    B, H, L, D = 1, 2, 8, 4
    torch.manual_seed(0)
    q = torch.randn(B, H, L, D, dtype=torch.float64, device=device, requires_grad=True)
    k = torch.randn(B, H, L, D, dtype=torch.float64, device=device, requires_grad=True)
    v = torch.randn(B, H, L, D, dtype=torch.float64, device=device, requires_grad=True)

    def wrapper(q_, k_, v_):
        return fn(q_, k_, v_, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None)

    try:
        return torch.autograd.gradcheck(wrapper, (q, k, v), eps=1e-6, atol=1e-4, rtol=1e-3)
    except Exception:
        return False
