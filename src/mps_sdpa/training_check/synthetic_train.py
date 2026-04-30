"""Tiny pure-PyTorch trainer for convergence equivalence. No Lightning, no W&B."""
from __future__ import annotations

import contextlib

import torch
import torch.nn as nn

from ..api import sdpa_opt


class _TinyAttention(nn.Module):
    def __init__(self, dim: int = 64, heads: int = 2, use_opt: bool = False):
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads
        self.scale = self.dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.use_opt = use_opt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.to_qkv(x).view(B, L, 3, self.heads, self.dim_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.use_opt:
            out = sdpa_opt(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
                scale=self.scale, backend="auto",
            )
        else:
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=self.scale,
            )
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.to_out(out)


def build_tiny_module(
    *, dim: int = 64, heads: int = 2, depth: int = 2, use_opt: bool = False,
) -> nn.Module:
    layers: list[nn.Module] = []
    for _ in range(depth):
        layers.append(_TinyAttention(dim, heads, use_opt))
        layers.append(nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)))
    return nn.Sequential(*layers)


def train(
    *, steps: int = 500, seed: int = 0, dtype: str = "fp32", use_opt: bool = False,
    device: str = "cpu", lr: float = 1e-3, B: int = 2, L: int = 512, D: int = 64,
) -> list[float]:
    torch.manual_seed(seed)
    model = build_tiny_module(dim=D, use_opt=use_opt).to(device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(B, L, D, generator=g).to(device=device)
    y = torch.randn(B, L, D, generator=g).to(device=device)

    autocast_ctx = contextlib.nullcontext()
    if dtype == "bf16" and device != "cpu":
        autocast_ctx = torch.amp.autocast(device_type=device, dtype=torch.bfloat16)
    elif dtype == "fp16" and device != "cpu":
        autocast_ctx = torch.amp.autocast(device_type=device, dtype=torch.float16)

    losses: list[float] = []
    for _ in range(steps):
        with autocast_ctx:
            pred = model(x)
            loss = (pred - y).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses
