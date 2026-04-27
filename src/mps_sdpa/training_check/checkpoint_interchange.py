"""Checkpoint interchange: save with one path, load with the other, compare outputs."""
from __future__ import annotations
import torch
from . import synthetic_train as st
from ..harness import tolerances


def _train_and_serialize(*, use_opt: bool, device: str, steps: int, seed: int):
    torch.manual_seed(seed)
    model = st.build_tiny_module(use_opt=use_opt).to(device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(2, 512, 64, generator=g).to(device=device)
    y = torch.randn(2, 512, 64, generator=g).to(device=device)
    for _ in range(steps):
        pred = model(x)
        loss = (pred - y).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model.state_dict(), x


def _load_and_forward(sd: dict, use_opt: bool, x: torch.Tensor, device: str) -> torch.Tensor:
    model = st.build_tiny_module(use_opt=use_opt).to(device=device)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval()
    with torch.no_grad():
        return model(x)


def run_interchange(*, device: str, steps: int = 5, seed: int = 0) -> dict:
    dtype = "fp32"
    atol, rtol = tolerances.forward_tol(dtype)

    sd_opt, x1 = _train_and_serialize(use_opt=True, device=device, steps=steps, seed=seed)
    out_as_opt = _load_and_forward(sd_opt, use_opt=True, x=x1, device=device)
    out_as_stock = _load_and_forward(sd_opt, use_opt=False, x=x1, device=device)
    diff_opt_to_stock = (out_as_opt - out_as_stock).abs().max().item()
    passed_1 = torch.allclose(out_as_opt, out_as_stock, atol=atol, rtol=rtol)

    sd_stock, x2 = _train_and_serialize(use_opt=False, device=device, steps=steps, seed=seed)
    out_as_stock2 = _load_and_forward(sd_stock, use_opt=False, x=x2, device=device)
    out_as_opt2 = _load_and_forward(sd_stock, use_opt=True, x=x2, device=device)
    diff_stock_to_opt = (out_as_stock2 - out_as_opt2).abs().max().item()
    passed_2 = torch.allclose(out_as_stock2, out_as_opt2, atol=atol, rtol=rtol)

    return {
        "opt_to_stock": {"passed": passed_1, "max_abs_err": diff_opt_to_stock},
        "stock_to_opt": {"passed": passed_2, "max_abs_err": diff_stock_to_opt},
    }
