import torch

from mps_sdpa.training_check import synthetic_train as st


def test_builder_creates_module_with_grad():
    m = st.build_tiny_module()
    x = torch.randn(2, 512, 64)
    y = m(x)
    loss = y.pow(2).mean()
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
    assert has_grad


def test_train_returns_loss_series_of_requested_length():
    losses = st.train(steps=10, seed=0, dtype="fp32", use_opt=False, device="cpu")
    assert len(losses) == 10


def test_train_two_runs_same_seed_produce_equal_losses():
    a = st.train(steps=5, seed=7, dtype="fp32", use_opt=False, device="cpu")
    b = st.train(steps=5, seed=7, dtype="fp32", use_opt=False, device="cpu")
    for la, lb in zip(a, b):
        assert abs(la - lb) < 1e-6, f"got {la} vs {lb}"
