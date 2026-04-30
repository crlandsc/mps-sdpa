"""Public API contract tests."""
import pytest
import torch

from mps_sdpa import available_backends, sdpa_opt, set_default_backend


@pytest.fixture
def qkv_cpu():
    torch.manual_seed(0)
    q = torch.randn(1, 4, 16, 32, dtype=torch.float32)
    k = torch.randn(1, 4, 16, 32, dtype=torch.float32)
    v = torch.randn(1, 4, 16, 32, dtype=torch.float32)
    return q, k, v


def test_stock_cpu_roundtrip(qkv_cpu):
    q, k, v = qkv_cpu
    out = sdpa_opt(q, k, v, backend="stock")
    assert out.shape == (1, 4, 16, 32)
    assert out.dtype == torch.float32


def test_auto_backend_on_cpu_falls_back_to_stock(qkv_cpu):
    q, k, v = qkv_cpu
    out = sdpa_opt(q, k, v, backend="auto")
    assert out.shape == (1, 4, 16, 32)


def test_unknown_backend_raises(qkv_cpu):
    q, k, v = qkv_cpu
    with pytest.raises(KeyError):
        sdpa_opt(q, k, v, backend="no_such")


def test_gqa_falls_back_to_stock(qkv_cpu):
    """GQA (Hq != Hkv) routes to stock with a one-time warning."""
    import warnings

    import torch.nn.functional as F

    from mps_sdpa import api as _api
    q, _, _ = qkv_cpu  # q has 4 heads
    # Reset one-time-warning flag so the test is deterministic.
    _api._gqa_warning_emitted = False
    k = torch.randn(1, 2, 16, 32, dtype=torch.float32)
    v = torch.randn(1, 2, 16, 32, dtype=torch.float32)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = sdpa_opt(q, k, v)
    # Reference: expand K/V the same way our fallback does
    k_expanded = k.repeat_interleave(2, dim=-3)
    v_expanded = v.repeat_interleave(2, dim=-3)
    ref = F.scaled_dot_product_attention(q, k_expanded, v_expanded)
    assert torch.allclose(out, ref)
    assert any("GQA" in str(x.message) for x in w)


def test_gqa_warning_emitted_only_once(qkv_cpu):
    """The GQA fallback warning must fire at most once per process."""
    import warnings

    from mps_sdpa import api as _api
    q, _, _ = qkv_cpu
    k = torch.randn(1, 2, 16, 32, dtype=torch.float32)
    v = torch.randn(1, 2, 16, 32, dtype=torch.float32)
    _api._gqa_warning_emitted = False
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sdpa_opt(q, k, v)
        sdpa_opt(q, k, v)
        sdpa_opt(q, k, v)
    gqa_warns = [x for x in w if "GQA" in str(x.message)]
    assert len(gqa_warns) == 1, f"expected 1 GQA warning, got {len(gqa_warns)}"


def test_is_causal_delegates_correctly(qkv_cpu):
    q, k, v = qkv_cpu
    out_causal = sdpa_opt(q, k, v, is_causal=True, backend="stock")
    out_manual = sdpa_opt(
        q, k, v,
        attn_mask=torch.ones(16, 16, dtype=torch.bool).tril(),
        backend="stock",
    )
    assert torch.allclose(out_causal, out_manual, atol=1e-6)


def test_available_backends_contains_stock():
    assert "stock" in available_backends()


def test_set_default_backend_persists():
    set_default_backend("stock")
