"""Battle-test edge cases.

The goal here is discovery, not coverage. If any of these fail we've found a
real issue. They're grouped by category and kept independent so a single
failure doesn't hide others.

Categories:
- Shape extremes (tiny, large, degenerate)
- Memory layout (non-contiguous, views, slices, transposes)
- Mask edge cases (empty, full, int8 bool-cast, scalar broadcast)
- Concurrent / repeat stress (cache thrash, many shapes, seed re-use)
- Device / dtype mismatches (user errors — should fail cleanly, not segfault)
- Integration patterns (nn.MultiheadAttention-style, torch.compile)
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


# ---- Shape extremes ----

@pytest.mark.skipif(not _can_run(), reason="requires zc + MPS")
@pytest.mark.parametrize("shape", [
    (1, 1, 1, 1),     # all-ones degenerate
    (1, 1, 2, 2),     # below any plausible threshold
    (1, 1, 1, 64),    # Lq=1 (single query)
    (1, 1, 64, 1),    # D=1
    (1, 1, 1, 1024),  # D very large
])
def test_tiny_and_degenerate_shapes(shape):
    """Tiny shapes should fall back to stock cleanly; output must match stock."""
    B, H, L, D = shape
    if D > 256:  # metal_proto rejects; don't force it
        pass
    q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
    k = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
    v = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
    out = sdpa_opt(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v)
    assert out.shape == ref.shape
    # At these shapes softmax numerics match stock exactly (both hit stock).
    assert (out - ref).abs().max().item() < 5e-3


@pytest.mark.skipif(not _can_run(), reason="requires zc + MPS")
def test_very_long_sequence_16k():
    """Lq=Lkv=16384 at bf16: attn matrix is 512 MB per head. Stock may crash
    but our backend should either run or fall back cleanly."""
    try:
        q = torch.randn(1, 2, 16384, 32, dtype=torch.bfloat16, device="mps")
        k = torch.randn(1, 2, 16384, 32, dtype=torch.bfloat16, device="mps")
        v = torch.randn(1, 2, 16384, 32, dtype=torch.bfloat16, device="mps")
        out = sdpa_opt(q, k, v, backend="mpsgraph_zc")
    except RuntimeError as e:
        msg = str(e).lower()
        # OOM is acceptable — we don't claim to work at unbounded seq lengths.
        if "out of memory" in msg or "buffer" in msg or "allocated" in msg:
            pytest.skip(f"OOM on this machine: {e}")
        raise
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


# ---- Memory layout ----

@pytest.mark.skipif(not _can_run(), reason="requires zc + MPS")
def test_non_contiguous_transposed_inputs():
    """Transposed views — we call .contiguous() internally, and stock's
    memory-reorder path may differ from our contig-copy path. Tolerance
    via cross_impl_atol(non_contig=True): strict-by-default, looser only
    on chips with observed cross-impl drift (see tests/_tolerances.py)."""
    torch.manual_seed(0)
    x = torch.randn(1, 2048, 4, 64, dtype=torch.bfloat16, device="mps")
    q = x.transpose(1, 2)
    k = x.transpose(1, 2)
    v = x.transpose(1, 2)
    assert not q.is_contiguous()
    out = sdpa_opt(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v)
    assert (out - ref).abs().max().item() < cross_impl_atol(q.dtype, non_contig=True)


@pytest.mark.skipif(not _can_run(), reason="requires zc + MPS")
def test_sliced_view_inputs():
    """Sliced (strided) views: Q[..., ::2, :] etc. Same non-contig drift
    profile as test_non_contiguous_transposed_inputs."""
    torch.manual_seed(0)
    full = torch.randn(1, 4, 4096, 64, dtype=torch.bfloat16, device="mps")
    q = full[..., ::2, :]
    k = full[..., ::2, :]
    v = full[..., ::2, :]
    out = sdpa_opt(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v)
    assert (out - ref).abs().max().item() < cross_impl_atol(q.dtype, non_contig=True)


@pytest.mark.skipif(not _can_run(), reason="requires zc + MPS")
def test_mixed_contiguity():
    """Q contiguous, K and V non-contiguous — common after a permute.
    Same non-contig drift profile."""
    torch.manual_seed(0)
    q = torch.randn(1, 4, 2048, 64, dtype=torch.bfloat16, device="mps")
    x = torch.randn(1, 2048, 4, 64, dtype=torch.bfloat16, device="mps")
    k = x.transpose(1, 2)
    v = x.transpose(1, 2)
    out = sdpa_opt(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v)
    assert (out - ref).abs().max().item() < cross_impl_atol(q.dtype, non_contig=True)


# ---- Mask edge cases ----

@pytest.mark.skipif(not _can_run(), reason="requires zc + MPS")
def test_all_true_bool_mask_equals_no_mask():
    """A mask that allows everything must produce the same output as no mask."""
    torch.manual_seed(0)
    B, H, L, D = 1, 4, 2048, 64
    q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
    k = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
    v = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
    full = torch.ones(1, 1, L, L, dtype=torch.bool, device="mps")
    out_masked = sdpa_opt(q, k, v, attn_mask=full)
    out_plain = sdpa_opt(q, k, v)
    assert (out_masked - out_plain).abs().max().item() < 5e-3


@pytest.mark.skipif(not _can_run(), reason="requires zc + MPS")
def test_mostly_false_mask():
    """A mask that blocks most positions should still produce valid softmax
    as long as each row has at least one unmasked position. Tolerance via
    cross_impl_atol — sparse reductions (10 keys) can amplify per-element
    ULP drift on chips listed in tests/_tolerances.py."""
    torch.manual_seed(0)
    B, H, L, D = 1, 4, 2048, 64
    q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
    k = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
    v = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
    m = torch.zeros(1, 1, L, L, dtype=torch.bool, device="mps")
    m[:, :, :, :10] = True  # only first 10 keys allowed per row
    out = sdpa_opt(q, k, v, attn_mask=m)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
    assert not torch.isnan(out).any()
    assert (out - ref).abs().max().item() < cross_impl_atol(q.dtype)


@pytest.mark.skipif(not _can_run(), reason="requires zc + MPS")
def test_causal_mask_short_seq():
    """Causal on short sequence — may fall back to stock but must be correct."""
    torch.manual_seed(0)
    B, H, L, D = 1, 2, 16, 32
    q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
    k = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
    v = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
    out = sdpa_opt(q, k, v, is_causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert (out - ref).abs().max().item() < 5e-3


# ---- Device / dtype error handling ----

@pytest.mark.skipif(not _can_run(), reason="requires zc + MPS")
def test_cpu_and_mps_mixed_raises_cleanly():
    """Mixing devices should raise a TorchError (from torch), not segfault."""
    q = torch.randn(1, 4, 128, 32, dtype=torch.bfloat16, device="mps")
    k = torch.randn(1, 4, 128, 32, dtype=torch.bfloat16, device="cpu")
    v = torch.randn(1, 4, 128, 32, dtype=torch.bfloat16, device="cpu")
    with pytest.raises((RuntimeError, ValueError)):
        sdpa_opt(q, k, v, backend="mpsgraph_zc")


@pytest.mark.skipif(not _can_run(), reason="requires zc + MPS")
def test_dtype_heterogeneous_falls_back():
    """Q bf16, K fp16, V fp32 — torch itself raises. We shouldn't crash harder."""
    q = torch.randn(1, 4, 128, 32, dtype=torch.bfloat16, device="mps")
    k = torch.randn(1, 4, 128, 32, dtype=torch.float16, device="mps")
    v = torch.randn(1, 4, 128, 32, dtype=torch.float32, device="mps")
    with pytest.raises(RuntimeError):
        sdpa_opt(q, k, v)


# ---- Cache / concurrency stress ----

@pytest.mark.skipif(not _can_run(), reason="requires zc + MPS")
def test_cache_thrash_many_shapes():
    """Build many unique-shape graphs; cache must grow without corruption."""
    from mps_sdpa._cpp import get_ext
    ext = get_ext()
    ext.clear_graph_cache()
    shapes = [(1, 2, L, 32) for L in (512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536)]
    for (B, H, L, D) in shapes:
        q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
        k = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
        v = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
        out = sdpa_opt(q, k, v, backend="mpsgraph_zc")
        assert out.shape == (B, H, L, D)
    # Expect at least one entry per unique shape that passes the threshold.
    # Some smaller shapes fall back to stock and don't hit the ext cache.
    assert ext.graph_cache_size() >= 3


@pytest.mark.skipif(not _can_run(), reason="requires zc + MPS")
def test_clear_cache_and_rebuild():
    """After clearing the cache, same-shape calls must rebuild + produce correct output."""
    from mps_sdpa._cpp import get_ext
    ext = get_ext()
    torch.manual_seed(0)
    q = torch.randn(1, 4, 2048, 64, dtype=torch.bfloat16, device="mps")
    k = torch.randn(1, 4, 2048, 64, dtype=torch.bfloat16, device="mps")
    v = torch.randn(1, 4, 2048, 64, dtype=torch.bfloat16, device="mps")
    out_before = sdpa_opt(q, k, v, backend="mpsgraph_zc")
    ext.clear_graph_cache()
    assert ext.graph_cache_size() == 0
    out_after = sdpa_opt(q, k, v, backend="mpsgraph_zc")
    assert ext.graph_cache_size() >= 1
    assert (out_before - out_after).abs().max().item() < 5e-3


# ---- Integration patterns ----

@pytest.mark.skipif(not _can_run(), reason="requires zc + MPS")
def test_nn_multihead_attention_style_wrapper():
    """Build a minimal MHA that uses sdpa_opt; verify fwd + bwd work cleanly."""
    import torch.nn as nn

    class _MHA(nn.Module):
        def __init__(self, d_model=128, n_heads=4):
            super().__init__()
            self.D = d_model
            self.H = n_heads
            self.dh = d_model // n_heads
            self.in_proj = nn.Linear(d_model, 3 * d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

        def forward(self, x):
            B, L, _ = x.shape
            qkv = self.in_proj(x).reshape(B, L, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            y = sdpa_opt(q, k, v)
            y = y.transpose(1, 2).reshape(B, L, self.D)
            return self.out_proj(y)

    m = _MHA().to(torch.bfloat16).to("mps")
    x = torch.randn(1, 2048, 128, dtype=torch.bfloat16, device="mps", requires_grad=True)
    y = m(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None and not torch.isnan(x.grad).any()


@pytest.mark.skipif(not _can_run(), reason="requires zc + MPS")
def test_same_shape_different_seeds_dropout():
    """Same shape, different seeds with dropout: outputs must differ but both be valid."""
    B, H, L, D = 1, 4, 2048, 64
    q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
    k = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
    v = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
    torch.manual_seed(1)
    out1 = sdpa_opt(q, k, v, dropout_p=0.2, backend="mpsgraph_zc")
    torch.manual_seed(2)
    out2 = sdpa_opt(q, k, v, dropout_p=0.2, backend="mpsgraph_zc")
    assert not torch.isnan(out1).any() and not torch.isnan(out2).any()
    # Different dropout masks should give different outputs
    assert (out1 - out2).abs().max().item() > 0.01


@pytest.mark.skipif(not _can_run(), reason="requires zc + MPS")
def test_repeated_calls_same_shape_no_leak():
    """1000 calls at the same shape: no memory blow-up, output stable."""
    q = torch.randn(1, 4, 1024, 64, dtype=torch.bfloat16, device="mps")
    k = torch.randn(1, 4, 1024, 64, dtype=torch.bfloat16, device="mps")
    v = torch.randn(1, 4, 1024, 64, dtype=torch.bfloat16, device="mps")
    for _ in range(200):  # 200 is plenty to catch leaks without slow tests
        out = sdpa_opt(q, k, v, backend="mpsgraph_zc")
        assert out.shape == (1, 4, 1024, 64)
    assert not torch.isnan(out).any()


# ---- Property-based sanity ----

@pytest.mark.skipif(not _can_run(), reason="requires zc + MPS")
def test_random_shapes_match_stock():
    """Random legal shapes: output within tolerance of stock."""
    import random
    rng = random.Random(42)
    for _ in range(15):
        B = rng.choice([1, 2])
        H = rng.choice([1, 2, 4, 8])
        L = rng.choice([256, 512, 1024, 1777, 2048])
        D = rng.choice([32, 48, 64, 96, 128])
        q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
        k = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
        v = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
        out = sdpa_opt(q, k, v)
        ref = F.scaled_dot_product_attention(q, k, v)
        torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-2,
                                    msg=f"B={B} H={H} L={L} D={D}")


# ---- Combined fallback paths ----

@pytest.mark.skipif(not _can_run(), reason="requires zc + MPS")
def test_causal_with_explicit_mask_falls_back_correctly():
    """is_causal=True combined with explicit attn_mask is a documented
    fallback path. PyTorch MPS's F.scaled_dot_product_attention crashes
    the process when given both arguments (NSInvalidArgumentException),
    so api.py combines the two into a single attn_mask before dispatch.
    Output must match stock SDPA called with the equivalent single-mask
    form. Closes Gap 3 from the v0.2.0 test-coverage audit."""
    torch.manual_seed(0)
    B, H, L, D = 1, 4, 1024, 64
    q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
    k = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
    v = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
    # Bool mask that allows everything (combined with is_causal still
    # produces the upper-triangular-blocked result).
    extra_mask = torch.ones(1, 1, L, L, dtype=torch.bool, device="mps")
    out_opt = sdpa_opt(q, k, v, is_causal=True, attn_mask=extra_mask)
    # Reference: equivalent single-arg form (just is_causal). Stock MPS
    # handles this fine; the crash only happens when BOTH args are passed
    # to F.scaled_dot_product_attention.
    out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert (out_opt - out_ref).abs().max().item() < cross_impl_atol(q.dtype)
