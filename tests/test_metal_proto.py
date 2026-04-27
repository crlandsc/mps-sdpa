"""Metal prototype kernel tests ().

The naive Metal kernel is correct but 60× slower than Apple's fused op via
our mpsgraph_zc backend — documented here so anyone who wants to try a
full FA2 implementation has a starting point + measurement harness.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


def _can_run() -> bool:
    try:
        return torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")
    except Exception:
        return False


@pytest.mark.skipif(not _can_run(), reason="requires MPS + compile_shader API")
def test_metal_proto_forward_correctness():
    from mps_sdpa import sdpa_opt
    torch.manual_seed(0)
    B, H, L, D = 1, 4, 512, 64
    q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
    k = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
    v = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
    out = sdpa_opt(q, k, v, backend="metal_proto")
    ref = F.scaled_dot_product_attention(q, k, v)
    diff = (out - ref).abs().max().item()
    assert diff < 5e-3, f"diff={diff}"


@pytest.mark.skipif(not _can_run(), reason="requires MPS + compile_shader API")
def test_metal_proto_not_in_auto_pref_above_mpsgraph():
    """Auto dispatch must NOT pick metal_proto over mpsgraph_zc / mpsgraph.

    Pin this so that a future reordering of the preference list doesn't
    accidentally promote the naive kernel.
    """
    from mps_sdpa import api as _api
    q = torch.randn(1, 4, 2048, 64, dtype=torch.bfloat16, device="mps")
    picked = _api._pick_auto(q)
    assert picked in ("mpsgraph_zc", "mpsgraph"), (
        f"auto picked {picked!r}; expected zc or pyobjc"
    )


@pytest.mark.skipif(not _can_run(), reason="requires MPS + compile_shader API")
def test_metal_proto_rejects_unsupported_features():
    """Probe kernel only handles bf16 no-mask no-dropout. Other calls must raise."""
    from mps_sdpa.backends.metal_proto import metal_proto_sdpa
    q = torch.randn(1, 4, 128, 64, dtype=torch.float32, device="mps")
    k = torch.randn(1, 4, 128, 64, dtype=torch.float32, device="mps")
    v = torch.randn(1, 4, 128, 64, dtype=torch.float32, device="mps")
    with pytest.raises(NotImplementedError):
        metal_proto_sdpa(q, k, v)  # fp32 not supported

    q = q.to(torch.bfloat16)
    k = k.to(torch.bfloat16)
    v = v.to(torch.bfloat16)
    with pytest.raises(NotImplementedError):
        metal_proto_sdpa(q, k, v, is_causal=True)  # causal not supported

    with pytest.raises(NotImplementedError):
        metal_proto_sdpa(q, k, v, dropout_p=0.1)  # dropout not supported
