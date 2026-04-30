"""Zero-copy MPSGraph backend — C++/Obj-C++ extension wrapper.

Uses mps_sdpa._cpp.mpsgraph_zc (compiled on first import) which extracts
MTLBuffers directly from torch MPS tensors via ATen's getMTLBufferStorage()
instead of routing through a CPU memcpy like the pyobjc path.

Feature coverage vs the pyobjc `mpsgraph` backend:
  - ✅ Forward, bf16 / fp16 / fp32
  - ✅ Causal masking (caller-built additive mask)
  - ✅ Per-head / per-batch broadcast masks
  - ✅ Backward via MPSGraph-native backward graph (manual softmax bwd)
  - ⏳ Dropout — not yet; falls back to pyobjc

When dropout is requested we fall back to the pyobjc path (which has the
unfused dropout graph). Everything else goes through the zero-copy path.
"""
from __future__ import annotations

from typing import Optional

import torch

from . import _calibrate, register_backend
from . import mpsgraph as _pyobjc_mpsgraph


def _is_oom(exc: BaseException) -> bool:
    """True iff exc represents an MPS out-of-memory condition.

    torch.OutOfMemoryError is the canonical class (stable since torch 2.4);
    some MPS code paths still raise plain RuntimeError with a substring like
    'out of memory' or 'MPS allocated', so we cover both.
    """
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "mps allocated" in msg


_EXT = None
_REASON: str | None = None

try:
    from .._cpp import get_ext, load_error
    _EXT = get_ext()
    if _EXT is None:
        _REASON = f"extension build failed: {load_error()}"
except Exception as e:
    _REASON = f"extension import failed: {type(e).__name__}: {e}"


def _fallback_to_pyobjc(q, k, v, *, attn_mask, dropout_p, is_causal, scale):
    """Route to the pyobjc mpsgraph backend (which handles autograd + dropout)."""
    return _pyobjc_mpsgraph.mpsgraph_sdpa(
        q, k, v,
        attn_mask=attn_mask, dropout_p=dropout_p,
        is_causal=is_causal, scale=scale,
    )


def _build_causal_mask(Lq: int, Lkv: int, dtype: torch.dtype) -> torch.Tensor:
    m = torch.full((Lq, Lkv), float("-inf"), dtype=torch.float32)
    m = torch.triu(m, diagonal=1)
    return m.to(dtype=dtype, device="mps").view(1, 1, Lq, Lkv).contiguous()


def _build_additive_mask(bool_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    add = torch.zeros(bool_mask.shape, dtype=torch.float32, device=bool_mask.device)
    add.masked_fill_(~bool_mask, float("-inf"))
    return add.to(dtype=dtype).contiguous()


def mpsgraph_zc_sdpa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    *, attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Public entry. Dispatches among zero-copy ext / pyobjc fallback / stock."""
    # Extension load might have failed — route everything through the pyobjc
    # backend which has its own stock-fallback logic.
    if _EXT is None:
        return _fallback_to_pyobjc(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                                    is_causal=is_causal, scale=scale)

    # Non-MPS or unsupported dtype: let pyobjc backend handle (it falls back to stock).
    if q.device.type != "mps" or q.dtype not in _pyobjc_mpsgraph._MPS_DTYPE:
        return _fallback_to_pyobjc(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                                    is_causal=is_causal, scale=scale)

    B, H, Lq, D = q.shape
    _, _, Lkv, _ = k.shape

    # Short-seq: zero-copy has much lower overhead, so the crossover shifts
    # toward the left. For now we use half the pyobjc threshold as a simple
    # default. (Full auto-calibration for zc is future work.)
    elem_size = q.element_size()
    attn_bytes = Lq * Lkv * elem_size
    dkey = _calibrate.dtype_key(q.dtype)
    thresholds = _calibrate.get_thresholds()
    fused_min = thresholds["fused_min_bytes"].get(dkey) if dkey else None
    if fused_min is None:
        return _fallback_to_pyobjc(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                                    is_causal=is_causal, scale=scale)
    # zc overhead is ~3x lower than pyobjc-copy at 1024^2 bf16, so halve threshold
    zc_threshold = max(fused_min // 4, 64 * 1024)  # min 64 KB safety floor
    if attn_bytes < zc_threshold:
        return _fallback_to_pyobjc(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                                    is_causal=is_causal, scale=scale)

    # Build mask if needed (same logic as pyobjc backend, just reused)
    mask_tensor: Optional[torch.Tensor] = None
    if is_causal and attn_mask is not None:
        return _fallback_to_pyobjc(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                                    is_causal=is_causal, scale=scale)
    if is_causal:
        mask_tensor = _build_causal_mask(Lq, Lkv, q.dtype)
    elif attn_mask is not None:
        m = attn_mask
        while m.dim() < 4:
            m = m.unsqueeze(0)
        if m.shape[0] not in {1, B} or m.shape[1] not in {1, H}:
            return _fallback_to_pyobjc(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                                        is_causal=is_causal, scale=scale)
        if m.shape[-2] != Lq or m.shape[-1] != Lkv:
            return _fallback_to_pyobjc(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                                        is_causal=is_causal, scale=scale)
        if m.dtype == torch.bool:
            mask_tensor = _build_additive_mask(m, q.dtype)
        else:
            mask_tensor = m.to(dtype=q.dtype).contiguous()

    # Apply scale: the graph uses D^-0.5; if user gave a different scale, pre-scale Q.
    default_scale = D ** -0.5
    q_eff = q
    if scale is not None and abs(scale - default_scale) > 1e-9:
        q_eff = q * (scale / default_scale)

    # Build the dropout mask here (pre-scaled 1/(1-p)) so we can pass to both
    # forward and backward graphs. Only materialize when dropout_p > 0.
    dropout_mask: Optional[torch.Tensor] = None
    if dropout_p > 0:
        keep = 1.0 - dropout_p
        rand = torch.rand(B, H, Lq, Lkv, device="mps", dtype=q.dtype)
        dropout_mask = (rand < keep).to(q.dtype) / keep

    try:
        # This is the true "zero-copy path succeeded" counter — reached only
        # after every short-seq / dtype / mask / dropout fallback check has passed.
        try:
            from ..api import _call_counts
            _call_counts["mpsgraph_zc"] += 1
        except Exception:
            pass
        # Both grad and no-grad paths route through the registered custom op
        # for torch.compile compatibility. The op's register_autograd handles
        # the backward; register_fake handles FakeTensor shape propagation.
        from . import torch_compile_op as _tco
        return _tco.sdpa_forward_op(
            q_eff.contiguous(), k.contiguous(), v.contiguous(),
            mask_tensor, dropout_mask,
        )
    except (torch.OutOfMemoryError, RuntimeError) as e:
        # Graceful OOM fallback (unfused dropout graph can OOM at very long seqs).
        # Free any cached blocks before retry so the fallback has the best chance.
        if _is_oom(e):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
            return _fallback_to_pyobjc(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                                        is_causal=is_causal, scale=scale)
        raise


_AVAILABLE = _EXT is not None
register_backend("mpsgraph_zc", mpsgraph_zc_sdpa, available=_AVAILABLE, reason=_REASON)
