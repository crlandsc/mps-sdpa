"""MPSGraph SDPA backend — wraps Apple's native scaledDotProductAttention op via pyobjc.

Bridges torch MPS tensors to MPSGraph by routing data through CPU memcpy. Graphs
are cached per (dtype, shape, mask kind, dropout) signature. Custom autograd
Function provides forward + a manually-built backward graph.

Why this exists: PyTorch's stock MPS SDPA path (`sdpa_general_mps`) does not
dispatch to Apple's dedicated `MPSGraph.scaledDotProductAttention` op.
Wrapping that op directly produces large wins on long-sequence cases.

This pyobjc-based path is a fallback. The primary path is the Obj-C++
extension in mpsgraph_zc.py, which avoids the CPU memcpy entirely via
ATen's `getMTLBufferStorage`.
"""
from __future__ import annotations
import ctypes
import threading
from typing import Optional
import torch
from torch.autograd.function import once_differentiable

from . import register_backend

_AVAILABLE = False
_UNAVAILABLE_REASON: str | None = None

try:
    import objc
    from Metal import MTLCreateSystemDefaultDevice
    from MetalPerformanceShadersGraph import (
        MPSGraph, MPSGraphTensorData, MPSGraphDevice,
    )
    _AVAILABLE = True
except Exception as e:
    _UNAVAILABLE_REASON = f"pyobjc import failed: {type(e).__name__}: {e}"


# The MPSGraph SDPA op (scaledDotProductAttentionWithQueryTensor:...) was added in
# macOS 15.0 (Sequoia). On macOS 14.x, the framework class exists but the method
# doesn't. We can't segfault if a caller ends up dispatching to us — detect at
# import and mark the backend unavailable with a clear reason. pyobjc exposes
# Obj-C methods as snake_cased Python attributes, so hasattr() is a reliable probe.
_SDPA_METHOD_NAMES = (
    "scaledDotProductAttentionWithQueryTensor_keyTensor_valueTensor_scale_name_",
    "scaledDotProductAttentionWithQueryTensor_keyTensor_valueTensor_maskTensor_scale_name_",
)


def _check_mpsgraph_sdpa_available() -> tuple[bool, str | None]:
    """Probe whether MPSGraph exposes the scaledDotProductAttention op.

    Returns (available, reason). On macOS 14 the method is missing; on macOS 15+
    it is present. We also verify the masked variant (used for causal masking).
    """
    if not _AVAILABLE:
        return False, _UNAVAILABLE_REASON
    try:
        g = MPSGraph.alloc().init()
    except Exception as e:  # pragma: no cover - defensive
        return False, f"MPSGraph() init failed: {type(e).__name__}: {e}"
    for attr in _SDPA_METHOD_NAMES:
        if not hasattr(g, attr):
            return False, (
                f"MPSGraph missing method {attr!r} — macOS 15.0+ (Sequoia) required"
            )
    return True, None

# MPSDataType enum values from MPSCoreTypes.h:
#   MPSDataTypeFloatBit               = 0x10000000
#   MPSDataTypeAlternateEncodingBit   = 0x80000000
#   MPSDataTypeFloat32                = FloatBit | 32   = 0x10000020
#   MPSDataTypeFloat16                = FloatBit | 16   = 0x10000010
#   MPSDataTypeBFloat16               = AlternateEncodingBit | Float16 = 0x90000010
_MPS_DTYPE = {
    torch.float32: 0x10000020,
    torch.float16: 0x10000010,
    torch.bfloat16: 0x90000010,
}

# MTLResourceStorageModeShared
_MTL_STORAGE_SHARED = 0

# Cached state (lazy-init on first call)
_device = None
_gdev = None
_command_queue = None
_objc_msgSend = None
_sel_newBufferWithBytesNoCopy = None
_graph_cache: dict = {}
_bwd_graph_cache: dict = {}
# Protect cache miss → build → insert. CPython dict __getitem__/__setitem__ are
# atomic under GIL, but the compound "if key not in cache: cache[key] = build()"
# pattern races under threading. We also serialize _init_runtime().
_graph_cache_lock = threading.Lock()
_bwd_graph_cache_lock = threading.Lock()
_runtime_lock = threading.Lock()


def _init_runtime():
    global _device, _gdev, _command_queue, _objc_msgSend, _sel_newBufferWithBytesNoCopy
    if _device is not None:
        return
    with _runtime_lock:
        if _device is not None:
            return
        _device = MTLCreateSystemDefaultDevice()
        _gdev = MPSGraphDevice.deviceWithMTLDevice_(_device)
        _command_queue = _device.newCommandQueue()
        lib = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        _objc_msgSend = lib.objc_msgSend
        _objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_void_p, ctypes.c_size_t,
                                  ctypes.c_uint64, ctypes.c_void_p]
        _objc_msgSend.restype = ctypes.c_void_p
        sel_reg = lib.sel_registerName
        sel_reg.restype = ctypes.c_void_p
        sel_reg.argtypes = [ctypes.c_char_p]
        _sel_newBufferWithBytesNoCopy = sel_reg(
            b"newBufferWithBytesNoCopy:length:options:deallocator:"
        )


def _copy_tensor_to_tensor_data(t: torch.Tensor, expected_dtype_val: int):
    """Copy a torch MPS tensor into a fresh MTLBuffer and wrap as MPSGraphTensorData.

    Not zero-copy — allocates fresh buffer + memcpy via CPU. Used because direct
    zero-copy wrap of torch's MPS buffers segfaults (torch's opaque storage layout).

    Returns (tensor_data, mtlbuffer). Caller must keep both alive for the graph run.
    """
    if not t.is_contiguous():
        t = t.contiguous()
    nbytes = t.numel() * t.element_size()
    buf = _device.newBufferWithLength_options_(nbytes, _MTL_STORAGE_SHARED)
    # Route via CPU. For bfloat16, view as uint16 bytes so numpy can handle.
    cpu = t.cpu()
    if cpu.dtype == torch.bfloat16:
        arr_bytes = cpu.view(torch.int16).numpy().tobytes()
    else:
        arr_bytes = cpu.numpy().tobytes()
    # Copy into the buffer using pyobjc's varlist.as_buffer → memoryview interface.
    mv = buf.contents().as_buffer(nbytes)
    mv[:] = arr_bytes
    td = MPSGraphTensorData.alloc().initWithMTLBuffer_shape_dataType_(
        buf, list(t.shape), expected_dtype_val
    )
    return td, buf


def _copy_tensor_data_to_torch(buf, out: torch.Tensor):
    """Copy MTLBuffer contents into a pre-allocated torch MPS tensor (via CPU)."""
    nbytes = out.numel() * out.element_size()
    mv = buf.contents().as_buffer(nbytes)
    data = bytes(mv)  # snapshot
    if out.dtype == torch.bfloat16:
        cpu = torch.frombuffer(bytearray(data), dtype=torch.int16).view(out.shape).view(torch.bfloat16)
    else:
        cpu = torch.frombuffer(bytearray(data), dtype=out.dtype).view(out.shape)
    out.copy_(cpu.to("mps"))


def _build_graph(
    B, H, Lq, Lkv, D, dtype_val, mask_kind: str,
    mask_shape: Optional[tuple[int, int]] = None, dropout: bool = False,
):
    """Build or retrieve cached graph.

    mask_kind: 'none' | 'mask_present' (fused op via Apple's SDPA).
    mask_shape: (B_m, H_m) where B_m in {1, B} and H_m in {1, H}. Only used
                when mask_kind != 'none'. Allows per-head or per-batch masks.
    dropout: unfused graph with explicit dropout mask.
    """
    key = (dtype_val, B, H, Lq, Lkv, D, mask_kind, mask_shape, dropout)
    cached = _graph_cache.get(key)
    if cached is not None:
        return cached
    with _graph_cache_lock:
        cached = _graph_cache.get(key)
        if cached is not None:
            return cached
        return _build_graph_unlocked(
            key, B, H, Lq, Lkv, D, dtype_val, mask_kind, mask_shape, dropout,
        )


def _build_graph_unlocked(
    key, B, H, Lq, Lkv, D, dtype_val, mask_kind, mask_shape, dropout,
):
    g = MPSGraph.alloc().init()
    q_ph = g.placeholderWithShape_dataType_name_([B, H, Lq, D], dtype_val, "q")
    k_ph = g.placeholderWithShape_dataType_name_([B, H, Lkv, D], dtype_val, "k")
    v_ph = g.placeholderWithShape_dataType_name_([B, H, Lkv, D], dtype_val, "v")
    scale = 1.0 / (D ** 0.5)
    mask_ph = None
    drop_ph = None
    mask_pl_shape = None
    if mask_kind != "none":
        B_m, H_m = mask_shape if mask_shape is not None else (1, 1)
        mask_pl_shape = [B_m, H_m, Lq, Lkv]
    if not dropout:
        if mask_kind != "none":
            mask_ph = g.placeholderWithShape_dataType_name_(mask_pl_shape, dtype_val, "mask")
            out_ph = g.scaledDotProductAttentionWithQueryTensor_keyTensor_valueTensor_maskTensor_scale_name_(
                q_ph, k_ph, v_ph, mask_ph, scale, "sdpa_masked"
            )
        else:
            out_ph = g.scaledDotProductAttentionWithQueryTensor_keyTensor_valueTensor_scale_name_(
                q_ph, k_ph, v_ph, scale, "sdpa"
            )
    else:
        # Dropout path: unfused build. Use the same mathematical form as MATH backend.
        scale_t = g.constantWithScalar_dataType_(scale, dtype_val)
        k_T = g.transposeTensor_dimension_withDimension_name_(k_ph, -1, -2, "kT")
        qk = g.matrixMultiplicationWithPrimaryTensor_secondaryTensor_name_(q_ph, k_T, "qk")
        scores = g.multiplicationWithPrimaryTensor_secondaryTensor_name_(qk, scale_t, "scores")
        if mask_kind != "none":
            mask_ph = g.placeholderWithShape_dataType_name_(mask_pl_shape, dtype_val, "mask")
            scores = g.additionWithPrimaryTensor_secondaryTensor_name_(scores, mask_ph, "scores_masked")
        attn = g.softMaxWithTensor_axis_name_(scores, -1, "attn")
        # Dropout mask is shape [B, H, Lq, Lkv]; already scaled by 1/(1-p).
        drop_ph = g.placeholderWithShape_dataType_name_([B, H, Lq, Lkv], dtype_val, "drop_mask")
        attn_dropped = g.multiplicationWithPrimaryTensor_secondaryTensor_name_(attn, drop_ph, "attn_dropped")
        out_ph = g.matrixMultiplicationWithPrimaryTensor_secondaryTensor_name_(attn_dropped, v_ph, "out")
    _graph_cache[key] = (g, q_ph, k_ph, v_ph, mask_ph, drop_ph, out_ph)
    return _graph_cache[key]


def _build_bwd_graph(
    B, H, Lq, Lkv, D, dtype_val, mask_kind: str,
    mask_shape: Optional[tuple[int, int]] = None, dropout: bool = False,
):
    """Build a compiled MPSGraph that takes (Q, K, V, grad_out[, mask[, dropout_mask]])
    and returns (dQ, dK, dV). Recomputes attn internally.
    mask_shape: (B_m, H_m) with B_m in {1,B} H_m in {1,H}, or None when mask_kind='none'.
    """
    key = (dtype_val, B, H, Lq, Lkv, D, mask_kind, mask_shape, dropout)
    cached = _bwd_graph_cache.get(key)
    if cached is not None:
        return cached
    with _bwd_graph_cache_lock:
        cached = _bwd_graph_cache.get(key)
        if cached is not None:
            return cached
        return _build_bwd_graph_unlocked(
            key, B, H, Lq, Lkv, D, dtype_val, mask_kind, mask_shape, dropout,
        )


def _build_bwd_graph_unlocked(
    key, B, H, Lq, Lkv, D, dtype_val, mask_kind, mask_shape, dropout,
):
    g = MPSGraph.alloc().init()
    q_ph = g.placeholderWithShape_dataType_name_([B, H, Lq, D], dtype_val, "q")
    k_ph = g.placeholderWithShape_dataType_name_([B, H, Lkv, D], dtype_val, "k")
    v_ph = g.placeholderWithShape_dataType_name_([B, H, Lkv, D], dtype_val, "v")
    go_ph = g.placeholderWithShape_dataType_name_([B, H, Lq, D], dtype_val, "grad_out")
    mask_ph = None
    drop_ph = None
    if mask_kind != "none":
        B_m, H_m = mask_shape if mask_shape is not None else (1, 1)
        mask_ph = g.placeholderWithShape_dataType_name_(
            [B_m, H_m, Lq, Lkv], dtype_val, "mask",
        )
    if dropout:
        drop_ph = g.placeholderWithShape_dataType_name_([B, H, Lq, Lkv], dtype_val, "drop_mask")

    scale = 1.0 / (D ** 0.5)
    scale_t = g.constantWithScalar_dataType_(scale, dtype_val)

    k_T = g.transposeTensor_dimension_withDimension_name_(k_ph, -1, -2, "k_T")
    qk = g.matrixMultiplicationWithPrimaryTensor_secondaryTensor_name_(q_ph, k_T, "qk")
    scores = g.multiplicationWithPrimaryTensor_secondaryTensor_name_(qk, scale_t, "scores")
    if mask_ph is not None:
        scores = g.additionWithPrimaryTensor_secondaryTensor_name_(scores, mask_ph, "scores_masked")

    # attn_raw = softmax(scores) — this is what goes through softmax bwd
    attn_raw = g.softMaxWithTensor_axis_name_(scores, -1, "attn_raw")

    # Effective attn used in forward: attn_raw * dropout_mask (post-softmax)
    if drop_ph is not None:
        attn_used = g.multiplicationWithPrimaryTensor_secondaryTensor_name_(attn_raw, drop_ph, "attn_dropped")
    else:
        attn_used = attn_raw

    # dV = attn_used^T @ grad_out
    attn_T = g.transposeTensor_dimension_withDimension_name_(attn_used, -1, -2, "attn_T")
    dV_t = g.matrixMultiplicationWithPrimaryTensor_secondaryTensor_name_(attn_T, go_ph, "dV")

    # d_attn_used = grad_out @ V^T
    v_T = g.transposeTensor_dimension_withDimension_name_(v_ph, -1, -2, "v_T")
    d_attn_used = g.matrixMultiplicationWithPrimaryTensor_secondaryTensor_name_(go_ph, v_T, "d_attn_used")

    # Chain through dropout: d_attn_raw = d_attn_used * dropout_mask
    if drop_ph is not None:
        d_attn_raw = g.multiplicationWithPrimaryTensor_secondaryTensor_name_(d_attn_used, drop_ph, "d_attn_raw")
    else:
        d_attn_raw = d_attn_used

    # Softmax backward manually: d_scores = attn_raw * (d_attn_raw - sum(attn_raw * d_attn_raw))
    ad = g.multiplicationWithPrimaryTensor_secondaryTensor_name_(attn_raw, d_attn_raw, "attn_d_attn")
    sum_ad = g.reductionSumWithTensor_axis_name_(ad, -1, "sum_attn_d_attn")
    diff = g.subtractionWithPrimaryTensor_secondaryTensor_name_(d_attn_raw, sum_ad, "d_attn_minus_sum")
    d_scores = g.multiplicationWithPrimaryTensor_secondaryTensor_name_(attn_raw, diff, "d_scores")
    d_scores_scaled = g.multiplicationWithPrimaryTensor_secondaryTensor_name_(d_scores, scale_t, "d_scores_scaled")

    # dQ = d_scores_scaled @ K
    dQ_t = g.matrixMultiplicationWithPrimaryTensor_secondaryTensor_name_(d_scores_scaled, k_ph, "dQ")
    # dK = d_scores_scaled^T @ Q
    d_scores_T = g.transposeTensor_dimension_withDimension_name_(d_scores_scaled, -1, -2, "d_scores_T")
    dK_t = g.matrixMultiplicationWithPrimaryTensor_secondaryTensor_name_(d_scores_T, q_ph, "dK")

    _bwd_graph_cache[key] = (g, q_ph, k_ph, v_ph, go_ph, mask_ph, drop_ph, dQ_t, dK_t, dV_t)
    return _bwd_graph_cache[key]


def _mpsgraph_backward_inner(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    grad_out: torch.Tensor, mask_tensor: Optional[torch.Tensor],
    dropout_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the backward graph and return (dQ, dK, dV)."""
    B, H, Lq, D = q.shape
    _, _, Lkv, _ = k.shape
    dtype_val = _MPS_DTYPE[q.dtype]
    mask_kind = "mask_present" if mask_tensor is not None else "none"
    dropout = dropout_mask is not None
    mask_shape = (mask_tensor.shape[0], mask_tensor.shape[1]) if mask_tensor is not None else None

    (g, q_ph, k_ph, v_ph, go_ph, mask_ph, drop_ph,
     dQ_t, dK_t, dV_t) = _build_bwd_graph(
        B, H, Lq, Lkv, D, dtype_val, mask_kind,
        mask_shape=mask_shape, dropout=dropout,
    )

    q_td, q_buf = _copy_tensor_to_tensor_data(q.contiguous(), dtype_val)
    k_td, k_buf = _copy_tensor_to_tensor_data(k.contiguous(), dtype_val)
    v_td, v_buf = _copy_tensor_to_tensor_data(v.contiguous(), dtype_val)
    go_td, go_buf = _copy_tensor_to_tensor_data(grad_out.contiguous(), dtype_val)
    feeds = {q_ph: q_td, k_ph: k_td, v_ph: v_td, go_ph: go_td}
    if mask_ph is not None:
        m_td, m_buf = _copy_tensor_to_tensor_data(mask_tensor, dtype_val)
        feeds[mask_ph] = m_td
    if drop_ph is not None:
        d_td, d_buf = _copy_tensor_to_tensor_data(dropout_mask.contiguous(), dtype_val)
        feeds[drop_ph] = d_td

    q_nbytes = q.numel() * q.element_size()
    kv_nbytes = k.numel() * k.element_size()
    dQ_buf = _device.newBufferWithLength_options_(q_nbytes, _MTL_STORAGE_SHARED)
    dK_buf = _device.newBufferWithLength_options_(kv_nbytes, _MTL_STORAGE_SHARED)
    dV_buf = _device.newBufferWithLength_options_(kv_nbytes, _MTL_STORAGE_SHARED)
    dQ_td = MPSGraphTensorData.alloc().initWithMTLBuffer_shape_dataType_(dQ_buf, list(q.shape), dtype_val)
    dK_td = MPSGraphTensorData.alloc().initWithMTLBuffer_shape_dataType_(dK_buf, list(k.shape), dtype_val)
    dV_td = MPSGraphTensorData.alloc().initWithMTLBuffer_shape_dataType_(dV_buf, list(v.shape), dtype_val)
    results = {dQ_t: dQ_td, dK_t: dK_td, dV_t: dV_td}

    g.runWithMTLCommandQueue_feeds_targetOperations_resultsDictionary_(
        _command_queue, feeds, None, results,
    )

    dQ = torch.empty(q.shape, dtype=q.dtype, device="mps")
    dK = torch.empty(k.shape, dtype=k.dtype, device="mps")
    dV = torch.empty(v.shape, dtype=v.dtype, device="mps")
    _copy_tensor_data_to_torch(dQ_buf, dQ)
    _copy_tensor_data_to_torch(dK_buf, dK)
    _copy_tensor_data_to_torch(dV_buf, dV)
    torch.mps.synchronize()
    return dQ, dK, dV


def _build_causal_mask(Lq: int, Lkv: int, dtype: torch.dtype) -> torch.Tensor:
    m = torch.full((Lq, Lkv), float("-inf"), dtype=torch.float32)
    m = torch.triu(m, diagonal=1)
    return m.to(dtype=dtype, device="mps").view(1, 1, Lq, Lkv).contiguous()


def _build_additive_mask(bool_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    add = torch.zeros(bool_mask.shape, dtype=torch.float32, device=bool_mask.device)
    add.masked_fill_(~bool_mask, float("-inf"))
    return add.to(dtype=dtype).contiguous()


# Shape thresholds: short sequences lose to copy overhead; fall back to stock.
# Thresholds are now auto-calibrated per (chip, os, torch) via _calibrate.get_thresholds(),
# which runs a one-time micro-benchmark on first import and caches to
# ~/.cache/mps_sdpa/thresholds.json. Set MPS_SDPA_SKIP_CALIBRATION=1 to force the
# conservative M4-tuned defaults (used by tests to keep imports fast).
from . import _calibrate  # noqa: E402


import logging as _logging
import os as _os
_logger = _logging.getLogger("mps_sdpa.mpsgraph")


def _log_fallback(reason: str) -> None:
    """Log a fallback-to-stock event in a unified format AND count it.

    Silent by default (DEBUG level). Opt in via:
      MPS_SDPA_LOG_FALLBACKS=1   -> logged at INFO level, prefix '[mps_sdpa.mpsgraph]'
      MPS_SDPA_LOG_FALLBACKS=warn -> logged at WARNING level (noisier)
    """
    # Increment the stock-fallback counter so api.get_call_stats() reflects the
    # true number of calls that ended up on stock via the mpsgraph path.
    try:
        from ..api import _call_counts
        _call_counts["stock_fallback"] += 1
    except Exception:
        pass
    mode = _os.environ.get("MPS_SDPA_LOG_FALLBACKS", "").lower()
    msg = f"[mps_sdpa.mpsgraph] falling back to stock: {reason}"
    if mode in ("1", "true", "info"):
        _logger.info(msg)
    elif mode in ("warn", "warning"):
        _logger.warning(msg)
    else:
        _logger.debug(msg)


def _fallback_stock(q, k, v, *, attn_mask, dropout_p, is_causal, scale, reason: str):
    """Internal stock fallback for cases mpsgraph can't or shouldn't handle.
    reason is a short human-readable description of why; logged via _log_fallback.
    """
    import torch.nn.functional as F
    _log_fallback(reason)
    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask, dropout_p=dropout_p,
        is_causal=is_causal, scale=scale,
    )


class _MpsGraphSDPAFunction(torch.autograd.Function):
    """Custom autograd function: MPSGraph SDPA forward, manual pure-torch backward.

    Forward runs Apple's dedicated scaledDotProductAttention MPSGraph op (fast).
    Backward recomputes the softmax (cheap) and does 4 matmuls in torch (MPS).
    Gradients are correct to fp16/bf16/fp32 tolerance.

    Why manual backward instead of letting torch handle it: torch.autograd would need
    a differentiable forward. Our forward path goes through MPSGraph which is outside
    torch's autograd. So we wrap as a Function and implement backward explicitly.

    Why recompute softmax instead of saving attn: the attn matrix is [B,H,Lq,Lkv]
    which for S3 (1,8,4096,4096,bf16) = 256 MB per layer. Recomputing from saved QK
    at backward time is faster than fetching a 256 MB save from memory.
    """

    @staticmethod
    def forward(ctx, q, k, v, mask_tensor, is_causal, scale, dropout_mask):
        ctx.save_for_backward(q, k, v)
        ctx.mask = mask_tensor
        ctx.dropout_mask = dropout_mask
        ctx.is_causal = is_causal
        ctx.scale = scale
        with torch.no_grad():
            return _mpsgraph_forward_inner(q, k, v, mask_tensor, is_causal, scale, dropout_mask)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        """Backward runs MPSGraph-native bwd graph. Not differentiable itself —
        @once_differentiable makes torch raise a clear error if a caller asks
        for second-order gradients via create_graph=True."""
        q, k, v = ctx.saved_tensors
        mask = ctx.mask
        dropout_mask = ctx.dropout_mask
        scale = ctx.scale
        D = q.shape[-1]

        if scale is not None and abs(scale - D ** -0.5) > 1e-9:
            s_ratio = scale / (D ** -0.5)
            q_scaled = q * s_ratio
            dQ, dK, dV = _mpsgraph_backward_inner(q_scaled, k, v, grad_out, mask, dropout_mask)
            dQ = dQ * s_ratio
            return (dQ, dK, dV, None, None, None, None)

        dQ, dK, dV = _mpsgraph_backward_inner(q, k, v, grad_out, mask, dropout_mask)
        return (dQ, dK, dV, None, None, None, None)


def mpsgraph_sdpa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    *, attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Public entry point. Dispatches among:
    - Stock fallback (short seqs, unsupported dtypes/masks, dropout, non-MPS)
    - Forward-only mpsgraph path (requires_grad=False)
    - Autograd-wrapped mpsgraph path (requires_grad=True)
    """
    if not _AVAILABLE:
        return _fallback_stock(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                               is_causal=is_causal, scale=scale,
                               reason="backend unavailable")
    if q.device.type != "mps":
        return _fallback_stock(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                               is_causal=is_causal, scale=scale,
                               reason=f"non-MPS device ({q.device.type})")

    B, H, Lq, D = q.shape
    _, _, Lkv, _ = k.shape
    if q.dtype not in _MPS_DTYPE:
        return _fallback_stock(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                               is_causal=is_causal, scale=scale,
                               reason=f"unsupported dtype {q.dtype}")

    # Short-seq cases: copy overhead dominates compute savings; fall back.
    elem_size = q.element_size()
    attn_bytes = Lq * Lkv * elem_size
    dkey = _calibrate.dtype_key(q.dtype)
    thresholds = _calibrate.get_thresholds()
    fused_min = thresholds["fused_min_bytes"].get(dkey) if dkey else None
    if fused_min is None or attn_bytes < fused_min:
        return _fallback_stock(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                               is_causal=is_causal, scale=scale,
                               reason=f"short-seq ({attn_bytes} < {fused_min} bytes)")

    # Handle masks — fall back for combinations we don't support. We compute the
    # additive mask tensor here (used by both the autograd and forward paths).
    # Accepts broadcast shapes (B_m, H_m, Lq, Lkv) where B_m ∈ {1,B}, H_m ∈ {1,H}.
    mask_tensor: Optional[torch.Tensor] = None
    if is_causal and attn_mask is not None:
        return _fallback_stock(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                               is_causal=is_causal, scale=scale,
                               reason="is_causal + explicit attn_mask unsupported")
    if is_causal:
        mask_tensor = _build_causal_mask(Lq, Lkv, q.dtype)
    elif attn_mask is not None:
        m = attn_mask
        while m.dim() < 4:
            m = m.unsqueeze(0)
        # Reject shapes we can't express as [B_m, H_m, Lq, Lkv] placeholders.
        if m.shape[0] not in {1, B} or m.shape[1] not in {1, H}:
            return _fallback_stock(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                                   is_causal=is_causal, scale=scale,
                                   reason=f"mask shape {tuple(m.shape)} not broadcast-compatible")
        if m.shape[-2] != Lq or m.shape[-1] != Lkv:
            return _fallback_stock(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                                   is_causal=is_causal, scale=scale,
                                   reason=f"mask spatial shape {tuple(m.shape[-2:])} != (Lq, Lkv)")
        if m.dtype == torch.bool:
            mask_tensor = _build_additive_mask(m, q.dtype)
        else:
            mask_tensor = m.to(dtype=q.dtype).contiguous()

    _init_runtime()

    # Dropout: uses an UNFUSED MPSGraph (Apple's fused SDPA op doesn't expose attn
    # pre-softmax so we can't post-multiply dropout). Unfused graph materializes
    # multiple Lq*Lkv intermediates which (a) OOM on very-long seqs, (b) actually
    # LOSE to stock at short seqs because stock has a well-tuned dropout kernel.
    # Empirical window where unfused mpsg dropout wins: ~16-64 MB attn matrix.
    # (Lq=Lkv=4096 bf16 = 32MB: 1.21x win. Lq=Lkv=2048 bf16 = 8MB: 0.93x loss.
    #  Lq=Lkv=8192 bf16 = 128MB: OOM.) Outside window, fall back to stock.
    dropout_mask: Optional[torch.Tensor] = None
    if dropout_p > 0:
        attn_bytes_drop = Lq * Lkv * elem_size
        drop_min = thresholds.get("dropout_min_bytes", 16 * 1024**2)
        drop_max = thresholds.get("dropout_max_bytes", 64 * 1024**2)
        if attn_bytes_drop < drop_min or attn_bytes_drop > drop_max:
            return _fallback_stock(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                                   is_causal=is_causal, scale=scale,
                                   reason=f"dropout attn_bytes {attn_bytes_drop} "
                                          f"outside window [{drop_min}, {drop_max}]")
        keep = 1.0 - dropout_p
        dropout_mask = (torch.rand(B, H, Lq, Lkv, device="mps", dtype=q.dtype) < keep).to(q.dtype) / keep

    # Autograd path — this is the true "mpsgraph succeeded" counter increment
    try:
        try:
            from ..api import _call_counts
            _call_counts["mpsgraph"] += 1
        except Exception:
            pass
        if q.requires_grad or k.requires_grad or v.requires_grad:
            return _MpsGraphSDPAFunction.apply(q, k, v, mask_tensor, is_causal, scale, dropout_mask)
        return _mpsgraph_forward_inner(q, k, v, mask_tensor, is_causal, scale, dropout_mask)
    except RuntimeError as e:
        # Graceful OOM fallback (torch raises RuntimeError on MPS OOM).
        if "out of memory" in str(e).lower() or "mps allocated" in str(e).lower():
            return _fallback_stock(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                                   is_causal=is_causal, scale=scale,
                                   reason=f"OOM recovery: {type(e).__name__}")
        raise


def _mpsgraph_forward_inner(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    mask_tensor: Optional[torch.Tensor],
    is_causal: bool, scale: Optional[float],
    dropout_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    B, H, Lq, D = q.shape
    _, _, Lkv, _ = k.shape

    default_scale = D ** -0.5
    q_eff = q
    if scale is not None and abs(scale - default_scale) > 1e-9:
        q_eff = q * (scale / default_scale)

    mask_kind = "mask_present" if mask_tensor is not None else "none"
    dropout = dropout_mask is not None
    mask_shape = (mask_tensor.shape[0], mask_tensor.shape[1]) if mask_tensor is not None else None

    dtype_val = _MPS_DTYPE[q.dtype]
    g, q_ph, k_ph, v_ph, mask_ph, drop_ph, out_ph = _build_graph(
        B, H, Lq, Lkv, D, dtype_val, mask_kind=mask_kind,
        mask_shape=mask_shape, dropout=dropout,
    )

    q_td, q_buf = _copy_tensor_to_tensor_data(q_eff.contiguous(), dtype_val)
    k_td, k_buf = _copy_tensor_to_tensor_data(k.contiguous(), dtype_val)
    v_td, v_buf = _copy_tensor_to_tensor_data(v.contiguous(), dtype_val)

    feeds = {q_ph: q_td, k_ph: k_td, v_ph: v_td}
    if mask_ph is not None:
        m_td, m_buf = _copy_tensor_to_tensor_data(mask_tensor, dtype_val)
        feeds[mask_ph] = m_td
    if drop_ph is not None:
        d_td, d_buf = _copy_tensor_to_tensor_data(dropout_mask.contiguous(), dtype_val)
        feeds[drop_ph] = d_td

    out_nbytes = B * H * Lq * D * torch.zeros(1, dtype=q.dtype).element_size()
    out_buf = _device.newBufferWithLength_options_(out_nbytes, _MTL_STORAGE_SHARED)
    out_td = MPSGraphTensorData.alloc().initWithMTLBuffer_shape_dataType_(
        out_buf, [B, H, Lq, D], dtype_val
    )
    results = {out_ph: out_td}

    g.runWithMTLCommandQueue_feeds_targetOperations_resultsDictionary_(
        _command_queue, feeds, None, results,
    )

    out = torch.empty((B, H, Lq, D), dtype=q.dtype, device="mps")
    _copy_tensor_data_to_torch(out_buf, out)
    torch.mps.synchronize()
    return out


_AVAILABLE, _UNAVAILABLE_REASON = _check_mpsgraph_sdpa_available()
register_backend(
    "mpsgraph", mpsgraph_sdpa, available=_AVAILABLE, reason=_UNAVAILABLE_REASON,
)
