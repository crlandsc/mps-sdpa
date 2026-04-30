"""Naive custom Metal compute-shader SDPA — research/experimentation only.

This is a single-pass online-softmax SDPA kernel: one thread per
`(batch*head, q_idx)` output row. It is NOT FlashAttention-2 — no tiling,
no threadgroup shared memory, no simdgroup matrix ops. As such it is
~60× slower than the `mpsgraph_zc` zero-copy bridge to Apple's fused op
on M4 / bf16 / typical transformer shapes.

Why it ships: kept as a starting point for anyone investigating a full
custom Metal kernel. Correct (matches stock within bf16 tolerance) but
not auto-selected — `api._pick_auto()` skips it. To exercise:

    sdpa_opt(q, k, v, backend="metal_proto")

See `docs/design/custom-kernel-experiment.md` for the rationale.
"""
from __future__ import annotations

import torch

from . import register_backend

_SRC = r"""
#include <metal_stdlib>
using namespace metal;

kernel void sdpa_naive_bf16(
    device const bfloat* Q [[buffer(0)]],
    device const bfloat* K [[buffer(1)]],
    device const bfloat* V [[buffer(2)]],
    device bfloat* O [[buffer(3)]],
    constant uint4& params [[buffer(4)]],  // (BH, Lq, Lkv, D)
    uint2 gid [[thread_position_in_grid]]
) {
    const uint bh = gid.x, q_idx = gid.y;
    const uint BH = params[0], Lq = params[1], Lkv = params[2], D = params[3];
    if (bh >= BH || q_idx >= Lq) return;

    const device bfloat* q_row = Q + (bh * Lq + q_idx) * D;
    const device bfloat* k_base = K + bh * Lkv * D;
    const device bfloat* v_base = V + bh * Lkv * D;
    device bfloat* o_row = O + (bh * Lq + q_idx) * D;

    const float scale = 1.0f / sqrt((float)D);
    float q_reg[256];
    for (uint d = 0; d < D; ++d) q_reg[d] = (float)q_row[d];

    float running_max = -INFINITY, running_denom = 0.0f;
    float out_acc[256];
    for (uint d = 0; d < D; ++d) out_acc[d] = 0.0f;

    // Online softmax: single pass over Lkv.
    for (uint k_idx = 0; k_idx < Lkv; ++k_idx) {
        const device bfloat* k_row = k_base + k_idx * D;
        float s = 0.0f;
        for (uint d = 0; d < D; ++d) s += q_reg[d] * (float)k_row[d];
        s *= scale;
        float new_max = max(running_max, s);
        float alpha = exp(running_max - new_max);
        float beta = exp(s - new_max);
        running_denom = running_denom * alpha + beta;
        const device bfloat* v_row = v_base + k_idx * D;
        for (uint d = 0; d < D; ++d) {
            out_acc[d] = out_acc[d] * alpha + beta * (float)v_row[d];
        }
        running_max = new_max;
    }
    for (uint d = 0; d < D; ++d) o_row[d] = (bfloat)(out_acc[d] / running_denom);
}
"""

_lib = None


def _get_lib():
    global _lib
    if _lib is None:
        _lib = torch.mps.compile_shader(_SRC)
    return _lib


def metal_proto_sdpa(
    q, k, v, *, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
) -> torch.Tensor:
    # Only bf16 supported in this probe; everything else raises NotImplementedError
    # so api.py falls back to stock/mpsgraph.
    if q.dtype != torch.bfloat16:
        raise NotImplementedError("metal_proto: only bf16 supported in this probe")
    if attn_mask is not None or is_causal or dropout_p > 0:
        raise NotImplementedError("metal_proto: no mask/causal/dropout support yet")
    if q.shape[-1] > 256:
        raise NotImplementedError(f"metal_proto: D>256 ({q.shape[-1]}) exceeds register budget")

    if scale is not None:
        default_scale = q.shape[-1] ** -0.5
        if abs(scale - default_scale) > 1e-9:
            q = q * (scale / default_scale)

    B, H, Lq, D = q.shape
    _, _, Lkv, _ = k.shape
    out = torch.empty_like(q)
    q_flat = q.contiguous().view(B * H, Lq, D)
    k_flat = k.contiguous().view(B * H, Lkv, D)
    v_flat = v.contiguous().view(B * H, Lkv, D)
    o_flat = out.view(B * H, Lq, D)
    params = torch.tensor([B * H, Lq, Lkv, D], dtype=torch.uint32, device="mps")
    _get_lib().sdpa_naive_bf16(
        q_flat, k_flat, v_flat, o_flat, params,
        threads=(B * H, Lq, 1),
    )
    return out


_AVAILABLE = torch.backends.mps.is_available()
register_backend(
    "metal_proto", metal_proto_sdpa, available=_AVAILABLE,
    reason=None if _AVAILABLE else "MPS device unavailable",
)
