"""Reserved name for a future fully-custom torch op backed by Metal kernels.

Currently a stub. The zero-copy C++ extension (`mpsgraph_zc`) already
provides the speedup envelope this was originally intended to probe, so
this name is kept reserved but unimplemented. Registered as unavailable
with a reason string so it shows up cleanly in the backend banner without
being auto-selected.
"""
from __future__ import annotations

from . import register_backend


def metal_op_sdpa(q, k, v, *, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    raise NotImplementedError(
        "metal_op backend not implemented — use mpsgraph_zc (the default)."
    )


register_backend(
    "metal_op", metal_op_sdpa, available=False,
    reason="reserved name; not implemented — use mpsgraph_zc instead",
)
