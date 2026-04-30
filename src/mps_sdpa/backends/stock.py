"""Stock backend: delegates to torch.nn.functional.scaled_dot_product_attention."""
from __future__ import annotations

import torch.nn.functional as F

from . import register_backend


def stock_sdpa(q, k, v, *, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


register_backend("stock", stock_sdpa, available=True)
