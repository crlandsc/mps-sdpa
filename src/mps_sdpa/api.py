"""Public API: drop-in replacement for F.scaled_dot_product_attention on MPS."""
from __future__ import annotations
import warnings
from typing import Optional
import torch

from . import backends as _backends

_default_backend = "auto"
_gqa_warning_emitted = False
_banner_printed = False

# Per-process dispatch counters. Increment inside sdpa_opt based on which
# path was actually taken. Useful for confirming opt is live during training.
from collections import Counter as _Counter
_call_counts: "_Counter[str]" = _Counter()


def get_call_stats() -> dict:
    """Snapshot of dispatch counters since process start.

    Returns e.g. {"mpsgraph_zc": 1234, "mpsgraph": 0, "stock_fallback": 45, ...}.
    'stock_fallback' means opt was requested but routed to stock (e.g. short seq,
    GQA, unsupported dtype). 'stock_explicit' means caller passed backend="stock".
    """
    return dict(_call_counts)


def reset_call_stats() -> None:
    _call_counts.clear()


def print_call_stats(tag: str = "") -> None:
    """Print a one-line summary of dispatch counts."""
    c = _call_counts
    total = sum(c.values())
    if total == 0:
        print(f"[mps-sdpa stats{' '+tag if tag else ''}] no calls yet")
        return
    parts = [f"{name}={cnt} ({cnt*100//total}%)" for name, cnt in c.most_common()]
    print(f"[mps-sdpa stats{' '+tag if tag else ''}] total={total}  " + "  ".join(parts),
          flush=True)


def backend_status(backend: str = "auto", device: str = "mps") -> dict:
    """Return a structured report of what backend will handle calls.

    Useful for runtime sanity checking that the opt path is actually active
    (vs silently falling back to stock).

    Keys:
      - requested_backend: what the user asked for (e.g., "auto")
      - picked: the backend name that will actually be used
      - active: True if the picked backend is a real acceleration, False if stock
      - available: list of currently-available backend names
      - unavailable: dict of {name: reason} for backends that failed to register
      - torch_version / mps_sdpa_version: for reproducibility
    """
    from . import __version__ as _version
    available = _backends.available_backends()
    known = list(_backends._REGISTRY.keys())
    unavailable = {
        name: (_backends.backend_reason(name) or "no reason recorded")
        for name in known
        if name not in available
    }
    if backend == "auto":
        if device == "mps":
            picked = _pick_auto(torch.zeros(1, device="mps") if torch.backends.mps.is_available()
                                else torch.zeros(1))
        else:
            picked = "stock"
    else:
        picked = backend
    active = picked in {"mpsgraph_zc", "mpsgraph"}  # metal_proto is slower-than-stock
    return {
        "requested_backend": backend,
        "picked": picked,
        "active": active,
        "available": available,
        "unavailable": unavailable,
        "torch_version": torch.__version__,
        "mps_sdpa_version": _version,
    }


def print_backend_banner(
    backend: str = "auto", device: str = "mps", tag: str = "", once: bool = True,
) -> None:
    """Print a one-glance preflight banner showing whether mps-sdpa is active.

    Prints once per process by default (set once=False to force re-print).
    Cheap — no forward pass, just inspects the registry. Intended to be called
    at model-construction time when the opt flag is enabled.
    """
    global _banner_printed
    if once and _banner_printed:
        return
    _banner_printed = True

    status = backend_status(backend=backend, device=device)
    bar = "=" * 68
    picked = status["picked"]
    backend_desc = {
        "mpsgraph_zc": "zero-copy C++ extension (fastest)",
        "mpsgraph": "pyobjc bridge (fallback)",
        "stock": "PyTorch F.scaled_dot_product_attention",
        "metal_proto": "naive Metal probe kernel (slow — not for production)",
    }.get(picked, picked)

    lines = [bar]
    lines.append(f"  mps-sdpa preflight" + (f"  [{tag}]" if tag else ""))
    lines.append(f"  torch: {status['torch_version']}   "
                 f"mps-sdpa: {status['mps_sdpa_version']}")
    lines.append(f"  requested backend: {status['requested_backend']!r}  ->  "
                 f"picked: {picked}  ({backend_desc})")
    if status["unavailable"]:
        lines.append(f"  unavailable backends:")
        for name, reason in status["unavailable"].items():
            short = reason.split(":")[0] if ":" in reason else reason
            lines.append(f"    - {name}: {short}")
    if status["active"]:
        lines.append(f"  STATUS: ACTIVE  [OK]  (calls route through mps-sdpa)")
    else:
        lines.append(f"  STATUS: INACTIVE  [!]  (falling back to stock torch SDPA)")
    lines.append(bar)
    print("\n".join(lines), flush=True)


def set_default_backend(name: str) -> None:
    """Process-global default for ``backend="auto"`` fallback."""
    global _default_backend
    _default_backend = name


def available_backends() -> list[str]:
    return _backends.available_backends()


def _pick_auto(q: torch.Tensor) -> str:
    if q.device.type != "mps":
        return "stock"
    # Preference order: zero-copy C++ ext > pyobjc-copy > experimental Metal
    # prototypes > stock. mpsgraph_zc transparently falls back to mpsgraph for
    # features it doesn't yet handle (autograd, dropout).
    pref = ["mpsgraph_zc", "mpsgraph", "metal_op", "metal_proto", "stock"]
    for name in pref:
        if name in _backends.available_backends():
            return name
    return "stock"


def _emit_gqa_warning_once(Hq: int, Hkv: int) -> None:
    global _gqa_warning_emitted
    if _gqa_warning_emitted:
        return
    _gqa_warning_emitted = True
    warnings.warn(
        f"[mps_sdpa] GQA detected (Hq={Hq}, Hkv={Hkv}); falling back to stock "
        f"F.scaled_dot_product_attention. The mpsgraph backend does not accelerate "
        f"grouped/multi-query attention. This warning is emitted once per process.",
        RuntimeWarning,
        stacklevel=3,
    )


def sdpa_opt(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    backend: str = "auto",
) -> torch.Tensor:
    """Drop-in replacement for ``F.scaled_dot_product_attention`` on MPS.

    GQA (Hq != Hkv) routes through stock F.scaled_dot_product_attention with a
    one-time warning — the mpsgraph backend is MHA-only.

    Robustness: if the selected backend raises because a feature isn't supported
    (e.g., some MPSGraph wrappers may not accept additive float masks), the dispatch
    falls back to stock rather than propagating the error. This keeps the behavior
    safe by default; pass ``backend="stock"`` explicitly to disable fallback.
    """
    # Coerce additive float masks to query dtype if they differ (bool masks
    # stay bool; downstream handlers convert to additive in the correct dtype).
    # Without this, PyTorch's stock SDPA raises on a dtype mismatch, breaking
    # drop-in replacement for code that uses fp32 masks with bf16 models.
    if (
        attn_mask is not None
        and attn_mask.dtype != torch.bool
        and attn_mask.is_floating_point()
        and attn_mask.dtype != query.dtype
    ):
        attn_mask = attn_mask.to(dtype=query.dtype)

    # is_causal=True combined with an explicit attn_mask: PyTorch's MPS
    # F.scaled_dot_product_attention crashes the process here with
    # NSInvalidArgumentException (torch issue, not ours). Combine the
    # two ourselves before any backend dispatch, then clear is_causal.
    # Logical AND for bool masks; additive (-inf where blocked) for float.
    if is_causal and attn_mask is not None:
        Lq = query.shape[-2]
        Lkv = key.shape[-2]
        causal = torch.ones(Lq, Lkv, dtype=torch.bool, device=query.device).tril()
        if attn_mask.dtype == torch.bool:
            # Broadcast-friendly AND. attn_mask may be [..., Lq, Lkv] of any
            # leading shape; causal is [Lq, Lkv] and broadcasts cleanly.
            attn_mask = attn_mask & causal
        else:
            # Additive float mask: -inf where the causal mask blocks.
            additive_causal = torch.zeros_like(attn_mask)
            additive_causal.masked_fill_(~causal, float("-inf"))
            attn_mask = attn_mask + additive_causal
        is_causal = False

    Hq, Hkv = query.shape[-3], key.shape[-3]
    if Hq != Hkv:
        _call_counts["gqa_fallback"] += 1
        _emit_gqa_warning_once(Hq, Hkv)
        # GQA: expand K/V to Hq heads via repeat_interleave, then route to stock.
        # This is the standard GQA materialization — trades memory for simplicity.
        # (torch >= 2.5 supports enable_gqa=True natively but we target torch 2.11+.)
        if Hq % Hkv != 0:
            raise ValueError(
                f"GQA requires Hq ({Hq}) to be divisible by Hkv ({Hkv})"
            )
        repeat = Hq // Hkv
        key = key.repeat_interleave(repeat, dim=-3)
        value = value.repeat_interleave(repeat, dim=-3)
        stock_fn = _backends.get_backend("stock")
        return stock_fn(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                        is_causal=is_causal, scale=scale)

    # Pick backend name
    if backend == "auto":
        name = _pick_auto(query)
    else:
        name = backend

    # Validate backend exists (raises KeyError if not found)
    fn = _backends.get_backend(name)

    # NOTE: per-backend counters are incremented INSIDE each backend's dispatch,
    # not here, so they reflect the true final path (the mpsgraph_zc backend
    # may internally fall back to pyobjc/stock for short-seq / dropout-window
    # edge cases). See mpsgraph_zc.py, mpsgraph.py, and _log_fallback().

    # Non-MPS devices always use stock
    if query.device.type != "mps":
        _call_counts["stock"] += 1
        stock_fn = _backends.get_backend("stock")
        return stock_fn(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                        is_causal=is_causal, scale=scale)

    # MPS device: try selected backend, fallback to stock if NotImplementedError
    try:
        return fn(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                  is_causal=is_causal, scale=scale)
    except NotImplementedError:
        if name == "stock":
            raise
        stock_fn = _backends.get_backend("stock")
        return stock_fn(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                        is_causal=is_causal, scale=scale)
