"""Backend registry for mps_sdpa."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, Optional

if TYPE_CHECKING:
    import torch  # noqa: F401  (string forward-ref in BackendFn annotation)

BackendFn = Callable[..., "torch.Tensor"]

_REGISTRY: Dict[str, BackendFn] = {}
_AVAILABLE: Dict[str, bool] = {}
_REASONS: Dict[str, str] = {}


def register_backend(
    name: str,
    fn: BackendFn,
    *,
    available: bool,
    reason: Optional[str] = None,
) -> None:
    _REGISTRY[name] = fn
    _AVAILABLE[name] = bool(available)
    if reason:
        _REASONS[name] = reason
    elif name in _REASONS:
        del _REASONS[name]


def available_backends() -> list[str]:
    return sorted(n for n, ok in _AVAILABLE.items() if ok)


def backend_reason(name: str) -> Optional[str]:
    """Return unavailability reason for a backend, or None if available / unknown."""
    return _REASONS.get(name)


def get_backend(name: str) -> BackendFn:
    if name not in _REGISTRY:
        raise KeyError(f"unknown backend {name!r}; known: {sorted(_REGISTRY)}")
    if not _AVAILABLE.get(name, False):
        reason = _REASONS.get(name, "no reason recorded")
        raise RuntimeError(
            f"backend {name!r} is registered but not available on this system: {reason}"
        )
    return _REGISTRY[name]


# Backend modules import below register themselves into _REGISTRY at import time;
# they must come AFTER register_backend / get_backend definitions above.
from . import (  # noqa: E402  (modules self-register via functions defined above)
    metal_op,  # noqa: F401
    metal_proto,  # noqa: F401
    mpsgraph,  # noqa: F401
    mpsgraph_zc,  # noqa: F401
    stock,  # noqa: F401
)

# Register the torch.library custom_op surface for torch.compile compatibility.
# Available on torch >= 2.4 (we pin >= 2.11 so it's always present); the try/
# except is belt-and-suspenders for unforeseen import-time issues — if
# registration fails for any reason, sdpa_opt still works through the eager
# dispatch path, just without the torch.compile-friendly trace surface.
try:
    from . import torch_compile_op  # noqa: F401, E402
except Exception:
    pass
