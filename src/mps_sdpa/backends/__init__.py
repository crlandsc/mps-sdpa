"""Backend registry for mps_sdpa."""
from __future__ import annotations
from typing import Callable, Dict, Optional

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


from . import stock  # noqa: E402,F401
from . import mpsgraph  # noqa: E402,F401
from . import mpsgraph_zc  # noqa: E402,F401
from . import metal_proto  # noqa: E402,F401
from . import metal_op  # noqa: E402,F401
