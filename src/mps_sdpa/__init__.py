"""mps_sdpa: fast scaled-dot-product attention on Apple silicon MPS."""
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("mps-sdpa")
except _PackageNotFoundError:
    __version__ = "0.0.0+unknown"

# Optionally force W&B offline mode at import. Off by default — a library
# should never silently override its caller's W&B configuration. Set
# MPS_SDPA_FORCE_WANDB_OFFLINE=1 to opt in (useful for sandboxed test runs
# that must not contact wandb.ai).
import os as _os

if _os.environ.get("MPS_SDPA_FORCE_WANDB_OFFLINE") == "1":
    from .utils import env as _env
    _env.force_wandb_offline()

# Forward-import the public API. Wrapped in try/except so that partial / WIP
# imports of sibling subpackages don't blow up if `.api` isn't present yet.
try:
    from .api import (  # noqa: E402
        available_backends,
        backend_status,
        get_fallback_stats,
        print_backend_banner,
        print_fallback_stats,
        reset_fallback_stats,
        sdpa_opt,
        set_default_backend,
    )
    __all__ = [
        "sdpa_opt",
        "set_default_backend",
        "available_backends",
        "backend_status",
        "print_backend_banner",
        "get_fallback_stats",
        "print_fallback_stats",
        "reset_fallback_stats",
    ]
except ImportError:
    __all__ = []
