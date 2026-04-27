"""Environment management: W&B kill-switch, MPS env knobs, preflight."""
from __future__ import annotations
import os
import sys
import platform
import contextlib
from typing import Iterator, Optional


def force_wandb_offline() -> None:
    """Force W&B offline mode. Called at package import time."""
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_SILENT"] = "true"


def assert_wandb_not_imported() -> None:
    """Raise if wandb has been imported in this interpreter."""
    mod = sys.modules.get("wandb")
    if mod is not None:
        raise RuntimeError(
            "wandb was imported; mps_sdpa project forbids any W&B usage. "
            "Remove the import or set WANDB_MODE=offline and restart."
        )


@contextlib.contextmanager
def mps_env(
    fast_math: Optional[bool] = None,
    prefer_metal: Optional[bool] = None,
    enable_fallback: Optional[bool] = None,
) -> Iterator[None]:
    """Set PyTorch MPS env knobs for the duration of a block.

    Preserves prior values and restores them on exit.
    """
    keys = {
        "PYTORCH_MPS_FAST_MATH": _bool_env(fast_math),
        "PYTORCH_MPS_PREFER_METAL": _bool_env(prefer_metal),
        "PYTORCH_ENABLE_MPS_FALLBACK": _bool_env(enable_fallback),
    }
    prior: dict[str, Optional[str]] = {}
    try:
        for k, v in keys.items():
            if v is None:
                continue
            prior[k] = os.environ.get(k)
            os.environ[k] = v
        yield
    finally:
        for k, v in prior.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _bool_env(v: Optional[bool]) -> Optional[str]:
    if v is None:
        return None
    return "1" if v else "0"


def preflight_check() -> None:
    """Assert macOS >= 14.0 on Apple silicon. Raises on mismatch."""
    if platform.system() != "Darwin":
        raise RuntimeError(f"macOS required; got {platform.system()}")
    if platform.machine() != "arm64":
        raise RuntimeError(f"Apple silicon required; got arch {platform.machine()}")
    mac_ver = platform.mac_ver()[0]
    if not mac_ver:
        raise RuntimeError("could not detect macOS version")
    major = int(mac_ver.split(".")[0])
    if major < 14:
        raise RuntimeError(f"macOS >= 14.0 required; got {mac_ver}")
