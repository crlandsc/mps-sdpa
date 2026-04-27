"""Auto-calibration for mpsgraph fused-path shape thresholds.

The hardcoded defaults (4 MB fused, 16-64 MB dropout window) were tuned on
M4 + torch 2.13 nightly + macOS 26. On other Apple silicon (M1/M2/M3/Max) the
compute/bandwidth trade-off differs and the crossover point shifts. This module
measures the stock-vs-mpsgraph crossover at import time and caches the result
keyed by (chip, os, torch, schema) so it runs once per (machine, stack).

Env vars:
  MPS_SDPA_SKIP_CALIBRATION=1   Use hardcoded defaults, don't touch cache/bench.
  MPS_SDPA_FORCE_CALIBRATE=1    Ignore cache, re-calibrate and overwrite.
"""
from __future__ import annotations
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Optional

import torch

_CACHE_DIR = Path.home() / ".cache" / "mps_sdpa"
_CACHE_FILE = _CACHE_DIR / "thresholds.json"
_CACHE_SCHEMA_VERSION = 1

# Conservative defaults (M4-calibrated). Used when calibration is skipped,
# MPS unavailable, cache miss + bench fails, or cross-machine.
_DEFAULT_THRESHOLDS: dict = {
    "fused_min_bytes": {
        "bf16": 4 * 1024**2,
        "fp16": 4 * 1024**2,
        "fp32": 8 * 1024**2,  # fp32 has 2x copy cost; crossover later
    },
    "dropout_min_bytes": 16 * 1024**2,
    "dropout_max_bytes": 64 * 1024**2,
    "calibrated": False,
}


def _fingerprint() -> dict:
    """Return (chip, os, torch, schema) identity for cache keying."""
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stderr=subprocess.DEVNULL, timeout=2,
        ).decode().strip()
    except Exception:
        chip = platform.processor() or "unknown"
    return {
        "chip": chip,
        "os": platform.mac_ver()[0] or "unknown",
        "torch": torch.__version__,
        "schema": _CACHE_SCHEMA_VERSION,
    }


def _load_cache() -> Optional[dict]:
    if not _CACHE_FILE.exists():
        return None
    try:
        data = json.loads(_CACHE_FILE.read_text())
    except Exception:
        return None
    if data.get("fingerprint") != _fingerprint():
        return None
    t = data.get("thresholds")
    if not isinstance(t, dict):
        return None
    fmb = t.get("fused_min_bytes")
    if not isinstance(fmb, dict):
        return None
    if not all(k in fmb for k in ("bf16", "fp16", "fp32")):
        return None
    return t


def _save_cache(thresholds: dict) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {"fingerprint": _fingerprint(), "thresholds": thresholds}
        _CACHE_FILE.write_text(json.dumps(payload, indent=2))
    except Exception:
        pass


def _bench_shape(L: int, dtype: torch.dtype, use_mpsg: bool) -> float:
    """Return min-of-N wall-clock seconds for a single SDPA call."""
    import torch.nn.functional as F
    from . import mpsgraph as _mg

    q = torch.randn(1, 8, L, 64, dtype=dtype, device="mps")
    k = torch.randn(1, 8, L, 64, dtype=dtype, device="mps")
    v = torch.randn(1, 8, L, 64, dtype=dtype, device="mps")
    if use_mpsg:
        _mg._init_runtime()
        fn = lambda: _mg._mpsgraph_forward_inner(q, k, v, None, False, None, None)  # noqa: E731
    else:
        fn = lambda: F.scaled_dot_product_attention(q, k, v)  # noqa: E731

    for _ in range(3):
        fn()
    torch.mps.synchronize()
    times = []
    for _ in range(8):
        t0 = time.perf_counter()
        fn()
        torch.mps.synchronize()
        times.append(time.perf_counter() - t0)
    return min(times)


def _calibrate_dtype(dtype: torch.dtype, elem_size: int) -> int:
    """Find smallest L in {256, 1024, 2048} where mpsgraph wins by >= 5%.

    Returns the bytes threshold (inclusive): L * L * elem_size. If no shape
    wins, returns a very large value so the mpsgraph path is effectively
    disabled for this dtype.
    """
    SAFETY_RATIO = 1.05  # need at least 5% win over stock to justify switching
    for L in (256, 1024, 2048):
        try:
            t_stock = _bench_shape(L, dtype, use_mpsg=False)
            t_mpsg = _bench_shape(L, dtype, use_mpsg=True)
        except Exception:
            # Any bench failure -> abort calibration for this dtype; caller falls
            # back to conservative default.
            raise
        if t_mpsg <= 0:
            continue
        ratio = t_stock / t_mpsg
        if ratio >= SAFETY_RATIO:
            return L * L * elem_size
    # Nothing we measured wins. Set an effectively-infinite threshold.
    return 2**40  # 1 TB


def _can_calibrate() -> bool:
    """True iff we can run benchmarks (MPS available + backend up)."""
    try:
        if not torch.backends.mps.is_available():
            return False
    except Exception:
        return False
    # mpsgraph import must have succeeded
    try:
        from . import mpsgraph as _mg
        if not _mg._AVAILABLE:
            return False
    except Exception:
        return False
    return True


def _calibrate() -> dict:
    """Run bench and return a fresh thresholds dict. Raises on bench failure."""
    elem_sizes = {
        torch.bfloat16: 2,
        torch.float16: 2,
        torch.float32: 4,
    }
    dtype_keys = {
        torch.bfloat16: "bf16",
        torch.float16: "fp16",
        torch.float32: "fp32",
    }
    out: dict = {
        "fused_min_bytes": {},
        "dropout_min_bytes": 16 * 1024**2,
        "dropout_max_bytes": 64 * 1024**2,
        "calibrated": True,
    }
    for dtype, elem in elem_sizes.items():
        out["fused_min_bytes"][dtype_keys[dtype]] = _calibrate_dtype(dtype, elem)
    return out


_cached_thresholds: Optional[dict] = None


def get_thresholds() -> dict:
    """Return thresholds dict. First call may run calibration (~0.5-1s).

    Cached in memory after first call; cached on disk keyed by (chip, os, torch).
    """
    global _cached_thresholds
    if _cached_thresholds is not None:
        return _cached_thresholds

    if os.environ.get("MPS_SDPA_SKIP_CALIBRATION") == "1":
        _cached_thresholds = _DEFAULT_THRESHOLDS
        return _cached_thresholds

    if os.environ.get("MPS_SDPA_FORCE_CALIBRATE") != "1":
        cached = _load_cache()
        if cached is not None:
            _cached_thresholds = cached
            return _cached_thresholds

    if not _can_calibrate():
        _cached_thresholds = _DEFAULT_THRESHOLDS
        return _cached_thresholds

    try:
        fresh = _calibrate()
    except Exception:
        _cached_thresholds = _DEFAULT_THRESHOLDS
        return _cached_thresholds

    _save_cache(fresh)
    _cached_thresholds = fresh
    return fresh


def dtype_key(dtype: torch.dtype) -> Optional[str]:
    """Map torch dtype to thresholds.fused_min_bytes key, or None if unsupported."""
    return {
        torch.bfloat16: "bf16",
        torch.float16: "fp16",
        torch.float32: "fp32",
    }.get(dtype)


def invalidate() -> None:
    """Drop in-memory + on-disk cache. For tests / recalibration."""
    global _cached_thresholds
    _cached_thresholds = None
    try:
        _CACHE_FILE.unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass
