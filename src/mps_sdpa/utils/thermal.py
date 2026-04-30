"""Thermal / system-load reader. Uses pyobjc for thermal; psutil for the rest."""
from __future__ import annotations

import os

import psutil

_THERMAL_STATE_NAMES = {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}


def thermal_state() -> str:
    """Read macOS thermal state via NSProcessInfo; returns 'unknown' if unavailable."""
    try:
        from Foundation import NSProcessInfo
        state = int(NSProcessInfo.processInfo().thermalState())
        return _THERMAL_STATE_NAMES.get(state, "unknown")
    except Exception:
        return "unknown"


def is_nominal(state: str) -> bool:
    return state == "nominal"


def snapshot() -> dict:
    """Return a snapshot of thermal, load, and RAM status."""
    load1, load5, load15 = os.getloadavg()
    vm = psutil.virtual_memory()
    return {
        "thermal": thermal_state(),
        "load1": float(load1),
        "load5": float(load5),
        "load15": float(load15),
        "free_ram_gb": float(vm.available) / 1024**3,
        "total_ram_gb": float(vm.total) / 1024**3,
    }
