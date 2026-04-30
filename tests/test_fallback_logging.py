"""Unified fallback logging.

All fallbacks go through _log_fallback with a
consistent `[mps_sdpa.mpsgraph] falling back to stock: <reason>` format.
Default log level is DEBUG (silent). Opt in via env var.
"""
from __future__ import annotations

import logging
import os

import pytest
import torch

from mps_sdpa.backends import mpsgraph as _mg


def _can_run() -> bool:
    try:
        return _mg._AVAILABLE and torch.backends.mps.is_available()
    except Exception:
        return False


def _capture(log_env_val: str | None = None):
    """Run mpsgraph_sdpa directly on MPS tensors with a short seq to trigger
    the short-seq fallback path, and return captured log records."""
    if log_env_val is None:
        os.environ.pop("MPS_SDPA_LOG_FALLBACKS", None)
    else:
        os.environ["MPS_SDPA_LOG_FALLBACKS"] = log_env_val
    # Use MPS tensors + short seq so the short-seq fallback path fires.
    q = torch.randn(1, 4, 128, 32, dtype=torch.bfloat16, device="mps")
    k = torch.randn(1, 4, 128, 32, dtype=torch.bfloat16, device="mps")
    v = torch.randn(1, 4, 128, 32, dtype=torch.bfloat16, device="mps")
    logger = logging.getLogger("mps_sdpa.mpsgraph")
    records: list[logging.LogRecord] = []

    class _H(logging.Handler):
        def emit(self, record):
            records.append(record)

    h = _H()
    logger.addHandler(h)
    prev_level = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        # Call the mpsgraph backend directly (bypasses api.py non-MPS short-circuit)
        _mg.mpsgraph_sdpa(q, k, v)
    finally:
        logger.removeHandler(h)
        logger.setLevel(prev_level)
    return records


@pytest.mark.skipif(not _can_run(), reason="requires MPS device")
def test_fallback_message_format_prefixed():
    """Every fallback log message has the unified prefix."""
    records = _capture(log_env_val="1")  # opt-in at INFO level
    assert len(records) >= 1, "expected at least one fallback log"
    for r in records:
        msg = r.getMessage()
        assert msg.startswith("[mps_sdpa.mpsgraph] falling back to stock:"), msg


@pytest.mark.skipif(not _can_run(), reason="requires MPS device")
def test_fallback_info_level_when_env_set():
    records = _capture(log_env_val="1")
    info_records = [r for r in records if r.levelno == logging.INFO]
    assert len(info_records) >= 1


@pytest.mark.skipif(not _can_run(), reason="requires MPS device")
def test_fallback_warning_level_when_env_set_to_warn():
    records = _capture(log_env_val="warn")
    warning_records = [r for r in records if r.levelno == logging.WARNING]
    assert len(warning_records) >= 1


@pytest.mark.skipif(not _can_run(), reason="requires MPS device")
def test_fallback_silent_by_default():
    """Without the env var, fallbacks still log at DEBUG but no INFO/WARNING."""
    records = _capture(log_env_val=None)
    loud = [r for r in records if r.levelno >= logging.INFO]
    assert not loud, (
        f"expected silent fallback by default; got {len(loud)} INFO+ records"
    )
