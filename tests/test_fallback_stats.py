"""Per-reason fallback counter API."""
from __future__ import annotations

import pytest
import torch

import mps_sdpa
from mps_sdpa import sdpa_opt
from mps_sdpa.backends import mpsgraph_zc as _zc


def _can_run() -> bool:
    try:
        return _zc._AVAILABLE and torch.backends.mps.is_available()
    except Exception:
        return False


@pytest.mark.skipif(not _can_run(), reason="requires zc + MPS")
def test_short_seq_increments_counter():
    """A call below the calibrated threshold should bucket as 'short-seq'."""
    mps_sdpa.reset_fallback_stats()
    # Tiny shape — guaranteed below any reasonable threshold
    q = torch.randn(1, 2, 16, 32, dtype=torch.bfloat16, device="mps")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    _ = sdpa_opt(q, k, v)
    stats = mps_sdpa.get_fallback_stats()
    assert "short-seq" in stats, f"expected 'short-seq' bucket, got {stats}"
    assert stats["short-seq"] >= 1


def test_reset_clears_counter():
    from mps_sdpa.api import _fallback_counters
    _fallback_counters["test"] = 5
    mps_sdpa.reset_fallback_stats()
    assert mps_sdpa.get_fallback_stats() == {}


def test_print_stats_with_no_calls(capsys):
    mps_sdpa.reset_fallback_stats()
    mps_sdpa.print_fallback_stats()
    out = capsys.readouterr().out
    assert "no fallbacks recorded" in out


def test_print_stats_with_calls(capsys):
    from mps_sdpa.api import _fallback_counters
    mps_sdpa.reset_fallback_stats()
    _fallback_counters["short-seq"] = 50
    _fallback_counters["dropout-window"] = 25
    mps_sdpa.print_fallback_stats(tag="run-X")
    out = capsys.readouterr().out
    assert "total=75" in out
    assert "short-seq=50" in out
    assert "dropout-window=25" in out
    assert "run-X" in out
