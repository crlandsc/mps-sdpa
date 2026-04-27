"""Thermal state reader tests."""
import pytest
from mps_sdpa.utils import thermal


def test_thermal_state_returns_known_value():
    s = thermal.thermal_state()
    assert s in {"nominal", "fair", "serious", "critical", "unknown"}


def test_is_nominal_matches_nominal():
    assert thermal.is_nominal("nominal") is True
    assert thermal.is_nominal("serious") is False
    assert thermal.is_nominal("critical") is False


def test_snapshot_has_required_fields():
    snap = thermal.snapshot()
    assert set(snap).issuperset({"thermal", "load1", "load5", "load15", "free_ram_gb"})
    assert isinstance(snap["thermal"], str)
    assert isinstance(snap["load1"], float)
    assert isinstance(snap["free_ram_gb"], float)
