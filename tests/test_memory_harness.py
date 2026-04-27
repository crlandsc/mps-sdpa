import pytest
import torch
from mps_sdpa.harness import memory


def test_snapshot_has_required_fields_cpu():
    snap = memory.mps_snapshot()
    assert set(snap).issuperset({"current", "driver", "recommended_max"})


def test_measure_region_records_before_and_after():
    def run():
        x = torch.zeros(1024, 1024)
        return x.sum()
    rec = memory.measure_region(run, device="cpu")
    assert "before" in rec
    assert "after" in rec
    assert "delta_current" in rec
    assert "delta_driver" in rec
