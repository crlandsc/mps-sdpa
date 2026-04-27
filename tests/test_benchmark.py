import pytest
import time
import torch
from mps_sdpa.harness import benchmark as bm


def _tiny_fn():
    q = torch.randn(1, 2, 8, 16)
    k = torch.randn(1, 2, 8, 16)
    v = torch.randn(1, 2, 8, 16)
    def run():
        return (q @ k.transpose(-1, -2)).softmax(dim=-1) @ v
    return run


def test_time_region_returns_stats():
    stats = bm.time_region(_tiny_fn(), warmup=2, min_iters=5, min_seconds=0.0, device="cpu")
    assert "median" in stats
    assert "p10" in stats
    assert "p90" in stats
    assert stats["n"] >= 5


def test_time_region_respects_min_seconds():
    t0 = time.time()
    bm.time_region(_tiny_fn(), warmup=1, min_iters=2, min_seconds=0.1, device="cpu")
    assert time.time() - t0 >= 0.1


def test_paired_ab_ratio_near_one_for_identical_fns():
    fn = _tiny_fn()
    res = bm.paired_ab(fn, fn, n_pairs=3, warmup=2, min_iters=3, min_seconds=0.0, device="cpu")
    assert 0.5 < res["paired_geomean_ratio"] < 2.0
