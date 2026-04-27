import pytest
import torch
from mps_sdpa.harness import cold_latency as cl


def test_measure_cold_returns_ms_positive():
    spec = {
        "backend": "stock", "device": "cpu",
        "B": 1, "H": 2, "Lq": 16, "Lkv": 16, "D": 32,
        "dtype": "fp32", "mask": "none", "is_causal": False,
    }
    ms = cl.measure_cold(spec, python_executable=None)
    assert ms > 0
    assert ms < 60_000


def test_measure_cold_captures_import_time():
    spec = {
        "backend": "stock", "device": "cpu",
        "B": 1, "H": 2, "Lq": 16, "Lkv": 16, "D": 32,
        "dtype": "fp32", "mask": "none", "is_causal": False,
    }
    ms = cl.measure_cold(spec)
    assert ms > 10, f"cold latency suspiciously low: {ms}ms (should include torch import)"
