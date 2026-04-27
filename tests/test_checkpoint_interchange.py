import pytest
import torch
from mps_sdpa.training_check import checkpoint_interchange as ci


def test_interchange_stock_to_opt_cpu_passes():
    verdict = ci.run_interchange(device="cpu", steps=5, seed=0)
    assert verdict["opt_to_stock"]["passed"] is True
    assert verdict["stock_to_opt"]["passed"] is True


def test_interchange_reports_max_err():
    verdict = ci.run_interchange(device="cpu", steps=5, seed=0)
    assert "max_abs_err" in verdict["opt_to_stock"]
    assert "max_abs_err" in verdict["stock_to_opt"]
