import pytest
import torch
from mps_sdpa.harness import gradcheck as gc


def test_stock_passes_gradcheck_cpu():
    ok = gc.run_gradcheck(backend_name="stock", device="cpu")
    assert ok is True
