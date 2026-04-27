"""Cold-latency measurement via fresh subprocess (eliminates intra-process warm-up cache)."""
from __future__ import annotations
import json
import subprocess
import sys
import time
from pathlib import Path

_RUNNER = r"""
import json, sys, time
spec = json.loads(sys.argv[1])
import torch
from mps_sdpa.backends import get_backend
from mps_sdpa.harness import tensor_factory
from mps_sdpa.suites.correctness_shapes import Case

case = Case(
    case_id="cold", B=spec["B"], H=spec["H"], Lq=spec["Lq"], Lkv=spec["Lkv"], D=spec["D"],
    dtype=spec["dtype"], mask=spec["mask"], contiguous=True, dropout_p=0.0,
)
q, k, v, mask = tensor_factory.build(case, device=spec["device"], seed=0)
fn = get_backend(spec["backend"])
if spec["device"] == "mps":
    torch.mps.synchronize()
t0 = time.perf_counter_ns()
fn(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=spec["is_causal"], scale=None)
if spec["device"] == "mps":
    torch.mps.synchronize()
t1 = time.perf_counter_ns()
print((t1 - t0) / 1e6)
"""


def measure_cold(spec: dict, python_executable: str | None = None) -> float:
    py = python_executable or sys.executable
    t0 = time.perf_counter()
    r = subprocess.run(
        [py, "-c", _RUNNER, json.dumps(spec)],
        capture_output=True, text=True, timeout=120,
    )
    t1 = time.perf_counter()
    if r.returncode != 0:
        raise RuntimeError(f"cold-latency subprocess failed: {r.stderr}")
    total_ms = (t1 - t0) * 1000
    return total_ms
