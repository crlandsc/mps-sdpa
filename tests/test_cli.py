import json
import pathlib
import subprocess
import sys
import pytest


def _run(*args, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", "mps_sdpa.cli", *args],
        capture_output=True, text=True, cwd=cwd,
    )


def test_cli_help_exits_zero():
    r = _run("--help")
    assert r.returncode == 0


def test_cli_correctness_stock_cpu_produces_json(tmp_path):
    r = _run("correctness", "--backend", "stock", "--device", "cpu",
             "--suite", "correctness", "--limit", "3",
             "--out", str(tmp_path / "corr.json"))
    assert r.returncode == 0, r.stderr
    data = json.loads((tmp_path / "corr.json").read_text())
    assert "n" in data


def test_cli_list_backends():
    r = _run("list-backends")
    assert r.returncode == 0
    assert "stock" in r.stdout


def test_cli_benchmark_stock_vs_stock_cpu_runs(tmp_path):
    out = tmp_path / "bench.csv"
    r = _run("benchmark", "--backend", "stock", "--baseline", "stock",
             "--device", "cpu", "--suite", "realistic", "--limit", "2",
             "--n-pairs", "2", "--warmup", "2", "--min-iters", "2",
             "--min-seconds", "0.0", "--out", str(out))
    assert r.returncode == 0, r.stderr
    assert out.exists()
    contents = out.read_text()
    assert "paired_geomean_ratio" in contents


def test_cli_self_test_runs_and_produces_json():
    """self-test must produce valid JSON with status field, regardless of device."""
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    r = _run("self-test", "--device", device)
    assert r.returncode in (0, 1), r.stderr  # 1 allowed if MPS unavailable
    data = json.loads(r.stdout)
    assert "status" in data
    assert "available_backends" in data
    assert "stock" in data["available_backends"]
    assert "elapsed_s" in data
    assert data["elapsed_s"] < 30, f"self-test took {data['elapsed_s']}s (limit 30)"
