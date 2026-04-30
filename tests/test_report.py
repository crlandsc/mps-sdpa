import json

from mps_sdpa.harness import report


def test_write_case_result_json(tmp_path):
    result = {
        "candidate": "stock", "phase": 1, "env": "stable", "case_id": "T1",
        "params": {
            "B": 1, "H": 8, "Lq": 512, "Lkv": 512,
            "D": 64, "dtype": "bf16", "mask": "none",
        },
        "correctness": {"passed": True, "max_abs_err_fwd": 0.001, "max_rel_err_fwd": 0.01},
        "latency_ms": {
            "hot_median": 1.5, "p10": 1.4, "p50": 1.5,
            "p90": 1.6, "stddev": 0.05, "n": 50,
        },
        "memory_bytes": {
            "current_before": 0, "current_after": 0,
            "driver_before": 100, "driver_after": 110,
        },
        "contamination": {"thermal_max": "nominal", "load_max": 1.0, "accepted": True},
    }
    out = tmp_path / "res.json"
    report.write_case_result(result, out)
    loaded = json.loads(out.read_text())
    assert loaded == result


def test_write_progress_markdown(tmp_path):
    p = tmp_path / "progress.md"
    report.append_progress(p, "harness smoke test complete", phase=1)
    content = p.read_text()
    assert "harness smoke" in content
    assert "smoke test" in content


def test_weighted_geomean_ratio():
    ratios = [(1, 0.8), (2, 0.5), (4, 0.25)]
    g = report.weighted_geomean_ratio(ratios)
    assert 0.3 < g < 0.45
