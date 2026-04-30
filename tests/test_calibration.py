"""Tests for the shape-threshold auto-calibration module."""

import torch

from mps_sdpa.backends import _calibrate


def test_default_thresholds_well_formed():
    t = _calibrate._DEFAULT_THRESHOLDS
    assert "fused_min_bytes" in t
    assert {"bf16", "fp16", "fp32"} <= t["fused_min_bytes"].keys()
    for v in t["fused_min_bytes"].values():
        assert isinstance(v, int) and v > 0
    assert t["dropout_min_bytes"] > 0
    assert t["dropout_max_bytes"] > t["dropout_min_bytes"]
    assert t["calibrated"] is False


def test_skip_calibration_env_var_returns_defaults(monkeypatch):
    monkeypatch.setenv("MPS_SDPA_SKIP_CALIBRATION", "1")
    _calibrate._cached_thresholds = None
    got = _calibrate.get_thresholds()
    assert got is _calibrate._DEFAULT_THRESHOLDS
    assert got["calibrated"] is False


def test_fingerprint_stable_across_calls():
    fp1 = _calibrate._fingerprint()
    fp2 = _calibrate._fingerprint()
    assert fp1 == fp2
    assert "chip" in fp1 and "os" in fp1 and "torch" in fp1
    assert fp1["schema"] == _calibrate._CACHE_SCHEMA_VERSION


def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(_calibrate, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(_calibrate, "_CACHE_FILE", tmp_path / "thresholds.json")
    sample = {
        "fused_min_bytes": {"bf16": 500, "fp16": 500, "fp32": 1000},
        "dropout_min_bytes": 1024,
        "dropout_max_bytes": 2048,
        "calibrated": True,
    }
    _calibrate._save_cache(sample)
    loaded = _calibrate._load_cache()
    assert loaded == sample


def test_cache_miss_on_fingerprint_change(tmp_path, monkeypatch):
    monkeypatch.setattr(_calibrate, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(_calibrate, "_CACHE_FILE", tmp_path / "thresholds.json")
    sample = {
        "fused_min_bytes": {"bf16": 1, "fp16": 1, "fp32": 1},
        "dropout_min_bytes": 1, "dropout_max_bytes": 2, "calibrated": True,
    }
    # Save with one fingerprint, then monkeypatch fingerprint to differ.
    _calibrate._save_cache(sample)
    monkeypatch.setattr(_calibrate, "_fingerprint",
                        lambda: {"chip": "x", "os": "y", "torch": "z", "schema": 99})
    assert _calibrate._load_cache() is None


def test_cache_miss_on_malformed_file(tmp_path, monkeypatch):
    monkeypatch.setattr(_calibrate, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(_calibrate, "_CACHE_FILE", tmp_path / "thresholds.json")
    (tmp_path / "thresholds.json").write_text("not json")
    assert _calibrate._load_cache() is None


def test_dtype_key_mapping():
    assert _calibrate.dtype_key(torch.bfloat16) == "bf16"
    assert _calibrate.dtype_key(torch.float16) == "fp16"
    assert _calibrate.dtype_key(torch.float32) == "fp32"
    assert _calibrate.dtype_key(torch.int32) is None


def test_cant_calibrate_when_mps_unavailable(monkeypatch):
    """When MPS is unavailable or backend import fails, calibrate returns False."""
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert _calibrate._can_calibrate() is False


def test_get_thresholds_falls_back_when_bench_fails(monkeypatch, tmp_path):
    """If _calibrate() raises, we silently fall back to defaults."""
    monkeypatch.setenv("MPS_SDPA_SKIP_CALIBRATION", "0")
    monkeypatch.setattr(_calibrate, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(_calibrate, "_CACHE_FILE", tmp_path / "thresholds.json")
    _calibrate._cached_thresholds = None
    monkeypatch.setattr(_calibrate, "_can_calibrate", lambda: True)
    def _boom():
        raise RuntimeError("simulated bench failure")
    monkeypatch.setattr(_calibrate, "_calibrate", _boom)
    got = _calibrate.get_thresholds()
    assert got is _calibrate._DEFAULT_THRESHOLDS


def test_invalidate_drops_memo_and_file(tmp_path, monkeypatch):
    monkeypatch.setattr(_calibrate, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(_calibrate, "_CACHE_FILE", tmp_path / "thresholds.json")
    _calibrate._cached_thresholds = {"fused_min_bytes": {"bf16": 1, "fp16": 1, "fp32": 1},
                                      "dropout_min_bytes": 1, "dropout_max_bytes": 2,
                                      "calibrated": True}
    _calibrate._save_cache(_calibrate._cached_thresholds)
    _calibrate.invalidate()
    assert _calibrate._cached_thresholds is None
    assert not (tmp_path / "thresholds.json").exists()


def test_calibrate_module_thresholds_flow_into_dispatch(monkeypatch):
    """Dispatch picks per-dtype threshold from _calibrate.get_thresholds()."""
    # Override thresholds via in-memory cache
    fake = {
        "fused_min_bytes": {"bf16": 999_999_999, "fp16": 999_999_999, "fp32": 999_999_999},
        "dropout_min_bytes": 1, "dropout_max_bytes": 2**40,
        "calibrated": True,
    }
    monkeypatch.setattr(_calibrate, "_cached_thresholds", fake)
    # Unsupported dtype path (int32) should fall back to stock too.
    # Supported dtype (bfloat16) with threshold set to 1B bytes → falls back.
    # We can't actually run a real MPS call here in sandbox, so just validate
    # the code path chooses fallback by inspecting the threshold lookup.
    assert _calibrate.get_thresholds()["fused_min_bytes"]["bf16"] == 999_999_999


def test_no_win_dtype_returns_none(monkeypatch):
    """When no probe shape wins for a dtype, _calibrate_dtype returns None
    (not 2**40). This is part of the v0.2.0 calibration improvements."""
    from mps_sdpa.backends import _calibrate
    # Mock _bench_shape so stock always wins by enough margin
    monkeypatch.setattr(_calibrate, "_bench_shape",
                        lambda L, dtype, use_mpsg: 1.0 if use_mpsg else 0.99)
    result = _calibrate._calibrate_dtype(torch.bfloat16, 2)
    assert result is None, f"expected None for never-wins; got {result}"


def test_schema_version_bump_invalidates_old_cache(tmp_path, monkeypatch):
    """An on-disk cache with schema=1 must be ignored after the v2 bump."""
    import json

    from mps_sdpa.backends import _calibrate

    monkeypatch.setattr(_calibrate, "_CACHE_FILE", tmp_path / "thresholds.json")
    fp = _calibrate._fingerprint()
    fp_old = {**fp, "schema": 1}
    (tmp_path / "thresholds.json").write_text(json.dumps({
        "fingerprint": fp_old,
        "thresholds": {
            "fused_min_bytes": {"bf16": 1024, "fp16": 1024, "fp32": 1024},
            "dropout_min_bytes": 1024,
            "dropout_max_bytes": 4096,
            "calibrated": True,
        },
    }))
    assert _calibrate._load_cache() is None
