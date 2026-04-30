"""Environment utility tests."""
import os
import sys

import pytest

from mps_sdpa.utils import env


def test_force_wandb_offline_sets_env_vars(monkeypatch):
    for k in ("WANDB_MODE", "WANDB_DISABLED", "WANDB_SILENT"):
        monkeypatch.delenv(k, raising=False)
    env.force_wandb_offline()
    assert os.environ["WANDB_MODE"] == "offline"
    assert os.environ["WANDB_DISABLED"] == "true"
    assert os.environ["WANDB_SILENT"] == "true"


def test_mps_env_knobs_context_manager_restores_prior():
    os.environ.pop("PYTORCH_MPS_FAST_MATH", None)
    with env.mps_env(fast_math=True, prefer_metal=False):
        assert os.environ["PYTORCH_MPS_FAST_MATH"] == "1"
        assert os.environ["PYTORCH_MPS_PREFER_METAL"] == "0"
    assert "PYTORCH_MPS_FAST_MATH" not in os.environ
    assert "PYTORCH_MPS_PREFER_METAL" not in os.environ


def test_mps_env_knobs_preserves_existing_value():
    os.environ["PYTORCH_MPS_FAST_MATH"] = "prior"
    try:
        with env.mps_env(fast_math=True):
            assert os.environ["PYTORCH_MPS_FAST_MATH"] == "1"
        assert os.environ["PYTORCH_MPS_FAST_MATH"] == "prior"
    finally:
        del os.environ["PYTORCH_MPS_FAST_MATH"]


def test_assert_wandb_not_imported_passes_when_not_imported(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    sys.modules.pop("wandb", None)
    env.assert_wandb_not_imported()


def test_assert_wandb_not_imported_fails_when_imported(monkeypatch):
    import types
    fake = types.ModuleType("wandb")
    monkeypatch.setitem(sys.modules, "wandb", fake)
    with pytest.raises(RuntimeError, match="wandb was imported"):
        env.assert_wandb_not_imported()


def test_preflight_asserts_apple_silicon_macos(monkeypatch):
    import platform
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(platform, "mac_ver", lambda: ("14.4", ("", "", ""), "arm64"))
    env.preflight_check()


def test_preflight_rejects_non_darwin(monkeypatch):
    import platform
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    with pytest.raises(RuntimeError, match="macOS required"):
        env.preflight_check()
