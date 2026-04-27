import pytest
from mps_sdpa.training_check import loss_compare as lc


def test_identical_curves_pass():
    a = [1.0 - 0.01 * i for i in range(100)]
    b = list(a)
    verdict = lc.compare(a, b, per_step_tol=1e-4)
    assert verdict["step_diffs_within_tol_frac"] == 1.0
    assert verdict["pearson"] > 0.99


def test_completely_different_curves_fail():
    a = [1.0] * 50
    b = [0.0] * 50
    verdict = lc.compare(a, b, per_step_tol=1e-4)
    assert verdict["step_diffs_within_tol_frac"] == 0.0


def test_similar_curves_with_small_noise_pass():
    a = [1.0 - 0.01 * i for i in range(100)]
    b = [la + 1e-5 for la in a]
    verdict = lc.compare(a, b, per_step_tol=1e-4)
    assert verdict["step_diffs_within_tol_frac"] == 1.0
