import pytest

from mps_sdpa.harness import tolerances


def test_forward_tolerance_fp32():
    atol, rtol = tolerances.forward_tol("fp32")
    assert atol == 5e-6
    assert rtol == 5e-5


def test_forward_tolerance_fp16_bf16_same():
    assert tolerances.forward_tol("fp16") == tolerances.forward_tol("bf16")


def test_backward_tolerance_is_2x_forward():
    f = tolerances.forward_tol("fp16")
    b = tolerances.backward_tol("fp16")
    assert b[0] == 2 * f[0]
    assert b[1] == 2 * f[1]


def test_unknown_dtype_raises():
    with pytest.raises(KeyError):
        tolerances.forward_tol("int8")
