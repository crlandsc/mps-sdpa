import torch

from mps_sdpa.harness import correctness
from mps_sdpa.suites.correctness_shapes import Case


def _small_case(mask="none", dtype="fp32"):
    return Case(
        case_id="s", B=1, H=2, Lq=8, Lkv=8, D=16,
        dtype=dtype, mask=mask, contiguous=True, dropout_p=0.0,
    )


def test_stock_vs_math_reference_passes():
    result = correctness.check_case(backend_name="stock", case=_small_case(), device="cpu")
    assert result["passed"] is True
    assert result["failure_class"] is None


def test_broken_backend_marks_hard_fail():
    from mps_sdpa import backends
    def broken(*a, **kw):
        return torch.zeros(1, 2, 8, 16)
    backends.register_backend("broken_test", broken, available=True)
    result = correctness.check_case(backend_name="broken_test", case=_small_case(), device="cpu")
    assert result["passed"] is False
    assert result["failure_class"] in {"soft", "hard"}
