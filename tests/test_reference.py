import torch

from mps_sdpa.harness import reference


def test_math_reference_matches_manual_softmax():
    torch.manual_seed(0)
    q = torch.randn(1, 2, 8, 16, dtype=torch.float32)
    k = torch.randn(1, 2, 8, 16, dtype=torch.float32)
    v = torch.randn(1, 2, 8, 16, dtype=torch.float32)
    out = reference.math_reference(q, k, v)
    scores = (q @ k.transpose(-1, -2)) / (16 ** 0.5)
    manual = torch.softmax(scores, dim=-1) @ v
    assert torch.allclose(out, manual, atol=1e-6)


def test_fp64_spot_check_within_tolerance_of_math_fp32():
    torch.manual_seed(0)
    q = torch.randn(1, 2, 8, 16, dtype=torch.float32)
    k = torch.randn(1, 2, 8, 16, dtype=torch.float32)
    v = torch.randn(1, 2, 8, 16, dtype=torch.float32)
    out_f32 = reference.math_reference(q, k, v)
    out_f64 = reference.cpu_fp64_reference(q, k, v)
    assert torch.allclose(out_f32, out_f64.float(), atol=1e-6)
