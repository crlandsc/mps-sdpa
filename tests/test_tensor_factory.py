import torch

from mps_sdpa.harness import tensor_factory as tf
from mps_sdpa.suites.correctness_shapes import Case


def _case(mask="none", dtype="fp32"):
    return Case(
        case_id="t", B=1, H=2, Lq=4, Lkv=4, D=8,
        dtype=dtype, mask=mask, contiguous=True, dropout_p=0.0,
    )


def test_build_produces_correct_shapes():
    q, k, v, mask = tf.build(_case(), device="cpu", seed=0)
    assert q.shape == (1, 2, 4, 8)
    assert k.shape == (1, 2, 4, 8)
    assert v.shape == (1, 2, 4, 8)
    assert mask is None


def test_bool_mask_shape():
    q, k, v, mask = tf.build(_case(mask="bool_b1lk"), device="cpu", seed=0)
    assert mask.shape == (1, 1, 4, 4)
    assert mask.dtype == torch.bool


def test_causal_returns_none_mask():
    q, k, v, mask = tf.build(_case(mask="causal"), device="cpu", seed=0)
    assert mask is None


def test_empty_row_mask_has_at_least_one_all_false_row():
    _, _, _, mask = tf.build(_case(mask="empty_row"), device="cpu", seed=0)
    assert mask is not None
    assert bool((~mask.any(dim=-1)).any())


def test_dtype_fp16_produces_half():
    q, k, v, _ = tf.build(_case(dtype="fp16"), device="cpu", seed=0)
    assert q.dtype == torch.float16


def test_deterministic_same_seed():
    q1, *_ = tf.build(_case(), device="cpu", seed=42)
    q2, *_ = tf.build(_case(), device="cpu", seed=42)
    assert torch.equal(q1, q2)
