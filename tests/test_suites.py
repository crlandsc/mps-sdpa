"""Shape suite tests."""
from mps_sdpa.suites import correctness_shapes, general_shapes, realistic_shapes


def test_correctness_suite_non_empty():
    cases = list(correctness_shapes.iter_cases())
    assert len(cases) > 0


def test_correctness_case_has_required_fields():
    for case in correctness_shapes.iter_cases():
        assert case.B > 0
        assert case.H > 0
        assert case.Lq > 0
        assert case.Lkv > 0
        assert case.D > 0
        assert case.dtype in {"fp16", "bf16", "fp32"}
        assert case.mask in {
            "none", "bool_bhlk", "bool_b1lk", "causal", "additive_float", "empty_row",
        }


def test_correctness_includes_causal_cases():
    cases = list(correctness_shapes.iter_cases())
    assert any(c.mask == "causal" for c in cases)


def test_correctness_includes_empty_row_mask():
    cases = list(correctness_shapes.iter_cases())
    assert any(c.mask == "empty_row" for c in cases)


def test_correctness_includes_all_three_dtypes():
    dtypes = {c.dtype for c in correctness_shapes.iter_cases()}
    assert {"fp16", "bf16", "fp32"}.issubset(dtypes)


def test_correctness_includes_d_variants():
    ds = {c.D for c in correctness_shapes.iter_cases()}
    assert {32, 64, 128}.issubset(ds)


def test_correctness_includes_cross_attention_lq_ne_lkv():
    cases = list(correctness_shapes.iter_cases())
    assert any(c.Lq != c.Lkv for c in cases)


def test_realistic_suite_non_empty():
    cases = list(realistic_shapes.iter_cases())
    assert len(cases) > 0


def test_realistic_cases_have_weights():
    for c in realistic_shapes.iter_cases():
        assert c.weight > 0


def test_realistic_long_seq_weights_dominate():
    cases = list(realistic_shapes.iter_cases())
    long_w = sum(c.weight for c in cases if max(c.Lq, c.Lkv) >= 2048)
    short_w = sum(c.weight for c in cases if max(c.Lq, c.Lkv) < 2048)
    assert long_w > short_w, f"long={long_w} short={short_w}"


def test_realistic_includes_cross_attention():
    cases = list(realistic_shapes.iter_cases())
    assert any(c.Lq != c.Lkv for c in cases)


def test_realistic_includes_bool_mask_case():
    cases = list(realistic_shapes.iter_cases())
    assert any(c.mask == "bool_b1lk" for c in cases)


def test_realistic_no_causal_in_ranking():
    for c in realistic_shapes.iter_cases():
        assert c.mask != "causal", "causal cases must not appear in the realistic ranking suite"


def test_general_suite_non_empty():
    cases = list(general_shapes.iter_cases())
    assert len(cases) > 0


def test_general_includes_causal_cases():
    cases = list(general_shapes.iter_cases())
    assert any(c.mask == "causal" for c in cases)


def test_general_includes_all_three_dtypes():
    dtypes = {c.dtype for c in general_shapes.iter_cases()}
    assert {"fp16", "bf16", "fp32"}.issubset(dtypes)
