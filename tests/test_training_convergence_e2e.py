"""End-to-end: stock (opt=False) vs stock-via-opt (opt=True) on CPU should be identical."""
from mps_sdpa.training_check import loss_compare as lc
from mps_sdpa.training_check import synthetic_train as st


def test_stock_vs_opt_cpu_fp32_identical():
    a = st.train(steps=50, seed=0, dtype="fp32", use_opt=False, device="cpu")
    b = st.train(steps=50, seed=0, dtype="fp32", use_opt=True, device="cpu")
    v = lc.compare(a, b, per_step_tol=1e-5)
    assert v["step_diffs_within_tol_frac"] >= 0.98
    assert v["pearson"] > 0.99
