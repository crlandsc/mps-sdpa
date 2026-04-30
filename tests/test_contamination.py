from mps_sdpa.harness import contamination as cm


def test_judge_accepts_nominal_low_load():
    samples = [{"thermal": "nominal", "load1": 0.5, "free_ram_gb": 10.0}] * 10
    verdict = cm.judge(samples)
    assert verdict["accepted"] is True


def test_judge_rejects_serious_thermal():
    samples = [{"thermal": "serious", "load1": 0.5, "free_ram_gb": 10.0}]
    verdict = cm.judge(samples)
    assert verdict["accepted"] is False
    assert "thermal" in verdict["reasons"][0]


def test_judge_rejects_high_load():
    samples = [{"thermal": "nominal", "load1": 5.0, "free_ram_gb": 10.0}]
    verdict = cm.judge(samples, max_load1=3.0)
    assert verdict["accepted"] is False


def test_judge_rejects_low_ram():
    samples = [{"thermal": "nominal", "load1": 0.5, "free_ram_gb": 1.0}]
    verdict = cm.judge(samples)
    assert verdict["accepted"] is False


def test_judge_flags_distribution_instability():
    verdict = cm.judge_distribution([{"p10": 1.0, "p90": 2.0}], max_p90_over_p10=1.25)
    assert verdict["accepted"] is False
