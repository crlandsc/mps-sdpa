"""Benchmark timing: warmup, hot-latency distribution, paired A/B."""
from __future__ import annotations

import math
import statistics
import time
from typing import Callable

import torch


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def time_region(
    fn: Callable[[], torch.Tensor],
    *, warmup: int = 25, min_iters: int = 50, min_seconds: float = 1.0,
    device: str = "mps",
) -> dict:
    for _ in range(warmup):
        fn()
    _sync(device)

    samples: list[float] = []
    t_start = time.perf_counter_ns()
    while True:
        _sync(device)
        t0 = time.perf_counter_ns()
        fn()
        _sync(device)
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1e6)
        if len(samples) >= min_iters and (time.perf_counter_ns() - t_start) / 1e9 >= min_seconds:
            break
    samples_sorted = sorted(samples)
    return {
        "median": float(statistics.median(samples_sorted)),
        "p10": float(samples_sorted[int(0.1 * len(samples_sorted))]),
        "p50": float(samples_sorted[int(0.5 * len(samples_sorted))]),
        "p90": float(samples_sorted[int(0.9 * len(samples_sorted))]),
        "stddev": float(statistics.stdev(samples)) if len(samples) > 1 else 0.0,
        "n": len(samples),
    }


def paired_ab(
    baseline_fn: Callable[[], torch.Tensor],
    candidate_fn: Callable[[], torch.Tensor],
    *, n_pairs: int = 10, warmup: int = 25, min_iters: int = 50,
    min_seconds: float = 1.0, device: str = "mps",
) -> dict:
    ratios: list[float] = []
    per_pair: list[tuple[float, float]] = []
    for _ in range(n_pairs):
        b = time_region(
            baseline_fn, warmup=warmup, min_iters=min_iters,
            min_seconds=min_seconds, device=device,
        )
        c = time_region(
            candidate_fn, warmup=warmup, min_iters=min_iters,
            min_seconds=min_seconds, device=device,
        )
        per_pair.append((b["median"], c["median"]))
        ratios.append(c["median"] / b["median"])
    log_mean = sum(math.log(r) for r in ratios) / len(ratios)
    return {
        "paired_geomean_ratio": math.exp(log_mean),
        "per_pair": per_pair,
        "n_pairs": n_pairs,
    }
