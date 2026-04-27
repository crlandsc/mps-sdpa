"""mps_sdpa command-line interface."""
from __future__ import annotations
import argparse
import itertools
import json
import pathlib
import sys

from . import available_backends
from .harness import correctness, report
from .suites import correctness_shapes, realistic_shapes, general_shapes


def _suite_iter(name: str):
    return {
        "correctness": correctness_shapes.iter_cases,
        "realistic": realistic_shapes.iter_cases,
        "general": general_shapes.iter_cases,
    }[name]()


def _cmd_correctness(args):
    cases = _suite_iter(args.suite)
    if args.limit:
        cases = itertools.islice(cases, args.limit)
    summary = correctness.run_suite(backend_name=args.backend, cases=list(cases), device=args.device)
    out = pathlib.Path(args.out) if args.out else None
    if out:
        report.write_case_result(summary, out)
    print(json.dumps({"n": summary["n"], "n_pass": summary["n_pass"],
                      "n_hard": summary["n_hard"], "n_soft": summary["n_soft"]}))
    return 0 if summary["n_hard"] == 0 else 1


def _cmd_list_backends(args):
    for b in available_backends():
        print(b)
    return 0


def _cmd_self_test(args):
    """Quick end-to-end validation: env + imports + backends + correctness + bench."""
    import time
    import torch
    import torch.nn.functional as F
    from . import available_backends as _avail_backends
    from .backends import backend_reason
    from .utils import env as _envmod

    t0 = time.perf_counter()
    results: dict = {}

    # 1. Environment preflight
    try:
        if args.device == "mps":
            _envmod.preflight_check()
        results["env"] = "ok"
    except Exception as e:
        results["env"] = f"fail: {type(e).__name__}: {e}"
        print(json.dumps(results, indent=2))
        return 1

    # 2. Backend availability
    backends = _avail_backends()
    results["available_backends"] = backends
    if args.device == "mps":
        if "mpsgraph" not in backends:
            results["mpsgraph_reason"] = backend_reason("mpsgraph")
            results["status"] = "fail"
            print(json.dumps(results, indent=2))
            return 1

    # 3. Correctness per supported dtype
    corr_results = {}
    shapes = {"bf16": (1, 4, 2048, 64), "fp16": (1, 4, 2048, 64), "fp32": (1, 4, 2048, 64)}
    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    for name, t_dt in dtypes.items():
        B, H, L, D = shapes[name]
        try:
            torch.manual_seed(0)
            q = torch.randn(B, H, L, D, dtype=t_dt, device=args.device)
            k = torch.randn(B, H, L, D, dtype=t_dt, device=args.device)
            v = torch.randn(B, H, L, D, dtype=t_dt, device=args.device)
            from . import sdpa_opt
            out = sdpa_opt(q, k, v, backend="auto")
            ref = F.scaled_dot_product_attention(q, k, v)
            diff = (out - ref).abs().max().item()
            tol = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-3}[name]
            corr_results[name] = {
                "max_abs_diff": diff,
                "tol": tol,
                "pass": diff < tol,
            }
        except Exception as e:
            corr_results[name] = {"error": f"{type(e).__name__}: {e}"}
    results["correctness"] = corr_results

    # 4. Quick benchmark
    bench_results: dict = {}
    if args.device == "mps" and "mpsgraph" in backends:
        import time as _t
        from . import sdpa_opt
        B, H, L, D = 1, 8, 2048, 64
        q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
        k = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")
        v = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="mps")

        def _bench(fn):
            for _ in range(3):
                fn()
            torch.mps.synchronize()
            times = []
            for _ in range(10):
                s = _t.perf_counter()
                fn()
                torch.mps.synchronize()
                times.append(_t.perf_counter() - s)
            return min(times)

        try:
            t_stock = _bench(lambda: F.scaled_dot_product_attention(q, k, v))
            # Benchmark the auto-picked backend (what users actually get) plus
            # each individual backend for full visibility.
            t_auto = _bench(lambda: sdpa_opt(q, k, v, backend="auto"))
            from . import api as _api
            picked = _api._pick_auto(q)
            bench_results = {
                "shape": f"B{B}H{H}Lq{L}Lkv{L}D{D}_bf16",
                "stock_ms": round(t_stock * 1000, 3),
                "auto_backend": picked,
                "auto_ms": round(t_auto * 1000, 3),
                "speedup_auto_vs_stock": round(t_stock / t_auto, 2),
            }
            # Include individual backend times when present for full picture.
            if "mpsgraph_zc" in backends:
                t_zc = _bench(lambda: sdpa_opt(q, k, v, backend="mpsgraph_zc"))
                bench_results["mpsgraph_zc_ms"] = round(t_zc * 1000, 3)
            if "mpsgraph" in backends:
                t_mpsg = _bench(lambda: sdpa_opt(q, k, v, backend="mpsgraph"))
                bench_results["mpsgraph_ms"] = round(t_mpsg * 1000, 3)
        except Exception as e:
            bench_results = {"error": f"{type(e).__name__}: {e}"}
    results["benchmark"] = bench_results

    # 5. Overall status
    corr_ok = all(c.get("pass", False) for c in corr_results.values())
    results["status"] = "pass" if corr_ok else "fail"
    results["elapsed_s"] = round(time.perf_counter() - t0, 2)
    print(json.dumps(results, indent=2))
    return 0 if corr_ok else 1


def _cmd_benchmark(args):
    import itertools as _it
    import json as _json
    from .backends import get_backend
    from .harness import benchmark as bm
    from .harness import memory as mm
    from .harness import tensor_factory, cold_latency, report as rep
    cases = list(_it.islice(_suite_iter(args.suite), args.limit or None))
    baseline_fn = get_backend(args.baseline)
    candidate_fn = get_backend(args.backend)
    rows = []
    for case in cases:
        q, k, v, mask = tensor_factory.build(case, device=args.device, seed=0)
        is_causal = getattr(case, "mask", "none") == "causal"

        def run_baseline():
            return baseline_fn(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=is_causal, scale=None)

        def run_candidate():
            return candidate_fn(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=is_causal, scale=None)

        try:
            mem = mm.measure_region(run_candidate, device=args.device)
            pair = bm.paired_ab(run_baseline, run_candidate,
                                n_pairs=args.n_pairs, warmup=args.warmup,
                                min_iters=args.min_iters, min_seconds=args.min_seconds,
                                device=args.device)
            cold_spec = {
                "backend": args.backend, "device": args.device,
                "B": case.B, "H": case.H, "Lq": case.Lq, "Lkv": case.Lkv, "D": case.D,
                "dtype": case.dtype, "mask": case.mask, "is_causal": is_causal,
            }
            cold_ms = None
            if args.measure_cold:
                try:
                    cold_ms = cold_latency.measure_cold(cold_spec)
                except Exception as e:
                    cold_ms = f"err:{type(e).__name__}"
            rows.append({
                "case_id": case.case_id,
                "paired_geomean_ratio": pair["paired_geomean_ratio"],
                "cold_ms": cold_ms,
                "delta_driver_bytes": mem["delta_driver"],
                "delta_current_bytes": mem["delta_current"],
                "weight": getattr(case, "weight", 1),
                "note": "",
            })
        except RuntimeError as e:
            rows.append({
                "case_id": case.case_id,
                "paired_geomean_ratio": None,
                "cold_ms": None,
                "delta_driver_bytes": None,
                "delta_current_bytes": None,
                "weight": getattr(case, "weight", 1),
                "note": f"runtime_error:{type(e).__name__}:{str(e)[:120]}",
            })
    if args.out:
        rep.write_bench_csv(rows, args.out)
    wr = [(r["weight"], r["paired_geomean_ratio"]) for r in rows if r["paired_geomean_ratio"]]
    geomean = rep.weighted_geomean_ratio(wr) if wr else 1.0
    print(_json.dumps({"n_cases": len(rows), "weighted_geomean_candidate_over_baseline": geomean}))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mps_sdpa")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("correctness", help="Run correctness suite against a backend")
    c.add_argument("--backend", required=True)
    c.add_argument("--device", default="mps")
    c.add_argument("--suite", default="correctness", choices=["correctness", "realistic", "general"])
    c.add_argument("--limit", type=int, default=None)
    c.add_argument("--out", default=None)
    c.set_defaults(func=_cmd_correctness)

    b = sub.add_parser("benchmark", help="Paired A/B benchmark: candidate vs baseline")
    b.add_argument("--backend", required=True, help="candidate backend")
    b.add_argument("--baseline", default="stock")
    b.add_argument("--device", default="mps")
    b.add_argument("--suite", default="realistic", choices=["correctness", "realistic", "general"])
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--n-pairs", type=int, default=5, dest="n_pairs")
    b.add_argument("--warmup", type=int, default=25)
    b.add_argument("--min-iters", type=int, default=50, dest="min_iters")
    b.add_argument("--min-seconds", type=float, default=1.0, dest="min_seconds")
    b.add_argument("--measure-cold", action="store_true", dest="measure_cold")
    b.add_argument("--out", default=None, help="CSV path")
    b.set_defaults(func=_cmd_benchmark)

    l = sub.add_parser("list-backends", help="List available backends")
    l.set_defaults(func=_cmd_list_backends)

    s = sub.add_parser("self-test", help="Quick end-to-end validation (<30s)")
    s.add_argument("--device", default="mps")
    s.set_defaults(func=_cmd_self_test)

    args = p.parse_args(argv)
    from .utils import env as _envmod
    _envmod.assert_wandb_not_imported()
    if getattr(args, "device", "") == "mps":
        _envmod.preflight_check()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
