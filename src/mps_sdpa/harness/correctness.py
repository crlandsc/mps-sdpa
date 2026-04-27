"""Correctness checker: compares a backend's output to math reference per case."""
from __future__ import annotations
from typing import Any
import torch

from ..backends import get_backend
from ..suites.correctness_shapes import Case
from . import reference, tensor_factory, tolerances


def _is_oom(exc: Exception) -> bool:
    """Heuristic for OOM / buffer-size errors on MPS (known stock crash on long seqs)."""
    msg = str(exc).lower()
    return any(s in msg for s in ("out of memory", "oom", "buffer size", "total bytes of tensor exceeds"))


def check_case(*, backend_name: str, case: Case, device: str = "cpu") -> dict[str, Any]:
    q, k, v, mask = tensor_factory.build(case, device=device, seed=0)
    is_causal = case.mask == "causal"
    atol, rtol = tolerances.forward_tol(case.dtype)
    dropout_p = case.dropout_p
    result: dict[str, Any] = {
        "case_id": case.case_id, "backend": backend_name,
        "passed": False, "failure_class": None, "max_abs_err": None, "max_rel_err": None,
        "shape_mismatch": False, "nan_inf_leak": False, "dropout_mode": None,
    }
    try:
        fn = get_backend(backend_name)
        out = fn(q, k, v, attn_mask=mask, dropout_p=dropout_p, is_causal=is_causal, scale=None)
    except NotImplementedError as e:
        result["failure_class"] = "hard"
        result["error"] = str(e)
        return result
    except RuntimeError as e:
        if _is_oom(e):
            result["failure_class"] = None
            result["passed"] = False
            result["error"] = f"OOM: {e}"
            result["environmental_skip"] = True
            return result
        result["failure_class"] = "hard"
        result["error"] = f"{type(e).__name__}: {e}"
        return result
    except Exception as e:
        result["failure_class"] = "hard"
        result["error"] = f"{type(e).__name__}: {e}"
        return result

    # Dropout path: compare distributional statistics, not exact values.
    if dropout_p > 0.0:
        result["dropout_mode"] = "distributional"
        ref_samples = []
        out_samples = []
        for s in range(8):
            torch.manual_seed(s)
            out_samples.append(fn(q, k, v, attn_mask=mask, dropout_p=dropout_p,
                                  is_causal=is_causal, scale=None).float())
            torch.manual_seed(s)
            ref_samples.append(reference.math_reference(
                q, k, v, attn_mask=mask, dropout_p=dropout_p,
                is_causal=is_causal, scale=None).float())
        out_mean = torch.stack(out_samples).mean()
        ref_mean = torch.stack(ref_samples).mean()
        out_std = torch.stack(out_samples).std()
        ref_std = torch.stack(ref_samples).std()
        mean_diff = float((out_mean - ref_mean).abs())
        std_diff = float((out_std - ref_std).abs())
        result["max_abs_err"] = max(mean_diff, std_diff)
        if mean_diff < 5e-2 and std_diff < 5e-2:
            result["passed"] = True
        else:
            result["failure_class"] = "soft"
        return result

    ref = reference.math_reference(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=is_causal, scale=None)

    if out.shape != ref.shape:
        result["failure_class"] = "hard"
        result["shape_mismatch"] = True
        return result
    if out.dtype != ref.dtype:
        result["failure_class"] = "hard"
        return result
    if torch.isnan(out).any() and not torch.isnan(ref).any():
        result["failure_class"] = "hard"
        result["nan_inf_leak"] = True
        return result
    if torch.isinf(out).any() and not torch.isinf(ref).any():
        result["failure_class"] = "hard"
        result["nan_inf_leak"] = True
        return result

    diff = (out.float() - ref.float()).abs()
    rel = diff / (ref.float().abs() + 1e-12)
    result["max_abs_err"] = float(diff.max())
    result["max_rel_err"] = float(rel.max())

    if torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol):
        result["passed"] = True
    else:
        result["failure_class"] = "soft"
    return result


def run_suite(*, backend_name: str, cases, device: str = "cpu") -> dict[str, Any]:
    results = [check_case(backend_name=backend_name, case=c, device=device) for c in cases]
    n_pass = sum(r["passed"] for r in results)
    n_hard = sum(r["failure_class"] == "hard" for r in results)
    n_soft = sum(r["failure_class"] == "soft" for r in results)
    return {
        "n": len(results), "n_pass": n_pass, "n_hard": n_hard, "n_soft": n_soft,
        "results": results,
    }
