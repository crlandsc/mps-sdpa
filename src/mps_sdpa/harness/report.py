"""Report writers: JSON per-case, CSV benchmark rows, markdown progress."""
from __future__ import annotations
import csv
import json
import math
import pathlib
from datetime import datetime
from typing import Iterable


def write_case_result(result: dict, path: pathlib.Path | str) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))


def write_bench_csv(rows: Iterable[dict], path: pathlib.Path | str) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def append_progress(path: pathlib.Path | str, msg: str, *, phase: int | None = None) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    line = f"- `{ts}` " + (f"[Phase {phase}] " if phase is not None else "") + msg + "\n"
    with path.open("a") as f:
        f.write(line)


def weighted_geomean_ratio(weighted_ratios: Iterable[tuple[float, float]]) -> float:
    weighted_ratios = list(weighted_ratios)
    total_w = sum(w for w, _ in weighted_ratios)
    if total_w == 0:
        return 1.0
    log_sum = sum(w * math.log(r) for w, r in weighted_ratios if r > 0)
    return math.exp(log_sum / total_w)
