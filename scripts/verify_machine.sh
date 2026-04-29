#!/usr/bin/env bash
# verify_machine.sh — capture machine-specific verification artifacts for
# adding a new chip / OS / torch combination to COMPAT.md or CHANGELOG.md.
#
# Usage:
#   scripts/verify_machine.sh                              # full run
#   OUT_DIR=/tmp/m3max scripts/verify_machine.sh           # custom output dir
#   SKIP_BENCH=1 scripts/verify_machine.sh                 # skip benchmark
#                                                          # (use when GPU is busy)
#
# Output: a timestamped directory containing env, pytest, correctness,
# benchmark, and thresholds artifacts. All intermediate output is also
# echoed to stdout so you can watch in real time.
#
# IMPORTANT: do not run the benchmark step under GPU contention (e.g.,
# while a training run is active). Calibration would also cache bad
# thresholds. Either wait for the GPU to be idle, or pass SKIP_BENCH=1.

set -euo pipefail

OUT_DIR="${OUT_DIR:-./verify-results-$(date +%Y%m%d-%H%M%S)}"
SKIP_BENCH="${SKIP_BENCH:-0}"

# --- Preflight ---------------------------------------------------------------
# mps_sdpa must be importable in the active Python env. Fail loudly with
# setup instructions rather than half-running and hitting a ModuleNotFoundError
# buried in the env-metadata step.
if ! python3 -c "import mps_sdpa" 2>/dev/null; then
    PY_PREFIX=$(python3 -c 'import sys; print(sys.prefix)' 2>/dev/null || echo unknown)
    cat >&2 <<EOF
ERROR: mps_sdpa is not importable in the active Python environment.
Active env: $PY_PREFIX

From the repo root, set up per CONTRIBUTING.md:

    python -m venv .venv && source .venv/bin/activate
    pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
    pip install -e ".[dev]"

Then re-run scripts/verify_machine.sh.
EOF
    exit 1
fi

mkdir -p "$OUT_DIR"
echo "Output dir: $OUT_DIR"
echo

# --- 1. Environment metadata --------------------------------------------------
echo "=== Environment ==="
{
    echo "--- sw_vers ---"
    sw_vers
    echo
    echo "--- Hardware ---"
    echo "Chip:      $(sysctl -n machdep.cpu.brand_string)"
    echo "CPU:       $(sysctl -n hw.ncpu) total ($(sysctl -n hw.physicalcpu) physical)"
    MEM_BYTES=$(sysctl -n hw.memsize)
    MEM_GB=$(( MEM_BYTES / 1024 / 1024 / 1024 ))
    echo "Memory:    ${MEM_GB} GB"
    GPU_CORES=$(system_profiler SPDisplaysDataType 2>/dev/null \
        | awk -F': ' '/Total Number of Cores/ {print $2; exit}')
    echo "GPU cores: ${GPU_CORES:-unknown}"
    echo
    echo "--- Software ---"
    python3 --version
    python3 -c "import torch; print('torch:', torch.__version__)"
    python3 -c "import mps_sdpa; print('mps_sdpa:', mps_sdpa.__version__)"
} | tee "$OUT_DIR/env.txt"
echo

# --- 2. Self-test + backend availability --------------------------------------
echo "=== Self-test ==="
mps-sdpa self-test --device mps 2>&1 | tee "$OUT_DIR/self-test.txt"
echo

echo "=== Backend availability ==="
mps-sdpa list-backends 2>&1 | tee "$OUT_DIR/backends.txt"
echo

# --- 3. Force calibration (skip if SKIP_BENCH=1) ------------------------------
# Calibration measures stock-vs-mpsgraph crossover thresholds, then caches
# them. Running under GPU contention writes wrong thresholds — they stick
# until the next MPS_SDPA_FORCE_CALIBRATE=1 run.
if [[ "$SKIP_BENCH" == "1" ]]; then
    echo "=== Calibration: SKIPPED (SKIP_BENCH=1) ==="
    echo
else
    echo "=== Force calibration (overwrites ~/.cache/mps_sdpa/thresholds.json) ==="
    MPS_SDPA_FORCE_CALIBRATE=1 python3 -c "
import torch
import mps_sdpa
# Trigger calibration via a real call
q = torch.zeros(1, 8, 1024, 64, dtype=torch.bfloat16, device='mps')
k = torch.zeros_like(q)
v = torch.zeros_like(q)
mps_sdpa.sdpa_opt(q, k, v)
print('calibration complete')
" 2>&1 | tee "$OUT_DIR/calibrate.txt"
    cp ~/.cache/mps_sdpa/thresholds.json "$OUT_DIR/thresholds.json"
    echo "--- thresholds.json ---"
    cat "$OUT_DIR/thresholds.json"
    echo
fi

# --- 4. Full pytest -----------------------------------------------------------
echo "=== Pytest suite (213 tests expected) ==="
pytest tests/ -q 2>&1 | tee "$OUT_DIR/pytest.txt"
echo

# --- 5. Correctness suite -----------------------------------------------------
echo "=== Correctness suite ==="
mps-sdpa correctness --backend mpsgraph_zc --device mps --suite realistic \
    2>&1 | tee "$OUT_DIR/correctness.txt"
echo

# --- 6. Benchmark (skip if SKIP_BENCH=1) --------------------------------------
# --measure-cold captures cold-launch latency and per-call driver memory
# deltas across all shapes in the realistic suite (in the bench.csv output).
if [[ "$SKIP_BENCH" == "1" ]]; then
    echo "=== Benchmark: SKIPPED (SKIP_BENCH=1) ==="
else
    echo "=== Benchmark ==="
    mps-sdpa benchmark --backend mpsgraph_zc --baseline stock --device mps \
        --suite realistic --n-pairs 3 --measure-cold \
        --out "$OUT_DIR/bench.csv" \
        2>&1 | tee "$OUT_DIR/bench.txt"
fi
echo

# --- 7. Memory probe at headline shapes ---------------------------------------
# Pin the README's "≫128×/≫64×/32× reduction" claim at L=2048/4096/8192.
# These shapes are the marketing headline; if the algorithmic memory advantage
# of the fused op were ever lost, this is where you'd see it first. Math-only
# (no timing), so safe even under contention — but skipped if SKIP_BENCH=1
# because the calibration step it depends on is also skipped.
if [[ "$SKIP_BENCH" == "1" ]]; then
    echo "=== Memory probe: SKIPPED (SKIP_BENCH=1) ==="
else
    echo "=== Memory probe (driver_allocated_memory deltas at L=2048/4096/8192) ==="
    python3 - 2>&1 << 'PY' | tee "$OUT_DIR/memory.txt"
import torch
import torch.nn.functional as F
from mps_sdpa import sdpa_opt

def measure(fn, L):
    torch.mps.empty_cache()
    torch.mps.synchronize()
    before = torch.mps.driver_allocated_memory()
    q = torch.randn(1, 8, L, 64, dtype=torch.bfloat16, device="mps")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = fn(q, k, v)
    torch.mps.synchronize()
    after = torch.mps.driver_allocated_memory()
    del q, k, v, out
    return (after - before) / (1024 * 1024)  # MB

print(f"{'L':>6} {'stock_MB':>12} {'mps_sdpa_MB':>14} {'reduction':>11}")
print("-" * 48)
for L in (2048, 4096, 8192):
    s = measure(F.scaled_dot_product_attention, L)
    o = measure(sdpa_opt, L)
    ratio = s / max(o, 0.001)
    print(f"{L:>6} {s:>12.1f} {o:>14.1f} {ratio:>10.0f}x")
PY
fi
echo

# --- Summary ------------------------------------------------------------------
echo "================================================================"
echo "All results saved to: $OUT_DIR"
echo "================================================================"
ls -la "$OUT_DIR"
