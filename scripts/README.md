# mps_sdpa scripts

## bootstrap_envs.sh

Creates two pyenv virtualenvs for benchmark A/B testing:
- `mps_sdpa_stable` — latest stable PyTorch.
- `mps_sdpa_nightly` — latest PyTorch nightly.

Usage:

```bash
./mps_sdpa/scripts/bootstrap_envs.sh
```

Re-running is safe (idempotent). Override Python version with `PYVER=3.12.1 ./bootstrap_envs.sh`.

## Activating

```bash
pyenv activate mps_sdpa_stable
# or
pyenv activate mps_sdpa_nightly
```

## verify_machine.sh

Captures all machine-specific verification artifacts in one shot — useful
when adding a new chip / OS / torch combination to COMPAT.md or CHANGELOG.md.

Usage:

```bash
scripts/verify_machine.sh                              # full run
OUT_DIR=/tmp/m3max scripts/verify_machine.sh           # custom output dir
SKIP_BENCH=1 scripts/verify_machine.sh                 # skip bench + calib
                                                        # (use when GPU is busy)
```

Outputs to a timestamped directory: env, self-test, calibration thresholds,
pytest log, correctness log, benchmark CSV. Do not run the benchmark step
under GPU contention — calibration would also cache wrong thresholds.
