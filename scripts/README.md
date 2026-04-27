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
