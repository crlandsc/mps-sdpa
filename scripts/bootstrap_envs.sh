#!/usr/bin/env bash
# Create two pyenv virtualenvs for mps_sdpa benchmark A/B: stable + nightly.
# Idempotent: safe to re-run.
set -euo pipefail

PYVER="${PYVER:-3.11.9}"
STABLE_NAME="${STABLE_NAME:-mps_sdpa_stable}"
NIGHTLY_NAME="${NIGHTLY_NAME:-mps_sdpa_nightly}"

if ! command -v pyenv >/dev/null 2>&1; then
    echo "ERROR: pyenv not found. Install: https://github.com/pyenv/pyenv" >&2
    exit 1
fi

# Install the base Python if missing.
pyenv versions | grep -q "^\* *$PYVER\b\|^ *$PYVER\b" || pyenv install -s "$PYVER"

# Create each env (idempotent via -f).
for name in "$STABLE_NAME" "$NIGHTLY_NAME"; do
    if ! pyenv virtualenvs | grep -q "$name\b"; then
        pyenv virtualenv "$PYVER" "$name"
    fi
done

# Install stable PyTorch in STABLE_NAME.
echo ">> Installing stable PyTorch into $STABLE_NAME"
PYENV_VERSION="$STABLE_NAME" pip install --upgrade pip
PYENV_VERSION="$STABLE_NAME" pip install "torch>=2.11.0" "numpy>=1.24" "psutil>=5.9" \
    "pyobjc-core>=10.0" "pyobjc-framework-Metal>=10.0" \
    "pyobjc-framework-MetalPerformanceShadersGraph>=10.0" \
    "pytest>=7.4" "scipy>=1.11" "matplotlib>=3.7"
PYENV_VERSION="$STABLE_NAME" pip install -e .
PYENV_VERSION="$STABLE_NAME" pip freeze > "$(dirname "$0")/../runs/env_stable_freeze.txt" 2>/dev/null || true

# Install nightly PyTorch in NIGHTLY_NAME.
echo ">> Installing nightly PyTorch into $NIGHTLY_NAME"
PYENV_VERSION="$NIGHTLY_NAME" pip install --upgrade pip
PYENV_VERSION="$NIGHTLY_NAME" pip install --pre torch \
    --index-url https://download.pytorch.org/whl/nightly/cpu
PYENV_VERSION="$NIGHTLY_NAME" pip install "numpy>=1.24" "psutil>=5.9" \
    "pyobjc-core>=10.0" "pyobjc-framework-Metal>=10.0" \
    "pyobjc-framework-MetalPerformanceShadersGraph>=10.0" \
    "pytest>=7.4" "scipy>=1.11" "matplotlib>=3.7"
PYENV_VERSION="$NIGHTLY_NAME" pip install -e .
PYENV_VERSION="$NIGHTLY_NAME" pip freeze > "$(dirname "$0")/../runs/env_nightly_freeze.txt" 2>/dev/null || true

echo ">> Done. Activate with: pyenv activate $STABLE_NAME  (or $NIGHTLY_NAME)"
