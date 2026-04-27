"""Shared pytest fixtures + env setup for mps_sdpa tests."""
import os

# Skip auto-calibration in tests by default — keeps test startup fast and
# independent of machine-specific benchmark outputs. Tests that exercise the
# calibration path clear this var themselves.
os.environ.setdefault("MPS_SDPA_SKIP_CALIBRATION", "1")
