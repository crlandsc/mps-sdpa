"""Smoke tests for tests/_tolerances.py — the chip-aware cross-impl helper.

These don't require MPS — they monkeypatch the chip-brand probe so the
helper's behavior on each chip family is deterministic regardless of where
the suite runs. The aim is to pin the policy: strict-by-default, loose
only for explicitly-listed chips.

Distinct from tests/test_tolerances.py, which tests the package's internal
mps_sdpa.harness.tolerances module.
"""
from __future__ import annotations

import pytest
import torch

from tests import _tolerances
from tests._tolerances import cross_impl_atol


# Pinned values — if these change, the helper's constants need to change
# too. These tests are the regression check that the policy is intact.
_STRICT_BF16 = 5e-3
_STRICT_FP16 = 5e-3
_STRICT_FP32 = 5e-6
_LOOSE_M1_BF16 = 5e-2
_NON_CONTIG_FLOOR_BF16 = 1e-2


@pytest.fixture
def force_chip(monkeypatch):
    """Override the chip-brand probe in the helper."""
    def _set(brand: str) -> None:
        monkeypatch.setattr(_tolerances, "_chip_brand", lambda: brand)
    return _set


# ---- Default-strict policy --------------------------------------------------

def test_unknown_chip_is_strict(force_chip):
    """Off-Apple environments and any unrecognized chip default to strict."""
    force_chip("unknown")
    assert cross_impl_atol(torch.bfloat16) == _STRICT_BF16
    assert cross_impl_atol(torch.float16) == _STRICT_FP16
    assert cross_impl_atol(torch.float32) == _STRICT_FP32


@pytest.mark.parametrize("brand", [
    "Apple M2",
    "Apple M3",
    "Apple M3 Max",
    "Apple M3 Pro",
    "Apple M4",
    "Apple M4 Pro",
    "Apple M4 Max",
])
def test_m2_through_m4_are_strict(force_chip, brand):
    """M2-M4 default to strict — no observed cross-impl drift > 5e-3 yet.
    If this test starts failing on a real machine, drift was observed
    empirically and the chip should be added to _LOOSE_CHIPS_BF16 with a
    reference to the failing test (see COMPAT.md "Test tolerance policy")."""
    force_chip(brand)
    assert cross_impl_atol(torch.bfloat16) == _STRICT_BF16
    assert cross_impl_atol(torch.float16) == _STRICT_FP16


# ---- Non-contig knob --------------------------------------------------------

def test_strict_chip_non_contig_bumps_to_floor(force_chip):
    """On strict-default chips, non_contig=True bumps to the chip-independent
    1e-2 floor (contig-copy path drift documented since v0.1.0)."""
    force_chip("unknown")
    assert cross_impl_atol(torch.bfloat16, non_contig=True) == _NON_CONTIG_FLOOR_BF16
    assert cross_impl_atol(torch.float16, non_contig=True) == _NON_CONTIG_FLOOR_BF16


def test_strict_chip_contig_stays_strict(force_chip):
    """non_contig=False (default) preserves the strict bound."""
    force_chip("Apple M4")
    assert cross_impl_atol(torch.bfloat16) == _STRICT_BF16
    assert cross_impl_atol(torch.bfloat16, non_contig=False) == _STRICT_BF16


# ---- M1 family loose policy -------------------------------------------------

@pytest.mark.parametrize("brand", [
    "Apple M1",
    "Apple M1 Pro",
    "Apple M1 Max",
    "Apple M1 Ultra",
])
def test_m1_family_is_loose(force_chip, brand):
    force_chip(brand)
    assert cross_impl_atol(torch.bfloat16) == _LOOSE_M1_BF16
    assert cross_impl_atol(torch.float16) == _LOOSE_M1_BF16


def test_m1_non_contig_does_not_drop_below_loose(force_chip):
    """The non_contig floor (1e-2) is strictly less than the M1 loose value
    (5e-2). max() picks the chip-loose value; non_contig must not narrow it."""
    force_chip("Apple M1")
    assert cross_impl_atol(torch.bfloat16, non_contig=True) == _LOOSE_M1_BF16


def test_m1_regex_does_not_match_future_m11(force_chip):
    """The M1 pattern is anchored so a hypothetical 'Apple M11' would not
    be silently classified as M1 family."""
    force_chip("Apple M11")
    assert cross_impl_atol(torch.bfloat16) == _STRICT_BF16


# ---- fp32 + dtype validation ------------------------------------------------

@pytest.mark.parametrize("brand", [
    "unknown", "Apple M1", "Apple M1 Max", "Apple M3 Max", "Apple M4",
])
def test_fp32_always_strict(force_chip, brand):
    """fp32 has enough mantissa precision that intermediate-rounding ULPs
    stay well within 5e-6; no chip family observed to need looser fp32."""
    force_chip(brand)
    assert cross_impl_atol(torch.float32) == _STRICT_FP32


def test_unsupported_dtype_raises(force_chip):
    force_chip("unknown")
    with pytest.raises(ValueError, match="unsupported dtype"):
        cross_impl_atol(torch.int32)
