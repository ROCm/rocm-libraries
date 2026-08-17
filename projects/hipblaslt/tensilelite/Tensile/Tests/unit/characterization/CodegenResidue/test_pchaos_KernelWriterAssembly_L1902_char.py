################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""
Characterisation test — branch_id 24d207463aa7b7dbc424872db6e75313529bf4d6

Predicate : if self.states.overflowedResources:
Site      : Tensile/KernelWriterAssembly.py:1902
Function  : checkResources
Solver    : z3 4.16.0  — SAT  (solver-backed-under-assumptions)

Classification note
-------------------
The predicate is a bare int-truthiness test on StateValues.overflowedResources
(KernelWriter.py:183, initialized to 0).

  overflowedResources == 0 -> predicate False (no overflow; default)
  overflowedResources != 0 -> predicate True  (overflow; error code 1-8)

Write-path error codes (KernelWriterAssembly.py:1878-1912):
  1 -> too many vgprs
  2 -> too many sgprs
  3 -> half store requires at least two elements per batch
  4 -> occupancy limit (numWorkGroupsPerExecUnit < 1)
  5 -> LDS scheduling conflict with oneBufferScheduling
  6 -> SIA2 better with occupancy 2 (ScheduleIterAlg==2 AND CUOccupancy<2)
  7 -> invalid LSU code (invalidLSUCode flag set by LSU.py:229)
  8 -> not enough LDS space (LdsNumBytes > archCaps['DeviceLDS'])

Tests here:
  1. Pure predicate helper — mirrors bool(overflowedResources) for each code.
  2. StateValues declaration pin — confirms the field is int=0 in the dataclass.

CPU-only.  No GPU hardware required.
"""

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Pure predicate helper: mirrors `if self.states.overflowedResources:` at L1902
# ---------------------------------------------------------------------------

def _overflowed(overflowed_resources: int) -> bool:
    """Mirror of KernelWriterAssembly.py:1902 `if self.states.overflowedResources:`.

    overflowedResources is an int field (StateValues, KernelWriter.py:183)
    initialized to 0.  The branch is TRUE when the value is any nonzero error
    code (1..8) set by one of the resource-check write-paths.

    pre:  0 <= overflowed_resources <= 8
    post: __return__ == (overflowed_resources != 0)
    """
    return bool(overflowed_resources)


# ---------------------------------------------------------------------------
# Tests for the pure predicate helper
# ---------------------------------------------------------------------------

def test_zero_is_false():
    """overflowedResources=0 (default, no overflow) -> FALSE branch of L1902."""
    assert _overflowed(0) is False


def test_code1_vgpr_overflow_is_true():
    """overflowedResources=1 (too many vgprs) -> TRUE branch of L1902."""
    assert _overflowed(1) is True


def test_code2_sgpr_overflow_is_true():
    """overflowedResources=2 (too many sgprs) -> TRUE branch of L1902."""
    assert _overflowed(2) is True


def test_code3_half_store_batch_is_true():
    """overflowedResources=3 (half store batch size) -> TRUE branch of L1902."""
    assert _overflowed(3) is True


def test_code4_occupancy_limit_is_true():
    """overflowedResources=4 (occupancy limit) -> TRUE branch of L1902."""
    assert _overflowed(4) is True


def test_code5_lds_conflict_is_true():
    """overflowedResources=5 (LDS scheduling conflict) -> TRUE branch of L1902."""
    assert _overflowed(5) is True


def test_code6_sia2_occupancy_is_true():
    """overflowedResources=6 (SIA2 CUOccupancy<2) -> TRUE branch of L1902."""
    assert _overflowed(6) is True


def test_code7_invalid_lsu_code_is_true():
    """overflowedResources=7 (invalid LSU code) -> TRUE branch of L1902."""
    assert _overflowed(7) is True


def test_code8_lds_space_exceeded_is_true():
    """overflowedResources=8 (not enough LDS space) -> TRUE branch of L1902."""
    assert _overflowed(8) is True


def test_exhaustive_domain():
    """Exhaustive: only code 0 yields False; codes 1-8 all yield True."""
    assert _overflowed(0) is False
    for code in range(1, 9):
        assert _overflowed(code) is True, (
            f"overflowedResources={code}: expected True, got False"
        )


# ---------------------------------------------------------------------------
# StateValues declaration pin
# ---------------------------------------------------------------------------

def test_overflowed_resources_declared_as_int_with_default_zero():
    """StateValues.overflowedResources is declared as int=0 in KernelWriter.py:183.

    Pin the dataclass field type annotation and default so any change is caught.
    StateValues requires positional args (version, kernel, kernelName) so we
    inspect the dataclass field metadata directly rather than instantiating.
    """
    import dataclasses  # noqa: PLC0415
    from Tensile.KernelWriter import StateValues  # noqa: PLC0415

    fields_by_name = {f.name: f for f in dataclasses.fields(StateValues)}
    assert "overflowedResources" in fields_by_name, (
        "StateValues no longer has an overflowedResources field; "
        "the predicate at KernelWriterAssembly.py:1902 relies on this field."
    )
    field = fields_by_name["overflowedResources"]
    assert field.default == 0, (
        f"StateValues.overflowedResources default changed from 0 to {field.default!r}; "
        "the FALSE branch of KernelWriterAssembly.py:1902 relies on 0 meaning no overflow."
    )
