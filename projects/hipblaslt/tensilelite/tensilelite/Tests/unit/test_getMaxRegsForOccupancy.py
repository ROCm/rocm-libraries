# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for getMaxRegsForOccupancy register growth logic.

Tests the critical VGPR growth loop that maximizes register usage while
preserving occupancy. With vgprs not a multiple of the 4-VGPR (or 8-VGPR
for doubleVgpr) allocation granule, getMaxRegsForOccupancy grows vgprs
to the largest value that keeps the initial occupancy.

The function must handle:
1. 4-VGPR and 8-VGPR allocation granules
2. Zero-occupancy fallback (initOccupancy==0 → return (vgprs, 1))
3. MaxVgpr limit boundary
4. Occupancy preservation during growth

An off-by-one error at a granule or occupancy boundary would silently
produce wrong register counts, leading to suboptimal or invalid kernels.
"""

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock
from math import ceil

pytestmark = pytest.mark.unit


def _make_writer(max_vgpr=256, max_sgpr=102, max_waves_per_simd=10, device_lds=65536):
    """Create mock KernelWriterAssembly with occupancy calculation."""
    writer = MagicMock()
    writer.states = SimpleNamespace(
        regCaps={
            "MaxVgpr": max_vgpr,
            "PhysicalMaxSgpr": max_sgpr,
        },
        archCaps={
            "MaxWavesPerSimd": max_waves_per_simd,
            "DeviceLDS": device_lds,
        }
    )

    # Implement real getVgprOccupancy logic
    def getVgprOccupancy(numThreads, vgprs, doubleVgpr=False):
        multiplier = int(ceil(max(numThreads, 256) / 256.0))
        maxOccupancy = max_waves_per_simd // multiplier

        vgprAllocateAligned = 4 if not doubleVgpr else 8
        totalVgprs = max_vgpr if not doubleVgpr else max_vgpr * 2
        vgprsAligned = int(ceil(vgprs / vgprAllocateAligned)) * vgprAllocateAligned
        vgprsAligned *= multiplier

        if vgprsAligned > totalVgprs:
            return 0
        elif vgprsAligned < 1:
            return maxOccupancy
        occupancy = min(totalVgprs // vgprsAligned, maxOccupancy)

        return occupancy

    writer.getVgprOccupancy = getVgprOccupancy

    # Implement simplified getOccupancy (just returns VGPR occupancy for these tests)
    def getOccupancy(numThreads, vgprs, sgprs, ldsSize, accvgprs=0, doubleVgpr=False):
        # For simplicity, assume VGPR is the binding constraint (no SGPR/LDS limits)
        considerAccVgprs = 0 if not doubleVgpr else accvgprs
        return getVgprOccupancy(numThreads, vgprs + considerAccVgprs, doubleVgpr)

    writer.getOccupancy = getOccupancy

    return writer


# ---------------------------------------------------------------------------
# Basic VGPR growth tests (4-VGPR granule)
# ---------------------------------------------------------------------------

def test_vgpr_growth_to_granule_boundary():
    """vgprs=41 should grow to 44 (next 4-VGPR boundary) if occupancy permits."""
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    writer = _make_writer(max_vgpr=256)

    # 41 VGPRs: aligned to 44 (ceil(41/4)*4 = 11*4 = 44)
    # With 256 total, occupancy = 256/44 = 5 waves
    # Growing to 44, 45, 46, 47 all give same occupancy (5 waves)
    # At 48, occupancy stays 5 (256/48 = 5)
    # So maxVgprs should grow until occupancy drops

    maxVgprs, occupancy = KernelWriterAssembly.getMaxRegsForOccupancy(
        writer, numThreads=256, vgprs=41, sgprs=32, ldsSize=0, doubleVgpr=False
    )

    # Should grow to at least 44 (4-VGPR boundary)
    assert maxVgprs >= 44, f"Expected vgprs >= 44, got {maxVgprs}"
    assert occupancy > 0, f"Expected positive occupancy, got {occupancy}"


def test_vgpr_at_granule_boundary_no_growth_needed():
    """vgprs=44 is already at 4-VGPR boundary, but can still grow if occupancy permits."""
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    writer = _make_writer(max_vgpr=256)

    maxVgprs, occupancy = KernelWriterAssembly.getMaxRegsForOccupancy(
        writer, numThreads=256, vgprs=44, sgprs=32, ldsSize=0, doubleVgpr=False
    )

    # 44 is already aligned, should be >= 44
    assert maxVgprs >= 44, f"Expected vgprs >= 44, got {maxVgprs}"
    assert occupancy > 0, f"Expected positive occupancy, got {occupancy}"


def test_vgpr_growth_stops_at_occupancy_drop():
    """Growth stops when adding more VGPRs would reduce occupancy."""
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    writer = _make_writer(max_vgpr=256)

    # With 256 total VGPRs, occupancy drops at certain boundaries:
    # 52 VGPRs: 256/52 = 4.9 → 4 waves
    # 64 VGPRs: 256/64 = 4 waves
    # 65 VGPRs: ceil(65/4)*4 = 68, 256/68 = 3.7 → 3 waves (drops!)

    # Start at 50 (occupancy = 5 waves: 256/52 aligned to 5)
    maxVgprs, occupancy = KernelWriterAssembly.getMaxRegsForOccupancy(
        writer, numThreads=256, vgprs=50, sgprs=32, ldsSize=0, doubleVgpr=False
    )

    initOccupancy = writer.getOccupancy(256, 50, 32, 0, 0, False)

    # Should not grow beyond where occupancy drops
    finalOccupancy = writer.getVgprOccupancy(256, maxVgprs, False)
    assert finalOccupancy >= initOccupancy, \
        f"Final occupancy {finalOccupancy} < initial {initOccupancy}"


def test_vgpr_growth_stops_at_max_vgpr():
    """Growth stops at MaxVgpr limit even if occupancy permits more."""
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    writer = _make_writer(max_vgpr=128)  # Lower limit

    # Start at 120, should not exceed 128
    maxVgprs, occupancy = KernelWriterAssembly.getMaxRegsForOccupancy(
        writer, numThreads=256, vgprs=120, sgprs=32, ldsSize=0, doubleVgpr=False
    )

    assert maxVgprs <= 128, f"Expected maxVgprs <= 128, got {maxVgprs}"


# ---------------------------------------------------------------------------
# doubleVgpr mode (8-VGPR granule)
# ---------------------------------------------------------------------------

def test_vgpr_growth_double_vgpr_8_granule():
    """With doubleVgpr=True, alignment is 8-VGPR granule, not 4."""
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    writer = _make_writer(max_vgpr=256)

    # 41 VGPRs with doubleVgpr: aligned to 48 (ceil(41/8)*8 = 6*8 = 48)
    maxVgprs, occupancy = KernelWriterAssembly.getMaxRegsForOccupancy(
        writer, numThreads=256, vgprs=41, sgprs=32, ldsSize=0, doubleVgpr=True
    )

    # Should grow to at least 48 (8-VGPR boundary)
    assert maxVgprs >= 48, f"Expected vgprs >= 48 with doubleVgpr, got {maxVgprs}"
    assert occupancy > 0, f"Expected positive occupancy, got {occupancy}"


def test_vgpr_growth_double_vgpr_at_boundary():
    """With doubleVgpr=True, vgprs=48 is at 8-VGPR boundary."""
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    writer = _make_writer(max_vgpr=256)

    maxVgprs, occupancy = KernelWriterAssembly.getMaxRegsForOccupancy(
        writer, numThreads=256, vgprs=48, sgprs=32, ldsSize=0, doubleVgpr=True
    )

    # 48 is already 8-VGPR aligned
    assert maxVgprs >= 48, f"Expected vgprs >= 48, got {maxVgprs}"


def test_double_vgpr_uses_2x_total_vgprs():
    """With doubleVgpr=True, totalVgprs = MaxVgpr * 2."""
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    writer = _make_writer(max_vgpr=256)

    # With doubleVgpr, totalVgprs = 512, so more room to grow
    maxVgprs, occupancy = KernelWriterAssembly.getMaxRegsForOccupancy(
        writer, numThreads=256, vgprs=200, sgprs=32, ldsSize=0, doubleVgpr=True
    )

    # Should grow to the logical limit (MaxVgpr=256) in doubled mode
    # Loop condition: vgprs < MaxVgpr, so max is 256
    # Initial occupancy with vgprs=200: 512/200 = 2
    assert maxVgprs == 256, f"Expected maxVgprs=256 (logical limit in doubled mode), got {maxVgprs}"
    assert occupancy == 2, f"Expected occupancy=2 (preserved during growth), got {occupancy}"


# ---------------------------------------------------------------------------
# Zero occupancy fallback
# ---------------------------------------------------------------------------

def test_zero_occupancy_returns_vgprs_and_1():
    """When initial occupancy is 0, return (vgprs, 1) immediately."""
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    writer = _make_writer(max_vgpr=256)

    # Request 300 VGPRs (above MaxVgpr=256) → occupancy = 0
    maxVgprs, occupancy = KernelWriterAssembly.getMaxRegsForOccupancy(
        writer, numThreads=256, vgprs=300, sgprs=32, ldsSize=0, doubleVgpr=False
    )

    # Should return original vgprs and occupancy=1
    assert maxVgprs == 300, f"Expected maxVgprs=300 (unchanged), got {maxVgprs}"
    assert occupancy == 1, f"Expected occupancy=1 for zero-occupancy case, got {occupancy}"


def test_zero_occupancy_with_double_vgpr():
    """Zero occupancy fallback works with doubleVgpr=True."""
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    writer = _make_writer(max_vgpr=256)

    # Request 600 VGPRs (above totalVgprs=512) → occupancy = 0
    maxVgprs, occupancy = KernelWriterAssembly.getMaxRegsForOccupancy(
        writer, numThreads=256, vgprs=600, sgprs=32, ldsSize=0, doubleVgpr=True
    )

    assert maxVgprs == 600, f"Expected maxVgprs=600 (unchanged), got {maxVgprs}"
    assert occupancy == 1, f"Expected occupancy=1, got {occupancy}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_vgprs_equal_to_max_vgpr():
    """When vgprs == MaxVgpr, no growth possible."""
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    writer = _make_writer(max_vgpr=256)

    maxVgprs, occupancy = KernelWriterAssembly.getMaxRegsForOccupancy(
        writer, numThreads=256, vgprs=256, sgprs=32, ldsSize=0, doubleVgpr=False
    )

    # Should not grow beyond MaxVgpr
    assert maxVgprs == 256, f"Expected maxVgprs=256, got {maxVgprs}"


def test_small_vgprs_can_grow_significantly():
    """Small vgprs value should grow to maximize occupancy."""
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    writer = _make_writer(max_vgpr=256)

    # Start at 10 VGPRs, lots of room to grow
    maxVgprs, occupancy = KernelWriterAssembly.getMaxRegsForOccupancy(
        writer, numThreads=256, vgprs=10, sgprs=32, ldsSize=0, doubleVgpr=False
    )

    # Should grow significantly (10 → at least to next occupancy boundary)
    assert maxVgprs >= 10, f"Expected growth, got {maxVgprs}"

    initOccupancy = writer.getOccupancy(256, 10, 32, 0, 0, False)
    finalOccupancy = writer.getVgprOccupancy(256, maxVgprs, False)
    assert finalOccupancy >= initOccupancy, \
        f"Final occupancy {finalOccupancy} < initial {initOccupancy}"


def test_multiple_workgroup_sizes():
    """Test with different numThreads values (affects multiplier)."""
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    writer = _make_writer(max_vgpr=256)

    # numThreads=512 → multiplier=2, different occupancy calculation
    maxVgprs_512, occupancy_512 = KernelWriterAssembly.getMaxRegsForOccupancy(
        writer, numThreads=512, vgprs=41, sgprs=32, ldsSize=0, doubleVgpr=False
    )

    # numThreads=256 → multiplier=1
    maxVgprs_256, occupancy_256 = KernelWriterAssembly.getMaxRegsForOccupancy(
        writer, numThreads=256, vgprs=41, sgprs=32, ldsSize=0, doubleVgpr=False
    )

    # Both should grow, but may differ due to multiplier effect
    assert maxVgprs_512 >= 41, f"Expected growth for 512 threads, got {maxVgprs_512}"
    assert maxVgprs_256 >= 41, f"Expected growth for 256 threads, got {maxVgprs_256}"


# ---------------------------------------------------------------------------
# Occupancy preservation validation
# ---------------------------------------------------------------------------

def test_returned_vgprs_preserves_occupancy():
    """Verify final maxVgprs actually preserves the initial occupancy."""
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    writer = _make_writer(max_vgpr=256)

    test_vgprs = [20, 41, 63, 80, 100, 127]

    for vgprs in test_vgprs:
        maxVgprs, returnedOccupancy = KernelWriterAssembly.getMaxRegsForOccupancy(
            writer, numThreads=256, vgprs=vgprs, sgprs=32, ldsSize=0, doubleVgpr=False
        )

        # Compute initial occupancy
        initOccupancy = writer.getOccupancy(256, vgprs, 32, 0, 0, False)

        # Compute final occupancy with returned maxVgprs
        finalOccupancy = writer.getVgprOccupancy(256, maxVgprs, False)

        assert finalOccupancy >= initOccupancy, \
            f"vgprs={vgprs}: final occupancy {finalOccupancy} < initial {initOccupancy}"

        assert returnedOccupancy == initOccupancy, \
            f"vgprs={vgprs}: returned occupancy {returnedOccupancy} != initial {initOccupancy}"


def test_growth_stops_one_vgpr_before_occupancy_drop():
    """Verify growth stops exactly at the last VGPR that preserves occupancy."""
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    writer = _make_writer(max_vgpr=256)

    maxVgprs, occupancy = KernelWriterAssembly.getMaxRegsForOccupancy(
        writer, numThreads=256, vgprs=50, sgprs=32, ldsSize=0, doubleVgpr=False
    )

    # vgprs=50 aligned to 52 (4-VGPR granule) gives occupancy=256/52=4
    # Maximum vgprs that preserves occupancy=4: ceil(vgprs/4)*4 <= 256/4=64
    # So max unaligned vgprs is 64 (aligned to 64)
    # At vgprs=65 (aligned to 68), occupancy drops to 256/68=3
    assert maxVgprs == 64, f"Expected maxVgprs=64 (max for occupancy=4), got {maxVgprs}"
    assert occupancy == 4, f"Expected occupancy=4 (preserved during growth), got {occupancy}"


# ---------------------------------------------------------------------------
# Granule boundary systematic tests
# ---------------------------------------------------------------------------

def test_granule_boundary_systematic():
    """Systematic test of 4-VGPR granule boundaries."""
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    writer = _make_writer(max_vgpr=256)

    # Test vgprs just below 4-VGPR boundaries
    test_cases = [
        (41, 44),  # 41 → aligns to 44
        (42, 44),  # 42 → aligns to 44
        (43, 44),  # 43 → aligns to 44
        (45, 48),  # 45 → aligns to 48
        (61, 64),  # 61 → aligns to 64
    ]

    for vgprs, expected_min_aligned in test_cases:
        maxVgprs, occupancy = KernelWriterAssembly.getMaxRegsForOccupancy(
            writer, numThreads=256, vgprs=vgprs, sgprs=32, ldsSize=0, doubleVgpr=False
        )

        # Should grow to at least the 4-VGPR boundary
        assert maxVgprs >= expected_min_aligned, \
            f"vgprs={vgprs}: expected >= {expected_min_aligned}, got {maxVgprs}"


# ---------------------------------------------------------------------------
# doubleVgpr 8-VGPR boundary systematic tests
# ---------------------------------------------------------------------------

def test_double_vgpr_8_boundary_vgprs_1():
    """With doubleVgpr=True, vgprs=1 aligns to 8."""
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    writer = _make_writer(max_vgpr=256)

    maxVgprs, occupancy = KernelWriterAssembly.getMaxRegsForOccupancy(
        writer, numThreads=256, vgprs=1, sgprs=16, ldsSize=0, doubleVgpr=True
    )

    # vgprs=1: ceil(1/8)*8 = 8
    # With doubleVgpr=True, should grow to at least 8
    assert maxVgprs >= 8, f"Expected vgprs >= 8 with doubleVgpr=True, got {maxVgprs}"
    assert occupancy > 0, f"Expected positive occupancy, got {occupancy}"


def test_double_vgpr_8_boundary_vgprs_9():
    """With doubleVgpr=True, vgprs=9 aligns to 16."""
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    writer = _make_writer(max_vgpr=256)

    maxVgprs, occupancy = KernelWriterAssembly.getMaxRegsForOccupancy(
        writer, numThreads=256, vgprs=9, sgprs=16, ldsSize=0, doubleVgpr=True
    )

    # vgprs=9: ceil(9/8)*8 = 2*8 = 16
    assert maxVgprs >= 16, f"Expected vgprs >= 16 with doubleVgpr=True, got {maxVgprs}"
    assert occupancy > 0, f"Expected positive occupancy, got {occupancy}"


def test_double_vgpr_8_boundary_vgprs_15():
    """With doubleVgpr=True, vgprs=15 aligns to 16."""
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    writer = _make_writer(max_vgpr=256)

    maxVgprs, occupancy = KernelWriterAssembly.getMaxRegsForOccupancy(
        writer, numThreads=256, vgprs=15, sgprs=16, ldsSize=0, doubleVgpr=True
    )

    # vgprs=15: ceil(15/8)*8 = 2*8 = 16
    assert maxVgprs >= 16, f"Expected vgprs >= 16 with doubleVgpr=True, got {maxVgprs}"
    assert occupancy > 0, f"Expected positive occupancy, got {occupancy}"


def test_double_vgpr_8_boundary_vgprs_16_aligned():
    """With doubleVgpr=True, vgprs=16 is already 8-aligned."""
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    writer = _make_writer(max_vgpr=256)

    maxVgprs, occupancy = KernelWriterAssembly.getMaxRegsForOccupancy(
        writer, numThreads=256, vgprs=16, sgprs=16, ldsSize=0, doubleVgpr=True
    )

    # vgprs=16: already 8-aligned, ceil(16/8)*8 = 16
    assert maxVgprs >= 16, f"Expected vgprs >= 16, got {maxVgprs}"
    assert occupancy > 0, f"Expected positive occupancy, got {occupancy}"


def test_double_vgpr_8_boundary_systematic():
    """Systematic test of 8-VGPR granule boundaries with doubleVgpr=True."""
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    writer = _make_writer(max_vgpr=256)

    # Test vgprs around 8-VGPR boundaries
    # Format: (vgprs, expected_min_aligned)
    test_cases = [
        (1, 8),    # 1 → aligns to 8
        (7, 8),    # 7 → aligns to 8
        (8, 8),    # 8 → already aligned
        (9, 16),   # 9 → aligns to 16
        (15, 16),  # 15 → aligns to 16
        (16, 16),  # 16 → already aligned
        (17, 24),  # 17 → aligns to 24
        (23, 24),  # 23 → aligns to 24
        (24, 24),  # 24 → already aligned
        (25, 32),  # 25 → aligns to 32
    ]

    for vgprs, expected_min_aligned in test_cases:
        maxVgprs, occupancy = KernelWriterAssembly.getMaxRegsForOccupancy(
            writer, numThreads=256, vgprs=vgprs, sgprs=16, ldsSize=0, doubleVgpr=True
        )

        # Should grow to at least the 8-VGPR boundary
        assert maxVgprs >= expected_min_aligned, \
            f"doubleVgpr=True, vgprs={vgprs}: expected >= {expected_min_aligned}, got {maxVgprs}"

        # Verify occupancy is positive
        assert occupancy > 0, \
            f"doubleVgpr=True, vgprs={vgprs}: expected positive occupancy, got {occupancy}"


def test_double_vgpr_with_accvgprs_detects_incorrect_rounding():
    """Verify 8-register rounding in growth loop when accvgprs > 0.

    With accvgprs=0, incorrect 4-register rounding in the growth loop can be
    hidden because getVgprOccupancy applies 8-register rounding again. Adding
    accvgprs=1 exposes the bug: correct 8-alignment gives (40, 10), while
    incorrect 4-alignment gives (44, 10).

    Correct 8-alignment:
      vgprs=40: aligned=40, total=41, final=48, occ=512/48=10
      vgprs=41: aligned=48, total=49, final=56, occ=512/56=9 (drops!)

    Incorrect 4-alignment (bug):
      vgprs=44: aligned=44, total=45, final=48, occ=512/48=10
      vgprs=45: aligned=48, total=49, final=56, occ=512/56=9 (drops!)
    """
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    writer = _make_writer(max_vgpr=256)

    maxVgprs, occupancy = KernelWriterAssembly.getMaxRegsForOccupancy(
        writer, numThreads=256, vgprs=1, sgprs=16, ldsSize=0,
        accvgprs=1, doubleVgpr=True
    )

    # With correct 8-alignment: max is 40 (aligned=40, +1 acc=41, final=48, occ=10)
    # With incorrect 4-alignment: max is 44 (aligned=44, +1 acc=45, final=48, occ=10)
    assert maxVgprs == 40, f"Expected maxVgprs=40 (8-aligned growth), got {maxVgprs}"
    assert occupancy == 10, f"Expected occupancy=10, got {occupancy}"


def test_double_vgpr_alignment_vs_no_alignment():
    """Compare doubleVgpr=True (8-aligned) vs False (4-aligned) for same vgprs."""
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    writer = _make_writer(max_vgpr=256)

    # vgprs=9 with doubleVgpr=True → aligns to 16
    maxVgprs_double, occ_double = KernelWriterAssembly.getMaxRegsForOccupancy(
        writer, numThreads=256, vgprs=9, sgprs=16, ldsSize=0, doubleVgpr=True
    )

    # vgprs=9 with doubleVgpr=False → aligns to 12 (4-VGPR granule)
    maxVgprs_single, occ_single = KernelWriterAssembly.getMaxRegsForOccupancy(
        writer, numThreads=256, vgprs=9, sgprs=16, ldsSize=0, doubleVgpr=False
    )

    # doubleVgpr=True should align to higher boundary (16 vs 12)
    assert maxVgprs_double >= 16, f"doubleVgpr=True: expected >= 16, got {maxVgprs_double}"
    assert maxVgprs_single >= 12, f"doubleVgpr=False: expected >= 12, got {maxVgprs_single}"

    # Both should have positive occupancy
    assert occ_double > 0 and occ_single > 0, "Both should have positive occupancy"
