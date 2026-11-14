# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for SizeMapping.synchronizerSizePerWG calculation.

Tests the three-part formula at Contractions.py:839-841:

    synchronizerSizePerWG = ceil(
        (MIWaveTile[0]*MIWaveTile[1] if EnableMatrixInstruction else ThreadTile0*ThreadTile1)
        * (ceil(NumElementsPerThread / NumElementsPerBatchStore) if NumElementsPerBatchStore != 0 else NumElementsPerThread)
        * ceil(NumThreads / WavefrontSize)
    )

Coverage gaps addressed:
1. Non-multiple NumElementsPerThread / NumElementsPerBatchStore (e.g., 6/4 = 1.5 → ceil = 2)
2. EnableMatrixInstruction=True path using MIWaveTile dimensions
3. EnableMatrixInstruction=False path using ThreadTile dimensions
4. NumElementsPerBatchStore=0 path using NumElementsPerThread directly
5. Non-multiple NumThreads / WavefrontSize for wave count

All existing tests either:
- Use NumElementsPerBatchStore=0 (skipping the ceil branch)
- Never assert synchronizerSizePerWG value
- Only check isinstance(dict) on the state

This test locks down the arithmetic to prevent silent bugs in synchronizer sizing.
"""

import pytest
from tensilelite.Contractions import SizeMapping
from math import ceil


def minimal_size_mapping_state():
    """Base state dict from test_streamk_force_dp_only.py with all required keys."""
    return {
        "_GlobalAccumulation": "PartialsBuffer",
        "_WorkspaceSizePerElemBias": 0,
        "_WorkspaceSizePerElemC": 4,
        "_staggerStrideShift": 0,
        "ActivationFused": True,
        "AdaptiveGemmGSUA": 0,
        "ClusterDim": [1, 1],
        "CUOccupancy": 1,
        "CustomKernelName": "",
        "DepthU": 64,
        "DirectToLdsA": False,
        "DirectToLdsB": False,
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "EnableMatrixInstruction": True,
        "ExpertSchedulingMode": 0,
        "GlobalReadVectorWidthA": 1,
        "GlobalReadVectorWidthB": 1,
        "GlobalSplitU": 0,
        "GlobalSplitUCoalesced": False,
        "GlobalSplitUWorkGroupMappingRoundRobin": False,
        "KernelLanguage": "Assembly",
        "LocalSplitU": 1,
        "MIWaveGroup": [2, 2],
        "MIWaveTile": [2, 2],
        "MacroTile0": 128,
        "MacroTile1": 128,
        "MagicDivAlg": 1,
        "MathClocksUnrolledLoop": 0,
        "MatrixInstruction": [16, 16, 32, 1],
        "NonTemporalA": 0,
        "NonTemporalB": 0,
        "NonTemporalD": 0,
        "NumElementsPerBatchStore": 0,
        "NumElementsPerThread": 1,
        "NumLoadsCoalescedA": 1,
        "NumLoadsCoalescedB": 1,
        "NumThreads": 256,
        "PrefetchGlobalRead": 1,
        "SpaceFillingAlgo": [],
        "StaggerU": 0,
        "StaggerUMapping": 0,
        "StoreVectorWidth": 1,
        "StreamK": 3,
        "StreamKAtomic": 0,
        "ThreadTile": [1, 1],
        "ThreadTile0": 1,
        "ThreadTile1": 1,
        "UnrollLoopSwapGlobalReadOrder": 0,
        "UseCustomMainLoopSchedule": 0,
        "VectorWidthA": 1,
        "VectorWidthB": 1,
        "WaveSeparateGlobalReadA": False,
        "WaveSeparateGlobalReadB": False,
        "WavefrontSize": 64,
        "WorkGroup": [16, 16, 1],
        "WorkGroupMapping": 1,
        "WorkGroupMappingXCC": 0,
        "WorkGroupMappingXCCGroup": 0,
    }


# ---------------------------------------------------------------------------
# Matrix Instruction (MI) path tests
# ---------------------------------------------------------------------------

def test_synchronizer_size_with_non_multiple_batch_store():
    """Non-multiple NumElementsPerThread/NumElementsPerBatchStore uses ceil (6/4 → 2)."""
    state = minimal_size_mapping_state()
    state["EnableMatrixInstruction"] = True
    state["MIWaveTile"] = [2, 2]
    state["NumElementsPerThread"] = 6
    state["NumElementsPerBatchStore"] = 4
    state["NumThreads"] = 96
    state["WavefrontSize"] = 64

    # Formula: ceil(MIWaveTile[0]*MIWaveTile[1] * ceil(6/4) * ceil(96/64))
    #        = ceil(2*2 * ceil(1.5) * ceil(1.5))
    #        = ceil(4 * 2 * 2)
    #        = 16
    sm = SizeMapping.FromOriginalState(state)
    assert sm.synchronizerSizePerWG == 16, \
        f"Expected 16 for non-multiple batch store, got {sm.synchronizerSizePerWG}"


def test_synchronizer_size_with_exact_batch_store_divisor():
    """Exact divisor NumElementsPerThread/NumElementsPerBatchStore (8/4 → 2)."""
    state = minimal_size_mapping_state()
    state["EnableMatrixInstruction"] = True
    state["MIWaveTile"] = [2, 2]
    state["NumElementsPerThread"] = 8
    state["NumElementsPerBatchStore"] = 4
    state["NumThreads"] = 128
    state["WavefrontSize"] = 64

    # Formula: ceil(2*2 * ceil(8/4) * ceil(128/64))
    #        = ceil(4 * 2 * 2)
    #        = 16
    sm = SizeMapping.FromOriginalState(state)
    assert sm.synchronizerSizePerWG == 16


def test_synchronizer_size_with_batch_store_zero():
    """NumElementsPerBatchStore=0 uses NumElementsPerThread directly (no ceil division)."""
    state = minimal_size_mapping_state()
    state["EnableMatrixInstruction"] = True
    state["MIWaveTile"] = [3, 3]
    state["NumElementsPerThread"] = 5
    state["NumElementsPerBatchStore"] = 0  # Triggers else branch
    state["NumThreads"] = 192
    state["WavefrontSize"] = 64

    # Formula: ceil(3*3 * 5 * ceil(192/64))
    #        = ceil(9 * 5 * 3)
    #        = 135
    sm = SizeMapping.FromOriginalState(state)
    assert sm.synchronizerSizePerWG == 135


def test_synchronizer_size_with_asymmetric_mi_wave_tile():
    """Asymmetric MIWaveTile dimensions (1x4 vs 4x1)."""
    state = minimal_size_mapping_state()
    state["EnableMatrixInstruction"] = True
    state["MIWaveTile"] = [1, 4]
    state["NumElementsPerThread"] = 2
    state["NumElementsPerBatchStore"] = 0
    state["NumThreads"] = 256
    state["WavefrontSize"] = 64

    # Formula: ceil(1*4 * 2 * ceil(256/64))
    #        = ceil(4 * 2 * 4)
    #        = 32
    sm = SizeMapping.FromOriginalState(state)
    assert sm.synchronizerSizePerWG == 32


def test_synchronizer_size_with_non_multiple_wave_count():
    """Non-multiple NumThreads/WavefrontSize for wave count (100/64 → 2)."""
    state = minimal_size_mapping_state()
    state["EnableMatrixInstruction"] = True
    state["MIWaveTile"] = [2, 2]
    state["NumElementsPerThread"] = 3
    state["NumElementsPerBatchStore"] = 0
    state["NumThreads"] = 100  # ceil(100/64) = 2
    state["WavefrontSize"] = 64

    # Formula: ceil(2*2 * 3 * ceil(100/64))
    #        = ceil(4 * 3 * 2)
    #        = 24
    sm = SizeMapping.FromOriginalState(state)
    assert sm.synchronizerSizePerWG == 24


# ---------------------------------------------------------------------------
# Non-MI (ThreadTile) path tests
# ---------------------------------------------------------------------------

def test_synchronizer_size_without_matrix_instruction():
    """EnableMatrixInstruction=False uses ThreadTile0*ThreadTile1 instead of MIWaveTile."""
    state = minimal_size_mapping_state()
    state["EnableMatrixInstruction"] = False
    state["ThreadTile0"] = 4
    state["ThreadTile1"] = 8
    state["NumElementsPerThread"] = 2
    state["NumElementsPerBatchStore"] = 0
    state["NumThreads"] = 128
    state["WavefrontSize"] = 64

    # Formula: ceil(4*8 * 2 * ceil(128/64))
    #        = ceil(32 * 2 * 2)
    #        = 128
    sm = SizeMapping.FromOriginalState(state)
    assert sm.synchronizerSizePerWG == 128


def test_synchronizer_size_non_mi_with_batch_store():
    """Non-MI path with non-zero NumElementsPerBatchStore."""
    state = minimal_size_mapping_state()
    state["EnableMatrixInstruction"] = False
    state["ThreadTile0"] = 2
    state["ThreadTile1"] = 3
    state["NumElementsPerThread"] = 7
    state["NumElementsPerBatchStore"] = 3  # ceil(7/3) = 3
    state["NumThreads"] = 96
    state["WavefrontSize"] = 64

    # Formula: ceil(2*3 * ceil(7/3) * ceil(96/64))
    #        = ceil(6 * 3 * 2)
    #        = 36
    sm = SizeMapping.FromOriginalState(state)
    assert sm.synchronizerSizePerWG == 36


def test_synchronizer_size_non_mi_asymmetric_thread_tile():
    """Non-MI with asymmetric ThreadTile (1x12)."""
    state = minimal_size_mapping_state()
    state["EnableMatrixInstruction"] = False
    state["ThreadTile0"] = 1
    state["ThreadTile1"] = 12
    state["NumElementsPerThread"] = 4
    state["NumElementsPerBatchStore"] = 0
    state["NumThreads"] = 64
    state["WavefrontSize"] = 64

    # Formula: ceil(1*12 * 4 * ceil(64/64))
    #        = ceil(12 * 4 * 1)
    #        = 48
    sm = SizeMapping.FromOriginalState(state)
    assert sm.synchronizerSizePerWG == 48


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_synchronizer_size_minimal_values():
    """All dimensions at minimum (1x1 tile, 1 element, 1 wave)."""
    state = minimal_size_mapping_state()
    state["EnableMatrixInstruction"] = True
    state["MIWaveTile"] = [1, 1]
    state["NumElementsPerThread"] = 1
    state["NumElementsPerBatchStore"] = 0
    state["NumThreads"] = 64
    state["WavefrontSize"] = 64

    # Formula: ceil(1*1 * 1 * ceil(64/64))
    #        = ceil(1 * 1 * 1)
    #        = 1
    sm = SizeMapping.FromOriginalState(state)
    assert sm.synchronizerSizePerWG == 1


def test_synchronizer_size_large_values():
    """Large dimensions to verify no overflow/truncation."""
    state = minimal_size_mapping_state()
    state["EnableMatrixInstruction"] = True
    state["MIWaveTile"] = [8, 8]
    state["NumElementsPerThread"] = 16
    state["NumElementsPerBatchStore"] = 0
    state["NumThreads"] = 512
    state["WavefrontSize"] = 64

    # Formula: ceil(8*8 * 16 * ceil(512/64))
    #        = ceil(64 * 16 * 8)
    #        = 8192
    sm = SizeMapping.FromOriginalState(state)
    assert sm.synchronizerSizePerWG == 8192


def test_synchronizer_size_batch_store_equals_one():
    """NumElementsPerBatchStore=1 (every element is a batch)."""
    state = minimal_size_mapping_state()
    state["EnableMatrixInstruction"] = True
    state["MIWaveTile"] = [2, 2]
    state["NumElementsPerThread"] = 5
    state["NumElementsPerBatchStore"] = 1  # ceil(5/1) = 5
    state["NumThreads"] = 64
    state["WavefrontSize"] = 64

    # Formula: ceil(2*2 * ceil(5/1) * ceil(64/64))
    #        = ceil(4 * 5 * 1)
    #        = 20
    sm = SizeMapping.FromOriginalState(state)
    assert sm.synchronizerSizePerWG == 20


# ---------------------------------------------------------------------------
# Parametrized comprehensive test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "enable_mi,mi_wave_tile,thread_tile,num_elems,batch_store,num_threads,wave_size,expected",
    [
        # MI path: non-multiple batch store (6/4 → 2)
        (True, [2, 2], [1, 1], 6, 4, 96, 64, 16),
        # MI path: exact divisor (8/4 → 2)
        (True, [2, 2], [1, 1], 8, 4, 128, 64, 16),
        # MI path: batch_store=0 uses NumElementsPerThread
        (True, [3, 3], [1, 1], 5, 0, 192, 64, 135),
        # MI path: asymmetric MIWaveTile
        (True, [1, 4], [1, 1], 2, 0, 256, 64, 32),
        # MI path: non-multiple wave count (100/64 → 2)
        (True, [2, 2], [1, 1], 3, 0, 100, 64, 24),
        # Non-MI path: ThreadTile0*ThreadTile1
        (False, [2, 2], [4, 8], 2, 0, 128, 64, 128),
        # Non-MI path: with batch store
        (False, [2, 2], [2, 3], 7, 3, 96, 64, 36),
        # Non-MI path: asymmetric ThreadTile
        (False, [2, 2], [1, 12], 4, 0, 64, 64, 48),
        # Edge: minimal all 1s
        (True, [1, 1], [1, 1], 1, 0, 64, 64, 1),
        # Edge: large values
        (True, [8, 8], [1, 1], 16, 0, 512, 64, 8192),
        # Edge: batch_store=1
        (True, [2, 2], [1, 1], 5, 1, 64, 64, 20),
    ],
)
def test_synchronizer_size_parametrized(
    enable_mi, mi_wave_tile, thread_tile, num_elems, batch_store, num_threads, wave_size, expected
):
    """Parametrized test covering all branches and edge cases."""
    state = minimal_size_mapping_state()
    state["EnableMatrixInstruction"] = enable_mi
    state["MIWaveTile"] = mi_wave_tile
    state["ThreadTile0"] = thread_tile[0]
    state["ThreadTile1"] = thread_tile[1]
    state["NumElementsPerThread"] = num_elems
    state["NumElementsPerBatchStore"] = batch_store
    state["NumThreads"] = num_threads
    state["WavefrontSize"] = wave_size

    sm = SizeMapping.FromOriginalState(state)
    assert sm.synchronizerSizePerWG == expected, (
        f"EnableMI={enable_mi}, MIWaveTile={mi_wave_tile}, ThreadTile={thread_tile}, "
        f"NumElems={num_elems}, BatchStore={batch_store}, NumThreads={num_threads}, "
        f"WaveSize={wave_size}: expected {expected}, got {sm.synchronizerSizePerWG}"
    )


# ---------------------------------------------------------------------------
# Manual oracle verification test
# ---------------------------------------------------------------------------

def test_synchronizer_size_hand_computed_oracle():
    """Hand-computed oracle from agent's gap report.

    NumElementsPerThread=6, NumElementsPerBatchStore=4, MIWaveTile=[2,2],
    NumThreads=96, WavefrontSize=64.

    Expected: 4*2*2 = 16

    Breakdown:
    - MIWaveTile: 2*2 = 4
    - BatchStore ceil: ceil(6/4) = ceil(1.5) = 2
    - Wave count ceil: ceil(96/64) = ceil(1.5) = 2
    - Product: 4 * 2 * 2 = 16
    """
    state = minimal_size_mapping_state()
    state["EnableMatrixInstruction"] = True
    state["MIWaveTile"] = [2, 2]
    state["NumElementsPerThread"] = 6
    state["NumElementsPerBatchStore"] = 4
    state["NumThreads"] = 96
    state["WavefrontSize"] = 64

    sm = SizeMapping.FromOriginalState(state)

    # Verify each component of the formula
    tile_factor = 2 * 2  # MIWaveTile[0] * MIWaveTile[1]
    batch_ceil = ceil(6 / 4)  # ceil(1.5) = 2
    wave_ceil = ceil(96 / 64)  # ceil(1.5) = 2

    assert tile_factor == 4
    assert batch_ceil == 2
    assert wave_ceil == 2
    assert sm.synchronizerSizePerWG == 16
