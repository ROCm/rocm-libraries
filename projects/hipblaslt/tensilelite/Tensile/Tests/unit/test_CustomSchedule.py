################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

import pytest
from unittest.mock import MagicMock

from Tensile.Components.CustomSchedule import hasCustomSchedule, ScheduleInfo
from Tensile.Components.CMSValidator import isValid, SchedulePosition, ValidationContext, ValidationConcern, active_concerns
from Tensile.Common import IsaVersion
from cms_test_utils import make_mock_id_map, make_mock_mfma_code


def _make_context(kernel, schedule_info):
    """Build a ValidationContext from mock idMap + mfmaCode.

    The per-shape positive-validation tests that used to live in this file
    (TestCustomScheduleBF16 / TestCustomScheduleTF32) were removed because
    they reproduced what `Tests/common/gemm/gfx950/custom_mainloop_scheduling*.yaml`
    already exercise via the full Tensile pipeline (which runs both the
    structural-rule isValid AND the dataflow-graph compare_graphs +
    validate_edge_wait_coverage as production gates). The remaining tests
    here exercise validator API surface, schedule-position ordering, and
    gfx1151 dispatch — none of which need the real-data kernel-writer path.
    """
    return ValidationContext(
        kernel=kernel,
        id_map=make_mock_id_map(schedule_info, kernel),
        mfma_code=make_mock_mfma_code(schedule_info.numMfma),
    )


# Helper to create a mock data type
def _mock_dtype(is_16bit=False, is_8bit=False, num_bytes=4):
    mock = MagicMock()
    mock.isHalf.return_value = is_16bit
    mock.isBFloat16.return_value = False # Assuming isHalf is enough for is16bit
    mock.isInt8.return_value = is_8bit
    mock.is8bitFloat.return_value = False # Assuming isInt8 is enough for is8bit
    mock.numBytes.return_value = num_bytes
    return mock

# Base kernel configuration factory
def create_base_kernel():
    kernel = {
        "UseCustomMainLoopSchedule": True,
        "EnableMatrixInstruction": True,
        "UnrollLoopSwapGlobalReadOrder": False,
        "ISA": IsaVersion(9,5,0),
        "WavefrontSize": 64,
        "ProblemType": {
            "DataType": _mock_dtype(),
            "DataTypeA": _mock_dtype(),
            "DataTypeB": _mock_dtype(),
            "TransposeA": False,
            "TransposeB": False,
            "TLUA": True,
            "TLUB": False,
        },
        "MacroTile0": 0, "MacroTile1": 0, "DepthU": 64,
        "PrefetchGlobalRead": 0, "PrefetchLocalRead": 0, "DirectToLds": 1,  "DtlPlusLdsBuf": False,
        "GlobalReadVectorWidthA": 0, "GlobalReadVectorWidthB": 0,
        "LocalReadVectorWidthA": 0, "LocalReadVectorWidthB": 0,
        "WaveSeparateGlobalReadA": 0,
        "WaveSeparateGlobalReadB": 0,
        "Use64bShadowLimit" : 1,
        "MatrixInstruction": [16,16,32,1],
        "MIWaveGroup": [],
        "LDSTrInst": False,
        "TransposeLDS": 0,
        "ForceUnrollSubIter": False,
        "SwapGlobalReadOrder": False, # For asserting it gets set
        "UsePLRPack": False, # For asserting it gets set
        "UseF32XEmulation": False,
        "VectorWidthA": 1,
        "VectorWidthB": 1,
        "MIWaveTileA": 2,
        "MIWaveTileB": 2,
    }
    return kernel

def update_kernel(kernel, updates):
    """Update kernel dict, auto-deriving TLUA/TLUB from TransposeA/TransposeB.

    Args:
        kernel: kernel dict to modify in-place
        updates: dict mirroring the kernel structure. "ProblemType" key (if present)
                 is applied via kernel["ProblemType"].update(); all other keys are
                 applied via kernel.update(). If TransposeA or TransposeB appear in
                 the ProblemType sub-dict, TLUA and TLUB are auto-derived.
    """
    if "ProblemType" in updates:
        pt = updates.pop("ProblemType")
        if "TransposeA" in pt or "TransposeB" in pt:
            transA = pt.get("TransposeA", kernel["ProblemType"]["TransposeA"])
            transB = pt.get("TransposeB", kernel["ProblemType"]["TransposeB"])
            pt["TLUA"] = not transA
            pt["TLUB"] = transB
        kernel["ProblemType"].update(pt)
    kernel.update(updates)


class TestCustomScheduleValidation:
    def test_no_custom_schedule(self):
        """A kernel that doesn't match any registered schedule returns False."""
        kernel = create_base_kernel()
        # An empty kernel should not have a custom schedule
        has_schedule, schedule_info = hasCustomSchedule(kernel)
        assert not has_schedule
        assert schedule_info is None

    def test_ascending_order_catches_invalid_schedule(self):
        """A non-ascending schedule is caught by the ordering concern."""
        kernel = create_base_kernel()
        invalid_schedule = {"P": [[3, 2, 1]]}

        si = ScheduleInfo(1, None, invalid_schedule, None, None, None, None)
        status, message = isValid(si, _make_context(kernel, si))
        assert status == False
        assert "Non-descending-order" in message

    def test_active_concerns_gfx950_with_packs(self):
        """gfx950 kernel with packs includes PACK_DATA_READY."""
        kernel = create_base_kernel()
        idmap = {"LRA0": [], "PackA0": [], "GRA": [], "GRIncA": [], "SYNC": []}
        concerns = active_concerns(kernel, idmap)
        assert ValidationConcern.PACK_DATA_READY in concerns
        assert ValidationConcern.LR_DATA_READY in concerns
        assert ValidationConcern.INSTRUCTION_ORDERING in concerns
        assert ValidationConcern.SCHEDULE_COMPLETENESS in concerns
        assert ValidationConcern.SCALAR_REGISTER_SAFETY in concerns

    def test_active_concerns_gfx950_tf32(self):
        """gfx950 kernel with TF32 includes QUAD_CYCLE_TIMING."""
        kernel = create_base_kernel()
        kernel["UseF32XEmulation"] = True
        idmap = {"LRA0": [], "PackA0": [], "GRA": []}
        concerns = active_concerns(kernel, idmap)
        assert ValidationConcern.QUAD_CYCLE_TIMING in concerns

    def test_active_concerns_gfx1151(self):
        """gfx1151 currently only has INSTRUCTION_ORDERING in its catalog.

        Other concerns (LW_ORDERING, GR_VGPR_READY, LDS coherence, etc.) are
        declared as future work in the catalog comments but not yet active —
        they will be added in stage 12 when the corresponding rules are built.
        """
        kernel = _gfx1151_base_kernel()
        kernel["MacroTile0"] = 96
        kernel["MacroTile1"] = 128
        idmap = {"LRA0": [], "GRA": [], "SYNC": []}
        concerns = active_concerns(kernel, idmap)
        assert ValidationConcern.INSTRUCTION_ORDERING in concerns
        # No other concerns are active yet for gfx1151
        assert ValidationConcern.PACK_DATA_READY not in concerns
        assert ValidationConcern.QUAD_CYCLE_TIMING not in concerns
        assert ValidationConcern.SCALAR_REGISTER_SAFETY not in concerns
        assert ValidationConcern.SCHEDULE_COMPLETENESS not in concerns

    def test_active_concerns_unknown_isa(self):
        """Unknown ISA returns empty set (graceful fallback)."""
        kernel = create_base_kernel()
        kernel["ISA"] = (99, 99, 99)
        concerns = active_concerns(kernel, {})
        assert concerns == set()

class TestSchedulePositionOrdering:
    """Test that SchedulePosition comparison handles vmfma_index=-1 correctly.

    vmfma_index=-1 is a wrap-around position: it represents instructions
    scheduled between iterations (after the last VMFMA of the previous
    iteration, before the first VMFMA of the current iteration). With
    explicit loop_index tracking, -1 naturally sorts before 0 within the
    same loop via integer comparison.
    """

    def test_neg1_after_last_vmfma_same_loop(self):
        """vmfma=-1 in loop 0 must be < vmfma=0 in loop 0."""
        a = SchedulePosition(loop_index=0, vmfma_index=-1, sub_index=0)
        b = SchedulePosition(loop_index=0, vmfma_index=0, sub_index=0)
        assert a < b
        assert b > a

    def test_neg1_loop1_after_last_vmfma_loop0(self):
        """vmfma=-1 in loop 1 must be > vmfma=num_vmfma-1 in loop 0."""
        a = SchedulePosition(loop_index=1, vmfma_index=-1, sub_index=0)
        b = SchedulePosition(loop_index=0, vmfma_index=2, sub_index=0)
        assert a > b
        assert b < a

    def test_neg1_before_next_loop_vmfma0(self):
        """vmfma=-1 in loop 0 must be < vmfma=0 in loop 1."""
        a = SchedulePosition(loop_index=0, vmfma_index=-1, sub_index=0)
        b = SchedulePosition(loop_index=1, vmfma_index=0, sub_index=0)
        assert a < b
        assert b > a

    def test_equal_positions(self):
        """Identical positions must be equal."""
        a = SchedulePosition(loop_index=0, vmfma_index=3, sub_index=1)
        b = SchedulePosition(loop_index=0, vmfma_index=3, sub_index=1)
        assert a == b
        assert not (a != b)
        assert not (a < b)
        assert not (a > b)
        assert a <= b
        assert a >= b

    def test_not_equal_different_sub_index(self):
        """Same (loop, vmfma) but different sub_index must not be equal."""
        a = SchedulePosition(loop_index=0, vmfma_index=3, sub_index=0)
        b = SchedulePosition(loop_index=0, vmfma_index=3, sub_index=1)
        assert a != b
        assert a < b

    def test_sub_index_ordering(self):
        """Within same (loop, vmfma), sub_index determines order."""
        a = SchedulePosition(loop_index=0, vmfma_index=2, sub_index=0)
        b = SchedulePosition(loop_index=0, vmfma_index=2, sub_index=5)
        c = SchedulePosition(loop_index=0, vmfma_index=2, sub_index=10)
        assert a < b < c
        assert c > b > a


# ----------------------------------------------------------------------------
# gfx1151 (RDNA 3.5) WMMA schedule tests
# ----------------------------------------------------------------------------
# These cover the schedules defined at the bottom of CustomSchedule.py.
# Coverage goals:
#   - dispatcher: hasCustomSchedule picks a gfx1151 schedule for gfx1151 kernels
#     and does NOT mis-dispatch CDNA 4 kernels to a gfx1151 schedule.
#   - predicate  : non-TN kernels and non-16bit dtypes are not routed to
#     TN-only fp16/bf16 gfx1151 schedules.
#   - validator  : active_concerns() auto-skips rules that don't apply to
#     gfx1151's ISA catalog, so isValid returns True with the correct
#     subset of validation (ordering checks, not CDNA-4 timing).


def _gfx1151_base_kernel():
    """Canonical gfx1151 kernel shell (MT / tile params filled in per case)."""
    dt = _mock_dtype(is_16bit=True, num_bytes=2)
    return {
        "UseCustomMainLoopSchedule": True,
        "EnableMatrixInstruction": True,
        "UnrollLoopSwapGlobalReadOrder": False,
        "ISA": IsaVersion(11, 5, 1),
        "WavefrontSize": 32,
        "ProblemType": {
            "DataType": dt, "DataTypeA": dt, "DataTypeB": dt,
            "TransposeA": True, "TransposeB": False,
            "TLUA": False, "TLUB": False,
        },
        "MacroTile0": 0, "MacroTile1": 0, "DepthU": 32,
        "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
        "DirectToLds": 0, "DtlPlusLdsBuf": False,
        "GlobalReadVectorWidthA": 8, "GlobalReadVectorWidthB": 8,
        "LocalReadVectorWidthA": 16, "LocalReadVectorWidthB": 16,
        "WaveSeparateGlobalReadA": 0, "WaveSeparateGlobalReadB": 0,
        "Use64bShadowLimit": 1,
        "MatrixInstruction": [16, 16, 16, 1],
        "MIWaveGroup": [2, 2],
        "LDSTrInst": False, "TransposeLDS": 1,
        "ForceUnrollSubIter": False,
        "SwapGlobalReadOrder": False,
        "UsePLRPack": False,
        "UseF32XEmulation": False,
        "MIWaveTileA": 1, "MIWaveTileB": 1,
        "1LDSBuffer": 0,
    }


# Tile parameters for a representative subset of gfx1151 schedules.
# Format: (MT0, MT1, DU, PGR, PLR, MIWG, WTA, WTB, MIK)
#   MIK=matrixInstK (16 for the bulk of fp16 tiles, 128 for the 16x16x128 case).
GFX1151_TILES = [
    (96,  128, 32,  2, 1, [2, 2], 3, 4, 16),
    (128, 96,  32,  2, 1, [2, 2], 4, 3, 16),
    (192, 64,  32,  2, 1, [4, 1], 3, 4, 16),
    (64,  192, 32,  2, 1, [1, 4], 4, 3, 16),
    (64,  128, 32,  2, 1, [2, 2], 2, 4, 16),
    (128, 64,  64,  2, 1, [2, 2], 4, 2, 16),
    (32,  128, 64,  2, 1, [1, 4], 2, 2, 16),
    (16,  16,  128, 2, 1, [1, 1], 1, 1, 16),
]


class TestCustomScheduleGfx1151:
    """Tests for gfx1151 (RDNA 3.5) WMMA schedules."""

    # ---- Dispatcher / predicate ----

    def test_dispatch_gfx1151_TN_16bit(self):
        """A TN fp16 gfx1151 kernel should pick up a gfx1151 schedule."""
        k = _gfx1151_base_kernel()
        update_kernel(k, {
            "MacroTile0": 96, "MacroTile1": 128, "DepthU": 32,
            "MIWaveTileA": 3, "MIWaveTileB": 4,
        })
        has, info = hasCustomSchedule(k)
        assert has
        assert isinstance(info, ScheduleInfo)

    def test_non_TN_does_not_dispatch(self):
        """gfx1151 schedules in this commit are TN-only; NN kernels must not match."""
        k = _gfx1151_base_kernel()
        update_kernel(k, {
            "ProblemType": {"TransposeA": False, "TransposeB": False},
            "MacroTile0": 96, "MacroTile1": 128, "DepthU": 32,
            "MIWaveTileA": 3, "MIWaveTileB": 4,
        })
        has, info = hasCustomSchedule(k)
        assert not has
        assert info is None

    def test_cdna4_isa_does_not_dispatch_to_gfx1151(self):
        """A CDNA 4 (gfx950) kernel must not match a gfx1151-registered tile."""
        k = _gfx1151_base_kernel()
        update_kernel(k, {
            "ISA": IsaVersion(9, 5, 0),
            "WavefrontSize": 64,
            "DirectToLds": 1,
            "MacroTile0": 96, "MacroTile1": 128, "DepthU": 32,
            "MIWaveTileA": 3, "MIWaveTileB": 4,
        })
        has, info = hasCustomSchedule(k)
        assert not has

    def test_fp32_does_not_dispatch_to_16bit_gfx1151(self):
        """The 16-bit gfx1151 schedules must not match a fp32 kernel."""
        k = _gfx1151_base_kernel()
        dt32 = _mock_dtype(is_16bit=False, is_8bit=False, num_bytes=4)
        update_kernel(k, {
            "ProblemType": {
                "DataType": dt32, "DataTypeA": dt32, "DataTypeB": dt32,
            },
            "MacroTile0": 96, "MacroTile1": 128, "DepthU": 32,
            "MIWaveTileA": 3, "MIWaveTileB": 4,
        })
        has, info = hasCustomSchedule(k)
        assert not has

    # ---- Schedule shape ----

    @pytest.mark.parametrize("MT0, MT1, DU, PGR, PLR, MIWG, WTA, WTB, MIK", GFX1151_TILES)
    def test_schedule_shape(self, MT0, MT1, DU, PGR, PLR, MIWG, WTA, WTB, MIK):
        """Each gfx1151 tile produces a well-formed ScheduleInfo."""
        k = _gfx1151_base_kernel()
        update_kernel(k, {
            "MacroTile0": MT0, "MacroTile1": MT1, "DepthU": DU,
            "PrefetchGlobalRead": PGR, "PrefetchLocalRead": PLR,
            "MatrixInstruction": [16, 16, MIK, 1],
            "MIWaveGroup": MIWG, "MIWaveTileA": WTA, "MIWaveTileB": WTB,
        })
        has, info = hasCustomSchedule(k)
        assert has, f"no schedule dispatched for MT{MT0}x{MT1}x{DU}"
        assert isinstance(info, ScheduleInfo)
        assert info.numMfma > 0
        assert info.numCodePaths >= 1
        assert "SYNC" in info.optSchedule

    # ---- Validator coverage ----

    @pytest.mark.parametrize("MT0, MT1, DU, PGR, PLR, MIWG, WTA, WTB, MIK", GFX1151_TILES)
    def test_validator_passes_gfx1151(
            self, MT0, MT1, DU, PGR, PLR, MIWG, WTA, WTB, MIK):
        """Applicable validator rules must run and succeed on gfx1151 schedules."""
        k = _gfx1151_base_kernel()
        update_kernel(k, {
            "MacroTile0": MT0, "MacroTile1": MT1, "DepthU": DU,
            "PrefetchGlobalRead": PGR, "PrefetchLocalRead": PLR,
            "MatrixInstruction": [16, 16, MIK, 1],
            "MIWaveGroup": MIWG, "MIWaveTileA": WTA, "MIWaveTileB": WTB,
        })
        has, info = hasCustomSchedule(k)
        assert has
        valid, msg = isValid(info, _make_context(k, info))
        assert valid, f"MT{MT0}x{MT1}x{DU}: isValid said: {msg}"

    def test_gfx1151_concerns_skip_timeline_rules(self):
        """gfx1151 catalog only has INSTRUCTION_ORDERING, so all timeline rules auto-skip."""
        k = _gfx1151_base_kernel()
        k["MacroTile0"] = 96
        k["MacroTile1"] = 128
        idmap = {"LRA0": [], "GRA": [], "SYNC": []}
        concerns = active_concerns(k, idmap)
        # No timeline rule concerns match, so timeline construction is skipped entirely.
        assert ValidationConcern.PACK_DATA_READY not in concerns
        assert ValidationConcern.LR_DATA_READY not in concerns
        assert ValidationConcern.LDS_WRITE_AFTER_READ not in concerns
        assert ValidationConcern.LDS_READ_AFTER_WRITE not in concerns
        # Ordering is the only active concern
        assert concerns == {ValidationConcern.INSTRUCTION_ORDERING}
