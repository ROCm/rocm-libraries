# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for SourceSwap (SS1) on a non-square MatrixInstruction.

SourceSwap feeds B into the wide src0 and A into the narrow src1, so the
physical ``v_wmma_..MxN`` opcode produces a transposed accumulator. After the
swap, ``MatrixInstM``/``MatrixInstN`` hold the *effective* (transposed) tiling
extents; the *physical* opcode dims live in ``MIBlock[0]``/``MIBlock[1]``.

These tests pin ``effectiveMatrixInstMN`` — the single place that decides the
swap — and the MacroTile geometry that follows from it.
"""

import pytest

from Tensile.Common import effectiveMatrixInstMN


class TestEffectiveMatrixInstMN:
    @pytest.mark.parametrize(
        "matrixInstM,matrixInstN,sourceSwap,expected",
        [
            # non-square + SS1 -> extents swap
            (32, 16, True,  (16, 32)),
            (16, 32, True,  (32, 16)),
            # non-square + SS0 -> identity (no transpose)
            (32, 16, False, (32, 16)),
            (16, 32, False, (16, 32)),
            # square MI -> identity regardless of SourceSwap
            (16, 16, True,  (16, 16)),
            (16, 16, False, (16, 16)),
            (32, 32, True,  (32, 32)),
        ],
        ids=[
            "32x16_SS1_swaps", "16x32_SS1_swaps",
            "32x16_SS0_identity", "16x32_SS0_identity",
            "16x16_SS1_identity", "16x16_SS0_identity", "32x32_SS1_identity",
        ],
    )
    def test_effective_extents(self, matrixInstM, matrixInstN, sourceSwap, expected):
        assert effectiveMatrixInstMN(matrixInstM, matrixInstN, sourceSwap) == expected

    def test_inputs_not_mutated(self):
        # The helper returns fresh values and never mutates its inputs. Callers
        # pass MIBlock[0]/[1] (the physical opcode dims), which must be preserved.
        m, n = 32, 16
        effectiveMatrixInstMN(m, n, True)
        assert (m, n) == (32, 16)


class TestMacroTileGeometry:
    """MI 32x16 SS1 with WT=[2,1], WG=[2,1] -> MT64x32 (see PR example).

    MT_M = MEff * WT0 * WG0,  MT_N = NEff * WT1 * WG1.
    MEff/NEff swap to 16/32 under SS1, so the MacroTile matches the transposed
    16M x 32N accumulator instead of the physical 32x16.
    """

    def test_mi32x16_ss1_macrotile(self):
        mEff, nEff = effectiveMatrixInstMN(32, 16, True)
        waveTile = (2, 1)
        waveGroup = (2, 1)
        mtM = mEff * waveTile[0] * waveGroup[0]
        mtN = nEff * waveTile[1] * waveGroup[1]
        assert (mtM, mtN) == (64, 32)

    def test_mi32x16_ss0_macrotile(self):
        # Same MI/tiling without SourceSwap keeps the physical orientation.
        mEff, nEff = effectiveMatrixInstMN(32, 16, False)
        mtM = mEff * 2 * 2
        mtN = nEff * 1 * 1
        assert (mtM, mtN) == (128, 16)


def _piap_mi32x16(source_swap):
    """Run assignProblemIndependentDerivedParameters on an MI 32x16 state and
    return the per-thread input counts (A, B) after derivation.

    Kept out of the pure tests above: this needs a C++ toolchain + isaInfoMap.
    """
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Common.Types import IsaVersion
    from Tensile.Common.DataType import DataType
    from Tensile.Toolchain.Validators import validateToolchain
    from Tensile.SolutionStructs.Solution import Solution

    isa = IsaVersion(12, 5, 0)
    isaInfoMap = makeIsaInfoMap([isa], validateToolchain("amdclang++"))
    state = {
        "ISA": isa,
        "ScheduleIterAlg": 1,
        "WavefrontSize": 32,
        "ProblemType": {
            "StridedBatched": True,
            "Batched": True,
            "OperationType": "GEMM",
            "DataType": DataType("s"),
            "DataTypeA": DataType("s"),
            "DataTypeB": DataType("s"),
            "HighPrecisionAccumulate": False,
            "MXBlockA": 0,
            "MXBlockB": 0,
            "Sparse": 0,
            "TransposeA": False,
            "TransposeB": False,
            "SwizzleTensorA": False,
            "SwizzleTensorB": False,
            "TLUA": True,
            "TLUB": True,
        },
        "EnableMatrixInstruction": True,
        "MIBlock": [32, 16, 128, 1, 1, 1],
        "MIWaveGroup": [1, 1],
        "MIWaveTile": [1, 1],
        "WorkGroup": [32, 1, 1],
        "WaveSplitK": False,
        # Distinct A/B so the swap is observable (physical: A=M-side, B=N-side).
        "MIInputPerThread": 128,
        "MIInputPerThreadA": 128,
        "MIInputPerThreadB": 64,
        "SourceSwap": source_swap,
        "DirectToLds": 0,
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "DirectToVgprMXSA": False,
        "DirectToVgprMXSB": False,
        "BufferLoad": True,
        "AssertSummationElementMultiple": 1,
        "AssertFree0ElementMultiple": 1,
        "AssertFree1ElementMultiple": 1,
        "UseF32XEmulation": False,
        "UseSubtileImpl": False,
        "ClusterDim": [1, 1],
        "ClusterBarrier": False,
        "TDMInst": 0,
        "DepthU": 128,
    }
    Solution.assignProblemIndependentDerivedParameters(state, False, isaInfoMap)
    return state["MIInputPerThreadA"], state["MIInputPerThreadB"]


class TestMIInputPerThreadSwap:
    """SS1 non-square swaps MIInputPerThreadA/B (A sized for the N-side, B for the
    M-side) to match the operands fed to the WMMA in mfmaIter. Regression guard
    for the "invalid operand for instruction" failure when B is under-sized.
    """

    def test_ss1_nonsquare_swaps_inputs(self):
        assert _piap_mi32x16(source_swap=True) == (64, 128)

    def test_ss0_nonsquare_keeps_inputs(self):
        assert _piap_mi32x16(source_swap=False) == (128, 64)
