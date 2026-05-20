################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Solution-level gating for the subtile BF16 any-K tail.

Pins:
  - Non-MX subtile BF16 accepts ASEM ∈ {1, 2, 4, 8} without bumping.
  - At ASEM=1 (strictest case: triggers the `aemA*bpeA % 4 != 0` outer
    guard in `Solution.py`), `NonDTLTailLoop{A,B}` stays False because
    the inner guards include `and not state["UseSubtileImpl"]`.
  - MX subtile still bumps to 32 via `minASEMforMX`.
"""
import pytest
import yaml

from Tensile.Common.Architectures import SUPPORTED_ISA
from Tensile.Common.Capabilities import makeIsaInfoMap
from Tensile.Common.Types import IsaVersion
from Tensile.Common.GlobalParameters import defaultSolution
from Tensile.Toolchain.Validators import validateToolchain
from Tensile.SolutionStructs.Validators.MatrixInstruction import matrixInstructionToMIParameters
from Tensile.SolutionStructs.Problem import ProblemType
from Tensile.SolutionStructs.Solution import Solution


_cxxCompiler = validateToolchain("amdclang++")
_isaInfoMap = makeIsaInfoMap(SUPPORTED_ISA, _cxxCompiler)


# ── State factories ──────────────────────────────────────────────────────────

def _build_subtile_bf16_state(*, asem=8, depthU=64, mt0=128, mt1=128):
    """State factory for a subtile BF16 (TN) kernel, no MX scales."""
    config = yaml.safe_load(
        """
        ProblemType:
          OperationType: GEMM
          DataType: b
          DestDataType: b
          ComputeDataType: s
          HighPrecisionAccumulate: True
          TransposeA: True
          TransposeB: False
          UseBeta: True
          Batched: True
          StridedBatched: True
          ActivationFuncCall: True
        """
    )
    isa = IsaVersion(9, 5, 0)
    mi = [16, 16, 32, 1, 1, mt0 // 64, mt1 // 64, 2, 2]
    mi_params = matrixInstructionToMIParameters(
        mi, isa, 64, config["ProblemType"], [16, 16, 1], _isaInfoMap
    )

    state = dict(defaultSolution)
    state.update({
        "ISA": isa,
        "WavefrontSize": 64,
        "ScheduleIterAlg": 3,
        "UseSubtileImpl": True,
        "StreamK": 3,
        "DepthU": depthU,
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 0,
        "DirectToLds": 1,
        "StaggerU": 0,
        "LocalSplitU": 1,
        "AssertSummationElementMultiple": asem,
        "AssertFree0ElementMultiple": 1,
        "AssertFree1ElementMultiple": 1,
        "NoReject": False,
        "MatrixInstruction": mi,
        "EnableMatrixInstruction": True,
        "UseF32XEmulation": False,
    })
    state.update(mi_params)
    state["ProblemType"] = ProblemType(config["ProblemType"], False)
    return state


def _build_subtile_mx_state(*, asem=16, depthU=256, mt0=128, mt1=128):
    """State factory for a subtile MXFP4 (TN) kernel."""
    config = yaml.safe_load(
        """
        ProblemType:
          OperationType: GEMM
          DataType: F4
          DestDataType: b
          ComputeDataType: s
          HighPrecisionAccumulate: True
          MXBlockA: 32
          MXBlockB: 32
          TransposeA: True
          TransposeB: False
          UseBeta: True
          Batched: True
          StridedBatched: True
          ActivationFuncCall: True
        """
    )
    isa = IsaVersion(9, 5, 0)
    mi = [16, 16, 128, 1, 1, mt0 // 64, mt1 // 64, 2, 2]
    mi_params = matrixInstructionToMIParameters(
        mi, isa, 64, config["ProblemType"], [16, 16, 1], _isaInfoMap
    )

    state = dict(defaultSolution)
    state.update({
        "ISA": isa,
        "WavefrontSize": 64,
        "ScheduleIterAlg": 3,
        "UseSubtileImpl": True,
        "StreamK": 3,
        "DepthU": depthU,
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 0,
        "DirectToLds": 1,
        "StaggerU": 0,
        "LocalSplitU": 1,
        "AssertSummationElementMultiple": asem,
        "AssertFree0ElementMultiple": 1,
        "AssertFree1ElementMultiple": 1,
        "NoReject": False,
        "MatrixInstruction": mi,
        "EnableMatrixInstruction": True,
        "UseF32XEmulation": False,
    })
    state.update(mi_params)
    state["ProblemType"] = ProblemType(config["ProblemType"], False)
    return state


def _run_assign_problem_independent(state):
    """Run `assignProblemIndependentDerivedParameters`. Tolerates
    downstream KeyErrors: the gates we pin execute before any field
    that varies across setups.
    """
    try:
        Solution.assignProblemIndependentDerivedParameters(state, False, _isaInfoMap)
    except KeyError:
        pass


# ── Tests: bf16 ASEM acceptance ──────────────────────────────────────────────

class TestSubtileBf16AcceptsAnyK:
    """ASEM ∈ {1, 2, 4, 8} must survive `assignProblemIndependent…`
    without being bumped. Pins the property in case a future commit
    introduces a non-MX ASEM bump.
    """

    def test_subtile_bf16_accepts_asem_8(self):
        state = _build_subtile_bf16_state(asem=8, depthU=64)
        _run_assign_problem_independent(state)
        actual = state["AssertSummationElementMultiple"]
        assert actual == 8, (
            f"BF16 subtile ASEM=8: AssertSummationElementMultiple was "
            f"bumped to {actual} (expected to remain 8). No non-MX bump "
            f"should exist."
        )

    def test_subtile_bf16_accepts_asem_4(self):
        state = _build_subtile_bf16_state(asem=4, depthU=64)
        _run_assign_problem_independent(state)
        actual = state["AssertSummationElementMultiple"]
        assert actual == 4, (
            f"BF16 subtile ASEM=4: AssertSummationElementMultiple was "
            f"bumped to {actual} (expected to remain 4)."
        )

    def test_subtile_bf16_accepts_asem_2(self):
        state = _build_subtile_bf16_state(asem=2, depthU=64)
        _run_assign_problem_independent(state)
        actual = state["AssertSummationElementMultiple"]
        assert actual == 2, (
            f"BF16 subtile ASEM=2: AssertSummationElementMultiple was "
            f"bumped to {actual} (expected to remain 2)."
        )

    def test_subtile_bf16_accepts_asem_1(self):
        state = _build_subtile_bf16_state(asem=1, depthU=64)
        _run_assign_problem_independent(state)
        actual = state["AssertSummationElementMultiple"]
        assert actual == 1, (
            f"BF16 subtile ASEM=1: AssertSummationElementMultiple was "
            f"bumped to {actual} (expected to remain 1)."
        )


# ── Tests: bf16 DTL preservation ─────────────────────────────────────────────

class TestSubtileBf16KeepsDTL:
    """At ASEM=1 the `aemA*bpeA % 4 != 0` outer guard in `Solution.py`
    fires. The inner guards must NOT then set `NonDTLTailLoop{A,B} =
    True` for a subtile kernel; the subtile tail-loop emit path is
    structurally DTL-only and masks its own tail at sub-dword
    granularity.
    """

    def test_subtile_bf16_asem_1_keeps_dtl(self):
        state = _build_subtile_bf16_state(asem=1, depthU=64)
        _run_assign_problem_independent(state)

        assert state.get("NonDTLTailLoopA") is False, (
            f"bf16 subtile ASEM=1: NonDTLTailLoopA = "
            f"{state.get('NonDTLTailLoopA')} (expected False)."
        )
        assert state.get("NonDTLTailLoopB") is False, (
            f"bf16 subtile ASEM=1: NonDTLTailLoopB = "
            f"{state.get('NonDTLTailLoopB')} (expected False)."
        )


# ── Tests: MX bump still works ───────────────────────────────────────────────

class TestSubtileMxBumpRegression:
    """Regression net: the MX path's `minASEMforMX = 32` bump must
    still fire for MX subtile kernels.
    """

    def test_subtile_mx_still_bumps_to_32(self):
        state = _build_subtile_mx_state(asem=16, depthU=256)
        _run_assign_problem_independent(state)
        actual = state["AssertSummationElementMultiple"]
        assert actual == 32, (
            f"MX subtile ASEM=16: AssertSummationElementMultiple = "
            f"{actual} (expected 32). The MX ASEM bump must still fire."
        )


# ── Tests: NoTailLoop derivation ─────────────────────────────────────────────

class TestSubtileBf16NoTailLoopDerive:
    """`NoTailLoop = (ASEM % DepthU == 0)`. For bf16 subtile with
    DepthU=64 and ASEM ∈ {1, 2, 4, 8}, this must be False so the
    tail-loop emit fires.
    """

    @pytest.mark.parametrize("asem", [1, 2, 4, 8])
    def test_subtile_bf16_no_tail_loop_derives_false(self, asem):
        state = _build_subtile_bf16_state(asem=asem, depthU=64)
        _run_assign_problem_independent(state)
        no_tail_loop_derived = (
            state["AssertSummationElementMultiple"] % state["DepthU"] == 0
        )
        assert no_tail_loop_derived is False, (
            f"ASEM={asem} DepthU=64 should leave NoTailLoop=False so "
            f"the subtile tail emits. Got "
            f"ASEM={state['AssertSummationElementMultiple']}, "
            f"DepthU={state['DepthU']}, "
            f"ASEM % DepthU == 0 → {no_tail_loop_derived}."
        )
