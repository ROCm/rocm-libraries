# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Solution-level pins for the subtile BF16 any-K tail on large MT shapes.

The large symmetric MT 320x320 (10x10 wavetile, 2x2 WG) config and the
existing MT 320x288 repro shape must not be rejected at the
`assignProblemIndependent…` gate across all three `PrefetchGlobalRead`
values and the three ASEM ∈ {1, 2, 8} modes.

The yaml smoke `subtile_bf16_anyk_largemt_extended.yaml` covers the
runtime build path. These unit pins cover the Solution-level gate
ordering: they catch a regression that would silently drop the large
MT from the built solution set (a VGPR-overflow rejection on this MT
would otherwise silently report success with errorCode 0).

The pins do NOT cover the late-stage VGPR-budget rejection that
happens inside `KernelWriter.py` after register allocation -- that
needs the full kernel-write pass and is exercised end-to-end by the
yaml smoke instead.
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


def _build_largemt_state(*, mi, asem, pgr, depthU=64):
    """State factory for a subtile BF16 large-MT TN kernel.

    `mi` is the full 9-tuple MatrixInstruction (mi_m, mi_n, mi_k,
    mi_b, mi_w, wt_m, wt_n, wg_m, wg_n) so callers control both the
    wave tile and the work-group shape (asymmetric WG matters for
    320x288).
    """
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
    mi_params = matrixInstructionToMIParameters(
        list(mi), isa, 64, config["ProblemType"], [16, 16, 1], _isaInfoMap
    )

    state = dict(defaultSolution)
    state.update({
        "ISA": isa,
        "WavefrontSize": 64,
        "ScheduleIterAlg": 3,
        "UseSubtileImpl": True,
        "StreamK": 3,
        "DepthU": depthU,
        "PrefetchGlobalRead": pgr,
        "PrefetchLocalRead": 0,
        "DirectToLds": 1,
        "StaggerU": 0,
        "LocalSplitU": 1,
        "AssertSummationElementMultiple": asem,
        "AssertFree0ElementMultiple": 1,
        "AssertFree1ElementMultiple": 1,
        "NoReject": False,
        "MatrixInstruction": list(mi),
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


# ── Large MT shapes covered ─────────────────────────────────────────────────

# (label, mi-9-tuple). MT = mi_m*wt_m*wg_m × mi_n*wt_n*wg_n.
_LARGE_MT_SHAPES = [
    ("MT_320x320_2x2WG", [16, 16, 32, 1, 1, 10, 10, 2, 2]),
    ("MT_320x288_4x1WG", [16, 16, 32, 1,  1, 5, 18,  4, 1]),
]


# ── Tests ───────────────────────────────────────────────────────────────────

class TestSubtileBf16LargeMTNotRejected:
    """Pin: the Solution-level gate accepts MT 320x320 and MT 320x288
    at ASEM ∈ {1, 2, 8} so no silent drop happens before the
    kernel-write stage. MT 320x320 in particular sits near the VGPR
    budget ceiling and is the shape most at risk of a silent reject.
    """

    @pytest.mark.parametrize("label,mi", _LARGE_MT_SHAPES)
    @pytest.mark.parametrize("asem", [1, 2, 8])
    def test_largemt_solution_accepted(self, label, mi, asem):
        state = _build_largemt_state(mi=mi, asem=asem, pgr=0)
        _run_assign_problem_independent(state)
        assert state.get("Valid") is not False, (
            f"Large MT {label} ASEM={asem}: Solution.py rejected the "
            f"kernel at the assignProblemIndependent gate (state['Valid'] "
            f"is False). This is a silent failure mode -- a VGPR-overflow "
            f"reject on this MT would still report errorCode 0."
        )


class TestSubtileBf16LargeMTAcceptsAllPGR:
    """Pin: the Solution-level gate accepts MT 320x320 and MT 320x288
    at PrefetchGlobalRead ∈ {0, 1, 2} so the per-prefetch-depth
    GR_inc paths all get a chance to run during the yaml smoke. The
    tail-loop GR_inc differs across prefetch depths, so this gate has
    to hold for all three.
    """

    @pytest.mark.parametrize("label,mi", _LARGE_MT_SHAPES)
    @pytest.mark.parametrize("pgr", [0, 1, 2])
    def test_largemt_pgr_accepted(self, label, mi, pgr):
        state = _build_largemt_state(mi=mi, asem=2, pgr=pgr)
        _run_assign_problem_independent(state)
        assert state.get("Valid") is not False, (
            f"Large MT {label} PGR={pgr}: Solution.py rejected the "
            f"kernel at the assignProblemIndependent gate. The yaml "
            f"sweep would silently drop this config from the build set."
        )


class TestSubtileBf16LargeMTKeepsDTL:
    """The subtile DTL-preservation guarantee must hold for the
    large-MT shapes too: a regression that flips
    `NonDTLTailLoop{A,B}=True` here would push the kernel onto the
    legacy non-subtile path. Pinned for ASEM=1 (the strictest gate)
    on both 320x320 and 320x288.
    """

    @pytest.mark.parametrize("label,mi", _LARGE_MT_SHAPES)
    def test_largemt_asem_1_keeps_dtl(self, label, mi):
        state = _build_largemt_state(mi=mi, asem=1, pgr=0)
        _run_assign_problem_independent(state)
        assert state.get("NonDTLTailLoopA") is False, (
            f"Large MT {label} ASEM=1: NonDTLTailLoopA = "
            f"{state.get('NonDTLTailLoopA')} (expected False). The "
            f"subtile DTL preservation gate must hold for large MTs."
        )
        assert state.get("NonDTLTailLoopB") is False, (
            f"Large MT {label} ASEM=1: NonDTLTailLoopB = "
            f"{state.get('NonDTLTailLoopB')} (expected False)."
        )


class TestSubtileBf16LargeMTNoTailLoopDerive:
    """Mirror of `TestSubtileBf16NoTailLoopDerive` for the large-MT
    shapes: `NoTailLoop = (ASEM % DepthU == 0)` must stay False so
    the tail emit path fires.
    """

    @pytest.mark.parametrize("label,mi", _LARGE_MT_SHAPES)
    @pytest.mark.parametrize("asem", [1, 2, 4, 8])
    def test_largemt_no_tail_loop_false(self, label, mi, asem):
        state = _build_largemt_state(mi=mi, asem=asem, pgr=0, depthU=64)
        _run_assign_problem_independent(state)
        no_tail_loop_derived = (
            state["AssertSummationElementMultiple"] % state["DepthU"] == 0
        )
        assert no_tail_loop_derived is False, (
            f"Large MT {label} ASEM={asem} DepthU=64 should leave "
            f"NoTailLoop=False so the subtile tail emits. Got "
            f"ASEM={state['AssertSummationElementMultiple']}, "
            f"DepthU={state['DepthU']}, "
            f"ASEM % DepthU == 0 → {no_tail_loop_derived}."
        )
