# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Solution-level gating for the subtile BF16 any-K tail.

Pins:
  - Non-MX subtile BF16 accepts ASEM ∈ {1, 2, 4, 8} without bumping.
  - At ASEM=1 (strictest case: triggers the `aemA*bpeA % 4 != 0` outer
    guard in `Solution.py`), `NonDTLTailLoop{A,B}` stays False because
    the inner guards include `and not state["UseSubtileImpl"]`.
  - The UseSubtileImpl narrowing to gfx950 must run BEFORE the
    NonDTLTailLoop gates (covered by `TestSubtileBf16NonGfx950Gate`).
  - The B-side gate also fires when TLUB is False (covered by
    `_build_subtile_bf16_state_nt`).
  - MX subtile still bumps to 32 via `minASEMforMX`.
  - Large symmetric MT 320x320 (10x10 wavetile, 2x2 WG) and the
    asymmetric MT 320x288 (4x1 WG) shapes are accepted at the
    Solution-level gate across ASEM ∈ {1, 2, 8} and PGR ∈ {0, 1, 2}
    (`TestSubtileBf16LargeMT*` classes below). The yaml smoke
    `subtile_bf16_anyk_largemt_extended.yaml` exercises the runtime
    build path; these pins catch a silent reject at the Solution
    gate that would otherwise drop the MT from the built solution
    set with errorCode 0.

  Note: the large-MT pins do NOT cover the late-stage VGPR-budget
  rejection that happens inside `KernelWriter.py` after register
  allocation -- that needs the full kernel-write pass and is
  exercised end-to-end by the yaml smoke instead.
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

def _build_subtile_bf16_state(*, asem=8, depthU=64, mt0=128, mt1=128,
                              transA=True, transB=False, isa=None):
    """State factory for a subtile BF16 kernel, no MX scales.

    Defaults to TN (transA=True, transB=False) for backward compat with
    pre-existing tests; `transA=False, transB=True` produces the NT
    layout that exercises the B-side `NonDTLTailLoopB` gate.
    `isa` overrides the target ISA (default gfx950).
    """
    config = yaml.safe_load(
        f"""
        ProblemType:
          OperationType: GEMM
          DataType: b
          DestDataType: b
          ComputeDataType: s
          HighPrecisionAccumulate: True
          TransposeA: {transA}
          TransposeB: {transB}
          UseBeta: True
          Batched: True
          StridedBatched: True
          ActivationFuncCall: True
        """
    )
    if isa is None:
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


def _build_subtile_bf16_state_nt(*, asem=8, depthU=64, mt0=128, mt1=128,
                                 isa=None):
    """NT-layout (transA=False, transB=True) factory: exercises the
    B-side gate at Solution.py because TLUB is False here.
    """
    return _build_subtile_bf16_state(
        asem=asem, depthU=depthU, mt0=mt0, mt1=mt1,
        transA=False, transB=True, isa=isa,
    )


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

    TN layout exercises the A-side gate (`TLUA=False`); NT layout
    exercises the B-side gate (`TLUB=False`). Both must stay on DTL.
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

    def test_subtile_bf16_asem_1_keeps_dtl_nt(self):
        """NT mirror of the TN test. On NT (`TLUB=True`) the B-side
        inner gate's `not TLUB` clause is False, so the gate never
        reaches `not UseSubtileImpl`. The result is still
        `NonDTLTailLoopB=False`, but via a different code path. This
        pins the property end-to-end across both layouts.
        """
        state = _build_subtile_bf16_state_nt(asem=1, depthU=64)
        _run_assign_problem_independent(state)

        assert state.get("NonDTLTailLoopA") is False, (
            f"bf16 subtile NT ASEM=1: NonDTLTailLoopA = "
            f"{state.get('NonDTLTailLoopA')} (expected False)."
        )
        assert state.get("NonDTLTailLoopB") is False, (
            f"bf16 subtile NT ASEM=1: NonDTLTailLoopB = "
            f"{state.get('NonDTLTailLoopB')} (expected False)."
        )


# ── Tests: UseSubtileImpl narrowing happens before NonDTL gate ──────────────

class TestSubtileBf16NonGfx950Gate:
    """Pin the ordering between the gfx950 narrowing and the NonDTL
    gates: on a non-gfx950 ISA an explicit `UseSubtileImpl=True` must
    be narrowed to False BEFORE the NonDTL gates run, so the kernel
    ends up on the legacy NonDTL path *with* `NonDTLTailLoop`
    populated rather than silently on the legacy path with the flag
    still False.

    The B-side inner gate requires `not TLUB`; on the TN layout
    (`TransposeA=True, TransposeB=False`) both `TLUA` and `TLUB` are
    False, so both A- and B-side gates can fire. NT (`TLUB=True`)
    short-circuits both inner gates before they ever look at
    `UseSubtileImpl`, so it cannot exercise the ordering bug.
    """

    def test_non_gfx950_falls_back_with_nondtl_tail(self):
        # Build a valid gfx950 MI-params dict, then override the ISA
        # to gfx900 so the narrowing fires when derivation runs. TN
        # layout (`TLUA=False, TLUB=False`) is required for the inner
        # gates to reach the `not UseSubtileImpl` clause.
        state = _build_subtile_bf16_state(asem=1, depthU=64,
                                          transA=True, transB=False)
        state["ISA"] = IsaVersion(9, 0, 0)
        state["UseSubtileImpl"] = True

        _run_assign_problem_independent(state)

        assert state.get("UseSubtileImpl") is False, (
            "Non-gfx950 ISA must narrow UseSubtileImpl to False, got "
            f"{state.get('UseSubtileImpl')}."
        )
        assert state.get("NonDTLTailLoopA") is True, (
            "Non-gfx950 + TN + ASEM=1 must end up on the legacy "
            "NonDTL path with NonDTLTailLoopA=True. Got "
            f"{state.get('NonDTLTailLoopA')}. The narrowing of "
            "UseSubtileImpl must run BEFORE the NonDTL gate."
        )
        assert state.get("NonDTLTailLoopB") is True, (
            "Non-gfx950 + TN + ASEM=1 must end up on the legacy "
            "NonDTL path with NonDTLTailLoopB=True. Got "
            f"{state.get('NonDTLTailLoopB')}. The narrowing of "
            "UseSubtileImpl must run BEFORE the NonDTL gate."
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


# ── Large-MT state factory ──────────────────────────────────────────────────

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


# (label, mi-9-tuple). MT = mi_m*wt_m*wg_m × mi_n*wt_n*wg_n.
_LARGE_MT_SHAPES = [
    ("MT_320x320_2x2WG", [16, 16, 32, 1, 1, 10, 10, 2, 2]),
    ("MT_320x288_4x1WG", [16, 16, 32, 1,  1, 5, 18,  4, 1]),
]


# ── Tests: large-MT Solution-level gates ────────────────────────────────────

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


# ── Single-wave WG=(1,1) + large WT VGPR-budget reject ─────────────────────

# (label, mi-9-tuple, depthU). MT = mi_m*wt_m*wg_m × mi_n*wt_n*wg_n.
# Surfaced by the MT x DU x WG sweep on PR #7661: every WG=(1,1) shape
# below builds at NoTailLoop=True (ASEM==DepthU) but the with-tail kernel
# (ASEM<DepthU) overflows the wave-64 256-VGPR budget at codegen with
# vgprs in [272, 280]. None of these are in production yamls today
# (largest WG=(1,1) wavetile in subtile_bf16*.yaml is (4,4), area 16).
# The Solution-level reject preempts the late-stage codegen overflow
# warning so the build set never carries an unbuildable kernel.
_SINGLE_WAVE_REGRESSION_SHAPES = [
    ("MT_112x240_WT_7x15",  [16, 16, 32, 1, 1,  7, 15, 1, 1], 64),
    ("MT_128x208_WT_8x13",  [16, 16, 32, 1, 1,  8, 13, 1, 1], 64),
    ("MT_208x128_WT_13x8",  [16, 16, 32, 1, 1, 13,  8, 1, 1], 64),
    ("MT_240x112_WT_15x7",  [16, 16, 32, 1, 1, 15,  7, 1, 1], 64),
    ("MT_160x160_WT_10x10", [16, 16, 32, 1, 1, 10, 10, 1, 1], 128),
]


class TestSubtileBf16SingleWaveLargeWtRejected:
    """The MT x DU sweep on PR #7661 surfaced 5 (MT, DU, WG=(1,1), WT)
    tuples where the no-tail kernel builds cleanly (ASEM==DepthU,
    NoTailLoop=True) but the with-tail kernel overflows the wave-64
    256-VGPR budget at codegen (vgprs in [272, 280]). The
    `Solution.assignProblemIndependentDerivedParameters` gate rejects
    these proactively so the build never carries a kernel that the
    assembler would later drop with a confusing
    `overflowed resources, msg="too many vgprs"` warning.

    Predicate: `MIWaveGroup == [1, 1] and AssertSummationElementMultiple
    < DepthU and MIWaveTile area >= 100`. The area threshold of 100
    captures the 5 shapes (areas 100, 104, 104, 105, 105) while leaving
    every WG=(1,1) wavetile <= (9, 9) (area 81) accepted -- the largest
    WG=(1,1) wavetile in any current production yaml is (4, 4) so the
    gate is structurally above the production envelope.
    """

    @pytest.mark.parametrize("label,mi,depthU", _SINGLE_WAVE_REGRESSION_SHAPES)
    @pytest.mark.parametrize("asem", [1, 2])
    def test_single_wave_with_tail_is_rejected(self, label, mi, depthU, asem):
        state = _build_largemt_state(mi=mi, asem=asem, pgr=2, depthU=depthU)
        _run_assign_problem_independent(state)
        assert state.get("Valid") is False, (
            f"{label} ASEM={asem} DepthU={depthU}: expected Solution-level "
            f"reject (state['Valid']=False) because single-wave WG=(1,1) + "
            f"WT area >= 100 + tail overflows the 256-VGPR budget at "
            f"codegen. Got state['Valid']={state.get('Valid')}."
        )


class TestSubtileBf16SingleWaveLargeWtAcceptedAtAsemEqDu:
    """The reject predicate gates on `ASEM < DepthU`. With `ASEM ==
    DepthU` the kernel emits with `NoTailLoop=True` (no tail body) and
    the no-tail baseline fits within the 256-VGPR budget (e.g. MT
    112x240 builds at 249 VGPRs in this mode). The Solution gate must
    NOT reject these `ASEM==DepthU` configurations -- they're the
    aligned-K hot path and represent the kernels actually useful to
    ship for these problem shapes (a producer who wants these MTs
    would simply pin ASEM to the K alignment).
    """

    @pytest.mark.parametrize("label,mi,depthU", _SINGLE_WAVE_REGRESSION_SHAPES)
    def test_single_wave_no_tail_is_accepted(self, label, mi, depthU):
        state = _build_largemt_state(mi=mi, asem=depthU, pgr=2, depthU=depthU)
        _run_assign_problem_independent(state)
        assert state.get("Valid") is not False, (
            f"{label} ASEM={depthU} DepthU={depthU}: expected the "
            f"NoTailLoop=True (aligned-K) path to be accepted at the "
            f"Solution gate. The reject predicate must gate on "
            f"ASEM<DepthU, not on the (MT, WG) shape alone, so the "
            f"aligned-K kernel remains buildable for downstream "
            f"producers who pin K to the DepthU alignment. Got "
            f"state['Valid']={state.get('Valid')}."
        )


class TestSubtileBf16MultiWaveLargeWtNotRejected:
    """The reject predicate gates specifically on `MIWaveGroup ==
    [1, 1]`. Multi-wave large-MT shapes (e.g. MT 320x320 with
    WG=(2,2)+WT=(10,10), area 100; MT 320x288 with WG=(4,1)+WT=(5,18),
    area 90) must NOT be rejected by this gate -- their D-accumulator
    is split across multiple waves so the per-wave VGPR footprint is
    well below the budget even with the tail scaffold. Pinning this
    explicitly is the dual to `TestSubtileBf16LargeMTNotRejected`
    above: that test pins acceptance over `WG != (1,1)` shapes against
    the pre-existing gates; this test pins it against the NEW
    single-wave gate.
    """

    @pytest.mark.parametrize("label,mi", _LARGE_MT_SHAPES)
    @pytest.mark.parametrize("asem", [1, 2, 8])
    def test_multi_wave_large_wt_not_rejected_by_single_wave_gate(
        self, label, mi, asem
    ):
        state = _build_largemt_state(mi=mi, asem=asem, pgr=2)
        _run_assign_problem_independent(state)
        assert state.get("Valid") is not False, (
            f"Multi-wave large MT {label} ASEM={asem}: state['Valid'] = "
            f"{state.get('Valid')}. The new single-wave WG=(1,1) reject "
            f"gate must not catch multi-wave shapes -- the per-wave "
            f"D-accumulator pressure for WG != (1, 1) is well below "
            f"the 256-VGPR budget even with the tail scaffold's "
            f"persistent state."
        )


class TestSubtileBf16SmallSingleWaveNotRejected:
    """The reject predicate gates on `MIWaveTile area >= 100`. The
    current production envelope uses WG=(1,1) only with small
    wavetiles (largest is (4, 4) at MT 64x64 in `subtile_bf16*.yaml`).
    These small WG=(1,1) shapes must NOT be caught by the gate -- they
    have a small D-accumulator that leaves ample room for the tail
    scaffold. Pinning the (4, 4) case (area 16, well under 100) and
    a midrange (8, 8) case (area 64, still under 100) catches a
    regression that lowered the area threshold and accidentally
    rejected production shapes.
    """

    @pytest.mark.parametrize(
        "label,mi",
        [
            ("MT_64x64_WT_4x4",   [16, 16, 32, 1, 1, 4, 4, 1, 1]),
            ("MT_128x128_WT_8x8", [16, 16, 32, 1, 1, 8, 8, 1, 1]),
        ],
    )
    @pytest.mark.parametrize("asem", [1, 2, 8])
    def test_small_single_wave_with_tail_accepted(self, label, mi, asem):
        state = _build_largemt_state(mi=mi, asem=asem, pgr=2, depthU=64)
        _run_assign_problem_independent(state)
        assert state.get("Valid") is not False, (
            f"Small single-wave {label} ASEM={asem}: state['Valid'] = "
            f"{state.get('Valid')}. The new WG=(1,1) + WT area >= 100 "
            f"reject must not catch wavetiles below the threshold; "
            f"the production envelope (WG=(1,1) with WT <= (4,4)) "
            f"must remain buildable."
        )
