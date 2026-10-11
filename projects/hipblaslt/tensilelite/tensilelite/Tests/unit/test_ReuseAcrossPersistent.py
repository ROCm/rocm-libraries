################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
################################################################################
"""Parameter registration and solution-validation guards for ReuseAcrossPersistent.

RAP keeps A (and its MX scales) resident in VGPRs for the whole K extent across
persistent iterations. That is only sound when every tile a workgroup visits
reads the same A. The solution-independent half of that contract is a guard in
``Solution.depthUIteration``; the problem-size half is the runtime predicates
emitted from ``Contractions.ProblemPredicate.CompoundPredicates``. Both are
covered here.

The reject harness mirrors ``test_halfplr_streamk_rejects``: real gfx1250
capability maps from ``makeIsaInfoMap`` and a real assembler feed
``Solution.__init__``, which runs ``assignDerivedParameters`` end-to-end, and the
reject reason is captured from stdout via ``capsys``.

The last sections execute emitted scalar code on the CPU: the entry guard's peek
against DefaultWGM tile by tile, and the phase walk against a model of its plan.
"""

import collections
import copy
import itertools
import re
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from rocisa.code import Label, Module, TextBlock
from rocisa.container import sgpr

from tensilelite.Components.TDMFuse import tdmKDimField, tdmKDimTrackable
from tensilelite.KernelWriterAssembly import KernelWriterAssembly
from tensilelite.Components.PersistentLoop import PersistentKernelState
import tensilelite.Components.WorkAssignment as work_assignment
import tensilelite.Components.WorkGroupMappingAlgos as wgm_algos
from tensilelite.Component import Component
from tensilelite.Components.TileProcessingStrategy import DataParallel
from tensilelite.Components.WorkAssignment import StaticGrid
from tensilelite.Components.WorkGroupMappingAlgos import DefaultWGM

from tensilelite.Common.GlobalParameters import defaultSolution
from tensilelite.Common.RequiredParameters import getRequiredParametersMin
from tensilelite.Common.ValidParameters import validParameters
from tensilelite.SolutionStructs.Naming import getParameterNameAbbreviation
from tensilelite.SolutionStructs.Solution import Solution, validateParameterTypes

pytestmark = pytest.mark.unit


# Sibling unit tests mutate the process-global defaultSolution in place, which
# makes Solution.__init__'s `for key in defaultSolution` loop order-dependent.
_PRISTINE_DEFAULT_SOLUTION = copy.deepcopy(dict(defaultSolution))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def test_rap_is_valid_solution_parameter():
    assert validParameters["ReuseAcrossPersistent"] == [0, 1]
    assert defaultSolution["ReuseAcrossPersistent"] == 0
    validateParameterTypes({"ReuseAcrossPersistent": 1})


def test_rap_is_in_the_min_roster():
    # Without this, RAP=0 and RAP=1 hash to the same kernel name and one of the
    # two is silently dropped as a duplicate, so a [0,1] fork would benchmark
    # the same kernel twice.
    assert "ReuseAcrossPersistent" in getRequiredParametersMin()


def test_rap_name_abbreviation_is_unique():
    abbreviations = collections.defaultdict(list)
    for key in getRequiredParametersMin():
        abbreviations[getParameterNameAbbreviation(key)].append(key)
    assert abbreviations["RAP"] == ["ReuseAcrossPersistent"]


# ---------------------------------------------------------------------------
# Codegen gating. Emitters read kernel["ReuseAcrossPersistent"] directly, the
# way they read kernel["HalfPLR"], so RAP has no codegen predicate. PAP consumes
# its validated capability through the shared persistent state owner.
# ---------------------------------------------------------------------------
def test_rap_has_no_codegen_predicate_to_disagree_with_derivation():
    """Two places deciding whether RAP is on is one too many.

    While codegen recomputed the preconditions, a solution could carry the flag
    and still be emitted plain, so a guard missing from derivation showed up as a
    kernel named RAP that behaved like RAP 0 -- and the benchmark, which ranks on
    gflops alone, would rank it against a real one. Emitters now read the flag,
    which makes assignDerivedParameters the only authority; this pins that there
    is nothing left for it to drift against.
    """
    from tensilelite.KernelWriter import KernelWriter

    assert not hasattr(KernelWriter, "isReuseAcrossPersistentEnabled")


def _codegenKernel(**overrides):
    kernel = {
        "ReuseAcrossPersistent": 1,
        "PrefetchAcrossPersistent": 1,
        "TileProcessingStrategy": "DataParallel",
        "WorkAssignment": "StaticGrid",
        "PrefetchGlobalRead": 2,
        "UseCustomMainLoopSchedule": 0,
        "SuppressNoLoadLoop": True,
        "HalfPLR": 0,
    }
    kernel.update(overrides)
    return kernel


def _papEnabled(**overrides):
    from tensilelite.KernelWriter import KernelWriter

    assert issubclass(KernelWriter, PersistentKernelState)
    return PersistentKernelState.isPrefetchAcrossPersistentEnabled(
        SimpleNamespace(), _codegenKernel(**overrides)
    )


def test_pap_codegen_consumes_the_validated_capability():
    """Backend conditions are resolved once, before code generation."""
    assert _papEnabled(_PrefetchAcrossPersistentEnabled=True)
    assert _papEnabled(_PrefetchAcrossPersistentEnabled=True, ReuseAcrossPersistent=0, HalfPLR=1)
    assert not _papEnabled(_PrefetchAcrossPersistentEnabled=False)
    assert not _papEnabled(_PrefetchAcrossPersistentEnabled=False, ReuseAcrossPersistent=0)


def test_pap_is_off_when_its_own_flag_is_off_whatever_rap_says():
    # RAP 1 with PAP 0 is supported, so RAP must not switch PAP back on.
    assert not _papEnabled(PrefetchAcrossPersistent=0)
    assert not _papEnabled(PrefetchAcrossPersistent=0, ReuseAcrossPersistent=0)


# ---------------------------------------------------------------------------
# Toolchain fixtures (real gfx1250 caps + assembler)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def gfx1250_iim():
    from tensilelite.Common.Architectures import gfxToIsa
    from tensilelite.Common.Capabilities import makeIsaInfoMap
    from tensilelite.Toolchain.Validators import validateToolchain

    cxx = validateToolchain("amdclang++")
    isa = gfxToIsa("gfx1250")
    iim = makeIsaInfoMap([isa], cxx)
    if not iim[isa].asmCaps["SupportedISA"]:
        pytest.skip("amdclang++ in this environment does not support gfx1250")
    return iim


@pytest.fixture(scope="module")
def assembler():
    from tensilelite.Toolchain.Assembly import makeAssemblyToolchain
    from tensilelite.Toolchain.Validators import validateToolchain, ToolchainDefaults

    cxx = validateToolchain("amdclang++")
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    return makeAssemblyToolchain(cxx, bundler, "default").assembler


@pytest.fixture(scope="module")
def _gp_gfx1250(gfx1250_iim):
    """Assign process-global parameters for gfx1250; restore after module."""
    from tensilelite.Common.GlobalParameters import globalParameters, assignGlobalParameters

    saved_gp = copy.deepcopy(dict(globalParameters))
    saved_vp = copy.deepcopy(dict(validParameters))
    saved_ds = copy.deepcopy(dict(defaultSolution))
    defaultSolution.clear()
    defaultSolution.update(copy.deepcopy(_PRISTINE_DEFAULT_SOLUTION))
    assignGlobalParameters({}, gfx1250_iim)
    yield
    globalParameters.clear()
    globalParameters.update(saved_gp)
    validParameters.clear()
    validParameters.update(saved_vp)
    defaultSolution.clear()
    defaultSolution.update(saved_ds)


# ---------------------------------------------------------------------------
# Base solution: the MT64x256 RAP candidate, minus the MX scaling (irrelevant to
# these guards). Each negative test flips exactly one knob.
# ---------------------------------------------------------------------------
def _make_params(gfx1250_iim, **overrides):
    from tensilelite.Common.Architectures import gfxToIsa
    from tensilelite.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
    )

    isa = gfxToIsa("gfx1250")
    # [M, N, K, B, ?, MIWaveTile0, MIWaveTile1, WaveGroup0, WaveGroup1]
    # -> MacroTile 64x256, NumWaves 4.
    mi = [16, 16, 128, 1, 1, 2, 8, 2, 2]
    pt = overrides.pop("ProblemType", {})
    problem_type = {
        "OperationType": "GEMM",
        "DataType": "F8",
        "DestDataType": "s",
        "ComputeDataType": "s",
        "HighPrecisionAccumulate": True,
        "TransposeA": True,
        "TransposeB": False,
        "UseBeta": True,
        "Batched": True,
    }
    problem_type.update(pt)

    params = {
        "ProblemType": problem_type,
        "ISA": isa,
        "MatrixInstruction": mi,
        "WorkGroup": [16, 16, 1],
        "WavefrontSize": 32,
        "DepthU": 256,
        "AssertSummationElementMultiple": 256,
        "KernelLanguage": "Assembly",
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 1,
        "ScheduleIterAlg": 4,
        "StaggerU": 0,
        "GlobalSplitU": 0,
        "InnerUnroll": 1,
        "TransposeLDS": -1,
        "LdsPadA": -1,
        "LdsPadB": -1,
        "LdsBlockSizePerPadA": -1,
        "LdsBlockSizePerPadB": -1,
        "1LDSBuffer": 0,
        "VectorWidthA": -1,
        "VectorWidthB": -1,
        "StoreVectorWidth": -1,
        "GlobalReadVectorWidthA": -1,
        "GlobalReadVectorWidthB": -1,
        "LocalReadVectorWidth": -1,
        "SourceSwap": True,
        "ExpandPointerSwap": False,
        "GlobalSplitUAlgorithm": "MultipleBuffer",
        "TDMInst": 3,
        "LDSTrInst": False,
        "TileProcessingStrategy": "DataParallel",
        "WorkAssignment": "StaticGrid",
        "PrefetchAcrossPersistent": 1,
        "ReuseAcrossPersistent": 1,
        "UseSubtileImpl": False,
        "StoreRemapVectorWidth": 0,
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "DirectToVgprSparseMetadata": False,
        "WorkGroupMapping": 1,
        "HalfPLR": 0,
    }
    params.update(overrides)
    mi_params = matrixInstructionToMIParameters(
        mi, isa, params["WavefrontSize"], problem_type, params["WorkGroup"], gfx1250_iim
    )
    params.update(mi_params)
    return params


def _derive(gfx1250_iim, assembler, capsys, **overrides):
    """Construct a Solution with reject printing on; return (sol, stdout)."""
    params = _make_params(gfx1250_iim, **overrides)
    sol = Solution(params, False, True, False, assembler, gfx1250_iim)
    return sol, capsys.readouterr().out


def _pin_store_budget(monkeypatch, kTiles):
    """Make the store-budget model report kTiles, so a test can pick the count.

    The derivation takes max(PrefetchGlobalRead + 1, model), so pinning the model
    is how a test reaches a specific k without a production override to set it.
    """
    monkeypatch.setattr(
        Solution, "rapMaxResidentKTiles", staticmethod(lambda *a, **k: kTiles)
    )


# ---------------------------------------------------------------------------
# Positive: the known-good combination must be accepted (guard vs over-reject).
# ---------------------------------------------------------------------------
def test_rap_resident_ktiles_never_falls_below_the_section_floor(
    _gp_gfx1250, gfx1250_iim, assembler, capsys
):
    """The emit-once sections supply PrefetchGlobalRead + 1 k-tiles for free.

    rapMaxResidentKTiles may raise the count above that, but never lower it: a
    model that guessed low must only leave K on the table, not take away a k that
    already worked.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert sol["_RAPNumResidentKTiles"] >= sol["PrefetchGlobalRead"] + 1


def test_rap_requires_a_persistent_loop(
    _gp_gfx1250, gfx1250_iim, assembler, capsys
):
    """An explicit RAP request requires a persistent execution strategy."""
    with pytest.raises(ValueError, match="ReuseAcrossPersistent requires a persistent"):
        _derive(gfx1250_iim, assembler, capsys, TileProcessingStrategy="None",
                PrefetchAcrossPersistent=0, GlobalSplitU=1)


def test_rap_off_does_not_derive_a_resident_block(
    _gp_gfx1250, gfx1250_iim, assembler, capsys
):
    sol, out = _derive(gfx1250_iim, assembler, capsys, ReuseAcrossPersistent=0)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert "_RAPNumResidentKTiles" not in sol._state


def test_rap_accepts_a_kernel_without_prefetch_across_persistent(
    _gp_gfx1250, gfx1250_iim, assembler, capsys
):
    """RAP rides on the persistent loop, not on PAP.

    StreamK 3 with DP-only tiles is what gives a workgroup several tiles in a
    row, which is the whole basis for holding A. PrefetchAcrossPersistent
    overlaps the next tile's loads with this tile's compute on that same loop --
    useful, but independent. RAP used to require it only because the two were
    derived together.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys, PrefetchAcrossPersistent=0)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert sol["_RAPNumResidentKTiles"] >= sol["PrefetchGlobalRead"] + 1


def test_rap_accepts_the_shallower_prefetch_base(
    _gp_gfx1250, gfx1250_iim, assembler, capsys
):
    """Pins the PGR=1 accept path that the ExpandPointerSwap reject below builds on.

    PGR=1 lowers the floor to 2 but not the store's slack, so the count itself is
    whatever the model affords -- only the floor moves with prefetch depth.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys, PrefetchGlobalRead=1)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert sol["_RAPNumResidentKTiles"] >= 2


def test_rap_accepts_a_swept_ktile_count_above_the_section_floor(
    _gp_gfx1250, gfx1250_iim, assembler, capsys, monkeypatch
):
    """Above the floor the count is free: the loop shell emits one body per tile."""
    _pin_store_budget(monkeypatch, 4)
    sol, out = _derive(gfx1250_iim, assembler, capsys)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert sol["_RAPNumResidentKTiles"] == 4


def test_rap_never_derives_fewer_ktiles_than_the_section_count(
    _gp_gfx1250, gfx1250_iim, assembler, capsys, monkeypatch
):
    """Below PrefetchGlobalRead + 1 the loop is never entered.

    The loop-entry guard skips straight to the pre-loop escapes when there are
    fewer than PGR + 1 k-tiles to walk, and those escapes reload A over the
    resident registers. RAP does not implement them, so the count must never land
    there -- the sections supply PGR + 1 whether the store budget affords them or
    not, and the codegen guard is what refuses the kernel when even those do not
    fit.

    Pinned below the floor on purpose: the derivation has to raise it back, not
    pass it through.
    """
    _pin_store_budget(monkeypatch, 1)
    sol, out = _derive(gfx1250_iim, assembler, capsys)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert sol["_RAPNumResidentKTiles"] == sol["PrefetchGlobalRead"] + 1


def test_rap_leaves_both_free_dims_alone(
    _gp_gfx1250, gfx1250_iim, assembler, capsys
):
    """K is the only dim RAP constrains. M and N carry nothing.

    The entry guard carries what the M == MacroTile0 predicate used to: which A a
    tile needs is (M-tile, batch), and it compares both. Whatever a partial
    M-tile's rows past M hold in the resident registers cannot reach a C element
    that matters -- row m of C depends only on row m of A, and the edge store
    masks the rows past M. An N edge reaches only the store.

    Asserting the absence rather than a tag, because re-adding a constraint on
    either dim is how the support would silently narrow again. Absence is not
    unconstrained: AssertFree0/1ElementMultiple are emitted elsewhere, still
    apply, and are not RAP's to relax -- hence only the compound set here.
    """
    import tensilelite.Contractions as C

    sol, out = _derive(gfx1250_iim, assembler, capsys)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"

    problemType = C.ProblemType.FromOriginalState(sol["ProblemType"])
    preds = C.ProblemPredicate.CompoundPredicates(sol, problemType)

    freePreds = {(p.index, p.tag, p.value) for p in preds if p.index in (0, 1)}
    assert freePreds == set(), f"M and N should carry no RAP predicate, got {freePreds}"

    # And K still does, so this is not just an empty predicate list.
    kIdx = sol["ProblemType"]["NumIndicesC"]
    assert any(p.index == kIdx for p in preds), "K must still be constrained"


def test_rap_k_predicates_admit_every_k_up_to_the_resident_block(
    _gp_gfx1250, gfx1250_iim, assembler, capsys, monkeypatch
):
    """K spans 1 .. kTiles * DepthU, and both ends are one-off sensitive.

    The loop runs ceil(K / DepthU) resident k-tiles and the TDM clamp stops a
    partial last one at K, so K needs no DepthU multiple -- ASEM's own
    BoundSizeMultiple sets the granularity. Both bounds are silent when wrong, and
    wrong in opposite directions: too high a ceiling admits a K whose top k-tile
    was never filled and multiplies whatever the previous tile left in those
    registers, while admitting K = 0 skips the loop, and with it the clone that
    zeroes C. Neither shows up as a build failure.

    SizeGreaterThan and SizeLessThan are strict in the C++ evaluator
    (ContractionProblemPredicates.hpp), hence 0 and the ceiling + 1.
    """
    import tensilelite.Contractions as C

    _pin_store_budget(monkeypatch, 4)
    sol, out = _derive(gfx1250_iim, assembler, capsys, AssertSummationElementMultiple=32)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"

    depthU = sol["DepthU"]
    kTiles = sol["_RAPNumResidentKTiles"]
    kIdx = sol["ProblemType"]["NumIndicesC"]

    problemType = C.ProblemType.FromOriginalState(sol["ProblemType"])
    preds = C.ProblemPredicate.CompoundPredicates(sol, problemType)
    kPreds = {(p.tag, p.value) for p in preds if p.index == kIdx}

    assert kPreds == {("SizeGreaterThan", 0), ("SizeLessThan", kTiles * depthU + 1)}

    # Spelled out as the accepted set, so a bound that drifts by one fails here
    # even if someone rewrites the pair above to match.
    accepted = {k for k in range(0, (kTiles + 2) * depthU + 1)
                if k > 0 and k < kTiles * depthU + 1}
    assert accepted == set(range(1, kTiles * depthU + 1))


@pytest.mark.parametrize(
    "rap, asem, folds, noTailLoop",
    [
        pytest.param(1, 32, True, True, id="rap_with_a_remainder"),
        pytest.param(1, 256, False, True, id="rap_without_a_remainder"),
        pytest.param(0, 32, False, False, id="no_rap_with_a_remainder"),
        pytest.param(0, 256, False, True, id="no_rap_without_a_remainder"),
    ],
)
def test_tail_folding_is_rap_meeting_a_k_remainder(
    _gp_gfx1250, gfx1250_iim, assembler, capsys, rap, asem, folds, noTailLoop
):
    """Tail folding is how RAP takes a K remainder, and nothing else folds.

    NoTailLoop and _TailFolding are derived apart -- NoTailLoop inside
    depthUIteration, _TailFolding after the ASEM overrides that follow it -- so
    this pins that they agree: a RAP kernel never has a tail loop, and folds
    exactly when its ASEM leaves a remainder. NoTailLoop then means only that no
    tail is emitted: the partial last k-tile takes a resident slot of its own.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       ReuseAcrossPersistent=rap, AssertSummationElementMultiple=asem)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert sol.get("_TailFolding", False) is folds
    assert sol["NoTailLoop"] is noTailLoop
    # Only RAP folds, so only RAP solutions carry the key: every other solution
    # keeps the shape the Solution characterization snapshots pin.
    assert ("_TailFolding" in sol) is bool(rap)


def test_a_rap_kernel_never_reaches_the_tail_loop_emitter(
    _gp_gfx1250, gfx1250_iim, assembler
):
    """Derivation never sends RAP to the tail loop; codegen refuses if it ever does.

    A tail loop under RAP needs registers RAP does not budget -- the resident A
    blocks leave it no ValuA to load into -- so a derivation change that let one
    through would otherwise surface as a pool overflow, or as a kernel computing
    from the wrong registers.
    """
    import shutil
    import rocisa
    from tensilelite.Common.Types import DebugConfig
    from tensilelite.SolutionStructs.Naming import getKernelFileBase
    from tensilelite.TensileCreateLibrary.Run import generateKernelObjectsFromSolutions
    from tensilelite.Tests.rocisa_test_state import preserve_rocisa_kernel_state

    sol = Solution(_make_params(gfx1250_iim, AssertSummationElementMultiple=32),
                   False, False, False, assembler, gfx1250_iim)
    assert sol.get("Valid") is True
    with preserve_rocisa_kernel_state():
        (kernel,) = generateKernelObjectsFromSolutions([sol])
        ri = rocisa.rocIsa.getInstance()
        ri.init(tuple(kernel["ISA"]), shutil.which("amdclang++") or "/usr/bin/amdclang++")
        ri.setKernel(tuple(kernel["ISA"]), kernel["WavefrontSize"])
        kernel.duplicate = False
        kernel["BaseName"] = getKernelFileBase(False, kernel)
        kernel["NoTailLoop"] = False
        kwa = KernelWriterAssembly(assembler, DebugConfig())
        kwa.setRocIsa(ri.getData(), ri.getOutputOptions())
        with pytest.raises(AssertionError, match="tail-folds the K remainder"):
            kwa.getSourceFileString(kernel)


_MX_PROBLEM_TYPE = {
    "MacDataTypeA": "F8", "MacDataTypeB": "F8",
    "MXBlockA": 32, "MXBlockB": 32, "DataTypeMXSA": "E8", "DataTypeMXSB": "E8",
}
_K_FIELD_REJECT = ("ReuseAcrossPersistent with AssertSummationElementMultiple % DepthU != 0 "
                   "requires TDM descriptor sets that keep K in one field")


def test_tail_folding_sees_the_asem_the_fp4_override_leaves(
    _gp_gfx1250, gfx1250_iim, assembler, capsys
):
    """Asked for 256, an F8xF4 kernel ends up at 32 and has to fold.

    The FP4 override runs after depthUIteration, where NoTailLoop is derived, so
    _TailFolding is derived after the override instead. Read off the config's
    256 it would come out False, and the partial last k-tile the predicate
    admits would load past K with no K dim shrink to clamp it. F8xF4 can fold at
    all because A and B, though they step K by different amounts, both count it
    in bytes: each wave steps by its own address increment.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys, AssertSummationElementMultiple=256,
                       ProblemType=dict(_MX_PROBLEM_TYPE, MacDataTypeB="F4"))
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert sol["AssertSummationElementMultiple"] == 32
    assert sol["_TailFolding"] is True
    assert sol["NoTailLoop"] is True


@pytest.mark.parametrize(
    "overrides",
    [
        # TDM moves a tile-major B only with the transposing LDS read.
        pytest.param({"ProblemType": {"TransposeB": True}, "LDSTrInst": True},
                     id="tile_major_b"),
        # PAP rejects every TDMFuse other than 0 on its own, so ask without it.
        pytest.param({"ProblemType": dict(_MX_PROBLEM_TYPE), "TDMFuse": 1,
                      "PrefetchAcrossPersistent": 0}, id="tdmfuse_1"),
    ],
)
def test_rap_rejects_a_k_remainder_its_descriptors_cannot_clamp(
    _gp_gfx1250, gfx1250_iim, assembler, capsys, overrides
):
    """One subtract per descriptor set shrinks K only if every member keeps it alike.

    Each case is accepted without a remainder, so what rejects it is the remainder.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys, **copy.deepcopy(overrides))
    assert sol.get("Valid") is True, f"control rejected with: {out!r}"
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       AssertSummationElementMultiple=32, **copy.deepcopy(overrides))
    assert sol.get("Valid") is False, f"expected reject for {overrides}"
    assert _K_FIELD_REJECT in out, f"rejected for another reason: {out!r}"


def test_rap_checks_the_k_remainder_against_the_final_asem(
    _gp_gfx1250, gfx1250_iim, assembler, capsys
):
    """FP4 overrides ASEM after RAP's other checks have run.

    Asked for 256, an FP4 kernel still ends up at 32, so it carries a remainder
    whatever the config said. Checked against the config's value, this grouping
    would pass derivation and fail in codegen instead.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys, AssertSummationElementMultiple=256,
                       TDMFuse=1, PrefetchAcrossPersistent=0,
                       ProblemType=dict(_MX_PROBLEM_TYPE, MacDataTypeB="F4"))
    assert sol["AssertSummationElementMultiple"] == 32
    assert sol.get("Valid") is False, "the remainder check saw the config's ASEM"
    assert _K_FIELD_REJECT in out, f"rejected for another reason: {out!r}"


# ---------------------------------------------------------------------------
# Negative: each precondition of "every tile reads the same A" is enforced.
#
# These are the whole contract, not a sample of it. Codegen reads the flag and
# recomputes nothing, so a precondition that stops rejecting here reaches the
# emitters rather than quietly turning RAP off.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "overrides, reason",
    [
        pytest.param(
            {"TileProcessingStrategy": "StreamK"},
            "ReuseAcrossPersistent requires DataParallel/StaticGrid",
            id="without_dp_only",
        ),
        pytest.param(
            # PGR>=2 forces ExpandPointerSwap off before the guard runs, so this
            # reject is only reachable at PGR=1 (accepted on its own above).
            {"PrefetchGlobalRead": 1, "ExpandPointerSwap": True},
            "ReuseAcrossPersistent requires ExpandPointerSwap = 0",
            id="with_expand_pointer_swap",
        ),
        pytest.param(
            {"InnerUnroll": 2},
            "ReuseAcrossPersistent requires InnerUnroll = 1",
            id="with_inner_unroll",
        ),
        pytest.param(
            # Asked with PAP off, or the subtile path's own gfx950 audit gate
            # rejects first and this guard is never reached.
            {"UseSubtileImpl": True, "PrefetchAcrossPersistent": 0},
            "ReuseAcrossPersistent is not implemented for the subtile path",
            id="with_subtile",
        ),
    ],
)
def test_rap_rejects_unsupported_combinations(
    _gp_gfx1250, gfx1250_iim, assembler, capsys, overrides, reason
):
    sol, out = _derive(gfx1250_iim, assembler, capsys, **overrides)
    assert sol.get("Valid") is False, f"expected reject for {overrides}"
    assert reason in out, f"expected {reason!r} in reject output, got: {out!r}"


# ---------------------------------------------------------------------------
# StaggerU. RAP holds A across every tile the persistent workgroup visits, so a
# summation start that moves from tile to tile multiplies resident A against a B
# reloaded at a different K offset; mapping 1 was measured returning wrong
# results on gfx1250. Rejecting a non-zero StaggerU would not be enough, because
# the compile-time value is not where the kernel reads its stagger: with
# SupportCustomStaggerU set it reads the runtime field and nothing else, so a
# kernel built at StaggerU 0 still takes whatever the host packs. RAP therefore
# joins the other features that turn the whole path off.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({}, id="already_zero_at_compile_time"),
        pytest.param(
            {"StaggerU": 32, "StaggerUMapping": 1, "StaggerUStride": 256},
            id="asked_for_the_mapping_measured_wrong",
        ),
        pytest.param(
            # Mapping 0 is the same offset for every tile RAP visits, so it would
            # survive on its own. Turned off anyway, and pinned here so the rule
            # is not later narrowed to the mappings that bite.
            {"StaggerU": 32, "StaggerUMapping": 0, "StaggerUStride": 256},
            id="asked_for_an_otherwise_safe_mapping",
        ),
    ],
)
def test_rap_turns_the_runtime_stagger_path_off(
    _gp_gfx1250, gfx1250_iim, assembler, capsys, overrides
):
    # PAP off, or PAP's own TDM disable does this first and the test would pass
    # without RAP having done anything.
    sol, out = _derive(
        gfx1250_iim, assembler, capsys, PrefetchAcrossPersistent=0, **overrides
    )
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert sol["ReuseAcrossPersistent"] == 1
    assert (sol["StaggerU"], sol["StaggerUMapping"], sol["StaggerUStride"]) == (0, 0, 0)
    assert sol["InternalSupportParams"]["SupportCustomStaggerU"] is False


# ---------------------------------------------------------------------------
# WorkGroupMapping. A's address is built from WorkGroup0 after DefaultWGM, and
# the entry guard peeks the same remapped M-tile (see the guard peek tests), so RAP
# keeps the runtime path: the value the kernel sees may come from the solution,
# from the host's prediction (0), or from TENSILE_FIXED_WGM.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rap", [0, 1])
@pytest.mark.parametrize("wgm", [8, 0, 1, -4])
def test_rap_keeps_the_runtime_wgm_path(
    _gp_gfx1250, gfx1250_iim, assembler, capsys, rap, wgm
):
    sol, out = _derive(
        gfx1250_iim, assembler, capsys, ReuseAcrossPersistent=rap, WorkGroupMapping=wgm
    )
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert sol["ReuseAcrossPersistent"] == rap
    assert sol["WorkGroupMapping"] == wgm
    assert sol["InternalSupportParams"]["SupportCustomWGM"] is True


def test_rap_is_rejected_with_space_filling_algo(
    _gp_gfx1250, gfx1250_iim, assembler, capsys
):
    # SpaceFillingCurveWalk is DefaultWGM's sibling, chosen at codegen, and the
    # guard only knows DefaultWGM's remap. Rejecting the compile-time list is
    # enough: which algorithm gets emitted is baked into the assembly, only the
    # WGM *value* arrives from a runtime field.
    sol, out = _derive(gfx1250_iim, assembler, capsys, SpaceFillingAlgo=[0])
    assert sol.get("Valid") is not True, "expected reject"
    assert "SpaceFillingAlgo" in out


# ---------------------------------------------------------------------------
# k_max model. Pinned to the two audited gfx1250 configs, whose true bounds were
# measured by sweeping the resident k-tile count until rapCheckStoreNeutrality
# fired: MacroTile 64x256 clears k=8 and trips at 9, 64x512 clears k=3 at 4.
# Driving the model directly rather than through Solution keeps the arithmetic
# pinned to those numbers; the terms below are the ones codegen actually reported.
# ---------------------------------------------------------------------------
_MODEL_ISA = (12, 5, 0)


def _modelTerms(threadTile1):
    from tensilelite.Common.DataType import DataType

    state = {
        "ThreadTile0": 16, "ThreadTile1": threadTile1,
        "StoreVectorWidth": 2, "BufferStore": True,
        "GlobalSplitU": 0, "_GlobalAccumulation": None,
        "LoopIters": 2, "ClusterLocalRead": 1, "PrefetchLocalRead": 1,
        "MIWaveTileA": 2, "MIInputPerThreadA": 64,
        "MIWaveTileMXSA": 2, "MIInputPerThreadMXSA": 4,
        "GroupLoadStore": False,
    }
    problemType = {
        "ComputeDataType": DataType("s"), "DestDataType": DataType("s"),
        "DataType": DataType("F8"), "MacDataTypeA": DataType("F8"),
        "MXBlockA": 32, "UseInitialStridesCD": False,
    }
    return state, problemType


def _kMax(threadTile1, **problemTypeOverrides):
    state, problemType = _modelTerms(threadTile1)
    problemType.update(problemTypeOverrides)
    isaInfoMap = {_MODEL_ISA: SimpleNamespace(regCaps={"MaxVgpr": 1024})}
    return Solution.rapMaxResidentKTiles(
        state, problemType, _MODEL_ISA, isaInfoMap, False, False
    )


def test_rap_kmax_model_reproduces_the_measured_bound_on_mt64x256():
    # ValuC 128, E 64, V 4, R 68 -> (1024 - 128 - 256 - 40) / 68
    assert _kMax(8) == 8


def test_rap_kmax_model_reproduces_the_measured_bound_on_mt64x512():
    # Twice the accumulators halves the slack: (1024 - 256 - 512 - 40) / 68.
    assert _kMax(16) == 3


def test_rap_kmax_model_declines_when_the_store_cannot_be_priced():
    # A bias adds an address and a data term to numVgprsPerElement. Modelling V
    # low would raise k past what the store can afford, so the model abstains and
    # the caller keeps the section floor.
    assert _kMax(8, UseBias=1) is None


# ---------------------------------------------------------------------------
# Store-neutrality guard. Real configs do not reach the reject branch -- that is
# the point of the guard -- so it is driven directly here.
# ---------------------------------------------------------------------------
_STORE_GUARD_ELEMENTS = 128
_STORE_GUARD_VGPRS_PER_ELEMENT = 4
_STORE_GUARD_WITHHELD = 204  # 3 k-tiles x (64 ValuA + 4 MXSA)


def _runStoreGuard(numVgprAvailable, beta=True, edge=False, withheld=_STORE_GUARD_WITHHELD):
    writer = SimpleNamespace(
        states=SimpleNamespace(overflowedResources=0, rapStoreNeutralityMsg=""),
        rapStoreWithheldVgprs=lambda kernel: withheld,
        rapResidentKTiles=lambda kernel: 3,
    )
    KernelWriterAssembly.rapCheckStoreNeutrality(
        writer,
        {"DepthU": 256},
        [None] * _STORE_GUARD_ELEMENTS,
        SimpleNamespace(numVgprsPerElement=_STORE_GUARD_VGPRS_PER_ELEMENT),
        numVgprAvailable,
        numVgprAvailable // _STORE_GUARD_VGPRS_PER_ELEMENT,
        beta,
        edge,
    )
    return writer.states


def test_store_guard_accepts_a_store_that_still_fits_in_one_batch():
    states = _runStoreGuard(600)
    assert states.overflowedResources == 0


def test_store_guard_rejects_when_residency_splits_the_store():
    states = _runStoreGuard(400)
    assert states.overflowedResources == 9
    # The message has to name both K values, or a tuning run cannot act on it.
    assert "largest store-neutral K is 256" in states.rapStoreNeutralityMsg
    assert "needs 768" in states.rapStoreNeutralityMsg


@pytest.mark.parametrize("beta, edge", [(False, False), (True, True), (False, True)])
def test_store_guard_examines_only_the_beta_non_edge_path(beta, edge):
    # beta=1/edge=0 is the tightest of the non-edge variants, and the only one
    # the guard looks at. Not because the size predicates exclude the rest --
    # since N was unconstrained they do not, and the edge path is reachable -- but
    # because pricing the edge path would cost resident k-tiles for every problem
    # to cover the ones that reach it. Measured, an edge store takes one batch
    # more under RAP than without it; see rapCheckStoreNeutrality for the figures
    # and for what this guard would do if it were let near that path.
    assert _runStoreGuard(400, beta=beta, edge=edge).overflowedResources == 0


# ---------------------------------------------------------------------------
# K dim tracking. A partial last k-tile runs through the resident loop, and the
# TDM descriptor clamps it at K only if each set's K dim, measured from the
# moving base, shrinks as the address advances.
# ---------------------------------------------------------------------------
def _kDimKernel(fuse=0, a="F8", b="F8", mx=True, depthU=256, **overrides):
    from tensilelite.Common.DataType import DataType

    kernel = {
        "TDMFuse": fuse, "NumWaves": 4, "TDMInst": 3, "TDMSplit": False,
        "enableTDMA": True, "enableTDMB": True, "UseSubtileImpl": False,
        "DepthU": depthU, "MatrixInstK": 128,
        "ReuseAcrossPersistent": 1, "AssertSummationElementMultiple": 32,
        "ProblemType": {
            "DataTypeA": DataType(a), "DataTypeB": DataType(b),
            "TLUA": False, "TLUB": False, "Sparse": 0,
            "MXBlockA": 32 if mx else 0, "MXBlockB": 32 if mx else 0,
        },
    }
    kernel.update(overrides)
    return kernel


@pytest.mark.parametrize("kernelArgs, trackable, fields", [
    pytest.param({}, True, {"A": (1, 256), "MXSB": (2, 2)}, id="mx_f8_default_grouping"),
    # FP4 packs two elements per byte and the dim counts bytes. 256 bytes and 128
    # bytes: each is its own address increment.
    pytest.param({"b": "F4"}, True, {"B": (1, 128)}, id="f8_by_f4_both_in_bytes"),
    pytest.param({"fuse": 1}, False, {}, id="tdm_fuse_1"),
    pytest.param({"fuse": 2}, False, {}, id="tdm_fuse_2"),
    pytest.param({"fuse": 3}, False, {}, id="tdm_fuse_3"),
    pytest.param({"mx": False}, True, {}, id="f8_no_mx"),
    pytest.param({"a": "F4", "b": "F4"}, True, {}, id="f4_by_f4"),
    # 256 elements either way, so one step fits both.
    pytest.param({"b": "H", "mx": False}, True, {}, id="f8_by_half_same_elements"),
    # 128 bytes against 256 half elements: no single register holds both.
    pytest.param({"a": "F4", "b": "H", "mx": False}, False, {}, id="f4_by_half"),
    pytest.param({"tileMajorB": True}, False, {}, id="tile_major_b"),
])
def test_tdm_k_dim_fields_and_tracking(kernelArgs, trackable, fields):
    """Where each tensor keeps K in its descriptor, and whether a set can track it.

    One subtract per descriptor set clamps the partial last k-tile, so every
    member has to keep K in one field and step it alike: by the same element
    count, or in bytes by its own address increment. Only the default grouping
    (TDMFuse 0) keeps one K field per set.
    """
    args = dict(kernelArgs)
    tileMajorB = args.pop("tileMajorB", False)
    kernel = _kDimKernel(**args)
    if tileMajorB:
        kernel["ProblemType"]["TLUB"] = True
    for tensor, field in fields.items():
        assert tdmKDimField(kernel, tensor) == field
    assert tdmKDimTrackable(kernel) is trackable


class _ShrinkWriter:
    """Just enough writer to render tailFoldingShrinkTdmKDim."""

    isTdmWaveSeparated = KernelWriterAssembly.isTdmWaveSeparated
    _tdmPairedParityOrder = KernelWriterAssembly._tdmPairedParityOrder
    tailFoldingMxsKSplitOffsetExceedsOneGroup = KernelWriterAssembly.tailFoldingMxsKSplitOffsetExceedsOneGroup

    def rapResidentKTiles(self, kernel):
        return 8

    @contextmanager
    def allocTmpSgpr(self, num, alignment=None, tag=None):
        yield SimpleNamespace(idx=90, size=num)


def _shrink(kernel, tcA, tcB):
    return str(KernelWriterAssembly.tailFoldingShrinkTdmKDim(
        _ShrinkWriter(), kernel, {"tensorChar": tcA}, {"tensorChar": tcB}))


@pytest.mark.parametrize("kernelArgs, tensors, emitted, absent", [
    pytest.param({}, ("A", "B"),
                 ["s_sub_u32 s[sgprtdmAGroup1+1], s[sgprtdmAGroup1+1], 0x1000000"], ["s_and_b32"],
                 id="data_set_shrinks_dim0_by_one_k_tile"),
    pytest.param({"b": "F4"}, ("A", "B"),
                 ["s[sgprtdmABIncs]", "s_sub_u32 s[sgprtdmAGroup1+1], s[sgprtdmAGroup1+1], s90"], [],
                 id="data_set_in_bytes_shrinks_by_the_wave_address_increment"),
    # DepthU 256 is two K groups, one per scale wave: the second wave starts a
    # group in and can never outrun the last k-tile's one-or-two groups.
    pytest.param({}, ("MXSA", "MXSB"),
                 ["s_sub_u32 s[sgprtdmMXSAGroup1+2], s[sgprtdmMXSAGroup1+2], 0x20000"], ["s_and_b32"],
                 id="scale_set_shrinks_dim1_by_its_k_groups"),
    # DepthU 512 gives each of the two scale waves two groups, so the second
    # starts two groups in and runs out when the last k-tile has only one.
    pytest.param({"depthU": 512}, ("MXSA", "MXSB"),
                 ["s_sub_u32 s[sgprtdmMXSAGroup1+2], s[sgprtdmMXSAGroup1+2], 0x40000",
                  "s_and_b32 s[sgprtdmMXSAGroup1+2], s[sgprtdmMXSAGroup1+2], s90"], [],
                 id="scale_wave_offset_past_one_group_clamps_instead_of_wrapping"),
])
def test_tail_folding_shrinks_each_sets_k_dim(kernelArgs, tensors, emitted, absent):
    rendered = _shrink(_kDimKernel(**kernelArgs), *tensors)
    for text in emitted:
        assert text in rendered
    for text in absent:
        assert text not in rendered


# ---------------------------------------------------------------------------
# Emitted scalar code on the CPU. Covers the SALU subset the guard peek and the
# phase walk emit, with labels and branches. The scalar divide macros expand to
# VALU sequences; _exact_scalar_divides swaps them for a marker _run_scalar
# evaluates exactly, so a test checks the arithmetic around a divide rather than
# the divide itself.
# ---------------------------------------------------------------------------
_MASK = 0xFFFFFFFF
_DIVIDE = re.compile(r"/\*DIVIDE q=(\S+) d=(\S+) div=(\S+) r=(\S+) rem=([01])\*/")
_DIVIDE_MACROS = ("scalarUInt32DivideAndRemainder", "scalarUInt24DivideAndRemainder")


def _exact_scalar_divides(monkeypatch, *modules):
    """Make the divide macros imported by ``modules`` emit exact-divide markers."""
    def divide(qReg, dReg, divReg, rReg, tmpVgprRes=None, wavewidth=64, doRemainder=True, **kwargs):
        module = Module("exact divide")
        module.add(TextBlock("/*DIVIDE q=%s d=%s div=%s r=%s rem=%d*/\n" % (
            sgpr(qReg), sgpr(dReg), sgpr(divReg), sgpr(rReg), 1 if doRemainder else 0)))
        return module

    for module in modules:
        for name in _DIVIDE_MACROS:
            if hasattr(module, name):
                monkeypatch.setattr(module, name, divide)


def _signed(value):
    return value - (1 << 32) if value & 0x80000000 else value


_COMPARES = {
    "SCmpEQU32": lambda a, b: a == b,
    "SCmpLgU32": lambda a, b: a != b,
    "SCmpLtU32": lambda a, b: a < b,
    "SCmpLeU32": lambda a, b: a <= b,
    "SCmpGtU32": lambda a, b: a > b,
    "SCmpGeU32": lambda a, b: a >= b,
    "SCmpEQI32": lambda a, b: _signed(a) == _signed(b),
    "SCmpLgI32": lambda a, b: _signed(a) != _signed(b),
    "SCmpLtI32": lambda a, b: _signed(a) < _signed(b),
    "SCmpLeI32": lambda a, b: _signed(a) <= _signed(b),
    "SCmpGtI32": lambda a, b: _signed(a) > _signed(b),
    "SCmpGeI32": lambda a, b: _signed(a) >= _signed(b),
}
_IGNORED = {"SNop", "SWaitAlu", "SWaitCnt"}


def _run_scalar(module, registers, max_steps=1_000_000):
    """Execute ``module`` on a register file keyed by rendered operand text.

    Reading a register that was never written raises, so a test notices code
    that depends on a value it did not set up.
    """
    items = list(module.flatitems())
    targets = {item.getLabelName(): i for i, item in enumerate(items) if isinstance(item, Label)}
    regs = dict(registers)
    scc = 0

    def value(operand):
        text = str(operand)
        return regs[text] if text in regs else int(text, 0) & _MASK

    pc = 0
    for _ in range(max_steps):
        if pc >= len(items):
            return regs
        item = items[pc]
        pc += 1
        if isinstance(item, TextBlock):
            match = _DIVIDE.search(str(item))
            if match:
                q, d, divisor, r, remainder = match.groups()
                dividend, div = value(d), value(divisor)
                regs[q] = dividend // div
                if remainder == "1":
                    regs[r] = dividend % div
            continue
        if isinstance(item, Label) or not hasattr(item, "getParams"):
            continue
        name = type(item).__name__
        if name in _IGNORED:
            continue
        params = [str(p) for p in item.getParams()]
        if name == "SBranch":
            pc = targets[params[0]]
        elif name in ("SCBranchSCC0", "SCBranchSCC1"):
            if scc == (name == "SCBranchSCC1"):
                pc = targets[params[0]]
        elif name in _COMPARES:
            scc = int(_COMPARES[name](value(params[0]), value(params[1])))
        elif name == "SCSelectB32":
            regs[params[0]] = value(params[1]) if scc else value(params[2])
        elif name == "SCMovB32":
            if scc:
                regs[params[0]] = value(params[1])
        elif name == "SMovB32":
            regs[params[0]] = value(params[1])
        elif name == "SAbsI32":
            regs[params[0]] = abs(_signed(value(params[1]))) & _MASK
            scc = int(regs[params[0]] != 0)
        elif name == "SSExtI16toI32":
            low = value(params[1]) & 0xFFFF
            regs[params[0]] = (low - 0x10000 if low & 0x8000 else low) & _MASK
        else:
            a, b = value(params[1]), value(params[2])
            if name == "SAddU32":
                result = a + b
                scc = int(result > _MASK)
            elif name == "SSubU32":
                result = a - b
                scc = int(a < b)
            elif name == "SMulI32":
                result = a * b
            elif name == "SMulHIU32":
                result = (a * b) >> 32
            elif name in ("SAndB32", "SOrB32", "SXorB32"):
                result = {"SAndB32": a & b, "SOrB32": a | b, "SXorB32": a ^ b}[name]
                scc = int(result != 0)
            elif name in ("SLShiftLeftB32", "SLShiftRightB32"):
                result = (a << (b & 31) if name == "SLShiftLeftB32" else a >> (b & 31)) & _MASK
                scc = int(result != 0)
            elif name in ("SMinU32", "SMaxU32"):
                result = min(a, b) if name == "SMinU32" else max(a, b)
                scc = int(result == a)
            else:
                raise AssertionError("unsupported instruction in scalar test: %s" % str(item).strip())
            regs[params[0]] = result & _MASK
    raise AssertionError("scalar code did not finish within %d steps" % max_steps)


class _Pool:
    """Hands out fresh indices, so no two live temporaries can alias."""

    def __init__(self, start):
        self.next = start

    def checkOut(self, size, tag="", *args, **kwargs):
        idx = self.next
        self.next += size
        return idx

    def checkOutAligned(self, size, alignment, tag="", *args, **kwargs):
        self.next += (-self.next) % alignment
        return self.checkOut(size, tag)

    def checkIn(self, idx):
        pass


class _Labels:
    def __init__(self):
        self.count = 0

    def getNameInc(self, name):
        self.count += 1
        return "%s_%d" % (name, self.count)


class _PeekWriter:
    def __init__(self):
        self.labels = _Labels()
        self.sgprPool = _Pool(100)
        self.vgprPool = _Pool(0)
        self.states = SimpleNamespace(WGMTransformLevels=-1)

    @contextmanager
    def allocTmpSgpr(self, size, alignment=1, tag=""):
        yield SimpleNamespace(idx=self.sgprPool.checkOutAligned(size, alignment or 1), size=size)

    @contextmanager
    def allocTmpSgprList(self, nums, alignmentList=None, tag=""):
        yield [SimpleNamespace(idx=self.sgprPool.checkOut(n), size=n) for n in nums]


def _peekKernel(wgmxcc):
    return {
        "WorkGroupMappingXCC": wgmxcc,
        "ClusterDim": [1, 1],
        "WavefrontSize": 32,
    }


def _wgm_field(wgm, wgmxcc):
    """Pack WGM the way the host does, with a neighbouring field set to prove it is masked."""
    if wgmxcc == -1:
        return (0x3 << 10) | (wgm & 0x3FF)
    return (0x5 << 16) | (wgm & 0xFFFF)


def _wgm_map(wg0, wg1, n0, n1, wgm):
    """DefaultWGM in Python: the coordinates a tile ends up with."""
    if wgm > 1:
        first, second, nFirst, nSecond = wg0, wg1, n0, n1
    elif wgm >= 0:
        return wg0, wg1
    else:
        first, second, nFirst, nSecond = wg1, wg0, n1, n0
    width = abs(wgm)
    block = second // width
    serial = (second % width) * nFirst + first
    fullBlocks = nSecond // width
    divisor = (nSecond - fullBlocks * width or width) if block >= fullBlocks else width
    newFirst, newSecond = serial // divisor, serial % divisor + block * width
    return (newFirst, newSecond) if wgm > 1 else (newSecond, newFirst)


@pytest.fixture
def _cpu(monkeypatch):
    _exact_scalar_divides(monkeypatch, work_assignment, wgm_algos)
    monkeypatch.setattr(Component.TileProcessingStrategy, "find", lambda writer: DataParallel())


# ---------------------------------------------------------------------------
# Guard peek. A tile reads the A of its WorkGroup0 after DefaultWGM, which folds
# the N-tile in, so the entry guard -- which runs before the tile is claimed --
# has to arrive at the same M-tile from the tile cursor. Both emitters run on
# the CPU and are compared tile by tile.
# ---------------------------------------------------------------------------
_TILE_IDENTITY_CASES = list(itertools.product(
    # One M-tile, and an M that a block of 3 or 4 divides or leaves short.
    [1, 3, 5],                                          # NumWorkGroups0
    # A block of 4 that fits N exactly, and one that leaves a short last block.
    [4, 7],                                             # NumWorkGroups1
    # Both signs, the identity range, and blocks wider than either grid dim.
    [-9, -4, -3, -1, 0, 1, 3, 4, 24],                   # WGM
    [-1, 1],                                            # WorkGroupMappingXCC: the two field layouts
))


@pytest.mark.parametrize("n0, n1, wgm, wgmxcc", _TILE_IDENTITY_CASES)
def test_guard_peek_names_the_m_tile_default_wgm_gives_the_tile(_cpu, n0, n1, wgm, wgmxcc):
    batches = 2
    kernel = _peekKernel(wgmxcc)
    peek = StaticGrid().peekTileBatch(_PeekWriter(), kernel, "RAPResidentBatch", "RAPResidentMTile")
    remap = DefaultWGM(_PeekWriter(), kernel, "WGM")
    common = {
        str(sgpr("NumWorkGroups0")): n0,
        str(sgpr("NumWorkGroups1")): n1,
        str(sgpr("WGM")): _wgm_field(wgm, wgmxcc),
    }
    for tile in range(n0 * n1 * batches):
        peeked = _run_scalar(peek, {**common, str(sgpr("NextTile")): tile})
        batch, inBatch = divmod(tile, n0 * n1)
        remapped = _run_scalar(remap, {**common,
                                       str(sgpr("WorkGroup0")): inBatch % n0,
                                       str(sgpr("WorkGroup1")): inBatch // n0})
        mTile = remapped[str(sgpr("WorkGroup0"))]
        assert mTile == _wgm_map(inBatch % n0, inBatch // n0, n0, n1, wgm)[0]
        assert peeked[str(sgpr("RAPResidentBatch"))] == batch
        assert peeked[str(sgpr("RAPResidentMTile"))] == mTile, (
            "tile %d: guard expects M-tile %d, the tile reads %d"
            % (tile, peeked[str(sgpr("RAPResidentMTile"))], mTile))
        for claimed in ("WorkGroup0", "WorkGroup1", "WorkGroup2"):
            assert str(sgpr(claimed)) not in peeked, "the peek claimed %s" % claimed


# ---------------------------------------------------------------------------
# Phase walk. A workgroup owns tiles rank + k*grid, and the plain stride changes
# M-tile at almost every tile once grid is not a multiple of the period of the
# (remapped) M-tile along the cursor. StaticGrid.rapPhaseWalkInit picks a stride
# that keeps the M-tile and an order of phases. The emitted init and per-tile
# advance run on the CPU against a model of the same plan, and against what the
# plan is for: the same tiles per workgroup, every tile exactly once, and one
# M-tile per phase.
# ---------------------------------------------------------------------------
_PARTITION = DataParallel().staticPartition()
_CURSOR, _BOUND, _GRID = str(sgpr(_PARTITION.cursor)), str(sgpr(_PARTITION.bound)), str(sgpr(_PARTITION.grid))


def _extended_gcd(a, b):
    oldR, r, oldS, s = a, b, 1, 0
    while r:
        q = oldR // r
        oldR, r = r, oldR - q * r
        oldS, s = s, oldS - q * s
    return oldR, oldS


def _decoded(wgm, wgmxcc):
    """The WGM the kernel reads back: the field is 10 bits wide when WorkGroupMappingXCC is -1."""
    bits = 10 if wgmxcc == -1 else 16
    value = wgm & ((1 << bits) - 1)
    return value - (1 << bits) if value >> (bits - 1) else value


def _phase_plan(grid, tiles, n0, n1, wgm):
    """(stride, step) the kernel should derive; the plain walk is (grid, 0)."""
    period, negative = n0, False
    if wgm > 1:
        period = n0 * min(wgm, n1)
    elif wgm < -1:
        negative = True
        height = min(-wgm, n0)
        if n0 % height:
            return grid, 0
        period = n0 * height
    if n0 == 1 or period > 0xFFFFFF:
        return grid, 0
    gcd, coeff = _extended_gcd(grid, period)
    phases = period // gcd
    if tiles > n0 * n1:
        sparse = 2 * phases > -(-(n0 * n1) // grid)
    elif negative:
        sparse = 2 * phases > -(-tiles // grid)
    else:
        sparse = phases > 2 * -(-tiles // grid)
    if sparse:
        return grid, 0
    return phases * grid, (coeff % phases) * grid


def _walk(rank, tiles, stride, step):
    order, cursor, offset = [], rank, 0
    while cursor < tiles:
        order.append(cursor)
        cursor += stride
        while cursor >= tiles:
            offset = (offset + step) % stride
            if offset == 0:
                return order
            cursor = rank + offset
    return order


def _emit(kernel):
    writer = _PeekWriter()
    init = StaticGrid().rapPhaseWalkInit(writer, kernel, _PARTITION)
    advance = StaticGrid().rapPhaseWalkAdvance(writer, kernel, _PARTITION)
    return init, advance


def _run_walk(init, advance, rank, grid, tiles, n0, n1, field):
    regs = _run_scalar(init, {
        _CURSOR: rank, _GRID: grid, _BOUND: tiles,
        str(sgpr("NumWorkGroups0")): n0, str(sgpr("NumWorkGroups1")): n1,
        str(sgpr("WGM")): field,
    })
    packed = regs[str(sgpr("RAPPhasePlan"))]
    plan = ((packed >> 16) * grid, (packed & 0xFFFF) * grid)
    order = []
    while regs[_CURSOR] < tiles:
        order.append(regs[_CURSOR])
        assert len(order) <= tiles, "the walk does not terminate"
        regs = _run_scalar(advance, regs)
    assert regs[_CURSOR] == tiles, "an exhausted walk leaves the cursor at the bound"
    return plan, order


# (NumWorkGroups0, NumWorkGroups1, batches, grid, WGM)
_PHASE_WALK_CASES = [
    (3, 16, 1, 256, 1), (3, 16, 1, 32, 1), (3, 16, 1, 32, 8), (3, 16, 1, 31, 8),
    (3, 16, 2, 32, 8), (5, 10, 1, 32, 4), (6, 8, 1, 17, 1), (6, 8, 1, 17, 16),
    (6, 8, 2, 20, 3), (2, 16, 1, 32, 8), (4, 9, 1, 23, 3), (1, 40, 1, 7, 8),
    (3, 37, 1, 16, 8), (3, 37, 1, 16, 1000), (5, 12, 1, 9, -5), (6, 7, 1, 13, -3),
    (6, 7, 1, 13, -4), (6, 7, 1, 13, -1), (6, 7, 1, 13, 0), (3, 8, 1, 64, 8),
    (3, 8, 1, 1, 8), (7, 5, 2, 11, -2), (4, 25, 1, 30, 24), (3, 300, 1, 255, 300),
]


@pytest.mark.parametrize("wgmxcc", [-1, 1])
@pytest.mark.parametrize("n0, n1, batches, grid, wgm", _PHASE_WALK_CASES)
def test_phase_walk_matches_the_plan_and_keeps_each_workgroups_tiles(
    _cpu, n0, n1, batches, grid, wgm, wgmxcc
):
    tiles = n0 * n1 * batches
    init, advance = _emit(_peekKernel(wgmxcc))
    stride, step = _phase_plan(grid, tiles, n0, n1, _decoded(wgm, wgmxcc))
    assert step < stride
    seen = []
    for rank in range(min(grid, tiles)):
        plan, order = _run_walk(init, advance, rank, grid, tiles, n0, n1, _wgm_field(wgm, wgmxcc))
        assert plan == (stride, step)
        assert order == _walk(rank, tiles, stride, step)
        assert sorted(order) == list(range(rank, tiles, grid)), "the walk changed which tiles a workgroup owns"
        seen.extend(order)
    assert sorted(seen) == list(range(tiles)), "some tile is not processed exactly once"


def _m_tile(tile, n0, n1, wgm):
    batch, inBatch = divmod(tile, n0 * n1)
    return batch, _wgm_map(inBatch % n0, inBatch // n0, n0, n1, wgm)[0]


# Full WGM blocks and one batch: every phase reads a single M-tile, and with
# WGM > 1 the phases come in M-tile order, wrapping around once.
@pytest.mark.parametrize("n0, n1, grid, wgm", [
    (3, 16, 32, 1), (3, 16, 32, 8), (3, 128, 31, 8), (5, 64, 32, 4), (6, 64, 20, 16),
    (6, 8, 20, 4), (3, 32, 24, 16), (4, 66, 23, 3), (6, 80, 13, -3), (5, 90, 9, -5),
])
def test_each_phase_reads_one_m_tile(_cpu, n0, n1, grid, wgm):
    tiles = n0 * n1
    init, advance = _emit(_peekKernel(-1))
    stride, step = _phase_plan(grid, tiles, n0, n1, wgm)
    assert stride > grid, "this case is meant to take the phased walk"
    for rank in range(min(grid, tiles)):
        _, order = _run_walk(init, advance, rank, grid, tiles, n0, n1, _wgm_field(wgm, -1))
        phases = {}
        for tile in order:
            phases.setdefault((tile - rank) % stride, []).append(_m_tile(tile, n0, n1, wgm))
        for offset, mTiles in phases.items():
            assert len(set(mTiles)) == 1, "rank %d, phase at +%d reads %s" % (rank, offset, mTiles)
        if wgm > 1:
            visited = [mTiles[0][1] for mTiles in phases.values()]
            drops = sum(later < earlier for earlier, later in zip(visited, visited[1:]))
            assert drops <= 1, "rank %d visits M-tiles out of order: %s" % (rank, visited)


def test_the_walk_keeps_to_its_sgpr_budget(_cpu):
    """Two sgprs for the whole kernel, and four temporaries at kernel start.

    The init runs with every persistent register and kernel argument live, where
    a RAP kernel's sgpr high-water mark tends to be, so whatever it borrows there
    adds to the kernel's sgpr count one for one.
    """
    writer = _PeekWriter()
    first = writer.sgprPool.next
    init = StaticGrid().rapPhaseWalkInit(writer, _peekKernel(-1), _PARTITION)
    assert writer.sgprPool.next - first <= 4
    advance = StaticGrid().rapPhaseWalkAdvance(writer, _peekKernel(-1), _PARTITION)
    assert set(re.findall(r"sgpr(RAPPhase\w+)", str(init) + str(advance))) == {"RAPPhasePlan", "RAPPhaseStart"}


def test_non_rap_activation_keeps_the_plain_stride():
    """Only RAP kernels carry the phase walk; every other StaticGrid kernel is unchanged."""
    # Stubbed on the instance, not in a subclass: Component registers every
    # subclass as an implementation, and a second DataParallel would make
    # TileProcessingStrategy.find fail for every later test in the process.
    processing = DataParallel()
    processing.materializeTile = lambda *args, **kwargs: Module("materialize stub")
    writer = _PeekWriter()
    rendered = str(StaticGrid().activateReservedOrAcquire(
        writer, {"ReuseAcrossPersistent": 0, "SpaceFillingAlgo": []}, processing, None, None))
    lines = [line.split("//")[0].strip() for line in rendered.splitlines()
             if line.strip() and not line.strip().startswith("/*")]
    assert lines == [
        "s_add_u32 s[sgprNextTile], s[sgprNextTile], s[sgprPersistentGrid]",
        "s_cselect_b32 s[sgprNextTile], s[sgprTotalTiles], s[sgprNextTile]",
    ]
