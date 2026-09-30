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
"""

import collections
import copy
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from Tensile.Components.TDMFuse import tdmKDimField, tdmKDimTrackable
from Tensile.KernelWriterAssembly import KernelWriterAssembly

from Tensile.Common.GlobalParameters import defaultSolution
from Tensile.Common.RequiredParameters import getRequiredParametersMin
from Tensile.Common.ValidParameters import validParameters
from Tensile.SolutionStructs.Naming import getParameterNameAbbreviation
from Tensile.SolutionStructs.Solution import Solution, validateParameterTypes

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
# way they read kernel["HalfPLR"], so RAP has no codegen predicate. PAP keeps
# one, and RAP has to appear in it.
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
    from Tensile.KernelWriter import KernelWriter

    assert not hasattr(KernelWriter, "isReuseAcrossPersistentEnabled")


def _codegenKernel(**overrides):
    kernel = {
        "ReuseAcrossPersistent": 1,
        "PrefetchAcrossPersistent": 1,
        "StreamK": 3,
        "PrefetchGlobalRead": 2,
        "UseCustomMainLoopSchedule": 0,
        "SuppressNoLoadLoop": True,
        "HalfPLR": 0,
    }
    kernel.update(overrides)
    return kernel


def _papEnabled(**overrides):
    from Tensile.KernelWriter import KernelWriter

    return KernelWriter.isPrefetchAcrossPersistentEnabled(
        SimpleNamespace(), _codegenKernel(**overrides)
    )


def test_pap_survives_the_suppressed_nll_for_halfplr_and_rap():
    """Suppressing the NLL removes where PAP's next-tile prefetch used to live.

    Only the features that re-emit it elsewhere may keep PAP. RAP is one of them
    -- derivation turns SuppressNoLoadLoop on for it -- so without RAP in this
    predicate every RAP kernel would silently lose PAP at codegen.
    """
    assert _papEnabled()
    assert _papEnabled(ReuseAcrossPersistent=0, HalfPLR=1)
    assert not _papEnabled(ReuseAcrossPersistent=0, HalfPLR=0)
    assert _papEnabled(ReuseAcrossPersistent=0, HalfPLR=0, SuppressNoLoadLoop=False)


def test_pap_is_off_when_its_own_flag_is_off_whatever_rap_says():
    # RAP 1 with PAP 0 is supported, so RAP must not switch PAP back on.
    assert not _papEnabled(PrefetchAcrossPersistent=0)
    assert not _papEnabled(PrefetchAcrossPersistent=0, ReuseAcrossPersistent=0)


# ---------------------------------------------------------------------------
# Toolchain fixtures (real gfx1250 caps + assembler)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def gfx1250_iim():
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Toolchain.Validators import validateToolchain

    cxx = validateToolchain("amdclang++")
    isa = gfxToIsa("gfx1250")
    iim = makeIsaInfoMap([isa], cxx)
    if not iim[isa].asmCaps["SupportedISA"]:
        pytest.skip("amdclang++ in this environment does not support gfx1250")
    return iim


@pytest.fixture(scope="module")
def assembler():
    from Tensile.Toolchain.Assembly import makeAssemblyToolchain
    from Tensile.Toolchain.Validators import validateToolchain, ToolchainDefaults

    cxx = validateToolchain("amdclang++")
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    return makeAssemblyToolchain(cxx, bundler, "default").assembler


@pytest.fixture(scope="module")
def _gp_gfx1250(gfx1250_iim):
    """Assign process-global parameters for gfx1250; restore after module."""
    from Tensile.Common.GlobalParameters import globalParameters, assignGlobalParameters

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
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.SolutionStructs.Validators.MatrixInstruction import (
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
        "StreamK": 3,
        "StreamKForceDPOnly": 1,
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
def test_rap_base_solution_is_accepted(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    sol, out = _derive(gfx1250_iim, assembler, capsys)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"


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


def test_rap_flag_is_cleared_when_there_is_no_persistent_loop(
    _gp_gfx1250, gfx1250_iim, assembler, capsys
):
    """Without Stream-K there are no persistent iterations to reuse across.

    Derivation clears the flag here rather than rejecting, which is the safe
    direction now that codegen trusts it: the alternative is a kernel that
    reaches the emitters claiming residency it cannot have.
    """
    # GlobalSplitU picks up the split Stream-K was providing; without one of the
    # two the solution is rejected before RAP is looked at.
    sol, out = _derive(gfx1250_iim, assembler, capsys, StreamK=0, GlobalSplitU=1)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert sol["ReuseAcrossPersistent"] == 0


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
    import Tensile.Contractions as C

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
    import Tensile.Contractions as C

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


def test_rap_runs_a_k_remainder_through_the_resident_loop(
    _gp_gfx1250, gfx1250_iim, assembler, capsys
):
    """ASEM below DepthU leaves a partial last k-tile, and RAP emits no tail for it.

    NoTailLoop then means only that no tail is emitted: StreamK keeps the loop
    count at ceil(K / DepthU) instead of peeling a k-tile off for a tail, and the
    partial k-tile takes a resident slot of its own.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys, AssertSummationElementMultiple=32)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert sol["AssertSummationElementMultiple"] % sol["DepthU"] != 0
    assert sol["NoTailLoop"] is True


def test_a_k_remainder_still_derives_a_tail_without_rap(
    _gp_gfx1250, gfx1250_iim, assembler, capsys
):
    sol, out = _derive(gfx1250_iim, assembler, capsys,
                       AssertSummationElementMultiple=32, ReuseAcrossPersistent=0)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert sol["NoTailLoop"] is False


_MX_PROBLEM_TYPE = {
    "MacDataTypeA": "F8", "MacDataTypeB": "F8",
    "MXBlockA": 32, "MXBlockB": 32, "DataTypeMXSA": "E8", "DataTypeMXSB": "E8",
}
_K_FIELD_REJECT = ("ReuseAcrossPersistent with AssertSummationElementMultiple % DepthU != 0 "
                   "requires TDM descriptor sets that keep K in one field")


def test_rap_runs_a_k_remainder_on_mx_f8_by_f4(
    _gp_gfx1250, gfx1250_iim, assembler, capsys
):
    """A and B step K by different amounts, but both count it in bytes.

    Each wave's step is then its own address increment, which the A/B set
    already selects by parity, so the remainder needs nothing the set lacks.
    """
    sol, out = _derive(gfx1250_iim, assembler, capsys, AssertSummationElementMultiple=32,
                       ProblemType=dict(_MX_PROBLEM_TYPE, MacDataTypeB="F4"))
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
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
            {"StreamKForceDPOnly": 0},
            "ReuseAcrossPersistent requires StreamK = 3 and StreamKForceDPOnly = 1",
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
# WorkGroupMapping. The entry guard recomputes this tile's M index from
# StreamKIter as tileIdx % NumWorkGroups0, while A's address is built from
# WorkGroup0 *after* DefaultWGM has remapped it. That remap folds WorkGroup1 --
# which moves every persistent iteration -- into WorkGroup0, so with it live the
# two disagree and the guard can pass a tile whose A has changed. This is the
# invariant the M predicate rests on: with more than one M-tile the guard's
# recomputation has to be the WorkGroup0 A's address was built from, and it is
# only because no remap is emitted.
#
# Rejecting a non-1 WorkGroupMapping would not be enough, for the same reason it
# was not enough for StaggerU: 0 means "the host predicts it at runtime" and
# TENSILE_FIXED_WGM overrides whatever the solution asked for. RAP therefore
# clears the capability flag, which stops the host packing the field at all.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"WorkGroupMapping": 8}, id="asked_for_the_library_default"),
        pytest.param(
            # 0 is "predicted at runtime", so the value the kernel sees is never
            # in the solution at all. Legal on StreamK, which RAP requires.
            {"WorkGroupMapping": 0},
            id="asked_for_the_runtime_prediction",
        ),
        pytest.param(
            # Already 1 at compile time: the case a reject could not have caught,
            # because there is nothing to reject and the runtime field still wins.
            {"WorkGroupMapping": 1},
            id="already_one_at_compile_time",
        ),
    ],
)
def test_rap_turns_the_runtime_wgm_path_off(
    _gp_gfx1250, gfx1250_iim, assembler, capsys, overrides
):
    sol, out = _derive(gfx1250_iim, assembler, capsys, **overrides)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert sol["ReuseAcrossPersistent"] == 1
    assert sol["WorkGroupMapping"] == 1
    assert sol["InternalSupportParams"]["SupportCustomWGM"] is False


def test_non_rap_keeps_the_runtime_wgm_path(
    _gp_gfx1250, gfx1250_iim, assembler, capsys
):
    # The opt-out has to be RAP's, not something every StreamK kernel now pays.
    sol, out = _derive(
        gfx1250_iim, assembler, capsys, ReuseAcrossPersistent=0, WorkGroupMapping=8
    )
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert sol["WorkGroupMapping"] == 8
    assert sol["InternalSupportParams"]["SupportCustomWGM"] is True


def test_rap_is_rejected_with_space_filling_algo(
    _gp_gfx1250, gfx1250_iim, assembler, capsys
):
    # SpaceFillingCurveWalk is DefaultWGM's sibling, chosen at codegen, so the
    # SupportCustomWGM opt-out inside DefaultWGM never runs for it. Rejecting the
    # compile-time list is enough here where rejecting StaggerU was not: which
    # algorithm gets emitted is baked into the assembly, only the WGM *value*
    # arrives from a runtime field.
    sol, out = _derive(gfx1250_iim, assembler, capsys, SpaceFillingAlgo=[0])
    assert sol.get("Valid") is not True, "expected reject"
    assert "SpaceFillingAlgo" in out


def _wgm_writer():
    """Minimal writer for DefaultWGM: pools, labels and the tmp-sgpr context."""
    from Tensile.Common.RegisterPool import RegisterPool, RegisterType
    from contextlib import contextmanager

    next_tmp = [200]

    @contextmanager
    def _alloc_tmp_sgpr_list(nums, alignmentList=None, tag=""):
        out = []
        for size in nums:
            out.append(SimpleNamespace(idx=next_tmp[0], size=size))
            next_tmp[0] += size
        yield out

    return SimpleNamespace(
        sgprPool=RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=False, printRP=False),
        vgprPool=RegisterPool(0, RegisterType.Vgpr, defaultPreventOverflow=False, printRP=False),
        labels=SimpleNamespace(getNameInc=lambda name: name),
        allocTmpSgprList=_alloc_tmp_sgpr_list,
        states=SimpleNamespace(WGMTransformLevels=-1),
    )


def _wgm_kernel(supportCustomWGM):
    return {
        "InternalSupportParams": {"SupportCustomWGM": supportCustomWGM},
        "WorkGroupMappingXCC": 1,
        "ClusterDim": [1, 1],
        "WavefrontSize": 32,
    }


@pytest.mark.parametrize(
    "supported, want_instructions",
    [
        pytest.param(False, False, id="capability_off_emits_nothing"),
        pytest.param(True, True, id="capability_on_still_emits"),
    ],
)
def test_default_wgm_honours_the_capability_flag(supported, want_instructions):
    """The device half of the opt-out.

    Clearing SupportCustomWGM stops the host packing the field, which would
    already leave the kernel reading a zero and taking the identity branch. The
    kernel is made to carry no remap at all so the invariant can be checked by
    reading this kernel rather than by trusting what the host sent -- the same
    two-sided arrangement SupportCustomStaggerU has.
    """
    from Tensile.Components.WorkGroupMappingAlgos import DefaultWGM

    module = DefaultWGM(_wgm_writer(), _wgm_kernel(supported), "WGM")
    # Comments are not instructions; count what the wave would actually execute.
    text = str(module)
    emitted = [
        line for line in text.splitlines()
        if line.strip() and not line.strip().startswith("/*")
    ]
    if want_instructions:
        assert emitted, "expected DefaultWGM to emit the remap"
    else:
        assert not emitted, f"expected no remap, got:\n{text}"


# ---------------------------------------------------------------------------
# k_max model. Pinned to the two audited gfx1250 configs, whose true bounds were
# measured by sweeping the resident k-tile count until rapCheckStoreNeutrality
# fired: MacroTile 64x256 clears k=8 and trips at 9, 64x512 clears k=3 at 4.
# Driving the model directly rather than through Solution keeps the arithmetic
# pinned to those numbers; the terms below are the ones codegen actually reported.
# ---------------------------------------------------------------------------
_MODEL_ISA = (12, 5, 0)


def _modelTerms(threadTile1):
    from Tensile.Common.DataType import DataType

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
    from Tensile.Common.DataType import DataType

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


def test_k_dim_fields_of_data_and_scale_tensors():
    kernel = _kDimKernel()
    assert tdmKDimField(kernel, "A") == (1, 256)
    assert tdmKDimField(kernel, "MXSB") == (2, 2)
    # FP4 packs two elements per byte and the dim counts bytes.
    assert tdmKDimField(_kDimKernel(b="F4"), "B") == (1, 128)


@pytest.mark.parametrize("fuse, trackable", [(0, True), (1, False), (2, False), (3, False)])
def test_only_the_default_grouping_keeps_one_k_field_per_set(fuse, trackable):
    assert tdmKDimTrackable(_kDimKernel(fuse=fuse)) is trackable


def test_k_dims_track_when_members_step_alike_or_by_bytes():
    assert tdmKDimTrackable(_kDimKernel(mx=False))
    assert tdmKDimTrackable(_kDimKernel(a="F4", b="F4"))
    # 256 bytes and 128 bytes: each is its own address increment.
    assert tdmKDimTrackable(_kDimKernel(b="F4"))
    # 256 elements either way, so one step fits both.
    assert tdmKDimTrackable(_kDimKernel(b="H", mx=False))
    # 128 bytes against 256 half elements: no single register holds both.
    assert not tdmKDimTrackable(_kDimKernel(a="F4", b="H", mx=False))
    tileMajorB = _kDimKernel()
    tileMajorB["ProblemType"]["TLUB"] = True
    assert not tdmKDimTrackable(tileMajorB)


@pytest.mark.parametrize("rap, asem, tracks", [(1, 32, True), (1, 256, False), (0, 32, False)])
def test_rap_tracks_the_k_dim_only_where_a_partial_k_tile_can_occur(rap, asem, tracks):
    kernel = _kDimKernel(ReuseAcrossPersistent=rap, AssertSummationElementMultiple=asem)
    assert KernelWriterAssembly.rapTracksKDim(SimpleNamespace(), kernel) is tracks


class _ShrinkWriter:
    """Just enough writer to render rapTdmShrinkKDim."""

    isTdmWaveSeparated = KernelWriterAssembly.isTdmWaveSeparated
    _tdmPairedParityOrder = KernelWriterAssembly._tdmPairedParityOrder
    rapMxsKSplitOffsetExceedsOneGroup = KernelWriterAssembly.rapMxsKSplitOffsetExceedsOneGroup

    def rapResidentKTiles(self, kernel):
        return 8

    @contextmanager
    def allocTmpSgpr(self, num, alignment=None, tag=None):
        yield SimpleNamespace(idx=90, size=num)


def _shrink(kernel, tcA, tcB):
    return str(KernelWriterAssembly.rapTdmShrinkKDim(
        _ShrinkWriter(), kernel, {"tensorChar": tcA}, {"tensorChar": tcB}))


def test_the_data_set_shrinks_k_in_dim0_by_one_k_tile():
    rendered = _shrink(_kDimKernel(), "A", "B")
    assert "s_sub_u32 s[sgprtdmAGroup1+1], s[sgprtdmAGroup1+1], 0x1000000" in rendered
    assert "s_and_b32" not in rendered


def test_a_data_set_stepping_by_bytes_shrinks_by_the_wave_address_increment():
    rendered = _shrink(_kDimKernel(b="F4"), "A", "B")
    assert "s[sgprtdmABIncs]" in rendered
    assert "s_sub_u32 s[sgprtdmAGroup1+1], s[sgprtdmAGroup1+1], s90" in rendered


def test_the_scale_set_shrinks_k_in_dim1_by_its_k_groups():
    # DepthU 256 is two K groups, one per scale wave: the second wave starts a
    # group in and can never outrun the last k-tile's one-or-two groups.
    rendered = _shrink(_kDimKernel(), "MXSA", "MXSB")
    assert "s_sub_u32 s[sgprtdmMXSAGroup1+2], s[sgprtdmMXSAGroup1+2], 0x20000" in rendered
    assert "s_and_b32" not in rendered


def test_a_scale_wave_offset_past_one_group_clamps_instead_of_wrapping():
    # DepthU 512 gives each of the two scale waves two groups, so the second
    # starts two groups in and runs out when the last k-tile has only one.
    rendered = _shrink(_kDimKernel(depthU=512), "MXSA", "MXSB")
    assert "s_sub_u32 s[sgprtdmMXSAGroup1+2], s[sgprtdmMXSAGroup1+2], 0x40000" in rendered
    assert "s_and_b32 s[sgprtdmMXSAGroup1+2], s[sgprtdmMXSAGroup1+2], s90" in rendered
