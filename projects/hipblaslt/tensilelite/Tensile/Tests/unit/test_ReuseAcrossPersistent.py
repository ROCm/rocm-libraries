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
``Solution.depthUIteration``; the problem-size half is three runtime predicates
emitted from ``Contractions.ProblemPredicate.CompoundPredicates``. Both are
covered here.

The reject harness mirrors ``test_halfplr_streamk_rejects``: real gfx1250
capability maps from ``makeIsaInfoMap`` and a real assembler feed
``Solution.__init__``, which runs ``assignDerivedParameters`` end-to-end, and the
reject reason is captured from stdout via ``capsys``.
"""

import collections
import copy
from types import SimpleNamespace

import pytest

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
# Base solution: the MT64x256 RAP candidate from mxf8mxf4_gfx1250_rap.yaml,
# minus the MX scaling (irrelevant to these guards). Each negative test flips
# exactly one knob.
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


def test_rap_off_does_not_derive_a_resident_block(
    _gp_gfx1250, gfx1250_iim, assembler, capsys
):
    sol, out = _derive(gfx1250_iim, assembler, capsys, ReuseAcrossPersistent=0)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert "_RAPNumResidentKTiles" not in sol._state


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
    monkeypatch.setenv("TENSILE_RAP_KTILES", "4")
    sol, out = _derive(gfx1250_iim, assembler, capsys)
    assert sol.get("Valid") is True, f"expected accept, rejected with: {out!r}"
    assert sol["_RAPNumResidentKTiles"] == 4


def test_rap_rejects_fewer_ktiles_than_the_section_count(
    _gp_gfx1250, gfx1250_iim, assembler, capsys, monkeypatch
):
    """Below PrefetchGlobalRead + 1 the sections outnumber the k-tiles.

    max(1, k - PGR) body copies plus the PGR drain sections is never fewer than
    PGR + 1, so a smaller k would leave two sections owning one k-tile while each
    still consumed a k-tile of K -- more K than the SizeEqual(K) predicate claims.
    """
    monkeypatch.setenv("TENSILE_RAP_KTILES", "2")
    sol, out = _derive(gfx1250_iim, assembler, capsys)
    assert sol.get("Valid") is False, "k below the section floor must not build"
    assert "at least PrefetchGlobalRead + 1 resident k-tiles" in out


# ---------------------------------------------------------------------------
# Negative: each precondition of "every tile reads the same A" is enforced.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "overrides, reason",
    [
        pytest.param(
            {"PrefetchAcrossPersistent": 0},
            "ReuseAcrossPersistent requires PrefetchAcrossPersistent",
            id="without_pap",
        ),
        pytest.param(
            {"StreamKForceDPOnly": 0},
            "ReuseAcrossPersistent requires StreamK = 3 and StreamKForceDPOnly = 1",
            id="without_dp_only",
        ),
        pytest.param(
            {"AssertSummationElementMultiple": 32},
            "ReuseAcrossPersistent requires NoTailLoop",
            id="with_tail_loop",
        ),
        pytest.param(
            # PGR>=2 forces ExpandPointerSwap off before the guard runs, so this
            # reject is only reachable at PGR=1 (accepted on its own above).
            {"PrefetchGlobalRead": 1, "ExpandPointerSwap": True},
            "ReuseAcrossPersistent requires ExpandPointerSwap = 0",
            id="with_expand_pointer_swap",
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
# k_max model. Pinned to the two audited gfx1250 configs, whose true bounds were
# measured by sweeping TENSILE_RAP_KTILES until rapCheckStoreNeutrality fired:
# MacroTile 64x256 clears k=8 and trips at 9, 64x512 clears k=3 and trips at 4.
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
def test_store_guard_only_examines_the_path_the_predicates_allow(beta, edge):
    # beta=1/edge=0 is the tightest variant this problem can reach; checking the
    # others would reject on paths the size predicates already exclude.
    assert _runStoreGuard(400, beta=beta, edge=edge).overflowedResources == 0
