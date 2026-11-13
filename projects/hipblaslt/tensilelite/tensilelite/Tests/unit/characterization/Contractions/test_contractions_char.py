################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Characterization tests for ``tensilelite.Contractions``: the index value classes
and the ``ProblemType`` / predicate / ``SizeMapping`` / ``Solution`` builders,
driven from the vendored LibraryIO logic fixture's original-state dicts."""

import importlib
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

import tensilelite.LibraryIO as L

C = importlib.import_module("tensilelite.Contractions")

pytestmark = pytest.mark.unit

_FIXTURE = Path(__file__).parent.parent / "LibraryIO" / "data" / "logic_gfx942_HSS_BH.yaml"


@pytest.fixture(scope="module")
def raw():
    # read_text() is on its own line so the marker covers only the yaml.load call.
    doc = _FIXTURE.read_text()
    return yaml.load(doc, Loader=L.StrictTypeLoader)  # nosec B506


@pytest.fixture
def problem_type(raw):
    return C.ProblemType.FromOriginalState(raw[4])


@pytest.fixture(scope="module")
def solution_state(assembler, isa_info_map):
    # The fully-derived solution state (all keys MasterSolutionLibrary needs),
    # obtained by parsing the fixture into a real Solution.
    sol = L.parseLibraryLogicFile(
        str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    ).solutions[0]
    return dict(sol._state)


# --- index value classes ----------------------------------------------------

def test_index_classes(snapshot):
    from tensilelite.Common.Utilities import state
    fi = C.FreeIndex(isA=True, i=0, c=0, d=0)
    bi = C.BatchIndex(a=0, b=1, c=0, d=0)
    bo = C.BoundIndex(a=2, b=2, aMirror=True, bMirror=False)
    assert {"free": state(fi), "batch": state(bi), "bound": state(bo)} == snapshot


# --- ProblemType ------------------------------------------------------------

def test_problem_type_index_names(problem_type, snapshot):
    assert problem_type.indexNames == snapshot          # property


def test_problem_type_operation_identifier(problem_type, snapshot):
    assert problem_type.operationIdentifier == snapshot  # property


def test_problem_type_placeholder_str(problem_type, snapshot):
    assert {
        "plain": problem_type.placeholderStr(),
        "full": problem_type.placeholderStr(includeBatch=True, includeOperation=True, includeType=True),
    } == snapshot


def test_problem_type_predicates(problem_type, snapshot):
    preds = problem_type.predicates(includeBatch=True, includeOperation=True, includeType=True)
    assert {"count": len(preds), "tags": sorted({p.tag for p in preds})} == snapshot


# --- SizeMapping / InternalArgsSupport / Solution ---------------------------

def test_size_mapping(solution_state, snapshot):
    from tensilelite.Common.Utilities import state
    sm = C.SizeMapping.FromOriginalState(solution_state)
    assert isinstance(state(sm), dict)


@pytest.mark.parametrize("mode", [0, 1, 2, 3, 5, 6, 7])
def test_internal_args_support_does_not_duplicate_alpha_capability(solution_state, mode):
    from tensilelite.Common.Utilities import state

    source = deepcopy(solution_state)
    source["ProblemType"]["UseScaleAlphaVec"] = mode
    serialized = state(C.InternalArgsSupport.FromOriginalState(source))
    assert set(serialized) == {
        "version", "persistentLoopArgsVersion", "gsu", "wgm", "staggerU",
        "perTileExtraIters", "useUniversalArgs", "useSFC",
    }
    problem = C.ProblemType.FromOriginalState(source["ProblemType"])
    assert problem.useScaleAlphaVec == mode


def test_problem_predicate_compound(problem_type, solution_state, snapshot):
    preds = C.ProblemPredicate.CompoundPredicates(solution_state, problem_type)
    assert {"count": len(preds), "tags": sorted({p.tag for p in preds})} == snapshot


def _gl2_saddr_predicates(problem_type, solution_state, **overrides):
    source = deepcopy(solution_state)
    source.update(overrides)
    preds = C.ProblemPredicate.CompoundPredicates(source, problem_type)
    return [p for p in preds if p.tag == "GL2PrefetchSAddrOffsetLimitCheck"]


def test_gl2_saddr_offset_check_scales_free_extent_by_cluster(problem_type, solution_state):
    (pred,) = _gl2_saddr_predicates(problem_type, solution_state, PrefetchGL2=1,
                                    PrefetchGL2SAddr=True, ClusterDim=[4, 2])
    pt = solution_state["ProblemType"]
    assert pred.value == {
        "TLUA": bool(pt["TLUA"]),
        "TLUB": bool(pt["TLUB"]),
        "DepthUA": solution_state["_DepthUA"],
        "DepthUB": solution_state["_DepthUB"],
        "ClusterMT0": solution_state["MacroTile0"] * 4,
        "ClusterMT1": solution_state["MacroTile1"] * 2,
        "DepthU": solution_state["DepthU"],
        "MatrixInstK": solution_state["MatrixInstK"],
        "MXBlockA": 0,
        "MXBlockB": 0,
        "MXBlockFreeA": 1,
        "MXBlockFreeB": 1,
        "Metadata": False,
        "TLUMetadata": False,
        "DepthUMetadata": 0,
        "ClusterMTMetadata": 0,
    }


def test_gl2_saddr_offset_check_carries_scales_and_metadata(problem_type, solution_state):
    pt = deepcopy(solution_state["ProblemType"])
    for key, value in dict(MXBlockA=32, MXBlockB=32, MXBlockFreeB=128, Sparse=2,
                           TLUMetadata=True).items():
        pt[key] = value
    (pred,) = _gl2_saddr_predicates(problem_type, solution_state, PrefetchGL2=1,
                                    PrefetchGL2SAddr=True, ClusterDim=[4, 2], ProblemType=pt,
                                    enableTDMMetadata=True, _DepthUMetadata=64,
                                    MacroTileMetadata=solution_state["MacroTile1"])
    assert {k: pred.value[k] for k in ("MXBlockA", "MXBlockB", "MXBlockFreeA", "MXBlockFreeB",
                                       "Metadata", "TLUMetadata", "DepthUMetadata",
                                       "ClusterMTMetadata")} == {
        "MXBlockA": 32,
        "MXBlockB": 32,
        "MXBlockFreeA": 1,
        "MXBlockFreeB": 128,
        "Metadata": True,
        "TLUMetadata": True,
        "DepthUMetadata": 64,
        # Sparse B: the metadata follows B's tile and cluster axis.
        "ClusterMTMetadata": solution_state["MacroTile1"] * 2,
    }


@pytest.mark.parametrize("prefetchGL2, saddr", [(0, True), (1, False)])
def test_gl2_saddr_offset_check_only_for_saddr_prefetch(problem_type, solution_state,
                                                        prefetchGL2, saddr):
    assert not _gl2_saddr_predicates(problem_type, solution_state, PrefetchGL2=prefetchGL2,
                                     PrefetchGL2SAddr=saddr)
