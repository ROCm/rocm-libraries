################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Characterization tests for ``Tensile.Contractions``: the index value classes
and the ``ProblemType`` / predicate / ``SizeMapping`` / ``Solution`` builders,
driven from the vendored LibraryIO logic fixture's original-state dicts."""

import importlib
from pathlib import Path

import pytest
import yaml

import Tensile.LibraryIO as L

C = importlib.import_module("Tensile.Contractions")

pytestmark = pytest.mark.unit

_FIXTURE = Path(__file__).parent.parent / "LibraryIO" / "data" / "logic_gfx942_HSS_BH.yaml"


@pytest.fixture(scope="module")
def raw():
    return yaml.load(_FIXTURE.read_text(), Loader=L.StrictTypeLoader)


@pytest.fixture
def problem_type(raw):
    return C.ProblemType.FromOriginalState(raw[4])


@pytest.fixture(scope="module")
def solution_state():
    # The fully-derived solution state (all keys MasterSolutionLibrary needs),
    # obtained by parsing the fixture into a real Solution.
    from Tensile.Common.Architectures import SUPPORTED_ISA
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Toolchain.Assembly import makeAssemblyToolchain
    from Tensile.Toolchain.Validators import validateToolchain, ToolchainDefaults

    cxx = validateToolchain("amdclang++")
    iim = makeIsaInfoMap(SUPPORTED_ISA, cxx)
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    asm = makeAssemblyToolchain(cxx, bundler, "default").assembler
    sol = L.parseLibraryLogicFile(str(_FIXTURE), asm, False, False, False, iim, False).solutions[0]
    return dict(sol._state)


# --- index value classes ----------------------------------------------------

def test_index_classes(snapshot):
    from Tensile.Common.Utilities import state
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
    from Tensile.Common.Utilities import state
    sm = C.SizeMapping.FromOriginalState(solution_state)
    assert isinstance(state(sm), dict)


def test_internal_args_support(solution_state):
    ias = C.InternalArgsSupport.FromOriginalState(solution_state)
    assert ias is not None


def test_problem_predicate_compound(problem_type, solution_state, snapshot):
    preds = C.ProblemPredicate.CompoundPredicates(solution_state, problem_type)
    assert {"count": len(preds), "tags": sorted({p.tag for p in preds})} == snapshot


# --- StreamK cluster-reduction split-barrier selection guard --
#
# ClusterReductionIterCheck must be emitted only for StreamKClusterReduction
# solutions with a real reduction cluster (pure reduction [1, C], so
# Ck = ClusterDim[1] > 1); its value carries [DepthU, Ck] so the host predicate
# can reject problems whose itersPerTile = ceil(K/DepthU) is not a multiple of
# Ck (split-barrier over-signal). It must NOT be emitted for non-cluster or
# multicast ([C,1]) solutions.

def _preds_for(solution_state, problem_type, **overrides):
    st = dict(solution_state)
    st.update(overrides)
    return C.ProblemPredicate.CompoundPredicates(st, problem_type)


def _cluster_iter_pred(preds):
    return next((p for p in preds if p.tag == "ClusterReductionIterCheck"), None)


def test_cluster_reduction_iter_check_emitted(problem_type, solution_state):
    # Pure reduction is ClusterDim = [1, C] (Ck = ClusterDim[1] = C).
    preds = _preds_for(solution_state, problem_type,
                       StreamKClusterReduction=1, ClusterDim=[1, 4], DepthU=256)
    p = _cluster_iter_pred(preds)
    assert p is not None, "cluster-reduction solution must emit ClusterReductionIterCheck"
    # value = [DepthU, Ck] so the host can compute itersPerTile % Ck.
    assert p.value == [256, 4]


def test_cluster_reduction_iter_check_not_emitted_when_off(problem_type, solution_state):
    # StreamKClusterReduction off -> no guard (derived flag off).
    preds = _preds_for(solution_state, problem_type,
                       StreamKClusterReduction=0, ClusterDim=[1, 4], DepthU=256)
    assert _cluster_iter_pred(preds) is None


def test_cluster_reduction_iter_check_not_emitted_without_cluster(problem_type, solution_state):
    # Reduction requested but Ck == 1 (no real reduction cluster) -> no guard.
    preds = _preds_for(solution_state, problem_type,
                       StreamKClusterReduction=1, ClusterDim=[1, 1], DepthU=256)
    assert _cluster_iter_pred(preds) is None
