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
