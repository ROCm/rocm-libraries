# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import copy
from unittest.mock import MagicMock

import pytest

from Tensile.TensileCreateLibrary.Run import _includeGemmA2AFusionProblemType


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "problem_type,enabled,expected",
    [
        ({"FusedGemmA2A": False}, False, True),
        ({"FusedGemmA2A": True}, False, False),
        ({"FusedGemmA2A": True}, True, True),
    ],
)
def test_gemm_a2a_logic_requires_explicit_enable(problem_type, enabled, expected):
    assert _includeGemmA2AFusionProblemType(problem_type, enabled) is expected


@pytest.fixture
def restore_type_mismatch_collector():
    """``generateLogicDataAndSolutions`` replaces Solution.py's module-level type
    mismatch collector with its own aggregate. Put the caller's back."""
    from Tensile.SolutionStructs.Solution import (
        getTypeMismatchCollector,
        mergeTypeMismatchCollector,
        resetTypeMismatchCollector,
    )

    saved = copy.deepcopy(getTypeMismatchCollector())
    yield
    resetTypeMismatchCollector()
    mergeTypeMismatchCollector(saved)


def test_disabled_build_drops_fused_logic_and_reports_it(
    monkeypatch, restore_type_mismatch_collector
):
    """Runs the real merge loop over one fused logic file with the gate off.

    Reporting the count matters as much as the drop: an empty library is
    otherwise the only symptom, which is how a filter that once over-matched
    still passed as a successful build.
    """
    import Tensile.LibraryIO as LibraryIO
    import Tensile.TensileCreateLibrary.Run as RunModule

    parsed = [
        LibraryIO.LibraryLogic(
            schedule="Aldebaran_Cijk_Ailk_Bljk_SB",
            architecture="gfx942",
            problemType={"FusedGemmA2A": True},
            solutions=[],
            exactLogic=None,
            library=MagicMock(solutions={}, lazyLibraries={}),
            typeMismatches={},
        )
    ]
    monkeypatch.setattr(RunModule, "ParallelMap2", lambda *a, **kw: parsed)
    reported = []
    monkeypatch.setattr(RunModule, "print1", reported.append)

    _, masterLibraries, _ = RunModule.generateLogicDataAndSolutions(
        ["fused.yaml"],
        {
            "Architecture": "gfx942",
            "CodeObjectVersion": "4",
            "LazyLibraryLoading": True,
            "GenSolTable": False,
            "EnableGemmA2AFusion": False,
        },
        MagicMock(),
        {},
    )

    assert masterLibraries == {}
    assert any("filtered 1 logic files" in message for message in reported)
