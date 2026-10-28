# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the lighter logic loading in TensileCreateLibrary: shared
solution wrappers in matching tables, and the parse wrapper the loading
workers run."""
import pytest

import tensilelite.TensileCreateLibrary.Run as Run
from tensilelite import LibraryIO
from tensilelite.SolutionLibrary import IndexSolutionLibrary, MatchingLibrary

pytestmark = pytest.mark.unit


class _Sol:
    def __init__(self, index):
        self.index = index


def _matching(table):
    return {"indexOrder": [0, 1], "distance": "Equality", "table": table}


def test_rows_for_one_solution_share_its_wrapper():
    sols = [_Sol(0), _Sol(1)]
    lib = MatchingLibrary.FromOriginalState(
        _matching([[[1, 64], [0, 0.9]], [[1, 128], [1, 0.8]], [[1, 256], [0, 0.7]]]), sols
    )
    wrappers = [row["index"] for row in lib.table]
    assert all(isinstance(w, IndexSolutionLibrary) for w in wrappers)
    assert wrappers[0] is wrappers[2]
    assert wrappers[0] is not wrappers[1]
    assert [w.state() for w in wrappers] == [0, 1, 0]


def test_shared_wrapper_follows_solution_reindexing():
    sols = [_Sol(0)]
    lib = MatchingLibrary.FromOriginalState(
        _matching([[[1, 64], [0, 0.9]], [[1, 256], [0, 0.7]]]), sols
    )
    sols[0].index = 5
    assert [row["index"].state() for row in lib.table] == [5, 5]


def test_rows_without_a_solution_are_still_skipped():
    lib = MatchingLibrary.FromOriginalState(_matching([[[1, 64], [3, 0.9]]]), {0: _Sol(0)})
    assert lib.table == []


def test_parse_wrapper_drops_only_what_tcl_discards(monkeypatch):
    calls = []

    def fakeParse(*args):
        calls.append(args)
        return ("schedule", "gfx942", "problemType", ["solution"], ["exactLogic"], "library", {"k": 1})

    monkeypatch.setattr(LibraryIO, "parseLibraryLogicFile", fakeParse)
    logic = Run._parseLogicFileForLibrary("logic.yaml", "assembler", False)
    assert calls == [("logic.yaml", "assembler", False)]
    assert logic == LibraryIO.LibraryLogic(
        "schedule", "gfx942", "problemType", None, None, "library", {"k": 1}
    )
