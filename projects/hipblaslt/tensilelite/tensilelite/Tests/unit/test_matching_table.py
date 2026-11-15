# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""MatchingTable: column storage for MatchingLibrary rows, equivalent to the legacy list of dicts."""

import pickle
import random

import numpy as np
import pytest

from tensilelite.Common.Utilities import state
from tensilelite.MatchingTable import MatchingTable
from tensilelite.SolutionLibrary import IndexSolutionLibrary, MatchingLibrary


class _Sol:
    def __init__(self, index):
        self.index = index


def _legacyRows(n, keyLen=4, speed=True, seed=0, maxKey=5000):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        key = [rng.randrange(1, maxKey) for _ in range(keyLen)]
        if rng.random() < 0.3 and rows:  # duplicate keys exercise sort stability
            key = list(rows[rng.randrange(len(rows))]["key"])
        row = {"key": key, "index": IndexSolutionLibrary(_Sol(i))}
        if speed:
            row["speed"] = rng.random() * 100
        rows.append(row)
    return rows


def _legacySorted(rows):
    out = list(rows)
    out.sort(key=lambda r: r["key"])
    return out


@pytest.mark.parametrize("speed", [True, False])
def test_state_matches_legacy_sorted_rows(speed):
    rows = _legacyRows(500, speed=speed)
    table = MatchingTable.fromRows(rows)
    assert table is not None
    assert table.state() == state(_legacySorted(rows))
    assert len(table) == 500


def test_rows_materialize_in_legacy_form():
    rows = _legacyRows(20)
    table = MatchingTable.fromRows(rows)
    expected = _legacySorted(rows)
    for i in range(20):
        row = table[i]
        assert row["key"] == expected[i]["key"]
        assert isinstance(row["index"], IndexSolutionLibrary)
        assert row["index"].solution is expected[i]["index"].solution
        assert row["speed"] == expected[i]["speed"]
        assert type(row["speed"]) is float
        assert all(type(k) is int for k in row["key"])
    assert [r["key"] for r in table] == [r["key"] for r in expected]
    assert [r["key"] for r in table[0:2]] == [table[0]["key"], table[1]["key"]]
    gridRows = _legacyRows(5, speed=False)
    assert "speed" not in MatchingTable.fromRows(gridRows)[0]


def test_merge_matches_legacy_concat_and_sort():
    a, b = _legacyRows(300, seed=1), _legacyRows(200, seed=2)
    ta, tb = MatchingTable.fromRows(a), MatchingTable.fromRows(b)
    ta += tb
    ta.sort(key=lambda r: r["key"])
    legacy = _legacySorted(a)
    legacy += _legacySorted(b)
    legacy.sort(key=lambda r: r["key"])
    assert ta.state() == state(legacy)
    assert MatchingTable.fromRows([]) is None  # an empty table stays a legacy list
    empty = MatchingTable(np.empty((0, 0), dtype=np.int64), [], None)
    empty += MatchingTable.fromRows(a)
    assert empty.state() == state(_legacySorted(a))
    tb += MatchingTable(np.empty((0, 0), dtype=np.int64), [], None)
    assert tb.state() == state(_legacySorted(b))


def test_merge_mismatched_shapes_is_an_error():
    ta = MatchingTable.fromRows(_legacyRows(3, keyLen=4))
    with pytest.raises(ValueError):
        ta += MatchingTable.fromRows(_legacyRows(3, keyLen=3))
    with pytest.raises(ValueError):
        ta += MatchingTable.fromRows(_legacyRows(3, speed=False))


def test_mixed_int_and_float_speeds_keep_their_types():
    rows = _legacyRows(40)
    for i in range(0, 40, 3):
        rows[i]["speed"] = 0  # unmeasured sizes carry an int 0 in the logic files
    table = MatchingTable.fromRows(rows)
    assert table is not None
    assert table.state() == state(_legacySorted(rows))
    assert {type(r["speed"]) for r in table} == {int, float}
    # Merging an all-float table with an all-int table must not float the ints.
    floats, ints = _legacyRows(10, seed=5), _legacyRows(10, seed=6)
    for r in ints:
        r["speed"] = 2
    tf, ti = MatchingTable.fromRows(floats), MatchingTable.fromRows(ints)
    assert tf.speedIsInt is None and ti.speedIsInt.all()
    tf += ti
    tf.sort()
    legacy = _legacySorted(floats) + _legacySorted(ints)
    legacy.sort(key=lambda r: r["key"])
    assert tf.state() == state(legacy)
    ti2 = MatchingTable.fromRows(ints)
    ti2 += MatchingTable.fromRows(floats)
    ti2.sort()
    assert ti2.state() == state(legacy)
    clone = pickle.loads(pickle.dumps(tf, pickle.HIGHEST_PROTOCOL))
    assert clone.state() == tf.state()


def test_values_that_cannot_round_trip_exactly_are_refused():
    rows = _legacyRows(5)
    rows[2]["key"][0] = True
    assert MatchingTable.fromRows(rows) is None
    rows = _legacyRows(5)
    rows[3]["speed"] = "fast"
    assert MatchingTable.fromRows(rows) is None
    rows = _legacyRows(5)
    rows[3]["speed"] = 2**60 + 1  # not exact in float64
    assert MatchingTable.fromRows(rows) is None
    rows = _legacyRows(5)
    rows[0]["key"] = rows[0]["key"][:-1]
    assert MatchingTable.fromRows(rows) is None
    rows = _legacyRows(5)
    rows[1]["key"][1] = 2**70
    assert MatchingTable.fromRows(rows) is None
    rows = _legacyRows(5)
    for r in rows:
        r["speed"] = 3  # all ints is fine
    table = MatchingTable.fromRows(rows)
    assert table is not None and type(table[0]["speed"]) is int
    assert table.state() == state(_legacySorted(rows))


def test_rows_of_one_solution_share_a_wrapper():
    sol = _Sol(0)
    rows = [
        {"key": [1, 64], "index": IndexSolutionLibrary(sol), "speed": 0.9},
        {"key": [1, 128], "index": IndexSolutionLibrary(_Sol(1)), "speed": 0.8},
        {"key": [1, 256], "index": IndexSolutionLibrary(sol), "speed": 0.7},
    ]
    table = MatchingTable.fromRows(rows)
    wrappers = [row["index"] for row in table]
    assert wrappers[0] is wrappers[2] and wrappers[0] is not wrappers[1]
    assert wrappers[0].solution is sol
    clone = pickle.loads(pickle.dumps(table, pickle.HIGHEST_PROTOCOL))
    w = [row["index"] for row in clone]
    assert w[0] is w[2] and w[0].solution is clone.solutions[0]
    clone.solutions[0].index = 7
    assert [x.state() for x in w] == [7, 1, 7]


def test_pickle_round_trip():
    table = MatchingTable.fromRows(_legacyRows(50))
    clone = pickle.loads(pickle.dumps(table, pickle.HIGHEST_PROTOCOL))
    assert clone.state() == table.state()


def test_matching_library_uses_column_storage_and_serializes_identically():
    sols = [_Sol(i) for i in range(6)]
    origTable = [
        [[64, 128, 1, 256], [0, 1.5]],
        [[32, 32, 1, 64], [1, 2.5]],
        [[64, 128, 1, 256], [2, 0.5]],
    ]
    d = {"indexOrder": [2, 3, 0, 1], "distance": "Equality", "table": origTable}
    lib = MatchingLibrary.FromOriginalState(d, sols)
    assert isinstance(lib.table, MatchingTable)
    legacy = MatchingLibrary.FromOriginalState(d, sols)
    legacy.table = legacy.table.toRows()
    assert state(lib) == state(legacy)
    assert lib.table[0]["key"] <= lib.table[1]["key"]
    other = MatchingLibrary.FromOriginalState(
        {
            "indexOrder": [2, 3, 0, 1],
            "distance": "Equality",
            "table": [[[8, 8, 1, 8], [5, 9.0]]],
        },
        sols,
    )
    lib.merge(other)
    legacyOther = MatchingLibrary.FromOriginalState(
        {
            "indexOrder": [2, 3, 0, 1],
            "distance": "Equality",
            "table": [[[8, 8, 1, 8], [5, 9.0]]],
        },
        sols,
    )
    legacyOther.table = legacyOther.table.toRows()
    legacy.merge(legacyOther)
    assert state(lib) == state(legacy)
    assert len(lib.table) == 4 and lib.table[0]["key"] == [8, 8, 1, 8]
