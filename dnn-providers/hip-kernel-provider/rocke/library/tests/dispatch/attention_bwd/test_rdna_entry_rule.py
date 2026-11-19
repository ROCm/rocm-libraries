# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The RDNA tuned-table entry rule (:mod:`dispatch.attention_bwd.rdna_entry_rule`).

CPU only, synthetic records (no device data, no real tuning keys): exercises
every branch of the selection rule, including the ones the real judge
documents never take (every shipped RDNA finalist has always been correct,
A/A-clean and not slower than shipped on every cell of its key -- see
``rdna_entries_manifest.json``), and checks the module against the real
historical judge documents it reimplements where those are available on disk.
"""

from __future__ import annotations

import json
import math
import os

import pytest

from dispatch.attention_bwd.rdna_entry_rule import (
    CellVerdict,
    load_cell_verdicts,
    select_rdna_general_entries,
)

_KEY_A = (
    "gfx1151",
    "general",
    64,
    "bf16",
    "none",
    "mha",
    "square",
    "fixed",
    8,
    "t64",
    "over",
)
_KEY_B = (
    "gfx1151",
    "general",
    128,
    "bf16",
    "causal_br",
    "gqa",
    "square",
    "fixed",
    8,
    "t>64",
    "over",
)


def _cell(
    key=_KEY_A,
    finalist="f1",
    correct=True,
    aa_clean=True,
    verdict="pass",
    ratio=0.8,
    cells=1,
):
    return CellVerdict(
        key=key,
        finalist=finalist,
        correct=correct,
        aa_clean=aa_clean,
        verdict=verdict,
        ratio_vs_shipped=ratio,
        cells_checked=cells,
    )


def test_a_faster_correct_clean_finalist_wins():
    entries = select_rdna_general_entries([_cell(ratio=0.7)], tie_eps=0.0)
    assert set(entries) == {_KEY_A}
    assert entries[_KEY_A].finalist == "f1"
    assert entries[_KEY_A].geomean_ratio_vs_shipped == pytest.approx(0.7)
    assert entries[_KEY_A].cells_checked == 1


def test_an_incorrect_cell_rejects_its_finalist():
    cells = [_cell(correct=False, verdict="fail", ratio=0.5)]
    assert select_rdna_general_entries(cells, tie_eps=0.0) == {}


def test_a_dirty_aa_control_rejects_its_finalist():
    cells = [_cell(aa_clean=False, ratio=0.5)]
    assert select_rdna_general_entries(cells, tie_eps=0.0) == {}


def test_slower_than_shipped_rejects_its_finalist():
    cells = [_cell(verdict="fail", ratio=1.2)]
    assert select_rdna_general_entries(cells, tie_eps=0.0) == {}


def test_a_slower_cell_rejects_the_finalist_even_if_other_cells_of_the_key_pass():
    """Every cell of the key must pass; one bad cell sinks the finalist."""
    cells = [
        _cell(finalist="f1", ratio=0.8, verdict="pass"),
        _cell(finalist="f1", ratio=1.3, verdict="fail"),
    ]
    assert select_rdna_general_entries(cells, tie_eps=0.0) == {}


def test_ambiguous_within_tie_eps_counts_as_not_slower():
    """An ``ambiguous`` verdict slightly *above* 1 (noise, not a real
    regression) still qualifies the finalist when within ``tie_eps``; it
    only ships because its geomean is also below 1."""
    cells = [_cell(verdict="ambiguous", ratio=0.99)]
    entries = select_rdna_general_entries(cells, tie_eps=0.03)
    assert set(entries) == {_KEY_A}
    assert entries[_KEY_A].geomean_ratio_vs_shipped == pytest.approx(0.99)


def test_ambiguous_within_tie_eps_but_geomean_not_below_1_ships_nothing():
    """``tie_eps`` only answers "not slower"; the separate "entry only when
    the winning geomean is below 1" gate still applies -- a finalist tied
    with shipped (ratio > 1, within tie_eps) qualifies but does not ship."""
    cells = [_cell(verdict="ambiguous", ratio=1.02)]
    assert select_rdna_general_entries(cells, tie_eps=0.03) == {}


def test_ambiguous_beyond_tie_eps_rejects_the_finalist():
    cells = [_cell(verdict="ambiguous", ratio=1.05)]
    assert select_rdna_general_entries(cells, tie_eps=0.03) == {}


def test_lowest_geomean_finalist_wins_among_qualifiers():
    cells = [
        _cell(finalist="f1", ratio=0.9),
        _cell(finalist="f2", ratio=0.6),
        _cell(
            finalist="f3", correct=False, ratio=0.1
        ),  # disqualified despite being fastest
    ]
    entries = select_rdna_general_entries(cells, tie_eps=0.0)
    assert entries[_KEY_A].finalist == "f2"


def test_geomean_is_computed_across_the_finalists_own_cells():
    cells = [
        _cell(finalist="f1", ratio=0.5, cells=1),
        _cell(finalist="f1", ratio=0.8, cells=1),
    ]
    entries = select_rdna_general_entries(cells, tie_eps=0.0)
    assert entries[_KEY_A].geomean_ratio_vs_shipped == pytest.approx(math.sqrt(0.4))
    assert entries[_KEY_A].cells_checked == 2


def test_a_geomean_of_exactly_1_ships_nothing():
    """A finalist that nets out equal to shipped (a 2x / 0.5x pair) changes
    nothing, even though every individual cell "passed"."""
    cells = [
        _cell(finalist="f1", ratio=0.5, cells=1),
        _cell(finalist="f1", ratio=2.0, cells=1),
    ]
    assert select_rdna_general_entries(cells, tie_eps=0.0) == {}


def test_a_finalist_no_faster_than_shipped_overall_ships_nothing():
    cells = [_cell(ratio=1.0)]
    assert select_rdna_general_entries(cells, tie_eps=0.0) == {}


def test_a_key_with_no_qualifying_finalist_is_absent():
    cells = [
        _cell(correct=False, ratio=0.5),
        _cell(finalist="f2", aa_clean=False, ratio=0.4),
    ]
    assert select_rdna_general_entries(cells, tie_eps=0.0) == {}


def test_keys_are_independent():
    cells = [_cell(key=_KEY_A, ratio=0.9), _cell(key=_KEY_B, finalist="g", ratio=0.4)]
    entries = select_rdna_general_entries(cells, tie_eps=0.0)
    assert set(entries) == {_KEY_A, _KEY_B}
    assert entries[_KEY_B].finalist == "g"


def test_load_cell_verdicts_folds_a_nonempty_why_into_incorrect():
    doc = {
        "slices": {
            "d64": {
                "keys": {
                    str(list(_KEY_A)): {
                        "cells": 3,
                        "finalists": {
                            "f1": {"ok": True, "geomean_vs_shipped": 0.8, "why": []},
                            "f2": {
                                "ok": False,
                                "geomean_vs_shipped": 0.5,
                                "why": ["cell 2: ambiguous"],
                            },
                        },
                    }
                }
            }
        }
    }
    cells = load_cell_verdicts(doc)
    by_finalist = {c.finalist: c for c in cells}
    assert by_finalist["f1"].correct and by_finalist["f1"].aa_clean
    assert by_finalist["f1"].cells_checked == 3
    assert not by_finalist["f2"].correct
    entries = select_rdna_general_entries(cells, tie_eps=0.0)
    assert set(entries) == {_KEY_A} and entries[_KEY_A].finalist == "f1"


_REAL_JUDGE_DIR = os.path.expanduser("~/tasks/bwd_rdna_gpu")


@pytest.mark.parametrize("arch", ["gfx1151", "gfx1201"])
def test_reimplements_the_historical_judge_selection(arch):
    """Where the real on-device judge documents for this PR's sweeps are
    available (outside the source tree -- CI and most checkouts skip this),
    the in-repo rule picks exactly the finalist the historical run shipped,
    for every key."""
    import ast

    path = os.path.join(_REAL_JUDGE_DIR, f"judge_{arch}.json")
    if not os.path.exists(path):
        pytest.skip(f"{path} not present on this host")
    with open(path) as f:
        doc = json.load(f)
    cells = load_cell_verdicts(doc)
    got = select_rdna_general_entries(cells, tie_eps=0.0)
    want = {}
    for slice_doc in doc["slices"].values():
        for key_str, rec in slice_doc["keys"].items():
            if rec.get("entry"):
                want[tuple(ast.literal_eval(key_str))] = rec["entry"]
    assert {k: v.finalist for k, v in got.items()} == want
