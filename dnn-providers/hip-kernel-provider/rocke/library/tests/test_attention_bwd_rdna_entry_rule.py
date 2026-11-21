# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The RDNA tuned-table entry rule
(:mod:`benchmarks.common.attention_bwd_rdna_entry_rule`).

CPU only, synthetic records (no device data): exercises every branch of the
selection rule, including the ones the shipped RDNA entries never took (every
shipped finalist was correct, A/A-clean and not slower than shipped on every
cell of its key -- see ``rdna_entries_manifest.json``). Each test is built so
that dropping the check it names changes the outcome.
"""

from __future__ import annotations

import math

import pytest

from benchmarks.common.attention_bwd_rdna_entry_rule import (
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
    entries = select_rdna_general_entries([_cell(ratio=0.7)])
    assert set(entries) == {_KEY_A}
    assert entries[_KEY_A].finalist == "f1"
    assert entries[_KEY_A].geomean_ratio_vs_shipped == pytest.approx(0.7)
    assert entries[_KEY_A].cells_checked == 1


def test_an_incorrect_cell_rejects_its_finalist():
    cells = [_cell(correct=False, verdict="fail", ratio=0.5)]
    assert select_rdna_general_entries(cells) == {}


def test_a_dirty_aa_control_rejects_its_finalist():
    cells = [_cell(aa_clean=False, ratio=0.5)]
    assert select_rdna_general_entries(cells) == {}


def test_slower_than_shipped_rejects_its_finalist():
    cells = [_cell(verdict="fail", ratio=1.2)]
    assert select_rdna_general_entries(cells) == {}


def test_a_slower_cell_rejects_the_finalist_even_if_other_cells_of_the_key_pass():
    """Every cell of the key must pass; one bad cell sinks the finalist even
    though its geomean over the key (sqrt(0.5 * 1.3) < 1) would ship it."""
    cells = [
        _cell(finalist="f1", ratio=0.5, verdict="pass"),
        _cell(finalist="f1", ratio=1.3, verdict="fail"),
    ]
    assert math.sqrt(0.5 * 1.3) < 1.0
    assert select_rdna_general_entries(cells) == {}


def test_an_unresolved_verdict_is_refused():
    """The bench resolves ``ambiguous`` by its own ``TIE_EPS`` before the
    rule runs; an unresolved verdict reaching the rule is an error, not a
    silent pass or fail."""
    with pytest.raises(ValueError, match="verdict"):
        _cell(verdict="ambiguous", ratio=0.99)


def test_lowest_geomean_finalist_wins_among_qualifiers():
    cells = [
        _cell(finalist="f1", ratio=0.9),
        _cell(finalist="f2", ratio=0.6),
        _cell(
            finalist="f3", correct=False, ratio=0.1
        ),  # disqualified despite being fastest
    ]
    entries = select_rdna_general_entries(cells)
    assert entries[_KEY_A].finalist == "f2"


def test_geomean_is_computed_across_the_finalists_own_cells():
    cells = [
        _cell(finalist="f1", ratio=0.5, cells=1),
        _cell(finalist="f1", ratio=0.8, cells=1),
    ]
    entries = select_rdna_general_entries(cells)
    assert entries[_KEY_A].geomean_ratio_vs_shipped == pytest.approx(math.sqrt(0.4))
    assert entries[_KEY_A].cells_checked == 2


def test_a_geomean_of_exactly_1_ships_nothing():
    """A finalist that nets out equal to shipped (a 2x / 0.5x pair) changes
    nothing, even though every individual cell "passed"."""
    cells = [
        _cell(finalist="f1", ratio=0.5, cells=1),
        _cell(finalist="f1", ratio=2.0, cells=1),
    ]
    assert select_rdna_general_entries(cells) == {}


def test_a_finalist_no_faster_than_shipped_overall_ships_nothing():
    cells = [_cell(ratio=1.0)]
    assert select_rdna_general_entries(cells) == {}


def test_a_key_with_no_qualifying_finalist_is_absent():
    cells = [
        _cell(correct=False, ratio=0.5),
        _cell(finalist="f2", aa_clean=False, ratio=0.4),
    ]
    assert select_rdna_general_entries(cells) == {}


def test_keys_are_independent():
    cells = [_cell(key=_KEY_A, ratio=0.9), _cell(key=_KEY_B, finalist="g", ratio=0.4)]
    entries = select_rdna_general_entries(cells)
    assert set(entries) == {_KEY_A, _KEY_B}
    assert entries[_KEY_B].finalist == "g"


def test_load_cell_verdicts_folds_a_nonempty_why_into_a_dirty_aa_control():
    doc = {
        "slices": {
            "d64": {
                "keys": {
                    str(list(_KEY_A)): {
                        "cells": 3,
                        "finalists": {
                            "f1": {"ok": True, "geomean_vs_shipped": 0.8, "why": []},
                            # faster, and ok, but its why names a failed
                            # A/A control: the loader must not ship it
                            "f2": {
                                "ok": True,
                                "geomean_vs_shipped": 0.5,
                                "why": ["cell 2: aa"],
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
    assert not by_finalist["f2"].aa_clean
    entries = select_rdna_general_entries(cells)
    assert set(entries) == {_KEY_A} and entries[_KEY_A].finalist == "f1"
