################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
# P4 round 1 — characterization of TensileMergeLibrary.mergeLogic (the core merge
# loop, source lines ~223-373). ADD-ONLY: a NEW file alongside the existing
# test_tensile_merge_library_char.py (which intentionally left mergeLogic as
# resistance). Logic data is list-indexed: data[5]=solutions, data[7]=size map.
# These pin the ACTUAL current merge behavior (size add / efficiency update /
# forceMerge / noEff) — no source change.
################################################################################
import importlib

import pytest

pytestmark = pytest.mark.unit

M = importlib.import_module("Tensile.TensileMergeLibrary")


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(M, "verbosity", 1)


def _sol(idx, name="s", kname="k", **extra):
    d = {"SolutionIndex": idx, "SolutionNameMin": name, "KernelNameMin": kname}
    d.update(extra)
    return d


def _make_data(solutions, sizes):
    """Build minimal logic data: [_, _, _, _, _, solutions, _, sizes]."""
    d = [None] * 8
    d[5] = solutions
    d[7] = sizes
    return d


def test_merge_logic_empty_incremental():
    """Empty incremental data: base unchanged."""
    base_sols = [_sol(0, "s0"), _sol(1, "s1")]
    base_sizes = [[[1, 2, 3, 4], [0, 0.9]], [[5, 6, 7, 8], [1, 0.8]]]
    base_data = _make_data(base_sols, base_sizes)

    inc_data = _make_data([], [])

    merged, sizes_added, sols_added, sols_removed = M.mergeLogic(base_data, inc_data, forceMerge=False)
    assert sizes_added == 0
    assert sols_added == 0


def test_merge_logic_new_size():
    """Incremental has new size not in base."""
    base_sols = [_sol(0, "s0")]
    base_sizes = [[[1, 2, 3, 4], [0, 0.9]]]
    base_data = _make_data(base_sols, base_sizes)

    inc_sols = [_sol(0, "s1")]
    inc_sizes = [[[5, 6, 7, 8], [0, 0.8]]]  # new size
    inc_data = _make_data(inc_sols, inc_sizes)

    merged, sizes_added, sols_added, sols_removed = M.mergeLogic(base_data, inc_data, forceMerge=False)
    assert sizes_added == 1
    assert sols_added == 1
    assert len(merged[7]) == 2


def test_merge_logic_existing_size_improved():
    """Existing size with better efficiency -> update."""
    base_sols = [_sol(0, "s0")]
    base_sizes = [[[1, 2, 3, 4], [0, 0.5]]]  # eff=0.5
    base_data = _make_data(base_sols, base_sizes)

    inc_sols = [_sol(0, "s1")]
    inc_sizes = [[[1, 2, 3, 4], [0, 0.9]]]  # same size, better eff=0.9
    inc_data = _make_data(inc_sols, inc_sizes)

    merged, sizes_added, sols_added, sols_removed = M.mergeLogic(base_data, inc_data, forceMerge=False)
    assert sizes_added == 0
    assert sols_added == 1


def test_merge_logic_existing_size_not_improved():
    """Existing size with worse efficiency -> skip."""
    base_sols = [_sol(0, "s0")]
    base_sizes = [[[1, 2, 3, 4], [0, 0.9]]]  # eff=0.9
    base_data = _make_data(base_sols, base_sizes)

    inc_sols = [_sol(0, "s1")]
    inc_sizes = [[[1, 2, 3, 4], [0, 0.5]]]  # same size, worse eff=0.5
    inc_data = _make_data(inc_sols, inc_sizes)

    merged, sizes_added, sols_added, sols_removed = M.mergeLogic(base_data, inc_data, forceMerge=False)
    assert sizes_added == 0
    assert sols_added == 0


def test_merge_logic_force_merge():
    """With forceMerge=True, accept non-improved sizes."""
    base_sols = [_sol(0, "s0")]
    base_sizes = [[[1, 2, 3, 4], [0, 0.9]]]  # eff=0.9
    base_data = _make_data(base_sols, base_sizes)

    inc_sols = [_sol(0, "s1")]
    inc_sizes = [[[1, 2, 3, 4], [0, 0.5]]]  # same size, worse eff=0.5
    inc_data = _make_data(inc_sols, inc_sizes)

    merged, sizes_added, sols_added, sols_removed = M.mergeLogic(base_data, inc_data, forceMerge=True)
    assert sols_added == 1


def test_merge_logic_no_eff_mode():
    """With noEff=True, store eff as 0.0 instead of actual value."""
    base_sols = [_sol(0, "s0")]
    base_sizes = [[[1, 2, 3, 4], [0, 0.5]]]
    base_data = _make_data(base_sols, base_sizes)

    inc_sols = [_sol(0, "s1")]
    inc_sizes = [[[5, 6, 7, 8], [0, 0.9]]]  # new size, high eff
    inc_data = _make_data(inc_sols, inc_sizes)

    merged, sizes_added, sols_added, sols_removed = M.mergeLogic(base_data, inc_data, forceMerge=False, noEff=True)
    assert merged[7][1][1][1] == 0.0  # noEff should set eff to 0.0
