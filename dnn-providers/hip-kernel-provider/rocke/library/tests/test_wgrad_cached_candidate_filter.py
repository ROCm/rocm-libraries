# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Regression test for the cached-candidate wgrad two-stage filter.

``_run_from_cache`` offers AOT-cached two-stage (scratch + reduce) wgrad
kernels next to the packed-atomic ones for every C/groups shape, unlike the
JIT sweep which only *builds* two-stage kernels for odd C/groups. Only an
explicit ``--two-stage never`` is allowed to drop them. A regression here
would silently reinstate the old odd-C/groups-only filter and drop valid
even-shape two-stage candidates from "auto"/"always" runs.
"""

from __future__ import annotations

from benchmarks.common.benchmark_implicit_gemm_conv import (
    _filter_wgrad_two_stage_candidates,
)


class _FakeIdent:
    def __init__(self, two_stage: bool):
        self.two_stage = two_stage


def _candidates():
    # One atomic (non-two-stage) candidate and one two-stage candidate; the
    # path/meta slots are unused by the filter.
    return [
        (_FakeIdent(two_stage=False), None, None),
        (_FakeIdent(two_stage=True), None, None),
    ]


def test_auto_and_always_preserve_every_candidate():
    # The filter takes no problem/C-groups argument -- it cannot see parity --
    # so "auto" and "always" are expected to behave identically here.
    for mode in ("auto", "always"):
        candidates = _candidates()
        filtered = _filter_wgrad_two_stage_candidates(candidates, mode)
        assert filtered == candidates, mode
        assert any(c[0].two_stage for c in filtered), mode
        assert any(not c[0].two_stage for c in filtered), mode


def test_never_drops_only_the_two_stage_candidates():
    candidates = _candidates()
    filtered = _filter_wgrad_two_stage_candidates(candidates, "never")
    assert all(not c[0].two_stage for c in filtered)
    assert len(filtered) == 1
    assert filtered[0] is candidates[0]


def test_never_leaves_an_all_atomic_candidate_list_untouched():
    candidates = [(_FakeIdent(two_stage=False), None, None)]
    assert _filter_wgrad_two_stage_candidates(candidates, "never") == candidates
