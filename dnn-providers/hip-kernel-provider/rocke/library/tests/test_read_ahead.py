# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``read_ahead`` issue order (pure Python, no IR builder)."""

from __future__ import annotations

import pytest

from kernels.common._read_ahead import read_ahead


def _trace(n, depth):
    events = []

    def issue(i):
        events.append(("issue", i))
        return i

    take = read_ahead(n, depth, issue)
    events.append(("primed", None))
    for i in range(n):
        assert take(i) == i
        events.append(("take", i))
    return events


def test_primes_depth_steps_then_issues_one_ahead_per_take():
    assert _trace(4, 2) == [
        ("issue", 0),
        ("issue", 1),
        ("primed", None),
        ("issue", 2),
        ("take", 0),
        ("issue", 3),
        ("take", 1),
        ("take", 2),
        ("take", 3),
    ]


@pytest.mark.parametrize("n, depth", [(3, 0), (3, 5), (0, 2), (1, 1)])
def test_every_step_is_issued_once_and_before_it_is_taken(n, depth):
    events = _trace(n, depth)
    issued = [i for kind, i in events if kind == "issue"]
    assert issued == list(range(n))
    for i in range(n):
        assert events.index(("issue", i)) < events.index(("take", i))
