# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The gfx11 default plan: a smaller ladder rung outranks a larger one.

Within one (op, dtype, class, dropout) the priority of a ``gfx11-generic`` backward row falls strictly with its tile, so the
heuristic's default order takes the smallest rung that covers the problem and never falls back to the kernel-id tie-break.
Pure Python.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pytest

FLYDSL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLYDSL))
from generators._instances import instances_for  # noqa: E402


@pytest.mark.parametrize("kind", ["prep", "dkdv", "dq"])
def test_priority_falls_strictly_with_tile(kind: str) -> None:
    groups = defaultdict(dict)
    for row in instances_for(f"sdpa_bwd_{kind}", "gfx11-generic"):
        knobs = row.knobs
        key = (row.dtype, knobs["causal"], knobs["has_bias"], knobs["dropout"])
        tile = knobs["head_dim"]
        assert tile not in groups[key], f"duplicate tile {tile} in {key}"
        groups[key][tile] = row.priority
    assert groups
    for key, by_tile in groups.items():
        ordered = [by_tile[tile] for tile in sorted(by_tile)]
        assert all(a > b for a, b in zip(ordered, ordered[1:])), (key, by_tile)
