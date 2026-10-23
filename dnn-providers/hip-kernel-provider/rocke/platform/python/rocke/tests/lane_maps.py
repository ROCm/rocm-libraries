# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test helper: evaluate an MMA lane map on integers.

A :class:`~rocke.core.arch.LayoutMap` emits the index arithmetic of one
(lane, slot) -> coordinate rule through an IR builder. Handed :class:`_Evaluate`
instead of a builder, the same calls compute the coordinates on the host, so a
test can read the whole map, compare it to a formula, or use it to place a
matrix into each lane's registers.
"""

from __future__ import annotations

__all__ = ["lane_placements"]


class _Evaluate:
    """The builder calls a lane map makes, as integer arithmetic."""

    const_i32 = staticmethod(int)
    mod = staticmethod(lambda a, b: a % b)
    div = staticmethod(lambda a, b: a // b)
    add = staticmethod(lambda a, b: a + b)
    mul = staticmethod(lambda a, b: a * b)


def lane_placements(layout, wave_size: int) -> list[tuple[int, int]]:
    """The coordinate of every (lane, slot) of ``layout``, lane-major."""
    return [
        layout.coord(_Evaluate, lane, slot)
        for lane in range(wave_size)
        for slot in range(layout.frag_len)
    ]
