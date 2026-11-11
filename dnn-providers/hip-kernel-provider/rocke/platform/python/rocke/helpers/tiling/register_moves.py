# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Compile-time register shuffles on a fragment's storage vector.

A register move copies element ``src`` of one vector to element ``dst`` of another, with
``vec_extract`` / ``vec_insert``. :func:`move_registers_into` writes the moves into an existing
vector; :func:`move_registers` writes them into a new zero vector. Their users: ``load_fragment``
(:mod:`rocke.helpers.tiling.emit`) places each loaded run into the fragment's registers, the register
reorder (:func:`rocke.helpers.tiling.transforms.transform_fragment`) permutes them, and the MMA
driver slices an atom's operand out of them. Which storage registers hold which logical registers
is :func:`rocke.helpers.tiling.fragments.storage_slots`.
"""

from __future__ import annotations

from collections.abc import Iterable

from ...core.ir import IRBuilder, Value

__all__ = ["move_registers", "move_registers_into"]


def move_registers_into(
    b: IRBuilder, target: Value, source: Value, moves: Iterable[tuple[int, int]]
) -> Value:
    """`target` with, for each ``(src, dst)`` of `moves` in order, element ``src`` of `source`
    written at element ``dst``: one ``vec_extract`` and one ``vec_insert`` per move. Elements no
    move writes keep their value in `target`. `source` and `target` share an element type.
    """
    for src, dst in moves:
        target = b.vec_insert(target, b.vec_extract(source, src), dst)
    return target


def move_registers(
    b: IRBuilder, source: Value, moves: Iterable[tuple[int, int]], count: int
) -> Value:
    """A new ``count``-element vector of `source`'s element type holding the `moves` of
    :func:`move_registers_into`; elements no move writes are zero."""
    return move_registers_into(b, b.zero_vec(source.type.elem, count), source, moves)
