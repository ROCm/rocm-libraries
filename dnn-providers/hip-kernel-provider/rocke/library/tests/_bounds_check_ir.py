# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Structural checks that a built kernel bounds its paged reads to its sequence.

An attention kernel walks fixed-size KV tiles and fixed-height Q blocks, so it
runs past the sequence whenever the length is not a tile multiple. Scores for
those positions are masked to -inf and contribute nothing, but the masking
happens *after* the address is formed: an unbounded block-table read returns an
arbitrary page id and the K/V load built from it is an illegal access (AICK-2955
on the gfx942 D128 body, D256 q=129 on the lean body).

These helpers walk the builder's IR and assert the index DAG of each read passes
through the clamp, so the contract is checked on any host without a GPU. The GPU
oracles are the matching ``*_bounds_numeric`` suites.

Two clamp shapes are recognised:

  * ``select(cmp, x, load(param) - 1)`` -- bound to the last key or token, and
  * ``smin(x, (load(param) - 1) >> log2(block_size))`` -- bound to the last page.

The page form pins the shift: one short would bound to a page past the last.
"""

from __future__ import annotations


def walk_ops(region):
    """Every op in ``region``, descending into nested regions."""
    for op in region.ops:
        yield op
        for sub in op.regions:
            yield from walk_ops(sub)


def global_loads(kernel, param):
    """Every ``memref.global_load`` in ``kernel`` whose base pointer is ``param``."""
    return [
        op
        for op in walk_ops(kernel.body)
        if op.name.startswith("memref.global_load")
        and op.operands[0].name == f"%{param}"
    ]


def is_last_index_of(value, param):
    """``value`` is ``load(param, ...) - 1`` (the last valid key / token)."""
    op = value.op
    if op is None or op.name != "arith.sub":
        return False
    lhs, rhs = op.operands
    return (
        lhs.op is not None
        and lhs.op.name.startswith("memref.global_load")
        and lhs.op.operands[0].name == f"%{param}"
        and rhs.op is not None
        and rhs.op.name == "arith.constant"
        and rhs.op.attrs.get("value") == 1
    )


def is_last_page_of(value, param, block_size):
    """``value`` is ``(load(param, ...) - 1) >> log2(block_size)`` (the last valid
    page). The shift is pinned: one short would bound to a page past the last."""
    op = value.op
    if op is None or op.name != "arith.lshr":
        return False
    shift = op.operands[1].op
    return (
        is_last_index_of(op.operands[0], param)
        and shift is not None
        and shift.name == "arith.constant"
        and shift.attrs.get("value") == block_size.bit_length() - 1
    )


def clamped_to_last_of(value, param, block_size=None):
    """The index DAG of ``value`` passes through ``select(cmp, x, load(param)-1)``
    (last key / token) or, given ``block_size``, through
    ``smin(x, (load(param)-1) >> log2(block_size))`` (last page)."""
    seen, stack = set(), [value]
    while stack:
        v = stack.pop()
        if id(v) in seen or v.op is None:
            continue
        seen.add(id(v))
        op = v.op
        if op.name == "arith.select" and is_last_index_of(op.operands[2], param):
            return True
        if (
            block_size is not None
            and op.name == "arith.smin"
            and any(is_last_page_of(x, param, block_size) for x in op.operands)
        ):
            return True
        stack.extend(op.operands)
    return False
