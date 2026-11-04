# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Software read-ahead over an unrolled loop, as a pure emission-order helper.

An unrolled loop whose step ``i`` consumes operands loaded by ``issue(i)``
hides the load latency by issuing step ``i + depth``'s loads before step ``i``
consumes its own. :func:`read_ahead` owns only that bookkeeping: it never
touches the IR builder, so every op it causes comes from the caller's
``issue`` callback, in exactly the order a hand-written prefetch would emit
it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

__all__ = ["read_ahead"]

T = TypeVar("T")


def read_ahead(n: int, depth: int, issue: Callable[[int], T]) -> Callable[[int], T]:
    """Prime steps ``[0, min(depth, n))`` now; return ``take``.

    ``take(i)`` -- called once per step, in order -- first issues step
    ``i + depth`` (when it exists) and then hands back step ``i``'s operands.
    ``depth == 0`` degenerates to issuing each step just before it is taken.
    """
    pending: dict[int, T] = {i: issue(i) for i in range(min(depth, n))}

    def take(i: int) -> T:
        if i + depth < n:
            pending[i + depth] = issue(i + depth)
        return pending.pop(i)

    return take
