# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Address functions: a tensor's strides applied to coordinates, in pure Python.

These are the golden: every address the tiling layer emits is tested against them. They compute
in Python integers, so there is no width limit and no wrap-around to hide a wrong address.

- ``address_of(tensor, position, launch_values=())`` -- one int per tensor axis, in axis order;
  returns ``sum((position[a] + begin[a]) * strides[a])``, in elements.
- ``transform_address_of(graph, tensor, coords, launch_values=())`` -- one int per graph input,
  in ``inputs(...)`` order; the same as ``address_of(tensor, graph.evaluate(coords))``.

``begin`` is the tensor's start in its parent, per axis (all 0 for a tensor that is not a slice),
so the address of a slice is the parent's address at ``position + begin``. A runtime stride or a
runtime begin has no value at build time: ``launch_values`` gives the int each one takes at
launch, as ``(Value, int)`` pairs matched by identity -- the stride's or the begin's Value. (Pairs,
not a dict: an IR value is not hashable.)

Both are positional: a mapping (or any non-tuple) is refused, never matched by name. Neither
checks the position against the tensor's lengths: they answer "which element would this lane
address", valid or not, which is what a test of the emitted address needs. Whether a position is
in bounds is ``required_checks``'s question.

Strides must be non-negative, as for the emitted structured address (which splits an address into
parts that are each ``>= 0``); a negative stride is refused here too, so the golden never accepts a
tensor the emitter cannot address.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from rocke.core.ir import Value

from .graph import (
    TransformGraph,
    _require_input_coordinates,
    _require_outputs_match_rank,
)

if TYPE_CHECKING:
    from ..descriptors import TensorDesc

__all__ = ["address_of", "transform_address_of"]


def address_of(
    tensor: TensorDesc,
    position: tuple[int, ...],
    launch_values: Sequence[tuple[Value, int]] = (),
) -> int:
    """The element address of ``position`` in ``tensor``:
    ``sum((position[a] + begin[a]) * strides[a])``.

    ``position`` holds one int per tensor axis, in tensor-axis order. ``launch_values`` gives, as
    ``(Value, int)`` pairs, the int every runtime stride and every runtime begin takes at launch.
    """
    if not isinstance(position, tuple):
        raise TypeError(
            "address_of takes the position as a tuple in tensor-axis order, got "
            f"{type(position).__name__} -- coordinates are positional"
        )
    rank = len(tensor.strides)
    if len(position) != rank:
        raise ValueError(
            f"address_of got {len(position)} coordinates but the tensor has rank {rank} -- "
            "give one int per tensor axis, in axis order"
        )
    strides = [
        _concrete(stride, axis, "stride", launch_values)
        for axis, stride in enumerate(tensor.strides)
    ]
    begins = [
        _concrete(begin, axis, "begin", launch_values)
        for axis, begin in enumerate(tensor.begin)
    ]
    for axis, stride in enumerate(strides):
        _require_nonnegative_stride(axis, stride)
    return sum((p + o) * s for p, o, s in zip(position, begins, strides))


def transform_address_of(
    graph: TransformGraph,
    tensor: TensorDesc,
    coords: tuple[int, ...],
    launch_values: Sequence[tuple[Value, int]] = (),
) -> int:
    """The element address ``graph`` reaches in ``tensor`` from ``coords``.

    ``coords`` holds one int per graph input, in ``inputs(...)`` order; graph output ``i`` is
    tensor axis ``i``. Equals ``address_of(tensor, graph.evaluate(coords), launch_values)``.
    """
    _require_input_coordinates(graph, "transform_address_of", coords)
    _require_outputs_match_rank(graph, len(tensor.strides))
    return address_of(tensor, graph.evaluate(coords), launch_values)


def _concrete(
    entry: int | Value,
    axis: int,
    what: str,
    launch_values: Sequence[tuple[Value, int]],
) -> int:
    """The int a stride or a begin takes: itself, or its value from ``launch_values``."""
    if not isinstance(entry, Value):
        return entry
    for key, number in launch_values:
        if key is entry:
            return number
    raise ValueError(
        f"address_of needs the value of the runtime {what} of axis {axis} -- pass it in "
        "launch_values"
    )


def _require_nonnegative_stride(axis: int, stride: int) -> None:
    if stride < 0:
        raise ValueError(
            f"structured address needs non-negative strides; tensor axis {axis} has stride "
            f"{stride}"
        )


def _require_nonnegative_strides(tensor: TensorDesc) -> None:
    """Refuse a negative int stride: the structured address splits into parts that are each
    >= 0. A runtime stride is stated ``>= 1``; the emitter checks its stated range itself.
    """
    for axis, stride in enumerate(tensor.strides):
        if not isinstance(stride, Value):
            _require_nonnegative_stride(axis, stride)
