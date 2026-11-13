# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Address functions: a tensor's strides applied to coordinates, in pure Python.

These are the golden: every address the tiling layer emits is tested against them. They compute
in Python integers, so there is no width limit and no wrap-around to hide a wrong address.

- ``address_of(tensor, position)`` -- one int per tensor axis, in axis order;
  returns ``sum(position[a] * tensor.strides[a])``, in elements.
- ``transform_address_of(graph, tensor, coords)`` -- one int per graph input, in ``inputs(...)``
  order; the same as ``address_of(tensor, graph.evaluate(coords))``.

Both are positional: a mapping (or any non-tuple) is refused, never matched by name. Neither
checks the position against the tensor's lengths: they answer "which element would this lane
address", valid or not, which is what a test of the emitted address needs. Whether a position is
in bounds is ``required_checks``'s question.

Strides must be non-negative, as for the emitted structured address (which splits an address into
parts that are each ``>= 0``); a negative stride is refused here too, so the golden never accepts a
tensor the emitter cannot address.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .graph import (
    TransformGraph,
    _require_input_coordinates,
    _require_outputs_match_rank,
)

if TYPE_CHECKING:
    from ..descriptors import TensorDesc

__all__ = ["address_of", "transform_address_of"]


def address_of(tensor: TensorDesc, position: tuple[int, ...]) -> int:
    """The element address of ``position`` in ``tensor``: ``sum(position[a] * strides[a])``.

    ``position`` holds one int per tensor axis, in tensor-axis order.
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
    _require_nonnegative_strides(tensor)
    return sum(p * s for p, s in zip(position, tensor.strides))


def transform_address_of(
    graph: TransformGraph, tensor: TensorDesc, coords: tuple[int, ...]
) -> int:
    """The element address ``graph`` reaches in ``tensor`` from ``coords``.

    ``coords`` holds one int per graph input, in ``inputs(...)`` order; graph output ``i`` is
    tensor axis ``i``. Equals ``address_of(tensor, graph.evaluate(coords))``.
    """
    _require_input_coordinates(graph, "transform_address_of", coords)
    _require_outputs_match_rank(graph, len(tensor.strides))
    return address_of(tensor, graph.evaluate(coords))


def _require_nonnegative_strides(tensor: TensorDesc) -> None:
    """Refuse a negative stride: the structured address splits into parts that are each >= 0."""
    for axis, stride in enumerate(tensor.strides):
        if stride < 0:
            raise ValueError(
                f"structured address needs non-negative strides; tensor axis {axis} has stride "
                f"{stride}"
            )
