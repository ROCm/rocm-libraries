# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Interpreter for the MMA catalog's hardware-notation layout strings (test helper).

A layout string such as ``K{2, 4} L{K1M} V{K2K0}`` says which matrix element each (lane, register
slot) of one wave holds. The grammar is documented in ``helpers/tiling/traits/README.md``
("Layout strings"); in short:

- ``X{s0}`` / ``X{s1, s0}`` splits dim ``X`` (``M``, ``N``, ``K`` or ``KP``) into sub-dims: ``X0``
  of size ``s0`` (fastest), ``X1`` of size ``s1``, and the last sub-dim takes the rest of ``X``.
- ``L{...}`` lists the dims and sub-dims that make up the lane index, most significant first.
- ``V{...}`` lists those that make up the register slot index, most significant first.

A bare name is a whole dim; ``B`` is the block index, ``R`` a replication index (every replica holds
the same element, so ``R`` is not part of the element's coordinate). A sub-dim's value combines with
its siblings into the dim's coordinate: ``X = X0 + s0 * (X1 + s1 * X2)``.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Mapping

from rocke.helpers.tiling.traits import MmaTraits

__all__ = ["interpret_layout", "operand_layout_strings"]

_SPLIT = re.compile(r"(KP|[KMN])\{(\d+)(?:, (\d+))?\}")
_GROUP = re.compile(r"([LV])\{([^}]*)\}")
_NAME = re.compile(r"(KP|[KMNBR])(\d?)")


def operand_layout_strings(traits: MmaTraits) -> dict[str, str]:
    """The A, B and C/D layout strings of one catalog row, with ``b_layout: "same"`` spelled out
    (A's string with ``M`` read as ``N``)."""
    b_layout = traits.b_layout
    if b_layout == "same":
        b_layout = traits.a_layout.replace("M", "N")
    return {"A": traits.a_layout, "B": b_layout, "C": traits.c_d_layout}


def _parse_names(text: str, layout: str) -> list[tuple[str, str]]:
    names = _NAME.findall(text)
    if "".join(dim + index for dim, index in names) != text:
        raise ValueError(f"layout {layout!r}: cannot read the index list {text!r}")
    return names


def interpret_layout(
    layout: str,
    sizes: Mapping[str, int],
    wave_size: int,
    element_dims: tuple[str, ...],
) -> dict[tuple[int, int], tuple[tuple[str, int], ...]]:
    """Map every ``(lane, slot)`` of one wave to the element it holds.

    ``sizes`` gives each dim's full size (``M``, ``N``, ``K``, ``KP``, ``B``, ``R``). The element is a
    sorted tuple of ``(dim, coordinate)`` pairs: one per dim of ``element_dims`` (a dim of size 1
    the string does not name is at coordinate 0), plus ``B`` when the string names it. Raises
    ``ValueError`` if the string is malformed or its lanes do not cover ``wave_size``.
    """
    splits: dict[str, list[int]] = {}
    rest = layout
    for match in _SPLIT.finditer(layout):
        dim, first, second = match.group(1), int(match.group(2)), match.group(3)
        factors = [first] if second is None else [int(second), first]  # fastest first
        product = 1
        for factor in factors:
            product *= factor
        if sizes[dim] % product:
            raise ValueError(f"layout {layout!r}: {dim}={sizes[dim]} not divisible")
        splits[dim] = factors + [sizes[dim] // product]
        rest = rest.replace(match.group(0), "", 1)
    groups = {tag: _parse_names(text, layout) for tag, text in _GROUP.findall(rest)}
    if _GROUP.sub("", rest).strip() or "L" not in groups:
        raise ValueError(
            f"layout {layout!r}: expected splits, then L{{...}} and V{{...}}"
        )
    lane_names, slot_names = groups["L"], groups.get("V", [])

    def extent(dim: str, index: str) -> int:
        if index == "":
            return sizes[dim]
        if dim not in splits:
            raise ValueError(f"layout {layout!r}: {dim}{index} names a dim never split")
        return splits[dim][int(index)]

    lane_extents = [extent(*name) for name in lane_names]
    slot_extents = [extent(*name) for name in slot_names]
    lanes = 1
    for value in lane_extents:
        lanes *= value
    if lanes != wave_size:
        raise ValueError(f"layout {layout!r}: {lanes} lanes, expected {wave_size}")

    def flatten(values: tuple[int, ...], extents: list[int]) -> int:
        index = 0
        for value, size in zip(values, extents):
            index = index * size + value
        return index

    names = lane_names + slot_names
    mapping: dict[tuple[int, int], tuple[tuple[str, int], ...]] = {}
    for lane_values in itertools.product(*(range(e) for e in lane_extents)):
        for slot_values in itertools.product(*(range(e) for e in slot_extents)):
            parts: dict[str, dict[str, int]] = {}
            for (dim, index), value in zip(names, lane_values + slot_values):
                parts.setdefault(dim, {})[index] = value
            element = []
            for dim, by_index in parts.items():
                if dim == "R":  # replicas hold the same element
                    continue
                if "" in by_index:
                    coordinate = by_index[""]
                else:
                    coordinate, stride = 0, 1
                    for position, size in enumerate(splits[dim]):
                        coordinate += by_index.get(str(position), 0) * stride
                        stride *= size
                element.append((dim, coordinate))
            for dim in element_dims:
                if dim not in parts:
                    if sizes[dim] != 1:
                        raise ValueError(f"layout {layout!r}: never names dim {dim!r}")
                    element.append((dim, 0))
            key = (
                flatten(lane_values, lane_extents),
                flatten(slot_values, slot_extents),
            )
            mapping[key] = tuple(sorted(element))
    return mapping
