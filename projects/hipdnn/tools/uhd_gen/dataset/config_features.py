# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Turning a kernel's configuration string into features a model can select on.

The integers in a descriptor become positional slots cfg0..cfgN so a model can generalise
over knobs; `scope_by` gives each group its own columns because slot meaning differs per
kernel. Which positions a group uses is read from the corpus, never from the library.

The non-numeric words stay a string: the training tool encodes categories under RFC 0019
§6.5 (`categorical_encoding`, covered by `features_hash`), so this module must not.
"""

from __future__ import annotations

import re
from typing import Iterable

__all__ = [
    "ABSENT",
    "numeric_slots",
    "word_key",
    "required_slots",
    "expand",
    "slots_used_by",
]

#: A missing slot. Not NaN, so "no such field" is a state a tree can split on.
ABSENT = -1

#: A number is a standalone token, never digits inside an identifier (`bf16`, `Filter1x1Pad0`).
_NUMBER = re.compile(r"(?<![A-Za-z0-9_])-?\d+(?![A-Za-z0-9_])")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def numeric_slots(descriptor: str, slots: int) -> list[int]:
    """The integers in `descriptor`, in order, padded to `slots` with `ABSENT`."""
    found = _NUMBER.findall(descriptor or "")
    return [int(found[i]) if i < len(found) else ABSENT for i in range(slots)]


def word_key(descriptor: str) -> str:
    """The descriptor's non-numeric shape, as one string.

    Joined because the words are a combination. Returned as text: §6.5 says who numbers it.
    """
    return "|".join(_WORD.findall(descriptor or ""))


def required_slots(descriptors: Iterable[str]) -> int:
    """How many numeric slots this corpus needs: the widest descriptor in it."""
    return max((len(_NUMBER.findall(text or "")) for text in descriptors), default=0)


def slots_used_by(
    rows: Iterable[list[int]], groups: Iterable
) -> dict[object, list[int]]:
    """Which positions each group actually fills, observed from the corpus."""
    used: dict[object, set[int]] = {}
    for row, group in zip(rows, groups):
        seen = used.setdefault(group, set())
        for index, value in enumerate(row):
            if value != ABSENT:
                seen.add(index)
    return {group: sorted(positions) for group, positions in used.items()}


def expand(
    descriptors: Iterable[str],
    *,
    slots: int | None = None,
) -> tuple[list[list[int]], list[str], int]:
    """Expand descriptors into `(rows_of_slots, word_shapes, slots)`; shapes stay strings."""
    texts = ["" if d is None else str(d) for d in descriptors]
    width = required_slots(texts) if slots is None else slots
    return (
        [numeric_slots(text, width) for text in texts],
        [word_key(text) for text in texts],
        width,
    )
