# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Turning a kernel's configuration string into features a model can select on.

A producer records the configuration it measured as one opaque string -- a tuning tuple, a C++
template instantiation, a bare index -- and passing it through unchanged makes every
configuration of a kernel *identical* to a model, because nothing in the feature row tells them
apart. A two-layer UHD then has a second layer with nothing to choose between, which is not
visible as an error: it ranks, it just cannot prefer. The cost is the whole tuning gain.

Configurations are described positionally: the integers in a descriptor become cfg0..cfgN in
order. An opaque id would only be memorisable, so a configuration absent from training would be
unrankable, where the numbers inside the string are the knobs that actually move performance and
a model can generalise over them.

The positions mean different things for different kernels -- one solver's descriptor is a
36-field tuple, another's a template with six numbers -- so slot 3 is a tile width for one and a
stage count for another. `scope_by` resolves that by giving each group its own columns, which
costs width and buys columns that mean one thing. Measured on a gfx942 corpus it moved total
regret from 0.1007 to 0.0889 on 1,095 unseen problems, effectively all of it in the *group*
decision: the first layer sees every row, so it was the one being asked to split on a column
with no consistent meaning.

Unscoped is still correct for an engine whose configuration schema does not vary, and for a
grouped model the second layer is unaffected either way -- its trees only ever see one group's
rows, where the positions were already consistent.

Which positions a group uses is read from the corpus, never from the library that produced it.
A column named `s107_f12` says "group 107's twelfth number" and claims nothing about what that
number means; the tuning dataset is the whole input, and a schema recovered from a library's
headers or its shipped metadata would be a second source to keep in step with it.

**The non-numeric part stays a string.** The words in a descriptor (`Default`, `OddC`,
`Filter1x1Pad0`) are a category, and RFC 0019 §6.5 already says what happens to one: the
*training tool* observes the values and ships a `categorical_encoding` in the UHD, covered by
`features_hash` and applied by the runtime. Assigning codes here instead would put a second
string-to-number mechanism outside that contract -- per-corpus, unhashed, and re-derived on every
import, so adding one kernel silently renumbers every code while the signature text, and
therefore the fingerprint, stays identical. That is precisely the divergence §6.5 exists to
close, so this module does not encode: it names the shape and leaves the number to the tool that
can put it under the contract.
"""

from __future__ import annotations

import re
from typing import Iterable

__all__ = ["ABSENT", "numeric_slots", "word_key", "required_slots", "expand", "slots_used_by"]

#: A missing slot. Not NaN: "this kernel has no such field" is a real state a tree can split on,
#: where NaN is routed by the learner's own default direction and so cannot be distinguished
#: from a field whose value was simply not recorded.
ABSENT = -1

#: A number is a standalone token, never digits inside an identifier. `bf16` is a dtype, not the
#: value 16, and `Filter1x1Pad0` is one variant name rather than three knobs -- harvesting those
#: digits shifts every slot after them and encodes, as a number, information the word key
#: already carries.
_NUMBER = re.compile(r"(?<![A-Za-z0-9_])-?\d+(?![A-Za-z0-9_])")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def numeric_slots(descriptor: str, slots: int) -> list[int]:
    """The integers in `descriptor`, in order, padded to `slots` with `ABSENT`."""
    found = _NUMBER.findall(descriptor or "")
    return [int(found[i]) if i < len(found) else ABSENT for i in range(slots)]


def word_key(descriptor: str) -> str:
    """The descriptor's non-numeric shape, as one string.

    Joined rather than kept per-position because the words are a *combination* -- `Default` in a
    template that also says `OddC` is not the same kernel as `Default` alone -- and because
    their count varies between kernels that share a slot layout.

    Returned as text, not a code. The value is a category and §6.5 says who numbers one.
    """
    return "|".join(_WORD.findall(descriptor or ""))


def required_slots(descriptors: Iterable[str]) -> int:
    """How many numeric slots this corpus needs: the widest descriptor in it.

    Sized from the data rather than a constant, so a kernel with more knobs than today's widest
    is not silently truncated to fit a number chosen when the tool was written.
    """
    return max((len(_NUMBER.findall(text or "")) for text in descriptors), default=0)


def slots_used_by(rows: Iterable[list[int]], groups: Iterable) -> dict[object, list[int]]:
    """Which positions each group actually fills, observed from the corpus.

    A group's schema is whatever its descriptors turned out to carry -- a position no member
    ever fills is not part of it. Read here rather than declared, because the tuning dataset is
    the whole input: a schema taken from the producing library would be a second source that has
    to be kept in step with the corpus, and would be wrong for exactly the corpus that predates
    a library change.
    """
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
    """Expand descriptors into numeric slots and a word shape.

    Returns `(rows_of_slots, word_shapes, slots)`. The shapes are strings: the training tool
    turns them into codes and ships the map, so one import cannot fix a numbering that a later
    one would contradict.
    """
    texts = ["" if d is None else str(d) for d in descriptors]
    width = required_slots(texts) if slots is None else slots
    return [numeric_slots(text, width) for text in texts], [word_key(text) for text in texts], width
