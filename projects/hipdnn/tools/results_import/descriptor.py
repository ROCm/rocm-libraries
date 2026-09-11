# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Turning a kernel's configuration string into features a model can select on.

A producer records the configuration it measured as one opaque string -- a tuning tuple, a C++
template instantiation, a bare index -- and passing it through unchanged makes every
configuration of a kernel *identical* to a model, because nothing in the feature row tells them
apart. A two-layer UHD then has a second layer with nothing to choose between, which is not
visible as an error: it ranks, it just cannot prefer. The cost is the whole tuning gain.

Configurations are described positionally rather than by an opaque id. An id can only be
memorised, so a configuration absent from training would be unrankable; the numbers inside the
string are the knobs that actually move performance, and a model can generalise over them.

The positions mean different things for different kernels -- one solver's descriptor is a
36-field tuple, another's a template with six numbers. That is sound because these features are
read by a *grouped* model, whose per-group trees only ever see rows from one group, so a slot
need only be consistent within a group. It is not sound to feed them to a single flat model,
where slot 3 would be a tile width for one kernel and a stage count for another.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping

__all__ = ["MissingVocabularyEntry", "numeric_slots", "word_key", "build_vocabulary", "expand"]

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


class MissingVocabularyEntry(Exception):
    """A descriptor whose word-shape was not in the supplied vocabulary.

    Raised rather than assigned a fresh code. The vocabulary exists so that a corpus encodes the
    same way twice -- a code invented here would not match the one training used, and the row
    would land at an arbitrary point in a split learned from other values, which no error
    downstream would attribute back to here.

    The token sequence is determined by the kernel, so an exhaustive corpus yields a closed
    vocabulary. Reaching this means that assumption no longer holds.
    """


def numeric_slots(descriptor: str, slots: int) -> list[int]:
    """The integers in `descriptor`, in order, padded to `slots` with `ABSENT`."""
    found = _NUMBER.findall(descriptor or "")
    return [int(found[i]) if i < len(found) else ABSENT for i in range(slots)]


def word_key(descriptor: str) -> str:
    """The descriptor's non-numeric shape, as one key.

    Joined rather than kept per-position because the words are a *combination* -- `Default` in a
    template that also says `OddC` is not the same kernel as `Default` alone -- and because
    their count varies between kernels that share a slot layout.
    """
    return "|".join(_WORD.findall(descriptor or ""))


def required_slots(descriptors: Iterable[str]) -> int:
    """How many numeric slots this corpus needs: the widest descriptor in it.

    Sizing from the data rather than a constant, so a kernel with more knobs than today's widest
    is not silently truncated to fit a number chosen when the tool was written.
    """
    return max((len(_NUMBER.findall(text or "")) for text in descriptors), default=0)


def build_vocabulary(descriptors: Iterable[str]) -> dict[str, int]:
    """Assign a code to each distinct word-shape, ordered so the result is reproducible.

    Sorted rather than first-seen: a code that depended on row order would change when shards
    merged in a different sequence, silently re-wiring every split the model learned on it.
    """
    return {key: index for index, key in enumerate(sorted({word_key(d) for d in descriptors}))}


def expand(
    descriptors: Iterable[str],
    *,
    slots: int | None = None,
    vocabulary: Mapping[str, int] | None = None,
) -> tuple[list[list[int]], list[int], dict[str, int], int]:
    """Expand descriptors into numeric slots and a variant code.

    Returns `(rows_of_slots, variant_codes, vocabulary, slots)`. A vocabulary is built from the
    data when not supplied; when supplied it is authoritative, and a word-shape outside it
    raises rather than extending it.
    """
    texts = ["" if d is None else str(d) for d in descriptors]
    width = required_slots(texts) if slots is None else slots

    if vocabulary is None:
        vocabulary = build_vocabulary(texts)
        codes = [vocabulary[word_key(text)] for text in texts]
    else:
        codes = []
        for text in texts:
            key = word_key(text)
            if key not in vocabulary:
                raise MissingVocabularyEntry(
                    f"descriptor shape {key!r} is not in the supplied vocabulary (from "
                    f"{descriptors if len(texts) < 4 else 'this corpus'}). The vocabulary is "
                    "closed by construction, so this corpus was not measured against the same "
                    "kernels as the one it will be scored beside."
                )
            codes.append(vocabulary[key])

    return [numeric_slots(text, width) for text in texts], codes, dict(vocabulary), width
