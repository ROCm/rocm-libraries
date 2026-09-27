# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""What a knob ordinal means, learned from the engine rather than re-derived.

RFC 0019 §13.2: "The generation UED therefore exposes every addressable KMD field, making
the knob tuple equal to the metadata tuple and every catalog entry individually reachable."

A knob value is an integer end to end -- the bench CLI parses `--knob name=value` as a
number, the backend carries an int64, and the ingestor matches it against the kernel's
metadata. So `int` fields address themselves and the other four KMD types (`bool`, `float`,
`string`, `int_list`) are addressed by an INDEX into the engine's value set for that field.

**The engine owns that numbering, and this module does not reproduce it.** Enumeration
already returns, for every candidate, both halves of the mapping:

    knob_settings   {"dtype": 0, "block_m": 256, ...}   <- what addresses this kernel
    kernel_features {"kernel.dtype": "BF16", ...}       <- what the kernel actually is

so `0 -> "BF16"` is an observation, not a derivation. An earlier version of this module
computed the table a second time from the descriptor tree and held the two in agreement by
contract: sort the same way, apply KMD defaults the same way, span the same packs. Each of
those is a way to disagree silently -- a mismatch produces a valid integer that addresses
the wrong kernel -- and the defaults rule had already been got wrong once.

What is observed is exactly what is addressable: a value no candidate carried cannot be
pinned, so its absence from the table costs nothing.
"""
from __future__ import annotations

#: KMD metadata types, per `hkp_pack/descriptors.py::_METADATA_TYPES`.
METADATA_TYPES = ("bool", "int", "float", "string", "int_list")

#: Types that address themselves: the value IS the pin.
NATIVE_TYPES = ("int",)

KERNEL_PREFIX = "kernel."


def _hashable(value):
    """A JSON value as a dict key: `int_list` arrives as a list, which is not one."""
    return tuple(value) if isinstance(value, list) else value


def observe(candidates, table: dict | None = None) -> dict:
    """Extend `table` with the (knob, ordinal) -> value pairs these candidates show.

    `candidates` are enumerated catalog entries as the bench emits them: each carries a
    complete `knob_settings` tuple and the `kernel_features` of the kernel that tuple
    addresses. A knob whose pinned integer equals the kernel's own value is a native `int`
    field and is recorded as such, so a reader can tell "the value is 256" from "the value
    is the one at index 0".

    Raises on a contradiction rather than overwriting. Two candidates disagreeing about
    what ordinal 1 means is the engine's numbering shifting mid-corpus, which would make
    every recorded row before the shift address a different kernel on replay.
    """
    table = {} if table is None else table
    for candidate in candidates:
        knobs = candidate.get("knob_settings") or {}
        features = candidate.get("kernel_features") or {}
        for name, pinned in knobs.items():
            actual = features.get(KERNEL_PREFIX + name)
            if actual is None:
                # The knob names no kernel field this candidate published; nothing to learn.
                continue
            entry = table.setdefault(name, {})
            known = entry.get(pinned)
            observed = _hashable(actual)
            if known is not None and known != observed:
                raise ValueError(
                    f"knob {name!r} ordinal {pinned} addressed {known!r} and then {observed!r}; "
                    "the engine's numbering changed during collection, so rows recorded "
                    "before the change no longer address the kernels they measured")
            entry[pinned] = observed
    return table


def is_ordinal(table: dict, name: str) -> bool:
    """Does this knob address by index rather than by its own value?

    True when any observed pin differs from the value it addressed -- which is what an
    ordinal IS. An `int` field pins its own value, so every pair agrees.
    """
    return any(pinned != value for pinned, value in table.get(name, {}).items())


def decode(table: dict, name: str, pinned: int):
    """The value an ordinal addressed, for reading a recorded row back.

    Refuses an unobserved ordinal rather than guessing: an index the corpus never saw
    names no kernel here, and returning the neighbour that happens to sit at it is how a
    replay silently measures something else.
    """
    entry = table.get(name)
    if entry is None or pinned not in entry:
        raise ValueError(f"knob {name!r} has no observed value for ordinal {pinned}")
    return entry[pinned]


def as_manifest(table: dict) -> dict:
    """The table in a form that survives JSON and stays readable.

    Ordinals are dict keys, which JSON stringifies, so each knob becomes a list ordered by
    ordinal with explicit indices. Recorded for reading, not for use: the runtime derives
    its own numbering and the corpus is unreadable without knowing what the integers meant.
    """
    manifest = {}
    for name, entry in sorted(table.items()):
        manifest[name] = {
            "ordinal": is_ordinal(table, name),
            "values": [{"pin": pin, "value": list(value) if isinstance(value, tuple) else value}
                       for pin, value in sorted(entry.items())],
        }
    return manifest


def unaddressable(exposed_knobs, table: dict) -> list[str]:
    """Exposed knobs no candidate ever pinned.

    Reported rather than ignored: a knob the engine advertises and never uses addresses
    nothing, and the tuple collision it eventually causes surfaces far from this cause.
    """
    return sorted(name for name in exposed_knobs if not table.get(name))
