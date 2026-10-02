# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Knob ordinal -> value tables, observed from enumeration rather than re-derived.

Non-int KMD fields are pinned by an index into the engine's value set (RFC 0019 §13.2).
The engine owns that numbering, so this module only records what candidates show.
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

    Raises on a contradiction: it means the engine's numbering shifted mid-corpus.
    """
    table = {} if table is None else table
    for candidate in candidates:
        knobs = candidate.get("knob_settings") or {}
        features = candidate.get("kernel_features") or {}
        for name, pinned in knobs.items():
            actual = features.get(KERNEL_PREFIX + name)
            if actual is None:
                continue
            entry = table.setdefault(name, {})
            known = entry.get(pinned)
            observed = _hashable(actual)
            if known is not None and known != observed:
                raise ValueError(
                    f"knob {name!r} ordinal {pinned} addressed {known!r} and then {observed!r}; "
                    "the engine's numbering changed during collection, so rows recorded "
                    "before the change no longer address the kernels they measured"
                )
            entry[pinned] = observed
    return table


def is_ordinal(table: dict, name: str) -> bool:
    """True when the knob addresses by index rather than by its own value."""
    return any(pinned != value for pinned, value in table.get(name, {}).items())


def decode(table: dict, name: str, pinned: int):
    """The value an ordinal addressed; refuses an ordinal the corpus never observed."""
    entry = table.get(name)
    if entry is None or pinned not in entry:
        raise ValueError(f"knob {name!r} has no observed value for ordinal {pinned}")
    return entry[pinned]


def as_manifest(table: dict) -> dict:
    """The table as JSON: per knob, a list of {pin, value} ordered by ordinal."""
    manifest = {}
    for name, entry in sorted(table.items()):
        manifest[name] = {
            "ordinal": is_ordinal(table, name),
            "values": [
                {
                    "pin": pin,
                    "value": list(value) if isinstance(value, tuple) else value,
                }
                for pin, value in sorted(entry.items())
            ],
        }
    return manifest


def merge_manifests(manifests) -> dict:
    """One addressing table from several collections' `as_manifest` records.

    Shards record subsets of one numbering; a pin that maps to different values is refused.
    """
    table: dict = {}
    for index, manifest in enumerate(manifests):
        for name, record in (manifest or {}).items():
            entry = table.setdefault(name, {})
            for item in record.get("values", []):
                pinned, observed = item["pin"], _hashable(item["value"])
                known = entry.get(pinned)
                if known is not None and known != observed:
                    raise ValueError(
                        f"knob {name!r} ordinal {pinned} addressed {known!r} in one collection and "
                        f"{observed!r} in another (collection {index}); the engine numbered them "
                        "differently, so they cannot train one model"
                    )
                entry[pinned] = observed
    return as_manifest(table)


def unaddressable(exposed_knobs, table: dict) -> list[str]:
    """Exposed knobs no candidate ever pinned."""
    return sorted(name for name in exposed_knobs if not table.get(name))
