# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Make every kernel addressable by a knob tuple, whatever its metadata types are.

RFC 0019 §13.2: "The generation UED therefore exposes every addressable KMD field, making the
knob tuple equal to the metadata tuple and every catalog entry individually reachable. A field
the exposure cannot carry leaves its kernels sharing a tuple, and generation stops on that
collision rather than timing an ambiguous candidate."

A knob value is an integer end to end — the bench CLI parses `--knob name=value` as a number,
the backend carries an int64, and the ingestor matches it against the kernel's metadata with
`std::get_if<int64_t>`. So `int` fields address themselves and the other four KMD types
(`bool`, `float`, `string`, `int_list`) cannot be pinned at all. Exposing only `int` was the
honest response to that; it is also why two kernels differing solely in `dtype` share a tuple
and abort the run.

**Ordinal encoding** closes the gap without changing the wire type. For a non-integer field,
the pinned integer is an INDEX into that field's sorted distinct values, and the kernel whose
value sits at that index is the kernel addressed.

Two properties make this safe to compute independently on both sides rather than shipping a
table:

* **The value set is the engine's, not the graph's.** Indices are assigned over every kernel
  the engine owns, not over the catalog that matched one graph. A per-graph set would give the
  same index different meanings for different graphs, so a recorded tuple would address a
  different kernel when replayed.
* **The order is total and specified.** `bool` false before true, `float` and `int` numerically,
  `string` by code point, `int_list` lexicographically by element. Both sides sort the same
  values the same way, so both arrive at the same index without exchanging anything.

The encoding is recorded in the generation manifest anyway — not because the runtime needs it,
but because an index in a stored row is unreadable without it, and a mismatch between the two
sides should be diagnosable rather than merely wrong.
"""
from __future__ import annotations

from pathlib import Path

from .provenance import descriptor_id, load_descriptor_tree


def engine_kernels(tree: Path, ued_id: str, arch: str | None = None) -> list[dict]:
    """Every kernel the engine owns, under the same arch rule the runtime loads by.

    The ordinal tables are the ENGINE's value set, so they must be built over the same packs
    the runtime will join into the catalog: the arch filter here mirrors
    `provenance_for_engine`, and a pack for another arch contributes no values because its
    kernels never become candidates on this machine.
    """
    index = load_descriptor_tree(tree)
    kernels: list[dict] = []
    for path, pack in index["kdp"].values():
        if descriptor_id(pack.get("engine"), f"{path}.engine") != ued_id:
            continue
        arches = pack.get("arch", [])
        if arch not in (None, "default") and arches and arch not in arches:
            continue
        kernels.extend(entry for entry in pack.get("kernelDescriptors", []) if isinstance(entry, dict))
    return kernels

#: KMD metadata types, per `hkp_pack/descriptors.py::_METADATA_TYPES`.
METADATA_TYPES = ("bool", "int", "float", "string", "int_list")

#: Types that address themselves: the value IS the pin.
NATIVE_TYPES = ("int",)


def _sort_key(field_type: str, value):
    """A total order per type, so two implementations agree without sharing state."""
    if field_type == "bool":
        return (0, int(bool(value)))
    if field_type in ("int", "float"):
        return (0, float(value))
    if field_type == "string":
        # By code point, not by locale: a locale-dependent collation would assign different
        # indices on two machines and silently address different kernels.
        return (0, str(value))
    if field_type == "int_list":
        return (0, tuple(int(item) for item in value))
    raise ValueError(f"unknown KMD field type {field_type!r}")


def _canonical(field_type: str, value):
    """The value as it is compared and recorded, independent of JSON spelling."""
    if field_type == "bool":
        return bool(value)
    if field_type == "int":
        return int(value)
    if field_type == "float":
        return float(value)
    if field_type == "string":
        return str(value)
    if field_type == "int_list":
        return tuple(int(item) for item in value)
    raise ValueError(f"unknown KMD field type {field_type!r}")


def encodings(kmd: dict, kernels: list[dict]) -> dict[str, list]:
    """Ordinal tables for every non-native field the KMD declares, in index order.

    `kernels` is the engine's whole kernel inventory -- every `kernelDescriptors` entry of
    every pack that joins it.

    **KMD defaults are applied first.** The runtime builds its catalog from completed metadata
    (`KernelIngestorStateManager::validateAndIndexPacks` -> `completeMetadata`, which fills a
    field the kernel omits from the KMD's `default`), so a table built from authored values
    alone would be missing exactly the values the defaults supply. The two sides would then
    number the same field differently and a recorded pin would address a different kernel --
    silently, because both indices are valid integers. A kernel that omits a field the KMD
    gives no default for is a load error there and contributes nothing here.
    """
    tables: dict[str, list] = {}
    for field in kmd.get("fields", []):
        name, field_type = field["name"], field["type"]
        if field_type in NATIVE_TYPES:
            continue
        if field_type not in METADATA_TYPES:
            raise ValueError(f"KMD field {name!r} has unknown type {field_type!r}")
        has_default = "default" in field
        observed = set()
        for kernel in kernels:
            metadata = kernel.get("metadata")
            if not isinstance(metadata, dict):
                continue
            if name in metadata:
                observed.add(_canonical(field_type, metadata[name]))
            elif has_default:
                observed.add(_canonical(field_type, field["default"]))
        if observed:
            tables[name] = [value for value in sorted(observed, key=lambda v: _sort_key(field_type, v))]
    return tables


def exposable(kmd: dict, tables: dict[str, list]) -> list[str]:
    """The knob names the collection UED exposes: native fields plus every encoded one."""
    names = []
    for field in kmd.get("fields", []):
        if field["type"] in NATIVE_TYPES or field["name"] in tables:
            names.append(field["name"])
    return names


def pin(field_type: str, value, table: list | None) -> int:
    """The integer that addresses `value` for a field of this type.

    A value the table does not contain is a caller error rather than a miss: the table was
    built from the same inventory the kernel came from, so an absent value means the two sides
    disagree about what the engine owns -- which is exactly the condition that would otherwise
    address the wrong kernel silently.
    """
    if field_type in NATIVE_TYPES:
        return int(value)
    canonical = _canonical(field_type, value)
    if table is None:
        raise ValueError(f"no ordinal table for a {field_type} field")
    try:
        return table.index(canonical)
    except ValueError as error:
        raise ValueError(
            f"value {canonical!r} is not in the engine's value set for this field; "
            "the generator and the descriptor set disagree about the kernel inventory"
        ) from error


def decode(field_type: str, index: int, table: list | None):
    """The value a pin addressed -- for reading a recorded row back."""
    if field_type in NATIVE_TYPES:
        return int(index)
    if table is None or not 0 <= index < len(table):
        raise ValueError(f"ordinal {index} is outside this field's value set")
    return table[index]


def unaddressable(kmd: dict, tables: dict[str, list]) -> list[str]:
    """Fields no kernel carries, which therefore address nothing.

    Reported rather than silently omitted: a field declared and never populated is a pack
    defect, and the collision it eventually causes is far from the cause.
    """
    return [field["name"] for field in kmd.get("fields", [])
            if field["type"] not in NATIVE_TYPES and field["name"] not in tables]
