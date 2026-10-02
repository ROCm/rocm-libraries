# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Compare a regenerated bundle against the shipped one, UUID-blind.

The no-op gate. Before a profile change can be trusted to preserve a shipped
catalog, regeneration has to be shown to reproduce it -- and the one comparison
that cannot be used is byte equality, because ``codegen/generator.py`` mints
every descriptor id with ``uuid.uuid4()``: two runs of the SAME config differ in
every id. So the ids are masked, and everything they point at is compared
instead.

    catalog_diff.py --shipped <a.kdp.json> --regenerated <b.kdp.json> [--ignore-ids]

The two halves are compared by different rules, because they are produced by
different things (see the generator's own split: only ``packs[].kernels`` comes
from the dispatcher):

  * the CATALOG -- ``kernelDescriptors`` -- as a multiset of
    ``(name, kernel_source.spec, metadata, priority)``. Order is not meaningful,
    but multiplicity is: two entries that differ only by id are two compiled
    kernels, and dropping one silently halves what ships.
  * the ENGINE-LEVEL blocks -- ``engine``/``dispatch``/``matchers`` references
    and every native symbol string -- compared with ids masked but symbols
    byte-equal, since a symbol is the entire contract with the native pack.

Exit 0 on no differences, 1 on any difference (with a report), 2 on bad usage.

What a pass does NOT establish: that the regenerated bundle is CORRECT, only
that it is the same catalog. A profile that is wrong in the same way twice
passes. And a KDP carries no UUID stability of its own -- the ``--ignore-ids``
mask is why this exits 0 at all, so preserving the SHIPPED ids through a splice
is a separate obligation (``extend.md``: resolve by UUID, not filename), checked
by ``--expect-ids``.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

#: Keys whose values are minted per run. Masked, never compared.
#: ``engine_id`` / ``kmd_id`` appear inside the KDP's specialization_contract;
#: they reference the same freshly-minted UED and KMD and differ for the same
#: reason as the top-level ``id`` field.
_ID_KEYS = ("id", "engine_id", "kmd_id")
#: KDP keys that reference another descriptor by id.
_REFERENCE_KEYS = ("engine", "dispatch", "matchers")


class CatalogDiffError(RuntimeError):
    """Bad usage or unreadable input. Never a difference between two catalogs."""


def _load(path: str, label: str) -> dict:
    try:
        text = Path(path).read_text()
    except OSError as exc:
        raise CatalogDiffError(f"cannot read the {label} KDP: {exc}") from exc
    try:
        kdp = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CatalogDiffError(f"the {label} KDP is not valid JSON: {exc}") from exc
    if "kernelDescriptors" not in kdp:
        raise CatalogDiffError(
            f"the {label} file has no 'kernelDescriptors'; is it a KDP?"
        )
    return kdp


def _canonical(value) -> str:
    """Stable text for a nested value, so dicts can key a Counter."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def entry_key(entry: dict, spec_fields: frozenset | None = None) -> tuple:
    """What makes two catalog entries the same kernel. Deliberately excludes
    ``id`` (minted per run) and includes ``priority`` (it orders selection).

    ``spec_fields``: when given, only these spec fields are compared (the
    metadata_fields from the KDP's specialization_contract). The shipped catalog
    is hand-authored and may omit policy-resolved knobs that the parity tool
    carries; both represent the same compiled binary, so the comparison key is
    the metadata-discriminating subset that the matcher actually uses.
    """
    spec = entry.get("kernel_source", {}).get("spec") or {}
    if spec_fields is not None:
        spec = {k: v for k, v in spec.items() if k in spec_fields}
    return (
        entry.get("name"),
        _canonical(spec),
        _canonical(entry.get("metadata")),
        entry.get("priority"),
    )


def compare_catalog(shipped: dict, regenerated: dict) -> list[str]:
    """Multiset difference over the kernel entries.

    The comparison key uses only the metadata-discriminating spec fields from
    the specialization_contract -- the same subset the C++ matcher reads. The
    shipped catalog may omit policy-resolved knobs the parity tool resolved;
    both represent the same compiled binary, so the full spec is not the key.
    """
    # Derive the spec fields that actually discriminate between kernels.
    # Fall back to None (full spec) when the contract is absent.
    contract = (shipped.get("provenance") or {}).get("specialization_contract")
    spec_fields: frozenset | None = None
    if contract:
        consumers = contract.get("consumers") or []
        all_fields: set[str] = set()
        for c in consumers:
            all_fields.update(c.get("metadata_fields") or [])
            all_fields.update(c.get("matcher_only_fields") or [])
        if all_fields:
            spec_fields = frozenset(all_fields)

    left = collections.Counter(
        entry_key(e, spec_fields) for e in shipped["kernelDescriptors"]
    )
    right = collections.Counter(
        entry_key(e, spec_fields) for e in regenerated["kernelDescriptors"]
    )
    problems: list[str] = []

    total_left, total_right = sum(left.values()), sum(right.values())
    if total_left != total_right:
        problems.append(
            f"kernel count: shipped {total_left}, regenerated {total_right} "
            f"({total_right - total_left:+d})"
        )

    missing = left - right
    added = right - left
    for key, count in sorted(missing.items())[:20]:
        problems.append(f"  only in shipped    (x{count}): {key[0]}")
    if len(missing) > 20:
        problems.append(f"  ... and {len(missing) - 20} more shipped-only entries")
    for key, count in sorted(added.items())[:20]:
        problems.append(f"  only in regenerated(x{count}): {key[0]}")
    if len(added) > 20:
        problems.append(f"  ... and {len(added) - 20} more regenerated-only entries")
    return problems


def find_duplicate_specs(kdp: dict) -> list[str]:
    """Two entries naming one binary.

    The packer dedups compilation on (source, builder, spec), so two catalog
    entries with the same spec are one kernel under two names -- and if their
    completed metadata also collides, the loader keeps one and the catalog
    silently ships less than it claims.
    """
    contract = (kdp.get("provenance") or {}).get("specialization_contract")
    spec_fields: frozenset | None = None
    if contract:
        consumers = contract.get("consumers") or []
        all_fields: set[str] = set()
        for c in consumers:
            all_fields.update(c.get("metadata_fields") or [])
            all_fields.update(c.get("matcher_only_fields") or [])
        if all_fields:
            spec_fields = frozenset(all_fields)

    by_spec: dict[str, list[str]] = collections.defaultdict(list)
    for entry in kdp["kernelDescriptors"]:
        spec = entry.get("kernel_source", {}).get("spec") or {}
        if spec_fields is not None:
            spec = {k: v for k, v in spec.items() if k in spec_fields}
        by_spec[_canonical(spec)].append(entry.get("name", "<unnamed>"))
    return [
        f"  {len(names)} entries share one spec: {', '.join(sorted(names)[:4])}"
        + (" ..." if len(names) > 4 else "")
        for names in by_spec.values()
        if len(names) > 1
    ]


def _masked(value, ignore_ids: bool):
    """Recursively blank anything that is a minted id."""
    if not ignore_ids:
        return value
    if isinstance(value, dict):
        return {
            k: ("<id>" if k in _ID_KEYS else _masked(v, ignore_ids))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_masked(v, ignore_ids) for v in value]
    return value


def compare_engine_blocks(
    shipped: dict, regenerated: dict, ignore_ids: bool
) -> list[str]:
    """Everything in the KDP except the catalog: name, arch, references, and the
    specialization contract. References are masked when ids are ignored -- a
    reference to a freshly minted UED is expected to differ -- but their
    PRESENCE and arity are not: a KDP that lost its matcher list is a different
    engine, however equal the kernels are.
    """
    problems: list[str] = []
    for key in ("name", "arch", "version"):
        if shipped.get(key) != regenerated.get(key):
            problems.append(
                f"{key}: shipped {shipped.get(key)!r}, "
                f"regenerated {regenerated.get(key)!r}"
            )
    for key in _REFERENCE_KEYS:
        left, right = shipped.get(key), regenerated.get(key)
        if (left is None) != (right is None):
            problems.append(f"{key}: present on one side only")
            continue
        if isinstance(left, list) != isinstance(right, list):
            problems.append(f"{key}: list on one side, scalar on the other")
        elif isinstance(left, list) and len(left) != len(right):
            problems.append(
                f"{key}: shipped names {len(left)} reference(s), "
                f"regenerated {len(right)}"
            )
        elif not ignore_ids and left != right:
            problems.append(f"{key}: {left!r} vs {right!r}")

    left_spec = _masked(
        (shipped.get("provenance") or {}).get("specialization_contract"), ignore_ids
    )
    right_spec = _masked(
        (regenerated.get("provenance") or {}).get("specialization_contract"),
        ignore_ids,
    )
    if _canonical(left_spec) != _canonical(right_spec):
        problems.append(
            "provenance.specialization_contract differs (metadata_fields, "
            "bindings or vocabulary changed) -- the agreement between "
            "kernel_source.spec and metadata is not the one that shipped"
        )
    return problems


def compare_expected_ids(shipped: dict, regenerated: dict) -> list[str]:
    """``--expect-ids``: the regenerated tree must REUSE the shipped ids.

    Only meaningful after a splice, never straight off ``generate.py``. This is
    what makes ``extend.md``'s 'preserve old names, UUIDs, symbols' checkable
    instead of a review convention.
    """
    problems: list[str] = []
    for key in ("id",) + _REFERENCE_KEYS:
        if shipped.get(key) != regenerated.get(key):
            problems.append(
                f"{key}: shipped {shipped.get(key)!r} was not preserved "
                f"(regenerated {regenerated.get(key)!r})"
            )
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare a regenerated descriptor catalog against the shipped one."
    )
    parser.add_argument("--shipped", required=True, help="the shipped *.kdp.json")
    parser.add_argument(
        "--regenerated", required=True, help="the freshly generated *.kdp.json"
    )
    parser.add_argument(
        "--ignore-ids",
        action="store_true",
        help="mask minted UUIDs. Required for a straight generate.py output, "
        "whose ids are new by construction.",
    )
    parser.add_argument(
        "--expect-ids",
        action="store_true",
        help="additionally require the shipped ids to be PRESERVED. For checking "
        "a splice, where reusing the shipped UUIDs is the obligation.",
    )
    args = parser.parse_args(argv)

    try:
        shipped = _load(args.shipped, "shipped")
        regenerated = _load(args.regenerated, "regenerated")
    except CatalogDiffError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    sections: list[tuple[str, list[str]]] = [
        ("catalog", compare_catalog(shipped, regenerated)),
        (
            "engine-level descriptors",
            compare_engine_blocks(shipped, regenerated, args.ignore_ids),
        ),
        ("duplicate specs (regenerated)", find_duplicate_specs(regenerated)),
    ]
    if args.expect_ids:
        sections.append(
            ("preserved ids", compare_expected_ids(shipped, regenerated))
        )

    print("catalog diff")
    print(f"  shipped      {len(shipped['kernelDescriptors'])} kernels")
    print(f"  regenerated  {len(regenerated['kernelDescriptors'])} kernels")
    print(f"  ids          {'masked' if args.ignore_ids else 'compared'}")

    failed = False
    for label, problems in sections:
        if problems:
            failed = True
            print(f"\n  {label}:")
            for line in problems:
                print(f"    {line}")

    if failed:
        print("\nDIFFER", file=sys.stderr)
        return 1
    print("\n  no differences")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
