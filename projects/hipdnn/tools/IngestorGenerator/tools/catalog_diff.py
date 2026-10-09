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

KMD comparison (``--shipped-kmd`` / ``--regenerated-kmd``):

The KDP carries only field NAMES via the ``specialization_contract``; the KMD
JSON holds field types and ``default_value`` entries, which are not visible in
the KDP. A ``default_value`` change (e.g., 512 → 256 for ``block_n``) silently
changes how the matcher completes partial records, so it is a semantic change that
the catalog comparison misses without the KMD. Pass both KMD files to catch it.
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


def entry_key(entry: dict, shipped_spec_keys: frozenset | None = None) -> tuple:
    """What makes two catalog entries the same kernel. Deliberately excludes
    ``id`` (minted per run) and includes ``priority`` (it orders selection).

    ``shipped_spec_keys``: when given (from the shipped entry's own spec), only
    spec fields present in the SHIPPED entry are compared. Regenerated-only
    fields (e.g. newly defaulted knobs like ``lds_k_row_pad``) are ignored --
    they are not part of what the shipped catalog committed to. But ALL shipped
    fields are compared, not only the metadata_fields subset: a pin flip to
    ``attention_dense_persist`` changes ``persistent`` (a shipped field), and
    a knobs-list rollback would likewise alter shipped fields that the
    metadata_fields filter would miss.
    """
    spec = entry.get("kernel_source", {}).get("spec") or {}
    if shipped_spec_keys is not None:
        spec = {k: v for k, v in spec.items() if k in shipped_spec_keys}
    ks = entry.get("kernel_source", {})
    return (
        entry.get("name"),
        _canonical(spec),
        _canonical(entry.get("metadata")),
        entry.get("priority"),
        ks.get("kind"),
        ks.get("source"),
        ks.get("builder"),
        entry.get("version"),
    )


def compare_catalog(shipped: dict, regenerated: dict) -> list[str]:
    """Multiset difference over the kernel entries.

    For each shipped entry, the spec key is restricted to the fields that entry
    ITSELF carries: regenerated-only fields (newly added defaults) are ignored so
    they do not produce false differences. But ALL shipped fields are compared,
    not just the metadata_fields subset -- a pin flip or knob-list rollback
    changes shipped fields that a narrower filter would miss.
    """
    # Build a per-entry key using the shipped entry's own spec field names.
    # Regenerated entries use those same keys so the comparison is symmetric.
    left_keys: list = []
    shipped_spec_keys_per_entry: list[frozenset | None] = []
    for e in shipped["kernelDescriptors"]:
        spec_keys = frozenset((e.get("kernel_source", {}).get("spec") or {}).keys())
        shipped_spec_keys_per_entry.append(spec_keys if spec_keys else None)
        left_keys.append(entry_key(e, spec_keys if spec_keys else None))

    # Derive the union of all shipped spec keys so the regenerated side is
    # filtered to the same vocabulary; entries with no shipped counterpart get
    # None (full spec) so they never silently collapse.
    all_shipped_keys: frozenset | None = (
        frozenset().union(*shipped_spec_keys_per_entry)  # type: ignore[arg-type]
        if shipped_spec_keys_per_entry
        else None
    )

    left = collections.Counter(left_keys)
    right = collections.Counter(
        entry_key(e, all_shipped_keys) for e in regenerated["kernelDescriptors"]
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
    silently ships less than it claims. Uses the full spec (not a filtered
    subset) since the packer dedup itself uses the full spec.
    """
    by_spec: dict[str, list[str]] = collections.defaultdict(list)
    for entry in kdp["kernelDescriptors"]:
        spec = entry.get("kernel_source", {}).get("spec") or {}
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


def _load_kmd(path: str, label: str) -> dict:
    try:
        text = Path(path).read_text()
    except OSError as exc:
        raise CatalogDiffError(f"cannot read the {label} KMD: {exc}") from exc
    try:
        kmd = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CatalogDiffError(f"the {label} KMD is not valid JSON: {exc}") from exc
    if "fields" not in kmd:
        raise CatalogDiffError(f"the {label} file has no 'fields'; is it a KMD?")
    return kmd


def compare_kmd_fields(shipped_kmd: dict, regenerated_kmd: dict) -> list[str]:
    """Compare KMD field names, types, and default_value entries.

    The KDP comparison catches field NAME drift through the specialization_contract,
    but the KMD holds field TYPES and default_value entries that are invisible in
    the KDP. A changed default_value changes how the C++ matcher completes partial
    records -- e.g., a record that omits block_n falls back to the KMD default, so
    512 vs 256 dispatches to a different binary even though the catalog looks identical.
    """
    problems: list[str] = []

    def _field_key(f: dict) -> str:
        return f.get("name", "")

    shipped_by_name = {
        f["name"]: f for f in shipped_kmd.get("fields", []) if "name" in f
    }
    regen_by_name = {
        f["name"]: f for f in regenerated_kmd.get("fields", []) if "name" in f
    }

    only_shipped = set(shipped_by_name) - set(regen_by_name)
    only_regen = set(regen_by_name) - set(shipped_by_name)
    for name in sorted(only_shipped):
        problems.append(f"  kmd field only in shipped:     {name!r}")
    for name in sorted(only_regen):
        problems.append(f"  kmd field only in regenerated: {name!r}")

    for name in sorted(set(shipped_by_name) & set(regen_by_name)):
        sf, rf = shipped_by_name[name], regen_by_name[name]
        for attr in ("type", "default_value"):
            sv, rv = sf.get(attr), rf.get(attr)
            if sv != rv:
                problems.append(
                    f"  kmd field {name!r} {attr}: shipped {sv!r}, regenerated {rv!r}"
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
    parser.add_argument(
        "--shipped-kmd",
        default=None,
        help="shipped *.kmd.json; when given, KMD field types and defaults are "
        "compared against --regenerated-kmd.",
    )
    parser.add_argument(
        "--regenerated-kmd",
        default=None,
        help="regenerated *.kmd.json to compare against --shipped-kmd.",
    )
    args = parser.parse_args(argv)

    if bool(args.shipped_kmd) != bool(args.regenerated_kmd):
        print(
            "FAIL: --shipped-kmd and --regenerated-kmd must be given together",
            file=sys.stderr,
        )
        return 2

    try:
        shipped = _load(args.shipped, "shipped")
        regenerated = _load(args.regenerated, "regenerated")
    except CatalogDiffError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    shipped_kmd: dict | None = None
    regenerated_kmd: dict | None = None
    if args.shipped_kmd:
        try:
            shipped_kmd = _load_kmd(args.shipped_kmd, "shipped")
            regenerated_kmd = _load_kmd(args.regenerated_kmd, "regenerated")
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
    if shipped_kmd is not None:
        sections.append(
            (
                "kmd field types and defaults",
                compare_kmd_fields(shipped_kmd, regenerated_kmd),
            )
        )
    if args.expect_ids:
        sections.append(("preserved ids", compare_expected_ids(shipped, regenerated)))

    print("catalog diff")
    print(f"  shipped      {len(shipped['kernelDescriptors'])} kernels")
    print(f"  regenerated  {len(regenerated['kernelDescriptors'])} kernels")
    print(f"  ids          {'masked' if args.ignore_ids else 'compared'}")
    print(
        f"  kmd          {'compared' if shipped_kmd is not None else 'not provided (types/defaults unchecked)'}"
    )

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
