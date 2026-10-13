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


def entry_key(entry: dict, spec_keys: frozenset | None = None) -> tuple:
    """What makes two catalog entries the same kernel. Deliberately excludes
    ``id`` (minted per run) and includes ``priority`` (it orders selection).

    ``spec_keys``: when given, only those spec fields are compared. ``None``
    compares the FULL spec (the ``--strict-spec`` mode, see ``compare_catalog``).
    """
    spec = entry.get("kernel_source", {}).get("spec") or {}
    if spec_keys is not None:
        spec = {k: v for k, v in spec.items() if k in spec_keys}
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


def compare_catalog(
    shipped: dict, regenerated: dict, strict_spec: bool = False
) -> list[str]:
    """Multiset difference over the kernel entries.

    By default the comparison is keyed on the spec fields the SHIPPED catalog
    carries: a registry whose spec dataclass grew knobs after the catalog was
    shipped (hkp_pack hydrates ``Spec(**fields)``, so omitted fields take their
    defaults) is the same catalog as long as every shipped field matches. A
    field the shipped side has and the regenerated side lost, or a changed
    shipped value, is reported either way.

    The caller owns the claim that fields the shipped catalog does not carry
    sit at their expected values (``test_registry_regeneration`` pins them to a
    checked-in literal snapshot). ``strict_spec=True`` compares the FULL spec
    of the two catalogs both ways, so any field only one side carries is a
    difference.
    """
    spec_keys: frozenset | None = None
    if not strict_spec:
        spec_keys = frozenset().union(
            *(
                (e.get("kernel_source", {}).get("spec") or {}).keys()
                for e in shipped["kernelDescriptors"]
            )
        )

    left = collections.Counter(
        entry_key(e, spec_keys) for e in shipped["kernelDescriptors"]
    )
    right = collections.Counter(
        entry_key(e, spec_keys) for e in regenerated["kernelDescriptors"]
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


def _id_index(kdp: dict) -> tuple[dict, list[str]]:
    """Map each kernel entry to its id. Keyed by name; a name used more than
    once is keyed by (name, full spec) instead so repeats are not conflated.
    Returns (index, names still ambiguous after that)."""
    entries = kdp["kernelDescriptors"]
    name_counts = collections.Counter(e.get("name") for e in entries)
    index: dict = {}
    ambiguous: list[str] = []
    for e in entries:
        name = e.get("name")
        key = (
            name
            if name_counts[name] == 1
            else (name, _canonical(e.get("kernel_source", {}).get("spec") or {}))
        )
        if key in index:
            ambiguous.append(str(name))
        index[key] = e.get("id")
    return index, ambiguous


def compare_expected_ids(shipped: dict, regenerated: dict) -> list[str]:
    """``--expect-ids``: the regenerated tree must REUSE the shipped ids.

    Only meaningful after a splice, never straight off ``generate.py``. This is
    what makes ``extend.md``'s 'preserve old names, UUIDs, symbols' checkable
    instead of a review convention. Covers the KDP's own id and references, the
    ``engine_id`` / ``kmd_id`` in its specialization contract, and the id of
    EVERY shipped kernel descriptor -- a loader that resolves kernels by UUID
    breaks on one renumbered entry just as it does on a renumbered engine.
    Kernels missing from the regenerated side are the catalog comparison's
    finding, not repeated here.
    """
    problems: list[str] = []
    for key in ("id",) + _REFERENCE_KEYS:
        if shipped.get(key) != regenerated.get(key):
            problems.append(
                f"{key}: shipped {shipped.get(key)!r} was not preserved "
                f"(regenerated {regenerated.get(key)!r})"
            )

    left_contract = (shipped.get("provenance") or {}).get(
        "specialization_contract"
    ) or {}
    right_contract = (regenerated.get("provenance") or {}).get(
        "specialization_contract"
    ) or {}
    for key in ("engine_id", "kmd_id"):
        if left_contract.get(key) != right_contract.get(key):
            problems.append(
                f"specialization_contract.{key}: shipped {left_contract.get(key)!r} "
                f"was not preserved (regenerated {right_contract.get(key)!r})"
            )

    left_ids, left_amb = _id_index(shipped)
    right_ids, right_amb = _id_index(regenerated)
    for name in sorted(set(left_amb) | set(right_amb)):
        problems.append(
            f"kernel {name!r}: name and spec do not identify one entry, "
            "cannot check its id"
        )
    changed = [
        (k, v, right_ids[k])
        for k, v in left_ids.items()
        if k in right_ids and v != right_ids[k]
    ]
    for key, old, new in changed[:20]:
        label = key if isinstance(key, str) else key[0]
        problems.append(
            f"kernel {label!r}: id {old!r} was not preserved (regenerated {new!r})"
        )
    if len(changed) > 20:
        problems.append(f"... and {len(changed) - 20} more kernel ids not preserved")
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
        "--strict-spec",
        action="store_true",
        help="compare the FULL kernel_source.spec both ways, so a field only "
        "one catalog carries is a difference. By default only the fields the "
        "shipped catalog carries are compared (a dropped or changed shipped "
        "field is always reported). It sees only what the regenerated catalog "
        "holds: a generator config built with the manifest's enumerate."
        "spec_fields projects the spec to those fields, so the comparison is "
        "full-spec only once the shipped catalog carries the full spec.",
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
        (
            "catalog",
            compare_catalog(shipped, regenerated, args.strict_spec),
        ),
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
        f"  spec fields  {'strict (full spec)' if args.strict_spec else 'shipped fields (regenerated-only fields not compared)'}"
    )
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
