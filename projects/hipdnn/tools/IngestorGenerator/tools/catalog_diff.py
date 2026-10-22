# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Compare a regenerated bundle against the shipped one, UUID-blind.

The no-op gate. Before a manifest or registry change can be trusted to preserve a
shipped catalog, regeneration has to be shown to reproduce it -- and the one
comparison that cannot be used is byte equality, because ``codegen/generator.py`` mints
every descriptor id with ``uuid.uuid4()``: two runs of the SAME config differ in
every id. So the ids are masked, and everything they point at is compared
instead.

    catalog_diff.py --shipped <a.kdp.json> --regenerated <b.kdp.json> [--ignore-ids]
                    [--strict-spec | --subset]

The two halves are compared by different rules, because they are produced by
different things (see the generator's own split: only ``packs[].kernels`` comes
from the dispatcher):

  * the CATALOG -- ``kernelDescriptors`` -- as a multiset of ``(name, kernel_source
    (every key; the spec filtered as below), arch, metadata, priority, version)``. Order is not meaningful,
    but multiplicity is: two entries that differ only by id are two compiled
    kernels, and dropping one silently halves what ships.
  * the ENGINE-LEVEL blocks -- ``engine``/``dispatch``/``matchers`` reference
    arity and ``provenance.specialization_contract`` content -- compared with ids
    masked. Native symbol strings in the UED/UHD/UDD/UMD are NOT compared here
    (only the KDP is read); symbol drift is a separate follow-up gate.

Exit 0 on no differences, 1 on any difference (with a report), 2 on bad usage or
unreadable input, including an ``--expect-ids`` name present in both catalogs used a
different number of times.

``--subset`` requires only shipped ⊆ regenerated: kernels only the regenerated
catalog holds are listed as registry extras, not differences.

Without ``--strict-spec`` (including ``--subset``) the spec comparison is keyed on
the fields the SHIPPED catalog carries, so a regenerated kernel whose extra field
holds a non-default value still passes. The ``spec fields`` header line lists those
regenerated-only fields; whoever reads a pass owns the claim that they sit at their
defaults (``test_registry_regeneration`` asserts it for the shipped engine).

What a pass does NOT establish: that the regenerated bundle is CORRECT, only
that it is the same catalog. A manifest that is wrong in the same way twice
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
#: ``engine_id`` / ``kmd_id`` appear in each consumer of the KDP's
#: specialization_contract; they reference the same freshly-minted UED and KMD
#: and differ for the same reason as the top-level ``id`` field.
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
    if not isinstance(kdp, dict) or "kernelDescriptors" not in kdp:
        raise CatalogDiffError(
            f"the {label} file has no 'kernelDescriptors'; is it a KDP?"
        )
    if not isinstance(kdp["kernelDescriptors"], list):
        raise CatalogDiffError(
            f"the {label} KDP 'kernelDescriptors' must be an array, "
            f"got {type(kdp['kernelDescriptors']).__name__}"
        )
    for index, entry in enumerate(kdp["kernelDescriptors"]):
        where = f"the {label} KDP kernelDescriptors[{index}]"
        if not isinstance(entry, dict):
            raise CatalogDiffError(f"{where} must be an object")
        source = entry.get("kernel_source", {})
        if not isinstance(source, dict):
            raise CatalogDiffError(f"{where} 'kernel_source' must be an object")
        if not isinstance(source.get("spec", {}), dict):
            raise CatalogDiffError(f"{where} 'kernel_source.spec' must be an object")
        if "name" in entry and not isinstance(entry["name"], str):
            raise CatalogDiffError(f"{where} 'name' must be a string")
        for scalar in ("priority", "version"):
            if not isinstance(entry.get(scalar), (str, int, float, bool, type(None))):
                raise CatalogDiffError(f"{where} '{scalar}' must be a scalar")
    _check_provenance(kdp.get("provenance"), f"the {label} KDP")
    return kdp


def _check_provenance(provenance, where: str) -> None:
    """``provenance.specialization_contract.consumers[]`` as the comparisons read it."""
    if provenance is None:
        return
    if not isinstance(provenance, dict):
        raise CatalogDiffError(f"{where} 'provenance' must be an object")
    contract = provenance.get("specialization_contract")
    if contract is None:
        return
    if not isinstance(contract, dict):
        raise CatalogDiffError(
            f"{where} 'provenance.specialization_contract' must be an object"
        )
    consumers = contract.get("consumers")
    if consumers is not None and (
        not isinstance(consumers, list)
        or not all(isinstance(c, dict) for c in consumers)
    ):
        raise CatalogDiffError(
            f"{where} 'provenance.specialization_contract.consumers' must be an "
            f"array of objects"
        )


def _canonical(value) -> str:
    """Stable text for a nested value, so dicts can key a Counter."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _spec_fields(kdp: dict) -> frozenset:
    """Every ``kernel_source.spec`` field name any entry of ``kdp`` carries."""
    return frozenset().union(
        *(
            (e.get("kernel_source", {}).get("spec") or {}).keys()
            for e in kdp["kernelDescriptors"]
        )
    )


def _entry_counters(
    shipped: dict, regenerated: dict, spec_keys: frozenset | None
) -> tuple[collections.Counter, collections.Counter]:
    """The shipped and regenerated entries as multisets over ``entry_key``."""
    return (
        collections.Counter(
            entry_key(e, spec_keys) for e in shipped["kernelDescriptors"]
        ),
        collections.Counter(
            entry_key(e, spec_keys) for e in regenerated["kernelDescriptors"]
        ),
    )


def binary_identity(entry: dict, spec_keys: frozenset | None = None) -> tuple:
    """What names the kernel an entry ships: its whole ``kernel_source`` (every key,
    whatever the kind) with the spec reduced to ``spec_keys`` (``None``: the full
    spec), and the entry's ``arch``."""
    source = dict(entry.get("kernel_source") or {})
    spec = source.get("spec") or {}
    if spec_keys is not None:
        spec = {k: v for k, v in spec.items() if k in spec_keys}
    source["spec"] = spec
    return (_canonical(source), _canonical(entry.get("arch")))


def entry_key(entry: dict, spec_keys: frozenset | None = None) -> tuple:
    """What makes two catalog entries the same kernel. Deliberately excludes
    ``id`` (minted per run) and includes ``priority`` (it orders selection).

    ``spec_keys``: when given, only those spec fields are compared. ``None``
    compares the FULL spec (the ``--strict-spec`` mode, see ``compare_catalog``).
    """
    return (
        entry.get("name"),
        *binary_identity(entry, spec_keys),
        _canonical(entry.get("metadata")),
        entry.get("priority"),
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
    sit at their defaults (``test_registry_regeneration`` asserts it against
    the spec dataclass). ``strict_spec=True`` compares the FULL spec both ways
    instead, so any regenerated-only field is a difference.
    """
    spec_keys = None if strict_spec else _spec_fields(shipped)
    left, right = _entry_counters(shipped, regenerated, spec_keys)
    problems: list[str] = []

    total_left, total_right = sum(left.values()), sum(right.values())
    if total_left != total_right:
        problems.append(
            f"kernel count: shipped {total_left}, regenerated {total_right} "
            f"({total_right - total_left:+d})"
        )

    missing = left - right
    added = right - left
    for key, count in sorted(missing.items(), key=lambda kv: repr(kv[0]))[:20]:
        problems.append(f"  only in shipped    (x{count}): {key[0]}")
    if len(missing) > 20:
        problems.append(f"  ... and {len(missing) - 20} more shipped-only entries")
    for key, count in sorted(added.items(), key=lambda kv: repr(kv[0]))[:20]:
        problems.append(f"  only in regenerated(x{count}): {key[0]}")
    if len(added) > 20:
        problems.append(f"  ... and {len(added) - 20} more regenerated-only entries")
    return problems


def compare_subset(shipped: dict, regenerated: dict) -> tuple[list[str], list[str]]:
    """(shipped entries the regenerated catalog lacks, regenerated-only names)."""
    left, right = _entry_counters(shipped, regenerated, _spec_fields(shipped))
    missing = left - right
    problems = [
        f"  only in shipped    (x{count}): {key[0]}"
        for key, count in sorted(missing.items(), key=lambda kv: repr(kv[0]))[:20]
    ]
    if len(missing) > 20:
        problems.append(f"  ... and {len(missing) - 20} more shipped-only entries")
    extras = [
        str(key[0])
        for key, count in sorted((right - left).items(), key=lambda kv: repr(kv[0]))
        for _ in range(count)
    ]
    return problems, extras


def find_duplicate_specs(kdp: dict) -> list[str]:
    """Two entries naming one binary.

    The packer dedups compilation on the kernel's source and full spec, so two
    catalog entries with one `binary_identity` are one kernel under two names --
    and if their completed metadata also collides, the loader keeps one and the
    catalog silently ships less than it claims. Uses the full spec (not a filtered
    subset) since the packer dedup itself uses the full spec.
    """
    by_binary: dict[tuple, list[str]] = collections.defaultdict(list)
    for entry in kdp["kernelDescriptors"]:
        by_binary[binary_identity(entry)].append(entry.get("name", "<unnamed>"))
    return [
        f"  {len(names)} entries share one spec: {', '.join(sorted(names)[:4])}"
        + (" ..." if len(names) > 4 else "")
        for names in by_binary.values()
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
    if not isinstance(kmd, dict) or "fields" not in kmd:
        raise CatalogDiffError(f"the {label} file has no 'fields'; is it a KMD?")
    if not isinstance(kmd["fields"], list) or not all(
        isinstance(field, dict) for field in kmd["fields"]
    ):
        raise CatalogDiffError(f"the {label} KMD 'fields' must be an array of objects")
    for index, field in enumerate(kmd["fields"]):
        if "name" in field and not isinstance(field["name"], str):
            raise CatalogDiffError(
                f"the {label} KMD fields[{index}] 'name' must be a string"
            )
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


def _name_counts(kdp: dict) -> collections.Counter:
    return collections.Counter(e.get("name") for e in kdp["kernelDescriptors"])


def _id_index(kdp: dict) -> tuple[dict, list[str]]:
    """Map each kernel entry to its id. Keyed by name; a name used more than
    once is keyed by (name, binary identity) instead, so repeats -- per-arch twins
    included -- are not conflated.
    Returns (index, names still ambiguous after that)."""
    entries = kdp["kernelDescriptors"]
    name_counts = _name_counts(kdp)
    index: dict = {}
    ambiguous: list[str] = []
    for e in entries:
        name = e.get("name")
        key = name if name_counts[name] == 1 else (name, *binary_identity(e))
        if key in index:
            ambiguous.append(str(name))
        index[key] = e.get("id")
    return index, ambiguous


def _contract_ids(kdp: dict) -> list[tuple]:
    """``(engine_id, kmd_id)`` of each ``specialization_contract.consumers[]``."""
    contract = (kdp.get("provenance") or {}).get("specialization_contract") or {}
    return [
        (consumer.get("engine_id"), consumer.get("kmd_id"))
        for consumer in contract.get("consumers") or []
    ]


def compare_expected_ids(shipped: dict, regenerated: dict) -> list[str]:
    """``--expect-ids``: the regenerated tree must REUSE the shipped ids.

    Only meaningful after a splice, never straight off ``generate.py``. This is
    what makes ``extend.md``'s 'preserve old names, UUIDs, symbols' checkable
    instead of a review convention. Covers the KDP's own id and references, the
    ``engine_id`` / ``kmd_id`` of each contract consumer, and the id of
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

    left_consumers, right_consumers = _contract_ids(shipped), _contract_ids(regenerated)
    if left_consumers != right_consumers:
        problems.append(
            "specialization_contract.consumers[] engine_id/kmd_id: shipped "
            f"{left_consumers!r} was not preserved (regenerated {right_consumers!r})"
        )

    left_counts, right_counts = _name_counts(shipped), _name_counts(regenerated)
    repeated = sorted(
        str(name)
        for name, count in left_counts.items()
        if name in right_counts and right_counts[name] != count
    )
    if repeated:
        raise CatalogDiffError(
            f"--expect-ids cannot check the id of {repeated}: the name is used a "
            f"different number of times in the shipped and regenerated catalogs, so "
            f"its entries cannot be paired"
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


def _spec_mode(shipped: dict, regenerated: dict, strict_spec: bool) -> str:
    """The header line saying which spec fields the catalog comparison covered.

    Outside ``--strict-spec`` a field only the regenerated catalog carries is
    never compared, so it names them: a clean pass must not read as covering them.
    """
    if strict_spec:
        return "strict (full spec)"
    uncompared = sorted(_spec_fields(regenerated) - _spec_fields(shipped))
    if not uncompared:
        return "shipped fields (the regenerated catalog adds no spec field)"
    shown = ", ".join(uncompared[:10]) + (" ..." if len(uncompared) > 10 else "")
    return (
        f"shipped fields ({len(uncompared)} regenerated-only spec fields NOT "
        f"compared: {shown})"
    )


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
        "the regenerated catalog carries is a difference. By default (and with "
        "--subset) only the fields the shipped catalog carries are compared, a "
        "dropped or changed shipped field is always reported, and the header "
        "names the regenerated-only fields that were NOT compared.",
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
    parser.add_argument(
        "--subset",
        action="store_true",
        help="require only shipped ⊆ regenerated; regenerated-only kernels are "
        "listed as registry extras. Excludes --strict-spec.",
    )
    args = parser.parse_args(argv)

    if args.subset and args.strict_spec:
        print(
            "FAIL: --subset and --strict-spec are mutually exclusive",
            file=sys.stderr,
        )
        return 2

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

    extras: list[str] = []
    if args.subset:
        catalog_problems, extras = compare_subset(shipped, regenerated)
    else:
        catalog_problems = compare_catalog(shipped, regenerated, args.strict_spec)
    sections: list[tuple[str, list[str]]] = [
        ("catalog", catalog_problems),
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
        try:
            sections.append(
                ("preserved ids", compare_expected_ids(shipped, regenerated))
            )
        except CatalogDiffError as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 2

    print("catalog diff")
    print(f"  shipped      {len(shipped['kernelDescriptors'])} kernels")
    print(f"  regenerated  {len(regenerated['kernelDescriptors'])} kernels")
    print(f"  ids          {'masked' if args.ignore_ids else 'compared'}")
    print(f"  spec fields  {_spec_mode(shipped, regenerated, args.strict_spec)}")
    print(
        f"  catalog      {'subset (shipped kernels must all be regenerated)' if args.subset else 'equal'}"
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

    if args.subset:
        print(f"\n  registry extras (not shipped): {len(extras)}")
        for name in extras[:20]:
            print(f"    {name}")
        if len(extras) > 20:
            print(f"    ... and {len(extras) - 20} more")

    if failed:
        print("\nDIFFER", file=sys.stderr)
        return 1
    print("\n  no differences")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
