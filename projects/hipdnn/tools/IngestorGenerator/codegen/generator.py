# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Renders a full descriptor bundle for one ``IngestorConfig``.

Descriptor JSON is built as plain Python dicts and serialized with
``json.dumps`` -- not Jinja2 -- because every key emitted must be on that type's
allow-list and an unknown key is a hard error at load, so a dict literal makes
"every key this function writes is a key I chose" a property of the code rather
than of template whitespace. Jinja2 renders the C++ stub/test files and the
CMake/registration text fragments, where hand-tuned whitespace against
``.clang-format`` actually matters.

UUIDs are minted exactly once per run, in :func:`mint_ids`, and threaded
through every cross-reference from that one dict -- never retyped.
"""

import json
import uuid
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .models import (
    KERNEL_SOURCE_KIND_KPACK,
    IngestorConfig,
    KernelSpec,
    PackSpec,
)

#: Two-line AMD copyright + SPDX header every emitted C++/CMake file opens with.
#:
#: It carries U+00A9, which is why every emitted file is written with an EXPLICIT
#: ``encoding="utf-8"``. ``Path.write_text`` otherwise picks the platform's locale
#: codec, and a copyright sign written as cp1252 is a lone ``0xa9`` byte that is not
#: valid UTF-8. `unfilled_placeholders` skips what it cannot decode, so on a
#: non-UTF-8 locale the whole emitted set is undecodable and the placeholder gate
#: reports an empty scan -- green because it read nothing.
CPP_COPYRIGHT_HEADER = (
    "// Copyright \u00a9 Advanced Micro Devices, Inc., or its affiliates.\n"
    "// SPDX-License-Identifier:  MIT\n"
)
CMAKE_COPYRIGHT_HEADER = (
    "# Copyright \u00a9 Advanced Micro Devices, Inc., or its affiliates.\n"
    "# SPDX-License-Identifier:  MIT\n"
)


def mint_ids(config: IngestorConfig) -> dict:
    """Mint every UUID this bundle needs, once, in one dict.

    Every other function in this module reads ids from this dict rather than
    calling ``uuid.uuid4()`` itself.

    Ids are RANDOM, deliberately: deriving one from a name or from metadata makes
    it only as unique as whatever it is keyed on, and the config guarantees
    uniqueness for neither. The properties it does need are enforced by explicit
    checks in `config_loader` that fail loudly and name the offender.

    Indexed by POSITION, not by name, so the lookup stays correct even for a config
    whose names repeat.
    """
    ids = {
        "kmd": str(uuid.uuid4()),
        "ued": str(uuid.uuid4()),
        "kernel_match": str(uuid.uuid4()),
    }
    if config.engine.has_heuristic:
        ids["uhd"] = str(uuid.uuid4())
    ids["udd"] = str(uuid.uuid4())
    for pack_index, pack in enumerate(config.packs):
        ids[("pack", pack_index)] = str(uuid.uuid4())
        if config.is_multi_pack:
            ids[("operation_umd", pack_index)] = str(uuid.uuid4())
        for kernel_index, _kernel in enumerate(pack.kernels):
            ids[("kernel", pack_index, kernel_index)] = str(uuid.uuid4())
    return ids


def _dump(obj: dict) -> str:
    return json.dumps(obj, indent=2, sort_keys=False) + "\n"


def build_kmd(config: IngestorConfig, ids: dict) -> dict:
    fields = []
    for f in config.kmd_fields:
        entry = {"name": f.name, "type": f.type}
        if not f.is_mandatory:
            entry["default_value"] = f.default_value
        fields.append(entry)
    return {
        "version": "1.0",
        "id": ids["kmd"],
        "name": f"{config.engine.local_name} variant fields",
        "fields": fields,
    }


def build_uhd(config: IngestorConfig, ids: dict) -> dict | None:
    if not config.engine.has_heuristic:
        return None
    return {
        "version": "1.0",
        "id": ids["uhd"],
        "name": f"{config.engine.local_name} selector",
        "kind": "native",
        "payload": config.score_symbol,
    }


def build_ued(config: IngestorConfig, ids: dict) -> dict:
    """The engine descriptor.

    ``sdk_version`` is emitted UNCONDITIONALLY, including at the baseline
    ``"1.0.0"``. The runtime parses it onto ``EngineDescriptor::sdkVersion`` and
    ``GenericPlanBuilder::understandsGraph()`` gates on it at match time, so a
    descriptor without one is silently matched against the loader's own baseline
    -- invisibly, and for exactly the engines whose author never thought about it
    -- and a later change of that baseline would re-target every shipped bundle.
    """
    ued = {
        "version": "1.0",
        "id": ids["ued"],
        "name": config.engine.name,
        "sdk_version": config.engine.sdk_version,
        "graph_match": {"native": config.graph_match_symbol},
        "metadata": ids["kmd"],
    }
    if config.engine.has_heuristic:
        ued["heuristic"] = ids["uhd"]
    if config.engine.knobs:
        ued["knobs"] = list(config.engine.knobs)
    if config.engine.behavior_notes:
        ued["behavior_notes"] = list(config.engine.behavior_notes)
    return ued


def build_udd(config: IngestorConfig, ids: dict) -> dict:
    return {
        "version": "1.0",
        "id": ids["udd"],
        "name": f"{config.engine.local_name} dispatch",
        "dispatch_symbol": config.dispatch_symbol,
    }


def build_kernel_match_umd(config: IngestorConfig, ids: dict) -> dict:
    """The shared kernel-scoped dtype matcher -- one per engine, referenced by
    every pack's KDP. A per-*kernel* applicability check the UED's graph_match
    cannot express, having no kernel in scope."""
    return {
        "version": "1.0",
        "id": ids["kernel_match"],
        "name": "kernel dtype matches the graph's dtype",
        "scope": "kernel",
        "match_symbol": config.kernel_match_symbol,
    }


#: KMD sentinel meaning "not set -- the kernel's own policy decides".
#:
#: A CONFIG-AUTHORING convenience that must never reach a descriptor: every compiled
#: kernel has a definite setting for every knob, so a descriptor claiming "unset"
#: mis-describes its own binary.
UNSET_SENTINEL = -1


def _canonical_metadata_value(value, kind: str | None):
    """One value in the spelling its DESTINATION KMD field's declared type ships.

    Canonicalization is DESTINATION-DIRECTED, never value-directed, because the
    same Python value means different things in different fields.

    ``bool`` -> ``int`` only where the field is declared ``int``: the loader's
    ``MetadataField`` compares an int alternative, so a boolean reaching an int
    field never matches anything. Projecting unconditionally would ship ``1``
    where a ``bool`` field's matcher holds ``true`` and decline every graph --
    the same silent decline one field type over.

    ``int`` -> ``float`` where the field is declared ``float``: ``1`` and ``1.0``
    compare equal in Python but serialize to different bytes, so an identity keyed
    on the emitted JSON sees a duplicate catalog tuple.

    Mirrors ``hkp_pack.agreement.canonical``; anything else passes through untouched
    and reaches the config loader's own type check with its authored spelling.
    """
    if kind == "int" and isinstance(value, bool):
        return int(value)
    if (
        kind == "float"
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        return float(value)
    return value


def _resolved_metadata(kernel: KernelSpec, config: IngestorConfig) -> dict:
    """Metadata DERIVED from the spec that built the kernel, not carried beside it.

    A tri-state knob lives in three layers that must agree:

      * ``kernel_source.spec[k]`` -- decides the COMPILED BINARY. Absent means the
        kernel's own policy resolves it at build time.
      * ``metadata[k]``           -- what the MATCHER compares; the catalog key.
      * the KMD field's ``default_value`` -- substituted for anything ABSENT at load.

    Authored metadata can disagree with the binary it labels, silently and in both
    directions, so the spec wins -- but only for knobs the config left UNRESOLVED.
    An authored value is never overwritten: metadata carries the hipDNN spelling the
    matcher compares (`"BF16"`) and the spec carries the builder's (`"bf16"`), so
    copying the spec over one breaks matching while the engine still loads and every
    count still reconciles.

    Resolving a POLICY default is deliberately out of scope: that resolution is
    arch- and shape-dependent and is not importable from this tool. An unresolved
    tri-state is rejected instead -- see `_check_metadata_resolved`. Surviving
    values are canonicalized by `_canonical_metadata_value`.
    """
    types = {f.name: f.type for f in config.kmd_fields}
    spec = kernel.kernel_source.spec or {}
    out = {
        name: _canonical_metadata_value(value, types.get(name))
        for name, value in kernel.metadata.items()
    }
    for field_spec in config.kmd_fields:
        name = field_spec.name
        authored = out.get(name, UNSET_SENTINEL)
        if authored != UNSET_SENTINEL:
            # The author stated it, in the vocabulary the matcher reads. Leave it.
            continue
        if name in spec and spec[name] is not None:
            # Unresolved in metadata but pinned in the spec: the binary is definite,
            # so state that rather than ship "undecided".
            out[name] = _canonical_metadata_value(spec[name], field_spec.type)
    return out


def _check_metadata_resolved(
    kernel: KernelSpec, metadata: dict, config: IngestorConfig
) -> None:
    """Refuse to emit a descriptor whose knob does not describe its binary.

    Two ways a knob can fail to describe the binary, and only one of them is
    visible in the emitted document.

    THE STATED SENTINEL. The author wrote ``-1``. A descriptor is a claim about a
    compiled artifact, and every artifact has a definite setting, so shipping the
    sentinel publishes a claim that is not true of anything.

    THE ABSENT KNOB, which is the same defect with nothing to grep for -- and so the
    one that actually gets shipped. An optional field absent from ``metadata`` AND
    unpinned in ``kernel_source.spec`` looks clean, but the loader keys the catalog
    on the KMD's ``default_value`` while the BINARY was built from the builder
    dataclass's own default. Those two defaults are not required to agree: when they
    disagree the descriptor names one kernel and advertises another, and when they
    agree two entries collapse onto one tuple (see `_completed_metadata`).

    Mandatory fields are not checked here -- the config loader already refuses a
    kernel that omits one, and its diagnostic is better because it names the pack.
    """
    unresolved = sorted(k for k, v in metadata.items() if v == UNSET_SENTINEL)
    if unresolved:
        raise ValueError(
            f"kernel {kernel.name!r} ships the unset sentinel "
            f"({UNSET_SENTINEL}) for {unresolved}: metadata must state the value the "
            f"kernel was BUILT with. Pin the knob in kernel_source.spec, or write the "
            f"resolved value in metadata -- the descriptor cannot say 'undecided' "
            f"about a binary that already decided."
        )

    spec = kernel.kernel_source.spec or {}
    undeclared = sorted(
        f.name
        for f in config.kmd_fields
        if not f.is_mandatory and f.name not in metadata and spec.get(f.name) is None
    )
    if undeclared:
        raise ValueError(
            f"kernel {kernel.name!r} states {undeclared} in neither its metadata nor "
            f"its kernel_source.spec, so nothing here decides the value. The loader "
            f"will substitute the KMD default_value as the catalog key while the "
            f"kernel is compiled from the builder's own default -- two independent "
            f"defaults that are not required to agree, and whose disagreement is "
            f"silent. Pin the knob in kernel_source.spec if the binary should carry "
            f"it, or write the resolved value in metadata if the builder's default "
            f"is what you mean."
        )


def _completed_metadata(metadata: dict, config: IngestorConfig) -> dict:
    """The metadata tuple AS THE LOADER COMPLETES IT, not as it is written.

    The emitted document is not the catalog key. ``DescriptorLoader`` substitutes
    each absent field's KMD ``default_value`` before comparing, so two descriptors
    that differ only in WHICH LAYER stated a value -- one omitting a field, one
    stating it at exactly the KMD default -- are one tuple to the runtime and two
    documents here. Completing the tuple the same way the loader does is what makes
    that collision visible to this tool at all.

    A mandatory field the config omitted has no default to substitute and is left
    out rather than invented -- the config loader already refuses that kernel.
    """
    completed = {}
    for kmd_field in config.kmd_fields:
        if kmd_field.name in metadata:
            value = metadata[kmd_field.name]
        elif kmd_field.is_mandatory:
            continue
        else:
            value = kmd_field.default_value
        completed[kmd_field.name] = _canonical_metadata_value(value, kmd_field.type)
    return completed


def _dedup_key(metadata: dict, config: IngestorConfig) -> str:
    """Identity of a descriptor AS THE MATCHER SEES IT.

    Keyed on the COMPLETED tuple rather than the emitted document, because the
    matcher compares the completed one -- see `_completed_metadata`.

    Architecture is deliberately NOT part of this key. Two candidates with the same
    tuple on disjoint devices are not duplicates at all, and one that overlaps is a
    conflict rather than a droppable repeat; both decisions need to see the whole
    group that shares a tuple, so `build_kdp` makes them against the arch coverage
    it records beside each key.
    """
    return json.dumps(_completed_metadata(metadata, config), sort_keys=True)


def _candidate_identity(kernel: KernelSpec) -> str:
    """What a candidate IS, beyond the tuple that selects it.

    Two entries sharing a completed tuple are the same candidate only if they also
    name the same binary at the same priority. The confusable pair is a knob left
    absent from ``kernel_source.spec`` (the kernel's own policy decides it) beside
    one pinned to the value the policy happened to choose: the SPECS differ, so the
    compiled binaries differ, while the matcher sees one tuple. `build_kdp` refuses
    and names both, since dropping one discards a distinct binary and keeping both
    ships a duplicate tuple that drops the engine at load.
    """
    return json.dumps(
        {
            "kernel_source": kernel.kernel_source.as_document(),
            "priority": kernel.priority,
        },
        sort_keys=True,
    )


def _arch_overlaps(left: list, right: list) -> bool:
    """Whether two arch coverages can select on the same device.

    An EMPTY list is the loader's wildcard -- absent ``arch`` means "any device" --
    so it overlaps everything, including a concrete id. Treating empty as "no
    coverage" would let a wildcard candidate sit beside a concrete one holding the
    same tuple. Mirrors ``hkp_pack.agreement.overlap``.
    """
    return not left or not right or bool(set(left) & set(right))


def _pack_index(config: IngestorConfig, pack: PackSpec) -> int:
    """This pack's position in the config.

    Ids are keyed on position rather than name because nothing guarantees pack names
    are unique, and a name-keyed lookup silently returns another pack's id. Identity
    comparison, not equality: two packs may legitimately compare equal by value.
    """
    for index, candidate in enumerate(config.packs):
        if candidate is pack:
            return index
    raise ValueError(f"pack {pack.name!r} is not part of this config")


def build_operation_umd(
    config: IngestorConfig, pack: PackSpec, ids: dict
) -> dict | None:
    """UMD policy: emitted only for genuine per-pack narrowing, i.e. only
    when the engine has more than one pack. A single-pack engine gets zero
    graph-scoped UMDs -- TestConvFwdPack.cpp asserts exactly this."""
    if not config.is_multi_pack:
        return None
    return {
        "version": "1.0",
        "id": ids[("operation_umd", _pack_index(config, pack))],
        "name": f"graph operation is {pack.discriminator}",
        "scope": "graph",
        "match_symbol": config.operation_match_symbol(pack),
    }


def build_specialization_contract(config: IngestorConfig, ids: dict) -> dict:
    """The ``provenance.specialization_contract`` this bundle's KDP carries.

    The declaration is DATA, and it is self-contained on purpose: a machine checking
    a shipped bundle must be able to say which metadata fields the producing compiler
    specialized on, and how each is read off the builder object, WITHOUT the original
    rocKE installed -- otherwise the check is unrunnable on the machine that received
    the archive rather than the one that built it.

    ``engine_id`` and ``kmd_id`` are the MINTED ids, threaded from `mint_ids`;
    re-deriving them from names would point the declaration at whatever engine
    happened to share a name. Everything else here is authored and loader-validated.

    A missing declaration is an ERROR, never an empty contract: an empty consumers
    list reads as "this engine specializes on nothing", which is the claim under
    which an unverifiable rocKE bundle would sail through a full-mode check.
    """
    declaration = config.specialization
    if not declaration:
        raise ValueError(
            f"engine {config.engine.name!r} emits descriptors but declares no "
            f"top-level 'specialization' block, so nothing here states which "
            f"metadata fields the producing compiler specialized on. A UKD without "
            f"provenance.specialization_contract cannot be checked against the "
            f"builder it was compiled from -- on the receiving machine there is no "
            f"builder to ask. Declare the partition: 'metadata_fields' with a "
            f"'bindings' entry each for the fields the builder consumes, and "
            f"'matcher_only_fields' for the rest."
        )
    return {
        "schema_version": 1,
        "consumers": [
            {
                "engine_id": ids["ued"],
                "kmd_id": ids["kmd"],
                "metadata_fields": list(declaration.get("metadata_fields") or []),
                "matcher_only_fields": list(
                    declaration.get("matcher_only_fields") or []
                ),
                "bindings": {
                    name: dict(binding)
                    for name, binding in (declaration.get("bindings") or {}).items()
                },
                "vocabulary": {
                    name: dict(spellings)
                    for name, spellings in (declaration.get("vocabulary") or {}).items()
                },
            }
        ],
    }


def build_kdp(
    config: IngestorConfig,
    pack: PackSpec,
    ids: dict,
    seen_metadata: dict | None = None,
) -> dict:
    """One pack's KDP.

    ``seen_metadata`` carries de-duplication state ACROSS the packs of one engine.
    Pass the same dict to every ``build_kdp`` call for a config; omit it and each
    pack de-duplicates against itself alone, which is the behaviour that ships a
    duplicate.

    Why engine-wide and not per-pack: the loader collects packs into one
    ``DescriptorSet`` BY ENGINE ID, so two entries with identical matcher-visible
    metadata are one candidate to the runtime whichever pack produced them -- a
    duplicate CATALOG TUPLE, which drops the whole engine at load, not one entry.

    A coverage gap is therefore served by widening the union rather than by shipping
    a second bundle beside the first: native symbol names are derived from the engine
    name, so a new slug mints a new engine that registers, loads, and matches nothing.
    """
    pack_index = _pack_index(config, pack)
    matchers = [ids["kernel_match"]]
    if config.is_multi_pack:
        matchers.insert(0, ids[("operation_umd", pack_index)])
    # Several generation expressions may target one engine and are EXPECTED to
    # overlap -- an author writes "the model-trace shapes" and "the published-sweep
    # shapes" without hand-partitioning them.
    #
    # Three outcomes, not one, because a shared tuple means three different things:
    #
    #   * DISJOINT arch coverage -- not duplicates at all. Two devices, two
    #     binaries; dropping either leaves a coverage hole no count reconciles.
    #   * OVERLAPPING coverage, same candidate -- one entry, de-duplicated here.
    #   * OVERLAPPING coverage, DIFFERENT candidate -- refused; see
    #     `_candidate_identity`.
    #   * OVERLAPPING but UNEQUAL coverage, SAME candidate -- also refused, with its
    #     own diagnostic: nothing about the candidate differs, and de-duplicating
    #     would silently widen one entry's coverage to devices its author never
    #     claimed.
    kernel_descriptors = []
    if seen_metadata is None:
        seen_metadata = {}
    duplicates: list = []
    contract = build_specialization_contract(config, ids)
    for index, kernel in enumerate(pack.kernels):
        # Resolve FIRST, then key on the resolved form: the dedup key and the emitted
        # document are derived from the same values, so they cannot drift apart.
        metadata = _resolved_metadata(kernel, config)
        _check_metadata_resolved(kernel, metadata, config)
        key = _dedup_key(metadata, config)
        # A kernel stating no arch inherits its pack's, which is the KDP convention
        # the loader reads. Comparing the authored (often empty) list instead would
        # make every kernel of an arch-scoped pack look like a wildcard.
        arch = list(kernel.arch or pack.arch)
        identity = _candidate_identity(kernel)
        already = None
        for prior in seen_metadata.setdefault(key, []):
            if not _arch_overlaps(arch, prior["arch"]):
                continue
            if prior["identity"] == identity:
                if sorted(prior["arch"]) == sorted(arch):
                    already = prior
                    break
                raise ValueError(
                    f"kernel {kernel.name!r} (pack {pack.name!r}) and kernel "
                    f"{prior['name']!r} (pack {prior['pack']!r}) are the SAME "
                    f"candidate and complete to the SAME catalog tuple "
                    f"{json.loads(key)}, but their architectures overlap without "
                    f"being equal ({arch or ['<any>']} vs "
                    f"{prior['arch'] or ['<any>']}). On the shared architectures "
                    f"the matcher would see one tuple twice, which drops the whole "
                    f"engine at load; coalescing them would instead advertise one "
                    f"of the two on devices its arch list never claimed. Give the "
                    f"two entries the SAME arch list so they de-duplicate, or make "
                    f"them disjoint."
                )
            raise ValueError(
                f"kernel {kernel.name!r} (pack {pack.name!r}) and kernel "
                f"{prior['name']!r} (pack {prior['pack']!r}) complete to the SAME "
                f"catalog tuple {json.loads(key)} on overlapping architectures "
                f"({arch or ['<any>']} vs {prior['arch'] or ['<any>']}), but they "
                f"are not the same candidate: their kernel_source/priority differ, "
                f"so they name different binaries. The matcher compares the tuple "
                f"and would see one entry twice -- a duplicate tuple drops the "
                f"whole engine at load, and dropping one of them here would "
                f"discard a binary that was deliberately built. Distinguish them "
                f"in metadata (a knob the spec pins belongs in the tuple), narrow "
                f"one of the arch lists, or do not ship both."
            )
        if already is not None:
            duplicates.append((kernel.name, already["name"]))
            continue
        seen_metadata[key].append(
            {
                "name": kernel.name,
                "pack": pack.name,
                "arch": arch,
                "identity": identity,
            }
        )
        entry = {
            "version": "1.0",
            "id": ids[("kernel", pack_index, index)],
            "name": kernel.name,
            # Per-kind keys, never the union: the runtime loader hard-fails an
            # unknown key and hkp_pack validates a closed set per kind.
            "kernel_source": kernel.kernel_source.as_document(),
            "metadata": metadata,
            "priority": kernel.priority,
        }
        if kernel.arch:
            entry["arch"] = list(kernel.arch)
        kernel_descriptors.append(entry)
    if duplicates:
        shown = ", ".join(f"{d} == {k}" for d, k in duplicates[:3])
        more = f" (+{len(duplicates) - 3} more)" if len(duplicates) > 3 else ""
        print(
            f"  pack '{pack.name}': dropped {len(duplicates)} duplicate "
            f"variant(s) with metadata already emitted: {shown}{more}"
        )
    kdp = {
        "version": "1.0",
        "id": ids[("pack", pack_index)],
        "name": f"{config.engine.namespace}:{config.kdp_stem(pack)}",
        "matchers": matchers,
        "engine": ids["ued"],
        "dispatch": ids["udd"],
        # Declared ONCE for the whole pack, and emitted AFTER minting so the ids in
        # it are this bundle's real ids. Every kernel below shares one engine, one
        # KMD and one field partition, so per-kernel copies would be identical
        # boilerplate -- 2.8x the size on a 2733-kernel pack. A reader resolves a
        # kernel's own declaration first and this one second
        # (``hkp_pack.agreement.resolved_contract``), so a kernel needing different
        # terms can still state them.
        "provenance": {"specialization_contract": contract},
        "kernelDescriptors": kernel_descriptors,
    }
    if pack.arch:
        kdp["arch"] = list(pack.arch)
    elif config.is_packaged:
        # hkp_pack REQUIRES arch on a KDP (_validate_kdp), unlike the runtime
        # loader, which treats absence as a wildcard. The config loader rejects
        # this earlier with a better message; reaching here means that check was
        # bypassed, so keep the key present and empty and let the packager's own
        # diagnostic be the one the author sees.
        kdp["arch"] = []
    return kdp


def build_kdp_documents(config: IngestorConfig, ids: dict) -> list:
    """Every pack's KDP, ``[(pack, document), ...]``, de-duplicated ENGINE-WIDE.

    The one de-duplication scope, in one place: building the packs separately,
    each against its own fresh state, is exactly the arrangement that ships a
    duplicate. See `build_kdp`.
    """
    seen_metadata: dict = {}
    return [
        (pack, build_kdp(config, pack, ids, seen_metadata)) for pack in config.packs
    ]


#: Inventory key for descriptors that name no architecture at all.
#:
#: An absent ``arch`` is the loader's WILDCARD -- the descriptor ships for every
#: device -- so it cannot be filed under a concrete id, and dropping it from the
#: per-arch view would understate what ships by exactly the entries nobody
#: restricted.
ARCH_WILDCARD = "*"


def emitted_inventory(config: IngestorConfig, kdp_documents: list) -> dict:
    """What this bundle ACTUALLY SHIPS, keyed by architecture.

    Built from the FINALIZED KDP documents -- after metadata resolution, the
    resolved-knob check and the engine-wide de-duplication -- never from the config.
    Those three passes are exactly where the authored count and the emitted count
    diverge, and a census keyed on the authored number reports a shortfall on every
    run and trains its reader to ignore it.

    ``source_kind`` is what the RUNTIME will see, not what was authored: a packaged
    bundle is lowered to ``kpack`` by hkp_pack before the loader reads it.

    A descriptor stating no ``arch`` of its own is filed under its pack's, because
    that is the coverage the loader gives it; a pack stating none either is filed
    under `ARCH_WILDCARD`.

    ``kdp_documents`` is `build_kdp_documents`' ``[(pack, document), ...]``; the
    pack carries the stem the file is named for, which the document does not.

    A descriptor is filed under every arch it names, and each arch reports its
    distinct names, so an engine contributes one entry per arch it covers.
    """
    arches: dict[str, dict] = {}

    def bucket(arch: str) -> dict:
        return arches.setdefault(arch, {"descriptors": [], "pack_names": set()})

    total = 0
    for pack, document in kdp_documents:
        pack_arch = list(document.get("arch") or []) or [ARCH_WILDCARD]
        stem = config.kdp_stem(pack)
        for arch in pack_arch:
            bucket(arch)["pack_names"].add(stem)
        for descriptor in document["kernelDescriptors"]:
            total += 1
            for arch in list(descriptor.get("arch") or []) or pack_arch:
                bucket(arch)["descriptors"].append(descriptor["name"])

    return {
        "sdk_version": config.engine.sdk_version,
        "source_kind": (
            KERNEL_SOURCE_KIND_KPACK
            if config.is_packaged
            else config.kernel_source_kind
        ),
        # Both the list and the count come from the distinct names, matching the
        # std::set the generated census loads them into. Two descriptors of one
        # engine cannot share a name -- the loader rejects that engine-wide -- so
        # the set collapses nothing a bundle can actually ship.
        "arches": {
            arch: {
                "descriptor_names": sorted(set(entry["descriptors"])),
                "descriptor_count": len(set(entry["descriptors"])),
                "pack_names": sorted(entry["pack_names"]),
                "pack_count": len(entry["pack_names"]),
            }
            for arch, entry in sorted(arches.items())
        },
        "total_descriptor_count": total,
    }


class IngestorGenerator:
    """Renders every file of one engine's descriptor bundle for a given
    :class:`IngestorConfig`, writing into ``output_dir``."""

    def __init__(self, template_dir: Path):
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
            # An unset UUID cross-reference must fail loudly at generation time,
            # not render "" and fail later at the loader with a message about an
            # empty string instead of a missing field.
            undefined=StrictUndefined,
        )

    def preview_files(self, config: IngestorConfig) -> list[str]:
        """The file list :meth:`render` would write, without writing anything."""
        slug = config.engine.slug
        ddir = config.descriptor_dir
        files = [
            f"{ddir}/{slug}.kmd.json",
            f"{ddir}/{slug}.ued.json",
            f"{ddir}/{slug}.udd.json",
        ]
        if config.engine.has_heuristic:
            files.append(f"{ddir}/{slug}.uhd.json")
        files.append(f"{ddir}/kernel_dtype_matches_graph.umd.json")
        for pack in config.packs:
            files.append(f"{ddir}/{config.kdp_stem(pack)}.kdp.json")
            if config.is_multi_pack:
                files.append(f"{ddir}/operation_is_{pack.discriminator}.umd.json")
        files.append(f"packs/{config.native_class_name}Native.cpp")
        files.append(f"tests/Test{config.engine.pascal_name}Packs.cpp")
        files.append(f"tests/Test{config.engine.pascal_name}Matchers.cpp")
        for fragment in FRAGMENT_FILENAMES:
            files.append(f"fragments/{fragment}")
        return files

    def render(self, config: IngestorConfig, output_dir: Path) -> list[str]:
        """Mint ids, write every descriptor JSON, the native/test C++ stubs,
        and the five CMake/registration fragments. Returns the list of
        relative paths written.

        Every KDP is BUILT before anything is rendered, because the templates are
        given the emitted inventory (`emitted_inventory`) and that view is only
        truthful once the engine-wide de-duplication has run over all of them.
        """
        ids = mint_ids(config)
        written: list[str] = []
        slug = config.engine.slug
        ddir = config.descriptor_dir
        (output_dir / ddir).mkdir(parents=True, exist_ok=True)

        def write_json(rel: str, obj: dict) -> None:
            path = output_dir / rel
            path.write_text(_dump(obj), encoding="utf-8")
            written.append(rel)

        kdp_documents = build_kdp_documents(config, ids)
        emitted = emitted_inventory(config, kdp_documents)

        write_json(f"{ddir}/{slug}.kmd.json", build_kmd(config, ids))
        write_json(f"{ddir}/{slug}.ued.json", build_ued(config, ids))
        write_json(f"{ddir}/{slug}.udd.json", build_udd(config, ids))
        uhd = build_uhd(config, ids)
        if uhd is not None:
            write_json(f"{ddir}/{slug}.uhd.json", uhd)
        write_json(
            f"{ddir}/kernel_dtype_matches_graph.umd.json",
            build_kernel_match_umd(config, ids),
        )
        for pack, document in kdp_documents:
            write_json(f"{ddir}/{config.kdp_stem(pack)}.kdp.json", document)
            op_umd = build_operation_umd(config, pack, ids)
            if op_umd is not None:
                write_json(
                    f"{ddir}/operation_is_{pack.discriminator}.umd.json",
                    op_umd,
                )

        # --- C++ stubs/tests ---
        packs_dir = output_dir / "packs"
        packs_dir.mkdir(parents=True, exist_ok=True)
        tests_dir = output_dir / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)

        native_rel = f"packs/{config.native_class_name}Native.cpp"
        (output_dir / native_rel).write_text(
            self._render_template("native.cpp.j2", config, ids=ids, emitted=emitted),
            encoding="utf-8",
        )
        written.append(native_rel)

        packs_test_rel = f"tests/Test{config.engine.pascal_name}Packs.cpp"
        (output_dir / packs_test_rel).write_text(
            self._render_template(
                "test_packs.cpp.j2", config, ids=ids, emitted=emitted
            ),
            encoding="utf-8",
        )
        written.append(packs_test_rel)

        matchers_test_rel = f"tests/Test{config.engine.pascal_name}Matchers.cpp"
        (output_dir / matchers_test_rel).write_text(
            self._render_template(
                "test_matchers.cpp.j2", config, ids=ids, emitted=emitted
            ),
            encoding="utf-8",
        )
        written.append(matchers_test_rel)

        # --- fragments ---
        fragments_dir = output_dir / "fragments"
        fragments_dir.mkdir(parents=True, exist_ok=True)
        for template_name, out_name in FRAGMENT_TEMPLATES:
            content = self._render_template(
                template_name, config, ids=ids, emitted=emitted
            )
            (fragments_dir / out_name).write_text(content, encoding="utf-8")
            written.append(f"fragments/{out_name}")

        return written

    #: Emitted files that are splice INSTRUCTIONS, not shipped source. They are
    #: pasted into existing files by hand and never exist as files in the tree,
    #: so they are excluded from the located/missing accounting.
    _NON_SHIPPED_PREFIXES = ("fragments/",)

    #: Where each emitted directory is SPLICED TO, relative to a root it was spliced
    #: into. Derived from this generator's own CMake fragments, which are the
    #: instructions the splice follows: ``packs/`` keeps its name per
    #: ``cmake_target_sources``, and ``tests/`` nests under ``packs/`` per
    #: ``cmake_test_sources``. Descriptors are handled in `_accepted_destinations`.
    _SPLICE_DESTINATIONS: dict[str, tuple[str, ...]] = {
        "packs": ("packs",),
        "tests": ("tests", "packs"),
    }

    @classmethod
    def _accepted_destinations(cls, rel: str) -> tuple[str, ...]:
        """The relative paths ``rel`` may legitimately have been spliced to.

        Every one of them keeps the ENGINE-SPECIFIC component of the emitted path
        -- the descriptor subpath, or a filename built from the engine's class or
        Pascal name -- because that is what makes a match evidence about THIS
        engine.
        """
        head, _, tail = rel.partition("/")
        if head in ("descriptors", "test_descriptors"):
            # A root pointed AT the `descriptors/` or `test_descriptors/` tree sees
            # the subpath alone, while a root above it sees the whole thing. Both
            # spellings keep the authored subpath and the engine's own directory.
            return (rel, tail)
        return tuple(
            f"{destination}/{tail}"
            for destination in cls._SPLICE_DESTINATIONS.get(head, (head,))
        )

    @classmethod
    def locate_emitted(
        cls, roots: list[Path], written: list[str]
    ) -> tuple[dict[str, Path], list[str], dict[str, list[Path]]]:
        """``({relative path: real path}, [not found], {relative path: [ambiguous]})``
        for the shippable files in ``written``, searched across ``roots``.

        Not ``root / rel``, because the provider splits this tool's flat ``packs/``
        + ``tests/`` layout: packs land in the engine directory and the test stubs
        under ``src/tests/engines/.../packs/``, as ``cmake_test_sources`` instructs.
        That split is why ``roots`` is a LIST -- the two trees may have no common
        ancestor worth scanning, and widening the root until they do drags in build
        trees and stale copies.

        Not by basename either: any file sharing a name would satisfy a missing
        target and manufacture ambiguity against some other engine's files. A hit
        must be at the engine-specific SPLICED RELATIVE PATH this generator's own
        fragments name (see `_accepted_destinations`).

        Two matches for one relative path is an ERROR, not a pick, and a
        non-existent root is a hard error: picking by directory order lets a filled
        stale copy report green, and zero hits with no complaint is a gate that
        passes because it looked nowhere.
        """
        roots = [Path(root) for root in roots]
        if not roots:
            raise ValueError(
                "locate_emitted needs at least one root to search; an empty root "
                "list finds nothing and would report every file missing."
            )
        absent = [str(root) for root in roots if not root.is_dir()]
        if absent:
            raise ValueError(
                f"emitted root(s) {absent} do not exist (or are not directories). "
                f"A root that cannot be read contributes no hits, so the scan would "
                f"report the files it should have found there as unfilled-free "
                f"simply by never seeing them."
            )
        shippable = [
            rel for rel in written if not rel.startswith(cls._NON_SHIPPED_PREFIXES)
        ]
        # Index by basename first: the suffix comparison below is the real test,
        # but only files that could possibly match need to reach it, and a spliced
        # provider tree is large.
        by_name: dict[str, list[Path]] = {}
        seen_paths: set = set()
        for root in roots:
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                # Two roots may nest. The same file reached twice is one file, not
                # an ambiguity -- reporting it as one would fail a correct tree.
                resolved = path.resolve()
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)
                by_name.setdefault(path.name, []).append(path)
        hits: dict[str, list[Path]] = {}
        for rel in shippable:
            accepted = cls._accepted_destinations(rel)
            matched = [
                path
                for path in by_name.get(Path(rel).name, [])
                if any(
                    path.as_posix() == candidate
                    or path.as_posix().endswith("/" + candidate)
                    for candidate in accepted
                )
            ]
            if matched:
                hits[rel] = matched
        found = {rel: paths[0] for rel, paths in hits.items() if len(paths) == 1}
        ambiguous = {rel: paths for rel, paths in hits.items() if len(paths) > 1}
        missing = [rel for rel in shippable if rel not in hits]
        return found, missing, ambiguous

    @classmethod
    def unfilled_placeholders(
        cls, roots: list[Path], written: list[str]
    ) -> dict[str, int]:
        """``{relative path: placeholder count}`` for every LOCATED file that
        still carries an unfilled stub marker, worst first.

        Lives here because this object is the only one that knows the full emitted
        set. A hand-written ``grep -c`` over a transcribed glob drifts the moment
        that set changes; the generator cannot fall behind itself.

        An empty result means "nothing unfilled AMONG THE FILES FOUND", never
        "nothing left to do": a file that was not located is counted by neither
        this nor any caller reading only this. The gate therefore reads
        `locate_emitted`'s missing/ambiguous lists too, and fails on them.
        """
        located, _missing, _ambiguous = cls.locate_emitted(roots, written)
        counts: dict[str, int] = {}
        for rel, path in located.items():
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            n = text.count(PLACEHOLDER_MARKER)
            if n:
                counts[rel] = n
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def _render_template(
        self, template_name: str, config: IngestorConfig, **extra
    ) -> str:
        """Render one template against the standard context.

        Every template sees ``config``, plus ``ids`` and ``emitted`` threaded
        through ``extra``. A caller that renders one template on its own gets
        ``emitted`` computed here rather than left undefined, because
        ``StrictUndefined`` would otherwise raise about a missing name rather than
        about the caller that skipped a step.
        """
        if "emitted" not in extra:
            ids = extra.setdefault("ids", mint_ids(config))
            extra["emitted"] = emitted_inventory(
                config, build_kdp_documents(config, ids)
            )
        try:
            template = self.env.get_template(template_name)
            return template.render(config=config, **extra)
        except Exception as e:
            raise RuntimeError(
                f"Failed to render template '{template_name}' for engine "
                f"'{config.engine.name}': {e}"
            ) from e


#: The marker every unfilled stub body carries. Templates emit it; the reader
#: replaces it. One spelling, defined once, so a scan cannot look for a string
#: the templates stopped writing.
PLACEHOLDER_MARKER = "FILL THIS OUT"

FRAGMENT_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("fragments/cmake_descriptor_files.j2", "cmake_descriptor_files.txt"),
    ("fragments/cmake_target_sources.j2", "cmake_target_sources.txt"),
    ("fragments/cmake_test_sources.j2", "cmake_test_sources.txt"),
    ("fragments/ingestor_packs_hpp.j2", "ingestor_packs.hpp.txt"),
    ("fragments/ingestor_packs_cpp.j2", "ingestor_packs.cpp.txt"),
)
FRAGMENT_FILENAMES: tuple[str, ...] = tuple(name for _, name in FRAGMENT_TEMPLATES)
