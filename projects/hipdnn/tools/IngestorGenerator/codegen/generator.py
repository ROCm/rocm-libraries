# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Renders a full descriptor bundle for one ``IngestorConfig``.

Descriptor JSON is built as dicts and serialized with ``json.dumps``; Jinja2
renders only the C++ stubs and the CMake/registration fragments. UUIDs are
minted once per run in :func:`mint_ids`.
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

#: Header every emitted C++/CMake file opens with. It contains ©, so emitted
#: files are written as UTF-8.
CPP_COPYRIGHT_HEADER = (
    "// Copyright \u00a9 Advanced Micro Devices, Inc., or its affiliates.\n"
    "// SPDX-License-Identifier:  MIT\n"
)
CMAKE_COPYRIGHT_HEADER = (
    "# Copyright \u00a9 Advanced Micro Devices, Inc., or its affiliates.\n"
    "# SPDX-License-Identifier:  MIT\n"
)


_CPP_NAMED_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}


def cpp_escape(value) -> str:
    """Escape ``value`` for use inside a C++ string literal, without the
    surrounding quotes.

    Control characters use three-digit octal: C++ hex escapes are
    maximal-munch, so ``"\\x1f32"`` would absorb the following text.
    """
    text = value if isinstance(value, str) else str(value)
    out = []
    for character in text:
        escape = _CPP_NAMED_ESCAPES.get(character)
        if escape is not None:
            out.append(escape)
        elif ord(character) < 0x20 or ord(character) == 0x7F:
            out.append(f"\\{ord(character):03o}")
        else:
            out.append(character)
    return "".join(out)


def mint_ids(config: IngestorConfig) -> dict:
    """Generate random UUIDs once per bundle, indexed by pack/kernel position
    so duplicate names remain distinct."""
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
    """The engine descriptor. ``sdk_version`` is always emitted, including the
    ``"1.0.0"`` baseline, because ``GenericPlanBuilder::understandsGraph()``
    gates on it at match time."""
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
    """The shared kernel-scoped dtype matcher: one per engine, referenced by
    every pack's KDP. Expresses the per-kernel applicability check the UED's
    graph_match cannot, having no kernel in scope."""
    return {
        "version": "1.0",
        "id": ids["kernel_match"],
        "name": "kernel dtype matches the graph's dtype",
        "scope": "kernel",
        "match_symbol": config.kernel_match_symbol,
    }


#: KMD sentinel meaning "not set; the kernel's own policy decides". Never
#: legal in an emitted descriptor.
UNSET_SENTINEL = -1


def _canonical_metadata_value(value, kind: str | None):
    """Canonicalize ``value`` for a KMD field declared ``kind``: ``bool`` ->
    ``int`` for an ``int`` field, ``int`` -> ``float`` for a ``float`` field,
    anything else unchanged. Mirrors ``hkp_pack.agreement.canonical``."""
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
    """Metadata for one kernel, with unresolved knobs filled from the spec that
    built it.

    A tri-state knob spans three layers: ``kernel_source.spec[k]`` decides the
    binary (absent = the kernel's policy decides at build time), ``metadata[k]``
    is the catalog key the matcher compares, and the KMD ``default_value``
    fills anything absent at load.

    An authored metadata value is never overwritten, since the layers use
    different spellings (matcher ``"BF16"``, builder ``"bf16"``).
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
            # Authored in the vocabulary the matcher reads; leave it.
            continue
        if name in spec and spec[name] is not None:
            # Unresolved in metadata but pinned in the spec: state the
            # definite binary value rather than ship "undecided".
            out[name] = _canonical_metadata_value(spec[name], field_spec.type)
    return out


def _check_metadata_resolved(
    kernel: KernelSpec, metadata: dict, config: IngestorConfig
) -> None:
    """Refuse a descriptor whose knob does not describe its binary: a stated
    ``-1`` sentinel, or an optional field absent from both ``metadata`` and
    ``kernel_source.spec``.

    An absent knob takes the KMD ``default_value`` as its catalog key while the
    binary was built from the builder dataclass's default. Mandatory fields are
    left to the config loader.
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


#: What each declared KMD type accepts, mirroring ``coerceToDeclaredType``
#: (``DescriptorLoader.hpp``): the JSON kind must be the declared type, with
#: one widening, an integer into a ``float`` field. A JSON ``bool`` is not an
#: ``int``; `_canonical_metadata_value` has already projected that case.
_METADATA_TYPE_ACCEPTS = {
    "bool": lambda v: isinstance(v, bool),
    "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "float": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "string": lambda v: isinstance(v, str),
    "int_list": lambda v: isinstance(v, list)
    and all(isinstance(item, int) and not isinstance(item, bool) for item in v),
}


def _check_metadata_types(
    kernel: KernelSpec, metadata: dict, config: IngestorConfig
) -> None:
    """Type-check metadata as projected, not as authored: `_resolved_metadata`
    projects spec values in after the config loader's own check.

    At load, ``coerceKernelMetadata`` refuses a mismatch and drops the whole
    pack. Fields the KMD does not declare are left to the config loader.
    """
    declared = {f.name: f.type for f in config.kmd_fields}
    for name, value in sorted(metadata.items()):
        accepts = _METADATA_TYPE_ACCEPTS.get(declared.get(name))
        if accepts is None or accepts(value):
            continue
        raise ValueError(
            f"kernel {kernel.name!r} resolves metadata field '{name}' to {value!r}, "
            f"which its kmd_fields entry declares as type '{declared[name]}'. The "
            f"value is projected from kernel_source.spec after the config loader's "
            f"own type check has run, so nothing before this point compared it with "
            f"the declared type. At load, coerceKernelMetadata refuses the mismatch "
            f"and drops the whole pack. Pin the knob in the spec using the field's "
            f"declared type, or state the matcher-visible value in metadata."
        )


def _completed_metadata(metadata: dict, config: IngestorConfig) -> dict:
    """The metadata tuple as the loader completes it, substituting each absent
    field's KMD ``default_value``, so descriptors differing only in which layer
    stated a value collide here as they do at runtime.

    A mandatory field the config omitted has no default and is left out.
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
    """Identity of a descriptor as the matcher sees it: the completed tuple,
    not the emitted document.

    Architecture is not part of this key; `build_kdp` decides overlap against
    the arch coverage recorded beside each key.
    """
    return json.dumps(_completed_metadata(metadata, config), sort_keys=True)


def _candidate_identity(kernel: KernelSpec) -> str:
    """What a candidate is beyond its tuple: two entries sharing a completed
    tuple are the same candidate only if they name the same binary at the same
    priority. A knob absent from ``kernel_source.spec`` and one pinned to the
    value the policy chose are different binaries under one tuple."""
    return json.dumps(
        {
            "kernel_source": kernel.kernel_source.as_document(),
            "priority": kernel.priority,
        },
        sort_keys=True,
    )


def _arch_overlaps(left: list, right: list) -> bool:
    """Whether two arch coverages can select on the same device. An empty list
    is the loader's wildcard and overlaps everything. Mirrors
    ``hkp_pack.agreement.overlap``."""
    return not left or not right or bool(set(left) & set(right))


def _pack_index(config: IngestorConfig, pack: PackSpec) -> int:
    """This pack's position in the config, by identity: pack names are not
    guaranteed unique and two packs may compare equal by value."""
    for index, candidate in enumerate(config.packs):
        if candidate is pack:
            return index
    raise ValueError(f"pack {pack.name!r} is not part of this config")


def build_operation_umd(
    config: IngestorConfig, pack: PackSpec, ids: dict
) -> dict | None:
    """UMD policy: emitted only for genuine per-pack narrowing, i.e. only when
    the engine has more than one pack. A single-pack engine gets zero
    graph-scoped UMDs."""
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

    Self-contained data, so a bundle can be checked without rocKE installed.
    ``engine_id`` and ``kmd_id`` come from `mint_ids`; the rest is authored.
    A missing declaration is an error, never an empty contract.
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

    ``seen_metadata`` carries de-duplication state across the packs of one
    engine; pass the same dict to every ``build_kdp`` call for a config.
    De-duplication is engine-wide because the loader collects packs into one
    ``DescriptorSet`` by engine id, and a duplicate tuple drops the engine at
    load. Widen this engine's union to close a coverage gap; a new slug mints
    a different engine.
    """
    pack_index = _pack_index(config, pack)
    matchers = [ids["kernel_match"]]
    if config.is_multi_pack:
        matchers.insert(0, ids[("operation_umd", pack_index)])
    # Generation expressions targeting one engine may overlap, so a shared
    # tuple has four outcomes: disjoint arch coverage is not a duplicate;
    # overlapping coverage with the same candidate and equal arch de-duplicates
    # here; a different candidate, or unequal coverage, is refused.
    kernel_descriptors = []
    if seen_metadata is None:
        seen_metadata = {}
    duplicates: list = []
    contract = build_specialization_contract(config, ids)
    for index, kernel in enumerate(pack.kernels):
        # Resolve first, then key on the resolved form, so the dedup key and
        # the emitted document derive from the same values.
        metadata = _resolved_metadata(kernel, config)
        # Before the type check: the unset sentinel is an int, so a type check
        # would report "-1 is not a string" for a knob nobody decided.
        _check_metadata_resolved(kernel, metadata, config)
        _check_metadata_types(kernel, metadata, config)
        key = _dedup_key(metadata, config)
        # A kernel stating no arch inherits its pack's, per the KDP convention
        # the loader reads; comparing the authored list would make every kernel
        # of an arch-scoped pack look like a wildcard.
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
            # Carry the prior kernel's own pack: a drop that empties this pack
            # is reported against the pack that absorbed it.
            duplicates.append((kernel.name, already["name"], already["pack"]))
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
        shown = ", ".join(f"{d} == {k}" for d, k, _p in duplicates[:3])
        more = f" (+{len(duplicates) - 3} more)" if len(duplicates) > 3 else ""
        print(
            f"  pack '{pack.name}': dropped {len(duplicates)} duplicate "
            f"variant(s) with metadata already emitted: {shown}{more}"
        )
    # The descriptor list is final here: after resolution, the metadata checks
    # and the engine-wide de-duplication. The loader drops a KDP with no
    # kernels while the census counts every KDP written, so an empty pack is
    # refused rather than silently dropped.
    if not kernel_descriptors:
        if duplicates:
            absorbed = ", ".join(
                f"{name} == {prior} (pack '{prior_pack}')"
                for name, prior, prior_pack in duplicates
            )
            absorbing = sorted({prior_pack for _n, _p, prior_pack in duplicates})
            cause = (
                f"all {len(duplicates)} of its kernels de-duplicated against "
                f"kernels already emitted by pack(s) {absorbing}: {absorbed}"
            )
        else:
            cause = "it contributed no kernels at all"
        raise ValueError(
            f"pack {pack.name!r} would ship a KDP with ZERO kernel descriptors: "
            f"{cause}. The loader drops a pack that declares no kernels, so the "
            f"emitted census would assert a pack the runtime never holds and the "
            f"generated test would fail on this bundle. Give the pack at least one "
            f"kernel the rest of the engine does not already emit, or do not "
            f"declare it."
        )
    kdp = {
        "version": "1.0",
        "id": ids[("pack", pack_index)],
        "name": f"{config.engine.namespace}:{config.kdp_stem(pack)}",
        "matchers": matchers,
        "engine": ids["ued"],
        "dispatch": ids["udd"],
        # Declared once per pack, after minting, since every kernel shares one
        # engine, KMD and field partition. A reader resolves a kernel's own
        # declaration first (``hkp_pack.agreement.resolved_contract``), so a
        # kernel needing different terms can still state them.
        "provenance": {"specialization_contract": contract},
        "kernelDescriptors": kernel_descriptors,
    }
    if pack.arch:
        kdp["arch"] = list(pack.arch)
    elif config.is_packaged:
        # hkp_pack requires arch on a KDP (_validate_kdp) where the runtime
        # loader treats absence as a wildcard. The config loader rejects this
        # earlier, so keep the key present and empty.
        kdp["arch"] = []
    return kdp


def build_kdp_documents(config: IngestorConfig, ids: dict) -> list:
    """Every pack's KDP, ``[(pack, document), ...]``, de-duplicated engine-wide.
    The single de-duplication scope; see `build_kdp`."""
    seen_metadata: dict = {}
    return [
        (pack, build_kdp(config, pack, ids, seen_metadata)) for pack in config.packs
    ]


#: Inventory key for descriptors naming no architecture. An absent ``arch`` is
#: the loader's wildcard, so it cannot be filed under a concrete id.
ARCH_WILDCARD = "*"


def emitted_inventory(config: IngestorConfig, kdp_documents: list) -> dict:
    """What this bundle ships, keyed by architecture.

    Built from the finalized KDP documents, never from the config, since
    resolution, the resolved-knob check and de-duplication all change the
    count. ``source_kind`` is what the runtime sees: a packaged bundle is
    lowered to ``kpack`` first. A descriptor with no ``arch`` is filed under
    its pack's; a pack with none under `ARCH_WILDCARD`.

    Each concrete arch row is unioned with the wildcard row, one way only,
    because a wildcard entry ships on that device too.
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

    wildcard = arches.get(ARCH_WILDCARD)
    if wildcard is not None:
        for arch, entry in arches.items():
            if arch == ARCH_WILDCARD:
                continue
            entry["descriptors"].extend(wildcard["descriptors"])
            entry["pack_names"].update(wildcard["pack_names"])

    return {
        "sdk_version": config.engine.sdk_version,
        "source_kind": (
            KERNEL_SOURCE_KIND_KPACK
            if config.is_packaged
            else config.kernel_source_kind
        ),
        # List and count both come from the distinct names, matching the
        # std::set the generated census loads them into. Two descriptors of one
        # engine cannot share a name, so the set collapses nothing shippable.
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
            # An unset UUID cross-reference fails loudly at generation time
            # rather than rendering "" and failing later at the loader.
            undefined=StrictUndefined,
        )
        # JSON is serialized by `json.dumps`, never by a template, so there is
        # no counterpart filter.
        self.env.filters["cpp_escape"] = cpp_escape

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
        """Mint ids, write every descriptor JSON, the native/test C++ stubs and
        the CMake/registration fragments. Returns the relative paths written.

        Every KDP is built before rendering, since the templates are given the
        emitted inventory and that view needs the engine-wide de-duplication.
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

    #: Emitted files that are splice instructions, not shipped source. They are
    #: pasted into existing files by hand, so they are excluded from the
    #: located/missing accounting.
    _NON_SHIPPED_PREFIXES = ("fragments/",)

    #: Where each emitted directory is spliced to, per this generator's CMake
    #: fragments: ``packs/`` keeps its name (``cmake_target_sources``),
    #: ``tests/`` nests under ``packs/`` (``cmake_test_sources``).
    _SPLICE_DESTINATIONS: dict[str, tuple[str, ...]] = {
        "packs": ("packs",),
        "tests": ("tests", "packs"),
    }

    @classmethod
    def _accepted_destinations(cls, rel: str) -> tuple[str, ...]:
        """The relative paths ``rel`` may legitimately have been spliced to.
        Each keeps the engine-specific component of the emitted path, so a
        match is evidence about this engine."""
        head, _, tail = rel.partition("/")
        if head in ("descriptors", "test_descriptors"):
            # A root pointed at the ``descriptors/``/``test_descriptors/`` tree
            # sees the subpath alone; a root above it sees the whole path.
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

        ``roots`` is a list because the provider splits this tool's flat
        ``packs/`` + ``tests/`` layout, test stubs landing under
        ``src/tests/engines/.../packs/`` as ``cmake_test_sources`` instructs.
        A hit must sit at an engine-specific spliced path (see
        `_accepted_destinations`), never merely share a basename.

        Two matches for one relative path, or a non-existent root, is an error.
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
        # Index by basename first: the suffix comparison below is the real
        # test, and a spliced provider tree is large.
        by_name: dict[str, list[Path]] = {}
        seen_paths: set = set()
        for root in roots:
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                # Two roots may nest; the same file reached twice is one file,
                # not an ambiguity.
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
        """``{relative path: placeholder count}`` for every located file that
        still carries an unfilled stub marker, worst first.

        Counts only located files, so the gate also fails on `locate_emitted`'s
        missing/ambiguous lists. An unreadable located file is an error naming
        every such path.
        """
        located, _missing, _ambiguous = cls.locate_emitted(roots, written)
        counts: dict[str, int] = {}
        unreadable: list[str] = []
        for rel, path in located.items():
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError) as e:
                unreadable.append(f"{rel} (at {path}): {e}")
                continue
            n = text.count(PLACEHOLDER_MARKER)
            if n:
                counts[rel] = n
        if unreadable:
            listed = "; ".join(sorted(unreadable))
            raise ValueError(
                f"{len(unreadable)} located file(s) could not be read, so the "
                f"placeholder scan did not cover them: {listed}. A file the scan "
                f"skipped reports exactly as a file it read and found clean, which "
                f"is the false green this gate exists to prevent. Emitted files are "
                f"written UTF-8; re-generate the bundle, or fix the copy that is "
                f"not."
            )
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def _render_template(
        self, template_name: str, config: IngestorConfig, **extra
    ) -> str:
        """Render one template with ``config`` plus ``ids`` and ``emitted``
        from ``extra``. A caller rendering one template alone gets ``emitted``
        computed here rather than a ``StrictUndefined`` error."""
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
#: the templates no longer write.
PLACEHOLDER_MARKER = "FILL THIS OUT"

FRAGMENT_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("fragments/cmake_descriptor_files.j2", "cmake_descriptor_files.txt"),
    ("fragments/cmake_target_sources.j2", "cmake_target_sources.txt"),
    ("fragments/cmake_test_sources.j2", "cmake_test_sources.txt"),
    ("fragments/ingestor_packs_hpp.j2", "ingestor_packs.hpp.txt"),
    ("fragments/ingestor_packs_cpp.j2", "ingestor_packs.cpp.txt"),
)
FRAGMENT_FILENAMES: tuple[str, ...] = tuple(name for _, name in FRAGMENT_TEMPLATES)
