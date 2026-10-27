# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Renders a full descriptor bundle for one ``IngestorConfig``.

Descriptor JSON is built as dicts and serialized with ``json.dumps``; Jinja2
renders only the C++ stubs and the CMake/registration fragments.

Ids are assigned once per run by :func:`mint_ids`: an id a previous version of
the bundle already carries is reused when its stable key matches (see
:func:`kernel_key`), and only a descriptor with no match gets a fresh
``uuid4``. :func:`resolve_ids` finds that previous version, in the output
directory or in a directory named by the caller, and reports what it reused.
"""

import copy
import json
import os
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import NamedTuple

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from . import generic_targets as gtmod
from . import prior_ids
from .models import (
    KERNEL_SOURCE_KIND_HSACO,
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


def kernel_key(
    metadata: dict, arch, config: IngestorConfig
) -> tuple[str, tuple[str, ...]]:
    """A kernel's catalog identity: its metadata tuple completed with the CURRENT
    config's KMD fields, plus its effective arch. It is the key the loader uses,
    so it is unique among emitted kernels, and names and positions play no part.

    `mint_ids` matches on the whole key first and falls back to the tuple alone,
    so widening or narrowing a kernel's arch, or renaming it, keeps its id
    while a metadata fix or a different tuple gets a new one. A match is reused
    only if the kernel's source fields are unchanged outside ``spec``: the same
    tuple built from different compiled bytes is a retune and gets a new id (see
    `mint_ids`)."""
    return (_dedup_key(metadata, config), tuple(sorted(arch or [])))


class _Held(NamedTuple):
    """A kernel of a previous version, as `mint_ids` matches it."""

    key: tuple
    id: str
    source: dict  # `_entry_source_fields` of the entry
    supplied: bool  # read from the supplement rather than the primary prior


def _launch_covers(prior: dict, new: dict) -> bool:
    """Whether one ``kernel_source`` object still names the binary ``prior`` did.

    Only ``spec`` may grow: a spec field only ``new`` carries is a builder knob
    stated at its default, which the loader hydrates the same way, so it is no
    difference. Every other key must be equal in both directions, because a key
    the source gains outside ``spec`` (say a stated ``target``) changes what is
    compiled. A spec field ``prior`` had that ``new`` drops or changes is a
    difference."""
    prior_spec, new_spec = prior.get("spec") or {}, new.get("spec") or {}
    return {k: v for k, v in prior.items() if k != "spec"} == {
        k: v for k, v in new.items() if k != "spec"
    } and all(k in new_spec and new_spec[k] == v for k, v in prior_spec.items())


def _source_covers(prior: dict, new: dict) -> bool:
    """Whether ``new`` source fields still name what ``prior`` named.

    Both are source fields as emitted (`KernelSource.descriptor_fields`): a
    ``kernel_source`` object, or a ``sources`` map of launch name to such an
    object. A kernel changing between the two, or a multi-launch kernel gaining
    or losing a launch, is a difference. Each launch is compared by
    `_launch_covers`, so the rule is the one `tools/catalog_diff.py` applies per
    kernel, with the spec keyed on the previous kernel's own fields."""
    if prior.keys() != new.keys():
        return False
    if "kernel_source" in prior:
        return _launch_covers(prior["kernel_source"], new["kernel_source"])
    before, after = prior["sources"], new["sources"]
    return before.keys() == after.keys() and all(
        _launch_covers(before[name], after[name]) for name in before
    )


def _prior_key(kdp, entry: dict, config: IngestorConfig) -> tuple:
    """`kernel_key` of a kernel entry read from a previous KDP."""
    return kernel_key(entry.get("metadata") or {}, kdp.effective_arch(entry), config)


def mint_ids(
    config: IngestorConfig,
    prior: "prior_ids.PriorBundle | None" = None,
    siblings: "Sequence[prior_ids.PriorBundle]" = (),
    supplement: "prior_ids.PriorBundle | None" = None,
) -> dict:
    """Assign every descriptor an id, once per bundle.

    Keys are the engine singletons (``kmd``, ``ued``, ``kernel_match``, ``udd``,
    ``uhd``) plus ``("pack", i)``, ``("operation_umd", i)`` and
    ``("kernel", pack_i, kernel_i)`` indexed by position, so duplicate names stay
    distinct.

    With no ``prior`` and no ``siblings`` every id is a fresh ``uuid4``.

    * Engine-level ids come from the engine's home bundle, ``prior`` or one of
      ``siblings``, whichever holds the UED.
    * A pack takes the id of the ``prior`` KDP with the same name; a sibling's KDP
      id is never reused, since two KDPs sharing an id are a conflict at load.
    * A kernel takes the id of the ``prior`` kernel with the same
      :func:`kernel_key`. A key held by several prior kernels is ambiguous and
      reuses none of them. A kernel left over takes the id of a left-over prior
      kernel with the same metadata tuple at a different arch, but only when each
      side has exactly one such candidate: a tuple shipped at several arch tiers
      is told apart by arch alone.
    * A kernel whose source changes or drops something the prior kernel's said (see
      `_source_covers`) is a retune, the same tuple built from different compiled
      bytes, and gets a new id. RFC 0017 leaves open whether an id may stay with
      changed bytes, and a plan that records only which kernel was picked would
      resolve a reused id to different code on replay. A ``spec`` field only the
      new source carries is not a retune: it cannot be told from a knob stated at
      its default. Any other added or changed field is, and for a multi-launch
      kernel each launch is judged alone.
    * ``supplement`` is a second previous version, a scratch pass of this bundle.
      It fills gaps: a pack or kernel ``prior`` has no match for may take its id
      from there. ``prior`` wins wherever its kernel of the key still covers the
      new source, so ids already shipped never change; where it does not (a
      retune), the supplement's kernel of that key and source keeps the id minted
      for the retune across reruns.
    * Each prior id is assigned at most once, in config order, so no id reaches
      two descriptors. A tuple's id follows it across an arch change (the second
      pass above), so a removed kernel's id can reach a different kernel that
      carries the same tuple at another arch.
    """
    engine = next(
        (
            bundle.engine
            for bundle in (prior, *siblings)
            if bundle is not None and bundle.engine is not None
        ),
        None,
    )
    claimed: set[str] = set()

    def take(existing: str | None) -> str:
        identity = existing if existing not in (None, *claimed) else str(uuid.uuid4())
        claimed.add(identity)
        return identity

    ids = {
        "kmd": take(engine.kmd["id"] if engine else None),
        "ued": take(engine.ued["id"] if engine else None),
        "kernel_match": take(engine.kernel_match_umd["id"] if engine else None),
    }
    if config.engine.has_heuristic:
        ids["uhd"] = take(engine.uhd["id"] if engine and engine.uhd else None)
    ids["udd"] = take(engine.udd["id"] if engine else None)

    prior_packs = {kdp.name: kdp for kdp in prior.kdps} if prior else {}

    # Prior kernels in file order, then the supplement's that the prior's ids do
    # not already hold.
    held: list[_Held] = []
    for kdp in prior.kdps if prior else ():
        for entry in kdp.entries:
            held.append(
                _Held(
                    _prior_key(kdp, entry, config),
                    entry["id"],
                    _entry_source_fields(entry),
                    False,
                )
            )
    primary_keys = {h.key for h in held}
    supplement_packs = {}
    if supplement is not None:
        supplement_packs = {kdp.name: kdp for kdp in supplement.kdps}
        primary_ids = {h.id for h in held}
        for kdp in supplement.kdps:
            for entry in kdp.entries:
                if entry["id"] not in primary_ids:
                    held.append(
                        _Held(
                            _prior_key(kdp, entry, config),
                            entry["id"],
                            _entry_source_fields(entry),
                            True,
                        )
                    )
    by_key: dict[tuple, list[_Held]] = {}
    for h in held:
        by_key.setdefault(h.key, []).append(h)

    # A kernel the de-duplication will drop is not a candidate for an id: it
    # would count as a second claimant of its tuple and read an arch move as
    # ambiguous.
    seen_candidates = {
        (_prior_key(kdp, entry, config), _entry_identity(entry))
        for bundle in siblings
        for kdp in bundle.kdps
        for entry in kdp.entries
    }
    wanted: list[tuple[tuple, tuple, dict]] = []  # (slot, key, source) per kernel
    dropped: list[tuple] = []
    for pack_index, pack in enumerate(config.packs):
        name = f"{config.engine.namespace}:{config.kdp_stem(pack)}"
        matched = prior_packs.get(name) or supplement_packs.get(name)
        ids[("pack", pack_index)] = take(matched.id if matched else None)
        if config.is_multi_pack:
            ids[("operation_umd", pack_index)] = take(
                matched.operation_umd_id if matched else None
            )
        for kernel_index, kernel in enumerate(pack.kernels):
            slot = ("kernel", pack_index, kernel_index)
            key = source = None
            if held:
                key = kernel_key(
                    _resolved_metadata(kernel, config),
                    kernel.arch or pack.arch,
                    config,
                )
                source = kernel.kernel_source.descriptor_fields()
                candidate = (key, _candidate_identity(kernel))
                if candidate in seen_candidates:
                    dropped.append(slot)
                    continue
                seen_candidates.add(candidate)
            wanted.append((slot, key, source))

    def pick(group: list[_Held], source: dict) -> str | None:
        """The id of the one kernel of ``group`` that still covers ``source``."""
        if len(group) == 1 and group[0].id not in claimed:
            if _source_covers(group[0].source, source):
                return group[0].id
        return None

    # Pass 1: the whole key, unambiguous on the prior side, built from the same
    # source. The primary prior answers first; the supplement answers only where
    # it does not (a retune whose id an earlier scratch pass already minted).
    left_over = []
    for slot, key, source in wanted:
        held_here = by_key.get(key, ())
        found = None
        for supplied in (False, True):
            group = [h for h in held_here if h.supplied == supplied]
            if len(group) > 1:
                break
            found = pick(group, source)
            if found is not None:
                break
        if found is not None:
            ids[slot] = take(found)
        else:
            left_over.append((slot, key, source))
    # Pass 2: the tuple alone, for kernels whose arch moved. Each side must offer
    # exactly one candidate, or the pairing would be a guess. A supplement kernel
    # at a key the prior also holds is no candidate: it is that key's scratch pass.
    unclaimed: dict[str, list[_Held]] = {}
    for h in held:
        if h.id not in claimed and (not h.supplied or h.key not in primary_keys):
            unclaimed.setdefault(h.key[0], []).append(h)
    leftovers: dict[str, int] = {}
    for _slot, key, _source in left_over:
        if key is not None:
            leftovers[key[0]] = leftovers.get(key[0], 0) + 1
    for slot, key, source in left_over:
        candidates = unclaimed.get(key[0], ()) if key is not None else ()
        if len(candidates) == 1 and leftovers[key[0]] == 1:
            ids[slot] = take(pick(candidates, source))
        else:
            ids[slot] = take(None)
    for slot in dropped:
        ids[slot] = take(None)
    assert len(set(ids.values())) == len(ids), "mint_ids assigned an id twice"
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
    # RFC 0019 §4.1: a non-static adapter must declare its objective. The loader drops a
    # UHD carrying `kind`/`payload` keys.
    return {
        "version": "1.0",
        "id": ids["uhd"],
        "name": f"{config.engine.local_name} selector",
        "adapter": "native",
        "objective": "max",
        "native": {"symbol": config.score_symbol},
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
        # RFC 0019 §3.1: the loader rejects a bare top-level `heuristic` pointer.
        ued["sort_kernel_catalog"] = {"default": ids["uhd"]}
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
    # A prebuilt hsaco object has no spec to pin a knob in; only metadata decides.
    prebuilt = kernel.kernel_source.kind == KERNEL_SOURCE_KIND_HSACO
    unresolved = sorted(k for k, v in metadata.items() if v == UNSET_SENTINEL)
    if unresolved:
        remedy = (
            "Write the resolved value in metadata"
            if prebuilt
            else "Pin the knob in kernel_source.spec, or write the resolved value in "
            "metadata"
        )
        raise ValueError(
            f"kernel {kernel.name!r} ships the unset sentinel "
            f"({UNSET_SENTINEL}) for {unresolved}: metadata must state the value the "
            f"kernel was BUILT with. {remedy} -- the descriptor cannot say "
            f"'undecided' about a binary that already decided."
        )

    spec = kernel.kernel_source.spec or {}
    undeclared = sorted(
        f.name
        for f in config.kmd_fields
        if not f.is_mandatory and f.name not in metadata and spec.get(f.name) is None
    )
    if undeclared:
        where = (
            "its metadata"
            if prebuilt
            else "neither its metadata nor its kernel_source.spec"
        )
        remedy = (
            "Write the resolved value in metadata"
            if prebuilt
            else "Pin the knob in kernel_source.spec if the binary should carry it, "
            "or write the resolved value in metadata"
        )
        if prebuilt:
            tail = (
                f"so nothing here decides the value. The loader will substitute the "
                f"KMD default_value as the catalog key, which need not match what "
                f"the prebuilt object was built with, and the disagreement is "
                f"silent. {remedy}."
            )
        else:
            tail = (
                f"so nothing here decides the value. The loader will substitute the "
                f"KMD default_value as the catalog key while the kernel is compiled "
                f"from the builder's own default -- two independent defaults that "
                f"are not required to agree, and whose disagreement is silent. "
                f"{remedy} if the builder's default is what you mean."
            )
        raise ValueError(
            f"kernel {kernel.name!r} states {undeclared} in {where}, {tail}"
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

    Architecture is not part of this key; `build_kdp` compares arch separately.
    """
    return json.dumps(_completed_metadata(metadata, config), sort_keys=True)


def _entry_source_fields(entry: dict) -> dict:
    """The source fields of a UKD read from disk: ``kernel_source`` or ``sources``."""
    return {key: entry[key] for key in ("kernel_source", "sources") if key in entry}


def _entry_identity(entry: dict) -> str:
    """`_candidate_identity_doc` of a UKD read from disk."""
    return _candidate_identity_doc(
        _entry_source_fields(entry), entry.get("priority", 0)
    )


def _candidate_identity_doc(source_fields: dict, priority: int) -> str:
    """What a candidate is beyond its tuple: two entries sharing a completed
    tuple are the same candidate only if they name the same binary at the same
    priority. A knob absent from ``kernel_source.spec`` and one pinned to the
    value the policy chose are different binaries under one tuple. A kernel's
    launches are part of what it names: two kernels differing only in a launch
    are different candidates.

    Takes the emitted source fields (`KernelSource.descriptor_fields`: either
    ``kernel_source`` or ``sources``), so a prior descriptor read from disk is
    identified the same way as a `KernelSpec`."""
    return json.dumps({**source_fields, "priority": priority}, sort_keys=True)


def _candidate_identity(kernel: KernelSpec) -> str:
    return _candidate_identity_doc(
        kernel.kernel_source.descriptor_fields(), kernel.priority
    )


def _arch_competes(left: list, right: list) -> bool:
    """Whether two arch lists tie on some device at the same tier."""
    return gtmod.compete(left, right, gtmod.default_table())


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
    ``engine_id`` and ``kmd_id`` come from `mint_ids` (reused from the previous
    bundle when there is one); the rest is authored.
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
    # tuple is a duplicate only when arch coverage competes: the same candidate
    # with equal arch de-duplicates, anything else is refused.
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
            if not _arch_competes(arch, prior["arch"]):
                continue
            if prior["identity"] == identity:
                if sorted(prior["arch"]) == sorted(arch):
                    already = prior
                    break
                raise ValueError(
                    f"kernel {kernel.name!r} (pack {pack.name!r}) and kernel "
                    f"{prior['name']!r} (pack {prior['pack']!r}) are the SAME "
                    f"candidate and complete to the SAME catalog tuple "
                    f"{json.loads(key)}, but they compete at the same arch tier "
                    f"without having equal arch lists ({arch or ['<any>']} vs "
                    f"{prior['arch'] or ['<any>']}). On the shared architectures "
                    f"the matcher would see one tuple twice, which drops the whole "
                    f"engine at load; coalescing them would instead advertise one "
                    f"of the two on devices its arch list never claimed. Give the "
                    f"two entries the SAME arch list so they de-duplicate, or put "
                    f"them at different arch tiers."
                )
            raise ValueError(
                f"kernel {kernel.name!r} (pack {pack.name!r}) and kernel "
                f"{prior['name']!r} (pack {prior['pack']!r}) complete to the SAME "
                f"catalog tuple {json.loads(key)} at the same arch tier "
                f"({arch or ['<any>']} vs {prior['arch'] or ['<any>']}), but they "
                f"are not the same candidate: their kernel_source/priority differ, "
                f"so they name different binaries. The matcher compares the tuple "
                f"and would see one entry twice -- a duplicate tuple drops the "
                f"whole engine at load, and dropping one of them here would "
                f"discard a binary that was deliberately built. Distinguish them "
                f"in metadata (a knob the spec pins belongs in the tuple), narrow "
                f"one of the arch lists, move one to a different arch tier, or do "
                f"not ship both."
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
            # 1.1 only on a UKD carrying `sources`; every other stays 1.0.
            "version": kernel.kernel_source.ukd_version,
            "id": ids[("kernel", pack_index, index)],
            "name": kernel.name,
            # Per-kind keys, never the union: the runtime loader hard-fails an
            # unknown key and hkp_pack validates a closed set per kind. A
            # multi-launch kernel writes `sources` in place of `kernel_source`.
            **kernel.kernel_source.descriptor_fields(),
            "metadata": metadata,
            "priority": kernel.priority,
        }
        # hkp_pack validates the kernel's own arch for hsaco and rejects a
        # wildcard, so the inherited pack arch is stated on the descriptor.
        if kernel.kernel_source.kind == KERNEL_SOURCE_KIND_HSACO:
            entry["arch"] = arch
        elif kernel.arch:
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


def build_kdp_documents(
    config: IngestorConfig, ids: dict, seed: dict | None = None
) -> list:
    """Every pack's KDP, ``[(pack, document), ...]``, de-duplicated engine-wide.
    The single de-duplication scope; see `build_kdp`.

    ``seed`` pre-fills the de-duplication state with the kernels of sibling
    bundles of the same engine (see `resolve_ids`); it is copied, not mutated."""
    seen_metadata: dict = copy.deepcopy(seed) if seed else {}
    return [
        (pack, build_kdp(config, pack, ids, seen_metadata)) for pack in config.packs
    ]


@dataclass(frozen=True)
class IdReport:
    own_prior: Path | None  # where this bundle's ids came from; None = nothing reused
    home: Path | None  # the root holding the engine-level files, if any
    reused: dict[str, int]  # engine / packs / operation_umds / kernels
    minted: int
    retired: tuple[str, ...]  # prior kernels and KDPs whose ids were not reused
    warnings: tuple[str, ...] = ()  # things the caller should see but may proceed past
    arch_moved: tuple[str, ...] = ()  # kernels that kept an id across an arch change


@dataclass(frozen=True)
class IdResolution:
    ids: dict
    satellite: bool  # the engine-level files live in a sibling; emit the KDP only
    kdp_documents: list  # build_kdp_documents(config, ids, seed), built once here
    report: (
        IdReport  # counted from kdp_documents: a de-duplicated kernel is not emitted
    )


def _normal_parts(path: "Path | PurePosixPath") -> tuple[str, ...]:
    return tuple(os.path.normcase(part) for part in path.parts)


def _ends_with(path: Path, relative: str) -> bool:
    tail = _normal_parts(PurePosixPath(relative))
    # Not resolve(): a symlinked checkout or directory would otherwise be compared
    # by where it points rather than by where the caller put it.
    parts = _normal_parts(Path(os.path.abspath(path)))
    return parts[len(parts) - len(tail) :] == tail


def _family_of(path: Path) -> str:
    """The folder under ``descriptors/`` that ``path`` sits in: ``hkp_pack``
    excludes a disabled family whole, so every bundle of one engine must share it."""
    # Not resolve(), for the reason given in `_ends_with`.
    parts = Path(os.path.abspath(path)).parts
    for index in range(len(parts) - 2, -1, -1):
        if parts[index] == "descriptors":
            return os.path.normcase(parts[index + 1])
    raise ValueError(
        f"{path} has no 'descriptors/<family>' component, so its family folder "
        f"cannot be told. Pass the bundle directory inside the provider's "
        f"descriptors/ tree."
    )


def _check_seed_entry(path: Path, entry: dict) -> None:
    """A sibling KDP's kernel entry must carry what the de-duplication reads. The
    reader has already checked its source, metadata and arch (`prior_ids`)."""
    if not isinstance(entry.get("name"), str):
        raise ValueError(f"{path}: kernel {entry.get('id')!r} has no string 'name'")


def _check_sibling_mode(config: IngestorConfig) -> None:
    if not config.is_packaged:
        raise prior_ids.NotOverridable(
            "--engine-ids-from needs a packaged bundle: each test_descriptors/<set> "
            "is its own pack root, so a satellite's engine, dispatch and matcher "
            "references could not resolve there."
        )
    if config.is_multi_pack:
        raise prior_ids.NotOverridable(
            "sibling bundles need a single-pack bundle: a multi-pack engine's "
            "operation matcher names a symbol only that bundle's own Native.cpp "
            "registers. Another bundle of this engine, named to the run or found "
            "beside this one in the provider tree, puts the config in sibling mode."
        )


def resolve_ids(
    config: IngestorConfig,
    output_dir: Path,
    ids_from: Path | None = None,
    engine_ids_from: Sequence[Path] = (),
    *,
    fresh: bool = False,
    also_from: Path | None = None,
) -> IdResolution:
    """Find this bundle's previous version, assign every id, and build the KDPs.

    ``ids_from`` is the previous version of this bundle and is strict: any
    problem reading it raises. Without it, ``output_dir / config.descriptor_dir``
    is read when it holds descriptors, so regenerating into an existing tree
    keeps its ids; a missing or empty directory there means a new bundle.

    ``engine_ids_from`` names every OTHER bundle of the same engine. Exactly one
    bundle of an engine, the home, holds the engine-level files; the others are
    satellites holding only their KDP. The engine-level ids are taken from the
    home, and the siblings' kernels seed the engine-wide de-duplication. A
    sibling's KDP and kernel ids are never reused, since a repeated id with
    different content is a conflict at load; a KDP id or kernel id held by two of
    the bundles read is refused.

    ``also_from`` is a second, lenient source: a previous run of this bundle (say
    the scratch directory being iterated in) whose ids fill in what ``ids_from``
    does not know, namely the kernels added since it was installed. It never
    overrides ``ids_from``, and it is ignored with a warning when it belongs to a
    different engine id or cannot be read.

    ``fresh`` reads nothing and gives every descriptor a new id: the explicit way
    to start a bundle over. It excludes the other two inputs.

    Raises ValueError, before anything is written.
    """
    output_dir = Path(output_dir)
    sibling_roots = [Path(root) for root in engine_ids_from]
    if fresh:
        if ids_from is not None or sibling_roots:
            raise ValueError(
                "fresh ids contradict --ids-from and --engine-ids-from, which exist "
                "to reuse them"
            )
        ids = mint_ids(config)
        kdp_documents = build_kdp_documents(config, ids)
        report = _report(config, ids, kdp_documents, None, None, None, None)
        return IdResolution(
            ids=ids,
            satellite=False,
            kdp_documents=kdp_documents,
            report=report,
        )
    if sibling_roots:
        _check_sibling_mode(config)

    strict = ids_from is not None
    if strict:
        own_root = Path(ids_from)
        if not _ends_with(own_root, config.descriptor_dir):
            raise ValueError(
                f"--ids-from {own_root} is not this bundle's directory (this config "
                f"writes to a path ending {config.descriptor_dir!r}). For another "
                f"bundle of the same engine use --engine-ids-from."
            )
    else:
        own_root = output_dir / config.descriptor_dir
    for sibling in sibling_roots:
        if sibling.resolve() == own_root.resolve():
            raise ValueError(f"the sibling {sibling} is this bundle")
        if _ends_with(sibling, config.descriptor_dir):
            raise ValueError(
                f"sibling {sibling} ends with this config's descriptor directory "
                f"{config.descriptor_dir!r}: a satellite needs its own "
                f"authored_subpath, or its KDP would be written into the sibling's "
                f"folder."
            )

    use_own = strict or prior_ids.holds_docs(own_root, "ued", "kdp")
    roots = ([own_root] if use_own else []) + sibling_roots
    homes = []
    for root in roots:
        engine = prior_ids.find_engine(root, config.engine.name)
        if engine is not None:
            homes.append((root, engine))
    if len(homes) > 1:
        raise ValueError(
            f"the UED of engine {config.engine.name!r} is in {len(homes)} roots, "
            f"{[str(root) for root, _ in homes]}: exactly one bundle of an engine "
            f"holds the engine-level files, and hkp_pack rejects a repeated id."
        )

    warnings = []
    own = None
    siblings: list = []
    home_root = None
    engine = None
    if homes:
        home_root, engine = homes[0]
        engine_id = engine.ued["id"]
        match_id = engine.kernel_match_umd["id"]
        if use_own:
            own = prior_ids.read_prior_bundle(own_root, engine_id, match_id)
        siblings = [
            prior_ids.read_prior_bundle(root, engine_id, match_id)
            for root in sibling_roots
        ]
        bundles = [bundle for bundle in (own, *siblings) if bundle is not None]
        owner: dict[str, Path] = {}
        kernel_owner: dict[str, Path] = {}
        for bundle in bundles:
            for kdp in bundle.kdps:
                if kdp.id in owner:
                    raise ValueError(
                        f"KDP id {kdp.id} is in both {owner[kdp.id]} and "
                        f"{bundle.root}: the sibling is this bundle, or a copy of it."
                    )
                owner[kdp.id] = bundle.root
                for entry in kdp.entries:
                    if entry["id"] in kernel_owner:
                        raise ValueError(
                            f"kernel id {entry['id']} is in {kernel_owner[entry['id']]} "
                            f"and also in {kdp.path}: two bundles of one engine "
                            f"cannot share a kernel id, and hkp_pack rejects it. "
                            f"Remove the kernel from one of the two files."
                        )
                    kernel_owner[entry["id"]] = kdp.path
                if kdp.dispatch != engine.udd["id"]:
                    raise ValueError(
                        f"{kdp.path}: dispatch {kdp.dispatch} is not the home's UDD "
                        f"{engine.udd['id']}"
                    )
                if sibling_roots and kdp.operation_umd_id is not None:
                    raise ValueError(
                        f"{kdp.path} carries a graph-scope operation matcher, which "
                        f"names a symbol only its own bundle's Native.cpp registers: "
                        f"sibling mode is for single-pack bundles."
                    )
        if own is not None and any(pack.kernels for pack in config.packs):
            packs = {
                f"{config.engine.namespace}:{config.kdp_stem(pack)}"
                for pack in config.packs
                if pack.kernels
            }
            warnings.extend(
                f"{kdp.path} holds no kernels, so every kernel id of the pack "
                f"{kdp.name!r} would be minted new. A truncated or placeholder "
                f"KDP is the usual cause: restore it (dvc pull, or the last "
                f"shipped copy) to keep the ids."
                for kdp in own.kdps
                if not kdp.entries and kdp.name in packs
            )
    elif sibling_roots:
        raise ValueError(
            f"none of the siblings {[str(root) for root in sibling_roots]} holds a "
            f"UED named {config.engine.name!r}: no sibling holds the engine's UED"
        )
    elif strict:
        raise ValueError(
            f"no UED named {config.engine.name!r} under --ids-from {own_root}"
        )
    elif use_own:
        satellite_hint = (
            f" It holds KDP(s) but no engine-level files: this looks like a "
            f"satellite bundle; pass --engine-ids-from <the home bundle's directory>."
            if prior_ids.list_docs(own_root, "kdp")
            else ""
        )
        raise ValueError(
            f"{own_root} holds descriptors but no UED named "
            f"{config.engine.name!r}.{satellite_hint} If the engine was renamed, "
            f"pass --new-ids to start its ids over, or clear that directory."
        )

    supplement = None
    if (
        also_from is not None
        and engine is not None
        and not (use_own and also_from.resolve() == own_root.resolve())
    ):
        try:
            other = prior_ids.find_engine(also_from, config.engine.name)
            if other is not None and other.ued["id"] != engine.ued["id"]:
                warnings.append(
                    f"{also_from} holds engine {config.engine.name!r} under a "
                    f"different id than {home_root}, so it was made without those ids; "
                    f"it is ignored."
                )
            else:
                supplement = prior_ids.read_prior_bundle(
                    also_from, engine.ued["id"], engine.kernel_match_umd["id"]
                )
        except ValueError as e:
            warnings.append(f"{also_from} is ignored as a source of ids: {e}")

    satellite = bool(sibling_roots) and home_root is not own_root
    if sibling_roots:
        home_family = (
            _family_of(home_root)
            if home_root is not own_root
            else PurePosixPath(config.descriptor_dir).parts[1]
        )
        families = {PurePosixPath(config.descriptor_dir).parts[1]} | {
            _family_of(root) for root in sibling_roots
        }
        if {os.path.normcase(f) for f in families} != {os.path.normcase(home_family)}:
            raise ValueError(
                f"bundles of one engine must share a family folder under "
                f"descriptors/, found {sorted(families)}: hkp_pack excludes a "
                f"disabled family whole, which would leave the others' references "
                f"dangling."
            )

    if own is not None or siblings:
        for root in ([own_root] if own is not None else []) + sibling_roots:
            warnings.extend(prior_ids.dvc_mismatches(root))
    if engine is not None and config.is_packaged:
        named = ([own_root] if use_own else []) + sibling_roots
        beside = {root.parent for root in named}
        for parent in sorted(beside):
            warnings.extend(prior_ids.unfetched_dvc(parent, named))
            for other in prior_ids.unnamed_siblings(parent, engine.ued["id"], named):
                warnings.append(
                    f"{other} holds another bundle of engine {config.engine.name!r} "
                    f"that is not named with --engine-ids-from. Left out, its kernels "
                    f"drop out of the engine-wide duplicate check, and a tuple "
                    f"competing with one of them drops the whole engine at load."
                )

    seed = None
    if siblings:
        seed = {}
        for bundle in siblings:
            for kdp in bundle.kdps:
                for entry in kdp.entries:
                    _check_seed_entry(kdp.path, entry)
                    metadata = entry.get("metadata") or {}
                    seed.setdefault(_dedup_key(metadata, config), []).append(
                        {
                            "name": entry["name"],
                            "pack": kdp.name,
                            "arch": list(kdp.effective_arch(entry) or []),
                            "identity": _entry_identity(entry),
                        }
                    )

    ids = mint_ids(config, own, siblings, supplement)

    if satellite:
        built = {
            "ued": (build_ued(config, ids), engine.ued),
            "kmd": (build_kmd(config, ids), engine.kmd),
            "udd": (build_udd(config, ids), engine.udd),
            "kernel match UMD": (
                build_kernel_match_umd(config, ids),
                engine.kernel_match_umd,
            ),
            "uhd": (build_uhd(config, ids), engine.uhd),
        }
        for name, (generated, home_doc) in built.items():
            if generated != home_doc:
                raise ValueError(
                    f"this config's {name} differs from the home bundle's ({home_root}). "
                    f"Change the engine block in the home bundle and regenerate every "
                    f"satellite after it: a satellite's KDPs depend on the home's KMD "
                    f"through completion and through specialization_contract.kmd_id."
                )

    kdp_documents = build_kdp_documents(config, ids, seed)
    report = _report(
        config,
        ids,
        kdp_documents,
        own,
        engine,
        own_root,
        home_root,
        supplement,
    )
    report = replace(report, warnings=tuple(warnings))
    return IdResolution(
        ids=ids,
        satellite=satellite,
        kdp_documents=kdp_documents,
        report=report,
    )


def _report(
    config,
    ids,
    kdp_documents,
    own,
    engine,
    own_root,
    home_root,
    supplement=None,
):
    """Count what was reused from what is actually emitted."""
    previous = [bundle for bundle in (own, supplement) if bundle is not None]
    known = set().union(*(bundle.all_ids for bundle in previous))
    if engine is not None:
        known.update(
            doc["id"]
            for doc in (
                engine.ued,
                engine.kmd,
                engine.udd,
                engine.kernel_match_umd,
                engine.uhd,
            )
            if doc
        )
    # What each previous kernel's effective arch was, to see which reuse moved it.
    previous_arch = {
        entry["id"]: sorted(kdp.effective_arch(entry) or [])
        for bundle in previous
        for kdp in bundle.kdps
        for entry in kdp.entries
    }
    reused = {"engine": 0, "packs": 0, "operation_umds": 0, "kernels": 0}
    total = 0
    for key in ("kmd", "ued", "kernel_match", "udd", "uhd"):
        if key in ids:
            total += 1
            reused["engine"] += ids[key] in known
    emitted: set[str] = set()
    arch_moved = []
    for pack, document in kdp_documents:
        index = _pack_index(config, pack)
        total += 1
        emitted.add(document["id"])
        reused["packs"] += document["id"] in known
        if config.is_multi_pack:
            total += 1
            reused["operation_umds"] += ids[("operation_umd", index)] in known
        for entry in document["kernelDescriptors"]:
            total += 1
            emitted.add(entry["id"])
            reused["kernels"] += entry["id"] in known
            now = sorted(entry.get("arch") or document.get("arch") or [])
            if entry["id"] in previous_arch and previous_arch[entry["id"]] != now:
                arch_moved.append(entry["name"])
    retired = []
    seen: set[str] = set()
    for bundle in previous:
        for kdp in bundle.kdps:
            if kdp.id not in emitted and kdp.id not in seen:
                retired.append(kdp.name)
            seen.add(kdp.id)
            for e in kdp.entries:
                if e["id"] not in emitted and e["id"] not in seen:
                    retired.append(e["name"])
                seen.add(e["id"])
    return IdReport(
        own_prior=own_root if own else None,
        home=home_root,
        reused=reused,
        minted=total - sum(reused.values()),
        retired=tuple(retired),
        arch_moved=tuple(arch_moved),
    )


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

    Each generic's descriptors also fill the rows of its members; the wildcard
    row fills every row.
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

    table = gtmod.default_table()
    generic_rows = {
        name: (list(arches[name]["descriptors"]), set(arches[name]["pack_names"]))
        for name in table.names()
        if name in arches
    }
    for name in generic_rows:
        for member in table.members(name):
            bucket(member)
    for arch, entry in arches.items():
        for name, (descriptors, pack_names) in generic_rows.items():
            if arch in table.members(name):
                entry["descriptors"].extend(descriptors)
                entry["pack_names"].update(pack_names)

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

    def preview_files(
        self, config: IngestorConfig, *, resolution: IdResolution | None = None
    ) -> list[str]:
        """The shippable file list :meth:`render` would write, without writing
        anything. ``resolution=None`` previews the home layout; a satellite
        resolution lists its KDP alone, since the C++ stubs and fragments it also
        writes are discarded."""
        slug = config.engine.slug
        ddir = config.descriptor_dir
        if resolution is not None and resolution.satellite:
            return [f"{ddir}/{config.kdp_stem(pack)}.kdp.json" for pack in config.packs]
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

    @staticmethod
    def refuse_second_kdp(config: IngestorConfig, output_dir: Path) -> None:
        """Raises ValueError when this run would write the top-level KDP of a pack
        whose KDP ``output_dir`` already holds under ``artifacts/`` (a DVC output).

        Both files would then carry the pack's id, and every later run would
        refuse the tree for the repeated id."""
        bundle = Path(output_dir) / config.descriptor_dir
        for pack in config.packs:
            stem = config.kdp_stem(pack)
            existing = bundle / "artifacts" / f"{stem}.kdp.json"
            if existing.is_file():
                raise ValueError(
                    f"{existing.as_posix()} already holds the KDP of pack "
                    f"{pack.name!r} as a DVC artifact: writing {stem}.kdp.json "
                    f"beside it would leave two KDPs with one id. Generate into a "
                    f"scratch directory instead and splice the additions into the "
                    f"artifact, then dvc add it."
                )

    def render(
        self,
        config: IngestorConfig,
        output_dir: Path,
        *,
        resolution: IdResolution | None = None,
    ) -> list[str]:
        """Write every descriptor JSON, the native/test C++ stubs and the
        CMake/registration fragments. Returns the relative paths written.

        ``resolution`` carries the ids and the KDPs built from them (see
        `resolve_ids`); ``None`` resolves them from ``output_dir``, so
        regenerating into an existing tree keeps its ids. A satellite writes
        its KDP only: the home bundle holds the engine-level files, and
        ``hkp_pack`` rejects a repeated id even when the content is identical.

        Every KDP is built before rendering, since the templates are given the
        emitted inventory and that view needs the engine-wide de-duplication.
        """
        output_dir = Path(output_dir)
        self.refuse_second_kdp(config, output_dir)
        if resolution is None:
            resolution = resolve_ids(config, output_dir)
        ids = resolution.ids
        written: list[str] = []
        slug = config.engine.slug
        ddir = config.descriptor_dir
        (output_dir / ddir).mkdir(parents=True, exist_ok=True)

        def write_text(rel: str, text: str) -> None:
            # LF on every platform: committed descriptors and sources are LF, and
            # text mode would write CRLF on Windows.
            (output_dir / rel).write_text(text, encoding="utf-8", newline="\n")
            written.append(rel)

        def write_json(rel: str, obj: dict) -> None:
            write_text(rel, _dump(obj))

        kdp_documents = resolution.kdp_documents
        emitted = emitted_inventory(config, kdp_documents)

        if not resolution.satellite:
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
        write_text(
            native_rel,
            self._render_template("native.cpp.j2", config, ids=ids, emitted=emitted),
        )

        packs_test_rel = f"tests/Test{config.engine.pascal_name}Packs.cpp"
        write_text(
            packs_test_rel,
            self._render_template(
                "test_packs.cpp.j2", config, ids=ids, emitted=emitted
            ),
        )

        matchers_test_rel = f"tests/Test{config.engine.pascal_name}Matchers.cpp"
        write_text(
            matchers_test_rel,
            self._render_template(
                "test_matchers.cpp.j2", config, ids=ids, emitted=emitted
            ),
        )

        # --- fragments ---
        fragments_dir = output_dir / "fragments"
        fragments_dir.mkdir(parents=True, exist_ok=True)
        for template_name, out_name in FRAGMENT_TEMPLATES:
            content = self._render_template(
                template_name, config, ids=ids, emitted=emitted
            )
            write_text(f"fragments/{out_name}", content)

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
