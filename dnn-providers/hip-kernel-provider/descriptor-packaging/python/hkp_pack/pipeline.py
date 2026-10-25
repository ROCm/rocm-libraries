import copy
import hashlib
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import agreement, provenance_sidecar, sources, toolchain
from .hip_compile import (
    compile_hip_variant,
    hip_source_relpath,
    hip_variant_key,
)
from .hsaco_source import hsaco_file_identity, hsaco_variant_key, resolve_hsaco_file
from .rocke_compile import (
    ROCKE_TARGET_NATIVE,
    compile_rocke_variant,
    rocke_compile_arch,
    rocke_variant_key,
)
from .descriptors import (
    KPACK_DIR_NAME,
    canonical_id,
    _selected_entries,
    kdp_arch_matches,
    kdp_survives,
    load_flat_input,
    reachable_generic_ids,
)
from .errors import HkpPackError
from .generic_targets import GenericTargets
from .kernel_signature import kernel_signature
from .kpack_resolver import load_kpack

# The archive group every root packs under unless it names its own. One archive ships per
# (group, arch), and the filename carries both, so two roots staged into one descriptor
# tree must not share a group -- otherwise they emit the same
# `<arch>/kpack/<group>_<arch>.kpack` and whichever copy lands second silently overwrites
# the other, leaving descriptors naming an archive that no longer holds their kernels.
GROUP_NAME = "hip_kernel_provider"

# The kernel_source kinds the packer ships as authored. No producer runs for
# them, so they contribute no code object and no archive entry.
_PASSTHROUGH_KINDS = frozenset({"embedded_source"})


@dataclass
class InlineUKD:
    id: str
    name: str
    metadata: dict
    priority: object
    source: str
    entry: str
    build: dict
    symbol: str
    variant_key: str
    extra: dict = field(default_factory=dict)
    origin_kind: str = "hip"
    builder: object = None
    spec: object = None
    provenance: dict = field(default_factory=dict)
    observations: dict = field(default_factory=dict)
    consumers: list = field(default_factory=list)
    # hsaco only: the authored file's resolved root-relative identity.
    rel_file: object = None
    # rocke only: the processor the code object was compiled for -- the shard
    # arch for target "native", its LLVM generic processor for "generic".
    compile_target: object = None
    # rocke `sources` only: `{launch name: LaunchEntry}`, in authored order. Then
    # `variant_key`, `symbol`, `source`, `builder`, `spec`, `observations` and
    # `compile_target` above belong to the launches, not to the UKD.
    launches: dict = field(default_factory=dict)


@dataclass
class LaunchEntry:
    """One launch of a `sources` UKD, as compiled."""

    variant_key: str
    symbol: str
    source: str
    builder: str
    spec: dict
    compile_target: str
    observations: dict
    # The builder's companion kernel this launch takes; None is the primary.
    role: object = None


@dataclass
class StandaloneUKD(InlineUKD):
    """A UKD authored as its own `<name>.ukd.json` and referenced by a KDP.

    Carries the same compiled fields as an inline UKD plus the original filename,
    since it stays a standalone file in the shipped shard (rather than being
    folded into the KDP).
    """

    filename: str = ""
    rel_dir: Path = Path(".")


@dataclass
class PassthroughUKD:
    """A UKD the packer emits without running a producer.

    `embedded_source` names a key into a table that the consuming binary
    compiles in. The packer builds no code object for it, so the authored
    document is what ships, with `arch` narrowed to the shard. `rel_dir` is the
    descriptor's position under the source root, which provenance records: for
    an inline UKD it is the directory of the KDP that holds it, and for a
    standalone UKD its own. `filename` is set only for the standalone form,
    which stays its own file in the shard.
    """

    doc: dict
    filename: str = ""
    rel_dir: Path = Path(".")


@dataclass
class ArchKDP:
    id: str
    filename: str
    header: dict
    rel_dir: Path = Path(".")
    ukds: list = field(default_factory=list)
    # Ordered kernelDescriptors output spec: each element is an InlineUKD
    # (rewritten inline in the shipped KDP), a PassthroughUKD (shipped inline as
    # authored), or a str (a standalone-UKD id ref, kept verbatim). Preserves
    # authored order across the heterogeneous vector.
    entries: list = field(default_factory=list)


@dataclass
class IntermediateArch:
    arch: str
    directory: Path
    kdps: list = field(default_factory=list)
    variant_co: dict = field(default_factory=dict)
    variant_symbol: dict = field(default_factory=dict)
    standalone_ukds: dict = field(default_factory=dict)
    # Standalone UKDs of a pass-through kind. These carry no code object, no
    # toc_key and no symbol.
    passthrough_standalone_ukds: dict = field(default_factory=dict)


@dataclass
class ArchResult:
    arch: str
    out_dir: Path
    kpack_path: Path
    skipped: bool = False
    # Generic-pass archives copied into this member's folder.
    generic_kpack_paths: tuple = ()


_EXC_TEXT_LIMIT = 200


def _bounded_repr(exc):
    """repr(exc), cut to a length that keeps one error message readable."""
    text = repr(exc)
    if len(text) <= _EXC_TEXT_LIMIT:
        return text
    return f"{text[:_EXC_TEXT_LIMIT]}... ({len(text)} chars)"


def _sha256(data):
    """Digest of a packed blob, recorded on the shipped UKD.

    Hashed here over the decompressed code object, before the archive's
    compressor sees it, because that is the buffer the runtime hashes back:
    KpackModuleCache compares this digest against what it decompressed and
    refuses the load on a mismatch. Hashing the compressed blob or the source
    file instead would make every kpack load fail.

    `expected_sha256` cross-checks it at pack time as well, so a disagreement
    is caught at the producing end rather than only at the consuming one.
    """
    return hashlib.sha256(data).hexdigest()


def _kpack_filename(arch, group=GROUP_NAME):
    return f"{group}_{arch}.kpack"


def _kpack_rel(arch, rel_dir=Path("."), group=GROUP_NAME):
    """`library` for a descriptor living at `rel_dir` within the arch shard.

    The runtime resolves this as `originDirectory / library`, where
    originDirectory is the parent directory of the descriptor FILE. The archive
    itself lives once per arch, at the arch root.

    So a nested descriptor has to climb back out to the arch root before
    descending into `kpack/`. A root-relative value happens to be correct only
    when rel_dir is "." -- true of every flat layout, which is why this was not
    caught until descriptors could nest.

    Climbing out of the descriptor's own directory is legal because the runtime
    anchors containment on the descriptor TREE, not on the individual
    descriptor's folder (`IngestorKernelCode.hpp`, the KPACK case). The archive
    is a sibling inside that tree by construction.
    """
    rel_dir = Path(rel_dir)
    prefix = (
        ""
        if rel_dir in (Path("."), Path(""))
        else "/".join([".."] * len(rel_dir.parts)) + "/"
    )
    return f"{prefix}{KPACK_DIR_NAME}/{_kpack_filename(arch, group)}"


def _kdp_header(doc):
    return {k: v for k, v in doc.items() if k != "kernelDescriptors"}


def _ukd_extra(ukd):
    return {
        k: v
        for k, v in ukd.items()
        if k
        not in (
            "id",
            "name",
            "kernel_source",
            "sources",
            "metadata",
            "priority",
            "build",
            "arch",
            "provenance",
        )
    }


def _compile_rocke_launch(
    ks,
    vk,
    where,
    arch,
    inter_arch_dir,
    variant_co,
    variant_symbol,
    observation_requests,
    variant_observations,
    origins,
):
    """Compile one rocke kernel_source once per variant key, then check the result.

    Returns `(primary symbol, observations, compile_target)`. A reused compile
    result is checked as hard as a fresh one, for EVERY consumer: the first's
    agreement says nothing about a second completing different metadata from the
    same decisions. The shard binding stays the shard arch even for a generic
    compile: the object is packed for, and its evidence bound to, this shard alone.
    """
    target = ks.get("target", ROCKE_TARGET_NATIVE)
    if vk not in variant_co:
        co_path, captured, observations = compile_rocke_variant(
            ks["source"],
            ks["builder"],
            ks["spec"],
            arch,
            inter_arch_dir,
            observation_requests.get(vk, {}),
            origins,
            target,
            # The variant may serve a `sources` UKD, for which a field one launch's
            # spec lacks is simply not reported by it. The compare that follows holds
            # a single-launch UKD to every field.
            tolerate_missing=True,
        )
        variant_co[vk] = co_path
        variant_symbol[vk] = captured
        variant_observations[vk] = observations
    observations = variant_observations[vk]
    symbol = variant_symbol[vk]
    if observations.get("arch") != arch:
        raise HkpPackError(
            f"{where}: compile observations were taken for "
            f"'{observations.get('arch')}', not '{arch}'"
        )
    compile_target = rocke_compile_arch(arch, target)
    if observations.get("compile_target") != compile_target:
        raise HkpPackError(
            f"{where}: compile observations were compiled for "
            f"'{observations.get('compile_target')}', not '{compile_target}' "
            f"(kernel_source.target '{target}' on shard '{arch}')"
        )
    if observations.get("symbol") != symbol:
        raise HkpPackError(
            f"{where}: compile observations name symbol "
            f"'{observations.get('symbol')}', not '{symbol}'"
        )
    return symbol, observations, compile_target


def _launch_label(names):
    """`launch 'a'` or `launches ['a', 'b']`, for a message about one compile."""
    if len(names) == 1:
        return f"launch '{names[0]}'"
    return f"launches {sorted(names)}"


def _compile_sources_ukd(
    ukd,
    where,
    arch,
    inter_arch_dir,
    variant_co,
    variant_symbol,
    observation_requests,
    consumer_records,
    variant_observations,
    origins,
):
    """Compile every launch of a `sources` UKD; entries naming one (source,
    builder, spec, target) compile once and share a code object.

    Returns `(None, None, record_fields)`: the UKD has no variant key or symbol of
    its own, only `launches`. The claim check runs per variant the UKD uses, per
    consumer like the checks above it: the variant key does not include the roles
    a UKD claims, so each UKD is held to the compiled kernel set.
    """
    entries = sources.kernel_sources(ukd)
    by_variant = {}
    for name, ks in entries.items():
        if ks.get("kind") != "rocke":
            raise sources.refuse_authored_kind(where, name, ks.get("kind"))
        by_variant.setdefault(_source_variant_key(ks), []).append(name)
    compiled = {}
    for vk, names in by_variant.items():
        try:
            compiled[vk] = _compile_rocke_launch(
                entries[names[0]],
                vk,
                where,
                arch,
                inter_arch_dir,
                variant_co,
                variant_symbol,
                observation_requests,
                variant_observations,
                origins,
            )
        except HkpPackError as exc:
            # One declaration speaks for every launch, so a field a launch's own
            # builder cannot show fails here, and must say which launch.
            raise HkpPackError(f"{where}: {_launch_label(names)}: {exc}") from exc
        first = entries[names[0]]
        sources.check_claims(
            where,
            first["builder"],
            names,
            [entries[n].get("role") for n in names],
            compiled[vk][1],
        )
    consumers = consumer_records.get(ukd["id"], [])
    # One declaration speaks for every launch: each field must be reported by at
    # least one launch and match wherever reported.
    for record in consumers:
        try:
            agreement.compare_launches(
                record["declaration"],
                record["kmd"],
                ukd["metadata"],
                {
                    name: compiled[_source_variant_key(ks)][1]
                    for name, ks in entries.items()
                },
            )
        except HkpPackError as exc:
            raise HkpPackError(f"{where}: {exc}") from exc
    launches = {}
    for name, ks in entries.items():
        vk = _source_variant_key(ks)
        _symbol, observations, compile_target = compiled[vk]
        role = ks.get("role")
        launches[name] = LaunchEntry(
            variant_key=vk,
            symbol=sources.entry_symbol(role, observations),
            source=ks["source"],
            builder=ks["builder"],
            spec=ks["spec"],
            compile_target=compile_target,
            observations=observations,
            role=role,
        )
    fields = {
        "origin_kind": "rocke",
        "source": None,
        "entry": None,
        "build": None,
        "builder": None,
        "spec": None,
        "observations": {},
        "consumers": consumers,
        "launches": launches,
        "provenance": copy.deepcopy(ukd.get("provenance", {})),
    }
    return None, None, fields


def _compile_ukd_variant(
    ukd,
    where,
    source_root,
    rel_dir,
    arch,
    hipcc,
    inter_arch_dir,
    variant_co,
    variant_symbol,
    observation_requests,
    consumer_records,
    variant_observations,
    origins,
):
    """Compile one UKD variant for arch, deduped into variant_co per kind.

    Dispatches on kernel_source.kind — producer selection is per-UKD, never
    per-folder. hip resolves its source relative to the descriptor that named it
    (`source_root / rel_dir / source`) and keys on (source, build). rocke keys on
    (source, builder, spec) and is location-independent (its source is a dotted
    module resolved by import), so source_root/rel_dir are accepted only for
    signature uniformity. hsaco resolves its `file` the way hip resolves its
    source, keys on the file's resolved root-relative path, and runs no
    producer: the authored path itself is recorded as the variant's code object.
    Returns (variant_key, symbol, record_fields).
    """
    entries = sources.kernel_sources(ukd)
    if None not in entries:
        return _compile_sources_ukd(
            ukd,
            where,
            arch,
            inter_arch_dir,
            variant_co,
            variant_symbol,
            observation_requests,
            consumer_records,
            variant_observations,
            origins,
        )
    ks = entries[None]
    kind = ks["kind"]
    # Only the two producer kinds carry a `source`, so it is read per-arm:
    # hoisting the read above the dispatch turns the unsupported-kind raise
    # below into a KeyError.
    if kind == "hip":
        source = ks["source"]
        entry = ks["entry"]
        build = ks["build"]
        vk = _variant_key_for(ukd, rel_dir)
        if vk not in variant_co:
            variant_co[vk] = compile_hip_variant(
                hipcc,
                source_root,
                rel_dir,
                source,
                build,
                arch,
                inter_arch_dir,
            )
            variant_symbol[vk] = entry
        symbol = entry
        fields = {
            "origin_kind": "hip",
            "source": source,
            "entry": entry,
            "build": build,
            "builder": None,
            "spec": None,
        }
    elif kind == "rocke":
        source = ks["source"]
        builder = ks["builder"]
        spec = ks["spec"]
        vk = _variant_key_for(ukd, rel_dir)
        symbol, observations, compile_target = _compile_rocke_launch(
            ks,
            vk,
            where,
            arch,
            inter_arch_dir,
            variant_co,
            variant_symbol,
            observation_requests,
            variant_observations,
            origins,
        )
        # A builder returning a module also compiled its companions, which a plain
        # `kernel_source` has no way to claim: that is `sources`, and shipping only
        # the primary would leave them unreachable behind evidence naming them.
        sources.check_claims(where, builder, ["kernel_source"], [None], observations)
        # Empty only for a UKD whose pack authors no engine, which `_agreement_inputs`
        # has already established carries no catalog to disagree with.
        consumers = consumer_records.get(ukd["id"], [])
        for record in consumers:
            agreement.compare(
                record["declaration"], record["kmd"], ukd["metadata"], observations
            )
        fields = {
            "origin_kind": "rocke",
            "source": source,
            "entry": None,
            "build": None,
            "builder": builder,
            "spec": spec,
            "observations": observations,
            "compile_target": compile_target,
            "consumers": consumers,
        }
    elif kind == "hsaco":
        # The packer never reads the object's target processor; hsaco kernels declare
        # `arch` and are tested on device (descriptor-packaging/README.md,
        # "hsaco kernels and arch").
        file = ks["file"]
        symbol = ks["symbol"]
        path = resolve_hsaco_file(source_root, rel_dir, file, where)
        rel_file = hsaco_file_identity(Path(source_root).resolve(), path)
        vk = hsaco_variant_key(rel_file)
        if vk not in variant_co:
            variant_co[vk] = path
            variant_symbol[vk] = symbol
        elif variant_co[vk] != path:
            raise HkpPackError(
                f"{where}: toc_key collision: hsaco key '{vk}' already names "
                f"'{variant_co[vk]}', and this UKD resolves to '{path}'"
            )
        fields = {
            "origin_kind": "hsaco",
            "source": None,
            "entry": None,
            "build": None,
            "builder": None,
            "spec": None,
            "rel_file": rel_file,
        }
    else:
        raise HkpPackError(f"{where} kernel_source has unsupported kind '{kind}'")
    fields["provenance"] = copy.deepcopy(ukd.get("provenance", {}))
    return vk, symbol, fields


def _is_passthrough(ukd):
    """Whether a UKD ships as authored instead of through a producer."""
    return sources.kind_of(ukd) in _PASSTHROUGH_KINDS


def _dest_at(base, rel_dir, name):
    """Destination for an authored file, preserving its subpath under base.

    The authored subpath is meaningful: it scopes producers and integrations in
    the source tree, and the staged and installed trees mirror it verbatim. Two
    descriptors cannot share a path within the single source root, so the
    destination is unique by construction — no de-duplication or renaming.
    """
    dest = Path(base) / rel_dir / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest


def _write_bytes_at(base, rel_dir, name, data):
    _dest_at(base, rel_dir, name).write_bytes(data)


def _write_text_at(base, rel_dir, name, text):
    _dest_at(base, rel_dir, name).write_text(text, encoding="utf-8")


def _compact_json(doc):
    """`doc` as compact JSON, its key order kept.

    A packed KDP keeps `kernelDescriptors` as its last key (pack_arch builds it
    so): the runtime loader reads the header in the same single pass that
    streams the kernels, and falls back to a second full parse when a header key
    follows them. Sorting keys would put header keys after it: the load would
    only be slower, nothing would fail.
    """
    return json.dumps(doc, separators=(",", ":")) + "\n"


def _write_packed_at(base, rel_dir, name, doc):
    """Write one packed descriptor compact, its UKDs' provenance to its sidecar,
    and the packed marker into its directory.

    `doc` is final here: detach digests each UKD exactly as it is written, so
    nothing may change it between the two.
    """
    sidecar_name, data = provenance_sidecar.detach(name, doc)
    _dest_at(base, rel_dir, sidecar_name).write_bytes(data)
    _dest_at(base, rel_dir, provenance_sidecar.PACKED_MARKER).write_bytes(b"")
    _write_text_at(base, rel_dir, name, _compact_json(doc))


@dataclass(frozen=True)
class _VariantJob:
    """One distinct variant the prewarm compiles, as a picklable record.

    Every field is a str or a plain dict so the record crosses a process
    boundary without a custom reducer. `hipcc` and `arch` travel by value
    rather than through an environment variable or a module global, so the
    worker is a pure function of its argument.
    """

    vk: str
    kind: str
    ukd: dict
    rel_dir: str
    source_root: str
    out_dir: str
    hipcc: str
    arch: str
    requests: dict = field(default_factory=dict)
    # The launch of a `sources` UKD this variant was first needed by; None for a
    # single-launch UKD.
    launch: object = None


def shipped_engines(flat, arches):
    """The engine names each arch's shard would carry: `{arch: [name, ...]}`.

    What configure asks before wiring a pack. Uses the same survival rule as the
    walk, over the same filtered input, so a root whose every arch maps to an
    empty list is one the pack would skip, and an engine absent for an arch is one
    that shard does not ship. A shard carries what survives its own pass and what
    survives the pass of each generic that `generic_passes` merges into it. A KDP
    whose engine names no UED contributes its Id.
    """
    names = {d.id: d.doc.get("name", d.id) for d in flat.by_type("ued")}
    table = flat.generic_targets
    generics = generic_passes(flat, arches)
    return {
        arch: sorted(
            {
                names.get(k.doc.get("engine"), k.doc.get("engine"))
                for k in flat.kdps()
                if any(
                    kdp_survives(k.doc, flat, p)
                    for p in (
                        arch,
                        *(g for g in generics if arch in table.members(g)),
                    )
                )
            }
        )
        for arch in arches
    }


def offered_engines(flat):
    """The engine names `flat` carries for at least one arch, sorted.

    The answer is independent of the build's arches: it is what host-side gates
    read, since the host must not depend on which arch content a build packs. A
    KDP is offered iff it survives some arch named anywhere under `flat` (a KDP or
    UKD `arch` list, inline or standalone; a generic's own name is such an arch) or
    the wildcard sentinel `"*"`, which `arch_matches` accepts only for wildcards.
    Names are built as in `shipped_engines`, and `shipped_engines(flat, arches)`
    values are always a subset of this list.
    """
    candidates = {"*"}
    ukd_docs = [d.doc for d in flat.ukds()]
    for k in flat.kdps():
        candidates.update(k.doc.get("arch") or ())
        ukd_docs.extend(
            e for e in k.doc.get("kernelDescriptors", []) if isinstance(e, dict)
        )
    for doc in ukd_docs:
        candidates.update(doc.get("arch") or ())
    names = {d.id: d.doc.get("name", d.id) for d in flat.by_type("ued")}
    return sorted(
        {
            names.get(k.doc.get("engine"), k.doc.get("engine"))
            for k in flat.kdps()
            if any(kdp_survives(k.doc, flat, a) for a in sorted(candidates))
        }
    )


def _agreement_inputs(flat, arch):
    """Every consumer's declaration and observation request, before any compile.

    The requests belong to the whole selected set, since two variants of one
    builder share a compile result and two KDPs can reference one standalone UKD;
    collecting up front lets a single pass over the builder object capture every
    consumer's readouts. References resolve by UUID. Returns
    `({ukd id: [consumer record]}, {variant key: {digest: request}})`. A KDP
    authoring `engine` as null leaves an EMPTY obligation, not a waived one, so a
    contract declared under it is rejected.
    """
    generics = flat.generic_by_id()
    # Keyed as authored: specialization consumers name their KMD by that spelling.
    schemas = {d.id: d.doc for d in generics.values() if d.type == "kmd"}
    ukd_by_id = flat.ukd_by_id()
    records, requests = {}, {}
    for kdp in flat.kdps():
        engine_id = kdp.doc["engine"]
        if engine_id is None:
            if agreement.resolved_contract(kdp.doc) is not None:
                raise HkpPackError(
                    f"KDP {kdp.path.name}: a specialization contract names an "
                    "engine, and this KDP authors none"
                )
            # The KDP declares nothing (just checked), so there is nothing for a
            # kernel to inherit and each speaks only for itself.
            for _sid, ukd, _sdesc in _selected_entries(
                kdp.doc, arch, ukd_by_id, flat.generic_targets
            ):
                if agreement.resolved_contract(ukd) is not None:
                    raise HkpPackError(
                        f"UKD {ukd['id']} in {kdp.path.name}: a specialization "
                        "contract names an engine, and this KDP authors none"
                    )
            continue
        if canonical_id(engine_id) not in generics:
            raise HkpPackError(
                f"KDP {kdp.path.name}: engine '{engine_id}' resolves to no descriptor"
            )
        engine = generics[canonical_id(engine_id)].doc
        kmd_id = engine.get("metadata")
        kmd_desc = generics.get(canonical_id(kmd_id))
        if kmd_desc is None or kmd_desc.type != "kmd":
            raise HkpPackError(
                f"engine '{engine_id}': metadata '{kmd_id}' resolves to no KMD"
            )
        kmd = kmd_desc.doc
        header = _kdp_header(kdp.doc)
        header["arch"] = [arch]
        for sid, ukd, sdesc in _selected_entries(
            kdp.doc, arch, ukd_by_id, flat.generic_targets
        ):
            # Completion is checked against the KMD the chain actually resolved to,
            # so a mis-typed or missing mandatory value fails before a compile.
            agreement.complete_metadata(ukd["metadata"], kmd)
            kind = sources.kind_of(ukd)
            # A passthrough kind runs no producer, so there is no producing compiler
            # whose specialization a contract could state.
            if kind in _PASSTHROUGH_KINDS:
                continue
            # A standalone UKD is its own file and several KDPs may reference it,
            # so it inherits from none of them and states its own declaration.
            enclosing = kdp.doc if sid is None else None
            declaration = agreement.select_declaration(
                ukd, engine, kmd, schemas, enclosing
            )
            if kind != "rocke" and declaration["metadata_fields"]:
                raise HkpPackError(
                    f"UKD {ukd['id']}: a '{kind}' source cannot fulfil compiled "
                    f"specialization bindings for {sorted(declaration['metadata_fields'])}; "
                    "declare them matcher-only or supply a compiling source"
                )
            records.setdefault(ukd["id"], []).append(
                agreement.consumer_record(ukd, engine, kmd, header, arch, declaration)
            )
            if kind != "rocke":
                continue
            rel_dir = sdesc.rel_dir if sid is not None else kdp.rel_dir
            request = agreement.observation_request(declaration, kmd)
            # Every launch's compile observes the same request: one declaration
            # speaks for the whole UKD, so a field a launch's builder cannot show
            # fails that compile loudly rather than being exempted.
            for vk, _name, _launch in _launch_variants(ukd, rel_dir):
                requests.setdefault(vk, {})[agreement.digest(request)] = request
    return (
        {uid: agreement.canonical_records(entries) for uid, entries in records.items()},
        requests,
    )


def _prewarm_jobs(flat, source_root, arch, observation_requests=None):
    """The distinct variant jobs the walk will compile.

    Selection comes from the generator the walk consumes, and dedup on the variant
    key keeps first-seen (walk) order. A kind `_variant_key_for` declines is dropped
    here, leaving the walk to report it; `out_dir` and `hipcc` are filled in by
    `_prewarm_variants`. hsaco is declined deliberately: it has no compile, so it
    yields no job, and the walk keys it via `hsaco_variant_key`.
    """
    ukd_by_id = flat.ukd_by_id()
    if observation_requests is None:
        _, observation_requests = _agreement_inputs(flat, arch)
    jobs = []
    seen = set()
    for kdp in flat.kdps():
        for sid, ukd, sdesc in _selected_entries(
            kdp.doc, arch, ukd_by_id, flat.generic_targets
        ):
            rel_dir = sdesc.rel_dir if sid is not None else kdp.rel_dir
            for vk, name, launch in _launch_variants(ukd, rel_dir):
                if vk is None or vk in seen:
                    continue
                seen.add(vk)
                jobs.append(
                    _VariantJob(
                        launch=name,
                        vk=vk,
                        kind=launch["kernel_source"]["kind"],
                        ukd=launch,
                        rel_dir=Path(rel_dir).as_posix(),
                        source_root=str(source_root),
                        out_dir="",
                        hipcc="",
                        arch=arch,
                        requests=observation_requests.get(vk, {}),
                    )
                )
    return jobs


def _compile_one_variant(job):
    """Compile one variant in a worker process, returning a result tuple.

    `(vk, co_path, symbol, None, observations, origins)` on success,
    `(vk, None, None, "Name: text", None, {})` on any failure.
    Failures are returned rather than raised: rocke and comgr
    exceptions are not guaranteed picklable, and an exception that cannot cross
    the process boundary takes the diagnosis with it.

    `observations` carries the compiler observations checked for every consumer.
    `origins` is this variant's producer file identities in picklable form, the
    only route to `OriginObserver.absorb`.

    No key is computed here. `job.vk` was computed in the parent, under
    whatever key functions were in force there; a key recomputed in the child
    would resolve the real functions and disagree with the walk.
    """
    origins = agreement.OriginObserver()
    try:
        observations = {}
        ks = job.ukd["kernel_source"]
        if job.kind == "hip":
            co_path = compile_hip_variant(
                job.hipcc,
                job.source_root,
                job.rel_dir,
                ks["source"],
                ks["build"],
                job.arch,
                job.out_dir,
            )
            # The hip symbol is authored, not captured from the artifact, which
            # is why it is read here rather than returned by the producer.
            symbol = ks["entry"]
        elif job.kind == "rocke":
            co_path, symbol, observations = compile_rocke_variant(
                ks["source"],
                ks["builder"],
                ks["spec"],
                job.arch,
                job.out_dir,
                job.requests,
                origins,
                ks.get("target", ROCKE_TARGET_NATIVE),
                tolerate_missing=True,
            )
        else:
            # Unreachable while `_variant_key_for` keys only these two kinds. A
            # raise rather than a fall-through, because a third keyable kind
            # added there would otherwise compile as rocke and yield a plausible
            # artefact from the wrong producer.
            raise HkpPackError(
                f"no variant compiler for kernel source kind '{job.kind}'"
            )
    except Exception as exc:
        return job.vk, None, None, f"{type(exc).__name__}: {exc}", None, {}
    return job.vk, str(co_path), symbol, None, observations, origins.exported()


def _cgroup_v2_cpu_quota():
    """Whole CPUs allowed by cgroup v2, or None when unlimited or absent.

    Linux only; the other probes in `_cpu_budget` bound the pool elsewhere.
    Read from the filesystem because no Python API reports it. A container
    given `--cpus=8` keeps a full affinity mask and is throttled instead, so
    this is the only source that sees the limit at all.

    Walked from this process's own cgroup up to the root, taking the tightest
    limit: where the limit sits depends on whether a cgroup namespace is in
    play, not on which framework imposed it. With one (Docker, Kubernetes) the
    container's cgroup appears as the root and `/sys/fs/cgroup/cpu.max` carries
    the limit; without one (Slurm, a systemd login session) the process sits in
    a nested scope whose root has no `cpu.max` at all, so reading only the root
    would report "unlimited" for every limited job on such a host.
    """
    root = Path("/sys/fs/cgroup")
    try:
        # The v2 line is `0::<path>`; a v1-only host has no such line.
        rel = next(
            line[3:].strip().lstrip("/")
            for line in Path("/proc/self/cgroup").read_text().splitlines()
            if line.startswith("0::")
        )
    except (OSError, StopIteration):
        return None

    quotas = []
    node = root.joinpath(rel) if rel else root
    while True:
        quotas.append(_read_cpu_max(node / "cpu.max"))
        if node == root or node.parent == node:
            break
        node = node.parent

    quotas = [q for q in quotas if q]
    return min(quotas) if quotas else None


def _read_cpu_max(path):
    """Whole CPUs from one `cpu.max` file, or None when unlimited or absent."""
    try:
        quota, period = path.read_text().split()
    except (OSError, ValueError):
        return None
    if quota == "max":
        return None
    try:
        # Floors to at least one: a sub-CPU allocation would otherwise disable
        # the pool entirely.
        return max(1, int(quota) // int(period))
    except (ValueError, ZeroDivisionError):
        return None


def _cpu_budget():
    """CPUs this process may actually use, rather than the host's core count.

    `os.cpu_count()` reports the machine, but a containerised or scheduled pack
    -- Docker, Kubernetes and Slurm alike -- is usually given a slice of it, and
    one worker per host core oversubscribes that slice badly. Each source below
    sees a limit the others cannot, so the smallest wins.
    """
    limits = [_cgroup_v2_cpu_quota()]

    process_cpu_count = getattr(os, "process_cpu_count", None)
    if process_cpu_count is not None:
        # 3.13+, and honours the affinity mask on every platform that has one.
        limits.append(process_cpu_count())
    elif hasattr(os, "sched_getaffinity"):
        # Linux only, so it cannot be the only affinity source -- this packer
        # runs on Windows too, where the host count below is all there is.
        limits.append(len(os.sched_getaffinity(0)))

    limits = [limit for limit in limits if limit]
    return min(limits) if limits else (os.cpu_count() or 1)


def _pack_jobs():
    """Worker count for the prewarm, from `HKP_PACK_JOBS`.

    Unset means `min(32, _cpu_budget())`. The cap is per-worker startup, not
    memory: a worker costs roughly 250 ms to stand up its rocke+comgr import,
    so past a few dozen the next worker costs more than the compile work it
    takes on. Measured on gfx942, a 1672-variant pack bottoms out near 48
    workers -- 19.5 s against 402 s serial -- and by 128 is back to its
    16-worker time, while peaking at 2.2 GiB throughout.

    32 is a compromise, not an optimum: the turnover moves with pack size, and
    the same host wants about 12 workers for 200 variants and 48 for 1672.
    Overshooting costs far more than undershooting -- 200 variants take 6.8 s
    on 12 workers and 16.9 s on 32 -- so the constant sits below the large-pack
    optimum deliberately.

    `HKP_PACK_JOBS=1` forces serial execution: the prewarm returns without a
    pool and the walk compiles every variant itself, which is the escape hatch
    for debugging a compile failure with a single clean traceback.

    Anything that is not an integer of 1 or more is a hard error, including `0`
    and negatives. Neither is clamped, because clamping lands them on the serial
    path: `-4` would read as "compiled on 4 workers" and run on one, and `0`
    means "auto" elsewhere in this repository while it would mean "serial" here.

    An explicit value is never reduced to `_cpu_budget()` -- choosing to
    oversubscribe is the caller's call to make.

    An environment variable rather than a CLI flag or a CMake cache variable
    because it reaches through the CMake custom command with no plumbing.
    """
    env = os.environ.get("HKP_PACK_JOBS")
    if env is None:
        return min(32, _cpu_budget())
    try:
        jobs = int(env)
    except ValueError:
        raise HkpPackError(f"HKP_PACK_JOBS must be an integer, got '{env}'") from None
    if jobs < 1:
        raise HkpPackError(f"HKP_PACK_JOBS must be 1 or greater, got {jobs}")
    return jobs


def _variant_key_for(ukd, rel_dir):
    """Content-hash variant key, or None for a kind the prewarm skips.

    The single place the hip and rocke key functions are called, so the walk and
    the prewarm cannot key one variant two ways.

    `hip_variant_key` and `rocke_variant_key` are resolved as globals of this
    module on every call -- never through a function-local import, an alias
    bound at import time, or a recomputation inside a worker process. Each of
    those shortcuts binds the real function past any substitution made on this
    module.

    A kind that carries no compilable source yields None rather than raising.
    The walk stays the sole reporter of whatever failure such a UKD produces:
    were this to raise, the prewarm would pre-empt it with an error of its own,
    from a different call site and with a different traceback.

    hsaco is deliberately None as well: it has no compile, so there is no
    prewarm job, and the walk keys it via `hsaco_variant_key`.
    """
    ks = sources.kernel_sources(ukd).get(None)
    if ks is None:
        raise HkpPackError(
            f"UKD '{ukd.get('id', '?')}' has sources, which key one variant per "
            "launch; key each launch's own kernel_source instead"
        )
    return _source_variant_key(ks, rel_dir)


def _source_variant_key(ks, rel_dir=None):
    """The variant key of one kernel_source: hip needs the descriptor's `rel_dir`."""
    kind = ks["kind"]
    if kind == "hip":
        return hip_variant_key(hip_source_relpath(rel_dir, ks["source"]), ks["build"])
    if kind == "rocke":
        return rocke_variant_key(
            ks["source"],
            ks["builder"],
            ks["spec"],
            ks.get("target", ROCKE_TARGET_NATIVE),
        )
    return None


def _launch_variants(ukd, rel_dir):
    """`(variant key, launch name, single-launch UKD document)` per launch of a UKD.

    A single-launch UKD yields itself. Each entry of `sources` yields the UKD
    with that entry as its `kernel_source`, minus the authored-only `role`, which
    is no part of what gets compiled. Entries on one compiled module repeat a key.
    """
    for name, ks in sources.kernel_sources(ukd).items():
        if name is None:
            yield _variant_key_for(ukd, rel_dir), None, ukd
            continue
        doc = {k: v for k, v in ukd.items() if k != "sources"}
        doc["kernel_source"] = {k: v for k, v in ks.items() if k != "role"}
        yield _source_variant_key(doc["kernel_source"], rel_dir), name, doc


def _prewarm_variants(
    flat,
    source_root,
    arch,
    hipcc,
    inter_arch_dir,
    variant_co,
    variant_symbol,
    variant_observations,
    observation_requests,
    log=print,
):
    """Compile this arch's distinct variants concurrently into the caches.

    The walk then finds each key already present and skips the expensive call.
    Records, symbols and doc rewriting stay entirely the walk's: this fills the
    code-object, symbol and observation caches. Returns one producer-origin map
    per compiled variant, which the caller must fold into the observer spanning
    the arch, since a prewarmed variant is observed nowhere else.

    Fails fast, matching the serial path: the first failing variant in walk
    order raises and the queued jobs are cancelled, so a broken builder costs
    the jobs already in flight rather than the whole pack. Nothing is written
    into any cache when that happens -- a partly-filled cache would let the
    walk skip compiles whose artefacts were never produced.
    """
    jobs = [
        replace(job, out_dir=str(inter_arch_dir), hipcc=str(hipcc))
        for job in _prewarm_jobs(flat, source_root, arch, observation_requests)
    ]

    workers = _pack_jobs()
    if len(jobs) < 2 or workers < 2:
        # Nothing was compiled here, so the walk observes every producer against
        # the caller's own observer.
        return []
    workers = min(workers, len(jobs))
    log(f"hkp_pack: compiling {len(jobs)} variants for {arch} on {workers} workers")

    # The pool is built here, per arch, and never kept in a module global.
    # Workers snapshot `sys.path` at process start, so a pool reused across
    # runs would freeze whichever path list existed when it was first built and
    # break imports the parent has since arranged.
    pool = ProcessPoolExecutor(max_workers=workers)
    results = []
    failure = None
    try:
        # Consumed lazily rather than collected with `list`, so the first
        # failure stops the pack instead of riding out every remaining job.
        # `map` yields in submission order, which is walk order, so the variant
        # this stops on is the one the serial path would have named -- breaking
        # on the first result to *complete* would hand that choice to the
        # scheduler.
        for result in pool.map(_compile_one_variant, jobs, chunksize=1):
            if result[3] is not None:
                failure = result
                break
            results.append(result)
    except Exception as exc:
        # A worker that is killed rather than raising surfaces here, and
        # the exception is not an HkpPackError -- so unconverted it would
        # sail past run_pipeline's per-arch handler and discard every other
        # arch's work.
        raise HkpPackError(
            f"the variant compile pool for {arch} failed "
            f"({type(exc).__name__}: {exc}); a worker process died -- an OOM "
            "kill, a crash inside the toolchain, or an interrupt. Retrying "
            "with a lower HKP_PACK_JOBS addresses the first of those"
        ) from exc
    finally:
        # `map` submits every job up front, so the queue outlives the break.
        # Finalising its result generator already cancels the pending futures;
        # `cancel_futures` restates that next to the `wait` it qualifies. Jobs
        # already running finish either way, bounding the waste at the worker
        # count rather than the pack size.
        pool.shutdown(wait=True, cancel_futures=True)

    if failure is not None:
        # One variant named, no failure count: the pack stops at the first
        # failure, so a count would describe how far the pool happened to get
        # rather than how many variants are broken. Named by UKD id, which is
        # what a reader can look up in the descriptors; the content-hash key is
        # not.
        failed = next((job for job in jobs if job.vk == failure[0]), None)
        named = None if failed is None else failed.ukd.get("id")
        launch = (
            ""
            if failed is None or failed.launch is None
            else (f" launch '{failed.launch}'")
        )
        raise HkpPackError(
            f"variant '{named or failure[0]}'{launch} failed to compile for {arch}: "
            f"{failure[3]}"
        )

    origins = []
    for vk, co_path, symbol, _err, observations, variant_origins in results:
        variant_co[vk] = Path(co_path)
        variant_symbol[vk] = symbol
        variant_observations[vk] = observations
        origins.append(variant_origins)
    return origins


def compile_intermediate(flat, source_root, arch, hipcc, inter_arch_dir, log=print):
    """Compile every hip and rocke UKD in the KDPs targeting arch and stage a per-arch tree.

    Writes inter_arch_dir with: hsaco-form KDP JSON (inline UKDs rewritten
    hip->hsaco, build lifted to top-level) + one .co per distinct (source,build)
    variant + every generic copied through + any non-matching KDP copied in its
    authored hip form (so pruning has a KDP to drop). Standalone UKDs a surviving
    KDP references by Id are compiled here too and tracked per arch, to be
    emitted as their own files by pack_arch. Returns an IntermediateArch carrying
    the origin data the pack step needs for provenance.

    A UKD of a pass-through kind runs no producer. It stays in the intermediate
    JSON as authored and is recorded for verbatim emission.

    An hsaco UKD runs no producer either and keeps its authored form in the
    intermediate JSON. No copy of its code object is staged: its bytes are read
    from the authored file at pack time.
    """
    inter_arch_dir = Path(inter_arch_dir)
    inter_arch_dir.mkdir(parents=True, exist_ok=True)

    variant_co = {}
    variant_symbol = {}
    variant_observations = {}
    origins = agreement.OriginObserver()
    consumer_records, observation_requests = _agreement_inputs(flat, arch)
    arch_kdps = []
    standalone_ukds = {}
    passthrough_standalone_ukds = {}
    ukd_by_id = flat.ukd_by_id()

    prewarmed_origins = _prewarm_variants(
        flat,
        source_root,
        arch,
        hipcc,
        inter_arch_dir,
        variant_co,
        variant_symbol,
        variant_observations,
        observation_requests,
        log,
    )
    for variant_origins in prewarmed_origins:
        origins.absorb(variant_origins)

    for kdp in flat.kdps():
        doc = kdp.doc
        if not kdp_arch_matches(doc, arch, flat.generic_targets):
            _write_bytes_at(
                inter_arch_dir, kdp.rel_dir, kdp.path.name, kdp.path.read_bytes()
            )
            continue

        new_doc = copy.deepcopy(doc)
        ukds = []
        entries = []
        new_kds = []
        for sid, entry, sdesc in _selected_entries(
            new_doc, arch, ukd_by_id, flat.generic_targets
        ):
            if sid is not None:
                # A reference to a standalone UKD: compile it once per arch and
                # keep the string in the KDP; it ships as its own file. Listing
                # it precedes the compile-once check, so a UKD two KDPs
                # reference appears in both and is compiled for the first only.
                entries.append(sid)
                new_kds.append(sid)
                key = canonical_id(sid)
                if key in passthrough_standalone_ukds:
                    continue
                sukd = entry
                where = f"standalone UKD {sdesc.path.name}"
                if _is_passthrough(sukd):
                    kind = sources.kind_of(sukd)
                    log(f"{where}: emitting kind '{kind}' as authored")
                    passthrough_standalone_ukds[key] = PassthroughUKD(
                        doc=copy.deepcopy(sukd),
                        filename=sdesc.path.name,
                        rel_dir=sdesc.rel_dir,
                    )
                    continue
                vk, symbol, fields = _compile_ukd_variant(
                    sukd,
                    where,
                    source_root,
                    sdesc.rel_dir,
                    arch,
                    hipcc,
                    inter_arch_dir,
                    variant_co,
                    variant_symbol,
                    observation_requests,
                    consumer_records,
                    variant_observations,
                    origins,
                )
                if key in standalone_ukds:
                    continue
                standalone_ukds[key] = StandaloneUKD(
                    id=sukd.get("id"),
                    name=sukd.get("name"),
                    metadata=sukd.get("metadata"),
                    priority=sukd.get("priority"),
                    symbol=symbol,
                    variant_key=vk,
                    extra=_ukd_extra(sukd),
                    filename=sdesc.path.name,
                    rel_dir=sdesc.rel_dir,
                    **fields,
                )
                continue

            ukd = entry
            where = f"UKD '{ukd.get('id')}' in {kdp.path.name}"
            if _is_passthrough(ukd):
                kind = sources.kind_of(ukd)
                log(f"{where}: emitting kind '{kind}' as authored")
                new_kds.append(ukd)
                entries.append(PassthroughUKD(doc=ukd, rel_dir=kdp.rel_dir))
                continue
            vk, symbol, fields = _compile_ukd_variant(
                ukd,
                where,
                source_root,
                kdp.rel_dir,
                arch,
                hipcc,
                inter_arch_dir,
                variant_co,
                variant_symbol,
                observation_requests,
                consumer_records,
                variant_observations,
                origins,
            )
            if fields.get("launches"):
                ukd["sources"] = {
                    name: {
                        "kind": "hsaco",
                        "file": f"{launch.variant_key}.co",
                        "symbol": launch.symbol,
                    }
                    for name, launch in fields["launches"].items()
                }
            elif fields["origin_kind"] != "hsaco":
                ukd["kernel_source"] = {
                    "kind": "hsaco",
                    "file": f"{vk}.co",
                    "symbol": symbol,
                }
            if fields["build"] is not None:
                ukd["build"] = fields["build"]
            new_kds.append(ukd)
            record = InlineUKD(
                id=ukd.get("id"),
                name=ukd.get("name"),
                metadata=ukd.get("metadata"),
                priority=ukd.get("priority"),
                symbol=symbol,
                variant_key=vk,
                extra=_ukd_extra(ukd),
                **fields,
            )
            ukds.append(record)
            entries.append(record)
        # A KDP whose UKDs all filter out for this arch is dropped from the
        # shard: no intermediate JSON, no record, and its exclusive generics
        # prune away with it.
        if not new_kds:
            log(f"KDP {kdp.path.name}: all UKDs filtered out for {arch}, dropping")
            continue
        new_doc["kernelDescriptors"] = new_kds
        _write_text_at(
            inter_arch_dir,
            kdp.rel_dir,
            kdp.path.name,
            _compact_json(new_doc),
        )
        arch_kdps.append(
            ArchKDP(
                id=doc.get("id"),
                filename=kdp.path.name,
                header=_kdp_header(doc),
                rel_dir=kdp.rel_dir,
                ukds=ukds,
                entries=entries,
            )
        )

    for generic in flat.generics():
        _write_bytes_at(
            inter_arch_dir,
            generic.rel_dir,
            generic.path.name,
            generic.path.read_bytes(),
        )

    # Sidecars are copied unpruned like their generics; pruning happens in pack_arch.
    for sidecar in flat.sidecars_for(flat.generics()):
        _write_bytes_at(
            inter_arch_dir,
            sidecar.rel_dir,
            sidecar.name,
            sidecar.source.read_bytes(),
        )

    origins.stable()
    return IntermediateArch(
        arch=arch,
        directory=inter_arch_dir,
        kdps=arch_kdps,
        variant_co=variant_co,
        variant_symbol=variant_symbol,
        standalone_ukds=standalone_ukds,
        passthrough_standalone_ukds=passthrough_standalone_ukds,
    )


@dataclass
class PruneResult:
    surviving_kdp_ids: set
    reachable_generic_ids: set


def prune(flat, arch):
    """Compute the surviving KDP and generic Ids for arch (wildcard-aware)."""
    surviving = [k for k in flat.kdps() if kdp_survives(k.doc, flat, arch)]
    return PruneResult(
        surviving_kdp_ids={k.id for k in surviving},
        reachable_generic_ids=reachable_generic_ids(flat, surviving),
    )


def _rewrite_ukd_kpack(
    ukd,
    arch,
    toc_key,
    sha256,
    *,
    signature,
    packed_launches=None,
    toolchain_fields=None,
    rel_dir=Path("."),
    group=GROUP_NAME,
):
    """Rewrite a compiled UKD into shipped kpack form.

    A UKD with `launches` ships `sources` instead of `kernel_source`: each entry a
    full kpack kernel_source (no `role`), its symbol read off the compiled launch
    and its `toc_key`, `sha256` and `signature` from `packed_launches`
    (`{launch: {toc_key, sha256, signature}}`, read off the object). Then
    `toc_key`, `sha256` and `signature` above are unused, and the UKD reads at
    document version 1.1.

    `toolchain_fields` carries the fields describing what actually produced the kernel
    (hipcc version, resolved comgr, rocKE wheel digest) as opposed to what the
    descriptor asked for. Merged into provenance rather than the variant key: in
    the key, a wheel bump would rename every rocKE artifact including ones it
    could not affect.

    An hsaco UKD records the authored file, its digest and its symbol.
    """
    if "effective_spec" in ukd.provenance:
        raise HkpPackError(
            f"UKD '{ukd.id}': authored input supplies provenance.effective_spec, "
            "which only the producing compiler creates"
        )
    if ukd.launches:
        provenance = {
            "origin_kind": "rocke",
            # Per launch, since launches may differ in builder and spec. `role` is
            # recorded only where authored: it is what binds an entry's symbol to
            # its own kernel in the compiled module.
            "sources": {
                name: {
                    "source": launch.source,
                    "builder": launch.builder,
                    "spec": launch.spec,
                    "compile_target": launch.compile_target,
                    **({} if launch.role is None else {"role": launch.role}),
                }
                for name, launch in ukd.launches.items()
            },
        }
    elif ukd.origin_kind == "rocke":
        provenance = {
            "origin_kind": "rocke",
            "source": ukd.source,
            "builder": ukd.builder,
            "spec": ukd.spec,
            # The processor the shipped object was built for: the shard arch for a
            # native compile, the family's generic processor for a generic one.
            "compile_target": ukd.compile_target,
        }
    elif ukd.origin_kind == "hsaco":
        provenance = {
            "origin_kind": "hsaco",
            "file": ukd.rel_file,
            "sha256": sha256,
            "symbol": ukd.symbol,
        }
    else:
        provenance = {
            "origin_kind": "hip",
            "source": ukd.source,
            "entry": ukd.entry,
            "build": ukd.build,
        }
    provenance = {**ukd.provenance, **provenance}
    if toolchain_fields:
        provenance.update(toolchain_fields)
    library = _kpack_rel(arch, rel_dir, group)
    if ukd.launches:
        kernel_key = "sources"
        kernel = {
            name: {
                "kind": "kpack",
                "library": library,
                "toc_key": packed_launches[name]["toc_key"],
                "symbol": launch.symbol,
                "sha256": packed_launches[name]["sha256"],
                "signature": packed_launches[name]["signature"],
            }
            for name, launch in ukd.launches.items()
        }
    else:
        kernel_key = "kernel_source"
        kernel = {
            "kind": "kpack",
            "library": library,
            "toc_key": toc_key,
            "symbol": ukd.symbol,
            "sha256": sha256,
            "signature": signature,
        }
    doc = {
        "id": ukd.id,
        "name": ukd.name,
        kernel_key: kernel,
        "metadata": ukd.metadata,
        "priority": ukd.priority,
        "provenance": provenance,
    }
    if set(doc) & ukd.extra.keys():
        raise HkpPackError(
            f"UKD '{ukd.id}': authored extra names produced UKD field(s) "
            f"{sorted(set(doc) & ukd.extra.keys())}"
        )
    doc.update(ukd.extra)
    if ukd.launches and not sources.reads_sources(doc["version"]):
        # Before `publish`, so the digest binds the version that ships.
        doc["version"] = sources.SOURCES_VERSION
    # Every shipped UKD carries the single shard arch, matching the KDP. Set it
    # after the extra passthrough so a source multi-arch list can't leak through.
    doc["arch"] = [arch]
    if ukd.origin_kind == "rocke":
        # Published last, onto the finished document, so `descriptor_digest` binds
        # the bytes that actually ship.
        observations = (
            {name: launch.observations for name, launch in ukd.launches.items()}
            if ukd.launches
            else ukd.observations
        )
        agreement.publish(doc, observations, ukd.consumers)
    return doc


def _rewrite_passthrough_ukd(passthrough, arch, source_label=None):
    """Rewrite a pass-through UKD into shipped form.

    One field may change: `arch` narrows to the shard. `kernel_source` ships as
    authored, so the `source_file` the descriptor names is the key the consuming
    binary's source table is keyed on.

    The provenance block records the authored values for a person reading the
    build output. `rewritten` names the fields whose emitted value differs from
    the authored one. It holds nothing machine-specific and nothing
    time-varying, so two runs over one source tree write the same bytes. Nothing
    at runtime reads it. The caller moves it to the descriptor's provenance
    sidecar, because `kernel_source` accepts only the keys the loader parses.
    """
    authored = passthrough.doc
    rel_dir = Path(passthrough.rel_dir).as_posix()
    kernel_source = authored["kernel_source"]
    source_file = kernel_source["source_file"]
    if not source_label:
        raise HkpPackError(
            "source_label is required for pass-through descriptor "
            f"'{rel_dir}/{source_file}': pass --source-label (or source_label=) "
            "naming the build rule that packs this root."
        )
    authored_arch = list(authored.get("arch") or [])

    doc = {k: v for k, v in authored.items() if k != "provenance"}
    doc["kernel_source"] = dict(kernel_source)

    rewritten = []
    if authored_arch != [arch]:
        rewritten.append("arch")

    provenance = {
        "origin_kind": kernel_source["kind"],
        "source_label": source_label,
    }
    provenance.update(
        {
            "rel_dir": rel_dir,
            "source_file": source_file,
            "authored_arch": authored_arch,
            "rewritten": rewritten,
        }
    )
    doc["provenance"] = provenance
    doc["arch"] = [arch]
    return doc


def _toolchain_for(ukd, hipcc, rocke_wheel_stamp):
    """Toolchain provenance for one UKD, dispatched on its producer.

    A prebuilt hsaco object carries no toolchain claim: nothing here produced it.
    """
    if ukd.origin_kind == "hsaco":
        return None
    if ukd.origin_kind == "rocke":
        return toolchain.rocke_provenance(rocke_wheel_stamp)
    return toolchain.hip_provenance(hipcc)


def pack_arch(
    flat,
    inter,
    out_arch_dir,
    kpack_mod,
    comp,
    expected_sha256=None,
    hipcc=None,
    rocke_wheel_stamp=None,
    group=GROUP_NAME,
    source_label=None,
):
    """Pack a pruned intermediate arch into the shipped kpack release tree.

    Each distinct (source,build) variant .co staged by compile_intermediate is
    packed once under its own toc_key; an authored hsaco UKD is packed from its
    own file instead. Both are rewritten to kpack; inline compiled UKDs go
    hsaco->kpack, stamping toc_key + sha256 + signature and moving build into
    provenance. Guarded against toc_key collisions (distinct inputs mapping to
    one key).

    A UKD of a pass-through kind takes the shard arch, keeps its authored
    kernel_source, and gets a provenance block naming its authored values. No
    packed UKD carries provenance inline: each descriptor file's UKD provenance
    ships in a sidecar beside the file (see provenance_sidecar). A shard with no
    compiled variant holds no archive and no `kpack/` directory, and its
    ArchResult carries kpack_path=None.

    An hsaco UKD's bytes are read from its authored file and packed as-is.
    """
    arch = inter.arch
    out_arch_dir = Path(out_arch_dir)
    out_arch_dir.mkdir(parents=True, exist_ok=True)

    standalone = list(inter.standalone_ukds.values())

    def _all_ukds():
        for kdp in inter.kdps:
            for ukd in kdp.ukds:
                yield ukd
        for ukd in standalone:
            yield ukd

    def _units(ukd):
        """`(who, variant key, symbol, input signature)` per compiled kernel.

        The signature must cover everything that determines the compiled bytes,
        per producer. Keying on (source, build) alone is blind on the rocke path,
        where build is ALWAYS None: two rocke UKDs sharing a source module but
        differing in builder or spec would present identical signatures, so a
        genuine toc_key collision would pass undetected and one kernel would
        silently ship the other's bytes -- the same silent-substitution class as
        the cross-root collision this work removed. An hsaco UKD's bytes are its
        authored file, so the file's root-relative identity is its signature.
        """
        who = f"UKD '{ukd.id}'"
        if ukd.launches:
            return [
                (
                    f"{who} launch '{name}'",
                    launch.variant_key,
                    launch.symbol,
                    (
                        launch.source,
                        launch.builder,
                        json.dumps(launch.spec, sort_keys=True),
                        launch.compile_target,
                    ),
                )
                for name, launch in ukd.launches.items()
            ]
        if ukd.origin_kind == "rocke":
            sig = (
                ukd.source,
                ukd.builder,
                json.dumps(ukd.spec, sort_keys=True),
                ukd.compile_target,
            )
        elif ukd.origin_kind == "hsaco":
            sig = ("hsaco", ukd.rel_file)
        else:
            sig = (ukd.source, json.dumps(ukd.build, sort_keys=True))
        return [(who, ukd.variant_key, ukd.symbol, sig)]

    variant_bytes = {}
    variant_sha = {}
    variant_signature = {}
    variant_source_build = {}
    for ukd in _all_ukds():
        for who, vk, symbol, sig in _units(ukd):
            toc_key = vk
            if vk in variant_source_build and variant_source_build[vk] != sig:
                raise HkpPackError(
                    f"toc_key collision: '{vk}' maps to two distinct "
                    f"inputs {variant_source_build[vk]} and {sig}"
                )
            variant_source_build[vk] = sig
            if vk not in variant_bytes:
                try:
                    data = inter.variant_co[vk].read_bytes()
                except OSError as exc:
                    if ukd.origin_kind != "hsaco":
                        raise
                    raise HkpPackError(
                        f"UKD '{ukd.id}': cannot read hsaco file "
                        f"'{ukd.rel_file}': {exc}"
                    ) from exc
                digest = _sha256(data)
                if expected_sha256 and toc_key in expected_sha256:
                    if digest != expected_sha256[toc_key]:
                        raise HkpPackError(
                            f"sha256 mismatch for toc_key '{toc_key}': expected "
                            f"{expected_sha256[toc_key]}, packed blob is {digest}"
                        )
                variant_bytes[vk] = data
                variant_sha[vk] = digest
            if symbol.encode("ascii") not in variant_bytes[vk]:
                raise HkpPackError(
                    f"{who} declares symbol '{symbol}' not present "
                    f"in code object for variant '{vk}'"
                )
            # Keyed on (variant, symbol), not on the variant alone: two UKDs
            # differing only by entry point share one blob and one toc_key, and
            # each has its own argument list. Caching per variant would give the
            # second one the first's signature, which no fixture with one symbol
            # per variant can catch. The launches of one `sources` UKD are that
            # case by construction.
            signature_key = (vk, symbol)
            if signature_key not in variant_signature:
                if ukd.origin_kind == "hsaco":
                    # Authored bytes are arbitrary: a truncated or corrupt object
                    # fails the metadata parse with a parser-specific exception.
                    try:
                        variant_signature[signature_key] = kernel_signature(
                            variant_bytes[vk], symbol, who
                        )
                    except HkpPackError:
                        raise
                    except Exception as exc:
                        raise HkpPackError(
                            f"{who}: cannot read the AMDGPU metadata of "
                            f"hsaco file '{ukd.rel_file}': {_bounded_repr(exc)}"
                        ) from exc
                else:
                    variant_signature[signature_key] = kernel_signature(
                        variant_bytes[vk], symbol, who
                    )

    def _packed(ukd):
        """The `_rewrite_ukd_kpack` arguments read off the packed objects."""
        if ukd.launches:
            return (None, None), {
                "signature": None,
                "packed_launches": {
                    name: {
                        "toc_key": launch.variant_key,
                        "sha256": variant_sha[launch.variant_key],
                        "signature": variant_signature[
                            (launch.variant_key, launch.symbol)
                        ],
                    }
                    for name, launch in ukd.launches.items()
                },
            }
        return (ukd.variant_key, variant_sha[ukd.variant_key]), {
            "signature": variant_signature[(ukd.variant_key, ukd.symbol)]
        }

    kpack_path = None
    if variant_bytes:
        archive = kpack_mod.PackedKernelArchive(
            group_name=group,
            gfx_arch_family=arch,
            gfx_arches=[arch],
            compressor=comp.ZstdCompressor(compression_level=3),
        )
        for vk, data in variant_bytes.items():
            prepared = archive.prepare_kernel(
                relative_path=vk,
                gfx_arch=arch,
                hsaco_data=data,
                metadata={"variant_key": vk},
            )
            archive.add_kernel(prepared)
        archive.finalize_archive()

        kpack_dir = out_arch_dir / KPACK_DIR_NAME
        kpack_dir.mkdir(parents=True, exist_ok=True)
        kpack_path = kpack_dir / _kpack_filename(arch, group)
        archive.write(kpack_path)

    for kdp in inter.kdps:
        out_doc = dict(kdp.header)
        # Each shard targets exactly its own arch, so narrow the authored arch
        # list (which may span several arches, or be empty for a wildcard) to the
        # single arch this shard is for. The descriptor's logical key is
        # (id, arch): the same KDP/UKD id ships under multiple arch shards with
        # per-arch content, unique per arch rather than globally.
        out_doc["arch"] = [arch]
        # Preserve the authored heterogeneous vector: compiled inline UKDs are
        # rewritten to kpack form, pass-through inline UKDs take the shard arch,
        # and standalone-UKD id refs are kept as bare strings (those UKDs ship
        # as their own files below).
        out_kds = []
        for e in kdp.entries:
            if isinstance(e, str):
                out_kds.append(e)
            elif isinstance(e, PassthroughUKD):
                # An inline UKD whose arch reaches past its pack's arch makes
                # the loader reject the whole KDP.
                out_kds.append(_rewrite_passthrough_ukd(e, arch, source_label))
            else:
                args, packed = _packed(e)
                out_kds.append(
                    _rewrite_ukd_kpack(
                        e,
                        arch,
                        *args,
                        **packed,
                        toolchain_fields=_toolchain_for(e, hipcc, rocke_wheel_stamp),
                        # An inline UKD ships INSIDE this KDP file, so the
                        # runtime anchors its library on the KDP's directory,
                        # not the UKD's own notion of where it came from.
                        rel_dir=kdp.rel_dir,
                        group=group,
                    )
                )
        out_doc["kernelDescriptors"] = out_kds
        _write_packed_at(out_arch_dir, kdp.rel_dir, kdp.filename, out_doc)

    # A standalone UKD stays its own file in the shard, rewritten to kpack form
    # with this arch's kpack details. It is emitted only for arches whose
    # surviving KDPs referenced it (compile_intermediate only records those).
    for ukd in standalone:
        args, packed = _packed(ukd)
        out_doc = _rewrite_ukd_kpack(
            ukd,
            arch,
            *args,
            **packed,
            toolchain_fields=_toolchain_for(ukd, hipcc, rocke_wheel_stamp),
            # A standalone UKD is its own file, so it anchors on its own dir.
            rel_dir=ukd.rel_dir,
            group=group,
        )
        _write_packed_at(out_arch_dir, ukd.rel_dir, ukd.filename, out_doc)

    # A pass-through standalone UKD is its own file. It takes this shard's arch,
    # matching the KDP that references it.
    for ukd in inter.passthrough_standalone_ukds.values():
        out_doc = _rewrite_passthrough_ukd(ukd, arch, source_label)
        _write_packed_at(out_arch_dir, ukd.rel_dir, ukd.filename, out_doc)

    prune_result = prune(flat, arch)
    surviving_generics = [
        generic
        for generic in flat.generics()
        if canonical_id(generic.id) in prune_result.reachable_generic_ids
    ]
    for generic in surviving_generics:
        _write_bytes_at(
            out_arch_dir,
            generic.rel_dir,
            generic.path.name,
            generic.path.read_bytes(),
        )

    # Derived from the surviving descriptors: reachable_generic_ids holds ids only, so a
    # sidecar's fate is exactly its descriptor's.
    for sidecar in flat.sidecars_for(surviving_generics):
        _write_bytes_at(
            out_arch_dir,
            sidecar.rel_dir,
            sidecar.name,
            sidecar.source.read_bytes(),
        )

    return ArchResult(arch=arch, out_dir=out_arch_dir, kpack_path=kpack_path)


def _merge_generic_into_member(generic_dir, member_dir, staging_dir):
    """Merge a generic pass's tree into a member folder atomically via `staging_dir`.

    An existing destination file is skipped when byte-identical and refused when
    it differs. Any failure leaves `member_dir` untouched.
    """
    generic_dir, member_dir, staging_dir = (
        Path(generic_dir),
        Path(member_dir),
        Path(staging_dir),
    )
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    try:
        if member_dir.exists():
            shutil.copytree(member_dir, staging_dir, copy_function=shutil.copyfile)
        else:
            staging_dir.mkdir(parents=True)
        for src in sorted(p for p in generic_dir.rglob("*") if p.is_file()):
            rel = src.relative_to(generic_dir)
            dest = staging_dir / rel
            if dest.exists():
                if dest.read_bytes() == src.read_bytes():
                    continue
                raise HkpPackError(
                    f"{rel.as_posix()} would be written by shard "
                    f"'{member_dir.name}' ({member_dir / rel}) and shard "
                    f"'{generic_dir.name}' ({src}) with different bytes; the "
                    "shards' member sets intersect"
                )
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
        if member_dir.exists():
            shutil.rmtree(member_dir)
        staging_dir.rename(member_dir)
    except HkpPackError:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    except OSError as exc:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise HkpPackError(
            f"cannot merge '{generic_dir.name}' into '{member_dir}': {exc}"
        ) from exc


def generic_passes(flat, arches):
    """The generics `run_pipeline` compiles for @arches, in table order.

    A generic is compiled when one of @arches is its member and a KDP ships for it.
    """
    table = flat.generic_targets
    return [
        generic
        for generic in table.names()
        if any(a in table.members(generic) for a in arches)
        and any(kdp_survives(k.doc, flat, generic) for k in flat.kdps())
    ]


def run_pipeline(
    source_root,
    arches,
    out_root,
    hipcc,
    generic_targets_json,
    rocm_kpack_dir=None,
    inter_root=None,
    expected_sha256=None,
    rocke_wheel_stamp=None,
    group=GROUP_NAME,
    source_label=None,
    exclude_folders=(),
    disabled_kinds=(),
    log=print,
):
    """One invocation over the full arch list: compile, prune, pack, install.

    Loads the one source root once — recursively, preserving each descriptor's
    authored subpath — then for each arch compiles the targeting KDPs' variants,
    prunes, and packs. Producer selection is per-UKD on `kernel_source.kind`, so
    hip and rocKE descriptors coexist under one root (in child folders that scope
    them) and combine into one kpack per arch. A hip UKD's source resolves
    relative to the descriptor that named it. An hsaco UKD's `file` resolves the
    same way, and its prebuilt bytes are packed as-is into the same kpack. A UKD
    of a pass-through kind runs no producer and is emitted as authored, so a
    root that holds only pass-through UKDs writes descriptors and no archive. An arch with no
    surviving KDP is skipped cleanly (no folder, no kpack) and logged with 'no
    kernels for <arch>, skipping'; every arch skipping packs nothing, which is a
    clean skip too. Empty arch list installs nothing (exit 0).

    `generic_targets_json` is the generic target table; generic passes are
    merged into member folders (descriptor-packaging/README.md).

    `exclude_folders` names top-level child folders of the root this build does
    not pack, and `disabled_kinds` the kernel_source kinds it has no producer for;
    both are pruned at load (load_flat_input) exactly as arch pruning prunes, so
    nothing downstream sees them.
    """
    out_root = Path(out_root)
    results = {}
    if not arches:
        return results

    kpack_mod, comp = load_kpack(rocm_kpack_dir)
    generic_targets = GenericTargets.load(generic_targets_json)
    flat = load_flat_input(
        source_root,
        generic_targets,
        log=log,
        exclude_folders=exclude_folders,
        disabled_kinds=disabled_kinds,
    )

    if inter_root is None:
        raise HkpPackError(
            "inter_root is required: pass --inter-root (or inter_root=) naming a "
            "build-only directory. It must not be derived from out_root, which is a "
            "staged output tree."
        )
    inter_root = Path(inter_root)

    failures = {}
    for arch in arches:
        surviving = [k for k in flat.kdps() if kdp_survives(k.doc, flat, arch)]
        out_arch_dir = out_root / arch
        if not surviving:
            log(f"no kernels for {arch}, skipping")
            if out_arch_dir.exists():
                shutil.rmtree(out_arch_dir)
            results[arch] = ArchResult(
                arch=arch, out_dir=out_arch_dir, kpack_path=None, skipped=True
            )
            continue
        try:
            inter = compile_intermediate(
                flat, source_root, arch, hipcc, inter_root / arch, log=log
            )
            # Stage this arch into a sibling temp dir and rename it into place
            # only once pack_arch returns cleanly. pack_arch creates the arch
            # directory before it validates anything, so writing in place
            # leaves a present-but-empty arch directory behind on failure -- and
            # install(DIRECTORY ... OPTIONAL) skips only a MISSING directory, so
            # that partial tree would install. Rename is atomic within a
            # filesystem, and both paths are under out_root by construction.
            staging = out_root / f".{arch}.staging"
            if staging.exists():
                shutil.rmtree(staging)
            result = pack_arch(
                flat,
                inter,
                staging,
                kpack_mod,
                comp,
                expected_sha256=expected_sha256,
                hipcc=hipcc,
                rocke_wheel_stamp=rocke_wheel_stamp,
                group=group,
                source_label=source_label,
            )
            if out_arch_dir.exists():
                shutil.rmtree(out_arch_dir)
            staging.rename(out_arch_dir)
            kpack_path = (
                None
                if result.kpack_path is None
                else out_arch_dir / KPACK_DIR_NAME / _kpack_filename(arch, group)
            )
            results[arch] = replace(
                result,
                out_dir=out_arch_dir,
                kpack_path=kpack_path,
            )
        except HkpPackError as exc:
            # One arch failing must not destroy the other arches' work: a
            # wildcard-arch UKD hitting an arch-restricted builder should shrink
            # one shard, not fail every shard. The failed arch's staged output is
            # discarded rather than left half-written, so install(... OPTIONAL)
            # skips it cleanly instead of shipping a partial tree. Its
            # intermediate dir stays for debugging -- build-only, never shipped.
            failures[arch] = str(exc)
            log(f"ERROR: {arch} failed: {exc}")
            # Discard the half-written staging dir AND any previous good output
            # for this arch: shipping a stale shard beside fresh ones would be a
            # subtler lie than shipping none.
            staging = out_root / f".{arch}.staging"
            if staging.exists():
                shutil.rmtree(staging)
            if out_arch_dir.exists():
                shutil.rmtree(out_arch_dir)
            results[arch] = ArchResult(
                arch=arch, out_dir=out_arch_dir, kpack_path=None, skipped=True
            )

    for generic in generic_passes(flat, arches):
        members = [a for a in arches if a in generic_targets.members(generic)]
        generic_out = inter_root / ".generic-out" / generic
        try:
            inter = compile_intermediate(
                flat, source_root, generic, hipcc, inter_root / generic, log=log
            )
            if generic_out.exists():
                shutil.rmtree(generic_out)
            generic_result = pack_arch(
                flat,
                inter,
                generic_out,
                kpack_mod,
                comp,
                expected_sha256=expected_sha256,
                hipcc=hipcc,
                rocke_wheel_stamp=rocke_wheel_stamp,
                group=group,
                source_label=source_label,
            )
        except HkpPackError as exc:
            failures[generic] = str(exc)
            log(f"ERROR: {generic} failed: {exc}")
            if generic_out.exists():
                shutil.rmtree(generic_out)
            continue
        for member in members:
            if member in failures:
                continue
            member_dir = out_root / member
            try:
                _merge_generic_into_member(
                    generic_out, member_dir, out_root / f".{member}.merge.staging"
                )
            except HkpPackError as exc:
                failures[member] = str(exc)
                log(f"ERROR: {member} failed merging {generic}: {exc}")
                break
            previous = results.get(member)
            kpack_path = None if previous is None else previous.kpack_path
            copied = ()
            if generic_result.kpack_path is not None:
                copied = (
                    member_dir / KPACK_DIR_NAME / _kpack_filename(generic, group),
                )
            results[member] = ArchResult(
                arch=member,
                out_dir=member_dir,
                kpack_path=kpack_path,
                skipped=False,
                generic_kpack_paths=(
                    (() if previous is None else previous.generic_kpack_paths) + copied
                ),
            )

    if failures:
        # Non-zero exit with partial output: the build fails loudly, but a
        # developer can still inspect what did succeed. Exiting 0 here would
        # resurrect the silent-empty-package class of defect.
        detail = "; ".join(f"{a}: {r}" for a, r in sorted(failures.items()))
        raise HkpPackError(
            f"packing failed for {len(failures)} failure(s) across {len(arches)} "
            f"requested arch(es) [{detail}]. Arches that succeeded were written; "
            "the failed arches' output was discarded."
        )

    # Nothing surviving any arch is a root with nothing to pack for this build --
    # arch-pruned, or all in disabled folders or kinds -- so it packs nothing and that
    # is not an error. Configure asks shipped_engines() first and leaves such a root
    # dormant rather than wiring it.
    arch_list = ", ".join(arches)
    if all(r.skipped for r in results.values()):
        log(f"packing '{source_root}': nothing to pack for [{arch_list}], skipping")
        return results

    # An internal invariant no input reaches: a shard that shipped an archive-producing UKD
    # always writes an archive. Only the packer knows which shards shipped and
    # what each selected.
    ukd_by_id = flat.ukd_by_id()
    for arch, result in results.items():
        if result.skipped or result.kpack_path is not None:
            continue
        for kdp in flat.kdps():
            if not kdp_survives(kdp.doc, flat, arch):
                continue
            for sid, ukd, _sdesc in _selected_entries(
                kdp.doc, arch, ukd_by_id, generic_targets
            ):
                if not _is_passthrough(ukd):
                    raise HkpPackError(
                        f"packing '{source_root}' wrote descriptors but no "
                        f"archive for {arch}, whose shard selected UKD "
                        f"'{sid or ukd.get('id')}' of an archive-producing kind "
                        "('hip', 'rocke' or 'hsaco')."
                    )
    return results
