import copy
import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .hip_compile import compile_hip_variant, hip_variant_key
from .rocke_compile import compile_rocke_variant, rocke_variant_key
from .descriptors import (
    arch_matches,
    kdp_survives,
    load_flat_inputs,
    reachable_generic_ids,
)
from .errors import HkpPackError
from .kpack_resolver import load_kpack

GROUP_NAME = "hip_kernel_provider"


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


@dataclass
class StandaloneUKD(InlineUKD):
    """A UKD authored as its own `<name>.ukd.json` and referenced by a KDP.

    Carries the same compiled fields as an inline UKD plus the original filename,
    since it stays a standalone file in the shipped shard (rather than being
    folded into the KDP).
    """

    filename: str = ""


@dataclass
class ArchKDP:
    id: str
    filename: str
    header: dict
    ukds: list = field(default_factory=list)
    # Ordered kernelDescriptors output spec: each element is either an InlineUKD
    # (rewritten inline in the shipped KDP) or a str (a standalone-UKD id ref,
    # kept verbatim). Preserves authored order across the heterogeneous vector.
    entries: list = field(default_factory=list)


@dataclass
class IntermediateArch:
    arch: str
    directory: Path
    kdps: list = field(default_factory=list)
    variant_co: dict = field(default_factory=dict)
    variant_symbol: dict = field(default_factory=dict)
    standalone_ukds: dict = field(default_factory=dict)


@dataclass
class ArchResult:
    arch: str
    out_dir: Path
    kpack_path: Path
    skipped: bool = False


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _kpack_filename(arch):
    return f"{GROUP_NAME}_{arch}.kpack"


def _kpack_rel(arch):
    return f"kpack/{_kpack_filename(arch)}"


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
            "metadata",
            "priority",
            "build",
            "arch",
            "provenance",
        )
    }


def _compile_ukd_variant(
    ukd,
    where,
    origin_root,
    origin_index,
    arch,
    hipcc,
    inter_arch_dir,
    variant_co,
    variant_symbol,
):
    """Compile one UKD variant for arch, deduped into variant_co per kind.

    Dispatches on kernel_source.kind: hip resolves its source against
    origin_root (the root this UKD's descriptor loaded from) and keys on
    (source, build, origin_index), the positional ordinal of that root, so two
    roots carrying an identically named source at the same relative path but
    different bytes stay distinct. rocke keys on (source, builder, spec) and is
    origin-independent (the source is a dotted module resolved by import), so
    origin_root/origin_index are accepted only for signature uniformity. Returns
    (variant_key, symbol, record_fields).
    """
    ks = ukd["kernel_source"]
    kind = ks["kind"]
    source = ks["source"]
    if kind == "hip":
        entry = ks["entry"]
        build = ks["build"]
        vk = hip_variant_key(source, build, origin_index)
        if vk not in variant_co:
            variant_co[vk] = compile_hip_variant(
                hipcc,
                origin_root,
                source,
                build,
                arch,
                inter_arch_dir,
                key_origin_index=origin_index,
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
        builder = ks["builder"]
        spec = ks["spec"]
        vk = rocke_variant_key(source, builder, spec)
        if vk not in variant_co:
            co_path, captured = compile_rocke_variant(
                origin_root, source, builder, spec, arch, inter_arch_dir
            )
            variant_co[vk] = co_path
            variant_symbol[vk] = captured
        symbol = variant_symbol[vk]
        fields = {
            "origin_kind": "rocke",
            "source": source,
            "entry": None,
            "build": None,
            "builder": builder,
            "spec": spec,
        }
    else:
        raise HkpPackError(f"{where} kernel_source has unsupported kind '{kind}'")
    return vk, symbol, fields


def compile_intermediate(flat, arch, hipcc, inter_arch_dir, log=print):
    """Compile every hip UKD in the KDPs targeting arch and stage a per-arch tree.

    Writes inter_arch_dir with: hsaco-form KDP JSON (inline UKDs rewritten
    hip->hsaco, build lifted to top-level) + one .co per distinct (source,build)
    variant + every generic copied through + any non-matching KDP copied in its
    authored hip form (so pruning has a KDP to drop). Standalone UKDs a surviving
    KDP references by Id are compiled here too and tracked per arch, to be
    emitted as their own files by pack_arch. Returns an IntermediateArch carrying
    the origin data the pack step needs for provenance.
    """
    inter_arch_dir = Path(inter_arch_dir)
    inter_arch_dir.mkdir(parents=True, exist_ok=True)

    variant_co = {}
    variant_symbol = {}
    arch_kdps = []
    standalone_ukds = {}
    ukd_by_id = flat.ukd_by_id()

    for kdp in flat.kdps():
        doc = kdp.doc
        if not arch_matches(doc, arch):
            (inter_arch_dir / kdp.path.name).write_bytes(kdp.path.read_bytes())
            continue

        new_doc = copy.deepcopy(doc)
        ukds = []
        entries = []
        new_kds = []
        for entry in new_doc["kernelDescriptors"]:
            if isinstance(entry, str):
                # A reference to a standalone UKD: compile it once per arch and
                # keep the string in the KDP; it ships as its own file. Skip it
                # in this shard unless its own arch applies here.
                sdesc = ukd_by_id[entry]
                if not arch_matches(sdesc.doc, arch):
                    continue
                entries.append(entry)
                new_kds.append(entry)
                if entry in standalone_ukds:
                    continue
                sukd = sdesc.doc
                where = f"standalone UKD {sdesc.path.name}"
                vk, symbol, fields = _compile_ukd_variant(
                    sukd,
                    where,
                    sdesc.origin_root,
                    sdesc.origin_index,
                    arch,
                    hipcc,
                    inter_arch_dir,
                    variant_co,
                    variant_symbol,
                )
                standalone_ukds[entry] = StandaloneUKD(
                    id=sukd.get("id"),
                    name=sukd.get("name"),
                    metadata=sukd.get("metadata"),
                    priority=sukd.get("priority"),
                    symbol=symbol,
                    variant_key=vk,
                    extra=_ukd_extra(sukd),
                    filename=sdesc.path.name,
                    **fields,
                )
                continue

            ukd = entry
            # An inline UKD ships in this shard only when its own arch applies.
            if not arch_matches(ukd, arch):
                continue
            where = f"UKD '{ukd.get('id')}' in {kdp.path.name}"
            vk, symbol, fields = _compile_ukd_variant(
                ukd,
                where,
                kdp.origin_root,
                kdp.origin_index,
                arch,
                hipcc,
                inter_arch_dir,
                variant_co,
                variant_symbol,
            )
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
        (inter_arch_dir / kdp.path.name).write_text(
            json.dumps(new_doc, indent=2) + "\n", encoding="utf-8"
        )
        arch_kdps.append(
            ArchKDP(
                id=doc.get("id"),
                filename=kdp.path.name,
                header=_kdp_header(doc),
                ukds=ukds,
                entries=entries,
            )
        )

    for generic in flat.generics():
        (inter_arch_dir / generic.path.name).write_bytes(generic.path.read_bytes())

    return IntermediateArch(
        arch=arch,
        directory=inter_arch_dir,
        kdps=arch_kdps,
        variant_co=variant_co,
        variant_symbol=variant_symbol,
        standalone_ukds=standalone_ukds,
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


def _rewrite_ukd_kpack(ukd, arch, toc_key, sha256):
    if ukd.origin_kind == "rocke":
        provenance = {
            "origin_kind": "rocke",
            "source": ukd.source,
            "builder": ukd.builder,
            "spec": ukd.spec,
        }
    else:
        provenance = {
            "origin_kind": "hip",
            "source": ukd.source,
            "entry": ukd.entry,
            "build": ukd.build,
        }
    doc = {
        "id": ukd.id,
        "name": ukd.name,
        "kernel_source": {
            "kind": "kpack",
            "library": _kpack_rel(arch),
            "toc_key": toc_key,
            "symbol": ukd.symbol,
            "sha256": sha256,
        },
        "metadata": ukd.metadata,
        "priority": ukd.priority,
        "provenance": provenance,
    }
    doc.update(ukd.extra)
    # Every shipped UKD carries the single shard arch, matching the KDP. Set it
    # after the extra passthrough so a source multi-arch list can't leak through.
    doc["arch"] = [arch]
    return doc


def pack_arch(flat, inter, out_arch_dir, kpack_mod, comp, expected_sha256=None):
    """Pack a pruned intermediate arch into the shipped kpack release tree.

    Each distinct (source,build) variant .co is packed once under its own
    toc_key; inline UKDs are rewritten hsaco->kpack, stamping toc_key + sha256
    and moving build into a sibling provenance block. Guarded against toc_key
    collisions (distinct inputs mapping to one key).
    """
    arch = inter.arch
    out_arch_dir = Path(out_arch_dir)
    kpack_dir = out_arch_dir / "kpack"
    kpack_dir.mkdir(parents=True, exist_ok=True)

    standalone = list(inter.standalone_ukds.values())

    def _all_ukds():
        for kdp in inter.kdps:
            for ukd in kdp.ukds:
                yield ukd
        for ukd in standalone:
            yield ukd

    variant_bytes = {}
    variant_sha = {}
    variant_source_build = {}
    for ukd in _all_ukds():
        vk = ukd.variant_key
        toc_key = vk
        sig = (ukd.source, json.dumps(ukd.build, sort_keys=True))
        if vk in variant_source_build and variant_source_build[vk] != sig:
            raise HkpPackError(
                f"toc_key collision: '{vk}' maps to two distinct "
                f"(source,build) inputs {variant_source_build[vk]} and {sig}"
            )
        variant_source_build[vk] = sig
        if vk not in variant_bytes:
            data = inter.variant_co[vk].read_bytes()
            digest = _sha256(data)
            if expected_sha256 and toc_key in expected_sha256:
                if digest != expected_sha256[toc_key]:
                    raise HkpPackError(
                        f"sha256 mismatch for toc_key '{toc_key}': expected "
                        f"{expected_sha256[toc_key]}, packed blob is {digest}"
                    )
            variant_bytes[vk] = data
            variant_sha[vk] = digest
        if ukd.symbol.encode("ascii") not in variant_bytes[vk]:
            raise HkpPackError(
                f"UKD '{ukd.id}' declares symbol '{ukd.symbol}' not present "
                f"in code object for variant '{vk}'"
            )

    archive = kpack_mod.PackedKernelArchive(
        group_name=GROUP_NAME,
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

    kpack_path = kpack_dir / _kpack_filename(arch)
    archive.write(kpack_path)

    for kdp in inter.kdps:
        out_doc = dict(kdp.header)
        # Each shard targets exactly its own arch, so narrow the authored arch
        # list (which may span several arches, or be empty for a wildcard) to the
        # single arch this shard is for. The descriptor's logical key is
        # (id, arch): the same KDP/UKD id ships under multiple arch shards with
        # per-arch content, unique per arch rather than globally.
        out_doc["arch"] = [arch]
        # Preserve the authored heterogeneous vector: inline UKDs are rewritten
        # to kpack form, standalone-UKD id refs are kept as bare strings (those
        # UKDs ship as their own files below).
        out_kds = []
        for e in kdp.entries:
            if isinstance(e, str):
                out_kds.append(e)
            else:
                out_kds.append(
                    _rewrite_ukd_kpack(
                        e, arch, e.variant_key, variant_sha[e.variant_key]
                    )
                )
        out_doc["kernelDescriptors"] = out_kds
        (out_arch_dir / kdp.filename).write_text(
            json.dumps(out_doc, indent=2) + "\n", encoding="utf-8"
        )

    # A standalone UKD stays its own file in the shard, rewritten to kpack form
    # with this arch's kpack details. It is emitted only for arches whose
    # surviving KDPs referenced it (compile_intermediate only records those).
    for ukd in standalone:
        out_doc = _rewrite_ukd_kpack(
            ukd, arch, ukd.variant_key, variant_sha[ukd.variant_key]
        )
        (out_arch_dir / ukd.filename).write_text(
            json.dumps(out_doc, indent=2) + "\n", encoding="utf-8"
        )

    prune_result = prune(flat, arch)
    for generic in flat.generics():
        if generic.id in prune_result.reachable_generic_ids:
            (out_arch_dir / generic.path.name).write_bytes(generic.path.read_bytes())

    return ArchResult(arch=arch, out_dir=out_arch_dir, kpack_path=kpack_path)


def run_pipeline(
    source_roots,
    arches,
    out_root,
    hipcc,
    rocm_kpack_dir=None,
    inter_root=None,
    expected_sha256=None,
    log=print,
):
    """One invocation over the full arch list: compile, prune, pack, install.

    Loads and merges every source folder once, then for each arch compiles the
    targeting KDPs' variants, prunes, and packs. Each hip UKD's source resolves
    against its own root, and all roots' descriptors combine into one kpack per
    arch. An arch with no surviving KDP is skipped cleanly (no folder, no kpack)
    and logged with 'no kernels for <arch>, skipping'. Empty arch list installs
    nothing (exit 0).
    """
    out_root = Path(out_root)
    results = {}
    if not arches:
        return results

    kpack_mod, comp = load_kpack(rocm_kpack_dir)
    flat = load_flat_inputs(source_roots, log=log)

    if inter_root is None:
        inter_root = out_root.parent / "hkp-intermediate"
    inter_root = Path(inter_root)

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
        inter = compile_intermediate(flat, arch, hipcc, inter_root / arch, log=log)
        results[arch] = pack_arch(
            flat, inter, out_arch_dir, kpack_mod, comp, expected_sha256=expected_sha256
        )
    return results
