import copy
import hashlib
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import agreement, toolchain
from .hip_compile import (
    compile_hip_variant,
    hip_source_relpath,
    hip_variant_key,
)
from .rocke_compile import compile_rocke_variant, rocke_variant_key
from .descriptors import (
    KPACK_DIR_NAME,
    arch_matches,
    kdp_survives,
    load_flat_input,
    reachable_generic_ids,
)
from .errors import HkpPackError
from .kernel_signature import kernel_signature
from .kpack_resolver import load_kpack

# The archive group every root packs under unless it names its own. One archive
# ships per (group, arch) and the filename carries both, so two roots staged into
# one descriptor tree must not share a group: the second would overwrite the first.
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


@dataclass
class StandaloneUKD(InlineUKD):
    """A UKD authored as its own `<name>.ukd.json` and referenced by a KDP: an
    inline UKD's compiled fields plus the original filename it keeps in the shard.
    """

    filename: str = ""
    rel_dir: Path = Path(".")


@dataclass
class PassthroughUKD:
    """A UKD the packer emits without running a producer.

    The authored document ships with `arch` narrowed to the shard. `rel_dir` is the
    descriptor's position under the source root (the KDP's directory for an inline
    UKD, its own for a standalone); `filename` is set only for the standalone form.
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
    # Ordered kernelDescriptors output spec, preserving authored order across the
    # heterogeneous vector: an InlineUKD (rewritten inline), a PassthroughUKD
    # (shipped as authored), or a str (a standalone-UKD id ref, kept verbatim).
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


def _sha256(data):
    """Digest of a packed blob, recorded on the shipped UKD. Hashed over the
    decompressed code object, the buffer KpackModuleCache hashes back;
    `expected_sha256` cross-checks it at pack time.
    """
    return hashlib.sha256(data).hexdigest()


def _kpack_filename(arch, group=GROUP_NAME):
    return f"{group}_{arch}.kpack"


def _kpack_rel(arch, rel_dir=Path("."), group=GROUP_NAME):
    """`library` for a descriptor living at `rel_dir` within the arch shard.

    The runtime resolves `originDirectory / library` against the descriptor FILE's
    parent, while the archive lives once per arch at the arch root, so a nested
    descriptor climbs back out. Containment is anchored on the descriptor TREE
    (`IngestorKernelCode.hpp`, KPACK case), which makes the climb legal.
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

    Dispatches on kernel_source.kind: hip resolves its source relative to the
    descriptor that named it and keys on (source, build); rocke keys on
    (source, builder, spec) and is location-independent, so source_root/rel_dir are
    accepted only for signature uniformity. Returns (variant_key, symbol, fields).
    """
    ks = ukd["kernel_source"]
    kind = ks["kind"]
    # Only the two producer kinds carry a `source`, so it is read per-arm: hoisting
    # the read above the dispatch turns the unsupported-kind raise into a KeyError.
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
        if vk not in variant_co:
            co_path, captured, observations = compile_rocke_variant(
                source,
                builder,
                spec,
                arch,
                inter_arch_dir,
                observation_requests.get(vk, {}),
                origins,
            )
            variant_co[vk] = co_path
            variant_symbol[vk] = captured
            variant_observations[vk] = observations
        observations = variant_observations[vk]
        symbol = variant_symbol[vk]
        # A reused compile result is checked as hard as a fresh one, for EVERY
        # consumer: the first's agreement says nothing about a second completing
        # different metadata from the same decisions.
        if observations.get("arch") != arch:
            raise HkpPackError(
                f"{where}: compile observations were taken for "
                f"'{observations.get('arch')}', not '{arch}'"
            )
        if observations.get("symbol") != symbol:
            raise HkpPackError(
                f"{where}: compile observations name symbol "
                f"'{observations.get('symbol')}', not '{symbol}'"
            )
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
            "consumers": consumers,
        }
    else:
        raise HkpPackError(f"{where} kernel_source has unsupported kind '{kind}'")
    fields["provenance"] = copy.deepcopy(ukd.get("provenance", {}))
    return vk, symbol, fields


def _is_passthrough(ukd):
    """Whether a UKD ships as authored instead of through a producer."""
    return ukd["kernel_source"]["kind"] in _PASSTHROUGH_KINDS


def _root_holds_compiling_source(flat):
    """Whether the authored root holds a UKD that a producer compiles, and so an
    archive. The complement of _is_passthrough over both authoring forms, so the
    two spellings cannot drift.
    """
    for desc in flat.ukds():
        if not _is_passthrough(desc.doc):
            return True
    for kdp in flat.kdps():
        for entry in kdp.doc.get("kernelDescriptors", []):
            if isinstance(entry, dict) and not _is_passthrough(entry):
                return True
    return False


def _dest_at(base, rel_dir, name):
    """Destination for an authored file, preserving its subpath under base: the
    staged and installed trees mirror the authored subpath verbatim, and two
    descriptors cannot share a path under one source root.
    """
    dest = Path(base) / rel_dir / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest


def _write_bytes_at(base, rel_dir, name, data):
    _dest_at(base, rel_dir, name).write_bytes(data)


def _write_text_at(base, rel_dir, name, text):
    _dest_at(base, rel_dir, name).write_text(text, encoding="utf-8")


@dataclass(frozen=True)
class _VariantJob:
    """One distinct variant the prewarm compiles, as a picklable record: every
    field is a str or plain dict and `hipcc`/`arch` travel by value, so the worker
    is a pure function of its argument.
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


def _selected_entries(doc, arch, ukd_by_id):
    """Yield the entries of doc that ship for arch.

    Yields `(entry_id, ukd_doc, sdesc)`: for a standalone-UKD id ref the id, its doc
    and its Descriptor; for an inline UKD `None`, the entry dict and `None`. All
    three arch filters live here alone, so the prewarm and the serial walk cannot
    select different variant sets, and `ukd_by_id` is a parameter so an orphan
    standalone UKD is unreachable.
    """
    if not arch_matches(doc, arch):
        return
    for entry in doc["kernelDescriptors"]:
        if isinstance(entry, str):
            sdesc = ukd_by_id[entry]
            if arch_matches(sdesc.doc, arch):
                yield entry, sdesc.doc, sdesc
        elif arch_matches(entry, arch):
            yield None, entry, None


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
    generics = {d.id: d.doc for d in flat.generics()}
    schemas = {d.id: d.doc for d in flat.generics() if d.type == "kmd"}
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
            for _sid, ukd, _sdesc in _selected_entries(kdp.doc, arch, ukd_by_id):
                if agreement.resolved_contract(ukd) is not None:
                    raise HkpPackError(
                        f"UKD {ukd['id']} in {kdp.path.name}: a specialization "
                        "contract names an engine, and this KDP authors none"
                    )
            continue
        if engine_id not in generics:
            raise HkpPackError(
                f"KDP {kdp.path.name}: engine '{engine_id}' resolves to no descriptor"
            )
        engine = generics[engine_id]
        kmd_id = engine.get("metadata")
        if kmd_id not in schemas:
            raise HkpPackError(
                f"engine '{engine_id}': metadata '{kmd_id}' resolves to no KMD"
            )
        kmd = schemas[kmd_id]
        header = _kdp_header(kdp.doc)
        header["arch"] = [arch]
        for sid, ukd, sdesc in _selected_entries(kdp.doc, arch, ukd_by_id):
            # Completion is checked against the KMD the chain actually resolved to,
            # so a mis-typed or missing mandatory value fails before a compile.
            agreement.complete_metadata(ukd["metadata"], kmd)
            kind = ukd["kernel_source"]["kind"]
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
            vk = _variant_key_for(ukd, rel_dir)
            request = agreement.observation_request(declaration, kmd)
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
    `_prewarm_variants`.
    """
    ukd_by_id = flat.ukd_by_id()
    if observation_requests is None:
        _, observation_requests = _agreement_inputs(flat, arch)
    jobs = []
    seen = set()
    for kdp in flat.kdps():
        for sid, ukd, sdesc in _selected_entries(kdp.doc, arch, ukd_by_id):
            rel_dir = sdesc.rel_dir if sid is not None else kdp.rel_dir
            vk = _variant_key_for(ukd, rel_dir)
            if vk is None or vk in seen:
                continue
            seen.add(vk)
            jobs.append(
                _VariantJob(
                    vk=vk,
                    kind=ukd["kernel_source"]["kind"],
                    ukd=ukd,
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
    `(vk, None, None, "Name: text", None, {})` on failure: failures are returned
    rather than raised because rocke and comgr exceptions may not pickle. `origins`
    is this variant's producer file identities in picklable form, the only route to
    `OriginObserver.absorb`. No key is computed here, since a child-side
    recomputation would resolve the real functions and disagree with the walk.
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
            # The hip symbol is authored, not captured from the artifact.
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
            )
        else:
            # Unreachable while `_variant_key_for` keys only these two kinds; a
            # raise rather than a fall-through, since a third keyable kind would
            # otherwise compile as rocke from the wrong producer.
            raise HkpPackError(
                f"no variant compiler for kernel source kind '{job.kind}'"
            )
    except Exception as exc:
        return job.vk, None, None, f"{type(exc).__name__}: {exc}", None, {}
    return job.vk, str(co_path), symbol, None, observations, origins.exported()


def _cgroup_v2_cpu_quota():
    """Whole CPUs allowed by cgroup v2, or None when unlimited or absent.

    Linux only, read from the filesystem because no API reports it and a container
    given `--cpus=8` keeps a full affinity mask. Walked from this process's cgroup
    to the root taking the tightest limit: without a cgroup namespace (Slurm, a
    systemd session) the root carries no `cpu.max` at all.
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
        # Floors to at least one: a sub-CPU allocation would otherwise disable the
        # pool entirely.
        return max(1, int(quota) // int(period))
    except (ValueError, ZeroDivisionError):
        return None


def _cpu_budget():
    """CPUs this process may actually use, rather than the host's core count: a
    containerised or scheduled pack gets a slice of the machine, and each source
    below sees a limit the others cannot, so the smallest wins.
    """
    limits = [_cgroup_v2_cpu_quota()]

    process_cpu_count = getattr(os, "process_cpu_count", None)
    if process_cpu_count is not None:
        # 3.13+, and honours the affinity mask on every platform that has one.
        limits.append(process_cpu_count())
    elif hasattr(os, "sched_getaffinity"):
        # Linux only, so it cannot be the only affinity source -- this packer runs
        # on Windows too, where the host count below is all there is.
        limits.append(len(os.sched_getaffinity(0)))

    limits = [limit for limit in limits if limit]
    return min(limits) if limits else (os.cpu_count() or 1)


def _pack_jobs():
    """Worker count for the prewarm, from `HKP_PACK_JOBS`.

    Unset means `min(32, _cpu_budget())`. The cap is per-worker startup (~250 ms of
    rocke+comgr import), and the optimum moves with pack size -- about 12 workers
    for 200 variants, 48 for 1672 -- so 32 sits below the large-pack optimum, since
    overshooting costs more than undershooting.

    `HKP_PACK_JOBS=1` forces serial execution, the escape hatch for a single clean
    traceback. Anything not an integer of 1 or more is a hard error, including `0`
    and negatives, and an explicit value is never reduced to `_cpu_budget()`. An
    environment variable because it reaches through the CMake custom command with
    no plumbing.
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

    The single place either key function is called, so the walk and the prewarm
    cannot key one variant two ways; both resolve as module globals on every call,
    since an alias or child-side recomputation would bind past any substitution. A
    kind with no compilable source yields None, leaving the walk the sole reporter.
    """
    ks = ukd["kernel_source"]
    kind = ks["kind"]
    if kind == "hip":
        return hip_variant_key(hip_source_relpath(rel_dir, ks["source"]), ks["build"])
    if kind == "rocke":
        return rocke_variant_key(ks["source"], ks["builder"], ks["spec"])
    return None


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

    The walk then finds each key present and skips the expensive call; records,
    symbols and doc rewriting stay the walk's. Returns one producer-origin map per
    compiled variant, which the caller must fold into the observer spanning the
    arch, since a prewarmed variant is observed nowhere else. Fails fast and writes
    nothing into either cache when it does: a partly-filled cache would let the
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

    # Built per arch and never kept in a module global: workers snapshot `sys.path`
    # at process start, so a reused pool would freeze the path list it was built
    # with.
    pool = ProcessPoolExecutor(max_workers=workers)
    results = []
    failure = None
    try:
        # Consumed lazily so the first failure stops the pack. `map` yields in
        # submission (walk) order, so this stops on the variant the serial path
        # would have named.
        for result in pool.map(_compile_one_variant, jobs, chunksize=1):
            if result[3] is not None:
                failure = result
                break
            results.append(result)
    except Exception as exc:
        # A killed worker surfaces here as a non-HkpPackError, which unconverted
        # would sail past run_pipeline's per-arch handler and discard every other
        # arch's work.
        raise HkpPackError(
            f"the variant compile pool for {arch} failed "
            f"({type(exc).__name__}: {exc}); a worker process died -- an OOM "
            "kill, a crash inside the toolchain, or an interrupt. Retrying "
            "with a lower HKP_PACK_JOBS addresses the first of those"
        ) from exc
    finally:
        # `map` submits every job up front, so the queue outlives the break.
        # Finalising its result generator already cancels pending futures;
        # `cancel_futures` restates that beside the `wait` it qualifies.
        pool.shutdown(wait=True, cancel_futures=True)

    if failure is not None:
        # One variant named, no failure count: the pack stops at the first failure,
        # so a count would only say how far the pool got. Named by UKD id, which a
        # reader can look up; the content-hash key is not.
        named = next((job.ukd.get("id") for job in jobs if job.vk == failure[0]), None)
        raise HkpPackError(
            f"variant '{named or failure[0]}' failed to compile for {arch}: "
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
    """Compile every hip UKD in the KDPs targeting arch and stage a per-arch tree.

    Writes inter_arch_dir with hsaco-form KDP JSON (inline UKDs rewritten
    hip->hsaco, build lifted to top-level), one .co per distinct (source,build)
    variant, every generic copied through, and any non-matching KDP copied in its
    authored form so pruning has a KDP to drop. Standalone UKDs a surviving KDP
    references are compiled and tracked per arch for pack_arch. A pass-through kind
    runs no producer and stays as authored.
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
        if not arch_matches(doc, arch):
            _write_bytes_at(
                inter_arch_dir, kdp.rel_dir, kdp.path.name, kdp.path.read_bytes()
            )
            continue

        new_doc = copy.deepcopy(doc)
        ukds = []
        entries = []
        new_kds = []
        for sid, entry, sdesc in _selected_entries(new_doc, arch, ukd_by_id):
            if sid is not None:
                # A standalone-UKD reference: compiled once per arch, shipped as its
                # own file, and kept as a string here. Listing precedes the
                # compile-once check, so a UKD two KDPs reference appears in both.
                entries.append(sid)
                new_kds.append(sid)
                if sid in passthrough_standalone_ukds:
                    continue
                sukd = entry
                where = f"standalone UKD {sdesc.path.name}"
                if _is_passthrough(sukd):
                    kind = sukd["kernel_source"]["kind"]
                    log(f"{where}: emitting kind '{kind}' as authored")
                    passthrough_standalone_ukds[sid] = PassthroughUKD(
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
                if sid in standalone_ukds:
                    continue
                standalone_ukds[sid] = StandaloneUKD(
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
                kind = ukd["kernel_source"]["kind"]
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
        # A KDP whose UKDs all filter out for this arch is dropped from the shard:
        # no intermediate JSON, no record, and its exclusive generics prune with it.
        if not new_kds:
            log(f"KDP {kdp.path.name}: all UKDs filtered out for {arch}, dropping")
            continue
        new_doc["kernelDescriptors"] = new_kds
        _write_text_at(
            inter_arch_dir,
            kdp.rel_dir,
            kdp.path.name,
            json.dumps(new_doc, indent=2) + "\n",
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
    toolchain_fields=None,
    rel_dir=Path("."),
    group=GROUP_NAME,
):
    """Rewrite a compiled UKD into shipped kpack form.

    `toolchain_fields` (hipcc version, resolved comgr, rocKE wheel digest) merges
    into provenance rather than the variant key, where a wheel bump would rename
    every rocKE artifact. A `specialization_contract` declared once on the enclosing
    KDP stays there, since the KDP header ships verbatim.

    Producer evidence is reserved: an authored input may not supply
    `provenance.effective_spec`, authored fields merge UNDER the produced block, and
    the authored `extra` passthrough cannot name a key this function writes.
    """
    if "effective_spec" in ukd.provenance:
        raise HkpPackError(
            f"UKD '{ukd.id}': authored input supplies provenance.effective_spec, "
            "which only the producing compiler creates"
        )
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
    provenance = {**ukd.provenance, **provenance}
    if toolchain_fields:
        provenance.update(toolchain_fields)
    doc = {
        "id": ukd.id,
        "name": ukd.name,
        "kernel_source": {
            "kind": "kpack",
            "library": _kpack_rel(arch, rel_dir, group),
            "toc_key": toc_key,
            "symbol": ukd.symbol,
            "sha256": sha256,
            "signature": signature,
        },
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
    # Every shipped UKD carries the single shard arch, matching the KDP. Set after
    # the extra passthrough so a source multi-arch list cannot leak through.
    doc["arch"] = [arch]
    if ukd.origin_kind == "rocke":
        # Published last, onto the finished document, so `descriptor_digest` binds
        # the bytes that actually ship.
        agreement.publish(doc, ukd.observations, ukd.consumers)
    return doc


def _rewrite_passthrough_ukd(passthrough, arch, source_label=None):
    """Rewrite a pass-through UKD into shipped form.

    Only `arch` changes, narrowing to the shard; `kernel_source` ships as authored,
    so `source_file` stays the consuming binary's table key. The provenance block
    records the authored values and the rewritten fields, holds nothing
    machine-specific or time-varying, and sits at the top level because
    `kernel_source` accepts only loader-parsed keys.
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
    """Toolchain provenance for one UKD, dispatched on its producer."""
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

    Each distinct (source,build) variant .co is packed once under its own toc_key;
    inline UKDs are rewritten hsaco->kpack, stamping toc_key + sha256 + signature
    and moving build into a sibling provenance block, guarded against toc_key
    collisions. A pass-through UKD takes the shard arch and keeps its authored
    kernel_source. A shard with no compiled variant holds no archive, no `kpack/`
    directory, and an ArchResult with kpack_path=None.
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

    variant_bytes = {}
    variant_sha = {}
    variant_signature = {}
    variant_source_build = {}
    for ukd in _all_ukds():
        vk = ukd.variant_key
        toc_key = vk
        # The signature must cover everything that determines the compiled bytes.
        # (source, build) alone is blind on the rocke path, where build is ALWAYS
        # None: two rocke UKDs sharing a source module would present identical
        # signatures and a real toc_key collision would pass undetected.
        if ukd.origin_kind == "rocke":
            sig = (
                ukd.source,
                ukd.builder,
                json.dumps(ukd.spec, sort_keys=True),
            )
        else:
            sig = (ukd.source, json.dumps(ukd.build, sort_keys=True))
        if vk in variant_source_build and variant_source_build[vk] != sig:
            raise HkpPackError(
                f"toc_key collision: '{vk}' maps to two distinct "
                f"inputs {variant_source_build[vk]} and {sig}"
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
        # Keyed on (variant, symbol), not on the variant alone: two UKDs
        # differing only by entry point share one blob and one toc_key, and each
        # has its own argument list. Caching per variant would give the second
        # one the first's signature, which no fixture with one symbol per
        # variant can catch.
        signature_key = (vk, ukd.symbol)
        if signature_key not in variant_signature:
            variant_signature[signature_key] = kernel_signature(
                variant_bytes[vk], ukd.symbol, f"UKD '{ukd.id}'"
            )

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
                out_kds.append(
                    _rewrite_ukd_kpack(
                        e,
                        arch,
                        e.variant_key,
                        variant_sha[e.variant_key],
                        signature=variant_signature[(e.variant_key, e.symbol)],
                        toolchain_fields=_toolchain_for(e, hipcc, rocke_wheel_stamp),
                        # An inline UKD ships INSIDE this KDP file, so the
                        # runtime anchors its library on the KDP's directory,
                        # not the UKD's own notion of where it came from.
                        rel_dir=kdp.rel_dir,
                        group=group,
                    )
                )
        out_doc["kernelDescriptors"] = out_kds
        _write_text_at(
            out_arch_dir,
            kdp.rel_dir,
            kdp.filename,
            json.dumps(out_doc, indent=2) + "\n",
        )

    # A standalone UKD stays its own file in the shard, rewritten to kpack form
    # with this arch's kpack details. It is emitted only for arches whose
    # surviving KDPs referenced it (compile_intermediate only records those).
    for ukd in standalone:
        out_doc = _rewrite_ukd_kpack(
            ukd,
            arch,
            ukd.variant_key,
            variant_sha[ukd.variant_key],
            signature=variant_signature[(ukd.variant_key, ukd.symbol)],
            toolchain_fields=_toolchain_for(ukd, hipcc, rocke_wheel_stamp),
            # A standalone UKD is its own file, so it anchors on its own dir.
            rel_dir=ukd.rel_dir,
            group=group,
        )
        _write_text_at(
            out_arch_dir,
            ukd.rel_dir,
            ukd.filename,
            json.dumps(out_doc, indent=2) + "\n",
        )

    # A pass-through standalone UKD is its own file. It takes this shard's arch,
    # matching the KDP that references it.
    for ukd in inter.passthrough_standalone_ukds.values():
        out_doc = _rewrite_passthrough_ukd(ukd, arch, source_label)
        _write_text_at(
            out_arch_dir,
            ukd.rel_dir,
            ukd.filename,
            json.dumps(out_doc, indent=2) + "\n",
        )

    prune_result = prune(flat, arch)
    for generic in flat.generics():
        if generic.id in prune_result.reachable_generic_ids:
            _write_bytes_at(
                out_arch_dir,
                generic.rel_dir,
                generic.path.name,
                generic.path.read_bytes(),
            )

    return ArchResult(arch=arch, out_dir=out_arch_dir, kpack_path=kpack_path)


def run_pipeline(
    source_root,
    arches,
    out_root,
    hipcc,
    rocm_kpack_dir=None,
    inter_root=None,
    expected_sha256=None,
    rocke_wheel_stamp=None,
    group=GROUP_NAME,
    source_label=None,
    log=print,
):
    """One invocation over the full arch list: compile, prune, pack, install.

    Loads the one source root once, recursively, preserving each descriptor's
    authored subpath, then per arch compiles, prunes and packs. Producer selection
    is per-UKD on `kernel_source.kind`, so hip and rocKE descriptors combine into
    one kpack per arch, and a hip UKD's source resolves relative to the descriptor
    that named it. A pass-through kind is emitted as authored, so a root holding
    only those writes descriptors and no archive. An arch with no surviving KDP is
    skipped cleanly and logged 'no kernels for <arch>, skipping'; every arch
    skipping is a failure. An empty arch list installs nothing (exit 0).
    """
    out_root = Path(out_root)
    results = {}
    if not arches:
        return results

    kpack_mod, comp = load_kpack(rocm_kpack_dir)
    flat = load_flat_input(source_root, log=log)

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

    if failures:
        # Non-zero exit with partial output: the build fails loudly, but a
        # developer can still inspect what did succeed. Exiting 0 here would
        # resurrect the silent-empty-package class of defect.
        detail = "; ".join(f"{a}: {r}" for a, r in sorted(failures.items()))
        raise HkpPackError(
            f"packing failed for {len(failures)} of {len(arches)} arch(es) "
            f"[{detail}]. Arches that succeeded were written; the failed arches' "
            "output was discarded."
        )

    # The archive clause keys on what the root holds rather than on what survived
    # pruning: a compiling UKD that prunes out of every requested arch is
    # indistinguishable downstream from one whose archive went missing. Nothing
    # downstream restates either clause -- the build edge's OUTPUT is a stamp its
    # recipe touches unconditionally, and a staged tree holding nothing reads the
    # same there as a root that is legitimately empty. Only the packer knows the
    # kinds it walked and which arches pruned.
    arch_list = ", ".join(arches)
    if all(r.skipped for r in results.values()):
        raise HkpPackError(
            f"packing '{source_root}' produced nothing: no KDP survived arch "
            f"pruning for any of the {len(arches)} requested arch(es) "
            f"[{arch_list}], so every arch was skipped. A root wired to a pack "
            "was wired to ship descriptors, so this is a failure and not a "
            "clean skip. Check each KDP's 'arch' list against the requested "
            "arches. Archives are a separate matter and are not always "
            "required -- a root of only pass-through kinds legitimately packs "
            "descriptors and no archive -- but descriptors always are."
        )

    if _root_holds_compiling_source(flat) and not any(
        r.kpack_path for r in results.values()
    ):
        raise HkpPackError(
            f"packing '{source_root}' wrote descriptors but no archive, while "
            "the root holds at least one UKD of a compiling kind ('hip' or "
            f"'rocke'). Requested arch(es) [{arch_list}]; none produced a "
            "kpack. Either those UKDs pruned out of every shard that shipped, "
            "or their KDPs ship without them."
        )
    return results
