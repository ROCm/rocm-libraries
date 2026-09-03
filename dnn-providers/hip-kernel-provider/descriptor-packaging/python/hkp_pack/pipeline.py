import copy
import hashlib
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import toolchain
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
from .kpack_resolver import load_kpack

# The archive group every root packs under unless it names its own. One archive ships per
# (group, arch), and the filename carries both, so two roots staged into one descriptor
# tree must not share a group -- otherwise they emit the same
# `<arch>/kpack/<group>_<arch>.kpack` and whichever copy lands second silently overwrites
# the other, leaving descriptors naming an archive that no longer holds their kernels.
# That collision is the whole reason the build used to stage one root under a `production/`
# subdirectory; naming the group per root removes the cause instead of dodging it.
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
    rel_dir: Path = Path(".")


@dataclass
class ArchKDP:
    id: str
    filename: str
    header: dict
    rel_dir: Path = Path(".")
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
    """Digest of a packed blob, recorded on the shipped UKD.

    Provenance only. The runtime parses `sha256` into KernelSource and never
    reads it back (Descriptors.hpp: "Carried, not checked"), so it is not an
    integrity guarantee on the consuming side.

    Still worth computing: `expected_sha256` cross-checks it at pack time, and
    it names the exact bytes a shipped kernel came from.
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
    when rel_dir is "." -- which is every flat layout, and is why this was not
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
):
    """Compile one UKD variant for arch, deduped into variant_co per kind.

    Dispatches on kernel_source.kind — producer selection is per-UKD, never
    per-folder. hip resolves its source relative to the descriptor that named it
    (`source_root / rel_dir / source`) and keys on (source, build). rocke keys on
    (source, builder, spec) and is location-independent (its source is a dotted
    module resolved by import), so source_root/rel_dir are accepted only for
    signature uniformity. Returns (variant_key, symbol, record_fields).
    """
    ks = ukd["kernel_source"]
    kind = ks["kind"]
    source = ks["source"]
    if kind == "hip":
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
        builder = ks["builder"]
        spec = ks["spec"]
        vk = _variant_key_for(ukd, rel_dir)
        if vk not in variant_co:
            co_path, captured = compile_rocke_variant(
                source, builder, spec, arch, inter_arch_dir
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


def _selected_entries(doc, arch, ukd_by_id):
    """Yield the entries of doc that ship for arch.

    Yields `(entry_id, ukd_doc, sdesc)`: for a standalone-UKD id ref, the id
    string, that UKD's doc, and its Descriptor; for an inline UKD, `None`, the
    entry dict itself, and `None`. The Descriptor rather than its rel_dir,
    because the walk needs its `path.name` for the error context and for the
    shipped filename as well as its `rel_dir` for the variant key.

    All three arch filters live here and nowhere else, so the prewarm and the
    serial walk cannot select different variant sets. `ukd_by_id` arrives as a
    parameter rather than being reached for through `flat`, which leaves the
    generator no way to enumerate a standalone UKD no KDP references: an orphan
    is legal input the walk never compiles.
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


def _prewarm_jobs(flat, source_root, arch):
    """The distinct variant jobs the walk will compile.

    Selection comes from the same generator the walk consumes, so the job set
    equals the walk's set by construction rather than by a second reading of
    the same filters. Dedup is on the variant key and keeps first-seen order,
    which is walk order: the pool compiles each distinct variant once, in the
    order the serial path would have reached them.

    A kind `_variant_key_for` declines to key is dropped here, leaving the walk
    to reach it and report whatever it produces.

    `out_dir` and `hipcc` are left empty because they belong to the pack run
    rather than to the selection; `_prewarm_variants` fills them in before
    dispatch.
    """
    ukd_by_id = flat.ukd_by_id()
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
                )
            )
    return jobs


def _compile_one_variant(job):
    """Compile one variant in a worker process, returning a result tuple.

    `(vk, co_path, symbol, None)` on success, `(vk, None, None, "Name: text")`
    on any failure. Failures are returned rather than raised: rocke and comgr
    exceptions are not guaranteed picklable, and an exception that cannot cross
    the process boundary takes the diagnosis with it.

    No key is computed here. `job.vk` was computed in the parent, under
    whatever key functions were in force there; a key recomputed in the child
    would resolve the real functions and disagree with the walk.
    """
    try:
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
            co_path, symbol = compile_rocke_variant(
                ks["source"],
                ks["builder"],
                ks["spec"],
                job.arch,
                job.out_dir,
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
        return job.vk, None, None, f"{type(exc).__name__}: {exc}"
    return job.vk, str(co_path), symbol, None


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

    The single place either key function is called, so the walk and the prewarm
    cannot key one variant two ways.

    `hip_variant_key` and `rocke_variant_key` are resolved as globals of this
    module on every call -- never through a function-local import, an alias
    bound at import time, or a recomputation inside a worker process. Each of
    those shortcuts binds the real function past any substitution made on this
    module.

    A kind that carries no compilable source yields None rather than raising.
    The walk stays the sole reporter of whatever failure such a UKD produces:
    were this to raise, the prewarm would pre-empt it with an error of its own,
    from a different call site and with a different traceback.
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
    log=print,
):
    """Compile this arch's distinct variants concurrently into the caches.

    The walk then finds each key already present and skips the expensive call.
    Records, symbols and doc rewriting stay entirely the walk's: this only
    populates two dicts.

    Fails fast, matching the serial path: the first failing variant in walk
    order raises and the queued jobs are cancelled, so a broken builder costs
    the jobs already in flight rather than the whole pack. Nothing is written
    into either cache when that happens -- a partly-filled cache would let the
    walk skip compiles whose artefacts were never produced.
    """
    jobs = [
        replace(job, out_dir=str(inter_arch_dir), hipcc=str(hipcc))
        for job in _prewarm_jobs(flat, source_root, arch)
    ]

    workers = _pack_jobs()
    if len(jobs) < 2 or workers < 2:
        return
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
        named = next((job.ukd.get("id") for job in jobs if job.vk == failure[0]), None)
        raise HkpPackError(
            f"variant '{named or failure[0]}' failed to compile for {arch}: "
            f"{failure[3]}"
        )

    for vk, co_path, symbol, _err in results:
        variant_co[vk] = Path(co_path)
        variant_symbol[vk] = symbol


def compile_intermediate(flat, source_root, arch, hipcc, inter_arch_dir, log=print):
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

    _prewarm_variants(
        flat,
        source_root,
        arch,
        hipcc,
        inter_arch_dir,
        variant_co,
        variant_symbol,
        log,
    )

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
                # A reference to a standalone UKD: compile it once per arch and
                # keep the string in the KDP; it ships as its own file. Listing
                # it precedes the compile-once check, so a UKD two KDPs
                # reference appears in both and is compiled for the first only.
                entries.append(sid)
                new_kds.append(sid)
                if sid in standalone_ukds:
                    continue
                sukd = entry
                where = f"standalone UKD {sdesc.path.name}"
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
                )
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


def _rewrite_ukd_kpack(
    ukd,
    arch,
    toc_key,
    sha256,
    toolchain_fields=None,
    rel_dir=Path("."),
    group=GROUP_NAME,
):
    """Rewrite a compiled UKD into shipped kpack form.

    `toolchain_fields` carries the fields describing what actually produced the kernel
    (hipcc version, resolved comgr, rocKE wheel digest) as opposed to what the
    descriptor asked for. Merged into provenance rather than the variant key: in
    the key, a wheel bump would rename every rocKE artifact including ones it
    could not affect.
    """
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
):
    """Pack a pruned intermediate arch into the shipped kpack release tree.

    Each distinct (source,build) variant .co is packed once under its own
    toc_key; inline UKDs are rewritten hsaco->kpack, stamping toc_key + sha256
    and moving build into a sibling provenance block. Guarded against toc_key
    collisions (distinct inputs mapping to one key).
    """
    arch = inter.arch
    out_arch_dir = Path(out_arch_dir)
    kpack_dir = out_arch_dir / KPACK_DIR_NAME
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
        # The signature must cover everything that determines the compiled
        # bytes, per producer. Keying on (source, build) alone is blind on the
        # rocke path, where build is ALWAYS None: two rocke UKDs sharing a
        # source module but differing in builder or spec would present identical
        # signatures, so a genuine toc_key collision would pass undetected and
        # one kernel would silently ship the other's bytes -- the same
        # silent-substitution class as the cross-root collision this work
        # removed.
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
                        e,
                        arch,
                        e.variant_key,
                        variant_sha[e.variant_key],
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
    log=print,
):
    """One invocation over the full arch list: compile, prune, pack, install.

    Loads the one source root once — recursively, preserving each descriptor's
    authored subpath — then for each arch compiles the targeting KDPs' variants,
    prunes, and packs. Producer selection is per-UKD on `kernel_source.kind`, so
    hip and rocKE descriptors coexist under one root (in child folders that scope
    them) and combine into one kpack per arch. A hip UKD's source resolves
    relative to the descriptor that named it. An arch with no surviving KDP is
    skipped cleanly (no folder, no kpack) and logged with 'no kernels for <arch>,
    skipping'. Empty arch list installs nothing (exit 0).
    """
    out_root = Path(out_root)
    results = {}
    if not arches:
        return results

    kpack_mod, comp = load_kpack(rocm_kpack_dir)
    flat = load_flat_input(source_root, log=log)

    if inter_root is None:
        inter_root = out_root.parent / "hkp-intermediate"
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
            # only once pack_arch returns cleanly. pack_arch creates
            # <out>/kpack/ before it validates anything, so writing in place
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
            )
            if out_arch_dir.exists():
                shutil.rmtree(out_arch_dir)
            staging.rename(out_arch_dir)
            results[arch] = replace(
                result,
                out_dir=out_arch_dir,
                kpack_path=out_arch_dir / KPACK_DIR_NAME / _kpack_filename(arch, group),
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
    return results
