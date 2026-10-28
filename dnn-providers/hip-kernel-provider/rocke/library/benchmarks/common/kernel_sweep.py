# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The two AOT benchmark modes: build the cache, then run any shape from it.

``--compile-all`` walks the configuration grid for an arch/dtype, builds and
compiles every variant that validates, and stores the HSACOs in an
:class:`~benchmarks.common.kernel_cache.KernelCache`. No GPU is needed and no
shape is involved --
that is what makes the artifacts reusable.

``--run-from-cache`` takes a concrete problem, asks the cache which of its
kernels can run it, and benchmarks those. Nothing is compiled.

The split exists because AOT kernels are shape-generic: the expensive step
(compilation) no longer depends on the problem, so it can be done once and
amortised over every shape the cache is later asked about.

The grid ``--compile-all`` walks, and how one identity is rebuilt and built,
is :mod:`kernels.conv.aot_jobs` (it sits below ``dispatch`` so the
dispatcher can build a registered instance too); this module enumerates,
compiles, caches and launches it.
"""

from __future__ import annotations

import gc
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    as_completed,
    wait,
)
from concurrent.futures.process import BrokenProcessPool
from typing import (
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)  # noqa: F401 (Tuple used in annotations)

from benchmarks.common.kernel_cache import (
    KernelCache,
    KernelIdentity,
    comgr_input_key,
    current_comgr_id,
    current_emitter_digest,
    current_llvm_flavor,
)
from kernels.conv.aot_jobs import (  # noqa: F401 (the grid and builder, re-exported)
    CACHE_DGRAD_MAX_SUB_GEMMS,
    CACHE_EPILOGUES,
    CACHE_GROUPED,
    CACHE_MAX_ACC_REGS,
    CACHE_PIPELINES,
    CACHE_TILE_K,
    CACHE_TILE_MN,
    CACHE_VECS,
    CACHE_WARP_MN,
    CACHE_WARP_TILE_MN,
    CACHE_WGRAD_GROUP_MERGES,
    CACHE_WGRAD_SPLIT_KS,
    CACHE_WGRAD_STREAMK_MODES,
    CACHE_WGRAD_STREAMK_REDUCTIONS,
    BuildJob,
    _direction_jobs,
    _geometry_list,
    _spec_is_valid,
    count_jobs,
    job_for_identity,
    kernel_for_job,
)


def _enumerate_chunk(
    direction: str,
    arch: str,
    dtype: str,
    target,
    split_ks: Sequence[int],
    max_sub_gemms: int,
    validate: bool,
    llvm_flavor: str,
    geometries: Sequence[tuple],
    streamk: Tuple[Sequence[str], Sequence[str]] = ((), ()),
) -> Tuple[int, List[Tuple[str, BuildJob]]]:
    """Walk one direction over ``geometries``.

    Returns the raw candidate count and the ``(key, job)`` pairs that are
    valid and first-seen within the chunk, in generator order. Duplicates
    across chunks are resolved by the caller, which merges the chunks in order
    and so keeps the same first occurrence a single serial walk would.
    """
    out: List[Tuple[str, BuildJob]] = []
    seen = set()
    n_raw = 0
    for job in _direction_jobs(
        direction,
        arch,
        dtype,
        target,
        split_ks,
        max_sub_gemms,
        geometries=geometries,
        llvm_flavor=llvm_flavor,
        streamk_modes=streamk[0],
        streamk_reductions=streamk[1],
    ):
        n_raw += 1
        key = job.identity.stable_hash()
        if key in seen:
            continue
        # An identity fixes the spec kwargs and capabilities the validator
        # reads, so a repeat would get the same verdict: skip it either way.
        seen.add(key)
        if validate and not _spec_is_valid(job, arch, dtype):
            continue
        out.append((key, job))
    return n_raw, out


# Per-process geometry lists, so a worker running many chunks rebuilds the
# grid once rather than once per chunk.
_WORKER_GEOMETRIES: Dict[Tuple[str, str, str], List[tuple]] = {}


def _enumerate_worker(payload):
    (
        direction,
        arch,
        dtype,
        split_ks,
        max_sub_gemms,
        validate,
        flavor,
        streamk,
        lo,
        hi,
    ) = payload
    from rocke.core.arch import ArchTarget

    target = ArchTarget.from_gfx(arch)
    geos = _WORKER_GEOMETRIES.get((arch, dtype, direction))
    if geos is None:
        geos = _WORKER_GEOMETRIES[(arch, dtype, direction)] = _geometry_list(
            target, dtype, direction
        )
    return _enumerate_chunk(
        direction,
        arch,
        dtype,
        target,
        split_ks,
        max_sub_gemms,
        validate,
        flavor,
        geos[lo:hi],
        streamk,
    )


def _worker_init() -> None:
    """Pool initializer: stop a compiler abort from writing a core dump.

    COMGR runs in-process, so an ``LLVM ERROR`` is an ``abort()`` in the
    worker. With the default core limit each one leaves a core the size of the
    worker's address space -- hundreds of MB -- and a grid that trips the same
    backend bug in thousands of variants fills the disk before the run ends.
    The crash is still reported: the broken pool is what the caller detects.
    """
    try:
        import resource  # POSIX only

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, OSError, ValueError):
        # No RLIMIT_CORE here (non-POSIX, or a hard limit we may not lower).
        # Cores are a disk-space problem, never a correctness one.
        pass


def enumerate_jobs(
    *,
    arch: str,
    dtype: str,
    target,
    directions: Sequence[str],
    split_ks: Sequence[int] = CACHE_WGRAD_SPLIT_KS,
    max_sub_gemms: int = CACHE_DGRAD_MAX_SUB_GEMMS,
    validate: bool = True,
    jobs: int = 1,
    log=None,
    log_every_s: float = 5.0,
    streamk_modes: Sequence[str] = (),
    streamk_reductions: Sequence[str] = CACHE_WGRAD_STREAMK_REDUCTIONS,
) -> List[BuildJob]:
    """All buildable jobs for the requested directions, deduped by identity.

    ``streamk_modes`` (empty by default) adds the opt-in wgrad stream-K
    kernels for those launch modes x ``streamk_reductions``.

    With ``validate=True`` (the default) each candidate is run through its
    direction's spec validator first, so the returned list is what will
    actually compile rather than the raw cross product.

    The raw cross product runs to millions of candidates. It is split into
    chunks of tile geometries and, with ``jobs > 1``, the chunks are walked in
    a process pool -- the work is pure-Python CPU, so threads would serialise
    on the GIL. The result does not depend on ``jobs``: chunks are merged in
    order, so the list (including its order) is the one a single serial walk
    produces. With ``log`` set a progress line is emitted at most every
    ``log_every_s`` seconds.
    """
    for direction in directions:
        if direction not in ("fwd", "wgrad", "dgrad"):
            raise ValueError(f"unknown direction {direction!r}")

    flavor = current_llvm_flavor()
    jobs = max(1, int(jobs))
    streamk = (tuple(streamk_modes), tuple(streamk_reductions))
    # Chunk bounds are per direction: wgrad's grid is longer than fwd's and
    # dgrad's (see _grid_axes), so one shared set of bounds would truncate it.
    payloads = []
    for direction in directions:
        n_geos = len(_geometry_list(target, dtype, direction))
        # Enough chunks per direction to keep every worker busy despite uneven
        # chunk costs, and to give the serial walk regular progress points.
        n_chunks = max(1, min(n_geos, jobs * 8 if jobs > 1 else 64))
        payloads += [
            (
                direction,
                arch,
                dtype,
                tuple(split_ks),
                max_sub_gemms,
                validate,
                flavor,
                streamk,
            )
            + (n_geos * i // n_chunks, n_geos * (i + 1) // n_chunks)
            for i in range(n_chunks)
        ]

    total = 0
    if log is not None:
        total = count_jobs(
            dtype=dtype,
            target=target,
            directions=directions,
            split_ks=split_ks,
            streamk_modes=streamk_modes,
            streamk_reductions=streamk_reductions,
        )
        log(f"  {total} candidates to check with {jobs} process(es)")

    n_raw = n_valid = 0
    started = last_log = time.perf_counter()
    results: List[Optional[List[Tuple[str, BuildJob]]]] = [None] * len(payloads)

    def _done(idx: int, chunk_raw: int, chunk_out) -> None:
        nonlocal n_raw, n_valid, last_log
        results[idx] = chunk_out
        n_raw += chunk_raw
        n_valid += len(chunk_out)
        if log is None:
            return
        now = time.perf_counter()
        if now - last_log >= log_every_s:
            last_log = now
            pct = 100.0 * n_raw / total if total else 0.0
            log(
                f"  enumerating: {n_raw}/{total} candidates checked "
                f"({pct:.1f}%), ~{n_valid} valid ({now - started:.0f}s)"
            )

    if jobs <= 1:
        # In-process, with the caller's target and the module's own grid.
        grids: Dict[str, List[tuple]] = {}
        for idx, payload in enumerate(payloads):
            direction, lo, hi = payload[0], payload[-2], payload[-1]
            geos = grids.get(direction)
            if geos is None:
                geos = grids[direction] = _geometry_list(target, dtype, direction)
            _done(
                idx,
                *_enumerate_chunk(
                    direction,
                    arch,
                    dtype,
                    target,
                    split_ks,
                    max_sub_gemms,
                    validate,
                    flavor,
                    geos[lo:hi],
                    streamk,
                ),
            )
    else:
        with ProcessPoolExecutor(max_workers=jobs, initializer=_worker_init) as pool:
            futures = {
                pool.submit(_enumerate_worker, payload): idx
                for idx, payload in enumerate(payloads)
            }
            for fut in as_completed(futures):
                _done(futures[fut], *fut.result())

    # Merge each direction's chunks in order, keeping the first occurrence.
    seen = set()
    per_direction: Dict[str, List[BuildJob]] = {d: [] for d in directions}
    for payload, chunk_out in zip(payloads, results):
        bucket = per_direction[payload[0]]
        for key, job in chunk_out:
            if key in seen:
                continue
            seen.add(key)
            bucket.append(job)

    # Round-robin across the requested directions rather than draining one
    # before starting the next. The full grid is far larger than any single
    # cache build, so callers routinely truncate it with --limit; taking them
    # in order would make a truncated build contain only forward kernels and
    # silently leave the backward directions unserved.
    out: List[BuildJob] = []
    streams = [iter(per_direction[d]) for d in directions]
    while streams:
        still_running = []
        for stream in streams:
            job = next(stream, None)
            if job is not None:
                out.append(job)
                still_running.append(stream)
        streams = still_running

    if log is not None:
        log(
            f"  enumerated {n_raw} candidates -> {len(out)} unique valid "
            f"({time.perf_counter() - started:.0f}s)"
        )
    return out


def _emit(build, job: BuildJob, arch: str, dtype: str):
    """Build and lower one job: ``(ComgrInput, content key, meta)``."""
    from rocke.helpers.compile import lower_kernel_for_comgr

    kernel, meta = build(job, arch, dtype)
    comgr_input = lower_kernel_for_comgr(kernel, arch=arch)
    return comgr_input, comgr_input_key(comgr_input), meta


def _emit_worker(payload):
    """Phase 1 worker: the job's binary content key, without compiling.

    Returns ``(job, key, meta, error)``. The lowered IR itself is not sent
    back -- it is large and most keys are already compiled; phase 2 re-emits
    the few it needs.
    """
    build, job, arch, dtype = payload
    try:
        _, key, meta = _emit(build, job, arch, dtype)
        return job, key, meta, None
    except Exception as exc:  # noqa: BLE001 - reported per job, never fatal
        return job, None, None, f"{type(exc).__name__}: {exc}"


def _compile_worker(payload):
    """Phase 2 worker: compile one binary. Returns ``(key, hsaco, name, error)``."""
    from rocke.runtime.comgr import build_hsaco_from_llvm_ir

    build, job, arch, dtype, key = payload
    try:
        comgr_input, got, _ = _emit(build, job, arch, dtype)
        if got != key:
            # The emitter is deterministic; a different key here means the
            # sources changed between the two phases.
            raise RuntimeError("kernel IR changed between emit and compile")
        hsaco, _ = build_hsaco_from_llvm_ir(
            comgr_input.llvm_text,
            isa=comgr_input.isa,
            options=list(comgr_input.options),
            fresh_process=comgr_input.fresh_process,
        )
        return key, hsaco, comgr_input.kernel_name, None
    except Exception as exc:  # noqa: BLE001 - reported per binary, never fatal
        return key, None, None, f"{type(exc).__name__}: {exc}"


# A main compile pool that keeps breaking is not a single bad kernel.
_MAX_POOL_RESTARTS = 20

_WORKER_DIED = (
    "worker process died: either the OOM killer, or the compiler aborted on "
    "this kernel's IR (an LLVM ERROR -- e.g. an instruction form the target "
    "does not have -- calls abort() rather than returning a diagnostic)"
)


# Each single-worker pool holds a handful of pipes; the slot count is capped so
# the reruns fit the process's open-file limit (often 1024 in containers).
_FDS_PER_POOL = 16
_MAX_ISOLATED_SLOTS = 16


def _isolated_slots(jobs: int) -> int:
    try:
        import resource  # POSIX only

        soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        if soft == resource.RLIM_INFINITY:
            soft = 1 << 16
    except (ImportError, OSError, ValueError):
        soft = 1024
    by_fds = max(1, (soft // 2) // _FDS_PER_POOL)
    return max(1, min(jobs, _MAX_ISOLATED_SLOTS, by_fds))


def _died_result(kind: str, payload):
    if kind == "emit":
        return (payload[1], None, None, _WORKER_DIED)
    return (payload[-1], None, None, _WORKER_DIED)


def _run_isolated(suspects, jobs: int):
    """Run each ``(kind, payload)`` with nothing else in its worker.

    A few single-worker pools ("slots") take the suspects one at a time, so a
    worker that dies takes down exactly one job. That job gets the worker's
    usual result tuple with :data:`_WORKER_DIED` as its error -- reported like
    any other failure instead of ending the run -- and only its slot gets a
    fresh pool. Yields ``(kind, payload, result)``.
    """
    pending = list(suspects)
    pending.reverse()
    slots = [
        ProcessPoolExecutor(max_workers=1, initializer=_worker_init)
        for _ in range(_isolated_slots(jobs))
    ]
    running: Dict = {}  # future -> (slot index, kind, payload)

    def submit(i: int) -> None:
        kind, payload = pending.pop()
        fn = _emit_worker if kind == "emit" else _compile_worker
        running[slots[i].submit(fn, payload)] = (i, kind, payload)

    try:
        for i in range(len(slots)):
            if pending:
                submit(i)
        while running:
            done, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for fut in done:
                i, kind, payload = running.pop(fut)
                try:
                    result = fut.result()
                except BrokenProcessPool:
                    result = _died_result(kind, payload)
                    slots[i].shutdown(wait=True)
                    slots[i] = ProcessPoolExecutor(
                        max_workers=1, initializer=_worker_init
                    )
                yield kind, payload, result
                if pending:
                    submit(i)
    finally:
        for pool in slots:
            pool.shutdown(wait=True)


def _pool_map(fn, payloads, jobs: int, on_result) -> None:
    if jobs <= 1:
        for payload in payloads:
            on_result(fn(payload))
        return
    with ProcessPoolExecutor(max_workers=jobs, initializer=_worker_init) as pool:
        futures = [pool.submit(fn, p) for p in payloads]
        for fut in as_completed(futures):
            on_result(fut.result())


def compile_all(
    *,
    cache: KernelCache,
    arch: str,
    dtype: str,
    target,
    directions: Sequence[str],
    jobs: int = 1,
    limit: Optional[int] = None,
    log=print,
    streamk_modes: Sequence[str] = (),
    streamk_reductions: Sequence[str] = CACHE_WGRAD_STREAMK_REDUCTIONS,
) -> int:
    """Populate ``cache`` with every implicit-GEMM variant for ``arch``/``dtype``.

    ``streamk_modes`` opts the wgrad stream-K kernels in (see
    :data:`CACHE_WGRAD_STREAMK_MODES`).

    Compilation is embarrassingly parallel and dominated by LLVM, so it fans
    out over processes; ``jobs`` defaults to one but the caller normally passes
    ``os.cpu_count()``.
    """
    log(
        f"AOT compile-all: enumerating variants for {arch}/{dtype} "
        f"({', '.join(directions)}) -- this walks the full tuning grid before "
        f"compilation starts"
    )
    return compile_jobs(
        cache=cache,
        all_jobs=enumerate_jobs(
            arch=arch,
            dtype=dtype,
            target=target,
            directions=directions,
            jobs=jobs,
            log=log,
            streamk_modes=streamk_modes,
            streamk_reductions=streamk_reductions,
        ),
        build=kernel_for_job,
        arch=arch,
        dtype=dtype,
        directions=directions,
        jobs=jobs,
        limit=limit,
        log=log,
    )


# (label, how to read it off an identity) for describe_jobs. Axes that take a
# single value within a direction are not printed.
_BREAKDOWN_AXES = (
    ("variant", lambda i: i.algorithm),
    ("pipeline", lambda i: i.pipeline),
    ("epilogue", lambda i: i.epilogue),
    (
        "k-loop",
        lambda i: "async_dma" if i.async_dma else "unroll_k" if i.unroll_k else "plain",
    ),
    ("grouped", lambda i: "yes" if i.grouped else "no"),
    (
        "split-K",
        lambda i: ("two-stage" if i.two_stage else "atomic") if i.split_k > 1 else "no",
    ),
    ("group_merge", lambda i: i.group_merge),
    (
        "stream-K",
        lambda i: f"{i.streamk}/{i.streamk_reduction}" if i.streamk != "off" else "no",
    ),
    ("stride x dilation", lambda i: f"{i.stride_h}x{i.dilation_h}"),
    ("filter", lambda i: f"{i.filter_h}x{i.filter_w}"),
    ("pad", lambda i: i.pad_h),
    ("cpg/kpg", lambda i: f"{i.cpg}/{i.kpg}"),
    ("atom", lambda i: f"{i.warp_tile_m}x{i.warp_tile_n}x{i.warp_tile_k}"),
    ("vec a/b/c", lambda i: f"{i.vector_size_a}/{i.vector_size_b}/{i.vector_size_c}"),
)


def describe_jobs(jobs: Sequence[BuildJob], log=print) -> None:
    """Log what a job list covers: per direction, the kernel count and how it
    splits along every swept axis (axes with a single value are omitted)."""
    by_direction: Dict[str, List[KernelIdentity]] = {}
    for job in jobs:
        by_direction.setdefault(job.identity.direction, []).append(job.identity)
    for direction, idents in by_direction.items():
        tiles = {(i.tile_m, i.tile_n, i.tile_k) for i in idents}
        warps = {(i.warp_m, i.warp_n) for i in idents}
        log(f"  {direction}: {len(idents)} kernels")
        if tiles != {(0, 0, 0)}:
            log(
                f"    geometry: {len(tiles)} block tiles (MxNxK) x "
                f"{len(warps)} warp layouts"
            )
        for label, get in _BREAKDOWN_AXES:
            counts: Dict[object, int] = {}
            for ident in idents:
                key = get(ident)
                counts[key] = counts.get(key, 0) + 1
            if len(counts) > 1:
                log(
                    f"    {label}: "
                    + ", ".join(f"{k} {n}" for k, n in sorted(counts.items()))
                )


def compile_jobs(
    *,
    cache: KernelCache,
    all_jobs: Sequence[BuildJob],
    build,
    arch: str,
    dtype: str,
    directions: Sequence[str],
    jobs: int = 1,
    limit: Optional[int] = None,
    log=print,
    log_every_s: float = 5.0,
) -> int:
    """Bring the cache up to date for ``all_jobs``; compile only what changed.

    Shared by every kernel family: the family only decides which jobs exist
    and how one's kernel is built (``build(job, arch, dtype) -> (kernel,
    meta)``). Three steps:

    1. **Up to date.** An entry built from this process's emitter sources and
       COMGR, whose binary is present, is skipped outright -- after a rebuild
       with nothing changed, nothing below runs.
    2. **Emit** (cheap). Every remaining job is built and lowered to LLVM IR
       and keyed by :func:`comgr_input_key`, without compiling. A job whose
       key already has a binary is linked to it on the spot.
    3. **Compile** (expensive). The first job to emit a key with no binary
       compiles it, in the same process pool as the emits, as soon as its
       emit is done; every job with that key is linked when it lands. So
       after an emitter change only the kernels whose code changed are
       recompiled, identities that emit the same code share one compile, and
       an interrupted run keeps every entry it finished.

    A worker that dies outright (OOM kill, native crash in COMGR) breaks the
    pool: its in-flight jobs are rerun one per process, a job that kills its
    worker again is reported as failed, and the run goes on.

    ``limit`` caps how many jobs go through steps 2-3 (smoke tests).
    """
    digest = current_emitter_digest()
    comgr_id = current_comgr_id()
    # Only the directions these jobs live in: a shared cache can hold ~10^6
    # entries of other families, and every one parsed here is resident in the
    # process the compile workers fork from.
    index = cache.index({j.identity.direction for j in all_jobs})

    def _up_to_date(job) -> bool:
        meta = index.get(job.identity.stable_hash())
        return (
            meta is not None
            and meta.get("emitter_digest") == digest
            and meta.get("comgr_id") == comgr_id
        )

    pending = [j for j in all_jobs if not _up_to_date(j)]
    up_to_date = len(all_jobs) - len(pending)
    to_check = f"{len(pending)} to check"
    if limit is not None and len(pending) > limit:
        # The limit counts this call's work only -- each dtype gets its own
        # --limit -- and up-to-date entries never count against it.
        to_check = f"{limit} of {len(pending)} out-of-date to check (--limit {limit})"
        pending = pending[:limit]

    describe_jobs(all_jobs, log)
    log(
        f"AOT compile-all: {len(all_jobs)} variants for {arch}/{dtype} "
        f"({', '.join(directions)}); {up_to_date} up to date, "
        f"{to_check} with {jobs} job(s)"
    )
    if not pending:
        log("AOT compile-all done: nothing to do")
        return 0

    started = last_log = time.perf_counter()
    state = dict(emitted=0, rejected=0, compiled=0, failed=0, linked=0, reused=0)
    members: Dict[str, List[Tuple[BuildJob, dict]]] = {}
    in_flight: set = set()

    def _link(key: str, job: BuildJob, meta: dict) -> None:
        cache.link(
            job.identity, key, dict(meta, emitter_digest=digest, comgr_id=comgr_id)
        )
        state["linked"] += 1

    def _progress(force: bool = False) -> None:
        nonlocal last_log
        now = time.perf_counter()
        if not force and now - last_log < log_every_s:
            return
        last_log = now
        log(
            f"  emitted {state['emitted']}/{len(pending)} "
            f"({100.0 * state['emitted'] / len(pending):.1f}%), "
            f"{len(members)} distinct binaries: {state['reused']} already "
            f"compiled, {state['compiled']} compiled, {len(in_flight)} compiling "
            f"({now - started:.0f}s)"
        )

    def on_emitted(result, submit_compile) -> None:
        job, key, meta, err = result
        state["emitted"] += 1
        if err is not None:
            state["rejected"] += 1
            if state["rejected"] <= 10:
                log(f"  [skip] {job.identity.label()}: {err}")
        else:
            first = key not in members
            members.setdefault(key, []).append((job, meta))
            if cache.has_blob(key):
                # Already compiled (by an earlier run, or earlier in this
                # one): link straight away.
                if first and key not in in_flight:
                    state["reused"] += 1
                _link(key, job, meta)
            elif key not in in_flight:
                # The first job to emit a missing binary compiles it; any
                # later job with the same key is linked when it lands.
                in_flight.add(key)
                submit_compile((build, job, arch, dtype, key))
        _progress()

    def on_compiled(result) -> None:
        key, hsaco, kernel_name, err = result
        in_flight.discard(key)
        if err is not None:
            state["failed"] += 1
            if state["failed"] <= 10:
                job = members[key][0][0]
                log(f"  [fail] {job.identity.label()}: {err}")
            return
        cache.put_blob(key, hsaco, kernel_name)
        state["compiled"] += 1
        # Entries are written as each binary lands, so an interrupted run
        # keeps everything it finished.
        for job, meta in members[key]:
            _link(key, job, meta)
        _progress()

    emit_payloads = [(build, j, arch, dtype) for j in pending]
    if jobs <= 1:
        for payload in emit_payloads:
            on_emitted(_emit_worker(payload), lambda p: on_compiled(_compile_worker(p)))
    else:
        # One pool for both steps: a binary is compiled as soon as the first
        # job emitting it is done, so a few slow-to-emit kernels never hold
        # the compile workers idle.
        #
        # Emits are fed through a bounded window rather than submitted up
        # front: the executor's queue is FIFO, so a compile submitted behind
        # hundreds of thousands of queued emits would not start until every
        # emit ran (and submitting them all takes seconds by itself). With
        # the window a compile waits behind at most ~2*jobs emits.
        window = 2 * jobs
        # Forked workers share this process's heap copy-on-write, but a
        # garbage collection in a worker writes to every tracked object's
        # header and so copies every page holding one: with a large cache
        # index resident, each of the 64 workers grew a private copy of it
        # (GBs apiece) and the run was OOM-killed. Frozen objects are left
        # alone by the collector.
        gc.freeze()
        emit_iter = iter(emit_payloads)
        # Work handed over from a pool that broke (see below).
        carry: List[Tuple[str, tuple]] = []
        restarts = 0
        while True:
            broken: List[Tuple[str, tuple]] = []
            with ProcessPoolExecutor(
                max_workers=jobs, initializer=_worker_init
            ) as pool:
                futures: Dict = {}

                def submit(kind: str, payload) -> None:
                    fn = _emit_worker if kind == "emit" else _compile_worker
                    futures[pool.submit(fn, payload)] = (kind, payload)

                def submit_compile(payload) -> None:
                    submit("compile", payload)

                def refill() -> None:
                    while len(futures) < window:
                        payload = next(emit_iter, None)
                        if payload is None:
                            return
                        submit("emit", payload)

                for kind, payload in carry:
                    submit(kind, payload)
                carry = []
                refill()
                try:
                    while futures:
                        # The timeout keeps the progress line coming while only
                        # a few slow kernels are left and nothing finishes for
                        # minutes.
                        ready, _ = wait(
                            list(futures),
                            timeout=log_every_s,
                            return_when=FIRST_COMPLETED,
                        )
                        _progress()
                        for fut in ready:
                            kind, payload = futures[fut]
                            result = fut.result()
                            del futures[fut]
                            if kind == "emit":
                                on_emitted(result, submit_compile)
                            else:
                                on_compiled(result)
                        refill()
                except BrokenProcessPool:
                    # A worker died outright -- killed by the OOM killer, or a
                    # native crash inside the compiler -- and took the whole
                    # pool with it. Every job still in flight is a suspect.
                    broken = list(futures.values())
                    futures.clear()
            if not broken:
                break
            restarts += 1
            if restarts > _MAX_POOL_RESTARTS:
                raise RuntimeError(
                    f"compile workers died {restarts} times; giving up. Either "
                    f"memory pressure -- rerun with a lower --jobs (was {jobs}) "
                    f"-- or this grid emits IR the backend aborts on for a whole "
                    f"family of variants, in which case the [fail] lines above "
                    f"name them and the emitter needs the fix, not the run."
                )
            log(
                f"  [warn] a worker process died (out of memory, or the compiler "
                f"aborting on its IR); rerunning its {len(broken)} in-flight "
                f"job(s) one per process to find the culprit. Lower --jobs if "
                f"this repeats."
            )
            # Rerun the suspects one per single-worker pool, so a job that
            # kills its worker again breaks only its own pool and is reported.
            # Compiles they trigger go to the next main pool.
            for kind, payload, result in _run_isolated(broken, jobs):
                if kind == "emit":
                    on_emitted(result, lambda p: carry.append(("compile", p)))
                else:
                    on_compiled(result)
            _progress()

    _progress(force=True)
    log(
        f"AOT compile-all done: {state['compiled']} compiled, "
        f"{state['linked']} entries updated, {up_to_date} up to date, "
        f"{state['rejected']} rejected, {state['failed']} failed "
        f"({time.perf_counter() - started:.1f}s)"
    )
    return 0


# ---------------------------------------------------------------------
# running a shape out of the cache
# ---------------------------------------------------------------------


def _launch_values_for(direction, problem, identity, ptrs, sizes, extras):
    from kernels.common.conv_args import ConvArgs

    tm, tn, tk = identity.tile_m, identity.tile_n, identity.tile_k
    if direction == "fwd":
        return ConvArgs.from_problem(problem, tile_m=tm, tile_n=tn).to_launch_values(
            *ptrs, *sizes
        )
    if direction == "wgrad":
        return ConvArgs.from_problem(
            problem, direction="wgrad", tile_m=tm, tile_n=tn, tile_k=tk
        ).to_launch_values(
            *ptrs,
            *sizes,
            split_k=extras.get("split_k", 1),
            ws_ptr=extras.get("ws_ptr"),
            ws_bytes=extras.get("ws_bytes"),
        )
    return ConvArgs.from_problem(
        problem, direction="dgrad", tile_m=tm, tile_n=tn
    ).to_launch_values(
        *ptrs,
        *sizes,
        sub_gemm_buf=extras["sub_gemm_buf"],
        num_sub_gemms=extras["num_sub_gemms"],
    )


def describe_cache(
    cache: KernelCache, log=print, directions: Optional[Sequence[str]] = None
) -> int:
    """Print what is in the cache, grouped by direction.

    ``directions`` limits the listing to the directions a run will use: every
    entry is a metadata file to parse, and a full implicit-GEMM cache holds
    hundreds of thousands of them -- reading all of them costs tens of seconds
    before a run that needs a few hundred.
    """
    by_direction: Dict[str, int] = {}
    entries = (
        cache.list_all()
        if directions is None
        else [e for d in directions for e in cache.list_all(d)]
    )
    for identity, _ in entries:
        by_direction[identity.direction] = by_direction.get(identity.direction, 0) + 1
    if not by_direction:
        if directions is None:
            log("AOT cache is empty.")
        else:
            log(f"AOT cache has no {', '.join(directions)} kernels.")
        return 2
    stale_by_direction = {}
    for direction, count in sorted(by_direction.items()):
        stale = cache.stale_entries(direction)
        stale_by_direction[direction] = stale
        note = f" ({stale} from older emitter sources)" if stale else ""
        log(f"  {direction:8s} {count} kernels{note}")
    stale_dirs = [d for d, n in stale_by_direction.items() if n]
    if stale_dirs:
        log(
            f"  [warn] {sum(stale_by_direction.values())} entries "
            f"({', '.join(stale_dirs)}) were built from different emitter sources "
            f"than this checkout -- built before a code change and not rebuilt "
            f"since. They still run; rerun --compile-all for those directions to "
            f"refresh them (only kernels whose code changed are recompiled)"
        )
    # What a rebuild from the names below reproduces against: the binary is a
    # function of these as well as of the kernel's identity.
    log(
        f"  this checkout: emitter {current_emitter_digest()[:12]}, "
        f"LLVM {current_llvm_flavor()}, comgr {current_comgr_id()}"
    )
    log(
        "  kernel names are complete identities; rebuild one with: "
        "python -m benchmarks.common.reproduce_kernel <name> --cache <cache dir>"
    )
    return 0
