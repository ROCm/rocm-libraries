# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Batch compile and on-disk compile cache of the attention backward kernels.

Used by the backward sweep driver, the backward timing harness and the
backward tests' session pre-build:

* The cache itself (:class:`CompileCache`, :func:`entry_key`,
  :func:`backward_source_hash`, :class:`Toolchain`) lives in
  :mod:`kernels.common.attention_bwd_cache`, so the run glue
  (``attn_bwd_kernels`` with ``ROCKE_BWD_HSACO_CACHE``) reads the entries
  written here and the other way round; it is re-exported here.
* :func:`toolchain_identity` names the toolchain a process actually uses. It
  never imports torch: it pins the torch-bundled comgr (the one every timed
  process resolves) by path, see ``kernels.common.attention_bwd_comgr``, so
  a compile process and a timed process name the same comgr.
* :func:`compile_batch` compiles many specs with a thread or a process pool
  (``mode``), writes them to the cache and returns per-spec results with an
  error token (``not_built`` for a value whose emission is not built,
  ``compile_crash`` for an item whose compiling process dies (process mode,
  or any mode for a kernel that compiles in a fresh process of its own; a
  crash of an in-process compile in thread or serial mode ends the caller's
  process),
  ``compile:<ExceptionName>`` otherwise).
* :func:`plan_entries` gives every kernel of a plan from the cache, compiling
  the missing ones or (``compile_missing=False``, the timing attempts) failing
  with ``cache_miss`` instead.

Nothing here prints; callers print counts and tokens only.
"""

from __future__ import annotations

import concurrent.futures as cf
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import Any

from kernels.common import attention_bwd_cache as _cache

# The shared cache (see the module doc), under the names callers already use.
STAGES = _cache.STAGES
CACHE_SCHEMA = _cache.CACHE_SCHEMA
CacheEntry = _cache.CacheEntry
CachedArtifact = _cache.CachedArtifact
CompileCache = _cache.CompileCache
Toolchain = _cache.Toolchain
backward_source_hash = _cache.backward_source_hash
entry_key = _cache.entry_key
source_files = _cache.source_files
spec_identity = _cache.spec_identity
_ROOT = _cache._ROOT
_sha256_file = _cache._sha256_file
_CachedArtifact = CachedArtifact  # earlier name, kept for callers

# Error token of a timing attempt whose kernel is not in the cache.
CACHE_MISS = "cache_miss"


# --------------------------------------------------------------------------- identity
def toolchain_identity() -> Toolchain:
    """Flavor and comgr this process uses; never imports torch.

    Every timed process imports torch, and rocKE then resolves the
    torch-bundled ``libamd_comgr``. A compile process takes the same library
    without importing torch: the identity pins it by path first
    (:func:`kernels.common.attention_bwd_cache.toolchain_identity`).
    """
    return _cache.toolchain_identity()


# --------------------------------------------------------------------------- compile
def _builder(stage: str) -> Callable:
    if stage == "main":
        from kernels.common.attention_bwd import build_attn_bwd_main

        return build_attn_bwd_main
    if stage == "dq":
        from kernels.common.attention_bwd_dq import build_attn_bwd_dq

        return build_attn_bwd_dq
    if stage == "dense":  # the dense family's main kernel (per-arch builder)
        from kernels.common.attention_bwd_run import _build_dense

        return _build_dense
    if stage == "dense_dq":  # the dense family's split dQ kernel
        from kernels.common.attention_bwd_run import _build_dense_dq

        return _build_dense_dq
    from kernels.common.attention_bwd_aux import (
        build_attn_bwd_convert,
        build_attn_bwd_prep,
        build_attn_bwd_reduce,
    )

    return {
        "prep": build_attn_bwd_prep,
        "convert": build_attn_bwd_convert,
        "reduce": build_attn_bwd_reduce,
    }[stage]


def not_built_reasons(stage: str, spec: Any) -> list[str]:
    """Knob values of a resolved main spec whose emission is not built."""
    if stage == "dense":
        from kernels.common.attention_dense_bwd_spec import (
            dense_bwd_not_built_reasons,
        )

        return dense_bwd_not_built_reasons(spec)
    if stage != "main":
        return []
    from kernels.common.attention_bwd import _not_built

    return list(_not_built(spec))


def compile_one(stage: str, spec: Any, arch: str) -> tuple[str, bytes, float]:
    """Build and compile one kernel: ``(kernel name, hsaco, seconds)``.

    Raises ``NotImplementedError`` for a value whose emission is not built
    (checked before building, so it is never confused with a compile error).
    """
    reasons = not_built_reasons(stage, spec)
    if reasons:
        raise NotImplementedError("not built: " + "; ".join(reasons))
    from rocke.helpers.compile import compile_kernel

    t0 = time.perf_counter()
    kernel = _builder(stage)(spec, arch=arch)
    art = compile_kernel(kernel, arch=arch, capture_ir_text=False, backend="python")
    return art.kernel_name, art.hsaco, time.perf_counter() - t0


_WORKER_TOOLCHAIN: Toolchain | None = None


def _worker_init(comgr_digests: Mapping | None = None) -> None:
    """Pool worker start-up: the caller's comgr digests (keyed by path, size
    and mtime, so a changed file is hashed again), then this process's own
    identity, which a result is tagged with and checked against the cache's."""
    global _WORKER_TOOLCHAIN
    _cache.seed_comgr_digests(comgr_digests or {})
    _WORKER_TOOLCHAIN = toolchain_identity()


def _identity_tag(tc: Toolchain) -> dict:
    return {"comgr": tc.comgr, "flavor": tc.flavor, "emit": tc.emit}


def _compile_task(
    stage: str, spec: Any, arch: str, tc: Toolchain | None = None
) -> dict:
    """Compile one item; tagged with the compiling process's toolchain.

    ``tc`` is the caller's identity for in-process (serial / thread) compiles;
    a pool worker uses the identity it took at start-up.
    """
    from rocke.runtime.comgr import ComgrProcessDied

    tc = tc or _WORKER_TOOLCHAIN or toolchain_identity()
    tag = _identity_tag(tc)
    try:
        name, blob, secs = compile_one(stage, spec, arch)
    except NotImplementedError as ex:
        return {"error": "not_built", "detail": str(ex)[:300], **tag}
    except ComgrProcessDied as ex:
        # a kernel compiled in a fresh process (helpers.compile) whose process
        # died: the same toolchain crash a dying pool worker reports
        return {"error": COMPILE_CRASH, "detail": str(ex)[:300], **tag}
    except Exception as ex:  # noqa: BLE001 - recorded per spec
        return {
            "error": f"compile:{type(ex).__name__}",
            "detail": (type(ex).__name__ + ": " + str(ex))[:300],
            **tag,
        }
    return {"kernel_name": name, "hsaco": blob, "seconds": secs, **tag}


# Error token of an item whose compiling process died (a toolchain crash, for
# example a comgr segfault), observed with the item alone in a fresh process.
COMPILE_CRASH = "compile_crash"


def _process_pool(workers: int) -> cf.ProcessPoolExecutor:
    """Spawned workers, each reused for many compiles (one interpreter per
    worker, not per kernel)."""
    import multiprocessing as mp

    return cf.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        initializer=_worker_init,
        initargs=(_cache.comgr_digests(),),
    )


@dataclass
class CompileResult:
    stage: str
    spec: Any
    entry: CacheEntry | None = None
    error: str | None = None
    detail: str = ""
    seconds: float = 0.0
    cached: bool = False

    @property
    def ok(self) -> bool:
        return self.entry is not None


def compile_batch(
    items: Sequence[tuple[str, Any]],
    *,
    arch: str,
    cache: CompileCache,
    workers: int = 16,
    mode: str = "thread",
    deadline: float | None = None,
) -> list[CompileResult]:
    """Compile ``(stage, resolved spec)`` items through ``cache``.

    ``mode`` is ``"thread"`` (one process; comgr compiles outside the GIL),
    ``"process"`` (spawned workers, none of which imports torch; a result
    whose flavor, comgr or emission switches differ from the cache's is
    rejected as ``toolchain_mismatch``; an
    item whose worker dies is retried alone in a fresh worker and comes back
    ``compile_crash`` if that one dies too) or ``"serial"``.
    Cache hits are not recompiled; duplicate items compile once. No new
    compile starts after ``deadline`` (a ``time.time()`` value); those items
    come back with error ``deadline``. Results are in the order of ``items``.
    """
    if mode not in ("process", "thread", "serial"):
        raise ValueError(f"mode {mode!r} not in process / thread / serial")
    results: list[CompileResult | None] = [None] * len(items)
    todo: list[int] = []
    first: dict[str, int] = {}
    for i, (stage, spec) in enumerate(items):
        k = cache.key(stage, spec, arch)
        if k in first:
            continue  # filled from the first occurrence below
        first[k] = i
        hit = cache.read(stage, spec, arch)
        if hit is not None:
            results[i] = CompileResult(stage, spec, hit, cached=True)
        else:
            todo.append(i)

    def finish(i: int, out: Mapping[str, Any]) -> None:
        stage, spec = items[i]
        if "error" in out:
            results[i] = CompileResult(
                stage, spec, None, out["error"], out.get("detail", "")
            )
        elif (out.get("comgr"), out.get("flavor"), out.get("emit", "")) != (
            cache.toolchain.comgr,
            cache.toolchain.flavor,
            cache.toolchain.emit,
        ):
            results[i] = CompileResult(
                stage, spec, None, "toolchain_mismatch", str(out.get("comgr"))
            )
        else:
            entry = cache.write(
                stage,
                spec,
                arch,
                kernel_name=out["kernel_name"],
                hsaco=out["hsaco"],
                extra={"compile_seconds": out["seconds"]},
            )
            results[i] = CompileResult(stage, spec, entry, seconds=out["seconds"])

    def expired() -> bool:
        return deadline is not None and time.time() > deadline

    def drain(pool: cf.Executor, queue: list[int], width: int, extra: tuple):
        """Compile the items of ``queue`` (consumed) on ``pool``; returns the
        items lost to a broken process pool (the queue keeps the rest)."""
        pending: dict[cf.Future, int] = {}
        lost: list[int] = []
        while (queue or pending) and not lost:
            while queue and len(pending) < 2 * width:
                i = queue.pop(0)
                if expired():
                    results[i] = CompileResult(*items[i], None, "deadline")
                    continue
                try:
                    pending[pool.submit(_compile_task, *items[i], arch, *extra)] = i
                except BrokenProcessPool:
                    lost.append(i)
                    break
            if not pending:
                continue
            done, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            for fut in done:
                i = pending.pop(fut)
                try:
                    finish(i, fut.result())
                except BrokenProcessPool:
                    lost.append(i)
                except Exception as ex:  # noqa: BLE001 - recorded per spec
                    results[i] = CompileResult(
                        *items[i],
                        None,
                        f"compile:{type(ex).__name__}",
                        str(ex)[:300],
                    )
        # every future still pending shares the broken pool's fate
        return lost + list(pending.values())

    if mode == "serial" or (workers <= 1 and mode == "thread"):
        for i in todo:
            if expired():
                results[i] = CompileResult(*items[i], None, "deadline")
                continue
            finish(i, _compile_task(*items[i], arch, cache.toolchain))
    elif mode == "thread":
        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            drain(pool, list(todo), workers, (cache.toolchain,))
    else:
        # A worker that dies (a toolchain crash) breaks the whole pool and
        # every compile in flight with it: those are re-run one per fresh
        # single-worker pool, so the crash is charged to the item that causes
        # it (COMPILE_CRASH) and the other items still compile.
        queue = list(todo)
        suspects: list[int] = []
        width = max(1, min(workers, len(queue)))  # no idle worker spawns
        while queue:
            with _process_pool(width) as pool:
                suspects += drain(pool, queue, width, ())
        for i in suspects:
            if expired():
                results[i] = CompileResult(*items[i], None, "deadline")
                continue
            with _process_pool(1) as pool:
                if drain(pool, [i], 1, ()):
                    results[i] = CompileResult(
                        *items[i],
                        None,
                        COMPILE_CRASH,
                        "the compiling process died (alone in its pool)",
                    )
    out = []
    for i, (stage, spec) in enumerate(items):
        r = results[i]
        if r is None:  # duplicate of an earlier item in this batch
            src = results[first[cache.key(stage, spec, arch)]]
            r = CompileResult(
                stage, spec, src.entry, src.error, src.detail, cached=True
            )
        out.append(r)
    return out


# --------------------------------------------------------------------------- run cache
def plan_entries(
    plan: Any,
    cache: CompileCache,
    *,
    workers: int = 1,
    mode: str = "serial",
    compile_missing: bool = True,
) -> dict[str, CacheEntry]:
    """Every kernel of ``plan`` from ``cache`` (compiled on a miss).

    Returns ``{kernel name: entry}``; raises ``RuntimeError`` naming the
    token of a kernel that could not be compiled. ``compile_missing=False``
    only reads: a kernel not in the cache raises ``RuntimeError`` with the
    token ``cache_miss`` and nothing is compiled.
    """
    items, names = [], []
    for step in plan.launches:
        if step.kernel in names:
            continue
        names.append(step.kernel)
        items.append((step.stage, plan.specs[step.stage]))
    if compile_missing:
        res = compile_batch(
            items, arch=plan.arch, cache=cache, workers=workers, mode=mode
        )
    else:
        res = []
        for stage, spec in items:
            hit = cache.read(stage, spec, plan.arch)
            res.append(
                CompileResult(stage, spec, hit, cached=True)
                if hit is not None
                else CompileResult(stage, spec, None, CACHE_MISS, "not compiled")
            )
    out = {}
    for name, r in zip(names, res):
        if not r.ok:
            raise RuntimeError(f"{r.error}: {r.detail}")
        if r.entry.kernel_name != name:
            raise AssertionError(
                f"cached kernel {r.entry.kernel_name} differs from the plan's {name}"
            )
        out[name] = r.entry
    return out


def run_cache_for(
    plan: Any, entries: Mapping[str, CacheEntry], toolchain: Toolchain
) -> dict:
    """A kernel cache for ``attn_bwd_kernels`` loaded from cache entries.

    The keys follow ``attention_bwd_run.attn_bwd_kernels`` (kernel name, arch,
    flavor, comgr ROCm vintage); the modules are loaded from the cached code
    objects, so the run path never recompiles.
    """
    from kernels.common import attention_bwd_run
    from rocke.runtime.hip_module import Runtime

    rt = Runtime()
    flavor = attention_bwd_run._flavor()
    comgr = attention_bwd_run._comgr_version()
    out = {}
    for name, entry in entries.items():
        mod = rt.load_module(entry.hsaco)
        out[(name, plan.arch, flavor, comgr)] = (
            CachedArtifact(entry),
            mod,
            mod.get_function(entry.kernel_name),
        )
    return out
