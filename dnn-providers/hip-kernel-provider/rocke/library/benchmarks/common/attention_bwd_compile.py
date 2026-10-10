# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Batch compile and on-disk compile cache of the attention backward kernels.

Used by the backward sweep driver and the backward timing harness, inside the
GPU job that times the compiled code:

* :func:`toolchain_identity` names the toolchain a process actually uses: the
  LLVM IR flavor of the lowering (``ROCKE_LLVM_FLAVOR`` or the autodetected
  one) and the ``libamd_comgr`` rocKE loads (resolved path, ROCm vintage and a
  content digest of the library file). torch is imported first, so the
  torch-bundled comgr is the one resolved (as in every timed process).
* :func:`backward_source_hash` is a content hash of every Python source the
  backward kernels can depend on under the Python lowering backend: the whole
  ``rocke`` package (IR, lowering, helpers, analysis, runtime) and every module
  under ``library/kernels``. Line endings are normalised, so a synced copy and
  the working tree hash alike. Any change invalidates every cache entry
  (conservative: an entry is never reused across a source change).
* :class:`CompileCache` stores one code object per (kernel stage, resolved
  spec, arch, flavor, comgr, source hash). An entry written under another
  flavor or comgr is never read: the identity is part of the key and is
  re-checked against the entry's metadata on every read.
* :func:`compile_batch` compiles many specs with a thread or a process pool
  (``mode``), writes them to the cache and returns per-spec results with an
  error token (``not_built`` for a value whose emission is not built,
  ``compile_crash`` for an item whose compiling process dies (process mode
  only; a crash in thread or serial mode ends the caller's process),
  ``compile:<ExceptionName>`` otherwise).

Nothing here prints; callers print counts and tokens only.
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STAGES = ("main", "prep", "convert")
CACHE_SCHEMA = "rocke_bwd_compile_cache/v1"

_ROOT = Path(__file__).resolve().parents[3]  # .../rocke


# --------------------------------------------------------------------------- identity
def _sha256_file(path: str | os.PathLike, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


@dataclass(frozen=True)
class Toolchain:
    """The LLVM flavor and the comgr library a process compiles with."""

    flavor: str
    comgr_path: str | None
    comgr_rocm: str | None  # ROCm vintage of the resolved library, "major.minor"
    comgr_sha: str | None  # content digest of the resolved library file

    @property
    def comgr(self) -> str:
        """Compact comgr identity used in cache keys and records."""
        name = Path(self.comgr_path).name if self.comgr_path else "none"
        return f"{name}@rocm{self.comgr_rocm or '?'}#{self.comgr_sha or '?'}"

    def record(self) -> dict:
        return {
            "rocke_llvm_flavor": self.flavor,
            "comgr": self.comgr,
            "comgr_path": self.comgr_path,
            "comgr_rocm": self.comgr_rocm,
        }


def toolchain_identity(*, import_torch: bool = True) -> Toolchain:
    """Flavor and comgr this process uses (imports torch first by default).

    rocKE prefers the torch-bundled ``libamd_comgr`` only when torch is
    already in the process; every timed process imports torch, so the identity
    of a compile process must be taken the same way.
    """
    if import_torch:
        try:
            from rocke.runtime.comgr import prefer_bundled_lib

            prefer_bundled_lib()
        except Exception:  # noqa: BLE001 - no torch: the system comgr is used
            pass
    flavor = os.environ.get("ROCKE_LLVM_FLAVOR")
    if not flavor:
        try:
            from rocke.core.lower_llvm import _detect_llvm_flavor

            flavor = _detect_llvm_flavor()
        except Exception:  # noqa: BLE001
            flavor = "unknown"
    path = rocm = sha = None
    try:
        from rocke.runtime.comgr import resolved_lib_path, resolved_lib_rocm_version

        path = resolved_lib_path()
        v = resolved_lib_rocm_version()
        rocm = f"{v[0]}.{v[1]}" if v else None
        if path and os.path.exists(path):
            path = os.path.realpath(path)
            sha = _sha256_file(path, 12)
    except Exception:  # noqa: BLE001 - recorded as unknown
        pass
    return Toolchain(flavor, path, rocm, sha)


def source_files(root: Path | None = None) -> list[Path]:
    """Every source the backward kernels can depend on (see module doc)."""
    base = Path(root) if root is not None else _ROOT
    out = []
    for sub in ("platform/python/rocke", "library/kernels"):
        d = base / sub
        if d.is_dir():
            out.extend(p for p in d.rglob("*.py") if "__pycache__" not in p.parts)
    return sorted(out, key=lambda p: p.relative_to(base).as_posix())


def backward_source_hash(root: Path | None = None) -> str:
    """Content hash of :func:`source_files` (paths and LF-normalised bytes)."""
    base = Path(root) if root is not None else _ROOT
    h = hashlib.sha256()
    for p in source_files(base):
        h.update(p.relative_to(base).as_posix().encode())
        h.update(b"\0")
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
        h.update(b"\0")
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- keys
def spec_identity(stage: str, spec: Any) -> str:
    """Stable text of a resolved spec (dataclass repr is field-ordered)."""
    if stage not in STAGES:
        raise ValueError(f"stage {stage!r} not in {STAGES}")
    return f"{stage}|{type(spec).__name__}|{spec!r}"


def entry_key(
    stage: str, spec: Any, *, arch: str, toolchain: Toolchain, source_hash: str
) -> str:
    """Cache key: spec, arch, flavor, comgr and source hash."""
    blob = "\n".join(
        (
            spec_identity(stage, spec),
            arch,
            toolchain.flavor,
            toolchain.comgr,
            source_hash,
        )
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


# --------------------------------------------------------------------------- cache
@dataclass
class CacheEntry:
    key: str
    kernel_name: str
    hsaco: bytes
    meta: dict

    @property
    def hsaco_sha(self) -> str:
        return hashlib.sha256(self.hsaco).hexdigest()[:16]


class CompileCache:
    """Code objects on disk: ``<dir>/<key[:2]>/<key>.{hsaco,json}``.

    Writes are atomic (temporary file, then rename), so concurrent jobs that
    share the directory never read a partial entry. ``read`` returns ``None``
    for an entry whose metadata names another flavor, comgr, arch or source
    hash than the caller's (such an entry can only exist after a key
    collision or a hand-copied file; it is never used).
    """

    def __init__(
        self, root: str | os.PathLike, *, toolchain: Toolchain, source_hash: str
    ) -> None:
        self.root = Path(root)
        self.toolchain = toolchain
        self.source_hash = source_hash
        self.hits = 0
        self.misses = 0
        self.rejected = 0

    def key(self, stage: str, spec: Any, arch: str) -> str:
        return entry_key(
            stage,
            spec,
            arch=arch,
            toolchain=self.toolchain,
            source_hash=self.source_hash,
        )

    def _paths(self, key: str) -> tuple[Path, Path]:
        d = self.root / key[:2]
        return d / f"{key}.hsaco", d / f"{key}.json"

    def read(self, stage: str, spec: Any, arch: str) -> CacheEntry | None:
        key = self.key(stage, spec, arch)
        hp, mp = self._paths(key)
        if not (hp.exists() and mp.exists()):
            self.misses += 1
            return None
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
            blob = hp.read_bytes()
        except (OSError, json.JSONDecodeError):
            self.misses += 1
            return None
        want = {
            "schema": CACHE_SCHEMA,
            "arch": arch,
            "flavor": self.toolchain.flavor,
            "comgr": self.toolchain.comgr,
            "source_hash": self.source_hash,
            "spec": spec_identity(stage, spec),
        }
        if any(meta.get(k) != v for k, v in want.items()) or (
            hashlib.sha256(blob).hexdigest()[:16] != meta.get("hsaco_sha")
        ):
            self.rejected += 1
            return None
        self.hits += 1
        return CacheEntry(key, meta["kernel_name"], blob, meta)

    def write(
        self,
        stage: str,
        spec: Any,
        arch: str,
        *,
        kernel_name: str,
        hsaco: bytes,
        extra: Mapping[str, Any] | None = None,
    ) -> CacheEntry:
        key = self.key(stage, spec, arch)
        hp, mp = self._paths(key)
        hp.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "schema": CACHE_SCHEMA,
            "arch": arch,
            "flavor": self.toolchain.flavor,
            "comgr": self.toolchain.comgr,
            "source_hash": self.source_hash,
            "spec": spec_identity(stage, spec),
            "kernel_name": kernel_name,
            "hsaco_sha": hashlib.sha256(hsaco).hexdigest()[:16],
            "written": time.time(),
            **dict(extra or {}),
        }
        tag = f".{os.getpid()}.{time.time_ns()}.tmp"
        th, tm = hp.with_name(hp.name + tag), mp.with_name(mp.name + tag)
        th.write_bytes(hsaco)
        tm.write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")
        os.replace(th, hp)
        os.replace(tm, mp)
        return CacheEntry(key, kernel_name, hsaco, meta)


# --------------------------------------------------------------------------- compile
def _builder(stage: str) -> Callable:
    if stage == "main":
        from kernels.common.attention_bwd import build_attn_bwd_main

        return build_attn_bwd_main
    from kernels.common.attention_bwd_aux import (
        build_attn_bwd_convert,
        build_attn_bwd_prep,
    )

    return {"prep": build_attn_bwd_prep, "convert": build_attn_bwd_convert}[stage]


def not_built_reasons(stage: str, spec: Any) -> list[str]:
    """Knob values of a resolved main spec whose emission is not built."""
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


def _worker_init() -> None:
    global _WORKER_TOOLCHAIN
    _WORKER_TOOLCHAIN = toolchain_identity()


def _compile_task(
    stage: str, spec: Any, arch: str, tc: Toolchain | None = None
) -> dict:
    """Compile one item; tagged with the compiling process's toolchain.

    ``tc`` is the caller's identity for in-process (serial / thread) compiles;
    a pool worker uses the identity it took at start-up.
    """
    tc = tc or _WORKER_TOOLCHAIN or toolchain_identity()
    tag = {"comgr": tc.comgr, "flavor": tc.flavor}
    try:
        name, blob, secs = compile_one(stage, spec, arch)
    except NotImplementedError as ex:
        return {"error": "not_built", "detail": str(ex)[:300], **tag}
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
    import multiprocessing as mp

    return cf.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        initializer=_worker_init,
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
    ``"process"`` (spawned workers; each worker imports torch, and a result
    whose flavor or comgr differs from the cache's is rejected as
    ``toolchain_mismatch``; an item whose worker dies is retried alone in a
    fresh worker and comes back ``compile_crash`` if that one dies too) or
    ``"serial"``.
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
        elif (out.get("comgr"), out.get("flavor")) != (
            cache.toolchain.comgr,
            cache.toolchain.flavor,
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
        while queue:
            with _process_pool(max(workers, 1)) as pool:
                suspects += drain(pool, queue, max(workers, 1), ())
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
class _CachedArtifact:
    """Stand-in for a ``KernelArtifact`` of a cached code object."""

    def __init__(self, entry: CacheEntry) -> None:
        self.kernel_name = entry.kernel_name
        self.hsaco = entry.hsaco
        self.cache_key = entry.key
        self.hsaco_sha = entry.hsaco_sha


def plan_entries(
    plan: Any, cache: CompileCache, *, workers: int = 1, mode: str = "serial"
) -> dict[str, CacheEntry]:
    """Every kernel of ``plan`` from ``cache`` (compiled on a miss).

    Returns ``{kernel name: entry}``; raises ``RuntimeError`` naming the
    token of a kernel that could not be compiled.
    """
    items, names = [], []
    for step in plan.launches:
        if step.kernel in names:
            continue
        names.append(step.kernel)
        items.append((step.stage, plan.specs[step.stage]))
    res = compile_batch(items, arch=plan.arch, cache=cache, workers=workers, mode=mode)
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
            _CachedArtifact(entry),
            mod,
            mod.get_function(entry.kernel_name),
        )
    return out
