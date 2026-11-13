# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Session pre-build of the backward kernels into the on-disk HSACO cache.

Opt-in: ``ROCKE_BWD_PREBUILD=1`` together with ``ROCKE_BWD_HSACO_CACHE=<dir>``
(the cache ``kernels.common.attention_bwd_run.attn_bwd_kernels`` reads). After
collection (``conftest.pytest_collection_finish``) this

1. dry-runs every selected test of a module that launches through the
   harness's ``run_case`` (``tests/_attention_bwd_harness.py``), in a small
   pool of spawned processes: ``run_case`` only plans (the same plan the real
   call launches) and returns the oracle gradients, and every compile or
   launch is refused, so each test walks as far as its control flow allows and
   names the kernels it would build (nothing touches the device and the
   pytest process keeps its module state);
2. compiles every distinct (stage, spec, arch) once with
   ``benchmarks.common.attention_bwd_compile.compile_batch`` in a process
   pool (each worker reused for many kernels, none importing torch), into
   the cache, with the toolchain identity of this process (the production
   comgr pinned by path; a worker whose toolchain differs is rejected, never
   written);
3. records a manifest (``<cache>/prebuild/<digest>.json``: the entry keys
   the selection needs, under a digest of the node ids, device arch, test /
   kernel / ``rocke`` sources, ``ROCKE_*`` / ``HIPDNN_*`` environment and
   toolchain), so a later pre-build of the same selection whose entries are
   all still cached skips steps 1 and 2 (no process is spawned);
4. prints one ``bwd-prebuild:`` line of counts.

The tests then load the cached code objects; a kernel the dry run did not
reach (a test that needs a fixture, or a run after a step the dry run cannot
take) still compiles in the test and is written to the cache. One process
pre-builds a cache directory at a time (a lock file), so pytest-xdist workers
and shards that share the directory compile each kernel once.

Environment: ``ROCKE_BWD_PREBUILD_WORKERS`` (default: the usable CPUs),
``ROCKE_BWD_PREBUILD_WAIT`` (seconds to wait for another pre-build's lock,
default 1800; a lock older than that is stale).
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import importlib
import os
import pickle
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

PREBUILD_ENV = "ROCKE_BWD_PREBUILD"
WORKERS_ENV = "ROCKE_BWD_PREBUILD_WORKERS"
WAIT_ENV = "ROCKE_BWD_PREBUILD_WAIT"
LOCK_NAME = ".prebuild.lock"
_HARNESS = "_attention_bwd_harness"
_DRY_RUN_PER_WORKER = 200  # tasks per dry-run worker before another is spawned
_DRY_RUN_MAX_WORKERS = 8


def enabled(env: Mapping[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    return env.get(PREBUILD_ENV, "").strip() not in ("", "0")


def default_workers() -> int:
    v = os.environ.get(WORKERS_ENV, "").strip()
    if v:
        return max(1, int(v))
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:  # not on Linux
        n = os.cpu_count() or 1
    return max(1, n)


# --------------------------------------------------------------------------- tasks
@dataclasses.dataclass(frozen=True)
class DryRunTask:
    """One selected test: its module, function and parameter values."""

    module: str
    function: str
    params: dict
    group: str  # tasks of one group share a dry-run worker (oracle reuse)


def _uses_harness(mod) -> bool:
    fn = getattr(mod, "run_case", None)
    return (
        callable(fn)
        and getattr(fn, "__name__", "") == "run_case"
        and getattr(fn, "__module__", "").rsplit(".", 1)[-1] == _HARNESS
    )


def _skip_marked(item) -> bool:
    if item.get_closest_marker("skip") is not None:
        return True
    for m in item.iter_markers("skipif"):
        conds = m.args or (m.kwargs.get("condition"),)
        if any(isinstance(c, bool) and c for c in conds):
            return True
    return False


def dry_run_tasks(items: Iterable[Any]) -> tuple[list[DryRunTask], int]:
    """``(tasks, items left out)`` of the pytest items the dry run can take:
    plain test functions of a module that launches through the harness's
    ``run_case``, not skip-marked, whose every argument is a parameter."""
    import inspect

    tasks, left = [], 0
    for item in items:
        mod = getattr(item, "module", None)
        fn = getattr(item, "function", None)
        if (
            mod is None
            or fn is None
            or getattr(item, "cls", None) is not None
            or not _uses_harness(mod)
            or _skip_marked(item)
        ):
            left += 1
            continue
        callspec = getattr(item, "callspec", None)
        params = dict(callspec.params) if callspec is not None else {}
        names = list(inspect.signature(fn).parameters)
        if any(n not in params for n in names):
            left += 1  # needs a fixture
            continue
        task = DryRunTask(
            module=mod.__name__,
            function=fn.__name__,
            params={n: params[n] for n in names},
            group=f"{mod.__name__}[{callspec.id}]" if callspec else item.nodeid,
        )
        try:
            pickle.dumps(task)
        except Exception:  # noqa: BLE001 - a parameter that cannot be sent
            left += 1
            continue
        tasks.append(task)
    return tasks, left


def _chunks(tasks: Sequence[DryRunTask], n: int) -> list[list[DryRunTask]]:
    """``n`` chunks, each group whole, balanced by task count."""
    groups: dict[str, list[DryRunTask]] = collections.defaultdict(list)
    for t in tasks:
        groups[t.group].append(t)
    out: list[list[DryRunTask]] = [[] for _ in range(max(1, n))]
    for g in sorted(groups.values(), key=len, reverse=True):
        min(out, key=len).extend(g)
    return [c for c in out if c]


# --------------------------------------------------------------------------- dry run
class _Planned(Exception):
    """Raised by the dry run in place of a compile or a launch."""


def _dry_run_chunk(tasks: Sequence[DryRunTask]) -> dict:
    """Run ``tasks`` with plan-only ``run_case``; ``{"specs", "counts"}``.

    Runs in a dry-run worker process (module state of the test modules is
    the worker's own). ``specs`` holds ``(stage, spec, arch)`` of every launch
    whose stage uses the shipped builder.
    """
    from unittest import mock

    import numpy as np

    from kernels.common import attention_bwd_run as run_mod

    specs: list[tuple] = []
    counts: collections.Counter = collections.Counter()

    def record(plan) -> None:
        for step in plan.launches:
            if run_mod._BUILDERS[step.stage] is run_mod._STOCK_BUILDERS[step.stage]:
                specs.append((step.stage, plan.specs[step.stage], plan.arch))

    def no_kernels(plan, *a, **k):
        record(plan)
        raise _Planned("launch")

    def no_compile(*a, **k):
        raise _Planned("compile")

    def plan_only(harness):
        def run_case(
            case,
            *,
            arch,
            config,
            overrides=None,
            request=None,
            thd=None,
            mutate_plan=None,
            inputs=None,
            **_device_only,
        ):
            plan, _lay = harness.case_plan(
                case,
                arch=arch,
                config=config,
                overrides=overrides,
                request=request,
                thd=thd,
                mutate_plan=mutate_plan,
            )
            record(plan)
            _inp, ref = harness.reference(case) if inputs is None else inputs
            return harness.CaseRun(
                case=case,
                plan=plan,
                dq=np.array(ref.dq, np.float64),
                dk=np.array(ref.dk, np.float64),
                dv=np.array(ref.dv, np.float64),
                ref=ref,
                integrity=[],
            )

        return run_case

    import rocke.helpers.compile as compile_mod

    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(run_mod, "attn_bwd_kernels", no_kernels))
        stack.enter_context(
            mock.patch.object(compile_mod, "compile_kernel", no_compile)
        )
        fakes: dict[str, Any] = {}  # harness module name -> plan-only run_case
        for t in tasks:
            try:
                mod = importlib.import_module(t.module)
                cur = mod.run_case
                if not any(cur is f for f in fakes.values()):
                    harness = sys.modules[cur.__module__]
                    if harness.__name__ not in fakes:
                        fakes[harness.__name__] = plan_only(harness)
                        stack.enter_context(
                            mock.patch.object(
                                harness, "run_case", fakes[harness.__name__]
                            )
                        )
                    if mod is not harness:
                        stack.enter_context(
                            mock.patch.object(mod, "run_case", fakes[harness.__name__])
                        )
                fn = getattr(mod, t.function)
            except Exception:  # noqa: BLE001 - counted; the test compiles itself
                counts["import_error"] += 1
                continue
            try:
                fn(**t.params)
                counts["completed"] += 1
            except KeyboardInterrupt:
                raise
            except BaseException:  # noqa: BLE001 - skip / fail / _Planned end it
                counts["stopped"] += 1
    return {"specs": specs, "counts": counts}


def collect_specs(
    tasks: Sequence[DryRunTask], workers: int
) -> tuple[list[tuple], collections.Counter]:
    """Dry-run ``tasks`` in spawned processes; ``(specs, counts)``."""
    import concurrent.futures as cf
    import multiprocessing as mp

    if not tasks:
        return [], collections.Counter()
    n = min(
        workers,
        _DRY_RUN_MAX_WORKERS,
        max(1, -(-len(tasks) // _DRY_RUN_PER_WORKER)),
    )
    specs: list[tuple] = []
    counts: collections.Counter = collections.Counter()
    with cf.ProcessPoolExecutor(
        max_workers=n, mp_context=mp.get_context("spawn")
    ) as pool:
        for out in pool.map(_dry_run_chunk, _chunks(tasks, n)):
            specs.extend(out["specs"])
            counts.update(out["counts"])
    return specs, counts


# --------------------------------------------------------------------------- lock
def _acquire(lock: Path, wait_s: float, stale_s: float | None = None) -> bool:
    """Take ``lock`` (create-exclusive); False when another holder outlives
    ``wait_s`` (the caller then pre-builds without it: duplicate compiles,
    never a wrong entry). A lock older than ``stale_s`` (default ``wait_s``)
    is a dead holder's and is taken over."""
    stale_s = wait_s if stale_s is None else stale_s
    deadline = time.time() + wait_s
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > stale_s:
                    lock.unlink()  # stale: its holder died
                    continue
            except OSError:
                continue  # released meanwhile
            if time.time() > deadline:
                return False
            time.sleep(1.0)
            continue
        with os.fdopen(fd, "w") as fh:
            fh.write(f"{os.getpid()}\n")
        return True


@contextlib.contextmanager
def prebuild_lock(root: Path, wait_s: float | None = None):
    root.mkdir(parents=True, exist_ok=True)
    lock = root / LOCK_NAME
    if wait_s is None:
        wait_s = float(os.environ.get(WAIT_ENV, "") or 1800)
    held = _acquire(lock, wait_s)
    try:
        yield held
    finally:
        if held:
            try:
                lock.unlink()
            except OSError:
                pass


# --------------------------------------------------------------------------- driver
def prebuild(
    items: Iterable[Any],
    *,
    cache_root: str | os.PathLike,
    workers: int | None = None,
    mode: str = "process",
) -> collections.Counter:
    """Dry-run ``items``, compile their kernels into ``cache_root`` once.

    Returns the counts printed on the ``bwd-prebuild:`` line: ``items``,
    ``dry_run`` (tasks), ``not_dry_run``, ``completed`` / ``stopped`` (dry
    runs), ``kernels`` (distinct), ``cached``, ``compiled`` and one count per
    error token of a kernel that did not compile. ``mode`` is the
    ``compile_batch`` mode (``process``; ``serial`` for tests).
    """
    from benchmarks.common import attention_bwd_compile as bc
    from kernels.common import attention_bwd_cache as ac

    items = list(items)
    workers = default_workers() if workers is None else max(1, workers)
    tasks, left = dry_run_tasks(items)
    counts: collections.Counter = collections.Counter(
        items=len(items), dry_run=len(tasks), not_dry_run=left
    )
    root = Path(cache_root).expanduser()
    with prebuild_lock(root) as held:
        if not held:
            counts["lock_wait_expired"] = 1
        cache = ac.CompileCache(
            root,
            toolchain=ac.toolchain_identity(),
            source_hash=ac.process_source_hash(),
        )
        manifest = root / "prebuild" / f"{manifest_digest(items, tasks, cache)}.json"
        done = _read_manifest(manifest, cache)
        if done is not None:  # this very selection was pre-built: no dry run
            counts.update(done)
            counts["manifest"] = 1
            return counts
        found, dry = collect_specs(tasks, workers)
        counts.update(dry)
        by_arch: dict[str, dict[str, tuple]] = collections.defaultdict(dict)
        for stage, spec, arch in found:
            by_arch[arch].setdefault(ac.spec_identity(stage, spec), (stage, spec))
        keys: list[str] = []
        errors: collections.Counter = collections.Counter()
        for arch in sorted(by_arch):
            batch = list(by_arch[arch].values())
            counts["kernels"] += len(batch)
            for r in bc.compile_batch(
                batch,
                arch=arch,
                cache=cache,
                workers=workers,
                mode=mode,
            ):
                if r.ok:
                    counts["cached" if r.cached else "compiled"] += 1
                    keys.append(r.entry.key)
                else:  # the test that launches it reports the failure
                    errors[r.error] += 1
        counts.update(errors)
        _write_manifest(manifest, keys, errors)
    return counts


# --------------------------------------------------------------------------- manifest
# Environment that cannot change which kernels a selection builds.
_MANIFEST_ENV_SKIP = frozenset(
    (PREBUILD_ENV, WORKERS_ENV, WAIT_ENV, "ROCKE_BWD_HSACO_CACHE")
)
# A file-naming variable whose file content the toolchain already digests
# (``Toolchain.comgr``): the name enters the digest, the large library is not
# read a second time.
_MANIFEST_ENV_NAME_ONLY = frozenset(("ROCKE_COMGR_LIB",))


def manifest_digest(items: Sequence[Any], tasks: Sequence[DryRunTask], cache) -> str:
    """Digest of everything the dry run's result depends on: the selected
    node ids, the device arch the test modules see, the test, ``rocke`` and
    kernel sources, the ``ROCKE_*`` / ``HIPDNN_*`` environment (with the
    content of a variable that names a file) and the cache's toolchain.

    A stale or wrong manifest can only skip the pre-build (the tests then
    compile what they miss); the tests never read code objects through it.
    """
    import hashlib

    h = hashlib.sha256()

    def put(*parts) -> None:
        for p in parts:
            h.update(str(p).encode())
            h.update(b"\0")

    for nodeid in sorted(getattr(i, "nodeid", repr(i)) for i in items):
        put("item", nodeid)
    archs = set()
    for t in tasks:
        mod = sys.modules.get(t.module)
        fn = getattr(mod, "run_case", None)
        harness = sys.modules.get(getattr(fn, "__module__", ""), None)
        archs.add(str(getattr(harness, "ARCH", None)))
    put("arch", *sorted(archs))
    tests_dir = Path(__file__).resolve().parent
    for p in sorted(tests_dir.rglob("*.py")):
        if "__pycache__" not in p.parts:
            put(p.relative_to(tests_dir).as_posix())
            h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    put("sources", cache.source_hash, "torch" in sys.modules)
    tc = cache.toolchain
    put("toolchain", tc.flavor, tc.comgr, tc.emit)
    for k in sorted(os.environ):
        if k in _MANIFEST_ENV_SKIP or not k.startswith(("ROCKE_", "HIPDNN_")):
            continue
        v = os.environ[k]
        put("env", k, v)
        if v and k not in _MANIFEST_ENV_NAME_ONLY and os.path.isfile(v):
            put(Path(v).read_bytes())
    return h.hexdigest()[:32]


def _read_manifest(path: Path, cache) -> dict | None:
    """Counts of an earlier pre-build of the same selection when every code
    object it built is still in the cache, else ``None``."""
    import json

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        keys = list(data["keys"])
        errors = dict(data.get("errors", {}))
    except (OSError, ValueError, KeyError, TypeError):
        return None
    for k in keys:
        hp, mp = cache._paths(k)
        if not (hp.exists() and mp.exists()):
            return None
    return {"kernels": len(keys) + sum(errors.values()), "cached": len(keys), **errors}


def _write_manifest(path: Path, keys: Sequence[str], errors: Mapping) -> None:
    import json

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        tmp.write_text(
            json.dumps({"keys": sorted(keys), "errors": dict(errors)}),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    except OSError:  # a manifest only saves the next dry run
        pass


def format_counts(counts: Mapping[str, int], toolchain: Any = None) -> str:
    keys = ["items", "dry_run", "not_dry_run", "completed", "stopped"]
    keys += ["import_error", "kernels", "cached", "compiled"]
    extra = sorted(k for k in counts if k not in keys)
    parts = [f"{k}={counts.get(k, 0)}" for k in keys + extra]
    if toolchain is not None:
        parts.append(f"flavor={toolchain.flavor} comgr={toolchain.comgr}")
    return "bwd-prebuild: " + " ".join(parts)


def prebuild_session(session) -> collections.Counter | None:
    """``pytest_collection_finish`` entry point (see the module doc)."""
    if not enabled():
        return None
    from kernels.common import attention_bwd_cache as ac

    root = os.environ.get(ac.HSACO_CACHE_ENV, "").strip()
    if not root:
        import pytest

        raise pytest.UsageError(f"{PREBUILD_ENV}=1 needs {ac.HSACO_CACHE_ENV}=<dir>")
    counts = prebuild(session.items, cache_root=root)
    line = format_counts(counts, ac.toolchain_identity())
    tr = session.config.pluginmanager.get_plugin("terminalreporter")
    if tr is not None:
        tr.write_line(line)
    else:
        print(line, flush=True)
    return counts
