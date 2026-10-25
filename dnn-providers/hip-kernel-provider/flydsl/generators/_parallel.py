# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Compile a generator's objects in parallel.

Only the compile runs in the pool. Each worker is a fresh process (``spawn``: no
MLIR state is forked) prepared by ``_flydsl_env.prepare(arch)``, as the generator's
own process is (one compile per worker), and inherits the environment the generator pinned before the pool
starts. The objects come back in table order, so checking them, writing them and
the manifest stay serial, and the bytes are what a serial build produces.

``FLYDSL_GEN_JOBS`` sets the worker count; the default is half the CPUs, since each
worker holds a FlyDSL/MLIR compile in memory. 1 compiles in this process.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from typing import Any

from . import _flydsl_env as env
from ._codeobject import GeneratorError


def _jobs(count: int) -> int:
    requested = os.environ.get("FLYDSL_GEN_JOBS")
    if requested and not requested.isdigit():
        raise GeneratorError(
            f"FLYDSL_GEN_JOBS must be a positive integer, got {requested!r}"
        )
    jobs = int(requested) if requested else max(1, (os.cpu_count() or 2) // 2)
    return max(1, min(jobs, count))


def _prepare_worker(arch: str) -> None:
    env.prepare(arch)


def compile_all(
    compile_one: Callable[[Any], Any],
    tasks: Sequence[Any],
    arch: str,
    label: Callable[[Any], str],
) -> list[Any]:
    """``compile_one(task)`` for every task, in table order.

    @p compile_one must be a module-level function and each task picklable. The
    first failure is raised and the tasks not yet started are cancelled.
    """
    jobs = _jobs(len(tasks))
    if jobs == 1:
        results = []
        for index, task in enumerate(tasks, start=1):
            print(f"[{index}/{len(tasks)}] {label(task)}", flush=True)
            results.append(compile_one(task))
        return results

    print(f"compiling {len(tasks)} object(s) on {jobs} worker(s)", flush=True)
    results: list[Any] = [None] * len(tasks)
    with ProcessPoolExecutor(
        max_workers=jobs,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_prepare_worker,
        initargs=(arch,),
        # One compile per process: FlyDSL restores a scoped LLVM option to its default
        # value, which LLVM reads as an explicit setting, so a reused worker would carry
        # the previous compile's options into the next object.
        max_tasks_per_child=1,
    ) as pool:
        futures = {
            pool.submit(compile_one, task): index for index, task in enumerate(tasks)
        }
        try:
            for done, future in enumerate(as_completed(futures), start=1):
                index = futures[future]
                try:
                    results[index] = future.result()
                except BrokenProcessPool as exc:
                    # A worker died outright (out of memory, a crash in the compiler):
                    # reported like any other failed object, not as a raw traceback.
                    raise GeneratorError(
                        f"{label(tasks[index])}: a compile worker died ({exc}); "
                        "retry with fewer workers (FLYDSL_GEN_JOBS)"
                    ) from exc
                print(f"[{done}/{len(tasks)}] {label(tasks[index])}", flush=True)
        except BaseException:
            pool.shutdown(wait=True, cancel_futures=True)
            raise
    return results
