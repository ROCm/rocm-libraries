# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Compile a round of configs in parallel, ahead of the evaluator.

Compiling dominates the wall time of a search on a live backend, and it is CPU
bound, so a pool of processes speeds it up. The evaluator still compiles, checks
and times one config at a time on a single thread (GPU state is thread-bound); a
backend that defines ``precompile(items)`` is offered each round's fresh compiles
first (see :class:`~.measure.Backend`), builds them here, and serves the results
from its ``compile``::

    pool = ProcessCompilePool(compile_worker, workers=16)   # worker(config) -> payload

    def compile_(config, key):
        r = pool.take(key)                    # None: not prebuilt, compile inline
        if r is None:
            payload = compile_worker(config)
        elif not r.ok:
            return CompileOutcome(False, detail=r.detail)
        else:
            payload = r.payload
        return CompileOutcome(True, artifact=load(payload))   # load on the main thread

    backend = CallableBackend(compile_, check_, measure_, precompile=pool.precompile)

``worker`` is a module-level function (it is pickled) that takes the decoded knob
dict and returns a picklable payload, for example the code-object bytes and the
kernel name. Workers are started with the ``spawn`` method, so they carry no GPU
or runtime state; do the loading (and any device work) in the evaluator's process.
Pick ``workers`` no larger than the CPU cores allocated to the job.
"""

from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["Precompiled", "ProcessCompilePool"]


@dataclass(frozen=True)
class Precompiled:
    """What one worker returned for one config."""

    ok: bool
    payload: Any = None
    detail: str = ""
    seconds: float = 0.0


def _run(worker: Callable[[Mapping[str, object]], Any], config: Mapping[str, object]):
    t0 = time.perf_counter()
    try:
        return Precompiled(True, worker(config), "", time.perf_counter() - t0)
    except Exception as e:  # noqa: BLE001 - a compile failure is a result
        return Precompiled(False, None, f"{type(e).__name__}: {str(e)[:200]}", time.perf_counter() - t0)


class ProcessCompilePool:
    """A spawn-context process pool that prebuilds a round of configs."""

    def __init__(
        self,
        worker: Callable[[Mapping[str, object]], Any],
        *,
        workers: int | None = None,
        initializer: Callable[..., Any] | None = None,
        initargs: tuple = (),
        timeout: float | None = None,
    ):
        if workers is not None and int(workers) < 1:
            raise ValueError(f"ProcessCompilePool workers={workers}, expected >= 1")
        self.worker = worker
        self.workers = int(workers) if workers else max(1, min(os.cpu_count() or 1, 16))
        self.timeout = timeout
        self._initializer = initializer
        self._initargs = initargs
        self._executor: cf.ProcessPoolExecutor | None = None
        self._results: dict[str, Precompiled] = {}

    def _pool(self) -> cf.ProcessPoolExecutor:
        if self._executor is None:
            self._executor = cf.ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=mp.get_context("spawn"),
                initializer=self._initializer,
                initargs=self._initargs,
            )
        return self._executor

    def precompile(self, items: Sequence[tuple[Mapping[str, object], str]]) -> None:
        """Build ``[(config, key), ...]`` and keep each result for :meth:`take`.

        Returns when every item is done. With a ``timeout`` an item that has not
        finished by then is recorded as a failure and its worker stays busy until
        it ends (a running compile cannot be cancelled).
        """
        todo = [(c, k) for c, k in items if k not in self._results]
        if not todo:
            return
        pool = self._pool()
        futures = {k: pool.submit(_run, self.worker, dict(c)) for c, k in todo}
        deadline = None if self.timeout is None else time.monotonic() + self.timeout
        for k, f in futures.items():
            left = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                self._results[k] = f.result(timeout=left)
            except cf.TimeoutError:
                f.cancel()
                self._results[k] = Precompiled(False, None, f"compile exceeded {self.timeout:g} s", 0.0)
            except Exception as e:  # noqa: BLE001 - a broken pool is a failed item
                self._results[k] = Precompiled(False, None, f"{type(e).__name__}: {str(e)[:200]}", 0.0)

    def take(self, key: str) -> Precompiled | None:
        """The prebuilt result of ``key`` (removed), or ``None`` if there is none."""
        return self._results.pop(key, None)

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        self._results.clear()

    def __enter__(self) -> "ProcessCompilePool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
