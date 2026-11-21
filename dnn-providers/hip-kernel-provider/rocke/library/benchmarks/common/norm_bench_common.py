# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared compile/launch/report plumbing for the norm2d + batchnorm2d AOT sweeps.

``case_results.ResultsWriter`` and ``early_stop.case_key`` are both typed for
conv: ``ResultsWriter.row`` rejects any ``engine`` not in conv's own
``ENGINES`` tuple (``"direct"``/``"implicit"``/``"winograd"``), and
``case_key`` imports ``ConvGeometry`` and only knows how to read a
``DirectConvProblem``/``ConvProblem``. Forcing either one to also speak
norm/batchnorm would mean widening conv-only machinery for two families that
do not share its shape -- so this module defines the minimal equivalents
instead: a JSON-Lines writer with no family allowlist, and a key built from
whatever dims dict the request already carries.

Timing reuses ``rocke.runtime.time_launches`` directly rather than
``benchmarks.common.batched_timing``'s two-round probe/chunk protocol: that
machinery exists to rank a few thousand conv tile/pipeline candidates behind
one device drain per chunk. A norm/batchnorm sweep's candidate count is the
block_size x vec grid (a few dozen at most), so a plain warmup+iters timing
loop per candidate is the lightweight tier these two AOT scripts were built
against -- no persistent HSACO cache-dir, no chunked ranking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

import statistics

from rocke.helpers.compile import compile_kernel
from rocke.runtime.launcher import KernelLauncher, LaunchConfig
from rocke.runtime import time_launches, time_launches_staged


def case_key(dims: Mapping[str, int], *, dtype: str, family: str, extra: str = "") -> str:
    """One joinable key for a norm/batchnorm case: family + dims + dtype (+ extra)."""
    fields = ",".join(f"{k}={v}" for k, v in sorted(dims.items()))
    tail = f",{extra}" if extra else ""
    return f"{family}/{dtype}/{fields}{tail}"


class JsonlResultsWriter:
    """Appends one JSON object per case to a JSON Lines file. No-op without a path.

    Unlike ``benchmarks.common.case_results.ResultsWriter`` this takes an
    arbitrary ``**fields`` row rather than a fixed conv-shaped signature, and
    does not restrict the caller to a closed set of named engines.
    """

    def __init__(self, path: Optional[str]) -> None:
        self._path = Path(path) if path else None
        self._fh = None
        self._n = 0
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self._path.open("w", encoding="utf-8")

    @property
    def n_rows(self) -> int:
        return self._n

    def row(self, **fields: Any) -> None:
        if self._fh is None:
            return
        self._fh.write(json.dumps(fields, sort_keys=True) + "\n")
        self._fh.flush()
        self._n += 1

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def compile_candidate(spec: Any, candidate: Any, *, arch: str):
    """Build + lower one ``(candidate, spec)`` pair to a ``KernelArtifact``.

    Mirrors what the conv AOT sweeps' compile step does per candidate, minus
    the HSACO cache-dir: ``compile_kernel`` is called fresh each time, same as
    every non-cached candidate in those sweeps.
    """
    kernel_def = candidate.build(spec, arch)
    return compile_kernel(kernel_def, arch=arch)


def time_candidate(
    artifact: Any,
    *,
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    args: Any,
    warmup: int = 5,
    iters: int = 25,
    timing: str = "staged",
) -> float:
    """Launch ``artifact`` and return a timing in ms for one candidate.

    ``timing="block"`` is :func:`rocke.runtime.time_launches`: ``iters``
    launches back to back between one event pair, averaged. ``timing="staged"``
    (the default) is :func:`rocke.runtime.time_launches_staged`: every call is
    its own stalled sample, the reported number is the median -- this is what
    ``batched_timing``'s docstring calls "what a number compared against
    hipDNN or MIOpen has to be," so a comparison against MIOpenDriver must use
    this protocol, not ``block``.
    """
    launcher = KernelLauncher.from_artifact(artifact)
    config = LaunchConfig(grid=grid, block=block)

    def call_once() -> None:
        launcher(args, config=config)

    if timing == "block":
        return time_launches(call_once, warmup=warmup, iters=iters)
    if timing == "staged":
        samples = time_launches_staged(call_once, warmup=warmup, iters=iters)
        return statistics.median(samples)
    raise ValueError(f"unknown timing protocol: {timing!r}")
