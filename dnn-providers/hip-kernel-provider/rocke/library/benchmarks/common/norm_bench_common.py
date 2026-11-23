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

That "a few dozen at most" stopped holding for batchnorm2d once
``n_groups > 1`` grew a masked tail (see ``instances/common/batchnorm2d.py``):
``is_valid_spec`` no longer restricts ``hw_threads`` to a fixed set, so a
shape's valid-candidate count can run into the hundreds. ``--fast-rank``
(:func:`add_fast_rank_args`, :func:`rank_candidates`) is the same fix conv
uses for the same symptom -- rank every candidate behind one device drain per
chunk, then time only the best ``--fast-rank-finalists`` in full -- wired
through :func:`~benchmarks.common.batched_timing.rank` rather than
:mod:`benchmarks.common.batched_timing` internals, so a sweep stays opt-in to
the chunked path instead of paying it by default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

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


def add_fast_rank_args(parser) -> None:
    """Add ``--fast-rank`` and its tuning flags to a norm/batchnorm sweep's parser.

    Mirrors ``benchmark_direct_conv.py``'s/``benchmark_implicit_gemm_conv.py``'s
    ``--fast-rank*`` flags, minus ``--fast-rank-scratch`` (no workspace
    accounting in this family). Defaults come straight from
    :mod:`benchmarks.common.batched_timing` so the two sweeps stay consistent.
    """
    from benchmarks.common import batched_timing as bt

    parser.add_argument(
        "--fast-rank", action="store_true",
        help="rank candidates with one device drain per chunk instead of "
        "timing every one in full; only the best --fast-rank-finalists are "
        "then measured with the full --warmup/--iters protocol. Use when the "
        "candidate grid is large enough that per-candidate drains dominate "
        "the sweep (see benchmarks.common.batched_timing).",
    )
    parser.add_argument("--fast-rank-finalists", type=int, default=bt.DEFAULT_FINALISTS)
    parser.add_argument("--fast-rank-chunk", type=int, default=bt.DEFAULT_CHUNK)
    parser.add_argument("--fast-rank-warmup", type=int, default=bt.DEFAULT_PROBE_WARMUP)
    parser.add_argument("--fast-rank-iters", type=int, default=bt.DEFAULT_PROBE_ITERS)
    parser.add_argument("--fast-rank-primer", type=int, default=bt.DEFAULT_PRIMER)


# One already-compiled candidate ready to rank: a label for reporting, its
# artifact, launch grid/block, and the kernarg dict every candidate in a
# fast-rank call shares (same request shape, same buffers -- ranking times,
# it does not verify, so sharing input buffers across candidates is fine).
Plan = Tuple[str, Any, Tuple[int, int, int], Tuple[int, int, int], Any]


def rank_candidates(
    plans: Sequence[Plan], *, args: Any, timing: str
) -> List[Any]:
    """Rank already-compiled ``plans`` with :mod:`benchmarks.common.batched_timing`.

    Mirrors ``benchmark_direct_conv.py``'s ``_fast_rank_plans``: every plan is
    probed with one device drain per chunk, the best ``--fast-rank-finalists``
    are kept, and only those are timed again with the caller's full
    ``--warmup``/``--iters`` under ``timing`` -- the same protocol
    :func:`time_candidate` uses, so a finalist's reported ``ms`` is directly
    comparable to a plain (non-ranked) sweep's. A plan the probe ranked out
    keeps no ``ms`` and is never reported as the best.

    Returns one :class:`~benchmarks.common.batched_timing.Ranked` per plan, in
    ``plans`` order (``ranked[i].index == i``).
    """
    from benchmarks.common import batched_timing

    n = len(plans)

    def make_run(i: int):
        _label, artifact, grid, block, call_args = plans[i]
        launcher = KernelLauncher.from_artifact(artifact)
        config = LaunchConfig(grid=grid, block=block)

        def run() -> None:
            launcher(call_args, config=config)

        return run

    def on_probe(r) -> None:
        if r.failed:
            return
        label = plans[r.index][0]
        print(f"  [{r.index + 1}/{n}] probe {r.probe_ms:8.3f} ms  {label}", flush=True)

    ranked = batched_timing.rank(
        n,
        make_run,
        warmup=args.warmup,
        iters=args.iters,
        probe_warmup=args.fast_rank_warmup,
        probe_iters=args.fast_rank_iters,
        finalists=args.fast_rank_finalists,
        chunk=args.fast_rank_chunk,
        primer=args.fast_rank_primer,
        timing=timing,
        on_probe=on_probe,
    )
    n_probed = sum(1 for r in ranked if r.probe_ms is not None)
    n_final = sum(1 for r in ranked if r.is_finalist)
    print(
        f"  probe: {n_probed}/{n} timed, {n_final} finalist(s) measured in "
        f"full ({timing} timing)",
        flush=True,
    )
    return ranked
