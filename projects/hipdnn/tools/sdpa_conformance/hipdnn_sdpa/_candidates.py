# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The candidates (kernels) an ingestor engine offers for a problem, from ``hipdnn_bench enumerate``.

hipDNN has no Graph binding that lists candidates, and the conformance tool adds none: it runs the
``hipdnn_bench`` of the hipDNN prefix under test (``$HIPDNN_SDPA_PREFIX/bin/hipdnn_bench``,
``tools/bench/HipdnnBench.cpp``)::

    hipdnn_bench enumerate --graph <file> --engine-name <engine> \\
        --plugin-dir <prefix>/lib/hipdnn_plugins/engines [--offset N] [--limit N]

``enumerate`` must be the first argument (``HipdnnBench.cpp:164``). The graph file holds the
problem's graph JSON. Stdout is one page, ``pageJson`` (``CandidatePageJson.hpp:246``): ``candidates``
(each ``id``, ``knob_settings`` as an object of knob name to value, ``kernel_features``),
``total_count``, ``offset`` and ``next_offset`` (null on the last page). A candidate's ``id`` is the
kernel id the ingestor logs when it builds that candidate (``prep=<id>|dkdv=<id>|dq=<id>`` for a
slotted engine). An engine that cannot enumerate exits 1 with
``Candidate enumeration failed: <message>`` on stderr: ``Engine does not support matched-catalog
enumeration`` (``EngineQueries.hpp:422-425``; ASM and every other non-ingestor engine) or the
``Ambiguous enrolled knob tuple ...`` error of a catalog whose candidates share their knobs
(``GenericPlanBuilder.hpp:738-752``).

:func:`list` pages until ``next_offset`` is null and caches per (problem, engine), least recently used out
after :data:`CACHE_LIMIT` entries: the result, a :class:`NotEnumerable` included, is returned as it was
first read. A ``hipdnn_bench`` that dies on a signal, exits without a message or times out raises
``RuntimeError`` and is not cached; only the engine's own refusal is a :class:`NotEnumerable`. The subprocess call is
:data:`_run`, replaced by tests.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import tempfile

import conformance_contract as contract

PAGE_LIMIT = 10000  # the largest page --limit accepts
_FAILED = "Candidate enumeration failed: "
CACHE_LIMIT = 64
# (sha1 of the graph JSON, engine) -> list of Candidate | NotEnumerable, least recently used first
_CACHE: dict = {}
_BENCH_TIMEOUT = 600  # seconds for one ``enumerate`` page
_run = subprocess.run


@dataclasses.dataclass(frozen=True)
class Candidate:
    """One kernel of an engine: ``id`` as the ingestor logs it, ``knobs`` as ``(name, value)``
    pairs in name order, the settings ``create_execution_plan_ext`` takes to build it.
    """

    id: str
    knobs: tuple


@dataclasses.dataclass(frozen=True)
class NotEnumerable:
    """The engine cannot enumerate this problem's candidates; ``message`` says why."""

    message: str


def _bench_command(graph_path: str, engine: str, offset: int) -> tuple:
    prefix = os.environ.get(contract.PREFIX_ENV)
    if not prefix:
        raise RuntimeError(
            f"{contract.PREFIX_ENV} is not set: it names the hipDNN prefix whose bin/ holds "
            "hipdnn_bench (the conformance runner exports --hipdnn-prefix there)"
        )
    return [
        f"{prefix}/{contract.BENCH_RELATIVE}",
        "enumerate",
        "--graph",
        graph_path,
        "--engine-name",
        engine,
        "--plugin-dir",
        f"{prefix}/{contract.BENCH_PLUGIN_DIR}",
        "--offset",
        str(offset),
        "--limit",
        str(PAGE_LIMIT),
    ]


def _failure(done) -> NotEnumerable:
    """The engine's own refusal (exit 1 with a message); anything else is a crash, see ``_read``."""
    text = (done.stderr or "").strip()
    if _FAILED in text:
        text = text.split(_FAILED, 1)[1].strip()
    return NotEnumerable(text or f"hipdnn_bench exited with status {done.returncode}")


def _read(graph_json: str, engine: str):
    entries, offset = [], 0
    with tempfile.TemporaryDirectory(prefix="sdpa-enumerate-") as scratch:
        graph_path = os.path.join(scratch, "graph.json")
        with open(graph_path, "w") as out:
            out.write(graph_json)
        while offset is not None:
            command = _bench_command(graph_path, engine, offset)
            try:
                done = _run(
                    command, capture_output=True, text=True, timeout=_BENCH_TIMEOUT
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    f"{command[0]} enumerate timed out after {_BENCH_TIMEOUT} s"
                ) from None
            except OSError as exc:
                return NotEnumerable(f"cannot run {command[0]}: {exc}")
            if done.returncode < 0 or (
                done.returncode != 0 and not (done.stderr or "").strip()
            ):
                raise RuntimeError(
                    f"{command[0]} enumerate died (status {done.returncode}) with no message"
                )
            if done.returncode != 0:
                return _failure(done)
            try:
                page = json.loads(done.stdout)
                entries += [
                    Candidate(c["id"], tuple(sorted(c["knob_settings"].items())))
                    for c in page["candidates"]
                ]
                offset = page["next_offset"]
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                return NotEnumerable(f"unreadable enumeration page: {exc!r}")
    return entries


def list(graph_json: str, engine: str):
    """The candidates of ``engine`` for the problem ``graph_json``, in catalog order: a list of
    :class:`Candidate`, or a :class:`NotEnumerable` carrying the reason. Cached per
    (problem, engine)."""
    key = (hashlib.sha1(graph_json.encode()).hexdigest(), engine)
    if key in _CACHE:
        _CACHE[key] = _CACHE.pop(key)  # most recently used
        return _CACHE[key]
    found = _CACHE[key] = _read(graph_json, engine)
    while len(_CACHE) > CACHE_LIMIT:
        del _CACHE[next(iter(_CACHE))]
    return found
