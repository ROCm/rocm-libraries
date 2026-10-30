# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Handle, plan cache and execution for a :class:`~._problem.Problem`.

One hipDNN handle per (device, HIP stream); the stream is
``torch.cuda.current_stream()`` at call time, so work lands on the caller's stream
(autograd's backward thread included). Plans are cached by ``Problem.key`` plus
engine and device; a decline is cached too (as plain data), so a shape the engine rejects is probed
once. The whole call runs under one lock: graph state and plans are not shared
across threads. A plan built without an engine pin also keeps the applicable engines' names in
heuristic rank order, the kernel id the build logged (``_kernel_log``) and the graph JSON; the
cache holds ``HIPDNN_SDPA_PLAN_CACHE`` entries (default 64), least recently used out first.
:func:`last_call` describes the calling thread's latest successful ``run`` of a direction.

Conformance hooks (names and formats in ``aotriton/conformance_contract.py``): every ``run``
rewrites an in-flight marker file (entry, once the ranked list is known, before ``execute``) so a
worker that dies leaves behind what it was doing; ``HIPDNN_SDPA_REPRO_CAPTURE`` makes ``run`` save
the call's inputs and graph right after the graph is built, before any hipDNN call that could
crash; ``HIPDNN_SDPA_TEST_CRASH`` exits the process in place of ``execute``.

There is no native fallback: any ``build_operation_graph`` / engine selection /
``check_support`` / ``build_plans`` failure raises :class:`EngineDeclined` with
hipDNN's message verbatim after a ``<stage>: `` prefix (``adapter:`` for pre-graph declines). An ``execute`` failure raises ``RuntimeError``.

Bindings used: ``Handle(stream)`` ``handle_bindings.cpp:193``; ``Graph``
``build_operation_graph`` ``graph_bindings.cpp:173``, ``create_execution_plans`` ``:182``,
``create_execution_plan_ext`` ``:186``, ``get_engine_configs`` ``:421`` (entries carry
``engine_id`` / ``engine_name``, ``autotune_bindings.cpp:121``), ``check_support`` ``:248``,
``build_plans`` ``:249``, ``get_workspace_size`` ``:255``, ``execute`` ``:268``,
``get_plan_name`` ``:340``, ``get_ranked_engine_ids`` ``:211``, ``get_execution_plan_engine_id`` ``:196``;
``Error.is_bad`` / ``get_message`` ``types_bindings.cpp:197-200``;
``HeuristicMode`` ``types_bindings.cpp:144``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import threading

import conformance_contract as contract
import torch

from . import _graph, _kernel_log, _repro
from ._problem import Problem
from ._types import EngineDeclined

_LOCK = threading.RLock()
_HANDLES: dict = {}  # (device index, stream pointer) -> Handle
# (Problem.key, engine, device index) -> _Plan, or a (direction, reason) decline, least recently
# used first. A decline is kept as plain data: a cached exception would keep its traceback, and
# with it the declined call's tensors, alive for the rest of the process.
_PLANS: dict = {}
DEFAULT_PLAN_CACHE = 64
# Ranked engine names of the case (pytest node id) being run, per direction: what the marker of a
# backward call says about the forward's applicable engines.
_CASE: dict = {"nodeid": None, "fwd": (), "bwd": ()}
_LAST = (
    threading.local()
)  # .calls: direction -> _Call of this thread's latest successful run


class _Plan:
    __slots__ = (
        "graph",
        "uids",
        "engine",
        "workspace_size",
        "ranked",
        "kernel",
        "graph_json",
    )

    def __init__(
        self, graph, uids, engine, workspace_size, ranked, kernel="", graph_json=""
    ):
        self.graph = graph
        self.uids = uids
        self.engine = engine
        self.workspace_size = workspace_size
        # applicable engine names, best first; () for a pinned plan
        self.ranked = ranked
        self.kernel = kernel  # ingestor kernel id the build logged; "" when none
        self.graph_json = graph_json


@dataclasses.dataclass(frozen=True)
class _Call(contract.CallRecord):
    """A :class:`~conformance_contract.CallRecord` that also holds the call's tensors.

    ``tensors`` (role -> tensor, every slot of the problem including derived ones such as
    sequence lengths and ragged offsets) lets ``save_repro`` write a replayable directory from the
    record alone, whichever thread asks.
    """

    tensors: dict = dataclasses.field(default=None, compare=False, repr=False)


def _handle(device: torch.device):
    hipdnn = _graph.frontend()
    index = device.index if device.index is not None else torch.cuda.current_device()
    with torch.cuda.device(index):
        stream = torch.cuda.current_stream(index).cuda_stream
        key = (index, stream)
        handle = _HANDLES.get(key)
        if handle is None:
            handle = _HANDLES[key] = hipdnn.Handle(stream)
    return handle


def _decline(direction: str, reason: str) -> EngineDeclined:
    return EngineDeclined(direction, reason)


def last_call(direction: str):
    """The calling thread's latest successful ``run`` of ``direction``, else ``None``.

    A run that declined or raised clears its direction's record, so a stale one is never read.
    """
    return getattr(_LAST, "calls", {}).get(direction)


def _ranked_names(graph, modes, handle) -> tuple:
    """Names of the applicable engines in heuristic rank order; empty when hipDNN cannot say."""
    try:
        names = {
            c.engine_id: c.engine_name for c in graph.get_engine_configs(modes, handle)
        }
        return tuple(names.get(i, str(i)) for i in graph.get_ranked_engine_ids(modes))
    except RuntimeError:
        return ()


def _nodeid() -> str:
    return os.environ.get("PYTEST_CURRENT_TEST", "").rsplit(" (", 1)[0]


def _graph_json(graph) -> str:
    try:
        return graph.to_json()
    except RuntimeError:
        return ""


def _enter_case(nodeid: str) -> None:
    if _CASE["nodeid"] != nodeid:
        _CASE.update(nodeid=nodeid, fwd=(), bwd=())


def _write_marker(stage, direction, engine, kernel, graph_json) -> None:
    """Rewrite this worker's in-flight marker atomically; a no-op without the directory env."""
    root = os.environ.get(contract.INFLIGHT_DIR_ENV)
    if not root:
        return
    directory = os.path.join(root, os.environ.get(contract.PASS_SLUG_ENV, ""))
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(
        directory, contract.marker_name(os.environ.get("PYTEST_XDIST_WORKER"))
    )
    marker = {
        "nodeid": _CASE["nodeid"],
        "direction": direction,
        "stage": stage,
        "engine": engine,
        "pin_fwd": os.environ.get("HIPDNN_FWD_ENGINE", ""),
        "pin_bwd": os.environ.get("HIPDNN_BWD_ENGINE", ""),
        "fwd_engines": list(_CASE["fwd"]),
        "bwd_engines": list(_CASE["bwd"]),
        "kernel": kernel,
        "pid": os.getpid(),
        "graph_json": graph_json,
    }
    tmp = f"{path}.tmp"
    with open(tmp, "w") as out:
        json.dump(marker, out)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)


def _note_ranked(direction, engine, ranked, graph_json) -> None:
    """The ranked list is known: remember it for the case and say so in the marker."""
    if engine is None:
        _CASE[direction] = ranked
    _write_marker(contract.STAGE_BUILD, direction, engine or "", "", graph_json)


def _select(graph, direction: str, engine: str | None, handle, graph_json="") -> tuple:
    """Pick, check and build a plan on a graph hipDNN has already finalized.

    Returns ``(plan name, workspace size, ranked engine names)``.
    """
    hipdnn = _graph.frontend()
    modes = [hipdnn.HeuristicMode.FALLBACK]
    ranked = ()
    if engine is None:
        ranked = _ranked_names(graph, modes, handle)
        _note_ranked(direction, engine, ranked, graph_json)
        err = graph.create_execution_plans(modes)
        if err.is_bad():
            raise _decline(direction, f"create_execution_plans: {err.get_message()}")
    else:
        _note_ranked(direction, engine, ranked, graph_json)
        try:
            configs = graph.get_engine_configs(modes, handle)
        except RuntimeError as exc:
            raise _decline(direction, f"get_engine_configs: {exc}") from None
        by_name = {c.engine_name: c.engine_id for c in configs}
        if engine not in by_name:
            raise _decline(
                direction,
                f"engine_not_applicable: engine {engine} is not applicable to this graph; applicable: "
                f"{', '.join(sorted(by_name)) or 'none'}",
            )
        err = graph.create_execution_plan_ext(by_name[engine])
        if err.is_bad():
            raise _decline(direction, f"create_execution_plan_ext: {err.get_message()}")

    err = graph.check_support()
    if err.is_bad():
        raise _decline(direction, f"check_support: {err.get_message()}")
    err = graph.build_plans()
    if err.is_bad():
        raise _decline(direction, f"build_plans: {err.get_message()}")

    if engine is not None:
        try:
            built = graph.get_execution_plan_engine_id()
        except RuntimeError as exc:
            raise _decline(
                direction, f"pin_not_honored: requested {engine}: {exc}"
            ) from None
        if built != by_name[engine]:
            raise _decline(
                direction,
                f"pin_not_honored: requested {engine}, built "
                f"{next((n for n, i in by_name.items() if i == built), built)}",
            )

    return graph.get_plan_name(handle), graph.get_workspace_size(), ranked


def _build_plan(prob: Problem, engine: str | None, handle) -> _Plan:
    direction = prob.direction
    graph, uids = _graph.build(prob)
    graph_json = _graph_json(graph)
    _repro.capture(prob, graph_json, engine)

    err = graph.build_operation_graph(handle)
    if err.is_bad():
        raise _decline(direction, f"build_operation_graph: {err.get_message()}")

    name, workspace_size, ranked = _select(graph, direction, engine, handle, graph_json)
    return _Plan(graph, uids, name, workspace_size, ranked, graph_json=graph_json)


def _cache_limit() -> int:
    raw = os.environ.get(contract.PLAN_CACHE_ENV)
    return max(1, int(raw)) if raw else DEFAULT_PLAN_CACHE


def _cache_put(key, value) -> None:
    _PLANS[key] = value
    while len(_PLANS) > _cache_limit():
        del _PLANS[next(iter(_PLANS))]


def _maybe_crash(direction: str, engine: str) -> None:
    if contract.parse_test_crash(os.environ.get(contract.TEST_CRASH_ENV)) == (
        direction,
        engine,
    ):
        os._exit(contract.TEST_CRASH_EXIT_CODE)


def run(prob: Problem, engine: str | None) -> str:
    """Execute ``prob`` in place on the current stream; returns the engine name used."""
    device = next(iter(prob.tensors.values())).device
    direction = prob.direction
    with _LOCK:
        calls = _LAST.__dict__.setdefault("calls", {})
        calls.pop(direction, None)
        _enter_case(_nodeid())
        handle = _handle(device)
        key = (prob.key, engine, device.index)
        plan = _PLANS.get(key)
        _write_marker(
            contract.STAGE_BUILD,
            direction,
            engine or "",
            "",
            plan.graph_json if isinstance(plan, _Plan) else "",
        )
        if plan is None:
            try:
                with _kernel_log.capture() as built:
                    plan = _build_plan(prob, engine, handle)
            except EngineDeclined as declined:
                _cache_put(key, (declined.direction, declined.reason))
                raise
            plan.kernel = built.kernel(plan.engine)
            _cache_put(key, plan)
        else:
            _PLANS[key] = _PLANS.pop(key)  # most recently used
            if isinstance(plan, tuple):
                raise EngineDeclined(*plan)
            if engine is None:
                _CASE[direction] = plan.ranked
            _repro.capture(prob, plan.graph_json, engine)

        workspace = (
            torch.empty(plan.workspace_size, dtype=torch.uint8, device=device)
            if plan.workspace_size > 0
            else None
        )
        variant_pack = {
            plan.uids[role]: t.data_ptr() for role, t in prob.tensors.items()
        }
        _write_marker(
            contract.STAGE_EXECUTE, direction, plan.engine, plan.kernel, plan.graph_json
        )
        _maybe_crash(direction, plan.engine)
        err = plan.graph.execute(
            handle, variant_pack, workspace.data_ptr() if workspace is not None else 0
        )
        if err.is_bad():
            raise RuntimeError(
                f"hipDNN {direction} execute failed: {err.get_message()}"
            )
        calls[direction] = _Call(
            direction,
            plan.engine,
            plan.kernel,
            plan.ranked,
            plan.graph_json,
            _repro.problem_meta(prob),
            prob.tensors,
        )
        return plan.engine
