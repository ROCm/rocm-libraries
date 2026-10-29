# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Handle, plan cache and execution for a :class:`~._problem.Problem`.

One hipDNN handle per (device, HIP stream); the stream is
``torch.cuda.current_stream()`` at call time, so work lands on the caller's stream
(autograd's backward thread included). Plans are cached by ``Problem.key`` plus
engine and device; a decline is cached too (as plain data), so a shape the engine rejects is probed
once. The whole call runs under one lock: graph state and plans are not shared
across threads.

There is no native fallback: any ``build_operation_graph`` / engine selection /
``check_support`` / ``build_plans`` failure raises :class:`EngineDeclined` with
hipDNN's message verbatim after a ``<stage>: `` prefix (``adapter:`` for pre-graph declines). An ``execute`` failure raises ``RuntimeError``.

Bindings used: ``Handle(stream)`` ``handle_bindings.cpp:193``; ``Graph``
``build_operation_graph`` ``graph_bindings.cpp:173``, ``create_execution_plans`` ``:182``,
``create_execution_plan_ext`` ``:186``, ``get_engine_configs`` ``:421`` (entries carry
``engine_id`` / ``engine_name``, ``autotune_bindings.cpp:121``), ``check_support`` ``:248``,
``build_plans`` ``:249``, ``get_workspace_size`` ``:255``, ``execute`` ``:268``,
``get_plan_name`` ``:340``; ``Error.is_bad`` / ``get_message`` ``types_bindings.cpp:197-200``;
``HeuristicMode`` ``types_bindings.cpp:144``.
"""

from __future__ import annotations

import threading

import torch

from . import _graph
from ._problem import Problem
from ._types import EngineDeclined

_LOCK = threading.RLock()
_HANDLES: dict = {}  # (device index, stream pointer) -> Handle
# (Problem.key, engine, device index) -> _Plan, or a (direction, reason) decline. A decline is
# kept as plain data: a cached exception would keep its traceback, and with it the declined
# call's tensors, alive for the rest of the process.
_PLANS: dict = {}


class _Plan:
    __slots__ = ("graph", "uids", "engine", "workspace_size")

    def __init__(self, graph, uids, engine, workspace_size):
        self.graph = graph
        self.uids = uids
        self.engine = engine
        self.workspace_size = workspace_size


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


def _build_plan(prob: Problem, engine: str | None, handle) -> _Plan:
    hipdnn = _graph.frontend()
    direction = prob.direction
    graph, uids = _graph.build(prob)

    err = graph.build_operation_graph(handle)
    if err.is_bad():
        raise _decline(direction, f"build_operation_graph: {err.get_message()}")

    modes = [hipdnn.HeuristicMode.FALLBACK]
    if engine is None:
        err = graph.create_execution_plans(modes)
        if err.is_bad():
            raise _decline(direction, f"create_execution_plans: {err.get_message()}")
    else:
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

    return _Plan(graph, uids, graph.get_plan_name(handle), graph.get_workspace_size())


def run(prob: Problem, engine: str | None) -> str:
    """Execute ``prob`` in place on the current stream; returns the engine name used."""
    device = next(iter(prob.tensors.values())).device
    with _LOCK:
        handle = _handle(device)
        key = (prob.key, engine, device.index)
        plan = _PLANS.get(key)
        if plan is None:
            try:
                plan = _build_plan(prob, engine, handle)
            except EngineDeclined as declined:
                _PLANS[key] = (declined.direction, declined.reason)
                raise
            _PLANS[key] = plan
        elif isinstance(plan, tuple):
            raise EngineDeclined(*plan)

        workspace = (
            torch.empty(plan.workspace_size, dtype=torch.uint8, device=device)
            if plan.workspace_size > 0
            else None
        )
        variant_pack = {
            plan.uids[role]: t.data_ptr() for role, t in prob.tensors.items()
        }
        err = plan.graph.execute(
            handle, variant_pack, workspace.data_ptr() if workspace is not None else 0
        )
        if err.is_bad():
            raise RuntimeError(
                f"hipDNN {prob.direction} execute failed: {err.get_message()}"
            )
        return plan.engine
