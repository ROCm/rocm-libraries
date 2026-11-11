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

Exhaustive hooks: ``HIPDNN_<DIR>_CANDIDATE=k`` beside an engine pin builds that engine's ``k``-th
kernel from ``hipdnn_bench enumerate`` (``_candidates``) through ``create_execution_plan_ext(engine_id,
knob_settings)`` and declines ``candidate_not_honored`` when the kernel the build logged is not
the candidate's; the cache key carries the index. ``HIPDNN_SDPA_EXHAUSTIVE`` makes the first
unpinned build of a problem count every ranked engine's candidates into ``last_call(...).candidates``.

There is no native fallback: any ``build_operation_graph`` / engine selection /
``check_support`` / ``build_plans`` failure raises :class:`EngineDeclined` with
hipDNN's message verbatim after a ``<stage>: `` prefix (``adapter:`` for pre-graph declines). An ``execute`` failure raises ``RuntimeError``. Any exception other than
:class:`EngineDeclined` leaves ``run`` with ``CALL_ERROR_ATTR`` set to the failing call's record
(``None`` before the graph exists); a pinned candidate is part of the record (``CallRecord.candidate``).

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

from . import _candidates, _graph, _kernel_log, _repro
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
        "candidate",
        "candidates",
    )

    def __init__(
        self,
        graph,
        uids,
        engine,
        workspace_size,
        ranked,
        kernel="",
        graph_json="",
        candidate=None,
    ):
        self.graph = graph
        self.uids = uids
        self.engine = engine
        self.workspace_size = workspace_size
        # applicable engine names, best first; () for a pinned plan
        self.ranked = ranked
        self.kernel = kernel  # ingestor kernel id the build logged; "" when none
        self.graph_json = graph_json
        # the pinned candidate as {"index", "id", "knobs"}; None when none is pinned
        self.candidate = candidate
        self.candidates = (
            {}
        )  # exhaustive pass 1: ranked engine -> candidate count | message


@dataclasses.dataclass(frozen=True)
class _Call(contract.CallRecord):
    """A :class:`~conformance_contract.CallRecord` that also holds the call's tensors.

    ``tensors`` (role -> tensor, every slot of the problem including derived ones such as
    sequence lengths and ragged offsets) lets ``save_repro`` write a replayable directory from the
    record alone, whichever thread asks.
    """

    tensors: dict = dataclasses.field(default=None, compare=False, repr=False)
    candidates: dict = dataclasses.field(
        default_factory=dict, compare=False, repr=False
    )


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


def _ranked_names(direction, graph, modes, handle) -> tuple:
    """Names of the applicable engines in heuristic rank order.

    A ranking failure declines with hipDNN's message: an empty ranking would read as "no engine
    applies" and hide the cause.
    """
    try:
        names = {
            c.engine_id: c.engine_name for c in graph.get_engine_configs(modes, handle)
        }
        return tuple(names.get(i, str(i)) for i in graph.get_ranked_engine_ids(modes))
    except RuntimeError as exc:
        raise _decline(direction, f"rank_engines: {exc}") from None


def _graph_json(direction, graph) -> str:
    """The graph's JSON; a graph that cannot serialize declines, so no record holds ``""``."""
    try:
        return graph.to_json()
    except RuntimeError as exc:
        raise _decline(direction, f"to_json: {exc}") from None


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
        "pin_fwd": os.environ.get(contract.ENGINE_ENV["fwd"], ""),
        "pin_bwd": os.environ.get(contract.ENGINE_ENV["bwd"], ""),
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


def _pinned_candidate(direction: str) -> int | None:
    """The candidate index ``HIPDNN_<DIR>_CANDIDATE`` pins for ``direction``, else ``None``."""
    raw = os.environ.get(contract.CANDIDATE_ENV[direction])
    return int(raw) if raw else None


def _candidate_pin(direction: str) -> dict | None:
    """``{"index": k}`` for a pinned candidate, else ``None``."""
    index = _pinned_candidate(direction)
    return None if index is None else {"index": index}


def _same_kernel(requested: str, built: str) -> bool:
    """Whether the kernel the backend logged is the candidate's; a slotted engine's
    ``prep=<id>|dkdv=<id>|dq=<id>`` compares by slot, not by order."""
    return sorted(requested.split("|")) == sorted(built.split("|"))


def _resolve_candidate(direction, engine, index, graph_json) -> dict:
    """``{"index", "id", "knobs"}`` of the engine's ``index``-th candidate."""
    found = (
        _candidates.list(graph_json, engine)
        if graph_json
        else _candidates.NotEnumerable("the graph has no JSON form")
    )
    if isinstance(found, _candidates.NotEnumerable):
        raise _decline(direction, f"not_enumerable: {found.message}")
    if index >= len(found):
        raise _decline(
            direction,
            f"candidate_out_of_range: candidate {index} requested, {engine} lists {len(found)}",
        )
    chosen = found[index]
    return {"index": index, "id": chosen.id, "knobs": dict(chosen.knobs)}


def _count_candidates(ranked, graph_json) -> dict:
    """``{engine: candidate count | why it cannot enumerate}`` for each ranked engine."""
    counts = {}
    for name in ranked:
        found = _candidates.list(graph_json, name)
        counts[name] = (
            found.message
            if isinstance(found, _candidates.NotEnumerable)
            else len(found)
        )
    return counts


def _select(
    graph,
    direction: str,
    engine: str | None,
    handle,
    graph_json="",
    candidate: dict | None = None,
    picked: list | None = None,
) -> tuple:
    """Pick, check and build a plan on a graph hipDNN has already finalized.

    ``candidate`` builds one kernel of the engine: ``{"index": k}`` resolves the ``k``-th entry of
    ``hipdnn_bench enumerate`` (``HIPDNN_SDPA_PREFIX``), a dict that also holds ``"knobs"`` (a
    saved candidate) is built as given. The resolved ``{"index", "id", "knobs"}`` is appended to
    ``picked``. Returns ``(plan name, workspace size, ranked engine names)``.
    """
    hipdnn = _graph.frontend()
    modes = [hipdnn.HeuristicMode.FALLBACK]
    ranked = ()
    if engine is None:
        if candidate is not None:
            raise _decline(
                direction,
                f"{contract.CANDIDATE_NOT_HONORED}: candidate {candidate['index']} is pinned without an engine",
            )
        ranked = _ranked_names(direction, graph, modes, handle)
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
        if candidate is None:
            err = graph.create_execution_plan_ext(by_name[engine])
        else:
            chosen = (
                candidate
                if "knobs" in candidate
                else _resolve_candidate(
                    direction, engine, candidate["index"], graph_json
                )
            )
            if picked is not None:
                picked.append(chosen)
            err = graph.create_execution_plan_ext(
                by_name[engine],
                [hipdnn.KnobSetting(n, v) for n, v in chosen["knobs"].items()],
            )
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


def _call(prob: Problem, engine: str, graph_json: str, plan=None, candidate=None):
    """The record of ``prob``'s call on ``plan`` (``None`` while it is still being built)."""
    return _Call(
        direction=prob.direction,
        engine=plan.engine if plan is not None else engine,
        kernel=plan.kernel if plan is not None else "",
        ranked=plan.ranked if plan is not None else (),
        graph_json=graph_json,
        problem_meta=_repro.problem_meta(prob),
        candidate=plan.candidate if plan is not None else candidate,
        tensors=prob.tensors,
        candidates=plan.candidates if plan is not None else {},
    )


def _tag(exc: BaseException, record) -> None:
    """Attach the failing call (or ``None``) to ``exc`` for the plugin (``CALL_ERROR_ATTR``)."""
    setattr(exc, contract.CALL_ERROR_ATTR, record)


def _build_plan(prob: Problem, engine: str | None, handle) -> _Plan:
    direction = prob.direction
    graph, uids = _graph.build(prob)
    graph_json = ""
    picked = []
    try:
        graph_json = _graph_json(direction, graph)
        _repro.capture(prob, graph_json, engine, _candidate_pin(direction))

        err = graph.build_operation_graph(handle)
        if err.is_bad():
            raise _decline(direction, f"build_operation_graph: {err.get_message()}")

        name, workspace_size, ranked = _select(
            graph,
            direction,
            engine,
            handle,
            graph_json,
            _candidate_pin(direction),
            picked,
        )
    except EngineDeclined:
        raise
    except Exception as exc:
        _tag(
            exc,
            _call(
                prob,
                engine or "",
                graph_json,
                candidate=picked[0] if picked else _candidate_pin(direction),
            ),
        )
        raise
    return _Plan(
        graph,
        uids,
        name,
        workspace_size,
        ranked,
        graph_json=graph_json,
        candidate=picked[0] if picked else None,
    )


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
    """Execute ``prob`` in place on the current stream; returns the engine name used.

    An exception other than :class:`EngineDeclined` leaves with ``CALL_ERROR_ATTR`` set to the
    failing call's record (``None`` when it failed before the graph was built).
    """
    try:
        return _run(prob, engine)
    except EngineDeclined:
        raise
    except Exception as exc:
        if not hasattr(exc, contract.CALL_ERROR_ATTR):
            _tag(exc, None)
        raise


def _run(prob: Problem, engine: str | None) -> str:
    device = next(iter(prob.tensors.values())).device
    direction = prob.direction
    with _LOCK:
        calls = _LAST.__dict__.setdefault("calls", {})
        calls.pop(direction, None)
        _enter_case(contract.current_nodeid())
        handle = _handle(device)
        key = (prob.key, engine, device.index, _pinned_candidate(direction))
        plan = _PLANS.get(key)
        _write_marker(
            contract.STAGE_BUILD,
            direction,
            engine or "",
            "",
            plan.graph_json if isinstance(plan, _Plan) else "",
        )
        fresh = plan is None
        if fresh:
            try:
                with _kernel_log.capture() as built:
                    plan = _build_plan(prob, engine, handle)
                plan.kernel = built.kernel(plan.engine)
                if plan.candidate is not None:
                    if not built.selected():
                        raise _decline(
                            direction,
                            f"{contract.CANDIDATE_NOT_HONORED}: the build logged no kernel selection; "
                            f"set {contract.KERNEL_LOG_ENV}=1 so the backend's kernel log is read",
                        )
                    if not _same_kernel(plan.candidate["id"], plan.kernel):
                        raise _decline(
                            direction,
                            f"{contract.CANDIDATE_NOT_HONORED}: requested kernel {plan.candidate['id']}, "
                            f"built {plan.kernel or 'none'}",
                        )
            except EngineDeclined as declined:
                _cache_put(key, (declined.direction, declined.reason))
                raise
        else:
            _PLANS[key] = _PLANS.pop(key)  # most recently used
            if isinstance(plan, tuple):
                raise EngineDeclined(*plan)

        try:
            if fresh:
                if engine is None and os.environ.get(contract.EXHAUSTIVE_ENV):
                    plan.candidates = _count_candidates(plan.ranked, plan.graph_json)
                _cache_put(key, plan)
            else:
                if engine is None:
                    _CASE[direction] = plan.ranked
                _repro.capture(prob, plan.graph_json, engine, plan.candidate)

            workspace = (
                torch.empty(plan.workspace_size, dtype=torch.uint8, device=device)
                if plan.workspace_size > 0
                else None
            )
            variant_pack = {
                plan.uids[role]: t.data_ptr() for role, t in prob.tensors.items()
            }
            _write_marker(
                contract.STAGE_EXECUTE,
                direction,
                plan.engine,
                plan.kernel,
                plan.graph_json,
            )
            _maybe_crash(direction, plan.engine)
            err = plan.graph.execute(
                handle,
                variant_pack,
                workspace.data_ptr() if workspace is not None else 0,
            )
            if err.is_bad():
                raise RuntimeError(
                    f"hipDNN {direction} execute failed: {err.get_message()}"
                )
        except EngineDeclined:
            raise
        except Exception as exc:
            _tag(exc, _call(prob, engine or "", plan.graph_json, plan))
            raise
        calls[direction] = _call(prob, engine or "", plan.graph_json, plan)
        return plan.engine
