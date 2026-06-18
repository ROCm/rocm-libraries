# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for forced-engine selection in the hipDNN Executor.

A forced/preferred engine is a SOFT request in hipDNN: when the requested id is
not among the engines the backend ranks as applicable, the frontend silently
runs the top-ranked engine while the caller still believes the forced engine
ran -- fabricating comparison rows where several "different" forced engines are
all the same fallback.

The executor avoids this by hard-selecting a forced engine via
``Graph.create_execution_plan_ext`` (which errors instead of falling back) and
reading back the engine that actually backs the built plan via
``Graph.get_execution_plan_engine_id``.
"""

import sys
import types
from unittest.mock import patch

import pytest

import dnn_benchmarking.execution.executor as executor_module
from dnn_benchmarking.config.benchmark_config import BenchmarkConfig
from dnn_benchmarking.common.exceptions import ExecutionError, UnsupportedGraphError


class _StubResult:
    """hipDNN Error stub with a configurable bad/message state."""

    def __init__(self, bad: bool = False, message: str = ""):
        self._bad = bad
        self._message = message

    def is_bad(self) -> bool:
        return self._bad

    def get_message(self) -> str:
        return self._message


class _StubGraph:
    """Minimal hipDNN Graph stub exercising the executor's plan lifecycle.

    ``create_execution_plan_ext`` records the hard-selected engine, returning a
    bad Error when ``hard_fails``; ``create_execution_plans`` flags the
    heuristic path; ``get_execution_plan_engine_id`` reports the engine backing
    the built plan.
    """

    def __init__(
        self,
        ranked,
        selected=None,
        hard_fails=False,
        rank_error=None,
        plans_fail=False,
        support_fails=False,
        build_fails=False,
    ):
        self._ranked = ranked
        self._selected = selected
        self._hard_fails = hard_fails
        self._rank_error = rank_error
        self._plans_fail = plans_fail
        self._support_fails = support_fails
        self._build_fails = build_fails
        self.plans_created = False
        self.hard_engine_id = None

    def from_json(self, _s):
        return _StubResult()

    def validate(self):
        return _StubResult()

    def build_operation_graph(self, _handle):
        return _StubResult()

    def get_ranked_engine_ids(self):
        if self._rank_error is not None:
            raise RuntimeError(self._rank_error)
        return list(self._ranked)

    def create_execution_plans(self):
        self.plans_created = True
        return _StubResult(bad=self._plans_fail, message="plan creation failed")

    def create_execution_plan_ext(self, engine_id):
        if self._hard_fails:
            return _StubResult(bad=True, message="Failed to finalize engine descriptor")
        self.hard_engine_id = engine_id
        return _StubResult()

    def get_execution_plan_engine_id(self):
        return self._selected

    def check_support(self):
        return _StubResult(bad=self._support_fails, message="not supported")

    def build_plans(self):
        return _StubResult(bad=self._build_fails, message="build failed")

    def get_workspace_size(self):
        return 0


def _executor():
    config = BenchmarkConfig(graph_path="dummy.json", warmup_iters=0, benchmark_iters=1)
    # "{}" -> empty graph dict: no data-type attrs / nodes to configure.
    return executor_module.Executor("{}", config)


def _fake_module(graph):
    fake = types.ModuleType("hipdnn_frontend")
    fake.Graph = lambda: graph
    return fake


def test_prepare_hard_select_records_actual_engine():
    """A forced, applicable engine is hard-selected (not soft-preferred) and the
    engine the backend reports as backing the plan is recorded."""
    executor = _executor()
    graph = _StubGraph(ranked=[999], selected=999)
    with patch.dict(sys.modules, {"hipdnn_frontend": _fake_module(graph)}):
        executor.prepare(handle=object(), engine_id=999)
    assert graph.hard_engine_id == 999  # hard selection was used
    assert graph.plans_created is False  # heuristic path not taken
    assert executor.selected_engine_id == 999


def test_prepare_hard_select_not_applicable_is_skip():
    """A hard-select failure (engine not applicable) becomes an
    UnsupportedGraphError, i.e. a clean skip rather than a silent fallback."""
    executor = _executor()
    graph = _StubGraph(ranked=[111], hard_fails=True)
    with patch.dict(sys.modules, {"hipdnn_frontend": _fake_module(graph)}):
        with pytest.raises(UnsupportedGraphError):
            executor.prepare(handle=object(), engine_id=999)


def test_prepare_discovery_uses_heuristic_plan_creation():
    """With no forced engine, prepare uses the heuristic create_execution_plans
    path and records whichever engine the backend selected."""
    executor = _executor()
    graph = _StubGraph(ranked=[111, 222], selected=111)
    with patch.dict(sys.modules, {"hipdnn_frontend": _fake_module(graph)}):
        executor.prepare(handle=object(), engine_id=None)
    assert graph.plans_created is True  # heuristic path taken
    assert graph.hard_engine_id is None  # hard selection not used
    assert executor.selected_engine_id == 111


def test_record_selected_engine_mismatch_raises():
    """If a forced engine differs from the engine actually selected, it is
    treated as an unsupported-graph skip rather than mislabeled timings."""
    executor = _executor()
    executor._graph = _StubGraph(ranked=[111], selected=111)
    with pytest.raises(UnsupportedGraphError) as exc:
        executor._record_selected_engine(999)
    assert "999" in str(exc.value) and "111" in str(exc.value)


def test_discover_engines_ranking_runtime_error_becomes_unsupported():
    """A backend RuntimeError while ranking surfaces as an unsupported-graph
    skip, not a hard error."""
    executor = _executor()
    graph = _StubGraph(ranked=[], rank_error="no engine has an applicable solution")
    with patch.dict(sys.modules, {"hipdnn_frontend": _fake_module(graph)}):
        with pytest.raises(UnsupportedGraphError) as exc:
            executor.discover_engines(handle=object())
    assert "applicable solution" in str(exc.value)


def test_prepare_forced_engine_mismatch_is_skip():
    """Driven through the public prepare() flow: hard-select succeeds but the
    backend reports a different engine backing the plan -> unsupported skip."""
    executor = _executor()
    graph = _StubGraph(ranked=[999], selected=111)  # hard select ok, read-back differs
    with patch.dict(sys.modules, {"hipdnn_frontend": _fake_module(graph)}):
        with pytest.raises(UnsupportedGraphError) as exc:
            executor.prepare(handle=object(), engine_id=999)
    assert graph.hard_engine_id == 999  # hard select was attempted
    assert "999" in str(exc.value) and "111" in str(exc.value)


def test_prepare_create_execution_plans_failure_is_execution_error():
    """A bad create_execution_plans() result on the discovery path is a hard
    ExecutionError, not an unsupported-graph skip."""
    executor = _executor()
    graph = _StubGraph(ranked=[1], plans_fail=True)
    with patch.dict(sys.modules, {"hipdnn_frontend": _fake_module(graph)}):
        with pytest.raises(ExecutionError) as exc:
            executor.prepare(handle=object(), engine_id=None)
    assert "plan creation failed" in str(exc.value)


def test_prepare_check_support_failure_is_unsupported():
    """A bad check_support() result is classified as an unsupported-graph skip."""
    executor = _executor()
    graph = _StubGraph(ranked=[1], selected=1, support_fails=True)
    with patch.dict(sys.modules, {"hipdnn_frontend": _fake_module(graph)}):
        with pytest.raises(UnsupportedGraphError) as exc:
            executor.prepare(handle=object(), engine_id=None)
    assert "not supported" in str(exc.value)


def test_prepare_build_plans_failure_is_execution_error():
    """A bad build_plans() result is a hard ExecutionError."""
    executor = _executor()
    graph = _StubGraph(ranked=[1], selected=1, build_fails=True)
    with patch.dict(sys.modules, {"hipdnn_frontend": _fake_module(graph)}):
        with pytest.raises(ExecutionError) as exc:
            executor.prepare(handle=object(), engine_id=None)
    assert "build failed" in str(exc.value)
