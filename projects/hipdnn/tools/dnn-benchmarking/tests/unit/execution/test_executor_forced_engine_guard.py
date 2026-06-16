# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the forced-engine applicability guard in Executor.

A preferred (forced) engine is a *soft* request in hipDNN: if the forced id is
not among the engines the backend ranks as applicable for a graph, hipDNN
silently falls back to the top-ranked engine and runs that, while the caller
still believes the forced engine ran. The benchmarking tool labels rows by the
*requested* engine, so that silent fallback fabricates comparison rows where
several "different" forced engines are all the same fallback engine.

``Executor._build_through_operation_graph`` guards against this by verifying the
forced id is in ``get_ranked_engine_ids()`` (exactly the predicate hipDNN uses
to honor a preferred engine) and raising ``UnsupportedGraphError`` otherwise,
which the suite runner turns into an honest skipped row.
"""

import sys
import types
from unittest.mock import patch

import pytest

import dnn_benchmarking.execution.executor as executor_module
from dnn_benchmarking.config.benchmark_config import BenchmarkConfig
from dnn_benchmarking.common.exceptions import UnsupportedGraphError


class _StubResult:
    """hipDNN Error stub that always reports success."""

    def is_bad(self) -> bool:
        return False

    def get_message(self) -> str:
        return ""


class _StubGraph:
    """Minimal hipDNN Graph stub with a configurable ranked-engine list."""

    def __init__(self, ranked):
        self._ranked = ranked
        self.preferred_engine_id = None

    def from_json(self, _s):
        return _StubResult()

    def set_preferred_engine_id_ext(self, engine_id):
        self.preferred_engine_id = engine_id

    def validate(self):
        return _StubResult()

    def build_operation_graph(self, _handle):
        return _StubResult()

    def get_ranked_engine_ids(self):
        return list(self._ranked)


def _executor():
    config = BenchmarkConfig(
        graph_path="dummy.json", warmup_iters=0, benchmark_iters=1
    )
    # "{}" -> empty graph dict: no data-type attrs / nodes to configure.
    return executor_module.Executor("{}", config)


def _fake_hipdnn(ranked):
    """Build a stub `hipdnn_frontend` module whose Graph has `ranked` engines."""
    fake = types.ModuleType("hipdnn_frontend")
    fake.Graph = lambda: _StubGraph(ranked)
    return fake


def test_forced_engine_not_in_ranked_list_raises_unsupported():
    """Forcing an engine the backend would not select is a clean skip, not a
    silent fallback."""
    executor = _executor()
    with patch.dict(sys.modules, {"hipdnn_frontend": _fake_hipdnn([111, 222])}):
        with pytest.raises(UnsupportedGraphError) as exc:
            executor._build_through_operation_graph(handle=object(), engine_id=999)
    msg = str(exc.value)
    assert "999" in msg
    assert "not applicable" in msg
    # The message surfaces what the backend would actually offer.
    assert "111" in msg and "222" in msg


def test_forced_engine_in_ranked_list_is_honored():
    """An applicable forced engine builds normally and is set as preferred."""
    executor = _executor()
    fake = _fake_hipdnn([111, 999, 222])
    with patch.dict(sys.modules, {"hipdnn_frontend": fake}):
        result = executor._build_through_operation_graph(
            handle=object(), engine_id=999
        )
        assert result is fake
    assert executor._graph.preferred_engine_id == 999


def test_discovery_path_no_engine_id_skips_guard():
    """Discovery (engine_id=None) must not trip the guard even though the guard
    queries the same ranked list."""
    executor = _executor()
    fake = _fake_hipdnn([])
    with patch.dict(sys.modules, {"hipdnn_frontend": fake}):
        # Would raise if the guard ran with engine_id=None.
        result = executor._build_through_operation_graph(
            handle=object(), engine_id=None
        )
        assert result is fake
    assert executor._graph.preferred_engine_id is None


def test_forced_engine_ranking_runtime_error_becomes_unsupported():
    """A backend RuntimeError while ranking is surfaced as an unsupported-graph
    skip, not a hard error."""

    class _RaisingGraph(_StubGraph):
        def get_ranked_engine_ids(self):
            raise RuntimeError("no engine has an applicable solution")

    fake = types.ModuleType("hipdnn_frontend")
    fake.Graph = lambda: _RaisingGraph(ranked=[])
    executor = _executor()
    with patch.dict(sys.modules, {"hipdnn_frontend": fake}):
        with pytest.raises(UnsupportedGraphError) as exc:
            executor._build_through_operation_graph(handle=object(), engine_id=7)
    assert "applicable solution" in str(exc.value)


class _HardStubGraph(_StubGraph):
    """Graph stub that exposes the hard-select + read-back bindings.

    Presence of ``create_execution_plan_ext`` is what makes the executor take
    the hard path; ``get_execution_plan_engine_id`` reports the engine that
    actually backs the built plan.
    """

    def __init__(self, ranked, selected=None, hard_raises=False):
        super().__init__(ranked)
        self._selected = selected
        self._hard_raises = hard_raises
        self.hard_engine_id = None

    def create_execution_plan_ext(self, engine_id):
        if self._hard_raises:
            raise RuntimeError("Failed to finalize engine descriptor")
        self.hard_engine_id = engine_id

    def get_execution_plan_engine_id(self):
        if self._selected is None:
            raise RuntimeError("no execution plan engine id")
        return self._selected

    # Plan lifecycle methods used by Executor.prepare.
    def check_support(self):
        return _StubResult()

    def build_plans(self):
        return _StubResult()

    def get_workspace_size(self):
        return 0


def test_hard_select_skips_soft_preferred_and_guard():
    """When the hard-select binding exists, the build neither sets a soft
    preference nor runs the membership guard (the hard plan creation in
    prepare is the authority)."""
    executor = _executor()
    fake = types.ModuleType("hipdnn_frontend")
    # 999 is NOT in the ranked list, yet the guard must not fire on the build.
    fake.Graph = lambda: _HardStubGraph(ranked=[111])
    with patch.dict(sys.modules, {"hipdnn_frontend": fake}):
        executor._build_through_operation_graph(handle=object(), engine_id=999)
    assert executor._used_hard_select is True
    assert executor._graph.preferred_engine_id is None  # soft path not taken


def test_prepare_hard_select_records_actual_engine():
    """prepare() uses create_execution_plan_ext and records the engine the
    backend reports as backing the plan."""
    executor = _executor()
    g = _HardStubGraph(ranked=[999], selected=999)
    fake = types.ModuleType("hipdnn_frontend")
    fake.Graph = lambda: g
    with patch.dict(sys.modules, {"hipdnn_frontend": fake}):
        executor.prepare(handle=object(), engine_id=999)
    assert g.hard_engine_id == 999  # hard selection was used
    assert executor.selected_engine_id == 999


def test_prepare_hard_select_not_applicable_is_skip():
    """A hard-select failure (engine not applicable) becomes an
    UnsupportedGraphError, i.e. a clean skip."""
    executor = _executor()
    fake = types.ModuleType("hipdnn_frontend")
    fake.Graph = lambda: _HardStubGraph(ranked=[111], hard_raises=True)
    with patch.dict(sys.modules, {"hipdnn_frontend": fake}):
        with pytest.raises(UnsupportedGraphError):
            executor.prepare(handle=object(), engine_id=999)


def test_record_selected_engine_mismatch_raises():
    """If a forced engine differs from the engine actually selected (only
    possible on the soft path), it's treated as an unsupported-graph skip."""
    executor = _executor()
    executor._graph = _HardStubGraph(ranked=[111], selected=111)
    with pytest.raises(UnsupportedGraphError) as exc:
        executor._record_selected_engine(999)
    assert "999" in str(exc.value) and "111" in str(exc.value)


def test_record_selected_engine_without_binding_is_noop():
    """Without get_execution_plan_engine_id the read-back is a no-op (older
    bindings); selected_engine_id stays None and nothing raises."""
    executor = _executor()
    executor._graph = _StubGraph(ranked=[111])  # no get_execution_plan_engine_id
    executor._record_selected_engine(999)
    assert executor.selected_engine_id is None
