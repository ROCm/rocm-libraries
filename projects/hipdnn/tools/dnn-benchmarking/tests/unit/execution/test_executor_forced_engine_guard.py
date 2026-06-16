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
