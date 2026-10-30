# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The plan cache keeps declines without keeping the declined call's tensors alive."""

import gc
import types
import weakref

import pytest
import torch

from hipdnn_sdpa import EngineDeclined, _runtime


class _Prob:
    def __init__(self, tensor, key):
        self.tensors = {"q": tensor}
        self.key = key
        self.direction = "fwd"


@pytest.fixture
def declining_runtime(monkeypatch):
    monkeypatch.setattr(_runtime, "_PLANS", {})
    monkeypatch.setattr(_runtime, "_handle", lambda device: object())

    def build(prob, engine, handle):
        raise EngineDeclined(prob.direction, "check_support: no engine")

    monkeypatch.setattr(_runtime, "_build_plan", build)


def _call_and_drop(key):
    """Run one declined call whose tensor is reachable only from this frame."""
    tensor = torch.empty(1024)
    ref = weakref.ref(tensor)
    with pytest.raises(EngineDeclined):
        _runtime.run(_Prob(tensor, key), None)
    return ref


def test_cached_decline_does_not_keep_tensors_alive(declining_runtime):
    ref = _call_and_drop(("shape", 1))
    gc.collect()
    assert ref() is None


def test_cached_decline_is_raised_again_with_the_same_reason(declining_runtime):
    _call_and_drop(("shape", 2))
    with pytest.raises(EngineDeclined) as again:
        _runtime.run(_Prob(torch.empty(1), ("shape", 2)), None)
    assert (again.value.direction, again.value.reason) == (
        "fwd",
        "check_support: no engine",
    )


class _Ok:
    def is_bad(self):
        return False


class _FakeGraph:
    """Engines ENG_A (id 1, ranked first) and ENG_B (id 2); `serves` is the id that really builds."""

    def __init__(self, serves):
        self.serves = serves
        self.pinned = None

    def build_operation_graph(self, handle):
        return _Ok()

    def get_engine_configs(self, modes, handle):
        return [
            types.SimpleNamespace(engine_id=2, engine_name="ENG_B"),
            types.SimpleNamespace(engine_id=1, engine_name="ENG_A"),
        ]

    def get_ranked_engine_ids(self, modes):
        return [1, 2]

    def create_execution_plans(self, modes):
        return _Ok()

    def create_execution_plan_ext(self, engine_id):
        self.pinned = engine_id
        return _Ok()

    def check_support(self):
        return _Ok()

    def build_plans(self):
        return _Ok()

    def get_execution_plan_engine_id(self):
        return self.serves if self.serves is not None else self.pinned

    def get_plan_name(self, handle):
        return "plan"

    def get_workspace_size(self):
        return 0


@pytest.fixture
def fake_graph(monkeypatch):
    holder = {}
    frontend = types.SimpleNamespace(
        HeuristicMode=types.SimpleNamespace(FALLBACK="fallback")
    )
    monkeypatch.setattr(_runtime._graph, "frontend", lambda: frontend)
    monkeypatch.setattr(_runtime._graph, "build", lambda prob: (holder["graph"], {}))
    return lambda serves: holder.setdefault("graph", _FakeGraph(serves))


def test_unpinned_plan_keeps_the_ranked_engine_names(fake_graph):
    fake_graph(None)
    plan = _runtime._build_plan(_Prob(torch.empty(1), "k"), None, object())
    assert plan.ranked == ("ENG_A", "ENG_B")


def test_pinned_plan_declines_when_another_engine_serves(fake_graph):
    fake_graph(2)  # the graph builds ENG_B whatever is requested
    with pytest.raises(EngineDeclined) as declined:
        _runtime._build_plan(_Prob(torch.empty(1), "k"), "ENG_A", object())
    assert declined.value.reason == "pin_not_honored: requested ENG_A, built ENG_B"


def test_pinned_plan_that_is_honored_builds(fake_graph):
    fake_graph(None)  # serves what was pinned
    plan = _runtime._build_plan(_Prob(torch.empty(1), "k"), "ENG_B", object())
    assert (plan.engine, plan.ranked) == ("plan", ())
