# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The plan cache keeps declines without keeping the declined call's tensors alive."""

import gc
import json
import os
import threading
import types
import weakref

import pytest
import torch

import hipdnn_sdpa
from hipdnn_sdpa import EngineDeclined, _kernel_log, _problem, _runtime

contract = _runtime.contract


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

    def to_json(self):
        return '{"graph": "fake"}'


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


# ----------------------------------------------------------------- run(): marker, hooks, LRU

KERNEL = "0b1c2d3e-0000-4000-8000-00000000000a"
ENGINE = "hipkernel:flydsl_sdpa_cdna"
SELECTED_LINE = f"ingestor: engine 'flydsl_sdpa_cdna' selected kernel {KERNEL} at rank 0 from 1 candidate(s)"


def nodeid():
    return os.environ["PYTEST_CURRENT_TEST"].rsplit(" (", 1)[0]


def fwd_problem(scale=0.5, o=None):
    q, k, v = (torch.randn(1, 2, 4, 8) for _ in range(3))
    o = torch.zeros(1, 2, 4, 8) if o is None else o
    return _problem.forward_problem(
        q, k, v, o, torch.zeros(1, 2, 4, 1), scale, None, None, None, None
    )


def bwd_problem(scale=0.5):
    q, k, v, o, do, dq, dk, dv = (torch.randn(1, 2, 4, 8) for _ in range(8))
    return _problem.backward_problem(
        q,
        k,
        v,
        o,
        do,
        torch.zeros(1, 2, 4, 1),
        dq,
        dk,
        dv,
        scale,
        None,
        None,
        None,
        None,
        None,
    )


class _RunGraph(_FakeGraph):
    """A graph for run(): `watch(point)` is called at each stage; `built` counts plan builds."""

    def __init__(self, watch, fail_at=None):
        super().__init__(None)
        self.watch = watch
        self.fail_at = fail_at
        self.executed = 0

    def _step(self, point):
        self.watch(point)
        return _Bad() if point == self.fail_at else _Ok()

    def build_operation_graph(self, handle):
        return self._step("build_operation_graph")

    def create_execution_plans(self, modes):
        return self._step("create_execution_plans")

    def check_support(self):
        return self._step("check_support")

    def build_plans(self):
        _kernel_log.on_line(SELECTED_LINE)
        return self._step("build_plans")

    def get_plan_name(self, handle):
        return ENGINE

    def execute(self, handle, variant_pack, workspace):
        self.watch("execute")
        self.executed += 1
        return _Ok()


class _Bad:
    def is_bad(self):
        return True

    def get_message(self):
        return "refused"


@pytest.fixture
def rt(monkeypatch, tmp_path):
    """Drives run() on fake graphs; `rt.marker()` reads the in-flight marker, `rt.points` the
    (stage, marker) pairs the graph saw."""
    state = types.SimpleNamespace(
        builds=0, graphs=[], fail_at=None, points=[], hooks=[]
    )
    monkeypatch.setattr(_runtime, "_PLANS", {})
    monkeypatch.setattr(_runtime, "_handle", lambda device: object())
    monkeypatch.setattr(_runtime, "_CASE", {"nodeid": None, "fwd": (), "bwd": ()})
    monkeypatch.setattr(_runtime, "_LAST", threading.local())
    monkeypatch.setattr(_kernel_log, "_LOCAL", threading.local())
    monkeypatch.setenv(contract.INFLIGHT_DIR_ENV, str(tmp_path / "inflight"))
    monkeypatch.setenv(contract.PASS_SLUG_ENV, "pass1")
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw7")
    for name in (
        contract.PLAN_CACHE_ENV,
        contract.TEST_CRASH_ENV,
        contract.REPRO_CAPTURE_ENV,
        "HIPDNN_FWD_ENGINE",
        "HIPDNN_BWD_ENGINE",
    ):
        monkeypatch.delenv(name, raising=False)
    frontend = types.SimpleNamespace(
        HeuristicMode=types.SimpleNamespace(FALLBACK="fallback")
    )
    monkeypatch.setattr(_runtime._graph, "frontend", lambda: frontend)

    def marker():
        path = tmp_path / "inflight" / "pass1" / "gw7.json"
        return json.loads(path.read_text()) if path.exists() else None

    def build(prob):
        state.builds += 1

        def watch(point):
            state.points.append((point, marker()))
            for hook in state.hooks:
                hook(point)

        graph = _RunGraph(watch, state.fail_at)
        state.graphs.append(graph)
        return graph, prob.uids()

    monkeypatch.setattr(_runtime._graph, "build", build)
    state.marker = marker
    state.tmp = tmp_path
    return state


def test_marker_is_written_at_entry_after_ranking_and_before_execute(rt):
    _runtime.run(fwd_problem(), None)
    seen = dict(rt.points)
    entry, ranked, execute = (
        seen["build_operation_graph"],
        seen["create_execution_plans"],
        seen["execute"],
    )
    assert set(entry) == set(contract.MARKER_FIELDS)
    assert (entry["nodeid"], entry["direction"], entry["stage"]) == (
        nodeid(),
        "fwd",
        "build",
    )
    assert entry["pid"] == os.getpid() and entry["engine"] == ""
    assert entry["fwd_engines"] == [] and entry["kernel"] == ""
    assert (ranked["stage"], ranked["fwd_engines"]) == ("build", ["ENG_A", "ENG_B"])
    assert ranked["graph_json"] == '{"graph": "fake"}'
    assert (execute["stage"], execute["engine"], execute["kernel"]) == (
        "execute",
        ENGINE,
        KERNEL,
    )
    assert execute["fwd_engines"] == ["ENG_A", "ENG_B"]
    assert sorted(p.name for p in (rt.tmp / "inflight" / "pass1").iterdir()) == [
        "gw7.json"
    ]


def test_marker_carries_the_pins_and_the_forward_list_into_the_backward(
    rt, monkeypatch
):
    monkeypatch.setenv("HIPDNN_FWD_ENGINE", "pin_f")
    monkeypatch.setenv("HIPDNN_BWD_ENGINE", "pin_b")
    _runtime.run(fwd_problem(), None)
    del rt.points[:]
    _runtime.run(bwd_problem(), None)
    entry = dict(rt.points)["build_operation_graph"]
    assert (entry["direction"], entry["pin_fwd"], entry["pin_bwd"]) == (
        "bwd",
        "pin_f",
        "pin_b",
    )
    assert entry["fwd_engines"] == ["ENG_A", "ENG_B"]


def test_marker_engine_lists_do_not_leak_into_the_next_case(rt, monkeypatch):
    _runtime.run(fwd_problem(), None)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "other.py::test_other (call)")
    del rt.points[:]
    _runtime.run(bwd_problem(), None)
    entry = dict(rt.points)["build_operation_graph"]
    assert (entry["nodeid"], entry["fwd_engines"]) == ("other.py::test_other", [])


def test_marker_is_not_written_without_a_directory(rt, monkeypatch):
    monkeypatch.delenv(contract.INFLIGHT_DIR_ENV)
    _runtime.run(fwd_problem(), None)
    assert not (rt.tmp / "inflight").exists()


class _Exited(BaseException):
    pass


def test_crash_hook_exits_after_the_execute_marker_in_place_of_execute(rt, monkeypatch):
    monkeypatch.setenv(contract.TEST_CRASH_ENV, f"fwd:{ENGINE}")
    codes = []

    def fake_exit(code):
        codes.append((code, rt.marker()))
        raise _Exited

    monkeypatch.setattr(os, "_exit", fake_exit)
    with pytest.raises(_Exited):
        _runtime.run(fwd_problem(), None)
    ((code, marker),) = codes
    assert code == contract.TEST_CRASH_EXIT_CODE
    assert (marker["stage"], marker["engine"]) == ("execute", ENGINE)
    assert rt.graphs[0].executed == 0


@pytest.mark.parametrize("spec", [f"bwd:{ENGINE}", "fwd:other:engine", "", "fwd"])
def test_crash_hook_ignores_other_directions_engines_and_malformed_values(
    rt, monkeypatch, spec
):
    monkeypatch.setenv(contract.TEST_CRASH_ENV, spec)
    monkeypatch.setattr(os, "_exit", lambda code: pytest.fail("exited"))
    _runtime.run(fwd_problem(), None)
    assert rt.graphs[0].executed == 1


def test_plan_cache_evicts_the_least_recently_used_plan(rt, monkeypatch):
    monkeypatch.setenv(contract.PLAN_CACHE_ENV, "2")
    for scale in (1.0, 2.0, 3.0):  # cache: 2, 3
        _runtime.run(fwd_problem(scale), None)
    assert rt.builds == 3
    _runtime.run(fwd_problem(2.0), None)  # hit; cache order: 3, 2
    assert rt.builds == 3
    _runtime.run(fwd_problem(1.0), None)  # miss; evicts 3
    assert rt.builds == 4
    _runtime.run(fwd_problem(2.0), None)  # still cached
    assert rt.builds == 4
    _runtime.run(fwd_problem(3.0), None)  # was evicted
    assert rt.builds == 5


def test_plan_cache_defaults_to_64_entries(rt):
    for scale in range(70):
        _runtime.run(fwd_problem(float(scale + 1)), None)
    assert len(_runtime._PLANS) == 64


def test_a_cache_hit_reports_the_same_ranked_list_and_kernel(rt):
    _runtime.run(fwd_problem(), None)
    first = hipdnn_sdpa.last_call("fwd")
    _runtime.run(fwd_problem(), None)
    assert rt.builds == 1
    second = hipdnn_sdpa.last_call("fwd")
    assert (second.engine, second.kernel, second.ranked) == (
        ENGINE,
        KERNEL,
        ("ENG_A", "ENG_B"),
    )
    assert second.graph_json == first.graph_json == '{"graph": "fake"}'


def test_last_call_describes_the_calling_threads_latest_call_per_direction(rt):
    _runtime.run(fwd_problem(scale=0.25), None)
    _runtime.run(bwd_problem(scale=0.75), None)
    fwd, bwd = hipdnn_sdpa.last_call("fwd"), hipdnn_sdpa.last_call("bwd")
    assert isinstance(fwd, contract.CallRecord)
    assert (fwd.direction, bwd.direction) == ("fwd", "bwd")
    assert (fwd.problem_meta["scale"], bwd.problem_meta["scale"]) == (0.25, 0.75)
    assert fwd.problem_meta["uids"]["q"] == 1 and "stats" in fwd.problem_meta["uids"]
    seen = []
    worker = threading.Thread(target=lambda: seen.append(hipdnn_sdpa.last_call("fwd")))
    worker.start()
    worker.join()
    assert seen == [None]


def test_last_call_is_cleared_by_a_declined_call(rt):
    _runtime.run(fwd_problem(), None)
    rt.fail_at = "check_support"
    with pytest.raises(EngineDeclined):
        _runtime.run(fwd_problem(scale=0.75), None)
    assert hipdnn_sdpa.last_call("fwd") is None


def test_last_call_is_none_before_any_call(rt):
    assert hipdnn_sdpa.last_call("bwd") is None


# ------------------------------------------------------------------------- repro capture


@pytest.fixture
def capture_root(rt, monkeypatch):
    root = rt.tmp / "repro"
    monkeypatch.setenv(contract.REPRO_CAPTURE_ENV, str(root))
    return root / contract.repro_case(nodeid(), "pass1")


def test_capture_writes_inputs_and_graph_before_the_graph_is_built(rt, capture_root):
    seen = {}

    def snapshot(point):
        if point == "build_operation_graph":
            seen["files"] = sorted(p.name for p in capture_root.iterdir())
            seen["graph"] = (capture_root / "fwd.graph.json").read_text()
            seen["tensors"] = torch.load(capture_root / "tensors.pt")
            seen["meta"] = json.loads((capture_root / "meta.json").read_text())

    rt.hooks.append(snapshot)
    rt.fail_at = "build_operation_graph"
    with pytest.raises(EngineDeclined):
        _runtime.run(fwd_problem(scale=0.125), None)
    assert seen["files"] == ["fwd.graph.json", "meta.json", "tensors.pt"]
    assert seen["graph"] == '{"graph": "fake"}'
    assert set(seen["tensors"]) == {"q", "k", "v"}  # outputs hold nothing to replay
    meta = seen["meta"]
    assert (meta["nodeid"], meta["pass"], meta["directions"]) == (
        nodeid(),
        "pass1",
        ["fwd"],
    )
    assert meta["calls"]["fwd"]["scale"] == 0.125


def test_capture_of_a_backward_keeps_its_forward_inputs_and_forward_outputs(
    rt, capture_root
):
    prob = fwd_problem()
    _runtime.run(prob, None)
    prob.tensors["o"].fill_(3.0)  # the forward's result, as the backward will read it
    bprob = bwd_problem()
    bprob.tensors["o"] = prob.tensors["o"]
    _runtime.run(bprob, None)
    tensors = torch.load(capture_root / "tensors.pt")
    assert set(tensors) == {"q", "k", "v", "o", "do", "stats"}
    assert (tensors["o"] == 3.0).all()
    meta = json.loads((capture_root / "meta.json").read_text())
    assert meta["directions"] == ["fwd", "bwd"]
    assert (capture_root / "bwd.graph.json").exists()
    assert not (capture_root / "outputs.pt").exists()


def test_capture_is_off_without_the_environment_variable(rt, monkeypatch):
    monkeypatch.delenv(contract.REPRO_CAPTURE_ENV, raising=False)
    _runtime.run(fwd_problem(), None)
    assert not (rt.tmp / "repro").exists()


def test_a_cache_hit_is_captured_too(rt, capture_root):
    _runtime.run(fwd_problem(), None)
    for entry in capture_root.iterdir():
        entry.unlink()
    _runtime.run(fwd_problem(), None)
    assert rt.builds == 1
    assert (capture_root / "fwd.graph.json").read_text() == '{"graph": "fake"}'
