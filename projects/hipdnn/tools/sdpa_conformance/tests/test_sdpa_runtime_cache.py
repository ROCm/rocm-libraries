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
from hipdnn_sdpa import EngineDeclined, _candidates, _kernel_log, _problem, _runtime
from test_candidates import FakeBench, entry

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
        self.knobs = None  # knob settings of the last create_execution_plan_ext

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

    def create_execution_plan_ext(self, engine_id, knob_settings=None):
        self.pinned = engine_id
        self.knobs = knob_settings
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
    graph = fake_graph(None)  # serves what was pinned
    # ENG_A is not first in the fake's config list: building the first applicable engine
    # instead of the pinned one would pin id 2.
    plan = _runtime._build_plan(_Prob(torch.empty(1), "k"), "ENG_A", object())
    assert (plan.engine, plan.ranked) == ("plan", ())
    assert graph.pinned == 1


# ----------------------------------------------------------------- run(): marker, hooks, LRU

KERNEL = "0b1c2d3e-0000-4000-8000-00000000000a"
ENGINE = "hipkernel:flydsl_sdpa_cdna"
SELECTED_LINE = f"ingestor: engine 'flydsl_sdpa_cdna' selected kernel {KERNEL} at rank 0 from 1 candidate(s)"


def nodeid():
    return contract.current_nodeid()


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
        return _Bad() if self.fail_at == "execute" else _Ok()


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
        contract.EXHAUSTIVE_ENV,
        *contract.CANDIDATE_ENV.values(),
        "HIPDNN_FWD_ENGINE",
        "HIPDNN_BWD_ENGINE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(_candidates, "_CACHE", {})
    monkeypatch.setenv(contract.PREFIX_ENV, "/opt/hipdnn")
    frontend = types.SimpleNamespace(
        HeuristicMode=types.SimpleNamespace(FALLBACK="fallback"),
        KnobSetting=lambda name, value: (name, value),
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


# ----------------------------------------------------------------- candidate pins (exhaustive)

NOT_SUPPORTED = "Engine does not support matched-catalog enumeration"


def pin_candidate(monkeypatch, direction, engine, index, catalog):
    """Pin ``engine`` and its ``index``-th candidate for ``direction``; ``catalog`` maps an
    engine name to its entries (or failure message) in the fake ``hipdnn_bench``."""
    monkeypatch.setattr(_candidates, "_run", FakeBench(catalog))
    monkeypatch.setenv(f"HIPDNN_{direction.upper()}_ENGINE", engine)
    monkeypatch.setenv(contract.CANDIDATE_ENV[direction], str(index))


def three_candidates():
    return {
        "ENG_A": [
            entry("other-0", block=64),
            entry(KERNEL, block=128),
            entry("other-2", block=256),
        ]
    }


def test_a_pinned_candidate_is_built_from_its_knob_settings_and_executed(
    rt, monkeypatch
):
    pin_candidate(monkeypatch, "fwd", "ENG_A", 1, three_candidates())
    assert _runtime.run(fwd_problem(), "ENG_A") == ENGINE
    (graph,) = rt.graphs
    assert graph.pinned == 1 and graph.knobs == [("block", 128)]
    assert graph.executed == 1
    assert _runtime.last_call("fwd").kernel == KERNEL


def test_a_candidate_whose_kernel_is_not_the_one_built_is_declined_not_honored(
    rt, monkeypatch
):
    pin_candidate(monkeypatch, "fwd", "ENG_A", 0, three_candidates())
    with pytest.raises(EngineDeclined) as declined:
        _runtime.run(fwd_problem(), "ENG_A")
    reason = declined.value.reason
    assert reason.startswith(contract.CANDIDATE_NOT_HONORED)
    assert "other-0" in reason and KERNEL in reason
    assert rt.graphs[0].executed == 0 and _runtime.last_call("fwd") is None


def test_the_slots_of_a_composite_kernel_compare_whatever_their_order(rt, monkeypatch):
    pin_candidate(
        monkeypatch,
        "bwd",
        "ENG_A",
        0,
        {"ENG_A": [entry("dq=U3|prep=U1|dkdv=U2", block=64)]},
    )

    class Slotted(_RunGraph):
        def build_plans(self):
            for slot, kernel in (("prep", "U1"), ("dkdv", "U2"), ("dq", "U3")):
                _kernel_log.on_line(
                    f"ingestor: engine 'flydsl_sdpa_cdna' slot '{slot}' selected kernel "
                    f"{kernel} at rank 0 from 1 candidate(s)"
                )
            return _Ok()

    monkeypatch.setattr(
        _runtime._graph, "build", lambda prob: (Slotted(lambda p: None), prob.uids())
    )
    assert _runtime.run(bwd_problem(), "ENG_A") == ENGINE
    assert _runtime.last_call("bwd").kernel == "prep=U1|dkdv=U2|dq=U3"


def test_an_engine_that_cannot_enumerate_declines_with_its_message(rt, monkeypatch):
    pin_candidate(monkeypatch, "fwd", "ENG_A", 0, {"ENG_A": NOT_SUPPORTED})
    with pytest.raises(EngineDeclined) as declined:
        _runtime.run(fwd_problem(), "ENG_A")
    assert declined.value.reason == f"not_enumerable: {NOT_SUPPORTED}"
    assert rt.graphs[0].executed == 0


def test_an_index_past_the_catalog_declines_out_of_range(rt, monkeypatch):
    pin_candidate(monkeypatch, "fwd", "ENG_A", 3, three_candidates())
    with pytest.raises(EngineDeclined) as declined:
        _runtime.run(fwd_problem(), "ENG_A")
    assert declined.value.reason.startswith("candidate_out_of_range")
    assert "3" in declined.value.reason


def test_a_candidate_pin_without_an_engine_pin_is_not_honored(rt, monkeypatch):
    monkeypatch.setenv(contract.CANDIDATE_ENV["fwd"], "0")
    with pytest.raises(EngineDeclined) as declined:
        _runtime.run(fwd_problem(), None)
    assert declined.value.reason.startswith(contract.CANDIDATE_NOT_HONORED)
    assert rt.graphs[0].executed == 0


def test_a_candidate_pin_binds_only_its_own_direction(rt, monkeypatch):
    pin_candidate(monkeypatch, "bwd", "ENG_A", 1, three_candidates())
    monkeypatch.delenv("HIPDNN_FWD_ENGINE", raising=False)
    assert _runtime.run(fwd_problem(), None) == ENGINE  # forward stays on the heuristic
    assert rt.graphs[0].knobs is None


def test_each_candidate_of_a_problem_is_its_own_plan(rt, monkeypatch):
    pin_candidate(monkeypatch, "fwd", "ENG_A", 1, three_candidates())
    prob = fwd_problem()
    _runtime.run(prob, "ENG_A")
    _runtime.run(prob, "ENG_A")
    assert rt.builds == 1
    monkeypatch.setenv(contract.CANDIDATE_ENV["fwd"], "0")
    with pytest.raises(EngineDeclined):
        _runtime.run(prob, "ENG_A")
    assert rt.builds == 2


def test_the_first_unpinned_call_counts_every_ranked_engines_candidates(
    rt, monkeypatch
):
    monkeypatch.setenv(contract.EXHAUSTIVE_ENV, "1")
    monkeypatch.setattr(
        _candidates,
        "_run",
        FakeBench({"ENG_A": three_candidates()["ENG_A"], "ENG_B": NOT_SUPPORTED}),
    )
    _runtime.run(fwd_problem(), None)
    assert _runtime.last_call("fwd").candidates == {"ENG_A": 3, "ENG_B": NOT_SUPPORTED}


def test_calls_count_no_candidates_unless_exhaustive_is_set(rt, monkeypatch):
    fake = FakeBench({"ENG_A": [], "ENG_B": []})
    monkeypatch.setattr(_candidates, "_run", fake)
    _runtime.run(fwd_problem(), None)
    assert _runtime.last_call("fwd").candidates == {} and fake.calls == []


def test_a_pinned_candidate_is_recorded_on_the_call_and_in_its_repro(
    rt, monkeypatch, tmp_path
):
    pin_candidate(monkeypatch, "fwd", "ENG_A", 1, three_candidates())
    _runtime.run(fwd_problem(), "ENG_A")
    call = _runtime.last_call("fwd")
    candidate = {"index": 1, "id": KERNEL, "knobs": {"block": 128}}
    assert call.candidate == candidate
    hipdnn_sdpa.save_repro(tmp_path, {"fwd": call})
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["calls"]["fwd"]["candidate"] == candidate
    assert meta["schema"] == 1


def test_an_unpinned_call_records_no_candidate(rt):
    _runtime.run(fwd_problem(), None)
    assert _runtime.last_call("fwd").candidate is None


def test_a_capture_names_the_pinned_candidate_index(rt, capture_root, monkeypatch):
    pin_candidate(monkeypatch, "fwd", "ENG_A", 2, three_candidates())
    with pytest.raises(EngineDeclined):  # entry 2 is not what the fake builds
        _runtime.run(fwd_problem(), "ENG_A")
    meta = json.loads((capture_root / "meta.json").read_text())
    assert meta["calls"]["fwd"]["candidate"] == {"index": 2}


# ------------------------------------------------- failures other than a decline carry the call


def failing_call(error):
    return getattr(error.value, contract.CALL_ERROR_ATTR)


def test_an_execute_failure_carries_the_failing_call(rt):
    rt.fail_at = "execute"
    with pytest.raises(RuntimeError, match="execute failed: refused") as error:
        _runtime.run(fwd_problem(), None)
    call = failing_call(error)
    assert isinstance(call, contract.CallRecord)
    assert (call.direction, call.engine, call.kernel, call.ranked) == (
        "fwd",
        ENGINE,
        KERNEL,
        ("ENG_A", "ENG_B"),
    )
    assert call.graph_json == '{"graph": "fake"}'
    assert {"q", "k", "v"} <= set(call.tensors)
    assert _runtime.last_call("fwd") is None


def test_an_error_during_the_build_carries_the_call_with_the_pinned_candidate(
    rt, monkeypatch
):
    pin_candidate(monkeypatch, "fwd", "ENG_A", 1, three_candidates())

    def boom(point):
        if point == "check_support":
            raise RuntimeError("backend exploded")

    rt.hooks.append(boom)
    with pytest.raises(RuntimeError, match="backend exploded") as error:
        _runtime.run(fwd_problem(), "ENG_A")
    call = failing_call(error)
    assert (call.direction, call.engine, call.graph_json) == (
        "fwd",
        "ENG_A",
        '{"graph": "fake"}',
    )
    assert call.candidate == {"index": 1, "id": KERNEL, "knobs": {"block": 128}}


def test_an_error_before_the_graph_exists_carries_none(rt, monkeypatch):
    def broken(prob):
        raise RuntimeError("bindings blew up")

    monkeypatch.setattr(_runtime._graph, "build", broken)
    with pytest.raises(RuntimeError) as error:
        _runtime.run(fwd_problem(), None)
    assert failing_call(error) is None


def test_a_decline_carries_no_call(rt):
    rt.fail_at = "check_support"
    with pytest.raises(EngineDeclined) as declined:
        _runtime.run(fwd_problem(), None)
    assert not hasattr(declined.value, contract.CALL_ERROR_ATTR)


# ------------------------------------------------- errors surface instead of an empty value


def test_a_ranking_failure_declines_with_its_message_not_an_empty_ranking(
    rt, monkeypatch
):
    def fail(self, modes):
        raise RuntimeError("no ranking today")

    monkeypatch.setattr(_RunGraph, "get_ranked_engine_ids", fail)
    with pytest.raises(EngineDeclined) as declined:
        _runtime.run(fwd_problem(), None)
    assert declined.value.reason == "rank_engines: no ranking today"


def test_a_graph_that_cannot_serialize_declines_instead_of_recording_an_empty_graph(
    rt, monkeypatch
):
    def fail(self):
        raise RuntimeError("no json")

    monkeypatch.setattr(_RunGraph, "to_json", fail)
    with pytest.raises(EngineDeclined) as declined:
        _runtime.run(fwd_problem(), None)
    assert declined.value.reason == "to_json: no json"
    assert rt.graphs[0].executed == 0


def test_a_pinned_candidate_whose_build_logged_no_selection_declines_naming_the_log(
    rt, monkeypatch
):
    pin_candidate(monkeypatch, "fwd", "ENG_A", 1, three_candidates())

    class Silent(_RunGraph):
        def build_plans(self):
            return _Ok()  # the kernel log is not installed: no selection line

    monkeypatch.setattr(
        _runtime._graph, "build", lambda prob: (Silent(lambda p: None), prob.uids())
    )
    with pytest.raises(EngineDeclined) as declined:
        _runtime.run(fwd_problem(), "ENG_A")
    reason = declined.value.reason
    assert reason.startswith(contract.CANDIDATE_NOT_HONORED)
    assert contract.KERNEL_LOG_ENV in reason and "kernel" in reason
