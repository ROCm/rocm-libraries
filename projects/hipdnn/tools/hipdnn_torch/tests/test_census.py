# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Pure-CPU, provider-free tests for the shared census/lock machinery in
``hipdnn_torch.base``: the ``grad``/``bwd`` census-key split and the totals it
feeds, ``note_aot``'s required ``engine`` argument, the backward-execute-failure
recompute row shape, ``_engine_name``'s handle-first lookup order, and
``_execute``'s stream-binding + ``_cached_graph``'s build-once caching with fake
handles/graphs -- no bootstrap, no GPU, no provider ``.so``.
"""

import logging
import threading
from types import SimpleNamespace

import pytest

from hipdnn_torch.base import (
    BWD_PREFIX,
    GRAD_PREFIX,
    NotApplicable,
    OpOverride,
    _engine_name,
)
from hipdnn_torch.bootstrap import _fnv1a64


def _make_override(op_name="test_op"):
    ov = OpOverride()
    ov.op_name = op_name
    return ov


# --------------------------------------------------------------------------- #
# totals() / bwd_totals() split                                               #
# --------------------------------------------------------------------------- #
def test_totals_and_bwd_totals_split_rows():
    ov = _make_override()
    ov.note_aot("Sq=1,Skv=1,D=1,dtype=f16", "E1")  # plain forward
    ov.note_aot(GRAD_PREFIX + "Sq=1,Skv=1,D=1,dtype=f16", "E2")  # grad-eligible forward
    ov.note_native(BWD_PREFIX + "Sq=1,Skv=1,D=1,dtype=f16", "some reason")  # bwd native
    ov.note_aot(BWD_PREFIX + "Sq=2,Skv=2,D=2,dtype=f16", "E3")  # bwd served

    assert ov.totals() == (2, 0)  # plain + grad rows only, bwd rows excluded
    assert ov.bwd_totals() == (1, 1)


def test_format_report_omits_bwd_total_without_bwd_rows():
    ov = _make_override("linear")
    ov.note_aot("K=4,N=8,dtype=f16", "HIPBLASLT_ENGINE")
    report = ov.format_report()
    assert "TOTAL" in report
    assert "BWD TOTAL" not in report


def test_format_report_includes_bwd_total_with_bwd_rows():
    ov = _make_override("scaled_dot_product_attention")
    ov.note_aot(GRAD_PREFIX + "Sq=4,Skv=4,D=8,dtype=f16", "E_fwd")
    ov.note_aot(BWD_PREFIX + "Sq=4,Skv=4,D=8,dtype=f16", "E_bwd")
    report = ov.format_report()
    assert "BWD TOTAL" in report
    # the bwd row's own aot/native, not the forward-only TOTAL, appears on that line
    bwd_total_line = next(line for line in report.splitlines() if "BWD TOTAL" in line)
    assert "aot=    1" in bwd_total_line
    assert "native=    0" in bwd_total_line


# --------------------------------------------------------------------------- #
# note_aot's required `engine` argument                                       #
# --------------------------------------------------------------------------- #
def test_note_aot_records_the_engine_passed_in():
    ov = _make_override()
    ov.note_aot("key", "ASM_SDPA_ENGINE")
    row = ov.census()["key"]
    assert row["aot"] == 1
    assert row["engines"] == {"ASM_SDPA_ENGINE"}


def test_note_aot_accumulates_distinct_engine_names_on_one_row():
    ov = _make_override()
    ov.note_aot("key", "ASM_SDPA_ENGINE")
    ov.note_aot("key", "hipkernel:flydsl_sdpa_cdna_dualwave")
    row = ov.census()["key"]
    assert row["aot"] == 2
    assert row["engines"] == {"ASM_SDPA_ENGINE", "hipkernel:flydsl_sdpa_cdna_dualwave"}


def test_note_aot_engine_none_adds_no_engines_set():
    ov = _make_override()
    ov.note_aot("key", None, biased=1)
    row = ov.census()["key"]
    assert row["aot"] == 1
    assert "engines" not in row
    assert row["biased"] == 1


# --------------------------------------------------------------------------- #
# Extras render in the human report (dropout, recompute, bwd_recompute)       #
# --------------------------------------------------------------------------- #
def test_format_report_renders_grad_and_bwd_extras():
    ov = _make_override("scaled_dot_product_attention")
    ov.note_aot(
        GRAD_PREFIX + "Sq=4,Skv=4,D=8,dtype=f16",
        "hipkernel:flydsl_sdpa_cdna_dualwave",
        dropout=1,
        recompute=1,
    )
    ov.note_aot(
        BWD_PREFIX + "Sq=4,Skv=4,D=8,dtype=f16", "hipkernel:flydsl_sdpa_bwd_split"
    )
    report = ov.format_report()
    assert "dropout=1" in report
    assert "recompute=1" in report
    assert "engine=hipkernel:flydsl_sdpa_cdna_dualwave" in report


# --------------------------------------------------------------------------- #
# A backward execute failure recomputed natively: a bwd-key note_native marks native, not aot #
# --------------------------------------------------------------------------- #
def test_p4_recompute_row_marks_native_not_aot():
    ov = _make_override("scaled_dot_product_attention")
    grad_key = GRAD_PREFIX + "Sq=1024,Skv=1024,D=64,dtype=bf16"
    bwd_key = BWD_PREFIX + "Sq=1024,Skv=1024,D=64,dtype=bf16"

    ov.note_aot(
        grad_key, "hipkernel:flydsl_sdpa_cdna_dualwave"
    )  # forward already served
    ov.note_native(
        bwd_key,
        "bwd execute failed, recomputed natively: HIP_ERROR_OUT_OF_MEMORY",
        level=logging.WARNING,
    )

    census = ov.census()
    assert census[bwd_key]["native"] == 1
    assert census[bwd_key]["aot"] == 0
    assert (
        census[grad_key]["aot"] == 1
    )  # the forward row is unaffected by the bwd failure
    assert (
        "bwd execute failed, recomputed natively: HIP_ERROR_OUT_OF_MEMORY"
        in ov.fallback_reasons()
    )
    assert ov.bwd_totals() == (0, 1)


# --------------------------------------------------------------------------- #
# _engine_name: handle-first, then local table, then module registry, hex     #
# --------------------------------------------------------------------------- #
class _FakeHandle:
    """Stand-in for ``hipdnn.Handle``: ``engine_id_to_name`` either returns a
    fixed name or raises, as configured."""

    def __init__(self, result):
        self._result = result

    def engine_id_to_name(self, engine_id):
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def test_engine_name_prefers_handle_over_everything():
    handle = _FakeHandle("hipkernel:custom_plugin_engine")
    # even an id the local table and a module registry would both resolve
    # differently -- the handle wins.
    known_id = _fnv1a64("ASM_SDPA_ENGINE")
    hipdnn = SimpleNamespace(engine_id_to_name=lambda eid: "MODULE_NAME")
    assert _engine_name(hipdnn, known_id, handle) == "hipkernel:custom_plugin_engine"


def test_engine_name_falls_back_to_local_table_when_handle_raises():
    handle = _FakeHandle(AttributeError("old wheel, no such method"))
    known_id = _fnv1a64("MIOPEN_ENGINE_DETERMINISTIC")
    assert _engine_name(None, known_id, handle) == "MIOPEN_ENGINE_DETERMINISTIC"


def test_engine_name_falls_back_to_module_registry_when_handle_returns_empty():
    handle = _FakeHandle("")  # falsy: tried, but nothing came back
    hipdnn = SimpleNamespace(engine_id_to_name=lambda eid: "MODULE_ENGINE")
    assert _engine_name(hipdnn, 0xDEADBEEF, handle) == "MODULE_ENGINE"


def test_engine_name_falls_back_to_hex_when_every_lookup_fails():
    handle = _FakeHandle(RuntimeError("no handle lookup"))

    def _raises(engine_id):
        raise RuntimeError("no module lookup either")

    hipdnn = SimpleNamespace(engine_id_to_name=_raises)
    assert _engine_name(hipdnn, 0xABCD, handle) == hex(0xABCD)


def test_engine_name_with_no_handle_still_checks_local_table_then_module():
    known_id = _fnv1a64("HIP_MLOPS_ENGINE")
    assert _engine_name(None, known_id) == "HIP_MLOPS_ENGINE"


# --------------------------------------------------------------------------- #
# _execute: stream binding under a fake handle/torch                          #
# --------------------------------------------------------------------------- #
class _FakeErr:
    def __init__(self, bad=False, message=""):
        self._bad = bad
        self._message = message

    def is_bad(self):
        return self._bad

    def get_message(self):
        return self._message


class _FakeGraph:
    def __init__(self):
        self.execute_calls = 0

    def execute(self, handle, variant_pack, ws_ptr):
        self.execute_calls += 1
        return _FakeErr(False)


class _FakeStreamHandle:
    """Stand-in for ``hipdnn.Handle``: records every ``set_stream`` call and
    optionally raises on a chosen stream id."""

    def __init__(self, raise_on_sid=None):
        self.set_stream_calls = []
        self._raise_on_sid = raise_on_sid

    def set_stream(self, sid):
        self.set_stream_calls.append(sid)
        if self._raise_on_sid is not None and sid == self._raise_on_sid:
            raise RuntimeError("hipStreamSet failed")


def _fake_state(handle, bound_stream, current_stream_id):
    torch_ns = SimpleNamespace(
        uint8="uint8",
        empty=lambda n, dtype, device: SimpleNamespace(data_ptr=lambda: 0xBEEF),
        cuda=SimpleNamespace(
            current_stream=lambda device: SimpleNamespace(cuda_stream=current_stream_id)
        ),
    )
    return SimpleNamespace(
        torch=torch_ns,
        handle=handle,
        bound_stream=bound_stream,
        lock=threading.Lock(),
    )


def test_execute_makes_no_set_stream_call_on_the_default_stream():
    handle = _FakeStreamHandle()
    ov = _make_override()
    ov.state = _fake_state(handle, bound_stream=0, current_stream_id=0)
    graph = _FakeGraph()
    ov._execute({"ws": 0, "graph": graph}, {}, device="cuda:0")
    assert handle.set_stream_calls == []
    assert graph.execute_calls == 1
    assert ov.state.bound_stream == 0


def test_execute_binds_a_new_stream_once_and_not_again():
    handle = _FakeStreamHandle()
    ov = _make_override()
    ov.state = _fake_state(handle, bound_stream=0, current_stream_id=7)
    graph = _FakeGraph()
    entry = {"ws": 0, "graph": graph}

    ov._execute(entry, {}, device="cuda:0")
    assert handle.set_stream_calls == [7]
    assert ov.state.bound_stream == 7

    ov._execute(entry, {}, device="cuda:0")  # same stream again
    assert handle.set_stream_calls == [7]  # not called a second time
    assert graph.execute_calls == 2


def test_execute_set_stream_failure_raises_not_applicable_with_prefixed_reason():
    handle = _FakeStreamHandle(raise_on_sid=3)
    ov = _make_override()
    ov.state = _fake_state(handle, bound_stream=0, current_stream_id=3)
    graph = _FakeGraph()

    with pytest.raises(NotApplicable) as exc_info:
        ov._execute({"ws": 0, "graph": graph}, {}, device="cuda:0")
    assert str(exc_info.value).startswith("set_stream:")
    assert graph.execute_calls == 0  # never reached execute
    assert ov.state.bound_stream == 0  # unchanged on failure


# --------------------------------------------------------------------------- #
# _cached_graph: miss path builds once and caches (fake graph)                #
# --------------------------------------------------------------------------- #
class _FakeHeuristicMode:
    FALLBACK = "FALLBACK"


class _FakeBuildGraph:
    def build_operation_graph(self, handle):
        return _FakeErr(False)

    def get_ranked_engine_ids(self, modes):
        return [1]

    def create_execution_plans(self, modes):
        return _FakeErr(False)

    def check_support(self):
        return _FakeErr(False)

    def build_plans(self):
        return _FakeErr(False)

    def get_workspace_size(self):
        return 0

    def get_execution_plan_engine_id(self):
        return 0xCAFE


def test_cached_graph_miss_path_builds_once_and_caches():
    build_count = 0
    graph = _FakeBuildGraph()

    def build():
        nonlocal build_count
        build_count += 1
        return graph

    ov = _make_override()
    ov.state = SimpleNamespace(
        handle=SimpleNamespace(engine_id_to_name=lambda eid: "HANDLE_ENGINE"),
        hipdnn=SimpleNamespace(HeuristicMode=_FakeHeuristicMode),
        select_mode="default",
        engine_id=None,
        engine_name=None,
        lock=threading.Lock(),
    )

    entry1 = ov._cached_graph(("k",), build, "desc")
    assert build_count == 1
    assert entry1 == {"graph": graph, "ws": 0, "engine": "HANDLE_ENGINE"}

    entry2 = ov._cached_graph(("k",), build, "desc")
    assert build_count == 1  # cached: build() not called again
    assert entry2 is entry1
