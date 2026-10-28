# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host checks for decode launch ordering and cleanup when a test scope fails."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from tests._decode_numeric_support import decode_launch_session


def _runtime(events, args, fault=None):
    def record(event):
        events.append(event)
        if fault == event:
            raise RuntimeError(event)

    def check_arguments(kwargs):
        assert kwargs.keys() == args.keys()
        assert all(kwargs[key] is value for key, value in args.items())

    owner = SimpleNamespace(
        launch=lambda: record("owner.launch"), close=lambda: record("owner.close")
    )
    graph = SimpleNamespace(replay=lambda: record("graph.replay"))

    def prepare(**kwargs):
        check_arguments(kwargs)
        record("prepare")
        return owner

    def run(**kwargs):
        check_arguments(kwargs)
        record("run")

    @contextmanager
    def capture(captured):
        assert captured is graph
        record("capture.enter")
        try:
            yield
        finally:
            record("capture.exit")

    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            CUDAGraph=lambda: graph,
            graph=capture,
            synchronize=lambda: record("synchronize"),
        )
    )
    au = SimpleNamespace(
        prepare_unified_attention_torch=prepare,
        run_unified_attention_torch=run,
        clear_attention_3d_graph_cache=lambda: record("clear"),
    )
    return torch, au


@pytest.mark.parametrize("mode", ["eager", "prepared", "graph", "external"])
def test_decode_launch_order_and_original_buffers(mode):
    events = []
    args = dict(
        out=object(), block_table=object(), lse=object(), use_graph=mode == "graph"
    )
    torch, au = _runtime(events, args)
    with decode_launch_session(torch, au, args, mode) as launch:
        launch()
        launch()
    setup = []
    if mode in ("prepared", "external"):
        setup.append("prepare")
    if mode == "external":
        setup += ["capture.enter", "owner.launch", "capture.exit"]
    call = {
        "eager": "run",
        "graph": "run",
        "prepared": "owner.launch",
        "external": "graph.replay",
    }[mode]
    teardown = (
        ["owner.close", "clear"] if mode in ("prepared", "external") else ["clear"]
    )
    assert events == setup + [call, "synchronize"] * 2 + teardown


@pytest.mark.parametrize(
    "fault", ["prepare", "capture.enter", "owner.launch", "assertion", "owner.close"]
)
def test_decode_scope_failure_closes_owner_and_clears_cache(fault):
    events = []
    args = dict(out=object(), block_table=object(), use_graph=False)
    torch, au = _runtime(events, args, fault=fault)
    mode = "external" if fault == "capture.enter" else "prepared"
    error = AssertionError if fault == "assertion" else RuntimeError
    with pytest.raises(error, match=fault):
        with decode_launch_session(torch, au, args, mode) as launch:
            if fault == "assertion":
                raise AssertionError("assertion")
            launch()
    assert events[-1] == "clear"
    if fault == "prepare":
        assert events == ["prepare", "clear"]
    else:
        assert events[-2:] == ["owner.close", "clear"]
