# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Graph binding guards preserve mutable storage and bounded ownership."""

import sys
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
from kernels.common import attention_unified as au
from test_attention_workspace import _arguments
from test_attention_workspace import runtime as _runtime

runtime = _runtime


class Tensor:
    def __init__(self, pointer):
        self.device = "cuda:0"
        self.dtype = "float16"
        self.shape = (1, 4, 64)
        self.strides = (256, 64, 1)
        self.pointer = pointer
        self.contents = 0

    def stride(self):
        return self.strides

    def data_ptr(self):
        return self.pointer


@pytest.fixture
def replay(runtime, monkeypatch):
    torch = sys.modules["torch"]
    operations = []

    class Graph:
        def replay(self):
            operations.append(("replay", self))

    @contextmanager
    def capture(graph):
        previous = runtime.stream
        runtime.stream = 41
        runtime.capturing = True
        try:
            yield
        finally:
            runtime.capturing = False
            runtime.stream = previous

    torch.cuda.CUDAGraph = Graph
    torch.cuda.graph = capture
    graphs = OrderedDict()
    monkeypatch.setattr(au, "_3D_GRAPHS", graphs)
    monkeypatch.setattr(au, "_3D_GRAPH_CAPACITY", 2)
    monkeypatch.setattr(au, "_enable_3d_graph_replay", lambda p, arch: True)
    monkeypatch.setattr(au, "_launch_3d_pipeline", lambda *a, **kw: None)
    monkeypatch.setattr(
        au, "wait_stream_and_release", lambda s: operations.append(("wait", s))
    )
    monkeypatch.setattr(
        au, "retain_for_stream", lambda s, e: operations.append(("retain", e))
    )
    arguments = _arguments()
    arguments["out"] = Tensor(256)
    return arguments, graphs, operations


@pytest.mark.parametrize("owned", [False, True])
def test_warm_replay_and_storage_rebinding(replay, runtime, owned):
    arguments, graphs, operations = replay
    owner = au.Attention3DExecution() if owned else None
    resident = owner._graphs if owned else graphs
    with au.no_fence():
        au._run_3d_tiled(**arguments, execution=owner)
        key = next(iter(resident))
        entry = resident[key]
        allocations = len(runtime.allocations)
        arguments["out"].contents += 1
        operations.clear()
        au._run_3d_tiled(**arguments, execution=owner)
        assert resident[key] is entry
        assert next(iter(resident)) is key
        assert len(runtime.allocations) == allocations
        assert operations == [("replay", entry.graph), ("retain", entry)]

        # Same tensor identity and geometry, different captured storage.
        arguments["out"].pointer += 256
        operations.clear()
        au._run_3d_tiled(**arguments, execution=owner)
        replacement = resident[key]
        assert replacement is not entry and len(resident) == 1
        assert operations[0] == ("wait", entry.stream)
        assert operations[-2:] == [
            ("replay", replacement.graph),
            ("retain", replacement),
        ]
        assert len(runtime.allocations) > allocations
    if owned:
        owner.close()
    else:
        au.clear_attention_3d_graph_cache()
    assert not resident


def test_gfx1250_preserves_context_and_direct_graph_storage_keys(
    replay, runtime, monkeypatch
):
    arguments, resident, operations = replay
    arguments["problem"] = replace(
        arguments["problem"], dtype="bf16", num_query_heads=8
    )
    monkeypatch.setattr(au, "_resolve_attention_arch", lambda: "gfx1250")
    empty = sys.modules["torch"].empty

    def initialized_empty(*args, **kwargs):
        tensor = empty(*args, **kwargs)
        tensor.fill_ = lambda value: operations.append(("fill", value))
        tensor.zero_ = lambda: operations.append(("zero",))
        return tensor

    monkeypatch.setattr(sys.modules["torch"], "empty", initialized_empty)
    arguments["q"] = Tensor(128)
    arguments["q"].device = SimpleNamespace(index=0)
    runtime.contexts.clear()
    au._run_3d_tiled(**arguments, stream=17, use_graph=True)
    assert runtime.contexts
    first_key = next(iter(resident))
    first = resident[first_key]
    assert len(first_key) == 26 and first_key[5] == first.bindings
    assert len(first.bindings) == 11
    assert first.bindings[0][0][:2] == (
        str(arguments["q"].device),
        str(arguments["q"].dtype),
    )
    allocations = len(runtime.allocations)
    arguments["q"].contents += 1
    au._run_3d_tiled(**arguments, stream=17, use_graph=True)
    assert len(runtime.allocations) == allocations and resident[first_key] is first
    arguments["out"].pointer += 256
    au._run_3d_tiled(**arguments, stream=17, use_graph=True)
    assert len(resident) == 2 and resident[first_key] is first
    au.clear_attention_3d_graph_cache()
    assert not resident


@pytest.mark.parametrize("field", ["shape", "strides", "dtype", "device"])
@pytest.mark.parametrize("owned", [False, True])
def test_same_tensor_metadata_change_rechecks_binding(replay, field, owned):
    arguments, graphs, operations = replay
    owner = au.Attention3DExecution() if owned else None
    resident = owner._graphs if owned else graphs
    au._run_3d_tiled(**arguments, execution=owner)
    entry = next(iter(resident.values()))
    changes = {
        "shape": (1, 2, 128),
        "strides": (512, 128, 1),
        "dtype": "bf16",
        "device": "cuda:1",
    }
    setattr(arguments["out"], field, changes[field])
    operations.clear()
    if owned:
        with pytest.raises(ValueError, match="prepare again"):
            au._run_3d_tiled(**arguments, execution=owner)
        assert next(iter(resident.values())) is entry
    else:
        au._run_3d_tiled(**arguments)
        assert next(iter(resident.values())) is not entry
        assert operations[0] == ("wait", entry.stream)


@pytest.mark.parametrize("change", ["scalar", "tensor", "stream", "problem"])
def test_changed_call_identity_does_not_replay_old_graph(replay, change):
    arguments, graphs, operations = replay
    au._run_3d_tiled(**arguments)
    original = next(iter(graphs.values()))
    if change == "scalar":
        arguments["softmax_scale"] *= 2
    elif change == "tensor":
        arguments["out"] = Tensor(512)
    elif change == "stream":
        arguments["stream"] = 29
    else:
        arguments["problem"] = replace(arguments["problem"], max_seqlen_k=128)
    operations.clear()
    au._run_3d_tiled(**arguments)
    assert operations[-1][0] == "wait"  # the default fence follows replay
    replayed = next(op[1] for op in operations if op[0] == "replay")
    assert replayed is not original.graph
    assert len(graphs) <= 2


def test_clear_invalidates_warm_binding(replay):
    arguments, graphs, _operations = replay
    au._run_3d_tiled(**arguments)
    original = next(iter(graphs.values()))
    au.clear_attention_3d_graph_cache()
    assert not graphs
    au._run_3d_tiled(**arguments)
    assert next(iter(graphs.values())) is not original


def test_storage_replacement_drains_the_original_device(replay, runtime, monkeypatch):
    arguments, graphs, _operations = replay
    arguments["q"].device = SimpleNamespace(index=0)
    au._run_3d_tiled(**arguments)
    original = next(iter(graphs.values()))
    arguments["q"].device = SimpleNamespace(index=1)
    runtime.device = 1
    drained_devices = []
    monkeypatch.setattr(
        au, "wait_stream_and_release", lambda s: drained_devices.append(runtime.device)
    )
    au._run_3d_tiled(**arguments)
    assert drained_devices[0] == 0
    assert runtime.device == 1
    assert next(iter(graphs.values())) is not original


def test_older_resident_replay_updates_lru_before_eviction(replay):
    arguments, graphs, operations = replay
    au._run_3d_tiled(**arguments)
    first_key = next(iter(graphs))
    first_entry = graphs[first_key]
    au._run_3d_tiled(**dict(arguments, softmax_scale=2.0))
    second_entry = next(reversed(graphs.values()))
    operations.clear()
    au._run_3d_tiled(**arguments)
    assert next(reversed(graphs)) is first_key
    assert next(op[1] for op in operations if op[0] == "replay") is first_entry.graph
    operations.clear()
    au._run_3d_tiled(**dict(arguments, softmax_scale=3.0))
    assert first_key in graphs and len(graphs) == 2
    assert second_entry not in graphs.values()
    assert operations[0] == ("wait", second_entry.stream)


@pytest.mark.parametrize("owned", [False, True])
def test_additive_bias_storage_replacement_refreshes_graph(replay, owned):
    arguments, graphs, operations = replay
    arguments["problem"] = replace(arguments["problem"], use_additive_bias=True)
    arguments["additive_bias"] = Tensor(8192)
    owner = au.Attention3DExecution() if owned else None
    resident = owner._graphs if owned else graphs
    with au.no_fence():
        au._run_3d_tiled(**arguments, execution=owner)
        key, entry = next(iter(resident.items()))
        assert entry.bindings[-1][1] == 8192
        assert arguments["additive_bias"] in entry.refs
        operations.clear()
        arguments["additive_bias"].contents += 1
        au._run_3d_tiled(**arguments, execution=owner)
        assert resident[key] is entry
        assert operations == [("replay", entry.graph), ("retain", entry)]
        arguments["additive_bias"].pointer += 1024
        operations.clear()
        au._run_3d_tiled(**arguments, execution=owner)
        assert len(resident) == 1 and resident[key] is not entry
        assert operations[0] == ("wait", entry.stream)
        assert resident[key].bindings[-1][1] == 9216


def test_prepared_execution_rejects_changed_additive_bias_strides(replay):
    arguments, _graphs, _operations = replay
    arguments["problem"] = replace(arguments["problem"], use_additive_bias=True)
    arguments["additive_bias"] = Tensor(8192)
    owner = au.Attention3DExecution()
    au._run_3d_tiled(**arguments, execution=owner)
    with pytest.raises(ValueError, match="prepare again"):
        au._run_3d_tiled(**arguments, execution=owner, additive_bias_head_stride=64)
