# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Ordinary eager reuse keeps submission, storage and completion ownership."""

from types import SimpleNamespace

import pytest
from kernels.common import attention_unified as au
from rocke.runtime.hip_module import Runtime
from test_attention_workspace import _arguments
from test_attention_workspace import runtime as _runtime

runtime = _runtime


class Tensor:
    def __init__(self, pointer):
        self.pointer = pointer
        self.device = "cuda:0"

    def data_ptr(self):
        return self.pointer

    def detach(self):
        return Tensor(self.pointer)


@pytest.fixture
def eager(runtime, monkeypatch):
    bindings, launches, retained = [], [], {}

    class Stage:
        def bind(self, values, *, config):
            bindings.append(values)

            def launch():
                launches.append((values, config.stream))

            return launch

    prepared = au._Attention3DPrepared(
        pipeline=SimpleNamespace(stages=(Stage(), Stage())),
        seg_config=au.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)),
        red_config=au.LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)),
    )
    monkeypatch.setattr(au, "_get_3d_pipeline", lambda *a, **kw: prepared)

    def retain(stream, op, *, resources):
        retained.setdefault(stream, {}).setdefault(id(op), (op, resources()))

    monkeypatch.setattr(
        au,
        "retain_until_drain",
        retain,
    )
    monkeypatch.setattr(
        au, "wait_stream_and_release", lambda stream: retained.pop(stream, None)
    )
    arguments = _arguments()
    for name in ("q", "k", "v", "out", "cu_seqlens_q", "seqused_k", "block_table"):
        arguments[name] = Tensor(256 * (len(bindings) + len(name)))
    return arguments, bindings, launches, retained


def test_ordinary_calls_reuse_scratch_and_immutable_pair(eager, runtime):
    args, bindings, launches, retained = eager
    start = len(runtime.allocations)
    with au.no_fence():
        for _ in range(50):
            result = au._run_3d_tiled(**args)
    assert result.launches == 2
    assert len(runtime.allocations) - start == 3
    assert len(bindings) == 2 and len(launches) == 100
    assert len(retained[17]) == 1


def test_pointer_and_scalar_changes_rebind_and_keep_prior_storage(eager):
    args, bindings, launches, retained = eager
    with au.no_fence():
        au._run_3d_tiled(**args)
        prior_pointer = bindings[1]["output_ptr"]
        prior_resources = next(iter(retained[17].values()))[1]
        args["out"].pointer += 1024
        args["softmax_scale"] *= 2
        au._run_3d_tiled(**args)
    assert len(bindings) == 4 and len(retained[17]) == 2
    assert prior_resources[3].pointer == prior_pointer
    assert bindings[3]["output_ptr"] == args["out"].pointer
    assert bindings[2]["scale"] == args["softmax_scale"]
    assert launches[-1][0] is bindings[3]


def test_streams_and_captures_get_separate_scratch(eager, runtime, monkeypatch):
    args, bindings, _launches, _retained = eager
    with au.no_fence():
        au._run_3d_tiled(**args, stream=17)
        au._run_3d_tiled(**args, stream=29)
    assert bindings[0]["segm_output_ptr"] is not bindings[2]["segm_output_ptr"]
    runtime.capturing = True
    captured = []
    monkeypatch.setattr(au, "_launch_3d_pipeline", lambda *a, **kw: captured.append(a))
    au._run_3d_tiled(**args)
    assert captured[0][1]["segm_output_ptr"] is not bindings[0]["segm_output_ptr"]
    assert len(bindings) == 4


def test_eviction_keeps_queued_bound_pair(eager, monkeypatch):
    args, _bindings, _launches, retained = eager
    monkeypatch.setattr(au, "_3D_EAGER_CAPACITY", 1)
    with au.no_fence():
        au._run_3d_tiled(**args, stream=17)
        prior = next(iter(retained[17].values()))
        au._run_3d_tiled(**args, stream=29)
    assert len(au._3D_EAGER_WORKSPACES) == 1
    assert prior in retained[17].values()


def test_capture_on_selected_stream_bypasses_eager_owner(eager, runtime, monkeypatch):
    args, bindings, _launches, _retained = eager
    # The ambient stream is idle; capture is visible only after switching to
    # the stream explicitly selected by this invocation.
    monkeypatch.setattr(au, "_torch_stream_capturing", lambda: runtime.stream == 29)
    captured = []
    monkeypatch.setattr(
        au, "_launch_3d_pipeline", lambda *a, **kw: captured.append((a, kw))
    )
    au._run_3d_tiled(**args, stream=29)
    assert not bindings
    assert captured[0][1]["capturing"]
    assert next(iter(au._3D_EAGER_WORKSPACES.values())).workspace is None


def test_same_stream_reentrant_call_gets_private_scratch(eager, monkeypatch):
    args, bindings, _launches, _retained = eager
    with au.no_fence():
        au._run_3d_tiled(**args)
        owner = next(iter(au._3D_EAGER_WORKSPACES.values()))
        original = owner.operation
        nested = []
        monkeypatch.setattr(
            au, "_launch_3d_pipeline", lambda *a, **kw: nested.append(a)
        )

        def reenter():
            au._run_3d_tiled(**args)
            original()

        owner.operation = reenter
        au._run_3d_tiled(**args)
    assert len(bindings) == 2
    assert nested[0][1]["segm_output_ptr"] is not owner.workspace[0]


def test_failed_pair_remains_retained_and_unlocks(eager, monkeypatch):
    args, _bindings, _launches, retained = eager
    with au.no_fence():
        au._run_3d_tiled(**args)
        owner = next(iter(au._3D_EAGER_WORKSPACES.values()))

        def fail():
            raise RuntimeError("partial pair")

        owner.operation = fail
        with pytest.raises(RuntimeError, match="partial pair"):
            au._run_3d_tiled(**args)
    assert any(entry[0] is fail for entry in retained[17].values())
    assert owner.lock.acquire(blocking=False)
    owner.lock.release()


def test_bound_retention_waits_for_explicit_successful_drain(monkeypatch):
    from rocke.runtime import hip_module

    monkeypatch.setattr(hip_module, "current_device", lambda: 0)
    rt = Runtime.__new__(Runtime)
    rt._pending_args = {}
    rt._pending_bound_args = {}
    owner = object()
    for _ in range(100):
        rt.retain_until_drain(17, owner)
    assert len(rt._pending_bound_args[(0, 17)]) == 1
    rt._reap_completed(17)
    assert rt._pending_bound_args[(0, 17)][id(owner)] is owner

    def fail(_stream):
        raise RuntimeError("drain failed")

    monkeypatch.setattr(rt, "stream_sync", fail)
    with pytest.raises(RuntimeError, match="drain failed"):
        rt.wait_stream(17)
    assert rt._pending_bound_args[(0, 17)][id(owner)] is owner
    monkeypatch.setattr(rt, "stream_sync", lambda stream: None)
    rt.wait_stream(17)
    assert not rt._pending_bound_args


def test_completed_request_storage_is_not_kept_by_binding_cache(eager):
    import gc
    import weakref

    args, _bindings, _launches, retained = eager
    tensor = weakref.ref(args["q"])
    au._run_3d_tiled(**args)
    assert not retained  # fenced call drained the completion registry
    del args["q"]
    gc.collect()
    assert tensor() is None


def test_resources_are_snapshotted_once_per_completion_batch(monkeypatch):
    from rocke.runtime import hip_module

    monkeypatch.setattr(hip_module, "current_device", lambda: 0)
    rt = Runtime.__new__(Runtime)
    rt._pending_args = {}
    rt._pending_bound_args = {}
    owner, snapshot = object(), object()
    calls = []

    def resources():
        calls.append(True)
        return snapshot

    for _ in range(100):
        rt.retain_until_drain(17, owner, resources=resources)
    assert calls == [True]
    assert rt._pending_bound_args[(0, 17)][id(owner)] == (owner, snapshot)
    rt.release_pending_for_stream(17)
    assert not rt._pending_bound_args


@pytest.mark.parametrize("device_sync", [False, True])
def test_null_stream_retention_is_device_scoped(monkeypatch, device_sync):
    from rocke.runtime import hip_module

    rt = Runtime.__new__(Runtime)
    rt._pending_args = {}
    rt._pending_bound_args = {}
    device = [0]
    monkeypatch.setattr(hip_module, "current_device", lambda: device[0])
    monkeypatch.setattr(hip_module, "_hipDeviceSynchronize", lambda: 0)
    monkeypatch.setattr(rt, "stream_sync", lambda stream: None)
    first, second = object(), object()
    rt.retain_until_drain(0, first)
    device[0] = 1
    rt.retain_until_drain(0, second)
    if device_sync:
        rt.sync()
    else:
        rt.wait_stream(0)
    assert rt._pending_bound_args == {(0, 0): {id(first): first}}
    device[0] = 0
    rt.wait_stream(0)
    assert not rt._pending_bound_args
