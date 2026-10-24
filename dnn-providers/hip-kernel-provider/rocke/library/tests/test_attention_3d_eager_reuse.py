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
    assert not au._3D_EAGER_WORKSPACES


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


@pytest.mark.parametrize(
    "change", ["pointer", "batch_stride", "head_stride", "sq_stride"]
)
def test_additive_bias_changes_rebind_and_retain_prior_storage(eager, change):
    from dataclasses import replace

    args, bindings, launches, retained = eager
    args["problem"] = replace(args["problem"], use_additive_bias=True)
    args["additive_bias"] = Tensor(8192)
    args["additive_bias_batch_stride"] = 256
    args["additive_bias_head_stride"] = 64
    args["additive_bias_sq_stride"] = 0
    with au.no_fence():
        au._run_3d_tiled(**args)
        prior_resources = next(iter(retained[17].values()))[1]
        au._run_3d_tiled(**args)
        assert len(bindings) == 2 and len(retained[17]) == 1
        if change == "pointer":
            args["additive_bias"].pointer += 1024
        else:
            args[f"additive_bias_{change}"] += 64
        au._run_3d_tiled(**args)
    assert len(bindings) == 4 and len(launches) == 6
    assert len(retained[17]) == 2
    assert prior_resources[-1].pointer == 8192
    assert bindings[2]["additive_bias_ptr"] == args["additive_bias"].pointer
    for name in ("batch_stride", "head_stride", "sq_stride"):
        assert bindings[2][f"additive_bias_{name}"] == args[f"additive_bias_{name}"]


def test_reused_pair_still_requires_additive_bias(eager):
    from dataclasses import replace

    args, bindings, launches, _retained = eager
    args["problem"] = replace(args["problem"], use_additive_bias=True)
    args["additive_bias"] = Tensor(8192)
    with au.no_fence():
        au._run_3d_tiled(**args)
        args["additive_bias"] = None
        with pytest.raises(ValueError, match="requires an additive_bias tensor"):
            au._run_3d_tiled(**args)
    assert len(bindings) == 2 and len(launches) == 2


@pytest.mark.parametrize("cleanup", ["wait_stream", "release_pending_for_stream"])
def test_explicit_stream_cleanup_after_caller_device_restored(monkeypatch, cleanup):
    import gc
    import weakref

    from rocke.runtime import hip_module

    rt = Runtime.__new__(Runtime)
    rt._pending_args = {}
    rt._pending_bound_args = {}
    device = [1]
    monkeypatch.setattr(hip_module, "current_device", lambda: device[0])
    drains = []
    monkeypatch.setattr(rt, "stream_sync", lambda stream: drains.append(stream))
    storage = Tensor(8192)
    storage_ref = weakref.ref(storage)
    first, second = object(), object()
    rt.retain_until_drain(17, first, resources=lambda: (storage,))
    rt.retain_until_drain(29, second)
    del storage
    device[0] = 0  # the launch wrapper restored the caller's device
    getattr(rt, cleanup)(17)
    gc.collect()
    assert storage_ref() is None
    assert rt._pending_bound_args == {(1, 29): {id(second): second}}
    assert device[0] == 0
    assert drains == ([17] if cleanup == "wait_stream" else [])


@pytest.mark.parametrize("stream", [0, 1, 2])
@pytest.mark.parametrize("cleanup", ["wait_stream", "release_pending_for_stream"])
def test_default_stream_cleanup_stays_device_scoped(monkeypatch, stream, cleanup):
    from rocke.runtime import hip_module

    rt = Runtime.__new__(Runtime)
    rt._pending_args = {}
    rt._pending_bound_args = {}
    device = [0]
    monkeypatch.setattr(hip_module, "current_device", lambda: device[0])
    monkeypatch.setattr(rt, "stream_sync", lambda stream: None)
    first, second = object(), object()
    first_key = rt._stream_key(stream)
    rt.retain_until_drain(stream, first)
    device[0] = 1
    rt.retain_until_drain(stream, second)
    getattr(rt, cleanup)(stream)
    assert rt._pending_bound_args == {first_key: {id(first): first}}


def test_cross_device_failed_drain_keeps_retained_storage(monkeypatch):
    from rocke.runtime import hip_module

    rt = Runtime.__new__(Runtime)
    rt._pending_args = {}
    rt._pending_bound_args = {}
    device = [1]
    monkeypatch.setattr(hip_module, "current_device", lambda: device[0])
    owner = object()
    rt.retain_until_drain(17, owner)
    device[0] = 0

    def fail(stream):
        raise RuntimeError("drain failed")

    monkeypatch.setattr(rt, "stream_sync", fail)
    with pytest.raises(RuntimeError, match="drain failed"):
        rt.wait_stream(17)
    assert rt._pending_bound_args == {(1, 17): {id(owner): owner}}
    monkeypatch.setattr(rt, "stream_sync", lambda stream: None)
    rt.wait_stream(17)
    assert not rt._pending_bound_args


def test_device_drain_keeps_other_device_explicit_streams(monkeypatch):
    from rocke.runtime import hip_module

    rt = Runtime.__new__(Runtime)
    rt._pending_args = {}
    rt._pending_bound_args = {}
    device = [0]
    monkeypatch.setattr(hip_module, "current_device", lambda: device[0])
    monkeypatch.setattr(hip_module, "_hipDeviceSynchronize", lambda: 0)
    first, second = object(), object()
    rt.retain_until_drain(17, first)
    device[0] = 1
    rt.retain_until_drain(29, second)
    rt.sync()
    assert rt._pending_bound_args == {(0, 17): {id(first): first}}


@pytest.mark.parametrize("stream", [0, 1, 2, 29])
@pytest.mark.parametrize("launch_kind", ["extra", "kernelparams"])
def test_device_drain_keeps_other_device_fifo_and_bound_storage(
    monkeypatch, stream, launch_kind
):
    import ctypes
    import gc
    import weakref

    from rocke.runtime import hip_module

    rt = Runtime.__new__(Runtime)
    rt._pending_args, rt._pending_bound_args = {}, {}
    device = [0]
    monkeypatch.setattr(hip_module, "current_device", lambda: device[0])
    monkeypatch.setattr(hip_module, "_hipDeviceSynchronize", lambda: 0)
    monkeypatch.setattr(rt, "_enqueue", lambda *a: ())
    refs = []
    for index in (0, 1):
        device[0] = index
        selected = stream if stream in (0, 1, 2) or index == 1 else 37
        fifo, bound = Tensor(8192), Tensor(16384)
        refs.append((weakref.ref(fifo), weakref.ref(bound)))
        if launch_kind == "extra":
            rt.launch(object(), (1, 1, 1), (64, 1, 1), b"args", stream=selected)
        else:
            rt.launch_kernelparams(
                object(),
                (1, 1, 1),
                (64, 1, 1),
                [ctypes.c_uint64(1)],
                stream=selected,
                record_event=False,
            )
        rt.retain_for_stream(selected, fifo)
        rt.retain_until_drain(selected, object(), resources=lambda: (bound,))
        del fifo, bound
    device[0] = 0
    rt.sync()
    gc.collect()
    assert all(ref() is None for ref in refs[0])
    assert all(ref() is not None for ref in refs[1])
    assert all(key[0] == 1 for key in rt._pending_args)
    assert all(key[0] == 1 for key in rt._pending_bound_args)
    device[0] = 1
    rt.sync()
    gc.collect()
    assert all(ref() is None for ref in refs[1])
    assert not rt._pending_args and not rt._pending_bound_args


@pytest.mark.parametrize("cleanup", ["wait_stream", "release_pending_for_stream"])
def test_explicit_stream_cleanup_releases_both_registries_after_device_restore(
    monkeypatch, cleanup
):
    from rocke.runtime import hip_module

    rt = Runtime.__new__(Runtime)
    rt._pending_args, rt._pending_bound_args = {}, {}
    device, destroyed, drains = [1], [], []
    monkeypatch.setattr(hip_module, "current_device", lambda: device[0])
    monkeypatch.setattr(rt, "stream_sync", drains.append)

    class Completion:
        def destroy(self):
            destroyed.append(True)

    rt._pending_args[(1, 17)] = [((object(),), Completion())]
    rt.retain_until_drain(17, object())
    rt.retain_for_stream(29, object())
    rt.retain_until_drain(29, object())
    device[0] = 0
    getattr(rt, cleanup)(17)
    assert destroyed == [True]
    assert drains == ([17] if cleanup == "wait_stream" else [])
    assert set(rt._pending_args) == set(rt._pending_bound_args) == {(1, 29)}
    assert device[0] == 0


@pytest.mark.parametrize("device_wide", [False, True])
def test_failed_drain_preserves_both_registries_for_retry(monkeypatch, device_wide):
    from rocke.runtime import hip_module

    rt = Runtime.__new__(Runtime)
    rt._pending_args, rt._pending_bound_args = {}, {}
    monkeypatch.setattr(hip_module, "current_device", lambda: 1)
    destroyed = []

    def fail(*args):
        raise RuntimeError("completion failed")

    rt._pending_args[(1, 17)] = [
        ((object(),), SimpleNamespace(destroy=lambda: destroyed.append(True)))
    ]
    rt.retain_until_drain(17, object())
    if device_wide:
        monkeypatch.setattr(hip_module, "_hipDeviceSynchronize", fail)
        drain = rt.sync
    else:
        monkeypatch.setattr(rt, "stream_sync", fail)
        drain = lambda: rt.wait_stream(17)
    with pytest.raises(RuntimeError, match="completion failed"):
        drain()
    assert set(rt._pending_args) == set(rt._pending_bound_args) == {(1, 17)}
    assert not destroyed
    monkeypatch.setattr(hip_module, "_hipDeviceSynchronize", lambda: 0)
    monkeypatch.setattr(rt, "stream_sync", lambda s: None)
    drain()
    assert destroyed == [True]
    assert not rt._pending_args and not rt._pending_bound_args


def test_maximum_length_within_same_selector_bucket_reuses_pair(eager, runtime):
    from dataclasses import replace

    args, bindings, launches, _retained = eager
    start = len(runtime.allocations)
    with au.no_fence():
        au._run_3d_tiled(**args)
        args["problem"] = replace(args["problem"], max_seqlen_k=66)
        au._run_3d_tiled(**args)
    assert len(au._3D_EAGER_WORKSPACES) == 1
    assert len(runtime.allocations) - start == 3
    assert len(bindings) == 2 and len(launches) == 4


@pytest.mark.parametrize("change", ["total_q", "segments"])
def test_larger_geometry_allocates_larger_scratch_and_retains_prior_pair(
    eager, runtime, monkeypatch, change
):
    from dataclasses import replace
    from math import prod
    import sys

    args, bindings, launches, retained = eager
    monkeypatch.setattr(au, "_num_segments", lambda *a: 128)
    torch = sys.modules["torch"]
    empty = torch.empty
    dtypes = []

    def record_empty(shape, *, dtype, device):
        dtypes.append(dtype)
        return empty(shape, dtype=dtype, device=device)

    monkeypatch.setattr(torch, "empty", record_empty)
    start = len(runtime.allocations)
    with au.no_fence():
        au._run_3d_tiled(**args, stream=17)
        prior = next(iter(retained[17].values()))
        small = runtime.allocations[start:]
        assert [t.shape for t in small] == [(1, 4, 128, 64), (1, 4, 128), (1, 4, 128)]
        assert sum(prod(t.shape) * 4 for t in small) == 135_168
        if change == "total_q":
            args["problem"] = replace(args["problem"], total_q=2)
            expected = [(2, 4, 128, 64), (2, 4, 128), (2, 4, 128)]
        else:
            monkeypatch.setattr(au, "_num_segments", lambda *a: 256)
            expected = [(1, 4, 256, 64), (1, 4, 256), (1, 4, 256)]
        au._run_3d_tiled(**args, stream=17)
        large = runtime.allocations[start + 3 :]
        assert [t.shape for t in large] == expected
        assert sum(prod(t.shape) * 4 for t in large) == 270_336
        assert all(a is not b for a, b in zip(small, large))
        assert bindings[2]["segm_output_ptr"] is large[0]
        au._run_3d_tiled(**args, stream=17)
        assert len(runtime.allocations) - start == 6
        assert len(bindings) == 4 and len(launches) == 6
        assert len(au._3D_EAGER_WORKSPACES) == len(retained[17]) == 2
        assert any(entry is prior for entry in retained[17].values())
        assert all(t is s for t, s in zip(prior[0].__defaults__[0], small))
    assert dtypes == [torch.float32] * 6
    assert all(t.stream == 17 for t in small + large)
    au.wait_stream_and_release(17)
    assert not retained


@pytest.mark.parametrize(
    "change", ["total_q", "num_seqs", "segments", "spec", "strided"]
)
def test_resolved_geometry_and_binary_identity_keep_separate_owners(
    eager, monkeypatch, change
):
    from dataclasses import replace

    args, bindings, _launches, _retained = eager
    with au.no_fence():
        au._run_3d_tiled(**args)
        if change in ("total_q", "num_seqs"):
            args["problem"] = replace(args["problem"], **{change: 2})
        elif change == "segments":
            monkeypatch.setattr(au, "_num_segments", lambda *a: 256)
        elif change == "spec":
            key = au._tiled_3d_cache_key(args["problem"], "gfx942")
            args["tuning_spec"] = SimpleNamespace(
                kernel_spec=SimpleNamespace(num_segments=128),
                cache_key=lambda: key + ("different_spec",),
            )
        else:
            args["kv_cache_layout"] = SimpleNamespace(arguments=dict)
        au._run_3d_tiled(**args)
    assert len(au._3D_EAGER_WORKSPACES) == 2
    assert bindings[0]["segm_output_ptr"] is not bindings[2]["segm_output_ptr"]


def test_clear_drains_once_per_stream_before_dropping_owners(
    eager, runtime, monkeypatch
):
    from dataclasses import replace

    from kernels import clear_attention_3d_eager_cache

    args, _bindings, _launches, retained = eager
    with au.no_fence():
        au._run_3d_tiled(**args)
        args["problem"] = replace(args["problem"], total_q=2)
        au._run_3d_tiled(**args)
        au._run_3d_tiled(**args, stream=29)
    graphs, pipelines = au._3D_GRAPHS, au._3D_PIPELINES
    drains = []

    def drain(stream):
        assert any(owner.stream == stream for owner in au._3D_EAGER_WORKSPACES.values())
        drains.append((runtime.device, stream))
        retained.pop(stream, None)

    monkeypatch.setattr(au, "wait_stream_and_release", drain)
    clear_attention_3d_eager_cache()
    assert drains == [(0, 17), (0, 29)]
    assert not au._3D_EAGER_WORKSPACES and not retained
    assert au._3D_GRAPHS is graphs and au._3D_PIPELINES is pipelines
    assert (runtime.device, runtime.stream) == (0, 17)


def test_clear_failed_drain_keeps_affected_owners_for_retry(eager, monkeypatch):
    args, _bindings, _launches, retained = eager
    with au.no_fence():
        au._run_3d_tiled(**args)
    owner = next(iter(au._3D_EAGER_WORKSPACES.values()))

    def fail(stream):
        raise RuntimeError("drain failed")

    monkeypatch.setattr(au, "wait_stream_and_release", fail)
    with pytest.raises(RuntimeError, match="drain failed"):
        au.clear_attention_3d_eager_cache()
    assert next(iter(au._3D_EAGER_WORKSPACES.values())) is owner
    assert retained
    monkeypatch.setattr(
        au, "wait_stream_and_release", lambda stream: retained.pop(stream, None)
    )
    au.clear_attention_3d_eager_cache()
    assert not au._3D_EAGER_WORKSPACES and not retained


def test_clear_rejects_selected_stream_capture_without_draining(eager, monkeypatch):
    args, _bindings, _launches, retained = eager
    with au.no_fence():
        au._run_3d_tiled(**args, stream=29)
    monkeypatch.setattr(
        au,
        "_torch_stream_capturing",
        lambda: __import__("torch").cuda.current_stream().cuda_stream == 29,
    )
    with pytest.raises(RuntimeError, match="during capture"):
        au.clear_attention_3d_eager_cache()
    assert len(au._3D_EAGER_WORKSPACES) == 1 and retained


def test_saved_executable_candidate_binding_uses_real_ordinary_runtime(eager, runtime):
    from dataclasses import replace

    from dispatch.attention import (
        AttentionRequest,
        attention_execution_candidates,
        attention_tuning_spec,
    )

    class _Tensor(Tensor):
        def __init__(self, shape, dtype, values=None):
            super().__init__(id(self))
            self.shape, self.dtype, self.values = shape, dtype, values

        def is_contiguous(self):
            return True

        def stride(self, dim):
            from math import prod

            return prod(self.shape[dim + 1 :])

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return self.values

    # Tensor transport and compilation are mocked; candidate binding, metadata
    # validation and the public runtime connection remain real.
    args, bindings, launches, retained = eager
    problem = replace(args["problem"], target_ctas=0, num_cus=8)
    req = AttentionRequest(
        batch=1,
        nhead_q=4,
        nhead_k=1,
        seqlen_q=1,
        seqlen_k=65,
        hdim_q=64,
        hdim_v=64,
        arch="gfx942",
        dtype="fp16",
        kv_block_size=16,
        num_cus=8,
    )
    candidate = next(
        c
        for c in attention_execution_candidates()
        if c.spec_id.startswith("gfx942_u3d")
    )
    spec = attention_tuning_spec(req, candidate.spec_id)
    tensors = {
        "problem": problem,
        "q": _Tensor((1, 4, 64), "torch.float16"),
        "out": _Tensor((1, 4, 64), "torch.float16"),
        "k": _Tensor((5, 16, 1, 64), "torch.float16"),
        "v": _Tensor((5, 16, 1, 64), "torch.float16"),
        "cu_seqlens_q": _Tensor((2,), "torch.int32", [0, 1]),
        "seqused_k": _Tensor((1,), "torch.int32", [65]),
        "block_table": _Tensor((1, 5), "torch.int32", [list(range(5))]),
    }
    bound = candidate.bind_torch(req, spec, tensors, stream=17)
    start = len(runtime.allocations)
    with au.no_fence():
        for _ in range(20):
            bound.launch()
    assert len(runtime.allocations) - start == 3
    assert len(bindings) == 2 and len(launches) == 40
    assert len(retained[17]) == 1


def test_clear_restores_device_and_preserves_unrelated_caches(
    eager, runtime, monkeypatch
):
    from collections import OrderedDict

    args, _bindings, _launches, _retained = eager
    with au.no_fence():
        au._run_3d_tiled(**args, stream=29)
        args["q"].device = "cuda:1"
        au._run_3d_tiled(**args, stream=31)
    graphs = OrderedDict(sentinel=object())
    pipelines = {("sentinel",): object()}
    monkeypatch.setattr(au, "_3D_GRAPHS", graphs)
    monkeypatch.setattr(au, "_3D_PIPELINES", dict(pipelines))
    drains = []
    monkeypatch.setattr(
        au,
        "wait_stream_and_release",
        lambda stream: drains.append((runtime.device, stream)),
    )
    au.clear_attention_3d_eager_cache()
    assert drains == [(0, 29), (1, 31)]
    assert (runtime.device, runtime.stream) == (0, 17)
    assert len(au._3D_GRAPHS) == 1 and au._3D_PIPELINES == pipelines


def test_clear_rejects_active_submission(eager):
    args, *_ = eager
    au._run_3d_tiled(**args)
    owner = next(iter(au._3D_EAGER_WORKSPACES.values()))
    with owner.lock:
        with pytest.raises(RuntimeError, match="during submission"):
            au.clear_attention_3d_eager_cache()
    assert len(au._3D_EAGER_WORKSPACES) == 1


@pytest.mark.parametrize("foreign_stream", [0, 2, 29])
@pytest.mark.parametrize("drain_fails", [False, True])
def test_clear_preserves_live_foreign_owner_and_reclaims_exited_thread(
    eager, runtime, monkeypatch, foreign_stream, drain_fails
):
    from threading import Event, Thread, current_thread

    args, *_ = eager
    runtime.stream = 0 if foreign_stream == 0 else 17
    au._run_3d_tiled(**args, stream=17)
    ready, finish = Event(), Event()
    drains, device_drains, errors = [], [], []
    monkeypatch.setattr(au, "wait_stream_and_release", drains.append)

    def drain_device():
        device_drains.append(runtime.device)
        if drain_fails and len(device_drains) == 1:
            raise RuntimeError("device drain failed")

    monkeypatch.setattr(au, "synchronize_and_release", drain_device)

    def worker():
        try:
            with au.no_fence():
                au._run_3d_tiled(**args, stream=foreign_stream)
            ready.set()
            assert finish.wait(5)
        except BaseException as error:
            errors.append(error)
            ready.set()

    thread = Thread(target=worker)
    thread.start()
    try:
        assert ready.wait(5) and not errors
        assert len(au._3D_EAGER_WORKSPACES) == 2
        au.clear_attention_3d_eager_cache()
        assert drains == [17] and not device_drains
        assert len(au._3D_EAGER_WORKSPACES) == 1
        assert next(iter(au._3D_EAGER_WORKSPACES))[-1] is thread
        assert next(iter(au._3D_EAGER_WORKSPACES))[-1] is not current_thread()
    finally:
        finish.set()
        thread.join(5)
    assert not thread.is_alive() and not errors
    if drain_fails:
        with pytest.raises(RuntimeError, match="device drain failed"):
            au.clear_attention_3d_eager_cache()
        assert len(au._3D_EAGER_WORKSPACES) == 1
    au.clear_attention_3d_eager_cache()
    assert not au._3D_EAGER_WORKSPACES
    assert device_drains == ([0, 0] if drain_fails else [0])
    assert drains == [17]
    assert (runtime.device, runtime.stream) == (0, 0 if foreign_stream == 0 else 17)


def test_clear_default_stream_owner_after_switching_current_stream(
    eager, runtime, monkeypatch
):
    args, *_ = eager
    runtime.stream = 0
    au._run_3d_tiled(**args, stream=0)
    runtime.stream = 29
    drains = []

    def drain(stream):
        # Launcher zero resolution must see the owner's default-stream context.
        drains.append((stream, runtime.stream))

    monkeypatch.setattr(au, "wait_stream_and_release", drain)
    au.clear_attention_3d_eager_cache()
    assert drains == [(0, 0)]
    assert not au._3D_EAGER_WORKSPACES
    assert (runtime.device, runtime.stream) == (0, 29)


@pytest.mark.parametrize("retention", ["retain_until_drain", "retain_for_stream"])
@pytest.mark.parametrize(
    "cleanup", ["wait_stream_and_release", "release_retained_for_stream"]
)
def test_public_zero_stream_retention_matches_cleanup(monkeypatch, retention, cleanup):
    import gc
    import sys
    import weakref

    from rocke.runtime import hip_module, launcher

    rt = Runtime.__new__(Runtime)
    rt._pending_args, rt._pending_bound_args = {}, {}
    monkeypatch.setattr(hip_module, "current_device", lambda: 0)
    monkeypatch.setattr(rt, "stream_sync", lambda stream: None)
    monkeypatch.setattr(launcher, "_runtime", lambda: rt)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                current_device=lambda: 0,
                current_stream=lambda device: SimpleNamespace(cuda_stream=17),
            )
        ),
    )
    storage = Tensor(8192)
    storage_ref = weakref.ref(storage)
    if retention == "retain_until_drain":
        launcher.retain_until_drain(0, object(), resources=lambda: (storage,))
    else:
        launcher.retain_for_stream(0, storage)
    del storage
    assert storage_ref() is not None
    getattr(launcher, cleanup)(0)
    gc.collect()
    assert storage_ref() is None
    assert not rt._pending_args and not rt._pending_bound_args


def test_ptds_owners_and_cleanup_are_isolated_between_threads(
    eager, runtime, monkeypatch
):
    from threading import Event, Thread

    from rocke.runtime import hip_module

    args, bindings, _launches, _retained = eager
    rt = Runtime.__new__(Runtime)
    rt._pending_args, rt._pending_bound_args = {}, {}
    monkeypatch.setattr(hip_module, "current_device", lambda: 0)
    monkeypatch.setattr(rt, "stream_sync", lambda stream: None)
    monkeypatch.setattr(au, "retain_until_drain", rt.retain_until_drain)
    ready, cleaned = Event(), Event()
    errors = []
    start = len(runtime.allocations)

    def first_thread():
        try:
            with au.no_fence():
                au._run_3d_tiled(**args, stream=2)
            rt.retain_for_stream(2, object())
            ready.set()
            assert cleaned.wait(5)
            assert len(rt._pending_bound_args) == len(rt._pending_args) == 1
            rt.wait_stream(2)
        except BaseException as exc:
            errors.append(exc)
            ready.set()

    def second_thread():
        try:
            assert ready.wait(5)
            with au.no_fence():
                au._run_3d_tiled(**args, stream=2)
            rt.retain_for_stream(2, object())
            assert len(rt._pending_bound_args) == len(rt._pending_args) == 2
            rt.wait_stream(2)
            assert len(rt._pending_bound_args) == len(rt._pending_args) == 1
        except BaseException as exc:
            errors.append(exc)
        finally:
            cleaned.set()

    threads = [Thread(target=first_thread), Thread(target=second_thread)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    assert not errors and all(not thread.is_alive() for thread in threads)
    assert not rt._pending_bound_args and not rt._pending_args
    assert len(runtime.allocations) - start == 6
    assert len(au._3D_EAGER_WORKSPACES) == 2
    assert bindings[0]["segm_output_ptr"] is not bindings[2]["segm_output_ptr"]
