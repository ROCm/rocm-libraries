# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only multi-device module/event and retained-argument lifetime checks."""

from __future__ import annotations

import ctypes
import gc
import weakref

import pytest
from rocke.runtime import HipError, Runtime, device_scope
from rocke.runtime import hip_module as hip


def launch(
    runtime: Runtime, *, stream: int = 0, event: bool = False
) -> hip.Event | None:
    """Enqueue an opaque test function through the real argument-packing path."""
    return runtime.launch(
        hip._HipFunctionHandle(),
        (1, 1, 1),
        (64, 1, 1),
        b"\x01\x02\x03\x04",
        stream=stream,
        record_event=event,
    )


def test_default_stream_retention_is_device_scoped(fake_hip) -> None:
    runtime = Runtime()
    launch(runtime)
    runtime.retain_for_stream(0, "device_zero")
    with device_scope(1):
        launch(runtime)
        runtime.retain_for_stream(0, "device_one")
        runtime.wait_stream(0)
    assert (0, 0) in runtime._pending_args
    assert (1, 0) not in runtime._pending_args
    assert runtime._pending_args[(0, 0)][0][0][-1] == "device_zero"
    runtime.wait_stream(0)
    assert not runtime._pending_args


def test_shared_buffer_owner_lives_until_every_consumer_drains(fake_hip) -> None:
    class BufferOwner:
        pass

    runtime = Runtime()
    owner = BufferOwner()
    reference = weakref.ref(owner)
    for device in [0, 1]:
        with device_scope(device):
            launch(runtime)
            runtime.retain_for_stream(0, owner)
    del owner
    runtime.sync()
    gc.collect()
    assert reference() is not None
    with device_scope(1):
        runtime.sync()
    gc.collect()
    assert reference() is None


def test_device_sync_does_not_release_other_device_events(fake_hip) -> None:
    runtime = Runtime()
    event_zero = launch(runtime, event=True)
    with device_scope(1):
        event_one = launch(runtime, event=True)
        runtime.sync()
    assert (0, 0) in runtime._pending_args
    assert (1, 0) not in runtime._pending_args
    assert ("event_destroy", 1, event_one.handle.p) in fake_hip.calls
    assert ("event_destroy", 0, event_zero.handle.p) not in fake_hip.calls


def test_reap_and_external_release_affect_only_current_device(fake_hip) -> None:
    runtime = Runtime()
    first = launch(runtime, event=True)
    with device_scope(1):
        second = launch(runtime, event=True)
        fake_hip.ready.add(first.handle.p)
        fake_hip.ready.add(second.handle.p)
        runtime._reap_completed(0)
    assert (0, 0) in runtime._pending_args
    with device_scope(1):
        runtime.release_pending_for_stream(0)
    assert (0, 0) in runtime._pending_args
    runtime._reap_completed(0)
    assert not runtime._pending_args


@pytest.mark.parametrize("kind", ["stream", "device"])
def test_failed_drain_preserves_retained_arguments(fake_hip, kind) -> None:
    runtime = Runtime()
    launch(runtime)
    key = ("stream_sync", 0, 0) if kind == "stream" else ("sync", 0)
    fake_hip.errors[key] = 700
    with pytest.raises(HipError):
        runtime.wait_stream(0) if kind == "stream" else runtime.sync()
    assert len(runtime._pending_args[(0, 0)]) == 1
    fake_hip.errors.clear()
    runtime.wait_stream(0)
    assert not runtime._pending_args


def test_blocking_launch_failed_sync_retains_packed_buffer(fake_hip) -> None:
    runtime = Runtime()
    fake_hip.errors[("stream_sync", 0, 0)] = 700
    with pytest.raises(HipError):
        runtime.launch_blocking(
            hip._HipFunctionHandle(), (1, 1, 1), (64, 1, 1), b"\x01\x02"
        )
    refs, event = runtime._pending_args[(0, 0)][0]
    assert refs[2][1] == ctypes.addressof(refs[0])
    assert event is None


@pytest.mark.parametrize("kind", ["create", "record"])
def test_post_enqueue_event_failure_preserves_arguments(fake_hip, kind) -> None:
    runtime = Runtime()
    key = ("event_create", 0) if kind == "create" else ("event_record", 0, 1000, 0)
    fake_hip.errors[key] = 700
    with pytest.raises(HipError):
        launch(runtime, event=True)
    assert len(runtime._pending_args[(0, 0)]) == 1
    assert runtime._pending_args[(0, 0)][0][1] is None
    if kind == "record":
        assert ("event_destroy", 0, 1000) in fake_hip.calls


def test_failed_enqueue_does_not_create_retention_entry(fake_hip) -> None:
    fake_hip.errors[("launch", 0)] = 700
    with pytest.raises(HipError):
        launch(Runtime())
    assert not Runtime._pending_args


def test_kernelparams_retention_is_also_device_scoped(fake_hip) -> None:
    runtime = Runtime()
    for device in [0, 1]:
        with device_scope(device):
            runtime.launch_kernelparams(
                hip._HipFunctionHandle(),
                (1, 1, 1),
                (64, 1, 1),
                [ctypes.c_int(device)],
                record_event=False,
            )
    assert set(runtime._pending_args) == {(0, 0), (1, 0)}
    runtime.wait_stream(0)
    assert set(runtime._pending_args) == {(1, 0)}


def test_module_operations_use_owner_and_restore_caller(fake_hip) -> None:
    runtime = Runtime()
    with device_scope(1):
        module = runtime.load_module(b"synthetic code object")
    assert module.device == 1
    function = module.get_function("entry")
    assert function._device == 1
    assert ("module_function", 1, b"entry") in fake_hip.calls
    module.unload()
    assert ("module_unload", 1) in fake_hip.calls
    assert fake_hip.device == 0


@pytest.mark.parametrize("method", ["launch", "launch_blocking", "launch_kernelparams"])
def test_wrong_device_function_rejected_before_enqueue(fake_hip, method) -> None:
    runtime = Runtime()
    with device_scope(1):
        function = runtime.load_module(b"synthetic code object").get_function("entry")
    args = [] if method == "launch_kernelparams" else b""
    with pytest.raises(HipError, match="function belongs to device 1"):
        getattr(runtime, method)(function, (1, 1, 1), (64, 1, 1), args)
    hip._hipModuleLaunchKernel.assert_not_called()
    assert not runtime._pending_args


def test_event_query_wait_destroy_use_owner_device(fake_hip) -> None:
    with device_scope(1):
        event = Runtime().event()
        event.record()
    assert not event.query()
    fake_hip.ready.add(event.handle.p)
    assert event.query()
    event.synchronize()
    event.destroy()
    assert ("event_query", 1, event.handle.p) in fake_hip.calls
    assert ("event_sync", 1, event.handle.p) in fake_hip.calls
    assert ("event_destroy", 1, event.handle.p) in fake_hip.calls
    assert fake_hip.device == 0


def test_event_record_rejects_wrong_device_before_hip(fake_hip) -> None:
    with device_scope(1):
        event = Runtime().event()
    with pytest.raises(HipError, match="event belongs to device 1"):
        event.record()
    hip._hipEventRecord.assert_not_called()


def test_elapsed_timing_rejects_cross_device_events(fake_hip) -> None:
    runtime = Runtime()
    start = runtime.event()
    with device_scope(1):
        end = runtime.event()
    with pytest.raises(ValueError, match="same device"):
        start.elapsed_to(end)
    hip._hipEventElapsedTime.assert_not_called()


def test_elapsed_timing_uses_same_device_and_restores_caller(fake_hip) -> None:
    with device_scope(1):
        start, end = Runtime().event(), Runtime().event()
    assert (
        start.elapsed_to(end) == 1.25
    )  # Synthetic value from the function-table double.
    assert fake_hip.device == 0
    assert ("event_elapsed", 1) in fake_hip.calls
