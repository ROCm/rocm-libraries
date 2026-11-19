# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only fake-HIP tests for owner-push streams, fences and command lifetimes."""

from __future__ import annotations

import ctypes
import gc
import weakref
from unittest.mock import Mock

import pytest
from rocke.runtime import PeerAsyncRuntime, PeerEvent, PeerStream
from rocke.runtime import async_peer as peer
from rocke.runtime import hip_module as hip


@pytest.fixture
def async_hip(fake_hip, monkeypatch):
    handles = {}

    def create(pointer, flags):
        status = fake_hip.status("stream_create", fake_hip.device, flags)
        if status == 0:
            # Reuse handle values across devices to exercise device/stream keys.
            value = handles.get(fake_hip.device, 10) + 1
            handles[fake_hip.device] = value
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p)).contents.value = value
        return status

    functions = {
        "_stream_create": create,
        "_stream_destroy": lambda stream: fake_hip.status(
            "stream_destroy", fake_hip.device, stream.value
        ),
        "_stream_wait_event": lambda stream, event, flags: fake_hip.status(
            "stream_wait", fake_hip.device, stream.value, event.p, flags
        ),
        "_copy_async": lambda dst, src, size, kind, stream: fake_hip.status(
            "local_copy",
            fake_hip.device,
            dst.value,
            src.value,
            size,
            kind,
            stream.value,
        ),
        "_copy_peer_async": lambda dst, dd, src, sd, size, stream: fake_hip.status(
            "peer_copy",
            fake_hip.device,
            dst.value,
            dd,
            src.value,
            sd,
            size,
            stream.value,
        ),
    }
    for name, function in functions.items():
        monkeypatch.setattr(peer, name, Mock(side_effect=function))
    return fake_hip


def test_owned_stream_defaults_and_legacy_launch(async_hip):
    runtime = PeerAsyncRuntime()
    stream = runtime.stream()
    assert isinstance(stream, PeerStream)
    assert stream.device == 0
    assert ("stream_create", 0, 1) in async_hip.calls
    module = runtime.load_module(b"fake hsaco")
    fn = module.get_function("entry")
    runtime.launch(fn, (1, 1, 1), (64, 1, 1), b"args", stream=int(stream))
    assert (0, stream.handle) in runtime._pending_args
    stream.synchronize()
    assert runtime._pending_args == {}
    stream.destroy()
    stream.destroy()
    peer._stream_destroy.assert_called_once()


def test_blocking_stream_is_explicit(async_hip):
    stream = PeerAsyncRuntime().stream(nonblocking=False)
    assert ("stream_create", 0, 0) in async_hip.calls
    stream.destroy()


def test_wrong_device_and_closed_stream_rejected(async_hip):
    runtime = PeerAsyncRuntime()
    stream = runtime.stream()
    with hip.device_scope(1), pytest.raises(hip.HipError, match="belongs to device"):
        runtime.memcpy_peer_async(200, 1, 100, 0, 16, stream=stream)
    peer._copy_peer_async.assert_not_called()
    assert runtime._pending_args == {}
    stream.destroy()
    with pytest.raises(hip.HipError, match="destroyed"):
        int(stream)


def test_destroy_selects_owner_and_failed_drain_is_retryable(async_hip):
    runtime = PeerAsyncRuntime()
    stream = runtime.stream()
    owner = object()
    runtime.memcpy_peer_async(200, 1, 100, 0, 16, stream=stream, owners=(owner,))
    async_hip.errors[("stream_sync", 0, stream.handle)] = 77
    with hip.device_scope(1):
        with pytest.raises(hip.HipError, match="hipStreamSynchronize"):
            stream.destroy()
        assert hip.get_device() == 1
        assert (0, stream.handle) in runtime._pending_args
        peer._stream_destroy.assert_not_called()
        del async_hip.errors[("stream_sync", 0, stream.handle)]
        stream.destroy()
        assert hip.get_device() == 1
    assert runtime._pending_args == {}


def test_failed_stream_destroy_retains_handle_for_retry(async_hip):
    stream = PeerAsyncRuntime().stream()
    async_hip.errors[("stream_destroy", 0, stream.handle)] = 77
    with pytest.raises(hip.HipError, match="hipStreamDestroy"):
        stream.destroy()
    assert int(stream) == stream.handle
    del async_hip.errors[("stream_destroy", 0, stream.handle)]
    stream.destroy()


def test_copy_orders_and_local_panel_are_explicit(async_hip):
    runtime = PeerAsyncRuntime()
    stream = runtime.stream()
    runtime.memcpy_peer_async(200, 1, 100, 0, 16, stream=stream)
    runtime.memcpy_peer_async(300, 0, 100, 0, 16, stream=stream)
    assert ("peer_copy", 0, 200, 1, 100, 0, 16, stream.handle) in async_hip.calls
    assert ("local_copy", 0, 300, 100, 16, 3, stream.handle) in async_hip.calls
    assert len(runtime._pending_args[(0, stream.handle)]) == 2
    assert all(
        event is None for _refs, event in runtime._pending_args[(0, stream.handle)]
    )
    stream.destroy()


def test_owner_retention_does_not_attach_to_prior_fence(async_hip):
    class Owner:
        pass

    runtime = PeerAsyncRuntime()
    stream = runtime.stream()
    earlier = runtime.event()
    earlier.record(int(stream))
    runtime._pending_args[(0, stream.handle)] = [(("earlier",), earlier)]
    owner = Owner()
    reference = weakref.ref(owner)
    runtime.memcpy_peer_async(200, 1, 100, 0, 16, stream=stream, owners=(owner,))
    del owner
    async_hip.ready.add(earlier.handle.p)
    runtime._reap_completed(int(stream))
    gc.collect()
    assert reference() is not None
    assert len(runtime._pending_args[(0, stream.handle)]) == 1
    stream.synchronize()
    gc.collect()
    assert reference() is None
    stream.destroy()


def test_copy_failure_retains_owners_until_successful_drain(async_hip):
    runtime = PeerAsyncRuntime()
    stream = runtime.stream()
    owner = object()
    async_hip.errors[("peer_copy", 0, 200, 1, 100, 0, 16, stream.handle)] = 77
    with pytest.raises(hip.HipError, match="hipMemcpyPeerAsync"):
        runtime.memcpy_peer_async(200, 1, 100, 0, 16, stream=stream, owners=(owner,))
    refs, event = runtime._pending_args[(0, stream.handle)][0]
    assert owner in refs and event is None
    stream.destroy()


def test_same_handle_different_devices_are_isolated(async_hip):
    runtime = PeerAsyncRuntime()
    stream0 = runtime.stream()
    runtime.memcpy_peer_async(200, 1, 100, 0, 16, stream=stream0)
    with hip.device_scope(1):
        stream1 = runtime.stream()
        assert stream1.handle == stream0.handle
        runtime.memcpy_peer_async(300, 0, 400, 1, 16, stream=stream1)
    stream0.destroy()
    assert (1, stream1.handle) in runtime._pending_args
    stream1.destroy()
    assert runtime._pending_args == {}


def test_cross_device_wait_requires_successful_record_and_all_waiter_drains(async_hip):
    runtime = PeerAsyncRuntime()
    source = runtime.stream()
    event = runtime.peer_event()
    assert isinstance(event, PeerEvent)
    with hip.device_scope(1):
        consumer1 = runtime.stream()
        with pytest.raises(hip.HipError, match="successfully recorded"):
            runtime.stream_wait_event(consumer1, event)
    event.record(source)
    async_hip.ready.add(event._event.handle.p)
    assert event.query()
    with hip.device_scope(1):
        runtime.stream_wait_event(consumer1, event)
    with hip.device_scope(2):
        consumer2 = runtime.stream()
        runtime.stream_wait_event(consumer2, event)
    source.synchronize()  # does not retire waits on other GPUs
    with pytest.raises(hip.HipError, match="pending waits"):
        event.record(source)
    with pytest.raises(hip.HipError, match="pending waits"):
        event.destroy()
    consumer1.synchronize()
    with pytest.raises(hip.HipError, match="pending waits"):
        event.destroy()
    consumer2.synchronize()
    event.record(source)
    event.destroy()
    event.destroy()
    with pytest.raises(hip.HipError, match="destroyed"):
        runtime.stream_wait_event(source, event)
    source.destroy()
    consumer1.destroy()
    consumer2.destroy()


def test_failed_wait_blocks_event_reuse_until_drain(async_hip):
    runtime = PeerAsyncRuntime()
    source = runtime.stream()
    event = runtime.peer_event()
    event.record(source)
    with hip.device_scope(1):
        consumer = runtime.stream()
        async_hip.errors[
            ("stream_wait", 1, consumer.handle, event._event.handle.p, 0)
        ] = 77
        with pytest.raises(hip.HipError, match="hipStreamWaitEvent"):
            runtime.stream_wait_event(consumer, event)
    with pytest.raises(hip.HipError, match="pending waits"):
        event.destroy()
    consumer.destroy()
    event.destroy()
    source.destroy()


def test_record_and_event_destroy_failures_are_retryable(async_hip):
    runtime = PeerAsyncRuntime()
    source = runtime.stream()
    event = runtime.peer_event()
    handle = event._event.handle.p
    async_hip.errors[("event_record", 0, handle, source.handle)] = 77
    with pytest.raises(hip.HipError, match="hipEventRecord"):
        event.record(source)
    with pytest.raises(hip.HipError, match="successfully recorded"):
        runtime.stream_wait_event(source, event)
    assert (0, source.handle) in runtime._pending_args
    del async_hip.errors[("event_record", 0, handle, source.handle)]
    event.record(source)
    async_hip.errors[("event_destroy", 0, handle)] = 77
    with pytest.raises(hip.HipError, match="hipEventDestroy"):
        event.destroy()
    del async_hip.errors[("event_destroy", 0, handle)]
    event.destroy()
    source.destroy()


@pytest.mark.parametrize(
    "src, dst, size",
    [(0, 200, 16), (100, 0, 16), (100, 200, 0), (-1, 200, 16), (100, 200, 1 << 64)],
)
def test_invalid_copy_ranges_never_enqueue(async_hip, src, dst, size):
    runtime = PeerAsyncRuntime()
    stream = runtime.stream()
    with pytest.raises(ValueError):
        runtime.memcpy_peer_async(dst, 1, src, 0, size, stream=stream)
    assert runtime._pending_args == {}
    peer._copy_peer_async.assert_not_called()
    stream.destroy()


def test_invalid_copy_owner_never_enqueue(async_hip):
    runtime = PeerAsyncRuntime()
    stream = runtime.stream()
    with pytest.raises(hip.HipError, match="source device"):
        runtime.memcpy_peer_async(200, 0, 100, 1, 16, stream=stream)
    assert runtime._pending_args == {}
    peer._copy_peer_async.assert_not_called()
    stream.destroy()
