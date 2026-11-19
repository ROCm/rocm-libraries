# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Owner-push HIP streams and fences, without a collective-library dependency.

This is an additive eager API over the existing Runtime. Every asynchronous
command gets a separate unfenced retention entry *before* submission; an earlier
launch event cannot retire a later copy's storage. Entries retire only after a
successful explicit stream/device drain. Raw allocations and peer mappings must
still survive every consumer, not just their owner's transfer stream.

Cross-device waits establish stream dependencies, not a remote data-publication
proof. Atomic-capable storage, system-scoped publication and GPU protocol stress
remain separate requirements. No graph-capture or scheduling-progress claim is
made here. Bookkeeping must be serialized per device/stream by the host caller.
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from dataclasses import dataclass, field
from operator import index
from typing import Any

from . import hip_module as hip

_stream_create = hip._b(
    "hipStreamCreateWithFlags", ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint
)
_stream_destroy = hip._b("hipStreamDestroy", ctypes.c_void_p)
_stream_wait_event = hip._b(
    "hipStreamWaitEvent", ctypes.c_void_p, hip._HipEventHandle, ctypes.c_uint
)
_copy_async = hip._b(
    "hipMemcpyAsync",
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_void_p,
)
_copy_peer_async = hip._b(
    "hipMemcpyPeerAsync",
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_size_t,
    ctypes.c_void_p,
)


def _positive_native_integer(value: int, name: str, width: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    result = index(value)
    if not 0 < result < 1 << width:
        raise ValueError(f"{name} must be positive and fit {width} bits")
    return result


@dataclass
class PeerStream:
    """Owned stream; int conversion checks active device and handle lifetime.

    Destruction first drains its owner stream and is retryable after a failed
    drain or destroy. It neither drains other GPUs nor removes peer mappings.
    Use PeerAsyncRuntime.stream() to create one; there is no GC destructor.
    """

    handle: int
    device: int
    _closed: bool = field(default=False, init=False, repr=False, compare=False)

    def __int__(self) -> int:
        if self._closed:
            raise hip.HipError("peer stream has been destroyed")
        if hip.get_device() != self.device:
            raise hip.HipError(
                f"peer stream belongs to device {self.device}; select its owner"
            )
        return self.handle

    def synchronize(self) -> None:
        with hip.device_scope(self.device):
            hip.Runtime().wait_stream(int(self))

    def destroy(self) -> None:
        if self._closed:
            return
        with hip.device_scope(self.device):
            hip.Runtime().wait_stream(int(self))
            hip._check(
                _stream_destroy(ctypes.c_void_p(self.handle)), "hipStreamDestroy"
            )
            self._closed = True


@dataclass(frozen=True)
class _PeerEventWait:
    event: PeerEvent


@dataclass
class PeerEvent:
    """Standalone fence protected against reuse/destroy while waits are retained.

    Unlike Runtime launch-completion events this fence is never a retention FIFO
    retirement marker. All waiting streams must drain before it can be re-recorded
    or destroyed. Destroy synchronizes the recording event first, if recorded.
    """

    _event: hip.Event
    _recorded: bool = field(default=False, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def device(self) -> int:
        if self._event.device is None:
            raise hip.HipError("peer event has no device owner")
        return self._event.device

    def _ensure_live(self) -> None:
        if self._closed:
            raise hip.HipError("peer event has been destroyed")

    def _ensure_unwaited(self) -> None:
        for bucket in hip.Runtime._pending_args.values():
            for refs, _completion in bucket:
                if any(
                    isinstance(ref, _PeerEventWait) and ref.event is self
                    for ref in refs
                ):
                    raise hip.HipError(
                        "peer event has pending waits; drain every waiting stream"
                    )

    def record(self, stream: PeerStream) -> None:
        self._ensure_live()
        self._ensure_unwaited()
        if not isinstance(stream, PeerStream):
            raise TypeError("an owned PeerStream is required")
        if stream.device != self.device:
            raise hip.HipError("peer event and recording stream have different owners")
        value = int(stream)
        self._recorded = False
        # Retain the standalone fence even when record reports an asynchronous error.
        PeerAsyncRuntime._enqueue(stream, (self,), lambda: self._event.record(value))
        self._recorded = True

    def query(self) -> bool:
        self._ensure_live()
        if not self._recorded:
            raise hip.HipError("peer event has not been successfully recorded")
        return self._event.query()

    def destroy(self) -> None:
        if self._closed:
            return
        self._ensure_unwaited()
        with hip.device_scope(self.device):
            # Also synchronize after a failed record attempt: never infer that
            # an asynchronous error means no command was accepted.
            self._event.synchronize()
            hip._check(hip._hipEventDestroy(self._event.handle), "hipEventDestroy")
            self._closed = True


class PeerAsyncRuntime(hip.Runtime):
    """Additive owned-stream API. Legacy Runtime launch/copy defaults are unchanged.

    Kernel submissions may use stream=int(peer_stream) after selecting its owner;
    keep record_event=False and record standalone PeerEvents at protocol frontiers.
    Resources retained by this API retire at quiescence, not by event polling.
    """

    def stream(self, *, nonblocking: bool = True) -> PeerStream:
        if not isinstance(nonblocking, bool):
            raise TypeError("nonblocking must be bool")
        device = hip.get_device()
        handle = ctypes.c_void_p()
        hip._check(
            _stream_create(ctypes.byref(handle), int(nonblocking)),
            "hipStreamCreateWithFlags",
        )
        if not handle.value:
            raise hip.HipError("hipStreamCreateWithFlags returned a null stream")
        return PeerStream(int(handle.value), device)

    def peer_event(self) -> PeerEvent:
        return PeerEvent(self.event())

    @staticmethod
    def _enqueue(
        stream: PeerStream, owners: tuple[Any, ...], action: Callable[[], None]
    ) -> None:
        if not isinstance(stream, PeerStream):
            raise TypeError("an owned PeerStream is required")
        value = int(stream)
        refs = tuple(owners) + (stream,)
        key = (stream.device, value)
        # Never merge into an earlier event-fenced entry. Retain before enqueue,
        # including ambiguous HIP failures, until a later successful drain.
        hip.Runtime._pending_args.setdefault(key, []).append((refs, None))
        action()

    def stream_wait_event(self, stream: PeerStream, event: PeerEvent) -> None:
        if not isinstance(event, PeerEvent):
            raise TypeError("a PeerEvent is required")
        event._ensure_live()
        if not event._recorded:
            raise hip.HipError("peer event has not been successfully recorded")
        # HIP waits may cross creation devices; timing comparisons may not.
        self._enqueue(
            stream,
            (_PeerEventWait(event),),
            lambda: hip._check(
                _stream_wait_event(
                    ctypes.c_void_p(int(stream)), event._event.handle, 0
                ),
                "hipStreamWaitEvent",
            ),
        )

    def memcpy_peer_async(
        self,
        dst: int,
        dst_device: int,
        src: int,
        src_device: int,
        nbytes: int,
        *,
        stream: PeerStream,
        owners: tuple[Any, ...] = (),
    ) -> None:
        """Push bytes on the source owner's stream, including the owner's local copy.

        Peer access must already be established. Raw malloc/free lifetimes remain
        caller-managed; supply owners for objects owning storage. Retained owners
        are not an acknowledgement that every GEMM consumer finished using data.
        """
        source_device = hip._device_ordinal(src_device)
        destination_device = hip._device_ordinal(dst_device)
        if not isinstance(stream, PeerStream):
            raise TypeError("an owned PeerStream is required")
        if stream.device != source_device:
            raise hip.HipError(
                "owner-push copy stream must belong to the source device"
            )
        width = ctypes.sizeof(ctypes.c_void_p) * 8
        source = _positive_native_integer(src, "source pointer", width)
        destination = _positive_native_integer(dst, "destination pointer", width)
        size = _positive_native_integer(
            nbytes, "copy size", ctypes.sizeof(ctypes.c_size_t) * 8
        )
        value = int(stream)

        def enqueue() -> None:
            if source_device == destination_device:
                status = _copy_async(
                    ctypes.c_void_p(destination),
                    ctypes.c_void_p(source),
                    size,
                    3,
                    ctypes.c_void_p(value),  # hipMemcpyDeviceToDevice
                )
                where = "hipMemcpyAsync(D2D)"
            else:
                status = _copy_peer_async(
                    ctypes.c_void_p(destination),
                    destination_device,
                    ctypes.c_void_p(source),
                    source_device,
                    size,
                    ctypes.c_void_p(value),
                )
                where = "hipMemcpyPeerAsync"
            hip._check(status, where)

        self._enqueue(stream, owners, enqueue)
