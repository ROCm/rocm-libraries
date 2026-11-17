# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Minimal ctypes wrapper over `libamdhip64.so` for the hipModule API.

This is the runtime twin of `comgr.py`: it takes the HSACO bytes that
comgr produced from our LLVM IR and runs the kernel via
`hipModuleLoadData` + `hipModuleLaunchKernel`. No host compilation,
no `<hip/hip_runtime.h>` parsing — the same code-object path AMDGPU
runtimes use for any pre-built fatbin or HSACO blob.

The wrapper stays torch-free and resolves HIP lazily:
- `get_device()` / `set_device()` / `device_scope()` select a host-thread device.
- `can_access_peer()` and Runtime peer methods establish directed access.
- `Runtime.alloc/free` and host/device copies operate on the current device.
- `Runtime.load_module()` records device ownership on modules and functions.
- `Runtime.launch()` retains arguments per device/stream until completion.
- `Runtime.event()` records event ownership and disallows cross-device timing.
"""

from __future__ import annotations

import ctypes
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from operator import index
from typing import Any, ClassVar

from ._ctypes_bind import _LazyFn
from .runtime_coexistence import _IS_WINDOWS, _add_dll_dir, _candidate_lib_paths

HIP_LAUNCH_PARAM_BUFFER_POINTER = ctypes.c_void_p(1)
HIP_LAUNCH_PARAM_BUFFER_SIZE = ctypes.c_void_p(2)
HIP_LAUNCH_PARAM_END = ctypes.c_void_p(3)

hipMemcpyHostToDevice = 1
hipMemcpyDeviceToHost = 2


class HipError(RuntimeError):
    pass


class _HipModuleHandle(ctypes.Structure):
    _fields_ = [("p", ctypes.c_void_p)]


class _HipFunctionHandle(ctypes.Structure):
    _fields_ = [("p", ctypes.c_void_p)]


class _HipEventHandle(ctypes.Structure):
    _fields_ = [("p", ctypes.c_void_p)]


def _load_lib() -> ctypes.CDLL:
    err = None
    for p in _candidate_lib_paths("amdhip64", "ROCKE_HIP_LIB", ["7"]):
        try:
            _add_dll_dir(p)
            return ctypes.CDLL(p)
        except OSError as e:
            err = e
    name = "amdhip64.dll" if _IS_WINDOWS else "libamdhip64.so"
    raise HipError(f"cannot load {name} ({err!r})")


# Holds the resolved ``libamdhip64`` handle. Stays ``None`` until the
# first HIP call actually goes through the runtime; this lets the user
# import rocke before importing torch (or vice versa) and still end
# up sharing torch's bundled HIP runtime instead of double-loading the
# system one.
_hip: ctypes.CDLL | None = None


def _resolve_hip() -> ctypes.CDLL:
    global _hip
    if _hip is None:
        _hip = _load_lib()
    return _hip


def _b(name: str, *argtypes, restype=ctypes.c_int) -> _LazyFn:
    return _LazyFn(name, list(argtypes), restype, _resolve_hip)


# HIP function table.
_hipGetErrorString = _b("hipGetErrorString", ctypes.c_int, restype=ctypes.c_char_p)
_hipInit = _b("hipInit", ctypes.c_uint)
_hipSetDevice = _b("hipSetDevice", ctypes.c_int)
_hipGetDevice = _b("hipGetDevice", ctypes.POINTER(ctypes.c_int))
_hipGetDeviceCount = _b("hipGetDeviceCount", ctypes.POINTER(ctypes.c_int))
_hipDeviceCanAccessPeer = _b(
    "hipDeviceCanAccessPeer", ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int
)
_hipDeviceEnablePeerAccess = _b(
    "hipDeviceEnablePeerAccess", ctypes.c_int, ctypes.c_uint
)
_hipDeviceDisablePeerAccess = _b("hipDeviceDisablePeerAccess", ctypes.c_int)
_hipModuleLoadData = _b(
    "hipModuleLoadData", ctypes.POINTER(_HipModuleHandle), ctypes.c_void_p
)
_hipModuleUnload = _b("hipModuleUnload", _HipModuleHandle)
_hipModuleGetFunction = _b(
    "hipModuleGetFunction",
    ctypes.POINTER(_HipFunctionHandle),
    _HipModuleHandle,
    ctypes.c_char_p,
)
_hipModuleLaunchKernel = _b(
    "hipModuleLaunchKernel",
    _HipFunctionHandle,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_void_p,  # sharedMemBytes, stream
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_void_p),
)
_hipMalloc = _b("hipMalloc", ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t)
_hipFree = _b("hipFree", ctypes.c_void_p)
_hipMemcpy = _b(
    "hipMemcpy", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
)
_hipMemset = _b("hipMemset", ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t)
_hipDeviceSynchronize = _b("hipDeviceSynchronize")
_hipStreamSynchronize = _b("hipStreamSynchronize", ctypes.c_void_p)
_hipEventCreate = _b("hipEventCreate", ctypes.POINTER(_HipEventHandle))
_hipEventDestroy = _b("hipEventDestroy", _HipEventHandle)
_hipEventRecord = _b("hipEventRecord", _HipEventHandle, ctypes.c_void_p)
_hipEventSynchronize = _b("hipEventSynchronize", _HipEventHandle)
_hipEventQuery = _b("hipEventQuery", _HipEventHandle)
_hipEventElapsedTime = _b(
    "hipEventElapsedTime",
    ctypes.POINTER(ctypes.c_float),
    _HipEventHandle,
    _HipEventHandle,
)


# hipError values relevant to event-based launch fencing. Kept here (not
# pulled from a header) so the runtime stays standard-library-only.
HIP_SUCCESS = 0
HIP_ERROR_NOT_READY = 600
HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704
HIP_ERROR_PEER_ACCESS_NOT_ENABLED = 705


def _check(s: int, where: str) -> None:
    if s != 0:
        msg = _hipGetErrorString(s)
        raise HipError(f"{where}: hipError({s}) {msg.decode() if msg else ''}")


_hip_inited = False
# Raw hipDeviceProp_t buffers, cached per device. The struct layout churns across
# ROCm releases (and the props symbol was versioned to ``...R0600`` in ROCm 6.x), so
# we keep the raw bytes and let each query read the field it needs rather than mirror
# the struct. See get_device_arch (gcnArchName) / get_device_name (name).
_device_props_cache: dict[int, bytes | None] = {}


def _ensure_hip_init() -> None:
    """Establish a HIP primary context exactly once.

    Without this, a fresh ctypes-only python process (no torch import,
    no prior HIP call) sees `hipModuleLoadData` return hipErrorNoDevice
    even though the GPU is visible to `rocminfo` — the runtime simply
    hasn't bound a device yet. Torch users get this for free as a side
    effect of `import torch`; the manifest-runner path doesn't.
    """
    global _hip_inited
    if _hip_inited:
        return
    _check(_hipInit(0), "hipInit")
    # Preserve a device the hosting process already selected (e.g. torch did
    # `hipSetDevice(N)`); only bind device 0 when no device is current yet.
    # hipGetDevice succeeds once a context exists, so a failure here means no
    # device is bound and we fall back to 0.
    cur = ctypes.c_int(-1)
    if _hipGetDevice(ctypes.byref(cur)) != HIP_SUCCESS or cur.value < 0:
        _check(_hipSetDevice(0), "hipSetDevice")
    _hip_inited = True


def _device_props(device: int = 0) -> bytes | None:
    """Raw ``hipDeviceProp_t`` bytes for a HIP device, or None if unavailable.

    Best-effort and **side-effect-free**: this is a *query*, not a context bind, so
    it deliberately does not call ``_ensure_hip_init()`` (which would ``hipSetDevice``
    and create a primary context). ``hipGetDeviceProperties*`` lazily inits the runtime
    internally and needs no bound context, so a pure ctypes process — no torch, no
    prior HIP call — still gets valid properties. Keeping it context-free means a probe
    can run before a later ``import torch`` without perturbing torch's device discovery.

    The struct layout changes across ROCm releases (the symbol was versioned to
    ``...R0600`` in ROCm 6.x), so we fill a generous zeroed buffer and let callers read
    the field they need rather than mirror the struct. The first properties symbol that
    returns success wins and its buffer is cached; we do not retry the legacy symbol once
    one has succeeded.
    """
    device = int(device)
    if device in _device_props_cache:
        return _device_props_cache[device]

    buf = ctypes.create_string_buffer(4096)
    for sym in ("hipGetDevicePropertiesR0600", "hipGetDeviceProperties"):
        fn = _b(sym, ctypes.c_void_p, ctypes.c_int)
        try:
            rc = fn(buf, device)
        except (AttributeError, OSError):
            continue
        if rc == 0:
            _device_props_cache[device] = buf.raw
            return buf.raw
    _device_props_cache[device] = None
    return None


def get_device_arch(device: int = 0) -> str | None:
    """Best-effort gfx string of a HIP device (e.g. ``"gfx942"``).

    Mirrors the ``Name`` field ``rocminfo`` prints for a GPU agent. Returns ``None``
    when it can't be determined (no GPU present, or the properties symbol is
    unavailable). Launch paths use this to compile for the device they will actually
    run on instead of defaulting to a fixed arch — building a gfx950 code object and
    launching it on gfx942 yields ``hipError(209) no kernel image``.

    ``gcnArchName`` carries the gfx token; the marketing ``name`` field (offset 0)
    contains no ``gfx`` token, so the first match in the raw buffer is the architecture
    name. The ``[0-9a-z]+`` class stops at the ``:`` feature-flag delimiter and the NUL
    terminator, yielding e.g. ``"gfx942"`` from ``"gfx942:sramecc+:xnack-"``.
    """
    import re

    raw = _device_props(device)
    if raw is None:
        return None
    m = re.search(rb"gfx[0-9a-z]+", raw)
    return m.group(0).decode("ascii") if m else None


def get_device_name(device: int = 0) -> str | None:
    """Marketing name of a HIP device — the string ``rocminfo`` labels "Marketing Name".

    Reads ``hipDeviceProp_t.name`` — the ``char name[256]`` at struct offset 0, which
    is stable across ROCm releases (unlike the churny ``gcnArchName`` offset). This is
    the same string ``rocminfo`` prints as "Marketing Name" and torch surfaces via
    ``torch.cuda.get_device_name``; reading it straight from HIP lets detection report
    the device without a torch dependency. Returns ``None`` when unavailable.
    """
    raw = _device_props(device)
    if raw is None:
        return None
    name = raw[:256].split(b"\0", 1)[0].decode("ascii", "replace")
    return name or None


def get_device_count() -> int:
    """Number of HIP devices visible to this process (respects ``HIP_VISIBLE_DEVICES``).

    Returns ``0`` when no device is visible or the runtime can't be queried — the count
    twin of ``get_device_arch`` returning ``None``. Best-effort and side-effect-free (no
    ``_ensure_hip_init``); ``hipGetDeviceCount`` lazily inits the runtime internally.
    A foundational HIP query, first consumed by the multi-GPU GEMM sweep test.
    """
    n = ctypes.c_int(0)
    try:
        rc = _hipGetDeviceCount(ctypes.byref(n))
    except (AttributeError, OSError):
        return 0
    return int(n.value) if rc == 0 else 0


def _device_ordinal(device: int) -> int:
    if isinstance(device, bool):
        raise TypeError("device ordinal must be a non-negative integer, not bool")
    ordinal = index(device)
    if not 0 <= ordinal <= 0x7FFFFFFF:
        raise ValueError(
            f"device ordinal must fit a non-negative HIP int, got {ordinal}"
        )
    return ordinal


def get_device() -> int:
    """Current HIP device for this host thread; unlike discovery, errors propagate."""
    _ensure_hip_init()
    device = ctypes.c_int(-1)
    _check(_hipGetDevice(ctypes.byref(device)), "hipGetDevice")
    return int(device.value)


def set_device(device: int) -> None:
    """Select a visible HIP ordinal on this host thread, without importing torch."""
    ordinal = _device_ordinal(device)
    _ensure_hip_init()
    _check(_hipSetDevice(ordinal), f"hipSetDevice({ordinal})")


@contextmanager
def device_scope(device: int) -> Iterator[None]:
    """Temporarily select a device, restoring the caller's selection on exit.

    HIP selection is host-thread-local. This scope does not synchronize devices
    or establish any ordering between their streams.
    """
    ordinal = _device_ordinal(device)
    previous = get_device()
    try:
        if ordinal != previous:
            set_device(ordinal)
        yield
    finally:
        if get_device() != previous:
            set_device(previous)


@contextmanager
def _resource_device_scope(device: int | None) -> Iterator[None]:
    if device is None:
        yield
    else:
        with device_scope(device):
            yield


def can_access_peer(device: int, peer_device: int) -> bool:
    """Query directed peer capability, not mapping state or execution correctness.

    A successful query returning False is distinct from a HIP failure. No device
    selection is changed, and no peer mapping is enabled by this function.
    """
    ordinal = _device_ordinal(device)
    peer = _device_ordinal(peer_device)
    capable = ctypes.c_int(0)
    _check(
        _hipDeviceCanAccessPeer(ctypes.byref(capable), ordinal, peer),
        f"hipDeviceCanAccessPeer({ordinal}, {peer})",
    )
    return bool(capable.value)


@dataclass
class Module:
    """Loaded code owned by a device; callers must drain uses before unloading."""

    handle: _HipModuleHandle
    device: int | None = None

    def get_function(self, name: str) -> _HipFunctionHandle:
        with _resource_device_scope(self.device):
            fn = _HipFunctionHandle()
            _check(
                _hipModuleGetFunction(
                    ctypes.byref(fn), self.handle, name.encode("utf-8")
                ),
                f"hipModuleGetFunction({name})",
            )
        fn._device = self.device
        return fn

    def unload(self) -> None:
        with _resource_device_scope(self.device):
            _check(_hipModuleUnload(self.handle), "hipModuleUnload")


@dataclass
class Event:
    """A completion event with its creation device, when created through Runtime."""

    handle: _HipEventHandle
    device: int | None = None

    def record(self, stream: int = 0) -> None:
        if self.device is not None and get_device() != self.device:
            raise HipError(
                f"event belongs to device {self.device}; select it before recording"
            )
        _check(_hipEventRecord(self.handle, ctypes.c_void_p(stream)), "hipEventRecord")

    def synchronize(self) -> None:
        with _resource_device_scope(self.device):
            _check(_hipEventSynchronize(self.handle), "hipEventSynchronize")

    def query(self) -> bool:
        """Non-blocking poll: return True iff the recorded work has completed.

        Returns ``hipSuccess`` -> True; ``hipErrorNotReady`` -> False; any
        other status raises :class:`HipError`. Used by
        :meth:`Runtime._reap_completed` to drop bucket entries whose
        kernels have finished without blocking.
        """
        with _resource_device_scope(self.device):
            s = _hipEventQuery(self.handle)
        if s == HIP_SUCCESS:
            return True
        if s == HIP_ERROR_NOT_READY:
            return False
        _check(s, "hipEventQuery")
        return False  # unreachable

    def elapsed_to(self, end: Event) -> float:
        if (
            self.device is not None
            and end.device is not None
            and self.device != end.device
        ):
            raise ValueError("elapsed timing requires events from the same device")
        ms = ctypes.c_float(0)
        with _resource_device_scope(self.device):
            _check(
                _hipEventElapsedTime(ctypes.byref(ms), self.handle, end.handle),
                "hipEventElapsedTime",
            )
        return float(ms.value)

    def destroy(self) -> None:
        with _resource_device_scope(self.device):
            _check(_hipEventDestroy(self.handle), "hipEventDestroy")


class Runtime:
    """HIP operations on the current device, with device-scoped launch retention.

    Load a module per device and keep raw allocations alive across all peer
    users. Drains affect only the selected device. Host orchestration must
    serialize bookkeeping for a given device/stream; selecting a different
    device does not wait for prior work.
    """

    # Per-device/stream FIFO of ``(refs_tuple, completion_event_or_None)``
    # entries: a backend-agnostic, stream-scoped **lifetime manager**.
    # Every async launch appends exactly one entry holding the ctypes
    # objects that launch enqueued; a caller may merge additional objects
    # to keep alive into the most-recent entry via
    # :meth:`retain_for_stream` so they share that launch's completion
    # event.
    #
    # This is NOT torch-specific, and must not migrate into a torch
    # module: the bucket only ever holds ctypes buffers and HIP
    # ``Event``s, :meth:`retain_for_stream` takes ``*objects: Any`` and
    # filters out ints/None, and ``Runtime`` never imports torch. Torch's
    # caching allocator (below) is the *motivating caller* of the retain
    # path, not something this runtime knows about.
    #
    # Why this exists
    # ---------------
    # Raw ``hipModuleLaunchKernel`` calls go through ctypes and are
    # invisible to Python's GC (and to torch's stream-aware caching
    # allocator, when torch owns the tensors). Two lifetimes must outlive
    # an in-flight kernel:
    #
    # 1. The ctypes args buffer, on the HIP_LAUNCH_PARAM_BUFFER_POINTER
    #    ("extra") path. That path does not promise to copy the
    #    packed-args buffer at enqueue time; observation on ROCm 6/7 is
    #    that the GPU command processor reads it later, when it actually
    #    starts the kernel. If the Python-owned buffer has been
    #    garbage-collected by then, the kernel reads stale memory and
    #    writes to whatever pointer those bytes now decode as. This
    #    applies to EVERY async launch -- numpy / manifest callers
    #    included -- so :meth:`launch` parks the buffer here
    #    unconditionally (torch-independent).
    #
    # 2. Any caller-owned object backing a kernel argument, retained via
    #    :meth:`retain_for_stream`. The motivating case: output /
    #    workspace tensors built with ``torch.empty(...)`` are tracked by
    #    torch's caching allocator against torch's *current* stream; once
    #    the Python reference drops, the allocator can recycle that memory
    #    while the raw HIP launch is still in flight, mutating the
    #    kernel's destination buffer. The launcher (the torch-aware layer)
    #    calls ``retain_for_stream`` for that; the mechanism here stays
    #    agnostic to what it holds.
    #
    # The mitigation in both cases is the same: tie the Python
    # references' lifetime to a HIP completion event recorded on the
    # same stream as the launch. Once the event has fired (queryable
    # without blocking via :meth:`Event.query`), it is safe to drop
    # every reference attached to that bucket entry.
    #
    # This mirrors CK Tile's ``stream_config`` + ``launch_kernel``
    # discipline (`include/ck_tile/host/stream_config.hpp`,
    # `include/ck_tile/host/kernel_launch.hpp`): every launch is paired
    # with a stream-bound synchronization primitive so the host never
    # observes a half-finished kernel. The Python analogue is
    # :meth:`_reap_completed` (eager non-blocking drain) and
    # :meth:`wait_stream` (event-blocking drain per stream).
    # Stream handles, particularly null stream 0, are not globally unique across
    # devices. A device drain must not release another device's in-flight buffers.
    _pending_args: ClassVar[
        dict[tuple[int, int], list[tuple[tuple[Any, ...], Event | None]]]
    ] = {}

    def enable_peer_access(self, peer_device: int) -> bool:
        """Enable current-device access to a peer; return whether we created it.

        An existing mapping is preserved and reported as False. Other HIP errors
        propagate. The caller must retain source allocations through all peer uses.
        """
        peer = _device_ordinal(peer_device)
        _ensure_hip_init()
        status = _hipDeviceEnablePeerAccess(peer, 0)
        if status == HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED:
            return False
        _check(status, f"hipDeviceEnablePeerAccess({peer})")
        return True

    def disable_peer_access(self, peer_device: int) -> bool:
        """Disable current-device access after all users are drained.

        Return False if no mapping exists; propagate all other HIP errors. This
        method does not synchronize pending work on either device.
        """
        peer = _device_ordinal(peer_device)
        _ensure_hip_init()
        status = _hipDeviceDisablePeerAccess(peer)
        if status == HIP_ERROR_PEER_ACCESS_NOT_ENABLED:
            return False
        _check(status, f"hipDeviceDisablePeerAccess({peer})")
        return True

    def load_module(self, blob: bytes) -> Module:
        device = get_device()
        buf = (ctypes.c_ubyte * len(blob)).from_buffer_copy(blob)
        handle = _HipModuleHandle()
        _check(
            _hipModuleLoadData(ctypes.byref(handle), ctypes.cast(buf, ctypes.c_void_p)),
            "hipModuleLoadData",
        )
        # Hold the buffer to keep memory alive.
        m = Module(handle, device=device)
        m._blob = buf  # type: ignore[attr-defined]
        return m

    def alloc(self, nbytes: int) -> int:
        p = ctypes.c_void_p(0)
        _check(_hipMalloc(ctypes.byref(p), nbytes), f"hipMalloc({nbytes})")
        return int(p.value)

    def free(self, ptr: int) -> None:
        _check(_hipFree(ctypes.c_void_p(ptr)), "hipFree")

    def memcpy_h2d(self, dst: int, src_buf: ctypes.Array, nbytes: int) -> None:
        _check(
            _hipMemcpy(ctypes.c_void_p(dst), src_buf, nbytes, hipMemcpyHostToDevice),
            "hipMemcpyH2D",
        )

    def memcpy_d2h(self, dst_buf: ctypes.Array, src: int, nbytes: int) -> None:
        _check(
            _hipMemcpy(dst_buf, ctypes.c_void_p(src), nbytes, hipMemcpyDeviceToHost),
            "hipMemcpyD2H",
        )

    def memset(self, ptr: int, value: int, nbytes: int) -> None:
        _check(_hipMemset(ctypes.c_void_p(ptr), value, nbytes), "hipMemset")

    def _reap_completed(self, stream: int, *, device: int | None = None) -> None:
        """Drop bucket entries whose completion events have fired.

        Non-blocking. Walks the FIFO from the head, destroying each
        event whose :meth:`Event.query` returns ``True`` and dropping
        its retained refs. Stops at the first un-fired event (FIFO
        ordering on the same stream guarantees nothing earlier in the
        queue is pending), or at the first entry with ``event is None``
        (a non-fenced legacy retain, only droppable by :meth:`sync`).
        """
        key = (get_device() if device is None else device, int(stream))
        bucket = self._pending_args.get(key)
        if not bucket:
            return
        while bucket:
            _refs, evt = bucket[0]
            if evt is None or not evt.query():
                break
            evt.destroy()
            bucket.pop(0)
        if not bucket:
            self._pending_args.pop(key, None)

    def stream_sync(self, stream: int) -> None:
        """``hipStreamSynchronize(stream)`` -- the cheap per-stream drain.

        On ROCm this typically costs <1 us when the stream is already
        idle. We use it for the synchronous-launch fast path
        (:meth:`launch_blocking` and ``LaunchConfig.fence=True``)
        instead of paying the ~40 us tax of a full
        ``hipEventCreate+Record+Synchronize+Destroy`` cycle just to
        wait on a single launch.

        Mirrors CK Tile's ``launch_kernel`` post-amble, which always
        ends with ``HIP_CHECK(hipStreamSynchronize(stream_config.stream_id_))``.
        """
        _check(
            _hipStreamSynchronize(ctypes.c_void_p(int(stream))), "hipStreamSynchronize"
        )

    def wait_stream(self, stream: int) -> None:
        """Per-stream drain + release all retained refs.

        Uses ``hipStreamSynchronize`` as the safe, cheap primitive that
        works on ROCm for raw ``hipModuleLaunchKernel`` work queued
        through ctypes. (``torch.cuda.synchronize()`` does not reliably
        drain that queue.) If the bucket also holds event-tagged
        entries from a prior async batch (e.g. inside
        :func:`time_launches`), every event will have fired by the
        time the stream-sync returns, and we destroy them as we drop
        the bucket.
        """
        s = int(stream)
        key = (get_device(), s)
        self.stream_sync(s)
        bucket = self._pending_args.pop(key, None)
        if not bucket:
            return
        for _refs, evt in bucket:
            if evt is not None:
                evt.destroy()

    def sync(self) -> None:
        """Drain the current device and release only its retained resources.

        Use :meth:`wait_stream` instead when the caller knows the
        target stream; this method is the broad hammer for benchmark
        harnesses that span multiple streams or want strong isolation
        between independent lanes (e.g. Triton 2D -> CK 2D -> Triton 3D
        -> CK 3D in the parity harness).
        """
        device = get_device()
        _check(_hipDeviceSynchronize(), "hipDeviceSynchronize")
        for key in list(self._pending_args):
            if key[0] != device:
                continue
            bucket = self._pending_args.pop(key)
            for _refs, evt in bucket:
                if evt is not None:
                    evt.destroy()

    def release_pending_for_stream(self, stream: int) -> None:
        """Drop refs held for ``stream`` after the caller has ensured
        the stream is drained (e.g. via :meth:`wait_stream`,
        :meth:`sync`, or an external event sync).

        Most callers should prefer :meth:`wait_stream`, which performs
        the wait *and* drops refs in one call. This method exists for
        callers that did their own sync via a different mechanism and
        only need the bookkeeping cleanup.
        """
        key = (get_device(), int(stream))
        bucket = self._pending_args.pop(key, None)
        if not bucket:
            return
        for _refs, evt in bucket:
            if evt is not None:
                evt.destroy()

    def retain_for_stream(self, stream: int, *objects: Any) -> None:
        """Keep ``objects`` alive until the most-recent launch on
        ``stream`` completes.

        Attaches to the most-recent bucket entry so the retained objects
        share the launch's HIP completion event. If there is no
        prior launch on ``stream``, parks the refs into a new entry
        with ``event=None`` (which only :meth:`sync` will release).

        Raw HIP launches issued through ctypes are invisible to
        Python's GC and mostly invisible to torch's stream-aware
        caching allocator: launcher code should call this for every
        tensor argument and workspace tensor it passes into a kernel.
        """
        keep = tuple(
            obj for obj in objects if obj is not None and not isinstance(obj, int)
        )
        if not keep:
            return
        key = (get_device(), int(stream))
        bucket = self._pending_args.setdefault(key, [])
        if bucket:
            refs, evt = bucket[-1]
            bucket[-1] = (refs + keep, evt)
        else:
            bucket.append((keep, None))

    def event(self) -> Event:
        device = get_device()
        h = _HipEventHandle()
        _check(_hipEventCreate(ctypes.byref(h)), "hipEventCreate")
        return Event(h, device=device)

    @staticmethod
    def _launch_key(fn: _HipFunctionHandle, stream: int) -> tuple[int, int]:
        device = get_device()
        owner = getattr(fn, "_device", None)
        if owner is not None and owner != device:
            raise HipError(
                f"function belongs to device {owner}, current device is {device}; "
                "load a module per device and select its owner before launching"
            )
        return device, int(stream)

    def _retain_launch(
        self, key: tuple[int, int], refs: tuple[Any, ...], record_event: bool
    ) -> Event | None:
        # Retain before creating/recording an event: either operation can fail
        # after the kernel was enqueued. A later successful drain is still needed.
        bucket = self._pending_args.setdefault(key, [])
        bucket.append((refs, None))
        if not record_event:
            return None
        event = self.event()
        try:
            event.record(stream=key[1])
        except BaseException:
            event.destroy()
            raise
        bucket[-1] = (refs, event)
        return event

    def launch(
        self,
        fn: _HipFunctionHandle,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args_packed: bytes,
        *,
        shared_bytes: int = 0,
        stream: int = 0,
        record_event: bool = False,
    ) -> Event | None:
        """Issue one kernel launch on ``stream`` (fire-and-forget).

        For *synchronous* launches that the host wants to fence on
        before reading outputs, use :meth:`launch_blocking` instead
        -- it pays only a single ``hipStreamSynchronize`` (~0.3 us)
        and never creates a HIP event.

        With ``record_event=False`` (default) no event is recorded.
        Bucket entries created without an event are released by
        :meth:`wait_stream` (which uses ``hipStreamSynchronize``) or
        by :meth:`sync` device-wide. This is the right setting for
        timed benchmark loops (see
        :func:`rocke.runtime.launcher.time_launches`), for the
        async path inside :class:`rocke.runtime.launcher.KernelLauncher`,
        and for the raw manifest runner.

        With ``record_event=True`` a HIP completion event is recorded
        on ``stream`` and stored alongside the args buffer. The
        returned :class:`Event` lets a caller wait on this specific
        launch and lets :meth:`_reap_completed` eagerly drop the
        bucket entry once the event has fired -- useful when the
        caller needs fine-grained per-launch observability rather
        than batch-wide stream drains. Costs ~1 us per launch on
        ROCm 7.
        """
        key = self._launch_key(fn, stream)
        s = key[1]
        # Eagerly reap only this device's stream.
        self._reap_completed(s, device=key[0])

        # Build the HIP "extra" array: [BUFFER_POINTER, &args, BUFFER_SIZE, &size, END].
        args_buf = (ctypes.c_ubyte * len(args_packed)).from_buffer_copy(args_packed)
        size_buf = ctypes.c_size_t(len(args_packed))
        extra = (ctypes.c_void_p * 5)(
            HIP_LAUNCH_PARAM_BUFFER_POINTER,
            ctypes.cast(args_buf, ctypes.c_void_p),
            HIP_LAUNCH_PARAM_BUFFER_SIZE,
            ctypes.cast(ctypes.pointer(size_buf), ctypes.c_void_p),
            HIP_LAUNCH_PARAM_END,
        )
        _check(
            _hipModuleLaunchKernel(
                fn,
                ctypes.c_uint(grid[0]),
                ctypes.c_uint(grid[1]),
                ctypes.c_uint(grid[2]),
                ctypes.c_uint(block[0]),
                ctypes.c_uint(block[1]),
                ctypes.c_uint(block[2]),
                ctypes.c_uint(shared_bytes),
                ctypes.c_void_p(s),
                None,
                extra,
            ),
            "hipModuleLaunchKernel",
        )

        return self._retain_launch(key, (args_buf, size_buf, extra), record_event)

    def launch_blocking(
        self,
        fn: _HipFunctionHandle,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args_packed: bytes,
        *,
        shared_bytes: int = 0,
        stream: int = 0,
    ) -> None:
        """Enqueue and drain the current device's stream, without creating events.

        A successful stream drain releases the packed arguments and any earlier
        retained resources. If synchronization fails after enqueue, references
        remain retained until a later successful drain instead of being freed
        while device work may still be using them.
        """
        self.launch(
            fn,
            grid,
            block,
            args_packed,
            shared_bytes=shared_bytes,
            stream=stream,
            record_event=False,
        )
        self.wait_stream(stream)

    def launch_kernelparams(
        self,
        fn: _HipFunctionHandle,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        ctypes_args: list,
        *,
        shared_bytes: int = 0,
        stream: int = 0,
        record_event: bool = True,
    ) -> Event | None:
        """Launch via the ``kernelParams`` path (an array of pointers to
        each parameter scalar) instead of the ``extra`` packed-buffer
        path. CUDA/HIP semantics guarantee ``kernelParams`` are *copied*
        into driver-owned memory at enqueue time, eliminating the
        host-buffer lifetime race the ``extra`` path is vulnerable to.
        See ``pack_args_kernelparams`` docstring for the full rationale.

        ``ctypes_args`` is a list of individual ``ctypes`` scalars (one
        per kernel argument, in declaration order), produced by
        ``pack_args_kernelparams``.

        Mirrors :meth:`launch`'s ``record_event`` contract: with
        ``record_event=True`` (default), a HIP completion event is
        recorded on ``stream`` and stored alongside the kept-alive
        params array in :attr:`_pending_args`. The retained objects
        are released by :meth:`_reap_completed` once the event fires.
        """
        key = self._launch_key(fn, stream)
        s = key[1]
        self._reap_completed(s, device=key[0])

        n = len(ctypes_args)
        # Build the void* params[] array. Keep both the per-arg
        # ctypes.pointer wrappers AND the underlying scalars alive in
        # `keep_alive` until hipModuleLaunchKernel has returned, which
        # is when the driver will have copied each parameter into its
        # own command-buffer memory.
        keep_alive = list(ctypes_args)
        ptrs = [ctypes.pointer(a) for a in ctypes_args]
        keep_alive.extend(ptrs)
        params_t = ctypes.c_void_p * n
        params = params_t(*[ctypes.cast(p, ctypes.c_void_p) for p in ptrs])
        _check(
            _hipModuleLaunchKernel(
                fn,
                ctypes.c_uint(grid[0]),
                ctypes.c_uint(grid[1]),
                ctypes.c_uint(grid[2]),
                ctypes.c_uint(block[0]),
                ctypes.c_uint(block[1]),
                ctypes.c_uint(block[2]),
                ctypes.c_uint(shared_bytes),
                ctypes.c_void_p(s),
                params,
                None,
            ),
            "hipModuleLaunchKernel(kernelParams)",
        )

        return self._retain_launch(key, (tuple(keep_alive), params), record_event)
