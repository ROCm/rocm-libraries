# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Minimal ctypes wrapper over `libamdhip64.so` for the hipModule API.

This is the runtime twin of `comgr.py`: it takes the HSACO bytes that
comgr produced from our LLVM IR and runs the kernel via
`hipModuleLoadData` + `hipModuleLaunchKernel`. No host compilation,
no `<hip/hip_runtime.h>` parsing — the same code-object path AMDGPU
runtimes use for any pre-built fatbin or HSACO blob.

We expose only what the GEMM kernel needs:
- `Runtime()` opens the library and caches function pointers.
- `Runtime.alloc(nbytes)` / `Runtime.free(ptr)` / `Runtime.memcpy(...)`.
- `Runtime.load_module(blob)` returns a `Module` with `get_function`.
- `Module.launch(fn, grid, block, args_bytes)` issues a launch.
- `Runtime.event()` / `Event.record()` / `Event.elapsed_to(other)` for timing.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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
_hip: Optional[ctypes.CDLL] = None


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
_hipDeviceGetAttribute = _b(
    "hipDeviceGetAttribute", ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int
)
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


def _check(s: int, where: str) -> None:
    if s != 0:
        msg = _hipGetErrorString(s)
        raise HipError(f"{where}: hipError({s}) {msg.decode() if msg else ''}")


_hip_inited = False
# Raw hipDeviceProp_t buffers, cached per device. The struct layout churns across
# ROCm releases (and the props symbol was versioned to ``...R0600`` in ROCm 6.x), so
# we keep the raw bytes and let each query read the field it needs rather than mirror
# the struct. See get_device_arch (gcnArchName) / get_device_name (name).
_device_props_cache: Dict[int, Optional[bytes]] = {}


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


def _device_props(device: int = 0) -> Optional[bytes]:
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


def get_device_arch(device: int = 0) -> Optional[str]:
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


def get_device_name(device: int = 0) -> Optional[str]:
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


# hipDeviceAttributeMultiprocessorCount. Part of the stable ``hipDeviceAttribute_t``
# ABI enum (AMD preserves numeric positions with ``...Unused`` placeholders), so this
# is far more durable than reading a field offset out of the churny hipDeviceProp_t.
_HIP_ATTR_MULTIPROCESSOR_COUNT = 63


def get_device_num_cus(device: int = 0) -> Optional[int]:
    """CU (multiprocessor) count of a HIP device, or None if unqueryable.

    Torch-free ctypes twin of ``get_device_arch`` / ``get_device_name``: uses
    ``hipDeviceGetAttribute`` (single ``int``, no struct-offset guessing) so the
    dispatch/benchmark layers can size split-KV device subscription without a torch
    dependency. Best-effort and side-effect-free (``hipDeviceGetAttribute`` lazily
    inits the runtime internally). NOTE: this reports CUs on CU-mode devices (CDNA);
    a WGP-mode (RDNA) device reports half the CU count — irrelevant to the current
    gfx942-only caller.
    """
    v = ctypes.c_int(0)
    try:
        rc = _hipDeviceGetAttribute(
            ctypes.byref(v), _HIP_ATTR_MULTIPROCESSOR_COUNT, int(device)
        )
    except (AttributeError, OSError):
        return None
    return int(v.value) if rc == 0 and v.value > 0 else None


@dataclass
class Module:
    handle: _HipModuleHandle

    def get_function(self, name: str) -> _HipFunctionHandle:
        fn = _HipFunctionHandle()
        _check(
            _hipModuleGetFunction(ctypes.byref(fn), self.handle, name.encode("utf-8")),
            f"hipModuleGetFunction({name})",
        )
        return fn

    def unload(self) -> None:
        _check(_hipModuleUnload(self.handle), "hipModuleUnload")


@dataclass
class Event:
    handle: _HipEventHandle

    def record(self, stream: int = 0) -> None:
        _check(_hipEventRecord(self.handle, ctypes.c_void_p(stream)), "hipEventRecord")

    def synchronize(self) -> None:
        _check(_hipEventSynchronize(self.handle), "hipEventSynchronize")

    def query(self) -> bool:
        """Non-blocking poll: return True iff the recorded work has completed.

        Returns ``hipSuccess`` -> True; ``hipErrorNotReady`` -> False; any
        other status raises :class:`HipError`. Used by
        :meth:`Runtime._reap_completed` to drop bucket entries whose
        kernels have finished without blocking.
        """
        s = _hipEventQuery(self.handle)
        if s == HIP_SUCCESS:
            return True
        if s == HIP_ERROR_NOT_READY:
            return False
        _check(s, "hipEventQuery")
        return False  # unreachable

    def elapsed_to(self, end: "Event") -> float:
        ms = ctypes.c_float(0)
        _check(
            _hipEventElapsedTime(ctypes.byref(ms), self.handle, end.handle),
            "hipEventElapsedTime",
        )
        return float(ms.value)

    def destroy(self) -> None:
        _check(_hipEventDestroy(self.handle), "hipEventDestroy")


class Runtime:
    # Per-stream FIFO of ``(refs_tuple, completion_event_or_None)``
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
    _pending_args: "Dict[int, List[Tuple[Tuple[Any, ...], Optional[Event]]]]" = {}

    def load_module(self, blob: bytes) -> Module:
        _ensure_hip_init()
        buf = (ctypes.c_ubyte * len(blob)).from_buffer_copy(blob)
        handle = _HipModuleHandle()
        _check(
            _hipModuleLoadData(ctypes.byref(handle), ctypes.cast(buf, ctypes.c_void_p)),
            "hipModuleLoadData",
        )
        # Hold the buffer to keep memory alive.
        m = Module(handle)
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

    def _reap_completed(self, stream: int) -> None:
        """Drop bucket entries whose completion events have fired.

        Non-blocking. Walks the FIFO from the head, destroying each
        event whose :meth:`Event.query` returns ``True`` and dropping
        its retained refs. Stops at the first un-fired event (FIFO
        ordering on the same stream guarantees nothing earlier in the
        queue is pending), or at the first entry with ``event is None``
        (a non-fenced legacy retain, only droppable by :meth:`sync`).
        """
        s = int(stream)
        bucket = self._pending_args.get(s)
        if not bucket:
            return
        while bucket:
            _refs, evt = bucket[0]
            if evt is None or not evt.query():
                break
            evt.destroy()
            bucket.pop(0)
        if not bucket:
            self._pending_args.pop(s, None)

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
        self.stream_sync(s)
        bucket = self._pending_args.pop(s, None)
        if not bucket:
            return
        for _refs, evt in bucket:
            if evt is not None:
                evt.destroy()

    def sync(self) -> None:
        """Device-wide drain: ``hipDeviceSynchronize`` + release everything.

        Use :meth:`wait_stream` instead when the caller knows the
        target stream; this method is the broad hammer for benchmark
        harnesses that span multiple streams or want strong isolation
        between independent lanes (e.g. Triton 2D -> CK 2D -> Triton 3D
        -> CK 3D in the parity harness).
        """
        _check(_hipDeviceSynchronize(), "hipDeviceSynchronize")
        for stream_id in list(self._pending_args.keys()):
            for _refs, evt in self._pending_args[stream_id]:
                if evt is not None:
                    evt.destroy()
        self._pending_args.clear()

    def release_pending_for_stream(self, stream: int) -> None:
        """Drop refs held for ``stream`` after the caller has ensured
        the stream is drained (e.g. via :meth:`wait_stream`,
        :meth:`sync`, or an external event sync).

        Most callers should prefer :meth:`wait_stream`, which performs
        the wait *and* drops refs in one call. This method exists for
        callers that did their own sync via a different mechanism and
        only need the bookkeeping cleanup.
        """
        s = int(stream)
        bucket = self._pending_args.pop(s, None)
        if not bucket:
            return
        for _refs, evt in bucket:
            if evt is not None:
                evt.destroy()

    def retain_for_stream(self, stream: int, *objects: Any) -> None:
        """Keep ``objects`` alive until the most-recent launch on
        ``stream`` completes.

        Attaches to the head bucket entry so the retained objects
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
        s = int(stream)
        bucket = self._pending_args.setdefault(s, [])
        if bucket:
            refs, evt = bucket[-1]
            bucket[-1] = (refs + keep, evt)
        else:
            bucket.append((keep, None))

    def event(self) -> Event:
        h = _HipEventHandle()
        _check(_hipEventCreate(ctypes.byref(h)), "hipEventCreate")
        return Event(h)

    def launch(
        self,
        fn: _HipFunctionHandle,
        grid: Tuple[int, int, int],
        block: Tuple[int, int, int],
        args_packed: bytes,
        *,
        shared_bytes: int = 0,
        stream: int = 0,
        record_event: bool = False,
    ) -> "Optional[Event]":
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
        s = int(stream)
        # Eagerly reap any prior launches on this stream that have
        # already completed. Keeps the bucket bounded in steady state.
        self._reap_completed(s)

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

        evt: Optional[Event] = None
        if record_event:
            evt = self.event()
            evt.record(stream=s)

        # Hold refs (args_buf MUST outlive the kernel for the "extra"
        # path) alongside the completion event. ``retain_for_stream``
        # appends tensors to the same entry.
        bucket = self._pending_args.setdefault(s, [])
        bucket.append(((args_buf, size_buf, extra), evt))
        return evt

    def launch_blocking(
        self,
        fn: _HipFunctionHandle,
        grid: Tuple[int, int, int],
        block: Tuple[int, int, int],
        args_packed: bytes,
        *,
        shared_bytes: int = 0,
        stream: int = 0,
    ) -> None:
        """Synchronous launch: enqueue, then ``hipStreamSynchronize``.

        This is the fast path for the default ``LaunchConfig.fence=True``
        contract. Unlike :meth:`launch`, no HIP event is created /
        recorded / destroyed -- a single ``hipStreamSynchronize`` is
        sufficient to (a) wait for the kernel and (b) guarantee the
        GPU command processor has finished reading the packed-args
        buffer. By the time this method returns, the args buffer can
        be safely freed by the Python frame's normal cleanup; no
        :attr:`_pending_args` bookkeeping is needed for the launch.

        Empirical cost on ROCm 7 / MI355X: ~0.3 us per call vs
        ~43 us for the event-based fence. Mirrors CK Tile's
        ``launch_kernel`` post-amble (``hipStreamSynchronize(stream_id_)``).
        """
        s = int(stream)
        # Eagerly reap any prior async launches that have completed.
        # Cheap (~0.1 us when the bucket is empty); keeps the bucket
        # from growing if the caller mixes fenced and unfenced launches.
        self._reap_completed(s)

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
        # Single ``hipStreamSynchronize`` is both the kernel-completion
        # wait and the args-buffer-drain barrier. After it returns,
        # ``args_buf``/``size_buf``/``extra`` can be dropped by Python's
        # frame cleanup -- the GPU is no longer reading them.
        _check(_hipStreamSynchronize(ctypes.c_void_p(s)), "hipStreamSynchronize")

    def launch_kernelparams(
        self,
        fn: _HipFunctionHandle,
        grid: Tuple[int, int, int],
        block: Tuple[int, int, int],
        ctypes_args: list,
        *,
        shared_bytes: int = 0,
        stream: int = 0,
        record_event: bool = True,
    ) -> "Optional[Event]":
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
        s = int(stream)
        self._reap_completed(s)

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

        evt: Optional[Event] = None
        if record_event:
            evt = self.event()
            evt.record(stream=s)

        # Belt-and-suspenders: keep keep_alive + params alive until the
        # completion event has fired, even though the driver should
        # have copied each parameter at enqueue.
        bucket = self._pending_args.setdefault(s, [])
        bucket.append(((tuple(keep_alive), params), evt))
        return evt
