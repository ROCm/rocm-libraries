# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Minimal ctypes wrapper over `libamdhip64.so` for the hipModule API.

This is the runtime twin of `_comgr.py`: it takes the HSACO bytes that
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
import glob
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


_IS_WINDOWS = sys.platform == "win32"


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


def _torch_bundled_lib(stem: str) -> Optional[str]:
    """Return path to ``<torch>/lib/lib<stem>.so`` if torch is in this process.

    Newer PyTorch+ROCm wheels (e.g. torch>=2.12 / ROCm 7.2) ship their
    own ``libamdhip64.so`` and ``libamd_comgr.so`` inside the wheel's
    ``torch/lib/`` directory. When torch is imported, those bundled
    libraries get loaded into the process and own torch's HIP context.
    A second copy of HIP loaded by rocke from ``/opt/rocm/lib`` is a
    *different* runtime instance with disjoint state — modules loaded
    via one are invisible to ``hipModuleGetFunction`` from the other,
    surfacing as ``hipError(500) named symbol not found`` even when the
    HSACO is well-formed and the symbol is present in its ELF.

    To keep both halves of the process talking to the same HIP/comgr
    runtime, prefer torch's bundled lib when torch is already imported.
    Avoids importing torch as a side effect: only honors a torch that
    is *already* in :data:`sys.modules`.
    """
    torch_mod = sys.modules.get("torch")
    if torch_mod is None:
        return None
    torch_file = getattr(torch_mod, "__file__", None)
    if not torch_file:
        return None
    libdir = os.path.join(os.path.dirname(torch_file), "lib")
    if _IS_WINDOWS:
        # ROCm-for-Windows torch wheels (TheRock / AMD nightlies) bundle
        # ``amdhip64.dll`` and a version-stamped ``amd_comgr*.dll`` (no
        # ``lib`` prefix). Prefer an exact match, else glob the versioned
        # comgr name.
        direct = os.path.join(libdir, f"{stem}.dll")
        if os.path.exists(direct):
            return direct
        matches = sorted(glob.glob(os.path.join(libdir, f"{stem}*.dll")))
        return matches[0] if matches else None
    candidate = os.path.join(libdir, f"lib{stem}.so")
    return candidate if os.path.exists(candidate) else None


def _rocm_sdk_dll(stem: str) -> Optional[str]:
    """Locate a ROCm runtime DLL shipped by the ``rocm-sdk-core`` wheel.

    ROCm-for-Windows torch nightlies (AMD's gfx1151 index / TheRock) put
    the HIP runtime and comgr in ``_rocm_sdk_core/bin`` with a version
    suffix (e.g. ``amdhip64_7.dll``, ``amd_comgr0702.dll``) rather than in
    ``torch/lib``. Returns the first match for ``<stem>*.dll`` or None.
    """
    if not _IS_WINDOWS:
        return None
    try:
        import importlib.util

        spec = importlib.util.find_spec("_rocm_sdk_core")
    except Exception:
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    for loc in spec.submodule_search_locations:
        bindir = os.path.join(loc, "bin")
        direct = os.path.join(bindir, f"{stem}.dll")
        if os.path.exists(direct):
            return direct
        matches = sorted(glob.glob(os.path.join(bindir, f"{stem}*.dll")))
        if matches:
            return matches[0]
    return None


def _candidate_lib_paths(stem: str, env_var: str, sonames: List[str]) -> List[str]:
    """Resolution order for the HIP runtime / COMGR shared libraries.

    Order:
      1. ``$ROCKE_HIP_LIB`` (explicit override, full path).
      2. ``<torch>/lib/lib<stem>.so`` if torch is already imported.
      3. ``/opt/rocm/lib/lib<stem>.so`` and the requested SONAME variants.
      4. Bare ``lib<stem>.so`` for the dynamic linker's search path.
    """
    paths: List[str] = []
    override = os.environ.get(env_var)
    if override:
        paths.append(override)
    bundled = _torch_bundled_lib(stem)
    if bundled is not None:
        paths.append(bundled)
    sdk = _rocm_sdk_dll(stem)
    if sdk is not None:
        paths.append(sdk)
    if _IS_WINDOWS:
        # ROCm-for-Windows / HIP SDK install: ``%HIP_PATH%\bin`` then the
        # bare DLL name (resolved via the default DLL search path). The
        # comgr DLL carries a version suffix, so glob it.
        for root_env in ("HIP_PATH", "ROCM_PATH"):
            root = os.environ.get(root_env)
            if not root:
                continue
            bindir = os.path.join(root, "bin")
            paths.append(os.path.join(bindir, f"{stem}.dll"))
            paths.extend(sorted(glob.glob(os.path.join(bindir, f"{stem}*.dll"))))
        paths.append(f"{stem}.dll")
        return paths
    paths.append(f"/opt/rocm/lib/lib{stem}.so")
    for soname in sonames:
        paths.append(f"/opt/rocm/lib/lib{stem}.so.{soname}")
    paths.append(f"lib{stem}.so")
    return paths


def _add_dll_dir(path: str) -> None:
    """On Windows, register a resolved DLL's own directory so its
    dependent DLLs (bundled alongside it in ``torch/lib`` or the HIP SDK
    ``bin``) are found by the loader. No-op off Windows or for bare names.
    """
    if not _IS_WINDOWS:
        return
    d = os.path.dirname(path)
    if d and os.path.isdir(d):
        try:
            os.add_dll_directory(d)
        except (OSError, AttributeError):
            pass


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


class _LazyFn:
    """Lazy ctypes function wrapper.

    Defers ``getattr`` on the underlying CDLL until the first call so
    that rocke and torch can be imported in any order without ending
    up with two HIP runtimes (see ``_torch_bundled_lib``). Resolved on
    first use; subsequent calls dispatch directly through the cached
    function pointer.

    ``lib_resolver`` returns the shared ``ctypes.CDLL`` for this lib
    family (HIP runtime / comgr / ...). It is invoked exactly once per
    function on first call.
    """

    __slots__ = ("_name", "_argtypes", "_restype", "_lib_resolver", "_fn")

    def __init__(
        self,
        name: str,
        argtypes: List[Any],
        restype: Any,
        lib_resolver: "Callable[[], ctypes.CDLL]",
    ) -> None:
        self._name = name
        self._argtypes = argtypes
        self._restype = restype
        self._lib_resolver = lib_resolver
        self._fn: Optional[Any] = None

    def _resolve(self) -> Any:
        fn = getattr(self._lib_resolver(), self._name)
        fn.argtypes = self._argtypes
        fn.restype = self._restype
        self._fn = fn
        return fn

    def __call__(self, *args: Any) -> Any:
        fn = self._fn or self._resolve()
        return fn(*args)


def _b(name: str, *argtypes, restype=ctypes.c_int) -> _LazyFn:
    return _LazyFn(name, list(argtypes), restype, _resolve_hip)


# HIP function table.
_hipGetErrorString = _b("hipGetErrorString", ctypes.c_int, restype=ctypes.c_char_p)
_hipInit = _b("hipInit", ctypes.c_uint)
_hipSetDevice = _b("hipSetDevice", ctypes.c_int)
_hipGetDevice = _b("hipGetDevice", ctypes.POINTER(ctypes.c_int))
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
_device_arch_cache: Dict[int, Optional[str]] = {}


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


def get_device_arch(device: int = 0) -> Optional[str]:
    """Best-effort gfx string of a HIP device (e.g. ``"gfx942"``).

    Returns ``None`` when it can't be determined (no GPU present, or the
    properties symbol is unavailable). Launch paths use this to compile for
    the device they will actually run on instead of defaulting to a fixed
    arch — building a gfx950 code object and launching it on gfx942 yields
    ``hipError(209) no kernel image``.

    The ``hipDeviceProp_t`` struct layout changes across ROCm releases (and
    the symbol was versioned to ``...R0600`` in ROCm 6.x), so rather than
    mirroring the struct we allocate a generous zeroed buffer, fill it via
    ``hipGetDeviceProperties*``, and scan it for the ``gfx<...>`` token that
    ``gcnArchName`` carries. ``name`` (the marketing string) contains no
    ``gfx`` token, so the first match is the architecture name.
    """
    import re

    device = int(device)
    if device in _device_arch_cache:
        return _device_arch_cache[device]

    buf = ctypes.create_string_buffer(4096)
    for sym in ("hipGetDevicePropertiesR0600", "hipGetDeviceProperties"):
        fn = _b(sym, ctypes.c_void_p, ctypes.c_int)
        try:
            rc = fn(buf, device)
        except (AttributeError, OSError):
            continue
        if rc != 0:
            continue
        m = re.search(rb"gfx[0-9a-z]+", buf.raw)
        if m:
            arch = m.group(0).decode("ascii")
            _device_arch_cache[device] = arch
            return arch
    _device_arch_cache[device] = None
    return None


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
    # entries. Every launch appends exactly one entry; tensor lifetimes
    # (set up via :meth:`retain_for_stream`) merge into the most-recent
    # entry so they share the launch's completion event.
    #
    # Why this exists
    # ---------------
    # Raw ``hipModuleLaunchKernel`` calls go through ctypes and are
    # invisible to torch's stream-aware caching allocator. Two failure
    # modes follow:
    #
    # 1. The HIP_LAUNCH_PARAM_BUFFER_POINTER ("extra") path does not
    #    promise to copy the packed-args buffer at enqueue time;
    #    observation on ROCm 6/7 is that the GPU command processor
    #    reads it later, when it actually starts the kernel. If the
    #    Python-owned ctypes buffer has been garbage-collected by then,
    #    the kernel reads stale memory and writes to whatever pointer
    #    those bytes now decode as.
    #
    # 2. Output / workspace tensors built with ``torch.empty(...)`` are
    #    tracked by torch's caching allocator against torch's
    #    *current* stream. Once the Python reference drops, the
    #    allocator can recycle that memory while the raw HIP launch is
    #    still in flight, mutating the kernel's destination buffer.
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
