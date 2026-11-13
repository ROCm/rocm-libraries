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
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core.arch import base_arch_from_target_id
from ..core.ir import gpu_check_word_symbol
from ._ctypes_bind import _LazyFn
from ._hip_device_properties import HipDevicePropR0600
from .runtime_coexistence import _IS_WINDOWS, _add_dll_dir, _candidate_lib_paths

HIP_LAUNCH_PARAM_BUFFER_POINTER = ctypes.c_void_p(1)
HIP_LAUNCH_PARAM_BUFFER_SIZE = ctypes.c_void_p(2)
HIP_LAUNCH_PARAM_END = ctypes.c_void_p(3)

hipMemcpyHostToDevice = 1
hipMemcpyDeviceToHost = 2


class HipError(RuntimeError):
    pass


class GpuCheckFailed(RuntimeError):
    """A ``gpu_checked_value`` fact was false in a launch the runtime waited
    for (see :class:`GpuCheckLedger`).

    Every output of that launch is undefined. The GPU stays usable.
    """


class GpuCheckCaptureRefused(RuntimeError):
    """A launch of a kernel with a GPU-check word, queued on a stream that is
    being captured into a graph. Nothing reads the word after a graph replay,
    so a false ``gpu_checked_value`` fact in it would go unseen: launch the
    kernel outside the capture, or build it without ``gpu_checked_value``."""


class _HipModuleHandle(ctypes.Structure):
    _fields_ = [("p", ctypes.c_void_p)]


class _HipFunctionHandle(ctypes.Structure):
    _fields_ = [("p", ctypes.c_void_p)]


class _HipEventHandle(ctypes.Structure):
    _fields_ = [("p", ctypes.c_void_p)]


# ``hipLaunchAttributeID`` value selecting ``val.clusterDim``.
_HIP_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = 4


class _HipClusterDim(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint), ("y", ctypes.c_uint), ("z", ctypes.c_uint)]


class _HipLaunchAttributeValue(ctypes.Union):
    # The HIP union is padded to 64 bytes and 8-byte aligned; only the
    # cluster member is used here.
    _fields_ = [("clusterDim", _HipClusterDim), ("pad", ctypes.c_uint64 * 8)]


class _HipLaunchAttribute(ctypes.Structure):
    _fields_ = [("id", ctypes.c_int), ("val", _HipLaunchAttributeValue)]


class _HipLaunchConfig(ctypes.Structure):
    """``HIP_LAUNCH_CONFIG`` as taken by ``hipDrvLaunchKernelEx``."""

    _fields_ = [
        ("gridDimX", ctypes.c_uint),
        ("gridDimY", ctypes.c_uint),
        ("gridDimZ", ctypes.c_uint),
        ("blockDimX", ctypes.c_uint),
        ("blockDimY", ctypes.c_uint),
        ("blockDimZ", ctypes.c_uint),
        ("sharedMemBytes", ctypes.c_uint),
        ("hStream", ctypes.c_void_p),
        ("attrs", ctypes.POINTER(_HipLaunchAttribute)),
        ("numAttrs", ctypes.c_uint),
    ]


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
_hipGetLastError = _b("hipGetLastError")
_hipModuleGetGlobal = _b(
    "hipModuleGetGlobal",
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_size_t),
    _HipModuleHandle,
    ctypes.c_char_p,
)
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
_hipDrvLaunchKernelEx = _b(
    "hipDrvLaunchKernelEx",
    ctypes.POINTER(_HipLaunchConfig),
    _HipFunctionHandle,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_void_p),
)
_hipMalloc = _b("hipMalloc", ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t)
_hipFree = _b("hipFree", ctypes.c_void_p)
_hipMemcpy = _b(
    "hipMemcpy", ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
)
_hipMemcpyAsync = _b(
    "hipMemcpyAsync",
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_void_p,
)
_hipMemset = _b("hipMemset", ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t)
_hipMemsetAsync = _b(
    "hipMemsetAsync",
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_size_t,
    ctypes.c_void_p,
)
_hipExtMallocWithFlags = _b(
    "hipExtMallocWithFlags",
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_size_t,
    ctypes.c_uint,
)
_hipStreamWaitValue32 = _b(
    "hipStreamWaitValue32",
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.c_uint,
    ctypes.c_uint32,
)
_hipDeviceSynchronize = _b("hipDeviceSynchronize")
_hipStreamSynchronize = _b("hipStreamSynchronize", ctypes.c_void_p)
_hipStreamCreate = _b("hipStreamCreate", ctypes.POINTER(ctypes.c_void_p))
_hipStreamDestroy = _b("hipStreamDestroy", ctypes.c_void_p)
_hipStreamIsCapturing = _b(
    "hipStreamIsCapturing", ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)
)
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
HIP_ERROR_NOT_FOUND = 500
HIP_ERROR_NOT_READY = 600
# hipStreamCaptureStatus: the stream is not being captured.
HIP_STREAM_CAPTURE_STATUS_NONE = 0


def _check(s: int, where: str) -> None:
    if s != 0:
        msg = _hipGetErrorString(s)
        raise HipError(f"{where}: hipError({s}) {msg.decode() if msg else ''}")


_hip_inited = False


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


def _device_props(device: int = 0) -> HipDevicePropR0600 | None:
    """Read HIP R0600 properties, or return ``None`` if unavailable.

    The function version must match the structure layout. Runtimes without
    ``hipGetDevicePropertiesR0600`` are unsupported by this query. Each call
    reads fresh properties from HIP.

    This does not call ``_ensure_hip_init()`` or select a device. HIP may
    initialize its runtime internally when handling the query.
    """
    device = int(device)
    props = HipDevicePropR0600()
    fn = _b(
        "hipGetDevicePropertiesR0600",
        ctypes.POINTER(HipDevicePropR0600),
        ctypes.c_int,
    )
    try:
        rc = fn(ctypes.byref(props), device)
    except (AttributeError, HipError, OSError):
        return None
    return props if rc == 0 else None


def get_device_target_id(device: int = 0) -> Optional[str]:
    """Read the target ID from HIP device properties, or return ``None``.

    Reads ``gcnArchName`` from :func:`_device_props`, preserving suffixes such
    as ``-strict`` and ``:sramecc+:xnack-``. HIP supplies this string from the
    device's ISA target ID; it does not include a COMGR ISA prefix.
    """
    props = _device_props(device)
    if props is None:
        return None
    return props.gcnArchName.decode("ascii", "replace") or None


def get_device_arch(device: int = 0) -> Optional[str]:
    """Return the device's base architecture, or ``None`` if unavailable.

    Applies :func:`base_arch_from_target_id` to :func:`get_device_target_id`
    so existing callers get names such as ``gfx942`` for catalog lookup and
    lowering. Use :func:`get_device_target_id` to retain profiles and features.
    """

    target_id = get_device_target_id(device)
    return base_arch_from_target_id(target_id) if target_id is not None else None


def get_device_name(device: int = 0) -> Optional[str]:
    """Read the HIP device name, or return ``None`` if unavailable.

    Reads ``name`` from :func:`_device_props`, the same field used by
    ``torch.cuda.get_device_name``, without importing torch.
    """
    props = _device_props(device)
    if props is None:
        return None
    return props.name.decode("ascii", "replace") or None


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


# hipDeviceAttributeMultiprocessorCount from HIP's stable hipDeviceAttribute_t enum.
_HIP_ATTR_MULTIPROCESSOR_COUNT = 63


def get_device_num_cus(device: int = 0) -> Optional[int]:
    """CU (multiprocessor) count of a HIP device, or None if unqueryable.

    Torch-free ctypes twin of ``get_device_arch`` / ``get_device_name``: uses
    ``hipDeviceGetAttribute`` (single ``int``, no struct-offset guessing) so the
    dispatch/benchmark layers can size split-KV device subscription without a torch
    dependency. Best-effort and side-effect-free (``hipDeviceGetAttribute`` lazily
    inits the runtime internally). NOTE: this reports CUs on CU-mode devices (CDNA);
    a WGP-mode (RDNA) device reports half the CU count — inert for the current
    CDNA-only callers (gfx942 / gfx950), but a non-CDNA caller must account for it.
    """
    v = ctypes.c_int(0)
    try:
        rc = _hipDeviceGetAttribute(
            ctypes.byref(v), _HIP_ATTR_MULTIPROCESSOR_COUNT, int(device)
        )
    except (AttributeError, OSError):
        return None
    return int(v.value) if rc == 0 and v.value > 0 else None


# Per-device ``clusterLaunch`` answers; a device's capability does not change
# during the process, and the property query is too slow for every launch.
_cluster_launch_support: Dict[int, Optional[bool]] = {}


def device_supports_cluster_launch(device: int = 0) -> Optional[bool]:
    """Whether HIP reports cluster launch support, or ``None`` if unknown.

    Reads ``clusterLaunch`` from :func:`_device_props`. The answer is cached
    per device.
    """
    device = int(device)
    if device not in _cluster_launch_support:
        props = _device_props(device)
        _cluster_launch_support[device] = (
            None if props is None else bool(props.clusterLaunch)
        )
    return _cluster_launch_support[device]


def current_device() -> int:
    """HIP's current device for this thread: the one rocke's runtime launches
    kernels on. Returns ``0``, HIP's default device, when the runtime cannot be
    queried."""
    cur = ctypes.c_int(0)
    try:
        rc = _hipGetDevice(ctypes.byref(cur))
    except (AttributeError, OSError):
        return 0
    return int(cur.value) if rc == HIP_SUCCESS and cur.value >= 0 else 0


def check_launch_cluster(
    grid: Tuple[int, int, int], cluster: object
) -> Tuple[int, int, int]:
    """Validate a launch cluster shape against ``grid`` and return it.

    The shape follows the same rules as the kernel's ``cluster_dims`` attr,
    and the grid (in workgroups) must be a whole number of clusters in every
    dimension.
    """
    from ..core.ir import check_cluster_dims

    shape = check_cluster_dims(cluster)
    for axis, g, c in zip("xyz", grid, shape):
        if int(g) % c != 0:
            raise ValueError(
                f"grid {tuple(int(v) for v in grid)} is not a multiple of "
                f"cluster {shape} in {axis}"
            )
    return shape


def _sync_stream(stream: int) -> None:
    """Wait for everything queued on ``stream`` (``hipStreamSynchronize``)."""
    _check(_hipStreamSynchronize(ctypes.c_void_p(int(stream))), "hipStreamSynchronize")


def _stream_is_capturing(stream: int) -> bool:
    """Whether ``stream`` is being captured into a graph."""
    status = ctypes.c_int(HIP_STREAM_CAPTURE_STATUS_NONE)
    _check(
        _hipStreamIsCapturing(ctypes.c_void_p(int(stream)), ctypes.byref(status)),
        "hipStreamIsCapturing",
    )
    return status.value != HIP_STREAM_CAPTURE_STATUS_NONE


@dataclass(frozen=True)
class GpuCheckWord:
    """The GPU-check word of one loaded kernel: an i32 in its code object,
    zero at load, that a lane sets to 1 when a ``gpu_checked_value`` fact is
    false, just before it returns. Every stream shares it.

    The I/O on one word; :class:`GpuCheckLedger` decides when to read and
    clear it.
    """

    kernel: str
    address: int

    def queue_read(self, stream: int, into: ctypes.c_int32) -> None:
        """Queue a copy of the word to ``into`` on ``stream``, so the copy
        follows every launch queued there; ``into`` holds the word once
        ``stream`` has been waited for."""
        _check(
            _hipMemcpyAsync(
                ctypes.byref(into),
                ctypes.c_void_p(self.address),
                4,
                hipMemcpyDeviceToHost,
                ctypes.c_void_p(int(stream)),
            ),
            f"hipMemcpyAsync({gpu_check_word_symbol(self.kernel)})",
        )

    def clear(self, stream: int) -> None:
        """Queue a store of 0 to the word on ``stream``."""
        _check(
            _hipMemsetAsync(
                ctypes.c_void_p(self.address), 0, 4, ctypes.c_void_p(int(stream))
            ),
            f"hipMemsetAsync({gpu_check_word_symbol(self.kernel)})",
        )


def _gpu_check_failed_text(kernel: str, shared: bool) -> str:
    """The :class:`GpuCheckFailed` text for ``kernel``'s set word; ``shared``
    when another stream also launched it since the word was last cleared."""
    where = (
        "on this stream or on another stream (the streams share the kernel's one "
        "check word, so it cannot say which)"
        if shared
        else "on this stream"
    )
    launches = "those launches" if shared else "that launch"
    return (
        f"{kernel}: a gpu_checked_value fact was false in a launch of {kernel} "
        f"{where}; every output of {launches} is undefined"
    )


# A (device, stream) pair: the null stream (0) of two devices are two keys.
_StreamKey = tuple[int, int]


class GpuCheckLedger:
    """Which GPU-check words a runtime wait reads, and when each is cleared.

    :meth:`Module.get_function` registers a kernel's word; :class:`Runtime`
    notes the word at each launch of the kernel (:meth:`note`) and reads it at
    its next wait on that stream -- :meth:`Runtime.launch_blocking`,
    :meth:`Runtime.stream_sync`, :meth:`Runtime.wait_stream`,
    :meth:`Runtime.sync`, :meth:`Runtime.release_pending_for_stream` and
    :meth:`Runtime.check_gpu_checks` -- raising :class:`GpuCheckFailed` when
    it is set. The wait does not clear it: the first launch of the kernel after
    every noted launch has been read clears it, queued ahead of that launch on
    its stream. So:

    * a set word is raised at the wait of every stream with a noted launch of
      the kernel that has not been read yet, since one word cannot say which
      of them failed;
    * no failure of a noted launch is lost between a read and a clear;
    * a word left set by a launch the runtime never saw is cleared, not
      blamed on the next launch.

    A launch queued while its stream is being captured into a graph is
    refused (:class:`GpuCheckCaptureRefused`): nothing reads the word after a
    replay. Not checked: a launch outside ``rocke.runtime`` is never noted, so
    nothing reads the word for it. A wait outside the runtime (a torch
    synchronize, an event) reads nothing either: call
    :meth:`Runtime.check_gpu_checks` after it.

    One instance serves every :class:`Runtime` and :class:`Module`: a wait
    reads what any runtime launched. ``current_device`` is how the ledger asks
    for the calling thread's HIP device; it asks only when it notes a word or
    holds an unread one.
    """

    def __init__(self, current_device: Callable[[], int]) -> None:
        self._current_device = current_device
        # By hipFunction_t address, the word of each kernel that has one.
        self._words_by_function: dict[int, GpuCheckWord] = {}
        # Per (device, stream), the words of the kernels launched there since
        # the last runtime wait on it, by word address.
        self._unread: dict[_StreamKey, dict[int, GpuCheckWord]] = {}
        # Per word address, the (device, stream) keys that noted a launch of its
        # kernel since the word was last cleared: more than one, and a set word
        # cannot say which of them failed.
        self._streams_by_word: dict[int, set[_StreamKey]] = {}

    def register(self, function: _HipFunctionHandle, word: GpuCheckWord) -> None:
        """Record that ``function`` (a :meth:`Module.get_function` handle) has
        ``word``."""
        self._words_by_function[int(function.p)] = word

    def word_of(self, function: Any) -> GpuCheckWord | None:
        """``function``'s GPU-check word, or None for a kernel without one and
        for any handle :meth:`Module.get_function` did not return."""
        if not isinstance(function, _HipFunctionHandle) or not function.p:
            return None
        return self._words_by_function.get(int(function.p))

    def note(self, function: Any, stream: int) -> None:
        """Note ``function``'s word, if it has one, for the next wait on
        ``stream``. Call before queueing the launch: when no noted launch of
        the kernel is unread, this queues the word's clear on ``stream`` first.

        Raises :class:`GpuCheckCaptureRefused` when ``stream`` is being
        captured.
        """
        word = self.word_of(function)
        if word is None:
            return
        if _stream_is_capturing(stream):
            raise GpuCheckCaptureRefused(
                f"{word.kernel}: this kernel checks gpu_checked_value facts, and "
                "its stream is being captured into a graph; nothing reads the "
                "kernel's GPU-check word after a replay, so a false fact would go "
                "unseen. Launch it outside the capture."
            )
        key = (self._current_device(), int(stream))
        if not any(word.address in words for words in self._unread.values()):
            word.clear(stream)
            self._streams_by_word[word.address] = set()
        self._streams_by_word.setdefault(word.address, set()).add(key)
        self._unread.setdefault(key, {})[word.address] = word

    def read(self, stream: int | None) -> None:
        """Read the words noted for ``stream`` on the current device (every
        stream of it when None) and raise :class:`GpuCheckFailed` naming every
        kernel whose word is set.

        Per stream, every copy is queued first and the stream is waited for
        once. A stream's notes are dropped only after its words were read.
        """
        if not self._unread:
            return
        device = self._current_device()
        if stream is None:
            keys = [key for key in self._unread if key[0] == device]
        else:
            keys = [(device, int(stream))]
        failed: list[str] = []
        for key in keys:
            words = list(self._unread.get(key, {}).values())
            if not words:
                continue
            values = [ctypes.c_int32(0) for _ in words]
            for word, value in zip(words, values):
                word.queue_read(key[1], value)
            _sync_stream(key[1])
            del self._unread[key]
            for word, value in zip(words, values):
                if value.value:
                    shared = len(self._streams_by_word.get(word.address, ())) > 1
                    failed.append(_gpu_check_failed_text(word.kernel, shared))
        if failed:
            raise GpuCheckFailed("; ".join(failed))

    def forget(self, words: Iterable[GpuCheckWord]) -> None:
        """Stop reading ``words`` and drop their kernels' registrations (their
        code object is being unloaded)."""
        gone = {word.address for word in words}
        for address in gone:
            self._streams_by_word.pop(address, None)
        for function in [
            f for f, word in self._words_by_function.items() if word.address in gone
        ]:
            del self._words_by_function[function]
        for key in list(self._unread):
            unread = self._unread[key]
            for address in gone.intersection(unread):
                del unread[address]
            if not unread:
                del self._unread[key]


_GPU_CHECKS = GpuCheckLedger(current_device)


@dataclass
class Module:
    handle: _HipModuleHandle
    # The GPU-check words handed out by get_function, by kernel name.
    gpu_checks: dict[str, GpuCheckWord] = field(default_factory=dict)

    def get_function(self, name: str) -> _HipFunctionHandle:
        """The kernel ``name``. When it has a GPU-check word
        (:meth:`gpu_check_word`), the runtime reads it after every launch of
        the returned handle (see :class:`GpuCheckLedger`)."""
        fn = _HipFunctionHandle()
        _check(
            _hipModuleGetFunction(ctypes.byref(fn), self.handle, name.encode("utf-8")),
            f"hipModuleGetFunction({name})",
        )
        word = self.gpu_check_word(name)
        if word is not None:
            _GPU_CHECKS.register(fn, word)
        return fn

    def gpu_check_word(self, name: str) -> GpuCheckWord | None:
        """``name``'s GPU-check word, or None when the code object does not
        define one (``hipModuleGetGlobal`` finds no such symbol)."""
        if name in self.gpu_checks:
            return self.gpu_checks[name]
        symbol = gpu_check_word_symbol(name)
        address = ctypes.c_void_p(0)
        size = ctypes.c_size_t(0)
        status = _hipModuleGetGlobal(
            ctypes.byref(address),
            ctypes.byref(size),
            self.handle,
            symbol.encode("utf-8"),
        )
        if status == HIP_ERROR_NOT_FOUND:
            # Clear the not-found status HIP keeps for hipGetLastError, or the
            # next caller that checks it (torch after a launch) reports it.
            _hipGetLastError()
            return None
        _check(status, f"hipModuleGetGlobal({symbol})")
        word = GpuCheckWord(name, int(address.value))
        self.gpu_checks[name] = word
        return word

    def unload(self) -> None:
        """Unload the code object. Its GPU-check words are no longer read:
        their memory goes with it."""
        _GPU_CHECKS.forget(self.gpu_checks.values())
        self.gpu_checks.clear()
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

    def check_gpu_checks(self, stream: int | None = None) -> None:
        """Raise :class:`GpuCheckFailed` if a ``gpu_checked_value`` fact was
        false in a launch queued through this runtime on ``stream`` (every
        stream of the current device when None) since the last runtime wait
        on it.

        The runtime's own waits read the words already. Call this after
        waiting some other way -- ``torch.cuda.synchronize()``, an event's
        ``synchronize``, a library that syncs the stream -- before using the
        outputs: no such wait reads them. It reads on ``stream`` (a copy
        queued there, then a stream sync), so it also waits for what is
        queued on ``stream``; with ``None`` it waits for each stream that has
        an unread launch.
        """
        _GPU_CHECKS.read(None if stream is None else int(stream))

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

    def memset_async(self, ptr: int, value: int, nbytes: int, stream: int = 0) -> None:
        """Queue a fill on ``stream`` instead of draining the device for it.

        :meth:`memset` is ``hipMemset``, which is synchronous with respect to
        the host and the device: it waits for everything already queued. That
        is the right primitive for setting a buffer up between measurements,
        and the wrong one *inside* one -- a kernel whose contract is "zero
        this, then launch" (a split-K epilogue accumulating into its output,
        a stream-K flag table) would drain the device once per launch, which
        serialises a batch that was queued precisely so it would not.

        Queued on one stream, the fill is ordered against that stream's
        launches by FIFO alone, so it is also what makes a batch of such
        kernels safe to queue back to back on shared buffers.
        """
        _check(
            _hipMemsetAsync(
                ctypes.c_void_p(ptr), value, nbytes, ctypes.c_void_p(stream)
            ),
            "hipMemsetAsync",
        )

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

    def create_stream(self) -> int:
        """A new HIP stream (``hipStreamCreate``), as the int handle every
        launch and wait takes. Destroy it with :meth:`destroy_stream`.

        A launch on it is not mapped to torch's current stream, as one on
        stream 0 is (``torch_interop.resolve_stream``), so a caller that does
        not use torch never imports or initializes it.
        """
        _ensure_hip_init()
        handle = ctypes.c_void_p(0)
        _check(_hipStreamCreate(ctypes.byref(handle)), "hipStreamCreate")
        return int(handle.value)

    def destroy_stream(self, stream: int) -> None:
        """Drain ``stream`` (:meth:`wait_stream`, which also drops what the
        runtime retained for it) and destroy it (``hipStreamDestroy``).

        The stream is destroyed even when the drain raises :class:`GpuCheckFailed`
        (a launch on it failed its GPU check); that error is then raised, unless
        destroying the stream fails too, whose error is raised instead."""
        try:
            self.wait_stream(stream)
        finally:
            _check(_hipStreamDestroy(ctypes.c_void_p(int(stream))), "hipStreamDestroy")

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

        Then reads the GPU-check words of the kernels launched on ``stream``
        and raises :class:`GpuCheckFailed` if one is set (see
        :class:`GpuCheckLedger`).
        """
        _sync_stream(stream)
        _GPU_CHECKS.read(int(stream))

    def wait_stream(self, stream: int) -> None:
        """Per-stream drain + release all retained refs.

        Uses ``hipStreamSynchronize`` as the safe, cheap primitive that
        works on ROCm for raw ``hipModuleLaunchKernel`` work queued
        through ctypes. (``torch.cuda.synchronize()`` does not reliably
        drain that queue.) If the bucket also holds event-tagged
        entries from a prior async batch (e.g. inside
        :func:`time_launches`), every event will have fired by the
        time the stream-sync returns, and we destroy them as we drop
        the bucket. Then reads the GPU-check words, as :meth:`stream_sync`
        does.
        """
        s = int(stream)
        _sync_stream(s)
        bucket = self._pending_args.pop(s, None)
        for _refs, evt in bucket or ():
            if evt is not None:
                evt.destroy()
        _GPU_CHECKS.read(s)

    def sync(self) -> None:
        """Device-wide drain: ``hipDeviceSynchronize`` + release everything.

        Use :meth:`wait_stream` instead when the caller knows the
        target stream; this method is the broad hammer for benchmark
        harnesses that span multiple streams or want strong isolation
        between independent lanes (e.g. Triton 2D -> CK 2D -> Triton 3D
        -> CK 3D in the parity harness).

        Then reads the GPU-check words of every stream, as
        :meth:`stream_sync` does for one.
        """
        _check(_hipDeviceSynchronize(), "hipDeviceSynchronize")
        for stream_id in list(self._pending_args.keys()):
            for _refs, evt in self._pending_args[stream_id]:
                if evt is not None:
                    evt.destroy()
        self._pending_args.clear()
        _GPU_CHECKS.read(None)

    def release_pending_for_stream(self, stream: int) -> None:
        """Drop refs held for ``stream`` after the caller has ensured
        the stream is drained (e.g. via :meth:`wait_stream`,
        :meth:`sync`, or an external event sync).

        Most callers should prefer :meth:`wait_stream`, which performs
        the wait *and* drops refs in one call. This method exists for
        callers that did their own sync via a different mechanism and
        only need the bookkeeping cleanup.

        Then reads the GPU-check words of the kernels launched on
        ``stream``, as :meth:`check_gpu_checks` does.
        """
        s = int(stream)
        bucket = self._pending_args.pop(s, None)
        for _refs, evt in bucket or ():
            if evt is not None:
                evt.destroy()
        _GPU_CHECKS.read(s)

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

    @staticmethod
    def _enqueue(
        fn: _HipFunctionHandle,
        grid: Tuple[int, int, int],
        block: Tuple[int, int, int],
        shared_bytes: int,
        stream: int,
        params: Any,
        extra: Any,
        cluster: Optional[Tuple[int, int, int]],
        where: str,
    ) -> Tuple[Any, ...]:
        """Enqueue one launch; return the ctypes objects it pointed HIP at.

        Without ``cluster`` this is a plain ``hipModuleLaunchKernel``. With
        one, the shape is validated (see :func:`check_launch_cluster`) and
        the launch goes through ``hipDrvLaunchKernelEx`` with a cluster
        dimension attribute. The returned config and attribute must be kept
        alive with the args buffer.
        """
        if cluster is None:
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
                    ctypes.c_void_p(stream),
                    params,
                    extra,
                ),
                where,
            )
            return ()

        cfg, attr = Runtime._cluster_launch_config(
            grid, block, shared_bytes, stream, cluster, where
        )
        Runtime._launch_cluster(ctypes.byref(cfg), fn, params, extra, where)
        return (cfg, attr)

    @staticmethod
    def _launch_cluster(
        cfg_ref: Any, fn: _HipFunctionHandle, params: Any, extra: Any, where: str
    ) -> None:
        """Issue a cluster launch built by :meth:`_cluster_launch_config`.

        The one ``hipDrvLaunchKernelEx`` call site: :meth:`_enqueue` and the
        callables of :meth:`prepare_launch` both go through it, so the two
        paths cannot drift apart.
        """
        _check(
            _hipDrvLaunchKernelEx(cfg_ref, fn, params, extra),
            f"hipDrvLaunchKernelEx({where})",
        )

    @staticmethod
    def _cluster_launch_config(
        grid: Tuple[int, int, int],
        block: Tuple[int, int, int],
        shared_bytes: int,
        stream: int,
        cluster: Tuple[int, int, int],
        where: str,
    ) -> Tuple[Any, Any]:
        """Validate ``cluster`` and build the ``hipDrvLaunchKernelEx`` config.

        Returns ``(cfg, attr)``: the launch config and the cluster-dimension
        attribute it points at, which must stay alive together. Raises before
        HIP sees anything for a bad shape (see :func:`check_launch_cluster`)
        or a device that reports no cluster launch support.
        """
        shape = check_launch_cluster(grid, cluster)
        device = current_device()
        if device_supports_cluster_launch(device) is False:
            raise HipError(
                f"{where}: device {device} does not support cluster launch "
                f"(cluster {shape})"
            )
        attr = _HipLaunchAttribute()
        attr.id = _HIP_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        attr.val.clusterDim.x, attr.val.clusterDim.y, attr.val.clusterDim.z = shape
        cfg = _HipLaunchConfig(
            gridDimX=grid[0],
            gridDimY=grid[1],
            gridDimZ=grid[2],
            blockDimX=block[0],
            blockDimY=block[1],
            blockDimZ=block[2],
            sharedMemBytes=shared_bytes,
            hStream=stream,
            attrs=ctypes.pointer(attr),
            numAttrs=1,
        )
        return cfg, attr

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
        cluster: Optional[Tuple[int, int, int]] = None,
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

        ``cluster=(x, y, z)`` launches workgroup clusters of that shape
        through ``hipDrvLaunchKernelEx``. It must match the kernel's compiled
        ``cluster_dims`` (the runtime does not check this) and evenly divide
        ``grid``. The same argument exists on :meth:`launch_blocking` and
        :meth:`launch_kernelparams`.
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
        _GPU_CHECKS.note(fn, s)
        cluster_refs = self._enqueue(
            fn,
            grid,
            block,
            shared_bytes,
            s,
            None,
            extra,
            cluster,
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
        bucket.append(((args_buf, size_buf, extra) + cluster_refs, evt))
        return evt

    def prepare_launch(
        self,
        fn: _HipFunctionHandle,
        grid: Tuple[int, int, int],
        block: Tuple[int, int, int],
        args_packed: bytes,
        *,
        shared_bytes: int = 0,
        stream: int = 0,
        cluster: Optional[Tuple[int, int, int]] = None,
    ) -> "Callable[[], None]":
        """Build every ctypes argument of one launch once; return a
        zero-argument callable that only enqueues it.

        ``cluster`` launches workgroup clusters of that shape, as in
        :meth:`launch` (``hipDrvLaunchKernelEx`` with a cluster attribute).

        For loops that launch the same kernel with the same arguments
        over and over -- the timed loop of a benchmark. :meth:`launch`
        rebuilds the args buffer, the ``extra`` array and the scalar
        ctypes on each call, which costs more host time than a small
        kernel runs, so a timed loop of those measures the host and not
        the GPU. The callable owns its args buffer, so it stays valid
        for as long as the callable is alive and is never written to
        again: the ``extra`` path's late host-buffer read (see
        :func:`rocke.runtime.packing.pack_args_kernelparams`) always
        sees the right bytes.
        """
        args_buf = (ctypes.c_ubyte * len(args_packed)).from_buffer_copy(args_packed)
        size_buf = ctypes.c_size_t(len(args_packed))
        extra = (ctypes.c_void_p * 5)(
            HIP_LAUNCH_PARAM_BUFFER_POINTER,
            ctypes.cast(args_buf, ctypes.c_void_p),
            HIP_LAUNCH_PARAM_BUFFER_SIZE,
            ctypes.cast(ctypes.pointer(size_buf), ctypes.c_void_p),
            HIP_LAUNCH_PARAM_END,
        )
        call_args = (
            fn,
            ctypes.c_uint(grid[0]),
            ctypes.c_uint(grid[1]),
            ctypes.c_uint(grid[2]),
            ctypes.c_uint(block[0]),
            ctypes.c_uint(block[1]),
            ctypes.c_uint(block[2]),
            ctypes.c_uint(shared_bytes),
            ctypes.c_void_p(int(stream)),
            None,
            extra,
        )
        keep = (args_buf, size_buf, extra)
        stream_id = int(stream)
        checked = _GPU_CHECKS.word_of(fn) is not None

        if cluster is not None:
            # Validated, device-checked and built once, like the plain path's
            # ctypes arguments: the callable only enqueues, through the same
            # launch helper as :meth:`_enqueue`.
            cfg, attr = self._cluster_launch_config(
                grid, block, shared_bytes, int(stream), cluster, "prepare_launch"
            )
            cfg_ref = ctypes.byref(cfg)

            def enqueue_cluster(_keep=keep + (cfg, attr)) -> None:
                if checked:
                    _GPU_CHECKS.note(fn, stream_id)
                Runtime._launch_cluster(cfg_ref, fn, None, extra, "prepare_launch")

            return enqueue_cluster

        def enqueue(_keep=keep) -> None:
            if checked:
                _GPU_CHECKS.note(fn, stream_id)
            _check(_hipModuleLaunchKernel(*call_args), "hipModuleLaunchKernel")

        return enqueue

    def launch_blocking(
        self,
        fn: _HipFunctionHandle,
        grid: Tuple[int, int, int],
        block: Tuple[int, int, int],
        args_packed: bytes,
        *,
        shared_bytes: int = 0,
        stream: int = 0,
        cluster: Optional[Tuple[int, int, int]] = None,
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

        Raises :class:`GpuCheckFailed` after the wait when a
        ``gpu_checked_value`` fact of ``fn`` (or of a kernel launched earlier
        on ``stream`` without a wait) was false; see :class:`GpuCheckLedger`.
        A kernel with a GPU-check word costs one ``hipStreamIsCapturing`` and
        one ``hipGetDevice`` at the launch (plus the word's clear when no
        earlier launch of it is unread), and at the wait one more
        ``hipGetDevice``, one ``hipMemcpyAsync`` and one more
        ``hipStreamSynchronize``. A kernel without one costs a dictionary
        lookup.
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
        _GPU_CHECKS.note(fn, s)
        cluster_refs = self._enqueue(
            fn,
            grid,
            block,
            shared_bytes,
            s,
            None,
            extra,
            cluster,
            "hipModuleLaunchKernel",
        )
        # Single ``hipStreamSynchronize`` is both the kernel-completion
        # wait and the args-buffer-drain barrier. After it returns,
        # ``args_buf``/``size_buf``/``extra`` (and ``cluster_refs``) can be
        # dropped by Python's frame cleanup -- the GPU is no longer reading
        # them.
        _sync_stream(s)
        del cluster_refs
        _GPU_CHECKS.read(s)

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
        cluster: Optional[Tuple[int, int, int]] = None,
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
        _GPU_CHECKS.note(fn, s)
        cluster_refs = self._enqueue(
            fn,
            grid,
            block,
            shared_bytes,
            s,
            params,
            None,
            cluster,
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
        bucket.append(((tuple(keep_alive), params) + cluster_refs, evt))
        return evt
