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
from dataclasses import dataclass
from typing import Tuple


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
    for p in [
        "/opt/rocm/lib/libamdhip64.so",
        "/opt/rocm/lib/libamdhip64.so.7",
        "libamdhip64.so",
    ]:
        try:
            return ctypes.CDLL(p)
        except OSError as e:
            err = e
    raise HipError(f"cannot load libamdhip64.so ({err!r})")


_hip = _load_lib()


def _b(name: str, *argtypes, restype=ctypes.c_int):
    fn = getattr(_hip, name)
    fn.argtypes = list(argtypes)
    fn.restype = restype
    return fn


# HIP function table.
_hipGetErrorString = _b("hipGetErrorString", ctypes.c_int, restype=ctypes.c_char_p)
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
_hipEventCreate = _b("hipEventCreate", ctypes.POINTER(_HipEventHandle))
_hipEventDestroy = _b("hipEventDestroy", _HipEventHandle)
_hipEventRecord = _b("hipEventRecord", _HipEventHandle, ctypes.c_void_p)
_hipEventSynchronize = _b("hipEventSynchronize", _HipEventHandle)
_hipEventElapsedTime = _b(
    "hipEventElapsedTime",
    ctypes.POINTER(ctypes.c_float),
    _HipEventHandle,
    _HipEventHandle,
)


def _check(s: int, where: str) -> None:
    if s != 0:
        msg = _hipGetErrorString(s)
        raise HipError(f"{where}: hipError({s}) {msg.decode() if msg else ''}")


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
    def load_module(self, blob: bytes) -> Module:
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

    def sync(self) -> None:
        _check(_hipDeviceSynchronize(), "hipDeviceSynchronize")

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
    ) -> None:
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
                ctypes.c_void_p(stream),
                None,
                extra,
            ),
            "hipModuleLaunchKernel",
        )
