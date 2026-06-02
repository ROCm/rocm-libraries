#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
T2.2 — Multi-kernel Python binding for the CK Tile GEMM Dispatcher.

Wraps ``libdispatcher_gemm.so`` (built from dispatcher_capi.h) via ctypes,
exposing a Pythonic interface for kernel enumeration, selection, and execution.

Build the shared library first::

    hipcc -fPIC -shared -o libdispatcher_gemm.so \\
          dispatcher_capi.cpp \\
          -I<ck_include_root> \\
          -include <output_dir>/<kernel_set>/dispatcher_wrappers/register_all_kernels.hpp

Usage::

    from dispatcher_binding import DispatcherLib
    import numpy as np

    lib = DispatcherLib("./libdispatcher_gemm.so")
    print(lib.kernel_count(), "kernels registered")
    names = lib.kernel_names()
    handle = lib.find_kernel(names[0])

    M, N, K = 1024, 1024, 1024
    a = np.random.rand(M, K).astype(np.float16)
    b = np.random.rand(K, N).astype(np.float16)
    c, elapsed_ms = lib.run_gemm(handle, a, b)
    print(f"TFLOP/s: {2*M*N*K / (elapsed_ms*1e-3) / 1e12:.2f}")
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Status codes (mirror dispatcher_capi.h)
# --------------------------------------------------------------------------- #

DISPATCHER_OK           =  0
DISPATCHER_ERR_NOT_FOUND = -1
DISPATCHER_ERR_INVALID   = -2
DISPATCHER_ERR_LAUNCH    = -3
DISPATCHER_ERR_OOM       = -4

_STATUS_NAMES = {
    DISPATCHER_OK:            "OK",
    DISPATCHER_ERR_NOT_FOUND: "NOT_FOUND",
    DISPATCHER_ERR_INVALID:   "INVALID",
    DISPATCHER_ERR_LAUNCH:    "LAUNCH_FAILED",
    DISPATCHER_ERR_OOM:       "OUT_OF_MEMORY",
}


class DispatcherError(RuntimeError):
    def __init__(self, status: int, msg: str = ""):
        name = _STATUS_NAMES.get(status, f"UNKNOWN({status})")
        super().__init__(f"Dispatcher error {name}: {msg}")
        self.status = status


def _check(status: int, context: str = "") -> None:
    if status != DISPATCHER_OK:
        raise DispatcherError(status, context)


# --------------------------------------------------------------------------- #
# ctypes signatures
# --------------------------------------------------------------------------- #

def _bind(lib: ctypes.CDLL) -> None:
    """Attach argtypes/restype to every C API function."""
    # dispatcher_kernel_count(int* count) -> int
    lib.dispatcher_kernel_count.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.dispatcher_kernel_count.restype  = ctypes.c_int

    # dispatcher_kernel_names(const char** names, int max_names) -> int
    lib.dispatcher_kernel_names.argtypes = [
        ctypes.POINTER(ctypes.c_char_p), ctypes.c_int
    ]
    lib.dispatcher_kernel_names.restype = ctypes.c_int

    # dispatcher_kernel_by_name(const char* name, int* handle) -> int
    lib.dispatcher_kernel_by_name.argtypes = [
        ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)
    ]
    lib.dispatcher_kernel_by_name.restype = ctypes.c_int

    # dispatcher_kernel_name_from_handle(int handle, const char** name) -> int
    lib.dispatcher_kernel_name_from_handle.argtypes = [
        ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)
    ]
    lib.dispatcher_kernel_name_from_handle.restype = ctypes.c_int

    # dispatcher_supports(int handle, int M, int N, int K) -> int
    lib.dispatcher_supports.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.dispatcher_supports.restype = ctypes.c_int

    # dispatcher_run_gemm(handle, M,N,K, a,b,c, sa,sb,sc, split_k, stream,
    #                     float* elapsed_ms) -> int
    lib.dispatcher_run_gemm.argtypes = [
        ctypes.c_int,                  # handle
        ctypes.c_int, ctypes.c_int, ctypes.c_int,  # M, N, K
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # a, b, c
        ctypes.c_int, ctypes.c_int, ctypes.c_int,  # stride_a, b, c
        ctypes.c_int,                  # split_k
        ctypes.c_void_p,               # stream (hipStream_t)
        ctypes.POINTER(ctypes.c_float),  # elapsed_ms
    ]
    lib.dispatcher_run_gemm.restype = ctypes.c_int

    # dispatcher_version() -> const char*
    lib.dispatcher_version.argtypes = []
    lib.dispatcher_version.restype  = ctypes.c_char_p


# --------------------------------------------------------------------------- #
# High-level wrapper
# --------------------------------------------------------------------------- #

class DispatcherLib:
    """Python interface to libdispatcher_gemm.so."""

    def __init__(self, so_path: str | Path):
        so_path = str(so_path)
        if not os.path.exists(so_path):
            raise FileNotFoundError(
                f"Shared library not found: {so_path}\n"
                "Build with:\n"
                "  hipcc -fPIC -shared -o libdispatcher_gemm.so \\\n"
                "        dispatcher_capi.cpp \\\n"
                "        -I<ck_include_root> \\\n"
                "        -include <output_dir>/<kernel_set>/dispatcher_wrappers/register_all_kernels.hpp"
            )
        self._lib = ctypes.CDLL(so_path)
        _bind(self._lib)

    def version(self) -> str:
        return self._lib.dispatcher_version().decode()

    def kernel_count(self) -> int:
        count = ctypes.c_int(0)
        _check(self._lib.dispatcher_kernel_count(ctypes.byref(count)), "kernel_count")
        return count.value

    def kernel_names(self) -> List[str]:
        n = self.kernel_count()
        if n == 0:
            return []
        arr = (ctypes.c_char_p * n)()
        written = self._lib.dispatcher_kernel_names(arr, n)
        if written < 0:
            raise DispatcherError(written, "kernel_names")
        return [arr[i].decode() for i in range(written)]

    def find_kernel(self, name: str) -> int:
        """Return an integer handle for a kernel identifier string."""
        handle = ctypes.c_int(-1)
        _check(
            self._lib.dispatcher_kernel_by_name(name.encode(), ctypes.byref(handle)),
            f"find_kernel({name!r})",
        )
        return handle.value

    def kernel_name(self, handle: int) -> str:
        name_ptr = ctypes.c_char_p()
        _check(
            self._lib.dispatcher_kernel_name_from_handle(handle, ctypes.byref(name_ptr)),
            f"kernel_name(handle={handle})",
        )
        return name_ptr.value.decode()

    def supports(self, handle: int, M: int, N: int, K: int) -> bool:
        """Return True if the kernel's supports() predicate accepts (M,N,K)."""
        status = self._lib.dispatcher_supports(handle, M, N, K)
        if status == DISPATCHER_OK:
            return True
        if status == DISPATCHER_ERR_INVALID:
            return False
        raise DispatcherError(status, f"supports(handle={handle}, {M}x{N}x{K})")

    def run_gemm(
        self,
        handle: int,
        a: np.ndarray,
        b: np.ndarray,
        split_k: int = 1,
        stream: Optional[int] = None,
    ) -> Tuple[np.ndarray, float]:
        """Run GEMM C = A @ B and return (C, elapsed_ms).

        Arrays must be fp16/bf16/fp8/int8 numpy arrays allocated on the GPU
        (via e.g. ``hip.malloc`` or ``torch.cuda.FloatTensor``).  For testing
        on host, pass CPU arrays — the binding forwards raw pointers; GPU access
        faults at kernel launch are propagated as DispatcherError.

        Layout: A is row-major (M×K), B is column-major (K×N), C is row-major.
        """
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("a and b must be 2-D arrays")
        M, K = a.shape
        K2, N = b.shape
        if K != K2:
            raise ValueError(f"Shape mismatch: a is {M}x{K}, b is {K2}x{N}")

        out_dtype = {
            np.float16: np.float16,
            np.float32: np.float32,
            np.int8:    np.int8,
        }.get(a.dtype.type, np.float16)
        c = np.zeros((M, N), dtype=out_dtype)

        elapsed = ctypes.c_float(0.0)
        stream_ptr = ctypes.c_void_p(stream if stream is not None else 0)

        _check(
            self._lib.dispatcher_run_gemm(
                handle,
                M, N, K,
                a.ctypes.data_as(ctypes.c_void_p),
                b.ctypes.data_as(ctypes.c_void_p),
                c.ctypes.data_as(ctypes.c_void_p),
                K,   # stride_a (row-major)
                K,   # stride_b (col-major: B[k,n] = ptr[n*K + k])
                N,   # stride_c (row-major)
                split_k,
                stream_ptr,
                ctypes.byref(elapsed),
            ),
            f"run_gemm(handle={handle}, {M}x{N}x{K})",
        )
        return c, elapsed.value

    def tflops(self, M: int, N: int, K: int, elapsed_ms: float) -> float:
        """Compute TFLOP/s from problem dimensions and elapsed time."""
        return 2.0 * M * N * K / (elapsed_ms * 1e-3) / 1e12
