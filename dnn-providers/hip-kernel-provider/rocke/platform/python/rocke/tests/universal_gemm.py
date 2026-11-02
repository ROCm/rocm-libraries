# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test helper: lower a universal GEMM spec with one engine and run it on the GPU.

Numeric GEMM tests lower the same spec with each engine (see
``rocke.tests.engines.available_engines``) and launch each result, so a defect in
either engine shows up as a wrong answer on hardware. The target arch is a required
keyword: a code object built for one arch must not be launched on another.
"""

from __future__ import annotations

import ctypes
import struct

import numpy as np

from rocke.core.backend import universal_gemm_spec_to_dict
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.instances.common.gemm_universal import (
    UniversalGemmSpec,
    build_universal_gemm,
)

__all__ = ["launch_universal_gemm", "lower_universal_gemm"]


def lower_universal_gemm(spec: UniversalGemmSpec, engine: str, *, arch: str) -> str:
    """The LLVM IR ``engine`` (``"python"`` or ``"cpp"``) emits for ``spec`` on
    ``arch``. Raises ``ValueError`` for any other engine name."""
    if engine == "python":
        kernel = build_universal_gemm(spec, arch=arch)
        return _lower_kernel_to_llvm_python(kernel, arch=arch)
    if engine != "cpp":
        raise ValueError(f"engine must be 'python' or 'cpp', got {engine!r}")
    import rocke_engine as native

    return native.gemm_lower_llvm(universal_gemm_spec_to_dict(spec), arch=arch)


def launch_universal_gemm(
    llvm_ir: str,
    spec: UniversalGemmSpec,
    a: np.ndarray,
    bt: np.ndarray,
    *,
    arch: str,
    b_device: np.ndarray | None = None,
    sorted_ids: np.ndarray | None = None,
    c: np.ndarray | None = None,
) -> np.ndarray:
    """Build ``llvm_ir`` for ``arch``, run it on the current GPU and return C.

    ``a`` is ``[batch,] M x K`` and ``bt`` is ``[batch,] N x K`` (B transposed).
    ``b_device`` replaces ``bt`` in device memory when the kernel reads B in another
    layout (a preshuffled B). ``sorted_ids`` adds the active-tile-skip arguments.
    ``c`` is C's initial contents; it defaults to NaN so a tile the kernel never
    writes is visible.
    """
    from rocke.runtime.comgr import build_hsaco_from_llvm_ir
    from rocke.runtime.hip_module import Runtime

    *batch, m, k = a.shape
    n = bt.shape[-2]
    batches = batch[0] if batch else 1
    blob, _ = build_hsaco_from_llvm_ir(llvm_ir, isa=f"amdgcn-amd-amdhsa--{arch}")
    runtime = Runtime()
    module = runtime.load_module(blob)
    if c is None:
        c = np.full((*batch, m, n), np.nan)
    arrays = [
        a.copy(),
        (bt if b_device is None else b_device).copy(),
        c.copy(),
    ]
    if sorted_ids is not None:
        arrays.append(sorted_ids.copy())
    ptrs = []
    try:
        for array in arrays:
            ptr = runtime.alloc(array.nbytes)
            ptrs.append(ptr)
            runtime.memcpy_h2d(
                ptr, (ctypes.c_ubyte * array.nbytes).from_buffer(array), array.nbytes
            )
        tile = spec.tile
        grid = (-(-n // tile.tile_n), -(-m // tile.tile_m), batches)
        if sorted_ids is None:
            args = struct.pack("3Q3i", *ptrs, m, n, k)
        else:
            args = struct.pack(
                "3Q6iQi", *ptrs[:3], m, n, k, m * k, n * k, m * n, ptrs[3], m
            )
        runtime.launch_blocking(
            module.get_function(spec.kernel_name()), grid, (spec.block_size, 1, 1), args
        )
        c = arrays[2]
        runtime.memcpy_d2h(
            (ctypes.c_ubyte * c.nbytes).from_buffer(c), ptrs[2], c.nbytes
        )
        return c
    finally:
        for ptr in ptrs:
            runtime.free(ptr)
        module.unload()
