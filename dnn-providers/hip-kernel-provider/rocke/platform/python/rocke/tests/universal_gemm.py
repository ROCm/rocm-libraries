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
from rocke.core.dtypes import dtype_info
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.instances.common.gemm_universal import (
    UniversalGemmSpec,
    build_universal_gemm,
)
from rocke.numeric.bf16 import f32_to_bf16

__all__ = [
    "launch_universal_gemm",
    "lower_universal_gemm",
    "universal_gemm_arguments",
    "universal_gemm_nan_output",
]

# The host array type of each storage dtype a universal GEMM reads and writes;
# bf16, which numpy lacks, is carried as its uint16 encoding.
_HOST_DTYPE = {
    "fp64": np.float64,
    "fp32": np.float32,
    "fp16": np.float16,
    "bf16": np.uint16,
}


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


def universal_gemm_nan_output(
    spec: UniversalGemmSpec, shape: tuple[int, ...]
) -> np.ndarray:
    """A ``shape`` C buffer of NaN in ``spec``'s storage dtype, so a tile the
    kernel never writes stays visible: ``float16`` NaN for fp16, the bf16 NaN
    encoding (``0x7FC0``) in ``uint16`` for bf16.

    Raises ``ValueError`` for a split-K spec, whose C is an fp32 workspace the
    caller must zero and cast, and for a dtype with no host array type here.
    """
    if spec.trait.split_k > 1:
        raise ValueError(
            f"split-K GEMM (split_k={spec.trait.split_k}) accumulates into a zeroed "
            "fp32 workspace; launch_universal_gemm runs only non-split-K kernels"
        )
    dtype = dtype_info(spec.data.dtype_c).name
    if dtype not in _HOST_DTYPE:
        raise ValueError(
            f"no host array type for C dtype {spec.data.dtype_c!r}; expected one of "
            f"{', '.join(_HOST_DTYPE)}"
        )
    if dtype == "bf16":
        return f32_to_bf16(np.full(shape, np.nan, dtype=np.float32))
    return np.full(shape, np.nan, dtype=_HOST_DTYPE[dtype])


def universal_gemm_arguments(
    spec: UniversalGemmSpec,
    pointers: tuple[int, ...],
    m: int,
    n: int,
    k: int,
) -> bytes:
    """The kernel-argument bytes of ``spec``'s kernel, in the order it declares them.

    ``pointers`` are the device addresses of A, B and C, then of the sorted
    token ids when the kernel skips inactive tiles. Every kernel takes A, B, C,
    M, N and K; a batched one adds the per-batch element strides of A, B and C
    (``M*K``, ``N*K``, ``M*N``); a batched one that skips inactive tiles adds
    the sorted token ids and the slot size (``M``).

    Raises ``ValueError`` when the sorted token ids are passed to a kernel that
    does not take them, or missing for one that does.
    """
    _require_sorted_ids_iff_taken(spec, len(pointers) == 4)
    layout, values = "3Q3i", [*pointers[:3], m, n, k]
    if spec.batched:
        layout += "3i"
        values += [m * k, n * k, m * n]
    if _skips_inactive_tiles(spec):
        layout += "Qi"
        values += [pointers[3], m]
    return struct.pack(layout, *values)


def _skips_inactive_tiles(spec: UniversalGemmSpec) -> bool:
    """Whether ``spec``'s kernel reads sorted token ids (the builder declares
    them for a batched kernel that skips inactive tiles)."""
    return spec.batched and spec.trait.active_tile_skip


def _require_sorted_ids_iff_taken(
    spec: UniversalGemmSpec, has_sorted_ids: bool
) -> None:
    """Refuse sorted token ids for a kernel that does not read them, and their
    absence for one that does."""
    if has_sorted_ids and not _skips_inactive_tiles(spec):
        raise ValueError(
            f"{spec.kernel_name()} does not read sorted token ids; pass sorted_ids "
            "only for a batched kernel with active_tile_skip"
        )
    if _skips_inactive_tiles(spec) and not has_sorted_ids:
        raise ValueError(
            f"{spec.kernel_name()} skips inactive tiles and reads sorted token ids; "
            "pass sorted_ids"
        )


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
    layout (a preshuffled B). ``sorted_ids`` is the sorted token ids a kernel
    that skips inactive tiles reads, and is required exactly for such a kernel.
    ``c`` is C's initial contents; it defaults to
    :func:`universal_gemm_nan_output`. The arguments are
    :func:`universal_gemm_arguments`.
    """
    from rocke.runtime.comgr import build_hsaco_from_llvm_ir
    from rocke.runtime.hip_module import Runtime

    _require_sorted_ids_iff_taken(spec, sorted_ids is not None)
    *batch, m, k = a.shape
    n = bt.shape[-2]
    batches = batch[0] if batch else 1
    if c is None:
        c = universal_gemm_nan_output(spec, (*batch, m, n))
    blob, _ = build_hsaco_from_llvm_ir(llvm_ir, isa=f"amdgcn-amd-amdhsa--{arch}")
    runtime = Runtime()
    module = runtime.load_module(blob)
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
        args = universal_gemm_arguments(spec, tuple(ptrs), m, n, k)
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
