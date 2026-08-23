# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Kernel-argument packing for the AMDGPU kernarg ABI.

Torch-agnostic by construction: imports only ``ctypes`` + ``struct`` and
duck-types device pointers on ``.data_ptr()``, so the numpy / manifest
(hip-only) path and the torch-tensor path share one packer. Two forms:

- :func:`pack_args` -> a single ``bytes`` blob for the
  ``HIP_LAUNCH_PARAM_BUFFER_POINTER`` ("extra") launch path.
- :func:`pack_args_kernelparams` -> a list of individual ``ctypes``
  scalars for the ``kernelParams`` launch path (driver-copied at enqueue).

Both respect the ABI's natural-alignment rule (8-byte ptr/i64, 4-byte
i32/f32) so a mixed ``(ptr ptr ptr i32 i32 i32 ptr)`` signature packs its
trailing pointer at the correct offset.
"""

from __future__ import annotations

import ctypes
import struct
from typing import Any, List, Mapping, Sequence, Tuple


def pack_args(
    signature: Sequence[Mapping[str, Any]], values: Mapping[str, Any]
) -> bytes:
    """Pack args from a manifest-style signature, respecting the AMDGPU
    kernarg ABI's natural-alignment rule.

    Each AMDGPU kernarg sits at an offset aligned to its own size:
    8-byte alignment for ``ptr`` / ``i64``, 4-byte alignment for
    ``i32`` / ``f32``. The flat ``struct.pack`` we used to use packed
    fields back-to-back with no inter-field padding, which is fine
    when the signature has only ptrs *or* only ints, but fails the
    instant a (ptr ptr ptr i32 i32 i32 ptr)-shaped signature shows
    up (e.g. a GEMM + bias-fused kernel) — the trailing pointer
    lands 4 bytes before its expected kernarg slot and the kernel
    reads garbage as the pointer, then segfaults on first access.

    Supported types: ``ptr<..., global>``, ``i32``, ``i64``, ``f32``.
    """
    # Map manifest type -> (struct format char, size in bytes, align).
    # The Python struct format is built with no implicit padding;
    # we insert explicit ``B`` bytes when alignment requires it.
    _TY_FMT: Mapping[str, Tuple[str, int, int]] = {
        "i32": ("i", 4, 4),
        "i64": ("q", 8, 8),
        "f32": ("f", 4, 4),
    }
    fmt_parts: List[str] = ["<"]
    packed: List[Any] = []
    offset = 0
    for arg in signature:
        name = str(arg["name"])
        ty = str(arg["type"])
        if name not in values:
            raise KeyError(f"missing kernel arg {name!r}")
        v = values[name]
        if ty.startswith("ptr<"):
            fmt_char, size, align = "Q", 8, 8
            arg_val: Any = _as_ptr(v)
        elif ty in _TY_FMT:
            fmt_char, size, align = _TY_FMT[ty]
            if ty == "f32":
                arg_val = float(v)
            else:
                arg_val = int(v)
        else:
            raise ValueError(f"unsupported kernel arg type {ty!r} for {name}")
        # Insert padding bytes so this arg lands at its natural alignment.
        pad = (-offset) % align
        if pad:
            fmt_parts.append(f"{pad}x")
            offset += pad
        fmt_parts.append(fmt_char)
        packed.append(arg_val)
        offset += size
    return struct.pack("".join(fmt_parts), *packed)


def pack_args_kernelparams(
    signature: Sequence[Mapping[str, Any]], values: Mapping[str, Any]
) -> List[Any]:
    """Pack args as a list of individual ``ctypes`` scalars for the
    ``kernelParams`` path of ``hipModuleLaunchKernel``.

    Returning one ``ctypes`` object per kernel argument lets the launcher
    build a ``void* params[]`` array whose entries point to each scalar.
    The CUDA/HIP semantics for ``kernelParams`` guarantee that the
    driver copies each parameter into driver-owned memory at enqueue
    time -- in contrast to the ``extra`` /
    ``HIP_LAUNCH_PARAM_BUFFER_POINTER`` path, whose copy semantics are
    underspecified and have been observed to read the host buffer
    *after* ``hipModuleLaunchKernel`` returns. The ``extra`` race
    produced the parity-harness "max_abs jumps to 2.5 / 512 on the
    second CK call" symptom investigated in
    ``ck/dsl/unified_attention_creative_results.md``.

    This is the same approach Triton's AMD driver uses (see
    ``triton/backends/amd/driver.py:392-402``: ``void *params[] = { ... };
    hipModuleLaunchKernel(..., params, 0);``).
    """
    out: List[Any] = []
    for arg in signature:
        name = str(arg["name"])
        ty = str(arg["type"])
        if name not in values:
            raise KeyError(f"missing kernel arg {name!r}")
        v = values[name]
        if ty.startswith("ptr<"):
            out.append(ctypes.c_uint64(_as_ptr(v)))
        elif ty == "i32":
            out.append(ctypes.c_int32(int(v)))
        elif ty == "i64":
            out.append(ctypes.c_int64(int(v)))
        elif ty == "f32":
            out.append(ctypes.c_float(float(v)))
        else:
            raise ValueError(f"unsupported kernel arg type {ty!r} for {name}")
    return out


def _as_ptr(v: Any) -> int:
    if isinstance(v, int):
        return v
    if hasattr(v, "data_ptr"):
        return int(v.data_ptr())
    # DeviceMem (RAII over hipMalloc, runtime.launcher) exposes its raw device
    # pointer via a ptr() method. Accepting it here lets the torch-free numpy /
    # manifest path pass a DeviceMem straight into a launcher's values dict, so
    # the async path's retain_for_stream() keeps the RAII buffer alive until the
    # stream drains -- closing the workspace-lifetime race by construction.
    ptr = getattr(v, "ptr", None)
    if callable(ptr):
        return int(ptr())
    if v is None:
        return 0
    raise TypeError(f"cannot convert {type(v).__name__} to device pointer")
