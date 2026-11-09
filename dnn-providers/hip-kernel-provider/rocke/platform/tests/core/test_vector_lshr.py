# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``vector.lshr`` is a logical shift in the element's own width.

LLVM's ``lshr <N x iW>`` shifts zeros in at bit ``W - 1``. The HIP lowering
reads each lane through the unsigned type of its width before shifting, so a
negative i8 or i16 lane does not shift its sign bits in. Covers the HIP source
of both engines, that hipcc accepts it, and, when a GPU is present, that the
LLVM route and hipcc on the HIP source both engines emit compute the same
lanes as numpy, on negative and positive lanes alike.
"""

from __future__ import annotations

import ctypes
import shutil
import struct
import subprocess

import numpy as np
import pytest
from rocke.core.arch.target import (
    base_arch_from_target_id,
    compiler_target_from_target_id,
)
from rocke.core.ir import I8, I16, I32, IRBuilder, PtrType
from rocke.tests.engines import lower_hip_with_both_engines

_LANES = 64

# (element type, lanes per vector, unsigned C type, numpy dtype of the element)
CASES = (
    (I8, 4, "uint8_t", np.int8),
    (I16, 2, "uint16_t", np.int16),
    (I32, 2, "uint32_t", np.int32),
)


def _id(case):
    dtype, n, _, _ = case
    return f"{n}x{dtype.name}"


def _lshr_kernel(dtype, n):
    """Lane ``l`` loads ``n`` elements of ``X`` and of ``S`` at element
    ``l * n``, shifts each ``X`` lane right by its ``S`` lane and stores the
    result to ``Y`` at the same place."""
    b = IRBuilder(f"vector_lshr_{dtype.name}")
    b.kernel.attrs["max_workgroup_size"] = _LANES
    x, s, y = (b.param(name, PtrType(I8, "global")) for name in "XSY")
    nbytes = n * np.dtype(_numpy(dtype)).itemsize
    total = b.const_i32(_LANES * nbytes)
    offset = b.mul(b.thread_id_x(), b.const_i32(nbytes))
    zero = b.const_i32(0)
    value = b.buffer_load_vN(b.buffer_rsrc(x, total), offset, zero, dtype, n)
    shift = b.buffer_load_vN(b.buffer_rsrc(s, total), offset, zero, dtype, n)
    b.buffer_store_vN(
        b.buffer_rsrc(y, total), offset, zero, b.vector_lshr(value, shift)
    )
    b.ret()
    return b.kernel


def _numpy(dtype):
    return next(case[3] for case in CASES if case[0] == dtype)


@pytest.mark.parametrize("case", CASES, ids=_id)
def test_hip_reads_each_lane_through_its_unsigned_type(case):
    dtype, n, unsigned, _ = case
    hip = lower_hip_with_both_engines(_lshr_kernel(dtype, n), "gfx942")
    shifts = [line.strip() for line in hip.splitlines() if " >> " in line]
    assert len(shifts) == n
    assert all(f"= (({unsigned})" in line for line in shifts), shifts


def _hipcc(kernel, target_id, out_dir):
    """Compile the HIP source both engines emit for ``kernel`` with hipcc;
    return the code object."""
    hipcc = shutil.which("hipcc")
    if hipcc is None:
        pytest.skip("hipcc not found")
    hip = lower_hip_with_both_engines(kernel, base_arch_from_target_id(target_id))
    source = out_dir / f"{kernel.name}.hip"
    blob = out_dir / f"{kernel.name}.hsaco"
    source.write_text(hip)
    subprocess.run(
        [hipcc, f"--offload-arch={compiler_target_from_target_id(target_id)}",
         "--genco", "-O3", str(source), "-o", str(blob)],
        check=True,
        capture_output=True,
        timeout=240,
    )  # fmt: skip
    return blob.read_bytes()


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_hip_source_compiles(tmp_path, arch):
    for dtype, n, _, _ in CASES:
        assert _hipcc(_lshr_kernel(dtype, n), arch, tmp_path)


def _gpu_arch():
    try:
        from rocke.runtime.hip_module import get_device_arch
    except ImportError:
        return None
    try:
        return get_device_arch()
    except Exception:  # noqa: BLE001 - no HIP runtime means no GPU test
        return None


def _run(blob, kernel_name, x, s):
    """Launch one workgroup on ``x`` and ``s``; return ``Y`` as ``x``'s dtype."""
    from rocke.runtime.hip_module import Runtime

    runtime = Runtime()
    module = runtime.load_module(blob)
    out = np.zeros_like(x)
    buffers = [runtime.alloc(x.nbytes) for _ in range(3)]
    try:
        for device, host in zip(buffers, (x, s)):
            runtime.memcpy_h2d(
                device, (ctypes.c_ubyte * host.nbytes).from_buffer(host), host.nbytes
            )
        runtime.launch_blocking(
            module.get_function(kernel_name),
            (1, 1, 1),
            (_LANES, 1, 1),
            struct.pack("3Q", *buffers),
        )
        runtime.memcpy_d2h(
            (ctypes.c_ubyte * out.nbytes).from_buffer(out), buffers[2], out.nbytes
        )
    finally:
        for device in buffers:
            runtime.free(device)
        module.unload()
    return out


@pytest.mark.parametrize("route", ["llvm", "hipcc"])
@pytest.mark.parametrize("case", CASES, ids=_id)
def test_gpu_matches_a_logical_shift(tmp_path, case, route):
    arch = _gpu_arch()
    if not arch:
        pytest.skip("no GPU")
    dtype, n, _, np_dtype = case
    kernel = _lshr_kernel(dtype, n)
    if route == "llvm":
        from rocke.helpers.compile import compile_kernel

        blob = compile_kernel(kernel, arch=arch).hsaco
    else:
        blob = _hipcc(kernel, arch, tmp_path)
    bits = np.dtype(np_dtype).itemsize * 8
    unsigned = np.dtype(f"u{bits // 8}")
    rng = np.random.default_rng(11)
    raw = rng.integers(0, 256, _LANES * n * bits // 8, dtype=np.uint8)
    x = raw.view(np_dtype).copy()
    x.view(unsigned)[0] = 1 << (bits - 1)  # the top bit alone, in every type
    assert (x < 0).any() and (x > 0).any()
    s = rng.integers(0, bits, x.size).astype(np_dtype)
    expected = (x.view(unsigned) >> s.astype(unsigned)).view(np_dtype)
    np.testing.assert_array_equal(_run(blob, kernel.name, x, s), expected)
