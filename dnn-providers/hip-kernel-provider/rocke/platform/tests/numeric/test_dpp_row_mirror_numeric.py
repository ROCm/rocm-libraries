# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Real-GPU checks for ``tile.dpp_row_mirror`` on one wave64 of CDNA lanes.

Each control word alone must move lanes as documented, and the
``xor 1, xor 2, half mirror, mirror`` butterfly must leave the 16-lane row sum
in every lane. Both engines lower the kernel; ``cpp`` needs the binding.
Machines without a gfx942 or gfx950 device skip.
"""

import ctypes
import importlib.util
import struct

import numpy as np
import pytest

from rocke.core.ir import I32, IRBuilder, PtrType
from rocke.helpers import compile_kernel
from rocke.runtime.hip_module import Runtime, get_device_arch

_WAVE = 64


@pytest.fixture(scope="module")
def arch():
    target = get_device_arch(0)
    if target not in ("gfx942", "gfx950"):
        pytest.skip(f"requires a gfx942 or gfx950 device; detected {target}")
    return target


@pytest.fixture(params=("python", "cpp"))
def backend(request):
    if request.param == "cpp" and importlib.util.find_spec("rocke_engine") is None:
        pytest.skip("cpp engine needs the rocke_engine binding")
    return request.param


def _kernel(name, body):
    b = IRBuilder(name)
    src = b.param("S", PtrType(I32, "global"), noalias=True, readonly=True)
    dst = b.param("D", PtrType(I32, "global"), noalias=True)
    tid = b.thread_id_x()
    b.global_store(dst, tid, body(b, b.global_load(src, tid, I32, align=4)), align=4)
    b.ret()
    return b.kernel


def _run(kernel, values, arch, backend):
    art = compile_kernel(kernel, arch=arch, backend=backend)
    rt = Runtime()
    module = rt.load_module(art.hsaco)
    nbytes = values.nbytes
    src = rt.alloc(nbytes)
    dst = rt.alloc(nbytes)
    try:
        host = (ctypes.c_uint8 * nbytes).from_buffer_copy(values.tobytes())
        rt.memcpy_h2d(src, host, nbytes)
        out = (ctypes.c_uint8 * nbytes)()
        rt.launch(
            module.get_function(art.kernel_name),
            (1, 1, 1),
            (_WAVE, 1, 1),
            struct.pack("<QQ", src, dst),
        )
        rt.sync()
        rt.memcpy_d2h(out, dst, nbytes)
        return np.frombuffer(bytes(out), dtype=np.int32)
    finally:
        rt.free(src)
        rt.free(dst)
        module.unload()


@pytest.mark.parametrize(
    "half,width", [(True, 8), (False, 16)], ids=["row_half_mirror", "row_mirror"]
)
def test_mirror_reads_documented_lane(arch, backend, half, width):
    # Lane-id inputs: lane i of each width-lane group reads lane width-1-i.
    kernel = _kernel(
        f"dpp_row_mirror_lanes_{width}", lambda b, v: b.dpp_row_mirror(v, half=half)
    )
    lanes = np.arange(_WAVE, dtype=np.int32)
    expected = (lanes & ~(width - 1)) | (width - 1 - (lanes & (width - 1)))
    np.testing.assert_array_equal(_run(kernel, lanes, arch, backend), expected)


def test_butterfly_leaves_row_sum_in_every_lane(arch, backend):
    def body(b, acc):
        acc = b.add(acc, b.warp_shuffle_xor_quad(acc, 1))
        acc = b.add(acc, b.warp_shuffle_xor_quad(acc, 2))
        acc = b.add(acc, b.dpp_row_mirror(acc, half=True))
        return b.add(acc, b.dpp_row_mirror(acc, half=False))

    kernel = _kernel("dpp_row_mirror_row_sum", body)
    values = np.random.default_rng(7).integers(-1000, 1000, _WAVE, dtype=np.int32)
    expected = np.repeat(values.reshape(-1, 16).sum(axis=1), 16).astype(np.int32)
    np.testing.assert_array_equal(_run(kernel, values, arch, backend), expected)
