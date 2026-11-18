# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Real-GPU checks for ``IRBuilder.warp_shuffle_xor_dpp`` on one wave64 of
gfx9 CDNA lanes, where each mask lowers to a base-DPP pattern (quad_perm for
1, 2, 3; row_half_mirror for 7; row_ror:8 for 8; row_mirror for 15).

Each mask alone must move lane ``i`` to read lane ``i ^ mask`` (i32 and f32),
and XOR butterflies whose masks span all four lane-index bits must leave the
16-lane row sum in every lane. Both engines lower the kernel; ``cpp`` needs
the binding. Machines without a gfx942 or gfx950 device skip.
"""

import ctypes
import importlib.util
import struct

import numpy as np
import pytest

from rocke.core.ir import F32, I32, IRBuilder, PtrType
from rocke.helpers import compile_kernel
from rocke.runtime.hip_module import Runtime, get_device_arch

_WAVE = 64
_GFX9_MASKS = (1, 2, 3, 7, 8, 15)


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


def _kernel(name, body, dtype=I32):
    b = IRBuilder(name)
    src = b.param("S", PtrType(dtype, "global"), noalias=True, readonly=True)
    dst = b.param("D", PtrType(dtype, "global"), noalias=True)
    tid = b.thread_id_x()
    b.global_store(dst, tid, body(b, b.global_load(src, tid, dtype, align=4)), align=4)
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
        return np.frombuffer(bytes(out), dtype=values.dtype)
    finally:
        rt.free(src)
        rt.free(dst)
        module.unload()


@pytest.mark.parametrize("mask", _GFX9_MASKS)
def test_mask_reads_xor_lane(arch, backend, mask):
    # Lane-id inputs: lane i must read lane i ^ mask, in every 16-lane row.
    kernel = _kernel(
        f"wsx_dpp_lanes_{mask}", lambda b, v: b.warp_shuffle_xor_dpp(v, mask)
    )
    lanes = np.arange(_WAVE, dtype=np.int32)
    np.testing.assert_array_equal(_run(kernel, lanes, arch, backend), lanes ^ mask)


@pytest.mark.parametrize("mask", _GFX9_MASKS)
def test_mask_moves_f32_bits(arch, backend, mask):
    # f32 goes through an i32 bitcast; the bits must arrive unchanged.
    kernel = _kernel(
        f"wsx_dpp_f32_{mask}", lambda b, v: b.warp_shuffle_xor_dpp(v, mask), F32
    )
    values = np.random.default_rng(mask).standard_normal(_WAVE).astype(np.float32)
    got = _run(kernel, values, arch, backend)
    np.testing.assert_array_equal(got, values[np.arange(_WAVE) ^ mask])


@pytest.mark.parametrize(
    "masks",
    [(1, 2, 7, 15), (1, 2, 8, 7), (15, 7, 2, 1)],
    ids=["1-2-7-15", "1-2-8-7", "15-7-2-1"],
)
def test_butterfly_leaves_row_sum_in_every_lane(arch, backend, masks):
    def body(b, acc):
        for m in masks:
            acc = b.add(acc, b.warp_shuffle_xor_dpp(acc, m))
        return acc

    kernel = _kernel("wsx_dpp_row_sum_" + "_".join(map(str, masks)), body)
    values = np.random.default_rng(7).integers(-1000, 1000, _WAVE, dtype=np.int32)
    expected = np.repeat(values.reshape(-1, 16).sum(axis=1), 16).astype(np.int32)
    np.testing.assert_array_equal(_run(kernel, values, arch, backend), expected)


def test_unencodable_mask_is_rejected(arch, backend):
    # Mask 4 has no register-only encoding on gfx9: lowering must fail loudly
    # in both engines rather than emit a row_xmask control the GPU lacks.
    kernel = _kernel("wsx_dpp_mask4", lambda b, v: b.warp_shuffle_xor_dpp(v, 4))
    with pytest.raises(Exception, match="no register-only DPP encoding"):
        compile_kernel(kernel, arch=arch, backend=backend)
