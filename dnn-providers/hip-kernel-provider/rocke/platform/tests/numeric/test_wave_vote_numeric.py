# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Real-GPU checks for ``IRBuilder.wave_all`` / ``wave_any`` on one full wave
of the detected device (wave32 on RDNA, wave64 on CDNA).

Each lane votes its input; every lane must receive the wave-wide AND/OR.
The cases where every lane votes true pin the wave32 full-mask comparison
(``ballot.i64`` sets only the low 32 bits there). Both engines lower the
kernel; ``cpp`` needs the binding. Machines without a HIP device skip.
"""

import ctypes
import importlib.util
import struct

import numpy as np
import pytest

from rocke.core.arch import ArchTarget
from rocke.core.ir import I32, IRBuilder, PtrType
from rocke.helpers import compile_kernel
from rocke.runtime.hip_module import Runtime, get_device_arch


@pytest.fixture(scope="module")
def arch():
    try:
        target = get_device_arch(0)
    except Exception as exc:  # noqa: BLE001 - no HIP runtime or device
        pytest.skip(f"requires a HIP device: {exc}")
    try:
        ArchTarget.from_gfx(target)
    except KeyError:
        pytest.skip(f"no rocKE arch target for {target}")
    return target


@pytest.fixture(params=("python", "cpp"))
def backend(request):
    if request.param == "cpp" and importlib.util.find_spec("rocke_engine") is None:
        pytest.skip("cpp engine needs the rocke_engine binding")
    return request.param


def _vote_kernel(vote):
    b = IRBuilder(f"wave_vote_{vote}")
    src = b.param("S", PtrType(I32, "global"), noalias=True, readonly=True)
    dst = b.param("D", PtrType(I32, "global"), noalias=True)
    tid = b.thread_id_x()
    pred = b.global_load(src, tid, I32, align=4)
    b.global_store(dst, tid, getattr(b, vote)(pred), align=4)
    b.ret()
    return b.kernel


def _run(vote, values, arch, backend):
    wave = ArchTarget.from_gfx(arch).wave_size
    art = compile_kernel(_vote_kernel(vote), arch=arch, backend=backend)
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
            (wave, 1, 1),
            struct.pack("<QQ", src, dst),
        )
        rt.sync()
        rt.memcpy_d2h(out, dst, nbytes)
        return np.frombuffer(bytes(out), dtype=values.dtype)
    finally:
        rt.free(src)
        rt.free(dst)
        module.unload()


def _votes(arch, case):
    wave = ArchTarget.from_gfx(arch).wave_size
    values = np.ones(wave, dtype=np.int32)
    if case == "one_false_first":
        values[0] = 0
    elif case == "one_false_last":
        values[-1] = 0
    elif case == "all_false":
        values[:] = 0
    elif case == "nonzero_true":
        values = np.arange(1, wave + 1, dtype=np.int32) * -3
    return values


_CASES = ("all_true", "nonzero_true", "one_false_first", "one_false_last", "all_false")


@pytest.mark.parametrize("case", _CASES)
def test_wave_all_is_wave_wide_and(arch, backend, case):
    values = _votes(arch, case)
    expected = np.full(values.shape, int(bool(np.all(values != 0))), dtype=np.int32)
    np.testing.assert_array_equal(_run("wave_all", values, arch, backend), expected)


@pytest.mark.parametrize("case", _CASES)
def test_wave_any_is_wave_wide_or(arch, backend, case):
    values = _votes(arch, case)
    expected = np.full(values.shape, int(bool(np.any(values != 0))), dtype=np.int32)
    np.testing.assert_array_equal(_run("wave_any", values, arch, backend), expected)
