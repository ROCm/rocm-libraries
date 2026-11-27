# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import ctypes
import struct

import numpy as np
import pytest
from rocke.instances.common.f64_mma_probe import F64MmaProbeSpec, build_f64_mma_probe
from rocke.tests.arch_marks import arch_specific
from rocke.tests.engines import available_engines, lower_with_engine
from rocke.tests.gpu import current_device_arch_in

pytestmark = pytest.mark.gpu

# The targets whose MMA catalog lists both f64 atoms and on whose hardware this
# test has passed. Written out rather than read from the catalog: a target that
# joins the catalog joins this test only once it has passed on that hardware.
_F64_MMA_ARCHES = ("gfx90a", "gfx942", "gfx950")


def _launch(ll, arch, name, a, b, c):
    from rocke.runtime.comgr import build_hsaco_from_llvm_ir
    from rocke.runtime.hip_module import Runtime

    blob, _ = build_hsaco_from_llvm_ir(ll, isa=f"amdgcn-amd-amdhsa--{arch}")
    runtime = Runtime()
    module = runtime.load_module(blob)
    arrays = [a.copy(), b.copy(), c.copy(), np.full_like(c, np.nan)]
    ptrs = []
    try:
        for array in arrays:
            ptr = runtime.alloc(array.nbytes)
            ptrs.append(ptr)
            runtime.memcpy_h2d(
                ptr, (ctypes.c_ubyte * array.nbytes).from_buffer(array), array.nbytes
            )
        runtime.launch_blocking(
            module.get_function(name), (1, 1, 1), (64, 1, 1), struct.pack("4Q", *ptrs)
        )
        d = arrays[3]
        runtime.memcpy_d2h(
            (ctypes.c_ubyte * d.nbytes).from_buffer(d), ptrs[3], d.nbytes
        )
        return d
    finally:
        for ptr in ptrs:
            runtime.free(ptr)
        module.unload()


def _shapes(m, k):
    if m == 16:
        return (16, k), (k, 16), (16, 16)
    # 4x4x4: four blocks stacked along rows; B is k/4 slabs of 16 x 4.
    return (16, k), (k // 4, 16, 4), (16, 4)


def _reference(m, a, b, c):
    """``a @ b + c`` per block, in the probe's memory layout."""
    if m == 16:
        return a @ b + c
    k = a.shape[1]
    a_blk = a.reshape(4, 4, k)
    b_blk = b.reshape(k // 4, 4, 4, 4).transpose(1, 0, 2, 3).reshape(4, k, 4)
    return (a_blk @ b_blk + c.reshape(4, 4, 4)).reshape(16, 4)


@arch_specific(
    _F64_MMA_ARCHES,
    reason="the f64 MMA probe builds only for the targets whose catalog lists the f64 atoms",
)
@pytest.mark.parametrize("m", [16, 4])
@pytest.mark.parametrize("k", [4, 8, 64])
def test_f64_mma_numeric(m, k):
    arch = current_device_arch_in(_F64_MMA_ARCHES)
    kernel = build_f64_mma_probe(F64MmaProbeSpec(m, k), arch)
    rng = np.random.default_rng(m * 100 + k)
    shapes = _shapes(m, k)
    exact = [rng.integers(-8, 9, shape).astype(np.float64) for shape in shapes]
    real = [rng.standard_normal(shape) for shape in shapes]
    for engine in available_engines():
        ll = lower_with_engine(kernel, arch, engine=engine)
        a, b, c = exact
        np.testing.assert_array_equal(
            _launch(ll, arch, kernel.name, a, b, c), _reference(m, a, b, c)
        )
        a, b, c = real
        d = _launch(ll, arch, kernel.name, a, b, c)
        scale = _reference(m, np.abs(a), np.abs(b), np.abs(c))
        error = np.abs(d - _reference(m, a, b, c)) / scale
        assert np.max(error) < k * np.finfo(np.float64).eps
