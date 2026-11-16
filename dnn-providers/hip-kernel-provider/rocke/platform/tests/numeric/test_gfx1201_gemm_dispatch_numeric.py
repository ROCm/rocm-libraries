# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Real FP16 dispatch selections against exact independent GEMM references.

The shapes exercise the direct-A classes (including M and N that are not
whole tiles and K that is not), the compv4 output/reduction bands it leaves,
short K, 16 KiB row strides and non-128/64 alignment. Integer inputs keep
accumulation exact; only C's fp16 store rounds. NaN-filled C exposes tiles a
selected kernel never writes, and NaN in the row padding of a strided request
exposes a read that strays into it.
"""

import numpy as np
import pytest

from rocke.dispatch import GemmRequest, dispatch_gemm_fp16
from rocke.tests.arch_marks import arch_specific
from rocke.tests.engines import available_engines
from rocke.tests.gpu import current_device_arch_in
from rocke.tests.universal_gemm import launch_universal_gemm, lower_universal_gemm


@arch_specific(("gfx1201",), reason="the measured FP16 compv4 palette targets gfx1201")
@pytest.mark.parametrize(
    "shape",
    (
        (512, 512, 512),
        (896, 896, 1024),
        (1024, 1024, 512),
        (128, 8192, 2048),
        (2048, 2560, 6144),
        (2048, 6144, 2560),
        (2048, 2048, 8192),
        (96, 1024, 128),
        (1024, 96, 128),
        (160, 160, 128),
        (256, 256, 48),
        (128, 128, 16),
        (256, 256, 32),
        (1000, 1000, 1024),
        (3000, 1500, 512),
        (8192, 128, 2048),
        (768, 1536, 2048),
        (1024, 1024, 8192),
        (704, 704, 8192),
        (256, 256, 8192),
        (300, 400, 640),
        (130, 128, 32),
        (77, 33, 16),
        (160, 160, 1024),
        (704, 704, 2048),
        (3072, 128, 4096),
        (2048, 2048, 1040),
        (1280, 1280, 10240),
        (480, 480, 8192),
        (800, 800, 8192),
    ),
    ids=lambda shape: "x".join(map(str, shape)),
)
def test_gfx1201_dispatched_gemm_matches_exact_reference(shape):
    arch = current_device_arch_in(("gfx1201",))
    result = dispatch_gemm_fp16(GemmRequest(*shape, arch=arch))
    m, n, k = shape
    rng = np.random.default_rng(np.random.SeedSequence([1201, m, n, k]))
    a_int = rng.integers(-2, 3, (m, k), dtype=np.int16)
    bt_int = rng.integers(-2, 3, (n, k), dtype=np.int16)
    expected = (a_int.astype(np.float64) @ bt_int.astype(np.float64).T).astype(
        np.float16
    )
    a, bt = a_int.astype(np.float16), bt_int.astype(np.float16)
    for engine in available_engines():
        llvm_ir = lower_universal_gemm(result.spec, engine, arch=arch)
        c = np.full((m, n), np.nan, dtype=np.float16)
        got = launch_universal_gemm(llvm_ir, result.spec, a, bt, arch=arch, c=c)
        np.testing.assert_array_equal(got, expected, err_msg=engine)


def _padded(values, stride):
    rows, k = values.shape
    out = np.full((rows, stride), np.nan, dtype=np.float16)
    out[:, :k] = values
    return out


@arch_specific(("gfx1201",), reason="row strides are a gfx1201 direct-A argument")
@pytest.mark.parametrize(
    "shape, lda, ldb",
    (
        ((512, 512, 256), 288, 256),
        ((300, 400, 640), 640, 656),
        ((1024, 1024, 8192), 8224, 8224),
        ((1000, 1000, 8192), 8192, 8224),
        ((1280, 1280, 10256), 10288, 10288),
    ),
    ids=lambda v: "x".join(map(str, v)) if isinstance(v, tuple) else str(v),
)
def test_gfx1201_dispatched_strided_gemm_matches_exact_reference(shape, lda, ldb):
    arch = current_device_arch_in(("gfx1201",))
    result = dispatch_gemm_fp16(GemmRequest(*shape, arch=arch, lda=lda, ldb=ldb))
    assert result.spec.trait.row_strides
    m, n, k = shape
    rng = np.random.default_rng(np.random.SeedSequence([1201, m, n, k, lda]))
    a_int = rng.integers(-2, 3, (m, k), dtype=np.int16)
    bt_int = rng.integers(-2, 3, (n, k), dtype=np.int16)
    expected = (a_int.astype(np.float64) @ bt_int.astype(np.float64).T).astype(
        np.float16
    )
    a, bt = _padded(a_int, lda), _padded(bt_int, ldb)
    for engine in available_engines():
        llvm_ir = lower_universal_gemm(result.spec, engine, arch=arch)
        c = np.full((m, n), np.nan, dtype=np.float16)
        got = launch_universal_gemm(llvm_ir, result.spec, a, bt, arch=arch, c=c, k=k)
        np.testing.assert_array_equal(got, expected, err_msg=engine)
