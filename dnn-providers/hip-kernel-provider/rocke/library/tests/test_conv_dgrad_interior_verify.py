# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# benchmark_implicit_gemm_conv._dxi_shell_bad: the dx_interior verify's shell
# check holds in every output dtype (the sentinel is compared bytewise).
# Host-only.

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from benchmarks.common.benchmark_implicit_gemm_conv import (
    _DXI_SENTINEL_BYTE,
    _dxi_shell_bad,
)


def _filled(dtype, shape=(1, 4, 4, 4, 8)):
    t = torch.empty(shape, dtype=dtype)
    t.view(torch.uint8).fill_(_DXI_SENTINEL_BYTE)
    return t


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_untouched_shell_passes_in_every_dtype(dtype):
    out = _filled(dtype)
    halo = torch.ones(out.shape[:-1], dtype=torch.bool)
    halo[:, 1:3, 1:3, 1:3] = False
    out[~halo] = 1.5  # the interior is written
    assert _dxi_shell_bad(out, halo, zero=False) == 0
    out[0, 0, 0, 0, 3] = 2.0  # one shell voxel written
    assert _dxi_shell_bad(out, halo, zero=False) == 1


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_zero_fill_shell(dtype):
    out = torch.zeros((1, 4, 4, 4, 8), dtype=dtype)
    halo = torch.ones(out.shape[:-1], dtype=torch.bool)
    halo[:, 1:3, 1:3, 1:3] = False
    out[~halo] = 1.5
    assert _dxi_shell_bad(out, halo, zero=True) == 0
    out[0, 3, 3, 3, 0] = 1.0
    assert _dxi_shell_bad(out, halo, zero=True) == 1
