# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 MXFP8 non-unity scaled MFMA numerical cases."""

import pytest


@pytest.mark.parametrize("dtype", ["fp8", "bf8"])
@pytest.mark.parametrize("output", ["bf16", "fp16"])
@pytest.mark.parametrize("route", ["comgr", "hip"])
@pytest.mark.parametrize("M,N,K", [(32, 64, 128), (64, 96, 256)])
def test_numeric(gfx950_numeric, dtype, output, route, M, N, K):
    gfx950_numeric(dtype, dtype, output, route, M, N, K)
