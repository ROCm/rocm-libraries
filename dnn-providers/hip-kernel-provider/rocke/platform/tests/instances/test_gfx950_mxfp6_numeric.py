# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 MXFP6 non-unity scaled MFMA numerical cases."""

import pytest


@pytest.mark.parametrize("dtype", ["fp6", "bf6"])
@pytest.mark.parametrize("output", ["bf16", "fp16"])
@pytest.mark.parametrize("route", ["comgr", "hip"])
@pytest.mark.parametrize("M,N,K", [(32, 64, 128), (64, 96, 256)])
def test_numeric(gfx950_numeric, dtype, output, route, M, N, K):
    gfx950_numeric(dtype, dtype, output, route, M, N, K)


@pytest.mark.parametrize(
    "a,b", [("fp6", "fp6"), ("bf6", "bf6"), ("fp6", "bf6"), ("bf6", "fp6")]
)
@pytest.mark.parametrize("route", ["comgr", "hip"])
def test_all_fp6_codes(gfx950_numeric, a, b, route):
    positions = tuple(k for start in range(0, 128, 32) for k in (start, start + 31))
    gfx950_numeric(
        a,
        b,
        "bf16",
        route,
        64,
        64,
        128,
        tuple(f"codes-{k}" for k in positions),
    )
