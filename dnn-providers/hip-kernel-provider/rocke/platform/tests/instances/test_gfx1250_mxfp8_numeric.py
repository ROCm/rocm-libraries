# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 MXFP8 numerical cases."""

import pytest


@pytest.mark.parametrize(
    "dtype,matrix_path,route,m,n,k,case,count",
    [
        (dtype, *case)
        for dtype in ("fp8", "bf8")
        for case in [
            ("wmma_scale", "comgr", 16, 16, 128, "all", 8),
            ("wmma_scale16", "comgr", 16, 16, 128, "all", 12),
            ("wmma_scale", "comgr", 32, 48, 256, "mixed", 1),
            ("wmma_scale16", "comgr", 32, 48, 256, "mixed", 1),
            ("wmma_scale", "hip", 32, 48, 256, "mixed", 1),
            ("wmma_scale16", "hip", 32, 48, 256, "mixed", 1),
            ("wmma", "comgr", 16, 16, 128, "mixed", 1),
        ]
        if dtype == "fp8" or case[0] != "wmma"
    ],
)
def test_numeric(numeric_case, dtype, matrix_path, route, m, n, k, case, count):
    numeric_case(dtype, matrix_path, route, m, n, k, case, count)
