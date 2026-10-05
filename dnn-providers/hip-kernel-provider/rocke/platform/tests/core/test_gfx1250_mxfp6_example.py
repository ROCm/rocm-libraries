# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The six-bit example defaults to both homogeneous encodings."""

import pytest
from rocke.examples.gfx1250.gemm import _scaled_gemm_example, mxfp6_gemm


@pytest.mark.parametrize(
    "argv,dtypes",
    [
        (["--dtype", "fp6e2m3"], ("fp6e2m3",)),
        (["--dtype", " FP6 "], ("fp6e2m3",)),
        (["--dtype", "fp6e3m2"], ("fp6e3m2",)),
        (["--dtype", " BF6 "], ("fp6e3m2",)),
        ([], ("fp6e2m3", "fp6e3m2")),
        (["--dtype", "both"], ("fp6e2m3", "fp6e3m2")),
        (["--dtype", "fp6"], ("fp6e2m3",)),
        (["--dtype", "bf6"], ("fp6e3m2",)),
    ],
)
def test_mxfp6_cli_selects_formats(monkeypatch, argv, dtypes):
    calls = []

    def run(spec, cases, **kwargs):
        calls.append((spec, cases, kwargs))
        return len(cases)

    monkeypatch.setattr(_scaled_gemm_example, "run_cases", run)
    assert (
        mxfp6_gemm.main(
            argv
            + [
                "--matrix-path",
                "wmma_scale16",
                "--compile-route",
                "hip",
                "--case",
                "all",
            ]
        )
        == 0
    )
    assert tuple(spec.dtype_a for spec, _, _ in calls) == dtypes
    for spec, cases, kwargs in calls:
        assert spec.dtype_a == spec.dtype_b
        assert spec.block_k == 16 and len(cases) == 20
        assert kwargs == {"compile_route": "hip"}
