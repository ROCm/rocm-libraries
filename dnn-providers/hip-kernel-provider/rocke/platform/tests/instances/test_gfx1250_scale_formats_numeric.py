# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 independent-scale fixtures and fixed-byte selector checks."""

from itertools import product
import subprocess
import sys

import pytest

from rocke.core.arch.target import _gfx1250_scale_combination_valid

_MODULE = "rocke.examples.gfx1250.gemm.scale_formats_gemm"

# Independent scale encodings through both COMGR and HIP.
_FORMATS = ("fp8e4m3", "bf8e5m2", "fp6", "bf6", "fp4")
_MIXED = [
    (a, b, sa, sb)
    for a, b, sa, sb in product(
        _FORMATS, _FORMATS, ("e8m0", "e5m3", "e4m3"), ("e8m0", "e5m3", "e4m3")
    )
    if (sa, sb) != ("e8m0", "e8m0") and _gfx1250_scale_combination_valid(a, b, sa, sb)
]


@pytest.mark.parametrize("matrix_path", ["wmma_scale", "wmma_scale16"])
@pytest.mark.parametrize("output_dtype", ["bf16", "fp16"])
@pytest.mark.parametrize(
    "a,b,sa,sb,route",
    [(*case, route) for case in _MIXED for route in ("comgr", "hip")],
)
def test_scale_format_wmma_numeric(
    gpu_env, matrix_path, output_dtype, a, b, sa, sb, route
):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            _MODULE,
            "--output-dtype",
            output_dtype,
            "--dtype-a",
            a,
            "--dtype-b",
            b,
            "--scale-dtype-a",
            sa,
            "--scale-dtype-b",
            sb,
            "--matrix-path",
            matrix_path,
            "--compile-route",
            route,
            "--m",
            "32",
            "--n",
            "48",
            "--k",
            "256",
            "--case",
            "all",
        ],
        env=gpu_env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    compiled = [line for line in output.splitlines() if "] compiled " in line]
    assert len(compiled) == 1, output
    assert f"_out_{output_dtype}_" in compiled[0], output
    count = 20 if matrix_path == "wmma_scale16" else 12
    assert f"PASS: verified {count} cases" in output, output
    assert output.count("bad=0") == count, output
    print(output, end="")


@pytest.mark.parametrize("dtype", _FORMATS)
@pytest.mark.parametrize("operand", ["a", "b"])
@pytest.mark.parametrize("block_k", [32, 16])
@pytest.mark.parametrize("route", ["comgr", "hip"])
def test_scale_selector_fixed_bytes_numeric(gpu_env, dtype, operand, block_k, route):
    """Hold encoded inputs fixed while changing the tested operand's selector."""
    import numpy as np
    import ml_dtypes

    from rocke.examples.gfx1250.gemm.block_scaled_gemm_verify import (
        _launch,
        check_result,
        pack_fp6_codes,
    )
    from rocke.helpers import compile_kernel
    from rocke.helpers.compile import compile_kernel_via_hipcc
    from rocke.instances.gfx1250.block_scaled_gemm import (
        BlockScaledGemmSpec,
        build_block_scaled_gemm,
    )
    from rocke.runtime.hip_module import Runtime

    unity = {"e8m0": 0x7F, "e5m3": 0x78, "e4m3": 0x38}
    # Literal predictions avoid the production and verifier scale decoders.
    predictions = {
        0x3C: {"e8m0": 2.0**-67, "e5m3": 3 / 512, "e4m3": 1.5},
        0x7C: {"e8m0": 0.125, "e5m3": 1.5, "e4m3": 384.0},
    }
    positions = (0, 15, 16, 31, 32, 63, 64, 95, 96, 127)

    def matrix(kind, k):
        codes = np.zeros((16, 128), dtype=np.uint8)
        if kind == "fp4":
            codes[:, k] = 2
            return codes[:, ::2] | (codes[:, 1::2] << 4)
        if kind in ("fp6", "bf6"):
            codes[:, k] = 8 if kind == "fp6" else 12
            return pack_fp6_codes(codes)
        codes[:, k] = 0x38 if kind == "fp8e4m3" else 0x3C
        return codes

    rt = Runtime()
    for scale in unity:
        # Sweep alternate selectors only on FP4, with a non-FP4 E8M0 partner.
        # Non-FP4 operands still get the fixed-byte E8M0 control.
        partner = "fp8e4m3" if dtype == "fp4" else "fp4"
        da, db = (dtype, partner) if operand == "a" else (partner, dtype)
        sa, sb = (scale, "e8m0") if operand == "a" else ("e8m0", scale)
        if not _gfx1250_scale_combination_valid(da, db, sa, sb):
            continue
        spec = BlockScaledGemmSpec(
            name="fixed_scale_bytes",
            M=16,
            N=16,
            K=128,
            dtype_a=da,
            dtype_b=db,
            scale_dtype="e8m0",
            scale_dtype_a=sa,
            scale_dtype_b=sb,
            block_k=block_k,
            matrix_path="wmma_scale16" if block_k == 16 else "wmma_scale",
        )
        compiler = compile_kernel if route == "comgr" else compile_kernel_via_hipcc
        artifact = compiler(build_block_scaled_gemm(spec), arch="gfx1250")
        module = rt.load_module(artifact.hsaco)
        try:
            fn = module.get_function(artifact.kernel_name)
            for k, code in product(positions, predictions):
                a, b = matrix(da, k), matrix(db, k)
                a_scales = np.full((16, 128 // block_k), unity[sa], dtype=np.uint8)
                b_scales = np.full((128 // block_k, 16), unity[sb], dtype=np.uint8)
                # Every tested scale byte is identical across selector variants.
                (a_scales if operand == "a" else b_scales).fill(code)
                got = _launch(rt, fn, spec, (a, b, a_scales, b_scales))
                expected = np.full(
                    (16, 16), predictions[code][scale], dtype=ml_dtypes.bfloat16
                ).astype(np.float32)
                check_result(got, expected, exact=True)
        finally:
            module.unload()
