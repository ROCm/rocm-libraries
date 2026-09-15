# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Verify packed NVFP4 GEMM with an independent block/tensor-scale reference.

Run with a gfx1250 device and LLVM 23 COMGR/HIP. The exact fixtures use bounded
dyadic block products; tensor-scale multiplication is explicitly rounded to
FP32 before the final BF16/FP16 conversion. No quantization policy is implied.
"""

from __future__ import annotations

import argparse

import numpy as np

from ....helpers import compile_kernel
from ....helpers.compile import compile_kernel_via_hipcc
from ....instances.gfx1250.nvfp4_gemm import (
    NvFp4GemmSpec,
    build_nvfp4_gemm,
    is_valid_spec,
)
from ....runtime.hip_module import Runtime, get_device_arch
from .block_scaled_gemm_verify import _launch, check_result, decode_fp4, pack_fp4_codes


def decode_e4m3_scales(encoded: np.ndarray) -> np.ndarray:
    """Decode finite nonnegative E4M3 bytes, including zero and subnormals."""
    if encoded.dtype != np.uint8 or np.any(encoded > 126):
        raise ValueError("expected finite nonnegative E4M3 scale bytes")
    exponent = (encoded >> 3).astype(np.int32)
    fraction = (encoded & 7).astype(np.float64) / 8
    return np.where(
        exponent == 0, np.ldexp(fraction, -6), np.ldexp(1 + fraction, exponent - 7)
    )


def reference_result(a, b, sa, sb, tensor_scales, *, dtype_c="bf16") -> np.ndarray:
    """Decode A/B, apply K=16 block scales, accumulate, then apply tensor scales.

    Tensor scales are multiplicative dequantization factors. The kernel computes
    their product in FP32 and multiplies the FP32 accumulator before conversion.
    Float64 accumulation here is an oracle for the exact fixtures, not a promise
    of bitwise agreement for arbitrary NVFP4 inputs.
    """
    import ml_dtypes

    da, db = decode_fp4(a), decode_fp4(b)
    m, k = da.shape
    n, kb = db.shape
    if k != kb or k % 16 or sa.shape != (m, k // 16) or sb.shape != (k // 16, n):
        raise ValueError("incompatible packed matrix or K=16 block-scale shapes")
    factors = np.asarray(tensor_scales, dtype=np.float32)
    if factors.shape != (2,) or not np.isfinite(factors).all():
        raise ValueError("expected two finite FP32 tensor dequantization factors")
    a_real = da * np.repeat(decode_e4m3_scales(sa), 16, axis=1)
    b_real = db * np.repeat(decode_e4m3_scales(sb).T, 16, axis=1)
    acc = (a_real @ b_real.T).astype(np.float32)
    result = acc * np.float32(factors[0] * factors[1])
    if dtype_c not in ("bf16", "fp16", "f16"):
        raise ValueError("output dtype must be bf16 or fp16")
    out_type = ml_dtypes.bfloat16 if dtype_c == "bf16" else np.float16
    return result.astype(out_type).astype(np.float32)


def make_case_inputs(spec: NvFp4GemmSpec, case: str):
    """Construct bounded encoded inputs without calling a quantizer."""
    if spec.K not in (128, 256):
        raise ValueError("exact fixtures require K=128 or K=256")
    rng = np.random.default_rng(0x4E16)
    a = rng.integers(0, 16, (spec.M, spec.K), dtype=np.uint8)
    b = rng.integers(0, 16, (spec.N, spec.K), dtype=np.uint8)
    # E4M3: 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2. Non-power scales catch E8M0 decoding.
    codes = np.array([48, 52, 56, 58, 60, 62, 64], dtype=np.uint8)
    sa = rng.choice(codes, (spec.M, spec.K // 16))
    sb = rng.choice(codes, (spec.K // 16, spec.N))
    factors = (1.0, 1.0)
    if case == "neutral":
        sa.fill(56)
        sb.fill(56)
    elif case == "a-only":
        sb.fill(56)
    elif case == "b-only":
        sa.fill(56)
    elif case == "tensor":
        factors = (0.5, 1.5)
    elif case == "tensor-rounding":
        factors = (1.3, 0.7)
    elif case == "zero-tensor":
        factors = (0.0, 1.5)
    elif case == "zero-block":
        sa[:, ::2] = 0
        sb[1::2, :] = 0
    elif case.startswith("group-"):
        group = int(case.removeprefix("group-"))
        if not 0 <= group < spec.K // 16:
            raise ValueError("scale group outside the matrix")
        active = np.arange(spec.K) // 16 == group
        a[:, ~active] = 0
        b[:, ~active] = 0
    elif case.startswith("codes-"):
        k = int(case.removeprefix("codes-"))
        if not 0 <= k < spec.K or min(spec.M, spec.N) < 16:
            raise ValueError("codes-N requires a valid K index and M/N >=16")
        a.fill(0)
        b.fill(0)
        a[:, k] = np.arange(spec.M) % 16
        b[:, k] = np.arange(spec.N) % 16
    elif case != "mixed":
        raise ValueError(f"unknown case {case!r}")
    return (pack_fp4_codes(a), pack_fp4_codes(b), sa, sb), factors


def run_cases(
    spec: NvFp4GemmSpec, cases: tuple[str, ...], *, compile_route="comgr"
) -> int:
    ok, reason = is_valid_spec(spec)
    if not ok:
        raise ValueError(reason)
    if not cases or compile_route not in ("comgr", "hip"):
        raise ValueError("provide cases and a comgr or hip compile route")
    arch = get_device_arch(0)
    if arch != "gfx1250":
        raise RuntimeError(f"visible HIP device 0 must be gfx1250, got {arch!r}")
    kernel = build_nvfp4_gemm(spec, arch=arch)
    compile_fn = (
        compile_kernel if compile_route == "comgr" else compile_kernel_via_hipcc
    )
    art = compile_fn(kernel, arch=arch)
    rt = Runtime()
    module = rt.load_module(art.hsaco)
    try:
        fn = module.get_function(art.kernel_name)
        for case in cases:
            inputs, factors = make_case_inputs(spec, case)
            expected = reference_result(*inputs, factors, dtype_c=spec.dtype_c)
            got = _launch(rt, fn, spec.block_spec(), inputs, tensor_scales=factors)
            check_result(got, expected, exact=True)
            print(
                f"PASS: nvfp4/{compile_route}/{spec.dtype_c}/{case} "
                f"{spec.M}x{spec.N}x{spec.K} bad=0",
                flush=True,
            )
    finally:
        module.unload()
    return len(cases)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m", type=int, default=32)
    p.add_argument("--n", type=int, default=48)
    p.add_argument("--k", type=int, default=256)
    p.add_argument("--dtype-c", choices=("bf16", "fp16"), default="bf16")
    p.add_argument("--compile-route", choices=("comgr", "hip"), default="comgr")
    p.add_argument("--case", default="all")
    args = p.parse_args(argv)
    spec = NvFp4GemmSpec("verify_nvfp4", args.m, args.n, args.k, args.dtype_c)
    cases = (
        (
            "neutral",
            "a-only",
            "b-only",
            "mixed",
            "tensor",
            "tensor-rounding",
            "zero-tensor",
            "zero-block",
        )
        + tuple(f"group-{g}" for g in range(spec.K // 16))
        + tuple(f"codes-{k}" for k in (0, 31, 32, 63, 64, 95, 96, 127))
        if args.case == "all"
        else (args.case,)
    )
    count = run_cases(spec, cases, compile_route=args.compile_route)
    print(f"PASS: verified {count} cases", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
