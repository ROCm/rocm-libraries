# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 scaled MFMA numerical verification with independent non-unity scales."""

from __future__ import annotations

import ctypes
import struct

import numpy as np

from ....helpers import compile_kernel
from ....helpers.compile import compile_kernel_via_hipcc
from ....instances.gfx950.block_scaled_gemm import (
    BlockScaledGemmSpec,
    block_scaled_gemm_grid,
    build_block_scaled_gemm,
    is_valid_spec,
)
from ....runtime.hip_module import Runtime, get_device_arch
from ...common.scaled_gemm_reference import (
    _output_dtype,
    check_result,
    make_case_inputs,
    pack_fp4_codes,
    pack_fp6_codes,
    reference_result,
)


def make_inputs(spec, case):
    """Use asymmetric K32 scales from 1/4, 1/2, 2, 4.

    Non-FP4 mantissas have magnitude <=1 and quantum >=1/16; FP4
    mantissas have magnitude <=6 and quantum 1/2. At K<=256, scaled
    product quantum counts are bounded by 9*2**20 for FP4 x FP4,
    3*2**22 for mixed FP4 pairs, and 2**24 otherwise. Every integer
    count through 2**24 is exactly representable in FP32, so each
    partial sum is exact before output rounding.
    """
    a, b, sa, sb = make_case_inputs(spec, "mixed" if case == "positive" else case)
    groups = spec.K // 32
    values = np.array([125, 126, 128, 129], dtype=np.uint8)
    g = np.arange(groups)[None, :]
    rows = np.arange(spec.M)[:, None]
    cols = np.arange(spec.N)[None, :]
    # Include tile index and step index without repeating the same K64 halves.
    sa[:] = values[(rows + rows // 32 + g + g // 2) % 4]
    sb[:] = values[(2 * cols + cols // 32 + 3 * g.T + g.T // 2 + 1) % 4]
    if case in ("neutral", "b-only") or case.startswith("codes-"):
        sa.fill(127)
    if case in ("neutral", "a-only") or case.startswith("codes-"):
        sb.fill(127)
    if case == "positive":
        # Positive dyadic mantissas expose lost scaling without cancellation.
        def positive(dtype, count):
            codes = (np.arange(count)[:, None] + np.arange(spec.K)[None, :]) % 3 + 1
            if dtype == "fp4e2m1":
                return pack_fp4_codes(codes.astype(np.uint8))
            if dtype in ("fp6e2m3", "fp6e3m2"):
                return pack_fp6_codes(codes.astype(np.uint8))
            import ml_dtypes

            return (codes * 0.25).astype(
                ml_dtypes.float8_e4m3fn if dtype == "fp8e4m3" else ml_dtypes.float8_e5m2
            )

        a = positive(spec.dtype_a, spec.M)
        b = positive(spec.dtype_b, spec.N)
    return a, b, sa, sb


def assert_scale_coverage(spec, case, sa, sb):
    if sa.shape != (spec.M, spec.K // 32) or sb.shape != (spec.K // 32, spec.N):
        raise AssertionError("incorrect scale fixture shape")
    if case == "neutral" or case.startswith("codes-"):
        return
    for active, scales in ((case != "b-only", sa), (case != "a-only", sb)):
        if active and not ({125, 126, 128, 129} <= set(np.unique(scales))):
            raise AssertionError(
                "non-trivial scale fixture must include scales below/above unity"
            )


def _u8_buffer(array: np.ndarray):
    array = np.ascontiguousarray(array)
    return (ctypes.c_uint8 * int(array.nbytes)).from_buffer_copy(array)


def _launch(rt, fn, spec, inputs):
    output_dtype = _output_dtype(spec.dtype_c)

    # A NaN sentinel makes missing output stores fail, including expected zeros.
    out = np.full((spec.M, spec.N), np.nan, dtype=output_dtype)
    allocations = []
    try:
        for array in (*inputs, out):
            ptr = rt.alloc(array.nbytes)
            allocations.append(ptr)
            rt.memcpy_h2d(ptr, _u8_buffer(array), array.nbytes)
        packed = struct.pack("<QQQQQiii", *allocations, spec.M, spec.N, spec.K)
        rt.launch(fn, block_scaled_gemm_grid(spec), (spec.block_size, 1, 1), packed)
        rt.sync()
        host_out = (ctypes.c_uint8 * int(out.nbytes))()
        rt.memcpy_d2h(host_out, allocations[-1], out.nbytes)
        return (
            np.frombuffer(bytes(host_out), dtype=output_dtype)
            .reshape(spec.M, spec.N)
            .astype(np.float32)
        )
    finally:
        for ptr in allocations:
            rt.free(ptr)


def run_cases(
    spec: BlockScaledGemmSpec,
    cases: tuple[str, ...],
    *,
    compile_route: str = "comgr",
    tol: float = 2e-2,
) -> int:
    """Compile once and launch each case; return the verified case count."""
    ok, reason = is_valid_spec(spec, arch="gfx950")
    if not ok:
        raise ValueError(reason)
    native = spec.resolved_matrix_path() != "wmma"
    if native and spec.K not in (64, 128, 256):
        raise ValueError("exact native fixtures require K=64, K=128 or K=256")
    if not cases:
        raise ValueError("at least one verification case is required")
    if compile_route not in ("comgr", "hip"):
        raise ValueError(f"unknown compile route {compile_route!r}")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")
    arch = get_device_arch(0)
    if arch != "gfx950":
        raise RuntimeError(f"visible HIP device 0 must be gfx950, got {arch!r}")
    kernel = build_block_scaled_gemm(spec, arch=arch)
    compile_fn = (
        compile_kernel if compile_route == "comgr" else compile_kernel_via_hipcc
    )
    art = compile_fn(kernel, arch=arch)
    print(
        f"[{arch}/{compile_route}] compiled {art.kernel_name} isa={art.isa}", flush=True
    )
    rt = Runtime()
    module = rt.load_module(art.hsaco)
    try:
        fn = module.get_function(art.kernel_name)
        for case in cases:
            inputs = make_inputs(spec, case)
            assert_scale_coverage(spec, case, inputs[2], inputs[3])
            expected = reference_result(
                *inputs,
                spec.block_k,
                native=native,
                dtype_a=spec.dtype_a,
                dtype_b=spec.dtype_b,
                dtype_c=spec.dtype_c,
            )
            label = (
                f"{spec.resolved_matrix_path()}/{spec.dtype_a}x{spec.dtype_b}/{compile_route}/{case} "
                f"{spec.M}x{spec.N}x{spec.K} bk{spec.block_k}"
            )
            got = _launch(rt, fn, spec, inputs)
            try:
                check_result(got, expected, exact=native, tol=tol)
            except AssertionError as exc:
                raise AssertionError(f"{label}: {exc}") from exc
            print(
                f"PASS: {label} bad=0 nonunity_a={np.count_nonzero(inputs[2] != 127)} nonunity_b={np.count_nonzero(inputs[3] != 127)}",
                flush=True,
            )
    finally:
        module.unload()
    return len(cases)
