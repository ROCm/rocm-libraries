# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Launch gfx1250 block-scaled GEMM and compare with an independent reference.

The default invocation keeps the K=64 FP8/BF8 WMMA + FP32-scale verifier.
Native ``--matrix-path wmma_scale`` / ``wmma_scale16`` use independent FP8/BF8, FP6/BF6, or FP4 operands and
encoded scales with K=32 / K=16 groups. Native fixtures cover K=128 or 256 and use
bounded dyadic values, permitting exact comparison after output-type rounding.
Numerical equality treats positive and negative zero as equal; output zero-sign
and signed-zero accumulation semantics are not verified.

Run on visible HIP device 0 (must be gfx1250), for example::

    python -m rocke.examples.gfx1250.gemm.block_scaled_gemm_verify \
        --matrix-path wmma_scale16 --case all
    python -m rocke.examples.gfx1250.gemm.block_scaled_gemm_verify \
        --matrix-path wmma_scale --m 32 --n 48 --k 256 --compile-route hip

Native COMGR lowering requires ROCKE_LLVM_FLAVOR=llvm23 and a matching
compiler. HIP verification requires a hipcc supporting the scaled builtins.
NumPy and ml_dtypes are required; torch is not used.
"""

from __future__ import annotations

import argparse
import ctypes
import struct

import numpy as np

from ....core.dtypes import normalize_dtype
from ....helpers import compile_kernel
from ....helpers.compile import compile_kernel_via_hipcc
from ....instances.gfx1250.block_scaled_gemm import (
    _LOWBIT_DTYPES,
    BlockScaledGemmSpec,
    block_scaled_gemm_grid,
    build_block_scaled_gemm,
    is_valid_spec,
)
from ....runtime.hip_module import Runtime, get_device_arch

# Keep the reference helpers importable from this established example module.
from ...common.scaled_gemm_reference import (
    _output_dtype,
    check_result,
    decode_e8m0,
    decode_fp4,
    decode_fp6,
    decode_scale,
    make_case_inputs,
    pack_fp4_codes,
    pack_fp6_codes,
    reference_result,
)

__all__ = [
    "check_result",
    "decode_e8m0",
    "decode_fp4",
    "decode_fp6",
    "decode_scale",
    "main",
    "make_case_inputs",
    "pack_fp4_codes",
    "pack_fp6_codes",
    "reference_result",
    "run_cases",
]


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
    ok, reason = is_valid_spec(spec, arch="gfx1250")
    if not ok:
        raise ValueError(reason)
    native = spec.resolved_matrix_path() != "wmma"
    if native and spec.K not in (128, 256):
        raise ValueError("exact native fixtures require K=128 or K=256")
    if not cases:
        raise ValueError("at least one verification case is required")
    if compile_route not in ("comgr", "hip"):
        raise ValueError(f"unknown compile route {compile_route!r}")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")
    arch = get_device_arch(0)
    if arch != "gfx1250":
        raise RuntimeError(f"visible HIP device 0 must be gfx1250, got {arch!r}")
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
            inputs = make_case_inputs(spec, case)
            da, db = spec.resolved_scale_dtypes()
            expected = reference_result(
                *inputs,
                spec.block_k,
                native=native,
                dtype_a=spec.dtype_a,
                dtype_b=spec.dtype_b,
                dtype_c=spec.dtype_c,
                scale_dtype_a=da,
                scale_dtype_b=db,
            )
            label = (
                f"{spec.resolved_matrix_path()}/{spec.dtype_a}x{spec.dtype_b}/{compile_route}/{case} "
                f"{spec.M}x{spec.N}x{spec.K} bk{spec.block_k} scales={da}/{db}"
            )
            got = _launch(rt, fn, spec, inputs)
            try:
                check_result(got, expected, exact=native, tol=tol)
            except AssertionError as exc:
                raise AssertionError(f"{label}: {exc}") from exc
            print(f"PASS: {label} bad=0", flush=True)
    finally:
        module.unload()
    return len(cases)


def main(argv: list[str] | None = None) -> int:
    """Run the legacy verifier or a bounded native SCALE/SCALE16 case set."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", default="gfx1250", choices=("gfx1250",))
    p.add_argument("--m", type=int, default=16)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--block-k", type=int, default=None)
    p.add_argument(
        "--dtype",
        default="fp8e4m3",
        type=normalize_dtype,
        choices=sorted(_LOWBIT_DTYPES),
    )
    p.add_argument("--tol", type=float, default=2e-2, help="legacy WMMA tolerance only")
    p.add_argument(
        "--matrix-path", default="wmma", choices=("wmma", "wmma_scale", "wmma_scale16")
    )
    p.add_argument("--compile-route", default="comgr", choices=("comgr", "hip"))
    p.add_argument(
        "--prefetch",
        action="store_true",
        help="add the data-prefetch hints (llvm23); results must not change",
    )
    p.add_argument(
        "--case",
        default="mixed",
        help="mixed, neutral, a-only, b-only, group-N, or all (including every K group)",
    )
    args = p.parse_args(argv)
    native = args.matrix_path != "wmma"
    block_k = args.block_k
    if block_k is None:
        block_k = {"wmma": 128, "wmma_scale": 32, "wmma_scale16": 16}[args.matrix_path]
    spec = BlockScaledGemmSpec(
        name="verify_gfx1250",
        M=args.m,
        N=args.n,
        K=args.k,
        dtype_a=args.dtype,
        dtype_b=args.dtype,
        dtype_c="bf16",
        scale_dtype="e8m0" if native else "fp32",
        block_k=block_k,
        matrix_path=args.matrix_path,
        prefetch=args.prefetch,
    )
    ok, reason = is_valid_spec(spec, arch=args.arch)
    if not ok:
        p.error(reason)
    cases = (
        ("neutral", "a-only", "b-only", "mixed")
        + tuple(f"group-{g}" for g in range(spec.K // block_k))
        if args.case == "all"
        else (args.case,)
    )
    count = run_cases(spec, cases, compile_route=args.compile_route, tol=args.tol)
    print(f"PASS: verified {count} cases", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
