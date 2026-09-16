# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Launch gfx1250 block-scaled GEMM and compare with an independent reference.

The default invocation keeps the K=64 FP8/BF8 WMMA + FP32-scale verifier.
Native ``--matrix-path wmma_scale`` / ``wmma_scale16`` use FP8 and E8M0
scales with K=32 / K=16 groups. Native fixtures cover K=128 or 256 and use
bounded dyadic values, permitting exact comparison after BF16 rounding.

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

from ....helpers import compile_kernel
from ....helpers.compile import compile_kernel_via_hipcc
from ....instances.gfx1250.block_scaled_gemm import (
    BlockScaledGemmSpec,
    block_scaled_gemm_grid,
    build_block_scaled_gemm,
    is_valid_spec,
)
from ....runtime.hip_module import Runtime, get_device_arch


def decode_e8m0(encoded: np.ndarray) -> np.ndarray:
    """Decode finite unsigned E8M0 bytes without using kernel packing logic."""
    if encoded.dtype != np.uint8 or np.any(encoded == 0xFF):
        raise ValueError("expected finite uint8 E8M0 scales (0xff encodes NaN)")
    return np.ldexp(np.ones(encoded.shape), encoded.astype(np.int32) - 127)


def reference_result(
    a: np.ndarray,
    b: np.ndarray,
    a_scale: np.ndarray,
    b_scale: np.ndarray,
    block_k: int,
    *,
    native: bool,
) -> np.ndarray:
    """Expand scales onto logical A/B elements, multiply, then round to BF16.

    The native fixtures use quarter-integer inputs of magnitude <=1 and
    scales 2**[-2,3]. At K<=256, even the sum of absolute products fits in
    2**22 units of 2**-8, so every FP32 partial sum is exact. Float64 host
    arithmetic and a single BF16 rounding provide an independent oracle.
    """
    import ml_dtypes

    sa = decode_e8m0(a_scale) if native else a_scale.astype(np.float64)
    sb = decode_e8m0(b_scale) if native else b_scale.astype(np.float64)
    scaled_a = a.astype(np.float64) * np.repeat(sa, block_k, axis=1)
    scaled_b = b.astype(np.float64) * np.repeat(sb.T, block_k, axis=1)
    ref = scaled_a @ scaled_b.T
    if native:
        ref = ref.astype(ml_dtypes.bfloat16)
    return ref.astype(np.float32)


def make_case_inputs(
    spec: BlockScaledGemmSpec, case: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create asymmetric mantissas and scales, optionally isolating a K group."""
    import ml_dtypes

    native = spec.resolved_matrix_path() != "wmma"
    if native and spec.K not in (128, 256):
        raise ValueError("exact native fixtures require K=128 or K=256")
    groups = spec.K // spec.block_k
    lowbit_types = {
        "fp8": ml_dtypes.float8_e4m3fn,
        "fp8e4m3": ml_dtypes.float8_e4m3fn,
        "bf8": ml_dtypes.float8_e5m2,
        "bf8e5m2": ml_dtypes.float8_e5m2,
    }
    rng = np.random.default_rng(0xB10C)
    magnitude = 0.25 if native else 0.5
    a = (rng.integers(-4, 5, size=(spec.M, spec.K)) * magnitude).astype(
        lowbit_types[spec.dtype_a]
    )
    b = (rng.integers(-4, 5, size=(spec.N, spec.K)) * magnitude).astype(
        lowbit_types[spec.dtype_b]
    )
    if native:
        sa = rng.integers(125, 131, size=(spec.M, groups), dtype=np.uint8)
        sb = rng.integers(125, 131, size=(groups, spec.N), dtype=np.uint8)
        neutral = 127
    else:
        sa = rng.uniform(0.5, 1.5, size=(spec.M, groups)).astype(np.float32)
        sb = rng.uniform(0.5, 1.5, size=(groups, spec.N)).astype(np.float32)
        neutral = 1.0
    if case in ("neutral", "b-only"):
        sa.fill(neutral)
    if case in ("neutral", "a-only"):
        sb.fill(neutral)
    if case.startswith("group-"):
        group = int(case.removeprefix("group-"))
        if not 0 <= group < groups:
            raise ValueError(f"scale group {group} outside [0, {groups})")
        active = (np.arange(spec.K) // spec.block_k) == group
        a[:, ~active] = 0
        b[:, ~active] = 0
    elif case not in ("neutral", "a-only", "b-only", "mixed"):
        raise ValueError(f"unknown verification case {case!r}")
    return a, b, sa, sb


def check_result(
    got: np.ndarray, expected: np.ndarray, *, exact: bool, tol: float = 2e-2
) -> None:
    """Fail on incomplete/non-finite output or a numerical mismatch."""
    if got.shape != expected.shape:
        raise AssertionError(f"output shape {got.shape} != {expected.shape}")
    if not np.isfinite(expected).all() or not np.isfinite(got).all():
        raise AssertionError("unexpected non-finite output or reference")
    diff = np.abs(got.astype(np.float64) - expected.astype(np.float64))
    bad = got != expected if exact else diff > tol * np.maximum(np.abs(expected), 1)
    if np.any(bad):
        worst = np.unravel_index(int(np.argmax(diff)), diff.shape)
        raise AssertionError(
            f"bad={np.count_nonzero(bad)}/{diff.size}; worst={worst}: "
            f"got={got[worst]}, expected={expected[worst]}, diff={diff[worst]}"
        )


def _u8_buffer(array: np.ndarray):
    array = np.ascontiguousarray(array)
    return (ctypes.c_uint8 * int(array.nbytes)).from_buffer_copy(array)


def _launch(rt, fn, spec, inputs):
    import ml_dtypes

    # A NaN sentinel makes missing output stores fail, including expected zeros.
    out = np.full((spec.M, spec.N), np.nan, dtype=ml_dtypes.bfloat16)
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
            np.frombuffer(bytes(host_out), dtype=ml_dtypes.bfloat16)
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
    if spec.dtype_c != "bf16":
        raise ValueError("this verifier requires BF16 output")
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
            expected = reference_result(*inputs, spec.block_k, native=native)
            label = (
                f"{spec.resolved_matrix_path()}/{compile_route}/{case} "
                f"{spec.M}x{spec.N}x{spec.K} bk{spec.block_k}"
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
    p.add_argument("--dtype", default="fp8e4m3", choices=("fp8e4m3", "bf8e5m2"))
    p.add_argument("--tol", type=float, default=2e-2, help="legacy WMMA tolerance only")
    p.add_argument(
        "--matrix-path", default="wmma", choices=("wmma", "wmma_scale", "wmma_scale16")
    )
    p.add_argument("--compile-route", default="comgr", choices=("comgr", "hip"))
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
