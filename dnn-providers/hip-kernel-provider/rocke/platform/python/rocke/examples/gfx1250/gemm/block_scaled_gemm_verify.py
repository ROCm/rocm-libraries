# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Launch gfx1250 block-scaled GEMM and compare with an independent reference.

The default invocation keeps the K=64 FP8/BF8 WMMA + FP32-scale verifier.
Native ``--matrix-path wmma_scale`` / ``wmma_scale16`` use homogeneous FP8 or FP6 and
E8M0 scales with K=32 / K=16 groups. Native fixtures cover K=128 or 256 and use
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
from ....core.dtypes import normalize_dtype
from ....core.storage import BitPacking
from ....instances.gfx1250.block_scaled_gemm import (
    _LOWBIT_DTYPES,
    BlockScaledGemmSpec,
    _canon_lowbit,
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


def pack_fp6_codes(codes: np.ndarray) -> np.ndarray:
    """Pack four consecutive six-bit codes into three little-endian bytes."""
    if (
        codes.dtype != np.uint8
        or codes.ndim != 2
        or codes.shape[1] % 4
        or np.any(codes > 63)
    ):
        raise ValueError(
            "expected rank-2 uint8 FP6 codes in [0, 63] and K divisible by 4"
        )
    packed = b"".join(BitPacking(6).pack(row.tolist()) for row in codes)
    return np.frombuffer(packed, dtype=np.uint8).reshape(codes.shape[0], -1)


def decode_fp6(packed: np.ndarray, dtype: str = "fp6") -> np.ndarray:
    """Decode packed E2M3 or E3M2, including finite extrema and signed zero."""
    if packed.dtype != np.uint8 or packed.ndim != 2 or packed.shape[1] % 3:
        raise ValueError("expected rank-2 uint8 FP6 bytes with row size divisible by 3")
    if dtype not in ("fp6", "fp6e2m3", "bf6", "fp6e3m2"):
        raise ValueError("FP6 dtype must be fp6/E2M3 or bf6/E3M2")
    # Bit addressing is independent of the packer's four-code groups.
    bits = np.unpackbits(packed, axis=1, bitorder="little").reshape(
        packed.shape[0], -1, 6
    )
    codes = (bits * (1 << np.arange(6))).sum(axis=-1)
    mantissa_bits, bias = (3, 1) if _canon_lowbit(dtype) == "fp6" else (2, 3)
    fraction = (codes & ((1 << mantissa_bits) - 1)) / (1 << mantissa_bits)
    exponent = (codes & 31) >> mantissa_bits
    magnitude = np.where(
        exponent == 0,
        np.ldexp(fraction, 1 - bias),
        np.ldexp(1 + fraction, exponent - bias),
    )
    return np.copysign(magnitude, np.where(codes & 32, -1.0, 1.0))


def reference_result(
    a: np.ndarray,
    b: np.ndarray,
    a_scale: np.ndarray,
    b_scale: np.ndarray,
    block_k: int,
    *,
    native: bool,
    dtype_a: str | None = None,
    dtype_b: str | None = None,
) -> np.ndarray:
    """Expand scales onto logical A/B elements, multiply, then round to BF16.

    The native fixtures use quarter-integer inputs of magnitude <=1 and
    scales 2**[-2,3]. At K<=256, even the sum of absolute products fits in
    2**22 units of 2**-8, so every FP32 partial sum is exact. Float64 host
    arithmetic and a single BF16 rounding provide an independent oracle.
    FP6 all-code fixtures isolate one K element, avoiding accumulation error.
    """
    import ml_dtypes

    sa = decode_e8m0(a_scale) if native else a_scale.astype(np.float64)
    sb = decode_e8m0(b_scale) if native else b_scale.astype(np.float64)

    def matrix_values(data, dtype):
        if dtype in ("fp6", "fp6e2m3", "bf6", "fp6e3m2"):
            return decode_fp6(data, dtype)
        return data.astype(np.float64)

    a_values = matrix_values(a, dtype_a)
    b_values = matrix_values(b, dtype_b)
    scaled_a = a_values * np.repeat(sa, block_k, axis=1)
    scaled_b = b_values * np.repeat(sb.T, block_k, axis=1)
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

    def operand(dtype, rows):
        kind = _canon_lowbit(dtype)
        if kind in ("fp6", "bf6"):
            # Small dyadic values ensure exact FP32 partial sums in these tests.
            codes = rng.integers(
                0, 9 if kind == "fp6" else 13, size=(rows, spec.K), dtype=np.uint8
            )
            return codes | (rng.integers(0, 2, size=codes.shape, dtype=np.uint8) << 5)
        magnitude = 0.25 if native else 0.5
        return (rng.integers(-4, 5, size=(rows, spec.K)) * magnitude).astype(
            lowbit_types[dtype]
        )

    a, b = operand(spec.dtype_a, spec.M), operand(spec.dtype_b, spec.N)
    if native:
        small = any(
            _canon_lowbit(d) in ("fp6", "bf6") for d in (spec.dtype_a, spec.dtype_b)
        )

        sa = rng.integers(
            125, 129 if small else 131, size=(spec.M, groups), dtype=np.uint8
        )
        sb = rng.integers(
            125, 129 if small else 131, size=(groups, spec.N), dtype=np.uint8
        )
        neutral_a = neutral_b = 127
    else:
        sa = rng.uniform(0.5, 1.5, size=(spec.M, groups)).astype(np.float32)
        sb = rng.uniform(0.5, 1.5, size=(groups, spec.N)).astype(np.float32)
        neutral_a = neutral_b = 1.0
    if case in ("neutral", "b-only"):
        sa.fill(neutral_a)
    if case in ("neutral", "a-only"):
        sb.fill(neutral_b)
    if case.startswith("codes-"):
        if (
            _canon_lowbit(spec.dtype_a) not in ("fp6", "bf6")
            or _canon_lowbit(spec.dtype_b) not in ("fp6", "bf6")
            or min(spec.M, spec.N) < 64
        ):
            raise ValueError("codes-N requires FP6/BF6 operands and M/N >= 64")
        k = int(case.removeprefix("codes-"))
        if not 0 <= k < spec.K:
            raise ValueError("codes-N K index outside the matrix")
        a.fill(0)
        b.fill(0)
        a[:, k] = np.arange(spec.M, dtype=np.uint8) % 64
        b[:, k] = np.arange(spec.N, dtype=np.uint8) % 64
        sa.fill(neutral_a)
        sb.fill(neutral_b)
    elif case.startswith("group-"):
        group = int(case.removeprefix("group-"))
        if not 0 <= group < groups:
            raise ValueError(f"scale group {group} outside [0, {groups})")
        active = (np.arange(spec.K) // spec.block_k) == group
        a[:, ~active] = 0
        b[:, ~active] = 0
    elif case not in ("neutral", "a-only", "b-only", "mixed"):
        raise ValueError(f"unknown verification case {case!r}")

    def pack(data, dtype):
        kind = _canon_lowbit(dtype)
        return pack_fp6_codes(data) if kind in ("fp6", "bf6") else data

    a, b = pack(a, spec.dtype_a), pack(b, spec.dtype_b)
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
            expected = reference_result(
                *inputs,
                spec.block_k,
                native=native,
                dtype_a=spec.dtype_a,
                dtype_b=spec.dtype_b,
            )
            label = (
                f"{spec.resolved_matrix_path()}/{spec.dtype_a}/{compile_route}/{case} "
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
