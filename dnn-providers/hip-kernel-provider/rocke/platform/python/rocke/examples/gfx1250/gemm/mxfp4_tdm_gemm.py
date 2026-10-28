# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Verify the configurable double-buffered TDM/LDS MX FP4 GEMM on gfx1250."""

from __future__ import annotations

import argparse
import ctypes
import struct

import numpy as np

from ....examples.common.scaled_gemm_reference import decode_e8m0, pack_fp4_codes
from ....helpers.compile import compile_kernel
from ....instances.gfx1250.mxfp4_tdm_gemm import (
    MxFp4TdmGemmSpec,
    build_mxfp4_tdm_gemm,
    is_valid_spec,
    mxfp4_tdm_gemm_grid,
)
from ....instances.gfx1250.mxfp4_tdm_gemm_variants import (
    BASE_VARIANT,
    variant_names,
    variant_spec,
)
from ....runtime.hip_module import Runtime, get_device_arch


def make_inputs(spec: MxFp4TdmGemmSpec, seed: int = 0x32F4):
    """Create finite, exactly representable FP4/E8M0 verification inputs."""
    import ml_dtypes

    rng = np.random.default_rng(seed)
    a_codes = rng.integers(0, 2, (spec.M, spec.K), dtype=np.uint8) * 2
    b_codes = rng.integers(0, 2, (spec.N, spec.K), dtype=np.uint8) * 2
    a_scale = rng.integers(125, 129, (spec.M, spec.K // spec.block_k), dtype=np.uint8)
    b_scale = rng.integers(125, 129, (spec.K // spec.block_k, spec.N), dtype=np.uint8)
    c_values = rng.integers(0, 3, (spec.M, spec.N), dtype=np.uint8).astype(np.float32)
    c = c_values.astype(ml_dtypes.float8_e4m3fn)
    return (
        pack_fp4_codes(a_codes),
        pack_fp4_codes(b_codes),
        a_scale,
        b_scale,
        c,
        a_codes,
        b_codes,
    )


def reference(
    spec: MxFp4TdmGemmSpec,
    inputs,
    *,
    alpha: np.float32,
    beta: np.float32,
):
    """Independent grouped-scale FP32 reference rounded to FP8 E4M3."""
    import ml_dtypes

    _, _, a_scale, b_scale, c, a_codes, b_codes = inputs
    a = (a_codes == 2).astype(np.float32)
    b = (b_codes == 2).astype(np.float32)
    sa = decode_e8m0(a_scale).astype(np.float32)
    sb = decode_e8m0(b_scale).astype(np.float32)
    out = np.zeros((spec.M, spec.N), dtype=np.float32)
    for group in range(spec.K // spec.block_k):
        begin = group * spec.block_k
        end = begin + spec.block_k
        out += (a[:, begin:end] * sa[:, group, None]) @ (
            b[:, begin:end] * sb[group, :, None]
        ).T
    out = alpha * out + beta * c.astype(np.float32)
    return out.astype(ml_dtypes.float8_e4m3fn)


def _u8_buffer(array: np.ndarray):
    array = np.ascontiguousarray(array)
    return (ctypes.c_uint8 * int(array.nbytes)).from_buffer_copy(array)


def run_case(
    spec: MxFp4TdmGemmSpec,
    *,
    alpha: float | None = None,
    beta: float = 0.5,
) -> None:
    """Compile, launch, and require byte-exact FP8 output."""
    import ml_dtypes

    ok, reason = is_valid_spec(spec)
    if not ok:
        raise ValueError(reason)
    arch = get_device_arch(0)
    if arch != "gfx1250":
        raise RuntimeError(f"visible HIP device 0 must be gfx1250, got {arch!r}")

    alpha32 = np.float32(1.0 / max(spec.K, 256) if alpha is None else alpha)
    beta32 = np.float32(beta)
    inputs = make_inputs(spec)
    expected = reference(spec, inputs, alpha=alpha32, beta=beta32)
    out = np.zeros((spec.M, spec.N), dtype=ml_dtypes.float8_e4m3fn)

    artifact = compile_kernel(
        build_mxfp4_tdm_gemm(spec, arch=arch),
        arch=arch,
        backend="python",
    )
    runtime = Runtime()
    module = runtime.load_module(artifact.hsaco)
    allocations = []
    try:
        function = module.get_function(artifact.kernel_name)
        for array in (*inputs[:5], out):
            pointer = runtime.alloc(array.nbytes)
            allocations.append(pointer)
            runtime.memcpy_h2d(pointer, _u8_buffer(array), array.nbytes)
        arguments = struct.pack(
            "<QQQQQQiiiff",
            *allocations,
            spec.M,
            spec.N,
            spec.K,
            float(alpha32),
            float(beta32),
        )
        cluster = spec.cluster_dim
        runtime.launch(
            function,
            mxfp4_tdm_gemm_grid(spec),
            (spec.block_size, 1, 1),
            arguments,
            cluster=cluster,
        )
        runtime.sync()
        host = (ctypes.c_uint8 * int(out.nbytes))()
        runtime.memcpy_d2h(host, allocations[-1], out.nbytes)
        got = np.frombuffer(bytes(host), dtype=np.uint8).reshape(spec.M, spec.N)
    finally:
        module.unload()
        for pointer in allocations:
            runtime.free(pointer)

    expected_bits = expected.view(np.uint8)
    mismatch = np.argwhere(got != expected_bits)
    if mismatch.size:
        row, col = mismatch[0]
        raise AssertionError(
            f"{mismatch.shape[0]} FP8 mismatches; first at ({row}, {col}): "
            f"got=0x{got[row, col]:02x}, expected=0x{expected_bits[row, col]:02x}"
        )
    print(
        f"PASS: {artifact.kernel_name} {spec.M}x{spec.N}x{spec.K} bad=0",
        flush=True,
    )


def _geometry_kwargs(args: argparse.Namespace) -> dict:
    return {
        "tile_m": args.tile_m,
        "tile_n": args.tile_n,
        "tile_k": args.tile_k,
        "warp_m": args.warp_m,
        "warp_n": args.warp_n,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--tile-m", type=int, default=128)
    parser.add_argument("--tile-n", type=int, default=64)
    parser.add_argument("--tile-k", type=int, default=128)
    parser.add_argument("--warp-m", type=int, default=2)
    parser.add_argument("--warp-n", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument(
        "--variant",
        default=BASE_VARIANT,
        help=(
            f"named cumulative kernel variant ({', '.join(variant_names())}) "
            "or 'all'"
        ),
    )
    args = parser.parse_args(argv)
    geo = _geometry_kwargs(args)
    names = variant_names() if args.variant == "all" else [args.variant]
    for name in names:
        spec = variant_spec(name, args.m, args.n, args.k, **geo)
        run_case(spec, alpha=args.alpha, beta=args.beta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
