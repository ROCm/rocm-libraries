# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Compare pool_tile=4x8,tk16 baseline vs pool_tile=4x4,tk32 variant.

Uses the same compile/launch path as deep_fused_conv_pool_verify.py:
build_deep_fused_conv_pool -> compile_kernel -> Runtime events.
Both configs are verified against the numpy reference before timing.
"""

import ctypes
import sys

import numpy as np

from rocke.core.arch import ArchTarget
from rocke.helpers import compile_kernel
from rocke.instances.gfx950.deep_fused_conv_pool import (
    build_deep_fused_conv_pool,
    deep_fused_conv_pool_grid,
    is_valid_spec,
    make_deep_fused_conv_pool_spec,
)
from rocke.examples.gfx950.deep_conv_fusion.deep_fused_conv_pool_verify import (
    _make_inputs,
    _reference_conv1x1_relu_pool,
    _pack_args,
    _useful_flops,
)
from rocke.runtime.hip_module import Runtime


ARCH = "gfx950"
_ATOM_K = (
    ArchTarget.from_gfx(ARCH)
    .mma.select_largest_k(a_dtype="fp16", b_dtype="fp16", c_dtype="fp32", m=32, n=32)
    .k
)


def _as_u8_buffer(array):
    return (ctypes.c_uint8 * int(array.nbytes)).from_buffer(array)


def make_spec(pool_tile_h, pool_tile_w, tile_k, tile_n, warp_m, warp_n, **kw):
    spec = make_deep_fused_conv_pool_spec(
        h=2160,
        w=3840,
        c=8,
        k0=32,
        k1=24,
        pool_tile_h=pool_tile_h,
        pool_tile_w=pool_tile_w,
        tile_k=tile_k,
        tile_n=tile_n,
        warp_m=warp_m,
        warp_n=warp_n,
        warp_tile_k=_ATOM_K,
        **kw,
    )
    ok, why = is_valid_spec(spec, arch=ARCH)
    if not ok:
        raise ValueError(
            f"invalid spec ({pool_tile_h}x{pool_tile_w}, tk{tile_k}): {why}"
        )
    return spec


def run_config(name, spec, *, warmup=100, iters=200, seed=123, tol=1e-2):
    print(f"\n=== {name} ===")
    print(
        f"  pool_tile={spec.pool_tile_h}x{spec.pool_tile_w} "
        f"tile_n={spec.tile_n} tile_k={spec.tile_k} "
        f"warp={spec.warp_m}x{spec.warp_n}"
    )
    kernel = build_deep_fused_conv_pool(spec, arch=ARCH)
    artifact = compile_kernel(kernel, arch=ARCH)
    grid = deep_fused_conv_pool_grid(spec)
    block = (spec.block_size, 1, 1)
    print(f"  grid={grid} block={block}")

    conv = spec.problem.conv
    A, B0, W1, Y = _make_inputs(spec, seed=seed)

    rt = Runtime()
    mod = rt.load_module(artifact.hsaco)
    fn = mod.get_function(artifact.kernel_name)
    A_dev = rt.alloc(A.nbytes)
    B_dev = rt.alloc(B0.nbytes)
    Y_dev = rt.alloc(Y.nbytes)
    W1_dev = rt.alloc(W1.nbytes)
    try:
        rt.memcpy_h2d(A_dev, _as_u8_buffer(A), A.nbytes)
        rt.memcpy_h2d(B_dev, _as_u8_buffer(B0), B0.nbytes)
        rt.memcpy_h2d(W1_dev, _as_u8_buffer(W1), W1.nbytes)
        rt.memset(Y_dev, 0, Y.nbytes)
        args = _pack_args(A_dev, B_dev, Y_dev, W1_dev, A, B0, Y, W1)

        # verify
        rt.launch_blocking(fn, grid, block, args)
        rt.memcpy_d2h(_as_u8_buffer(Y), Y_dev, Y.nbytes)
        ref = _reference_conv1x1_relu_pool(A, B0, W1, conv, spec.problem)
        diff = np.abs(Y.astype(np.float32) - ref.astype(np.float32))
        bad = int(np.count_nonzero(diff > tol))
        max_diff = float(diff.max()) if diff.size else 0.0
        print(f"  verify: max_abs_diff={max_diff:.6g} bad={bad}/{Y.size}")
        if bad:
            print("  !! verification FAILED", file=sys.stderr)

        # bench
        for _ in range(warmup):
            rt.launch(fn, grid, block, args)
        rt.sync()
        start = rt.event()
        end = rt.event()
        start.record()
        for _ in range(iters):
            rt.launch(fn, grid, block, args)
        end.record()
        end.synchronize()
        ms = start.elapsed_to(end) / iters
        start.destroy()
        end.destroy()
        rt.sync()
    finally:
        rt.free(A_dev)
        rt.free(B_dev)
        rt.free(Y_dev)
        rt.free(W1_dev)

    tflops = _useful_flops(spec) / 1e9 / ms
    print(f"  mean_ms={ms:.6g} useful_TFLOPS={tflops:.3f}")
    return ms, tflops


if __name__ == "__main__":
    print("Full target shape: H=2160 W=3840 C=8 K0=32 K1=24")

    configs = [
        ("4x4 tk32 mem (baseline)", make_spec(4, 4, 32, 32, 2, 1)),
        ("4x4 tk32 unroll_k", make_spec(4, 4, 32, 32, 2, 1, unroll_k=True)),
    ]
    results = [(name, *run_config(name, spec)) for name, spec in configs]

    print("\n=== Summary ===")
    base_ms, base_tf = results[0][1], results[0][2]
    for name, ms, tf in results:
        delta = (tf / base_tf - 1.0) * 100
        print(f"{name:28s}: {ms:.6g} ms  {tf:6.1f} TFLOP/s  ({delta:+.1f}%)")
