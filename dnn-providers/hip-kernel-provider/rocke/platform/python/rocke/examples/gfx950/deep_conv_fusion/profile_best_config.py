# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Single blocking dispatch of the current best config for rocprofv3 capture.

Best config: pool_tile=4x4, tile_m=64, tile_k=32, tile_n=32, warp=2x1, mem.
Mirrors compare_pool_tile_configs.py compile/launch path but does exactly one
verified launch so counter passes see minimal noise.
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


def main():
    spec = make_deep_fused_conv_pool_spec(
        h=2160,
        w=3840,
        c=8,
        k0=32,
        k1=24,
        pool_tile_h=4,
        pool_tile_w=4,
        tile_k=32,
        tile_n=32,
        warp_m=2,
        warp_n=1,
        warp_tile_k=_ATOM_K,
    )
    ok, why = is_valid_spec(spec, arch=ARCH)
    if not ok:
        print(f"invalid spec: {why}", file=sys.stderr)
        return 2

    kernel = build_deep_fused_conv_pool(spec, arch=ARCH)
    artifact = compile_kernel(kernel, arch=ARCH)
    grid = deep_fused_conv_pool_grid(spec)
    block = (spec.block_size, 1, 1)
    print(
        f"pool_tile={spec.pool_tile_h}x{spec.pool_tile_w} tile_m={spec.tile_m} "
        f"tile_n={spec.tile_n} tile_k={spec.tile_k} warp={spec.warp_m}x{spec.warp_n} "
        f"grid={grid} block={block}"
    )

    conv = spec.problem.conv
    A, B0, W1, Y = _make_inputs(spec, seed=123)
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
        rt.launch_blocking(fn, grid, block, args)
        rt.memcpy_d2h(_as_u8_buffer(Y), Y_dev, Y.nbytes)
        ref = _reference_conv1x1_relu_pool(A, B0, W1, conv, spec.problem)
        diff = np.abs(Y.astype(np.float32) - ref.astype(np.float32))
        bad = int(np.count_nonzero(diff > 1e-2))
        print(f"verify: max_abs_diff={float(diff.max()):.6g} bad={bad}/{Y.size}")
    finally:
        rt.free(A_dev)
        rt.free(B_dev)
        rt.free(Y_dev)
        rt.free(W1_dev)
    return 0


if __name__ == "__main__":
    sys.exit(main())
