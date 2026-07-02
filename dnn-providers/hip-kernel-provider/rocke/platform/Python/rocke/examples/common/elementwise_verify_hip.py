# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""HIP-path (hipcc) elementwise-add numeric verify -- the "simple op" check.

Builds a binary elementwise-add f16 kernel, compiles it through the HIP-C++ ->
hipcc backend (``compile_kernel_via_hipcc``), runs it on the GPU, and compares
to numpy. Confirms the non-MMA portion of the HIP lowering path (loads, vec
pack/store, f32 promote, store) is numerically correct on the target arch.

  PYTHONPATH=Python python3 -m rocke.examples.common.elementwise_verify_hip --arch gfx942
"""

from __future__ import annotations

import argparse
import ctypes
import struct

import numpy as np

from rocke.helpers.compile import compile_kernel_via_hipcc
from rocke.instances.common.elementwise import (
    ElementwiseSpec,
    build_elementwise,
    elementwise_grid,
)
from rocke.runtime.hip_module import Runtime


def _as_u8(a):
    return (ctypes.c_uint8 * int(a.nbytes)).from_buffer(a)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", default="gfx942")
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--vec", type=int, default=8)
    p.add_argument("--tol", type=float, default=0.0)
    args = p.parse_args()

    spec = ElementwiseSpec(
        op="add",
        dtype="f16",
        block_size=args.block_size,
        vec=args.vec,
        name="ew_hipverify",
    )
    art = compile_kernel_via_hipcc(build_elementwise(spec), arch=args.arch)
    print(
        f"[{args.arch}] HIP-path built {art.kernel_name} ({art.hsaco_bytes} B) "
        f"hipcc={art.timings.get('hipcc', 0):.0f}ms"
    )

    rng = np.random.default_rng(0x1234)
    import torch

    A_f32 = rng.integers(-8, 9, size=(args.n,), dtype=np.int16).astype(np.float32)
    B_f32 = rng.integers(-8, 9, size=(args.n,), dtype=np.int16).astype(np.float32)
    A_u16 = (
        torch.from_numpy(A_f32)
        .to(torch.float16)
        .view(torch.int16)
        .numpy()
        .view(np.uint16)
    )
    B_u16 = (
        torch.from_numpy(B_f32)
        .to(torch.float16)
        .view(torch.int16)
        .numpy()
        .view(np.uint16)
    )

    rt = Runtime()
    mod = rt.load_module(art.hsaco)
    fn = mod.get_function(art.kernel_name)
    A_dev = rt.alloc(A_u16.nbytes)
    B_dev = rt.alloc(B_u16.nbytes)
    C_dev = rt.alloc(args.n * 2)
    rt.memcpy_h2d(A_dev, _as_u8(A_u16), A_u16.nbytes)
    rt.memcpy_h2d(B_dev, _as_u8(B_u16), B_u16.nbytes)
    rt.memset(C_dev, 0, args.n * 2)

    grid = elementwise_grid(args.n, spec)
    block = (spec.block_size, 1, 1)
    pack = struct.pack("<QQQi", A_dev, B_dev, C_dev, args.n)
    rt.launch(fn, grid, block, pack, stream=0)
    rt.stream_sync(0)

    out_buf = (ctypes.c_uint8 * (args.n * 2))()
    rt.memcpy_d2h(out_buf, C_dev, args.n * 2)
    C_out = (
        torch.from_numpy(np.frombuffer(bytes(out_buf), dtype=np.int16).copy())
        .view(torch.float16)
        .to(torch.float32)
        .numpy()
    )
    A_h = (
        torch.from_numpy(A_u16.view(np.int16))
        .view(torch.float16)
        .to(torch.float32)
        .numpy()
    )
    B_h = (
        torch.from_numpy(B_u16.view(np.int16))
        .view(torch.float16)
        .to(torch.float32)
        .numpy()
    )
    ref = torch.from_numpy(A_h + B_h).to(torch.float16).to(torch.float32).numpy()

    rt.free(A_dev)
    rt.free(B_dev)
    rt.free(C_dev)

    diff = np.abs(C_out - ref)
    max_abs = float(diff.max())
    bad = int((diff > args.tol).sum())
    ok = max_abs <= args.tol
    print(
        f"[{args.arch}] HIP-path elementwise-add N={args.n}: "
        f"max_abs_diff={max_abs:.3e} bad={bad}/{args.n} tol={args.tol:.0e} "
        f"-> {'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
