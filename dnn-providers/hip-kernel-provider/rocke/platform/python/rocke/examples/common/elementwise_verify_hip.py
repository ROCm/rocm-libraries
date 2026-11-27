# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""HIP-path (hipcc) elementwise-add numeric verify -- the "simple op" check.

Builds a binary elementwise-add f16 kernel, compiles it through the HIP-C++ ->
hipcc backend (``compile_kernel_via_hipcc``), runs it on the GPU, and compares
to numpy. Confirms the non-MMA portion of the HIP lowering path (loads, vec
pack/store, f32 promote, store) is numerically correct on the target arch.

  PYTHONPATH=python python3 -m rocke.examples.common.elementwise_verify_hip --arch gfx942
"""

from __future__ import annotations

import argparse
import struct

import numpy as np

from rocke.helpers.compile import compile_kernel_via_hipcc
from rocke.instances.common.elementwise import (
    ElementwiseSpec,
    build_elementwise,
    elementwise_grid,
)
from rocke.numeric import storage
from rocke.runtime.hip_module import Runtime
from rocke.runtime.host_buffers import as_u8_buffer


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
    A_f16 = storage.encode(rng.integers(-8, 9, size=(args.n,), dtype=np.int16), "fp16")
    B_f16 = storage.encode(rng.integers(-8, 9, size=(args.n,), dtype=np.int16), "fp16")

    rt = Runtime()
    mod = rt.load_module(art.hsaco)
    fn = mod.get_function(art.kernel_name)
    A_dev = rt.alloc(A_f16.nbytes)
    B_dev = rt.alloc(B_f16.nbytes)
    C_dev = rt.alloc(args.n * 2)
    rt.memcpy_h2d(A_dev, as_u8_buffer(A_f16), A_f16.nbytes)
    rt.memcpy_h2d(B_dev, as_u8_buffer(B_f16), B_f16.nbytes)
    rt.memset(C_dev, 0, args.n * 2)

    grid = elementwise_grid(args.n, spec)
    block = (spec.block_size, 1, 1)
    pack = struct.pack("<QQQi", A_dev, B_dev, C_dev, args.n)
    rt.launch(fn, grid, block, pack, stream=0)
    rt.stream_sync(0)

    C_f16 = np.empty(args.n, dtype=np.float16)
    rt.memcpy_d2h(as_u8_buffer(C_f16), C_dev, C_f16.nbytes)
    C_out = storage.decode(C_f16, "fp16")
    ref = storage.round_to(
        storage.decode(A_f16, "fp16") + storage.decode(B_f16, "fp16"), "fp16"
    )

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
