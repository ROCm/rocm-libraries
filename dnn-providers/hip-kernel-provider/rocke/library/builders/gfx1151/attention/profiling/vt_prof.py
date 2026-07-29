#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Single-config swapqk launcher for rocprofv3, with the transposed-V toggle.

Issues exactly ``--iters`` launches of one head-chunk so counter attribution is
clean. Used to answer whether collapsing the V-gather (256 -> 32 vector-memory
instructions per K-loop iteration) actually relieves the memory unit.

Usage:
    rocprofv3 --pmc MemUnitBusy GPUBusy -d /tmp/vt -f csv -- \\
        python3 vt_prof.py --seqlen 8192 --vt 1 --iters 3
"""

from __future__ import annotations

import argparse
import ctypes
import math
import struct

import numpy as np

from rocke.helpers import compile_kernel
from rocke.runtime.hip_module import Runtime

from kernels.gfx1151.wmma_fmha_swapqk import (
    SwapQKCfg,
    build_wmma_fmha_swapqk,
    swapqk_transpose_v,
)


def _u8(a):
    return (ctypes.c_uint8 * int(a.nbytes)).from_buffer(np.ascontiguousarray(a))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seqlen", type=int, default=8192)
    ap.add_argument("--head-size", type=int, default=128)
    ap.add_argument("--heads", type=int, default=24)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=1)
    ap.add_argument("--mq", type=int, default=1)
    ap.add_argument("--of16", type=int, default=0)
    ap.add_argument("--block-n", type=int, default=64)
    ap.add_argument("--ilp", type=int, default=2)
    ap.add_argument("--waves", type=int, default=2)
    ap.add_argument("--vt", type=int, default=0)
    ap.add_argument("--vkb", type=int, default=0)
    ap.add_argument("--dual", type=int, default=1)
    ap.add_argument("--qkdo", type=int, default=0)
    ap.add_argument("--bg", type=int, default=0)
    ap.add_argument(
        "--vpf",
        type=int,
        default=0,
        help="V-gathers kept in flight across the PV steps",
    )
    ap.add_argument(
        "--wpe", type=int, default=0, help="waves_per_eu hint (0 = compiler default)"
    )
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    L, D, H, B = args.seqlen, args.head_size, args.heads, args.batch
    cfg = SwapQKCfg(
        head_size=D,
        num_query_heads=H,
        num_kv_heads=0,
        mask_mode="none",
        n_waves=args.waves,
        q_block=args.mq,
        o_f16=bool(args.of16),
        block_n=args.block_n,
        qk_ilp=args.ilp,
        sched_mode="pingpong",
        buffer_gather=True,
        dual_gather=bool(args.dual),
        fast_exp2=True,
        v_transposed=bool(args.vt),
        v_kblock=args.vkb,
        v_prefetch=args.vpf,
        qk_douter=bool(args.qkdo),
        bcast_group=args.bg,
        waves_per_eu=(args.wpe or None),
    )

    rng = np.random.default_rng(0xA11E)
    Q = (rng.standard_normal((B, L, H, D)) * 0.3).astype(np.float16)
    Kk = (rng.standard_normal((B, L, H, D)) * 0.3).astype(np.float16)
    Vv = (rng.standard_normal((B, L, H, D)) * 0.3).astype(np.float16)
    Out = np.zeros((B, L, H, D), dtype=np.float16)
    Vdev = swapqk_transpose_v(Vv, args.vkb) if args.vt else Vv
    scale_log2 = float(1.0 / math.sqrt(D) * math.log2(math.e))

    art = compile_kernel(build_wmma_fmha_swapqk(cfg, arch="gfx1151"), arch="gfx1151")
    rt = Runtime()
    module = rt.load_module(art.hsaco)
    fn = module.get_function(art.kernel_name)

    qd, kd, vd, od = (rt.alloc(x.nbytes) for x in (Q, Kk, Vdev, Out))
    rt.memcpy_h2d(qd, _u8(Q), Q.nbytes)
    rt.memcpy_h2d(kd, _u8(Kk), Kk.nbytes)
    rt.memcpy_h2d(vd, _u8(Vdev), Vdev.nbytes)

    stride_token, stride_head = H * D, D
    v_head_elems = D * L if args.vt else D
    nqb = L // cfg.q_rows_per_cta
    block = (cfg.block_size, 1, 1)

    # one chunk only: the profiled dispatch is a single representative head group
    h0 = 0
    packed = struct.pack(
        "<QQQQfiiiiiiiiii",
        qd + h0 * D * 2,
        kd + h0 * D * 2,
        vd + h0 * v_head_elems * 2,
        od + h0 * D * 2,
        scale_log2,
        L,
        L,
        stride_token,
        stride_head,
        stride_token,
        stride_head,
        stride_token,
        stride_head,
        stride_token,
        stride_head,
    )
    grid = (nqb, args.chunk, B)

    rt.launch(fn, grid, block, packed)
    rt.sync()
    for _ in range(args.iters):
        rt.launch(fn, grid, block, packed)
    rt.sync()

    print(
        f"vt={args.vt} vkb={args.vkb} L={L} grid={grid} block={block} kernel={art.kernel_name}"
    )
    for ptr in (qd, kd, vd, od):
        rt.free(ptr)
    module.unload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
