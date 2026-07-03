# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Interleaved same-process A/B of the two gfx1151 WMMA-GEMM dispatch orders.

The question: does mapping ``block_id.x -> M-tile`` (grid_order "MN") run
faster or slower than ``block_id.x -> N-tile`` ("NM", the universal-GEMM
order) on gfx1151? RDNA dispatches workgroups in x-fastest order, so the
choice changes which tiles are co-resident and therefore L2 reuse of the
shared A/B rows.

This box auto-clocks +-25-30%, so only same-process interleaved ratios are
valid. This harness builds both kernels once, verifies each against the
numpy reference (C = A @ B.T), then benches them round-robin.

  PYTHONPATH=Python python3 -m rocke.examples.gfx1151.wmma_gemm_compare_orders \
      --m 4096 --n 4096 --k 4096 --rounds 5 --iters 100
"""

from __future__ import annotations

import argparse
import ctypes
import statistics
import struct

import numpy as np

from rocke.helpers import compile_kernel
from rocke.instances.gfx1151.wmma_gemm import WmmaGemmSpec, build_wmma_gemm
from rocke.runtime.hip_module import Runtime

ARCH = "gfx1151"


def _as_u8_buffer(a):
    return (ctypes.c_uint8 * int(a.nbytes)).from_buffer(a)


class Cfg:
    def __init__(self, name, block_x_is_m, M, N):
        self.name = name
        spec = WmmaGemmSpec(name="rocke_wmma_gemm", block_x_is_m=block_x_is_m)
        self.spec = spec
        self.kernel = build_wmma_gemm(spec, arch=ARCH)
        self.artifact = compile_kernel(self.kernel, arch=ARCH)
        # one wave per 16x16 tile; gx-fastest dispatch differs per order.
        mt = (M + 15) // 16
        nt = (N + 15) // 16
        self.grid = (mt, nt, 1) if block_x_is_m else (nt, mt, 1)
        self.block = (spec.block_size, 1, 1)
        self.samples = []


def _prep(rt, cfg, A, B, ref, M, N, K):
    mod = rt.load_module(cfg.artifact.hsaco)
    fn = mod.get_function(cfg.artifact.kernel_name)
    C = np.zeros((M, N), dtype=np.float16)
    A_dev = rt.alloc(A.nbytes)
    B_dev = rt.alloc(B.nbytes)
    C_dev = rt.alloc(C.nbytes)
    rt.memcpy_h2d(A_dev, _as_u8_buffer(A), A.nbytes)
    rt.memcpy_h2d(B_dev, _as_u8_buffer(B), B.nbytes)
    rt.memset(C_dev, 0, C.nbytes)
    args = struct.pack("<QQQiii", A_dev, B_dev, C_dev, M, N, K)
    rt.launch_blocking(fn, cfg.grid, cfg.block, args)
    rt.memcpy_d2h(_as_u8_buffer(C), C_dev, C.nbytes)
    bad = int(np.count_nonzero(np.abs(C.astype(np.float32) - ref) > 1e-2))
    return fn, args, (A_dev, B_dev, C_dev), bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=4096)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    M, N, K = args.m, args.n, args.k
    for d, nm in ((M, "M"), (N, "N"), (K, "K")):
        if d % 16:
            raise SystemExit(f"{nm}={d} must be a multiple of 16")

    rng = np.random.default_rng(args.seed)
    A = rng.integers(-5, 6, size=(M, K), dtype=np.int16).astype(np.float16)
    B = rng.integers(-5, 6, size=(N, K), dtype=np.int16).astype(np.float16)
    ref = (
        (A.astype(np.float32) @ B.astype(np.float32).T)
        .astype(np.float16)
        .astype(np.float32)
    )
    flop = 2.0 * M * N * K

    configs = [
        Cfg("xn (block.x->N, NM)", block_x_is_m=False, M=M, N=N),
        Cfg("xm (block.x->M, MN)", block_x_is_m=True, M=M, N=N),
    ]

    print(f"shape M={M} N={N} K={K}")
    rt = Runtime()
    live = []
    for cfg in configs:
        fn, cfgargs, devs, bad = _prep(rt, cfg, A, B, ref, M, N, K)
        status = "OK" if bad == 0 else f"BAD={bad}"
        print(f"built {cfg.name:24s} grid={cfg.grid} block={cfg.block} verify={status}")
        live.append((cfg, fn, cfgargs, devs))

    for cfg, fn, cfgargs, _ in live:
        for _ in range(args.warmup):
            rt.launch(fn, cfg.grid, cfg.block, cfgargs)
    rt.sync()

    for _ in range(args.rounds):
        for cfg, fn, cfgargs, _ in live:
            start = rt.event()
            end = rt.event()
            start.record()
            for _ in range(args.iters):
                rt.launch(fn, cfg.grid, cfg.block, cfgargs)
            end.record()
            end.synchronize()
            cfg.samples.append(start.elapsed_to(end) / args.iters)
            start.destroy()
            end.destroy()
        rt.sync()

    for cfg, _, _, devs in live:
        for d in devs:
            rt.free(d)

    print("\n=== Summary (median of rounds) ===")
    base = statistics.median(configs[0].samples)
    for cfg in configs:
        med = statistics.median(cfg.samples)
        spread = (max(cfg.samples) - min(cfg.samples)) / med * 100
        tflops = flop / 1e9 / med
        delta = (base / med - 1.0) * 100
        print(
            f"{cfg.name:24s} med={med:.5g} ms  spread={spread:4.1f}%  "
            f"{tflops:7.2f} TFLOP/s  ({delta:+.1f}% vs base)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
