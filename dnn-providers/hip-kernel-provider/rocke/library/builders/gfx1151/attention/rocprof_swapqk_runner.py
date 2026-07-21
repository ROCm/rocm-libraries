# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Fixed-shape launch runner for rocprofv3 probing of ``fmha_swapqk``.

Launches ONE swapqk dispatch ``--iters`` times on the null stream (no timing, no
verify) so ``rocprofv3 --pmc ... -- python3 -m ...rocprof_swapqk_runner`` attributes
counters cleanly to the single attention kernel. Defaults to the swept best config
(w2, block_n=32, ilp2, pingpong) at D128 L2048 H24 B1.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import struct


def main() -> int:
    import numpy as np

    from rocke.helpers import compile_kernel
    from rocke.runtime.hip_module import Runtime, get_device_arch
    from .fmha_swapqk import SwapQKCfg, build_wmma_fmha_swapqk, swapqk_grid

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--head-size", type=int, default=128)
    ap.add_argument("--seqlen-q", type=int, default=2048)
    ap.add_argument("--seqlen-k", type=int, default=2048)
    ap.add_argument("--heads", type=int, default=24)
    ap.add_argument("--kv-heads", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--waves", type=int, default=2)
    ap.add_argument("--ilp", type=int, default=2)
    ap.add_argument("--block-n", type=int, default=32)
    ap.add_argument("--sched", default="pingpong")
    ap.add_argument("--prefetch-v", type=int, default=0)
    ap.add_argument("--static", type=int, default=0)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--arch", default=None)
    args = ap.parse_args()

    arch = args.arch or get_device_arch() or "gfx1151"
    cfg = SwapQKCfg(
        head_size=args.head_size,
        num_query_heads=args.heads,
        num_kv_heads=args.kv_heads,
        mask_mode="causal" if args.causal else "none",
        n_waves=args.waves,
        sched_mode=args.sched,
        qk_ilp=args.ilp,
        block_n=args.block_n,
        prefetch_v=bool(args.prefetch_v),
        static_shape=bool(args.static),
    )
    art = compile_kernel(build_wmma_fmha_swapqk(cfg, arch=arch), arch=arch)
    print(f"[probe] built {art.kernel_name} arch={arch}", flush=True)

    B, Hq = args.batch, args.heads
    Hk = args.kv_heads or args.heads
    D, Sq, Sk = args.head_size, args.seqlen_q, args.seqlen_k
    rng = np.random.default_rng(0xA11E)
    Q = (rng.standard_normal((B, Sq, Hq, D)) * 0.3).astype(np.float16)
    K = (rng.standard_normal((B, Sk, Hk, D)) * 0.3).astype(np.float16)
    V = (rng.standard_normal((B, Sk, Hk, D)) * 0.3).astype(np.float16)
    Out = np.zeros((B, Sq, Hq, D), dtype=np.float16)
    scale_log2 = float(1.0 / math.sqrt(D) * math.log2(math.e))

    grid = swapqk_grid(cfg, seqlen_q=Sq, batch=B)
    block = (cfg.block_size, 1, 1)

    rt = Runtime()
    module = rt.load_module(art.hsaco)
    fn = module.get_function(art.kernel_name)

    def u8(a):
        return (ctypes.c_uint8 * int(a.nbytes)).from_buffer(np.ascontiguousarray(a))

    qd, kd, vd, od = (rt.alloc(x.nbytes) for x in (Q, K, V, Out))
    rt.memcpy_h2d(qd, u8(Q), Q.nbytes)
    rt.memcpy_h2d(kd, u8(K), K.nbytes)
    rt.memcpy_h2d(vd, u8(V), V.nbytes)
    rt.memset(od, 0, Out.nbytes)
    packed = struct.pack(
        "<QQQQfiiiiiiiiii",
        qd,
        kd,
        vd,
        od,
        scale_log2,
        Sq,
        Sk,
        Hq * D,
        D,
        Hk * D,
        D,
        Hk * D,
        D,
        Hq * D,
        D,
    )

    rt.launch(fn, grid, block, packed)
    rt.sync()
    print(f"[probe] launching {args.iters}x grid={grid} block={block}", flush=True)
    for _ in range(args.iters):
        rt.launch(fn, grid, block, packed)
    rt.sync()
    print("[probe] done", flush=True)
    for ptr in (qd, kd, vd, od):
        rt.free(ptr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
