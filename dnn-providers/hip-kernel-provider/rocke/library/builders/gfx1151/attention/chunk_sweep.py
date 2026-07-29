# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Head-chunk x o_nt sweep for the large-Sq MALL-residency regime.

At L>=8K the per-head KV (``Sk * D * 2 tensors * 2 B``) is the reused working set we
want resident in the 32 MB MALL. Head-chunking bounds the concurrent KV by launching
only ``C`` heads at a time (serialized, single-stream, offset Q/K/V/O pointers,
``grid.Y = C``), trading launch serialization (``ceil(H/C)`` launches) for cache
residency.

The measured optimum was C=2 (16 MB) at L16K even though MALL is 32 MB -- because the
write-once O stream also allocates MALL lines and evicts KV. ``o_nt`` marks that store
non-temporal so it streams past MALL, which should free the headroom for C=4 and halve
the launch count (12 -> 6). This harness sweeps the (C, o_nt) grid to test that.

Usage:
    python3 chunk_sweep.py --seqlen 16384 --chunks 1,2,4,6 --ont 0,1
    python3 chunk_sweep.py --seqlen 2048 --chunks 2,4 --ont 0,1 --verify
"""

from __future__ import annotations

import argparse
import ctypes
import math
import struct
import sys

import numpy as np

from rocke.helpers import compile_kernel
from rocke.runtime.hip_module import Runtime
from rocke.runtime.launcher import time_launches

from kernels.gfx1151.wmma_fmha_swapqk import (
    SwapQKCfg,
    build_wmma_fmha_swapqk,
    swapqk_grid,
    swapqk_transpose_v,
)
from builders.gfx1151.attention.bench_v_staging import _ref_attention


def _u8(a):
    return (ctypes.c_uint8 * int(a.nbytes)).from_buffer(np.ascontiguousarray(a))


def run_chunked(
    cfg,
    rt,
    fn,
    bufs,
    *,
    seqlen,
    heads,
    head_size,
    chunk,
    batch,
    scale_log2,
    warmup,
    iters,
):
    """Time ceil(heads/chunk) serialized launches, each covering `chunk` heads.

    Head-chunking offsets the Q/K/V/O base pointers by ``h0 * head_size`` elements
    (layout [B, S, H, D] -> head stride is head_size) and sets ``grid.Y = chunk``, so
    only `chunk` heads' KV is live in MALL at a time. All launches go on the default
    stream so they stay serialized -- running them concurrently defeats the bound.

    Under ``cfg.v_transposed`` V is [B, H, D, S], so its head stride is
    ``head_size * seqlen`` rather than ``head_size``.
    """
    qd, kd, vd, od = bufs
    nqb = seqlen // cfg.q_rows_per_cta
    block = (cfg.block_size, 1, 1)
    stride_token = heads * head_size
    stride_head = head_size
    elem = 2  # f16
    v_head_elems = head_size * seqlen if cfg.v_transposed else head_size

    packs = []
    for h0 in range(0, heads, chunk):
        c = min(chunk, heads - h0)
        off = h0 * head_size * elem
        packs.append(
            (
                (nqb, c, batch),
                struct.pack(
                    "<QQQQfiiiiiiiiii",
                    qd + off,
                    kd + off,
                    vd + h0 * v_head_elems * elem,
                    od + off,
                    scale_log2,
                    seqlen,
                    seqlen,
                    stride_token,
                    stride_head,
                    stride_token,
                    stride_head,
                    stride_token,
                    stride_head,
                    stride_token,
                    stride_head,
                ),
            )
        )

    def launch_all():
        for grid, packed in packs:
            rt.launch(fn, grid, block, packed)

    launch_all()
    rt.sync()
    ms = time_launches(launch_all, warmup=warmup, iters=iters)
    return ms, len(packs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seqlen", type=int, default=16384)
    ap.add_argument("--head-size", type=int, default=128)
    ap.add_argument("--heads", type=int, default=24)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--chunks", default="1,2,4,6", help="comma-separated C values")
    ap.add_argument("--ont", default="0,1", help="comma-separated o_nt values (0/1)")
    ap.add_argument("--mq", type=int, default=2)
    ap.add_argument("--block-n", type=int, default=32)
    ap.add_argument("--ilp", type=int, default=2)
    ap.add_argument("--of16", type=int, default=1, help="f16 O-carry (0/1)")
    ap.add_argument(
        "--wpe", type=int, default=0, help="waves_per_eu (0=compiler default)"
    )
    ap.add_argument("--dual", type=int, default=1, help="dual-subtile V gather (0/1)")
    ap.add_argument("--waves", type=int, default=2, help="n_waves per CTA")
    ap.add_argument("--q-lds", type=int, default=0, help="stage Q in LDS (0/1)")
    ap.add_argument("--kv-lds", type=int, default=0, help="stage K/V in LDS (0/1)")
    ap.add_argument(
        "--vt", type=int, default=0, help="transposed-V layout [B,H,D,S] (0/1)"
    )
    ap.add_argument(
        "--vkb",
        type=int,
        default=0,
        help="key-blocked V [B,H,S/KB,D,KB]; 0=full transpose",
    )
    ap.add_argument(
        "--qkdo",
        type=int,
        default=0,
        help="QK loop d-outer/kv-inner (Q loaded once per d)",
    )
    ap.add_argument(
        "--vpf",
        type=int,
        default=0,
        help="V-gathers kept in flight across the PV steps",
    )
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument(
        "--verify",
        action="store_true",
        help="check vs numpy reference (only tractable at small L)",
    )
    args = ap.parse_args()

    L, D, H, B = args.seqlen, args.head_size, args.heads, args.batch
    chunks = [int(x) for x in args.chunks.split(",")]
    onts = [bool(int(x)) for x in args.ont.split(",")]

    kv_per_head_mb = L * D * 2 * 2 / 2**20
    print(f"shape: B={B} H={H} L={L} D={D}   per-head KV = {kv_per_head_mb:.1f} MB")
    print(
        f"{'C':>3} {'o_nt':>5} {'launches':>9} {'concKV':>8} "
        f"{'us':>10} {'TF':>7}  {'max_abs':>9}"
    )
    print("-" * 62)

    rng = np.random.default_rng(0xA11E)
    Q = (rng.standard_normal((B, L, H, D)) * 0.3).astype(np.float16)
    Kk = (rng.standard_normal((B, L, H, D)) * 0.3).astype(np.float16)
    Vv = (rng.standard_normal((B, L, H, D)) * 0.3).astype(np.float16)
    Out = np.zeros((B, L, H, D), dtype=np.float16)
    scale_log2 = float(1.0 / math.sqrt(D) * math.log2(math.e))

    ref = None
    if args.verify:
        ref = np.empty_like(Out)
        for bi in range(B):
            ref[bi] = _ref_attention(Q[bi], Kk[bi], Vv[bi], causal=False)

    rt = Runtime()
    # the reference above consumes V as [B,S,H,D]; only the device copy is relaid.
    Vdev = swapqk_transpose_v(Vv, args.vkb) if args.vt else Vv
    qd, kd, vd, od = (rt.alloc(x.nbytes) for x in (Q, Kk, Vdev, Out))
    rt.memcpy_h2d(qd, _u8(Q), Q.nbytes)
    rt.memcpy_h2d(kd, _u8(Kk), Kk.nbytes)
    rt.memcpy_h2d(vd, _u8(Vdev), Vdev.nbytes)

    flops = 4.0 * B * H * L * L * D
    best = None

    for ont in onts:
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
            o_nt=ont,
            q_lds=bool(args.q_lds),
            kv_lds=bool(args.kv_lds),
            v_transposed=bool(args.vt),
            v_kblock=args.vkb,
            v_prefetch=args.vpf,
            qk_douter=bool(args.qkdo),
            waves_per_eu=(args.wpe or None),
        )
        art = compile_kernel(
            build_wmma_fmha_swapqk(cfg, arch="gfx1151"), arch="gfx1151"
        )
        module = rt.load_module(art.hsaco)
        fn = module.get_function(art.kernel_name)

        for c in chunks:
            rt.memset(od, 0, Out.nbytes)
            try:
                ms, nlaunch = run_chunked(
                    cfg,
                    rt,
                    fn,
                    (qd, kd, vd, od),
                    seqlen=L,
                    heads=H,
                    head_size=D,
                    chunk=c,
                    batch=B,
                    scale_log2=scale_log2,
                    warmup=args.warmup,
                    iters=args.iters,
                )
            except Exception as e:  # noqa: BLE001
                print(
                    f"{c:>3} {int(ont):>5} {'-':>9} {'-':>8} "
                    f"FAIL {type(e).__name__}: {e}"
                )
                continue

            tf = flops / (ms * 1e-3) / 1e12
            mab = "-"
            if args.verify:
                rt.memcpy_d2h(_u8(Out), od, Out.nbytes)
                v = float(np.abs(Out.astype(np.float32) - ref.astype(np.float32)).max())
                mab = f"{v:.2e}"
            conc = c * kv_per_head_mb
            print(
                f"{c:>3} {int(ont):>5} {nlaunch:>9} {conc:>7.0f}M "
                f"{ms * 1e3:>10.1f} {tf:>7.2f}  {mab:>9}"
            )
            if best is None or tf > best[0]:
                best = (tf, c, ont)

        module.unload()

    for ptr in (qd, kd, vd, od):
        rt.free(ptr)

    if best:
        print(f"\nbest: {best[0]:.2f} TF at C={best[1]} o_nt={int(best[2])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
