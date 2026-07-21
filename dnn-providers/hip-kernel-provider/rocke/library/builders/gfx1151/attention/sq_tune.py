# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Driver for the transposed-QK WMMA FMHA kernel (``fmha_swapqk``)."""

from __future__ import annotations

import argparse
import ctypes
import math
import struct

from rocke.helpers import compile_kernel
from rocke.runtime.hip_module import Runtime
from rocke.runtime.launcher import time_launches

from .bench_v_staging import _find_objdump, _ref_attention
from .fmha_swapqk import SwapQKCfg, build_wmma_fmha_swapqk, swapqk_grid
from .tune import Shape, _mem_counts, _resource_counts


def verify_and_time(
    cfg: SwapQKCfg,
    shape: Shape,
    *,
    warmup=15,
    iters=100,
    tol=2e-2,
    objdump=None,
    arch="gfx1151",
    verify=True,
    emit_dir=None,
    prebuilt_dir=None,
):
    import numpy as np
    import os

    kname = cfg.kernel_name()
    if prebuilt_dir is not None:
        with open(os.path.join(prebuilt_dir, kname + ".hsaco"), "rb") as f:
            hsaco = f.read()

        class _Art:
            pass

        art = _Art()
        art.hsaco = hsaco
        art.kernel_name = kname
        isa = {}
    else:
        art = compile_kernel(build_wmma_fmha_swapqk(cfg, arch=arch), arch=arch)
        if emit_dir is not None:
            os.makedirs(emit_dir, exist_ok=True)
            path = os.path.join(emit_dir, art.kernel_name + ".hsaco")
            with open(path, "wb") as f:
                f.write(art.hsaco)
            return {
                "cfg": cfg,
                "ok": True,
                "max_abs": -1.0,
                "us": 0.0,
                "tflops": 0.0,
                "grid": None,
                "emit": path,
            }
        isa = _mem_counts(art.hsaco, art.kernel_name, objdump)
        isa.update(_resource_counts(art.hsaco))

    B, Hq, Hk, D = shape.batch, shape.heads, shape.kvh, shape.head_size
    Sq, Sk = shape.seqlen_q, shape.seqlen_k
    rng = np.random.default_rng(0xA11E)
    Q = (rng.standard_normal((B, Sq, Hq, D)) * 0.3).astype(np.float16)
    Kk = (rng.standard_normal((B, Sk, Hk, D)) * 0.3).astype(np.float16)
    Vv = (rng.standard_normal((B, Sk, Hk, D)) * 0.3).astype(np.float16)
    Out = np.zeros((B, Sq, Hq, D), dtype=np.float16)
    scale_log2 = float(1.0 / math.sqrt(D) * math.log2(math.e))

    grid = swapqk_grid(cfg, seqlen_q=Sq, batch=B)
    block = (cfg.block_size, 1, 1)

    rt = Runtime()
    module = rt.load_module(art.hsaco)
    fn = module.get_function(art.kernel_name)

    def u8(a):
        return (ctypes.c_uint8 * int(a.nbytes)).from_buffer(np.ascontiguousarray(a))

    qd, kd, vd, od = (rt.alloc(x.nbytes) for x in (Q, Kk, Vv, Out))
    rt.memcpy_h2d(qd, u8(Q), Q.nbytes)
    rt.memcpy_h2d(kd, u8(Kk), Kk.nbytes)
    rt.memcpy_h2d(vd, u8(Vv), Vv.nbytes)
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
    if verify:
        rt.memcpy_d2h(u8(Out), od, Out.nbytes)
        ref = np.empty_like(Out)
        for bi in range(B):
            if Hk != Hq:
                rep = Hq // Hk
                Kb = np.repeat(Kk[bi], rep, axis=1)
                Vb = np.repeat(Vv[bi], rep, axis=1)
            else:
                Kb, Vb = Kk[bi], Vv[bi]
            ref[bi] = _ref_attention(Q[bi], Kb, Vb, causal=shape.causal)
        max_abs = float(np.abs(Out.astype(np.float32) - ref.astype(np.float32)).max())
        ok = max_abs <= tol
    else:
        max_abs = -1.0
        ok = True

    ms = time_launches(
        lambda: rt.launch(fn, grid, block, packed), warmup=warmup, iters=iters
    )

    for ptr in (qd, kd, vd, od):
        rt.free(ptr)
    module.unload()

    flops = 4.0 * B * Hq * Sq * Sk * D
    if shape.causal:
        flops *= 0.5
    tflops = flops / (ms * 1e-3) / 1e12
    return {
        "cfg": cfg,
        "ok": ok,
        "max_abs": max_abs,
        "us": ms * 1e3,
        "tflops": tflops,
        "grid": grid,
        **isa,
    }


def _fmt(r):
    c = r["cfg"]
    vpe = c.waves_per_eu if c.waves_per_eu is not None else "def"
    return (
        f"w{c.n_waves} vpe={vpe:>3} {c.sched_mode:>8} ilp{c.qk_ilp} bn{c.block_n:>2} "
        f"pf{int(c.prefetch_v)} st{int(c.static_shape)} bg{int(c.buffer_gather)} | "
        f"{'Y' if r['ok'] else 'N'} {r['max_abs']:.2e} "
        f"{r['us']:8.1f}us {r['tflops']:7.2f} TF | "
        f"gld={r.get('gld', '-')} dsld={r.get('dsld', '-')} dsst={r.get('dsst', '-')} "
        f"wmma={r.get('wmma', '-')} instr={r.get('instr', '-')} "
        f"vgpr={r.get('vgpr', '-')} spill={r.get('vspill', '-')}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlen-q", type=int, default=512)
    ap.add_argument("--seqlen-k", type=int, default=512)
    ap.add_argument("--head-size", type=int, default=128)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv-heads", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--waves", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--wpe", type=int, nargs="+", default=[0])
    ap.add_argument(
        "--sched", nargs="+", default=["pingpong"], choices=["none", "pingpong"]
    )
    ap.add_argument("--ilp", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--block-n", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--pfv", type=int, nargs="+", default=[0], help="prefetch_v: 0/1")
    ap.add_argument(
        "--static", type=int, nargs="+", default=[0], help="static_shape: 0/1"
    )
    ap.add_argument(
        "--buf",
        type=int,
        nargs="+",
        default=[1],
        help="buffer_gather D16: 0/1 (only wins at w2/bn>=32)",
    )
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--arch", default="gfx1151")
    ap.add_argument("--emit", default=None)
    ap.add_argument("--prebuilt", default=None)
    args = ap.parse_args()

    shape = Shape(
        batch=args.batch,
        heads=args.heads,
        kv_heads=args.kv_heads,
        seqlen_q=args.seqlen_q,
        seqlen_k=args.seqlen_k,
        head_size=args.head_size,
        causal=args.causal,
    )
    objdump = _find_objdump()
    print(
        f"shape: B{shape.batch} Sq{shape.seqlen_q} Sk{shape.seqlen_k} D{shape.head_size} "
        f"Hq{shape.heads} Hk{shape.kvh} causal={shape.causal}"
    )
    best = None
    for w in args.waves:
        for wpe in args.wpe:
            for sched in args.sched:
                for il in args.ilp:
                    for bn in args.block_n:
                        for pf in args.pfv:
                            for st in args.static:
                                for bg in args.buf:
                                    cfg = SwapQKCfg(
                                        head_size=shape.head_size,
                                        num_query_heads=shape.heads,
                                        num_kv_heads=shape.kv_heads,
                                        mask_mode="causal" if shape.causal else "none",
                                        n_waves=w,
                                        waves_per_eu=(wpe or None),
                                        sched_mode=sched,
                                        qk_ilp=il,
                                        block_n=bn,
                                        prefetch_v=bool(pf),
                                        static_shape=bool(st),
                                        buffer_gather=bool(bg),
                                    )
                                    try:
                                        r = verify_and_time(
                                            cfg,
                                            shape,
                                            objdump=objdump,
                                            verify=not args.no_verify,
                                            arch=args.arch,
                                            emit_dir=args.emit,
                                            prebuilt_dir=args.prebuilt,
                                        )
                                    except Exception as e:  # noqa: BLE001
                                        print(
                                            f"w{w} wpe={wpe} {sched} ilp{il} bn{bn} pf{pf} bg{bg}: BUILD/RUN FAIL: {e}"
                                        )
                                        continue
                                    print(_fmt(r))
                                    if r["ok"] and (
                                        best is None or r["tflops"] > best["tflops"]
                                    ):
                                        best = r
    if best:
        print("\nBEST:", _fmt(best))


if __name__ == "__main__":
    raise SystemExit(main())
