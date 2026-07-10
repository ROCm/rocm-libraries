# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Cross-arch TFLOPs benchmark for the single-launch fused-MoE mega-kernel.

Times the ONE fused kernel (gate+up+silu+down+reduce) end-to-end with HIP
events and reports achieved TFLOPs at a Qwen3-30B-A3B-style decode shape. The
same FLOP accounting (2 * M * (2*I*K + H_out*I), silu negligible) is used on
both targets so the gfx950 MFMA mega and the gfx1250 WMMA mega are directly
comparable:

  gfx1250 (remote):  ... -m rocke.examples.gfx1250.fused_mega_moe.fused_mega_moe_bench --arch gfx1250
  gfx950  (local):   ... -m rocke.examples.gfx1250.fused_mega_moe.fused_mega_moe_bench --arch gfx950

``M`` is the total sorted-token rows (num_m_blocks * 16). Each m-block is
assigned an expert (round-robin), every slot is a valid token so the atomic
reduce fires -- i.e. the timed kernel does the full per-tile work.
"""

from __future__ import annotations

import argparse
import ctypes
import struct


def _build(
    arch,
    dtype,
    tile_n_inter,
    tile_n_down,
    warp_n=4,
    pipeline="mem",
    double_buffer=False,
    waves_per_eu=None,
    tile_m=16,
):
    if arch == "gfx1250":
        from rocke.instances.gfx1250.fused_moe_mega_wmma import (
            FusedMegaWmmaSpec,
            build_moe_fused_mega_wmma,
            moe_fused_mega_wmma_grid,
        )

        spec = FusedMegaWmmaSpec(
            name=f"bench_{arch}",
            dtype=dtype,
            tile_m=tile_m,
            tile_n_inter=tile_n_inter,
            tile_n_down=tile_n_down,
            warp_n=warp_n,
            pipeline=pipeline,
            double_buffer=double_buffer,
            waves_per_eu=waves_per_eu,
        )
        return (
            spec,
            build_moe_fused_mega_wmma(spec, arch=arch),
            moe_fused_mega_wmma_grid,
        )
    from rocke.instances.common.moe_fused_mega import (
        FusedMegaKernelSpec,
        build_moe_fused_mega_gemm,
        moe_fused_mega_grid,
    )

    spec = FusedMegaKernelSpec(
        name=f"bench_{arch}",
        dtype=dtype,
        tile_m=tile_m,
        tile_n_inter=tile_n_inter,
        tile_n_down=tile_n_down,
    )
    return spec, build_moe_fused_mega_gemm(spec, arch=arch), moe_fused_mega_grid


def main() -> int:
    from rocke.runtime.comgr import prefer_bundled_lib

    prefer_bundled_lib()  # pin newest comgr/LLVM flavor before lowering (gfx1250 needs ROCm>=7.2)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", default="gfx1250", choices=("gfx1250", "gfx950"))
    p.add_argument("--tokens", type=int, default=512)  # sorted rows (mult of 16)
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--hidden", type=int, default=2048)  # K
    p.add_argument("--inter", type=int, default=768)  # I (mult of tile_n_inter)
    p.add_argument("--hout", type=int, default=2048)  # H_out (mult of tile_n_down)
    p.add_argument("--tile-m", type=int, default=16)  # rows/m-block (16,32,64)
    p.add_argument("--tile-n-inter", type=int, default=256)
    p.add_argument("--tile-n-down", type=int, default=256)
    p.add_argument("--dtype", default="bf16", choices=("bf16", "fp16"))
    p.add_argument("--warp-n", type=int, default=4)  # gfx1250 only (4->128t, 8->256t)
    p.add_argument("--pipeline", default="mem")  # gfx1250: 'mem' | 'wmma_v1'
    p.add_argument("--double-buffer", action="store_true")  # gfx1250 LDS ping-pong
    p.add_argument("--waves-per-eu", type=int, default=None)  # gfx1250 occupancy hint
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=30)
    args = p.parse_args()

    import numpy as np

    from rocke.helpers import compile_kernel
    from rocke.runtime.hip_module import Runtime

    T, E, K, I, H_out = (
        args.tokens,
        args.experts,
        args.hidden,
        args.inter,
        args.hout,
    )  # noqa: E741
    TILE_M = args.tile_m
    if T % TILE_M:
        raise SystemExit(f"tokens must be a multiple of tile_m={TILE_M}")
    if I % args.tile_n_inter or H_out % args.tile_n_down:
        raise SystemExit("inter % tile_n_inter and hout % tile_n_down must be 0")

    spec, kdef, grid_fn = _build(
        args.arch,
        args.dtype,
        args.tile_n_inter,
        args.tile_n_down,
        warp_n=args.warp_n,
        pipeline=args.pipeline,
        double_buffer=args.double_buffer,
        waves_per_eu=args.waves_per_eu,
        tile_m=args.tile_m,
    )
    art = compile_kernel(kdef, arch=args.arch)
    print(f"[{args.arch}] built {art.kernel_name} ({art.hsaco_bytes} B)")

    num_m_blocks = T // TILE_M
    st = (
        np.dtype(np.float16) if args.dtype == "fp16" else np.dtype(np.uint16)
    )  # noqa: F841
    elem_b = 2

    block_expert_ids = (np.arange(num_m_blocks) % E).astype(np.int32)
    sorted_token_ids = np.arange(T, dtype=np.int32)
    sorted_weights = np.ones(T, np.float32)

    rt = Runtime()
    module = rt.load_module(art.hsaco)
    fn = module.get_function(art.kernel_name)

    def u8(a):
        a = np.ascontiguousarray(a)
        return (ctypes.c_uint8 * int(a.nbytes)).from_buffer_copy(a)

    a_nb = T * K * elem_b
    wg_nb = E * I * K * elem_b
    wd_nb = E * H_out * I * elem_b
    y_nb = T * H_out * 4

    ad = rt.alloc(a_nb)
    wgd = rt.alloc(wg_nb)
    wud = rt.alloc(wg_nb)
    wdd = rt.alloc(wd_nb)
    stid = rt.alloc(sorted_token_ids.nbytes)
    swd = rt.alloc(sorted_weights.nbytes)
    beid = rt.alloc(block_expert_ids.nbytes)
    yd = rt.alloc(y_nb)
    for ptr, nb in ((ad, a_nb), (wgd, wg_nb), (wud, wg_nb), (wdd, wd_nb), (yd, y_nb)):
        rt.memset(ptr, 0, nb)
    rt.memcpy_h2d(stid, u8(sorted_token_ids), sorted_token_ids.nbytes)
    rt.memcpy_h2d(swd, u8(sorted_weights), sorted_weights.nbytes)
    rt.memcpy_h2d(beid, u8(block_expert_ids), block_expert_ids.nbytes)

    stride_b_gate = I * K
    stride_b_down = H_out * I
    packed = struct.pack(
        "<8Q10i",
        ad,
        wgd,
        wud,
        wdd,
        stid,
        swd,
        beid,
        yd,
        num_m_blocks * TILE_M,
        I,
        K,
        H_out,
        0,
        stride_b_gate,
        stride_b_gate,
        stride_b_down,
        0,
        T,
    )
    grid = grid_fn(num_m_blocks, I, spec)
    block = (spec.block_size, 1, 1)

    for _ in range(args.warmup):
        rt.launch(fn, grid, block, packed)
    rt.sync()

    start = rt.event()
    end = rt.event()
    start.record()
    for _ in range(args.iters):
        rt.launch(fn, grid, block, packed)
    end.record()
    end.synchronize()
    ms_total = start.elapsed_to(end)
    rt.sync()

    per_us = (ms_total / args.iters) * 1e3
    M = num_m_blocks * TILE_M
    flops = 2.0 * M * (2.0 * I * K + H_out * I)
    tflops = flops / (per_us * 1e-6) / 1e12

    for ptr in (ad, wgd, wud, wdd, stid, swd, beid, yd):
        rt.free(ptr)
    module.unload()

    print(
        f"[{args.arch}] fused-mega {args.dtype} pl={args.pipeline} M{M} K{K} I{I} "
        f"Hout{H_out} E{E} grid={grid} block={block[0]}: {per_us:.2f} us/iter  "
        f"{tflops:.2f} TFLOPs  ({flops / 1e9:.2f} GFLOP/iter)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
