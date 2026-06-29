# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build + numeric-verify the gfx1250 WMMA single-launch fused-MoE mega-kernel.

Runs the ONE fused kernel (gate+up+silu -> LDS Hidden -> down -> weighted atomic
reduce) on a gfx1250 device and compares against a numpy bf16/f32 reference that
models the kernel's storage (bf16 operands + bf16 LDS Hidden, f32 accumulation).
A PASS confirms the whole fused path: the dual-B shared-A gate+up WMMA k-loop,
the SiLU->Hidden cshuffle stage (column-distributed WMMA accumulator scatter),
the LDS-resident-A down WMMA k-loop, and the token-masked weighted reduce.

Must run on a gfx1250 device:

  PYTHONPATH=Python python3 -m rocke.examples.gfx1250.fused_mega_moe.fused_mega_moe_wmma_verify \
      --tokens 16 --experts 1 --hidden 64 --inter 64 --hout 64
"""

from __future__ import annotations

import argparse
import ctypes
import struct


def main() -> int:
    from rocke.runtime.comgr import prefer_bundled_lib

    prefer_bundled_lib()  # pin newest comgr/LLVM flavor before lowering (gfx1250 needs ROCm>=7.2)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", default="gfx1250")
    p.add_argument("--tokens", type=int, default=16)
    p.add_argument("--experts", type=int, default=1)
    p.add_argument("--hidden", type=int, default=64)  # K (contraction of gate/up)
    p.add_argument("--inter", type=int, default=64)  # I / N (gate-up out, down K)
    p.add_argument("--hout", type=int, default=64)  # H_out (down output)
    p.add_argument("--tile-n-inter", type=int, default=64)
    p.add_argument("--tile-n-down", type=int, default=64)
    p.add_argument("--dtype", default="bf16", choices=("bf16", "fp16"))
    p.add_argument("--pipeline", default="mem")  # 'mem' | 'wmma_v1'
    p.add_argument("--double-buffer", action="store_true")
    p.add_argument("--waves-per-eu", type=int, default=None)
    p.add_argument("--tile-m", type=int, default=16)
    p.add_argument("--tol", type=float, default=3e-2)
    args = p.parse_args()

    import ml_dtypes
    import numpy as np

    from rocke.helpers import compile_kernel
    from rocke.instances.gfx1250.fused_moe_mega_wmma import (
        FusedMegaWmmaSpec,
        build_moe_fused_mega_wmma,
        moe_fused_mega_wmma_grid,
    )
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
        raise SystemExit(f"tokens={T} must be a multiple of {TILE_M}")
    if I % args.tile_n_inter:
        raise SystemExit("inter must be a multiple of tile_n_inter")
    if H_out % args.tile_n_down:
        raise SystemExit("hout must be a multiple of tile_n_down")

    st = ml_dtypes.bfloat16 if args.dtype == "bf16" else np.float16
    rng = np.random.default_rng(0xB0BA)

    # One token -> one expert (topk=1); experts assigned round-robin per m-block.
    num_m_blocks = T // TILE_M
    block_expert_ids = (np.arange(num_m_blocks) % E).astype(np.int32)
    # sorted token id per slot == slot row (identity gather); weight = 1.
    sorted_token_ids = np.arange(T, dtype=np.int32)
    sorted_weights = rng.uniform(0.4, 1.0, size=T).astype(np.float32)

    # Activations + per-expert weights. Small magnitudes -> low bf16 noise.
    X = (rng.standard_normal((T, K)) * 0.3).astype(np.float32)
    Wg = (rng.standard_normal((E, I, K)) * 0.1).astype(np.float32)
    Wu = (rng.standard_normal((E, I, K)) * 0.1).astype(np.float32)
    Wd = (rng.standard_normal((E, H_out, I)) * 0.1).astype(np.float32)

    # Cast to the kernel's storage dtype (bf16/f16) and back -> reference operands.
    def cast(a):
        return a.astype(st).astype(np.float32)

    Xs, Wgs, Wus, Wds = cast(X), cast(Wg), cast(Wu), cast(Wd)

    # ---- numpy reference modelling bf16 storage + bf16 LDS Hidden -------
    ref = np.zeros((T, H_out), np.float32)
    for slot in range(T):
        t = int(sorted_token_ids[slot])
        e = int(block_expert_ids[slot // TILE_M])
        w = float(sorted_weights[slot])
        gate = Xs[t] @ Wgs[e].T  # (I,)
        up = Xs[t] @ Wus[e].T
        hid = (gate / (1.0 + np.exp(-gate))) * up
        hidq = cast(hid)  # Hidden is stored bf16 in LDS
        ref[t] += w * (hidq @ Wds[e].T)

    # ---- build + launch the mega-kernel --------------------------------
    spec = FusedMegaWmmaSpec(
        name=f"verify_{args.arch}",
        dtype=args.dtype,
        tile_m=TILE_M,
        tile_n_inter=args.tile_n_inter,
        tile_n_down=args.tile_n_down,
        pipeline=args.pipeline,
        double_buffer=args.double_buffer,
        waves_per_eu=args.waves_per_eu,
    )
    art = compile_kernel(
        build_moe_fused_mega_wmma(spec, arch=args.arch), arch=args.arch
    )
    print(f"[{args.arch}] built {art.kernel_name} ({art.hsaco_bytes} B, isa={art.isa})")

    rt = Runtime()
    module = rt.load_module(art.hsaco)
    fn = module.get_function(art.kernel_name)

    def u8(a):
        a = np.ascontiguousarray(a)
        return (ctypes.c_uint8 * int(a.nbytes)).from_buffer_copy(a)

    A_st = X.astype(st)
    Wg_st = Wg.astype(st)
    Wu_st = Wu.astype(st)
    Wd_st = Wd.astype(st)
    Y = np.zeros((T, H_out), np.float32)

    ad = rt.alloc(A_st.nbytes)
    wgd = rt.alloc(Wg_st.nbytes)
    wud = rt.alloc(Wu_st.nbytes)
    wdd = rt.alloc(Wd_st.nbytes)
    stid = rt.alloc(sorted_token_ids.nbytes)
    swd = rt.alloc(sorted_weights.nbytes)
    beid = rt.alloc(block_expert_ids.nbytes)
    yd = rt.alloc(Y.nbytes)
    rt.memcpy_h2d(ad, u8(A_st.view(np.uint8)), A_st.nbytes)
    rt.memcpy_h2d(wgd, u8(Wg_st.view(np.uint8)), Wg_st.nbytes)
    rt.memcpy_h2d(wud, u8(Wu_st.view(np.uint8)), Wu_st.nbytes)
    rt.memcpy_h2d(wdd, u8(Wd_st.view(np.uint8)), Wd_st.nbytes)
    rt.memcpy_h2d(stid, u8(sorted_token_ids), sorted_token_ids.nbytes)
    rt.memcpy_h2d(swd, u8(sorted_weights), sorted_weights.nbytes)
    rt.memcpy_h2d(beid, u8(block_expert_ids), block_expert_ids.nbytes)
    rt.memset(yd, 0, Y.nbytes)

    M = num_m_blocks * TILE_M
    N = I  # inter dim (gate/up N == down K)
    stride_b_gate = I * K
    stride_b_up = I * K
    stride_b_down = H_out * I
    slot_size = 0
    tokens = T

    grid = moe_fused_mega_wmma_grid(num_m_blocks, I, spec)
    block = (spec.block_size, 1, 1)
    # 8 pointers + 10 i32 scalars (signature order).
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
        M,
        N,
        K,
        H_out,
        0,
        stride_b_gate,
        stride_b_up,
        stride_b_down,
        slot_size,
        tokens,
    )
    rt.launch(fn, grid, block, packed)
    rt.sync()
    out_buf = (ctypes.c_uint8 * int(Y.nbytes))()
    rt.memcpy_d2h(out_buf, yd, Y.nbytes)
    got = np.frombuffer(bytes(out_buf), dtype=np.float32).reshape(T, H_out)

    for ptr in (ad, wgd, wud, wdd, stid, swd, beid, yd):
        rt.free(ptr)
    module.unload()

    diff = np.abs(got - ref)
    denom = np.maximum(np.abs(ref), 1.0)
    max_rel = float((diff / denom).max())
    max_abs = float(diff.max())
    ok = max_rel <= args.tol
    print(
        f"[{args.arch}] fused-mega WMMA {args.dtype} T{T} E{E} K{K} I{I} Hout{H_out} "
        f"grid={grid}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
        f"tol={args.tol:.0e} -> {'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
