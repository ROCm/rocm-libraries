# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build, numeric-verify and time the gfx1151 transposed-QK WMMA FMHA (swapqk).

The production counterpart to ``wmma_fmha_fwd_verify``: builds the kernel from
:class:`SwapQKCfg` (whose defaults are the winning knobs), launches it via the
HIP runtime, compares against the same dense-attention numpy reference the rest
of this package uses, and reports achieved throughput.

    PYTHONPATH=python python3 -m builders.gfx1151.attention.wmma_fmha_swapqk_verify \
        --seqlen-q 2048 --seqlen-k 2048 --head-size 128 --heads 24 --batch 1

The reference materializes a full Sq x Sk score matrix, so it is only tractable
to about L=4096; use ``--no-verify`` for longer sequences (the kernel logic is
seqlen-agnostic and is covered by the small-L verifies).

Must run on a gfx1151 device. The kernel COMPILES on any host (comgr targets
gfx1151 regardless of the build GPU) but must EXECUTE on the board, so
``--emit DIR`` writes the hsaco without a GPU and ``--prebuilt DIR`` loads it
back on the board.

Accumulation order differs from the fp32 reference, so parity is judged within a
tolerance (default ``2e-2``), not bit-for-bit.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import struct

from rocke.helpers import compile_kernel
from rocke.runtime.hip_module import Runtime
from rocke.runtime.launcher import time_launches

from kernels.gfx1151.wmma_fmha_swapqk import (
    SwapQKCfg,
    build_wmma_fmha_swapqk,
    is_valid_spec,
    swapqk_grid,
    swapqk_transpose_v,
)


def _ref_attention(Q, K, V, *, causal: bool):
    """Dense attention reference, Q/K/V shape ``(seqlen, heads, head_size)``.

    Mirrors ``parity_extended_kernels._ref_attention`` (fp32 math, fp16 out).
    """
    import numpy as np

    d = Q.shape[-1]
    scores = np.einsum("ihd,jhd->ihj", Q.astype(np.float32), K.astype(np.float32))
    scores /= math.sqrt(d)
    if causal:
        q_pos = np.arange(Q.shape[0])[:, None, None]
        k_pos = np.arange(K.shape[0])[None, None, :]
        scores = np.where(k_pos <= q_pos, scores, -1e30)
    scores -= scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    out = np.einsum("ihj,jhd->ihd", probs, V.astype(np.float32))
    return out.astype(np.float16)


def _build_paged_v(np, V, block_size: int, *, shuffle: bool = True):
    """Relay V [B,Sk,Hk,D] into a fake vLLM-shaped paged cache + block table.

    Mirrors what ``reshape_and_cache`` already produces at KV-cache-write time:
    ``[num_blocks, Hk, D, block_size]``, i.e. token FASTEST -- which is exactly
    the order the PV A-fragment wants, and the reason this kernel can drop the
    caller's per-layer transpose entirely.

    Two deliberate hostilities, because the failure mode is a kernel that
    ignores the block table and still returns plausible numbers:

    * pages are SHUFFLED and over-allocated, so physical order != logical order
      and a kernel that treats the cache as contiguous reads a different
      request's keys rather than its own;
    * unused pages (and the padding of used ones) are POISONED with 7777.0
      instead of zero, so an off-by-one page lands on a value that blows past
      any tolerance instead of quietly attenuating the output.
    """
    B, Sk, Hk, D = V.shape
    if Sk % block_size:
        raise SystemExit(
            f"seqlen_k={Sk} must be a multiple of kv_block_size={block_size}"
        )
    per_req = Sk // block_size
    needed = B * per_req
    # Slack pages exist only to be poison: they prove nothing is read past the
    # table, and they push the used ids off the identity mapping.
    num_blocks = needed + max(4, needed // 2)

    rng = np.random.default_rng(0x9A9ED)
    ids = rng.permutation(num_blocks)[:needed] if shuffle else np.arange(needed)
    block_table = ids.astype(np.int32).reshape(B, per_req)

    cache = np.full((num_blocks, Hk, D, block_size), 7777.0, dtype=np.float16)
    # [B,Sk,Hk,D] -> [B,per_req,Hk,D,BS]; the last axis is the in-page token.
    src = V.reshape(B, per_req, block_size, Hk, D).transpose(0, 1, 3, 4, 2)
    cache[block_table] = src
    return np.ascontiguousarray(cache), np.ascontiguousarray(block_table)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", default="gfx1151")
    p.add_argument("--seqlen-q", type=int, default=2048)
    p.add_argument("--seqlen-k", type=int, default=2048)
    p.add_argument("--head-size", type=int, default=128)
    p.add_argument("--heads", type=int, default=24)
    p.add_argument("--kv-heads", type=int, default=0, help="0 -> MHA (== heads)")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--causal", action="store_true")
    # Production tunables (see SwapQKCfg for the measured defaults).
    p.add_argument("--n-waves", type=int, default=2)
    p.add_argument("--block-n", type=int, default=64)
    p.add_argument("--qk-ilp", type=int, default=2)
    p.add_argument(
        "--row-major-v",
        action="store_true",
        help="build against V as [B,S,H,D] instead of the transposed default",
    )
    p.add_argument(
        "--o-nt",
        action="store_true",
        help="stream the O store past the MALL (large-L lever, +3-14%%)",
    )
    p.add_argument(
        "--qk-douter",
        type=int,
        default=0,
        choices=(0, 1),
        help="d-outer QK loop; measured a loss at both corners, default off",
    )
    p.add_argument("--sched-mode", default="pingpong", choices=("pingpong", "none"))
    p.add_argument("--lazy-rescale", type=int, default=1, choices=(0, 1))
    p.add_argument(
        "--gqa-fuse",
        type=int,
        default=1,
        help="query heads sharing a kv head fused into one CTA (wave-per-head)",
    )
    p.add_argument(
        "--k-lds",
        action="store_true",
        help="stage the K tile in shared LDS (block_n=64 only; V stays on the gather)",
    )
    p.add_argument(
        "--v-paged",
        action="store_true",
        help="read V from a fake paged cache [num_blocks,Hk,D,BS] via a block table",
    )
    p.add_argument(
        "--kv-block-size",
        type=int,
        default=16,
        help="tokens per physical page under --v-paged (multiple of 16)",
    )
    p.add_argument(
        "--shuffle-blocks",
        type=int,
        default=1,
        choices=(0, 1),
        help="scatter the pages non-contiguously (default on: a kernel that "
        "ignores the block table still PASSES against an identity table)",
    )
    p.add_argument("--tol", type=float, default=2e-2)
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--emit", default=None, metavar="DIR", help="write hsaco, no run")
    p.add_argument("--prebuilt", default=None, metavar="DIR", help="load hsaco + run")
    args = p.parse_args()

    import numpy as np

    kvh = args.kv_heads or args.heads
    cfg = SwapQKCfg(
        head_size=args.head_size,
        num_query_heads=args.heads,
        num_kv_heads=args.kv_heads,
        mask_mode="causal" if args.causal else "none",
        n_waves=args.n_waves,
        block_n=args.block_n,
        qk_ilp=args.qk_ilp,
        v_transposed=not args.row_major_v,
        o_nt=args.o_nt,
        qk_douter=bool(args.qk_douter),
        sched_mode=args.sched_mode,
        lazy_rescale=bool(args.lazy_rescale),
        gqa_fuse=args.gqa_fuse,
        k_lds=args.k_lds,
        v_paged=args.v_paged,
        kv_block_size=args.kv_block_size if args.v_paged else 0,
    )
    ok, why = is_valid_spec(cfg, args.arch)
    if not ok:
        raise SystemExit(f"invalid config: {why}")

    B, Hq, Hk, D = args.batch, args.heads, kvh, args.head_size
    Sq, Sk = args.seqlen_q, args.seqlen_k
    if Sk % cfg.block_n:
        raise SystemExit(f"seqlen_k={Sk} must be a multiple of block_n={cfg.block_n}")

    if args.prebuilt is not None:
        name = cfg.kernel_name()
        with open(os.path.join(args.prebuilt, name + ".hsaco"), "rb") as f:
            hsaco = f.read()
        kernel_name = name
        print(f"[{args.arch}] loaded prebuilt {name} ({len(hsaco)} B)")
    else:
        art = compile_kernel(
            build_wmma_fmha_swapqk(cfg, arch=args.arch), arch=args.arch
        )
        hsaco, kernel_name = art.hsaco, art.kernel_name
        print(
            f"[{args.arch}] built {kernel_name} ({art.hsaco_bytes} B, isa={art.isa}) "
            f"total={art.timings.get('total', 0):.1f}ms"
        )
        if args.emit is not None:
            os.makedirs(args.emit, exist_ok=True)
            path = os.path.join(args.emit, kernel_name + ".hsaco")
            with open(path, "wb") as f:
                f.write(hsaco)
            print(f"[{args.arch}] emitted {path} (no run)")
            return 0

    rng = np.random.default_rng(0xA11E)
    Q = (rng.standard_normal((B, Sq, Hq, D)) * 0.3).astype(np.float16)
    K = (rng.standard_normal((B, Sk, Hk, D)) * 0.3).astype(np.float16)
    V = (rng.standard_normal((B, Sk, Hk, D)) * 0.3).astype(np.float16)
    Out = np.zeros((B, Sq, Hq, D), dtype=np.float16)

    # The reference consumes V row-major; only the device copy is relaid.
    if cfg.v_paged:
        V_dev, block_table = _build_paged_v(
            np, V, cfg.kv_block_size, shuffle=bool(args.shuffle_blocks)
        )
        bt_stride = block_table.shape[1]
        bt_num_entries = block_table.size
        print(
            f"[{args.arch}] paged V: cache{V_dev.shape} bs={cfg.kv_block_size} "
            f"block_table{block_table.shape} shuffled={bool(args.shuffle_blocks)}"
        )
    else:
        V_dev = swapqk_transpose_v(V) if cfg.v_transposed else V
        block_table, bt_stride, bt_num_entries = None, 0, 0
    scale_log2 = float(1.0 / math.sqrt(D) * math.log2(math.e))

    grid = swapqk_grid(cfg, seqlen_q=Sq, batch=B)
    block = (cfg.block_size, 1, 1)

    rt = Runtime()
    module = rt.load_module(hsaco)
    fn = module.get_function(kernel_name)

    def u8(a):
        return (ctypes.c_uint8 * int(a.nbytes)).from_buffer(np.ascontiguousarray(a))

    qd, kd, vd, od = (rt.alloc(x.nbytes) for x in (Q, K, V_dev, Out))
    rt.memcpy_h2d(qd, u8(Q), Q.nbytes)
    rt.memcpy_h2d(kd, u8(K), K.nbytes)
    rt.memcpy_h2d(vd, u8(V_dev), V_dev.nbytes)
    rt.memset(od, 0, Out.nbytes)
    btd = 0
    if block_table is not None:
        btd = rt.alloc(block_table.nbytes)
        rt.memcpy_h2d(btd, u8(block_table), block_table.nbytes)

    # Within-batch element strides; the kernel folds the batch axis in itself.
    # These stay the row-major values under v_transposed -- the transposed
    # addressing is derived from seqlen inside the kernel.
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
    if cfg.v_paged:
        # VBlockTable/bt_stride/bt_num_entries are appended LAST, so everything
        # above stays byte-identical. The "4x" is not cosmetic: the kernarg ABI
        # aligns each argument to its own size, and this pointer follows an ODD
        # number of i32s -- without the pad it lands 4 bytes early and the
        # kernel dereferences garbage.
        packed += struct.pack("<4xQii", btd, bt_stride, bt_num_entries)

    def launch_once():
        rt.launch(fn, grid, block, packed)

    launch_once()
    rt.sync()

    max_abs, bad = -1.0, 0
    if not args.no_verify:
        rt.memcpy_d2h(u8(Out), od, Out.nbytes)
        ref = np.empty_like(Out)
        for bi in range(B):
            if Hk != Hq:
                rep = Hq // Hk
                Kb = np.repeat(K[bi], rep, axis=1)
                Vb = np.repeat(V[bi], rep, axis=1)
            else:
                Kb, Vb = K[bi], V[bi]
            ref[bi] = _ref_attention(Q[bi], Kb, Vb, causal=args.causal)
        diff = np.abs(Out.astype(np.float32) - ref.astype(np.float32))
        max_abs = float(diff.max())
        bad = int(np.count_nonzero(diff > args.tol))

    ms = time_launches(launch_once, warmup=args.warmup, iters=args.iters)
    flops = 4.0 * B * Hq * Sq * Sk * D * (0.5 if args.causal else 1.0)
    tflops = flops / (ms * 1e-3) / 1e12

    for ptr in (qd, kd, vd, od):
        rt.free(ptr)
    module.unload()

    ok = args.no_verify or max_abs <= args.tol
    verdict = "SKIP" if args.no_verify else ("PASS" if ok else "FAIL")
    print(
        f"[{args.arch}] swapqk Sq={Sq} Sk={Sk} D={D} Hq={Hq} Hk={Hk} "
        f"causal={args.causal} bn={cfg.block_n} w={cfg.n_waves} "
        f"vt={int(cfg.v_transposed)} o_nt={int(cfg.o_nt)}: "
        f"{ms * 1e3:.1f}us {tflops:.2f} TF | verify={verdict}"
    )
    if not args.no_verify:
        print(
            f"    max_abs_diff={max_abs:.3e} bad={bad}/{Out.size} "
            f"tol={args.tol:.0e}"
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
