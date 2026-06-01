# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""HIP-path (hipcc) FMHA forward build + numeric verify on CDNA and RDNA.

Builds the unified tiled FMHA forward kernel
(:func:`ck_dsl.instances.common.fmha_mfma.build_fmha_fwd_mfma` -- one body,
MFMA on CDNA / WMMA on RDNA wave32), compiles it through the **HIP-C++ ->
hipcc** backend (``compile_kernel_via_hipcc``) instead of the LLVM-IR ->
libamd_comgr backend, launches it via the HIP runtime, and compares the
output against a torch-free numpy dense-attention reference.

This is the HIP-path analogue of
``ck_dsl.examples.gfx1151.wmma_fmha_fwd_verify`` (which exercises the
LLVM->comgr path). It closes the HIP-path FMHA numeric gate:

  * gfx950 / gfx942: ``__builtin_amdgcn_mfma_*`` C++ builtins (wave64).
  * gfx1151:         ``__builtin_amdgcn_wmma_*_w32`` C++ builtins (wave32).

The accumulation order of the f32 matmul chain differs from numpy, so parity
is judged within a tolerance (default ``2e-2``, matching the attention parity
gate), not bit-for-bit.

Must run on a device matching ``--arch`` (e.g. alola
``--gres=gpu:gfx950-mi350x:1`` or ``--gres=gpu:gfx1151:1``).

    PYTHONPATH=python python3 -m ck_dsl.examples.common.fmha_fwd_verify_hip \
        --arch gfx950 --seqlen-q 64 --seqlen-k 64 --head-size 64 --heads 4
"""

from __future__ import annotations

import argparse
import ctypes
import math
import struct

from ck_dsl.core.arch import ArchTarget
from ck_dsl.helpers.compile import compile_kernel_via_hipcc
from ck_dsl.instances import FmhaCommonSpec, FmhaShape
from ck_dsl.instances.common.fmha_mfma import (
    FmhaMfmaSpec,
    build_fmha_fwd_mfma,
    fmha_fwd_mfma_grid,
    is_valid_spec,
)
from ck_dsl.runtime.hip_module import Runtime


def _ref_attention(Q, K, V, *, causal: bool):
    """Dense attention reference, Q/K/V shape ``(seqlen, heads, head_size)``.

    Mirrors ``examples.gfx1151.wmma_fmha_fwd_verify._ref_attention`` (fp32
    math, fp16 out) so the two harnesses share an oracle.
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", default="gfx950")
    p.add_argument("--seqlen-q", type=int, default=64)
    p.add_argument("--seqlen-k", type=int, default=64)
    p.add_argument("--head-size", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--kv-heads", type=int, default=0, help="0 -> MHA (== heads)")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument(
        "--causal",
        action="store_true",
        help="causal mask; correct for any batch (G4: the unified body uses "
        "the within-batch query position q_pos_base for the mask while "
        "addressing Q/Out at the global-batched row)",
    )
    p.add_argument("--tol", type=float, default=2e-2)
    p.add_argument(
        "--stats",
        action="store_true",
        help="opt-in softmax-stats: build with generate_stats=True, store "
        "per-(query-row, query-head) M / L, and verify them against a numpy "
        "log2-domain reference (flat [B*Sq, HQ] f32 layout)",
    )
    p.add_argument("--no-verify", action="store_true")
    args = p.parse_args()

    import numpy as np

    # G4: causal / sliding-window masking is now correct for batch > 1. The
    # unified body addresses Q/Out at the global-batched row (q_tile_base) but
    # masks against the within-batch query position (q_pos_base), so each
    # batch's row 0 is causal position 0. No batch restriction needed.

    for d, name in (
        (args.seqlen_q, "seqlen_q"),
        (args.seqlen_k, "seqlen_k"),
        (args.head_size, "head_size"),
    ):
        if d % 16:
            raise SystemExit(f"{name}={d} must be a multiple of 16 (16x16 tile)")

    target = ArchTarget.from_gfx(args.arch)
    family = "WMMA" if target.wave_size == 32 else "MFMA"
    kvh = args.kv_heads or args.heads
    common = FmhaCommonSpec(
        FmhaShape(
            head_size=args.head_size,
            num_query_heads=args.heads,
            num_kv_heads=kvh,
            block_size_q=16,
            block_size_k=64,
        ),
        dtype="f16",
        mask_mode="causal" if args.causal else "none",
        generate_stats=args.stats,
    )
    spec = FmhaMfmaSpec(
        common=common,
        seqlen_q=args.seqlen_q,
        seqlen_k=args.seqlen_k,
        name=f"ck_dsl_fmha_fwd_hip_{args.arch}",
    )

    ok, why = is_valid_spec(spec, args.arch)
    print(
        f"[{args.arch}] HIP-path {family} FMHA fwd "
        f"H{args.head_size} HQ{args.heads} HK{kvh} "
        f"Q{args.seqlen_q} K{args.seqlen_k} wave={target.wave_size} "
        f"validate -> {ok} ({why})"
    )
    if not ok:
        return 2

    art = compile_kernel_via_hipcc(
        build_fmha_fwd_mfma(spec, arch=args.arch), arch=args.arch
    )
    print(
        f"[{args.arch}] HIP-path built {art.kernel_name} "
        f"({art.hsaco_bytes} B, isa={art.isa}) "
        f"hipcc={art.timings.get('hipcc', 0):.0f}ms "
        f"total={art.timings.get('total', 0):.0f}ms"
    )

    if args.no_verify:
        print(f"[{args.arch}] HIP-path build OK (verify skipped)")
        return 0

    B, Hq, Hk, D = args.batch, args.heads, kvh, args.head_size
    Sq, Sk = args.seqlen_q, args.seqlen_k
    rng = np.random.default_rng(0xA11E)

    # Per-batch tensors: [batch, seqlen, heads, head_size] row-major.
    Q = (rng.standard_normal((B, Sq, Hq, D)) * 0.3).astype(np.float16)
    K = (rng.standard_normal((B, Sk, Hk, D)) * 0.3).astype(np.float16)
    V = (rng.standard_normal((B, Sk, Hk, D)) * 0.3).astype(np.float16)
    Out = np.zeros((B, Sq, Hq, D), dtype=np.float16)

    # Element strides (row-major within a batch). The kernel folds the batch
    # axis in via seqlen * batch_idx, so the host strides are the within-batch
    # strides.
    stride_q_token = Hq * D
    stride_q_head = D
    stride_k_token = Hk * D
    stride_k_head = D
    stride_v_token = Hk * D
    stride_v_head = D
    stride_o_token = Hq * D
    stride_o_head = D
    scale_log2 = float(1.0 / math.sqrt(D) * math.log2(math.e))

    grid = fmha_fwd_mfma_grid(spec, batch=B)
    block = (target.wave_size, 1, 1)

    rt = Runtime()
    module = rt.load_module(art.hsaco)
    fn = module.get_function(art.kernel_name)

    def u8(a):
        return (ctypes.c_uint8 * int(a.nbytes)).from_buffer(np.ascontiguousarray(a))

    qd = rt.alloc(Q.nbytes)
    kd = rt.alloc(K.nbytes)
    vd = rt.alloc(V.nbytes)
    od = rt.alloc(Out.nbytes)
    rt.memcpy_h2d(qd, u8(Q), Q.nbytes)
    rt.memcpy_h2d(kd, u8(K), K.nbytes)
    rt.memcpy_h2d(vd, u8(V), V.nbytes)
    rt.memset(od, 0, Out.nbytes)

    # Opt-in softmax-stats: a single f32 natural-log LSE array in the
    # contract-standard head-major [B, Hq, Sq] layout, indexed
    # ((b*Hq + h)*Sq + q) -- matching the hipDNN stats descriptor that the
    # bwd graph's ``stats`` tensor uses.
    LSE = np.zeros((B * Hq * Sq,), dtype=np.float32)
    if args.stats:
        lse_d = rt.alloc(LSE.nbytes)
        rt.memset(lse_d, 0, LSE.nbytes)

    # The opt-in stats pointer appends AFTER the 8 strides, matching the
    # tail-of-ABI declaration order in fmha_mfma._declare_params.
    pack_fmt = "<QQQQfiiiiiiiiii"
    pack_args = [
        qd,
        kd,
        vd,
        od,
        scale_log2,
        Sq,
        Sk,
        stride_q_token,
        stride_q_head,
        stride_k_token,
        stride_k_head,
        stride_v_token,
        stride_v_head,
        stride_o_token,
        stride_o_head,
    ]
    if args.stats:
        # The f32 LSE pointer is an 8-byte-aligned kernarg. The preceding
        # 10 i32 scalars end at byte 76, so 4 pad bytes are needed to reach
        # the pointer's natural offset 80 (the kernarg layout aligns each
        # pointer to 8; the provider's LaunchAbi packer does this
        # automatically).
        pack_fmt += "4xQ"
        pack_args += [lse_d]
    packed = struct.pack(pack_fmt, *pack_args)
    rt.launch(fn, grid, block, packed)
    rt.sync()
    rt.memcpy_d2h(u8(Out), od, Out.nbytes)
    if args.stats:
        rt.memcpy_d2h(u8(LSE), lse_d, LSE.nbytes)

    # Reference per batch.
    ref = np.empty_like(Out)
    for bi in range(B):
        if Hk != Hq:  # GQA: expand kv heads to query heads for the reference.
            rep = Hq // Hk
            Kb = np.repeat(K[bi], rep, axis=1)
            Vb = np.repeat(V[bi], rep, axis=1)
        else:
            Kb, Vb = K[bi], V[bi]
        ref[bi] = _ref_attention(Q[bi], Kb, Vb, causal=args.causal)

    diff = np.abs(Out.astype(np.float32) - ref.astype(np.float32))
    max_abs = float(diff.max())
    bad = int(np.count_nonzero(diff > args.tol))

    # Opt-in softmax-stats verify: compare the kernel's stored natural-log
    # LSE against a numpy reference computed with the SAME mask as the run.
    #   scores_nat = (Q @ K^T) * (1/sqrt(D))            [natural domain, post-mask]
    #   LSE_nat = logsumexp_j scores_nat = max + ln(sum(exp(s - max)))
    # flattened head-major to ((b*Hq + h)*Sq + q). Absolute compare at tol 1e-2.
    stats_ok = True
    if args.stats:
        # Head-major [B, Hq, Sq] so the flatten matches the kernel's
        # ((b*Hq + h)*Sq + q) store layout.
        LSE_ref = np.empty((B, Hq, Sq), dtype=np.float32)
        for bi in range(B):
            if Hk != Hq:  # GQA: expand kv heads to query heads for the ref.
                rep = Hq // Hk
                Kb = np.repeat(K[bi], rep, axis=1)
            else:
                Kb = K[bi]
            # scores: (Sq, Hq, Sk) in the natural domain.
            scores = np.einsum(
                "ihd,jhd->ihj", Q[bi].astype(np.float32), Kb.astype(np.float32)
            )
            scores = scores * (1.0 / math.sqrt(D))
            if args.causal:
                q_pos = np.arange(Sq)[:, None, None]
                k_pos = np.arange(Sk)[None, None, :]
                scores = np.where(k_pos <= q_pos, scores, -1e30)
            mx = scores.max(axis=-1)
            # lse_q_major: (Sq, Hq) -> transpose to (Hq, Sq) for head-major.
            lse_q_major = mx + np.log(np.exp(scores - mx[..., None]).sum(axis=-1))
            LSE_ref[bi] = lse_q_major.T
        LSE_ref_flat = LSE_ref.reshape(B * Hq * Sq)
        lse_abs = float(np.abs(LSE - LSE_ref_flat).max())
        stats_ok = lse_abs <= 1e-2
        print(
            f"[{args.arch}] HIP-path {family} FMHA stats: LSE_max_abs_diff="
            f"{lse_abs:.3e} (tol 1e-2 -> {'PASS' if stats_ok else 'FAIL'})"
        )

    free_ptrs = [qd, kd, vd, od]
    if args.stats:
        free_ptrs += [lse_d]
    for ptr in free_ptrs:
        rt.free(ptr)
    module.unload()

    ok = (max_abs <= args.tol) and stats_ok
    tag = "PASS" if ok else "FAIL"
    print(
        f"[{args.arch}] HIP-path {family} FMHA Sq={Sq} Sk={Sk} D={D} "
        f"Hq={Hq} Hk={Hk} causal={args.causal}: max_abs_diff={max_abs:.3e} "
        f"bad={bad}/{Out.size} tol={args.tol:.0e} -> {tag}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
