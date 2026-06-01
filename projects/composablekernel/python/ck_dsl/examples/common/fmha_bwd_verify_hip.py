# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""HIP-path (hipcc) FMHA backward build + numeric verify on CDNA.

Builds the warp-distributed FMHA backward kernel
(:func:`ck_dsl.instances.common.fmha_bwd.build_fmha_bwd`), compiles it
through the **HIP-C++ -> hipcc** backend (``compile_kernel_via_hipcc``),
launches it via the HIP runtime, and compares dQ / dK / dV against a
torch-free numpy reference (the correctness oracle is computed in fp32).

This is the backward analogue of
``ck_dsl.examples.common.fmha_fwd_verify_hip``: it closes the HIP-path
FMHA-backward numeric gate. The bwd body is warp-distributed scalar (one
wave64 warp per ``(q_token, head)`` CTA, head-dim distributed across
lanes) and accumulates dQ / dK / dV via f32 global atomics, so the
emitted IR is identical on gfx942 and gfx950.

The kernel recomputes ``P = softmax(Q @ K^T / sqrt(D))`` from saved
softmax statistics ``(M, L)``. Two equivalent encodings are supported via
``--stats-mode``:

  * native: ``M = max_k (S * log2 e)``, ``L = sum_k exp2(S*log2 e - M)``
    (the running max / denominator a fused forward emits).
  * bridge: ``M = log2(sum_k exp2(S*log2 e))`` (= LSE in the log2 domain),
    ``L = 1.0``. This is the encoding the hipDNN provider feeds the
    kernel (M = stats_LSE2, L = ones); ``p = exp2(s_log2 - M) / L``
    collapses to ``exp2(s_log2 - LSE2)`` which equals the true softmax
    probability. Both modes must yield identical dQ / dK / dV.

The accumulation order of the f32 atomic chains differs from numpy, so
parity is judged within a tolerance (default ``5e-2``, fp16 backward),
not bit-for-bit.

Must run on a device matching ``--arch`` (e.g. a gfx942 box).

    PYTHONPATH=python python3 -m ck_dsl.examples.common.fmha_bwd_verify_hip \
        --arch gfx942 --seqlen-q 64 --seqlen-k 64 --head-size 64 --heads 4
"""

from __future__ import annotations

import argparse
import ctypes
import math
import struct

from ck_dsl.core.arch import ArchTarget
from ck_dsl.helpers.compile import compile_kernel_via_hipcc
from ck_dsl.instances import FmhaCommonSpec, FmhaShape
from ck_dsl.instances.common.fmha_bwd import (
    FmhaBwdSpec,
    build_fmha_bwd,
    fmha_bwd_grid,
    is_valid_spec,
)
from ck_dsl.runtime.hip_module import Runtime


def _ref_backward(Q, K, V, dO, *, scale_inv, causal, Hkv):
    """Dense attention backward reference (fp32 oracle).

    Q / dO shape ``(Sq, Hq, D)``; K / V shape ``(Skv, Hkv, D)``.
    Returns (dQ, dK, dV) as fp32 arrays matching the kernel's gradient
    layouts: dQ ``(Sq, Hq, D)``, dK / dV ``(Skv, Hkv, D)``.

    For kv-head mapping ``hk = h // (Hq // Hkv)`` each query head h folds
    its dK / dV contribution into the kv head hk it shares; dQ is
    per-query-head.
    """
    import numpy as np

    Sq, Hq, D = Q.shape
    Skv = K.shape[0]
    rep = Hq // Hkv

    dQ = np.zeros((Sq, Hq, D), dtype=np.float32)
    dK = np.zeros((Skv, Hkv, D), dtype=np.float32)
    dV = np.zeros((Skv, Hkv, D), dtype=np.float32)

    for h in range(Hq):
        hk = h // rep
        Qh = Q[:, h, :].astype(np.float32)
        Kk = K[:, hk, :].astype(np.float32)
        Vk = V[:, hk, :].astype(np.float32)
        dOh = dO[:, h, :].astype(np.float32)

        S = (Qh @ Kk.T) * scale_inv  # [Sq, Skv]
        if causal:
            q_pos = np.arange(Sq)[:, None]
            k_pos = np.arange(Skv)[None, :]
            S = np.where(k_pos <= q_pos, S, -np.inf)
        # Row softmax (fp32, numerically stable).
        S = S - S.max(axis=1, keepdims=True)
        P = np.exp(S)
        P = P / P.sum(axis=1, keepdims=True)  # [Sq, Skv]

        dV[:, hk, :] += P.T @ dOh
        dP = dOh @ Vk.T  # [Sq, Skv]
        dS = P * (dP - (dP * P).sum(axis=1, keepdims=True))  # softmax jacobian
        dQ[:, h, :] = (dS @ Kk) * scale_inv
        dK[:, hk, :] += (dS.T @ Qh) * scale_inv

    return dQ, dK, dV


def _ref_stats(Q, K, *, scale_inv, causal, Hq, Hkv, mode):
    """Saved softmax statistics M / L, flat length ``Sq*Hq`` (q*Hq+h).

    Computed from the SAME forward (and SAME mask) as the reference. For
    masked entries exp2 contributes 0.

    native: ``Slog2 = S * log2 e``; ``M = max_k Slog2`` (over unmasked k),
            ``L = sum_k exp2(Slog2 - M)``.
    bridge: ``M = log2(sum_k exp2(Slog2))`` (LSE in log2 domain),
            ``L = 1.0``.
    """
    import numpy as np

    log2e = math.log2(math.e)
    Sq = Q.shape[0]
    Skv = K.shape[0]
    rep = Hq // Hkv

    M = np.empty((Sq, Hq), dtype=np.float32)
    L = np.empty((Sq, Hq), dtype=np.float32)

    for h in range(Hq):
        hk = h // rep
        Qh = Q[:, h, :].astype(np.float32)
        Kk = K[:, hk, :].astype(np.float32)
        S = (Qh @ Kk.T) * scale_inv  # [Sq, Skv]
        Slog2 = S * log2e
        if causal:
            q_pos = np.arange(Sq)[:, None]
            k_pos = np.arange(Skv)[None, :]
            Slog2 = np.where(k_pos <= q_pos, Slog2, -np.inf)
        row_max = Slog2.max(axis=1)  # [Sq]
        denom = np.exp2(Slog2 - row_max[:, None]).sum(axis=1)  # [Sq]
        if mode == "native":
            M[:, h] = row_max
            L[:, h] = denom
        else:  # bridge: M = LSE2 = row_max + log2(denom), L = 1.0
            M[:, h] = row_max + np.log2(denom)
            L[:, h] = 1.0

    # Flat layout indexed q*Hq + h (q-major).
    return M.reshape(-1), L.reshape(-1)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", default="gfx950")
    p.add_argument("--seqlen-q", type=int, default=64)
    p.add_argument("--seqlen-k", type=int, default=64)
    p.add_argument("--head-size", type=int, default=64)
    p.add_argument("--heads", type=int, default=4, help="Hq")
    p.add_argument("--kv-heads", type=int, default=0, help="Hkv; 0 -> MHA (== heads)")
    p.add_argument(
        "--batch",
        type=int,
        default=1,
        help="batch size; only 1 is supported (the bwd grid has no batch "
        "axis -- multi-batch needs a per-batch launch loop, out of scope "
        "for this kernel harness)",
    )
    p.add_argument("--causal", action="store_true", help="top-left causal mask")
    p.add_argument(
        "--stats-mode",
        choices=("native", "bridge"),
        default="native",
        help="saved-statistics encoding for M / L; both yield the same "
        "dQ / dK / dV (bridge = provider's M=LSE2, L=ones)",
    )
    p.add_argument("--tol", type=float, default=5e-2)
    p.add_argument("--seed", type=int, default=0xA11E)
    args = p.parse_args()

    import numpy as np

    # batch = 1 ONLY: the bwd grid is (seqlen_q, num_query_heads, 1) with no
    # batch axis, so each launch handles exactly one batch element.
    if args.batch != 1:
        raise SystemExit(
            f"--batch={args.batch} unsupported: the fmha_bwd grid has no batch "
            "axis (B=1 only; multi-batch needs a per-batch launch loop)"
        )

    # head_size % 64 == 0 (wave64 warp body): D in {64, 128, 192, 256}.
    if args.head_size % 64:
        raise SystemExit(
            f"head_size={args.head_size} must be a multiple of 64 (wave64 warp "
            "body); supported D in {64, 128, 192, 256}"
        )
    for d, name in ((args.seqlen_q, "seqlen_q"), (args.seqlen_k, "seqlen_k")):
        if d % 16:
            raise SystemExit(f"{name}={d} must be a multiple of 16")

    Hq = args.heads
    Hkv = args.kv_heads or args.heads
    if Hq % Hkv:
        raise SystemExit(f"heads={Hq} must be divisible by kv-heads={Hkv} (GQA)")

    target = ArchTarget.from_gfx(args.arch)
    D = args.head_size
    Sq, Skv = args.seqlen_q, args.seqlen_k

    common = FmhaCommonSpec(
        shape=FmhaShape(head_size=D, num_query_heads=Hq, num_kv_heads=Hkv),
        dtype="f16",
        mask_mode="causal" if args.causal else "none",
    )
    spec = FmhaBwdSpec(common=common, seqlen_q=Sq, seqlen_k=Skv)

    ok, why = is_valid_spec(spec, args.arch)
    print(
        f"[{args.arch}] HIP-path FMHA bwd "
        f"H{D} HQ{Hq} HK{Hkv} Q{Sq} K{Skv} "
        f"mask={common.mask_mode} stats={args.stats_mode} "
        f"wave={target.wave_size} validate -> {ok} ({why})"
    )
    if not ok:
        return 2

    art = compile_kernel_via_hipcc(build_fmha_bwd(spec, arch=args.arch), arch=args.arch)
    print(
        f"[{args.arch}] HIP-path built {art.kernel_name} "
        f"({art.hsaco_bytes} B, isa={art.isa}) "
        f"hipcc={art.timings.get('hipcc', 0):.0f}ms "
        f"total={art.timings.get('total', 0):.0f}ms"
    )

    rng = np.random.default_rng(args.seed)

    # BSHD, batch=1. Q / dO: [Sq, Hq, D]; K / V: [Skv, Hkv, D]; small fp16.
    Q = (rng.standard_normal((Sq, Hq, D)) * 0.1).astype(np.float16)
    K = (rng.standard_normal((Skv, Hkv, D)) * 0.1).astype(np.float16)
    V = (rng.standard_normal((Skv, Hkv, D)) * 0.1).astype(np.float16)
    dO = (rng.standard_normal((Sq, Hq, D)) * 0.1).astype(np.float16)

    # Gradient accumulators (f32) -- the kernel atomic-adds into these, so
    # they MUST be zero-initialised before launch.
    dQ = np.zeros((Sq, Hq, D), dtype=np.float32)
    dK = np.zeros((Skv, Hkv, D), dtype=np.float32)
    dV = np.zeros((Skv, Hkv, D), dtype=np.float32)

    # Element strides (row-major). Head stride is implicit for the gradient
    # tensors (== Q head stride for dQ, == K / V head stride for dK / dV),
    # so the ABI only carries the per-tensor token strides for dQ / dK / dV.
    stride_q_token = Hq * D
    stride_q_head = D
    stride_k_token = Hkv * D
    stride_k_head = D
    stride_v_token = Hkv * D
    stride_v_head = D
    stride_do_token = Hq * D
    stride_do_head = D
    stride_dq_token = Hq * D
    stride_dk_token = Hkv * D
    stride_dv_token = Hkv * D

    scale_inv = float(1.0 / math.sqrt(D))
    scale_log2 = float(scale_inv * math.log2(math.e))

    # Saved softmax statistics M / L, flat length Sq*Hq indexed q*Hq + h.
    M_saved, L_saved = _ref_stats(
        Q, K, scale_inv=scale_inv, causal=args.causal, Hq=Hq, Hkv=Hkv,
        mode=args.stats_mode,
    )

    grid = fmha_bwd_grid(spec)  # (seqlen_q, num_query_heads, 1)
    block = (target.wave_size, 1, 1)

    rt = Runtime()
    module = rt.load_module(art.hsaco)
    fn = module.get_function(art.kernel_name)

    def u8(a):
        return (ctypes.c_uint8 * int(a.nbytes)).from_buffer(np.ascontiguousarray(a))

    qd = rt.alloc(Q.nbytes)
    kd = rt.alloc(K.nbytes)
    vd = rt.alloc(V.nbytes)
    dod = rt.alloc(dO.nbytes)
    md = rt.alloc(M_saved.nbytes)
    ld = rt.alloc(L_saved.nbytes)
    dqd = rt.alloc(dQ.nbytes)
    dkd = rt.alloc(dK.nbytes)
    dvd = rt.alloc(dV.nbytes)

    rt.memcpy_h2d(qd, u8(Q), Q.nbytes)
    rt.memcpy_h2d(kd, u8(K), K.nbytes)
    rt.memcpy_h2d(vd, u8(V), V.nbytes)
    rt.memcpy_h2d(dod, u8(dO), dO.nbytes)
    rt.memcpy_h2d(md, u8(M_saved), M_saved.nbytes)
    rt.memcpy_h2d(ld, u8(L_saved), L_saved.nbytes)
    # dQ / dK / dV are atomic-accumulated -> zero-initialise on device.
    rt.memset(dqd, 0, dQ.nbytes)
    rt.memset(dkd, 0, dK.nbytes)
    rt.memset(dvd, 0, dV.nbytes)

    # Kernarg pack (24 args): 9 ptrs, 2 f32, then 13 i32 (seqlen_q, seqlen_k
    # + 11 strides). All naturally aligned (9*8=72, +2*4=80, +13*4=132 B);
    # no extra pad. Order mirrors fmha_bwd._declare_params exactly.
    packed = struct.pack(
        "<9Q2f13i",
        qd, kd, vd, dod, md, ld, dqd, dkd, dvd,
        scale_log2, scale_inv,
        Sq, Skv,
        stride_q_token, stride_q_head,
        stride_k_token, stride_k_head,
        stride_v_token, stride_v_head,
        stride_do_token, stride_do_head,
        stride_dq_token, stride_dk_token, stride_dv_token,
    )
    rt.launch(fn, grid, block, packed)
    rt.sync()

    rt.memcpy_d2h(u8(dQ), dqd, dQ.nbytes)
    rt.memcpy_d2h(u8(dK), dkd, dK.nbytes)
    rt.memcpy_d2h(u8(dV), dvd, dV.nbytes)

    dQ_ref, dK_ref, dV_ref = _ref_backward(
        Q, K, V, dO, scale_inv=scale_inv, causal=args.causal, Hkv=Hkv
    )

    for ptr in (qd, kd, vd, dod, md, ld, dqd, dkd, dvd):
        rt.free(ptr)
    module.unload()

    results = {}
    all_ok = True
    for name, got, ref in (
        ("dQ", dQ, dQ_ref),
        ("dK", dK, dK_ref),
        ("dV", dV, dV_ref),
    ):
        diff = np.abs(got.astype(np.float32) - ref.astype(np.float32))
        max_abs = float(diff.max())
        results[name] = max_abs
        all_ok = all_ok and (max_abs <= args.tol)

    tag = "PASS" if all_ok else "FAIL"
    print(
        f"[{args.arch}] HIP-path FMHA bwd Sq={Sq} Sk={Skv} D={D} "
        f"Hq={Hq} Hk={Hkv} causal={args.causal} stats={args.stats_mode}: "
        f"dQ={results['dQ']:.3e} dK={results['dK']:.3e} dV={results['dV']:.3e} "
        f"tol={args.tol:.0e} -> {tag}"
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
