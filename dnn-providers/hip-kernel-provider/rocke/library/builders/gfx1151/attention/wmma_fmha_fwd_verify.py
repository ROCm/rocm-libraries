# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build + numeric-verify the gfx1151 WMMA FMHA forward on a Strix Halo node.

Builds the RDNA3.5 WMMA attention kernel, launches it via the HIP runtime, and
compares the output against a torch dense-attention reference (the same math as
``rocke.examples.common.parity_extended_kernels._ref_attention``):

    scores = (Q @ K^T) / sqrt(d)            [+ causal mask]
    probs  = softmax(scores, dim=-1)
    Out      = probs @ V

Must run on a gfx1151 device (e.g. ``--gres=gpu:gfx1151:1`` on a SLURM cluster).

    PYTHONPATH=python python3 -m builders.gfx1151.attention.wmma_fmha_fwd_verify \
        --seqlen-q 64 --seqlen-k 64 --head-size 64 --heads 4

``--dtype {fp16,bf16}`` selects the Q/K/V/O I/O type; the WMMA accumulator and
the softmax are f32 on both. bf16 has no numpy host type, so it is carried as
raw ``uint16`` and converted by hand (``_f32_to_bf16`` / ``_bf16_to_f32``).

The accumulation order of the WMMA f32 chain differs from torch, so parity is
judged within a tolerance (fp16 ``2e-2``, matching the attention parity gate),
not bit-for-bit; see ``_DEFAULT_TOL`` for the per-dtype budgets and the gfx1151
measurements they were set from.

Both RDNA targets are numerically verified: pass ``--arch gfx1151`` or
``--arch gfx1201`` (the module lives under ``builders/gfx1151/`` but is
arch-parameterised, and ``--arch`` selects the matching WMMA atom).
"""

from __future__ import annotations

import argparse
import ctypes
import math
import struct

from rocke.helpers import compile_kernel
from kernels.gfx1151.wmma_fmha_fwd import (
    WmmaFmhaFwdSpec,
    build_wmma_fmha_fwd,
    wmma_fmha_fwd_grid,
)
from rocke.numeric.bf16 import bf16_to_f32, f32_to_bf16
from rocke.runtime.hip_module import Runtime


# Per-dtype parity tolerance. numpy is the only hard dependency, so bf16 has no
# native host type: it is carried as raw uint16 and converted by hand (see
# ``_f32_to_bf16`` / ``_bf16_to_f32``), per TESTING.md -- "bf16 gets a
# hand-rolled encoding or an explicit NotImplementedError, never a silent
# upcast". bf16 keeps 8 mantissa bits against fp16's 11, so it gets its own,
# looser budget rather than reusing fp16's -- a shared tolerance would let a
# structural bug hide inside bf16 quantisation noise (TESTING.md gap G5).
#
# Measured on gfx1151 and gfx1201, B=2 Sq=Sk=64, inputs ~N(0, 0.3).
# Both arches produce identical figures:
#
#   bf16  D=64  Hq=Hk=4  causal   max_abs = 1.95e-03  (2^-9)
#   bf16  D=128 Hq=8 Hk=2 MHA/GQA max_abs = 4.88e-04  (2^-11)
#   fp16  D=64  Hq=Hk=4  causal   max_abs = 2.44e-04  (2^-12)
#
# Those are exact powers of two -- each is a half-ULP of the output dtype at the
# magnitude of the largest output. The kernel therefore agrees with the fp32
# reference to within one rounding step of the I/O type, and the residual is
# output quantisation rather than arithmetic divergence. That is also why the
# two arches agree exactly despite different WMMA generations: the quantisation
# is deterministic given the same inputs. A structural bug would land orders of
# magnitude above this floor, not one ULP above it.
#
# bf16's budget is the worst observed 1.95e-03 with ~5x headroom for other
# shapes and seeds. That is deliberately far tighter than fp16's long-standing
# 2e-2 gate: a tolerance loose enough to swallow bf16 quantisation noise would
# also swallow a structural bug (TESTING.md gap G5). Judge on absolute error --
# max_rel runs to ~3e-01 purely on near-zero outputs, where it is meaningless.
_DEFAULT_TOL = {"fp16": 2e-2, "bf16": 1e-2}


# The bf16 host codec is shared with the library/tests numeric gate; see
# rocke.numeric.bf16 for why there is exactly one implementation.
_f32_to_bf16 = f32_to_bf16
_bf16_to_f32 = bf16_to_f32


def _to_storage(a_f32, dtype: str):
    """fp32 -> the on-device storage array (np.float16, or raw uint16 for bf16)."""
    import numpy as np

    if dtype == "bf16":
        return _f32_to_bf16(a_f32)
    return a_f32.astype(np.float16)


def _from_storage(a, dtype: str):
    """Storage array -> fp32."""
    import numpy as np

    if dtype == "bf16":
        return _bf16_to_f32(a)
    return a.astype(np.float32)


def _ref_attention(Q, K, V, *, causal: bool, dtype: str):
    """Dense attention reference, Q/K/V shape ``(seqlen, heads, head_size)``.

    Mirrors ``parity_extended_kernels._ref_attention``: fp32 math throughout,
    truncated to the I/O dtype only at the end (matching the kernel, whose WMMA
    accumulator and softmax are f32 on every dtype). ``Q``/``K``/``V`` are fp32.
    """
    import numpy as np

    d = Q.shape[-1]
    scores = np.einsum("ihd,jhd->ihj", Q, K)
    scores /= math.sqrt(d)
    if causal:
        q_pos = np.arange(Q.shape[0])[:, None, None]
        k_pos = np.arange(K.shape[0])[None, None, :]
        scores = np.where(k_pos <= q_pos, scores, -1e30)
    scores -= scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    out = np.einsum("ihj,jhd->ihd", probs, V)
    return _to_storage(out, dtype)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", default="gfx1151")
    p.add_argument("--seqlen-q", type=int, default=64)
    p.add_argument("--seqlen-k", type=int, default=64)
    p.add_argument("--head-size", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--kv-heads", type=int, default=0, help="0 -> MHA (== heads)")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--causal", action="store_true")
    p.add_argument(
        "--dtype",
        default="fp16",
        choices=("fp16", "bf16"),
        help="Q/K/V/O I/O dtype; accumulate is fp32 either way",
    )
    p.add_argument(
        "--tol",
        type=float,
        default=None,
        help="parity tolerance; default is per-dtype (see _DEFAULT_TOL)",
    )
    p.add_argument("--no-verify", action="store_true")
    args = p.parse_args()
    if args.tol is None:
        args.tol = _DEFAULT_TOL[args.dtype]

    import numpy as np

    for d, name in (
        (args.seqlen_q, "seqlen_q"),
        (args.seqlen_k, "seqlen_k"),
        (args.head_size, "head_size"),
    ):
        if d % 16:
            raise SystemExit(f"{name}={d} must be a multiple of 16 (WMMA 16x16 tile)")

    kvh = args.kv_heads or args.heads
    spec = WmmaFmhaFwdSpec(
        head_size=args.head_size,
        num_query_heads=args.heads,
        num_kv_heads=kvh,
        dtype=args.dtype,
        mask_mode="causal" if args.causal else "none",
        name=f"wmma_fmha_{args.arch}",
    )
    art = compile_kernel(build_wmma_fmha_fwd(spec, arch=args.arch), arch=args.arch)
    print(
        f"[{args.arch}] built {art.kernel_name} ({art.hsaco_bytes} B, isa={art.isa}) "
        f"total={art.timings.get('total', 0):.1f}ms"
    )

    if args.no_verify:
        print(f"[{args.arch}] build OK (verify skipped)")
        return 0

    B, Hq, Hk, D = args.batch, args.heads, kvh, args.head_size
    Sq, Sk = args.seqlen_q, args.seqlen_k
    rng = np.random.default_rng(0xA11E)

    # Per-batch tensors: [batch, seqlen, heads, head_size] row-major.
    # Quantise to the I/O dtype on the host, then decode back to fp32 so the
    # reference sees exactly the values the kernel reads -- otherwise the input
    # rounding would show up as "error" on top of the kernel's own.
    Q = _to_storage(
        (rng.standard_normal((B, Sq, Hq, D)) * 0.3).astype(np.float32), args.dtype
    )
    K = _to_storage(
        (rng.standard_normal((B, Sk, Hk, D)) * 0.3).astype(np.float32), args.dtype
    )
    V = _to_storage(
        (rng.standard_normal((B, Sk, Hk, D)) * 0.3).astype(np.float32), args.dtype
    )
    Out = np.zeros((B, Sq, Hq, D), dtype=Q.dtype)
    Qf, Kf, Vf = (_from_storage(t, args.dtype) for t in (Q, K, V))

    # Element strides (row-major within a batch). The kernel folds the batch
    # axis in via seqlen * batch_idx, so the host strides are the within-batch
    # strides (the kernel multiplies the batch index by seqlen internally).
    stride_q_token = Hq * D
    stride_q_head = D
    stride_k_token = Hk * D
    stride_k_head = D
    stride_v_token = Hk * D
    stride_v_head = D
    stride_o_token = Hq * D
    stride_o_head = D
    scale_log2 = float(1.0 / math.sqrt(D) * math.log2(math.e))

    grid = wmma_fmha_fwd_grid(spec, seqlen_q=Sq, batch=B)
    block = (spec.block_size, 1, 1)

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

    packed = struct.pack(
        "<QQQQfiiiiiiiiii",
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
    )
    rt.launch(fn, grid, block, packed)
    rt.sync()
    rt.memcpy_d2h(u8(Out), od, Out.nbytes)

    # Reference per batch.
    ref = np.empty_like(Out)
    for bi in range(B):
        # GQA: expand kv heads to query heads for the reference.
        if Hk != Hq:
            n_rep = Hq // Hk
            Kb = np.repeat(Kf[bi], n_rep, axis=1)
            Vb = np.repeat(Vf[bi], n_rep, axis=1)
        else:
            Kb, Vb = Kf[bi], Vf[bi]
        ref[bi] = _ref_attention(Qf[bi], Kb, Vb, causal=args.causal, dtype=args.dtype)

    diff = np.abs(_from_storage(Out, args.dtype) - _from_storage(ref, args.dtype))
    max_abs = float(diff.max())
    bad = int(np.count_nonzero(diff > args.tol))
    for ptr in (qd, kd, vd, od):
        rt.free(ptr)
    module.unload()

    ok = max_abs <= args.tol
    tag = "PASS" if ok else "FAIL"
    print(
        f"[{args.arch}] WMMA FMHA {args.dtype} Sq={Sq} Sk={Sk} D={D} Hq={Hq} Hk={Hk} "
        f"causal={args.causal}: max_abs_diff={max_abs:.3e} "
        f"bad={bad}/{Out.size} tol={args.tol:.0e} -> {tag}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
