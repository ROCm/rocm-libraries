# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""End-to-end parity harness for the extended CK Tile DSL kernel family.

This is the sibling of :mod:`ck_dsl.examples.ck_tile_parity` (which
covers elementwise / layernorm / transpose / batched GEMM / grouped
GEMM) and :mod:`ck_dsl.examples.attention.parity_unified_attention`
(which covers the tiled MFMA attention paths). This harness validates
the v1 scalar-inner kernels added across the extended kernel family:

* Fused MoE: ``moe_gather``, ``moe_silu_mul``, ``moe_topk_weighted_reduce``.
* FMHA expansion: ``fmha_fwd_varlen``, ``fmha_fwd_head_grouping``,
  ``fmha_appendkv`` (with + without rotary), ``fmha_fwd_paged_prefill``,
  ``fmha_fwd_splitkv_decode`` (two-launch), ``fmha_fwd_fp8``.
* Sage attention (CK Tile ``49_sageattention``): ``fp16_bf16``,
  ``fp8_bf16`` quant modes (the ``i8`` / ``i4`` codebook variants are
  smoke-tested at the IR level only; per-tensor codebook reference
  setup is a follow-on).
* Sparse attention (CK Tile ``50_sparse_attn``): ``jenga`` (block-sparse)
  and ``vsa`` (variable-size).

Caveats:

* The v1 scalar-inner kernels are *correctness oracles*, not perf
  targets. The harness launches them with ``block=(1, 1, 1)`` (one
  thread per CTA) because the inner body doesn't distribute work
  across threads. With this launch shape, the kernel takes Nx the
  ideal MFMA wall time but matches the torch reference bit-for-bit
  (or within f16/exp2 ULP).
* The ``streamk_gemm`` v1 (persistent grid + atomic accumulate)
  fails parity at this size in the current scalar-inner form and is
  excluded from this harness pending the MFMA v2 hoist.
* No torch-baseline timing is reported here: the v1 kernels' wall
  time is dominated by the scalar inner loop and isn't comparable to
  a tiled MFMA torch op.

Usage::

    /workspace/dsl_bake_off/venv/bin/python3 \\
        python/ck_dsl/examples/parity_extended_kernels.py [--op all]

The harness exits non-zero if any kernel's max abs diff exceeds the
per-kernel tolerance, so it doubles as a smoke gate for CI.
"""

# Standard attention notation: ``O`` is the output tensor (Q, K, V, O).
# Ruff's ambiguous-name lint (E741) is silenced for this file because the
# math notation is the primary readability win.
# ruff: noqa: E741

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path
from typing import Callable, Dict

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

import torch  # noqa: E402

from ck_dsl.helpers import QkScaleSpec, compile_kernel  # noqa: E402
from ck_dsl.helpers.rotary import RotarySpec  # noqa: E402
from ck_dsl.instances import (  # noqa: E402
    BlockScaleGemmSpec,
    FmhaAppendKvSpec,
    FmhaBwdSpec,
    FmhaCommonSpec,
    FmhaFwdFp8Spec,
    FmhaFwdHeadGroupingSpec,
    FmhaFwdPagedPrefillSpec,
    FmhaFwdSplitKvDecodeSpec,
    FmhaFwdVarlenSpec,
    FmhaShape,
    FusedMoeSpec,
    JengaSparseSpec,
    MxGemmSpec,
    SageAttentionSpec,
    StreamKGemmSpec,
    VsaSparseSpec,
    build_block_scale_gemm,
    build_fmha_bwd,
    build_fmha_fwd_appendkv,
    build_fmha_fwd_fp8,
    build_fmha_fwd_head_grouping,
    build_fmha_fwd_paged_prefill,
    build_fmha_fwd_splitkv_decode_reduce,
    build_fmha_fwd_splitkv_decode_segment,
    build_fmha_fwd_varlen,
    build_jenga_sparse_attention,
    build_moe_gather,
    build_moe_silu_mul,
    build_moe_topk_weighted_reduce,
    build_mx_gemm,
    build_sage_attention,
    build_streamk_gemm,
    build_vsa_sparse_attention,
    block_scale_gemm_grid,
    block_scale_gemm_signature,
    fmha_appendkv_grid,
    fmha_appendkv_signature,
    fmha_bwd_grid,
    fmha_bwd_signature,
    fmha_fwd_fp8_grid,
    fmha_fwd_fp8_signature,
    fmha_fwd_head_grouping_grid,
    fmha_fwd_head_grouping_signature,
    fmha_fwd_paged_prefill_grid,
    fmha_fwd_paged_prefill_signature,
    fmha_fwd_splitkv_decode_reduce_grid,
    fmha_fwd_splitkv_decode_reduce_signature,
    fmha_fwd_splitkv_decode_segment_grid,
    fmha_fwd_splitkv_decode_segment_signature,
    fmha_fwd_varlen_grid,
    fmha_fwd_varlen_signature,
    jenga_sparse_attention_grid,
    jenga_sparse_attention_signature,
    moe_gather_grid,
    moe_gather_signature,
    moe_silu_mul_grid,
    moe_silu_mul_signature,
    moe_topk_weighted_reduce_grid,
    moe_topk_weighted_reduce_signature,
    mx_gemm_grid,
    mx_gemm_signature,
    sage_attention_grid,
    sage_attention_signature,
    streamk_gemm_grid,
    streamk_gemm_signature,
    vsa_sparse_attention_grid,
    vsa_sparse_attention_signature,
)
from ck_dsl.runtime.launcher import KernelLauncher, LaunchConfig  # noqa: E402


@dataclass
class Result:
    name: str
    passed: bool
    max_abs_diff: float
    rel_max: float
    range_min: float
    range_max: float
    note: str = ""


def _summarise(O, O_ref, *, tol: float, note: str = "") -> Result:
    diff = (O.float() - O_ref.float()).abs()
    max_d = float(diff.max().item())
    ref_max = float(O_ref.abs().max().item())
    rel = max_d / (ref_max + 1e-9)
    O_min = float(O.min().item())
    O_max = float(O.max().item())
    # Sanity: O must be non-trivial when ref is non-trivial.
    if max(abs(O_min), abs(O_max)) < 0.001 and ref_max > 0.01:
        return Result(
            name="",
            passed=False,
            max_abs_diff=max_d,
            rel_max=rel,
            range_min=O_min,
            range_max=O_max,
            note=f"output is trivially zero (ref range ~{ref_max:.3f})",
        )
    return Result(
        name="",
        passed=(max_d <= tol),
        max_abs_diff=max_d,
        rel_max=rel,
        range_min=O_min,
        range_max=O_max,
        note=note,
    )


def _launch(launcher, args, *, grid, block=(64, 1, 1)):
    """Launch with wave64 block-size by default (the FMHA / sage / sparse
    kernels distribute the head-dim axis across the wave64). Per-kernel
    overrides pass ``block=(spec.block_size, 1, 1)`` explicitly when the
    kernel uses thread-id distribution at a different granularity (e.g.
    appendkv / moe_gather scatter kernels)."""
    launcher(args, config=LaunchConfig(grid=grid, block=block))
    torch.cuda.synchronize()


def _ref_attention(Q, K, V, *, causal: bool = False) -> torch.Tensor:
    """Dense attention reference. Q/K/V shape: ``(seqlen, H, D)``."""
    d = Q.shape[-1]
    scores = torch.einsum("ihd,jhd->ihj", Q.float(), K.float()) / math.sqrt(d)
    if causal:
        q_pos = torch.arange(Q.shape[0], device=Q.device).view(-1, 1, 1)
        k_pos = torch.arange(K.shape[0], device=K.device).view(1, 1, -1)
        scores = torch.where(k_pos <= q_pos, scores, torch.full_like(scores, -1e30))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("ihj,jhd->ihd", probs, V.float()).to(Q.dtype)


# ---------------------------------------------------------------------
# Fused MoE
# ---------------------------------------------------------------------


def case_moe_gather() -> Result:
    spec = FusedMoeSpec(
        tokens=32,
        experts=4,
        topk=2,
        hidden=256,
        intermediate=512,
        dtype="f16",
        block_size=64,
        vec=4,
    )
    kernel = build_moe_gather(spec)
    art = compile_kernel(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=moe_gather_signature(spec),
    )
    torch.manual_seed(0)
    X = torch.randn(spec.tokens, spec.hidden, dtype=torch.float16, device="cuda")
    sids = [(b % spec.tokens) for b in range(spec.total_pairs)]
    sids[3] = -1  # mask out one bucket
    SortedTokenIds = torch.tensor(sids, dtype=torch.int32, device="cuda")
    GroupedInput = torch.zeros(
        spec.total_pairs,
        spec.hidden,
        dtype=torch.float16,
        device="cuda",
    )
    _launch(
        launcher,
        {
            "X": X,
            "SortedTokenIds": SortedTokenIds,
            "GroupedInput": GroupedInput,
            "tokens": spec.tokens,
            "hidden": spec.hidden,
        },
        grid=moe_gather_grid(spec),
        block=(spec.block_size, 1, 1),
    )
    Ref = torch.zeros_like(GroupedInput)
    for b in range(spec.total_pairs):
        tid = sids[b]
        if tid >= 0:
            Ref[b, :] = X[tid, :]
    r = _summarise(GroupedInput, Ref, tol=0.0)
    r.name = "moe_gather"
    return r


def case_moe_silu_mul() -> Result:
    spec = FusedMoeSpec(
        tokens=32,
        experts=4,
        topk=2,
        hidden=256,
        intermediate=512,
        dtype="f16",
        block_size=64,
        vec=4,
    )
    kernel = build_moe_silu_mul(spec)
    art = compile_kernel(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=moe_silu_mul_signature(spec),
    )
    torch.manual_seed(1)
    GateOut = torch.randn(
        spec.total_pairs,
        spec.intermediate,
        dtype=torch.float16,
        device="cuda",
    )
    UpOut = torch.randn(
        spec.total_pairs,
        spec.intermediate,
        dtype=torch.float16,
        device="cuda",
    )
    Hidden = torch.zeros_like(GateOut)
    _launch(
        launcher,
        {
            "GateOut": GateOut,
            "UpOut": UpOut,
            "Hidden": Hidden,
            "total_pairs": spec.total_pairs,
            "intermediate": spec.intermediate,
        },
        grid=moe_silu_mul_grid(spec),
        block=(spec.block_size, 1, 1),
    )
    g32 = GateOut.float()
    silu = g32 * torch.sigmoid(g32)
    Ref = (silu * UpOut.float()).to(torch.float16)
    r = _summarise(Hidden, Ref, tol=5e-3, note="f16 silu via exp2 ULP")
    r.name = "moe_silu_mul"
    return r


def case_moe_topk_weighted_reduce() -> Result:
    spec = FusedMoeSpec(
        tokens=32,
        experts=4,
        topk=2,
        hidden=256,
        intermediate=512,
        dtype="f16",
        block_size=64,
        vec=4,
    )
    kernel = build_moe_topk_weighted_reduce(spec)
    art = compile_kernel(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=moe_topk_weighted_reduce_signature(spec),
    )
    torch.manual_seed(2)
    DownOut = torch.randn(
        spec.total_pairs,
        spec.hidden,
        dtype=torch.float16,
        device="cuda",
    )
    sids = [b // spec.topk for b in range(spec.total_pairs)]
    SortedTokenIds = torch.tensor(sids, dtype=torch.int32, device="cuda")
    SortedWeights = torch.rand(
        spec.total_pairs,
        dtype=torch.float32,
        device="cuda",
    )
    Y = torch.zeros(
        spec.tokens,
        spec.hidden,
        dtype=torch.float32,
        device="cuda",
    )
    _launch(
        launcher,
        {
            "DownOut": DownOut,
            "SortedTokenIds": SortedTokenIds,
            "SortedWeights": SortedWeights,
            "Y": Y,
            "total_pairs": spec.total_pairs,
            "hidden": spec.hidden,
            "tokens": spec.tokens,
        },
        grid=moe_topk_weighted_reduce_grid(spec),
        block=(spec.block_size, 1, 1),
    )
    Ref = torch.zeros_like(Y)
    for b in range(spec.total_pairs):
        tid = sids[b]
        if tid >= 0:
            Ref[tid, :] += SortedWeights[b].item() * DownOut[b, :].float()
    r = _summarise(Y, Ref, tol=5e-2)
    r.name = "moe_topk_weighted_reduce"
    return r


# ---------------------------------------------------------------------
# FMHA expansion
# ---------------------------------------------------------------------


def case_fmha_appendkv_norope() -> Result:
    head_size = 64
    HK = 4
    batch = 1  # v1 cache is flat (no per-batch base); use batch=1
    new_lens = [3]
    seqlen_kv_init = [7]
    cache_capacity = 16
    total_new = sum(new_lens)

    spec = FmhaAppendKvSpec(
        common=FmhaCommonSpec(
            shape=FmhaShape(head_size, HK, HK),
            dtype="f16",
        ),
        batch=batch,
        rotary=None,
        block_size=64,
    )
    kernel = build_fmha_fwd_appendkv(spec)
    art = compile_kernel(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=fmha_appendkv_signature(spec),
    )
    torch.manual_seed(10)
    K_new = torch.randn(
        total_new,
        HK,
        head_size,
        dtype=torch.float16,
        device="cuda",
    )
    V_new = torch.randn(
        total_new,
        HK,
        head_size,
        dtype=torch.float16,
        device="cuda",
    )
    K_cache = torch.zeros(
        cache_capacity,
        HK,
        head_size,
        dtype=torch.float16,
        device="cuda",
    )
    V_cache = torch.zeros_like(K_cache)
    seqlen_kv = torch.tensor(
        seqlen_kv_init,
        dtype=torch.int32,
        device="cuda",
    )
    cu = torch.tensor(
        [0] + list(accumulate(new_lens)),
        dtype=torch.int32,
        device="cuda",
    )
    _launch(
        launcher,
        {
            "K_new": K_new,
            "V_new": V_new,
            "K_cache": K_cache,
            "V_cache": V_cache,
            "seqlen_kv": seqlen_kv,
            "cu_seqlens_new": cu,
            "total_new_q": total_new,
            "batch": batch,
            "stride_in_token": HK * head_size,
            "stride_in_head": head_size,
            "stride_cache_token": HK * head_size,
            "stride_cache_head": head_size,
        },
        grid=fmha_appendkv_grid(spec, total_new_q=total_new),
        block=(spec.block_size, 1, 1),
    )
    K_ref = torch.zeros_like(K_cache)
    V_ref = torch.zeros_like(V_cache)
    for local in range(new_lens[0]):
        dst = seqlen_kv_init[0] + local
        K_ref[dst] = K_new[local]
        V_ref[dst] = V_new[local]
    r = _summarise(
        torch.cat([K_cache.flatten(), V_cache.flatten()]),
        torch.cat([K_ref.flatten(), V_ref.flatten()]),
        tol=0.0,
    )
    r.name = "fmha_appendkv (no rotary)"
    return r


def case_fmha_appendkv_rotary() -> Result:
    head_size = 64
    HK = 2
    batch = 1
    new_lens = [3]
    seqlen_kv_init = [4]
    cache_capacity = 32
    total_new = sum(new_lens)

    rotary = RotarySpec(head_size=head_size, layout="half")
    spec = FmhaAppendKvSpec(
        common=FmhaCommonSpec(
            shape=FmhaShape(head_size, HK, HK),
            dtype="f16",
        ),
        batch=batch,
        rotary=rotary,
        block_size=64,
    )
    kernel = build_fmha_fwd_appendkv(spec)
    art = compile_kernel(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=fmha_appendkv_signature(spec),
    )
    torch.manual_seed(11)
    K_new = torch.randn(
        total_new,
        HK,
        head_size,
        dtype=torch.float16,
        device="cuda",
    )
    V_new = torch.randn(
        total_new,
        HK,
        head_size,
        dtype=torch.float16,
        device="cuda",
    )
    K_cache = torch.zeros(
        cache_capacity,
        HK,
        head_size,
        dtype=torch.float16,
        device="cuda",
    )
    V_cache = torch.zeros_like(K_cache)
    seqlen_kv = torch.tensor(
        seqlen_kv_init,
        dtype=torch.int32,
        device="cuda",
    )
    cu = torch.tensor(
        [0] + list(accumulate(new_lens)),
        dtype=torch.int32,
        device="cuda",
    )
    pair_count = head_size // 2
    pos = torch.arange(cache_capacity, dtype=torch.float32, device="cuda")
    inv_freq = 1.0 / (
        10000.0
        ** (
            torch.arange(0, pair_count, dtype=torch.float32, device="cuda")
            * (2.0 / head_size)
        )
    )
    freqs = torch.outer(pos, inv_freq)
    cos_tbl = torch.cos(freqs).contiguous()
    sin_tbl = torch.sin(freqs).contiguous()
    _launch(
        launcher,
        {
            "K_new": K_new,
            "V_new": V_new,
            "K_cache": K_cache,
            "V_cache": V_cache,
            "seqlen_kv": seqlen_kv,
            "cu_seqlens_new": cu,
            "cos_table": cos_tbl,
            "sin_table": sin_tbl,
            "total_new_q": total_new,
            "batch": batch,
            "stride_in_token": HK * head_size,
            "stride_in_head": head_size,
            "stride_cache_token": HK * head_size,
            "stride_cache_head": head_size,
        },
        grid=fmha_appendkv_grid(spec, total_new_q=total_new),
        block=(spec.block_size, 1, 1),
    )
    K_ref = torch.zeros_like(K_cache)
    V_ref = torch.zeros_like(V_cache)
    for local in range(new_lens[0]):
        dst = seqlen_kv_init[0] + local
        c = cos_tbl[dst]
        s = sin_tbl[dst]
        k = K_new[local].float()
        lo, hi = k[:, :pair_count], k[:, pair_count:]
        new_lo = lo * c - hi * s
        new_hi = lo * s + hi * c
        K_ref[dst] = torch.cat([new_lo, new_hi], dim=-1).to(torch.float16)
        V_ref[dst] = V_new[local]
    r = _summarise(K_cache, K_ref, tol=5e-3, note="K rotated; V copied")
    r.name = "fmha_appendkv (rotary)"
    return r


def case_fmha_varlen_causal() -> Result:
    head_size = 64
    HQ = HK = 2
    # MFMA varlen requires per-sequence seqlen_q to be a multiple of
    # BLOCK_M (16). Each sequence's Q tile stays within one sequence.
    seq_lens = [16, 32]
    total_q = sum(seq_lens)
    spec = FmhaFwdVarlenSpec(
        common=FmhaCommonSpec(
            shape=FmhaShape(head_size, HQ, HK),
            dtype="f16",
            mask_mode="causal",
        ),
        max_seqlen_q=max(seq_lens),
        max_seqlen_k=max(seq_lens),
        batch=len(seq_lens),
    )
    kernel = build_fmha_fwd_varlen(spec)
    art = compile_kernel(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=fmha_fwd_varlen_signature(spec),
    )
    torch.manual_seed(20)
    Q = torch.randn(total_q, HQ, head_size, dtype=torch.float16, device="cuda")
    K = torch.randn(total_q, HK, head_size, dtype=torch.float16, device="cuda")
    V = torch.randn(total_q, HK, head_size, dtype=torch.float16, device="cuda")
    O = torch.zeros_like(Q)
    cu = torch.tensor(
        [0] + list(accumulate(seq_lens)),
        dtype=torch.int32,
        device="cuda",
    )
    _launch(
        launcher,
        {
            "Q": Q,
            "K": K,
            "V": V,
            "O": O,
            "cu_seqlens_q": cu,
            "cu_seqlens_k": cu,
            "scale_log2": math.log2(math.e) / math.sqrt(head_size),
            "total_q": total_q,
            "batch": len(seq_lens),
            "stride_q_token": HQ * head_size,
            "stride_q_head": head_size,
            "stride_k_token": HK * head_size,
            "stride_k_head": head_size,
            "stride_v_token": HK * head_size,
            "stride_v_head": head_size,
            "stride_o_token": HQ * head_size,
            "stride_o_head": head_size,
        },
        grid=fmha_fwd_varlen_grid(spec, total_q=total_q),
    )
    O_ref = torch.zeros_like(O)
    for seq in range(len(seq_lens)):
        lo, hi = sum(seq_lens[:seq]), sum(seq_lens[: seq + 1])
        O_ref[lo:hi] = _ref_attention(
            Q[lo:hi],
            K[lo:hi],
            V[lo:hi],
            causal=True,
        )
    r = _summarise(O, O_ref, tol=5e-3)
    r.name = "fmha_fwd_varlen (causal)"
    return r


def case_fmha_head_grouping_gqa() -> Result:
    head_size = 64
    HQ = 4
    HK = 2
    # BLOCK_M = 16 alignment for the MFMA-tiled body.
    seq_q = seq_k = 16
    batch = 2
    spec = FmhaFwdHeadGroupingSpec(
        common=FmhaCommonSpec(
            shape=FmhaShape(head_size, HQ, HK),
            dtype="f16",
            mask_mode="causal",
        ),
        seqlen_q=seq_q,
        seqlen_k=seq_k,
    )
    kernel = build_fmha_fwd_head_grouping(spec)
    art = compile_kernel(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=fmha_fwd_head_grouping_signature(spec),
    )
    torch.manual_seed(30)
    Q = torch.randn(
        batch * seq_q,
        HQ,
        head_size,
        dtype=torch.float16,
        device="cuda",
    )
    K = torch.randn(
        batch * seq_k,
        HK,
        head_size,
        dtype=torch.float16,
        device="cuda",
    )
    V = torch.randn(
        batch * seq_k,
        HK,
        head_size,
        dtype=torch.float16,
        device="cuda",
    )
    O = torch.zeros_like(Q)
    _launch(
        launcher,
        {
            "Q": Q,
            "K": K,
            "V": V,
            "O": O,
            "scale_log2": math.log2(math.e) / math.sqrt(head_size),
            "seqlen_q": seq_q,
            "seqlen_k": seq_k,
            "stride_q_token": HQ * head_size,
            "stride_q_head": head_size,
            "stride_k_token": HK * head_size,
            "stride_k_head": head_size,
            "stride_v_token": HK * head_size,
            "stride_v_head": head_size,
            "stride_o_token": HQ * head_size,
            "stride_o_head": head_size,
        },
        grid=fmha_fwd_head_grouping_grid(spec, batch=batch),
    )
    O_ref = torch.zeros_like(O)
    for b in range(batch):
        ql, qh = b * seq_q, (b + 1) * seq_q
        kl, kh = b * seq_k, (b + 1) * seq_k
        for h in range(HQ):
            kh_idx = h // (HQ // HK)
            scores = (
                Q[ql:qh, h, :].float() @ K[kl:kh, kh_idx, :].float().t()
            ) / math.sqrt(head_size)
            q_pos = torch.arange(seq_q, device=Q.device).view(-1, 1)
            k_pos = torch.arange(seq_k, device=K.device).view(1, -1)
            scores = torch.where(
                k_pos <= q_pos,
                scores,
                torch.full_like(scores, -1e30),
            )
            probs = torch.softmax(scores, dim=-1)
            O_ref[ql:qh, h, :] = (probs @ V[kl:kh, kh_idx, :].float()).to(torch.float16)
    r = _summarise(O, O_ref, tol=5e-3, note=f"GQA HQ={HQ} HK={HK}")
    r.name = "fmha_fwd_head_grouping (GQA causal)"
    return r


def case_fmha_fwd_fp8() -> Result:
    head_size = 64
    HQ = HK = 2
    # MFMA path: seqlen_q must be a multiple of BLOCK_M (16).
    seq_q = 16
    seq_k = 16
    spec = FmhaFwdFp8Spec(
        common=FmhaCommonSpec(
            shape=FmhaShape(head_size, HQ, HK),
            dtype="f16",
            mask_mode="none",
        ),
        kv_dtype="fp8e4m3",
        seqlen_q=seq_q,
    )
    kernel = build_fmha_fwd_fp8(spec)
    art = compile_kernel(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=fmha_fwd_fp8_signature(spec),
    )
    torch.manual_seed(40)
    Q = torch.randn(seq_q, HQ, head_size, dtype=torch.float16, device="cuda")
    # MI355X uses OCP FP8 (e4m3fn, bias=7) -- ``e4m3fnuz`` is the
    # older AMD pre-OCP format and would give 2x the values after cvt.
    K_f32 = (
        torch.randn(
            seq_k,
            HK,
            head_size,
            dtype=torch.float32,
            device="cuda",
        )
        * 0.5
    )
    V_f32 = (
        torch.randn(
            seq_k,
            HK,
            head_size,
            dtype=torch.float32,
            device="cuda",
        )
        * 0.5
    )
    K_fp8 = K_f32.to(torch.float8_e4m3fn).contiguous()
    V_fp8 = V_f32.to(torch.float8_e4m3fn).contiguous()
    O = torch.zeros_like(Q)
    k_scale_val, v_scale_val = 1.5, 0.75
    _launch(
        launcher,
        {
            "Q": Q,
            "K": K_fp8,
            "V": V_fp8,
            "O": O,
            "k_scale": k_scale_val,
            "v_scale": v_scale_val,
            "scale_log2": math.log2(math.e) / math.sqrt(head_size),
            "seqlen_q": seq_q,
            "seqlen_k": seq_k,
            "stride_q_token": HQ * head_size,
            "stride_q_head": head_size,
            "stride_k_token": HK * head_size,
            "stride_k_head": head_size,
            "stride_v_token": HK * head_size,
            "stride_v_head": head_size,
            "stride_o_token": HQ * head_size,
            "stride_o_head": head_size,
        },
        grid=fmha_fwd_fp8_grid(spec),
    )
    K_dq = K_fp8.float() * k_scale_val
    V_dq = V_fp8.float() * v_scale_val
    O_ref = torch.zeros_like(O)
    for h in range(HQ):
        kh = h // (HQ // HK)
        scores = (Q[:, h, :].float() @ K_dq[:, kh, :].t()) / math.sqrt(head_size)
        probs = torch.softmax(scores, dim=-1)
        O_ref[:, h, :] = (probs @ V_dq[:, kh, :]).to(torch.float16)
    r = _summarise(O, O_ref, tol=5e-3, note="OCP fp8e4m3fn (MI355X)")
    r.name = "fmha_fwd_fp8 (per-tensor scales)"
    return r


def case_fmha_fwd_splitkv_decode() -> Result:
    head_size = 64
    HQ = HK = 2
    batch = 2
    num_segments = 4
    seqlen_k_per_batch = [16, 12]
    max_k = max(seqlen_k_per_batch)
    spec = FmhaFwdSplitKvDecodeSpec(
        common=FmhaCommonSpec(
            shape=FmhaShape(head_size, HQ, HK),
            dtype="f16",
            mask_mode="none",
        ),
        batch=batch,
        num_segments=num_segments,
    )
    seg_k = build_fmha_fwd_splitkv_decode_segment(spec)
    red_k = build_fmha_fwd_splitkv_decode_reduce(spec)
    seg_art = compile_kernel(seg_k)
    red_art = compile_kernel(red_k)
    seg_launcher = KernelLauncher(
        hsaco=seg_art.hsaco,
        kernel_name=seg_k.name,
        signature=fmha_fwd_splitkv_decode_segment_signature(spec),
    )
    red_launcher = KernelLauncher(
        hsaco=red_art.hsaco,
        kernel_name=red_k.name,
        signature=fmha_fwd_splitkv_decode_reduce_signature(spec),
    )
    torch.manual_seed(50)
    Q = torch.randn(batch, HQ, head_size, dtype=torch.float16, device="cuda")
    K = torch.randn(
        batch * max_k,
        HK,
        head_size,
        dtype=torch.float16,
        device="cuda",
    )
    V = torch.randn(
        batch * max_k,
        HK,
        head_size,
        dtype=torch.float16,
        device="cuda",
    )
    O = torch.zeros_like(Q)
    seqlens_k = torch.tensor(
        seqlen_k_per_batch,
        dtype=torch.int32,
        device="cuda",
    )
    ws_size = num_segments * batch * HQ
    ws_m = torch.zeros(ws_size, dtype=torch.float32, device="cuda")
    ws_l = torch.zeros(ws_size, dtype=torch.float32, device="cuda")
    ws_acc = torch.zeros(
        ws_size * head_size,
        dtype=torch.float32,
        device="cuda",
    )
    scale_log2 = math.log2(math.e) / math.sqrt(head_size)
    _launch(
        seg_launcher,
        {
            "Q": Q,
            "K": K,
            "V": V,
            "seqlens_k": seqlens_k,
            "ws_m": ws_m,
            "ws_l": ws_l,
            "ws_acc": ws_acc,
            "scale_log2": scale_log2,
            "batch": batch,
            "stride_q_seq": HQ * head_size,
            "stride_q_head": head_size,
            "stride_k_token": HK * head_size,
            "stride_k_head": head_size,
            "stride_v_token": HK * head_size,
            "stride_v_head": head_size,
        },
        grid=fmha_fwd_splitkv_decode_segment_grid(spec),
    )
    _launch(
        red_launcher,
        {
            "ws_m": ws_m,
            "ws_l": ws_l,
            "ws_acc": ws_acc,
            "O": O,
            "batch": batch,
            "stride_o_seq": HQ * head_size,
            "stride_o_head": head_size,
        },
        grid=fmha_fwd_splitkv_decode_reduce_grid(spec),
    )
    O_ref = torch.zeros_like(O)
    for seq in range(batch):
        L = seqlen_k_per_batch[seq]
        for h in range(HQ):
            kh = h // (HQ // HK)
            scores = (Q[seq, h, :].float() @ K[:L, kh, :].float().t()) / math.sqrt(
                head_size
            )
            probs = torch.softmax(scores, dim=-1)
            O_ref[seq, h, :] = (probs @ V[:L, kh, :].float()).to(torch.float16)
    r = _summarise(O, O_ref, tol=5e-3, note=f"segments={num_segments}")
    r.name = "fmha_fwd_splitkv_decode (segment+reduce)"
    return r


def case_fmha_fwd_paged_prefill() -> Result:
    """Multi-block paged prefill with non-contiguous block_table.

    Exercises real CK Tile paged-KV indirection: the sequence's
    ``seqlen_k = 48`` tokens live in 3 paged blocks of size 16, which
    map (via ``block_table``) to physical blocks ``[3, 1, 5]`` in a
    cache that has 6 total blocks. A correct kernel must read K/V
    from those three non-contiguous physical blocks.
    """
    head_size = 64
    HQ = HK = 2
    page_block_size = 16
    seqlen_q = 4
    seqlen_k = 48  # three paged blocks
    batch = 1
    spec = FmhaFwdPagedPrefillSpec(
        common=FmhaCommonSpec(
            shape=FmhaShape(head_size, HQ, HK),
            dtype="f16",
            mask_mode="none",
        ),
        page_block_size=page_block_size,
        max_blocks_per_seq=4,
        batch=batch,
    )
    kernel = build_fmha_fwd_paged_prefill(spec)
    art = compile_kernel(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=fmha_fwd_paged_prefill_signature(spec),
    )
    torch.manual_seed(60)
    Q = torch.randn(seqlen_q, HQ, head_size, dtype=torch.float16, device="cuda")
    num_blocks = 6
    K_cache = torch.randn(
        num_blocks,
        page_block_size,
        HK,
        head_size,
        dtype=torch.float16,
        device="cuda",
    )
    V_cache = torch.randn(
        num_blocks,
        page_block_size,
        HK,
        head_size,
        dtype=torch.float16,
        device="cuda",
    )
    O = torch.zeros_like(Q)
    # Non-contiguous physical block assignment -- this is the test that
    # caught the v1 "first-block-only" bug.
    block_table = torch.tensor(
        [[3, 1, 5, 0]],
        dtype=torch.int32,
        device="cuda",
    )
    cu_seqlens_q = torch.tensor([0, seqlen_q], dtype=torch.int32, device="cuda")
    seqlens_k = torch.tensor([seqlen_k], dtype=torch.int32, device="cuda")

    stride_block = page_block_size * HK * head_size
    stride_page = HK * head_size
    stride_kv_head = head_size

    _launch(
        launcher,
        {
            "Q": Q,
            "K_cache": K_cache,
            "V_cache": V_cache,
            "O": O,
            "block_table": block_table,
            "cu_seqlens_q": cu_seqlens_q,
            "seqlens_k": seqlens_k,
            "scale_log2": math.log2(math.e) / math.sqrt(head_size),
            "total_q": seqlen_q,
            "batch": batch,
            "stride_q_token": HQ * head_size,
            "stride_q_head": head_size,
            "stride_block": stride_block,
            "stride_page": stride_page,
            "stride_kv_head": stride_kv_head,
            "stride_v_block": stride_block,
            "stride_v_page": stride_page,
            "stride_v_kv_head": stride_kv_head,
            "stride_o_token": HQ * head_size,
            "stride_o_head": head_size,
            "block_table_stride": 4,
        },
        grid=fmha_fwd_paged_prefill_grid(spec, total_q=seqlen_q),
    )
    # Reference: gather K/V from the physical blocks the block_table points to.
    physical_blocks = [3, 1, 5]
    K_seq = torch.cat(
        [K_cache[p] for p in physical_blocks], dim=0
    )  # (3 * page_block_size, HK, head_size)
    V_seq = torch.cat([V_cache[p] for p in physical_blocks], dim=0)
    O_ref = _ref_attention(Q, K_seq, V_seq, causal=False)
    r = _summarise(
        O,
        O_ref,
        tol=5e-3,
        note="3 non-contiguous paged blocks via block_table=[3,1,5]",
    )
    r.name = "fmha_fwd_paged_prefill"
    return r


# ---------------------------------------------------------------------
# Sage attention (49_sageattention)
# ---------------------------------------------------------------------


def case_sage_attention_fp16() -> Result:
    head_size = 64
    HQ = HK = 2
    seq_q = 8
    seq_k = 32
    q_blocks = max(1, seq_q // 8)
    k_blocks = max(1, seq_k // 32)
    q_scale_spec = QkScaleSpec(
        layout="per_block",
        scale_block=8,
        stride_batch=q_blocks * HQ,
        stride_head=1,
        stride_block=HQ,
    )
    k_scale_spec = QkScaleSpec(
        layout="per_block",
        scale_block=32,
        stride_batch=k_blocks * HK,
        stride_head=1,
        stride_block=HK,
    )
    spec = SageAttentionSpec(
        common=FmhaCommonSpec(
            shape=FmhaShape(head_size, HQ, HK),
            dtype="f16",
            mask_mode="none",
        ),
        quant_mode="fp16_bf16",
        q_scale=q_scale_spec,
        k_scale=k_scale_spec,
        seqlen_q=seq_q,
        seqlen_k=seq_k,
    )
    kernel = build_sage_attention(spec)
    art = compile_kernel(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=sage_attention_signature(spec),
    )
    torch.manual_seed(70)
    Q = torch.randn(seq_q, HQ, head_size, dtype=torch.float16, device="cuda")
    K = torch.randn(seq_k, HK, head_size, dtype=torch.float16, device="cuda")
    V = torch.randn(seq_k, HK, head_size, dtype=torch.float16, device="cuda")
    O = torch.zeros_like(Q)
    qs = torch.ones(q_blocks * HQ, dtype=torch.float32, device="cuda")
    ks = torch.ones(k_blocks * HK, dtype=torch.float32, device="cuda")
    _launch(
        launcher,
        {
            "Q": Q,
            "K": K,
            "V": V,
            "O": O,
            "q_scale": qs,
            "k_scale": ks,
            "scale_log2": math.log2(math.e) / math.sqrt(head_size),
            "seqlen_q": seq_q,
            "seqlen_k": seq_k,
            "stride_q_token": HQ * head_size,
            "stride_q_head": head_size,
            "stride_k_token": HK * head_size,
            "stride_k_head": head_size,
            "stride_v_token": HK * head_size,
            "stride_v_head": head_size,
            "stride_o_token": HQ * head_size,
            "stride_o_head": head_size,
        },
        grid=sage_attention_grid(spec),
    )
    O_ref = _ref_attention(Q, K, V, causal=False)
    r = _summarise(O, O_ref, tol=5e-3, note="baseline scales=1")
    r.name = "sage_attention fp16_bf16 (scales=1)"
    return r


def case_sage_attention_fp8() -> Result:
    head_size = 64
    HQ = HK = 2
    seq_q = 8
    seq_k = 32
    q_blocks = seq_q // 8
    k_blocks = seq_k // 32
    q_scale_spec = QkScaleSpec(
        layout="per_block",
        scale_block=8,
        stride_batch=q_blocks * HQ,
        stride_head=1,
        stride_block=HQ,
    )
    k_scale_spec = QkScaleSpec(
        layout="per_block",
        scale_block=32,
        stride_batch=k_blocks * HK,
        stride_head=1,
        stride_block=HK,
    )
    spec = SageAttentionSpec(
        common=FmhaCommonSpec(
            shape=FmhaShape(head_size, HQ, HK),
            dtype="f16",
            mask_mode="none",
        ),
        quant_mode="fp8_bf16",
        q_scale=q_scale_spec,
        k_scale=k_scale_spec,
        seqlen_q=seq_q,
        seqlen_k=seq_k,
    )
    kernel = build_sage_attention(spec)
    art = compile_kernel(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=sage_attention_signature(spec),
    )
    torch.manual_seed(80)
    Q = torch.randn(seq_q, HQ, head_size, dtype=torch.float16, device="cuda")
    K_f32 = (
        torch.randn(
            seq_k,
            HK,
            head_size,
            dtype=torch.float32,
            device="cuda",
        )
        * 0.5
    )
    V_f32 = (
        torch.randn(
            seq_k,
            HK,
            head_size,
            dtype=torch.float32,
            device="cuda",
        )
        * 0.5
    )
    K_fp8 = K_f32.to(torch.float8_e4m3fn).contiguous()
    V_fp8 = V_f32.to(torch.float8_e4m3fn).contiguous()
    O = torch.zeros_like(Q)
    qs_val, ks_val = 1.5, 0.7
    qs = torch.full((q_blocks * HQ,), qs_val, dtype=torch.float32, device="cuda")
    ks = torch.full((k_blocks * HK,), ks_val, dtype=torch.float32, device="cuda")
    _launch(
        launcher,
        {
            "Q": Q,
            "K": K_fp8,
            "V": V_fp8,
            "O": O,
            "q_scale": qs,
            "k_scale": ks,
            "scale_log2": math.log2(math.e) / math.sqrt(head_size),
            "seqlen_q": seq_q,
            "seqlen_k": seq_k,
            "stride_q_token": HQ * head_size,
            "stride_q_head": head_size,
            "stride_k_token": HK * head_size,
            "stride_k_head": head_size,
            "stride_v_token": HK * head_size,
            "stride_v_head": head_size,
            "stride_o_token": HQ * head_size,
            "stride_o_head": head_size,
        },
        grid=sage_attention_grid(spec),
    )
    K_dq = K_fp8.float()
    V_dq = V_fp8.float()
    O_ref = torch.zeros_like(O)
    for h in range(HQ):
        kh = h // (HQ // HK)
        scores = (Q[:, h, :].float() @ K_dq[:, kh, :].t()) / math.sqrt(head_size)
        # Sage applies per-block QK scales to the SCORE, not to K/V.
        scores = scores * qs_val * ks_val
        probs = torch.softmax(scores, dim=-1)
        O_ref[:, h, :] = (probs @ V_dq[:, kh, :]).to(torch.float16)
    r = _summarise(O, O_ref, tol=5e-3, note=f"per-block scales q={qs_val} k={ks_val}")
    r.name = "sage_attention fp8_bf16 (per-block scales)"
    return r


# ---------------------------------------------------------------------
# Sparse attention (50_sparse_attn)
# ---------------------------------------------------------------------


def case_jenga_sparse() -> Result:
    head_size = 64
    HQ = HK = 2
    # MFMA path: seqlen_q must be BLOCK_M (16) aligned; block_k must be
    # a multiple of MFMA BLOCK_K (16).
    seq_q = 16
    seq_k = 64
    block_k = 32
    num_kb = seq_k // block_k
    spec = JengaSparseSpec(
        common=FmhaCommonSpec(
            shape=FmhaShape(head_size, HQ, HK),
            dtype="f16",
            mask_mode="none",
        ),
        seqlen_q=seq_q,
        seqlen_k=seq_k,
        block_q=1,
        block_k=block_k,
    )
    kernel = build_jenga_sparse_attention(spec)
    art = compile_kernel(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=jenga_sparse_attention_signature(spec),
    )
    torch.manual_seed(90)
    Q = torch.randn(seq_q, HQ, head_size, dtype=torch.float16, device="cuda")
    K = torch.randn(seq_k, HK, head_size, dtype=torch.float16, device="cuda")
    V = torch.randn(seq_k, HK, head_size, dtype=torch.float16, device="cuda")
    O = torch.zeros_like(Q)
    mask = torch.zeros(seq_q, num_kb, dtype=torch.int8, device="cuda")
    mask[:, 1] = 1
    _launch(
        launcher,
        {
            "Q": Q,
            "K": K,
            "V": V,
            "O": O,
            "mask": mask,
            "scale_log2": math.log2(math.e) / math.sqrt(head_size),
            "seqlen_q": seq_q,
            "seqlen_k": seq_k,
            "stride_q_token": HQ * head_size,
            "stride_q_head": head_size,
            "stride_k_token": HK * head_size,
            "stride_k_head": head_size,
            "stride_v_token": HK * head_size,
            "stride_v_head": head_size,
            "stride_o_token": HQ * head_size,
            "stride_o_head": head_size,
        },
        grid=jenga_sparse_attention_grid(spec),
    )
    O_ref = _ref_attention(
        Q,
        K[block_k : 2 * block_k],
        V[block_k : 2 * block_k],
        causal=False,
    )
    r = _summarise(O, O_ref, tol=5e-3, note="attend to second k-block only")
    r.name = "jenga_sparse_attention"
    return r


def case_vsa_sparse() -> Result:
    head_size = 64
    HQ = HK = 2
    seq_q = 16  # BLOCK_M-aligned for MFMA path
    seq_k = 128
    block_k = 32
    spec = VsaSparseSpec(
        common=FmhaCommonSpec(
            shape=FmhaShape(head_size, HQ, HK),
            dtype="f16",
            mask_mode="none",
        ),
        seqlen_q=seq_q,
        seqlen_k=seq_k,
        block_q=1,
        block_k=block_k,
        max_blocks_per_q=2,
    )
    kernel = build_vsa_sparse_attention(spec)
    art = compile_kernel(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=vsa_sparse_attention_signature(spec),
    )
    torch.manual_seed(100)
    Q = torch.randn(seq_q, HQ, head_size, dtype=torch.float16, device="cuda")
    K = torch.randn(seq_k, HK, head_size, dtype=torch.float16, device="cuda")
    V = torch.randn(seq_k, HK, head_size, dtype=torch.float16, device="cuda")
    O = torch.zeros_like(Q)
    block_lut = torch.tensor(
        [[0, 3]] * seq_q,
        dtype=torch.int32,
        device="cuda",
    )
    block_count = torch.tensor(
        [2] * seq_q,
        dtype=torch.int32,
        device="cuda",
    )
    _launch(
        launcher,
        {
            "Q": Q,
            "K": K,
            "V": V,
            "O": O,
            "block_lut": block_lut,
            "block_count": block_count,
            "scale_log2": math.log2(math.e) / math.sqrt(head_size),
            "seqlen_q": seq_q,
            "seqlen_k": seq_k,
            "stride_q_token": HQ * head_size,
            "stride_q_head": head_size,
            "stride_k_token": HK * head_size,
            "stride_k_head": head_size,
            "stride_v_token": HK * head_size,
            "stride_v_head": head_size,
            "stride_o_token": HQ * head_size,
            "stride_o_head": head_size,
        },
        grid=vsa_sparse_attention_grid(spec),
    )
    K_chunks = torch.cat([K[0:block_k], K[3 * block_k : 4 * block_k]], dim=0)
    V_chunks = torch.cat([V[0:block_k], V[3 * block_k : 4 * block_k]], dim=0)
    O_ref = _ref_attention(Q, K_chunks, V_chunks, causal=False)
    r = _summarise(O, O_ref, tol=5e-3, note="LUT={0, 3}")
    r.name = "vsa_sparse_attention"
    return r


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Sage int variants (i8 / i4 codebook dequant)
# ---------------------------------------------------------------------


def _sage_int_case(quant_mode: str) -> Result:
    head_size = 128 if quant_mode == "i4_fp8_bf16" else 64
    HQ = HK = 2
    seq_q = 8
    seq_k = 32
    q_blocks = max(1, seq_q // 8)
    k_blocks = max(1, seq_k // 32)
    q_scale_spec = QkScaleSpec(
        layout="per_block",
        scale_block=8,
        stride_batch=q_blocks * HQ,
        stride_head=1,
        stride_block=HQ,
    )
    k_scale_spec = QkScaleSpec(
        layout="per_block",
        scale_block=32,
        stride_batch=k_blocks * HK,
        stride_head=1,
        stride_block=HK,
    )
    spec = SageAttentionSpec(
        common=FmhaCommonSpec(
            shape=FmhaShape(head_size, HQ, HK),
            dtype="f16",
            mask_mode="none",
        ),
        quant_mode=quant_mode,
        q_scale=q_scale_spec,
        k_scale=k_scale_spec,
        seqlen_q=seq_q,
        seqlen_k=seq_k,
    )
    kernel = build_sage_attention(spec)
    art = compile_kernel(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=sage_attention_signature(spec),
    )
    torch.manual_seed(110 if quant_mode == "i8_fp8_bf16" else 120)
    Q = torch.randn(seq_q, HQ, head_size, dtype=torch.float16, device="cuda")
    O = torch.zeros_like(Q)
    qs_val, ks_val = 0.6, 1.3
    qs = torch.full((q_blocks * HQ,), qs_val, dtype=torch.float32, device="cuda")
    ks = torch.full((k_blocks * HK,), ks_val, dtype=torch.float32, device="cuda")

    if quant_mode == "i8_fp8_bf16":
        # K/V quantised to i8 in [-127, 127]; codebook holds the
        # f32-domain value for each (i8 + 128) index.
        codebook_vals = (
            torch.arange(-128, 128, dtype=torch.float32, device="cuda") / 64.0
        )
        K_i8 = torch.randint(
            -127, 128, (seq_k, HK, head_size), dtype=torch.int8, device="cuda"
        )
        V_i8 = torch.randint(
            -127, 128, (seq_k, HK, head_size), dtype=torch.int8, device="cuda"
        )
        cb_k = codebook_vals
        cb_v = codebook_vals
        K_buf, V_buf = K_i8, V_i8
        # The codebook output gets rounded to fp8e4m3, then read back
        # to f32 in the kernel. Apply that same round-trip in the
        # reference so we compare against what the kernel actually
        # accumulates.
        K_dq_dom = codebook_vals[K_i8.int() + 128]
        V_dq_dom = codebook_vals[V_i8.int() + 128]
        K_dq = K_dq_dom.to(torch.float8_e4m3fn).float()
        V_dq = V_dq_dom.to(torch.float8_e4m3fn).float()
    else:  # i4_fp8_bf16
        # 16-entry codebook (one per i4 value -8..7).
        codebook_vals = torch.arange(-8, 8, dtype=torch.float32, device="cuda") / 4.0
        # Each byte holds two nibbles: low nibble = lo elem, high = hi.
        n_bytes = seq_k * HK * head_size // 2
        K_packed = torch.randint(
            0,
            256,
            (n_bytes,),
            dtype=torch.uint8,
            device="cuda",
        ).to(torch.int8)
        V_packed = torch.randint(
            0,
            256,
            (n_bytes,),
            dtype=torch.uint8,
            device="cuda",
        ).to(torch.int8)
        cb_k = codebook_vals
        cb_v = codebook_vals
        K_buf = K_packed
        V_buf = V_packed

        # Reference: dequant each packed byte to two f32 values, then
        # round through fp8e4m3 (matching the kernel's post-codebook cast).
        def unpack_to_f32(packed_i8):
            byte_i32 = packed_i8.int()
            lo_nib = byte_i32 & 0xF
            # Sign-extend low nibble: if bit3 set, subtract 16.
            lo_signed = torch.where(lo_nib >= 8, lo_nib - 16, lo_nib)
            # Sign-extend high nibble: shift right with sign-fill.
            hi_signed = (byte_i32 << 24) >> 28
            return lo_signed.long(), hi_signed.long()

        K_lo_i4, K_hi_i4 = unpack_to_f32(K_packed)
        V_lo_i4, V_hi_i4 = unpack_to_f32(V_packed)
        # Each lane owns one packed byte = head_dim slots [2*tid, 2*tid+1].
        # Reassemble the dequantised K/V as (seq_k, HK, head_size).
        # The byte-flat layout is (seq_k, HK, head_size/2) -- consecutive
        # bytes are consecutive lanes, which means head_dim slots
        # [2*tid, 2*tid+1] in (head_dim) order.
        K_dq_dom = torch.zeros(seq_k, HK, head_size, dtype=torch.float32, device="cuda")
        V_dq_dom = torch.zeros_like(K_dq_dom)
        idx = torch.arange(n_bytes, device="cuda")
        kk = idx % (head_size // 2)
        h = (idx // (head_size // 2)) % HK
        s = idx // (HK * head_size // 2)
        K_dq_dom[s, h, 2 * kk] = codebook_vals[K_lo_i4 + 8]
        K_dq_dom[s, h, 2 * kk + 1] = codebook_vals[K_hi_i4 + 8]
        V_dq_dom[s, h, 2 * kk] = codebook_vals[V_lo_i4 + 8]
        V_dq_dom[s, h, 2 * kk + 1] = codebook_vals[V_hi_i4 + 8]
        K_dq = K_dq_dom.to(torch.float8_e4m3fn).float()
        V_dq = V_dq_dom.to(torch.float8_e4m3fn).float()

    # i4 / i8 store packed bytes -- the K/V stride is element-stride
    # in the *byte* space. For i4 each byte holds two elements so the
    # stride halves; i8 keeps stride == head_size as one byte = one
    # element.
    kv_bytes_per_elem = 0.5 if quant_mode == "i4_fp8_bf16" else 1
    sk_token = int(HK * head_size * kv_bytes_per_elem)
    sk_head = int(head_size * kv_bytes_per_elem)
    launch_args = {
        "Q": Q,
        "K": K_buf,
        "V": V_buf,
        "O": O,
        "q_scale": qs,
        "k_scale": ks,
        "codebook_k": cb_k,
        "codebook_v": cb_v,
        "scale_log2": math.log2(math.e) / math.sqrt(head_size),
        "seqlen_q": seq_q,
        "seqlen_k": seq_k,
        "stride_q_token": HQ * head_size,
        "stride_q_head": head_size,
        "stride_k_token": sk_token,
        "stride_k_head": sk_head,
        "stride_v_token": sk_token,
        "stride_v_head": sk_head,
        "stride_o_token": HQ * head_size,
        "stride_o_head": head_size,
    }
    _launch(launcher, launch_args, grid=sage_attention_grid(spec))

    O_ref = torch.zeros_like(O)
    for h in range(HQ):
        kh = h // (HQ // HK)
        scores = (Q[:, h, :].float() @ K_dq[:, kh, :].t()) / math.sqrt(head_size)
        scores = scores * qs_val * ks_val
        probs = torch.softmax(scores, dim=-1)
        O_ref[:, h, :] = (probs @ V_dq[:, kh, :]).to(torch.float16)

    r = _summarise(O, O_ref, tol=5e-2, note=f"quant_mode={quant_mode}")
    r.name = f"sage_attention {quant_mode}"
    return r


def case_sage_attention_i8() -> Result:
    return _sage_int_case("i8_fp8_bf16")


def case_sage_attention_i4() -> Result:
    return _sage_int_case("i4_fp8_bf16")


# ---------------------------------------------------------------------
# GEMM family: StreamK, block-scale, MX
# ---------------------------------------------------------------------


def case_streamk_gemm() -> Result:
    """f16 GEMM via StreamK + MFMA atomic split-K."""
    M = N = K = 64
    # tile_m=tile_n=16 matches the f16_16x16x16 atom; tile_k=32 = 2
    # MFMA atoms per macro tile (4 macro tiles span the K=64 axis).
    spec = StreamKGemmSpec(
        M=M,
        N=N,
        K=K,
        tile_m=16,
        tile_n=16,
        tile_k=32,
        dtype="f16",
        num_cus=8,
        blocks_per_cu=1,
    )
    kernel = build_streamk_gemm(spec)
    art = compile_kernel(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=streamk_gemm_signature(spec),
    )
    torch.manual_seed(200)
    # Integer-valued so f16 multiplies are exact (no rounding).
    A = torch.randint(-3, 4, (M, K), dtype=torch.int32, device="cuda").to(torch.float16)
    B = torch.randint(-3, 4, (K, N), dtype=torch.int32, device="cuda").to(torch.float16)
    Cf32 = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    Counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    _launch(
        launcher,
        {"A": A, "B": B, "Cf32": Cf32, "Counter": Counter},
        grid=streamk_gemm_grid(spec),
        block=(spec.block_size, 1, 1),
    )
    C_ref = A.float() @ B.float()
    r = _summarise(Cf32, C_ref, tol=0.0)
    r.name = "streamk_gemm (atomic strategy)"
    return r


def case_fmha_fwd_mfma() -> Result:
    """MFMA-tiled FMHA forward (16x16 Q tile, MFMA QK + softmax + PV)."""
    from ck_dsl.instances import (
        FmhaMfmaSpec,
        build_fmha_fwd_mfma,
        fmha_fwd_mfma_grid,
        fmha_fwd_mfma_signature,
    )

    head_size = 64
    HQ = HK = 2
    seqlen_q = 16
    seqlen_k = 16
    batch = 1
    spec = FmhaMfmaSpec(
        common=FmhaCommonSpec(
            shape=FmhaShape(head_size, HQ, HK),
            dtype="f16",
            mask_mode="none",
        ),
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
    )
    kernel = build_fmha_fwd_mfma(spec)
    art = compile_kernel(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=fmha_fwd_mfma_signature(spec),
    )
    torch.manual_seed(400)
    Q = torch.randn(
        batch * seqlen_q,
        HQ,
        head_size,
        dtype=torch.float16,
        device="cuda",
    )
    K = torch.randn(
        batch * seqlen_k,
        HK,
        head_size,
        dtype=torch.float16,
        device="cuda",
    )
    V = torch.randn(
        batch * seqlen_k,
        HK,
        head_size,
        dtype=torch.float16,
        device="cuda",
    )
    O = torch.zeros_like(Q)
    _launch(
        launcher,
        {
            "Q": Q,
            "K": K,
            "V": V,
            "O": O,
            "scale_log2": math.log2(math.e) / math.sqrt(head_size),
            "seqlen_q": seqlen_q,
            "seqlen_k": seqlen_k,
            "stride_q_token": HQ * head_size,
            "stride_q_head": head_size,
            "stride_k_token": HK * head_size,
            "stride_k_head": head_size,
            "stride_v_token": HK * head_size,
            "stride_v_head": head_size,
            "stride_o_token": HQ * head_size,
            "stride_o_head": head_size,
        },
        grid=fmha_fwd_mfma_grid(spec, batch=batch),
    )
    # Per-head dense attention reference.
    O_ref = torch.zeros_like(O)
    for h in range(HQ):
        kh = h // (HQ // HK)
        scores = (Q[:, h, :].float() @ K[:, kh, :].float().t()) / math.sqrt(head_size)
        probs = torch.softmax(scores, dim=-1)
        O_ref[:, h, :] = (probs @ V[:, kh, :].float()).to(torch.float16)
    r = _summarise(
        O,
        O_ref,
        tol=5e-3,
        note="mfma_f32_16x16x16_f16 QK + softmax + PV chain",
    )
    r.name = "fmha_fwd_mfma (16x16 atom)"
    return r


def case_mfma_gemm() -> Result:
    """f16 GEMM via mfma_f32_16x16x16_f16 atom (production density)."""
    from ck_dsl.instances import (
        MfmaGemmSpec,
        build_mfma_gemm,
        mfma_gemm_grid,
        mfma_gemm_signature,
    )

    M = N = K = 64
    spec = MfmaGemmSpec(M=M, N=N, K=K, dtype="f16")
    kernel = build_mfma_gemm(spec)
    art = compile_kernel(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=mfma_gemm_signature(spec),
    )
    torch.manual_seed(230)
    # Integer-valued so f16 multiplies are exact (no rounding).
    A = torch.randint(-3, 4, (M, K), dtype=torch.int32, device="cuda").to(torch.float16)
    B = torch.randint(-3, 4, (K, N), dtype=torch.int32, device="cuda").to(torch.float16)
    C = torch.zeros(M, N, dtype=torch.float16, device="cuda")
    _launch(
        launcher,
        {"A": A, "B": B, "C": C, "M": M, "N": N, "K": K},
        grid=mfma_gemm_grid(spec),
        block=(spec.block_size, 1, 1),
    )
    C_ref = (A.float() @ B.float()).to(torch.float16)
    r = _summarise(
        C,
        C_ref,
        tol=0.0,
        note="mfma_f32_16x16x16_f16 atom; one MFMA per K-tile",
    )
    r.name = "mfma_gemm (f16 16x16x16 atom)"
    return r


def case_block_scale_gemm() -> Result:
    """Block-scaled FP8 GEMM with a/b/abquant per-group scales."""
    M = N = K = 64
    spec = BlockScaleGemmSpec(
        M=M,
        N=N,
        K=K,
        block_tile_m=16,
        block_tile_n=16,
        quant_mode="abquant",
        mantissa_dtype="fp8e4m3",
        group_size_mnk=(1, 1, K),
    )
    kernel = build_block_scale_gemm(spec)
    art = compile_kernel(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=block_scale_gemm_signature(spec),
    )
    torch.manual_seed(210)
    A_f32 = torch.randn(M, K, dtype=torch.float32, device="cuda") * 0.3
    B_f32 = torch.randn(K, N, dtype=torch.float32, device="cuda") * 0.3
    A_fp8 = A_f32.to(torch.float8_e4m3fn)
    B_fp8 = B_f32.to(torch.float8_e4m3fn)
    a_scale = torch.full((M, 1), 1.5, dtype=torch.float32, device="cuda")
    b_scale = torch.full((1, N), 0.7, dtype=torch.float32, device="cuda")
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    _launch(
        launcher,
        {
            "A": A_fp8,
            "AScale": a_scale,
            "B": B_fp8,
            "BScale": b_scale,
            "C": C,
            "M": M,
            "N": N,
            "K": K,
        },
        grid=block_scale_gemm_grid(spec),
        block=(spec.block_size, 1, 1),
    )
    # Reference: f32 GEMM on the dequantised + scaled values.
    A_dq = A_fp8.float() * a_scale
    B_dq = B_fp8.float() * b_scale
    C_ref = A_dq @ B_dq
    r = _summarise(C, C_ref, tol=1e-2, note="fp8 abquant 1x1xK")
    r.name = "block_scale_gemm (fp8 abquant)"
    return r


def case_mx_gemm() -> Result:
    """OCP MX shared-exponent GEMM, fp8e4m3 mantissa, group_k=32."""
    M = N = 32
    K = 64  # 2 MX blocks
    spec = MxGemmSpec(M=M, N=N, K=K, mantissa_dtype="fp8e4m3")
    kernel = build_mx_gemm(spec)
    art = compile_kernel(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=mx_gemm_signature(spec),
    )
    torch.manual_seed(220)
    A_f32 = torch.randn(M, K, dtype=torch.float32, device="cuda") * 0.3
    B_f32 = torch.randn(K, N, dtype=torch.float32, device="cuda") * 0.3
    A_fp8 = A_f32.to(torch.float8_e4m3fn)
    B_fp8 = B_f32.to(torch.float8_e4m3fn)
    # MX shared-exponent E8M0 byte: bias=127. e=127 => scale 1.0.
    a_scale = torch.full((M, K // 32), 127, dtype=torch.uint8, device="cuda").to(
        torch.int8
    )
    b_scale = torch.full((K // 32, N), 127, dtype=torch.uint8, device="cuda").to(
        torch.int8
    )
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    _launch(
        launcher,
        {
            "A": A_fp8,
            "AScale": a_scale,
            "B": B_fp8,
            "BScale": b_scale,
            "C": C,
            "M": M,
            "N": N,
            "K": K,
        },
        grid=mx_gemm_grid(spec),
        block=(spec.block_size, 1, 1),
    )
    # With scale=1 the dequant is the fp8 round-trip.
    C_ref = A_fp8.float() @ B_fp8.float()
    r = _summarise(C, C_ref, tol=1e-2, note="MX e8m0=127 (scale=1)")
    r.name = "mx_gemm (fp8 e8m0=127)"
    return r


# ---------------------------------------------------------------------
# FMHA backward
# ---------------------------------------------------------------------


def case_fmha_bwd() -> Result:
    """dQ / dK / dV via atomic fp32 accumulate vs torch.autograd."""
    head_size = 64
    HQ = HK = 2
    seq_q = seq_k = 16
    spec = FmhaBwdSpec(
        common=FmhaCommonSpec(
            shape=FmhaShape(head_size, HQ, HK),
            dtype="f16",
            mask_mode="none",
        ),
        seqlen_q=seq_q,
        seqlen_k=seq_k,
    )
    kernel = build_fmha_bwd(spec)
    art = compile_kernel(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=fmha_bwd_signature(spec),
    )
    torch.manual_seed(300)
    Q = torch.randn(
        seq_q,
        HQ,
        head_size,
        dtype=torch.float16,
        device="cuda",
        requires_grad=False,
    )
    K = torch.randn(
        seq_k,
        HK,
        head_size,
        dtype=torch.float16,
        device="cuda",
    )
    V = torch.randn(
        seq_k,
        HK,
        head_size,
        dtype=torch.float16,
        device="cuda",
    )
    dO = torch.randn(
        seq_q,
        HQ,
        head_size,
        dtype=torch.float16,
        device="cuda",
    )

    # Compute saved (M, L) from a forward pass.
    M_saved = torch.zeros(seq_q, HQ, dtype=torch.float32, device="cuda")
    L_saved = torch.zeros(seq_q, HQ, dtype=torch.float32, device="cuda")
    for h in range(HQ):
        kh = h // (HQ // HK)
        scores = (Q[:, h, :].float() @ K[:, kh, :].float().t()) / math.sqrt(head_size)
        scores_log2 = scores * math.log2(math.e)
        M_saved[:, h] = scores_log2.max(dim=-1).values
        L_saved[:, h] = torch.exp2(scores_log2 - M_saved[:, h : h + 1]).sum(dim=-1)

    dQ = torch.zeros(seq_q, HQ, head_size, dtype=torch.float32, device="cuda")
    dK = torch.zeros(seq_k, HK, head_size, dtype=torch.float32, device="cuda")
    dV = torch.zeros(seq_k, HK, head_size, dtype=torch.float32, device="cuda")
    _launch(
        launcher,
        {
            "Q": Q,
            "K": K,
            "V": V,
            "dO": dO,
            "M_saved": M_saved.contiguous(),
            "L_saved": L_saved.contiguous(),
            "dQ": dQ,
            "dK": dK,
            "dV": dV,
            "scale_log2": math.log2(math.e) / math.sqrt(head_size),
            "scale_inv": 1.0 / math.sqrt(head_size),
            "seqlen_q": seq_q,
            "seqlen_k": seq_k,
            "stride_q_token": HQ * head_size,
            "stride_q_head": head_size,
            "stride_k_token": HK * head_size,
            "stride_k_head": head_size,
            "stride_v_token": HK * head_size,
            "stride_v_head": head_size,
            "stride_do_token": HQ * head_size,
            "stride_do_head": head_size,
            "stride_dq_token": HQ * head_size,
            "stride_dk_token": HK * head_size,
            "stride_dv_token": HK * head_size,
        },
        grid=fmha_bwd_grid(spec),
    )

    # Torch reference via autograd.
    Qg = Q.float().clone().requires_grad_(True)
    Kg = K.float().clone().requires_grad_(True)
    Vg = V.float().clone().requires_grad_(True)
    out = torch.zeros(seq_q, HQ, head_size, dtype=torch.float32, device="cuda")
    for h in range(HQ):
        kh = h // (HQ // HK)
        scores = (Qg[:, h, :] @ Kg[:, kh, :].t()) / math.sqrt(head_size)
        probs = torch.softmax(scores, dim=-1)
        out[:, h, :] = probs @ Vg[:, kh, :]
    loss = (out * dO.float()).sum()
    loss.backward()
    dQ_ref = Qg.grad
    dK_ref = Kg.grad
    dV_ref = Vg.grad

    # The kernel's bwd doesn't sum per-head_idx across the HQ heads when
    # HQ == HK; each Q head writes to its own dQ row. Compare each.
    diff_dQ = (dQ - dQ_ref).abs().max().item()
    diff_dK = (dK - dK_ref).abs().max().item()
    diff_dV = (dV - dV_ref).abs().max().item()
    max_d = max(diff_dQ, diff_dK, diff_dV)
    ref_max = max(
        dQ_ref.abs().max().item(),
        dK_ref.abs().max().item(),
        dV_ref.abs().max().item(),
    )
    rel = max_d / (ref_max + 1e-9)
    print(f"    dQ max={diff_dQ:.4g} dK max={diff_dK:.4g} dV max={diff_dV:.4g}")
    r = Result(
        name="fmha_bwd (atomic accumulate)",
        passed=(max_d < 5e-2),
        max_abs_diff=max_d,
        rel_max=rel,
        range_min=min(float(dQ.min()), float(dK.min()), float(dV.min())),
        range_max=max(float(dQ.max()), float(dK.max()), float(dV.max())),
        note=f"dQ={diff_dQ:.3g} dK={diff_dK:.3g} dV={diff_dV:.3g}",
    )
    return r


ALL_CASES: Dict[str, Callable[[], Result]] = {
    "moe_gather": case_moe_gather,
    "moe_silu_mul": case_moe_silu_mul,
    "moe_topk_weighted_reduce": case_moe_topk_weighted_reduce,
    "fmha_appendkv_norope": case_fmha_appendkv_norope,
    "fmha_appendkv_rotary": case_fmha_appendkv_rotary,
    "fmha_varlen_causal": case_fmha_varlen_causal,
    "fmha_head_grouping_gqa": case_fmha_head_grouping_gqa,
    "fmha_fwd_fp8": case_fmha_fwd_fp8,
    "fmha_fwd_splitkv_decode": case_fmha_fwd_splitkv_decode,
    "fmha_fwd_paged_prefill": case_fmha_fwd_paged_prefill,
    "fmha_bwd": case_fmha_bwd,
    "fmha_fwd_mfma": case_fmha_fwd_mfma,
    "sage_attention_fp16": case_sage_attention_fp16,
    "sage_attention_fp8": case_sage_attention_fp8,
    "sage_attention_i8": case_sage_attention_i8,
    "sage_attention_i4": case_sage_attention_i4,
    "jenga_sparse": case_jenga_sparse,
    "vsa_sparse": case_vsa_sparse,
    "streamk_gemm": case_streamk_gemm,
    "mfma_gemm": case_mfma_gemm,
    "block_scale_gemm": case_block_scale_gemm,
    "mx_gemm": case_mx_gemm,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--op",
        default="all",
        help="Comma-separated case names (or 'all').",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA / ROCm not available; skipping.")
        return 0

    if args.op == "all":
        cases = list(ALL_CASES.items())
    else:
        wanted = {n.strip() for n in args.op.split(",")}
        cases = [(n, fn) for n, fn in ALL_CASES.items() if n in wanted]
        if not cases:
            print(f"no cases match {args.op!r}; valid: {list(ALL_CASES)}")
            return 2

    results = []
    for name, fn in cases:
        print(f"\n=== {name} ===")
        try:
            r = fn()
            results.append(r)
            print(f"  {r.name}")
            print(
                f"    max_abs={r.max_abs_diff:.4g}  rel={r.rel_max:.4g}  "
                f"range=({r.range_min:.4f}, {r.range_max:.4f})  "
                f"{'PASS' if r.passed else 'FAIL'}"
                f"{('  ' + r.note) if r.note else ''}"
            )
        except Exception as exc:
            import traceback

            traceback.print_exc()
            results.append(
                Result(
                    name=name,
                    passed=False,
                    max_abs_diff=float("inf"),
                    rel_max=float("inf"),
                    range_min=0.0,
                    range_max=0.0,
                    note=f"EXCEPTION: {exc}",
                )
            )

    print("\n" + "=" * 60 + "\nSUMMARY\n" + "=" * 60)
    for r in results:
        tag = "PASS" if r.passed else "FAIL"
        print(f"  {tag}  {r.name:50s} max_abs={r.max_abs_diff:.4g}")
    n_pass = sum(1 for r in results if r.passed)
    print(f"\n{n_pass}/{len(results)} pass")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
