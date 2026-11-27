# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""End-to-end parity harness for the attention kernel family (library layer).

This module covers FMHA / Sage / Sparse attention kernels that live in the
``kernels`` package (library layer).  Platform MoE / GEMM cases live in
``rocke.examples.common.parity_extended_kernels`` (platform layer).

The shared data and case-runner helpers come from
``rocke.examples.common.parity_support`` -- library importing from platform is
legal per the one-way layering rule.

Usage::

    python -m builders.common.parity_fmha_extended [--arch gfx950]

The harness exits non-zero if any kernel's max abs diff exceeds the
per-kernel tolerance, so it doubles as a smoke gate for CI.

Inputs are numpy draws from a fixed per-case seed, stored in the kernel's
element type and launched from ``DeviceArray`` buffers. The attention
references are :func:`rocke.numeric.sdpa_reference.sdpa_reference` and
:func:`~rocke.numeric.sdpa_reference.sdpa_reference_bwd` on the stored values.
"""

# Standard attention notation: ``O`` is the output tensor (Q, K, V, O).
# Ruff's ambiguous-name lint (E741) is silenced for this file because the
# math notation is the primary readability win.
# ruff: noqa: E741

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from itertools import accumulate

import numpy as np

from rocke.helpers import QkScaleSpec  # noqa: E402
from rocke.helpers.rotary import RotarySpec  # noqa: E402
from rocke.numeric import storage
from rocke.numeric.compare import max_abs_error
from rocke.numeric.sdpa_reference import sdpa_reference, sdpa_reference_bwd
from rocke.runtime.device_array import DeviceArray
from rocke.runtime.launcher import KernelLauncher  # noqa: E402
from kernels import (  # noqa: E402
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
    sage_attention_grid,
    sage_attention_signature,
    vsa_sparse_attention_grid,
    vsa_sparse_attention_signature,
)
from kernels import (  # noqa: E402
    FmhaAppendKvSpec,
    FmhaBwdSpec,
    FmhaCommonSpec,
    FmhaFwdFp8Spec,
    FmhaFwdHeadGroupingSpec,
    FmhaFwdPagedPrefillSpec,
    FmhaFwdSplitKvDecodeSpec,
    FmhaFwdVarlenSpec,
    FmhaShape,
    JengaSparseSpec,
    SageAttentionSpec,
    VsaSparseSpec,
    build_fmha_bwd,
    build_fmha_fwd_appendkv,
    build_fmha_fwd_fp8,
    build_fmha_fwd_head_grouping,
    build_fmha_fwd_paged_prefill,
    build_fmha_fwd_splitkv_decode_reduce,
    build_fmha_fwd_splitkv_decode_segment,
    build_fmha_fwd_varlen,
    build_jenga_sparse_attention,
    build_sage_attention,
    build_vsa_sparse_attention,
)
from kernels import fmha_fwd_mfma_grid, fmha_fwd_mfma_signature  # noqa: E402
from kernels import FmhaMfmaSpec, build_fmha_fwd_mfma  # noqa: E402

from rocke.examples.common.parity_support import (  # noqa: E402
    Result,
    default_arch,
    device_available,
    device_values,
    draw,
    launch,
    require_ocp_fp8_arch,
    round_to_f16,
    run_cases,
    summarise,
    values,
)
from rocke.helpers import compile_kernel  # noqa: E402

# Target gfx arch for all compiles in this harness.  Set by main() from
# --arch / default_arch() before any case runs.
_ARCH = "gfx950"


def _compile(kernel):
    """Compile a kernel for the harness-selected arch (see ``_ARCH``)."""
    return compile_kernel(kernel, arch=_ARCH)


# ---------------------------------------------------------------------------
# Dense attention reference
# ---------------------------------------------------------------------------


def _ref_attention(Q, K, V, *, causal: bool = False, scale=None) -> np.ndarray:
    """Attention of one sequence, rounded to fp16 as the kernels store ``O``.

    ``Q`` is ``(seqlen_q, HQ, D)`` and ``K`` / ``V`` are ``(seqlen_k, HK, D)``
    values; query head ``h`` reads KV head ``h // (HQ // HK)``. ``causal`` keeps
    keys ``k <= q`` (top-left). ``scale`` multiplies the scores (default
    ``1 / sqrt(D)``).
    """
    out, _ = sdpa_reference(
        Q,
        K,
        V,
        scale=scale,
        causal=causal,
        q_offsets=[0, Q.shape[0]],
        kv_offsets=[0, K.shape[0]],
    )
    return round_to_f16(out)


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
    kernel = build_fmha_fwd_appendkv(spec, arch=_ARCH)
    art = _compile(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=fmha_appendkv_signature(spec),
    )
    rng = np.random.default_rng(10)
    K_new = draw(rng, total_new, HK, head_size)
    V_new = draw(rng, total_new, HK, head_size)
    K_cache = DeviceArray.zeros((cache_capacity, HK, head_size), "fp16")
    V_cache = DeviceArray.zeros((cache_capacity, HK, head_size), "fp16")
    launch(
        launcher,
        {
            "K_new": DeviceArray(K_new),
            "V_new": DeviceArray(V_new),
            "K_cache": K_cache,
            "V_cache": V_cache,
            "seqlen_kv": DeviceArray(np.array(seqlen_kv_init, dtype=np.int32)),
            "cu_seqlens_new": DeviceArray(
                np.array([0] + list(accumulate(new_lens)), dtype=np.int32)
            ),
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
    K_ref = np.zeros(K_cache.shape)
    V_ref = np.zeros(V_cache.shape)
    for local in range(new_lens[0]):
        dst = seqlen_kv_init[0] + local
        K_ref[dst] = values(K_new[local], "fp16")
        V_ref[dst] = values(V_new[local], "fp16")
    r = summarise(
        np.concatenate(
            [device_values(K_cache).ravel(), device_values(V_cache).ravel()]
        ),
        np.concatenate([K_ref.ravel(), V_ref.ravel()]),
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
    kernel = build_fmha_fwd_appendkv(spec, arch=_ARCH)
    art = _compile(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=fmha_appendkv_signature(spec),
    )
    rng = np.random.default_rng(11)
    K_new = draw(rng, total_new, HK, head_size)
    V_new = draw(rng, total_new, HK, head_size)
    K_cache = DeviceArray.zeros((cache_capacity, HK, head_size), "fp16")
    V_cache = DeviceArray.zeros((cache_capacity, HK, head_size), "fp16")
    pair_count = head_size // 2
    pos = np.arange(cache_capacity, dtype=np.float32)
    inv_freq = np.float32(1.0) / (
        np.float32(10000.0)
        ** (np.arange(0, pair_count, dtype=np.float32) * np.float32(2.0 / head_size))
    )
    freqs = np.outer(pos, inv_freq)
    cos_tbl = np.cos(freqs)
    sin_tbl = np.sin(freqs)
    launch(
        launcher,
        {
            "K_new": DeviceArray(K_new),
            "V_new": DeviceArray(V_new),
            "K_cache": K_cache,
            "V_cache": V_cache,
            "seqlen_kv": DeviceArray(np.array(seqlen_kv_init, dtype=np.int32)),
            "cu_seqlens_new": DeviceArray(
                np.array([0] + list(accumulate(new_lens)), dtype=np.int32)
            ),
            "cos_table": DeviceArray(cos_tbl),
            "sin_table": DeviceArray(sin_tbl),
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
    K_ref = np.zeros(K_cache.shape)
    for local in range(new_lens[0]):
        dst = seqlen_kv_init[0] + local
        c = cos_tbl[dst].astype(np.float64)
        s = sin_tbl[dst].astype(np.float64)
        k = values(K_new[local], "fp16")
        lo, hi = k[:, :pair_count], k[:, pair_count:]
        new_lo = lo * c - hi * s
        new_hi = lo * s + hi * c
        K_ref[dst] = round_to_f16(np.concatenate([new_lo, new_hi], axis=-1))
    r = summarise(device_values(K_cache), K_ref, tol=5e-3, note="K rotated; V copied")
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
    kernel = build_fmha_fwd_varlen(spec, arch=_ARCH)
    art = _compile(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=fmha_fwd_varlen_signature(spec),
    )
    rng = np.random.default_rng(20)
    Q = draw(rng, total_q, HQ, head_size)
    K = draw(rng, total_q, HK, head_size)
    V = draw(rng, total_q, HK, head_size)
    O = DeviceArray.zeros(Q.shape, "fp16")
    cu = DeviceArray(np.array([0] + list(accumulate(seq_lens)), dtype=np.int32))
    launch(
        launcher,
        {
            "Q": DeviceArray(Q),
            "K": DeviceArray(K),
            "V": DeviceArray(V),
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
    q, k, v = (values(x, "fp16") for x in (Q, K, V))
    O_ref = np.zeros(Q.shape)
    for seq in range(len(seq_lens)):
        lo, hi = sum(seq_lens[:seq]), sum(seq_lens[: seq + 1])
        O_ref[lo:hi] = _ref_attention(q[lo:hi], k[lo:hi], v[lo:hi], causal=True)
    r = summarise(device_values(O), O_ref, tol=5e-3)
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
    kernel = build_fmha_fwd_head_grouping(spec, arch=_ARCH)
    art = _compile(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=fmha_fwd_head_grouping_signature(spec),
    )
    rng = np.random.default_rng(30)
    Q = draw(rng, batch * seq_q, HQ, head_size)
    K = draw(rng, batch * seq_k, HK, head_size)
    V = draw(rng, batch * seq_k, HK, head_size)
    O = DeviceArray.zeros(Q.shape, "fp16")
    launch(
        launcher,
        {
            "Q": DeviceArray(Q),
            "K": DeviceArray(K),
            "V": DeviceArray(V),
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
    q, k, v = (values(x, "fp16") for x in (Q, K, V))
    O_ref = np.zeros(Q.shape)
    for b in range(batch):
        ql, qh = b * seq_q, (b + 1) * seq_q
        kl, kh = b * seq_k, (b + 1) * seq_k
        O_ref[ql:qh] = _ref_attention(q[ql:qh], k[kl:kh], v[kl:kh], causal=True)
    r = summarise(device_values(O), O_ref, tol=5e-3, note=f"GQA HQ={HQ} HK={HK}")
    r.name = "fmha_fwd_head_grouping (GQA causal)"
    return r


def case_fmha_fwd_fp8() -> Result:
    require_ocp_fp8_arch("fmha_fwd_fp8", _ARCH)
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
    kernel = build_fmha_fwd_fp8(spec, arch=_ARCH)
    art = _compile(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=fmha_fwd_fp8_signature(spec),
    )
    rng = np.random.default_rng(40)
    Q = draw(rng, seq_q, HQ, head_size)
    # gfx950 uses OCP FP8 (e4m3fn, bias=7) -- ``e4m3fnuz`` is the
    # older AMD pre-OCP format and would give 2x the values after cvt.
    K_fp8 = draw(rng, seq_k, HK, head_size, dtype="fp8e4m3", scale=0.5)
    V_fp8 = draw(rng, seq_k, HK, head_size, dtype="fp8e4m3", scale=0.5)
    O = DeviceArray.zeros(Q.shape, "fp16")
    k_scale_val, v_scale_val = 1.5, 0.75
    launch(
        launcher,
        {
            "Q": DeviceArray(Q),
            "K": DeviceArray(K_fp8, "fp8e4m3"),
            "V": DeviceArray(V_fp8, "fp8e4m3"),
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
    K_dq = values(K_fp8, "fp8e4m3") * k_scale_val
    V_dq = values(V_fp8, "fp8e4m3") * v_scale_val
    O_ref = _ref_attention(values(Q, "fp16"), K_dq, V_dq)
    r = summarise(device_values(O), O_ref, tol=5e-3, note="OCP fp8e4m3fn (gfx950)")
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
    seg_k = build_fmha_fwd_splitkv_decode_segment(spec, arch=_ARCH)
    red_k = build_fmha_fwd_splitkv_decode_reduce(spec, arch=_ARCH)
    seg_art = _compile(seg_k)
    red_art = _compile(red_k)
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
    rng = np.random.default_rng(50)
    Q = draw(rng, batch, HQ, head_size)
    K = draw(rng, batch * max_k, HK, head_size)
    V = draw(rng, batch * max_k, HK, head_size)
    O = DeviceArray.zeros(Q.shape, "fp16")
    ws_size = num_segments * batch * HQ
    ws_m = DeviceArray.zeros((ws_size,), "fp32")
    ws_l = DeviceArray.zeros((ws_size,), "fp32")
    ws_acc = DeviceArray.zeros((ws_size * head_size,), "fp32")
    scale_log2 = math.log2(math.e) / math.sqrt(head_size)
    launch(
        seg_launcher,
        {
            "Q": DeviceArray(Q),
            "K": DeviceArray(K),
            "V": DeviceArray(V),
            "seqlens_k": DeviceArray(np.array(seqlen_k_per_batch, dtype=np.int32)),
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
    launch(
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
    q, k, v = (values(x, "fp16") for x in (Q, K, V))
    O_ref = np.zeros(Q.shape)
    for seq in range(batch):
        # Every sequence reads its keys from the start of K / V.
        L = seqlen_k_per_batch[seq]
        O_ref[seq] = _ref_attention(q[seq : seq + 1], k[:L], v[:L])[0]
    r = summarise(device_values(O), O_ref, tol=5e-3, note=f"segments={num_segments}")
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
    kernel = build_fmha_fwd_paged_prefill(spec, arch=_ARCH)
    art = _compile(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=fmha_fwd_paged_prefill_signature(spec),
    )
    rng = np.random.default_rng(60)
    Q = draw(rng, seqlen_q, HQ, head_size)
    num_blocks = 6
    K_cache = draw(rng, num_blocks, page_block_size, HK, head_size)
    V_cache = draw(rng, num_blocks, page_block_size, HK, head_size)
    O = DeviceArray.zeros(Q.shape, "fp16")
    # Non-contiguous physical block assignment -- this is the test that
    # caught the v1 "first-block-only" bug.
    block_table = np.array([[3, 1, 5, 0]], dtype=np.int32)

    stride_block = page_block_size * HK * head_size
    stride_page = HK * head_size
    stride_kv_head = head_size

    launch(
        launcher,
        {
            "Q": DeviceArray(Q),
            "K_cache": DeviceArray(K_cache),
            "V_cache": DeviceArray(V_cache),
            "O": O,
            "block_table": DeviceArray(block_table),
            "cu_seqlens_q": DeviceArray(np.array([0, seqlen_q], dtype=np.int32)),
            "seqlens_k": DeviceArray(np.array([seqlen_k], dtype=np.int32)),
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
    K_seq = values(np.concatenate([K_cache[p] for p in physical_blocks]), "fp16")
    V_seq = values(np.concatenate([V_cache[p] for p in physical_blocks]), "fp16")
    O_ref = _ref_attention(values(Q, "fp16"), K_seq, V_seq)
    r = summarise(
        device_values(O),
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
    kernel = build_sage_attention(spec, arch=_ARCH)
    art = _compile(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=sage_attention_signature(spec),
    )
    rng = np.random.default_rng(70)
    Q = draw(rng, seq_q, HQ, head_size)
    K = draw(rng, seq_k, HK, head_size)
    V = draw(rng, seq_k, HK, head_size)
    O = DeviceArray.zeros(Q.shape, "fp16")
    launch(
        launcher,
        {
            "Q": DeviceArray(Q),
            "K": DeviceArray(K),
            "V": DeviceArray(V),
            "O": O,
            "q_scale": DeviceArray(np.ones(q_blocks * HQ, dtype=np.float32)),
            "k_scale": DeviceArray(np.ones(k_blocks * HK, dtype=np.float32)),
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
    O_ref = _ref_attention(*(values(x, "fp16") for x in (Q, K, V)))
    r = summarise(device_values(O), O_ref, tol=5e-3, note="baseline scales=1")
    r.name = "sage_attention fp16_bf16 (scales=1)"
    return r


def case_sage_attention_fp8() -> Result:
    require_ocp_fp8_arch("sage_attention fp8_bf16", _ARCH)
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
    kernel = build_sage_attention(spec, arch=_ARCH)
    art = _compile(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=sage_attention_signature(spec),
    )
    rng = np.random.default_rng(80)
    Q = draw(rng, seq_q, HQ, head_size)
    K_fp8 = draw(rng, seq_k, HK, head_size, dtype="fp8e4m3", scale=0.5)
    V_fp8 = draw(rng, seq_k, HK, head_size, dtype="fp8e4m3", scale=0.5)
    O = DeviceArray.zeros(Q.shape, "fp16")
    qs_val, ks_val = 1.5, 0.7
    launch(
        launcher,
        {
            "Q": DeviceArray(Q),
            "K": DeviceArray(K_fp8, "fp8e4m3"),
            "V": DeviceArray(V_fp8, "fp8e4m3"),
            "O": O,
            "q_scale": DeviceArray(np.full(q_blocks * HQ, qs_val, dtype=np.float32)),
            "k_scale": DeviceArray(np.full(k_blocks * HK, ks_val, dtype=np.float32)),
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
    # Sage applies per-block QK scales to the SCORE, not to K/V.
    O_ref = _ref_attention(
        values(Q, "fp16"),
        values(K_fp8, "fp8e4m3"),
        values(V_fp8, "fp8e4m3"),
        scale=qs_val * ks_val / math.sqrt(head_size),
    )
    r = summarise(
        device_values(O),
        O_ref,
        tol=5e-3,
        note=f"per-block scales q={qs_val} k={ks_val}",
    )
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
    kernel = build_jenga_sparse_attention(spec, arch=_ARCH)
    art = _compile(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=jenga_sparse_attention_signature(spec),
    )
    rng = np.random.default_rng(90)
    Q = draw(rng, seq_q, HQ, head_size)
    K = draw(rng, seq_k, HK, head_size)
    V = draw(rng, seq_k, HK, head_size)
    O = DeviceArray.zeros(Q.shape, "fp16")
    mask = np.zeros((seq_q, num_kb), dtype=np.int8)
    mask[:, 1] = 1
    launch(
        launcher,
        {
            "Q": DeviceArray(Q),
            "K": DeviceArray(K),
            "V": DeviceArray(V),
            "O": O,
            "mask": DeviceArray(mask),
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
    q, k, v = (values(x, "fp16") for x in (Q, K, V))
    O_ref = _ref_attention(q, k[block_k : 2 * block_k], v[block_k : 2 * block_k])
    r = summarise(
        device_values(O), O_ref, tol=5e-3, note="attend to second k-block only"
    )
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
    kernel = build_vsa_sparse_attention(spec, arch=_ARCH)
    art = _compile(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=vsa_sparse_attention_signature(spec),
    )
    rng = np.random.default_rng(100)
    Q = draw(rng, seq_q, HQ, head_size)
    K = draw(rng, seq_k, HK, head_size)
    V = draw(rng, seq_k, HK, head_size)
    O = DeviceArray.zeros(Q.shape, "fp16")
    launch(
        launcher,
        {
            "Q": DeviceArray(Q),
            "K": DeviceArray(K),
            "V": DeviceArray(V),
            "O": O,
            "block_lut": DeviceArray(np.array([[0, 3]] * seq_q, dtype=np.int32)),
            "block_count": DeviceArray(np.array([2] * seq_q, dtype=np.int32)),
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
    q, k, v = (values(x, "fp16") for x in (Q, K, V))
    K_chunks = np.concatenate([k[0:block_k], k[3 * block_k : 4 * block_k]])
    V_chunks = np.concatenate([v[0:block_k], v[3 * block_k : 4 * block_k]])
    O_ref = _ref_attention(q, K_chunks, V_chunks)
    r = summarise(device_values(O), O_ref, tol=5e-3, note="LUT={0, 3}")
    r.name = "vsa_sparse_attention"
    return r


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
    kernel = build_sage_attention(spec, arch=_ARCH)
    art = _compile(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=sage_attention_signature(spec),
    )
    rng = np.random.default_rng(110 if quant_mode == "i8_fp8_bf16" else 120)
    Q = draw(rng, seq_q, HQ, head_size)
    O = DeviceArray.zeros(Q.shape, "fp16")
    qs_val, ks_val = 0.6, 1.3

    if quant_mode == "i8_fp8_bf16":
        # K/V quantised to i8 in [-127, 127]; codebook holds the
        # f32-domain value for each (i8 + 128) index.
        codebook_vals = np.arange(-128, 128, dtype=np.float32) / np.float32(64.0)
        K_buf = rng.integers(-127, 128, (seq_k, HK, head_size)).astype(np.int8)
        V_buf = rng.integers(-127, 128, (seq_k, HK, head_size)).astype(np.int8)
        K_dq_dom = codebook_vals[K_buf.astype(np.int32) + 128]
        V_dq_dom = codebook_vals[V_buf.astype(np.int32) + 128]
    else:  # i4_fp8_bf16
        # 16-entry codebook (one per i4 value -8..7).
        codebook_vals = np.arange(-8, 8, dtype=np.float32) / np.float32(4.0)
        # Each byte holds two nibbles: low nibble = lo elem, high = hi.
        # The byte-flat layout is (seq_k, HK, head_size/2) -- consecutive
        # bytes are consecutive lanes, and lane ``tid`` owns head_dim slots
        # [2*tid, 2*tid+1] (lo, hi).
        n_bytes = seq_k * HK * head_size // 2
        K_bytes = rng.integers(0, 256, n_bytes).astype(np.uint8)
        V_bytes = rng.integers(0, 256, n_bytes).astype(np.uint8)
        K_buf, V_buf = K_bytes.view(np.int8), V_bytes.view(np.int8)

        def dequant(packed: np.ndarray) -> np.ndarray:
            """The codebook values of every nibble, in head_dim order."""
            nibbles = np.stack([packed & 0xF, packed >> 4], axis=-1).astype(np.int32)
            signed = np.where(nibbles >= 8, nibbles - 16, nibbles)
            return codebook_vals[signed + 8].reshape(seq_k, HK, head_size)

        K_dq_dom = dequant(K_bytes)
        V_dq_dom = dequant(V_bytes)
    # The codebook output gets rounded to fp8e4m3, then read back to f32 in
    # the kernel. Apply that same round-trip in the reference so we compare
    # against what the kernel actually accumulates.
    K_dq = storage.round_to(K_dq_dom, "fp8e4m3").astype(np.float64)
    V_dq = storage.round_to(V_dq_dom, "fp8e4m3").astype(np.float64)

    # i4 / i8 store packed bytes -- the K/V stride is element-stride
    # in the *byte* space. For i4 each byte holds two elements so the
    # stride halves; i8 keeps stride == head_size as one byte = one
    # element.
    kv_bytes_per_elem = 0.5 if quant_mode == "i4_fp8_bf16" else 1
    sk_token = int(HK * head_size * kv_bytes_per_elem)
    sk_head = int(head_size * kv_bytes_per_elem)
    launch_args = {
        "Q": DeviceArray(Q),
        "K": DeviceArray(K_buf),
        "V": DeviceArray(V_buf),
        "O": O,
        "q_scale": DeviceArray(np.full(q_blocks * HQ, qs_val, dtype=np.float32)),
        "k_scale": DeviceArray(np.full(k_blocks * HK, ks_val, dtype=np.float32)),
        "codebook_k": DeviceArray(codebook_vals),
        "codebook_v": DeviceArray(codebook_vals),
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
    launch(launcher, launch_args, grid=sage_attention_grid(spec))

    O_ref = _ref_attention(
        values(Q, "fp16"), K_dq, V_dq, scale=qs_val * ks_val / math.sqrt(head_size)
    )
    r = summarise(device_values(O), O_ref, tol=5e-2, note=f"quant_mode={quant_mode}")
    r.name = f"sage_attention {quant_mode}"
    return r


def case_sage_attention_i8() -> Result:
    return _sage_int_case("i8_fp8_bf16")


def case_sage_attention_i4() -> Result:
    return _sage_int_case("i4_fp8_bf16")


# ---------------------------------------------------------------------
# FMHA MFMA tiled forward
# ---------------------------------------------------------------------


def case_fmha_fwd_mfma() -> Result:
    """MFMA-tiled FMHA forward (16x16 Q tile, MFMA QK + softmax + PV)."""
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
    kernel = build_fmha_fwd_mfma(spec, arch=_ARCH)
    art = _compile(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=fmha_fwd_mfma_signature(spec),
    )
    rng = np.random.default_rng(400)
    Q = draw(rng, batch * seqlen_q, HQ, head_size)
    K = draw(rng, batch * seqlen_k, HK, head_size)
    V = draw(rng, batch * seqlen_k, HK, head_size)
    O = DeviceArray.zeros(Q.shape, "fp16")
    launch(
        launcher,
        {
            "Q": DeviceArray(Q),
            "K": DeviceArray(K),
            "V": DeviceArray(V),
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
    O_ref = _ref_attention(*(values(x, "fp16") for x in (Q, K, V)))
    r = summarise(
        device_values(O),
        O_ref,
        tol=5e-3,
        note="mfma_f32_16x16x16_f16 QK + softmax + PV chain",
    )
    r.name = "fmha_fwd_mfma (16x16 atom)"
    return r


# ---------------------------------------------------------------------
# FMHA backward
# ---------------------------------------------------------------------


def case_fmha_bwd() -> Result:
    """dQ / dK / dV via atomic fp32 accumulate vs the float64 SDPA backward."""
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
    kernel = build_fmha_bwd(spec, arch=_ARCH)
    art = _compile(kernel)
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=fmha_bwd_signature(spec),
    )
    rng = np.random.default_rng(300)
    Q = draw(rng, seq_q, HQ, head_size)
    K = draw(rng, seq_k, HK, head_size)
    V = draw(rng, seq_k, HK, head_size)
    dO = draw(rng, seq_q, HQ, head_size)
    q, k, v, do = (values(x, "fp16") for x in (Q, K, V, dO))

    # The saved forward statistics in the log2 domain: the row max M of the
    # scaled scores and the softmax denominator L relative to it.
    scores_log2 = np.einsum("qhd,khd->qhk", q, k) / math.sqrt(head_size)
    scores_log2 *= math.log2(math.e)
    M_saved = scores_log2.max(axis=-1)
    L_saved = np.exp2(scores_log2 - M_saved[..., None]).sum(axis=-1)

    dQ = DeviceArray.zeros((seq_q, HQ, head_size), "fp32")
    dK = DeviceArray.zeros((seq_k, HK, head_size), "fp32")
    dV = DeviceArray.zeros((seq_k, HK, head_size), "fp32")
    launch(
        launcher,
        {
            "Q": DeviceArray(Q),
            "K": DeviceArray(K),
            "V": DeviceArray(V),
            "dO": DeviceArray(dO),
            "M_saved": DeviceArray(M_saved.astype(np.float32)),
            "L_saved": DeviceArray(L_saved.astype(np.float32)),
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

    grads = sdpa_reference_bwd(
        q,
        k,
        v,
        o=None,
        do=do,
        q_offsets=[0, seq_q],
        kv_offsets=[0, seq_k],
        recompute_lse=True,
    )
    dQ_out, dK_out, dV_out = (device_values(x) for x in (dQ, dK, dV))

    # The kernel's bwd doesn't sum per-head_idx across the HQ heads when
    # HQ == HK; each Q head writes to its own dQ row. Compare each.
    diff_dQ = max_abs_error(dQ_out, grads.dq)
    diff_dK = max_abs_error(dK_out, grads.dk)
    diff_dV = max_abs_error(dV_out, grads.dv)
    max_d = max(diff_dQ, diff_dK, diff_dV)
    ref_max = max(float(np.abs(g).max()) for g in (grads.dq, grads.dk, grads.dv))
    rel = max_d / (ref_max + 1e-9)
    print(f"    dQ max={diff_dQ:.4g} dK max={diff_dK:.4g} dV max={diff_dV:.4g}")
    r = Result(
        name="fmha_bwd (atomic accumulate)",
        passed=(max_d < 5e-2),
        max_abs_diff=max_d,
        rel_max=rel,
        range_min=min(float(x.min()) for x in (dQ_out, dK_out, dV_out)),
        range_max=max(float(x.max()) for x in (dQ_out, dK_out, dV_out)),
        note=f"dQ={diff_dQ:.3g} dK={diff_dK:.3g} dV={diff_dV:.3g}",
    )
    return r


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------

ALL_CASES: dict[str, Callable[[], Result]] = {
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
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--op",
        default="all",
        help="Comma-separated case names (or 'all').",
    )
    parser.add_argument(
        "--arch",
        default=None,
        help="gfx target for codegen (default: running device, else gfx950).",
    )
    args = parser.parse_args()

    global _ARCH
    _ARCH = args.arch or default_arch()
    print(f"codegen arch: {_ARCH}")

    if not device_available():
        print("HIP device not available; skipping.")
        return 0

    if args.op == "all":
        cases = list(ALL_CASES.items())
    else:
        wanted = {n.strip() for n in args.op.split(",")}
        cases = [(n, fn) for n, fn in ALL_CASES.items() if n in wanted]
        if not cases:
            print(f"no cases match {args.op!r}; valid: {list(ALL_CASES)}")
            return 2
    return run_cases(cases, arch=_ARCH)


if __name__ == "__main__":
    raise SystemExit(main())
