# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Append-KV kernel (CK Tile ``01_fmha`` appendkv parity).

Writes a new K / V token (or block of tokens) into a pre-allocated KV
cache at the position implied by the per-sequence ``seqlen_kv`` array;
the corresponding fwd attention kernel then reads the updated cache.

Optionally applies rotary embedding to K (the standard pattern for
LLaMA-style RoPE caches).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ..core.ir import F32, I32, IRBuilder, KernelDef, PtrType
from ..helpers.io import io_ir_type, load_scalar_as_f32, store_scalar_from_f32
from ..helpers.rotary import (
    RotarySpec,
    apply_rotary_pair_f32,
    load_cos_sin,
    pair_indices,
)
from ..helpers.spec import SignatureBuilder, ceil_div_grid, kernel_name_join
from ._fmha_common import FmhaCommonSpec, validate_common_spec


__all__ = [
    "FmhaAppendKvSpec",
    "build_fmha_fwd_appendkv",
    "fmha_appendkv_grid",
    "fmha_appendkv_signature",
    "is_valid_spec",
]


@dataclass(frozen=True)
class FmhaAppendKvSpec:
    common: FmhaCommonSpec
    batch: int
    rotary: RotarySpec | None = None
    block_size: int = 256
    name: str = "ck_dsl_fmha_appendkv"

    def kernel_name(self) -> str:
        s = self.common.shape
        rot = "rope" if self.rotary is not None else "norope"
        return kernel_name_join(
            self.name,
            f"H{s.head_size}",
            f"HK{s.num_kv_heads}",
            self.common.dtype,
            f"B{self.batch}",
            rot,
            f"b{self.block_size}",
        )


def is_valid_spec(spec: FmhaAppendKvSpec) -> Tuple[bool, str]:
    ok, why = validate_common_spec(spec.common)
    if not ok:
        return False, why
    if spec.batch <= 0:
        return False, f"batch must be > 0 (got {spec.batch})"
    if spec.rotary is not None and spec.rotary.head_size != spec.common.shape.head_size:
        return False, (
            f"rotary head_size ({spec.rotary.head_size}) != "
            f"common head_size ({spec.common.shape.head_size})"
        )
    return True, "ok"


def build_fmha_fwd_appendkv(spec: FmhaAppendKvSpec) -> KernelDef:
    """Append K / V tokens to a pre-allocated KV cache."""
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid fmha_appendkv spec: {why}")
    s = spec.common
    H = s.shape.head_size
    BS = spec.block_size
    dtype = s.dtype
    ty = io_ir_type(dtype)

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BS

    K_new = b.param(
        "K_new", PtrType(ty, "global"), noalias=True, readonly=True, align=16
    )
    V_new = b.param(
        "V_new", PtrType(ty, "global"), noalias=True, readonly=True, align=16
    )
    K_cache = b.param("K_cache", PtrType(ty, "global"), noalias=True, align=16)
    V_cache = b.param("V_cache", PtrType(ty, "global"), noalias=True, align=16)
    seqlen_kv = b.param(
        "seqlen_kv",
        PtrType(I32, "global"),
        noalias=True,
        readonly=True,
        align=4,
    )
    cu_seqlens_new = b.param(
        "cu_seqlens_new",
        PtrType(I32, "global"),
        noalias=True,
        readonly=True,
        align=4,
    )
    cos_table = (
        b.param("cos_table", PtrType(F32, "global"), readonly=True, align=4)
        if spec.rotary is not None
        else None
    )
    sin_table = (
        b.param("sin_table", PtrType(F32, "global"), readonly=True, align=4)
        if spec.rotary is not None
        else None
    )
    total_new_q = b.param("total_new_q", I32)
    _batch = b.param("batch", I32)  # noqa: F841 - ABI
    stride_in_token = b.param("stride_in_token", I32)
    stride_in_head = b.param("stride_in_head", I32)
    stride_cache_token = b.param("stride_cache_token", I32)
    stride_cache_head = b.param("stride_cache_head", I32)

    tid = b.thread_id_x()
    bid = b.block_id_x()
    kv_head_idx = b.block_id_y()
    new_token = b.add(b.mul(bid, b.const_i32(BS)), tid)

    # Bounds-check: the workgroup width is ``block_size`` but the
    # last block may have only a partial slab of tokens. Threads
    # past ``total_new_q`` must skip the body or they'd read OOB
    # metadata and scatter garbage into the cache.
    in_bounds = b.cmp_lt(new_token, total_new_q)
    with b.scf_if(in_bounds):
        _appendkv_body(
            b,
            spec=spec,
            H=H,
            dtype=dtype,
            new_token=new_token,
            kv_head_idx=kv_head_idx,
            K_new=K_new,
            V_new=V_new,
            K_cache=K_cache,
            V_cache=V_cache,
            seqlen_kv=seqlen_kv,
            cu_seqlens_new=cu_seqlens_new,
            cos_table=cos_table,
            sin_table=sin_table,
            stride_in_token=stride_in_token,
            stride_in_head=stride_in_head,
            stride_cache_token=stride_cache_token,
            stride_cache_head=stride_cache_head,
        )
    b.ret()
    return b.kernel


def _appendkv_body(
    b,
    *,
    spec,
    H,
    dtype,
    new_token,
    kv_head_idx,
    K_new,
    V_new,
    K_cache,
    V_cache,
    seqlen_kv,
    cu_seqlens_new,
    cos_table,
    sin_table,
    stride_in_token,
    stride_in_head,
    stride_cache_token,
    stride_cache_head,
):
    """The per-thread appendkv body extracted so the bounds-check
    wrapper above stays compact.
    """

    seq_loop = b.scf_for_iter(
        b.const_i32(0),
        b.const_i32(spec.batch),
        b.const_i32(1),
        [("seq", b.const_i32(0))],
        iv_name="bs_i",
    )
    with seq_loop as (iv, (seq,)):
        cuq_next = b.global_load_i32(cu_seqlens_new, b.add(iv, b.const_i32(1)))
        is_in_seq = b.cmp_lt(new_token, cuq_next)
        b.scf_yield(b.select(is_in_seq, seq, b.add(seq, b.const_i32(1))))
    seq_idx = seq_loop.results[0]
    cu_base = b.global_load_i32(cu_seqlens_new, seq_idx)
    local_new = b.sub(new_token, cu_base)
    seqlen_cur = b.global_load_i32(seqlen_kv, seq_idx)
    dst_pos = b.add(seqlen_cur, local_new)

    in_row_base = b.add(
        b.mul(new_token, stride_in_token), b.mul(kv_head_idx, stride_in_head)
    )
    cache_row_base = b.add(
        b.mul(dst_pos, stride_cache_token), b.mul(kv_head_idx, stride_cache_head)
    )

    if spec.rotary is not None:
        for pair in range(spec.rotary.pair_count):
            lo_d, hi_d = pair_indices(spec.rotary, pair)
            k_lo = load_scalar_as_f32(
                b, K_new, b.add(in_row_base, b.const_i32(lo_d)), dtype=dtype
            )
            k_hi = load_scalar_as_f32(
                b, K_new, b.add(in_row_base, b.const_i32(hi_d)), dtype=dtype
            )
            cos_v, sin_v = load_cos_sin(
                b,
                cos_table,
                sin_table,
                token_pos=dst_pos,
                pair_idx=b.const_i32(pair),
                spec=spec.rotary,
            )
            new_lo, new_hi = apply_rotary_pair_f32(b, k_lo, k_hi, cos_v, sin_v)
            store_scalar_from_f32(
                b,
                K_cache,
                b.add(cache_row_base, b.const_i32(lo_d)),
                new_lo,
                dtype=dtype,
            )
            store_scalar_from_f32(
                b,
                K_cache,
                b.add(cache_row_base, b.const_i32(hi_d)),
                new_hi,
                dtype=dtype,
            )
    else:
        for d in range(H):
            v = load_scalar_as_f32(
                b, K_new, b.add(in_row_base, b.const_i32(d)), dtype=dtype
            )
            store_scalar_from_f32(
                b,
                K_cache,
                b.add(cache_row_base, b.const_i32(d)),
                v,
                dtype=dtype,
            )

    for d in range(H):
        v = load_scalar_as_f32(
            b, V_new, b.add(in_row_base, b.const_i32(d)), dtype=dtype
        )
        store_scalar_from_f32(
            b,
            V_cache,
            b.add(cache_row_base, b.const_i32(d)),
            v,
            dtype=dtype,
        )


def fmha_appendkv_grid(
    spec: FmhaAppendKvSpec, *, total_new_q: int
) -> Tuple[int, int, int]:
    gx, _, _ = ceil_div_grid((total_new_q, spec.block_size))
    return (gx, spec.common.shape.num_kv_heads, 1)


def fmha_appendkv_signature(spec: FmhaAppendKvSpec):
    sig = (
        SignatureBuilder()
        .ptr("K_new", spec.common.dtype)
        .ptr("V_new", spec.common.dtype)
        .ptr("K_cache", spec.common.dtype)
        .ptr("V_cache", spec.common.dtype)
        .ptr("seqlen_kv", "i32")
        .ptr("cu_seqlens_new", "i32")
    )
    if spec.rotary is not None:
        sig = sig.ptr("cos_table", "f32").ptr("sin_table", "f32")
    return (
        sig.scalar("total_new_q", "i32")
        .scalar("batch", "i32")
        .scalar("stride_in_token", "i32")
        .scalar("stride_in_head", "i32")
        .scalar("stride_cache_token", "i32")
        .scalar("stride_cache_head", "i32")
        .build()
    )
