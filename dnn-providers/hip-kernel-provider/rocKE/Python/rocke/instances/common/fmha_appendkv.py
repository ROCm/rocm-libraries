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

from ...core.ir import F32, I32, IRBuilder, KernelDef, PtrType
from ...helpers.io import (
    io_ir_type,
    load_scalar_as_f32,
    pack_f32_to,
    store_scalar_from_f32,
    vector_row_copy,
)
from ...helpers.rotary import (
    RotarySpec,
    apply_rotary_pair_f32,
    load_cos_sin,
    pair_indices,
)
from ...helpers.spec import SignatureBuilder, ceil_div_grid, kernel_name_join
from ._fmha_common import FmhaCommonSpec, validate_common_spec


# CK Tile's appendkv default policy reads 16 bytes at a time
# (``GetAlignmentK = 16 / sizeof(KDataType) = 8`` for f16/bf16; see
# ``include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_appendkv_pipeline_default_policy.hpp``).
# The corresponding scalar load chain we replace below issues ``H`` 2-byte
# loads per thread; the vector path collapses that to ``H/_VEC`` 16-byte
# loads, the same shape ``buffer_load_dwordx4`` lowers to in the C++
# reference.
_VEC = 8  # 16-byte vector for f16 / bf16


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
    name: str = "rocke_fmha_appendkv"

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


def is_valid_spec(spec: FmhaAppendKvSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for one append-KV config on ``arch``.

    This kernel is a pure vectorised KV-cache scatter (optionally fused
    with rotary): it issues no MFMA atoms and allocates no LDS, so the
    only architecture fact that matters is the per-WG thread cap. It is
    sourced from :class:`rocke.core.arch.ArchTarget` so the predicate
    rejects an unknown arch (or an over-budget ``block_size``) with a
    structured reason rather than failing later at lower/launch time.
    The kernel builds identically on gfx942 and gfx950 (the f16 / bf16
    vector load / store path is shared CDNA hardware), so gfx950
    behavior is unchanged.
    """
    from ...core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)

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
    if spec.block_size > target.max_threads_per_block:
        return False, (
            f"block_size {spec.block_size} > max_threads_per_block "
            f"{target.max_threads_per_block} on {arch}"
        )
    return True, "ok"


def build_fmha_fwd_appendkv(spec: FmhaAppendKvSpec, arch: str = "gfx950") -> KernelDef:
    """Append K / V tokens to a pre-allocated KV cache.

    This is a pure vectorised KV-cache scatter (optionally fused with
    rotary): it emits no MFMA atoms and allocates no LDS, so the f16 /
    bf16 vector load / store path is identical on gfx942 and gfx950.
    ``arch`` is threaded only through :func:`is_valid_spec` (per-WG
    thread-cap check); the emitted IR is byte-identical across arches.
    """
    ok, why = is_valid_spec(spec, arch)
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

    Address math + sequence lookup runs first; the actual K / V
    copy is delegated to the vectorised helpers below so the body
    stays readable and the scalar / vector choice lives in one spot.
    """

    # Per-token sequence lookup. ``new_token`` is wave-non-uniform so
    # the scan result is *per-thread* (no readfirstlane). Python-
    # unrolling the loop (``spec.batch`` is compile-time constant)
    # drops the scf.for overhead — the compiler emits an unrolled
    # chain of ``v_cmp / v_cndmask`` instead of a back-edge loop.
    # This mirrors the CK Tile reference where ``i_batch`` is on
    # ``blockIdx.z`` and the per-batch metadata is one
    # ``buffer_load_dword`` per CTA.
    seq = b.const_i32(0)
    for i in range(spec.batch):
        cuq_next = b.global_load_i32(cu_seqlens_new, b.const_i32(i + 1))
        is_in_seq = b.cmp_lt(new_token, cuq_next)
        seq = b.select(is_in_seq, seq, b.add(seq, b.const_i32(1)))
    seq_idx = seq
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

    _appendkv_copy_k(
        b,
        spec=spec,
        H=H,
        dtype=dtype,
        K_new=K_new,
        K_cache=K_cache,
        cos_table=cos_table,
        sin_table=sin_table,
        dst_pos=dst_pos,
        in_row_base=in_row_base,
        cache_row_base=cache_row_base,
    )
    _appendkv_copy_v(
        b,
        H=H,
        dtype=dtype,
        V_new=V_new,
        V_cache=V_cache,
        in_row_base=in_row_base,
        cache_row_base=cache_row_base,
    )


def _copy_row_vec(
    b,
    *,
    H,
    dtype,
    src_ptr,
    dst_ptr,
    src_row_base,
    dst_row_base,
):
    """Vectorised row copy along the head dim.

    Replaces the scalar ``for d in range(H): load + store`` with
    ``H // _VEC`` 16-byte vector copies (each
    ``buffer_load_dwordx4`` + ``buffer_store_dwordx4`` on AMDGPU);
    the residual ``H % _VEC`` tail falls back to scalar so any
    future head_size we add can still go through this path.

    Matches CK Tile's ``store_tile(k_dram_block_window, knew_tile)``
    semantics in
    ``include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_appendkv_pipeline.hpp``:
    one head_dim row is moved with the widest aligned vector the
    pointer's ``align=16`` declaration supports.

    Delegates to :func:`rocke.helpers.io.vector_row_copy` (whose
    docstring names this function as its promotion source); the helper
    emits the identical ``H // _VEC`` vector copies + scalar tail. The
    ``vec_bytes=16`` argument maps to ``_VEC = 16 / 2`` for f16 / bf16.
    """
    vector_row_copy(
        b,
        src=src_ptr,
        dst=dst_ptr,
        src_base=src_row_base,
        dst_base=dst_row_base,
        H=H,
        dtype=dtype,
        vec_bytes=_VEC * 2,
    )


def _appendkv_copy_k(
    b,
    *,
    spec,
    H,
    dtype,
    K_new,
    K_cache,
    cos_table,
    sin_table,
    dst_pos,
    in_row_base,
    cache_row_base,
):
    """K-cache copy, optionally fused with rotary embedding.

    No-rotary path is a straight vector copy
    (:func:`_copy_row_vec`). Rotary paths read the K row with vec
    loads, apply the 2x2 rotation per pair on f32, then pack +
    vector-store. CK Tile's
    ``BlockRotaryEmbedding<RotaryEnum>::apply`` (see
    ``include/ck_tile/ops/fmha/block/block_rotary_embedding.hpp``)
    does the same: load a knew tile vectorised, rotate, store back
    vectorised.
    """
    if spec.rotary is None:
        _copy_row_vec(
            b,
            H=H,
            dtype=dtype,
            src_ptr=K_new,
            dst_ptr=K_cache,
            src_row_base=in_row_base,
            dst_row_base=cache_row_base,
        )
        return

    ty = io_ir_type(dtype)
    elem_bytes = 2
    layout = spec.rotary.layout
    pair_count = spec.rotary.pair_count

    if layout == "half" and (H // 2) % _VEC == 0:
        # LLaMA-2/3/Qwen style: pair (i, i + H/2). The lo halves are
        # contiguous in [0, H/2), the hi halves are contiguous in
        # [H/2, H), so each side gets its own dwordx4 load + store.
        half = H // 2
        n_chunks = half // _VEC
        for c in range(n_chunks):
            d_lo = c * _VEC
            d_hi = half + c * _VEC
            lo_vec = b.global_load_vN(
                K_new,
                b.add(in_row_base, b.const_i32(d_lo)),
                ty,
                _VEC,
                align=_VEC * elem_bytes,
            )
            hi_vec = b.global_load_vN(
                K_new,
                b.add(in_row_base, b.const_i32(d_hi)),
                ty,
                _VEC,
                align=_VEC * elem_bytes,
            )
            out_lo_f32 = []
            out_hi_f32 = []
            for j in range(_VEC):
                pair = c * _VEC + j
                cos_v, sin_v = load_cos_sin(
                    b,
                    cos_table,
                    sin_table,
                    token_pos=dst_pos,
                    pair_idx=b.const_i32(pair),
                    spec=spec.rotary,
                )
                lo_f32 = b.cast_to_f32(b.vec_extract(lo_vec, j))
                hi_f32 = b.cast_to_f32(b.vec_extract(hi_vec, j))
                new_lo, new_hi = apply_rotary_pair_f32(b, lo_f32, hi_f32, cos_v, sin_v)
                out_lo_f32.append(new_lo)
                out_hi_f32.append(new_hi)
            lo_pack = pack_f32_to(b, out_lo_f32, dtype=dtype)
            hi_pack = pack_f32_to(b, out_hi_f32, dtype=dtype)
            b.global_store_vN(
                K_cache,
                b.add(cache_row_base, b.const_i32(d_lo)),
                lo_pack,
                _VEC,
                align=_VEC * elem_bytes,
            )
            b.global_store_vN(
                K_cache,
                b.add(cache_row_base, b.const_i32(d_hi)),
                hi_pack,
                _VEC,
                align=_VEC * elem_bytes,
            )
        return

    if layout == "interleaved" and H % _VEC == 0 and _VEC % 2 == 0:
        # GPT-J / LLaMA-1 style: pair (2i, 2i+1). Adjacent in the head
        # dim so ``_VEC = 8`` covers 4 pairs per chunk; we load the
        # chunk once, rotate each pair on f32, then pack + store the
        # rotated chunk (same dwordx4 store the no-rotary path uses).
        pairs_per_chunk = _VEC // 2
        n_chunks = H // _VEC
        for c in range(n_chunks):
            d = c * _VEC
            vec = b.global_load_vN(
                K_new,
                b.add(in_row_base, b.const_i32(d)),
                ty,
                _VEC,
                align=_VEC * elem_bytes,
            )
            out_f32 = []
            for j in range(pairs_per_chunk):
                pair = c * pairs_per_chunk + j
                cos_v, sin_v = load_cos_sin(
                    b,
                    cos_table,
                    sin_table,
                    token_pos=dst_pos,
                    pair_idx=b.const_i32(pair),
                    spec=spec.rotary,
                )
                lo_f32 = b.cast_to_f32(b.vec_extract(vec, 2 * j))
                hi_f32 = b.cast_to_f32(b.vec_extract(vec, 2 * j + 1))
                new_lo, new_hi = apply_rotary_pair_f32(b, lo_f32, hi_f32, cos_v, sin_v)
                out_f32.append(new_lo)
                out_f32.append(new_hi)
            packed = pack_f32_to(b, out_f32, dtype=dtype)
            b.global_store_vN(
                K_cache,
                b.add(cache_row_base, b.const_i32(d)),
                packed,
                _VEC,
                align=_VEC * elem_bytes,
            )
        return

    # Catch-all fallback: keep the original scalar pair loop so any
    # exotic head_size that doesn't align to _VEC still compiles.
    for pair in range(pair_count):
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
            b, K_cache, b.add(cache_row_base, b.const_i32(lo_d)), new_lo, dtype=dtype
        )
        store_scalar_from_f32(
            b, K_cache, b.add(cache_row_base, b.const_i32(hi_d)), new_hi, dtype=dtype
        )


def _appendkv_copy_v(
    b,
    *,
    H,
    dtype,
    V_new,
    V_cache,
    in_row_base,
    cache_row_base,
):
    """V cache copy is always a plain row copy (no rotary on V)."""
    _copy_row_vec(
        b,
        H=H,
        dtype=dtype,
        src_ptr=V_new,
        dst_ptr=V_cache,
        src_row_base=in_row_base,
        dst_row_base=cache_row_base,
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
