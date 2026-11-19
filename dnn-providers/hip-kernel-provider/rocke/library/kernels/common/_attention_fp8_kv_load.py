# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The shared fp8 KV sync-dequant tile loader for the dense prefill kernels.

An fp8 KV cache cannot ride the `buffer_load ... lds` DMA path the bf16 KV
loaders use: that instruction moves HBM straight into LDS with no register
stop, and the dequant needs one. Every rocKE kernel that serves fp8 KV
therefore trades the DMA for the loop here -- a flat 8-element global load,
an in-register dequant, and an LDS store of the dequantised ``dtype`` values
into the *same* LDS layout the bf16 path fills. Downstream of the store
nothing in the kernel knows the KV cache was fp8, so the MFMA operand
readers, the LDS strides and the tile pipeline are all shared with bf16.

The tiled_3d kernels reached the same trade independently
(``_issue_fp8_dequant_loads``, in both ``kernels/gfx942/attention_tiled_3d.py``
and ``kernels/gfx950/attention_tiled_3d.py``, which switch their async/wide
loader off under fp8). They are deliberately left on their own copy, and not
only because converting them would move their goldens: they decompose the
chunk index differently (chunk -> ``HD//8`` row, then an 8-wide column, rather
than a flat element offset divided by ``head_size``) and they address KV
through a paged descriptor instead of a flat ``base + row * stride``. This
helper exposes no seam to inject that addressing, so absorbing them needs a
pluggable offset callback rather than a reordering. That pair duplicates each
other far more closely than either duplicates this, and is the better
consolidation target if one is taken up. See
:func:`issue_fp8_kv_dequant_tile_load` for why the emission order is fixed.

Encoding: the ``cvt_pk_f32_fp8x4`` chain this emits is identical IR on every
target, but the *hardware* decodes the byte per its native fp8 format --
e4m3fnuz on the ``gfx9_mfma`` family, OCP e4m3fn on gfx950. The caller is
responsible for having checked that the KV tensor's encoding matches the
target, which is what
``kernels.common.attention_dense_spec.require_fp8_kv_dtype`` does. See the G3
note in ``kernels.common.fmha_fwd_fp8``.
"""

from __future__ import annotations

from collections.abc import Callable

from rocke.core.ir import IRBuilder, Type, Value

from rocke.helpers.attention import dequant_fp8x8_to_dtype

__all__ = [
    "FP8_KV_CHUNK_ELEMS",
    "fp8_kv_chunks_per_thread",
    "issue_fp8_kv_dequant_tile_load",
]

#: Elements one thread loads, dequantises and stores per chunk. Fixed at the
#: width of :func:`dequant_fp8x8_to_dtype`, which is two ``cvt_pk_f32_fp8x4``.
FP8_KV_CHUNK_ELEMS = 8


def fp8_kv_chunks_per_thread(*, block_n: int, head_size: int, threads: int) -> int:
    """Chunks one thread owns for a ``block_n x head_size`` KV tile, or raise.

    Separated from the emitter so a spec-validation path can reject an
    unservable tile shape at host time, before any IR is built.
    """
    total = block_n * head_size
    stride = threads * FP8_KV_CHUNK_ELEMS
    if total % stride != 0:
        raise ValueError(
            f"fp8 KV sync loader: block_n*head_size={total} must be divisible "
            f"by threads*{FP8_KV_CHUNK_ELEMS}={stride} (block_n={block_n}, "
            f"head_size={head_size}, threads={threads}); a tail chunk would "
            f"need a bounds guard the loader does not emit"
        )
    return total // stride


def issue_fp8_kv_dequant_tile_load(
    b: IRBuilder,
    *,
    ptr: Value,
    lds: Value,
    buf_val: Value,
    tile_key0: Value,
    scale: Value,
    kv_base: Value,
    kv_row_stride: Callable[[], Value],
    seqlen_kv: Value,
    tid: Value,
    threads: int,
    block_n: int,
    head_size: int,
    kv_dtype: Type,
    dtype: Type,
    packed_group: int = 1,
) -> None:
    """Emit one KV tile's fp8 load -> unfused-scale dequant -> ``dtype`` LDS store.

    ``kv_row_stride`` is a *callable*, not a Value, and is invoked once per
    chunk. On the runtime-shape path the head counts are kernargs, so two
    specs differing only in head counts share a cache key and a symbol; baking
    a hoisted stride here made them lower to different IR under that one key
    and the launcher cache then served whichever binary compiled first to
    both. Calling per chunk also keeps the emission order identical to the
    inlined loaders this was extracted from.

    ``packed_group > 1`` selects the row-group-packed K_lds layout (``key``
    decomposes into a group row and a within-group column offset); the default
    is the natural ``[token, dim]`` layout both V and unpacked K use.

    The global load is *unbounded* -- a raw pointer load, not the
    bounds-checked buffer load the bf16 DMA path uses, so nothing returns 0
    past the end of the cache for free. Keys past ``seqlen_kv`` are therefore
    clamped to row 0, which is always inside the same (batch, kv head) slice
    and so always inside the allocation.

    Whether the clamp ever fires depends on the caller's tile loop. gfx950
    over-prefetches one tile past the last consumed KV tile and relies on this
    (that tile's data is never read). gfx942 bounds its loop by a floor divide,
    so no accepted spec addresses a key at or past ``seqlen_kv`` and the clamp
    is purely defensive there. Keep it on both: it is what makes the unbounded
    load safe to reuse, and it costs one compare and one select per chunk.

    ADDRESSING LIMIT: the index is a 32-bit ELEMENT offset, so this loader
    reads a KV cache of at most 2^31 elements, and past that the address wraps
    onto whatever it lands on rather than returning zeros the way an
    over-large i32 buffer voffset does against its ``num_records``. Both dense
    callers are already gated by ``check_dense_spec_preflight``, which refuses
    a cache at or past ``INT32_LIMIT`` *bytes* computed with
    ``spec.kv_elem_bytes``; at 1 byte per fp8 element that byte bound is
    exactly this element bound, so no separate fp8 threshold exists or should
    be added. A new caller must keep that property or bring its own gate.
    There is no i64 variant here yet: the paged kernels solve the same problem
    with ``use_i64_kv_addr`` in ``kernels.common.attention_unified``, folding
    the offset into a 64-bit buffer base.

    Emission order is load-bearing: this is the promotion of the loader the
    gfx950 dense kernel carried inline, and it reproduces that op sequence
    exactly so the extraction moves no golden. Reorder nothing here without
    re-blessing the dense goldens.
    """
    chunks = fp8_kv_chunks_per_thread(
        block_n=block_n, head_size=head_size, threads=threads
    )
    for it in range(chunks):
        e0 = b.mul(
            b.add(b.const_i32(it * threads), tid), b.const_i32(FP8_KV_CHUNK_ELEMS)
        )
        key = b.div(e0, b.const_i32(head_size))
        d0 = b.mod(e0, b.const_i32(head_size))
        kv_row = b.add(tile_key0, key)
        kv_row = b.select(b.cmp_lt(kv_row, seqlen_kv), kv_row, b.const_i32(0))
        addr = b.add(
            b.add(kv_base, b.mul(kv_row, kv_row_stride())),
            d0,
        )
        deq = dequant_fp8x8_to_dtype(
            b,
            b.global_load_vN(ptr, addr, kv_dtype, FP8_KV_CHUNK_ELEMS),
            scale,
            dtype,
        )
        if packed_group > 1:
            row = b.div(key, b.const_i32(packed_group))
            col = b.add(
                b.mul(b.mod(key, b.const_i32(packed_group)), b.const_i32(head_size)),
                d0,
            )
        else:
            row = key
            col = d0
        b.smem_store_vN(lds, [buf_val, row, col], deq, FP8_KV_CHUNK_ELEMS)
