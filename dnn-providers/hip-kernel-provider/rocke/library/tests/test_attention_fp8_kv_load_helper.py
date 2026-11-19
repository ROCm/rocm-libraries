# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The shared fp8 KV sync-dequant tile loader (CPU only).

* ``fp8_kv_chunks_per_thread`` deals the tile evenly or refuses the shape at
  host time, before any IR exists;
* the emitted loop is one global load, one dequant and one LDS store per
  chunk, and never the ``tile.async_buffer_load_lds_addr`` DMA the bf16 KV
  loaders use -- that instruction has no register stop to dequant at, which
  is the whole reason this loader exists;
* the scale is applied as one explicit ``arith.fmul`` per element, never the
  fused ``cvt_scalef32`` (E8M0-only, truncates a non-power-of-two scale);
* ``kv_row_stride`` is invoked once per chunk rather than hoisted, which is
  what keeps two runtime-shape specs sharing a cache key from lowering to
  different IR;
* ``packed_group`` selects the row-group-packed K_lds layout and costs
  exactly one extra div/mod/mul/add per chunk.
"""

from __future__ import annotations

from collections import Counter

import pytest
from kernels.common._attention_fp8_kv_load import (
    FP8_KV_CHUNK_ELEMS,
    fp8_kv_chunks_per_thread,
    issue_fp8_kv_dequant_tile_load,
)
from rocke.core.ir import BF16, F32, FP8E4M3, I32, IRBuilder, PtrType

THREADS = 256


def _emit(*, block_n, head_size, threads=THREADS, packed_group=1, stride_calls=None):
    """Emit one tile load into a bare kernel; return the ops it added."""
    b = IRBuilder("k")
    ptr = b.param("k", PtrType(FP8E4M3, "global"))
    scale = b.param("k_scale", F32)
    kv_base = b.param("kv_base", I32)
    stride = b.param("kv_row_stride", I32)
    seqlen_kv = b.param("seqlen_kv", I32)
    tid = b.param("tid", I32)
    rows = block_n // packed_group
    lds = b.smem_alloc(BF16, [1, rows, head_size * packed_group])

    def _stride():
        if stride_calls is not None:
            stride_calls.append(stride)
        return stride

    n0 = len(b.kernel.body.ops)
    issue_fp8_kv_dequant_tile_load(
        b,
        ptr=ptr,
        lds=lds,
        buf_val=b.const_i32(0),
        tile_key0=b.const_i32(0),
        scale=scale,
        kv_base=kv_base,
        kv_row_stride=_stride,
        seqlen_kv=seqlen_kv,
        tid=tid,
        threads=threads,
        block_n=block_n,
        head_size=head_size,
        kv_dtype=FP8E4M3,
        dtype=BF16,
        packed_group=packed_group,
    )
    return Counter(o.name for o in b.kernel.body.ops[n0:])


# --- the host-side deal -----------------------------------------------------


@pytest.mark.parametrize(
    "block_n,head_size,threads,want",
    [
        (32, 64, 256, 1),  # exactly one chunk per thread
        (128, 64, 256, 4),
        (128, 128, 256, 8),
        (256, 128, 256, 16),
        (64, 64, 64, 8),  # a single wave
    ],
)
def test_chunks_per_thread_deals_the_whole_tile(block_n, head_size, threads, want):
    assert (
        fp8_kv_chunks_per_thread(block_n=block_n, head_size=head_size, threads=threads)
        == want
    )
    # the deal is exact: every element of the tile is owned once
    assert want * threads * FP8_KV_CHUNK_ELEMS == block_n * head_size


@pytest.mark.parametrize(
    "block_n,head_size,threads",
    [
        (100, 64, 256),  # 6400 % 2048 != 0
        (128, 65, 256),  # an odd head_size cannot align to 8-element chunks
        (16, 64, 256),  # fewer elements than one chunk per thread
        (128, 64, 100),  # a thread count that does not divide the tile
    ],
)
def test_chunks_per_thread_refuses_an_unservable_tile(block_n, head_size, threads):
    with pytest.raises(ValueError, match="must be divisible"):
        fp8_kv_chunks_per_thread(block_n=block_n, head_size=head_size, threads=threads)


def test_chunk_width_matches_the_dequant_helper():
    """The chunk is the width of ``dequant_fp8x8_to_dtype``: two x4 converts."""
    assert FP8_KV_CHUNK_ELEMS == 8


# --- the emitted loop -------------------------------------------------------


@pytest.mark.parametrize(
    "block_n,head_size,chunks", [(32, 64, 1), (128, 64, 4), (128, 128, 8)]
)
def test_one_load_dequant_store_per_chunk(block_n, head_size, chunks):
    ops = _emit(block_n=block_n, head_size=head_size)
    assert ops["memref.global_load_vN"] == chunks
    assert ops["tile.smem_store_vN"] == chunks
    # dequant_fp8x8_to_dtype is two cvt_pk_f32_fp8x4 per 8-element chunk
    assert ops["arith.cvt_pk_f32_fp8x4"] == 2 * chunks
    # the out-of-range key clamp: one compare and one select per chunk
    assert ops["arith.cmp"] == chunks
    assert ops["arith.select"] == chunks


def test_never_emits_the_async_dma():
    """The DMA has no register stop, so an fp8 dequant cannot ride it."""
    ops = _emit(block_n=128, head_size=64)
    assert ops["tile.async_buffer_load_lds_addr"] == 0


@pytest.mark.parametrize(
    "block_n,head_size,chunks", [(32, 64, 1), (128, 64, 4), (128, 128, 8)]
)
def test_scale_is_one_unfused_fmul_per_element(block_n, head_size, chunks):
    """Explicit fmul, never the fused cvt_scalef32 (E8M0-only, truncating).

    Deleting the scale multiply from ``dequant_fp8x8_to_dtype`` drops this
    count to zero and fails here.
    """
    ops = _emit(block_n=block_n, head_size=head_size)
    assert ops["arith.fmul"] == chunks * FP8_KV_CHUNK_ELEMS
    assert ops["arith.cast_f32_to"] == chunks * FP8_KV_CHUNK_ELEMS


@pytest.mark.parametrize("chunks,block_n", [(1, 32), (4, 128), (8, 256)])
def test_row_stride_is_called_once_per_chunk(chunks, block_n):
    """Invoked per chunk, never hoisted.

    Two runtime-shape specs differing only in head counts share a cache key
    and a symbol; a hoisted stride made them lower to different IR under that
    one key, and the launcher cache then served whichever compiled first to
    both.
    """
    calls = []
    _emit(block_n=block_n, head_size=64, stride_calls=calls)
    assert len(calls) == chunks


def test_packed_group_costs_one_extra_div_mod_mul_add_per_chunk():
    """The row-group-packed K_lds layout, against the natural [token, dim] one."""
    flat = _emit(block_n=128, head_size=64, packed_group=1)
    packed = _emit(block_n=128, head_size=64, packed_group=4)
    chunks = 4
    for op in ("arith.div", "arith.mod", "arith.mul", "arith.add"):
        assert packed[op] - flat[op] == chunks, op
    # the data movement itself is unchanged
    for op in (
        "memref.global_load_vN",
        "tile.smem_store_vN",
        "arith.cvt_pk_f32_fp8x4",
        "arith.fmul",
    ):
        assert packed[op] == flat[op], op


def test_emitter_refuses_an_unservable_tile_before_emitting():
    """The host-side deal gates the emitter, so no partial IR is left behind."""
    b = IRBuilder("k")
    args = dict(
        ptr=b.param("k", PtrType(FP8E4M3, "global")),
        lds=b.smem_alloc(BF16, [1, 100, 64]),
        buf_val=b.const_i32(0),
        tile_key0=b.const_i32(0),
        scale=b.param("s", F32),
        kv_base=b.param("kb", I32),
        seqlen_kv=b.param("sl", I32),
        tid=b.param("tid", I32),
    )
    stride = b.param("st", I32)
    n0 = len(b.kernel.body.ops)  # after every argument is built
    with pytest.raises(ValueError, match="must be divisible"):
        issue_fp8_kv_dequant_tile_load(
            b,
            kv_row_stride=lambda: stride,
            threads=THREADS,
            block_n=100,
            head_size=64,
            kv_dtype=FP8E4M3,
            dtype=BF16,
            **args,
        )
    assert len(b.kernel.body.ops) == n0
