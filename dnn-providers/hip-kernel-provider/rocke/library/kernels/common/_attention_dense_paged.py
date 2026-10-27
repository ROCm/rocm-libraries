# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Paged K/V (block table) pieces shared by the gfx942 and gfx950 dense prefill kernels.

The cache is ``[num_kv_blocks, block_size, num_kv_heads, head_size]``; sequence
``s``'s logical page ``j`` lives at physical block ``block_tables[s][j]``. The
spec guarantees every wave's rows of a K/V tile fall in one page
(``block_size >= block_n // num_waves``, both powers of two), so one block-table
lookup per wave serves all of the wave's rows.

The emitters follow the gfx950 kernel's original op order, so either kernel can
call them without moving its IR.
"""

from __future__ import annotations

from rocke.core.ir import I32, PtrType


def declare_paged_params(b):
    """Declare ``block_tables``, ``kv_lens``, ``block_table_stride`` (in that ABI order)."""
    block_tables = b.param(
        "block_tables", PtrType(I32, "global"), noalias=True, readonly=True, align=4
    )
    kv_lens = b.param(
        "kv_lens", PtrType(I32, "global"), noalias=True, readonly=True, align=4
    )
    bt_stride = b.param("block_table_stride", I32)
    return block_tables, kv_lens, bt_stride


def paged_cache_bytes(spec) -> int:
    """Byte extent of one paged K (or V) cache: the buffer-resource bound.

    The whole cache, not one sequence's logical length: any page id the table can
    name must be in range, and loads past ``num_records`` silently return 0.
    Elements are 2 bytes: paged K/V is fp16/bf16 only, like the loaders' byte offsets.
    """
    return spec.num_kv_blocks * spec.block_size * spec.num_kv_heads * spec.head_size * 2


def paged_seq_setup(b, bt, kv_lens, bt_stride, block_size: int):
    """Return ``(seq_base, n_pages)``: the sequence's block-table row offset and its
    live page count ``ceil(kv_lens[bt] / block_size)``, which bounds every lookup."""
    seq_base = b.mul(bt, bt_stride)
    kv_len = b.global_load_i32(kv_lens, bt)
    n_pages = b.div(
        b.add(kv_len, b.const_i32(block_size - 1)),
        b.const_i32(block_size),
    )
    return seq_base, n_pages


def paged_wave_row_base(b, block_tables, seq_base, n_pages, wave_key0, block_size: int):
    """Physical row of the page holding key ``wave_key0`` (``block_id * block_size``).

    ``wave_key0`` is the wave's first key of the tile; it must be non-negative, so
    the page split is a shift and a mask. A page past ``n_pages`` reads block 0.
    The block id is not range-checked: the K/V buffer resource spans the whole cache
    (:func:`paged_cache_bytes`), so a bad id reads 0 instead of out of bounds. An
    i64 path that folds the block into a 64-bit base loses that and must check it.
    """
    page = b.lshr(wave_key0, b.const_i32(block_size.bit_length() - 1))
    block_id = b.masked_global_load(
        block_tables,
        b.add(seq_base, page),
        b.cmp_lt(page, n_pages),
        b.const_i32(0),
        dtype=I32,
        align=4,
    )
    return b.mul(block_id, b.const_i32(block_size))


def paged_kv_row(b, row_base, key, block_size: int):
    """Cache row of logical key ``key`` (non-negative) inside the page at ``row_base``."""
    return b.add(row_base, b.land(key, b.const_i32(block_size - 1)))


def validate_paged_launch(spec, k, v, block_tables, kv_lens, validate_paged: bool):
    """Host checks of a paged dense launch, before compile/launch.

    The cache shape must match the spec, because the buffer bound comes from the
    spec (:func:`paged_cache_bytes`): a smaller cache would let a table entry read
    past the allocation. With ``validate_paged`` the table contents are checked too
    (a device->host sync; pass False on the hot path): every block id the kernel
    dereferences must be in ``[0, num_kv_blocks)``, and ``kv_lens[i]`` must equal
    ``seqlen_kv``, since the kernel visits all ``seqlen_kv`` keys.
    """
    if spec.paged and (block_tables is None or kv_lens is None):
        raise ValueError("paged=True requires block_tables and kv_lens")
    if not spec.paged and (block_tables is not None or kv_lens is not None):
        raise ValueError("block_tables/kv_lens provided but spec.paged is False")
    if not spec.paged:
        return
    want = (spec.num_kv_blocks, spec.block_size, spec.num_kv_heads, spec.head_size)
    for name, t in (("k", k), ("v", v)):
        got = tuple(t.shape)
        if got != want:
            raise ValueError(
                f"paged {name} cache shape {got} != spec-derived "
                f"[num_kv_blocks, block_size, num_kv_heads, head_size]={want}; "
                "a mismatch mis-sizes the buffer-resource bound and can read OOB"
            )
    if not validate_paged:
        return
    kv_lens_host = kv_lens.tolist() if hasattr(kv_lens, "tolist") else list(kv_lens)
    for i in range(spec.batch):
        kv_len = int(kv_lens_host[i])
        if kv_len != spec.seqlen_kv:
            raise ValueError(
                f"paged kv_lens[{i}]={kv_len} != seqlen_kv={spec.seqlen_kv}; the "
                "kernel reads all seqlen_kv tiles, so a shorter kv_len leaves "
                "uncovered tiles reading page 0 -> silently wrong output"
            )
        n_pages = (kv_len + spec.block_size - 1) // spec.block_size
        used = block_tables[i][:n_pages]
        used = used.tolist() if hasattr(used, "tolist") else list(used)
        for block_id in used:
            if not 0 <= int(block_id) < spec.num_kv_blocks:
                raise ValueError(
                    f"paged block_tables[{i}] physical block id {int(block_id)} "
                    f"outside [0, num_kv_blocks={spec.num_kv_blocks}); a "
                    "malformed entry reads 0 via the bounds-checked cache "
                    "SRD -> silently wrong output"
                )


def paged_launch_args(block_tables, kv_lens) -> dict:
    """Kernel argument values of the paged params (:func:`declare_paged_params`)."""
    return {
        "block_tables": block_tables,
        "kv_lens": kv_lens,
        "block_table_stride": int(block_tables.stride(0)),
    }
