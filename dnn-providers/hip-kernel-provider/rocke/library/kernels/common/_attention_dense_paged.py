# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Paged K/V (block table) pieces shared by the gfx942 and gfx950 dense prefill kernels.

The cache is ``[num_kv_blocks, block_size, num_kv_heads, head_size]``; sequence
``s``'s logical page ``j`` lives at physical block ``block_tables[s][j]``. The
spec's paged validation keeps every wave's rows of a K/V tile in one page, so one
block-table lookup per wave serves all of the wave's rows.
"""

from __future__ import annotations

from rocke.core.ir import I32, I64, PtrType


def declare_paged_params(b):
    """Declare ``block_tables``, ``kv_lens``, ``block_table_stride``, in ABI order."""
    block_tables = b.param(
        "block_tables", PtrType(I32, "global"), noalias=True, readonly=True, align=4
    )
    kv_lens = b.param(
        "kv_lens", PtrType(I32, "global"), noalias=True, readonly=True, align=4
    )
    bt_stride = b.param("block_table_stride", I32)
    return block_tables, kv_lens, bt_stride


_TORCH_CACHE_DTYPES = {"fp16": "torch.float16", "bf16": "torch.bfloat16"}


def paged_page_bytes(spec) -> int:
    """Bytes of one cache page (``block_size`` rows of all kv heads)."""
    return spec.block_size * spec.num_kv_heads * spec.head_size * spec.kv_elem_bytes


def paged_cache_bytes(spec) -> int:
    """Byte extent of one paged K (or V) cache: the i32 buffer-resource bound.

    The whole cache, not one sequence's logical length: any page id the table can
    name must be in range, and loads past ``num_records`` silently return 0. The
    spec's 2 GiB check uses it too.
    """
    return spec.num_kv_blocks * paged_page_bytes(spec)


def paged_tile_check(block_size: int, block_m: int, block_n: int) -> tuple[bool, str]:
    """Whether pages of ``block_size`` keys fit a ``block_m`` x ``block_n`` tile: whole
    pages per KV tile, and each wave's rows of a tile (one wave per 32 query rows)
    inside one page, so one lookup per wave serves its rows."""
    if block_n % block_size:
        return False, (
            f"block_n ({block_n}) must be a multiple of page block_size ({block_size})"
        )
    rows_per_wave = block_n // (block_m // 32)
    if block_size < rows_per_wave or block_size % rows_per_wave:
        return False, (
            f"paged block_size ({block_size}) must be >= and a multiple of "
            f"ROWS_PER_WAVE ({rows_per_wave})"
        )
    return True, ""


def paged_seq_setup(b, bt, kv_lens, bt_stride, block_size: int):
    """Return ``(seq_base, n_pages, kv_len)``: the sequence's block-table row offset,
    its live page count ``ceil(kv_lens[bt] / block_size)``, which bounds every
    lookup, and ``kv_lens[bt]`` itself."""
    seq_base = b.mul(bt, bt_stride)
    kv_len = b.global_load_i32(kv_lens, bt)
    n_pages = b.div(
        b.add(kv_len, b.const_i32(block_size - 1)),
        b.const_i32(block_size),
    )
    return seq_base, n_pages, kv_len


def paged_wave_block_id(b, block_tables, seq_base, n_pages, wave_key0, block_size: int):
    """Physical block id of the page holding key ``wave_key0``.

    ``wave_key0`` is the wave's first key of the tile; it must be non-negative, so
    the page split is a shift and a mask. A page past ``n_pages`` reads block 0.
    The id itself is not range-checked here: the i32 form bounds it by the
    whole-cache resource (:func:`paged_wave_row_base`), the i64 form clamps it
    (:func:`paged_page_rsrc`).
    """
    page = b.lshr(wave_key0, b.const_i32(block_size.bit_length() - 1))
    return b.masked_global_load(
        block_tables,
        b.add(seq_base, page),
        b.cmp_lt(page, n_pages),
        b.const_i32(0),
        dtype=I32,
        align=4,
    )


def paged_wave_row_base(b, block_tables, seq_base, n_pages, wave_key0, block_size: int):
    """Physical row of the page holding key ``wave_key0`` (``block_id * block_size``).

    The K/V buffer resource spans the whole cache (:func:`paged_cache_bytes`), so
    an id slightly past it reads 0, but a large one wraps in the i32 offset and can
    land on another page: only the launcher's ``validate_paged`` content check
    catches a corrupt table.
    """
    block_id = paged_wave_block_id(
        b, block_tables, seq_base, n_pages, wave_key0, block_size
    )
    return b.mul(block_id, b.const_i32(block_size))


def paged_page_bytes_ir(b, block_size: int, kv_row_bytes):
    """:func:`paged_page_bytes` as an IR value: ``kv_row_bytes`` is the byte stride of
    one cache row (all kv heads), a kernarg-derived value on the runtime-shape path."""
    return b.mul(kv_row_bytes, b.const_i32(block_size))


def paged_cache_bytes_ir(b, num_kv_blocks, page_bytes):
    """:func:`paged_cache_bytes` as an IR value, from the ``num_kv_blocks`` kernarg.

    Only for the i32 form: the spec admits it only while the whole cache fits the i32
    buffer range, so the product cannot wrap.
    """
    return b.mul(num_kv_blocks, page_bytes)


def paged_page_rsrc(b, base_ptr, block_id, num_kv_blocks, page_bytes):
    """Buffer resource over the one page ``block_id`` of the cache at ``base_ptr``
    (``use_i64_kv_addr``): the 64-bit page base goes into the resource, so the
    voffset stays within a page (:func:`paged_in_page_row`). ``num_kv_blocks`` and
    ``page_bytes`` are i32 IR values (:func:`paged_page_bytes_ir`).

    ``block_id`` must be wave-uniform (one page per wave and tile). Unlike the i32
    form, nothing bounds a bad id here: it would be an arbitrary 64-bit address. So
    it is clamped (unsigned, so a negative id clamps too) to the last block: a
    corrupt table reads a real page instead of faulting; ``validate_paged`` still
    reports it on the host. ``num_records`` is one page, never the whole cache: a
    range past the page's allocation could fault.
    """
    block_id = b.readfirstlane(block_id)
    block_id = b.select(
        b.cmp_ult(block_id, num_kv_blocks),
        block_id,
        b.sub(num_kv_blocks, b.const_i32(1)),
    )
    page_base = b.mul(b.zext(block_id, I64), b.zext(page_bytes, I64))
    return b.buffer_rsrc(b.global_ptr_add(base_ptr, page_base), page_bytes)


def paged_wave_page_rsrc(
    b, base_ptr, block_tables, seq_base, n_pages, wave_key0, block_size: int, bounds
):
    """Resource over the page holding key ``wave_key0`` (``use_i64_kv_addr``): one
    per wave and tile, shared by all of the wave's copies of the tile, rows then
    addressed within the page. ``bounds`` is ``(num_kv_blocks, page_bytes)`` as
    :func:`paged_page_rsrc` takes them."""
    block_id = paged_wave_block_id(
        b, block_tables, seq_base, n_pages, wave_key0, block_size
    )
    return paged_page_rsrc(b, base_ptr, block_id, *bounds)


def paged_in_page_row(b, key, block_size: int):
    """Row of logical key ``key`` (non-negative) within its page."""
    return b.land(key, b.const_i32(block_size - 1))


def paged_kv_row(b, row_base, key, block_size: int):
    """Cache row of logical key ``key`` (non-negative) in the page at ``row_base``."""
    return b.add(row_base, paged_in_page_row(b, key, block_size))


def paged_v_tail_offset(b, voff_bytes, key, kv_len):
    """V copy byte offset of logical key ``key``, out of range past ``kv_len``.

    Keys past the sequence's end read stale rows of its last page, or block 0 past
    its last page, which may hold anything. Their scores are masked, but P = 0 times
    a NaN V row is still NaN in P*V, so V must be zero there: an offset past the
    buffer bound makes the async copy write zeros into LDS. K needs no such offset,
    since the causal select replaces a stale key's score.
    """
    return b.select(b.cmp_lt(key, kv_len), voff_bytes, b.const_i32(0x7FFFFFFF))


def validate_paged_launch(
    spec,
    k,
    v,
    block_tables,
    kv_lens,
    validate_paged: bool,
    *,
    q=None,
    out=None,
    cu_seqlens_q=None,
):
    """Host checks of a paged dense launch, before compile/launch.

    Always (metadata only), as the unified binding checks its paged inputs:
    ``block_tables`` [batch, >= pages] and ``kv_lens`` [batch] are int32 tensors with
    unit inner stride on the K/V device, and the K/V caches are contiguous GPU tensors
    on one device with the spec's shape. The kernel bakes the cache layout and takes
    the cache extent ``num_kv_blocks`` from the spec (a kernel argument on the
    runtime-shape bodies): the i32 form bounds its loads by the whole cache
    (:func:`paged_cache_bytes`), the i64 form clamps the page id to ``num_kv_blocks``
    and bounds each page (:func:`paged_page_rsrc`). Either way a strided view or a
    smaller cache would be read at the wrong addresses or past the allocation. Paged varlen also needs
    packed contiguous ``q``/``out`` [total_q, Hq, D] and an int32 ``cu_seqlens_q``
    [batch + 1] there. With ``validate_paged`` the contents are checked too
    (device->host copies; pass False on the hot path): every block id the kernel
    dereferences must be in ``[0, num_kv_blocks)``. A single sequence needs
    ``kv_lens[0] == seqlen_kv``, since the kernel visits all ``seqlen_kv`` keys;
    paged varlen needs ``cu_seqlens_q`` to span the packed Q in order, and each
    sequence's lengths within the spec maxima (``q_len <= kv_len`` under
    ``causal_bottom_right``, where a negative shift would hide keys).
    Without ``validate_paged`` nothing checks the contents: the caller guarantees
    them, and a violation reads wrong pages without a fault.
    """
    if type(validate_paged) is not bool:
        raise ValueError(
            f"validate_paged must be a bool, got {type(validate_paged).__name__}"
        )
    if spec.paged and (block_tables is None or kv_lens is None):
        raise ValueError("paged=True requires block_tables and kv_lens")
    if not spec.paged and (block_tables is not None or kv_lens is not None):
        raise ValueError("block_tables/kv_lens provided but spec.paged is False")
    if not spec.paged:
        return
    if spec.varlen and cu_seqlens_q is None:
        raise ValueError("paged varlen requires cu_seqlens_q (int32 [batch+1])")
    n_pages = -(-spec.seqlen_kv // spec.block_size)
    bt_shape = tuple(block_tables.shape)
    if len(bt_shape) != 2 or bt_shape[0] != spec.batch or bt_shape[1] < n_pages:
        raise ValueError(
            f"paged block_tables shape {bt_shape} must be [batch={spec.batch}, "
            f">= {n_pages} pages] (the kernel reads that many entries)"
        )
    kv_lens_shape = tuple(kv_lens.shape)
    if kv_lens_shape != (spec.batch,):
        raise ValueError(
            f"paged kv_lens shape {kv_lens_shape} must be [batch={spec.batch}]"
        )
    metadata = [("block_tables", block_tables), ("kv_lens", kv_lens)]
    if spec.varlen:
        cu_shape = tuple(cu_seqlens_q.shape)
        if cu_shape != (spec.batch + 1,):
            raise ValueError(
                f"paged cu_seqlens_q shape {cu_shape} must be [batch+1={spec.batch + 1}]"
            )
        metadata.append(("cu_seqlens_q", cu_seqlens_q))
    for name, t in metadata:
        check_paged_metadata(name, t, k.device)
    if not bt_shape[1] <= block_tables.stride(0) <= 0x7FFF_FFFF:
        raise ValueError(
            "paged block_tables row stride must be non-overlapping and fit int32"
        )
    want = (spec.num_kv_blocks, spec.block_size, spec.num_kv_heads, spec.head_size)
    for name, t in (("k", k), ("v", v)):
        got = tuple(t.shape)
        if got != want:
            raise ValueError(
                f"paged {name} cache shape {got} != spec-derived "
                f"[num_kv_blocks, block_size, num_kv_heads, head_size]={want}; "
                "a mismatch mis-sizes the buffer-resource bound and can read OOB"
            )
        if not t.is_contiguous():
            raise ValueError(
                f"paged {name} cache must be contiguous: the kernel bakes its layout"
            )
        if not t.is_cuda or str(t.device) != str(k.device):
            raise ValueError(
                f"paged {name} cache must be a GPU tensor on the K device {k.device}, "
                f"got {t.device}"
            )
        want_dtype = _TORCH_CACHE_DTYPES[spec.dtype]
        if str(t.dtype) != want_dtype:
            raise ValueError(
                f"paged {name} cache dtype {t.dtype} must be {want_dtype} (spec dtype "
                f"{spec.dtype}): the buffer bound assumes its element size"
            )
    if spec.varlen:
        _check_packed_queries(spec, q, out, k.device)
    if not validate_paged:
        return
    # One copy per tensor, not per row.
    used_rows = block_tables[:, :n_pages].tolist()
    kv_values = [int(kv_len) for kv_len in kv_lens.tolist()]
    if spec.varlen:
        q_lens = _packed_query_lengths(spec, cu_seqlens_q.tolist(), int(q.shape[0]))
    for i, (kv_len, used) in enumerate(zip(kv_values, used_rows)):
        if not spec.varlen:
            if kv_len != spec.seqlen_kv:
                raise ValueError(
                    f"paged kv_lens[{i}]={kv_len} != seqlen_kv={spec.seqlen_kv}; the "
                    "kernel reads all seqlen_kv tiles, so a shorter kv_len leaves "
                    "uncovered tiles reading page 0 -> silently wrong output"
                )
            live_pages = n_pages
        else:
            if not 0 <= kv_len <= spec.seqlen_kv:
                raise ValueError(
                    f"paged kv_lens[{i}]={kv_len} outside [0, seqlen_kv="
                    f"{spec.seqlen_kv}]: the table holds only that many keys"
                )
            if spec.causal_bottom_right and q_lens[i] > kv_len:
                raise ValueError(
                    f"paged varlen sequence {i}: q_len={q_lens[i]} > "
                    f"kv_lens[{i}]={kv_len}; bottom-right causal needs "
                    "q_len <= kv_len per sequence"
                )
            # Columns past the sequence's own pages are never read: any value.
            live_pages = -(-kv_len // spec.block_size)
        for block_id in used[:live_pages]:
            if not 0 <= int(block_id) < spec.num_kv_blocks:
                raise ValueError(
                    f"paged block_tables[{i}] physical block id {int(block_id)} "
                    f"outside [0, num_kv_blocks={spec.num_kv_blocks}); a "
                    "malformed entry reads a wrong page -> silently wrong output"
                )


def check_paged_metadata(name: str, t, device) -> None:
    """Structural rules of one int32 metadata tensor; the dense binding also
    applies them to a ``cu_seqlens_q`` it drops before the runner."""
    if str(t.dtype) != "torch.int32":
        raise ValueError(f"paged {name} dtype must be int32, got {t.dtype}")
    if not t.is_cuda:
        raise ValueError(f"paged {name} must be a GPU tensor")
    if str(t.device) != str(device):
        raise ValueError(
            f"paged {name} must be on the K/V device {device}, got {t.device}"
        )
    if t.stride(-1) != 1:
        raise ValueError(f"paged {name} innermost stride must be 1")


def _check_packed_queries(spec, q, out, device) -> None:
    """Packed Q/out of a paged varlen launch: the kernel addresses both as
    contiguous [total_q, Hq, D] and bounds Q reads by ``cu_seqlens_q[batch]``."""
    if q is None or out is None:
        raise ValueError("paged varlen validation needs q and out")
    q_shape = tuple(q.shape)
    if (
        len(q_shape) != 3
        or q_shape[1:] != (spec.num_query_heads, spec.head_size)
        or q_shape[0] > spec.batch * spec.seqlen_q
    ):
        raise ValueError(
            f"paged varlen q shape {q_shape} must be packed [total_q <= batch*"
            f"seqlen_q={spec.batch * spec.seqlen_q}, {spec.num_query_heads}, "
            f"{spec.head_size}]"
        )
    if tuple(out.shape) != q_shape:
        raise ValueError(
            f"paged varlen out shape {tuple(out.shape)} must equal q shape {q_shape}"
        )
    for name, t in (("q", q), ("out", out)):
        if not t.is_contiguous():
            raise ValueError(f"paged varlen {name} must be contiguous")
        if str(t.device) != str(device):
            raise ValueError(
                f"paged varlen {name} must be on the K/V device {device}, "
                f"got {t.device}"
            )


def _packed_query_lengths(spec, offsets, total_q: int) -> list:
    """Per-sequence query lengths of ``cu_seqlens_q``; rejects offsets that do not
    span the packed Q in order or exceed ``seqlen_q``."""
    offsets = [int(x) for x in offsets]
    if offsets[0] != 0 or offsets[-1] != total_q:
        raise ValueError(
            f"paged cu_seqlens_q must start at 0 and end at total_q={total_q}, "
            f"got {offsets[0]} .. {offsets[-1]}"
        )
    lengths = [stop - start for start, stop in zip(offsets, offsets[1:])]
    for i, length in enumerate(lengths):
        if not 0 <= length <= spec.seqlen_q:
            raise ValueError(
                f"paged cu_seqlens_q sequence {i} length {length} outside "
                f"[0, seqlen_q={spec.seqlen_q}] (offsets must be monotone)"
            )
    return lengths


def paged_launch_args(block_tables, kv_lens) -> dict:
    """Kernel argument values of the paged params (:func:`declare_paged_params`)."""
    return {
        "block_tables": block_tables,
        "kv_lens": kv_lens,
        "block_table_stride": int(block_tables.stride(0)),
    }
