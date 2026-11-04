# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared prep / convert kernels of the attention backward (all families).

Python-only family glue (no C++ twin; packs pin ``ROCKE_BACKEND=python``).
One spec, :class:`AttnBwdAuxSpec`, drives four builders:

* ``build_attn_bwd_prep``: per query row ``lse2 = lse * LOG2E`` and
  ``Dsum = rowsum(dO * O)`` (fp32), both gated on one
  ``row_live = q < len_q && lse > -inf && lse < +inf`` (ordered compares, so a
  NaN LSE is dead): dead rows get ``+inf`` / ``0``.  It zeroes the dQ
  workspace rows (``zero_dq``) and, in the contiguous kv role
  (``zero_kv``), the dK / dV workspace rows.  Batched: every row
  ``< S_q_max`` (``< S_kv_max`` for kv) is initialised; THD: rows inside the
  segment only.  User tensors are read only in CTAs with a live row.
* ``build_attn_bwd_convert``: ``DST = cast(mult * WS)`` at arbitrary element
  strides.  Batched rows ``[len, S_max)`` are written as exact zeros; THD rows
  outside the segment are not touched.
* ``build_attn_bwd_reduce`` (``dkv_sink = "slices"``): the convert of the sum
  of ``n_slices`` workspace slices ``s_stride`` floats apart, added in slice
  order (deterministic); the dK / dV slices need no zero fill, so the kv-zero
  role of prep is off with it.
* ``build_attn_bwd_page_zero`` / ``build_attn_bwd_page_convert``
  (``kv_target = "paged"``): walk ``j < ceil(len_kv / page)`` logical blocks
  of every sequence, read ``block_table[s][j]`` once per CTA, and zero /
  convert the fp32 page workspace rows of that physical page (slot
  ``page_id * page + t``).  Shared physical pages are visited once per
  referencing sequence; the duplicate writes are identical.  A block-table
  entry outside ``[0, page_rows / page)`` (device data the host check
  rejects) is never dereferenced: no workspace row and no gradient-cache row
  of it is written, and every address uses block 0 instead.

Workspace addressing (``[h][rows][X]`` head-major or ``[rows][h][X]``
token-major, ``X = D`` or 1) goes through ``_attention_bwd_addr``: the
per-(head, base row) offset is an i64 pointer rebase, the in-slab index is
i32, and rows at or past ``ws_rows`` are never written (reads give 0).
Token-major addressing needs ``S_max * H * D < 2**31`` for its in-slab index
(head-major: ``S_max * D``); the host plan checks it.

The work-list role of prep (THD, compile-time ``worklist``, runtime
``use_worklist``): CTA ``(0, 0, 0)``, after its q-side rows, writes the live
``(sequence, kv tile)`` items of the batch heaviest first into ``WORKLIST``
(layout and order: ``_attention_bwd_worklist``). Each thread owns sequences
``tid, tid + T, ...``; per owned sequence it sums the live kv tiles
(``ceil(len_kv / wl_kn0)``) of every sequence that sorts before it, reading
the (query length, tile count) pairs of ``T`` sequences per pass from LDS,
then stores its sequence's items from that start. ``n_batch`` sequences, no
host read; ``ceil(n_batch / T)^2`` passes of ``T`` LDS pairs. Without the
switch no role IR is emitted and ``WORKLIST``, ``use_worklist``, ``n_batch``
and ``wl_kn0`` are not read; with it and ``use_worklist = 0`` the role does
not run.

Kernel arguments follow the ``rocke.attn_bwd.v3`` (prep, convert) and
``rocke.attn_bwd_unified.v1`` (page roles) lists exactly; see
``attn_bwd_aux_params``.  No pointer is ``noalias``; inputs are ``readonly``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from rocke.core.arch import ArchTarget
from rocke.core.ir import BF16, F16, F32, I32, I64, IRBuilder, KernelDef, PtrType, Value
from rocke.helpers.attention_fwd_ext import LOG2E

from ._attention_bwd_addr import (
    PagedCacheAddr,
    StridedTensorAddr,
    WorkspaceAddr,
    block_table_entry,
    decode_batched,
    decode_thd,
)
from ._attention_bwd_caps import BWD_BARRIER_FENCE
from ._attention_bwd_worklist import (
    worklist_sorts_before,
    worklist_store_count,
    worklist_store_item,
    worklist_tiles,
)

__all__ = [
    "ATTN_BWD_ABI",
    "ATTN_BWD_UNIFIED_ABI",
    "AUX_STAGES",
    "CONVERT_VECS",
    "AttnBwdAuxSpec",
    "attn_bwd_aux_block_threads",
    "attn_bwd_aux_params",
    "attn_bwd_aux_rows_per_cta",
    "attn_bwd_convert_grid",
    "attn_bwd_page_grid",
    "attn_bwd_prep_grid",
    "build_attn_bwd_convert",
    "build_attn_bwd_page_convert",
    "build_attn_bwd_page_zero",
    "build_attn_bwd_prep",
    "build_attn_bwd_reduce",
]

ATTN_BWD_ABI = "rocke.attn_bwd.v3"
ATTN_BWD_UNIFIED_ABI = "rocke.attn_bwd_unified.v1"
AUX_STAGES = ("prep", "convert", "page_zero", "page_convert", "reduce")

_HEAD_SIZES = (32, 64, 128)
_DTYPES = {"fp16": F16, "bf16": BF16}
_SEQ_MODES = ("batched", "thd")
_WS_LAYOUTS = ("head_major", "token_major")
_KV_TARGETS = ("contiguous", "paged")
_MAX_BLOCK = 1024
# Elements per lane of the convert kernel: 8 (one 16-byte output store, two
# 16-byte workspace loads), 4 (half; twice the lanes per row) or 16 (two of
# each; half the lanes per row). The other roles move 8 elements per lane.
CONVERT_VECS = (8, 4, 16)
# The lowering's flat-work-group-size attribute has a fixed minimum of 64.
_MIN_WG_ATTR = 64


@dataclass(frozen=True)
class AttnBwdAuxSpec:
    """Compile-time key of the prep / convert / page-role kernels."""

    head_size: int
    dtype: str = "fp16"
    seq_mode: str = "batched"
    stage_vec: int = 8
    ws_layout: str = "head_major"
    kv_target: str = "contiguous"
    rows_per_cta: int | None = None
    convert_vec: int = 8  # elements per lane of the convert (CONVERT_VECS)
    # prep (THD, contiguous kv): also emit the work-list role (runtime
    # use_worklist); False emits none of it
    worklist: bool = False
    # user-tensor access width (elements) of a stage_vec = 1 kernel (1, 2, 4)
    narrow_vec: int = 1
    name: str = "rocke_attn_bwd"

    def __post_init__(self) -> None:
        if self.head_size not in _HEAD_SIZES:
            raise ValueError(f"head_size must be one of {_HEAD_SIZES}")
        if self.dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {tuple(_DTYPES)}")
        if self.seq_mode not in _SEQ_MODES:
            raise ValueError(f"seq_mode must be one of {_SEQ_MODES}")
        if self.stage_vec not in (8, 1):
            raise ValueError("stage_vec must be 8 or 1")
        if self.ws_layout not in _WS_LAYOUTS:
            raise ValueError(f"ws_layout must be one of {_WS_LAYOUTS}")
        if self.kv_target not in _KV_TARGETS:
            raise ValueError(f"kv_target must be one of {_KV_TARGETS}")
        if self.rows_per_cta is not None and self.rows_per_cta < 1:
            raise ValueError("rows_per_cta must be positive")
        if self.convert_vec not in CONVERT_VECS:
            raise ValueError(f"convert_vec must be one of {CONVERT_VECS}")
        if not isinstance(self.worklist, bool):
            raise ValueError(f"worklist must be a bool, got {self.worklist!r}")
        if self.worklist and (self.seq_mode != "thd" or self.kv_target != "contiguous"):
            raise ValueError(
                "worklist=True needs seq_mode='thd' and kv_target='contiguous'"
            )
        if self.narrow_vec not in (1, 2, 4) or (
            self.narrow_vec != 1 and self.stage_vec != 1
        ):
            raise ValueError("narrow_vec must be 1, 2 or 4 (not 1: stage_vec = 1 only)")

    def kernel_name(self, stage: str) -> str:
        """Kernel symbol; every non-default compile-time field salts the name."""
        if stage not in AUX_STAGES:
            raise ValueError(f"stage must be one of {AUX_STAGES}")
        parts = [self.name, stage, self.dtype, f"d{self.head_size}"]
        if stage in ("prep", "convert", "reduce"):
            parts.append(self.seq_mode)
        if self.stage_vec != 8:
            parts.append(f"sv{self.stage_vec}")
        if self.ws_layout != "head_major":
            parts.append("tm")
        if stage == "prep" and self.kv_target != "contiguous":
            parts.append("paged")
        if self.rows_per_cta is not None:
            parts.append(f"r{self.rows_per_cta}")
        if self.convert_vec != 8:
            parts.append(f"v{self.convert_vec}")
        if self.worklist:
            parts.append("wl")
        if self.narrow_vec != 1:
            parts.append(f"nv{self.narrow_vec}")
        return "_".join(parts)


# ---------------------------------------------------------------------------
# Launch geometry (host)
# ---------------------------------------------------------------------------


def attn_bwd_aux_rows_per_cta(spec: AttnBwdAuxSpec, *, arch: str) -> int:
    """Rows per CTA: ``None`` resolves to ``max(wave, 64) / (D / v)``.

    One row is ``D / v`` lanes (``v = convert_vec`` elements per lane; 8 for
    every role but a non-default convert).  An explicit value must fill whole
    waves and keep the block at most 1024 threads.
    """
    wave = ArchTarget.from_gfx(arch).wave_size
    lanes = _lanes_per_row(spec)
    if spec.rows_per_cta is None:
        return max(wave, _MIN_WG_ATTR) // lanes
    threads = spec.rows_per_cta * lanes
    if threads % wave or threads > _MAX_BLOCK:
        raise ValueError(
            f"rows_per_cta={spec.rows_per_cta} gives {threads} threads; need a "
            f"multiple of the wave size {wave} and at most {_MAX_BLOCK}"
        )
    return spec.rows_per_cta


def attn_bwd_aux_block_threads(spec: AttnBwdAuxSpec, *, arch: str) -> int:
    return attn_bwd_aux_rows_per_cta(spec, arch=arch) * _lanes_per_row(spec)


def _lanes_per_row(spec: AttnBwdAuxSpec) -> int:
    return spec.head_size // spec.convert_vec


def _ceil(a: int, b: int) -> int:
    return -(-a // b)


def attn_bwd_prep_grid(
    spec: AttnBwdAuxSpec,
    *,
    arch: str,
    s_q_max: int,
    s_kv_max: int,
    h_q: int,
    h_k: int,
    h_v: int,
    batch: int,
    zero_kv: bool,
) -> tuple[int, int, int]:
    rows = attn_bwd_aux_rows_per_cta(spec, arch=arch)
    if zero_kv and spec.kv_target != "contiguous":
        raise ValueError("the kv-zero role exists only for kv_target='contiguous'")
    y = h_q + (h_k + h_v if zero_kv else 0)
    return (_ceil(max(s_q_max, s_kv_max, 1), rows), y, batch)


def attn_bwd_convert_grid(
    spec: AttnBwdAuxSpec, *, arch: str, s_max: int, heads: int, batch: int
) -> tuple[int, int, int]:
    rows = attn_bwd_aux_rows_per_cta(spec, arch=arch)
    return (_ceil(max(s_max, 1), rows), heads, batch)


def attn_bwd_page_grid(
    spec: AttnBwdAuxSpec,
    *,
    arch: str,
    max_blocks_per_seq: int,
    page: int,
    h_kv: int,
    num_seqs: int,
) -> tuple[int, int, int]:
    rows = attn_bwd_aux_rows_per_cta(spec, arch=arch)
    if page % rows:
        raise ValueError(f"page {page} is not a multiple of rows_per_cta {rows}")
    return (_ceil(max(max_blocks_per_seq * page, 1), rows), h_kv, num_seqs)


# ---------------------------------------------------------------------------
# Kernel-argument lists
# ---------------------------------------------------------------------------

_PREP_PTRS = (
    "O", "dO", "LSE", "SEQ_Q", "SEQ_KV", "OFF_Q", "OFF_KV",
    "WS_LSE2", "WS_DSUM", "WS_DQ", "WS_DK", "WS_DV", "WORKLIST",
)  # fmt: skip
# Batch / head strides are i64 kernargs (a tensor may span more than 2^31
# elements between batches or heads); token strides stay i32.
_PREP_I64 = ("o_b", "o_h", "do_b", "do_h", "l_b", "l_h")
_PREP_I32 = (
    "o_t", "do_t", "l_t",
    "h_q", "h_k", "h_v", "S_q_max", "S_kv_max", "has_len", "len_stride",
    "off64", "q_mult", "q_div", "kv_mult", "kv_div", "ws_rows_q", "ws_rows_kv",
    "ws_seg", "zero_kv", "zero_dq", "use_worklist", "n_batch", "wl_kn0",
)  # fmt: skip
_CONVERT_PTRS = ("SRC", "DST", "SEQ", "OFF")
_CONVERT_I64 = ("d_b", "d_h")
_CONVERT_I32 = (
    "H", "d_t", "S_max", "has_len", "len_stride", "off64",
    "tok_mult", "tok_div", "ws_rows", "ws_seg",
)  # fmt: skip
# The slice reduce: the convert's arguments plus the slice stride (floats,
# i64) and the slice count.
_REDUCE_I64 = _CONVERT_I64 + ("s_stride",)
_REDUCE_I32 = _CONVERT_I32 + ("n_slices",)
_PAGE_ZERO_PTRS = ("SEQ_LENS_KV", "BLOCK_TABLE", "WS_PDK", "WS_PDV")
_PAGE_I32 = ("h_kv", "num_seqs", "max_blocks_per_seq", "bt_stride", "page", "page_rows")
_PAGE_CONVERT_PTRS = ("SRC", "SEQ_LENS_KV", "BLOCK_TABLE", "DST")
_PAGE_CONVERT_I32 = _PAGE_I32 + ("d_blk", "d_tok", "d_h")

_READONLY = {
    "prep": {"O", "dO", "LSE", "SEQ_Q", "SEQ_KV", "OFF_Q", "OFF_KV"},
    "convert": {"SRC", "SEQ", "OFF"},
    "reduce": {"SRC", "SEQ", "OFF"},
    "page_zero": {"SEQ_LENS_KV", "BLOCK_TABLE"},
    "page_convert": {"SRC", "SEQ_LENS_KV", "BLOCK_TABLE"},
}


def attn_bwd_aux_params(
    stage: str, spec: AttnBwdAuxSpec
) -> tuple[tuple[str, str], ...]:
    """``(name, struct format)`` in kernel-argument order.

    ``"Q"`` = 64-bit pointer, ``"q"`` = i64, ``"f"`` = f32, ``"i"`` = i32
    (pointers first, so the i64 block is 8-byte aligned without padding).
    """
    if stage == "prep":
        lists = (_PREP_PTRS, _PREP_I64, (), _PREP_I32)
    elif stage == "convert":
        lists = (_CONVERT_PTRS, _CONVERT_I64, ("mult",), _CONVERT_I32)
    elif stage == "reduce":
        lists = (_CONVERT_PTRS, _REDUCE_I64, ("mult",), _REDUCE_I32)
    elif stage == "page_zero":
        lists = (_PAGE_ZERO_PTRS, (), (), _PAGE_I32)
    elif stage == "page_convert":
        lists = (_PAGE_CONVERT_PTRS, (), ("mult",), _PAGE_CONVERT_I32)
    else:
        raise ValueError(f"stage must be one of {AUX_STAGES}")
    ptrs, i64s, f32s, i32s = lists
    return (
        tuple((n, "Q") for n in ptrs)
        + tuple((n, "q") for n in i64s)
        + tuple((n, "f") for n in f32s)
        + tuple((n, "i") for n in i32s)
    )


def _ptr_type(name: str, io) -> PtrType:
    if name in ("O", "dO", "DST"):
        return PtrType(io, "global")
    if name in ("SEQ_Q", "SEQ_KV", "OFF_Q", "OFF_KV", "SEQ", "OFF", "WORKLIST"):
        return PtrType(I32, "global")
    if name in ("SEQ_LENS_KV", "BLOCK_TABLE"):
        return PtrType(I32, "global")
    return PtrType(F32, "global")


def _declare(b: IRBuilder, stage: str, spec: AttnBwdAuxSpec) -> dict[str, Value]:
    io = _DTYPES[spec.dtype]
    args: dict[str, Value] = {}
    for name, fmt in attn_bwd_aux_params(stage, spec):
        if fmt == "Q":
            attrs = {"readonly": True} if name in _READONLY[stage] else {}
            args[name] = b.param(name, _ptr_type(name, io), **attrs)
        elif fmt == "q":
            args[name] = b.param(name, I64)
        elif fmt == "f":
            args[name] = b.param(name, F32)
        else:
            args[name] = b.param(name, I32)
    return args


# ---------------------------------------------------------------------------
# Emission helpers
# ---------------------------------------------------------------------------


class _Lanes:
    """Row / segment decomposition of a CTA: ``D / v`` lanes per row
    (``v = convert_vec`` elements per lane)."""

    def __init__(self, b: IRBuilder, spec: AttnBwdAuxSpec, rows: int) -> None:
        vec = spec.convert_vec
        self.lanes = spec.head_size // vec
        tid = b.thread_id_x()
        self.row = b.div(tid, b.const_i32(self.lanes))  # row inside the CTA
        self.seg = b.mod(tid, b.const_i32(self.lanes))  # v-element segment
        self.col0 = b.mul(self.seg, b.const_i32(vec))
        self.tile_row0 = b.mul(b.block_id_x(), b.const_i32(rows))
        self.r = b.add(self.tile_row0, self.row)


def _zero_vec8(b: IRBuilder) -> Value:
    z = b.const_f32(0.0)
    return b.vec_pack([z] * 8, F32)


def _seq(b: IRBuilder, spec: AttnBwdAuxSpec, batch, *, s_max, seq_ptr, off_ptr,
         a: dict, mult: str, div: str):  # fmt: skip
    if spec.seq_mode == "thd":
        return decode_thd(
            b, batch, s_max=s_max, off_ptr=off_ptr, off64=a["off64"],
            mult=a[mult], div=a[div], seq_ptr=seq_ptr, has_len=a["has_len"],
            len_stride=a["len_stride"], ws_seg=a["ws_seg"],
        )  # fmt: skip
    return decode_batched(
        b, batch, s_max=s_max, seq_ptr=seq_ptr, has_len=a["has_len"],
        len_stride=a["len_stride"],
    )  # fmt: skip


def _finish(b: IRBuilder, spec: AttnBwdAuxSpec, arch: str) -> KernelDef:
    b.ret()
    threads = attn_bwd_aux_block_threads(spec, arch=arch)
    b.kernel.attrs["max_workgroup_size"] = max(threads, _MIN_WG_ATTR)
    return b.kernel


def _check_vec8(spec: AttnBwdAuxSpec, stage: str) -> None:
    if spec.convert_vec != 8:
        raise ValueError(
            f"convert_vec={spec.convert_vec} applies to the convert only, not {stage}"
        )


def _check_arch(arch: str) -> None:
    try:
        ArchTarget.from_gfx(arch)
    except KeyError:
        raise ValueError(f"unknown arch {arch!r}") from None


# ---------------------------------------------------------------------------
# prep
# ---------------------------------------------------------------------------


def _prep_q_role(b, spec, a, ln: _Lanes, h, z, io) -> None:
    D = spec.head_size
    seq = _seq(
        b, spec, z, s_max=a["S_q_max"], seq_ptr=a["SEQ_Q"], off_ptr=a["OFF_Q"],
        a=a, mult="q_mult", div="q_div",
    )  # fmt: skip
    layout = spec.ws_layout
    nh = a["h_q"] if layout == "token_major" else None

    def ws(ptr, width):
        return WorkspaceAddr(
            b, ptr, ws_rows=a["ws_rows_q"], width=width, row0=seq.ws_row0,
            layout=layout, n_heads=nh,
        ).slab(h, ln.tile_row0)  # fmt: skip

    lse2_ws = ws(a["WS_LSE2"], 1)
    dsum_ws = ws(a["WS_DSUM"], 1)
    rr = ln.row
    in_role = b.cmp_lt(ln.r, seq.s_max if spec.seq_mode == "batched" else seq.length)
    leader = b.cmp_eq(ln.seg, b.const_i32(0))
    store_ok = b.land(b.land(leader, in_role), lse2_ws.row_ok(rr))
    live_cta = b.cmp_lt(ln.tile_row0, seq.length)
    inf = b.const_f32(float("inf"))
    zero = b.const_f32(0.0)

    def live_body():
        def tensor(ptr, prefix, dtype):
            strides = (a[f"{prefix}_b"], a[f"{prefix}_h"], a[f"{prefix}_t"])
            return StridedTensorAddr(
                b,
                ptr,
                dtype=dtype,
                strides=strides,
                seq=seq,
                stage_vec=spec.stage_vec,
                narrow_vec=spec.narrow_vec,
            ).slab(h, ln.tile_row0)

        o = tensor(a["O"], "o", io).load_vec(ln.r, ln.col0)
        do = tensor(a["dO"], "do", io).load_vec(ln.r, ln.col0)
        acc = zero
        for i in range(8):
            acc = b.fma(
                b.cast_to_f32(b.vec_extract(o, i)),
                b.cast_to_f32(b.vec_extract(do, i)),
                acc,
            )
        m = ln.lanes // 2
        while m >= 1:
            acc = b.fadd(acc, b.warp_shuffle_xor(acc, m))
            m //= 2
        lse = tensor(a["LSE"], "l", F32).load(ln.r)
        row_live = b.land(
            b.cmp_lt(ln.r, seq.length),
            b.land(
                b.fcmp("ogt", lse, b.const_f32(float("-inf"))),
                b.fcmp("olt", lse, inf),
            ),
        )
        lse2 = b.select(row_live, b.fmul(lse, b.const_f32(LOG2E)), inf)
        dsum = b.select(row_live, acc, zero)
        with b.scf_if(store_ok):
            lse2_ws.store(rr, 0, lse2)
            dsum_ws.store(rr, 0, dsum)

    if spec.seq_mode == "batched":
        with b.scf_if_else(live_cta) as (then_ctx, else_ctx):
            with then_ctx:
                live_body()
            with else_ctx, b.scf_if(store_ok):
                lse2_ws.store(rr, 0, inf)
                dsum_ws.store(rr, 0, zero)
    else:
        with b.scf_if(live_cta):
            live_body()

    dq_ws = ws(a["WS_DQ"], D)
    zero_dq = b.cmp_ne(a["zero_dq"], b.const_i32(0))
    with b.scf_if(b.land(zero_dq, b.land(in_role, dq_ws.row_ok(rr)))):
        dq_ws.store_vec(rr, ln.col0, _zero_vec8(b))


def _prep_kv_role(b, spec, a, ln: _Lanes, y, z) -> None:
    D = spec.head_size
    seq = _seq(
        b, spec, z, s_max=a["S_kv_max"], seq_ptr=a["SEQ_KV"], off_ptr=a["OFF_KV"],
        a=a, mult="kv_mult", div="kv_div",
    )  # fmt: skip
    y2 = b.sub(y, a["h_q"])
    is_dk = b.cmp_lt(y2, a["h_k"])
    head = b.select(is_dk, y2, b.sub(y2, a["h_k"]))
    ptr = b.select(is_dk, a["WS_DK"], a["WS_DV"])
    nh = None
    if spec.ws_layout == "token_major":
        nh = b.select(is_dk, a["h_k"], a["h_v"])
    slab = WorkspaceAddr(
        b, ptr, ws_rows=a["ws_rows_kv"], width=D, row0=seq.ws_row0,
        layout=spec.ws_layout, n_heads=nh,
    ).slab(head, ln.tile_row0)  # fmt: skip
    bound = seq.s_max if spec.seq_mode == "batched" else seq.length
    ok = b.land(b.cmp_lt(ln.r, bound), slab.row_ok(ln.row))
    ok = b.land(ok, b.cmp_ne(a["zero_kv"], b.const_i32(0)))
    ok = b.land(ok, b.cmp_lt(y2, b.add(a["h_k"], a["h_v"])))
    with b.scf_if(ok):
        slab.store_vec(ln.row, ln.col0, _zero_vec8(b))


def _wl_weight(b, a, s):
    """(query length, live kv tiles) of THD sequence ``s``, decoded exactly as
    the main kernel decodes its lengths (the workspace base is not used)."""
    lens = []
    for side in ("q", "kv"):
        smax, off, seq = {
            "q": ("S_q_max", "OFF_Q", "SEQ_Q"),
            "kv": ("S_kv_max", "OFF_KV", "SEQ_KV"),
        }[side]
        info = decode_thd(
            b, s, s_max=a[smax], off_ptr=a[off], off64=a["off64"],
            mult=a[f"{side}_mult"], div=a[f"{side}_div"], seq_ptr=a[seq],
            has_len=a["has_len"], len_stride=a["len_stride"],
        )  # fmt: skip
        lens.append(info.length)
    return lens[0], worklist_tiles(b, lens[1], a["wl_kn0"])


def _worklist_role(b, a, threads: int) -> None:
    """Write the work list (see the module docstring); every thread of the CTA
    runs it (CTA-uniform trip counts around the barriers)."""
    zero, one = b.const_i32(0), b.const_i32(1)
    tc = b.const_i32(threads)
    tid = b.thread_id_x()
    nb = a["n_batch"]
    lds = b.smem_alloc(I32, [2 * threads], name_hint="wl_lds")
    n_pass = b.div(b.add(nb, b.const_i32(threads - 1)), tc)

    def lds_at(i):
        return b.vec_extract(b.smem_load_vN(lds, i, dtype=I32, n=1), 0)

    outer = b.scf_for_iter(
        zero, n_pass, one, [("wl_count", zero)], iv_name="wl_sb",
        elide_trailing_barrier=False,
    )  # fmt: skip
    with outer as (sb, _cur):
        s = b.add(b.mul(sb, tc), tid)
        s_ok = b.cmp_lt(s, nb)
        key_s, nt_s = _wl_weight(b, a, b.select(s_ok, s, zero))
        chunks = b.scf_for_iter(
            zero, n_pass, one, [("wl_start", zero), ("wl_total", zero)],
            iv_name="wl_c", elide_trailing_barrier=False,
        )  # fmt: skip
        with chunks as (c, (start, total)):
            c0 = b.mul(c, tc)
            o = b.add(c0, tid)
            o_ok = b.cmp_lt(o, nb)
            key_o, nt_o = _wl_weight(b, a, b.select(o_ok, o, zero))
            b.sync()  # the previous pass's LDS reads are done
            b.smem_store_vN(lds, [tid], b.select(o_ok, key_o, b.const_i32(-1)), 1)
            b.smem_store_vN(lds, [b.add(tid, tc)], b.select(o_ok, nt_o, zero), 1)
            b.sync()
            scan = b.scf_for_iter(
                zero, tc, one, [("wl_st", start), ("wl_tt", total)],
                iv_name="wl_j", elide_trailing_barrier=False,
            )  # fmt: skip
            with scan as (j, (st, tt)):
                n_j = lds_at(b.add(j, tc))
                before = worklist_sorts_before(
                    b, key_o=lds_at(j), o=b.add(c0, j), key_s=key_s, s=s
                )
                b.scf_yield(b.add(st, b.select(before, n_j, zero)), b.add(tt, n_j))
            b.scf_yield(*scan.results)
        start, total = chunks.results
        # the owned sequence's items from its start (none for s >= n_batch)
        writes = b.scf_for(zero, b.select(s_ok, nt_s, zero), one, iv_name="wl_t")
        with writes as t:
            worklist_store_item(b, a["WORKLIST"], b.add(start, t), s, t)
        b.scf_yield(total)
    with b.scf_if(b.cmp_eq(tid, zero)):
        worklist_store_count(b, a["WORKLIST"], outer.results[0])


def build_attn_bwd_prep(spec: AttnBwdAuxSpec, *, arch: str) -> KernelDef:
    """q-side stats (lse2, Dsum) and dQ-workspace zero, plus the kv-zero role
    and, with ``spec.worklist``, the work-list role."""
    _check_arch(arch)
    _check_vec8(spec, "prep")
    rows = attn_bwd_aux_rows_per_cta(spec, arch=arch)
    b = IRBuilder(spec.kernel_name("prep"))
    b.fence_barriers = BWD_BARRIER_FENCE
    a = _declare(b, "prep", spec)
    io = _DTYPES[spec.dtype]
    ln = _Lanes(b, spec, rows)
    y = b.block_id_y()
    z = b.block_id_z()
    if spec.kv_target == "contiguous":
        with b.scf_if_else(b.cmp_lt(y, a["h_q"])) as (then_ctx, else_ctx):
            with then_ctx:
                _prep_q_role(b, spec, a, ln, y, z, io)
            with else_ctx:
                _prep_kv_role(b, spec, a, ln, y, z)
    else:
        with b.scf_if(b.cmp_lt(y, a["h_q"])):
            _prep_q_role(b, spec, a, ln, y, z, io)
    if spec.worklist:
        # one CTA: (0, 0, 0), a CTA-uniform branch around the role's barriers
        zero = b.const_i32(0)
        first = b.land(
            b.land(b.cmp_eq(b.block_id_x(), zero), b.cmp_eq(y, zero)),
            b.cmp_eq(z, zero),
        )
        with b.scf_if(b.land(b.cmp_ne(a["use_worklist"], zero), first)):
            _worklist_role(b, a, attn_bwd_aux_block_threads(spec, arch=arch))
    return _finish(b, spec, arch)


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------


def _ws_vec8(b, slab, rr, col0, n: int = 8) -> list:
    """``n`` (a multiple of 4) fp32 workspace values (0 for rows outside the
    workspace), read as 16-byte pieces."""
    ok = slab.row_ok(rr)
    zero = b.const_f32(0.0)
    out = []
    for half in range(n // 4):
        col = col0 if half == 0 and n == 4 else b.add(col0, b.const_i32(4 * half))
        v4 = slab.load_vec4(rr, col)
        out += [b.select(ok, b.vec_extract(v4, i), zero) for i in range(4)]
    return out


def build_attn_bwd_convert(spec: AttnBwdAuxSpec, *, arch: str) -> KernelDef:
    """fp32 workspace times ``mult`` to the output dtype at arbitrary strides."""
    return _build_convert(spec, arch, "convert")


def build_attn_bwd_reduce(spec: AttnBwdAuxSpec, *, arch: str) -> KernelDef:
    """The convert of ``n_slices`` fp32 workspace slices (``dkv_sink =
    "slices"``): ``DST = cast(mult * (s_0 + s_1 + ... + s_{n-1}))``.

    Slice ``i`` starts ``i * s_stride`` floats after ``SRC`` and is laid out
    like the convert's workspace. The sum runs in slice order, one fp32 add
    chain per element, so the result does not depend on which CTA wrote a
    slice first (deterministic). Rows and the zero rule are the convert's.
    """
    return _build_convert(spec, arch, "reduce")


def _build_convert(spec: AttnBwdAuxSpec, arch: str, stage: str) -> KernelDef:
    _check_arch(arch)
    if spec.worklist:
        raise ValueError("worklist=True applies to the prep kernel only")
    rows = attn_bwd_aux_rows_per_cta(spec, arch=arch)
    b = IRBuilder(spec.kernel_name(stage))
    b.fence_barriers = BWD_BARRIER_FENCE
    a = _declare(b, stage, spec)
    io = _DTYPES[spec.dtype]
    ln = _Lanes(b, spec, rows)
    h = b.block_id_y()
    z = b.block_id_z()
    seq = _seq(
        b, spec, z, s_max=a["S_max"], seq_ptr=a["SEQ"], off_ptr=a["OFF"], a=a,
        mult="tok_mult", div="tok_div",
    )  # fmt: skip

    def ws_slab(src):
        return WorkspaceAddr(
            b, src, ws_rows=a["ws_rows"], width=spec.head_size, row0=seq.ws_row0,
            layout=spec.ws_layout,
            n_heads=a["H"] if spec.ws_layout == "token_major" else None,
        ).slab(h, ln.tile_row0)  # fmt: skip

    slab = ws_slab(a["SRC"])
    dst = StridedTensorAddr(
        b, a["DST"], dtype=io, strides=(a["d_b"], a["d_h"], a["d_t"]), seq=seq,
        stage_vec=spec.stage_vec, narrow_vec=spec.narrow_vec,
    ).slab(h, ln.tile_row0, clamp=False)  # fmt: skip
    mult = a["mult"]
    zero = b.const_f32(0.0)

    vec = spec.convert_vec

    def read_sum():
        vals = _ws_vec8(b, slab, ln.row, ln.col0, vec)
        if stage == "convert":
            return vals
        # slices 1 .. n - 1 added in order onto slice 0: slice i is slice 0's
        # slab moved by i * s_stride floats (same rows, same bounds)
        stride = b.mul(a["s_stride"], b.const_i64(4))
        one = b.const_i32(1)
        loop = b.scf_for_iter(
            one, a["n_slices"], one,
            [(f"rsum{i}", v) for i, v in enumerate(vals)], iv_name="rslice",
        )  # fmt: skip
        with loop as (it, cur):
            ptr = b.global_ptr_add(slab.ptr, b.mul(b.zext(it, I64), stride))
            part = _ws_vec8(b, replace(slab, ptr=ptr), ln.row, ln.col0, vec)
            b.scf_yield(*[b.fadd(x, y) for x, y in zip(cur, part)])
        return list(loop.results)

    def write_row(valid):
        vals = read_sum()
        vals = [b.fmul(mult, v) for v in vals]
        if valid is not None:
            vals = [b.select(valid, v, zero) for v in vals]
        if vec == 16:
            # two 8-element output pieces (16-byte stores)
            for part in range(2):
                piece = b.vec_pack(vals[8 * part : 8 * part + 8], F32)
                col = ln.col0 if part == 0 else b.add(ln.col0, b.const_i32(8))
                dst.store_vec(ln.r, col, b.vec_cast_f32_to(piece, io))
            return
        out = b.vec_cast_f32_to(b.vec_pack(vals, F32), io)
        dst.store_vec(ln.r, ln.col0, out)

    if spec.seq_mode == "batched":
        with b.scf_if(b.cmp_lt(ln.r, seq.s_max)):
            write_row(b.cmp_lt(ln.r, seq.length))
    else:
        cta_live = b.cmp_lt(ln.tile_row0, seq.length)
        with b.scf_if(cta_live), b.scf_if(b.cmp_lt(ln.r, seq.length)):
            write_row(None)
    return _finish(b, spec, arch)


# ---------------------------------------------------------------------------
# paged workspace roles
# ---------------------------------------------------------------------------


def _page_walk(b, spec, a, ln: _Lanes):
    """(sequence, logical block, in-page first slot, live) of this CTA.

    The kv length is device data: a value outside ``[0, max_blocks_per_seq *
    page]`` is clamped into it (the defined behaviour for out-of-contract
    data; the host check ``check_attn_bwd_unified_inputs`` rejects a length
    past ``max_seqlen_kv`` before launch).
    """
    s = b.block_id_z()
    page = a["page"]
    slot0 = ln.tile_row0
    j = b.div(slot0, page)
    in0 = b.mod(slot0, page)
    span = b.mul(a["max_blocks_per_seq"], page)
    raw = b.global_load_i32(a["SEQ_LENS_KV"], s)
    len_kv = b.smin(b.smax(raw, b.const_i32(0)), span)
    n_blocks = b.div(b.add(len_kv, b.sub(page, b.const_i32(1))), page)
    live = b.cmp_lt(j, n_blocks)
    return s, j, in0, live


def _page_slab(b, spec, a, ptr, s, j, in0, h):
    """(guarded block id, page-workspace slab, block id in range) of block
    ``j`` of sequence ``s``.

    The block-table entry is device data: an entry outside ``[0, num_blocks)``
    (``num_blocks = page_rows / page``; negative ids included) is replaced by
    block 0 for every address, and its workspace slot is moved to
    ``page_rows`` (past the workspace), so its slab has no row inside the
    workspace; the caller also predicates its stores on the returned check.
    """
    raw = block_table_entry(b, a["BLOCK_TABLE"], s, j, bt_stride=a["bt_stride"])
    zero = b.const_i32(0)
    num_blocks = b.div(a["page_rows"], a["page"])
    ok = b.land(b.cmp_ge(raw, zero), b.cmp_lt(raw, num_blocks))
    pid = b.select(ok, raw, zero)
    phys0 = b.add(b.mul(b.zext(pid, I64), b.zext(a["page"], I64)), b.zext(in0, I64))
    phys0 = b.select(ok, phys0, b.zext(a["page_rows"], I64))
    ws = WorkspaceAddr(
        b, ptr, ws_rows=a["page_rows"], width=spec.head_size, row0=phys0
    ).slab(h)
    return pid, ws, ok


def _check_paged(spec: AttnBwdAuxSpec) -> None:
    if spec.kv_target != "paged":
        raise ValueError("page roles need kv_target='paged'")
    if spec.ws_layout != "head_major":
        raise ValueError("the page workspace is head-major")


def build_attn_bwd_page_zero(spec: AttnBwdAuxSpec, *, arch: str) -> KernelDef:
    """Zero the fp32 page-workspace slots of every live (sequence, block)."""
    _check_arch(arch)
    _check_paged(spec)
    _check_vec8(spec, "page_zero")
    rows = attn_bwd_aux_rows_per_cta(spec, arch=arch)
    b = IRBuilder(spec.kernel_name("page_zero"))
    b.fence_barriers = BWD_BARRIER_FENCE
    a = _declare(b, "page_zero", spec)
    ln = _Lanes(b, spec, rows)
    h = b.block_id_y()
    s, j, in0, live = _page_walk(b, spec, a, ln)
    with b.scf_if(live):
        _pid, ws_k, id_ok = _page_slab(b, spec, a, a["WS_PDK"], s, j, in0, h)
        _pid2, ws_v, _ok2 = _page_slab(b, spec, a, a["WS_PDV"], s, j, in0, h)
        ok = b.land(b.cmp_lt(b.add(in0, ln.row), a["page"]), ws_k.row_ok(ln.row))
        ok = b.land(ok, id_ok)
        with b.scf_if(ok):
            ws_k.store_vec(ln.row, ln.col0, _zero_vec8(b))
            ws_v.store_vec(ln.row, ln.col0, _zero_vec8(b))
    return _finish(b, spec, arch)


def build_attn_bwd_page_convert(spec: AttnBwdAuxSpec, *, arch: str) -> KernelDef:
    """Write ``mult * slot`` to the paged gradient cache for every live block."""
    _check_arch(arch)
    _check_paged(spec)
    _check_vec8(spec, "page_convert")
    rows = attn_bwd_aux_rows_per_cta(spec, arch=arch)
    io = _DTYPES[spec.dtype]
    b = IRBuilder(spec.kernel_name("page_convert"))
    b.fence_barriers = BWD_BARRIER_FENCE
    a = _declare(b, "page_convert", spec)
    ln = _Lanes(b, spec, rows)
    h = b.block_id_y()
    s, j, in0, live = _page_walk(b, spec, a, ln)
    with b.scf_if(live):
        pid, ws, id_ok = _page_slab(b, spec, a, a["SRC"], s, j, in0, h)
        dst = PagedCacheAddr(
            b, a["DST"], dtype=io, strides=(a["d_blk"], a["d_tok"], a["d_h"]),
            stage_vec=spec.stage_vec,
        ).slab(pid, h, in0)  # fmt: skip
        ok = b.land(b.cmp_lt(b.add(in0, ln.row), a["page"]), ws.row_ok(ln.row))
        # a block id outside the cache is never written (dK / dV of a bad entry)
        ok = b.land(ok, id_ok)
        with b.scf_if(ok):
            vals = [b.fmul(a["mult"], v) for v in _ws_vec8(b, ws, ln.row, ln.col0)]
            out = b.vec_cast_f32_to(b.vec_pack(vals, F32), io)
            dst.store_vec(b.add(in0, ln.row), ln.col0, out)
    return _finish(b, spec, arch)
