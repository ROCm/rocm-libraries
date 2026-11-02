# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Deterministic split-K for the dgrad ``pp_boxes`` kernels (``pp_box_det``).

Stage 1 is the dgrad kernel with ``DgradConvSpec.pp_box_det``: every slice of a
split box stores its f32 accumulators -- as the lanes hold them (the MFMA
register layout, 16 bytes per lane per accumulator) -- into its own workspace
slot, slot = the workgroup's logical block id. Unsplit boxes store dX as
before. Stage 2 (:func:`build_dgrad_box_det_reduce`) runs one workgroup per
split tile with the *same* warp grid: each lane loads its own accumulators from
the tile's slots in slice order (0, 1, ..., S-1), adds them in f32 in that fixed
order, and stores the tile once through the cshuffle epilogue at the dX pixels
of the tile's rows. No atomics, no dX clear, one rounding: bitwise
deterministic.

The sub-GEMM buffer of a ``pp_box_det`` launch (:func:`pp_box_det_pack`)::

    records | magic table | header: slot-0 word, counter word | row tables
    (pp_boxes 7) | stage-2 table: 8 words per split tile | stage-2 rows:
    tile_m per split tile | (pad to 64 words) | slots: tile_m * tile_n f32
    (bf16: pp_box_det 2/4) per block | one i32 counter per block (3/4)

Stage-2 table entry: ``[slot0, S, slot_step, rows_word, n_off, m_valid, 0, 0]``
-- the tile's slice-s slot is ``slot0 + s * slot_step``, its row r's dX pixel
index (``(n*Di + d)*Hi*Wi + h*Wi + w``) is word ``rows_word + r``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace as dc_replace
from typing import List, Tuple

from rocke.core.ir import BF16, F32, I32, IRBuilder, KernelDef, PtrType
from rocke.helpers.atoms import mfma_atom
from rocke.helpers.epilogues import CShuffleEpilogue
from rocke.helpers.geometry import WarpGrid
from rocke.helpers.spec import SignatureBuilder, kernel_name_join
from rocke.helpers.tensor_view import make_buffer_resource

PBD_ENTRY_WORDS = 8
_WS_ALIGN_WORDS = 64


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _record_pixels(sg, problem) -> List[int]:
    """dX pixel index of every row of a pp_boxes record (box or pixel list)."""
    if sg.row_list is not None:
        return list(sg.row_list)
    p = problem
    Di = p.Di if p.is_3d else 1
    ds, db = sg.a_embed_h_coeff, sg.a_embed_w_coeff
    hs, ws = sg.h_tilde_slice, sg.w_tilde_slice
    hb, wb = sg.d_h_offset, sg.d_w_offset
    if not p.is_3d:
        ds, db = 1, 0
    out = []
    for m in range(sg.gemm_m):
        nd, r = divmod(m, hs * ws)
        h, w = divmod(r, ws)
        img, d = divmod(nd, ds)
        out.append(((img * Di + d + db) * p.Hi + h + hb) * p.Wi + w + wb)
    return out


def pp_box_det_pack(
    sub_gemms, problem, *, tile_m: int, tile_n: int, pp_box_order: int, slot_bytes: int = 4
) -> Tuple[List[int], dict]:
    """The i32 words of a ``pp_box_det`` launch's sub-GEMM buffer up to the
    workspace, and its layout: ``ws_word`` (slot 0), ``ws_words`` (all slots),
    ``tab_word`` / ``n_tiles`` (the stage-2 table), ``blocks``."""
    from kernels.common.conv_implicit_gemm_dgrad import (
        _MAGIC_FIELDS,
        _RECORD_FIELDS,
        pack_sub_gemm_buffer,
    )

    buf = pack_sub_gemm_buffer(sub_gemms, tile_m, tile_n)
    hdr = len(sub_gemms) * (_RECORD_FIELDS + _MAGIC_FIELDS)
    buf[hdr:hdr] = [0, 0]  # header: slot-0 word, counter word (set below)
    nn = _ceil_div(problem.cpg, tile_n)
    entries, rows = [], []
    for sg in sub_gemms:
        S = max(1, sg.b_x_stride)
        if S <= 1:
            continue
        nm = _ceil_div(sg.gemm_m, tile_m)
        pix = _record_pixels(sg, problem)
        for m in range(nm):
            r = pix[m * tile_m:(m + 1) * tile_m]
            for n in range(nn):
                if pp_box_order & 1:
                    slot0, step = sg.block_start + (n * S) * nm + m, nm
                else:
                    slot0, step = sg.block_start + (m * nn + n) * S, 1
                _S = S
                if os.environ.get("ROCKE_PBD_DEBUG_S"):  # timing experiments only: wrong dX
                    _S = min(S, int(os.environ["ROCKE_PBD_DEBUG_S"]))
                entries.append([slot0, _S, step, len(rows), n * tile_n, len(r), 0, 0])
                rows.extend(r + [r[0]] * (tile_m - len(r)))
                # (rows_word is made absolute below)
    tab_word = len(buf)
    rows_word0 = tab_word + PBD_ENTRY_WORDS * len(entries)
    for e in entries:
        e[3] += rows_word0
        buf.extend(e)
    buf.extend(rows)
    ws_word = _ceil_div(len(buf), _WS_ALIGN_WORDS) * _WS_ALIGN_WORDS
    buf[hdr] = ws_word
    blocks = sub_gemms[-1].block_end
    ws_words = blocks * tile_m * tile_n * slot_bytes // 4
    cnt_word = ws_word + ws_words  # per-tile counters (pp_box_det 3/4), zeroed once
    buf[hdr + 1] = cnt_word
    if (cnt_word + blocks) * 4 >= 0x7FFFFFF0:
        raise ValueError("pp_box_det: workspace past the 2 GiB buffer range")
    return buf, dict(
        ws_word=ws_word, ws_words=ws_words, tab_word=tab_word, cnt_word=cnt_word,
        n_tiles=len(entries), blocks=blocks, total_words=cnt_word + blocks,
    )


@dataclass(frozen=True)
class DgradBoxDetReduceSpec:
    tile_m: int
    tile_n: int
    warp_m: int
    warp_n: int
    dtype_d: str = "bf16"
    store_vec: int = 8
    m_chunks: int = 1
    wave_size: int = 64
    # Workgroups per split tile (grid y): workgroup q folds and stores the
    # rows of MFMA row blocks [q*g, (q+1)*g) of every warp (g = the warp's
    # row blocks / msplit) -- more workgroups on the same bytes.
    msplit: int = 1
    # Slices per loop trip (loads issued together; past S out of range).
    unroll: int = 1
    # One wave per workgroup (grid z = the stage-1 wave): each folds and
    # stores its own warp tile's rows -- small workgroups, little LDS.
    wave_wg: bool = False
    # Slot element: 4 = f32 (pp_box_det 1), 2 = bf16 (pp_box_det 2).
    slot_bytes: int = 4
    # Cache-policy aux bits of the slot loads (0 = default; 2 = NT streaming).
    load_aux: int = 0
    # Load the store rows' dX pixel indices before the slot loads (m_chunks
    # 1): their latency overlaps the fold instead of preceding each store.
    pix_pre: bool = False
    name: str = "conv_dgrad_box_det_reduce"

    @property
    def block_size(self) -> int:
        return self.wave_size if self.wave_wg else self.warp_m * self.warp_n * self.wave_size

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name, f"t{self.tile_m}x{self.tile_n}", f"w{self.warp_m}x{self.warp_n}",
            f"v{self.store_vec}", f"mch{self.m_chunks}", f"ms{self.msplit}", f"u{self.unroll}", "wwg" if self.wave_wg else "", "sb2" if self.slot_bytes == 2 else "",
            f"la{self.load_aux}" if self.load_aux else "", "pp" if self.pix_pre else "", self.dtype_d,
        )


def build_dgrad_box_det_reduce(spec: DgradBoxDetReduceSpec, *, arch: str = "gfx942") -> KernelDef:  # noqa: ARG001
    """Stage 2 of ``pp_box_det``: one workgroup per split tile (grid x =
    ``n_tiles``), the stage-1 warp grid; see the module docstring."""
    from rocke.helpers.io import io_ir_type

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
    ws_ptr = b.param("ws", PtrType(I32, "global"), noalias=True, readonly=True, align=16)
    dx_ptr = b.param(
        "dx", PtrType(io_ir_type(spec.dtype_d), "global"), noalias=True, writeonly=True, align=16
    )
    tab_word = b.param("tab_word", I32)
    ws_word = b.param("ws_word", I32)
    c_C = b.param("C", I32)
    dx_bytes = b.param("dx_bytes", I32)

    atom = mfma_atom("bf16", 16, 16, 16)
    full = WarpGrid.from_atom(
        atom, tile_m=spec.tile_m, tile_n=spec.tile_n, tile_k=16,
        warp_m=spec.warp_m, warp_n=spec.warp_n, wave_size=spec.wave_size,
    )
    mfm, mfn = full.mfmas_per_warp_m, full.mfmas_per_warp_n
    if mfm % spec.msplit:
        raise ValueError(f"msplit {spec.msplit} does not divide {mfm} row blocks")
    g = mfm // spec.msplit
    if spec.wave_wg:
        grid = WarpGrid.from_atom(
            atom, tile_m=g * 16, tile_n=mfn * 16, tile_k=16,
            warp_m=1, warp_n=1, wave_size=spec.wave_size,
        ).bind(b, block_m_axis="y", block_n_axis="x")
    else:
        grid = WarpGrid.from_atom(
            atom, tile_m=spec.tile_m // spec.msplit, tile_n=spec.tile_n, tile_k=16,
            warp_m=spec.warp_m, warp_n=spec.warp_n, wave_size=spec.wave_size,
        ).bind(b, block_m_axis="y", block_n_axis="x")
    tid, lane = grid.tid, grid.lane
    c0 = b.const_i32(0)
    c1 = b.const_i32(1)
    q = b.block_id_y() if spec.msplit > 1 else c0

    e = b.add(tab_word, b.mul(b.block_id_x(), b.const_i32(PBD_ENTRY_WORDS)))

    def _ld(i):
        return b.readfirstlane(b.global_load_i32(ws_ptr, b.add(e, b.const_i32(i))))

    slot0, S, step, rows_word, n_off, m_valid = (_ld(i) for i in range(6))

    n_w = spec.warp_m * spec.warp_n
    tile_elems = spec.tile_m * spec.tile_n
    ws = make_buffer_resource(b, ws_ptr, num_bytes=b.const_i32(0x7FFFFFF0))
    if spec.wave_wg:
        wave = b.block_id_z()
    else:
        wave = b.readfirstlane(b.div(tid, b.const_i32(spec.wave_size)))
    eb = spec.slot_bytes
    lane_off = b.mul(b.add(b.mul(wave, b.const_i32(64)), lane), b.const_i32(4 * eb))
    acc_stride = n_w * 64 * 4 * eb
    ws_bytes0 = b.add(
        b.mul(ws_word, b.const_i32(4)),
        b.mul(q, b.const_i32(g * mfn * acc_stride)),
    )

    pre = None
    if spec.pix_pre and spec.m_chunks == 1:
        # The cshuffle store's per-thread vectors (CShuffleEpilogue._store_vectors
        # order): vector e of thread t is row (e*threads + t) // (tile_n/sv).
        sv = CShuffleEpilogue.from_grid(
            atom=atom, grid=grid, max_store_vec=spec.store_vec, out_dtype=spec.dtype_d
        ).store_vec
        nvec = grid.tile_m * grid.tile_n // sv // grid.block_size
        pre = []
        for e_ in range(nvec):
            row = b.div(
                b.add(b.const_i32(e_ * grid.block_size), tid), b.const_i32(grid.tile_n // sv)
            )
            pre.append(row)  # mapped to pixels below, once _row exists
    if spec.wave_wg:
        wm = b.div(wave, b.const_i32(spec.warp_n))
        wn = b.sub(wave, b.mul(wm, b.const_i32(spec.warp_n)))
        grid = dc_replace(
            grid, block_m_off=c0,
            block_n_off=b.add(n_off, b.mul(wn, b.const_i32(mfn * 16))),
        )
        row0 = b.add(b.mul(wm, b.const_i32(mfm * 16)), b.mul(q, b.const_i32(g * 16)))
    else:
        grid = dc_replace(grid, block_m_off=c0, block_n_off=n_off)
    dx = make_buffer_resource(b, dx_ptr, num_bytes=dx_bytes)
    c_g16 = b.const_i32(g * 16)
    q_off = b.mul(q, c_g16)

    def _row(b_: IRBuilder, m_val):
        """Sub-tile row -> the split tile's row."""
        if spec.wave_wg:
            return b_.add(row0, m_val)
        if spec.msplit == 1:
            return m_val
        wm = b_.div(m_val, c_g16)
        return b_.add(
            b_.add(b_.mul(wm, b_.const_i32(mfm * 16)), q_off), b_.sub(m_val, b_.mul(wm, c_g16))
        )

    pre_pix = None
    if pre is not None:
        pre_pix = []
        for row in pre:
            r = _row(b, row)
            pre_pix.append((r, b.global_load_i32(ws_ptr, b.add(rows_word, r))))
        pre_pix.reverse()

    it = [(f"pbd_a{j}", b.zero_vec_f32(4)) for j in range(g * mfn)]
    U = spec.unroll
    c_oob = b.const_i32(0x7FFFFFF0)
    red = b.scf_for_iter(c0, S, b.const_i32(U), it, iv_name="s")
    with red as (s, accs_s):
        # U slices per trip, all loads first; a slice past S reads out of
        # range (0.0, exact). The adds stay in slice order.
        parts = []
        for u in range(U):
            su = b.add(s, b.const_i32(u)) if u else s
            slot = b.add(slot0, b.mul(su, step))
            base = b.add(
                b.add(ws_bytes0, b.mul(slot, b.const_i32(tile_elems * eb))), lane_off
            )
            if u:
                base = b.select(b.cmp_lt(su, S), base, c_oob)
            if eb == 2:
                raw = [
                    b.buffer_load_vN(ws.rsrc, b.add(base, b.const_i32(j * acc_stride)), c0, BF16, 4, aux=spec.load_aux)
                    for j in range(len(accs_s))
                ]
                parts.append([
                    b.vec_pack([b.cast_to_f32(b.vec_extract(v, i)) for i in range(4)], F32)
                    for v in raw
                ])
            else:
                parts.append([
                    b.buffer_load_vN(ws.rsrc, b.add(base, b.const_i32(j * acc_stride)), c0, F32, 4, aux=spec.load_aux)
                    for j in range(len(accs_s))
                ])
        new = list(accs_s)
        for u in range(U):
            new = [b.fadd(a, pv) for a, pv in zip(new, parts[u])]
        b.scf_yield(*new)
    accs = list(red.results)


    def dx_addr(b_: IRBuilder, m_val, n_val):
        if pre_pix:
            r, pix = pre_pix.pop()
        else:
            r = _row(b_, m_val)
            pix = b_.global_load_i32(ws_ptr, b_.add(rows_word, r))
        ok = b_.cmp_lt(r, m_valid) if (spec.msplit > 1 or spec.wave_wg) else None
        return b_.add(b_.mul(pix, c_C), n_val), ok

    epi = CShuffleEpilogue.from_grid(
        atom=atom, grid=grid, max_store_vec=spec.store_vec, out_dtype=spec.dtype_d
    )
    epi = dc_replace(epi, war_barriers=1, m_chunks=spec.m_chunks)
    bound_m = m_valid if (spec.msplit == 1 and not spec.wave_wg) else b.const_i32(spec.tile_m)
    epi.store(b, accs=accs, addr_fn=dx_addr, d_rsrc=dx.rsrc, bounds=(bound_m, c_C))
    return b.kernel


def dgrad_box_det_reduce_signature(spec: DgradBoxDetReduceSpec) -> list:
    return (
        SignatureBuilder()
        .ptr("ws", "i32")
        .ptr("dx", spec.dtype_d)
        .scalar("tab_word", "i32")
        .scalar("ws_word", "i32")
        .scalar("C", "i32")
        .scalar("dx_bytes", "i32")
        .build()
    )
