#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/attention_bwd_layout_emit.py -- Python reference for the C ports
# of the attention backward host decision helpers that emit no IR:
# kernels/common/_attention_bwd_caps.py (arch caps, atoms, the check_*
# functions, the asm-MFMA tables, check_agpr_alloc), _attention_bwd_tiles.py
# (warp grids), _attention_bwd_lds.py (XtLayout, RowXorLayout, XtWriter,
# DmaSlabLayout, the conflict-predictor choice of an XT swizzle, BwdLdsRequest
# and plan_bwd_lds over raw requests) and the ABI lists of attention_bwd.py.
# Each config prints a canonical text dump of one group of results; the C twin
# attention_bwd_layout_emit.c prints the same dump and run_diff compares the
# two byte for byte (sha256), like a .ll.
#
# Config kinds ("<kind>|<args>"):
#   caps|ARCH     bwd_arch_caps, bwd_atoms / select_bwd_atom per dtype, every
#                 check_*, the asm-MFMA tables, check_agpr_alloc, BwdArchFacts,
#                 and per backward atom the C-fragment facts from its lane map
#                 (stats_vec runs with check_stats_vec_layout's message, quad
#                 transpose)
#   proofs|ARCH   BwdTileGeometry.from_spec over every atom / transpose /
#                 route combination: the C port mirrors its checks without the
#                 lane-map proofs, so both must agree that those never refuse
#   tiles|W       g02_grids / g13_grids / natural_grids / kv_operand_split /
#                 replicated_g02 / check_lds_route_grid over W waves
#   small|D       small_tile_grids over every (kM0, kN0, W)
#   xt|R,C,E,K    XtLayout / RowXorLayout over every swizzle (rows, cols,
#                 elem_bytes, chunk_bytes)
#   writer|N      XtWriter inits and default_xt_writer sweeps (group N)
#   choose|R,C    choose_xt_swizzle / planner_xt_swizzle over threads, atom K
#                 and arch for one (rows, cols) image
#   access|0      xt_read_accesses / xt_write_accesses / dma_slab_read_accesses
#                 with a wave32 size and a writer that does not match its image
#   dma|R,C,E     DmaSlabLayout: offsets, dma_source, read accesses, conflicts
#   plan|ARCH|S|FIELDS  BwdLdsRequest(**FIELDS) and plan_bwd_lds(strict=S)
#   abi|0         attn_bwd_params of every stage, static_unit_mults
#
# Only mode "ll" (the dump) is supported.
import dataclasses
import functools
import sys

from kernels.common import _attention_bwd_caps as caps
from kernels.common import _attention_bwd_lds as lds
from kernels.common import _attention_bwd_tiles as tiles
from kernels.common._attention_bwd_body import BwdTileGeometry
from kernels.common._attention_bwd_frag import c_quad_transposable
from kernels.common._attention_bwd_valu import check_stats_vec_layout
from kernels.common.attention_bwd import (
    AttnBwdSpec,
    attn_bwd_arch_facts,
    attn_bwd_params,
    static_unit_mults,
)

# The XT swizzle search predicts the same instruction many times over: the
# planner repeats choose_xt_swizzle, and the reads do not depend on the writer.
# Memoize the per-instruction predictor in this process; results are unchanged.
_predict_max = lds._predict_max


@functools.cache
def _predict_max_cached(arch, opcode, accesses):
    return _predict_max(arch, opcode, accesses)


lds._predict_max = lambda arch, opcode, accesses: _predict_max_cached(
    arch, opcode, tuple(accesses)
)

ARCHES = ("gfx942", "gfx950", "gfx1151", "gfx1201")

_PLAN_REQUESTS = (
    ("gfx942", True, "head_size:64;k_m0:32;k_n0:128;waves:8"),
    ("gfx942", True, "head_size:64;k_m0:32;k_n0:128;waves:8;transpose_source:lds_plain"),
    ("gfx942", False, "head_size:128;k_m0:16;k_n0:128;waves:8;kt_source:lds"),
    ("gfx942", True, "head_size:128;k_m0:16;k_n0:128;waves:8;kt_source:lds;transpose_source:lds_plain"),
    ("gfx942", True, "head_size:64;k_m0:32;k_n0:64;waves:4;kv_residency:lds;kt_source:lds;lds_order:phase_low"),
    ("gfx942", True, "head_size:32;k_m0:32;k_n0:128;waves:4;acc_in_lds:True;dkv_epilogue:lds_staged_pair"),
    ("gfx942", True, "head_size:64;k_m0:32;k_n0:128;waves:8;ring_depth:2;transpose_source:lds_plain;lds_swizzle:q=pad8,do=pad16"),
    ("gfx942", True, "head_size:64;k_m0:32;k_n0:128;waves:8;dq_mode:split;pt_route:lds;transpose_source:lds_plain"),
    ("gfx942", True, "head_size:64;k_m0:32;k_n0:128;waves:8;dq_mode:split;transpose_source:xt_lds;kt_source:reg"),
    ("gfx942", True, "head_size:64;k_m0:16;k_n0:64;waves:4;pt_route:lds;ds_slots:2;dkv_epilogue:direct;transpose_source:lds_plain"),
    ("gfx942", True, "head_size:64;k_m0:32;k_n0:128;waves:8;lds_swizzle: k_stage = xor , qt=pad32,dot=none,kt_stage=pad8,dkv_out=xor,dkv_out=pad16"),
    ("gfx942", True, "head_size:128;k_m0:64;k_n0:256;waves:8;kv_residency:lds;kt_source:lds;transpose_source:lds_plain"),
    ("gfx942", False, "head_size:128;k_m0:64;k_n0:256;waves:8;kv_residency:lds;kt_source:lds;transpose_source:lds_plain;lds_order:phase_low"),
    ("gfx950", True, "head_size:64;k_m0:32;k_n0:256;waves:8;kt_source:lds;transpose_source:tr_read;atom_k_g4:32"),
    ("gfx950", True, "head_size:64;k_m0:32;k_n0:128;waves:8;transpose_source:tr_read;global_path:dma;ring_depth:3"),
    ("gfx950", True, "head_size:128;k_m0:16;k_n0:128;waves:8;transpose_source:tr_read;global_path:dma;ring_depth:2;lds_swizzle:ds=pad32"),
    ("gfx950", True, "head_size:32;k_m0:32;k_n0:128;waves:4;transpose_source:xt_lds;atom_k_g13:32;atom_k_g4:32;kt_source:lds"),
    ("gfx950", True, "head_size:64;k_m0:16;k_n0:64;waves:4;transpose_source:lds_plain;kv_residency:lds;kt_source:lds;acc_in_lds:True;dq_mode:split"),
    ("gfx1201", True, "head_size:64;k_m0:16;k_n0:16;waves:1;wave_size:32;kv_residency:lds;kt_source:lds;transpose_source:wmma_lds;pt_route:lds;acc_in_lds:True"),
    ("gfx1151", True, "head_size:128;k_m0:16;k_n0:16;waves:1;wave_size:32;kv_residency:lds;kt_source:lds;transpose_source:xt_lds;pt_route:lds;acc_in_lds:True"),
    ("gfx942", True, "head_size:64;k_m0:32;k_n0:128;waves:8;dtype:fp32"),
    ("gfx942", True, "head_size:64;k_m0:32;k_n0:128;waves:8;kv_residency:hbm"),
    ("gfx942", True, "head_size:64;k_m0:32;k_n0:128;waves:8;kt_source:smem"),
    ("gfx942", True, "head_size:64;k_m0:32;k_n0:128;waves:8;kv_residency:lds"),
    ("gfx942", True, "head_size:64;k_m0:32;k_n0:128;waves:8;transpose_source:ldsx"),
    ("gfx942", True, "head_size:64;k_m0:32;k_n0:128;waves:8;ring_depth:4"),
    ("gfx942", True, "head_size:64;k_m0:32;k_n0:128;waves:8;global_path:tdm"),
    ("gfx942", True, "head_size:64;k_m0:32;k_n0:128;waves:8;global_path:dma"),
    ("gfx942", True, "head_size:64;k_m0:32;k_n0:128;waves:8;pt_route:regs"),
    ("gfx942", True, "head_size:64;k_m0:32;k_n0:128;waves:8;dq_mode:none"),
    ("gfx942", True, "head_size:64;k_m0:32;k_n0:128;waves:8;dkv_epilogue:lds"),
    ("gfx942", True, "head_size:64;k_m0:32;k_n0:128;waves:8;lds_order:high"),
    ("gfx942", True, "head_size:64;k_m0:32;k_n0:128;waves:8;ds_slots:3"),
    ("gfx942", True, "head_size:64;k_m0:32;k_n0:128;waves:8;lds_swizzle:q=pad4"),
    ("gfx942", True, "head_size:64;k_m0:32;k_n0:128;waves:8;atom_k_g13:8"),
    ("gfx942", True, "head_size:64;k_m0:0;k_n0:128;waves:8"),
    ("gfx942", True, "head_size:64;k_m0:32;k_n0:128;waves:8;transpose_source:lds_plain;lds_swizzle:q=xor"),
    ("gfx942", True, "head_size:64;k_m0:32;k_n0:48;waves:8;lds_swizzle:kt_stage=xor"),
    ("gfx942", True, "head_size:64;k_m0:32;k_n0:128;waves:8;transpose_source:lds_plain;lds_swizzle:ds=xor"),
    ("gfx950", True, "head_size:64;k_m0:16;k_n0:256;waves:8;transpose_source:tr_read;global_path:dma"),
)  # fmt: skip

_XT_SHAPES = (
    (64, 16, 2, 8), (64, 32, 2, 8), (128, 16, 2, 8), (32, 64, 2, 8),
    (128, 128, 2, 8), (64, 192, 2, 8), (64, 48, 2, 8), (64, 16, 2, 16),
    (16, 8, 2, 4), (0, 16, 2, 8), (64, 16, 2, 12), (64, 24, 2, 16),
    (32, 256, 2, 16), (16, 4, 2, 8), (16, 4, 2, 16), (16, 16, 3, 8),
)  # fmt: skip

_CHOOSE_SHAPES = tuple(
    (r, c) for r in (32, 64, 128) for c in (16, 32, 64, 128, 192, 256)
)

_DMA_SHAPES = (
    (16, 64, 2), (32, 64, 2), (16, 128, 2), (64, 128, 2), (16, 32, 2),
    (8, 32, 2), (16, 40, 2), (16, 72, 2), (0, 64, 2), (16, 256, 2), (32, 16, 2),
    (16, 36, 2),
)  # fmt: skip


def _configs():
    out = [f"caps|{a}" for a in (*ARCHES, "gfx90a")]
    out += [f"proofs|{a}" for a in ARCHES]
    out += [f"tiles|{w}" for w in range(9)]
    out += [f"small|{d}" for d in (32, 64, 128)]
    out += ["xt|" + ",".join(map(str, s)) for s in _XT_SHAPES]
    out += [f"writer|{n}" for n in range(3)]
    out += ["choose|" + ",".join(map(str, s)) for s in _CHOOSE_SHAPES]
    out += ["dma|" + ",".join(map(str, s)) for s in _DMA_SHAPES]
    out += [f"plan|{a}|{int(s)}|{f}" for a, s, f in _PLAN_REQUESTS]
    out += ["abi|0", "access|0"]
    return tuple(out)


_CONFIGS = _configs()


def _exc(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"


def _try(lines, label, fn, *args, **kwargs):
    try:
        lines.append(f"{label}: {fn(*args, **kwargs)}")
    except (KeyError, ValueError) as e:
        lines.append(f"{label}: {_exc(e)}")


def _op_id(arch, dt, k):
    return caps.select_bwd_atom(arch, dt, k).op_id


def _layout_repr(plan, name):
    return repr(plan.layout(name))


def _unit_mults(sc, mode):
    return ",".join(sorted(static_unit_mults(sc, mode)))


def _t(x):
    return ", ".join(str(v) for v in x)


# ------------------------------------------------------------------- caps


def _caps(arch):
    lines = []
    try:
        c = caps.bwd_arch_caps(arch)
    except ValueError as e:
        lines.append("caps: " + _exc(e))
    else:
        lines.append(
            f"caps: wave={c.wave_size} path={c.matrix_path} lds={c.lds_capacity_bytes} "
            f"dma={c.lds_dma_dwords!r} max_dma={c.max_lds_dma_dwords} "
            f"tr64={c.has_tr_read_b64} tr128={c.has_tr_read_b128} "
            f"has_agprs={c.has_agprs} vgprs={c.arch_vgprs} agprs={c.agprs} "
            f"atomic={c.native_fp32_atomic_add} waves={c.legal_waves!r}"
        )
    for dt in ("fp16", "f16", "bf16", "fp32"):
        try:
            ops = caps.bwd_atoms(arch, dt)
            lines.append(
                f"atoms {dt}: "
                + " ".join(
                    f"{o.op_id}/k{o.k}/c{o.c_frag_len}/w{o.wave_size}" for o in ops
                )
            )
        except ValueError as e:
            lines.append(f"atoms {dt}: {_exc(e)}")
        for k in (16, 32, 8):
            _try(
                lines,
                f"select {dt} k{k}",
                _op_id,
                arch,
                dt,
                k,
            )
        _try(lines, f"wide {dt}", caps.wide_k_atom_available, arch, dt)
    for w in range(10):
        _try(lines, f"check_waves {w}", caps.check_waves, arch, w)
    for n in (-1, 0, 65536, 65537, 163840, 163841):
        _try(lines, f"check_lds_bytes {n}", caps.check_lds_bytes, arch, n)
    for d in range(6):
        _try(lines, f"check_lds_dma_width {d}", caps.check_lds_dma_width, arch, d)
    for bits in (32, 64, 128):
        _try(
            lines,
            f"check_transpose_read {bits}",
            caps.check_transpose_read,
            arch,
            bits,
        )
    for k in (16, 32, 8):
        lines.append(f"AGPR_ASM_TAIL_NOP {k}: {caps.AGPR_ASM_TAIL_NOP.get(k)!r}")
        _try(lines, f"wait_states k{k}", caps.mfma_result_wait_states, arch, k)
        for dt in ("fp16", "bf16", "fp32"):
            _try(
                lines,
                f"mnemonic {dt} k{k}",
                caps.mfma_asm_mnemonic,
                arch,
                dt,
                k,
            )
            _try(lines, f"nop_min {dt} k{k}", caps.asm_tail_nop_min, arch, dt, k)
            for nop in (5, 6, 7, 15, 16):
                _try(
                    lines,
                    f"check_nop {dt} k{k} {nop}",
                    caps.check_asm_tail_nop,
                    arch,
                    dt,
                    k,
                    nop,
                )
    lines.append(f"VALU_LEAD: {caps.ASM_MFMA_VALU_LEAD_WAIT_STATES}")
    for a in (
        None,
        "backend",
        (0, 0),
        (128, 128),
        (256, 256),
        (64, 64),
        "auto",
        (128, 512),
    ):
        _try(lines, f"check_agpr_alloc {a!r}", caps.check_agpr_alloc, arch, a)
    try:
        f = attn_bwd_arch_facts(arch)
    except ValueError as e:
        lines.append("facts: " + _exc(e))
    else:
        lines.append(
            f"facts: wave={f.wave_size} path={f.matrix_path} lds={f.lds_capacity_bytes} "
            f"vgprs={f.arch_vgprs} agprs={f.agprs} tr={f.has_tr_read} "
            f"dma={f.has_lds_dma} wide={f.wide_k_atom} waves={f.legal_waves!r} "
            f"atoms={f.legal_atoms!r} transposes={f.legal_transpose_sources!r} "
            f"paths={f.legal_global_paths!r} rings={f.legal_ring_depths!r}"
        )
    for dt in ("fp16", "bf16"):
        try:
            ops = caps.bwd_atoms(arch, dt)
        except ValueError:
            continue
        for op in ops:
            ok = []
            for w in (1, 2, 4):
                try:
                    check_stats_vec_layout(op, w)
                    ok.append(w)
                except ValueError:
                    pass
            lines.append(
                f"c_facts {op.op_id}: stats_vecs={ok!r} quad={c_quad_transposable(op)}"
            )
            for w in (1, 2, 3, 4, 8):
                _try(
                    lines,
                    f"check_stats_vec_layout {op.op_id} {w}",
                    check_stats_vec_layout,
                    op,
                    w,
                )
    return lines


# ----------------------------------------------------------------- proofs


def _proof_specs(arch):
    f = attn_bwd_arch_facts(arch)
    mfma = f.matrix_path == "mfma"
    w = 4 if mfma else 2
    for dt in ("fp16", "bf16"):
        for a02 in f.legal_atoms:
            for a13 in f.legal_atoms:
                for a4 in f.legal_atoms:
                    for ts in f.legal_transpose_sources:
                        for route in ("relabel", "lds") if mfma else ("lds",):
                            for g02, g13 in (
                                ((1, w), (w, 1)),
                                ((2, w // 2), (2, w // 2)),
                            ):
                                if route == "relabel" and g02 != (1, w):
                                    continue
                                yield {
                                    "head_size": 64, "dtype": dt, "waves": w,
                                    "block_m": 32, "block_n": 64, "block_k4": 32,
                                    "warp_grid_g4": (1, w), "atom_g02": a02,
                                    "atom_g13": a13, "atom_g4": a4,
                                    "pt_route": route, "transpose_source": ts,
                                    "kv_residency": "reg" if mfma else "lds",
                                    "kt_source": "reg" if mfma else "lds",
                                    "acc_in_lds": not mfma, "ring_depth": 1,
                                    "global_path": "vgpr", "sched": "none",
                                    "warp_grid_g02": g02, "warp_grid_g13": g13,
                                }  # fmt: skip


def _proofs(arch):
    lines = []
    for fields in _proof_specs(arch):
        label = (
            f"{fields['dtype']} {fields['atom_g02']}/{fields['atom_g13']}/"
            f"{fields['atom_g4']} {fields['transpose_source']} {fields['pt_route']} "
            f"{fields['warp_grid_g02']}"
        )
        try:
            r = AttnBwdSpec(**fields).resolved(arch)
            BwdTileGeometry.from_spec(r, arch)
            lines.append(f"geometry {label}: ok")
        except ValueError as e:
            lines.append(f"geometry {label}: {_exc(e)}")
    return lines


# ------------------------------------------------------------------ tiles


def _tiles(w):
    lines = [
        f"g02_grids: {tiles.g02_grids(w)!r}",
        f"g13_grids: {tiles.g13_grids(w)!r}",
        f"natural: {tiles.natural_grids(w)!r}",
    ]
    for g in tiles.g02_grids(w) + ((3, 1), (1, 16)):
        lines.append(
            f"grid {g!r}: waves={tiles.grid_waves(g)} split={tiles.kv_operand_split(g, w)} "
            f"replicated={tiles.replicated_g02(g, w)}"
        )
    nat02, nat13 = tiles.natural_grids(w)
    cands = tuple(
        dict.fromkeys(tiles.g02_grids(w) + tiles.g13_grids(w) + ((4, 4), (1, 3)))
    )
    for gname, nat in (("warp_grid_g02", nat02), ("warp_grid_g13", nat13)):
        for g in cands:
            for rows, cols in ((16, 16), (32, 64), (64, 128)):
                for route in ("lds", "relabel"):
                    _try(
                        lines,
                        f"route {gname} {g!r} {rows}x{cols} {route}",
                        tiles.check_lds_route_grid,
                        gname,
                        g,
                        nat,
                        w,
                        rows,
                        cols,
                        route,
                    )
    return lines


def _small(d):
    lines = []
    for km in (16, 32, 64):
        for kn in (16, 32, 64, 128, 192, 256):
            for w in (1, 2, 4, 8):
                t = tiles.small_tile_grids(d, km, kn, w)
                lines.append(f"small {km}x{kn} w{w}: {t!r}")
    return lines


# --------------------------------------------------------------------- xt


def _xt(rows, cols, eb, chunk):
    lines = []
    for sw in ("xor", "none", "pad8", "pad16", "pad32", "pad12"):
        try:
            lay = lds.XtLayout(rows, cols, eb, sw, chunk)
        except ValueError as e:
            lines.append(f"xt {sw}: {_exc(e)}")
            continue
        lines.append(f"xt {sw}: pitch={lay.row_pitch_bytes} size={lay.size_bytes}")
        offs = [
            lay.offset(r, c)
            for r in range(0, rows, max(1, rows // 16))
            for c in range(0, cols, max(1, cols // 16))
        ]
        lines.append(f"xt {sw} offsets: {_t(offs)}")
        for r, c in ((rows, 0), (0, cols), (-1, 0)):
            _try(lines, f"xt {sw} offset {r},{c}", lay.offset, r, c)
        for c in (0, 1, 2, 4):
            for n in (1, 2, 4, 8):
                _try(lines, f"xt {sw} vector {c},{n}", lay.check_vector, c, n)
    try:
        rx = lds.RowXorLayout(rows, cols, eb, chunk)
    except ValueError as e:
        lines.append(f"rowxor: {_exc(e)}")
    else:
        lines.append(f"rowxor: pitch={rx.row_pitch_bytes} size={rx.size_bytes}")
        offs = [
            rx.offset(r, c)
            for r in range(0, rows, max(1, rows // 16))
            for c in range(0, cols, max(1, cols // 16))
        ]
        lines.append(f"rowxor offsets: {_t(offs)}")
        _try(lines, "rowxor vector 4,8", rx.check_vector, 4, 8)
    return lines


def _writer(group):
    lines = []
    if group == 0:
        for args in (
            (16, 64, 256, 4, 2, 16, 2), (32, 128, 512, 8, 4, 16, 2), (16, 64, 64, 4, 2, 8, 2),
            (16, 64, 256, 3, 2, 16, 2), (16, 64, 256, 4, 3, 16, 2), (16, 60, 256, 4, 2, 16, 2),
            (16, 64, 256, 16, 2, 16, 2), (16, 64, 256, 4, 16, 16, 2), (16, 64, 256, 4, 2, 12, 2),
            (16, 64, 0, 4, 2, 16, 2), (128, 32, 64, 8, 2, 4, 2), (16, 128, 128, 2, 4, 16, 2),
        ):  # fmt: skip
            try:
                wr = lds.XtWriter(*args)
            except ValueError as e:
                lines.append(f"writer {args!r}: {_exc(e)}")
                continue
            lines.append(
                f"writer {args!r}: n_dg={wr.n_dg} n_sg={wr.n_sg} d_lanes={wr.d_lanes_eff} "
                f"units={wr.units} it={wr.iterations} wb={wr.write_bytes} "
                f"stores={wr.stores_per_thread} loads={wr.loads_per_thread}"
            )
            coords = [
                wr.unit_coords(t, i)
                for i in range(wr.iterations + 1)
                for t in range(0, wr.threads, 7)
            ]
            lines.append(
                "coords: "
                + " ".join("-" if c is None else f"{c[0]},{c[1]}" for c in coords)
            )
    else:
        threads = (64, 128, 256, 512) if group == 1 else (32, 96, 1024)
        for seq in (16, 32, 64, 128, 192, 256):
            for depth in (32, 64, 128):
                for t in threads:
                    for mvb in (16, 8):
                        try:
                            wr = lds.default_xt_writer(seq, depth, t, max_vec_bytes=mvb)
                            lines.append(
                                f"default {seq},{depth},{t},{mvb}: vec_d={wr.vec_d} "
                                f"kpt={wr.k_per_thread} d_lanes={wr.d_lanes}"
                            )
                        except ValueError as e:
                            lines.append(f"default {seq},{depth},{t},{mvb}: {_exc(e)}")
    return lines


def _choose(rows, cols):
    lines = []
    for arch in ("gfx942", "gfx950", "gfx1201", "gfx90a"):
        for threads in (64, 96, 128, 256, 512):
            for atom_k in (16, 32):
                label = f"{arch} t{threads} k{atom_k}"
                try:
                    wr = lds.default_xt_writer(cols, rows, threads)
                    best, table = lds.choose_xt_swizzle(
                        rows, cols, wr, arch, atom_k=atom_k
                    )
                except (KeyError, ValueError) as e:
                    lines.append(f"choose {label}: {_exc(e)}")
                    continue
                parts = []
                for sw, row in table.items():
                    if "illegal" in row:
                        parts.append(f"{sw}=illegal({row['illegal']})")
                    else:
                        p = row["prediction"]
                        pred = "None" if p is None else f"r{p['read']}w{p['write']}"
                        parts.append(f"{sw}={pred}/{row['bytes']}")
                lines.append(f"choose {label}: {best} " + " ".join(parts))
                _try(
                    lines,
                    f"planner {label}",
                    lds.planner_xt_swizzle,
                    rows,
                    cols,
                    threads,
                    arch,
                    2,
                    atom_k,
                )
        lay = lds.XtLayout(rows, cols, 2, "xor", 8)
        _try(
            lines,
            f"xt_conflicts {arch} no writer",
            lds.xt_conflicts,
            lay,
            None,
            arch,
        )
    return lines


def _dma(rows, cols, eb):
    lines = []
    try:
        lay = lds.DmaSlabLayout(rows, cols, eb)
    except ValueError as e:
        return [f"dma: {_exc(e)}"]
    lines.append(
        f"dma: row_bytes={lay.row_bytes} fold={lay.fold} slabs={lay.slabs} "
        f"size={lay.size_bytes}"
    )
    offs = [lay.offset(r, c) for r in range(rows) for c in range(0, cols, 8)]
    lines.append(f"dma offsets: {_t(offs)}")
    _try(lines, "dma offset out", lay.offset, rows, 0)
    src = [lay.dma_source(s) for s in range(lay.size_bytes // 16)]
    lines.append("dma sources: " + " ".join(f"{r},{c}" for r, c in src))
    _try(lines, "dma source out", lay.dma_source, lay.size_bytes // 16)
    for atom_k in (16, 32):
        try:
            acc = lds.dma_slab_read_accesses(lay, m0=0, k0=0, atom_k=atom_k)
            lines.append(
                f"dma reads k{atom_k}: " + " ".join(f"{a}:{b}:{w}" for a, b, w in acc)
            )
        except ValueError as e:
            lines.append(f"dma reads k{atom_k}: {_exc(e)}")
        for arch in ("gfx942", "gfx950", "gfx1151"):
            _try(
                lines,
                f"dma conflicts {arch} k{atom_k}",
                lds.dma_slab_read_conflicts,
                lay,
                arch,
                atom_k=atom_k,
            )
    return lines


def _accesses(label, fn, *args, **kwargs):
    try:
        acc = fn(*args, **kwargs)
    except ValueError as e:
        return f"{label}: {_exc(e)}"
    return f"{label}: " + " ".join(f"{a}:{b}:{w}" for a, b, w in acc)


def _access():
    lines = []
    lay = lds.XtLayout(64, 16, 2, "xor", 8)
    for ws in (64, 32):
        for k in (16, 32):
            lines.append(
                _accesses(
                    f"xt reads w{ws} k{k}",
                    lds.xt_read_accesses,
                    lay,
                    n0=16,
                    k0=0,
                    atom_k=k,
                    wave_size=ws,
                )
            )
    wr = lds.default_xt_writer(16, 64, 256)
    for wave, ws in ((0, 64), (3, 64), (1, 32), (7, 32)):
        lines.append(
            _accesses(
                f"xt writes wave{wave} w{ws}",
                lds.xt_write_accesses,
                lay,
                wr,
                wave=wave,
                iteration=0,
                j=1,
                wave_size=ws,
            )
        )
    for seq, depth in ((32, 64), (16, 32)):
        other = lds.default_xt_writer(seq, depth, 256)
        lines.append(
            _accesses(
                f"xt writes {seq}x{depth}",
                lds.xt_write_accesses,
                lay,
                other,
                wave=0,
                iteration=0,
                j=0,
            )
        )
    slab = lds.DmaSlabLayout(16, 64, 2)
    for ws in (64, 32):
        lines.append(
            _accesses(
                f"dma reads w{ws}",
                lds.dma_slab_read_accesses,
                slab,
                m0=0,
                k0=8,
                atom_k=32,
                wave_size=ws,
            )
        )
    return lines


# ------------------------------------------------------------------- plan

_REQ_INT = {"head_size", "k_m0", "k_n0", "waves", "wave_size", "ring_depth", "ds_slots",
            "atom_k_g13", "atom_k_g4"}  # fmt: skip


def _request(text):
    fields = {}
    for item in text.split(";"):
        name, _, v = item.partition(":")
        if name in _REQ_INT:
            fields[name] = int(v)
        elif name == "acc_in_lds":
            fields[name] = v == "True"
        elif name == "lds_swizzle":
            fields[name] = lds.BwdLdsRequest.canonical_swizzle(v)
        else:
            fields[name] = v
    return lds.BwdLdsRequest(**fields)


def _plan(arch, strict, text):
    lines = []
    try:
        req = _request(text)
    except ValueError as e:
        return [f"request: {_exc(e)}"]
    lines.append(f"request: elem_bytes={req.elem_bytes} threads={req.threads}")
    for name in ("q", "kt_stage", "dkv_out", "zz"):
        lines.append(f"swizzle_of {name}: {req.swizzle_of(name, 'dflt')}")
    try:
        plan = lds.plan_bwd_lds(req, arch=arch, strict=strict)
    except (KeyError, ValueError) as e:
        return lines + [f"plan: {_exc(e)}"]
    lines.append(
        f"plan: total={plan.total_bytes} resident={plan.resident_bytes} "
        f"capacity={plan.capacity_bytes} fits={plan.fits} "
        f"phase_offset={plan.phase_offset} unaliased={plan.unaliased_bytes}"
    )
    lines.append("plan phases: " + " ".join(f"{p}={n}" for p, n in plan.phase_bytes))
    for b in plan.buffers:
        lines.append(
            f"plan buffer: {b.name}[{b.slot}] offset={b.offset} nbytes={b.nbytes} "
            f"phases={','.join(b.phases)} layout={b.layout} resident={b.resident}"
        )
    for name, lay in plan.layouts:
        lines.append(f"plan layout: {name} = {lay!r}")
    for name in ("q", "ds", "dkv_out", "k_res", "nope"):
        lines.append(f"has {name}: {plan.has(name)}")
        _try(lines, f"offset {name}[1]", plan.offset, name, 1)
        lines.append(f"ring_offsets {name}: {plan.ring_offsets(name)!r}")
        _try(lines, f"ring_stride {name}", plan.ring_stride, name)
        _try(lines, f"layout {name}", _layout_repr, plan, name)
    for phase in lds.PHASES:
        lines.append(
            f"live {phase}: "
            + " ".join(f"{b.name}[{b.slot}]" for b in plan.live(phase))
        )
    _try(lines, "check_capacity", plan.check_capacity)
    bad = dataclasses.replace(plan, total_bytes=plan.total_bytes - 128)
    _try(lines, "check shrunk", bad.check)
    return lines


def _abi():
    lines = []
    for stage in ("main", "prep", "convert", "reduce", "dq", "bwd"):
        try:
            lines.append(
                f"params {stage}: "
                + " ".join(f"{n}:{f}" for n, f in attn_bwd_params(stage))
            )
        except ValueError as e:
            lines.append(f"params {stage}: {_exc(e)}")
    for sc in ("fold_ds", "dq_before_atomic", "convert", "runtime", "never"):
        for mode in ("direct", "atomic"):
            _try(
                lines,
                f"unit_mults {sc} {mode}",
                _unit_mults,
                sc,
                mode,
            )
    return lines


def dump(config: str) -> list:
    kind, _, arg = config.partition("|")
    if kind == "caps":
        return _caps(arg)
    if kind == "proofs":
        return _proofs(arg)
    if kind == "tiles":
        return _tiles(int(arg))
    if kind == "small":
        return _small(int(arg))
    if kind == "xt":
        return _xt(*map(int, arg.split(",")))
    if kind == "writer":
        return _writer(int(arg))
    if kind == "choose":
        return _choose(*map(int, arg.split(",")))
    if kind == "dma":
        return _dma(*map(int, arg.split(",")))
    if kind == "plan":
        arch, strict, text = arg.split("|", 2)
        return _plan(arch, strict == "1", text)
    if kind == "access":
        return _access()
    return _abi()


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: attention_bwd_layout_emit.py <config_index> [ll]\n")
        return 2
    idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
    if mode != "ll":
        sys.stderr.write(f"unknown mode {mode}\n")
        return 2
    if not 0 <= idx < len(_CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    config = _CONFIGS[idx]
    sys.stdout.write(f"config: {config}\n")
    sys.stdout.write("".join(line + "\n" for line in dump(config)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
