#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/attention_bwd_frag_emit.py -- Python reference emitter for the
# attention backward fragment and block-GEMM helpers
# (kernels.common._attention_bwd_frag and kernels.common._attention_bwd_gemm).
# The helpers have no kernel of their own, so this builds a small probe kernel
# per config that calls the helpers; the C twin attention_bwd_frag_emit.c
# builds the same kernel through the C++ helpers
# (helper_rocke.instances.common._attention_bwd_frag.h / _gemm.h) and the two
# .ll must match. Host-only answers (lane-map facts, relabel plans, transpose
# read plans, view alignment) are stored as i32 constants so they are compared
# too.
#
# A config is (kind, arch, dtype, fields):
#   facts   - atom facts, operand coordinates, relabel plans, K-permuted
#             coordinates and transpose-read plans of the arch's 16x16 atoms
#   gemm    - one BlockGemm: LDS / register operands, intrinsic or inline-asm
#             AGPR route, plus mma_frag, acc_coords, load_k_slice and
#             mma_k_sliced
#   relabel - G0 (S = Q K^T) -> relabel_acc_to_a -> G1 with the (K-permuted)
#             B coordinates, plain or transpose reads, intrinsic or asm route
#   views   - LdsView / DmaSlabView / XorLdsView index, load, store,
#             tr16_load, plus load_frag_lds / load_frag_lds_tr16 with int and
#             runtime origins
# The geometries come from the shipped defaults and the tuned tables of
# dispatch/attention_bwd (gfx942: 16 x 32 / 16 x 64 / 16 x 128 tiles, 2 or 4
# waves, 16x16x16 atoms, asm_g13; gfx950: 32 x 128 / 32 x 256 / 16 x 192
# tiles, 4 or 8 waves, 16x16x32 G0 / G4 atoms, tr_read, asm_g13), plus
# gfx1151 / gfx1201 WMMA for the xlane16 and gfx12 direct relabels. The
# caps answers (atom, wave size, legal waves, asm
# mnemonic, tail nop, lead wait, transpose read) are explicit inputs of the C
# API; the C emitter fills them from its own table.
#
# Config index = position in CONFIGS (keep in sync with the .c table; the
# list is printed by `attention_bwd_frag_emit.py list`).
#
# argv[2] selects the output: ll (default), ir, verify (see _emit_common.py).
import sys

from _emit_common import run_emit
from kernels.common._attention_bwd_caps import select_bwd_atom
from kernels.common._attention_bwd_frag import (
    DmaSlabView,
    LdsView,
    XorLdsView,
    atom_facts,
    c_quad_transposable,
    c_row_runs,
    check_tr16_coords,
    kperm_b_coords,
    load_frag_lds,
    load_frag_lds_tr16,
    operand_coords,
    plan_c_to_a,
    simulate_relabel,
    slab_offset,
    tr16_lane_address,
    tr16_model_coords,
    tr16_read_plan,
)
from kernels.common._attention_bwd_gemm import (
    BlockGemm,
    LdsOperand,
    check_relabel_grids,
    kperm_b_operand_coords,
    lane_and_wave,
    mma_agpr_asm_steps,
    relabel_acc_to_a,
)
from rocke.core.ir import BF16, F16, F32, I32, IRBuilder, PtrType

# --------------------------------------------------------------------------
# Configs
# --------------------------------------------------------------------------


def _gemm(arch, dtype, m, n, k, grid, atom_k, **kw):
    f = {
        "m": m,
        "n": n,
        "k": k,
        "grid": grid,
        "atom_k": atom_k,
        "a": "lds",
        "b": "inner",
        "mem": "alloc",
        "pad": 0,
        "route": "intrinsic",
        "cluster": "k_step",
        "zero": True,
        "tail_nop": None,
        "agpr_base": None,
        "lead_wait": None,
        "pre": False,
        "extra": False,
        "slice": None,
        "b_all": False,
        "src": None,
        "tid": False,
        "asm_steps": None,
        "first_s": 0,
    }
    f.update(kw)
    return ("gemm", arch, dtype, f)


def _relabel(arch, dtype, km, kn, d, waves, k02, k13, **kw):
    f = {
        "km": km,
        "kn": kn,
        "d": d,
        "waves": waves,
        "k02": k02,
        "k13": k13,
        "cast": "generic",
        "reader": "plain",
        "route": "intrinsic",
        "grid1": None,
    }
    f.update(kw)
    return ("relabel", arch, dtype, f)


def _facts(arch, dtype, atoms, pairs, tr16=()):
    return ("facts", arch, dtype, {"atoms": atoms, "pairs": pairs, "tr16": tr16})


def _views(arch, dtype, view, base, pitch, rows, **kw):
    f = {
        "view": view,
        "base": base,
        "pitch": pitch,
        "rows": rows,
        "scale": 1,
        "shift": False,
        "chunk": 4,
        "ns": (1, 2, 4, 8),
        "tr16": False,
        "frag": (),
    }
    f.update(kw)
    return ("views", arch, dtype, f)


CONFIGS = (
    # --- facts ------------------------------------------------------------
    _facts("gfx942", "fp16", (16,), ((16, 16),)),
    _facts("gfx942", "bf16", (16,), ((16, 16),)),
    _facts(
        "gfx950",
        "fp16",
        (16, 32),
        ((16, 16), (16, 32), (32, 32), (32, 16)),
        ((16, "a"), (16, "b"), (32, "a"), (32, "b")),
    ),
    _facts("gfx950", "bf16", (16, 32), ((16, 32), (32, 32)), ((32, "b"),)),
    _facts("gfx1151", "fp16", (16,), ((16, 16),)),
    _facts("gfx1201", "bf16", (16,), ((16, 16),)),
    # --- gemm: shipped default / tuned geometries --------------------------
    # gfx942 tuned: 16 x 32 tile, 2 waves, G0 (1, 2) over D 128
    _gemm("gfx942", "fp16", 16, 32, 128, (1, 2), 16),
    # gfx942 G1 (2, 1): kN0 x D over kM0, B k-outer plain, asm_g13
    _gemm("gfx942", "bf16", 32, 128, 16, (2, 1), 16, b="outer", route="asm"),
    # gfx942 4 waves, G4 (1, 4): dQ = dS K, K k-outer
    _gemm("gfx942", "fp16", 16, 128, 64, (1, 4), 16, b="outer", mem="view", pad=8),
    # gfx950 tuned: G0 32 x 128 over D 128 on 16x16x32, (1, 4)
    _gemm("gfx950", "bf16", 32, 128, 128, (1, 4), 32, mem="view", pad=8),
    # gfx950 G1 (4, 1), 16x16x16, tr_read K-outer B, asm_g13 k_step
    _gemm(
        "gfx950",
        "bf16",
        128,
        128,
        32,
        (4, 1),
        16,
        b="tr16",
        mem="view",
        route="asm",
        agpr_base=128,
    ),
    # gfx950 G4 16x16x32 tr16 with a gemm-wide asm cluster and zero acc
    _gemm(
        "gfx950",
        "fp16",
        32,
        64,
        64,
        (1, 4),
        32,
        b="tr16",
        mem="view",
        route="asm",
        cluster="gemm",
        zero=False,
        pre=True,
    ),
    # register operands: load_all_a + mma, load_operands + mma
    _gemm("gfx942", "fp16", 32, 32, 32, (2, 1), 16, a="regs_all", extra=True),
    _gemm(
        "gfx950", "fp16", 32, 64, 64, (2, 2), 32, a="regs_ops", b="outer", extra=True
    ),
    # (2, W/2) grid, padded rows (vector reads split), mma_frag / acc_coords
    _gemm("gfx950", "bf16", 64, 64, 32, (2, 2), 16, mem="view", pad=4, extra=True),
    # XOR and DMA-slab views as GEMM operands
    _gemm("gfx950", "fp16", 32, 64, 64, (1, 2), 16, mem="xor"),
    _gemm("gfx950", "bf16", 32, 64, 64, (1, 2), 32, mem="dma"),
    _gemm("gfx942", "bf16", 16, 64, 64, (1, 4), 16, mem="dma", b="outer"),
    # ring slot shift (runtime) on a view
    _gemm("gfx942", "fp16", 16, 32, 64, (1, 2), 16, mem="shift"),
    # asm knobs: tail nop, lead wait 0, one fragment (s_nop between steps)
    _gemm(
        "gfx942",
        "fp16",
        16,
        16,
        64,
        (1, 1),
        16,
        route="asm",
        tail_nop=6,
        lead_wait=0,
        zero=False,
    ),
    _gemm(
        "gfx950",
        "bf16",
        16,
        16,
        64,
        (1, 1),
        32,
        route="asm",
        cluster="gemm",
        lead_wait=4,
        tail_nop=0,
    ),
    _gemm("gfx950", "fp16", 32, 128, 64, (1, 4), 16, route="asm", pre=True, extra=True),
    # K slicing (g4_slicing): slice / slice_db / slice_db_g3 (first)
    _gemm("gfx942", "fp16", 16, 64, 64, (1, 2), 16, b="outer", slice=(32, 1, False)),
    _gemm(
        "gfx950",
        "bf16",
        32,
        128,
        128,
        (1, 4),
        32,
        b="tr16",
        mem="view",
        slice=(64, 2, False),
    ),
    _gemm("gfx950", "fp16", 32, 64, 128, (1, 4), 32, b="outer", slice=(32, 2, True)),
    _gemm("gfx942", "bf16", 16, 32, 64, (1, 2), 16, slice=(16, 1, True), extra=True),
    # wave32 WMMA
    _gemm("gfx1151", "fp16", 16, 64, 64, (1, 4), 16),
    _gemm("gfx1201", "bf16", 32, 32, 32, (2, 1), 16, b="outer", extra=True),
    # one fragment, gemm-wide cluster: s_nop between the K steps
    _gemm("gfx950", "fp16", 16, 16, 64, (1, 1), 32, route="asm", cluster="gemm"),
    # register B: load_all_b
    _gemm("gfx942", "bf16", 32, 32, 64, (2, 1), 16, b="outer", b_all=True, extra=True),
    # gfx950 8 waves (tuned): G0 (1, 8), G1 (8, 1) tr16, G4 (2, 4) sliced
    _gemm("gfx950", "bf16", 32, 128, 64, (1, 8), 32, mem="view", pad=8),
    _gemm("gfx950", "fp16", 128, 64, 32, (8, 1), 16, b="tr16", mem="view"),
    _gemm("gfx950", "bf16", 32, 128, 256, (2, 4), 32, b="outer", slice=(64, 2, False)),
    # gfx950 16 x 192 tile (tuned): G0 (1, 4) frags_n 3, G1 asm pinned
    _gemm("gfx950", "fp16", 16, 192, 128, (1, 4), 32, mem="view", pad=8),
    _gemm(
        "gfx950",
        "bf16",
        192,
        128,
        16,
        (4, 1),
        16,
        b="tr16",
        mem="view",
        route="asm",
        agpr_base=64,
    ),
    # gfx942 D 128, 16 x 128 tile, 4 waves (tuned): G1 (4, 1) asm pinned
    _gemm(
        "gfx942",
        "fp16",
        128,
        128,
        16,
        (4, 1),
        16,
        b="outer",
        route="asm",
        agpr_base=128,
    ),
    # LdsOperand mn0 / k0 / align through BlockGemm, on an alloc and a view
    _gemm("gfx942", "fp16", 16, 32, 32, (1, 2), 16, src=(16, 16, 4), extra=True),
    _gemm(
        "gfx950",
        "bf16",
        32,
        64,
        64,
        (1, 2),
        32,
        b="outer",
        mem="view",
        src=(8, 32, 2),
        slice=(32, 1, True),
    ),
    # lane_and_wave with a caller tid; mma_agpr_asm_steps fed by load_k_slice
    _gemm("gfx942", "bf16", 16, 32, 64, (1, 2), 16, tid=True, asm_steps="ok"),
    _gemm(
        "gfx950",
        "fp16",
        32,
        64,
        64,
        (2, 2),
        32,
        b="tr16",
        mem="view",
        route="asm",
        zero=False,
        asm_steps="ok",
    ),
    # --- relabel ------------------------------------------------------------
    _relabel("gfx942", "fp16", 16, 32, 128, 2, 16, 16),
    _relabel("gfx942", "bf16", 16, 64, 64, 4, 16, 16, cast="rne_noguard", route="asm"),
    _relabel("gfx950", "bf16", 32, 128, 128, 4, 32, 16, reader="tr16", route="asm"),
    _relabel("gfx950", "fp16", 32, 64, 64, 4, 16, 32),
    _relabel(
        "gfx950", "bf16", 32, 64, 64, 2, 32, 32, reader="tr16", cast="rne_noguard"
    ),
    _relabel("gfx950", "fp16", 32, 128, 64, 4, 16, 32, reader="tr16", route="asm"),
    _relabel("gfx1151", "fp16", 16, 32, 64, 2, 16, 16),
    _relabel("gfx1151", "bf16", 16, 64, 32, 4, 16, 16, cast="rne_noguard"),
    _relabel("gfx1201", "bf16", 16, 32, 64, 2, 16, 16),
    # gfx950 tuned: 8 waves (32 x 128 and 32 x 256), 16 x 192 asm; gfx942 asm
    _relabel("gfx950", "bf16", 32, 128, 128, 8, 32, 16, reader="tr16"),
    _relabel("gfx950", "fp16", 32, 256, 64, 8, 32, 16, reader="tr16"),
    _relabel("gfx950", "bf16", 16, 192, 128, 4, 32, 16, reader="tr16", route="asm"),
    _relabel("gfx942", "fp16", 16, 128, 128, 4, 16, 16, route="asm"),
    # --- views --------------------------------------------------------------
    _views("gfx950", "fp16", "plain", 0, 72, 32, frag=(("a", 16, "k_inner", 0, 0, 8),)),
    _views(
        "gfx942",
        "bf16",
        "plain",
        64,
        36,
        16,
        ns=(1, 2, 4, 8),
        frag=(("b", 16, "k_outer", "rt", 16, 8), ("a", 16, "k_inner", 16, "rt", 4)),
    ),
    _views("gfx950", "fp16", "f32", 128, 64, 16, scale=2, ns=(1, 2, 4)),
    _views("gfx950", "bf16", "plain", 256, 128, 32, shift=True, tr16=True),
    _views(
        "gfx950",
        "bf16",
        "dma",
        512,
        128,
        32,
        ns=(1, 4, 8),
        tr16=True,
        frag=(("b", 32, "tr16", 0, "rt", 0), ("a", 32, "k_inner", "rt", 0, 8)),
    ),
    _views("gfx950", "fp16", "dma", 0, 32, 16, ns=(1, 2, 8)),
    _views("gfx942", "fp16", "dma", 64, 64, 8, ns=(8,), shift=True),
    _views("gfx950", "fp16", "xor", 0, 64, 32, chunk=8, tr16=True),
    _views("gfx950", "bf16", "xor", 128, 32, 64, chunk=4, ns=(1, 4, 8)),
    _views("gfx942", "fp16", "xor", 64, 128, 16, chunk=2, ns=(2, 8)),
    _views(
        "gfx950",
        "fp16",
        "plain",
        0,
        64,
        64,
        tr16=True,
        frag=(("b", 16, "tr16", "rt", 4, 0), ("b", 32, "tr16", 8, 0, 0)),
    ),
    _views(
        "gfx1151", "fp16", "plain", 0, 40, 16, frag=(("a", 16, "k_inner", 0, 16, 8),)
    ),
    # --- rejected configs (both engines must refuse) ------------------------
    # tr16 reader on gfx942 (no ds_read_tr16_b64)
    _gemm("gfx942", "fp16", 16, 32, 64, (1, 2), 16, b="tr16", mem="view"),
    # illegal warp grid
    _gemm("gfx950", "fp16", 32, 64, 64, (4, 2), 16),
    # 16x16x32 atom on gfx942
    _gemm("gfx942", "bf16", 16, 32, 64, (1, 2), 32),
    # asm route on RDNA
    _gemm("gfx1151", "fp16", 16, 32, 32, (1, 2), 16, route="asm"),
    # pinned AGPRs past a255
    _gemm("gfx950", "fp16", 64, 64, 32, (1, 1), 16, route="asm", agpr_base=250),
    # K slice not a multiple of the atom K
    _gemm("gfx950", "bf16", 32, 64, 128, (1, 4), 32, slice=(16, 1, False)),
    # asm knob on the intrinsic route
    _gemm("gfx942", "fp16", 16, 32, 64, (1, 2), 16, tail_nop=7),
    # tail nop out of range
    _gemm("gfx942", "fp16", 16, 32, 64, (1, 2), 16, route="asm", tail_nop=16),
    # relabel with a (2, W/2) consumer grid
    _relabel("gfx950", "fp16", 32, 64, 64, 4, 16, 16, grid1=(2, 2)),
    # XOR view with a shift
    _views("gfx950", "fp16", "xor", 0, 64, 32, shift=True),
    # DMA slab view with a pitch that is not whole chunks
    _views("gfx950", "fp16", "dma", 0, 36, 32),
    # misaligned transpose-read column origin
    _views("gfx950", "fp16", "plain", 0, 64, 64, frag=(("b", 16, "tr16", 2, 0, 0),)),
    # unknown dtype
    _gemm("gfx942", "fp32", 16, 32, 64, (1, 2), 16),
    # tr16 reader on a K-inner tile (LdsOperand.__post_init__)
    _gemm("gfx950", "fp16", 32, 64, 64, (1, 2), 16, b="tr16_inner", mem="view"),
    # an asm step with one A fragment short
    _gemm("gfx950", "bf16", 32, 64, 64, (2, 2), 32, asm_steps="bad"),
    # lead wait out of range
    _gemm("gfx950", "fp16", 32, 64, 64, (1, 2), 16, route="asm", lead_wait=17),
    # first K slice index outside the slices
    _gemm("gfx942", "fp16", 16, 32, 64, (1, 2), 16, slice=(32, 1, True), first_s=2),
)


_RT = "rt"
_ELEM = {"fp16": F16, "f16": F16, "bf16": BF16}

# --------------------------------------------------------------------------
# Probe helpers
# --------------------------------------------------------------------------


class _Out:
    """One i32 and one f32 output buffer; every stored value is kept live."""

    def __init__(self, b):
        self.b = b
        self.oi = b.param("out_i", PtrType(I32, "global"))
        self.of = b.param("out_f", PtrType(F32, "global"))
        self.ni = 0
        self.nf = 0

    def i(self, v):
        b = self.b
        idx = b.const_i32(self.ni)
        self.ni += 1
        if isinstance(v, (bool, int)):
            v = b.const_i32(_wrap(int(v)))
        b.global_store(self.oi, idx, v)

    def f(self, v):
        b = self.b
        idx = b.const_i32(self.nf)
        self.nf += 1
        if v.type.name.startswith("vec"):
            v = b.vec_extract(v, 0)
        v = b.cast_to_f32(v)
        b.global_store(self.of, idx, v)


def _wrap(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v


def _coords_vals(out, fc):
    for dm, dk in fc.deltas:
        out.i(dm)
        out.i(dk)
    for lane in (0, 17, fc.wave_size - 1):
        for mn, k in fc.int_coords(lane):
            out.i(mn * 1000 + k)
    out.i(fc.kperm)


def _plan_vals(out, plan):
    out.i(plan.kind == "xlane16")
    out.i(plan.c_tiles)
    out.i(plan.c_tile_rows)
    for t, cs in plan.slot_src:
        out.i(t * 100 + cs)
    for j, s0, s1 in plan.xlane:
        out.i(j)
        out.i(s0)
        out.i(s1)
    for k in plan.k_perm:
        out.i(k)
    out.i(plan.needs_k_perm)


def _tr16_vals(out, plan):
    out.i(plan.kind == "stacked")
    out.i(plan.g_stride)
    for off in plan.k_offsets:
        out.i(off)


def _probe_facts(b, out, arch, dtype, f, lane):
    for k in f["atoms"]:
        op = select_bwd_atom(arch, dtype, k)
        fa = atom_facts(op)
        for x in (fa.a_contig, fa.b_contig, fa.affine, fa.c_is_aT, fa.ab_same_k):
            out.i(x)
        for run in (1, 2, 4, 8):
            out.i(c_row_runs(op, run))
        out.i(c_quad_transposable(op))
        for role in ("a", "b"):
            fc = operand_coords(op, role)
            _coords_vals(out, fc)
            mn, kk = fc.emit_base(b, lane)
            out.i(mn)
            out.i(kk)
    for ck, ak in f["pairs"]:
        c_op = select_bwd_atom(arch, dtype, ck)
        a_op = select_bwd_atom(arch, dtype, ak)
        plan = plan_c_to_a(c_op, a_op)
        _plan_vals(out, plan)
        sim = simulate_relabel(plan, c_op, a_op)
        acc = 0
        for li, row in enumerate(sim):
            for si, (r, kk) in enumerate(row):
                acc = (acc * 31 + li * 7 + si * 5 + r * 3 + kk) & 0x7FFFFFFF
        out.i(acc)
        if plan.needs_k_perm:
            kc = kperm_b_coords(plan, c_op, a_op)
            _coords_vals(out, kc)
            mn, kk = kc.emit_base(b, lane)
            out.i(mn)
            out.i(kk)
            if kc.wave_size == 64:
                _tr16_vals(out, tr16_read_plan(kc))
    for k, role in f["tr16"]:
        fc = operand_coords(select_bwd_atom(arch, dtype, k), role)
        out.i(check_tr16_coords(fc))
        _tr16_vals(out, tr16_read_plan(fc))
    acc = 0
    for kpl in (4, 8):
        for ln in range(64):
            for t in range(kpl // 4):
                kk, mn = tr16_lane_address(ln, t, kpl)
                acc = (acc * 17 + kk * 3 + mn) & 0x7FFFFFFF
            for mn, kk in tr16_model_coords(ln, kpl):
                acc = (acc * 13 + mn * 5 + kk) & 0x7FFFFFFF
    out.i(acc)


def _mem_view(b, mem, smem, base, pitch, rows, shift):
    if mem == "xor":
        return XorLdsView(smem, base, pitch, rows, chunk=4)
    if mem == "dma":
        return DmaSlabView(smem, base, pitch, rows)
    return LdsView(smem, base, pitch, rows, shift=shift if mem == "shift" else None)


def _align64(x: int) -> int:
    return (x + 63) // 64 * 64


def _probe_gemm(b, out, arch, dtype, f, tid, lane, wave):
    elem = _ELEM.get(dtype, F16)
    g = BlockGemm(arch, dtype, f["m"], f["n"], f["k"], f["grid"], f["atom_k"])
    origin = g.wave_origin(b, wave)
    m, n, k = f["m"], f["n"], f["k"]
    bmode = f["b"]
    inner = bmode in ("inner", "tr16_inner")
    b_storage = "k_inner" if inner else "k_outer"
    b_rows, b_cols = (n, k) if inner else (k, n)
    mem = f["mem"]
    so = {}
    if f["src"] is not None:
        so = dict(zip(("mn0", "k0", "align"), f["src"]))
    if mem == "alloc":
        e_mn, e_k = so.get("mn0", 0), so.get("k0", 0)
        sa = b.smem_alloc(elem, (m + e_mn, k + e_k), "lds_a")
        if inner:
            b_shape = (b_rows + e_mn, b_cols + e_k)
        else:
            b_shape = (b_rows + e_k, b_cols + e_mn)
        sb = b.smem_alloc(elem, b_shape, "lds_b")
        a_src = LdsOperand(sa, **so)
        b_src = LdsOperand(sb, b_storage, **so)
    else:
        shift = b.param("shift", I32) if mem == "shift" else None
        pa = k + f["pad"]
        pb = b_cols + f["pad"]
        base_b = _align64(m * pa)
        total = base_b + _align64(b_rows * pb)
        smem = b.smem_alloc(elem, (total,), "bwd_lds")
        va = _mem_view(b, mem, smem, 0, pa, m, shift)
        vb = _mem_view(b, mem, smem, base_b, pb, b_rows, shift)
        out.i(va.align_elems)
        out.i(vb.align_elems)
        a_src = LdsOperand(va, **so)
        reader = "tr16" if bmode in ("tr16", "tr16_inner") else "plain"
        b_src = LdsOperand(vb, b_storage, reader=reader, **so)
    a_op, b_op = a_src, b_src
    if f["a"] == "regs_all":
        a_op = g.load_all_a(b, a_src, lane, origin[0])
    elif f["a"] == "regs_ops":
        a_op, b_op = g.load_operands(b, a_src, b_src, lane=lane, origin=origin)
    if f["b_all"]:
        b_op = g.load_all_b(b, b_src, lane, origin[1])
    kw = {}
    if f["route"] == "asm":
        kw["route"] = "agpr_asm"
        kw["asm_cluster"] = f["cluster"]
        if f["agpr_base"] is not None:
            kw["asm_agpr_base"] = f["agpr_base"]
        if f["lead_wait"] is not None:
            kw["asm_lead_wait"] = f["lead_wait"]
    if f["tail_nop"] is not None:
        kw["asm_tail_nop"] = f["tail_nop"]
    if f["pre"]:

        def pre():
            out.i(lane)

        kw["pre_issue"] = pre
    acc0 = g.zero_acc(b) if f["zero"] else None
    acc = g.mma(b, acc0, a_op, b_op, lane=lane, origin=origin, **kw)
    for row in acc:
        for v in row:
            out.f(v)
    fm, fn = g.frags_m - 1, g.frags_n - 1
    if f["extra"]:
        fr = g.mma_frag(b, acc[fm][fn], a_op, b_op, fm, fn, lane=lane, origin=origin)
        out.f(fr)
        for r, c in g.acc_coords(b, lane, origin, fm, fn):
            out.i(r)
            out.i(c)
        out.i(g.threads)
        out.i(g.mma_per_wave)
    if f["slice"] is not None:
        k_slice, buffers, first = f["slice"]
        sl = None
        if first:
            s0 = f["first_s"]
            sl = g.load_k_slice(b, a_op, b_op, s0, k_slice, lane=lane, origin=origin)
        acc2 = g.mma_k_sliced(
            b, g.zero_acc(b), a_op, b_op, k_slice=k_slice, buffers=buffers, lane=lane,
            origin=origin, first=sl,
        )  # fmt: skip
        for row in acc2:
            for v in row:
                out.f(v)
    if f["asm_steps"] is not None:
        steps = g.load_k_slice(b, a_op, b_op, 0, g.k, lane=lane, origin=origin)
        if f["asm_steps"] == "bad":
            steps[0] = (steps[0][0][:-1], steps[0][1])
        acc3 = mma_agpr_asm_steps(g, b, acc, steps)
        for row in acc3:
            for v in row:
                out.f(v)


def _probe_relabel(b, out, arch, dtype, f, tid, lane, wave):
    elem = _ELEM[dtype]
    km, kn, d, w = f["km"], f["kn"], f["d"], f["waves"]
    g0 = BlockGemm(arch, dtype, km, kn, d, (1, w), f["k02"])
    grid1 = f["grid1"] or (w, 1)
    g1 = BlockGemm(arch, dtype, kn, d, km, grid1, f["k13"])
    o0 = g0.wave_origin(b, wave)
    o1 = g1.wave_origin(b, wave)
    base_k = _align64(km * d)
    base_do = base_k + _align64(kn * d)
    smem = b.smem_alloc(elem, (base_do + _align64(km * d),), "bwd_lds")
    vq = LdsView(smem, 0, d, km)
    vk = LdsView(smem, base_k, d, kn)
    vdo = LdsView(smem, base_do, d, km)
    s = g0.mma(b, g0.zero_acc(b), LdsOperand(vq), LdsOperand(vk), lane=lane, origin=o0)
    plan = check_relabel_grids(g0, g1)
    a_p, plan2 = relabel_acc_to_a(b, g0, g1, s, lane, bf16_cast=f["cast"])
    _plan_vals(out, plan2)
    out.i(plan == plan2)
    coords = kperm_b_operand_coords(plan, g0, g1)
    _coords_vals(out, coords)
    src = LdsOperand(vdo, "k_outer", coords=coords, reader=f["reader"])
    kw = {"route": "agpr_asm"} if f["route"] == "asm" else {}
    acc0 = None if f["route"] == "asm" else g1.zero_acc(b)
    dv = g1.mma(b, acc0, a_p, src, lane=lane, origin=o1, **kw)
    for row in dv:
        for v in row:
            out.f(v)


def _iv(x, rt):
    return rt if x == _RT else x


def _probe_views(b, out, arch, dtype, f, tid, lane, wave):
    elem = _ELEM[dtype]
    r_rt = b.param("row", I32)
    c_rt = b.param("col", I32)
    shift = b.param("shift", I32) if f["shift"] else None
    base, pitch, rows = f["base"], f["pitch"], f["rows"]
    size = _align64(base + pitch * rows * f["scale"]) + 64
    smem = b.smem_alloc(elem, (size,), "bwd_lds")
    kind = f["view"]
    if kind == "xor":
        v = XorLdsView(smem, base, pitch, rows, shift=shift, chunk=f["chunk"])
    elif kind == "dma":
        v = DmaSlabView(smem, base, pitch, rows, shift=shift)
    else:
        v = LdsView(smem, base, pitch, rows, f["scale"], shift)
    out.i(v.align_elems)
    vdt = F32 if f["scale"] == 2 else elem
    pts = ((0, 0), (1, 8), (rows - 1, 0), (2, _RT), (_RT, 0), (_RT, 16), (_RT, _RT))
    for r, c in pts:
        out.i(v.index(b, _iv(r, r_rt), _iv(c, c_rt)))
    if kind == "xor":
        for r in (0, 1, 5, rows - 1):
            out.i(v.row_mask(r))
        out.i(v.emit_index(b, r_rt, 8))
    if kind == "dma":
        for r, c in ((0, 0), (rows - 1, pitch - 1), (3, 17)):
            out.i(slab_offset(rows, pitch, r, c))
    for n in f["ns"]:
        for r, c in ((1, 0), (_RT, 8), (2, _RT)):
            x = v.load(b, _iv(r, r_rt), _iv(c, c_rt), dtype=vdt, n=n)
            out.f(x)
            v.store(b, _iv(r, r_rt), _iv(c, c_rt), x, n)
    if f["tr16"]:
        out.f(v.tr16_load(b, r_rt, 4, dtype=elem))
        out.f(v.tr16_load(b, 3, c_rt, dtype=elem))
    for role, atom_k, storage, mn0, k0, align in f["frag"]:
        fc = operand_coords(select_bwd_atom(arch, dtype, atom_k), role)
        if storage == "tr16":
            x = load_frag_lds_tr16(
                b,
                fc,
                v,
                lane,
                dtype=dtype,
                arch=arch,
                mn0=_iv(mn0, c_rt),
                k0=_iv(k0, r_rt),
            )
        else:
            x = load_frag_lds(
                b, fc, v, lane, dtype=dtype, mn0=_iv(mn0, c_rt), k0=_iv(k0, r_rt),
                storage=storage, align=align,
            )  # fmt: skip
        out.f(x)
        if storage != "tr16":
            s2 = b.smem_alloc(elem, (64, 64), "lds_2d")
            y = load_frag_lds(
                b, fc, s2, lane, dtype=dtype, mn0=_iv(mn0, c_rt), k0=_iv(k0, r_rt),
                storage=storage, align=align,
            )  # fmt: skip
            out.f(y)


_PROBES = {
    "facts": None,
    "gemm": _probe_gemm,
    "relabel": _probe_relabel,
    "views": _probe_views,
}


def _spec(idx: int):
    if not 0 <= idx < len(CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    return idx, CONFIGS[idx][1]


def _wave_size(arch: str) -> int:
    return 32 if arch.startswith(("gfx11", "gfx12")) else 64


def build_attention_bwd_frag_kernel(idx, *, arch: str):
    """The probe kernel of config ``idx``."""
    kind, _arch, dtype, f = CONFIGS[idx]
    b = IRBuilder("attn_bwd_frag_probe")
    out = _Out(b)
    tid_in = b.param("tid", I32) if kind == "gemm" and f["tid"] else None
    tid, lane, wave = lane_and_wave(b, _wave_size(arch), tid=tid_in)
    if kind == "facts":
        _probe_facts(b, out, arch, dtype, f, lane)
    else:
        _PROBES[kind](b, out, arch, dtype, f, tid, lane, wave)
    b.ret()
    return b.kernel


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "list":
        for i, (kind, arch, dtype, f) in enumerate(CONFIGS):
            print(f"{i:3d} {kind:8s} {arch:8s} {dtype:5s} {f}")
        return 0
    return run_emit(
        _spec,
        build_attention_bwd_frag_kernel,
        usage="usage: attention_bwd_frag_emit.py <config_index> [ll|ir|verify]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
