# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""XT transposes (``transpose_source = "xt_lds"``) and the ``xor`` LDS swizzle.

CPU checks of the layout-only levers:

* the swizzled index the body emits (``XorLdsView``, both the runtime-row
  program and the build-time fold) is the planner's ``XtLayout`` /
  ``RowXorLayout`` offset, a bijection that keeps every chunk contiguous;
* an LDS simulator runs the real emitters (row copy, ``write_xt``, the block
  GEMM operand loads) on symbolic element labels for every thread and lane:
  the G1 / G3 / G4 B fragments read from the XT images, and every operand read
  from an ``xor`` row-major buffer, deliver exactly the catalog coordinates of
  the plain reader (so the MMAs see the same operands in the same order);
* bank-conflict prediction: the B reads (and the per-step writer stores) the
  kernels emit on the XOR XT images are conflict-free in the production
  predictor on gfx942 and gfx950, while the same accesses on the unswizzled
  image conflict;
* build-time: the values are built (no ``_not_built`` entry), ``xt_lds``
  leaves no single-element LDS read in the q loop, the spelled-out default
  swizzle is the default kernel, and the illegal couplings stay refused.
"""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager

import pytest
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower

from kernels.common._attention_bwd_body import BwdLds, BwdTileGeometry, write_xt
from kernels.common._attention_bwd_frag import (
    IntLaneBuilder,
    XorLdsView,
    ir_type,
    operand_coords,
)
from kernels.common._attention_bwd_gemm import LdsOperand
from kernels.common._attention_bwd_lds import (
    RowXorLayout,
    XtLayout,
    _predict_max,
    choose_xt_swizzle,
    default_xt_writer,
    xt_conflicts,
)
from kernels.common.attention_bwd import (
    INERT_KNOBS,
    AttnBwdSpec,
    _not_built,
    attn_bwd_arch_facts,
    build_attn_bwd_main,
    lds_plan,
    validate_attn_bwd_spec,
)

_INT = IntLaneBuilder()
ARCHS = ("gfx942", "gfx950", "gfx1151", "gfx1201")
CDNA = ("gfx942", "gfx950")
# Every row-major buffer of the planner with the xor swizzle.
ROW_XOR = (
    "dkv_out=xor,do=xor,ds=xor,k_res=xor,k_stage=xor,pt=xor,q=xor,v_res=xor,"
    "v_stage=xor"
)


def _xt_legal(arch: str) -> bool:
    return "xt_lds" in attn_bwd_arch_facts(arch).legal_transpose_sources


# ---------------------------------------------------------------------------
# the emitted index is the planner's layout
# ---------------------------------------------------------------------------

XOR_SHAPES = sorted(
    {(d, s) for d in (32, 64, 128) for s in (16, 32, 64, 128, 192, 256)}
    | {(s, d) for d in (32, 64, 128) for s in (16, 32, 64, 128)}
)


def _layouts(rows, cols):
    out = []
    for lay in (XtLayout(rows, cols, 2, "xor"), RowXorLayout(rows, cols)):
        try:
            lay.offset(0, 0)
        except ValueError:
            continue
        out.append(lay)
    return out


@pytest.mark.parametrize("rows,cols", XOR_SHAPES)
def test_xor_view_index_is_the_planner_offset(rows, cols):
    for lay in _layouts(rows, cols):
        for base in (0, 64 * 3):
            view = XorLdsView(
                None, base, cols, rows, chunk=lay.chunk_bytes // 2
            )  # smem unused
            seen = set()
            for r in range(rows):
                for c in range(cols):
                    want = base + lay.offset(r, c) // 2
                    # runtime row: the emitted program, evaluated on ints
                    assert view.emit_index(_INT, r, c) == want
                    # build-time row: the folded mask
                    assert view.index(_INT, r, c) == want
                    seen.add(want)
            assert seen == set(range(base, base + rows * cols))
            # a chunk stays contiguous, so a chunk-wide access is one vector
            ch = view.chunk
            assert view.align_elems <= ch
            for r in range(rows):
                for c in range(0, cols, ch):
                    first = view.emit_index(_INT, r, c)
                    assert first % ch == 0
                    assert [view.emit_index(_INT, r, c + i) for i in range(ch)] == [
                        first + i for i in range(ch)
                    ]


def test_xor_view_rejects_shapes_the_planner_rejects():
    with pytest.raises(ValueError):
        XorLdsView(None, 0, 48, 16, chunk=4)  # 96-byte rows: no line period
    with pytest.raises(ValueError):
        XorLdsView(None, 32, 64, 16, chunk=4)  # origin not on a 128-byte line
    with pytest.raises(ValueError):
        XorLdsView(None, 0, 64, 16, chunk=4, scale=2)
    with pytest.raises(ValueError):
        RowXorLayout(16, 48)  # same rule in the planner
    with pytest.raises(ValueError):
        XtLayout(48, 48, 2, "xor")


# ---------------------------------------------------------------------------
# LDS simulator: the real emitters on symbolic element labels
# ---------------------------------------------------------------------------


class Sym:
    """A runtime i32 / i1 of the simulator (not an ``int``, so every emitter
    takes its runtime-value path)."""

    __slots__ = ("v",)

    def __init__(self, v):
        self.v = int(v)


def _v(x):
    return x.v if isinstance(x, Sym) else x


class _VecType:
    def __init__(self, count):
        self.count = count
        self.elem = None


class Vec(tuple):
    """A vector of element labels (``.type.count`` as an IR vector has)."""

    @property
    def type(self):
        return _VecType(len(self))


class LdsSim:
    """Duck-typed ``IRBuilder`` for the LDS emitters: integer ops on
    :class:`Sym`, vectors as tuples of labels, one dict per allocation.

    Every vector access must be naturally aligned (the lowering's claim) and a
    load of an unwritten cell returns ``None``; stores inside a false
    ``scf_if`` are dropped.
    """

    def __init__(self):
        self.mem = {}
        self.dead = 0
        self.loads = []  # (index, n) of every load, for the access checks

    # integer ops
    def const_i32(self, v):
        return Sym(v)

    def _bin(f):
        def op(self, a, b):
            return Sym(f(_v(a), _v(b)))

        return op

    add = _bin(lambda a, b: a + b)
    sub = _bin(lambda a, b: a - b)
    mul = _bin(lambda a, b: a * b)
    div = _bin(lambda a, b: a // b)
    mod = _bin(lambda a, b: a % b)
    xor = _bin(lambda a, b: a ^ b)
    lshr = _bin(lambda a, b: a >> b)
    shl = _bin(lambda a, b: a << b)
    land = _bin(lambda a, b: a & b)
    cmp_lt = _bin(lambda a, b: int(a < b))
    cmp_ge = _bin(lambda a, b: int(a >= b))
    del _bin

    @contextmanager
    def scf_if(self, cond):
        live = bool(_v(cond))
        self.dead += 0 if live else 1
        try:
            yield
        finally:
            self.dead -= 0 if live else 1

    # vectors
    def vec_extract(self, vec, i):
        return vec[i]

    def vec_pack(self, elems, elem):
        return Vec(elems)

    def vec_concat(self, a, b):
        return Vec(tuple(a) + tuple(b))

    # LDS
    def smem_alloc(self, elem, shape, name="smem", exclusive=False):
        return name

    def smem_load_vN(self, smem, idx, *, dtype, n):
        i = _v(idx)
        assert n == 1 or i % n == 0, f"misaligned {n}-element load at {i}"
        self.loads.append((i, n))
        return Vec(self.mem.get(i + k) for k in range(n))

    def smem_store_vN(self, smem, indices, value, n):
        (idx,) = indices
        i = _v(idx)
        assert n == 1 or i % n == 0, f"misaligned {n}-element store at {i}"
        vals = (value,) if n == 1 else tuple(value)
        assert len(vals) == n
        if not self.dead:
            for k, x in enumerate(vals):
                self.mem[i + k] = x


def _fill(b, view, rows, cols, tag, threads):
    """Fill a row-major view through the staging copy (``vec``-wide stores);
    element ``(r, c)`` holds the label ``(tag, r, c)``."""
    from kernels.common._attention_bwd_body import _copy_rows

    vec = 8 if cols % 8 == 0 else 1

    def load(r, c):
        r, c = _v(r), _v(c)
        return tuple((tag, r, c + i) for i in range(vec))

    for t in range(threads):
        _copy_rows(
            b, load, view, rows=rows, cols=cols, tid=Sym(t), threads=threads, vec=vec
        )


def _b_fragments(b, gemm, src, lane_count):
    """``{(wave, lane, fn, ks): tuple}`` of every B fragment a block GEMM reads."""
    out = {}
    for w in range(gemm.waves):
        _r0, c0 = gemm.wave_origin(b, Sym(w))
        for lane in range(lane_count):
            for ks in range(gemm.k_steps):
                frags = gemm.load_b(b, src, Sym(lane), c0, ks)
                for fn, f in enumerate(frags):
                    out[(w, lane, fn, ks)] = f
    return out


def _expected_b(gemm, coords, tag, lane_count):
    """The catalog B fragments of a ``[K][N]`` operand labelled ``(tag, k, n)``."""
    b = _INT
    out = {}
    for w in range(gemm.waves):
        _r0, c0 = gemm.wave_origin(b, w)
        for lane in range(lane_count):
            for ks in range(gemm.k_steps):
                for fn in range(gemm.frags_n):
                    mn0 = c0 + fn * gemm.op.n
                    k0 = ks * gemm.atom_k
                    out[(w, lane, fn, ks)] = tuple(
                        (tag, k + k0, mn + mn0) for mn, k in coords.int_coords(lane)
                    )
    return out


def _sim_specs():
    out = []
    for arch in ARCHS:
        wmma = attn_bwd_arch_facts(arch).matrix_path == "wmma"
        for d in (32, 64, 128):
            kws = [dict(lds_swizzle=ROW_XOR)]
            if _xt_legal(arch):
                kws += [
                    dict(transpose_source="xt_lds"),
                    dict(transpose_source="xt_lds", lds_swizzle=ROW_XOR),
                    dict(
                        transpose_source="xt_lds",
                        lds_swizzle="dot=pad16,kt_res=none,kt_stage=pad8,qt=pad32",
                    ),
                ]
                if not wmma:
                    kws += [
                        dict(transpose_source="xt_lds", pt_route="lds"),
                        dict(transpose_source="xt_lds", kt_source="lds"),
                        dict(transpose_source="xt_lds", stage_vec=1),
                        # gfx950: K-permuted 16x16x32 G1 / G3 B (relabel)
                        dict(transpose_source="xt_lds", atom_g13="16x16x32"),
                    ]
            for i, kw in enumerate(kws):
                try:  # LDS budget: a padded image may not fit (d128 on gfx942)
                    validate_attn_bwd_spec(AttnBwdSpec(head_size=d, **kw), arch)
                except ValueError:
                    continue
                out.append(pytest.param(arch, d, kw, id=f"{arch}-d{d}-{i}"))
    return out


@pytest.mark.parametrize("arch,d,kw", _sim_specs())
def test_simulated_operands_equal_the_catalog_coordinates(arch, d, kw):
    s = validate_attn_bwd_spec(AttnBwdSpec(head_size=d, dtype="bf16", **kw), arch)
    assert not _not_built(s)
    g = BwdTileGeometry.from_spec(s, arch)
    plan = lds_plan(s, arch)
    b = LdsSim()
    lds = BwdLds(b, plan, s.dtype)
    m0, n0, threads, ws = s.block_m, s.block_n, g.threads, g.wave_size
    elem = ir_type(s.dtype)
    v_q, v_do, v_ds = lds.view("q", d), lds.view("do", d), lds.view("ds", n0)
    v_k = lds.view("k_res", d) if lds.has("k_res") else lds.view("k_stage", d)
    for name in ("q", "do", "ds", "k_stage", "k_res"):
        if not lds.has(name):
            continue
        want = plan.buffer(name).layout
        swz = dict(i.split("=") for i in (s.lds_swizzle or "").split(",") if i)
        assert want == f"rowmajor:{swz.get(name, 'none')}"
        if want == "rowmajor:xor":
            view = {"q": v_q, "do": v_do, "ds": v_ds}.get(name, v_k)
            assert isinstance(view, XorLdsView)
    xt = g.reader == "xt"
    coords13 = g.b13_coords
    rd = g.reader
    if rd == "tr16":
        # the transpose read is a cross-lane instruction: its per-lane 8-byte
        # groups stay whole under every layout (the index test); the plain
        # reader shares its addressing
        rd = "plain"
    # stage 0 (K phase): K staged, K^T image written, G4 B read
    _fill(b, v_k, n0, d, "k", threads)
    if xt:
        v_qt, v_dot = lds.view("qt", m0), lds.view("dot", m0)
        kt_name = "kt_res" if lds.has("kt_res") else "kt_stage"
        v_kt = lds.view(kt_name, n0)
        for view, name in ((v_qt, "qt"), (v_dot, "dot"), (v_kt, kt_name)):
            sw = plan.layout(name).swizzle
            assert isinstance(view, XorLdsView) == (sw == "xor"), (name, sw)
        for t in range(threads):
            write_xt(
                b, v_k, v_kt, seq=n0, depth=d, tid=Sym(t), threads=threads, elem=elem
            )
        b4 = LdsOperand(v_kt, "k_inner")
        resident_k = kt_name == "kt_res"
    else:
        b4 = LdsOperand(v_k, "k_outer", reader=rd)
        resident_k = lds.has("k_res")
    want4 = _expected_b(g.g4, operand_coords(g.g4.op, "b"), "k", ws)
    assert _b_fragments(b, g.g4, b4, ws) == want4
    # q loop phase (aliases stage 0): Q / dO / dS staged, Q^T / dO^T written
    _fill(b, v_q, m0, d, "q", threads)
    _fill(b, v_do, m0, d, "do", threads)
    _fill(b, v_ds, m0, n0, "ds", threads)
    if resident_k:  # a resident K^T source survives the loop phase
        assert _b_fragments(b, g.g4, b4, ws) == want4
    # G0 / G2 A (Q, dO) and G4 A (dS): K-inner reads of the row-major tiles
    for tile, view, gemm in (("q", v_q, g.g0), ("ds", v_ds, g.g4)):
        coords = operand_coords(gemm.op, "a")
        for w in range(gemm.waves):
            r0, _c0 = gemm.wave_origin(b, Sym(w))
            for lane in range(ws):
                for ks in range(gemm.k_steps):
                    got = gemm.load_a(b, LdsOperand(view), Sym(lane), r0, ks)
                    for fm, f in enumerate(got):
                        mn0 = _v(r0) + fm * gemm.op.m
                        assert f == tuple(
                            (tile, mn + mn0, k + ks * gemm.atom_k)
                            for mn, k in coords.int_coords(lane)
                        ), (tile, w, lane, ks, fm)
    if xt:
        for t in range(threads):
            kw_ = dict(tid=Sym(t), threads=threads, elem=elem)
            write_xt(b, v_q, v_qt, seq=m0, depth=d, **kw_)
            write_xt(b, v_do, v_dot, seq=m0, depth=d, **kw_)
        b13 = {
            "q": LdsOperand(v_qt, "k_inner", coords=coords13),
            "do": LdsOperand(v_dot, "k_inner", coords=coords13),
        }
    else:
        b13 = {
            t: LdsOperand(v, "k_outer", coords=coords13, reader=rd)
            for t, v in (("q", v_q), ("do", v_do))
        }
    for tile, src in b13.items():
        got = _b_fragments(b, g.g1, src, ws)
        assert got == _expected_b(g.g1, coords13, tile, ws), tile
    if xt:
        # every XT B read is a vector of at least 4 elements (one 8-byte read)
        b.loads.clear()
        _b_fragments(b, g.g1, b13["q"], ws)
        assert min(n for _i, n in b.loads) >= min(4, g.g1.op.b_frag_len)


# ---------------------------------------------------------------------------
# bank-conflict prediction of the emitted accesses
# ---------------------------------------------------------------------------


def _index(view, row, col):
    """The emitted index of ``(row, col)`` evaluated on ints (the runtime-row
    program of an XOR view)."""
    if isinstance(view, XorLdsView):
        return view.emit_index(_INT, row, col)
    return view.index(_INT, row, col)


_READ = {4: "ds_read_b32", 8: "ds_read_b64", 16: "ds_read_b128"}
_WRITE = {4: "ds_write_b32", 8: "ds_write_b64", 16: "ds_write_b128"}


def _read_instrs(view, gemm, coords):
    """``[(width, [(lane, byte, width)])]``: one entry per LDS read instruction
    of the B fragments of every wave, from the emitted index of ``view``."""
    from math import gcd

    from kernels.common._attention_bwd_frag import _runs

    out = []
    runs = _runs(coords.deltas, 1)  # K-inner: runs along K
    # load_frag_lds's width: the alignment of every lane's run starts
    g = gcd(8, view.align_elems)
    for lane in range(coords.wave_size):
        cc = coords.int_coords(lane)
        for r in runs:
            g = gcd(g, cc[r[0]][1])
    (piece,) = {
        next(w for w in (8, 4, 2, 1) if w <= len(r) and g % w == 0 and len(r) % w == 0)
        for r in runs
    }
    for w in range(gemm.waves):
        _r0, c0 = gemm.wave_origin(_INT, w)
        for ks in range(gemm.k_steps):
            for fn in range(gemm.frags_n):
                for r in runs:
                    for p in range(0, len(r), piece):
                        acc = []
                        for lane in range(gemm.wave_size):
                            mn, k = coords.int_coords(lane)[r[p]]
                            row = mn + c0 + fn * gemm.op.n
                            col = k + ks * gemm.atom_k
                            acc.append((lane, 2 * _index(view, row, col), 2 * piece))
                        out.append((2 * piece, acc))
    return out


def _writer_instrs(view, seq, depth, threads):
    w = default_xt_writer(seq, depth, threads)
    out = []
    for it in range(w.iterations):
        for wave in range(-(-threads // 64)):
            for j in range(w.vec_d):
                acc = []
                for lane in range(64):
                    co = w.unit_coords(wave * 64 + lane, it)
                    if co is None:
                        continue
                    s0, d0 = co
                    acc.append((lane, 2 * _index(view, d0 + j, s0), w.write_bytes))
                if acc:
                    out.append((w.write_bytes, acc))
    return out


def _worst(arch, instrs, table):
    return max(_predict_max(arch, table[wd], acc) for wd, acc in instrs)


def _start_points():
    pts = []
    for arch in CDNA:
        for d in (32, 64, 128):
            for kt in (None, "lds"):
                pts.append((arch, d, kt))
    return pts


_IMAGES = ("qt", "dot", "kt_res", "kt_stage")


def _image_specs(s, sw):
    """``s`` with every XT image at swizzle ``sw`` (``None``: the planner's)."""
    if sw is None:
        return s
    return dataclasses.replace(s, lds_swizzle=",".join(f"{n}={sw}" for n in _IMAGES))


@pytest.mark.parametrize("arch,d,kt", _start_points())
def test_emitted_xt_accesses_are_the_conflict_model(arch, d, kt):
    """Bank-conflict prediction of the XT images, from the accesses the kernel
    emits: every B-fragment read (G1 / G3 on Q^T / dO^T, G4 on K^T, each LDS
    instruction of every wave) and every writer store, addressed through the
    emitted index, run through the production predictor.

    * the emitted accesses predict exactly what the layout model
      (``xt_conflicts``, which ranks the swizzles) predicts, for ``xor``,
      ``none`` and the planner's choice;
    * ``xor`` is conflict-free on every gfx942 start point (reads and the
      per-step writer stores), and on gfx950 for the per-step images; the
      packed unswizzled image conflicts (the check is not vacuous);
    * the planner's choice has the fewest predicted read conflicts of every
      candidate (gfx950 K^T: a row pad, the XOR hash being 2-way there).
    """
    pytest.importorskip("rocke.analysis.lds")
    kw = dict(head_size=d, transpose_source="xt_lds")
    if kt:
        kw["kt_source"] = kt
    base = validate_attn_bwd_spec(AttnBwdSpec(**kw), arch)
    g = BwdTileGeometry.from_spec(base, arch)
    m0, n0 = base.block_m, base.block_n
    seen = {}
    for sw in ("xor", "none", None):
        s = validate_attn_bwd_spec(_image_specs(base, sw), arch)
        plan = lds_plan(s, arch)
        lds = BwdLds(LdsSim(), plan, s.dtype)
        kt_name = "kt_res" if lds.has("kt_res") else "kt_stage"
        images = (
            ("qt", m0, g.g1, g.b13_coords),
            ("dot", m0, g.g1, g.b13_coords),
            (kt_name, n0, g.g4, operand_coords(g.g4.op, "b")),
        )
        for name, seq, gemm, coords in images:
            lay = plan.layout(name)
            assert sw is None or lay.swizzle == sw
            view = lds.view(name, seq)
            assert isinstance(view, XorLdsView) == (lay.swizzle == "xor")
            got = {
                "read": _worst(arch, _read_instrs(view, gemm, coords), _READ),
                "write": _worst(arch, _writer_instrs(view, seq, d, g.threads), _WRITE),
            }
            writer = default_xt_writer(seq, d, g.threads)
            assert got == xt_conflicts(lay, writer, arch, atom_k=gemm.atom_k), (
                name,
                lay.swizzle,
            )
            seen[(name, sw)] = got
            if sw is None:
                _best, table = choose_xt_swizzle(
                    d, seq, writer, arch, atom_k=gemm.atom_k
                )
                reads = [
                    v["prediction"]["read"] for v in table.values() if "prediction" in v
                ]
                assert got["read"] == min(reads), (name, lay.swizzle)
    for name in ("qt", "dot"):
        assert seen[(name, "xor")] == {"read": 1, "write": 1}, name
        assert seen[(name, "none")]["read"] > 1, name
    kt_name = next(n for n, _sw in seen if n.startswith("kt"))
    assert seen[(kt_name, "none")]["read"] > 1
    if arch == "gfx942":
        assert seen[(kt_name, "xor")]["read"] == 1
        assert seen[(kt_name, None)] == seen[(kt_name, "xor")]


# ---------------------------------------------------------------------------
# build-time facts
# ---------------------------------------------------------------------------


def test_values_are_built():
    for arch in ARCHS:
        kws = [dict(lds_swizzle=ROW_XOR)]
        if _xt_legal(arch):
            kws += [dict(transpose_source="xt_lds"), dict(lds_swizzle="qt=xor")]
        for kw in kws:
            try:
                s = validate_attn_bwd_spec(AttnBwdSpec(head_size=64, **kw), arch)
            except ValueError as exc:
                # the one-wave gfx1151 d64 tile spills with the XT images
                # (LAYOUT_SPILLING_COMBOS); d128 carries the check there
                assert "LAYOUT_SPILLING_COMBOS" in str(exc), (arch, kw, exc)
                s = validate_attn_bwd_spec(AttnBwdSpec(head_size=128, **kw), arch)
            assert not _not_built(s), (arch, kw)
            assert build_attn_bwd_main(s, arch=arch)


def test_xt_lds_is_legal_on_every_mfma_target_and_excludes_dma():
    for arch in CDNA:
        assert _xt_legal(arch)
    with pytest.raises(ValueError, match="dma"):
        AttnBwdSpec(head_size=64, transpose_source="xt_lds", global_path="dma")


def _ir_without_name(spec, arch):
    name = validate_attn_bwd_spec(spec, arch).kernel_name("main")
    ir = lower(build_attn_bwd_main(spec, arch=arch), arch=arch, llvm_flavor="llvm22")
    return ir.replace(name, "KNAME")


@pytest.mark.parametrize("arch", ARCHS)
def test_spelled_out_default_swizzles_are_the_default_kernel(arch):
    """``none`` on a row-major buffer, and the planner's choice on an XT image
    (``xor`` on gfx942), are the defaults: spelled out, the kernel equals the
    one without ``lds_swizzle``; the levers themselves change it. The
    ``INERT_KNOBS`` predicate of ``lds_swizzle`` says exactly that (it is what
    the sweep deduplicates on)."""
    inert = INERT_KNOBS["lds_swizzle"]

    def check(spec, ref, same: bool):
        assert (_ir_without_name(spec, arch) == ref) is same, spec.lds_swizzle
        assert inert(validate_attn_bwd_spec(spec, arch), arch) is same

    base = AttnBwdSpec(head_size=64, mask_class="band")
    ref = _ir_without_name(base, arch)
    check(dataclasses.replace(base, lds_swizzle="do=none,q=none"), ref, True)
    check(dataclasses.replace(base, lds_swizzle="qt=pad8,dot=xor"), ref, True)
    check(dataclasses.replace(base, lds_swizzle="q=xor"), ref, False)
    if _xt_legal(arch):
        xt = dataclasses.replace(base, transpose_source="xt_lds")
        try:
            validate_attn_bwd_spec(xt, arch)
        except ValueError as exc:
            # the one-wave gfx1151 d64 tile spills with the XT images: the
            # validator refuses it (LAYOUT_SPILLING_COMBOS); d128 carries it
            assert "LAYOUT_SPILLING_COMBOS" in str(exc), exc
            base = dataclasses.replace(base, head_size=128)
            ref = _ir_without_name(base, arch)
            xt = dataclasses.replace(base, transpose_source="xt_lds")
        xt_ref = _ir_without_name(xt, arch)
        assert xt_ref != ref
        plan = lds_plan(validate_attn_bwd_spec(xt, arch), arch)
        chosen = {n: plan.layout(n).swizzle for n in _IMAGES if plan.has(n)}
        if arch == "gfx942":
            assert set(chosen.values()) == {"xor"}
        spelled = ",".join(f"{n}={v}" for n, v in sorted(chosen.items()))
        check(dataclasses.replace(xt, lds_swizzle=spelled), xt_ref, True)
        other = "pad8" if chosen["qt"] != "pad8" else "xor"
        check(dataclasses.replace(xt, lds_swizzle=f"qt={other}"), xt_ref, False)


def _walk(ops):
    for op in ops:
        yield op
        for r in op.regions:
            yield from _walk(r.ops)


def _q_loop_lds_loads(kernel):
    loops = [
        op
        for op in _walk(kernel.body.ops)
        if op.name == "scf.for"
        and any(o.name == "tile.mma" for o in _walk(op.regions[0].ops))
    ]
    return [
        op
        for lp in loops
        for op in _walk(lp.regions[0].ops)
        if op.name.startswith("tile.smem_load")
    ]


def _all_lds_loads(kernel):
    return [op for op in _walk(kernel.body.ops) if op.name.startswith("tile.smem_load")]


@pytest.mark.parametrize("kt", ("reg", "lds"))
@pytest.mark.parametrize("arch", CDNA)
@pytest.mark.parametrize("d", (32, 64, 128))
def test_xt_lds_reads_k_outer_operands_as_vectors(arch, d, kt):
    """The point of the lever: with ``lds_plain`` the K-outer operands (G1 /
    G3 B in the q loop, G4 B once per CTA or every step) are single-element
    LDS reads; with ``xt_lds`` no LDS read of the kernel is a single element
    (the B reads are the vectors the simulator test checks; the writer reads
    ``vec_d`` elements)."""
    base = AttnBwdSpec(head_size=d, transpose_source="lds_plain", kt_source=kt)
    try:
        validate_attn_bwd_spec(base, arch)
    except ValueError as ex:
        pytest.skip(f"no lds_plain twin: {ex}")
    plain = build_attn_bwd_main(base, arch=arch)
    assert any(op.attrs.get("vec") == 1 for op in _q_loop_lds_loads(plain))
    xt = build_attn_bwd_main(
        dataclasses.replace(base, transpose_source="xt_lds"), arch=arch
    )
    assert _q_loop_lds_loads(xt)
    assert all(op.attrs.get("vec") > 1 for op in _all_lds_loads(xt))


def test_lds_route_reads_p_and_ds_with_the_plain_reader():
    """``xt_lds`` with ``pt_route = "lds"``: P^T / dS^T have no XT image and are
    read K-outer element by element, as under ``lds_plain``."""
    s = AttnBwdSpec(head_size=64, transpose_source="xt_lds", pt_route="lds")
    loads = _q_loop_lds_loads(build_attn_bwd_main(s, arch="gfx942"))
    assert any(op.attrs.get("vec") == 1 for op in loads)
    assert lds_plan(validate_attn_bwd_spec(s, "gfx942"), "gfx942").has("pt")
