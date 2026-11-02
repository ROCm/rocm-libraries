# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""N2 (true-mono WP17, DESIGN 12.3): the out_proj K-split into its producers' epilogues.

Accuracy-gated lever (PLAN decision 1; tolerance class: the out_proj summation order changes). Two device functions
that replace a served pair of the hybrid's stage functions and must be linked together (sidecar ``requires`` /
``pair``, ATOM ``extern_stages``):

* ``b3`` (:func:`build_b3n2_kernel`): the rocKE B3 (``stage_b3``, read-only; its ``_B3.emit`` is reused verbatim by
  an anchored source patch, :func:`_patched_emit`) whose epilogue, instead of publishing the gated output's MXFP8 pack
  (``d8`` / ``de``) to C1, multiplies the HEAD's whole 128-value pack by its out_proj slice and publishes f32 partial
  products. Each sibling task (value head ``vh``, split ``s`` of ``S``) already holds the head's 128 f32 core rows
  after the sibling exchange (it needs them for the RMS); it also loads the head's 128 z values (one more qkvz poll,
  B1's tiles), forms the gated output of all four 32-row blocks with exactly the stock epilogue's operations (so its
  own block's pack is bit for bit the one the stock B3 publishes, and every sibling computes the same pack for the
  others' blocks), and then runs one scaled MFMA (16x16x128, MXFP8 x MXFP4, the head = one k_iter of C1's chain) per
  16-column tile of its share ``160 / S`` of the 160 output tiles. The partial ``P[r][t][vh][c]`` (request row ``r``,
  tile ``t``, head ``vh``, column ``c``) goes to a private tagged-pair region (:data:`N2_OFF`) of the scratch.
* ``c1`` (:func:`build_c1n2_kernel`): C1 without its GEMV: task ``t`` polls the 48 heads' partials of its 16 columns
  and request rows, sums them in a FIXED order (``order=seq``: per thread ``HPT`` heads in head order, then the ``G``
  group sums in group order; ``order=chain``: the stock C1's own order, part ``p`` = heads ``p, p + 4, ..., p + 44``
  in sequence, then ``((p0 + p1) + p2) + p3``: bitwise to the stock C1 iff the MFMA's accumulate equals an f32 add of
  the rounded per-head dot; deterministic either way, no float atomics), then the stock HC write (``res + bf16(bf16(proj) * inj)``, ``w`` and
  ``whs`` puts) through the WP2b emitter's own code.

What changes numerically: the stock C1 accumulates the 48 head k_iters inside 4 MFMA accumulator chains (wave
``part`` holds heads ``part, part + 4, ...``) and adds the 4 chains ``((w0 + w1) + w2) + w3``. N2 rounds every head's
dot to f32 first and adds the 48 partials in another order. The quantized operands (activation pack, weights,
scales) are the same bits; the rounding point to bf16 stays in C1.

Region (plain scratch, ``numa=1`` builds only): every mailbox of a ``numa=1`` build lives in ATOM's NUMA area, and
the plain mailbox part's second half holds only the QSA layers' index scores and B3's ``xchg=numa2`` exchange
(:data:`stage_b3.XCHG_OFFSET`, 4 rows): the N2 region starts past that exchange and ends below
``layout.mailbox_bytes(MAX_ROWS)`` (checked by :func:`check_region`). A DeltaNet launch never writes the QSA scores,
and a tag is unique per launch, so a stale pair of another launch never matches.

Knobs (:class:`N2Knobs`): ``issue`` (where the B3 tasks issue their out_proj weight loads: ``epi`` = at the start
of the epilogue, before the head's core poll; ``loader`` = with the head-z poll; ``entry`` = at the stage entry),
``hpt`` (C1 heads per thread, 0 = by rows), ``w_cpol`` / ``s_cpol`` (cache bits of the weight / scale loads),
``pub`` (1: B3 also publishes the stock ``d8`` / ``de`` pack: gate builds only, the T0 golden reads it),
``mutate`` (test-only must-fail mutants: ``drop`` C1 leaves head 0 out; ``dup`` C1 adds head 0 twice; ``kit`` B3
multiplies by head ``vh ^ 1``'s weights; ``ascale`` B3 uses the next block's A exponent).
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import textwrap
from dataclasses import dataclass, fields, replace
from types import SimpleNamespace

from rocke.core.ir import F32, I32, I64, IRBuilder, VectorType
from rocke.helpers import mailbox as M
from rocke.helpers import mx_quant as mxq
from rocke.helpers.lds_arena import LdsArena, LdsEntry

from .. import stage_b3 as SB
from . import gemv as GV

__all__ = [
    "N2_OFF",
    "N2_ROW_PAIRS",
    "N2Knobs",
    "build_b3n2_kernel",
    "build_c1n2_kernel",
    "check_region",
    "knobs_from_spec",
    "knobs_spec",
    "pair_id",
    "write_n2_functions",
]

HEADS = SB.HV  # 48 value heads = 48 k_iters of out_proj (128 K each)
TILES = SB.H // 16  # 160 output tiles of 16 columns
KI = 2 * ((SB.VALUE + 255) // 256)  # 48 k_iters (C1's chain)
CH = KI // 4  # scale dwords per tile (4 k_iters' scale bytes each)
N2_ROW_PAIRS = TILES * HEADS * 16  # 122880 pairs per request row
_XCHG_END = SB.XCHG_OFFSET + 4 * SB.XCHG_ROW_PAIRS * M.PAIR_BYTES
N2_OFF = -(-_XCHG_END // 16384) * 16384  # 5505024: first 16 KiB boundary past B3's numa2 exchange
MUTANTS = ("drop", "dup", "kit", "ascale")
ISSUES = ("epi", "loader", "entry")
ORDERS = ("seq", "chain")


def check_region(mailbox_bytes_4: int, scratch_bytes: int, rows: int = 4) -> None:
    """The region fits in the plain mailbox part (``layout.mailbox_bytes(MAX_ROWS)``) past B3's exchange."""
    end = N2_OFF + rows * N2_ROW_PAIRS * M.PAIR_BYTES
    if N2_OFF < _XCHG_END or end > mailbox_bytes_4 or mailbox_bytes_4 // 2 > N2_OFF:
        raise ValueError(f"N2 region [{N2_OFF}, {end}) outside the plain second half [{mailbox_bytes_4 // 2}, "
                         f"{mailbox_bytes_4}) past the exchange end {_XCHG_END}")  # fmt: skip
    if scratch_bytes and end > scratch_bytes:
        raise ValueError("N2 region past the scratch")


@dataclass(frozen=True)
class N2Knobs:
    issue: str = "epi"
    hpt: int = 0
    w_cpol: int = 0
    s_cpol: int = 0
    pub: int = 0
    order: str = "seq"
    mutate: str = ""

    def __post_init__(self) -> None:
        if self.order not in ORDERS:
            raise ValueError(f"order must be one of {ORDERS}")
        if self.order == "chain" and self.hpt:
            raise ValueError("order=chain fixes the grouping (4 parts of 12 heads)")
        if self.issue not in ISSUES:
            raise ValueError(f"issue must be one of {ISSUES}")
        if self.hpt and (HEADS % self.hpt or self.hpt < 1):
            raise ValueError("hpt must divide 48")
        if self.w_cpol not in GV.CPOLS or self.s_cpol not in GV.CPOLS:
            raise ValueError(f"w_cpol / s_cpol must be one of {GV.CPOLS}")
        if self.pub not in (0, 1):
            raise ValueError("pub: 0 | 1")
        if self.mutate and self.mutate not in MUTANTS:
            raise ValueError(f"mutate must be one of {MUTANTS}")


def knobs_spec(k: N2Knobs) -> str:
    d = N2Knobs()
    return ",".join(f"{f.name}={getattr(k, f.name)}" for f in fields(k) if getattr(k, f.name) != getattr(d, f.name))


def knobs_from_spec(spec: str) -> N2Knobs:
    kn = {}
    for kv in filter(None, (x.strip() for x in (spec or "").split(","))):
        key, _, v = kv.partition("=")
        cur = getattr(N2Knobs(), key)
        kn[key] = int(v) if isinstance(cur, int) else v
    return N2Knobs(**kn)


def pair_id(n2: N2Knobs) -> str:
    """Pair id shared by a generation's B3 and C1 (the region layout and the mutant, which both sides must agree on
    to be a meaningful pair)."""
    s = f"n2v1:{N2_OFF}:{N2_ROW_PAIRS}:{n2.mutate}"
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def plan(rows: int, hpt: int = 0, order: str = "seq") -> tuple[int, int]:
    """C1's ``(groups G, heads per thread HPT)``: ``rows * 16 * G <= 512`` threads. ``order=chain``: the stock C1's
    grouping (group ``p`` = heads ``p, p + 4, ...``, the k_iters wave ``p`` chains), G = 4."""
    if order == "chain":
        return 4, HEADS // 4
    if hpt:
        g = HEADS // hpt
        if rows * 16 * g > 512:
            raise ValueError(f"hpt {hpt} needs {rows * 16 * g} threads")
        return g, hpt
    g = {1: 24, 2: 16, 3: 8, 4: 8}[rows]
    return g, HEADS // g


# ============================================================================ B3 with the N2 epilogue


def _patched_emit():
    """``stage_b3._B3.emit`` with three hooks, each guarded by ``self._n2_on`` (off: the same Python path, so the
    stock build's IR): the head-z loader before the qkvz barrier, the entry weight issue, and the epilogue."""
    src = textwrap.dedent(inspect.getsource(SB._B3.emit))

    def sub(text, old, new):
        if text.count(old) != 1:
            raise RuntimeError(f"stage_b3 anchor not unique / missing: {old[:60]!r}")
        return text.replace(old, new)

    a1 = "        b.smem_store_vN(io.lds.smem, [b.select(is_qk, w_qk, w_vz)], vec, 4)\n    self.barrier()\n"
    src = sub(src, a1, a1.replace("    self.barrier()\n",
                                  "    if self._n2_on:\n        self._n2_loader(locals())\n    self.barrier()\n"))
    a2 = '    mb_d8, mb_de = self.mb("d8"), self.mb("de")\n'
    src = sub(src, a2, a2 + "    if self._n2_on and self.n2.issue == 'entry':\n        self._n2_issue(locals())\n")
    head = "    # Epilogue: wave r"
    tail = '    if k.stores == "end":'
    i, j = src.index(head), src.index(tail)
    if src.count(head) != 1 or src.count(tail) != 1:
        raise RuntimeError("stage_b3 epilogue anchors")
    seg = src[i:j]
    src = src[:i] + "    if self._n2_on:\n        self._n2_epilogue(locals())\n    else:\n" + textwrap.indent(seg, "    ") + src[j:]
    ns = dict(vars(SB))
    exec(compile(src, f"<n2 patched {SB.__file__}>", "exec"), ns)  # noqa: S102
    return ns["emit"]


class _B3N2(SB._B3):
    emit = _patched_emit()

    def __init__(self, io, shape, knobs, n2: N2Knobs | None):
        super().__init__(io, shape, knobs)
        self.n2 = n2
        self._n2_on = n2 is not None
        self._n2_w = None
        self.TPT = TILES // self.S  # tiles per task
        self.TPW = self.TPT // SB.WAVES  # tiles per wave

    # -- weights of the task's tiles (wave w: tiles s * TPT + w * TPW + i), head vh's k_iter
    def _n2_issue(self, L):
        if self._n2_w is not None:
            return
        b, c, io, n2 = self.b, self.c, self.io, self.n2
        vh, s, wave, lane = L["vh"], L["s"], self.io.wave, self.io.lane
        kit = b.xor(vh, c(1)) if n2.mutate == "kit" else vh
        rs_w = self.rsrc(io.addr("w_out"), SB.H * SB.VALUE // 2)
        rs_s = self.rsrc(io.addr("s_out"), SB.H * SB.VALUE // 32)
        t0 = b.add(b.mul(s, c(self.TPT)), b.mul(wave, c(self.TPW)))
        ws, ss = [], []
        for i in range(self.TPW):
            t = b.add(t0, c(i))
            el = b.add(b.mul(b.add(b.mul(t, c(KI)), kit), c(64)), lane)
            ws.append(b.buffer_load_cp(rs_w, b.mul(el, c(16)), c(0), dwords=4, aux=n2.w_cpol))
            sl = b.add(b.mul(b.add(b.mul(t, c(CH)), b.lshr(kit, c(2))), c(64)), lane)
            ss.append(b.buffer_load_cp(rs_s, b.mul(sl, c(4)), c(0), dwords=1, aux=n2.s_cpol))
        self._n2_w = (t0, kit, ws, ss)

    # -- the head's 128 z values (split > 1: the other siblings' rows are B1 tiles this task did not poll)
    def _n2_loader(self, L):
        if self.n2.issue == "loader":
            self._n2_issue(L)
        if self.S == 1:
            return
        b, c = self.b, self.c
        rows, tid, vh = self.rows, self.io.tid, L["vh"]
        lo = L["NL"] * rows
        with b.scf_if(b.land(b.cmp_ge(tid, c(lo)), b.cmp_lt(tid, c(lo + 32 * rows)))):
            q = b.sub(tid, c(lo))
            i = q if rows == 1 else b.land(q, c(31))
            ri = 0 if rows == 1 else b.lshr(q, c(5))
            src = b.add(c(SB.QKV // 2), b.add(b.mul(vh, c(SB.DV // 2)), b.mul(i, c(2))))
            v = self.poll([(L["mb_qkvz"], self.poll_ix("qkvz", self.ix(src, ri)), 2)])[0]
            z0, z1 = M.bf2_f32(b, v[0])
            z2, z3 = M.bf2_f32(b, v[1])
            self.stv("n2_z", b.mul(i, c(4)), b.vec_pack([z0, z1, z2, z3], F32), ri)

    def _n2_epilogue(self, L):
        b, c, io, k, n2 = self.b, self.c, self.io, self.k, self.n2
        rows, S, RT, NB = self.rows, self.S, self.RT, self.NB
        lane, wave, vh, s, rw = io.lane, io.wave, L["vh"], L["s"], L["rw"]
        if n2.issue == "epi":
            self._n2_issue(L)
        g16 = b.land(lane, c(15))
        hb = b.lshr(g16, c(2))  # head block of this lane (0..3)
        m4 = b.land(g16, c(3))
        hr0 = b.add(b.mul(hb, c(32)), b.mul(m4, c(8)))  # its first head row
        r_nw = L["r_nw"]
        epi = b.cmp_eq(wave, c(0)) if rows == 1 else b.cmp_lt(wave, c(rows))
        with b.scf_if(epi):
            if S > 1:
                got = self.poll(
                    [(L["mb_core_poll"], L["xpoll"](b.add(b.mul(vh, c(SB.DV)), b.mul(lane, c(2))), rw), 2)],
                    L["xk_poll"],
                )[0]
                two = b.mul(lane, c(2))
                self.stf("b3_cores", two, b.bitcast(got[0], F32), rw)
                self.stf("b3_cores", b.add(two, c(1)), b.bitcast(got[1], F32), rw)
                cname = "b3_cores"
            else:
                cname = "b3_core"
            tot = b.fadd(self.sumsq64(cname, 0, rw, lane), self.sumsq64(cname, 64, rw, lane))
            inv_rms = b.rsqrt(b.fadd(b.fmul(tot, self.f(1.0 / SB.DV)), self.f(SB.EPS)))
            cv = self.ldf4(cname, hr0, rw) + self.ldf4(cname, b.add(hr0, c(4)), rw)
            nw = b.buffer_load_vN(r_nw, b.mul(hr0, c(2)), c(0), I32, 4)
            zname, zoff = ("n2_z", 0) if S > 1 else ("b3_vz", RT)
            zv = self.ldf4(zname, b.add(hr0, c(zoff)), rw) + self.ldf4(zname, b.add(hr0, c(zoff + 4)), rw)
            zsig = [self.sigmoid(z) for z in zv]
            wv = self.bf16x8([b.vec_extract(nw, e) for e in range(4)])
            out = [self.bf16r(b.fmul(b.fmul(b.fmul(cv[e], inv_rms), wv[e]), zsig[e])) for e in range(8)]
            amax = b.fabs(out[0])
            for e in range(1, 8):
                amax = b.fmax(amax, b.fabs(out[e]))
            amax = self.xmax(self.xmax(amax, 1), 2)
            ex, scale = mxq.e8m0_fp8(b, amax)
            cvt = k.cvt
            if cvt == "intrinsic":
                from ...mx_quant import intrinsic_cvt

                cvt = intrinsic_cvt
            lo = mxq.pack_fp8x4(b, out[0], out[1], out[2], out[3], scale, cvt=cvt)
            hi = mxq.pack_fp8x4(b, out[4], out[5], out[6], out[7], scale, cvt=cvt)
            with b.scf_if(b.cmp_lt(lane, c(16))):
                wx = b.add(b.mul(hb, c(8)), b.mul(m4, c(2)))
                self.b.smem_store_vN(io.lds.smem, [self.lw("n2_x8", wx, rw)], lo, 1)
                self.b.smem_store_vN(io.lds.smem, [self.lw("n2_x8", b.add(wx, c(1)), rw)], hi, 1)
                with b.scf_if(b.cmp_eq(m4, c(0))):
                    self.b.smem_store_vN(io.lds.smem, [self.lw("n2_xe", hb, rw)], ex, 1)
                if n2.pub:
                    # the stock publish of this task's own blocks (gate builds: T0 reads the pack)
                    mine = b.land(b.cmp_ge(hb, b.mul(s, c(NB))), b.cmp_lt(hb, b.mul(b.add(s, c(1)), c(NB))))
                    with b.scf_if(mine):
                        M.put2(b, L["mb_d8"],
                               self.put_ix("d8", self.ix(b.add(b.add(b.mul(vh, c(32)), b.mul(hb, c(8))), b.mul(m4, c(2))), rw)),
                               lo, hi, io.tag, self.put_knobs)  # fmt: skip
                        with b.scf_if(b.cmp_eq(m4, c(0))):
                            M.put(b, L["mb_de"], self.put_ix("de", self.ix(b.add(b.mul(vh, c(4)), hb), rw)), ex, io.tag,
                                  self.put_knobs)  # fmt: skip
        self.barrier()
        # every wave: TPW tiles, one scaled MFMA each; A row m = request row m // 4 (lane group g of the result holds
        # request row g in element 0), A K block (lane >> 4) = head block (lane >> 4)
        t0, kit, ws, ss = self._n2_w
        tok = b.smin(b.lshr(g16, c(2)), c(rows - 1)) if rows > 1 else c(0)
        xr = self.lw("n2_x8", 0, tok)
        er = self.lw("n2_xe", 0, tok)
        blk = b.lshr(lane, c(4))
        off = b.add(xr, b.mul(blk, c(4)))
        a0 = b.smem_load_vN(io.lds.smem, off, dtype=I32, n=4)
        a1 = b.smem_load_vN(io.lds.smem, b.add(off, c(16)), dtype=I32, n=4)
        aop = b.vec_concat(a0, a1)
        eb = b.land(b.add(blk, c(1)), c(3)) if n2.mutate == "ascale" else blk
        sa = b.vec_extract(b.smem_load_vN(io.lds.smem, b.add(er, eb), dtype=I32, n=1), 0)
        sh = b.mul(b.land(kit, c(3)), c(8))
        mb = M.mailbox_from_i64(b, b.add(io.scratch, b.const_i64(N2_OFF)), c(0x7FFFFFF0), self.mbk)
        g = b.lshr(lane, c(4))
        col = g16
        zero = b.zero_vec_f32(4)
        vals = []
        for i in range(self.TPW):
            sb = b.lshr(ss[i], sh)
            acc = b.mfma_scale_mx(aop, ws[i], zero, sa, sb, shape="16x16x128", fmt_a="fp8", fmt_b="fp4",
                                  opsel_a=0, opsel_b=0)  # fmt: skip
            vals.append(b.vec_extract(acc, 0))
        with b.scf_if(b.cmp_lt(g, c(rows))):
            rbase = b.mul(g, c(N2_ROW_PAIRS)) if rows > 1 else c(0)
            for i in range(self.TPW):
                t = b.add(t0, c(i))
                idx = b.add(rbase, b.add(b.mul(b.add(b.mul(t, c(HEADS)), vh), c(16)), col))
                M.put(b, mb, idx, b.bitcast(vals[i], I32), io.tag, self.mbk)


def _b3n2_arena(shape, knobs, rows_pad):
    ents = list(SB.b3_lds_entries(shape, knobs))
    ents += [LdsEntry("n2_x8", 32, True), LdsEntry("n2_xe", 4, True)]
    if knobs.split > 1:
        ents.append(LdsEntry("n2_z", SB.DV, True))
    ents = [LdsEntry("b3_pad", rows_pad)] + ents
    return LdsArena({"b3": ents}, rows=shape.rows)


def _b3n2_body(b, a, layout, shape, knobs, n2, part="body"):
    pad = SB.MONO_B3_LDS_START[shape.rows]
    arena = _b3n2_arena(shape, knobs, pad) if n2 is not None else SB._arena(shape, knobs, pad=pad)
    if arena.words > SB.MONO_LDS_WORDS[shape.rows]:
        raise ValueError(f"B3 N2 arena {arena.words} words exceeds the mono's {SB.MONO_LDS_WORDS[shape.rows]}")
    lds = arena.emit(b)
    io = SB.B3Io(
        b=b, tid=a.tid, lane=a.lane, wave=a.wave, task=a.task, tag=a.tag,
        scratch=b.readfirstlane(a.scratch), lds=lds, addr=lambda n: getattr(a, n),
        sc={n: layout.sc[n] for n in SB.B3_REGIONS}, row_pairs=layout.row_pairs,
    )  # fmt: skip
    with b.scf_if(b.cmp_lt(a.task, b.const_i32(SB.b3_tasks(knobs)))):
        e = _B3N2(io, shape, knobs, n2)
        if part == "pre":
            e.emit_pre()
        else:
            e.emit()


def build_b3n2_kernel(shape, knobs, n2: N2Knobs | None, layout, part="body"):
    """``rocke_q38_b3_<variant>`` (or its ``b3pre`` prologue). ``n2=None``: the stock B3 through the patched emit
    (the CPU test checks it lowers to the same IR as ``stage_b3``)."""
    from ..stage_fn import ABI_PARAMS

    if n2 is not None and not knobs.numa:
        raise ValueError("N2 addresses the plain scratch's second half: only under ATOM's numa=1 lever")
    if n2 is not None and shape.group:
        raise ValueError("N2: narrow builds only")
    stem = "b3" if part == "body" else "b3pre"
    b = IRBuilder(f"rocke_q38_{stem}_{shape.variant}")
    b.kernel.attrs["max_workgroup_size"] = layout.threads
    vals = {}
    for name, t in ABI_PARAMS:
        vals[name] = b.param(name, I32 if t == "i32" else I64)
    tid = b.thread_id_x()
    for name in ("tag", "task", "layer"):
        vals[name] = b.readfirstlane(vals[name])
    a = SimpleNamespace(**vals, tid=tid, lane=b.land(tid, b.const_i32(63)),
                        wave=b.readfirstlane(b.lshr(tid, b.const_i32(6))))  # fmt: skip
    _b3n2_body(b, a, layout, shape, knobs, n2, part)
    b.ret()
    return b.kernel


def lower_b3n2_function(shape, knobs, n2, layout, *, arch="gfx950", flavor=None, part="body"):
    from ..device_function import DeviceFunctionKnobs, lower_device_function

    stem = "b3" if part == "body" else "b3pre"
    return lower_device_function(build_b3n2_kernel(shape, knobs, n2, layout, part), arch=arch,
                                 symbol=f"rocke_q38_{stem}_{shape.variant}", llvm_flavor=flavor,
                                 knobs=DeviceFunctionKnobs(lds="arg"))  # fmt: skip


# ============================================================================ C1 = fixed-order reduce + HC write


def _c1n2(e, n2: N2Knobs) -> None:
    b, a, g = e.b, e.a, e.g
    with b.scf_if(b.cmp_lt(a.task, e.c(TILES))):
        _c1n2_body(e, n2)


def _c1n2_body(e, n2: N2Knobs) -> None:
    b, a, g, k = e.b, e.a, e.g, e.k
    tid, rows, t = a.tid, e.rows, a.task
    if rows == 1:
        bs, rt = b.lshr(b.smin(tid, e.c(31)), e.c(3)), e.c(0)
    else:
        tt = b.smin(tid, e.c(32 * rows - 1))
        bs, rt = b.lshr(b.land(tt, e.c(31)), e.c(3)), b.lshr(tt, e.c(5))
    word = b.add(b.add(b.mul(bs, e.c(g.row_words)), b.mul(t, e.c(8))), b.land(tid, e.c(7)))
    if e.s.kind == "delta_ple":
        res = e.poll([("ph", e.mbix(word, rt), 1)])[0][0]
    else:
        rs_h = e.rsrc(a.h_in, g.hc_width * 2 * rows)
        hw = b.add(word, b.mul(rt, e.c(g.hc_words))) if rows > 1 else word
        res = e.load(rs_h, b.mul(hw, e.c(4)), 1)
    inj = b.bitcast(e.poll([("inj", e.mbix(bs, rt), 1)])[0][0], F32)

    G, HPT = plan(rows, n2.hpt, n2.order)
    mb = M.mailbox_from_i64(b, b.add(e.scratch, b.const_i64(N2_OFF)), e.c(0x7FFFFFF0), e.mbk)
    with b.scf_if(b.cmp_lt(tid, e.c(rows * 16 * G))):
        span = 16 * G
        r = e.c(0) if rows == 1 else b.div(tid, e.c(span))
        rem = tid if rows == 1 else b.mod(tid, e.c(span))
        gg = b.lshr(rem, e.c(4))
        cc = b.land(rem, e.c(15))
        base = b.add(b.mul(b.mul(t, e.c(HEADS)), e.c(16)), cc)
        if rows > 1:
            base = b.add(base, b.mul(r, e.c(N2_ROW_PAIRS)))
        if n2.order == "chain":
            heads = [b.add(gg, e.c(4 * j)) for j in range(HPT)]  # the stock chain of wave part gg: k_iters gg + 4 j
        else:
            h0 = b.mul(gg, e.c(HPT))
            heads = [b.add(h0, e.c(j)) for j in range(HPT)]
        specs = [(mb, b.add(base, b.mul(h, e.c(16))), 1) for h in heads]
        vals = []
        for first in range(0, len(specs), GV.POLL_MAX):
            batch = specs[first : first + GV.POLL_MAX]
            got = e._poll_asm(batch) if k.mb_impl == "asm" else e._poll_ir(batch)
            vals += [b.bitcast(x[0], F32) for x in got]
        acc = vals[0]
        if n2.mutate == "drop":
            acc = b.select(b.cmp_eq(gg, e.c(0)), e.f(0.0), acc)
        if n2.mutate == "dup":
            acc = b.select(b.cmp_eq(gg, e.c(0)), b.fadd(acc, vals[0]), acc)
        for v in vals[1:]:
            acc = b.fadd(acc, v)
        e.stw(e.ent("x8", b.add(b.mul(gg, e.c(16)), cc), r), b.bitcast(acc, I32))
    e.barrier()
    with b.scf_if(b.cmp_lt(tid, e.c(32 * rows))):
        j = b.mul(b.land(tid, e.c(7)), e.c(2))

        def part(gi, col):
            return e.ldw(e.ent("x8", b.add(e.c(16 * gi), col), rt), F32)

        p0, p1 = part(0, j), part(0, b.add(j, e.c(1)))
        for gi in range(1, G):
            p0 = b.fadd(p0, part(gi, j))
            p1 = b.fadd(p1, part(gi, b.add(j, e.c(1))))
        r0, r1 = M.bf2_f32(b, res)
        o0 = b.fadd(r0, e.bf16r(b.fmul(e.bf16r(p0), inj)))
        o1 = b.fadd(r1, e.bf16r(b.fmul(e.bf16r(p1), inj)))
        out = M.bf16_pair(b, o0, o1)
        e.put("w", e.mbix(word, rt), out)
        v0, v1 = M.bf2_f32(b, out)
        sq = b.fadd(b.fmul(v0, v0), b.fmul(v1, v1))
        for o in (1, 2, 4):
            sq = b.fadd(sq, e.xshfl(sq, o))
        with b.scf_if(b.cmp_eq(b.land(tid, e.c(7)), e.c(0))):
            e.put("whs", e.mbix(b.add(b.mul(bs, e.c(g.tasks["c1"])), t), rt), sq)


def build_c1n2_kernel(shape: GV.GemvShape, knobs: GV.GemvKnobs, n2: N2Knobs, g=None):
    g = g or GV.geom_from_atom()
    if shape.stage != "c1" or shape.wide or shape.group:
        raise ValueError("C1 N2: narrow c1 builds only")
    if not knobs.numa:
        raise ValueError("C1 N2: numa=1 builds only (the region is the plain scratch's second half)")
    if knobs.store_out or knobs.mutate:
        raise ValueError("C1 N2: store_out / GEMV mutants do not apply")
    plan(shape.rows, n2.hpt)
    b = IRBuilder(shape.symbol)
    b.kernel.attrs["max_workgroup_size"] = g.threads
    b.fence_barriers = "compiler"
    vals = {}
    for name, t in GV._abi_params():
        vals[name] = b.param(name, I32 if t == "i32" else I64)
    tid = b.thread_id_x()
    for name in ("tag", "task", "layer", "l0", "l1", "l2", "l3"):
        vals[name] = b.readfirstlane(vals[name])
    a = SimpleNamespace(**vals, tid=tid, lane=b.land(tid, b.const_i32(63)),
                        wave=b.readfirstlane(b.lshr(tid, b.const_i32(6))))  # fmt: skip
    pw = GV._pool_words(g, shape, knobs)
    e = GV._E(b, a, g, shape, knobs, pw)
    _c1n2(e, n2)
    b.ret()
    return b.kernel


def lower_c1n2_function(shape, knobs, n2, g=None, *, arch="gfx950", flavor=None):
    from ..device_function import DeviceFunctionKnobs, lower_device_function

    return lower_device_function(build_c1n2_kernel(shape, knobs, n2, g), arch=arch, symbol=shape.symbol,
                                 llvm_flavor=flavor, knobs=DeviceFunctionKnobs(lds="arg", inline="always"))  # fmt: skip


# ============================================================================ C1 streaming (the bitwise twin)
#
# ``build_c1s_kernel``: the stock WP2b C1 (same weights issue, same MFMA chain: wave ``part`` = k_iters ``part + 4 j``
# in order, same accumulator, same scales, same reduce and HC write) whose A operands are POLLED per step straight from
# B3's / Q7's ``d8`` / ``de`` mailboxes into registers, instead of staging the whole pack in LDS behind two CTA
# barriers before the first MFMA. The chain starts on the first head that lands and its post-poll tail is the last
# head's step, not the whole pack. Same bits as the stock C1 by construction (same MFMA sequence and operands; the
# in-situ gate is bitwise). Knob ``sgrp`` (:class:`C1sKnobs`): steps polled per batch (1, 2, 3, 4, 6, 12).


@dataclass(frozen=True)
class C1sKnobs:
    sgrp: int = 1
    mutate: str = ""  # test only: "kit" (step 1 polls k_iter + 4), "exp" (step 1 takes the next block's exponent)

    def __post_init__(self) -> None:
        if self.sgrp not in (1, 2, 3, 4, 6, 12):
            raise ValueError("sgrp: 1, 2, 3, 4, 6 or 12 (12: every step in ONE tagged poll, mb_impl ir / irf)")
        if self.mutate not in ("", "kit", "exp"):
            raise ValueError("mutate: kit | exp")


def c1s_spec(k: C1sKnobs) -> str:
    d = C1sKnobs()
    return ",".join(f"{f.name}={getattr(k, f.name)}" for f in fields(k) if getattr(k, f.name) != getattr(d, f.name))


def c1s_from_spec(spec: str) -> C1sKnobs:
    kn = {}
    for kv in filter(None, (x.strip() for x in (spec or "").split(","))):
        key, _, v = kv.partition("=")
        kn[key] = int(v) if key == "sgrp" else v
    return C1sKnobs(**kn)


def _c1s(e, sk: C1sKnobs) -> None:
    b = e.b
    with b.scf_if(b.cmp_lt(e.a.task, e.c(e.g.tasks["c1"]))):
        GV._split(e, 4, lambda role: _c1s_body(e, role, sk))


def _c1s_body(e, role, sk: C1sKnobs) -> None:
    b, a, g, k = e.b, e.a, e.g, e.k
    tid, lane, wave = a.tid, a.lane, a.wave
    rows = e.rows
    t = a.task
    part = b.land(wave, e.c(3))
    off = e.oob_off() if role is None else e.c(0)
    rs_w = e.rsrc(a.w_out, g.hidden * g.delta_value // 2)
    rs_s = e.rsrc(a.s_out, g.hidden * g.delta_value // 32)

    def weights(off):
        ws = []
        for j in range(CH):
            el = b.add(b.add(b.mul(b.add(b.add(b.mul(t, e.c(KI)), part), e.c(4 * j)), e.c(64)), lane), e.c(0))
            el = b.add(b.mul(el, e.c(4)), off)
            ws.append(e.load(rs_w, b.mul(el, e.c(4)), 4, k.w_cpol))
        ss = []
        for j in range(CH):
            el = b.add(b.add(b.mul(b.add(b.mul(t, e.c(CH)), e.c(j)), e.c(64)), lane), off)
            ss.append(e.load(rs_s, b.mul(el, e.c(4)), 1, k.s_cpol))
        return ws, ss

    wss = weights(off) if role != "idle" else None
    if rows == 1:
        bs, rt = b.lshr(b.smin(tid, e.c(31)), e.c(3)), e.c(0)
    else:
        tt = b.smin(tid, e.c(32 * rows - 1))
        bs, rt = b.lshr(b.land(tt, e.c(31)), e.c(3)), b.lshr(tt, e.c(5))
    word = b.add(b.add(b.mul(bs, e.c(g.row_words)), b.mul(t, e.c(8))), b.land(tid, e.c(7)))
    if e.s.kind == "delta_ple":
        res = e.poll([("ph", e.mbix(word, rt), 1)])[0][0]
    else:
        rs_h = e.rsrc(a.h_in, g.hc_width * 2 * rows)
        hw = b.add(word, b.mul(rt, e.c(g.hc_words))) if rows > 1 else word
        res = e.load(rs_h, b.mul(hw, e.c(4)), 1)
    inj = b.bitcast(e.poll([("inj", e.mbix(bs, rt), 1)])[0][0], F32)

    def mm():
        ws, ss = wss
        sh = b.mul(part, e.c(8))
        if k.bsop:
            sbs = []
            for j0 in range(0, CH, 4):
                packed = e.pack4([b.lshr(ss[j], sh) for j in range(j0, j0 + 4)])
                sbs += [(packed, q) for q in range(4)]
        else:
            sbs = [(b.lshr(ss[j], sh), 0) for j in range(CH)]
        tok = b.smin(b.land(lane, e.c(15)), e.c(rows - 1)) if rows > 1 else e.c(0)
        gq = b.lshr(lane, e.c(4))
        acc = b.zero_vec_f32(4)
        for j0 in range(0, CH, sk.sgrp):
            js = list(range(j0, min(CH, j0 + sk.sgrp)))
            specs = []
            for j in js:
                kit = b.add(part, e.c(4 * j))
                if sk.mutate == "kit" and j == 1:
                    kit = b.add(kit, e.c(4))
                w0 = b.add(b.mul(kit, e.c(32)), b.mul(gq, e.c(4)))
                specs += [("d8", e.mbix(w0, tok), 2), ("d8", e.mbix(b.add(w0, e.c(2)), tok), 2),
                          ("d8", e.mbix(b.add(w0, e.c(16)), tok), 2), ("d8", e.mbix(b.add(w0, e.c(18)), tok), 2)]  # fmt: skip
                eg = b.land(b.add(gq, e.c(1)), e.c(3)) if (sk.mutate == "exp" and j == 1) else gq
                specs.append(("de", e.mbix(b.add(b.mul(kit, e.c(4)), eg), tok), 1))
            if sk.sgrp == 12 and k.mb_impl != "asm":
                # one compiler-visible poll loop over all 60 pairs-of-words (no POLL_MAX batching)
                got = e._poll_ir([(e.mb(r), e.poll_ix(r, i), nw) for r, i, nw in specs])
            else:
                got = e.poll(specs)
            for q, j in enumerate(js):
                v = got[5 * q : 5 * q + 5]
                aop = b.vec_pack([v[0][0], v[0][1], v[1][0], v[1][1], v[2][0], v[2][1], v[3][0], v[3][1]], I32)
                sb, selb = sbs[j]
                acc = e.mma(acc, aop, ws[j], v[4][0], sb, 0, selb, last=(j == CH - 1), valu_src=True)
        GV._store_red(e, acc)

    GV._role_if(e, role, 4, mm)
    e.barrier()

    with b.scf_if(b.cmp_lt(tid, e.c(32 * rows))):
        j = b.mul(b.land(tid, e.c(7)), e.c(2))

        def red(i):
            return e.ldw(e.ent("red", i, rt), F32)

        p0, p1 = red(j), red(b.add(j, e.c(1)))
        for w in range(1, 4):
            p0 = b.fadd(p0, red(b.add(j, e.c(w * 16))))
            p1 = b.fadd(p1, red(b.add(j, e.c(w * 16 + 1))))
        r0, r1 = M.bf2_f32(b, res)
        o0 = b.fadd(r0, e.bf16r(b.fmul(e.bf16r(p0), inj)))
        o1 = b.fadd(r1, e.bf16r(b.fmul(e.bf16r(p1), inj)))
        out = M.bf16_pair(b, o0, o1)
        e.put("w", e.mbix(word, rt), out)
        v0, v1 = M.bf2_f32(b, out)
        sq = b.fadd(b.fmul(v0, v0), b.fmul(v1, v1))
        for o in (1, 2, 4):
            sq = b.fadd(sq, e.xshfl(sq, o))
        with b.scf_if(b.cmp_eq(b.land(tid, e.c(7)), e.c(0))):
            e.put("whs", e.mbix(b.add(b.mul(bs, e.c(g.tasks["c1"])), t), rt), sq)


def build_c1s_kernel(shape: GV.GemvShape, knobs: GV.GemvKnobs, sk: C1sKnobs, g=None):
    g = g or GV.geom_from_atom()
    if shape.stage != "c1" or shape.wide or shape.group:
        raise ValueError("C1 streaming: narrow c1 builds only")
    if knobs.store_out or knobs.mutate:
        raise ValueError("C1 streaming: store_out / GEMV mutants do not apply")
    b = IRBuilder(shape.symbol)
    b.kernel.attrs["max_workgroup_size"] = g.threads
    b.fence_barriers = "compiler"
    vals = {}
    for name, t in GV._abi_params():
        vals[name] = b.param(name, I32 if t == "i32" else I64)
    tid = b.thread_id_x()
    for name in ("tag", "task", "layer", "l0", "l1", "l2", "l3"):
        vals[name] = b.readfirstlane(vals[name])
    a = SimpleNamespace(**vals, tid=tid, lane=b.land(tid, b.const_i32(63)),
                        wave=b.readfirstlane(b.lshr(tid, b.const_i32(6))))  # fmt: skip
    pw = GV._pool_words(g, shape, knobs)
    e = GV._E(b, a, g, shape, knobs, pw)
    _c1s(e, sk)
    b.ret()
    return b.kernel


def write_c1s_functions(out_dir, cells, c1_knobs_for, sk: C1sKnobs, g=None, flavor=None) -> list:
    """``<out_dir>/c1_<kind>_<variant>.ll`` per (kind, variant) of ``cells``; sidecar as WP2b's plus ``c1s``."""
    from ..device_function import DeviceFunctionKnobs, lower_device_function

    g = g or GV.geom_from_atom()
    out = []
    for kd, v in cells:
        rows, seq, group = GV.parse_variant(v)
        sh = GV.GemvShape("c1", kd, rows, seq, group)
        ck = c1_knobs_for(kd, v)
        fn = lower_device_function(build_c1s_kernel(sh, ck, sk, g), arch="gfx950", symbol=sh.symbol,
                                   llvm_flavor=flavor, knobs=DeviceFunctionKnobs(lds="arg", inline="always"))  # fmt: skip
        _write(fn, out_dir, sh.stem, {**GV.sidecar_extra(sh, ck), "c1s": c1s_spec(sk)})
        out.append(sh.stem)
    return out


# ============================================================================ writers


def _write(fn, out_dir, stem, extra):
    from ..device_function import write_device_function

    paths = write_device_function(fn, out_dir, stem=stem, bitcode=False)
    with open(paths["json"]) as f:
        meta = json.load(f)
    meta.update(extra)
    with open(paths["json"], "w", newline="\n") as f:
        json.dump(meta, f, indent=1)
        f.write("\n")
    return paths


def write_n2_functions(out_dir, variants, b3_knobs_for, c1_knobs_for, n2: N2Knobs, layout=None, g=None,
                       flavor=None, kinds=("delta", "delta_ple")) -> list:  # fmt: skip
    """``<out_dir>/b3/b3_<variant>.ll`` (+ ``b3pre``) and ``<out_dir>/c1/c1_<kind>_<variant>.ll`` for every variant.
    ATOM: ``b3=<out>/b3,b3pre=<out>/b3,c1=<out>/c1/c1_{kind}_{variant}.ll``. The sidecars pair them (``requires``)."""
    from ..mono_layout import from_atom

    layout = layout or from_atom(None)
    g = g or GV.geom_from_atom()
    pid = pair_id(n2)
    spec = knobs_spec(n2)
    written = []
    for v in variants:
        rows, seq, group = GV.parse_variant(v)
        shape = SB.B3Shape(rows, seq, group)
        bk = b3_knobs_for(v)
        needs = ["b3pre"] if bk.pre == "dma" else []
        for part in ("body", "pre") if needs else ("body",):
            stem = ("b3_" if part == "body" else "b3pre_") + v
            fn = lower_b3n2_function(shape, bk, n2, layout, flavor=flavor, part=part)
            extra = {"stage_knobs": SB.knobs_spec(bk), **SB.sidecar_numa(bk), "n2": spec, "n2_off": N2_OFF}
            if part == "body":
                extra.update(needs=needs, requires={"c1": pid}, pair=pid)
            else:
                extra["prologue_of"] = "b3"
            _write(fn, os.path.join(out_dir, "b3"), stem, extra)
            written.append(f"b3/{stem}")
        for kd in kinds:
            sh = GV.GemvShape("c1", kd, rows, seq, group)
            ck = c1_knobs_for(kd, v)
            fn = lower_c1n2_function(sh, ck, n2, g, flavor=flavor)
            extra = {**GV.sidecar_extra(sh, ck), "n2": spec, "n2_off": N2_OFF, "pair": pid,
                     "requires": {"b3": pid}}  # fmt: skip
            _write(fn, os.path.join(out_dir, "c1"), sh.stem, extra)
            written.append(f"c1/{sh.stem}")
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="emit the N2 (out_proj K-split) B3 + C1 stage functions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="r1,r4s")
    ap.add_argument("--b3", default="", help="per-variant B3 knob specs as json {variant: spec}, or one spec")
    ap.add_argument("--c1", default="", help="per (kind.variant) GemvKnobs specs as json, or one spec")
    ap.add_argument("--n2", default="", help="N2Knobs k=v,...")
    ap.add_argument("--kinds", default="delta,delta_ple")
    ap.add_argument("--flavor")
    a = ap.parse_args(argv)

    def table(text):
        text = text.strip()
        if text.startswith("{"):
            return json.loads(text)
        if os.path.isfile(text):
            return json.load(open(text))
        return {"*": text}

    tb, tc = table(a.b3), table(a.c1)

    def b3k(v):
        return SB.knobs_from_spec(tb.get(v, tb.get("*", "")))

    def c1k(kd, v):
        return GV.knobs_from_spec(tc.get(f"{kd}.{v}", tc.get(v, tc.get("*", ""))))

    out = write_n2_functions(a.out, a.variants.split(","), b3k, c1k, knobs_from_spec(a.n2), flavor=a.flavor,
                             kinds=tuple(a.kinds.split(",")))  # fmt: skip
    print("wrote", len(out), "functions:", " ".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
