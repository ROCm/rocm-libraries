# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""WP2d: B3 (gated delta rule) of the grouped wide builds (rows 5 .. 16, ``r6g3`` .. ``r16g4``) as a rocKE stage.

The rocKE B3 (:mod:`..stage_b3`) serves rows <= 4 (``r4g2`` included). Above 4 rows the mono runs FlyDSL's
``delta_group_wide.b3``: the narrow body with every per-row step past the CTA's threads in rounds (the qkvz loads
:func:`_qkvz_round`, the b and a chains 4 request rows at a time on rotating wave pairs, the q / k norms), the mixed
rows held as bf16 words (``bax``: all 16 rows fit one pass), each row's q and k loaded with its recurrence step and
its state stored at once, and the epilogue in passes of 8 request rows. This module is the same stage on
:mod:`..stage_b3`'s helpers (the emitter class ``_B3``: the mono's math, lane exchanges, mailbox / table access),
bitwise to the mono's B3 by construction (same operations, same order per value; every row's recurrence the
grouped build's, the state chained inside a group of ``S`` rows), with the wide structure's choices as knobs:

* ``bax``: the mixed rows ``bf16`` (FlyDSL wide: one word per two values, converted per chain step) | ``f32``
  (the narrow body's: converted once at staging; rows <= 8 only, the arena);
* ``qk_at``: ``step`` (FlyDSL wide: each row's q / k read from LDS at its recurrence step) | ``early`` (every row's
  q / k in registers before the recurrence, the narrow body's; register pressure grows with the rows);
* ``stores``: ``row`` (FlyDSL wide: each row's state stored right after its step) | ``end`` (after the epilogue);
* ``order``: the recurrence order: ``row`` (rows 0 .. R - 1, FlyDSL's program order) | ``group`` (one step of every
  group at a time: the groups' independent chains interleaved);
* the shared knobs of :class:`..stage_b3.B3Knobs` that apply at wide builds: ``split``, ``head_map``, ``wait``,
  ``state_nt``, ``barrier``, ``shfl4``, ``cvt``, ``state_issue``, ``norms``, ``table``, ``hoist``, ``mailbox``
  (``numa``, ``xchg=numa2``, ``pre=dma`` and ``core_put=direct`` are narrow only: ATOM drops the numa lever above 4
  rows, the private exchange region overlaps the wide mailbox part, the wide B1 hosts no prologue).

The function's arena starts at word 0: a wide build's B3 is a separate LDS phase (``layout_wide.PHASE``) with a CTA
barrier ahead of it (``layout_wide.barrier_before``).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, fields, replace
from types import SimpleNamespace

from rocke.core.ir import CPOL_NT, F32, I32, I64, IRBuilder, VectorType
from rocke.helpers import device_ops as dops
from rocke.helpers import mailbox as M
from rocke.helpers import mx_quant as mxq
from rocke.helpers.lds_arena import LdsArena, LdsEntry

from .. import stage_b3 as SB

WIDE_ROWS = 16
MUTANTS = ("chain_head", "ba_row", "epi_row", "qk_row")
# chain_head: a group's second row starts from the group's input state, not from the first row's result
# ba_row: the b / a chain sums of request row 4 t + i stored for row 4 t + (i + 1) % n
# epi_row: the epilogue of pass p > 0 publishes row wave + 8 (p - 1)'s gated output for row wave + 8 p
# qk_row: row r's recurrence reads row (r + 1) % R's q / k


@dataclass(frozen=True)
class WideB3Knobs:
    """Wide B3 knobs (first value = FlyDSL wide's choice; every value keeps the bits). ``b3``: the shared
    :class:`B3Knobs` (only the fields named in the module docstring may leave their defaults)."""

    b3: SB.B3Knobs = field(default_factory=SB.B3Knobs)
    bax: str = "bf16"
    qk_at: str = "step"
    stores: str = "row"
    order: str = "row"
    mutate: str = ""

    def __post_init__(self) -> None:
        k = self.b3
        if k.numa or k.xchg != "core" or k.pre != "none" or k.core_put != "lds":
            raise ValueError("wide B3: numa, xchg=numa2, pre=dma and core_put=direct are narrow only")
        if k.stores != "early":
            raise ValueError("wide B3: the store point is the wide knob 'stores' (row | end)")
        if self.bax not in ("bf16", "f32"):
            raise ValueError("bax must be bf16|f32")
        if self.qk_at not in ("step", "early"):
            raise ValueError("qk_at must be step|early")
        if self.stores not in ("row", "end"):
            raise ValueError("stores must be row|end")
        if self.order not in ("row", "group"):
            raise ValueError("order must be row|group")
        if self.mutate not in ("",) + MUTANTS:
            raise ValueError(f"mutate must be '' or one of {MUTANTS}")


def knobs_spec(k: WideB3Knobs) -> str:
    parts = [SB.knobs_spec(k.b3)] if SB.knobs_spec(k.b3) else []
    d = WideB3Knobs()
    for f in fields(k):
        if f.name == "b3":
            continue
        v = getattr(k, f.name)
        if v != getattr(d, f.name):
            parts.append(f"w_{f.name}={v}")
    return ",".join(parts)


def knobs_from_spec(spec: str) -> WideB3Knobs:
    """``k=v,...``: :func:`..stage_b3.knobs_from_spec` fields, plus ``w_<field>=v`` for the wide knobs."""
    wide, rest = {}, []
    for kv in filter(None, (x.strip() for x in (spec or "").split(","))):
        k, _, v = kv.partition("=")
        if k.startswith("w_"):
            wide[k[2:]] = v
        else:
            rest.append(kv)
    return WideB3Knobs(b3=SB.knobs_from_spec(",".join(rest)), **wide)


@dataclass(frozen=True)
class WideShape:
    """``rows`` (5 .. 16) request rows in groups of ``group`` rows (the grouped verify of ``rows / group``
    requests)."""

    rows: int
    group: int

    def __post_init__(self) -> None:
        if not 4 < self.rows <= WIDE_ROWS:
            raise ValueError(f"wide rows must be in 5..{WIDE_ROWS}")
        if self.group < 2 or self.rows % self.group or self.rows < 2 * self.group:
            raise ValueError(f"bad group {self.group} for rows {self.rows}")

    @property
    def chain(self) -> int:
        return self.group

    @property
    def seq(self) -> bool:
        return False

    @property
    def variant(self) -> str:
        return f"r{self.rows}g{self.group}"


def lds_entries(shape: WideShape, wk: WideB3Knobs) -> tuple:
    k = wk.b3
    rt = SB.DV // k.split
    ents = [
        LdsEntry("b3_qk", 2 * SB.DK, True),
        LdsEntry("b3_vz", 2 * rt, True),
        LdsEntry("b3_core", rt, True),
        LdsEntry("b3_sums", 4 * SB.WAVES, True),
    ]
    if k.split > 1:
        ents.append(LdsEntry("b3_cores", SB.DV, True))
    ents.append(LdsEntry("b3_baw", 2 * SB.RW_ROW))
    if wk.bax == "bf16":
        ents.append(LdsEntry("b3_baxw", SB.RW_ROW, True))
    else:
        ents.append(LdsEntry("b3_bax", len(SB.ROUTER_PARTS) * SB.RX_PART, True))
    return tuple(ents)


class _B3W(SB._B3):
    """:class:`..stage_b3._B3` at 5 .. 16 rows (``emit`` replaced; the helpers reused)."""

    def __init__(self, io, shape: WideShape, wk: WideB3Knobs):
        super().__init__(io, shape, wk.b3)
        self.wk = wk

    def emit(self) -> None:  # noqa: C901 - one stage, the mono's order
        io, b, k, wk = self.io, self.b, self.k, self.wk
        c = self.c
        rows, S, RT, NJ, NB, NL = self.rows, self.S, self.RT, self.NJ, self.NB, self.NL
        G = self.shape.chain
        W8 = SB.WAVES
        tid, lane, wave = io.tid, io.lane, io.wave
        vh, s = self.head(io.task)
        vh = b.readfirstlane(vh)
        s = b.readfirstlane(s)
        kh = b.readfirstlane(b.div(vh, c(SB.GQA)))
        row0 = b.lshr(tid, c(4))
        kl = b.land(tid, c(15))
        mb_xbf, mb_qkvz = self.mb("xbf"), self.mb("qkvz")
        mb_d8, mb_de = self.mb("d8"), self.mb("de")
        mb_core = self.mb("core")
        pk = self.put_knobs
        R = range(rows)

        def load_state():
            nbytes = SB.HV * SB.DV * SB.DK * 4
            rec_in, rec_out = {}, []
            if k.table == "smem":
                base = b.readfirstlane(io.addr("conv_in"))
                for r0_ in range(0, rows, 8):  # at most 8 rows' table words a scalar batch
                    rr = list(range(r0_, min(rows, r0_ + 8)))
                    lines = [f"s_load_dwordx4 ${i}, ${len(rr)}, {(4 * r + 2) * 8}" for i, r in enumerate(rr)]
                    lines.append("s_waitcnt lgkmcnt(0)")
                    outs = b.inline_asm_multi(
                        "\n".join(lines), ",".join(["=&s"] * len(rr) + ["s"]), [base],
                        result_types=[VectorType(I32, 4)] * len(rr), result_name_hint="stab",
                    )  # fmt: skip
                    for i, r in enumerate(rr):
                        w = [b.vec_extract(outs[i], e) for e in range(4)]
                        a_in = b.lor(b.zext(w[0], I64), b.shl(b.zext(w[1], I64), b.const_i64(32)))
                        a_out = b.lor(b.zext(w[2], I64), b.shl(b.zext(w[3], I64), b.const_i64(32)))
                        rec_out.append(self.rsrc(a_out, nbytes))
                        if r % G == 0:
                            rec_in[r] = self.rsrc(a_in, nbytes)
            else:
                tbl = dops.ptr_from_i64(b, b.readfirstlane(io.addr("conv_in")), pointee=I64)
                for r in R:
                    rec_out.append(self.rsrc(b.global_load_i64(tbl, c(4 * r + 3)), nbytes))
                    if r % G == 0:
                        rec_in[r] = self.rsrc(b.global_load_i64(tbl, c(4 * r + 2)), nbytes)
            hrow = [b.add(b.add(b.mul(s, c(RT)), row0), c(32 * j)) for j in range(NJ)]
            sbase = [b.add(b.mul(b.add(b.mul(vh, c(SB.DV)), hrow[j]), c(SB.DK)), kl) for j in range(NJ)]
            st = {}
            for r in R:
                if r % G:
                    continue
                st[r] = [[b.buffer_load(rec_in[r], b.mul(b.add(sbase[j], c(16 * e)), c(4)), c(0), F32)
                          for e in range(8)] for j in range(NJ)]  # fmt: skip
            return rec_out, sbase, st

        if k.state_issue == "entry":
            rec_out, sbase, st = load_state()

        r_alog = self.rsrc(io.addr("a_log"), SB.HV * 4)
        r_dtb = self.rsrc(io.addr("dt_bias"), SB.HV * 4)
        r_nw = self.rsrc(io.addr("norm_w"), SB.DV * 2)
        r_ba = self.rsrc(io.addr("ba_w"), 2 * SB.HV * SB.H * 2)
        vh4 = b.mul(vh, c(4))
        a_log = b.buffer_load(r_alog, vh4, c(0), F32)
        dt_bias = b.buffer_load(r_dtb, vh4, c(0), F32)
        g16 = b.land(lane, c(15))
        bk = b.smin(b.lshr(g16, c(2)), c(NB - 1))
        m4 = b.land(g16, c(3))
        r0 = b.add(b.mul(bk, c(32)), b.mul(m4, c(8)))
        nw_row = b.add(b.mul(s, c(RT)), r0)
        nw = b.buffer_load_vN(r_nw, b.mul(nw_row, c(2)), c(0), I32, 4)

        # b and a: every row's mixed row (bf16 words or f32) and both weight rows into the router's 16-part
        # layout (threads 128 .. 447)
        bj = b.smin(b.smax(b.sub(tid, c(128)), c(0)), c(SB.H // 8 - 1))
        in_ba = b.land(b.cmp_ge(tid, c(128)), b.cmp_lt(tid, c(128 + SB.H // 8)))
        with b.scf_if(in_ba):
            woff = b.mul(b.add(b.mul(vh, c(SB.H // 2)), b.mul(bj, c(4))), c(4))
            wb = b.buffer_load_vN(r_ba, woff, c(0), I32, 4)
            wa = b.buffer_load_vN(r_ba, b.add(woff, c(SB.HV * SB.H * 2)), c(0), I32, 4)
            b.sched_barrier(0)
            specs = []
            for r in R:
                specs += [
                    (mb_xbf, self.poll_ix("xbf", self.ix(b.mul(bj, c(4)), r)), 2),
                    (mb_xbf, self.poll_ix("xbf", self.ix(b.add(b.mul(bj, c(4)), c(2)), r)), 2),
                ]
            xv = self.poll(specs)
            part, first = self._part(b.mul(bj, c(8)))
            off = b.add(b.mul(part, c(SB.RW_PART)), b.lshr(b.sub(b.mul(bj, c(8)), first), c(1)))
            for r in R:
                if wk.bax == "bf16":
                    words = [xv[2 * r][0], xv[2 * r][1], xv[2 * r + 1][0], xv[2 * r + 1][1]]
                    self.stv("b3_baxw", off, b.vec_pack(words, I32), r)
                else:
                    for h in range(2):
                        j = b.add(b.mul(bj, c(2)), c(h))
                        x0, x1 = M.bf2_f32(b, xv[2 * r + h][0])
                        x2, x3 = M.bf2_f32(b, xv[2 * r + h][1])
                        p2, f2 = self._part(b.mul(j, c(4)))
                        dst = b.sub(b.add(b.mul(p2, c(SB.RX_PART)), b.mul(j, c(4))), f2)
                        self.stv("b3_bax", dst, b.vec_pack([x0, x1, x2, x3], F32), r)
            self.stv("b3_baw", off, wb)
            self.stv("b3_baw", b.add(off, c(SB.RW_ROW)), wa)
        self.barrier()

        if k.state_issue == "after_ba":
            rec_out, sbase, st = load_state()

        # b, a chains, 4 request rows at a time: tile t on waves (1 + 2 t, 2 + 2 t) mod 8 (b, a), lane 16 i + q
        # part q of request row 4 t + i (FlyDSL wide ``_ba_chains``)
        for tile in range(-(-rows // 4)):
            n = min(4, rows - 4 * tile)
            ab = b.land(b.add(wave, c(W8 - 1 - 2 * tile)), c(W8 - 1))
            with b.scf_if(b.land(b.cmp_lt(ab, c(2)), b.cmp_lt(lane, c(16 * n)))):
                q = b.land(lane, c(15))
                rr = b.add(c(4 * tile), b.lshr(lane, c(4)))
                big = b.cmp_lt(q, c(8))
                wrow = b.add(b.mul(ab, c(SB.RW_ROW)), b.mul(q, c(SB.RW_PART)))
                acc = self.f(0.0)
                for st_ in range(SB.CHAIN_STEPS):
                    w8 = self.bf16x8(self.ldi4("b3_baw", b.add(wrow, c(4 * st_))))
                    if wk.bax == "bf16":
                        x8 = self.bf16x8(self.ldi4("b3_baxw", b.add(b.mul(q, c(SB.RW_PART)), c(4 * st_)), rr))
                    else:
                        xr = b.mul(q, c(SB.RX_PART))
                        x8 = self.ldf4("b3_bax", b.add(xr, c(8 * st_)), rr) + self.ldf4(
                            "b3_bax", b.add(xr, c(8 * st_ + 4)), rr)  # fmt: skip
                    new = acc
                    for e in range(8):
                        new = b.fma(w8[e], x8[e], new)
                    acc = new if st_ < SB.SHORT_STEPS else b.select(big, new, acc)
                bits = b.bitcast(acc, I32)
                for i in range(n):
                    total = b.bitcast(b.readlane(bits, c(16 * i)), F32)
                    for qq in range(1, 16):
                        total = b.fadd(total, b.bitcast(b.readlane(bits, c(16 * i + qq)), F32))
                    dst_row = 4 * tile + ((i + 1) % n if wk.mutate == "ba_row" else i)
                    with b.scf_if(b.cmp_eq(lane, c(0))):
                        self.stf("b3_sums", ab, self.bf16r(total), dst_row)

        def gates():
            decay_, beta_ = [], []
            for r in R:
                bb = self.ldf("b3_sums", 0, r)
                aa = self.ldf("b3_sums", 1, r)
                sp = self.softplus(b.fadd(aa, dt_bias))
                decay_.append(self.exp_decay(b.fmul(sp, b.fsub(self.f(0.0), self.exp_decay(a_log)))))
                beta_.append(self.sigmoid(bb))
            sv0_ = {(r, j): [b.fmul(st[r][j][e], decay_[r]) for e in range(8)]
                    for r in R if r % G == 0 for j in range(NJ)}  # fmt: skip
            return decay_, beta_, sv0_

        if k.hoist != "none":
            self.barrier()
            decay, beta, sv0 = gates()

        # qkvz: wait for B1's tiles, then rounds: thread tid + 512 m = NL r + i polls row r's 4 values
        ntile = 4 + 2 * max(1, RT // 64)
        if k.wait == "pre_poll":
            tiles = self._tiles(kh, vh, s)

            def hint(m):
                kk = b.mod(m, c(ntile))
                tile = tiles[-1]
                for i in range(ntile - 2, -1, -1):
                    tile = b.select(b.cmp_eq(kk, c(i)), tiles[i], tile)
                return mb_qkvz, self.poll_ix("qkvz", self.ix(b.add(b.mul(tile, c(SB.B1_TILE_PAIRS)), c(31)),
                                                                b.div(m, c(ntile))))  # fmt: skip

            M.pre_poll(b, hint, ntile * rows, io.tag, self.mbk)
        else:
            b.s_barrier_bare(fence="compiler")
        nv = RT // 4
        vbase = b.add(b.mul(vh, c(SB.DV // 2)), b.mul(s, c(RT // 2)))
        for m in range(-(-NL * rows // SB.THREADS)):
            li = b.add(tid, c(SB.THREADS * m))
            with b.scf_if(b.cmp_lt(li, c(NL * rows))):
                i = b.mod(li, c(NL))
                ri = b.div(li, c(NL))
                src = b.select(
                    b.cmp_lt(i, c(64 + nv)),
                    b.select(
                        b.cmp_lt(i, c(64)),
                        b.select(
                            b.cmp_lt(i, c(32)),
                            b.add(b.mul(kh, c(64)), b.mul(i, c(2))),
                            b.add(c(SB.KEY // 2), b.add(b.mul(kh, c(64)), b.mul(b.sub(i, c(32)), c(2)))),
                        ),
                        b.add(c(SB.KEY), b.add(vbase, b.mul(b.sub(i, c(64)), c(2)))),
                    ),
                    b.add(c(SB.QKV // 2), b.add(vbase, b.mul(b.sub(i, c(64 + nv)), c(2)))),
                )  # fmt: skip
                v = self.poll([(mb_qkvz, self.poll_ix("qkvz", self.ix(src, ri)), 2)])[0]
                q0, q1 = M.bf2_f32(b, v[0])
                q2, q3 = M.bf2_f32(b, v[1])
                vec = b.vec_pack([q0, q1, q2, q3], F32)
                w_qk = self.lw("b3_qk", b.mul(i, c(4)), ri)
                w_vz = self.lw("b3_vz", b.mul(b.sub(i, c(64)), c(4)), ri)
                b.smem_store_vN(io.lds.smem, [b.select(b.cmp_lt(i, c(64)), w_qk, w_vz)], vec, 4)
        self.barrier()

        if k.hoist == "none":
            decay, beta, sv0 = gates()

        # q / k L2 norms: rounds, wave w + 8 m = 2 r + n takes row r's q (n = 0) or k (``_qk_norms``)
        rsq128 = self.fbits(SB.RSQ128)
        if k.norms == "redundant":
            inv_q, inv_k = [], []
            for r in R:
                sq = b.fadd(self.sumsq64("b3_qk", 0, r, lane), self.sumsq64("b3_qk", 64, r, lane))
                sk = b.fadd(self.sumsq64("b3_qk", 128, r, lane), self.sumsq64("b3_qk", 192, r, lane))
                inv_q.append(b.fmul(b.rsqrt(b.fadd(sq, self.f(SB.EPS))), rsq128))
                inv_k.append(b.rsqrt(b.fadd(sk, self.f(SB.EPS))))
        else:
            for m in range(-(-2 * rows // W8)):
                qw = b.add(wave, c(W8 * m))
                with b.scf_if(b.cmp_lt(qw, c(2 * rows))):
                    nk = b.land(qw, c(1))
                    wr = b.lshr(qw, c(1))
                    base = b.add(b.mul(nk, c(128)), b.mul(b.lshr(lane, c(5)), c(64)))
                    halves = self.sumsq64("b3_qk", base, wr, lane)
                    rs = b.rsqrt(b.fadd(self.xadd(halves, 32), self.f(SB.EPS)))
                    rs = b.select(b.cmp_eq(nk, c(0)), b.fmul(rs, rsq128), rs)
                    with b.scf_if(b.cmp_eq(lane, c(0))):
                        self.stf("b3_sums", b.add(c(2), nk), rs, wr)
            self.barrier()
            inv_q = [self.ldf("b3_sums", 2, r) for r in R]
            inv_k = [self.ldf("b3_sums", 3, r) for r in R]

        def qk(r):
            rq = (r + 1) % rows if wk.mutate == "qk_row" else r
            qh_ = [b.fmul(self.ldf("b3_qk", b.add(kl, c(16 * e)), rq), inv_q[r]) for e in range(8)]
            kh_ = [b.fmul(self.ldf("b3_qk", b.add(kl, c(128 + 16 * e)), rq), inv_k[r]) for e in range(8)]
            return qh_, kh_

        pre_qk = {r: qk(r) for r in R} if wk.qk_at == "early" else None

        def store_rows(rs_):
            for r in rs_:
                for j in range(NJ):
                    for e in range(8):
                        voff = b.mul(b.add(sbase[j], c(16 * e)), c(4))
                        if k.state_nt:
                            b.buffer_store_cp(b.bitcast(sn[(r, j)][e], I32), rec_out[r], voff, c(0), aux=CPOL_NT)
                        else:
                            b.buffer_store_f32(rec_out[r], voff, c(0), sn[(r, j)][e])

        # the recurrence, the rows of a group chained (a group's first row from the group's state)
        if wk.order == "row":
            seq = list(R)
        else:
            seq = [g * G + p for p in range(G) for g in range(rows // G)]
        sn, core = {}, {}
        for r in seq:
            qh_r, kh_r = pre_qk[r] if pre_qk is not None else qk(r)
            for j in range(NJ):
                trow = b.add(row0, c(32 * j))
                head = r % G == 0 or (wk.mutate == "chain_head" and r % G == 1)
                if head:
                    hr = r - r % G
                    sv = sv0[(hr, j)] if r % G == 0 else [b.fmul(st[hr][j][e], decay[r]) for e in range(8)]
                else:
                    sv = [b.fmul(sn[(r - 1, j)][e], decay[r]) for e in range(8)]
                pr = self.f(0.0)
                for e in range(8):
                    pr = b.fma(sv[e], kh_r[e], pr)
                vv = self.ldf("b3_vz", trow, r)
                delta = b.fmul(beta[r], b.fsub(vv, self.sum16(pr)))
                sn[(r, j)] = [b.fma(delta, kh_r[e], sv[e]) for e in range(8)]
                cr = self.f(0.0)
                for e in range(8):
                    cr = b.fma(sn[(r, j)][e], qh_r[e], cr)
                core[(r, j)] = self.sum16(cr)
            if wk.stores == "row":
                store_rows([r])
        with b.scf_if(b.cmp_eq(kl, c(0))):
            for r in R:
                for j in range(NJ):
                    self.stf("b3_core", b.add(row0, c(32 * j)), core[(r, j)], r)
        self.barrier()

        # siblings exchange the head's core rows (split > 1): put2 threads in rounds
        if S > 1:
            npt = RT // 2
            for m in range(-(-npt * rows // SB.THREADS)):
                li = b.add(tid, c(SB.THREADS * m))
                with b.scf_if(b.cmp_lt(li, c(npt * rows))):
                    jj = b.mod(li, c(npt))
                    rj = b.div(li, c(npt))
                    c0 = self.ldf("b3_core", b.mul(jj, c(2)), rj)
                    c1 = self.ldf("b3_core", b.add(b.mul(jj, c(2)), c(1)), rj)
                    pi = b.add(b.add(b.mul(vh, c(SB.DV)), b.mul(s, c(RT))), b.mul(jj, c(2)))
                    M.put2(b, mb_core, self.put_ix("core", self.ix(pi, rj)), c0, c1, io.tag, pk)

        # epilogue: pass p, wave w finishes request row w + 8 p
        wv = self.bf16x8([b.vec_extract(nw, e) for e in range(4)])
        cvt = k.cvt
        if cvt == "intrinsic":
            from ..mx_quant import intrinsic_cvt

            cvt = intrinsic_cvt
        for p in range(-(-rows // W8)):
            rw = b.add(wave, c(W8 * p))
            pub = rw
            if wk.mutate == "epi_row" and p > 0:
                rw = b.add(wave, c(W8 * (p - 1)))
            with b.scf_if(b.cmp_lt(pub, c(rows))):
                if S > 1:
                    got = self.poll([(mb_core, self.poll_ix("core", self.ix(b.add(b.mul(vh, c(SB.DV)),
                                                                                     b.mul(lane, c(2))), rw)), 2)])[0]  # fmt: skip
                    two = b.mul(lane, c(2))
                    self.stf("b3_cores", two, b.bitcast(got[0], F32), rw)
                    self.stf("b3_cores", b.add(two, c(1)), b.bitcast(got[1], F32), rw)
                    cname = "b3_cores"
                else:
                    cname = "b3_core"
                tot = b.fadd(self.sumsq64(cname, 0, rw, lane), self.sumsq64(cname, 64, rw, lane))
                inv_rms = b.rsqrt(b.fadd(b.fmul(tot, self.f(1.0 / SB.DV)), self.f(SB.EPS)))
                cv = self.ldf4("b3_core", r0, rw) + self.ldf4("b3_core", b.add(r0, c(4)), rw)
                zv = self.ldf4("b3_vz", b.add(r0, c(RT)), rw) + self.ldf4("b3_vz", b.add(r0, c(RT + 4)), rw)
                zsig = [self.sigmoid(z) for z in zv]
                out = [self.bf16r(b.fmul(b.fmul(b.fmul(cv[e], inv_rms), wv[e]), zsig[e])) for e in range(8)]
                amax = b.fabs(out[0])
                for e in range(1, 8):
                    amax = b.fmax(amax, b.fabs(out[e]))
                amax = self.xmax(self.xmax(amax, 1), 2)
                ex, scale = mxq.e8m0_fp8(b, amax)
                lo = mxq.pack_fp8x4(b, out[0], out[1], out[2], out[3], scale, cvt=cvt)
                hi = mxq.pack_fp8x4(b, out[4], out[5], out[6], out[7], scale, cvt=cvt)
                blk = b.add(b.mul(s, c(NB)), bk)
                with b.scf_if(b.cmp_lt(lane, c(4 * NB))):
                    M.put2(b, mb_d8, self.put_ix("d8", self.ix(b.add(b.add(b.mul(vh, c(32)), b.mul(blk, c(8))),
                                                                     b.mul(m4, c(2))), pub)), lo, hi, io.tag, pk)  # fmt: skip
                    with b.scf_if(b.cmp_eq(m4, c(0))):
                        M.put(b, mb_de, self.put_ix("de", self.ix(b.add(b.mul(vh, c(4)), blk), pub)), ex, io.tag, pk)
        if wk.stores == "end":
            b.s_barrier_bare(fence="compiler")
            store_rows(list(R))


def _arena(shape: WideShape, wk: WideB3Knobs) -> LdsArena:
    return LdsArena({"b3": lds_entries(shape, wk)}, rows=shape.rows)


def build_function_kernel(shape: WideShape, wk: WideB3Knobs, layout, arena_words: int | None = None):
    """The hybrid function's ``KernelDef`` (``stage_fn.ABI_PARAMS``)."""
    from ..stage_fn import ABI_PARAMS

    arena = _arena(shape, wk)
    if arena_words is not None and arena.words > arena_words:
        raise ValueError(f"wide B3 {shape.variant}: LDS arena {arena.words} words > the layer's {arena_words}")
    b = IRBuilder(f"rocke_q38_b3_{shape.variant}")
    b.kernel.attrs["max_workgroup_size"] = layout.threads
    vals = {}
    for name, t in ABI_PARAMS:
        vals[name] = b.param(name, I32 if t == "i32" else I64)
    tid = b.thread_id_x()
    for name in ("tag", "task", "layer"):
        vals[name] = b.readfirstlane(vals[name])
    lds = arena.emit(b)
    io = SB.B3Io(
        b=b, tid=tid, lane=b.land(tid, b.const_i32(63)), wave=b.readfirstlane(b.lshr(tid, b.const_i32(6))),
        task=vals["task"], tag=vals["tag"], scratch=b.readfirstlane(vals["scratch"]), lds=lds,
        addr=lambda n: vals[n], sc={n: layout.sc[n] for n in SB.B3_REGIONS}, row_pairs=layout.row_pairs,
    )  # fmt: skip
    with b.scf_if(b.cmp_lt(io.task, b.const_i32(SB.b3_tasks(wk.b3)))):
        _B3W(io, shape, wk).emit()
    b.ret()
    return b.kernel, arena.words


def write_functions(out_dir, shapes, knobs_for, layout, flavor=None, arena="atom", atom_root=None) -> list:
    """``b3_<variant>.ll`` / ``.json`` per wide shape (sidecar ``stage_knobs``, ``needs`` []). ``arena``: ``"atom"``
    (the FlyDSL wide arena, ``moe_wide.arena_words``), an int or None."""
    from ..device_function import DeviceFunctionKnobs, lower_device_function, write_device_function
    from .moe_wide import arena_words

    out = []
    for shape in shapes:
        wk = knobs_for(shape)
        have = arena_words(shape.rows, shape.group, atom_root) if arena == "atom" else arena
        kern, words = build_function_kernel(shape, wk, layout, have)
        fn = lower_device_function(kern, arch="gfx950", symbol=f"rocke_q38_b3_{shape.variant}", llvm_flavor=flavor,
                                   knobs=DeviceFunctionKnobs(lds="arg"))  # fmt: skip
        paths = write_device_function(fn, out_dir, stem=f"b3_{shape.variant}", bitcode=False)
        meta = json.load(open(paths["json"]))
        meta["stage_knobs"] = knobs_spec(wk)
        meta["needs"] = []
        meta["wide"] = True
        with open(paths["json"], "w", newline="\n") as f:
            json.dump(meta, f, indent=1)
            f.write("\n")
        out.append({"variant": shape.variant, "lds_words": words, **paths})
    return out


def main(argv=None) -> int:
    from ..mono_layout import from_atom

    ap = argparse.ArgumentParser(description="emit the WP2d wide B3 stage functions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="r6g3,r8g4,r9g3,r12g3,r12g4,r16g4")
    ap.add_argument("--knobs", default="")
    ap.add_argument("--flavor", default=None)
    ap.add_argument("--atom-root", default=None)
    a = ap.parse_args(argv)
    wk = knobs_from_spec(a.knobs)
    import re

    shapes = []
    for v in a.variants.split(","):
        m = re.fullmatch(r"r(\d+)g(\d+)", v)
        shapes.append(WideShape(int(m.group(1)), int(m.group(2))))
    print(json.dumps(write_functions(a.out, shapes, lambda s: wk, from_atom(a.atom_root), flavor=a.flavor,
                                     atom_root=a.atom_root), indent=1))  # fmt: skip
    return 0


__all__ = ["MUTANTS", "WIDE_ROWS", "WideB3Knobs", "WideShape", "build_function_kernel", "knobs_from_spec",
           "knobs_spec", "lds_entries", "main", "write_functions"]  # fmt: skip

if __name__ == "__main__":
    sys.exit(main())
