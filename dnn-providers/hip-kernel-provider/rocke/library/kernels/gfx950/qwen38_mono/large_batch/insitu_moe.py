# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""WP-C5 fix round: the MoE-down all-reduce in situ (DESIGN-CCL 6.3 / 4.9), gfx950.

P-L4's large-batch expert-major MoE (:mod:`.moe_lb`, unchanged) at its TP rank shape (the rank's half of the
intermediate, ``tp=2``), with the all-reduce of the rank's partial MoE output folded into the launch, the same way
:mod:`.insitu` folds the out-projection's:

* **Producer = the combine.** A combine unit sums its 11 slot partials in slot order and rounds them to bf16 exactly as
  the plain kernel does, then, instead of storing ``y``, pushes the bit complement of the canonicalised bf16 values
  (``0xFFFF`` -> ``0x7FC0``) into every rank's receive slot with ``global_store_dwordx2 ... sc0 sc1`` through 64-bit
  peer bases (the lamport form, the in-situ GEMM's default). Every combine variant (``phase``, ``gbar``, ``last``) goes
  through :meth:`combine_cols`, so all of them push.
* **Consumer = after the layer's combine, before the next layer's grid barrier.** Every wave of the grid, after its own
  combine units, takes 1 KiB slices (512 bf16 of one token: H = 2560 = 5 slices) of the exchange: poll the slice of
  every rank in the own arena until no element is empty, reset it, add the ranks in f32 in rank order, store bf16 to
  ``dst[layer][token]``. Bitwise equal to a host rank-order sum of every rank's plain ``y``.
* Slots, tags, epoch, errors, abort: as :mod:`.insitu` (two slots, ``tag = epoch (+) layer``; control words in the
  arena; the launch's last CTA advances them; bounded polls; ``ctrl`` error record + ABORT to every rank). The WAR
  argument needs the kernel's inter-layer grid barrier: launch with ``gsync=1`` (and ``layers >= 1``).

Arena map (bytes, same on every rank): control words ``[0, 256)``; ``ONES`` at 4096 (1 KiB, host-filled with bf16 1.0);
``JUNK`` at 8192 (16 KiB); payload at 65536, per (slot, src rank) ``bucket x 2560`` bf16 in one of two layouts (``lay``):
``unit`` (the default where it applies: combine unit major, ``[unit][16-column tile][16 tokens][4 lane groups x 8 B]``, so
each push instruction writes 512 contiguous bytes; a consumer slice is two of a unit's tiles) or ``tok`` (``[token][2560]``:
the combine's 8-byte pushes land as 32-byte pieces of 16 token rows; 0.92-0.97 slower in the WP-C5 fix round).

Mutants: ``noreset`` (the consumer does not reset its slot: stale-read twin); the MoE's own mutants (``moe.mutant``,
e.g. ``slot``: the combine skips a slot) carry over. Kernel parameters: the MoE's (:func:`.moe_lb.moe_params`), then
``peer0..peer{N-1}``, ``ctrl``, ``arena``, ``dst`` (i64) and i32 ``rank``, ``dl_first``, ``dl``.

Python engine only; names ``q38lbxm_*``; nothing served imports it. Measurements:
``experiments/qwen38-true-mono/large-batch/WP-C5.md`` (Fix round).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from rocke.core.ir import CPOL_SC1, F32, I32, I64, KernelDef, VectorType
from rocke.helpers import device_ops as D

from ..ccl import wire as W
from ..ccl.collectives import _dwords_of_f32, _f32_of_dwords, _rank_sum, pack_kernargs
from . import insitu as X
from . import moe_lb as M

__all__ = ["MoeInsituKnobs", "arena_map", "build_moe_insitu_kernel", "moe_insitu_name", "mutant", "pack_kernargs"]

PAY_OFF = 1 << 16
MUTANTS = ("none", "noreset")


@dataclass(frozen=True)
class MoeInsituKnobs:
    moe: M.MoeKnobs
    ranks: int = 2
    pause: str = "s_nop"
    mutant: str = "none"
    cons: str = "pass"  # "pass": consumer slices after the wave's combine units | "combine" (combine=phase): the combine
    #   unit consumes its own unit right after pushing it: own bf16 values from registers, polls the other ranks' pieces of
    #   the same (tokens, columns) in the own arena, sums in rank order, stores dst; no consumer pass, so a unit's exchange
    #   overlaps the other waves' expert / combine work. Deadlock-free: unit u runs on wave gw(u) on every rank (the same
    #   deal), each wave runs its units in the same order, and a combine unit's other inputs are local
    lay: str = "auto"  # arena layout: "auto" (= "unit" where it applies, else "tok": the WP-C5 fix round's measured
    #   choice, a WIN of 0.92-0.97 against "tok" at rows 8-128 on both rocke nodes) | "tok" ([token][2560] per (slot, src): the combine's 8-byte stores land as 32-byte
    #   pieces of 16 token rows) | "unit" (combine=phase only: [combine unit][d][16 tokens][4 lane groups x 8 bytes], so each
    #   of the unit's dt store instructions writes 512 contiguous bytes; a consumer slice = 2 of a unit's 16-column tiles)
    hz: bool = True  # :func:`.insitu.pad_asm_vmem` (s_nop 4 before inline-asm VMEM on SGPR operands; moe_lb's own
    #   ``hz`` already pads its atomics / polls, which the pass leaves alone)
    tag: str = ""

    @property
    def default_layout(self) -> str:
        return "unit" if self.moe.combine == "phase" and self.moe.dt % 2 == 0 else "tok"

    @property
    def layout(self) -> str:
        return self.default_layout if self.lay == "auto" else self.lay

    def check(self) -> None:
        self.moe.check()
        if not 1 <= self.ranks <= 8:
            raise ValueError("ranks 1..8")
        if self.mutant not in MUTANTS:
            raise ValueError(f"mutant in {MUTANTS}")
        if self.pause not in W._PAUSE:
            raise ValueError("pause")
        if self.moe.stamp or self.moe.ustamp or self.moe.diag != "none" or self.moe.null != "none":
            raise ValueError("no stamp / diag / null builds")
        if self.lay not in ("auto", "tok", "unit"):
            raise ValueError("lay auto | tok | unit")
        if self.cons not in ("pass", "combine"):
            raise ValueError("cons pass | combine")
        if self.cons == "combine" and self.moe.combine != "phase":
            raise ValueError("cons=combine: combine=phase")
        if self.lay == "unit" and (self.moe.combine != "phase" or self.moe.dt % 2):
            raise ValueError("lay=unit: combine=phase (dt 16-column tiles per combine unit, dt even)")


def moe_insitu_name(kn: MoeInsituKnobs) -> str:
    s = M.kernel_name(kn.moe).replace("q38lb_moe_", f"q38lbxm_lamport_n{kn.ranks}_")
    if kn.pause != "s_nop":
        s += f"_p{kn.pause}"
    if kn.layout != kn.default_layout:
        s += f"_l{kn.layout}"
    if kn.cons != "pass":
        s += f"_c{kn.cons}"
    if kn.mutant != "none":
        s += f"_mut{kn.mutant}"
    if not kn.hz:
        s += "_nohz"
    if kn.tag:
        s += f"_x{kn.tag}"
    return s


def arena_map(kn: MoeInsituKnobs) -> dict:
    region = 2 * kn.ranks * kn.moe.bucket * M.H * 2
    return dict(pay=PAY_OFF, region=region, nbytes=PAY_OFF + -(-region // X.ALIGN) * X.ALIGN, ones=X.ONES_OFF,
                junk=X.JUNK_OFF, junk_bytes=X.JUNK_BYTES)


def _c(b, v: int):
    return b.const_i32(int(v))


def poll_lamport_d2(b, base, voffs, deadline, abort_addr, *, pause: str = "s_nop"):
    """Lamport poll of 8-byte pieces (4 bf16) at every offset of the own arena (``global_load_dword x 2 ... sc0 sc1``
    per piece, every bf16 element checked; the abort word and deadline as the CCL polls; each dword its own output
    operand, cf. :func:`.insitu.poll_lamport_dw`). Returns (the pieces as <2 x i32>, status)."""
    n = len(voffs)
    D2 = 2 * n
    mk, it, st, acc, tv, ab, now = (f"${D2 + k}" for k in range(7))
    i0 = D2 + 7
    vo = [f"${i0 + r}" for r in range(n)]
    bs, dl, aa = (f"${i0 + n + k}" for k in range(3))
    s = "y"
    lines = [f"s_mov_b32 {it}, 0", W._lbl(s, "top") + ":", f"global_load_dword {ab}, {aa}, off sc0 sc1"]
    for r in range(n):
        for d in range(2):
            off = f" offset:{4 * d}" if d else ""
            lines.append(f"global_load_dword ${2 * r + d}, {vo[r]}, {bs}{off} sc0 sc1")
    lines.append("s_waitcnt vmcnt(0)")
    lines += W._abort_check(s, ab, mk)
    miss = [f"v_mov_b32 {acc}, 0"]
    for k in range(D2):
        miss += [f"v_subrev_u32_e32 {tv}, 0x10001, ${k}", f"v_bfi_b32 {tv}, ${k}, 0, {tv}", f"v_or_b32_e32 {acc}, {acc}, {tv}"]
    miss += [f"v_and_b32_e32 {acc}, 0x80008000, {acc}", f"v_cmp_ne_u32_e64 {mk}, 0, {acc}", f"s_and_b64 {mk}, {mk}, exec"]
    lines += miss + ["s_cbranch_scc0 " + W._lbl(s, "ok")]
    lines += W._tail(s, pause, dl, mk, now, it, st)
    lines += [W._lbl(s, "ok") + ":", f"s_mov_b32 {st}, 0", "s_branch " + W._lbl(s, "end"),
              W._lbl(s, "timeout") + ":", f"s_mov_b32 {st}, 1", "s_branch " + W._lbl(s, "miss"),
              W._lbl(s, "abort") + ":", f"s_mov_b32 {st}, 2", W._lbl(s, "miss") + ":", *miss, W._lbl(s, "end") + ":"]
    cons = ",".join(["=&v"] * D2 + ["=&s", "=&s", "=&s", "=&v", "=&v", "=&v", "=&s"] + ["v"] * n + ["s", "v", "v"])
    outs = b.inline_asm_multi("\n".join(x for x in lines if x), cons + ",~{scc},~{memory}", [*voffs, base, deadline, abort_addr],
                              result_types=[I32] * D2 + [I64, I32, I32, I32, I32, I32, I64], result_name_hint="polld2")
    return [b.vec_pack([outs[2 * r + d] for d in range(2)], I32) for r in range(n)], outs[D2 + 2]


def G_udiv(b, x, d: int):
    """x // d for a power-of-two d (unsigned)."""
    assert d & (d - 1) == 0
    return b.lshr(x, _c(b, d.bit_length() - 1))


class _MoeInsitu(M._Moe):
    # the in-situ GEMM's protocol helpers (they use only the attributes set below)
    _record = X._Insitu._record
    _encode2 = X._Insitu._encode2
    _tag = X._Insitu._tag
    _slot = X._Insitu._slot

    def __init__(self, kn: MoeInsituKnobs):
        kn.check()
        self.ix = kn
        self.lay = arena_map(kn)
        super().__init__(kn.moe)
        b = self.b
        b.kernel.name = moe_insitu_name(kn)
        N = kn.ranks
        self.peers = [b.param(f"peer{r}", I64) for r in range(N)]
        self.ctrl = b.param("ctrl", I64)
        self.own = b.param("arena", I64)
        self.dst = b.param("dst", I64)
        self.rank = b.readfirstlane(b.param("rank", I32))
        self.dl_first = b.readfirstlane(b.param("dl_first", I32))
        self.dl = b.readfirstlane(b.param("dl", I32))
        self.epoch = b.readfirstlane(W.load_word_sys(b, self.own, _c(b, X.OFF_EPOCH)))
        self.slotbase = b.readfirstlane(W.load_word_sys(b, self.own, _c(b, X.OFF_SLOTBASE)))
        self.abort_addr = b.add(self.own, b.const_i64(X.OFF_ABORT))

    def _payu(self, slot, src, cu):
        """lay=unit: byte offset of combine unit ``cu`` of (slot, src rank) (dt x 512 bytes per unit)."""
        b, k = self.b, self.k
        src = src if not isinstance(src, int) else _c(b, src)
        i = b.add(b.mul(b.add(b.mul(slot, _c(b, self.ix.ranks)), src), _c(b, (k.bucket // M.TILE) * k.nrb)), cu)
        return b.add(_c(b, PAY_OFF), b.mul(i, _c(b, k.dt * 512)))

    def _pay(self, slot, src, tok):
        """Byte offset of token ``tok`` of (slot, src rank) in the arena."""
        b = self.b
        src = src if not isinstance(src, int) else _c(b, src)
        i = b.add(b.mul(b.add(b.mul(slot, _c(b, self.ix.ranks)), src), _c(b, self.k.bucket)), tok)
        return b.add(_c(b, PAY_OFF), b.mul(i, _c(b, M.H * 2)))

    # ------------------------------------------------------------------ producer: the combine pushes
    def combine_cols(self, layer, tok, cols0, ok=None, ndt=None, aux=CPOL_SC1) -> None:
        """moe_lb's combine (11 slot partials in slot order, bf16 rounding), pushed to every rank instead of ``y``."""
        b, k = self.b, self.k
        slot = self._slot(layer)
        own = []
        if self.ix.layout == "unit":  # [unit = tt nrb + rb][d][l16][g] x 8 bytes
            cu = b.add(b.mul(G_udiv(b, tok, M.TILE), _c(b, k.nrb)), G_udiv(b, cols0, k.dt * M.TILE))
            base = b.add(self._payu(slot, self.rank, cu), b.add(b.mul(b.land(tok, _c(b, 15)), _c(b, 32)), b.mul(self.g, _c(b, 8))))
        else:
            base = self._pay(slot, self.rank, tok)
        for d in range(k.dt if ndt is None else ndt):
            col = b.add(cols0, b.add(_c(b, d * M.TILE), b.mul(self.g, _c(b, 4))))
            tot = None
            for s_ in range(M.SLOTS):
                if k.mutant == "slot" and s_ == M.TOPK - 1:
                    continue
                o = b.mul(b.add(b.mul(b.add(b.mul(tok, _c(b, M.SLOTS)), _c(b, s_)), _c(b, M.H)), col), _c(b, 4))
                v = b.vec_bitcast(b.buffer_load_cp(self.part_rs, o, _c(b, 0), dwords=4, aux=aux), VectorType(F32, 4))
                vs = [b.vec_extract(v, i) for i in range(4)]
                tot = vs if tot is None else [b.fadd(tot[i], vs[i]) for i in range(4)]
            yv = b.vec_bitcast(b.vec_trunc_f32_to_bf16(b.vec_pack(tot, F32)), VectorType(I32, 2))
            off = b.add(base, _c(b, d * 512)) if self.ix.layout == "unit" else b.add(base, b.mul(col, _c(b, 2)))
            if ok is not None:
                junk = b.add(_c(b, X.JUNK_OFF), b.shl(b.land(b.add(self.tid, _c(b, 512 * d)), _c(b, X.JUNK_BYTES // 8 - 1)), _c(b, 3)))
                off = b.select(ok, off, junk)
            if self.ix.cons == "combine":
                if self.ix.ranks > 1:
                    self._store2([p_ for _r, p_ in self._others()], off, self._encode2(yv))
                own.append((yv, off, col))
            else:
                self._store2(self.peers, off, self._encode2(yv))
        if self.ix.cons == "combine":
            self._combine_consume(layer, tok, own, ok)

    def _others(self):
        """The other ranks (rank + 1, rank + 2, .. mod N; uniform) and their arena bases."""
        b, N = self.b, self.ix.ranks
        out = []
        for i in range(1, N):
            r = b.readfirstlane(b.mod(b.add(self.rank, _c(b, i)), _c(b, N)))
            p_ = self.peers[0]
            for k_ in range(1, N):
                p_ = b.select(b.cmp_eq(r, _c(b, k_)), self.peers[k_], p_)
            out.append((r, p_))
        return out

    def _combine_consume(self, layer, tok, own, ok) -> None:
        """cons=combine: poll the other ranks' copies of this lane's pieces (the same offsets: every rank pushes the
        unit's piece to the same place of every arena), reset them, add all ranks' bf16 values in f32 in rank order
        (own from registers), store bf16 to dst[layer][tok][col..col + 3]."""
        b, ix, N = self.b, self.ix, self.ix.ranks
        zero = _c(b, 0)
        oth = self._others()
        slot = self._slot(layer)
        dly = b.select(b.cmp_eq(layer, zero), self.dl_first, self.dl)
        dl = b.add(D.memrealtime(b), b.zext(dly, I64))
        live = ok if ok is not None else b.cmp_eq(zero, zero)
        ones = b.add(_c(b, X.ONES_OFF), b.shl(b.land(self.lane, _c(b, 127)), _c(b, 3)))
        got = []
        if N > 1:
            offs = []
            for r, _p in oth:  # the other rank r's piece d of this lane: its own-rank region (src = r)
                for d, (_yv, off_own, _col) in enumerate(own):
                    o = b.add(b.sub(off_own, self._srcbase(slot, self.rank)), self._srcbase(slot, r))
                    offs.append(b.select(live, o, ones))
            raw, st = poll_lamport_d2(b, self.own, offs, dl, self.abort_addr, pause=ix.pause)
            self._record(st, layer)
            if ix.mutant != "noreset":
                junk = b.add(_c(b, X.JUNK_OFF), b.shl(b.land(self.tid, _c(b, X.JUNK_BYTES // 8 - 1)), _c(b, 3)))
                zero2 = b.vec_pack([zero, zero], I32)
                self._store2([self.own], None, zero2, offs=[b.select(live, o, junk) for o in offs])
            got = [b.vec_pack([b.xor(b.vec_extract(x, i), _c(b, -1)) for i in range(2)], I32) for x in raw]
        dlay = b.add(self.dst, b.mul(b.zext(layer, I64), b.const_i64(self.k.bucket * M.H * 2)))
        d_rs = M._rsrc(b, dlay, _c(b, self.k.bucket * M.H * 2))
        nd = len(own)
        for d, (yv, _off, col) in enumerate(own):
            own_f = _f32_of_dwords(b, [b.vec_extract(yv, i) for i in range(2)], "bf16")
            parts = []
            for r in range(N):
                vals = own_f
                for i, (rr, _p) in enumerate(oth):
                    g_ = _f32_of_dwords(b, [b.vec_extract(got[i * nd + d], j) for j in range(2)], "bf16")
                    hit = b.cmp_eq(rr, _c(b, r))
                    vals = [b.select(hit, x, y) for x, y in zip(g_, vals)]
                parts.append(vals)
            out = b.vec_pack(_dwords_of_f32(b, _rank_sum(b, parts), "bf16"), I32)
            doff = b.mul(b.add(b.mul(tok, _c(b, M.H)), col), _c(b, 2))
            b.buffer_store_cp(out, d_rs, b.select(live, doff, _c(b, M.OOB)), zero)

    def _srcbase(self, slot, src):
        """Offset of (slot, src)'s region (the layouts differ only inside it)."""
        b, k = self.b, self.k
        per = 2 * k.bucket * M.H if self.ix.layout == "tok" else (k.bucket // M.TILE) * k.nrb * k.dt * 512
        src = src if not isinstance(src, int) else _c(b, src)
        return b.mul(b.add(b.mul(slot, _c(b, self.ix.ranks)), src), _c(b, per))

    def _store2(self, bases, voff, data, offs=None):
        """``global_store_dwordx2 ... sc0 sc1``: ``data`` at ``voff`` of every arena in ``bases`` (or at every offset of
        ``offs`` of ``bases[0]``)."""
        b = self.b
        if offs is None:
            return X._Insitu._store2(self, bases, voff, data)
        lines = [f"global_store_dwordx2 ${i}, ${len(offs)}, ${len(offs) + 1} sc0 sc1" for i in range(len(offs))] + ["s_nop 1"]
        b.inline_asm("\n".join(lines), ",".join(["v"] * len(offs) + ["v", "s"]) + ",~{memory}", [*offs, data, bases[0]], None)

    # ------------------------------------------------------------------ consumer
    def consume(self, layer) -> None:
        """This wave's 1 KiB consumer slices of exchange use ``layer``: 5 per token (H = 2560 = 5 x 512 bf16)."""
        b, ix = self.b, self.ix
        N = ix.ranks
        zero = _c(b, 0)
        SPT = M.H // 512
        if ix.layout == "unit":
            U = b.readfirstlane(b.mul(b.lshr(b.add(self.rows, _c(b, 15)), _c(b, 4)), _c(b, self.k.nrb * (self.k.dt // 2))))
        else:
            U = b.readfirstlane(b.mul(self.rows, _c(b, SPT)))
        gw = b.readfirstlane(b.add(b.mul(self.cta, _c(b, M.WAVES)), self.wave))
        tw = _c(b, M.GRID * M.WAVES)
        left = b.sub(U, gw)
        nk = b.readfirstlane(b.select(b.cmp_gt(left, zero), b.div(b.add(left, _c(b, M.GRID * M.WAVES - 1)), tw), zero))
        slot = self._slot(layer)
        dly = b.select(b.cmp_eq(layer, zero), self.dl_first, self.dl)
        dl = b.add(D.memrealtime(b), b.zext(dly, I64))
        lane16 = b.shl(self.lane, _c(b, 4))
        ones = b.add(_c(b, X.ONES_OFF), lane16)
        junk = b.add(_c(b, X.JUNK_OFF), lane16)
        zv = b.vec_pack([zero] * 4, I32)
        dlay = b.add(self.dst, b.mul(b.zext(layer, I64), b.const_i64(self.k.bucket * M.H * 2)))
        d_rs = M._rsrc(b, dlay, _c(b, self.k.bucket * M.H * 2))
        with b.scf_for(zero, nk, _c(b, 1), iv_name="xmk") as kk:
            u = b.readfirstlane(b.add(gw, b.mul(kk, tw)))
            if ix.layout == "unit":  # slice u = (unit u / (dt / 2), tile pair); lane: tile d, token l16, column half gp
                k = self.k
                hpu = k.dt // 2
                cu = b.div(u, _c(b, hpu))
                d = b.add(b.mul(b.sub(u, b.mul(cu, _c(b, hpu))), _c(b, 2)), b.lshr(self.lane, _c(b, 5)))
                l16 = b.land(b.lshr(self.lane, _c(b, 1)), _c(b, 15))
                gp = b.land(self.lane, _c(b, 1))
                tt = b.div(cu, _c(b, k.nrb))
                tok = b.add(b.mul(tt, _c(b, M.TILE)), l16)
                col = b.add(b.add(b.mul(b.sub(cu, b.mul(tt, _c(b, k.nrb))), _c(b, k.dt * M.TILE)), b.mul(d, _c(b, M.TILE))),
                            b.mul(gp, _c(b, 8)))
                live = b.cmp_lt(tok, self.rows)
                inner = b.add(b.mul(b.add(b.mul(d, _c(b, 16)), l16), _c(b, 32)), b.mul(gp, _c(b, 16)))
                offs = [b.select(live, b.add(self._payu(slot, r, cu), inner), ones) for r in range(N)]
                doff = b.mul(b.add(b.mul(tok, _c(b, M.H)), col), _c(b, 2))
            else:
                tok = b.div(u, _c(b, SPT))
                inner = b.add(b.mul(b.sub(u, b.mul(tok, _c(b, SPT))), _c(b, 1024)), lane16)
                live = b.cmp_lt(tok, self.rows)
                offs = [b.select(live, b.add(self._pay(slot, r, tok), inner), ones) for r in range(N)]
                doff = b.add(b.mul(tok, _c(b, M.H * 2)), inner)
            raw, _mask, st = W.poll_lamport(b, self.own, offs, 1, "bf16", dl, self.abort_addr, pause=ix.pause)
            self._record(st, layer)
            if ix.mutant != "noreset":
                W.store_zero_lines(b, self.own, [b.select(live, o, junk) for o in offs], 1, zv)
            vecs = [b.vec_pack([b.xor(b.vec_extract(x, i), _c(b, -1)) for i in range(4)], I32) for x in raw]
            parts = [_f32_of_dwords(b, [b.vec_extract(v, i) for i in range(4)], "bf16") for v in vecs]
            s = _rank_sum(b, parts)
            out = b.vec_pack(_dwords_of_f32(b, s, "bf16"), I32)
            b.buffer_store_cp(out, d_rs, b.select(live, doff, _c(b, M.OOB)), zero)

    def combine_phase(self, layer) -> None:
        super().combine_phase(layer)
        if self.ix.cons == "pass":
            self.consume(layer)

    def combine_gbar(self, layer) -> None:
        super().combine_gbar(layer)
        self.consume(layer)

    def build(self) -> KernelDef:
        if self.k.combine == "last":
            raise ValueError("combine=last has no per-wave combine pass to hang the consumer on")
        kd = super().build()
        # the launch's last CTA advances the exchange epoch / slot base by the layers walked (cf. insitu.finish)
        b = self.b
        nxt = W.tag_add(b, self.epoch, self.L)
        sb = b.readfirstlane(b.land(b.add(self.slotbase, self.L), _c(b, 1)))
        with b.scf_if(D.is_thread0(b)):
            old = D.atomic_add_scoped(b, self.own, _c(b, X.OFF_EXIT), _c(b, 1), scope="agent", returns=True)
            with b.scf_if(b.cmp_eq(old, _c(b, M.GRID - 1))):
                W.store_word_sys(b, self.own, _c(b, X.OFF_EXIT), _c(b, 0))
                W.store_word_sys(b, self.own, _c(b, X.OFF_EPOCH), nxt)
                W.store_word_sys(b, self.own, _c(b, X.OFF_SLOTBASE), sb)
                W.drain(b)
        return kd


def build_moe_insitu_kernel(kn: MoeInsituKnobs) -> KernelDef:
    kd = _MoeInsitu(kn).build()
    if kn.hz:
        X.pad_asm_vmem(kd)
    return kd


def mutant(kn: MoeInsituKnobs) -> MoeInsituKnobs:
    """The stale-read must-fail twin (lamport ``noreset``)."""
    return replace(kn, mutant="noreset")
