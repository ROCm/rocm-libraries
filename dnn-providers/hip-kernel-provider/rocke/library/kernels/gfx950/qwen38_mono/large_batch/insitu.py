# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""WP-C5: in-situ GEMM + all-reduce in one persistent launch (DESIGN-CCL 6.3, 4.9), gfx950.

The large-batch dense GEMM of P-L3 (:mod:`.gemm`, unchanged) at its TP-rank shape, with the exchange of the row-parallel
partials folded into the launch:

* **Producer = the GEMM epilogue.** The unit that finishes an output block (the split-K fold's last arriver, or the
  unit itself at ``splits=1``) rounds its rows to bf16 exactly as the plain GEMM does and, instead of writing Y,
  pushes them (no extra phase):

  - ``form="lamport"`` (CCL-LM's flagless one-shot): every rank's receive slot gets the bit complement of the
    canonicalised bf16 partial (``0xFFFF`` -> the quiet NaN ``0x7FC0`` first, so no pushed element reads as empty =
    zero) with ``global_store_dwordx2 ... sc0 sc1`` through 64-bit peer bases; no drain, no flag.
  - ``form="pull"`` (CCL-LM's pull, aiter's 1-stage scheme): the partial goes to this rank's own arena, the CTA drains
    and one lane per rank stores the use's tag into the block's flag on that rank.

* **Consumer = after the layer's last unit, before the layer's grid barrier** (the layer shape of CCL-LM: the next
  layer needs the whole vector). The consumer units are 1 KiB slices of the exchange (``512 / R`` tokens x ``R`` rows,
  16 bytes = 8 rows per lane), spread over every wave of the grid (``u = gw + k G W``), including CTAs without GEMM
  units. ``lamport``: poll the slice of every rank in the own arena until no bf16 element is zero (every element
  checked), reset it to zero, decode. ``pull``: poll the block flags (the tag of this use), then read every rank's
  slice from its arena (system scope). Both add the ranks' partials in f32 in rank order 0..N-1 and store the bf16
  sum to ``dst[layer][token][row]``: every rank gets the same bits, and a host rank-order sum of the per-rank bf16 Y of
  the plain GEMM reproduces them.

* **``cons="fold"`` (knob):** no consumer pass. The fold's last arriver of a block folds 8 rows x 1 token per thread
  (same per-element split order and rounding), pushes them to the OTHER ranks only (lamport) or stages them and
  publishes the block flag (pull), then polls / reads the other ranks' copies of the same pieces and sums all ranks in
  rank order with its own partial from registers. A block's exchange then overlaps the other CTAs' GEMM work; only the
  last blocks' flight shows. Deadlock-free only while every CTA runs at most one unit per layer (``check`` refuses
  otherwise): a polling CTA must not hold a later unit that another rank waits on.

* **Slots and tags.** Exchange use = layer; slot = ``(slotbase + layer) & 1``, tag = ``epoch (+) layer``
  (:func:`..ccl.wire.tag_add`); ``epoch`` / ``slotbase`` live in the arena's control words (``rocke.runtime.peer``
  offsets) and the last CTA of a launch advances them by ``layers`` (graph-replay safe, no host reset). Two slots are
  enough (WAR): a push of use u + 2 on rank P follows P's grid barrier of use u + 1, which follows P's consumption of
  use u + 1, which needs rank Q's pushes of use u + 1, which follow Q's grid barrier of use u, which follows every Q
  consumer of use u (the lamport reset stores are drained by that barrier's ``vmcnt(0)``).

* **Errors.** Every poll is bounded (``s_memrealtime`` deadline: ``dl_first`` for layer 0, ``dl`` after) and checks
  the local abort word; a failed poll claims the rank's error record (``ctrl``, the ``PeerGroup`` layout) and pushes
  ABORT to every rank. The GEMM's grid barrier has its own deadline (``gs`` word 0).

Arena map (bytes from the arena base, same on every rank): control words ``[0, 256)``; ``ONES`` (1 KiB of bf16
``1.0``, host-filled: the dead lanes' lamport poll target) at 4096; ``JUNK`` (16 KiB: dead pushes / resets) at 8192;
pull flags ``[2 slot][N src][NBLK]`` u32 at 32768; lamport payload ``[2][N][NBLK][BLKB]`` at ``pay_lam``; pull payload at
``pay_pull`` (:func:`arena_map`). ``NBLK = (m / R) x ntb`` output blocks (row group, token block) of ``BLKB = 16 tbt R 2``
bytes, layout ``[token in block][row in group]``.

Mutants (must fail): ``noreset`` (lamport: the consumer does not reset its slot, so the next use of the slot reads the
use before it: a stale-read twin), ``slot`` (pull: payload read from the other slot: a stale-read twin); the GEMM's own
mutants (``gemm.mutant``) carry over. Kernel parameters: the GEMM's (:func:`.gemm.gemm_params`; ``gsync`` is ignored:
the in-situ build always synchronises layers), then ``peer0..peer{N-1}``, ``ctrl``, ``arena``, ``dst`` (i64) and i32
``rank``, ``dl_first``, ``dl`` (100 MHz ticks). Pack with :func:`..ccl.collectives.pack_kernargs`.

Python engine only (the GEMM's ``buffer_*_cp`` / ``int_to_ptr``); names ``q38lbx_*``; nothing served imports it.
Measurements: ``experiments/qwen38-true-mono/large-batch/WP-C5.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from rocke.core.ir import F32, I32, I64, KernelDef, VectorType
from rocke.helpers import device_ops as D
from rocke.helpers.lds_arena import ONE_CTA_PER_CU_WORDS

from ..ccl import wire as W
from ..ccl.collectives import _dwords_of_f32, _f32_of_dwords, _rank_sum, pack_kernargs
from . import gemm as G

__all__ = ["CONS", "FORMS", "InsituKnobs", "arena_map", "build_insitu_kernel", "insitu_name", "pack_kernargs", "MUTANTS"]

FORMS = ("lamport", "pull", "none", "line")  # none: the exchange-free twin (the fold writes Y like the plain GEMM)
GBS = ("gemm", "lite")
CONS = ("pass", "fold", "phase")
PUBLISH = ("unit_end", "comm_wave", "counted")  # form=line only (DESIGN-CCL 4.2.1)
ARS = ("oneshot", "twoshot", "auto")
MUTANTS = ("none", "noreset", "slot")
# ar="auto": (ranks, rows bucket) -> knob values, from the WP-C5 fix round's in-situ measurements
# (experiments/qwen38-true-mono/large-batch/WP-C5.md, "Fix round"); a missing key resolves to the defaults.
# TP2, rows 5-128 (buckets 16-128), both rocke nodes: lamport one-shot with the consumer pass wins or ties against every
# other built value (form=line x publish unit_end / comm_wave / counted, pull, ar=twoshot, cons=fold / phase, gb=lite);
# cons=fold ties within 1 % at 128 rows and keeps its one-unit-per-CTA precondition, so the pass stays.
# TP4 (ONE run, n07-17, 4 GPUs: provisional until a second node confirms): the pass at rows 8-16, cons=fold at 32-64
# (0.955-0.978 of the pass), ar=twoshot at 128 (0.883); at 32-128 every in-situ value still loses to GEMM + aiter.
_TP2 = dict(form="lamport", publish="unit_end", cons="pass", ar="oneshot")
_TP4F = dict(form="lamport", publish="unit_end", cons="fold", ar="oneshot")
AUTO: dict = {(2, 16): _TP2, (2, 32): _TP2, (2, 64): _TP2, (2, 128): _TP2,
              (4, 16): _TP2, (4, 32): _TP4F, (4, 64): _TP4F, (4, 128): dict(_TP2, ar="twoshot")}
LDS_LIMIT = 160 * 1024 - 256
OFF_EPOCH, OFF_ABORT, OFF_EXIT, OFF_CLAIM, OFF_LAUNCHES, OFF_SLOTBASE = 0, 4, 8, 12, 16, 20  # rocke.runtime.peer
ONES_OFF, JUNK_OFF, JUNK_BYTES, FLAG_OFF = 4096, 8192, 16384, 32768
ALIGN = 1 << 16
STAGE_INSITU = 11  # error record stage word
ERR_TIMEOUT, ERR_ABORT, ERR_AHEAD = -3, -6, -7


def _c(b, v: int):
    return b.const_i32(int(v))


def poll_lamport_dw(b, base, voffs, deadline, abort_addr, *, pause: str = "s_nop"):
    """:func:`..ccl.wire.poll_lamport` (bf16, one 16-byte vector per source) with every dword its own output operand
    instead of fixed VGPRs (``v96+``): inside the GEMM's fold, with its prefetch registers live, the fixed-register form
    failed register allocation (COMGR ``CODEGEN_BC_TO_RELOCATABLE``, bucket 64 / 128 builds of ``cons=fold``). Four
    ``global_load_dword ... sc0 sc1`` per source and round (16 B, system scope), every bf16 element checked, the local
    abort word and the deadline as in the CCL polls. Returns (the sources' vectors as <4 x i32>, status)."""
    n = len(voffs)
    D4 = 4 * n
    mk, it, st, acc, tv, ab, now = (f"${D4 + k}" for k in range(7))
    i0 = D4 + 7
    vo = [f"${i0 + r}" for r in range(n)]
    bs, dl, aa = (f"${i0 + n + k}" for k in range(3))
    s = "x"
    lines = [f"s_mov_b32 {it}, 0", W._lbl(s, "top") + ":", f"global_load_dword {ab}, {aa}, off sc0 sc1"]
    for r in range(n):
        for d in range(4):
            off = f" offset:{4 * d}" if d else ""
            lines.append(f"global_load_dword ${4 * r + d}, {vo[r]}, {bs}{off} sc0 sc1")
    lines.append("s_waitcnt vmcnt(0)")
    lines += W._abort_check(s, ab, mk)
    miss = [f"v_mov_b32 {acc}, 0"]
    for k in range(D4):
        miss += [f"v_subrev_u32_e32 {tv}, 0x10001, ${k}", f"v_bfi_b32 {tv}, ${k}, 0, {tv}", f"v_or_b32_e32 {acc}, {acc}, {tv}"]
    miss += [f"v_and_b32_e32 {acc}, 0x80008000, {acc}", f"v_cmp_ne_u32_e64 {mk}, 0, {acc}", f"s_and_b64 {mk}, {mk}, exec"]
    lines += miss + ["s_cbranch_scc0 " + W._lbl(s, "ok")]
    lines += W._tail(s, pause, dl, mk, now, it, st)
    lines += [W._lbl(s, "ok") + ":", f"s_mov_b32 {st}, 0", "s_branch " + W._lbl(s, "end"),
              W._lbl(s, "timeout") + ":", f"s_mov_b32 {st}, 1", "s_branch " + W._lbl(s, "miss"),
              W._lbl(s, "abort") + ":", f"s_mov_b32 {st}, 2", W._lbl(s, "miss") + ":", *miss, W._lbl(s, "end") + ":"]
    cons = ",".join(["=&v"] * D4 + ["=&s", "=&s", "=&s", "=&v", "=&v", "=&v", "=&s"] + ["v"] * n + ["s", "v", "v"])
    outs = b.inline_asm_multi("\n".join(x for x in lines if x), cons + ",~{scc},~{memory}", [*voffs, base, deadline, abort_addr],
                              result_types=[I32] * D4 + [I64, I32, I32, I32, I32, I32, I64], result_name_hint="pollx")
    vecs = [b.vec_pack([outs[4 * r + d] for d in range(4)], I32) for r in range(n)]
    return vecs, outs[D4 + 2]


@dataclass(frozen=True)
class InsituKnobs:
    """One in-situ build: the GEMM plan (``gemm``; split-K needs ``fold="last"``) and the exchange."""

    gemm: G.GemmKnobs
    ranks: int = 2
    m: int = 2560  # output rows (the all-reduced vector), a multiple of the GEMM's row group
    form: str = "lamport"
    pause: str = "s_nop"  # poll pause (ccl.wire pauses)
    mutant: str = "none"
    gb: str = "gemm"  # layer grid barrier: "gemm" (P-L3's: returning agent-scope add, target via LDS) | "lite" (the
    #   target is (launches + 1) x CTAs per stripe from the arena's launch counter: a non-returning add, no LDS hop;
    #   needs the ``gs`` buffer zeroed together with the arena and used by in-situ ``lite`` launches only)
    cons: str = "pass"  # "pass": consumer waves after the layer's last unit | "fold": the fold's last arriver consumes its
    #   own block right after pushing it (own partial from registers, polls the other ranks only), so a block's exchange
    #   overlaps the other CTAs' GEMM work; needs splits > 1 and every CTA at most one unit per layer (deadlock freedom)
    #   | "phase" (FlyTape's shape): the consumer pass runs behind an extra grid barrier of this rank (every CTA's
    #   pushes issued first); the ``gs`` buffer then needs 8 (2 layers + 2) + 64 words (rows ``layers + bnd``)
    publish: str = "unit_end"  # form=line: "unit_end" (every wave drains, CTA barrier, then the block's line flags) |
    #   "comm_wave" (the fold stages the block in LDS; one wave pushes it, drains its own stores, flags; the other waves
    #   never drain) | "counted" (per wave, no CTA barrier: line j's flag follows the wave's wait on fold iteration
    #   j + 1's partial loads, issued after line j's stores; the last line drains)
    hz: bool = True  # pad every inline-asm block that issues VMEM on an SGPR operand (the GEMM's grid-barrier atomic /
    #   poll, the CCL pushes / polls / resets, the error record) with a leading ``s_nop 4``: a VALU write of an SGPR (the
    #   compiler's ``v_readlane`` restore of a spilled peer / arena base) needs 5 wait states before a VMEM reads it as a
    #   base, and LLVM does not pad VMEM inside inline asm (alola rules 176 / 182; P-L4's ``hz``). The WP-C5 fix round's
    #   ISA scan found unpadded hits in TP2 must-fail twins and in most TP4 builds; ``hz=False`` = the pre-fix emission
    ar: str = "oneshot"  # "oneshot" (every producer pushes to every rank) | "twoshot" (lamport: block b goes to its owner
    #   rank b mod N only; the owner's consumers sum in rank order and push the bf16 sum to every rank; every rank's
    #   consumers then read the owner's sum) | "auto" (resolved per (ranks, bucket) from :data:`AUTO`)
    tag: str = ""

    @property
    def R(self) -> int:
        return self.gemm.rows_per_unit

    @property
    def tpu(self) -> int:  # tokens per 1 KiB consumer unit
        return 512 // self.R

    def check(self) -> None:
        g = self.gemm
        g.check()
        if self.gb not in GBS:
            raise ValueError(f"gb in {GBS}")
        if self.form == "none" and (self.cons != "pass" or self.mutant != "none"):
            raise ValueError("form=none: no consumer, no exchange mutants")
        if self.form not in FORMS:
            raise ValueError(f"form in {FORMS}")
        if self.mutant not in MUTANTS:
            raise ValueError(f"mutant in {MUTANTS}")
        if self.mutant == "noreset" and self.form != "lamport":
            raise ValueError("noreset is a lamport mutant")
        if self.mutant == "slot" and self.form not in ("pull", "line"):
            raise ValueError("slot is a pull / line mutant")
        if not 1 <= self.ranks <= 8:
            raise ValueError("ranks 1..8")
        if g.splits > 1 and g.fold != "last":
            raise ValueError("in-situ push needs fold=last (the fold's last arriver pushes)")
        if self.R > 512 or 512 % self.R or self.m % self.R:
            raise ValueError("rows per unit must divide 512 and m")
        if (g.tbt * 16) % self.tpu:
            raise ValueError("token block not a multiple of the consumer unit's tokens")
        if g.stamp or g.tstamp or g.diag != "none":
            raise ValueError("no stamp / diag builds")
        if self.pause not in W._PAUSE:
            raise ValueError("pause")
        if self.cons not in CONS:
            raise ValueError(f"cons in {CONS}")
        if self.cons == "fold":
            if g.splits < 2:
                raise ValueError("cons=fold needs split-K (the fold's last arriver consumes)")
            if (self.m // self.R) * g.ntb * g.splits > g.grid:
                raise ValueError("cons=fold needs at most one unit per CTA per layer (units <= grid): a CTA polling in "
                                 "its fold must not hold a later unit another rank waits on")
            if self.R % 8:
                raise ValueError("cons=fold: rows per unit a multiple of 8")
        if self.publish not in PUBLISH:
            raise ValueError(f"publish in {PUBLISH}")
        if self.ar not in ARS:
            raise ValueError(f"ar in {ARS}")
        if self.ar == "auto":
            raise ValueError("ar=auto: build resolve(knobs)")
        if self.publish != "unit_end" and self.form != "line":
            raise ValueError("publish variants are form=line's (lamport needs no publish; pull is unit_end)")
        if self.cons == "phase" and (self.gb != "gemm" or self.form == "none"):
            raise ValueError("cons=phase: gb=gemm and an exchange")
        if self.form == "line":
            if g.splits < 2 or self.R % 8:
                raise ValueError("form=line: split-K (the fold writes 16 bytes = 8 rows per lane) and rows per unit % 8")
            if self.cons == "fold":
                raise ValueError("form=line: cons pass | phase")
            if self.publish == "comm_wave":
                if self.lines_per_block * self.ranks > 64:
                    raise ValueError("comm_wave: one lane per (line, rank) flag")
                if g.lds_bytes() + self.blkb + (ONE_CTA_PER_CU_WORDS * 4 if g.occ1 else 0) > LDS_LIMIT:
                    raise ValueError("comm_wave: the LDS block tile does not fit")
        if self.ar == "twoshot" and (self.form != "lamport" or self.cons == "fold" or self.ranks < 2):
            raise ValueError("ar=twoshot: form=lamport, cons pass | phase, ranks >= 2")

    @property
    def blkb(self) -> int:  # bytes of one output block (row group x token block) of bf16
        return self.gemm.tbt * 16 * self.R * 2

    @property
    def lines_per_block(self) -> int:
        return self.blkb // 1024


def resolve(kn: InsituKnobs) -> InsituKnobs:
    """``ar="auto"`` -> the measured choice for (ranks, rows bucket) (:data:`AUTO`); anything else unchanged."""
    if kn.ar != "auto":
        return kn
    return replace(kn, **{"ar": "oneshot", **AUTO.get((kn.ranks, kn.gemm.bucket), {})})


def insitu_name(kn: InsituKnobs) -> str:
    kn = resolve(kn)
    s = G.kernel_name(kn.gemm).replace("q38lb_gemm_", f"q38lbx_{kn.form}_n{kn.ranks}_m{kn.m}_")
    if kn.gb != "gemm":
        s += f"_gb{kn.gb}"
    if kn.cons != "pass":
        s += f"_c{kn.cons}"
    if kn.publish != "unit_end":
        s += f"_pub{kn.publish}"
    if kn.ar != "oneshot":
        s += "_ar2"
    if not kn.hz:
        s += "_nohz"
    if kn.pause != "s_nop":
        s += f"_p{kn.pause}"
    if kn.mutant != "none":
        s += f"_mut{kn.mutant}"
    if kn.tag:
        s += f"_{kn.tag}"
    return s


def arena_map(kn: InsituKnobs) -> dict:
    """Byte offsets / sizes of the arena regions (host and device use the same numbers)."""
    g = kn.gemm
    nblk = (kn.m // kn.R) * g.ntb
    blkb = g.tbt * 16 * kn.R * 2
    flags = 2 * kn.ranks * nblk * 4
    region = 2 * kn.ranks * nblk * blkb
    pay_lam = -(-(FLAG_OFF + flags) // ALIGN) * ALIGN
    pay_pull = pay_lam + -(-region // ALIGN) * ALIGN
    total = pay_pull + -(-region // ALIGN) * ALIGN
    # fix-round regions, after the original ones (their offsets, and so the original builds, are unchanged):
    # form=line payload [2][N][NBLK][BLKB] and line flags [2][N][NBLK][lines per block] u32; twoshot's all-gather
    # payload [2][NBLK][BLKB] (its reduce-scatter half uses the lamport region)
    lpb = max(1, blkb // 1024)
    lflags = 2 * kn.ranks * nblk * lpb * 4
    pay_line = total
    lflag = pay_line + -(-region // ALIGN) * ALIGN
    pay_ag = lflag + -(-lflags // ALIGN) * ALIGN
    total = pay_ag + -(-(2 * nblk * blkb) // ALIGN) * ALIGN
    return dict(nblk=nblk, blkb=blkb, flags=flags, region=region, pay_lam=pay_lam, pay_pull=pay_pull, nbytes=total,
                ones=ONES_OFF, junk=JUNK_OFF, junk_bytes=JUNK_BYTES, flag=FLAG_OFF, pay_line=pay_line, lflag=lflag,
                lflags=lflags, lpb=lpb, pay_ag=pay_ag, ag_region=2 * nblk * blkb)


class _Insitu(G._Gemm):
    def __init__(self, kn: InsituKnobs):
        kn.check()
        self.ix = kn
        self.lay = arena_map(kn)
        super().__init__(kn.gemm)
        b = self.b
        b.kernel.name = insitu_name(kn)
        N = kn.ranks
        self.peers = [b.param(f"peer{r}", I64) for r in range(N)]
        self.ctrl = b.param("ctrl", I64)
        self.own = b.param("arena", I64)
        self.dst = b.param("dst", I64)
        self.rank = b.readfirstlane(b.param("rank", I32))
        self.dl_first = b.readfirstlane(b.param("dl_first", I32))
        self.dl = b.readfirstlane(b.param("dl", I32))
        zero = _c(b, 0)
        # the in-situ build always synchronises layers (WAR of the 2 slots rests on the grid barrier)
        self.gsync = b.cmp_eq(zero, zero)
        self.nogs = b.cmp_ne(zero, zero)
        self.epoch = b.readfirstlane(W.load_word_sys(b, self.own, _c(b, OFF_EPOCH)))
        self.slotbase = b.readfirstlane(W.load_word_sys(b, self.own, _c(b, OFF_SLOTBASE)))
        if kn.gb == "lite":
            self.launches = b.readfirstlane(W.load_word_sys(b, self.own, _c(b, OFF_LAUNCHES)))
        self.abort_addr = b.add(self.own, b.const_i64(OFF_ABORT))
        lmod = b.mod(self.lane, _c(b, N)) if N > 1 else zero
        self.lane_src = lmod
        lp = self.peers[0]
        for r in range(1, N):
            lp = b.select(b.cmp_eq(lmod, _c(b, r)), self.peers[r], lp)
        self.lane_peer = lp
        self._ivn = 0
        if kn.form == "line" and kn.publish == "comm_wave":  # the fold's block tile (bf16, [token][row])
            self.cw = b.smem_alloc(I32, [self.lay["blkb"] // 4], name_hint="xcw")

    # ------------------------------------------------------------------ helpers
    def _iv(self, stem: str) -> str:
        self._ivn += 1
        return f"{stem}{self._ivn}"

    def _slot(self, layer):
        b = self.b
        return b.land(b.add(self.slotbase, layer), _c(b, 1))

    def _tag(self, layer):
        return W.tag_add(self.b, self.epoch, layer)

    def _pay(self, slot, src, blk, pull: bool):
        b, lay = self.b, self.lay
        base = lay["pay_pull"] if pull else lay["pay_lam"]
        src = src if not isinstance(src, int) else _c(b, src)
        i = b.add(b.mul(b.add(b.mul(slot, _c(b, self.ix.ranks)), src), _c(b, lay["nblk"])), blk)
        return b.add(_c(b, base), b.mul(i, _c(b, lay["blkb"])))

    def _payl(self, slot, src, blk):
        """form=line payload of (slot, src rank, block)."""
        b, lay = self.b, self.lay
        src = src if not isinstance(src, int) else _c(b, src)
        i = b.add(b.mul(b.add(b.mul(slot, _c(b, self.ix.ranks)), src), _c(b, lay["nblk"])), blk)
        return b.add(_c(b, lay["pay_line"]), b.mul(i, _c(b, lay["blkb"])))

    def _payag(self, slot, blk):
        """twoshot all-gather payload of (slot, block): the owner's bf16 sum."""
        b, lay = self.b, self.lay
        return b.add(_c(b, lay["pay_ag"]), b.mul(b.add(b.mul(slot, _c(b, lay["nblk"])), blk), _c(b, lay["blkb"])))

    def _lflag(self, slot, src, blk, j):
        """form=line flag of line ``j`` of (slot, src rank, block)."""
        b, lay = self.b, self.lay
        i = b.add(b.mul(b.add(b.mul(b.add(b.mul(slot, _c(b, self.ix.ranks)), src), _c(b, lay["nblk"])), blk), _c(b, lay["lpb"])), j)
        return b.add(_c(b, lay["lflag"]), b.shl(i, _c(b, 2)))

    def _peer_of(self, r):
        """The arena base of rank ``r`` (uniform)."""
        b = self.b
        p = self.peers[0]
        for k in range(1, self.ix.ranks):
            p = b.select(b.cmp_eq(r, _c(b, k)), self.peers[k], p)
        return p

    def _encode4(self, v4):
        """lamport wire of a <4 x i32> (8 bf16)."""
        b = self.b
        lo = self._encode2(b.vec_pack([b.vec_extract(v4, 0), b.vec_extract(v4, 1)], I32))
        hi = self._encode2(b.vec_pack([b.vec_extract(v4, 2), b.vec_extract(v4, 3)], I32))
        return b.vec_pack([b.vec_extract(lo, 0), b.vec_extract(lo, 1), b.vec_extract(hi, 0), b.vec_extract(hi, 1)], I32)

    def _flag(self, slot, src, blk):
        b, lay = self.b, self.lay
        i = b.add(b.mul(b.add(b.mul(slot, _c(b, self.ix.ranks)), src), _c(b, lay["nblk"])), blk)
        return b.add(_c(b, FLAG_OFF), b.shl(i, _c(b, 2)))

    def _record(self, st, layer) -> None:
        """A failed poll (``st`` uniform != 0): first error of the rank -> ``ctrl``; ABORT to every rank."""
        b = self.b
        # uniform asm operands (tag_add: SGPR constraints) before any one-lane region: inside it LLVM may hand the asm a
        # VGPR (``s_add_u32 s9, s31, v52``: "cannot compile inline asm", the cons=fold lamport builds at 2 fold iterations)
        tag = self._tag(layer)
        with b.scf_if(b.cmp_ne(st, _c(b, 0))):
            code = b.select(b.cmp_eq(st, _c(b, W.ST_TIMEOUT)), _c(b, ERR_TIMEOUT),
                            b.select(b.cmp_eq(st, _c(b, W.ST_ABORT)), _c(b, ERR_ABORT), _c(b, ERR_AHEAD)))
            with b.scf_if(b.cmp_eq(self.lane, _c(b, 0))):
                old = D.atomic_add_scoped(b, self.own, _c(b, OFF_CLAIM), _c(b, 1), scope="agent", returns=True)
                with b.scf_if(b.cmp_eq(old, _c(b, 0))):
                    for i, v in ((1, code), (2, _c(b, STAGE_INSITU)), (3, layer), (4, _c(b, -1)), (5, tag),
                                 (7, self.cta), (10, self.rank)):
                        W.store_word_sys(b, self.ctrl, _c(b, 4 * i), v)
                    W.drain(b)
                    W.store_word_sys(b, self.ctrl, _c(b, 0), _c(b, 1))
                for p in self.peers:
                    W.store_word_sys(b, p, _c(b, OFF_ABORT), _c(b, 1))
                W.drain(b)

    def _encode2(self, dw):
        """lamport wire of a <2 x i32> (4 bf16): canonicalise 0xFFFF -> 0x7FC0, complement."""
        b = self.b
        out = []
        m16 = _c(b, 0xFFFF)
        for i in range(2):
            d = b.vec_extract(dw, i)
            lo = b.land(d, m16)
            hi = b.lshr(d, _c(b, 16))
            lo = b.select(b.cmp_eq(lo, m16), _c(b, 0x7FC0), lo)
            hi = b.select(b.cmp_eq(hi, m16), _c(b, 0x7FC0), hi)
            out.append(b.xor(b.lor(lo, b.shl(hi, _c(b, 16))), _c(b, -1)))
        return b.vec_pack(out, I32)

    def _store2(self, bases, voff, data) -> None:
        """``global_store_dwordx2 ... sc0 sc1`` of this lane's 8 bytes at ``voff`` of every arena in ``bases``."""
        b = self.b
        lines = [f"global_store_dwordx2 $0, $1, ${2 + i} sc0 sc1" for i in range(len(bases))] + ["s_nop 1"]
        b.inline_asm("\n".join(lines), "v,v," + ",".join(["s"] * len(bases)) + ",~{memory}", [voff, data, *bases], None)

    def _push(self, u, tl, r_in, vals4, ok, junk_idx) -> None:
        """Push one piece: 4 consecutive rows (``r_in`` within the row group) of token ``tl`` of the unit's block."""
        b, k, ix = self.b, self.k, self.ix
        bf = b.vec_trunc_f32_to_bf16(b.vec_pack(vals4, F32))
        dw = b.vec_bitcast(bf, VectorType(I32, 2))
        slot = self._slot(u.layer)
        blk = b.add(b.mul(u.rg, _c(b, k.ntb)), u.tbk)
        pull = ix.form == "pull"
        off = b.add(self._pay(slot, self.rank, blk, pull), b.mul(b.add(b.mul(tl, _c(b, ix.R)), r_in), _c(b, 2)))
        junk = b.add(_c(b, JUNK_OFF), b.shl(b.land(junk_idx, _c(b, JUNK_BYTES // 8 - 1)), _c(b, 3)))
        off = b.select(ok, off, junk)
        if pull:
            self._store2([self.own], off, dw)
        elif ix.ar == "twoshot":  # reduce-scatter: the block's owner rank only
            owner = b.readfirstlane(b.mod(blk, _c(b, ix.ranks)))
            self._store2([self._peer_of(owner)], off, self._encode2(dw))
        else:
            self._store2(self.peers, off, self._encode2(dw))

    def _publish(self, u) -> None:
        """pull: every thread's own-arena stores of the block are performed, then lanes 0..N-1 store the tag into the
        block's flag on rank ``lane``."""
        b, k = self.b, self.k
        W.drain(b)
        b.sync_lds_only()
        slot = self._slot(u.layer)
        tag = self._tag(u.layer)
        blk = b.add(b.mul(u.rg, _c(b, k.ntb)), u.tbk)
        fo = self._flag(slot, self.rank, blk)
        with b.scf_if(b.land(u.valid, b.cmp_lt(self.tid, _c(b, self.ix.ranks)))):
            W.store_flags_lanes(b, b.add(self.lane_peer, b.zext(fo, I64)), tag)

    # ------------------------------------------------------------------ producer (overrides)
    def gbar(self, bnd) -> None:
        if self.ix.gb != "lite":
            super().gbar(bnd)
            return
        b, k = self.b, self.k
        per = k.grid // G.XCDS
        target = b.readfirstlane(b.mul(b.add(self.launches, _c(b, 1)), _c(b, per)))
        b.s_waitcnt(vmcnt=0)
        b.sync_lds_only()
        gsa = b.readfirstlane(self.a["gs"])
        row = b.mul(b.add(_c(b, 1), bnd), _c(b, 8))
        with b.scf_if(D.is_thread0(b)):
            D.atomic_add_scoped(b, gsa, b.mul(b.add(row, G._umod(b, self.cta, G.XCDS)), _c(b, 4)), _c(b, 1), scope="agent")
        with b.scf_if(b.cmp_lt(self.tid, _c(b, G.XCDS))):
            dl = b.add(D.memrealtime(b), b.const_i64(200_000_000))
            st = W.poll_count_ge(b, gsa, b.mul(b.add(row, self.tid), _c(b, 4)), target, dl, pause="s_sleep1")
            with b.scf_if(b.land(b.cmp_ne(st, _c(b, 0)), D.is_thread0(b))):
                D.atomic_add_scoped(b, gsa, _c(b, 0), _c(b, 1), scope="agent")
        b.sync_lds_only()

    def epilogue(self, u, acc) -> None:
        b, k = self.b, self.k
        if self.ix.form == "none":
            G._Gemm.epilogue(self, u, acc)
            return
        if k.splits > 1:
            super().epilogue(u, acc)  # partials, arrival, last arriver -> self.fold (pushes)
            return
        for mi in range(k.mt):
            r_in = b.add(b.sub(self.wrow[mi], self.l16), b.mul(self.g, _c(b, 4)))
            for t in range(k.tbt):
                tl = b.add(_c(b, t * G.TILE), self.l16)
                tok = b.add(u.tok0, tl)
                ok = b.land(u.valid, b.cmp_lt(tok, self.rows))
                vals = [b.vec_extract(acc[mi][t], e) for e in range(4)]
                self._push(u, tl, r_in, vals, ok, b.add(self.tid, _c(b, 512 * (mi * k.tbt + t))))
        if self.ix.form == "pull":
            self._publish(u)

    def fold(self, u, p_rs) -> None:
        if self.ix.form == "none":
            G._Gemm.fold(self, u, p_rs)
            return
        if self.ix.cons == "fold":
            self._fold_consume(u, p_rs)
            return
        if self.ix.form == "line":
            self._fold_line(u, p_rs)
            return
        self._fold_push(u, p_rs)

    def _fold_push(self, u, p_rs) -> None:
        """The plain GEMM's fold (same loads, same split order, same bf16 rounding), pushing instead of writing Y."""
        b, k = self.b, self.k
        TP = k.tiles * G.TILE
        R = k.rows_per_unit
        q4 = R // 4
        pieces = k.tbt * G.TILE * q4
        S = k.splits - (1 if k.mutant == "fold" else 0)
        for it in range(-(-pieces // k.threads)):
            v = b.add(self.tid, _c(b, it * k.threads))
            tl = G._udiv(b, v, q4)
            r_in = b.mul(G._umod(b, v, q4), _c(b, 4))
            row = b.add(b.mul(u.rg, _c(b, R)), r_in)
            tok = b.add(u.tok0, tl)
            ok = b.land(b.cmp_lt(v, _c(b, pieces)), b.cmp_lt(tok, self.rows))
            tot = None
            for s0 in range(0, S, G.FOLD_GROUP):
                loads = []
                for s in range(s0, min(S, s0 + G.FOLD_GROUP)):
                    off = b.mul(b.add(b.mul(b.add(_c(b, s * TP), tok), self.Mpad), row), _c(b, 4))
                    loads.append(b.buffer_load_cp(p_rs, b.select(ok, off, _c(b, G.OOB)), _c(b, 0), dwords=4, aux=G.CPOL_SC1))
                for ld in loads:
                    f = b.vec_bitcast(ld, VectorType(F32, 4))
                    if tot is None:
                        tot = [b.vec_extract(f, e) for e in range(4)]
                    else:
                        tot = [b.fadd(tot[e], b.vec_extract(f, e)) for e in range(4)]
            self._push(u, tl, r_in, tot, ok, v)
        if self.ix.form == "pull":
            self._publish(u)

    def _fold_line(self, u, p_rs) -> None:
        """form=line: the plain GEMM's fold (same per-element split order and bf16 rounding) of 8 rows x 1 token per
        thread: thread ``v`` holds bytes ``[16 v, 16 v + 16)`` of the block ([token][row] layout), so wave ``w`` of fold
        iteration ``i`` writes the block's 1 KiB line ``i W + w``. Every line goes to every rank's line payload with
        ``global_store_dwordx4 ... sc0 sc1``; its flag (the use's tag, on every rank) follows the publish variant:
        ``unit_end`` (every wave drains, CTA barrier, lanes ``(line, rank)`` store the block's flags), ``counted`` (per
        wave, no barrier: line ``i``'s flag after the wave's wait on iteration ``i + 1``'s partial loads, which were
        issued after line ``i``'s stores; the last line drains), ``comm_wave`` (the fold writes the block into LDS, the
        last wave pushes it, drains only its own stores and stores the flags; the other waves never drain)."""
        b, k, ix = self.b, self.k, self.ix
        N = ix.ranks
        TP = k.tiles * G.TILE
        R = k.rows_per_unit
        q8 = R // 8
        pieces = k.tbt * G.TILE * q8
        lpb = self.lay["lpb"]
        S = k.splits - (1 if k.mutant == "fold" else 0)
        slot = self._slot(u.layer)
        tag = self._tag(u.layer)
        blk = b.add(b.mul(u.rg, _c(b, k.ntb)), u.tbk)
        base = self._payl(slot, self.rank, blk)
        zero = _c(b, 0)
        pub = ix.publish
        nit = -(-pieces // k.threads)

        def wave_flag(it):  # counted: the flag of this wave's line of fold iteration ``it``
            line = b.add(_c(b, it * k.waves), self.wave)
            W.drain(b)
            with b.scf_if(b.land(b.cmp_lt(line, _c(b, lpb)), b.cmp_lt(self.lane, _c(b, N)))):
                fo = self._lflag(slot, self.rank, blk, line)
                W.store_flags_lanes(b, b.add(self.lane_peer, b.zext(fo, I64)), tag)

        for it in range(nit):
            v = b.add(self.tid, _c(b, it * k.threads))
            tl = G._udiv(b, v, q8)
            r_in = b.mul(G._umod(b, v, q8), _c(b, 8))
            row = b.add(b.mul(u.rg, _c(b, R)), r_in)
            tok = b.add(u.tok0, tl)
            inb = b.cmp_lt(v, _c(b, pieces))
            ok = b.land(inb, b.cmp_lt(tok, self.rows))
            tot = None
            grp_ = max(1, G.FOLD_GROUP // 2)
            for s0 in range(0, S, grp_):
                loads = []
                for s_ in range(s0, min(S, s0 + grp_)):
                    off = b.mul(b.add(b.mul(b.add(_c(b, s_ * TP), tok), self.Mpad), row), _c(b, 4))
                    for h in range(2):
                        o = b.select(ok, b.add(off, _c(b, 16 * h)), _c(b, G.OOB))
                        loads.append(b.buffer_load_cp(p_rs, o, zero, dwords=4, aux=G.CPOL_SC1))
                for j in range(0, len(loads), 2):
                    f = [b.vec_extract(b.vec_bitcast(loads[j + h], VectorType(F32, 4)), e) for h in range(2) for e in range(4)]
                    tot = f if tot is None else [b.fadd(x, y) for x, y in zip(tot, f)]
            own = b.vec_pack(_dwords_of_f32(b, tot, "bf16"), I32)
            if pub == "comm_wave":
                with b.scf_if(inb):
                    b.smem_store_vN(self.cw, [b.shl(v, _c(b, 2))], own, 4)
                continue
            if pub == "counted" and it > 0:
                wave_flag(it - 1)
            junk = b.add(_c(b, JUNK_OFF), b.shl(b.land(v, _c(b, JUNK_BYTES // 16 - 1)), _c(b, 4)))
            W.push_vecs(b, self.peers, b.select(ok, b.add(base, b.shl(v, _c(b, 4))), junk), [own])
        if pub == "counted":
            wave_flag(nit - 1)
        elif pub == "unit_end":
            W.drain(b)
            b.sync_lds_only()
            with b.scf_if(b.cmp_lt(self.tid, _c(b, lpb * N))):
                fo = self._lflag(slot, self.rank, blk, b.div(self.tid, _c(b, N)))
                W.store_flags_lanes(b, b.add(self.lane_peer, b.zext(fo, I64)), tag)
        else:  # comm_wave
            b.sync_lds_only()
            with b.scf_if(b.cmp_eq(self.wave, _c(b, k.waves - 1))):
                lane16 = b.shl(self.lane, _c(b, 4))
                for j in range(lpb):
                    d = b.smem_load_vN(self.cw, b.add(_c(b, 256 * j), b.shl(self.lane, _c(b, 2))), dtype=I32, n=4)
                    W.push_vecs(b, self.peers, b.add(base, b.add(_c(b, 1024 * j), lane16)), [d])
                W.drain(b)
                with b.scf_if(b.cmp_lt(self.lane, _c(b, lpb * N))):
                    fo = self._lflag(slot, self.rank, blk, b.div(self.lane, _c(b, N)))
                    W.store_flags_lanes(b, b.add(self.lane_peer, b.zext(fo, I64)), tag)

    def _others(self):
        """The other ranks in the order (rank + 1, rank + 2, ..) mod N (uniform) and their arena bases."""
        b, N = self.b, self.ix.ranks
        out = []
        for i in range(1, N):
            r = b.readfirstlane(b.mod(b.add(self.rank, _c(b, i)), _c(b, N)))
            p = self.peers[0]
            for k in range(1, N):
                p = b.select(b.cmp_eq(r, _c(b, k)), self.peers[k], p)
            out.append((r, p))
        return out

    def _fold_consume(self, u, p_rs) -> None:
        """cons=fold: the last arriver of block (rg, tbk) folds 8 rows x 1 token per thread (the same per-element split
        order and bf16 rounding as the plain GEMM), pushes them to the other ranks (lamport) or stages them in its own
        arena and publishes the block flag (pull), then polls / reads the other ranks' copies of the SAME pieces and adds
        all ranks' bf16 partials in f32 in rank order (its own from registers) -> dst. Deadlock-free because every CTA
        runs at most one unit per layer (``check``): a polling CTA holds no later unit, and every rank's pushes of the
        block come from a fold that waits on nothing remote."""
        b, k, ix = self.b, self.k, self.ix
        N = ix.ranks
        TP = k.tiles * G.TILE
        R = k.rows_per_unit
        q8 = R // 8
        pieces = k.tbt * G.TILE * q8
        S = k.splits - (1 if k.mutant == "fold" else 0)
        layer = u.layer
        slot = self._slot(layer)
        tag = self._tag(layer)
        blk = b.add(b.mul(u.rg, _c(b, k.ntb)), u.tbk)
        pull = ix.form == "pull"
        oth = self._others()
        zero = _c(b, 0)
        its = []
        for it in range(-(-pieces // k.threads)):
            v = b.add(self.tid, _c(b, it * k.threads))
            tl = G._udiv(b, v, q8)
            r_in = b.mul(G._umod(b, v, q8), _c(b, 8))
            row = b.add(b.mul(u.rg, _c(b, R)), r_in)
            tok = b.add(u.tok0, tl)
            ok = b.land(b.cmp_lt(v, _c(b, pieces)), b.cmp_lt(tok, self.rows))
            tot = None
            grp_ = max(1, G.FOLD_GROUP // 2)
            for s0 in range(0, S, grp_):
                loads = []
                for s_ in range(s0, min(S, s0 + grp_)):
                    off = b.mul(b.add(b.mul(b.add(_c(b, s_ * TP), tok), self.Mpad), row), _c(b, 4))
                    for h in range(2):
                        o = b.select(ok, b.add(off, _c(b, 16 * h)), _c(b, G.OOB))
                        loads.append(b.buffer_load_cp(p_rs, o, zero, dwords=4, aux=G.CPOL_SC1))
                for j in range(0, len(loads), 2):
                    f = [b.vec_extract(b.vec_bitcast(loads[j + h], VectorType(F32, 4)), e) for h in range(2) for e in range(4)]
                    tot = f if tot is None else [b.fadd(x, y) for x, y in zip(tot, f)]
            own = b.vec_pack(_dwords_of_f32(b, tot, "bf16"), I32)  # 8 bf16 = 4 dwords
            inner = b.mul(b.add(b.mul(tl, _c(b, R)), r_in), _c(b, 2))
            junk = b.add(_c(b, JUNK_OFF), b.shl(b.land(v, _c(b, JUNK_BYTES // 16 - 1)), _c(b, 4)))
            if pull:
                off = b.select(ok, b.add(self._pay(slot, self.rank, blk, True), inner), junk)
                W.push_vecs(b, [self.own], off, [own])
            elif N > 1:
                lo = self._encode2(b.vec_pack([b.vec_extract(own, 0), b.vec_extract(own, 1)], I32))
                hi = self._encode2(b.vec_pack([b.vec_extract(own, 2), b.vec_extract(own, 3)], I32))
                enc = b.vec_pack([b.vec_extract(lo, 0), b.vec_extract(lo, 1), b.vec_extract(hi, 0), b.vec_extract(hi, 1)], I32)
                off = b.select(ok, b.add(self._pay(slot, self.rank, blk, False), inner), junk)
                W.push_vecs(b, [p for _r, p in oth], off, [enc])
            its.append(own)  # only the partial stays live across the polls: the piece geometry is recomputed below
            #   (keeping the per-iteration lane masks live failed COMGR register allocation at 2+ fold iterations)
        if pull:  # every thread's stores of the block performed, then the block flag on every rank
            W.drain(b)
            b.sync_lds_only()
            fo = self._flag(slot, self.rank, blk)
            with b.scf_if(b.cmp_lt(self.tid, _c(b, N))):
                W.store_flags_lanes(b, b.add(self.lane_peer, b.zext(fo, I64)), tag)
            if N > 1:  # the other ranks' flags of this block (lane l polls rank (rank + 1 + l mod (N - 1)) mod N)
                lr = b.mod(b.add(b.add(self.rank, _c(b, 1)), b.mod(self.lane, _c(b, N - 1))), _c(b, N))
                dly = b.select(b.cmp_eq(layer, zero), self.dl_first, self.dl)
                dl = b.add(D.memrealtime(b), b.zext(dly, I64))
                faddr = b.add(self.own, b.zext(self._flag(slot, lr, blk), I64))
                _s, _m, st = W.poll_flags(b, faddr, tag, dl, self.abort_addr, _c(b, 2), pause=ix.pause)
                self._record(st, layer)
        dly = b.select(b.cmp_eq(layer, zero), self.dl_first, self.dl)
        dl = b.add(D.memrealtime(b), b.zext(dly, I64))
        TPb = k.tiles * G.TILE
        dlay = b.add(self.dst, b.mul(b.zext(layer, I64), b.const_i64(TPb * ix.m * 2)))
        d_rs = G._rsrc(b, dlay, _c(b, TPb * ix.m * 2))
        rslot = b.sub(_c(b, 1), slot) if ix.mutant == "slot" else slot
        for it, own in enumerate(its):
            v = b.add(self.tid, _c(b, it * k.threads))
            tl = G._udiv(b, v, q8)
            r_in = b.mul(G._umod(b, v, q8), _c(b, 8))
            tok = b.add(u.tok0, tl)
            ok = b.land(b.cmp_lt(v, _c(b, pieces)), b.cmp_lt(tok, self.rows))
            inner = b.mul(b.add(b.mul(tl, _c(b, R)), r_in), _c(b, 2))
            ones = b.add(_c(b, ONES_OFF), b.shl(b.land(v, _c(b, 63)), _c(b, 4)))
            if N > 1 and pull:
                offs = [b.select(ok, b.add(self._pay(rslot, r, blk, True), inner), ones) for r, _p in oth]
                vecs = W.load_lines_peers(b, [p for _r, p in oth], offs, 1)
            elif N > 1:
                offs = [b.select(ok, b.add(self._pay(slot, r, blk, False), inner), ones) for r, _p in oth]
                raw, st = poll_lamport_dw(b, self.own, offs, dl, self.abort_addr, pause=ix.pause)
                self._record(st, layer)
                if ix.mutant != "noreset":
                    junk = b.add(_c(b, JUNK_OFF), b.shl(b.land(v, _c(b, JUNK_BYTES // 16 - 1)), _c(b, 4)))
                    W.store_zero_lines(b, self.own, [b.select(ok, o, junk) for o in offs], 1, b.vec_pack([zero] * 4, I32))
                vecs = [b.vec_pack([b.xor(b.vec_extract(x, i), _c(b, -1)) for i in range(4)], I32) for x in raw]
            else:
                vecs = []
            own_f = _f32_of_dwords(b, [b.vec_extract(own, i) for i in range(4)], "bf16")
            got = [_f32_of_dwords(b, [b.vec_extract(x, i) for i in range(4)], "bf16") for x in vecs]
            # the parts in rank order: rank r is this rank's own partial when r == rank, else the other rank's
            parts = []
            for r in range(N):
                vals = own_f
                for i, (rr, _p) in enumerate(oth):
                    hit = b.cmp_eq(rr, _c(b, r))
                    vals = [b.select(hit, x, y) for x, y in zip(got[i], vals)]
                parts.append(vals)
            out = b.vec_pack(_dwords_of_f32(b, _rank_sum(b, parts), "bf16"), I32)
            doff = b.mul(b.add(b.mul(tok, _c(b, ix.m)), b.add(b.mul(u.rg, _c(b, R)), r_in)), _c(b, 2))
            b.buffer_store_cp(out, d_rs, b.select(ok, doff, _c(b, G.OOB)), zero)

    # ------------------------------------------------------------------ consumer
    def consume(self, layer) -> None:
        """This wave's consumer units of exchange use ``layer`` (uniform)."""
        b, k, ix = self.b, self.k, self.ix
        if ix.ar == "twoshot":
            self._consume2(layer)
            return
        N, R, TPU = ix.ranks, ix.R, ix.tpu
        TB16 = k.tbt * G.TILE
        RG = ix.m // R
        zero = _c(b, 0)
        UG = b.readfirstlane(G._cdiv(b, self.rows, _c(b, TPU)))
        U = b.mul(UG, _c(b, RG))
        gw = b.readfirstlane(b.add(b.mul(self.cta, _c(b, k.waves)), self.wave))
        tw = _c(b, k.grid * k.waves)
        left = b.sub(U, gw)
        nk = b.readfirstlane(b.select(b.cmp_gt(left, zero), G._cdiv(b, left, tw), zero))
        tag = self._tag(layer)
        slot = self._slot(layer)
        dly = b.select(b.cmp_eq(layer, zero), self.dl_first, self.dl)
        dl = b.add(D.memrealtime(b), b.zext(dly, I64))
        pull = ix.form == "pull"
        rslot = b.sub(_c(b, 1), slot) if ix.mutant == "slot" else slot
        e = b.mul(self.lane, _c(b, 8))
        t_in = G._udiv(b, e, R)
        r_in = G._umod(b, e, R)
        lane16 = b.shl(self.lane, _c(b, 4))
        TPb = k.tiles * G.TILE
        dlay = b.add(self.dst, b.mul(b.zext(layer, I64), b.const_i64(TPb * ix.m * 2)))
        d_rs = G._rsrc(b, dlay, _c(b, TPb * ix.m * 2))
        with b.scf_for(zero, nk, _c(b, 1), iv_name=self._iv("xk")) as kk:
            u = b.readfirstlane(b.add(gw, b.mul(kk, tw)))
            rg = b.div(u, UG)
            grp = b.sub(u, b.mul(rg, UG))
            tokstart = b.mul(grp, _c(b, TPU))
            tbk = G._udiv(b, tokstart, TB16)
            tib0 = b.sub(tokstart, b.mul(tbk, _c(b, TB16)))
            blk = b.add(b.mul(rg, _c(b, k.ntb)), tbk)
            tok = b.add(tokstart, t_in)
            live = b.cmp_lt(tok, self.rows)
            inner = b.mul(b.add(b.mul(b.add(tib0, t_in), _c(b, R)), r_in), _c(b, 2))
            ones = b.add(_c(b, ONES_OFF), lane16)
            if pull:
                faddr = b.add(self.own, b.zext(self._flag(slot, self.lane_src, blk), I64))
                _seen, _mask, st = W.poll_flags(b, faddr, tag, dl, self.abort_addr, _c(b, 2), pause=ix.pause)
                self._record(st, layer)
                offs = [b.select(live, b.add(self._pay(rslot, r, blk, True), inner), ones) for r in range(N)]
                vecs = W.load_lines_peers(b, self.peers, offs, 1)
            elif ix.form == "line":  # poll this unit's line flag of every rank, then read the lines from the own arena
                line = b.readfirstlane(G._udiv(b, tib0, TPU))
                faddr = b.add(self.own, b.zext(self._lflag(slot, self.lane_src, blk, line), I64))
                _seen, _mask, st = W.poll_flags(b, faddr, tag, dl, self.abort_addr, _c(b, 2), pause=ix.pause)
                self._record(st, layer)
                offs = [b.select(live, b.add(self._payl(rslot, r, blk), inner), ones) for r in range(N)]
                vecs = W.load_lines(b, self.own, offs)
            else:
                offs = [b.select(live, b.add(self._pay(slot, r, blk, False), inner), ones) for r in range(N)]
                raw, _mask, st = W.poll_lamport(b, self.own, offs, 1, "bf16", dl, self.abort_addr, pause=ix.pause)
                self._record(st, layer)
                if ix.mutant != "noreset":
                    junk = b.add(_c(b, JUNK_OFF), lane16)
                    zoffs = [b.select(live, o, junk) for o in offs]
                    W.store_zero_lines(b, self.own, zoffs, 1, b.vec_pack([zero] * 4, I32))
                ones_v = _c(b, -1)
                vecs = [b.vec_pack([b.xor(b.vec_extract(x, i), ones_v) for i in range(4)], I32) for x in raw]
            parts = [_f32_of_dwords(b, [b.vec_extract(v, i) for i in range(4)], "bf16") for v in vecs]
            s = _rank_sum(b, parts)
            out = b.vec_pack(_dwords_of_f32(b, s, "bf16"), I32)
            doff = b.mul(b.add(b.mul(tok, _c(b, ix.m)), b.add(b.mul(rg, _c(b, R)), r_in)), _c(b, 2))
            b.buffer_store_cp(out, d_rs, b.select(live, doff, _c(b, G.OOB)), zero)

    def _consume2(self, layer) -> None:
        """ar=twoshot (lamport): stage 1, this wave's units of the blocks this rank owns (block mod N): poll the N
        partials in the own arena, reset, sum in rank order (f32), round to bf16 and push the sum to every rank's
        all-gather slot; stage 2, every unit: poll the owner's sum in the own arena, reset, store it to ``dst``. A wave
        runs all its stage-1 units before any stage-2 unit, and stage 1 waits only on folds (no cycle)."""
        b, k, ix = self.b, self.k, self.ix
        N, R, TPU = ix.ranks, ix.R, ix.tpu
        TB16 = k.tbt * G.TILE
        RG = ix.m // R
        zero = _c(b, 0)
        UG = b.readfirstlane(G._cdiv(b, self.rows, _c(b, TPU)))
        U = b.mul(UG, _c(b, RG))
        gw = b.readfirstlane(b.add(b.mul(self.cta, _c(b, k.waves)), self.wave))
        tw = _c(b, k.grid * k.waves)
        left = b.sub(U, gw)
        nk = b.readfirstlane(b.select(b.cmp_gt(left, zero), G._cdiv(b, left, tw), zero))
        slot = self._slot(layer)
        dly = b.select(b.cmp_eq(layer, zero), self.dl_first, self.dl)
        dl = b.add(D.memrealtime(b), b.zext(dly, I64))
        e = b.mul(self.lane, _c(b, 8))
        t_in = G._udiv(b, e, R)
        r_in = G._umod(b, e, R)
        lane16 = b.shl(self.lane, _c(b, 4))
        ones = b.add(_c(b, ONES_OFF), lane16)
        junk = b.add(_c(b, JUNK_OFF), lane16)
        zv = b.vec_pack([zero] * 4, I32)
        TPb = k.tiles * G.TILE
        dlay = b.add(self.dst, b.mul(b.zext(layer, I64), b.const_i64(TPb * ix.m * 2)))
        d_rs = G._rsrc(b, dlay, _c(b, TPb * ix.m * 2))

        def geo(kk):
            u = b.readfirstlane(b.add(gw, b.mul(kk, tw)))
            rg = b.div(u, UG)
            tokstart = b.mul(b.sub(u, b.mul(rg, UG)), _c(b, TPU))
            tbk = G._udiv(b, tokstart, TB16)
            tib0 = b.sub(tokstart, b.mul(tbk, _c(b, TB16)))
            blk = b.readfirstlane(b.add(b.mul(rg, _c(b, k.ntb)), tbk))
            tok = b.add(tokstart, t_in)
            inner = b.mul(b.add(b.mul(b.add(tib0, t_in), _c(b, R)), r_in), _c(b, 2))
            return rg, blk, tok, b.cmp_lt(tok, self.rows), inner

        def decode(raw):
            return [b.vec_pack([b.xor(b.vec_extract(x, i), _c(b, -1)) for i in range(4)], I32) for x in raw]

        with b.scf_for(zero, nk, _c(b, 1), iv_name=self._iv("xr")) as kk:
            _rg, blk, _tok, live, inner = geo(kk)
            with b.scf_if(b.cmp_eq(b.mod(blk, _c(b, N)), self.rank)):
                offs = [b.select(live, b.add(self._pay(slot, r, blk, False), inner), ones) for r in range(N)]
                raw, _mask, st = W.poll_lamport(b, self.own, offs, 1, "bf16", dl, self.abort_addr, pause=ix.pause)
                self._record(st, layer)
                if ix.mutant != "noreset":
                    W.store_zero_lines(b, self.own, [b.select(live, o, junk) for o in offs], 1, zv)
                parts = [_f32_of_dwords(b, [b.vec_extract(v, i) for i in range(4)], "bf16") for v in decode(raw)]
                out = b.vec_pack(_dwords_of_f32(b, _rank_sum(b, parts), "bf16"), I32)
                W.push_vecs(b, self.peers, b.select(live, b.add(self._payag(slot, blk), inner), junk), [self._encode4(out)])
        with b.scf_for(zero, nk, _c(b, 1), iv_name=self._iv("xg")) as kk:
            rg, blk, tok, live, inner = geo(kk)
            off = b.select(live, b.add(self._payag(slot, blk), inner), ones)
            raw, _mask, st = W.poll_lamport(b, self.own, [off], 1, "bf16", dl, self.abort_addr, pause=ix.pause)
            self._record(st, layer)
            if ix.mutant != "noreset":
                W.store_zero_lines(b, self.own, [b.select(live, off, junk)], 1, zv)
            out = decode(raw)[0]
            doff = b.mul(b.add(b.mul(tok, _c(b, ix.m)), b.add(b.mul(rg, _c(b, R)), r_in)), _c(b, 2))
            b.buffer_store_cp(out, d_rs, b.select(live, doff, _c(b, G.OOB)), zero)

    def finish(self) -> None:
        """Last CTA: epoch (+)= layers, slotbase ^= layers & 1, exit counter back to zero."""
        b = self.b
        nxt = W.tag_add(b, self.epoch, self.L)
        sb = b.readfirstlane(b.land(b.add(self.slotbase, self.L), _c(b, 1)))
        last = _c(b, self.k.grid - 1)
        with b.scf_if(D.is_thread0(b)):
            old = D.atomic_add_scoped(b, self.own, _c(b, OFF_EXIT), _c(b, 1), scope="agent", returns=True)
            with b.scf_if(b.cmp_eq(old, last)):
                W.store_word_sys(b, self.own, _c(b, OFF_EXIT), _c(b, 0))
                W.store_word_sys(b, self.own, _c(b, OFF_EPOCH), nxt)
                W.store_word_sys(b, self.own, _c(b, OFF_SLOTBASE), sb)
                if self.ix.gb == "lite":
                    W.store_word_sys(b, self.own, _c(b, OFF_LAUNCHES), b.add(self.launches, _c(b, 1)))
                W.drain(b)

    # ------------------------------------------------------------------ main loop (the GEMM's, plus consume + finish)
    def build(self) -> KernelDef:
        b, k = self.b, self.k
        D_, XA, KC = k.depth, k.xa, k.kc
        lds = k.xsrc == "lds"
        units = {}

        def unit(n, d):
            if d not in units:
                units[d] = G._Unit(self, b.add(n, _c(b, d)) if d else n)
            return units[d]

        def wpos(n, p, xg=None):
            d, c = divmod(p, KC)
            return self.wissue(unit(n, d), c, xg)

        def xpos(n, p, gate=None):
            d, c = divmod(p, KC)
            return self.xissue(unit(n, d), c, gate)

        def sel(cond, new, old):
            return [b.select(cond, x, y) for x, y in zip(new, old)]

        def parity(n, c):
            if KC % 2 == 0 or not k.xu:
                return _c(b, c % 2)
            return b.land(b.add(n, _c(b, c)), _c(b, 1))

        zero = _c(b, 0)
        n0 = zero
        units.clear()
        self.cur_layer = unit(n0, 0).layer
        wq = [wpos(n0, p) for p in range(D_)] if k.xu else []
        xq = []
        if lds and k.xu:
            first = xpos(n0, 0)
            self.xput(first, zero)
            xq = [xpos(n0, p) for p in range(1, XA + 1)]
            b.sync_lds_only()
        lens_w = [len(g) for g in wq]
        lens_x = [len(g) for g in xq]
        carry = [v for g in wq for v in g] + [v for g in xq for v in g]
        iters = [(f"c{i}", v) for i, v in enumerate(carry)]
        loop = b.scf_for_iter(zero, self.ncta, _c(b, 1), iters, iv_name="un")
        with loop as lv:
            n, vals = lv if iters else (lv, [])
            units.clear()
            vals = list(vals)
            u = unit(n, 0)
            self.cur_layer = u.layer
            cross = b.land(self.gsync, b.land(b.cmp_eq(u.i, zero), b.cmp_gt(u.layer, zero)))
            with b.scf_if(cross):
                prev = b.sub(u.layer, _c(b, 1))
                if self.ix.cons == "phase":  # FlyTape's shape: every CTA of this rank has pushed before any consumer
                    G._Gemm.gbar(self, b.add(prev, self.L))
                if self.ix.cons in ("pass", "phase") and self.ix.form != "none":
                    self.consume(prev)
                self.gbar(prev)
            pos = 0
            if k.xu:
                wq = []
                for ln in lens_w:
                    wq.append(vals[pos : pos + ln])
                    pos += ln
                xq = []
                for ln in lens_x:
                    xq.append(vals[pos : pos + ln])
                    pos += ln
                if lds:
                    f0 = xpos(n, 0, cross)
                    with b.scf_if(cross):
                        self.xput(f0, parity(n, 0))
                        b.sync_lds_only()
                    xq = [sel(cross, xpos(n, p, b.land(cross, self.xgate(unit(n, p // KC)))), xq[p - 1]) for p in range(1, XA + 1)]
                if k.sop:
                    for p in range(D_):
                        d_, c_ = divmod(p, KC)
                        if c_ % 4:
                            continue
                        fresh = self.wissue(unit(n, d_), c_, b.land(cross, self.xgate(unit(n, d_))), xonly=True)
                        nx = len(fresh)
                        wq[p] = wq[p][: len(wq[p]) - nx] + sel(cross, fresh, wq[p][len(wq[p]) - nx :])
                elif k.xsrc == "global" or k.xscale == "glb":
                    nw = k.mt * (2 if k.w == "mx" else 4)
                    for p in range(D_):
                        d_, c_ = divmod(p, KC)
                        fresh = self.wissue(unit(n, d_), c_, b.land(cross, self.xgate(unit(n, d_))), xonly=True)
                        wq[p] = wq[p][:nw] + sel(cross, fresh, wq[p][nw:])
            else:
                wq = [wpos(n, p) for p in range(min(D_, KC))]
                if lds:
                    first = xpos(n, 0)
                    b.sync_lds_only()
                    self.xput(first, zero)
                    xq = [xpos(n, p) for p in range(1, min(XA, KC - 1) + 1)]
                    b.sync_lds_only()
            acc = [[b.zero_vec_f32(4) for _ in range(k.tbt)] for _ in range(k.mt)]
            for c in range(KC):
                if k.sb:
                    b.sched_barrier(0)
                if k.xu or c + D_ < KC:
                    wq.append(wpos(n, c + D_))
                if lds and (k.xu or c + 1 + XA < KC):
                    xq.append(xpos(n, c + 1 + XA))
                if k.sb:
                    b.sched_barrier(0)
                cur = wq.pop(0)
                skip = k.mutant == "kskip" and c == KC - 1
                acc = self.chunk_math(acc, cur, parity(n, c) if lds else None, skip, c % 4)
                if lds and (k.xu or c + 1 < KC):
                    nxt = xq.pop(0)
                    self.xput(nxt, parity(n, c + 1))
                    b.sync_lds_only()
            if k.sb:
                b.sched_barrier(0)
            self.epilogue(u, acc)
            carry = [v for g in wq for v in g] + [v for g in xq for v in g]
            if not k.xu:
                carry = []
            b.scf_yield(*carry)
        # a CTA without units in a layer consumes and synchronises every layer too
        with b.scf_if(b.cmp_eq(self.cnt, zero)):
            with b.scf_for(zero, b.sub(self.L, _c(b, 1)), _c(b, 1), iv_name="xgb") as bnd:
                if self.ix.cons == "phase":
                    G._Gemm.gbar(self, b.add(bnd, self.L))
                if self.ix.cons in ("pass", "phase") and self.ix.form != "none":
                    self.consume(bnd)
                self.gbar(bnd)
        if self.ix.cons == "phase":
            G._Gemm.gbar(self, b.add(b.sub(self.L, _c(b, 1)), self.L))
        if self.ix.cons in ("pass", "phase") and self.ix.form != "none":
            self.consume(b.sub(self.L, _c(b, 1)))
        self.finish()
        return b.kernel


_VMEM = ("global_", "buffer_", "flat_", "scratch_")


def pad_asm_vmem(kd: KernelDef) -> int:
    """Prefix ``s_nop 4`` to every inline-asm op whose template issues a VMEM instruction and that takes an SGPR input
    (``s`` constraint): the gfx9 VALU-writes-SGPR -> VMEM-reads-it hazard (5 wait states) that LLVM does not pad inside
    inline asm. Returns the number of padded ops."""
    n = 0

    def walk(region) -> None:
        nonlocal n
        for op in region.ops:
            if op.name == "tile.inline_asm":
                t = op.attrs.get("template", "")
                ins = [c.strip() for c in op.attrs.get("constraints", "").split(",")]
                if any(v in t for v in _VMEM) and "s" in ins and not t.startswith("s_nop 4\n"):
                    op.attrs["template"] = "s_nop 4\n" + t
                    n += 1
            for r in op.regions:
                walk(r)

    walk(kd.body)
    return n


def build_insitu_kernel(kn: InsituKnobs) -> KernelDef:
    kn = resolve(kn)
    kd = _Insitu(kn).build()
    if kn.hz:
        pad_asm_vmem(kd)
    return kd


def mutant(kn: InsituKnobs) -> InsituKnobs:
    """The stale-read must-fail twin of a build (lamport ``noreset``, pull ``slot``)."""
    if kn.form == "none":
        raise ValueError("form=none has no exchange twin")
    kn = resolve(kn)
    return replace(kn, mutant="noreset" if kn.form == "lamport" else "slot")
