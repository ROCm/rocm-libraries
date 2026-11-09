# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The sparse MLA merge (``attn.m``, ``smerge``) of monofly's DeepSeek-V4.1 compressed consumer layers as ONE rocKE
stage function (hybrid path: monofly links it into its FlyDSL layer kernel in place of the ``<name>.m`` stage body,
``MLASparseAttn(rk=...)``, off by default).

Bit-identical re-implementation of monofly ``blocks/gfx950/attention/mla_sparse.py`` ``_smerge`` / ``_msoftmax`` /
``_mpv`` / ``_smerge_tail`` for the configuration its consumer layers run (``smerge``, ``vtr``, ``lean``; no
``ckv4`` / ``ring8`` / step entries; the selection a plain arg). A task is (row, 16-head tile, head group of
``mheads`` heads, 64-dim slice):

* V of the slice for all 640 keys into LDS, key-major rows, XOR-swizzled (``_vk_word``): the window's 128 from the
  ring (positions before the step's first), the step's own keys from ``kvn`` (tagged pairs: an attempt at task
  start, polled again only if stale), the 512 compressed keys from the cache by the selection (-1 pads: zeros);
* the raw scores of the task's heads in all ten chunks polled (one round trip), the softmax with the producers' thread
  mapping (32 threads a head, two keys a thread a chunk): scale, mask, each chunk's max by a 5-level butterfly,
  the running max over the chunks in order (from -1e30), P = exp(s - m) (bf16 into LDS), each chunk's sum(P);
* P V of each chunk (``v_mfma_f32_16x16x32_bf16``, B by LDS transpose reads), the reference's recurrence in chunk
  order (mul then add, not fused), the sink, the division, bf16, the inverse RoPE of the last 64 dims (fma pairs as
  PyTorch's GPU complex multiply), the MXFP8 QAT (group 32) -> the output channel's word / code pairs.

Every floating-point op is monofly's, in its order (``mfabi`` module docstring for the semantics).

ABI (``rocke_v41mf_smerge_m<mheads>_<variant>``, ``lds="arg"``: LDS operands are word offsets from ``lds_base``, the
merge's own layout (monofly ``_SPlan.m_*``))::

    void (i32 tag, i32 task, i32 tiles, i32 spec, i32 ring_slots, i32 cache_len, i32 ckv_big, i32 unrope_f32,
          i64 pos, i64 rope, i64 sink, i64 ring, i64 ckv, i64 sel,
          i64 s, i32 s_loc, i32 s_fl, i64 kvn, i32 kvn_loc, i32 kvn_fl, i64 oq, i32 oq_loc, i32 oq_fl,
          i64 os, i32 os_loc, i32 os_fl, i64 md, i32 md_loc, i32 md_fl, i64 scr, i64 dbg, i32 ntasks,
          i32 lds_base)

Regions (monofly names): ``s`` = ``<name>.s`` (scores, polled), ``kvn`` (the step's keys, polled), ``oq`` / ``os`` =
the output channel's ``.q`` / ``.s`` (put), ``md`` = ``<name>.mdone`` (put, thread 0). Every int argument except
``tag`` / ``task`` / the ``*_loc`` is a build constant on the monofly side (they fold after inlining). ``scr``: the
scratch buffer's base (every region lies in it: ``srsrc``). ``ntasks``: the stage's tasks (``endbar``). ``dbg``: monofly's
``<name>.m.probe`` (int64 [CTA][8]) in a probe build (``stamps``), else 0.

Knobs (:class:`SmergeKnobs`): ``mheads``; ``bar`` (``fence``: FlyDSL's barriers, ``lds``); ``pause`` (poll retry:
0 = ``s_nop 0``); ``mutate`` (test only, :data:`MUTANTS`); ``stamps`` (1: monofly's merge probe points, thread 0,
s_memrealtime into ``dbg``: 0 start, 1 V rows landed, 2 scores landed, 7 maxima exchanged, 3 softmax done, 6 P V
MFMAs done (wave 0), 4 P V done, 5 end). Structural levers, every one the same bits (the default 0 is monofly's
order):

* ``premask``: the score masks (the selection's LDS flags, the window bound) and the tail's sink / RoPE loads before
  the score poll instead of after it (off the path after the scores land);
* ``msreuse``: the tail's running maxima from the softmax's registers (a thread of the tail's head computed the same
  ``maximum`` chain from the same LDS words) instead of 10 LDS loads and the chain again;
* ``early``: the corrections exp(m_{c-1} - m_c) and the denominator's chain (they need only the maxima and the
  chunks' sums) before the P V MFMAs instead of after the t exchange;
* ``tsep``: t (the P V partials) in LDS words of their own past the merge's layout instead of over V: no barrier
  between the MFMAs and the t stores (the stage's LDS grows by ``NCT * mheads * OP`` words);
* ``srsrc``: one buffer resource over scratch for the five mailbox regions (their offsets the scalar offsets) instead
  of one a region: fewer SGPRs in the layer (it spills SGPRs into VGPR lanes);
* ``seldirect``: 1: each lane loads its 8 compressed keys' selection entries straight from the plain arg (one dword a
  chunk, beside the copy into LDS the masks read), so the compressed V loads do not wait for the copy, an LDS round
  trip and a barrier (runbook 8.3: barriers only for cross-thread visibility); 2: the lanes' compressed keys remapped to
  8 consecutive entries of the selection (lane slot k: keys 8 k .. + 8, one chunk; a wave's load still covers 8 keys
  x 128 contiguous bytes), so a lane's 8 entries are two 16-byte loads, and its piece-0 lane writes them into the LDS
  copy itself (no copy loads). Either way, with ``ownskip`` too (no barrier left between the copy and the masks' reads
  of other threads' entries) a barrier goes before the masks;
* ``ownskip``: the window's V store skips the step's own slots (the ring holds zeros there; the own rows land in
  them) instead of zeroing them first: no write-after-write between threads, so no barrier between the V stores and
  the own rows; the ring-read flag (``mdone``) moves after the maxima exchange's barrier (every lane's ring loads
  have landed by then);
* ``endbar``: the task's closing barrier only when this CTA runs another task of the stage (task + 256 < ``ntasks``):
  the emitter joins the CTA before the next LDS stage anyway;
* ``tail1``: the tail (the recurrence, the division, the inverse RoPE, the QAT, the puts) one dim a thread (64 threads a
  head: ``mheads`` x 64 threads, mheads <= 8) instead of two (32 a head): twice the waves on the tail's serial chain,
  half its work a thread (WaveScope at6: the tail ran on 2 of 8 waves at mheads 4, ~1 us of VALU). The RoPE pair's
  other value and the MX word's other three by lane exchanges; the group's amax over 32 lanes (max is exact): the same
  bits. Its maxima come from LDS (``msreuse`` holds another thread's head);
* ``sattempt``: the raw scores' loads issued once at task start (an attempt, as monofly's own rows' poll is), before
  the V loads; where the scores are polled, a wave whose attempt holds every tag current uses it, else the wave polls
  as before (its words staged in the P rows' LDS, free then; mheads <= 8). In situ the producers' scores are out before
  the merge task starts (section 2 of the lane's RESULTS: by ~2 us), so the poll's round trip (~0.5 us after V
  landed) leaves the path. ``sattempt=2``: the attempt lands in LDS by the buffer -> LDS DMA (``buffer_load ... lds``,
  16 bytes a lane, past the merge's layout: ``SATT_WORDS`` words a spec and wave), so no register holds it across the
  V landing (in situ ``sattempt=1`` cost 17 SGPR spills at 6 rows); the DMA has no device-scope (sc1) bit, so it may
  read a stale line: the tag check then re-polls (sc1), the same words either way;
* ``vdma``: V's window and compressed rows land in LDS by the buffer -> LDS DMA (``buffer_load ... lds``, 16 bytes a
  lane) instead of through registers and ``ds_write``: a wave's load of one chunk covers 8 keys x 128 contiguous bytes,
  which ``_vk_word`` places in 1 KB of LDS, so lane L loads key L / 8's piece (L % 8) ^ swizzle (the swizzle moves to
  the source addresses). Masked keys (positions out of the window's ring range, -1 selection pads) load from past the
  buffer's extent (zeros: the same words as the register path's selects). With ``ownskip`` the step's own slots are
  lanes off in EXEC (no write), so no wave waits for its V before the score poll: the poll's round trip overlaps the
  V landing (the bubble ``sattempt`` aimed at, with no registers held); every wave waits for its DMA (vmcnt) before
  the maxima exchange's barrier, the first point another thread's V is read. Without ``ownskip`` the own slots load
  zeros and the V barrier waits for the DMA first. No registers carry V (``seldirect=2``'s key order is not
  contiguous per wave: vdma is off with it).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, fields

from rocke.core.ir import BF16, F32, I32, I64, IRBuilder, VectorType

from .mfabi import CM_DEV, Emit, Region

HD, ROPE, WINDOW, CH, TILE, SEL = 512, 64, 128, 64, 16, 512
NCH = SEL // CH  # compressed chunks
NCT = 2 + NCH  # all chunks
SLICE = 64
NSL = HD // SLICE
VTP = NCT * CH // 2 + 4
PP = NCT * CH // 2 + 4
OP = SLICE + 4
THREADS, WAVES = 512, 8
NEG_INIT = -1.0e30
M_VT = 0
M_P = SLICE * VTP
M_I = M_P + TILE * PP
M_X = M_I + SEL  # (no step entries: m_n == m_i + SEL)
SYMBOL_PREFIX = "rocke_v41mf_"
STAGE = "smerge"
ABI = (
    ("tag", "i32"), ("task", "i32"), ("tiles", "i32"), ("spec", "i32"), ("ring_slots", "i32"), ("cache_len", "i32"),
    ("ckv_big", "i32"), ("unrope_f32", "i32"),
    ("pos", "i64"), ("rope", "i64"), ("sink", "i64"), ("ring", "i64"), ("ckv", "i64"), ("sel", "i64"),
    ("s", "i64"), ("s_loc", "i32"), ("s_fl", "i32"), ("kvn", "i64"), ("kvn_loc", "i32"), ("kvn_fl", "i32"),
    ("oq", "i64"), ("oq_loc", "i32"), ("oq_fl", "i32"), ("os", "i64"), ("os_loc", "i32"), ("os_fl", "i32"),
    ("md", "i64"), ("md_loc", "i32"), ("md_fl", "i32"), ("scr", "i64"), ("dbg", "i64"), ("ntasks", "i32"),
)  # fmt: skip
# Test-only mutants (each must make the bitwise A/B differ or fail the gate):
#   out_bit: every output word's low bit flipped        code:  every output code + 1
#   rec:     the last compressed chunk left out of the recurrence
#   sink:    no sink in the denominator                   own:   the step's own keys not landed (ring zeros)
#   pv_kst:  each chunk's P V without its second 32-key step
#   vsw:     (vdma) the source pieces not swizzled          vwait: (vdma) no DMA wait before the maxima barrier
MUTANTS = ("out_bit", "code", "rec", "sink", "own", "pv_kst", "vsw", "vwait")


def m_l(mheads: int) -> int:
    return M_X + mheads * NCT


SATT_WORDS = 64 * 4  # sattempt=2: a wave's 16 bytes a lane of one spec


def nspec(mheads: int) -> int:
    return -(-NCT // (THREADS // (mheads * 32)))


def lds_words(mheads: int, tsep: int = 0, satt: int = 0) -> int:
    """The merge's LDS words (monofly ``_SPlan.m_lds``; ``tsep``: + the t region past it; ``satt`` 2: + the score
    attempt's DMA landing, ``WAVES x nspec x SATT_WORDS``)."""
    base = m_l(mheads) + mheads * NCT + (NCT * mheads * OP if tsep else 0)
    return base + (WAVES * nspec(mheads) * SATT_WORDS if satt2_fits(mheads, tsep, satt) else 0)


LDS_CAP = 37 * 1024  # words a merge layout may take (monofly's arena adds its context, guards and stamps past it)


def satt2_fits(mheads: int, tsep: int, satt: int) -> bool:
    """sattempt=2's DMA landing fits (mheads <= 8 and the layout under ``LDS_CAP``): else the attempt goes to
    registers (sattempt=1's form)."""
    base = m_l(mheads) + mheads * NCT + (NCT * mheads * OP if tsep else 0)
    return satt == 2 and nspec(mheads) <= 5 and base + WAVES * nspec(mheads) * SATT_WORDS <= LDS_CAP


@dataclass(frozen=True)
class SmergeKnobs:
    mheads: int = 4
    bar: str = "fence"
    pause: int = 0
    poll_max: int = 12
    mutate: str = ""
    stamps: int = 0
    premask: int = 0
    msreuse: int = 0
    early: int = 0
    tsep: int = 0
    srsrc: int = 0
    seldirect: int = 0
    ownskip: int = 0
    endbar: int = 0
    tail1: int = 0
    sattempt: int = 0
    vdma: int = 0

    def __post_init__(self) -> None:
        if self.mheads not in (1, 2, 4, 8, 16):
            raise ValueError("mheads: 1, 2, 4, 8 or 16")
        if self.bar not in ("fence", "lds"):
            raise ValueError("bar: fence | lds")
        if not 1 <= self.poll_max <= 16:
            raise ValueError("poll_max: 1..16")
        if self.mutate and self.mutate not in MUTANTS:
            raise ValueError(f"mutate: one of {MUTANTS}")

    def spec(self) -> str:
        d = SmergeKnobs()
        return ",".join(f"{f.name}={getattr(self, f.name)}" for f in fields(self)
                        if f.name != "mheads" and getattr(self, f.name) != getattr(d, f.name))  # fmt: skip

    def variant(self) -> str:
        s = self.spec()
        return f"m{self.mheads}_" + ("base" if not s else s.replace(",", "_").replace("=", "").replace("-", "m"))


def knobs_from_spec(spec: str) -> SmergeKnobs:
    kw, types = {}, {f.name: f.type for f in fields(SmergeKnobs)}
    for part in (spec or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition("=")
        if k not in types:
            raise ValueError(f"unknown knob {k!r}")
        kw[k] = int(v) if types[k] in ("int", int) else v
    return SmergeKnobs(**kw)


def symbol(knobs: SmergeKnobs) -> str:
    return f"{SYMBOL_PREFIX}{STAGE}_{knobs.variant()}"


def build_smerge(knobs: SmergeKnobs = SmergeKnobs()):
    k = knobs
    M = k.mheads
    G = THREADS // (M * 32)  # chunk groups (threads of a head: 32)
    NSPEC = -(-NCT // G)
    MG = TILE // M  # head groups a tile
    ML = m_l(M)
    b = IRBuilder(symbol(k))
    b.kernel.attrs["max_workgroup_size"] = THREADS
    b.fence_barriers = "compiler"
    p = {n: b.param(n, I32 if t == "i32" else I64) for n, t in ABI}
    e = Emit(b, max(lds_words(16, 1), lds_words(M, k.tsep, k.sattempt)), bar=k.bar, pause=k.pause, poll_max=k.poll_max)
    M_T = lds_words(M) if k.tsep else M_VT  # t: past the layout (tsep) or over V
    c, f = e.c, e.f
    rfl = b.readfirstlane
    # every int argument is wave-uniform in the caller (tag, task: SGPRs; the rest build constants)
    tag, task = p["tag"], p["task"]
    tiles, spec, rs, cl = (p[n] for n in ("tiles", "spec", "ring_slots", "cache_len"))
    big, unr = p["ckv_big"], p["unrope_f32"]
    shared = (e.rsrc(p["scr"]), p["scr"]) if k.srsrc else None
    reg = {n: Region(e, p[n], p[f"{n}_loc"], p[f"{n}_fl"], shared) for n in ("s", "kvn", "oq", "os", "md")}
    tid = b.thread_id_x()
    lane = b.land(tid, c(63))
    wave = rfl(b.lshr(tid, c(6)))
    heads = b.mul(tiles, c(TILE))
    r_dbg = e.rsrc(p["dbg"]) if k.stamps else None

    def stamp(point):
        """(stamps) thread 0: s_memrealtime -> dbg[CTA 8 + point] (monofly ``_pstamp``)."""
        if r_dbg is None:
            return
        with b.scf_if(b.cmp_eq(tid, c(0))):
            at = b.mul(b.add(b.mul(b.block_id_x(), c(8)), c(point)), c(8))
            b.buffer_store_cp(b.vec_bitcast(b.vec_pack([b.memrealtime()], I64), VectorType(I32, 2)), r_dbg, at, c(0),
                              aux=0)  # fmt: skip

    def vk_word(key, word):
        """monofly ``_vk_word``: the LDS word of a key's row word (rows of 32 words, 4-word pieces XOR-swizzled)."""
        sw = b.lor(b.land(key, c(2)), b.land(b.lshr(key, c(1)), c(4)))
        return b.add(b.add(c(M_VT), b.mul(key, c(SLICE // 2))),
                     b.add(b.shl(b.xor(sw, b.lshr(word, c(2))), c(2)), b.land(word, c(3))))  # fmt: skip

    # ---- the task: (row, tile) tt, head group hg, slice s
    per_tt = MG * NSL
    tt = b.div(task, c(per_tt))
    rem = b.sub(task, b.mul(tt, c(per_tt)))
    hg = b.div(rem, c(NSL))
    s = b.sub(rem, b.mul(hg, c(NSL)))
    r = b.div(tt, tiles)
    ht = b.sub(tt, b.mul(r, tiles))
    bq = b.div(r, spec)
    r_pos = e.rsrc(p["pos"])
    base = rfl(b.buffer_load_cp(r_pos, b.mul(bq, c(4)), c(0), dwords=1, aux=0))
    pos = b.add(base, b.sub(r, b.mul(bq, spec)))
    wlo = b.sub(pos, c(WINDOW - 1))  # the window's first position
    stamp(0)

    SA = bool(k.sattempt) and 2 * NSPEC * THREADS <= TILE * PP  # (the re-poll's words staged in P's rows)
    SA2 = SA and satt2_fits(M, k.tsep, k.sattempt)
    M_SA = lds_words(M, k.tsep)  # (sattempt=2: the attempt's DMA landing, past the layout)
    VD = bool(k.vdma) and k.seldirect != 2  # (vdma: V by the buffer -> LDS DMA)
    OOB = 0x7FFFFFF0  # (a voffset at the resources' extent: the load returns zeros)

    def score_specs():
        gch_ = b.div(tid, c(M * 32))
        hh_ = b.add(b.mul(hg, c(M)), b.land(b.lshr(tid, c(5)), c(M - 1)))
        kk_ = b.mul(b.land(tid, c(31)), c(2))
        ccs_ = [b.smin(b.add(gch_, c(G * i)), c(NCT - 1)) for i in range(NSPEC)]
        return [(reg["s"], b.add(b.mul(b.add(b.mul(b.add(b.mul(tt, c(NCT)), cc), c(TILE)), hh_), c(CH)), kk_), 2)
                for cc in ccs_]  # fmt: skip

    if SA:  # the scores' attempt, first (loads return in order: nothing after it waits longer for it)
        sspecs = score_specs()
        if SA2:  # into LDS by the DMA: no registers held across the V landing
            lds0 = b.smem_addr_of(e.pool)
            for i, (reg_, pr, _) in enumerate(sspecs):
                dst = b.smem_ptr_add(lds0, b.zext(b.mul(b.add(c(M_SA + i * SATT_WORDS), b.mul(wave, c(NSPEC * SATT_WORDS))), c(4)), I64))
                b.async_buffer_load_lds_addr(reg_.rsrc, dst, reg_.load_off(b.mul(pr, c(8))), c(0),
                                             4, 0)  # fmt: skip
            satt = None
        else:
            satt = [reg_.load(b.mul(pr, c(8)), 4) for reg_, pr, _ in sspecs]

    # ---- V loads: key kq of each 64 (lane pairs: consecutive keys), words 4 piece .. + 4 of the slice
    par = b.land(lane, c(1))
    piece = b.land(b.lshr(lane, c(1)), c(7))
    kq = b.add(b.add(b.mul(wave, c(8)), b.mul(b.lshr(lane, c(4)), c(2))), par)
    r_ring = e.rsrc(p["ring"])
    soff = b.add(b.mul(s, c(SLICE // 2)), b.mul(piece, c(4)))  # words in a row
    wv = []
    if VD:  # lane L: key L // 8 of the wave's 8, piece (L % 8) ^ the key's swizzle (bits 1 and 3 of the key: a chunk
        # starts at a multiple of 64, so the lane's key in the wave's 8 and the wave's parity give them)
        kl = b.lshr(lane, c(3))
        kq_d = b.add(b.mul(wave, c(8)), kl)
        sw_d = b.lor(b.land(kq_d, c(2)), b.land(b.lshr(kq_d, c(1)), c(4)))
        if k.mutate == "vsw":
            sw_d = c(0)
        soff_d = b.add(b.mul(s, c(SLICE // 2)), b.mul(b.xor(b.land(lane, c(7)), sw_d), c(4)))
        lds_v = b.smem_addr_of(e.pool)

        def vdma(rsrc_, key0, voff):
            """The wave's 8 keys key0 .. + 8 (1 KB of LDS: _vk_word's rows), 16 bytes a lane."""
            dst = b.smem_ptr_add(lds_v, b.zext(b.mul(b.add(c(M_VT), b.mul(key0, c(SLICE // 2))), c(4)), I64))
            b.async_buffer_load_lds_addr(rsrc_, dst, voff, c(0), 4, 0)

        for i in range(WINDOW // CH):
            kp = b.add(wlo, b.add(c(i * CH), kq_d))
            slot = b.mod(b.add(kp, rs), rs)
            row = b.add(b.mul(bq, rs), slot)
            ok = b.land(b.cmp_ge(kp, c(0)), b.cmp_lt(kp, base))
            voff = b.select(ok, b.mul(b.add(b.mul(row, c(HD // 2)), soff_d), c(4)), c(OOB))
            key0 = b.add(c(i * CH), b.mul(wave, c(8)))
            if k.ownskip:  # the step's own slots: lanes off (no write), the own rows land there
                with b.scf_if(b.cmp_lt(kp, base)):
                    vdma(r_ring, key0, voff)
            else:
                vdma(r_ring, key0, voff)
    for i in range(0 if VD else WINDOW // CH):
        kp = b.add(wlo, b.add(c(i * CH), kq))
        slot = b.mod(b.add(kp, rs), rs)  # kp >= -(WINDOW - 1) > -ring_slots: the unsigned modulo's value
        row = b.add(b.mul(bq, rs), slot)
        wv.append(b.buffer_load_cp(r_ring, b.mul(b.add(b.mul(row, c(HD // 2)), soff), c(4)), c(0), dwords=4, aux=0))
    # the step's own keys (kvn): thread -> row tid // 16 (clamped), words 2 (tid % 16) .. + 2: an attempt now
    jo = b.smin(b.lshr(tid, c(4)), b.sub(spec, c(1)))
    wo = b.mul(b.land(tid, c(15)), c(2))
    own_pair = b.add(b.add(b.mul(b.add(b.mul(bq, spec), jo), c(HD // 2)), b.mul(s, c(SLICE // 2))), wo)
    own_ok = b.land(b.cmp_lt(b.lshr(tid, c(4)), spec), b.cmp_le(b.add(base, jo), pos))
    own_raw = reg["kvn"].load(b.mul(own_pair, c(8)), 4)
    # the row's 512 selected positions into LDS (a thread each)
    r_sel = e.rsrc(p["sel"])
    if k.seldirect == 2:  # the lane's 8 consecutive entries (keys 8 kq .. + 8): two 16-byte loads
        sb = b.mul(b.add(b.mul(r, c(SEL)), b.mul(kq, c(8))), c(4))
        i2 = [b.buffer_load_cp(r_sel, b.add(sb, c(16 * h_)), c(0), dwords=4, aux=0) for h_ in range(2)]
        idxd = [b.vec_extract(i2[i_ // 4], i_ % 4) for i_ in range(NCH)]
        with b.scf_if(b.cmp_eq(piece, c(0))):  # the copy the masks read
            for h_ in range(2):
                e.st(b.add(c(M_I + 4 * h_), b.mul(kq, c(8))), i2[h_], 4)
    else:
        w = b.buffer_load_cp(r_sel, b.mul(b.add(b.mul(r, c(SEL)), tid), c(4)), c(0), dwords=1, aux=0)
        if k.seldirect:  # the lane's own 8 entries (chunk c, key kq), straight from the arg
            kqs = kq_d if VD else kq
            idxd = [b.buffer_load_cp(r_sel, b.mul(b.add(b.mul(r, c(SEL)), b.add(c(ci * CH), kqs)), c(4)), c(0), dwords=1,
                                     aux=0) for ci in range(NCH)]  # fmt: skip
        e.st(b.add(c(M_I), tid), w)
    if not k.seldirect:
        e.barrier()
    # the compressed keys by the selection (a request's slot of a big cache from its own base)
    slot_bytes = b.mul(b.sext(cl, I64), b.const_i64(HD * 2))
    ckv_b = b.select(b.cmp_ne(big, c(0)), b.add(p["ckv"], b.mul(b.sext(bq, I64), slot_bytes)), p["ckv"])
    r_ckv = e.rsrc(ckv_b)
    cv, oks = [], []
    if VD:  # the compressed rows: the same lane mapping, -1 pads past the extent (zeros)
        for ci in range(NCH):
            idx = idxd[ci] if k.seldirect else e.ld(b.add(c(M_I + ci * CH), kq_d))
            slot = b.smin(b.smax(idx, c(0)), b.sub(cl, c(1)))
            row = b.select(b.cmp_ne(big, c(0)), slot, b.add(b.mul(bq, cl), slot))
            voff = b.select(b.cmp_ge(idx, c(0)), b.mul(b.add(b.mul(row, c(HD // 2)), soff_d), c(4)), c(OOB))
            vdma(r_ckv, b.add(c((2 + ci) * CH), b.mul(wave, c(8))), voff)
    for ci in range(0 if VD else NCH):
        idx = idxd[ci] if k.seldirect else e.ld(b.add(c(M_I + ci * CH), kq))
        slot = b.smin(b.smax(idx, c(0)), b.sub(cl, c(1)))
        oks.append(b.cmp_ge(idx, c(0)))
        row = b.select(b.cmp_ne(big, c(0)), slot, b.add(b.mul(bq, cl), slot))
        cv.append(b.buffer_load_cp(r_ckv, b.mul(b.add(b.mul(row, c(HD // 2)), soff), c(4)), c(0), dwords=4, aux=0))
    pw = b.mul(piece, c(4))
    for i in range(0 if VD else WINDOW // CH):
        kp = b.add(wlo, b.add(c(i * CH), kq))
        ok = b.land(b.cmp_ge(kp, c(0)), b.cmp_lt(kp, base))
        words = [b.select(ok, b.vec_extract(wv[i], q), c(0)) for q in range(4)]
        if k.ownskip:  # the step's own slots (kp >= base) are the own rows': not zeroed first
            with b.scf_if(b.cmp_lt(kp, base)):
                e.st(vk_word(b.add(c(i * CH), kq), pw), b.vec_pack(words, I32), 4)
        else:
            e.st(vk_word(b.add(c(i * CH), kq), pw), b.vec_pack(words, I32), 4)
    for ci in range(0 if VD else NCH):
        words = [b.select(oks[ci], b.vec_extract(cv[ci], q), c(0)) for q in range(4)]
        ckey = b.add(c(2 * CH), b.add(b.mul(kq, c(8)), c(ci))) if k.seldirect == 2 else b.add(c((2 + ci) * CH), kq)
        e.st(vk_word(ckey, pw), b.vec_pack(words, I32), 4)
    if not k.ownskip:
        if VD:  # this wave's DMA landed before the barrier publishes it
            b.s_waitcnt(vmcnt=0)
        e.barrier()  # every ring row (zeros at the step's own slots) is in before the own rows overwrite them
    # the own rows: the attempt's words, polled again by a thread whose tags were stale
    g = b.sub(b.add(base, jo), wlo)  # the key's index in the window
    at = vk_word(g, wo)
    stale = b.lor(b.cmp_ne(b.vec_extract(own_raw, 1), tag), b.cmp_ne(b.vec_extract(own_raw, 3), tag))
    if k.mutate != "own":
        with b.scf_if(b.land(own_ok, b.lnot(stale))):
            e.st(at, b.vec_pack([b.vec_extract(own_raw, 0), b.vec_extract(own_raw, 2)], I32), 2)
        with b.scf_if(b.land(own_ok, stale)):
            got = e.poll(tag, [(reg["kvn"], own_pair, 2)])[0]
            e.st(at, b.vec_pack([b.vec_extract(got, 0), b.vec_extract(got, 2)], I32), 2)
    if not k.ownskip:
        with b.scf_if(b.cmp_eq(tid, c(0))):  # the ring is read (RingWrite)
            reg["md"].put(task, [c(1), tag])
    stamp(1)

    # ---- the softmax of the task's heads over all ten chunks (the producers' thread mapping)
    gch = b.div(tid, c(M * 32))
    h = b.land(b.lshr(tid, c(5)), c(M - 1))
    j = b.land(tid, c(31))
    hh = b.add(b.mul(hg, c(M)), h)
    ccs = [b.add(gch, c(G * i)) for i in range(NSPEC)]
    okc = [b.cmp_lt(cc, c(NCT)) for cc in ccs]
    ccs = [b.smin(cc, c(NCT - 1)) for cc in ccs]
    kk = b.mul(j, c(2))
    specs = [(reg["s"], b.add(b.mul(b.add(b.mul(b.add(b.mul(tt, c(NCT)), cc), c(TILE)), hh), c(CH)), kk), 2)
             for cc in ccs]  # fmt: skip
    scale = f(1.0 / math.sqrt(HD))
    ninf = b.const_f32(float("-inf"))

    def masks():
        out = []
        for cc in ccs:
            ci = b.smin(b.smax(b.sub(cc, c(2)), c(0)), c(NCH - 1))
            row = []
            for q in range(2):
                win = b.land(b.cmp_lt(cc, c(2)), b.cmp_ge(b.add(b.add(wlo, b.mul(cc, c(CH))), b.add(kk, c(q))), c(0)))
                sel = e.ld(b.add(b.add(c(M_I), b.mul(ci, c(CH))), b.add(kk, c(q))))
                row.append(b.lor(win, b.land(b.cmp_ge(cc, c(2)), b.cmp_ge(sel, c(0)))))
            out.append(row)
        return out

    # the tail's per-thread loads (sink, the RoPE pair): issued before the poll (premask) or after the softmax
    T1 = bool(k.tail1) and M * 64 <= THREADS  # (tail1: one dim a thread, 64 threads a head)
    if T1:
        hh2 = b.smin(b.lshr(tid, c(6)), c(M - 1))
        dl = b.land(tid, c(63))
        d0 = b.add(b.mul(s, c(SLICE)), dl)  # the thread's dim
    else:
        hh2 = b.smin(b.lshr(tid, c(5)), c(M - 1))
        d0 = b.add(b.mul(s, c(SLICE)), kk)
    head = b.add(b.add(b.mul(ht, c(TILE)), b.mul(hg, c(M))), hh2)

    def tail_loads():
        sink_ = b.bitcast(b.buffer_load_cp(e.rsrc(p["sink"]), b.mul(head, c(4)), c(0), dwords=1, aux=0), F32)
        first = b.lshr(b.smax(b.sub(d0, c(HD - ROPE)), c(0)), c(1))
        cs_ = b.buffer_load_cp(e.rsrc(p["rope"]), b.mul(b.add(b.mul(pos, c(ROPE)), b.mul(first, c(2))), c(4)), c(0),
                               dwords=2, aux=0)  # fmt: skip
        return sink_, cs_

    if k.seldirect and k.ownskip:  # the LDS copy of the selection, written by other threads, visible to the masks
        e.barrier()
    if k.premask:
        oks2 = masks()
        sink, cs = tail_loads()
    if SA2:  # the DMA landed (vmcnt): each lane's 4 words at its slot of its wave's spec block
        b.s_waitcnt(vmcnt=0)
        satt = [e.ld4(b.add(c(M_SA + i * SATT_WORDS), b.add(b.mul(wave, c(NSPEC * SATT_WORDS)), b.mul(lane, c(4)))))
                for i in range(NSPEC)]  # fmt: skip
    if SA:  # every tag current in the attempt: its words; else this wave polls (staged through LDS)
        stale = None
        for v in satt:
            st_ = b.lor(b.cmp_ne(b.vec_extract(v, 1), tag), b.cmp_ne(b.vec_extract(v, 3), tag))
            stale = st_ if stale is None else b.lor(stale, st_)
        anys = b.cmp_ne(b.wave_any(b.select(stale, c(1), c(0))), c(0))
        stg = b.add(c(M_P), b.mul(tid, c(2 * NSPEC)))
        with b.scf_if(anys):
            pol = e.poll(tag, specs)
            for i, v in enumerate(pol):
                e.st(b.add(stg, c(2 * i)), b.vec_pack([b.vec_extract(v, 0), b.vec_extract(v, 2)], I32), 2)
        got = []
        for i, v in enumerate(satt):
            lw = e.b.smem_load_vN(e.pool, b.add(stg, c(2 * i)), dtype=I32, n=2)
            w0 = b.select(anys, b.vec_extract(lw, 0), b.vec_extract(v, 0))
            w1 = b.select(anys, b.vec_extract(lw, 1), b.vec_extract(v, 2))
            got.append(b.vec_pack([w0, tag, w1, tag], I32))
    else:
        got = e.poll(tag, specs)
    stamp(2)
    if not k.premask:
        oks2 = masks()
    svs = []
    for i, cc in enumerate(ccs):
        sv = []
        for q in range(2):
            raw = b.fmul(b.bitcast(b.vec_extract(got[i], 2 * q), F32), scale)
            sv.append(b.select(oks2[i][q], raw, ninf))
        svs.append(sv)
    mxs = e.bfly([e.fmaximum(sv[0], sv[1]) for sv in svs], (1, 2, 4, 8, 16), e.fmaximum)
    j0 = b.cmp_eq(j, c(0))
    for i, cc in enumerate(ccs):
        with b.scf_if(b.land(okc[i], j0)):
            e.stf(b.add(b.add(c(M_X), b.mul(h, c(NCT))), cc), mxs[i])
    if VD and k.mutate != "vwait":  # (vdma) this wave's V DMA landed: the first point another thread reads V
        b.s_waitcnt(vmcnt=0)
    e.barrier()
    stamp(7)
    if k.ownskip:  # (after a barrier every thread passed after its ring loads landed)
        with b.scf_if(b.cmp_eq(tid, c(0))):  # the ring is read (RingWrite)
            reg["md"].put(task, [c(1), tag])
    m_prev, ms = f(NEG_INIT), []
    for cidx in range(NCT):
        m_prev = e.fmaximum(m_prev, e.ldf(b.add(c(M_X + cidx), b.mul(h, c(NCT)))))
        ms.append(m_prev)

    def pick(vals, i):
        out = vals[0]
        for q in range(1, len(vals)):
            out = b.select(b.cmp_eq(i, c(q)), vals[q], out)
        return out

    pvs = [[b.exp(b.fsub(v, pick(ms, cc))) for v in sv] for sv, cc in zip(svs, ccs)]
    lss = e.bfly([b.fadd(pv[0], pv[1]) for pv in pvs], (1, 2, 4, 8, 16), b.fadd)
    for i, cc in enumerate(ccs):
        with b.scf_if(okc[i]):
            e.st(b.add(b.add(c(M_P), b.mul(h, c(PP))), b.add(b.mul(cc, c(CH // 2)), j)), e.bf_pair(*pvs[i]))
        with b.scf_if(b.land(okc[i], j0)):
            e.stf(b.add(b.add(c(ML), b.mul(h, c(NCT))), cc), lss[i])

    # ---- the tail: P V, the recurrence, the output (thread: head hh2, dims d0, d0 + 1)
    if not k.premask:
        sink, cs = tail_loads()
    e.barrier()
    stamp(3)

    def scalars():
        """The recurrence's maxima, corrections and denominator (the reference's order: the window's two chunks from
        -1e30, then the compressed ones) -> (corrs, den, m)."""
        lsv = [e.ldf(b.add(c(ML + cidx), b.mul(hh2, c(NCT)))) for cidx in range(NCT)]
        if k.msreuse and not T1:  # the softmax's chain (the same maximum ops on the same words): m after chunk c = ms[c]
            mvals = ms
        else:
            mxv = [e.ldf(b.add(c(M_X + cidx), b.mul(hh2, c(NCT)))) for cidx in range(NCT)]
            mvals, m_ = [], f(NEG_INIT)
            for cidx in range(NCT):
                m_ = e.fmaximum(m_, mxv[cidx])
                mvals.append(m_)
        corrs_, den_, m_prev_ = [], b.const_f32(0.0), f(NEG_INIT)
        for cidx in range(NCT - (1 if k.mutate == "rec" else 0)):
            corr_ = b.exp(b.fsub(m_prev_, mvals[cidx]))
            den_ = b.fadd(b.fmul(den_, corr_), lsv[cidx])
            corrs_.append(corr_)
            m_prev_ = mvals[cidx]
        return corrs_, den_, m_prev_

    if k.early:
        corrs, den, m = scalars()
    # P V of every chunk for the slice: wave w the 16 dims 16 (w % 4) .., the chunks of half w // 4
    lm, ldq = b.land(lane, c(15)), b.lshr(lane, c(4))
    cb, half = b.land(wave, c(3)), b.lshr(wave, c(2))
    dim = b.add(b.mul(cb, c(16)), lm)
    vkw = vk_word(b.add(b.mul(ldq, c(8)), b.lshr(lm, c(2))), b.add(b.mul(cb, c(8)), b.mul(b.land(lm, c(3)), c(2))))
    ts = []
    for i in range(NCT // 2):
        cc = b.add(b.mul(half, c(NCT // 2)), c(i))
        acc = b.vec_pack([b.const_f32(0.0)] * 4, F32)
        for kst in range(CH // 32):
            if k.mutate == "pv_kst" and kst == 1:
                continue
            a = b.vec_bitcast(e.ld4(b.add(b.add(c(M_P), b.mul(lm, c(PP))),
                                          b.add(b.mul(cc, c(CH // 2)), b.add(c(kst * 16), b.mul(ldq, c(4)))))),
                              VectorType(BF16, 8))  # fmt: skip
            off = b.mul(b.add(b.mul(cc, c(CH)), c(kst * 32)), c(SLICE // 2))  # words: (cc 64 + 32 kst) x 64 bf16
            w0 = e.tr16(b.add(vkw, off))
            w1 = e.tr16(b.add(vkw, b.add(off, c(4 * SLICE // 2))))
            bv = b.vec_bitcast(b.vec_concat(w0, w1), VectorType(BF16, 8))
            acc = b.mfma_f32_16x16x32_bf16(a, bv, acc)
        ts.append((cc, acc))
    if not k.tsep:
        e.barrier()  # every wave's V reads are done: t overlays them
    if k.stamps:
        with b.scf_if(b.cmp_eq(wave, c(0))):
            stamp(6)
    for cc, acc in ts:
        for q in range(4):
            if q < M:  # (lanes with ld 4 + q >= mheads store nothing)
                with b.scf_if(b.cmp_lt(b.add(b.mul(ldq, c(4)), c(q)), c(M))):
                    e.stf(b.add(b.add(c(M_T), b.mul(b.add(b.mul(cc, c(M)), b.add(b.mul(ldq, c(4)), c(q))), c(OP))),
                                dim), b.vec_extract(acc, q))  # fmt: skip
    e.barrier()
    stamp(4)
    # the reference's recurrence in chunk order (num from 0: the window's state as ``softmax_all`` builds it)
    QN = 1 if T1 else 2  # dims a thread
    dcol = dl if T1 else kk
    tv = [[e.ldf(b.add(b.add(c(M_T + cidx * M * OP), b.mul(hh2, c(OP))), b.add(dcol, c(q)))) for q in range(QN)]
          for cidx in range(NCT)]  # fmt: skip
    if not k.early:
        corrs, den, m = scalars()
    zero = b.const_f32(0.0)
    num = [b.fadd(b.fmul(b.fadd(b.fmul(zero, corrs[0]), tv[0][q]), corrs[1]), tv[1][q]) for q in range(QN)]
    for cidx in range(2, len(corrs)):
        num = [b.fadd(b.fmul(num[q], corrs[cidx]), tv[cidx][q]) for q in range(QN)]
    if k.mutate != "sink":
        den = b.fadd(den, b.exp(b.fsub(sink, m)))
    o = [e.bfr(b.fdiv(num[q], den)) for q in range(QN)]
    is_rope = b.cmp_ge(d0, c(HD - ROPE))
    cosv = b.bitcast(b.vec_extract(cs, 0), F32)
    sinv = b.bitcast(b.vec_extract(cs, 1), F32)
    s2 = b.fneg(sinv)  # the conjugate (inverse RoPE)
    keep = b.cmp_ne(unr, c(0))

    def xo(v, off):  # lane ^ off's value
        return b.bitcast(e.xshfl_i(b.bitcast(v, I32), off), F32)

    def absf(v):  # monofly abs_f32: max(x, -x) (LLVM folds it to |x| either way)
        return e.fmaximum(v, b.fneg(v))

    if T1:  # one dim a lane: the pair's other value from lane ^ 1 (even lane: the real part, odd: the imaginary)
        oo = xo(o[0], 1)
        even = b.cmp_eq(b.land(dl, c(1)), c(0))
        o0, o1 = b.select(even, o[0], oo), b.select(even, oo, o[0])
        a_ = e.ffma(o0, cosv, b.fneg(b.fmul(o1, s2)))
        b_ = e.ffma(o1, cosv, b.fmul(o0, s2))
        a_ = b.select(keep, a_, e.bfr(a_))
        b_ = b.select(keep, b_, e.bfr(b_))
        ov = b.select(is_rope, b.select(even, a_, b_), o[0])
        am = e.bfly([absf(ov)], (1, 2, 4, 8, 16), e.fmaximum)[0]  # the 32-dim group: 32 lanes
        code = e.e8m0(am)
        word = e.quant_word([ov, xo(ov, 1), xo(ov, 2), xo(ov, 3)], code)
        wlead, clead, nthr = b.cmp_eq(b.land(dl, c(3)), c(0)), b.cmp_eq(b.land(dl, c(31)), c(0)), M * 64
    else:
        a_ = e.ffma(o[0], cosv, b.fneg(b.fmul(o[1], s2)))
        b_ = e.ffma(o[1], cosv, b.fmul(o[0], s2))
        a_ = b.select(keep, a_, e.bfr(a_))
        b_ = b.select(keep, b_, e.bfr(b_))
        o = [b.select(is_rope, a_, o[0]), b.select(is_rope, b_, o[1])]
        am = e.bfly([e.fmaximum(absf(o[0]), absf(o[1]))], (1, 2, 4, 8), e.fmaximum)[0]
        code = e.e8m0(am)
        nb = [xo(o[q], 1) for q in range(2)]
        word = e.quant_word([o[0], o[1], nb[0], nb[1]], code)
        wlead, clead, nthr = b.cmp_eq(b.land(j, c(1)), c(0)), b.cmp_eq(b.land(j, c(15)), c(0)), M * 32
    if k.mutate == "out_bit":
        word = b.xor(word, c(1))
    if k.mutate == "code":
        code = b.add(code, c(1))
    e0 = b.add(b.add(b.mul(b.mul(r, heads), c(HD)), b.mul(head, c(HD))), d0)
    okt = b.cmp_lt(tid, c(nthr))
    with b.scf_if(b.land(wlead, okt)):
        reg["oq"].put(b.lshr(e0, c(2)), [word, tag])
    with b.scf_if(b.land(clead, okt)):
        reg["os"].put(b.lshr(e0, c(5)), [code, tag])
    if k.endbar:  # only when this CTA runs another task (the emitter joins the CTA before the next LDS stage)
        with b.scf_if(b.cmp_lt(b.add(task, c(256)), p["ntasks"])):
            e.barrier()
    else:
        e.barrier()  # the next task reuses the LDS
    stamp(5)
    b.ret()
    return b.kernel


def lower_smerge_function(knobs: SmergeKnobs = SmergeKnobs(), *, arch="gfx950", flavor=None):
    from ..qwen38_mono.device_function import DeviceFunctionKnobs, lower_device_function

    return lower_device_function(build_smerge(knobs), arch=arch, symbol=symbol(knobs), llvm_flavor=flavor,
                                 knobs=DeviceFunctionKnobs(lds="arg", inline="always"))  # fmt: skip


def write_smerge_function(out: str, knobs: SmergeKnobs = SmergeKnobs(), *, arch="gfx950", flavor=None,
                          bitcode="auto"):  # fmt: skip
    from ..qwen38_mono.device_function import write_device_function

    fn = lower_smerge_function(knobs, arch=arch, flavor=flavor)
    paths = write_device_function(fn, out, stem=f"{STAGE}_{knobs.variant()}", bitcode=bitcode)
    with open(paths["json"]) as fh:
        meta = json.load(fh)
    meta.update(stage=STAGE, stage_knobs=knobs.spec(), mheads=knobs.mheads, abi_v41mf=[list(x) for x in ABI],
                lds_words=lds_words(knobs.mheads, knobs.tsep, knobs.sattempt), mutate=knobs.mutate, stamps=knobs.stamps)  # fmt: skip
    with open(paths["json"], "w", newline="\n") as fh:
        json.dump(meta, fh, indent=1)
        fh.write("\n")
    return paths


def write_set(out: str, spec: str = "", mheads=(2, 4, 8, 16), *, arch="gfx950", flavor=None, bitcode="auto",
              name: str = ""):  # fmt: skip
    """One function per ``mheads`` for the knobs ``spec`` and an index ``<name or variant>.json`` ({"stage",
    "knobs", "variants": {mheads: json path}}): the value of monofly's ``MLASparseAttn(rk=)``."""
    import os

    var = {}
    for mh in mheads:
        kn = knobs_from_spec(",".join(x for x in (spec, f"mheads={mh}") if x))
        var[str(mh)] = os.path.abspath(write_smerge_function(out, kn, arch=arch, flavor=flavor, bitcode=bitcode)["json"])
    stem = name or ("set_" + (knobs_from_spec(spec).variant().split("_", 1)[1]))
    path = os.path.join(out, f"{STAGE}_{stem}.json")
    with open(path, "w", newline="\n") as fh:
        json.dump({"stage": STAGE, "knobs": spec, "variants": var}, fh, indent=1)
        fh.write("\n")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="emit the rocKE V4.1 sparse-merge stage function sets")
    ap.add_argument("--out", required=True)
    ap.add_argument("--knobs", action="append", default=[], help="k=v,.. (repeatable: one set each)")
    ap.add_argument("--mheads", default="2,4,8,16")
    ap.add_argument("--flavor", default=None)
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--bitcode", choices=("auto", "yes", "no"), default="no")
    a = ap.parse_args(argv)
    bc = {"auto": "auto", "yes": True, "no": False}[a.bitcode]
    mh = tuple(int(x) for x in a.mheads.split(","))
    out = [{"knobs": spec, "set": write_set(a.out, spec, mh, arch=a.arch, flavor=a.flavor, bitcode=bc)}
           for spec in (a.knobs or [""])]  # fmt: skip
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
