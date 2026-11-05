# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""WP-L2f / E10: the large-batch MXFP4 head with a fused greedy argmax (rows 5-128, TP1 and vocab-parallel TP2), gfx950.

The GEMM is WP-L2a's E1 (:mod:`.e1`, imported and subclassed, never edited): ``logits[token, row] = X W^T`` over the
rank's vocabulary rows (TP1 248320, TP2 124160 = rank ``r`` owns global rows ``[r M, (r + 1) M)``), any E1 plan with one
K split and the 16x16x128 MFMA (every head pick of ``e1_plans.PLANS``). Only the epilogue changes (``logits=out`` keeps
E1's bf16 store as well):

* **Unit argmax.** Each lane folds its ``4 mt`` rows of every token tile in increasing row order (strict ``>``: the lowest
  row wins a tie), keys of ATOM's rule (``akey="bf16"``: the signed-order key of the bf16-rounded logit, ``-0 == +0``,
  ``mx_decode_gemv._bf16_key``; ``"f32"``: FlyTape's f32 compare, a tolerance lever); rows past ``M`` (row-group padding)
  never win. The four lane groups of a token fold by ``xor 16 / 32`` shuffles (larger key, then lower row), the waves of
  the CTA through LDS (same rule: order-free, deterministic).
* **Cross-unit merge** (``mrg="amax"``): one 64-bit ``global_atomic_umax_x2`` per (unit, token) of
  ``(key ^ 0x80000000) << 32 | ~row`` into ``amax[layer][token]``: unsigned max = larger key, then lower row, i.e. ATOM's
  ``tie="low"`` independent of arrival order (deterministic). After its atomics complete (``vmcnt(0)``) the unit arrives on
  its token block's counter ``acnt[layer][block]``; the last of the ``RG`` row groups reads AND clears the word with one
  returning ``global_atomic_swap_x2`` (zero at rest, any build may follow), and zeroes the counter.
* **TP2 merge** (the CCL argmax merge, ``DESIGN-CCL`` WP-C2's pair form, fused): that last arriver pushes the rank's
  ``(key, global id)`` pair of each token, tagged, into both ranks' IPC arenas (``ccl.wire.push_v4``, ``sc0 sc1``), polls
  both ranks' pairs in its own arena (``ccl.wire.poll_pairs``, deadline + abort word) and merges in rank order (larger
  key, then lower global id: rank 0 wins ties), so both ranks write the same token. Slots
  ``[layer][launch parity][src rank][token]`` (16 bytes); tag ``= epoch L + layer + 1`` from the arena's own epoch word,
  advanced by the launch's last CTA (graph-replay safe, no host reset). A rank can be at most one launch ahead of its peer
  per slot parity (it needs the peer's pair of launch ``e + 1`` before it finishes launch ``e + 1``), so two parities
  suffice for every walk (stream and phase).
* Output ``aout[layer][token] = (global id, key)`` i32 pairs (TP2: the merged pair, identical on both ranks).

Knobs (:class:`HeadKnobs`, default-off levers in the name): ``akey``, ``lout`` (also the bf16 logits), ``xdelay`` (diag:
rank 1 sleeps before its push), and the E1 plan (any E1 lever through ``e1``).

Mutants (must fail; the bench plants ties and an all-negative token): ``tie`` (in-lane ``>=``: a lane's tie goes to the
higher row), ``pad`` (padding rows not masked: an all-negative token picks a padding row, TP2 rows-512 plans),
``ulo`` (the packed low word is ``row``: cross-unit ties go to the higher row), ``early`` (the merge runs at ``RG - 1``
arrivals while CTA 0 starts every unit late: a stale / missing contribution), ``xpeer`` (TP2: the peer's pair ignored),
``xorder`` (TP2: equal keys take rank 1's id), ``xstale`` (TP2: the poll accepts any non-zero tag and rank 1 pushes late:
the previous launch's pair is read; the deterministic stale-read twin), plus E1's (``e1.mutant``: ``kskip``, ...).

**W4A16** (``w4a16``, the fix round's ATOM-precision knob, default off): ATOM's mono head keeps the activations bf16
(``Mxfp4LMHead``: ``tl.dot_scaled`` bf16 x e2m1 up to 16 rows, the bf16 GEMM on the same MXFP4-valued weights above).
With ``w4a16`` the E1 core is built for bf16 activations (``e1.w="bf16"``: X staged in LDS as bf16, the bf16
16x16x32 MFMA) while the weights stay MXFP4 (codes + E8M0 scales, ``wlay`` row or tile, never ``sop``): each lane's 16
code bytes (32 E2M1 values at K ``128 q + 32 g``) are decoded in registers with ``v_cvt_scalef32_pk_bf16_fp4`` (exact:
every MXFP4 value is a bf16) into the four ``<8 x bf16>`` A fragments of K pieces ``8 m``, the same K order as E1's
bf16 path, so a ``w4a16`` head and the ``e1.w="bf16"`` head over the dequantized weights issue the same MFMAs on the
same values (bitwise-equal logits). Mutant ``w4scale``: the decode ignores the block scale.

Python engine only (inline asm, ``int_to_ptr``); names ``q38lbhd_*``; nothing served imports it. Measurements:
``experiments/qwen38-true-mono/large-batch/WP-L2f.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

from dataclasses import replace as _rep

from rocke.core.ir import F32, I32, I64, KernelDef, PtrType, Value
from rocke.helpers import device_ops as D

from . import e1 as E1M

__all__ = [
    "HeadKnobs",
    "HEAD_MUTANTS",
    "VOCAB",
    "arena_bytes",
    "build_head_kernel",
    "head_kernel_name",
    "head_params",
    "ref_keys_bf16",
    "ref_keys_f32",
    "ARENA_OFF",
    "W4_MUTANTS",
    "w4a16_e1",
    "plan_w4a16",
    "HEAD_W4_PLANS",
    "HEAD_W16_PLANS",
]

VOCAB = 248320
HEAD_MUTANTS = ("none", "tie", "pad", "ulo", "early", "xpeer", "xorder", "xstale")
W4_MUTANTS = ("w4scale",)  # w4a16 only: the decode ignores the block scale
ARENA_OFF = 1 << 16  # first byte of the head's slots in the IPC arena (the CCL control words live below)
INT_MIN = -0x80000000
SLOT = 16


def _c(b, v: int) -> Value:
    return b.const_i32(int(v))


@dataclass(frozen=True)
class HeadKnobs:
    e1: E1M.E1Knobs = field(default_factory=E1M.E1Knobs)
    tp: int = 1  # 1 | 2 (vocab-parallel: each rank's M rows, the TP2 pair merge in the kernel)
    akey: str = "bf16"  # "bf16" (ATOM: argmax of the bf16 logits) | "f32" (FlyTape's f32 compare)
    lout: bool = False  # also store the bf16 logits (E1's Y)
    xdelay: int = 0  # diag: rank 1 sleeps xdelay x s_sleep 127 before its TP2 push
    # cross-unit merge: "unit" (every unit's winners -> one umax atomic per token) | "cta" (a CTA folds its units of
    # the layer in LDS first and issues one atomic per token at its last unit; the counter then counts CTAs: fewer
    # atomics on the hot token words; needs one token block a layer, e1.tb in (0, tiles))
    amrg: str = "unit"
    # W4A16: bf16 activations (e1.w="bf16") x MXFP4 weights decoded to bf16 in registers (ATOM's head precision)
    w4a16: bool = False
    mutant: str = "none"
    tag: str = ""

    def check(self) -> None:
        k = self.e1
        k.check()
        if k.splits != 1 or k.mfma != 16 or k.yq or k.kseg or k.chain or k.diag != "none" or k.kw != 1:
            raise ValueError("head: one K split, mfma 16, kw 1, no yq / kseg / chain / diag")
        if self.tp not in (1, 2):
            raise ValueError("tp 1 | 2")
        if self.akey not in ("bf16", "f32"):
            raise ValueError("akey bf16 | f32")
        if self.mutant not in HEAD_MUTANTS + W4_MUTANTS:
            raise ValueError(f"mutant in {HEAD_MUTANTS + W4_MUTANTS}")
        if self.mutant in W4_MUTANTS and not self.w4a16:
            raise ValueError("the w4scale mutant needs w4a16")
        if self.w4a16 and not (k.w == "bf16" and not k.sop and k.xsrc == "lds" and k.xslay == "row"
                               and k.xscale == "lds" and k.wsb == "shift" and not k.ragged
                               and k.mutant in ("none", "kskip")):
            raise ValueError("w4a16: e1.w=bf16 (X), no sop, xsrc=lds, xslay=row, xscale=lds, wsb=shift, K a multiple of "
                             "128, e1 mutant none | kskip")
        if self.mutant in ("xpeer", "xorder", "xstale") and self.tp != 2:
            raise ValueError("the TP2 merge mutants need tp=2")
        if not 0 <= self.xdelay <= 64:
            raise ValueError("xdelay 0..64")
        if self.amrg not in ("unit", "cta"):
            raise ValueError("amrg unit | cta")
        if self.amrg == "cta" and k.tb not in (0, k.tiles):
            raise ValueError("amrg=cta: one token block a layer (tb 0)")

    @property
    def tpb(self) -> int:  # tokens of the bucket (token tiles x 16)
        return self.e1.tiles * E1M.TILE

    def suffix(self) -> str:
        parts = [f"tp{self.tp}"]
        d = HeadKnobs()
        for f in fields(self):
            if f.name in ("e1", "tp", "tag"):
                continue
            v = getattr(self, f.name)
            if v == getattr(d, f.name):
                continue
            parts.append((f.name if v else f"no{f.name}") if isinstance(v, bool) else f"{f.name}{v}")
        if self.tag:
            parts.append(self.tag)
        return "_".join(parts)


def head_kernel_name(hk: HeadKnobs) -> str:
    s = hk.e1.suffix()
    w16 = "_w16" if hk.e1.w == "bf16" and not hk.w4a16 else ""  # E1's suffix omits w: the bf16-weight head is marked
    return f"q38lbhd_k{hk.e1.k}_b{hk.tpb}_{hk.suffix()}{w16}" + (f"_{s}" if s else "")


def head_params(hk: HeadKnobs) -> list[str]:
    """E1's parameters (:func:`e1.e1_params`: i64 pointers, then i32 ``m, rows, layers, gsync``), then i64 ``amax, acnt,
    aout`` (+ ``peer0, peer1, arena`` at TP2), then i32 ``rank`` (host packs ``<Q..iiii Q.. i``; pad to 8)."""
    p = E1M.e1_params(hk.e1) + ["amax", "acnt", "aout"]
    if hk.tp == 2:
        p += ["peer0", "peer1", "arena"]
    return p + ["rank"]


def arena_bytes(layers: int, tpb: int) -> int:
    """IPC arena bytes the TP2 head needs past ``ARENA_OFF``: slots ``[layer][2][2][tpb]`` x 16 + control words."""
    return ARENA_OFF + layers * 4 * tpb * SLOT + 1024


CTL_EP = ARENA_OFF - 1024  # epoch word (advanced by the launch's last CTA)
CTL_END = ARENA_OFF - 768  # launch-end arrival counter (zero at rest)
CTL_ABORT = ARENA_OFF - 512  # stays zero: the poll's abort word


def ref_keys_bf16(logits_bf16_bits):
    """Host twin of the device key (numpy uint16 bf16 bits -> int32 signed-order keys, -0 == +0)."""
    import numpy as np

    b = logits_bf16_bits.astype(np.uint32) << np.uint32(16)
    b = np.where(b == 0x80000000, np.uint32(0), b).astype(np.uint32)
    return np.where(b & 0x80000000, b ^ 0x7FFFFFFF, b).astype(np.uint32).view(np.int32)


def ref_keys_f32(logits_f32):
    import numpy as np

    b = np.asarray(logits_f32, dtype=np.float32).view(np.uint32)
    b = np.where(b == 0x80000000, np.uint32(0), b).astype(np.uint32)
    return np.where(b & 0x80000000, b ^ 0x7FFFFFFF, b).astype(np.uint32).view(np.int32)


class _Head(E1M._E1):
    def __init__(self, hk: HeadKnobs):
        hk.check()
        self.hk = hk
        super().__init__(hk.e1)
        b, k = self.b, hk.e1
        b.kernel.name = head_kernel_name(hk)
        self.ha = {n: b.param(n, I64) for n in head_params(hk)[len(E1M.e1_params(k)) : -1]}
        self.rank = b.readfirstlane(b.param("rank", I32))
        self.ntok = k.tbt * E1M.TILE  # tokens of one unit / token block
        self.red = b.smem_alloc(I32, [k.waves, self.ntok, 2], name_hint="hdred")
        self.hflag = b.smem_alloc(I32, [2], name_hint="hdflag")
        if hk.amrg == "cta":
            self.cbest = b.smem_alloc(I32, [self.ntok, 2], name_hint="hdcb")
        if hk.w4a16:
            # MXFP4 weight strides (the base built the bf16 ones from e1.w; those values go unused)
            Mi = E1M._i64(b, self.M)
            if k.wlay == "tile":
                MT = E1M._i64(b, E1M._cdiv(b, self.M, _c(b, E1M.TILE)))
                self.wl = b.mul(MT, b.const_i64(k.nq * 1024))
                self.sl = b.mul(MT, b.const_i64(k.nq * 64))
            else:
                self.wl = b.mul(Mi, b.const_i64(k.k // 2))
                self.sl = b.mul(Mi, b.const_i64(E1M.ksp(k.k)))
            self.wl32 = b.trunc(self.wl, I32)
            self.sl32 = b.trunc(self.sl, I32)
            self._kmx = _rep(k, w="mx")  # E1's mx weight loads (non-sop), read through this view
        if hk.tp == 2:
            # this launch's epoch (every CTA reads it before it arrives at the launch end: the advance comes after all reads)
            from ..ccl import wire as W

            ar = b.readfirstlane(self.ha["arena"])
            self.ep = b.readfirstlane(W.load_word_sys(b, ar, _c(b, CTL_EP)))

    # ------------------------------------------------------------------ W4A16 (MXFP4 weights, bf16 activations)
    def wissue(self, u, c, xonly_gate=None, xonly=False):
        if not self.hk.w4a16:
            return super().wissue(u, c, xonly_gate, xonly)
        if xonly:
            return []  # X is staged in LDS (xsrc=lds): nothing rides with the weights
        b = self.b
        if not hasattr(u, "s_rs"):
            u.s_rs = E1M._rsrc(b, b.add(self.a["ws"], b.mul(E1M._i64(b, u.layer), self.sl)), self.sl32)
        kb = self.k
        self.k = self._kmx
        try:
            return super().wissue(u, c)  # per weight tile: 16 code bytes, then the scale dword
        finally:
            self.k = kb

    def _w4frags(self, wl, skip):
        """Per weight tile the four ``<8 x bf16>`` A fragments (K pieces ``8 m`` of the lane's 32)."""
        b, k = self.b, self.k
        out = []
        for mi in range(k.mt):
            codes, sw = wl[2 * mi], wl[2 * mi + 1]
            if self.hk.mutant == "w4scale":
                sc = b.bitcast(_c(b, 127 << 23), F32)
            else:
                sc = b.bitcast(b.shl(b.land(b.lshr(sw, self.sh8g), _c(b, 0xFF)), _c(b, 23)), F32)
            cv = b.vec_pack([_c(b, 0)] * 4, I32) if skip else codes
            fr = []
            for m in range(4):
                word = b.vec_extract(cv, m)
                p = [b.cvt_scalef32_pk_bf16_fp4(word, sc, i) for i in range(4)]
                fr.append(b.vec_concat(b.vec_concat(p[0], p[1]), b.vec_concat(p[2], p[3])))
            out.append(fr)
        return out

    def chunk_math(self, acc, wl, buf, skip, j=0, u=None):
        if not self.hk.w4a16:
            return super().chunk_math(acc, wl, buf, skip, j, u)
        b, k = self.b, self.k
        acc = [list(r) for r in acc]
        frags = self._w4frags(wl, skip)

        def ops_of(t):
            return [self.bop(t, buf, None, m) for m in range(4)]

        ahead = {}
        if k.bpre:
            for t in range(k.bpre):
                ahead[t] = ops_of(t)
            if k.sb:
                b.sched_barrier(0)
        for t in range(k.tbt):
            if k.bpre:
                if t + k.bpre < k.tbt:
                    ahead[t + k.bpre] = ops_of(t + k.bpre)
                cur = ahead.pop(t)
            else:
                cur = ops_of(t)
            for m in range(4):
                for mi in range(k.mt):
                    acc[mi][t] = b.mfma_f32_16x16x32_bf16(frags[mi][m], cur[m], acc[mi][t])
            if k.bpre and k.sb:
                b.sched_barrier(0)
        return acc

    # ------------------------------------------------------------------ keys
    def _key(self, v: Value) -> Value:
        b = self.b
        if self.hk.akey == "bf16":
            h = b.bitcast(b.trunc_f32_to_bf16(v), _I16())
            bits = b.shl(b.zext(h, I32), _c(b, 16))
        else:
            bits = b.bitcast(v, I32)
        bits = b.select(b.cmp_eq(bits, _c(b, INT_MIN)), _c(b, 0), bits)
        return b.select(b.cmp_lt(bits, _c(b, 0)), b.xor(bits, _c(b, 0x7FFFFFFF)), bits)

    def _pick(self, k1, r1, k2, r2):
        """(k2, r2) beats (k1, r1): larger key, then lower row."""
        b = self.b
        w = b.lor(b.cmp_gt(k2, k1), b.land(b.cmp_eq(k2, k1), b.cmp_lt(r2, r1)))
        return b.select(w, k2, k1), b.select(w, r2, r1)

    # ------------------------------------------------------------------ epilogue
    def epilogue(self, u, acc, segv=None) -> None:
        b, hk, k = self.b, self.hk, self.k
        if hk.lout:
            super().epilogue(u, acc, segv)
        R = k.rows_per_unit
        lastrow = self.M
        bests = []
        for t in range(k.tbt):
            bk = br = None
            for mi in range(k.mt):
                row0 = b.add(b.add(b.mul(u.rg, _c(b, R)), b.sub(self.wrow[mi], self.l16)), b.mul(self.g, _c(b, 4)))
                for e in range(4):
                    row = b.add(row0, _c(b, e))
                    key = self._key(b.vec_extract(acc[mi][t], e))
                    if hk.mutant != "pad":
                        key = b.select(b.cmp_lt(row, lastrow), key, _c(b, INT_MIN))
                    if bk is None:
                        bk, br = key, row
                    else:
                        take = b.cmp_ge(key, bk) if hk.mutant == "tie" else b.cmp_gt(key, bk)
                        bk, br = b.select(take, key, bk), b.select(take, row, br)
            for m in (16, 32):
                bk, br = self._pick(bk, br, b.warp_shuffle_xor(bk, m), b.warp_shuffle_xor(br, m))
            bests.append((bk, br))
        g0 = b.cmp_eq(self.g, _c(b, 0))
        with b.scf_if(g0):
            for t, (bk, br) in enumerate(bests):
                tl = b.add(_c(b, t * E1M.TILE), self.l16)
                b.smem_store_vN(self.red, [self.wave, tl, _c(b, 0)], bk, 1)
                b.smem_store_vN(self.red, [self.wave, tl, _c(b, 1)], br, 1)
        b.sync_lds_only()
        tid = self.tid
        amax = b.readfirstlane(b.add(self.ha["amax"], b.mul(E1M._i64(b, u.layer), b.const_i64(hk.tpb * 8))))
        with b.scf_if(b.cmp_lt(tid, _c(b, self.ntok))):
            tl = tid
            kb = b.vec_extract(b.smem_load_vN(self.red, _c(b, 0), tl, _c(b, 0), dtype=I32, n=1), 0)
            rb = b.vec_extract(b.smem_load_vN(self.red, _c(b, 0), tl, _c(b, 1), dtype=I32, n=1), 0)
            for wv in range(1, k.waves):
                k2 = b.vec_extract(b.smem_load_vN(self.red, _c(b, wv), tl, _c(b, 0), dtype=I32, n=1), 0)
                r2 = b.vec_extract(b.smem_load_vN(self.red, _c(b, wv), tl, _c(b, 1), dtype=I32, n=1), 0)
                kb, rb = self._pick(kb, rb, k2, r2)
            tok = b.add(u.tok0, tl)
            ok = b.land(u.valid, b.cmp_lt(tok, self.rows))
            if hk.amrg == "cta":
                # fold this unit into the CTA's running best (the CTA's first unit of the layer starts it); only the
                # CTA's last unit of the layer issues the atomic
                first = b.cmp_eq(u.i, _c(b, 0))
                k0 = b.vec_extract(b.smem_load_vN(self.cbest, tl, _c(b, 0), dtype=I32, n=1), 0)
                r0 = b.vec_extract(b.smem_load_vN(self.cbest, tl, _c(b, 1), dtype=I32, n=1), 0)
                k0 = b.select(first, _c(b, INT_MIN), k0)
                r0 = b.select(first, _c(b, 0x7FFFFFFF), r0)
                kb, rb = self._pick(k0, r0, kb, rb)
                b.smem_store_vN(self.cbest, [tl, _c(b, 0)], kb, 1)
                b.smem_store_vN(self.cbest, [tl, _c(b, 1)], rb, 1)
                ok = b.land(ok, b.cmp_eq(u.i, b.sub(self.cnt, _c(b, 1))))
            with b.scf_if(ok):
                hi = b.xor(kb, _c(b, INT_MIN))
                lo = rb if hk.mutant == "ulo" else b.xor(rb, _c(b, -1))
                packed = b.lor(b.shl(b.zext(hi, I64), b.const_i64(32)), b.zext(lo, I64))
                b.inline_asm("global_atomic_umax_x2 $0, $1, $2", "v,v,s,~{memory}", [b.mul(tok, _c(b, 8)), packed, amax], None)
        b.sync_lds_only()  # the next unit's red stores wait for this unit's reads
        if hk.amrg == "cta":
            with b.scf_if(b.cmp_eq(u.i, b.sub(self.cnt, _c(b, 1)))):
                self._merge(u)
        else:
            self._merge(u)

    def _merge(self, u) -> None:
        """Arrival on the token block's counter; the last of the RG row groups reads + clears each token's word."""
        b, hk, k = self.b, self.hk, self.k
        b.s_waitcnt(vmcnt=0)  # this CTA's umax atomics completed before its arrival
        b.sync_lds_only()
        cbase = b.readfirstlane(b.add(self.ha["acnt"], b.mul(E1M._i64(b, u.layer), b.const_i64(64 * 4))))
        cidx = b.mul(u.tbk, _c(b, 4))
        with b.scf_if(D.is_thread0(b)):
            b.smem_store_vN(self.hflag, [_c(b, 0)], _c(b, -1), 1)
            with b.scf_if(u.valid):
                old = D.atomic_add_scoped(b, cbase, cidx, _c(b, 1), scope="agent", returns=True)
                b.smem_store_vN(self.hflag, [_c(b, 0)], old, 1)
        b.sync_lds_only()
        seen = b.readfirstlane(b.vec_extract(b.smem_load_vN(self.hflag, _c(b, 0), dtype=I32, n=1), 0))
        arrivals = b.smin(self.U, _c(b, self.k.grid)) if hk.amrg == "cta" else self.RG  # CTAs with a unit / row groups
        need = b.sub(arrivals, _c(b, 2 if hk.mutant == "early" else 1))
        with b.scf_if(b.cmp_eq(seen, need)):
            D.acquire_l2(b, "agent")
            amax = b.readfirstlane(b.add(self.ha["amax"], b.mul(E1M._i64(b, u.layer), b.const_i64(hk.tpb * 8))))
            nact = self.ntok
            with b.scf_if(b.cmp_lt(self.tid, _c(b, nact))):
                tl = b.smin(self.tid, _c(b, self.ntok - 1))
                tok = b.add(u.tok0, tl)
                live = b.land(b.cmp_lt(self.tid, _c(b, self.ntok)), b.cmp_lt(tok, self.rows))
                old = b.inline_asm(
                    "global_atomic_swap_x2 $0, $1, $2, $3 sc0\ns_waitcnt vmcnt(0)",
                    "=&v,v,v,s,~{memory}",
                    [b.mul(tok, _c(b, 8)), b.const_i64(0), amax],
                    I64,
                    result_name_hint="hdswap",
                )
                key = b.xor(b.trunc(b.lshr(old, b.const_i64(32)), I32), _c(b, INT_MIN))
                row = b.trunc(old, I32) if hk.mutant == "ulo" else b.xor(b.trunc(old, I32), _c(b, -1))
                gid = b.add(b.mul(self.rank, self.M), row) if hk.tp == 2 else row
                if hk.tp == 2:
                    key, gid = self._xmerge(u, tok, key, gid)
                aout = b.readfirstlane(b.add(self.ha["aout"], b.mul(E1M._i64(b, u.layer), b.const_i64(hk.tpb * 8))))
                with b.scf_if(live):
                    p = b.int_to_ptr(aout, PtrType(I32, "global"))
                    b.store(gid, p, b.mul(tok, _c(b, 2)))
                    b.store(key, p, b.add(b.mul(tok, _c(b, 2)), _c(b, 1)))
            with b.scf_if(D.is_thread0(b)):
                b.store(_c(b, 0), b.int_to_ptr(cbase, PtrType(I32, "global")), u.tbk)

    def _xmerge(self, u, tok, key, gid):
        """TP2: push (key, gid) tagged into both arenas, poll both ranks' pairs here, merge in rank order."""
        from ..ccl import wire as W

        b, hk = self.b, self.hk
        L = self.L
        seq = b.add(b.mul(self.ep, L), u.layer)
        tag = b.readfirstlane(W.tag_add(b, seq, _c(b, 1)))
        par = b.land(self.ep, _c(b, 1))
        base = b.add(_c(b, ARENA_OFF), b.mul(b.add(b.mul(u.layer, _c(b, 2)), par), _c(b, 2 * hk.tpb * SLOT)))

        def off(src):
            return b.add(base, b.mul(b.add(b.mul(src, _c(b, hk.tpb)), tok), _c(b, SLOT)))

        delayed = hk.xdelay or (hk.mutant == "xstale")
        if delayed:
            with b.scf_if(b.cmp_eq(self.rank, _c(b, 1))):
                b.inline_asm("\n".join(["s_sleep 127"] * (hk.xdelay or 32)), "~{memory}", [], None)
        peers = [b.readfirstlane(self.ha["peer0"]), b.readfirstlane(self.ha["peer1"])]
        W.push_v4(b, peers, off(self.rank), b.vec_pack([key, tag, gid, tag], I32))
        own = b.readfirstlane(self.ha["arena"])
        t64 = b.zext(tag, I64)
        neg = b.sub(b.const_i64(0), b.add(t64, b.const_i64(1)))
        win = b.const_i64(1 << 30)
        dl = b.add(D.memrealtime(b), b.const_i64(200_000_000))  # 2 s
        abort = b.add(own, b.const_i64(CTL_ABORT))
        pairs, _mask, st = W.poll_pairs(b, own, [off(_c(b, 0)), off(_c(b, 1))], t64, neg, win, dl, abort,
                                        mutant="any" if hk.mutant == "xstale" else "none")
        with b.scf_if(b.land(b.cmp_ne(st, _c(b, 0)), b.cmp_eq(b.land(self.tid, _c(b, 63)), _c(b, 0)))):
            D.atomic_add_scoped(b, b.readfirstlane(self.a["gs"]), _c(b, 0), _c(b, 1), scope="agent")
        k0, i0 = b.trunc(pairs[0], I32), b.trunc(pairs[1], I32)
        k1, i1 = b.trunc(pairs[2], I32), b.trunc(pairs[3], I32)
        if hk.mutant == "xpeer":
            mine = b.cmp_eq(self.rank, _c(b, 0))
            return b.select(mine, k0, k1), b.select(mine, i0, i1)
        if hk.mutant == "xorder":
            w = b.lor(b.cmp_gt(k1, k0), b.land(b.cmp_eq(k1, k0), b.cmp_gt(i1, i0)))
            w = b.lor(w, b.cmp_eq(k1, k0))
        else:
            w = b.lor(b.cmp_gt(k1, k0), b.land(b.cmp_eq(k1, k0), b.cmp_lt(i1, i0)))
        return b.select(w, k1, k0), b.select(w, i1, i0)

    def bardrop_delay(self) -> None:
        super().bardrop_delay()

    # ------------------------------------------------------------------ build
    def build(self) -> KernelDef:
        b, hk = self.b, self.hk
        if hk.mutant == "early":
            # CTA 0 starts every unit late (E1's bardrop sleep, hooked into the unit start below), so a merge released one
            # arrival early misses its row group's contribution deterministically
            self._early = True
        kd = super().build()
        if hk.tp == 2:
            # launch end: one returning arrival per CTA after its last unit; the last CTA zeroes the counter and advances
            # the epoch (after every CTA read it at its start)
            ar = b.readfirstlane(self.ha["arena"])
            b.s_waitcnt(vmcnt=0)
            b.sync_lds_only()
            with b.scf_if(D.is_thread0(b)):
                old = D.atomic_add_scoped(b, ar, _c(b, CTL_END), _c(b, 1), scope="agent", returns=True)
                with b.scf_if(b.cmp_eq(old, _c(b, self.k.grid - 1))):
                    from ..ccl import wire as W

                    W.store_word_sys(b, ar, _c(b, CTL_END), _c(b, 0))
                    W.store_word_sys(b, ar, _c(b, CTL_EP), b.add(self.ep, _c(b, 1)))
        return kd


def _I16():
    from rocke.core.ir import I16

    return I16


def build_head_kernel(hk: HeadKnobs) -> KernelDef:
    h = _Head(hk)
    if hk.mutant == "early":
        orig = h.seg_loads

        def seg_loads(u):  # the unit start hook E1 calls once per unit: CTA 0 sleeps there
            h.bardrop_delay()
            return orig(u)

        h.seg_loads = seg_loads
    kd = h.build()
    if hk.e1.hz:
        from .insitu import pad_asm_vmem

        pad_asm_vmem(kd)
    return kd


# ------------------------------------------------------------------ measured plan table

# (tp, row bucket of e1_plans.row_key) -> HeadKnobs overrides on WP-L2a's head pick ("e1": E1 fields). Only changes that
# WON on both rocke nodes (DESIGN 9.1 win rule, same-job controls) are entered; the ledger is WP-L2f.md.
HEAD_PLANS: dict = {
    # round 1 (base: WP-L2a pick + argmax): waves=8 at TP2 r5, 0.923 (n07-17) / 0.930 (n01-09)
    # round 2 (base: round 1): amrg=cta pairs, WIN on both nodes (n01-09 / n07-17)
    (1, 5): {"e1": {"map": "xcd"}, "amrg": "cta"},  # 0.966 / 0.970
    (1, 16): {"e1": {"xa": 2}, "amrg": "cta"},  # round 2 (n01-09) 0.934 / round 3 (n07-17, same base) 0.939
    (1, 128): {"e1": {"wcp": "nt"}, "amrg": "cta"},  # 0.934 / 0.944
    (2, 5): {"e1": {"waves": 8, "xu": True}, "amrg": "cta"},  # 0.945 / 0.943
    (2, 16): {"e1": {"bpre": 1}, "amrg": "cta"},  # 0.960 / 0.961
    (2, 128): {"e1": {"map": "xcd"}, "amrg": "cta"},  # round 4: 0.962 (n01-09) / 0.968 (n07-17)
}


def plan(tp: int, rows: int) -> HeadKnobs:
    """The MXFP8-activation head build for ``rows`` live tokens at ``tp``: WP-L2a's E1 pick of the bucket + the measured
    overrides. Fix round: on real text it FAILED the pre-registered head-precision gate against ATOM's served head (mean
    KL 0.098-0.099 B > 0.05 B, top-1 disagreement 0.90-0.91 % on gsm8k positions; WP-L2f.md "Fix round"), so the head
    of a pass that must match ATOM is :func:`plan_w4a16`; this one stays as a knob for an accuracy-budget decision."""
    from dataclasses import replace as _rep

    from . import e1_plans as EP

    e = EP.plan("head", tp, rows)
    ov = HEAD_PLANS.get((tp, EP.row_key(rows)), {})
    if ov.get("e1"):
        e = _rep(e, **ov["e1"])
    hk = HeadKnobs(e1=e, tp=tp, **{k: v for k, v in ov.items() if k != "e1"})
    hk.check()
    return hk


def w4a16_e1(e: "E1M.E1Knobs") -> "E1M.E1Knobs":
    """An mx head pick's E1 knobs for bf16 activations (W4A16 / the bf16-weight head): the same grid, waves, mt, depth,
    token blocking, map, cache policy and weight layout; X staged in LDS (no sop / chunk-major X scales)."""
    return _rep(e, w="bf16", sop=False, xsrc="lds", xslay="row", xscale="lds", xsb="shift", wsb="shift", act="fp8")


# (tp, row bucket) -> E1 overrides on the register-safe W4A16 / bf16-weight base; only both-node WINs on the same base
# (DESIGN 9.1 rule per job; ledger: WP-L2f.md "Fix round"). Fix round A (TP1: n01-09 job v2a3, n07-17 job r3b).
HEAD_W4_PLANS: dict = {
    (1, 32): {"sb": False},  # 0.968 / 0.956
    # fix round B (n01-09 job wBa, n07-17 job wBb; TP1 re-tested on the round-A base: nothing further)
    (2, 32): {"mt": 4},  # 0.899 / 0.906
    (2, 128): {"waves": 8},  # 0.763 / 0.757
}
HEAD_W16_PLANS: dict = {  # the bf16-weight reference head: plain loads (nt is the MXFP4 picks' policy) 0.63-0.78
    (1, 5): {"wcp": "default"},
    (1, 8): {"wcp": "default"},
    (1, 16): {"wcp": "default"},
    (1, 32): {"wcp": "default"},
    (1, 64): {"wcp": "default"},
    (1, 128): {"wcp": "default"},
    # fix round B (TP2): 0.638-0.809 on both nodes
    (2, 5): {"wcp": "default"},
    (2, 8): {"wcp": "default"},
    (2, 16): {"wcp": "default"},
    (2, 32): {"wcp": "default"},
    (2, 64): {"wcp": "default"},
    (2, 128): {"wcp": "default", "bpre": 1},  # fix round C' (on the round-B base): bpre=1 0.884 / 0.883
}


def plan_w4a16(tp: int, rows: int, w16: bool = False, tuned: bool = True) -> HeadKnobs:
    """The head with ATOM's activation precision (the fix round's default for an ATOM-matching pass: 100 % top-1
    agreement with ATOM's served head on 29,696 real positions): ``plan(tp, rows)`` with ``w4a16`` (MXFP4 weights
    decoded in registers), or with ``w16`` the bf16-weight head (E1's bf16 path over the dequantized MXFP4 weights: ATOM's head above 16 rows and
    FlyTape's weight bytes). Register-safe base (no spill, gfx950 COMGR): the bf16-weight head keeps at most 2 weight
    tiles per wave (4 x 16 bytes of each tile per chunk in flight; with 8 waves and 4+ token tiles at most 3 chunks in
    flight), the W4A16 head at most 2 from 2 token tiles on (the bf16 B operands; venv COMGR spilled mt 4 at 32 and 128
    rows, WP-L2f fix round job v1b); the bench's lever arms search around it."""
    p = plan(tp, rows)
    e = w4a16_e1(p.e1)
    if (w16 or e.tbt >= 2) and e.mt > 2:
        e = _rep(e, mt=2)
    if w16 and e.waves == 8 and e.tbt >= 4 and e.depth > 3:
        e = _rep(e, depth=3)
    from . import e1_plans as EP

    # tuned=False: the register-safe base alone (the logits twin of every W4A16 / w16 arm of the bucket: the K order of
    # each output's sum does not depend on waves / mt / depth, so their logits are bitwise equal)
    ov = (HEAD_W16_PLANS if w16 else HEAD_W4_PLANS).get((tp, EP.row_key(rows)), {}) if tuned else {}
    if ov:
        e = _rep(e, **ov)
    hk = _rep(p, e1=e, w4a16=not w16)
    hk.check()
    return hk
