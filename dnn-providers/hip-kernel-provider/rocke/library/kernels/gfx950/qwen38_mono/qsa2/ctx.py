# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Per-stage context of a QSA stage function: thread ids, request rows, the
mono's mailboxes, LDS arena and caches, and the FlyDSL mono's device helpers
(``device_ops``, ``stages/common.py``, ``hc.py``, ``moe.mexp_fma``, ``p2``),
each emitting the same LLVM operations in the same order as the mono's, so
that a function linked into the mono kernel and compiled by the same LLVM
gives the same bits.

Exact correspondences (FlyDSL -> here):

- ``xshfl(v, off)``: ``update.dpp`` with the mono's controls and row / bank
  masks for ``off`` < 16; ``ds_bpermute`` (``shuffle_xor``) above.
- ``xred(v, off, op)``: ``op(v, xshfl(v, off))`` below 16, else
  ``op(lhs, rhs)`` of ``permlane{16,32}_swap(v, v)``.
- ``fx.max`` on f32: ``llvm.maximum.f32``; ``fma``: ``llvm.fma.f32``;
  ``_fma64``: ``llvm.fma.f64`` (``lower.QsaLowerer``).
- ``mexp_fma``: ``exp2_hw`` of ``x log2 e``, below -126 of ``fma(x, log2 e,
  64)`` scaled by 2^-64; ``sigmoid_acc``: ``1 / (1 + exp(0 - x))``.
- ``mbx.put`` / ``put2`` / ``poll`` / ``pre_poll``: ``rocke.helpers.mailbox``
  (same pairs, tags and cache bits; the poll loop is the rocKE one).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from rocke.core.ir import BF16, F32, F64, I32, I64, IRBuilder, PtrType, Value, VectorType
from rocke.helpers import device_ops as D
from rocke.helpers import mailbox as M

from . import lower as X

SIGN = -(2**31)
NEG_MAX = -3.4028234663852886e38
SCALE_BITS = 0x3D800000  # 1/16
R128 = 0x3C000000
R256 = 0x3B800000
SQRT128 = 0x413504F3
EPS = 0x358637BD
L2E = 1.4426950408889634
TWO_M64 = 2.0**-64
RATIO_BITS = 0x3B124925  # fl(1/448)
CM_SC1 = 16
CPOL_NT = 2
# ATOM's NUMA area (``kernels/numa.py``; ``moe_fused``'s constants): the header past the mailbox part and the
# stamp / blame / spin regions, the area at the next 2 MiB boundary past header + 256; byte x of the mailbox part
# at x + (x & ~8191) (copy 0) and + 8192 (copy 1).
NUMA_PAGE = 1 << 21
NUMA_CHUNK = 8192


def _numa_hdr_default():
    from .. import moe_fused

    return moe_fused.NUMA_HDR


try:  # noqa: SIM105
    NUMA_HDR_DEFAULT = _numa_hdr_default()
except Exception:  # pragma: no cover - moe_fused needs its own imports
    NUMA_HDR_DEFAULT = 0


@dataclass(frozen=True)
class _Rg:
    """A mailbox region: static byte offset in the scratch's mailbox part."""

    off: int

# FlyDSL ``device_ops.xshfl`` DPP controls: off -> [(ctrl, row_mask, bank_mask)]
# (two steps for 8 and 4: the second's ``old`` is the first's result)
_XSHFL = {
    8: ((0x118, 0xF, 0xC), (0x108, 0xF, 0x3)),
    4: ((0x114, 0xF, 0xA), (0x104, 0xF, 0x5)),
    2: ((0x4E, 0xF, 0xF),),
    1: ((0xB1, 0xF, 0xF),),
}


@dataclass
class QCtx:
    b: IRBuilder
    a: object  # stage_fn ABI namespace
    lay: object  # layout.QsaLayout
    k: object  # knobs.QsaKnobs
    stage: str

    def __post_init__(self):
        b = self.b
        self.rows, self.seq = self.lay.rows, self.lay.seq
        self.tid = self.a.tid
        self.lane = self.a.lane
        self.wave = self.a.wave
        self.tag = self.a.tag
        if self.k.mailbox_impl in ("ir", "irf"):
            from .. import mailbox_ir  # noqa: F401  (registers ir / irf)
        self.fly = self.k.mailbox_impl == "fly"
        self.mbk = M.MailboxKnobs(
            impl="asm" if self.fly else self.k.mailbox_impl, pause=self.k.pause, poll_max=12,
            put_nt=self.k.put_nt,
        )
        self._pool = None
        self._mbs = {}
        self.ls = self.a.ls[: (self.rows if self.lay.seq or self.rows > 1 else 1)]
        self._zero = b.const_i32(0)
        # the mailbox base(s) at the function entry: a value made inside a
        # runtime branch would not dominate its uses in another
        self._base()

    # ------------------------------------------------------------ ints
    def c(self, v: int) -> Value:
        return self.b.const_i32(v)

    def f(self, v: float) -> Value:
        return self.b.const_f32(v)

    def fbits(self, bits: int) -> Value:
        return self.b.bitcast(self.b.const_i32(bits), F32)

    def _v(self, x):
        return self.c(x) if isinstance(x, int) else x

    def add(self, *xs):
        b = self.b
        out = self._v(xs[0])
        for x in xs[1:]:
            out = b.add(out, self._v(x))
        return out

    def mul(self, x, y):
        return self.b.mul(self._v(x), self._v(y))

    def lt(self, x, y):
        return self.b.cmp_lt(self._v(x), self._v(y))

    def le(self, x, y):
        return self.b.cmp_le(self._v(x), self._v(y))

    def gt(self, x, y):
        return self.b.cmp_gt(self._v(x), self._v(y))

    def ge(self, x, y):
        return self.b.cmp_ge(self._v(x), self._v(y))

    def eq(self, x, y):
        return self.b.cmp_eq(self._v(x), self._v(y))

    def ne(self, x, y):
        return self.b.cmp_ne(self._v(x), self._v(y))

    def sel(self, cond, x, y):
        return self.b.select(cond, self._v(x), self._v(y))

    def smin(self, x, y):
        return self.b.smin(self._v(x), self._v(y))

    def smax(self, x, y):
        return self.b.smax(self._v(x), self._v(y))

    def div(self, x, d: int):
        """``x // d`` for ``x`` >= 0 (every use here)."""
        if d & (d - 1) == 0:
            return self.b.lshr(self._v(x), self.c(d.bit_length() - 1))
        return self.b.div(self._v(x), self.c(d))

    def mod(self, x, d: int):
        if d & (d - 1) == 0:
            return self.b.land(self._v(x), self.c(d - 1))
        return self.b.mod(self._v(x), self.c(d))

    def land(self, x, y):
        return self.b.land(self._v(x), self._v(y))

    def lor(self, x, y):
        return self.b.lor(self._v(x), self._v(y))

    def band(self, x, y):
        """i1 and."""
        return self.b.land(x, y)

    def uni(self, v):
        return self.b.readfirstlane(v)

    # ------------------------------------------------------------ rows
    def row_len(self, r):
        """Request row ``r``'s ``L`` (``qsa.row_len``)."""
        ls = self.ls
        if isinstance(r, int):
            return ls[r]
        L = ls[0]
        for q in range(1, len(ls)):
            L = self.b.select(self.eq(r, q), ls[q], L)
        return L

    def request(self, t, per: int):
        """``qsa._request``: (request row, its task)."""
        if self.rows == 1:
            return 0, t
        return self.smin(self.div(t, per), self.rows - 1), self.mod(t, per)

    def mb_ix(self, i, r):
        if isinstance(r, int) and r == 0:
            return self._v(i)
        return self.add(i, self.mul(r, self.lay.row_pairs))

    def list_len(self, L):
        """(cb, n, ak)."""
        n1 = self.add(L, 1)
        cb = self.div(n1, 4)
        n = self.smin(cb, self.lay.sel)
        ak = self.add(self.mul(n, 4), self.mod(n1, 4))
        return cb, n, ak

    # ------------------------------------------------------------ state table
    def _entry(self, i):
        """Entry ``i`` of the ``qsa`` table (wave-uniform i64): ``qsa._entry``,
        a 2-dword vector load and ``readfirstlane`` of each word."""
        b = self.b
        rs = self.rsrc(self.a.qsa)
        off = self.mul(self._v(i), 8)
        v = b.buffer_load_cp(rs, off, self._zero, dwords=2, aux=0)
        lo = self.uni(b.vec_extract(v, 0))
        hi = self.uni(b.vec_extract(v, 1))
        return b.lor(b.zext(lo, I64), b.shl(b.zext(hi, I64), b.const_i64(32)))

    def qsa_entry(self, r, name):
        """``qsa.qsa_row`` (row 0's from ``qsa_state``)."""
        idx = self.lay.state_index(name)
        if self.seq or (isinstance(r, int) and r == 0):
            return self._entry(idx)
        return self._entry(self.add(self.mul(r, len(self.lay.state)), idx))

    # ------------------------------------------------------------ buffers
    def rsrc(self, addr64):
        b = self.b
        return b.buffer_rsrc(b.int_to_ptr(addr64, PtrType(I32, "global")), self.c(-1))

    def load(self, rs, word, n=1, aux=0):
        """``bo.buffer_load(rs, word, vec_width=n)``: word offset (i32 units)."""
        v = self.b.buffer_load_cp(rs, self.mul(word, 4), self._zero, dwords=n, aux=aux)
        return v

    def load_words(self, rs, word, n, aux=0):
        v = self.load(rs, word, n, aux)
        if n == 1:
            return [v]
        return [self.b.vec_extract(v, e) for e in range(n)]

    def store(self, rs, word, words, aux=0):
        b = self.b
        data = words[0] if len(words) == 1 else b.vec_pack(list(words), I32)
        b.buffer_store_cp(data, rs, self.mul(word, 4), self._zero, aux=aux)

    def sload(self, addr64, word):
        """A scalar load of i32 ``word`` (``bo.buffer_load(..., is_scalar=True)``):
        ``s_load`` through a constant-space pointer (``tbl=smem`` of the q2
        port; the mono uses ``s_buffer_load``)."""
        b = self.b
        if self.k.tbl == "vmem":
            v = self.load(self.rsrc(addr64), word, 1)
            return self.uni(v)
        p = b.int_to_ptr(addr64, PtrType(I32, "constant"))
        return self.uni(b.global_load_i32(p, self._v(word)))

    def page_block(self, a_pages, blk, scalar=False):
        """``qsa.page_block``: ``pages[b / 4] * 4 + b % 4``."""
        pb = self.lay.page_blocks
        q = self.div(blk, pb)
        if scalar:
            page = self.sload(a_pages, q)
        else:
            page = self.load(self.rsrc(a_pages), q, 1)
        return self.add(self.mul(page, pb), self.mod(blk, pb))

    # ------------------------------------------------------------ mailboxes
    def mb(self, region: str, half: bool = False, fsc: str | None = None):
        """A mailbox region: its static byte offset in the scratch's mailbox
        part (``half``: the second half, Q3 / Q4's ``isc``; ``fsc``: a
        ``qsa_fast`` mailbox there). Pairs are addressed through one base
        (the scratch, or with ``numa=1`` ATOM's NUMA area)."""
        off = self.lay.sc[region]
        if half or fsc:
            off += self.lay.score_shift
        if fsc:
            off += self.lay.fsc[fsc] - self.lay.sc["isc"]
        return _Rg(off)

    def _base(self):
        b = self.b
        if "plain" not in self._mbs:
            self._mbs["plain"] = M.mailbox_from_i64(b, self.a.scratch, self.c(-1), self.mbk)
            if self.fly:
                self._mbs["rs_plain"] = self.rsrc(self.a.scratch)
            if self.k.numa:
                hdr = self.k.numa_hdr or NUMA_HDR_DEFAULT
                a_ = b.add(self.a.scratch, b.const_i64(hdr + 256 + NUMA_PAGE - 1))
                area = b.land(a_, b.const_i64(-NUMA_PAGE))
                self._mbs["area"] = M.mailbox_from_i64(b, area, self.c(-1), self.mbk)
                if self.fly:
                    self._mbs["rs_area"] = self.rsrc(area)
                hp = b.int_to_ptr(b.add(self.a.scratch, b.const_i64(hdr)), PtrType(I32, "constant"))
                mask = self.uni(b.global_load_i32(hp, self.c(0)))
                self._sel_mask = self.uni(b.xor(mask, b.sub(self.c(0), M.iod_id(b))))
        return self._mbs["plain"]

    def _pair(self, rg, i):
        """Logical pair index (from the scratch base) of pair ``i`` of ``rg``."""
        return self.add(i, rg.off // 8) if not isinstance(i, int) else self.c(rg.off // 8 + i)

    def _copy0(self, p):
        """ATOM ``numa.phys``: area pair index of copy 0 of logical pair ``p``."""
        b = self.b
        x_ = b.mul(p, self.c(8))
        return b.lshr(b.add(x_, b.land(x_, self.c(-NUMA_CHUNK))), self.c(3))

    def _local(self, p):
        """Area pair index of the copy of logical pair ``p`` homed at this CTA's IOD."""
        b = self.b
        x_ = b.mul(p, self.c(8))
        xi = b.add(x_, b.land(x_, self.c(-NUMA_CHUNK)))
        sel = b.land(b.lshr(self._sel_mask, b.lshr(xi, self.c(21))), self.c(1))
        return b.lshr(b.add(xi, b.shl(sel, self.c(13))), self.c(3))

    def _fly_store(self, p, words):
        """FlyDSL ``mbx.put`` / ``put2``: one buffer store of the pair(s) with sc1 (``numa``: both copies)."""
        b = self.b
        data = b.vec_pack([b.bitcast(w, I32) if w.type == F32 else w for w in words], I32)
        aux = CM_SC1 | (CPOL_NT if self.k.put_nt else 0)
        if self.k.numa:
            off = b.mul(self._copy0(p), self.c(8))
            rs = self._mbs["rs_area"]
            b.buffer_store_cp(data, rs, off, self._zero, aux=aux)
            b.buffer_store_cp(data, rs, b.add(off, self.c(NUMA_CHUNK)), self._zero, aux=aux)
        else:
            b.buffer_store_cp(data, self._mbs["rs_plain"], b.mul(p, self.c(8)), self._zero, aux=aux)

    def put(self, rg, i, v):
        base = self._base()
        p = self._pair(rg, i)
        if self.fly:
            return self._fly_store(p, [v, self.tag])
        if self.k.numa:
            M.put(self.b, self._mbs["area"], self._copy0(p), v, self.tag, replace(self.mbk, numa_copies=2))
        else:
            M.put(self.b, base, p, v, self.tag, self.mbk)

    def put2(self, rg, i, v0, v1):
        base = self._base()
        p = self._pair(rg, i)
        if self.fly:
            return self._fly_store(p, [v0, self.tag, v1, self.tag])
        if self.k.numa:
            M.put2(self.b, self._mbs["area"], self._copy0(p), v0, v1, self.tag, replace(self.mbk, numa_copies=2))
        else:
            M.put2(self.b, base, p, v0, v1, self.tag, self.mbk)

    def _specs(self, specs):
        base = self._base()
        out = []
        for rg, i, n in specs:
            p = self._pair(rg, i)
            out.append((self._mbs["area"], self._local(p), n) if self.k.numa else (base, p, n))
        return out

    def poll(self, specs):
        if not self.fly:
            return M.poll(self.b, self._specs(specs), self.tag, self.mbk)
        # FlyDSL ``mbx.poll``: batches of POLL_MAX (12) specs, each its own spin loop
        b = self.b
        self._base()
        out = []
        for first in range(0, len(specs), 12):
            fs = []
            for rg, i, n in specs[first : first + 12]:
                p = self._pair(rg, i)
                if self.k.numa:
                    fs.append((self._mbs["rs_area"], b.mul(self._local(p), self.c(8)), 2 * n))
                else:
                    fs.append((self._mbs["rs_plain"], b.mul(p, self.c(8)), 2 * n))
            from rocke.helpers import device_ops as _D

            vecs = X.poll_fly(b, self.tag, fs, aux=CM_SC1, pause=_D.PAUSES[self.k.pause] or "")
            for v, (_rg, _i, n) in zip(vecs, specs[first : first + 12]):
                out.append([b.vec_extract(v, 2 * q) for q in range(n)])
        return out

    def pre_poll(self, hint, n: int):
        """``mailbox.pre_poll``: wave 0 polls one hint pair per producer, then
        the CTA barrier (``pbar``: ``workgroup`` = the mono's fenced
        ``gpu.barrier``, ``compiler`` = no fence)."""
        b = self.b
        assert 0 < n <= 64 * 12
        if self.stage in self.k.prepoll_skip:
            # knob prepoll: no hint throttle; the payload poll spins itself
            # (one round trip after the producers instead of two)
            self.barrier(self.k.pbar)
            return
        with b.scf_if(self.eq(self.wave, 0)):
            specs = []
            for r in range((n + 63) // 64):
                m = self.smin(self.add(self.lane, 64 * r), n - 1)
                rg, i = hint(m)
                specs.append((rg, i, 1))
            self.poll(specs)
        self.barrier(self.k.pbar)

    def barrier(self, fence=None):
        """``gpu.barrier()``: fence release, ``s_barrier``, fence acquire."""
        self.b.s_barrier_bare(fence=fence or self.k.bar)

    # ------------------------------------------------------------ LDS
    def pool(self):
        if self._pool is None:
            self._pool = self.b.smem_alloc(I32, [self.lay.lds_words], name_hint="qsa")
        return self._pool

    def lw(self, name: str, idx=0):
        """Word index of ``name[idx]`` in the mono's arena."""
        base = self.lay.lds[name]
        if isinstance(idx, int):
            return self.c(base + idx)
        return self.add(idx, base)

    def ld(self, word) -> Value:
        b = self.b
        return b.vec_extract(b.smem_load_vN(self.pool(), self._v(word), dtype=I32, n=1), 0)

    def ldf(self, word) -> Value:
        return self.b.bitcast(self.ld(word), F32)

    def ld4(self, word) -> list:
        b = self.b
        v = b.smem_load_vN(self.pool(), self._v(word), dtype=I32, n=4)
        return [b.vec_extract(v, e) for e in range(4)]

    def st(self, word, v: Value) -> None:
        if v.type == F32:
            v = self.b.bitcast(v, I32)
        self.b.smem_store_vN(self.pool(), [self._v(word)], v, 1)

    def stv(self, word, vals) -> None:
        b = self.b
        ws = [b.bitcast(v, I32) if v.type == F32 else v for v in vals]
        b.smem_store_vN(self.pool(), [self._v(word)], b.vec_pack(ws, I32), len(ws))

    def atomic_add(self, word, v) -> None:
        self.b.lds_atomic_add(self.pool(), [self._v(word)], self._v(v))

    # ------------------------------------------------------------ math
    def bf16r(self, x):
        b = self.b
        return b.cast_to_f32(b.trunc_f32_to_bf16(x))

    def bf2(self, w):
        return M.bf2_f32(self.b, w)

    def unpack(self, words):
        out = []
        for w in words:
            out += list(self.bf2(w))
        return out

    def bf16x8(self, words):
        return self.unpack(words)

    def bf16_pair(self, x, y):
        """``mbx.bf16_pair``: ``fptrunc <2 x float>`` to bf16, as an i32."""
        b = self.b
        v = b.vec_trunc_f32_to_bf16(b.vec_pack([x, y], F32))
        return b.vec_extract(b.vec_bitcast(v, VectorType(I32, 1)), 0)

    def fma(self, x, y, z):
        return self.b.fma(x, y, z)

    def fmax(self, x, y):
        return self.b.fmax(x, y)

    def exp2_hw(self, x):
        return self.b.exp2_fast(x)

    def mexp_fma(self, x):
        b = self.b
        p = b.fmul(x, self.f(L2E))
        small = b.fmul(self.exp2_hw(b.fma(x, self.f(L2E), self.f(64.0))), self.f(TWO_M64))
        return b.select(b.fcmp("olt", p, self.f(-126.0)), small, self.exp2_hw(p))

    def sigmoid_acc(self, x):
        b = self.b
        return b.fdiv(self.f(1.0), b.fadd(self.f(1.0), b.exp(b.fsub(self.f(0.0), x))))

    # ------------------------------------------------------------ lanes
    def xshfl(self, v, off: int):
        b = self.b
        is_f = v.type == F32
        if off >= 16:
            x = b.bitcast(v, I32) if is_f else v
            r = b.ds_bpermute(b.shl(b.xor(self.lane, self.c(off)), self.c(2)), x)
            return b.bitcast(r, F32) if is_f else r
        src = b.bitcast(v, I32) if is_f else v
        steps = _XSHFL[off]
        r = X.update_dpp(b, src, src, *steps[0])
        for ctrl, rm, bm in steps[1:]:
            r = X.update_dpp(b, r, src, ctrl, rm, bm)
        return b.bitcast(r, F32) if is_f else r

    def swap(self, x, y, off: int):
        """``v_permlane{32,16}_swap(x, y)``: (struct field 0, field 1)."""
        fn = self.b.permlane32_swap if off == 32 else self.b.permlane16_swap
        return fn(x, y)

    def xred(self, v, off: int, op):
        b = self.b
        if off < 16:
            return op(v, self.xshfl(v, off))
        is_f = v.type == F32
        s = b.bitcast(v, I32) if is_f else v
        lo, hi = self.swap(s, s, off)
        if is_f:
            lo, hi = b.bitcast(lo, F32), b.bitcast(hi, F32)
        return op(lo, hi)

    def wave_max(self, v):
        for off in (32, 16, 8, 4, 2, 1):
            v = self.xred(v, off, self.fmax)
        return v

    def wave_sum(self, v):
        for off in (32, 16, 8, 4, 2, 1):
            v = self.xred(v, off, self.b.fadd)
        return v

    def natural_sum(self, v):
        for off in (1, 2, 4, 8, 16, 32):
            v = self.xred(v, off, self.b.fadd)
        return v

    # ------------------------------------------------------------ MXFP8
    def e8m0_fp8(self, amax):
        b = self.b
        bits = b.bitcast(b.fmul(amax, self.fbits(RATIO_BITS)), I32)
        e = b.land(b.lshr(bits, self.c(23)), self.c(0xFF))
        e = b.add(e, b.select(self.ne(b.land(bits, self.c(0x7FFFFF)), 0), self.c(1), self.c(0)))
        e = b.smin(b.smax(e, self.c(1)), self.c(254))
        return e, b.bitcast(b.shl(e, self.c(23)), F32)

    def fp8_word(self, v0, v1, v2, v3, scale):
        b = self.b
        w = b.cvt_scalef32_pk_fp8_f32(self.c(0), v0, v1, scale, hi=False)
        return b.cvt_scalef32_pk_fp8_f32(w, v2, v3, scale, hi=True)


__all__ = ["QCtx", "SIGN", "NEG_MAX", "SCALE_BITS", "R128", "SQRT128", "BF16", "F64"]
