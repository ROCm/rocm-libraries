# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""monofly's device ABI for rocKE stage functions linked into its gfx950 layer kernels.

* Mailboxes: ``(payload, tag)`` 8-byte pairs at device scope (cache bits sc1), polled until every tag matches.
* A region is passed as three values: its copy-0 address (``i64``), ``local`` (``i32``: this CTA's copy offset, 0 or
  ``HALF``; 0 for a single-copy region) and ``flags`` (``i32``: bit 0 dual copy, bit 1 spread over runs). The monofly
  side passes the flags as constants (they fold once the call is inlined) and ``local`` from its dual-copy hooks.
  A load reads ``place(o) + local``, a store writes ``place(o)`` and, for a dual region, ``place(o) + HALF``;
  ``place(o) = o + (o & RUN_MASK)`` for a spread region (monofly ``device/gfx950/dual.py``).
* Floating point with FlyDSL's semantics: ``fx.max`` / ``fx.min`` are ``llvm.maximum`` / ``llvm.minimum``
  (NaN-propagating, -0 < +0: rocKE's ``fmaximum`` / ``fminimum``); ``fmath.fma`` is ``llvm.fma`` (rocKE's
  ``fma_fused``; its ``fma`` is ``llvm.fmuladd``); ``exp`` is ``llvm.exp.f32``;
  ``/`` a plain ``fdiv``; bf16 rounding is ``fptrunc`` (RNE).
* Lane exchange: the value of lane ``lane ^ off`` (DPP within a 16-lane row: 1, 2, 8 one move, 4 = 3 then 7; the
  permlane swaps for 16 / 32 as monofly's ``xred``): the same values, so reductions in monofly's offset order give
  its bits.
"""

from __future__ import annotations

import struct

from rocke.core.ir import BF16, F32, I16, I32, I64, PtrType, VectorType

HALF = 8192
RUN_MASK = 0x3FFFE000
CM_DEV = 16
FP8_MAX = 448.0
ZERO_CODE = 127


def f32c(v: float) -> float:
    """The f32 value of a Python float (as FlyDSL's f32 constants hold it)."""
    return struct.unpack("<f", struct.pack("<f", v))[0]


class Region:
    """A monofly region passed by (address, local, flags). ``shared`` = (resource, base address): the region reached
    through one resource over its buffer (scratch), its byte offset in it as the instructions' scalar offset (one
    resource for every region instead of one each: SGPRs), as monofly's ``shared_mailbox`` does."""

    def __init__(self, e: "Emit", addr, local, flags, shared=None):
        self.e = e
        b = e.b
        if shared is None:
            self.rsrc, self.soff = e.rsrc(addr), e.c(0)
        else:
            self.rsrc, self.soff = shared[0], b.trunc(b.sub(addr, shared[1]), I32)
        self.local = local
        self.flags = flags

    def place(self, byte):
        b, c = self.e.b, self.e.c
        spread = b.cmp_ne(b.land(self.flags, c(2)), c(0))
        return b.select(spread, b.add(byte, b.land(byte, c(RUN_MASK))), byte)

    def load(self, byte, dwords, aux=CM_DEV):
        b = self.e.b
        return b.buffer_load_cp(self.rsrc, b.add(self.place(byte), self.local), self.soff, dwords=dwords, aux=aux)

    def load_off(self, byte):
        """The byte offset (from the resource's base) a load of logical ``byte`` uses (``tagged_poll`` specs)."""
        b = self.e.b
        return b.add(b.add(self.place(byte), self.local), self.soff)

    def store(self, data, byte, aux=CM_DEV):
        b, c = self.e.b, self.e.c
        o = self.place(byte)
        b.buffer_store_cp(data, self.rsrc, o, self.soff, aux=aux)
        with b.scf_if(b.cmp_ne(b.land(self.flags, c(1)), c(0))):
            b.buffer_store_cp(data, self.rsrc, b.add(o, c(HALF)), self.soff, aux=aux)

    def put(self, pair, words):
        """(payload, tag) pairs from pair ``pair``: ``words`` = [w0, tag] or [w0, tag, w1, tag]."""
        b = self.e.b
        self.store(b.vec_pack(words, I32), b.mul(pair, self.e.c(8)))


class Emit:
    """Builder helpers with monofly's semantics (module docstring)."""

    def __init__(self, b, pool_words: int, *, bar: str = "fence", pause: int = 0, poll_max: int = 12):
        self.b, self.bar, self.pause, self.poll_max = b, bar, pause, poll_max
        self.pool = b.smem_alloc(I32, [pool_words], name_hint="mono_lds")

    # ---- constants / resources
    def c(self, v):
        return self.b.const_i32(int(v))

    def f(self, v):
        return self.b.const_f32(f32c(v))

    def rsrc(self, addr, nbytes=0x7FFFFFF0, uniform=True):
        """A buffer resource over ``addr`` (``uniform``: the caller's value is wave-uniform after inlining, as monofly's
        args and region addresses are: no readfirstlane)."""
        b = self.b
        ptr = b.int_to_ptr(addr if uniform else b.readfirstlane(addr), PtrType(I32, "global"))
        return b.buffer_rsrc(ptr, self.c(nbytes))

    def barrier(self):
        if self.bar == "lds":  # LDS-only ordering (mailbox stores need no fence)
            self.b.sync_lds_only(fence="compiler")
        else:
            self.b.s_barrier_bare(fence="workgroup")  # FlyDSL gpu.barrier: fences + s_barrier

    def pause_text(self) -> str:
        return "s_nop 0" if self.pause == 0 else f"s_sleep {self.pause}"

    # ---- floats (FlyDSL semantics)
    def fmaximum(self, a, x):
        """``llvm.maximum.f32`` (FlyDSL ``fx.max``: NaN-propagating, max(-0, +0) = +0)."""
        return self.b.fmaximum(a, x)

    def fminimum(self, a, x):
        """``llvm.minimum.f32`` (FlyDSL ``fx.min``)."""
        return self.b.fminimum(a, x)

    def ffma(self, a, x, y):
        """a x + y with one rounding (``llvm.fma.f32``: FlyDSL ``fmath.fma``)."""
        return self.b.fma_fused(a, x, y)

    def bfr(self, v):
        return self.b.cast_to_f32(self.b.trunc_f32_to_bf16(v))

    def bf_pair(self, v0, v1):
        b = self.b
        lo = b.zext(b.bitcast(b.trunc_f32_to_bf16(v0), I16), I32)
        hi = b.zext(b.bitcast(b.trunc_f32_to_bf16(v1), I16), I32)
        return b.lor(lo, b.shl(hi, self.c(16)))

    def e8m0(self, amax):
        """monofly ``mx.e8m0_code``."""
        b, c = self.b, self.c
        bits = b.bitcast(amax, I32)
        ex = b.land(b.lshr(bits, c(23)), c(255))
        man = b.land(bits, c(0x7FFFFF))
        code = b.add(b.sub(ex, c(8)), b.select(b.cmp_gt(man, c(0x600000)), c(1), c(0)))
        code = b.smax(b.smin(code, c(254)), c(1))
        return b.select(b.fcmp("oeq", amax, b.const_f32(0.0)), c(ZERO_CODE), code)

    def quant_word(self, vals, code):
        """monofly ``mx.quant_word``: x 2^-(code - 127), clamped to +-448 (maximum / minimum), RNE E4M3."""
        b = self.b
        inv = b.bitcast(b.shl(b.sub(self.c(254), code), self.c(23)), F32)
        q = [self.fmaximum(self.fminimum(b.fmul(v, inv), self.f(FP8_MAX)), self.f(-FP8_MAX)) for v in vals]
        return b.bitcast(b.cvt_pk_fp8_f32x4(b.vec_pack(q, F32)), I32)

    # ---- lane exchange (values of lane ^ off)
    def xshfl_i(self, x, off):
        b = self.b
        if off in (1, 2, 3, 7, 8, 15):  # (gfx950 DPP: quad_perm 1-3, row_half_mirror 7, row_ror 8, row_mirror 15)
            return b.dpp_xor(x, off)
        if off == 4:  # lane ^ 4 = (lane ^ 3) ^ 7: two DPP moves (gfx950 has no row_xmask), as monofly's two
            return b.dpp_xor(b.dpp_xor(x, 3), 7)
        raise ValueError(off)

    def xred_f(self, v, off, op):
        """op(v, v of lane ^ off) (monofly ``xred``)."""
        b = self.b
        x = b.bitcast(v, I32)
        if off < 16:
            return op(v, b.bitcast(self.xshfl_i(x, off), F32))
        swap = b.permlane16_swap if off == 16 else b.permlane32_swap
        lo, hi = swap(x, x)
        return op(b.bitcast(lo, F32), b.bitcast(hi, F32))

    def bfly(self, vs, offsets, op):
        """monofly ``_bfly``: each of ``vs`` reduced over ``offsets`` in order, level by level."""
        for off in offsets:
            vs = [self.xred_f(v, off, op) for v in vs]
        return vs

    # ---- LDS (word offsets in the caller's arena)
    def ld(self, at):
        return self.b.vec_extract(self.b.smem_load_vN(self.pool, at, dtype=I32, n=1), 0)

    def ldf(self, at):
        return self.b.vec_extract(self.b.smem_load_vN(self.pool, at, dtype=F32, n=1), 0)

    def ld4(self, at):
        return self.b.smem_load_vN(self.pool, at, dtype=I32, n=4)

    def st(self, at, v, n=1):
        self.b.smem_store_vN(self.pool, [at], v, n)

    def stf(self, at, v):
        self.b.smem_store_vN(self.pool, [at], self.b.bitcast(v, I32), 1)

    def tr16(self, at):
        """``ds_read_b64_tr_b16`` at word ``at`` -> <2 x i32>."""
        b = self.b
        return b.vec_bitcast(b.ds_read_tr16_b64(self.pool, at, dtype=BF16), VectorType(I32, 2))

    # ---- mailboxes
    def poll(self, tag, specs):
        """[(Region, pair, npairs)] -> one ``<2 npairs x i32>`` per spec (payloads at even indices); ``poll_max``
        specs a round trip."""
        b, out = self.b, []
        for b0 in range(0, len(specs), self.poll_max):
            batch = specs[b0 : b0 + self.poll_max]
            out += b.tagged_poll(tag, [(reg.rsrc, reg.load_off(b.mul(pr, self.c(8))), 2 * n) for reg, pr, n in batch],
                                 aux=CM_DEV, pause=self.pause_text(), reload="launder")  # fmt: skip
        return out

    def i64(self, v):
        return self.b.sext(v, I64)
