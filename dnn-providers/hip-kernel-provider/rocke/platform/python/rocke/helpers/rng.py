# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Philox4x32-10 counter-based RNG in IR.

A (key, counter) pair gives four uniform ``u32`` words after 10 rounds, with no
per-thread state, so a kernel can regenerate any word from its arguments alone.

The IR's i32 ``add`` / ``mul`` carry ``nsw``, so a wrapping u32 sum or product
written with them is poison. Philox wraps on nearly every call, so its low
product and key schedule are built in i64 with operands small enough that no
step overflows, then truncated; LLVM narrows the sequence back to 32-bit ops.
"""

from __future__ import annotations

from typing import Tuple

from ..core.ir import I32, I64, IRBuilder, Value


__all__ = [
    "PHILOX_M0",
    "PHILOX_M1",
    "PHILOX_W0",
    "PHILOX_W1",
    "PHILOX_ROUNDS",
    "PhiloxCall",
    "add_u32_wrap",
    "mul_lo_u32_const",
    "mulhilo_u32_wide",
    "philox4x32_10",
    "philox_counter_add",
]


# Philox4x32-10 multipliers and key increments (Random123).
PHILOX_M0 = 0xD2511F53
PHILOX_M1 = 0xCD9E8D57
PHILOX_W0 = 0x9E3779B9
PHILOX_W1 = 0xBB67AE85
PHILOX_ROUNDS = 10


def _u32_const(b: IRBuilder, value: int) -> Value:
    """Materialise an i32 constant from an unsigned-32 value.

    Python ints can be arbitrarily large; the IR's ``const_i32`` accepts
    a signed-i32 view of the bit pattern. Force the two's-complement
    reinterpretation so the round constants land bit-correct.
    """
    masked = int(value) & 0xFFFFFFFF
    if masked >= 0x80000000:
        masked -= 0x100000000
    return b.const_i32(masked)


def add_u32_wrap(b: IRBuilder, a: Value, c: Value) -> Value:
    """``(a + c) mod 2**32`` of two i32 values read as u32, free of ``nsw`` poison.

    The sum of two zero-extended words is below ``2**33``, so the i64 ``add nsw``
    cannot overflow; the truncation drops the carry.
    """
    s = b.add(b.zext(a, I64), b.zext(c, I64))
    return b.trunc(s, I32)


def mul_lo_u32_const(b: IRBuilder, a: Value, c: int) -> Value:
    """Low 32 bits of ``a * c`` for i32 ``a`` read as u32 and a u32 constant ``c``.

    Split ``c`` into 16-bit halves: ``a * c_lo`` and ``a * c_hi`` stay below
    ``2**48`` in i64, and only the low 16 bits of the second product reach the
    low word. No step overflows, so no ``nsw`` poison.
    """
    c &= 0xFFFFFFFF
    a64 = b.zext(a, I64)
    p_lo = b.mul(a64, b.const_i64(c & 0xFFFF))
    p_hi = b.mul(a64, b.const_i64(c >> 16))
    p_hi16 = b.shl(b.land(p_hi, b.const_i64(0xFFFF)), b.const_i64(16))
    return b.trunc(b.add(p_lo, p_hi16), I32)


def _mulhilo32(b: IRBuilder, a: Value, c: int) -> Tuple[Value, Value]:
    """``(hi, lo) = a * c`` for i32 ``a`` and unsigned constant ``c``.

    ``umul_hi_i32`` lowers to a flag-free i64 multiply (``v_mul_hi_u32``);
    the low word is :func:`mul_lo_u32_const` (``v_mul_lo_u32`` after LLVM
    narrows it).
    """
    hi = b.umul_hi_i32(a, _u32_const(b, c))
    lo = mul_lo_u32_const(b, a, c)
    return hi, lo


def mulhilo_u32_wide(b: IRBuilder, a: Value, c: int) -> Tuple[Value, Value]:
    """``(hi, lo)`` of ``a * c`` (u32 ``a``, u32 constant ``c``) from one u64 product.

    With both halves cut from the same i64 product LLVM selects one
    ``v_mad_u64_u32`` instead of ``v_mul_hi_u32`` + ``v_mul_lo_u32``. The i64
    ``mul`` carries ``nsw``, so the product is built from parts that cannot
    overflow: ``a * (c >> 1)`` stays below ``2**63``, doubled by a shift and
    completed by ``add nuw`` (below ``2**64``); LLVM folds it back to one multiply.
    """
    c &= 0xFFFFFFFF
    a64 = b.zext(a, I64)
    prod = b.shl(b.mul(a64, b.const_i64(c >> 1)), b.const_i64(1))
    if c & 1:
        prod = b.add_nuw(prod, a64)
    return b.trunc(b.lshr(prod, b.const_i64(32)), I32), b.trunc(prod, I32)


def philox4x32_10(
    b: IRBuilder,
    c0: Value,
    c1: Value,
    c2: Value,
    c3: Value,
    k0: Value,
    k1: Value,
) -> Tuple[Value, Value, Value, Value]:
    """Philox4x32-10 of counter words ``c0..c3`` and key words ``k0, k1`` (all i32 as u32).

    Bit-exact against ``rocke.numeric.philox_dropout.philox4x32_10``.
    """
    for _ in range(PHILOX_ROUNDS):
        hi0, lo0 = _mulhilo32(b, c0, PHILOX_M0)
        hi1, lo1 = _mulhilo32(b, c2, PHILOX_M1)
        c0, c1, c2, c3 = (
            b.xor(b.xor(hi1, c1), k0),
            lo1,
            b.xor(b.xor(hi0, c3), k1),
            lo0,
        )
        k0 = add_u32_wrap(b, k0, _u32_const(b, PHILOX_W0))
        k1 = add_u32_wrap(b, k1, _u32_const(b, PHILOX_W1))
    return c0, c1, c2, c3


def philox_counter_add(
    b: IRBuilder, base_lo: Value, base_hi: Value, k: Value
) -> Tuple[Value, Value]:
    """Words ``(lo, hi)`` of the u64 counter ``base + k`` (``k`` an i32 read as u32), wrapping mod ``2**64``.

    Per-element counters differ from a hoisted row counter by ``kv // 4``; the
    carry into the high word is ``lo < k``.
    """
    lo = add_u32_wrap(b, base_lo, k)
    carry = b.zext(b.cmp_ult(lo, k), I32)
    return lo, add_u32_wrap(b, base_hi, carry)


class PhiloxCall:
    """One Philox4x32-10 call of the counter ``(ctr_lo, ctr_hi, 0, 0)``, emitted a
    round at a time.

    A body can spread a tile's calls over its MFMA sequence (``step``) and take
    the words once all rounds are emitted (``finish``); the result equals
    :func:`philox4x32_10` of the counter ``(ctr_lo, ctr_hi, 0, 0)``.

    ``xor3`` (targets with ``v_bitop3_b32``): each round's two ``hi ^ c ^ k``
    become one ``v_bitop3_b32`` (LLVM forms none from the xor pair). ``mad64``:
    each product's two halves come from one u64 product (one ``v_mad_u64_u32``).
    Counter words 2 and 3 start at zero, and the round-0 products of word 2 stay
    zero: those words are tracked here so no multiply or xor is emitted for them.
    """

    def __init__(
        self, b: IRBuilder, *, seed_lo, seed_hi, ctr_lo, ctr_hi, xor3=False, mad64=False
    ):
        self.b = b
        self.xor3 = xor3
        self.mad64 = mad64
        self.c = (ctr_lo, ctr_hi, None, None)  # None = known zero
        self.k = (seed_lo, seed_hi)
        self.rounds = 0

    @property
    def done(self) -> bool:
        return self.rounds == PHILOX_ROUNDS

    def _mulhilo(self, a, m):
        if a is None:
            return None, None
        if not self.mad64:
            return _mulhilo32(self.b, a, m)
        return mulhilo_u32_wide(self.b, a, m)

    def _xor(self, *vals):
        vals = [v for v in vals if v is not None]
        b = self.b
        if len(vals) == 3 and self.xor3:
            return b.bitop3_b32(*vals, 0x96)
        out = vals[0]
        for v in vals[1:]:
            out = b.xor(out, v)
        return out

    def step(self) -> None:
        """Emit the next round (no-op once all rounds are emitted)."""
        if self.done:
            return
        b = self.b
        c0, c1, c2, c3 = self.c
        k0, k1 = self.k
        hi0, lo0 = self._mulhilo(c0, PHILOX_M0)
        hi1, lo1 = self._mulhilo(c2, PHILOX_M1)
        self.c = (self._xor(hi1, c1, k0), lo1, self._xor(hi0, c3, k1), lo0)
        self.k = (
            add_u32_wrap(b, k0, _u32_const(b, PHILOX_W0)),
            add_u32_wrap(b, k1, _u32_const(b, PHILOX_W1)),
        )
        self.rounds += 1

    def finish(self) -> Tuple[Value, Value, Value, Value]:
        """Emit the remaining rounds; the four output words."""
        while not self.done:
            self.step()
        zero = self.b.const_i32(0)
        return tuple(zero if c is None else c for c in self.c)
