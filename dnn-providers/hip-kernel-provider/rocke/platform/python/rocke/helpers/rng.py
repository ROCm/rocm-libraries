# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Philox4x32 RNG for FMHA dropout.

Philox4x32-10 is a counter-based RNG: each ``(seed, subseq, offset,
counter)`` quadruple deterministically produces four independent
uniform ``u32`` outputs after 10 rounds of a Feistel-like
permutation. Because there is no per-thread state, the same RNG
output is reproducible from the kernel arguments alone -- the FMHA
forward and backward passes can independently regenerate the same
dropout mask without explicit state hand-off.
"""

from __future__ import annotations

from typing import Tuple

from ..core.ir import IRBuilder, Value


__all__ = [
    "PHILOX_M0",
    "PHILOX_M1",
    "PHILOX_W0",
    "PHILOX_W1",
    "PHILOX_ROUNDS",
    "dropout_mask_pair_f32",
    "philox_u32_quartet",
    "philox_uniform_f32_quartet",
]


# Philox4x32-10 round constants -- match PyTorch + CK Tile bit-for-bit.
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


def _mulhilo32(b: IRBuilder, a: Value, c: int) -> Tuple[Value, Value]:
    """``(hi, lo) = a * c`` for i32 ``a`` and unsigned constant ``c``.

    Both halves lower to single AMDGPU instructions (``v_mul_hi_u32`` +
    ``v_mul_lo_u32``).
    """
    c_v = _u32_const(b, c)
    lo = b.mul(a, c_v)
    hi = b.umul_hi_i32(a, c_v)
    return hi, lo


def philox_u32_quartet(
    b: IRBuilder,
    *,
    seed_lo: Value,
    seed_hi: Value,
    subseq: Value,
    offset: Value,
) -> Tuple[Value, Value, Value, Value]:
    """One Philox4x32-10 round bank.

    Inputs: four i32 values forming the seed + counter.
    Outputs: four i32 uniform-random outputs.

    Pure function: bit-for-bit reproducible from the inputs.
    """
    c0, c1, c2, c3 = offset, b.const_i32(0), subseq, b.const_i32(0)
    k0, k1 = seed_lo, seed_hi

    for _ in range(PHILOX_ROUNDS):
        hi0, lo0 = _mulhilo32(b, c0, PHILOX_M0)
        hi1, lo1 = _mulhilo32(b, c2, PHILOX_M1)
        new_c0 = b.xor(b.xor(hi1, c1), k0)
        new_c1 = lo1
        new_c2 = b.xor(b.xor(hi0, c3), k1)
        new_c3 = lo0
        c0, c1, c2, c3 = new_c0, new_c1, new_c2, new_c3
        k0 = b.add(k0, _u32_const(b, PHILOX_W0))
        k1 = b.add(k1, _u32_const(b, PHILOX_W1))

    return c0, c1, c2, c3


def philox_uniform_f32_quartet(
    b: IRBuilder,
    *,
    seed_lo: Value,
    seed_hi: Value,
    subseq: Value,
    offset: Value,
) -> Tuple[Value, Value, Value, Value]:
    """Four uniform ``[0, 1)`` f32 values from one Philox quartet.

    Conversion is ``(u >> 8) * 2^-24`` -- 24 bits of randomness
    matching IEEE-754 single-precision significand.
    """
    r0, r1, r2, r3 = philox_u32_quartet(
        b, seed_lo=seed_lo, seed_hi=seed_hi, subseq=subseq, offset=offset
    )
    inv24 = b.const_f32(2.0**-24)
    shift8 = b.const_i32(8)

    def to_f32(u: Value) -> Value:
        shifted = b.lshr(u, shift8)
        return b.fmul(b.sitofp_f32(shifted), inv24)

    return to_f32(r0), to_f32(r1), to_f32(r2), to_f32(r3)


def dropout_mask_pair_f32(
    b: IRBuilder,
    *,
    seed_lo: Value,
    seed_hi: Value,
    subseq: Value,
    offset: Value,
    keep_prob_f32: Value,
) -> Tuple[Value, Value, Value, Value]:
    """Two ``(keep, scale)`` masks from one Philox quartet.

    Returns ``(keep0, scale, keep1, scale)``. The kernel applies the
    mask via one ``score = score * keep * scale`` per element.
    """
    u0, u1, _u2, _u3 = philox_uniform_f32_quartet(
        b, seed_lo=seed_lo, seed_hi=seed_hi, subseq=subseq, offset=offset
    )
    scale = b.rcp(keep_prob_f32)
    one_f = b.const_f32(1.0)
    zero_f = b.const_f32(0.0)
    keep0 = b.select(b.fcmp("olt", u0, keep_prob_f32), one_f, zero_f)
    keep1 = b.select(b.fcmp("olt", u1, keep_prob_f32), one_f, zero_f)
    return keep0, scale, keep1, scale
