# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Magic-number division: an unsigned integer divide as mul-hi, add, shift.

The AMDGPU backend has no integer-division instruction, so a divide by a
divisor known on the host (at build time or at launch) is emitted as
``(umul_hi(n, multiplier) + n) >> shift`` with a ``(multiplier, shift)`` pair
computed on the host. This mirrors CK Tile's
``ck_tile/core/utility/magic_div.hpp``.

- :func:`calculate_magic_numbers` computes the pair on the host.
- :func:`do_magic_division` emits the sequence with the pair baked in as
  constants.
- :func:`do_magic_division_dynamic` emits it with the pair as runtime values
  (for example kernel arguments the launcher fills from
  :func:`calculate_magic_numbers`).

The coordinate-transform DAG in ``rocke.helpers.transforms`` is one caller;
kernels that split a flat index call it directly.
"""

from __future__ import annotations

from typing import Tuple

from ..core.ir import IRBuilder, Value


def calculate_magic_numbers(divisor: int) -> Tuple[int, int]:
    """Compute the ``(multiplier, shift)`` pair for unsigned magic division.

    Direct port of CK Tile's
    ``magic_division32_bit_range::calculate_magic_numbers`` (the
    ``magic_division`` alias used by ``merge_v2_magic_division``). The
    returned constants reproduce ``dividend // divisor`` via the mul-hi
    sequence in :func:`do_magic_division` for any ``dividend`` that fits
    in the 31-bit unsigned range (the same restriction CK Tile documents).

    The math:

    .. code-block:: text

        shift      = ceil(log2(divisor))           (smallest s with 2**s >= divisor)
        multiplier = ((2**shift - divisor) << 32) // divisor + 1

    Both values are plain Python ``int`` (compile-time constants); they are
    baked into the emitted IR as ``const_i32`` operands, exactly like the
    C++ constexpr path.
    """
    if divisor < 1:
        raise ValueError(f"magic division requires divisor >= 1, got {divisor}")
    shift = 0
    while (1 << shift) < divisor:
        shift += 1
    multiplier = (((1 << shift) - divisor) << 32) // divisor + 1
    return multiplier, shift


def do_magic_division(
    b: IRBuilder, dividend: Value, multiplier: int, shift: int
) -> Value:
    """Emit ``dividend // divisor`` using the magic ``(multiplier, shift)``.

    Mirrors CK Tile's ``do_magic_division`` device path
    (``tmp = __umulhi(dividend, multiplier); (tmp + dividend) >> shift``),
    which the AMDGPU backend lowers without an integer-division instruction.
    ``dividend`` is an i32 SSA value interpreted as unsigned; the result is
    the i32 quotient. ``multiplier`` / ``shift`` come from
    :func:`calculate_magic_numbers`.

    The sum ``tmp + dividend`` is emitted as a no-unsigned-wrap add: for a
    dividend near 2**31 it passes the signed maximum (from 1,610,612,736 for
    divisor 3) while staying below 2**32, so a no-signed-wrap add would make
    the quotient poison inside the documented range.
    """
    # ``multiplier`` is a uint32 that may exceed the i32 signed range; bake
    # it into the i32 constant as its two's-complement bit pattern so the
    # unsigned ``umul_hi`` sees the right bits (matches the C++ uint32 path).
    mult_i32 = multiplier - (1 << 32) if multiplier >= (1 << 31) else multiplier
    tmp = b.umul_hi_i32(dividend, b.const_i32(mult_i32))
    summed = b.add_nuw(tmp, dividend)
    if shift == 0:
        return summed
    return b.lshr(summed, b.const_i32(shift))


def do_magic_division_dynamic(
    b: IRBuilder, dividend: Value, multiplier: Value, shift: Value
) -> Value:
    """Like :func:`do_magic_division` but ``multiplier`` and ``shift`` are
    runtime SSA ``Value`` objects (e.g. kernel arguments) instead of Python
    ``int``. Emits the same ``umul_hi + add + lshr`` sequence; ``lshr`` is
    always emitted (shift == 0 produces a no-op right-shift). The sum is a
    no-unsigned-wrap add, as in :func:`do_magic_division`.
    """
    tmp = b.umul_hi_i32(dividend, multiplier)
    summed = b.add_nuw(tmp, dividend)
    return b.lshr(summed, shift)


__all__ = [
    "calculate_magic_numbers",
    "do_magic_division",
    "do_magic_division_dynamic",
]
