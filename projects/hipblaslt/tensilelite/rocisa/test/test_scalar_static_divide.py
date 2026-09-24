# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Execution tests for the scalar magic-number division helpers of f_math.

``scalarStaticDivideAndRemainder`` and ``scalarStaticCeilDivide`` emit a fixed
SALU sequence that derives ``dividend / divisor`` from a compile-time magic
constant. These tests interpret the emitted sequence with a small SGPR
simulator and compare against Python integer division, so what is checked is
the arithmetic the kernel actually executes, not just the instruction text.
"""

import re

import pytest

from rocisa.container import ContinuousRegister, sgpr
from rocisa.functions import scalarStaticCeilDivide, scalarStaticDivideAndRemainder

U32 = 0xFFFFFFFF

# Register assignment shared by all cases: quotient, remainder, dividend, and a
# two-SGPR aligned temporary pair.
Q_REG, R_REG, D_REG, TMP_REG = 0, 1, 2, 4
TMP_RES = ContinuousRegister(TMP_REG, 2)

# Non-power-of-2 divisors take the magic-number path: small odd and prime
# divisors (3 has a magic constant above INT32_MAX), macrotile-like sizes,
# larger odd/prime sizes, and divisors above 16 bits. Powers of 2 (including 1)
# take the shift/AND fast path.
NON_POW2_DIVISORS = [
    3, 5, 7, 11, 13,
    96, 160, 192, 272, 320,
    97, 1009, 1023,
    65537, 100003,
]
POW2_DIVISORS = [1, 2, 16, 32, 64, 128, 256]


def _magic(divisor):
    """Magic constant and shift used by the helpers for ``divisor``."""
    shift = 33
    return ((1 << shift) // divisor) + 1, shift


def _low_half_overflow_threshold(divisor):
    """Smallest dividend whose product with the low 16 bits of magic exceeds 32 bits."""
    magic, _ = _magic(divisor)
    magic_lo = magic & 0xFFFF
    return -(-(1 << 32) // magic_lo)  # ceil division


def _exactness_limit(divisor):
    """Largest dividend for which magic division is guaranteed exact."""
    magic, shift = _magic(divisor)
    return ((1 << shift) - 1) // (magic * divisor - (1 << shift))


def _dividends(divisor):
    """Dividends bracketing the low-half overflow threshold, plus small values
    and the largest dividend the magic constant is guaranteed exact for."""
    threshold = _low_half_overflow_threshold(divisor)
    limit = min(_exactness_limit(divisor), U32)
    assert threshold + 1 <= limit, f"overflow threshold not testable for {divisor}"
    values = [
        0,
        1,
        divisor - 1,
        divisor,
        divisor + 1,
        threshold - divisor,
        threshold - 1,
        threshold,
        threshold + 1,
        threshold + divisor,
        # Exact multiples of the divisor above the threshold: these are the
        # cases where a too-low quotient aliases the remainder onto 0.
        (threshold // divisor + 1) * divisor,
        (threshold // divisor + 1) * divisor - 1,
        (threshold // divisor + 1) * divisor + 1,
        4 * threshold,
        4 * threshold + divisor - 1,
        limit - divisor,
        limit - 1,
        limit,
    ]
    return sorted({v for v in values if 0 <= v <= limit})


# ---------------------------------------------------------------------------
# Minimal SGPR interpreter for the emitted instructions.
# ---------------------------------------------------------------------------


def _parse_operand(text):
    """Return ``("reg", idx, count)`` or ``("imm", value, 1)``."""
    text = text.strip().rstrip(",")
    pair = re.fullmatch(r"s\[(\d+):(\d+)\]", text)
    if pair:
        lo, hi = int(pair.group(1)), int(pair.group(2))
        return ("reg", lo, hi - lo + 1)
    single = re.fullmatch(r"s(\d+)", text)
    if single:
        return ("reg", int(single.group(1)), 1)
    return ("imm", int(text, 0), 1)


class _Sim:
    def __init__(self):
        self.regs = {}
        self.scc = 0

    def read(self, operand):
        kind, value, count = operand
        if kind == "imm":
            return value & U32
        return sum(self.regs.get(value + i, 0) << (32 * i) for i in range(count))

    def write(self, operand, value):
        _, idx, count = operand
        for i in range(count):
            self.regs[idx + i] = (value >> (32 * i)) & U32

    def run(self, module):
        for line in str(module).splitlines():
            line = line.split("//")[0].strip()
            if not line:
                continue
            opcode, _, rest = line.partition(" ")
            ops = [_parse_operand(o) for o in rest.split(",")] if rest.strip() else []
            self._exec(opcode, ops)
        return self

    def _exec(self, opcode, ops):
        if opcode == "s_mov_b32":
            self.write(ops[0], self.read(ops[1]))
        elif opcode == "s_mul_i32":
            self.write(ops[0], (self.read(ops[1]) * self.read(ops[2])) & U32)
        elif opcode == "s_mul_hi_u32":
            self.write(ops[0], (self.read(ops[1]) * self.read(ops[2])) >> 32 & U32)
        elif opcode == "s_lshl_b64":
            result = (self.read(ops[1]) << (self.read(ops[2]) & 0x3F)) & ((1 << 64) - 1)
            self.write(ops[0], result)
            self.scc = int(result != 0)
        elif opcode == "s_lshr_b64":
            result = self.read(ops[1]) >> (self.read(ops[2]) & 0x3F)
            self.write(ops[0], result)
            self.scc = int(result != 0)
        elif opcode == "s_lshr_b32":
            result = (self.read(ops[1]) >> (self.read(ops[2]) & 0x1F)) & U32
            self.write(ops[0], result)
            self.scc = int(result != 0)
        elif opcode == "s_and_b32":
            result = self.read(ops[1]) & self.read(ops[2])
            self.write(ops[0], result)
            self.scc = int(result != 0)
        elif opcode == "s_add_u32":
            total = self.read(ops[1]) + self.read(ops[2])
            self.write(ops[0], total & U32)
            self.scc = int(total > U32)
        elif opcode == "s_addc_u32":
            total = self.read(ops[1]) + self.read(ops[2]) + self.scc
            self.write(ops[0], total & U32)
            self.scc = int(total > U32)
        elif opcode == "s_sub_u32":
            diff = self.read(ops[1]) - self.read(ops[2])
            self.write(ops[0], diff & U32)
            self.scc = int(diff < 0)
        elif opcode == "s_cmp_lg_u32":
            self.scc = int(self.read(ops[0]) != self.read(ops[1]))
        else:
            raise AssertionError(f"unhandled opcode in emitted sequence: {opcode}")


def _run_divide_regs(divisor, dividend, q_reg, r_reg, d_reg, do_remainder):
    module = scalarStaticDivideAndRemainder(
        q_reg, r_reg, d_reg, divisor, TMP_RES, do_remainder
    )
    sim = _Sim()
    sim.regs[d_reg] = dividend
    sim.run(module)
    return sim


def _run_divide(divisor, dividend, do_remainder=1):
    sim = _run_divide_regs(divisor, dividend, Q_REG, R_REG, D_REG, do_remainder)
    return sim.regs.get(Q_REG), sim.regs.get(R_REG)


def _run_ceil(divisor, dividend):
    module = scalarStaticCeilDivide(sgpr(Q_REG), sgpr(D_REG), divisor, TMP_RES)
    sim = _Sim()
    sim.regs[D_REG] = dividend
    sim.run(module)
    return sim.regs.get(Q_REG)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("divisor", NON_POW2_DIVISORS)
def test_divide_and_remainder_non_power_of_2(divisor):
    for dividend in _dividends(divisor):
        quotient, remainder = _run_divide(divisor, dividend)
        assert quotient == dividend // divisor, (divisor, dividend)
        assert remainder == dividend % divisor, (divisor, dividend)


@pytest.mark.parametrize("divisor", NON_POW2_DIVISORS)
def test_divide_quotient_only_non_power_of_2(divisor):
    for dividend in _dividends(divisor):
        quotient, _ = _run_divide(divisor, dividend, do_remainder=0)
        assert quotient == dividend // divisor, (divisor, dividend)


@pytest.mark.parametrize("divisor", NON_POW2_DIVISORS)
def test_divide_remainder_only_non_power_of_2(divisor):
    for dividend in _dividends(divisor):
        _, remainder = _run_divide(divisor, dividend, do_remainder=2)
        assert remainder == dividend % divisor, (divisor, dividend)


def test_magic_constant_above_int32_max():
    """Divisor 3 has magic 0xAAAAAAAB, which does not fit in a signed 32-bit int."""
    assert _magic(3)[0] > 0x7FFFFFFF
    for dividend in [1, 2, 3, 4, 5, 100, 12345, 1 << 31, U32 - 1, U32]:
        quotient, remainder = _run_divide(3, dividend)
        assert (quotient, remainder) == divmod(dividend, 3), dividend
        assert _run_ceil(3, dividend) == -(-dividend // 3), dividend


# Register aliasing used by kernel-writer call sites: (qReg, rReg, dReg,
# doRemainder, outputs checked).
ALIAS_CASES = {
    "quotient_and_remainder_share_reg": (Q_REG, Q_REG, D_REG, 1, "r"),
    "quotient_in_tmp_lo": (TMP_REG, R_REG, D_REG, 2, "r"),
    "quotient_overwrites_dividend": (D_REG, R_REG, D_REG, 0, "q"),
}


@pytest.mark.parametrize("case", list(ALIAS_CASES))
@pytest.mark.parametrize("divisor", [3, 7, 192, 320])
def test_divide_register_aliasing(case, divisor):
    q_reg, r_reg, d_reg, do_remainder, outputs = ALIAS_CASES[case]
    for dividend in _dividends(divisor):
        sim = _run_divide_regs(divisor, dividend, q_reg, r_reg, d_reg, do_remainder)
        if "q" in outputs:
            assert sim.regs[q_reg] == dividend // divisor, (divisor, dividend)
        if "r" in outputs:
            assert sim.regs[r_reg] == dividend % divisor, (divisor, dividend)


@pytest.mark.parametrize("divisor", POW2_DIVISORS)
def test_divide_and_remainder_power_of_2(divisor):
    for dividend in [0, 1, divisor - 1, divisor, 123456, 109227, 1 << 30, U32]:
        quotient, remainder = _run_divide(divisor, dividend)
        assert quotient == dividend // divisor, (divisor, dividend)
        assert remainder == dividend % divisor, (divisor, dividend)


@pytest.mark.parametrize("divisor", NON_POW2_DIVISORS)
def test_ceil_divide_non_power_of_2(divisor):
    for dividend in _dividends(divisor):
        expected = -(-dividend // divisor)
        assert _run_ceil(divisor, dividend) == expected, (divisor, dividend)


@pytest.mark.parametrize("divisor", POW2_DIVISORS)
def test_ceil_divide_power_of_2(divisor):
    for dividend in [0, 1, divisor - 1, divisor, 123456, 109227, 1 << 30]:
        expected = -(-dividend // divisor)
        assert _run_ceil(divisor, dividend) == expected, (divisor, dividend)


@pytest.mark.parametrize("divisor", NON_POW2_DIVISORS)
def test_last_element_tile_index_pattern(divisor):
    """(size - 1) % macroTile, the tail-loop reload index that motivated the fix."""
    for size in [109227, 109226, 196609, 327681, 1000000]:
        dividend = size - 1
        if dividend > _exactness_limit(divisor):
            continue
        _, remainder = _run_divide(divisor, dividend)
        assert remainder == dividend % divisor, (divisor, size)


@pytest.mark.parametrize("divisor", NON_POW2_DIVISORS)
def test_magic_product_kept_at_64_bits(divisor):
    """The magic product must be formed with a high-half multiply, not truncated."""
    for text in (
        str(scalarStaticDivideAndRemainder(Q_REG, R_REG, D_REG, divisor, TMP_RES, 1)),
        str(scalarStaticCeilDivide(sgpr(Q_REG), sgpr(D_REG), divisor, TMP_RES)),
    ):
        assert "s_mul_hi_u32" in text, text
