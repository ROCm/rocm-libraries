# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GPU execution tests for the scalar magic-number division helpers of f_math.

``scalarStaticDivideAndRemainder`` and ``scalarStaticCeilDivide`` emit a fixed
SALU sequence that derives ``dividend / divisor`` from a compile-time magic
constant. Each test assembles the emitted sequences into a single-wave kernel
that loops over a buffer of dividends, runs it on the device, and compares
every result against Python integer division.

Usage:
    pytest test_scalar_static_divide.py -v
    python test_scalar_static_divide.py
"""

import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

from gpu_test_helpers import (
    assemble_and_run,
    generate_kernel_asm,
    generate_load_params,
    init_rocisa,
    requires_gpu,
)

from rocisa.code import Module, TextBlock
from rocisa.container import ContinuousRegister, sgpr
from rocisa.functions import scalarStaticCeilDivide, scalarStaticDivideAndRemainder

U32 = 0xFFFFFFFF

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

# ---------------------------------------------------------------------------
# Kernel register map
# ---------------------------------------------------------------------------
# s[0:1] kernarg pointer, s[4:5] dividend cursor, s[6:7] output cursor,
# s10 remaining count, s11 current dividend, s[12:13] shared aligned tmp pair.
# Results are staged in v[OUT_VGPR:], which must be even-aligned for dwordx4.
IN_PTR, OUT_PTR, COUNT, DIVIDEND, TMP = 4, 6, 10, 11, 12
OUT_VGPR = 4
TMP_RES = ContinuousRegister(TMP, 2)

# One helper call per output column: (column name, mode, qReg, rReg, dReg,
# output register). mode is doRemainder (0/1/2) or "ceil". Every call reads its
# own copy of the dividend, so calls that overwrite dReg do not affect others.
CALLS = [
    ("quotient",           1,      16, 17, 18, 16),
    ("remainder",          1,      16, 17, 18, 17),
    ("quotient_only",      0,      19, 20, 18, 19),
    ("remainder_only",     2,      21, 22, 18, 22),
    ("ceil",               "ceil", 23, None, 18, 23),
    # Register aliasing used by kernel-writer call sites.
    ("alias_q_eq_r",       1,      24, 24, 25, 24),
    ("alias_q_eq_tmp_lo",  2,      TMP, 26, 27, 26),
    ("alias_q_eq_d",       0,      28, 29, 28, 28),
]
NUM_COLUMNS = len(CALLS)


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


def _is_pow2(divisor):
    return divisor & (divisor - 1) == 0


def _dividends(divisor):
    """Dividends bracketing the low-half overflow threshold, the exactness
    limit, and a seeded sample of the exact range."""
    rng = np.random.default_rng(divisor)
    if _is_pow2(divisor):
        values = [0, 1, divisor - 1, divisor, divisor + 1, 109227, 1 << 30, 1 << 31, U32]
        values += rng.integers(0, U32, 256, endpoint=True).tolist()
        return np.array(sorted(set(values)), dtype=np.uint32)

    threshold = _low_half_overflow_threshold(divisor)
    limit = min(_exactness_limit(divisor), U32)
    assert threshold + 1 <= limit, f"overflow threshold not testable for {divisor}"
    values = [
        0,
        1,
        divisor - 1,
        divisor,
        divisor + 1,
        # Exact multiples of the divisor above the threshold: these are the
        # cases where a too-low quotient aliases the remainder onto 0.
        (threshold // divisor + 1) * divisor,
        (threshold // divisor + 1) * divisor - 1,
        (threshold // divisor + 1) * divisor + 1,
        4 * threshold,
        4 * threshold + divisor - 1,
        1 << 31,
        U32 - 1,
        U32,
        limit - divisor,
        limit - 1,
        limit,
    ]
    values += range(threshold - 2 * divisor, threshold + 2 * divisor)
    values += rng.integers(0, limit, 256, endpoint=True).tolist()
    return np.array(sorted({v for v in values if 0 <= v <= limit}), dtype=np.uint32)


def _expected(divisor, dividends):
    d = dividends.astype(np.uint64)
    q, r = d // divisor, d % divisor
    ceil = q + (r != 0)
    columns = {
        "quotient": q,
        "remainder": r,
        "quotient_only": q,
        "remainder_only": r,
        "ceil": ceil,
        "alias_q_eq_r": r,
        "alias_q_eq_tmp_lo": r,
        "alias_q_eq_d": q,
    }
    return np.stack([columns[name] for name, *_ in CALLS], axis=1).astype(np.uint32)


# ---------------------------------------------------------------------------
# Kernel generation
# ---------------------------------------------------------------------------


def _helper_module(divisor, mode, q, r, d):
    if mode == "ceil":
        return scalarStaticCeilDivide(sgpr(q), sgpr(d), divisor, TMP_RES)
    return scalarStaticDivideAndRemainder(q, r, d, divisor, TMP_RES, mode)


def generate_divide_kernel(divisor):
    """Single-wave kernel: for each dividend, run every helper call in CALLS
    and store the NUM_COLUMNS results as consecutive dwords."""
    init_rocisa()
    writer = SimpleNamespace(sgprs={})

    body = Module("divide body")
    body.add(TextBlock(f"  s_load_dword s{DIVIDEND}, s[{IN_PTR}:{IN_PTR + 1}], 0x0\n"))
    body.add(TextBlock("  s_waitcnt lgkmcnt(0)\n"))
    for _, mode, q, r, d, _ in CALLS:
        body.add(TextBlock(f"  s_mov_b32 s{d}, s{DIVIDEND}\n"))
        body.add(_helper_module(divisor, mode, q, r, d))
    for col, (*_, out) in enumerate(CALLS):
        body.add(TextBlock(f"  v_mov_b32 v{OUT_VGPR + col}, s{out}\n"))
    for col in range(0, NUM_COLUMNS, 4):
        body.add(TextBlock(
            f"  global_store_dwordx4 v0, v[{OUT_VGPR + col}:{OUT_VGPR + col + 3}], "
            f"s[{OUT_PTR}:{OUT_PTR + 1}] offset:{col * 4}\n"))

    inner = Module("scalar static divide")
    inner.add(generate_load_params([
        (IN_PTR, 2, 0, "dividends"),
        (OUT_PTR, 2, 8, "results"),
        (COUNT, 1, 16, "count"),
    ]))
    inner.add(TextBlock("  s_mov_b64 exec, 1\n  v_mov_b32 v0, 0\n"))
    inner.add(TextBlock("label_divide_loop:\n"))
    inner.add(body)
    inner.add(TextBlock(
        f"  s_add_u32 s{IN_PTR}, s{IN_PTR}, 4\n"
        f"  s_addc_u32 s{IN_PTR + 1}, s{IN_PTR + 1}, 0\n"
        f"  s_add_u32 s{OUT_PTR}, s{OUT_PTR}, {NUM_COLUMNS * 4}\n"
        f"  s_addc_u32 s{OUT_PTR + 1}, s{OUT_PTR + 1}, 0\n"
        f"  s_sub_u32 s{COUNT}, s{COUNT}, 1\n"
        f"  s_cmp_lg_u32 s{COUNT}, 0\n"
        "  s_cbranch_scc1 label_divide_loop\n"))

    args = (
        ("dividends_ptr", 8, "global_buffer", "u32"),
        ("output_ptr",    8, "global_buffer", "u32"),
        ("count",         4, "by_value",      "u32"),
    )
    return generate_kernel_asm(str(inner), writer, args, lds_size=0, num_threads=64)


def run_divide(divisor, tmp_path):
    """Run the kernel for ``divisor``; return (dividends, results, expected)."""
    dividends = _dividends(divisor)
    kernel_asm = generate_divide_kernel(divisor)
    output_size = len(dividends) * NUM_COLUMNS * 4
    raw = assemble_and_run(kernel_asm, tmp_path, f"scalar_div_{divisor}", output_size,
                           inputs=(dividends,), scalars=(len(dividends),), num_threads=64)
    results = np.frombuffer(raw, dtype=np.uint32).reshape(len(dividends), NUM_COLUMNS)
    return dividends, results, _expected(divisor, dividends)


def verify(divisor, dividends, results, expected, max_report=10):
    """Return a list of mismatch descriptions."""
    errors = []
    for row, col in zip(*np.nonzero(results != expected)):
        if len(errors) < max_report:
            errors.append(f"{CALLS[col][0]}({dividends[row]} / {divisor}): "
                          f"got {results[row, col]}, expected {expected[row, col]}")
        else:
            errors.append("...")
            break
    return errors


# ---------------------------------------------------------------------------
# Pytest tests
# ---------------------------------------------------------------------------


@requires_gpu
class TestScalarStaticDivide:

    @pytest.mark.parametrize("divisor", NON_POW2_DIVISORS + POW2_DIVISORS)
    def test_scalar_static_divide(self, divisor, tmp_path):
        errors = verify(divisor, *run_divide(divisor, tmp_path))
        assert not errors, f"divisor {divisor}: " + "; ".join(errors)


@pytest.mark.parametrize("divisor", NON_POW2_DIVISORS)
def test_magic_product_kept_at_64_bits(divisor):
    """The magic product must be formed with a high-half multiply, not truncated."""
    for text in (
        str(scalarStaticDivideAndRemainder(16, 17, 18, divisor, TMP_RES, 1)),
        str(scalarStaticCeilDivide(sgpr(16), sgpr(18), divisor, TMP_RES)),
    ):
        assert "s_mul_hi_u32" in text, text


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    total_errors = 0
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = type('P', (), {'__truediv__': lambda s, n: os.path.join(tmp_dir, n)})()
        for divisor in NON_POW2_DIVISORS + POW2_DIVISORS:
            dividends, results, expected = run_divide(divisor, tmp_path)
            errors = verify(divisor, dividends, results, expected)
            status = "PASS" if not errors else f"FAIL: {'; '.join(errors)}"
            print(f"divisor {divisor:>6} ({len(dividends)} dividends): {status}")
            total_errors += len(errors)
    sys.exit(1 if total_errors else 0)
