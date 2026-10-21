# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU test for ``scalarUInt32DivideAndRemainder`` (ROCM-31826).

Stream-K kernels use this helper to split a tile index into workgroup
coordinates, so it must be exact for every 32-bit dividend and nonzero
divisor. The test emits the helper for gfx942, runs the emitted instructions
through a one-lane emulator of just the opcodes it uses, and compares the
quotient and remainder with Python integer division. ``v_rcp_iflag_f32`` is
modeled as a correctly rounded float32 reciprocal.
"""

import random
import shutil
import struct

import pytest

from rocisa import rocIsa
from rocisa.container import ContinuousRegister
from rocisa.functions import scalarUInt32DivideAndRemainder

pytestmark = pytest.mark.unit

U32 = 0xFFFFFFFF
U24 = 0xFFFFFF


def _f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _cvt_u32(bits):
    value = _f32(bits)
    if value != value or value <= 0:
        return 0
    return min(int(value), U32)


VALU = {
    "v_cvt_f32_u32": lambda a: _bits(float(a)),
    "v_rcp_iflag_f32": lambda a: _bits(1.0 / _f32(a)),
    "v_cvt_u32_f32": _cvt_u32,
    "v_mov_b32": lambda a: a,
    "v_mul_f32": lambda a, b: _bits(_f32(a) * _f32(b)),
    "v_mul_u32_u24": lambda a, b: ((a & U24) * (b & U24)) & U32,
    "v_mul_lo_u32": lambda a, b: (a * b) & U32,
    "v_mul_hi_u32": lambda a, b: (a * b) >> 32,
    "v_add_u32": lambda a, b: (a + b) & U32,
    "v_sub_u32": lambda a, b: (a - b) & U32,
}
CMPX = {
    "v_cmpx_eq_u32": lambda a, b: a == b,
    "v_cmpx_gt_u32": lambda a, b: a > b,
    "v_cmpx_ge_u32": lambda a, b: a >= b,
}


def _run(asm, sgprs):
    """Run the emitted text for one uniform lane; returns the SGPR file."""
    regs = {f"s{i}": v for i, v in sgprs.items()}
    active = True

    def read(operand):
        return regs[operand] if operand in regs else int(operand, 0) & U32

    for line in asm.splitlines():
        line = line.split("//")[0].strip()
        if not line:
            continue
        op, _, rest = line.partition(" ")
        dst, *srcs = [x.strip() for x in rest.split(",")]
        if op in CMPX:
            active = CMPX[op](*map(read, srcs))
        elif op == "s_mov_b64" and dst == "exec":
            active = read(srcs[0]) == U32
        elif op == "v_readfirstlane_b32":
            regs[dst] = regs[srcs[0]]
        elif op in VALU:
            if active:
                regs[dst] = VALU[op](*map(read, srcs))
        else:
            raise AssertionError(f"emulator does not model: {line}")
    return regs


def _cases():
    edges = [0, 1, 2, 3, (1 << 24) - 1, 1 << 24, (1 << 24) + 1, (1 << 31) - 1,
             1 << 31, (1 << 31) + 1, U32 - 1, U32]
    cases = [(d, v) for d in edges for v in edges if v]
    # 65,536 x 65,552 at MT16x16 has 4096 * 4097 = 16,781,312 tiles.
    cases += [(16781311, 16781312), (16781312, 16781312), (16781311, 4096)]
    # Ticket shape: tiles the old helper mapped wrong on hardware (nWG0 = 1,212,416).
    cases += [(16973825, 1212416), (18186241, 1212416)]
    rng = random.Random(31826)
    for _ in range(400):
        cases.append((rng.getrandbits(32), rng.getrandbits(rng.randint(1, 32)) or 1))
    return cases


@pytest.fixture(scope="module", autouse=True)
def _gfx942():
    ri = rocIsa.getInstance()
    ri.init((9, 4, 2), shutil.which("amdclang++") or "/usr/bin/amdclang++")
    ri.setKernel((9, 4, 2), 64)


# (qReg, dReg, divReg, rReg): distinct registers, and the StreamK form where
# the quotient overwrites the dividend.
LAYOUTS = [(4, 12, 13, 14), (12, 12, 13, 14)]


@pytest.mark.parametrize("q,d,div,r", LAYOUTS)
@pytest.mark.parametrize("doRemainder", [True, False])
def test_divide_matches_python(q, d, div, r, doRemainder):
    asm = str(scalarUInt32DivideAndRemainder(
        qReg=q, dReg=d, divReg=div, rReg=r, tmpVgprRes=ContinuousRegister(idx=8, size=2),
        wavewidth=64, doRemainder=doRemainder))
    wrong = []
    for dividend, divisor in _cases():
        regs = _run(asm, {d: dividend, div: divisor})
        got = (regs[f"s{q}"], regs[f"s{r}"] if doRemainder else None)
        want = (dividend // divisor, dividend % divisor if doRemainder else None)
        if got != want:
            wrong.append(f"{dividend} / {divisor}: got {got}, want {want}")
    assert not wrong, f"{len(wrong)} wrong, first: " + "; ".join(wrong[:5])
