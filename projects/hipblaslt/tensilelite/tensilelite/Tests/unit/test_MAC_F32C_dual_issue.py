#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Unit tests for VOPD dual-issue in the ComplexFloat MAC component
# (tensilelite/Components/MAC_F32C.py).
#
# A complex multiply-accumulate is four real f32 FMAs, issued as two
# v_dual_fmac_f32 pairs:
#     pair 1: c.re += a.re*b.re    c.im += a.im*b.re
#     pair 2: c.re -= a.im*b.im    c.im += a.re*b.im
# VOPD cannot negate an operand, so the minus sits on b, in a temp made with xor.
# The temps are the last registers of ValuB, so their parity relative to b.re/b.im
# is the same wherever ValuB lands (main loop or tail loop).
#
# VOPD rules the assembler enforces: the two destinations differ in parity, the two
# src0 differ in bank (bank = register mod 4), the two src1 differ in bank or are the
# same register.
#
# Both halves of a pair may use the same src1 register. gfx12 VOPD allows that; gfx11 wants different
# banks even then, which these tests do not check. The stub writer below mirrors KernelWriter's ValuB
# sizing; the config tests in Tests/common/gemm/cgemm_mac.yaml cover the real writer.
#
# Usage:
#   pytest test_MAC_F32C_dual_issue.py -v
################################################################################

import random
import re
import shutil

import pytest
from types import SimpleNamespace

from tensilelite.Tests.rocisa_test_state import preserve_rocisa_kernel_state

pytestmark = pytest.mark.unit

WAVESIZE_32 = 32
VOPD_ARCHS = ["gfx1100", "gfx1201"]


def _init_rocisa(arch):
    from rocisa import rocIsa
    from tensilelite.Common.Architectures import gfxToIsa
    ri = rocIsa.getInstance()
    isa = gfxToIsa(arch)
    ri.init(isa, shutil.which("amdclang++") or "/usr/bin/amdclang++")
    ri.setKernel(isa, WAVESIZE_32)


@pytest.fixture(scope="module", autouse=True, params=VOPD_ARCHS)
def _rocisa_arch(request):
    with preserve_rocisa_kernel_state():
        _init_rocisa(request.param)
        yield


def _component():
    import tensilelite.Component  # noqa: F401  -- populates the component registry
    from tensilelite.Components.MAC_F32C import MAC_F32C_Plain
    return MAC_F32C_Plain()


class _StubWriter:
    def __init__(self, kernel):
        iu = kernel["InnerUnroll"]
        blocks = 2 * iu
        valuB = 2 * kernel["ThreadTile1"] * blocks
        negOff = valuB if kernel["UseDualFMAC"] else None
        self.states = SimpleNamespace(kernel=kernel, b=SimpleNamespace(
            numVgprValu=valuB + (2 * iu if negOff is not None else 0), valuNegTempOffset=negOff))


def _kernel(tt0=2, tt1=2, ccA=False, ccB=False, dual=True, iu=1):
    return {
        "ThreadTile0": tt0,
        "ThreadTile1": tt1,
        "InnerUnroll": iu,
        "UseDualFMAC": dual,
        "ProblemType": {"ComplexConjugateA": ccA, "ComplexConjugateB": ccB},
    }


def _emit(kernel, m=0):
    tP = {"tile01Idx": 1}
    return str(_component()(_StubWriter(kernel), tP, tP, m, kernel["InnerUnroll"]))


def _counts(asm):
    """-> (vopd_pairs, single_issue_fmas).

    Count LINES, not mnemonic occurrences: a VOPD line carries the mnemonic twice
    (`v_dual_fmac_f32 ... :: v_dual_fmac_f32 ...`).
    """
    lines = asm.splitlines()
    pairs = sum(1 for ln in lines if "::" in ln and "v_dual_fmac_f32" in ln)
    singles = sum(1 for ln in lines if "v_fmac_f32" in ln and "::" not in ln)
    return pairs, singles


ALL_CONJUGATES = [(False, False), (False, True), (True, False), (True, True)]


def _resolver(kernel, a_base, b_base):
    """Map the macro's symbolic operands to physical vgprs, like the .set lines do."""
    iu = kernel["InnerUnroll"]
    blkA, blkB = 2 * kernel["ThreadTile0"], 2 * kernel["ThreadTile1"]

    def resolve(tok):
        mm = re.fullmatch(r"(-?)v\[vgprValu([ABC])(?:_X(\d+)_I(\d+))?\+(\d+)\]", tok.strip())
        assert mm, f"unexpected operand {tok!r}"
        neg, kind, x, i, off = mm.groups()
        if kind == "C":
            return neg, 2000 + int(off)
        base, blk = (a_base, blkA) if kind == "A" else (b_base, blkB)
        return neg, base + (int(x) * iu + int(i)) * blk + int(off)
    return resolve


def _ops(asm, resolve):
    """-> list of (mnemonic, dst, [(neg, src)...]) with VOPD split into its two halves."""
    out = []
    for ln in asm.splitlines():
        ln = ln.split("//")[0].strip()
        for half in ln.split("::"):
            half = half.strip()
            mn = half.split(" ", 1)[0]
            if mn not in ("v_xor_b32", "v_dual_fmac_f32", "v_fmac_f32"):
                continue
            args = [t for t in half.split(" ", 1)[1].split(",")]
            dst = resolve(args[0])[1]
            srcs = [("", int(t, 16)) if t.strip().startswith("0x") else resolve(t) for t in args[1:]]
            out.append((mn, dst, srcs, "::" in ln))
    return out


@pytest.mark.parametrize("b_base", [24, 25])   # tail-loop (even) and main-loop (odd) bases
@pytest.mark.parametrize("iu", [1, 2])
@pytest.mark.parametrize("ccA,ccB", ALL_CONJUGATES)
def test_vopd_bank_rules_hold_for_any_valub_base(ccA, ccB, iu, b_base):
    kernel = _kernel(2, 4, ccA, ccB, iu=iu)
    for m in range(2):
        asm = _emit(kernel, m)
        dual = [l for l in asm.splitlines() if "::" in l]
        resolve = _resolver(kernel, 16, b_base)
        for ln in dual:
            x, y = [_ops(h, resolve)[0] for h in ln.split("::")]
            assert (x[1] % 2) != (y[1] % 2), f"dst parity: {ln}"
            for k in range(2):  # src0, src1: identical register or different bank
                rx, ry = x[2][k][1], y[2][k][1]
                assert rx == ry or rx % 4 != ry % 4, f"src{k} bank clash at base {b_base}: {ln}"


def _run(asm, resolve, regs):
    """Tiny interpreter: enough to check the arithmetic of the emitted stream."""
    def val(neg, r):
        if r >= 0x80000000:
            return r
        return -regs[r] if neg else regs[r]
    for mn, dst, srcs, _ in _ops(asm, resolve):
        if mn == "v_xor_b32":
            assert srcs[0][1] == 0x80000000
            regs[dst] = -val(*srcs[1])
        else:
            regs[dst] = regs.get(dst, 0.0) + val(*srcs[0]) * val(*srcs[1])


@pytest.mark.parametrize("dual", [True, False])
@pytest.mark.parametrize("ccA,ccB", ALL_CONJUGATES)
def test_emitted_stream_computes_the_complex_mac(ccA, ccB, dual):
    tt0, tt1 = 2, 4
    kernel = _kernel(tt0, tt1, ccA, ccB, dual=dual)
    resolve = _resolver(kernel, 16, 25)
    rnd = random.Random(1234)
    regs = {r: rnd.uniform(-2, 2) for r in range(16, 16 + 64)}
    A = [complex(regs[16 + 2*a], regs[17 + 2*a]) for a in range(tt0)]
    B = [complex(regs[25 + 2*b], regs[26 + 2*b]) for b in range(tt1)]
    _run(_emit(kernel), resolve, regs)
    for b in range(tt1):
        for a in range(tt0):
            want = (A[a].conjugate() if ccA else A[a]) * (B[b].conjugate() if ccB else B[b])
            got = complex(regs[2000 + 2*(a + b*tt0)], regs[2001 + 2*(a + b*tt0)])
            assert abs(got - want) < 1e-12, f"a={a} b={b}: {got} != {want}"


@pytest.mark.parametrize("iu", [1, 2])
@pytest.mark.parametrize("ccA,ccB", ALL_CONJUGATES)
def test_all_four_fmas_are_dual_issued(ccA, ccB, iu):
    """Every real FMA lands in a VOPD pair for all four conjugate combinations,
    including conjugate-both, which used to emit no VOPD at all."""
    tt0 = tt1 = 2
    pairs, singles = _counts(_emit(_kernel(tt0, tt1, ccA, ccB, iu=iu)))
    assert pairs == 2 * tt0 * tt1 * iu
    assert singles == 0


@pytest.mark.parametrize("ccA,ccB", ALL_CONJUGATES)
def test_dual_issue_off_emits_four_plain_fmas(ccA, ccB):
    """With the flag off: four plain FMAs per element and no negation temps."""
    tt0 = tt1 = 2
    asm = _emit(_kernel(tt0, tt1, ccA, ccB, dual=False))
    assert _counts(asm) == (0, 4 * tt0 * tt1)
    assert "v_xor_b32" not in asm


def _instruction_stream(asm, resolve):
    """One entry per emitted instruction: its ops (a VOPD line carries two); comment and s_setprio lines are skipped."""
    return [ops for ops in (_ops(ln, resolve) for ln in asm.splitlines()) if ops]


@pytest.mark.parametrize("tt0", [2, 4])
@pytest.mark.parametrize("ccA,ccB", ALL_CONJUGATES)
def test_second_pair_is_emitted_one_row_after_the_first(ccA, ccB, tt0):
    """The second pair writes the accumulators the first pair just wrote, and a dependent dual-issue
    instruction waits several cycles. So a whole row of one pair is emitted before the other: at least
    tt0 instructions sit between the two writers of every accumulator register."""
    kernel = _kernel(tt0, 2, ccA, ccB)
    stream = _instruction_stream(_emit(kernel), _resolver(kernel, 16, 25))
    writers = {}
    for i, ops in enumerate(stream):
        for _mn, dst, _srcs, _vopd in ops:
            if dst >= 2000:
                writers.setdefault(dst, []).append(i)
    assert len(writers) == 2 * tt0 * 2
    for dst, idx in writers.items():
        assert len(idx) == 2, f"accumulator {dst} written {len(idx)} times"
        assert idx[1] - idx[0] >= tt0, f"accumulator {dst}: pairs only {idx[1] - idx[0]} instructions apart"


@pytest.mark.parametrize("iu", [1, 2])
@pytest.mark.parametrize("ccA,ccB", ALL_CONJUGATES)
def test_negation_temps_are_the_only_extra_registers_written(ccA, ccB, iu):
    """The xor writes only the reserved temps after the last ValuB block, never b data or A."""
    tt0, tt1 = 2, 2
    kernel = _kernel(tt0, tt1, ccA, ccB, iu=iu)
    writer = _StubWriter(kernel)
    b_base = 24
    resolve = _resolver(kernel, 16, b_base)
    first, end = b_base + writer.states.b.valuNegTempOffset, b_base + writer.states.b.numVgprValu
    xor_dsts = [dst for ops in _instruction_stream(_emit(kernel), resolve) for mn, dst, _s, _v in ops if mn == "v_xor_b32"]
    assert xor_dsts
    assert all(first <= dst < end for dst in xor_dsts), f"xor wrote outside [{first}, {end}): {xor_dsts}"


@pytest.mark.parametrize("ccA,ccB", ALL_CONJUGATES)
def test_pair_order_puts_the_pair_without_a_fresh_xor_first(ccA, ccB):
    """With only A conjugated, only pair 1 needs a fresh xor (-b.re), so pair 2 is emitted first.
    Otherwise pair 1 goes first. Pair 1 reads a.re as src0 of its first half, pair 2 reads a.im."""
    kernel = _kernel(2, 1, ccA, ccB)
    first_dual = next(l for l in _emit(kernel).splitlines() if "::" in l).split("::")[0]
    src0_offset = int(re.search(r"vgprValuA_X\d+_I\d+\+(\d+)", first_dual).group(1)) % 2
    pair2_first = ccA and not ccB
    assert src0_offset == (1 if pair2_first else 0)


def test_dual_issue_pairs_keep_src0_on_adjacent_valua():
    """src0 of both halves are the ADJACENT ValuA real/imag registers, so the src0 bank
    rule holds by construction."""
    asm = _emit(_kernel(2, 2, True, True))
    for ln in [l for l in asm.splitlines() if "::" in l and "v_dual_fmac_f32" in l]:
        x, y = ln.split("::", 1)
        xa = re.search(r"vgprValuA_X\d+_I\d+\+(\d+)", x)
        ya = re.search(r"vgprValuA_X\d+_I\d+\+(\d+)", y)
        assert xa and ya, f"both halves must read ValuA as src0: {ln}"
        assert abs(int(xa.group(1)) - int(ya.group(1))) == 1, ln
