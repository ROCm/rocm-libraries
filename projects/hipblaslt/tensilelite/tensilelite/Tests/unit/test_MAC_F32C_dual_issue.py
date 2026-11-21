#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Unit tests for VOPD dual-issue in the ComplexFloat MAC component
# (tensilelite/Components/MAC_F32C.py).
#
# A complex multiply-accumulate is four real f32 FMAs, issued as two
# v_dual_fmac_f32 pairs. X works on column b, Y on its neighbour column b^1 ('):
#     pair 1: c.re += a.re*b.re      c'.im += a.im*b'.re
#     pair 2: c.re -= a.im*b.im      c'.im += a.re*b'.im
# VOPD cannot negate an operand, so the minus sits on b, in vgprValuBNeg made with xor.
# vgprValuBNeg lives inside the ValuB allocation, so its bank relative to b is the
# same wherever ValuB lands (main loop or tail loop).
#
# VOPD rules: the two destinations differ in parity, the two src0 differ in bank
# (bank = register mod 4), the two src1 differ in bank. gfx12 also accepts the same
# register in both halves; gfx11 does not, so these tests hold the gfx11 rule.
# The stub writer below mirrors KernelWriter's ValuB sizing; the config tests in
# Tests/common/gemm/cgemm_mac.yaml cover the real writer.
#
# Usage:
#   pytest test_MAC_F32C_dual_issue.py -v
################################################################################

import os
import random
import re
import shutil
import subprocess

import pytest
from types import SimpleNamespace

from tensilelite.Tests.rocisa_test_state import preserve_rocisa_kernel_state

pytestmark = pytest.mark.unit

WAVESIZE_32 = 32
VOPD_ARCHS = ["gfx1100", "gfx1201"]
NEG_TEMPS = 4


def _assembler():
    return shutil.which("amdclang++") or "/opt/rocm/llvm/bin/amdclang++"


def _init_rocisa(arch):
    from rocisa import rocIsa
    from tensilelite.Common.Architectures import gfxToIsa
    ri = rocIsa.getInstance()
    isa = gfxToIsa(arch)
    ri.init(isa, _assembler())
    ri.setKernel(isa, WAVESIZE_32)


@pytest.fixture(scope="module", autouse=True, params=VOPD_ARCHS)
def _rocisa_arch(request):
    with preserve_rocisa_kernel_state():
        _init_rocisa(request.param)
        yield request.param


def _component():
    import tensilelite.Component  # noqa: F401  -- populates the component registry
    from tensilelite.Components.MAC_F32C import MAC_F32C_Plain
    return MAC_F32C_Plain()


class _StubWriter:
    def __init__(self, kernel):
        blocks = 2 * kernel["InnerUnroll"]  # PrefetchLocalRead=1
        valuB = 2 * kernel["ThreadTile1"] * blocks
        negOff = valuB if kernel["UseDualFMAC"] else None
        self.states = SimpleNamespace(kernel=kernel, b=SimpleNamespace(
            numVgprValu=valuB + (NEG_TEMPS * kernel["InnerUnroll"] if negOff is not None else 0),
            valuNegTempOffset=negOff))


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


def _bases(kernel, a_base, b_base, c_base=2000):
    """Physical vgpr of every name the macro uses, laid out like KernelWriterAssembly's .set lines."""
    iu = kernel["InnerUnroll"]
    blkA, blkB = 2 * kernel["ThreadTile0"], 2 * kernel["ThreadTile1"]
    names = {"ValuC": c_base}
    for x in range(2):
        for i in range(iu):
            names["ValuA_X%d_I%d" % (x, i)] = a_base + (x * iu + i) * blkA
            names["ValuB_X%d_I%d" % (x, i)] = b_base + (x * iu + i) * blkB
    if kernel["UseDualFMAC"]:
        names["ValuBNeg"] = b_base + _StubWriter(kernel).states.b.valuNegTempOffset
    return names


def _resolver(kernel, a_base, b_base):
    """Map the macro's symbolic operands to physical vgprs, like the .set lines do."""
    names = _bases(kernel, a_base, b_base)

    def resolve(tok):
        mm = re.fullmatch(r"(-?)v\[vgpr(\w+)\+(\d+)\]", tok.strip())
        assert mm and mm.group(2) in names, f"unexpected operand {tok!r}"
        neg, name, off = mm.groups()
        return neg, names[name] + int(off)
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
            args = half.split(" ", 1)[1].split(",")
            dst = resolve(args[0])[1]
            srcs = [("", int(t, 16)) if t.strip().startswith("0x") else resolve(t) for t in args[1:]]
            out.append((mn, dst, srcs, "::" in ln))
    return out


@pytest.mark.parametrize("b_base", [24, 25, 26, 27])   # every bank ValuB can start on
@pytest.mark.parametrize("iu", [1, 2])
@pytest.mark.parametrize("ccA,ccB", ALL_CONJUGATES)
def test_vopd_bank_rules_hold_for_any_valub_base(ccA, ccB, iu, b_base):
    kernel = _kernel(2, 4, ccA, ccB, iu=iu)
    for m in range(2):
        asm = _emit(kernel, m)
        dual = [l for l in asm.splitlines() if "::" in l]
        assert dual
        resolve = _resolver(kernel, 16, b_base)
        for ln in dual:
            x, y = [_ops(h, resolve)[0] for h in ln.split("::")]
            assert (x[1] % 2) != (y[1] % 2), f"dst parity: {ln}"
            for k in range(2):  # src0, src1: different bank, even for the same register (gfx11)
                rx, ry = x[2][k][1], y[2][k][1]
                assert rx % 4 != ry % 4, f"src{k} bank clash at base {b_base}: {ln}"


@pytest.mark.parametrize("ccA,ccB", ALL_CONJUGATES)
def test_real_assembler_accepts_the_stream(_rocisa_arch, ccA, ccB, tmp_path):
    """The assembler is the authority on VOPD rules; ValuB starts odd in the main loop, even in the tail loop."""
    asmPath = _assembler()
    if not os.path.exists(asmPath):
        pytest.skip("no amdclang++")
    kernel = _kernel(4, 4, ccA, ccB, iu=2)
    for b_base in (129, 130):
        sets = "".join(".set vgpr%s, %d\n" % kv for kv in _bases(kernel, 16, b_base, c_base=64).items())
        src = tmp_path / "mac.s"
        src.write_text(sets + _emit(kernel, 0) + _emit(kernel, 1))
        r = subprocess.run([asmPath, "-x", "assembler", "-target", "amdgcn-amd-amdhsa", "-mcpu=" + _rocisa_arch,
                            "-Xclang", "-target-feature", "-Xclang", "+wavefrontsize32",
                            "-c", str(src), "-o", str(tmp_path / "mac.o")],
                           capture_output=True, text=True, stdin=subprocess.DEVNULL)
        assert r.returncode == 0, f"{_rocisa_arch} ValuB base {b_base}:\n{r.stderr}"


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


@pytest.mark.parametrize("iu", [1, 2])
@pytest.mark.parametrize("dual", [True, False])
@pytest.mark.parametrize("ccA,ccB", ALL_CONJUGATES)
def test_emitted_stream_computes_the_complex_mac(ccA, ccB, dual, iu):
    tt0, tt1 = 2, 4
    kernel = _kernel(tt0, tt1, ccA, ccB, dual=dual, iu=iu)
    names = _bases(kernel, 16, 125)
    resolve = _resolver(kernel, 16, 125)
    rnd = random.Random(1234)
    # only the A and B inputs are filled: reading anything else (vgprValuBNeg before its xor) raises KeyError
    regs = {}
    for name, base in names.items():
        if name.startswith(("ValuA_", "ValuB_X")):
            regs.update({base + k: rnd.uniform(-2, 2) for k in range(2 * (tt0 if name.startswith("ValuA") else tt1))})

    def cplx(name, k, conj):
        r = names[name] + 2*k
        z = complex(regs[r], regs[r + 1])
        return z.conjugate() if conj else z
    want = {(a, b): sum(cplx("ValuA_X0_I%d" % i, a, ccA) * cplx("ValuB_X0_I%d" % i, b, ccB) for i in range(iu))
            for a in range(tt0) for b in range(tt1)}
    _run(_emit(kernel), resolve, regs)
    for (a, b), w in want.items():
        got = complex(regs[2000 + 2*(a + b*tt0)], regs[2001 + 2*(a + b*tt0)])
        assert abs(got - w) < 1e-12, f"a={a} b={b}: {got} != {w}"


def test_odd_threadtile1_is_rejected():
    """The neighbour column b ^ 1 would fall outside an odd tile. The Solution gate keeps these out;
    the component also refuses instead of reading B registers that are not there."""
    with pytest.raises(AssertionError, match="ThreadTile1 must be even"):
        _emit(_kernel(2, 3))


@pytest.mark.parametrize("iu", [1, 2])
@pytest.mark.parametrize("ccA,ccB", ALL_CONJUGATES)
def test_all_four_fmas_are_dual_issued(ccA, ccB, iu):
    """Every real FMA lands in a VOPD pair for all four conjugate combinations."""
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
    """The xor writes only vgprValuBNeg, after the last ValuB block, never b data or A."""
    kernel = _kernel(2, 4, ccA, ccB, iu=iu)
    writer = _StubWriter(kernel)
    b_base = 24
    resolve = _resolver(kernel, 16, b_base)
    first, end = b_base + writer.states.b.valuNegTempOffset, b_base + writer.states.b.numVgprValu
    assert end - first == NEG_TEMPS * iu
    xor_dsts = [dst for ops in _instruction_stream(_emit(kernel), resolve) for mn, dst, _s, _v in ops if mn == "v_xor_b32"]
    assert xor_dsts
    assert all(first <= dst < end for dst in xor_dsts), f"xor wrote outside [{first}, {end}): {xor_dsts}"
    # each InnerUnroll writes only its own four temps
    assert sorted({(dst - first) // NEG_TEMPS for dst in xor_dsts}) == list(range(iu))


@pytest.mark.parametrize("ccA,ccB", ALL_CONJUGATES)
def test_pair_order_puts_the_pair_without_a_fresh_xor_first(ccA, ccB):
    """With only A conjugated, only pair 1 needs a fresh xor (-b.re), so pair 2 is emitted first.
    Otherwise pair 1 goes first. Pair 1 reads a.re as src0 of its first half, pair 2 reads a.im."""
    kernel = _kernel(2, 2, ccA, ccB)
    first_dual = next(l for l in _emit(kernel).splitlines() if "::" in l).split("::")[0]
    src0_offset = int(re.search(r"vgprValuA_X\d+_I\d+\+(\d+)", first_dual).group(1)) % 2
    pair2_first = ccA and not ccB
    assert src0_offset == (1 if pair2_first else 0)
