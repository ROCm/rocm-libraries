# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Lowering of 64-bit integer adds on targets without a native 64-bit add.

SAddU64 and VAddNCU64 lower to a pair of 32-bit adds on targets that lack s_add_u64 and
v_add_nc_u64 (gfx942, gfx950, ...): the low add sets the carry and the high add consumes it.
A register pair splits into its two halves. An immediate must split into its low 32 bits and
the sign extension of those into the high half, never the immediate again: with it, ptr += imm
would also add imm * 2^32 and move the pointer hundreds of GiB (AMD-ROCm-Internal PR 2557,
class C2 on AIHPBLAS-4988).
"""

import os
import shutil
import subprocess

import pytest

import rocisa
from rocisa.container import sgpr, vgpr
from rocisa.instruction import SAddU64, VAddNCU64

# Targets without a native 64-bit add, so both instructions take the lowered path.
_ISAS = [(9, 0, 10), (9, 4, 2), (9, 5, 0)]


def _init(isa):
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    search_path = os.pathsep.join(
        [os.path.join(rocm_path, "bin"), os.path.join(rocm_path, "lib", "llvm", "bin")]
    )
    assembler = shutil.which("amdclang++", path=search_path) or "amdclang++"
    ri = rocisa.rocIsa.getInstance()
    ri.init(isa, assembler, False)
    ri.setKernel(isa, 64)
    return ri


@pytest.fixture(params=_ISAS, ids=lambda isa: rocisa.isaToGfx(isa))
def lowered_isa(request):
    ri = _init(request.param)
    caps = ri.getAsmCaps()
    if caps["s_add_u64"] or caps["v_add_nc_u64"]:
        pytest.skip(f"{rocisa.isaToGfx(request.param)} has a native 64-bit add")
    yield request.param
    _init((9, 4, 2))


def _lines(inst):
    """The emitted instructions, without comments, as (mnemonic, operands) pairs."""
    out = []
    for line in str(inst).splitlines():
        line = line.split("//")[0].strip()
        if line:
            mnemonic, _, rest = line.partition(" ")
            out.append((mnemonic, [o.strip() for o in rest.split(",")]))
    return out


def _as_int(text):
    return int(text, 0)


# (immediate, expected low 32 bits as a signed value, expected high 32 bits)
_IMMEDIATES = [
    (5, 5, 0),
    (0, 0, 0),
    (-8, -8, -1),
    (0x7FFFFFFF, 0x7FFFFFFF, 0),
    (-(2**31), -(2**31), -1),
    # Above 2^31: Python passes these through as doubles, which must split as 64-bit integers.
    (2**31 + 5, -(2**31) + 5, 0),
    (2**32 + 3, 3, 1),
    (3 * 2**32 + 7, 7, 3),
    # The largest magnitudes a double carries exactly.
    (2**53 - 1, -1, 2**21 - 1),
    (-(2**53 - 1), 1, -(2**21)),
]


def test_saddu64_registers_lower_to_a_carry_chain(lowered_isa):
    lines = _lines(SAddU64(dst=sgpr(4, 2), src0=sgpr(6, 2), src1=sgpr(8, 2)))
    assert lines == [("s_add_u32", ["s4", "s6", "s8"]), ("s_addc_u32", ["s5", "s7", "s9"])]


@pytest.mark.parametrize("imm,low,high", _IMMEDIATES)
def test_saddu64_immediate_high_half_is_the_sign_extension(lowered_isa, imm, low, high):
    lines = _lines(SAddU64(dst=sgpr(4, 2), src0=sgpr(4, 2), src1=imm))
    assert [m for m, _ in lines] == ["s_add_u32", "s_addc_u32"]
    (_, lo_ops), (_, hi_ops) = lines
    assert lo_ops[:2] == ["s4", "s4"] and hi_ops[:2] == ["s5", "s5"]
    assert _as_int(lo_ops[2]) == low
    assert _as_int(hi_ops[2]) == high, f"high half of {imm:#x} must be {high}, got {hi_ops[2]}"


def test_vaddncu64_registers_lower_to_a_carry_chain(lowered_isa):
    lines = _lines(VAddNCU64(dst=vgpr(4, 2), src0=vgpr(6, 2), src1=vgpr(8, 2)))
    assert [m for m, _ in lines] == ["v_add_co_u32", "v_addc_co_u32"]
    (_, lo_ops), (_, hi_ops) = lines
    assert lo_ops[0] == "v4" and lo_ops[2:] == ["v6", "v8"]
    assert hi_ops[0] == "v5" and hi_ops[2:4] == ["v7", "v9"]
    # The low add writes the carry and the high add reads the same carry register.
    assert lo_ops[1] == hi_ops[1] == hi_ops[4]


def _inline(v):
    return -16 <= v <= 64


@pytest.mark.parametrize("imm,low,high", _IMMEDIATES)
def test_vaddncu64_immediate_high_half_is_the_sign_extension(lowered_isa, imm, low, high):
    inst = VAddNCU64(dst=vgpr(4, 2), src0=vgpr(4, 2), src1=imm)
    if not _inline(high):
        # v_addc_co_u32 reads vcc, which leaves no room for a literal next to it.
        with pytest.raises(Exception, match="not an inline constant"):
            str(inst)
        return
    lines = _lines(inst)
    assert [m for m, _ in lines] == ["v_add_co_u32", "v_addc_co_u32"]
    (_, lo_ops), (_, hi_ops) = lines
    # VOP2 takes a literal only in src0, so the immediate comes first.
    assert _as_int(lo_ops[2]) == low and lo_ops[3] == "v4"
    assert _as_int(hi_ops[2]) == high and hi_ops[3] == "v5"


def _assemble(isa, text, tmp_path):
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    search_path = os.pathsep.join(
        [os.path.join(rocm_path, "bin"), os.path.join(rocm_path, "lib", "llvm", "bin")]
    )
    clang = shutil.which("amdclang++", path=search_path) or shutil.which("amdclang++")
    if not clang:
        pytest.skip("amdclang++ not found")
    src = tmp_path / "lowered.s"
    src.write_text(text + "\n")
    return subprocess.run(
        [clang, "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
         f"-mcpu={rocisa.isaToGfx(isa)}", "-c", str(src), "-o", str(tmp_path / "lowered.o")],
        capture_output=True, text=True,
    )  # fmt: skip


# String checks miss an operand the hardware cannot encode; the assembler does not.
@pytest.mark.parametrize("imm,low,high", _IMMEDIATES)
def test_lowered_immediate_adds_assemble(lowered_isa, imm, low, high, tmp_path):
    insts = [SAddU64(dst=sgpr(4, 2), src0=sgpr(4, 2), src1=imm)]
    if _inline(high):
        insts.append(VAddNCU64(dst=vgpr(4, 2), src0=vgpr(4, 2), src1=imm))
    text = "\n".join(str(i) for i in insts)
    result = _assemble(lowered_isa, text, tmp_path)
    assert result.returncode == 0, f"{text}\n{result.stderr}"


def test_symbolic_immediate_is_rejected(lowered_isa):
    with pytest.raises(Exception, match="cannot split"):
        str(SAddU64(dst=sgpr(4, 2), src0=sgpr(4, 2), src1="SomeSymbol"))


# A double cannot tell these from a neighbor: 2^53 + 1 and 2^63 - 1 round to 2^53 and 2^63.
@pytest.mark.parametrize("imm", [2**53, 2**53 + 1, -(2**53), 2**63 - 1, -(2**63), 2**63])
def test_immediate_a_double_cannot_hold_exactly_is_rejected(lowered_isa, imm):
    with pytest.raises(Exception, match="cannot split"):
        str(SAddU64(dst=sgpr(4, 2), src0=sgpr(4, 2), src1=imm))
