# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""VOPD (dual issue VALU) rules for VDualOp, per ISA family.

RDNA3/3.5 (gfx11, RDNA3.5 ISA 7.6) and RDNA4 (gfx120x, RDNA4 ISA 7.8) differ in
which VGPRs two halves may share, in which half may write the other's sources,
and in the max/min mnemonics. vopdCheck models each family; dualOrSingle uses it
to pair two instructions only where the target allows it.

Every rule has one legal and one illegal case. A rule that differs by family runs
on both families with its two verdicts; bank rules also run on both, since the
gfx120x shared-VGPR exception sits next to them. Other rules run once. Where the
assembler checks a rule too, its verdict must match the model's.
"""

import os
import shutil
import subprocess
from copy import deepcopy

import pytest

import rocisa
from rocisa.container import DPPModifiers, SDWAModifiers, VOP3PModifiers, sgpr, vgpr
from rocisa.enum import HighBitSel
from rocisa.instruction import (
    VAddF32,
    VAddU32,
    VAndB32,
    VCndMaskB32,
    VDot2F32BF16,
    VDot2F32F16,
    VDualOp,
    VLShiftLeftB32,
    VMacF32,
    VMaxF32,
    VMinF32,
    VMulF32,
    VSubF32,
    dualOrSingle,
    vopdCheck,
)

GFX11 = (11, 5, 1)
GFX12 = (12, 0, 1)


def _init(isa, wavefront):
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    search_path = os.pathsep.join(
        [os.path.join(rocm_path, "bin"), os.path.join(rocm_path, "lib", "llvm", "bin")]
    )
    assembler = shutil.which("amdclang++", path=search_path) or shutil.which("amdclang++") or "amdclang++"
    ri = rocisa.rocIsa.getInstance()
    ri.init(isa, assembler, False)
    ri.setKernel(isa, wavefront)


@pytest.fixture
def isa():
    """Call with (isa, wavefront); restores the gfx942 default afterwards."""
    yield _init
    _init((9, 4, 2), 64)


def _clang():
    """amdclang++, or skip: without it rocisa cannot probe the assembler caps
    (v_add_nc_u32, v_fmac_f32) that decide some VOPD forms."""
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    search_path = os.pathsep.join(
        [os.path.join(rocm_path, "bin"), os.path.join(rocm_path, "lib", "llvm", "bin")]
    )
    clang = shutil.which("amdclang++", path=search_path) or shutil.which("amdclang++")
    if not clang:
        pytest.skip("amdclang++ not found")
    return clang


def _assemble(isa, wavefront, text, tmp_path):
    clang = _clang()
    src = tmp_path / "vopd.s"
    src.write_text(text + "\n")
    wave = ["-mwavefrontsize64"] if wavefront == 64 else []
    return subprocess.run(
        [clang, "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
         f"-mcpu={rocisa.isaToGfx(isa)}", *wave, "-c", str(src), "-o", str(tmp_path / "vopd.o")],
        capture_output=True, text=True,
    )  # fmt: skip


def _neg(r):
    r.setMinus(True)
    return r


def _abs(r):
    r.setAbs(True)
    return r


def _half(r):
    r.setHalfSelect(HighBitSel.LOW)
    return r


# A pair is ((opX, dstX, src0X, src1X), (opY, dstY, src0Y, src1Y)), built lazily
# so operand objects are fresh per test.
def _x(op="mul_f32", d=0, s0=2, s1=4):
    return lambda: (op, _op(d), _op(s0), _op(s1))


def _y(op="add_f32", d=1, s0=3, s1=5):
    return lambda: (op, _op(d), _op(s0), _op(s1))


def _op(v):
    """An int is a VGPR index; wrap an immediate in a lambda."""
    if callable(v):
        return v()
    return vgpr(v) if isinstance(v, int) else v


_LIT = 0x1234

# (id, X, Y, {isa or (isa, wave): legal}, assembler checks this rule)
_RULES = [
    # Rules that differ by family.
    ("shared_src0_vgpr", _x(s0=2), _y(s0=2), {GFX11: False, GFX12: True}, True),
    ("shared_vsrc1_vgpr", _x(s1=4), _y(s1=4), {GFX11: False, GFX12: True}, True),
    ("x_writes_y_src0", _x(d=3), _y(d=0, s0=3), {GFX11: True, GFX12: False}, False),
    ("x_writes_y_vsrc1", _x(d=5), _y(d=0, s1=5), {GFX11: True, GFX12: False}, False),
    ("y_writes_x_src0", _x(s0=3), _y(d=3, s0=2), {GFX11: True, GFX12: True}, True),
    # Banks, on both families.
    ("src0_same_bank", _x(s0=2), _y(s0=6), {GFX11: False, GFX12: False}, True),
    ("vsrc1_same_bank", _x(s1=4), _y(s1=8), {GFX11: False, GFX12: False}, True),
    # Shared rules, once.
    ("dst_same_parity", _x(), _y(d=6), {GFX11: False}, True),
    ("dst_x_odd_y_even", _x(d=1), _y(d=0), {GFX11: True}, True),
    ("src0_sgpr_vs_vgpr_same_index", _x(s0=lambda: sgpr(2)), _y(s0=2), {GFX11: True}, True),
    ("vsrc1_sgpr", _x(s1=lambda: sgpr(4)), _y(), {GFX11: False}, True),
    ("vcc_and_one_sgpr_twice", _x("cndmask_b32", s0=lambda: sgpr(2)), _y(s0=lambda: sgpr(2)),
     {GFX11: True}, True),
    ("vcc_and_two_sgprs", _x("cndmask_b32", s0=lambda: sgpr(2)), _y(s0=lambda: sgpr(3)),
     {GFX11: False}, True),
    ("vcc_sgpr_and_exec", _x("cndmask_b32", s0=lambda: sgpr(2)), _y(s0="exec_lo"),
     {GFX12: False}, True),
    ("vcc_sgpr_and_inline", _x("cndmask_b32", s0=lambda: sgpr(2)), _y(s0=1.0), {GFX11: True}, True),
    ("sgpr_and_literal", _x(s0=lambda: sgpr(2)), _y(s0=lambda: _LIT), {GFX11: True}, True),
    ("shared_literal", _x(s0=lambda: _LIT), _y(s0=lambda: _LIT), {GFX11: True}, True),
    ("two_literals", _x(s0=lambda: _LIT), _y(s0=lambda: _LIT + 1), {GFX11: False}, True),
    ("vcc_sgpr_and_literal", _x("cndmask_b32", s0=lambda: sgpr(2)), _y(s0=lambda: _LIT),
     {GFX11: False}, True),
    ("neg_operand", _x(s0=lambda: _neg(vgpr(2))), _y(), {GFX11: False}, True),
    ("abs_operand", _x(s0=lambda: _abs(vgpr(2))), _y(), {GFX11: False}, True),
    ("true16_operand", _x(s0=lambda: _half(vgpr(2))), _y(), {GFX11: False}, True),
    ("wave64", _x(), _y(), {(GFX11, 64): False}, True),
    # CDNA5 (gfx1250) also has VOPD but is not modelled yet.
    ("gfx1250_not_modelled", _x(), _y(), {(12, 5, 0): False}, False),
    ("gfx942_no_vopd", _x(), _y(), {(9, 4, 2): False}, True),
    # Opcode slots. mov_b32 is VOPD but has one source: VDualOp cannot encode it.
    ("mov_b32_one_source", _x("mov_b32"), _y(), {GFX11: False}, False),
    ("sub_nc_u32_not_vopd", _x(), _y("sub_nc_u32"), {GFX11: False}, True),
]

_XY_OPS = ["fmac_f32", "mul_f32", "add_f32", "sub_f32", "subrev_f32", "mul_dx9_zero_f32",
           "cndmask_b32", "dot2acc_f32_f16", "dot2acc_f32_bf16"]  # fmt: skip
_Y_ONLY = ["add_nc_u32", "lshlrev_b32", "and_b32"]
_MAX_MIN = {GFX11: ["max_f32", "min_f32"], GFX12: ["max_num_f32", "min_num_f32"]}
for _isa, _other in ((GFX11, GFX12), (GFX12, GFX11)):
    _g = rocisa.isaToGfx(_isa)
    for _o in _XY_OPS + _MAX_MIN[_isa]:
        _RULES.append((f"{_o}_in_x_{_g}", _x(_o), _y(), {_isa: True}, True))
        _RULES.append((f"{_o}_in_y_{_g}", _x(), _y(_o), {_isa: True}, True))
    for _o in _Y_ONLY:
        _RULES.append((f"{_o}_in_y_{_g}", _x(), _y(_o), {_isa: True}, True))
        _RULES.append((f"{_o}_in_x_{_g}", _x(_o), _y(), {_isa: False}, True))
    for _o in _MAX_MIN[_other]:
        _RULES.append((f"{_o}_in_x_{_g}", _x(_o), _y(), {_isa: False}, True))

_CASES = []
for _id, _xf, _yf, _verdicts, _asm in _RULES:
    for _key, _legal in _verdicts.items():
        _target, _wave = _key if isinstance(_key[0], tuple) else (_key, 32)
        _name = f"{_id}-{rocisa.isaToGfx(_target)}" + ("-w64" if _wave == 64 else "")
        _CASES.append(pytest.param(_xf, _yf, _target, _wave, _legal, _asm, id=_name))


def _text(x, y):
    s = lambda v: v if isinstance(v, str) else str(v)  # noqa: E731
    return f"v_dual_{x[0]} {s(x[1])}, {s(x[2])}, {s(x[3])} :: v_dual_{y[0]} {s(y[1])}, {s(y[2])}, {s(y[3])}"


@pytest.mark.parametrize("xf,yf,target,wave,legal,asm", _CASES)
def test_vopd_rules(isa, xf, yf, target, wave, legal, asm, tmp_path):
    isa(target, wave)
    x, y = xf(), yf()
    reason = vopdCheck(*x, *y)
    assert (reason == "") == legal, f"{_text(x, y)}: {reason!r}"
    if asm:
        result = _assemble(target, wave, _text(x, y), tmp_path)
        assert (result.returncode == 0) == legal, f"{_text(x, y)}\n{result.stderr}"


def test_vdual_op_rejects_illegal_pair(isa):
    isa(GFX11, 32)
    x, y = _x()(), _y(d=6)()
    with pytest.raises(Exception, match="one even and one odd"):
        VDualOp(*x, *y)
    assert "one even and one odd" in vopdCheck(*x, *y)


@pytest.mark.parametrize("field,value", [("dpp", DPPModifiers(row_shr=1)),
                                         ("sdwa", SDWAModifiers()),
                                         ("vop3", VOP3PModifiers(op_sel=[0, 1]))])  # fmt: skip
def test_vdual_op_rejects_modifiers_set_after_construction(isa, field, value):
    isa(GFX11, 32)
    inst = VDualOp(*_x()(), *_y()())
    setattr(inst, field, value)
    with pytest.raises(Exception, match="no DPP, SDWA or VOP3"):
        str(inst)


# dualOrSingle takes two instructions in program order.


def _lines(insts):
    return [str(i).split("//")[0].strip() for i in insts]


def test_dual_or_single_pairs_where_the_target_allows(isa):
    isa(GFX11, 32)
    out = dualOrSingle(VMulF32(vgpr(0), vgpr(2), vgpr(4)), VAddF32(vgpr(1), vgpr(3), vgpr(5)))
    assert _lines(out) == ["v_dual_mul_f32 v0, v2, v4 :: v_dual_add_f32 v1, v3, v5"]


def test_dual_or_single_keeps_both_on_an_unmodelled_target(isa):
    isa((12, 5, 0), 32)
    first, second = VMulF32(vgpr(0), vgpr(2), vgpr(4)), VAddF32(vgpr(1), vgpr(3), vgpr(5))
    out = dualOrSingle(first, second)
    assert [type(i) for i in out] == [VMulF32, VAddF32]
    assert _lines(out) == ["v_mul_f32 v0, v2, v4", "v_add_f32 v1, v3, v5"]


@pytest.mark.parametrize("target", [GFX11, GFX12], ids=rocisa.isaToGfx)
def test_dual_or_single_keeps_order_when_second_reads_first(isa, target):
    # In a VOPD the second would read the old v0.
    isa(target, 32)
    out = dualOrSingle(VMulF32(vgpr(0), vgpr(2), vgpr(4)), VAddF32(vgpr(1), vgpr(0), vgpr(5)))
    assert _lines(out) == ["v_mul_f32 v0, v2, v4", "v_add_f32 v1, v0, v5"]


@pytest.mark.parametrize("target,paired", [(GFX11, True), (GFX12, False)], ids=["gfx1151", "gfx1201"])
def test_dual_or_single_slot_swap_follows_the_family_rule(isa, target, paired):
    # add_nc_u32 only fits Y, so the mul goes to X; the mul then writes a source
    # of Y (v3), which gfx11 allows (old value) and gfx120x forbids.
    _clang()
    isa(target, 32)
    out = dualOrSingle(VAddU32(vgpr(0), vgpr(3), vgpr(4)), VMulF32(vgpr(3), vgpr(2), vgpr(5)))
    if paired:
        assert _lines(out) == ["v_dual_mul_f32 v3, v2, v5 :: v_dual_add_nc_u32 v0, v3, v4"]
    else:
        assert _lines(out) == ["v_add_nc_u32 v0, v3, v4", "v_mul_f32 v3, v2, v5"]


# Each instruction class with a VOPD form, paired with a v_mul_f32 that fits any
# slot. VSubF32 with an SGPR in src1 needs subrev; VMaxF32 is spelled per family.
_CLASSES = [
    (lambda: VMulF32(vgpr(1), vgpr(3), vgpr(5)), {GFX11: "mul_f32 v1, v3, v5"}),
    (lambda: VAddF32(vgpr(1), vgpr(3), vgpr(5)), {GFX11: "add_f32 v1, v3, v5"}),
    (lambda: VSubF32(vgpr(1), vgpr(5), sgpr(7)), {GFX11: "subrev_f32 v1, s7, v5"}),
    (lambda: VMaxF32(vgpr(1), vgpr(3), vgpr(5)),
     {GFX11: "max_f32 v1, v3, v5", GFX12: "max_num_f32 v1, v3, v5"}),
    (lambda: VMinF32(vgpr(1), vgpr(3), vgpr(5)),
     {GFX11: "min_f32 v1, v3, v5", GFX12: "min_num_f32 v1, v3, v5"}),
    (lambda: VAddU32(vgpr(1), vgpr(3), vgpr(5)), {GFX11: "add_nc_u32 v1, v3, v5"}),
    (lambda: VAndB32(vgpr(1), vgpr(3), vgpr(5)), {GFX11: "and_b32 v1, v3, v5"}),
    (lambda: VLShiftLeftB32(vgpr(1), 4, vgpr(5)), {GFX11: "lshlrev_b32 v1, 4, v5"}),
    (lambda: VMacF32(vgpr(1), vgpr(3), vgpr(5)), {GFX11: "fmac_f32 v1, v3, v5"}),
    (lambda: VCndMaskB32(vgpr(1), vgpr(3), vgpr(5)), {GFX11: "cndmask_b32 v1, v3, v5"}),
    (lambda: VDot2F32F16(vgpr(1), vgpr(3), vgpr(5), vgpr(1)), {GFX11: "dot2acc_f32_f16 v1, v3, v5"}),
    (lambda: VDot2F32BF16(vgpr(1), vgpr(3), vgpr(5), vgpr(1)),
     {GFX11: "dot2acc_f32_bf16 v1, v3, v5"}),
]
_CLASS_CASES = [
    pytest.param(make, target, half, id=f"{half.split()[0]}-{rocisa.isaToGfx(target)}")
    for make, halves in _CLASSES
    for target, half in halves.items()
]


@pytest.mark.parametrize("make,target,half", _CLASS_CASES)
def test_dual_or_single_maps_each_class(isa, make, target, half, tmp_path):
    _clang()
    isa(target, 32)
    out = dualOrSingle(VMulF32(vgpr(0), vgpr(2), vgpr(4)), make())
    assert len(out) == 1 and isinstance(out[0], VDualOp), _lines(out)
    halves = [h.strip() for h in _lines(out)[0].split("::")]
    assert f"v_dual_{half}" in halves
    result = _assemble(target, 32, _lines(out)[0], tmp_path)
    assert result.returncode == 0, f"{_lines(out)[0]}\n{result.stderr}"


def test_dual_or_single_keeps_an_instruction_with_modifiers(isa):
    isa(GFX11, 32)
    second = VAddF32(vgpr(1), vgpr(3), vgpr(5))
    second.dpp = DPPModifiers(row_shr=1)
    out = dualOrSingle(VMulF32(vgpr(0), vgpr(2), vgpr(4)), second)
    assert [type(i) for i in out] == [VMulF32, VAddF32]


def test_vdual_op_emission(isa):
    isa(GFX11, 32)
    inst = VDualOp("mul_f32", vgpr(0), vgpr(2), vgpr(4), "add_nc_u32", vgpr(1), vgpr(3), vgpr(5),
                   comment="pair")  # fmt: skip
    assert str(inst).split("//")[0].rstrip() == (
        "v_dual_mul_f32 v0, v2, v4 :: v_dual_add_nc_u32 v1, v3, v5"
    )
    assert "// pair" in str(inst)
    copy = deepcopy(inst)
    inst.setSrc(2, vgpr(7))
    assert (copy.opX, copy.opY) == ("mul_f32", "add_nc_u32")
    assert str(copy) == str(VDualOp("mul_f32", vgpr(0), vgpr(2), vgpr(4), "add_nc_u32", vgpr(1),
                                    vgpr(3), vgpr(5), comment="pair"))  # fmt: skip
