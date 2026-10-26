# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The llvm-mc cross-check driver (`tools/isa_xcheck.py`): operands and
instructions print in LLVM syntax from the description, variant lines parse,
and the report counts verdicts. CPU-only: llvm-mc itself does not run."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PLATFORM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLATFORM / "tools"))

import isa_xcheck

CRATE = PLATFORM / "rust" / "rocke-codeobj"


def variant(
    name: str, defs: list[str], uses: list[str], imm: int = 0, mods: int = 0
) -> isa_xcheck.Variant:
    return isa_xcheck.Variant(0, name, None, defs, uses, imm, mods, "", "")


def test_operands_print_in_llvm_syntax():
    assert isa_xcheck.reg("v4x2") == "v[4:5]"
    assert isa_xcheck.reg("v7x1", half=True) == "v7.h"
    assert isa_xcheck.reg("ifffffff0") == "-16"
    assert isa_xcheck.reg("i3f800000", float_text=True) == "1.0"
    assert (
        isa_xcheck.dpp16_text(0x00FF_08B1)
        == "quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1"
    )
    assert not isa_xcheck.float_source("v_cvt_f32_i32", 0)
    assert isa_xcheck.float_source("v_fma_f32", 2)
    assert isa_xcheck.float_source("v_ldexp_f64", 0)
    assert not isa_xcheck.float_source("v_ldexp_f64", 1)


def test_variant_lines_parse():
    (v,) = isa_xcheck.parse_variants(
        "3\tSAddU32\t-\ts4x1\ts6x1,i00000005\t0\t0\t80048506\t80048506\n"
    )
    assert (v.index, v.name, v.code, v.defs, v.uses, v.imm, v.mods) == (
        3,
        "SAddU32",
        None,
        ["s4x1"],
        ["s6x1", "i00000005"],
        0,
        0,
    )
    assert v.words() == [0x80048506]


def test_instructions_render_from_the_description():
    info = isa_xcheck.load_info(CRATE)
    vcmp, vopd = isa_xcheck.vcmp_names(), isa_xcheck.vopd_names(CRATE)
    assert vopd[7] == "v_dual_mul_dx9_zero_f32"
    sop2 = variant("SAddU32", ["s4x1"], ["s6x1", "i00000005"])
    assert isa_xcheck.render(sop2, info, vcmp, vopd) == "s_add_u32 s4, s6, 5"
    dual = variant(
        "VDual", ["v0x1", "v3x1"], ["v1x1", "v2x1", "v6x1", "v7x1"], imm=7 << 8, mods=2
    )
    assert (
        isa_xcheck.render(dual, info, vcmp, vopd)
        == "v_dual_fmac_f32 v0, v1, v2 :: v_dual_mul_dx9_zero_f32 v3, v6, v7"
    )
    dual.imm = 99 << 8
    with pytest.raises(isa_xcheck.Unprintable):
        isa_xcheck.render(dual, info, vcmp, vopd)


def test_report_counts_verdicts(tmp_path):
    vs = [
        variant("SAddU32", ["s4x1"], ["s6x1", "s7x1"]),
        variant("SAddU32", ["s4x1"], ["s6x1", "i3f800000"]),
    ]
    vs[0].verdict, vs[1].verdict = "match", "unprintable"
    text = isa_xcheck.report(vs, tmp_path / "report.md")
    assert (
        "| match | 1 |" in text
        and "| unprintable | 1 |" in text
        and "| mismatch | 0 |" in text
    )
    assert (tmp_path / "report.md").read_text(encoding="utf-8") == text
