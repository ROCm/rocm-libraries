# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Offline tests for the LSE store epilogue factories (no GPU needed)."""

from __future__ import annotations

import re

import pytest
from ._lse_store_support import factory_for

import _attention_fwd_ext_harness as H
from kernels.common import _lse_store as L
from rocke.core.ir import F32, I32, IRBuilder, PtrType
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower
from rocke.helpers.attention_fwd_ext import RowEpilogue

ARCHS = ("gfx942", "gfx950", "gfx1151", "gfx1201")
SLOTS = {"gfx942": 4, "gfx950": 4, "gfx1151": 8, "gfx1201": 8}


def _ir(arch, flavor="llvm22", **kw):
    kernel, _ = H.make_probe_kernel("lse_ir", arch=arch, head_size=64, **kw)
    return lower(kernel, arch=arch, llvm_flavor=flavor)


def _calls(ir, intrinsic):
    return len(re.findall(r"call [^\n]*@llvm\." + re.escape(intrinsic) + r"\(", ir))


@pytest.mark.parametrize("flavor", ("llvm20", "llvm22", "llvm23"))
@pytest.mark.parametrize("arch", ARCHS)
def test_helper_hook_lowers_with_one_log2_per_slot_and_no_scratch(arch, flavor):
    ir = _ir(arch, flavor, lse_hook_factory=factory_for(arch))
    assert _calls(ir, "log2.f32") == SLOTS[arch]
    assert "alloca" not in ir


@pytest.mark.parametrize("arch", ARCHS)
def test_helper_matches_the_inline_store_emission(arch):
    inline = _ir(arch)
    helper = _ir(arch, lse_hook_factory=factory_for(arch))
    for token in ("log2.f32", "exp2.f32"):
        assert _calls(inline, token) == _calls(helper, token)
    assert inline.count("fcmp ogt") == helper.count("fcmp ogt")
    assert inline.count("store float") == helper.count("store float")


@pytest.mark.parametrize("arch", ARCHS)
def test_strided_layouts_differ_only_in_index_math(arch):
    bhs = _ir(arch, lse_hook_factory=factory_for(arch, "bhs"))
    bsh = _ir(arch, lse_hook_factory=factory_for(arch, "bsh"))
    assert _calls(bhs, "log2.f32") == _calls(bsh, "log2.f32")
    assert bhs.count("store float") == bsh.count("store float")


def _row(b, slot):
    one = b.const_f32(1.0)
    ok = b.cmp_lt(b.const_i32(0), b.const_i32(1))
    return RowEpilogue(slot, b.const_i32(3), ok, ok, one, one, ok)


def _bare(name):
    b = IRBuilder(name)
    lse = b.param("LSE", PtrType(F32, "global"))
    head = b.param("head", I32)
    return b, lse, head


def test_slot_beyond_the_body_row_count_is_rejected():
    b, lse, head = _bare("lse_unit")
    hook = L.make_mfma_lse_epilogue(b, lse, head_idx=head, row_stride=1, head_stride=8)
    hook(b, _row(b, 3))
    with pytest.raises(ValueError, match="slot 4"):
        hook(b, _row(b, 4))
    wide = L.make_wmma_lse_epilogue(b, lse, head_idx=head, row_stride=1, head_stride=8)
    wide(b, _row(b, 7))
    with pytest.raises(ValueError, match="slot 8"):
        wide(b, _row(b, 8))


def test_hook_returns_the_extra_validity_unchanged():
    b, lse, head = _bare("lse_unit2")
    extra = b.cmp_lt(b.const_i32(1), b.const_i32(2))
    seen = []

    def extra_valid(bb, e):
        seen.append(e.slot)
        return extra

    hook = L.make_mfma_lse_epilogue(
        b, lse, head_idx=head, row_stride=1, head_stride=8, extra_valid=extra_valid
    )
    assert hook(b, _row(b, 0)) is extra
    assert seen == [0]
    plain = L.make_mfma_lse_epilogue(b, lse, head_idx=head, row_stride=1, head_stride=8)
    assert plain(b, _row(b, 0)) is None


def test_batch_arguments_must_come_together():
    b, lse, head = _bare("lse_unit3")
    with pytest.raises(ValueError, match="together"):
        L.make_mfma_lse_epilogue(
            b, lse, head_idx=head, row_stride=1, head_stride=1, batch_idx=head
        )


def test_batch_rebase_is_emitted_once_with_a_64_bit_offset():
    b, lse, head = _bare("lse_unit4")
    batch = b.param("batch", I32)
    bs = b.param("bs", I32)
    hook = L.make_mfma_lse_epilogue(
        b,
        lse,
        head_idx=head,
        row_stride=64,
        head_stride=1,
        batch_idx=batch,
        batch_stride=bs,
        row_base=bs,
    )
    for s in range(4):
        hook(b, _row(b, s))
    b.ret()
    ir = lower(b.kernel, arch="gfx942", llvm_flavor="llvm22")
    assert ir.count("getelementptr inbounds i8") == 1
    assert ir.count("store float") == 4


def test_no_exported_helper_looks_like_a_builder():
    assert not [n for n in L.__all__ if n.startswith("build_")]
