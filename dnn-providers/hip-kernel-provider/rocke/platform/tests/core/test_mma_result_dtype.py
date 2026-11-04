# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""MMA instructions whose result dtype differs from their accumulator.

A catalog row records the result dtype as ``dst`` only where it differs from
the accumulator ``c``; :class:`~rocke.core.arch.MmaOp` exposes it as
``dst_dtype``, equal to ``c_dtype`` everywhere else. The queries of
:class:`~rocke.core.arch.MmaCatalog` take ``dst_dtype``; leaving it out means
"equal to the accumulator", so every query that names no result dtype keeps
its answer. The one such instruction today is gfx1250
``wmma_gfx1250_bf16f32_16x16x32_bf16``: bf16 x bf16, an fp32 accumulator and a
bf16 result. These tests pin:

- that every other catalog op's result dtype is its accumulator's;
- the gfx1250 query answers with and without ``dst_dtype``;
- that ``IRBuilder.mma`` sizes the result from the result dtype, for an op and
  for a bare op_id;
- the lowered call in both engines at every LLVM flavor: an fp32 accumulator
  in, a bf16 result out, every modifier 0;
- that the module compiles and links for gfx1250 and selects the instruction
  (compile-verified, not run; needs a compiler that accepts gfx1250).

The C++ engine's rows and queries are pinned to the same answers by
``tests/core/mma_result_dtype.cpp``.
"""

from __future__ import annotations

import pytest
from rocke.core.arch import ArchTarget, known_arches
from rocke.core.ir import BF16, F32, IRBuilder, PtrType, VectorType
from rocke.core.lower_llvm import LLVM_FLAVORS
from rocke.tests.engines import lower_with_both_engines
from rocke.tests.ir_validity import assert_target_accepts, target_mnemonics

_OP_ID = "wmma_gfx1250_bf16f32_16x16x32_bf16"
_F32_RESULT_OP_ID = "wmma_gfx1250_f32_16x16x32_bf16"
_INTRINSIC = "llvm.amdgcn.wmma.bf16f32.16x16x32.bf16.v8bf16.v16bf16.v8f32"
_QUERY = {"family": "wmma", "a_dtype": "bf16", "b_dtype": "bf16", "c_dtype": "fp32"}


def _catalog():
    return ArchTarget.from_gfx("gfx1250").mma


def _kernel(op):
    """One MMA between bf16 operands loaded from memory and a zero fp32
    accumulator, every result element stored, so that optimization keeps the
    instruction. ``op`` is an MmaOp or a bare op_id."""
    b = IRBuilder("gfx1250_bf16f32")
    b.kernel.attrs["max_workgroup_size"] = 32
    operands = []
    for name in ("A", "B"):
        ptr = b.param(name, PtrType(BF16, "global"))
        index = b.mul(b.thread_id_x(), b.const_i32(16))
        operands.append(b.global_load_vN(ptr, index, BF16, 16))
    out = b.param("D", PtrType(BF16, "global"))
    d = b.mma(op, *operands, b.zero_vec(F32, 8))
    base = b.mul(b.thread_id_x(), b.const_i32(8))
    for slot in range(8):
        b.global_store(out, b.add(base, b.const_i32(slot)), b.vec_extract(d, slot))
    b.ret()
    return b.kernel, d


def test_every_other_op_returns_its_accumulator_dtype():
    differing = {
        (gfx, op.op_id)
        for gfx in known_arches()
        for op in ArchTarget.from_gfx(gfx).mma.ops
        if op.dst_dtype != op.c_dtype
    }
    assert differing == {("gfx1250", _OP_ID)}


def test_the_catalog_row_names_an_fp32_accumulator_and_a_bf16_result():
    op = _catalog().by_op_id(_OP_ID)
    assert (op.family, op.a_dtype, op.b_dtype, op.c_dtype, op.dst_dtype) == (
        "wmma",
        "bf16",
        "bf16",
        "fp32",
        "bf16",
    )
    assert (op.m, op.n, op.k) == (16, 16, 32)
    assert (op.a_frag_len, op.b_frag_len, op.c_frag_len, op.wave_size) == (
        16,
        16,
        8,
        32,
    )


@pytest.mark.parametrize(
    "dst_dtype, expected",
    [(None, _F32_RESULT_OP_ID), ("fp32", _F32_RESULT_OP_ID), ("bf16", _OP_ID)],
)
def test_a_query_selects_the_bf16_result_only_when_it_names_it(dst_dtype, expected):
    """Without a result dtype a query means "equal to the accumulator", so the
    fp32-result instruction keeps every answer it had."""
    catalog = _catalog()
    query = dict(_QUERY, dst_dtype=dst_dtype, m=16, n=16)
    assert [op.op_id for op in catalog.enumerate(**query)] == [expected]
    assert catalog.select_largest_k(**query).op_id == expected
    assert catalog.select_largest_k(**query, k_max=32).op_id == expected
    assert catalog.op_for_shape(**query, k=32).op_id == expected
    assert catalog.has_shape(**query, k=32)
    assert not catalog.has_shape(**query, k=16)


def test_a_result_dtype_no_instruction_returns_selects_nothing():
    catalog = _catalog()
    assert catalog.enumerate(**_QUERY, dst_dtype="fp16") == []
    assert catalog.select_largest_k(**_QUERY, dst_dtype="fp16", m=16, n=16) is None


@pytest.mark.parametrize("by_op_id", [False, True], ids=["mma_op", "op_id"])
def test_the_ir_result_is_eight_bf16_values(by_op_id):
    op = _catalog().by_op_id(_OP_ID)
    _, d = _kernel(op.op_id if by_op_id else op)
    assert d.type == VectorType(BF16, 8)


@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
def test_the_call_takes_an_fp32_accumulator_and_returns_bf16(flavor):
    """Both engines emit one call: bf16 A and B, every modifier 0, the eight
    fp32 accumulator values, and eight bf16 results bound to the IR result
    without a conversion."""
    kernel, _ = _kernel(_catalog().by_op_id(_OP_ID))
    text = lower_with_both_engines(kernel, "gfx1250", flavor)
    assert (
        f"declare <8 x bfloat> @{_INTRINSIC}(i1 immarg, <16 x bfloat>, i1 immarg, "
        "<16 x bfloat>, i16 immarg, <8 x float>, i1 immarg, i1 immarg)"
    ) in text
    (call,) = [
        line.strip()
        for line in text.splitlines()
        if f"call <8 x bfloat> @{_INTRINSIC}(" in line
    ]
    arguments = call.split(f"@{_INTRINSIC}(", 1)[1].rsplit(")", 1)[0].split(", ")
    assert [a.rsplit(" ", 1)[0] for a in arguments] == [
        "i1",
        "<16 x bfloat>",
        "i1",
        "<16 x bfloat>",
        "i16",
        "<8 x float>",
        "i1",
        "i1",
    ]
    assert [a for a in arguments if a.rsplit(" ", 1)[0] in ("i1", "i16")] == [
        "i1 0",
        "i1 0",
        "i16 0",
        "i1 0",
        "i1 0",
    ]
    result = call.split(" = ", 1)[0]
    assert f"<8 x bfloat> {result}" in text.split(call, 1)[1]


def test_the_op_has_no_lane_maps_yet():
    op = _catalog().by_op_id(_OP_ID)
    for layout in (op.a_layout, op.b_layout, op.c_layout):
        with pytest.raises(NotImplementedError, match="no verified"):
            layout()


def test_gfx1250_compiles_links_and_selects_the_instruction():
    """Lowered at the host's LLVM flavor and compiled for gfx1250; skips, naming
    the reason, where the compiler does not accept gfx1250. Compile-verified,
    not run."""
    kernel, _ = _kernel(_catalog().by_op_id(_OP_ID))
    text = lower_with_both_engines(kernel, "gfx1250")
    assert_target_accepts(text, "gfx1250")
    mnemonics = target_mnemonics(text, "gfx1250")
    assert [m for m in mnemonics if m.startswith("v_wmma")] == [
        "v_wmma_bf16f32_16x16x32_bf16"
    ]
