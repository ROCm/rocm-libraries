# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Every instruction the backend coverage record says rocke issues, driven through TileMma.

The pairs are read from the record (``traits/data/backend_coverage.json``), never written by hand: each
``issuable`` (instruction, target) pair of a family the driver feeds, once per recorded operand-dtype
variant, drives a single-instruction TileMma whose A, B, C and result dtypes are the instruction's own.
The tile must lower to that instruction's LLVM intrinsic, and its result Fragment must carry the
instruction's result dtype (which differs from the accumulator's for one row).

Lowering pins the Python engine (``_lower_kernel_to_llvm_python``); the C++ engine is checked by the
byte-identity gate. Where the host clang targets the arch, the module is also compiled and linked
(compile-verified, never run).
"""

from __future__ import annotations

import re

import pytest

from rocke.core.ir import I8, U8, IRBuilder, PtrType, dtype_to_ir_type
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.helpers.tiling import (
    fill_fragment,
    make_fragment,
    make_tensor_desc,
    make_window,
    store_fragment,
)
from rocke.helpers.tiling.mma import TileMma
from rocke.helpers.tiling.mma.driver import _DRIVEN_FAMILIES
from rocke.helpers.tiling.traits import load_backend_coverage, load_mma_traits

from .test_mma_issue import _compile_or_skip

_CATALOG = load_mma_traits()
_COVERAGE = load_backend_coverage()


def _intrinsic_calls(ir: str, intrinsic: str) -> list[str]:
    """The call lines of ``intrinsic``, with or without LLVM's overload type suffixes (``.v8i32``,
    ``.f16``): another intrinsic that merely starts with the same name does not count.
    """
    pattern = re.compile(rf"@{re.escape(intrinsic)}(\.v?\d*(?:i|f|bf)\d+)*\(")
    return [line for line in ir.splitlines() if "call" in line and pattern.search(line)]


def _driven_issuable_pairs():
    """Every issuable pair of a driven family, once per recorded operand-dtype variant, as
    ``(op_id, target, operand_dtypes)``."""
    for (op_id, target), entries in sorted(_COVERAGE.entries.items()):
        if _CATALOG.get(op_id).family not in _DRIVEN_FAMILIES:
            continue
        for entry in entries:
            if entry.status == "issuable":
                yield op_id, target, entry.operand_dtypes


def _params(pairs):
    for op_id, target, variant in pairs:
        label = f"{op_id}-{target}" + (f"-{'-'.join(variant)}" if variant else "")
        yield pytest.param(op_id, target, variant, id=label)


_PAIRS = list(_driven_issuable_pairs())


def _plan(op_id: str, target: str, operand_dtypes) -> TileMma:
    """A single-instruction TileMma pinned to ``op_id``; a selectable-signedness integer instruction
    takes the operand dtypes of its recorded variant (u8 picks the unsigned form)."""
    dtypes = {}
    if operand_dtypes is not None:
        dtypes = dict(zip("ab", map(dtype_to_ir_type, operand_dtypes)))
    mma = TileMma(target=target, atom_override=op_id, **dtypes)
    assert mma.plan.mfma_count == 1
    return mma


def _zero_fragment(b: IRBuilder, desc, dtype):
    """A zero Fragment of ``dtype``. fill_fragment zeroes every dtype but u8, which core's zero_vec
    has no form for; a u8 zero is the i8 zero vector reinterpreted, bit for bit."""
    fragment = make_fragment(desc, dtype)
    if dtype == U8:
        count = fragment.storage_type.count
        fragment.value = b.bitcast(b.zero_vec(I8, count), fragment.storage_type)
    else:
        fill_fragment(b, fragment, 0)
    return fragment


def _single_instruction_tile(b: IRBuilder, mma: TileMma):
    """Zero A, B and C fragments of the instruction's own dtypes and descriptors, one MMA, the result
    stored to a tensor of the result dtype. Returns the result Fragment."""
    b.kernel.attrs["max_workgroup_size"] = mma.wave_size
    d_ptr = b.param("D", PtrType(mma.d_dtype, "global"), noalias=True, writeonly=True)
    lane = b.thread_id_x()
    zero = b.const_i32(0)
    a = _zero_fragment(b, mma.a_desc, mma.a_dtype)
    bf = _zero_fragment(b, mma.b_desc, mma.b_dtype)
    acc = _zero_fragment(b, mma.c_desc, mma.c_dtype)
    result = mma(b, a, bf, acc)
    traits = mma.traits
    d_td = make_tensor_desc((traits.m, traits.n), (traits.n, 1), mma.d_dtype)
    store_fragment(b, d_ptr, make_window(d_td, (zero, zero)), result, lane)
    b.ret()
    return result


@pytest.mark.parametrize("op_id,target,operand_dtypes", list(_params(_PAIRS)))
def test_issuable_instruction_drives_a_tile_that_lowers_to_it(
    op_id, target, operand_dtypes
) -> None:
    mma = _plan(op_id, target, operand_dtypes)
    b = IRBuilder("issuable_pair")
    result = _single_instruction_tile(b, mma)
    # The result carries the instruction's result dtype, the accumulator's for every row but the
    # bf16-from-f32 one.
    assert result.dtype == mma.d_dtype == dtype_to_ir_type(mma.traits.d_dtype)
    ir = _lower_kernel_to_llvm_python(b.kernel, arch=target)
    assert len(_intrinsic_calls(ir, mma.traits.llvm_intrinsic)) == 1
    _compile_or_skip(b.kernel, target)


def test_issuable_population() -> None:
    # Locks the pair list read from the record, so a record or catalog change is seen here.
    assert len(_PAIRS) == 339
