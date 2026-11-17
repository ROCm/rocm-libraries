# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A global load_fragment that moves one element per load addresses it as store_fragment does.

When the layer does not emit one wide load per run (a dtype it keeps scalar, a stride the run
width does not divide, or a tile that may overhang its tensor), ``load_fragment`` issues one
load per element and leaves merging them to the backend. The backend vectorizers can merge
only loads whose addresses they can relate, so the lane's base index is computed once and every
other element's index is that base plus a compile-time constant (``add nsw base, C``), exactly
the form ``store_fragment`` gives its stores. Rebuilding the full address per element instead
left the backend merging the loads partly or not at all, differently per LLVM version. A masked
load takes the same form, but its per-element bound check still keeps the backend from merging
it.

These tests lower a tile copy (``load_fragment`` then ``store_fragment``) with each engine and
check, in the LLVM IR, that every per-element load reads the base or ``add nsw base, C`` off one
shared base, and that the index each load reads is the element the layout puts in that register,
for every lane. Host only: nothing is compiled or launched.
"""

from __future__ import annotations

import numpy as np
import pytest

from rocke.core.ir import F16, U8
from rocke.tests.engines import ENGINES, Engine, lower_with_engine
from rocke.tests.llvm_ir import KernelIR

from ._copy_kernels import LANES_PER_ROW, CopyCase, copy_kernel

_CASES = [
    # u8 is outside the dtypes the layer moves with wide loads, so its run stays scalar.
    CopyCase("u8_scalar_dtype", U8, run=8),
    # Row stride 257 is odd: no run width divides it, so the f16 run stays scalar.
    CopyCase("f16_odd_row_stride", F16, run=4, row_stride=257),
    # 60 rows is not a multiple of the 8-row tile: every load is masked.
    CopyCase("f16_masked_rows", F16, run=4, rows=60),
]


def _lowered(case: CopyCase, engine: Engine) -> KernelIR:
    kernel, _, _ = copy_kernel(case)
    return KernelIR(lower_with_engine(kernel, "gfx942", engine=engine))


def _load_indices(ir: KernelIR) -> list[str]:
    """The i32 element index of every load from ``%src``, in IR order. A masked load's index is
    ``select %in_bounds, <index>, 0``; its in-bounds arm is returned."""
    indices = []
    for name in ir.find("load"):
        # load <type>, ptr <gep>, align <n>
        _, gep, _ = ir.operands(name)
        # getelementptr inbounds <type>, ptr <pointer>, i32 <index>
        _, pointer, index = ir.operands(gep)
        if pointer != "%src":
            continue
        if ir.opcode(index) == "select":
            select = index
            _, index, out_of_bounds = ir.operands(select)
            assert out_of_bounds == "0", ir.definitions[select]
        indices.append(index)
    return indices


def _expected_indices(case: CopyCase, lanes: np.ndarray) -> list[np.ndarray]:
    """The element index the layout puts in each register, per lane, for the tile at block 0:
    register ``r * run + i`` holds row ``lane_row + lane_rows * r`` and column
    ``lane_column * run + i``."""
    lane_row, lane_column = lanes // LANES_PER_ROW, lanes % LANES_PER_ROW
    return [
        (lane_row + case.lane_rows * repeat) * case.stride
        + lane_column * case.run
        + column_offset
        for repeat in range(case.rows_per_lane)
        for column_offset in range(case.run)
    ]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_every_per_element_load_is_one_lane_base_plus_a_constant(
    case: CopyCase, engine: Engine
) -> None:
    ir = _lowered(case, engine)
    indices = _load_indices(ir)
    assert (
        len(indices) == case.rows_per_lane * case.run
    ), "expected one load per element"
    base = indices[0]
    for index in indices[1:]:
        definition = ir.definitions[index]
        assert definition.startswith("add nsw i32 "), definition
        assert ir.operands(index)[0] == base, definition
        assert ir.operands(index)[1].isdigit(), definition


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_every_per_element_load_reads_its_registers_element(
    case: CopyCase, engine: Engine
) -> None:
    ir = _lowered(case, engine)
    lanes = np.arange(case.wave_size)
    inputs = {
        ir.call_named("llvm.amdgcn.workitem.id.x"): lanes,
        ir.call_named("llvm.amdgcn.workgroup.id.x"): 0,
        ir.call_named("llvm.amdgcn.workgroup.id.y"): 0,
    }
    read = [ir.evaluate(index, inputs) for index in _load_indices(ir)]
    for register, (got, want) in enumerate(zip(read, _expected_indices(case, lanes))):
        np.testing.assert_array_equal(got, want, err_msg=f"register {register}")
