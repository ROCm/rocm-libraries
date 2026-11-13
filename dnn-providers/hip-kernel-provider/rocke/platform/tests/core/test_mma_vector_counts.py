# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""MMA result/padded bridges reject malformed counts before type equality."""

from types import SimpleNamespace

import pytest
from rocke.core.arch import ArchTarget
from rocke.core.ir import I32, Op, Value, VectorType
from rocke.core.isa.matrix_operand_forms import ZeroPadded
from rocke.core.mma_operand import mma_result_needs_reinterpret


@pytest.mark.parametrize("bridge", ["result", "padded"])
@pytest.mark.parametrize("count", [4.0, True, False, "4", None])
@pytest.mark.parametrize("field", ["value", "register", "both"])
def test_bridges_share_integer_count_admission(bridge, count, field):
    op_id = "wmma_i32_16x16x16_iu8"
    valid = VectorType(I32, 4)
    malformed = VectorType(I32, count)
    actual = malformed if field in ("value", "both") else valid
    register = malformed if field in ("register", "both") else valid
    arch = ArchTarget.from_gfx("gfx1151")
    kind = (
        ("result" if bridge == "result" else "operand")
        if field != "register"
        else "register"
    )
    label = (
        f"MMA result of {op_id!r}"
        if bridge == "result"
        else f"MMA operand a of {op_id!r}"
    )
    with pytest.raises(ValueError) as caught:
        if bridge == "result":
            mma_result_needs_reinterpret(op_id, actual, register, arch)
        else:
            # Construct a valid form, then inject malformed register metadata
            # so the same check covers both values at the final admission gate.
            form = ZeroPadded(valid, 4)
            object.__setattr__(form, "register_type", register)
            op = Op(name=f"tile.{op_id}", operands=[Value("%a", actual)])
            form.check_operand(SimpleNamespace(_flavor="llvm24"), op, "a")
    assert str(caught.value) == (
        f"{label} requires an integer {kind} vector count, got {count!r}"
    )
