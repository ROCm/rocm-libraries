# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The HIP lowering's refusal of a packed atomic add whose value is not a pair.

The HIP lowering addresses a ``global_atomic_add_pk_f16`` / ``_bf16`` value as
one packed ``<2 x elem>``. Serialized IR reaches the lowering without passing
the builder, so the lowering checks the value's type with the builder's rule.
Both engines must refuse with the same type and text.
"""

from __future__ import annotations

import pytest
from rocke.core.ir import BF16, F16, IRBuilder, KernelDef, PtrType, Type, VectorType
from rocke.core.ir_serialize import serialize
from rocke.tests.engines import HipRefusal, hip_outcomes_of_both_engines


def _packed_atomic_kernel(elem: Type, value_type: Type) -> KernelDef:
    """A ``global_atomic_add_pk_<elem>`` of a ``value_type`` value loaded from
    an ``elem`` buffer, written past the builder."""
    b = IRBuilder("packed_atomic_add_raw")
    values = b.param("P", PtrType(elem, "global"))
    lane = b.thread_id_x()
    value = b.global_load(values, lane, value_type, align=2)
    b._op(
        f"memref.global_atomic_add_pk_{elem.name}",
        [values, lane, value],
        [value_type],
        attrs={"elem_type": elem.name, "vec": 2, "ordering": "monotonic"},
    )
    b.ret()
    return b.kernel


@pytest.mark.parametrize(
    "elem,value_type",
    [
        (F16, F16),
        (BF16, BF16),
        (F16, VectorType(F16, 4)),
        (BF16, VectorType(F16, 2)),
    ],
)
def test_a_packed_atomic_add_of_anything_but_a_pair_is_refused(elem, value_type):
    kernel = _packed_atomic_kernel(elem, value_type)
    message = (
        f"global_atomic_add_pk_{elem.name} expects <2 x {elem.name}> input, "
        f"got {value_type.name}"
    )
    outcomes = hip_outcomes_of_both_engines(serialize(kernel), "gfx942")
    assert outcomes == (HipRefusal("ValueError", message),) * 2, outcomes


@pytest.mark.parametrize("elem", [F16, BF16])
def test_the_builder_refuses_a_packed_atomic_add_of_a_scalar(elem):
    b = IRBuilder("packed_atomic_add_scalar")
    values = b.param("P", PtrType(elem, "global"))
    lane = b.thread_id_x()
    scalar = b.global_load(values, lane, elem, align=2)
    add_packed = getattr(b, f"global_atomic_add_pk_{elem.name}")
    message = (
        f"global_atomic_add_pk_{elem.name} expects <2 x {elem.name}> input, "
        f"got {elem.name}"
    )
    with pytest.raises(ValueError) as error:
        add_packed(values, lane, scalar)
    assert error.value.args[0] == message
