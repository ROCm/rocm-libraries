# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The HIP lowering's refusals for a typed buffer store it cannot emit.

The typed buffer stores (``tile.buffer_store_f16`` / ``_bf16`` / ``_f32`` and
``tile.buffer_store_vN_f16`` / ``_bf16`` / ``_f32``) share one emitter in each
engine. Serialized IR reaches that emitter without passing the builder, so it
takes each store's byte count from the IR's rule for them, which checks the
operand and result counts and a vector store's ``dwords`` attr. Both engines
must refuse with the same type and text.
"""

from __future__ import annotations

import pytest
from rocke.core.ir import I8, IRBuilder, KernelDef, PtrType
from rocke.core.ir_serialize import serialize
from rocke.tests.engines import HipRefusal, hip_outcomes_of_both_engines

TYPED_VECTOR_STORES = (
    "tile.buffer_store_vN_f16",
    "tile.buffer_store_vN_bf16",
    "tile.buffer_store_vN_f32",
)


def _typed_store_kernel(
    op_name: str, operand_count: int, attrs: dict[str, int]
) -> KernelDef:
    """A typed buffer store ``op_name`` with the first ``operand_count`` of
    (descriptor, vector offset, scalar offset, a 16-byte value) and ``attrs``,
    written past the builder."""
    b = IRBuilder("typed_buffer_store_raw")
    bytes_param = b.param("X", PtrType(I8, "global"))
    rsrc = b.buffer_rsrc(bytes_param, b.const_i32(64))
    zero = b.const_i32(0)
    value = b.vector_splat(b.const_f32(1.0), 4)
    operands = [rsrc, zero, zero, value]
    b._op(op_name, operands[:operand_count], attrs=attrs)
    b.ret()
    return b.kernel


@pytest.mark.parametrize("op_name", TYPED_VECTOR_STORES)
def test_a_typed_vector_store_of_an_unsupported_dword_count_is_refused(op_name):
    kernel = _typed_store_kernel(op_name, 4, {"dwords": 3})
    message = f"{op_name} failed - dwords=3, expected one of (1, 2, 4)"
    outcomes = hip_outcomes_of_both_engines(serialize(kernel), "gfx942")
    assert outcomes == (HipRefusal("ValueError", message),) * 2, outcomes


@pytest.mark.parametrize("op_name", TYPED_VECTOR_STORES)
def test_a_typed_vector_store_without_a_dword_count_is_refused(op_name):
    kernel = _typed_store_kernel(op_name, 4, {})
    message = f"{op_name} failed - dwords attr missing, expected one of (1, 2, 4)"
    outcomes = hip_outcomes_of_both_engines(serialize(kernel), "gfx942")
    assert outcomes == (HipRefusal("ValueError", message),) * 2, outcomes


@pytest.mark.parametrize(
    "op_name,attrs",
    [
        ("tile.buffer_store_f16", {}),
        ("tile.buffer_store_bf16", {}),
        ("tile.buffer_store_f32", {}),
        ("tile.buffer_store_vN_f32", {"dwords": 4}),
    ],
)
def test_a_typed_store_with_a_missing_operand_is_refused(op_name, attrs):
    """The operand count is checked before any operand is read, so a short
    store is a refusal, not a crash, in both engines."""
    kernel = _typed_store_kernel(op_name, 3, attrs)
    message = f"{op_name}: expected 4 operands / 0 results, got 3 / 0"
    outcomes = hip_outcomes_of_both_engines(serialize(kernel), "gfx942")
    assert outcomes == (HipRefusal("ValueError", message),) * 2, outcomes
