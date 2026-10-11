# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``store_fragment`` performs no element-type conversion of its own beyond the legacy f32
-> f16 / bf16 narrowing, which is not extended: any other dtype pair (a widening such as f32
-> f64, or a reinterpretation such as f32 -> i32) is refused before any IR is emitted.
Offline, no GPU; lowering pins the Python engine.
"""

from __future__ import annotations

import pytest

from rocke.core.ir import F16, F32, F64, I32, IRBuilder, PtrType
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.helpers.tiling import (
    fill_fragment,
    make_fragment,
    make_tensor_desc,
    make_window,
    store_fragment,
)
from rocke.helpers.tiling.memory import cooperative_load_desc
from rocke.tests.ir_validity import assert_target_accepts

# A real (rows, cols) cooperative tile over one 64-lane wave.
_TILE = cooperative_load_desc(64, 16, 1, vw=4)


def _store(b: IRBuilder, fragment_dtype, memory_dtype) -> None:
    out = b.param("OUT", PtrType(memory_dtype, "global"))
    tensor = make_tensor_desc((64, 16), (16, 1), memory_dtype)
    fragment = make_fragment(_TILE, fragment_dtype)
    fill_fragment(b, fragment, 0)
    store_fragment(b, out, make_window(tensor, (0, 0)), fragment, b.thread_id_x())


def test_a_fragment_stores_into_a_tensor_of_its_own_element_type() -> None:
    b = IRBuilder("store_f32_as_f32")
    _store(b, F32, F32)
    # The C++ engine does not mirror this yet.
    assert_target_accepts(
        _lower_kernel_to_llvm_python(b.kernel, arch="gfx90a"), "gfx90a"
    )


@pytest.mark.parametrize(
    "fragment_dtype,memory_dtype,message",
    [
        (
            F32,
            F64,
            (
                "store_fragment does not convert element types: the fragment is f32, "
                "the tensor is f64; store a fragment whose element type matches the tensor"
            ),
        ),
        (
            F32,
            I32,
            (
                "store_fragment does not convert element types: the fragment is f32, "
                "the tensor is i32; store a fragment whose element type matches the tensor"
            ),
        ),
        (
            F16,
            F32,
            (
                "store_fragment does not convert element types: the fragment is f16, "
                "the tensor is f32; store a fragment whose element type matches the tensor"
            ),
        ),
    ],
)
def test_conversions_are_refused_before_any_ir(
    fragment_dtype, memory_dtype, message
) -> None:
    b = IRBuilder(f"store_{fragment_dtype.name}_as_{memory_dtype.name}")
    out = b.param("OUT", PtrType(memory_dtype, "global"))
    tensor = make_tensor_desc((64, 16), (16, 1), memory_dtype)
    fragment = make_fragment(_TILE, fragment_dtype)
    fill_fragment(b, fragment, 0)
    thread = b.thread_id_x()
    emitted = len(b.kernel.body.ops)
    with pytest.raises(ValueError, match=f"^{message}$"):
        store_fragment(b, out, make_window(tensor, (0, 0)), fragment, thread)
    assert len(b.kernel.body.ops) == emitted
