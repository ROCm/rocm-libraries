# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Packed-only types have no IR value.

I4 and U4 values only ever live packed in 32-bit words: a kernel keeps the
words in i32 storage and unpacks them explicitly. So the IR builder of each
engine refuses every op result, kernel parameter and LDS tile of i4 or u4
elements with the same text, and the lowerings, which have no rendering for
either type, still refuse one that reaches them.

The serialized-IR seam (rocke.tests.engines) lowers kernels the Python builder
built, so the C++ builder's refusals are asked of the rocke_ir_types test
binary (ROCKE_IR_TYPES_TEST).
"""

import os
import subprocess

import pytest

from rocke.core.ir import I32, IRBuilder, Param, PtrType, VectorType, dtype_to_ir_type
from rocke.tests.engines import (
    assert_both_engines_reject,
    assert_both_engines_reject_hip,
)

PACKED_ONLY_DTYPES = ["i4", "u4"]

# Builders that would make an IR value, or an LDS tile, of the element type
# they are given, with the IR op each refusal names. rocke_ir_types builds the
# same calls under the same names.
_BUILDERS = {
    "param": ("param", lambda b, t: b.param("X", t)),
    "param_vector": ("param", lambda b, t: b.param("X", VectorType(t, 8))),
    "zero_vec": ("zero_vec", lambda b, t: b.zero_vec(t, 8)),
    "undef_vec": ("vector.undef", lambda b, t: b.undef_vec(t, 8)),
    "trunc": ("arith.trunc", lambda b, t: b.trunc(b.const_i32(0), t)),
    "vec_bitcast": (
        "vector.bitcast",
        lambda b, t: b.vec_bitcast(b.zero_vec(I32, 1), VectorType(t, 8)),
    ),
    "global_load": (
        "memref.global_load_typed",
        lambda b, t: b.global_load(
            b.param("P", PtrType(t, "global")), b.const_i32(0), t
        ),
    ),
    "smem_alloc": ("tile.smem_alloc", lambda b, t: b.smem_alloc(t, (64,))),
}


def _refusal(op: str, dtype: str) -> str:
    return (
        f"{op}: {dtype} values travel packed in 32-bit words and have no scalar "
        "or vector form -- keep them in i32 storage and unpack explicitly"
    )


@pytest.mark.parametrize("dtype", PACKED_ONLY_DTYPES)
@pytest.mark.parametrize("builder", sorted(_BUILDERS))
def test_builder_refuses_a_packed_only_value(builder, dtype):
    op, build = _BUILDERS[builder]
    b = IRBuilder("packed_only")
    with pytest.raises(TypeError) as error:
        build(b, dtype_to_ir_type(dtype))
    assert str(error.value) == _refusal(op, dtype)


@pytest.mark.parametrize("dtype", PACKED_ONLY_DTYPES)
@pytest.mark.parametrize("builder", sorted(_BUILDERS))
def test_native_builder_refuses_a_packed_only_value(builder, dtype):
    executable = os.environ.get("ROCKE_IR_TYPES_TEST")
    if not executable:
        pytest.skip("set ROCKE_IR_TYPES_TEST to the built rocke_ir_types test")
    native = subprocess.run(
        [executable, "--packed-only", builder, dtype],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.rstrip("\n")
    op, _ = _BUILDERS[builder]
    assert native == f"error: {_refusal(op, dtype)}"


@pytest.mark.parametrize("dtype", PACKED_ONLY_DTYPES)
def test_lowerings_refuse_a_packed_only_value(dtype):
    # The builder refuses the parameter, so the kernel is assembled by hand to
    # reach the lowerings' own refusal: the type has no LLVM or HIP spelling.
    b = IRBuilder(f"{dtype}_value")
    b.ret()
    b.kernel.params.append(Param(name="X", type=dtype_to_ir_type(dtype)))
    assert_both_engines_reject(
        b.kernel, "gfx90a", NotImplementedError, f"no LLVM mapping for type {dtype}"
    )
    assert_both_engines_reject_hip(
        b.kernel, "gfx90a", NotImplementedError, f"no HIP mapping for type {dtype}"
    )
