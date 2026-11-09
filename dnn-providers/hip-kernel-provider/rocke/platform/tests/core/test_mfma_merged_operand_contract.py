# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""New fixed MFMA signatures retain the shared operand validation boundary."""

import pytest

from rocke.core.ir import BF16, F32, F64, I32, IRBuilder, VectorType
from rocke.core.ir_serialize import serialize
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python


SIGNATURES = [
    ("mfma_f64_16x16x4_f64", F64, VectorType(F64, 4)),
    ("mfma_f64_4x4x4_f64", F64, VectorType(F64, 1)),
    ("mfma_f32_4x4x4_bf16", VectorType(BF16, 4), VectorType(F32, 4)),
]


def _kernel(signature, concrete):
    op_id, source, accumulator = signature
    b = IRBuilder("fixed_mfma_contract")
    a = b.param("a", source)
    bb = b.param("b", source)
    c = b.param("c", accumulator)
    b.mma(op_id, a, bb, c)
    op = b.kernel.body.ops[-1]
    if concrete:
        op.name = f"tile.{op.attrs.pop('op_id')}"
    b.ret()
    return b.kernel, op


@pytest.mark.parametrize("signature", SIGNATURES)
@pytest.mark.parametrize("role", ["src0", "src1", "src2", "dst", "arity"])
@pytest.mark.parametrize(
    "engine,concrete",
    [
        ("python", False),
        ("python", True),
        ("hip", False),
        ("hip", True),
        ("native", False),
    ],
)
def test_new_signatures_reject_malformed_actual_values(
    signature, concrete, role, engine
):
    kernel, op = _kernel(signature, concrete)
    if role == "arity":
        op.operands.pop()
        error = "MFMA expects 3 operands and 1 result"
    else:
        if role == "dst":
            op.result.type = I32
        else:
            index = int(role[-1])
            op.operands[index].type = I32
            kernel.params[index].type = I32
        expected = signature[1] if role in ("src0", "src1") else signature[2]
        error = f"MFMA requires {'src0 and src1' if role in ('src0', 'src1') else 'src2 and dst'} to be {expected.name}"
    if engine == "native":
        native = pytest.importorskip("rocke_engine")
        with pytest.raises(RuntimeError) as caught:
            native.lower_serialized_ir(
                serialize(kernel), arch="gfx942", flavor="llvm23"
            )
        assert str(caught.value).endswith(f"(status 1): {error}")
    else:
        with pytest.raises(ValueError) as caught:
            if engine == "hip":
                lower_kernel_to_hip(kernel, arch="gfx942")
            else:
                _lower_kernel_to_llvm_python(
                    kernel, arch="gfx942", llvm_flavor="llvm23"
                )
        assert str(caught.value) == error


@pytest.mark.parametrize("signature", SIGNATURES)
@pytest.mark.parametrize("concrete", [False, True])
@pytest.mark.parametrize("flavor", ["llvm20", "llvm22", "llvm23"])
def test_new_signatures_keep_valid_emission(signature, concrete, flavor):
    kernel, _ = _kernel(signature, concrete)
    ll = _lower_kernel_to_llvm_python(kernel, arch="gfx942", llvm_flavor=flavor)
    assert "@llvm.amdgcn.mfma." in ll
    native = pytest.importorskip("rocke_engine")
    # Native serialization uses neutral tile.mma; Python also supports concrete
    # instruction names. Compare both Python dispatch forms with the native form.
    if concrete:
        op = kernel.body.ops[0]
        op.name = "tile.mma"
        op.attrs["op_id"] = signature[0]
    assert (
        native.lower_serialized_ir(serialize(kernel), arch="gfx942", flavor=flavor)
        == ll
    )


@pytest.mark.parametrize("signature", SIGNATURES)
@pytest.mark.parametrize("role", ["src0", "src2", "dst"])
@pytest.mark.parametrize("engine", ["python", "hip"])
def test_fixed_signatures_reject_non_integer_vector_counts(signature, role, engine):
    kernel, op = _kernel(signature, False)
    expected = signature[1] if role == "src0" else signature[2]
    element = expected.elem if isinstance(expected, VectorType) else expected
    count = expected.count if isinstance(expected, VectorType) else 1
    malformed = VectorType(element, float(count))
    if role == "dst":
        op.result.type = malformed
    else:
        index = int(role[-1])
        op.operands[index].type = malformed
        kernel.params[index].type = malformed
    with pytest.raises(ValueError, match="MFMA requires"):
        if engine == "hip":
            lower_kernel_to_hip(kernel, arch="gfx942")
        else:
            _lower_kernel_to_llvm_python(kernel, arch="gfx942", llvm_flavor="llvm23")
