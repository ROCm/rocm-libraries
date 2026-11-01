# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Both HIP lowerers agree op by op: the same source, or the same refusal.

The family-wide HIP source identity test compares whatever the kernel families
build. The cases here pin the ops those families may not reach on every
target: the s_waitcnt refusal on a target without the monolithic instruction.
Each case lowers one small kernel through the Python lowerer and, through the
rocke_engine binding, the C++ one, and compares the text or the error.
"""

import pytest
from rocke.core.ir import F32, IRBuilder, PtrType
from rocke.core.ir_serialize import serialize
from rocke.core.lower_hip import lower_kernel_to_hip

# rocke_status_t (cpp/include/rocke/ir.h): the C++ error code the binding reports
# for each Python exception type the lowerers raise.
_NATIVE_STATUS = {ValueError: 1, TypeError: 2, KeyError: 3, NotImplementedError: 5}


def _native_hip(builder: IRBuilder, gfx: str) -> str:
    rocke_engine = pytest.importorskip("rocke_engine")
    return rocke_engine.lower_serialized_ir_hip(serialize(builder.kernel), gfx)


def _assert_both_refuse(
    builder: IRBuilder, gfx: str, error: type[Exception], message: str
) -> None:
    """Python raises `error` with exactly `message`; C++ reports the matching
    status with the same text."""
    with pytest.raises(error) as python_error:
        lower_kernel_to_hip(builder.kernel, arch=gfx)
    assert python_error.value.args[0] == message
    with pytest.raises(RuntimeError) as native_error:
        _native_hip(builder, gfx)
    assert str(native_error.value).endswith(
        f"(status {_NATIVE_STATUS[error]}): {message}"
    )


def _assert_same_source(builder: IRBuilder, gfx: str) -> str:
    python_hip = lower_kernel_to_hip(builder.kernel, arch=gfx)
    assert _native_hip(builder, gfx) == python_hip
    return python_hip


def _waitcnt_kernel(op: str) -> IRBuilder:
    builder = IRBuilder(f"{op}_kernel")
    builder.param("out", PtrType(F32, "global"))
    if op == "s_waitcnt":
        builder.s_waitcnt(lgkmcnt=0)
    else:
        builder.sync_lds_only()
    builder.ret()
    return builder


@pytest.mark.parametrize("op", ["s_waitcnt", "sync_lds_only"])
def test_waitcnt_is_refused_where_the_target_has_split_wait_counters(op):
    _assert_both_refuse(
        _waitcnt_kernel(op),
        "gfx1250",
        NotImplementedError,
        "s_waitcnt is not available on gfx1250: the target replaced it with "
        "split wait counters, which the HIP lowering does not emit; lower "
        "through the LLVM path",
    )


@pytest.mark.parametrize("op", ["s_waitcnt", "sync_lds_only"])
@pytest.mark.parametrize("gfx", ["gfx942", "gfx950", "gfx1151", "gfx1201"])
def test_waitcnt_still_lowers_where_the_target_has_it(op, gfx):
    hip = _assert_same_source(_waitcnt_kernel(op), gfx)
    assert "__builtin_amdgcn_s_waitcnt(" in hip
