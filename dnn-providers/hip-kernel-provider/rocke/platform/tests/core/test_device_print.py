# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from copy import deepcopy

import pytest

from rocke.core.ir import (
    DEVICE_PRINT_MAX_LITERAL_BYTES,
    F32,
    IRBuilder,
    PrintValue,
    PtrType,
)
from rocke.core.ir_serialize import parse, serialize
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.core.verify import verify
from rocke.runtime.comgr import ComgrTimings


def _build_print_kernel():
    b = IRBuilder("print_proto")
    pointer = b.param("p", PtrType(F32, "global"))
    integer = b.const_i32(-5)
    floating = b.const_f32(6.5)
    predicate = b.cmp_eq(integer, integer)
    b.device_print(
        "state=",
        integer,
        " unsigned=",
        PrintValue(integer, "u32"),
        " f=",
        floating,
        " ok=",
        predicate,
        " p=",
        pointer,
        predicate=predicate,
    )
    return b.kernel


def test_device_print_canonical_record_roundtrip() -> None:
    kernel = _build_print_kernel()
    assert verify(kernel) == []
    text = serialize(kernel)
    assert text == serialize(parse(text))
    assert 'kind = s:"text", text = s:"state="' in text
    assert 'format = s:"u32", kind = s:"value"' in text
    assert "predicate_operand = i:5" in text
    assert 'text = s:"\\n"' in text


def test_device_print_lowering_uses_canonical_formats_and_predicate() -> None:
    llvm = lower_kernel_to_llvm(_build_print_kernel(), arch="gfx950")
    assert b"state=%lld unsigned=%llu f=%.9g ok=%s p=%p" in llvm.encode()
    assert 'c"true\\00"' in llvm
    assert 'c"false\\00"' in llvm
    assert "i64 5, i64 6" in llvm
    assert "br i1 %eq3, label %device.print." in llvm
    assert "sext i32 -5 to i64" in llvm
    assert "zext i32 -5 to i64" in llvm
    assert "fpext float 0x401A000000000000 to double" in llvm
    assert "bitcast double %printf_f64.3 to i64" in llvm
    assert "ptrtoint ptr addrspace(1) %p to i64" in llvm
    assert "call i64 @__ockl_printf_append_args" in llvm
    assert llvm.count("call i64 @__ockl_printf_append_string_n") == 2


def test_device_print_seven_and_eight_value_packet_boundary() -> None:
    for count, expected_groups in ((7, (7,)), (8, (7, 1))):
        b = IRBuilder(f"print_{count}")
        value = b.const_i32(1)
        b.device_print(*(PrintValue(value, "i32") for _ in range(count)))
        llvm = lower_kernel_to_llvm(b.kernel, arch="gfx950")
        calls = [
            line
            for line in llvm.splitlines()
            if "call i64 @__ockl_printf_append_args" in line
        ]
        assert tuple(
            int(line.split("i32 ", 1)[1].split(",", 1)[0]) for line in calls
        ) == (expected_groups)


def test_device_print_verifier_rejects_mutated_record() -> None:
    kernel = deepcopy(_build_print_kernel())
    op = kernel.body.ops[-1]
    op.attrs["items"][1]["format"] = "f32"
    messages = [d.message for d in verify(kernel)]
    assert any("incompatible" in message for message in messages)


@pytest.mark.parametrize("attribute", ["predicate_operand", "operand"])
def test_device_print_verifier_rejects_boolean_operand_indexes(attribute: str) -> None:
    kernel = deepcopy(_build_print_kernel())
    op = kernel.body.ops[-1]
    if attribute == "predicate_operand":
        op.attrs[attribute] = True
    else:
        op.attrs["items"][1][attribute] = False

    messages = [diagnostic.message for diagnostic in verify(kernel)]
    assert any("out of range" in message for message in messages)


def test_device_print_builder_rejects_invalid_contracts() -> None:
    b = IRBuilder("invalid_print")
    integer = b.const_i32(1)
    with pytest.raises(TypeError, match="not compatible"):
        b.device_print(PrintValue(integer, "f32"))
    with pytest.raises(ValueError, match=str(DEVICE_PRINT_MAX_LITERAL_BYTES)):
        b.device_print("x" * (DEVICE_PRINT_MAX_LITERAL_BYTES + 1))
    with pytest.raises(ValueError, match="ASCII"):
        b.device_print("lambda=λ")
    with pytest.raises(ValueError, match="termination"):
        b.device_print("x", termination="sometimes")


def test_device_print_percent_is_literal_text() -> None:
    b = IRBuilder("print_text")
    b.device_print("100% complete")
    llvm = lower_kernel_to_llvm(b.kernel, arch="gfx950")
    assert "100%% complete" in llvm


def test_compile_propagates_device_library_feature(monkeypatch) -> None:
    from rocke.helpers import compile as compile_module

    seen: list[bool] = []

    def fake_build_hsaco(llvm_text: str, **kwargs) -> tuple[bytes, ComgrTimings]:
        assert "@__ockl_printf_begin" in llvm_text
        seen.append(kwargs["link_device_libraries"])
        return b"hsaco", ComgrTimings()

    monkeypatch.setattr(compile_module, "build_hsaco_from_llvm_ir", fake_build_hsaco)
    artifact = compile_module.compile_kernel(_build_print_kernel(), backend="python")
    assert artifact.hsaco == b"hsaco"
    assert seen == [True]


def test_device_library_feature_is_structural() -> None:
    from rocke.helpers.compile import _kernel_requires_device_libraries

    assert _kernel_requires_device_libraries(_build_print_kernel())
    assert not _kernel_requires_device_libraries(IRBuilder("ordinary").kernel)
