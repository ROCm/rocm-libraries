# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from copy import deepcopy

import pytest

from rocke.core.ir import (
    F32,
    IRBuilder,
    PrintValue,
    PtrType,
    _DEVICE_PRINT_DEFAULT_MAX_LITERAL_BYTES,
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
    assert b"state=%lld unsigned=%llu f=%.9g ok=%c p=%p" in llvm.encode()
    assert "select i1 %eq3, i64 116, i64 102" in llvm
    assert 'c"true\\00"' not in llvm
    assert 'c"false\\00"' not in llvm
    assert "br i1 %eq3, label %device.print." in llvm
    assert "sext i32 -5 to i64" in llvm
    assert "zext i32 -5 to i64" in llvm
    assert "fpext float 0x401A000000000000 to double" in llvm
    assert "bitcast double %printf_f64.3 to i64" in llvm
    assert "ptrtoint ptr addrspace(1) %p to i64" in llvm
    assert "call i64 @__ockl_printf_append_args" in llvm
    assert llvm.count("call i64 @__ockl_printf_append_string_n") == 1


def test_device_print_text_transport_has_one_trailing_nul() -> None:
    builder = IRBuilder("print_text_terminator")
    builder.device_print("abc", termination="none")
    llvm = lower_kernel_to_llvm(builder.kernel, arch="gfx950")
    assert (
        "@.rocke.printf.0 = private unnamed_addr addrspace(4) constant "
        '[4 x i8] c"abc\\00", align 1'
    ) in llvm
    assert (
        "ptr addrspacecast (ptr addrspace(4) @.rocke.printf.0 to ptr), " "i64 4, i32 1)"
    ) in llvm


def test_device_print_seven_and_eight_value_packet_boundary() -> None:
    for count, expected_groups in ((7, (7,)), (8, (7, 1))):
        b = IRBuilder(f"print_{count}")
        value = b.const_i32(1)
        true_value = b.cmp_eq(value, value)
        false_value = b.cmp_ne(value, value)
        items = [true_value]
        items.extend(PrintValue(value, "i32") for _ in range(count - 2))
        items.append(false_value)
        b.device_print(*items)
        llvm = lower_kernel_to_llvm(b.kernel, arch="gfx950")
        assert "%c" + "%lld" * (count - 2) + "%c\\0A" in llvm
        assert llvm.count("i64 116, i64 102") == 2
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


def test_device_print_verifier_rejects_mutated_text_with_nul() -> None:
    kernel = deepcopy(_build_print_kernel())
    op = kernel.body.ops[-1]
    op.attrs["items"][0]["text"] = "state=\x00hidden"
    messages = [diagnostic.message for diagnostic in verify(kernel)]
    assert any("contains NUL" in message for message in messages)


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
    with pytest.raises(ValueError, match=str(_DEVICE_PRINT_DEFAULT_MAX_LITERAL_BYTES)):
        b.device_print("x" * (_DEVICE_PRINT_DEFAULT_MAX_LITERAL_BYTES + 1))
    with pytest.raises(ValueError, match="ASCII"):
        b.device_print("lambda=λ")
    with pytest.raises(ValueError, match="NUL"):
        b.device_print("visible\x00hidden")
    with pytest.raises(ValueError, match="termination"):
        b.device_print("x", termination="sometimes")


def test_device_print_limits_are_internal() -> None:
    import rocke
    import rocke.core

    assert not hasattr(rocke, "DEVICE_PRINT_MAX_LITERAL_BYTES")
    assert not hasattr(rocke, "DEVICE_PRINT_MAX_VALUES")
    assert not hasattr(rocke.core, "DEVICE_PRINT_MAX_LITERAL_BYTES")
    assert not hasattr(rocke.core, "DEVICE_PRINT_MAX_VALUES")


def test_device_print_environment_limit_overrides(monkeypatch) -> None:
    monkeypatch.setenv("ROCKE_ENGINE_DEVICE_PRINT_MAX_LITERAL_BYTES", " 4 ")
    accepted = IRBuilder("literal_limit_accepted")
    accepted.device_print("1234", termination="none")
    rejected = IRBuilder("literal_limit_rejected")
    with pytest.raises(ValueError, match="exceeds 4 literal bytes"):
        rejected.device_print("12345", termination="none")

    raised_limit = _DEVICE_PRINT_DEFAULT_MAX_LITERAL_BYTES + 1
    monkeypatch.setenv("ROCKE_ENGINE_DEVICE_PRINT_MAX_LITERAL_BYTES", str(raised_limit))
    accepted = IRBuilder("raised_literal_limit")
    accepted.device_print("x" * raised_limit, termination="none")

    monkeypatch.setenv("ROCKE_ENGINE_DEVICE_PRINT_MAX_VALUE_COUNT", "1")
    rejected = IRBuilder("value_limit_rejected")
    first = rejected.const_i32(1)
    second = rejected.const_i32(2)
    with pytest.raises(ValueError, match="exceeds 1 expanded values"):
        rejected.device_print(first, second)


@pytest.mark.parametrize("value", ["0", "-1", "1_0", "abc", "2147483648"])
def test_device_print_environment_limit_rejects_invalid_values(
    monkeypatch, value: str
) -> None:
    name = "ROCKE_ENGINE_DEVICE_PRINT_MAX_LITERAL_BYTES"
    monkeypatch.setenv(name, value)
    builder = IRBuilder("invalid_environment_limit")
    with pytest.raises(ValueError, match=name):
        builder.device_print("x")


def test_device_print_environment_empty_is_unset(monkeypatch) -> None:
    monkeypatch.setenv("ROCKE_ENGINE_DEVICE_PRINT_MAX_LITERAL_BYTES", "   ")
    builder = IRBuilder("empty_environment_limit")
    with pytest.raises(ValueError, match=str(_DEVICE_PRINT_DEFAULT_MAX_LITERAL_BYTES)):
        builder.device_print("x" * (_DEVICE_PRINT_DEFAULT_MAX_LITERAL_BYTES + 1))


def test_device_print_verifier_uses_environment_limits(monkeypatch) -> None:
    kernel = _build_print_kernel()
    monkeypatch.setenv("ROCKE_ENGINE_DEVICE_PRINT_MAX_VALUE_COUNT", "1")
    messages = [diagnostic.message for diagnostic in verify(kernel)]
    assert any("expanded value count exceeds 1" in message for message in messages)


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
