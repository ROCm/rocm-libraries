# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The compile-and-link test helper gives the three verdicts it promises.

Each case is a small hand-written module, so the expected verdict is fixed by the
module itself and not by the current state of rocke's lowering. A host without a
usable clang skips, which is the helper's own contract.
"""

from __future__ import annotations

import pytest

from rocke.tests.ir_validity import arbiter_verdict, assert_target_accepts

_VALID = """\
target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @k(ptr addrspace(1) %p) {
  store float 1.0, ptr addrspace(1) %p
  ret void
}
"""

# An MFMA operand printed as `float` but defined as `<1 x float>`: the parser
# rejects it before any code generation.
_WRONG_OPERAND_TYPE = """\
target triple = "amdgcn-amd-amdhsa"
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float, float, <4 x float>, i32, i32, i32)
define amdgpu_kernel void @k(ptr addrspace(1) %p, <1 x float> %a) {
  %r = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %a, float 1.0, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  store <4 x float> %r, ptr addrspace(1) %p
  ret void
}
"""

# A call to an intrinsic that does not exist compiles to an object; only the
# link step reports it.
_UNDEFINED_INTRINSIC = """\
target triple = "amdgcn-amd-amdhsa"
declare float @llvm.amdgcn.not.an.intrinsic(float)
define amdgpu_kernel void @k(ptr addrspace(1) %p) {
  %r = call float @llvm.amdgcn.not.an.intrinsic(float 1.0)
  store float %r, ptr addrspace(1) %p
  ret void
}
"""


def _verdict_or_skip(ir_text: str, target: str) -> tuple[str, str]:
    verdict, text = arbiter_verdict(ir_text, target)
    if verdict == "cannot_judge":
        pytest.skip(text)
    return verdict, text


def test_a_valid_module_is_accepted():
    assert _verdict_or_skip(_VALID, "gfx90a") == ("accepted", "")
    assert_target_accepts(_VALID, "gfx90a")


def test_a_mistyped_operand_is_rejected_with_the_parser_error():
    verdict, text = _verdict_or_skip(_WRONG_OPERAND_TYPE, "gfx90a")
    assert verdict == "rejected"
    assert "defined with type '<1 x float>' but expected 'float'" in text


def test_a_call_to_a_nonexistent_intrinsic_is_rejected_at_link():
    verdict, text = _verdict_or_skip(_UNDEFINED_INTRINSIC, "gfx90a")
    assert verdict == "rejected"
    assert "undefined symbol: llvm.amdgcn.not.an.intrinsic" in text


def test_a_target_the_host_clang_does_not_know_cannot_be_judged():
    verdict, text = arbiter_verdict(_VALID, "gfx9999")
    if verdict == "cannot_judge" and "no host clang" in text:
        pytest.skip(text)
    assert (verdict, text) == ("cannot_judge", "host clang cannot target gfx9999")


def test_assert_target_accepts_fails_on_a_rejected_module():
    _verdict_or_skip(_WRONG_OPERAND_TYPE, "gfx90a")
    with pytest.raises(AssertionError, match="LLVM rejected the lowered IR for gfx90a"):
        assert_target_accepts(_WRONG_OPERAND_TYPE, "gfx90a")


def test_a_clang_that_cannot_run_its_linker_is_not_a_verdict_on_the_module():
    # The message a distro clang without an AMDGPU linker prints. It says nothing about the
    # module, so it must not be reported as a rejection.
    from rocke.tests.ir_validity import _toolchain_incomplete

    assert _toolchain_incomplete(
        'clang: error: unable to execute command: Executable "ld.lld" doesn\'t exist!'
    )
    assert not _toolchain_incomplete(
        "ld.lld: error: undefined symbol: llvm.amdgcn.not.an.intrinsic"
    )
    assert not _toolchain_incomplete(
        "module.ll:4:64: error: '%a' defined with type '<1 x float>' but expected 'float'"
    )
