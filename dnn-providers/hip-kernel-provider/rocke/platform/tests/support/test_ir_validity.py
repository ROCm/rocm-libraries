# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The compile-and-link test helper gives the three verdicts it promises.

Each case is a small hand-written module, so the expected verdict is fixed by the
module itself and not by the current state of rocke's lowering. A host without a
usable clang skips, which is the helper's own contract.

The helper finds the IR-validity tool wherever these tests run, refuses a tools
directory without it, and loads each directory's own copy.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from unittest import mock

import pytest
from rocke.assets import platform_tools_dir
from rocke.tests.ir_validity import (
    _validity_tool,
    arbiter_verdict,
    assert_target_accepts,
    target_mnemonics,
)

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
    # Only a host that can judge a known target can say why it cannot judge an
    # unknown one; any other host skips with its own reason.
    _verdict_or_skip(_VALID, "gfx90a")
    verdict, text = arbiter_verdict(_VALID, "gfx9999")
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


def test_target_mnemonics_disassembles_the_code_object():
    _verdict_or_skip(_VALID, "gfx90a")
    mnemonics = target_mnemonics(_VALID, "gfx90a")
    assert "s_endpgm" in mnemonics
    assert any(m.startswith("global_store") for m in mnemonics)


def test_target_mnemonics_fails_on_a_rejected_module():
    _verdict_or_skip(_WRONG_OPERAND_TYPE, "gfx90a")
    with pytest.raises(AssertionError, match="LLVM rejected the lowered IR for gfx90a"):
        target_mnemonics(_WRONG_OPERAND_TYPE, "gfx90a")


def test_the_tool_is_found_where_the_tests_run():
    """Every layout these tests run in ships the IR-validity tool: ``tools/`` in
    a checkout, and in an install the copy staged under ``tests/arch_domain``,
    where ``conftest.py`` points :func:`rocke.assets.platform_tools_dir`.
    Without it every compile check would skip, so a missing tool fails here."""
    assert (
        _validity_tool() is not None
    ), f"no IR-validity tool in {platform_tools_dir()}"


def test_a_tools_dir_without_the_tool_is_refused(tmp_path):
    """A tools directory that holds no tool is refused, so a wrong path fails
    loudly instead of turning every compile check into a skip."""
    with (
        mock.patch.dict(os.environ, {"ROCKE_PLATFORM_TOOLS_DIR": str(tmp_path)}),
        pytest.raises(FileNotFoundError) as refused,
    ):
        arbiter_verdict(_VALID, "gfx90a")
    assert str(refused.value) == (
        f"the platform tools directory {tmp_path.resolve()} holds no "
        "check_ir_validity.py; point ROCKE_PLATFORM_TOOLS_DIR at a directory that "
        "does, or unset it"
    )


def test_each_tools_dir_gets_its_own_copy_of_the_tool(tmp_path):
    """Two directories each hold a copy of the tool and the modules it imports.
    The helper loads the tool from whichever directory is named, also after
    loading the other; the tool's imports come from its own directory, and the
    modules of the same names imported before are put back afterwards."""
    tool = _validity_tool()
    assert tool is not None, f"no IR-validity tool in {platform_tools_dir()}"
    source = Path(tool.__file__).parent
    beside = {module_file.stem for module_file in source.glob("*.py")}
    imported_before = {
        name: module for name, module in sys.modules.items() if name in beside
    }
    assert imported_before, "the tool imports no module beside it"
    for directory in (tmp_path / "first", tmp_path / "second"):
        directory.mkdir()
        for module_file in source.glob("*.py"):
            shutil.copy(module_file, directory)
        with mock.patch.dict(os.environ, {"ROCKE_PLATFORM_TOOLS_DIR": str(directory)}):
            loaded = _validity_tool()
        assert Path(loaded.__file__).parent == directory.resolve()
        for imported in (
            loaded._requires_newer_flavor,
            loaded.available_cpus,
            loaded.tool_identity,
        ):
            assert Path(imported.__code__.co_filename).parent == directory.resolve()
        assert {name: sys.modules.get(name) for name in imported_before} == (
            imported_before
        )
