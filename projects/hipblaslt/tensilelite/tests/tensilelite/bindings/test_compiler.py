# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the tensilelite_jit Python bindings."""

import pytest
import tensilelite_jit as jit

PROCESSOR = "gfx950"
TARGET = "amdgcn-amd-amdhsa--" + PROCESSOR
ELF_MAGIC = b"\x7fELF"

KERNEL_IR = """target triple = "amdgcn-amd-amdhsa"

declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @jit_python_kernel(ptr addrspace(1) %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %idx64 = sext i32 %tid to i64
  %ptr = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %idx64
  store i32 %tid, ptr addrspace(1) %ptr, align 4
  ret void
}
"""

INVALID_IR = """target triple = "amdgcn-amd-amdhsa"

define void @f() {
  this is not ir
}
"""

MALFORMED_ASM = """
  .amdgcn_target "amdgcn-amd-amdhsa--gfx950"
  .amdhsa_code_object_version 6
  .text
broken:
  v_not_an_instruction v0, v1
"""


def amdgcn_target_directive(asm):
    """Returns the .amdgcn_target directive line of the assembly."""
    begin = asm.index(b".amdgcn_target ")
    end = asm.find(b"\n", begin)
    return asm[begin:] if end == -1 else asm[begin:end]


def test_opt_level_members():
    assert [level.name for level in jit.OptLevel] == ["o0", "o1", "o2", "o3"]


def test_code_object_version_members():
    assert [version.name for version in jit.CodeObjectVersion] == ["v4", "v5", "v6"]


def test_compile_options_defaults():
    options = jit.CompileOptions()
    assert options.optLevel == jit.OptLevel.o3
    assert options.codeObjectVersion == jit.CodeObjectVersion.v6
    assert options.extraOptions == []


def test_compile_options_round_trip():
    options = jit.CompileOptions()
    options.optLevel = jit.OptLevel.o1
    options.codeObjectVersion = jit.CodeObjectVersion.v5
    options.extraOptions = ["-g"]
    assert options.optLevel == jit.OptLevel.o1
    assert options.codeObjectVersion == jit.CodeObjectVersion.v5
    assert options.extraOptions == ["-g"]


def test_compile_options_keywords():
    options = jit.CompileOptions(
        optLevel=jit.OptLevel.o0,
        codeObjectVersion=jit.CodeObjectVersion.v4,
        extraOptions=["-g"],
    )
    assert options.optLevel == jit.OptLevel.o0
    assert options.codeObjectVersion == jit.CodeObjectVersion.v4
    assert options.extraOptions == ["-g"]


def test_compile_to_asm():
    asm = jit.compileToASM(KERNEL_IR, TARGET)
    assert isinstance(asm, bytes)
    # LLVM spells the triple's empty environment differently across versions.
    directive = amdgcn_target_directive(asm)
    assert directive.startswith(b'.amdgcn_target "amdgcn-amd-amdhsa-')
    assert directive.endswith(b"-" + PROCESSOR.encode() + b'"')
    assert b"s_endpgm" in asm


def test_compile_to_asm_honours_code_object_version():
    options = jit.CompileOptions(codeObjectVersion=jit.CodeObjectVersion.v5)
    asm = jit.compileToASM(llvmIR=KERNEL_IR, target=TARGET, options=options)
    assert b".amdhsa_code_object_version 5" in asm


def test_compile_to_hsaco():
    assert jit.compileToHSACO(KERNEL_IR, TARGET).startswith(ELF_MAGIC)


def test_assemble_to_hsaco():
    asm = jit.compileToASM(KERNEL_IR, TARGET)
    hsaco = jit.assembleToHSACO(assembly=asm.decode(), target=TARGET)
    assert hsaco.startswith(ELF_MAGIC)


@pytest.mark.parametrize("compile_ir", [jit.compileToASM, jit.compileToHSACO])
def test_invalid_ir_raises_runtime_error(compile_ir):
    with pytest.raises(RuntimeError):
        compile_ir(INVALID_IR, TARGET)


def test_malformed_asm_raises_runtime_error():
    with pytest.raises(RuntimeError):
        jit.assembleToHSACO(MALFORMED_ASM, TARGET)


def test_invalid_code_object_version_raises_type_error():
    with pytest.raises(TypeError):
        jit.CompileOptions(codeObjectVersion=7)
