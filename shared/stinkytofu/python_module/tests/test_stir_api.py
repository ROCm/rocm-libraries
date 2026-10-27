# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Physical STIR load, pass run, and the stinkytofu command. No GPU."""

import importlib.machinery
import importlib.util
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../build/lib"))

import stinkytofu
from stinkytofu.cli import main

KERNEL = """\
st.func @k() {
^entry:
  v0 = "st.v_mov_b32"(v1) { issueCycles = 1, latencyCycles = 1 }
  v2 = "st.v_add_f32"(v0, v1) { issueCycles = 1, latencyCycles = 1 }
}
"""


def test_physical_round_trip():
    printed = stinkytofu.load_stir(KERNEL).serialize()
    assert stinkytofu.load_stir(printed).serialize() == printed
    assert "st.v_mov_b32" in printed
    again = stinkytofu.load_stir(printed)
    opcodes = [
        inst.opcode
        for fn in again.functions()
        for block in fn.blocks()
        for inst in block.instructions()
    ]
    assert opcodes == ["st.v_mov_b32", "st.v_add_f32"]
    mov = next(
        inst
        for fn in again.functions()
        for block in fn.blocks()
        for inst in block.instructions()
    )
    assert mov.defs == [("v", 0, 1)]
    assert ("v", 1, 1) in mov.uses


def test_ssa_view_is_refused():
    module = stinkytofu.load_stir(KERNEL)
    module.run_passes(["LiftAsmRegistersToSSAPass"])
    with pytest.raises(ValueError, match="does not round-trip"):
        stinkytofu.load_stir(module.serialize(ssa=True))


def test_lift_exposes_value_ids():
    module = stinkytofu.load_stir(KERNEL)
    module.run_passes(["LiftAsmRegistersToSSAPass"])
    add = next(
        inst
        for fn in module.functions()
        for block in fn.blocks()
        for inst in block.instructions()
        if inst.opcode == "st.v_add_f32"
    )
    assert add.defs
    value_id, reg_class = add.defs[0]
    assert isinstance(value_id, int)
    assert reg_class == "v"
    assert all(isinstance(use[0], int) for use in add.uses if isinstance(use, tuple))


def test_unknown_pass_names_a_real_pass():
    module = stinkytofu.load_stir(KERNEL)
    with pytest.raises(ValueError, match="LiftAsmRegistersToSSAPass"):
        module.run_passes(["NotARealPass"])


def test_unknown_pass_runs_nothing():
    module = stinkytofu.load_stir(KERNEL)
    before = module.serialize()
    with pytest.raises(ValueError, match="unknown pass 'NotARealPass'"):
        module.run_passes(["RemoveInstructionPass=v_mov_b32", "NotARealPass"])
    assert module.serialize() == before


def test_pass_names_lists_the_lift():
    assert "LiftAsmRegistersToSSAPass" in stinkytofu.pass_names()


def test_missed_remark_raises():
    module = stinkytofu.load_stir(KERNEL)
    with pytest.raises(RuntimeError, match="no attached SSA"):
        module.run_passes(["RegisterAllocationPass"])


def test_analysis_failure_raises():
    text = """\
st.func @k() {
^entry:
  s0 = "st.s_mov_b32"(s1) { issueCycles = 1, latencyCycles = 1 }
  "st.tensor_load_to_lds"(s[0:3], s[4:11]) { issueCycles = 1, latencyCycles = 1 }
}
"""
    module = stinkytofu.load_stir(text)
    with pytest.raises(
        RuntimeError, match="HazardGapAnalysisPass reported an analysis failure"
    ):
        module.run_passes(["HazardGapAnalysisPass"])


def test_hazard_pass_gets_gfx1250_caps():
    text = """\
st.func @k() {
^entry:
  v9 = "st.buffer_load_b32"(v0) { issueCycles = 1, latencyCycles = 1 }
  v3 = "st.global_atomic_inc_u32"(v1, v2, s[14:15]) { issueCycles = 1, latencyCycles = 1 }
}
"""
    module = stinkytofu.load_stir(text)
    module.run_passes(["Gfx1250HazardPass"])
    opcodes = [
        inst.opcode
        for fn in module.functions()
        for block in fn.blocks()
        for inst in block.instructions()
    ]
    assert "st.s_wait_xcnt" in opcodes


def test_view_survives_removal():
    module = stinkytofu.load_stir(KERNEL)
    mov = next(
        inst
        for fn in module.functions()
        for block in fn.blocks()
        for inst in block.instructions()
    )
    module.run_passes(["RemoveInstructionPass=v_mov_b32"])
    assert mov.opcode == "st.v_mov_b32"
    assert mov.defs == [("v", 0, 1)]
    opcodes = [
        inst.opcode
        for fn in module.functions()
        for block in fn.blocks()
        for inst in block.instructions()
    ]
    assert opcodes == ["st.v_add_f32"]


def test_symbolic_literal_operand():
    text = """\
st.func @k() {
^entry:
  s2 = "st.s_mov_b32"(ttmp9) { issueCycles = 1, latencyCycles = 2 }
  s82 = "st.s_mul_i32"(MT1, s3) { issueCycles = 1, latencyCycles = 2 }
}
"""
    module = stinkytofu.load_stir(text)
    mov, mul = [
        inst
        for fn in module.functions()
        for block in fn.blocks()
        for inst in block.instructions()
    ]
    assert mov.uses == ["ttmp9"]
    assert mul.uses == ["MT1", ("s", 3, 1)]
    assert stinkytofu.load_stir(module.serialize()).serialize() == module.serialize()
    with pytest.raises(ValueError, match="Unknown register type"):
        stinkytofu.load_stir(
            'st.func @k() {\n^entry:\n  s0 = "st.s_mov_b32"(Foo[1])\n}\n'
        )


def test_hwreg_operand():
    text = """\
st.func @k() {
^entry:
  hwreg(26, 0, 2) = "st.s_setreg_IMM32_b32"(2) { issueCycles = 1, latencyCycles = 1 }
}
"""
    module = stinkytofu.load_stir(text)
    inst = next(
        inst
        for fn in module.functions()
        for block in fn.blocks()
        for inst in block.instructions()
    )
    assert inst.defs == [("hwreg", 26, 0, 2)]
    assert 2 in inst.uses


MODULE = """\
st.module @Cijk_MT16x16_-Qev53=sp {
  st.func @main() {
    ^entry:
      "st.s_nop"(0) { issueCycles = 1, latencyCycles = 1 }
  }

  st.func @helper() {
    ^entry:
      "st.s_nop"(0) { issueCycles = 1, latencyCycles = 1 }
  }
}
"""

ASM = """\
s_nop 0
loop_begin:
s_nop 0
v_mov_b32 v0, v1
loop_end:
s_endpgm
"""

ROUNDTRIP_KERNEL = os.path.join(
    os.path.dirname(__file__), "../../tests/filecheck/raw_asm_roundtrip.s"
)


def test_module_name_survives_reload():
    printed = stinkytofu.load_stir(MODULE).serialize()
    assert printed.startswith("st.module @Cijk_MT16x16_-Qev53=sp {\n")
    again = stinkytofu.load_stir(printed)
    assert again.serialize() == printed
    assert [fn.name for fn in again.functions()] == ["main", "helper"]


def test_unnamed_function_round_trips():
    text = 'st.func @() {\n^entry:\n  "st.s_nop"(0) { issueCycles = 1, latencyCycles = 1 }\n}\n'
    printed = stinkytofu.load_stir(text).serialize()
    again = stinkytofu.load_stir(printed)
    assert again.serialize() == printed
    assert again.functions()[0].name == ""


def test_typo_in_an_opcode_raises():
    with pytest.raises(ValueError, match="4:3: error: Unknown mnemonic 'v_add_f33'"):
        stinkytofu.load_stir(KERNEL.replace("st.v_add_f32", "st.v_add_f33"))


def test_dangling_successor_raises():
    text = """\
st.func @k() {
^entry:
  "st.s_nop"(0) { issueCycles = 1, latencyCycles = 1 }
  Successors: ^loop
}
"""
    with pytest.raises(ValueError, match=r"Successor '\^loop' of block '\^entry'"):
        stinkytofu.load_stir(text)


def test_load_warnings_keep_what_loaded():
    module = stinkytofu.load_stir('st.func @k() {\n^entry:\n  "st.s_nop"(0)\n}\n')
    assert module.load_warnings() == [
        "3:16: warning: Instruction missing attributes (issueCycles, latencyCycles)"
    ]
    assert len(module.functions()[0].blocks()[0].instructions()) == 1
    assert stinkytofu.load_stir(KERNEL).load_warnings() == []


def test_modifiers_are_visible():
    text = """\
st.func @k() {
^entry:
  v[0:3] = "st.ds_load_b128"(v40) { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, offset = 64, gds = false } }
}
"""
    inst = stinkytofu.load_stir(text).functions()[0].blocks()[0].instructions()[0]
    assert inst.modifiers == ["mod.ds = { na = 1, offset = 64, gds = false }"]


def test_load_asm_runs_passes_and_emits():
    module = stinkytofu.load_asm(ASM)
    assert module.load_warnings() == []
    assert module.emit_asm() == ASM
    module.run_passes(["StinkyRemoveNopPass"])
    assert module.emit_asm() == "loop_begin:\nv_mov_b32 v0, v1\nloop_end:\ns_endpgm\n"


def test_load_asm_region():
    module = stinkytofu.load_asm(ASM, from_label="loop_begin", to_label="loop_end")
    assert module.emit_asm() == "loop_begin:\ns_nop 0\nv_mov_b32 v0, v1\nloop_end:\n"
    with pytest.raises(ValueError, match="must be given together"):
        stinkytofu.load_asm(ASM, from_label="loop_begin")
    with pytest.raises(ValueError, match="end label 'nowhere' not found"):
        stinkytofu.load_asm(ASM, from_label="loop_begin", to_label="nowhere")


def test_unknown_asm_mnemonic_is_kept_as_text():
    module = stinkytofu.load_asm("s_nop 0\ns_bogus_op 0\n")
    assert module.load_warnings() == [
        "2:1: warning: Unknown mnemonic 's_bogus_op'; storing as text"
    ]
    assert "s_bogus_op 0" in module.emit_asm()


def test_kernel_header_round_trips():
    module = stinkytofu.load_asm_file(ROUNDTRIP_KERNEL)
    first = module.emit_asm()
    assert ".amdhsa_kernel Cijk_Alik_Bljk" in first
    assert ".amdhsa_kernel" not in module.emit_asm(signature=False)
    assert stinkytofu.load_asm(first).emit_asm() == first
    region = stinkytofu.load_asm_file(
        ROUNDTRIP_KERNEL, from_label="label_LoopBeginL", to_label="label_LoopEndL"
    )
    assert ".amdhsa_kernel" not in region.emit_asm()


@pytest.mark.skipif(not stinkytofu.hasComgrSupport(), reason="built without amd_comgr")
def test_assemble_returns_the_assembler_log():
    assert stinkytofu.load_asm("s_nop 0\ns_endpgm\n").assemble() is None
    assert "invalid instruction" in stinkytofu.load_asm("s_bogus_op 0\n").assemble()


@pytest.mark.skipif(stinkytofu.hasComgrSupport(), reason="built with amd_comgr")
def test_assemble_needs_comgr():
    with pytest.raises(RuntimeError, match="no amd_comgr"):
        stinkytofu.load_asm("s_nop 0\n").assemble()


def test_cli_passes(capsys):
    assert main(["passes"]) == 0
    assert "LiftAsmRegistersToSSAPass" in capsys.readouterr().out


def test_cli_run_round_trips(tmp_path, capsys):
    path = tmp_path / "k.stir"
    path.write_text(KERNEL)
    assert main(["run", str(path), "--pass", "LiftAsmRegistersToSSAPass"]) == 0
    printed = capsys.readouterr().out
    stinkytofu.load_stir(printed)


def test_cli_runs_assembler_source(tmp_path, capsys):
    path = tmp_path / "k.s"
    path.write_text(ASM)
    assert main(["run", str(path), "--pass", "StinkyRemoveNopPass"]) == 0
    assert (
        capsys.readouterr().out
        == "loop_begin:\nv_mov_b32 v0, v1\nloop_end:\ns_endpgm\n"
    )


def test_cli_emits_assembly_from_stir(tmp_path, capsys):
    path = tmp_path / "k.stir"
    path.write_text(KERNEL)
    assert main(["run", str(path), "--emit-asm"]) == 0
    assert "v_mov_b32 v0, v1" in capsys.readouterr().out


def test_cli_load_error_exits_2(tmp_path, capsys):
    path = tmp_path / "k.stir"
    path.write_text(KERNEL.replace("st.v_add_f32", "st.v_add_f33"))
    assert main(["run", str(path)]) == 2
    assert "Unknown mnemonic 'v_add_f33'" in capsys.readouterr().err


def test_cli_unknown_pass(tmp_path, capsys):
    path = tmp_path / "k.stir"
    path.write_text(KERNEL)
    assert main(["run", str(path), "--pass", "NotARealPass"]) == 2
    assert "LiftAsmRegistersToSSAPass" in capsys.readouterr().err


def test_cli_refused_pass_exits_1(tmp_path, capsys):
    path = tmp_path / "k.stir"
    path.write_text(KERNEL)
    assert main(["run", str(path), "--pass", "RegisterAllocationPass"]) == 1
    assert "no attached SSA" in capsys.readouterr().err


def test_cli_analysis_failure_exits_1(tmp_path, capsys):
    path = tmp_path / "k.stir"
    path.write_text(
        """\
st.func @k() {
^entry:
  s0 = "st.s_mov_b32"(s1) { issueCycles = 1, latencyCycles = 1 }
  "st.tensor_load_to_lds"(s[0:3], s[4:11]) { issueCycles = 1, latencyCycles = 1 }
}
"""
    )
    assert main(["run", str(path), "--pass", "HazardGapAnalysisPass"]) == 1
    assert "analysis failure" in capsys.readouterr().err


def _launcher():
    path = os.path.join(os.path.dirname(__file__), "../bin/stinkytofu")
    loader = importlib.machinery.SourceFileLoader("stinkytofu_launcher", path)
    spec = importlib.util.spec_from_loader("stinkytofu_launcher", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_launcher_finds_build_and_install_layouts(tmp_path):
    launcher = _launcher()
    build = tmp_path / "build"
    (build / "bin").mkdir(parents=True)
    (build / "lib" / "stinkytofu").mkdir(parents=True)
    (build / "lib" / "stinkytofu" / "__init__.py").write_text("")
    assert launcher.module_dir(str(build / "bin" / "stinkytofu")) == str(build / "lib")

    version = "python%d.%d" % sys.version_info[:2]
    prefix = tmp_path / "prefix"
    package = prefix / "lib" / version / "dist-packages" / "stinkytofu"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (prefix / "lib" / "stinkytofu").mkdir()
    (prefix / "lib" / "stinkytofu" / "intrinsics.st.bc").write_text("")
    (prefix / "bin").mkdir()
    assert launcher.module_dir(str(prefix / "bin" / "stinkytofu")) == str(
        package.parent
    )
