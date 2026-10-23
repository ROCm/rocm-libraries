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
    opcodes = [inst.opcode for fn in again.functions() for block in fn.blocks() for inst in block.instructions()]
    assert opcodes == ["st.v_mov_b32", "st.v_add_f32"]
    mov = next(inst for fn in again.functions() for block in fn.blocks() for inst in block.instructions())
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
    with pytest.raises(RuntimeError, match="HazardGapAnalysisPass reported an analysis failure"):
        module.run_passes(["HazardGapAnalysisPass"])


def test_view_survives_removal():
    module = stinkytofu.load_stir(KERNEL)
    mov = next(
        inst for fn in module.functions() for block in fn.blocks() for inst in block.instructions()
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


def test_hwreg_operand():
    text = """\
st.func @k() {
^entry:
  hwreg(26, 0, 2) = "st.s_setreg_IMM32_b32"(2) { issueCycles = 1, latencyCycles = 1 }
}
"""
    module = stinkytofu.load_stir(text)
    inst = next(
        inst for fn in module.functions() for block in fn.blocks() for inst in block.instructions()
    )
    assert inst.defs == [("hwreg", 26, 0, 2)]
    assert 2 in inst.uses


def test_cli_passes(capsys):
    assert main(["passes"]) == 0
    assert "LiftAsmRegistersToSSAPass" in capsys.readouterr().out


def test_cli_run_round_trips(tmp_path, capsys):
    path = tmp_path / "k.stir"
    path.write_text(KERNEL)
    assert main(["run", str(path), "--pass", "LiftAsmRegistersToSSAPass"]) == 0
    printed = capsys.readouterr().out
    stinkytofu.load_stir(printed)


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
    assert launcher.module_dir(str(prefix / "bin" / "stinkytofu")) == str(package.parent)
