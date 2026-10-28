# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Physical STIR load, pass run, and the stinkytofu command. No GPU."""

import importlib.machinery
import importlib.util
import os
import shlex
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
    message = (
        r"4:3: error: Unknown mnemonic 'v_add_f33' for gfx1250 \[unknown-mnemonic\]"
    )
    with pytest.raises(ValueError, match=message):
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
        " [missing-attributes]"
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
        "2:1: warning: Unknown mnemonic 's_bogus_op'; storing as text [unknown-mnemonic]"
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


@pytest.mark.skipif(not stinkytofu.hasComgrSupport(), reason="built without amd_comgr")
def test_assemble_reports_an_undefined_label():
    # comgr prints this one to stderr instead of its log.
    log = stinkytofu.load_asm("s_branch label_nowhere\ns_endpgm\n").assemble()
    assert "undefined label 'label_nowhere'" in log


@pytest.mark.skipif(stinkytofu.hasComgrSupport(), reason="built with amd_comgr")
def test_assemble_needs_comgr():
    with pytest.raises(RuntimeError, match="no amd_comgr"):
        stinkytofu.load_asm("s_nop 0\n").assemble()


@pytest.mark.parametrize("build_cfg", [False, True])
def test_asm_load_survives_stir(build_cfg):
    module = stinkytofu.load_asm_file(ROUNDTRIP_KERNEL)
    if build_cfg:
        module.run_passes(["CFGBuilderPass"])
    printed = module.serialize()
    reloaded = stinkytofu.load_stir(printed)
    assert reloaded.load_warnings() == []
    assert reloaded.serialize() == printed
    # STIR carries no kernel header; every label and directive comes back.
    assert reloaded.emit_asm() == module.emit_asm(signature=False)


STIR_LOOP = """\
st.func @k() {
^entry:
  s0 = "st.s_mov_b32"(0) { issueCycles = 1, latencyCycles = 2 }
  Successors: ^loop
^loop:
  s0 = "st.s_add_u32"(s0, 1) { issueCycles = 1, latencyCycles = 2 }
  "st.s_cmp_lt_u32"(s0, 4) { issueCycles = 1, latencyCycles = 2 }
  "st.s_cbranch_scc1"(loop) { issueCycles = 1, latencyCycles = 1 }
  Successors: ^loop, ^exit
^exit:
  "st.s_endpgm"() { issueCycles = 1, latencyCycles = 1 }
}
"""


def test_a_branch_target_block_gets_its_label():
    module = stinkytofu.load_stir(STIR_LOOP)
    assert "\nloop:\ns_add_u32" in module.emit_asm()
    assert "^loop: { label }\n" in module.serialize()
    assert "^exit:\n" in module.serialize()


@pytest.mark.skipif(not stinkytofu.hasComgrSupport(), reason="built without amd_comgr")
def test_a_stir_loop_assembles():
    assert stinkytofu.load_stir(STIR_LOOP).assemble() is None


def test_a_branch_to_any_label_is_an_instruction():
    module = stinkytofu.load_asm(
        "s_cbranch_scc1 .LBB0_1\ns_nop 0\n.LBB0_1:\ns_endpgm\n"
    )
    assert module.load_warnings() == []
    first = module.functions()[0].blocks()[0].instructions()[0]
    assert first.opcode == "st.s_cbranch_scc1"
    assert first.uses == [".LBB0_1"]
    module.run_passes(["CFGBuilderPass"])
    assert ".LBB0_1" in module.functions()[0].blocks()[0].successors()
    assert stinkytofu.load_stir(module.serialize()).emit_asm() == module.emit_asm()


def test_a_known_mnemonic_kept_as_text_warns():
    text = "tensor_load_to_lds s[12:15], s[16:23] th:TH_LOAD_NT\ns_endpgm\n"
    module = stinkytofu.load_asm(text)
    assert module.load_warnings() == [
        "1:1: warning: Could not read the operands of 'tensor_load_to_lds'; storing as text"
        " [unreadable-operands]"
    ]
    assert module.emit_asm() == text


# Same stream as tests/filecheck/tdm_load_wave_sync_single_wave.stir: one
# barrier with two waves, none with one.
TDM_STREAM = """\
st.func @k() {
^entry:
  "st.tensor_load_to_lds"(s[0:3], s[4:11]) { issueCycles = 1, latencyCycles = 1, mod.memtoken = { tokens = [0] } }
  "st.tensor_load_to_lds"(s[12:15], s[16:23]) { issueCycles = 1, latencyCycles = 1, mod.memtoken = { tokens = [0] } }
  "st.tensor_load_to_lds"(s[24:27], s[28:35]) { issueCycles = 1, latencyCycles = 1, mod.memtoken = { tokens = [1] } }
  "st.s_wait_tensorcnt"() { issueCycles = 1, latencyCycles = 1, mod.swaittensorcnt = { tlcnt = 1 }, mod.memtoken = { tokens = [0] } }
}
"""


def test_a_pass_returns_its_remarks():
    [record] = stinkytofu.load_stir(KERNEL).run_passes(["LiftAsmRegistersToSSAPass"])
    assert (record.pass_name, record.function) == ("LiftAsmRegistersToSSAPass", "k")
    [remark] = record.remarks
    assert (remark.kind, remark.remark_name) == ("passed", "LiftedToSSA")
    assert remark.message.startswith("@k: lifted")
    # The lift attaches SSA values. The physical STIR it prints is unchanged.
    assert not record.changed
    assert record.before is None and record.after is None


def test_a_removal_changes_the_function_without_a_remark():
    [record] = stinkytofu.load_stir(KERNEL).run_passes(
        ["RemoveInstructionPass=v_mov_b32"]
    )
    assert record.changed
    assert record.remarks == []


def test_a_missed_remark_stays_beside_a_success():
    text = """\
st.func @k() {
^entry:
  exec_lo0 = "st.s_mov_b32"(vcc_lo0) { issueCycles = 1, latencyCycles = 2 }
  v11 = "st.v_mov_b32"(0) { issueCycles = 1, latencyCycles = 2 }
  exec_lo0 = "st.s_mov_b32"(-1) { issueCycles = 1, latencyCycles = 2 }
  exec_lo0 = "st.s_mov_b32"(vcc_lo0) { issueCycles = 1, latencyCycles = 2 }
}
"""
    [record] = stinkytofu.load_stir(text).run_passes(["TieExecMaskedWritesPass"])
    assert [(remark.kind, remark.remark_name) for remark in record.remarks] == [
        ("passed", "TiedWrites"),
        ("missed", "UnmatchedExecSpan"),
    ]


def test_one_record_per_pass_per_function():
    records = stinkytofu.load_stir(MODULE).run_passes(
        ["StinkyRemoveNopPass", "RemoveInstructionPass=v_mov_b32"]
    )
    assert [(r.function, r.pass_name, r.changed) for r in records] == [
        ("main", "StinkyRemoveNopPass", True),
        ("main", "RemoveInstructionPass", False),
        ("helper", "StinkyRemoveNopPass", True),
        ("helper", "RemoveInstructionPass", False),
    ]


def test_function_runs_one_function():
    module = stinkytofu.load_stir(MODULE)
    records = module.run_passes(["StinkyRemoveNopPass"], function="helper")
    assert [(r.function, r.changed) for r in records] == [("helper", True)]
    main_fn, helper = module.functions()
    assert [i.opcode for b in main_fn.blocks() for i in b.instructions()] == [
        "st.s_nop"
    ]
    assert [i.opcode for b in helper.blocks() for i in b.instructions()] == []


def test_unknown_function_runs_nothing():
    module = stinkytofu.load_stir(MODULE)
    before = module.serialize()
    with pytest.raises(
        ValueError, match="no function 'nowhere'; functions: 'main', 'helper'"
    ):
        module.run_passes(["StinkyRemoveNopPass"], function="nowhere")
    assert module.serialize() == before


def test_rejected_arguments_run_nothing():
    module = stinkytofu.load_stir(KERNEL)
    before = module.serialize()
    with pytest.raises(
        ValueError, match="RemoveInstructionPass rejected its arguments"
    ):
        module.run_passes(["RemoveInstructionPass=v_mov_b32", "RemoveInstructionPass"])
    assert module.serialize() == before


def test_tile_reaches_the_pass():
    one_wave = stinkytofu.load_stir(TDM_STREAM)
    [record] = one_wave.run_passes(["TDMLoadWaveSyncPass"])
    assert not record.changed
    two_waves = stinkytofu.load_stir(TDM_STREAM)
    [record] = two_waves.run_passes(["TDMLoadWaveSyncPass"], tile={"NumWaves": 2})
    assert record.changed
    assert "st.s_barrier_signal" in two_waves.serialize()


def test_features_take_the_two_flags():
    features = {"unroll_gemm": False, "distribute_global_read": False}
    records = stinkytofu.load_stir(KERNEL).run_passes(
        ["StinkyRemoveNopPass"], features=features
    )
    assert len(records) == 1


@pytest.mark.parametrize(
    "options, error, message",
    [
        ({"print_before": "text"}, ValueError, 'print_before must be "stir" or "asm"'),
        ({"tile": {"NumWave": 2}}, ValueError, "unknown tile key 'NumWave'"),
        ({"tile": {"NumWaves": -1}}, ValueError, r"must be in \[0, 4294967295\]"),
        ({"tile": {"NumWaves": 2**32}}, ValueError, r"must be in \[0, 4294967295\]"),
        ({"tile": {"NumWaves": "2"}}, TypeError, "must be an int"),
        ({"features": {"unroll": True}}, ValueError, "unknown features key 'unroll'"),
        ({"features": {"unroll_gemm": 1}}, TypeError, "must be a bool"),
    ],
)
def test_bad_run_options_run_nothing(options, error, message):
    module = stinkytofu.load_stir(KERNEL)
    before = module.serialize()
    with pytest.raises(error, match=message):
        module.run_passes(["RemoveInstructionPass=v_mov_b32"], **options)
    assert module.serialize() == before


def test_print_before_and_after():
    [record] = stinkytofu.load_stir(KERNEL).run_passes(
        ["RemoveInstructionPass=v_mov_b32"], print_before="stir", print_after="asm"
    )
    assert "*** Before Pass: RemoveInstructionPass ***" in record.before
    assert '"st.v_mov_b32"(v1)' in record.before
    assert record.after == (
        "\n*** After Pass: RemoveInstructionPass ***\nv_add_f32 v2, v0, v1\n"
    )


def test_entry_dump_includes_callees():
    records = stinkytofu.load_stir(MODULE).run_passes(
        ["StinkyRemoveNopPass"], print_before="asm"
    )
    assert [r.before.count("callee Function: helper") for r in records] == [1, 0]


EDIT_KERNEL = """\
st.func @k() {
^entry:
  v[0:3] = "st.ds_load_b128"(v40) { issueCycles = 4, latencyCycles = 56 }
  "st.s_cbranch_scc1"(loop) { issueCycles = 1, latencyCycles = 1 }
  Successors: ^loop, ^exit
^loop:
  "st.s_nop"(0) { issueCycles = 1, latencyCycles = 1 }
  Successors: ^exit
^exit:
  "st.s_endpgm"() { issueCycles = 1, latencyCycles = 1 }
}
"""

NOP = '  "st.s_nop"({}) {{ issueCycles = 1, latencyCycles = 1 }}\n'


def test_verify_is_empty_when_nothing_is_wrong():
    assert stinkytofu.load_stir(KERNEL).verify() == []
    assert stinkytofu.load_stir(EDIT_KERNEL).verify(function="k") == []


def test_set_operand_shows_in_functions_and_verify():
    module = stinkytofu.load_stir(EDIT_KERNEL)
    module.set_operand("k", "entry", 0, ("defs", 0), ("v", 0, 2))
    load = module.functions()[0].blocks()[0].instructions()[0]
    assert load.defs == [("v", 0, 2)]

    [diagnostic] = module.verify()
    assert (diagnostic.level, diagnostic.id) == ("error", "register-width")
    assert (diagnostic.function, diagnostic.block, diagnostic.index) == (
        "k",
        "entry",
        0,
    )
    assert (diagnostic.line, diagnostic.column) == (None, None)
    assert str(diagnostic) == (
        "@k ^entry #0: error: 'ds_load_b128' operand dest[0] has register width 2,"
        " expected 4 [register-width]"
    )

    module.set_operand("k", "entry", 0, ("defs", 0), ("v", 0, 4))
    module.set_operand("k", "loop", 0, ("uses", 0), 7)
    assert module.verify() == []
    assert '"st.s_nop"(7)' in module.serialize()


@pytest.mark.parametrize(
    "args, error, message",
    [
        (("k", "loop", 0, ("uses", 0), True), ValueError, "is not an operand"),
        (("k", "loop", 0, ("uses", 0), (3, "v")), ValueError, "is an SSA value"),
        (("k", "loop", 0, ("uses", 0), ("q", 0, 1)), ValueError, "is not an operand"),
        (("k", "loop", 0, ("uses", 0), 2**40), ValueError, "does not fit"),
        (("k", "loop", 0, ("srcs", 0), 1), ValueError, r"\(\"defs\", i\)"),
        (("k", "loop", 1, ("uses", 0), 1), IndexError, "no instruction 1"),
        (("k", "loop", 0, ("uses", 1), 1), IndexError, r"no uses\[1\]"),
        (("k", "nowhere", 0, ("uses", 0), 1), ValueError, "blocks: '\\^entry'"),
    ],
)
def test_set_operand_rejects_what_it_cannot_write(args, error, message):
    module = stinkytofu.load_stir(EDIT_KERNEL)
    before = module.serialize()
    with pytest.raises(error, match=message):
        module.set_operand(*args)
    assert module.serialize() == before


def test_set_operand_refuses_a_lifted_function():
    module = stinkytofu.load_stir(KERNEL)
    module.run_passes(["LiftAsmRegistersToSSAPass"])
    with pytest.raises(ValueError, match="attached SSA"):
        module.set_operand("k", "entry", 0, ("uses", 0), ("v", 3, 1))


def test_replace_block_keeps_the_header_and_reloads():
    module = stinkytofu.load_stir(EDIT_KERNEL)
    body = NOP.format(3) + '  "st.s_branch"(exit)\n'
    assert module.replace_block("k", "loop", body) == [
        "2:22: warning: Instruction missing attributes (issueCycles, latencyCycles)"
        " [missing-attributes]"
    ]
    loop = module.functions()[0].blocks()[1]
    assert [inst.opcode for inst in loop.instructions()] == ["st.s_nop", "st.s_branch"]
    assert loop.successors() == ["exit"]
    printed = module.serialize()
    assert "^loop: { label }" in printed
    # The new branch names ^exit, so ^exit now starts with its label.
    assert "^exit: { label }" in printed
    assert stinkytofu.load_stir(printed).serialize() == printed
    assert module.verify() == []


@pytest.mark.parametrize(
    "body, message",
    [
        (
            NOP.format(0) + NOP.format(1).replace("s_nop", "s_nopp"),
            r"\[unknown-mnemonic\]",
        ),
        (NOP.format(0) + "  Successors: ^exit\n", r"2:3: error: .*\[syntax\]"),
        ("^loop:\n" + NOP.format(0), r"\[syntax\]"),
    ],
)
def test_failed_replace_block_leaves_the_module(body, message):
    module = stinkytofu.load_stir(EDIT_KERNEL)
    before = module.serialize()
    with pytest.raises(ValueError, match=message):
        module.replace_block("k", "loop", body)
    assert module.serialize() == before


def test_replace_block_drops_attached_ssa():
    module = stinkytofu.load_stir(KERNEL)
    module.run_passes(["LiftAsmRegistersToSSAPass"])
    module.replace_block(
        "k",
        "entry",
        '  v0 = "st.v_mov_b32"(v1) { issueCycles = 1, latencyCycles = 1 }\n',
    )
    [inst] = module.functions()[0].blocks()[0].instructions()
    assert inst.defs == [("v", 0, 1)]


def test_branch_to_a_missing_label_is_reported():
    module = stinkytofu.load_stir(EDIT_KERNEL)
    module.replace_block(
        "k", "loop", '  "st.s_branch"(nowhere) { issueCycles = 1, latencyCycles = 1 }\n'
    )
    [diagnostic] = module.verify()
    assert diagnostic.id == "undefined-label"
    assert (diagnostic.block, diagnostic.index) == ("loop", 0)
    assert "'nowhere'" in diagnostic.message


CHAIN_KERNEL = """\
st.func @k() {
^entry:
  "st.s_nop"(0) { issueCycles = 1, latencyCycles = 1 }
  Successors: ^mid
^mid:
  "st.s_nop"(1) { issueCycles = 1, latencyCycles = 1 }
  Successors: ^last
^last:
  "st.s_endpgm"() { issueCycles = 1, latencyCycles = 1 }
}
"""


def test_block_lines_rebuild_the_block_unchanged():
    module = stinkytofu.load_stir(EDIT_KERNEL)
    printed = module.serialize()
    assert module.block_lines("k", "entry") == [
        'v[0:3] = "st.ds_load_b128"(v40) { issueCycles = 4, latencyCycles = 56 }',
        '"st.s_cbranch_scc1"(loop) { issueCycles = 1, latencyCycles = 1 }',
    ]
    # ^loop's own label is the { label } on its header, not a body line.
    assert module.block_lines("k", "loop") == [NOP.format(0).strip()]
    for block in module.functions()[0].blocks():
        lines = module.block_lines("k", block.name)
        module.replace_block("k", block.name, "\n".join(lines))
    assert module.serialize() == printed


def test_block_lines_hold_every_line_of_assembler_source():
    module = stinkytofu.load_asm_file(ROUNDTRIP_KERNEL)
    emitted, printed = module.emit_asm(), module.serialize()
    [function] = module.functions()
    [block] = function.blocks()
    lines = module.block_lines(function.name, block.name)
    directives = [line for line in lines if line.startswith('"st.asm_directive"(')]
    labels = [line for line in lines if line.endswith(":") or ": { alignment" in line]
    assert len(block.instructions()) == 1938
    assert (len(lines), len(directives), len(labels)) == (2203, 218, 47)
    assert sum(line.startswith('"st.asm_directive"(".set"') for line in lines) == 82
    module.replace_block(function.name, block.name, "\n".join(lines))
    assert (module.emit_asm(), module.serialize()) == (emitted, printed)


def test_delete_block_hands_a_falling_predecessor_on():
    module = stinkytofu.load_stir(CHAIN_KERNEL)
    module.delete_block("k", "mid")
    entry, last = module.functions()[0].blocks()
    assert (entry.name, last.name) == ("entry", "last")
    assert entry.successors() == ["last"]
    printed = module.serialize()
    assert stinkytofu.load_stir(printed).serialize() == printed


@pytest.mark.parametrize(
    "block, message",
    [
        ("entry", "it is the entry block of function 'k'"),
        ("loop", r"'st.s_cbranch_scc1' at \^entry #1 branches to 'loop'"),
        ("nowhere", "no block '\\^nowhere'"),
    ],
)
def test_delete_block_refuses_the_entry_and_a_branch_target(block, message):
    module = stinkytofu.load_stir(EDIT_KERNEL)
    before = module.serialize()
    with pytest.raises(ValueError, match=message):
        module.delete_block("k", block)
    assert module.serialize() == before


def test_delete_function_keeps_the_entry():
    module = stinkytofu.load_stir(MODULE)
    with pytest.raises(ValueError, match="'main' cannot be deleted: it is the entry"):
        module.delete_function("main")
    with pytest.raises(
        ValueError, match="no function 'nope'; functions: 'main', 'helper'"
    ):
        module.delete_function("nope")
    module.delete_function("helper")
    assert [function.name for function in module.functions()] == ["main"]
    assert "@helper" not in module.serialize()


def test_live_ranges_and_pressure_read_the_lift():
    module = stinkytofu.load_stir(KERNEL)
    module.run_passes(["LiftAsmRegistersToSSAPass"])
    mov, add = module.functions()[0].blocks()[0].instructions()
    ranges = module.live_ranges("k")
    by_value = {live.value: live for live in ranges.ranges}
    assert set(by_value) == {mov.defs[0][0], add.defs[0][0], add.uses[1][0]}

    v0 = by_value[mov.defs[0][0]]
    assert (v0.reg_class, v0.width, v0.register) == ("v", 1, ("v", 0, 1))
    assert v0.segments == [(v0.start, v0.end)]
    assert v0.length == v0.end - v0.start
    assert str(v0) == f"%{v0.value}:v [3d,5d)"
    # The mov writes v0, and the add is the last to read it.
    assert ranges.place(v0.start) == ("entry", 0, "def")
    assert ranges.place(v0.end - 1) == ("entry", 1, "use")
    assert ranges.place(0) == ("entry", None, "use")
    assert ranges.place(ranges.slot_count) is None

    pressure = module.pressure("k")
    assert (pressure.vgpr, pressure.sgpr) == (2, 0)
    peak = pressure.peaks["v"]
    assert (peak.reg_class, peak.dwords, peak.place) == ("v", 2, ("entry", 0, "def"))
    assert ranges.place(peak.slot) == peak.place


@pytest.mark.parametrize("query", ["live_ranges", "pressure"])
def test_live_ranges_and_pressure_need_a_lift(query):
    module = stinkytofu.load_stir(KERNEL)
    with pytest.raises(ValueError, match="run LiftAsmRegistersToSSAPass first"):
        getattr(module, query)("k")
    with pytest.raises(ValueError, match="no function 'nowhere'"):
        getattr(module, query)("nowhere")


HAZARD_STREAM = """\
st.func @k() {
^entry:
  s0 = "st.s_mov_b32"(s1) { issueCycles = 1, latencyCycles = 1 }
  "st.tensor_load_to_lds"(s[0:3], s[4:11]) { issueCycles = 1, latencyCycles = 1 }
}
"""


def test_hazards_report_every_pair():
    [gap] = stinkytofu.load_stir(HAZARD_STREAM).hazards("k")
    assert (gap.rule, gap.required, gap.gap, gap.ok) == (
        "SaluSgprToMemAddr",
        8,
        0,
        False,
    )
    assert (gap.block, gap.producer, gap.consumer) == ("entry", 0, 1)
    assert (gap.producer_opcode, gap.consumer_opcode) == (
        "st.s_mov_b32",
        "st.tensor_load_to_lds",
    )

    waited = HAZARD_STREAM.replace(
        '  "st.tensor',
        '  "st.s_nop"(7) { issueCycles = 8, latencyCycles = 8 }\n  "st.tensor',
    )
    [gap] = stinkytofu.load_stir(waited).hazards("k")
    assert (gap.gap, gap.ok, gap.consumer) == (8, True, 2)
    assert stinkytofu.load_stir(KERNEL).hazards("k") == []


def test_size_lays_functions_out_in_order():
    size = stinkytofu.load_stir(KERNEL).size("k")
    assert (size.start, size.bytes, size.cycles, size.instructions) == (0, 8, 2, 2)
    assert size.labels == {}

    labelled = MODULE.replace(
        "@helper() {\n    ^entry:", "@helper() {\n    ^entry: { label }"
    )
    module = stinkytofu.load_stir(labelled)
    main, helper = module.size("main"), module.size("helper")
    assert helper.start == main.bytes
    assert helper.labels == {"entry": helper.start}


def test_copy_is_whole_and_separate():
    module = stinkytofu.load_asm_file(ROUNDTRIP_KERNEL)
    copy = module.copy()
    assert copy.serialize() == module.serialize()
    assert copy.emit_asm() == module.emit_asm()
    vgprs = "SignatureKernelDescriptor.totalVgprs"
    assert copy.getMetaDataU64(vgprs) == module.getMetaDataU64(vgprs)
    assert module.getMetaDataU64(vgprs) is not None
    assert copy.load_warnings() == module.load_warnings()

    kernel = stinkytofu.load_stir(KERNEL)
    before = kernel.serialize()
    edited = kernel.copy()
    edited.run_passes(["RemoveInstructionPass=v_mov_b32"])
    assert kernel.serialize() == before
    assert edited.serialize() != before


def test_run_info_records_each_run():
    module = stinkytofu.load_stir(KERNEL)
    assert module.run_info() == []
    module.run_passes(["StinkyRemoveNopPass"])
    module.run_passes(
        ["RemoveInstructionPass=v_mov_b32"],
        function="k",
        features={"unroll_gemm": False},
        tile={"NumWaves": 2},
    )
    module.run_passes([])
    first, second = module.run_info()
    assert (first.passes, first.function, first.features, first.tile) == (
        ["StinkyRemoveNopPass"],
        None,
        {},
        {},
    )
    assert (second.arch, second.version) == ((12, 5, 0), stinkytofu.version())
    assert (second.function, second.features, second.tile) == (
        "k",
        {"unroll_gemm": False},
        {"NumWaves": 2},
    )
    assert second.command == (
        "stinkytofu run --arch 12,5,0 --pass RemoveInstructionPass=v_mov_b32"
        " --function k --feature unroll_gemm=false --tile NumWaves=2"
    )
    assert str(second) == "stinkytofu %s: %s" % (
        stinkytofu.version(),
        second.command[len("stinkytofu ") :],
    )

    printed = module.serialize(comments=True)
    assert printed.splitlines()[:2] == ["// %s" % first, "// %s" % second]
    assert not module.serialize().startswith("//")
    assert stinkytofu.load_stir(printed).serialize() == module.serialize()
    assert module.emit_asm(comments=True).startswith("// %s\n// %s\n" % (first, second))
    assert [str(run) for run in module.copy().run_info()] == [str(first), str(second)]


def test_run_info_quotes_words_and_keeps_a_run_that_raised():
    module = stinkytofu.load_stir(
        'st.func @() {\n^entry:\n  "st.s_nop"(0) { issueCycles = 1, latencyCycles = 1 }\n}\n'
    )
    module.run_passes(["RemoveInstructionPass=s_nop, s_endpgm"], function="")
    assert module.run_info()[0].command.endswith(
        "--pass 'RemoveInstructionPass=s_nop, s_endpgm' --function ''"
    )
    with pytest.raises(ValueError, match="unknown pass"):
        module.run_passes(["NotARealPass"])
    assert len(module.run_info()) == 1
    # A refused pass still ran, and may have run passes before it.
    with pytest.raises(RuntimeError, match="no attached SSA"):
        module.run_passes(["RegisterAllocationPass"])
    assert module.run_info()[-1].passes == ["RegisterAllocationPass"]


def test_compare_runs_each_list_on_its_own_copy():
    module = stinkytofu.load_stir(KERNEL)
    before = module.serialize()
    result = stinkytofu.compare(
        module, ["StinkyRemoveNopPass"], ["LiftAsmRegistersToSSAPass"], function="k"
    )
    assert module.serialize() == before
    assert module.run_info() == []
    assert module.functions()[0].has_attached_ssa is False

    assert [record.pass_name for record in result.a.records] == ["StinkyRemoveNopPass"]
    assert [record.pass_name for record in result.b.records] == [
        "LiftAsmRegistersToSSAPass"
    ]
    assert result.b.module.functions()[0].has_attached_ssa is True
    assert [run.passes for run in result.b.module.run_info()] == [
        ["LiftAsmRegistersToSSAPass"]
    ]
    # The nop pass leaves the function unlifted, so side a has no pressure.
    assert result.a.pressure == {"k": None}
    assert result.b.pressure["k"].vgpr == 2
    assert result.a.size["k"].bytes == result.b.size["k"].bytes == 8
    assert result.a.hazards == result.b.hazards == {"k": []}
    assert str(result) == (
        "k: vgpr - -> 2, sgpr - -> 0, bytes 8 -> 8, cycles 2 -> 2, hazard violations 0 -> 0"
    )

    both = stinkytofu.compare(stinkytofu.load_stir(HAZARD_STREAM), [], [])
    assert str(both).endswith("hazard violations 1 -> 1")


def test_compare_names_the_side_that_failed():
    module = stinkytofu.load_stir(KERNEL)
    with pytest.raises(RuntimeError, match="^passes_b: missed: RegisterAllocation"):
        stinkytofu.compare(module, ["StinkyRemoveNopPass"], ["RegisterAllocationPass"])
    with pytest.raises(ValueError, match="^passes_a: unknown pass 'NotARealPass'"):
        stinkytofu.compare(module, ["NotARealPass"], [])
    assert module.run_info() == []


NOISY_KERNEL = """\
st.func @k() {
^entry:
  "st.s_nop"(0) { issueCycles = 1, latencyCycles = 1 }
  v0 = "st.v_mov_b32"(v1) { issueCycles = 1, latencyCycles = 1 }
  "st.s_nop"(1) { issueCycles = 1, latencyCycles = 1 }
  v2 = "st.v_add_f32"(v0, v1) { issueCycles = 1, latencyCycles = 1 }
  "st.s_nop"(2) { issueCycles = 1, latencyCycles = 1 }
  "st.s_branch"(tail) { issueCycles = 1, latencyCycles = 1 }
  Successors: ^tail
^tail:
  "st.s_endpgm"() { issueCycles = 1, latencyCycles = 1 }
}
"""


def opcodes_of(module):
    return [
        inst.opcode
        for function in module.functions()
        for block in function.blocks()
        for inst in block.instructions()
    ]


def adds_v1(module):
    return any(
        inst.opcode == "st.v_add_f32" and ("v", 1, 1) in inst.uses
        for function in module.functions()
        for block in function.blocks()
        for inst in block.instructions()
    )


def test_reduce_keeps_only_what_the_check_needs():
    module = stinkytofu.load_stir(NOISY_KERNEL)
    before = module.serialize()
    log = []
    result = stinkytofu.reduce(module, adds_v1, log=log.append)
    assert result.module.serialize() == (
        "st.module @k {\n  st.func @k() {\n    ^entry:\n"
        '      v2 = "st.v_add_f32"(v0, v1) { issueCycles = 1, latencyCycles = 1 }\n'
        "  }\n}\n"
    )
    assert (result.functions, result.blocks, result.lines) == ((1, 1), (2, 1), (7, 1))
    assert result.complete
    assert str(result) == (
        "functions 1 -> 1, blocks 2 -> 1, lines 7 -> 1 after %d checks" % result.checks
    )
    # ^tail goes once the branch naming it has gone.
    assert log[0] == "removed 3 lines of ^entry in 'k'; 3 left"
    assert log[-1] == "removed 1 block of 'k'; 1 left"
    assert module.serialize() == before


def test_reduce_removes_callable_functions():
    def main_has_a_nop(module):
        [main] = [
            function for function in module.functions() if function.name == "main"
        ]
        return any(
            inst.opcode == "st.s_nop"
            for block in main.blocks()
            for inst in block.instructions()
        )

    result = stinkytofu.reduce(stinkytofu.load_stir(MODULE), main_has_a_nop)
    assert [function.name for function in result.module.functions()] == ["main"]
    assert result.functions == (2, 1)


def test_reduce_offers_no_candidate_verify_rejects():
    module = stinkytofu.load_asm(
        "s_nop 0\ns_cbranch_scc1 label_x\ns_nop 1\nlabel_x:\ns_endpgm\n"
    )
    seen = []

    def branches(candidate):
        seen.append([diagnostic.id for diagnostic in candidate.verify()])
        return "st.s_cbranch_scc1" in opcodes_of(candidate)

    result = stinkytofu.reduce(module, branches)
    # The label stays: without it, the branch is an undefined-label.
    assert result.module.block_lines("", "entry") == [
        '"st.s_cbranch_scc1"(label_x) { issueCycles = 1, latencyCycles = 1 }',
        "label_x:",
    ]
    assert len(seen) > 1 and all(ids == [] for ids in seen)


def test_reduce_checks_the_module_first():
    module = stinkytofu.load_stir(NOISY_KERNEL)
    with pytest.raises(ValueError, match="check does not report the module as failing"):
        stinkytofu.reduce(module, lambda candidate: False)
    with pytest.raises(ZeroDivisionError):
        stinkytofu.reduce(module, lambda candidate: 1 / 0)
    with pytest.raises(ValueError, match="max_checks must be at least 1"):
        stinkytofu.reduce(module, adds_v1, max_checks=0)


def test_reduce_takes_a_check_that_raises_as_not_failing():
    calls = []

    def only_the_first(candidate):
        calls.append(candidate)
        if len(calls) > 1:
            raise RuntimeError("missed: RegisterAllocation")
        return True

    module = stinkytofu.load_stir(NOISY_KERNEL)
    result = stinkytofu.reduce(module, only_the_first)
    assert result.module.serialize() == module.serialize()
    assert result.complete and result.checks == len(calls) > 1


def test_reduce_stops_at_max_checks():
    result = stinkytofu.reduce(
        stinkytofu.load_stir(NOISY_KERNEL), adds_v1, max_checks=2
    )
    assert (result.checks, result.complete) == (2, False)
    assert str(result).endswith(" after 2 checks, stopped at max_checks")
    assert adds_v1(result.module)


def write_test(tmp_path, body):
    """A test command: a Python script that reads the candidate as text."""
    script = tmp_path / "fails.py"
    script.write_text(
        "import os, signal, sys\ntext = open(sys.argv[1]).read()\n" + body
    )
    return [sys.executable, str(script)]


def test_command_check_runs_a_command_on_each_candidate(tmp_path):
    test = write_test(
        tmp_path,
        'ok = sys.argv[1].endswith(".stir") and "v_add_f32" in text\n'
        "sys.exit(0 if ok else 1)\n",
    )
    result = stinkytofu.reduce(
        stinkytofu.load_stir(NOISY_KERNEL), stinkytofu.command_check(test)
    )
    assert opcodes_of(result.module) == ["st.v_add_f32"]


def test_command_check_writes_assembly_for_asm(tmp_path):
    test = write_test(
        tmp_path,
        'ok = sys.argv[1].endswith(".s") and "v_mov_b32 v0, v1" in text\n'
        "sys.exit(0 if ok else 1)\n",
    )
    check = stinkytofu.command_check(test, form="asm")
    result = stinkytofu.reduce(stinkytofu.load_asm(ASM), check)
    assert result.module.emit_asm() == "v_mov_b32 v0, v1\n"


def test_command_check_outlives_a_test_that_dies(tmp_path):
    # A candidate without the v_mov kills the test process, as an abort in a
    # pass would; that candidate only counts as not failing.
    test = write_test(
        tmp_path,
        'if "v_mov_b32" not in text:\n'
        "    os.kill(os.getpid(), signal.SIGKILL)\n"
        'sys.exit(0 if "v_add_f32" in text else 1)\n',
    )
    result = stinkytofu.reduce(
        stinkytofu.load_stir(NOISY_KERNEL), stinkytofu.command_check(test)
    )
    assert opcodes_of(result.module) == ["st.v_mov_b32", "st.v_add_f32"]


def test_command_check_stops_a_slow_test(tmp_path):
    check = stinkytofu.command_check(
        write_test(tmp_path, "import time\ntime.sleep(30)\n"), timeout=0.2
    )
    assert check(stinkytofu.load_stir(KERNEL)) is False


def test_command_check_refuses_what_it_cannot_run(tmp_path):
    with pytest.raises(ValueError, match='form must be "stir" or "asm"'):
        stinkytofu.command_check(["true"], form="llvm")
    missing = stinkytofu.command_check([str(tmp_path / "missing")])
    with pytest.raises(OSError):
        stinkytofu.reduce(stinkytofu.load_stir(KERNEL), missing)


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


def test_cli_prints_dumps_to_stderr(tmp_path, capsys):
    path = tmp_path / "k.stir"
    path.write_text(KERNEL)
    argv = ["run", str(path), "--pass", "RemoveInstructionPass=v_mov_b32"]
    assert main(argv + ["--print-before", "asm", "--print-after", "stir"]) == 0
    captured = capsys.readouterr()
    assert (
        "*** Before Pass: RemoveInstructionPass ***\nv_mov_b32 v0, v1\n" in captured.err
    )
    assert "*** After Pass: RemoveInstructionPass ***" in captured.err
    assert "***" not in captured.out
    stinkytofu.load_stir(captured.out)


def test_cli_repeats_a_recorded_run(tmp_path, capsys):
    path = tmp_path / "k.stir"
    path.write_text(TDM_STREAM)
    module = stinkytofu.load_stir(TDM_STREAM)
    module.run_passes(
        ["TDMLoadWaveSyncPass"], features={"unroll_gemm": False}, tile={"NumWaves": 2}
    )
    [run] = module.run_info()
    assert main(shlex.split(run.command)[1:] + [str(path)]) == 0
    printed = capsys.readouterr().out
    assert printed == module.serialize()
    assert "st.s_barrier_signal" in printed


@pytest.mark.parametrize(
    "setting, message",
    [
        (["--feature", "unroll_gemm=maybe"], "must be true or false"),
        (["--feature", "unroll_gemm"], "takes NAME=VALUE"),
        (["--tile", "NumWaves=two"], "must be an integer"),
        (["--tile", "Bogus=1"], "unknown tile key 'Bogus'"),
    ],
)
def test_cli_rejects_a_bad_setting(tmp_path, capsys, setting, message):
    path = tmp_path / "k.stir"
    path.write_text(KERNEL)
    assert main(["run", str(path), "--pass", "StinkyRemoveNopPass"] + setting) == 2
    assert message in capsys.readouterr().err


def test_cli_runs_one_function(tmp_path, capsys):
    path = tmp_path / "m.stir"
    path.write_text(MODULE)
    argv = ["run", str(path), "--pass", "StinkyRemoveNopPass", "--function", "helper"]
    assert main(argv) == 0
    printed = stinkytofu.load_stir(capsys.readouterr().out)
    counts = [
        sum(len(b.instructions()) for b in fn.blocks()) for fn in printed.functions()
    ]
    assert counts == [1, 0]
    assert main(["run", str(path), "--function", "nowhere"]) == 2
    assert "no function 'nowhere'" in capsys.readouterr().err


def test_cli_reduces_a_stir_file(tmp_path, capsys):
    path = tmp_path / "k.stir"
    path.write_text(NOISY_KERNEL)
    python, script = write_test(tmp_path, 'sys.exit(0 if "v_add_f32" in text else 1)\n')
    out = tmp_path / "small.stir"
    argv = ["reduce", str(path), "--test", python, "--test-arg", script]
    assert main(argv + ["-o", str(out)]) == 0
    assert opcodes_of(stinkytofu.load_stir(out.read_text())) == ["st.v_add_f32"]
    err = capsys.readouterr().err
    assert "removed 3 lines of ^entry in 'k'; 3 left\n" in err
    assert "functions 1 -> 1, blocks 2 -> 1, lines 7 -> 1 after " in err

    assert main(argv + ["--max-checks", "1"]) == 0
    captured = capsys.readouterr()
    assert captured.out == stinkytofu.load_stir(NOISY_KERNEL).serialize()
    assert captured.err.endswith("after 1 check, stopped at max_checks\n")


def test_cli_reduces_assembler_source(tmp_path, capsys):
    path = tmp_path / "k.s"
    path.write_text(ASM)
    python, script = write_test(tmp_path, 'sys.exit(0 if "v_mov_b32" in text else 1)\n')
    assert main(["reduce", str(path), "--test", python, "--test-arg", script]) == 0
    assert capsys.readouterr().out == "v_mov_b32 v0, v1\n"


def test_cli_reduce_needs_a_test_that_fails_on_the_input(tmp_path, capsys):
    path = tmp_path / "k.stir"
    path.write_text(KERNEL)
    python, script = write_test(tmp_path, "sys.exit(1)\n")
    assert main(["reduce", str(path), "--test", python, "--test-arg", script]) == 2
    assert "check does not report the module as failing" in capsys.readouterr().err
    assert main(["reduce", str(path), "--test", str(tmp_path / "missing")]) == 2
    assert "No such file or directory" in capsys.readouterr().err


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
