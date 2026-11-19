# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for tensilelite.SgprHazards, the gfx1200/gfx1201 SGPR read-hazard wait insertion.

The expected waits follow LLVM's AMDGPUWaitSGPRHazards (llvm/test/CodeGen/AMDGPU/
valu-read-sgpr-hazard.mir); the TensileLite sequences come from the gfx1201 kernels of ROCM-33234.
"""

import textwrap

import pytest

from tensilelite.SgprHazards import (SgprHazardError, analyze, insertSgprHazardWaits, waitText,
                                     withSgprHazardWaits)

pytestmark = pytest.mark.unit

SA = "s_wait_alu depctr_sa_sdst(0)"
VA = "s_wait_alu depctr_va_sdst(0)"
VCC = "s_wait_alu depctr_va_vcc(0)"


def waits(asm):
    """[(instruction, wait)] for every instruction that needs a wait before it."""
    lines = textwrap.dedent(asm).strip("\n").split("\n")
    ops, need = analyze(lines)
    return [(ops[i].raw.split("//")[0].strip(), waitText(w)) for i, w in sorted(need.items())]


def insert(asm):
    return insertSgprHazardWaits(textwrap.dedent(asm).strip("\n"))


def test_salu_write_then_salu_read_of_valu_read_pair():
    # hazard_vcc1: the VALU read of s2 tracks the pair s[2:3].
    assert waits("""
        v_cndmask_b32_e64 v1, v1, v2, s2
        s_cselect_b32 s3, -1, 0
        s_add_u32 s4, s3, 0
        s_endpgm
        """) == [("s_add_u32 s4, s3, 0", SA)]


def test_untracked_pair_needs_no_wait():
    # hazard_valu_write1_no_hazard: a VALU write does not track the pair, only a VALU read does.
    assert waits("""
        v_cmp_eq_f32_e64 s1, v0, v1
        s_cselect_b32 s1, -1, 0
        s_add_u32 s2, s1, 0
        s_endpgm
        """) == []


def test_vcc_written_by_salu_and_valu_on_merging_paths():
    # hazard_merge_vcc
    assert waits("""
          v_cndmask_b32_e32 v1, v1, v2, vcc_lo
          s_cbranch_scc0 label_a
          s_branch label_b
        label_a:
          s_cselect_b32 vcc_lo, -1, 0
          s_branch label_c
        label_b:
          v_cmp_eq_f32_e64 vcc_lo, v0, v1
          s_branch label_c
        label_c:
          v_cndmask_b32_e32 v3, v4, v5, vcc_lo
          s_endpgm
        """) == [("v_cndmask_b32_e32 v3, v4, v5, vcc_lo",
                  "s_wait_alu depctr_va_vcc(0) depctr_sa_sdst(0)")]


def test_vcc_valu_write_then_valu_read():
    # A VOPC _e32 writes VCC; the next VALU read of VCC needs va_vcc, not va_sdst.
    assert waits("""
        v_cndmask_b32_e32 v1, v1, v2, vcc_lo
        v_cmp_eq_u32_e32 vcc_lo, v0, v1
        v_cndmask_b32_e32 v3, v4, v5, vcc_lo
        v_cmp_eq_u32_e32 v0, v1
        v_dual_cndmask_b32 v6, v7, v8 :: v_dual_mov_b32 v9, v10
        s_endpgm
        """) == [("v_cndmask_b32_e32 v3, v4, v5, vcc_lo", VCC),
                 ("v_dual_cndmask_b32 v6, v7, v8 :: v_dual_mov_b32 v9, v10", VCC)]


def test_cbranch_vccz_reads_vcc():
    assert waits("""
          v_cndmask_b32_e32 v1, v1, v2, vcc_lo
          s_and_b32 vcc_lo, exec_lo, s3
          s_cbranch_vccz label_x
        label_x:
          s_endpgm
        """) == [("s_cbranch_vccz label_x", SA)]


def test_tail_loop_lane_masks():
    # bf16 MT32x128x64 DTVB1 tail loop: the loop back edge tracks s66, so both mask reads need
    # va_sdst; the remainder update needs sa_sdst before its compare, which also covers the
    # v_cmp at the top of the next iteration.
    assert waits("""
        label_TailLoopBeginL:
          v_cmp_ge_i32_e64 s66, v197, s12
          v_cndmask_b32_e64 v180, v180, 0, s66
          v_cmp_ge_i32_e64 s66, v197, s12
          v_cndmask_b32_e64 v52, v52, 0, s66
          s_sub_co_i32 s12, s12, 16
          s_cmp_le_i32 s12, 0
          s_cbranch_scc0 label_TailLoopBeginL
          s_endpgm
        """) == [("v_cndmask_b32_e64 v180, v180, 0, s66", VA),
                 ("v_cndmask_b32_e64 v52, v52, 0, s66", VA),
                 ("s_cmp_le_i32 s12, 0", SA)]


def test_epilogue_scratch_sgpr():
    # Deepbench kernel epilogue: s8 is read by VALU, then reused for the activation compare, the
    # bias/ScaleAlphaVec LDS offset and the bf16 NaN mask.
    assert waits("""
        v_add_nc_u32_e32 v136, s8, v222
        s_mov_b32 s8, 3
        s_cmp_eq_u32 s56, s8
        s_mul_i32 s8, 0x80, s2
        v_sub_nc_u32_e64 v147, v128, s8
        v_cmp_u_f32_e64 s8, v152, v152
        v_cndmask_b32_e64 v152, v141, v143, s8
        s_endpgm
        """) == [("s_cmp_eq_u32 s56, s8", SA),
                 ("v_sub_nc_u32_e64 v147, v128, s8", SA),
                 ("v_cndmask_b32_e64 v152, v141, v143, s8", VA)]


def test_symbolic_sgpr_operands():
    assert waits("""
        .set sgprSrdD, 16
        .set sgprAlpha, sgprSrdD+20
        v_mov_b32 v1, s[sgprAlpha]
        s_mov_b32 s[sgprAlpha], 1.0
        v_mul_f32_e64 v2, v3, s36
        v_mov_b32 v7, s[sgprSrdD]
        s_mov_b32 s[sgprSrdD+1], 0
        v_cndmask_b32_e64 v4, v5, v6, s[sgprSrdD:sgprSrdD+1]
        s_endpgm
        """) == [("v_mul_f32_e64 v2, v3, s36", SA),
                 ("v_cndmask_b32_e64 v4, v5, v6, s[sgprSrdD:sgprSrdD+1]", SA)]


def test_sgpr_operand_followed_by_a_modifier():
    # rocisa appends VOP3P modifiers after a space, inside the last operand.
    assert waits("""
        v_mov_b32 v1, s40
        s_mov_b32 s40, 1.0
        v_pk_mul_f32 v[0:1], v[0:1], s[40:41] op_sel_hi:[1,0,1]
        s_endpgm
        """) == [("v_pk_mul_f32 v[0:1], v[0:1], s[40:41] op_sel_hi:[1,0,1]", SA)]


def test_sendmsg_rtn_writes_its_destination():
    assert waits("""
        v_mov_b32 v1, s20
        s_sendmsg_rtn_b32 s20, sendmsg(MSG_RTN_GET_REALTIME)
        v_mov_b32 v2, s20
        s_endpgm
        """) == [("v_mov_b32 v2, s20", SA)]


@pytest.mark.parametrize("asm", [
    "s_mov_b32 s[sgprUnknown], 0",                    # unresolved symbol
    "v_mov_b32 v1, s5.l",                             # SGPR spelling the parser does not know
    "s_bitreplicate_b64_b32 s[0:1], s2",              # SALU mnemonic of unknown shape
    "v_new_op_b32 s0, v1",                            # VALU writing an SGPR it does not know
    "s_movrels_b32 s0, s1",                           # M0-relative SGPR access
    "my_macro s1, s2",                                # not an instruction
    ".if 1\ns_nop 0\n.endif",                         # conditional assembly
    ".long 0xbf800000",                               # raw words in code
    ".macro M a\ns_nop 0\n.endm\nM 1",                # macro invocation
    "s_wait_alu foo",                                 # unknown s_wait_alu operand
])
def test_parser_fails_closed(asm):
    with pytest.raises(SgprHazardError):
        waits(asm)


@pytest.mark.parametrize("between, needsWait", [
    ("s_load_b32 s0, s[2:3], 0x0", False),           # SMEM retires pending writes
    ("buffer_load_b32 v9, v8, s[20:23], 0 offen", False),  # so does non-FLAT VMEM
    ("global_load_b32 v9, v8, s[20:21]", True),      # FLAT does not
    ("ds_load_b32 v9, v8", True),
    ("s_wait_alu depctr_sa_sdst(0)", False),
    ("s_wait_alu depctr_va_vdst(0)", True),
    ("s_wait_alu 0xfffe", False),
    ("s_wait_alu 0xf1ff", True),                     # va_sdst only
    ("s_nop 7", True),
])
def test_what_retires_a_pending_salu_write(between, needsWait):
    found = waits(f"""
        v_mov_b32 v1, s4
        s_mov_b32 s4, 1
        {between}
        v_mov_b32 v2, s4
        s_endpgm
        """)
    assert found == ([("v_mov_b32 v2, s4", SA)] if needsWait else [])


@pytest.mark.parametrize("between, needsWait", [
    ("s_wait_alu 0xf1ff", False),                    # va_sdst(0)
    ("s_wait_alu 0xfffd", True),                     # va_vcc(0) only
    ("s_wait_alu depctr_va_sdst(0)", False),
])
def test_what_retires_a_pending_valu_write(between, needsWait):
    found = waits(f"""
        v_mov_b32 v1, s4
        v_readfirstlane_b32 s4, v2
        {between}
        v_mov_b32 v3, s4
        s_endpgm
        """)
    assert found == ([("v_mov_b32 v3, s4", VA)] if needsWait else [])


def test_salu_read_retires_valu_writes():
    assert waits("""
        v_mov_b32 v1, s4
        v_readfirstlane_b32 s4, v2
        s_add_u32 s5, s4, 1
        v_mov_b32 v3, s4
        s_endpgm
        """) == []


def test_indirect_jump_flushes_and_callee_entry_is_conservative():
    # Activation calls: the call target's address is taken, the call writes s[28:29] and the callee
    # returns through it at once.
    assert waits("""
          v_mov_b32 v1, s9
          s_mov_b32 s9, 1
          s_getpc_b64 s[12:13]
          s_add_i32 s14, label_fn, 4
          s_add_u32 s12, s12, s14
          s_swappc_b64 s[28:29], s[12:13]
          s_endpgm
        label_fn:
          s_setpc_b64 s[28:29]
        """) == [("s_swappc_b64 s[28:29], s[12:13]", SA), ("s_setpc_b64 s[28:29]", SA)]


def test_long_branch_target_is_entered_conservatively():
    # The SLongBranchPositive shape: label_far is only reached through s_setpc_b64, so it is entered
    # with every SGPR's write pending; the jump arithmetic itself uses untracked SGPRs.
    assert waits("""
          v_mov_b32 v1, s9
          s_getpc_b64 s[80:81]
          s_add_i32 s82, label_far, 4
          s_add_u32 s80, s80, s82
          s_addc_u32 s81, s81, 0
          s_setpc_b64 s[80:81]
        label_far:
          v_mov_b32 v2, s9
          s_endpgm
        """) == [("v_mov_b32 v2, s9", SA)]


def test_wait_never_splits_an_s_getpc_sequence():
    with pytest.raises(SgprHazardError):
        insert("""
            v_mov_b32 v1, s4
            s_mov_b32 s4, 1
            s_getpc_b64 s[12:13]
            s_add_u32 s5, s4, 4
            s_endpgm
            """)


def test_label_and_instruction_on_one_line():
    out, n = insert("""
          v_mov_b32 v1, s4
          s_mov_b32 s4, 1
          s_cbranch_scc0 label_x
          s_endpgm
        label_x: v_mov_b32 v2, s4
          s_endpgm
        """)
    assert n == 1
    assert f"label_x:\n{SA} // SGPR read hazard\nv_mov_b32 v2, s4" in out


def test_wait_goes_before_a_delay_hint_and_folds_into_a_wait():
    out, n = insert("""
        v_mov_b32 v1, s4
        v_mov_b32 v5, s6
        s_mov_b32 s4, 1
        s_delay_alu instid0(SALU_CYCLE_1)
        s_add_u32 s5, s4, 1
        v_readfirstlane_b32 s6, v2
        s_wait_alu depctr_vm_vsrc(0) // keep
        v_mov_b32 v3, s6
        s_endpgm
        """)
    assert n == 2
    assert f"{SA} // SGPR read hazard\ns_delay_alu instid0(SALU_CYCLE_1)\ns_add_u32" in out
    assert "s_wait_alu depctr_vm_vsrc(0) depctr_va_sdst(0) // keep\nv_mov_b32 v3, s6" in out


def test_metadata_and_block_comments_are_not_code():
    assert waits("""
        /* temp0 = (wg//CU_Count) */
        v_mov_b32 v1, s4 /* ok */
        s_mov_b32 s4, 1
        .amdgpu_metadata
        amdhsa.version:
          - 1
        .end_amdgpu_metadata
        v_mov_b32 v2, s4
        s_endpgm
        """) == [("v_mov_b32 v2, s4", SA)]


def test_objdump_branch_targets_come_from_the_encoding():
    # llvm-objdump prints one name per address; the branch below names an alias of label_b.
    assert waits("""
        0000000000000100 <label_a>:
        \tv_mov_b32_e32 v1, s4                // 000000000100: 7E020204
        \ts_mov_b32 s4, 1                     // 000000000104: BE840081
        \ts_branch label_alias                // 000000000108: BFA00000
        000000000000010c <label_b>:
        \tv_mov_b32_e32 v2, s4                // 00000000010C: 7E040204
        \ts_endpgm                            // 000000000110: BFB00000
        """) == [("v_mov_b32_e32 v2, s4", SA)]


def test_with_waits_follows_the_cap():
    asm = "v_mov_b32 v1, s4\ns_mov_b32 s4, 1\nv_mov_b32 v2, s4\ns_endpgm"
    assert SA in withSgprHazardWaits(asm, {"HasVALUReadSGPRHazard": 1})
    assert withSgprHazardWaits(asm, {"HasVALUReadSGPRHazard": 0}) == asm
    with pytest.raises(KeyError):
        withSgprHazardWaits(asm, {})


def test_insert_is_idempotent_and_keeps_indentation():
    asm = textwrap.dedent("""
        label_x:
          v_cmp_ge_i32_e64 s66, v197, s12
          v_cndmask_b32_e64 v180, v180, 0, s66
          v_cmp_ge_i32_e64 s66, v197, s12
          v_cndmask_b32_e64 v52, v52, 0, s66 // B mask
          s_endpgm
        """)
    out, n = insertSgprHazardWaits(asm)
    assert n == 1
    assert "  s_wait_alu depctr_va_sdst(0) // SGPR read hazard\n  v_cndmask_b32_e64 v52" in out
    assert insertSgprHazardWaits(out) == (out, 0)
