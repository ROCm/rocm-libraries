# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for tensilelite.SgprHazards, the gfx1200/gfx1201 SGPR read-hazard wait insertion.

The expected waits follow LLVM's AMDGPUWaitSGPRHazards (llvm/test/CodeGen/AMDGPU/
valu-read-sgpr-hazard.mir); the TensileLite sequences come from the gfx1201 kernels of ROCM-33234.
"""

import textwrap

import pytest

from tensilelite.SgprHazards import analyze, insertSgprHazardWaits, waitText

SA = "s_wait_alu depctr_sa_sdst(0)"
VA = "s_wait_alu depctr_va_sdst(0)"


def waits(asm):
    """[(instruction, wait)] for every instruction that needs a wait before it."""
    lines = textwrap.dedent(asm).strip("\n").split("\n")
    ops, need = analyze(lines)
    return [(ops[i].raw.split("//")[0].strip(), waitText(w)) for i, w in sorted(need.items())]


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


def test_unresolved_symbolic_sgpr_is_an_error():
    with pytest.raises(ValueError):
        waits("s_mov_b32 s[sgprUnknown], 0")


@pytest.mark.parametrize("between, needsWait", [
    ("s_load_b32 s0, s[2:3], 0x0", False),           # SMEM retires pending writes
    ("buffer_load_b32 v9, v8, s[20:23], 0 offen", False),  # so does non-FLAT VMEM
    ("global_load_b32 v9, v8, s[20:21]", True),      # FLAT does not
    ("ds_load_b32 v9, v8", True),
    ("s_wait_alu depctr_sa_sdst(0)", False),
    ("s_wait_alu depctr_va_vdst(0)", True),
    ("s_wait_alu 0xfffe", False),
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


def test_salu_read_retires_valu_writes():
    assert waits("""
        v_mov_b32 v1, s4
        v_readfirstlane_b32 s4, v2
        s_add_u32 s5, s4, 1
        v_mov_b32 v3, s4
        s_endpgm
        """) == []


def test_valu_write_then_valu_read():
    assert waits("""
        v_mov_b32 v1, s4
        v_readfirstlane_b32 s4, v2
        v_mov_b32 v3, s4
        s_endpgm
        """) == [("v_mov_b32 v3, s4", VA)]


def test_indirect_jump_flushes_and_callee_entry_is_conservative():
    # Activation calls: the call target's address is taken, the call writes s[28:29] and the callee
    # returns through it at once.
    assert waits("""
          v_mov_b32 v1, s9
          s_mov_b32 s9, 1
          s_getpc_b64 s[12:13]
          s_add_i32 s14, label_fn-label_pc, 4
          s_add_u32 s12, s12, s14
          s_swappc_b64 s[28:29], s[12:13]
          s_endpgm
        label_fn:
          s_setpc_b64 s[28:29]
        """) == [("s_swappc_b64 s[28:29], s[12:13]", SA), ("s_setpc_b64 s[28:29]", SA)]


def test_vopd_and_macros():
    assert waits("""
        .macro M
          v_mov_b32 v0, s4
        .endm
        v_dual_mov_b32 v1, s4 :: v_dual_mov_b32 v2, v3
        s_mov_b32 s4, 0
        s_add_u32 s5, s4, 1
        s_endpgm
        """) == [("s_add_u32 s5, s4, 1", SA)]


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
