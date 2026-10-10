# RUN: %stinkytofu-opt --arch gfx1250 %s --whole-function --InsertWaitAluPass --emit-asm 2>/dev/null
#
# In expert mode an SALU write to M0 that a named-barrier op reads needs
# s_wait_alu depctr_sa_sdst(0) in between (sp3 p.302; asm_verify fence-m0).
# InsertWaitAlu turns expert mode on for the whole kernel, including code the
# input ran in mode 0 without the fence, so it adds it: on a straight line, and
# at a join where only one path wrote M0. An input fence is not duplicated.
# Before: no wait before either s_barrier_join / s_barrier_signal.

# CHECK:      s_mov_b32 m0, s26
# CHECK-NEXT: s_wait_alu depctr_sa_sdst(0)
# CHECK-NEXT: s_barrier_join m0
# CHECK-LABEL: .L_join:
# CHECK-NEXT: s_wait_alu depctr_sa_sdst(0)
# CHECK-NEXT: s_barrier_signal m0
# CHECK:      s_mov_b32 m0, s28
# CHECK-NEXT: s_wait_alu depctr_sa_sdst(0)
# CHECK-NEXT: s_barrier_wait m0
# CHECK-NEXT: s_endpgm

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
k:
s_mov_b32 m0, s26
s_barrier_join m0
s_cmp_lt_u32 s28, 4
s_cbranch_scc0 .L_join
s_mov_b32 m0, s27
.L_join:
s_barrier_signal m0
s_mov_b32 m0, s28
s_wait_alu depctr_sa_sdst(0)
s_barrier_wait m0
s_endpgm
