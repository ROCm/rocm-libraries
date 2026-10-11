# RUN: %stinkytofu-opt --arch gfx1250 %s --LongBranchLoweringPass --CFGBuilderPass --StinkyUnreachableBlockElimPass --from-label region_begin --to-label region_end --emit-asm
#
# rocisa's dispatching SLongBranch: the negative arm falls through from
# s_cbranch_scc1 and the positive arm starts at its label, so a backward walk
# that stops at the first branch recognised neither s_setpc_b64. Both then had
# no CFG successor, and unreachable-block elimination deleted the live target.

# CHECK:      region_begin
# CHECK:      s_cbranch_scc1 label_Pos
# CHECK:      s_setpc_b64 s[62:63]
# CHECK:      label_Pos
# CHECK:      s_setpc_b64 s[62:63]
# CHECK:      label_Target
# CHECK-NEXT: s_mov_b32 s5, 1
# CHECK:      region_end
# CHECK-NEXT: s_endpgm

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

region_begin:
s_getpc_b64 s[62:63]                         // addr of next instr
s_add_i32 s64, label_Target, 4               // target branch offset
s_cmp_ge_i32 s64, 0                          // check positive or negative
s_cbranch_scc1 label_Pos                     // jump when positive
s_abs_i32 s64, s64                           // abs offset
s_sub_u32 s62, s62, s64                      // sub target branch offset
s_subb_u32 s63, s63, 0                       // sub high and carry
s_setpc_b64 s[62:63]                         // branch to label_Target
label_Pos:
s_add_u32 s62, s62, s64                      // add target branch offset
s_addc_u32 s63, s63, 0                       // add high and carry
s_setpc_b64 s[62:63]                         // branch to label_Target
label_Target:
s_mov_b32 s5, 1
region_end:
s_endpgm
