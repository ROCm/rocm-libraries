# RUN: %stinkytofu-opt --arch gfx1250 %s --vgpr-msb-mode=msb8 --vgpr-msb-unused-slots-dont-care --CFGBuilderPass --InsertVgprMsbPass --from-label label_start --to-label label_end --emit-asm
#
# Under VgprMsbUnusedSlotsDontCare, labels and basic blocks still reset what
# InsertVgprMsbPass knows, and a setter is never rewritten from another block.
#
# - The setter after label_start (65) stays 65: the fall-through block needs
#   src1 and src2 as well, but gets its own setter instead of rewriting that one.
# - label_next gets an s_nop and its own setter: its predecessors leave 65 (the
#   taken branch) and 85 (the fall-through).
# - The last v_add, behind an SALU, uses only slots 85 already selects.

# CHECK-LABEL: label_start:
# CHECK-NEXT: s_nop 0
# CHECK-NEXT: s_set_vgpr_msb 65
# CHECK-NEXT: v_mov_b32 v[257-256], v[258-256]
# CHECK-NEXT: s_cbranch_scc1 label_next
# CHECK-NEXT: s_set_vgpr_msb 85
# CHECK-NEXT: v_fma_f32 v[259-256], v[260-256], v[261-256], v[262-256]
# CHECK-NEXT: label_next:
# CHECK-NEXT: s_nop 0
# CHECK-NEXT: s_set_vgpr_msb 85
# CHECK-NEXT: v_fma_f32 v[263-256], v[264-256], v[265-256], v[266-256]
# CHECK-NEXT: s_add_u32 s0, s0, 1
# CHECK-NEXT: v_add_f32 v[267-256], v[268-256], v[269-256]
# CHECK-NEXT: label_end:

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_msb_labels
.p2align 8
.type test_msb_labels,@function
test_msb_labels:
label_start:
v_mov_b32 v257, v258
s_cbranch_scc1 label_next
v_fma_f32 v259, v260, v261, v262
label_next:
v_fma_f32 v263, v264, v265, v266
s_add_u32 s0, s0, 1
v_add_f32 v267, v268, v269
label_end:
s_endpgm
