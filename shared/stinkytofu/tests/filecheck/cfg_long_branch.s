# RUN: %stinkytofu-opt --arch gfx1250 %s --LongBranchLoweringPass --from-label region_begin --to-label region_end --emit-asm
#
# Long-branch idiom round-trip:
#
# LongBranchLoweringPass recognizes the rocisa long-branch sequence
#
#     s_getpc_b64  s[D:D+1]
#     s_add_i32    sX, LBL, 4
#     s_add_u32    sD,   sD,   sX
#     s_addc_u32   sD+1, sD+1, 0
#     s_setpc_b64  s[D:D+1]
#
# and stamps a LabelData{LBL} modifier on the terminating s_setpc_b64 so
# CFGBuilder can wire a normal direct-branch edge.
#
# The pass only attaches metadata, it does NOT rewrite the instructions.
# So the round-tripped assembly must:
#   1. Preserve every instruction of the idiom (no DCE; no rewriting).
#   2. Preserve the original operand spelling (label name + literal 4).
#   3. Still terminate with s_setpc_b64 s[62:63].
#
# CHECK: region_begin
# CHECK: s_getpc_b64 s[62:63]
# CHECK: s_add_i32 s64, label_PrefetchEnd, 4
# CHECK: s_add_u32 s62, s62, s64
# CHECK: s_addc_u32 s63, s63, 0
# CHECK: s_setpc_b64 s[62:63]
# CHECK: label_PrefetchEnd
# CHECK: region_end

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text

region_begin:
s_getpc_b64 s[62:63]                         // addr of next instr
s_add_i32 s64, label_PrefetchEnd, 4          // target branch offset
s_add_u32 s62, s62, s64                      // add target branch offset
s_addc_u32 s63, s63, 0                       // add high and carry
s_setpc_b64 s[62:63]                         // branch to label_PrefetchEnd
label_PrefetchEnd:
s_nop 0
region_end:
s_endpgm
