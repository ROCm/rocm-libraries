# RUN: %stinkytofu-opt --arch gfx1250 %s --CFGBuilderPass --StinkyUnreachableBlockElimPass --from-label region_begin --to-label region_end --emit-asm
#
# Unreachable-block elimination keeps address-taken blocks. This indirect
# jump is not a recognised long branch (the carry adds s65, not 0), so the CFG
# has no edge to label_Target. Its label is an s_add_i32 operand, though, so
# the s_setpc_b64 can reach it: it and its fall-through successor stay, and
# only the unreferenced label_Dead goes.

# CHECK:      s_setpc_b64 s[62:63]
# CHECK-NOT:  label_Dead
# CHECK:      label_Target:
# CHECK-NEXT: s_mov_b32 s5, 1
# CHECK-NEXT: label_Next:
# CHECK-NEXT: s_mov_b32 s6, 2
# CHECK-NEXT: region_end:

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
region_begin:
s_getpc_b64 s[62:63]
s_add_i32 s64, label_Target, 4
s_mov_b32 s65, 0
s_add_u32 s62, s62, s64
s_addc_u32 s63, s63, s65
s_setpc_b64 s[62:63]
label_Dead:
s_nop 0
label_Target:
s_mov_b32 s5, 1
label_Next:
s_mov_b32 s6, 2
region_end:
s_endpgm
