# RUN: %stinkytofu-opt --arch gfx1250 %s --RaiseVgprMsbPass --InsertVgprMsbPass --from-label label_start --to-label label_end --emit-asm
#
# Round-trip through RaiseVgprMsbPass + InsertVgprMsbPass on a kernel that
# uses VGPRs above index 255. Each such operand requires an s_set_vgpr_msb
# state instruction to select the upper index bits for the relevant field
# (src0/src1/src2/dst). Raise dissolves the state into per-operand full
# indices; Insert restores the state instructions and re-encodes the
# operands back to byte form `v[fullIdx - msb*256]`.
#
# The input exercises three state values (1, then 2, then reset to 0) so
# the test fails if Insert miscounts, omits a state change, or emits the
# operands without the byte-form offset.

# CHECK-LABEL: label_start:
# CHECK:       s_set_vgpr_msb 1
# CHECK:       v_mov_b32 v0, v[257-256]
# CHECK:       s_set_vgpr_msb 258
# CHECK:       v_mov_b32 v3, v[516-512]
# CHECK:       s_set_vgpr_msb 512
# CHECK:       v_mov_b32 v0, v1
# CHECK-LABEL: label_end:

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_msb
.p2align 8
.type test_msb,@function
.set vgprHi1, 257
.set vgprHi2, 516
test_msb:
label_start:
s_set_vgpr_msb 1
v_mov_b32 v0, v[vgprHi1-256]
s_set_vgpr_msb 2
v_mov_b32 v3, v[vgprHi2-512]
s_set_vgpr_msb 0
v_mov_b32 v0, v1
label_end:
s_endpgm
