# RUN: %stinkytofu-opt --arch gfx1250 %s --RaiseVgprMsbPass --from-label label_start --to-label label_end
#
# Hex s_set_vgpr_msb immediates (what LLVM and our generators print) must raise
# exactly like their decimal spelling. 0x55 selects bank 1 for src0/src1/src2/dst;
# 0xc3 selects bank 3 for src0 and dst. The parser keeps hex spelled as text, so
# the pass used to see "no value", keep MSB 0 and still erase the setter.

# CHECK-LABEL: label_start:
# CHECK-NEXT:  v_add_f32 v257, v258, v259
# CHECK-NEXT:  v_add_f32 v257, v258, v259
# CHECK-LABEL: label_x:
# CHECK-NEXT:  v_mov_b32 v775, v776
# CHECK-NOT:   s_set_vgpr_msb

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
label_start:
s_set_vgpr_msb 0x55
v_add_f32 v1, v2, v3
s_set_vgpr_msb 85
v_add_f32 v1, v2, v3
label_x:
s_set_vgpr_msb 0xc3
v_mov_b32 v7, v8
label_end:
s_endpgm
