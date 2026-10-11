# RUN: %stinkytofu-opt --arch gfx1250 %s --whole-function --RaiseVgprMsbPass --InsertVgprMsbPass --emit-asm 2>/dev/null
#
# s_endpgm is reached with VGPR MSB mode 0: the mode is MODE register state
# (sp3 7.4.27), so a high-VGPR op right before the end sets it back.
# Before: the kernel ended in mode 0x45. 17664 = 0x4500 (mode 0, previous 0x45).

# CHECK:      s_set_vgpr_msb 69{{$}}
# CHECK-NEXT: v_add_f32 v[261-256], v[262-256], v[263-256]
# CHECK-NEXT: s_set_vgpr_msb 17664{{$}}
# CHECK-NEXT: s_endpgm

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
k:
s_set_vgpr_msb 0x45
v_add_f32 v5, v6, v7
s_endpgm
