# RUN: %stinkytofu-opt --arch gfx1250 %s --whole-function --InsertWaitAluPass 2>&1; echo "--- assumeInOrderVaVdstAcrossPipes"; %stinkytofu-opt --arch gfx1250 %s --whole-function --InsertWaitAluPass=assumeInOrderVaVdstAcrossPipes 2>&1
#
# The bf16 pattern: a VALU write, then WMMAs, then a ds_load that overwrites a
# WMMA source (va_vdst(3) for its XDL followers), then a ds_load addressed by
# the VALU result. A wait counted on the XDL pipe proves nothing about the VALU
# on another pipe (per-pipe va_vdst, as asm_verify checks), so the second
# ds_load gets its own wait by default. The unsafe in-order opt-in suppresses
# it; gfx1250_va_vdst_cross_pipe demonstrates that gfx1250 does not provide the
# assumed ordering.

# CHECK:      v_wmma_f32_16x16x32_bf16 v[32:39]
# CHECK-NEXT: s_wait_alu depctr_va_vdst(3)
# CHECK-NEXT: ds_load_b32 v200, v50
# CHECK-NEXT: s_wait_alu depctr_va_vdst(0)
# CHECK-NEXT: ds_load_b32 v60, v10

# CHECK:      --- assumeInOrderVaVdstAcrossPipes
# CHECK:      v_wmma_f32_16x16x32_bf16 v[32:39]
# CHECK-NEXT: s_wait_alu depctr_va_vdst(3)
# CHECK-NEXT: ds_load_b32 v200, v50
# CHECK-NEXT: ds_load_b32 v60, v10

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
k:
v_add_u32 v10, v1, v2
v_wmma_f32_16x16x32_bf16 v[0:7], v[200:207], v[208:215], v[0:7]
v_wmma_f32_16x16x32_bf16 v[16:23], v[216:223], v[208:215], v[16:23]
v_wmma_f32_16x16x32_bf16 v[24:31], v[216:223], v[208:215], v[24:31]
v_wmma_f32_16x16x32_bf16 v[32:39], v[216:223], v[208:215], v[32:39]
ds_load_b32 v200, v50
ds_load_b32 v60, v10
s_endpgm
