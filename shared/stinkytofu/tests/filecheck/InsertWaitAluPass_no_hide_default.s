# RUN: %stinkytofu-opt --arch gfx1250 %s --whole-function --InsertWaitAluPass 2>&1; echo "--- waitHide"; %stinkytofu-opt --arch gfx1250 %s --whole-function --InsertWaitAluPass=waitHide 2>&1
#
# in P's K loop a VALU result used as a ds_load address got no va_vdst
# wait because 12 matrix ops had issued since (HWModel::WaitHide, a timing
# assumption), and a ds_load's address VGPR was overwritten after 11 later DS
# reads with no vm_vsrc wait. The ISA makes counter waits mandatory, so the
# hides are opt-in (=waitHide).

# CHECK:      v_add_u32 v200, s4, v201
# CHECK:      v_wmma_f32_16x16x128_fp8_fp8 v[96:103]
# CHECK-NEXT: s_wait_alu depctr_va_vdst({{[0-9]+}})
# CHECK-NEXT: ds_load_b128 v[160:163], v200 offset:0
# CHECK:      ds_load_b128 v[212:215], v203 offset:176
# CHECK-NEXT: s_wait_alu depctr_vm_vsrc({{[0-9]+}})
# CHECK-NEXT: v_add_u32 v202, s4, v201

# CHECK:      --- waitHide
# CHECK:      v_wmma_f32_16x16x128_fp8_fp8 v[96:103]
# CHECK-NEXT: ds_load_b128 v[160:163], v200 offset:0
# CHECK:      ds_load_b128 v[212:215], v203 offset:176
# CHECK-NEXT: v_add_u32 v202, s4, v201

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
k:
v_add_u32 v200, s4, v201
v_wmma_f32_16x16x128_fp8_fp8 v[0:7], v[128:143], v[144:159], v[0:7]
v_wmma_f32_16x16x128_fp8_fp8 v[8:15], v[128:143], v[144:159], v[8:15]
v_wmma_f32_16x16x128_fp8_fp8 v[16:23], v[128:143], v[144:159], v[16:23]
v_wmma_f32_16x16x128_fp8_fp8 v[24:31], v[128:143], v[144:159], v[24:31]
v_wmma_f32_16x16x128_fp8_fp8 v[32:39], v[128:143], v[144:159], v[32:39]
v_wmma_f32_16x16x128_fp8_fp8 v[40:47], v[128:143], v[144:159], v[40:47]
v_wmma_f32_16x16x128_fp8_fp8 v[48:55], v[128:143], v[144:159], v[48:55]
v_wmma_f32_16x16x128_fp8_fp8 v[56:63], v[128:143], v[144:159], v[56:63]
v_wmma_f32_16x16x128_fp8_fp8 v[64:71], v[128:143], v[144:159], v[64:71]
v_wmma_f32_16x16x128_fp8_fp8 v[72:79], v[128:143], v[144:159], v[72:79]
v_wmma_f32_16x16x128_fp8_fp8 v[80:87], v[128:143], v[144:159], v[80:87]
v_wmma_f32_16x16x128_fp8_fp8 v[88:95], v[128:143], v[144:159], v[88:95]
v_wmma_f32_16x16x128_fp8_fp8 v[96:103], v[128:143], v[144:159], v[96:103]
ds_load_b128 v[160:163], v200 offset:0
ds_load_b128 v[164:167], v202 offset:16
ds_load_b128 v[168:171], v203 offset:16
ds_load_b128 v[172:175], v203 offset:32
ds_load_b128 v[176:179], v203 offset:48
ds_load_b128 v[180:183], v203 offset:64
ds_load_b128 v[184:187], v203 offset:80
ds_load_b128 v[188:191], v203 offset:96
ds_load_b128 v[192:195], v203 offset:112
ds_load_b128 v[196:199], v203 offset:128
ds_load_b128 v[204:207], v203 offset:144
ds_load_b128 v[208:211], v203 offset:160
ds_load_b128 v[212:215], v203 offset:176
v_add_u32 v202, s4, v201
s_endpgm
