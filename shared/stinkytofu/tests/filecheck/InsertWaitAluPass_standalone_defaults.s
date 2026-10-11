# RUN: %stinkytofu-opt --arch gfx1250 %s --whole-function --InsertWaitAluPass 2>&1; echo "--- noTrackValuVsrc"; %stinkytofu-opt --arch gfx1250 %s --whole-function --InsertWaitAluPass=noTrackValuVsrc 2>&1
#
# the standalone --InsertWaitAluPass ran with every refinement off and no
# VALU-source tracking, so a ds_load overwriting the B matrix a WMMA had just
# read got no wait (the 16 wmma-war findings on P in oracle mode). It now runs
# with the gfx1250 pipeline's options and tracks the source WAR; VA_VDST is
# the ISA's proxy for a VALU having read its sources.

# CHECK:      v_wmma_f32_16x16x128_fp8_fp8 v[0:7], v[128:143], v[144:159], v[0:7]
# CHECK-NEXT: s_wait_alu depctr_va_vdst({{[0-9]+}})
# CHECK-NEXT: ds_load_b128 v[144:147], v200 offset:0

# CHECK:      --- noTrackValuVsrc
# CHECK:      v_wmma_f32_16x16x128_fp8_fp8 v[0:7], v[128:143], v[144:159], v[0:7]
# CHECK-NEXT: ds_load_b128 v[144:147], v200 offset:0

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
k:
v_wmma_f32_16x16x128_fp8_fp8 v[0:7], v[128:143], v[144:159], v[0:7]
ds_load_b128 v[144:147], v200 offset:0
s_endpgm
