# RUN: %stinkytofu-opt --arch gfx1250 %s --StinkyDAGSchedulerPass --InsertHazardGapNopPass --from-label region_begin --to-label region_end 2>/dev/null; echo "=== before"; %stinkytofu-opt --arch gfx1250 %s --StinkyDAGSchedulerPass --HazardGapAnalysisPass --from-label region_begin --to-label region_end 2>&1 >/dev/null | grep "SaluSgprToMemAddr (" ; echo "=== after"; %stinkytofu-opt --arch gfx1250 %s --StinkyDAGSchedulerPass --InsertHazardGapNopPass --HazardGapAnalysisPass --from-label region_begin --to-label region_end 2>&1 >/dev/null | grep "SaluSgprToMemAddr ("
#
# A SALU-written SGPR read by tensor_load_to_lds needs 8 cycles between them.
# The scheduler only models that as a stall it may pay, so it leaves
# both tensor_loads closer: the second one directly after its s_add_u32, the
# tie-break case makes more likely. InsertHazardGapNopPass pads the
# shortfall with s_nop: 7 states after one VALU cycle, 8 with none.

# CHECK:      s_addc_u32 s33, s33, 0
# CHECK-NEXT: v_add_f32 v4, v2, v3
# CHECK-NEXT: s_nop 6
# CHECK-NEXT: tensor_load_to_lds s[32:35], s[40:47]
# CHECK:      s_add_u32 s48, s48, 0x1000
# CHECK-NEXT: s_nop 7
# CHECK-NEXT: tensor_load_to_lds s[48:51], s[56:63]
# CHECK:      === before
# CHECK-NEXT: SaluSgprToMemAddr (>=8 cyc):  2 pair(s) checked,  2 VIOLATION(s)
# CHECK:      === after
# CHECK-NEXT: SaluSgprToMemAddr (>=8 cyc):  2 pair(s) checked,  0 VIOLATION(s)

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
region_begin:
s_mov_b32 s5, 1
s_add_u32 s32, s32, 0x1000
s_addc_u32 s33, s33, 0
v_add_f32 v1, v2, v3
v_add_f32 v4, v2, v3
tensor_load_to_lds s[32:35], s[40:47]
ds_load_b128 v[8:11], v20
global_load_b128 v[12:15], v21, s[50:51]
s_add_u32 s48, s48, 0x1000
tensor_load_to_lds s[48:51], s[56:63]
region_end:
s_endpgm
