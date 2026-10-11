# RUN: %stinkytofu-opt --arch gfx1250 %s --RemoveWaitAluPass --InsertWaitAluPass --from-label region_begin --to-label region_end
#
# RemoveWaitAlu clears only va_vdst / vm_vsrc, which InsertWaitAlu regenerates.
# sa_sdst (and va_sdst, va_ssrc, va_vcc) are live expert-mode counters nothing
# re-inserts: the survivor keeps sa_sdst, and InsertWaitAlu must not fold the
# survivor's hold_cnt into its own wait and erase sa_sdst with it.

# CHECK-LABEL: region_begin:
# CHECK:       v_add_f32 v1, v2, v3
# CHECK-NEXT:  s_wait_alu {{.*}}sa_sdst(0)
# CHECK-NEXT:  s_wait_alu depctr_va_vdst(0)
# CHECK-NEXT:  buffer_store_b32 v1, v2, s[4:7], s3 offen offset:16

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
s_nop 0
region_begin:
v_add_f32 v1, v2, v3
s_wait_alu depctr_va_vdst(0) depctr_sa_sdst(0) depctr_hold_cnt(0)
buffer_store_b32 v1, v2, s[4:7], s3 offen offset:16
region_end:
s_endpgm
