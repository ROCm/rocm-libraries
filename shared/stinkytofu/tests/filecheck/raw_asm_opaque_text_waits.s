# RUN: %stinkytofu-opt --arch gfx1250 %s --StinkyRemoveWaitCntPass --from-label region_begin --to-label region_end 2>&1; echo "exit=$?"; %stinkytofu-opt --arch gfx1250 %s --StinkyRemoveWaitCntPass --allow-opaque-text --from-label region_begin --to-label region_end 2>&1

# future_global_load_b32 has no table entry, so the parser keeps it as text. Without
# --allow-opaque-text the driver refuses to run passes over it. With it,
# RemoveWaitCnt must keep s_wait_loadcnt: the load is invisible to wait
# insertion, which would never put the wait back.

# CHECK: Error: {{.*}}contains an instruction kept as text
# CHECK-NEXT: future_global_load_b32 v4, off, s0 offset:16
# CHECK: exit=1
# CHECK: Warning: StinkyRemoveWaitCntPass: {{.*}}keeping its wait counters
# CHECK: region_begin:
# CHECK-NEXT: global_load_b32 v1, v2, s[0:1] offset:16
# CHECK-NEXT: future_global_load_b32 v4, off, s0 offset:16
# CHECK-NEXT: s_wait_loadcnt 0x0
# CHECK-NEXT: v_add_f32 v3, v1, v1
# CHECK-NEXT: v_add_f32 v6, v4, v4

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
region_begin:
global_load_b32 v1, v2, s[0:1] offset:16
future_global_load_b32 v4, off, s0 offset:16
s_wait_loadcnt 0x0
v_add_f32 v3, v1, v1
v_add_f32 v6, v4, v4
region_end:
s_endpgm
