# RUN: %stinkytofu-opt --arch gfx1250 %s --from-label region_begin --to-label region_end 2>&1; %stinkytofu-opt --arch gfx1250-strict %s --from-label region_begin --to-label region_end 2>&1 >/dev/null | grep -iE "warning|error" | sed "s/^/diag: /"; echo END
#
# SCC readers and writers that were missing from both gfx1250 tables. As
# opaque text they were invisible to the SCC rules of the scheduler and of
# InsertClusterBarrierPass (our bigtile --cu-map emits s_bitcmp0_b32). Every
# line must parse as an instruction and round-trip as written.

# CHECK-NOT:  Warning
# CHECK:      region_begin:
# CHECK-NEXT: s_bitcmp0_b32 s1, 3
# CHECK-NEXT: s_bitcmp0_b64 s[2:3], 5
# CHECK-NEXT: s_bitcmp1_b64 s[2:3], s4
# CHECK-NEXT: s_cmovk_i32 s0, 0x1234
# CHECK-NEXT: s_barrier_signal_isfirst -1
# CHECK-NEXT: s_barrier_leave
# CHECK-NEXT: s_get_barrier_state s5, m0
# CHECK-NEXT: region_end:
# CHECK-NOT:  diag:
# CHECK:      END

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
region_begin:
s_bitcmp0_b32 s1, 3
s_bitcmp0_b64 s[2:3], 5
s_bitcmp1_b64 s[2:3], s4
s_cmovk_i32 s0, 0x1234
s_barrier_signal_isfirst -1
s_barrier_leave
s_get_barrier_state s5, m0
region_end:
s_endpgm
