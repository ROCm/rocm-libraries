# RUN: %stinkytofu-opt --arch gfx1250 %s --from-label region_start --to-label region_end --InsertAsanCheckPass --emit-asm
#
# InsertAsanCheckPass: debug-only ASan shadow-memory bounds check, gfx1250 only.
#
# - Fires on every MUBUF load/store whose SRD is one of the tracked tensor
#   buffers (sgprSrdA/B/C/D), resolved via the `.set` symbol table since
#   operand identity is lost post-conversion (physical index only).
# - Does NOT fire on an untracked SRD (e.g. bias) -- scope reduction, not a bug.
# - The skip label must be anchored immediately before the guarded access, not
#   appended at the end of the block (AsmIRBuilder::createLabel() has no
#   insertBefore param and defaults to end-of-block) -- otherwise a passing
#   check on an earlier access branches clean past every later real access
#   (and past s_endpgm). This is why the label for the SrdA check precedes
#   the SrdA load, not the SrdB store two accesses later.
# - global_store_b64 (the violation report write) is followed by
#   s_wait_storecnt 0 before s_trap, so the write is guaranteed to land in
#   memory before the wave halts.

# CHECK-LABEL: region_start:
# CHECK: v_add_co_u32 v50, vcc_lo, s20, v4
# CHECK-NEXT: v_add_co_ci_u32 v51, vcc_lo, s21, 0, vcc_lo
# CHECK-NEXT: v_lshrrev_b32 v50, 3, v50
# CHECK-NEXT: v_lshl_or_b32 v50, v51, 29, v50
# CHECK-NEXT: v_lshrrev_b32 v51, 3, v51
# CHECK-NEXT: v_or_b32 v50, 2147450880, v50
# CHECK-NEXT: global_load_u8 v52, v[50:51], off
# CHECK-NEXT: s_wait_loadcnt 0
# CHECK-NEXT: v_cmp_ne_u32 vcc_lo, 0, v52
# CHECK-NEXT: s_cbranch_vccz label_asanCheckOk_0
# CHECK-NEXT: s_getpc_b64 s[102:103]
# CHECK-NEXT: v_mov_b32 v54, s100
# CHECK-NEXT: v_mov_b32 v55, s101
# CHECK-NEXT: v_mov_b32 v56, s102
# CHECK-NEXT: v_mov_b32 v57, s103
# CHECK-NEXT: global_store_b64 v[54:55], v[56:57], off
# CHECK-NEXT: s_wait_storecnt 0
# CHECK-NEXT: s_trap 2
# CHECK-NEXT: label_asanCheckOk_0:
# CHECK-NEXT: buffer_load_b128 v[0:3], v4, s[20:23], null offen offset:0

# The bias access is on an untracked SRD -- no check between the two loads.
# CHECK-NOT: v_add_co_u32
# CHECK: buffer_load_b128 v[8:11], v12, s[28:31], null offen offset:0

# Second tracked access (SrdB, a store): gets its own check + its own skip
# label anchored immediately before it, not reused/shared from the first.
# CHECK: v_add_co_u32 v50, vcc_lo, s24, v20
# CHECK: s_cbranch_vccz label_asanCheckOk_1
# CHECK: s_trap 2
# CHECK-NEXT: label_asanCheckOk_1:
# CHECK-NEXT: buffer_store_b128 v[16:19], v20, s[24:27], null offen offset:0
# CHECK-NEXT: s_endpgm

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
region_start:
.set sgprSrdA, 20
.set sgprSrdB, 24
.set sgprSrdBias, 28
.set sgprAsanReportBuf, 100
.set sgprAsanTmp, 102
.set vgprAsanTmp, 50

buffer_load_b128 v[0:3], v4, s[sgprSrdA:sgprSrdA+3], 0 offen
buffer_load_b128 v[8:11], v12, s[sgprSrdBias:sgprSrdBias+3], 0 offen
buffer_store_b128 v[16:19], v20, s[sgprSrdB:sgprSrdB+3], 0 offen
s_endpgm
region_end:
