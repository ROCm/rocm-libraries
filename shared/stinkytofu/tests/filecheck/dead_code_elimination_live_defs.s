# RUN: %stinkytofu-opt --arch gfx1250 %s --from-label region_begin --to-label region_end --DeadCodeEliminationPass --emit-asm; echo CHAINED; %stinkytofu-opt --arch gfx1250 %s --from-label region_begin --to-label region_end --CFGBuilderPass --BuildUseDefChainPass --DeadCodeEliminationPass --emit-asm
#
# DCE keeps defs that are still live: a 64-bit def whose high dword is read after a 32-bit
# rewrite of the low one, a DS atomic's return value, an M0 the LDS-DMA load reads implicitly,
# and a VGPR def whose later rewrite runs under a different EXEC. With def-use chains built it
# must not abort. The fully overwritten v10 = 1 is still removed.
#
# CHECK-NOT: v_mov_b32 v10, 1
# CHECK: v_mov_b32 v10, 2
# CHECK: v_lshlrev_b64 v[0:1], 2, v[4:5]
# CHECK: v_mov_b32 v0, 0
# CHECK: ds_add_rtn_u32 v6, v7, v8
# CHECK: s_mov_b32 m0, s2
# CHECK: buffer_load_b32
# CHECK: s_mov_b32 m0, s3
# CHECK: v_mov_b32 v11, 1
# CHECK: s_mov_b32 exec_lo, s4
# CHECK: v_mov_b32 v11, 2
# CHECK: CHAINED
# CHECK-NOT: v_mov_b32 v10, 1
# CHECK: v_lshlrev_b64 v[0:1], 2, v[4:5]
# CHECK: ds_add_rtn_u32 v6, v7, v8
# CHECK: s_mov_b32 m0, s2
# CHECK: v_mov_b32 v11, 1
.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
region_begin:
v_mov_b32 v10, 1
v_mov_b32 v10, 2
v_lshlrev_b64 v[0:1], 2, v[4:5]
v_mov_b32 v0, 0
v_add_f32 v2, v1, v1
ds_add_rtn_u32 v6, v7, v8
v_mov_b32 v6, 1
s_mov_b32 m0, s2
buffer_load_b32 v9, v8, s[4:7], s3 offen lds
s_mov_b32 m0, s3
v_mov_b32 v11, 1
s_mov_b32 exec_lo, s4
v_mov_b32 v11, 2
region_end:
s_endpgm
