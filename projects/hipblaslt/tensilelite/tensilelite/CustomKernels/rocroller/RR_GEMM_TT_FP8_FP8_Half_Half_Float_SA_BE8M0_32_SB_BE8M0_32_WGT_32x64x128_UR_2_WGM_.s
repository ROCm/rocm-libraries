// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_,@function
RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__exec_begin:


s_load_dwordx2 s[20:21], s[0:1], 56 // Load scalar value
s_load_dwordx2 s[22:23], s[0:1], 64 // Load scalar value
s_load_dwordx2 s[24:25], s[0:1], 72 // Load scalar value
s_load_dwordx2 s[26:27], s[0:1], 80 // Load scalar value
s_load_dwordx2 s[28:29], s[0:1], 88 // Load scalar value
s_load_dwordx2 s[30:31], s[0:1], 96 // Load scalar value
s_load_dwordx2 s[32:33], s[0:1], 104 // Load scalar value
s_load_dwordx2 s[34:35], s[0:1], 112 // Load scalar value
s_load_dwordx2 s[36:37], s[0:1], 120 // Load scalar value
s_load_dwordx2 s[38:39], s[0:1], 128 // Load scalar value
s_load_dwordx2 s[40:41], s[0:1], 136 // Load scalar value
s_load_dwordx2 s[42:43], s[0:1], 144 // Load scalar value
s_load_dwordx2 s[44:45], s[0:1], 152 // Load scalar value
s_load_dwordx2 s[46:47], s[0:1], 160 // Load scalar value
s_load_dwordx2 s[48:49], s[0:1], 168 // Load scalar value
s_load_dword s50, s[0:1], 176 // Load scalar value

v_bfe_u32 v1, v0, 0, 10
v_bfe_u32 v2, v0, 10, 10


s_mul_hi_i32 s0, s16, s11 // (op 1196) 
s_mul_hi_i32 s1, s16, s11 // (op 1383) 
v_lshrrev_b32 v0, 7, v1 // (op 1252) 
v_and_b32 v3, 15, v1 // (op 1252) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 1252) 


v_and_b32 v0, 48, v1 // (op 1252) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 1252) 


v_lshrrev_b32 v0, 7, v1 // (op 1424) 
v_and_b32 v4, 15, v1 // (op 1424) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 1424) 


v_and_b32 v0, 48, v1 // (op 1424) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 1424) 


v_lshrrev_b32 v0, 7, v1 // (op 1196) 
s_add_i32 s18, s0, s16 // (op 1196) 

v_and_b32 v5, 63, v1 // (op 1196) 
s_ashr_i32 s0, s18, 31 // (op 1196) 
v_lshrrev_b32 v6, 4, v5 // (op 1196) 
s_cmp_eq_i32 s11, 0 // (op 1196) 
s_mov_b32 s19, scc // (op 1196) 
v_lshlrev_b32 v7, 2, v6 // (op 1196) 

s_cmp_eq_i32 s19, 0 // (op 1196) 
s_cselect_b32 s51, 0, -1 // (op 1196) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 1196) 


s_and_b32 s19, s12, 31 // (op 1196) 
v_lshrrev_b32 v0, 2, v5 // (op 1196) 

v_and_b32 v5, 3, v0 // (op 1196) 

v_and_b32 v0, 48, v1 // (op 1309) 
v_and_b32 v7, 63, v1 // (op 1309) 
v_sub_u32 v8, v7, v0 // (op 1309) 

v_lshrrev_b32 v7, 1, v8 // (op 1309) 
v_add_lshl_u32 v9, v0, v7, 6 // (op 1309) 


v_and_b32 v0, 1, v8 // (op 1309) 

v_lshlrev_b32 v7, 3, v0 // (op 1309) 

v_lshrrev_b32 v0, 6, v1 // (op 1309) 
v_and_b32 v8, 1, v0 // (op 1309) 

v_lshl_add_u32 v0, v8, 4, v7 // (op 1309) 


v_add_u32 v7, v9, v0 // (op 1309) 


v_lshrrev_b32 v0, 6, v1 // (op 1367) 
v_and_b32 v8, 1, v0 // (op 1367) 

v_and_b32 v0, 15, v1 // (op 1367) 
v_lshl_add_u32 v9, v8, 4, v0 // (op 1367) 


v_and_b32 v0, 63, v1 // (op 1367) 
v_lshrrev_b32 v8, 4, v0 // (op 1367) 

v_lshl_add_u32 v0, v8, 6, v9 // (op 1367) 


v_lshrrev_b32 v8, 7, v1 // (op 1383) 
v_and_b32 v9, 63, v1 // (op 1383) 
v_and_b32 v10, 48, v1 // (op 1468) 
v_and_b32 v11, 63, v1 // (op 1468) 
v_sub_u32 v12, v11, v10 // (op 1468) 

v_lshrrev_b32 v11, 1, v12 // (op 1468) 
v_add_lshl_u32 v13, v10, v11, 6 // (op 1468) 


v_and_b32 v10, 1, v12 // (op 1468) 

v_lshlrev_b32 v11, 3, v10 // (op 1468) 

v_lshrrev_b32 v10, 6, v1 // (op 1468) 
v_and_b32 v12, 1, v10 // (op 1468) 

v_lshl_add_u32 v10, v12, 4, v11 // (op 1468) 


v_add_u32 v11, v13, v10 // (op 1468) 


v_lshrrev_b32 v10, 6, v1 // (op 1497) 
v_and_b32 v12, 1, v10 // (op 1497) 

v_and_b32 v10, 15, v1 // (op 1497) 
v_lshl_add_u32 v13, v12, 4, v10 // (op 1497) 


v_and_b32 v10, 63, v1 // (op 1497) 
v_lshrrev_b32 v12, 4, v10 // (op 1497) 

v_lshl_add_u32 v10, v12, 6, v13 // (op 1497) 


s_lshl_b32 s52, 1, s19 // (op 1196) 
v_and_b32 v12, 3, v1 // (op 1196) 
s_add_i32 s53, s52, s51 // (op 1196) 


s_and_b32 s51, s0, s53 // (op 1196) 


s_add_i32 s0, s1, s16 // (op 1383) 

s_ashr_i32 s1, s0, 31 // (op 1383) 
v_lshl_add_u32 v13, v5, 2, v12 // (op 1196) 


s_add_i32 s52, s18, s51 // (op 1196) 


v_lshrrev_b32 v5, 6, v1 // (op 1196) 
s_ashr_i32 s18, s52, s19 // (op 1196) 


v_and_b32 v12, 1, v5 // (op 1196) 

s_lshl_b32 s19, s12, 24 // (op 1196) 
v_lshl_add_u32 v5, v12, 4, v13 // (op 1196) 


s_ashr_i32 s51, s19, 31 // (op 1196) 

s_xor_b32 s19, s18, s51 // (op 1196) 

s_sub_i32 s18, s19, s51 // (op 1196) 


s_mul_hi_i32 s19, s18, s13 // (op 1196) 
s_add_i32 s51, s19, s18 // (op 1196) 

s_ashr_i32 s19, s51, 31 // (op 1196) 
s_cmp_eq_i32 s13, 0 // (op 1196) 
s_mov_b32 s52, scc // (op 1196) 
s_cmp_eq_i32 s52, 0 // (op 1196) 
s_cselect_b32 s53, 0, -1 // (op 1196) 

s_and_b32 s52, s14, 31 // (op 1196) 
v_lshrrev_b32 v12, 4, v9 // (op 1383) 
s_cmp_eq_i32 s11, 0 // (op 1383) 
s_mov_b32 s54, scc // (op 1383) 
v_lshlrev_b32 v13, 2, v12 // (op 1383) 

s_cmp_eq_i32 s54, 0 // (op 1383) 
s_cselect_b32 s55, 0, -1 // (op 1383) 

s_lshl_b32 s54, 1, s52 // (op 1196) 
v_lshl_add_u32 v12, v8, 4, v13 // (op 1383) 


v_lshrrev_b32 v8, 2, v9 // (op 1383) 

s_add_i32 s56, s54, s53 // (op 1196) 


s_and_b32 s53, s19, s56 // (op 1196) 


s_add_i32 s19, s51, s53 // (op 1196) 


s_ashr_i32 s51, s19, s52 // (op 1196) 


s_lshl_b32 s19, s14, 24 // (op 1196) 
s_ashr_i32 s52, s19, 31 // (op 1196) 

s_xor_b32 s19, s51, s52 // (op 1196) 

s_sub_i32 s51, s19, s52 // (op 1196) 


s_and_b32 s19, s12, 31 // (op 1383) 
s_lshl_b32 s52, 1, s19 // (op 1383) 
v_and_b32 v9, 3, v8 // (op 1383) 

s_add_i32 s53, s52, s55 // (op 1383) 


v_and_b32 v8, 3, v1 // (op 1383) 
s_and_b32 s52, s1, s53 // (op 1383) 


v_lshl_add_u32 v13, v9, 2, v8 // (op 1383) 


s_add_i32 s1, s0, s52 // (op 1383) 


v_lshrrev_b32 v8, 6, v1 // (op 1383) 
s_ashr_i32 s0, s1, s19 // (op 1383) 


v_and_b32 v9, 1, v8 // (op 1383) 

s_lshl_b32 s1, s12, 24 // (op 1383) 
v_lshl_add_u32 v8, v9, 4, v13 // (op 1383) 


s_ashr_i32 s19, s1, 31 // (op 1383) 

s_xor_b32 s1, s0, s19 // (op 1383) 

s_sub_i32 s0, s1, s19 // (op 1383) 


s_mul_hi_i32 s1, s0, s13 // (op 1383) 
s_add_i32 s19, s1, s0 // (op 1383) 

s_ashr_i32 s1, s19, 31 // (op 1383) 
s_cmp_eq_i32 s13, 0 // (op 1383) 
s_mov_b32 s52, scc // (op 1383) 
s_cmp_eq_i32 s52, 0 // (op 1383) 
s_cselect_b32 s53, 0, -1 // (op 1383) 

s_and_b32 s52, s14, 31 // (op 1383) 
s_lshl_b32 s54, 1, s52 // (op 1383) 
s_add_i32 s55, s54, s53 // (op 1383) 


s_and_b32 s53, s1, s55 // (op 1383) 


s_add_i32 s1, s19, s53 // (op 1383) 


s_ashr_i32 s19, s1, s52 // (op 1383) 


s_lshl_b32 s1, s14, 24 // (op 1383) 
s_ashr_i32 s52, s1, 31 // (op 1383) 

s_xor_b32 s1, s19, s52 // (op 1383) 

s_sub_i32 s19, s1, s52 // (op 1383) 



s_waitcnt lgkmcnt(0)
s_mul_i32 s1, s18, s44 // (op 1196) 
s_sub_u32 s52, s16, s1 // (op 1196) 

s_mul_i32 s1, s51, s44 // (op 1196) 
s_add_u32 s1, s1, s52 // (op 1196) 

s_mul_hi_i32 s52, s16, s10 // (op 1196) 
s_add_i32 s53, s52, s16 // (op 1196) 

s_ashr_i32 s52, s53, 31 // (op 1196) 
s_cmp_eq_i32 s10, 0 // (op 1196) 
s_mov_b32 s54, scc // (op 1196) 
s_cmp_eq_i32 s54, 0 // (op 1196) 
s_cselect_b32 s55, 0, -1 // (op 1196) 

s_and_b32 s54, s47, 31 // (op 1196) 
s_lshl_b32 s56, 1, s54 // (op 1196) 
s_add_i32 s57, s56, s55 // (op 1196) 


s_and_b32 s55, s52, s57 // (op 1196) 


s_add_i32 s52, s53, s55 // (op 1196) 


s_ashr_i32 s53, s52, s54 // (op 1196) 


s_lshl_b32 s52, s47, 24 // (op 1196) 
s_ashr_i32 s54, s52, 31 // (op 1196) 

s_xor_b32 s52, s53, s54 // (op 1196) 

s_sub_i32 s53, s52, s54 // (op 1196) 


s_mul_i32 s52, s0, s44 // (op 1383) 
s_sub_u32 s54, s16, s52 // (op 1383) 

s_mul_i32 s52, s19, s44 // (op 1383) 
s_add_u32 s52, s52, s54 // (op 1383) 

s_mul_hi_i32 s54, s16, s10 // (op 1383) 
s_add_i32 s55, s54, s16 // (op 1383) 

s_ashr_i32 s54, s55, 31 // (op 1383) 
s_cmp_eq_i32 s10, 0 // (op 1383) 
s_mov_b32 s56, scc // (op 1383) 
s_cmp_eq_i32 s56, 0 // (op 1383) 
s_cselect_b32 s57, 0, -1 // (op 1383) 

s_and_b32 s56, s47, 31 // (op 1383) 
s_lshl_b32 s58, 1, s56 // (op 1383) 
s_add_i32 s59, s58, s57 // (op 1383) 


s_and_b32 s57, s54, s59 // (op 1383) 


s_add_i32 s54, s55, s57 // (op 1383) 


s_ashr_i32 s55, s54, s56 // (op 1383) 


s_lshl_b32 s54, s47, 24 // (op 1383) 
s_ashr_i32 s56, s54, 31 // (op 1383) 

s_xor_b32 s54, s55, s56 // (op 1383) 

s_sub_i32 s55, s54, s56 // (op 1383) 


s_add_u32 s56, s6, 64 // (op 1196) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 1196) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 1196) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 1196) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 1196) 

s_mul_i32 s54, s44, s56 // (op 1196) 
s_mul_i32 s58, s48, s54 // (op 1196) 

s_mul_i32 s54, s53, s58 // (op 1196) 

s_sub_u32 s58, s16, s54 // (op 1196) 

s_mov_b32 s54, s58 // (op 1196) 
s_mul_hi_i32 s59, s54, s15 // (op 1196) 
s_add_i32 s60, s59, s54 // (op 1196) 


s_ashr_i32 s54, s60, 31 // (op 1196) 
s_cmp_eq_i32 s15, 0 // (op 1196) 
s_mov_b32 s59, scc // (op 1196) 
s_cmp_eq_i32 s59, 0 // (op 1196) 
s_cselect_b32 s61, 0, -1 // (op 1196) 

s_and_b32 s59, s46, 31 // (op 1196) 
s_lshl_b32 s62, 1, s59 // (op 1196) 
s_add_i32 s63, s62, s61 // (op 1196) 


s_and_b32 s61, s54, s63 // (op 1196) 


s_add_i32 s54, s60, s61 // (op 1196) 


s_ashr_i32 s60, s54, s59 // (op 1196) 


s_lshl_b32 s54, s46, 24 // (op 1196) 
s_ashr_i32 s59, s54, 31 // (op 1196) 

s_xor_b32 s54, s60, s59 // (op 1196) 

s_sub_i32 s60, s54, s59 // (op 1196) 


s_mul_i32 s54, s45, s44 // (op 1196) 
s_add_u32 s62, s8, 32 // (op 1196) Start of Int64 add, locking SCC
s_addc_u32 s63, s9, 0 // (op 1196) End of Int64 add, unlocking SCC
s_sub_u32 s64, s62, 1 // (op 1196) least significant half; sets scc
s_subb_u32 s65, s63, 0 // (op 1196) most significant half; uses scc

s_ashr_i64 s[62:63], s[64:65], 5 // (op 1196) 

s_sub_i32 s59, s62, s54 // (op 1196) 

s_mul_i32 s61, s60, s59 // (op 1196) 

s_sub_u32 s59, s58, s61 // (op 1196) 


s_add_u32 s58, s59, s54 // (op 1196) 


s_cmp_eq_i32 s53, 0 // (op 1196) 
s_mov_b32 s54, scc // (op 1196) 

s_cmp_eq_i32 s54, 0 // (op 1196) 
s_cselect_b32 s53, s58, s1 // (op 1196) 


s_mul_i32 s1, s51, s56 // (op 1196) 


v_lshl_add_u32 v9, s53, 5, v6 // (op 1196) 


s_sub_i32 s51, s18, s1 // (op 1196) 


s_cmp_eq_i32 s54, 0 // (op 1196) 
s_cselect_b32 s1, s60, s51 // (op 1196) 



v_lshl_add_u32 v6, s1, 6, v5 // (op 1196) 


v_mul_lo_u32 v5, v6, s2 // (op 1196) 

v_add_lshl_u32 v6, v9, v5, 1 // (op 1196) 


s_add_u32 s56, s6, 64 // (op 1383) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 1383) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 1383) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 1383) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 1383) 

s_mul_i32 s1, s44, s56 // (op 1383) 
s_mul_i32 s18, s48, s1 // (op 1383) 

s_mul_i32 s1, s55, s18 // (op 1383) 

s_sub_u32 s18, s16, s1 // (op 1383) 

s_mov_b32 s1, s18 // (op 1383) 
s_mul_hi_i32 s51, s1, s15 // (op 1383) 
s_add_i32 s53, s51, s1 // (op 1383) 


s_ashr_i32 s1, s53, 31 // (op 1383) 
s_cmp_eq_i32 s15, 0 // (op 1383) 
s_mov_b32 s51, scc // (op 1383) 
s_cmp_eq_i32 s51, 0 // (op 1383) 
s_cselect_b32 s54, 0, -1 // (op 1383) 

s_and_b32 s51, s46, 31 // (op 1383) 
s_lshl_b32 s58, 1, s51 // (op 1383) 
s_add_i32 s59, s58, s54 // (op 1383) 


s_and_b32 s54, s1, s59 // (op 1383) 


s_add_i32 s1, s53, s54 // (op 1383) 


s_ashr_i32 s53, s1, s51 // (op 1383) 


s_lshl_b32 s1, s46, 24 // (op 1383) 
s_ashr_i32 s51, s1, 31 // (op 1383) 

s_xor_b32 s1, s53, s51 // (op 1383) 

s_sub_i32 s53, s1, s51 // (op 1383) 


s_mul_i32 s1, s45, s44 // (op 1383) 
s_add_u32 s58, s8, 32 // (op 1383) Start of Int64 add, locking SCC
s_addc_u32 s59, s9, 0 // (op 1383) End of Int64 add, unlocking SCC
s_sub_u32 s60, s58, 1 // (op 1383) least significant half; sets scc
s_subb_u32 s61, s59, 0 // (op 1383) most significant half; uses scc

s_ashr_i64 s[58:59], s[60:61], 5 // (op 1383) 

s_sub_i32 s51, s58, s1 // (op 1383) 

s_mul_i32 s54, s53, s51 // (op 1383) 

s_sub_u32 s51, s18, s54 // (op 1383) 


s_add_u32 s18, s51, s1 // (op 1383) 


s_cmp_eq_i32 s55, 0 // (op 1383) 
s_mov_b32 s1, scc // (op 1383) 

s_cmp_eq_i32 s1, 0 // (op 1383) 
s_cselect_b32 s51, s18, s52 // (op 1383) 


s_mul_i32 s18, s19, s56 // (op 1383) 


v_lshl_add_u32 v5, s51, 5, v12 // (op 1383) 


s_sub_i32 s19, s0, s18 // (op 1383) 


s_cmp_eq_i32 s1, 0 // (op 1383) 
s_cselect_b32 s0, s53, s19 // (op 1383) 



v_lshl_add_u32 v9, s0, 6, v8 // (op 1383) 


v_mul_lo_u32 v8, v9, s4 // (op 1383) 

v_add_lshl_u32 v9, v5, v8, 1 // (op 1383) 


s_mul_i32 s1, s2, s7 // (op 1198) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 1198) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 1198) least significant
s_mul_hi_u32 s19, s2, s6 // (op 1198) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1198) most significant: sum
s_add_u32 s1, s1, s19 // (op 1198) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 1198) 

s_mul_i32 s1, s4, s7 // (op 1385) most significant: low of low * high
s_mul_i32 s51, s5, s6 // (op 1385) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 1385) least significant
s_mul_hi_u32 s52, s4, s6 // (op 1385) most significant: high of low * low
s_add_u32 s1, s1, s51 // (op 1385) most significant: sum
s_add_u32 s1, s1, s52 // (op 1385) most significant: sum


s_lshl_b64 s[52:53], s[0:1], 1 // (op 1385) 

s_mov_b32 s56, s20 // (op 1198) reg expression
s_mov_b32 s57, s21 // (op 1198) reg expression
s_mov_b32 s58, s18 // (op 1198) reg expression
s_mov_b32 s59, 131072 // (op 1198) call()

s_mov_b32 s60, s22 // (op 1385) reg expression

s_mov_b32 s61, s23 // (op 1385) reg expression
s_mov_b32 s62, s52 // (op 1385) reg expression
s_mov_b32 s63, 131072 // (op 1385) call()


v_mov_b32 v5, s49 // (op 2) Move value
v_mov_b32 v8, s50 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
v_accvgpr_write a4, 0.00000 // (op 310) call()
v_accvgpr_write a5, 0.00000 // (op 310) call()
v_accvgpr_write a6, 0.00000 // (op 310) call()
v_accvgpr_write a7, 0.00000 // (op 310) call()
s_mul_hi_i32 s0, s16, s11 // (op 963) 
v_lshrrev_b32 v12, 3, v1 // (op 963) 
s_add_i32 s1, s0, s16 // (op 963) 

v_and_b32 v13, -8, v1 // (op 963) 
s_ashr_i32 s0, s1, 31 // (op 963) 
v_sub_u32 v14, v1, v13 // (op 963) 

s_cmp_eq_i32 s11, 0 // (op 963) 
s_mov_b32 s18, scc // (op 963) 
v_lshlrev_b32 v13, 4, v14 // (op 963) 

s_cmp_eq_i32 s18, 0 // (op 963) 
s_cselect_b32 s19, 0, -1 // (op 963) 

v_mov_b32 v15, 0 // (op 963) convert
v_mov_b32 v14, v13 // (op 963) convert

s_and_b32 s18, s12, 31 // (op 963) 
s_lshl_b32 s20, 1, s18 // (op 963) 
s_add_i32 s21, s20, s19 // (op 963) 


s_and_b32 s19, s0, s21 // (op 963) 


s_add_i32 s0, s1, s19 // (op 963) 


s_ashr_i32 s1, s0, s18 // (op 963) 


s_lshl_b32 s0, s12, 24 // (op 963) 
s_ashr_i32 s18, s0, 31 // (op 963) 

s_xor_b32 s0, s1, s18 // (op 963) 

s_sub_i32 s1, s0, s18 // (op 963) 


s_mul_hi_i32 s0, s1, s13 // (op 963) 
s_add_i32 s18, s0, s1 // (op 963) 

s_ashr_i32 s0, s18, 31 // (op 963) 
s_cmp_eq_i32 s13, 0 // (op 963) 
s_mov_b32 s19, scc // (op 963) 
s_cmp_eq_i32 s19, 0 // (op 963) 
s_cselect_b32 s20, 0, -1 // (op 963) 

s_and_b32 s19, s14, 31 // (op 963) 
s_lshl_b32 s21, 1, s19 // (op 963) 
s_add_i32 s22, s21, s20 // (op 963) 


s_and_b32 s23, s0, s22 // (op 963) 


s_add_i32 s0, s18, s23 // (op 963) 


s_ashr_i32 s18, s0, s19 // (op 963) 


s_lshl_b32 s0, s14, 24 // (op 963) 
s_ashr_i32 s19, s0, 31 // (op 963) 

s_xor_b32 s0, s18, s19 // (op 963) 

s_sub_i32 s18, s0, s19 // (op 963) 


s_mul_i32 s0, s1, s44 // (op 963) 

s_sub_u32 s1, s16, s0 // (op 963) 

s_mul_i32 s0, s18, s44 // (op 963) 
s_add_u32 s0, s0, s1 // (op 963) 


s_mul_hi_i32 s1, s16, s10 // (op 963) 
s_add_i32 s18, s1, s16 // (op 963) 

s_ashr_i32 s1, s18, 31 // (op 963) 
s_cmp_eq_i32 s10, 0 // (op 963) 
s_mov_b32 s19, scc // (op 963) 
s_cmp_eq_i32 s19, 0 // (op 963) 
s_cselect_b32 s20, 0, -1 // (op 963) 

s_and_b32 s19, s47, 31 // (op 963) 
s_lshl_b32 s21, 1, s19 // (op 963) 
s_add_i32 s22, s21, s20 // (op 963) 


s_and_b32 s23, s1, s22 // (op 963) 


s_add_i32 s1, s18, s23 // (op 963) 


s_ashr_i32 s18, s1, s19 // (op 963) 


s_lshl_b32 s1, s47, 24 // (op 963) 
s_ashr_i32 s19, s1, 31 // (op 963) 

s_xor_b32 s1, s18, s19 // (op 963) 

s_sub_i32 s18, s1, s19 // (op 963) 


s_add_u32 s22, s6, 64 // (op 963) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 963) End of Int64 add, unlocking SCC
s_sub_u32 s20, s22, 1 // (op 963) least significant half; sets scc
s_subb_u32 s21, s23, 0 // (op 963) most significant half; uses scc

s_ashr_i64 s[22:23], s[20:21], 6 // (op 963) 

s_mul_i32 s1, s44, s22 // (op 963) 

s_mul_i32 s19, s48, s1 // (op 963) 

s_mul_i32 s1, s18, s19 // (op 963) 

s_sub_u32 s19, s16, s1 // (op 963) 

s_mov_b32 s1, s19 // (op 963) 
s_mul_hi_i32 s20, s1, s15 // (op 963) 
s_add_i32 s21, s20, s1 // (op 963) 


s_ashr_i32 s1, s21, 31 // (op 963) 
s_cmp_eq_i32 s15, 0 // (op 963) 
s_mov_b32 s20, scc // (op 963) 
s_cmp_eq_i32 s20, 0 // (op 963) 
s_cselect_b32 s22, 0, -1 // (op 963) 

s_and_b32 s20, s46, 31 // (op 963) 
s_lshl_b32 s23, 1, s20 // (op 963) 
s_add_i32 s49, s23, s22 // (op 963) 


s_and_b32 s22, s1, s49 // (op 963) 


s_add_i32 s1, s21, s22 // (op 963) 


s_ashr_i32 s21, s1, s20 // (op 963) 


s_lshl_b32 s1, s46, 24 // (op 963) 
s_ashr_i32 s20, s1, 31 // (op 963) 

s_xor_b32 s1, s21, s20 // (op 963) 

s_sub_i32 s21, s1, s20 // (op 963) 


s_mul_i32 s1, s45, s44 // (op 963) 
s_add_u32 s22, s8, 32 // (op 963) Start of Int64 add, locking SCC
s_addc_u32 s23, s9, 0 // (op 963) End of Int64 add, unlocking SCC
s_sub_u32 s54, s22, 1 // (op 963) least significant half; sets scc
s_subb_u32 s55, s23, 0 // (op 963) most significant half; uses scc

s_ashr_i64 s[22:23], s[54:55], 5 // (op 963) 

s_sub_i32 s20, s22, s1 // (op 963) 

s_mul_i32 s22, s21, s20 // (op 963) 


s_sub_u32 s23, s19, s22 // (op 963) 


s_add_u32 s19, s23, s1 // (op 963) 


s_cmp_eq_i32 s18, 0 // (op 963) 
s_mov_b32 s1, scc // (op 963) 

s_cmp_eq_i32 s1, 0 // (op 963) 
s_cselect_b32 s18, s19, s0 // (op 963) 



v_lshl_add_u32 v13, s18, 5, v12 // (op 963) 


v_mul_lo_u32 v17, v13, s25 // (op 963) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 963) most significant: high of low * low
v_mul_lo_u32 v16, v13, s24 // (op 963) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 963) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 963) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 963) most significant half



s_mul_i32 s1, s24, s9 // (op 965) most significant: low of low * high
s_mul_i32 s18, s25, s8 // (op 965) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 965) least significant
s_mul_hi_u32 s19, s24, s8 // (op 965) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 965) most significant: sum
s_add_u32 s1, s1, s19 // (op 965) most significant: sum


s_mov_b32 s20, s26 // (op 965) reg expression
s_mov_b32 s21, s27 // (op 965) reg expression
s_mov_b32 s22, s0 // (op 965) reg expression
s_mov_b32 s23, 131072 // (op 965) call()

s_mul_hi_i32 s0, s16, s11 // (op 982) 
v_lshrrev_b32 v12, 3, v1 // (op 982) 
s_add_i32 s1, s0, s16 // (op 982) 

v_and_b32 v13, -8, v1 // (op 982) 
s_ashr_i32 s0, s1, 31 // (op 982) 
v_sub_u32 v14, v1, v13 // (op 982) 

s_cmp_eq_i32 s11, 0 // (op 982) 
s_mov_b32 s18, scc // (op 982) 
v_lshlrev_b32 v13, 4, v14 // (op 982) 

s_cmp_eq_i32 s18, 0 // (op 982) 
s_cselect_b32 s19, 0, -1 // (op 982) 

v_mov_b32 v15, 0 // (op 982) convert
v_mov_b32 v14, v13 // (op 982) convert

s_and_b32 s18, s12, 31 // (op 982) 
s_lshl_b32 s49, 1, s18 // (op 982) 
s_add_i32 s50, s49, s19 // (op 982) 


s_and_b32 s19, s0, s50 // (op 982) 


s_add_i32 s0, s1, s19 // (op 982) 


s_ashr_i32 s1, s0, s18 // (op 982) 


s_lshl_b32 s0, s12, 24 // (op 982) 
s_ashr_i32 s18, s0, 31 // (op 982) 

s_xor_b32 s0, s1, s18 // (op 982) 

s_sub_i32 s1, s0, s18 // (op 982) 


s_mul_hi_i32 s0, s1, s13 // (op 982) 
s_add_i32 s18, s0, s1 // (op 982) 

s_ashr_i32 s0, s18, 31 // (op 982) 
s_cmp_eq_i32 s13, 0 // (op 982) 
s_mov_b32 s19, scc // (op 982) 
s_cmp_eq_i32 s19, 0 // (op 982) 
s_cselect_b32 s49, 0, -1 // (op 982) 

s_and_b32 s19, s14, 31 // (op 982) 
s_lshl_b32 s50, 1, s19 // (op 982) 
s_add_i32 s51, s50, s49 // (op 982) 


s_and_b32 s49, s0, s51 // (op 982) 


s_add_i32 s0, s18, s49 // (op 982) 


s_ashr_i32 s18, s0, s19 // (op 982) 


s_lshl_b32 s0, s14, 24 // (op 982) 
s_ashr_i32 s19, s0, 31 // (op 982) 

s_xor_b32 s0, s18, s19 // (op 982) 

s_sub_i32 s18, s0, s19 // (op 982) 


s_mul_i32 s0, s1, s44 // (op 982) 

s_sub_u32 s1, s16, s0 // (op 982) 

s_mul_i32 s0, s18, s44 // (op 982) 
s_add_u32 s0, s0, s1 // (op 982) 


s_mul_hi_i32 s1, s16, s10 // (op 982) 
s_add_i32 s18, s1, s16 // (op 982) 

s_ashr_i32 s1, s18, 31 // (op 982) 
s_cmp_eq_i32 s10, 0 // (op 982) 
s_mov_b32 s19, scc // (op 982) 
s_cmp_eq_i32 s19, 0 // (op 982) 
s_cselect_b32 s49, 0, -1 // (op 982) 

s_and_b32 s19, s47, 31 // (op 982) 
s_lshl_b32 s50, 1, s19 // (op 982) 
s_add_i32 s51, s50, s49 // (op 982) 


s_and_b32 s49, s1, s51 // (op 982) 


s_add_i32 s1, s18, s49 // (op 982) 


s_ashr_i32 s18, s1, s19 // (op 982) 


s_lshl_b32 s1, s47, 24 // (op 982) 
s_ashr_i32 s19, s1, 31 // (op 982) 

s_xor_b32 s1, s18, s19 // (op 982) 

s_sub_i32 s18, s1, s19 // (op 982) 


s_add_u32 s54, s6, 64 // (op 982) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 982) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 982) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 982) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 6 // (op 982) 

s_mul_i32 s1, s44, s50 // (op 982) 

s_mul_i32 s19, s48, s1 // (op 982) 

s_mul_i32 s1, s18, s19 // (op 982) 

s_sub_u32 s19, s16, s1 // (op 982) 

s_mov_b32 s1, s19 // (op 982) 
s_mul_hi_i32 s49, s1, s15 // (op 982) 
s_add_i32 s50, s49, s1 // (op 982) 


s_ashr_i32 s1, s50, 31 // (op 982) 
s_cmp_eq_i32 s15, 0 // (op 982) 
s_mov_b32 s49, scc // (op 982) 
s_cmp_eq_i32 s49, 0 // (op 982) 
s_cselect_b32 s51, 0, -1 // (op 982) 

s_and_b32 s49, s46, 31 // (op 982) 
s_lshl_b32 s52, 1, s49 // (op 982) 
s_add_i32 s53, s52, s51 // (op 982) 


s_and_b32 s51, s1, s53 // (op 982) 


s_add_i32 s1, s50, s51 // (op 982) 


s_ashr_i32 s50, s1, s49 // (op 982) 


s_lshl_b32 s1, s46, 24 // (op 982) 
s_ashr_i32 s49, s1, 31 // (op 982) 

s_xor_b32 s1, s50, s49 // (op 982) 

s_sub_i32 s50, s1, s49 // (op 982) 


s_mul_i32 s1, s45, s44 // (op 982) 
s_add_u32 s54, s8, 32 // (op 982) Start of Int64 add, locking SCC
s_addc_u32 s55, s9, 0 // (op 982) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 982) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 982) most significant half; uses scc

s_ashr_i64 s[54:55], s[52:53], 5 // (op 982) 

s_sub_i32 s49, s54, s1 // (op 982) 

s_mul_i32 s51, s50, s49 // (op 982) 


s_sub_u32 s49, s19, s51 // (op 982) 


s_add_u32 s19, s49, s1 // (op 982) 


s_cmp_eq_i32 s18, 0 // (op 982) 
s_mov_b32 s1, scc // (op 982) 

s_cmp_eq_i32 s1, 0 // (op 982) 
s_cselect_b32 s18, s19, s0 // (op 982) 



v_lshl_add_u32 v13, s18, 5, v12 // (op 982) 


v_mul_lo_u32 v19, v13, s25 // (op 982) most significant: low of low * high
v_mul_hi_u32 v12, v13, s24 // (op 982) most significant: high of low * low
v_mul_lo_u32 v18, v13, s24 // (op 982) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 982) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v14 // (op 982) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 982) most significant half



s_mul_i32 s1, s24, s9 // (op 984) most significant: low of low * high
s_mul_i32 s18, s25, s8 // (op 984) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 984) least significant
s_mul_hi_u32 s19, s24, s8 // (op 984) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 984) most significant: sum
s_add_u32 s1, s1, s19 // (op 984) most significant: sum


s_mov_b32 s20, s26 // (op 984) reg expression
s_mov_b32 s21, s27 // (op 984) reg expression
s_mov_b32 s22, s0 // (op 984) reg expression
s_mov_b32 s23, 131072 // (op 984) call()

s_mul_hi_i32 s0, s16, s11 // (op 1027) 
v_and_b32 v12, -4, v1 // (op 1027) 
s_add_i32 s1, s0, s16 // (op 1027) 

v_sub_u32 v13, v1, v12 // (op 1027) 

s_ashr_i32 s0, s1, 31 // (op 1027) 
v_lshlrev_b32 v12, 4, v13 // (op 1027) 

s_cmp_eq_i32 s11, 0 // (op 1027) 
s_mov_b32 s18, scc // (op 1027) 
v_lshrrev_b32 v13, 2, v1 // (op 1027) 
s_cmp_eq_i32 s18, 0 // (op 1027) 
s_cselect_b32 s19, 0, -1 // (op 1027) 

s_and_b32 s18, s12, 31 // (op 1027) 
s_lshl_b32 s49, 1, s18 // (op 1027) 
s_add_i32 s50, s49, s19 // (op 1027) 


s_and_b32 s19, s0, s50 // (op 1027) 


s_add_i32 s0, s1, s19 // (op 1027) 


s_ashr_i32 s1, s0, s18 // (op 1027) 


s_lshl_b32 s0, s12, 24 // (op 1027) 
s_ashr_i32 s18, s0, 31 // (op 1027) 

s_xor_b32 s0, s1, s18 // (op 1027) 

s_sub_i32 s1, s0, s18 // (op 1027) 


s_mul_hi_i32 s0, s1, s13 // (op 1027) 
s_add_i32 s18, s0, s1 // (op 1027) 

s_ashr_i32 s0, s18, 31 // (op 1027) 
s_cmp_eq_i32 s13, 0 // (op 1027) 
s_mov_b32 s19, scc // (op 1027) 
s_cmp_eq_i32 s19, 0 // (op 1027) 
s_cselect_b32 s49, 0, -1 // (op 1027) 

s_and_b32 s19, s14, 31 // (op 1027) 
s_lshl_b32 s50, 1, s19 // (op 1027) 
s_add_i32 s51, s50, s49 // (op 1027) 


s_and_b32 s49, s0, s51 // (op 1027) 


s_add_i32 s0, s18, s49 // (op 1027) 


s_ashr_i32 s18, s0, s19 // (op 1027) 


s_lshl_b32 s0, s14, 24 // (op 1027) 
s_ashr_i32 s19, s0, 31 // (op 1027) 

s_xor_b32 s0, s18, s19 // (op 1027) 

s_sub_i32 s18, s0, s19 // (op 1027) 


s_add_u32 s54, s6, 64 // (op 1027) Start of Int64 add, locking SCC
s_addc_u32 s55, s7, 0 // (op 1027) End of Int64 add, unlocking SCC
s_sub_u32 s52, s54, 1 // (op 1027) least significant half; sets scc
s_subb_u32 s53, s55, 0 // (op 1027) most significant half; uses scc

s_ashr_i64 s[50:51], s[52:53], 6 // (op 1027) 

s_mul_i32 s0, s18, s50 // (op 1027) 

s_sub_i32 s49, s1, s0 // (op 1027) 


s_mul_hi_i32 s0, s16, s10 // (op 1027) 
s_add_i32 s1, s0, s16 // (op 1027) 

s_ashr_i32 s0, s1, 31 // (op 1027) 
s_cmp_eq_i32 s10, 0 // (op 1027) 
s_mov_b32 s18, scc // (op 1027) 
s_cmp_eq_i32 s18, 0 // (op 1027) 
s_cselect_b32 s19, 0, -1 // (op 1027) 

s_and_b32 s18, s47, 31 // (op 1027) 
s_lshl_b32 s52, 1, s18 // (op 1027) 
s_add_i32 s53, s52, s19 // (op 1027) 


s_and_b32 s19, s0, s53 // (op 1027) 


s_add_i32 s0, s1, s19 // (op 1027) 


s_ashr_i32 s1, s0, s18 // (op 1027) 


s_lshl_b32 s0, s47, 24 // (op 1027) 
s_ashr_i32 s18, s0, 31 // (op 1027) 

s_xor_b32 s0, s1, s18 // (op 1027) 

s_sub_i32 s1, s0, s18 // (op 1027) 


s_mul_i32 s0, s44, s50 // (op 1027) 

s_mul_i32 s18, s48, s0 // (op 1027) 

s_mul_i32 s0, s1, s18 // (op 1027) 

s_sub_u32 s18, s16, s0 // (op 1027) 

s_mul_hi_i32 s0, s18, s15 // (op 1027) 
s_add_i32 s19, s0, s18 // (op 1027) 


s_ashr_i32 s0, s19, 31 // (op 1027) 
s_cmp_eq_i32 s15, 0 // (op 1027) 
s_mov_b32 s18, scc // (op 1027) 
s_cmp_eq_i32 s18, 0 // (op 1027) 
s_cselect_b32 s50, 0, -1 // (op 1027) 

s_and_b32 s18, s46, 31 // (op 1027) 
s_lshl_b32 s51, 1, s18 // (op 1027) 
s_add_i32 s52, s51, s50 // (op 1027) 


s_and_b32 s50, s0, s52 // (op 1027) 


s_add_i32 s0, s19, s50 // (op 1027) 


s_ashr_i32 s19, s0, s18 // (op 1027) 


s_lshl_b32 s0, s46, 24 // (op 1027) 
s_ashr_i32 s18, s0, 31 // (op 1027) 

s_xor_b32 s0, s19, s18 // (op 1027) 

s_sub_i32 s19, s0, s18 // (op 1027) 


s_cmp_eq_i32 s1, 0 // (op 1027) 
s_mov_b32 s0, scc // (op 1027) 

s_cmp_eq_i32 s0, 0 // (op 1027) 
s_cselect_b32 s1, s19, s49 // (op 1027) 



v_lshl_add_u32 v14, s1, 6, v12 // (op 1027) 


v_mov_b32 v21, 0 // (op 1027) convert
v_mov_b32 v20, v14 // (op 1027) convert

v_mul_lo_u32 v15, v13, s29 // (op 1027) most significant: low of low * high
v_mul_hi_u32 v12, v13, s28 // (op 1027) most significant: high of low * low
v_mul_lo_u32 v14, v13, s28 // (op 1027) least significant: low of low * low
v_add_u32 v15, v15, v12 // (op 1027) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v20 // (op 1027) least significant half
v_addc_co_u32 v15, s[0:1], v15, v21, s[0:1] // (op 1027) most significant half



s_mul_i32 s1, s28, s33 // (op 1029) most significant: low of low * high
s_mul_i32 s18, s29, s32 // (op 1029) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1029) least significant
s_mul_hi_u32 s19, s28, s32 // (op 1029) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1029) most significant: sum
s_add_u32 s1, s1, s19 // (op 1029) most significant: sum


s_mov_b32 s52, s30 // (op 1029) reg expression
s_mov_b32 s53, s31 // (op 1029) reg expression
s_mov_b32 s54, s0 // (op 1029) reg expression
s_mov_b32 s55, 131072 // (op 1029) call()

s_mul_hi_i32 s0, s16, s11 // (op 1046) 
v_and_b32 v12, -4, v1 // (op 1046) 
s_add_i32 s1, s0, s16 // (op 1046) 

v_sub_u32 v13, v1, v12 // (op 1046) 

s_ashr_i32 s0, s1, 31 // (op 1046) 
v_lshlrev_b32 v12, 4, v13 // (op 1046) 

s_cmp_eq_i32 s11, 0 // (op 1046) 
s_mov_b32 s18, scc // (op 1046) 
v_lshrrev_b32 v13, 2, v1 // (op 1046) 
s_cmp_eq_i32 s18, 0 // (op 1046) 
s_cselect_b32 s19, 0, -1 // (op 1046) 

s_and_b32 s18, s12, 31 // (op 1046) 
s_lshl_b32 s49, 1, s18 // (op 1046) 
s_add_i32 s50, s49, s19 // (op 1046) 


s_and_b32 s19, s0, s50 // (op 1046) 


s_add_i32 s0, s1, s19 // (op 1046) 


s_ashr_i32 s1, s0, s18 // (op 1046) 


s_lshl_b32 s0, s12, 24 // (op 1046) 
s_ashr_i32 s18, s0, 31 // (op 1046) 

s_xor_b32 s0, s1, s18 // (op 1046) 

s_sub_i32 s1, s0, s18 // (op 1046) 


s_mul_hi_i32 s0, s1, s13 // (op 1046) 
s_add_i32 s18, s0, s1 // (op 1046) 

s_ashr_i32 s0, s18, 31 // (op 1046) 
s_cmp_eq_i32 s13, 0 // (op 1046) 
s_mov_b32 s19, scc // (op 1046) 
s_cmp_eq_i32 s19, 0 // (op 1046) 
s_cselect_b32 s49, 0, -1 // (op 1046) 

s_and_b32 s19, s14, 31 // (op 1046) 
s_lshl_b32 s50, 1, s19 // (op 1046) 
s_add_i32 s51, s50, s49 // (op 1046) 


s_and_b32 s49, s0, s51 // (op 1046) 


s_add_i32 s0, s18, s49 // (op 1046) 


s_ashr_i32 s18, s0, s19 // (op 1046) 


s_lshl_b32 s0, s14, 24 // (op 1046) 
s_ashr_i32 s19, s0, 31 // (op 1046) 

s_xor_b32 s0, s18, s19 // (op 1046) 

s_sub_i32 s18, s0, s19 // (op 1046) 


s_add_u32 s50, s6, 64 // (op 1046) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1046) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 1046) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 1046) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 6 // (op 1046) 

s_mul_i32 s0, s18, s50 // (op 1046) 

s_sub_i32 s49, s1, s0 // (op 1046) 


s_mul_hi_i32 s0, s16, s10 // (op 1046) 
s_add_i32 s1, s0, s16 // (op 1046) 

s_ashr_i32 s0, s1, 31 // (op 1046) 
s_cmp_eq_i32 s10, 0 // (op 1046) 
s_mov_b32 s18, scc // (op 1046) 
s_cmp_eq_i32 s18, 0 // (op 1046) 
s_cselect_b32 s19, 0, -1 // (op 1046) 

s_and_b32 s18, s47, 31 // (op 1046) 
s_lshl_b32 s64, 1, s18 // (op 1046) 
s_add_i32 s65, s64, s19 // (op 1046) 


s_and_b32 s19, s0, s65 // (op 1046) 


s_add_i32 s0, s1, s19 // (op 1046) 


s_ashr_i32 s1, s0, s18 // (op 1046) 


s_lshl_b32 s0, s47, 24 // (op 1046) 
s_ashr_i32 s18, s0, 31 // (op 1046) 

s_xor_b32 s0, s1, s18 // (op 1046) 

s_sub_i32 s1, s0, s18 // (op 1046) 


s_mul_i32 s0, s44, s50 // (op 1046) 

s_mul_i32 s18, s48, s0 // (op 1046) 

s_mul_i32 s0, s1, s18 // (op 1046) 

s_sub_u32 s18, s16, s0 // (op 1046) 

s_mul_hi_i32 s0, s18, s15 // (op 1046) 
s_add_i32 s19, s0, s18 // (op 1046) 


s_ashr_i32 s0, s19, 31 // (op 1046) 
s_cmp_eq_i32 s15, 0 // (op 1046) 
s_mov_b32 s18, scc // (op 1046) 
s_cmp_eq_i32 s18, 0 // (op 1046) 
s_cselect_b32 s50, 0, -1 // (op 1046) 

s_and_b32 s18, s46, 31 // (op 1046) 
s_lshl_b32 s51, 1, s18 // (op 1046) 
s_add_i32 s64, s51, s50 // (op 1046) 


s_and_b32 s50, s0, s64 // (op 1046) 


s_add_i32 s0, s19, s50 // (op 1046) 


s_ashr_i32 s19, s0, s18 // (op 1046) 


s_lshl_b32 s0, s46, 24 // (op 1046) 
s_ashr_i32 s18, s0, 31 // (op 1046) 

s_xor_b32 s0, s19, s18 // (op 1046) 

s_sub_i32 s19, s0, s18 // (op 1046) 


s_cmp_eq_i32 s1, 0 // (op 1046) 
s_mov_b32 s0, scc // (op 1046) 

s_cmp_eq_i32 s0, 0 // (op 1046) 
s_cselect_b32 s1, s19, s49 // (op 1046) 



v_lshl_add_u32 v20, s1, 6, v12 // (op 1046) 


v_mov_b32 v23, 0 // (op 1046) convert
v_mov_b32 v22, v20 // (op 1046) convert

v_mul_lo_u32 v21, v13, s29 // (op 1046) most significant: low of low * high
v_mul_hi_u32 v12, v13, s28 // (op 1046) most significant: high of low * low
v_mul_lo_u32 v20, v13, s28 // (op 1046) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 1046) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v22 // (op 1046) least significant half
v_addc_co_u32 v21, s[0:1], v21, v23, s[0:1] // (op 1046) most significant half



s_mul_i32 s1, s28, s33 // (op 1048) most significant: low of low * high
s_mul_i32 s18, s29, s32 // (op 1048) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1048) least significant
s_mul_hi_u32 s19, s28, s32 // (op 1048) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1048) most significant: sum
s_add_u32 s1, s1, s19 // (op 1048) most significant: sum


s_mov_b32 s52, s30 // (op 1048) reg expression
s_mov_b32 s53, s31 // (op 1048) reg expression
s_mov_b32 s54, s0 // (op 1048) reg expression
s_mov_b32 s55, 131072 // (op 1048) call()

s_mul_hi_i32 s0, s16, s11 // (op 1091) 
v_lshrrev_b32 v12, 7, v1 // (op 1091) 
s_add_i32 s1, s0, s16 // (op 1091) 

v_and_b32 v13, 15, v1 // (op 1091) 
s_ashr_i32 s0, s1, 31 // (op 1091) 
v_lshl_add_u32 v22, v12, 4, v13 // (op 1091) 


s_cmp_eq_i32 s11, 0 // (op 1091) 
s_mov_b32 s18, scc // (op 1091) 
v_and_b32 v12, 63, v1 // (op 1091) 
s_cmp_eq_i32 s18, 0 // (op 1091) 
s_cselect_b32 s19, 0, -1 // (op 1091) 

v_lshrrev_b32 v13, 4, v12 // (op 1091) 

s_and_b32 s18, s12, 31 // (op 1091) 
s_lshl_b32 s49, 1, s18 // (op 1091) 
s_add_i32 s50, s49, s19 // (op 1091) 


s_and_b32 s19, s0, s50 // (op 1091) 


s_add_i32 s0, s1, s19 // (op 1091) 


s_ashr_i32 s1, s0, s18 // (op 1091) 


s_lshl_b32 s0, s12, 24 // (op 1091) 
s_ashr_i32 s18, s0, 31 // (op 1091) 

s_xor_b32 s0, s1, s18 // (op 1091) 

s_sub_i32 s1, s0, s18 // (op 1091) 


s_mul_hi_i32 s0, s1, s13 // (op 1091) 
s_add_i32 s18, s0, s1 // (op 1091) 

s_ashr_i32 s0, s18, 31 // (op 1091) 
s_cmp_eq_i32 s13, 0 // (op 1091) 
s_mov_b32 s19, scc // (op 1091) 
s_cmp_eq_i32 s19, 0 // (op 1091) 
s_cselect_b32 s49, 0, -1 // (op 1091) 

s_and_b32 s19, s14, 31 // (op 1091) 
s_lshl_b32 s50, 1, s19 // (op 1091) 
s_add_i32 s51, s50, s49 // (op 1091) 


s_and_b32 s49, s0, s51 // (op 1091) 


s_add_i32 s0, s18, s49 // (op 1091) 


s_ashr_i32 s18, s0, s19 // (op 1091) 


s_lshl_b32 s0, s14, 24 // (op 1091) 
s_ashr_i32 s19, s0, 31 // (op 1091) 

s_xor_b32 s0, s18, s19 // (op 1091) 

s_sub_i32 s18, s0, s19 // (op 1091) 


s_mul_i32 s0, s1, s44 // (op 1091) 

s_sub_u32 s1, s16, s0 // (op 1091) 

s_mul_i32 s0, s18, s44 // (op 1091) 
s_add_u32 s0, s0, s1 // (op 1091) 


s_mul_hi_i32 s1, s16, s10 // (op 1091) 
s_add_i32 s18, s1, s16 // (op 1091) 

s_ashr_i32 s1, s18, 31 // (op 1091) 
s_cmp_eq_i32 s10, 0 // (op 1091) 
s_mov_b32 s19, scc // (op 1091) 
s_cmp_eq_i32 s19, 0 // (op 1091) 
s_cselect_b32 s49, 0, -1 // (op 1091) 

s_and_b32 s19, s47, 31 // (op 1091) 
s_lshl_b32 s50, 1, s19 // (op 1091) 
s_add_i32 s51, s50, s49 // (op 1091) 


s_and_b32 s49, s1, s51 // (op 1091) 


s_add_i32 s1, s18, s49 // (op 1091) 


s_ashr_i32 s18, s1, s19 // (op 1091) 


s_lshl_b32 s1, s47, 24 // (op 1091) 
s_ashr_i32 s19, s1, 31 // (op 1091) 

s_xor_b32 s1, s18, s19 // (op 1091) 

s_sub_i32 s18, s1, s19 // (op 1091) 


s_add_u32 s50, s6, 64 // (op 1091) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1091) End of Int64 add, unlocking SCC
s_sub_u32 s64, s50, 1 // (op 1091) least significant half; sets scc
s_subb_u32 s65, s51, 0 // (op 1091) most significant half; uses scc

s_ashr_i64 s[50:51], s[64:65], 6 // (op 1091) 

s_mul_i32 s1, s44, s50 // (op 1091) 

s_mul_i32 s19, s48, s1 // (op 1091) 

s_mul_i32 s1, s18, s19 // (op 1091) 

s_sub_u32 s19, s16, s1 // (op 1091) 

s_mov_b32 s1, s19 // (op 1091) 
s_mul_hi_i32 s49, s1, s15 // (op 1091) 
s_add_i32 s50, s49, s1 // (op 1091) 


s_ashr_i32 s1, s50, 31 // (op 1091) 
s_cmp_eq_i32 s15, 0 // (op 1091) 
s_mov_b32 s49, scc // (op 1091) 
s_cmp_eq_i32 s49, 0 // (op 1091) 
s_cselect_b32 s51, 0, -1 // (op 1091) 

s_and_b32 s49, s46, 31 // (op 1091) 
s_lshl_b32 s64, 1, s49 // (op 1091) 
s_add_i32 s65, s64, s51 // (op 1091) 


s_and_b32 s51, s1, s65 // (op 1091) 


s_add_i32 s1, s50, s51 // (op 1091) 


s_ashr_i32 s50, s1, s49 // (op 1091) 


s_lshl_b32 s1, s46, 24 // (op 1091) 
s_ashr_i32 s49, s1, 31 // (op 1091) 

s_xor_b32 s1, s50, s49 // (op 1091) 

s_sub_i32 s50, s1, s49 // (op 1091) 


s_mul_i32 s1, s45, s44 // (op 1091) 
s_add_u32 s64, s8, 32 // (op 1091) Start of Int64 add, locking SCC
s_addc_u32 s65, s9, 0 // (op 1091) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 1091) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 1091) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 5 // (op 1091) 

s_sub_i32 s49, s64, s1 // (op 1091) 

s_mul_i32 s51, s50, s49 // (op 1091) 


s_sub_u32 s49, s19, s51 // (op 1091) 


s_add_u32 s19, s49, s1 // (op 1091) 


s_cmp_eq_i32 s18, 0 // (op 1091) 
s_mov_b32 s1, scc // (op 1091) 

s_cmp_eq_i32 s1, 0 // (op 1091) 
s_cselect_b32 s18, s19, s0 // (op 1091) 



v_lshl_add_u32 v12, s18, 5, v22 // (op 1091) 


v_mul_lo_u32 v22, v12, s34 // (op 1091) 
v_add_u32 v22, v22, v13 // (op 1091) 


s_mul_i32 s1, s34, s9 // (op 1093) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 1093) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1093) least significant
s_mul_hi_u32 s19, s34, s8 // (op 1093) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1093) most significant: sum
s_add_u32 s1, s1, s19 // (op 1093) most significant: sum


s_mov_b32 s64, s36 // (op 1093) reg expression
s_mov_b32 s65, s37 // (op 1093) reg expression
s_mov_b32 s66, s0 // (op 1093) reg expression
s_mov_b32 s67, 131072 // (op 1093) call()

s_mul_hi_i32 s0, s16, s11 // (op 1110) 
v_lshrrev_b32 v12, 7, v1 // (op 1110) 
s_add_i32 s1, s0, s16 // (op 1110) 

v_and_b32 v13, 15, v1 // (op 1110) 
s_ashr_i32 s0, s1, 31 // (op 1110) 
v_lshl_add_u32 v23, v12, 4, v13 // (op 1110) 


s_cmp_eq_i32 s11, 0 // (op 1110) 
s_mov_b32 s18, scc // (op 1110) 
v_and_b32 v12, 63, v1 // (op 1110) 
s_cmp_eq_i32 s18, 0 // (op 1110) 
s_cselect_b32 s19, 0, -1 // (op 1110) 

v_lshrrev_b32 v13, 4, v12 // (op 1110) 

s_and_b32 s18, s12, 31 // (op 1110) 
s_lshl_b32 s49, 1, s18 // (op 1110) 
s_add_i32 s50, s49, s19 // (op 1110) 


s_and_b32 s19, s0, s50 // (op 1110) 


s_add_i32 s0, s1, s19 // (op 1110) 


s_ashr_i32 s1, s0, s18 // (op 1110) 


s_lshl_b32 s0, s12, 24 // (op 1110) 
s_ashr_i32 s18, s0, 31 // (op 1110) 

s_xor_b32 s0, s1, s18 // (op 1110) 

s_sub_i32 s1, s0, s18 // (op 1110) 


s_mul_hi_i32 s0, s1, s13 // (op 1110) 
s_add_i32 s18, s0, s1 // (op 1110) 

s_ashr_i32 s0, s18, 31 // (op 1110) 
s_cmp_eq_i32 s13, 0 // (op 1110) 
s_mov_b32 s19, scc // (op 1110) 
s_cmp_eq_i32 s19, 0 // (op 1110) 
s_cselect_b32 s49, 0, -1 // (op 1110) 

s_and_b32 s19, s14, 31 // (op 1110) 
s_lshl_b32 s50, 1, s19 // (op 1110) 
s_add_i32 s51, s50, s49 // (op 1110) 


s_and_b32 s49, s0, s51 // (op 1110) 


s_add_i32 s0, s18, s49 // (op 1110) 


s_ashr_i32 s18, s0, s19 // (op 1110) 


s_lshl_b32 s0, s14, 24 // (op 1110) 
s_ashr_i32 s19, s0, 31 // (op 1110) 

s_xor_b32 s0, s18, s19 // (op 1110) 

s_sub_i32 s18, s0, s19 // (op 1110) 


s_mul_i32 s0, s1, s44 // (op 1110) 

s_sub_u32 s1, s16, s0 // (op 1110) 

s_mul_i32 s0, s18, s44 // (op 1110) 
s_add_u32 s0, s0, s1 // (op 1110) 


s_mul_hi_i32 s1, s16, s10 // (op 1110) 
s_add_i32 s18, s1, s16 // (op 1110) 

s_ashr_i32 s1, s18, 31 // (op 1110) 
s_cmp_eq_i32 s10, 0 // (op 1110) 
s_mov_b32 s19, scc // (op 1110) 
s_cmp_eq_i32 s19, 0 // (op 1110) 
s_cselect_b32 s49, 0, -1 // (op 1110) 

s_and_b32 s19, s47, 31 // (op 1110) 
s_lshl_b32 s50, 1, s19 // (op 1110) 
s_add_i32 s51, s50, s49 // (op 1110) 


s_and_b32 s49, s1, s51 // (op 1110) 


s_add_i32 s1, s18, s49 // (op 1110) 


s_ashr_i32 s18, s1, s19 // (op 1110) 


s_lshl_b32 s1, s47, 24 // (op 1110) 
s_ashr_i32 s19, s1, 31 // (op 1110) 

s_xor_b32 s1, s18, s19 // (op 1110) 

s_sub_i32 s18, s1, s19 // (op 1110) 


s_add_u32 s50, s6, 64 // (op 1110) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1110) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 1110) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 1110) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 6 // (op 1110) 

s_mul_i32 s1, s44, s50 // (op 1110) 

s_mul_i32 s19, s48, s1 // (op 1110) 

s_mul_i32 s1, s18, s19 // (op 1110) 

s_sub_u32 s19, s16, s1 // (op 1110) 

s_mov_b32 s1, s19 // (op 1110) 
s_mul_hi_i32 s49, s1, s15 // (op 1110) 
s_add_i32 s50, s49, s1 // (op 1110) 


s_ashr_i32 s1, s50, 31 // (op 1110) 
s_cmp_eq_i32 s15, 0 // (op 1110) 
s_mov_b32 s49, scc // (op 1110) 
s_cmp_eq_i32 s49, 0 // (op 1110) 
s_cselect_b32 s51, 0, -1 // (op 1110) 

s_and_b32 s49, s46, 31 // (op 1110) 
s_lshl_b32 s68, 1, s49 // (op 1110) 
s_add_i32 s69, s68, s51 // (op 1110) 


s_and_b32 s51, s1, s69 // (op 1110) 


s_add_i32 s1, s50, s51 // (op 1110) 


s_ashr_i32 s50, s1, s49 // (op 1110) 


s_lshl_b32 s1, s46, 24 // (op 1110) 
s_ashr_i32 s49, s1, 31 // (op 1110) 

s_xor_b32 s1, s50, s49 // (op 1110) 

s_sub_i32 s50, s1, s49 // (op 1110) 


s_mul_i32 s1, s45, s44 // (op 1110) 
s_add_u32 s68, s8, 32 // (op 1110) Start of Int64 add, locking SCC
s_addc_u32 s69, s9, 0 // (op 1110) End of Int64 add, unlocking SCC
s_sub_u32 s70, s68, 1 // (op 1110) least significant half; sets scc
s_subb_u32 s71, s69, 0 // (op 1110) most significant half; uses scc

s_ashr_i64 s[68:69], s[70:71], 5 // (op 1110) 

s_sub_i32 s49, s68, s1 // (op 1110) 

s_mul_i32 s51, s50, s49 // (op 1110) 


s_sub_u32 s49, s19, s51 // (op 1110) 


s_add_u32 s19, s49, s1 // (op 1110) 


s_cmp_eq_i32 s18, 0 // (op 1110) 
s_mov_b32 s1, scc // (op 1110) 

s_cmp_eq_i32 s1, 0 // (op 1110) 
s_cselect_b32 s18, s19, s0 // (op 1110) 



v_lshl_add_u32 v12, s18, 5, v23 // (op 1110) 


v_mul_lo_u32 v23, v12, s34 // (op 1110) 
v_add_u32 v23, v23, v13 // (op 1110) 


s_mul_i32 s1, s34, s9 // (op 1112) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 1112) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1112) least significant
s_mul_hi_u32 s19, s34, s8 // (op 1112) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1112) most significant: sum
s_add_u32 s1, s1, s19 // (op 1112) most significant: sum


s_mov_b32 s64, s36 // (op 1112) reg expression
s_mov_b32 s65, s37 // (op 1112) reg expression
s_mov_b32 s66, s0 // (op 1112) reg expression
s_mov_b32 s67, 131072 // (op 1112) call()

s_mul_hi_i32 s0, s16, s11 // (op 1155) 
v_and_b32 v12, -64, v1 // (op 1155) 
s_add_i32 s1, s0, s16 // (op 1155) 

v_sub_u32 v13, v1, v12 // (op 1155) 

s_ashr_i32 s0, s1, 31 // (op 1155) 
v_lshrrev_b32 v12, 6, v1 // (op 1155) 
s_cmp_eq_i32 s11, 0 // (op 1155) 
s_mov_b32 s18, scc // (op 1155) 
s_cmp_eq_i32 s18, 0 // (op 1155) 
s_cselect_b32 s19, 0, -1 // (op 1155) 

s_and_b32 s18, s12, 31 // (op 1155) 
s_lshl_b32 s49, 1, s18 // (op 1155) 
s_add_i32 s50, s49, s19 // (op 1155) 


s_and_b32 s19, s0, s50 // (op 1155) 


s_add_i32 s0, s1, s19 // (op 1155) 


s_ashr_i32 s1, s0, s18 // (op 1155) 


s_lshl_b32 s0, s12, 24 // (op 1155) 
s_ashr_i32 s18, s0, 31 // (op 1155) 

s_xor_b32 s0, s1, s18 // (op 1155) 

s_sub_i32 s1, s0, s18 // (op 1155) 


s_mul_hi_i32 s0, s1, s13 // (op 1155) 
s_add_i32 s18, s0, s1 // (op 1155) 

s_ashr_i32 s0, s18, 31 // (op 1155) 
s_cmp_eq_i32 s13, 0 // (op 1155) 
s_mov_b32 s19, scc // (op 1155) 
s_cmp_eq_i32 s19, 0 // (op 1155) 
s_cselect_b32 s49, 0, -1 // (op 1155) 

s_and_b32 s19, s14, 31 // (op 1155) 
s_lshl_b32 s50, 1, s19 // (op 1155) 
s_add_i32 s51, s50, s49 // (op 1155) 


s_and_b32 s49, s0, s51 // (op 1155) 


s_add_i32 s0, s18, s49 // (op 1155) 


s_ashr_i32 s18, s0, s19 // (op 1155) 


s_lshl_b32 s0, s14, 24 // (op 1155) 
s_ashr_i32 s19, s0, 31 // (op 1155) 

s_xor_b32 s0, s18, s19 // (op 1155) 

s_sub_i32 s18, s0, s19 // (op 1155) 


s_add_u32 s50, s6, 64 // (op 1155) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1155) End of Int64 add, unlocking SCC
s_sub_u32 s68, s50, 1 // (op 1155) least significant half; sets scc
s_subb_u32 s69, s51, 0 // (op 1155) most significant half; uses scc

s_ashr_i64 s[50:51], s[68:69], 6 // (op 1155) 

s_mul_i32 s0, s18, s50 // (op 1155) 

s_sub_i32 s49, s1, s0 // (op 1155) 


s_mul_hi_i32 s0, s16, s10 // (op 1155) 
s_add_i32 s1, s0, s16 // (op 1155) 

s_ashr_i32 s0, s1, 31 // (op 1155) 
s_cmp_eq_i32 s10, 0 // (op 1155) 
s_mov_b32 s18, scc // (op 1155) 
s_cmp_eq_i32 s18, 0 // (op 1155) 
s_cselect_b32 s19, 0, -1 // (op 1155) 

s_and_b32 s18, s47, 31 // (op 1155) 
s_lshl_b32 s68, 1, s18 // (op 1155) 
s_add_i32 s69, s68, s19 // (op 1155) 


s_and_b32 s19, s0, s69 // (op 1155) 


s_add_i32 s0, s1, s19 // (op 1155) 


s_ashr_i32 s1, s0, s18 // (op 1155) 


s_lshl_b32 s0, s47, 24 // (op 1155) 
s_ashr_i32 s18, s0, 31 // (op 1155) 

s_xor_b32 s0, s1, s18 // (op 1155) 

s_sub_i32 s1, s0, s18 // (op 1155) 


s_mul_i32 s0, s44, s50 // (op 1155) 

s_mul_i32 s18, s48, s0 // (op 1155) 

s_mul_i32 s0, s1, s18 // (op 1155) 

s_sub_u32 s18, s16, s0 // (op 1155) 

s_mul_hi_i32 s0, s18, s15 // (op 1155) 
s_add_i32 s19, s0, s18 // (op 1155) 


s_ashr_i32 s0, s19, 31 // (op 1155) 
s_cmp_eq_i32 s15, 0 // (op 1155) 
s_mov_b32 s18, scc // (op 1155) 
s_cmp_eq_i32 s18, 0 // (op 1155) 
s_cselect_b32 s50, 0, -1 // (op 1155) 

s_and_b32 s18, s46, 31 // (op 1155) 
s_lshl_b32 s51, 1, s18 // (op 1155) 
s_add_i32 s68, s51, s50 // (op 1155) 


s_and_b32 s50, s0, s68 // (op 1155) 


s_add_i32 s0, s19, s50 // (op 1155) 


s_ashr_i32 s19, s0, s18 // (op 1155) 


s_lshl_b32 s0, s46, 24 // (op 1155) 
s_ashr_i32 s18, s0, 31 // (op 1155) 

s_xor_b32 s0, s19, s18 // (op 1155) 

s_sub_i32 s19, s0, s18 // (op 1155) 


s_cmp_eq_i32 s1, 0 // (op 1155) 
s_mov_b32 s0, scc // (op 1155) 

s_cmp_eq_i32 s0, 0 // (op 1155) 
s_cselect_b32 s1, s19, s49 // (op 1155) 



v_lshl_add_u32 v24, s1, 6, v13 // (op 1155) 


v_mul_lo_u32 v13, v12, s38 // (op 1155) 
v_add_u32 v13, v13, v24 // (op 1155) 


s_mul_i32 s1, s38, s43 // (op 1157) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 1157) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 1157) least significant
s_mul_hi_u32 s19, s38, s42 // (op 1157) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1157) most significant: sum
s_add_u32 s1, s1, s19 // (op 1157) most significant: sum


s_mov_b32 s68, s40 // (op 1157) reg expression
s_mov_b32 s69, s41 // (op 1157) reg expression
s_mov_b32 s70, s0 // (op 1157) reg expression
s_mov_b32 s71, 131072 // (op 1157) call()

s_mul_hi_i32 s0, s16, s11 // (op 1174) 
v_and_b32 v12, -64, v1 // (op 1174) 
s_add_i32 s1, s0, s16 // (op 1174) 

v_sub_u32 v24, v1, v12 // (op 1174) 

s_ashr_i32 s0, s1, 31 // (op 1174) 
v_lshrrev_b32 v12, 6, v1 // (op 1174) 
s_cmp_eq_i32 s11, 0 // (op 1174) 
s_mov_b32 s18, scc // (op 1174) 
s_cmp_eq_i32 s18, 0 // (op 1174) 
s_cselect_b32 s19, 0, -1 // (op 1174) 

s_and_b32 s18, s12, 31 // (op 1174) 
s_lshl_b32 s49, 1, s18 // (op 1174) 
s_add_i32 s50, s49, s19 // (op 1174) 


s_and_b32 s19, s0, s50 // (op 1174) 


s_add_i32 s0, s1, s19 // (op 1174) 


s_ashr_i32 s1, s0, s18 // (op 1174) 


s_lshl_b32 s0, s12, 24 // (op 1174) 
s_ashr_i32 s18, s0, 31 // (op 1174) 

s_xor_b32 s0, s1, s18 // (op 1174) 

s_sub_i32 s1, s0, s18 // (op 1174) 


s_mul_hi_i32 s0, s1, s13 // (op 1174) 
s_add_i32 s18, s0, s1 // (op 1174) 

s_ashr_i32 s0, s18, 31 // (op 1174) 
s_cmp_eq_i32 s13, 0 // (op 1174) 
s_mov_b32 s19, scc // (op 1174) 
s_cmp_eq_i32 s19, 0 // (op 1174) 
s_cselect_b32 s49, 0, -1 // (op 1174) 

s_and_b32 s19, s14, 31 // (op 1174) 
s_lshl_b32 s50, 1, s19 // (op 1174) 
s_add_i32 s51, s50, s49 // (op 1174) 


s_and_b32 s49, s0, s51 // (op 1174) 


s_add_i32 s0, s18, s49 // (op 1174) 


s_ashr_i32 s18, s0, s19 // (op 1174) 


s_lshl_b32 s0, s14, 24 // (op 1174) 
s_ashr_i32 s19, s0, 31 // (op 1174) 

s_xor_b32 s0, s18, s19 // (op 1174) 

s_sub_i32 s18, s0, s19 // (op 1174) 


s_add_u32 s50, s6, 64 // (op 1174) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1174) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1174) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1174) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1174) 

s_mul_i32 s0, s18, s50 // (op 1174) 

s_sub_i32 s49, s1, s0 // (op 1174) 


s_mul_hi_i32 s0, s16, s10 // (op 1174) 
s_add_i32 s1, s0, s16 // (op 1174) 

s_ashr_i32 s0, s1, 31 // (op 1174) 
s_cmp_eq_i32 s10, 0 // (op 1174) 
s_mov_b32 s18, scc // (op 1174) 
s_cmp_eq_i32 s18, 0 // (op 1174) 
s_cselect_b32 s19, 0, -1 // (op 1174) 

s_and_b32 s18, s47, 31 // (op 1174) 
s_lshl_b32 s72, 1, s18 // (op 1174) 
s_add_i32 s73, s72, s19 // (op 1174) 


s_and_b32 s19, s0, s73 // (op 1174) 


s_add_i32 s0, s1, s19 // (op 1174) 


s_ashr_i32 s1, s0, s18 // (op 1174) 


s_lshl_b32 s0, s47, 24 // (op 1174) 
s_ashr_i32 s18, s0, 31 // (op 1174) 

s_xor_b32 s0, s1, s18 // (op 1174) 

s_sub_i32 s1, s0, s18 // (op 1174) 


s_mul_i32 s0, s44, s50 // (op 1174) 

s_mul_i32 s18, s48, s0 // (op 1174) 

s_mul_i32 s0, s1, s18 // (op 1174) 

s_sub_u32 s18, s16, s0 // (op 1174) 

s_mul_hi_i32 s0, s18, s15 // (op 1174) 
s_add_i32 s19, s0, s18 // (op 1174) 


s_ashr_i32 s0, s19, 31 // (op 1174) 
s_cmp_eq_i32 s15, 0 // (op 1174) 
s_mov_b32 s18, scc // (op 1174) 
s_cmp_eq_i32 s18, 0 // (op 1174) 
s_cselect_b32 s50, 0, -1 // (op 1174) 

s_and_b32 s18, s46, 31 // (op 1174) 
s_lshl_b32 s51, 1, s18 // (op 1174) 
s_add_i32 s72, s51, s50 // (op 1174) 


s_and_b32 s50, s0, s72 // (op 1174) 


s_add_i32 s0, s19, s50 // (op 1174) 


s_ashr_i32 s19, s0, s18 // (op 1174) 


s_lshl_b32 s0, s46, 24 // (op 1174) 
s_ashr_i32 s18, s0, 31 // (op 1174) 

s_xor_b32 s0, s19, s18 // (op 1174) 

s_sub_i32 s19, s0, s18 // (op 1174) 


s_cmp_eq_i32 s1, 0 // (op 1174) 
s_mov_b32 s0, scc // (op 1174) 

s_cmp_eq_i32 s0, 0 // (op 1174) 
s_cselect_b32 s1, s19, s49 // (op 1174) 



v_lshl_add_u32 v25, s1, 6, v24 // (op 1174) 


v_mul_lo_u32 v24, v12, s38 // (op 1174) 
v_add_u32 v24, v24, v25 // (op 1174) 


s_mul_i32 s1, s38, s43 // (op 1176) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 1176) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 1176) least significant
s_mul_hi_u32 s19, s38, s42 // (op 1176) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1176) most significant: sum
s_add_u32 s1, s1, s19 // (op 1176) most significant: sum


s_mov_b32 s68, s40 // (op 1176) reg expression
s_mov_b32 s69, s41 // (op 1176) reg expression
s_mov_b32 s70, s0 // (op 1176) reg expression
s_mov_b32 s71, 131072 // (op 1176) call()




buffer_load_ubyte v12, v22, s[64:67], 0 offen offset: 0 // (op 769) Load value
v_and_b32 v25, -4, v1 // (op 1292) 
v_sub_u32 v26, v1, v25 // (op 1292) 

v_lshlrev_b32 v25, 4, v26 // (op 1292) 

v_lshrrev_b32 v26, 2, v1 // (op 1292) 
v_lshl_add_u32 v27, v26, 6, v25 // (op 1292) 


s_nop 0
v_readfirstlane_b32 s0, v27 // (op 1292) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[28:29], 6 // (op 911) 
v_mov_b64 v[26:27], s[18:19] // (op 911) convert


v_mov_b64 v[28:29], v[14:15] // (op 911) reg expression
s_mov_b32 m0, s0 // (op 911) reg expression
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 911) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 911) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 911) most significant half

s_add_u32 m0, m0, 4096 // (op 911) 
buffer_load_dwordx4 v28, s[52:55], 0 offen offset: 0 lds // (op 911) Load value direct to lds





v_mov_b32 v25, v13 // (op 776) reg expression

buffer_load_ubyte v26, v25, s[68:71], 0 offen offset: 0 // (op 776) Load value

v_and_b32 v25, -8, v1 // (op 1240) 
v_sub_u32 v27, v1, v25 // (op 1240) 

v_lshlrev_b32 v25, 4, v27 // (op 1240) 

v_lshrrev_b32 v27, 3, v1 // (op 1240) 
v_lshl_add_u32 v28, v27, 7, v25 // (op 1240) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 1240) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 8192, s0 // (op 916) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 916) reg expression
buffer_load_dwordx4 v16, s[20:23], 0 offen offset: 0 lds // (op 916) Load value direct to lds





buffer_load_ubyte v25, v22, s[64:67], 0 offen offset:4 // (op 783) Load value

v_and_b32 v22, -4, v1 // (op 1451) 
v_sub_u32 v27, v1, v22 // (op 1451) 

v_lshlrev_b32 v22, 4, v27 // (op 1451) 

v_lshrrev_b32 v27, 2, v1 // (op 1451) 
v_lshl_add_u32 v28, v27, 6, v22 // (op 1451) 


s_nop 0
v_readfirstlane_b32 s0, v28 // (op 1451) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[18:19], s[28:29], 6 // (op 921) 
v_mov_b64 v[28:29], s[18:19] // (op 921) convert


s_lshl_b64 s[18:19], s[28:29], 7 // (op 921) 
v_mov_b32 v22, s18 // (op 921) 
v_mov_b32 v27, s19 // (op 921) 
v_add_co_u32 v30, s[50:51], v14, v22 // (op 921) least significant half
v_addc_co_u32 v31, s[50:51], v15, v27, s[50:51] // (op 921) most significant half




s_add_u32 s1, 12288, s0 // (op 921) 
s_mov_b32 m0, s1 // (op 921) reg expression
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 921) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 921) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 921) most significant half

s_add_u32 m0, m0, 4096 // (op 921) 
buffer_load_dwordx4 v30, s[52:55], 0 offen offset: 0 lds // (op 921) Load value direct to lds







s_lshl_b32 s0, s38, 2 // (op 790) 
v_add_u32 v22, v13, s0 // (op 790) 


buffer_load_ubyte v14, v22, s[68:71], 0 offen offset: 0 // (op 790) Load value


v_and_b32 v13, -8, v1 // (op 1412) 
v_sub_u32 v15, v1, v13 // (op 1412) 

v_lshlrev_b32 v13, 4, v15 // (op 1412) 

v_lshrrev_b32 v15, 3, v1 // (op 1412) 
v_lshl_add_u32 v22, v15, 7, v13 // (op 1412) 


s_nop 0
v_readfirstlane_b32 s0, v22 // (op 1412) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s1, 20480, s0 // (op 923) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s1 // (op 923) reg expression
s_mov_b32 s18, 128 // (op 923) call()
buffer_load_dwordx4 v16, s[20:23], s18 offen offset: 0 lds // (op 923) Load value direct to lds




v_and_b32 v13, -64, v1 // (op 1350) 
v_sub_u32 v22, v1, v13 // (op 1350) 

v_and_b32 v13, 4294967232, v1 // (op 1350) 
v_add_u32 v15, v13, v22 // (op 1350) 





s_waitcnt vmcnt(6)
ds_write_b8 v15, v26 offset:24576 // (op 796) Store local data 

s_barrier  // (op 1554) 



ds_read_b64_tr_b8 v[28:29], v7 // (op 814) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v7 offset:512 // (op 814) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v7 offset:4096 // (op 814) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v7 offset:4608 // (op 814) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1570) 


ds_read_u8 v13, v0 offset:24576 // (op 824) Load local data 



ds_read_b64_tr_b8 v[36:37], v7 offset:32 // (op 819) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v7 offset:544 // (op 819) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v7 offset:4128 // (op 819) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v7 offset:4640 // (op 819) Transpose load local data 


ds_read_u8 v22, v0 offset:24608 // (op 829) Load local data 
s_waitcnt vmcnt(5)
s_barrier  // (op 1538) 



ds_read_b128 v[44:47], v3 offset:8192 // (op 810) Load local data 
ds_read_b128 v[48:51], v3 offset:8256 // (op 810) Load local data 
v_and_b32 v27, -8, v1 // (op 1212) 
v_sub_u32 v15, v1, v27 // (op 1212) 

v_lshlrev_b32 v27, 4, v15 // (op 1212) 

v_lshrrev_b32 v15, 3, v1 // (op 1212) 
v_lshl_add_u32 v16, v15, 7, v27 // (op 1212) 


s_nop 0
v_readfirstlane_b32 s0, v16 // (op 1212) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v27, -4, v1 // (op 1264) 
v_sub_u32 v15, v1, v27 // (op 1264) 

v_lshlrev_b32 v27, 4, v15 // (op 1264) 

v_lshrrev_b32 v15, 2, v1 // (op 1264) 
v_lshl_add_u32 v16, v15, 6, v27 // (op 1264) 


s_nop 0
v_readfirstlane_b32 s1, v16 // (op 1264) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v27, -64, v1 // (op 1321) 
v_sub_u32 v15, v1, v27 // (op 1321) 

v_and_b32 v27, 4294967232, v1 // (op 1321) 
v_add_u32 v16, v27, v15 // (op 1321) 


v_and_b32 v15, -8, v1 // (op 1397) 
v_sub_u32 v17, v1, v15 // (op 1397) 

v_lshlrev_b32 v15, 4, v17 // (op 1397) 

v_lshrrev_b32 v17, 3, v1 // (op 1397) 
v_lshl_add_u32 v27, v17, 7, v15 // (op 1397) 


s_nop 0
v_readfirstlane_b32 s18, v27 // (op 1397) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v15, -4, v1 // (op 1436) 
v_sub_u32 v17, v1, v15 // (op 1436) 

v_lshlrev_b32 v15, 4, v17 // (op 1436) 

v_lshrrev_b32 v17, 2, v1 // (op 1436) 
v_lshl_add_u32 v27, v17, 6, v15 // (op 1436) 


s_nop 0
v_readfirstlane_b32 s19, v27 // (op 1436) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v15, -64, v1 // (op 1480) 
v_sub_u32 v17, v1, v15 // (op 1480) 

v_and_b32 v15, 4294967232, v1 // (op 1480) 
v_add_u32 v27, v15, v17 // (op 1480) 


s_mov_b32 s49, 0 // (op 312) call()
s_ashr_i64 s[50:51], s[32:33], 7 // (op 311) 
s_mov_b32 s72, s50 // (op 311) convert

s_and_b32 s50, s72, 4294967294 // (op 311) 

s_cmp_lt_u32 s49, s50 // (op 311) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopBottom_KLoop_311 // (op 311) Condition: Top (jump to RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopBottom_KLoop_311 if false)
RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopTop_KLoop_311:
s_add_u32 s49, s49, 2 // (op 313) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:51], v[28:35], a[0:3], v12, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 84) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[44:51], v[36:43], a[4:7], v12, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 347) 


v_mov_b32 v15, v23 // (op 77) 

buffer_load_ubyte v12, v15, s[64:67], 0 offen offset:8 // (op 77) Load value
s_barrier  // (op 1557) 

s_lshl_b64 s[50:51], s[28:29], 6 // (op 883) 
v_mov_b64 v[52:53], s[50:51] // (op 883) convert


v_mov_b64 v[54:55], v[20:21] // (op 883) 
s_lshl_b64 s[50:51], s[28:29], 7 // (op 883) 
s_lshl_b64 s[72:73], s[50:51], 1 // (op 883) 

v_mov_b32 v17, s72 // (op 883) 
v_mov_b32 v56, s73 // (op 883) 
v_add_co_u32 v58, s[50:51], v54, v17 // (op 883) least significant half
v_addc_co_u32 v59, s[50:51], v55, v56, s[50:51] // (op 883) most significant half




s_mov_b32 m0, s1 // (op 883) reg expression
buffer_load_dwordx4 v58, s[52:55], 0 offen offset: 0 lds // (op 883) Load value direct to lds
v_add_co_u32 v58, s[50:51], v58, v52 // (op 883) least significant half
v_addc_co_u32 v59, s[50:51], 0, v59, s[50:51] // (op 883) most significant half

s_add_u32 m0, m0, 4096 // (op 883) 
buffer_load_dwordx4 v58, s[52:55], 0 offen offset: 0 lds // (op 883) Load value direct to lds




v_mov_b32 v17, v24 // (op 21) 
s_lshl_b32 s50, s38, 3 // (op 21) 
v_add_u32 v52, v17, s50 // (op 21) 


buffer_load_ubyte v26, v52, s[68:71], 0 offen offset: 0 // (op 21) Load value

s_barrier  // (op 1541) 


v_mov_b64 v[52:53], v[18:19] // (op 878) 
s_add_u32 s50, 8192, s0 // (op 878) 

s_mov_b32 m0, s50 // (op 878) reg expression
s_mov_b32 s51, 256 // (op 878) call()
buffer_load_dwordx4 v52, s[20:23], s51 offen offset: 0 lds // (op 878) Load value direct to lds


s_barrier  // (op 1602) 



s_waitcnt vmcnt(6)
ds_write_b8 v27, v14 offset:24832 // (op 240) Store local data 
s_barrier  // (op 1593) 



ds_read_b64_tr_b8 v[28:29], v11 offset:12288 // (op 233) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v11 offset:12800 // (op 233) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v11 offset:16384 // (op 233) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v11 offset:16896 // (op 233) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1599) 


ds_read_u8 v13, v10 offset:24832 // (op 237) Load local data 



ds_read_b64_tr_b8 v[36:37], v11 offset:12320 // (op 352) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v11 offset:12832 // (op 352) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v11 offset:16416 // (op 352) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v11 offset:16928 // (op 352) Transpose load local data 


ds_read_u8 v22, v10 offset:24864 // (op 355) Load local data 
s_barrier  // (op 1657) 
s_waitcnt vmcnt(5)
s_barrier  // (op 1587) 



ds_read_b128 v[44:47], v4 offset:20480 // (op 231) Load local data 
ds_read_b128 v[48:51], v4 offset:20544 // (op 231) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:51], v[28:35], a[0:3], v25, v13 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 238) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[44:51], v[36:43], a[4:7], v25, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 348) 



buffer_load_ubyte v25, v15, s[64:67], 0 offen offset:12 // (op 235) Load value

s_barrier  // (op 1596) 
v_add_u32 v23, 8, v23 // (op 1102) 

s_lshl_b64 s[50:51], s[28:29], 6 // (op 906) 
v_mov_b64 v[56:57], s[50:51] // (op 906) convert


s_lshl_b64 s[50:51], s[28:29], 7 // (op 906) 
s_mul_i32 s73, 3, s51 // (op 906) most significant: low of low * high
s_mul_i32 s72, 3, s50 // (op 906) least significant
s_mul_hi_u32 s74, 3, s50 // (op 906) most significant: high of low * low
s_add_u32 s73, s73, s74 // (op 906) most significant: sum


v_mov_b32 v15, s72 // (op 906) 
v_mov_b32 v58, s73 // (op 906) 
v_add_co_u32 v60, s[50:51], v54, v15 // (op 906) least significant half
v_addc_co_u32 v61, s[50:51], v55, v58, s[50:51] // (op 906) most significant half




s_add_u32 s50, 12288, s19 // (op 906) 
s_nop 0
s_mov_b32 m0, s50 // (op 906) reg expression
buffer_load_dwordx4 v60, s[52:55], 0 offen offset: 0 lds // (op 906) Load value direct to lds
v_add_co_u32 v60, s[72:73], v60, v56 // (op 906) least significant half
v_addc_co_u32 v61, s[72:73], 0, v61, s[72:73] // (op 906) most significant half

s_add_u32 m0, m0, 4096 // (op 906) 
buffer_load_dwordx4 v60, s[52:55], 0 offen offset: 0 lds // (op 906) Load value direct to lds






s_lshl_b32 s50, s38, 2 // (op 242) 
s_mul_i32 s51, 3, s50 // (op 242) 

v_add_u32 v15, v17, s51 // (op 242) 


s_lshl_b64 s[50:51], s[28:29], 7 // (op 1038) 
s_lshl_b64 s[72:73], s[50:51], 1 // (op 1038) 

v_mov_b32 v54, s72 // (op 1038) 
v_mov_b32 v55, s73 // (op 1038) 
v_add_co_u32 v20, s[50:51], v20, v54 // (op 1038) least significant half
v_addc_co_u32 v21, s[50:51], v21, v55, s[50:51] // (op 1038) most significant half




buffer_load_ubyte v14, v15, s[68:71], 0 offen offset: 0 // (op 242) Load value


s_barrier  // (op 1590) 
s_lshl_b32 s50, s38, 3 // (op 1166) 


v_add_u32 v24, v24, s50 // (op 1166) 

s_add_u32 s50, 20480, s18 // (op 901) 

s_mov_b32 m0, s50 // (op 901) reg expression
s_mov_b32 s51, 384 // (op 901) call()
buffer_load_dwordx4 v52, s[20:23], s51 offen offset: 0 lds // (op 901) Load value direct to lds



s_barrier  // (op 1573) 
v_add_co_u32 v18, vcc, 256, v18 // (op 974) least significant half



v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 974) most significant half
s_waitcnt vmcnt(6)
ds_write_b8 v16, v26 offset:24576 // (op 26) Store local data 
s_barrier  // (op 1563) 



ds_read_b64_tr_b8 v[28:29], v7 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[30:31], v7 offset:512 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[32:33], v7 offset:4096 // (op 72) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v7 offset:4608 // (op 72) Transpose load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1580) 


ds_read_u8 v13, v0 offset:24576 // (op 81) Load local data 



ds_read_b64_tr_b8 v[36:37], v7 offset:32 // (op 343) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v7 offset:544 // (op 343) Transpose load local data 
ds_read_b64_tr_b8 v[40:41], v7 offset:4128 // (op 343) Transpose load local data 
ds_read_b64_tr_b8 v[42:43], v7 offset:4640 // (op 343) Transpose load local data 


ds_read_u8 v22, v0 offset:24608 // (op 346) Load local data 
s_waitcnt vmcnt(5)
s_barrier  // (op 1547) 



ds_read_b128 v[44:47], v3 offset:8192 // (op 67) Load local data 
ds_read_b128 v[48:51], v3 offset:8256 // (op 67) Load local data 
s_ashr_i64 s[50:51], s[32:33], 7 // (op 311) 
s_mov_b32 s72, s50 // (op 311) convert

s_and_b32 s50, s72, 4294967294 // (op 311) 

s_cmp_lt_u32 s49, s50 // (op 311) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopTop_KLoop_311 // (op 311) Condition: Bottom (jump to RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopTop_KLoop_311 if true)
RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopBottom_KLoop_311:













s_barrier  // (op 1641) 
s_mul_hi_i32 s0, s16, s11 // (op 936) 
v_lshrrev_b32 v12, 3, v1 // (op 936) 
s_add_i32 s1, s0, s16 // (op 936) 

v_and_b32 v15, -8, v1 // (op 936) 
s_ashr_i32 s0, s1, 31 // (op 936) 
v_sub_u32 v16, v1, v15 // (op 936) 

s_cmp_eq_i32 s11, 0 // (op 936) 
s_mov_b32 s18, scc // (op 936) 
v_lshlrev_b32 v15, 4, v16 // (op 936) 

s_cmp_eq_i32 s18, 0 // (op 936) 
s_cselect_b32 s19, 0, -1 // (op 936) 

s_and_b32 s18, s12, 31 // (op 936) 
s_lshl_b32 s49, 1, s18 // (op 936) 
s_add_i32 s50, s49, s19 // (op 936) 


s_and_b32 s19, s0, s50 // (op 936) 


s_add_i32 s0, s1, s19 // (op 936) 


s_ashr_i32 s1, s0, s18 // (op 936) 


s_lshl_b32 s0, s12, 24 // (op 936) 
s_ashr_i32 s18, s0, 31 // (op 936) 

s_xor_b32 s0, s1, s18 // (op 936) 

s_sub_i32 s1, s0, s18 // (op 936) 


s_mul_hi_i32 s0, s1, s13 // (op 936) 
s_add_i32 s18, s0, s1 // (op 936) 

s_ashr_i32 s0, s18, 31 // (op 936) 
s_cmp_eq_i32 s13, 0 // (op 936) 
s_mov_b32 s19, scc // (op 936) 
s_cmp_eq_i32 s19, 0 // (op 936) 
s_cselect_b32 s49, 0, -1 // (op 936) 

s_and_b32 s19, s14, 31 // (op 936) 
s_lshl_b32 s50, 1, s19 // (op 936) 
s_add_i32 s51, s50, s49 // (op 936) 


s_and_b32 s49, s0, s51 // (op 936) 


s_add_i32 s0, s18, s49 // (op 936) 


s_ashr_i32 s18, s0, s19 // (op 936) 


s_lshl_b32 s0, s14, 24 // (op 936) 
s_ashr_i32 s19, s0, 31 // (op 936) 

s_xor_b32 s0, s18, s19 // (op 936) 

s_sub_i32 s18, s0, s19 // (op 936) 


s_mul_i32 s0, s1, s44 // (op 936) 

s_sub_u32 s1, s16, s0 // (op 936) 

s_mul_i32 s0, s18, s44 // (op 936) 
s_add_u32 s0, s0, s1 // (op 936) 


s_mul_hi_i32 s1, s16, s10 // (op 936) 
s_add_i32 s18, s1, s16 // (op 936) 

s_ashr_i32 s1, s18, 31 // (op 936) 
s_cmp_eq_i32 s10, 0 // (op 936) 
s_mov_b32 s19, scc // (op 936) 
s_cmp_eq_i32 s19, 0 // (op 936) 
s_cselect_b32 s49, 0, -1 // (op 936) 

s_and_b32 s19, s47, 31 // (op 936) 
s_lshl_b32 s50, 1, s19 // (op 936) 
s_add_i32 s51, s50, s49 // (op 936) 


s_and_b32 s49, s1, s51 // (op 936) 


s_add_i32 s1, s18, s49 // (op 936) 


s_ashr_i32 s18, s1, s19 // (op 936) 


s_lshl_b32 s1, s47, 24 // (op 936) 
s_ashr_i32 s19, s1, 31 // (op 936) 

s_xor_b32 s1, s18, s19 // (op 936) 

s_sub_i32 s18, s1, s19 // (op 936) 


s_add_u32 s50, s6, 64 // (op 936) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 936) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 936) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 936) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 936) 

s_mul_i32 s1, s44, s50 // (op 936) 

s_mul_i32 s19, s48, s1 // (op 936) 

s_mul_i32 s1, s18, s19 // (op 936) 

s_sub_u32 s19, s16, s1 // (op 936) 

s_mov_b32 s1, s19 // (op 936) 
s_mul_hi_i32 s49, s1, s15 // (op 936) 
s_add_i32 s50, s49, s1 // (op 936) 


s_ashr_i32 s1, s50, 31 // (op 936) 
s_cmp_eq_i32 s15, 0 // (op 936) 
s_mov_b32 s49, scc // (op 936) 
s_cmp_eq_i32 s49, 0 // (op 936) 
s_cselect_b32 s51, 0, -1 // (op 936) 

s_and_b32 s49, s46, 31 // (op 936) 
s_lshl_b32 s72, 1, s49 // (op 936) 
s_add_i32 s73, s72, s51 // (op 936) 


s_and_b32 s51, s1, s73 // (op 936) 


s_add_i32 s1, s50, s51 // (op 936) 


s_ashr_i32 s50, s1, s49 // (op 936) 


s_lshl_b32 s1, s46, 24 // (op 936) 
s_ashr_i32 s49, s1, 31 // (op 936) 

s_xor_b32 s1, s50, s49 // (op 936) 

s_sub_i32 s50, s1, s49 // (op 936) 


s_mul_i32 s1, s45, s44 // (op 936) 
s_add_u32 s72, s8, 32 // (op 936) Start of Int64 add, locking SCC
s_addc_u32 s73, s9, 0 // (op 936) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 936) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 936) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 5 // (op 936) 

s_sub_i32 s49, s72, s1 // (op 936) 

s_mul_i32 s51, s50, s49 // (op 936) 


s_sub_u32 s49, s19, s51 // (op 936) 


s_add_u32 s19, s49, s1 // (op 936) 


s_cmp_eq_i32 s18, 0 // (op 936) 
s_mov_b32 s1, scc // (op 936) 

s_cmp_eq_i32 s1, 0 // (op 936) 
s_cselect_b32 s18, s19, s0 // (op 936) 



v_lshl_add_u32 v16, s18, 5, v12 // (op 936) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 936) 
s_lshr_b32 s18, s0, 1 // (op 936) 

v_lshl_add_u32 v12, s18, 8, v15 // (op 936) 


v_mov_b32 v27, 0 // (op 936) convert
v_mov_b32 v26, v12 // (op 936) convert

v_mul_lo_u32 v21, v16, s25 // (op 936) most significant: low of low * high
v_mul_hi_u32 v12, v16, s24 // (op 936) most significant: high of low * low
v_mul_lo_u32 v20, v16, s24 // (op 936) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 936) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v26 // (op 936) least significant half
v_addc_co_u32 v21, s[0:1], v21, v27, s[0:1] // (op 936) most significant half



s_mul_i32 s1, s24, s9 // (op 938) most significant: low of low * high
s_mul_i32 s18, s25, s8 // (op 938) most significant: low of high * low
s_mul_i32 s0, s24, s8 // (op 938) least significant
s_mul_hi_u32 s19, s24, s8 // (op 938) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 938) most significant: sum
s_add_u32 s1, s1, s19 // (op 938) most significant: sum


s_mov_b32 s20, s26 // (op 938) reg expression
s_mov_b32 s21, s27 // (op 938) reg expression
s_mov_b32 s22, s0 // (op 938) reg expression
s_mov_b32 s23, 131072 // (op 938) call()

s_mul_hi_i32 s0, s16, s11 // (op 1004) 
v_and_b32 v12, -4, v1 // (op 1004) 
s_add_i32 s1, s0, s16 // (op 1004) 

v_sub_u32 v15, v1, v12 // (op 1004) 

s_ashr_i32 s0, s1, 31 // (op 1004) 
v_lshlrev_b32 v12, 4, v15 // (op 1004) 

s_cmp_eq_i32 s11, 0 // (op 1004) 
s_mov_b32 s18, scc // (op 1004) 
v_lshrrev_b32 v15, 2, v1 // (op 1004) 
s_cmp_eq_i32 s18, 0 // (op 1004) 
s_cselect_b32 s19, 0, -1 // (op 1004) 

s_and_b32 s18, s12, 31 // (op 1004) 
s_lshl_b32 s49, 1, s18 // (op 1004) 
s_add_i32 s50, s49, s19 // (op 1004) 


s_and_b32 s19, s0, s50 // (op 1004) 


s_add_i32 s0, s1, s19 // (op 1004) 


s_ashr_i32 s1, s0, s18 // (op 1004) 


s_lshl_b32 s0, s12, 24 // (op 1004) 
s_ashr_i32 s18, s0, 31 // (op 1004) 

s_xor_b32 s0, s1, s18 // (op 1004) 

s_sub_i32 s1, s0, s18 // (op 1004) 


s_mul_hi_i32 s0, s1, s13 // (op 1004) 
s_add_i32 s18, s0, s1 // (op 1004) 

s_ashr_i32 s0, s18, 31 // (op 1004) 
s_cmp_eq_i32 s13, 0 // (op 1004) 
s_mov_b32 s19, scc // (op 1004) 
s_cmp_eq_i32 s19, 0 // (op 1004) 
s_cselect_b32 s49, 0, -1 // (op 1004) 

s_and_b32 s19, s14, 31 // (op 1004) 
s_lshl_b32 s50, 1, s19 // (op 1004) 
s_add_i32 s51, s50, s49 // (op 1004) 


s_and_b32 s49, s0, s51 // (op 1004) 


s_add_i32 s0, s18, s49 // (op 1004) 


s_ashr_i32 s18, s0, s19 // (op 1004) 


s_lshl_b32 s0, s14, 24 // (op 1004) 
s_ashr_i32 s19, s0, 31 // (op 1004) 

s_xor_b32 s0, s18, s19 // (op 1004) 

s_sub_i32 s18, s0, s19 // (op 1004) 


s_add_u32 s50, s6, 64 // (op 1004) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1004) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1004) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1004) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1004) 

s_mul_i32 s0, s18, s50 // (op 1004) 

s_sub_i32 s49, s1, s0 // (op 1004) 


s_mul_hi_i32 s0, s16, s10 // (op 1004) 
s_add_i32 s1, s0, s16 // (op 1004) 

s_ashr_i32 s0, s1, 31 // (op 1004) 
s_cmp_eq_i32 s10, 0 // (op 1004) 
s_mov_b32 s18, scc // (op 1004) 
s_cmp_eq_i32 s18, 0 // (op 1004) 
s_cselect_b32 s19, 0, -1 // (op 1004) 

s_and_b32 s18, s47, 31 // (op 1004) 
s_lshl_b32 s72, 1, s18 // (op 1004) 
s_add_i32 s73, s72, s19 // (op 1004) 


s_and_b32 s19, s0, s73 // (op 1004) 


s_add_i32 s0, s1, s19 // (op 1004) 


s_ashr_i32 s1, s0, s18 // (op 1004) 


s_lshl_b32 s0, s47, 24 // (op 1004) 
s_ashr_i32 s18, s0, 31 // (op 1004) 

s_xor_b32 s0, s1, s18 // (op 1004) 

s_sub_i32 s1, s0, s18 // (op 1004) 


s_mul_i32 s0, s44, s50 // (op 1004) 

s_mul_i32 s18, s48, s0 // (op 1004) 

s_mul_i32 s0, s1, s18 // (op 1004) 

s_sub_u32 s18, s16, s0 // (op 1004) 

s_mul_hi_i32 s0, s18, s15 // (op 1004) 
s_add_i32 s19, s0, s18 // (op 1004) 


s_ashr_i32 s0, s19, 31 // (op 1004) 
s_cmp_eq_i32 s15, 0 // (op 1004) 
s_mov_b32 s18, scc // (op 1004) 
s_cmp_eq_i32 s18, 0 // (op 1004) 
s_cselect_b32 s50, 0, -1 // (op 1004) 

s_and_b32 s18, s46, 31 // (op 1004) 
s_lshl_b32 s51, 1, s18 // (op 1004) 
s_add_i32 s72, s51, s50 // (op 1004) 


s_and_b32 s50, s0, s72 // (op 1004) 


s_add_i32 s0, s19, s50 // (op 1004) 


s_ashr_i32 s19, s0, s18 // (op 1004) 


s_lshl_b32 s0, s46, 24 // (op 1004) 
s_ashr_i32 s18, s0, 31 // (op 1004) 

s_xor_b32 s0, s19, s18 // (op 1004) 

s_sub_i32 s19, s0, s18 // (op 1004) 


s_cmp_eq_i32 s1, 0 // (op 1004) 
s_mov_b32 s0, scc // (op 1004) 

s_cmp_eq_i32 s0, 0 // (op 1004) 
s_cselect_b32 s1, s19, s49 // (op 1004) 



v_lshl_add_u32 v16, s1, 6, v12 // (op 1004) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 1004) 
v_mov_b32 v19, 0 // (op 1004) convert
v_mov_b32 v18, v16 // (op 1004) convert

s_lshr_b32 s18, s0, 1 // (op 1004) 

v_lshl_add_u32 v12, s18, 8, v15 // (op 1004) 


v_mul_lo_u32 v17, v12, s29 // (op 1004) most significant: low of low * high
v_mul_hi_u32 v15, v12, s28 // (op 1004) most significant: high of low * low
v_mul_lo_u32 v16, v12, s28 // (op 1004) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 1004) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v18 // (op 1004) least significant half
v_addc_co_u32 v17, s[0:1], v17, v19, s[0:1] // (op 1004) most significant half



s_mul_i32 s1, s28, s33 // (op 1006) most significant: low of low * high
s_mul_i32 s18, s29, s32 // (op 1006) most significant: low of high * low
s_mul_i32 s0, s28, s32 // (op 1006) least significant
s_mul_hi_u32 s19, s28, s32 // (op 1006) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1006) most significant: sum
s_add_u32 s1, s1, s19 // (op 1006) most significant: sum


s_mov_b32 s52, s30 // (op 1006) reg expression
s_mov_b32 s53, s31 // (op 1006) reg expression
s_mov_b32 s54, s0 // (op 1006) reg expression
s_mov_b32 s55, 131072 // (op 1006) call()

s_mul_hi_i32 s0, s16, s11 // (op 1068) 
v_lshrrev_b32 v12, 7, v1 // (op 1068) 
s_add_i32 s1, s0, s16 // (op 1068) 

v_and_b32 v15, 15, v1 // (op 1068) 
s_ashr_i32 s0, s1, 31 // (op 1068) 
v_lshl_add_u32 v18, v12, 4, v15 // (op 1068) 


s_cmp_eq_i32 s11, 0 // (op 1068) 
s_mov_b32 s18, scc // (op 1068) 
v_and_b32 v12, 63, v1 // (op 1068) 
s_cmp_eq_i32 s18, 0 // (op 1068) 
s_cselect_b32 s19, 0, -1 // (op 1068) 

v_lshrrev_b32 v15, 4, v12 // (op 1068) 

s_and_b32 s18, s12, 31 // (op 1068) 
s_lshl_b32 s49, 1, s18 // (op 1068) 
s_add_i32 s50, s49, s19 // (op 1068) 


s_and_b32 s19, s0, s50 // (op 1068) 


s_add_i32 s0, s1, s19 // (op 1068) 


s_ashr_i32 s1, s0, s18 // (op 1068) 


s_lshl_b32 s0, s12, 24 // (op 1068) 
s_ashr_i32 s18, s0, 31 // (op 1068) 

s_xor_b32 s0, s1, s18 // (op 1068) 

s_sub_i32 s1, s0, s18 // (op 1068) 


s_mul_hi_i32 s0, s1, s13 // (op 1068) 
s_add_i32 s18, s0, s1 // (op 1068) 

s_ashr_i32 s0, s18, 31 // (op 1068) 
s_cmp_eq_i32 s13, 0 // (op 1068) 
s_mov_b32 s19, scc // (op 1068) 
s_cmp_eq_i32 s19, 0 // (op 1068) 
s_cselect_b32 s49, 0, -1 // (op 1068) 

s_and_b32 s19, s14, 31 // (op 1068) 
s_lshl_b32 s50, 1, s19 // (op 1068) 
s_add_i32 s51, s50, s49 // (op 1068) 


s_and_b32 s49, s0, s51 // (op 1068) 


s_add_i32 s0, s18, s49 // (op 1068) 


s_ashr_i32 s18, s0, s19 // (op 1068) 


s_lshl_b32 s0, s14, 24 // (op 1068) 
s_ashr_i32 s19, s0, 31 // (op 1068) 

s_xor_b32 s0, s18, s19 // (op 1068) 

s_sub_i32 s18, s0, s19 // (op 1068) 


s_mul_i32 s0, s1, s44 // (op 1068) 

s_sub_u32 s1, s16, s0 // (op 1068) 

s_mul_i32 s0, s18, s44 // (op 1068) 
s_add_u32 s0, s0, s1 // (op 1068) 


s_mul_hi_i32 s1, s16, s10 // (op 1068) 
s_add_i32 s18, s1, s16 // (op 1068) 

s_ashr_i32 s1, s18, 31 // (op 1068) 
s_cmp_eq_i32 s10, 0 // (op 1068) 
s_mov_b32 s19, scc // (op 1068) 
s_cmp_eq_i32 s19, 0 // (op 1068) 
s_cselect_b32 s49, 0, -1 // (op 1068) 

s_and_b32 s19, s47, 31 // (op 1068) 
s_lshl_b32 s50, 1, s19 // (op 1068) 
s_add_i32 s51, s50, s49 // (op 1068) 


s_and_b32 s49, s1, s51 // (op 1068) 


s_add_i32 s1, s18, s49 // (op 1068) 


s_ashr_i32 s18, s1, s19 // (op 1068) 


s_lshl_b32 s1, s47, 24 // (op 1068) 
s_ashr_i32 s19, s1, 31 // (op 1068) 

s_xor_b32 s1, s18, s19 // (op 1068) 

s_sub_i32 s18, s1, s19 // (op 1068) 


s_add_u32 s50, s6, 64 // (op 1068) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1068) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1068) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1068) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1068) 

s_mul_i32 s1, s44, s50 // (op 1068) 

s_mul_i32 s19, s48, s1 // (op 1068) 

s_mul_i32 s1, s18, s19 // (op 1068) 

s_sub_u32 s19, s16, s1 // (op 1068) 

s_mov_b32 s1, s19 // (op 1068) 
s_mul_hi_i32 s49, s1, s15 // (op 1068) 
s_add_i32 s50, s49, s1 // (op 1068) 


s_ashr_i32 s1, s50, 31 // (op 1068) 
s_cmp_eq_i32 s15, 0 // (op 1068) 
s_mov_b32 s49, scc // (op 1068) 
s_cmp_eq_i32 s49, 0 // (op 1068) 
s_cselect_b32 s51, 0, -1 // (op 1068) 

s_and_b32 s49, s46, 31 // (op 1068) 
s_lshl_b32 s72, 1, s49 // (op 1068) 
s_add_i32 s73, s72, s51 // (op 1068) 


s_and_b32 s51, s1, s73 // (op 1068) 


s_add_i32 s1, s50, s51 // (op 1068) 


s_ashr_i32 s50, s1, s49 // (op 1068) 


s_lshl_b32 s1, s46, 24 // (op 1068) 
s_ashr_i32 s49, s1, 31 // (op 1068) 

s_xor_b32 s1, s50, s49 // (op 1068) 

s_sub_i32 s50, s1, s49 // (op 1068) 


s_mul_i32 s1, s45, s44 // (op 1068) 
s_add_u32 s72, s8, 32 // (op 1068) Start of Int64 add, locking SCC
s_addc_u32 s73, s9, 0 // (op 1068) End of Int64 add, unlocking SCC
s_sub_u32 s74, s72, 1 // (op 1068) least significant half; sets scc
s_subb_u32 s75, s73, 0 // (op 1068) most significant half; uses scc

s_ashr_i64 s[72:73], s[74:75], 5 // (op 1068) 

s_sub_i32 s49, s72, s1 // (op 1068) 

s_mul_i32 s51, s50, s49 // (op 1068) 


s_sub_u32 s49, s19, s51 // (op 1068) 


s_add_u32 s19, s49, s1 // (op 1068) 


s_cmp_eq_i32 s18, 0 // (op 1068) 
s_mov_b32 s1, scc // (op 1068) 

s_cmp_eq_i32 s1, 0 // (op 1068) 
s_cselect_b32 s18, s19, s0 // (op 1068) 



v_lshl_add_u32 v12, s18, 5, v18 // (op 1068) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 1068) 
s_lshr_b32 s18, s0, 1 // (op 1068) 

v_lshl_add_u32 v18, s18, 3, v15 // (op 1068) 


v_mul_lo_u32 v15, v12, s34 // (op 1068) 
v_add_u32 v15, v15, v18 // (op 1068) 


s_mul_i32 s1, s34, s9 // (op 1070) most significant: low of low * high
s_mul_i32 s18, s35, s8 // (op 1070) most significant: low of high * low
s_mul_i32 s0, s34, s8 // (op 1070) least significant
s_mul_hi_u32 s19, s34, s8 // (op 1070) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1070) most significant: sum
s_add_u32 s1, s1, s19 // (op 1070) most significant: sum


s_mov_b32 s64, s36 // (op 1070) reg expression
s_mov_b32 s65, s37 // (op 1070) reg expression
s_mov_b32 s66, s0 // (op 1070) reg expression
s_mov_b32 s67, 131072 // (op 1070) call()

s_mul_hi_i32 s0, s16, s11 // (op 1132) 
v_and_b32 v12, -64, v1 // (op 1132) 
s_add_i32 s1, s0, s16 // (op 1132) 

v_sub_u32 v18, v1, v12 // (op 1132) 

s_ashr_i32 s0, s1, 31 // (op 1132) 
v_lshrrev_b32 v12, 6, v1 // (op 1132) 
s_cmp_eq_i32 s11, 0 // (op 1132) 
s_mov_b32 s18, scc // (op 1132) 
s_cmp_eq_i32 s18, 0 // (op 1132) 
s_cselect_b32 s19, 0, -1 // (op 1132) 

s_and_b32 s18, s12, 31 // (op 1132) 
s_lshl_b32 s49, 1, s18 // (op 1132) 
s_add_i32 s50, s49, s19 // (op 1132) 


s_and_b32 s19, s0, s50 // (op 1132) 


s_add_i32 s0, s1, s19 // (op 1132) 


s_ashr_i32 s1, s0, s18 // (op 1132) 


s_lshl_b32 s0, s12, 24 // (op 1132) 
s_ashr_i32 s18, s0, 31 // (op 1132) 

s_xor_b32 s0, s1, s18 // (op 1132) 

s_sub_i32 s1, s0, s18 // (op 1132) 


s_mul_hi_i32 s0, s1, s13 // (op 1132) 
s_add_i32 s18, s0, s1 // (op 1132) 

s_ashr_i32 s0, s18, 31 // (op 1132) 
s_cmp_eq_i32 s13, 0 // (op 1132) 
s_mov_b32 s19, scc // (op 1132) 
s_cmp_eq_i32 s19, 0 // (op 1132) 
s_cselect_b32 s49, 0, -1 // (op 1132) 

s_and_b32 s19, s14, 31 // (op 1132) 
s_lshl_b32 s50, 1, s19 // (op 1132) 
s_add_i32 s51, s50, s49 // (op 1132) 


s_and_b32 s49, s0, s51 // (op 1132) 


s_add_i32 s0, s18, s49 // (op 1132) 


s_ashr_i32 s18, s0, s19 // (op 1132) 


s_lshl_b32 s0, s14, 24 // (op 1132) 
s_ashr_i32 s19, s0, 31 // (op 1132) 

s_xor_b32 s0, s18, s19 // (op 1132) 

s_sub_i32 s18, s0, s19 // (op 1132) 


s_add_u32 s50, s6, 64 // (op 1132) Start of Int64 add, locking SCC
s_addc_u32 s51, s7, 0 // (op 1132) End of Int64 add, unlocking SCC
s_sub_u32 s72, s50, 1 // (op 1132) least significant half; sets scc
s_subb_u32 s73, s51, 0 // (op 1132) most significant half; uses scc

s_ashr_i64 s[50:51], s[72:73], 6 // (op 1132) 

s_mul_i32 s0, s18, s50 // (op 1132) 

s_sub_i32 s49, s1, s0 // (op 1132) 


s_mul_hi_i32 s0, s16, s10 // (op 1132) 
s_add_i32 s1, s0, s16 // (op 1132) 

s_ashr_i32 s0, s1, 31 // (op 1132) 
s_cmp_eq_i32 s10, 0 // (op 1132) 
s_mov_b32 s18, scc // (op 1132) 
s_cmp_eq_i32 s18, 0 // (op 1132) 
s_cselect_b32 s19, 0, -1 // (op 1132) 

s_and_b32 s18, s47, 31 // (op 1132) 
s_lshl_b32 s72, 1, s18 // (op 1132) 
s_add_i32 s73, s72, s19 // (op 1132) 


s_and_b32 s19, s0, s73 // (op 1132) 


s_add_i32 s0, s1, s19 // (op 1132) 


s_ashr_i32 s1, s0, s18 // (op 1132) 


s_lshl_b32 s0, s47, 24 // (op 1132) 
s_ashr_i32 s18, s0, 31 // (op 1132) 

s_xor_b32 s0, s1, s18 // (op 1132) 

s_sub_i32 s1, s0, s18 // (op 1132) 


s_mul_i32 s0, s44, s50 // (op 1132) 

s_mul_i32 s18, s48, s0 // (op 1132) 

s_mul_i32 s0, s1, s18 // (op 1132) 

s_sub_u32 s18, s16, s0 // (op 1132) 

s_mul_hi_i32 s0, s18, s15 // (op 1132) 
s_add_i32 s19, s0, s18 // (op 1132) 


s_ashr_i32 s0, s19, 31 // (op 1132) 
s_cmp_eq_i32 s15, 0 // (op 1132) 
s_mov_b32 s18, scc // (op 1132) 
s_cmp_eq_i32 s18, 0 // (op 1132) 
s_cselect_b32 s50, 0, -1 // (op 1132) 

s_and_b32 s18, s46, 31 // (op 1132) 
s_lshl_b32 s51, 1, s18 // (op 1132) 
s_add_i32 s72, s51, s50 // (op 1132) 


s_and_b32 s50, s0, s72 // (op 1132) 


s_add_i32 s0, s19, s50 // (op 1132) 


s_ashr_i32 s19, s0, s18 // (op 1132) 


s_lshl_b32 s0, s46, 24 // (op 1132) 
s_ashr_i32 s18, s0, 31 // (op 1132) 

s_xor_b32 s0, s19, s18 // (op 1132) 

s_sub_i32 s19, s0, s18 // (op 1132) 


s_cmp_eq_i32 s1, 0 // (op 1132) 
s_mov_b32 s0, scc // (op 1132) 

s_cmp_eq_i32 s0, 0 // (op 1132) 
s_cselect_b32 s1, s19, s49 // (op 1132) 



v_lshl_add_u32 v19, s1, 6, v18 // (op 1132) 


s_ashr_i64 s[0:1], s[32:33], 7 // (op 1132) 
s_lshr_b32 s18, s0, 1 // (op 1132) 

v_lshl_add_u32 v18, s18, 3, v12 // (op 1132) 


v_mul_lo_u32 v12, v18, s38 // (op 1132) 
v_add_u32 v12, v12, v19 // (op 1132) 


s_mul_i32 s1, s38, s43 // (op 1134) most significant: low of low * high
s_mul_i32 s18, s39, s42 // (op 1134) most significant: low of high * low
s_mul_i32 s0, s38, s42 // (op 1134) least significant
s_mul_hi_u32 s19, s38, s42 // (op 1134) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1134) most significant: sum
s_add_u32 s1, s1, s19 // (op 1134) most significant: sum


s_mov_b32 s68, s40 // (op 1134) reg expression
s_mov_b32 s69, s41 // (op 1134) reg expression
s_mov_b32 s70, s0 // (op 1134) reg expression
s_mov_b32 s71, 131072 // (op 1134) call()

v_and_b32 v18, -8, v1 // (op 1225) 
v_sub_u32 v19, v1, v18 // (op 1225) 

v_lshlrev_b32 v18, 4, v19 // (op 1225) 

v_lshrrev_b32 v19, 3, v1 // (op 1225) 
v_lshl_add_u32 v23, v19, 7, v18 // (op 1225) 


s_nop 0
v_readfirstlane_b32 s0, v23 // (op 1225) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, -4, v1 // (op 1277) 
v_sub_u32 v19, v1, v18 // (op 1277) 

v_lshlrev_b32 v18, 4, v19 // (op 1277) 

v_lshrrev_b32 v19, 2, v1 // (op 1277) 
v_lshl_add_u32 v23, v19, 6, v18 // (op 1277) 


s_nop 0
v_readfirstlane_b32 s1, v23 // (op 1277) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v18, -64, v1 // (op 1334) 
v_sub_u32 v19, v1, v18 // (op 1334) 

v_and_b32 v18, 4294967232, v1 // (op 1334) 
v_add_u32 v23, v18, v19 // (op 1334) 


s_ashr_i64 s[18:19], s[32:33], 7 // (op 317) 
s_and_b32 s49, s18, 4294967294 // (op 317) 

s_ashr_i64 s[18:19], s[32:33], 7 // (op 316) 
s_mov_b32 s50, s18 // (op 316) convert

s_cmp_lt_u32 s49, s50 // (op 316) 








s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_316 // (op 316) Condition: Top (jump to RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_316 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopTop_KLoopTail_316:
s_barrier  // (op 1544) 
s_add_u32 s49, s49, 1 // (op 318) 


v_mov_b64 v[18:19], v[20:21] // (op 888) 
s_add_u32 s18, 8192, s0 // (op 888) 

s_mov_b32 m0, s18 // (op 888) reg expression
buffer_load_dwordx4 v18, s[20:23], 0 offen offset: 0 lds // (op 888) Load value direct to lds


v_add_co_u32 v20, vcc, 128, v20 // (op 926) least significant half
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 926) most significant half
s_barrier  // (op 1560) 

s_lshl_b64 s[18:19], s[28:29], 6 // (op 894) 
v_mov_b64 v[18:19], s[18:19] // (op 894) convert


v_mov_b64 v[24:25], v[16:17] // (op 894) 
v_mov_b64 v[26:27], v[24:25] // (op 894) reg expression
s_mov_b32 m0, s1 // (op 894) reg expression
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 894) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v18 // (op 894) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 894) most significant half

s_add_u32 m0, m0, 4096 // (op 894) 
buffer_load_dwordx4 v26, s[52:55], 0 offen offset: 0 lds // (op 894) Load value direct to lds



s_lshl_b64 s[18:19], s[28:29], 7 // (op 996) 
v_mov_b32 v22, s18 // (op 996) 
v_mov_b32 v13, s19 // (op 996) 
v_add_co_u32 v16, s[50:51], v16, v22 // (op 996) least significant half
v_addc_co_u32 v17, s[50:51], v17, v13, s[50:51] // (op 996) most significant half






v_mov_b32 v22, v12 // (op 184) 
v_mov_b32 v13, v22 // (op 184) reg expression

buffer_load_ubyte v14, v13, s[68:71], 0 offen offset: 0 // (op 184) Load value


s_lshl_b32 s18, s38, 2 // (op 1124) 
v_add_u32 v12, v12, s18 // (op 1124) 

s_barrier  // (op 1576) 
s_waitcnt vmcnt(3)
s_barrier  // (op 1550) 






ds_read_b128 v[24:27], v3 offset:8192 // (op 173) Load local data 
ds_read_b128 v[28:31], v3 offset:8256 // (op 173) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v23, v14 offset:24576 // (op 182) Store local data 

s_barrier  // (op 1566) 



ds_read_b64_tr_b8 v[32:33], v7 // (op 175) Transpose load local data 
ds_read_b64_tr_b8 v[34:35], v7 offset:512 // (op 175) Transpose load local data 
ds_read_b64_tr_b8 v[36:37], v7 offset:4096 // (op 175) Transpose load local data 
ds_read_b64_tr_b8 v[38:39], v7 offset:4608 // (op 175) Transpose load local data 


v_mov_b32 v22, v15 // (op 177) 

buffer_load_ubyte v13, v22, s[64:67], 0 offen offset: 0 // (op 177) Load value

v_add_u32 v15, 4, v15 // (op 1060) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1583) 


ds_read_u8 v14, v0 offset:24576 // (op 179) Load local data 



ds_read_b64_tr_b8 v[40:41], v7 offset:32 // (op 324) Transpose load local data 
s_waitcnt vmcnt(0) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[24:31], v[32:39], a[0:3], v13, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 180) 
ds_read_b64_tr_b8 v[42:43], v7 offset:544 // (op 324) Transpose load local data 


ds_read_b64_tr_b8 v[44:45], v7 offset:4128 // (op 324) Transpose load local data 
ds_read_b64_tr_b8 v[46:47], v7 offset:4640 // (op 324) Transpose load local data 


ds_read_u8 v14, v0 offset:24608 // (op 327) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[24:31], v[40:47], a[4:7], v13, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 328) 




s_ashr_i64 s[18:19], s[32:33], 7 // (op 316) 
s_mov_b32 s50, s18 // (op 316) convert

s_cmp_lt_u32 s49, s50 // (op 316) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopTop_KLoopTail_316 // (op 316) Condition: Bottom (jump to RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopTop_KLoopTail_316 if true)
RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_316:






















s_barrier  // (op 1645) 



















v_cmp_eq_f32 vcc, 0.00000, v5 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ConditionalFalse_0_beta_125
v_accvgpr_read v10, a0 // (op 46) 
v_accvgpr_read v11, a1 // (op 46) 
v_accvgpr_read v12, a2 // (op 46) 
v_accvgpr_read v13, a3 // (op 46) 
v_mul_f32 v14, v8, v10 // (op 46) 
v_mul_f32 v15, v8, v11 // (op 46) 
v_mul_f32 v16, v8, v12 // (op 46) 
v_mul_f32 v17, v8, v13 // (op 46) 

v_cvt_f16_f32 v10, v14 // (op 52) 
v_cvt_f16_f32 v11, v15 // (op 52) 
v_cvt_f16_f32 v12, v16 // (op 52) 
v_cvt_f16_f32 v13, v17 // (op 52) 


v_pack_b32_f16 v14, v10, v11 // (op 54) 
v_pack_b32_f16 v15, v12, v13 // (op 54) 

v_mov_b32 v0, v9 // (op 54) reg expression

buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 54) Store value



v_accvgpr_read v10, a4 // (op 508) 
v_accvgpr_read v11, a5 // (op 508) 
v_accvgpr_read v12, a6 // (op 508) 
v_accvgpr_read v13, a7 // (op 508) 
v_mul_f32 v14, v8, v10 // (op 508) 
v_mul_f32 v15, v8, v11 // (op 508) 
v_mul_f32 v16, v8, v12 // (op 508) 
v_mul_f32 v17, v8, v13 // (op 508) 

v_cvt_f16_f32 v10, v14 // (op 510) 
v_cvt_f16_f32 v11, v15 // (op 510) 
v_cvt_f16_f32 v12, v16 // (op 510) 
v_cvt_f16_f32 v13, v17 // (op 510) 


v_pack_b32_f16 v14, v10, v11 // (op 511) 
v_pack_b32_f16 v15, v12, v13 // (op 511) 

s_lshl_b32 s0, s4, 6 // (op 511) 
v_add_u32 v0, v9, s0 // (op 511) 


buffer_store_dwordx2 v[14:15], v0, s[60:63], 0 offen // (op 511) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ConditionalBottom_0_beta_125
RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ConditionalFalse_0_beta_125:


v_mov_b32 v0, v6 // (op 138) reg expression

buffer_load_dwordx2 v[10:11], v0, s[56:59], 0 offen // (op 138) Load value

v_accvgpr_read v12, a0 // (op 134) 
v_accvgpr_read v13, a1 // (op 134) 
v_accvgpr_read v14, a2 // (op 134) 
v_accvgpr_read v15, a3 // (op 134) 
v_mul_f32 v16, v8, v12 // (op 134) 
v_mul_f32 v17, v8, v13 // (op 134) 
v_mul_f32 v18, v8, v14 // (op 134) 
v_mul_f32 v19, v8, v15 // (op 134) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v10 // (op 139) 
v_lshrrev_b32 v13, 16, v10 // (op 139) 
v_cvt_f32_f16 v13, v13 // (op 139) 
v_mul_f32 v20, v5, v12 // (op 139) 
v_mul_f32 v21, v5, v13 // (op 139) 
v_cvt_f32_f16 v12, v11 // (op 139) 
v_lshrrev_b32 v13, 16, v11 // (op 139) 
v_cvt_f32_f16 v13, v13 // (op 139) 
v_mul_f32 v22, v5, v12 // (op 139) 
v_mul_f32 v23, v5, v13 // (op 139) 


v_add_f32 v10, v20, v16 // (op 135) 
v_add_f32 v11, v21, v17 // (op 135) 
v_add_f32 v12, v22, v18 // (op 135) 
v_add_f32 v13, v23, v19 // (op 135) 


v_cvt_f16_f32 v14, v10 // (op 136) 
v_cvt_f16_f32 v15, v11 // (op 136) 
v_cvt_f16_f32 v16, v12 // (op 136) 
v_cvt_f16_f32 v17, v13 // (op 136) 


v_pack_b32_f16 v10, v14, v15 // (op 137) 
v_pack_b32_f16 v11, v16, v17 // (op 137) 

v_mov_b32 v0, v9 // (op 137) reg expression

buffer_store_dwordx2 v[10:11], v0, s[60:63], 0 offen // (op 137) Store value





s_lshl_b32 s0, s2, 6 // (op 535) 
v_add_u32 v0, v6, s0 // (op 535) 


buffer_load_dwordx2 v[10:11], v0, s[56:59], 0 offen // (op 535) Load value

v_accvgpr_read v12, a4 // (op 531) 
v_accvgpr_read v13, a5 // (op 531) 
v_accvgpr_read v14, a6 // (op 531) 
v_accvgpr_read v15, a7 // (op 531) 
v_mul_f32 v16, v8, v12 // (op 531) 
v_mul_f32 v17, v8, v13 // (op 531) 
v_mul_f32 v18, v8, v14 // (op 531) 
v_mul_f32 v19, v8, v15 // (op 531) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v12, v10 // (op 536) 
v_lshrrev_b32 v13, 16, v10 // (op 536) 
v_cvt_f32_f16 v13, v13 // (op 536) 
v_mul_f32 v20, v5, v12 // (op 536) 
v_mul_f32 v21, v5, v13 // (op 536) 
v_cvt_f32_f16 v12, v11 // (op 536) 
v_lshrrev_b32 v13, 16, v11 // (op 536) 
v_cvt_f32_f16 v13, v13 // (op 536) 
v_mul_f32 v22, v5, v12 // (op 536) 
v_mul_f32 v23, v5, v13 // (op 536) 


v_add_f32 v10, v20, v16 // (op 532) 
v_add_f32 v11, v21, v17 // (op 532) 
v_add_f32 v12, v22, v18 // (op 532) 
v_add_f32 v13, v23, v19 // (op 532) 


v_cvt_f16_f32 v14, v10 // (op 533) 
v_cvt_f16_f32 v15, v11 // (op 533) 
v_cvt_f16_f32 v16, v12 // (op 533) 
v_cvt_f16_f32 v17, v13 // (op 533) 


v_pack_b32_f16 v10, v14, v15 // (op 534) 
v_pack_b32_f16 v11, v16, v17 // (op 534) 

s_lshl_b32 s0, s4, 6 // (op 534) 
v_add_u32 v0, v9, s0 // (op 534) 


buffer_store_dwordx2 v[10:11], v0, s[60:63], 0 offen // (op 534) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ConditionalBottom_0_beta_125:










s_endpgm  // End of RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_
.LRR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__end:
.size RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_, .LRR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__end-RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 72
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 25088 // lds bytes
  .amdhsa_accum_offset 64
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_user_sgpr_kernarg_preload_length 14
  .amdhsa_user_sgpr_kernarg_preload_offset 0
.amdhsa_system_sgpr_workgroup_id_x 1
.amdhsa_system_sgpr_workgroup_id_y 1
.amdhsa_system_sgpr_workgroup_id_z 0
.amdhsa_system_sgpr_workgroup_info 0
.amdhsa_system_vgpr_workitem_id 1
.end_amdhsa_kernel

.amdgpu_metadata
---
custom.config:
  Source:
    Origin: rocroller
  Version: 1.0.0
  Features:
    SupportsUserArgs: false
    SupportsBias: false
    SupportsActivation: false
    SupportsScaleAlpha: false
    SupportsGSU: false
  InternalSupportParams:
    KernArgsVersion: 0
  MXScaleFormat: NoSwizzle
  ProblemType:
    OperationType: GEMM
    DataType: F8
    DataTypeA: F8
    DataTypeB: F8
    DestDataType: h
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: True
    TransposeB: True
    UseBeta: True
    Batched: True
    UseBias: 0
    Activation: False
    UseScaleAlphaVec: 0
    MXBlockA: 32
    MXBlockB: 32
  CustomKernel:
    args: [ { type: int64, semantic: StrideC0 },
            { type: int64, semantic: StrideD0 },
            { type: int64, semantic: SizeFree1 },
            { type: int64, semantic: SizeFree0 },
            { type: int32, semantic: RR_MagicMultipleWgmMainBlock },
            { type: int32, semantic: RR_MagicMultipleWgm },
            { type: uint32, semantic: RR_MagicShiftAndSignWgm },
            { type: int32, semantic: RR_MagicMultipleNumTilesN },
            { type: uint32, semantic: RR_MagicShiftAndSignNumTilesN },
            { type: int32, semantic: RR_MagicMultipleWgmTail },
            { type: address, semantic: AddressC },
            { type: address, semantic: AddressD },
            { type: int64, semantic: StrideA0 },
            { type: address, semantic: AddressA },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: SizeSum },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: SizeSumScaleA },
            { type: int32, semantic: RR_WorkgroupMapping },
            { type: int32, semantic: RR_QuotientTilesMByWgm },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmTail },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmMainBlock },
            { type: int32, semantic: RR_QuotientTilesByBlock },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha } ]
    macrotile: [32, 64, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 2]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_
    .symbol: RR_GEMM_TT_FP8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_.kd
    .kernarg_segment_size: 180
    .group_segment_fixed_size: 25088
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 62
    .agpr_count: 8
    .max_flat_workgroup_size: 256
    .wavefront_size: 64
    .args:
      - .name: Tensor_10_stride_1_11
        .size: 8
        .offset: 0
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 8
        .value_kind: by_value
      - .name: Tensor_2_size_1_12
        .size: 8
        .offset: 16
        .value_kind: by_value
      - .name: Tensor_0_size_0_1
        .size: 8
        .offset: 24
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 32
        .value_kind: by_value
      - .name: MagicMultiple_22
        .size: 4
        .offset: 36
        .value_kind: by_value
      - .name: MagicShiftAndSign_23
        .size: 4
        .offset: 40
        .value_kind: by_value
      - .name: MagicMultiple_28
        .size: 4
        .offset: 44
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 48
        .value_kind: by_value
      - .name: MagicMultiple_30
        .size: 4
        .offset: 52
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 56
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 64
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_stride_0_0
        .size: 8
        .offset: 72
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 80
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_stride_0_3
        .size: 8
        .offset: 88
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 96
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_4
        .size: 8
        .offset: 104
        .value_kind: by_value
      - .name: Tensor_4_stride_0_6
        .size: 8
        .offset: 112
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 120
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_7_stride_0_8
        .size: 8
        .offset: 128
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 136
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_size_1_9
        .size: 8
        .offset: 144
        .value_kind: by_value
      - .name: WGM_18
        .size: 4
        .offset: 152
        .value_kind: by_value
      - .name: Subtract_24
        .size: 4
        .offset: 156
        .value_kind: by_value
      - .name: MagicShiftAndSign_31
        .size: 4
        .offset: 160
        .value_kind: by_value
      - .name: MagicShiftAndSign_27
        .size: 4
        .offset: 164
        .value_kind: by_value
      - .name: Subtract_21
        .size: 4
        .offset: 168
        .value_kind: by_value
      - .name: user_Float_Value_14
        .size: 4
        .offset: 172
        .value_kind: by_value
      - .name: user_Float_Value_12
        .size: 4
        .offset: 176
        .value_kind: by_value
...

.end_amdgpu_metadata

