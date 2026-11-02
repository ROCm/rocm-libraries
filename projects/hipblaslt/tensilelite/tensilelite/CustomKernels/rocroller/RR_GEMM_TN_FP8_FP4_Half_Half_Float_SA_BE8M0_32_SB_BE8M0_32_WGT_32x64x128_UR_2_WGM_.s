// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT



.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_,@function
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_:
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
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__exec_begin:


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
s_load_dword s48, s[0:1], 168 // Load scalar value

v_bfe_u32 v1, v0, 0, 10
v_bfe_u32 v2, v0, 10, 10


s_mul_hi_i32 s0, s16, s10 // (op 1135) 
s_mul_hi_i32 s1, s16, s10 // (op 1324) 
v_lshrrev_b32 v0, 7, v1 // (op 1192) 
v_and_b32 v3, 15, v1 // (op 1192) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 1192) 


v_and_b32 v0, 48, v1 // (op 1192) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 1192) 


v_lshrrev_b32 v0, 7, v1 // (op 1350) 
v_and_b32 v4, 15, v1 // (op 1350) 
v_lshl_add_u32 v5, v0, 4, v4 // (op 1350) 


v_and_b32 v0, 48, v1 // (op 1350) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 1350) 


v_lshrrev_b32 v0, 7, v1 // (op 1135) 
s_add_i32 s18, s0, s16 // (op 1135) 

v_and_b32 v5, 63, v1 // (op 1135) 
s_ashr_i32 s0, s18, 31 // (op 1135) 
v_lshrrev_b32 v6, 4, v5 // (op 1135) 
s_cmp_eq_i32 s10, 0 // (op 1135) 
s_mov_b32 s19, scc // (op 1135) 
v_lshlrev_b32 v7, 2, v6 // (op 1135) 

s_cmp_eq_i32 s19, 0 // (op 1135) 
s_cselect_b32 s49, 0, -1 // (op 1135) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 1135) 


s_and_b32 s19, s13, 31 // (op 1135) 
v_lshrrev_b32 v0, 2, v5 // (op 1135) 

v_and_b32 v5, 3, v0 // (op 1135) 

v_lshrrev_b32 v0, 6, v1 // (op 1250) 
v_and_b32 v7, 1, v0 // (op 1250) 

v_and_b32 v0, 15, v1 // (op 1250) 
v_lshl_add_u32 v8, v7, 4, v0 // (op 1250) 


v_and_b32 v0, 63, v1 // (op 1250) 
v_lshrrev_b32 v7, 4, v0 // (op 1250) 

v_lshlrev_b32 v0, 5, v7 // (op 1250) 

v_lshl_add_u32 v7, v8, 7, v0 // (op 1250) 


v_lshlrev_b32 v0, 2, v7 // (op 1250) 

v_lshrrev_b32 v7, 3, v0 // (op 1250) 

v_lshrrev_b32 v0, 6, v1 // (op 1308) 
v_and_b32 v8, 1, v0 // (op 1308) 

v_and_b32 v0, 15, v1 // (op 1308) 
v_lshl_add_u32 v9, v8, 4, v0 // (op 1308) 


v_and_b32 v0, 63, v1 // (op 1308) 
v_lshrrev_b32 v8, 4, v0 // (op 1308) 

v_lshl_add_u32 v0, v9, 2, v8 // (op 1308) 


v_lshrrev_b32 v8, 7, v1 // (op 1324) 
v_and_b32 v9, 63, v1 // (op 1324) 
v_lshrrev_b32 v10, 6, v1 // (op 1379) 
v_and_b32 v11, 1, v10 // (op 1379) 

v_and_b32 v10, 15, v1 // (op 1379) 
v_lshl_add_u32 v12, v11, 4, v10 // (op 1379) 


v_and_b32 v10, 63, v1 // (op 1379) 
v_lshrrev_b32 v11, 4, v10 // (op 1379) 

v_lshlrev_b32 v10, 5, v11 // (op 1379) 

v_lshl_add_u32 v11, v12, 7, v10 // (op 1379) 


v_lshlrev_b32 v10, 2, v11 // (op 1379) 

v_lshrrev_b32 v11, 3, v10 // (op 1379) 

v_lshrrev_b32 v10, 6, v1 // (op 1408) 
v_and_b32 v12, 1, v10 // (op 1408) 

v_and_b32 v10, 15, v1 // (op 1408) 
v_lshl_add_u32 v13, v12, 4, v10 // (op 1408) 


v_and_b32 v10, 63, v1 // (op 1408) 
v_lshrrev_b32 v12, 4, v10 // (op 1408) 

v_lshl_add_u32 v10, v13, 2, v12 // (op 1408) 


s_lshl_b32 s50, 1, s19 // (op 1135) 
v_and_b32 v12, 3, v1 // (op 1135) 
s_add_i32 s51, s50, s49 // (op 1135) 


s_and_b32 s49, s0, s51 // (op 1135) 


s_add_i32 s0, s1, s16 // (op 1324) 

s_ashr_i32 s1, s0, 31 // (op 1324) 
v_lshl_add_u32 v13, v5, 2, v12 // (op 1135) 


s_add_i32 s50, s18, s49 // (op 1135) 


v_lshrrev_b32 v5, 6, v1 // (op 1135) 
s_ashr_i32 s18, s50, s19 // (op 1135) 


v_and_b32 v12, 1, v5 // (op 1135) 

s_lshl_b32 s19, s13, 24 // (op 1135) 
v_lshl_add_u32 v5, v12, 4, v13 // (op 1135) 


s_ashr_i32 s49, s19, 31 // (op 1135) 

s_xor_b32 s19, s18, s49 // (op 1135) 

s_sub_i32 s18, s19, s49 // (op 1135) 


s_mul_hi_i32 s19, s18, s14 // (op 1135) 
s_add_i32 s49, s19, s18 // (op 1135) 

s_ashr_i32 s19, s49, 31 // (op 1135) 
s_cmp_eq_i32 s14, 0 // (op 1135) 
s_mov_b32 s50, scc // (op 1135) 
s_cmp_eq_i32 s50, 0 // (op 1135) 
s_cselect_b32 s51, 0, -1 // (op 1135) 

s_and_b32 s50, s15, 31 // (op 1135) 
v_lshrrev_b32 v12, 4, v9 // (op 1324) 
s_cmp_eq_i32 s10, 0 // (op 1324) 
s_mov_b32 s52, scc // (op 1324) 
v_lshlrev_b32 v13, 2, v12 // (op 1324) 

s_cmp_eq_i32 s52, 0 // (op 1324) 
s_cselect_b32 s53, 0, -1 // (op 1324) 

s_lshl_b32 s52, 1, s50 // (op 1135) 
v_lshl_add_u32 v12, v8, 4, v13 // (op 1324) 


v_lshrrev_b32 v8, 2, v9 // (op 1324) 

s_add_i32 s54, s52, s51 // (op 1135) 


s_and_b32 s51, s19, s54 // (op 1135) 


s_add_i32 s19, s49, s51 // (op 1135) 


s_ashr_i32 s49, s19, s50 // (op 1135) 


s_lshl_b32 s19, s15, 24 // (op 1135) 
s_ashr_i32 s50, s19, 31 // (op 1135) 

s_xor_b32 s19, s49, s50 // (op 1135) 

s_sub_i32 s49, s19, s50 // (op 1135) 


s_mul_i32 s19, s18, s11 // (op 1135) 
s_sub_u32 s50, s16, s19 // (op 1135) 

s_mul_i32 s19, s49, s11 // (op 1135) 
s_add_u32 s19, s19, s50 // (op 1135) 

s_mul_hi_i32 s50, s16, s9 // (op 1135) 
s_add_i32 s51, s50, s16 // (op 1135) 

s_ashr_i32 s50, s51, 31 // (op 1135) 
s_cmp_eq_i32 s9, 0 // (op 1135) 
s_mov_b32 s52, scc // (op 1135) 
s_cmp_eq_i32 s52, 0 // (op 1135) 
s_cselect_b32 s54, 0, -1 // (op 1135) 

s_and_b32 s52, s13, 31 // (op 1324) 
v_and_b32 v9, 3, v8 // (op 1324) 

s_lshl_b32 s55, 1, s52 // (op 1324) 
v_and_b32 v8, 3, v1 // (op 1324) 
s_add_i32 s56, s55, s53 // (op 1324) 


s_and_b32 s53, s1, s56 // (op 1324) 


v_lshl_add_u32 v13, v9, 2, v8 // (op 1324) 


s_add_i32 s1, s0, s53 // (op 1324) 


v_lshrrev_b32 v8, 6, v1 // (op 1324) 
s_ashr_i32 s0, s1, s52 // (op 1324) 


v_and_b32 v9, 1, v8 // (op 1324) 

s_lshl_b32 s1, s13, 24 // (op 1324) 
v_lshl_add_u32 v8, v9, 4, v13 // (op 1324) 


s_ashr_i32 s52, s1, 31 // (op 1324) 

s_xor_b32 s1, s0, s52 // (op 1324) 

s_sub_i32 s0, s1, s52 // (op 1324) 


s_mul_hi_i32 s1, s0, s14 // (op 1324) 
s_add_i32 s52, s1, s0 // (op 1324) 

s_ashr_i32 s1, s52, 31 // (op 1324) 
s_cmp_eq_i32 s14, 0 // (op 1324) 
s_mov_b32 s53, scc // (op 1324) 
s_cmp_eq_i32 s53, 0 // (op 1324) 
s_cselect_b32 s55, 0, -1 // (op 1324) 

s_and_b32 s53, s15, 31 // (op 1324) 
s_lshl_b32 s56, 1, s53 // (op 1324) 
s_add_i32 s57, s56, s55 // (op 1324) 


s_and_b32 s55, s1, s57 // (op 1324) 


s_add_i32 s1, s52, s55 // (op 1324) 


s_ashr_i32 s52, s1, s53 // (op 1324) 


s_lshl_b32 s1, s15, 24 // (op 1324) 
s_ashr_i32 s53, s1, 31 // (op 1324) 

s_xor_b32 s1, s52, s53 // (op 1324) 

s_sub_i32 s52, s1, s53 // (op 1324) 


s_mul_i32 s1, s0, s11 // (op 1324) 
s_sub_u32 s53, s16, s1 // (op 1324) 

s_mul_i32 s1, s52, s11 // (op 1324) 
s_add_u32 s1, s1, s53 // (op 1324) 

s_mul_hi_i32 s53, s16, s9 // (op 1324) 
s_add_i32 s55, s53, s16 // (op 1324) 

s_ashr_i32 s53, s55, 31 // (op 1324) 
s_cmp_eq_i32 s9, 0 // (op 1324) 
s_mov_b32 s56, scc // (op 1324) 
s_cmp_eq_i32 s56, 0 // (op 1324) 
s_cselect_b32 s57, 0, -1 // (op 1324) 


s_waitcnt lgkmcnt(0)
s_and_b32 s56, s45, 31 // (op 1135) 
s_lshl_b32 s58, 1, s56 // (op 1135) 
s_add_i32 s59, s58, s54 // (op 1135) 


s_and_b32 s54, s50, s59 // (op 1135) 


s_add_i32 s50, s51, s54 // (op 1135) 


s_ashr_i32 s51, s50, s56 // (op 1135) 


s_lshl_b32 s50, s45, 24 // (op 1135) 
s_ashr_i32 s54, s50, 31 // (op 1135) 

s_xor_b32 s50, s51, s54 // (op 1135) 

s_sub_i32 s51, s50, s54 // (op 1135) 


s_and_b32 s50, s45, 31 // (op 1324) 
s_lshl_b32 s54, 1, s50 // (op 1324) 
s_add_i32 s56, s54, s57 // (op 1324) 


s_and_b32 s54, s53, s56 // (op 1324) 


s_add_i32 s53, s55, s54 // (op 1324) 


s_ashr_i32 s54, s53, s50 // (op 1324) 


s_lshl_b32 s50, s45, 24 // (op 1324) 
s_ashr_i32 s53, s50, 31 // (op 1324) 

s_xor_b32 s50, s54, s53 // (op 1324) 

s_sub_i32 s54, s50, s53 // (op 1324) 


s_add_u32 s56, s6, 64 // (op 1135) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 1135) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 1135) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 1135) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 1135) 

s_mul_i32 s50, s11, s56 // (op 1135) 
s_mul_i32 s53, s8, s50 // (op 1135) 

s_mul_i32 s50, s51, s53 // (op 1135) 

s_sub_u32 s53, s16, s50 // (op 1135) 

s_mov_b32 s50, s53 // (op 1135) 
s_mul_hi_i32 s55, s50, s12 // (op 1135) 
s_add_i32 s58, s55, s50 // (op 1135) 


s_ashr_i32 s50, s58, 31 // (op 1135) 
s_cmp_eq_i32 s12, 0 // (op 1135) 
s_mov_b32 s55, scc // (op 1135) 
s_cmp_eq_i32 s55, 0 // (op 1135) 
s_cselect_b32 s59, 0, -1 // (op 1135) 

s_and_b32 s55, s44, 31 // (op 1135) 
s_lshl_b32 s60, 1, s55 // (op 1135) 
s_add_i32 s61, s60, s59 // (op 1135) 


s_and_b32 s59, s50, s61 // (op 1135) 


s_add_i32 s50, s58, s59 // (op 1135) 


s_ashr_i32 s58, s50, s55 // (op 1135) 


s_lshl_b32 s50, s44, 24 // (op 1135) 
s_ashr_i32 s55, s50, 31 // (op 1135) 

s_xor_b32 s50, s58, s55 // (op 1135) 

s_sub_i32 s58, s50, s55 // (op 1135) 


s_mul_i32 s50, s46, s11 // (op 1135) 
s_add_u32 s60, s20, 32 // (op 1135) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 1135) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 1135) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 1135) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 5 // (op 1135) 

s_sub_i32 s55, s60, s50 // (op 1135) 

s_mul_i32 s59, s58, s55 // (op 1135) 

s_sub_u32 s55, s53, s59 // (op 1135) 


s_add_u32 s53, s55, s50 // (op 1135) 


s_cmp_eq_i32 s51, 0 // (op 1135) 
s_mov_b32 s50, scc // (op 1135) 

s_cmp_eq_i32 s50, 0 // (op 1135) 
s_cselect_b32 s51, s53, s19 // (op 1135) 


s_mul_i32 s19, s49, s56 // (op 1135) 


v_lshl_add_u32 v9, s51, 5, v6 // (op 1135) 


s_sub_i32 s49, s18, s19 // (op 1135) 


s_cmp_eq_i32 s50, 0 // (op 1135) 
s_cselect_b32 s51, s58, s49 // (op 1135) 



v_lshl_add_u32 v6, s51, 6, v5 // (op 1135) 


v_mul_lo_u32 v5, v6, s2 // (op 1135) 

v_add_lshl_u32 v6, v9, v5, 1 // (op 1135) 


s_add_u32 s18, s6, 64 // (op 1324) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 1324) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 1324) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 1324) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 6 // (op 1324) 

s_mul_i32 s53, s11, s18 // (op 1324) 
s_mul_i32 s49, s8, s53 // (op 1324) 

s_mul_i32 s53, s54, s49 // (op 1324) 

s_sub_u32 s49, s16, s53 // (op 1324) 

s_mov_b32 s53, s49 // (op 1324) 
s_mul_hi_i32 s50, s53, s12 // (op 1324) 
s_add_i32 s51, s50, s53 // (op 1324) 


s_ashr_i32 s50, s51, 31 // (op 1324) 
s_cmp_eq_i32 s12, 0 // (op 1324) 
s_mov_b32 s53, scc // (op 1324) 
s_cmp_eq_i32 s53, 0 // (op 1324) 
s_cselect_b32 s55, 0, -1 // (op 1324) 

s_and_b32 s53, s44, 31 // (op 1324) 
s_lshl_b32 s56, 1, s53 // (op 1324) 
s_add_i32 s57, s56, s55 // (op 1324) 


s_and_b32 s55, s50, s57 // (op 1324) 


s_add_i32 s50, s51, s55 // (op 1324) 


s_ashr_i32 s51, s50, s53 // (op 1324) 


s_lshl_b32 s50, s44, 24 // (op 1324) 
s_ashr_i32 s53, s50, 31 // (op 1324) 

s_xor_b32 s50, s51, s53 // (op 1324) 

s_sub_i32 s51, s50, s53 // (op 1324) 


s_mul_i32 s50, s46, s11 // (op 1324) 
s_add_u32 s56, s20, 32 // (op 1324) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 1324) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 1324) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 1324) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 5 // (op 1324) 

s_sub_i32 s53, s56, s50 // (op 1324) 

s_mul_i32 s55, s51, s53 // (op 1324) 

s_sub_u32 s53, s49, s55 // (op 1324) 


s_add_u32 s49, s53, s50 // (op 1324) 


s_cmp_eq_i32 s54, 0 // (op 1324) 
s_mov_b32 s50, scc // (op 1324) 

s_cmp_eq_i32 s50, 0 // (op 1324) 
s_cselect_b32 s53, s49, s1 // (op 1324) 


s_mul_i32 s1, s52, s18 // (op 1324) 


v_lshl_add_u32 v5, s53, 5, v12 // (op 1324) 


s_sub_i32 s49, s0, s1 // (op 1324) 


s_cmp_eq_i32 s50, 0 // (op 1324) 
s_cselect_b32 s0, s51, s49 // (op 1324) 



v_lshl_add_u32 v9, s0, 6, v8 // (op 1324) 


v_mul_lo_u32 v8, v9, s4 // (op 1324) 

v_add_lshl_u32 v9, v5, v8, 1 // (op 1324) 


s_mul_i32 s1, s2, s7 // (op 1137) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 1137) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 1137) least significant
s_mul_hi_u32 s19, s2, s6 // (op 1137) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1137) most significant: sum
s_add_u32 s1, s1, s19 // (op 1137) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 1137) 

s_mul_i32 s1, s4, s7 // (op 1326) most significant: low of low * high
s_mul_i32 s49, s5, s6 // (op 1326) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 1326) least significant
s_mul_hi_u32 s50, s4, s6 // (op 1326) most significant: high of low * low
s_add_u32 s1, s1, s49 // (op 1326) most significant: sum
s_add_u32 s1, s1, s50 // (op 1326) most significant: sum


s_lshl_b64 s[50:51], s[0:1], 1 // (op 1326) 

s_mov_b32 s52, s22 // (op 1137) reg expression
s_mov_b32 s53, s23 // (op 1137) reg expression
s_mov_b32 s54, s18 // (op 1137) reg expression
s_mov_b32 s55, 131072 // (op 1137) call()

s_mov_b32 s56, s24 // (op 1326) reg expression

s_mov_b32 s57, s25 // (op 1326) reg expression
s_mov_b32 s58, s50 // (op 1326) reg expression
s_mov_b32 s59, 131072 // (op 1326) call()


v_mov_b32 v5, s47 // (op 2) Move value
v_mov_b32 v8, s48 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
v_accvgpr_write a4, 0.00000 // (op 310) call()
v_accvgpr_write a5, 0.00000 // (op 310) call()
v_accvgpr_write a6, 0.00000 // (op 310) call()
v_accvgpr_write a7, 0.00000 // (op 310) call()
s_mul_hi_i32 s0, s16, s10 // (op 902) 
v_lshrrev_b32 v12, 3, v1 // (op 902) 
s_add_i32 s1, s0, s16 // (op 902) 

v_and_b32 v13, -8, v1 // (op 902) 
s_ashr_i32 s0, s1, 31 // (op 902) 
v_sub_u32 v14, v1, v13 // (op 902) 

s_cmp_eq_i32 s10, 0 // (op 902) 
s_mov_b32 s18, scc // (op 902) 
v_lshlrev_b32 v13, 4, v14 // (op 902) 

s_cmp_eq_i32 s18, 0 // (op 902) 
s_cselect_b32 s19, 0, -1 // (op 902) 

s_and_b32 s18, s13, 31 // (op 902) 
s_lshl_b32 s22, 1, s18 // (op 902) 
s_add_i32 s23, s22, s19 // (op 902) 


s_and_b32 s19, s0, s23 // (op 902) 


s_add_i32 s0, s1, s19 // (op 902) 


s_ashr_i32 s1, s0, s18 // (op 902) 


s_lshl_b32 s0, s13, 24 // (op 902) 
s_ashr_i32 s18, s0, 31 // (op 902) 

s_xor_b32 s0, s1, s18 // (op 902) 

s_sub_i32 s1, s0, s18 // (op 902) 


s_mul_hi_i32 s0, s1, s14 // (op 902) 
s_add_i32 s18, s0, s1 // (op 902) 

s_ashr_i32 s0, s18, 31 // (op 902) 
s_cmp_eq_i32 s14, 0 // (op 902) 
s_mov_b32 s19, scc // (op 902) 
s_cmp_eq_i32 s19, 0 // (op 902) 
s_cselect_b32 s22, 0, -1 // (op 902) 

s_and_b32 s19, s15, 31 // (op 902) 
s_lshl_b32 s23, 1, s19 // (op 902) 
s_add_i32 s24, s23, s22 // (op 902) 


s_and_b32 s25, s0, s24 // (op 902) 


s_add_i32 s0, s18, s25 // (op 902) 


s_ashr_i32 s18, s0, s19 // (op 902) 


s_lshl_b32 s0, s15, 24 // (op 902) 
s_ashr_i32 s19, s0, 31 // (op 902) 

s_xor_b32 s0, s18, s19 // (op 902) 

s_sub_i32 s18, s0, s19 // (op 902) 


s_mul_i32 s0, s1, s11 // (op 902) 

s_sub_u32 s1, s16, s0 // (op 902) 

s_mul_i32 s0, s18, s11 // (op 902) 
s_add_u32 s0, s0, s1 // (op 902) 


s_mul_hi_i32 s1, s16, s9 // (op 902) 
s_add_i32 s18, s1, s16 // (op 902) 

s_ashr_i32 s1, s18, 31 // (op 902) 
s_cmp_eq_i32 s9, 0 // (op 902) 
s_mov_b32 s19, scc // (op 902) 
s_cmp_eq_i32 s19, 0 // (op 902) 
s_cselect_b32 s22, 0, -1 // (op 902) 

s_and_b32 s19, s45, 31 // (op 902) 
s_lshl_b32 s23, 1, s19 // (op 902) 
s_add_i32 s24, s23, s22 // (op 902) 


s_and_b32 s25, s1, s24 // (op 902) 


s_add_i32 s1, s18, s25 // (op 902) 


s_ashr_i32 s18, s1, s19 // (op 902) 


s_lshl_b32 s1, s45, 24 // (op 902) 
s_ashr_i32 s19, s1, 31 // (op 902) 

s_xor_b32 s1, s18, s19 // (op 902) 

s_sub_i32 s18, s1, s19 // (op 902) 


s_add_u32 s22, s6, 64 // (op 902) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 902) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 902) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 902) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 902) 

s_mul_i32 s1, s11, s22 // (op 902) 

s_mul_i32 s19, s8, s1 // (op 902) 

s_mul_i32 s1, s18, s19 // (op 902) 

s_sub_u32 s19, s16, s1 // (op 902) 

s_mov_b32 s1, s19 // (op 902) 
s_mul_hi_i32 s22, s1, s12 // (op 902) 
s_add_i32 s23, s22, s1 // (op 902) 


s_ashr_i32 s1, s23, 31 // (op 902) 
s_cmp_eq_i32 s12, 0 // (op 902) 
s_mov_b32 s22, scc // (op 902) 
s_cmp_eq_i32 s22, 0 // (op 902) 
s_cselect_b32 s24, 0, -1 // (op 902) 

s_and_b32 s22, s44, 31 // (op 902) 
s_lshl_b32 s25, 1, s22 // (op 902) 
s_add_i32 s47, s25, s24 // (op 902) 


s_and_b32 s24, s1, s47 // (op 902) 


s_add_i32 s1, s23, s24 // (op 902) 


s_ashr_i32 s23, s1, s22 // (op 902) 


s_lshl_b32 s1, s44, 24 // (op 902) 
s_ashr_i32 s22, s1, 31 // (op 902) 

s_xor_b32 s1, s23, s22 // (op 902) 

s_sub_i32 s23, s1, s22 // (op 902) 


s_mul_i32 s1, s46, s11 // (op 902) 
s_add_u32 s24, s20, 32 // (op 902) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 902) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 902) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 902) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 5 // (op 902) 

s_sub_i32 s22, s24, s1 // (op 902) 

s_mul_i32 s24, s23, s22 // (op 902) 


s_sub_u32 s25, s19, s24 // (op 902) 


s_add_u32 s19, s25, s1 // (op 902) 


s_cmp_eq_i32 s18, 0 // (op 902) 
s_mov_b32 s1, scc // (op 902) 

s_cmp_eq_i32 s1, 0 // (op 902) 
s_cselect_b32 s18, s19, s0 // (op 902) 



v_lshl_add_u32 v14, s18, 5, v12 // (op 902) 


v_mul_lo_u32 v12, v14, s26 // (op 902) 
v_add_u32 v12, v12, v13 // (op 902) 


s_mul_i32 s1, s26, s21 // (op 904) most significant: low of low * high
s_mul_i32 s18, s27, s20 // (op 904) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 904) least significant
s_mul_hi_u32 s19, s26, s20 // (op 904) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 904) most significant: sum
s_add_u32 s1, s1, s19 // (op 904) most significant: sum


s_mov_b32 s48, s28 // (op 904) reg expression
s_mov_b32 s49, s29 // (op 904) reg expression
s_mov_b32 s50, s0 // (op 904) reg expression
s_mov_b32 s51, 131072 // (op 904) call()

s_mul_hi_i32 s47, s16, s10 // (op 921) 
v_lshrrev_b32 v13, 3, v1 // (op 921) 
s_add_i32 s0, s47, s16 // (op 921) 

v_and_b32 v14, -8, v1 // (op 921) 
s_ashr_i32 s1, s0, 31 // (op 921) 
v_sub_u32 v15, v1, v14 // (op 921) 

s_cmp_eq_i32 s10, 0 // (op 921) 
s_mov_b32 s47, scc // (op 921) 
v_lshlrev_b32 v14, 4, v15 // (op 921) 

s_cmp_eq_i32 s47, 0 // (op 921) 
s_cselect_b32 s18, 0, -1 // (op 921) 

s_and_b32 s19, s13, 31 // (op 921) 
s_lshl_b32 s47, 1, s19 // (op 921) 
s_add_i32 s22, s47, s18 // (op 921) 


s_and_b32 s18, s1, s22 // (op 921) 


s_add_i32 s1, s0, s18 // (op 921) 


s_ashr_i32 s0, s1, s19 // (op 921) 


s_lshl_b32 s1, s13, 24 // (op 921) 
s_ashr_i32 s47, s1, 31 // (op 921) 

s_xor_b32 s1, s0, s47 // (op 921) 

s_sub_i32 s0, s1, s47 // (op 921) 


s_mul_hi_i32 s1, s0, s14 // (op 921) 
s_add_i32 s47, s1, s0 // (op 921) 

s_ashr_i32 s1, s47, 31 // (op 921) 
s_cmp_eq_i32 s14, 0 // (op 921) 
s_mov_b32 s18, scc // (op 921) 
s_cmp_eq_i32 s18, 0 // (op 921) 
s_cselect_b32 s19, 0, -1 // (op 921) 

s_and_b32 s18, s15, 31 // (op 921) 
s_lshl_b32 s22, 1, s18 // (op 921) 
s_add_i32 s23, s22, s19 // (op 921) 


s_and_b32 s19, s1, s23 // (op 921) 


s_add_i32 s1, s47, s19 // (op 921) 


s_ashr_i32 s19, s1, s18 // (op 921) 


s_lshl_b32 s1, s15, 24 // (op 921) 
s_ashr_i32 s18, s1, 31 // (op 921) 

s_xor_b32 s1, s19, s18 // (op 921) 

s_sub_i32 s19, s1, s18 // (op 921) 


s_mul_i32 s1, s0, s11 // (op 921) 

s_sub_u32 s0, s16, s1 // (op 921) 

s_mul_i32 s1, s19, s11 // (op 921) 
s_add_u32 s1, s1, s0 // (op 921) 


s_mul_hi_i32 s0, s16, s9 // (op 921) 
s_add_i32 s47, s0, s16 // (op 921) 

s_ashr_i32 s0, s47, 31 // (op 921) 
s_cmp_eq_i32 s9, 0 // (op 921) 
s_mov_b32 s18, scc // (op 921) 
s_cmp_eq_i32 s18, 0 // (op 921) 
s_cselect_b32 s19, 0, -1 // (op 921) 

s_and_b32 s18, s45, 31 // (op 921) 
s_lshl_b32 s22, 1, s18 // (op 921) 
s_add_i32 s23, s22, s19 // (op 921) 


s_and_b32 s19, s0, s23 // (op 921) 


s_add_i32 s0, s47, s19 // (op 921) 


s_ashr_i32 s19, s0, s18 // (op 921) 


s_lshl_b32 s0, s45, 24 // (op 921) 
s_ashr_i32 s18, s0, 31 // (op 921) 

s_xor_b32 s0, s19, s18 // (op 921) 

s_sub_i32 s19, s0, s18 // (op 921) 


s_add_u32 s22, s6, 64 // (op 921) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 921) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 921) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 921) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 921) 

s_mul_i32 s0, s11, s22 // (op 921) 

s_mul_i32 s18, s8, s0 // (op 921) 

s_mul_i32 s0, s19, s18 // (op 921) 

s_sub_u32 s18, s16, s0 // (op 921) 

s_mov_b32 s0, s18 // (op 921) 
s_mul_hi_i32 s47, s0, s12 // (op 921) 
s_add_i32 s22, s47, s0 // (op 921) 


s_ashr_i32 s0, s22, 31 // (op 921) 
s_cmp_eq_i32 s12, 0 // (op 921) 
s_mov_b32 s47, scc // (op 921) 
s_cmp_eq_i32 s47, 0 // (op 921) 
s_cselect_b32 s23, 0, -1 // (op 921) 

s_and_b32 s47, s44, 31 // (op 921) 
s_lshl_b32 s24, 1, s47 // (op 921) 
s_add_i32 s25, s24, s23 // (op 921) 


s_and_b32 s23, s0, s25 // (op 921) 


s_add_i32 s0, s22, s23 // (op 921) 


s_ashr_i32 s22, s0, s47 // (op 921) 


s_lshl_b32 s0, s44, 24 // (op 921) 
s_ashr_i32 s47, s0, 31 // (op 921) 

s_xor_b32 s0, s22, s47 // (op 921) 

s_sub_i32 s22, s0, s47 // (op 921) 


s_mul_i32 s0, s46, s11 // (op 921) 
s_add_u32 s24, s20, 32 // (op 921) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 921) End of Int64 add, unlocking SCC
s_sub_u32 s60, s24, 1 // (op 921) least significant half; sets scc
s_subb_u32 s61, s25, 0 // (op 921) most significant half; uses scc

s_ashr_i64 s[24:25], s[60:61], 5 // (op 921) 

s_sub_i32 s23, s24, s0 // (op 921) 

s_mul_i32 s47, s22, s23 // (op 921) 


s_sub_u32 s22, s18, s47 // (op 921) 


s_add_u32 s18, s22, s0 // (op 921) 


s_cmp_eq_i32 s19, 0 // (op 921) 
s_mov_b32 s0, scc // (op 921) 

s_cmp_eq_i32 s0, 0 // (op 921) 
s_cselect_b32 s19, s18, s1 // (op 921) 



v_lshl_add_u32 v15, s19, 5, v13 // (op 921) 


v_mul_lo_u32 v13, v15, s26 // (op 921) 
v_add_u32 v13, v13, v14 // (op 921) 


s_mul_i32 s1, s26, s21 // (op 923) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 923) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 923) least significant
s_mul_hi_u32 s18, s26, s20 // (op 923) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 923) most significant: sum
s_add_u32 s1, s1, s18 // (op 923) most significant: sum


s_mov_b32 s48, s28 // (op 923) reg expression
s_mov_b32 s49, s29 // (op 923) reg expression
s_mov_b32 s50, s0 // (op 923) reg expression
s_mov_b32 s51, 131072 // (op 923) call()

s_mul_hi_i32 s47, s16, s10 // (op 966) 
v_lshrrev_b32 v14, 2, v1 // (op 966) 
s_add_i32 s0, s47, s16 // (op 966) 

v_and_b32 v15, -4, v1 // (op 966) 
s_ashr_i32 s1, s0, 31 // (op 966) 
v_sub_u32 v16, v1, v15 // (op 966) 

s_cmp_eq_i32 s10, 0 // (op 966) 
s_mov_b32 s47, scc // (op 966) 
v_lshlrev_b32 v15, 5, v16 // (op 966) 

s_cmp_eq_i32 s47, 0 // (op 966) 
s_cselect_b32 s18, 0, -1 // (op 966) 

v_mov_b32 v17, 0 // (op 966) convert
v_mov_b32 v16, v15 // (op 966) convert

s_and_b32 s19, s13, 31 // (op 966) 
s_lshl_b32 s47, 1, s19 // (op 966) 
s_add_i32 s22, s47, s18 // (op 966) 


s_and_b32 s18, s1, s22 // (op 966) 


s_add_i32 s1, s0, s18 // (op 966) 


s_ashr_i32 s0, s1, s19 // (op 966) 


s_lshl_b32 s1, s13, 24 // (op 966) 
s_ashr_i32 s47, s1, 31 // (op 966) 

s_xor_b32 s1, s0, s47 // (op 966) 

s_sub_i32 s0, s1, s47 // (op 966) 


s_mul_hi_i32 s1, s0, s14 // (op 966) 
s_add_i32 s47, s1, s0 // (op 966) 

s_ashr_i32 s1, s47, 31 // (op 966) 
s_cmp_eq_i32 s14, 0 // (op 966) 
s_mov_b32 s18, scc // (op 966) 
s_cmp_eq_i32 s18, 0 // (op 966) 
s_cselect_b32 s19, 0, -1 // (op 966) 

s_and_b32 s18, s15, 31 // (op 966) 
s_lshl_b32 s22, 1, s18 // (op 966) 
s_add_i32 s23, s22, s19 // (op 966) 


s_and_b32 s19, s1, s23 // (op 966) 


s_add_i32 s1, s47, s19 // (op 966) 


s_ashr_i32 s19, s1, s18 // (op 966) 


s_lshl_b32 s1, s15, 24 // (op 966) 
s_ashr_i32 s18, s1, 31 // (op 966) 

s_xor_b32 s1, s19, s18 // (op 966) 

s_sub_i32 s19, s1, s18 // (op 966) 


s_add_u32 s22, s6, 64 // (op 966) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 966) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 966) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 966) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 966) 

s_mul_i32 s1, s19, s22 // (op 966) 

s_sub_i32 s47, s0, s1 // (op 966) 


s_mul_hi_i32 s0, s16, s9 // (op 966) 
s_add_i32 s1, s0, s16 // (op 966) 

s_ashr_i32 s0, s1, 31 // (op 966) 
s_cmp_eq_i32 s9, 0 // (op 966) 
s_mov_b32 s18, scc // (op 966) 
s_cmp_eq_i32 s18, 0 // (op 966) 
s_cselect_b32 s19, 0, -1 // (op 966) 

s_and_b32 s18, s45, 31 // (op 966) 
s_lshl_b32 s24, 1, s18 // (op 966) 
s_add_i32 s25, s24, s19 // (op 966) 


s_and_b32 s19, s0, s25 // (op 966) 


s_add_i32 s0, s1, s19 // (op 966) 


s_ashr_i32 s1, s0, s18 // (op 966) 


s_lshl_b32 s0, s45, 24 // (op 966) 
s_ashr_i32 s18, s0, 31 // (op 966) 

s_xor_b32 s0, s1, s18 // (op 966) 

s_sub_i32 s1, s0, s18 // (op 966) 


s_mul_i32 s0, s11, s22 // (op 966) 

s_mul_i32 s18, s8, s0 // (op 966) 

s_mul_i32 s0, s1, s18 // (op 966) 

s_sub_u32 s18, s16, s0 // (op 966) 

s_mul_hi_i32 s0, s18, s12 // (op 966) 
s_add_i32 s19, s0, s18 // (op 966) 


s_ashr_i32 s0, s19, 31 // (op 966) 
s_cmp_eq_i32 s12, 0 // (op 966) 
s_mov_b32 s18, scc // (op 966) 
s_cmp_eq_i32 s18, 0 // (op 966) 
s_cselect_b32 s22, 0, -1 // (op 966) 

s_and_b32 s18, s44, 31 // (op 966) 
s_lshl_b32 s23, 1, s18 // (op 966) 
s_add_i32 s24, s23, s22 // (op 966) 


s_and_b32 s25, s0, s24 // (op 966) 


s_add_i32 s0, s19, s25 // (op 966) 


s_ashr_i32 s19, s0, s18 // (op 966) 


s_lshl_b32 s0, s44, 24 // (op 966) 
s_ashr_i32 s18, s0, 31 // (op 966) 

s_xor_b32 s0, s19, s18 // (op 966) 

s_sub_i32 s19, s0, s18 // (op 966) 


s_cmp_eq_i32 s1, 0 // (op 966) 
s_mov_b32 s0, scc // (op 966) 

s_cmp_eq_i32 s0, 0 // (op 966) 
s_cselect_b32 s1, s19, s47 // (op 966) 



v_lshl_add_u32 v15, s1, 6, v14 // (op 966) 


v_mul_lo_u32 v19, v15, s31 // (op 966) most significant: low of low * high
v_mul_hi_u32 v14, v15, s30 // (op 966) most significant: high of low * low
v_mul_lo_u32 v18, v15, s30 // (op 966) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 966) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v18 // (op 966) least significant half
v_addc_co_u32 v15, s[0:1], v17, v19, s[0:1] // (op 966) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 966) 


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 966) 

v_mov_b32 v14, v16 // (op 966) convert

s_mul_i32 s1, s30, s7 // (op 968) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 968) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 968) least significant
s_mul_hi_u32 s18, s30, s6 // (op 968) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 968) most significant: sum
s_add_u32 s1, s1, s18 // (op 968) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 968) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 968) 

s_mov_b32 s60, s32 // (op 968) reg expression
s_mov_b32 s61, s33 // (op 968) reg expression
s_mov_b32 s62, s0 // (op 968) reg expression
s_mov_b32 s63, 131072 // (op 968) call()

s_mul_hi_i32 s47, s16, s10 // (op 985) 
v_lshrrev_b32 v15, 2, v1 // (op 985) 
s_add_i32 s0, s47, s16 // (op 985) 

v_and_b32 v16, -4, v1 // (op 985) 
s_ashr_i32 s1, s0, 31 // (op 985) 
v_sub_u32 v17, v1, v16 // (op 985) 

s_cmp_eq_i32 s10, 0 // (op 985) 
s_mov_b32 s47, scc // (op 985) 
v_lshlrev_b32 v16, 5, v17 // (op 985) 

s_cmp_eq_i32 s47, 0 // (op 985) 
s_cselect_b32 s18, 0, -1 // (op 985) 

v_mov_b32 v19, 0 // (op 985) convert
v_mov_b32 v18, v16 // (op 985) convert

s_and_b32 s19, s13, 31 // (op 985) 
s_lshl_b32 s47, 1, s19 // (op 985) 
s_add_i32 s22, s47, s18 // (op 985) 


s_and_b32 s18, s1, s22 // (op 985) 


s_add_i32 s1, s0, s18 // (op 985) 


s_ashr_i32 s0, s1, s19 // (op 985) 


s_lshl_b32 s1, s13, 24 // (op 985) 
s_ashr_i32 s47, s1, 31 // (op 985) 

s_xor_b32 s1, s0, s47 // (op 985) 

s_sub_i32 s0, s1, s47 // (op 985) 


s_mul_hi_i32 s1, s0, s14 // (op 985) 
s_add_i32 s47, s1, s0 // (op 985) 

s_ashr_i32 s1, s47, 31 // (op 985) 
s_cmp_eq_i32 s14, 0 // (op 985) 
s_mov_b32 s18, scc // (op 985) 
s_cmp_eq_i32 s18, 0 // (op 985) 
s_cselect_b32 s19, 0, -1 // (op 985) 

s_and_b32 s18, s15, 31 // (op 985) 
s_lshl_b32 s22, 1, s18 // (op 985) 
s_add_i32 s23, s22, s19 // (op 985) 


s_and_b32 s19, s1, s23 // (op 985) 


s_add_i32 s1, s47, s19 // (op 985) 


s_ashr_i32 s19, s1, s18 // (op 985) 


s_lshl_b32 s1, s15, 24 // (op 985) 
s_ashr_i32 s18, s1, 31 // (op 985) 

s_xor_b32 s1, s19, s18 // (op 985) 

s_sub_i32 s19, s1, s18 // (op 985) 


s_add_u32 s22, s6, 64 // (op 985) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 985) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 985) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 985) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 985) 

s_mul_i32 s1, s19, s22 // (op 985) 

s_sub_i32 s47, s0, s1 // (op 985) 


s_mul_hi_i32 s0, s16, s9 // (op 985) 
s_add_i32 s1, s0, s16 // (op 985) 

s_ashr_i32 s0, s1, 31 // (op 985) 
s_cmp_eq_i32 s9, 0 // (op 985) 
s_mov_b32 s18, scc // (op 985) 
s_cmp_eq_i32 s18, 0 // (op 985) 
s_cselect_b32 s19, 0, -1 // (op 985) 

s_and_b32 s18, s45, 31 // (op 985) 
s_lshl_b32 s24, 1, s18 // (op 985) 
s_add_i32 s25, s24, s19 // (op 985) 


s_and_b32 s19, s0, s25 // (op 985) 


s_add_i32 s0, s1, s19 // (op 985) 


s_ashr_i32 s1, s0, s18 // (op 985) 


s_lshl_b32 s0, s45, 24 // (op 985) 
s_ashr_i32 s18, s0, 31 // (op 985) 

s_xor_b32 s0, s1, s18 // (op 985) 

s_sub_i32 s1, s0, s18 // (op 985) 


s_mul_i32 s0, s11, s22 // (op 985) 

s_mul_i32 s18, s8, s0 // (op 985) 

s_mul_i32 s0, s1, s18 // (op 985) 

s_sub_u32 s18, s16, s0 // (op 985) 

s_mul_hi_i32 s0, s18, s12 // (op 985) 
s_add_i32 s19, s0, s18 // (op 985) 


s_ashr_i32 s0, s19, 31 // (op 985) 
s_cmp_eq_i32 s12, 0 // (op 985) 
s_mov_b32 s18, scc // (op 985) 
s_cmp_eq_i32 s18, 0 // (op 985) 
s_cselect_b32 s22, 0, -1 // (op 985) 

s_and_b32 s18, s44, 31 // (op 985) 
s_lshl_b32 s23, 1, s18 // (op 985) 
s_add_i32 s24, s23, s22 // (op 985) 


s_and_b32 s25, s0, s24 // (op 985) 


s_add_i32 s0, s19, s25 // (op 985) 


s_ashr_i32 s19, s0, s18 // (op 985) 


s_lshl_b32 s0, s44, 24 // (op 985) 
s_ashr_i32 s18, s0, 31 // (op 985) 

s_xor_b32 s0, s19, s18 // (op 985) 

s_sub_i32 s19, s0, s18 // (op 985) 


s_cmp_eq_i32 s1, 0 // (op 985) 
s_mov_b32 s0, scc // (op 985) 

s_cmp_eq_i32 s0, 0 // (op 985) 
s_cselect_b32 s1, s19, s47 // (op 985) 



v_lshl_add_u32 v16, s1, 6, v15 // (op 985) 


v_mul_lo_u32 v21, v16, s31 // (op 985) most significant: low of low * high
v_mul_hi_u32 v15, v16, s30 // (op 985) most significant: high of low * low
v_mul_lo_u32 v20, v16, s30 // (op 985) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 985) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v20 // (op 985) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 985) most significant half

v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 985) 


v_lshrrev_b64 v[18:19], 3, v[16:17] // (op 985) 

v_mov_b32 v15, v18 // (op 985) convert

s_mul_i32 s1, s30, s7 // (op 987) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 987) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 987) least significant
s_mul_hi_u32 s18, s30, s6 // (op 987) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 987) most significant: sum
s_add_u32 s1, s1, s18 // (op 987) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 987) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 987) 

s_mov_b32 s60, s32 // (op 987) reg expression
s_mov_b32 s61, s33 // (op 987) reg expression
s_mov_b32 s62, s0 // (op 987) reg expression
s_mov_b32 s63, 131072 // (op 987) call()

s_mul_hi_i32 s47, s16, s10 // (op 1030) 
v_lshrrev_b32 v16, 7, v1 // (op 1030) 
s_add_i32 s0, s47, s16 // (op 1030) 

v_and_b32 v17, 15, v1 // (op 1030) 
s_ashr_i32 s1, s0, 31 // (op 1030) 
v_lshl_add_u32 v18, v16, 4, v17 // (op 1030) 


s_cmp_eq_i32 s10, 0 // (op 1030) 
s_mov_b32 s47, scc // (op 1030) 
v_and_b32 v16, 63, v1 // (op 1030) 
s_cmp_eq_i32 s47, 0 // (op 1030) 
s_cselect_b32 s18, 0, -1 // (op 1030) 

v_lshrrev_b32 v17, 4, v16 // (op 1030) 

s_and_b32 s19, s13, 31 // (op 1030) 
s_lshl_b32 s47, 1, s19 // (op 1030) 
s_add_i32 s22, s47, s18 // (op 1030) 


s_and_b32 s18, s1, s22 // (op 1030) 


s_add_i32 s1, s0, s18 // (op 1030) 


s_ashr_i32 s0, s1, s19 // (op 1030) 


s_lshl_b32 s1, s13, 24 // (op 1030) 
s_ashr_i32 s47, s1, 31 // (op 1030) 

s_xor_b32 s1, s0, s47 // (op 1030) 

s_sub_i32 s0, s1, s47 // (op 1030) 


s_mul_hi_i32 s1, s0, s14 // (op 1030) 
s_add_i32 s47, s1, s0 // (op 1030) 

s_ashr_i32 s1, s47, 31 // (op 1030) 
s_cmp_eq_i32 s14, 0 // (op 1030) 
s_mov_b32 s18, scc // (op 1030) 
s_cmp_eq_i32 s18, 0 // (op 1030) 
s_cselect_b32 s19, 0, -1 // (op 1030) 

s_and_b32 s18, s15, 31 // (op 1030) 
s_lshl_b32 s22, 1, s18 // (op 1030) 
s_add_i32 s23, s22, s19 // (op 1030) 


s_and_b32 s19, s1, s23 // (op 1030) 


s_add_i32 s1, s47, s19 // (op 1030) 


s_ashr_i32 s19, s1, s18 // (op 1030) 


s_lshl_b32 s1, s15, 24 // (op 1030) 
s_ashr_i32 s18, s1, 31 // (op 1030) 

s_xor_b32 s1, s19, s18 // (op 1030) 

s_sub_i32 s19, s1, s18 // (op 1030) 


s_mul_i32 s1, s0, s11 // (op 1030) 

s_sub_u32 s0, s16, s1 // (op 1030) 

s_mul_i32 s1, s19, s11 // (op 1030) 
s_add_u32 s1, s1, s0 // (op 1030) 


s_mul_hi_i32 s0, s16, s9 // (op 1030) 
s_add_i32 s47, s0, s16 // (op 1030) 

s_ashr_i32 s0, s47, 31 // (op 1030) 
s_cmp_eq_i32 s9, 0 // (op 1030) 
s_mov_b32 s18, scc // (op 1030) 
s_cmp_eq_i32 s18, 0 // (op 1030) 
s_cselect_b32 s19, 0, -1 // (op 1030) 

s_and_b32 s18, s45, 31 // (op 1030) 
s_lshl_b32 s22, 1, s18 // (op 1030) 
s_add_i32 s23, s22, s19 // (op 1030) 


s_and_b32 s19, s0, s23 // (op 1030) 


s_add_i32 s0, s47, s19 // (op 1030) 


s_ashr_i32 s19, s0, s18 // (op 1030) 


s_lshl_b32 s0, s45, 24 // (op 1030) 
s_ashr_i32 s18, s0, 31 // (op 1030) 

s_xor_b32 s0, s19, s18 // (op 1030) 

s_sub_i32 s19, s0, s18 // (op 1030) 


s_add_u32 s22, s6, 64 // (op 1030) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1030) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1030) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1030) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 1030) 

s_mul_i32 s0, s11, s22 // (op 1030) 

s_mul_i32 s18, s8, s0 // (op 1030) 

s_mul_i32 s0, s19, s18 // (op 1030) 

s_sub_u32 s18, s16, s0 // (op 1030) 

s_mov_b32 s0, s18 // (op 1030) 
s_mul_hi_i32 s47, s0, s12 // (op 1030) 
s_add_i32 s22, s47, s0 // (op 1030) 


s_ashr_i32 s0, s22, 31 // (op 1030) 
s_cmp_eq_i32 s12, 0 // (op 1030) 
s_mov_b32 s47, scc // (op 1030) 
s_cmp_eq_i32 s47, 0 // (op 1030) 
s_cselect_b32 s23, 0, -1 // (op 1030) 

s_and_b32 s47, s44, 31 // (op 1030) 
s_lshl_b32 s24, 1, s47 // (op 1030) 
s_add_i32 s25, s24, s23 // (op 1030) 


s_and_b32 s23, s0, s25 // (op 1030) 


s_add_i32 s0, s22, s23 // (op 1030) 


s_ashr_i32 s22, s0, s47 // (op 1030) 


s_lshl_b32 s0, s44, 24 // (op 1030) 
s_ashr_i32 s47, s0, 31 // (op 1030) 

s_xor_b32 s0, s22, s47 // (op 1030) 

s_sub_i32 s22, s0, s47 // (op 1030) 


s_mul_i32 s0, s46, s11 // (op 1030) 
s_add_u32 s24, s20, 32 // (op 1030) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1030) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 1030) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 1030) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 5 // (op 1030) 

s_sub_i32 s23, s24, s0 // (op 1030) 

s_mul_i32 s47, s22, s23 // (op 1030) 


s_sub_u32 s22, s18, s47 // (op 1030) 


s_add_u32 s18, s22, s0 // (op 1030) 


s_cmp_eq_i32 s19, 0 // (op 1030) 
s_mov_b32 s0, scc // (op 1030) 

s_cmp_eq_i32 s0, 0 // (op 1030) 
s_cselect_b32 s19, s18, s1 // (op 1030) 



v_lshl_add_u32 v16, s19, 5, v18 // (op 1030) 


v_mul_lo_u32 v18, v16, s34 // (op 1030) 
v_add_u32 v18, v18, v17 // (op 1030) 


s_mul_i32 s1, s34, s21 // (op 1032) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 1032) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 1032) least significant
s_mul_hi_u32 s18, s34, s20 // (op 1032) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1032) most significant: sum
s_add_u32 s1, s1, s18 // (op 1032) most significant: sum


s_mov_b32 s64, s36 // (op 1032) reg expression
s_mov_b32 s65, s37 // (op 1032) reg expression
s_mov_b32 s66, s0 // (op 1032) reg expression
s_mov_b32 s67, 131072 // (op 1032) call()

s_mul_hi_i32 s47, s16, s10 // (op 1049) 
v_lshrrev_b32 v16, 7, v1 // (op 1049) 
s_add_i32 s0, s47, s16 // (op 1049) 

v_and_b32 v17, 15, v1 // (op 1049) 
s_ashr_i32 s1, s0, 31 // (op 1049) 
v_lshl_add_u32 v19, v16, 4, v17 // (op 1049) 


s_cmp_eq_i32 s10, 0 // (op 1049) 
s_mov_b32 s47, scc // (op 1049) 
v_and_b32 v16, 63, v1 // (op 1049) 
s_cmp_eq_i32 s47, 0 // (op 1049) 
s_cselect_b32 s18, 0, -1 // (op 1049) 

v_lshrrev_b32 v17, 4, v16 // (op 1049) 

s_and_b32 s19, s13, 31 // (op 1049) 
s_lshl_b32 s47, 1, s19 // (op 1049) 
s_add_i32 s22, s47, s18 // (op 1049) 


s_and_b32 s18, s1, s22 // (op 1049) 


s_add_i32 s1, s0, s18 // (op 1049) 


s_ashr_i32 s0, s1, s19 // (op 1049) 


s_lshl_b32 s1, s13, 24 // (op 1049) 
s_ashr_i32 s47, s1, 31 // (op 1049) 

s_xor_b32 s1, s0, s47 // (op 1049) 

s_sub_i32 s0, s1, s47 // (op 1049) 


s_mul_hi_i32 s1, s0, s14 // (op 1049) 
s_add_i32 s47, s1, s0 // (op 1049) 

s_ashr_i32 s1, s47, 31 // (op 1049) 
s_cmp_eq_i32 s14, 0 // (op 1049) 
s_mov_b32 s18, scc // (op 1049) 
s_cmp_eq_i32 s18, 0 // (op 1049) 
s_cselect_b32 s19, 0, -1 // (op 1049) 

s_and_b32 s18, s15, 31 // (op 1049) 
s_lshl_b32 s22, 1, s18 // (op 1049) 
s_add_i32 s23, s22, s19 // (op 1049) 


s_and_b32 s19, s1, s23 // (op 1049) 


s_add_i32 s1, s47, s19 // (op 1049) 


s_ashr_i32 s19, s1, s18 // (op 1049) 


s_lshl_b32 s1, s15, 24 // (op 1049) 
s_ashr_i32 s18, s1, 31 // (op 1049) 

s_xor_b32 s1, s19, s18 // (op 1049) 

s_sub_i32 s19, s1, s18 // (op 1049) 


s_mul_i32 s1, s0, s11 // (op 1049) 

s_sub_u32 s0, s16, s1 // (op 1049) 

s_mul_i32 s1, s19, s11 // (op 1049) 
s_add_u32 s1, s1, s0 // (op 1049) 


s_mul_hi_i32 s0, s16, s9 // (op 1049) 
s_add_i32 s47, s0, s16 // (op 1049) 

s_ashr_i32 s0, s47, 31 // (op 1049) 
s_cmp_eq_i32 s9, 0 // (op 1049) 
s_mov_b32 s18, scc // (op 1049) 
s_cmp_eq_i32 s18, 0 // (op 1049) 
s_cselect_b32 s19, 0, -1 // (op 1049) 

s_and_b32 s18, s45, 31 // (op 1049) 
s_lshl_b32 s22, 1, s18 // (op 1049) 
s_add_i32 s23, s22, s19 // (op 1049) 


s_and_b32 s19, s0, s23 // (op 1049) 


s_add_i32 s0, s47, s19 // (op 1049) 


s_ashr_i32 s19, s0, s18 // (op 1049) 


s_lshl_b32 s0, s45, 24 // (op 1049) 
s_ashr_i32 s18, s0, 31 // (op 1049) 

s_xor_b32 s0, s19, s18 // (op 1049) 

s_sub_i32 s19, s0, s18 // (op 1049) 


s_add_u32 s22, s6, 64 // (op 1049) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1049) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1049) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1049) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 1049) 

s_mul_i32 s0, s11, s22 // (op 1049) 

s_mul_i32 s18, s8, s0 // (op 1049) 

s_mul_i32 s0, s19, s18 // (op 1049) 

s_sub_u32 s18, s16, s0 // (op 1049) 

s_mov_b32 s0, s18 // (op 1049) 
s_mul_hi_i32 s47, s0, s12 // (op 1049) 
s_add_i32 s22, s47, s0 // (op 1049) 


s_ashr_i32 s0, s22, 31 // (op 1049) 
s_cmp_eq_i32 s12, 0 // (op 1049) 
s_mov_b32 s47, scc // (op 1049) 
s_cmp_eq_i32 s47, 0 // (op 1049) 
s_cselect_b32 s23, 0, -1 // (op 1049) 

s_and_b32 s47, s44, 31 // (op 1049) 
s_lshl_b32 s24, 1, s47 // (op 1049) 
s_add_i32 s25, s24, s23 // (op 1049) 


s_and_b32 s23, s0, s25 // (op 1049) 


s_add_i32 s0, s22, s23 // (op 1049) 


s_ashr_i32 s22, s0, s47 // (op 1049) 


s_lshl_b32 s0, s44, 24 // (op 1049) 
s_ashr_i32 s47, s0, 31 // (op 1049) 

s_xor_b32 s0, s22, s47 // (op 1049) 

s_sub_i32 s22, s0, s47 // (op 1049) 


s_mul_i32 s0, s46, s11 // (op 1049) 
s_add_u32 s24, s20, 32 // (op 1049) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1049) End of Int64 add, unlocking SCC
s_sub_u32 s68, s24, 1 // (op 1049) least significant half; sets scc
s_subb_u32 s69, s25, 0 // (op 1049) most significant half; uses scc

s_ashr_i64 s[24:25], s[68:69], 5 // (op 1049) 

s_sub_i32 s23, s24, s0 // (op 1049) 

s_mul_i32 s47, s22, s23 // (op 1049) 


s_sub_u32 s22, s18, s47 // (op 1049) 


s_add_u32 s18, s22, s0 // (op 1049) 


s_cmp_eq_i32 s19, 0 // (op 1049) 
s_mov_b32 s0, scc // (op 1049) 

s_cmp_eq_i32 s0, 0 // (op 1049) 
s_cselect_b32 s19, s18, s1 // (op 1049) 



v_lshl_add_u32 v16, s19, 5, v19 // (op 1049) 


v_mul_lo_u32 v19, v16, s34 // (op 1049) 
v_add_u32 v19, v19, v17 // (op 1049) 


s_mul_i32 s1, s34, s21 // (op 1051) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 1051) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 1051) least significant
s_mul_hi_u32 s18, s34, s20 // (op 1051) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1051) most significant: sum
s_add_u32 s1, s1, s18 // (op 1051) most significant: sum


s_mov_b32 s64, s36 // (op 1051) reg expression
s_mov_b32 s65, s37 // (op 1051) reg expression
s_mov_b32 s66, s0 // (op 1051) reg expression
s_mov_b32 s67, 131072 // (op 1051) call()

s_mul_hi_i32 s47, s16, s10 // (op 1094) 
v_lshrrev_b32 v16, 2, v1 // (op 1094) 
s_add_i32 s0, s47, s16 // (op 1094) 

v_and_b32 v17, -4, v1 // (op 1094) 
s_ashr_i32 s1, s0, 31 // (op 1094) 
v_sub_u32 v20, v1, v17 // (op 1094) 

s_cmp_eq_i32 s10, 0 // (op 1094) 
s_mov_b32 s47, scc // (op 1094) 
s_cmp_eq_i32 s47, 0 // (op 1094) 
s_cselect_b32 s18, 0, -1 // (op 1094) 

s_and_b32 s19, s13, 31 // (op 1094) 
s_lshl_b32 s47, 1, s19 // (op 1094) 
s_add_i32 s22, s47, s18 // (op 1094) 


s_and_b32 s18, s1, s22 // (op 1094) 


s_add_i32 s1, s0, s18 // (op 1094) 


s_ashr_i32 s0, s1, s19 // (op 1094) 


s_lshl_b32 s1, s13, 24 // (op 1094) 
s_ashr_i32 s47, s1, 31 // (op 1094) 

s_xor_b32 s1, s0, s47 // (op 1094) 

s_sub_i32 s0, s1, s47 // (op 1094) 


s_mul_hi_i32 s1, s0, s14 // (op 1094) 
s_add_i32 s47, s1, s0 // (op 1094) 

s_ashr_i32 s1, s47, 31 // (op 1094) 
s_cmp_eq_i32 s14, 0 // (op 1094) 
s_mov_b32 s18, scc // (op 1094) 
s_cmp_eq_i32 s18, 0 // (op 1094) 
s_cselect_b32 s19, 0, -1 // (op 1094) 

s_and_b32 s18, s15, 31 // (op 1094) 
s_lshl_b32 s22, 1, s18 // (op 1094) 
s_add_i32 s23, s22, s19 // (op 1094) 


s_and_b32 s19, s1, s23 // (op 1094) 


s_add_i32 s1, s47, s19 // (op 1094) 


s_ashr_i32 s19, s1, s18 // (op 1094) 


s_lshl_b32 s1, s15, 24 // (op 1094) 
s_ashr_i32 s18, s1, 31 // (op 1094) 

s_xor_b32 s1, s19, s18 // (op 1094) 

s_sub_i32 s19, s1, s18 // (op 1094) 


s_add_u32 s22, s6, 64 // (op 1094) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1094) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1094) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1094) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 1094) 

s_mul_i32 s1, s19, s22 // (op 1094) 

s_sub_i32 s47, s0, s1 // (op 1094) 


s_mul_hi_i32 s0, s16, s9 // (op 1094) 
s_add_i32 s1, s0, s16 // (op 1094) 

s_ashr_i32 s0, s1, 31 // (op 1094) 
s_cmp_eq_i32 s9, 0 // (op 1094) 
s_mov_b32 s18, scc // (op 1094) 
s_cmp_eq_i32 s18, 0 // (op 1094) 
s_cselect_b32 s19, 0, -1 // (op 1094) 

s_and_b32 s18, s45, 31 // (op 1094) 
s_lshl_b32 s24, 1, s18 // (op 1094) 
s_add_i32 s25, s24, s19 // (op 1094) 


s_and_b32 s19, s0, s25 // (op 1094) 


s_add_i32 s0, s1, s19 // (op 1094) 


s_ashr_i32 s1, s0, s18 // (op 1094) 


s_lshl_b32 s0, s45, 24 // (op 1094) 
s_ashr_i32 s18, s0, 31 // (op 1094) 

s_xor_b32 s0, s1, s18 // (op 1094) 

s_sub_i32 s1, s0, s18 // (op 1094) 


s_mul_i32 s0, s11, s22 // (op 1094) 

s_mul_i32 s18, s8, s0 // (op 1094) 

s_mul_i32 s0, s1, s18 // (op 1094) 

s_sub_u32 s18, s16, s0 // (op 1094) 

s_mul_hi_i32 s0, s18, s12 // (op 1094) 
s_add_i32 s19, s0, s18 // (op 1094) 


s_ashr_i32 s0, s19, 31 // (op 1094) 
s_cmp_eq_i32 s12, 0 // (op 1094) 
s_mov_b32 s18, scc // (op 1094) 
s_cmp_eq_i32 s18, 0 // (op 1094) 
s_cselect_b32 s22, 0, -1 // (op 1094) 

s_and_b32 s18, s44, 31 // (op 1094) 
s_lshl_b32 s23, 1, s18 // (op 1094) 
s_add_i32 s24, s23, s22 // (op 1094) 


s_and_b32 s25, s0, s24 // (op 1094) 


s_add_i32 s0, s19, s25 // (op 1094) 


s_ashr_i32 s19, s0, s18 // (op 1094) 


s_lshl_b32 s0, s44, 24 // (op 1094) 
s_ashr_i32 s18, s0, 31 // (op 1094) 

s_xor_b32 s0, s19, s18 // (op 1094) 

s_sub_i32 s19, s0, s18 // (op 1094) 


s_cmp_eq_i32 s1, 0 // (op 1094) 
s_mov_b32 s0, scc // (op 1094) 

s_cmp_eq_i32 s0, 0 // (op 1094) 
s_cselect_b32 s1, s19, s47 // (op 1094) 



v_lshl_add_u32 v17, s1, 6, v16 // (op 1094) 


v_mul_lo_u32 v16, v17, s38 // (op 1094) 

v_add_u32 v17, v20, v16 // (op 1094) 


s_mul_i32 s1, s38, s7 // (op 1096) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 1096) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 1096) least significant
s_mul_hi_u32 s18, s38, s6 // (op 1096) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1096) most significant: sum
s_add_u32 s1, s1, s18 // (op 1096) most significant: sum


s_mov_b32 s68, s40 // (op 1096) reg expression
s_mov_b32 s69, s41 // (op 1096) reg expression
s_mov_b32 s70, s0 // (op 1096) reg expression
s_mov_b32 s71, 131072 // (op 1096) call()

s_mul_hi_i32 s47, s16, s10 // (op 1113) 
v_lshrrev_b32 v16, 2, v1 // (op 1113) 
s_add_i32 s0, s47, s16 // (op 1113) 

v_and_b32 v20, -4, v1 // (op 1113) 
s_ashr_i32 s1, s0, 31 // (op 1113) 
v_sub_u32 v21, v1, v20 // (op 1113) 

s_cmp_eq_i32 s10, 0 // (op 1113) 
s_mov_b32 s47, scc // (op 1113) 
s_cmp_eq_i32 s47, 0 // (op 1113) 
s_cselect_b32 s18, 0, -1 // (op 1113) 

s_and_b32 s19, s13, 31 // (op 1113) 
s_lshl_b32 s47, 1, s19 // (op 1113) 
s_add_i32 s22, s47, s18 // (op 1113) 


s_and_b32 s18, s1, s22 // (op 1113) 


s_add_i32 s1, s0, s18 // (op 1113) 


s_ashr_i32 s0, s1, s19 // (op 1113) 


s_lshl_b32 s1, s13, 24 // (op 1113) 
s_ashr_i32 s47, s1, 31 // (op 1113) 

s_xor_b32 s1, s0, s47 // (op 1113) 

s_sub_i32 s0, s1, s47 // (op 1113) 


s_mul_hi_i32 s1, s0, s14 // (op 1113) 
s_add_i32 s47, s1, s0 // (op 1113) 

s_ashr_i32 s1, s47, 31 // (op 1113) 
s_cmp_eq_i32 s14, 0 // (op 1113) 
s_mov_b32 s18, scc // (op 1113) 
s_cmp_eq_i32 s18, 0 // (op 1113) 
s_cselect_b32 s19, 0, -1 // (op 1113) 

s_and_b32 s18, s15, 31 // (op 1113) 
s_lshl_b32 s22, 1, s18 // (op 1113) 
s_add_i32 s23, s22, s19 // (op 1113) 


s_and_b32 s19, s1, s23 // (op 1113) 


s_add_i32 s1, s47, s19 // (op 1113) 


s_ashr_i32 s19, s1, s18 // (op 1113) 


s_lshl_b32 s1, s15, 24 // (op 1113) 
s_ashr_i32 s18, s1, 31 // (op 1113) 

s_xor_b32 s1, s19, s18 // (op 1113) 

s_sub_i32 s19, s1, s18 // (op 1113) 


s_add_u32 s22, s6, 64 // (op 1113) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1113) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1113) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1113) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 1113) 

s_mul_i32 s1, s19, s22 // (op 1113) 

s_sub_i32 s47, s0, s1 // (op 1113) 


s_mul_hi_i32 s0, s16, s9 // (op 1113) 
s_add_i32 s1, s0, s16 // (op 1113) 

s_ashr_i32 s0, s1, 31 // (op 1113) 
s_cmp_eq_i32 s9, 0 // (op 1113) 
s_mov_b32 s18, scc // (op 1113) 
s_cmp_eq_i32 s18, 0 // (op 1113) 
s_cselect_b32 s19, 0, -1 // (op 1113) 

s_and_b32 s18, s45, 31 // (op 1113) 
s_lshl_b32 s24, 1, s18 // (op 1113) 
s_add_i32 s25, s24, s19 // (op 1113) 


s_and_b32 s19, s0, s25 // (op 1113) 


s_add_i32 s0, s1, s19 // (op 1113) 


s_ashr_i32 s1, s0, s18 // (op 1113) 


s_lshl_b32 s0, s45, 24 // (op 1113) 
s_ashr_i32 s18, s0, 31 // (op 1113) 

s_xor_b32 s0, s1, s18 // (op 1113) 

s_sub_i32 s1, s0, s18 // (op 1113) 


s_mul_i32 s0, s11, s22 // (op 1113) 

s_mul_i32 s18, s8, s0 // (op 1113) 

s_mul_i32 s0, s1, s18 // (op 1113) 

s_sub_u32 s18, s16, s0 // (op 1113) 

s_mul_hi_i32 s0, s18, s12 // (op 1113) 
s_add_i32 s19, s0, s18 // (op 1113) 


s_ashr_i32 s0, s19, 31 // (op 1113) 
s_cmp_eq_i32 s12, 0 // (op 1113) 
s_mov_b32 s18, scc // (op 1113) 
s_cmp_eq_i32 s18, 0 // (op 1113) 
s_cselect_b32 s22, 0, -1 // (op 1113) 

s_and_b32 s18, s44, 31 // (op 1113) 
s_lshl_b32 s23, 1, s18 // (op 1113) 
s_add_i32 s24, s23, s22 // (op 1113) 


s_and_b32 s25, s0, s24 // (op 1113) 


s_add_i32 s0, s19, s25 // (op 1113) 


s_ashr_i32 s19, s0, s18 // (op 1113) 


s_lshl_b32 s0, s44, 24 // (op 1113) 
s_ashr_i32 s18, s0, 31 // (op 1113) 

s_xor_b32 s0, s19, s18 // (op 1113) 

s_sub_i32 s19, s0, s18 // (op 1113) 


s_cmp_eq_i32 s1, 0 // (op 1113) 
s_mov_b32 s0, scc // (op 1113) 

s_cmp_eq_i32 s0, 0 // (op 1113) 
s_cselect_b32 s1, s19, s47 // (op 1113) 



v_lshl_add_u32 v20, s1, 6, v16 // (op 1113) 


v_mul_lo_u32 v16, v20, s38 // (op 1113) 

v_add_u32 v20, v21, v16 // (op 1113) 


s_mul_i32 s1, s38, s7 // (op 1115) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 1115) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 1115) least significant
s_mul_hi_u32 s18, s38, s6 // (op 1115) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1115) most significant: sum
s_add_u32 s1, s1, s18 // (op 1115) most significant: sum


s_mov_b32 s68, s40 // (op 1115) reg expression
s_mov_b32 s69, s41 // (op 1115) reg expression
s_mov_b32 s70, s0 // (op 1115) reg expression
s_mov_b32 s71, 131072 // (op 1115) call()




buffer_load_ubyte v16, v18, s[64:67], 0 offen offset: 0 // (op 769) Load value



buffer_load_dwordx4 v[22:25], v14, s[60:63], 0 offen // (op 773) Load value



buffer_load_ubyte v21, v17, s[68:71], 0 offen offset: 0 // (op 776) Load value



buffer_load_dwordx4 v[26:29], v12, s[48:51], 0 offen // (op 779) Load value



buffer_load_ubyte v30, v18, s[64:67], 0 offen offset:4 // (op 783) Load value




buffer_load_dwordx4 v[32:35], v14, s[60:63], 0 offen offset:64 // (op 787) Load value




buffer_load_ubyte v14, v17, s[68:71], 0 offen offset:4 // (op 790) Load value




buffer_load_dwordx4 v[36:39], v12, s[48:51], 0 offen offset:128 // (op 793) Load value

v_and_b32 v12, -4, v1 // (op 1233) 
v_sub_u32 v31, v1, v12 // (op 1233) 

v_lshlrev_b32 v12, 5, v31 // (op 1233) 

v_lshrrev_b32 v31, 2, v1 // (op 1233) 
v_lshl_add_u32 v17, v31, 7, v12 // (op 1233) 


v_lshlrev_b32 v12, 2, v17 // (op 1233) 

v_lshrrev_b32 v31, 3, v12 // (op 1233) 




s_waitcnt vmcnt(6)
ds_write_b128 v31, v[22:25] // (op 795) Store local data 

v_and_b32 v12, -4, v1 // (op 1291) 
v_sub_u32 v31, v1, v12 // (op 1291) 

v_and_b32 v12, 4294967292, v1 // (op 1291) 
v_add_u32 v17, v12, v31 // (op 1291) 





s_waitcnt vmcnt(5)
ds_write_b8 v17, v21 offset:4096 // (op 796) Store local data 

v_and_b32 v12, -8, v1 // (op 1180) 
v_sub_u32 v31, v1, v12 // (op 1180) 

v_lshlrev_b32 v12, 4, v31 // (op 1180) 

v_lshrrev_b32 v31, 3, v1 // (op 1180) 
v_lshl_add_u32 v17, v31, 7, v12 // (op 1180) 





s_waitcnt vmcnt(4)
ds_write_b128 v17, v[26:29] offset:4352 // (op 797) Store local data 

v_and_b32 v12, -8, v1 // (op 1151) 
v_sub_u32 v31, v1, v12 // (op 1151) 

v_lshlrev_b32 v12, 4, v31 // (op 1151) 

v_lshrrev_b32 v31, 3, v1 // (op 1151) 
v_lshl_add_u32 v17, v31, 7, v12 // (op 1151) 


v_and_b32 v12, -4, v1 // (op 1204) 
v_sub_u32 v18, v1, v12 // (op 1204) 

v_lshlrev_b32 v12, 5, v18 // (op 1204) 

v_lshrrev_b32 v18, 2, v1 // (op 1204) 
v_lshl_add_u32 v31, v18, 7, v12 // (op 1204) 


v_lshlrev_b32 v12, 2, v31 // (op 1204) 

v_lshrrev_b32 v18, 3, v12 // (op 1204) 

v_and_b32 v12, -4, v1 // (op 1262) 
v_sub_u32 v31, v1, v12 // (op 1262) 

v_and_b32 v12, 4294967292, v1 // (op 1262) 
v_add_u32 v40, v12, v31 // (op 1262) 


v_and_b32 v12, -8, v1 // (op 1338) 
v_sub_u32 v31, v1, v12 // (op 1338) 

v_lshlrev_b32 v12, 4, v31 // (op 1338) 

v_lshrrev_b32 v31, 3, v1 // (op 1338) 
v_lshl_add_u32 v41, v31, 7, v12 // (op 1338) 


v_and_b32 v12, -4, v1 // (op 1362) 
v_sub_u32 v31, v1, v12 // (op 1362) 

v_lshlrev_b32 v12, 5, v31 // (op 1362) 

v_lshrrev_b32 v31, 2, v1 // (op 1362) 
v_lshl_add_u32 v42, v31, 7, v12 // (op 1362) 


v_lshlrev_b32 v12, 2, v42 // (op 1362) 

v_lshrrev_b32 v31, 3, v12 // (op 1362) 

v_and_b32 v12, -4, v1 // (op 1391) 
v_sub_u32 v42, v1, v12 // (op 1391) 

v_and_b32 v12, 4294967292, v1 // (op 1391) 
v_add_u32 v43, v12, v42 // (op 1391) 


s_mov_b32 s47, 0 // (op 312) call()
s_ashr_i64 s[0:1], s[42:43], 7 // (op 311) 
s_mov_b32 s18, s0 // (op 311) convert

s_and_b32 s19, s18, 4294967294 // (op 311) 

s_cmp_lt_u32 s47, s19 // (op 311) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopBottom_KLoop_311 // (op 311) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopBottom_KLoop_311 if false)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopTop_KLoop_311:
s_add_u32 s47, s47, 2 // (op 313) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1449) 



ds_read_b128 v[44:47], v3 offset:4352 // (op 67) Load local data 
ds_read_b128 v[48:51], v3 offset:4416 // (op 67) Load local data 
s_barrier  // (op 1462) 



ds_read_b128 v[52:55], v7 // (op 72) Load local data 
s_barrier  // (op 1476) 



ds_read_u8 v12, v0 offset:4096 // (op 81) Load local data 



ds_read_b128 v[56:59], v7 offset:2048 // (op 343) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:51], v[52:55], a[0:3], v16, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 84) 





ds_read_u8 v12, v0 offset:4224 // (op 346) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[44:51], v[56:59], a[4:7], v16, v12 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 347) 





v_mov_b32 v12, v19 // (op 77) 

buffer_load_ubyte v16, v12, s[64:67], 0 offen offset:8 // (op 77) Load value


v_mov_b32 v42, v15 // (op 10) 

buffer_load_dwordx4 v[22:25], v42, s[60:63], 0 offen offset:128 // (op 10) Load value


v_mov_b32 v44, v20 // (op 21) 

buffer_load_ubyte v21, v44, s[68:71], 0 offen offset:8 // (op 21) Load value


v_mov_b32 v45, v13 // (op 4) 

buffer_load_dwordx4 v[26:29], v45, s[48:51], 0 offen offset:256 // (op 4) Load value
s_barrier  // (op 1499) 



s_waitcnt vmcnt(6)
ds_write_b128 v31, v[32:35] offset:8448 // (op 239) Store local data 
s_barrier  // (op 1505) 



s_waitcnt vmcnt(5)
ds_write_b8 v43, v14 offset:12544 // (op 240) Store local data 
s_barrier  // (op 1493) 



s_waitcnt vmcnt(4)
ds_write_b128 v41, v[36:39] offset:12800 // (op 229) Store local data 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 1490) 



ds_read_b128 v[46:49], v4 offset:12800 // (op 231) Load local data 
ds_read_b128 v[50:53], v4 offset:12864 // (op 231) Load local data 
s_barrier  // (op 1496) 
s_barrier  // (op 1546) 



ds_read_b128 v[54:57], v11 offset:8448 // (op 233) Load local data 
s_barrier  // (op 1502) 



ds_read_u8 v58, v10 offset:12544 // (op 237) Load local data 



ds_read_b128 v[60:63], v11 offset:10496 // (op 352) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[46:53], v[54:57], a[0:3], v30, v58 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 238) 





s_barrier  // (op 1580) 
ds_read_u8 v54, v10 offset:12672 // (op 355) Load local data 
s_barrier  // (op 1584) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[46:53], v[60:63], a[4:7], v30, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 348) 






buffer_load_ubyte v30, v12, s[64:67], 0 offen offset:12 // (op 235) Load value




v_add_u32 v19, 8, v19 // (op 1041) 
buffer_load_dwordx4 v[32:35], v42, s[60:63], 0 offen offset:192 // (op 241) Load value




v_add_u32 v15, 128, v15 // (op 977) 
buffer_load_ubyte v14, v44, s[68:71], 0 offen offset:12 // (op 242) Load value




v_add_u32 v20, 8, v20 // (op 1105) 
buffer_load_dwordx4 v[36:39], v45, s[48:51], 0 offen offset:384 // (op 228) Load value

s_barrier  // (op 1465) 
v_add_u32 v13, 256, v13 // (op 913) 



s_waitcnt vmcnt(6)
ds_write_b128 v18, v[22:25] // (op 13) Store local data 
s_barrier  // (op 1479) 



s_waitcnt vmcnt(5)
ds_write_b8 v40, v21 offset:4096 // (op 26) Store local data 
s_barrier  // (op 1452) 



s_waitcnt vmcnt(4)
ds_write_b128 v17, v[26:29] offset:4352 // (op 6) Store local data 
s_ashr_i64 s[0:1], s[42:43], 7 // (op 311) 
s_mov_b32 s18, s0 // (op 311) convert

s_and_b32 s19, s18, 4294967294 // (op 311) 

s_cmp_lt_u32 s47, s19 // (op 311) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopTop_KLoop_311 // (op 311) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopTop_KLoop_311 if true)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopBottom_KLoop_311:















s_mul_hi_i32 s47, s16, s10 // (op 875) 
v_lshrrev_b32 v31, 3, v1 // (op 875) 
s_add_i32 s0, s47, s16 // (op 875) 

v_and_b32 v12, -8, v1 // (op 875) 
s_ashr_i32 s1, s0, 31 // (op 875) 
v_sub_u32 v13, v1, v12 // (op 875) 

s_cmp_eq_i32 s10, 0 // (op 875) 
s_mov_b32 s47, scc // (op 875) 
v_lshlrev_b32 v12, 4, v13 // (op 875) 

s_cmp_eq_i32 s47, 0 // (op 875) 
s_cselect_b32 s18, 0, -1 // (op 875) 

s_and_b32 s19, s13, 31 // (op 875) 
s_lshl_b32 s47, 1, s19 // (op 875) 
s_add_i32 s22, s47, s18 // (op 875) 


s_and_b32 s18, s1, s22 // (op 875) 


s_add_i32 s1, s0, s18 // (op 875) 


s_ashr_i32 s0, s1, s19 // (op 875) 


s_lshl_b32 s1, s13, 24 // (op 875) 
s_ashr_i32 s47, s1, 31 // (op 875) 

s_xor_b32 s1, s0, s47 // (op 875) 

s_sub_i32 s0, s1, s47 // (op 875) 


s_mul_hi_i32 s1, s0, s14 // (op 875) 
s_add_i32 s47, s1, s0 // (op 875) 

s_ashr_i32 s1, s47, 31 // (op 875) 
s_cmp_eq_i32 s14, 0 // (op 875) 
s_mov_b32 s18, scc // (op 875) 
s_cmp_eq_i32 s18, 0 // (op 875) 
s_cselect_b32 s19, 0, -1 // (op 875) 

s_and_b32 s18, s15, 31 // (op 875) 
s_lshl_b32 s22, 1, s18 // (op 875) 
s_add_i32 s23, s22, s19 // (op 875) 


s_and_b32 s19, s1, s23 // (op 875) 


s_add_i32 s1, s47, s19 // (op 875) 


s_ashr_i32 s19, s1, s18 // (op 875) 


s_lshl_b32 s1, s15, 24 // (op 875) 
s_ashr_i32 s18, s1, 31 // (op 875) 

s_xor_b32 s1, s19, s18 // (op 875) 

s_sub_i32 s19, s1, s18 // (op 875) 


s_mul_i32 s1, s0, s11 // (op 875) 

s_sub_u32 s0, s16, s1 // (op 875) 

s_mul_i32 s1, s19, s11 // (op 875) 
s_add_u32 s1, s1, s0 // (op 875) 


s_mul_hi_i32 s0, s16, s9 // (op 875) 
s_add_i32 s47, s0, s16 // (op 875) 

s_ashr_i32 s0, s47, 31 // (op 875) 
s_cmp_eq_i32 s9, 0 // (op 875) 
s_mov_b32 s18, scc // (op 875) 
s_cmp_eq_i32 s18, 0 // (op 875) 
s_cselect_b32 s19, 0, -1 // (op 875) 

s_and_b32 s18, s45, 31 // (op 875) 
s_lshl_b32 s22, 1, s18 // (op 875) 
s_add_i32 s23, s22, s19 // (op 875) 


s_and_b32 s19, s0, s23 // (op 875) 


s_add_i32 s0, s47, s19 // (op 875) 


s_ashr_i32 s19, s0, s18 // (op 875) 


s_lshl_b32 s0, s45, 24 // (op 875) 
s_ashr_i32 s18, s0, 31 // (op 875) 

s_xor_b32 s0, s19, s18 // (op 875) 

s_sub_i32 s19, s0, s18 // (op 875) 


s_add_u32 s22, s6, 64 // (op 875) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 875) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 875) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 875) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 875) 

s_mul_i32 s0, s11, s22 // (op 875) 

s_mul_i32 s18, s8, s0 // (op 875) 

s_mul_i32 s0, s19, s18 // (op 875) 

s_sub_u32 s18, s16, s0 // (op 875) 

s_mov_b32 s0, s18 // (op 875) 
s_mul_hi_i32 s47, s0, s12 // (op 875) 
s_add_i32 s22, s47, s0 // (op 875) 


s_ashr_i32 s0, s22, 31 // (op 875) 
s_cmp_eq_i32 s12, 0 // (op 875) 
s_mov_b32 s47, scc // (op 875) 
s_cmp_eq_i32 s47, 0 // (op 875) 
s_cselect_b32 s23, 0, -1 // (op 875) 

s_and_b32 s47, s44, 31 // (op 875) 
s_lshl_b32 s24, 1, s47 // (op 875) 
s_add_i32 s25, s24, s23 // (op 875) 


s_and_b32 s23, s0, s25 // (op 875) 


s_add_i32 s0, s22, s23 // (op 875) 


s_ashr_i32 s22, s0, s47 // (op 875) 


s_lshl_b32 s0, s44, 24 // (op 875) 
s_ashr_i32 s47, s0, 31 // (op 875) 

s_xor_b32 s0, s22, s47 // (op 875) 

s_sub_i32 s22, s0, s47 // (op 875) 


s_mul_i32 s0, s46, s11 // (op 875) 
s_add_u32 s24, s20, 32 // (op 875) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 875) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 875) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 875) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 5 // (op 875) 

s_sub_i32 s23, s24, s0 // (op 875) 

s_mul_i32 s47, s22, s23 // (op 875) 


s_sub_u32 s22, s18, s47 // (op 875) 


s_add_u32 s18, s22, s0 // (op 875) 


s_cmp_eq_i32 s19, 0 // (op 875) 
s_mov_b32 s0, scc // (op 875) 

s_cmp_eq_i32 s0, 0 // (op 875) 
s_cselect_b32 s19, s18, s1 // (op 875) 



v_lshl_add_u32 v13, s19, 5, v31 // (op 875) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 875) 
s_lshr_b32 s47, s0, 1 // (op 875) 

v_lshl_add_u32 v31, s47, 8, v12 // (op 875) 


v_mul_lo_u32 v12, v13, s26 // (op 875) 
v_add_u32 v12, v12, v31 // (op 875) 


s_mul_i32 s1, s26, s21 // (op 877) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 877) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 877) least significant
s_mul_hi_u32 s18, s26, s20 // (op 877) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 877) most significant: sum
s_add_u32 s1, s1, s18 // (op 877) most significant: sum


s_mov_b32 s48, s28 // (op 877) reg expression
s_mov_b32 s49, s29 // (op 877) reg expression
s_mov_b32 s50, s0 // (op 877) reg expression
s_mov_b32 s51, 131072 // (op 877) call()

s_mul_hi_i32 s47, s16, s10 // (op 943) 
v_lshrrev_b32 v13, 2, v1 // (op 943) 
s_add_i32 s0, s47, s16 // (op 943) 

v_and_b32 v31, -4, v1 // (op 943) 
s_ashr_i32 s1, s0, 31 // (op 943) 
v_sub_u32 v15, v1, v31 // (op 943) 

s_cmp_eq_i32 s10, 0 // (op 943) 
s_mov_b32 s47, scc // (op 943) 
v_lshlrev_b32 v31, 5, v15 // (op 943) 

s_cmp_eq_i32 s47, 0 // (op 943) 
s_cselect_b32 s18, 0, -1 // (op 943) 

s_and_b32 s19, s13, 31 // (op 943) 
s_lshl_b32 s47, 1, s19 // (op 943) 
s_add_i32 s22, s47, s18 // (op 943) 


s_and_b32 s18, s1, s22 // (op 943) 


s_add_i32 s1, s0, s18 // (op 943) 


s_ashr_i32 s0, s1, s19 // (op 943) 


s_lshl_b32 s1, s13, 24 // (op 943) 
s_ashr_i32 s47, s1, 31 // (op 943) 

s_xor_b32 s1, s0, s47 // (op 943) 

s_sub_i32 s0, s1, s47 // (op 943) 


s_mul_hi_i32 s1, s0, s14 // (op 943) 
s_add_i32 s47, s1, s0 // (op 943) 

s_ashr_i32 s1, s47, 31 // (op 943) 
s_cmp_eq_i32 s14, 0 // (op 943) 
s_mov_b32 s18, scc // (op 943) 
s_cmp_eq_i32 s18, 0 // (op 943) 
s_cselect_b32 s19, 0, -1 // (op 943) 

s_and_b32 s18, s15, 31 // (op 943) 
s_lshl_b32 s22, 1, s18 // (op 943) 
s_add_i32 s23, s22, s19 // (op 943) 


s_and_b32 s19, s1, s23 // (op 943) 


s_add_i32 s1, s47, s19 // (op 943) 


s_ashr_i32 s19, s1, s18 // (op 943) 


s_lshl_b32 s1, s15, 24 // (op 943) 
s_ashr_i32 s18, s1, 31 // (op 943) 

s_xor_b32 s1, s19, s18 // (op 943) 

s_sub_i32 s19, s1, s18 // (op 943) 


s_add_u32 s22, s6, 64 // (op 943) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 943) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 943) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 943) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 943) 

s_mul_i32 s1, s19, s22 // (op 943) 

s_sub_i32 s47, s0, s1 // (op 943) 


s_mul_hi_i32 s0, s16, s9 // (op 943) 
s_add_i32 s1, s0, s16 // (op 943) 

s_ashr_i32 s0, s1, 31 // (op 943) 
s_cmp_eq_i32 s9, 0 // (op 943) 
s_mov_b32 s18, scc // (op 943) 
s_cmp_eq_i32 s18, 0 // (op 943) 
s_cselect_b32 s19, 0, -1 // (op 943) 

s_and_b32 s18, s45, 31 // (op 943) 
s_lshl_b32 s24, 1, s18 // (op 943) 
s_add_i32 s25, s24, s19 // (op 943) 


s_and_b32 s19, s0, s25 // (op 943) 


s_add_i32 s0, s1, s19 // (op 943) 


s_ashr_i32 s1, s0, s18 // (op 943) 


s_lshl_b32 s0, s45, 24 // (op 943) 
s_ashr_i32 s18, s0, 31 // (op 943) 

s_xor_b32 s0, s1, s18 // (op 943) 

s_sub_i32 s1, s0, s18 // (op 943) 


s_mul_i32 s0, s11, s22 // (op 943) 

s_mul_i32 s18, s8, s0 // (op 943) 

s_mul_i32 s0, s1, s18 // (op 943) 

s_sub_u32 s18, s16, s0 // (op 943) 

s_mul_hi_i32 s0, s18, s12 // (op 943) 
s_add_i32 s19, s0, s18 // (op 943) 


s_ashr_i32 s0, s19, 31 // (op 943) 
s_cmp_eq_i32 s12, 0 // (op 943) 
s_mov_b32 s18, scc // (op 943) 
s_cmp_eq_i32 s18, 0 // (op 943) 
s_cselect_b32 s22, 0, -1 // (op 943) 

s_and_b32 s18, s44, 31 // (op 943) 
s_lshl_b32 s23, 1, s18 // (op 943) 
s_add_i32 s24, s23, s22 // (op 943) 


s_and_b32 s25, s0, s24 // (op 943) 


s_add_i32 s0, s19, s25 // (op 943) 


s_ashr_i32 s19, s0, s18 // (op 943) 


s_lshl_b32 s0, s44, 24 // (op 943) 
s_ashr_i32 s18, s0, 31 // (op 943) 

s_xor_b32 s0, s19, s18 // (op 943) 

s_sub_i32 s19, s0, s18 // (op 943) 


s_cmp_eq_i32 s1, 0 // (op 943) 
s_mov_b32 s0, scc // (op 943) 

s_cmp_eq_i32 s0, 0 // (op 943) 
s_cselect_b32 s1, s19, s47 // (op 943) 



v_lshl_add_u32 v15, s1, 6, v13 // (op 943) 


v_mul_lo_u32 v17, v15, s31 // (op 943) most significant: low of low * high
v_mul_hi_u32 v13, v15, s30 // (op 943) most significant: high of low * low
v_mul_lo_u32 v16, v15, s30 // (op 943) least significant: low of low * low
v_add_u32 v17, v17, v13 // (op 943) most significant: sum


s_ashr_i64 s[0:1], s[42:43], 7 // (op 943) 
s_lshr_b32 s47, s0, 1 // (op 943) 

v_lshl_add_u32 v13, s47, 8, v31 // (op 943) 


v_mov_b32 v19, 0 // (op 943) convert
v_mov_b32 v18, v13 // (op 943) convert

v_add_co_u32 v20, s[0:1], v18, v16 // (op 943) least significant half
v_addc_co_u32 v21, s[0:1], v19, v17, s[0:1] // (op 943) most significant half

v_lshlrev_b64 v[20:21], 2, v[20:21] // (op 943) 


v_lshrrev_b64 v[18:19], 3, v[20:21] // (op 943) 

v_mov_b32 v13, v18 // (op 943) convert

s_mul_i32 s1, s30, s7 // (op 945) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 945) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 945) least significant
s_mul_hi_u32 s18, s30, s6 // (op 945) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 945) most significant: sum
s_add_u32 s1, s1, s18 // (op 945) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 945) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 945) 

s_mov_b32 s60, s32 // (op 945) reg expression
s_mov_b32 s61, s33 // (op 945) reg expression
s_mov_b32 s62, s0 // (op 945) reg expression
s_mov_b32 s63, 131072 // (op 945) call()

s_mul_hi_i32 s47, s16, s10 // (op 1007) 
v_lshrrev_b32 v31, 7, v1 // (op 1007) 
s_add_i32 s0, s47, s16 // (op 1007) 

v_and_b32 v15, 15, v1 // (op 1007) 
s_ashr_i32 s1, s0, 31 // (op 1007) 
v_lshl_add_u32 v16, v31, 4, v15 // (op 1007) 


s_cmp_eq_i32 s10, 0 // (op 1007) 
s_mov_b32 s47, scc // (op 1007) 
v_and_b32 v15, 63, v1 // (op 1007) 
s_cmp_eq_i32 s47, 0 // (op 1007) 
s_cselect_b32 s18, 0, -1 // (op 1007) 

v_lshrrev_b32 v31, 4, v15 // (op 1007) 

s_and_b32 s19, s13, 31 // (op 1007) 
s_lshl_b32 s47, 1, s19 // (op 1007) 
s_add_i32 s22, s47, s18 // (op 1007) 


s_and_b32 s18, s1, s22 // (op 1007) 


s_add_i32 s1, s0, s18 // (op 1007) 


s_ashr_i32 s0, s1, s19 // (op 1007) 


s_lshl_b32 s1, s13, 24 // (op 1007) 
s_ashr_i32 s47, s1, 31 // (op 1007) 

s_xor_b32 s1, s0, s47 // (op 1007) 

s_sub_i32 s0, s1, s47 // (op 1007) 


s_mul_hi_i32 s1, s0, s14 // (op 1007) 
s_add_i32 s47, s1, s0 // (op 1007) 

s_ashr_i32 s1, s47, 31 // (op 1007) 
s_cmp_eq_i32 s14, 0 // (op 1007) 
s_mov_b32 s18, scc // (op 1007) 
s_cmp_eq_i32 s18, 0 // (op 1007) 
s_cselect_b32 s19, 0, -1 // (op 1007) 

s_and_b32 s18, s15, 31 // (op 1007) 
s_lshl_b32 s22, 1, s18 // (op 1007) 
s_add_i32 s23, s22, s19 // (op 1007) 


s_and_b32 s19, s1, s23 // (op 1007) 


s_add_i32 s1, s47, s19 // (op 1007) 


s_ashr_i32 s19, s1, s18 // (op 1007) 


s_lshl_b32 s1, s15, 24 // (op 1007) 
s_ashr_i32 s18, s1, 31 // (op 1007) 

s_xor_b32 s1, s19, s18 // (op 1007) 

s_sub_i32 s19, s1, s18 // (op 1007) 


s_mul_i32 s1, s0, s11 // (op 1007) 

s_sub_u32 s0, s16, s1 // (op 1007) 

s_mul_i32 s1, s19, s11 // (op 1007) 
s_add_u32 s1, s1, s0 // (op 1007) 


s_mul_hi_i32 s0, s16, s9 // (op 1007) 
s_add_i32 s47, s0, s16 // (op 1007) 

s_ashr_i32 s0, s47, 31 // (op 1007) 
s_cmp_eq_i32 s9, 0 // (op 1007) 
s_mov_b32 s18, scc // (op 1007) 
s_cmp_eq_i32 s18, 0 // (op 1007) 
s_cselect_b32 s19, 0, -1 // (op 1007) 

s_and_b32 s18, s45, 31 // (op 1007) 
s_lshl_b32 s22, 1, s18 // (op 1007) 
s_add_i32 s23, s22, s19 // (op 1007) 


s_and_b32 s19, s0, s23 // (op 1007) 


s_add_i32 s0, s47, s19 // (op 1007) 


s_ashr_i32 s19, s0, s18 // (op 1007) 


s_lshl_b32 s0, s45, 24 // (op 1007) 
s_ashr_i32 s18, s0, 31 // (op 1007) 

s_xor_b32 s0, s19, s18 // (op 1007) 

s_sub_i32 s19, s0, s18 // (op 1007) 


s_add_u32 s22, s6, 64 // (op 1007) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1007) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1007) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1007) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 1007) 

s_mul_i32 s0, s11, s22 // (op 1007) 

s_mul_i32 s18, s8, s0 // (op 1007) 

s_mul_i32 s0, s19, s18 // (op 1007) 

s_sub_u32 s18, s16, s0 // (op 1007) 

s_mov_b32 s0, s18 // (op 1007) 
s_mul_hi_i32 s47, s0, s12 // (op 1007) 
s_add_i32 s22, s47, s0 // (op 1007) 


s_ashr_i32 s0, s22, 31 // (op 1007) 
s_cmp_eq_i32 s12, 0 // (op 1007) 
s_mov_b32 s47, scc // (op 1007) 
s_cmp_eq_i32 s47, 0 // (op 1007) 
s_cselect_b32 s23, 0, -1 // (op 1007) 

s_and_b32 s47, s44, 31 // (op 1007) 
s_lshl_b32 s24, 1, s47 // (op 1007) 
s_add_i32 s25, s24, s23 // (op 1007) 


s_and_b32 s23, s0, s25 // (op 1007) 


s_add_i32 s0, s22, s23 // (op 1007) 


s_ashr_i32 s22, s0, s47 // (op 1007) 


s_lshl_b32 s0, s44, 24 // (op 1007) 
s_ashr_i32 s47, s0, 31 // (op 1007) 

s_xor_b32 s0, s22, s47 // (op 1007) 

s_sub_i32 s22, s0, s47 // (op 1007) 


s_mul_i32 s0, s46, s11 // (op 1007) 
s_add_u32 s24, s20, 32 // (op 1007) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1007) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 1007) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 1007) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 5 // (op 1007) 

s_sub_i32 s23, s24, s0 // (op 1007) 

s_mul_i32 s47, s22, s23 // (op 1007) 


s_sub_u32 s22, s18, s47 // (op 1007) 


s_add_u32 s18, s22, s0 // (op 1007) 


s_cmp_eq_i32 s19, 0 // (op 1007) 
s_mov_b32 s0, scc // (op 1007) 

s_cmp_eq_i32 s0, 0 // (op 1007) 
s_cselect_b32 s19, s18, s1 // (op 1007) 



v_lshl_add_u32 v15, s19, 5, v16 // (op 1007) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 1007) 
s_lshr_b32 s47, s0, 1 // (op 1007) 

v_lshl_add_u32 v16, s47, 3, v31 // (op 1007) 


v_mul_lo_u32 v31, v15, s34 // (op 1007) 
v_add_u32 v31, v31, v16 // (op 1007) 


s_mul_i32 s1, s34, s21 // (op 1009) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 1009) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 1009) least significant
s_mul_hi_u32 s18, s34, s20 // (op 1009) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1009) most significant: sum
s_add_u32 s1, s1, s18 // (op 1009) most significant: sum


s_mov_b32 s64, s36 // (op 1009) reg expression
s_mov_b32 s65, s37 // (op 1009) reg expression
s_mov_b32 s66, s0 // (op 1009) reg expression
s_mov_b32 s67, 131072 // (op 1009) call()

s_mul_hi_i32 s47, s16, s10 // (op 1071) 
v_lshrrev_b32 v15, 2, v1 // (op 1071) 
s_add_i32 s0, s47, s16 // (op 1071) 

v_and_b32 v16, -4, v1 // (op 1071) 
s_ashr_i32 s1, s0, 31 // (op 1071) 
v_sub_u32 v17, v1, v16 // (op 1071) 

s_cmp_eq_i32 s10, 0 // (op 1071) 
s_mov_b32 s47, scc // (op 1071) 
s_cmp_eq_i32 s47, 0 // (op 1071) 
s_cselect_b32 s18, 0, -1 // (op 1071) 

s_and_b32 s19, s13, 31 // (op 1071) 
s_lshl_b32 s47, 1, s19 // (op 1071) 
s_add_i32 s22, s47, s18 // (op 1071) 


s_and_b32 s18, s1, s22 // (op 1071) 


s_add_i32 s1, s0, s18 // (op 1071) 


s_ashr_i32 s0, s1, s19 // (op 1071) 


s_lshl_b32 s1, s13, 24 // (op 1071) 
s_ashr_i32 s47, s1, 31 // (op 1071) 

s_xor_b32 s1, s0, s47 // (op 1071) 

s_sub_i32 s0, s1, s47 // (op 1071) 


s_mul_hi_i32 s1, s0, s14 // (op 1071) 
s_add_i32 s47, s1, s0 // (op 1071) 

s_ashr_i32 s1, s47, 31 // (op 1071) 
s_cmp_eq_i32 s14, 0 // (op 1071) 
s_mov_b32 s18, scc // (op 1071) 
s_cmp_eq_i32 s18, 0 // (op 1071) 
s_cselect_b32 s19, 0, -1 // (op 1071) 

s_and_b32 s18, s15, 31 // (op 1071) 
s_lshl_b32 s22, 1, s18 // (op 1071) 
s_add_i32 s23, s22, s19 // (op 1071) 


s_and_b32 s19, s1, s23 // (op 1071) 


s_add_i32 s1, s47, s19 // (op 1071) 


s_ashr_i32 s19, s1, s18 // (op 1071) 


s_lshl_b32 s1, s15, 24 // (op 1071) 
s_ashr_i32 s18, s1, 31 // (op 1071) 

s_xor_b32 s1, s19, s18 // (op 1071) 

s_sub_i32 s19, s1, s18 // (op 1071) 


s_add_u32 s22, s6, 64 // (op 1071) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1071) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1071) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1071) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 1071) 

s_mul_i32 s1, s19, s22 // (op 1071) 

s_sub_i32 s47, s0, s1 // (op 1071) 


s_mul_hi_i32 s0, s16, s9 // (op 1071) 
s_add_i32 s1, s0, s16 // (op 1071) 

s_ashr_i32 s0, s1, 31 // (op 1071) 
s_cmp_eq_i32 s9, 0 // (op 1071) 
s_mov_b32 s18, scc // (op 1071) 
s_cmp_eq_i32 s18, 0 // (op 1071) 
s_cselect_b32 s19, 0, -1 // (op 1071) 

s_and_b32 s18, s45, 31 // (op 1071) 
s_lshl_b32 s24, 1, s18 // (op 1071) 
s_add_i32 s25, s24, s19 // (op 1071) 


s_and_b32 s19, s0, s25 // (op 1071) 


s_add_i32 s0, s1, s19 // (op 1071) 


s_ashr_i32 s1, s0, s18 // (op 1071) 


s_lshl_b32 s0, s45, 24 // (op 1071) 
s_ashr_i32 s18, s0, 31 // (op 1071) 

s_xor_b32 s0, s1, s18 // (op 1071) 

s_sub_i32 s1, s0, s18 // (op 1071) 


s_mul_i32 s0, s11, s22 // (op 1071) 

s_mul_i32 s18, s8, s0 // (op 1071) 

s_mul_i32 s0, s1, s18 // (op 1071) 

s_sub_u32 s18, s16, s0 // (op 1071) 

s_mul_hi_i32 s0, s18, s12 // (op 1071) 
s_add_i32 s19, s0, s18 // (op 1071) 


s_ashr_i32 s0, s19, 31 // (op 1071) 
s_cmp_eq_i32 s12, 0 // (op 1071) 
s_mov_b32 s18, scc // (op 1071) 
s_cmp_eq_i32 s18, 0 // (op 1071) 
s_cselect_b32 s22, 0, -1 // (op 1071) 

s_and_b32 s18, s44, 31 // (op 1071) 
s_lshl_b32 s23, 1, s18 // (op 1071) 
s_add_i32 s24, s23, s22 // (op 1071) 


s_and_b32 s25, s0, s24 // (op 1071) 


s_add_i32 s0, s19, s25 // (op 1071) 


s_ashr_i32 s19, s0, s18 // (op 1071) 


s_lshl_b32 s0, s44, 24 // (op 1071) 
s_ashr_i32 s18, s0, 31 // (op 1071) 

s_xor_b32 s0, s19, s18 // (op 1071) 

s_sub_i32 s19, s0, s18 // (op 1071) 


s_cmp_eq_i32 s1, 0 // (op 1071) 
s_mov_b32 s0, scc // (op 1071) 

s_cmp_eq_i32 s0, 0 // (op 1071) 
s_cselect_b32 s1, s19, s47 // (op 1071) 



v_lshl_add_u32 v16, s1, 6, v15 // (op 1071) 


v_mul_lo_u32 v15, v16, s38 // (op 1071) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 1071) 
s_lshr_b32 s47, s0, 1 // (op 1071) 

v_lshl_add_u32 v16, s47, 3, v17 // (op 1071) 


v_add_u32 v17, v16, v15 // (op 1071) 


s_mul_i32 s1, s38, s7 // (op 1073) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 1073) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 1073) least significant
s_mul_hi_u32 s18, s38, s6 // (op 1073) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1073) most significant: sum
s_add_u32 s1, s1, s18 // (op 1073) most significant: sum


s_mov_b32 s68, s40 // (op 1073) reg expression
s_mov_b32 s69, s41 // (op 1073) reg expression
s_mov_b32 s70, s0 // (op 1073) reg expression
s_mov_b32 s71, 131072 // (op 1073) call()

v_and_b32 v15, -8, v1 // (op 1164) 
v_sub_u32 v16, v1, v15 // (op 1164) 

v_lshlrev_b32 v15, 4, v16 // (op 1164) 

v_lshrrev_b32 v16, 3, v1 // (op 1164) 
v_lshl_add_u32 v18, v16, 7, v15 // (op 1164) 


v_and_b32 v15, -4, v1 // (op 1217) 
v_sub_u32 v16, v1, v15 // (op 1217) 

v_lshlrev_b32 v15, 5, v16 // (op 1217) 

v_lshrrev_b32 v16, 2, v1 // (op 1217) 
v_lshl_add_u32 v19, v16, 7, v15 // (op 1217) 


v_lshlrev_b32 v15, 2, v19 // (op 1217) 

v_lshrrev_b32 v16, 3, v15 // (op 1217) 

v_and_b32 v15, -4, v1 // (op 1275) 
v_sub_u32 v19, v1, v15 // (op 1275) 

v_and_b32 v15, 4294967292, v1 // (op 1275) 
v_add_u32 v20, v15, v19 // (op 1275) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 317) 
s_and_b32 s47, s0, 4294967294 // (op 317) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 316) 
s_mov_b32 s18, s0 // (op 316) convert

s_cmp_lt_u32 s47, s18 // (op 316) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_316 // (op 316) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_316 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopTop_KLoopTail_316:


s_add_u32 s47, s47, 1 // (op 318) 
v_mov_b32 v19, v12 // (op 170) 

buffer_load_dwordx4 v[22:25], v19, s[48:51], 0 offen // (op 170) Load value

v_add_u32 v12, 128, v12 // (op 865) 


s_barrier  // (op 1455) 
v_mov_b32 v19, v13 // (op 183) 




buffer_load_dwordx4 v[26:29], v19, s[60:63], 0 offen // (op 183) Load value

v_add_u32 v13, 64, v13 // (op 935) 
s_waitcnt vmcnt(1)
ds_write_b128 v18, v[22:25] offset:4352 // (op 171) Store local data 



v_mov_b32 v19, v17 // (op 184) 

s_barrier  // (op 1468) 
buffer_load_ubyte v30, v19, s[68:71], 0 offen offset: 0 // (op 184) Load value




v_add_u32 v17, 4, v17 // (op 1063) 
s_waitcnt vmcnt(1)
ds_write_b128 v16, v[26:29] // (op 181) Store local data 

s_waitcnt lgkmcnt(1)
s_barrier  // (op 1458) 



s_barrier  // (op 1482) 
ds_read_b128 v[22:25], v3 offset:4352 // (op 173) Load local data 



ds_read_b128 v[26:29], v3 offset:4416 // (op 173) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v20, v30 offset:4096 // (op 182) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 1472) 



ds_read_b128 v[32:35], v7 // (op 175) Load local data 


v_mov_b32 v19, v31 // (op 177) 

buffer_load_ubyte v21, v19, s[64:67], 0 offen offset: 0 // (op 177) Load value

v_add_u32 v31, 4, v31 // (op 999) 
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1486) 



ds_read_u8 v19, v0 offset:4096 // (op 179) Load local data 



ds_read_b128 v[36:39], v7 offset:2048 // (op 324) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:29], v[32:35], a[0:3], v21, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 180) 





ds_read_u8 v19, v0 offset:4224 // (op 327) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[22:29], v[36:39], a[4:7], v21, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 328) 




s_ashr_i64 s[0:1], s[42:43], 7 // (op 316) 
s_mov_b32 s18, s0 // (op 316) convert

s_cmp_lt_u32 s47, s18 // (op 316) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopTop_KLoopTail_316 // (op 316) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopTop_KLoopTail_316 if true)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_316:





















s_barrier  // (op 1568) 



















v_cmp_eq_f32 vcc, 0.00000, v5 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ConditionalFalse_0_beta_125
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

buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 54) Store value



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


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 511) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ConditionalBottom_0_beta_125
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ConditionalFalse_0_beta_125:


v_mov_b32 v0, v6 // (op 138) reg expression

buffer_load_dwordx2 v[10:11], v0, s[52:55], 0 offen // (op 138) Load value

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

buffer_store_dwordx2 v[10:11], v0, s[56:59], 0 offen // (op 137) Store value





s_lshl_b32 s0, s2, 6 // (op 535) 
v_add_u32 v0, v6, s0 // (op 535) 


buffer_load_dwordx2 v[10:11], v0, s[52:55], 0 offen // (op 535) Load value

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


buffer_store_dwordx2 v[10:11], v0, s[56:59], 0 offen // (op 534) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ConditionalBottom_0_beta_125:










s_endpgm  // End of RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_
.LRR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__end:
.size RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_, .LRR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__end-RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 72
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 16896 // lds bytes
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
  ProblemType:
    OperationType: GEMM
    DataType: F8
    DataTypeA: F8
    DataTypeB: F4
    DestDataType: h
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: True
    TransposeB: False
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
            { type: int32, semantic: RR_QuotientTilesByBlock },
            { type: int32, semantic: RR_MagicMultipleWgmMainBlock },
            { type: int32, semantic: RR_MagicMultipleWgm },
            { type: int32, semantic: RR_WorkgroupMapping },
            { type: int32, semantic: RR_MagicMultipleWgmTail },
            { type: uint32, semantic: RR_MagicShiftAndSignWgm },
            { type: int32, semantic: RR_MagicMultipleNumTilesN },
            { type: uint32, semantic: RR_MagicShiftAndSignNumTilesN },
            { type: int64, semantic: SizeFree0 },
            { type: address, semantic: AddressC },
            { type: address, semantic: AddressD },
            { type: int64, semantic: StrideA0 },
            { type: address, semantic: AddressA },
            { type: int64, semantic: StrideB0 },
            { type: address, semantic: AddressB },
            { type: int64, semantic: StrideScaleA0 },
            { type: address, semantic: AddressMXScaleA },
            { type: int64, semantic: StrideScaleB0 },
            { type: address, semantic: AddressMXScaleB },
            { type: int64, semantic: SizeSum },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmTail },
            { type: uint32, semantic: RR_MagicShiftAndSignWgmMainBlock },
            { type: int32, semantic: RR_QuotientTilesMByWgm },
            { type: float32, semantic: Beta },
            { type: float32, semantic: Alpha } ]
    macrotile: [32, 64, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 2]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 16896
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 74
    .vgpr_count: 64
    .agpr_count: 8
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {is-null: true}, {is-null: true}]
    .dynamic_sharedmemory_bytes: {type: LiteralValue.UInt32, dataType: UInt32, value: 0}
    .args:
      - .name: Tensor_10_stride_1_10
        .size: 8
        .offset: 0
        .expression: {type: CommandArgument, size: 8, offset: 192, name: Tensor_10_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 8
        .expression: {type: CommandArgument, size: 8, offset: 240, name: Tensor_21_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_size_1_4
        .size: 8
        .offset: 16
        .expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Subtract_21
        .size: 4
        .offset: 24
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 28
        .expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_22
        .size: 4
        .offset: 32
        .expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: WGM_18
        .size: 4
        .offset: 36
        .expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_30
        .size: 4
        .offset: 40
        .expression: {type: MagicMultiple, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_24, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 160, size: 4}, rhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_23
        .size: 4
        .offset: 44
        .expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_28
        .size: 4
        .offset: 48
        .expression: {type: MagicMultiple, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 52
        .expression: {type: MagicShiftAndSign, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_0_size_0_1
        .size: 8
        .offset: 56
        .expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 64
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: Half, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Half, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 72
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: Half, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Half, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_stride_0_0
        .size: 8
        .offset: 80
        .expression: {type: CommandArgument, size: 8, offset: 24, name: Tensor_0_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 88
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: FP8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP8, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_stride_1_3
        .size: 8
        .offset: 96
        .expression: {type: CommandArgument, size: 8, offset: 72, name: Tensor_2_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 104
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP4, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP4, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_stride_0_6
        .size: 8
        .offset: 112
        .expression: {type: CommandArgument, size: 8, offset: 104, name: Tensor_4_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 120
        .expression: {type: CommandArgument, size: 8, offset: 80, name: Tensor_4_pointer, variableType: {dataType: E8M0, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: E8M0, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_7_stride_1_8
        .size: 8
        .offset: 128
        .expression: {type: CommandArgument, size: 8, offset: 152, name: Tensor_7_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 136
        .expression: {type: CommandArgument, size: 8, offset: 120, name: Tensor_7_pointer, variableType: {dataType: E8M0, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: E8M0, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_14
        .size: 8
        .offset: 144
        .expression: {type: CommandArgument, size: 8, offset: 16, name: Tensor_0_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_31
        .size: 4
        .offset: 152
        .expression: {type: MagicShiftAndSign, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_24, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 160, size: 4}, rhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_27
        .size: 4
        .offset: 156
        .expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: Subtract_24
        .size: 4
        .offset: 160
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 5}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: user_Float_Value_14
        .size: 4
        .offset: 164
        .expression: {type: CommandArgument, size: 4, offset: 204, name: user_Float_Value_14, variableType: {dataType: Float, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Float, pointerType: Value}
        .value_kind: by_value
      - .name: user_Float_Value_12
        .size: 4
        .offset: 168
        .expression: {type: CommandArgument, size: 4, offset: 200, name: user_Float_Value_12, variableType: {dataType: Float, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Float, pointerType: Value}
        .value_kind: by_value
...

.end_amdgpu_metadata

