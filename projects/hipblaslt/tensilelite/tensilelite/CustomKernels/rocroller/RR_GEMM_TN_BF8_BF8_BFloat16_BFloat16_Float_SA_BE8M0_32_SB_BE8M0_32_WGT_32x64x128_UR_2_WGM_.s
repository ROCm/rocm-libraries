// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_,@function
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_:
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
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 1196) 
s_mul_hi_i32 s1, s16, s10 // (op 1383) 
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
s_cmp_eq_i32 s10, 0 // (op 1196) 
s_mov_b32 s19, scc // (op 1196) 
v_lshlrev_b32 v7, 2, v6 // (op 1196) 

s_cmp_eq_i32 s19, 0 // (op 1196) 
s_cselect_b32 s49, 0, -1 // (op 1196) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 1196) 


s_and_b32 s19, s13, 31 // (op 1196) 
v_lshrrev_b32 v0, 2, v5 // (op 1196) 

v_and_b32 v5, 3, v0 // (op 1196) 

v_lshrrev_b32 v0, 6, v1 // (op 1309) 
v_and_b32 v7, 1, v0 // (op 1309) 

v_and_b32 v0, 15, v1 // (op 1309) 
v_lshl_add_u32 v8, v7, 4, v0 // (op 1309) 


v_and_b32 v0, 48, v1 // (op 1309) 
v_lshl_add_u32 v7, v8, 7, v0 // (op 1309) 


v_lshrrev_b32 v0, 6, v1 // (op 1367) 
v_and_b32 v8, 1, v0 // (op 1367) 

v_and_b32 v0, 15, v1 // (op 1367) 
v_lshl_add_u32 v9, v8, 4, v0 // (op 1367) 


v_and_b32 v0, 63, v1 // (op 1367) 
v_lshrrev_b32 v8, 4, v0 // (op 1367) 

v_lshl_add_u32 v0, v9, 2, v8 // (op 1367) 


v_lshrrev_b32 v8, 7, v1 // (op 1383) 
v_and_b32 v9, 63, v1 // (op 1383) 
v_lshrrev_b32 v10, 6, v1 // (op 1468) 
v_and_b32 v11, 1, v10 // (op 1468) 

v_and_b32 v10, 15, v1 // (op 1468) 
v_lshl_add_u32 v12, v11, 4, v10 // (op 1468) 


v_and_b32 v10, 48, v1 // (op 1468) 
v_lshl_add_u32 v11, v12, 7, v10 // (op 1468) 


v_lshrrev_b32 v10, 6, v1 // (op 1497) 
v_and_b32 v12, 1, v10 // (op 1497) 

v_and_b32 v10, 15, v1 // (op 1497) 
v_lshl_add_u32 v13, v12, 4, v10 // (op 1497) 


v_and_b32 v10, 63, v1 // (op 1497) 
v_lshrrev_b32 v12, 4, v10 // (op 1497) 

v_lshl_add_u32 v10, v13, 2, v12 // (op 1497) 


s_lshl_b32 s50, 1, s19 // (op 1196) 
v_and_b32 v12, 3, v1 // (op 1196) 
s_add_i32 s51, s50, s49 // (op 1196) 


s_and_b32 s49, s0, s51 // (op 1196) 


s_add_i32 s0, s1, s16 // (op 1383) 

s_ashr_i32 s1, s0, 31 // (op 1383) 
v_lshl_add_u32 v13, v5, 2, v12 // (op 1196) 


s_add_i32 s50, s18, s49 // (op 1196) 


v_lshrrev_b32 v5, 6, v1 // (op 1196) 
s_ashr_i32 s18, s50, s19 // (op 1196) 


v_and_b32 v12, 1, v5 // (op 1196) 

s_lshl_b32 s19, s13, 24 // (op 1196) 
v_lshl_add_u32 v5, v12, 4, v13 // (op 1196) 


s_ashr_i32 s49, s19, 31 // (op 1196) 

s_xor_b32 s19, s18, s49 // (op 1196) 

s_sub_i32 s18, s19, s49 // (op 1196) 


s_mul_hi_i32 s19, s18, s14 // (op 1196) 
s_add_i32 s49, s19, s18 // (op 1196) 

s_ashr_i32 s19, s49, 31 // (op 1196) 
s_cmp_eq_i32 s14, 0 // (op 1196) 
s_mov_b32 s50, scc // (op 1196) 
s_cmp_eq_i32 s50, 0 // (op 1196) 
s_cselect_b32 s51, 0, -1 // (op 1196) 

s_and_b32 s50, s15, 31 // (op 1196) 
v_lshrrev_b32 v12, 4, v9 // (op 1383) 
s_cmp_eq_i32 s10, 0 // (op 1383) 
s_mov_b32 s52, scc // (op 1383) 
v_lshlrev_b32 v13, 2, v12 // (op 1383) 

s_cmp_eq_i32 s52, 0 // (op 1383) 
s_cselect_b32 s53, 0, -1 // (op 1383) 

s_lshl_b32 s52, 1, s50 // (op 1196) 
v_lshl_add_u32 v12, v8, 4, v13 // (op 1383) 


v_lshrrev_b32 v8, 2, v9 // (op 1383) 

s_add_i32 s54, s52, s51 // (op 1196) 


s_and_b32 s51, s19, s54 // (op 1196) 


s_add_i32 s19, s49, s51 // (op 1196) 


s_ashr_i32 s49, s19, s50 // (op 1196) 


s_lshl_b32 s19, s15, 24 // (op 1196) 
s_ashr_i32 s50, s19, 31 // (op 1196) 

s_xor_b32 s19, s49, s50 // (op 1196) 

s_sub_i32 s49, s19, s50 // (op 1196) 


s_mul_i32 s19, s18, s11 // (op 1196) 
s_sub_u32 s50, s16, s19 // (op 1196) 

s_mul_i32 s19, s49, s11 // (op 1196) 
s_add_u32 s19, s19, s50 // (op 1196) 

s_mul_hi_i32 s50, s16, s9 // (op 1196) 
s_add_i32 s51, s50, s16 // (op 1196) 

s_ashr_i32 s50, s51, 31 // (op 1196) 
s_cmp_eq_i32 s9, 0 // (op 1196) 
s_mov_b32 s52, scc // (op 1196) 
s_cmp_eq_i32 s52, 0 // (op 1196) 
s_cselect_b32 s54, 0, -1 // (op 1196) 

s_and_b32 s52, s13, 31 // (op 1383) 
v_and_b32 v9, 3, v8 // (op 1383) 

s_lshl_b32 s55, 1, s52 // (op 1383) 
v_and_b32 v8, 3, v1 // (op 1383) 
s_add_i32 s56, s55, s53 // (op 1383) 


s_and_b32 s53, s1, s56 // (op 1383) 


v_lshl_add_u32 v13, v9, 2, v8 // (op 1383) 


s_add_i32 s1, s0, s53 // (op 1383) 


v_lshrrev_b32 v8, 6, v1 // (op 1383) 
s_ashr_i32 s0, s1, s52 // (op 1383) 


v_and_b32 v9, 1, v8 // (op 1383) 

s_lshl_b32 s1, s13, 24 // (op 1383) 
v_lshl_add_u32 v8, v9, 4, v13 // (op 1383) 


s_ashr_i32 s52, s1, 31 // (op 1383) 

s_xor_b32 s1, s0, s52 // (op 1383) 

s_sub_i32 s0, s1, s52 // (op 1383) 


s_mul_hi_i32 s1, s0, s14 // (op 1383) 
s_add_i32 s52, s1, s0 // (op 1383) 

s_ashr_i32 s1, s52, 31 // (op 1383) 
s_cmp_eq_i32 s14, 0 // (op 1383) 
s_mov_b32 s53, scc // (op 1383) 
s_cmp_eq_i32 s53, 0 // (op 1383) 
s_cselect_b32 s55, 0, -1 // (op 1383) 

s_and_b32 s53, s15, 31 // (op 1383) 
s_lshl_b32 s56, 1, s53 // (op 1383) 
s_add_i32 s57, s56, s55 // (op 1383) 


s_and_b32 s55, s1, s57 // (op 1383) 


s_add_i32 s1, s52, s55 // (op 1383) 


s_ashr_i32 s52, s1, s53 // (op 1383) 


s_lshl_b32 s1, s15, 24 // (op 1383) 
s_ashr_i32 s53, s1, 31 // (op 1383) 

s_xor_b32 s1, s52, s53 // (op 1383) 

s_sub_i32 s52, s1, s53 // (op 1383) 


s_mul_i32 s1, s0, s11 // (op 1383) 
s_sub_u32 s53, s16, s1 // (op 1383) 

s_mul_i32 s1, s52, s11 // (op 1383) 
s_add_u32 s1, s1, s53 // (op 1383) 

s_mul_hi_i32 s53, s16, s9 // (op 1383) 
s_add_i32 s55, s53, s16 // (op 1383) 

s_ashr_i32 s53, s55, 31 // (op 1383) 
s_cmp_eq_i32 s9, 0 // (op 1383) 
s_mov_b32 s56, scc // (op 1383) 
s_cmp_eq_i32 s56, 0 // (op 1383) 
s_cselect_b32 s57, 0, -1 // (op 1383) 


s_waitcnt lgkmcnt(0)
s_and_b32 s56, s45, 31 // (op 1196) 
s_lshl_b32 s58, 1, s56 // (op 1196) 
s_add_i32 s59, s58, s54 // (op 1196) 


s_and_b32 s54, s50, s59 // (op 1196) 


s_add_i32 s50, s51, s54 // (op 1196) 


s_ashr_i32 s51, s50, s56 // (op 1196) 


s_lshl_b32 s50, s45, 24 // (op 1196) 
s_ashr_i32 s54, s50, 31 // (op 1196) 

s_xor_b32 s50, s51, s54 // (op 1196) 

s_sub_i32 s51, s50, s54 // (op 1196) 


s_and_b32 s50, s45, 31 // (op 1383) 
s_lshl_b32 s54, 1, s50 // (op 1383) 
s_add_i32 s56, s54, s57 // (op 1383) 


s_and_b32 s54, s53, s56 // (op 1383) 


s_add_i32 s53, s55, s54 // (op 1383) 


s_ashr_i32 s54, s53, s50 // (op 1383) 


s_lshl_b32 s50, s45, 24 // (op 1383) 
s_ashr_i32 s53, s50, 31 // (op 1383) 

s_xor_b32 s50, s54, s53 // (op 1383) 

s_sub_i32 s54, s50, s53 // (op 1383) 


s_add_u32 s56, s6, 64 // (op 1196) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 1196) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 1196) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 1196) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 1196) 

s_mul_i32 s50, s11, s56 // (op 1196) 
s_mul_i32 s53, s8, s50 // (op 1196) 

s_mul_i32 s50, s51, s53 // (op 1196) 

s_sub_u32 s53, s16, s50 // (op 1196) 

s_mov_b32 s50, s53 // (op 1196) 
s_mul_hi_i32 s55, s50, s12 // (op 1196) 
s_add_i32 s58, s55, s50 // (op 1196) 


s_ashr_i32 s50, s58, 31 // (op 1196) 
s_cmp_eq_i32 s12, 0 // (op 1196) 
s_mov_b32 s55, scc // (op 1196) 
s_cmp_eq_i32 s55, 0 // (op 1196) 
s_cselect_b32 s59, 0, -1 // (op 1196) 

s_and_b32 s55, s44, 31 // (op 1196) 
s_lshl_b32 s60, 1, s55 // (op 1196) 
s_add_i32 s61, s60, s59 // (op 1196) 


s_and_b32 s59, s50, s61 // (op 1196) 


s_add_i32 s50, s58, s59 // (op 1196) 


s_ashr_i32 s58, s50, s55 // (op 1196) 


s_lshl_b32 s50, s44, 24 // (op 1196) 
s_ashr_i32 s55, s50, 31 // (op 1196) 

s_xor_b32 s50, s58, s55 // (op 1196) 

s_sub_i32 s58, s50, s55 // (op 1196) 


s_mul_i32 s50, s46, s11 // (op 1196) 
s_add_u32 s60, s20, 32 // (op 1196) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 1196) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 1196) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 1196) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 5 // (op 1196) 

s_sub_i32 s55, s60, s50 // (op 1196) 

s_mul_i32 s59, s58, s55 // (op 1196) 

s_sub_u32 s55, s53, s59 // (op 1196) 


s_add_u32 s53, s55, s50 // (op 1196) 


s_cmp_eq_i32 s51, 0 // (op 1196) 
s_mov_b32 s50, scc // (op 1196) 

s_cmp_eq_i32 s50, 0 // (op 1196) 
s_cselect_b32 s51, s53, s19 // (op 1196) 


s_mul_i32 s19, s49, s56 // (op 1196) 


v_lshl_add_u32 v9, s51, 5, v6 // (op 1196) 


s_sub_i32 s49, s18, s19 // (op 1196) 


s_cmp_eq_i32 s50, 0 // (op 1196) 
s_cselect_b32 s51, s58, s49 // (op 1196) 



v_lshl_add_u32 v6, s51, 6, v5 // (op 1196) 


v_mul_lo_u32 v5, v6, s2 // (op 1196) 

v_add_lshl_u32 v6, v9, v5, 1 // (op 1196) 


s_add_u32 s18, s6, 64 // (op 1383) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 1383) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 1383) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 1383) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 6 // (op 1383) 

s_mul_i32 s53, s11, s18 // (op 1383) 
s_mul_i32 s49, s8, s53 // (op 1383) 

s_mul_i32 s53, s54, s49 // (op 1383) 

s_sub_u32 s49, s16, s53 // (op 1383) 

s_mov_b32 s53, s49 // (op 1383) 
s_mul_hi_i32 s50, s53, s12 // (op 1383) 
s_add_i32 s51, s50, s53 // (op 1383) 


s_ashr_i32 s50, s51, 31 // (op 1383) 
s_cmp_eq_i32 s12, 0 // (op 1383) 
s_mov_b32 s53, scc // (op 1383) 
s_cmp_eq_i32 s53, 0 // (op 1383) 
s_cselect_b32 s55, 0, -1 // (op 1383) 

s_and_b32 s53, s44, 31 // (op 1383) 
s_lshl_b32 s56, 1, s53 // (op 1383) 
s_add_i32 s57, s56, s55 // (op 1383) 


s_and_b32 s55, s50, s57 // (op 1383) 


s_add_i32 s50, s51, s55 // (op 1383) 


s_ashr_i32 s51, s50, s53 // (op 1383) 


s_lshl_b32 s50, s44, 24 // (op 1383) 
s_ashr_i32 s53, s50, 31 // (op 1383) 

s_xor_b32 s50, s51, s53 // (op 1383) 

s_sub_i32 s51, s50, s53 // (op 1383) 


s_mul_i32 s50, s46, s11 // (op 1383) 
s_add_u32 s56, s20, 32 // (op 1383) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 1383) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 1383) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 1383) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 5 // (op 1383) 

s_sub_i32 s53, s56, s50 // (op 1383) 

s_mul_i32 s55, s51, s53 // (op 1383) 

s_sub_u32 s53, s49, s55 // (op 1383) 


s_add_u32 s49, s53, s50 // (op 1383) 


s_cmp_eq_i32 s54, 0 // (op 1383) 
s_mov_b32 s50, scc // (op 1383) 

s_cmp_eq_i32 s50, 0 // (op 1383) 
s_cselect_b32 s53, s49, s1 // (op 1383) 


s_mul_i32 s1, s52, s18 // (op 1383) 


v_lshl_add_u32 v5, s53, 5, v12 // (op 1383) 


s_sub_i32 s49, s0, s1 // (op 1383) 


s_cmp_eq_i32 s50, 0 // (op 1383) 
s_cselect_b32 s0, s51, s49 // (op 1383) 



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
s_mul_i32 s49, s5, s6 // (op 1385) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 1385) least significant
s_mul_hi_u32 s50, s4, s6 // (op 1385) most significant: high of low * low
s_add_u32 s1, s1, s49 // (op 1385) most significant: sum
s_add_u32 s1, s1, s50 // (op 1385) most significant: sum


s_lshl_b64 s[50:51], s[0:1], 1 // (op 1385) 

s_mov_b32 s52, s22 // (op 1198) reg expression
s_mov_b32 s53, s23 // (op 1198) reg expression
s_mov_b32 s54, s18 // (op 1198) reg expression
s_mov_b32 s55, 131072 // (op 1198) call()

s_mov_b32 s56, s24 // (op 1385) reg expression

s_mov_b32 s57, s25 // (op 1385) reg expression
s_mov_b32 s58, s50 // (op 1385) reg expression
s_mov_b32 s59, 131072 // (op 1385) call()


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
s_mul_hi_i32 s0, s16, s10 // (op 963) 
v_lshrrev_b32 v12, 3, v1 // (op 963) 
s_add_i32 s1, s0, s16 // (op 963) 

v_and_b32 v13, -8, v1 // (op 963) 
s_ashr_i32 s0, s1, 31 // (op 963) 
v_sub_u32 v14, v1, v13 // (op 963) 

s_cmp_eq_i32 s10, 0 // (op 963) 
s_mov_b32 s18, scc // (op 963) 
v_lshlrev_b32 v13, 4, v14 // (op 963) 

s_cmp_eq_i32 s18, 0 // (op 963) 
s_cselect_b32 s19, 0, -1 // (op 963) 

v_mov_b32 v15, 0 // (op 963) convert
v_mov_b32 v14, v13 // (op 963) convert

s_and_b32 s18, s13, 31 // (op 963) 
s_lshl_b32 s22, 1, s18 // (op 963) 
s_add_i32 s23, s22, s19 // (op 963) 


s_and_b32 s19, s0, s23 // (op 963) 


s_add_i32 s0, s1, s19 // (op 963) 


s_ashr_i32 s1, s0, s18 // (op 963) 


s_lshl_b32 s0, s13, 24 // (op 963) 
s_ashr_i32 s18, s0, 31 // (op 963) 

s_xor_b32 s0, s1, s18 // (op 963) 

s_sub_i32 s1, s0, s18 // (op 963) 


s_mul_hi_i32 s0, s1, s14 // (op 963) 
s_add_i32 s18, s0, s1 // (op 963) 

s_ashr_i32 s0, s18, 31 // (op 963) 
s_cmp_eq_i32 s14, 0 // (op 963) 
s_mov_b32 s19, scc // (op 963) 
s_cmp_eq_i32 s19, 0 // (op 963) 
s_cselect_b32 s22, 0, -1 // (op 963) 

s_and_b32 s19, s15, 31 // (op 963) 
s_lshl_b32 s23, 1, s19 // (op 963) 
s_add_i32 s24, s23, s22 // (op 963) 


s_and_b32 s25, s0, s24 // (op 963) 


s_add_i32 s0, s18, s25 // (op 963) 


s_ashr_i32 s18, s0, s19 // (op 963) 


s_lshl_b32 s0, s15, 24 // (op 963) 
s_ashr_i32 s19, s0, 31 // (op 963) 

s_xor_b32 s0, s18, s19 // (op 963) 

s_sub_i32 s18, s0, s19 // (op 963) 


s_mul_i32 s0, s1, s11 // (op 963) 

s_sub_u32 s1, s16, s0 // (op 963) 

s_mul_i32 s0, s18, s11 // (op 963) 
s_add_u32 s0, s0, s1 // (op 963) 


s_mul_hi_i32 s1, s16, s9 // (op 963) 
s_add_i32 s18, s1, s16 // (op 963) 

s_ashr_i32 s1, s18, 31 // (op 963) 
s_cmp_eq_i32 s9, 0 // (op 963) 
s_mov_b32 s19, scc // (op 963) 
s_cmp_eq_i32 s19, 0 // (op 963) 
s_cselect_b32 s22, 0, -1 // (op 963) 

s_and_b32 s19, s45, 31 // (op 963) 
s_lshl_b32 s23, 1, s19 // (op 963) 
s_add_i32 s24, s23, s22 // (op 963) 


s_and_b32 s25, s1, s24 // (op 963) 


s_add_i32 s1, s18, s25 // (op 963) 


s_ashr_i32 s18, s1, s19 // (op 963) 


s_lshl_b32 s1, s45, 24 // (op 963) 
s_ashr_i32 s19, s1, 31 // (op 963) 

s_xor_b32 s1, s18, s19 // (op 963) 

s_sub_i32 s18, s1, s19 // (op 963) 


s_add_u32 s22, s6, 64 // (op 963) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 963) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 963) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 963) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 963) 

s_mul_i32 s1, s11, s22 // (op 963) 

s_mul_i32 s19, s8, s1 // (op 963) 

s_mul_i32 s1, s18, s19 // (op 963) 

s_sub_u32 s19, s16, s1 // (op 963) 

s_mov_b32 s1, s19 // (op 963) 
s_mul_hi_i32 s22, s1, s12 // (op 963) 
s_add_i32 s23, s22, s1 // (op 963) 


s_ashr_i32 s1, s23, 31 // (op 963) 
s_cmp_eq_i32 s12, 0 // (op 963) 
s_mov_b32 s22, scc // (op 963) 
s_cmp_eq_i32 s22, 0 // (op 963) 
s_cselect_b32 s24, 0, -1 // (op 963) 

s_and_b32 s22, s44, 31 // (op 963) 
s_lshl_b32 s25, 1, s22 // (op 963) 
s_add_i32 s47, s25, s24 // (op 963) 


s_and_b32 s24, s1, s47 // (op 963) 


s_add_i32 s1, s23, s24 // (op 963) 


s_ashr_i32 s23, s1, s22 // (op 963) 


s_lshl_b32 s1, s44, 24 // (op 963) 
s_ashr_i32 s22, s1, 31 // (op 963) 

s_xor_b32 s1, s23, s22 // (op 963) 

s_sub_i32 s23, s1, s22 // (op 963) 


s_mul_i32 s1, s46, s11 // (op 963) 
s_add_u32 s24, s20, 32 // (op 963) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 963) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 963) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 963) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 5 // (op 963) 

s_sub_i32 s22, s24, s1 // (op 963) 

s_mul_i32 s24, s23, s22 // (op 963) 


s_sub_u32 s25, s19, s24 // (op 963) 


s_add_u32 s19, s25, s1 // (op 963) 


s_cmp_eq_i32 s18, 0 // (op 963) 
s_mov_b32 s1, scc // (op 963) 

s_cmp_eq_i32 s1, 0 // (op 963) 
s_cselect_b32 s18, s19, s0 // (op 963) 



v_lshl_add_u32 v13, s18, 5, v12 // (op 963) 


v_mul_lo_u32 v17, v13, s27 // (op 963) most significant: low of low * high
v_mul_hi_u32 v12, v13, s26 // (op 963) most significant: high of low * low
v_mul_lo_u32 v16, v13, s26 // (op 963) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 963) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 963) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 963) most significant half



s_mul_i32 s1, s26, s21 // (op 965) most significant: low of low * high
s_mul_i32 s18, s27, s20 // (op 965) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 965) least significant
s_mul_hi_u32 s19, s26, s20 // (op 965) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 965) most significant: sum
s_add_u32 s1, s1, s19 // (op 965) most significant: sum


s_mov_b32 s48, s28 // (op 965) reg expression
s_mov_b32 s49, s29 // (op 965) reg expression
s_mov_b32 s50, s0 // (op 965) reg expression
s_mov_b32 s51, 131072 // (op 965) call()

s_mul_hi_i32 s47, s16, s10 // (op 982) 
v_lshrrev_b32 v12, 3, v1 // (op 982) 
s_add_i32 s0, s47, s16 // (op 982) 

v_and_b32 v13, -8, v1 // (op 982) 
s_ashr_i32 s1, s0, 31 // (op 982) 
v_sub_u32 v14, v1, v13 // (op 982) 

s_cmp_eq_i32 s10, 0 // (op 982) 
s_mov_b32 s47, scc // (op 982) 
v_lshlrev_b32 v13, 4, v14 // (op 982) 

s_cmp_eq_i32 s47, 0 // (op 982) 
s_cselect_b32 s18, 0, -1 // (op 982) 

v_mov_b32 v15, 0 // (op 982) convert
v_mov_b32 v14, v13 // (op 982) convert

s_and_b32 s19, s13, 31 // (op 982) 
s_lshl_b32 s47, 1, s19 // (op 982) 
s_add_i32 s22, s47, s18 // (op 982) 


s_and_b32 s18, s1, s22 // (op 982) 


s_add_i32 s1, s0, s18 // (op 982) 


s_ashr_i32 s0, s1, s19 // (op 982) 


s_lshl_b32 s1, s13, 24 // (op 982) 
s_ashr_i32 s47, s1, 31 // (op 982) 

s_xor_b32 s1, s0, s47 // (op 982) 

s_sub_i32 s0, s1, s47 // (op 982) 


s_mul_hi_i32 s1, s0, s14 // (op 982) 
s_add_i32 s47, s1, s0 // (op 982) 

s_ashr_i32 s1, s47, 31 // (op 982) 
s_cmp_eq_i32 s14, 0 // (op 982) 
s_mov_b32 s18, scc // (op 982) 
s_cmp_eq_i32 s18, 0 // (op 982) 
s_cselect_b32 s19, 0, -1 // (op 982) 

s_and_b32 s18, s15, 31 // (op 982) 
s_lshl_b32 s22, 1, s18 // (op 982) 
s_add_i32 s23, s22, s19 // (op 982) 


s_and_b32 s19, s1, s23 // (op 982) 


s_add_i32 s1, s47, s19 // (op 982) 


s_ashr_i32 s19, s1, s18 // (op 982) 


s_lshl_b32 s1, s15, 24 // (op 982) 
s_ashr_i32 s18, s1, 31 // (op 982) 

s_xor_b32 s1, s19, s18 // (op 982) 

s_sub_i32 s19, s1, s18 // (op 982) 


s_mul_i32 s1, s0, s11 // (op 982) 

s_sub_u32 s0, s16, s1 // (op 982) 

s_mul_i32 s1, s19, s11 // (op 982) 
s_add_u32 s1, s1, s0 // (op 982) 


s_mul_hi_i32 s0, s16, s9 // (op 982) 
s_add_i32 s47, s0, s16 // (op 982) 

s_ashr_i32 s0, s47, 31 // (op 982) 
s_cmp_eq_i32 s9, 0 // (op 982) 
s_mov_b32 s18, scc // (op 982) 
s_cmp_eq_i32 s18, 0 // (op 982) 
s_cselect_b32 s19, 0, -1 // (op 982) 

s_and_b32 s18, s45, 31 // (op 982) 
s_lshl_b32 s22, 1, s18 // (op 982) 
s_add_i32 s23, s22, s19 // (op 982) 


s_and_b32 s19, s0, s23 // (op 982) 


s_add_i32 s0, s47, s19 // (op 982) 


s_ashr_i32 s19, s0, s18 // (op 982) 


s_lshl_b32 s0, s45, 24 // (op 982) 
s_ashr_i32 s18, s0, 31 // (op 982) 

s_xor_b32 s0, s19, s18 // (op 982) 

s_sub_i32 s19, s0, s18 // (op 982) 


s_add_u32 s22, s6, 64 // (op 982) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 982) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 982) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 982) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 982) 

s_mul_i32 s0, s11, s22 // (op 982) 

s_mul_i32 s18, s8, s0 // (op 982) 

s_mul_i32 s0, s19, s18 // (op 982) 

s_sub_u32 s18, s16, s0 // (op 982) 

s_mov_b32 s0, s18 // (op 982) 
s_mul_hi_i32 s47, s0, s12 // (op 982) 
s_add_i32 s22, s47, s0 // (op 982) 


s_ashr_i32 s0, s22, 31 // (op 982) 
s_cmp_eq_i32 s12, 0 // (op 982) 
s_mov_b32 s47, scc // (op 982) 
s_cmp_eq_i32 s47, 0 // (op 982) 
s_cselect_b32 s23, 0, -1 // (op 982) 

s_and_b32 s47, s44, 31 // (op 982) 
s_lshl_b32 s24, 1, s47 // (op 982) 
s_add_i32 s25, s24, s23 // (op 982) 


s_and_b32 s23, s0, s25 // (op 982) 


s_add_i32 s0, s22, s23 // (op 982) 


s_ashr_i32 s22, s0, s47 // (op 982) 


s_lshl_b32 s0, s44, 24 // (op 982) 
s_ashr_i32 s47, s0, 31 // (op 982) 

s_xor_b32 s0, s22, s47 // (op 982) 

s_sub_i32 s22, s0, s47 // (op 982) 


s_mul_i32 s0, s46, s11 // (op 982) 
s_add_u32 s24, s20, 32 // (op 982) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 982) End of Int64 add, unlocking SCC
s_sub_u32 s60, s24, 1 // (op 982) least significant half; sets scc
s_subb_u32 s61, s25, 0 // (op 982) most significant half; uses scc

s_ashr_i64 s[24:25], s[60:61], 5 // (op 982) 

s_sub_i32 s23, s24, s0 // (op 982) 

s_mul_i32 s47, s22, s23 // (op 982) 


s_sub_u32 s22, s18, s47 // (op 982) 


s_add_u32 s18, s22, s0 // (op 982) 


s_cmp_eq_i32 s19, 0 // (op 982) 
s_mov_b32 s0, scc // (op 982) 

s_cmp_eq_i32 s0, 0 // (op 982) 
s_cselect_b32 s19, s18, s1 // (op 982) 



v_lshl_add_u32 v13, s19, 5, v12 // (op 982) 


v_mul_lo_u32 v19, v13, s27 // (op 982) most significant: low of low * high
v_mul_hi_u32 v12, v13, s26 // (op 982) most significant: high of low * low
v_mul_lo_u32 v18, v13, s26 // (op 982) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 982) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v14 // (op 982) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 982) most significant half



s_mul_i32 s1, s26, s21 // (op 984) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 984) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 984) least significant
s_mul_hi_u32 s18, s26, s20 // (op 984) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 984) most significant: sum
s_add_u32 s1, s1, s18 // (op 984) most significant: sum


s_mov_b32 s48, s28 // (op 984) reg expression
s_mov_b32 s49, s29 // (op 984) reg expression
s_mov_b32 s50, s0 // (op 984) reg expression
s_mov_b32 s51, 131072 // (op 984) call()

s_mul_hi_i32 s47, s16, s10 // (op 1027) 
v_lshrrev_b32 v12, 3, v1 // (op 1027) 
s_add_i32 s0, s47, s16 // (op 1027) 

v_and_b32 v13, -8, v1 // (op 1027) 
s_ashr_i32 s1, s0, 31 // (op 1027) 
v_sub_u32 v14, v1, v13 // (op 1027) 

s_cmp_eq_i32 s10, 0 // (op 1027) 
s_mov_b32 s47, scc // (op 1027) 
v_lshlrev_b32 v13, 4, v14 // (op 1027) 

s_cmp_eq_i32 s47, 0 // (op 1027) 
s_cselect_b32 s18, 0, -1 // (op 1027) 

v_mov_b32 v15, 0 // (op 1027) convert
v_mov_b32 v14, v13 // (op 1027) convert

s_and_b32 s19, s13, 31 // (op 1027) 
s_lshl_b32 s47, 1, s19 // (op 1027) 
s_add_i32 s22, s47, s18 // (op 1027) 


s_and_b32 s18, s1, s22 // (op 1027) 


s_add_i32 s1, s0, s18 // (op 1027) 


s_ashr_i32 s0, s1, s19 // (op 1027) 


s_lshl_b32 s1, s13, 24 // (op 1027) 
s_ashr_i32 s47, s1, 31 // (op 1027) 

s_xor_b32 s1, s0, s47 // (op 1027) 

s_sub_i32 s0, s1, s47 // (op 1027) 


s_mul_hi_i32 s1, s0, s14 // (op 1027) 
s_add_i32 s47, s1, s0 // (op 1027) 

s_ashr_i32 s1, s47, 31 // (op 1027) 
s_cmp_eq_i32 s14, 0 // (op 1027) 
s_mov_b32 s18, scc // (op 1027) 
s_cmp_eq_i32 s18, 0 // (op 1027) 
s_cselect_b32 s19, 0, -1 // (op 1027) 

s_and_b32 s18, s15, 31 // (op 1027) 
s_lshl_b32 s22, 1, s18 // (op 1027) 
s_add_i32 s23, s22, s19 // (op 1027) 


s_and_b32 s19, s1, s23 // (op 1027) 


s_add_i32 s1, s47, s19 // (op 1027) 


s_ashr_i32 s19, s1, s18 // (op 1027) 


s_lshl_b32 s1, s15, 24 // (op 1027) 
s_ashr_i32 s18, s1, 31 // (op 1027) 

s_xor_b32 s1, s19, s18 // (op 1027) 

s_sub_i32 s19, s1, s18 // (op 1027) 


s_add_u32 s22, s6, 64 // (op 1027) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1027) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1027) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1027) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 1027) 

s_mul_i32 s1, s19, s22 // (op 1027) 

s_sub_i32 s47, s0, s1 // (op 1027) 


s_mul_hi_i32 s0, s16, s9 // (op 1027) 
s_add_i32 s1, s0, s16 // (op 1027) 

s_ashr_i32 s0, s1, 31 // (op 1027) 
s_cmp_eq_i32 s9, 0 // (op 1027) 
s_mov_b32 s18, scc // (op 1027) 
s_cmp_eq_i32 s18, 0 // (op 1027) 
s_cselect_b32 s19, 0, -1 // (op 1027) 

s_and_b32 s18, s45, 31 // (op 1027) 
s_lshl_b32 s24, 1, s18 // (op 1027) 
s_add_i32 s25, s24, s19 // (op 1027) 


s_and_b32 s19, s0, s25 // (op 1027) 


s_add_i32 s0, s1, s19 // (op 1027) 


s_ashr_i32 s1, s0, s18 // (op 1027) 


s_lshl_b32 s0, s45, 24 // (op 1027) 
s_ashr_i32 s18, s0, 31 // (op 1027) 

s_xor_b32 s0, s1, s18 // (op 1027) 

s_sub_i32 s1, s0, s18 // (op 1027) 


s_mul_i32 s0, s11, s22 // (op 1027) 

s_mul_i32 s18, s8, s0 // (op 1027) 

s_mul_i32 s0, s1, s18 // (op 1027) 

s_sub_u32 s18, s16, s0 // (op 1027) 

s_mul_hi_i32 s0, s18, s12 // (op 1027) 
s_add_i32 s19, s0, s18 // (op 1027) 


s_ashr_i32 s0, s19, 31 // (op 1027) 
s_cmp_eq_i32 s12, 0 // (op 1027) 
s_mov_b32 s18, scc // (op 1027) 
s_cmp_eq_i32 s18, 0 // (op 1027) 
s_cselect_b32 s22, 0, -1 // (op 1027) 

s_and_b32 s18, s44, 31 // (op 1027) 
s_lshl_b32 s23, 1, s18 // (op 1027) 
s_add_i32 s24, s23, s22 // (op 1027) 


s_and_b32 s25, s0, s24 // (op 1027) 


s_add_i32 s0, s19, s25 // (op 1027) 


s_ashr_i32 s19, s0, s18 // (op 1027) 


s_lshl_b32 s0, s44, 24 // (op 1027) 
s_ashr_i32 s18, s0, 31 // (op 1027) 

s_xor_b32 s0, s19, s18 // (op 1027) 

s_sub_i32 s19, s0, s18 // (op 1027) 


s_cmp_eq_i32 s1, 0 // (op 1027) 
s_mov_b32 s0, scc // (op 1027) 

s_cmp_eq_i32 s0, 0 // (op 1027) 
s_cselect_b32 s1, s19, s47 // (op 1027) 



v_lshl_add_u32 v13, s1, 6, v12 // (op 1027) 


v_mul_lo_u32 v21, v13, s31 // (op 1027) most significant: low of low * high
v_mul_hi_u32 v12, v13, s30 // (op 1027) most significant: high of low * low
v_mul_lo_u32 v20, v13, s30 // (op 1027) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 1027) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v20 // (op 1027) least significant half
v_addc_co_u32 v13, s[0:1], v15, v21, s[0:1] // (op 1027) most significant half



s_mul_i32 s1, s30, s7 // (op 1029) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 1029) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 1029) least significant
s_mul_hi_u32 s18, s30, s6 // (op 1029) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1029) most significant: sum
s_add_u32 s1, s1, s18 // (op 1029) most significant: sum


s_mov_b32 s60, s32 // (op 1029) reg expression
s_mov_b32 s61, s33 // (op 1029) reg expression
s_mov_b32 s62, s0 // (op 1029) reg expression
s_mov_b32 s63, 131072 // (op 1029) call()

s_mul_hi_i32 s47, s16, s10 // (op 1046) 
v_lshrrev_b32 v14, 3, v1 // (op 1046) 
s_add_i32 s0, s47, s16 // (op 1046) 

v_and_b32 v15, -8, v1 // (op 1046) 
s_ashr_i32 s1, s0, 31 // (op 1046) 
v_sub_u32 v20, v1, v15 // (op 1046) 

s_cmp_eq_i32 s10, 0 // (op 1046) 
s_mov_b32 s47, scc // (op 1046) 
v_lshlrev_b32 v15, 4, v20 // (op 1046) 

s_cmp_eq_i32 s47, 0 // (op 1046) 
s_cselect_b32 s18, 0, -1 // (op 1046) 

v_mov_b32 v21, 0 // (op 1046) convert
v_mov_b32 v20, v15 // (op 1046) convert

s_and_b32 s19, s13, 31 // (op 1046) 
s_lshl_b32 s47, 1, s19 // (op 1046) 
s_add_i32 s22, s47, s18 // (op 1046) 


s_and_b32 s18, s1, s22 // (op 1046) 


s_add_i32 s1, s0, s18 // (op 1046) 


s_ashr_i32 s0, s1, s19 // (op 1046) 


s_lshl_b32 s1, s13, 24 // (op 1046) 
s_ashr_i32 s47, s1, 31 // (op 1046) 

s_xor_b32 s1, s0, s47 // (op 1046) 

s_sub_i32 s0, s1, s47 // (op 1046) 


s_mul_hi_i32 s1, s0, s14 // (op 1046) 
s_add_i32 s47, s1, s0 // (op 1046) 

s_ashr_i32 s1, s47, 31 // (op 1046) 
s_cmp_eq_i32 s14, 0 // (op 1046) 
s_mov_b32 s18, scc // (op 1046) 
s_cmp_eq_i32 s18, 0 // (op 1046) 
s_cselect_b32 s19, 0, -1 // (op 1046) 

s_and_b32 s18, s15, 31 // (op 1046) 
s_lshl_b32 s22, 1, s18 // (op 1046) 
s_add_i32 s23, s22, s19 // (op 1046) 


s_and_b32 s19, s1, s23 // (op 1046) 


s_add_i32 s1, s47, s19 // (op 1046) 


s_ashr_i32 s19, s1, s18 // (op 1046) 


s_lshl_b32 s1, s15, 24 // (op 1046) 
s_ashr_i32 s18, s1, 31 // (op 1046) 

s_xor_b32 s1, s19, s18 // (op 1046) 

s_sub_i32 s19, s1, s18 // (op 1046) 


s_add_u32 s22, s6, 64 // (op 1046) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1046) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1046) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1046) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 1046) 

s_mul_i32 s1, s19, s22 // (op 1046) 

s_sub_i32 s47, s0, s1 // (op 1046) 


s_mul_hi_i32 s0, s16, s9 // (op 1046) 
s_add_i32 s1, s0, s16 // (op 1046) 

s_ashr_i32 s0, s1, 31 // (op 1046) 
s_cmp_eq_i32 s9, 0 // (op 1046) 
s_mov_b32 s18, scc // (op 1046) 
s_cmp_eq_i32 s18, 0 // (op 1046) 
s_cselect_b32 s19, 0, -1 // (op 1046) 

s_and_b32 s18, s45, 31 // (op 1046) 
s_lshl_b32 s24, 1, s18 // (op 1046) 
s_add_i32 s25, s24, s19 // (op 1046) 


s_and_b32 s19, s0, s25 // (op 1046) 


s_add_i32 s0, s1, s19 // (op 1046) 


s_ashr_i32 s1, s0, s18 // (op 1046) 


s_lshl_b32 s0, s45, 24 // (op 1046) 
s_ashr_i32 s18, s0, 31 // (op 1046) 

s_xor_b32 s0, s1, s18 // (op 1046) 

s_sub_i32 s1, s0, s18 // (op 1046) 


s_mul_i32 s0, s11, s22 // (op 1046) 

s_mul_i32 s18, s8, s0 // (op 1046) 

s_mul_i32 s0, s1, s18 // (op 1046) 

s_sub_u32 s18, s16, s0 // (op 1046) 

s_mul_hi_i32 s0, s18, s12 // (op 1046) 
s_add_i32 s19, s0, s18 // (op 1046) 


s_ashr_i32 s0, s19, 31 // (op 1046) 
s_cmp_eq_i32 s12, 0 // (op 1046) 
s_mov_b32 s18, scc // (op 1046) 
s_cmp_eq_i32 s18, 0 // (op 1046) 
s_cselect_b32 s22, 0, -1 // (op 1046) 

s_and_b32 s18, s44, 31 // (op 1046) 
s_lshl_b32 s23, 1, s18 // (op 1046) 
s_add_i32 s24, s23, s22 // (op 1046) 


s_and_b32 s25, s0, s24 // (op 1046) 


s_add_i32 s0, s19, s25 // (op 1046) 


s_ashr_i32 s19, s0, s18 // (op 1046) 


s_lshl_b32 s0, s44, 24 // (op 1046) 
s_ashr_i32 s18, s0, 31 // (op 1046) 

s_xor_b32 s0, s19, s18 // (op 1046) 

s_sub_i32 s19, s0, s18 // (op 1046) 


s_cmp_eq_i32 s1, 0 // (op 1046) 
s_mov_b32 s0, scc // (op 1046) 

s_cmp_eq_i32 s0, 0 // (op 1046) 
s_cselect_b32 s1, s19, s47 // (op 1046) 



v_lshl_add_u32 v15, s1, 6, v14 // (op 1046) 


v_mul_lo_u32 v23, v15, s31 // (op 1046) most significant: low of low * high
v_mul_hi_u32 v14, v15, s30 // (op 1046) most significant: high of low * low
v_mul_lo_u32 v22, v15, s30 // (op 1046) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 1046) most significant: sum


v_add_co_u32 v14, s[0:1], v20, v22 // (op 1046) least significant half
v_addc_co_u32 v15, s[0:1], v21, v23, s[0:1] // (op 1046) most significant half



s_mul_i32 s1, s30, s7 // (op 1048) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 1048) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 1048) least significant
s_mul_hi_u32 s18, s30, s6 // (op 1048) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1048) most significant: sum
s_add_u32 s1, s1, s18 // (op 1048) most significant: sum


s_mov_b32 s60, s32 // (op 1048) reg expression
s_mov_b32 s61, s33 // (op 1048) reg expression
s_mov_b32 s62, s0 // (op 1048) reg expression
s_mov_b32 s63, 131072 // (op 1048) call()

s_mul_hi_i32 s47, s16, s10 // (op 1091) 
v_lshrrev_b32 v20, 7, v1 // (op 1091) 
s_add_i32 s0, s47, s16 // (op 1091) 

v_and_b32 v21, 15, v1 // (op 1091) 
s_ashr_i32 s1, s0, 31 // (op 1091) 
v_lshl_add_u32 v22, v20, 4, v21 // (op 1091) 


s_cmp_eq_i32 s10, 0 // (op 1091) 
s_mov_b32 s47, scc // (op 1091) 
v_and_b32 v20, 63, v1 // (op 1091) 
s_cmp_eq_i32 s47, 0 // (op 1091) 
s_cselect_b32 s18, 0, -1 // (op 1091) 

v_lshrrev_b32 v21, 4, v20 // (op 1091) 

s_and_b32 s19, s13, 31 // (op 1091) 
s_lshl_b32 s47, 1, s19 // (op 1091) 
s_add_i32 s22, s47, s18 // (op 1091) 


s_and_b32 s18, s1, s22 // (op 1091) 


s_add_i32 s1, s0, s18 // (op 1091) 


s_ashr_i32 s0, s1, s19 // (op 1091) 


s_lshl_b32 s1, s13, 24 // (op 1091) 
s_ashr_i32 s47, s1, 31 // (op 1091) 

s_xor_b32 s1, s0, s47 // (op 1091) 

s_sub_i32 s0, s1, s47 // (op 1091) 


s_mul_hi_i32 s1, s0, s14 // (op 1091) 
s_add_i32 s47, s1, s0 // (op 1091) 

s_ashr_i32 s1, s47, 31 // (op 1091) 
s_cmp_eq_i32 s14, 0 // (op 1091) 
s_mov_b32 s18, scc // (op 1091) 
s_cmp_eq_i32 s18, 0 // (op 1091) 
s_cselect_b32 s19, 0, -1 // (op 1091) 

s_and_b32 s18, s15, 31 // (op 1091) 
s_lshl_b32 s22, 1, s18 // (op 1091) 
s_add_i32 s23, s22, s19 // (op 1091) 


s_and_b32 s19, s1, s23 // (op 1091) 


s_add_i32 s1, s47, s19 // (op 1091) 


s_ashr_i32 s19, s1, s18 // (op 1091) 


s_lshl_b32 s1, s15, 24 // (op 1091) 
s_ashr_i32 s18, s1, 31 // (op 1091) 

s_xor_b32 s1, s19, s18 // (op 1091) 

s_sub_i32 s19, s1, s18 // (op 1091) 


s_mul_i32 s1, s0, s11 // (op 1091) 

s_sub_u32 s0, s16, s1 // (op 1091) 

s_mul_i32 s1, s19, s11 // (op 1091) 
s_add_u32 s1, s1, s0 // (op 1091) 


s_mul_hi_i32 s0, s16, s9 // (op 1091) 
s_add_i32 s47, s0, s16 // (op 1091) 

s_ashr_i32 s0, s47, 31 // (op 1091) 
s_cmp_eq_i32 s9, 0 // (op 1091) 
s_mov_b32 s18, scc // (op 1091) 
s_cmp_eq_i32 s18, 0 // (op 1091) 
s_cselect_b32 s19, 0, -1 // (op 1091) 

s_and_b32 s18, s45, 31 // (op 1091) 
s_lshl_b32 s22, 1, s18 // (op 1091) 
s_add_i32 s23, s22, s19 // (op 1091) 


s_and_b32 s19, s0, s23 // (op 1091) 


s_add_i32 s0, s47, s19 // (op 1091) 


s_ashr_i32 s19, s0, s18 // (op 1091) 


s_lshl_b32 s0, s45, 24 // (op 1091) 
s_ashr_i32 s18, s0, 31 // (op 1091) 

s_xor_b32 s0, s19, s18 // (op 1091) 

s_sub_i32 s19, s0, s18 // (op 1091) 


s_add_u32 s22, s6, 64 // (op 1091) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1091) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1091) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1091) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 1091) 

s_mul_i32 s0, s11, s22 // (op 1091) 

s_mul_i32 s18, s8, s0 // (op 1091) 

s_mul_i32 s0, s19, s18 // (op 1091) 

s_sub_u32 s18, s16, s0 // (op 1091) 

s_mov_b32 s0, s18 // (op 1091) 
s_mul_hi_i32 s47, s0, s12 // (op 1091) 
s_add_i32 s22, s47, s0 // (op 1091) 


s_ashr_i32 s0, s22, 31 // (op 1091) 
s_cmp_eq_i32 s12, 0 // (op 1091) 
s_mov_b32 s47, scc // (op 1091) 
s_cmp_eq_i32 s47, 0 // (op 1091) 
s_cselect_b32 s23, 0, -1 // (op 1091) 

s_and_b32 s47, s44, 31 // (op 1091) 
s_lshl_b32 s24, 1, s47 // (op 1091) 
s_add_i32 s25, s24, s23 // (op 1091) 


s_and_b32 s23, s0, s25 // (op 1091) 


s_add_i32 s0, s22, s23 // (op 1091) 


s_ashr_i32 s22, s0, s47 // (op 1091) 


s_lshl_b32 s0, s44, 24 // (op 1091) 
s_ashr_i32 s47, s0, 31 // (op 1091) 

s_xor_b32 s0, s22, s47 // (op 1091) 

s_sub_i32 s22, s0, s47 // (op 1091) 


s_mul_i32 s0, s46, s11 // (op 1091) 
s_add_u32 s24, s20, 32 // (op 1091) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1091) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 1091) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 1091) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 5 // (op 1091) 

s_sub_i32 s23, s24, s0 // (op 1091) 

s_mul_i32 s47, s22, s23 // (op 1091) 


s_sub_u32 s22, s18, s47 // (op 1091) 


s_add_u32 s18, s22, s0 // (op 1091) 


s_cmp_eq_i32 s19, 0 // (op 1091) 
s_mov_b32 s0, scc // (op 1091) 

s_cmp_eq_i32 s0, 0 // (op 1091) 
s_cselect_b32 s19, s18, s1 // (op 1091) 



v_lshl_add_u32 v20, s19, 5, v22 // (op 1091) 


v_mul_lo_u32 v22, v20, s34 // (op 1091) 
v_add_u32 v22, v22, v21 // (op 1091) 


s_mul_i32 s1, s34, s21 // (op 1093) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 1093) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 1093) least significant
s_mul_hi_u32 s18, s34, s20 // (op 1093) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1093) most significant: sum
s_add_u32 s1, s1, s18 // (op 1093) most significant: sum


s_mov_b32 s64, s36 // (op 1093) reg expression
s_mov_b32 s65, s37 // (op 1093) reg expression
s_mov_b32 s66, s0 // (op 1093) reg expression
s_mov_b32 s67, 131072 // (op 1093) call()

s_mul_hi_i32 s47, s16, s10 // (op 1110) 
v_lshrrev_b32 v20, 7, v1 // (op 1110) 
s_add_i32 s0, s47, s16 // (op 1110) 

v_and_b32 v21, 15, v1 // (op 1110) 
s_ashr_i32 s1, s0, 31 // (op 1110) 
v_lshl_add_u32 v23, v20, 4, v21 // (op 1110) 


s_cmp_eq_i32 s10, 0 // (op 1110) 
s_mov_b32 s47, scc // (op 1110) 
v_and_b32 v20, 63, v1 // (op 1110) 
s_cmp_eq_i32 s47, 0 // (op 1110) 
s_cselect_b32 s18, 0, -1 // (op 1110) 

v_lshrrev_b32 v21, 4, v20 // (op 1110) 

s_and_b32 s19, s13, 31 // (op 1110) 
s_lshl_b32 s47, 1, s19 // (op 1110) 
s_add_i32 s22, s47, s18 // (op 1110) 


s_and_b32 s18, s1, s22 // (op 1110) 


s_add_i32 s1, s0, s18 // (op 1110) 


s_ashr_i32 s0, s1, s19 // (op 1110) 


s_lshl_b32 s1, s13, 24 // (op 1110) 
s_ashr_i32 s47, s1, 31 // (op 1110) 

s_xor_b32 s1, s0, s47 // (op 1110) 

s_sub_i32 s0, s1, s47 // (op 1110) 


s_mul_hi_i32 s1, s0, s14 // (op 1110) 
s_add_i32 s47, s1, s0 // (op 1110) 

s_ashr_i32 s1, s47, 31 // (op 1110) 
s_cmp_eq_i32 s14, 0 // (op 1110) 
s_mov_b32 s18, scc // (op 1110) 
s_cmp_eq_i32 s18, 0 // (op 1110) 
s_cselect_b32 s19, 0, -1 // (op 1110) 

s_and_b32 s18, s15, 31 // (op 1110) 
s_lshl_b32 s22, 1, s18 // (op 1110) 
s_add_i32 s23, s22, s19 // (op 1110) 


s_and_b32 s19, s1, s23 // (op 1110) 


s_add_i32 s1, s47, s19 // (op 1110) 


s_ashr_i32 s19, s1, s18 // (op 1110) 


s_lshl_b32 s1, s15, 24 // (op 1110) 
s_ashr_i32 s18, s1, 31 // (op 1110) 

s_xor_b32 s1, s19, s18 // (op 1110) 

s_sub_i32 s19, s1, s18 // (op 1110) 


s_mul_i32 s1, s0, s11 // (op 1110) 

s_sub_u32 s0, s16, s1 // (op 1110) 

s_mul_i32 s1, s19, s11 // (op 1110) 
s_add_u32 s1, s1, s0 // (op 1110) 


s_mul_hi_i32 s0, s16, s9 // (op 1110) 
s_add_i32 s47, s0, s16 // (op 1110) 

s_ashr_i32 s0, s47, 31 // (op 1110) 
s_cmp_eq_i32 s9, 0 // (op 1110) 
s_mov_b32 s18, scc // (op 1110) 
s_cmp_eq_i32 s18, 0 // (op 1110) 
s_cselect_b32 s19, 0, -1 // (op 1110) 

s_and_b32 s18, s45, 31 // (op 1110) 
s_lshl_b32 s22, 1, s18 // (op 1110) 
s_add_i32 s23, s22, s19 // (op 1110) 


s_and_b32 s19, s0, s23 // (op 1110) 


s_add_i32 s0, s47, s19 // (op 1110) 


s_ashr_i32 s19, s0, s18 // (op 1110) 


s_lshl_b32 s0, s45, 24 // (op 1110) 
s_ashr_i32 s18, s0, 31 // (op 1110) 

s_xor_b32 s0, s19, s18 // (op 1110) 

s_sub_i32 s19, s0, s18 // (op 1110) 


s_add_u32 s22, s6, 64 // (op 1110) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1110) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1110) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1110) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 1110) 

s_mul_i32 s0, s11, s22 // (op 1110) 

s_mul_i32 s18, s8, s0 // (op 1110) 

s_mul_i32 s0, s19, s18 // (op 1110) 

s_sub_u32 s18, s16, s0 // (op 1110) 

s_mov_b32 s0, s18 // (op 1110) 
s_mul_hi_i32 s47, s0, s12 // (op 1110) 
s_add_i32 s22, s47, s0 // (op 1110) 


s_ashr_i32 s0, s22, 31 // (op 1110) 
s_cmp_eq_i32 s12, 0 // (op 1110) 
s_mov_b32 s47, scc // (op 1110) 
s_cmp_eq_i32 s47, 0 // (op 1110) 
s_cselect_b32 s23, 0, -1 // (op 1110) 

s_and_b32 s47, s44, 31 // (op 1110) 
s_lshl_b32 s24, 1, s47 // (op 1110) 
s_add_i32 s25, s24, s23 // (op 1110) 


s_and_b32 s23, s0, s25 // (op 1110) 


s_add_i32 s0, s22, s23 // (op 1110) 


s_ashr_i32 s22, s0, s47 // (op 1110) 


s_lshl_b32 s0, s44, 24 // (op 1110) 
s_ashr_i32 s47, s0, 31 // (op 1110) 

s_xor_b32 s0, s22, s47 // (op 1110) 

s_sub_i32 s22, s0, s47 // (op 1110) 


s_mul_i32 s0, s46, s11 // (op 1110) 
s_add_u32 s24, s20, 32 // (op 1110) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1110) End of Int64 add, unlocking SCC
s_sub_u32 s68, s24, 1 // (op 1110) least significant half; sets scc
s_subb_u32 s69, s25, 0 // (op 1110) most significant half; uses scc

s_ashr_i64 s[24:25], s[68:69], 5 // (op 1110) 

s_sub_i32 s23, s24, s0 // (op 1110) 

s_mul_i32 s47, s22, s23 // (op 1110) 


s_sub_u32 s22, s18, s47 // (op 1110) 


s_add_u32 s18, s22, s0 // (op 1110) 


s_cmp_eq_i32 s19, 0 // (op 1110) 
s_mov_b32 s0, scc // (op 1110) 

s_cmp_eq_i32 s0, 0 // (op 1110) 
s_cselect_b32 s19, s18, s1 // (op 1110) 



v_lshl_add_u32 v20, s19, 5, v23 // (op 1110) 


v_mul_lo_u32 v23, v20, s34 // (op 1110) 
v_add_u32 v23, v23, v21 // (op 1110) 


s_mul_i32 s1, s34, s21 // (op 1112) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 1112) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 1112) least significant
s_mul_hi_u32 s18, s34, s20 // (op 1112) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1112) most significant: sum
s_add_u32 s1, s1, s18 // (op 1112) most significant: sum


s_mov_b32 s64, s36 // (op 1112) reg expression
s_mov_b32 s65, s37 // (op 1112) reg expression
s_mov_b32 s66, s0 // (op 1112) reg expression
s_mov_b32 s67, 131072 // (op 1112) call()

s_mul_hi_i32 s47, s16, s10 // (op 1155) 
v_lshrrev_b32 v20, 2, v1 // (op 1155) 
s_add_i32 s0, s47, s16 // (op 1155) 

v_and_b32 v21, -4, v1 // (op 1155) 
s_ashr_i32 s1, s0, 31 // (op 1155) 
v_sub_u32 v24, v1, v21 // (op 1155) 

s_cmp_eq_i32 s10, 0 // (op 1155) 
s_mov_b32 s47, scc // (op 1155) 
s_cmp_eq_i32 s47, 0 // (op 1155) 
s_cselect_b32 s18, 0, -1 // (op 1155) 

s_and_b32 s19, s13, 31 // (op 1155) 
s_lshl_b32 s47, 1, s19 // (op 1155) 
s_add_i32 s22, s47, s18 // (op 1155) 


s_and_b32 s18, s1, s22 // (op 1155) 


s_add_i32 s1, s0, s18 // (op 1155) 


s_ashr_i32 s0, s1, s19 // (op 1155) 


s_lshl_b32 s1, s13, 24 // (op 1155) 
s_ashr_i32 s47, s1, 31 // (op 1155) 

s_xor_b32 s1, s0, s47 // (op 1155) 

s_sub_i32 s0, s1, s47 // (op 1155) 


s_mul_hi_i32 s1, s0, s14 // (op 1155) 
s_add_i32 s47, s1, s0 // (op 1155) 

s_ashr_i32 s1, s47, 31 // (op 1155) 
s_cmp_eq_i32 s14, 0 // (op 1155) 
s_mov_b32 s18, scc // (op 1155) 
s_cmp_eq_i32 s18, 0 // (op 1155) 
s_cselect_b32 s19, 0, -1 // (op 1155) 

s_and_b32 s18, s15, 31 // (op 1155) 
s_lshl_b32 s22, 1, s18 // (op 1155) 
s_add_i32 s23, s22, s19 // (op 1155) 


s_and_b32 s19, s1, s23 // (op 1155) 


s_add_i32 s1, s47, s19 // (op 1155) 


s_ashr_i32 s19, s1, s18 // (op 1155) 


s_lshl_b32 s1, s15, 24 // (op 1155) 
s_ashr_i32 s18, s1, 31 // (op 1155) 

s_xor_b32 s1, s19, s18 // (op 1155) 

s_sub_i32 s19, s1, s18 // (op 1155) 


s_add_u32 s22, s6, 64 // (op 1155) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1155) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1155) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1155) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 1155) 

s_mul_i32 s1, s19, s22 // (op 1155) 

s_sub_i32 s47, s0, s1 // (op 1155) 


s_mul_hi_i32 s0, s16, s9 // (op 1155) 
s_add_i32 s1, s0, s16 // (op 1155) 

s_ashr_i32 s0, s1, 31 // (op 1155) 
s_cmp_eq_i32 s9, 0 // (op 1155) 
s_mov_b32 s18, scc // (op 1155) 
s_cmp_eq_i32 s18, 0 // (op 1155) 
s_cselect_b32 s19, 0, -1 // (op 1155) 

s_and_b32 s18, s45, 31 // (op 1155) 
s_lshl_b32 s24, 1, s18 // (op 1155) 
s_add_i32 s25, s24, s19 // (op 1155) 


s_and_b32 s19, s0, s25 // (op 1155) 


s_add_i32 s0, s1, s19 // (op 1155) 


s_ashr_i32 s1, s0, s18 // (op 1155) 


s_lshl_b32 s0, s45, 24 // (op 1155) 
s_ashr_i32 s18, s0, 31 // (op 1155) 

s_xor_b32 s0, s1, s18 // (op 1155) 

s_sub_i32 s1, s0, s18 // (op 1155) 


s_mul_i32 s0, s11, s22 // (op 1155) 

s_mul_i32 s18, s8, s0 // (op 1155) 

s_mul_i32 s0, s1, s18 // (op 1155) 

s_sub_u32 s18, s16, s0 // (op 1155) 

s_mul_hi_i32 s0, s18, s12 // (op 1155) 
s_add_i32 s19, s0, s18 // (op 1155) 


s_ashr_i32 s0, s19, 31 // (op 1155) 
s_cmp_eq_i32 s12, 0 // (op 1155) 
s_mov_b32 s18, scc // (op 1155) 
s_cmp_eq_i32 s18, 0 // (op 1155) 
s_cselect_b32 s22, 0, -1 // (op 1155) 

s_and_b32 s18, s44, 31 // (op 1155) 
s_lshl_b32 s23, 1, s18 // (op 1155) 
s_add_i32 s24, s23, s22 // (op 1155) 


s_and_b32 s25, s0, s24 // (op 1155) 


s_add_i32 s0, s19, s25 // (op 1155) 


s_ashr_i32 s19, s0, s18 // (op 1155) 


s_lshl_b32 s0, s44, 24 // (op 1155) 
s_ashr_i32 s18, s0, 31 // (op 1155) 

s_xor_b32 s0, s19, s18 // (op 1155) 

s_sub_i32 s19, s0, s18 // (op 1155) 


s_cmp_eq_i32 s1, 0 // (op 1155) 
s_mov_b32 s0, scc // (op 1155) 

s_cmp_eq_i32 s0, 0 // (op 1155) 
s_cselect_b32 s1, s19, s47 // (op 1155) 



v_lshl_add_u32 v21, s1, 6, v20 // (op 1155) 


v_mul_lo_u32 v20, v21, s38 // (op 1155) 

v_add_u32 v21, v24, v20 // (op 1155) 


s_mul_i32 s1, s38, s7 // (op 1157) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 1157) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 1157) least significant
s_mul_hi_u32 s18, s38, s6 // (op 1157) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1157) most significant: sum
s_add_u32 s1, s1, s18 // (op 1157) most significant: sum


s_mov_b32 s68, s40 // (op 1157) reg expression
s_mov_b32 s69, s41 // (op 1157) reg expression
s_mov_b32 s70, s0 // (op 1157) reg expression
s_mov_b32 s71, 131072 // (op 1157) call()

s_mul_hi_i32 s47, s16, s10 // (op 1174) 
v_lshrrev_b32 v20, 2, v1 // (op 1174) 
s_add_i32 s0, s47, s16 // (op 1174) 

v_and_b32 v24, -4, v1 // (op 1174) 
s_ashr_i32 s1, s0, 31 // (op 1174) 
v_sub_u32 v25, v1, v24 // (op 1174) 

s_cmp_eq_i32 s10, 0 // (op 1174) 
s_mov_b32 s47, scc // (op 1174) 
s_cmp_eq_i32 s47, 0 // (op 1174) 
s_cselect_b32 s18, 0, -1 // (op 1174) 

s_and_b32 s19, s13, 31 // (op 1174) 
s_lshl_b32 s47, 1, s19 // (op 1174) 
s_add_i32 s22, s47, s18 // (op 1174) 


s_and_b32 s18, s1, s22 // (op 1174) 


s_add_i32 s1, s0, s18 // (op 1174) 


s_ashr_i32 s0, s1, s19 // (op 1174) 


s_lshl_b32 s1, s13, 24 // (op 1174) 
s_ashr_i32 s47, s1, 31 // (op 1174) 

s_xor_b32 s1, s0, s47 // (op 1174) 

s_sub_i32 s0, s1, s47 // (op 1174) 


s_mul_hi_i32 s1, s0, s14 // (op 1174) 
s_add_i32 s47, s1, s0 // (op 1174) 

s_ashr_i32 s1, s47, 31 // (op 1174) 
s_cmp_eq_i32 s14, 0 // (op 1174) 
s_mov_b32 s18, scc // (op 1174) 
s_cmp_eq_i32 s18, 0 // (op 1174) 
s_cselect_b32 s19, 0, -1 // (op 1174) 

s_and_b32 s18, s15, 31 // (op 1174) 
s_lshl_b32 s22, 1, s18 // (op 1174) 
s_add_i32 s23, s22, s19 // (op 1174) 


s_and_b32 s19, s1, s23 // (op 1174) 


s_add_i32 s1, s47, s19 // (op 1174) 


s_ashr_i32 s19, s1, s18 // (op 1174) 


s_lshl_b32 s1, s15, 24 // (op 1174) 
s_ashr_i32 s18, s1, 31 // (op 1174) 

s_xor_b32 s1, s19, s18 // (op 1174) 

s_sub_i32 s19, s1, s18 // (op 1174) 


s_add_u32 s22, s6, 64 // (op 1174) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1174) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1174) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1174) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 1174) 

s_mul_i32 s1, s19, s22 // (op 1174) 

s_sub_i32 s47, s0, s1 // (op 1174) 


s_mul_hi_i32 s0, s16, s9 // (op 1174) 
s_add_i32 s1, s0, s16 // (op 1174) 

s_ashr_i32 s0, s1, 31 // (op 1174) 
s_cmp_eq_i32 s9, 0 // (op 1174) 
s_mov_b32 s18, scc // (op 1174) 
s_cmp_eq_i32 s18, 0 // (op 1174) 
s_cselect_b32 s19, 0, -1 // (op 1174) 

s_and_b32 s18, s45, 31 // (op 1174) 
s_lshl_b32 s24, 1, s18 // (op 1174) 
s_add_i32 s25, s24, s19 // (op 1174) 


s_and_b32 s19, s0, s25 // (op 1174) 


s_add_i32 s0, s1, s19 // (op 1174) 


s_ashr_i32 s1, s0, s18 // (op 1174) 


s_lshl_b32 s0, s45, 24 // (op 1174) 
s_ashr_i32 s18, s0, 31 // (op 1174) 

s_xor_b32 s0, s1, s18 // (op 1174) 

s_sub_i32 s1, s0, s18 // (op 1174) 


s_mul_i32 s0, s11, s22 // (op 1174) 

s_mul_i32 s18, s8, s0 // (op 1174) 

s_mul_i32 s0, s1, s18 // (op 1174) 

s_sub_u32 s18, s16, s0 // (op 1174) 

s_mul_hi_i32 s0, s18, s12 // (op 1174) 
s_add_i32 s19, s0, s18 // (op 1174) 


s_ashr_i32 s0, s19, 31 // (op 1174) 
s_cmp_eq_i32 s12, 0 // (op 1174) 
s_mov_b32 s18, scc // (op 1174) 
s_cmp_eq_i32 s18, 0 // (op 1174) 
s_cselect_b32 s22, 0, -1 // (op 1174) 

s_and_b32 s18, s44, 31 // (op 1174) 
s_lshl_b32 s23, 1, s18 // (op 1174) 
s_add_i32 s24, s23, s22 // (op 1174) 


s_and_b32 s25, s0, s24 // (op 1174) 


s_add_i32 s0, s19, s25 // (op 1174) 


s_ashr_i32 s19, s0, s18 // (op 1174) 


s_lshl_b32 s0, s44, 24 // (op 1174) 
s_ashr_i32 s18, s0, 31 // (op 1174) 

s_xor_b32 s0, s19, s18 // (op 1174) 

s_sub_i32 s19, s0, s18 // (op 1174) 


s_cmp_eq_i32 s1, 0 // (op 1174) 
s_mov_b32 s0, scc // (op 1174) 

s_cmp_eq_i32 s0, 0 // (op 1174) 
s_cselect_b32 s1, s19, s47 // (op 1174) 



v_lshl_add_u32 v24, s1, 6, v20 // (op 1174) 


v_mul_lo_u32 v20, v24, s38 // (op 1174) 

v_add_u32 v24, v25, v20 // (op 1174) 


s_mul_i32 s1, s38, s7 // (op 1176) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 1176) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 1176) least significant
s_mul_hi_u32 s18, s38, s6 // (op 1176) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1176) most significant: sum
s_add_u32 s1, s1, s18 // (op 1176) most significant: sum


s_mov_b32 s68, s40 // (op 1176) reg expression
s_mov_b32 s69, s41 // (op 1176) reg expression
s_mov_b32 s70, s0 // (op 1176) reg expression
s_mov_b32 s71, 131072 // (op 1176) call()




buffer_load_ubyte v20, v22, s[64:67], 0 offen offset: 0 // (op 769) Load value
v_and_b32 v25, -8, v1 // (op 1292) 
v_sub_u32 v26, v1, v25 // (op 1292) 

v_lshlrev_b32 v25, 4, v26 // (op 1292) 

v_lshrrev_b32 v26, 3, v1 // (op 1292) 
v_lshl_add_u32 v27, v26, 7, v25 // (op 1292) 


s_nop 0
v_readfirstlane_b32 s47, v27 // (op 1292) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 5 // (op 911) 
v_mov_b64 v[26:27], s[0:1] // (op 911) convert


v_mov_b64 v[28:29], v[12:13] // (op 911) 
s_mov_b32 m0, s47 // (op 911) reg expression
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 911) Load value direct to lds
v_add_co_u32 v28, s[0:1], v28, v26 // (op 911) least significant half
v_addc_co_u32 v29, s[0:1], 0, v29, s[0:1] // (op 911) most significant half

s_add_u32 m0, m0, 4096 // (op 911) 
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 911) Load value direct to lds






buffer_load_ubyte v25, v21, s[68:71], 0 offen offset: 0 // (op 776) Load value
v_and_b32 v26, -8, v1 // (op 1240) 
v_sub_u32 v27, v1, v26 // (op 1240) 

v_lshlrev_b32 v26, 4, v27 // (op 1240) 

v_lshrrev_b32 v27, 3, v1 // (op 1240) 
v_lshl_add_u32 v28, v27, 7, v26 // (op 1240) 


s_nop 0
v_readfirstlane_b32 s47, v28 // (op 1240) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s0, 8192, s47 // (op 916) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s0 // (op 916) reg expression
buffer_load_dwordx4 v16, s[48:51], 0 offen offset: 0 lds // (op 916) Load value direct to lds





buffer_load_ubyte v26, v22, s[64:67], 0 offen offset:4 // (op 783) Load value

v_and_b32 v22, -8, v1 // (op 1451) 
v_sub_u32 v27, v1, v22 // (op 1451) 

v_lshlrev_b32 v22, 4, v27 // (op 1451) 

v_lshrrev_b32 v27, 3, v1 // (op 1451) 
v_lshl_add_u32 v28, v27, 7, v22 // (op 1451) 


s_nop 0
v_readfirstlane_b32 s47, v28 // (op 1451) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 5 // (op 921) 
v_mov_b64 v[28:29], s[0:1] // (op 921) convert


v_mov_b64 v[30:31], v[12:13] // (op 921) 
s_add_u32 s0, 12288, s47 // (op 921) 
s_mov_b32 m0, s0 // (op 921) reg expression
s_mov_b32 s1, 128 // (op 921) call()
buffer_load_dwordx4 v30, s[60:63], s1 offen offset: 0 lds // (op 921) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 921) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 921) most significant half

s_add_u32 m0, m0, 4096 // (op 921) 
buffer_load_dwordx4 v30, s[60:63], s1 offen offset: 0 lds // (op 921) Load value direct to lds









buffer_load_ubyte v22, v21, s[68:71], 0 offen offset:4 // (op 790) Load value

v_and_b32 v21, -8, v1 // (op 1412) 
v_sub_u32 v12, v1, v21 // (op 1412) 

v_lshlrev_b32 v13, 4, v12 // (op 1412) 

v_lshrrev_b32 v12, 3, v1 // (op 1412) 
v_lshl_add_u32 v21, v12, 7, v13 // (op 1412) 


s_nop 0
v_readfirstlane_b32 s47, v21 // (op 1412) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s0, 20480, s47 // (op 923) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s0 // (op 923) reg expression
s_mov_b32 s1, 128 // (op 923) call()
buffer_load_dwordx4 v16, s[48:51], s1 offen offset: 0 lds // (op 923) Load value direct to lds




v_and_b32 v21, -4, v1 // (op 1350) 
v_sub_u32 v12, v1, v21 // (op 1350) 

v_and_b32 v13, 4294967292, v1 // (op 1350) 
v_add_u32 v21, v13, v12 // (op 1350) 





s_waitcnt vmcnt(6)
ds_write_b8 v21, v25 offset:24576 // (op 796) Store local data 

s_barrier  // (op 1554) 



ds_read_b128 v[28:31], v7 // (op 814) Load local data 
ds_read_b128 v[32:35], v7 offset:64 // (op 814) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1570) 



ds_read_u8 v21, v0 offset:24576 // (op 824) Load local data 



ds_read_b128 v[36:39], v7 offset:4096 // (op 819) Load local data 
ds_read_b128 v[40:43], v7 offset:4160 // (op 819) Load local data 



ds_read_u8 v27, v0 offset:24704 // (op 829) Load local data 
s_waitcnt vmcnt(5)
s_barrier  // (op 1538) 



ds_read_b128 v[44:47], v3 offset:8192 // (op 810) Load local data 
ds_read_b128 v[48:51], v3 offset:8256 // (op 810) Load local data 
v_and_b32 v12, -8, v1 // (op 1212) 
v_sub_u32 v13, v1, v12 // (op 1212) 

v_lshlrev_b32 v12, 4, v13 // (op 1212) 

v_lshrrev_b32 v13, 3, v1 // (op 1212) 
v_lshl_add_u32 v16, v13, 7, v12 // (op 1212) 


s_nop 0
v_readfirstlane_b32 s47, v16 // (op 1212) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -8, v1 // (op 1264) 
v_sub_u32 v13, v1, v12 // (op 1264) 

v_lshlrev_b32 v12, 4, v13 // (op 1264) 

v_lshrrev_b32 v13, 3, v1 // (op 1264) 
v_lshl_add_u32 v16, v13, 7, v12 // (op 1264) 


s_nop 0
v_readfirstlane_b32 s0, v16 // (op 1264) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -4, v1 // (op 1321) 
v_sub_u32 v13, v1, v12 // (op 1321) 

v_and_b32 v12, 4294967292, v1 // (op 1321) 
v_add_u32 v16, v12, v13 // (op 1321) 


v_and_b32 v17, -8, v1 // (op 1397) 
v_sub_u32 v12, v1, v17 // (op 1397) 

v_lshlrev_b32 v13, 4, v12 // (op 1397) 

v_lshrrev_b32 v12, 3, v1 // (op 1397) 
v_lshl_add_u32 v17, v12, 7, v13 // (op 1397) 


s_nop 0
v_readfirstlane_b32 s1, v17 // (op 1397) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v17, -8, v1 // (op 1436) 
v_sub_u32 v12, v1, v17 // (op 1436) 

v_lshlrev_b32 v13, 4, v12 // (op 1436) 

v_lshrrev_b32 v12, 3, v1 // (op 1436) 
v_lshl_add_u32 v17, v12, 7, v13 // (op 1436) 


s_nop 0
v_readfirstlane_b32 s18, v17 // (op 1436) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v17, -4, v1 // (op 1480) 
v_sub_u32 v12, v1, v17 // (op 1480) 

v_and_b32 v13, 4294967292, v1 // (op 1480) 
v_add_u32 v17, v13, v12 // (op 1480) 


s_mov_b32 s19, 0 // (op 312) call()
s_ashr_i64 s[22:23], s[42:43], 7 // (op 311) 
s_mov_b32 s24, s22 // (op 311) convert

s_and_b32 s25, s24, 4294967294 // (op 311) 

s_cmp_lt_u32 s19, s25 // (op 311) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopBottom_KLoop_311 // (op 311) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopBottom_KLoop_311 if false)
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopTop_KLoop_311:
s_add_u32 s19, s19, 2 // (op 313) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:51], v[28:35], a[0:3], v20, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 84) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[44:51], v[36:43], a[4:7], v20, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 347) 


v_mov_b32 v12, v23 // (op 77) 

buffer_load_ubyte v20, v12, s[64:67], 0 offen offset:8 // (op 77) Load value
s_barrier  // (op 1557) 

s_lshl_b64 s[22:23], s[30:31], 5 // (op 883) 
v_mov_b64 v[52:53], s[22:23] // (op 883) convert


v_mov_b64 v[54:55], v[14:15] // (op 883) 
v_mov_b64 v[56:57], v[54:55] // (op 883) 
s_mov_b32 m0, s0 // (op 883) reg expression
s_mov_b32 s22, 256 // (op 883) call()
buffer_load_dwordx4 v56, s[60:63], s22 offen offset: 0 lds // (op 883) Load value direct to lds
v_add_co_u32 v56, s[24:25], v56, v52 // (op 883) least significant half
v_addc_co_u32 v57, s[24:25], 0, v57, s[24:25] // (op 883) most significant half

s_add_u32 m0, m0, 4096 // (op 883) 
buffer_load_dwordx4 v56, s[60:63], s22 offen offset: 0 lds // (op 883) Load value direct to lds





v_mov_b32 v13, v24 // (op 21) 

buffer_load_ubyte v25, v13, s[68:71], 0 offen offset:8 // (op 21) Load value
s_barrier  // (op 1541) 


v_mov_b64 v[52:53], v[18:19] // (op 878) 
s_add_u32 s22, 8192, s47 // (op 878) 

s_mov_b32 m0, s22 // (op 878) reg expression
s_mov_b32 s23, 256 // (op 878) call()
buffer_load_dwordx4 v52, s[48:51], s23 offen offset: 0 lds // (op 878) Load value direct to lds


s_barrier  // (op 1602) 



s_waitcnt vmcnt(6)
ds_write_b8 v17, v22 offset:24832 // (op 240) Store local data 
s_barrier  // (op 1593) 



ds_read_b128 v[28:31], v11 offset:12288 // (op 233) Load local data 
ds_read_b128 v[32:35], v11 offset:12352 // (op 233) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1599) 



ds_read_u8 v21, v10 offset:24832 // (op 237) Load local data 



ds_read_b128 v[36:39], v11 offset:16384 // (op 352) Load local data 
ds_read_b128 v[40:43], v11 offset:16448 // (op 352) Load local data 



ds_read_u8 v27, v10 offset:24960 // (op 355) Load local data 
s_barrier  // (op 1657) 
s_waitcnt vmcnt(5)
s_barrier  // (op 1587) 



ds_read_b128 v[44:47], v4 offset:20480 // (op 231) Load local data 
ds_read_b128 v[48:51], v4 offset:20544 // (op 231) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:51], v[28:35], a[0:3], v26, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 238) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[44:51], v[36:43], a[4:7], v26, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 348) 



buffer_load_ubyte v26, v12, s[64:67], 0 offen offset:12 // (op 235) Load value

s_barrier  // (op 1596) 
v_add_u32 v23, 8, v23 // (op 1102) 

s_lshl_b64 s[22:23], s[30:31], 5 // (op 906) 
v_mov_b64 v[56:57], s[22:23] // (op 906) convert


v_mov_b64 v[58:59], v[54:55] // (op 906) 
s_add_u32 s22, 12288, s18 // (op 906) 
s_mov_b32 m0, s22 // (op 906) reg expression
s_mov_b32 s23, 384 // (op 906) call()
buffer_load_dwordx4 v58, s[60:63], s23 offen offset: 0 lds // (op 906) Load value direct to lds
v_add_co_u32 v58, s[24:25], v58, v56 // (op 906) least significant half
v_addc_co_u32 v59, s[24:25], 0, v59, s[24:25] // (op 906) most significant half

s_add_u32 m0, m0, 4096 // (op 906) 
buffer_load_dwordx4 v58, s[60:63], s23 offen offset: 0 lds // (op 906) Load value direct to lds








v_add_co_u32 v14, vcc, 256, v14 // (op 1038) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 1038) most significant half
buffer_load_ubyte v22, v13, s[68:71], 0 offen offset:12 // (op 242) Load value

s_barrier  // (op 1590) 
v_add_u32 v24, 8, v24 // (op 1166) 


s_add_u32 s22, 20480, s1 // (op 901) 

s_mov_b32 m0, s22 // (op 901) reg expression
s_mov_b32 s23, 384 // (op 901) call()
buffer_load_dwordx4 v52, s[48:51], s23 offen offset: 0 lds // (op 901) Load value direct to lds



s_barrier  // (op 1573) 
v_add_co_u32 v18, vcc, 256, v18 // (op 974) least significant half



v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 974) most significant half
s_waitcnt vmcnt(6)
ds_write_b8 v16, v25 offset:24576 // (op 26) Store local data 
s_barrier  // (op 1563) 



ds_read_b128 v[28:31], v7 // (op 72) Load local data 
ds_read_b128 v[32:35], v7 offset:64 // (op 72) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1580) 



ds_read_u8 v21, v0 offset:24576 // (op 81) Load local data 



ds_read_b128 v[36:39], v7 offset:4096 // (op 343) Load local data 
ds_read_b128 v[40:43], v7 offset:4160 // (op 343) Load local data 



ds_read_u8 v27, v0 offset:24704 // (op 346) Load local data 
s_waitcnt vmcnt(5)
s_barrier  // (op 1547) 



ds_read_b128 v[44:47], v3 offset:8192 // (op 67) Load local data 
ds_read_b128 v[48:51], v3 offset:8256 // (op 67) Load local data 
s_ashr_i64 s[22:23], s[42:43], 7 // (op 311) 
s_mov_b32 s24, s22 // (op 311) convert

s_and_b32 s25, s24, 4294967294 // (op 311) 

s_cmp_lt_u32 s19, s25 // (op 311) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopTop_KLoop_311 // (op 311) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopTop_KLoop_311 if true)
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopBottom_KLoop_311:













s_barrier  // (op 1641) 
s_mul_hi_i32 s47, s16, s10 // (op 936) 
v_lshrrev_b32 v12, 3, v1 // (op 936) 
s_add_i32 s0, s47, s16 // (op 936) 

v_and_b32 v13, -8, v1 // (op 936) 
s_ashr_i32 s1, s0, 31 // (op 936) 
v_sub_u32 v14, v1, v13 // (op 936) 

s_cmp_eq_i32 s10, 0 // (op 936) 
s_mov_b32 s47, scc // (op 936) 
v_lshlrev_b32 v13, 4, v14 // (op 936) 

s_cmp_eq_i32 s47, 0 // (op 936) 
s_cselect_b32 s18, 0, -1 // (op 936) 

s_and_b32 s19, s13, 31 // (op 936) 
s_lshl_b32 s47, 1, s19 // (op 936) 
s_add_i32 s22, s47, s18 // (op 936) 


s_and_b32 s18, s1, s22 // (op 936) 


s_add_i32 s1, s0, s18 // (op 936) 


s_ashr_i32 s0, s1, s19 // (op 936) 


s_lshl_b32 s1, s13, 24 // (op 936) 
s_ashr_i32 s47, s1, 31 // (op 936) 

s_xor_b32 s1, s0, s47 // (op 936) 

s_sub_i32 s0, s1, s47 // (op 936) 


s_mul_hi_i32 s1, s0, s14 // (op 936) 
s_add_i32 s47, s1, s0 // (op 936) 

s_ashr_i32 s1, s47, 31 // (op 936) 
s_cmp_eq_i32 s14, 0 // (op 936) 
s_mov_b32 s18, scc // (op 936) 
s_cmp_eq_i32 s18, 0 // (op 936) 
s_cselect_b32 s19, 0, -1 // (op 936) 

s_and_b32 s18, s15, 31 // (op 936) 
s_lshl_b32 s22, 1, s18 // (op 936) 
s_add_i32 s23, s22, s19 // (op 936) 


s_and_b32 s19, s1, s23 // (op 936) 


s_add_i32 s1, s47, s19 // (op 936) 


s_ashr_i32 s19, s1, s18 // (op 936) 


s_lshl_b32 s1, s15, 24 // (op 936) 
s_ashr_i32 s18, s1, 31 // (op 936) 

s_xor_b32 s1, s19, s18 // (op 936) 

s_sub_i32 s19, s1, s18 // (op 936) 


s_mul_i32 s1, s0, s11 // (op 936) 

s_sub_u32 s0, s16, s1 // (op 936) 

s_mul_i32 s1, s19, s11 // (op 936) 
s_add_u32 s1, s1, s0 // (op 936) 


s_mul_hi_i32 s0, s16, s9 // (op 936) 
s_add_i32 s47, s0, s16 // (op 936) 

s_ashr_i32 s0, s47, 31 // (op 936) 
s_cmp_eq_i32 s9, 0 // (op 936) 
s_mov_b32 s18, scc // (op 936) 
s_cmp_eq_i32 s18, 0 // (op 936) 
s_cselect_b32 s19, 0, -1 // (op 936) 

s_and_b32 s18, s45, 31 // (op 936) 
s_lshl_b32 s22, 1, s18 // (op 936) 
s_add_i32 s23, s22, s19 // (op 936) 


s_and_b32 s19, s0, s23 // (op 936) 


s_add_i32 s0, s47, s19 // (op 936) 


s_ashr_i32 s19, s0, s18 // (op 936) 


s_lshl_b32 s0, s45, 24 // (op 936) 
s_ashr_i32 s18, s0, 31 // (op 936) 

s_xor_b32 s0, s19, s18 // (op 936) 

s_sub_i32 s19, s0, s18 // (op 936) 


s_add_u32 s22, s6, 64 // (op 936) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 936) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 936) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 936) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 936) 

s_mul_i32 s0, s11, s22 // (op 936) 

s_mul_i32 s18, s8, s0 // (op 936) 

s_mul_i32 s0, s19, s18 // (op 936) 

s_sub_u32 s18, s16, s0 // (op 936) 

s_mov_b32 s0, s18 // (op 936) 
s_mul_hi_i32 s47, s0, s12 // (op 936) 
s_add_i32 s22, s47, s0 // (op 936) 


s_ashr_i32 s0, s22, 31 // (op 936) 
s_cmp_eq_i32 s12, 0 // (op 936) 
s_mov_b32 s47, scc // (op 936) 
s_cmp_eq_i32 s47, 0 // (op 936) 
s_cselect_b32 s23, 0, -1 // (op 936) 

s_and_b32 s47, s44, 31 // (op 936) 
s_lshl_b32 s24, 1, s47 // (op 936) 
s_add_i32 s25, s24, s23 // (op 936) 


s_and_b32 s23, s0, s25 // (op 936) 


s_add_i32 s0, s22, s23 // (op 936) 


s_ashr_i32 s22, s0, s47 // (op 936) 


s_lshl_b32 s0, s44, 24 // (op 936) 
s_ashr_i32 s47, s0, 31 // (op 936) 

s_xor_b32 s0, s22, s47 // (op 936) 

s_sub_i32 s22, s0, s47 // (op 936) 


s_mul_i32 s0, s46, s11 // (op 936) 
s_add_u32 s24, s20, 32 // (op 936) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 936) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 936) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 936) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 5 // (op 936) 

s_sub_i32 s23, s24, s0 // (op 936) 

s_mul_i32 s47, s22, s23 // (op 936) 


s_sub_u32 s22, s18, s47 // (op 936) 


s_add_u32 s18, s22, s0 // (op 936) 


s_cmp_eq_i32 s19, 0 // (op 936) 
s_mov_b32 s0, scc // (op 936) 

s_cmp_eq_i32 s0, 0 // (op 936) 
s_cselect_b32 s19, s18, s1 // (op 936) 



v_lshl_add_u32 v14, s19, 5, v12 // (op 936) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 936) 
s_lshr_b32 s47, s0, 1 // (op 936) 

v_lshl_add_u32 v12, s47, 8, v13 // (op 936) 


v_mov_b32 v25, 0 // (op 936) convert
v_mov_b32 v24, v12 // (op 936) convert

v_mul_lo_u32 v13, v14, s27 // (op 936) most significant: low of low * high
v_mul_hi_u32 v23, v14, s26 // (op 936) most significant: high of low * low
v_mul_lo_u32 v12, v14, s26 // (op 936) least significant: low of low * low
v_add_u32 v13, v13, v23 // (op 936) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v24 // (op 936) least significant half
v_addc_co_u32 v13, s[0:1], v13, v25, s[0:1] // (op 936) most significant half



s_mul_i32 s1, s26, s21 // (op 938) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 938) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 938) least significant
s_mul_hi_u32 s18, s26, s20 // (op 938) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 938) most significant: sum
s_add_u32 s1, s1, s18 // (op 938) most significant: sum


s_mov_b32 s48, s28 // (op 938) reg expression
s_mov_b32 s49, s29 // (op 938) reg expression
s_mov_b32 s50, s0 // (op 938) reg expression
s_mov_b32 s51, 131072 // (op 938) call()

s_mul_hi_i32 s47, s16, s10 // (op 1004) 
v_lshrrev_b32 v14, 3, v1 // (op 1004) 
s_add_i32 s0, s47, s16 // (op 1004) 

v_and_b32 v15, -8, v1 // (op 1004) 
s_ashr_i32 s1, s0, 31 // (op 1004) 
v_sub_u32 v16, v1, v15 // (op 1004) 

s_cmp_eq_i32 s10, 0 // (op 1004) 
s_mov_b32 s47, scc // (op 1004) 
v_lshlrev_b32 v15, 4, v16 // (op 1004) 

s_cmp_eq_i32 s47, 0 // (op 1004) 
s_cselect_b32 s18, 0, -1 // (op 1004) 

s_and_b32 s19, s13, 31 // (op 1004) 
s_lshl_b32 s47, 1, s19 // (op 1004) 
s_add_i32 s22, s47, s18 // (op 1004) 


s_and_b32 s18, s1, s22 // (op 1004) 


s_add_i32 s1, s0, s18 // (op 1004) 


s_ashr_i32 s0, s1, s19 // (op 1004) 


s_lshl_b32 s1, s13, 24 // (op 1004) 
s_ashr_i32 s47, s1, 31 // (op 1004) 

s_xor_b32 s1, s0, s47 // (op 1004) 

s_sub_i32 s0, s1, s47 // (op 1004) 


s_mul_hi_i32 s1, s0, s14 // (op 1004) 
s_add_i32 s47, s1, s0 // (op 1004) 

s_ashr_i32 s1, s47, 31 // (op 1004) 
s_cmp_eq_i32 s14, 0 // (op 1004) 
s_mov_b32 s18, scc // (op 1004) 
s_cmp_eq_i32 s18, 0 // (op 1004) 
s_cselect_b32 s19, 0, -1 // (op 1004) 

s_and_b32 s18, s15, 31 // (op 1004) 
s_lshl_b32 s22, 1, s18 // (op 1004) 
s_add_i32 s23, s22, s19 // (op 1004) 


s_and_b32 s19, s1, s23 // (op 1004) 


s_add_i32 s1, s47, s19 // (op 1004) 


s_ashr_i32 s19, s1, s18 // (op 1004) 


s_lshl_b32 s1, s15, 24 // (op 1004) 
s_ashr_i32 s18, s1, 31 // (op 1004) 

s_xor_b32 s1, s19, s18 // (op 1004) 

s_sub_i32 s19, s1, s18 // (op 1004) 


s_add_u32 s22, s6, 64 // (op 1004) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1004) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1004) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1004) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 1004) 

s_mul_i32 s1, s19, s22 // (op 1004) 

s_sub_i32 s47, s0, s1 // (op 1004) 


s_mul_hi_i32 s0, s16, s9 // (op 1004) 
s_add_i32 s1, s0, s16 // (op 1004) 

s_ashr_i32 s0, s1, 31 // (op 1004) 
s_cmp_eq_i32 s9, 0 // (op 1004) 
s_mov_b32 s18, scc // (op 1004) 
s_cmp_eq_i32 s18, 0 // (op 1004) 
s_cselect_b32 s19, 0, -1 // (op 1004) 

s_and_b32 s18, s45, 31 // (op 1004) 
s_lshl_b32 s24, 1, s18 // (op 1004) 
s_add_i32 s25, s24, s19 // (op 1004) 


s_and_b32 s19, s0, s25 // (op 1004) 


s_add_i32 s0, s1, s19 // (op 1004) 


s_ashr_i32 s1, s0, s18 // (op 1004) 


s_lshl_b32 s0, s45, 24 // (op 1004) 
s_ashr_i32 s18, s0, 31 // (op 1004) 

s_xor_b32 s0, s1, s18 // (op 1004) 

s_sub_i32 s1, s0, s18 // (op 1004) 


s_mul_i32 s0, s11, s22 // (op 1004) 

s_mul_i32 s18, s8, s0 // (op 1004) 

s_mul_i32 s0, s1, s18 // (op 1004) 

s_sub_u32 s18, s16, s0 // (op 1004) 

s_mul_hi_i32 s0, s18, s12 // (op 1004) 
s_add_i32 s19, s0, s18 // (op 1004) 


s_ashr_i32 s0, s19, 31 // (op 1004) 
s_cmp_eq_i32 s12, 0 // (op 1004) 
s_mov_b32 s18, scc // (op 1004) 
s_cmp_eq_i32 s18, 0 // (op 1004) 
s_cselect_b32 s22, 0, -1 // (op 1004) 

s_and_b32 s18, s44, 31 // (op 1004) 
s_lshl_b32 s23, 1, s18 // (op 1004) 
s_add_i32 s24, s23, s22 // (op 1004) 


s_and_b32 s25, s0, s24 // (op 1004) 


s_add_i32 s0, s19, s25 // (op 1004) 


s_ashr_i32 s19, s0, s18 // (op 1004) 


s_lshl_b32 s0, s44, 24 // (op 1004) 
s_ashr_i32 s18, s0, 31 // (op 1004) 

s_xor_b32 s0, s19, s18 // (op 1004) 

s_sub_i32 s19, s0, s18 // (op 1004) 


s_cmp_eq_i32 s1, 0 // (op 1004) 
s_mov_b32 s0, scc // (op 1004) 

s_cmp_eq_i32 s0, 0 // (op 1004) 
s_cselect_b32 s1, s19, s47 // (op 1004) 



v_lshl_add_u32 v16, s1, 6, v14 // (op 1004) 


v_mul_lo_u32 v25, v16, s31 // (op 1004) most significant: low of low * high
v_mul_hi_u32 v14, v16, s30 // (op 1004) most significant: high of low * low
v_mul_lo_u32 v24, v16, s30 // (op 1004) least significant: low of low * low
v_add_u32 v25, v25, v14 // (op 1004) most significant: sum


s_ashr_i64 s[0:1], s[42:43], 7 // (op 1004) 
s_lshr_b32 s47, s0, 1 // (op 1004) 

v_lshl_add_u32 v14, s47, 8, v15 // (op 1004) 


v_mov_b32 v17, 0 // (op 1004) convert
v_mov_b32 v16, v14 // (op 1004) convert

v_add_co_u32 v14, s[0:1], v16, v24 // (op 1004) least significant half
v_addc_co_u32 v15, s[0:1], v17, v25, s[0:1] // (op 1004) most significant half



s_mul_i32 s1, s30, s7 // (op 1006) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 1006) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 1006) least significant
s_mul_hi_u32 s18, s30, s6 // (op 1006) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1006) most significant: sum
s_add_u32 s1, s1, s18 // (op 1006) most significant: sum


s_mov_b32 s60, s32 // (op 1006) reg expression
s_mov_b32 s61, s33 // (op 1006) reg expression
s_mov_b32 s62, s0 // (op 1006) reg expression
s_mov_b32 s63, 131072 // (op 1006) call()

s_mul_hi_i32 s47, s16, s10 // (op 1068) 
v_lshrrev_b32 v16, 7, v1 // (op 1068) 
s_add_i32 s0, s47, s16 // (op 1068) 

v_and_b32 v17, 15, v1 // (op 1068) 
s_ashr_i32 s1, s0, 31 // (op 1068) 
v_lshl_add_u32 v18, v16, 4, v17 // (op 1068) 


s_cmp_eq_i32 s10, 0 // (op 1068) 
s_mov_b32 s47, scc // (op 1068) 
v_and_b32 v16, 63, v1 // (op 1068) 
s_cmp_eq_i32 s47, 0 // (op 1068) 
s_cselect_b32 s18, 0, -1 // (op 1068) 

v_lshrrev_b32 v17, 4, v16 // (op 1068) 

s_and_b32 s19, s13, 31 // (op 1068) 
s_lshl_b32 s47, 1, s19 // (op 1068) 
s_add_i32 s22, s47, s18 // (op 1068) 


s_and_b32 s18, s1, s22 // (op 1068) 


s_add_i32 s1, s0, s18 // (op 1068) 


s_ashr_i32 s0, s1, s19 // (op 1068) 


s_lshl_b32 s1, s13, 24 // (op 1068) 
s_ashr_i32 s47, s1, 31 // (op 1068) 

s_xor_b32 s1, s0, s47 // (op 1068) 

s_sub_i32 s0, s1, s47 // (op 1068) 


s_mul_hi_i32 s1, s0, s14 // (op 1068) 
s_add_i32 s47, s1, s0 // (op 1068) 

s_ashr_i32 s1, s47, 31 // (op 1068) 
s_cmp_eq_i32 s14, 0 // (op 1068) 
s_mov_b32 s18, scc // (op 1068) 
s_cmp_eq_i32 s18, 0 // (op 1068) 
s_cselect_b32 s19, 0, -1 // (op 1068) 

s_and_b32 s18, s15, 31 // (op 1068) 
s_lshl_b32 s22, 1, s18 // (op 1068) 
s_add_i32 s23, s22, s19 // (op 1068) 


s_and_b32 s19, s1, s23 // (op 1068) 


s_add_i32 s1, s47, s19 // (op 1068) 


s_ashr_i32 s19, s1, s18 // (op 1068) 


s_lshl_b32 s1, s15, 24 // (op 1068) 
s_ashr_i32 s18, s1, 31 // (op 1068) 

s_xor_b32 s1, s19, s18 // (op 1068) 

s_sub_i32 s19, s1, s18 // (op 1068) 


s_mul_i32 s1, s0, s11 // (op 1068) 

s_sub_u32 s0, s16, s1 // (op 1068) 

s_mul_i32 s1, s19, s11 // (op 1068) 
s_add_u32 s1, s1, s0 // (op 1068) 


s_mul_hi_i32 s0, s16, s9 // (op 1068) 
s_add_i32 s47, s0, s16 // (op 1068) 

s_ashr_i32 s0, s47, 31 // (op 1068) 
s_cmp_eq_i32 s9, 0 // (op 1068) 
s_mov_b32 s18, scc // (op 1068) 
s_cmp_eq_i32 s18, 0 // (op 1068) 
s_cselect_b32 s19, 0, -1 // (op 1068) 

s_and_b32 s18, s45, 31 // (op 1068) 
s_lshl_b32 s22, 1, s18 // (op 1068) 
s_add_i32 s23, s22, s19 // (op 1068) 


s_and_b32 s19, s0, s23 // (op 1068) 


s_add_i32 s0, s47, s19 // (op 1068) 


s_ashr_i32 s19, s0, s18 // (op 1068) 


s_lshl_b32 s0, s45, 24 // (op 1068) 
s_ashr_i32 s18, s0, 31 // (op 1068) 

s_xor_b32 s0, s19, s18 // (op 1068) 

s_sub_i32 s19, s0, s18 // (op 1068) 


s_add_u32 s22, s6, 64 // (op 1068) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1068) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1068) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1068) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 1068) 

s_mul_i32 s0, s11, s22 // (op 1068) 

s_mul_i32 s18, s8, s0 // (op 1068) 

s_mul_i32 s0, s19, s18 // (op 1068) 

s_sub_u32 s18, s16, s0 // (op 1068) 

s_mov_b32 s0, s18 // (op 1068) 
s_mul_hi_i32 s47, s0, s12 // (op 1068) 
s_add_i32 s22, s47, s0 // (op 1068) 


s_ashr_i32 s0, s22, 31 // (op 1068) 
s_cmp_eq_i32 s12, 0 // (op 1068) 
s_mov_b32 s47, scc // (op 1068) 
s_cmp_eq_i32 s47, 0 // (op 1068) 
s_cselect_b32 s23, 0, -1 // (op 1068) 

s_and_b32 s47, s44, 31 // (op 1068) 
s_lshl_b32 s24, 1, s47 // (op 1068) 
s_add_i32 s25, s24, s23 // (op 1068) 


s_and_b32 s23, s0, s25 // (op 1068) 


s_add_i32 s0, s22, s23 // (op 1068) 


s_ashr_i32 s22, s0, s47 // (op 1068) 


s_lshl_b32 s0, s44, 24 // (op 1068) 
s_ashr_i32 s47, s0, 31 // (op 1068) 

s_xor_b32 s0, s22, s47 // (op 1068) 

s_sub_i32 s22, s0, s47 // (op 1068) 


s_mul_i32 s0, s46, s11 // (op 1068) 
s_add_u32 s24, s20, 32 // (op 1068) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 1068) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 1068) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 1068) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 5 // (op 1068) 

s_sub_i32 s23, s24, s0 // (op 1068) 

s_mul_i32 s47, s22, s23 // (op 1068) 


s_sub_u32 s22, s18, s47 // (op 1068) 


s_add_u32 s18, s22, s0 // (op 1068) 


s_cmp_eq_i32 s19, 0 // (op 1068) 
s_mov_b32 s0, scc // (op 1068) 

s_cmp_eq_i32 s0, 0 // (op 1068) 
s_cselect_b32 s19, s18, s1 // (op 1068) 



v_lshl_add_u32 v16, s19, 5, v18 // (op 1068) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 1068) 
s_lshr_b32 s47, s0, 1 // (op 1068) 

v_lshl_add_u32 v18, s47, 3, v17 // (op 1068) 


v_mul_lo_u32 v17, v16, s34 // (op 1068) 
v_add_u32 v17, v17, v18 // (op 1068) 


s_mul_i32 s1, s34, s21 // (op 1070) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 1070) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 1070) least significant
s_mul_hi_u32 s18, s34, s20 // (op 1070) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1070) most significant: sum
s_add_u32 s1, s1, s18 // (op 1070) most significant: sum


s_mov_b32 s64, s36 // (op 1070) reg expression
s_mov_b32 s65, s37 // (op 1070) reg expression
s_mov_b32 s66, s0 // (op 1070) reg expression
s_mov_b32 s67, 131072 // (op 1070) call()

s_mul_hi_i32 s47, s16, s10 // (op 1132) 
v_lshrrev_b32 v16, 2, v1 // (op 1132) 
s_add_i32 s0, s47, s16 // (op 1132) 

v_and_b32 v18, -4, v1 // (op 1132) 
s_ashr_i32 s1, s0, 31 // (op 1132) 
v_sub_u32 v19, v1, v18 // (op 1132) 

s_cmp_eq_i32 s10, 0 // (op 1132) 
s_mov_b32 s47, scc // (op 1132) 
s_cmp_eq_i32 s47, 0 // (op 1132) 
s_cselect_b32 s18, 0, -1 // (op 1132) 

s_and_b32 s19, s13, 31 // (op 1132) 
s_lshl_b32 s47, 1, s19 // (op 1132) 
s_add_i32 s22, s47, s18 // (op 1132) 


s_and_b32 s18, s1, s22 // (op 1132) 


s_add_i32 s1, s0, s18 // (op 1132) 


s_ashr_i32 s0, s1, s19 // (op 1132) 


s_lshl_b32 s1, s13, 24 // (op 1132) 
s_ashr_i32 s47, s1, 31 // (op 1132) 

s_xor_b32 s1, s0, s47 // (op 1132) 

s_sub_i32 s0, s1, s47 // (op 1132) 


s_mul_hi_i32 s1, s0, s14 // (op 1132) 
s_add_i32 s47, s1, s0 // (op 1132) 

s_ashr_i32 s1, s47, 31 // (op 1132) 
s_cmp_eq_i32 s14, 0 // (op 1132) 
s_mov_b32 s18, scc // (op 1132) 
s_cmp_eq_i32 s18, 0 // (op 1132) 
s_cselect_b32 s19, 0, -1 // (op 1132) 

s_and_b32 s18, s15, 31 // (op 1132) 
s_lshl_b32 s22, 1, s18 // (op 1132) 
s_add_i32 s23, s22, s19 // (op 1132) 


s_and_b32 s19, s1, s23 // (op 1132) 


s_add_i32 s1, s47, s19 // (op 1132) 


s_ashr_i32 s19, s1, s18 // (op 1132) 


s_lshl_b32 s1, s15, 24 // (op 1132) 
s_ashr_i32 s18, s1, 31 // (op 1132) 

s_xor_b32 s1, s19, s18 // (op 1132) 

s_sub_i32 s19, s1, s18 // (op 1132) 


s_add_u32 s22, s6, 64 // (op 1132) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 1132) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1132) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1132) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 1132) 

s_mul_i32 s1, s19, s22 // (op 1132) 

s_sub_i32 s47, s0, s1 // (op 1132) 


s_mul_hi_i32 s0, s16, s9 // (op 1132) 
s_add_i32 s1, s0, s16 // (op 1132) 

s_ashr_i32 s0, s1, 31 // (op 1132) 
s_cmp_eq_i32 s9, 0 // (op 1132) 
s_mov_b32 s18, scc // (op 1132) 
s_cmp_eq_i32 s18, 0 // (op 1132) 
s_cselect_b32 s19, 0, -1 // (op 1132) 

s_and_b32 s18, s45, 31 // (op 1132) 
s_lshl_b32 s24, 1, s18 // (op 1132) 
s_add_i32 s25, s24, s19 // (op 1132) 


s_and_b32 s19, s0, s25 // (op 1132) 


s_add_i32 s0, s1, s19 // (op 1132) 


s_ashr_i32 s1, s0, s18 // (op 1132) 


s_lshl_b32 s0, s45, 24 // (op 1132) 
s_ashr_i32 s18, s0, 31 // (op 1132) 

s_xor_b32 s0, s1, s18 // (op 1132) 

s_sub_i32 s1, s0, s18 // (op 1132) 


s_mul_i32 s0, s11, s22 // (op 1132) 

s_mul_i32 s18, s8, s0 // (op 1132) 

s_mul_i32 s0, s1, s18 // (op 1132) 

s_sub_u32 s18, s16, s0 // (op 1132) 

s_mul_hi_i32 s0, s18, s12 // (op 1132) 
s_add_i32 s19, s0, s18 // (op 1132) 


s_ashr_i32 s0, s19, 31 // (op 1132) 
s_cmp_eq_i32 s12, 0 // (op 1132) 
s_mov_b32 s18, scc // (op 1132) 
s_cmp_eq_i32 s18, 0 // (op 1132) 
s_cselect_b32 s22, 0, -1 // (op 1132) 

s_and_b32 s18, s44, 31 // (op 1132) 
s_lshl_b32 s23, 1, s18 // (op 1132) 
s_add_i32 s24, s23, s22 // (op 1132) 


s_and_b32 s25, s0, s24 // (op 1132) 


s_add_i32 s0, s19, s25 // (op 1132) 


s_ashr_i32 s19, s0, s18 // (op 1132) 


s_lshl_b32 s0, s44, 24 // (op 1132) 
s_ashr_i32 s18, s0, 31 // (op 1132) 

s_xor_b32 s0, s19, s18 // (op 1132) 

s_sub_i32 s19, s0, s18 // (op 1132) 


s_cmp_eq_i32 s1, 0 // (op 1132) 
s_mov_b32 s0, scc // (op 1132) 

s_cmp_eq_i32 s0, 0 // (op 1132) 
s_cselect_b32 s1, s19, s47 // (op 1132) 



v_lshl_add_u32 v18, s1, 6, v16 // (op 1132) 


v_mul_lo_u32 v16, v18, s38 // (op 1132) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 1132) 
s_lshr_b32 s47, s0, 1 // (op 1132) 

v_lshl_add_u32 v18, s47, 3, v19 // (op 1132) 


v_add_u32 v19, v18, v16 // (op 1132) 


s_mul_i32 s1, s38, s7 // (op 1134) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 1134) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 1134) least significant
s_mul_hi_u32 s18, s38, s6 // (op 1134) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1134) most significant: sum
s_add_u32 s1, s1, s18 // (op 1134) most significant: sum


s_mov_b32 s68, s40 // (op 1134) reg expression
s_mov_b32 s69, s41 // (op 1134) reg expression
s_mov_b32 s70, s0 // (op 1134) reg expression
s_mov_b32 s71, 131072 // (op 1134) call()

v_and_b32 v16, -8, v1 // (op 1225) 
v_sub_u32 v18, v1, v16 // (op 1225) 

v_lshlrev_b32 v16, 4, v18 // (op 1225) 

v_lshrrev_b32 v18, 3, v1 // (op 1225) 
v_lshl_add_u32 v20, v18, 7, v16 // (op 1225) 


s_nop 0
v_readfirstlane_b32 s47, v20 // (op 1225) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -8, v1 // (op 1277) 
v_sub_u32 v18, v1, v16 // (op 1277) 

v_lshlrev_b32 v16, 4, v18 // (op 1277) 

v_lshrrev_b32 v18, 3, v1 // (op 1277) 
v_lshl_add_u32 v20, v18, 7, v16 // (op 1277) 


s_nop 0
v_readfirstlane_b32 s0, v20 // (op 1277) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -4, v1 // (op 1334) 
v_sub_u32 v18, v1, v16 // (op 1334) 

v_and_b32 v16, 4294967292, v1 // (op 1334) 
v_add_u32 v20, v16, v18 // (op 1334) 


s_ashr_i64 s[18:19], s[42:43], 7 // (op 317) 
s_and_b32 s1, s18, 4294967294 // (op 317) 

s_ashr_i64 s[18:19], s[42:43], 7 // (op 316) 
s_mov_b32 s22, s18 // (op 316) convert

s_cmp_lt_u32 s1, s22 // (op 316) 








s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_316 // (op 316) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_316 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopTop_KLoopTail_316:
s_barrier  // (op 1544) 
s_add_u32 s1, s1, 1 // (op 318) 


v_mov_b64 v[22:23], v[12:13] // (op 888) 
s_add_u32 s18, 8192, s47 // (op 888) 

s_mov_b32 m0, s18 // (op 888) reg expression
buffer_load_dwordx4 v22, s[48:51], 0 offen offset: 0 lds // (op 888) Load value direct to lds


v_add_co_u32 v12, vcc, 128, v12 // (op 926) least significant half
v_addc_co_u32 v13, vcc, 0, v13, vcc // (op 926) most significant half
s_barrier  // (op 1560) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 894) 
v_mov_b64 v[22:23], s[18:19] // (op 894) convert


v_mov_b64 v[24:25], v[14:15] // (op 894) 
v_mov_b64 v[26:27], v[24:25] // (op 894) 
s_mov_b32 m0, s0 // (op 894) reg expression
buffer_load_dwordx4 v26, s[60:63], 0 offen offset: 0 lds // (op 894) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v22 // (op 894) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 894) most significant half

s_add_u32 m0, m0, 4096 // (op 894) 
buffer_load_dwordx4 v26, s[60:63], 0 offen offset: 0 lds // (op 894) Load value direct to lds



v_add_co_u32 v14, vcc, 128, v14 // (op 996) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 996) most significant half


v_mov_b32 v16, v19 // (op 184) 

buffer_load_ubyte v18, v16, s[68:71], 0 offen offset: 0 // (op 184) Load value

v_add_u32 v19, 4, v19 // (op 1124) 
s_barrier  // (op 1576) 
s_waitcnt vmcnt(3)
s_barrier  // (op 1550) 






ds_read_b128 v[22:25], v3 offset:8192 // (op 173) Load local data 
ds_read_b128 v[26:29], v3 offset:8256 // (op 173) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v20, v18 offset:24576 // (op 182) Store local data 

s_barrier  // (op 1566) 



ds_read_b128 v[30:33], v7 // (op 175) Load local data 
ds_read_b128 v[34:37], v7 offset:64 // (op 175) Load local data 


v_mov_b32 v16, v17 // (op 177) 

buffer_load_ubyte v18, v16, s[64:67], 0 offen offset: 0 // (op 177) Load value

v_add_u32 v17, 4, v17 // (op 1060) 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1583) 



ds_read_u8 v16, v0 offset:24576 // (op 179) Load local data 



s_waitcnt vmcnt(0) lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:29], v[30:37], a[0:3], v18, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 180) 
ds_read_b128 v[38:41], v7 offset:4096 // (op 324) Load local data 


ds_read_b128 v[42:45], v7 offset:4160 // (op 324) Load local data 



ds_read_u8 v16, v0 offset:24704 // (op 327) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[22:29], v[38:45], a[4:7], v18, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 328) 




s_ashr_i64 s[18:19], s[42:43], 7 // (op 316) 
s_mov_b32 s22, s18 // (op 316) convert

s_cmp_lt_u32 s1, s22 // (op 316) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopTop_KLoopTail_316 // (op 316) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopTop_KLoopTail_316 if true)
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_316:





















s_barrier  // (op 1645) 



















v_cmp_eq_f32 vcc, 0.00000, v5 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ConditionalFalse_0_beta_125
v_accvgpr_read v10, a0 // (op 46) 
v_accvgpr_read v11, a1 // (op 46) 
v_accvgpr_read v12, a2 // (op 46) 
v_accvgpr_read v13, a3 // (op 46) 
v_mul_f32 v14, v8, v10 // (op 46) 
v_mul_f32 v15, v8, v11 // (op 46) 
v_mul_f32 v16, v8, v12 // (op 46) 
v_mul_f32 v17, v8, v13 // (op 46) 

v_lshrrev_b32 v10, 16, v14 // (op 52) 
v_lshrrev_b32 v11, 16, v15 // (op 52) 
v_lshrrev_b32 v12, 16, v16 // (op 52) 
v_lshrrev_b32 v13, 16, v17 // (op 52) 


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

v_lshrrev_b32 v10, 16, v14 // (op 510) 
v_lshrrev_b32 v11, 16, v15 // (op 510) 
v_lshrrev_b32 v12, 16, v16 // (op 510) 
v_lshrrev_b32 v13, 16, v17 // (op 510) 


v_pack_b32_f16 v14, v10, v11 // (op 511) 
v_pack_b32_f16 v15, v12, v13 // (op 511) 

s_lshl_b32 s0, s4, 6 // (op 511) 
v_add_u32 v0, v9, s0 // (op 511) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 511) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ConditionalBottom_0_beta_125
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ConditionalFalse_0_beta_125:


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
v_and_b32 v12, 65535, v10 // (op 139) 
v_lshrrev_b32 v13, 16, v10 // (op 139) 
v_lshlrev_b32 v12, 16, v12 // (op 139) 
v_lshlrev_b32 v13, 16, v13 // (op 139) 
v_mul_f32 v20, v5, v12 // (op 139) 
v_mul_f32 v21, v5, v13 // (op 139) 
v_and_b32 v12, 65535, v11 // (op 139) 
v_lshrrev_b32 v13, 16, v11 // (op 139) 
v_lshlrev_b32 v12, 16, v12 // (op 139) 
v_lshlrev_b32 v13, 16, v13 // (op 139) 
v_mul_f32 v22, v5, v12 // (op 139) 
v_mul_f32 v23, v5, v13 // (op 139) 


v_add_f32 v10, v20, v16 // (op 135) 
v_add_f32 v11, v21, v17 // (op 135) 
v_add_f32 v12, v22, v18 // (op 135) 
v_add_f32 v13, v23, v19 // (op 135) 


v_lshrrev_b32 v14, 16, v10 // (op 136) 
v_lshrrev_b32 v15, 16, v11 // (op 136) 
v_lshrrev_b32 v16, 16, v12 // (op 136) 
v_lshrrev_b32 v17, 16, v13 // (op 136) 


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
v_and_b32 v12, 65535, v10 // (op 536) 
v_lshrrev_b32 v13, 16, v10 // (op 536) 
v_lshlrev_b32 v12, 16, v12 // (op 536) 
v_lshlrev_b32 v13, 16, v13 // (op 536) 
v_mul_f32 v20, v5, v12 // (op 536) 
v_mul_f32 v21, v5, v13 // (op 536) 
v_and_b32 v12, 65535, v11 // (op 536) 
v_lshrrev_b32 v13, 16, v11 // (op 536) 
v_lshlrev_b32 v12, 16, v12 // (op 536) 
v_lshlrev_b32 v13, 16, v13 // (op 536) 
v_mul_f32 v22, v5, v12 // (op 536) 
v_mul_f32 v23, v5, v13 // (op 536) 


v_add_f32 v10, v20, v16 // (op 532) 
v_add_f32 v11, v21, v17 // (op 532) 
v_add_f32 v12, v22, v18 // (op 532) 
v_add_f32 v13, v23, v19 // (op 532) 


v_lshrrev_b32 v14, 16, v10 // (op 533) 
v_lshrrev_b32 v15, 16, v11 // (op 533) 
v_lshrrev_b32 v16, 16, v12 // (op 533) 
v_lshrrev_b32 v17, 16, v13 // (op 533) 


v_pack_b32_f16 v10, v14, v15 // (op 534) 
v_pack_b32_f16 v11, v16, v17 // (op 534) 

s_lshl_b32 s0, s4, 6 // (op 534) 
v_add_u32 v0, v9, s0 // (op 534) 


buffer_store_dwordx2 v[10:11], v0, s[56:59], 0 offen // (op 534) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__ConditionalBottom_0_beta_125:










s_endpgm  // End of RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_
.LRR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__end:
.size RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_, .LRR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM__end-RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 68
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 25088 // lds bytes
  .amdhsa_accum_offset 60
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
    DataType: B8
    DataTypeA: B8
    DataTypeB: B8
    DestDataType: b
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
  - .name: RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_BF8_BF8_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x64x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 25088
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 74
    .vgpr_count: 60
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
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: BFloat16, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BFloat16, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 72
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: BFloat16, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BFloat16, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: BF8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF8, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: BF8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF8, pointerType: PointerGlobal}
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

