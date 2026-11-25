// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_,@function
RR_GEMM_TN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_:
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
RR_GEMM_TN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 1167) 
s_mul_hi_i32 s1, s16, s10 // (op 1355) 
v_lshrrev_b32 v0, 6, v1 // (op 1281) 
v_and_b32 v3, 1, v0 // (op 1281) 

v_and_b32 v0, 15, v1 // (op 1281) 
v_lshl_add_u32 v4, v3, 4, v0 // (op 1281) 


v_and_b32 v0, 63, v1 // (op 1281) 
v_lshrrev_b32 v3, 4, v0 // (op 1281) 

v_lshlrev_b32 v0, 5, v3 // (op 1281) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 1281) 


v_mul_lo_u32 v0, v3, 6 // (op 1281) 

v_lshrrev_b32 v3, 3, v0 // (op 1281) 

v_lshrrev_b32 v0, 6, v1 // (op 1425) 
v_and_b32 v4, 1, v0 // (op 1425) 

v_and_b32 v0, 15, v1 // (op 1425) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 1425) 


v_and_b32 v0, 63, v1 // (op 1425) 
v_lshrrev_b32 v4, 4, v0 // (op 1425) 

v_lshlrev_b32 v0, 5, v4 // (op 1425) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 1425) 


v_mul_lo_u32 v0, v4, 6 // (op 1425) 

v_lshrrev_b32 v4, 3, v0 // (op 1425) 

v_lshrrev_b32 v0, 7, v1 // (op 1167) 
s_add_i32 s18, s0, s16 // (op 1167) 

v_and_b32 v5, 63, v1 // (op 1167) 
s_ashr_i32 s0, s18, 31 // (op 1167) 
v_lshrrev_b32 v6, 4, v5 // (op 1167) 
s_cmp_eq_i32 s10, 0 // (op 1167) 
s_mov_b32 s19, scc // (op 1167) 
v_lshlrev_b32 v7, 2, v6 // (op 1167) 

s_cmp_eq_i32 s19, 0 // (op 1167) 
s_cselect_b32 s49, 0, -1 // (op 1167) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 1167) 


s_and_b32 s19, s13, 31 // (op 1167) 
v_lshrrev_b32 v0, 2, v5 // (op 1167) 

v_and_b32 v5, 3, v0 // (op 1167) 

v_lshrrev_b32 v0, 7, v1 // (op 1228) 
v_and_b32 v7, 15, v1 // (op 1228) 
v_lshl_add_u32 v8, v0, 4, v7 // (op 1228) 


v_and_b32 v0, 48, v1 // (op 1228) 
v_lshl_add_u32 v7, v8, 7, v0 // (op 1228) 


v_lshrrev_b32 v0, 7, v1 // (op 1339) 
v_and_b32 v8, 15, v1 // (op 1339) 
v_lshl_add_u32 v9, v0, 4, v8 // (op 1339) 


v_and_b32 v0, 63, v1 // (op 1339) 
v_lshrrev_b32 v8, 4, v0 // (op 1339) 

v_lshl_add_u32 v0, v9, 2, v8 // (op 1339) 


v_lshrrev_b32 v8, 7, v1 // (op 1355) 
v_and_b32 v9, 63, v1 // (op 1355) 
v_lshrrev_b32 v10, 7, v1 // (op 1401) 
v_and_b32 v11, 15, v1 // (op 1401) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 1401) 


v_and_b32 v10, 48, v1 // (op 1401) 
v_lshl_add_u32 v11, v12, 7, v10 // (op 1401) 


v_lshrrev_b32 v10, 7, v1 // (op 1454) 
v_and_b32 v12, 15, v1 // (op 1454) 
v_lshl_add_u32 v13, v10, 4, v12 // (op 1454) 


v_and_b32 v10, 63, v1 // (op 1454) 
v_lshrrev_b32 v12, 4, v10 // (op 1454) 

v_lshl_add_u32 v10, v13, 2, v12 // (op 1454) 


s_lshl_b32 s50, 1, s19 // (op 1167) 
v_and_b32 v12, 3, v1 // (op 1167) 
s_add_i32 s51, s50, s49 // (op 1167) 


s_and_b32 s49, s0, s51 // (op 1167) 


s_add_i32 s0, s1, s16 // (op 1355) 

s_ashr_i32 s1, s0, 31 // (op 1355) 
v_lshl_add_u32 v13, v5, 2, v12 // (op 1167) 


s_add_i32 s50, s18, s49 // (op 1167) 


v_lshrrev_b32 v5, 6, v1 // (op 1167) 
s_ashr_i32 s18, s50, s19 // (op 1167) 


v_and_b32 v12, 1, v5 // (op 1167) 

s_lshl_b32 s19, s13, 24 // (op 1167) 
v_lshl_add_u32 v5, v12, 4, v13 // (op 1167) 


s_ashr_i32 s49, s19, 31 // (op 1167) 

s_xor_b32 s19, s18, s49 // (op 1167) 

s_sub_i32 s18, s19, s49 // (op 1167) 


s_mul_hi_i32 s19, s18, s14 // (op 1167) 
s_add_i32 s49, s19, s18 // (op 1167) 

s_ashr_i32 s19, s49, 31 // (op 1167) 
s_cmp_eq_i32 s14, 0 // (op 1167) 
s_mov_b32 s50, scc // (op 1167) 
s_cmp_eq_i32 s50, 0 // (op 1167) 
s_cselect_b32 s51, 0, -1 // (op 1167) 

s_and_b32 s50, s15, 31 // (op 1167) 
v_lshrrev_b32 v12, 4, v9 // (op 1355) 
s_cmp_eq_i32 s10, 0 // (op 1355) 
s_mov_b32 s52, scc // (op 1355) 
v_lshlrev_b32 v13, 2, v12 // (op 1355) 

s_cmp_eq_i32 s52, 0 // (op 1355) 
s_cselect_b32 s53, 0, -1 // (op 1355) 

s_lshl_b32 s52, 1, s50 // (op 1167) 
v_lshl_add_u32 v12, v8, 4, v13 // (op 1355) 


v_lshrrev_b32 v8, 2, v9 // (op 1355) 

s_add_i32 s54, s52, s51 // (op 1167) 


s_and_b32 s51, s19, s54 // (op 1167) 


s_add_i32 s19, s49, s51 // (op 1167) 


s_ashr_i32 s49, s19, s50 // (op 1167) 


s_lshl_b32 s19, s15, 24 // (op 1167) 
s_ashr_i32 s50, s19, 31 // (op 1167) 

s_xor_b32 s19, s49, s50 // (op 1167) 

s_sub_i32 s49, s19, s50 // (op 1167) 


s_mul_i32 s19, s18, s11 // (op 1167) 
s_sub_u32 s50, s16, s19 // (op 1167) 

s_mul_i32 s19, s49, s11 // (op 1167) 
s_add_u32 s19, s19, s50 // (op 1167) 

s_mul_hi_i32 s50, s16, s9 // (op 1167) 
s_add_i32 s51, s50, s16 // (op 1167) 

s_ashr_i32 s50, s51, 31 // (op 1167) 
s_cmp_eq_i32 s9, 0 // (op 1167) 
s_mov_b32 s52, scc // (op 1167) 
s_cmp_eq_i32 s52, 0 // (op 1167) 
s_cselect_b32 s54, 0, -1 // (op 1167) 

s_and_b32 s52, s13, 31 // (op 1355) 
v_and_b32 v9, 3, v8 // (op 1355) 

s_lshl_b32 s55, 1, s52 // (op 1355) 
v_and_b32 v8, 3, v1 // (op 1355) 
s_add_i32 s56, s55, s53 // (op 1355) 


s_and_b32 s53, s1, s56 // (op 1355) 


v_lshl_add_u32 v13, v9, 2, v8 // (op 1355) 


s_add_i32 s1, s0, s53 // (op 1355) 


v_lshrrev_b32 v8, 6, v1 // (op 1355) 
s_ashr_i32 s0, s1, s52 // (op 1355) 


v_and_b32 v9, 1, v8 // (op 1355) 

s_lshl_b32 s1, s13, 24 // (op 1355) 
v_lshl_add_u32 v8, v9, 4, v13 // (op 1355) 


s_ashr_i32 s52, s1, 31 // (op 1355) 

s_xor_b32 s1, s0, s52 // (op 1355) 

s_sub_i32 s0, s1, s52 // (op 1355) 


s_mul_hi_i32 s1, s0, s14 // (op 1355) 
s_add_i32 s52, s1, s0 // (op 1355) 

s_ashr_i32 s1, s52, 31 // (op 1355) 
s_cmp_eq_i32 s14, 0 // (op 1355) 
s_mov_b32 s53, scc // (op 1355) 
s_cmp_eq_i32 s53, 0 // (op 1355) 
s_cselect_b32 s55, 0, -1 // (op 1355) 

s_and_b32 s53, s15, 31 // (op 1355) 
s_lshl_b32 s56, 1, s53 // (op 1355) 
s_add_i32 s57, s56, s55 // (op 1355) 


s_and_b32 s55, s1, s57 // (op 1355) 


s_add_i32 s1, s52, s55 // (op 1355) 


s_ashr_i32 s52, s1, s53 // (op 1355) 


s_lshl_b32 s1, s15, 24 // (op 1355) 
s_ashr_i32 s53, s1, 31 // (op 1355) 

s_xor_b32 s1, s52, s53 // (op 1355) 

s_sub_i32 s52, s1, s53 // (op 1355) 


s_mul_i32 s1, s0, s11 // (op 1355) 
s_sub_u32 s53, s16, s1 // (op 1355) 

s_mul_i32 s1, s52, s11 // (op 1355) 
s_add_u32 s1, s1, s53 // (op 1355) 

s_mul_hi_i32 s53, s16, s9 // (op 1355) 
s_add_i32 s55, s53, s16 // (op 1355) 

s_ashr_i32 s53, s55, 31 // (op 1355) 
s_cmp_eq_i32 s9, 0 // (op 1355) 
s_mov_b32 s56, scc // (op 1355) 
s_cmp_eq_i32 s56, 0 // (op 1355) 
s_cselect_b32 s57, 0, -1 // (op 1355) 


s_waitcnt lgkmcnt(0)
s_and_b32 s56, s45, 31 // (op 1167) 
s_lshl_b32 s58, 1, s56 // (op 1167) 
s_add_i32 s59, s58, s54 // (op 1167) 


s_and_b32 s54, s50, s59 // (op 1167) 


s_add_i32 s50, s51, s54 // (op 1167) 


s_ashr_i32 s51, s50, s56 // (op 1167) 


s_lshl_b32 s50, s45, 24 // (op 1167) 
s_ashr_i32 s54, s50, 31 // (op 1167) 

s_xor_b32 s50, s51, s54 // (op 1167) 

s_sub_i32 s51, s50, s54 // (op 1167) 


s_and_b32 s50, s45, 31 // (op 1355) 
s_lshl_b32 s54, 1, s50 // (op 1355) 
s_add_i32 s56, s54, s57 // (op 1355) 


s_and_b32 s54, s53, s56 // (op 1355) 


s_add_i32 s53, s55, s54 // (op 1355) 


s_ashr_i32 s54, s53, s50 // (op 1355) 


s_lshl_b32 s50, s45, 24 // (op 1355) 
s_ashr_i32 s53, s50, 31 // (op 1355) 

s_xor_b32 s50, s54, s53 // (op 1355) 

s_sub_i32 s54, s50, s53 // (op 1355) 


s_add_u32 s56, s4, 32 // (op 1167) Start of Int64 add, locking SCC
s_addc_u32 s57, s5, 0 // (op 1167) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 1167) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 1167) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 5 // (op 1167) 

s_mul_i32 s50, s11, s56 // (op 1167) 
s_mul_i32 s53, s8, s50 // (op 1167) 

s_mul_i32 s50, s51, s53 // (op 1167) 

s_sub_u32 s53, s16, s50 // (op 1167) 

s_mov_b32 s50, s53 // (op 1167) 
s_mul_hi_i32 s55, s50, s12 // (op 1167) 
s_add_i32 s58, s55, s50 // (op 1167) 


s_ashr_i32 s50, s58, 31 // (op 1167) 
s_cmp_eq_i32 s12, 0 // (op 1167) 
s_mov_b32 s55, scc // (op 1167) 
s_cmp_eq_i32 s55, 0 // (op 1167) 
s_cselect_b32 s59, 0, -1 // (op 1167) 

s_and_b32 s55, s44, 31 // (op 1167) 
s_lshl_b32 s60, 1, s55 // (op 1167) 
s_add_i32 s61, s60, s59 // (op 1167) 


s_and_b32 s59, s50, s61 // (op 1167) 


s_add_i32 s50, s58, s59 // (op 1167) 


s_ashr_i32 s58, s50, s55 // (op 1167) 


s_lshl_b32 s50, s44, 24 // (op 1167) 
s_ashr_i32 s55, s50, 31 // (op 1167) 

s_xor_b32 s50, s58, s55 // (op 1167) 

s_sub_i32 s58, s50, s55 // (op 1167) 


s_mul_i32 s50, s46, s11 // (op 1167) 
s_add_u32 s60, s6, 64 // (op 1167) Start of Int64 add, locking SCC
s_addc_u32 s61, s7, 0 // (op 1167) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 1167) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 1167) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 6 // (op 1167) 

s_sub_i32 s55, s60, s50 // (op 1167) 

s_mul_i32 s59, s58, s55 // (op 1167) 

s_sub_u32 s55, s53, s59 // (op 1167) 


s_add_u32 s53, s55, s50 // (op 1167) 


s_cmp_eq_i32 s51, 0 // (op 1167) 
s_mov_b32 s50, scc // (op 1167) 

s_cmp_eq_i32 s50, 0 // (op 1167) 
s_cselect_b32 s51, s53, s19 // (op 1167) 


s_mul_i32 s19, s49, s56 // (op 1167) 


v_lshl_add_u32 v9, s51, 6, v6 // (op 1167) 


s_sub_i32 s49, s18, s19 // (op 1167) 


s_cmp_eq_i32 s50, 0 // (op 1167) 
s_cselect_b32 s51, s58, s49 // (op 1167) 



v_lshl_add_u32 v6, s51, 5, v5 // (op 1167) 


v_mul_lo_u32 v5, v6, s2 // (op 1167) 

v_add_lshl_u32 v6, v9, v5, 1 // (op 1167) 


s_add_u32 s18, s4, 32 // (op 1355) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 1355) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 1355) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 1355) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 5 // (op 1355) 

s_mul_i32 s53, s11, s18 // (op 1355) 
s_mul_i32 s49, s8, s53 // (op 1355) 

s_mul_i32 s53, s54, s49 // (op 1355) 

s_sub_u32 s49, s16, s53 // (op 1355) 

s_mov_b32 s53, s49 // (op 1355) 
s_mul_hi_i32 s50, s53, s12 // (op 1355) 
s_add_i32 s51, s50, s53 // (op 1355) 


s_ashr_i32 s50, s51, 31 // (op 1355) 
s_cmp_eq_i32 s12, 0 // (op 1355) 
s_mov_b32 s53, scc // (op 1355) 
s_cmp_eq_i32 s53, 0 // (op 1355) 
s_cselect_b32 s55, 0, -1 // (op 1355) 

s_and_b32 s53, s44, 31 // (op 1355) 
s_lshl_b32 s56, 1, s53 // (op 1355) 
s_add_i32 s57, s56, s55 // (op 1355) 


s_and_b32 s55, s50, s57 // (op 1355) 


s_add_i32 s50, s51, s55 // (op 1355) 


s_ashr_i32 s51, s50, s53 // (op 1355) 


s_lshl_b32 s50, s44, 24 // (op 1355) 
s_ashr_i32 s53, s50, 31 // (op 1355) 

s_xor_b32 s50, s51, s53 // (op 1355) 

s_sub_i32 s51, s50, s53 // (op 1355) 


s_mul_i32 s50, s46, s11 // (op 1355) 
s_add_u32 s56, s6, 64 // (op 1355) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 1355) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 1355) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 1355) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 1355) 

s_sub_i32 s53, s56, s50 // (op 1355) 

s_mul_i32 s55, s51, s53 // (op 1355) 

s_sub_u32 s53, s49, s55 // (op 1355) 


s_add_u32 s49, s53, s50 // (op 1355) 


s_cmp_eq_i32 s54, 0 // (op 1355) 
s_mov_b32 s50, scc // (op 1355) 

s_cmp_eq_i32 s50, 0 // (op 1355) 
s_cselect_b32 s53, s49, s1 // (op 1355) 


s_mul_i32 s1, s52, s18 // (op 1355) 


v_lshl_add_u32 v5, s53, 6, v12 // (op 1355) 


s_sub_i32 s49, s0, s1 // (op 1355) 


s_cmp_eq_i32 s50, 0 // (op 1355) 
s_cselect_b32 s0, s51, s49 // (op 1355) 



v_lshl_add_u32 v9, s0, 5, v8 // (op 1355) 


v_mul_lo_u32 v8, v9, s20 // (op 1355) 

v_add_lshl_u32 v9, v5, v8, 1 // (op 1355) 


s_mul_i32 s1, s2, s5 // (op 1169) most significant: low of low * high
s_mul_i32 s18, s3, s4 // (op 1169) most significant: low of high * low
s_mul_i32 s0, s2, s4 // (op 1169) least significant
s_mul_hi_u32 s19, s2, s4 // (op 1169) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 1169) most significant: sum
s_add_u32 s1, s1, s19 // (op 1169) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 1169) 

s_mul_i32 s1, s20, s5 // (op 1357) most significant: low of low * high
s_mul_i32 s49, s21, s4 // (op 1357) most significant: low of high * low
s_mul_i32 s0, s20, s4 // (op 1357) least significant
s_mul_hi_u32 s50, s20, s4 // (op 1357) most significant: high of low * low
s_add_u32 s1, s1, s49 // (op 1357) most significant: sum
s_add_u32 s1, s1, s50 // (op 1357) most significant: sum


s_lshl_b64 s[50:51], s[0:1], 1 // (op 1357) 

s_mov_b32 s52, s22 // (op 1169) reg expression
s_mov_b32 s53, s23 // (op 1169) reg expression
s_mov_b32 s54, s18 // (op 1169) reg expression
s_mov_b32 s55, 131072 // (op 1169) call()

s_mov_b32 s56, s24 // (op 1357) reg expression

s_mov_b32 s57, s25 // (op 1357) reg expression
s_mov_b32 s58, s50 // (op 1357) reg expression
s_mov_b32 s59, 131072 // (op 1357) call()


v_mov_b32 v5, s47 // (op 2) Move value
v_mov_b32 v8, s48 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 61) call()
v_accvgpr_write a1, 0.00000 // (op 61) call()
v_accvgpr_write a2, 0.00000 // (op 61) call()
v_accvgpr_write a3, 0.00000 // (op 61) call()
v_accvgpr_write a4, 0.00000 // (op 313) call()
v_accvgpr_write a5, 0.00000 // (op 313) call()
v_accvgpr_write a6, 0.00000 // (op 313) call()
v_accvgpr_write a7, 0.00000 // (op 313) call()
s_mul_hi_i32 s0, s16, s10 // (op 934) 
v_lshrrev_b32 v12, 3, v1 // (op 934) 
s_add_i32 s1, s0, s16 // (op 934) 

v_and_b32 v13, -8, v1 // (op 934) 
s_ashr_i32 s0, s1, 31 // (op 934) 
v_sub_u32 v14, v1, v13 // (op 934) 

s_cmp_eq_i32 s10, 0 // (op 934) 
s_mov_b32 s18, scc // (op 934) 
v_lshlrev_b32 v13, 4, v14 // (op 934) 

s_cmp_eq_i32 s18, 0 // (op 934) 
s_cselect_b32 s19, 0, -1 // (op 934) 

v_mov_b32 v15, 0 // (op 934) convert
v_mov_b32 v14, v13 // (op 934) convert

s_and_b32 s18, s13, 31 // (op 934) 
s_lshl_b32 s22, 1, s18 // (op 934) 
s_add_i32 s23, s22, s19 // (op 934) 


s_and_b32 s19, s0, s23 // (op 934) 


s_add_i32 s0, s1, s19 // (op 934) 


s_ashr_i32 s1, s0, s18 // (op 934) 


s_lshl_b32 s0, s13, 24 // (op 934) 
s_ashr_i32 s18, s0, 31 // (op 934) 

s_xor_b32 s0, s1, s18 // (op 934) 

s_sub_i32 s1, s0, s18 // (op 934) 


s_mul_hi_i32 s0, s1, s14 // (op 934) 
s_add_i32 s18, s0, s1 // (op 934) 

s_ashr_i32 s0, s18, 31 // (op 934) 
s_cmp_eq_i32 s14, 0 // (op 934) 
s_mov_b32 s19, scc // (op 934) 
s_cmp_eq_i32 s19, 0 // (op 934) 
s_cselect_b32 s22, 0, -1 // (op 934) 

s_and_b32 s19, s15, 31 // (op 934) 
s_lshl_b32 s23, 1, s19 // (op 934) 
s_add_i32 s24, s23, s22 // (op 934) 


s_and_b32 s25, s0, s24 // (op 934) 


s_add_i32 s0, s18, s25 // (op 934) 


s_ashr_i32 s18, s0, s19 // (op 934) 


s_lshl_b32 s0, s15, 24 // (op 934) 
s_ashr_i32 s19, s0, 31 // (op 934) 

s_xor_b32 s0, s18, s19 // (op 934) 

s_sub_i32 s18, s0, s19 // (op 934) 


s_mul_i32 s0, s1, s11 // (op 934) 

s_sub_u32 s1, s16, s0 // (op 934) 

s_mul_i32 s0, s18, s11 // (op 934) 
s_add_u32 s0, s0, s1 // (op 934) 


s_mul_hi_i32 s1, s16, s9 // (op 934) 
s_add_i32 s18, s1, s16 // (op 934) 

s_ashr_i32 s1, s18, 31 // (op 934) 
s_cmp_eq_i32 s9, 0 // (op 934) 
s_mov_b32 s19, scc // (op 934) 
s_cmp_eq_i32 s19, 0 // (op 934) 
s_cselect_b32 s22, 0, -1 // (op 934) 

s_and_b32 s19, s45, 31 // (op 934) 
s_lshl_b32 s23, 1, s19 // (op 934) 
s_add_i32 s24, s23, s22 // (op 934) 


s_and_b32 s25, s1, s24 // (op 934) 


s_add_i32 s1, s18, s25 // (op 934) 


s_ashr_i32 s18, s1, s19 // (op 934) 


s_lshl_b32 s1, s45, 24 // (op 934) 
s_ashr_i32 s19, s1, 31 // (op 934) 

s_xor_b32 s1, s18, s19 // (op 934) 

s_sub_i32 s18, s1, s19 // (op 934) 


s_add_u32 s22, s4, 32 // (op 934) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 934) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 934) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 934) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 934) 

s_mul_i32 s1, s11, s22 // (op 934) 

s_mul_i32 s19, s8, s1 // (op 934) 

s_mul_i32 s1, s18, s19 // (op 934) 

s_sub_u32 s19, s16, s1 // (op 934) 

s_mov_b32 s1, s19 // (op 934) 
s_mul_hi_i32 s22, s1, s12 // (op 934) 
s_add_i32 s23, s22, s1 // (op 934) 


s_ashr_i32 s1, s23, 31 // (op 934) 
s_cmp_eq_i32 s12, 0 // (op 934) 
s_mov_b32 s22, scc // (op 934) 
s_cmp_eq_i32 s22, 0 // (op 934) 
s_cselect_b32 s24, 0, -1 // (op 934) 

s_and_b32 s22, s44, 31 // (op 934) 
s_lshl_b32 s25, 1, s22 // (op 934) 
s_add_i32 s47, s25, s24 // (op 934) 


s_and_b32 s24, s1, s47 // (op 934) 


s_add_i32 s1, s23, s24 // (op 934) 


s_ashr_i32 s23, s1, s22 // (op 934) 


s_lshl_b32 s1, s44, 24 // (op 934) 
s_ashr_i32 s22, s1, 31 // (op 934) 

s_xor_b32 s1, s23, s22 // (op 934) 

s_sub_i32 s23, s1, s22 // (op 934) 


s_mul_i32 s1, s46, s11 // (op 934) 
s_add_u32 s24, s6, 64 // (op 934) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 934) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 934) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 934) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 934) 

s_sub_i32 s22, s24, s1 // (op 934) 

s_mul_i32 s24, s23, s22 // (op 934) 


s_sub_u32 s25, s19, s24 // (op 934) 


s_add_u32 s19, s25, s1 // (op 934) 


s_cmp_eq_i32 s18, 0 // (op 934) 
s_mov_b32 s1, scc // (op 934) 

s_cmp_eq_i32 s1, 0 // (op 934) 
s_cselect_b32 s18, s19, s0 // (op 934) 



v_lshl_add_u32 v13, s18, 6, v12 // (op 934) 


v_mul_lo_u32 v17, v13, s27 // (op 934) most significant: low of low * high
v_mul_hi_u32 v12, v13, s26 // (op 934) most significant: high of low * low
v_mul_lo_u32 v16, v13, s26 // (op 934) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 934) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 934) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 934) most significant half



s_mul_i32 s1, s26, s7 // (op 936) most significant: low of low * high
s_mul_i32 s18, s27, s6 // (op 936) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 936) least significant
s_mul_hi_u32 s19, s26, s6 // (op 936) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 936) most significant: sum
s_add_u32 s1, s1, s19 // (op 936) most significant: sum


s_mov_b32 s48, s28 // (op 936) reg expression
s_mov_b32 s49, s29 // (op 936) reg expression
s_mov_b32 s50, s0 // (op 936) reg expression
s_mov_b32 s51, 131072 // (op 936) call()

s_mul_hi_i32 s47, s16, s10 // (op 953) 
v_lshrrev_b32 v12, 3, v1 // (op 953) 
s_add_i32 s0, s47, s16 // (op 953) 

v_and_b32 v13, -8, v1 // (op 953) 
s_ashr_i32 s1, s0, 31 // (op 953) 
v_sub_u32 v14, v1, v13 // (op 953) 

s_cmp_eq_i32 s10, 0 // (op 953) 
s_mov_b32 s47, scc // (op 953) 
v_lshlrev_b32 v13, 4, v14 // (op 953) 

s_cmp_eq_i32 s47, 0 // (op 953) 
s_cselect_b32 s18, 0, -1 // (op 953) 

v_mov_b32 v15, 0 // (op 953) convert
v_mov_b32 v14, v13 // (op 953) convert

s_and_b32 s19, s13, 31 // (op 953) 
s_lshl_b32 s47, 1, s19 // (op 953) 
s_add_i32 s22, s47, s18 // (op 953) 


s_and_b32 s18, s1, s22 // (op 953) 


s_add_i32 s1, s0, s18 // (op 953) 


s_ashr_i32 s0, s1, s19 // (op 953) 


s_lshl_b32 s1, s13, 24 // (op 953) 
s_ashr_i32 s47, s1, 31 // (op 953) 

s_xor_b32 s1, s0, s47 // (op 953) 

s_sub_i32 s0, s1, s47 // (op 953) 


s_mul_hi_i32 s1, s0, s14 // (op 953) 
s_add_i32 s47, s1, s0 // (op 953) 

s_ashr_i32 s1, s47, 31 // (op 953) 
s_cmp_eq_i32 s14, 0 // (op 953) 
s_mov_b32 s18, scc // (op 953) 
s_cmp_eq_i32 s18, 0 // (op 953) 
s_cselect_b32 s19, 0, -1 // (op 953) 

s_and_b32 s18, s15, 31 // (op 953) 
s_lshl_b32 s22, 1, s18 // (op 953) 
s_add_i32 s23, s22, s19 // (op 953) 


s_and_b32 s19, s1, s23 // (op 953) 


s_add_i32 s1, s47, s19 // (op 953) 


s_ashr_i32 s19, s1, s18 // (op 953) 


s_lshl_b32 s1, s15, 24 // (op 953) 
s_ashr_i32 s18, s1, 31 // (op 953) 

s_xor_b32 s1, s19, s18 // (op 953) 

s_sub_i32 s19, s1, s18 // (op 953) 


s_mul_i32 s1, s0, s11 // (op 953) 

s_sub_u32 s0, s16, s1 // (op 953) 

s_mul_i32 s1, s19, s11 // (op 953) 
s_add_u32 s1, s1, s0 // (op 953) 


s_mul_hi_i32 s0, s16, s9 // (op 953) 
s_add_i32 s47, s0, s16 // (op 953) 

s_ashr_i32 s0, s47, 31 // (op 953) 
s_cmp_eq_i32 s9, 0 // (op 953) 
s_mov_b32 s18, scc // (op 953) 
s_cmp_eq_i32 s18, 0 // (op 953) 
s_cselect_b32 s19, 0, -1 // (op 953) 

s_and_b32 s18, s45, 31 // (op 953) 
s_lshl_b32 s22, 1, s18 // (op 953) 
s_add_i32 s23, s22, s19 // (op 953) 


s_and_b32 s19, s0, s23 // (op 953) 


s_add_i32 s0, s47, s19 // (op 953) 


s_ashr_i32 s19, s0, s18 // (op 953) 


s_lshl_b32 s0, s45, 24 // (op 953) 
s_ashr_i32 s18, s0, 31 // (op 953) 

s_xor_b32 s0, s19, s18 // (op 953) 

s_sub_i32 s19, s0, s18 // (op 953) 


s_add_u32 s22, s4, 32 // (op 953) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 953) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 953) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 953) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 953) 

s_mul_i32 s0, s11, s22 // (op 953) 

s_mul_i32 s18, s8, s0 // (op 953) 

s_mul_i32 s0, s19, s18 // (op 953) 

s_sub_u32 s18, s16, s0 // (op 953) 

s_mov_b32 s0, s18 // (op 953) 
s_mul_hi_i32 s47, s0, s12 // (op 953) 
s_add_i32 s22, s47, s0 // (op 953) 


s_ashr_i32 s0, s22, 31 // (op 953) 
s_cmp_eq_i32 s12, 0 // (op 953) 
s_mov_b32 s47, scc // (op 953) 
s_cmp_eq_i32 s47, 0 // (op 953) 
s_cselect_b32 s23, 0, -1 // (op 953) 

s_and_b32 s47, s44, 31 // (op 953) 
s_lshl_b32 s24, 1, s47 // (op 953) 
s_add_i32 s25, s24, s23 // (op 953) 


s_and_b32 s23, s0, s25 // (op 953) 


s_add_i32 s0, s22, s23 // (op 953) 


s_ashr_i32 s22, s0, s47 // (op 953) 


s_lshl_b32 s0, s44, 24 // (op 953) 
s_ashr_i32 s47, s0, 31 // (op 953) 

s_xor_b32 s0, s22, s47 // (op 953) 

s_sub_i32 s22, s0, s47 // (op 953) 


s_mul_i32 s0, s46, s11 // (op 953) 
s_add_u32 s24, s6, 64 // (op 953) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 953) End of Int64 add, unlocking SCC
s_sub_u32 s60, s24, 1 // (op 953) least significant half; sets scc
s_subb_u32 s61, s25, 0 // (op 953) most significant half; uses scc

s_ashr_i64 s[24:25], s[60:61], 6 // (op 953) 

s_sub_i32 s23, s24, s0 // (op 953) 

s_mul_i32 s47, s22, s23 // (op 953) 


s_sub_u32 s22, s18, s47 // (op 953) 


s_add_u32 s18, s22, s0 // (op 953) 


s_cmp_eq_i32 s19, 0 // (op 953) 
s_mov_b32 s0, scc // (op 953) 

s_cmp_eq_i32 s0, 0 // (op 953) 
s_cselect_b32 s19, s18, s1 // (op 953) 



v_lshl_add_u32 v13, s19, 6, v12 // (op 953) 


v_mul_lo_u32 v19, v13, s27 // (op 953) most significant: low of low * high
v_mul_hi_u32 v12, v13, s26 // (op 953) most significant: high of low * low
v_mul_lo_u32 v18, v13, s26 // (op 953) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 953) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v14 // (op 953) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 953) most significant half



s_mul_i32 s1, s26, s7 // (op 955) most significant: low of low * high
s_mul_i32 s47, s27, s6 // (op 955) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 955) least significant
s_mul_hi_u32 s18, s26, s6 // (op 955) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 955) most significant: sum
s_add_u32 s1, s1, s18 // (op 955) most significant: sum


s_mov_b32 s48, s28 // (op 955) reg expression
s_mov_b32 s49, s29 // (op 955) reg expression
s_mov_b32 s50, s0 // (op 955) reg expression
s_mov_b32 s51, 131072 // (op 955) call()

s_mul_hi_i32 s47, s16, s10 // (op 998) 
v_lshrrev_b32 v12, 3, v1 // (op 998) 
s_add_i32 s0, s47, s16 // (op 998) 

v_and_b32 v13, -8, v1 // (op 998) 
s_ashr_i32 s1, s0, 31 // (op 998) 
v_sub_u32 v14, v1, v13 // (op 998) 

s_cmp_eq_i32 s10, 0 // (op 998) 
s_mov_b32 s47, scc // (op 998) 
v_lshlrev_b32 v13, 4, v14 // (op 998) 

s_cmp_eq_i32 s47, 0 // (op 998) 
s_cselect_b32 s18, 0, -1 // (op 998) 

v_mov_b32 v15, 0 // (op 998) convert
v_mov_b32 v14, v13 // (op 998) convert

s_and_b32 s19, s13, 31 // (op 998) 
s_lshl_b32 s47, 1, s19 // (op 998) 
s_add_i32 s22, s47, s18 // (op 998) 


s_and_b32 s18, s1, s22 // (op 998) 


s_add_i32 s1, s0, s18 // (op 998) 


s_ashr_i32 s0, s1, s19 // (op 998) 


s_lshl_b32 s1, s13, 24 // (op 998) 
s_ashr_i32 s47, s1, 31 // (op 998) 

s_xor_b32 s1, s0, s47 // (op 998) 

s_sub_i32 s0, s1, s47 // (op 998) 


s_mul_hi_i32 s1, s0, s14 // (op 998) 
s_add_i32 s47, s1, s0 // (op 998) 

s_ashr_i32 s1, s47, 31 // (op 998) 
s_cmp_eq_i32 s14, 0 // (op 998) 
s_mov_b32 s18, scc // (op 998) 
s_cmp_eq_i32 s18, 0 // (op 998) 
s_cselect_b32 s19, 0, -1 // (op 998) 

s_and_b32 s18, s15, 31 // (op 998) 
s_lshl_b32 s22, 1, s18 // (op 998) 
s_add_i32 s23, s22, s19 // (op 998) 


s_and_b32 s19, s1, s23 // (op 998) 


s_add_i32 s1, s47, s19 // (op 998) 


s_ashr_i32 s19, s1, s18 // (op 998) 


s_lshl_b32 s1, s15, 24 // (op 998) 
s_ashr_i32 s18, s1, 31 // (op 998) 

s_xor_b32 s1, s19, s18 // (op 998) 

s_sub_i32 s19, s1, s18 // (op 998) 


s_add_u32 s22, s4, 32 // (op 998) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 998) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 998) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 998) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 998) 

s_mul_i32 s1, s19, s22 // (op 998) 

s_sub_i32 s47, s0, s1 // (op 998) 


s_mul_hi_i32 s0, s16, s9 // (op 998) 
s_add_i32 s1, s0, s16 // (op 998) 

s_ashr_i32 s0, s1, 31 // (op 998) 
s_cmp_eq_i32 s9, 0 // (op 998) 
s_mov_b32 s18, scc // (op 998) 
s_cmp_eq_i32 s18, 0 // (op 998) 
s_cselect_b32 s19, 0, -1 // (op 998) 

s_and_b32 s18, s45, 31 // (op 998) 
s_lshl_b32 s24, 1, s18 // (op 998) 
s_add_i32 s25, s24, s19 // (op 998) 


s_and_b32 s19, s0, s25 // (op 998) 


s_add_i32 s0, s1, s19 // (op 998) 


s_ashr_i32 s1, s0, s18 // (op 998) 


s_lshl_b32 s0, s45, 24 // (op 998) 
s_ashr_i32 s18, s0, 31 // (op 998) 

s_xor_b32 s0, s1, s18 // (op 998) 

s_sub_i32 s1, s0, s18 // (op 998) 


s_mul_i32 s0, s11, s22 // (op 998) 

s_mul_i32 s18, s8, s0 // (op 998) 

s_mul_i32 s0, s1, s18 // (op 998) 

s_sub_u32 s18, s16, s0 // (op 998) 

s_mul_hi_i32 s0, s18, s12 // (op 998) 
s_add_i32 s19, s0, s18 // (op 998) 


s_ashr_i32 s0, s19, 31 // (op 998) 
s_cmp_eq_i32 s12, 0 // (op 998) 
s_mov_b32 s18, scc // (op 998) 
s_cmp_eq_i32 s18, 0 // (op 998) 
s_cselect_b32 s22, 0, -1 // (op 998) 

s_and_b32 s18, s44, 31 // (op 998) 
s_lshl_b32 s23, 1, s18 // (op 998) 
s_add_i32 s24, s23, s22 // (op 998) 


s_and_b32 s25, s0, s24 // (op 998) 


s_add_i32 s0, s19, s25 // (op 998) 


s_ashr_i32 s19, s0, s18 // (op 998) 


s_lshl_b32 s0, s44, 24 // (op 998) 
s_ashr_i32 s18, s0, 31 // (op 998) 

s_xor_b32 s0, s19, s18 // (op 998) 

s_sub_i32 s19, s0, s18 // (op 998) 


s_cmp_eq_i32 s1, 0 // (op 998) 
s_mov_b32 s0, scc // (op 998) 

s_cmp_eq_i32 s0, 0 // (op 998) 
s_cselect_b32 s1, s19, s47 // (op 998) 



v_lshl_add_u32 v13, s1, 5, v12 // (op 998) 


v_mul_lo_u32 v21, v13, s31 // (op 998) most significant: low of low * high
v_mul_hi_u32 v12, v13, s30 // (op 998) most significant: high of low * low
v_mul_lo_u32 v20, v13, s30 // (op 998) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 998) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v20 // (op 998) least significant half
v_addc_co_u32 v13, s[0:1], v15, v21, s[0:1] // (op 998) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 998) most significant: low of high * low
v_mul_hi_u32 v20, v12, 6 // (op 998) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 998) least significant: low of low * low
v_add_u32 v15, v15, v20 // (op 998) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 998) 

v_mov_b32 v14, v12 // (op 998) convert

s_mul_i32 s1, s5, 6 // (op 1000) most significant: low of high * low
s_mul_i32 s0, s4, 6 // (op 1000) least significant
s_mul_hi_u32 s47, s4, 6 // (op 1000) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1000) most significant: sum

s_mul_i32 s19, s30, s1 // (op 1000) most significant: low of low * high
s_mul_i32 s47, s31, s0 // (op 1000) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 1000) least significant
s_mul_hi_u32 s22, s30, s0 // (op 1000) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 1000) most significant: sum
s_add_u32 s19, s19, s22 // (op 1000) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1000) 

s_mov_b32 s60, s32 // (op 1000) reg expression
s_mov_b32 s61, s33 // (op 1000) reg expression
s_mov_b32 s62, s0 // (op 1000) reg expression
s_mov_b32 s63, 131072 // (op 1000) call()

s_mul_hi_i32 s47, s16, s10 // (op 1017) 
v_lshrrev_b32 v15, 3, v1 // (op 1017) 
s_add_i32 s0, s47, s16 // (op 1017) 

v_and_b32 v12, -8, v1 // (op 1017) 
s_ashr_i32 s1, s0, 31 // (op 1017) 
v_sub_u32 v13, v1, v12 // (op 1017) 

s_cmp_eq_i32 s10, 0 // (op 1017) 
s_mov_b32 s47, scc // (op 1017) 
v_lshlrev_b32 v12, 4, v13 // (op 1017) 

s_cmp_eq_i32 s47, 0 // (op 1017) 
s_cselect_b32 s18, 0, -1 // (op 1017) 

v_mov_b32 v21, 0 // (op 1017) convert
v_mov_b32 v20, v12 // (op 1017) convert

s_and_b32 s19, s13, 31 // (op 1017) 
s_lshl_b32 s47, 1, s19 // (op 1017) 
s_add_i32 s22, s47, s18 // (op 1017) 


s_and_b32 s18, s1, s22 // (op 1017) 


s_add_i32 s1, s0, s18 // (op 1017) 


s_ashr_i32 s0, s1, s19 // (op 1017) 


s_lshl_b32 s1, s13, 24 // (op 1017) 
s_ashr_i32 s47, s1, 31 // (op 1017) 

s_xor_b32 s1, s0, s47 // (op 1017) 

s_sub_i32 s0, s1, s47 // (op 1017) 


s_mul_hi_i32 s1, s0, s14 // (op 1017) 
s_add_i32 s47, s1, s0 // (op 1017) 

s_ashr_i32 s1, s47, 31 // (op 1017) 
s_cmp_eq_i32 s14, 0 // (op 1017) 
s_mov_b32 s18, scc // (op 1017) 
s_cmp_eq_i32 s18, 0 // (op 1017) 
s_cselect_b32 s19, 0, -1 // (op 1017) 

s_and_b32 s18, s15, 31 // (op 1017) 
s_lshl_b32 s22, 1, s18 // (op 1017) 
s_add_i32 s23, s22, s19 // (op 1017) 


s_and_b32 s19, s1, s23 // (op 1017) 


s_add_i32 s1, s47, s19 // (op 1017) 


s_ashr_i32 s19, s1, s18 // (op 1017) 


s_lshl_b32 s1, s15, 24 // (op 1017) 
s_ashr_i32 s18, s1, 31 // (op 1017) 

s_xor_b32 s1, s19, s18 // (op 1017) 

s_sub_i32 s19, s1, s18 // (op 1017) 


s_add_u32 s22, s4, 32 // (op 1017) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1017) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1017) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1017) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 1017) 

s_mul_i32 s1, s19, s22 // (op 1017) 

s_sub_i32 s47, s0, s1 // (op 1017) 


s_mul_hi_i32 s0, s16, s9 // (op 1017) 
s_add_i32 s1, s0, s16 // (op 1017) 

s_ashr_i32 s0, s1, 31 // (op 1017) 
s_cmp_eq_i32 s9, 0 // (op 1017) 
s_mov_b32 s18, scc // (op 1017) 
s_cmp_eq_i32 s18, 0 // (op 1017) 
s_cselect_b32 s19, 0, -1 // (op 1017) 

s_and_b32 s18, s45, 31 // (op 1017) 
s_lshl_b32 s24, 1, s18 // (op 1017) 
s_add_i32 s25, s24, s19 // (op 1017) 


s_and_b32 s19, s0, s25 // (op 1017) 


s_add_i32 s0, s1, s19 // (op 1017) 


s_ashr_i32 s1, s0, s18 // (op 1017) 


s_lshl_b32 s0, s45, 24 // (op 1017) 
s_ashr_i32 s18, s0, 31 // (op 1017) 

s_xor_b32 s0, s1, s18 // (op 1017) 

s_sub_i32 s1, s0, s18 // (op 1017) 


s_mul_i32 s0, s11, s22 // (op 1017) 

s_mul_i32 s18, s8, s0 // (op 1017) 

s_mul_i32 s0, s1, s18 // (op 1017) 

s_sub_u32 s18, s16, s0 // (op 1017) 

s_mul_hi_i32 s0, s18, s12 // (op 1017) 
s_add_i32 s19, s0, s18 // (op 1017) 


s_ashr_i32 s0, s19, 31 // (op 1017) 
s_cmp_eq_i32 s12, 0 // (op 1017) 
s_mov_b32 s18, scc // (op 1017) 
s_cmp_eq_i32 s18, 0 // (op 1017) 
s_cselect_b32 s22, 0, -1 // (op 1017) 

s_and_b32 s18, s44, 31 // (op 1017) 
s_lshl_b32 s23, 1, s18 // (op 1017) 
s_add_i32 s24, s23, s22 // (op 1017) 


s_and_b32 s25, s0, s24 // (op 1017) 


s_add_i32 s0, s19, s25 // (op 1017) 


s_ashr_i32 s19, s0, s18 // (op 1017) 


s_lshl_b32 s0, s44, 24 // (op 1017) 
s_ashr_i32 s18, s0, 31 // (op 1017) 

s_xor_b32 s0, s19, s18 // (op 1017) 

s_sub_i32 s19, s0, s18 // (op 1017) 


s_cmp_eq_i32 s1, 0 // (op 1017) 
s_mov_b32 s0, scc // (op 1017) 

s_cmp_eq_i32 s0, 0 // (op 1017) 
s_cselect_b32 s1, s19, s47 // (op 1017) 



v_lshl_add_u32 v12, s1, 5, v15 // (op 1017) 


v_mul_lo_u32 v23, v12, s31 // (op 1017) most significant: low of low * high
v_mul_hi_u32 v13, v12, s30 // (op 1017) most significant: high of low * low
v_mul_lo_u32 v22, v12, s30 // (op 1017) least significant: low of low * low
v_add_u32 v23, v23, v13 // (op 1017) most significant: sum


v_add_co_u32 v12, s[0:1], v20, v22 // (op 1017) least significant half
v_addc_co_u32 v13, s[0:1], v21, v23, s[0:1] // (op 1017) most significant half



v_mul_lo_u32 v21, v13, 6 // (op 1017) most significant: low of high * low
v_mul_hi_u32 v15, v12, 6 // (op 1017) most significant: high of low * low
v_mul_lo_u32 v20, v12, 6 // (op 1017) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 1017) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[20:21] // (op 1017) 

v_mov_b32 v15, v12 // (op 1017) convert

s_mul_i32 s1, s5, 6 // (op 1019) most significant: low of high * low
s_mul_i32 s0, s4, 6 // (op 1019) least significant
s_mul_hi_u32 s47, s4, 6 // (op 1019) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1019) most significant: sum

s_mul_i32 s19, s30, s1 // (op 1019) most significant: low of low * high
s_mul_i32 s47, s31, s0 // (op 1019) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 1019) least significant
s_mul_hi_u32 s22, s30, s0 // (op 1019) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 1019) most significant: sum
s_add_u32 s19, s19, s22 // (op 1019) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 1019) 

s_mov_b32 s60, s32 // (op 1019) reg expression
s_mov_b32 s61, s33 // (op 1019) reg expression
s_mov_b32 s62, s0 // (op 1019) reg expression
s_mov_b32 s63, 131072 // (op 1019) call()

s_mul_hi_i32 s47, s16, s10 // (op 1062) 
v_lshrrev_b32 v12, 2, v1 // (op 1062) 
s_add_i32 s0, s47, s16 // (op 1062) 

v_and_b32 v13, -4, v1 // (op 1062) 
s_ashr_i32 s1, s0, 31 // (op 1062) 
v_sub_u32 v20, v1, v13 // (op 1062) 

s_cmp_eq_i32 s10, 0 // (op 1062) 
s_mov_b32 s47, scc // (op 1062) 
s_cmp_eq_i32 s47, 0 // (op 1062) 
s_cselect_b32 s18, 0, -1 // (op 1062) 

s_and_b32 s19, s13, 31 // (op 1062) 
s_lshl_b32 s47, 1, s19 // (op 1062) 
s_add_i32 s22, s47, s18 // (op 1062) 


s_and_b32 s18, s1, s22 // (op 1062) 


s_add_i32 s1, s0, s18 // (op 1062) 


s_ashr_i32 s0, s1, s19 // (op 1062) 


s_lshl_b32 s1, s13, 24 // (op 1062) 
s_ashr_i32 s47, s1, 31 // (op 1062) 

s_xor_b32 s1, s0, s47 // (op 1062) 

s_sub_i32 s0, s1, s47 // (op 1062) 


s_mul_hi_i32 s1, s0, s14 // (op 1062) 
s_add_i32 s47, s1, s0 // (op 1062) 

s_ashr_i32 s1, s47, 31 // (op 1062) 
s_cmp_eq_i32 s14, 0 // (op 1062) 
s_mov_b32 s18, scc // (op 1062) 
s_cmp_eq_i32 s18, 0 // (op 1062) 
s_cselect_b32 s19, 0, -1 // (op 1062) 

s_and_b32 s18, s15, 31 // (op 1062) 
s_lshl_b32 s22, 1, s18 // (op 1062) 
s_add_i32 s23, s22, s19 // (op 1062) 


s_and_b32 s19, s1, s23 // (op 1062) 


s_add_i32 s1, s47, s19 // (op 1062) 


s_ashr_i32 s19, s1, s18 // (op 1062) 


s_lshl_b32 s1, s15, 24 // (op 1062) 
s_ashr_i32 s18, s1, 31 // (op 1062) 

s_xor_b32 s1, s19, s18 // (op 1062) 

s_sub_i32 s19, s1, s18 // (op 1062) 


s_mul_i32 s1, s0, s11 // (op 1062) 

s_sub_u32 s0, s16, s1 // (op 1062) 

s_mul_i32 s1, s19, s11 // (op 1062) 
s_add_u32 s1, s1, s0 // (op 1062) 


s_mul_hi_i32 s0, s16, s9 // (op 1062) 
s_add_i32 s47, s0, s16 // (op 1062) 

s_ashr_i32 s0, s47, 31 // (op 1062) 
s_cmp_eq_i32 s9, 0 // (op 1062) 
s_mov_b32 s18, scc // (op 1062) 
s_cmp_eq_i32 s18, 0 // (op 1062) 
s_cselect_b32 s19, 0, -1 // (op 1062) 

s_and_b32 s18, s45, 31 // (op 1062) 
s_lshl_b32 s22, 1, s18 // (op 1062) 
s_add_i32 s23, s22, s19 // (op 1062) 


s_and_b32 s19, s0, s23 // (op 1062) 


s_add_i32 s0, s47, s19 // (op 1062) 


s_ashr_i32 s19, s0, s18 // (op 1062) 


s_lshl_b32 s0, s45, 24 // (op 1062) 
s_ashr_i32 s18, s0, 31 // (op 1062) 

s_xor_b32 s0, s19, s18 // (op 1062) 

s_sub_i32 s19, s0, s18 // (op 1062) 


s_add_u32 s22, s4, 32 // (op 1062) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1062) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1062) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1062) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 1062) 

s_mul_i32 s0, s11, s22 // (op 1062) 

s_mul_i32 s18, s8, s0 // (op 1062) 

s_mul_i32 s0, s19, s18 // (op 1062) 

s_sub_u32 s18, s16, s0 // (op 1062) 

s_mov_b32 s0, s18 // (op 1062) 
s_mul_hi_i32 s47, s0, s12 // (op 1062) 
s_add_i32 s22, s47, s0 // (op 1062) 


s_ashr_i32 s0, s22, 31 // (op 1062) 
s_cmp_eq_i32 s12, 0 // (op 1062) 
s_mov_b32 s47, scc // (op 1062) 
s_cmp_eq_i32 s47, 0 // (op 1062) 
s_cselect_b32 s23, 0, -1 // (op 1062) 

s_and_b32 s47, s44, 31 // (op 1062) 
s_lshl_b32 s24, 1, s47 // (op 1062) 
s_add_i32 s25, s24, s23 // (op 1062) 


s_and_b32 s23, s0, s25 // (op 1062) 


s_add_i32 s0, s22, s23 // (op 1062) 


s_ashr_i32 s22, s0, s47 // (op 1062) 


s_lshl_b32 s0, s44, 24 // (op 1062) 
s_ashr_i32 s47, s0, 31 // (op 1062) 

s_xor_b32 s0, s22, s47 // (op 1062) 

s_sub_i32 s22, s0, s47 // (op 1062) 


s_mul_i32 s0, s46, s11 // (op 1062) 
s_add_u32 s24, s6, 64 // (op 1062) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 1062) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 1062) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 1062) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 6 // (op 1062) 

s_sub_i32 s23, s24, s0 // (op 1062) 

s_mul_i32 s47, s22, s23 // (op 1062) 


s_sub_u32 s22, s18, s47 // (op 1062) 


s_add_u32 s18, s22, s0 // (op 1062) 


s_cmp_eq_i32 s19, 0 // (op 1062) 
s_mov_b32 s0, scc // (op 1062) 

s_cmp_eq_i32 s0, 0 // (op 1062) 
s_cselect_b32 s19, s18, s1 // (op 1062) 



v_lshl_add_u32 v13, s19, 6, v12 // (op 1062) 


v_mul_lo_u32 v12, v13, s34 // (op 1062) 
v_add_u32 v12, v12, v20 // (op 1062) 


s_mul_i32 s1, s34, s7 // (op 1064) most significant: low of low * high
s_mul_i32 s47, s35, s6 // (op 1064) most significant: low of high * low
s_mul_i32 s0, s34, s6 // (op 1064) least significant
s_mul_hi_u32 s18, s34, s6 // (op 1064) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1064) most significant: sum
s_add_u32 s1, s1, s18 // (op 1064) most significant: sum


s_mov_b32 s64, s36 // (op 1064) reg expression
s_mov_b32 s65, s37 // (op 1064) reg expression
s_mov_b32 s66, s0 // (op 1064) reg expression
s_mov_b32 s67, 131072 // (op 1064) call()

s_mul_hi_i32 s47, s16, s10 // (op 1081) 
v_lshrrev_b32 v13, 2, v1 // (op 1081) 
s_add_i32 s0, s47, s16 // (op 1081) 

v_and_b32 v20, -4, v1 // (op 1081) 
s_ashr_i32 s1, s0, 31 // (op 1081) 
v_sub_u32 v21, v1, v20 // (op 1081) 

s_cmp_eq_i32 s10, 0 // (op 1081) 
s_mov_b32 s47, scc // (op 1081) 
s_cmp_eq_i32 s47, 0 // (op 1081) 
s_cselect_b32 s18, 0, -1 // (op 1081) 

s_and_b32 s19, s13, 31 // (op 1081) 
s_lshl_b32 s47, 1, s19 // (op 1081) 
s_add_i32 s22, s47, s18 // (op 1081) 


s_and_b32 s18, s1, s22 // (op 1081) 


s_add_i32 s1, s0, s18 // (op 1081) 


s_ashr_i32 s0, s1, s19 // (op 1081) 


s_lshl_b32 s1, s13, 24 // (op 1081) 
s_ashr_i32 s47, s1, 31 // (op 1081) 

s_xor_b32 s1, s0, s47 // (op 1081) 

s_sub_i32 s0, s1, s47 // (op 1081) 


s_mul_hi_i32 s1, s0, s14 // (op 1081) 
s_add_i32 s47, s1, s0 // (op 1081) 

s_ashr_i32 s1, s47, 31 // (op 1081) 
s_cmp_eq_i32 s14, 0 // (op 1081) 
s_mov_b32 s18, scc // (op 1081) 
s_cmp_eq_i32 s18, 0 // (op 1081) 
s_cselect_b32 s19, 0, -1 // (op 1081) 

s_and_b32 s18, s15, 31 // (op 1081) 
s_lshl_b32 s22, 1, s18 // (op 1081) 
s_add_i32 s23, s22, s19 // (op 1081) 


s_and_b32 s19, s1, s23 // (op 1081) 


s_add_i32 s1, s47, s19 // (op 1081) 


s_ashr_i32 s19, s1, s18 // (op 1081) 


s_lshl_b32 s1, s15, 24 // (op 1081) 
s_ashr_i32 s18, s1, 31 // (op 1081) 

s_xor_b32 s1, s19, s18 // (op 1081) 

s_sub_i32 s19, s1, s18 // (op 1081) 


s_mul_i32 s1, s0, s11 // (op 1081) 

s_sub_u32 s0, s16, s1 // (op 1081) 

s_mul_i32 s1, s19, s11 // (op 1081) 
s_add_u32 s1, s1, s0 // (op 1081) 


s_mul_hi_i32 s0, s16, s9 // (op 1081) 
s_add_i32 s47, s0, s16 // (op 1081) 

s_ashr_i32 s0, s47, 31 // (op 1081) 
s_cmp_eq_i32 s9, 0 // (op 1081) 
s_mov_b32 s18, scc // (op 1081) 
s_cmp_eq_i32 s18, 0 // (op 1081) 
s_cselect_b32 s19, 0, -1 // (op 1081) 

s_and_b32 s18, s45, 31 // (op 1081) 
s_lshl_b32 s22, 1, s18 // (op 1081) 
s_add_i32 s23, s22, s19 // (op 1081) 


s_and_b32 s19, s0, s23 // (op 1081) 


s_add_i32 s0, s47, s19 // (op 1081) 


s_ashr_i32 s19, s0, s18 // (op 1081) 


s_lshl_b32 s0, s45, 24 // (op 1081) 
s_ashr_i32 s18, s0, 31 // (op 1081) 

s_xor_b32 s0, s19, s18 // (op 1081) 

s_sub_i32 s19, s0, s18 // (op 1081) 


s_add_u32 s22, s4, 32 // (op 1081) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1081) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1081) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1081) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 1081) 

s_mul_i32 s0, s11, s22 // (op 1081) 

s_mul_i32 s18, s8, s0 // (op 1081) 

s_mul_i32 s0, s19, s18 // (op 1081) 

s_sub_u32 s18, s16, s0 // (op 1081) 

s_mov_b32 s0, s18 // (op 1081) 
s_mul_hi_i32 s47, s0, s12 // (op 1081) 
s_add_i32 s22, s47, s0 // (op 1081) 


s_ashr_i32 s0, s22, 31 // (op 1081) 
s_cmp_eq_i32 s12, 0 // (op 1081) 
s_mov_b32 s47, scc // (op 1081) 
s_cmp_eq_i32 s47, 0 // (op 1081) 
s_cselect_b32 s23, 0, -1 // (op 1081) 

s_and_b32 s47, s44, 31 // (op 1081) 
s_lshl_b32 s24, 1, s47 // (op 1081) 
s_add_i32 s25, s24, s23 // (op 1081) 


s_and_b32 s23, s0, s25 // (op 1081) 


s_add_i32 s0, s22, s23 // (op 1081) 


s_ashr_i32 s22, s0, s47 // (op 1081) 


s_lshl_b32 s0, s44, 24 // (op 1081) 
s_ashr_i32 s47, s0, 31 // (op 1081) 

s_xor_b32 s0, s22, s47 // (op 1081) 

s_sub_i32 s22, s0, s47 // (op 1081) 


s_mul_i32 s0, s46, s11 // (op 1081) 
s_add_u32 s24, s6, 64 // (op 1081) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 1081) End of Int64 add, unlocking SCC
s_sub_u32 s68, s24, 1 // (op 1081) least significant half; sets scc
s_subb_u32 s69, s25, 0 // (op 1081) most significant half; uses scc

s_ashr_i64 s[24:25], s[68:69], 6 // (op 1081) 

s_sub_i32 s23, s24, s0 // (op 1081) 

s_mul_i32 s47, s22, s23 // (op 1081) 


s_sub_u32 s22, s18, s47 // (op 1081) 


s_add_u32 s18, s22, s0 // (op 1081) 


s_cmp_eq_i32 s19, 0 // (op 1081) 
s_mov_b32 s0, scc // (op 1081) 

s_cmp_eq_i32 s0, 0 // (op 1081) 
s_cselect_b32 s19, s18, s1 // (op 1081) 



v_lshl_add_u32 v20, s19, 6, v13 // (op 1081) 


v_mul_lo_u32 v13, v20, s34 // (op 1081) 
v_add_u32 v13, v13, v21 // (op 1081) 


s_mul_i32 s1, s34, s7 // (op 1083) most significant: low of low * high
s_mul_i32 s47, s35, s6 // (op 1083) most significant: low of high * low
s_mul_i32 s0, s34, s6 // (op 1083) least significant
s_mul_hi_u32 s18, s34, s6 // (op 1083) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1083) most significant: sum
s_add_u32 s1, s1, s18 // (op 1083) most significant: sum


s_mov_b32 s64, s36 // (op 1083) reg expression
s_mov_b32 s65, s37 // (op 1083) reg expression
s_mov_b32 s66, s0 // (op 1083) reg expression
s_mov_b32 s67, 131072 // (op 1083) call()

s_mul_hi_i32 s47, s16, s10 // (op 1126) 
v_lshrrev_b32 v20, 6, v1 // (op 1126) 
s_add_i32 s0, s47, s16 // (op 1126) 

v_and_b32 v21, 1, v20 // (op 1126) 

s_ashr_i32 s1, s0, 31 // (op 1126) 
v_and_b32 v20, 15, v1 // (op 1126) 
s_cmp_eq_i32 s10, 0 // (op 1126) 
s_mov_b32 s47, scc // (op 1126) 
v_lshl_add_u32 v22, v21, 4, v20 // (op 1126) 


s_cmp_eq_i32 s47, 0 // (op 1126) 
s_cselect_b32 s18, 0, -1 // (op 1126) 

v_and_b32 v20, 63, v1 // (op 1126) 
s_and_b32 s19, s13, 31 // (op 1126) 
v_lshrrev_b32 v21, 4, v20 // (op 1126) 

s_lshl_b32 s47, 1, s19 // (op 1126) 
s_add_i32 s22, s47, s18 // (op 1126) 


s_and_b32 s18, s1, s22 // (op 1126) 


s_add_i32 s1, s0, s18 // (op 1126) 


s_ashr_i32 s0, s1, s19 // (op 1126) 


s_lshl_b32 s1, s13, 24 // (op 1126) 
s_ashr_i32 s47, s1, 31 // (op 1126) 

s_xor_b32 s1, s0, s47 // (op 1126) 

s_sub_i32 s0, s1, s47 // (op 1126) 


s_mul_hi_i32 s1, s0, s14 // (op 1126) 
s_add_i32 s47, s1, s0 // (op 1126) 

s_ashr_i32 s1, s47, 31 // (op 1126) 
s_cmp_eq_i32 s14, 0 // (op 1126) 
s_mov_b32 s18, scc // (op 1126) 
s_cmp_eq_i32 s18, 0 // (op 1126) 
s_cselect_b32 s19, 0, -1 // (op 1126) 

s_and_b32 s18, s15, 31 // (op 1126) 
s_lshl_b32 s22, 1, s18 // (op 1126) 
s_add_i32 s23, s22, s19 // (op 1126) 


s_and_b32 s19, s1, s23 // (op 1126) 


s_add_i32 s1, s47, s19 // (op 1126) 


s_ashr_i32 s19, s1, s18 // (op 1126) 


s_lshl_b32 s1, s15, 24 // (op 1126) 
s_ashr_i32 s18, s1, 31 // (op 1126) 

s_xor_b32 s1, s19, s18 // (op 1126) 

s_sub_i32 s19, s1, s18 // (op 1126) 


s_add_u32 s22, s4, 32 // (op 1126) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1126) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1126) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1126) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 1126) 

s_mul_i32 s1, s19, s22 // (op 1126) 

s_sub_i32 s47, s0, s1 // (op 1126) 


s_mul_hi_i32 s0, s16, s9 // (op 1126) 
s_add_i32 s1, s0, s16 // (op 1126) 

s_ashr_i32 s0, s1, 31 // (op 1126) 
s_cmp_eq_i32 s9, 0 // (op 1126) 
s_mov_b32 s18, scc // (op 1126) 
s_cmp_eq_i32 s18, 0 // (op 1126) 
s_cselect_b32 s19, 0, -1 // (op 1126) 

s_and_b32 s18, s45, 31 // (op 1126) 
s_lshl_b32 s24, 1, s18 // (op 1126) 
s_add_i32 s25, s24, s19 // (op 1126) 


s_and_b32 s19, s0, s25 // (op 1126) 


s_add_i32 s0, s1, s19 // (op 1126) 


s_ashr_i32 s1, s0, s18 // (op 1126) 


s_lshl_b32 s0, s45, 24 // (op 1126) 
s_ashr_i32 s18, s0, 31 // (op 1126) 

s_xor_b32 s0, s1, s18 // (op 1126) 

s_sub_i32 s1, s0, s18 // (op 1126) 


s_mul_i32 s0, s11, s22 // (op 1126) 

s_mul_i32 s18, s8, s0 // (op 1126) 

s_mul_i32 s0, s1, s18 // (op 1126) 

s_sub_u32 s18, s16, s0 // (op 1126) 

s_mul_hi_i32 s0, s18, s12 // (op 1126) 
s_add_i32 s19, s0, s18 // (op 1126) 


s_ashr_i32 s0, s19, 31 // (op 1126) 
s_cmp_eq_i32 s12, 0 // (op 1126) 
s_mov_b32 s18, scc // (op 1126) 
s_cmp_eq_i32 s18, 0 // (op 1126) 
s_cselect_b32 s22, 0, -1 // (op 1126) 

s_and_b32 s18, s44, 31 // (op 1126) 
s_lshl_b32 s23, 1, s18 // (op 1126) 
s_add_i32 s24, s23, s22 // (op 1126) 


s_and_b32 s25, s0, s24 // (op 1126) 


s_add_i32 s0, s19, s25 // (op 1126) 


s_ashr_i32 s19, s0, s18 // (op 1126) 


s_lshl_b32 s0, s44, 24 // (op 1126) 
s_ashr_i32 s18, s0, 31 // (op 1126) 

s_xor_b32 s0, s19, s18 // (op 1126) 

s_sub_i32 s19, s0, s18 // (op 1126) 


s_cmp_eq_i32 s1, 0 // (op 1126) 
s_mov_b32 s0, scc // (op 1126) 

s_cmp_eq_i32 s0, 0 // (op 1126) 
s_cselect_b32 s1, s19, s47 // (op 1126) 



v_lshl_add_u32 v20, s1, 5, v22 // (op 1126) 


v_mul_lo_u32 v22, v20, s38 // (op 1126) 

v_add_u32 v20, v21, v22 // (op 1126) 


s_mul_i32 s1, s38, s5 // (op 1128) most significant: low of low * high
s_mul_i32 s47, s39, s4 // (op 1128) most significant: low of high * low
s_mul_i32 s0, s38, s4 // (op 1128) least significant
s_mul_hi_u32 s18, s38, s4 // (op 1128) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1128) most significant: sum
s_add_u32 s1, s1, s18 // (op 1128) most significant: sum


s_mov_b32 s68, s40 // (op 1128) reg expression
s_mov_b32 s69, s41 // (op 1128) reg expression
s_mov_b32 s70, s0 // (op 1128) reg expression
s_mov_b32 s71, 131072 // (op 1128) call()

s_mul_hi_i32 s47, s16, s10 // (op 1145) 
v_lshrrev_b32 v21, 6, v1 // (op 1145) 
s_add_i32 s0, s47, s16 // (op 1145) 

v_and_b32 v22, 1, v21 // (op 1145) 

s_ashr_i32 s1, s0, 31 // (op 1145) 
v_and_b32 v21, 15, v1 // (op 1145) 
s_cmp_eq_i32 s10, 0 // (op 1145) 
s_mov_b32 s47, scc // (op 1145) 
v_lshl_add_u32 v23, v22, 4, v21 // (op 1145) 


s_cmp_eq_i32 s47, 0 // (op 1145) 
s_cselect_b32 s18, 0, -1 // (op 1145) 

v_and_b32 v21, 63, v1 // (op 1145) 
s_and_b32 s19, s13, 31 // (op 1145) 
v_lshrrev_b32 v22, 4, v21 // (op 1145) 

s_lshl_b32 s47, 1, s19 // (op 1145) 
s_add_i32 s22, s47, s18 // (op 1145) 


s_and_b32 s18, s1, s22 // (op 1145) 


s_add_i32 s1, s0, s18 // (op 1145) 


s_ashr_i32 s0, s1, s19 // (op 1145) 


s_lshl_b32 s1, s13, 24 // (op 1145) 
s_ashr_i32 s47, s1, 31 // (op 1145) 

s_xor_b32 s1, s0, s47 // (op 1145) 

s_sub_i32 s0, s1, s47 // (op 1145) 


s_mul_hi_i32 s1, s0, s14 // (op 1145) 
s_add_i32 s47, s1, s0 // (op 1145) 

s_ashr_i32 s1, s47, 31 // (op 1145) 
s_cmp_eq_i32 s14, 0 // (op 1145) 
s_mov_b32 s18, scc // (op 1145) 
s_cmp_eq_i32 s18, 0 // (op 1145) 
s_cselect_b32 s19, 0, -1 // (op 1145) 

s_and_b32 s18, s15, 31 // (op 1145) 
s_lshl_b32 s22, 1, s18 // (op 1145) 
s_add_i32 s23, s22, s19 // (op 1145) 


s_and_b32 s19, s1, s23 // (op 1145) 


s_add_i32 s1, s47, s19 // (op 1145) 


s_ashr_i32 s19, s1, s18 // (op 1145) 


s_lshl_b32 s1, s15, 24 // (op 1145) 
s_ashr_i32 s18, s1, 31 // (op 1145) 

s_xor_b32 s1, s19, s18 // (op 1145) 

s_sub_i32 s19, s1, s18 // (op 1145) 


s_add_u32 s22, s4, 32 // (op 1145) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1145) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1145) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1145) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 1145) 

s_mul_i32 s1, s19, s22 // (op 1145) 

s_sub_i32 s47, s0, s1 // (op 1145) 


s_mul_hi_i32 s0, s16, s9 // (op 1145) 
s_add_i32 s1, s0, s16 // (op 1145) 

s_ashr_i32 s0, s1, 31 // (op 1145) 
s_cmp_eq_i32 s9, 0 // (op 1145) 
s_mov_b32 s18, scc // (op 1145) 
s_cmp_eq_i32 s18, 0 // (op 1145) 
s_cselect_b32 s19, 0, -1 // (op 1145) 

s_and_b32 s18, s45, 31 // (op 1145) 
s_lshl_b32 s24, 1, s18 // (op 1145) 
s_add_i32 s25, s24, s19 // (op 1145) 


s_and_b32 s19, s0, s25 // (op 1145) 


s_add_i32 s0, s1, s19 // (op 1145) 


s_ashr_i32 s1, s0, s18 // (op 1145) 


s_lshl_b32 s0, s45, 24 // (op 1145) 
s_ashr_i32 s18, s0, 31 // (op 1145) 

s_xor_b32 s0, s1, s18 // (op 1145) 

s_sub_i32 s1, s0, s18 // (op 1145) 


s_mul_i32 s0, s11, s22 // (op 1145) 

s_mul_i32 s18, s8, s0 // (op 1145) 

s_mul_i32 s0, s1, s18 // (op 1145) 

s_sub_u32 s18, s16, s0 // (op 1145) 

s_mul_hi_i32 s0, s18, s12 // (op 1145) 
s_add_i32 s19, s0, s18 // (op 1145) 


s_ashr_i32 s0, s19, 31 // (op 1145) 
s_cmp_eq_i32 s12, 0 // (op 1145) 
s_mov_b32 s18, scc // (op 1145) 
s_cmp_eq_i32 s18, 0 // (op 1145) 
s_cselect_b32 s22, 0, -1 // (op 1145) 

s_and_b32 s18, s44, 31 // (op 1145) 
s_lshl_b32 s23, 1, s18 // (op 1145) 
s_add_i32 s24, s23, s22 // (op 1145) 


s_and_b32 s25, s0, s24 // (op 1145) 


s_add_i32 s0, s19, s25 // (op 1145) 


s_ashr_i32 s19, s0, s18 // (op 1145) 


s_lshl_b32 s0, s44, 24 // (op 1145) 
s_ashr_i32 s18, s0, 31 // (op 1145) 

s_xor_b32 s0, s19, s18 // (op 1145) 

s_sub_i32 s19, s0, s18 // (op 1145) 


s_cmp_eq_i32 s1, 0 // (op 1145) 
s_mov_b32 s0, scc // (op 1145) 

s_cmp_eq_i32 s0, 0 // (op 1145) 
s_cselect_b32 s1, s19, s47 // (op 1145) 



v_lshl_add_u32 v21, s1, 5, v23 // (op 1145) 


v_mul_lo_u32 v23, v21, s38 // (op 1145) 

v_add_u32 v21, v22, v23 // (op 1145) 


s_mul_i32 s1, s38, s5 // (op 1147) most significant: low of low * high
s_mul_i32 s47, s39, s4 // (op 1147) most significant: low of high * low
s_mul_i32 s0, s38, s4 // (op 1147) least significant
s_mul_hi_u32 s18, s38, s4 // (op 1147) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1147) most significant: sum
s_add_u32 s1, s1, s18 // (op 1147) most significant: sum


s_mov_b32 s68, s40 // (op 1147) reg expression
s_mov_b32 s69, s41 // (op 1147) reg expression
s_mov_b32 s70, s0 // (op 1147) reg expression
s_mov_b32 s71, 131072 // (op 1147) call()




buffer_load_ubyte v22, v20, s[68:71], 0 offen offset: 0 // (op 782) Load value



buffer_load_dwordx3 v[24:26], v14, s[60:63], 0 offen // (op 786) Load value
v_and_b32 v23, -8, v1 // (op 1211) 
v_sub_u32 v27, v1, v23 // (op 1211) 

v_lshlrev_b32 v23, 4, v27 // (op 1211) 

v_lshrrev_b32 v27, 3, v1 // (op 1211) 
v_lshl_add_u32 v28, v27, 7, v23 // (op 1211) 


s_nop 0
v_readfirstlane_b32 s47, v28 // (op 1211) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 5 // (op 889) 
v_mov_b64 v[28:29], s[0:1] // (op 889) convert


v_mov_b64 v[30:31], v[16:17] // (op 889) 
s_mov_b32 m0, s47 // (op 889) reg expression
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 889) Load value direct to lds
v_add_co_u32 v30, s[0:1], v30, v28 // (op 889) least significant half
v_addc_co_u32 v31, s[0:1], 0, v31, s[0:1] // (op 889) most significant half

s_add_u32 m0, m0, 4096 // (op 889) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 889) Load value direct to lds






buffer_load_ubyte v23, v12, s[64:67], 0 offen offset: 0 // (op 792) Load value



buffer_load_ubyte v27, v20, s[68:71], 0 offen offset:4 // (op 796) Load value




buffer_load_dwordx3 v[28:30], v14, s[60:63], 0 offen offset:96 // (op 800) Load value

v_and_b32 v14, -8, v1 // (op 1384) 
v_sub_u32 v20, v1, v14 // (op 1384) 

v_lshlrev_b32 v14, 4, v20 // (op 1384) 

v_lshrrev_b32 v20, 3, v1 // (op 1384) 
v_lshl_add_u32 v31, v20, 7, v14 // (op 1384) 


s_nop 0
v_readfirstlane_b32 s47, v31 // (op 1384) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 5 // (op 894) 
v_mov_b64 v[32:33], s[0:1] // (op 894) convert


v_mov_b64 v[34:35], v[16:17] // (op 894) 
s_add_u32 s0, 8192, s47 // (op 894) 
s_mov_b32 m0, s0 // (op 894) reg expression
s_mov_b32 s1, 128 // (op 894) call()
buffer_load_dwordx4 v34, s[48:51], s1 offen offset: 0 lds // (op 894) Load value direct to lds
v_add_co_u32 v34, s[18:19], v34, v32 // (op 894) least significant half
v_addc_co_u32 v35, s[18:19], 0, v35, s[18:19] // (op 894) most significant half

s_add_u32 m0, m0, 4096 // (op 894) 
buffer_load_dwordx4 v34, s[48:51], s1 offen offset: 0 lds // (op 894) Load value direct to lds









buffer_load_ubyte v14, v12, s[64:67], 0 offen offset:4 // (op 806) Load value

v_and_b32 v12, -8, v1 // (op 1269) 
v_sub_u32 v20, v1, v12 // (op 1269) 

v_lshlrev_b32 v12, 4, v20 // (op 1269) 

v_lshrrev_b32 v20, 3, v1 // (op 1269) 
v_lshl_add_u32 v16, v20, 7, v12 // (op 1269) 


v_mul_lo_u32 v12, v16, 6 // (op 1269) 

v_lshrrev_b32 v20, 3, v12 // (op 1269) 




s_waitcnt vmcnt(8)
ds_write_b96 v20, v[24:26] offset:16384 // (op 808) Store local data 

v_and_b32 v12, -4, v1 // (op 1322) 
v_sub_u32 v20, v1, v12 // (op 1322) 

v_and_b32 v12, 4294967292, v1 // (op 1322) 
v_add_u32 v16, v12, v20 // (op 1322) 





s_waitcnt vmcnt(5)
ds_write_b8 v16, v23 offset:19456 // (op 810) Store local data 

v_and_b32 v12, -8, v1 // (op 1183) 
v_sub_u32 v20, v1, v12 // (op 1183) 

v_lshlrev_b32 v12, 4, v20 // (op 1183) 

v_lshrrev_b32 v20, 3, v1 // (op 1183) 
v_lshl_add_u32 v16, v20, 7, v12 // (op 1183) 


s_nop 0
v_readfirstlane_b32 s47, v16 // (op 1183) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -8, v1 // (op 1240) 
v_sub_u32 v20, v1, v12 // (op 1240) 

v_lshlrev_b32 v12, 4, v20 // (op 1240) 

v_lshrrev_b32 v20, 3, v1 // (op 1240) 
v_lshl_add_u32 v16, v20, 7, v12 // (op 1240) 


v_mul_lo_u32 v12, v16, 6 // (op 1240) 

v_lshrrev_b32 v20, 3, v12 // (op 1240) 

v_and_b32 v12, -4, v1 // (op 1293) 
v_sub_u32 v16, v1, v12 // (op 1293) 

v_and_b32 v12, 4294967292, v1 // (op 1293) 
v_add_u32 v17, v12, v16 // (op 1293) 


v_and_b32 v12, -8, v1 // (op 1369) 
v_sub_u32 v16, v1, v12 // (op 1369) 

v_lshlrev_b32 v12, 4, v16 // (op 1369) 

v_lshrrev_b32 v16, 3, v1 // (op 1369) 
v_lshl_add_u32 v31, v16, 7, v12 // (op 1369) 


s_nop 0
v_readfirstlane_b32 s0, v31 // (op 1369) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v12, -8, v1 // (op 1413) 
v_sub_u32 v16, v1, v12 // (op 1413) 

v_lshlrev_b32 v12, 4, v16 // (op 1413) 

v_lshrrev_b32 v16, 3, v1 // (op 1413) 
v_lshl_add_u32 v31, v16, 7, v12 // (op 1413) 


v_mul_lo_u32 v12, v31, 6 // (op 1413) 

v_lshrrev_b32 v16, 3, v12 // (op 1413) 

v_and_b32 v12, -4, v1 // (op 1437) 
v_sub_u32 v31, v1, v12 // (op 1437) 

v_and_b32 v12, 4294967292, v1 // (op 1437) 
v_add_u32 v32, v12, v31 // (op 1437) 


s_mov_b32 s1, 0 // (op 315) call()
s_ashr_i64 s[18:19], s[42:43], 7 // (op 314) 
s_mov_b32 s22, s18 // (op 314) convert

s_and_b32 s18, s22, 4294967294 // (op 314) 

s_cmp_lt_u32 s1, s18 // (op 314) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoop_314 // (op 314) Condition: Top (jump to RR_GEMM_TN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoop_314 if false)
RR_GEMM_TN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoop_314:
s_barrier  // (op 1495) 
s_add_u32 s1, s1, 2 // (op 316) 



ds_read_b128 v[34:37], v7 // (op 67) Load local data 
ds_read_b128 v[38:41], v7 offset:64 // (op 67) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1508) 



ds_read_b128 v[42:45], v3 offset:16384 // (op 72) Load local data 
ds_read_b64 v[46:47], v3 offset:16400 // (op 72) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1522) 



ds_read_u8 v12, v0 offset:19456 // (op 77) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:41], v[42:47], a[0:3], v12, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 84) 
ds_read_b128 v[48:51], v7 offset:4096 // (op 344) Load local data 


ds_read_b128 v[52:55], v7 offset:4160 // (op 344) Load local data 



ds_read_u8 v12, v0 offset:19584 // (op 348) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[48:55], v[42:47], a[4:7], v12, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 350) 





v_mov_b32 v12, v21 // (op 82) 

buffer_load_ubyte v22, v12, s[68:71], 0 offen offset:8 // (op 82) Load value


v_mov_b32 v31, v15 // (op 10) 

buffer_load_dwordx3 v[24:26], v31, s[60:63], 0 offen offset:192 // (op 10) Load value
s_barrier  // (op 1498) 

s_lshl_b64 s[18:19], s[26:27], 5 // (op 873) 
v_mov_b64 v[34:35], s[18:19] // (op 873) convert


v_mov_b64 v[36:37], v[18:19] // (op 873) 
v_mov_b64 v[38:39], v[36:37] // (op 873) 
s_mov_b32 m0, s47 // (op 873) reg expression
s_mov_b32 s18, 256 // (op 873) call()
buffer_load_dwordx4 v38, s[48:51], s18 offen offset: 0 lds // (op 873) Load value direct to lds
v_add_co_u32 v38, s[22:23], v38, v34 // (op 873) least significant half
v_addc_co_u32 v39, s[22:23], 0, v39, s[22:23] // (op 873) most significant half

s_add_u32 m0, m0, 4096 // (op 873) 
buffer_load_dwordx4 v38, s[48:51], s18 offen offset: 0 lds // (op 873) Load value direct to lds





v_mov_b32 v33, v13 // (op 17) 

buffer_load_ubyte v23, v33, s[64:67], 0 offen offset:8 // (op 17) Load value
s_barrier  // (op 1545) 



s_waitcnt vmcnt(8)
ds_write_b96 v16, v[28:30] offset:19712 // (op 239) Store local data 
s_barrier  // (op 1551) 



s_waitcnt vmcnt(5)
ds_write_b8 v32, v14 offset:22784 // (op 240) Store local data 
s_barrier  // (op 1536) 



ds_read_b128 v[38:41], v11 offset:8192 // (op 231) Load local data 
ds_read_b128 v[42:45], v11 offset:8256 // (op 231) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1542) 



ds_read_b128 v[46:49], v4 offset:19712 // (op 233) Load local data 
ds_read_b64 v[50:51], v4 offset:19728 // (op 233) Load local data 
s_barrier  // (op 1584) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 1548) 



ds_read_u8 v34, v10 offset:22784 // (op 235) Load local data 



ds_read_b128 v[52:55], v11 offset:12288 // (op 357) Load local data 
ds_read_b128 v[56:59], v11 offset:12352 // (op 357) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:45], v[46:51], a[0:3], v34, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 238) 





ds_read_u8 v34, v10 offset:22912 // (op 361) Load local data 
s_barrier  // (op 1622) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[52:59], v[46:51], a[4:7], v34, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 351) 






buffer_load_ubyte v27, v12, s[68:71], 0 offen offset:12 // (op 237) Load value




v_add_u32 v21, 8, v21 // (op 1137) 
buffer_load_dwordx3 v[28:30], v31, s[60:63], 0 offen offset:288 // (op 241) Load value

s_barrier  // (op 1539) 
v_add_u32 v15, 192, v15 // (op 1009) 

s_lshl_b64 s[18:19], s[26:27], 5 // (op 884) 
v_mov_b64 v[34:35], s[18:19] // (op 884) convert


v_mov_b64 v[38:39], v[36:37] // (op 884) 
s_add_u32 s18, 8192, s0 // (op 884) 
s_mov_b32 m0, s18 // (op 884) reg expression
s_mov_b32 s19, 384 // (op 884) call()
buffer_load_dwordx4 v38, s[48:51], s19 offen offset: 0 lds // (op 884) Load value direct to lds
v_add_co_u32 v38, s[22:23], v38, v34 // (op 884) least significant half
v_addc_co_u32 v39, s[22:23], 0, v39, s[22:23] // (op 884) most significant half

s_add_u32 m0, m0, 4096 // (op 884) 
buffer_load_dwordx4 v38, s[48:51], s19 offen offset: 0 lds // (op 884) Load value direct to lds








v_add_co_u32 v18, vcc, 256, v18 // (op 945) least significant half
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 945) most significant half
buffer_load_ubyte v14, v33, s[64:67], 0 offen offset:12 // (op 242) Load value

s_barrier  // (op 1511) 
v_add_u32 v13, 8, v13 // (op 1073) 



s_waitcnt vmcnt(8)
ds_write_b96 v20, v[24:26] offset:16384 // (op 13) Store local data 
s_barrier  // (op 1525) 



s_waitcnt vmcnt(5)
ds_write_b8 v17, v23 offset:19456 // (op 21) Store local data 
s_ashr_i64 s[18:19], s[42:43], 7 // (op 314) 
s_mov_b32 s22, s18 // (op 314) convert

s_and_b32 s18, s22, 4294967294 // (op 314) 

s_cmp_lt_u32 s1, s18 // (op 314) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoop_314 // (op 314) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoop_314 if true)
RR_GEMM_TN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoop_314:














s_barrier  // (op 1604) 
s_mul_hi_i32 s47, s16, s10 // (op 907) 
v_lshrrev_b32 v12, 3, v1 // (op 907) 
s_add_i32 s0, s47, s16 // (op 907) 

v_and_b32 v13, -8, v1 // (op 907) 
s_ashr_i32 s1, s0, 31 // (op 907) 
v_sub_u32 v15, v1, v13 // (op 907) 

s_cmp_eq_i32 s10, 0 // (op 907) 
s_mov_b32 s47, scc // (op 907) 
v_lshlrev_b32 v13, 4, v15 // (op 907) 

s_cmp_eq_i32 s47, 0 // (op 907) 
s_cselect_b32 s18, 0, -1 // (op 907) 

s_and_b32 s19, s13, 31 // (op 907) 
s_lshl_b32 s47, 1, s19 // (op 907) 
s_add_i32 s22, s47, s18 // (op 907) 


s_and_b32 s18, s1, s22 // (op 907) 


s_add_i32 s1, s0, s18 // (op 907) 


s_ashr_i32 s0, s1, s19 // (op 907) 


s_lshl_b32 s1, s13, 24 // (op 907) 
s_ashr_i32 s47, s1, 31 // (op 907) 

s_xor_b32 s1, s0, s47 // (op 907) 

s_sub_i32 s0, s1, s47 // (op 907) 


s_mul_hi_i32 s1, s0, s14 // (op 907) 
s_add_i32 s47, s1, s0 // (op 907) 

s_ashr_i32 s1, s47, 31 // (op 907) 
s_cmp_eq_i32 s14, 0 // (op 907) 
s_mov_b32 s18, scc // (op 907) 
s_cmp_eq_i32 s18, 0 // (op 907) 
s_cselect_b32 s19, 0, -1 // (op 907) 

s_and_b32 s18, s15, 31 // (op 907) 
s_lshl_b32 s22, 1, s18 // (op 907) 
s_add_i32 s23, s22, s19 // (op 907) 


s_and_b32 s19, s1, s23 // (op 907) 


s_add_i32 s1, s47, s19 // (op 907) 


s_ashr_i32 s19, s1, s18 // (op 907) 


s_lshl_b32 s1, s15, 24 // (op 907) 
s_ashr_i32 s18, s1, 31 // (op 907) 

s_xor_b32 s1, s19, s18 // (op 907) 

s_sub_i32 s19, s1, s18 // (op 907) 


s_mul_i32 s1, s0, s11 // (op 907) 

s_sub_u32 s0, s16, s1 // (op 907) 

s_mul_i32 s1, s19, s11 // (op 907) 
s_add_u32 s1, s1, s0 // (op 907) 


s_mul_hi_i32 s0, s16, s9 // (op 907) 
s_add_i32 s47, s0, s16 // (op 907) 

s_ashr_i32 s0, s47, 31 // (op 907) 
s_cmp_eq_i32 s9, 0 // (op 907) 
s_mov_b32 s18, scc // (op 907) 
s_cmp_eq_i32 s18, 0 // (op 907) 
s_cselect_b32 s19, 0, -1 // (op 907) 

s_and_b32 s18, s45, 31 // (op 907) 
s_lshl_b32 s22, 1, s18 // (op 907) 
s_add_i32 s23, s22, s19 // (op 907) 


s_and_b32 s19, s0, s23 // (op 907) 


s_add_i32 s0, s47, s19 // (op 907) 


s_ashr_i32 s19, s0, s18 // (op 907) 


s_lshl_b32 s0, s45, 24 // (op 907) 
s_ashr_i32 s18, s0, 31 // (op 907) 

s_xor_b32 s0, s19, s18 // (op 907) 

s_sub_i32 s19, s0, s18 // (op 907) 


s_add_u32 s22, s4, 32 // (op 907) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 907) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 907) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 907) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 907) 

s_mul_i32 s0, s11, s22 // (op 907) 

s_mul_i32 s18, s8, s0 // (op 907) 

s_mul_i32 s0, s19, s18 // (op 907) 

s_sub_u32 s18, s16, s0 // (op 907) 

s_mov_b32 s0, s18 // (op 907) 
s_mul_hi_i32 s47, s0, s12 // (op 907) 
s_add_i32 s22, s47, s0 // (op 907) 


s_ashr_i32 s0, s22, 31 // (op 907) 
s_cmp_eq_i32 s12, 0 // (op 907) 
s_mov_b32 s47, scc // (op 907) 
s_cmp_eq_i32 s47, 0 // (op 907) 
s_cselect_b32 s23, 0, -1 // (op 907) 

s_and_b32 s47, s44, 31 // (op 907) 
s_lshl_b32 s24, 1, s47 // (op 907) 
s_add_i32 s25, s24, s23 // (op 907) 


s_and_b32 s23, s0, s25 // (op 907) 


s_add_i32 s0, s22, s23 // (op 907) 


s_ashr_i32 s22, s0, s47 // (op 907) 


s_lshl_b32 s0, s44, 24 // (op 907) 
s_ashr_i32 s47, s0, 31 // (op 907) 

s_xor_b32 s0, s22, s47 // (op 907) 

s_sub_i32 s22, s0, s47 // (op 907) 


s_mul_i32 s0, s46, s11 // (op 907) 
s_add_u32 s24, s6, 64 // (op 907) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 907) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 907) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 907) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 6 // (op 907) 

s_sub_i32 s23, s24, s0 // (op 907) 

s_mul_i32 s47, s22, s23 // (op 907) 


s_sub_u32 s22, s18, s47 // (op 907) 


s_add_u32 s18, s22, s0 // (op 907) 


s_cmp_eq_i32 s19, 0 // (op 907) 
s_mov_b32 s0, scc // (op 907) 

s_cmp_eq_i32 s0, 0 // (op 907) 
s_cselect_b32 s19, s18, s1 // (op 907) 



v_lshl_add_u32 v15, s19, 6, v12 // (op 907) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 907) 
s_lshr_b32 s47, s0, 1 // (op 907) 

v_lshl_add_u32 v12, s47, 8, v13 // (op 907) 


v_mov_b32 v17, 0 // (op 907) convert
v_mov_b32 v16, v12 // (op 907) convert

v_mul_lo_u32 v13, v15, s27 // (op 907) most significant: low of low * high
v_mul_hi_u32 v18, v15, s26 // (op 907) most significant: high of low * low
v_mul_lo_u32 v12, v15, s26 // (op 907) least significant: low of low * low
v_add_u32 v13, v13, v18 // (op 907) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v16 // (op 907) least significant half
v_addc_co_u32 v13, s[0:1], v13, v17, s[0:1] // (op 907) most significant half



s_mul_i32 s1, s26, s7 // (op 909) most significant: low of low * high
s_mul_i32 s47, s27, s6 // (op 909) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 909) least significant
s_mul_hi_u32 s18, s26, s6 // (op 909) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 909) most significant: sum
s_add_u32 s1, s1, s18 // (op 909) most significant: sum


s_mov_b32 s48, s28 // (op 909) reg expression
s_mov_b32 s49, s29 // (op 909) reg expression
s_mov_b32 s50, s0 // (op 909) reg expression
s_mov_b32 s51, 131072 // (op 909) call()

s_mul_hi_i32 s47, s16, s10 // (op 975) 
v_lshrrev_b32 v15, 3, v1 // (op 975) 
s_add_i32 s0, s47, s16 // (op 975) 

v_and_b32 v16, -8, v1 // (op 975) 
s_ashr_i32 s1, s0, 31 // (op 975) 
v_sub_u32 v17, v1, v16 // (op 975) 

s_cmp_eq_i32 s10, 0 // (op 975) 
s_mov_b32 s47, scc // (op 975) 
v_lshlrev_b32 v16, 4, v17 // (op 975) 

s_cmp_eq_i32 s47, 0 // (op 975) 
s_cselect_b32 s18, 0, -1 // (op 975) 

s_and_b32 s19, s13, 31 // (op 975) 
s_lshl_b32 s47, 1, s19 // (op 975) 
s_add_i32 s22, s47, s18 // (op 975) 


s_and_b32 s18, s1, s22 // (op 975) 


s_add_i32 s1, s0, s18 // (op 975) 


s_ashr_i32 s0, s1, s19 // (op 975) 


s_lshl_b32 s1, s13, 24 // (op 975) 
s_ashr_i32 s47, s1, 31 // (op 975) 

s_xor_b32 s1, s0, s47 // (op 975) 

s_sub_i32 s0, s1, s47 // (op 975) 


s_mul_hi_i32 s1, s0, s14 // (op 975) 
s_add_i32 s47, s1, s0 // (op 975) 

s_ashr_i32 s1, s47, 31 // (op 975) 
s_cmp_eq_i32 s14, 0 // (op 975) 
s_mov_b32 s18, scc // (op 975) 
s_cmp_eq_i32 s18, 0 // (op 975) 
s_cselect_b32 s19, 0, -1 // (op 975) 

s_and_b32 s18, s15, 31 // (op 975) 
s_lshl_b32 s22, 1, s18 // (op 975) 
s_add_i32 s23, s22, s19 // (op 975) 


s_and_b32 s19, s1, s23 // (op 975) 


s_add_i32 s1, s47, s19 // (op 975) 


s_ashr_i32 s19, s1, s18 // (op 975) 


s_lshl_b32 s1, s15, 24 // (op 975) 
s_ashr_i32 s18, s1, 31 // (op 975) 

s_xor_b32 s1, s19, s18 // (op 975) 

s_sub_i32 s19, s1, s18 // (op 975) 


s_add_u32 s22, s4, 32 // (op 975) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 975) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 975) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 975) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 975) 

s_mul_i32 s1, s19, s22 // (op 975) 

s_sub_i32 s47, s0, s1 // (op 975) 


s_mul_hi_i32 s0, s16, s9 // (op 975) 
s_add_i32 s1, s0, s16 // (op 975) 

s_ashr_i32 s0, s1, 31 // (op 975) 
s_cmp_eq_i32 s9, 0 // (op 975) 
s_mov_b32 s18, scc // (op 975) 
s_cmp_eq_i32 s18, 0 // (op 975) 
s_cselect_b32 s19, 0, -1 // (op 975) 

s_and_b32 s18, s45, 31 // (op 975) 
s_lshl_b32 s24, 1, s18 // (op 975) 
s_add_i32 s25, s24, s19 // (op 975) 


s_and_b32 s19, s0, s25 // (op 975) 


s_add_i32 s0, s1, s19 // (op 975) 


s_ashr_i32 s1, s0, s18 // (op 975) 


s_lshl_b32 s0, s45, 24 // (op 975) 
s_ashr_i32 s18, s0, 31 // (op 975) 

s_xor_b32 s0, s1, s18 // (op 975) 

s_sub_i32 s1, s0, s18 // (op 975) 


s_mul_i32 s0, s11, s22 // (op 975) 

s_mul_i32 s18, s8, s0 // (op 975) 

s_mul_i32 s0, s1, s18 // (op 975) 

s_sub_u32 s18, s16, s0 // (op 975) 

s_mul_hi_i32 s0, s18, s12 // (op 975) 
s_add_i32 s19, s0, s18 // (op 975) 


s_ashr_i32 s0, s19, 31 // (op 975) 
s_cmp_eq_i32 s12, 0 // (op 975) 
s_mov_b32 s18, scc // (op 975) 
s_cmp_eq_i32 s18, 0 // (op 975) 
s_cselect_b32 s22, 0, -1 // (op 975) 

s_and_b32 s18, s44, 31 // (op 975) 
s_lshl_b32 s23, 1, s18 // (op 975) 
s_add_i32 s24, s23, s22 // (op 975) 


s_and_b32 s25, s0, s24 // (op 975) 


s_add_i32 s0, s19, s25 // (op 975) 


s_ashr_i32 s19, s0, s18 // (op 975) 


s_lshl_b32 s0, s44, 24 // (op 975) 
s_ashr_i32 s18, s0, 31 // (op 975) 

s_xor_b32 s0, s19, s18 // (op 975) 

s_sub_i32 s19, s0, s18 // (op 975) 


s_cmp_eq_i32 s1, 0 // (op 975) 
s_mov_b32 s0, scc // (op 975) 

s_cmp_eq_i32 s0, 0 // (op 975) 
s_cselect_b32 s1, s19, s47 // (op 975) 



v_lshl_add_u32 v17, s1, 5, v15 // (op 975) 


v_mul_lo_u32 v19, v17, s31 // (op 975) most significant: low of low * high
v_mul_hi_u32 v15, v17, s30 // (op 975) most significant: high of low * low
v_mul_lo_u32 v18, v17, s30 // (op 975) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 975) most significant: sum


s_ashr_i64 s[0:1], s[42:43], 7 // (op 975) 
s_lshr_b32 s47, s0, 1 // (op 975) 

v_lshl_add_u32 v15, s47, 8, v16 // (op 975) 


v_mov_b32 v17, 0 // (op 975) convert
v_mov_b32 v16, v15 // (op 975) convert

v_add_co_u32 v20, s[0:1], v16, v18 // (op 975) least significant half
v_addc_co_u32 v21, s[0:1], v17, v19, s[0:1] // (op 975) most significant half



v_mul_lo_u32 v19, v21, 6 // (op 975) most significant: low of high * low
v_mul_hi_u32 v15, v20, 6 // (op 975) most significant: high of low * low
v_mul_lo_u32 v18, v20, 6 // (op 975) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 975) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 975) 

v_mov_b32 v15, v16 // (op 975) convert

s_mul_i32 s1, s5, 6 // (op 977) most significant: low of high * low
s_mul_i32 s0, s4, 6 // (op 977) least significant
s_mul_hi_u32 s47, s4, 6 // (op 977) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 977) most significant: sum

s_mul_i32 s19, s30, s1 // (op 977) most significant: low of low * high
s_mul_i32 s47, s31, s0 // (op 977) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 977) least significant
s_mul_hi_u32 s22, s30, s0 // (op 977) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 977) most significant: sum
s_add_u32 s19, s19, s22 // (op 977) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 977) 

s_mov_b32 s60, s32 // (op 977) reg expression
s_mov_b32 s61, s33 // (op 977) reg expression
s_mov_b32 s62, s0 // (op 977) reg expression
s_mov_b32 s63, 131072 // (op 977) call()

s_mul_hi_i32 s47, s16, s10 // (op 1039) 
v_lshrrev_b32 v16, 2, v1 // (op 1039) 
s_add_i32 s0, s47, s16 // (op 1039) 

v_and_b32 v17, -4, v1 // (op 1039) 
s_ashr_i32 s1, s0, 31 // (op 1039) 
v_sub_u32 v18, v1, v17 // (op 1039) 

s_cmp_eq_i32 s10, 0 // (op 1039) 
s_mov_b32 s47, scc // (op 1039) 
s_cmp_eq_i32 s47, 0 // (op 1039) 
s_cselect_b32 s18, 0, -1 // (op 1039) 

s_and_b32 s19, s13, 31 // (op 1039) 
s_lshl_b32 s47, 1, s19 // (op 1039) 
s_add_i32 s22, s47, s18 // (op 1039) 


s_and_b32 s18, s1, s22 // (op 1039) 


s_add_i32 s1, s0, s18 // (op 1039) 


s_ashr_i32 s0, s1, s19 // (op 1039) 


s_lshl_b32 s1, s13, 24 // (op 1039) 
s_ashr_i32 s47, s1, 31 // (op 1039) 

s_xor_b32 s1, s0, s47 // (op 1039) 

s_sub_i32 s0, s1, s47 // (op 1039) 


s_mul_hi_i32 s1, s0, s14 // (op 1039) 
s_add_i32 s47, s1, s0 // (op 1039) 

s_ashr_i32 s1, s47, 31 // (op 1039) 
s_cmp_eq_i32 s14, 0 // (op 1039) 
s_mov_b32 s18, scc // (op 1039) 
s_cmp_eq_i32 s18, 0 // (op 1039) 
s_cselect_b32 s19, 0, -1 // (op 1039) 

s_and_b32 s18, s15, 31 // (op 1039) 
s_lshl_b32 s22, 1, s18 // (op 1039) 
s_add_i32 s23, s22, s19 // (op 1039) 


s_and_b32 s19, s1, s23 // (op 1039) 


s_add_i32 s1, s47, s19 // (op 1039) 


s_ashr_i32 s19, s1, s18 // (op 1039) 


s_lshl_b32 s1, s15, 24 // (op 1039) 
s_ashr_i32 s18, s1, 31 // (op 1039) 

s_xor_b32 s1, s19, s18 // (op 1039) 

s_sub_i32 s19, s1, s18 // (op 1039) 


s_mul_i32 s1, s0, s11 // (op 1039) 

s_sub_u32 s0, s16, s1 // (op 1039) 

s_mul_i32 s1, s19, s11 // (op 1039) 
s_add_u32 s1, s1, s0 // (op 1039) 


s_mul_hi_i32 s0, s16, s9 // (op 1039) 
s_add_i32 s47, s0, s16 // (op 1039) 

s_ashr_i32 s0, s47, 31 // (op 1039) 
s_cmp_eq_i32 s9, 0 // (op 1039) 
s_mov_b32 s18, scc // (op 1039) 
s_cmp_eq_i32 s18, 0 // (op 1039) 
s_cselect_b32 s19, 0, -1 // (op 1039) 

s_and_b32 s18, s45, 31 // (op 1039) 
s_lshl_b32 s22, 1, s18 // (op 1039) 
s_add_i32 s23, s22, s19 // (op 1039) 


s_and_b32 s19, s0, s23 // (op 1039) 


s_add_i32 s0, s47, s19 // (op 1039) 


s_ashr_i32 s19, s0, s18 // (op 1039) 


s_lshl_b32 s0, s45, 24 // (op 1039) 
s_ashr_i32 s18, s0, 31 // (op 1039) 

s_xor_b32 s0, s19, s18 // (op 1039) 

s_sub_i32 s19, s0, s18 // (op 1039) 


s_add_u32 s22, s4, 32 // (op 1039) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1039) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1039) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1039) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 1039) 

s_mul_i32 s0, s11, s22 // (op 1039) 

s_mul_i32 s18, s8, s0 // (op 1039) 

s_mul_i32 s0, s19, s18 // (op 1039) 

s_sub_u32 s18, s16, s0 // (op 1039) 

s_mov_b32 s0, s18 // (op 1039) 
s_mul_hi_i32 s47, s0, s12 // (op 1039) 
s_add_i32 s22, s47, s0 // (op 1039) 


s_ashr_i32 s0, s22, 31 // (op 1039) 
s_cmp_eq_i32 s12, 0 // (op 1039) 
s_mov_b32 s47, scc // (op 1039) 
s_cmp_eq_i32 s47, 0 // (op 1039) 
s_cselect_b32 s23, 0, -1 // (op 1039) 

s_and_b32 s47, s44, 31 // (op 1039) 
s_lshl_b32 s24, 1, s47 // (op 1039) 
s_add_i32 s25, s24, s23 // (op 1039) 


s_and_b32 s23, s0, s25 // (op 1039) 


s_add_i32 s0, s22, s23 // (op 1039) 


s_ashr_i32 s22, s0, s47 // (op 1039) 


s_lshl_b32 s0, s44, 24 // (op 1039) 
s_ashr_i32 s47, s0, 31 // (op 1039) 

s_xor_b32 s0, s22, s47 // (op 1039) 

s_sub_i32 s22, s0, s47 // (op 1039) 


s_mul_i32 s0, s46, s11 // (op 1039) 
s_add_u32 s24, s6, 64 // (op 1039) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 1039) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 1039) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 1039) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 6 // (op 1039) 

s_sub_i32 s23, s24, s0 // (op 1039) 

s_mul_i32 s47, s22, s23 // (op 1039) 


s_sub_u32 s22, s18, s47 // (op 1039) 


s_add_u32 s18, s22, s0 // (op 1039) 


s_cmp_eq_i32 s19, 0 // (op 1039) 
s_mov_b32 s0, scc // (op 1039) 

s_cmp_eq_i32 s0, 0 // (op 1039) 
s_cselect_b32 s19, s18, s1 // (op 1039) 



v_lshl_add_u32 v17, s19, 6, v16 // (op 1039) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 1039) 
s_lshr_b32 s47, s0, 1 // (op 1039) 

v_lshl_add_u32 v16, s47, 3, v18 // (op 1039) 


v_mul_lo_u32 v18, v17, s34 // (op 1039) 
v_add_u32 v18, v18, v16 // (op 1039) 


s_mul_i32 s1, s34, s7 // (op 1041) most significant: low of low * high
s_mul_i32 s47, s35, s6 // (op 1041) most significant: low of high * low
s_mul_i32 s0, s34, s6 // (op 1041) least significant
s_mul_hi_u32 s18, s34, s6 // (op 1041) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1041) most significant: sum
s_add_u32 s1, s1, s18 // (op 1041) most significant: sum


s_mov_b32 s64, s36 // (op 1041) reg expression
s_mov_b32 s65, s37 // (op 1041) reg expression
s_mov_b32 s66, s0 // (op 1041) reg expression
s_mov_b32 s67, 131072 // (op 1041) call()

s_mul_hi_i32 s47, s16, s10 // (op 1103) 
v_lshrrev_b32 v16, 6, v1 // (op 1103) 
s_add_i32 s0, s47, s16 // (op 1103) 

v_and_b32 v17, 1, v16 // (op 1103) 

s_ashr_i32 s1, s0, 31 // (op 1103) 
v_and_b32 v16, 15, v1 // (op 1103) 
s_cmp_eq_i32 s10, 0 // (op 1103) 
s_mov_b32 s47, scc // (op 1103) 
v_lshl_add_u32 v19, v17, 4, v16 // (op 1103) 


s_cmp_eq_i32 s47, 0 // (op 1103) 
s_cselect_b32 s18, 0, -1 // (op 1103) 

v_and_b32 v16, 63, v1 // (op 1103) 
s_and_b32 s19, s13, 31 // (op 1103) 
v_lshrrev_b32 v17, 4, v16 // (op 1103) 

s_lshl_b32 s47, 1, s19 // (op 1103) 
s_add_i32 s22, s47, s18 // (op 1103) 


s_and_b32 s18, s1, s22 // (op 1103) 


s_add_i32 s1, s0, s18 // (op 1103) 


s_ashr_i32 s0, s1, s19 // (op 1103) 


s_lshl_b32 s1, s13, 24 // (op 1103) 
s_ashr_i32 s47, s1, 31 // (op 1103) 

s_xor_b32 s1, s0, s47 // (op 1103) 

s_sub_i32 s0, s1, s47 // (op 1103) 


s_mul_hi_i32 s1, s0, s14 // (op 1103) 
s_add_i32 s47, s1, s0 // (op 1103) 

s_ashr_i32 s1, s47, 31 // (op 1103) 
s_cmp_eq_i32 s14, 0 // (op 1103) 
s_mov_b32 s18, scc // (op 1103) 
s_cmp_eq_i32 s18, 0 // (op 1103) 
s_cselect_b32 s19, 0, -1 // (op 1103) 

s_and_b32 s18, s15, 31 // (op 1103) 
s_lshl_b32 s22, 1, s18 // (op 1103) 
s_add_i32 s23, s22, s19 // (op 1103) 


s_and_b32 s19, s1, s23 // (op 1103) 


s_add_i32 s1, s47, s19 // (op 1103) 


s_ashr_i32 s19, s1, s18 // (op 1103) 


s_lshl_b32 s1, s15, 24 // (op 1103) 
s_ashr_i32 s18, s1, 31 // (op 1103) 

s_xor_b32 s1, s19, s18 // (op 1103) 

s_sub_i32 s19, s1, s18 // (op 1103) 


s_add_u32 s22, s4, 32 // (op 1103) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 1103) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 1103) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 1103) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 5 // (op 1103) 

s_mul_i32 s1, s19, s22 // (op 1103) 

s_sub_i32 s47, s0, s1 // (op 1103) 


s_mul_hi_i32 s0, s16, s9 // (op 1103) 
s_add_i32 s1, s0, s16 // (op 1103) 

s_ashr_i32 s0, s1, 31 // (op 1103) 
s_cmp_eq_i32 s9, 0 // (op 1103) 
s_mov_b32 s18, scc // (op 1103) 
s_cmp_eq_i32 s18, 0 // (op 1103) 
s_cselect_b32 s19, 0, -1 // (op 1103) 

s_and_b32 s18, s45, 31 // (op 1103) 
s_lshl_b32 s24, 1, s18 // (op 1103) 
s_add_i32 s25, s24, s19 // (op 1103) 


s_and_b32 s19, s0, s25 // (op 1103) 


s_add_i32 s0, s1, s19 // (op 1103) 


s_ashr_i32 s1, s0, s18 // (op 1103) 


s_lshl_b32 s0, s45, 24 // (op 1103) 
s_ashr_i32 s18, s0, 31 // (op 1103) 

s_xor_b32 s0, s1, s18 // (op 1103) 

s_sub_i32 s1, s0, s18 // (op 1103) 


s_mul_i32 s0, s11, s22 // (op 1103) 

s_mul_i32 s18, s8, s0 // (op 1103) 

s_mul_i32 s0, s1, s18 // (op 1103) 

s_sub_u32 s18, s16, s0 // (op 1103) 

s_mul_hi_i32 s0, s18, s12 // (op 1103) 
s_add_i32 s19, s0, s18 // (op 1103) 


s_ashr_i32 s0, s19, 31 // (op 1103) 
s_cmp_eq_i32 s12, 0 // (op 1103) 
s_mov_b32 s18, scc // (op 1103) 
s_cmp_eq_i32 s18, 0 // (op 1103) 
s_cselect_b32 s22, 0, -1 // (op 1103) 

s_and_b32 s18, s44, 31 // (op 1103) 
s_lshl_b32 s23, 1, s18 // (op 1103) 
s_add_i32 s24, s23, s22 // (op 1103) 


s_and_b32 s25, s0, s24 // (op 1103) 


s_add_i32 s0, s19, s25 // (op 1103) 


s_ashr_i32 s19, s0, s18 // (op 1103) 


s_lshl_b32 s0, s44, 24 // (op 1103) 
s_ashr_i32 s18, s0, 31 // (op 1103) 

s_xor_b32 s0, s19, s18 // (op 1103) 

s_sub_i32 s19, s0, s18 // (op 1103) 


s_cmp_eq_i32 s1, 0 // (op 1103) 
s_mov_b32 s0, scc // (op 1103) 

s_cmp_eq_i32 s0, 0 // (op 1103) 
s_cselect_b32 s1, s19, s47 // (op 1103) 



v_lshl_add_u32 v16, s1, 5, v19 // (op 1103) 


v_mul_lo_u32 v19, v16, s38 // (op 1103) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 1103) 
s_lshr_b32 s47, s0, 1 // (op 1103) 

v_lshl_add_u32 v16, s47, 3, v17 // (op 1103) 


v_add_u32 v17, v16, v19 // (op 1103) 


s_mul_i32 s1, s38, s5 // (op 1105) most significant: low of low * high
s_mul_i32 s47, s39, s4 // (op 1105) most significant: low of high * low
s_mul_i32 s0, s38, s4 // (op 1105) least significant
s_mul_hi_u32 s18, s38, s4 // (op 1105) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 1105) most significant: sum
s_add_u32 s1, s1, s18 // (op 1105) most significant: sum


s_mov_b32 s68, s40 // (op 1105) reg expression
s_mov_b32 s69, s41 // (op 1105) reg expression
s_mov_b32 s70, s0 // (op 1105) reg expression
s_mov_b32 s71, 131072 // (op 1105) call()

v_and_b32 v16, -8, v1 // (op 1196) 
v_sub_u32 v19, v1, v16 // (op 1196) 

v_lshlrev_b32 v16, 4, v19 // (op 1196) 

v_lshrrev_b32 v19, 3, v1 // (op 1196) 
v_lshl_add_u32 v20, v19, 7, v16 // (op 1196) 


s_nop 0
v_readfirstlane_b32 s47, v20 // (op 1196) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -8, v1 // (op 1253) 
v_sub_u32 v19, v1, v16 // (op 1253) 

v_lshlrev_b32 v16, 4, v19 // (op 1253) 

v_lshrrev_b32 v19, 3, v1 // (op 1253) 
v_lshl_add_u32 v20, v19, 7, v16 // (op 1253) 


v_mul_lo_u32 v16, v20, 6 // (op 1253) 

v_lshrrev_b32 v19, 3, v16 // (op 1253) 

v_and_b32 v16, -4, v1 // (op 1306) 
v_sub_u32 v20, v1, v16 // (op 1306) 

v_and_b32 v16, 4294967292, v1 // (op 1306) 
v_add_u32 v21, v16, v20 // (op 1306) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 320) 
s_and_b32 s18, s0, 4294967294 // (op 320) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 319) 
s_mov_b32 s19, s0 // (op 319) convert

s_cmp_lt_u32 s18, s19 // (op 319) 




s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_319 // (op 319) Condition: Top (jump to RR_GEMM_TN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_319 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoopTail_319:
s_barrier  // (op 1501) 
s_add_u32 s18, s18, 1 // (op 321) 

s_lshl_b64 s[0:1], s[26:27], 5 // (op 878) 
v_mov_b64 v[22:23], s[0:1] // (op 878) convert


v_mov_b64 v[24:25], v[12:13] // (op 878) 
v_mov_b64 v[26:27], v[24:25] // (op 878) 
s_mov_b32 m0, s47 // (op 878) reg expression
buffer_load_dwordx4 v26, s[48:51], 0 offen offset: 0 lds // (op 878) Load value direct to lds
v_add_co_u32 v26, s[0:1], v26, v22 // (op 878) least significant half
v_addc_co_u32 v27, s[0:1], 0, v27, s[0:1] // (op 878) most significant half

s_add_u32 m0, m0, 4096 // (op 878) 
buffer_load_dwordx4 v26, s[48:51], 0 offen offset: 0 lds // (op 878) Load value direct to lds



v_add_co_u32 v12, vcc, 128, v12 // (op 897) least significant half
v_addc_co_u32 v13, vcc, 0, v13, vcc // (op 897) most significant half


v_mov_b32 v14, v15 // (op 183) 

buffer_load_dwordx3 v[22:24], v14, s[60:63], 0 offen // (op 183) Load value

v_add_u32 v15, 96, v15 // (op 967) 


s_barrier  // (op 1514) 
v_mov_b32 v14, v18 // (op 184) 




s_waitcnt vmcnt(0)
ds_write_b96 v19, v[22:24] offset:16384 // (op 181) Store local data 
buffer_load_ubyte v16, v14, s[64:67], 0 offen offset: 0 // (op 184) Load value


v_add_u32 v18, 4, v18 // (op 1031) 
s_barrier  // (op 1504) 
s_barrier  // (op 1528) 






ds_read_b128 v[22:25], v7 // (op 173) Load local data 
ds_read_b128 v[26:29], v7 offset:64 // (op 173) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v21, v16 offset:19456 // (op 182) Store local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 1518) 



ds_read_b128 v[30:33], v3 offset:16384 // (op 175) Load local data 
ds_read_b64 v[34:35], v3 offset:16400 // (op 175) Load local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 1532) 



ds_read_u8 v14, v0 offset:19456 // (op 177) Load local data 


v_mov_b32 v16, v17 // (op 179) 

buffer_load_ubyte v20, v16, s[68:71], 0 offen offset: 0 // (op 179) Load value

v_add_u32 v17, 4, v17 // (op 1095) 



ds_read_b128 v[36:39], v7 offset:4096 // (op 325) Load local data 
ds_read_b128 v[40:43], v7 offset:4160 // (op 325) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:29], v[30:35], a[0:3], v14, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 180) 





ds_read_u8 v14, v0 offset:19584 // (op 329) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[36:43], v[30:35], a[4:7], v14, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b010 // (op 331) 




s_ashr_i64 s[0:1], s[42:43], 7 // (op 319) 
s_mov_b32 s19, s0 // (op 319) convert

s_cmp_lt_u32 s18, s19 // (op 319) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoopTail_319 // (op 319) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopTop_KLoopTail_319 if true)
RR_GEMM_TN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ForLoopBottom_KLoopTail_319:





















s_barrier  // (op 1608) 



















v_cmp_eq_f32 vcc, 0.00000, v5 // (op 125) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalFalse_0_beta_125 // (op 125) Condition: False, jump to RR_GEMM_TN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalFalse_0_beta_125
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


buffer_store_dwordx2 v[14:15], v9, s[56:59], 0 offen // (op 54) Store value


v_accvgpr_read v10, a4 // (op 517) 
v_accvgpr_read v11, a5 // (op 517) 
v_accvgpr_read v12, a6 // (op 517) 
v_accvgpr_read v13, a7 // (op 517) 
v_mul_f32 v14, v8, v10 // (op 517) 
v_mul_f32 v15, v8, v11 // (op 517) 
v_mul_f32 v16, v8, v12 // (op 517) 
v_mul_f32 v17, v8, v13 // (op 517) 

v_lshrrev_b32 v10, 16, v14 // (op 519) 
v_lshrrev_b32 v11, 16, v15 // (op 519) 
v_lshrrev_b32 v12, 16, v16 // (op 519) 
v_lshrrev_b32 v13, 16, v17 // (op 519) 


v_pack_b32_f16 v14, v10, v11 // (op 520) 
v_pack_b32_f16 v15, v12, v13 // (op 520) 


buffer_store_dwordx2 v[14:15], v9, s[56:59], 0 offen offset:64 // (op 520) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalBottom_0_beta_125 // (op 125) Condition: Done, jump to RR_GEMM_TN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalBottom_0_beta_125
RR_GEMM_TN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalFalse_0_beta_125:



buffer_load_dwordx2 v[10:11], v6, s[52:55], 0 offen offset:64 // (op 550) Load value
v_accvgpr_read v12, a4 // (op 546) 
v_accvgpr_read v13, a5 // (op 546) 
v_accvgpr_read v14, a6 // (op 546) 
v_accvgpr_read v15, a7 // (op 546) 
v_mul_f32 v16, v8, v12 // (op 546) 
v_mul_f32 v17, v8, v13 // (op 546) 
v_mul_f32 v18, v8, v14 // (op 546) 
v_mul_f32 v19, v8, v15 // (op 546) 

s_waitcnt vmcnt(0)
v_and_b32 v12, 65535, v10 // (op 551) 
v_lshrrev_b32 v13, 16, v10 // (op 551) 
v_lshlrev_b32 v12, 16, v12 // (op 551) 
v_lshlrev_b32 v13, 16, v13 // (op 551) 
v_mul_f32 v20, v5, v12 // (op 551) 
v_mul_f32 v21, v5, v13 // (op 551) 
v_and_b32 v12, 65535, v11 // (op 551) 
v_lshrrev_b32 v13, 16, v11 // (op 551) 
v_lshlrev_b32 v12, 16, v12 // (op 551) 
v_lshlrev_b32 v13, 16, v13 // (op 551) 
v_mul_f32 v22, v5, v12 // (op 551) 
v_mul_f32 v23, v5, v13 // (op 551) 


v_add_f32 v10, v20, v16 // (op 547) 
v_add_f32 v11, v21, v17 // (op 547) 
v_add_f32 v12, v22, v18 // (op 547) 
v_add_f32 v13, v23, v19 // (op 547) 


v_lshrrev_b32 v14, 16, v10 // (op 548) 
v_lshrrev_b32 v15, 16, v11 // (op 548) 
v_lshrrev_b32 v16, 16, v12 // (op 548) 
v_lshrrev_b32 v17, 16, v13 // (op 548) 


v_pack_b32_f16 v10, v14, v15 // (op 549) 
v_pack_b32_f16 v11, v16, v17 // (op 549) 


buffer_store_dwordx2 v[10:11], v9, s[56:59], 0 offen offset:64 // (op 549) Store value





buffer_load_dwordx2 v[10:11], v6, s[52:55], 0 offen // (op 138) Load value
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


buffer_store_dwordx2 v[10:11], v9, s[56:59], 0 offen // (op 137) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__ConditionalBottom_0_beta_125:










s_endpgm  // End of RR_GEMM_TN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_
.LRR_GEMM_TN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__end:
.size RR_GEMM_TN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_, .LRR_GEMM_TN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM__end-RR_GEMM_TN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 68
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 23040 // lds bytes
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
  MXScaleFormat: NoSwizzle
  ProblemType:
    OperationType: GEMM
    DataType: F8
    DataTypeA: F8
    DataTypeB: F6
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
            { type: int64, semantic: SizeFree1 },
            { type: int64, semantic: SizeFree0 },
            { type: int32, semantic: RR_QuotientTilesByBlock },
            { type: int32, semantic: RR_MagicMultipleWgmMainBlock },
            { type: int32, semantic: RR_MagicMultipleWgm },
            { type: int32, semantic: RR_WorkgroupMapping },
            { type: int32, semantic: RR_MagicMultipleWgmTail },
            { type: uint32, semantic: RR_MagicShiftAndSignWgm },
            { type: int32, semantic: RR_MagicMultipleNumTilesN },
            { type: uint32, semantic: RR_MagicShiftAndSignNumTilesN },
            { type: int64, semantic: StrideD0 },
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
    macrotile: [64, 32, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
    unroll: 2
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [2, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_FP8_FP6_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x32x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 23040
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 74
    .vgpr_count: 60
    .agpr_count: 8
    .max_flat_workgroup_size: 256
    .wavefront_size: 64
    .args:
      - .name: Tensor_10_stride_1_10
        .size: 8
        .offset: 0
        .value_kind: by_value
      - .name: Tensor_2_size_1_4
        .size: 8
        .offset: 8
        .value_kind: by_value
      - .name: Tensor_0_size_0_1
        .size: 8
        .offset: 16
        .value_kind: by_value
      - .name: Subtract_21
        .size: 4
        .offset: 24
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 28
        .value_kind: by_value
      - .name: MagicMultiple_22
        .size: 4
        .offset: 32
        .value_kind: by_value
      - .name: WGM_18
        .size: 4
        .offset: 36
        .value_kind: by_value
      - .name: MagicMultiple_30
        .size: 4
        .offset: 40
        .value_kind: by_value
      - .name: MagicShiftAndSign_23
        .size: 4
        .offset: 44
        .value_kind: by_value
      - .name: MagicMultiple_28
        .size: 4
        .offset: 48
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 52
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 56
        .value_kind: by_value
      - .name: Tensor_10_pointer
        .size: 8
        .offset: 64
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 72
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_stride_0_0
        .size: 8
        .offset: 80
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 88
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_stride_1_3
        .size: 8
        .offset: 96
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 104
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
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
      - .name: Tensor_7_stride_1_8
        .size: 8
        .offset: 128
        .value_kind: by_value
      - .name: Tensor_7_pointer
        .size: 8
        .offset: 136
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_0_size_1_14
        .size: 8
        .offset: 144
        .value_kind: by_value
      - .name: MagicShiftAndSign_31
        .size: 4
        .offset: 152
        .value_kind: by_value
      - .name: MagicShiftAndSign_27
        .size: 4
        .offset: 156
        .value_kind: by_value
      - .name: Subtract_24
        .size: 4
        .offset: 160
        .value_kind: by_value
      - .name: user_Float_Value_14
        .size: 4
        .offset: 164
        .value_kind: by_value
      - .name: user_Float_Value_12
        .size: 4
        .offset: 168
        .value_kind: by_value
...

.end_amdgpu_metadata

