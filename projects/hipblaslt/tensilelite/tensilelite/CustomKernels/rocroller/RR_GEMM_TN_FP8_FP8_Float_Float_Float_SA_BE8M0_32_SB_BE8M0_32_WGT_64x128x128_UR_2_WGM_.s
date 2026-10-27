// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM_,@function
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM_:
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
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 4501) 
s_mul_hi_i32 s1, s16, s10 // (op 4756) 
v_lshrrev_b32 v0, 7, v1 // (op 4562) 
v_and_b32 v3, 15, v1 // (op 4562) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 4562) 


v_and_b32 v0, 48, v1 // (op 4562) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 4562) 


v_lshrrev_b32 v0, 6, v1 // (op 4619) 
v_and_b32 v4, 1, v0 // (op 4619) 

v_and_b32 v0, 15, v1 // (op 4619) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 4619) 


v_and_b32 v0, 48, v1 // (op 4619) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 4619) 


v_lshrrev_b32 v0, 7, v1 // (op 4677) 
v_and_b32 v5, 15, v1 // (op 4677) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 4677) 


v_and_b32 v0, 63, v1 // (op 4677) 
v_lshrrev_b32 v5, 4, v0 // (op 4677) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 4677) 


v_lshrrev_b32 v5, 6, v1 // (op 4735) 
v_and_b32 v6, 1, v5 // (op 4735) 

v_and_b32 v5, 15, v1 // (op 4735) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 4735) 


v_and_b32 v5, 63, v1 // (op 4735) 
v_lshrrev_b32 v6, 4, v5 // (op 4735) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 4735) 


v_lshrrev_b32 v6, 7, v1 // (op 4802) 
v_and_b32 v7, 15, v1 // (op 4802) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 4802) 


v_and_b32 v6, 48, v1 // (op 4802) 
v_lshl_add_u32 v7, v8, 7, v6 // (op 4802) 


v_lshrrev_b32 v6, 6, v1 // (op 4846) 
v_and_b32 v8, 1, v6 // (op 4846) 

v_and_b32 v6, 15, v1 // (op 4846) 
v_lshl_add_u32 v9, v8, 4, v6 // (op 4846) 


v_and_b32 v6, 48, v1 // (op 4846) 
v_lshl_add_u32 v8, v9, 7, v6 // (op 4846) 


v_lshrrev_b32 v6, 7, v1 // (op 4875) 
v_and_b32 v9, 15, v1 // (op 4875) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 4875) 


v_and_b32 v6, 63, v1 // (op 4875) 
v_lshrrev_b32 v9, 4, v6 // (op 4875) 

v_lshl_add_u32 v6, v10, 2, v9 // (op 4875) 


v_lshrrev_b32 v9, 6, v1 // (op 4904) 
v_and_b32 v10, 1, v9 // (op 4904) 

v_and_b32 v9, 15, v1 // (op 4904) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 4904) 


v_and_b32 v9, 63, v1 // (op 4904) 
v_lshrrev_b32 v10, 4, v9 // (op 4904) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 4904) 


v_lshrrev_b32 v10, 7, v1 // (op 4501) 
s_add_i32 s18, s0, s16 // (op 4501) 

v_and_b32 v11, 63, v1 // (op 4501) 
s_ashr_i32 s0, s18, 31 // (op 4501) 
v_lshrrev_b32 v12, 4, v11 // (op 4501) 
s_cmp_eq_i32 s10, 0 // (op 4501) 
s_mov_b32 s19, scc // (op 4501) 
v_lshlrev_b32 v13, 2, v12 // (op 4501) 

s_cmp_eq_i32 s19, 0 // (op 4501) 
s_cselect_b32 s49, 0, -1 // (op 4501) 

v_lshl_add_u32 v12, v10, 4, v13 // (op 4501) 


s_and_b32 s19, s13, 31 // (op 4501) 
v_lshrrev_b32 v10, 2, v11 // (op 4501) 

v_and_b32 v11, 3, v10 // (op 4501) 

v_lshrrev_b32 v10, 7, v1 // (op 4756) 
v_and_b32 v13, 63, v1 // (op 4756) 
s_lshl_b32 s50, 1, s19 // (op 4501) 
v_and_b32 v14, 3, v1 // (op 4501) 
s_add_i32 s51, s50, s49 // (op 4501) 


s_and_b32 s49, s0, s51 // (op 4501) 


s_add_i32 s0, s1, s16 // (op 4756) 

s_ashr_i32 s1, s0, 31 // (op 4756) 
v_lshl_add_u32 v15, v11, 2, v14 // (op 4501) 


s_add_i32 s50, s18, s49 // (op 4501) 


v_lshrrev_b32 v11, 6, v1 // (op 4501) 
s_ashr_i32 s18, s50, s19 // (op 4501) 


v_and_b32 v14, 1, v11 // (op 4501) 

s_lshl_b32 s19, s13, 24 // (op 4501) 
v_lshl_add_u32 v11, v14, 4, v15 // (op 4501) 


s_ashr_i32 s49, s19, 31 // (op 4501) 

s_xor_b32 s19, s18, s49 // (op 4501) 

s_sub_i32 s18, s19, s49 // (op 4501) 


s_mul_hi_i32 s19, s18, s14 // (op 4501) 
s_add_i32 s49, s19, s18 // (op 4501) 

s_ashr_i32 s19, s49, 31 // (op 4501) 
s_cmp_eq_i32 s14, 0 // (op 4501) 
s_mov_b32 s50, scc // (op 4501) 
s_cmp_eq_i32 s50, 0 // (op 4501) 
s_cselect_b32 s51, 0, -1 // (op 4501) 

s_and_b32 s50, s15, 31 // (op 4501) 
v_lshrrev_b32 v14, 4, v13 // (op 4756) 
s_cmp_eq_i32 s10, 0 // (op 4756) 
s_mov_b32 s52, scc // (op 4756) 
v_lshlrev_b32 v15, 2, v14 // (op 4756) 

s_cmp_eq_i32 s52, 0 // (op 4756) 
s_cselect_b32 s53, 0, -1 // (op 4756) 

s_lshl_b32 s52, 1, s50 // (op 4501) 
v_lshl_add_u32 v14, v10, 4, v15 // (op 4756) 


v_lshrrev_b32 v10, 2, v13 // (op 4756) 

s_add_i32 s54, s52, s51 // (op 4501) 


s_and_b32 s51, s19, s54 // (op 4501) 


s_add_i32 s19, s49, s51 // (op 4501) 


s_ashr_i32 s49, s19, s50 // (op 4501) 


s_lshl_b32 s19, s15, 24 // (op 4501) 
s_ashr_i32 s50, s19, 31 // (op 4501) 

s_xor_b32 s19, s49, s50 // (op 4501) 

s_sub_i32 s49, s19, s50 // (op 4501) 


s_mul_i32 s19, s18, s11 // (op 4501) 
s_sub_u32 s50, s16, s19 // (op 4501) 

s_mul_i32 s19, s49, s11 // (op 4501) 
s_add_u32 s19, s19, s50 // (op 4501) 

s_mul_hi_i32 s50, s16, s9 // (op 4501) 
s_add_i32 s51, s50, s16 // (op 4501) 

s_ashr_i32 s50, s51, 31 // (op 4501) 
s_cmp_eq_i32 s9, 0 // (op 4501) 
s_mov_b32 s52, scc // (op 4501) 
s_cmp_eq_i32 s52, 0 // (op 4501) 
s_cselect_b32 s54, 0, -1 // (op 4501) 

s_and_b32 s52, s13, 31 // (op 4756) 
v_and_b32 v13, 3, v10 // (op 4756) 

s_lshl_b32 s55, 1, s52 // (op 4756) 
v_and_b32 v10, 3, v1 // (op 4756) 
s_add_i32 s56, s55, s53 // (op 4756) 


s_and_b32 s53, s1, s56 // (op 4756) 


v_lshl_add_u32 v15, v13, 2, v10 // (op 4756) 


s_add_i32 s1, s0, s53 // (op 4756) 


v_lshrrev_b32 v10, 6, v1 // (op 4756) 
s_ashr_i32 s0, s1, s52 // (op 4756) 


v_and_b32 v13, 1, v10 // (op 4756) 

s_lshl_b32 s1, s13, 24 // (op 4756) 
v_lshl_add_u32 v10, v13, 4, v15 // (op 4756) 


s_ashr_i32 s52, s1, 31 // (op 4756) 

s_xor_b32 s1, s0, s52 // (op 4756) 

s_sub_i32 s0, s1, s52 // (op 4756) 


s_mul_hi_i32 s1, s0, s14 // (op 4756) 
s_add_i32 s52, s1, s0 // (op 4756) 

s_ashr_i32 s1, s52, 31 // (op 4756) 
s_cmp_eq_i32 s14, 0 // (op 4756) 
s_mov_b32 s53, scc // (op 4756) 
s_cmp_eq_i32 s53, 0 // (op 4756) 
s_cselect_b32 s55, 0, -1 // (op 4756) 

s_and_b32 s53, s15, 31 // (op 4756) 
s_lshl_b32 s56, 1, s53 // (op 4756) 
s_add_i32 s57, s56, s55 // (op 4756) 


s_and_b32 s55, s1, s57 // (op 4756) 


s_add_i32 s1, s52, s55 // (op 4756) 


s_ashr_i32 s52, s1, s53 // (op 4756) 


s_lshl_b32 s1, s15, 24 // (op 4756) 
s_ashr_i32 s53, s1, 31 // (op 4756) 

s_xor_b32 s1, s52, s53 // (op 4756) 

s_sub_i32 s52, s1, s53 // (op 4756) 


s_mul_i32 s1, s0, s11 // (op 4756) 
s_sub_u32 s53, s16, s1 // (op 4756) 

s_mul_i32 s1, s52, s11 // (op 4756) 
s_add_u32 s1, s1, s53 // (op 4756) 

s_mul_hi_i32 s53, s16, s9 // (op 4756) 
s_add_i32 s55, s53, s16 // (op 4756) 

s_ashr_i32 s53, s55, 31 // (op 4756) 
s_cmp_eq_i32 s9, 0 // (op 4756) 
s_mov_b32 s56, scc // (op 4756) 
s_cmp_eq_i32 s56, 0 // (op 4756) 
s_cselect_b32 s57, 0, -1 // (op 4756) 


s_waitcnt lgkmcnt(0)
s_and_b32 s56, s45, 31 // (op 4501) 
s_lshl_b32 s58, 1, s56 // (op 4501) 
s_add_i32 s59, s58, s54 // (op 4501) 


s_and_b32 s54, s50, s59 // (op 4501) 


s_add_i32 s50, s51, s54 // (op 4501) 


s_ashr_i32 s51, s50, s56 // (op 4501) 


s_lshl_b32 s50, s45, 24 // (op 4501) 
s_ashr_i32 s54, s50, 31 // (op 4501) 

s_xor_b32 s50, s51, s54 // (op 4501) 

s_sub_i32 s51, s50, s54 // (op 4501) 


s_and_b32 s50, s45, 31 // (op 4756) 
s_lshl_b32 s54, 1, s50 // (op 4756) 
s_add_i32 s56, s54, s57 // (op 4756) 


s_and_b32 s54, s53, s56 // (op 4756) 


s_add_i32 s53, s55, s54 // (op 4756) 


s_ashr_i32 s54, s53, s50 // (op 4756) 


s_lshl_b32 s50, s45, 24 // (op 4756) 
s_ashr_i32 s53, s50, 31 // (op 4756) 

s_xor_b32 s50, s54, s53 // (op 4756) 

s_sub_i32 s54, s50, s53 // (op 4756) 


s_add_u32 s56, s6, 128 // (op 4501) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 4501) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 4501) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 4501) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 7 // (op 4501) 

s_mul_i32 s50, s11, s56 // (op 4501) 
s_mul_i32 s53, s8, s50 // (op 4501) 

s_mul_i32 s50, s51, s53 // (op 4501) 

s_sub_u32 s53, s16, s50 // (op 4501) 

s_mov_b32 s50, s53 // (op 4501) 
s_mul_hi_i32 s55, s50, s12 // (op 4501) 
s_add_i32 s58, s55, s50 // (op 4501) 


s_ashr_i32 s50, s58, 31 // (op 4501) 
s_cmp_eq_i32 s12, 0 // (op 4501) 
s_mov_b32 s55, scc // (op 4501) 
s_cmp_eq_i32 s55, 0 // (op 4501) 
s_cselect_b32 s59, 0, -1 // (op 4501) 

s_and_b32 s55, s44, 31 // (op 4501) 
s_lshl_b32 s60, 1, s55 // (op 4501) 
s_add_i32 s61, s60, s59 // (op 4501) 


s_and_b32 s59, s50, s61 // (op 4501) 


s_add_i32 s50, s58, s59 // (op 4501) 


s_ashr_i32 s58, s50, s55 // (op 4501) 


s_lshl_b32 s50, s44, 24 // (op 4501) 
s_ashr_i32 s55, s50, 31 // (op 4501) 

s_xor_b32 s50, s58, s55 // (op 4501) 

s_sub_i32 s58, s50, s55 // (op 4501) 


s_mul_i32 s50, s46, s11 // (op 4501) 
s_add_u32 s60, s20, 64 // (op 4501) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 4501) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 4501) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 4501) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 6 // (op 4501) 

s_sub_i32 s55, s60, s50 // (op 4501) 

s_mul_i32 s59, s58, s55 // (op 4501) 

s_sub_u32 s55, s53, s59 // (op 4501) 


s_add_u32 s53, s55, s50 // (op 4501) 


s_cmp_eq_i32 s51, 0 // (op 4501) 
s_mov_b32 s50, scc // (op 4501) 

s_cmp_eq_i32 s50, 0 // (op 4501) 
s_cselect_b32 s51, s53, s19 // (op 4501) 


s_mul_i32 s19, s49, s56 // (op 4501) 


v_lshl_add_u32 v13, s51, 6, v12 // (op 4501) 


s_sub_i32 s49, s18, s19 // (op 4501) 


s_cmp_eq_i32 s50, 0 // (op 4501) 
s_cselect_b32 s51, s58, s49 // (op 4501) 



v_lshl_add_u32 v12, s51, 7, v11 // (op 4501) 


v_mul_lo_u32 v11, v12, s2 // (op 4501) 

v_add_lshl_u32 v12, v13, v11, 2 // (op 4501) 


s_add_u32 s18, s6, 128 // (op 4756) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 4756) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 4756) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 4756) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 7 // (op 4756) 

s_mul_i32 s53, s11, s18 // (op 4756) 
s_mul_i32 s49, s8, s53 // (op 4756) 

s_mul_i32 s53, s54, s49 // (op 4756) 

s_sub_u32 s49, s16, s53 // (op 4756) 

s_mov_b32 s53, s49 // (op 4756) 
s_mul_hi_i32 s50, s53, s12 // (op 4756) 
s_add_i32 s51, s50, s53 // (op 4756) 


s_ashr_i32 s50, s51, 31 // (op 4756) 
s_cmp_eq_i32 s12, 0 // (op 4756) 
s_mov_b32 s53, scc // (op 4756) 
s_cmp_eq_i32 s53, 0 // (op 4756) 
s_cselect_b32 s55, 0, -1 // (op 4756) 

s_and_b32 s53, s44, 31 // (op 4756) 
s_lshl_b32 s56, 1, s53 // (op 4756) 
s_add_i32 s57, s56, s55 // (op 4756) 


s_and_b32 s55, s50, s57 // (op 4756) 


s_add_i32 s50, s51, s55 // (op 4756) 


s_ashr_i32 s51, s50, s53 // (op 4756) 


s_lshl_b32 s50, s44, 24 // (op 4756) 
s_ashr_i32 s53, s50, 31 // (op 4756) 

s_xor_b32 s50, s51, s53 // (op 4756) 

s_sub_i32 s51, s50, s53 // (op 4756) 


s_mul_i32 s50, s46, s11 // (op 4756) 
s_add_u32 s56, s20, 64 // (op 4756) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 4756) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 4756) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 4756) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 4756) 

s_sub_i32 s53, s56, s50 // (op 4756) 

s_mul_i32 s55, s51, s53 // (op 4756) 

s_sub_u32 s53, s49, s55 // (op 4756) 


s_add_u32 s49, s53, s50 // (op 4756) 


s_cmp_eq_i32 s54, 0 // (op 4756) 
s_mov_b32 s50, scc // (op 4756) 

s_cmp_eq_i32 s50, 0 // (op 4756) 
s_cselect_b32 s53, s49, s1 // (op 4756) 


s_mul_i32 s1, s52, s18 // (op 4756) 


v_lshl_add_u32 v11, s53, 6, v14 // (op 4756) 


s_sub_i32 s49, s0, s1 // (op 4756) 


s_cmp_eq_i32 s50, 0 // (op 4756) 
s_cselect_b32 s0, s51, s49 // (op 4756) 



v_lshl_add_u32 v13, s0, 7, v10 // (op 4756) 


v_mul_lo_u32 v10, v13, s4 // (op 4756) 

v_add_lshl_u32 v13, v11, v10, 2 // (op 4756) 


s_mul_i32 s1, s2, s7 // (op 4503) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 4503) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 4503) least significant
s_mul_hi_u32 s19, s2, s6 // (op 4503) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4503) most significant: sum
s_add_u32 s1, s1, s19 // (op 4503) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 4503) 

s_mul_i32 s1, s4, s7 // (op 4758) most significant: low of low * high
s_mul_i32 s49, s5, s6 // (op 4758) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 4758) least significant
s_mul_hi_u32 s50, s4, s6 // (op 4758) most significant: high of low * low
s_add_u32 s1, s1, s49 // (op 4758) most significant: sum
s_add_u32 s1, s1, s50 // (op 4758) most significant: sum


s_lshl_b64 s[50:51], s[0:1], 2 // (op 4758) 

s_mov_b32 s52, s22 // (op 4503) reg expression
s_mov_b32 s53, s23 // (op 4503) reg expression
s_mov_b32 s54, s18 // (op 4503) reg expression
s_mov_b32 s55, 131072 // (op 4503) call()

s_mov_b32 s56, s24 // (op 4758) reg expression

s_mov_b32 s57, s25 // (op 4758) reg expression
s_mov_b32 s58, s50 // (op 4758) reg expression
s_mov_b32 s59, 131072 // (op 4758) call()


v_mov_b32 v10, s47 // (op 2) Move value
v_mov_b32 v11, s48 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 63) call()
v_accvgpr_write a1, 0.00000 // (op 63) call()
v_accvgpr_write a2, 0.00000 // (op 63) call()
v_accvgpr_write a3, 0.00000 // (op 63) call()
v_accvgpr_write a4, 0.00000 // (op 336) call()
v_accvgpr_write a5, 0.00000 // (op 336) call()
v_accvgpr_write a6, 0.00000 // (op 336) call()
v_accvgpr_write a7, 0.00000 // (op 336) call()
v_accvgpr_write a8, 0.00000 // (op 512) call()
v_accvgpr_write a9, 0.00000 // (op 512) call()
v_accvgpr_write a10, 0.00000 // (op 512) call()
v_accvgpr_write a11, 0.00000 // (op 512) call()
v_accvgpr_write a12, 0.00000 // (op 688) call()
v_accvgpr_write a13, 0.00000 // (op 688) call()
v_accvgpr_write a14, 0.00000 // (op 688) call()
v_accvgpr_write a15, 0.00000 // (op 688) call()
v_accvgpr_write a16, 0.00000 // (op 952) call()
v_accvgpr_write a17, 0.00000 // (op 952) call()
v_accvgpr_write a18, 0.00000 // (op 952) call()
v_accvgpr_write a19, 0.00000 // (op 952) call()
v_accvgpr_write a20, 0.00000 // (op 1261) call()
v_accvgpr_write a21, 0.00000 // (op 1261) call()
v_accvgpr_write a22, 0.00000 // (op 1261) call()
v_accvgpr_write a23, 0.00000 // (op 1261) call()
v_accvgpr_write a24, 0.00000 // (op 1262) call()
v_accvgpr_write a25, 0.00000 // (op 1262) call()
v_accvgpr_write a26, 0.00000 // (op 1262) call()
v_accvgpr_write a27, 0.00000 // (op 1262) call()
v_accvgpr_write a28, 0.00000 // (op 1263) call()
v_accvgpr_write a29, 0.00000 // (op 1263) call()
v_accvgpr_write a30, 0.00000 // (op 1263) call()
v_accvgpr_write a31, 0.00000 // (op 1263) call()
s_mul_hi_i32 s0, s16, s10 // (op 4263) 
v_lshrrev_b32 v14, 3, v1 // (op 4263) 
s_add_i32 s1, s0, s16 // (op 4263) 

v_and_b32 v15, -8, v1 // (op 4263) 
s_ashr_i32 s0, s1, 31 // (op 4263) 
v_sub_u32 v16, v1, v15 // (op 4263) 

s_cmp_eq_i32 s10, 0 // (op 4263) 
s_mov_b32 s18, scc // (op 4263) 
v_lshlrev_b32 v15, 4, v16 // (op 4263) 

s_cmp_eq_i32 s18, 0 // (op 4263) 
s_cselect_b32 s19, 0, -1 // (op 4263) 

v_mov_b32 v17, 0 // (op 4263) convert
v_mov_b32 v16, v15 // (op 4263) convert

s_and_b32 s18, s13, 31 // (op 4263) 
s_lshl_b32 s22, 1, s18 // (op 4263) 
s_add_i32 s23, s22, s19 // (op 4263) 


s_and_b32 s19, s0, s23 // (op 4263) 


s_add_i32 s0, s1, s19 // (op 4263) 


s_ashr_i32 s1, s0, s18 // (op 4263) 


s_lshl_b32 s0, s13, 24 // (op 4263) 
s_ashr_i32 s18, s0, 31 // (op 4263) 

s_xor_b32 s0, s1, s18 // (op 4263) 

s_sub_i32 s1, s0, s18 // (op 4263) 


s_mul_hi_i32 s0, s1, s14 // (op 4263) 
s_add_i32 s18, s0, s1 // (op 4263) 

s_ashr_i32 s0, s18, 31 // (op 4263) 
s_cmp_eq_i32 s14, 0 // (op 4263) 
s_mov_b32 s19, scc // (op 4263) 
s_cmp_eq_i32 s19, 0 // (op 4263) 
s_cselect_b32 s22, 0, -1 // (op 4263) 

s_and_b32 s19, s15, 31 // (op 4263) 
s_lshl_b32 s23, 1, s19 // (op 4263) 
s_add_i32 s24, s23, s22 // (op 4263) 


s_and_b32 s25, s0, s24 // (op 4263) 


s_add_i32 s0, s18, s25 // (op 4263) 


s_ashr_i32 s18, s0, s19 // (op 4263) 


s_lshl_b32 s0, s15, 24 // (op 4263) 
s_ashr_i32 s19, s0, 31 // (op 4263) 

s_xor_b32 s0, s18, s19 // (op 4263) 

s_sub_i32 s18, s0, s19 // (op 4263) 


s_mul_i32 s0, s1, s11 // (op 4263) 

s_sub_u32 s1, s16, s0 // (op 4263) 

s_mul_i32 s0, s18, s11 // (op 4263) 
s_add_u32 s0, s0, s1 // (op 4263) 


s_mul_hi_i32 s1, s16, s9 // (op 4263) 
s_add_i32 s18, s1, s16 // (op 4263) 

s_ashr_i32 s1, s18, 31 // (op 4263) 
s_cmp_eq_i32 s9, 0 // (op 4263) 
s_mov_b32 s19, scc // (op 4263) 
s_cmp_eq_i32 s19, 0 // (op 4263) 
s_cselect_b32 s22, 0, -1 // (op 4263) 

s_and_b32 s19, s45, 31 // (op 4263) 
s_lshl_b32 s23, 1, s19 // (op 4263) 
s_add_i32 s24, s23, s22 // (op 4263) 


s_and_b32 s25, s1, s24 // (op 4263) 


s_add_i32 s1, s18, s25 // (op 4263) 


s_ashr_i32 s18, s1, s19 // (op 4263) 


s_lshl_b32 s1, s45, 24 // (op 4263) 
s_ashr_i32 s19, s1, 31 // (op 4263) 

s_xor_b32 s1, s18, s19 // (op 4263) 

s_sub_i32 s18, s1, s19 // (op 4263) 


s_add_u32 s22, s6, 128 // (op 4263) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4263) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 4263) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 4263) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 4263) 

s_mul_i32 s1, s11, s22 // (op 4263) 

s_mul_i32 s19, s8, s1 // (op 4263) 

s_mul_i32 s1, s18, s19 // (op 4263) 

s_sub_u32 s19, s16, s1 // (op 4263) 

s_mov_b32 s1, s19 // (op 4263) 
s_mul_hi_i32 s22, s1, s12 // (op 4263) 
s_add_i32 s23, s22, s1 // (op 4263) 


s_ashr_i32 s1, s23, 31 // (op 4263) 
s_cmp_eq_i32 s12, 0 // (op 4263) 
s_mov_b32 s22, scc // (op 4263) 
s_cmp_eq_i32 s22, 0 // (op 4263) 
s_cselect_b32 s24, 0, -1 // (op 4263) 

s_and_b32 s22, s44, 31 // (op 4263) 
s_lshl_b32 s25, 1, s22 // (op 4263) 
s_add_i32 s47, s25, s24 // (op 4263) 


s_and_b32 s24, s1, s47 // (op 4263) 


s_add_i32 s1, s23, s24 // (op 4263) 


s_ashr_i32 s23, s1, s22 // (op 4263) 


s_lshl_b32 s1, s44, 24 // (op 4263) 
s_ashr_i32 s22, s1, 31 // (op 4263) 

s_xor_b32 s1, s23, s22 // (op 4263) 

s_sub_i32 s23, s1, s22 // (op 4263) 


s_mul_i32 s1, s46, s11 // (op 4263) 
s_add_u32 s24, s20, 64 // (op 4263) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 4263) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 4263) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 4263) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 4263) 

s_sub_i32 s22, s24, s1 // (op 4263) 

s_mul_i32 s24, s23, s22 // (op 4263) 


s_sub_u32 s25, s19, s24 // (op 4263) 


s_add_u32 s19, s25, s1 // (op 4263) 


s_cmp_eq_i32 s18, 0 // (op 4263) 
s_mov_b32 s1, scc // (op 4263) 

s_cmp_eq_i32 s1, 0 // (op 4263) 
s_cselect_b32 s18, s19, s0 // (op 4263) 



v_lshl_add_u32 v15, s18, 6, v14 // (op 4263) 


v_mul_lo_u32 v19, v15, s27 // (op 4263) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 4263) most significant: high of low * low
v_mul_lo_u32 v18, v15, s26 // (op 4263) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 4263) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 4263) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 4263) most significant half



s_mul_i32 s1, s26, s21 // (op 4265) most significant: low of low * high
s_mul_i32 s18, s27, s20 // (op 4265) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 4265) least significant
s_mul_hi_u32 s19, s26, s20 // (op 4265) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4265) most significant: sum
s_add_u32 s1, s1, s19 // (op 4265) most significant: sum


s_mov_b32 s48, s28 // (op 4265) reg expression
s_mov_b32 s49, s29 // (op 4265) reg expression
s_mov_b32 s50, s0 // (op 4265) reg expression
s_mov_b32 s51, 131072 // (op 4265) call()

s_mul_hi_i32 s47, s16, s10 // (op 4282) 
v_lshrrev_b32 v14, 3, v1 // (op 4282) 
s_add_i32 s0, s47, s16 // (op 4282) 

v_and_b32 v15, -8, v1 // (op 4282) 
s_ashr_i32 s1, s0, 31 // (op 4282) 
v_sub_u32 v16, v1, v15 // (op 4282) 

s_cmp_eq_i32 s10, 0 // (op 4282) 
s_mov_b32 s47, scc // (op 4282) 
v_lshlrev_b32 v15, 4, v16 // (op 4282) 

s_cmp_eq_i32 s47, 0 // (op 4282) 
s_cselect_b32 s18, 0, -1 // (op 4282) 

v_mov_b32 v17, 0 // (op 4282) convert
v_mov_b32 v16, v15 // (op 4282) convert

s_and_b32 s19, s13, 31 // (op 4282) 
s_lshl_b32 s47, 1, s19 // (op 4282) 
s_add_i32 s22, s47, s18 // (op 4282) 


s_and_b32 s18, s1, s22 // (op 4282) 


s_add_i32 s1, s0, s18 // (op 4282) 


s_ashr_i32 s0, s1, s19 // (op 4282) 


s_lshl_b32 s1, s13, 24 // (op 4282) 
s_ashr_i32 s47, s1, 31 // (op 4282) 

s_xor_b32 s1, s0, s47 // (op 4282) 

s_sub_i32 s0, s1, s47 // (op 4282) 


s_mul_hi_i32 s1, s0, s14 // (op 4282) 
s_add_i32 s47, s1, s0 // (op 4282) 

s_ashr_i32 s1, s47, 31 // (op 4282) 
s_cmp_eq_i32 s14, 0 // (op 4282) 
s_mov_b32 s18, scc // (op 4282) 
s_cmp_eq_i32 s18, 0 // (op 4282) 
s_cselect_b32 s19, 0, -1 // (op 4282) 

s_and_b32 s18, s15, 31 // (op 4282) 
s_lshl_b32 s22, 1, s18 // (op 4282) 
s_add_i32 s23, s22, s19 // (op 4282) 


s_and_b32 s19, s1, s23 // (op 4282) 


s_add_i32 s1, s47, s19 // (op 4282) 


s_ashr_i32 s19, s1, s18 // (op 4282) 


s_lshl_b32 s1, s15, 24 // (op 4282) 
s_ashr_i32 s18, s1, 31 // (op 4282) 

s_xor_b32 s1, s19, s18 // (op 4282) 

s_sub_i32 s19, s1, s18 // (op 4282) 


s_mul_i32 s1, s0, s11 // (op 4282) 

s_sub_u32 s0, s16, s1 // (op 4282) 

s_mul_i32 s1, s19, s11 // (op 4282) 
s_add_u32 s1, s1, s0 // (op 4282) 


s_mul_hi_i32 s0, s16, s9 // (op 4282) 
s_add_i32 s47, s0, s16 // (op 4282) 

s_ashr_i32 s0, s47, 31 // (op 4282) 
s_cmp_eq_i32 s9, 0 // (op 4282) 
s_mov_b32 s18, scc // (op 4282) 
s_cmp_eq_i32 s18, 0 // (op 4282) 
s_cselect_b32 s19, 0, -1 // (op 4282) 

s_and_b32 s18, s45, 31 // (op 4282) 
s_lshl_b32 s22, 1, s18 // (op 4282) 
s_add_i32 s23, s22, s19 // (op 4282) 


s_and_b32 s19, s0, s23 // (op 4282) 


s_add_i32 s0, s47, s19 // (op 4282) 


s_ashr_i32 s19, s0, s18 // (op 4282) 


s_lshl_b32 s0, s45, 24 // (op 4282) 
s_ashr_i32 s18, s0, 31 // (op 4282) 

s_xor_b32 s0, s19, s18 // (op 4282) 

s_sub_i32 s19, s0, s18 // (op 4282) 


s_add_u32 s22, s6, 128 // (op 4282) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4282) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 4282) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 4282) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 4282) 

s_mul_i32 s0, s11, s22 // (op 4282) 

s_mul_i32 s18, s8, s0 // (op 4282) 

s_mul_i32 s0, s19, s18 // (op 4282) 

s_sub_u32 s18, s16, s0 // (op 4282) 

s_mov_b32 s0, s18 // (op 4282) 
s_mul_hi_i32 s47, s0, s12 // (op 4282) 
s_add_i32 s22, s47, s0 // (op 4282) 


s_ashr_i32 s0, s22, 31 // (op 4282) 
s_cmp_eq_i32 s12, 0 // (op 4282) 
s_mov_b32 s47, scc // (op 4282) 
s_cmp_eq_i32 s47, 0 // (op 4282) 
s_cselect_b32 s23, 0, -1 // (op 4282) 

s_and_b32 s47, s44, 31 // (op 4282) 
s_lshl_b32 s24, 1, s47 // (op 4282) 
s_add_i32 s25, s24, s23 // (op 4282) 


s_and_b32 s23, s0, s25 // (op 4282) 


s_add_i32 s0, s22, s23 // (op 4282) 


s_ashr_i32 s22, s0, s47 // (op 4282) 


s_lshl_b32 s0, s44, 24 // (op 4282) 
s_ashr_i32 s47, s0, 31 // (op 4282) 

s_xor_b32 s0, s22, s47 // (op 4282) 

s_sub_i32 s22, s0, s47 // (op 4282) 


s_mul_i32 s0, s46, s11 // (op 4282) 
s_add_u32 s24, s20, 64 // (op 4282) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 4282) End of Int64 add, unlocking SCC
s_sub_u32 s60, s24, 1 // (op 4282) least significant half; sets scc
s_subb_u32 s61, s25, 0 // (op 4282) most significant half; uses scc

s_ashr_i64 s[24:25], s[60:61], 6 // (op 4282) 

s_sub_i32 s23, s24, s0 // (op 4282) 

s_mul_i32 s47, s22, s23 // (op 4282) 


s_sub_u32 s22, s18, s47 // (op 4282) 


s_add_u32 s18, s22, s0 // (op 4282) 


s_cmp_eq_i32 s19, 0 // (op 4282) 
s_mov_b32 s0, scc // (op 4282) 

s_cmp_eq_i32 s0, 0 // (op 4282) 
s_cselect_b32 s19, s18, s1 // (op 4282) 



v_lshl_add_u32 v15, s19, 6, v14 // (op 4282) 


v_mul_lo_u32 v21, v15, s27 // (op 4282) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 4282) most significant: high of low * low
v_mul_lo_u32 v20, v15, s26 // (op 4282) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 4282) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v16 // (op 4282) least significant half
v_addc_co_u32 v21, s[0:1], v21, v17, s[0:1] // (op 4282) most significant half



s_mul_i32 s1, s26, s21 // (op 4284) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 4284) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 4284) least significant
s_mul_hi_u32 s18, s26, s20 // (op 4284) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 4284) most significant: sum
s_add_u32 s1, s1, s18 // (op 4284) most significant: sum


s_mov_b32 s48, s28 // (op 4284) reg expression
s_mov_b32 s49, s29 // (op 4284) reg expression
s_mov_b32 s50, s0 // (op 4284) reg expression
s_mov_b32 s51, 131072 // (op 4284) call()

s_mul_hi_i32 s47, s16, s10 // (op 4327) 
v_lshrrev_b32 v14, 3, v1 // (op 4327) 
s_add_i32 s0, s47, s16 // (op 4327) 

v_and_b32 v15, -8, v1 // (op 4327) 
s_ashr_i32 s1, s0, 31 // (op 4327) 
v_sub_u32 v16, v1, v15 // (op 4327) 

s_cmp_eq_i32 s10, 0 // (op 4327) 
s_mov_b32 s47, scc // (op 4327) 
v_lshlrev_b32 v15, 4, v16 // (op 4327) 

s_cmp_eq_i32 s47, 0 // (op 4327) 
s_cselect_b32 s18, 0, -1 // (op 4327) 

v_mov_b32 v17, 0 // (op 4327) convert
v_mov_b32 v16, v15 // (op 4327) convert

s_and_b32 s19, s13, 31 // (op 4327) 
s_lshl_b32 s47, 1, s19 // (op 4327) 
s_add_i32 s22, s47, s18 // (op 4327) 


s_and_b32 s18, s1, s22 // (op 4327) 


s_add_i32 s1, s0, s18 // (op 4327) 


s_ashr_i32 s0, s1, s19 // (op 4327) 


s_lshl_b32 s1, s13, 24 // (op 4327) 
s_ashr_i32 s47, s1, 31 // (op 4327) 

s_xor_b32 s1, s0, s47 // (op 4327) 

s_sub_i32 s0, s1, s47 // (op 4327) 


s_mul_hi_i32 s1, s0, s14 // (op 4327) 
s_add_i32 s47, s1, s0 // (op 4327) 

s_ashr_i32 s1, s47, 31 // (op 4327) 
s_cmp_eq_i32 s14, 0 // (op 4327) 
s_mov_b32 s18, scc // (op 4327) 
s_cmp_eq_i32 s18, 0 // (op 4327) 
s_cselect_b32 s19, 0, -1 // (op 4327) 

s_and_b32 s18, s15, 31 // (op 4327) 
s_lshl_b32 s22, 1, s18 // (op 4327) 
s_add_i32 s23, s22, s19 // (op 4327) 


s_and_b32 s19, s1, s23 // (op 4327) 


s_add_i32 s1, s47, s19 // (op 4327) 


s_ashr_i32 s19, s1, s18 // (op 4327) 


s_lshl_b32 s1, s15, 24 // (op 4327) 
s_ashr_i32 s18, s1, 31 // (op 4327) 

s_xor_b32 s1, s19, s18 // (op 4327) 

s_sub_i32 s19, s1, s18 // (op 4327) 


s_add_u32 s22, s6, 128 // (op 4327) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4327) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 4327) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 4327) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 4327) 

s_mul_i32 s1, s19, s22 // (op 4327) 

s_sub_i32 s47, s0, s1 // (op 4327) 


s_mul_hi_i32 s0, s16, s9 // (op 4327) 
s_add_i32 s1, s0, s16 // (op 4327) 

s_ashr_i32 s0, s1, 31 // (op 4327) 
s_cmp_eq_i32 s9, 0 // (op 4327) 
s_mov_b32 s18, scc // (op 4327) 
s_cmp_eq_i32 s18, 0 // (op 4327) 
s_cselect_b32 s19, 0, -1 // (op 4327) 

s_and_b32 s18, s45, 31 // (op 4327) 
s_lshl_b32 s24, 1, s18 // (op 4327) 
s_add_i32 s25, s24, s19 // (op 4327) 


s_and_b32 s19, s0, s25 // (op 4327) 


s_add_i32 s0, s1, s19 // (op 4327) 


s_ashr_i32 s1, s0, s18 // (op 4327) 


s_lshl_b32 s0, s45, 24 // (op 4327) 
s_ashr_i32 s18, s0, 31 // (op 4327) 

s_xor_b32 s0, s1, s18 // (op 4327) 

s_sub_i32 s1, s0, s18 // (op 4327) 


s_mul_i32 s0, s11, s22 // (op 4327) 

s_mul_i32 s18, s8, s0 // (op 4327) 

s_mul_i32 s0, s1, s18 // (op 4327) 

s_sub_u32 s18, s16, s0 // (op 4327) 

s_mul_hi_i32 s0, s18, s12 // (op 4327) 
s_add_i32 s19, s0, s18 // (op 4327) 


s_ashr_i32 s0, s19, 31 // (op 4327) 
s_cmp_eq_i32 s12, 0 // (op 4327) 
s_mov_b32 s18, scc // (op 4327) 
s_cmp_eq_i32 s18, 0 // (op 4327) 
s_cselect_b32 s22, 0, -1 // (op 4327) 

s_and_b32 s18, s44, 31 // (op 4327) 
s_lshl_b32 s23, 1, s18 // (op 4327) 
s_add_i32 s24, s23, s22 // (op 4327) 


s_and_b32 s25, s0, s24 // (op 4327) 


s_add_i32 s0, s19, s25 // (op 4327) 


s_ashr_i32 s19, s0, s18 // (op 4327) 


s_lshl_b32 s0, s44, 24 // (op 4327) 
s_ashr_i32 s18, s0, 31 // (op 4327) 

s_xor_b32 s0, s19, s18 // (op 4327) 

s_sub_i32 s19, s0, s18 // (op 4327) 


s_cmp_eq_i32 s1, 0 // (op 4327) 
s_mov_b32 s0, scc // (op 4327) 

s_cmp_eq_i32 s0, 0 // (op 4327) 
s_cselect_b32 s1, s19, s47 // (op 4327) 



v_lshl_add_u32 v15, s1, 7, v14 // (op 4327) 


v_mul_lo_u32 v23, v15, s31 // (op 4327) most significant: low of low * high
v_mul_hi_u32 v14, v15, s30 // (op 4327) most significant: high of low * low
v_mul_lo_u32 v22, v15, s30 // (op 4327) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 4327) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v22 // (op 4327) least significant half
v_addc_co_u32 v15, s[0:1], v17, v23, s[0:1] // (op 4327) most significant half



s_mul_i32 s1, s30, s7 // (op 4329) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 4329) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 4329) least significant
s_mul_hi_u32 s18, s30, s6 // (op 4329) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 4329) most significant: sum
s_add_u32 s1, s1, s18 // (op 4329) most significant: sum


s_mov_b32 s60, s32 // (op 4329) reg expression
s_mov_b32 s61, s33 // (op 4329) reg expression
s_mov_b32 s62, s0 // (op 4329) reg expression
s_mov_b32 s63, 131072 // (op 4329) call()

s_mul_hi_i32 s47, s16, s10 // (op 4346) 
v_lshrrev_b32 v16, 3, v1 // (op 4346) 
s_add_i32 s0, s47, s16 // (op 4346) 

v_and_b32 v17, -8, v1 // (op 4346) 
s_ashr_i32 s1, s0, 31 // (op 4346) 
v_sub_u32 v22, v1, v17 // (op 4346) 

s_cmp_eq_i32 s10, 0 // (op 4346) 
s_mov_b32 s47, scc // (op 4346) 
v_lshlrev_b32 v17, 4, v22 // (op 4346) 

s_cmp_eq_i32 s47, 0 // (op 4346) 
s_cselect_b32 s18, 0, -1 // (op 4346) 

v_mov_b32 v23, 0 // (op 4346) convert
v_mov_b32 v22, v17 // (op 4346) convert

s_and_b32 s19, s13, 31 // (op 4346) 
s_lshl_b32 s47, 1, s19 // (op 4346) 
s_add_i32 s22, s47, s18 // (op 4346) 


s_and_b32 s18, s1, s22 // (op 4346) 


s_add_i32 s1, s0, s18 // (op 4346) 


s_ashr_i32 s0, s1, s19 // (op 4346) 


s_lshl_b32 s1, s13, 24 // (op 4346) 
s_ashr_i32 s47, s1, 31 // (op 4346) 

s_xor_b32 s1, s0, s47 // (op 4346) 

s_sub_i32 s0, s1, s47 // (op 4346) 


s_mul_hi_i32 s1, s0, s14 // (op 4346) 
s_add_i32 s47, s1, s0 // (op 4346) 

s_ashr_i32 s1, s47, 31 // (op 4346) 
s_cmp_eq_i32 s14, 0 // (op 4346) 
s_mov_b32 s18, scc // (op 4346) 
s_cmp_eq_i32 s18, 0 // (op 4346) 
s_cselect_b32 s19, 0, -1 // (op 4346) 

s_and_b32 s18, s15, 31 // (op 4346) 
s_lshl_b32 s22, 1, s18 // (op 4346) 
s_add_i32 s23, s22, s19 // (op 4346) 


s_and_b32 s19, s1, s23 // (op 4346) 


s_add_i32 s1, s47, s19 // (op 4346) 


s_ashr_i32 s19, s1, s18 // (op 4346) 


s_lshl_b32 s1, s15, 24 // (op 4346) 
s_ashr_i32 s18, s1, 31 // (op 4346) 

s_xor_b32 s1, s19, s18 // (op 4346) 

s_sub_i32 s19, s1, s18 // (op 4346) 


s_add_u32 s22, s6, 128 // (op 4346) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4346) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 4346) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 4346) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 4346) 

s_mul_i32 s1, s19, s22 // (op 4346) 

s_sub_i32 s47, s0, s1 // (op 4346) 


s_mul_hi_i32 s0, s16, s9 // (op 4346) 
s_add_i32 s1, s0, s16 // (op 4346) 

s_ashr_i32 s0, s1, 31 // (op 4346) 
s_cmp_eq_i32 s9, 0 // (op 4346) 
s_mov_b32 s18, scc // (op 4346) 
s_cmp_eq_i32 s18, 0 // (op 4346) 
s_cselect_b32 s19, 0, -1 // (op 4346) 

s_and_b32 s18, s45, 31 // (op 4346) 
s_lshl_b32 s24, 1, s18 // (op 4346) 
s_add_i32 s25, s24, s19 // (op 4346) 


s_and_b32 s19, s0, s25 // (op 4346) 


s_add_i32 s0, s1, s19 // (op 4346) 


s_ashr_i32 s1, s0, s18 // (op 4346) 


s_lshl_b32 s0, s45, 24 // (op 4346) 
s_ashr_i32 s18, s0, 31 // (op 4346) 

s_xor_b32 s0, s1, s18 // (op 4346) 

s_sub_i32 s1, s0, s18 // (op 4346) 


s_mul_i32 s0, s11, s22 // (op 4346) 

s_mul_i32 s18, s8, s0 // (op 4346) 

s_mul_i32 s0, s1, s18 // (op 4346) 

s_sub_u32 s18, s16, s0 // (op 4346) 

s_mul_hi_i32 s0, s18, s12 // (op 4346) 
s_add_i32 s19, s0, s18 // (op 4346) 


s_ashr_i32 s0, s19, 31 // (op 4346) 
s_cmp_eq_i32 s12, 0 // (op 4346) 
s_mov_b32 s18, scc // (op 4346) 
s_cmp_eq_i32 s18, 0 // (op 4346) 
s_cselect_b32 s22, 0, -1 // (op 4346) 

s_and_b32 s18, s44, 31 // (op 4346) 
s_lshl_b32 s23, 1, s18 // (op 4346) 
s_add_i32 s24, s23, s22 // (op 4346) 


s_and_b32 s25, s0, s24 // (op 4346) 


s_add_i32 s0, s19, s25 // (op 4346) 


s_ashr_i32 s19, s0, s18 // (op 4346) 


s_lshl_b32 s0, s44, 24 // (op 4346) 
s_ashr_i32 s18, s0, 31 // (op 4346) 

s_xor_b32 s0, s19, s18 // (op 4346) 

s_sub_i32 s19, s0, s18 // (op 4346) 


s_cmp_eq_i32 s1, 0 // (op 4346) 
s_mov_b32 s0, scc // (op 4346) 

s_cmp_eq_i32 s0, 0 // (op 4346) 
s_cselect_b32 s1, s19, s47 // (op 4346) 



v_lshl_add_u32 v17, s1, 7, v16 // (op 4346) 


v_mul_lo_u32 v25, v17, s31 // (op 4346) most significant: low of low * high
v_mul_hi_u32 v16, v17, s30 // (op 4346) most significant: high of low * low
v_mul_lo_u32 v24, v17, s30 // (op 4346) least significant: low of low * low
v_add_u32 v25, v25, v16 // (op 4346) most significant: sum


v_add_co_u32 v16, s[0:1], v22, v24 // (op 4346) least significant half
v_addc_co_u32 v17, s[0:1], v23, v25, s[0:1] // (op 4346) most significant half



s_mul_i32 s1, s30, s7 // (op 4348) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 4348) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 4348) least significant
s_mul_hi_u32 s18, s30, s6 // (op 4348) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 4348) most significant: sum
s_add_u32 s1, s1, s18 // (op 4348) most significant: sum


s_mov_b32 s60, s32 // (op 4348) reg expression
s_mov_b32 s61, s33 // (op 4348) reg expression
s_mov_b32 s62, s0 // (op 4348) reg expression
s_mov_b32 s63, 131072 // (op 4348) call()

s_mul_hi_i32 s47, s16, s10 // (op 4391) 
v_lshrrev_b32 v22, 2, v1 // (op 4391) 
s_add_i32 s0, s47, s16 // (op 4391) 

v_and_b32 v23, -4, v1 // (op 4391) 
s_ashr_i32 s1, s0, 31 // (op 4391) 
v_sub_u32 v24, v1, v23 // (op 4391) 

s_cmp_eq_i32 s10, 0 // (op 4391) 
s_mov_b32 s47, scc // (op 4391) 
s_cmp_eq_i32 s47, 0 // (op 4391) 
s_cselect_b32 s18, 0, -1 // (op 4391) 

s_and_b32 s19, s13, 31 // (op 4391) 
s_lshl_b32 s47, 1, s19 // (op 4391) 
s_add_i32 s22, s47, s18 // (op 4391) 


s_and_b32 s18, s1, s22 // (op 4391) 


s_add_i32 s1, s0, s18 // (op 4391) 


s_ashr_i32 s0, s1, s19 // (op 4391) 


s_lshl_b32 s1, s13, 24 // (op 4391) 
s_ashr_i32 s47, s1, 31 // (op 4391) 

s_xor_b32 s1, s0, s47 // (op 4391) 

s_sub_i32 s0, s1, s47 // (op 4391) 


s_mul_hi_i32 s1, s0, s14 // (op 4391) 
s_add_i32 s47, s1, s0 // (op 4391) 

s_ashr_i32 s1, s47, 31 // (op 4391) 
s_cmp_eq_i32 s14, 0 // (op 4391) 
s_mov_b32 s18, scc // (op 4391) 
s_cmp_eq_i32 s18, 0 // (op 4391) 
s_cselect_b32 s19, 0, -1 // (op 4391) 

s_and_b32 s18, s15, 31 // (op 4391) 
s_lshl_b32 s22, 1, s18 // (op 4391) 
s_add_i32 s23, s22, s19 // (op 4391) 


s_and_b32 s19, s1, s23 // (op 4391) 


s_add_i32 s1, s47, s19 // (op 4391) 


s_ashr_i32 s19, s1, s18 // (op 4391) 


s_lshl_b32 s1, s15, 24 // (op 4391) 
s_ashr_i32 s18, s1, 31 // (op 4391) 

s_xor_b32 s1, s19, s18 // (op 4391) 

s_sub_i32 s19, s1, s18 // (op 4391) 


s_mul_i32 s1, s0, s11 // (op 4391) 

s_sub_u32 s0, s16, s1 // (op 4391) 

s_mul_i32 s1, s19, s11 // (op 4391) 
s_add_u32 s1, s1, s0 // (op 4391) 


s_mul_hi_i32 s0, s16, s9 // (op 4391) 
s_add_i32 s47, s0, s16 // (op 4391) 

s_ashr_i32 s0, s47, 31 // (op 4391) 
s_cmp_eq_i32 s9, 0 // (op 4391) 
s_mov_b32 s18, scc // (op 4391) 
s_cmp_eq_i32 s18, 0 // (op 4391) 
s_cselect_b32 s19, 0, -1 // (op 4391) 

s_and_b32 s18, s45, 31 // (op 4391) 
s_lshl_b32 s22, 1, s18 // (op 4391) 
s_add_i32 s23, s22, s19 // (op 4391) 


s_and_b32 s19, s0, s23 // (op 4391) 


s_add_i32 s0, s47, s19 // (op 4391) 


s_ashr_i32 s19, s0, s18 // (op 4391) 


s_lshl_b32 s0, s45, 24 // (op 4391) 
s_ashr_i32 s18, s0, 31 // (op 4391) 

s_xor_b32 s0, s19, s18 // (op 4391) 

s_sub_i32 s19, s0, s18 // (op 4391) 


s_add_u32 s22, s6, 128 // (op 4391) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4391) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 4391) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 4391) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 4391) 

s_mul_i32 s0, s11, s22 // (op 4391) 

s_mul_i32 s18, s8, s0 // (op 4391) 

s_mul_i32 s0, s19, s18 // (op 4391) 

s_sub_u32 s18, s16, s0 // (op 4391) 

s_mov_b32 s0, s18 // (op 4391) 
s_mul_hi_i32 s47, s0, s12 // (op 4391) 
s_add_i32 s22, s47, s0 // (op 4391) 


s_ashr_i32 s0, s22, 31 // (op 4391) 
s_cmp_eq_i32 s12, 0 // (op 4391) 
s_mov_b32 s47, scc // (op 4391) 
s_cmp_eq_i32 s47, 0 // (op 4391) 
s_cselect_b32 s23, 0, -1 // (op 4391) 

s_and_b32 s47, s44, 31 // (op 4391) 
s_lshl_b32 s24, 1, s47 // (op 4391) 
s_add_i32 s25, s24, s23 // (op 4391) 


s_and_b32 s23, s0, s25 // (op 4391) 


s_add_i32 s0, s22, s23 // (op 4391) 


s_ashr_i32 s22, s0, s47 // (op 4391) 


s_lshl_b32 s0, s44, 24 // (op 4391) 
s_ashr_i32 s47, s0, 31 // (op 4391) 

s_xor_b32 s0, s22, s47 // (op 4391) 

s_sub_i32 s22, s0, s47 // (op 4391) 


s_mul_i32 s0, s46, s11 // (op 4391) 
s_add_u32 s24, s20, 64 // (op 4391) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 4391) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 4391) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 4391) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 6 // (op 4391) 

s_sub_i32 s23, s24, s0 // (op 4391) 

s_mul_i32 s47, s22, s23 // (op 4391) 


s_sub_u32 s22, s18, s47 // (op 4391) 


s_add_u32 s18, s22, s0 // (op 4391) 


s_cmp_eq_i32 s19, 0 // (op 4391) 
s_mov_b32 s0, scc // (op 4391) 

s_cmp_eq_i32 s0, 0 // (op 4391) 
s_cselect_b32 s19, s18, s1 // (op 4391) 



v_lshl_add_u32 v23, s19, 6, v22 // (op 4391) 


v_mul_lo_u32 v22, v23, s34 // (op 4391) 
v_add_u32 v22, v22, v24 // (op 4391) 


s_mul_i32 s1, s34, s21 // (op 4393) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 4393) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 4393) least significant
s_mul_hi_u32 s18, s34, s20 // (op 4393) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 4393) most significant: sum
s_add_u32 s1, s1, s18 // (op 4393) most significant: sum


s_mov_b32 s64, s36 // (op 4393) reg expression
s_mov_b32 s65, s37 // (op 4393) reg expression
s_mov_b32 s66, s0 // (op 4393) reg expression
s_mov_b32 s67, 131072 // (op 4393) call()

s_mul_hi_i32 s47, s16, s10 // (op 4410) 
v_lshrrev_b32 v23, 2, v1 // (op 4410) 
s_add_i32 s0, s47, s16 // (op 4410) 

v_and_b32 v24, -4, v1 // (op 4410) 
s_ashr_i32 s1, s0, 31 // (op 4410) 
v_sub_u32 v25, v1, v24 // (op 4410) 

s_cmp_eq_i32 s10, 0 // (op 4410) 
s_mov_b32 s47, scc // (op 4410) 
s_cmp_eq_i32 s47, 0 // (op 4410) 
s_cselect_b32 s18, 0, -1 // (op 4410) 

s_and_b32 s19, s13, 31 // (op 4410) 
s_lshl_b32 s47, 1, s19 // (op 4410) 
s_add_i32 s22, s47, s18 // (op 4410) 


s_and_b32 s18, s1, s22 // (op 4410) 


s_add_i32 s1, s0, s18 // (op 4410) 


s_ashr_i32 s0, s1, s19 // (op 4410) 


s_lshl_b32 s1, s13, 24 // (op 4410) 
s_ashr_i32 s47, s1, 31 // (op 4410) 

s_xor_b32 s1, s0, s47 // (op 4410) 

s_sub_i32 s0, s1, s47 // (op 4410) 


s_mul_hi_i32 s1, s0, s14 // (op 4410) 
s_add_i32 s47, s1, s0 // (op 4410) 

s_ashr_i32 s1, s47, 31 // (op 4410) 
s_cmp_eq_i32 s14, 0 // (op 4410) 
s_mov_b32 s18, scc // (op 4410) 
s_cmp_eq_i32 s18, 0 // (op 4410) 
s_cselect_b32 s19, 0, -1 // (op 4410) 

s_and_b32 s18, s15, 31 // (op 4410) 
s_lshl_b32 s22, 1, s18 // (op 4410) 
s_add_i32 s23, s22, s19 // (op 4410) 


s_and_b32 s19, s1, s23 // (op 4410) 


s_add_i32 s1, s47, s19 // (op 4410) 


s_ashr_i32 s19, s1, s18 // (op 4410) 


s_lshl_b32 s1, s15, 24 // (op 4410) 
s_ashr_i32 s18, s1, 31 // (op 4410) 

s_xor_b32 s1, s19, s18 // (op 4410) 

s_sub_i32 s19, s1, s18 // (op 4410) 


s_mul_i32 s1, s0, s11 // (op 4410) 

s_sub_u32 s0, s16, s1 // (op 4410) 

s_mul_i32 s1, s19, s11 // (op 4410) 
s_add_u32 s1, s1, s0 // (op 4410) 


s_mul_hi_i32 s0, s16, s9 // (op 4410) 
s_add_i32 s47, s0, s16 // (op 4410) 

s_ashr_i32 s0, s47, 31 // (op 4410) 
s_cmp_eq_i32 s9, 0 // (op 4410) 
s_mov_b32 s18, scc // (op 4410) 
s_cmp_eq_i32 s18, 0 // (op 4410) 
s_cselect_b32 s19, 0, -1 // (op 4410) 

s_and_b32 s18, s45, 31 // (op 4410) 
s_lshl_b32 s22, 1, s18 // (op 4410) 
s_add_i32 s23, s22, s19 // (op 4410) 


s_and_b32 s19, s0, s23 // (op 4410) 


s_add_i32 s0, s47, s19 // (op 4410) 


s_ashr_i32 s19, s0, s18 // (op 4410) 


s_lshl_b32 s0, s45, 24 // (op 4410) 
s_ashr_i32 s18, s0, 31 // (op 4410) 

s_xor_b32 s0, s19, s18 // (op 4410) 

s_sub_i32 s19, s0, s18 // (op 4410) 


s_add_u32 s22, s6, 128 // (op 4410) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4410) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 4410) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 4410) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 4410) 

s_mul_i32 s0, s11, s22 // (op 4410) 

s_mul_i32 s18, s8, s0 // (op 4410) 

s_mul_i32 s0, s19, s18 // (op 4410) 

s_sub_u32 s18, s16, s0 // (op 4410) 

s_mov_b32 s0, s18 // (op 4410) 
s_mul_hi_i32 s47, s0, s12 // (op 4410) 
s_add_i32 s22, s47, s0 // (op 4410) 


s_ashr_i32 s0, s22, 31 // (op 4410) 
s_cmp_eq_i32 s12, 0 // (op 4410) 
s_mov_b32 s47, scc // (op 4410) 
s_cmp_eq_i32 s47, 0 // (op 4410) 
s_cselect_b32 s23, 0, -1 // (op 4410) 

s_and_b32 s47, s44, 31 // (op 4410) 
s_lshl_b32 s24, 1, s47 // (op 4410) 
s_add_i32 s25, s24, s23 // (op 4410) 


s_and_b32 s23, s0, s25 // (op 4410) 


s_add_i32 s0, s22, s23 // (op 4410) 


s_ashr_i32 s22, s0, s47 // (op 4410) 


s_lshl_b32 s0, s44, 24 // (op 4410) 
s_ashr_i32 s47, s0, 31 // (op 4410) 

s_xor_b32 s0, s22, s47 // (op 4410) 

s_sub_i32 s22, s0, s47 // (op 4410) 


s_mul_i32 s0, s46, s11 // (op 4410) 
s_add_u32 s24, s20, 64 // (op 4410) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 4410) End of Int64 add, unlocking SCC
s_sub_u32 s68, s24, 1 // (op 4410) least significant half; sets scc
s_subb_u32 s69, s25, 0 // (op 4410) most significant half; uses scc

s_ashr_i64 s[24:25], s[68:69], 6 // (op 4410) 

s_sub_i32 s23, s24, s0 // (op 4410) 

s_mul_i32 s47, s22, s23 // (op 4410) 


s_sub_u32 s22, s18, s47 // (op 4410) 


s_add_u32 s18, s22, s0 // (op 4410) 


s_cmp_eq_i32 s19, 0 // (op 4410) 
s_mov_b32 s0, scc // (op 4410) 

s_cmp_eq_i32 s0, 0 // (op 4410) 
s_cselect_b32 s19, s18, s1 // (op 4410) 



v_lshl_add_u32 v24, s19, 6, v23 // (op 4410) 


v_mul_lo_u32 v23, v24, s34 // (op 4410) 
v_add_u32 v23, v23, v25 // (op 4410) 


s_mul_i32 s1, s34, s21 // (op 4412) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 4412) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 4412) least significant
s_mul_hi_u32 s18, s34, s20 // (op 4412) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 4412) most significant: sum
s_add_u32 s1, s1, s18 // (op 4412) most significant: sum


s_mov_b32 s64, s36 // (op 4412) reg expression
s_mov_b32 s65, s37 // (op 4412) reg expression
s_mov_b32 s66, s0 // (op 4412) reg expression
s_mov_b32 s67, 131072 // (op 4412) call()

s_mul_hi_i32 s47, s16, s10 // (op 4455) 
v_lshrrev_b32 v24, 2, v1 // (op 4455) 
s_add_i32 s0, s47, s16 // (op 4455) 

v_lshlrev_b32 v25, 1, v24 // (op 4455) 

s_ashr_i32 s1, s0, 31 // (op 4455) 
v_and_b32 v24, -4, v1 // (op 4455) 
s_cmp_eq_i32 s10, 0 // (op 4455) 
s_mov_b32 s47, scc // (op 4455) 
v_sub_u32 v26, v1, v24 // (op 4455) 

s_cmp_eq_i32 s47, 0 // (op 4455) 
s_cselect_b32 s18, 0, -1 // (op 4455) 

s_and_b32 s19, s13, 31 // (op 4455) 
s_lshl_b32 s47, 1, s19 // (op 4455) 
s_add_i32 s22, s47, s18 // (op 4455) 


s_and_b32 s18, s1, s22 // (op 4455) 


s_add_i32 s1, s0, s18 // (op 4455) 


s_ashr_i32 s0, s1, s19 // (op 4455) 


s_lshl_b32 s1, s13, 24 // (op 4455) 
s_ashr_i32 s47, s1, 31 // (op 4455) 

s_xor_b32 s1, s0, s47 // (op 4455) 

s_sub_i32 s0, s1, s47 // (op 4455) 


s_mul_hi_i32 s1, s0, s14 // (op 4455) 
s_add_i32 s47, s1, s0 // (op 4455) 

s_ashr_i32 s1, s47, 31 // (op 4455) 
s_cmp_eq_i32 s14, 0 // (op 4455) 
s_mov_b32 s18, scc // (op 4455) 
s_cmp_eq_i32 s18, 0 // (op 4455) 
s_cselect_b32 s19, 0, -1 // (op 4455) 

s_and_b32 s18, s15, 31 // (op 4455) 
s_lshl_b32 s22, 1, s18 // (op 4455) 
s_add_i32 s23, s22, s19 // (op 4455) 


s_and_b32 s19, s1, s23 // (op 4455) 


s_add_i32 s1, s47, s19 // (op 4455) 


s_ashr_i32 s19, s1, s18 // (op 4455) 


s_lshl_b32 s1, s15, 24 // (op 4455) 
s_ashr_i32 s18, s1, 31 // (op 4455) 

s_xor_b32 s1, s19, s18 // (op 4455) 

s_sub_i32 s19, s1, s18 // (op 4455) 


s_add_u32 s22, s6, 128 // (op 4455) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4455) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 4455) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 4455) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 4455) 

s_mul_i32 s1, s19, s22 // (op 4455) 

s_sub_i32 s47, s0, s1 // (op 4455) 


s_mul_hi_i32 s0, s16, s9 // (op 4455) 
s_add_i32 s1, s0, s16 // (op 4455) 

s_ashr_i32 s0, s1, 31 // (op 4455) 
s_cmp_eq_i32 s9, 0 // (op 4455) 
s_mov_b32 s18, scc // (op 4455) 
s_cmp_eq_i32 s18, 0 // (op 4455) 
s_cselect_b32 s19, 0, -1 // (op 4455) 

s_and_b32 s18, s45, 31 // (op 4455) 
s_lshl_b32 s24, 1, s18 // (op 4455) 
s_add_i32 s25, s24, s19 // (op 4455) 


s_and_b32 s19, s0, s25 // (op 4455) 


s_add_i32 s0, s1, s19 // (op 4455) 


s_ashr_i32 s1, s0, s18 // (op 4455) 


s_lshl_b32 s0, s45, 24 // (op 4455) 
s_ashr_i32 s18, s0, 31 // (op 4455) 

s_xor_b32 s0, s1, s18 // (op 4455) 

s_sub_i32 s1, s0, s18 // (op 4455) 


s_mul_i32 s0, s11, s22 // (op 4455) 

s_mul_i32 s18, s8, s0 // (op 4455) 

s_mul_i32 s0, s1, s18 // (op 4455) 

s_sub_u32 s18, s16, s0 // (op 4455) 

s_mul_hi_i32 s0, s18, s12 // (op 4455) 
s_add_i32 s19, s0, s18 // (op 4455) 


s_ashr_i32 s0, s19, 31 // (op 4455) 
s_cmp_eq_i32 s12, 0 // (op 4455) 
s_mov_b32 s18, scc // (op 4455) 
s_cmp_eq_i32 s18, 0 // (op 4455) 
s_cselect_b32 s22, 0, -1 // (op 4455) 

s_and_b32 s18, s44, 31 // (op 4455) 
s_lshl_b32 s23, 1, s18 // (op 4455) 
s_add_i32 s24, s23, s22 // (op 4455) 


s_and_b32 s25, s0, s24 // (op 4455) 


s_add_i32 s0, s19, s25 // (op 4455) 


s_ashr_i32 s19, s0, s18 // (op 4455) 


s_lshl_b32 s0, s44, 24 // (op 4455) 
s_ashr_i32 s18, s0, 31 // (op 4455) 

s_xor_b32 s0, s19, s18 // (op 4455) 

s_sub_i32 s19, s0, s18 // (op 4455) 


s_cmp_eq_i32 s1, 0 // (op 4455) 
s_mov_b32 s0, scc // (op 4455) 

s_cmp_eq_i32 s0, 0 // (op 4455) 
s_cselect_b32 s1, s19, s47 // (op 4455) 



v_lshl_add_u32 v24, s1, 7, v25 // (op 4455) 


v_mul_lo_u32 v25, v24, s38 // (op 4455) 

v_add_u32 v24, v26, v25 // (op 4455) 


s_mul_i32 s1, s38, s7 // (op 4457) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 4457) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 4457) least significant
s_mul_hi_u32 s18, s38, s6 // (op 4457) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 4457) most significant: sum
s_add_u32 s1, s1, s18 // (op 4457) most significant: sum


s_mov_b32 s68, s40 // (op 4457) reg expression
s_mov_b32 s69, s41 // (op 4457) reg expression
s_mov_b32 s70, s0 // (op 4457) reg expression
s_mov_b32 s71, 131072 // (op 4457) call()

s_mul_hi_i32 s47, s16, s10 // (op 4474) 
v_lshrrev_b32 v25, 2, v1 // (op 4474) 
s_add_i32 s0, s47, s16 // (op 4474) 

v_lshlrev_b32 v26, 1, v25 // (op 4474) 

s_ashr_i32 s1, s0, 31 // (op 4474) 
v_and_b32 v25, -4, v1 // (op 4474) 
s_cmp_eq_i32 s10, 0 // (op 4474) 
s_mov_b32 s47, scc // (op 4474) 
v_sub_u32 v27, v1, v25 // (op 4474) 

s_cmp_eq_i32 s47, 0 // (op 4474) 
s_cselect_b32 s18, 0, -1 // (op 4474) 

s_and_b32 s19, s13, 31 // (op 4474) 
s_lshl_b32 s47, 1, s19 // (op 4474) 
s_add_i32 s22, s47, s18 // (op 4474) 


s_and_b32 s18, s1, s22 // (op 4474) 


s_add_i32 s1, s0, s18 // (op 4474) 


s_ashr_i32 s0, s1, s19 // (op 4474) 


s_lshl_b32 s1, s13, 24 // (op 4474) 
s_ashr_i32 s47, s1, 31 // (op 4474) 

s_xor_b32 s1, s0, s47 // (op 4474) 

s_sub_i32 s0, s1, s47 // (op 4474) 


s_mul_hi_i32 s1, s0, s14 // (op 4474) 
s_add_i32 s47, s1, s0 // (op 4474) 

s_ashr_i32 s1, s47, 31 // (op 4474) 
s_cmp_eq_i32 s14, 0 // (op 4474) 
s_mov_b32 s18, scc // (op 4474) 
s_cmp_eq_i32 s18, 0 // (op 4474) 
s_cselect_b32 s19, 0, -1 // (op 4474) 

s_and_b32 s18, s15, 31 // (op 4474) 
s_lshl_b32 s22, 1, s18 // (op 4474) 
s_add_i32 s23, s22, s19 // (op 4474) 


s_and_b32 s19, s1, s23 // (op 4474) 


s_add_i32 s1, s47, s19 // (op 4474) 


s_ashr_i32 s19, s1, s18 // (op 4474) 


s_lshl_b32 s1, s15, 24 // (op 4474) 
s_ashr_i32 s18, s1, 31 // (op 4474) 

s_xor_b32 s1, s19, s18 // (op 4474) 

s_sub_i32 s19, s1, s18 // (op 4474) 


s_add_u32 s22, s6, 128 // (op 4474) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4474) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 4474) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 4474) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 4474) 

s_mul_i32 s1, s19, s22 // (op 4474) 

s_sub_i32 s47, s0, s1 // (op 4474) 


s_mul_hi_i32 s0, s16, s9 // (op 4474) 
s_add_i32 s1, s0, s16 // (op 4474) 

s_ashr_i32 s0, s1, 31 // (op 4474) 
s_cmp_eq_i32 s9, 0 // (op 4474) 
s_mov_b32 s18, scc // (op 4474) 
s_cmp_eq_i32 s18, 0 // (op 4474) 
s_cselect_b32 s19, 0, -1 // (op 4474) 

s_and_b32 s18, s45, 31 // (op 4474) 
s_lshl_b32 s24, 1, s18 // (op 4474) 
s_add_i32 s25, s24, s19 // (op 4474) 


s_and_b32 s19, s0, s25 // (op 4474) 


s_add_i32 s0, s1, s19 // (op 4474) 


s_ashr_i32 s1, s0, s18 // (op 4474) 


s_lshl_b32 s0, s45, 24 // (op 4474) 
s_ashr_i32 s18, s0, 31 // (op 4474) 

s_xor_b32 s0, s1, s18 // (op 4474) 

s_sub_i32 s1, s0, s18 // (op 4474) 


s_mul_i32 s0, s11, s22 // (op 4474) 

s_mul_i32 s18, s8, s0 // (op 4474) 

s_mul_i32 s0, s1, s18 // (op 4474) 

s_sub_u32 s18, s16, s0 // (op 4474) 

s_mul_hi_i32 s0, s18, s12 // (op 4474) 
s_add_i32 s19, s0, s18 // (op 4474) 


s_ashr_i32 s0, s19, 31 // (op 4474) 
s_cmp_eq_i32 s12, 0 // (op 4474) 
s_mov_b32 s18, scc // (op 4474) 
s_cmp_eq_i32 s18, 0 // (op 4474) 
s_cselect_b32 s22, 0, -1 // (op 4474) 

s_and_b32 s18, s44, 31 // (op 4474) 
s_lshl_b32 s23, 1, s18 // (op 4474) 
s_add_i32 s24, s23, s22 // (op 4474) 


s_and_b32 s25, s0, s24 // (op 4474) 


s_add_i32 s0, s19, s25 // (op 4474) 


s_ashr_i32 s19, s0, s18 // (op 4474) 


s_lshl_b32 s0, s44, 24 // (op 4474) 
s_ashr_i32 s18, s0, 31 // (op 4474) 

s_xor_b32 s0, s19, s18 // (op 4474) 

s_sub_i32 s19, s0, s18 // (op 4474) 


s_cmp_eq_i32 s1, 0 // (op 4474) 
s_mov_b32 s0, scc // (op 4474) 

s_cmp_eq_i32 s0, 0 // (op 4474) 
s_cselect_b32 s1, s19, s47 // (op 4474) 



v_lshl_add_u32 v25, s1, 7, v26 // (op 4474) 


v_mul_lo_u32 v26, v25, s38 // (op 4474) 

v_add_u32 v25, v27, v26 // (op 4474) 


s_mul_i32 s1, s38, s7 // (op 4476) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 4476) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 4476) least significant
s_mul_hi_u32 s18, s38, s6 // (op 4476) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 4476) most significant: sum
s_add_u32 s1, s1, s18 // (op 4476) most significant: sum


s_mov_b32 s68, s40 // (op 4476) reg expression
s_mov_b32 s69, s41 // (op 4476) reg expression
s_mov_b32 s70, s0 // (op 4476) reg expression
s_mov_b32 s71, 131072 // (op 4476) call()

v_and_b32 v26, -8, v1 // (op 4602) 
v_sub_u32 v27, v1, v26 // (op 4602) 

v_lshlrev_b32 v26, 4, v27 // (op 4602) 

v_lshrrev_b32 v27, 3, v1 // (op 4602) 
v_lshl_add_u32 v28, v27, 7, v26 // (op 4602) 


s_nop 0
v_readfirstlane_b32 s47, v28 // (op 4602) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 5 // (op 4205) 
v_mov_b64 v[26:27], s[0:1] // (op 4205) convert


v_mov_b64 v[28:29], v[14:15] // (op 4205) 
s_mov_b32 m0, s47 // (op 4205) reg expression
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 4205) Load value direct to lds
v_add_co_u32 v28, s[0:1], v28, v26 // (op 4205) least significant half
v_addc_co_u32 v29, s[0:1], 0, v29, s[0:1] // (op 4205) most significant half

s_add_u32 m0, m0, 4096 // (op 4205) 
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 4205) Load value direct to lds
v_add_co_u32 v28, s[0:1], v28, v26 // (op 4205) least significant half
v_addc_co_u32 v29, s[0:1], 0, v29, s[0:1] // (op 4205) most significant half

s_add_u32 m0, m0, 4096 // (op 4205) 
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 4205) Load value direct to lds
v_add_co_u32 v28, s[0:1], v28, v26 // (op 4205) least significant half
v_addc_co_u32 v29, s[0:1], 0, v29, s[0:1] // (op 4205) most significant half

s_add_u32 m0, m0, 4096 // (op 4205) 
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 4205) Load value direct to lds




v_mov_b64 v[26:27], s[38:39] // (op 3987) convert

v_mov_b32 v28, v24 // (op 3987) 
buffer_load_ubyte v30, v28, s[68:71], 0 offen offset: 0 // (op 3987) Load value
v_add_u32 v28, v28, v26 // (op 3987) 
buffer_load_ubyte v31, v28, s[68:71], 0 offen offset: 0 // (op 3987) Load value


v_and_b32 v26, -8, v1 // (op 4545) 
v_sub_u32 v27, v1, v26 // (op 4545) 

v_lshlrev_b32 v26, 4, v27 // (op 4545) 

v_lshrrev_b32 v27, 3, v1 // (op 4545) 
v_lshl_add_u32 v28, v27, 7, v26 // (op 4545) 


s_nop 0
v_readfirstlane_b32 s47, v28 // (op 4545) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 5 // (op 4210) 
v_mov_b64 v[26:27], s[0:1] // (op 4210) convert


v_mov_b64 v[28:29], v[18:19] // (op 4210) 
s_add_u32 s0, 16384, s47 // (op 4210) 
s_mov_b32 m0, s0 // (op 4210) reg expression
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 4210) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 4210) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 4210) most significant half

s_add_u32 m0, m0, 4096 // (op 4210) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 4210) Load value direct to lds







buffer_load_ubyte v26, v22, s[64:67], 0 offen offset: 0 // (op 3993) Load value
v_and_b32 v27, -8, v1 // (op 4829) 
v_sub_u32 v28, v1, v27 // (op 4829) 

v_lshlrev_b32 v27, 4, v28 // (op 4829) 

v_lshrrev_b32 v28, 3, v1 // (op 4829) 
v_lshl_add_u32 v29, v28, 7, v27 // (op 4829) 


s_nop 0
v_readfirstlane_b32 s47, v29 // (op 4829) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 5 // (op 4215) 
v_mov_b64 v[28:29], s[0:1] // (op 4215) convert


v_mov_b64 v[32:33], v[14:15] // (op 4215) 
s_add_u32 s0, 24576, s47 // (op 4215) 
s_mov_b32 m0, s0 // (op 4215) reg expression
s_mov_b32 s1, 128 // (op 4215) call()
buffer_load_dwordx4 v32, s[60:63], s1 offen offset: 0 lds // (op 4215) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v28 // (op 4215) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 4215) most significant half

s_add_u32 m0, m0, 4096 // (op 4215) 
buffer_load_dwordx4 v32, s[60:63], s1 offen offset: 0 lds // (op 4215) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v28 // (op 4215) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 4215) most significant half

s_add_u32 m0, m0, 4096 // (op 4215) 
buffer_load_dwordx4 v32, s[60:63], s1 offen offset: 0 lds // (op 4215) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v28 // (op 4215) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 4215) most significant half

s_add_u32 m0, m0, 4096 // (op 4215) 
buffer_load_dwordx4 v32, s[60:63], s1 offen offset: 0 lds // (op 4215) Load value direct to lds







v_mov_b64 v[14:15], s[38:39] // (op 3999) convert

v_mov_b32 v27, v24 // (op 3999) 
buffer_load_ubyte v28, v27, s[68:71], 0 offen offset:4 // (op 3999) Load value
v_add_u32 v27, v27, v14 // (op 3999) 
buffer_load_ubyte v29, v27, s[68:71], 0 offen offset:4 // (op 3999) Load value



v_and_b32 v24, -8, v1 // (op 4785) 
v_sub_u32 v27, v1, v24 // (op 4785) 

v_lshlrev_b32 v24, 4, v27 // (op 4785) 

v_lshrrev_b32 v27, 3, v1 // (op 4785) 
v_lshl_add_u32 v14, v27, 7, v24 // (op 4785) 


s_nop 0
v_readfirstlane_b32 s47, v14 // (op 4785) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 5 // (op 4217) 
v_mov_b64 v[14:15], s[0:1] // (op 4217) convert


v_mov_b64 v[32:33], v[18:19] // (op 4217) 
s_add_u32 s0, 40960, s47 // (op 4217) 
s_mov_b32 m0, s0 // (op 4217) reg expression
s_mov_b32 s1, 128 // (op 4217) call()
buffer_load_dwordx4 v32, s[48:51], s1 offen offset: 0 lds // (op 4217) Load value direct to lds
v_add_co_u32 v32, s[18:19], v32, v14 // (op 4217) least significant half
v_addc_co_u32 v33, s[18:19], 0, v33, s[18:19] // (op 4217) most significant half

s_add_u32 m0, m0, 4096 // (op 4217) 
buffer_load_dwordx4 v32, s[48:51], s1 offen offset: 0 lds // (op 4217) Load value direct to lds









buffer_load_ubyte v24, v22, s[64:67], 0 offen offset:4 // (op 4005) Load value

v_and_b32 v22, -4, v1 // (op 4718) 
v_sub_u32 v27, v1, v22 // (op 4718) 

v_lshrrev_b32 v22, 2, v1 // (op 4718) 
v_lshl_add_u32 v14, v22, 3, v27 // (op 4718) 





s_waitcnt vmcnt(13)
ds_write_b8 v14, v30 offset:49152 // (op 4008) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v14, v31 offset:49156 // (op 4008) Store local data 

v_and_b32 v22, -4, v1 // (op 4660) 
v_sub_u32 v27, v1, v22 // (op 4660) 

v_and_b32 v22, 4294967292, v1 // (op 4660) 
v_add_u32 v14, v22, v27 // (op 4660) 





s_waitcnt vmcnt(9)
ds_write_b8 v14, v26 offset:49664 // (op 4010) Store local data 

s_barrier  // (op 5005) 



ds_read_b128 v[32:35], v4 // (op 4035) Load local data 
ds_read_b128 v[36:39], v4 offset:64 // (op 4035) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 5040) 



ds_read_u8 v22, v5 offset:49152 // (op 4065) Load local data 



ds_read_b128 v[40:43], v4 offset:4096 // (op 4040) Load local data 
ds_read_b128 v[44:47], v4 offset:4160 // (op 4040) Load local data 



ds_read_u8 v27, v5 offset:49280 // (op 4070) Load local data 



ds_read_b128 v[48:51], v4 offset:8192 // (op 4045) Load local data 
ds_read_b128 v[52:55], v4 offset:8256 // (op 4045) Load local data 



ds_read_u8 v14, v5 offset:49408 // (op 4075) Load local data 



ds_read_b128 v[56:59], v4 offset:12288 // (op 4050) Load local data 
ds_read_b128 v[60:63], v4 offset:12352 // (op 4050) Load local data 



ds_read_u8 v15, v5 offset:49536 // (op 4080) Load local data 
s_barrier  // (op 4989) 



ds_read_b128 v[64:67], v3 offset:16384 // (op 4025) Load local data 
ds_read_b128 v[68:71], v3 offset:16448 // (op 4025) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 5022) 



ds_read_u8 v18, v0 offset:49664 // (op 4055) Load local data 



ds_read_b128 v[72:75], v3 offset:20480 // (op 4030) Load local data 
ds_read_b128 v[76:79], v3 offset:20544 // (op 4030) Load local data 



ds_read_u8 v19, v0 offset:49792 // (op 4060) Load local data 
v_and_b32 v80, -8, v1 // (op 4530) 
v_sub_u32 v81, v1, v80 // (op 4530) 

v_lshlrev_b32 v80, 4, v81 // (op 4530) 

v_lshrrev_b32 v81, 3, v1 // (op 4530) 
v_lshl_add_u32 v82, v81, 7, v80 // (op 4530) 


s_nop 0
v_readfirstlane_b32 s47, v82 // (op 4530) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v80, -8, v1 // (op 4587) 
v_sub_u32 v81, v1, v80 // (op 4587) 

v_lshlrev_b32 v80, 4, v81 // (op 4587) 

v_lshrrev_b32 v81, 3, v1 // (op 4587) 
v_lshl_add_u32 v82, v81, 7, v80 // (op 4587) 


s_nop 0
v_readfirstlane_b32 s0, v82 // (op 4587) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v80, -4, v1 // (op 4644) 
v_sub_u32 v81, v1, v80 // (op 4644) 

v_and_b32 v80, 4294967292, v1 // (op 4644) 
v_add_u32 v82, v80, v81 // (op 4644) 


v_and_b32 v80, -4, v1 // (op 4702) 
v_sub_u32 v81, v1, v80 // (op 4702) 

v_lshrrev_b32 v80, 2, v1 // (op 4702) 
v_lshl_add_u32 v83, v80, 3, v81 // (op 4702) 


v_and_b32 v80, -8, v1 // (op 4770) 
v_sub_u32 v81, v1, v80 // (op 4770) 

v_lshlrev_b32 v80, 4, v81 // (op 4770) 

v_lshrrev_b32 v81, 3, v1 // (op 4770) 
v_lshl_add_u32 v84, v81, 7, v80 // (op 4770) 


s_nop 0
v_readfirstlane_b32 s1, v84 // (op 4770) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v80, -8, v1 // (op 4814) 
v_sub_u32 v81, v1, v80 // (op 4814) 

v_lshlrev_b32 v80, 4, v81 // (op 4814) 

v_lshrrev_b32 v81, 3, v1 // (op 4814) 
v_lshl_add_u32 v84, v81, 7, v80 // (op 4814) 


s_nop 0
v_readfirstlane_b32 s18, v84 // (op 4814) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v80, -4, v1 // (op 4858) 
v_sub_u32 v81, v1, v80 // (op 4858) 

v_and_b32 v80, 4294967292, v1 // (op 4858) 
v_add_u32 v84, v80, v81 // (op 4858) 


v_and_b32 v80, -4, v1 // (op 4887) 
v_sub_u32 v81, v1, v80 // (op 4887) 

v_lshrrev_b32 v80, 2, v1 // (op 4887) 
v_lshl_add_u32 v85, v80, 3, v81 // (op 4887) 


s_mov_b32 s19, 0 // (op 1047) call()
s_ashr_i64 s[22:23], s[42:43], 7 // (op 1046) 
s_mov_b32 s24, s22 // (op 1046) convert

s_and_b32 s25, s24, 4294967294 // (op 1046) 

s_cmp_lt_u32 s19, s25 // (op 1046) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__ForLoopBottom_KLoop_1046 // (op 1046) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__ForLoopBottom_KLoop_1046 if false)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__ForLoopTop_KLoop_1046:
s_add_u32 s19, s19, 2 // (op 1048) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[64:71], v[32:39], a[0:3], v18, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 88) 
s_barrier  // (op 5008) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[64:71], v[40:47], a[4:7], v18, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 376) 
s_lshl_b64 s[22:23], s[30:31], 5 // (op 4177) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[64:71], v[48:55], a[8:11], v18, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 552) 
v_mov_b64 v[80:81], s[22:23] // (op 4177) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[64:71], v[56:63], a[12:15], v18, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 728) 
v_mov_b64 v[86:87], v[16:17] // (op 4177) 
v_mov_b64 v[88:89], v[86:87] // (op 4177) 
s_mov_b32 m0, s0 // (op 4177) reg expression
s_mov_b32 s22, 256 // (op 4177) call()
buffer_load_dwordx4 v88, s[60:63], s22 offen offset: 0 lds // (op 4177) Load value direct to lds
v_add_co_u32 v88, s[24:25], v88, v80 // (op 4177) least significant half
v_addc_co_u32 v89, s[24:25], 0, v89, s[24:25] // (op 4177) most significant half

s_add_u32 m0, m0, 4096 // (op 4177) 
buffer_load_dwordx4 v88, s[60:63], s22 offen offset: 0 lds // (op 4177) Load value direct to lds
v_add_co_u32 v88, s[24:25], v88, v80 // (op 4177) least significant half
v_addc_co_u32 v89, s[24:25], 0, v89, s[24:25] // (op 4177) most significant half

s_add_u32 m0, m0, 4096 // (op 4177) 
buffer_load_dwordx4 v88, s[60:63], s22 offen offset: 0 lds // (op 4177) Load value direct to lds
v_add_co_u32 v88, s[24:25], v88, v80 // (op 4177) least significant half
v_addc_co_u32 v89, s[24:25], 0, v89, s[24:25] // (op 4177) most significant half

s_add_u32 m0, m0, 4096 // (op 4177) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:79], v[32:39], a[16:19], v19, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1223) 
buffer_load_dwordx4 v88, s[60:63], s22 offen offset: 0 lds // (op 4177) Load value direct to lds




v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[72:79], v[40:47], a[20:23], v19, v27 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1177) 
v_mov_b64 v[80:81], s[38:39] // (op 25) convert

v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[72:79], v[48:55], a[24:27], v19, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1131) 
v_mov_b32 v88, v25 // (op 25) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[72:79], v[56:63], a[28:31], v19, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1085) 
v_mov_b32 v89, v88 // (op 25) 
buffer_load_ubyte v30, v89, s[68:71], 0 offen offset:8 // (op 25) Load value
v_add_u32 v89, v89, v80 // (op 25) 
buffer_load_ubyte v31, v89, s[68:71], 0 offen offset:8 // (op 25) Load value


s_barrier  // (op 4992) 

s_lshl_b64 s[22:23], s[26:27], 5 // (op 4172) 
v_mov_b64 v[80:81], s[22:23] // (op 4172) convert


v_mov_b64 v[90:91], v[20:21] // (op 4172) 
v_mov_b64 v[92:93], v[90:91] // (op 4172) 
s_add_u32 s22, 16384, s47 // (op 4172) 
s_mov_b32 m0, s22 // (op 4172) reg expression
s_mov_b32 s23, 256 // (op 4172) call()
buffer_load_dwordx4 v92, s[48:51], s23 offen offset: 0 lds // (op 4172) Load value direct to lds
v_add_co_u32 v92, s[24:25], v92, v80 // (op 4172) least significant half
v_addc_co_u32 v93, s[24:25], 0, v93, s[24:25] // (op 4172) most significant half

s_add_u32 m0, m0, 4096 // (op 4172) 
buffer_load_dwordx4 v92, s[48:51], s23 offen offset: 0 lds // (op 4172) Load value direct to lds






v_mov_b32 v89, v23 // (op 17) 

buffer_load_ubyte v26, v89, s[64:67], 0 offen offset:8 // (op 17) Load value
s_barrier  // (op 5080) 



s_waitcnt vmcnt(13)
ds_write_b8 v85, v28 offset:49920 // (op 255) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v85, v29 offset:49924 // (op 255) Store local data 
s_barrier  // (op 5074) 



s_waitcnt vmcnt(9)
ds_write_b8 v84, v24 offset:50432 // (op 254) Store local data 
s_barrier  // (op 5063) 



ds_read_b128 v[92:95], v8 offset:24576 // (op 247) Load local data 
ds_read_b128 v[96:99], v8 offset:24640 // (op 247) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 5077) 



ds_read_u8 v80, v9 offset:49920 // (op 251) Load local data 



ds_read_b128 v[100:103], v8 offset:28672 // (op 381) Load local data 
ds_read_b128 v[104:107], v8 offset:28736 // (op 381) Load local data 



ds_read_u8 v81, v9 offset:50048 // (op 385) Load local data 



ds_read_b128 v[108:111], v8 offset:32768 // (op 557) Load local data 
ds_read_b128 v[112:115], v8 offset:32832 // (op 557) Load local data 



ds_read_u8 v116, v9 offset:50176 // (op 561) Load local data 



ds_read_b128 v[118:121], v8 offset:36864 // (op 733) Load local data 
ds_read_b128 v[122:125], v8 offset:36928 // (op 733) Load local data 



ds_read_u8 v117, v9 offset:50304 // (op 737) Load local data 
s_barrier  // (op 5057) 
s_barrier  // (op 5121) 



ds_read_b128 v[64:67], v7 offset:40960 // (op 245) Load local data 
ds_read_b128 v[68:71], v7 offset:41024 // (op 245) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 5070) 



ds_read_u8 v18, v6 offset:50432 // (op 249) Load local data 



ds_read_b128 v[72:75], v7 offset:45056 // (op 1088) Load local data 
ds_read_b128 v[76:79], v7 offset:45120 // (op 1088) Load local data 



ds_read_u8 v19, v6 offset:50560 // (op 1091) Load local data 
s_barrier  // (op 5147) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[64:71], v[92:99], a[0:3], v18, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 252) 
s_barrier  // (op 5067) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[64:71], v[100:107], a[4:7], v18, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 377) 
s_lshl_b64 s[22:23], s[30:31], 5 // (op 4200) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[64:71], v[108:115], a[8:11], v18, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 553) 
v_mov_b64 v[126:127], s[22:23] // (op 4200) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[64:71], v[118:125], a[12:15], v18, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 729) 
v_mov_b64 v[128:129], v[86:87] // (op 4200) 
s_add_u32 s22, 24576, s18 // (op 4200) 
s_mov_b32 m0, s22 // (op 4200) reg expression
s_mov_b32 s23, 384 // (op 4200) call()
buffer_load_dwordx4 v128, s[60:63], s23 offen offset: 0 lds // (op 4200) Load value direct to lds
v_add_co_u32 v128, s[24:25], v128, v126 // (op 4200) least significant half
v_addc_co_u32 v129, s[24:25], 0, v129, s[24:25] // (op 4200) most significant half

s_add_u32 m0, m0, 4096 // (op 4200) 
buffer_load_dwordx4 v128, s[60:63], s23 offen offset: 0 lds // (op 4200) Load value direct to lds
v_add_co_u32 v128, s[24:25], v128, v126 // (op 4200) least significant half
v_addc_co_u32 v129, s[24:25], 0, v129, s[24:25] // (op 4200) most significant half

s_add_u32 m0, m0, 4096 // (op 4200) 
buffer_load_dwordx4 v128, s[60:63], s23 offen offset: 0 lds // (op 4200) Load value direct to lds
v_add_co_u32 v128, s[24:25], v128, v126 // (op 4200) least significant half
v_addc_co_u32 v129, s[24:25], 0, v129, s[24:25] // (op 4200) most significant half

s_add_u32 m0, m0, 4096 // (op 4200) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:79], v[92:99], a[16:19], v19, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1224) 
buffer_load_dwordx4 v128, s[60:63], s23 offen offset: 0 lds // (op 4200) Load value direct to lds








v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[72:79], v[100:107], a[20:23], v19, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1178) 
v_add_co_u32 v16, vcc, 256, v16 // (op 4338) least significant half


v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 4338) most significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[72:79], v[108:115], a[24:27], v19, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1132) 
v_mov_b64 v[80:81], s[38:39] // (op 258) convert



v_mov_b32 v86, v88 // (op 258) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[72:79], v[118:125], a[28:31], v19, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1086) 
buffer_load_ubyte v28, v86, s[68:71], 0 offen offset:12 // (op 258) Load value


v_add_u32 v86, v86, v80 // (op 258) 
buffer_load_ubyte v29, v86, s[68:71], 0 offen offset:12 // (op 258) Load value



s_barrier  // (op 5060) 
v_add_u32 v25, 8, v25 // (op 4466) 

s_lshl_b64 s[22:23], s[26:27], 5 // (op 4195) 
v_mov_b64 v[80:81], s[22:23] // (op 4195) convert


v_mov_b64 v[86:87], v[90:91] // (op 4195) 
s_add_u32 s22, 40960, s1 // (op 4195) 
s_mov_b32 m0, s22 // (op 4195) reg expression
s_mov_b32 s23, 384 // (op 4195) call()
buffer_load_dwordx4 v86, s[48:51], s23 offen offset: 0 lds // (op 4195) Load value direct to lds
v_add_co_u32 v86, s[24:25], v86, v80 // (op 4195) least significant half
v_addc_co_u32 v87, s[24:25], 0, v87, s[24:25] // (op 4195) most significant half

s_add_u32 m0, m0, 4096 // (op 4195) 
buffer_load_dwordx4 v86, s[48:51], s23 offen offset: 0 lds // (op 4195) Load value direct to lds








v_add_co_u32 v20, vcc, 256, v20 // (op 4274) least significant half
buffer_load_ubyte v24, v89, s[64:67], 0 offen offset:12 // (op 257) Load value
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 4274) most significant half

s_barrier  // (op 5043) 
v_add_u32 v23, 8, v23 // (op 4402) 



s_waitcnt vmcnt(13)
ds_write_b8 v83, v30 offset:49152 // (op 30) Store local data 
s_waitcnt vmcnt(12)
ds_write_b8 v83, v31 offset:49156 // (op 30) Store local data 
s_barrier  // (op 5025) 



s_waitcnt vmcnt(9)
ds_write_b8 v82, v26 offset:49664 // (op 21) Store local data 
s_barrier  // (op 5014) 



ds_read_b128 v[32:35], v4 // (op 75) Load local data 
ds_read_b128 v[36:39], v4 offset:64 // (op 75) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 5050) 



ds_read_u8 v22, v5 offset:49152 // (op 85) Load local data 



ds_read_b128 v[40:43], v4 offset:4096 // (op 371) Load local data 
ds_read_b128 v[44:47], v4 offset:4160 // (op 371) Load local data 



ds_read_u8 v27, v5 offset:49280 // (op 375) Load local data 



ds_read_b128 v[48:51], v4 offset:8192 // (op 547) Load local data 
ds_read_b128 v[52:55], v4 offset:8256 // (op 547) Load local data 



ds_read_u8 v14, v5 offset:49408 // (op 551) Load local data 



ds_read_b128 v[56:59], v4 offset:12288 // (op 723) Load local data 
ds_read_b128 v[60:63], v4 offset:12352 // (op 723) Load local data 



ds_read_u8 v15, v5 offset:49536 // (op 727) Load local data 
s_barrier  // (op 4998) 



ds_read_b128 v[64:67], v3 offset:16384 // (op 70) Load local data 
ds_read_b128 v[68:71], v3 offset:16448 // (op 70) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 5032) 



ds_read_u8 v18, v0 offset:49664 // (op 80) Load local data 



ds_read_b128 v[72:75], v3 offset:20480 // (op 1080) Load local data 
ds_read_b128 v[76:79], v3 offset:20544 // (op 1080) Load local data 



ds_read_u8 v19, v0 offset:49792 // (op 1083) Load local data 
s_ashr_i64 s[22:23], s[42:43], 7 // (op 1046) 
s_mov_b32 s24, s22 // (op 1046) convert

s_and_b32 s25, s24, 4294967294 // (op 1046) 

s_cmp_lt_u32 s19, s25 // (op 1046) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__ForLoopTop_KLoop_1046 // (op 1046) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__ForLoopTop_KLoop_1046 if true)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__ForLoopBottom_KLoop_1046:

















s_barrier  // (op 5137) 
s_mul_hi_i32 s47, s16, s10 // (op 4230) 
v_lshrrev_b32 v16, 3, v1 // (op 4230) 
s_add_i32 s0, s47, s16 // (op 4230) 

v_and_b32 v17, -8, v1 // (op 4230) 
s_ashr_i32 s1, s0, 31 // (op 4230) 
v_sub_u32 v20, v1, v17 // (op 4230) 

s_cmp_eq_i32 s10, 0 // (op 4230) 
s_mov_b32 s47, scc // (op 4230) 
v_lshlrev_b32 v17, 4, v20 // (op 4230) 

s_cmp_eq_i32 s47, 0 // (op 4230) 
s_cselect_b32 s18, 0, -1 // (op 4230) 

s_and_b32 s19, s13, 31 // (op 4230) 
s_lshl_b32 s47, 1, s19 // (op 4230) 
s_add_i32 s22, s47, s18 // (op 4230) 


s_and_b32 s18, s1, s22 // (op 4230) 


s_add_i32 s1, s0, s18 // (op 4230) 


s_ashr_i32 s0, s1, s19 // (op 4230) 


s_lshl_b32 s1, s13, 24 // (op 4230) 
s_ashr_i32 s47, s1, 31 // (op 4230) 

s_xor_b32 s1, s0, s47 // (op 4230) 

s_sub_i32 s0, s1, s47 // (op 4230) 


s_mul_hi_i32 s1, s0, s14 // (op 4230) 
s_add_i32 s47, s1, s0 // (op 4230) 

s_ashr_i32 s1, s47, 31 // (op 4230) 
s_cmp_eq_i32 s14, 0 // (op 4230) 
s_mov_b32 s18, scc // (op 4230) 
s_cmp_eq_i32 s18, 0 // (op 4230) 
s_cselect_b32 s19, 0, -1 // (op 4230) 

s_and_b32 s18, s15, 31 // (op 4230) 
s_lshl_b32 s22, 1, s18 // (op 4230) 
s_add_i32 s23, s22, s19 // (op 4230) 


s_and_b32 s19, s1, s23 // (op 4230) 


s_add_i32 s1, s47, s19 // (op 4230) 


s_ashr_i32 s19, s1, s18 // (op 4230) 


s_lshl_b32 s1, s15, 24 // (op 4230) 
s_ashr_i32 s18, s1, 31 // (op 4230) 

s_xor_b32 s1, s19, s18 // (op 4230) 

s_sub_i32 s19, s1, s18 // (op 4230) 


s_mul_i32 s1, s0, s11 // (op 4230) 

s_sub_u32 s0, s16, s1 // (op 4230) 

s_mul_i32 s1, s19, s11 // (op 4230) 
s_add_u32 s1, s1, s0 // (op 4230) 


s_mul_hi_i32 s0, s16, s9 // (op 4230) 
s_add_i32 s47, s0, s16 // (op 4230) 

s_ashr_i32 s0, s47, 31 // (op 4230) 
s_cmp_eq_i32 s9, 0 // (op 4230) 
s_mov_b32 s18, scc // (op 4230) 
s_cmp_eq_i32 s18, 0 // (op 4230) 
s_cselect_b32 s19, 0, -1 // (op 4230) 

s_and_b32 s18, s45, 31 // (op 4230) 
s_lshl_b32 s22, 1, s18 // (op 4230) 
s_add_i32 s23, s22, s19 // (op 4230) 


s_and_b32 s19, s0, s23 // (op 4230) 


s_add_i32 s0, s47, s19 // (op 4230) 


s_ashr_i32 s19, s0, s18 // (op 4230) 


s_lshl_b32 s0, s45, 24 // (op 4230) 
s_ashr_i32 s18, s0, 31 // (op 4230) 

s_xor_b32 s0, s19, s18 // (op 4230) 

s_sub_i32 s19, s0, s18 // (op 4230) 


s_add_u32 s22, s6, 128 // (op 4230) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4230) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 4230) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 4230) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 4230) 

s_mul_i32 s0, s11, s22 // (op 4230) 

s_mul_i32 s18, s8, s0 // (op 4230) 

s_mul_i32 s0, s19, s18 // (op 4230) 

s_sub_u32 s18, s16, s0 // (op 4230) 

s_mov_b32 s0, s18 // (op 4230) 
s_mul_hi_i32 s47, s0, s12 // (op 4230) 
s_add_i32 s22, s47, s0 // (op 4230) 


s_ashr_i32 s0, s22, 31 // (op 4230) 
s_cmp_eq_i32 s12, 0 // (op 4230) 
s_mov_b32 s47, scc // (op 4230) 
s_cmp_eq_i32 s47, 0 // (op 4230) 
s_cselect_b32 s23, 0, -1 // (op 4230) 

s_and_b32 s47, s44, 31 // (op 4230) 
s_lshl_b32 s24, 1, s47 // (op 4230) 
s_add_i32 s25, s24, s23 // (op 4230) 


s_and_b32 s23, s0, s25 // (op 4230) 


s_add_i32 s0, s22, s23 // (op 4230) 


s_ashr_i32 s22, s0, s47 // (op 4230) 


s_lshl_b32 s0, s44, 24 // (op 4230) 
s_ashr_i32 s47, s0, 31 // (op 4230) 

s_xor_b32 s0, s22, s47 // (op 4230) 

s_sub_i32 s22, s0, s47 // (op 4230) 


s_mul_i32 s0, s46, s11 // (op 4230) 
s_add_u32 s24, s20, 64 // (op 4230) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 4230) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 4230) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 4230) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 6 // (op 4230) 

s_sub_i32 s23, s24, s0 // (op 4230) 

s_mul_i32 s47, s22, s23 // (op 4230) 


s_sub_u32 s22, s18, s47 // (op 4230) 


s_add_u32 s18, s22, s0 // (op 4230) 


s_cmp_eq_i32 s19, 0 // (op 4230) 
s_mov_b32 s0, scc // (op 4230) 

s_cmp_eq_i32 s0, 0 // (op 4230) 
s_cselect_b32 s19, s18, s1 // (op 4230) 



v_lshl_add_u32 v20, s19, 6, v16 // (op 4230) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 4230) 
s_lshr_b32 s47, s0, 1 // (op 4230) 

v_lshl_add_u32 v16, s47, 8, v17 // (op 4230) 


v_mov_b32 v23, 0 // (op 4230) convert
v_mov_b32 v22, v16 // (op 4230) convert

v_mul_lo_u32 v17, v20, s27 // (op 4230) most significant: low of low * high
v_mul_hi_u32 v21, v20, s26 // (op 4230) most significant: high of low * low
v_mul_lo_u32 v16, v20, s26 // (op 4230) least significant: low of low * low
v_add_u32 v17, v17, v21 // (op 4230) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v22 // (op 4230) least significant half
v_addc_co_u32 v17, s[0:1], v17, v23, s[0:1] // (op 4230) most significant half



s_mul_i32 s1, s26, s21 // (op 4232) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 4232) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 4232) least significant
s_mul_hi_u32 s18, s26, s20 // (op 4232) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 4232) most significant: sum
s_add_u32 s1, s1, s18 // (op 4232) most significant: sum


s_mov_b32 s48, s28 // (op 4232) reg expression
s_mov_b32 s49, s29 // (op 4232) reg expression
s_mov_b32 s50, s0 // (op 4232) reg expression
s_mov_b32 s51, 131072 // (op 4232) call()

s_mul_hi_i32 s47, s16, s10 // (op 4304) 
v_lshrrev_b32 v20, 3, v1 // (op 4304) 
s_add_i32 s0, s47, s16 // (op 4304) 

v_and_b32 v21, -8, v1 // (op 4304) 
s_ashr_i32 s1, s0, 31 // (op 4304) 
v_sub_u32 v22, v1, v21 // (op 4304) 

s_cmp_eq_i32 s10, 0 // (op 4304) 
s_mov_b32 s47, scc // (op 4304) 
v_lshlrev_b32 v21, 4, v22 // (op 4304) 

s_cmp_eq_i32 s47, 0 // (op 4304) 
s_cselect_b32 s18, 0, -1 // (op 4304) 

s_and_b32 s19, s13, 31 // (op 4304) 
s_lshl_b32 s47, 1, s19 // (op 4304) 
s_add_i32 s22, s47, s18 // (op 4304) 


s_and_b32 s18, s1, s22 // (op 4304) 


s_add_i32 s1, s0, s18 // (op 4304) 


s_ashr_i32 s0, s1, s19 // (op 4304) 


s_lshl_b32 s1, s13, 24 // (op 4304) 
s_ashr_i32 s47, s1, 31 // (op 4304) 

s_xor_b32 s1, s0, s47 // (op 4304) 

s_sub_i32 s0, s1, s47 // (op 4304) 


s_mul_hi_i32 s1, s0, s14 // (op 4304) 
s_add_i32 s47, s1, s0 // (op 4304) 

s_ashr_i32 s1, s47, 31 // (op 4304) 
s_cmp_eq_i32 s14, 0 // (op 4304) 
s_mov_b32 s18, scc // (op 4304) 
s_cmp_eq_i32 s18, 0 // (op 4304) 
s_cselect_b32 s19, 0, -1 // (op 4304) 

s_and_b32 s18, s15, 31 // (op 4304) 
s_lshl_b32 s22, 1, s18 // (op 4304) 
s_add_i32 s23, s22, s19 // (op 4304) 


s_and_b32 s19, s1, s23 // (op 4304) 


s_add_i32 s1, s47, s19 // (op 4304) 


s_ashr_i32 s19, s1, s18 // (op 4304) 


s_lshl_b32 s1, s15, 24 // (op 4304) 
s_ashr_i32 s18, s1, 31 // (op 4304) 

s_xor_b32 s1, s19, s18 // (op 4304) 

s_sub_i32 s19, s1, s18 // (op 4304) 


s_add_u32 s22, s6, 128 // (op 4304) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4304) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 4304) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 4304) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 4304) 

s_mul_i32 s1, s19, s22 // (op 4304) 

s_sub_i32 s47, s0, s1 // (op 4304) 


s_mul_hi_i32 s0, s16, s9 // (op 4304) 
s_add_i32 s1, s0, s16 // (op 4304) 

s_ashr_i32 s0, s1, 31 // (op 4304) 
s_cmp_eq_i32 s9, 0 // (op 4304) 
s_mov_b32 s18, scc // (op 4304) 
s_cmp_eq_i32 s18, 0 // (op 4304) 
s_cselect_b32 s19, 0, -1 // (op 4304) 

s_and_b32 s18, s45, 31 // (op 4304) 
s_lshl_b32 s24, 1, s18 // (op 4304) 
s_add_i32 s25, s24, s19 // (op 4304) 


s_and_b32 s19, s0, s25 // (op 4304) 


s_add_i32 s0, s1, s19 // (op 4304) 


s_ashr_i32 s1, s0, s18 // (op 4304) 


s_lshl_b32 s0, s45, 24 // (op 4304) 
s_ashr_i32 s18, s0, 31 // (op 4304) 

s_xor_b32 s0, s1, s18 // (op 4304) 

s_sub_i32 s1, s0, s18 // (op 4304) 


s_mul_i32 s0, s11, s22 // (op 4304) 

s_mul_i32 s18, s8, s0 // (op 4304) 

s_mul_i32 s0, s1, s18 // (op 4304) 

s_sub_u32 s18, s16, s0 // (op 4304) 

s_mul_hi_i32 s0, s18, s12 // (op 4304) 
s_add_i32 s19, s0, s18 // (op 4304) 


s_ashr_i32 s0, s19, 31 // (op 4304) 
s_cmp_eq_i32 s12, 0 // (op 4304) 
s_mov_b32 s18, scc // (op 4304) 
s_cmp_eq_i32 s18, 0 // (op 4304) 
s_cselect_b32 s22, 0, -1 // (op 4304) 

s_and_b32 s18, s44, 31 // (op 4304) 
s_lshl_b32 s23, 1, s18 // (op 4304) 
s_add_i32 s24, s23, s22 // (op 4304) 


s_and_b32 s25, s0, s24 // (op 4304) 


s_add_i32 s0, s19, s25 // (op 4304) 


s_ashr_i32 s19, s0, s18 // (op 4304) 


s_lshl_b32 s0, s44, 24 // (op 4304) 
s_ashr_i32 s18, s0, 31 // (op 4304) 

s_xor_b32 s0, s19, s18 // (op 4304) 

s_sub_i32 s19, s0, s18 // (op 4304) 


s_cmp_eq_i32 s1, 0 // (op 4304) 
s_mov_b32 s0, scc // (op 4304) 

s_cmp_eq_i32 s0, 0 // (op 4304) 
s_cselect_b32 s1, s19, s47 // (op 4304) 



v_lshl_add_u32 v22, s1, 7, v20 // (op 4304) 


v_mul_lo_u32 v31, v22, s31 // (op 4304) most significant: low of low * high
v_mul_hi_u32 v20, v22, s30 // (op 4304) most significant: high of low * low
v_mul_lo_u32 v30, v22, s30 // (op 4304) least significant: low of low * low
v_add_u32 v31, v31, v20 // (op 4304) most significant: sum


s_ashr_i64 s[0:1], s[42:43], 7 // (op 4304) 
s_lshr_b32 s47, s0, 1 // (op 4304) 

v_lshl_add_u32 v20, s47, 8, v21 // (op 4304) 


v_mov_b32 v23, 0 // (op 4304) convert
v_mov_b32 v22, v20 // (op 4304) convert

v_add_co_u32 v20, s[0:1], v22, v30 // (op 4304) least significant half
v_addc_co_u32 v21, s[0:1], v23, v31, s[0:1] // (op 4304) most significant half



s_mul_i32 s1, s30, s7 // (op 4306) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 4306) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 4306) least significant
s_mul_hi_u32 s18, s30, s6 // (op 4306) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 4306) most significant: sum
s_add_u32 s1, s1, s18 // (op 4306) most significant: sum


s_mov_b32 s60, s32 // (op 4306) reg expression
s_mov_b32 s61, s33 // (op 4306) reg expression
s_mov_b32 s62, s0 // (op 4306) reg expression
s_mov_b32 s63, 131072 // (op 4306) call()

s_mul_hi_i32 s47, s16, s10 // (op 4368) 
v_lshrrev_b32 v22, 2, v1 // (op 4368) 
s_add_i32 s0, s47, s16 // (op 4368) 

v_and_b32 v23, -4, v1 // (op 4368) 
s_ashr_i32 s1, s0, 31 // (op 4368) 
v_sub_u32 v25, v1, v23 // (op 4368) 

s_cmp_eq_i32 s10, 0 // (op 4368) 
s_mov_b32 s47, scc // (op 4368) 
s_cmp_eq_i32 s47, 0 // (op 4368) 
s_cselect_b32 s18, 0, -1 // (op 4368) 

s_and_b32 s19, s13, 31 // (op 4368) 
s_lshl_b32 s47, 1, s19 // (op 4368) 
s_add_i32 s22, s47, s18 // (op 4368) 


s_and_b32 s18, s1, s22 // (op 4368) 


s_add_i32 s1, s0, s18 // (op 4368) 


s_ashr_i32 s0, s1, s19 // (op 4368) 


s_lshl_b32 s1, s13, 24 // (op 4368) 
s_ashr_i32 s47, s1, 31 // (op 4368) 

s_xor_b32 s1, s0, s47 // (op 4368) 

s_sub_i32 s0, s1, s47 // (op 4368) 


s_mul_hi_i32 s1, s0, s14 // (op 4368) 
s_add_i32 s47, s1, s0 // (op 4368) 

s_ashr_i32 s1, s47, 31 // (op 4368) 
s_cmp_eq_i32 s14, 0 // (op 4368) 
s_mov_b32 s18, scc // (op 4368) 
s_cmp_eq_i32 s18, 0 // (op 4368) 
s_cselect_b32 s19, 0, -1 // (op 4368) 

s_and_b32 s18, s15, 31 // (op 4368) 
s_lshl_b32 s22, 1, s18 // (op 4368) 
s_add_i32 s23, s22, s19 // (op 4368) 


s_and_b32 s19, s1, s23 // (op 4368) 


s_add_i32 s1, s47, s19 // (op 4368) 


s_ashr_i32 s19, s1, s18 // (op 4368) 


s_lshl_b32 s1, s15, 24 // (op 4368) 
s_ashr_i32 s18, s1, 31 // (op 4368) 

s_xor_b32 s1, s19, s18 // (op 4368) 

s_sub_i32 s19, s1, s18 // (op 4368) 


s_mul_i32 s1, s0, s11 // (op 4368) 

s_sub_u32 s0, s16, s1 // (op 4368) 

s_mul_i32 s1, s19, s11 // (op 4368) 
s_add_u32 s1, s1, s0 // (op 4368) 


s_mul_hi_i32 s0, s16, s9 // (op 4368) 
s_add_i32 s47, s0, s16 // (op 4368) 

s_ashr_i32 s0, s47, 31 // (op 4368) 
s_cmp_eq_i32 s9, 0 // (op 4368) 
s_mov_b32 s18, scc // (op 4368) 
s_cmp_eq_i32 s18, 0 // (op 4368) 
s_cselect_b32 s19, 0, -1 // (op 4368) 

s_and_b32 s18, s45, 31 // (op 4368) 
s_lshl_b32 s22, 1, s18 // (op 4368) 
s_add_i32 s23, s22, s19 // (op 4368) 


s_and_b32 s19, s0, s23 // (op 4368) 


s_add_i32 s0, s47, s19 // (op 4368) 


s_ashr_i32 s19, s0, s18 // (op 4368) 


s_lshl_b32 s0, s45, 24 // (op 4368) 
s_ashr_i32 s18, s0, 31 // (op 4368) 

s_xor_b32 s0, s19, s18 // (op 4368) 

s_sub_i32 s19, s0, s18 // (op 4368) 


s_add_u32 s22, s6, 128 // (op 4368) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4368) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 4368) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 4368) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 4368) 

s_mul_i32 s0, s11, s22 // (op 4368) 

s_mul_i32 s18, s8, s0 // (op 4368) 

s_mul_i32 s0, s19, s18 // (op 4368) 

s_sub_u32 s18, s16, s0 // (op 4368) 

s_mov_b32 s0, s18 // (op 4368) 
s_mul_hi_i32 s47, s0, s12 // (op 4368) 
s_add_i32 s22, s47, s0 // (op 4368) 


s_ashr_i32 s0, s22, 31 // (op 4368) 
s_cmp_eq_i32 s12, 0 // (op 4368) 
s_mov_b32 s47, scc // (op 4368) 
s_cmp_eq_i32 s47, 0 // (op 4368) 
s_cselect_b32 s23, 0, -1 // (op 4368) 

s_and_b32 s47, s44, 31 // (op 4368) 
s_lshl_b32 s24, 1, s47 // (op 4368) 
s_add_i32 s25, s24, s23 // (op 4368) 


s_and_b32 s23, s0, s25 // (op 4368) 


s_add_i32 s0, s22, s23 // (op 4368) 


s_ashr_i32 s22, s0, s47 // (op 4368) 


s_lshl_b32 s0, s44, 24 // (op 4368) 
s_ashr_i32 s47, s0, 31 // (op 4368) 

s_xor_b32 s0, s22, s47 // (op 4368) 

s_sub_i32 s22, s0, s47 // (op 4368) 


s_mul_i32 s0, s46, s11 // (op 4368) 
s_add_u32 s24, s20, 64 // (op 4368) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 4368) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 4368) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 4368) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 6 // (op 4368) 

s_sub_i32 s23, s24, s0 // (op 4368) 

s_mul_i32 s47, s22, s23 // (op 4368) 


s_sub_u32 s22, s18, s47 // (op 4368) 


s_add_u32 s18, s22, s0 // (op 4368) 


s_cmp_eq_i32 s19, 0 // (op 4368) 
s_mov_b32 s0, scc // (op 4368) 

s_cmp_eq_i32 s0, 0 // (op 4368) 
s_cselect_b32 s19, s18, s1 // (op 4368) 



v_lshl_add_u32 v23, s19, 6, v22 // (op 4368) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 4368) 
s_lshr_b32 s47, s0, 1 // (op 4368) 

v_lshl_add_u32 v22, s47, 3, v25 // (op 4368) 


v_mul_lo_u32 v25, v23, s34 // (op 4368) 
v_add_u32 v25, v25, v22 // (op 4368) 


s_mul_i32 s1, s34, s21 // (op 4370) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 4370) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 4370) least significant
s_mul_hi_u32 s18, s34, s20 // (op 4370) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 4370) most significant: sum
s_add_u32 s1, s1, s18 // (op 4370) most significant: sum


s_mov_b32 s64, s36 // (op 4370) reg expression
s_mov_b32 s65, s37 // (op 4370) reg expression
s_mov_b32 s66, s0 // (op 4370) reg expression
s_mov_b32 s67, 131072 // (op 4370) call()

s_mul_hi_i32 s47, s16, s10 // (op 4432) 
v_lshrrev_b32 v26, 2, v1 // (op 4432) 
s_add_i32 s0, s47, s16 // (op 4432) 

v_lshlrev_b32 v22, 1, v26 // (op 4432) 

s_ashr_i32 s1, s0, 31 // (op 4432) 
v_and_b32 v23, -4, v1 // (op 4432) 
s_cmp_eq_i32 s10, 0 // (op 4432) 
s_mov_b32 s47, scc // (op 4432) 
v_sub_u32 v26, v1, v23 // (op 4432) 

s_cmp_eq_i32 s47, 0 // (op 4432) 
s_cselect_b32 s18, 0, -1 // (op 4432) 

s_and_b32 s19, s13, 31 // (op 4432) 
s_lshl_b32 s47, 1, s19 // (op 4432) 
s_add_i32 s22, s47, s18 // (op 4432) 


s_and_b32 s18, s1, s22 // (op 4432) 


s_add_i32 s1, s0, s18 // (op 4432) 


s_ashr_i32 s0, s1, s19 // (op 4432) 


s_lshl_b32 s1, s13, 24 // (op 4432) 
s_ashr_i32 s47, s1, 31 // (op 4432) 

s_xor_b32 s1, s0, s47 // (op 4432) 

s_sub_i32 s0, s1, s47 // (op 4432) 


s_mul_hi_i32 s1, s0, s14 // (op 4432) 
s_add_i32 s47, s1, s0 // (op 4432) 

s_ashr_i32 s1, s47, 31 // (op 4432) 
s_cmp_eq_i32 s14, 0 // (op 4432) 
s_mov_b32 s18, scc // (op 4432) 
s_cmp_eq_i32 s18, 0 // (op 4432) 
s_cselect_b32 s19, 0, -1 // (op 4432) 

s_and_b32 s18, s15, 31 // (op 4432) 
s_lshl_b32 s22, 1, s18 // (op 4432) 
s_add_i32 s23, s22, s19 // (op 4432) 


s_and_b32 s19, s1, s23 // (op 4432) 


s_add_i32 s1, s47, s19 // (op 4432) 


s_ashr_i32 s19, s1, s18 // (op 4432) 


s_lshl_b32 s1, s15, 24 // (op 4432) 
s_ashr_i32 s18, s1, 31 // (op 4432) 

s_xor_b32 s1, s19, s18 // (op 4432) 

s_sub_i32 s19, s1, s18 // (op 4432) 


s_add_u32 s22, s6, 128 // (op 4432) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 4432) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 4432) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 4432) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 4432) 

s_mul_i32 s1, s19, s22 // (op 4432) 

s_sub_i32 s47, s0, s1 // (op 4432) 


s_mul_hi_i32 s0, s16, s9 // (op 4432) 
s_add_i32 s1, s0, s16 // (op 4432) 

s_ashr_i32 s0, s1, 31 // (op 4432) 
s_cmp_eq_i32 s9, 0 // (op 4432) 
s_mov_b32 s18, scc // (op 4432) 
s_cmp_eq_i32 s18, 0 // (op 4432) 
s_cselect_b32 s19, 0, -1 // (op 4432) 

s_and_b32 s18, s45, 31 // (op 4432) 
s_lshl_b32 s24, 1, s18 // (op 4432) 
s_add_i32 s25, s24, s19 // (op 4432) 


s_and_b32 s19, s0, s25 // (op 4432) 


s_add_i32 s0, s1, s19 // (op 4432) 


s_ashr_i32 s1, s0, s18 // (op 4432) 


s_lshl_b32 s0, s45, 24 // (op 4432) 
s_ashr_i32 s18, s0, 31 // (op 4432) 

s_xor_b32 s0, s1, s18 // (op 4432) 

s_sub_i32 s1, s0, s18 // (op 4432) 


s_mul_i32 s0, s11, s22 // (op 4432) 

s_mul_i32 s18, s8, s0 // (op 4432) 

s_mul_i32 s0, s1, s18 // (op 4432) 

s_sub_u32 s18, s16, s0 // (op 4432) 

s_mul_hi_i32 s0, s18, s12 // (op 4432) 
s_add_i32 s19, s0, s18 // (op 4432) 


s_ashr_i32 s0, s19, 31 // (op 4432) 
s_cmp_eq_i32 s12, 0 // (op 4432) 
s_mov_b32 s18, scc // (op 4432) 
s_cmp_eq_i32 s18, 0 // (op 4432) 
s_cselect_b32 s22, 0, -1 // (op 4432) 

s_and_b32 s18, s44, 31 // (op 4432) 
s_lshl_b32 s23, 1, s18 // (op 4432) 
s_add_i32 s24, s23, s22 // (op 4432) 


s_and_b32 s25, s0, s24 // (op 4432) 


s_add_i32 s0, s19, s25 // (op 4432) 


s_ashr_i32 s19, s0, s18 // (op 4432) 


s_lshl_b32 s0, s44, 24 // (op 4432) 
s_ashr_i32 s18, s0, 31 // (op 4432) 

s_xor_b32 s0, s19, s18 // (op 4432) 

s_sub_i32 s19, s0, s18 // (op 4432) 


s_cmp_eq_i32 s1, 0 // (op 4432) 
s_mov_b32 s0, scc // (op 4432) 

s_cmp_eq_i32 s0, 0 // (op 4432) 
s_cselect_b32 s1, s19, s47 // (op 4432) 



v_lshl_add_u32 v23, s1, 7, v22 // (op 4432) 


v_mul_lo_u32 v22, v23, s38 // (op 4432) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 4432) 
s_lshr_b32 s47, s0, 1 // (op 4432) 

v_lshl_add_u32 v23, s47, 3, v26 // (op 4432) 


v_add_u32 v26, v23, v22 // (op 4432) 


s_mul_i32 s1, s38, s7 // (op 4434) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 4434) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 4434) least significant
s_mul_hi_u32 s18, s38, s6 // (op 4434) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 4434) most significant: sum
s_add_u32 s1, s1, s18 // (op 4434) most significant: sum


s_mov_b32 s68, s40 // (op 4434) reg expression
s_mov_b32 s69, s41 // (op 4434) reg expression
s_mov_b32 s70, s0 // (op 4434) reg expression
s_mov_b32 s71, 131072 // (op 4434) call()

v_and_b32 v22, -8, v1 // (op 4515) 
v_sub_u32 v23, v1, v22 // (op 4515) 

v_lshlrev_b32 v22, 4, v23 // (op 4515) 

v_lshrrev_b32 v23, 3, v1 // (op 4515) 
v_lshl_add_u32 v30, v23, 7, v22 // (op 4515) 


s_nop 0
v_readfirstlane_b32 s47, v30 // (op 4515) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -8, v1 // (op 4574) 
v_sub_u32 v23, v1, v22 // (op 4574) 

v_lshlrev_b32 v22, 4, v23 // (op 4574) 

v_lshrrev_b32 v23, 3, v1 // (op 4574) 
v_lshl_add_u32 v30, v23, 7, v22 // (op 4574) 


s_nop 0
v_readfirstlane_b32 s0, v30 // (op 4574) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v22, -4, v1 // (op 4631) 
v_sub_u32 v23, v1, v22 // (op 4631) 

v_and_b32 v22, 4294967292, v1 // (op 4631) 
v_add_u32 v30, v22, v23 // (op 4631) 


v_and_b32 v22, -4, v1 // (op 4689) 
v_sub_u32 v23, v1, v22 // (op 4689) 

v_lshrrev_b32 v22, 2, v1 // (op 4689) 
v_lshl_add_u32 v31, v22, 3, v23 // (op 4689) 


s_ashr_i64 s[18:19], s[42:43], 7 // (op 1021) 
s_and_b32 s1, s18, 4294967294 // (op 1021) 

s_ashr_i64 s[18:19], s[42:43], 7 // (op 1020) 
s_mov_b32 s22, s18 // (op 1020) convert

s_cmp_lt_u32 s1, s22 // (op 1020) 













s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__ForLoopBottom_KLoopTail_1020 // (op 1020) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__ForLoopBottom_KLoopTail_1020 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__ForLoopTop_KLoopTail_1020:
s_barrier  // (op 4995) 
s_add_u32 s1, s1, 1 // (op 1022) 

s_lshl_b64 s[18:19], s[26:27], 5 // (op 4182) 
v_mov_b64 v[14:15], s[18:19] // (op 4182) convert


v_mov_b64 v[18:19], v[16:17] // (op 4182) 
v_mov_b64 v[28:29], v[18:19] // (op 4182) 
s_add_u32 s18, 16384, s47 // (op 4182) 
s_mov_b32 m0, s18 // (op 4182) reg expression
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 4182) Load value direct to lds
v_add_co_u32 v28, s[22:23], v28, v14 // (op 4182) least significant half
v_addc_co_u32 v29, s[22:23], 0, v29, s[22:23] // (op 4182) most significant half

s_add_u32 m0, m0, 4096 // (op 4182) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 4182) Load value direct to lds




v_add_co_u32 v16, vcc, 128, v16 // (op 4220) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 4220) most significant half
s_barrier  // (op 5011) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 4188) 
v_mov_b64 v[14:15], s[18:19] // (op 4188) convert


v_mov_b64 v[18:19], v[20:21] // (op 4188) 
v_mov_b64 v[28:29], v[18:19] // (op 4188) 
s_mov_b32 m0, s0 // (op 4188) reg expression
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 4188) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v14 // (op 4188) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 4188) most significant half

s_add_u32 m0, m0, 4096 // (op 4188) 
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 4188) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v14 // (op 4188) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 4188) most significant half

s_add_u32 m0, m0, 4096 // (op 4188) 
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 4188) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v14 // (op 4188) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 4188) most significant half

s_add_u32 m0, m0, 4096 // (op 4188) 
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 4188) Load value direct to lds



v_add_co_u32 v20, vcc, 128, v20 // (op 4296) least significant half
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 4296) most significant half


v_mov_b32 v14, v25 // (op 191) 

buffer_load_ubyte v15, v14, s[64:67], 0 offen offset: 0 // (op 191) Load value

v_add_u32 v25, 4, v25 // (op 4360) 

s_barrier  // (op 5028) 
v_mov_b64 v[18:19], s[38:39] // (op 192) convert




v_mov_b32 v14, v26 // (op 192) 
v_mov_b32 v22, v14 // (op 192) 
buffer_load_ubyte v28, v22, s[68:71], 0 offen offset: 0 // (op 192) Load value
v_add_u32 v22, v22, v18 // (op 192) 
buffer_load_ubyte v29, v22, s[68:71], 0 offen offset: 0 // (op 192) Load value



v_add_u32 v26, 4, v26 // (op 4424) 
s_waitcnt vmcnt(2)
ds_write_b8 v30, v15 offset:49664 // (op 188) Store local data 

s_barrier  // (op 5001) 



s_barrier  // (op 5046) 
ds_read_b128 v[32:35], v3 offset:16384 // (op 179) Load local data 



ds_read_b128 v[36:39], v3 offset:16448 // (op 179) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v31, v28 offset:49152 // (op 189) Store local data 
s_barrier  // (op 5018) 
s_waitcnt vmcnt(0)
ds_write_b8 v31, v29 offset:49156 // (op 189) Store local data 




ds_read_b128 v[40:43], v4 // (op 181) Load local data 
ds_read_b128 v[44:47], v4 offset:64 // (op 181) Load local data 
s_waitcnt lgkmcnt(6)
s_barrier  // (op 5036) 



ds_read_u8 v14, v0 offset:49664 // (op 183) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 5053) 



ds_read_u8 v15, v5 offset:49152 // (op 185) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[32:39], v[40:47], a[0:3], v14, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 186) 
ds_read_b128 v[48:51], v4 offset:4096 // (op 350) Load local data 
ds_read_b128 v[52:55], v4 offset:4160 // (op 350) Load local data 



ds_read_u8 v18, v5 offset:49280 // (op 354) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[32:39], v[48:55], a[4:7], v14, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 355) 
ds_read_b128 v[56:59], v4 offset:8192 // (op 526) Load local data 
ds_read_b128 v[60:63], v4 offset:8256 // (op 526) Load local data 



ds_read_u8 v19, v5 offset:49408 // (op 530) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[32:39], v[56:63], a[8:11], v14, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 531) 
ds_read_b128 v[64:67], v4 offset:12288 // (op 702) Load local data 
ds_read_b128 v[68:71], v4 offset:12352 // (op 702) Load local data 



ds_read_u8 v22, v5 offset:49536 // (op 706) Load local data 



s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[32:39], v[64:71], a[12:15], v14, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 707) 
ds_read_b128 v[72:75], v3 offset:20480 // (op 964) Load local data 


ds_read_b128 v[76:79], v3 offset:20544 // (op 964) Load local data 



ds_read_u8 v14, v0 offset:49792 // (op 967) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[72:79], v[40:47], a[16:19], v14, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 969) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[72:79], v[48:55], a[20:23], v14, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1000) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[72:79], v[56:63], a[24:27], v14, v19 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1031) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[72:79], v[64:71], a[28:31], v14, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1062) 




s_ashr_i64 s[18:19], s[42:43], 7 // (op 1020) 
s_mov_b32 s22, s18 // (op 1020) convert

s_cmp_lt_u32 s1, s22 // (op 1020) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__ForLoopTop_KLoopTail_1020 // (op 1020) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__ForLoopTop_KLoopTail_1020 if true)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__ForLoopBottom_KLoopTail_1020:






















s_barrier  // (op 5129) 





















v_cmp_eq_f32 vcc, 0.00000, v10 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__ConditionalFalse_0_beta_133
v_accvgpr_read v6, a0 // (op 50) 
v_accvgpr_read v7, a1 // (op 50) 
v_accvgpr_read v8, a2 // (op 50) 
v_accvgpr_read v9, a3 // (op 50) 
v_mul_f32 v14, v11, v6 // (op 50) 
v_mul_f32 v15, v11, v7 // (op 50) 
v_mul_f32 v16, v11, v8 // (op 50) 
v_mul_f32 v17, v11, v9 // (op 50) 



v_mov_b32 v0, v13 // (op 56) reg expression

buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 56) Store value


v_accvgpr_read v6, a4 // (op 1869) 
v_accvgpr_read v7, a5 // (op 1869) 
v_accvgpr_read v8, a6 // (op 1869) 
v_accvgpr_read v9, a7 // (op 1869) 
v_mul_f32 v14, v11, v6 // (op 1869) 
v_mul_f32 v15, v11, v7 // (op 1869) 
v_mul_f32 v16, v11, v8 // (op 1869) 
v_mul_f32 v17, v11, v9 // (op 1869) 



s_lshl_b32 s0, s4, 7 // (op 1871) 
v_add_u32 v0, v13, s0 // (op 1871) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 1871) Store value


v_accvgpr_read v6, a8 // (op 1881) 
v_accvgpr_read v7, a9 // (op 1881) 
v_accvgpr_read v8, a10 // (op 1881) 
v_accvgpr_read v9, a11 // (op 1881) 
v_mul_f32 v14, v11, v6 // (op 1881) 
v_mul_f32 v15, v11, v7 // (op 1881) 
v_mul_f32 v16, v11, v8 // (op 1881) 
v_mul_f32 v17, v11, v9 // (op 1881) 



s_lshl_b32 s0, s4, 8 // (op 1883) 
v_add_u32 v0, v13, s0 // (op 1883) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 1883) Store value


v_accvgpr_read v6, a12 // (op 1893) 
v_accvgpr_read v7, a13 // (op 1893) 
v_accvgpr_read v8, a14 // (op 1893) 
v_accvgpr_read v9, a15 // (op 1893) 
v_mul_f32 v14, v11, v6 // (op 1893) 
v_mul_f32 v15, v11, v7 // (op 1893) 
v_mul_f32 v16, v11, v8 // (op 1893) 
v_mul_f32 v17, v11, v9 // (op 1893) 



s_lshl_b32 s0, s4, 7 // (op 1895) 
s_mul_i32 s1, 3, s0 // (op 1895) 

v_add_u32 v0, v13, s1 // (op 1895) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 1895) Store value


v_accvgpr_read v6, a16 // (op 1927) 
v_accvgpr_read v7, a17 // (op 1927) 
v_accvgpr_read v8, a18 // (op 1927) 
v_accvgpr_read v9, a19 // (op 1927) 
v_mul_f32 v14, v11, v6 // (op 1927) 
v_mul_f32 v15, v11, v7 // (op 1927) 
v_mul_f32 v16, v11, v8 // (op 1927) 
v_mul_f32 v17, v11, v9 // (op 1927) 



v_add_u32 v0, 128, v13 // (op 1930) 

buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 1930) Store value


v_accvgpr_read v6, a20 // (op 1940) 
v_accvgpr_read v7, a21 // (op 1940) 
v_accvgpr_read v8, a22 // (op 1940) 
v_accvgpr_read v9, a23 // (op 1940) 
v_mul_f32 v14, v11, v6 // (op 1940) 
v_mul_f32 v15, v11, v7 // (op 1940) 
v_mul_f32 v16, v11, v8 // (op 1940) 
v_mul_f32 v17, v11, v9 // (op 1940) 



s_lshl_b32 s0, s4, 7 // (op 1932) 
s_add_u32 s0, s0, 128 // (op 1932) 
v_add_u32 v0, v13, s0 // (op 1932) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 1932) Store value


v_accvgpr_read v6, a24 // (op 1945) 
v_accvgpr_read v7, a25 // (op 1945) 
v_accvgpr_read v8, a26 // (op 1945) 
v_accvgpr_read v9, a27 // (op 1945) 
v_mul_f32 v14, v11, v6 // (op 1945) 
v_mul_f32 v15, v11, v7 // (op 1945) 
v_mul_f32 v16, v11, v8 // (op 1945) 
v_mul_f32 v17, v11, v9 // (op 1945) 



s_lshl_b32 s0, s4, 8 // (op 1934) 
s_add_u32 s0, s0, 128 // (op 1934) 
v_add_u32 v0, v13, s0 // (op 1934) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 1934) Store value


v_accvgpr_read v6, a28 // (op 1950) 
v_accvgpr_read v7, a29 // (op 1950) 
v_accvgpr_read v8, a30 // (op 1950) 
v_accvgpr_read v9, a31 // (op 1950) 
v_mul_f32 v14, v11, v6 // (op 1950) 
v_mul_f32 v15, v11, v7 // (op 1950) 
v_mul_f32 v16, v11, v8 // (op 1950) 
v_mul_f32 v17, v11, v9 // (op 1950) 



s_lshl_b32 s0, s4, 7 // (op 1936) 
s_mul_i32 s1, 3, s0 // (op 1936) 
s_add_u32 s1, s1, 128 // (op 1936) 

v_add_u32 v0, v13, s1 // (op 1936) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 1936) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__ConditionalBottom_0_beta_133
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__ConditionalFalse_0_beta_133:


v_add_u32 v0, 128, v12 // (op 2108) 

buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 2108) Load value

v_accvgpr_read v14, a16 // (op 2096) 
v_accvgpr_read v15, a17 // (op 2096) 
v_accvgpr_read v16, a18 // (op 2096) 
v_accvgpr_read v17, a19 // (op 2096) 
v_mul_f32 v18, v11, v14 // (op 2096) 
v_mul_f32 v19, v11, v15 // (op 2096) 
v_mul_f32 v20, v11, v16 // (op 2096) 
v_mul_f32 v21, v11, v17 // (op 2096) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2107) 
v_mul_f32 v15, v10, v7 // (op 2107) 
v_mul_f32 v16, v10, v8 // (op 2107) 
v_mul_f32 v17, v10, v9 // (op 2107) 

v_add_f32 v6, v14, v18 // (op 2097) 
v_add_f32 v7, v15, v19 // (op 2097) 
v_add_f32 v8, v16, v20 // (op 2097) 
v_add_f32 v9, v17, v21 // (op 2097) 




v_add_u32 v0, 128, v13 // (op 2099) 

buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 2099) Store value




s_lshl_b32 s0, s2, 7 // (op 2112) 
s_add_u32 s0, s0, 128 // (op 2112) 
v_add_u32 v0, v12, s0 // (op 2112) 


buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 2112) Load value

v_accvgpr_read v14, a20 // (op 2122) 
v_accvgpr_read v15, a21 // (op 2122) 
v_accvgpr_read v16, a22 // (op 2122) 
v_accvgpr_read v17, a23 // (op 2122) 
v_mul_f32 v18, v11, v14 // (op 2122) 
v_mul_f32 v19, v11, v15 // (op 2122) 
v_mul_f32 v20, v11, v16 // (op 2122) 
v_mul_f32 v21, v11, v17 // (op 2122) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2110) 
v_mul_f32 v15, v10, v7 // (op 2110) 
v_mul_f32 v16, v10, v8 // (op 2110) 
v_mul_f32 v17, v10, v9 // (op 2110) 

v_add_f32 v6, v14, v18 // (op 2111) 
v_add_f32 v7, v15, v19 // (op 2111) 
v_add_f32 v8, v16, v20 // (op 2111) 
v_add_f32 v9, v17, v21 // (op 2111) 




s_lshl_b32 s0, s4, 7 // (op 2101) 
s_add_u32 s0, s0, 128 // (op 2101) 
v_add_u32 v0, v13, s0 // (op 2101) 


buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 2101) Store value




s_lshl_b32 s0, s2, 8 // (op 2116) 
s_add_u32 s0, s0, 128 // (op 2116) 
v_add_u32 v0, v12, s0 // (op 2116) 


buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 2116) Load value

v_accvgpr_read v14, a24 // (op 2124) 
v_accvgpr_read v15, a25 // (op 2124) 
v_accvgpr_read v16, a26 // (op 2124) 
v_accvgpr_read v17, a27 // (op 2124) 
v_mul_f32 v18, v11, v14 // (op 2124) 
v_mul_f32 v19, v11, v15 // (op 2124) 
v_mul_f32 v20, v11, v16 // (op 2124) 
v_mul_f32 v21, v11, v17 // (op 2124) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2114) 
v_mul_f32 v15, v10, v7 // (op 2114) 
v_mul_f32 v16, v10, v8 // (op 2114) 
v_mul_f32 v17, v10, v9 // (op 2114) 

v_add_f32 v6, v14, v18 // (op 2115) 
v_add_f32 v7, v15, v19 // (op 2115) 
v_add_f32 v8, v16, v20 // (op 2115) 
v_add_f32 v9, v17, v21 // (op 2115) 




s_lshl_b32 s0, s4, 8 // (op 2103) 
s_add_u32 s0, s0, 128 // (op 2103) 
v_add_u32 v0, v13, s0 // (op 2103) 


buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 2103) Store value




s_lshl_b32 s0, s2, 7 // (op 2120) 
s_mul_i32 s1, 3, s0 // (op 2120) 
s_add_u32 s1, s1, 128 // (op 2120) 

v_add_u32 v0, v12, s1 // (op 2120) 


buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 2120) Load value

v_accvgpr_read v14, a28 // (op 2126) 
v_accvgpr_read v15, a29 // (op 2126) 
v_accvgpr_read v16, a30 // (op 2126) 
v_accvgpr_read v17, a31 // (op 2126) 
v_mul_f32 v18, v11, v14 // (op 2126) 
v_mul_f32 v19, v11, v15 // (op 2126) 
v_mul_f32 v20, v11, v16 // (op 2126) 
v_mul_f32 v21, v11, v17 // (op 2126) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2118) 
v_mul_f32 v15, v10, v7 // (op 2118) 
v_mul_f32 v16, v10, v8 // (op 2118) 
v_mul_f32 v17, v10, v9 // (op 2118) 

v_add_f32 v6, v14, v18 // (op 2119) 
v_add_f32 v7, v15, v19 // (op 2119) 
v_add_f32 v8, v16, v20 // (op 2119) 
v_add_f32 v9, v17, v21 // (op 2119) 




s_lshl_b32 s0, s4, 7 // (op 2105) 
s_mul_i32 s1, 3, s0 // (op 2105) 
s_add_u32 s1, s1, 128 // (op 2105) 

v_add_u32 v0, v13, s1 // (op 2105) 


buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 2105) Store value




v_mov_b32 v0, v12 // (op 145) reg expression

s_nop 0
buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 145) Load value

v_accvgpr_read v14, a0 // (op 142) 
v_accvgpr_read v15, a1 // (op 142) 
v_accvgpr_read v16, a2 // (op 142) 
v_accvgpr_read v17, a3 // (op 142) 
v_mul_f32 v18, v11, v14 // (op 142) 
v_mul_f32 v19, v11, v15 // (op 142) 
v_mul_f32 v20, v11, v16 // (op 142) 
v_mul_f32 v21, v11, v17 // (op 142) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 146) 
v_mul_f32 v15, v10, v7 // (op 146) 
v_mul_f32 v16, v10, v8 // (op 146) 
v_mul_f32 v17, v10, v9 // (op 146) 

v_add_f32 v6, v14, v18 // (op 143) 
v_add_f32 v7, v15, v19 // (op 143) 
v_add_f32 v8, v16, v20 // (op 143) 
v_add_f32 v9, v17, v21 // (op 143) 




v_mov_b32 v0, v13 // (op 144) reg expression

buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 144) Store value




s_lshl_b32 s0, s2, 7 // (op 2022) 
v_add_u32 v0, v12, s0 // (op 2022) 


buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 2022) Load value

v_accvgpr_read v14, a4 // (op 2019) 
v_accvgpr_read v15, a5 // (op 2019) 
v_accvgpr_read v16, a6 // (op 2019) 
v_accvgpr_read v17, a7 // (op 2019) 
v_mul_f32 v18, v11, v14 // (op 2019) 
v_mul_f32 v19, v11, v15 // (op 2019) 
v_mul_f32 v20, v11, v16 // (op 2019) 
v_mul_f32 v21, v11, v17 // (op 2019) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2023) 
v_mul_f32 v15, v10, v7 // (op 2023) 
v_mul_f32 v16, v10, v8 // (op 2023) 
v_mul_f32 v17, v10, v9 // (op 2023) 

v_add_f32 v6, v14, v18 // (op 2020) 
v_add_f32 v7, v15, v19 // (op 2020) 
v_add_f32 v8, v16, v20 // (op 2020) 
v_add_f32 v9, v17, v21 // (op 2020) 




s_lshl_b32 s0, s4, 7 // (op 2021) 
v_add_u32 v0, v13, s0 // (op 2021) 


buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 2021) Store value




s_lshl_b32 s0, s2, 8 // (op 2034) 
v_add_u32 v0, v12, s0 // (op 2034) 


buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 2034) Load value

v_accvgpr_read v14, a8 // (op 2031) 
v_accvgpr_read v15, a9 // (op 2031) 
v_accvgpr_read v16, a10 // (op 2031) 
v_accvgpr_read v17, a11 // (op 2031) 
v_mul_f32 v18, v11, v14 // (op 2031) 
v_mul_f32 v19, v11, v15 // (op 2031) 
v_mul_f32 v20, v11, v16 // (op 2031) 
v_mul_f32 v21, v11, v17 // (op 2031) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2035) 
v_mul_f32 v15, v10, v7 // (op 2035) 
v_mul_f32 v16, v10, v8 // (op 2035) 
v_mul_f32 v17, v10, v9 // (op 2035) 

v_add_f32 v6, v14, v18 // (op 2032) 
v_add_f32 v7, v15, v19 // (op 2032) 
v_add_f32 v8, v16, v20 // (op 2032) 
v_add_f32 v9, v17, v21 // (op 2032) 




s_lshl_b32 s0, s4, 8 // (op 2033) 
v_add_u32 v0, v13, s0 // (op 2033) 


buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 2033) Store value




s_lshl_b32 s0, s2, 7 // (op 2046) 
s_mul_i32 s1, 3, s0 // (op 2046) 

v_add_u32 v0, v12, s1 // (op 2046) 


buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 2046) Load value

v_accvgpr_read v14, a12 // (op 2043) 
v_accvgpr_read v15, a13 // (op 2043) 
v_accvgpr_read v16, a14 // (op 2043) 
v_accvgpr_read v17, a15 // (op 2043) 
v_mul_f32 v18, v11, v14 // (op 2043) 
v_mul_f32 v19, v11, v15 // (op 2043) 
v_mul_f32 v20, v11, v16 // (op 2043) 
v_mul_f32 v21, v11, v17 // (op 2043) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 2047) 
v_mul_f32 v15, v10, v7 // (op 2047) 
v_mul_f32 v16, v10, v8 // (op 2047) 
v_mul_f32 v17, v10, v9 // (op 2047) 

v_add_f32 v6, v14, v18 // (op 2044) 
v_add_f32 v7, v15, v19 // (op 2044) 
v_add_f32 v8, v16, v20 // (op 2044) 
v_add_f32 v9, v17, v21 // (op 2044) 




s_lshl_b32 s0, s4, 7 // (op 2045) 
s_mul_i32 s1, 3, s0 // (op 2045) 

v_add_u32 v0, v13, s1 // (op 2045) 


buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 2045) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__ConditionalBottom_0_beta_133:
















s_endpgm  // End of RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM_
.LRR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__end:
.size RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM_, .LRR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM__end-RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 164
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 50688 // lds bytes
  .amdhsa_accum_offset 132
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
    DataTypeB: F8
    DestDataType: s
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
    macrotile: [64, 128, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [2, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x128x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 50688
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 74
    .vgpr_count: 130
    .agpr_count: 32
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {is-null: true}, {is-null: true}]
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
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 28
        .expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}}
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
        .expression: {type: MagicMultiple, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_24, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 160, size: 4}, rhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}}}
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
        .expression: {type: MagicMultiple, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 52
        .expression: {type: MagicShiftAndSign, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}
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
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_10_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_21_pointer
        .size: 8
        .offset: 72
        .expression: {type: CommandArgument, size: 8, offset: 208, name: Tensor_21_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP8, pointerType: PointerGlobal}
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
        .expression: {type: MagicShiftAndSign, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_24, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 160, size: 4}, rhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_27
        .size: 4
        .offset: 156
        .expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: Subtract_24
        .size: 4
        .offset: 160
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
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

