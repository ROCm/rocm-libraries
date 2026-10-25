// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT



.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM_,@function
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 6942) 
s_mul_hi_i32 s1, s16, s10 // (op 7081) 
v_lshrrev_b32 v0, 7, v1 // (op 7003) 
v_and_b32 v3, 15, v1 // (op 7003) 
v_lshl_add_u32 v4, v0, 6, v3 // (op 7003) 


v_and_b32 v0, 48, v1 // (op 7003) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 7003) 


v_lshrrev_b32 v0, 6, v1 // (op 7060) 
v_and_b32 v4, 1, v0 // (op 7060) 

v_and_b32 v0, 15, v1 // (op 7060) 
v_lshl_add_u32 v5, v4, 6, v0 // (op 7060) 


v_and_b32 v0, 63, v1 // (op 7060) 
v_lshrrev_b32 v4, 4, v0 // (op 7060) 

v_lshlrev_b32 v0, 5, v4 // (op 7060) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 7060) 


v_lshlrev_b32 v0, 2, v4 // (op 7060) 

v_lshrrev_b32 v4, 3, v0 // (op 7060) 

v_lshrrev_b32 v0, 7, v1 // (op 7127) 
v_and_b32 v5, 15, v1 // (op 7127) 
v_lshl_add_u32 v6, v0, 6, v5 // (op 7127) 


v_and_b32 v0, 48, v1 // (op 7127) 
v_lshl_add_u32 v5, v6, 7, v0 // (op 7127) 


v_lshrrev_b32 v0, 6, v1 // (op 7171) 
v_and_b32 v6, 1, v0 // (op 7171) 

v_and_b32 v0, 15, v1 // (op 7171) 
v_lshl_add_u32 v7, v6, 6, v0 // (op 7171) 


v_and_b32 v0, 63, v1 // (op 7171) 
v_lshrrev_b32 v6, 4, v0 // (op 7171) 

v_lshlrev_b32 v0, 5, v6 // (op 7171) 

v_lshl_add_u32 v6, v7, 7, v0 // (op 7171) 


v_lshlrev_b32 v0, 2, v6 // (op 7171) 

v_lshrrev_b32 v6, 3, v0 // (op 7171) 

v_lshrrev_b32 v0, 7, v1 // (op 6942) 
s_add_i32 s18, s0, s16 // (op 6942) 

v_and_b32 v7, 63, v1 // (op 6942) 
s_ashr_i32 s0, s18, 31 // (op 6942) 
v_lshrrev_b32 v8, 4, v7 // (op 6942) 
s_cmp_eq_i32 s10, 0 // (op 6942) 
s_mov_b32 s19, scc // (op 6942) 
v_lshlrev_b32 v9, 2, v8 // (op 6942) 

s_cmp_eq_i32 s19, 0 // (op 6942) 
s_cselect_b32 s49, 0, -1 // (op 6942) 

v_lshl_add_u32 v8, v0, 6, v9 // (op 6942) 


s_and_b32 s19, s13, 31 // (op 6942) 
v_lshrrev_b32 v0, 2, v7 // (op 6942) 

v_and_b32 v7, 3, v0 // (op 6942) 

v_lshrrev_b32 v0, 7, v1 // (op 7081) 
v_and_b32 v9, 63, v1 // (op 7081) 
s_lshl_b32 s50, 1, s19 // (op 6942) 
v_and_b32 v10, 3, v1 // (op 6942) 
s_add_i32 s51, s50, s49 // (op 6942) 


s_and_b32 s49, s0, s51 // (op 6942) 


s_add_i32 s0, s1, s16 // (op 7081) 

s_ashr_i32 s1, s0, 31 // (op 7081) 
v_lshl_add_u32 v11, v7, 2, v10 // (op 6942) 


s_add_i32 s50, s18, s49 // (op 6942) 


v_lshrrev_b32 v7, 6, v1 // (op 6942) 
s_ashr_i32 s18, s50, s19 // (op 6942) 


v_and_b32 v10, 1, v7 // (op 6942) 

s_lshl_b32 s19, s13, 24 // (op 6942) 
v_lshl_add_u32 v7, v10, 6, v11 // (op 6942) 


s_ashr_i32 s49, s19, 31 // (op 6942) 

s_xor_b32 s19, s18, s49 // (op 6942) 

s_sub_i32 s18, s19, s49 // (op 6942) 


s_mul_hi_i32 s19, s18, s14 // (op 6942) 
s_add_i32 s49, s19, s18 // (op 6942) 

s_ashr_i32 s19, s49, 31 // (op 6942) 
s_cmp_eq_i32 s14, 0 // (op 6942) 
s_mov_b32 s50, scc // (op 6942) 
s_cmp_eq_i32 s50, 0 // (op 6942) 
s_cselect_b32 s51, 0, -1 // (op 6942) 

s_and_b32 s50, s15, 31 // (op 6942) 
v_lshrrev_b32 v10, 4, v9 // (op 7081) 
s_cmp_eq_i32 s10, 0 // (op 7081) 
s_mov_b32 s52, scc // (op 7081) 
v_lshlrev_b32 v11, 2, v10 // (op 7081) 

s_cmp_eq_i32 s52, 0 // (op 7081) 
s_cselect_b32 s53, 0, -1 // (op 7081) 

s_lshl_b32 s52, 1, s50 // (op 6942) 
v_lshl_add_u32 v10, v0, 6, v11 // (op 7081) 


v_lshrrev_b32 v0, 2, v9 // (op 7081) 

s_add_i32 s54, s52, s51 // (op 6942) 


s_and_b32 s51, s19, s54 // (op 6942) 


s_add_i32 s19, s49, s51 // (op 6942) 


s_ashr_i32 s49, s19, s50 // (op 6942) 


s_lshl_b32 s19, s15, 24 // (op 6942) 
s_ashr_i32 s50, s19, 31 // (op 6942) 

s_xor_b32 s19, s49, s50 // (op 6942) 

s_sub_i32 s49, s19, s50 // (op 6942) 


s_mul_i32 s19, s18, s11 // (op 6942) 
s_sub_u32 s50, s16, s19 // (op 6942) 

s_mul_i32 s19, s49, s11 // (op 6942) 
s_add_u32 s19, s19, s50 // (op 6942) 

s_mul_hi_i32 s50, s16, s9 // (op 6942) 
s_add_i32 s51, s50, s16 // (op 6942) 

s_ashr_i32 s50, s51, 31 // (op 6942) 
s_cmp_eq_i32 s9, 0 // (op 6942) 
s_mov_b32 s52, scc // (op 6942) 
s_cmp_eq_i32 s52, 0 // (op 6942) 
s_cselect_b32 s54, 0, -1 // (op 6942) 

s_and_b32 s52, s13, 31 // (op 7081) 
v_and_b32 v9, 3, v0 // (op 7081) 

s_lshl_b32 s55, 1, s52 // (op 7081) 
v_and_b32 v0, 3, v1 // (op 7081) 
s_add_i32 s56, s55, s53 // (op 7081) 


s_and_b32 s53, s1, s56 // (op 7081) 


v_lshl_add_u32 v11, v9, 2, v0 // (op 7081) 


s_add_i32 s1, s0, s53 // (op 7081) 


v_lshrrev_b32 v0, 6, v1 // (op 7081) 
s_ashr_i32 s0, s1, s52 // (op 7081) 


v_and_b32 v9, 1, v0 // (op 7081) 

s_lshl_b32 s1, s13, 24 // (op 7081) 
v_lshl_add_u32 v0, v9, 6, v11 // (op 7081) 


s_ashr_i32 s52, s1, 31 // (op 7081) 

s_xor_b32 s1, s0, s52 // (op 7081) 

s_sub_i32 s0, s1, s52 // (op 7081) 


s_mul_hi_i32 s1, s0, s14 // (op 7081) 
s_add_i32 s52, s1, s0 // (op 7081) 

s_ashr_i32 s1, s52, 31 // (op 7081) 
s_cmp_eq_i32 s14, 0 // (op 7081) 
s_mov_b32 s53, scc // (op 7081) 
s_cmp_eq_i32 s53, 0 // (op 7081) 
s_cselect_b32 s55, 0, -1 // (op 7081) 

s_and_b32 s53, s15, 31 // (op 7081) 
s_lshl_b32 s56, 1, s53 // (op 7081) 
s_add_i32 s57, s56, s55 // (op 7081) 


s_and_b32 s55, s1, s57 // (op 7081) 


s_add_i32 s1, s52, s55 // (op 7081) 


s_ashr_i32 s52, s1, s53 // (op 7081) 


s_lshl_b32 s1, s15, 24 // (op 7081) 
s_ashr_i32 s53, s1, 31 // (op 7081) 

s_xor_b32 s1, s52, s53 // (op 7081) 

s_sub_i32 s52, s1, s53 // (op 7081) 


s_mul_i32 s1, s0, s11 // (op 7081) 
s_sub_u32 s53, s16, s1 // (op 7081) 

s_mul_i32 s1, s52, s11 // (op 7081) 
s_add_u32 s1, s1, s53 // (op 7081) 

s_mul_hi_i32 s53, s16, s9 // (op 7081) 
s_add_i32 s55, s53, s16 // (op 7081) 

s_ashr_i32 s53, s55, 31 // (op 7081) 
s_cmp_eq_i32 s9, 0 // (op 7081) 
s_mov_b32 s56, scc // (op 7081) 
s_cmp_eq_i32 s56, 0 // (op 7081) 
s_cselect_b32 s57, 0, -1 // (op 7081) 


s_waitcnt lgkmcnt(0)
s_and_b32 s56, s45, 31 // (op 6942) 
s_lshl_b32 s58, 1, s56 // (op 6942) 
s_add_i32 s59, s58, s54 // (op 6942) 


s_and_b32 s54, s50, s59 // (op 6942) 


s_add_i32 s50, s51, s54 // (op 6942) 


s_ashr_i32 s51, s50, s56 // (op 6942) 


s_lshl_b32 s50, s45, 24 // (op 6942) 
s_ashr_i32 s54, s50, 31 // (op 6942) 

s_xor_b32 s50, s51, s54 // (op 6942) 

s_sub_i32 s51, s50, s54 // (op 6942) 


s_and_b32 s50, s45, 31 // (op 7081) 
s_lshl_b32 s54, 1, s50 // (op 7081) 
s_add_i32 s56, s54, s57 // (op 7081) 


s_and_b32 s54, s53, s56 // (op 7081) 


s_add_i32 s53, s55, s54 // (op 7081) 


s_ashr_i32 s54, s53, s50 // (op 7081) 


s_lshl_b32 s50, s45, 24 // (op 7081) 
s_ashr_i32 s53, s50, 31 // (op 7081) 

s_xor_b32 s50, s54, s53 // (op 7081) 

s_sub_i32 s54, s50, s53 // (op 7081) 


s_add_u32 s56, s6, 128 // (op 6942) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 6942) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 6942) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 6942) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 7 // (op 6942) 

s_mul_i32 s50, s11, s56 // (op 6942) 
s_mul_i32 s53, s8, s50 // (op 6942) 

s_mul_i32 s50, s51, s53 // (op 6942) 

s_sub_u32 s53, s16, s50 // (op 6942) 

s_mov_b32 s50, s53 // (op 6942) 
s_mul_hi_i32 s55, s50, s12 // (op 6942) 
s_add_i32 s58, s55, s50 // (op 6942) 


s_ashr_i32 s50, s58, 31 // (op 6942) 
s_cmp_eq_i32 s12, 0 // (op 6942) 
s_mov_b32 s55, scc // (op 6942) 
s_cmp_eq_i32 s55, 0 // (op 6942) 
s_cselect_b32 s59, 0, -1 // (op 6942) 

s_and_b32 s55, s44, 31 // (op 6942) 
s_lshl_b32 s60, 1, s55 // (op 6942) 
s_add_i32 s61, s60, s59 // (op 6942) 


s_and_b32 s59, s50, s61 // (op 6942) 


s_add_i32 s50, s58, s59 // (op 6942) 


s_ashr_i32 s58, s50, s55 // (op 6942) 


s_lshl_b32 s50, s44, 24 // (op 6942) 
s_ashr_i32 s55, s50, 31 // (op 6942) 

s_xor_b32 s50, s58, s55 // (op 6942) 

s_sub_i32 s58, s50, s55 // (op 6942) 


s_mul_i32 s50, s46, s11 // (op 6942) 
s_add_u32 s60, s20, 128 // (op 6942) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 6942) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 6942) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 6942) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 7 // (op 6942) 

s_sub_i32 s55, s60, s50 // (op 6942) 

s_mul_i32 s59, s58, s55 // (op 6942) 

s_sub_u32 s55, s53, s59 // (op 6942) 


s_add_u32 s53, s55, s50 // (op 6942) 


s_cmp_eq_i32 s51, 0 // (op 6942) 
s_mov_b32 s50, scc // (op 6942) 

s_cmp_eq_i32 s50, 0 // (op 6942) 
s_cselect_b32 s51, s53, s19 // (op 6942) 


s_mul_i32 s19, s49, s56 // (op 6942) 


v_lshl_add_u32 v9, s51, 7, v8 // (op 6942) 


s_sub_i32 s49, s18, s19 // (op 6942) 


s_cmp_eq_i32 s50, 0 // (op 6942) 
s_cselect_b32 s51, s58, s49 // (op 6942) 



v_lshl_add_u32 v8, s51, 7, v7 // (op 6942) 


v_mul_lo_u32 v7, v8, s2 // (op 6942) 

v_add_lshl_u32 v8, v9, v7, 1 // (op 6942) 


s_add_u32 s18, s6, 128 // (op 7081) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 7081) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 7081) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 7081) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 7 // (op 7081) 

s_mul_i32 s53, s11, s18 // (op 7081) 
s_mul_i32 s49, s8, s53 // (op 7081) 

s_mul_i32 s53, s54, s49 // (op 7081) 

s_sub_u32 s49, s16, s53 // (op 7081) 

s_mov_b32 s53, s49 // (op 7081) 
s_mul_hi_i32 s50, s53, s12 // (op 7081) 
s_add_i32 s51, s50, s53 // (op 7081) 


s_ashr_i32 s50, s51, 31 // (op 7081) 
s_cmp_eq_i32 s12, 0 // (op 7081) 
s_mov_b32 s53, scc // (op 7081) 
s_cmp_eq_i32 s53, 0 // (op 7081) 
s_cselect_b32 s55, 0, -1 // (op 7081) 

s_and_b32 s53, s44, 31 // (op 7081) 
s_lshl_b32 s56, 1, s53 // (op 7081) 
s_add_i32 s57, s56, s55 // (op 7081) 


s_and_b32 s55, s50, s57 // (op 7081) 


s_add_i32 s50, s51, s55 // (op 7081) 


s_ashr_i32 s51, s50, s53 // (op 7081) 


s_lshl_b32 s50, s44, 24 // (op 7081) 
s_ashr_i32 s53, s50, 31 // (op 7081) 

s_xor_b32 s50, s51, s53 // (op 7081) 

s_sub_i32 s51, s50, s53 // (op 7081) 


s_mul_i32 s50, s46, s11 // (op 7081) 
s_add_u32 s56, s20, 128 // (op 7081) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 7081) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 7081) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 7081) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 7 // (op 7081) 

s_sub_i32 s53, s56, s50 // (op 7081) 

s_mul_i32 s55, s51, s53 // (op 7081) 

s_sub_u32 s53, s49, s55 // (op 7081) 


s_add_u32 s49, s53, s50 // (op 7081) 


s_cmp_eq_i32 s54, 0 // (op 7081) 
s_mov_b32 s50, scc // (op 7081) 

s_cmp_eq_i32 s50, 0 // (op 7081) 
s_cselect_b32 s53, s49, s1 // (op 7081) 


s_mul_i32 s1, s52, s18 // (op 7081) 


v_lshl_add_u32 v7, s53, 7, v10 // (op 7081) 


s_sub_i32 s49, s0, s1 // (op 7081) 


s_cmp_eq_i32 s50, 0 // (op 7081) 
s_cselect_b32 s0, s51, s49 // (op 7081) 



v_lshl_add_u32 v9, s0, 7, v0 // (op 7081) 


v_mul_lo_u32 v0, v9, s4 // (op 7081) 

v_add_lshl_u32 v9, v7, v0, 1 // (op 7081) 


s_mul_i32 s1, s2, s7 // (op 6944) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 6944) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 6944) least significant
s_mul_hi_u32 s19, s2, s6 // (op 6944) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 6944) most significant: sum
s_add_u32 s1, s1, s19 // (op 6944) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 6944) 

s_mul_i32 s1, s4, s7 // (op 7083) most significant: low of low * high
s_mul_i32 s49, s5, s6 // (op 7083) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 7083) least significant
s_mul_hi_u32 s50, s4, s6 // (op 7083) most significant: high of low * low
s_add_u32 s1, s1, s49 // (op 7083) most significant: sum
s_add_u32 s1, s1, s50 // (op 7083) most significant: sum


s_lshl_b64 s[50:51], s[0:1], 1 // (op 7083) 

s_mov_b32 s52, s22 // (op 6944) reg expression
s_mov_b32 s53, s23 // (op 6944) reg expression
s_mov_b32 s54, s18 // (op 6944) reg expression
s_mov_b32 s55, 131072 // (op 6944) call()

s_mov_b32 s56, s24 // (op 7083) reg expression

s_mov_b32 s57, s25 // (op 7083) reg expression
s_mov_b32 s58, s50 // (op 7083) reg expression
s_mov_b32 s59, 131072 // (op 7083) call()


v_mov_b32 v0, s47 // (op 2) Move value
v_mov_b32 v7, s48 // (op 35) Move value


v_accvgpr_write a0, 0.00000 // (op 57) call()
v_accvgpr_write a1, 0.00000 // (op 57) call()
v_accvgpr_write a2, 0.00000 // (op 57) call()
v_accvgpr_write a3, 0.00000 // (op 57) call()
v_accvgpr_write a4, 0.00000 // (op 281) call()
v_accvgpr_write a5, 0.00000 // (op 281) call()
v_accvgpr_write a6, 0.00000 // (op 281) call()
v_accvgpr_write a7, 0.00000 // (op 281) call()
v_accvgpr_write a8, 0.00000 // (op 417) call()
v_accvgpr_write a9, 0.00000 // (op 417) call()
v_accvgpr_write a10, 0.00000 // (op 417) call()
v_accvgpr_write a11, 0.00000 // (op 417) call()
v_accvgpr_write a12, 0.00000 // (op 553) call()
v_accvgpr_write a13, 0.00000 // (op 553) call()
v_accvgpr_write a14, 0.00000 // (op 553) call()
v_accvgpr_write a15, 0.00000 // (op 553) call()
v_accvgpr_write a16, 0.00000 // (op 777) call()
v_accvgpr_write a17, 0.00000 // (op 777) call()
v_accvgpr_write a18, 0.00000 // (op 777) call()
v_accvgpr_write a19, 0.00000 // (op 777) call()
v_accvgpr_write a20, 0.00000 // (op 1038) call()
v_accvgpr_write a21, 0.00000 // (op 1038) call()
v_accvgpr_write a22, 0.00000 // (op 1038) call()
v_accvgpr_write a23, 0.00000 // (op 1038) call()
v_accvgpr_write a24, 0.00000 // (op 1039) call()
v_accvgpr_write a25, 0.00000 // (op 1039) call()
v_accvgpr_write a26, 0.00000 // (op 1039) call()
v_accvgpr_write a27, 0.00000 // (op 1039) call()
v_accvgpr_write a28, 0.00000 // (op 1040) call()
v_accvgpr_write a29, 0.00000 // (op 1040) call()
v_accvgpr_write a30, 0.00000 // (op 1040) call()
v_accvgpr_write a31, 0.00000 // (op 1040) call()
v_accvgpr_write a32, 0.00000 // (op 1387) call()
v_accvgpr_write a33, 0.00000 // (op 1387) call()
v_accvgpr_write a34, 0.00000 // (op 1387) call()
v_accvgpr_write a35, 0.00000 // (op 1387) call()
v_accvgpr_write a36, 0.00000 // (op 1648) call()
v_accvgpr_write a37, 0.00000 // (op 1648) call()
v_accvgpr_write a38, 0.00000 // (op 1648) call()
v_accvgpr_write a39, 0.00000 // (op 1648) call()
v_accvgpr_write a40, 0.00000 // (op 1649) call()
v_accvgpr_write a41, 0.00000 // (op 1649) call()
v_accvgpr_write a42, 0.00000 // (op 1649) call()
v_accvgpr_write a43, 0.00000 // (op 1649) call()
v_accvgpr_write a44, 0.00000 // (op 1650) call()
v_accvgpr_write a45, 0.00000 // (op 1650) call()
v_accvgpr_write a46, 0.00000 // (op 1650) call()
v_accvgpr_write a47, 0.00000 // (op 1650) call()
v_accvgpr_write a48, 0.00000 // (op 1997) call()
v_accvgpr_write a49, 0.00000 // (op 1997) call()
v_accvgpr_write a50, 0.00000 // (op 1997) call()
v_accvgpr_write a51, 0.00000 // (op 1997) call()
v_accvgpr_write a52, 0.00000 // (op 2258) call()
v_accvgpr_write a53, 0.00000 // (op 2258) call()
v_accvgpr_write a54, 0.00000 // (op 2258) call()
v_accvgpr_write a55, 0.00000 // (op 2258) call()
v_accvgpr_write a56, 0.00000 // (op 2259) call()
v_accvgpr_write a57, 0.00000 // (op 2259) call()
v_accvgpr_write a58, 0.00000 // (op 2259) call()
v_accvgpr_write a59, 0.00000 // (op 2259) call()
v_accvgpr_write a60, 0.00000 // (op 2260) call()
v_accvgpr_write a61, 0.00000 // (op 2260) call()
v_accvgpr_write a62, 0.00000 // (op 2260) call()
v_accvgpr_write a63, 0.00000 // (op 2260) call()
s_mul_hi_i32 s0, s16, s10 // (op 6832) 
v_lshrrev_b32 v10, 3, v1 // (op 6832) 
s_add_i32 s1, s0, s16 // (op 6832) 

v_and_b32 v11, -8, v1 // (op 6832) 
s_ashr_i32 s0, s1, 31 // (op 6832) 
v_sub_u32 v12, v1, v11 // (op 6832) 

s_cmp_eq_i32 s10, 0 // (op 6832) 
s_mov_b32 s18, scc // (op 6832) 
v_lshlrev_b32 v11, 4, v12 // (op 6832) 

s_cmp_eq_i32 s18, 0 // (op 6832) 
s_cselect_b32 s19, 0, -1 // (op 6832) 

v_mov_b32 v13, 0 // (op 6832) convert
v_mov_b32 v12, v11 // (op 6832) convert

s_and_b32 s18, s13, 31 // (op 6832) 
s_lshl_b32 s22, 1, s18 // (op 6832) 
s_add_i32 s23, s22, s19 // (op 6832) 


s_and_b32 s19, s0, s23 // (op 6832) 


s_add_i32 s0, s1, s19 // (op 6832) 


s_ashr_i32 s1, s0, s18 // (op 6832) 


s_lshl_b32 s0, s13, 24 // (op 6832) 
s_ashr_i32 s18, s0, 31 // (op 6832) 

s_xor_b32 s0, s1, s18 // (op 6832) 

s_sub_i32 s1, s0, s18 // (op 6832) 


s_mul_hi_i32 s0, s1, s14 // (op 6832) 
s_add_i32 s18, s0, s1 // (op 6832) 

s_ashr_i32 s0, s18, 31 // (op 6832) 
s_cmp_eq_i32 s14, 0 // (op 6832) 
s_mov_b32 s19, scc // (op 6832) 
s_cmp_eq_i32 s19, 0 // (op 6832) 
s_cselect_b32 s22, 0, -1 // (op 6832) 

s_and_b32 s19, s15, 31 // (op 6832) 
s_lshl_b32 s23, 1, s19 // (op 6832) 
s_add_i32 s24, s23, s22 // (op 6832) 


s_and_b32 s25, s0, s24 // (op 6832) 


s_add_i32 s0, s18, s25 // (op 6832) 


s_ashr_i32 s18, s0, s19 // (op 6832) 


s_lshl_b32 s0, s15, 24 // (op 6832) 
s_ashr_i32 s19, s0, 31 // (op 6832) 

s_xor_b32 s0, s18, s19 // (op 6832) 

s_sub_i32 s18, s0, s19 // (op 6832) 


s_mul_i32 s0, s1, s11 // (op 6832) 

s_sub_u32 s1, s16, s0 // (op 6832) 

s_mul_i32 s0, s18, s11 // (op 6832) 
s_add_u32 s0, s0, s1 // (op 6832) 


s_mul_hi_i32 s1, s16, s9 // (op 6832) 
s_add_i32 s18, s1, s16 // (op 6832) 

s_ashr_i32 s1, s18, 31 // (op 6832) 
s_cmp_eq_i32 s9, 0 // (op 6832) 
s_mov_b32 s19, scc // (op 6832) 
s_cmp_eq_i32 s19, 0 // (op 6832) 
s_cselect_b32 s22, 0, -1 // (op 6832) 

s_and_b32 s19, s45, 31 // (op 6832) 
s_lshl_b32 s23, 1, s19 // (op 6832) 
s_add_i32 s24, s23, s22 // (op 6832) 


s_and_b32 s25, s1, s24 // (op 6832) 


s_add_i32 s1, s18, s25 // (op 6832) 


s_ashr_i32 s18, s1, s19 // (op 6832) 


s_lshl_b32 s1, s45, 24 // (op 6832) 
s_ashr_i32 s19, s1, 31 // (op 6832) 

s_xor_b32 s1, s18, s19 // (op 6832) 

s_sub_i32 s18, s1, s19 // (op 6832) 


s_add_u32 s22, s6, 128 // (op 6832) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 6832) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 6832) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 6832) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 6832) 

s_mul_i32 s1, s11, s22 // (op 6832) 

s_mul_i32 s19, s8, s1 // (op 6832) 

s_mul_i32 s1, s18, s19 // (op 6832) 

s_sub_u32 s19, s16, s1 // (op 6832) 

s_mov_b32 s1, s19 // (op 6832) 
s_mul_hi_i32 s22, s1, s12 // (op 6832) 
s_add_i32 s23, s22, s1 // (op 6832) 


s_ashr_i32 s1, s23, 31 // (op 6832) 
s_cmp_eq_i32 s12, 0 // (op 6832) 
s_mov_b32 s22, scc // (op 6832) 
s_cmp_eq_i32 s22, 0 // (op 6832) 
s_cselect_b32 s24, 0, -1 // (op 6832) 

s_and_b32 s22, s44, 31 // (op 6832) 
s_lshl_b32 s25, 1, s22 // (op 6832) 
s_add_i32 s47, s25, s24 // (op 6832) 


s_and_b32 s24, s1, s47 // (op 6832) 


s_add_i32 s1, s23, s24 // (op 6832) 


s_ashr_i32 s23, s1, s22 // (op 6832) 


s_lshl_b32 s1, s44, 24 // (op 6832) 
s_ashr_i32 s22, s1, 31 // (op 6832) 

s_xor_b32 s1, s23, s22 // (op 6832) 

s_sub_i32 s23, s1, s22 // (op 6832) 


s_mul_i32 s1, s46, s11 // (op 6832) 
s_add_u32 s24, s20, 128 // (op 6832) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 6832) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 6832) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 6832) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 7 // (op 6832) 

s_sub_i32 s22, s24, s1 // (op 6832) 

s_mul_i32 s24, s23, s22 // (op 6832) 


s_sub_u32 s25, s19, s24 // (op 6832) 


s_add_u32 s19, s25, s1 // (op 6832) 


s_cmp_eq_i32 s18, 0 // (op 6832) 
s_mov_b32 s1, scc // (op 6832) 

s_cmp_eq_i32 s1, 0 // (op 6832) 
s_cselect_b32 s18, s19, s0 // (op 6832) 



v_lshl_add_u32 v11, s18, 7, v10 // (op 6832) 


v_mul_lo_u32 v15, v11, s27 // (op 6832) most significant: low of low * high
v_mul_hi_u32 v10, v11, s26 // (op 6832) most significant: high of low * low
v_mul_lo_u32 v14, v11, s26 // (op 6832) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 6832) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 6832) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 6832) most significant half



s_mul_i32 s1, s26, s21 // (op 6834) most significant: low of low * high
s_mul_i32 s18, s27, s20 // (op 6834) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 6834) least significant
s_mul_hi_u32 s19, s26, s20 // (op 6834) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 6834) most significant: sum
s_add_u32 s1, s1, s19 // (op 6834) most significant: sum


s_mov_b32 s48, s28 // (op 6834) reg expression
s_mov_b32 s49, s29 // (op 6834) reg expression
s_mov_b32 s50, s0 // (op 6834) reg expression
s_mov_b32 s51, 131072 // (op 6834) call()

s_mul_hi_i32 s47, s16, s10 // (op 6851) 
v_lshrrev_b32 v10, 3, v1 // (op 6851) 
s_add_i32 s0, s47, s16 // (op 6851) 

v_and_b32 v11, -8, v1 // (op 6851) 
s_ashr_i32 s1, s0, 31 // (op 6851) 
v_sub_u32 v12, v1, v11 // (op 6851) 

s_cmp_eq_i32 s10, 0 // (op 6851) 
s_mov_b32 s47, scc // (op 6851) 
v_lshlrev_b32 v11, 4, v12 // (op 6851) 

s_cmp_eq_i32 s47, 0 // (op 6851) 
s_cselect_b32 s18, 0, -1 // (op 6851) 

v_mov_b32 v13, 0 // (op 6851) convert
v_mov_b32 v12, v11 // (op 6851) convert

s_and_b32 s19, s13, 31 // (op 6851) 
s_lshl_b32 s47, 1, s19 // (op 6851) 
s_add_i32 s22, s47, s18 // (op 6851) 


s_and_b32 s18, s1, s22 // (op 6851) 


s_add_i32 s1, s0, s18 // (op 6851) 


s_ashr_i32 s0, s1, s19 // (op 6851) 


s_lshl_b32 s1, s13, 24 // (op 6851) 
s_ashr_i32 s47, s1, 31 // (op 6851) 

s_xor_b32 s1, s0, s47 // (op 6851) 

s_sub_i32 s0, s1, s47 // (op 6851) 


s_mul_hi_i32 s1, s0, s14 // (op 6851) 
s_add_i32 s47, s1, s0 // (op 6851) 

s_ashr_i32 s1, s47, 31 // (op 6851) 
s_cmp_eq_i32 s14, 0 // (op 6851) 
s_mov_b32 s18, scc // (op 6851) 
s_cmp_eq_i32 s18, 0 // (op 6851) 
s_cselect_b32 s19, 0, -1 // (op 6851) 

s_and_b32 s18, s15, 31 // (op 6851) 
s_lshl_b32 s22, 1, s18 // (op 6851) 
s_add_i32 s23, s22, s19 // (op 6851) 


s_and_b32 s19, s1, s23 // (op 6851) 


s_add_i32 s1, s47, s19 // (op 6851) 


s_ashr_i32 s19, s1, s18 // (op 6851) 


s_lshl_b32 s1, s15, 24 // (op 6851) 
s_ashr_i32 s18, s1, 31 // (op 6851) 

s_xor_b32 s1, s19, s18 // (op 6851) 

s_sub_i32 s19, s1, s18 // (op 6851) 


s_mul_i32 s1, s0, s11 // (op 6851) 

s_sub_u32 s0, s16, s1 // (op 6851) 

s_mul_i32 s1, s19, s11 // (op 6851) 
s_add_u32 s1, s1, s0 // (op 6851) 


s_mul_hi_i32 s0, s16, s9 // (op 6851) 
s_add_i32 s47, s0, s16 // (op 6851) 

s_ashr_i32 s0, s47, 31 // (op 6851) 
s_cmp_eq_i32 s9, 0 // (op 6851) 
s_mov_b32 s18, scc // (op 6851) 
s_cmp_eq_i32 s18, 0 // (op 6851) 
s_cselect_b32 s19, 0, -1 // (op 6851) 

s_and_b32 s18, s45, 31 // (op 6851) 
s_lshl_b32 s22, 1, s18 // (op 6851) 
s_add_i32 s23, s22, s19 // (op 6851) 


s_and_b32 s19, s0, s23 // (op 6851) 


s_add_i32 s0, s47, s19 // (op 6851) 


s_ashr_i32 s19, s0, s18 // (op 6851) 


s_lshl_b32 s0, s45, 24 // (op 6851) 
s_ashr_i32 s18, s0, 31 // (op 6851) 

s_xor_b32 s0, s19, s18 // (op 6851) 

s_sub_i32 s19, s0, s18 // (op 6851) 


s_add_u32 s22, s6, 128 // (op 6851) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 6851) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 6851) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 6851) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 6851) 

s_mul_i32 s0, s11, s22 // (op 6851) 

s_mul_i32 s18, s8, s0 // (op 6851) 

s_mul_i32 s0, s19, s18 // (op 6851) 

s_sub_u32 s18, s16, s0 // (op 6851) 

s_mov_b32 s0, s18 // (op 6851) 
s_mul_hi_i32 s47, s0, s12 // (op 6851) 
s_add_i32 s22, s47, s0 // (op 6851) 


s_ashr_i32 s0, s22, 31 // (op 6851) 
s_cmp_eq_i32 s12, 0 // (op 6851) 
s_mov_b32 s47, scc // (op 6851) 
s_cmp_eq_i32 s47, 0 // (op 6851) 
s_cselect_b32 s23, 0, -1 // (op 6851) 

s_and_b32 s47, s44, 31 // (op 6851) 
s_lshl_b32 s24, 1, s47 // (op 6851) 
s_add_i32 s25, s24, s23 // (op 6851) 


s_and_b32 s23, s0, s25 // (op 6851) 


s_add_i32 s0, s22, s23 // (op 6851) 


s_ashr_i32 s22, s0, s47 // (op 6851) 


s_lshl_b32 s0, s44, 24 // (op 6851) 
s_ashr_i32 s47, s0, 31 // (op 6851) 

s_xor_b32 s0, s22, s47 // (op 6851) 

s_sub_i32 s22, s0, s47 // (op 6851) 


s_mul_i32 s0, s46, s11 // (op 6851) 
s_add_u32 s24, s20, 128 // (op 6851) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 6851) End of Int64 add, unlocking SCC
s_sub_u32 s60, s24, 1 // (op 6851) least significant half; sets scc
s_subb_u32 s61, s25, 0 // (op 6851) most significant half; uses scc

s_ashr_i64 s[24:25], s[60:61], 7 // (op 6851) 

s_sub_i32 s23, s24, s0 // (op 6851) 

s_mul_i32 s47, s22, s23 // (op 6851) 


s_sub_u32 s22, s18, s47 // (op 6851) 


s_add_u32 s18, s22, s0 // (op 6851) 


s_cmp_eq_i32 s19, 0 // (op 6851) 
s_mov_b32 s0, scc // (op 6851) 

s_cmp_eq_i32 s0, 0 // (op 6851) 
s_cselect_b32 s19, s18, s1 // (op 6851) 



v_lshl_add_u32 v11, s19, 7, v10 // (op 6851) 


v_mul_lo_u32 v17, v11, s27 // (op 6851) most significant: low of low * high
v_mul_hi_u32 v10, v11, s26 // (op 6851) most significant: high of low * low
v_mul_lo_u32 v16, v11, s26 // (op 6851) least significant: low of low * low
v_add_u32 v17, v17, v10 // (op 6851) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v12 // (op 6851) least significant half
v_addc_co_u32 v17, s[0:1], v17, v13, s[0:1] // (op 6851) most significant half



s_mul_i32 s1, s26, s21 // (op 6853) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 6853) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 6853) least significant
s_mul_hi_u32 s18, s26, s20 // (op 6853) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 6853) most significant: sum
s_add_u32 s1, s1, s18 // (op 6853) most significant: sum


s_mov_b32 s48, s28 // (op 6853) reg expression
s_mov_b32 s49, s29 // (op 6853) reg expression
s_mov_b32 s50, s0 // (op 6853) reg expression
s_mov_b32 s51, 131072 // (op 6853) call()

s_mul_hi_i32 s47, s16, s10 // (op 6896) 
v_lshrrev_b32 v10, 2, v1 // (op 6896) 
s_add_i32 s0, s47, s16 // (op 6896) 

v_and_b32 v11, -4, v1 // (op 6896) 
s_ashr_i32 s1, s0, 31 // (op 6896) 
v_sub_u32 v12, v1, v11 // (op 6896) 

s_cmp_eq_i32 s10, 0 // (op 6896) 
s_mov_b32 s47, scc // (op 6896) 
v_lshlrev_b32 v11, 5, v12 // (op 6896) 

s_cmp_eq_i32 s47, 0 // (op 6896) 
s_cselect_b32 s18, 0, -1 // (op 6896) 

v_mov_b32 v13, 0 // (op 6896) convert
v_mov_b32 v12, v11 // (op 6896) convert

s_and_b32 s19, s13, 31 // (op 6896) 
s_lshl_b32 s47, 1, s19 // (op 6896) 
s_add_i32 s22, s47, s18 // (op 6896) 


s_and_b32 s18, s1, s22 // (op 6896) 


s_add_i32 s1, s0, s18 // (op 6896) 


s_ashr_i32 s0, s1, s19 // (op 6896) 


s_lshl_b32 s1, s13, 24 // (op 6896) 
s_ashr_i32 s47, s1, 31 // (op 6896) 

s_xor_b32 s1, s0, s47 // (op 6896) 

s_sub_i32 s0, s1, s47 // (op 6896) 


s_mul_hi_i32 s1, s0, s14 // (op 6896) 
s_add_i32 s47, s1, s0 // (op 6896) 

s_ashr_i32 s1, s47, 31 // (op 6896) 
s_cmp_eq_i32 s14, 0 // (op 6896) 
s_mov_b32 s18, scc // (op 6896) 
s_cmp_eq_i32 s18, 0 // (op 6896) 
s_cselect_b32 s19, 0, -1 // (op 6896) 

s_and_b32 s18, s15, 31 // (op 6896) 
s_lshl_b32 s22, 1, s18 // (op 6896) 
s_add_i32 s23, s22, s19 // (op 6896) 


s_and_b32 s19, s1, s23 // (op 6896) 


s_add_i32 s1, s47, s19 // (op 6896) 


s_ashr_i32 s19, s1, s18 // (op 6896) 


s_lshl_b32 s1, s15, 24 // (op 6896) 
s_ashr_i32 s18, s1, 31 // (op 6896) 

s_xor_b32 s1, s19, s18 // (op 6896) 

s_sub_i32 s19, s1, s18 // (op 6896) 


s_add_u32 s22, s6, 128 // (op 6896) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 6896) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 6896) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 6896) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 6896) 

s_mul_i32 s1, s19, s22 // (op 6896) 

s_sub_i32 s47, s0, s1 // (op 6896) 


s_mul_hi_i32 s0, s16, s9 // (op 6896) 
s_add_i32 s1, s0, s16 // (op 6896) 

s_ashr_i32 s0, s1, 31 // (op 6896) 
s_cmp_eq_i32 s9, 0 // (op 6896) 
s_mov_b32 s18, scc // (op 6896) 
s_cmp_eq_i32 s18, 0 // (op 6896) 
s_cselect_b32 s19, 0, -1 // (op 6896) 

s_and_b32 s18, s45, 31 // (op 6896) 
s_lshl_b32 s24, 1, s18 // (op 6896) 
s_add_i32 s25, s24, s19 // (op 6896) 


s_and_b32 s19, s0, s25 // (op 6896) 


s_add_i32 s0, s1, s19 // (op 6896) 


s_ashr_i32 s1, s0, s18 // (op 6896) 


s_lshl_b32 s0, s45, 24 // (op 6896) 
s_ashr_i32 s18, s0, 31 // (op 6896) 

s_xor_b32 s0, s1, s18 // (op 6896) 

s_sub_i32 s1, s0, s18 // (op 6896) 


s_mul_i32 s0, s11, s22 // (op 6896) 

s_mul_i32 s18, s8, s0 // (op 6896) 

s_mul_i32 s0, s1, s18 // (op 6896) 

s_sub_u32 s18, s16, s0 // (op 6896) 

s_mul_hi_i32 s0, s18, s12 // (op 6896) 
s_add_i32 s19, s0, s18 // (op 6896) 


s_ashr_i32 s0, s19, 31 // (op 6896) 
s_cmp_eq_i32 s12, 0 // (op 6896) 
s_mov_b32 s18, scc // (op 6896) 
s_cmp_eq_i32 s18, 0 // (op 6896) 
s_cselect_b32 s22, 0, -1 // (op 6896) 

s_and_b32 s18, s44, 31 // (op 6896) 
s_lshl_b32 s23, 1, s18 // (op 6896) 
s_add_i32 s24, s23, s22 // (op 6896) 


s_and_b32 s25, s0, s24 // (op 6896) 


s_add_i32 s0, s19, s25 // (op 6896) 


s_ashr_i32 s19, s0, s18 // (op 6896) 


s_lshl_b32 s0, s44, 24 // (op 6896) 
s_ashr_i32 s18, s0, 31 // (op 6896) 

s_xor_b32 s0, s19, s18 // (op 6896) 

s_sub_i32 s19, s0, s18 // (op 6896) 


s_cmp_eq_i32 s1, 0 // (op 6896) 
s_mov_b32 s0, scc // (op 6896) 

s_cmp_eq_i32 s0, 0 // (op 6896) 
s_cselect_b32 s1, s19, s47 // (op 6896) 



v_lshl_add_u32 v11, s1, 7, v10 // (op 6896) 


v_mul_lo_u32 v19, v11, s31 // (op 6896) most significant: low of low * high
v_mul_hi_u32 v10, v11, s30 // (op 6896) most significant: high of low * low
v_mul_lo_u32 v18, v11, s30 // (op 6896) least significant: low of low * low
v_add_u32 v19, v19, v10 // (op 6896) most significant: sum


v_add_co_u32 v10, s[0:1], v12, v18 // (op 6896) least significant half
v_addc_co_u32 v11, s[0:1], v13, v19, s[0:1] // (op 6896) most significant half

v_lshlrev_b64 v[10:11], 2, v[10:11] // (op 6896) 


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 6896) 

s_mul_i32 s1, s30, s7 // (op 6898) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 6898) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 6898) least significant
s_mul_hi_u32 s18, s30, s6 // (op 6898) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 6898) most significant: sum
s_add_u32 s1, s1, s18 // (op 6898) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 6898) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 6898) 

s_mov_b32 s60, s32 // (op 6898) reg expression
s_mov_b32 s61, s33 // (op 6898) reg expression
s_mov_b32 s62, s0 // (op 6898) reg expression
s_mov_b32 s63, 131072 // (op 6898) call()

s_mul_hi_i32 s47, s16, s10 // (op 6915) 
v_lshrrev_b32 v10, 2, v1 // (op 6915) 
s_add_i32 s0, s47, s16 // (op 6915) 

v_and_b32 v11, -4, v1 // (op 6915) 
s_ashr_i32 s1, s0, 31 // (op 6915) 
v_sub_u32 v18, v1, v11 // (op 6915) 

s_cmp_eq_i32 s10, 0 // (op 6915) 
s_mov_b32 s47, scc // (op 6915) 
v_lshlrev_b32 v11, 5, v18 // (op 6915) 

s_cmp_eq_i32 s47, 0 // (op 6915) 
s_cselect_b32 s18, 0, -1 // (op 6915) 

v_mov_b32 v19, 0 // (op 6915) convert
v_mov_b32 v18, v11 // (op 6915) convert

s_and_b32 s19, s13, 31 // (op 6915) 
s_lshl_b32 s47, 1, s19 // (op 6915) 
s_add_i32 s22, s47, s18 // (op 6915) 


s_and_b32 s18, s1, s22 // (op 6915) 


s_add_i32 s1, s0, s18 // (op 6915) 


s_ashr_i32 s0, s1, s19 // (op 6915) 


s_lshl_b32 s1, s13, 24 // (op 6915) 
s_ashr_i32 s47, s1, 31 // (op 6915) 

s_xor_b32 s1, s0, s47 // (op 6915) 

s_sub_i32 s0, s1, s47 // (op 6915) 


s_mul_hi_i32 s1, s0, s14 // (op 6915) 
s_add_i32 s47, s1, s0 // (op 6915) 

s_ashr_i32 s1, s47, 31 // (op 6915) 
s_cmp_eq_i32 s14, 0 // (op 6915) 
s_mov_b32 s18, scc // (op 6915) 
s_cmp_eq_i32 s18, 0 // (op 6915) 
s_cselect_b32 s19, 0, -1 // (op 6915) 

s_and_b32 s18, s15, 31 // (op 6915) 
s_lshl_b32 s22, 1, s18 // (op 6915) 
s_add_i32 s23, s22, s19 // (op 6915) 


s_and_b32 s19, s1, s23 // (op 6915) 


s_add_i32 s1, s47, s19 // (op 6915) 


s_ashr_i32 s19, s1, s18 // (op 6915) 


s_lshl_b32 s1, s15, 24 // (op 6915) 
s_ashr_i32 s18, s1, 31 // (op 6915) 

s_xor_b32 s1, s19, s18 // (op 6915) 

s_sub_i32 s19, s1, s18 // (op 6915) 


s_add_u32 s22, s6, 128 // (op 6915) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 6915) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 6915) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 6915) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 6915) 

s_mul_i32 s1, s19, s22 // (op 6915) 

s_sub_i32 s47, s0, s1 // (op 6915) 


s_mul_hi_i32 s0, s16, s9 // (op 6915) 
s_add_i32 s1, s0, s16 // (op 6915) 

s_ashr_i32 s0, s1, 31 // (op 6915) 
s_cmp_eq_i32 s9, 0 // (op 6915) 
s_mov_b32 s18, scc // (op 6915) 
s_cmp_eq_i32 s18, 0 // (op 6915) 
s_cselect_b32 s19, 0, -1 // (op 6915) 

s_and_b32 s18, s45, 31 // (op 6915) 
s_lshl_b32 s24, 1, s18 // (op 6915) 
s_add_i32 s25, s24, s19 // (op 6915) 


s_and_b32 s19, s0, s25 // (op 6915) 


s_add_i32 s0, s1, s19 // (op 6915) 


s_ashr_i32 s1, s0, s18 // (op 6915) 


s_lshl_b32 s0, s45, 24 // (op 6915) 
s_ashr_i32 s18, s0, 31 // (op 6915) 

s_xor_b32 s0, s1, s18 // (op 6915) 

s_sub_i32 s1, s0, s18 // (op 6915) 


s_mul_i32 s0, s11, s22 // (op 6915) 

s_mul_i32 s18, s8, s0 // (op 6915) 

s_mul_i32 s0, s1, s18 // (op 6915) 

s_sub_u32 s18, s16, s0 // (op 6915) 

s_mul_hi_i32 s0, s18, s12 // (op 6915) 
s_add_i32 s19, s0, s18 // (op 6915) 


s_ashr_i32 s0, s19, 31 // (op 6915) 
s_cmp_eq_i32 s12, 0 // (op 6915) 
s_mov_b32 s18, scc // (op 6915) 
s_cmp_eq_i32 s18, 0 // (op 6915) 
s_cselect_b32 s22, 0, -1 // (op 6915) 

s_and_b32 s18, s44, 31 // (op 6915) 
s_lshl_b32 s23, 1, s18 // (op 6915) 
s_add_i32 s24, s23, s22 // (op 6915) 


s_and_b32 s25, s0, s24 // (op 6915) 


s_add_i32 s0, s19, s25 // (op 6915) 


s_ashr_i32 s19, s0, s18 // (op 6915) 


s_lshl_b32 s0, s44, 24 // (op 6915) 
s_ashr_i32 s18, s0, 31 // (op 6915) 

s_xor_b32 s0, s19, s18 // (op 6915) 

s_sub_i32 s19, s0, s18 // (op 6915) 


s_cmp_eq_i32 s1, 0 // (op 6915) 
s_mov_b32 s0, scc // (op 6915) 

s_cmp_eq_i32 s0, 0 // (op 6915) 
s_cselect_b32 s1, s19, s47 // (op 6915) 



v_lshl_add_u32 v11, s1, 7, v10 // (op 6915) 


v_mul_lo_u32 v21, v11, s31 // (op 6915) most significant: low of low * high
v_mul_hi_u32 v10, v11, s30 // (op 6915) most significant: high of low * low
v_mul_lo_u32 v20, v11, s30 // (op 6915) least significant: low of low * low
v_add_u32 v21, v21, v10 // (op 6915) most significant: sum


v_add_co_u32 v10, s[0:1], v18, v20 // (op 6915) least significant half
v_addc_co_u32 v11, s[0:1], v19, v21, s[0:1] // (op 6915) most significant half

v_lshlrev_b64 v[10:11], 2, v[10:11] // (op 6915) 


v_lshrrev_b64 v[18:19], 3, v[10:11] // (op 6915) 

s_mul_i32 s1, s30, s7 // (op 6917) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 6917) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 6917) least significant
s_mul_hi_u32 s18, s30, s6 // (op 6917) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 6917) most significant: sum
s_add_u32 s1, s1, s18 // (op 6917) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 6917) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 6917) 

s_mov_b32 s60, s32 // (op 6917) reg expression
s_mov_b32 s61, s33 // (op 6917) reg expression
s_mov_b32 s62, s0 // (op 6917) reg expression
s_mov_b32 s63, 131072 // (op 6917) call()

s_mul_hi_i32 s47, s16, s10 // (op 7193) 
v_and_b32 v10, 63, v1 // (op 7193) 
s_add_i32 s0, s47, s16 // (op 7193) 

v_lshrrev_b32 v11, 4, v10 // (op 7193) 
s_ashr_i32 s1, s0, 31 // (op 7193) 
v_and_b32 v20, 3, v11 // (op 7193) 

s_cmp_eq_i32 s10, 0 // (op 7193) 
s_mov_b32 s47, scc // (op 7193) 
v_and_b32 v11, 15, v1 // (op 7193) 
s_cmp_eq_i32 s47, 0 // (op 7193) 
s_cselect_b32 s18, 0, -1 // (op 7193) 

v_lshl_add_u32 v21, v20, 4, v11 // (op 7193) 


s_and_b32 s19, s13, 31 // (op 7193) 
v_lshrrev_b32 v11, 7, v1 // (op 7193) 
s_lshl_b32 s47, 1, s19 // (op 7193) 
v_lshl_add_u32 v20, v11, 6, v21 // (op 7193) 


s_add_i32 s22, s47, s18 // (op 7193) 


v_lshrrev_b32 v11, 6, v10 // (op 7193) 

s_and_b32 s18, s1, s22 // (op 7193) 


v_lshlrev_b32 v10, 2, v11 // (op 7193) 

s_add_i32 s1, s0, s18 // (op 7193) 


s_ashr_i32 s0, s1, s19 // (op 7193) 


s_lshl_b32 s1, s13, 24 // (op 7193) 
s_ashr_i32 s47, s1, 31 // (op 7193) 

s_xor_b32 s1, s0, s47 // (op 7193) 

s_sub_i32 s0, s1, s47 // (op 7193) 


s_mul_hi_i32 s1, s0, s14 // (op 7193) 
s_add_i32 s47, s1, s0 // (op 7193) 

s_ashr_i32 s1, s47, 31 // (op 7193) 
s_cmp_eq_i32 s14, 0 // (op 7193) 
s_mov_b32 s18, scc // (op 7193) 
s_cmp_eq_i32 s18, 0 // (op 7193) 
s_cselect_b32 s19, 0, -1 // (op 7193) 

s_and_b32 s18, s15, 31 // (op 7193) 
s_lshl_b32 s22, 1, s18 // (op 7193) 
s_add_i32 s23, s22, s19 // (op 7193) 


s_and_b32 s19, s1, s23 // (op 7193) 


s_add_i32 s1, s47, s19 // (op 7193) 


s_ashr_i32 s19, s1, s18 // (op 7193) 


s_lshl_b32 s1, s15, 24 // (op 7193) 
s_ashr_i32 s18, s1, 31 // (op 7193) 

s_xor_b32 s1, s19, s18 // (op 7193) 

s_sub_i32 s19, s1, s18 // (op 7193) 


s_mul_i32 s1, s0, s11 // (op 7193) 

s_sub_u32 s0, s16, s1 // (op 7193) 

s_mul_i32 s1, s19, s11 // (op 7193) 
s_add_u32 s1, s1, s0 // (op 7193) 


s_mul_hi_i32 s0, s16, s9 // (op 7193) 
s_add_i32 s47, s0, s16 // (op 7193) 

s_ashr_i32 s0, s47, 31 // (op 7193) 
s_cmp_eq_i32 s9, 0 // (op 7193) 
s_mov_b32 s18, scc // (op 7193) 
s_cmp_eq_i32 s18, 0 // (op 7193) 
s_cselect_b32 s19, 0, -1 // (op 7193) 

s_and_b32 s18, s45, 31 // (op 7193) 
s_lshl_b32 s22, 1, s18 // (op 7193) 
s_add_i32 s23, s22, s19 // (op 7193) 


s_and_b32 s19, s0, s23 // (op 7193) 


s_add_i32 s0, s47, s19 // (op 7193) 


s_ashr_i32 s19, s0, s18 // (op 7193) 


s_lshl_b32 s0, s45, 24 // (op 7193) 
s_ashr_i32 s18, s0, 31 // (op 7193) 

s_xor_b32 s0, s19, s18 // (op 7193) 

s_sub_i32 s19, s0, s18 // (op 7193) 


s_add_u32 s22, s6, 128 // (op 7193) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 7193) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 7193) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 7193) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 7193) 

s_mul_i32 s0, s11, s22 // (op 7193) 

s_mul_i32 s18, s8, s0 // (op 7193) 

s_mul_i32 s0, s19, s18 // (op 7193) 

s_sub_u32 s18, s16, s0 // (op 7193) 

s_mov_b32 s0, s18 // (op 7193) 
s_mul_hi_i32 s47, s0, s12 // (op 7193) 
s_add_i32 s22, s47, s0 // (op 7193) 


s_ashr_i32 s0, s22, 31 // (op 7193) 
s_cmp_eq_i32 s12, 0 // (op 7193) 
s_mov_b32 s47, scc // (op 7193) 
s_cmp_eq_i32 s47, 0 // (op 7193) 
s_cselect_b32 s23, 0, -1 // (op 7193) 

s_and_b32 s47, s44, 31 // (op 7193) 
s_lshl_b32 s24, 1, s47 // (op 7193) 
s_add_i32 s25, s24, s23 // (op 7193) 


s_and_b32 s23, s0, s25 // (op 7193) 


s_add_i32 s0, s22, s23 // (op 7193) 


s_ashr_i32 s22, s0, s47 // (op 7193) 


s_lshl_b32 s0, s44, 24 // (op 7193) 
s_ashr_i32 s47, s0, 31 // (op 7193) 

s_xor_b32 s0, s22, s47 // (op 7193) 

s_sub_i32 s22, s0, s47 // (op 7193) 


s_mul_i32 s0, s46, s11 // (op 7193) 
s_add_u32 s24, s20, 128 // (op 7193) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 7193) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 7193) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 7193) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 7 // (op 7193) 

s_sub_i32 s23, s24, s0 // (op 7193) 

s_mul_i32 s47, s22, s23 // (op 7193) 


s_sub_u32 s22, s18, s47 // (op 7193) 


s_add_u32 s18, s22, s0 // (op 7193) 


s_cmp_eq_i32 s19, 0 // (op 7193) 
s_mov_b32 s0, scc // (op 7193) 

s_cmp_eq_i32 s0, 0 // (op 7193) 
s_cselect_b32 s19, s18, s1 // (op 7193) 



v_lshl_add_u32 v11, s19, 7, v20 // (op 7193) 


v_mul_lo_u32 v20, v11, s34 // (op 7193) 
v_add_u32 v20, v20, v10 // (op 7193) 


s_mul_i32 s1, s34, s21 // (op 7195) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 7195) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 7195) least significant
s_mul_hi_u32 s18, s34, s20 // (op 7195) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 7195) most significant: sum
s_add_u32 s1, s1, s18 // (op 7195) most significant: sum


s_mov_b32 s64, s36 // (op 7195) reg expression
s_mov_b32 s65, s37 // (op 7195) reg expression
s_mov_b32 s66, s0 // (op 7195) reg expression
s_mov_b32 s67, 131072 // (op 7195) call()

s_mul_hi_i32 s47, s16, s10 // (op 7214) 
v_and_b32 v10, 63, v1 // (op 7214) 
s_add_i32 s0, s47, s16 // (op 7214) 

v_lshrrev_b32 v11, 4, v10 // (op 7214) 
s_ashr_i32 s1, s0, 31 // (op 7214) 
v_and_b32 v21, 3, v11 // (op 7214) 

s_cmp_eq_i32 s10, 0 // (op 7214) 
s_mov_b32 s47, scc // (op 7214) 
v_and_b32 v11, 15, v1 // (op 7214) 
s_cmp_eq_i32 s47, 0 // (op 7214) 
s_cselect_b32 s18, 0, -1 // (op 7214) 

v_lshl_add_u32 v22, v21, 4, v11 // (op 7214) 


s_and_b32 s19, s13, 31 // (op 7214) 
v_lshrrev_b32 v11, 7, v1 // (op 7214) 
s_lshl_b32 s47, 1, s19 // (op 7214) 
v_lshl_add_u32 v21, v11, 6, v22 // (op 7214) 


s_add_i32 s22, s47, s18 // (op 7214) 


v_lshrrev_b32 v11, 6, v10 // (op 7214) 

s_and_b32 s18, s1, s22 // (op 7214) 


v_lshlrev_b32 v10, 2, v11 // (op 7214) 

s_add_i32 s1, s0, s18 // (op 7214) 


s_ashr_i32 s0, s1, s19 // (op 7214) 


s_lshl_b32 s1, s13, 24 // (op 7214) 
s_ashr_i32 s47, s1, 31 // (op 7214) 

s_xor_b32 s1, s0, s47 // (op 7214) 

s_sub_i32 s0, s1, s47 // (op 7214) 


s_mul_hi_i32 s1, s0, s14 // (op 7214) 
s_add_i32 s47, s1, s0 // (op 7214) 

s_ashr_i32 s1, s47, 31 // (op 7214) 
s_cmp_eq_i32 s14, 0 // (op 7214) 
s_mov_b32 s18, scc // (op 7214) 
s_cmp_eq_i32 s18, 0 // (op 7214) 
s_cselect_b32 s19, 0, -1 // (op 7214) 

s_and_b32 s18, s15, 31 // (op 7214) 
s_lshl_b32 s22, 1, s18 // (op 7214) 
s_add_i32 s23, s22, s19 // (op 7214) 


s_and_b32 s19, s1, s23 // (op 7214) 


s_add_i32 s1, s47, s19 // (op 7214) 


s_ashr_i32 s19, s1, s18 // (op 7214) 


s_lshl_b32 s1, s15, 24 // (op 7214) 
s_ashr_i32 s18, s1, 31 // (op 7214) 

s_xor_b32 s1, s19, s18 // (op 7214) 

s_sub_i32 s19, s1, s18 // (op 7214) 


s_mul_i32 s1, s0, s11 // (op 7214) 

s_sub_u32 s0, s16, s1 // (op 7214) 

s_mul_i32 s1, s19, s11 // (op 7214) 
s_add_u32 s1, s1, s0 // (op 7214) 


s_mul_hi_i32 s0, s16, s9 // (op 7214) 
s_add_i32 s47, s0, s16 // (op 7214) 

s_ashr_i32 s0, s47, 31 // (op 7214) 
s_cmp_eq_i32 s9, 0 // (op 7214) 
s_mov_b32 s18, scc // (op 7214) 
s_cmp_eq_i32 s18, 0 // (op 7214) 
s_cselect_b32 s19, 0, -1 // (op 7214) 

s_and_b32 s18, s45, 31 // (op 7214) 
s_lshl_b32 s22, 1, s18 // (op 7214) 
s_add_i32 s23, s22, s19 // (op 7214) 


s_and_b32 s19, s0, s23 // (op 7214) 


s_add_i32 s0, s47, s19 // (op 7214) 


s_ashr_i32 s19, s0, s18 // (op 7214) 


s_lshl_b32 s0, s45, 24 // (op 7214) 
s_ashr_i32 s18, s0, 31 // (op 7214) 

s_xor_b32 s0, s19, s18 // (op 7214) 

s_sub_i32 s19, s0, s18 // (op 7214) 


s_add_u32 s22, s6, 128 // (op 7214) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 7214) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 7214) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 7214) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 7214) 

s_mul_i32 s0, s11, s22 // (op 7214) 

s_mul_i32 s18, s8, s0 // (op 7214) 

s_mul_i32 s0, s19, s18 // (op 7214) 

s_sub_u32 s18, s16, s0 // (op 7214) 

s_mov_b32 s0, s18 // (op 7214) 
s_mul_hi_i32 s47, s0, s12 // (op 7214) 
s_add_i32 s22, s47, s0 // (op 7214) 


s_ashr_i32 s0, s22, 31 // (op 7214) 
s_cmp_eq_i32 s12, 0 // (op 7214) 
s_mov_b32 s47, scc // (op 7214) 
s_cmp_eq_i32 s47, 0 // (op 7214) 
s_cselect_b32 s23, 0, -1 // (op 7214) 

s_and_b32 s47, s44, 31 // (op 7214) 
s_lshl_b32 s24, 1, s47 // (op 7214) 
s_add_i32 s25, s24, s23 // (op 7214) 


s_and_b32 s23, s0, s25 // (op 7214) 


s_add_i32 s0, s22, s23 // (op 7214) 


s_ashr_i32 s22, s0, s47 // (op 7214) 


s_lshl_b32 s0, s44, 24 // (op 7214) 
s_ashr_i32 s47, s0, 31 // (op 7214) 

s_xor_b32 s0, s22, s47 // (op 7214) 

s_sub_i32 s22, s0, s47 // (op 7214) 


s_mul_i32 s0, s46, s11 // (op 7214) 
s_add_u32 s24, s20, 128 // (op 7214) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 7214) End of Int64 add, unlocking SCC
s_sub_u32 s68, s24, 1 // (op 7214) least significant half; sets scc
s_subb_u32 s69, s25, 0 // (op 7214) most significant half; uses scc

s_ashr_i64 s[24:25], s[68:69], 7 // (op 7214) 

s_sub_i32 s23, s24, s0 // (op 7214) 

s_mul_i32 s47, s22, s23 // (op 7214) 


s_sub_u32 s22, s18, s47 // (op 7214) 


s_add_u32 s18, s22, s0 // (op 7214) 


s_cmp_eq_i32 s19, 0 // (op 7214) 
s_mov_b32 s0, scc // (op 7214) 

s_cmp_eq_i32 s0, 0 // (op 7214) 
s_cselect_b32 s19, s18, s1 // (op 7214) 



v_lshl_add_u32 v11, s19, 7, v21 // (op 7214) 


v_mul_lo_u32 v21, v11, s34 // (op 7214) 
v_add_u32 v21, v21, v10 // (op 7214) 


s_mul_i32 s1, s34, s21 // (op 7216) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 7216) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 7216) least significant
s_mul_hi_u32 s18, s34, s20 // (op 7216) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 7216) most significant: sum
s_add_u32 s1, s1, s18 // (op 7216) most significant: sum


s_mov_b32 s64, s36 // (op 7216) reg expression
s_mov_b32 s65, s37 // (op 7216) reg expression
s_mov_b32 s66, s0 // (op 7216) reg expression
s_mov_b32 s67, 131072 // (op 7216) call()

s_mul_hi_i32 s47, s16, s10 // (op 7245) 
v_and_b32 v10, 63, v1 // (op 7245) 
s_add_i32 s0, s47, s16 // (op 7245) 

v_lshrrev_b32 v11, 4, v10 // (op 7245) 
s_ashr_i32 s1, s0, 31 // (op 7245) 
v_and_b32 v22, 3, v11 // (op 7245) 

s_cmp_eq_i32 s10, 0 // (op 7245) 
s_mov_b32 s47, scc // (op 7245) 
v_and_b32 v11, 15, v1 // (op 7245) 
s_cmp_eq_i32 s47, 0 // (op 7245) 
s_cselect_b32 s18, 0, -1 // (op 7245) 

v_lshl_add_u32 v23, v22, 4, v11 // (op 7245) 


s_and_b32 s19, s13, 31 // (op 7245) 
v_lshrrev_b32 v11, 6, v1 // (op 7245) 
s_lshl_b32 s47, 1, s19 // (op 7245) 
v_and_b32 v22, 1, v11 // (op 7245) 

s_add_i32 s22, s47, s18 // (op 7245) 


v_lshl_add_u32 v11, v22, 6, v23 // (op 7245) 


s_and_b32 s18, s1, s22 // (op 7245) 


v_lshrrev_b32 v22, 6, v10 // (op 7245) 

s_add_i32 s1, s0, s18 // (op 7245) 


s_ashr_i32 s0, s1, s19 // (op 7245) 


s_lshl_b32 s1, s13, 24 // (op 7245) 
s_ashr_i32 s47, s1, 31 // (op 7245) 

s_xor_b32 s1, s0, s47 // (op 7245) 

s_sub_i32 s0, s1, s47 // (op 7245) 


s_mul_hi_i32 s1, s0, s14 // (op 7245) 
s_add_i32 s47, s1, s0 // (op 7245) 

s_ashr_i32 s1, s47, 31 // (op 7245) 
s_cmp_eq_i32 s14, 0 // (op 7245) 
s_mov_b32 s18, scc // (op 7245) 
s_cmp_eq_i32 s18, 0 // (op 7245) 
s_cselect_b32 s19, 0, -1 // (op 7245) 

s_and_b32 s18, s15, 31 // (op 7245) 
s_lshl_b32 s22, 1, s18 // (op 7245) 
s_add_i32 s23, s22, s19 // (op 7245) 


s_and_b32 s19, s1, s23 // (op 7245) 


s_add_i32 s1, s47, s19 // (op 7245) 


s_ashr_i32 s19, s1, s18 // (op 7245) 


s_lshl_b32 s1, s15, 24 // (op 7245) 
s_ashr_i32 s18, s1, 31 // (op 7245) 

s_xor_b32 s1, s19, s18 // (op 7245) 

s_sub_i32 s19, s1, s18 // (op 7245) 


s_add_u32 s22, s6, 128 // (op 7245) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 7245) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 7245) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 7245) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 7245) 

s_mul_i32 s1, s19, s22 // (op 7245) 

s_sub_i32 s47, s0, s1 // (op 7245) 


s_mul_hi_i32 s0, s16, s9 // (op 7245) 
s_add_i32 s1, s0, s16 // (op 7245) 

s_ashr_i32 s0, s1, 31 // (op 7245) 
s_cmp_eq_i32 s9, 0 // (op 7245) 
s_mov_b32 s18, scc // (op 7245) 
s_cmp_eq_i32 s18, 0 // (op 7245) 
s_cselect_b32 s19, 0, -1 // (op 7245) 

s_and_b32 s18, s45, 31 // (op 7245) 
s_lshl_b32 s24, 1, s18 // (op 7245) 
s_add_i32 s25, s24, s19 // (op 7245) 


s_and_b32 s19, s0, s25 // (op 7245) 


s_add_i32 s0, s1, s19 // (op 7245) 


s_ashr_i32 s1, s0, s18 // (op 7245) 


s_lshl_b32 s0, s45, 24 // (op 7245) 
s_ashr_i32 s18, s0, 31 // (op 7245) 

s_xor_b32 s0, s1, s18 // (op 7245) 

s_sub_i32 s1, s0, s18 // (op 7245) 


s_mul_i32 s0, s11, s22 // (op 7245) 

s_mul_i32 s18, s8, s0 // (op 7245) 

s_mul_i32 s0, s1, s18 // (op 7245) 

s_sub_u32 s18, s16, s0 // (op 7245) 

s_mul_hi_i32 s0, s18, s12 // (op 7245) 
s_add_i32 s19, s0, s18 // (op 7245) 


s_ashr_i32 s0, s19, 31 // (op 7245) 
s_cmp_eq_i32 s12, 0 // (op 7245) 
s_mov_b32 s18, scc // (op 7245) 
s_cmp_eq_i32 s18, 0 // (op 7245) 
s_cselect_b32 s22, 0, -1 // (op 7245) 

s_and_b32 s18, s44, 31 // (op 7245) 
s_lshl_b32 s23, 1, s18 // (op 7245) 
s_add_i32 s24, s23, s22 // (op 7245) 


s_and_b32 s25, s0, s24 // (op 7245) 


s_add_i32 s0, s19, s25 // (op 7245) 


s_ashr_i32 s19, s0, s18 // (op 7245) 


s_lshl_b32 s0, s44, 24 // (op 7245) 
s_ashr_i32 s18, s0, 31 // (op 7245) 

s_xor_b32 s0, s19, s18 // (op 7245) 

s_sub_i32 s19, s0, s18 // (op 7245) 


s_cmp_eq_i32 s1, 0 // (op 7245) 
s_mov_b32 s0, scc // (op 7245) 

s_cmp_eq_i32 s0, 0 // (op 7245) 
s_cselect_b32 s1, s19, s47 // (op 7245) 



v_lshl_add_u32 v10, s1, 7, v11 // (op 7245) 


v_mul_lo_u32 v11, v10, s38 // (op 7245) 

v_lshl_add_u32 v10, v22, 2, v11 // (op 7245) 


s_mul_i32 s1, s38, s7 // (op 7247) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 7247) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 7247) least significant
s_mul_hi_u32 s18, s38, s6 // (op 7247) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 7247) most significant: sum
s_add_u32 s1, s1, s18 // (op 7247) most significant: sum


s_mov_b32 s68, s40 // (op 7247) reg expression
s_mov_b32 s69, s41 // (op 7247) reg expression
s_mov_b32 s70, s0 // (op 7247) reg expression
s_mov_b32 s71, 131072 // (op 7247) call()

s_mul_hi_i32 s47, s16, s10 // (op 7266) 
v_and_b32 v11, 63, v1 // (op 7266) 
s_add_i32 s0, s47, s16 // (op 7266) 

v_lshrrev_b32 v22, 4, v11 // (op 7266) 
s_ashr_i32 s1, s0, 31 // (op 7266) 
v_and_b32 v23, 3, v22 // (op 7266) 

s_cmp_eq_i32 s10, 0 // (op 7266) 
s_mov_b32 s47, scc // (op 7266) 
v_and_b32 v22, 15, v1 // (op 7266) 
s_cmp_eq_i32 s47, 0 // (op 7266) 
s_cselect_b32 s18, 0, -1 // (op 7266) 

v_lshl_add_u32 v24, v23, 4, v22 // (op 7266) 


s_and_b32 s19, s13, 31 // (op 7266) 
v_lshrrev_b32 v22, 6, v1 // (op 7266) 
s_lshl_b32 s47, 1, s19 // (op 7266) 
v_and_b32 v23, 1, v22 // (op 7266) 

s_add_i32 s22, s47, s18 // (op 7266) 


v_lshl_add_u32 v22, v23, 6, v24 // (op 7266) 


s_and_b32 s18, s1, s22 // (op 7266) 


v_lshrrev_b32 v23, 6, v11 // (op 7266) 

s_add_i32 s1, s0, s18 // (op 7266) 


s_ashr_i32 s0, s1, s19 // (op 7266) 


s_lshl_b32 s1, s13, 24 // (op 7266) 
s_ashr_i32 s47, s1, 31 // (op 7266) 

s_xor_b32 s1, s0, s47 // (op 7266) 

s_sub_i32 s0, s1, s47 // (op 7266) 


s_mul_hi_i32 s1, s0, s14 // (op 7266) 
s_add_i32 s47, s1, s0 // (op 7266) 

s_ashr_i32 s1, s47, 31 // (op 7266) 
s_cmp_eq_i32 s14, 0 // (op 7266) 
s_mov_b32 s18, scc // (op 7266) 
s_cmp_eq_i32 s18, 0 // (op 7266) 
s_cselect_b32 s19, 0, -1 // (op 7266) 

s_and_b32 s18, s15, 31 // (op 7266) 
s_lshl_b32 s22, 1, s18 // (op 7266) 
s_add_i32 s23, s22, s19 // (op 7266) 


s_and_b32 s19, s1, s23 // (op 7266) 


s_add_i32 s1, s47, s19 // (op 7266) 


s_ashr_i32 s19, s1, s18 // (op 7266) 


s_lshl_b32 s1, s15, 24 // (op 7266) 
s_ashr_i32 s18, s1, 31 // (op 7266) 

s_xor_b32 s1, s19, s18 // (op 7266) 

s_sub_i32 s19, s1, s18 // (op 7266) 


s_add_u32 s22, s6, 128 // (op 7266) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 7266) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 7266) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 7266) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 7266) 

s_mul_i32 s1, s19, s22 // (op 7266) 

s_sub_i32 s47, s0, s1 // (op 7266) 


s_mul_hi_i32 s0, s16, s9 // (op 7266) 
s_add_i32 s1, s0, s16 // (op 7266) 

s_ashr_i32 s0, s1, 31 // (op 7266) 
s_cmp_eq_i32 s9, 0 // (op 7266) 
s_mov_b32 s18, scc // (op 7266) 
s_cmp_eq_i32 s18, 0 // (op 7266) 
s_cselect_b32 s19, 0, -1 // (op 7266) 

s_and_b32 s18, s45, 31 // (op 7266) 
s_lshl_b32 s24, 1, s18 // (op 7266) 
s_add_i32 s25, s24, s19 // (op 7266) 


s_and_b32 s19, s0, s25 // (op 7266) 


s_add_i32 s0, s1, s19 // (op 7266) 


s_ashr_i32 s1, s0, s18 // (op 7266) 


s_lshl_b32 s0, s45, 24 // (op 7266) 
s_ashr_i32 s18, s0, 31 // (op 7266) 

s_xor_b32 s0, s1, s18 // (op 7266) 

s_sub_i32 s1, s0, s18 // (op 7266) 


s_mul_i32 s0, s11, s22 // (op 7266) 

s_mul_i32 s18, s8, s0 // (op 7266) 

s_mul_i32 s0, s1, s18 // (op 7266) 

s_sub_u32 s18, s16, s0 // (op 7266) 

s_mul_hi_i32 s0, s18, s12 // (op 7266) 
s_add_i32 s19, s0, s18 // (op 7266) 


s_ashr_i32 s0, s19, 31 // (op 7266) 
s_cmp_eq_i32 s12, 0 // (op 7266) 
s_mov_b32 s18, scc // (op 7266) 
s_cmp_eq_i32 s18, 0 // (op 7266) 
s_cselect_b32 s22, 0, -1 // (op 7266) 

s_and_b32 s18, s44, 31 // (op 7266) 
s_lshl_b32 s23, 1, s18 // (op 7266) 
s_add_i32 s24, s23, s22 // (op 7266) 


s_and_b32 s25, s0, s24 // (op 7266) 


s_add_i32 s0, s19, s25 // (op 7266) 


s_ashr_i32 s19, s0, s18 // (op 7266) 


s_lshl_b32 s0, s44, 24 // (op 7266) 
s_ashr_i32 s18, s0, 31 // (op 7266) 

s_xor_b32 s0, s19, s18 // (op 7266) 

s_sub_i32 s19, s0, s18 // (op 7266) 


s_cmp_eq_i32 s1, 0 // (op 7266) 
s_mov_b32 s0, scc // (op 7266) 

s_cmp_eq_i32 s0, 0 // (op 7266) 
s_cselect_b32 s1, s19, s47 // (op 7266) 



v_lshl_add_u32 v11, s1, 7, v22 // (op 7266) 


v_mul_lo_u32 v22, v11, s38 // (op 7266) 

v_lshl_add_u32 v11, v23, 2, v22 // (op 7266) 


s_mul_i32 s1, s38, s7 // (op 7268) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 7268) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 7268) least significant
s_mul_hi_u32 s18, s38, s6 // (op 7268) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 7268) most significant: sum
s_add_u32 s1, s1, s18 // (op 7268) most significant: sum


s_mov_b32 s68, s40 // (op 7268) reg expression
s_mov_b32 s69, s41 // (op 7268) reg expression
s_mov_b32 s70, s0 // (op 7268) reg expression
s_mov_b32 s71, 131072 // (op 7268) call()

v_and_b32 v22, -4, v1 // (op 7043) 
v_sub_u32 v23, v1, v22 // (op 7043) 

v_lshlrev_b32 v22, 5, v23 // (op 7043) 

v_lshrrev_b32 v23, 2, v1 // (op 7043) 
v_lshl_add_u32 v24, v23, 7, v22 // (op 7043) 


v_lshlrev_b32 v22, 2, v24 // (op 7043) 

v_lshrrev_b32 v23, 3, v22 // (op 7043) 

s_nop 0
v_readfirstlane_b32 s47, v23 // (op 7043) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 6 // (op 6766) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 6766) 

v_lshrrev_b64 v[22:23], 3, s[18:19] // (op 6766) 


v_mov_b64 v[24:25], v[12:13] // (op 6766) 
s_mov_b32 m0, s47 // (op 6766) reg expression
buffer_load_dwordx4 v24, s[60:63], 0 offen offset: 0 lds // (op 6766) Load value direct to lds
v_add_co_u32 v24, s[0:1], v24, v22 // (op 6766) least significant half
v_addc_co_u32 v25, s[0:1], 0, v25, s[0:1] // (op 6766) most significant half

s_add_u32 m0, m0, 4096 // (op 6766) 
buffer_load_dwordx4 v24, s[60:63], 0 offen offset: 0 lds // (op 6766) Load value direct to lds



v_and_b32 v22, -8, v1 // (op 6986) 
v_sub_u32 v23, v1, v22 // (op 6986) 

v_lshlrev_b32 v22, 4, v23 // (op 6986) 

v_lshrrev_b32 v23, 3, v1 // (op 6986) 
v_lshl_add_u32 v24, v23, 7, v22 // (op 6986) 


s_nop 0
v_readfirstlane_b32 s47, v24 // (op 6986) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 5 // (op 6771) 
v_mov_b64 v[22:23], s[0:1] // (op 6771) convert


v_mov_b64 v[24:25], v[14:15] // (op 6771) 
s_add_u32 s0, 8192, s47 // (op 6771) 
s_mov_b32 m0, s0 // (op 6771) reg expression
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 6771) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v22 // (op 6771) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 6771) most significant half

s_add_u32 m0, m0, 4096 // (op 6771) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 6771) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v22 // (op 6771) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 6771) most significant half

s_add_u32 m0, m0, 4096 // (op 6771) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 6771) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v22 // (op 6771) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 6771) most significant half

s_add_u32 m0, m0, 4096 // (op 6771) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 6771) Load value direct to lds






v_mov_b32 v22, v10 // (op 76) reg expression

buffer_load_dwordx2 v[24:25], v22, s[68:71], 0 offen // (op 76) Load value




v_mov_b32 v10, v20 // (op 72) reg expression

buffer_load_dwordx2 v[22:23], v10, s[64:67], 0 offen // (op 72) Load value


v_and_b32 v10, -4, v1 // (op 7154) 
v_sub_u32 v20, v1, v10 // (op 7154) 

v_lshlrev_b32 v10, 5, v20 // (op 7154) 

v_lshrrev_b32 v20, 2, v1 // (op 7154) 
v_lshl_add_u32 v26, v20, 7, v10 // (op 7154) 


v_lshlrev_b32 v10, 2, v26 // (op 7154) 

v_lshrrev_b32 v20, 3, v10 // (op 7154) 

s_nop 0
v_readfirstlane_b32 s47, v20 // (op 7154) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 6 // (op 6776) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 6776) 

v_lshrrev_b64 v[26:27], 3, s[18:19] // (op 6776) 


v_mov_b64 v[28:29], v[12:13] // (op 6776) 
s_add_u32 s0, 24576, s47 // (op 6776) 
s_mov_b32 m0, s0 // (op 6776) reg expression
buffer_load_dwordx4 v28, s[60:63], 64 offen offset: 0 lds // (op 6776) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 6776) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 6776) most significant half

s_add_u32 m0, m0, 4096 // (op 6776) 
buffer_load_dwordx4 v28, s[60:63], 64 offen offset: 0 lds // (op 6776) Load value direct to lds





v_and_b32 v10, -8, v1 // (op 7110) 
v_sub_u32 v20, v1, v10 // (op 7110) 

v_lshlrev_b32 v10, 4, v20 // (op 7110) 

v_lshrrev_b32 v20, 3, v1 // (op 7110) 
v_lshl_add_u32 v12, v20, 7, v10 // (op 7110) 


s_nop 0
v_readfirstlane_b32 s47, v12 // (op 7110) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 5 // (op 6778) 
v_mov_b64 v[12:13], s[0:1] // (op 6778) convert


v_mov_b64 v[26:27], v[14:15] // (op 6778) 
s_add_u32 s0, 32768, s47 // (op 6778) 
s_mov_b32 m0, s0 // (op 6778) reg expression
s_mov_b32 s1, 128 // (op 6778) call()
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 6778) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 6778) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 6778) most significant half

s_add_u32 m0, m0, 4096 // (op 6778) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 6778) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 6778) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 6778) most significant half

s_add_u32 m0, m0, 4096 // (op 6778) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 6778) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 6778) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 6778) most significant half

s_add_u32 m0, m0, 4096 // (op 6778) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 6778) Load value direct to lds






s_waitcnt vmcnt(12)
s_barrier  // (op 7454) 



ds_read_b128 v[12:15], v4 // (op 6514) Load local data 



ds_read_b128 v[26:29], v4 offset:1024 // (op 6519) Load local data 



ds_read_b128 v[30:33], v4 offset:2048 // (op 6524) Load local data 



ds_read_b128 v[34:37], v4 offset:3072 // (op 6529) Load local data 
s_waitcnt vmcnt(8)
s_barrier  // (op 7438) 



ds_read_b128 v[38:41], v3 offset:8192 // (op 6494) Load local data 
ds_read_b128 v[42:45], v3 offset:8256 // (op 6494) Load local data 



ds_read_b128 v[46:49], v3 offset:10240 // (op 6499) Load local data 
ds_read_b128 v[50:53], v3 offset:10304 // (op 6499) Load local data 



ds_read_b128 v[54:57], v3 offset:12288 // (op 6504) Load local data 
ds_read_b128 v[58:61], v3 offset:12352 // (op 6504) Load local data 



ds_read_b128 v[62:65], v3 offset:14336 // (op 6509) Load local data 
ds_read_b128 v[66:69], v3 offset:14400 // (op 6509) Load local data 
v_and_b32 v10, -8, v1 // (op 6971) 
v_sub_u32 v20, v1, v10 // (op 6971) 

v_lshlrev_b32 v10, 4, v20 // (op 6971) 

v_lshrrev_b32 v20, 3, v1 // (op 6971) 
v_lshl_add_u32 v70, v20, 7, v10 // (op 6971) 


s_nop 0
v_readfirstlane_b32 s47, v70 // (op 6971) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -4, v1 // (op 7028) 
v_sub_u32 v20, v1, v10 // (op 7028) 

v_lshlrev_b32 v10, 5, v20 // (op 7028) 

v_lshrrev_b32 v20, 2, v1 // (op 7028) 
v_lshl_add_u32 v70, v20, 7, v10 // (op 7028) 


v_lshlrev_b32 v10, 2, v70 // (op 7028) 

v_lshrrev_b32 v20, 3, v10 // (op 7028) 

s_nop 0
v_readfirstlane_b32 s0, v20 // (op 7028) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -8, v1 // (op 7095) 
v_sub_u32 v20, v1, v10 // (op 7095) 

v_lshlrev_b32 v10, 4, v20 // (op 7095) 

v_lshrrev_b32 v20, 3, v1 // (op 7095) 
v_lshl_add_u32 v70, v20, 7, v10 // (op 7095) 


s_nop 0
v_readfirstlane_b32 s1, v70 // (op 7095) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v10, -4, v1 // (op 7139) 
v_sub_u32 v20, v1, v10 // (op 7139) 

v_lshlrev_b32 v10, 5, v20 // (op 7139) 

v_lshrrev_b32 v20, 2, v1 // (op 7139) 
v_lshl_add_u32 v70, v20, 7, v10 // (op 7139) 


v_lshlrev_b32 v10, 2, v70 // (op 7139) 

v_lshrrev_b32 v20, 3, v10 // (op 7139) 

s_nop 0
v_readfirstlane_b32 s18, v20 // (op 7139) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s19, 0 // (op 2080) call()
s_ashr_i64 s[22:23], s[42:43], 7 // (op 2079) 
s_mov_b32 s24, s22 // (op 2079) convert

s_and_b32 s25, s24, 4294967294 // (op 2079) 

s_cmp_lt_u32 s19, s25 // (op 2079) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ForLoopBottom_KLoop_2079 // (op 2079) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ForLoopBottom_KLoop_2079 if false)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ForLoopTop_KLoop_2079:
s_add_u32 s19, s19, 2 // (op 2081) 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 7457) 
s_waitcnt vmcnt(7)
v_mov_b32 v70, v24 // (op 6690) reg expression

v_mov_b32 v71, v25 // (op 6690) reg expression
s_lshl_b64 s[22:23], s[30:31], 6 // (op 6739) 
s_lshl_b64 s[24:25], s[22:23], 2 // (op 6739) 

v_lshrrev_b64 v[72:73], 3, s[24:25] // (op 6739) 


v_mov_b64 v[74:75], v[18:19] // (op 6739) 
v_mov_b64 v[76:77], v[74:75] // (op 6739) 
s_mov_b32 m0, s0 // (op 6739) reg expression
s_mov_b32 s22, 128 // (op 6739) call()
buffer_load_dwordx4 v76, s[60:63], s22 offen offset: 0 lds // (op 6739) Load value direct to lds
v_add_co_u32 v76, s[24:25], v76, v72 // (op 6739) least significant half
v_addc_co_u32 v77, s[24:25], 0, v77, s[24:25] // (op 6739) most significant half

s_add_u32 m0, m0, 4096 // (op 6739) 
s_waitcnt vmcnt(7)
v_mov_b32 v78, v22 // (op 6703) reg expression
v_mov_b32 v79, v23 // (op 6703) reg expression
buffer_load_dwordx4 v76, s[60:63], s22 offen offset: 0 lds // (op 6739) Load value direct to lds



v_bfe_u32 v80, v78, 0, 8 // (op 6114) 
v_bfe_u32 v81, v78, 8, 8 // (op 6114) 
v_bfe_u32 v82, v78, 16, 8 // (op 6114) 
v_bfe_u32 v83, v78, 24, 8 // (op 6114) 
v_permlane16_swap_b32 v80, v81 // (op 6114) 
s_nop 0
v_permlane16_swap_b32 v82, v83 // (op 6114) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v80, v82 // (op 6114) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v81, v83 // (op 6114) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 7441) 

s_lshl_b64 s[22:23], s[26:27], 5 // (op 6734) 
v_mov_b64 v[72:73], s[22:23] // (op 6734) convert


v_mov_b64 v[76:77], v[16:17] // (op 6734) 
v_mov_b64 v[84:85], v[76:77] // (op 6734) 
s_add_u32 s22, 8192, s47 // (op 6734) 
s_mov_b32 m0, s22 // (op 6734) reg expression
s_mov_b32 s23, 256 // (op 6734) call()
buffer_load_dwordx4 v84, s[48:51], s23 offen offset: 0 lds // (op 6734) Load value direct to lds
v_add_co_u32 v84, s[24:25], v84, v72 // (op 6734) least significant half
v_addc_co_u32 v85, s[24:25], 0, v85, s[24:25] // (op 6734) most significant half

s_add_u32 m0, m0, 4096 // (op 6734) 
v_bfe_u32 v86, v70, 0, 8 // (op 6177) 
v_bfe_u32 v87, v70, 8, 8 // (op 6177) 
v_bfe_u32 v88, v70, 16, 8 // (op 6177) 
v_bfe_u32 v89, v70, 24, 8 // (op 6177) 
v_permlane16_swap_b32 v86, v87 // (op 6177) 
s_nop 0
v_permlane16_swap_b32 v88, v89 // (op 6177) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v86, v88 // (op 6177) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v87, v89 // (op 6177) 
buffer_load_dwordx4 v84, s[48:51], s23 offen offset: 0 lds // (op 6734) Load value direct to lds
v_add_co_u32 v84, s[24:25], v84, v72 // (op 6734) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:45], v[12:15], a[0:3], v80, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 78) 
v_addc_co_u32 v85, s[24:25], 0, v85, s[24:25] // (op 6734) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[38:45], v[26:29], a[4:7], v80, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 315) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[38:45], v[30:33], a[8:11], v80, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 451) 
s_add_u32 m0, m0, 4096 // (op 6734) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[38:45], v[34:37], a[12:15], v80, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 587) 
buffer_load_dwordx4 v84, s[48:51], s23 offen offset: 0 lds // (op 6734) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[46:53], v[12:15], a[16:19], v81, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1007) 
v_add_co_u32 v84, s[24:25], v84, v72 // (op 6734) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[46:53], v[26:29], a[20:23], v81, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 969) 
v_addc_co_u32 v85, s[24:25], 0, v85, s[24:25] // (op 6734) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[46:53], v[30:33], a[24:27], v81, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 931) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[46:53], v[34:37], a[28:31], v81, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 893) 
s_add_u32 m0, m0, 4096 // (op 6734) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[54:61], v[12:15], a[32:35], v82, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1617) 
buffer_load_dwordx4 v84, s[48:51], s23 offen offset: 0 lds // (op 6734) Load value direct to lds






v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[54:61], v[26:29], a[36:39], v82, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1579) 
v_mov_b32 v10, v11 // (op 6698) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[54:61], v[30:33], a[40:43], v82, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1541) 
v_add_u32 v20, 8, v10 // (op 6698) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[54:61], v[34:37], a[44:47], v82, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1503) 
s_waitcnt vmcnt(10)
s_barrier  // (op 7476) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[62:69], v[12:15], a[48:51], v83, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2227) 
ds_read_b128 v[90:93], v6 offset:24576 // (op 216) Load local data 
buffer_load_dwordx2 v[24:25], v20, s[68:71], 0 offen // (op 6698) Load value





v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[62:69], v[26:29], a[52:55], v83, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2189) 
v_add_u32 v11, 8, v11 // (op 7256) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[62:69], v[30:33], a[56:59], v83, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2151) 
v_mov_b32 v10, v21 // (op 6711) 
v_add_u32 v20, 8, v10 // (op 6711) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[62:69], v[34:37], a[60:63], v83, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2113) 
ds_read_b128 v[94:97], v6 offset:25600 // (op 324) Load local data 





ds_read_b128 v[80:83], v6 offset:26624 // (op 460) Load local data 
buffer_load_dwordx2 v[22:23], v20, s[64:67], 0 offen // (op 6711) Load value





v_add_u32 v21, 8, v21 // (op 7204) 
ds_read_b128 v[84:87], v6 offset:27648 // (op 596) Load local data 
s_waitcnt vmcnt(8)
s_barrier  // (op 7470) 



ds_read_b128 v[38:41], v5 offset:32768 // (op 214) Load local data 
ds_read_b128 v[42:45], v5 offset:32832 // (op 214) Load local data 



ds_read_b128 v[46:49], v5 offset:34816 // (op 901) Load local data 
ds_read_b128 v[50:53], v5 offset:34880 // (op 901) Load local data 



ds_read_b128 v[54:57], v5 offset:36864 // (op 1511) Load local data 
ds_read_b128 v[58:61], v5 offset:36928 // (op 1511) Load local data 



ds_read_b128 v[98:101], v5 offset:38912 // (op 2121) Load local data 
ds_read_b128 v[102:105], v5 offset:38976 // (op 2121) Load local data 
v_bfe_u32 v106, v79, 0, 8 // (op 6129) 
v_bfe_u32 v107, v79, 8, 8 // (op 6129) 
v_bfe_u32 v108, v79, 16, 8 // (op 6129) 
v_bfe_u32 v109, v79, 24, 8 // (op 6129) 
v_permlane16_swap_b32 v106, v107 // (op 6129) 
s_nop 0
v_permlane16_swap_b32 v108, v109 // (op 6129) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v106, v108 // (op 6129) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v107, v109 // (op 6129) 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 7480) 


s_lshl_b64 s[22:23], s[30:31], 6 // (op 6761) 
s_lshl_b64 s[24:25], s[22:23], 2 // (op 6761) 

v_lshrrev_b64 v[72:73], 3, s[24:25] // (op 6761) 


v_mov_b64 v[78:79], v[74:75] // (op 6761) 
s_add_u32 s22, 24576, s18 // (op 6761) 
s_mov_b32 m0, s22 // (op 6761) reg expression
s_mov_b32 s23, 192 // (op 6761) call()
buffer_load_dwordx4 v78, s[60:63], s23 offen offset: 0 lds // (op 6761) Load value direct to lds
v_add_co_u32 v78, s[24:25], v78, v72 // (op 6761) least significant half
v_addc_co_u32 v79, s[24:25], 0, v79, s[24:25] // (op 6761) most significant half

s_add_u32 m0, m0, 4096 // (op 6761) 
v_bfe_u32 v110, v71, 0, 8 // (op 6192) 
v_bfe_u32 v111, v71, 8, 8 // (op 6192) 
v_bfe_u32 v112, v71, 16, 8 // (op 6192) 
v_bfe_u32 v113, v71, 24, 8 // (op 6192) 
v_permlane16_swap_b32 v110, v111 // (op 6192) 
s_nop 0
v_permlane16_swap_b32 v112, v113 // (op 6192) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v110, v112 // (op 6192) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v111, v113 // (op 6192) 
buffer_load_dwordx4 v78, s[60:63], s23 offen offset: 0 lds // (op 6761) Load value direct to lds






s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:45], v[90:93], a[0:3], v106, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 221) 
v_add_co_u32 v18, vcc, 128, v18 // (op 6907) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[38:45], v[94:97], a[4:7], v106, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 316) 
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 6907) most significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[38:45], v[80:83], a[8:11], v106, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 452) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 7473) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[38:45], v[84:87], a[12:15], v106, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 588) 
s_lshl_b64 s[22:23], s[26:27], 5 // (op 6756) 
v_mov_b64 v[78:79], s[22:23] // (op 6756) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[46:53], v[90:93], a[16:19], v107, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1008) 
v_mov_b64 v[88:89], v[76:77] // (op 6756) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[46:53], v[94:97], a[20:23], v107, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 970) 
s_add_u32 s22, 32768, s1 // (op 6756) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[46:53], v[80:83], a[24:27], v107, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 932) 
s_mov_b32 m0, s22 // (op 6756) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[46:53], v[84:87], a[28:31], v107, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 894) 
s_mov_b32 s23, 384 // (op 6756) call()
s_waitcnt vmcnt(8)
s_barrier  // (op 7463) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[54:61], v[90:93], a[32:35], v108, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1618) 
ds_read_b128 v[12:15], v4 // (op 67) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[54:61], v[94:97], a[36:39], v108, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1580) 
buffer_load_dwordx4 v88, s[48:51], s23 offen offset: 0 lds // (op 6756) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[54:61], v[80:83], a[40:43], v108, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1542) 
v_add_co_u32 v88, s[24:25], v88, v78 // (op 6756) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[54:61], v[84:87], a[44:47], v108, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1504) 
v_addc_co_u32 v89, s[24:25], 0, v89, s[24:25] // (op 6756) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[98:105], v[90:93], a[48:51], v109, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2228) 
s_add_u32 m0, m0, 4096 // (op 6756) 




v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[98:105], v[94:97], a[52:55], v109, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2190) 
ds_read_b128 v[26:29], v4 offset:1024 // (op 312) Load local data 

buffer_load_dwordx4 v88, s[48:51], s23 offen offset: 0 lds // (op 6756) Load value direct to lds



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[98:105], v[80:83], a[56:59], v109, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2152) 
ds_read_b128 v[30:33], v4 offset:2048 // (op 448) Load local data 

v_add_co_u32 v88, s[24:25], v88, v78 // (op 6756) least significant half



v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[98:105], v[84:87], a[60:63], v109, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2114) 
ds_read_b128 v[34:37], v4 offset:3072 // (op 584) Load local data 




v_addc_co_u32 v89, s[24:25], 0, v89, s[24:25] // (op 6756) most significant half

s_add_u32 m0, m0, 4096 // (op 6756) 
s_waitcnt vmcnt(6)
s_barrier  // (op 7447) 
buffer_load_dwordx4 v88, s[48:51], s23 offen offset: 0 lds // (op 6756) Load value direct to lds



ds_read_b128 v[38:41], v3 offset:8192 // (op 62) Load local data 
ds_read_b128 v[42:45], v3 offset:8256 // (op 62) Load local data 
v_add_co_u32 v88, s[24:25], v88, v78 // (op 6756) least significant half



ds_read_b128 v[46:49], v3 offset:10240 // (op 889) Load local data 
v_addc_co_u32 v89, s[24:25], 0, v89, s[24:25] // (op 6756) most significant half

s_add_u32 m0, m0, 4096 // (op 6756) 
ds_read_b128 v[50:53], v3 offset:10304 // (op 889) Load local data 
buffer_load_dwordx4 v88, s[48:51], s23 offen offset: 0 lds // (op 6756) Load value direct to lds








ds_read_b128 v[54:57], v3 offset:12288 // (op 1499) Load local data 
ds_read_b128 v[58:61], v3 offset:12352 // (op 1499) Load local data 
v_add_co_u32 v16, vcc, 256, v16 // (op 6843) least significant half



v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 6843) most significant half
ds_read_b128 v[62:65], v3 offset:14336 // (op 2109) Load local data 
ds_read_b128 v[66:69], v3 offset:14400 // (op 2109) Load local data 
s_ashr_i64 s[22:23], s[42:43], 7 // (op 2079) 
s_mov_b32 s24, s22 // (op 2079) convert

s_and_b32 s25, s24, 4294967294 // (op 2079) 

s_cmp_lt_u32 s19, s25 // (op 2079) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ForLoopTop_KLoop_2079 // (op 2079) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ForLoopTop_KLoop_2079 if true)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ForLoopBottom_KLoop_2079:











s_barrier  // (op 7544) 
s_mul_hi_i32 s47, s16, s10 // (op 6791) 
v_lshrrev_b32 v10, 3, v1 // (op 6791) 
s_add_i32 s0, s47, s16 // (op 6791) 

v_and_b32 v11, -8, v1 // (op 6791) 
s_ashr_i32 s1, s0, 31 // (op 6791) 
v_sub_u32 v16, v1, v11 // (op 6791) 

s_cmp_eq_i32 s10, 0 // (op 6791) 
s_mov_b32 s47, scc // (op 6791) 
v_lshlrev_b32 v11, 4, v16 // (op 6791) 

s_cmp_eq_i32 s47, 0 // (op 6791) 
s_cselect_b32 s18, 0, -1 // (op 6791) 

s_and_b32 s19, s13, 31 // (op 6791) 
s_lshl_b32 s47, 1, s19 // (op 6791) 
s_add_i32 s22, s47, s18 // (op 6791) 


s_and_b32 s18, s1, s22 // (op 6791) 


s_add_i32 s1, s0, s18 // (op 6791) 


s_ashr_i32 s0, s1, s19 // (op 6791) 


s_lshl_b32 s1, s13, 24 // (op 6791) 
s_ashr_i32 s47, s1, 31 // (op 6791) 

s_xor_b32 s1, s0, s47 // (op 6791) 

s_sub_i32 s0, s1, s47 // (op 6791) 


s_mul_hi_i32 s1, s0, s14 // (op 6791) 
s_add_i32 s47, s1, s0 // (op 6791) 

s_ashr_i32 s1, s47, 31 // (op 6791) 
s_cmp_eq_i32 s14, 0 // (op 6791) 
s_mov_b32 s18, scc // (op 6791) 
s_cmp_eq_i32 s18, 0 // (op 6791) 
s_cselect_b32 s19, 0, -1 // (op 6791) 

s_and_b32 s18, s15, 31 // (op 6791) 
s_lshl_b32 s22, 1, s18 // (op 6791) 
s_add_i32 s23, s22, s19 // (op 6791) 


s_and_b32 s19, s1, s23 // (op 6791) 


s_add_i32 s1, s47, s19 // (op 6791) 


s_ashr_i32 s19, s1, s18 // (op 6791) 


s_lshl_b32 s1, s15, 24 // (op 6791) 
s_ashr_i32 s18, s1, 31 // (op 6791) 

s_xor_b32 s1, s19, s18 // (op 6791) 

s_sub_i32 s19, s1, s18 // (op 6791) 


s_mul_i32 s1, s0, s11 // (op 6791) 

s_sub_u32 s0, s16, s1 // (op 6791) 

s_mul_i32 s1, s19, s11 // (op 6791) 
s_add_u32 s1, s1, s0 // (op 6791) 


s_mul_hi_i32 s0, s16, s9 // (op 6791) 
s_add_i32 s47, s0, s16 // (op 6791) 

s_ashr_i32 s0, s47, 31 // (op 6791) 
s_cmp_eq_i32 s9, 0 // (op 6791) 
s_mov_b32 s18, scc // (op 6791) 
s_cmp_eq_i32 s18, 0 // (op 6791) 
s_cselect_b32 s19, 0, -1 // (op 6791) 

s_and_b32 s18, s45, 31 // (op 6791) 
s_lshl_b32 s22, 1, s18 // (op 6791) 
s_add_i32 s23, s22, s19 // (op 6791) 


s_and_b32 s19, s0, s23 // (op 6791) 


s_add_i32 s0, s47, s19 // (op 6791) 


s_ashr_i32 s19, s0, s18 // (op 6791) 


s_lshl_b32 s0, s45, 24 // (op 6791) 
s_ashr_i32 s18, s0, 31 // (op 6791) 

s_xor_b32 s0, s19, s18 // (op 6791) 

s_sub_i32 s19, s0, s18 // (op 6791) 


s_add_u32 s22, s6, 128 // (op 6791) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 6791) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 6791) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 6791) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 6791) 

s_mul_i32 s0, s11, s22 // (op 6791) 

s_mul_i32 s18, s8, s0 // (op 6791) 

s_mul_i32 s0, s19, s18 // (op 6791) 

s_sub_u32 s18, s16, s0 // (op 6791) 

s_mov_b32 s0, s18 // (op 6791) 
s_mul_hi_i32 s47, s0, s12 // (op 6791) 
s_add_i32 s22, s47, s0 // (op 6791) 


s_ashr_i32 s0, s22, 31 // (op 6791) 
s_cmp_eq_i32 s12, 0 // (op 6791) 
s_mov_b32 s47, scc // (op 6791) 
s_cmp_eq_i32 s47, 0 // (op 6791) 
s_cselect_b32 s23, 0, -1 // (op 6791) 

s_and_b32 s47, s44, 31 // (op 6791) 
s_lshl_b32 s24, 1, s47 // (op 6791) 
s_add_i32 s25, s24, s23 // (op 6791) 


s_and_b32 s23, s0, s25 // (op 6791) 


s_add_i32 s0, s22, s23 // (op 6791) 


s_ashr_i32 s22, s0, s47 // (op 6791) 


s_lshl_b32 s0, s44, 24 // (op 6791) 
s_ashr_i32 s47, s0, 31 // (op 6791) 

s_xor_b32 s0, s22, s47 // (op 6791) 

s_sub_i32 s22, s0, s47 // (op 6791) 


s_mul_i32 s0, s46, s11 // (op 6791) 
s_add_u32 s24, s20, 128 // (op 6791) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 6791) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 6791) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 6791) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 7 // (op 6791) 

s_sub_i32 s23, s24, s0 // (op 6791) 

s_mul_i32 s47, s22, s23 // (op 6791) 


s_sub_u32 s22, s18, s47 // (op 6791) 


s_add_u32 s18, s22, s0 // (op 6791) 


s_cmp_eq_i32 s19, 0 // (op 6791) 
s_mov_b32 s0, scc // (op 6791) 

s_cmp_eq_i32 s0, 0 // (op 6791) 
s_cselect_b32 s19, s18, s1 // (op 6791) 



v_lshl_add_u32 v16, s19, 7, v10 // (op 6791) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 6791) 
s_lshr_b32 s47, s0, 1 // (op 6791) 

v_lshl_add_u32 v10, s47, 8, v11 // (op 6791) 


v_mov_b32 v21, 0 // (op 6791) convert
v_mov_b32 v20, v10 // (op 6791) convert

v_mul_lo_u32 v11, v16, s27 // (op 6791) most significant: low of low * high
v_mul_hi_u32 v17, v16, s26 // (op 6791) most significant: high of low * low
v_mul_lo_u32 v10, v16, s26 // (op 6791) least significant: low of low * low
v_add_u32 v11, v11, v17 // (op 6791) most significant: sum

v_add_co_u32 v10, s[0:1], v10, v20 // (op 6791) least significant half
v_addc_co_u32 v11, s[0:1], v11, v21, s[0:1] // (op 6791) most significant half



s_mul_i32 s1, s26, s21 // (op 6793) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 6793) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 6793) least significant
s_mul_hi_u32 s18, s26, s20 // (op 6793) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 6793) most significant: sum
s_add_u32 s1, s1, s18 // (op 6793) most significant: sum


s_mov_b32 s48, s28 // (op 6793) reg expression
s_mov_b32 s49, s29 // (op 6793) reg expression
s_mov_b32 s50, s0 // (op 6793) reg expression
s_mov_b32 s51, 131072 // (op 6793) call()

s_mul_hi_i32 s47, s16, s10 // (op 6873) 
v_lshrrev_b32 v16, 2, v1 // (op 6873) 
s_add_i32 s0, s47, s16 // (op 6873) 

v_and_b32 v17, -4, v1 // (op 6873) 
s_ashr_i32 s1, s0, 31 // (op 6873) 
v_sub_u32 v18, v1, v17 // (op 6873) 

s_cmp_eq_i32 s10, 0 // (op 6873) 
s_mov_b32 s47, scc // (op 6873) 
v_lshlrev_b32 v17, 5, v18 // (op 6873) 

s_cmp_eq_i32 s47, 0 // (op 6873) 
s_cselect_b32 s18, 0, -1 // (op 6873) 

s_and_b32 s19, s13, 31 // (op 6873) 
s_lshl_b32 s47, 1, s19 // (op 6873) 
s_add_i32 s22, s47, s18 // (op 6873) 


s_and_b32 s18, s1, s22 // (op 6873) 


s_add_i32 s1, s0, s18 // (op 6873) 


s_ashr_i32 s0, s1, s19 // (op 6873) 


s_lshl_b32 s1, s13, 24 // (op 6873) 
s_ashr_i32 s47, s1, 31 // (op 6873) 

s_xor_b32 s1, s0, s47 // (op 6873) 

s_sub_i32 s0, s1, s47 // (op 6873) 


s_mul_hi_i32 s1, s0, s14 // (op 6873) 
s_add_i32 s47, s1, s0 // (op 6873) 

s_ashr_i32 s1, s47, 31 // (op 6873) 
s_cmp_eq_i32 s14, 0 // (op 6873) 
s_mov_b32 s18, scc // (op 6873) 
s_cmp_eq_i32 s18, 0 // (op 6873) 
s_cselect_b32 s19, 0, -1 // (op 6873) 

s_and_b32 s18, s15, 31 // (op 6873) 
s_lshl_b32 s22, 1, s18 // (op 6873) 
s_add_i32 s23, s22, s19 // (op 6873) 


s_and_b32 s19, s1, s23 // (op 6873) 


s_add_i32 s1, s47, s19 // (op 6873) 


s_ashr_i32 s19, s1, s18 // (op 6873) 


s_lshl_b32 s1, s15, 24 // (op 6873) 
s_ashr_i32 s18, s1, 31 // (op 6873) 

s_xor_b32 s1, s19, s18 // (op 6873) 

s_sub_i32 s19, s1, s18 // (op 6873) 


s_add_u32 s22, s6, 128 // (op 6873) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 6873) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 6873) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 6873) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 6873) 

s_mul_i32 s1, s19, s22 // (op 6873) 

s_sub_i32 s47, s0, s1 // (op 6873) 


s_mul_hi_i32 s0, s16, s9 // (op 6873) 
s_add_i32 s1, s0, s16 // (op 6873) 

s_ashr_i32 s0, s1, 31 // (op 6873) 
s_cmp_eq_i32 s9, 0 // (op 6873) 
s_mov_b32 s18, scc // (op 6873) 
s_cmp_eq_i32 s18, 0 // (op 6873) 
s_cselect_b32 s19, 0, -1 // (op 6873) 

s_and_b32 s18, s45, 31 // (op 6873) 
s_lshl_b32 s24, 1, s18 // (op 6873) 
s_add_i32 s25, s24, s19 // (op 6873) 


s_and_b32 s19, s0, s25 // (op 6873) 


s_add_i32 s0, s1, s19 // (op 6873) 


s_ashr_i32 s1, s0, s18 // (op 6873) 


s_lshl_b32 s0, s45, 24 // (op 6873) 
s_ashr_i32 s18, s0, 31 // (op 6873) 

s_xor_b32 s0, s1, s18 // (op 6873) 

s_sub_i32 s1, s0, s18 // (op 6873) 


s_mul_i32 s0, s11, s22 // (op 6873) 

s_mul_i32 s18, s8, s0 // (op 6873) 

s_mul_i32 s0, s1, s18 // (op 6873) 

s_sub_u32 s18, s16, s0 // (op 6873) 

s_mul_hi_i32 s0, s18, s12 // (op 6873) 
s_add_i32 s19, s0, s18 // (op 6873) 


s_ashr_i32 s0, s19, 31 // (op 6873) 
s_cmp_eq_i32 s12, 0 // (op 6873) 
s_mov_b32 s18, scc // (op 6873) 
s_cmp_eq_i32 s18, 0 // (op 6873) 
s_cselect_b32 s22, 0, -1 // (op 6873) 

s_and_b32 s18, s44, 31 // (op 6873) 
s_lshl_b32 s23, 1, s18 // (op 6873) 
s_add_i32 s24, s23, s22 // (op 6873) 


s_and_b32 s25, s0, s24 // (op 6873) 


s_add_i32 s0, s19, s25 // (op 6873) 


s_ashr_i32 s19, s0, s18 // (op 6873) 


s_lshl_b32 s0, s44, 24 // (op 6873) 
s_ashr_i32 s18, s0, 31 // (op 6873) 

s_xor_b32 s0, s19, s18 // (op 6873) 

s_sub_i32 s19, s0, s18 // (op 6873) 


s_cmp_eq_i32 s1, 0 // (op 6873) 
s_mov_b32 s0, scc // (op 6873) 

s_cmp_eq_i32 s0, 0 // (op 6873) 
s_cselect_b32 s1, s19, s47 // (op 6873) 



v_lshl_add_u32 v18, s1, 7, v16 // (op 6873) 


v_mul_lo_u32 v21, v18, s31 // (op 6873) most significant: low of low * high
v_mul_hi_u32 v16, v18, s30 // (op 6873) most significant: high of low * low
v_mul_lo_u32 v20, v18, s30 // (op 6873) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 6873) most significant: sum


s_ashr_i64 s[0:1], s[42:43], 7 // (op 6873) 
s_lshr_b32 s47, s0, 1 // (op 6873) 

v_lshl_add_u32 v16, s47, 8, v17 // (op 6873) 


v_mov_b32 v19, 0 // (op 6873) convert
v_mov_b32 v18, v16 // (op 6873) convert

v_add_co_u32 v16, s[0:1], v18, v20 // (op 6873) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 6873) most significant half

v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 6873) 


v_lshrrev_b64 v[18:19], 3, v[16:17] // (op 6873) 

s_mul_i32 s1, s30, s7 // (op 6875) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 6875) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 6875) least significant
s_mul_hi_u32 s18, s30, s6 // (op 6875) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 6875) most significant: sum
s_add_u32 s1, s1, s18 // (op 6875) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 6875) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 6875) 

s_mov_b32 s60, s32 // (op 6875) reg expression
s_mov_b32 s61, s33 // (op 6875) reg expression
s_mov_b32 s62, s0 // (op 6875) reg expression
s_mov_b32 s63, 131072 // (op 6875) call()

v_and_b32 v16, -8, v1 // (op 6956) 
v_sub_u32 v17, v1, v16 // (op 6956) 

v_lshlrev_b32 v16, 4, v17 // (op 6956) 

v_lshrrev_b32 v17, 3, v1 // (op 6956) 
v_lshl_add_u32 v20, v17, 7, v16 // (op 6956) 


s_nop 0
v_readfirstlane_b32 s47, v20 // (op 6956) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -4, v1 // (op 7015) 
v_sub_u32 v17, v1, v16 // (op 7015) 

v_lshlrev_b32 v16, 5, v17 // (op 7015) 

v_lshrrev_b32 v17, 2, v1 // (op 7015) 
v_lshl_add_u32 v20, v17, 7, v16 // (op 7015) 


v_lshlrev_b32 v21, 2, v20 // (op 7015) 

v_lshrrev_b32 v20, 3, v21 // (op 7015) 

s_nop 0
v_readfirstlane_b32 s0, v20 // (op 7015) Wait state hazard: VALU Write Readlane Hazard

s_mul_hi_i32 s1, s16, s10 // (op 7293) 
v_and_b32 v16, 63, v1 // (op 7293) 
s_add_i32 s18, s1, s16 // (op 7293) 

v_lshrrev_b32 v17, 4, v16 // (op 7293) 
s_ashr_i32 s1, s18, 31 // (op 7293) 
v_and_b32 v20, 3, v17 // (op 7293) 

s_cmp_eq_i32 s10, 0 // (op 7293) 
s_mov_b32 s19, scc // (op 7293) 
v_and_b32 v17, 15, v1 // (op 7293) 
s_cmp_eq_i32 s19, 0 // (op 7293) 
s_cselect_b32 s22, 0, -1 // (op 7293) 

v_lshl_add_u32 v21, v20, 4, v17 // (op 7293) 


s_and_b32 s19, s13, 31 // (op 7293) 
v_lshrrev_b32 v17, 7, v1 // (op 7293) 
s_lshl_b32 s23, 1, s19 // (op 7293) 
v_lshl_add_u32 v20, v17, 6, v21 // (op 7293) 


s_add_i32 s24, s23, s22 // (op 7293) 


v_lshrrev_b32 v17, 6, v16 // (op 7293) 

s_and_b32 s25, s1, s24 // (op 7293) 


v_lshlrev_b32 v16, 2, v17 // (op 7293) 

s_add_i32 s1, s18, s25 // (op 7293) 


s_ashr_i32 s18, s1, s19 // (op 7293) 


s_lshl_b32 s1, s13, 24 // (op 7293) 
s_ashr_i32 s19, s1, 31 // (op 7293) 

s_xor_b32 s1, s18, s19 // (op 7293) 

s_sub_i32 s18, s1, s19 // (op 7293) 


s_mul_hi_i32 s1, s18, s14 // (op 7293) 
s_add_i32 s19, s1, s18 // (op 7293) 

s_ashr_i32 s1, s19, 31 // (op 7293) 
s_cmp_eq_i32 s14, 0 // (op 7293) 
s_mov_b32 s22, scc // (op 7293) 
s_cmp_eq_i32 s22, 0 // (op 7293) 
s_cselect_b32 s23, 0, -1 // (op 7293) 

s_and_b32 s22, s15, 31 // (op 7293) 
s_lshl_b32 s24, 1, s22 // (op 7293) 
s_add_i32 s25, s24, s23 // (op 7293) 


s_and_b32 s23, s1, s25 // (op 7293) 


s_add_i32 s1, s19, s23 // (op 7293) 


s_ashr_i32 s19, s1, s22 // (op 7293) 


s_lshl_b32 s1, s15, 24 // (op 7293) 
s_ashr_i32 s22, s1, 31 // (op 7293) 

s_xor_b32 s1, s19, s22 // (op 7293) 

s_sub_i32 s19, s1, s22 // (op 7293) 


s_mul_i32 s1, s18, s11 // (op 7293) 

s_sub_u32 s18, s16, s1 // (op 7293) 

s_mul_i32 s1, s19, s11 // (op 7293) 
s_add_u32 s1, s1, s18 // (op 7293) 


s_mul_hi_i32 s18, s16, s9 // (op 7293) 
s_add_i32 s19, s18, s16 // (op 7293) 

s_ashr_i32 s18, s19, 31 // (op 7293) 
s_cmp_eq_i32 s9, 0 // (op 7293) 
s_mov_b32 s22, scc // (op 7293) 
s_cmp_eq_i32 s22, 0 // (op 7293) 
s_cselect_b32 s23, 0, -1 // (op 7293) 

s_and_b32 s22, s45, 31 // (op 7293) 
s_lshl_b32 s24, 1, s22 // (op 7293) 
s_add_i32 s25, s24, s23 // (op 7293) 


s_and_b32 s23, s18, s25 // (op 7293) 


s_add_i32 s18, s19, s23 // (op 7293) 


s_ashr_i32 s19, s18, s22 // (op 7293) 


s_lshl_b32 s18, s45, 24 // (op 7293) 
s_ashr_i32 s22, s18, 31 // (op 7293) 

s_xor_b32 s18, s19, s22 // (op 7293) 

s_sub_i32 s19, s18, s22 // (op 7293) 


s_add_u32 s22, s6, 128 // (op 7293) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 7293) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 7293) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 7293) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 7293) 

s_mul_i32 s18, s11, s22 // (op 7293) 

s_mul_i32 s22, s8, s18 // (op 7293) 

s_mul_i32 s18, s19, s22 // (op 7293) 

s_sub_u32 s22, s16, s18 // (op 7293) 

s_mov_b32 s18, s22 // (op 7293) 
s_mul_hi_i32 s23, s18, s12 // (op 7293) 
s_add_i32 s24, s23, s18 // (op 7293) 


s_ashr_i32 s18, s24, 31 // (op 7293) 
s_cmp_eq_i32 s12, 0 // (op 7293) 
s_mov_b32 s23, scc // (op 7293) 
s_cmp_eq_i32 s23, 0 // (op 7293) 
s_cselect_b32 s25, 0, -1 // (op 7293) 

s_and_b32 s23, s44, 31 // (op 7293) 
s_lshl_b32 s64, 1, s23 // (op 7293) 
s_add_i32 s65, s64, s25 // (op 7293) 


s_and_b32 s25, s18, s65 // (op 7293) 


s_add_i32 s18, s24, s25 // (op 7293) 


s_ashr_i32 s24, s18, s23 // (op 7293) 


s_lshl_b32 s18, s44, 24 // (op 7293) 
s_ashr_i32 s23, s18, 31 // (op 7293) 

s_xor_b32 s18, s24, s23 // (op 7293) 

s_sub_i32 s24, s18, s23 // (op 7293) 


s_mul_i32 s18, s46, s11 // (op 7293) 
s_add_u32 s64, s20, 128 // (op 7293) Start of Int64 add, locking SCC
s_addc_u32 s65, s21, 0 // (op 7293) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 7293) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 7293) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 7 // (op 7293) 

s_sub_i32 s23, s64, s18 // (op 7293) 

s_mul_i32 s25, s24, s23 // (op 7293) 


s_sub_u32 s23, s22, s25 // (op 7293) 


s_add_u32 s22, s23, s18 // (op 7293) 


s_cmp_eq_i32 s19, 0 // (op 7293) 
s_mov_b32 s18, scc // (op 7293) 

s_cmp_eq_i32 s18, 0 // (op 7293) 
s_cselect_b32 s19, s22, s1 // (op 7293) 



v_lshl_add_u32 v17, s19, 7, v20 // (op 7293) 


s_ashr_i64 s[18:19], s[42:43], 7 // (op 7293) 
s_lshr_b32 s1, s18, 1 // (op 7293) 

v_lshl_add_u32 v20, s1, 3, v16 // (op 7293) 


v_mul_lo_u32 v16, v17, s34 // (op 7293) 
v_add_u32 v16, v16, v20 // (op 7293) 


s_mul_i32 s19, s34, s21 // (op 7295) most significant: low of low * high
s_mul_i32 s1, s35, s20 // (op 7295) most significant: low of high * low
s_mul_i32 s18, s34, s20 // (op 7295) least significant
s_mul_hi_u32 s22, s34, s20 // (op 7295) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 7295) most significant: sum
s_add_u32 s19, s19, s22 // (op 7295) most significant: sum


s_mov_b32 s64, s36 // (op 7295) reg expression
s_mov_b32 s65, s37 // (op 7295) reg expression
s_mov_b32 s66, s18 // (op 7295) reg expression
s_mov_b32 s67, 131072 // (op 7295) call()

s_mul_hi_i32 s1, s16, s10 // (op 7320) 
v_and_b32 v17, 63, v1 // (op 7320) 
s_add_i32 s18, s1, s16 // (op 7320) 

v_lshrrev_b32 v20, 4, v17 // (op 7320) 
s_ashr_i32 s1, s18, 31 // (op 7320) 
v_and_b32 v21, 3, v20 // (op 7320) 

s_cmp_eq_i32 s10, 0 // (op 7320) 
s_mov_b32 s19, scc // (op 7320) 
v_and_b32 v20, 15, v1 // (op 7320) 
s_cmp_eq_i32 s19, 0 // (op 7320) 
s_cselect_b32 s22, 0, -1 // (op 7320) 

v_lshl_add_u32 v70, v21, 4, v20 // (op 7320) 


s_and_b32 s19, s13, 31 // (op 7320) 
v_lshrrev_b32 v20, 6, v1 // (op 7320) 
s_lshl_b32 s23, 1, s19 // (op 7320) 
v_and_b32 v21, 1, v20 // (op 7320) 

s_add_i32 s24, s23, s22 // (op 7320) 


v_lshl_add_u32 v20, v21, 6, v70 // (op 7320) 


s_and_b32 s25, s1, s24 // (op 7320) 


v_lshrrev_b32 v21, 6, v17 // (op 7320) 

s_add_i32 s1, s18, s25 // (op 7320) 


v_lshlrev_b32 v17, 2, v21 // (op 7320) 

s_ashr_i32 s18, s1, s19 // (op 7320) 


s_lshl_b32 s1, s13, 24 // (op 7320) 
s_ashr_i32 s19, s1, 31 // (op 7320) 

s_xor_b32 s1, s18, s19 // (op 7320) 

s_sub_i32 s18, s1, s19 // (op 7320) 


s_mul_hi_i32 s1, s18, s14 // (op 7320) 
s_add_i32 s19, s1, s18 // (op 7320) 

s_ashr_i32 s1, s19, 31 // (op 7320) 
s_cmp_eq_i32 s14, 0 // (op 7320) 
s_mov_b32 s22, scc // (op 7320) 
s_cmp_eq_i32 s22, 0 // (op 7320) 
s_cselect_b32 s23, 0, -1 // (op 7320) 

s_and_b32 s22, s15, 31 // (op 7320) 
s_lshl_b32 s24, 1, s22 // (op 7320) 
s_add_i32 s25, s24, s23 // (op 7320) 


s_and_b32 s23, s1, s25 // (op 7320) 


s_add_i32 s1, s19, s23 // (op 7320) 


s_ashr_i32 s19, s1, s22 // (op 7320) 


s_lshl_b32 s1, s15, 24 // (op 7320) 
s_ashr_i32 s22, s1, 31 // (op 7320) 

s_xor_b32 s1, s19, s22 // (op 7320) 

s_sub_i32 s19, s1, s22 // (op 7320) 


s_add_u32 s22, s6, 128 // (op 7320) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 7320) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 7320) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 7320) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 7320) 

s_mul_i32 s1, s19, s22 // (op 7320) 

s_sub_i32 s19, s18, s1 // (op 7320) 


s_mul_hi_i32 s1, s16, s9 // (op 7320) 
s_add_i32 s18, s1, s16 // (op 7320) 

s_ashr_i32 s1, s18, 31 // (op 7320) 
s_cmp_eq_i32 s9, 0 // (op 7320) 
s_mov_b32 s24, scc // (op 7320) 
s_cmp_eq_i32 s24, 0 // (op 7320) 
s_cselect_b32 s25, 0, -1 // (op 7320) 

s_and_b32 s24, s45, 31 // (op 7320) 
s_lshl_b32 s68, 1, s24 // (op 7320) 
s_add_i32 s69, s68, s25 // (op 7320) 


s_and_b32 s25, s1, s69 // (op 7320) 


s_add_i32 s1, s18, s25 // (op 7320) 


s_ashr_i32 s18, s1, s24 // (op 7320) 


s_lshl_b32 s1, s45, 24 // (op 7320) 
s_ashr_i32 s24, s1, 31 // (op 7320) 

s_xor_b32 s1, s18, s24 // (op 7320) 

s_sub_i32 s18, s1, s24 // (op 7320) 


s_mul_i32 s1, s11, s22 // (op 7320) 

s_mul_i32 s22, s8, s1 // (op 7320) 

s_mul_i32 s1, s18, s22 // (op 7320) 

s_sub_u32 s22, s16, s1 // (op 7320) 

s_mul_hi_i32 s1, s22, s12 // (op 7320) 
s_add_i32 s23, s1, s22 // (op 7320) 


s_ashr_i32 s1, s23, 31 // (op 7320) 
s_cmp_eq_i32 s12, 0 // (op 7320) 
s_mov_b32 s22, scc // (op 7320) 
s_cmp_eq_i32 s22, 0 // (op 7320) 
s_cselect_b32 s24, 0, -1 // (op 7320) 

s_and_b32 s22, s44, 31 // (op 7320) 
s_lshl_b32 s25, 1, s22 // (op 7320) 
s_add_i32 s68, s25, s24 // (op 7320) 


s_and_b32 s24, s1, s68 // (op 7320) 


s_add_i32 s1, s23, s24 // (op 7320) 


s_ashr_i32 s23, s1, s22 // (op 7320) 


s_lshl_b32 s1, s44, 24 // (op 7320) 
s_ashr_i32 s22, s1, 31 // (op 7320) 

s_xor_b32 s1, s23, s22 // (op 7320) 

s_sub_i32 s23, s1, s22 // (op 7320) 


s_cmp_eq_i32 s18, 0 // (op 7320) 
s_mov_b32 s1, scc // (op 7320) 

s_cmp_eq_i32 s1, 0 // (op 7320) 
s_cselect_b32 s18, s23, s19 // (op 7320) 



v_lshl_add_u32 v21, s18, 7, v20 // (op 7320) 


v_mul_lo_u32 v20, v21, s38 // (op 7320) 

s_ashr_i64 s[18:19], s[42:43], 7 // (op 7320) 
s_lshr_b32 s1, s18, 1 // (op 7320) 

v_lshl_add_u32 v21, s1, 3, v17 // (op 7320) 


v_add_u32 v17, v21, v20 // (op 7320) 


s_mul_i32 s19, s38, s7 // (op 7322) most significant: low of low * high
s_mul_i32 s1, s39, s6 // (op 7322) most significant: low of high * low
s_mul_i32 s18, s38, s6 // (op 7322) least significant
s_mul_hi_u32 s22, s38, s6 // (op 7322) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 7322) most significant: sum
s_add_u32 s19, s19, s22 // (op 7322) most significant: sum


s_mov_b32 s68, s40 // (op 7322) reg expression
s_mov_b32 s69, s41 // (op 7322) reg expression
s_mov_b32 s70, s18 // (op 7322) reg expression
s_mov_b32 s71, 131072 // (op 7322) call()

s_ashr_i64 s[18:19], s[42:43], 7 // (op 157) 
s_and_b32 s1, s18, 4294967294 // (op 157) 

s_ashr_i64 s[18:19], s[42:43], 7 // (op 156) 
s_mov_b32 s22, s18 // (op 156) convert

s_cmp_lt_u32 s1, s22 // (op 156) 











s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ForLoopBottom_KLoopTail_156 // (op 156) Condition: Top (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ForLoopBottom_KLoopTail_156 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ForLoopTop_KLoopTail_156:
s_barrier  // (op 7444) 
s_add_u32 s1, s1, 1 // (op 158) 

s_lshl_b64 s[18:19], s[26:27], 5 // (op 6744) 
v_mov_b64 v[12:13], s[18:19] // (op 6744) convert


v_mov_b64 v[14:15], v[10:11] // (op 6744) 
v_mov_b64 v[20:21], v[14:15] // (op 6744) 
s_add_u32 s18, 8192, s47 // (op 6744) 
s_mov_b32 m0, s18 // (op 6744) reg expression
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 6744) Load value direct to lds
v_add_co_u32 v20, s[22:23], v20, v12 // (op 6744) least significant half
v_addc_co_u32 v21, s[22:23], 0, v21, s[22:23] // (op 6744) most significant half

s_add_u32 m0, m0, 4096 // (op 6744) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 6744) Load value direct to lds
v_add_co_u32 v20, s[22:23], v20, v12 // (op 6744) least significant half
v_addc_co_u32 v21, s[22:23], 0, v21, s[22:23] // (op 6744) most significant half

s_add_u32 m0, m0, 4096 // (op 6744) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 6744) Load value direct to lds
v_add_co_u32 v20, s[22:23], v20, v12 // (op 6744) least significant half
v_addc_co_u32 v21, s[22:23], 0, v21, s[22:23] // (op 6744) most significant half

s_add_u32 m0, m0, 4096 // (op 6744) 
buffer_load_dwordx4 v20, s[48:51], 0 offen offset: 0 lds // (op 6744) Load value direct to lds




v_add_co_u32 v10, vcc, 128, v10 // (op 6781) least significant half
v_addc_co_u32 v11, vcc, 0, v11, vcc // (op 6781) most significant half
s_barrier  // (op 7460) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 6750) 
s_lshl_b64 s[22:23], s[18:19], 2 // (op 6750) 

v_lshrrev_b64 v[12:13], 3, s[22:23] // (op 6750) 


v_mov_b64 v[14:15], v[18:19] // (op 6750) 
v_mov_b64 v[20:21], v[14:15] // (op 6750) 
s_mov_b32 m0, s0 // (op 6750) reg expression
buffer_load_dwordx4 v20, s[60:63], 0 offen offset: 0 lds // (op 6750) Load value direct to lds
v_add_co_u32 v20, s[18:19], v20, v12 // (op 6750) least significant half
v_addc_co_u32 v21, s[18:19], 0, v21, s[18:19] // (op 6750) most significant half

s_add_u32 m0, m0, 4096 // (op 6750) 
buffer_load_dwordx4 v20, s[60:63], 0 offen offset: 0 lds // (op 6750) Load value direct to lds



v_add_co_u32 v18, s[18:19], 64, v18 // (op 6865) least significant half
v_addc_co_u32 v19, s[18:19], 0, v19, s[18:19] // (op 6865) most significant half

s_waitcnt vmcnt(2)
s_barrier  // (op 7450) 



ds_read_b128 v[20:23], v3 offset:8192 // (op 164) Load local data 
ds_read_b128 v[24:27], v3 offset:8256 // (op 164) Load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 7466) 



ds_read_b128 v[12:15], v4 // (op 166) Load local data 


v_mov_b32 v28, v16 // (op 168) 
v_mov_b32 v29, v28 // (op 168) reg expression

buffer_load_dwordx2 v[30:31], v29, s[64:67], 0 offen // (op 168) Load value


v_add_u32 v16, 4, v16 // (op 7283) 


v_mov_b32 v28, v17 // (op 170) 
v_mov_b32 v29, v28 // (op 170) reg expression

buffer_load_dwordx2 v[32:33], v29, s[68:71], 0 offen // (op 170) Load value


v_add_u32 v17, 4, v17 // (op 7310) 
s_waitcnt vmcnt(1)
v_bfe_u32 v34, v30, 0, 8 // (op 6241) 



v_bfe_u32 v35, v30, 8, 8 // (op 6241) 
v_bfe_u32 v36, v30, 16, 8 // (op 6241) 
v_bfe_u32 v37, v30, 24, 8 // (op 6241) 
v_permlane16_swap_b32 v34, v35 // (op 6241) 
s_nop 0
v_permlane16_swap_b32 v36, v37 // (op 6241) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v34, v36 // (op 6241) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v35, v37 // (op 6241) 
ds_read_b128 v[38:41], v4 offset:1024 // (op 295) Load local data 




ds_read_b128 v[28:31], v4 offset:2048 // (op 431) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v42, v32, 0, 8 // (op 6271) 



v_bfe_u32 v43, v32, 8, 8 // (op 6271) 
v_bfe_u32 v44, v32, 16, 8 // (op 6271) 
v_bfe_u32 v45, v32, 24, 8 // (op 6271) 
v_permlane16_swap_b32 v42, v43 // (op 6271) 
ds_read_b128 v[46:49], v4 offset:3072 // (op 567) Load local data 
v_permlane16_swap_b32 v44, v45 // (op 6271) 



s_nop 1
v_permlane32_swap_b32 v42, v44 // (op 6271) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v43, v45 // (op 6271) 
ds_read_b128 v[50:53], v3 offset:10240 // (op 789) Load local data 

ds_read_b128 v[54:57], v3 offset:10304 // (op 789) Load local data 



s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:27], v[12:15], a[0:3], v34, v42 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 171) 
ds_read_b128 v[58:61], v3 offset:12288 // (op 1399) Load local data 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:27], v[38:41], a[4:7], v34, v43 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 298) 
ds_read_b128 v[62:65], v3 offset:12352 // (op 1399) Load local data 



s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:27], v[28:31], a[8:11], v34, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 434) 
ds_read_b128 v[66:69], v3 offset:14336 // (op 2009) Load local data 
ds_read_b128 v[70:73], v3 offset:14400 // (op 2009) Load local data 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:27], v[46:49], a[12:15], v34, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 570) 

s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[50:57], v[12:15], a[16:19], v35, v42 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 793) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[50:57], v[28:31], a[24:27], v35, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 847) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[50:57], v[46:49], a[28:31], v35, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 874) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[50:57], v[38:41], a[20:23], v35, v43 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 820) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[58:65], v[12:15], a[32:35], v36, v42 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1403) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[58:65], v[28:31], a[40:43], v36, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1457) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[58:65], v[46:49], a[44:47], v36, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1484) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[58:65], v[38:41], a[36:39], v36, v43 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 1430) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[66:73], v[12:15], a[48:51], v37, v42 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2013) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[66:73], v[28:31], a[56:59], v37, v44 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2067) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[66:73], v[46:49], a[60:63], v37, v45 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2094) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[66:73], v[38:41], a[52:55], v37, v43 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b100 // (op 2040) 




s_ashr_i64 s[18:19], s[42:43], 7 // (op 156) 
s_mov_b32 s22, s18 // (op 156) convert

s_cmp_lt_u32 s1, s22 // (op 156) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ForLoopTop_KLoopTail_156 // (op 156) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ForLoopTop_KLoopTail_156 if true)
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ForLoopBottom_KLoopTail_156:




















s_barrier  // (op 7510) 

















v_cmp_eq_f32 vcc, 0.00000, v0 // (op 116) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ConditionalFalse_0_beta_116 // (op 116) Condition: False, jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ConditionalFalse_0_beta_116
v_accvgpr_read v10, a0 // (op 42) 
v_accvgpr_read v11, a1 // (op 42) 
v_accvgpr_read v12, a2 // (op 42) 
v_accvgpr_read v13, a3 // (op 42) 
v_mul_f32 v14, v7, v10 // (op 42) 
v_mul_f32 v15, v7, v11 // (op 42) 
v_mul_f32 v16, v7, v12 // (op 42) 
v_mul_f32 v17, v7, v13 // (op 42) 

v_cvt_f16_f32 v10, v14 // (op 48) 
v_cvt_f16_f32 v11, v15 // (op 48) 
v_cvt_f16_f32 v12, v16 // (op 48) 
v_cvt_f16_f32 v13, v17 // (op 48) 


v_pack_b32_f16 v4, v10, v11 // (op 50) 
v_pack_b32_f16 v5, v12, v13 // (op 50) 

v_mov_b32 v3, v9 // (op 50) reg expression

buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 50) Store value



v_accvgpr_read v10, a4 // (op 2906) 
v_accvgpr_read v11, a5 // (op 2906) 
v_accvgpr_read v12, a6 // (op 2906) 
v_accvgpr_read v13, a7 // (op 2906) 
v_mul_f32 v14, v7, v10 // (op 2906) 
v_mul_f32 v15, v7, v11 // (op 2906) 
v_mul_f32 v16, v7, v12 // (op 2906) 
v_mul_f32 v17, v7, v13 // (op 2906) 

v_cvt_f16_f32 v10, v14 // (op 2908) 
v_cvt_f16_f32 v11, v15 // (op 2908) 
v_cvt_f16_f32 v12, v16 // (op 2908) 
v_cvt_f16_f32 v13, v17 // (op 2908) 


v_pack_b32_f16 v4, v10, v11 // (op 2909) 
v_pack_b32_f16 v5, v12, v13 // (op 2909) 

s_lshl_b32 s0, s4, 5 // (op 2909) 
v_add_u32 v3, v9, s0 // (op 2909) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 2909) Store value



v_accvgpr_read v10, a8 // (op 2920) 
v_accvgpr_read v11, a9 // (op 2920) 
v_accvgpr_read v12, a10 // (op 2920) 
v_accvgpr_read v13, a11 // (op 2920) 
v_mul_f32 v14, v7, v10 // (op 2920) 
v_mul_f32 v15, v7, v11 // (op 2920) 
v_mul_f32 v16, v7, v12 // (op 2920) 
v_mul_f32 v17, v7, v13 // (op 2920) 

v_cvt_f16_f32 v10, v14 // (op 2922) 
v_cvt_f16_f32 v11, v15 // (op 2922) 
v_cvt_f16_f32 v12, v16 // (op 2922) 
v_cvt_f16_f32 v13, v17 // (op 2922) 


v_pack_b32_f16 v4, v10, v11 // (op 2923) 
v_pack_b32_f16 v5, v12, v13 // (op 2923) 

s_lshl_b32 s0, s4, 6 // (op 2923) 
v_add_u32 v3, v9, s0 // (op 2923) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 2923) Store value



v_accvgpr_read v10, a12 // (op 2934) 
v_accvgpr_read v11, a13 // (op 2934) 
v_accvgpr_read v12, a14 // (op 2934) 
v_accvgpr_read v13, a15 // (op 2934) 
v_mul_f32 v14, v7, v10 // (op 2934) 
v_mul_f32 v15, v7, v11 // (op 2934) 
v_mul_f32 v16, v7, v12 // (op 2934) 
v_mul_f32 v17, v7, v13 // (op 2934) 

v_cvt_f16_f32 v10, v14 // (op 2936) 
v_cvt_f16_f32 v11, v15 // (op 2936) 
v_cvt_f16_f32 v12, v16 // (op 2936) 
v_cvt_f16_f32 v13, v17 // (op 2936) 


v_pack_b32_f16 v4, v10, v11 // (op 2937) 
v_pack_b32_f16 v5, v12, v13 // (op 2937) 

s_lshl_b32 s0, s4, 5 // (op 2937) 
s_mul_i32 s1, 3, s0 // (op 2937) 

v_add_u32 v3, v9, s1 // (op 2937) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 2937) Store value



v_accvgpr_read v10, a16 // (op 2970) 
v_accvgpr_read v11, a17 // (op 2970) 
v_accvgpr_read v12, a18 // (op 2970) 
v_accvgpr_read v13, a19 // (op 2970) 
v_mul_f32 v14, v7, v10 // (op 2970) 
v_mul_f32 v15, v7, v11 // (op 2970) 
v_mul_f32 v16, v7, v12 // (op 2970) 
v_mul_f32 v17, v7, v13 // (op 2970) 

v_cvt_f16_f32 v10, v14 // (op 2972) 
v_cvt_f16_f32 v11, v15 // (op 2972) 
v_cvt_f16_f32 v12, v16 // (op 2972) 
v_cvt_f16_f32 v13, v17 // (op 2972) 


v_pack_b32_f16 v4, v10, v11 // (op 2974) 
v_pack_b32_f16 v5, v12, v13 // (op 2974) 

v_add_u32 v3, 32, v9 // (op 2974) 

buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 2974) Store value



v_accvgpr_read v10, a20 // (op 2984) 
v_accvgpr_read v11, a21 // (op 2984) 
v_accvgpr_read v12, a22 // (op 2984) 
v_accvgpr_read v13, a23 // (op 2984) 
v_mul_f32 v14, v7, v10 // (op 2984) 
v_mul_f32 v15, v7, v11 // (op 2984) 
v_mul_f32 v16, v7, v12 // (op 2984) 
v_mul_f32 v17, v7, v13 // (op 2984) 

v_cvt_f16_f32 v10, v14 // (op 2986) 
v_cvt_f16_f32 v11, v15 // (op 2986) 
v_cvt_f16_f32 v12, v16 // (op 2986) 
v_cvt_f16_f32 v13, v17 // (op 2986) 


v_pack_b32_f16 v4, v10, v11 // (op 2976) 
v_pack_b32_f16 v5, v12, v13 // (op 2976) 

s_lshl_b32 s0, s4, 5 // (op 2976) 
s_add_u32 s0, s0, 32 // (op 2976) 
v_add_u32 v3, v9, s0 // (op 2976) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 2976) Store value



v_accvgpr_read v10, a24 // (op 2990) 
v_accvgpr_read v11, a25 // (op 2990) 
v_accvgpr_read v12, a26 // (op 2990) 
v_accvgpr_read v13, a27 // (op 2990) 
v_mul_f32 v14, v7, v10 // (op 2990) 
v_mul_f32 v15, v7, v11 // (op 2990) 
v_mul_f32 v16, v7, v12 // (op 2990) 
v_mul_f32 v17, v7, v13 // (op 2990) 

v_cvt_f16_f32 v10, v14 // (op 2992) 
v_cvt_f16_f32 v11, v15 // (op 2992) 
v_cvt_f16_f32 v12, v16 // (op 2992) 
v_cvt_f16_f32 v13, v17 // (op 2992) 


v_pack_b32_f16 v4, v10, v11 // (op 2978) 
v_pack_b32_f16 v5, v12, v13 // (op 2978) 

s_lshl_b32 s0, s4, 6 // (op 2978) 
s_add_u32 s0, s0, 32 // (op 2978) 
v_add_u32 v3, v9, s0 // (op 2978) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 2978) Store value



v_accvgpr_read v10, a28 // (op 2996) 
v_accvgpr_read v11, a29 // (op 2996) 
v_accvgpr_read v12, a30 // (op 2996) 
v_accvgpr_read v13, a31 // (op 2996) 
v_mul_f32 v14, v7, v10 // (op 2996) 
v_mul_f32 v15, v7, v11 // (op 2996) 
v_mul_f32 v16, v7, v12 // (op 2996) 
v_mul_f32 v17, v7, v13 // (op 2996) 

v_cvt_f16_f32 v10, v14 // (op 2998) 
v_cvt_f16_f32 v11, v15 // (op 2998) 
v_cvt_f16_f32 v12, v16 // (op 2998) 
v_cvt_f16_f32 v13, v17 // (op 2998) 


v_pack_b32_f16 v4, v10, v11 // (op 2980) 
v_pack_b32_f16 v5, v12, v13 // (op 2980) 

s_lshl_b32 s0, s4, 5 // (op 2980) 
s_mul_i32 s1, 3, s0 // (op 2980) 
s_add_u32 s1, s1, 32 // (op 2980) 

v_add_u32 v3, v9, s1 // (op 2980) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 2980) Store value



v_accvgpr_read v10, a32 // (op 3046) 
v_accvgpr_read v11, a33 // (op 3046) 
v_accvgpr_read v12, a34 // (op 3046) 
v_accvgpr_read v13, a35 // (op 3046) 
v_mul_f32 v14, v7, v10 // (op 3046) 
v_mul_f32 v15, v7, v11 // (op 3046) 
v_mul_f32 v16, v7, v12 // (op 3046) 
v_mul_f32 v17, v7, v13 // (op 3046) 

v_cvt_f16_f32 v10, v14 // (op 3048) 
v_cvt_f16_f32 v11, v15 // (op 3048) 
v_cvt_f16_f32 v12, v16 // (op 3048) 
v_cvt_f16_f32 v13, v17 // (op 3048) 


v_pack_b32_f16 v4, v10, v11 // (op 3050) 
v_pack_b32_f16 v5, v12, v13 // (op 3050) 

v_add_u32 v3, 64, v9 // (op 3050) 

buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 3050) Store value



v_accvgpr_read v10, a36 // (op 3060) 
v_accvgpr_read v11, a37 // (op 3060) 
v_accvgpr_read v12, a38 // (op 3060) 
v_accvgpr_read v13, a39 // (op 3060) 
v_mul_f32 v14, v7, v10 // (op 3060) 
v_mul_f32 v15, v7, v11 // (op 3060) 
v_mul_f32 v16, v7, v12 // (op 3060) 
v_mul_f32 v17, v7, v13 // (op 3060) 

v_cvt_f16_f32 v10, v14 // (op 3062) 
v_cvt_f16_f32 v11, v15 // (op 3062) 
v_cvt_f16_f32 v12, v16 // (op 3062) 
v_cvt_f16_f32 v13, v17 // (op 3062) 


v_pack_b32_f16 v4, v10, v11 // (op 3052) 
v_pack_b32_f16 v5, v12, v13 // (op 3052) 

s_lshl_b32 s0, s4, 5 // (op 3052) 
s_add_u32 s0, s0, 64 // (op 3052) 
v_add_u32 v3, v9, s0 // (op 3052) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 3052) Store value



v_accvgpr_read v10, a40 // (op 3066) 
v_accvgpr_read v11, a41 // (op 3066) 
v_accvgpr_read v12, a42 // (op 3066) 
v_accvgpr_read v13, a43 // (op 3066) 
v_mul_f32 v14, v7, v10 // (op 3066) 
v_mul_f32 v15, v7, v11 // (op 3066) 
v_mul_f32 v16, v7, v12 // (op 3066) 
v_mul_f32 v17, v7, v13 // (op 3066) 

v_cvt_f16_f32 v10, v14 // (op 3068) 
v_cvt_f16_f32 v11, v15 // (op 3068) 
v_cvt_f16_f32 v12, v16 // (op 3068) 
v_cvt_f16_f32 v13, v17 // (op 3068) 


v_pack_b32_f16 v4, v10, v11 // (op 3054) 
v_pack_b32_f16 v5, v12, v13 // (op 3054) 

s_lshl_b32 s0, s4, 6 // (op 3054) 
s_add_u32 s0, s0, 64 // (op 3054) 
v_add_u32 v3, v9, s0 // (op 3054) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 3054) Store value



v_accvgpr_read v10, a44 // (op 3072) 
v_accvgpr_read v11, a45 // (op 3072) 
v_accvgpr_read v12, a46 // (op 3072) 
v_accvgpr_read v13, a47 // (op 3072) 
v_mul_f32 v14, v7, v10 // (op 3072) 
v_mul_f32 v15, v7, v11 // (op 3072) 
v_mul_f32 v16, v7, v12 // (op 3072) 
v_mul_f32 v17, v7, v13 // (op 3072) 

v_cvt_f16_f32 v10, v14 // (op 3074) 
v_cvt_f16_f32 v11, v15 // (op 3074) 
v_cvt_f16_f32 v12, v16 // (op 3074) 
v_cvt_f16_f32 v13, v17 // (op 3074) 


v_pack_b32_f16 v4, v10, v11 // (op 3056) 
v_pack_b32_f16 v5, v12, v13 // (op 3056) 

s_lshl_b32 s0, s4, 5 // (op 3056) 
s_mul_i32 s1, 3, s0 // (op 3056) 
s_add_u32 s1, s1, 64 // (op 3056) 

v_add_u32 v3, v9, s1 // (op 3056) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 3056) Store value



v_accvgpr_read v10, a48 // (op 3122) 
v_accvgpr_read v11, a49 // (op 3122) 
v_accvgpr_read v12, a50 // (op 3122) 
v_accvgpr_read v13, a51 // (op 3122) 
v_mul_f32 v14, v7, v10 // (op 3122) 
v_mul_f32 v15, v7, v11 // (op 3122) 
v_mul_f32 v16, v7, v12 // (op 3122) 
v_mul_f32 v17, v7, v13 // (op 3122) 

v_cvt_f16_f32 v10, v14 // (op 3124) 
v_cvt_f16_f32 v11, v15 // (op 3124) 
v_cvt_f16_f32 v12, v16 // (op 3124) 
v_cvt_f16_f32 v13, v17 // (op 3124) 


v_pack_b32_f16 v4, v10, v11 // (op 3126) 
v_pack_b32_f16 v5, v12, v13 // (op 3126) 

v_add_u32 v3, 96, v9 // (op 3126) 

buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 3126) Store value



v_accvgpr_read v10, a52 // (op 3136) 
v_accvgpr_read v11, a53 // (op 3136) 
v_accvgpr_read v12, a54 // (op 3136) 
v_accvgpr_read v13, a55 // (op 3136) 
v_mul_f32 v14, v7, v10 // (op 3136) 
v_mul_f32 v15, v7, v11 // (op 3136) 
v_mul_f32 v16, v7, v12 // (op 3136) 
v_mul_f32 v17, v7, v13 // (op 3136) 

v_cvt_f16_f32 v10, v14 // (op 3138) 
v_cvt_f16_f32 v11, v15 // (op 3138) 
v_cvt_f16_f32 v12, v16 // (op 3138) 
v_cvt_f16_f32 v13, v17 // (op 3138) 


v_pack_b32_f16 v4, v10, v11 // (op 3128) 
v_pack_b32_f16 v5, v12, v13 // (op 3128) 

s_lshl_b32 s0, s4, 5 // (op 3128) 
s_add_u32 s0, s0, 96 // (op 3128) 
v_add_u32 v3, v9, s0 // (op 3128) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 3128) Store value



v_accvgpr_read v10, a56 // (op 3142) 
v_accvgpr_read v11, a57 // (op 3142) 
v_accvgpr_read v12, a58 // (op 3142) 
v_accvgpr_read v13, a59 // (op 3142) 
v_mul_f32 v14, v7, v10 // (op 3142) 
v_mul_f32 v15, v7, v11 // (op 3142) 
v_mul_f32 v16, v7, v12 // (op 3142) 
v_mul_f32 v17, v7, v13 // (op 3142) 

v_cvt_f16_f32 v10, v14 // (op 3144) 
v_cvt_f16_f32 v11, v15 // (op 3144) 
v_cvt_f16_f32 v12, v16 // (op 3144) 
v_cvt_f16_f32 v13, v17 // (op 3144) 


v_pack_b32_f16 v4, v10, v11 // (op 3130) 
v_pack_b32_f16 v5, v12, v13 // (op 3130) 

s_lshl_b32 s0, s4, 6 // (op 3130) 
s_add_u32 s0, s0, 96 // (op 3130) 
v_add_u32 v3, v9, s0 // (op 3130) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 3130) Store value



v_accvgpr_read v10, a60 // (op 3148) 
v_accvgpr_read v11, a61 // (op 3148) 
v_accvgpr_read v12, a62 // (op 3148) 
v_accvgpr_read v13, a63 // (op 3148) 
v_mul_f32 v14, v7, v10 // (op 3148) 
v_mul_f32 v15, v7, v11 // (op 3148) 
v_mul_f32 v16, v7, v12 // (op 3148) 
v_mul_f32 v17, v7, v13 // (op 3148) 

v_cvt_f16_f32 v10, v14 // (op 3150) 
v_cvt_f16_f32 v11, v15 // (op 3150) 
v_cvt_f16_f32 v12, v16 // (op 3150) 
v_cvt_f16_f32 v13, v17 // (op 3150) 


v_pack_b32_f16 v4, v10, v11 // (op 3132) 
v_pack_b32_f16 v5, v12, v13 // (op 3132) 

s_lshl_b32 s0, s4, 5 // (op 3132) 
s_mul_i32 s1, 3, s0 // (op 3132) 
s_add_u32 s1, s1, 96 // (op 3132) 

v_add_u32 v3, v9, s1 // (op 3132) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 3132) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ConditionalBottom_0_beta_116 // (op 116) Condition: Done, jump to RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ConditionalBottom_0_beta_116
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ConditionalFalse_0_beta_116:


v_add_u32 v3, 96, v8 // (op 3520) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 3520) Load value

v_accvgpr_read v10, a48 // (op 3507) 
v_accvgpr_read v11, a49 // (op 3507) 
v_accvgpr_read v12, a50 // (op 3507) 
v_accvgpr_read v13, a51 // (op 3507) 
v_mul_f32 v14, v7, v10 // (op 3507) 
v_mul_f32 v15, v7, v11 // (op 3507) 
v_mul_f32 v16, v7, v12 // (op 3507) 
v_mul_f32 v17, v7, v13 // (op 3507) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 3519) 
v_lshrrev_b32 v11, 16, v4 // (op 3519) 
v_cvt_f32_f16 v11, v11 // (op 3519) 
v_mul_f32 v18, v0, v10 // (op 3519) 
v_mul_f32 v19, v0, v11 // (op 3519) 
v_cvt_f32_f16 v10, v5 // (op 3519) 
v_lshrrev_b32 v11, 16, v5 // (op 3519) 
v_cvt_f32_f16 v11, v11 // (op 3519) 
v_mul_f32 v20, v0, v10 // (op 3519) 
v_mul_f32 v21, v0, v11 // (op 3519) 


v_add_f32 v10, v18, v14 // (op 3508) 
v_add_f32 v11, v19, v15 // (op 3508) 
v_add_f32 v12, v20, v16 // (op 3508) 
v_add_f32 v13, v21, v17 // (op 3508) 


v_cvt_f16_f32 v14, v10 // (op 3509) 
v_cvt_f16_f32 v15, v11 // (op 3509) 
v_cvt_f16_f32 v16, v12 // (op 3509) 
v_cvt_f16_f32 v17, v13 // (op 3509) 


v_pack_b32_f16 v10, v14, v15 // (op 3511) 
v_pack_b32_f16 v11, v16, v17 // (op 3511) 

v_add_u32 v3, 96, v9 // (op 3511) 

buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 3511) Store value





s_lshl_b32 s0, s2, 5 // (op 3525) 
s_add_u32 s0, s0, 96 // (op 3525) 
v_add_u32 v3, v8, s0 // (op 3525) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 3525) Load value

v_accvgpr_read v10, a52 // (op 3537) 
v_accvgpr_read v11, a53 // (op 3537) 
v_accvgpr_read v12, a54 // (op 3537) 
v_accvgpr_read v13, a55 // (op 3537) 
v_mul_f32 v14, v7, v10 // (op 3537) 
v_mul_f32 v15, v7, v11 // (op 3537) 
v_mul_f32 v16, v7, v12 // (op 3537) 
v_mul_f32 v17, v7, v13 // (op 3537) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 3522) 
v_lshrrev_b32 v11, 16, v4 // (op 3522) 
v_cvt_f32_f16 v11, v11 // (op 3522) 
v_mul_f32 v18, v0, v10 // (op 3522) 
v_mul_f32 v19, v0, v11 // (op 3522) 
v_cvt_f32_f16 v10, v5 // (op 3522) 
v_lshrrev_b32 v11, 16, v5 // (op 3522) 
v_cvt_f32_f16 v11, v11 // (op 3522) 
v_mul_f32 v20, v0, v10 // (op 3522) 
v_mul_f32 v21, v0, v11 // (op 3522) 


v_add_f32 v10, v18, v14 // (op 3523) 
v_add_f32 v11, v19, v15 // (op 3523) 
v_add_f32 v12, v20, v16 // (op 3523) 
v_add_f32 v13, v21, v17 // (op 3523) 


v_cvt_f16_f32 v14, v10 // (op 3524) 
v_cvt_f16_f32 v15, v11 // (op 3524) 
v_cvt_f16_f32 v16, v12 // (op 3524) 
v_cvt_f16_f32 v17, v13 // (op 3524) 


v_pack_b32_f16 v10, v14, v15 // (op 3513) 
v_pack_b32_f16 v11, v16, v17 // (op 3513) 

s_lshl_b32 s0, s4, 5 // (op 3513) 
s_add_u32 s0, s0, 96 // (op 3513) 
v_add_u32 v3, v9, s0 // (op 3513) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 3513) Store value





s_lshl_b32 s0, s2, 6 // (op 3530) 
s_add_u32 s0, s0, 96 // (op 3530) 
v_add_u32 v3, v8, s0 // (op 3530) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 3530) Load value

v_accvgpr_read v10, a56 // (op 3539) 
v_accvgpr_read v11, a57 // (op 3539) 
v_accvgpr_read v12, a58 // (op 3539) 
v_accvgpr_read v13, a59 // (op 3539) 
v_mul_f32 v14, v7, v10 // (op 3539) 
v_mul_f32 v15, v7, v11 // (op 3539) 
v_mul_f32 v16, v7, v12 // (op 3539) 
v_mul_f32 v17, v7, v13 // (op 3539) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 3527) 
v_lshrrev_b32 v11, 16, v4 // (op 3527) 
v_cvt_f32_f16 v11, v11 // (op 3527) 
v_mul_f32 v18, v0, v10 // (op 3527) 
v_mul_f32 v19, v0, v11 // (op 3527) 
v_cvt_f32_f16 v10, v5 // (op 3527) 
v_lshrrev_b32 v11, 16, v5 // (op 3527) 
v_cvt_f32_f16 v11, v11 // (op 3527) 
v_mul_f32 v20, v0, v10 // (op 3527) 
v_mul_f32 v21, v0, v11 // (op 3527) 


v_add_f32 v10, v18, v14 // (op 3528) 
v_add_f32 v11, v19, v15 // (op 3528) 
v_add_f32 v12, v20, v16 // (op 3528) 
v_add_f32 v13, v21, v17 // (op 3528) 


v_cvt_f16_f32 v14, v10 // (op 3529) 
v_cvt_f16_f32 v15, v11 // (op 3529) 
v_cvt_f16_f32 v16, v12 // (op 3529) 
v_cvt_f16_f32 v17, v13 // (op 3529) 


v_pack_b32_f16 v10, v14, v15 // (op 3515) 
v_pack_b32_f16 v11, v16, v17 // (op 3515) 

s_lshl_b32 s0, s4, 6 // (op 3515) 
s_add_u32 s0, s0, 96 // (op 3515) 
v_add_u32 v3, v9, s0 // (op 3515) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 3515) Store value





s_lshl_b32 s0, s2, 5 // (op 3535) 
s_mul_i32 s1, 3, s0 // (op 3535) 
s_add_u32 s1, s1, 96 // (op 3535) 

v_add_u32 v3, v8, s1 // (op 3535) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 3535) Load value

v_accvgpr_read v10, a60 // (op 3541) 
v_accvgpr_read v11, a61 // (op 3541) 
v_accvgpr_read v12, a62 // (op 3541) 
v_accvgpr_read v13, a63 // (op 3541) 
v_mul_f32 v14, v7, v10 // (op 3541) 
v_mul_f32 v15, v7, v11 // (op 3541) 
v_mul_f32 v16, v7, v12 // (op 3541) 
v_mul_f32 v17, v7, v13 // (op 3541) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 3532) 
v_lshrrev_b32 v11, 16, v4 // (op 3532) 
v_cvt_f32_f16 v11, v11 // (op 3532) 
v_mul_f32 v18, v0, v10 // (op 3532) 
v_mul_f32 v19, v0, v11 // (op 3532) 
v_cvt_f32_f16 v10, v5 // (op 3532) 
v_lshrrev_b32 v11, 16, v5 // (op 3532) 
v_cvt_f32_f16 v11, v11 // (op 3532) 
v_mul_f32 v20, v0, v10 // (op 3532) 
v_mul_f32 v21, v0, v11 // (op 3532) 


v_add_f32 v10, v18, v14 // (op 3533) 
v_add_f32 v11, v19, v15 // (op 3533) 
v_add_f32 v12, v20, v16 // (op 3533) 
v_add_f32 v13, v21, v17 // (op 3533) 


v_cvt_f16_f32 v14, v10 // (op 3534) 
v_cvt_f16_f32 v15, v11 // (op 3534) 
v_cvt_f16_f32 v16, v12 // (op 3534) 
v_cvt_f16_f32 v17, v13 // (op 3534) 


v_pack_b32_f16 v10, v14, v15 // (op 3517) 
v_pack_b32_f16 v11, v16, v17 // (op 3517) 

s_lshl_b32 s0, s4, 5 // (op 3517) 
s_mul_i32 s1, 3, s0 // (op 3517) 
s_add_u32 s1, s1, 96 // (op 3517) 

v_add_u32 v3, v9, s1 // (op 3517) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 3517) Store value





v_add_u32 v3, 64, v8 // (op 3433) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 3433) Load value

v_accvgpr_read v10, a32 // (op 3420) 
v_accvgpr_read v11, a33 // (op 3420) 
v_accvgpr_read v12, a34 // (op 3420) 
v_accvgpr_read v13, a35 // (op 3420) 
v_mul_f32 v14, v7, v10 // (op 3420) 
v_mul_f32 v15, v7, v11 // (op 3420) 
v_mul_f32 v16, v7, v12 // (op 3420) 
v_mul_f32 v17, v7, v13 // (op 3420) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 3432) 
v_lshrrev_b32 v11, 16, v4 // (op 3432) 
v_cvt_f32_f16 v11, v11 // (op 3432) 
v_mul_f32 v18, v0, v10 // (op 3432) 
v_mul_f32 v19, v0, v11 // (op 3432) 
v_cvt_f32_f16 v10, v5 // (op 3432) 
v_lshrrev_b32 v11, 16, v5 // (op 3432) 
v_cvt_f32_f16 v11, v11 // (op 3432) 
v_mul_f32 v20, v0, v10 // (op 3432) 
v_mul_f32 v21, v0, v11 // (op 3432) 


v_add_f32 v10, v18, v14 // (op 3421) 
v_add_f32 v11, v19, v15 // (op 3421) 
v_add_f32 v12, v20, v16 // (op 3421) 
v_add_f32 v13, v21, v17 // (op 3421) 


v_cvt_f16_f32 v14, v10 // (op 3422) 
v_cvt_f16_f32 v15, v11 // (op 3422) 
v_cvt_f16_f32 v16, v12 // (op 3422) 
v_cvt_f16_f32 v17, v13 // (op 3422) 


v_pack_b32_f16 v10, v14, v15 // (op 3424) 
v_pack_b32_f16 v11, v16, v17 // (op 3424) 

v_add_u32 v3, 64, v9 // (op 3424) 

buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 3424) Store value





s_lshl_b32 s0, s2, 5 // (op 3438) 
s_add_u32 s0, s0, 64 // (op 3438) 
v_add_u32 v3, v8, s0 // (op 3438) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 3438) Load value

v_accvgpr_read v10, a36 // (op 3450) 
v_accvgpr_read v11, a37 // (op 3450) 
v_accvgpr_read v12, a38 // (op 3450) 
v_accvgpr_read v13, a39 // (op 3450) 
v_mul_f32 v14, v7, v10 // (op 3450) 
v_mul_f32 v15, v7, v11 // (op 3450) 
v_mul_f32 v16, v7, v12 // (op 3450) 
v_mul_f32 v17, v7, v13 // (op 3450) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 3435) 
v_lshrrev_b32 v11, 16, v4 // (op 3435) 
v_cvt_f32_f16 v11, v11 // (op 3435) 
v_mul_f32 v18, v0, v10 // (op 3435) 
v_mul_f32 v19, v0, v11 // (op 3435) 
v_cvt_f32_f16 v10, v5 // (op 3435) 
v_lshrrev_b32 v11, 16, v5 // (op 3435) 
v_cvt_f32_f16 v11, v11 // (op 3435) 
v_mul_f32 v20, v0, v10 // (op 3435) 
v_mul_f32 v21, v0, v11 // (op 3435) 


v_add_f32 v10, v18, v14 // (op 3436) 
v_add_f32 v11, v19, v15 // (op 3436) 
v_add_f32 v12, v20, v16 // (op 3436) 
v_add_f32 v13, v21, v17 // (op 3436) 


v_cvt_f16_f32 v14, v10 // (op 3437) 
v_cvt_f16_f32 v15, v11 // (op 3437) 
v_cvt_f16_f32 v16, v12 // (op 3437) 
v_cvt_f16_f32 v17, v13 // (op 3437) 


v_pack_b32_f16 v10, v14, v15 // (op 3426) 
v_pack_b32_f16 v11, v16, v17 // (op 3426) 

s_lshl_b32 s0, s4, 5 // (op 3426) 
s_add_u32 s0, s0, 64 // (op 3426) 
v_add_u32 v3, v9, s0 // (op 3426) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 3426) Store value





s_lshl_b32 s0, s2, 6 // (op 3443) 
s_add_u32 s0, s0, 64 // (op 3443) 
v_add_u32 v3, v8, s0 // (op 3443) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 3443) Load value

v_accvgpr_read v10, a40 // (op 3452) 
v_accvgpr_read v11, a41 // (op 3452) 
v_accvgpr_read v12, a42 // (op 3452) 
v_accvgpr_read v13, a43 // (op 3452) 
v_mul_f32 v14, v7, v10 // (op 3452) 
v_mul_f32 v15, v7, v11 // (op 3452) 
v_mul_f32 v16, v7, v12 // (op 3452) 
v_mul_f32 v17, v7, v13 // (op 3452) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 3440) 
v_lshrrev_b32 v11, 16, v4 // (op 3440) 
v_cvt_f32_f16 v11, v11 // (op 3440) 
v_mul_f32 v18, v0, v10 // (op 3440) 
v_mul_f32 v19, v0, v11 // (op 3440) 
v_cvt_f32_f16 v10, v5 // (op 3440) 
v_lshrrev_b32 v11, 16, v5 // (op 3440) 
v_cvt_f32_f16 v11, v11 // (op 3440) 
v_mul_f32 v20, v0, v10 // (op 3440) 
v_mul_f32 v21, v0, v11 // (op 3440) 


v_add_f32 v10, v18, v14 // (op 3441) 
v_add_f32 v11, v19, v15 // (op 3441) 
v_add_f32 v12, v20, v16 // (op 3441) 
v_add_f32 v13, v21, v17 // (op 3441) 


v_cvt_f16_f32 v14, v10 // (op 3442) 
v_cvt_f16_f32 v15, v11 // (op 3442) 
v_cvt_f16_f32 v16, v12 // (op 3442) 
v_cvt_f16_f32 v17, v13 // (op 3442) 


v_pack_b32_f16 v10, v14, v15 // (op 3428) 
v_pack_b32_f16 v11, v16, v17 // (op 3428) 

s_lshl_b32 s0, s4, 6 // (op 3428) 
s_add_u32 s0, s0, 64 // (op 3428) 
v_add_u32 v3, v9, s0 // (op 3428) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 3428) Store value





s_lshl_b32 s0, s2, 5 // (op 3448) 
s_mul_i32 s1, 3, s0 // (op 3448) 
s_add_u32 s1, s1, 64 // (op 3448) 

v_add_u32 v3, v8, s1 // (op 3448) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 3448) Load value

v_accvgpr_read v10, a44 // (op 3454) 
v_accvgpr_read v11, a45 // (op 3454) 
v_accvgpr_read v12, a46 // (op 3454) 
v_accvgpr_read v13, a47 // (op 3454) 
v_mul_f32 v14, v7, v10 // (op 3454) 
v_mul_f32 v15, v7, v11 // (op 3454) 
v_mul_f32 v16, v7, v12 // (op 3454) 
v_mul_f32 v17, v7, v13 // (op 3454) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 3445) 
v_lshrrev_b32 v11, 16, v4 // (op 3445) 
v_cvt_f32_f16 v11, v11 // (op 3445) 
v_mul_f32 v18, v0, v10 // (op 3445) 
v_mul_f32 v19, v0, v11 // (op 3445) 
v_cvt_f32_f16 v10, v5 // (op 3445) 
v_lshrrev_b32 v11, 16, v5 // (op 3445) 
v_cvt_f32_f16 v11, v11 // (op 3445) 
v_mul_f32 v20, v0, v10 // (op 3445) 
v_mul_f32 v21, v0, v11 // (op 3445) 


v_add_f32 v10, v18, v14 // (op 3446) 
v_add_f32 v11, v19, v15 // (op 3446) 
v_add_f32 v12, v20, v16 // (op 3446) 
v_add_f32 v13, v21, v17 // (op 3446) 


v_cvt_f16_f32 v14, v10 // (op 3447) 
v_cvt_f16_f32 v15, v11 // (op 3447) 
v_cvt_f16_f32 v16, v12 // (op 3447) 
v_cvt_f16_f32 v17, v13 // (op 3447) 


v_pack_b32_f16 v10, v14, v15 // (op 3430) 
v_pack_b32_f16 v11, v16, v17 // (op 3430) 

s_lshl_b32 s0, s4, 5 // (op 3430) 
s_mul_i32 s1, 3, s0 // (op 3430) 
s_add_u32 s1, s1, 64 // (op 3430) 

v_add_u32 v3, v9, s1 // (op 3430) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 3430) Store value





v_add_u32 v3, 32, v8 // (op 3346) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 3346) Load value

v_accvgpr_read v10, a16 // (op 3333) 
v_accvgpr_read v11, a17 // (op 3333) 
v_accvgpr_read v12, a18 // (op 3333) 
v_accvgpr_read v13, a19 // (op 3333) 
v_mul_f32 v14, v7, v10 // (op 3333) 
v_mul_f32 v15, v7, v11 // (op 3333) 
v_mul_f32 v16, v7, v12 // (op 3333) 
v_mul_f32 v17, v7, v13 // (op 3333) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 3345) 
v_lshrrev_b32 v11, 16, v4 // (op 3345) 
v_cvt_f32_f16 v11, v11 // (op 3345) 
v_mul_f32 v18, v0, v10 // (op 3345) 
v_mul_f32 v19, v0, v11 // (op 3345) 
v_cvt_f32_f16 v10, v5 // (op 3345) 
v_lshrrev_b32 v11, 16, v5 // (op 3345) 
v_cvt_f32_f16 v11, v11 // (op 3345) 
v_mul_f32 v20, v0, v10 // (op 3345) 
v_mul_f32 v21, v0, v11 // (op 3345) 


v_add_f32 v10, v18, v14 // (op 3334) 
v_add_f32 v11, v19, v15 // (op 3334) 
v_add_f32 v12, v20, v16 // (op 3334) 
v_add_f32 v13, v21, v17 // (op 3334) 


v_cvt_f16_f32 v14, v10 // (op 3335) 
v_cvt_f16_f32 v15, v11 // (op 3335) 
v_cvt_f16_f32 v16, v12 // (op 3335) 
v_cvt_f16_f32 v17, v13 // (op 3335) 


v_pack_b32_f16 v10, v14, v15 // (op 3337) 
v_pack_b32_f16 v11, v16, v17 // (op 3337) 

v_add_u32 v3, 32, v9 // (op 3337) 

buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 3337) Store value





s_lshl_b32 s0, s2, 5 // (op 3351) 
s_add_u32 s0, s0, 32 // (op 3351) 
v_add_u32 v3, v8, s0 // (op 3351) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 3351) Load value

v_accvgpr_read v10, a20 // (op 3363) 
v_accvgpr_read v11, a21 // (op 3363) 
v_accvgpr_read v12, a22 // (op 3363) 
v_accvgpr_read v13, a23 // (op 3363) 
v_mul_f32 v14, v7, v10 // (op 3363) 
v_mul_f32 v15, v7, v11 // (op 3363) 
v_mul_f32 v16, v7, v12 // (op 3363) 
v_mul_f32 v17, v7, v13 // (op 3363) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 3348) 
v_lshrrev_b32 v11, 16, v4 // (op 3348) 
v_cvt_f32_f16 v11, v11 // (op 3348) 
v_mul_f32 v18, v0, v10 // (op 3348) 
v_mul_f32 v19, v0, v11 // (op 3348) 
v_cvt_f32_f16 v10, v5 // (op 3348) 
v_lshrrev_b32 v11, 16, v5 // (op 3348) 
v_cvt_f32_f16 v11, v11 // (op 3348) 
v_mul_f32 v20, v0, v10 // (op 3348) 
v_mul_f32 v21, v0, v11 // (op 3348) 


v_add_f32 v10, v18, v14 // (op 3349) 
v_add_f32 v11, v19, v15 // (op 3349) 
v_add_f32 v12, v20, v16 // (op 3349) 
v_add_f32 v13, v21, v17 // (op 3349) 


v_cvt_f16_f32 v14, v10 // (op 3350) 
v_cvt_f16_f32 v15, v11 // (op 3350) 
v_cvt_f16_f32 v16, v12 // (op 3350) 
v_cvt_f16_f32 v17, v13 // (op 3350) 


v_pack_b32_f16 v10, v14, v15 // (op 3339) 
v_pack_b32_f16 v11, v16, v17 // (op 3339) 

s_lshl_b32 s0, s4, 5 // (op 3339) 
s_add_u32 s0, s0, 32 // (op 3339) 
v_add_u32 v3, v9, s0 // (op 3339) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 3339) Store value





s_lshl_b32 s0, s2, 6 // (op 3356) 
s_add_u32 s0, s0, 32 // (op 3356) 
v_add_u32 v3, v8, s0 // (op 3356) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 3356) Load value

v_accvgpr_read v10, a24 // (op 3365) 
v_accvgpr_read v11, a25 // (op 3365) 
v_accvgpr_read v12, a26 // (op 3365) 
v_accvgpr_read v13, a27 // (op 3365) 
v_mul_f32 v14, v7, v10 // (op 3365) 
v_mul_f32 v15, v7, v11 // (op 3365) 
v_mul_f32 v16, v7, v12 // (op 3365) 
v_mul_f32 v17, v7, v13 // (op 3365) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 3353) 
v_lshrrev_b32 v11, 16, v4 // (op 3353) 
v_cvt_f32_f16 v11, v11 // (op 3353) 
v_mul_f32 v18, v0, v10 // (op 3353) 
v_mul_f32 v19, v0, v11 // (op 3353) 
v_cvt_f32_f16 v10, v5 // (op 3353) 
v_lshrrev_b32 v11, 16, v5 // (op 3353) 
v_cvt_f32_f16 v11, v11 // (op 3353) 
v_mul_f32 v20, v0, v10 // (op 3353) 
v_mul_f32 v21, v0, v11 // (op 3353) 


v_add_f32 v10, v18, v14 // (op 3354) 
v_add_f32 v11, v19, v15 // (op 3354) 
v_add_f32 v12, v20, v16 // (op 3354) 
v_add_f32 v13, v21, v17 // (op 3354) 


v_cvt_f16_f32 v14, v10 // (op 3355) 
v_cvt_f16_f32 v15, v11 // (op 3355) 
v_cvt_f16_f32 v16, v12 // (op 3355) 
v_cvt_f16_f32 v17, v13 // (op 3355) 


v_pack_b32_f16 v10, v14, v15 // (op 3341) 
v_pack_b32_f16 v11, v16, v17 // (op 3341) 

s_lshl_b32 s0, s4, 6 // (op 3341) 
s_add_u32 s0, s0, 32 // (op 3341) 
v_add_u32 v3, v9, s0 // (op 3341) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 3341) Store value





s_lshl_b32 s0, s2, 5 // (op 3361) 
s_mul_i32 s1, 3, s0 // (op 3361) 
s_add_u32 s1, s1, 32 // (op 3361) 

v_add_u32 v3, v8, s1 // (op 3361) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 3361) Load value

v_accvgpr_read v10, a28 // (op 3367) 
v_accvgpr_read v11, a29 // (op 3367) 
v_accvgpr_read v12, a30 // (op 3367) 
v_accvgpr_read v13, a31 // (op 3367) 
v_mul_f32 v14, v7, v10 // (op 3367) 
v_mul_f32 v15, v7, v11 // (op 3367) 
v_mul_f32 v16, v7, v12 // (op 3367) 
v_mul_f32 v17, v7, v13 // (op 3367) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 3358) 
v_lshrrev_b32 v11, 16, v4 // (op 3358) 
v_cvt_f32_f16 v11, v11 // (op 3358) 
v_mul_f32 v18, v0, v10 // (op 3358) 
v_mul_f32 v19, v0, v11 // (op 3358) 
v_cvt_f32_f16 v10, v5 // (op 3358) 
v_lshrrev_b32 v11, 16, v5 // (op 3358) 
v_cvt_f32_f16 v11, v11 // (op 3358) 
v_mul_f32 v20, v0, v10 // (op 3358) 
v_mul_f32 v21, v0, v11 // (op 3358) 


v_add_f32 v10, v18, v14 // (op 3359) 
v_add_f32 v11, v19, v15 // (op 3359) 
v_add_f32 v12, v20, v16 // (op 3359) 
v_add_f32 v13, v21, v17 // (op 3359) 


v_cvt_f16_f32 v14, v10 // (op 3360) 
v_cvt_f16_f32 v15, v11 // (op 3360) 
v_cvt_f16_f32 v16, v12 // (op 3360) 
v_cvt_f16_f32 v17, v13 // (op 3360) 


v_pack_b32_f16 v10, v14, v15 // (op 3343) 
v_pack_b32_f16 v11, v16, v17 // (op 3343) 

s_lshl_b32 s0, s4, 5 // (op 3343) 
s_mul_i32 s1, 3, s0 // (op 3343) 
s_add_u32 s1, s1, 32 // (op 3343) 

v_add_u32 v3, v9, s1 // (op 3343) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 3343) Store value





v_mov_b32 v3, v8 // (op 129) reg expression

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 129) Load value

v_accvgpr_read v10, a0 // (op 125) 
v_accvgpr_read v11, a1 // (op 125) 
v_accvgpr_read v12, a2 // (op 125) 
v_accvgpr_read v13, a3 // (op 125) 
v_mul_f32 v14, v7, v10 // (op 125) 
v_mul_f32 v15, v7, v11 // (op 125) 
v_mul_f32 v16, v7, v12 // (op 125) 
v_mul_f32 v17, v7, v13 // (op 125) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 130) 
v_lshrrev_b32 v11, 16, v4 // (op 130) 
v_cvt_f32_f16 v11, v11 // (op 130) 
v_mul_f32 v18, v0, v10 // (op 130) 
v_mul_f32 v19, v0, v11 // (op 130) 
v_cvt_f32_f16 v10, v5 // (op 130) 
v_lshrrev_b32 v11, 16, v5 // (op 130) 
v_cvt_f32_f16 v11, v11 // (op 130) 
v_mul_f32 v20, v0, v10 // (op 130) 
v_mul_f32 v21, v0, v11 // (op 130) 


v_add_f32 v10, v18, v14 // (op 126) 
v_add_f32 v11, v19, v15 // (op 126) 
v_add_f32 v12, v20, v16 // (op 126) 
v_add_f32 v13, v21, v17 // (op 126) 


v_cvt_f16_f32 v14, v10 // (op 127) 
v_cvt_f16_f32 v15, v11 // (op 127) 
v_cvt_f16_f32 v16, v12 // (op 127) 
v_cvt_f16_f32 v17, v13 // (op 127) 


v_pack_b32_f16 v10, v14, v15 // (op 128) 
v_pack_b32_f16 v11, v16, v17 // (op 128) 

v_mov_b32 v3, v9 // (op 128) reg expression

buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 128) Store value





s_lshl_b32 s0, s2, 5 // (op 3254) 
v_add_u32 v3, v8, s0 // (op 3254) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 3254) Load value

v_accvgpr_read v10, a4 // (op 3250) 
v_accvgpr_read v11, a5 // (op 3250) 
v_accvgpr_read v12, a6 // (op 3250) 
v_accvgpr_read v13, a7 // (op 3250) 
v_mul_f32 v14, v7, v10 // (op 3250) 
v_mul_f32 v15, v7, v11 // (op 3250) 
v_mul_f32 v16, v7, v12 // (op 3250) 
v_mul_f32 v17, v7, v13 // (op 3250) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 3255) 
v_lshrrev_b32 v11, 16, v4 // (op 3255) 
v_cvt_f32_f16 v11, v11 // (op 3255) 
v_mul_f32 v18, v0, v10 // (op 3255) 
v_mul_f32 v19, v0, v11 // (op 3255) 
v_cvt_f32_f16 v10, v5 // (op 3255) 
v_lshrrev_b32 v11, 16, v5 // (op 3255) 
v_cvt_f32_f16 v11, v11 // (op 3255) 
v_mul_f32 v20, v0, v10 // (op 3255) 
v_mul_f32 v21, v0, v11 // (op 3255) 


v_add_f32 v10, v18, v14 // (op 3251) 
v_add_f32 v11, v19, v15 // (op 3251) 
v_add_f32 v12, v20, v16 // (op 3251) 
v_add_f32 v13, v21, v17 // (op 3251) 


v_cvt_f16_f32 v14, v10 // (op 3252) 
v_cvt_f16_f32 v15, v11 // (op 3252) 
v_cvt_f16_f32 v16, v12 // (op 3252) 
v_cvt_f16_f32 v17, v13 // (op 3252) 


v_pack_b32_f16 v10, v14, v15 // (op 3253) 
v_pack_b32_f16 v11, v16, v17 // (op 3253) 

s_lshl_b32 s0, s4, 5 // (op 3253) 
v_add_u32 v3, v9, s0 // (op 3253) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 3253) Store value





s_lshl_b32 s0, s2, 6 // (op 3268) 
v_add_u32 v3, v8, s0 // (op 3268) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 3268) Load value

v_accvgpr_read v10, a8 // (op 3264) 
v_accvgpr_read v11, a9 // (op 3264) 
v_accvgpr_read v12, a10 // (op 3264) 
v_accvgpr_read v13, a11 // (op 3264) 
v_mul_f32 v14, v7, v10 // (op 3264) 
v_mul_f32 v15, v7, v11 // (op 3264) 
v_mul_f32 v16, v7, v12 // (op 3264) 
v_mul_f32 v17, v7, v13 // (op 3264) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 3269) 
v_lshrrev_b32 v11, 16, v4 // (op 3269) 
v_cvt_f32_f16 v11, v11 // (op 3269) 
v_mul_f32 v18, v0, v10 // (op 3269) 
v_mul_f32 v19, v0, v11 // (op 3269) 
v_cvt_f32_f16 v10, v5 // (op 3269) 
v_lshrrev_b32 v11, 16, v5 // (op 3269) 
v_cvt_f32_f16 v11, v11 // (op 3269) 
v_mul_f32 v20, v0, v10 // (op 3269) 
v_mul_f32 v21, v0, v11 // (op 3269) 


v_add_f32 v10, v18, v14 // (op 3265) 
v_add_f32 v11, v19, v15 // (op 3265) 
v_add_f32 v12, v20, v16 // (op 3265) 
v_add_f32 v13, v21, v17 // (op 3265) 


v_cvt_f16_f32 v14, v10 // (op 3266) 
v_cvt_f16_f32 v15, v11 // (op 3266) 
v_cvt_f16_f32 v16, v12 // (op 3266) 
v_cvt_f16_f32 v17, v13 // (op 3266) 


v_pack_b32_f16 v10, v14, v15 // (op 3267) 
v_pack_b32_f16 v11, v16, v17 // (op 3267) 

s_lshl_b32 s0, s4, 6 // (op 3267) 
v_add_u32 v3, v9, s0 // (op 3267) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 3267) Store value





s_lshl_b32 s0, s2, 5 // (op 3282) 
s_mul_i32 s1, 3, s0 // (op 3282) 

v_add_u32 v3, v8, s1 // (op 3282) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 3282) Load value

v_accvgpr_read v10, a12 // (op 3278) 
v_accvgpr_read v11, a13 // (op 3278) 
v_accvgpr_read v12, a14 // (op 3278) 
v_accvgpr_read v13, a15 // (op 3278) 
v_mul_f32 v14, v7, v10 // (op 3278) 
v_mul_f32 v15, v7, v11 // (op 3278) 
v_mul_f32 v16, v7, v12 // (op 3278) 
v_mul_f32 v17, v7, v13 // (op 3278) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 3283) 
v_lshrrev_b32 v11, 16, v4 // (op 3283) 
v_cvt_f32_f16 v11, v11 // (op 3283) 
v_mul_f32 v18, v0, v10 // (op 3283) 
v_mul_f32 v19, v0, v11 // (op 3283) 
v_cvt_f32_f16 v10, v5 // (op 3283) 
v_lshrrev_b32 v11, 16, v5 // (op 3283) 
v_cvt_f32_f16 v11, v11 // (op 3283) 
v_mul_f32 v20, v0, v10 // (op 3283) 
v_mul_f32 v21, v0, v11 // (op 3283) 


v_add_f32 v10, v18, v14 // (op 3279) 
v_add_f32 v11, v19, v15 // (op 3279) 
v_add_f32 v12, v20, v16 // (op 3279) 
v_add_f32 v13, v21, v17 // (op 3279) 


v_cvt_f16_f32 v14, v10 // (op 3280) 
v_cvt_f16_f32 v15, v11 // (op 3280) 
v_cvt_f16_f32 v16, v12 // (op 3280) 
v_cvt_f16_f32 v17, v13 // (op 3280) 


v_pack_b32_f16 v10, v14, v15 // (op 3281) 
v_pack_b32_f16 v11, v16, v17 // (op 3281) 

s_lshl_b32 s0, s4, 5 // (op 3281) 
s_mul_i32 s1, 3, s0 // (op 3281) 

v_add_u32 v3, v9, s1 // (op 3281) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 3281) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ConditionalBottom_0_beta_116:
























s_endpgm  // End of RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM_
.LRR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__end:
.size RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM_, .LRR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__end-RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 180
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 49152 // lds bytes
  .amdhsa_accum_offset 116
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
    macrotile: [128, 128, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [4, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_FP8_FP4_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 49152
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 72
    .vgpr_count: 114
    .agpr_count: 64
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {is-null: true}, {is-null: true}]
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
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 28
        .expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}}
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
        .expression: {type: MagicMultiple, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_24, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 160, size: 4}, rhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}}}
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
        .expression: {type: MagicShiftAndSign, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_24, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 160, size: 4}, rhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_27
        .size: 4
        .offset: 156
        .expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: Subtract_24
        .size: 4
        .offset: 160
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
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

