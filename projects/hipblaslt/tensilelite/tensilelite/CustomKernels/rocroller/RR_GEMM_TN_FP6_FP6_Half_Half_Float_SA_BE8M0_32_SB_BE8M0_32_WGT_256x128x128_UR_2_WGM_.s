// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT



.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM_,@function
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 13816) 
s_mul_hi_i32 s1, s16, s10 // (op 13957) 
v_and_b32 v0, 4294967168, v1 // (op 13878) 
v_and_b32 v3, 15, v1 // (op 13878) 
v_add_lshl_u32 v4, v0, v3, 7 // (op 13878) 


v_and_b32 v0, 63, v1 // (op 13878) 
v_lshrrev_b32 v3, 4, v0 // (op 13878) 

v_lshlrev_b32 v0, 5, v3 // (op 13878) 

v_add_u32 v3, v4, v0 // (op 13878) 


v_mul_lo_u32 v0, v3, 6 // (op 13878) 

v_lshrrev_b32 v3, 3, v0 // (op 13878) 

v_lshrrev_b32 v0, 6, v1 // (op 13936) 
v_and_b32 v4, 1, v0 // (op 13936) 

v_and_b32 v0, 15, v1 // (op 13936) 
v_lshl_add_u32 v5, v4, 6, v0 // (op 13936) 


v_and_b32 v0, 63, v1 // (op 13936) 
v_lshrrev_b32 v4, 4, v0 // (op 13936) 

v_lshlrev_b32 v0, 5, v4 // (op 13936) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 13936) 


v_mul_lo_u32 v0, v4, 6 // (op 13936) 

v_lshrrev_b32 v4, 3, v0 // (op 13936) 

v_and_b32 v0, 4294967168, v1 // (op 13988) 
v_and_b32 v5, 15, v1 // (op 13988) 
v_add_lshl_u32 v6, v0, v5, 7 // (op 13988) 


v_and_b32 v0, 63, v1 // (op 13988) 
v_lshrrev_b32 v5, 4, v0 // (op 13988) 

v_lshlrev_b32 v0, 5, v5 // (op 13988) 

v_add_u32 v5, v6, v0 // (op 13988) 


v_mul_lo_u32 v0, v5, 6 // (op 13988) 

v_lshrrev_b32 v5, 3, v0 // (op 13988) 

v_lshrrev_b32 v0, 6, v1 // (op 14017) 
v_and_b32 v6, 1, v0 // (op 14017) 

v_and_b32 v0, 15, v1 // (op 14017) 
v_lshl_add_u32 v7, v6, 6, v0 // (op 14017) 


v_and_b32 v0, 63, v1 // (op 14017) 
v_lshrrev_b32 v6, 4, v0 // (op 14017) 

v_lshlrev_b32 v0, 5, v6 // (op 14017) 

v_lshl_add_u32 v6, v7, 7, v0 // (op 14017) 


v_mul_lo_u32 v0, v6, 6 // (op 14017) 

v_lshrrev_b32 v6, 3, v0 // (op 14017) 

v_and_b32 v0, 4294967168, v1 // (op 13816) 
s_add_i32 s18, s0, s16 // (op 13816) 

v_and_b32 v7, 63, v1 // (op 13816) 
s_ashr_i32 s0, s18, 31 // (op 13816) 
v_lshrrev_b32 v8, 4, v7 // (op 13816) 
s_cmp_eq_i32 s10, 0 // (op 13816) 
s_mov_b32 s19, scc // (op 13816) 
v_lshlrev_b32 v9, 2, v8 // (op 13816) 

s_cmp_eq_i32 s19, 0 // (op 13816) 
s_cselect_b32 s49, 0, -1 // (op 13816) 

v_add_u32 v8, v0, v9 // (op 13816) 


s_and_b32 s19, s13, 31 // (op 13816) 
v_lshrrev_b32 v0, 2, v7 // (op 13816) 

v_and_b32 v7, 3, v0 // (op 13816) 

v_and_b32 v0, 4294967168, v1 // (op 13957) 
v_and_b32 v9, 63, v1 // (op 13957) 
s_lshl_b32 s50, 1, s19 // (op 13816) 
v_and_b32 v10, 3, v1 // (op 13816) 
s_add_i32 s51, s50, s49 // (op 13816) 


s_and_b32 s49, s0, s51 // (op 13816) 


s_add_i32 s0, s1, s16 // (op 13957) 

s_ashr_i32 s1, s0, 31 // (op 13957) 
v_lshl_add_u32 v11, v7, 2, v10 // (op 13816) 


s_add_i32 s50, s18, s49 // (op 13816) 


v_lshrrev_b32 v7, 6, v1 // (op 13816) 
s_ashr_i32 s18, s50, s19 // (op 13816) 


v_and_b32 v10, 1, v7 // (op 13816) 

s_lshl_b32 s19, s13, 24 // (op 13816) 
v_lshl_add_u32 v7, v10, 6, v11 // (op 13816) 


s_ashr_i32 s49, s19, 31 // (op 13816) 

s_xor_b32 s19, s18, s49 // (op 13816) 

s_sub_i32 s18, s19, s49 // (op 13816) 


s_mul_hi_i32 s19, s18, s14 // (op 13816) 
s_add_i32 s49, s19, s18 // (op 13816) 

s_ashr_i32 s19, s49, 31 // (op 13816) 
s_cmp_eq_i32 s14, 0 // (op 13816) 
s_mov_b32 s50, scc // (op 13816) 
s_cmp_eq_i32 s50, 0 // (op 13816) 
s_cselect_b32 s51, 0, -1 // (op 13816) 

s_and_b32 s50, s15, 31 // (op 13816) 
v_lshrrev_b32 v10, 4, v9 // (op 13957) 
s_cmp_eq_i32 s10, 0 // (op 13957) 
s_mov_b32 s52, scc // (op 13957) 
v_lshlrev_b32 v11, 2, v10 // (op 13957) 

s_cmp_eq_i32 s52, 0 // (op 13957) 
s_cselect_b32 s53, 0, -1 // (op 13957) 

s_lshl_b32 s52, 1, s50 // (op 13816) 
v_add_u32 v10, v0, v11 // (op 13957) 


v_lshrrev_b32 v0, 2, v9 // (op 13957) 

s_add_i32 s54, s52, s51 // (op 13816) 


s_and_b32 s51, s19, s54 // (op 13816) 


s_add_i32 s19, s49, s51 // (op 13816) 


s_ashr_i32 s49, s19, s50 // (op 13816) 


s_lshl_b32 s19, s15, 24 // (op 13816) 
s_ashr_i32 s50, s19, 31 // (op 13816) 

s_xor_b32 s19, s49, s50 // (op 13816) 

s_sub_i32 s49, s19, s50 // (op 13816) 


s_mul_i32 s19, s18, s11 // (op 13816) 
s_sub_u32 s50, s16, s19 // (op 13816) 

s_mul_i32 s19, s49, s11 // (op 13816) 
s_add_u32 s19, s19, s50 // (op 13816) 

s_mul_hi_i32 s50, s16, s9 // (op 13816) 
s_add_i32 s51, s50, s16 // (op 13816) 

s_ashr_i32 s50, s51, 31 // (op 13816) 
s_cmp_eq_i32 s9, 0 // (op 13816) 
s_mov_b32 s52, scc // (op 13816) 
s_cmp_eq_i32 s52, 0 // (op 13816) 
s_cselect_b32 s54, 0, -1 // (op 13816) 

s_and_b32 s52, s13, 31 // (op 13957) 
v_and_b32 v9, 3, v0 // (op 13957) 

s_lshl_b32 s55, 1, s52 // (op 13957) 
v_and_b32 v0, 3, v1 // (op 13957) 
s_add_i32 s56, s55, s53 // (op 13957) 


s_and_b32 s53, s1, s56 // (op 13957) 


v_lshl_add_u32 v11, v9, 2, v0 // (op 13957) 


s_add_i32 s1, s0, s53 // (op 13957) 


v_lshrrev_b32 v0, 6, v1 // (op 13957) 
s_ashr_i32 s0, s1, s52 // (op 13957) 


v_and_b32 v9, 1, v0 // (op 13957) 

s_lshl_b32 s1, s13, 24 // (op 13957) 
v_lshl_add_u32 v0, v9, 6, v11 // (op 13957) 


s_ashr_i32 s52, s1, 31 // (op 13957) 

s_xor_b32 s1, s0, s52 // (op 13957) 

s_sub_i32 s0, s1, s52 // (op 13957) 


s_mul_hi_i32 s1, s0, s14 // (op 13957) 
s_add_i32 s52, s1, s0 // (op 13957) 

s_ashr_i32 s1, s52, 31 // (op 13957) 
s_cmp_eq_i32 s14, 0 // (op 13957) 
s_mov_b32 s53, scc // (op 13957) 
s_cmp_eq_i32 s53, 0 // (op 13957) 
s_cselect_b32 s55, 0, -1 // (op 13957) 

s_and_b32 s53, s15, 31 // (op 13957) 
s_lshl_b32 s56, 1, s53 // (op 13957) 
s_add_i32 s57, s56, s55 // (op 13957) 


s_and_b32 s55, s1, s57 // (op 13957) 


s_add_i32 s1, s52, s55 // (op 13957) 


s_ashr_i32 s52, s1, s53 // (op 13957) 


s_lshl_b32 s1, s15, 24 // (op 13957) 
s_ashr_i32 s53, s1, 31 // (op 13957) 

s_xor_b32 s1, s52, s53 // (op 13957) 

s_sub_i32 s52, s1, s53 // (op 13957) 


s_mul_i32 s1, s0, s11 // (op 13957) 
s_sub_u32 s53, s16, s1 // (op 13957) 

s_mul_i32 s1, s52, s11 // (op 13957) 
s_add_u32 s1, s1, s53 // (op 13957) 

s_mul_hi_i32 s53, s16, s9 // (op 13957) 
s_add_i32 s55, s53, s16 // (op 13957) 

s_ashr_i32 s53, s55, 31 // (op 13957) 
s_cmp_eq_i32 s9, 0 // (op 13957) 
s_mov_b32 s56, scc // (op 13957) 
s_cmp_eq_i32 s56, 0 // (op 13957) 
s_cselect_b32 s57, 0, -1 // (op 13957) 


s_waitcnt lgkmcnt(0)
s_and_b32 s56, s45, 31 // (op 13816) 
s_lshl_b32 s58, 1, s56 // (op 13816) 
s_add_i32 s59, s58, s54 // (op 13816) 


s_and_b32 s54, s50, s59 // (op 13816) 


s_add_i32 s50, s51, s54 // (op 13816) 


s_ashr_i32 s51, s50, s56 // (op 13816) 


s_lshl_b32 s50, s45, 24 // (op 13816) 
s_ashr_i32 s54, s50, 31 // (op 13816) 

s_xor_b32 s50, s51, s54 // (op 13816) 

s_sub_i32 s51, s50, s54 // (op 13816) 


s_and_b32 s50, s45, 31 // (op 13957) 
s_lshl_b32 s54, 1, s50 // (op 13957) 
s_add_i32 s56, s54, s57 // (op 13957) 


s_and_b32 s54, s53, s56 // (op 13957) 


s_add_i32 s53, s55, s54 // (op 13957) 


s_ashr_i32 s54, s53, s50 // (op 13957) 


s_lshl_b32 s50, s45, 24 // (op 13957) 
s_ashr_i32 s53, s50, 31 // (op 13957) 

s_xor_b32 s50, s54, s53 // (op 13957) 

s_sub_i32 s54, s50, s53 // (op 13957) 


s_add_u32 s56, s6, 128 // (op 13816) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 13816) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 13816) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 13816) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 7 // (op 13816) 

s_mul_i32 s50, s11, s56 // (op 13816) 
s_mul_i32 s53, s8, s50 // (op 13816) 

s_mul_i32 s50, s51, s53 // (op 13816) 

s_sub_u32 s53, s16, s50 // (op 13816) 

s_mov_b32 s50, s53 // (op 13816) 
s_mul_hi_i32 s55, s50, s12 // (op 13816) 
s_add_i32 s58, s55, s50 // (op 13816) 


s_ashr_i32 s50, s58, 31 // (op 13816) 
s_cmp_eq_i32 s12, 0 // (op 13816) 
s_mov_b32 s55, scc // (op 13816) 
s_cmp_eq_i32 s55, 0 // (op 13816) 
s_cselect_b32 s59, 0, -1 // (op 13816) 

s_and_b32 s55, s44, 31 // (op 13816) 
s_lshl_b32 s60, 1, s55 // (op 13816) 
s_add_i32 s61, s60, s59 // (op 13816) 


s_and_b32 s59, s50, s61 // (op 13816) 


s_add_i32 s50, s58, s59 // (op 13816) 


s_ashr_i32 s58, s50, s55 // (op 13816) 


s_lshl_b32 s50, s44, 24 // (op 13816) 
s_ashr_i32 s55, s50, 31 // (op 13816) 

s_xor_b32 s50, s58, s55 // (op 13816) 

s_sub_i32 s58, s50, s55 // (op 13816) 


s_mul_i32 s50, s46, s11 // (op 13816) 
s_add_u32 s60, s20, 256 // (op 13816) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 13816) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 13816) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 13816) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 8 // (op 13816) 

s_sub_i32 s55, s60, s50 // (op 13816) 

s_mul_i32 s59, s58, s55 // (op 13816) 

s_sub_u32 s55, s53, s59 // (op 13816) 


s_add_u32 s53, s55, s50 // (op 13816) 


s_cmp_eq_i32 s51, 0 // (op 13816) 
s_mov_b32 s50, scc // (op 13816) 

s_cmp_eq_i32 s50, 0 // (op 13816) 
s_cselect_b32 s51, s53, s19 // (op 13816) 


s_mul_i32 s19, s49, s56 // (op 13816) 


v_lshl_add_u32 v9, s51, 8, v8 // (op 13816) 


s_sub_i32 s49, s18, s19 // (op 13816) 


s_cmp_eq_i32 s50, 0 // (op 13816) 
s_cselect_b32 s51, s58, s49 // (op 13816) 



v_lshl_add_u32 v8, s51, 7, v7 // (op 13816) 


v_mul_lo_u32 v7, v8, s2 // (op 13816) 

v_add_lshl_u32 v8, v9, v7, 1 // (op 13816) 


s_add_u32 s18, s6, 128 // (op 13957) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 13957) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 13957) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 13957) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 7 // (op 13957) 

s_mul_i32 s53, s11, s18 // (op 13957) 
s_mul_i32 s49, s8, s53 // (op 13957) 

s_mul_i32 s53, s54, s49 // (op 13957) 

s_sub_u32 s49, s16, s53 // (op 13957) 

s_mov_b32 s53, s49 // (op 13957) 
s_mul_hi_i32 s50, s53, s12 // (op 13957) 
s_add_i32 s51, s50, s53 // (op 13957) 


s_ashr_i32 s50, s51, 31 // (op 13957) 
s_cmp_eq_i32 s12, 0 // (op 13957) 
s_mov_b32 s53, scc // (op 13957) 
s_cmp_eq_i32 s53, 0 // (op 13957) 
s_cselect_b32 s55, 0, -1 // (op 13957) 

s_and_b32 s53, s44, 31 // (op 13957) 
s_lshl_b32 s56, 1, s53 // (op 13957) 
s_add_i32 s57, s56, s55 // (op 13957) 


s_and_b32 s55, s50, s57 // (op 13957) 


s_add_i32 s50, s51, s55 // (op 13957) 


s_ashr_i32 s51, s50, s53 // (op 13957) 


s_lshl_b32 s50, s44, 24 // (op 13957) 
s_ashr_i32 s53, s50, 31 // (op 13957) 

s_xor_b32 s50, s51, s53 // (op 13957) 

s_sub_i32 s51, s50, s53 // (op 13957) 


s_mul_i32 s50, s46, s11 // (op 13957) 
s_add_u32 s56, s20, 256 // (op 13957) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 13957) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 13957) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 13957) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 8 // (op 13957) 

s_sub_i32 s53, s56, s50 // (op 13957) 

s_mul_i32 s55, s51, s53 // (op 13957) 

s_sub_u32 s53, s49, s55 // (op 13957) 


s_add_u32 s49, s53, s50 // (op 13957) 


s_cmp_eq_i32 s54, 0 // (op 13957) 
s_mov_b32 s50, scc // (op 13957) 

s_cmp_eq_i32 s50, 0 // (op 13957) 
s_cselect_b32 s53, s49, s1 // (op 13957) 


s_mul_i32 s1, s52, s18 // (op 13957) 


v_lshl_add_u32 v7, s53, 8, v10 // (op 13957) 


s_sub_i32 s49, s0, s1 // (op 13957) 


s_cmp_eq_i32 s50, 0 // (op 13957) 
s_cselect_b32 s0, s51, s49 // (op 13957) 



v_lshl_add_u32 v9, s0, 7, v0 // (op 13957) 


v_mul_lo_u32 v0, v9, s4 // (op 13957) 

v_add_lshl_u32 v9, v7, v0, 1 // (op 13957) 


s_mul_i32 s1, s2, s7 // (op 13818) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 13818) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 13818) least significant
s_mul_hi_u32 s19, s2, s6 // (op 13818) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13818) most significant: sum
s_add_u32 s1, s1, s19 // (op 13818) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 13818) 

s_mul_i32 s1, s4, s7 // (op 13959) most significant: low of low * high
s_mul_i32 s49, s5, s6 // (op 13959) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 13959) least significant
s_mul_hi_u32 s50, s4, s6 // (op 13959) most significant: high of low * low
s_add_u32 s1, s1, s49 // (op 13959) most significant: sum
s_add_u32 s1, s1, s50 // (op 13959) most significant: sum


s_lshl_b64 s[50:51], s[0:1], 1 // (op 13959) 

s_mov_b32 s52, s22 // (op 13818) reg expression
s_mov_b32 s53, s23 // (op 13818) reg expression
s_mov_b32 s54, s18 // (op 13818) reg expression
s_mov_b32 s55, 131072 // (op 13818) call()

s_mov_b32 s56, s24 // (op 13959) reg expression

s_mov_b32 s57, s25 // (op 13959) reg expression
s_mov_b32 s58, s50 // (op 13959) reg expression
s_mov_b32 s59, 131072 // (op 13959) call()


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
v_accvgpr_write a64, 0.00000 // (op 2607) call()
v_accvgpr_write a65, 0.00000 // (op 2607) call()
v_accvgpr_write a66, 0.00000 // (op 2607) call()
v_accvgpr_write a67, 0.00000 // (op 2607) call()
v_accvgpr_write a68, 0.00000 // (op 2868) call()
v_accvgpr_write a69, 0.00000 // (op 2868) call()
v_accvgpr_write a70, 0.00000 // (op 2868) call()
v_accvgpr_write a71, 0.00000 // (op 2868) call()
v_accvgpr_write a72, 0.00000 // (op 2869) call()
v_accvgpr_write a73, 0.00000 // (op 2869) call()
v_accvgpr_write a74, 0.00000 // (op 2869) call()
v_accvgpr_write a75, 0.00000 // (op 2869) call()
v_accvgpr_write a76, 0.00000 // (op 2870) call()
v_accvgpr_write a77, 0.00000 // (op 2870) call()
v_accvgpr_write a78, 0.00000 // (op 2870) call()
v_accvgpr_write a79, 0.00000 // (op 2870) call()
v_accvgpr_write a80, 0.00000 // (op 3217) call()
v_accvgpr_write a81, 0.00000 // (op 3217) call()
v_accvgpr_write a82, 0.00000 // (op 3217) call()
v_accvgpr_write a83, 0.00000 // (op 3217) call()
v_accvgpr_write a84, 0.00000 // (op 3478) call()
v_accvgpr_write a85, 0.00000 // (op 3478) call()
v_accvgpr_write a86, 0.00000 // (op 3478) call()
v_accvgpr_write a87, 0.00000 // (op 3478) call()
v_accvgpr_write a88, 0.00000 // (op 3479) call()
v_accvgpr_write a89, 0.00000 // (op 3479) call()
v_accvgpr_write a90, 0.00000 // (op 3479) call()
v_accvgpr_write a91, 0.00000 // (op 3479) call()
v_accvgpr_write a92, 0.00000 // (op 3480) call()
v_accvgpr_write a93, 0.00000 // (op 3480) call()
v_accvgpr_write a94, 0.00000 // (op 3480) call()
v_accvgpr_write a95, 0.00000 // (op 3480) call()
v_accvgpr_write a96, 0.00000 // (op 3827) call()
v_accvgpr_write a97, 0.00000 // (op 3827) call()
v_accvgpr_write a98, 0.00000 // (op 3827) call()
v_accvgpr_write a99, 0.00000 // (op 3827) call()
v_accvgpr_write a100, 0.00000 // (op 4088) call()
v_accvgpr_write a101, 0.00000 // (op 4088) call()
v_accvgpr_write a102, 0.00000 // (op 4088) call()
v_accvgpr_write a103, 0.00000 // (op 4088) call()
v_accvgpr_write a104, 0.00000 // (op 4089) call()
v_accvgpr_write a105, 0.00000 // (op 4089) call()
v_accvgpr_write a106, 0.00000 // (op 4089) call()
v_accvgpr_write a107, 0.00000 // (op 4089) call()
v_accvgpr_write a108, 0.00000 // (op 4090) call()
v_accvgpr_write a109, 0.00000 // (op 4090) call()
v_accvgpr_write a110, 0.00000 // (op 4090) call()
v_accvgpr_write a111, 0.00000 // (op 4090) call()
v_accvgpr_write a112, 0.00000 // (op 4437) call()
v_accvgpr_write a113, 0.00000 // (op 4437) call()
v_accvgpr_write a114, 0.00000 // (op 4437) call()
v_accvgpr_write a115, 0.00000 // (op 4437) call()
v_accvgpr_write a116, 0.00000 // (op 4698) call()
v_accvgpr_write a117, 0.00000 // (op 4698) call()
v_accvgpr_write a118, 0.00000 // (op 4698) call()
v_accvgpr_write a119, 0.00000 // (op 4698) call()
v_accvgpr_write a120, 0.00000 // (op 4699) call()
v_accvgpr_write a121, 0.00000 // (op 4699) call()
v_accvgpr_write a122, 0.00000 // (op 4699) call()
v_accvgpr_write a123, 0.00000 // (op 4699) call()
v_accvgpr_write a124, 0.00000 // (op 4700) call()
v_accvgpr_write a125, 0.00000 // (op 4700) call()
v_accvgpr_write a126, 0.00000 // (op 4700) call()
v_accvgpr_write a127, 0.00000 // (op 4700) call()
s_mul_hi_i32 s0, s16, s10 // (op 13706) 
v_and_b32 v10, 4294967294, v1 // (op 13706) 
s_add_i32 s1, s0, s16 // (op 13706) 

v_and_b32 v11, -2, v1 // (op 13706) 
s_ashr_i32 s0, s1, 31 // (op 13706) 
v_sub_u32 v12, v1, v11 // (op 13706) 

s_cmp_eq_i32 s10, 0 // (op 13706) 
s_mov_b32 s18, scc // (op 13706) 
v_lshlrev_b32 v11, 6, v12 // (op 13706) 

s_cmp_eq_i32 s18, 0 // (op 13706) 
s_cselect_b32 s19, 0, -1 // (op 13706) 

v_mov_b32 v13, 0 // (op 13706) convert
v_mov_b32 v12, v11 // (op 13706) convert

s_and_b32 s18, s13, 31 // (op 13706) 
s_lshl_b32 s22, 1, s18 // (op 13706) 
s_add_i32 s23, s22, s19 // (op 13706) 


s_and_b32 s19, s0, s23 // (op 13706) 


s_add_i32 s0, s1, s19 // (op 13706) 


s_ashr_i32 s1, s0, s18 // (op 13706) 


s_lshl_b32 s0, s13, 24 // (op 13706) 
s_ashr_i32 s18, s0, 31 // (op 13706) 

s_xor_b32 s0, s1, s18 // (op 13706) 

s_sub_i32 s1, s0, s18 // (op 13706) 


s_mul_hi_i32 s0, s1, s14 // (op 13706) 
s_add_i32 s18, s0, s1 // (op 13706) 

s_ashr_i32 s0, s18, 31 // (op 13706) 
s_cmp_eq_i32 s14, 0 // (op 13706) 
s_mov_b32 s19, scc // (op 13706) 
s_cmp_eq_i32 s19, 0 // (op 13706) 
s_cselect_b32 s22, 0, -1 // (op 13706) 

s_and_b32 s19, s15, 31 // (op 13706) 
s_lshl_b32 s23, 1, s19 // (op 13706) 
s_add_i32 s24, s23, s22 // (op 13706) 


s_and_b32 s25, s0, s24 // (op 13706) 


s_add_i32 s0, s18, s25 // (op 13706) 


s_ashr_i32 s18, s0, s19 // (op 13706) 


s_lshl_b32 s0, s15, 24 // (op 13706) 
s_ashr_i32 s19, s0, 31 // (op 13706) 

s_xor_b32 s0, s18, s19 // (op 13706) 

s_sub_i32 s18, s0, s19 // (op 13706) 


s_mul_i32 s0, s1, s11 // (op 13706) 

s_sub_u32 s1, s16, s0 // (op 13706) 

s_mul_i32 s0, s18, s11 // (op 13706) 
s_add_u32 s0, s0, s1 // (op 13706) 


s_mul_hi_i32 s1, s16, s9 // (op 13706) 
s_add_i32 s18, s1, s16 // (op 13706) 

s_ashr_i32 s1, s18, 31 // (op 13706) 
s_cmp_eq_i32 s9, 0 // (op 13706) 
s_mov_b32 s19, scc // (op 13706) 
s_cmp_eq_i32 s19, 0 // (op 13706) 
s_cselect_b32 s22, 0, -1 // (op 13706) 

s_and_b32 s19, s45, 31 // (op 13706) 
s_lshl_b32 s23, 1, s19 // (op 13706) 
s_add_i32 s24, s23, s22 // (op 13706) 


s_and_b32 s25, s1, s24 // (op 13706) 


s_add_i32 s1, s18, s25 // (op 13706) 


s_ashr_i32 s18, s1, s19 // (op 13706) 


s_lshl_b32 s1, s45, 24 // (op 13706) 
s_ashr_i32 s19, s1, 31 // (op 13706) 

s_xor_b32 s1, s18, s19 // (op 13706) 

s_sub_i32 s18, s1, s19 // (op 13706) 


s_add_u32 s22, s6, 128 // (op 13706) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13706) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13706) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13706) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 13706) 

s_mul_i32 s1, s11, s22 // (op 13706) 

s_mul_i32 s19, s8, s1 // (op 13706) 

s_mul_i32 s1, s18, s19 // (op 13706) 

s_sub_u32 s19, s16, s1 // (op 13706) 

s_mov_b32 s1, s19 // (op 13706) 
s_mul_hi_i32 s22, s1, s12 // (op 13706) 
s_add_i32 s23, s22, s1 // (op 13706) 


s_ashr_i32 s1, s23, 31 // (op 13706) 
s_cmp_eq_i32 s12, 0 // (op 13706) 
s_mov_b32 s22, scc // (op 13706) 
s_cmp_eq_i32 s22, 0 // (op 13706) 
s_cselect_b32 s24, 0, -1 // (op 13706) 

s_and_b32 s22, s44, 31 // (op 13706) 
s_lshl_b32 s25, 1, s22 // (op 13706) 
s_add_i32 s47, s25, s24 // (op 13706) 


s_and_b32 s24, s1, s47 // (op 13706) 


s_add_i32 s1, s23, s24 // (op 13706) 


s_ashr_i32 s23, s1, s22 // (op 13706) 


s_lshl_b32 s1, s44, 24 // (op 13706) 
s_ashr_i32 s22, s1, 31 // (op 13706) 

s_xor_b32 s1, s23, s22 // (op 13706) 

s_sub_i32 s23, s1, s22 // (op 13706) 


s_mul_i32 s1, s46, s11 // (op 13706) 
s_add_u32 s24, s20, 256 // (op 13706) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 13706) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 13706) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 13706) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 8 // (op 13706) 

s_sub_i32 s22, s24, s1 // (op 13706) 

s_mul_i32 s24, s23, s22 // (op 13706) 


s_sub_u32 s25, s19, s24 // (op 13706) 


s_add_u32 s19, s25, s1 // (op 13706) 


s_cmp_eq_i32 s18, 0 // (op 13706) 
s_mov_b32 s1, scc // (op 13706) 

s_cmp_eq_i32 s1, 0 // (op 13706) 
s_cselect_b32 s18, s19, s0 // (op 13706) 



v_lshl_add_u32 v11, s18, 8, v10 // (op 13706) 


v_mul_lo_u32 v15, v11, s27 // (op 13706) most significant: low of low * high
v_mul_hi_u32 v10, v11, s26 // (op 13706) most significant: high of low * low
v_mul_lo_u32 v14, v11, s26 // (op 13706) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 13706) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 13706) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 13706) most significant half



v_mul_lo_u32 v11, v15, 6 // (op 13706) most significant: low of high * low
v_mul_hi_u32 v12, v14, 6 // (op 13706) most significant: high of low * low
v_mul_lo_u32 v10, v14, 6 // (op 13706) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 13706) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 13706) 

v_mov_b32 v10, v12 // (op 13706) convert

s_mul_i32 s1, s21, 6 // (op 13708) most significant: low of high * low
s_mul_i32 s0, s20, 6 // (op 13708) least significant
s_mul_hi_u32 s18, s20, 6 // (op 13708) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13708) most significant: sum

s_mul_i32 s19, s26, s1 // (op 13708) most significant: low of low * high
s_mul_i32 s22, s27, s0 // (op 13708) most significant: low of high * low
s_mul_i32 s18, s26, s0 // (op 13708) least significant
s_mul_hi_u32 s23, s26, s0 // (op 13708) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 13708) most significant: sum
s_add_u32 s19, s19, s23 // (op 13708) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 13708) 

s_mov_b32 s48, s28 // (op 13708) reg expression
s_mov_b32 s49, s29 // (op 13708) reg expression
s_mov_b32 s50, s0 // (op 13708) reg expression
s_mov_b32 s51, 131072 // (op 13708) call()

s_mul_hi_i32 s47, s16, s10 // (op 13725) 
v_and_b32 v11, 4294967294, v1 // (op 13725) 
s_add_i32 s0, s47, s16 // (op 13725) 

v_and_b32 v12, -2, v1 // (op 13725) 
s_ashr_i32 s1, s0, 31 // (op 13725) 
v_sub_u32 v13, v1, v12 // (op 13725) 

s_cmp_eq_i32 s10, 0 // (op 13725) 
s_mov_b32 s47, scc // (op 13725) 
v_lshlrev_b32 v12, 6, v13 // (op 13725) 

s_cmp_eq_i32 s47, 0 // (op 13725) 
s_cselect_b32 s18, 0, -1 // (op 13725) 

v_mov_b32 v15, 0 // (op 13725) convert
v_mov_b32 v14, v12 // (op 13725) convert

s_and_b32 s19, s13, 31 // (op 13725) 
s_lshl_b32 s47, 1, s19 // (op 13725) 
s_add_i32 s22, s47, s18 // (op 13725) 


s_and_b32 s18, s1, s22 // (op 13725) 


s_add_i32 s1, s0, s18 // (op 13725) 


s_ashr_i32 s0, s1, s19 // (op 13725) 


s_lshl_b32 s1, s13, 24 // (op 13725) 
s_ashr_i32 s47, s1, 31 // (op 13725) 

s_xor_b32 s1, s0, s47 // (op 13725) 

s_sub_i32 s0, s1, s47 // (op 13725) 


s_mul_hi_i32 s1, s0, s14 // (op 13725) 
s_add_i32 s47, s1, s0 // (op 13725) 

s_ashr_i32 s1, s47, 31 // (op 13725) 
s_cmp_eq_i32 s14, 0 // (op 13725) 
s_mov_b32 s18, scc // (op 13725) 
s_cmp_eq_i32 s18, 0 // (op 13725) 
s_cselect_b32 s19, 0, -1 // (op 13725) 

s_and_b32 s18, s15, 31 // (op 13725) 
s_lshl_b32 s22, 1, s18 // (op 13725) 
s_add_i32 s23, s22, s19 // (op 13725) 


s_and_b32 s19, s1, s23 // (op 13725) 


s_add_i32 s1, s47, s19 // (op 13725) 


s_ashr_i32 s19, s1, s18 // (op 13725) 


s_lshl_b32 s1, s15, 24 // (op 13725) 
s_ashr_i32 s18, s1, 31 // (op 13725) 

s_xor_b32 s1, s19, s18 // (op 13725) 

s_sub_i32 s19, s1, s18 // (op 13725) 


s_mul_i32 s1, s0, s11 // (op 13725) 

s_sub_u32 s0, s16, s1 // (op 13725) 

s_mul_i32 s1, s19, s11 // (op 13725) 
s_add_u32 s1, s1, s0 // (op 13725) 


s_mul_hi_i32 s0, s16, s9 // (op 13725) 
s_add_i32 s47, s0, s16 // (op 13725) 

s_ashr_i32 s0, s47, 31 // (op 13725) 
s_cmp_eq_i32 s9, 0 // (op 13725) 
s_mov_b32 s18, scc // (op 13725) 
s_cmp_eq_i32 s18, 0 // (op 13725) 
s_cselect_b32 s19, 0, -1 // (op 13725) 

s_and_b32 s18, s45, 31 // (op 13725) 
s_lshl_b32 s22, 1, s18 // (op 13725) 
s_add_i32 s23, s22, s19 // (op 13725) 


s_and_b32 s19, s0, s23 // (op 13725) 


s_add_i32 s0, s47, s19 // (op 13725) 


s_ashr_i32 s19, s0, s18 // (op 13725) 


s_lshl_b32 s0, s45, 24 // (op 13725) 
s_ashr_i32 s18, s0, 31 // (op 13725) 

s_xor_b32 s0, s19, s18 // (op 13725) 

s_sub_i32 s19, s0, s18 // (op 13725) 


s_add_u32 s22, s6, 128 // (op 13725) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13725) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13725) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13725) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 13725) 

s_mul_i32 s0, s11, s22 // (op 13725) 

s_mul_i32 s18, s8, s0 // (op 13725) 

s_mul_i32 s0, s19, s18 // (op 13725) 

s_sub_u32 s18, s16, s0 // (op 13725) 

s_mov_b32 s0, s18 // (op 13725) 
s_mul_hi_i32 s47, s0, s12 // (op 13725) 
s_add_i32 s22, s47, s0 // (op 13725) 


s_ashr_i32 s0, s22, 31 // (op 13725) 
s_cmp_eq_i32 s12, 0 // (op 13725) 
s_mov_b32 s47, scc // (op 13725) 
s_cmp_eq_i32 s47, 0 // (op 13725) 
s_cselect_b32 s23, 0, -1 // (op 13725) 

s_and_b32 s47, s44, 31 // (op 13725) 
s_lshl_b32 s24, 1, s47 // (op 13725) 
s_add_i32 s25, s24, s23 // (op 13725) 


s_and_b32 s23, s0, s25 // (op 13725) 


s_add_i32 s0, s22, s23 // (op 13725) 


s_ashr_i32 s22, s0, s47 // (op 13725) 


s_lshl_b32 s0, s44, 24 // (op 13725) 
s_ashr_i32 s47, s0, 31 // (op 13725) 

s_xor_b32 s0, s22, s47 // (op 13725) 

s_sub_i32 s22, s0, s47 // (op 13725) 


s_mul_i32 s0, s46, s11 // (op 13725) 
s_add_u32 s24, s20, 256 // (op 13725) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 13725) End of Int64 add, unlocking SCC
s_sub_u32 s60, s24, 1 // (op 13725) least significant half; sets scc
s_subb_u32 s61, s25, 0 // (op 13725) most significant half; uses scc

s_ashr_i64 s[24:25], s[60:61], 8 // (op 13725) 

s_sub_i32 s23, s24, s0 // (op 13725) 

s_mul_i32 s47, s22, s23 // (op 13725) 


s_sub_u32 s22, s18, s47 // (op 13725) 


s_add_u32 s18, s22, s0 // (op 13725) 


s_cmp_eq_i32 s19, 0 // (op 13725) 
s_mov_b32 s0, scc // (op 13725) 

s_cmp_eq_i32 s0, 0 // (op 13725) 
s_cselect_b32 s19, s18, s1 // (op 13725) 



v_lshl_add_u32 v12, s19, 8, v11 // (op 13725) 


v_mul_lo_u32 v17, v12, s27 // (op 13725) most significant: low of low * high
v_mul_hi_u32 v11, v12, s26 // (op 13725) most significant: high of low * low
v_mul_lo_u32 v16, v12, s26 // (op 13725) least significant: low of low * low
v_add_u32 v17, v17, v11 // (op 13725) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 13725) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 13725) most significant half



v_mul_lo_u32 v15, v17, 6 // (op 13725) most significant: low of high * low
v_mul_hi_u32 v11, v16, 6 // (op 13725) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 13725) least significant: low of low * low
v_add_u32 v15, v15, v11 // (op 13725) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 13725) 

v_mov_b32 v11, v12 // (op 13725) convert

s_mul_i32 s1, s21, 6 // (op 13727) most significant: low of high * low
s_mul_i32 s0, s20, 6 // (op 13727) least significant
s_mul_hi_u32 s47, s20, 6 // (op 13727) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13727) most significant: sum

s_mul_i32 s19, s26, s1 // (op 13727) most significant: low of low * high
s_mul_i32 s47, s27, s0 // (op 13727) most significant: low of high * low
s_mul_i32 s18, s26, s0 // (op 13727) least significant
s_mul_hi_u32 s22, s26, s0 // (op 13727) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 13727) most significant: sum
s_add_u32 s19, s19, s22 // (op 13727) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 13727) 

s_mov_b32 s48, s28 // (op 13727) reg expression
s_mov_b32 s49, s29 // (op 13727) reg expression
s_mov_b32 s50, s0 // (op 13727) reg expression
s_mov_b32 s51, 131072 // (op 13727) call()

s_mul_hi_i32 s47, s16, s10 // (op 13770) 
v_lshrrev_b32 v12, 1, v1 // (op 13770) 
s_add_i32 s0, s47, s16 // (op 13770) 

v_and_b32 v13, -2, v1 // (op 13770) 
s_ashr_i32 s1, s0, 31 // (op 13770) 
v_sub_u32 v14, v1, v13 // (op 13770) 

s_cmp_eq_i32 s10, 0 // (op 13770) 
s_mov_b32 s47, scc // (op 13770) 
v_lshlrev_b32 v13, 6, v14 // (op 13770) 

s_cmp_eq_i32 s47, 0 // (op 13770) 
s_cselect_b32 s18, 0, -1 // (op 13770) 

v_mov_b32 v15, 0 // (op 13770) convert
v_mov_b32 v14, v13 // (op 13770) convert

s_and_b32 s19, s13, 31 // (op 13770) 
s_lshl_b32 s47, 1, s19 // (op 13770) 
s_add_i32 s22, s47, s18 // (op 13770) 


s_and_b32 s18, s1, s22 // (op 13770) 


s_add_i32 s1, s0, s18 // (op 13770) 


s_ashr_i32 s0, s1, s19 // (op 13770) 


s_lshl_b32 s1, s13, 24 // (op 13770) 
s_ashr_i32 s47, s1, 31 // (op 13770) 

s_xor_b32 s1, s0, s47 // (op 13770) 

s_sub_i32 s0, s1, s47 // (op 13770) 


s_mul_hi_i32 s1, s0, s14 // (op 13770) 
s_add_i32 s47, s1, s0 // (op 13770) 

s_ashr_i32 s1, s47, 31 // (op 13770) 
s_cmp_eq_i32 s14, 0 // (op 13770) 
s_mov_b32 s18, scc // (op 13770) 
s_cmp_eq_i32 s18, 0 // (op 13770) 
s_cselect_b32 s19, 0, -1 // (op 13770) 

s_and_b32 s18, s15, 31 // (op 13770) 
s_lshl_b32 s22, 1, s18 // (op 13770) 
s_add_i32 s23, s22, s19 // (op 13770) 


s_and_b32 s19, s1, s23 // (op 13770) 


s_add_i32 s1, s47, s19 // (op 13770) 


s_ashr_i32 s19, s1, s18 // (op 13770) 


s_lshl_b32 s1, s15, 24 // (op 13770) 
s_ashr_i32 s18, s1, 31 // (op 13770) 

s_xor_b32 s1, s19, s18 // (op 13770) 

s_sub_i32 s19, s1, s18 // (op 13770) 


s_add_u32 s22, s6, 128 // (op 13770) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13770) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13770) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13770) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 13770) 

s_mul_i32 s1, s19, s22 // (op 13770) 

s_sub_i32 s47, s0, s1 // (op 13770) 


s_mul_hi_i32 s0, s16, s9 // (op 13770) 
s_add_i32 s1, s0, s16 // (op 13770) 

s_ashr_i32 s0, s1, 31 // (op 13770) 
s_cmp_eq_i32 s9, 0 // (op 13770) 
s_mov_b32 s18, scc // (op 13770) 
s_cmp_eq_i32 s18, 0 // (op 13770) 
s_cselect_b32 s19, 0, -1 // (op 13770) 

s_and_b32 s18, s45, 31 // (op 13770) 
s_lshl_b32 s24, 1, s18 // (op 13770) 
s_add_i32 s25, s24, s19 // (op 13770) 


s_and_b32 s19, s0, s25 // (op 13770) 


s_add_i32 s0, s1, s19 // (op 13770) 


s_ashr_i32 s1, s0, s18 // (op 13770) 


s_lshl_b32 s0, s45, 24 // (op 13770) 
s_ashr_i32 s18, s0, 31 // (op 13770) 

s_xor_b32 s0, s1, s18 // (op 13770) 

s_sub_i32 s1, s0, s18 // (op 13770) 


s_mul_i32 s0, s11, s22 // (op 13770) 

s_mul_i32 s18, s8, s0 // (op 13770) 

s_mul_i32 s0, s1, s18 // (op 13770) 

s_sub_u32 s18, s16, s0 // (op 13770) 

s_mul_hi_i32 s0, s18, s12 // (op 13770) 
s_add_i32 s19, s0, s18 // (op 13770) 


s_ashr_i32 s0, s19, 31 // (op 13770) 
s_cmp_eq_i32 s12, 0 // (op 13770) 
s_mov_b32 s18, scc // (op 13770) 
s_cmp_eq_i32 s18, 0 // (op 13770) 
s_cselect_b32 s22, 0, -1 // (op 13770) 

s_and_b32 s18, s44, 31 // (op 13770) 
s_lshl_b32 s23, 1, s18 // (op 13770) 
s_add_i32 s24, s23, s22 // (op 13770) 


s_and_b32 s25, s0, s24 // (op 13770) 


s_add_i32 s0, s19, s25 // (op 13770) 


s_ashr_i32 s19, s0, s18 // (op 13770) 


s_lshl_b32 s0, s44, 24 // (op 13770) 
s_ashr_i32 s18, s0, 31 // (op 13770) 

s_xor_b32 s0, s19, s18 // (op 13770) 

s_sub_i32 s19, s0, s18 // (op 13770) 


s_cmp_eq_i32 s1, 0 // (op 13770) 
s_mov_b32 s0, scc // (op 13770) 

s_cmp_eq_i32 s0, 0 // (op 13770) 
s_cselect_b32 s1, s19, s47 // (op 13770) 



v_lshl_add_u32 v13, s1, 7, v12 // (op 13770) 


v_mul_lo_u32 v17, v13, s31 // (op 13770) most significant: low of low * high
v_mul_hi_u32 v12, v13, s30 // (op 13770) most significant: high of low * low
v_mul_lo_u32 v16, v13, s30 // (op 13770) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 13770) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 13770) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 13770) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 13770) most significant: low of high * low
v_mul_hi_u32 v16, v12, 6 // (op 13770) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 13770) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 13770) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 13770) 

v_mov_b32 v14, v12 // (op 13770) convert

s_mul_i32 s1, s7, 6 // (op 13772) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 13772) least significant
s_mul_hi_u32 s47, s6, 6 // (op 13772) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13772) most significant: sum

s_mul_i32 s19, s30, s1 // (op 13772) most significant: low of low * high
s_mul_i32 s47, s31, s0 // (op 13772) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 13772) least significant
s_mul_hi_u32 s22, s30, s0 // (op 13772) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 13772) most significant: sum
s_add_u32 s19, s19, s22 // (op 13772) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 13772) 

s_mov_b32 s60, s32 // (op 13772) reg expression
s_mov_b32 s61, s33 // (op 13772) reg expression
s_mov_b32 s62, s0 // (op 13772) reg expression
s_mov_b32 s63, 131072 // (op 13772) call()

s_mul_hi_i32 s47, s16, s10 // (op 13789) 
v_lshrrev_b32 v12, 1, v1 // (op 13789) 
s_add_i32 s0, s47, s16 // (op 13789) 

v_and_b32 v13, -2, v1 // (op 13789) 
s_ashr_i32 s1, s0, 31 // (op 13789) 
v_sub_u32 v15, v1, v13 // (op 13789) 

s_cmp_eq_i32 s10, 0 // (op 13789) 
s_mov_b32 s47, scc // (op 13789) 
v_lshlrev_b32 v13, 6, v15 // (op 13789) 

s_cmp_eq_i32 s47, 0 // (op 13789) 
s_cselect_b32 s18, 0, -1 // (op 13789) 

v_mov_b32 v17, 0 // (op 13789) convert
v_mov_b32 v16, v13 // (op 13789) convert

s_and_b32 s19, s13, 31 // (op 13789) 
s_lshl_b32 s47, 1, s19 // (op 13789) 
s_add_i32 s22, s47, s18 // (op 13789) 


s_and_b32 s18, s1, s22 // (op 13789) 


s_add_i32 s1, s0, s18 // (op 13789) 


s_ashr_i32 s0, s1, s19 // (op 13789) 


s_lshl_b32 s1, s13, 24 // (op 13789) 
s_ashr_i32 s47, s1, 31 // (op 13789) 

s_xor_b32 s1, s0, s47 // (op 13789) 

s_sub_i32 s0, s1, s47 // (op 13789) 


s_mul_hi_i32 s1, s0, s14 // (op 13789) 
s_add_i32 s47, s1, s0 // (op 13789) 

s_ashr_i32 s1, s47, 31 // (op 13789) 
s_cmp_eq_i32 s14, 0 // (op 13789) 
s_mov_b32 s18, scc // (op 13789) 
s_cmp_eq_i32 s18, 0 // (op 13789) 
s_cselect_b32 s19, 0, -1 // (op 13789) 

s_and_b32 s18, s15, 31 // (op 13789) 
s_lshl_b32 s22, 1, s18 // (op 13789) 
s_add_i32 s23, s22, s19 // (op 13789) 


s_and_b32 s19, s1, s23 // (op 13789) 


s_add_i32 s1, s47, s19 // (op 13789) 


s_ashr_i32 s19, s1, s18 // (op 13789) 


s_lshl_b32 s1, s15, 24 // (op 13789) 
s_ashr_i32 s18, s1, 31 // (op 13789) 

s_xor_b32 s1, s19, s18 // (op 13789) 

s_sub_i32 s19, s1, s18 // (op 13789) 


s_add_u32 s22, s6, 128 // (op 13789) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13789) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13789) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13789) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 13789) 

s_mul_i32 s1, s19, s22 // (op 13789) 

s_sub_i32 s47, s0, s1 // (op 13789) 


s_mul_hi_i32 s0, s16, s9 // (op 13789) 
s_add_i32 s1, s0, s16 // (op 13789) 

s_ashr_i32 s0, s1, 31 // (op 13789) 
s_cmp_eq_i32 s9, 0 // (op 13789) 
s_mov_b32 s18, scc // (op 13789) 
s_cmp_eq_i32 s18, 0 // (op 13789) 
s_cselect_b32 s19, 0, -1 // (op 13789) 

s_and_b32 s18, s45, 31 // (op 13789) 
s_lshl_b32 s24, 1, s18 // (op 13789) 
s_add_i32 s25, s24, s19 // (op 13789) 


s_and_b32 s19, s0, s25 // (op 13789) 


s_add_i32 s0, s1, s19 // (op 13789) 


s_ashr_i32 s1, s0, s18 // (op 13789) 


s_lshl_b32 s0, s45, 24 // (op 13789) 
s_ashr_i32 s18, s0, 31 // (op 13789) 

s_xor_b32 s0, s1, s18 // (op 13789) 

s_sub_i32 s1, s0, s18 // (op 13789) 


s_mul_i32 s0, s11, s22 // (op 13789) 

s_mul_i32 s18, s8, s0 // (op 13789) 

s_mul_i32 s0, s1, s18 // (op 13789) 

s_sub_u32 s18, s16, s0 // (op 13789) 

s_mul_hi_i32 s0, s18, s12 // (op 13789) 
s_add_i32 s19, s0, s18 // (op 13789) 


s_ashr_i32 s0, s19, 31 // (op 13789) 
s_cmp_eq_i32 s12, 0 // (op 13789) 
s_mov_b32 s18, scc // (op 13789) 
s_cmp_eq_i32 s18, 0 // (op 13789) 
s_cselect_b32 s22, 0, -1 // (op 13789) 

s_and_b32 s18, s44, 31 // (op 13789) 
s_lshl_b32 s23, 1, s18 // (op 13789) 
s_add_i32 s24, s23, s22 // (op 13789) 


s_and_b32 s25, s0, s24 // (op 13789) 


s_add_i32 s0, s19, s25 // (op 13789) 


s_ashr_i32 s19, s0, s18 // (op 13789) 


s_lshl_b32 s0, s44, 24 // (op 13789) 
s_ashr_i32 s18, s0, 31 // (op 13789) 

s_xor_b32 s0, s19, s18 // (op 13789) 

s_sub_i32 s19, s0, s18 // (op 13789) 


s_cmp_eq_i32 s1, 0 // (op 13789) 
s_mov_b32 s0, scc // (op 13789) 

s_cmp_eq_i32 s0, 0 // (op 13789) 
s_cselect_b32 s1, s19, s47 // (op 13789) 



v_lshl_add_u32 v13, s1, 7, v12 // (op 13789) 


v_mul_lo_u32 v19, v13, s31 // (op 13789) most significant: low of low * high
v_mul_hi_u32 v12, v13, s30 // (op 13789) most significant: high of low * low
v_mul_lo_u32 v18, v13, s30 // (op 13789) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 13789) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v18 // (op 13789) least significant half
v_addc_co_u32 v13, s[0:1], v17, v19, s[0:1] // (op 13789) most significant half



v_mul_lo_u32 v17, v13, 6 // (op 13789) most significant: low of high * low
v_mul_hi_u32 v15, v12, 6 // (op 13789) most significant: high of low * low
v_mul_lo_u32 v16, v12, 6 // (op 13789) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 13789) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 13789) 

v_mov_b32 v15, v12 // (op 13789) convert

s_mul_i32 s1, s7, 6 // (op 13791) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 13791) least significant
s_mul_hi_u32 s47, s6, 6 // (op 13791) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13791) most significant: sum

s_mul_i32 s19, s30, s1 // (op 13791) most significant: low of low * high
s_mul_i32 s47, s31, s0 // (op 13791) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 13791) least significant
s_mul_hi_u32 s22, s30, s0 // (op 13791) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 13791) most significant: sum
s_add_u32 s19, s19, s22 // (op 13791) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 13791) 

s_mov_b32 s60, s32 // (op 13791) reg expression
s_mov_b32 s61, s33 // (op 13791) reg expression
s_mov_b32 s62, s0 // (op 13791) reg expression
s_mov_b32 s63, 131072 // (op 13791) call()

s_mul_hi_i32 s47, s16, s10 // (op 14039) 
v_and_b32 v12, 63, v1 // (op 14039) 
s_add_i32 s0, s47, s16 // (op 14039) 

v_lshrrev_b32 v13, 4, v12 // (op 14039) 
s_ashr_i32 s1, s0, 31 // (op 14039) 
v_and_b32 v16, 3, v13 // (op 14039) 

s_cmp_eq_i32 s10, 0 // (op 14039) 
s_mov_b32 s47, scc // (op 14039) 
v_and_b32 v13, 15, v1 // (op 14039) 
s_cmp_eq_i32 s47, 0 // (op 14039) 
s_cselect_b32 s18, 0, -1 // (op 14039) 

v_lshl_add_u32 v17, v16, 4, v13 // (op 14039) 


s_and_b32 s19, s13, 31 // (op 14039) 
v_and_b32 v13, 4294967168, v1 // (op 14039) 
s_lshl_b32 s47, 1, s19 // (op 14039) 
v_add_u32 v16, v13, v17 // (op 14039) 


s_add_i32 s22, s47, s18 // (op 14039) 


v_lshrrev_b32 v13, 6, v12 // (op 14039) 

s_and_b32 s18, s1, s22 // (op 14039) 


v_lshlrev_b32 v12, 2, v13 // (op 14039) 

s_add_i32 s1, s0, s18 // (op 14039) 


s_ashr_i32 s0, s1, s19 // (op 14039) 


s_lshl_b32 s1, s13, 24 // (op 14039) 
s_ashr_i32 s47, s1, 31 // (op 14039) 

s_xor_b32 s1, s0, s47 // (op 14039) 

s_sub_i32 s0, s1, s47 // (op 14039) 


s_mul_hi_i32 s1, s0, s14 // (op 14039) 
s_add_i32 s47, s1, s0 // (op 14039) 

s_ashr_i32 s1, s47, 31 // (op 14039) 
s_cmp_eq_i32 s14, 0 // (op 14039) 
s_mov_b32 s18, scc // (op 14039) 
s_cmp_eq_i32 s18, 0 // (op 14039) 
s_cselect_b32 s19, 0, -1 // (op 14039) 

s_and_b32 s18, s15, 31 // (op 14039) 
s_lshl_b32 s22, 1, s18 // (op 14039) 
s_add_i32 s23, s22, s19 // (op 14039) 


s_and_b32 s19, s1, s23 // (op 14039) 


s_add_i32 s1, s47, s19 // (op 14039) 


s_ashr_i32 s19, s1, s18 // (op 14039) 


s_lshl_b32 s1, s15, 24 // (op 14039) 
s_ashr_i32 s18, s1, 31 // (op 14039) 

s_xor_b32 s1, s19, s18 // (op 14039) 

s_sub_i32 s19, s1, s18 // (op 14039) 


s_mul_i32 s1, s0, s11 // (op 14039) 

s_sub_u32 s0, s16, s1 // (op 14039) 

s_mul_i32 s1, s19, s11 // (op 14039) 
s_add_u32 s1, s1, s0 // (op 14039) 


s_mul_hi_i32 s0, s16, s9 // (op 14039) 
s_add_i32 s47, s0, s16 // (op 14039) 

s_ashr_i32 s0, s47, 31 // (op 14039) 
s_cmp_eq_i32 s9, 0 // (op 14039) 
s_mov_b32 s18, scc // (op 14039) 
s_cmp_eq_i32 s18, 0 // (op 14039) 
s_cselect_b32 s19, 0, -1 // (op 14039) 

s_and_b32 s18, s45, 31 // (op 14039) 
s_lshl_b32 s22, 1, s18 // (op 14039) 
s_add_i32 s23, s22, s19 // (op 14039) 


s_and_b32 s19, s0, s23 // (op 14039) 


s_add_i32 s0, s47, s19 // (op 14039) 


s_ashr_i32 s19, s0, s18 // (op 14039) 


s_lshl_b32 s0, s45, 24 // (op 14039) 
s_ashr_i32 s18, s0, 31 // (op 14039) 

s_xor_b32 s0, s19, s18 // (op 14039) 

s_sub_i32 s19, s0, s18 // (op 14039) 


s_add_u32 s22, s6, 128 // (op 14039) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 14039) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 14039) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 14039) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 14039) 

s_mul_i32 s0, s11, s22 // (op 14039) 

s_mul_i32 s18, s8, s0 // (op 14039) 

s_mul_i32 s0, s19, s18 // (op 14039) 

s_sub_u32 s18, s16, s0 // (op 14039) 

s_mov_b32 s0, s18 // (op 14039) 
s_mul_hi_i32 s47, s0, s12 // (op 14039) 
s_add_i32 s22, s47, s0 // (op 14039) 


s_ashr_i32 s0, s22, 31 // (op 14039) 
s_cmp_eq_i32 s12, 0 // (op 14039) 
s_mov_b32 s47, scc // (op 14039) 
s_cmp_eq_i32 s47, 0 // (op 14039) 
s_cselect_b32 s23, 0, -1 // (op 14039) 

s_and_b32 s47, s44, 31 // (op 14039) 
s_lshl_b32 s24, 1, s47 // (op 14039) 
s_add_i32 s25, s24, s23 // (op 14039) 


s_and_b32 s23, s0, s25 // (op 14039) 


s_add_i32 s0, s22, s23 // (op 14039) 


s_ashr_i32 s22, s0, s47 // (op 14039) 


s_lshl_b32 s0, s44, 24 // (op 14039) 
s_ashr_i32 s47, s0, 31 // (op 14039) 

s_xor_b32 s0, s22, s47 // (op 14039) 

s_sub_i32 s22, s0, s47 // (op 14039) 


s_mul_i32 s0, s46, s11 // (op 14039) 
s_add_u32 s24, s20, 256 // (op 14039) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 14039) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 14039) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 14039) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 8 // (op 14039) 

s_sub_i32 s23, s24, s0 // (op 14039) 

s_mul_i32 s47, s22, s23 // (op 14039) 


s_sub_u32 s22, s18, s47 // (op 14039) 


s_add_u32 s18, s22, s0 // (op 14039) 


s_cmp_eq_i32 s19, 0 // (op 14039) 
s_mov_b32 s0, scc // (op 14039) 

s_cmp_eq_i32 s0, 0 // (op 14039) 
s_cselect_b32 s19, s18, s1 // (op 14039) 



v_lshl_add_u32 v13, s19, 8, v16 // (op 14039) 


v_mul_lo_u32 v16, v13, s34 // (op 14039) 
v_add_u32 v16, v16, v12 // (op 14039) 


s_mul_i32 s1, s34, s21 // (op 14041) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 14041) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 14041) least significant
s_mul_hi_u32 s18, s34, s20 // (op 14041) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 14041) most significant: sum
s_add_u32 s1, s1, s18 // (op 14041) most significant: sum


s_mov_b32 s64, s36 // (op 14041) reg expression
s_mov_b32 s65, s37 // (op 14041) reg expression
s_mov_b32 s66, s0 // (op 14041) reg expression
s_mov_b32 s67, 131072 // (op 14041) call()

s_mul_hi_i32 s47, s16, s10 // (op 14060) 
v_and_b32 v12, 63, v1 // (op 14060) 
s_add_i32 s0, s47, s16 // (op 14060) 

v_lshrrev_b32 v13, 4, v12 // (op 14060) 
s_ashr_i32 s1, s0, 31 // (op 14060) 
v_and_b32 v17, 3, v13 // (op 14060) 

s_cmp_eq_i32 s10, 0 // (op 14060) 
s_mov_b32 s47, scc // (op 14060) 
v_and_b32 v13, 15, v1 // (op 14060) 
s_cmp_eq_i32 s47, 0 // (op 14060) 
s_cselect_b32 s18, 0, -1 // (op 14060) 

v_lshl_add_u32 v18, v17, 4, v13 // (op 14060) 


s_and_b32 s19, s13, 31 // (op 14060) 
v_and_b32 v13, 4294967168, v1 // (op 14060) 
s_lshl_b32 s47, 1, s19 // (op 14060) 
v_add_u32 v17, v13, v18 // (op 14060) 


s_add_i32 s22, s47, s18 // (op 14060) 


v_lshrrev_b32 v13, 6, v12 // (op 14060) 

s_and_b32 s18, s1, s22 // (op 14060) 


v_lshlrev_b32 v12, 2, v13 // (op 14060) 

s_add_i32 s1, s0, s18 // (op 14060) 


s_ashr_i32 s0, s1, s19 // (op 14060) 


s_lshl_b32 s1, s13, 24 // (op 14060) 
s_ashr_i32 s47, s1, 31 // (op 14060) 

s_xor_b32 s1, s0, s47 // (op 14060) 

s_sub_i32 s0, s1, s47 // (op 14060) 


s_mul_hi_i32 s1, s0, s14 // (op 14060) 
s_add_i32 s47, s1, s0 // (op 14060) 

s_ashr_i32 s1, s47, 31 // (op 14060) 
s_cmp_eq_i32 s14, 0 // (op 14060) 
s_mov_b32 s18, scc // (op 14060) 
s_cmp_eq_i32 s18, 0 // (op 14060) 
s_cselect_b32 s19, 0, -1 // (op 14060) 

s_and_b32 s18, s15, 31 // (op 14060) 
s_lshl_b32 s22, 1, s18 // (op 14060) 
s_add_i32 s23, s22, s19 // (op 14060) 


s_and_b32 s19, s1, s23 // (op 14060) 


s_add_i32 s1, s47, s19 // (op 14060) 


s_ashr_i32 s19, s1, s18 // (op 14060) 


s_lshl_b32 s1, s15, 24 // (op 14060) 
s_ashr_i32 s18, s1, 31 // (op 14060) 

s_xor_b32 s1, s19, s18 // (op 14060) 

s_sub_i32 s19, s1, s18 // (op 14060) 


s_mul_i32 s1, s0, s11 // (op 14060) 

s_sub_u32 s0, s16, s1 // (op 14060) 

s_mul_i32 s1, s19, s11 // (op 14060) 
s_add_u32 s1, s1, s0 // (op 14060) 


s_mul_hi_i32 s0, s16, s9 // (op 14060) 
s_add_i32 s47, s0, s16 // (op 14060) 

s_ashr_i32 s0, s47, 31 // (op 14060) 
s_cmp_eq_i32 s9, 0 // (op 14060) 
s_mov_b32 s18, scc // (op 14060) 
s_cmp_eq_i32 s18, 0 // (op 14060) 
s_cselect_b32 s19, 0, -1 // (op 14060) 

s_and_b32 s18, s45, 31 // (op 14060) 
s_lshl_b32 s22, 1, s18 // (op 14060) 
s_add_i32 s23, s22, s19 // (op 14060) 


s_and_b32 s19, s0, s23 // (op 14060) 


s_add_i32 s0, s47, s19 // (op 14060) 


s_ashr_i32 s19, s0, s18 // (op 14060) 


s_lshl_b32 s0, s45, 24 // (op 14060) 
s_ashr_i32 s18, s0, 31 // (op 14060) 

s_xor_b32 s0, s19, s18 // (op 14060) 

s_sub_i32 s19, s0, s18 // (op 14060) 


s_add_u32 s22, s6, 128 // (op 14060) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 14060) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 14060) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 14060) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 14060) 

s_mul_i32 s0, s11, s22 // (op 14060) 

s_mul_i32 s18, s8, s0 // (op 14060) 

s_mul_i32 s0, s19, s18 // (op 14060) 

s_sub_u32 s18, s16, s0 // (op 14060) 

s_mov_b32 s0, s18 // (op 14060) 
s_mul_hi_i32 s47, s0, s12 // (op 14060) 
s_add_i32 s22, s47, s0 // (op 14060) 


s_ashr_i32 s0, s22, 31 // (op 14060) 
s_cmp_eq_i32 s12, 0 // (op 14060) 
s_mov_b32 s47, scc // (op 14060) 
s_cmp_eq_i32 s47, 0 // (op 14060) 
s_cselect_b32 s23, 0, -1 // (op 14060) 

s_and_b32 s47, s44, 31 // (op 14060) 
s_lshl_b32 s24, 1, s47 // (op 14060) 
s_add_i32 s25, s24, s23 // (op 14060) 


s_and_b32 s23, s0, s25 // (op 14060) 


s_add_i32 s0, s22, s23 // (op 14060) 


s_ashr_i32 s22, s0, s47 // (op 14060) 


s_lshl_b32 s0, s44, 24 // (op 14060) 
s_ashr_i32 s47, s0, 31 // (op 14060) 

s_xor_b32 s0, s22, s47 // (op 14060) 

s_sub_i32 s22, s0, s47 // (op 14060) 


s_mul_i32 s0, s46, s11 // (op 14060) 
s_add_u32 s24, s20, 256 // (op 14060) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 14060) End of Int64 add, unlocking SCC
s_sub_u32 s68, s24, 1 // (op 14060) least significant half; sets scc
s_subb_u32 s69, s25, 0 // (op 14060) most significant half; uses scc

s_ashr_i64 s[24:25], s[68:69], 8 // (op 14060) 

s_sub_i32 s23, s24, s0 // (op 14060) 

s_mul_i32 s47, s22, s23 // (op 14060) 


s_sub_u32 s22, s18, s47 // (op 14060) 


s_add_u32 s18, s22, s0 // (op 14060) 


s_cmp_eq_i32 s19, 0 // (op 14060) 
s_mov_b32 s0, scc // (op 14060) 

s_cmp_eq_i32 s0, 0 // (op 14060) 
s_cselect_b32 s19, s18, s1 // (op 14060) 



v_lshl_add_u32 v13, s19, 8, v17 // (op 14060) 


v_mul_lo_u32 v17, v13, s34 // (op 14060) 
v_add_u32 v17, v17, v12 // (op 14060) 


s_mul_i32 s1, s34, s21 // (op 14062) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 14062) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 14062) least significant
s_mul_hi_u32 s18, s34, s20 // (op 14062) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 14062) most significant: sum
s_add_u32 s1, s1, s18 // (op 14062) most significant: sum


s_mov_b32 s64, s36 // (op 14062) reg expression
s_mov_b32 s65, s37 // (op 14062) reg expression
s_mov_b32 s66, s0 // (op 14062) reg expression
s_mov_b32 s67, 131072 // (op 14062) call()

s_mul_hi_i32 s47, s16, s10 // (op 14091) 
v_and_b32 v12, 63, v1 // (op 14091) 
s_add_i32 s0, s47, s16 // (op 14091) 

v_lshrrev_b32 v13, 4, v12 // (op 14091) 
s_ashr_i32 s1, s0, 31 // (op 14091) 
v_and_b32 v18, 3, v13 // (op 14091) 

s_cmp_eq_i32 s10, 0 // (op 14091) 
s_mov_b32 s47, scc // (op 14091) 
v_and_b32 v13, 15, v1 // (op 14091) 
s_cmp_eq_i32 s47, 0 // (op 14091) 
s_cselect_b32 s18, 0, -1 // (op 14091) 

v_lshl_add_u32 v19, v18, 4, v13 // (op 14091) 


s_and_b32 s19, s13, 31 // (op 14091) 
v_lshrrev_b32 v13, 6, v1 // (op 14091) 
s_lshl_b32 s47, 1, s19 // (op 14091) 
v_and_b32 v18, 1, v13 // (op 14091) 

s_add_i32 s22, s47, s18 // (op 14091) 


v_lshl_add_u32 v13, v18, 6, v19 // (op 14091) 


s_and_b32 s18, s1, s22 // (op 14091) 


v_lshrrev_b32 v18, 6, v12 // (op 14091) 

s_add_i32 s1, s0, s18 // (op 14091) 


s_ashr_i32 s0, s1, s19 // (op 14091) 


s_lshl_b32 s1, s13, 24 // (op 14091) 
s_ashr_i32 s47, s1, 31 // (op 14091) 

s_xor_b32 s1, s0, s47 // (op 14091) 

s_sub_i32 s0, s1, s47 // (op 14091) 


s_mul_hi_i32 s1, s0, s14 // (op 14091) 
s_add_i32 s47, s1, s0 // (op 14091) 

s_ashr_i32 s1, s47, 31 // (op 14091) 
s_cmp_eq_i32 s14, 0 // (op 14091) 
s_mov_b32 s18, scc // (op 14091) 
s_cmp_eq_i32 s18, 0 // (op 14091) 
s_cselect_b32 s19, 0, -1 // (op 14091) 

s_and_b32 s18, s15, 31 // (op 14091) 
s_lshl_b32 s22, 1, s18 // (op 14091) 
s_add_i32 s23, s22, s19 // (op 14091) 


s_and_b32 s19, s1, s23 // (op 14091) 


s_add_i32 s1, s47, s19 // (op 14091) 


s_ashr_i32 s19, s1, s18 // (op 14091) 


s_lshl_b32 s1, s15, 24 // (op 14091) 
s_ashr_i32 s18, s1, 31 // (op 14091) 

s_xor_b32 s1, s19, s18 // (op 14091) 

s_sub_i32 s19, s1, s18 // (op 14091) 


s_add_u32 s22, s6, 128 // (op 14091) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 14091) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 14091) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 14091) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 14091) 

s_mul_i32 s1, s19, s22 // (op 14091) 

s_sub_i32 s47, s0, s1 // (op 14091) 


s_mul_hi_i32 s0, s16, s9 // (op 14091) 
s_add_i32 s1, s0, s16 // (op 14091) 

s_ashr_i32 s0, s1, 31 // (op 14091) 
s_cmp_eq_i32 s9, 0 // (op 14091) 
s_mov_b32 s18, scc // (op 14091) 
s_cmp_eq_i32 s18, 0 // (op 14091) 
s_cselect_b32 s19, 0, -1 // (op 14091) 

s_and_b32 s18, s45, 31 // (op 14091) 
s_lshl_b32 s24, 1, s18 // (op 14091) 
s_add_i32 s25, s24, s19 // (op 14091) 


s_and_b32 s19, s0, s25 // (op 14091) 


s_add_i32 s0, s1, s19 // (op 14091) 


s_ashr_i32 s1, s0, s18 // (op 14091) 


s_lshl_b32 s0, s45, 24 // (op 14091) 
s_ashr_i32 s18, s0, 31 // (op 14091) 

s_xor_b32 s0, s1, s18 // (op 14091) 

s_sub_i32 s1, s0, s18 // (op 14091) 


s_mul_i32 s0, s11, s22 // (op 14091) 

s_mul_i32 s18, s8, s0 // (op 14091) 

s_mul_i32 s0, s1, s18 // (op 14091) 

s_sub_u32 s18, s16, s0 // (op 14091) 

s_mul_hi_i32 s0, s18, s12 // (op 14091) 
s_add_i32 s19, s0, s18 // (op 14091) 


s_ashr_i32 s0, s19, 31 // (op 14091) 
s_cmp_eq_i32 s12, 0 // (op 14091) 
s_mov_b32 s18, scc // (op 14091) 
s_cmp_eq_i32 s18, 0 // (op 14091) 
s_cselect_b32 s22, 0, -1 // (op 14091) 

s_and_b32 s18, s44, 31 // (op 14091) 
s_lshl_b32 s23, 1, s18 // (op 14091) 
s_add_i32 s24, s23, s22 // (op 14091) 


s_and_b32 s25, s0, s24 // (op 14091) 


s_add_i32 s0, s19, s25 // (op 14091) 


s_ashr_i32 s19, s0, s18 // (op 14091) 


s_lshl_b32 s0, s44, 24 // (op 14091) 
s_ashr_i32 s18, s0, 31 // (op 14091) 

s_xor_b32 s0, s19, s18 // (op 14091) 

s_sub_i32 s19, s0, s18 // (op 14091) 


s_cmp_eq_i32 s1, 0 // (op 14091) 
s_mov_b32 s0, scc // (op 14091) 

s_cmp_eq_i32 s0, 0 // (op 14091) 
s_cselect_b32 s1, s19, s47 // (op 14091) 



v_lshl_add_u32 v12, s1, 7, v13 // (op 14091) 


v_mul_lo_u32 v13, v12, s38 // (op 14091) 

v_lshl_add_u32 v12, v18, 2, v13 // (op 14091) 


s_mul_i32 s1, s38, s7 // (op 14093) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 14093) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 14093) least significant
s_mul_hi_u32 s18, s38, s6 // (op 14093) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 14093) most significant: sum
s_add_u32 s1, s1, s18 // (op 14093) most significant: sum


s_mov_b32 s68, s40 // (op 14093) reg expression
s_mov_b32 s69, s41 // (op 14093) reg expression
s_mov_b32 s70, s0 // (op 14093) reg expression
s_mov_b32 s71, 131072 // (op 14093) call()

s_mul_hi_i32 s47, s16, s10 // (op 14112) 
v_and_b32 v13, 63, v1 // (op 14112) 
s_add_i32 s0, s47, s16 // (op 14112) 

v_lshrrev_b32 v18, 4, v13 // (op 14112) 
s_ashr_i32 s1, s0, 31 // (op 14112) 
v_and_b32 v19, 3, v18 // (op 14112) 

s_cmp_eq_i32 s10, 0 // (op 14112) 
s_mov_b32 s47, scc // (op 14112) 
v_and_b32 v18, 15, v1 // (op 14112) 
s_cmp_eq_i32 s47, 0 // (op 14112) 
s_cselect_b32 s18, 0, -1 // (op 14112) 

v_lshl_add_u32 v20, v19, 4, v18 // (op 14112) 


s_and_b32 s19, s13, 31 // (op 14112) 
v_lshrrev_b32 v18, 6, v1 // (op 14112) 
s_lshl_b32 s47, 1, s19 // (op 14112) 
v_and_b32 v19, 1, v18 // (op 14112) 

s_add_i32 s22, s47, s18 // (op 14112) 


v_lshl_add_u32 v18, v19, 6, v20 // (op 14112) 


s_and_b32 s18, s1, s22 // (op 14112) 


v_lshrrev_b32 v19, 6, v13 // (op 14112) 

s_add_i32 s1, s0, s18 // (op 14112) 


s_ashr_i32 s0, s1, s19 // (op 14112) 


s_lshl_b32 s1, s13, 24 // (op 14112) 
s_ashr_i32 s47, s1, 31 // (op 14112) 

s_xor_b32 s1, s0, s47 // (op 14112) 

s_sub_i32 s0, s1, s47 // (op 14112) 


s_mul_hi_i32 s1, s0, s14 // (op 14112) 
s_add_i32 s47, s1, s0 // (op 14112) 

s_ashr_i32 s1, s47, 31 // (op 14112) 
s_cmp_eq_i32 s14, 0 // (op 14112) 
s_mov_b32 s18, scc // (op 14112) 
s_cmp_eq_i32 s18, 0 // (op 14112) 
s_cselect_b32 s19, 0, -1 // (op 14112) 

s_and_b32 s18, s15, 31 // (op 14112) 
s_lshl_b32 s22, 1, s18 // (op 14112) 
s_add_i32 s23, s22, s19 // (op 14112) 


s_and_b32 s19, s1, s23 // (op 14112) 


s_add_i32 s1, s47, s19 // (op 14112) 


s_ashr_i32 s19, s1, s18 // (op 14112) 


s_lshl_b32 s1, s15, 24 // (op 14112) 
s_ashr_i32 s18, s1, 31 // (op 14112) 

s_xor_b32 s1, s19, s18 // (op 14112) 

s_sub_i32 s19, s1, s18 // (op 14112) 


s_add_u32 s22, s6, 128 // (op 14112) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 14112) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 14112) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 14112) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 14112) 

s_mul_i32 s1, s19, s22 // (op 14112) 

s_sub_i32 s47, s0, s1 // (op 14112) 


s_mul_hi_i32 s0, s16, s9 // (op 14112) 
s_add_i32 s1, s0, s16 // (op 14112) 

s_ashr_i32 s0, s1, 31 // (op 14112) 
s_cmp_eq_i32 s9, 0 // (op 14112) 
s_mov_b32 s18, scc // (op 14112) 
s_cmp_eq_i32 s18, 0 // (op 14112) 
s_cselect_b32 s19, 0, -1 // (op 14112) 

s_and_b32 s18, s45, 31 // (op 14112) 
s_lshl_b32 s24, 1, s18 // (op 14112) 
s_add_i32 s25, s24, s19 // (op 14112) 


s_and_b32 s19, s0, s25 // (op 14112) 


s_add_i32 s0, s1, s19 // (op 14112) 


s_ashr_i32 s1, s0, s18 // (op 14112) 


s_lshl_b32 s0, s45, 24 // (op 14112) 
s_ashr_i32 s18, s0, 31 // (op 14112) 

s_xor_b32 s0, s1, s18 // (op 14112) 

s_sub_i32 s1, s0, s18 // (op 14112) 


s_mul_i32 s0, s11, s22 // (op 14112) 

s_mul_i32 s18, s8, s0 // (op 14112) 

s_mul_i32 s0, s1, s18 // (op 14112) 

s_sub_u32 s18, s16, s0 // (op 14112) 

s_mul_hi_i32 s0, s18, s12 // (op 14112) 
s_add_i32 s19, s0, s18 // (op 14112) 


s_ashr_i32 s0, s19, 31 // (op 14112) 
s_cmp_eq_i32 s12, 0 // (op 14112) 
s_mov_b32 s18, scc // (op 14112) 
s_cmp_eq_i32 s18, 0 // (op 14112) 
s_cselect_b32 s22, 0, -1 // (op 14112) 

s_and_b32 s18, s44, 31 // (op 14112) 
s_lshl_b32 s23, 1, s18 // (op 14112) 
s_add_i32 s24, s23, s22 // (op 14112) 


s_and_b32 s25, s0, s24 // (op 14112) 


s_add_i32 s0, s19, s25 // (op 14112) 


s_ashr_i32 s19, s0, s18 // (op 14112) 


s_lshl_b32 s0, s44, 24 // (op 14112) 
s_ashr_i32 s18, s0, 31 // (op 14112) 

s_xor_b32 s0, s19, s18 // (op 14112) 

s_sub_i32 s19, s0, s18 // (op 14112) 


s_cmp_eq_i32 s1, 0 // (op 14112) 
s_mov_b32 s0, scc // (op 14112) 

s_cmp_eq_i32 s0, 0 // (op 14112) 
s_cselect_b32 s1, s19, s47 // (op 14112) 



v_lshl_add_u32 v13, s1, 7, v18 // (op 14112) 


v_mul_lo_u32 v18, v13, s38 // (op 14112) 

v_lshl_add_u32 v13, v19, 2, v18 // (op 14112) 


s_mul_i32 s1, s38, s7 // (op 14114) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 14114) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 14114) least significant
s_mul_hi_u32 s18, s38, s6 // (op 14114) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 14114) most significant: sum
s_add_u32 s1, s1, s18 // (op 14114) most significant: sum


s_mov_b32 s68, s40 // (op 14114) reg expression
s_mov_b32 s69, s41 // (op 14114) reg expression
s_mov_b32 s70, s0 // (op 14114) reg expression
s_mov_b32 s71, 131072 // (op 14114) call()




buffer_load_dwordx4 v[18:21], v14, s[60:63], 0 offen // (op 13060) Load value
buffer_load_dwordx2 v[22:23], v14, s[60:63], 0 offen offset:16 // (op 13060) Load value
buffer_load_dwordx4 v[24:27], v14, s[60:63], 0 offen offset:24 // (op 13060) Load value
buffer_load_dwordx2 v[28:29], v14, s[60:63], 0 offen offset:40 // (op 13060) Load value

s_mul_i32 s1, s27, 6 // (op 13063) most significant: low of high * low
s_mul_i32 s0, s26, 6 // (op 13063) least significant
s_mul_hi_u32 s47, s26, 6 // (op 13063) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13063) most significant: sum

v_lshrrev_b64 v[30:31], 3, s[0:1] // (op 13063) 


v_mov_b32 v32, v10 // (op 13063) 
buffer_load_dwordx4 v[34:37], v32, s[48:51], 0 offen // (op 13063) Load value
buffer_load_dwordx2 v[38:39], v32, s[48:51], 0 offen offset:16 // (op 13063) Load value
buffer_load_dwordx4 v[40:43], v32, s[48:51], 0 offen offset:24 // (op 13063) Load value
buffer_load_dwordx2 v[44:45], v32, s[48:51], 0 offen offset:40 // (op 13063) Load value
v_add_u32 v32, v32, v30 // (op 13063) 
buffer_load_dwordx4 v[46:49], v32, s[48:51], 0 offen // (op 13063) Load value
buffer_load_dwordx2 v[50:51], v32, s[48:51], 0 offen offset:16 // (op 13063) Load value
buffer_load_dwordx4 v[52:55], v32, s[48:51], 0 offen offset:24 // (op 13063) Load value
buffer_load_dwordx2 v[56:57], v32, s[48:51], 0 offen offset:40 // (op 13063) Load value




v_mov_b32 v30, v12 // (op 76) reg expression

buffer_load_dwordx2 v[32:33], v30, s[68:71], 0 offen // (op 76) Load value




s_lshl_b32 s47, s34, 6 // (op 2747) 
v_add_u32 v12, v16, s47 // (op 2747) 


buffer_load_dwordx2 v[30:31], v12, s[64:67], 0 offen // (op 2747) Load value



v_mov_b32 v12, v16 // (op 72) reg expression

buffer_load_dwordx2 v[58:59], v12, s[64:67], 0 offen // (op 72) Load value





buffer_load_dwordx4 v[60:63], v14, s[60:63], 0 offen offset:96 // (op 13066) Load value
buffer_load_dwordx2 v[64:65], v14, s[60:63], 0 offen offset:112 // (op 13066) Load value
buffer_load_dwordx4 v[66:69], v14, s[60:63], 0 offen offset:120 // (op 13066) Load value
buffer_load_dwordx2 v[70:71], v14, s[60:63], 0 offen offset:136 // (op 13066) Load value


s_mul_i32 s1, s27, 6 // (op 13069) most significant: low of high * low
s_mul_i32 s0, s26, 6 // (op 13069) least significant
s_mul_hi_u32 s47, s26, 6 // (op 13069) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13069) most significant: sum

v_lshrrev_b64 v[72:73], 3, s[0:1] // (op 13069) 


v_mov_b32 v12, v10 // (op 13069) 
buffer_load_dwordx4 v[74:77], v12, s[48:51], 0 offen offset:96 // (op 13069) Load value
buffer_load_dwordx2 v[78:79], v12, s[48:51], 0 offen offset:112 // (op 13069) Load value
buffer_load_dwordx4 v[80:83], v12, s[48:51], 0 offen offset:120 // (op 13069) Load value
buffer_load_dwordx2 v[84:85], v12, s[48:51], 0 offen offset:136 // (op 13069) Load value
v_add_u32 v12, v12, v72 // (op 13069) 
buffer_load_dwordx4 v[86:89], v12, s[48:51], 0 offen offset:96 // (op 13069) Load value
buffer_load_dwordx2 v[90:91], v12, s[48:51], 0 offen offset:112 // (op 13069) Load value
buffer_load_dwordx4 v[92:95], v12, s[48:51], 0 offen offset:120 // (op 13069) Load value
buffer_load_dwordx2 v[96:97], v12, s[48:51], 0 offen offset:136 // (op 13069) Load value



v_and_b32 v10, -2, v1 // (op 13919) 
v_sub_u32 v12, v1, v10 // (op 13919) 

v_lshlrev_b32 v10, 6, v12 // (op 13919) 

v_lshrrev_b32 v12, 1, v1 // (op 13919) 
v_lshl_add_u32 v14, v12, 7, v10 // (op 13919) 


v_mul_lo_u32 v10, v14, 6 // (op 13919) 

v_lshrrev_b32 v12, 3, v10 // (op 13919) 




s_waitcnt vmcnt(26)
ds_write_b128 v12, v[18:21] // (op 13071) Store local data 
s_waitcnt vmcnt(25)
ds_write_b64 v12, v[22:23] offset:16 // (op 13071) Store local data 
s_waitcnt vmcnt(24)
ds_write_b128 v12, v[24:27] offset:24 // (op 13071) Store local data 
s_waitcnt vmcnt(23)
ds_write_b64 v12, v[28:29] offset:40 // (op 13071) Store local data 

v_and_b32 v10, -2, v1 // (op 13861) 
v_sub_u32 v12, v1, v10 // (op 13861) 

v_lshlrev_b32 v10, 6, v12 // (op 13861) 

v_lshrrev_b32 v12, 1, v1 // (op 13861) 
v_lshl_add_u32 v14, v12, 8, v10 // (op 13861) 


v_mul_lo_u32 v10, v14, 6 // (op 13861) 

v_lshrrev_b32 v12, 3, v10 // (op 13861) 




s_waitcnt vmcnt(22)
ds_write_b128 v12, v[34:37] offset:12288 // (op 13072) Store local data 
s_waitcnt vmcnt(21)
ds_write_b64 v12, v[38:39] offset:12304 // (op 13072) Store local data 
s_waitcnt vmcnt(20)
ds_write_b128 v12, v[40:43] offset:12312 // (op 13072) Store local data 
s_waitcnt vmcnt(19)
ds_write_b64 v12, v[44:45] offset:12328 // (op 13072) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v12, v[46:49] offset:12384 // (op 13072) Store local data 
s_waitcnt vmcnt(17)
ds_write_b64 v12, v[50:51] offset:12400 // (op 13072) Store local data 
s_waitcnt vmcnt(16)
ds_write_b128 v12, v[52:55] offset:12408 // (op 13072) Store local data 
s_waitcnt vmcnt(15)
ds_write_b64 v12, v[56:57] offset:12424 // (op 13072) Store local data 

v_and_b32 v10, -2, v1 // (op 13832) 
v_sub_u32 v12, v1, v10 // (op 13832) 

v_lshlrev_b32 v10, 6, v12 // (op 13832) 

v_lshrrev_b32 v12, 1, v1 // (op 13832) 
v_lshl_add_u32 v14, v12, 8, v10 // (op 13832) 


v_mul_lo_u32 v10, v14, 6 // (op 13832) 

v_lshrrev_b32 v12, 3, v10 // (op 13832) 

v_and_b32 v10, -2, v1 // (op 13890) 
v_sub_u32 v14, v1, v10 // (op 13890) 

v_lshlrev_b32 v10, 6, v14 // (op 13890) 

v_lshrrev_b32 v14, 1, v1 // (op 13890) 
v_lshl_add_u32 v16, v14, 7, v10 // (op 13890) 


v_mul_lo_u32 v10, v16, 6 // (op 13890) 

v_lshrrev_b32 v14, 3, v10 // (op 13890) 

v_and_b32 v10, -2, v1 // (op 13971) 
v_sub_u32 v16, v1, v10 // (op 13971) 

v_lshlrev_b32 v10, 6, v16 // (op 13971) 

v_lshrrev_b32 v16, 1, v1 // (op 13971) 
v_lshl_add_u32 v72, v16, 8, v10 // (op 13971) 


v_mul_lo_u32 v10, v72, 6 // (op 13971) 

v_lshrrev_b32 v16, 3, v10 // (op 13971) 

v_and_b32 v10, -2, v1 // (op 14000) 
v_sub_u32 v72, v1, v10 // (op 14000) 

v_lshlrev_b32 v10, 6, v72 // (op 14000) 

v_lshrrev_b32 v72, 1, v1 // (op 14000) 
v_lshl_add_u32 v73, v72, 7, v10 // (op 14000) 


v_mul_lo_u32 v10, v73, 6 // (op 14000) 

v_lshrrev_b32 v72, 3, v10 // (op 14000) 

s_mov_b32 s47, 0 // (op 3300) call()
s_ashr_i64 s[0:1], s[42:43], 7 // (op 3299) 
s_mov_b32 s18, s0 // (op 3299) convert

s_and_b32 s19, s18, 4294967294 // (op 3299) 

s_cmp_lt_u32 s47, s19 // (op 3299) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ForLoopBottom_KLoop_3299 // (op 3299) Condition: Top (jump to RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ForLoopBottom_KLoop_3299 if false)
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ForLoopTop_KLoop_3299:
s_add_u32 s47, s47, 2 // (op 3301) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 14312) 



ds_read_b128 v[98:101], v3 offset:12288 // (op 62) Load local data 
ds_read_b64 v[102:103], v3 offset:12304 // (op 62) Load local data 
s_barrier  // (op 14325) 



ds_read_b128 v[104:107], v4 // (op 67) Load local data 
ds_read_b64 v[108:109], v4 offset:16 // (op 67) Load local data 



ds_read_b128 v[110:113], v4 offset:1536 // (op 312) Load local data 
ds_read_b64 v[114:115], v4 offset:1552 // (op 312) Load local data 
s_waitcnt vmcnt(14)
v_mov_b32 v116, v32 // (op 13575) reg expression



v_mov_b32 v117, v33 // (op 13575) reg expression
s_waitcnt vmcnt(12)
v_mov_b32 v118, v58 // (op 13602) reg expression
v_mov_b32 v119, v59 // (op 13602) reg expression
ds_read_b128 v[120:123], v4 offset:3072 // (op 448) Load local data 
ds_read_b64 v[124:125], v4 offset:3088 // (op 448) Load local data 
v_bfe_u32 v126, v118, 0, 8 // (op 12384) 



v_bfe_u32 v127, v118, 8, 8 // (op 12384) 
v_bfe_u32 v128, v118, 16, 8 // (op 12384) 
v_bfe_u32 v129, v118, 24, 8 // (op 12384) 
v_permlane16_swap_b32 v126, v127 // (op 12384) 
ds_read_b128 v[130:133], v4 offset:4608 // (op 584) Load local data 
v_permlane16_swap_b32 v128, v129 // (op 12384) 
s_nop 1
v_permlane32_swap_b32 v126, v128 // (op 12384) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v127, v129 // (op 12384) 
ds_read_b64 v[134:135], v4 offset:4624 // (op 584) Load local data 



v_bfe_u32 v136, v116, 0, 8 // (op 12511) 
v_bfe_u32 v137, v116, 8, 8 // (op 12511) 
v_bfe_u32 v138, v116, 16, 8 // (op 12511) 
v_bfe_u32 v139, v116, 24, 8 // (op 12511) 
v_permlane16_swap_b32 v136, v137 // (op 12511) 
ds_read_b128 v[140:143], v3 offset:13824 // (op 889) Load local data 
ds_read_b64 v[144:145], v3 offset:13840 // (op 889) Load local data 
v_permlane16_swap_b32 v138, v139 // (op 12511) 



s_nop 1
v_permlane32_swap_b32 v136, v138 // (op 12511) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v137, v139 // (op 12511) 
ds_read_b128 v[146:149], v3 offset:15360 // (op 1499) Load local data 
ds_read_b64 v[150:151], v3 offset:15376 // (op 1499) Load local data 



s_waitcnt lgkmcnt(10)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[98:103], v[104:109], a[0:3], v126, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 78) 
ds_read_b128 v[152:155], v3 offset:16896 // (op 2109) Load local data 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[98:103], v[110:115], a[4:7], v126, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 315) 
ds_read_b64 v[156:157], v3 offset:16912 // (op 2109) Load local data 



s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[98:103], v[120:125], a[8:11], v126, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 451) 
ds_read_b128 v[158:161], v3 offset:18432 // (op 2719) Load local data 
s_waitcnt lgkmcnt(7)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[98:103], v[130:135], a[12:15], v126, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 587) 
ds_read_b64 v[162:163], v3 offset:18448 // (op 2719) Load local data 






v_mov_b32 v10, v15 // (op 10) 

s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[140:145], v[110:115], a[20:23], v127, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 969) 
buffer_load_dwordx4 v[18:21], v10, s[60:63], 0 offen offset:192 // (op 10) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[140:145], v[120:125], a[24:27], v127, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 931) 
buffer_load_dwordx2 v[22:23], v10, s[60:63], 0 offen offset:208 // (op 10) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[140:145], v[130:135], a[28:31], v127, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 893) 
buffer_load_dwordx4 v[24:27], v10, s[60:63], 0 offen offset:216 // (op 10) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[140:145], v[104:109], a[16:19], v127, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1007) 
v_mov_b32 v98, v30 // (op 13588) reg expression

v_mov_b32 v99, v31 // (op 13588) reg expression
v_bfe_u32 v100, v98, 0, 8 // (op 12415) 
v_bfe_u32 v101, v98, 8, 8 // (op 12415) 
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[146:151], v[104:109], a[32:35], v128, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1617) 
v_bfe_u32 v102, v98, 16, 8 // (op 12415) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[146:151], v[110:115], a[36:39], v128, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1579) 
v_bfe_u32 v103, v98, 24, 8 // (op 12415) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[146:151], v[120:125], a[40:43], v128, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1541) 
v_permlane16_swap_b32 v100, v101 // (op 12415) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[146:151], v[130:135], a[44:47], v128, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1503) 
v_permlane16_swap_b32 v102, v103 // (op 12415) 

s_nop 1
v_permlane32_swap_b32 v100, v102 // (op 12415) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v101, v103 // (op 12415) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[152:157], v[104:109], a[48:51], v129, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2227) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[152:157], v[110:115], a[52:55], v129, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2189) 
ds_read_b128 v[140:143], v3 offset:19968 // (op 3329) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[152:157], v[120:125], a[56:59], v129, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2151) 
ds_read_b64 v[144:145], v3 offset:19984 // (op 3329) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[152:157], v[130:135], a[60:63], v129, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2113) 
ds_read_b128 v[146:149], v3 offset:21504 // (op 3939) Load local data 


ds_read_b64 v[150:151], v3 offset:21520 // (op 3939) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[158:163], v[110:115], a[68:71], v100, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2799) 
ds_read_b128 v[152:155], v3 offset:23040 // (op 4549) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[158:163], v[120:125], a[72:75], v100, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2761) 
ds_read_b64 v[156:157], v3 offset:23056 // (op 4549) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[158:163], v[130:135], a[76:79], v100, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2723) 
buffer_load_dwordx2 v[28:29], v10, s[60:63], 0 offen offset:232 // (op 10) Load value

v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[158:163], v[104:109], a[64:67], v100, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2837) 
s_mul_i32 s1, s27, 6 // (op 4) most significant: low of high * low

s_mul_i32 s0, s26, 6 // (op 4) least significant
s_mul_hi_u32 s18, s26, 6 // (op 4) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 4) most significant: sum

s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[140:145], v[110:115], a[84:87], v101, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 3409) 
v_lshrrev_b64 v[126:127], 3, s[0:1] // (op 4) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[140:145], v[120:125], a[88:91], v101, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 3371) 
v_mov_b32 v73, v11 // (op 4) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[140:145], v[130:135], a[92:95], v101, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 3333) 
v_mov_b32 v128, v73 // (op 4) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[140:145], v[104:109], a[80:83], v101, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 3447) 
buffer_load_dwordx4 v[34:37], v128, s[48:51], 0 offen offset:192 // (op 4) Load value

buffer_load_dwordx2 v[38:39], v128, s[48:51], 0 offen offset:208 // (op 4) Load value
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[146:151], v[110:115], a[100:103], v102, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 4019) 
buffer_load_dwordx4 v[40:43], v128, s[48:51], 0 offen offset:216 // (op 4) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[146:151], v[120:125], a[104:107], v102, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 3981) 
buffer_load_dwordx2 v[44:45], v128, s[48:51], 0 offen offset:232 // (op 4) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[146:151], v[130:135], a[108:111], v102, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 3943) 
v_add_u32 v128, v128, v126 // (op 4) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[146:151], v[104:109], a[96:99], v102, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 4057) 
buffer_load_dwordx4 v[46:49], v128, s[48:51], 0 offen offset:192 // (op 4) Load value

buffer_load_dwordx2 v[50:51], v128, s[48:51], 0 offen offset:208 // (op 4) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[152:157], v[110:115], a[116:119], v103, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 4629) 
buffer_load_dwordx4 v[52:55], v128, s[48:51], 0 offen offset:216 // (op 4) Load value

buffer_load_dwordx2 v[56:57], v128, s[48:51], 0 offen offset:232 // (op 4) Load value




v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[152:157], v[120:125], a[120:123], v103, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 4591) 
s_barrier  // (op 14348) 




v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[152:157], v[130:135], a[124:127], v103, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 4553) 
v_mov_b32 v110, v13 // (op 13583) 

v_add_u32 v111, 8, v110 // (op 13583) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[152:157], v[104:109], a[112:115], v103, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 4667) 
s_waitcnt vmcnt(23)
ds_write_b128 v72, v[60:63] offset:36864 // (op 222) Store local data 




s_waitcnt vmcnt(22)
ds_write_b64 v72, v[64:65] offset:36880 // (op 222) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v72, v[66:69] offset:36888 // (op 222) Store local data 
s_waitcnt vmcnt(20)
ds_write_b64 v72, v[70:71] offset:36904 // (op 222) Store local data 
buffer_load_dwordx2 v[32:33], v111, s[68:71], 0 offen // (op 13583) Load value


v_add_u32 v13, 8, v13 // (op 14102) 


s_barrier  // (op 14342) 
v_mov_b32 v100, v17 // (op 13597) 



s_lshl_b32 s0, s34, 6 // (op 13597) 
s_add_u32 s1, 8, s0 // (op 13597) 

v_add_u32 v101, v100, s1 // (op 13597) 


buffer_load_dwordx2 v[30:31], v101, s[64:67], 0 offen // (op 13597) Load value

s_waitcnt vmcnt(21)
ds_write_b128 v16, v[74:77] offset:49152 // (op 212) Store local data 


v_add_u32 v101, 8, v100 // (op 13610) 

buffer_load_dwordx2 v[58:59], v101, s[64:67], 0 offen // (op 13610) Load value

s_waitcnt vmcnt(21)
ds_write_b64 v16, v[78:79] offset:49168 // (op 212) Store local data 

s_waitcnt vmcnt(20)
ds_write_b128 v16, v[80:83] offset:49176 // (op 212) Store local data 
v_add_u32 v17, 8, v17 // (op 14050) 
s_waitcnt vmcnt(19)
ds_write_b64 v16, v[84:85] offset:49192 // (op 212) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v16, v[86:89] offset:49248 // (op 212) Store local data 
s_waitcnt vmcnt(17)
ds_write_b64 v16, v[90:91] offset:49264 // (op 212) Store local data 
s_waitcnt vmcnt(16)
ds_write_b128 v16, v[92:95] offset:49272 // (op 212) Store local data 
s_waitcnt vmcnt(15)
ds_write_b64 v16, v[96:97] offset:49288 // (op 212) Store local data 



s_waitcnt lgkmcnt(0)
s_barrier  // (op 14339) 
buffer_load_dwordx4 v[60:63], v10, s[60:63], 0 offen offset:288 // (op 223) Load value



ds_read_b128 v[100:103], v5 offset:49152 // (op 214) Load local data 
ds_read_b64 v[104:105], v5 offset:49168 // (op 214) Load local data 
buffer_load_dwordx2 v[64:65], v10, s[60:63], 0 offen offset:304 // (op 223) Load value
s_barrier  // (op 14345) 
buffer_load_dwordx4 v[66:69], v10, s[60:63], 0 offen offset:312 // (op 223) Load value



ds_read_b128 v[106:109], v6 offset:36864 // (op 216) Load local data 
ds_read_b64 v[110:111], v6 offset:36880 // (op 216) Load local data 
buffer_load_dwordx2 v[70:71], v10, s[60:63], 0 offen offset:328 // (op 223) Load value




v_bfe_u32 v112, v119, 0, 8 // (op 12399) 

v_bfe_u32 v113, v119, 8, 8 // (op 12399) 
v_bfe_u32 v114, v119, 16, 8 // (op 12399) 
v_bfe_u32 v115, v119, 24, 8 // (op 12399) 
v_permlane16_swap_b32 v112, v113 // (op 12399) 
v_add_u32 v15, 192, v15 // (op 13781) 
v_permlane16_swap_b32 v114, v115 // (op 12399) 
s_mul_i32 s1, s27, 6 // (op 211) most significant: low of high * low
s_mul_i32 s0, s26, 6 // (op 211) least significant
v_permlane32_swap_b32 v112, v114 // (op 12399) 
v_permlane32_swap_b32 v113, v115 // (op 12399) 
s_mul_hi_u32 s18, s26, 6 // (op 211) most significant: high of low * low

s_add_u32 s1, s1, s18 // (op 211) most significant: sum

v_lshrrev_b64 v[118:119], 3, s[0:1] // (op 211) 


v_mov_b32 v10, v73 // (op 211) 
v_bfe_u32 v120, v117, 0, 8 // (op 12526) 
v_bfe_u32 v121, v117, 8, 8 // (op 12526) 
v_bfe_u32 v122, v117, 16, 8 // (op 12526) 
v_bfe_u32 v123, v117, 24, 8 // (op 12526) 
v_permlane16_swap_b32 v120, v121 // (op 12526) 
ds_read_b128 v[124:127], v6 offset:38400 // (op 324) Load local data 
ds_read_b64 v[128:129], v6 offset:38416 // (op 324) Load local data 
buffer_load_dwordx4 v[74:77], v10, s[48:51], 0 offen offset:288 // (op 211) Load value



v_permlane16_swap_b32 v122, v123 // (op 12526) 
ds_read_b128 v[130:133], v6 offset:39936 // (op 460) Load local data 
s_nop 0
v_permlane32_swap_b32 v120, v122 // (op 12526) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v121, v123 // (op 12526) 
buffer_load_dwordx2 v[78:79], v10, s[48:51], 0 offen offset:304 // (op 211) Load value

ds_read_b64 v[134:135], v6 offset:39952 // (op 460) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[100:105], v[106:111], a[0:3], v112, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 221) 
ds_read_b128 v[136:139], v6 offset:41472 // (op 596) Load local data 
ds_read_b64 v[140:141], v6 offset:41488 // (op 596) Load local data 
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[100:105], v[124:129], a[4:7], v112, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 316) 



s_barrier  // (op 14397) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[100:105], v[130:135], a[8:11], v112, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 452) 
ds_read_b128 v[142:145], v5 offset:50688 // (op 901) Load local data 
ds_read_b64 v[146:147], v5 offset:50704 // (op 901) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[100:105], v[136:141], a[12:15], v112, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 588) 




ds_read_b128 v[100:103], v5 offset:52224 // (op 1511) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[142:147], v[106:111], a[16:19], v113, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1008) 
ds_read_b64 v[104:105], v5 offset:52240 // (op 1511) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[142:147], v[130:135], a[24:27], v113, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 932) 
ds_read_b128 v[148:151], v5 offset:53760 // (op 2121) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[142:147], v[136:141], a[28:31], v113, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 894) 
ds_read_b64 v[152:153], v5 offset:53776 // (op 2121) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[142:147], v[124:129], a[20:23], v113, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 970) 
ds_read_b128 v[154:157], v5 offset:55296 // (op 2731) Load local data 

ds_read_b64 v[158:159], v5 offset:55312 // (op 2731) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[100:105], v[106:111], a[32:35], v114, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1618) 
v_bfe_u32 v142, v99, 0, 8 // (op 12430) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[100:105], v[130:135], a[40:43], v114, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1542) 
v_bfe_u32 v143, v99, 8, 8 // (op 12430) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[100:105], v[136:141], a[44:47], v114, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1504) 
v_bfe_u32 v144, v99, 16, 8 // (op 12430) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[100:105], v[124:129], a[36:39], v114, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1580) 
v_bfe_u32 v145, v99, 24, 8 // (op 12430) 

v_permlane16_swap_b32 v142, v143 // (op 12430) 
ds_read_b128 v[100:103], v5 offset:56832 // (op 3341) Load local data 
v_permlane16_swap_b32 v144, v145 // (op 12430) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[148:153], v[106:111], a[48:51], v115, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2228) 
s_nop 0
v_permlane32_swap_b32 v142, v144 // (op 12430) Wait state hazard: VALU Write Readlane Hazard
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[148:153], v[130:135], a[56:59], v115, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2152) 
v_permlane32_swap_b32 v143, v145 // (op 12430) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[148:153], v[136:141], a[60:63], v115, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2114) 
ds_read_b64 v[104:105], v5 offset:56848 // (op 3341) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[148:153], v[124:129], a[52:55], v115, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2190) 
ds_read_b128 v[160:163], v5 offset:58368 // (op 3951) Load local data 


ds_read_b64 v[164:165], v5 offset:58384 // (op 3951) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[154:159], v[124:129], a[68:71], v142, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2800) 
ds_read_b128 v[112:115], v5 offset:59904 // (op 4561) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[154:159], v[106:111], a[64:67], v142, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2838) 
ds_read_b64 v[116:117], v5 offset:59920 // (op 4561) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[154:159], v[130:135], a[72:75], v142, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2762) 
s_barrier  // (op 14459) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[154:159], v[136:141], a[76:79], v142, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2724) 
buffer_load_dwordx4 v[80:83], v10, s[48:51], 0 offen offset:312 // (op 211) Load value

buffer_load_dwordx2 v[84:85], v10, s[48:51], 0 offen offset:328 // (op 211) Load value
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[100:105], v[124:129], a[84:87], v143, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 3410) 
v_add_u32 v10, v10, v118 // (op 211) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[100:105], v[106:111], a[80:83], v143, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 3448) 
buffer_load_dwordx4 v[86:89], v10, s[48:51], 0 offen offset:288 // (op 211) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[100:105], v[130:135], a[88:91], v143, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 3372) 
buffer_load_dwordx2 v[90:91], v10, s[48:51], 0 offen offset:304 // (op 211) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[100:105], v[136:141], a[92:95], v143, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 3334) 
buffer_load_dwordx4 v[92:95], v10, s[48:51], 0 offen offset:312 // (op 211) Load value

buffer_load_dwordx2 v[96:97], v10, s[48:51], 0 offen offset:328 // (op 211) Load value



s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[160:165], v[124:129], a[100:103], v144, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 4020) 
s_barrier  // (op 14328) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[160:165], v[106:111], a[96:99], v144, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 4058) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[160:165], v[130:135], a[104:107], v144, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 3982) 
v_add_u32 v11, 192, v11 // (op 13717) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[160:165], v[136:141], a[108:111], v144, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 3944) 
s_waitcnt vmcnt(26)
ds_write_b128 v14, v[18:21] // (op 13) Store local data 

s_waitcnt vmcnt(25)
ds_write_b64 v14, v[22:23] offset:16 // (op 13) Store local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[112:117], v[124:129], a[116:119], v145, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 4630) 
s_waitcnt vmcnt(24)
ds_write_b128 v14, v[24:27] offset:24 // (op 13) Store local data 

s_waitcnt vmcnt(23)
ds_write_b64 v14, v[28:29] offset:40 // (op 13) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[112:117], v[106:111], a[112:115], v145, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 4668) 

s_barrier  // (op 14315) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[112:117], v[130:135], a[120:123], v145, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 4592) 
s_waitcnt vmcnt(22)
ds_write_b128 v12, v[34:37] offset:12288 // (op 6) Store local data 

s_waitcnt vmcnt(21)
ds_write_b64 v12, v[38:39] offset:12304 // (op 6) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[112:117], v[136:141], a[124:127], v145, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 4554) 
s_waitcnt vmcnt(20)
ds_write_b128 v12, v[40:43] offset:12312 // (op 6) Store local data 




s_waitcnt vmcnt(19)
ds_write_b64 v12, v[44:45] offset:12328 // (op 6) Store local data 
s_waitcnt vmcnt(18)
ds_write_b128 v12, v[46:49] offset:12384 // (op 6) Store local data 
s_waitcnt vmcnt(17)
ds_write_b64 v12, v[50:51] offset:12400 // (op 6) Store local data 
s_waitcnt vmcnt(16)
ds_write_b128 v12, v[52:55] offset:12408 // (op 6) Store local data 
s_waitcnt vmcnt(15)
ds_write_b64 v12, v[56:57] offset:12424 // (op 6) Store local data 
s_ashr_i64 s[0:1], s[42:43], 7 // (op 3299) 
s_mov_b32 s18, s0 // (op 3299) convert

s_and_b32 s19, s18, 4294967294 // (op 3299) 

s_cmp_lt_u32 s47, s19 // (op 3299) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ForLoopTop_KLoop_3299 // (op 3299) Condition: Bottom (jump to RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ForLoopTop_KLoop_3299 if true)
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ForLoopBottom_KLoop_3299:













s_mul_hi_i32 s47, s16, s10 // (op 13649) 
v_and_b32 v10, 4294967294, v1 // (op 13649) 
s_add_i32 s0, s47, s16 // (op 13649) 

v_and_b32 v11, -2, v1 // (op 13649) 
s_ashr_i32 s1, s0, 31 // (op 13649) 
v_sub_u32 v12, v1, v11 // (op 13649) 

s_cmp_eq_i32 s10, 0 // (op 13649) 
s_mov_b32 s47, scc // (op 13649) 
v_lshlrev_b32 v11, 6, v12 // (op 13649) 

s_cmp_eq_i32 s47, 0 // (op 13649) 
s_cselect_b32 s18, 0, -1 // (op 13649) 

s_and_b32 s19, s13, 31 // (op 13649) 
s_lshl_b32 s47, 1, s19 // (op 13649) 
s_add_i32 s22, s47, s18 // (op 13649) 


s_and_b32 s18, s1, s22 // (op 13649) 


s_add_i32 s1, s0, s18 // (op 13649) 


s_ashr_i32 s0, s1, s19 // (op 13649) 


s_lshl_b32 s1, s13, 24 // (op 13649) 
s_ashr_i32 s47, s1, 31 // (op 13649) 

s_xor_b32 s1, s0, s47 // (op 13649) 

s_sub_i32 s0, s1, s47 // (op 13649) 


s_mul_hi_i32 s1, s0, s14 // (op 13649) 
s_add_i32 s47, s1, s0 // (op 13649) 

s_ashr_i32 s1, s47, 31 // (op 13649) 
s_cmp_eq_i32 s14, 0 // (op 13649) 
s_mov_b32 s18, scc // (op 13649) 
s_cmp_eq_i32 s18, 0 // (op 13649) 
s_cselect_b32 s19, 0, -1 // (op 13649) 

s_and_b32 s18, s15, 31 // (op 13649) 
s_lshl_b32 s22, 1, s18 // (op 13649) 
s_add_i32 s23, s22, s19 // (op 13649) 


s_and_b32 s19, s1, s23 // (op 13649) 


s_add_i32 s1, s47, s19 // (op 13649) 


s_ashr_i32 s19, s1, s18 // (op 13649) 


s_lshl_b32 s1, s15, 24 // (op 13649) 
s_ashr_i32 s18, s1, 31 // (op 13649) 

s_xor_b32 s1, s19, s18 // (op 13649) 

s_sub_i32 s19, s1, s18 // (op 13649) 


s_mul_i32 s1, s0, s11 // (op 13649) 

s_sub_u32 s0, s16, s1 // (op 13649) 

s_mul_i32 s1, s19, s11 // (op 13649) 
s_add_u32 s1, s1, s0 // (op 13649) 


s_mul_hi_i32 s0, s16, s9 // (op 13649) 
s_add_i32 s47, s0, s16 // (op 13649) 

s_ashr_i32 s0, s47, 31 // (op 13649) 
s_cmp_eq_i32 s9, 0 // (op 13649) 
s_mov_b32 s18, scc // (op 13649) 
s_cmp_eq_i32 s18, 0 // (op 13649) 
s_cselect_b32 s19, 0, -1 // (op 13649) 

s_and_b32 s18, s45, 31 // (op 13649) 
s_lshl_b32 s22, 1, s18 // (op 13649) 
s_add_i32 s23, s22, s19 // (op 13649) 


s_and_b32 s19, s0, s23 // (op 13649) 


s_add_i32 s0, s47, s19 // (op 13649) 


s_ashr_i32 s19, s0, s18 // (op 13649) 


s_lshl_b32 s0, s45, 24 // (op 13649) 
s_ashr_i32 s18, s0, 31 // (op 13649) 

s_xor_b32 s0, s19, s18 // (op 13649) 

s_sub_i32 s19, s0, s18 // (op 13649) 


s_add_u32 s22, s6, 128 // (op 13649) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13649) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13649) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13649) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 13649) 

s_mul_i32 s0, s11, s22 // (op 13649) 

s_mul_i32 s18, s8, s0 // (op 13649) 

s_mul_i32 s0, s19, s18 // (op 13649) 

s_sub_u32 s18, s16, s0 // (op 13649) 

s_mov_b32 s0, s18 // (op 13649) 
s_mul_hi_i32 s47, s0, s12 // (op 13649) 
s_add_i32 s22, s47, s0 // (op 13649) 


s_ashr_i32 s0, s22, 31 // (op 13649) 
s_cmp_eq_i32 s12, 0 // (op 13649) 
s_mov_b32 s47, scc // (op 13649) 
s_cmp_eq_i32 s47, 0 // (op 13649) 
s_cselect_b32 s23, 0, -1 // (op 13649) 

s_and_b32 s47, s44, 31 // (op 13649) 
s_lshl_b32 s24, 1, s47 // (op 13649) 
s_add_i32 s25, s24, s23 // (op 13649) 


s_and_b32 s23, s0, s25 // (op 13649) 


s_add_i32 s0, s22, s23 // (op 13649) 


s_ashr_i32 s22, s0, s47 // (op 13649) 


s_lshl_b32 s0, s44, 24 // (op 13649) 
s_ashr_i32 s47, s0, 31 // (op 13649) 

s_xor_b32 s0, s22, s47 // (op 13649) 

s_sub_i32 s22, s0, s47 // (op 13649) 


s_mul_i32 s0, s46, s11 // (op 13649) 
s_add_u32 s24, s20, 256 // (op 13649) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 13649) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 13649) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 13649) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 8 // (op 13649) 

s_sub_i32 s23, s24, s0 // (op 13649) 

s_mul_i32 s47, s22, s23 // (op 13649) 


s_sub_u32 s22, s18, s47 // (op 13649) 


s_add_u32 s18, s22, s0 // (op 13649) 


s_cmp_eq_i32 s19, 0 // (op 13649) 
s_mov_b32 s0, scc // (op 13649) 

s_cmp_eq_i32 s0, 0 // (op 13649) 
s_cselect_b32 s19, s18, s1 // (op 13649) 



v_lshl_add_u32 v12, s19, 8, v10 // (op 13649) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 13649) 
s_lshr_b32 s47, s0, 1 // (op 13649) 

v_lshl_add_u32 v10, s47, 8, v11 // (op 13649) 


v_mov_b32 v73, 0 // (op 13649) convert
v_mov_b32 v72, v10 // (op 13649) convert

v_mul_lo_u32 v11, v12, s27 // (op 13649) most significant: low of low * high
v_mul_hi_u32 v13, v12, s26 // (op 13649) most significant: high of low * low
v_mul_lo_u32 v10, v12, s26 // (op 13649) least significant: low of low * low
v_add_u32 v11, v11, v13 // (op 13649) most significant: sum

v_add_co_u32 v10, s[0:1], v10, v72 // (op 13649) least significant half
v_addc_co_u32 v11, s[0:1], v11, v73, s[0:1] // (op 13649) most significant half



v_mul_lo_u32 v73, v11, 6 // (op 13649) most significant: low of high * low
v_mul_hi_u32 v12, v10, 6 // (op 13649) most significant: high of low * low
v_mul_lo_u32 v72, v10, 6 // (op 13649) least significant: low of low * low
v_add_u32 v73, v73, v12 // (op 13649) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[72:73] // (op 13649) 

v_mov_b32 v12, v10 // (op 13649) convert

s_mul_i32 s1, s21, 6 // (op 13651) most significant: low of high * low
s_mul_i32 s0, s20, 6 // (op 13651) least significant
s_mul_hi_u32 s47, s20, 6 // (op 13651) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13651) most significant: sum

s_mul_i32 s19, s26, s1 // (op 13651) most significant: low of low * high
s_mul_i32 s47, s27, s0 // (op 13651) most significant: low of high * low
s_mul_i32 s18, s26, s0 // (op 13651) least significant
s_mul_hi_u32 s22, s26, s0 // (op 13651) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 13651) most significant: sum
s_add_u32 s19, s19, s22 // (op 13651) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 13651) 

s_mov_b32 s48, s28 // (op 13651) reg expression
s_mov_b32 s49, s29 // (op 13651) reg expression
s_mov_b32 s50, s0 // (op 13651) reg expression
s_mov_b32 s51, 131072 // (op 13651) call()

s_mul_hi_i32 s47, s16, s10 // (op 13747) 
v_lshrrev_b32 v10, 1, v1 // (op 13747) 
s_add_i32 s0, s47, s16 // (op 13747) 

v_and_b32 v11, -2, v1 // (op 13747) 
s_ashr_i32 s1, s0, 31 // (op 13747) 
v_sub_u32 v13, v1, v11 // (op 13747) 

s_cmp_eq_i32 s10, 0 // (op 13747) 
s_mov_b32 s47, scc // (op 13747) 
v_lshlrev_b32 v11, 6, v13 // (op 13747) 

s_cmp_eq_i32 s47, 0 // (op 13747) 
s_cselect_b32 s18, 0, -1 // (op 13747) 

s_and_b32 s19, s13, 31 // (op 13747) 
s_lshl_b32 s47, 1, s19 // (op 13747) 
s_add_i32 s22, s47, s18 // (op 13747) 


s_and_b32 s18, s1, s22 // (op 13747) 


s_add_i32 s1, s0, s18 // (op 13747) 


s_ashr_i32 s0, s1, s19 // (op 13747) 


s_lshl_b32 s1, s13, 24 // (op 13747) 
s_ashr_i32 s47, s1, 31 // (op 13747) 

s_xor_b32 s1, s0, s47 // (op 13747) 

s_sub_i32 s0, s1, s47 // (op 13747) 


s_mul_hi_i32 s1, s0, s14 // (op 13747) 
s_add_i32 s47, s1, s0 // (op 13747) 

s_ashr_i32 s1, s47, 31 // (op 13747) 
s_cmp_eq_i32 s14, 0 // (op 13747) 
s_mov_b32 s18, scc // (op 13747) 
s_cmp_eq_i32 s18, 0 // (op 13747) 
s_cselect_b32 s19, 0, -1 // (op 13747) 

s_and_b32 s18, s15, 31 // (op 13747) 
s_lshl_b32 s22, 1, s18 // (op 13747) 
s_add_i32 s23, s22, s19 // (op 13747) 


s_and_b32 s19, s1, s23 // (op 13747) 


s_add_i32 s1, s47, s19 // (op 13747) 


s_ashr_i32 s19, s1, s18 // (op 13747) 


s_lshl_b32 s1, s15, 24 // (op 13747) 
s_ashr_i32 s18, s1, 31 // (op 13747) 

s_xor_b32 s1, s19, s18 // (op 13747) 

s_sub_i32 s19, s1, s18 // (op 13747) 


s_add_u32 s22, s6, 128 // (op 13747) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13747) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13747) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13747) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 13747) 

s_mul_i32 s1, s19, s22 // (op 13747) 

s_sub_i32 s47, s0, s1 // (op 13747) 


s_mul_hi_i32 s0, s16, s9 // (op 13747) 
s_add_i32 s1, s0, s16 // (op 13747) 

s_ashr_i32 s0, s1, 31 // (op 13747) 
s_cmp_eq_i32 s9, 0 // (op 13747) 
s_mov_b32 s18, scc // (op 13747) 
s_cmp_eq_i32 s18, 0 // (op 13747) 
s_cselect_b32 s19, 0, -1 // (op 13747) 

s_and_b32 s18, s45, 31 // (op 13747) 
s_lshl_b32 s24, 1, s18 // (op 13747) 
s_add_i32 s25, s24, s19 // (op 13747) 


s_and_b32 s19, s0, s25 // (op 13747) 


s_add_i32 s0, s1, s19 // (op 13747) 


s_ashr_i32 s1, s0, s18 // (op 13747) 


s_lshl_b32 s0, s45, 24 // (op 13747) 
s_ashr_i32 s18, s0, 31 // (op 13747) 

s_xor_b32 s0, s1, s18 // (op 13747) 

s_sub_i32 s1, s0, s18 // (op 13747) 


s_mul_i32 s0, s11, s22 // (op 13747) 

s_mul_i32 s18, s8, s0 // (op 13747) 

s_mul_i32 s0, s1, s18 // (op 13747) 

s_sub_u32 s18, s16, s0 // (op 13747) 

s_mul_hi_i32 s0, s18, s12 // (op 13747) 
s_add_i32 s19, s0, s18 // (op 13747) 


s_ashr_i32 s0, s19, 31 // (op 13747) 
s_cmp_eq_i32 s12, 0 // (op 13747) 
s_mov_b32 s18, scc // (op 13747) 
s_cmp_eq_i32 s18, 0 // (op 13747) 
s_cselect_b32 s22, 0, -1 // (op 13747) 

s_and_b32 s18, s44, 31 // (op 13747) 
s_lshl_b32 s23, 1, s18 // (op 13747) 
s_add_i32 s24, s23, s22 // (op 13747) 


s_and_b32 s25, s0, s24 // (op 13747) 


s_add_i32 s0, s19, s25 // (op 13747) 


s_ashr_i32 s19, s0, s18 // (op 13747) 


s_lshl_b32 s0, s44, 24 // (op 13747) 
s_ashr_i32 s18, s0, 31 // (op 13747) 

s_xor_b32 s0, s19, s18 // (op 13747) 

s_sub_i32 s19, s0, s18 // (op 13747) 


s_cmp_eq_i32 s1, 0 // (op 13747) 
s_mov_b32 s0, scc // (op 13747) 

s_cmp_eq_i32 s0, 0 // (op 13747) 
s_cselect_b32 s1, s19, s47 // (op 13747) 



v_lshl_add_u32 v13, s1, 7, v10 // (op 13747) 


v_mul_lo_u32 v73, v13, s31 // (op 13747) most significant: low of low * high
v_mul_hi_u32 v10, v13, s30 // (op 13747) most significant: high of low * low
v_mul_lo_u32 v72, v13, s30 // (op 13747) least significant: low of low * low
v_add_u32 v73, v73, v10 // (op 13747) most significant: sum


s_ashr_i64 s[0:1], s[42:43], 7 // (op 13747) 
s_lshr_b32 s47, s0, 1 // (op 13747) 

v_lshl_add_u32 v10, s47, 8, v11 // (op 13747) 


v_mov_b32 v29, 0 // (op 13747) convert
v_mov_b32 v28, v10 // (op 13747) convert

v_add_co_u32 v10, s[0:1], v28, v72 // (op 13747) least significant half
v_addc_co_u32 v11, s[0:1], v29, v73, s[0:1] // (op 13747) most significant half



v_mul_lo_u32 v73, v11, 6 // (op 13747) most significant: low of high * low
v_mul_hi_u32 v13, v10, 6 // (op 13747) most significant: high of low * low
v_mul_lo_u32 v72, v10, 6 // (op 13747) least significant: low of low * low
v_add_u32 v73, v73, v13 // (op 13747) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[72:73] // (op 13747) 

v_mov_b32 v13, v10 // (op 13747) convert

s_mul_i32 s1, s7, 6 // (op 13749) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 13749) least significant
s_mul_hi_u32 s47, s6, 6 // (op 13749) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13749) most significant: sum

s_mul_i32 s19, s30, s1 // (op 13749) most significant: low of low * high
s_mul_i32 s47, s31, s0 // (op 13749) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 13749) least significant
s_mul_hi_u32 s22, s30, s0 // (op 13749) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 13749) most significant: sum
s_add_u32 s19, s19, s22 // (op 13749) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 13749) 

s_mov_b32 s60, s32 // (op 13749) reg expression
s_mov_b32 s61, s33 // (op 13749) reg expression
s_mov_b32 s62, s0 // (op 13749) reg expression
s_mov_b32 s63, 131072 // (op 13749) call()

v_and_b32 v10, -2, v1 // (op 13845) 
v_sub_u32 v11, v1, v10 // (op 13845) 

v_lshlrev_b32 v10, 6, v11 // (op 13845) 

v_lshrrev_b32 v11, 1, v1 // (op 13845) 
v_lshl_add_u32 v14, v11, 8, v10 // (op 13845) 


v_mul_lo_u32 v10, v14, 6 // (op 13845) 

v_lshrrev_b32 v11, 3, v10 // (op 13845) 

v_and_b32 v10, -2, v1 // (op 13903) 
v_sub_u32 v14, v1, v10 // (op 13903) 

v_lshlrev_b32 v10, 6, v14 // (op 13903) 

v_lshrrev_b32 v14, 1, v1 // (op 13903) 
v_lshl_add_u32 v15, v14, 7, v10 // (op 13903) 


v_mul_lo_u32 v10, v15, 6 // (op 13903) 

v_lshrrev_b32 v14, 3, v10 // (op 13903) 

s_mul_hi_i32 s47, s16, s10 // (op 14139) 
v_and_b32 v10, 63, v1 // (op 14139) 
s_add_i32 s0, s47, s16 // (op 14139) 

v_lshrrev_b32 v15, 4, v10 // (op 14139) 
s_ashr_i32 s1, s0, 31 // (op 14139) 
v_and_b32 v16, 3, v15 // (op 14139) 

s_cmp_eq_i32 s10, 0 // (op 14139) 
s_mov_b32 s47, scc // (op 14139) 
v_and_b32 v15, 15, v1 // (op 14139) 
s_cmp_eq_i32 s47, 0 // (op 14139) 
s_cselect_b32 s18, 0, -1 // (op 14139) 

v_lshl_add_u32 v17, v16, 4, v15 // (op 14139) 


s_and_b32 s19, s13, 31 // (op 14139) 
v_and_b32 v15, 4294967168, v1 // (op 14139) 
s_lshl_b32 s47, 1, s19 // (op 14139) 
v_add_u32 v16, v15, v17 // (op 14139) 


s_add_i32 s22, s47, s18 // (op 14139) 


v_lshrrev_b32 v15, 6, v10 // (op 14139) 

s_and_b32 s18, s1, s22 // (op 14139) 


v_lshlrev_b32 v10, 2, v15 // (op 14139) 

s_add_i32 s1, s0, s18 // (op 14139) 


s_ashr_i32 s0, s1, s19 // (op 14139) 


s_lshl_b32 s1, s13, 24 // (op 14139) 
s_ashr_i32 s47, s1, 31 // (op 14139) 

s_xor_b32 s1, s0, s47 // (op 14139) 

s_sub_i32 s0, s1, s47 // (op 14139) 


s_mul_hi_i32 s1, s0, s14 // (op 14139) 
s_add_i32 s47, s1, s0 // (op 14139) 

s_ashr_i32 s1, s47, 31 // (op 14139) 
s_cmp_eq_i32 s14, 0 // (op 14139) 
s_mov_b32 s18, scc // (op 14139) 
s_cmp_eq_i32 s18, 0 // (op 14139) 
s_cselect_b32 s19, 0, -1 // (op 14139) 

s_and_b32 s18, s15, 31 // (op 14139) 
s_lshl_b32 s22, 1, s18 // (op 14139) 
s_add_i32 s23, s22, s19 // (op 14139) 


s_and_b32 s19, s1, s23 // (op 14139) 


s_add_i32 s1, s47, s19 // (op 14139) 


s_ashr_i32 s19, s1, s18 // (op 14139) 


s_lshl_b32 s1, s15, 24 // (op 14139) 
s_ashr_i32 s18, s1, 31 // (op 14139) 

s_xor_b32 s1, s19, s18 // (op 14139) 

s_sub_i32 s19, s1, s18 // (op 14139) 


s_mul_i32 s1, s0, s11 // (op 14139) 

s_sub_u32 s0, s16, s1 // (op 14139) 

s_mul_i32 s1, s19, s11 // (op 14139) 
s_add_u32 s1, s1, s0 // (op 14139) 


s_mul_hi_i32 s0, s16, s9 // (op 14139) 
s_add_i32 s47, s0, s16 // (op 14139) 

s_ashr_i32 s0, s47, 31 // (op 14139) 
s_cmp_eq_i32 s9, 0 // (op 14139) 
s_mov_b32 s18, scc // (op 14139) 
s_cmp_eq_i32 s18, 0 // (op 14139) 
s_cselect_b32 s19, 0, -1 // (op 14139) 

s_and_b32 s18, s45, 31 // (op 14139) 
s_lshl_b32 s22, 1, s18 // (op 14139) 
s_add_i32 s23, s22, s19 // (op 14139) 


s_and_b32 s19, s0, s23 // (op 14139) 


s_add_i32 s0, s47, s19 // (op 14139) 


s_ashr_i32 s19, s0, s18 // (op 14139) 


s_lshl_b32 s0, s45, 24 // (op 14139) 
s_ashr_i32 s18, s0, 31 // (op 14139) 

s_xor_b32 s0, s19, s18 // (op 14139) 

s_sub_i32 s19, s0, s18 // (op 14139) 


s_add_u32 s22, s6, 128 // (op 14139) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 14139) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 14139) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 14139) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 14139) 

s_mul_i32 s0, s11, s22 // (op 14139) 

s_mul_i32 s18, s8, s0 // (op 14139) 

s_mul_i32 s0, s19, s18 // (op 14139) 

s_sub_u32 s18, s16, s0 // (op 14139) 

s_mov_b32 s0, s18 // (op 14139) 
s_mul_hi_i32 s47, s0, s12 // (op 14139) 
s_add_i32 s22, s47, s0 // (op 14139) 


s_ashr_i32 s0, s22, 31 // (op 14139) 
s_cmp_eq_i32 s12, 0 // (op 14139) 
s_mov_b32 s47, scc // (op 14139) 
s_cmp_eq_i32 s47, 0 // (op 14139) 
s_cselect_b32 s23, 0, -1 // (op 14139) 

s_and_b32 s47, s44, 31 // (op 14139) 
s_lshl_b32 s24, 1, s47 // (op 14139) 
s_add_i32 s25, s24, s23 // (op 14139) 


s_and_b32 s23, s0, s25 // (op 14139) 


s_add_i32 s0, s22, s23 // (op 14139) 


s_ashr_i32 s22, s0, s47 // (op 14139) 


s_lshl_b32 s0, s44, 24 // (op 14139) 
s_ashr_i32 s47, s0, 31 // (op 14139) 

s_xor_b32 s0, s22, s47 // (op 14139) 

s_sub_i32 s22, s0, s47 // (op 14139) 


s_mul_i32 s0, s46, s11 // (op 14139) 
s_add_u32 s24, s20, 256 // (op 14139) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 14139) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 14139) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 14139) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 8 // (op 14139) 

s_sub_i32 s23, s24, s0 // (op 14139) 

s_mul_i32 s47, s22, s23 // (op 14139) 


s_sub_u32 s22, s18, s47 // (op 14139) 


s_add_u32 s18, s22, s0 // (op 14139) 


s_cmp_eq_i32 s19, 0 // (op 14139) 
s_mov_b32 s0, scc // (op 14139) 

s_cmp_eq_i32 s0, 0 // (op 14139) 
s_cselect_b32 s19, s18, s1 // (op 14139) 



v_lshl_add_u32 v15, s19, 8, v16 // (op 14139) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 14139) 
s_lshr_b32 s47, s0, 1 // (op 14139) 

v_lshl_add_u32 v16, s47, 3, v10 // (op 14139) 


v_mul_lo_u32 v10, v15, s34 // (op 14139) 
v_add_u32 v10, v10, v16 // (op 14139) 


s_mul_i32 s1, s34, s21 // (op 14141) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 14141) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 14141) least significant
s_mul_hi_u32 s18, s34, s20 // (op 14141) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 14141) most significant: sum
s_add_u32 s1, s1, s18 // (op 14141) most significant: sum


s_mov_b32 s64, s36 // (op 14141) reg expression
s_mov_b32 s65, s37 // (op 14141) reg expression
s_mov_b32 s66, s0 // (op 14141) reg expression
s_mov_b32 s67, 131072 // (op 14141) call()

s_mul_hi_i32 s47, s16, s10 // (op 14166) 
v_and_b32 v15, 63, v1 // (op 14166) 
s_add_i32 s0, s47, s16 // (op 14166) 

v_lshrrev_b32 v16, 4, v15 // (op 14166) 
s_ashr_i32 s1, s0, 31 // (op 14166) 
v_and_b32 v17, 3, v16 // (op 14166) 

s_cmp_eq_i32 s10, 0 // (op 14166) 
s_mov_b32 s47, scc // (op 14166) 
v_and_b32 v16, 15, v1 // (op 14166) 
s_cmp_eq_i32 s47, 0 // (op 14166) 
s_cselect_b32 s18, 0, -1 // (op 14166) 

v_lshl_add_u32 v18, v17, 4, v16 // (op 14166) 


s_and_b32 s19, s13, 31 // (op 14166) 
v_lshrrev_b32 v16, 6, v1 // (op 14166) 
s_lshl_b32 s47, 1, s19 // (op 14166) 
v_and_b32 v17, 1, v16 // (op 14166) 

s_add_i32 s22, s47, s18 // (op 14166) 


v_lshl_add_u32 v16, v17, 6, v18 // (op 14166) 


s_and_b32 s18, s1, s22 // (op 14166) 


v_lshrrev_b32 v17, 6, v15 // (op 14166) 

s_add_i32 s1, s0, s18 // (op 14166) 


v_lshlrev_b32 v15, 2, v17 // (op 14166) 

s_ashr_i32 s0, s1, s19 // (op 14166) 


s_lshl_b32 s1, s13, 24 // (op 14166) 
s_ashr_i32 s47, s1, 31 // (op 14166) 

s_xor_b32 s1, s0, s47 // (op 14166) 

s_sub_i32 s0, s1, s47 // (op 14166) 


s_mul_hi_i32 s1, s0, s14 // (op 14166) 
s_add_i32 s47, s1, s0 // (op 14166) 

s_ashr_i32 s1, s47, 31 // (op 14166) 
s_cmp_eq_i32 s14, 0 // (op 14166) 
s_mov_b32 s18, scc // (op 14166) 
s_cmp_eq_i32 s18, 0 // (op 14166) 
s_cselect_b32 s19, 0, -1 // (op 14166) 

s_and_b32 s18, s15, 31 // (op 14166) 
s_lshl_b32 s22, 1, s18 // (op 14166) 
s_add_i32 s23, s22, s19 // (op 14166) 


s_and_b32 s19, s1, s23 // (op 14166) 


s_add_i32 s1, s47, s19 // (op 14166) 


s_ashr_i32 s19, s1, s18 // (op 14166) 


s_lshl_b32 s1, s15, 24 // (op 14166) 
s_ashr_i32 s18, s1, 31 // (op 14166) 

s_xor_b32 s1, s19, s18 // (op 14166) 

s_sub_i32 s19, s1, s18 // (op 14166) 


s_add_u32 s22, s6, 128 // (op 14166) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 14166) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 14166) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 14166) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 14166) 

s_mul_i32 s1, s19, s22 // (op 14166) 

s_sub_i32 s47, s0, s1 // (op 14166) 


s_mul_hi_i32 s0, s16, s9 // (op 14166) 
s_add_i32 s1, s0, s16 // (op 14166) 

s_ashr_i32 s0, s1, 31 // (op 14166) 
s_cmp_eq_i32 s9, 0 // (op 14166) 
s_mov_b32 s18, scc // (op 14166) 
s_cmp_eq_i32 s18, 0 // (op 14166) 
s_cselect_b32 s19, 0, -1 // (op 14166) 

s_and_b32 s18, s45, 31 // (op 14166) 
s_lshl_b32 s24, 1, s18 // (op 14166) 
s_add_i32 s25, s24, s19 // (op 14166) 


s_and_b32 s19, s0, s25 // (op 14166) 


s_add_i32 s0, s1, s19 // (op 14166) 


s_ashr_i32 s1, s0, s18 // (op 14166) 


s_lshl_b32 s0, s45, 24 // (op 14166) 
s_ashr_i32 s18, s0, 31 // (op 14166) 

s_xor_b32 s0, s1, s18 // (op 14166) 

s_sub_i32 s1, s0, s18 // (op 14166) 


s_mul_i32 s0, s11, s22 // (op 14166) 

s_mul_i32 s18, s8, s0 // (op 14166) 

s_mul_i32 s0, s1, s18 // (op 14166) 

s_sub_u32 s18, s16, s0 // (op 14166) 

s_mul_hi_i32 s0, s18, s12 // (op 14166) 
s_add_i32 s19, s0, s18 // (op 14166) 


s_ashr_i32 s0, s19, 31 // (op 14166) 
s_cmp_eq_i32 s12, 0 // (op 14166) 
s_mov_b32 s18, scc // (op 14166) 
s_cmp_eq_i32 s18, 0 // (op 14166) 
s_cselect_b32 s22, 0, -1 // (op 14166) 

s_and_b32 s18, s44, 31 // (op 14166) 
s_lshl_b32 s23, 1, s18 // (op 14166) 
s_add_i32 s24, s23, s22 // (op 14166) 


s_and_b32 s25, s0, s24 // (op 14166) 


s_add_i32 s0, s19, s25 // (op 14166) 


s_ashr_i32 s19, s0, s18 // (op 14166) 


s_lshl_b32 s0, s44, 24 // (op 14166) 
s_ashr_i32 s18, s0, 31 // (op 14166) 

s_xor_b32 s0, s19, s18 // (op 14166) 

s_sub_i32 s19, s0, s18 // (op 14166) 


s_cmp_eq_i32 s1, 0 // (op 14166) 
s_mov_b32 s0, scc // (op 14166) 

s_cmp_eq_i32 s0, 0 // (op 14166) 
s_cselect_b32 s1, s19, s47 // (op 14166) 



v_lshl_add_u32 v17, s1, 7, v16 // (op 14166) 


v_mul_lo_u32 v16, v17, s38 // (op 14166) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 14166) 
s_lshr_b32 s47, s0, 1 // (op 14166) 

v_lshl_add_u32 v17, s47, 3, v15 // (op 14166) 


v_add_u32 v15, v17, v16 // (op 14166) 


s_mul_i32 s1, s38, s7 // (op 14168) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 14168) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 14168) least significant
s_mul_hi_u32 s18, s38, s6 // (op 14168) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 14168) most significant: sum
s_add_u32 s1, s1, s18 // (op 14168) most significant: sum


s_mov_b32 s68, s40 // (op 14168) reg expression
s_mov_b32 s69, s41 // (op 14168) reg expression
s_mov_b32 s70, s0 // (op 14168) reg expression
s_mov_b32 s71, 131072 // (op 14168) call()

s_ashr_i64 s[0:1], s[42:43], 7 // (op 4525) 
s_and_b32 s47, s0, 4294967294 // (op 4525) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 4524) 
s_mov_b32 s18, s0 // (op 4524) convert

s_cmp_lt_u32 s47, s18 // (op 4524) 






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ForLoopBottom_KLoopTail_4524 // (op 4524) Condition: Top (jump to RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ForLoopBottom_KLoopTail_4524 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ForLoopTop_KLoopTail_4524:

s_add_u32 s47, s47, 1 // (op 4526) 
s_mul_i32 s1, s27, 6 // (op 161) most significant: low of high * low
s_mul_i32 s0, s26, 6 // (op 161) least significant
s_mul_hi_u32 s18, s26, 6 // (op 161) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 161) most significant: sum

v_lshrrev_b64 v[16:17], 3, s[0:1] // (op 161) 


v_mov_b32 v18, v12 // (op 161) 
v_mov_b32 v19, v18 // (op 161) 
buffer_load_dwordx4 v[20:23], v19, s[48:51], 0 offen // (op 161) Load value
buffer_load_dwordx2 v[24:25], v19, s[48:51], 0 offen offset:16 // (op 161) Load value
buffer_load_dwordx4 v[26:29], v19, s[48:51], 0 offen offset:24 // (op 161) Load value
buffer_load_dwordx2 v[30:31], v19, s[48:51], 0 offen offset:40 // (op 161) Load value
v_add_u32 v19, v19, v16 // (op 161) 
buffer_load_dwordx4 v[32:35], v19, s[48:51], 0 offen // (op 161) Load value
buffer_load_dwordx2 v[36:37], v19, s[48:51], 0 offen offset:16 // (op 161) Load value
buffer_load_dwordx4 v[38:41], v19, s[48:51], 0 offen offset:24 // (op 161) Load value
buffer_load_dwordx2 v[42:43], v19, s[48:51], 0 offen offset:40 // (op 161) Load value



v_add_u32 v12, 96, v12 // (op 13639) 


s_barrier  // (op 14318) 
v_mov_b32 v16, v13 // (op 173) 




buffer_load_dwordx4 v[44:47], v16, s[60:63], 0 offen // (op 173) Load value
s_waitcnt vmcnt(8)
ds_write_b128 v11, v[20:23] offset:12288 // (op 162) Store local data 
s_waitcnt vmcnt(7)
ds_write_b64 v11, v[24:25] offset:12304 // (op 162) Store local data 
s_waitcnt vmcnt(6)
ds_write_b128 v11, v[26:29] offset:12312 // (op 162) Store local data 
s_waitcnt vmcnt(5)
ds_write_b64 v11, v[30:31] offset:12328 // (op 162) Store local data 
s_waitcnt vmcnt(4)
ds_write_b128 v11, v[32:35] offset:12384 // (op 162) Store local data 
s_waitcnt vmcnt(3)
ds_write_b64 v11, v[36:37] offset:12400 // (op 162) Store local data 
s_waitcnt vmcnt(2)
ds_write_b128 v11, v[38:41] offset:12408 // (op 162) Store local data 
buffer_load_dwordx2 v[48:49], v16, s[60:63], 0 offen offset:16 // (op 173) Load value
buffer_load_dwordx4 v[50:53], v16, s[60:63], 0 offen offset:24 // (op 173) Load value
buffer_load_dwordx2 v[54:55], v16, s[60:63], 0 offen offset:40 // (op 173) Load value

v_add_u32 v13, 96, v13 // (op 13739) 
s_waitcnt vmcnt(4)
ds_write_b64 v11, v[42:43] offset:12424 // (op 162) Store local data 

s_barrier  // (op 14331) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 14321) 






ds_read_b128 v[16:19], v3 offset:12288 // (op 164) Load local data 
ds_read_b64 v[20:21], v3 offset:12304 // (op 164) Load local data 
s_waitcnt vmcnt(3)
ds_write_b128 v14, v[44:47] // (op 172) Store local data 
s_waitcnt vmcnt(2)
ds_write_b64 v14, v[48:49] offset:16 // (op 172) Store local data 
s_waitcnt vmcnt(1)
ds_write_b128 v14, v[50:53] offset:24 // (op 172) Store local data 
s_waitcnt vmcnt(0)
ds_write_b64 v14, v[54:55] offset:40 // (op 172) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 14335) 



ds_read_b128 v[22:25], v4 // (op 166) Load local data 
ds_read_b64 v[26:27], v4 offset:16 // (op 166) Load local data 


v_mov_b32 v28, v10 // (op 168) 
v_mov_b32 v29, v28 // (op 168) reg expression

buffer_load_dwordx2 v[30:31], v29, s[64:67], 0 offen // (op 168) Load value



v_mov_b32 v29, v15 // (op 170) 
v_mov_b32 v32, v29 // (op 170) reg expression

buffer_load_dwordx2 v[34:35], v32, s[68:71], 0 offen // (op 170) Load value


v_add_u32 v15, 4, v15 // (op 14156) 
s_waitcnt vmcnt(1)
v_bfe_u32 v36, v30, 0, 8 // (op 12607) 



v_bfe_u32 v37, v30, 8, 8 // (op 12607) 
v_bfe_u32 v38, v30, 16, 8 // (op 12607) 
v_bfe_u32 v39, v30, 24, 8 // (op 12607) 
v_permlane16_swap_b32 v36, v37 // (op 12607) 
ds_read_b128 v[40:43], v4 offset:1536 // (op 295) Load local data 
v_permlane16_swap_b32 v38, v39 // (op 12607) 
s_nop 1
v_permlane32_swap_b32 v36, v38 // (op 12607) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v37, v39 // (op 12607) 
ds_read_b64 v[44:45], v4 offset:1552 // (op 295) Load local data 




ds_read_b128 v[46:49], v4 offset:3072 // (op 431) Load local data 
ds_read_b64 v[50:51], v4 offset:3088 // (op 431) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v30, v34, 0, 8 // (op 12667) 



v_bfe_u32 v31, v34, 8, 8 // (op 12667) 
v_bfe_u32 v32, v34, 16, 8 // (op 12667) 
v_bfe_u32 v33, v34, 24, 8 // (op 12667) 
v_permlane16_swap_b32 v30, v31 // (op 12667) 
s_nop 0
v_permlane16_swap_b32 v32, v33 // (op 12667) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v30, v32 // (op 12667) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v31, v33 // (op 12667) 
ds_read_b128 v[52:55], v4 offset:4608 // (op 567) Load local data 

ds_read_b64 v[56:57], v4 offset:4624 // (op 567) Load local data 



s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[16:21], v[22:27], a[0:3], v36, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 171) 
ds_read_b128 v[58:61], v3 offset:13824 // (op 789) Load local data 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[16:21], v[40:45], a[4:7], v36, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 298) 
ds_read_b64 v[62:63], v3 offset:13840 // (op 789) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[16:21], v[46:51], a[8:11], v36, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 434) 
ds_read_b128 v[64:67], v3 offset:15360 // (op 1399) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[16:21], v[52:57], a[12:15], v36, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 570) 
ds_read_b64 v[68:69], v3 offset:15376 // (op 1399) Load local data 




s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[58:63], v[22:27], a[16:19], v37, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 793) 
ds_read_b128 v[16:19], v3 offset:16896 // (op 2009) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[58:63], v[46:51], a[24:27], v37, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 847) 
ds_read_b64 v[20:21], v3 offset:16912 // (op 2009) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[58:63], v[52:57], a[28:31], v37, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 874) 
ds_read_b128 v[70:73], v3 offset:18432 // (op 2619) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[58:63], v[40:45], a[20:23], v37, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 820) 
ds_read_b64 v[74:75], v3 offset:18448 // (op 2619) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[64:69], v[22:27], a[32:35], v38, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1403) 
s_lshl_b32 s0, s34, 6 // (op 2628) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[64:69], v[46:51], a[40:43], v38, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1457) 
v_add_u32 v29, v28, s0 // (op 2628) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[64:69], v[52:57], a[44:47], v38, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1484) 
buffer_load_dwordx2 v[34:35], v29, s[64:67], 0 offen // (op 2628) Load value


v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[64:69], v[40:45], a[36:39], v38, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1430) 




v_add_u32 v10, 4, v10 // (op 14129) 
ds_read_b128 v[58:61], v3 offset:19968 // (op 3229) Load local data 
ds_read_b64 v[62:63], v3 offset:19984 // (op 3229) Load local data 
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[16:21], v[22:27], a[48:51], v39, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2013) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[16:21], v[46:51], a[56:59], v39, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2067) 
s_waitcnt vmcnt(0)
v_bfe_u32 v64, v34, 0, 8 // (op 12621) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[16:21], v[52:57], a[60:63], v39, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2094) 
v_bfe_u32 v65, v34, 8, 8 // (op 12621) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[16:21], v[40:45], a[52:55], v39, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2040) 
v_bfe_u32 v66, v34, 16, 8 // (op 12621) 


v_bfe_u32 v67, v34, 24, 8 // (op 12621) 
v_permlane16_swap_b32 v64, v65 // (op 12621) 
s_nop 0
v_permlane16_swap_b32 v66, v67 // (op 12621) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v64, v66 // (op 12621) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v65, v67 // (op 12621) 
ds_read_b128 v[16:19], v3 offset:21504 // (op 3839) Load local data 

ds_read_b64 v[20:21], v3 offset:21520 // (op 3839) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[70:75], v[40:45], a[68:71], v64, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2650) 
ds_read_b128 v[34:37], v3 offset:23040 // (op 4449) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[70:75], v[22:27], a[64:67], v64, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2623) 
ds_read_b64 v[38:39], v3 offset:23056 // (op 4449) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[70:75], v[46:51], a[72:75], v64, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2677) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[70:75], v[52:57], a[76:79], v64, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2704) 

s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[58:63], v[40:45], a[84:87], v65, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 3260) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[58:63], v[22:27], a[80:83], v65, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 3233) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[58:63], v[46:51], a[88:91], v65, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 3287) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[58:63], v[52:57], a[92:95], v65, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 3314) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[16:21], v[40:45], a[100:103], v66, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 3870) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[16:21], v[22:27], a[96:99], v66, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 3843) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[16:21], v[46:51], a[104:107], v66, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 3897) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[16:21], v[52:57], a[108:111], v66, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 3924) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[34:39], v[40:45], a[116:119], v67, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 4480) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[34:39], v[22:27], a[112:115], v67, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 4453) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[34:39], v[46:51], a[120:123], v67, v32 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 4507) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[34:39], v[52:57], a[124:127], v67, v33 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 4534) 




s_ashr_i64 s[0:1], s[42:43], 7 // (op 4524) 
s_mov_b32 s18, s0 // (op 4524) convert

s_cmp_lt_u32 s47, s18 // (op 4524) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ForLoopTop_KLoopTail_4524 // (op 4524) Condition: Bottom (jump to RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ForLoopTop_KLoopTail_4524 if true)
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ForLoopBottom_KLoopTail_4524:




















s_barrier  // (op 14447) 

















v_cmp_eq_f32 vcc, 0.00000, v0 // (op 116) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ConditionalFalse_0_beta_116 // (op 116) Condition: False, jump to RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ConditionalFalse_0_beta_116
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



v_accvgpr_read v10, a4 // (op 5650) 
v_accvgpr_read v11, a5 // (op 5650) 
v_accvgpr_read v12, a6 // (op 5650) 
v_accvgpr_read v13, a7 // (op 5650) 
v_mul_f32 v14, v7, v10 // (op 5650) 
v_mul_f32 v15, v7, v11 // (op 5650) 
v_mul_f32 v16, v7, v12 // (op 5650) 
v_mul_f32 v17, v7, v13 // (op 5650) 

v_cvt_f16_f32 v10, v14 // (op 5652) 
v_cvt_f16_f32 v11, v15 // (op 5652) 
v_cvt_f16_f32 v12, v16 // (op 5652) 
v_cvt_f16_f32 v13, v17 // (op 5652) 


v_pack_b32_f16 v4, v10, v11 // (op 5653) 
v_pack_b32_f16 v5, v12, v13 // (op 5653) 

s_lshl_b32 s0, s4, 5 // (op 5653) 
v_add_u32 v3, v9, s0 // (op 5653) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5653) Store value



v_accvgpr_read v10, a8 // (op 5664) 
v_accvgpr_read v11, a9 // (op 5664) 
v_accvgpr_read v12, a10 // (op 5664) 
v_accvgpr_read v13, a11 // (op 5664) 
v_mul_f32 v14, v7, v10 // (op 5664) 
v_mul_f32 v15, v7, v11 // (op 5664) 
v_mul_f32 v16, v7, v12 // (op 5664) 
v_mul_f32 v17, v7, v13 // (op 5664) 

v_cvt_f16_f32 v10, v14 // (op 5666) 
v_cvt_f16_f32 v11, v15 // (op 5666) 
v_cvt_f16_f32 v12, v16 // (op 5666) 
v_cvt_f16_f32 v13, v17 // (op 5666) 


v_pack_b32_f16 v4, v10, v11 // (op 5667) 
v_pack_b32_f16 v5, v12, v13 // (op 5667) 

s_lshl_b32 s0, s4, 6 // (op 5667) 
v_add_u32 v3, v9, s0 // (op 5667) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5667) Store value



v_accvgpr_read v10, a12 // (op 5678) 
v_accvgpr_read v11, a13 // (op 5678) 
v_accvgpr_read v12, a14 // (op 5678) 
v_accvgpr_read v13, a15 // (op 5678) 
v_mul_f32 v14, v7, v10 // (op 5678) 
v_mul_f32 v15, v7, v11 // (op 5678) 
v_mul_f32 v16, v7, v12 // (op 5678) 
v_mul_f32 v17, v7, v13 // (op 5678) 

v_cvt_f16_f32 v10, v14 // (op 5680) 
v_cvt_f16_f32 v11, v15 // (op 5680) 
v_cvt_f16_f32 v12, v16 // (op 5680) 
v_cvt_f16_f32 v13, v17 // (op 5680) 


v_pack_b32_f16 v4, v10, v11 // (op 5681) 
v_pack_b32_f16 v5, v12, v13 // (op 5681) 

s_lshl_b32 s0, s4, 5 // (op 5681) 
s_mul_i32 s1, 3, s0 // (op 5681) 

v_add_u32 v3, v9, s1 // (op 5681) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5681) Store value



v_accvgpr_read v10, a16 // (op 5714) 
v_accvgpr_read v11, a17 // (op 5714) 
v_accvgpr_read v12, a18 // (op 5714) 
v_accvgpr_read v13, a19 // (op 5714) 
v_mul_f32 v14, v7, v10 // (op 5714) 
v_mul_f32 v15, v7, v11 // (op 5714) 
v_mul_f32 v16, v7, v12 // (op 5714) 
v_mul_f32 v17, v7, v13 // (op 5714) 

v_cvt_f16_f32 v10, v14 // (op 5716) 
v_cvt_f16_f32 v11, v15 // (op 5716) 
v_cvt_f16_f32 v12, v16 // (op 5716) 
v_cvt_f16_f32 v13, v17 // (op 5716) 


v_pack_b32_f16 v4, v10, v11 // (op 5718) 
v_pack_b32_f16 v5, v12, v13 // (op 5718) 

v_add_u32 v3, 32, v9 // (op 5718) 

buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5718) Store value



v_accvgpr_read v10, a20 // (op 5728) 
v_accvgpr_read v11, a21 // (op 5728) 
v_accvgpr_read v12, a22 // (op 5728) 
v_accvgpr_read v13, a23 // (op 5728) 
v_mul_f32 v14, v7, v10 // (op 5728) 
v_mul_f32 v15, v7, v11 // (op 5728) 
v_mul_f32 v16, v7, v12 // (op 5728) 
v_mul_f32 v17, v7, v13 // (op 5728) 

v_cvt_f16_f32 v10, v14 // (op 5730) 
v_cvt_f16_f32 v11, v15 // (op 5730) 
v_cvt_f16_f32 v12, v16 // (op 5730) 
v_cvt_f16_f32 v13, v17 // (op 5730) 


v_pack_b32_f16 v4, v10, v11 // (op 5720) 
v_pack_b32_f16 v5, v12, v13 // (op 5720) 

s_lshl_b32 s0, s4, 5 // (op 5720) 
s_add_u32 s0, s0, 32 // (op 5720) 
v_add_u32 v3, v9, s0 // (op 5720) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5720) Store value



v_accvgpr_read v10, a24 // (op 5734) 
v_accvgpr_read v11, a25 // (op 5734) 
v_accvgpr_read v12, a26 // (op 5734) 
v_accvgpr_read v13, a27 // (op 5734) 
v_mul_f32 v14, v7, v10 // (op 5734) 
v_mul_f32 v15, v7, v11 // (op 5734) 
v_mul_f32 v16, v7, v12 // (op 5734) 
v_mul_f32 v17, v7, v13 // (op 5734) 

v_cvt_f16_f32 v10, v14 // (op 5736) 
v_cvt_f16_f32 v11, v15 // (op 5736) 
v_cvt_f16_f32 v12, v16 // (op 5736) 
v_cvt_f16_f32 v13, v17 // (op 5736) 


v_pack_b32_f16 v4, v10, v11 // (op 5722) 
v_pack_b32_f16 v5, v12, v13 // (op 5722) 

s_lshl_b32 s0, s4, 6 // (op 5722) 
s_add_u32 s0, s0, 32 // (op 5722) 
v_add_u32 v3, v9, s0 // (op 5722) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5722) Store value



v_accvgpr_read v10, a28 // (op 5740) 
v_accvgpr_read v11, a29 // (op 5740) 
v_accvgpr_read v12, a30 // (op 5740) 
v_accvgpr_read v13, a31 // (op 5740) 
v_mul_f32 v14, v7, v10 // (op 5740) 
v_mul_f32 v15, v7, v11 // (op 5740) 
v_mul_f32 v16, v7, v12 // (op 5740) 
v_mul_f32 v17, v7, v13 // (op 5740) 

v_cvt_f16_f32 v10, v14 // (op 5742) 
v_cvt_f16_f32 v11, v15 // (op 5742) 
v_cvt_f16_f32 v12, v16 // (op 5742) 
v_cvt_f16_f32 v13, v17 // (op 5742) 


v_pack_b32_f16 v4, v10, v11 // (op 5724) 
v_pack_b32_f16 v5, v12, v13 // (op 5724) 

s_lshl_b32 s0, s4, 5 // (op 5724) 
s_mul_i32 s1, 3, s0 // (op 5724) 
s_add_u32 s1, s1, 32 // (op 5724) 

v_add_u32 v3, v9, s1 // (op 5724) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5724) Store value



v_accvgpr_read v10, a32 // (op 5790) 
v_accvgpr_read v11, a33 // (op 5790) 
v_accvgpr_read v12, a34 // (op 5790) 
v_accvgpr_read v13, a35 // (op 5790) 
v_mul_f32 v14, v7, v10 // (op 5790) 
v_mul_f32 v15, v7, v11 // (op 5790) 
v_mul_f32 v16, v7, v12 // (op 5790) 
v_mul_f32 v17, v7, v13 // (op 5790) 

v_cvt_f16_f32 v10, v14 // (op 5792) 
v_cvt_f16_f32 v11, v15 // (op 5792) 
v_cvt_f16_f32 v12, v16 // (op 5792) 
v_cvt_f16_f32 v13, v17 // (op 5792) 


v_pack_b32_f16 v4, v10, v11 // (op 5794) 
v_pack_b32_f16 v5, v12, v13 // (op 5794) 

v_add_u32 v3, 64, v9 // (op 5794) 

buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5794) Store value



v_accvgpr_read v10, a36 // (op 5804) 
v_accvgpr_read v11, a37 // (op 5804) 
v_accvgpr_read v12, a38 // (op 5804) 
v_accvgpr_read v13, a39 // (op 5804) 
v_mul_f32 v14, v7, v10 // (op 5804) 
v_mul_f32 v15, v7, v11 // (op 5804) 
v_mul_f32 v16, v7, v12 // (op 5804) 
v_mul_f32 v17, v7, v13 // (op 5804) 

v_cvt_f16_f32 v10, v14 // (op 5806) 
v_cvt_f16_f32 v11, v15 // (op 5806) 
v_cvt_f16_f32 v12, v16 // (op 5806) 
v_cvt_f16_f32 v13, v17 // (op 5806) 


v_pack_b32_f16 v4, v10, v11 // (op 5796) 
v_pack_b32_f16 v5, v12, v13 // (op 5796) 

s_lshl_b32 s0, s4, 5 // (op 5796) 
s_add_u32 s0, s0, 64 // (op 5796) 
v_add_u32 v3, v9, s0 // (op 5796) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5796) Store value



v_accvgpr_read v10, a40 // (op 5810) 
v_accvgpr_read v11, a41 // (op 5810) 
v_accvgpr_read v12, a42 // (op 5810) 
v_accvgpr_read v13, a43 // (op 5810) 
v_mul_f32 v14, v7, v10 // (op 5810) 
v_mul_f32 v15, v7, v11 // (op 5810) 
v_mul_f32 v16, v7, v12 // (op 5810) 
v_mul_f32 v17, v7, v13 // (op 5810) 

v_cvt_f16_f32 v10, v14 // (op 5812) 
v_cvt_f16_f32 v11, v15 // (op 5812) 
v_cvt_f16_f32 v12, v16 // (op 5812) 
v_cvt_f16_f32 v13, v17 // (op 5812) 


v_pack_b32_f16 v4, v10, v11 // (op 5798) 
v_pack_b32_f16 v5, v12, v13 // (op 5798) 

s_lshl_b32 s0, s4, 6 // (op 5798) 
s_add_u32 s0, s0, 64 // (op 5798) 
v_add_u32 v3, v9, s0 // (op 5798) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5798) Store value



v_accvgpr_read v10, a44 // (op 5816) 
v_accvgpr_read v11, a45 // (op 5816) 
v_accvgpr_read v12, a46 // (op 5816) 
v_accvgpr_read v13, a47 // (op 5816) 
v_mul_f32 v14, v7, v10 // (op 5816) 
v_mul_f32 v15, v7, v11 // (op 5816) 
v_mul_f32 v16, v7, v12 // (op 5816) 
v_mul_f32 v17, v7, v13 // (op 5816) 

v_cvt_f16_f32 v10, v14 // (op 5818) 
v_cvt_f16_f32 v11, v15 // (op 5818) 
v_cvt_f16_f32 v12, v16 // (op 5818) 
v_cvt_f16_f32 v13, v17 // (op 5818) 


v_pack_b32_f16 v4, v10, v11 // (op 5800) 
v_pack_b32_f16 v5, v12, v13 // (op 5800) 

s_lshl_b32 s0, s4, 5 // (op 5800) 
s_mul_i32 s1, 3, s0 // (op 5800) 
s_add_u32 s1, s1, 64 // (op 5800) 

v_add_u32 v3, v9, s1 // (op 5800) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5800) Store value



v_accvgpr_read v10, a48 // (op 5866) 
v_accvgpr_read v11, a49 // (op 5866) 
v_accvgpr_read v12, a50 // (op 5866) 
v_accvgpr_read v13, a51 // (op 5866) 
v_mul_f32 v14, v7, v10 // (op 5866) 
v_mul_f32 v15, v7, v11 // (op 5866) 
v_mul_f32 v16, v7, v12 // (op 5866) 
v_mul_f32 v17, v7, v13 // (op 5866) 

v_cvt_f16_f32 v10, v14 // (op 5868) 
v_cvt_f16_f32 v11, v15 // (op 5868) 
v_cvt_f16_f32 v12, v16 // (op 5868) 
v_cvt_f16_f32 v13, v17 // (op 5868) 


v_pack_b32_f16 v4, v10, v11 // (op 5870) 
v_pack_b32_f16 v5, v12, v13 // (op 5870) 

v_add_u32 v3, 96, v9 // (op 5870) 

buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5870) Store value



v_accvgpr_read v10, a52 // (op 5880) 
v_accvgpr_read v11, a53 // (op 5880) 
v_accvgpr_read v12, a54 // (op 5880) 
v_accvgpr_read v13, a55 // (op 5880) 
v_mul_f32 v14, v7, v10 // (op 5880) 
v_mul_f32 v15, v7, v11 // (op 5880) 
v_mul_f32 v16, v7, v12 // (op 5880) 
v_mul_f32 v17, v7, v13 // (op 5880) 

v_cvt_f16_f32 v10, v14 // (op 5882) 
v_cvt_f16_f32 v11, v15 // (op 5882) 
v_cvt_f16_f32 v12, v16 // (op 5882) 
v_cvt_f16_f32 v13, v17 // (op 5882) 


v_pack_b32_f16 v4, v10, v11 // (op 5872) 
v_pack_b32_f16 v5, v12, v13 // (op 5872) 

s_lshl_b32 s0, s4, 5 // (op 5872) 
s_add_u32 s0, s0, 96 // (op 5872) 
v_add_u32 v3, v9, s0 // (op 5872) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5872) Store value



v_accvgpr_read v10, a56 // (op 5886) 
v_accvgpr_read v11, a57 // (op 5886) 
v_accvgpr_read v12, a58 // (op 5886) 
v_accvgpr_read v13, a59 // (op 5886) 
v_mul_f32 v14, v7, v10 // (op 5886) 
v_mul_f32 v15, v7, v11 // (op 5886) 
v_mul_f32 v16, v7, v12 // (op 5886) 
v_mul_f32 v17, v7, v13 // (op 5886) 

v_cvt_f16_f32 v10, v14 // (op 5888) 
v_cvt_f16_f32 v11, v15 // (op 5888) 
v_cvt_f16_f32 v12, v16 // (op 5888) 
v_cvt_f16_f32 v13, v17 // (op 5888) 


v_pack_b32_f16 v4, v10, v11 // (op 5874) 
v_pack_b32_f16 v5, v12, v13 // (op 5874) 

s_lshl_b32 s0, s4, 6 // (op 5874) 
s_add_u32 s0, s0, 96 // (op 5874) 
v_add_u32 v3, v9, s0 // (op 5874) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5874) Store value



v_accvgpr_read v10, a60 // (op 5892) 
v_accvgpr_read v11, a61 // (op 5892) 
v_accvgpr_read v12, a62 // (op 5892) 
v_accvgpr_read v13, a63 // (op 5892) 
v_mul_f32 v14, v7, v10 // (op 5892) 
v_mul_f32 v15, v7, v11 // (op 5892) 
v_mul_f32 v16, v7, v12 // (op 5892) 
v_mul_f32 v17, v7, v13 // (op 5892) 

v_cvt_f16_f32 v10, v14 // (op 5894) 
v_cvt_f16_f32 v11, v15 // (op 5894) 
v_cvt_f16_f32 v12, v16 // (op 5894) 
v_cvt_f16_f32 v13, v17 // (op 5894) 


v_pack_b32_f16 v4, v10, v11 // (op 5876) 
v_pack_b32_f16 v5, v12, v13 // (op 5876) 

s_lshl_b32 s0, s4, 5 // (op 5876) 
s_mul_i32 s1, 3, s0 // (op 5876) 
s_add_u32 s1, s1, 96 // (op 5876) 

v_add_u32 v3, v9, s1 // (op 5876) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5876) Store value



v_accvgpr_read v10, a64 // (op 5942) 
v_accvgpr_read v11, a65 // (op 5942) 
v_accvgpr_read v12, a66 // (op 5942) 
v_accvgpr_read v13, a67 // (op 5942) 
v_mul_f32 v14, v7, v10 // (op 5942) 
v_mul_f32 v15, v7, v11 // (op 5942) 
v_mul_f32 v16, v7, v12 // (op 5942) 
v_mul_f32 v17, v7, v13 // (op 5942) 

v_cvt_f16_f32 v10, v14 // (op 5944) 
v_cvt_f16_f32 v11, v15 // (op 5944) 
v_cvt_f16_f32 v12, v16 // (op 5944) 
v_cvt_f16_f32 v13, v17 // (op 5944) 


v_pack_b32_f16 v4, v10, v11 // (op 5946) 
v_pack_b32_f16 v5, v12, v13 // (op 5946) 

v_add_u32 v3, 128, v9 // (op 5946) 

buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5946) Store value



v_accvgpr_read v10, a68 // (op 5956) 
v_accvgpr_read v11, a69 // (op 5956) 
v_accvgpr_read v12, a70 // (op 5956) 
v_accvgpr_read v13, a71 // (op 5956) 
v_mul_f32 v14, v7, v10 // (op 5956) 
v_mul_f32 v15, v7, v11 // (op 5956) 
v_mul_f32 v16, v7, v12 // (op 5956) 
v_mul_f32 v17, v7, v13 // (op 5956) 

v_cvt_f16_f32 v10, v14 // (op 5958) 
v_cvt_f16_f32 v11, v15 // (op 5958) 
v_cvt_f16_f32 v12, v16 // (op 5958) 
v_cvt_f16_f32 v13, v17 // (op 5958) 


v_pack_b32_f16 v4, v10, v11 // (op 5948) 
v_pack_b32_f16 v5, v12, v13 // (op 5948) 

s_lshl_b32 s0, s4, 5 // (op 5948) 
s_add_u32 s0, s0, 128 // (op 5948) 
v_add_u32 v3, v9, s0 // (op 5948) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5948) Store value



v_accvgpr_read v10, a72 // (op 5962) 
v_accvgpr_read v11, a73 // (op 5962) 
v_accvgpr_read v12, a74 // (op 5962) 
v_accvgpr_read v13, a75 // (op 5962) 
v_mul_f32 v14, v7, v10 // (op 5962) 
v_mul_f32 v15, v7, v11 // (op 5962) 
v_mul_f32 v16, v7, v12 // (op 5962) 
v_mul_f32 v17, v7, v13 // (op 5962) 

v_cvt_f16_f32 v10, v14 // (op 5964) 
v_cvt_f16_f32 v11, v15 // (op 5964) 
v_cvt_f16_f32 v12, v16 // (op 5964) 
v_cvt_f16_f32 v13, v17 // (op 5964) 


v_pack_b32_f16 v4, v10, v11 // (op 5950) 
v_pack_b32_f16 v5, v12, v13 // (op 5950) 

s_lshl_b32 s0, s4, 6 // (op 5950) 
s_add_u32 s0, s0, 128 // (op 5950) 
v_add_u32 v3, v9, s0 // (op 5950) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5950) Store value



v_accvgpr_read v10, a76 // (op 5968) 
v_accvgpr_read v11, a77 // (op 5968) 
v_accvgpr_read v12, a78 // (op 5968) 
v_accvgpr_read v13, a79 // (op 5968) 
v_mul_f32 v14, v7, v10 // (op 5968) 
v_mul_f32 v15, v7, v11 // (op 5968) 
v_mul_f32 v16, v7, v12 // (op 5968) 
v_mul_f32 v17, v7, v13 // (op 5968) 

v_cvt_f16_f32 v10, v14 // (op 5970) 
v_cvt_f16_f32 v11, v15 // (op 5970) 
v_cvt_f16_f32 v12, v16 // (op 5970) 
v_cvt_f16_f32 v13, v17 // (op 5970) 


v_pack_b32_f16 v4, v10, v11 // (op 5952) 
v_pack_b32_f16 v5, v12, v13 // (op 5952) 

s_lshl_b32 s0, s4, 5 // (op 5952) 
s_mul_i32 s1, 3, s0 // (op 5952) 
s_add_u32 s1, s1, 128 // (op 5952) 

v_add_u32 v3, v9, s1 // (op 5952) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5952) Store value



v_accvgpr_read v10, a80 // (op 6018) 
v_accvgpr_read v11, a81 // (op 6018) 
v_accvgpr_read v12, a82 // (op 6018) 
v_accvgpr_read v13, a83 // (op 6018) 
v_mul_f32 v14, v7, v10 // (op 6018) 
v_mul_f32 v15, v7, v11 // (op 6018) 
v_mul_f32 v16, v7, v12 // (op 6018) 
v_mul_f32 v17, v7, v13 // (op 6018) 

v_cvt_f16_f32 v10, v14 // (op 6020) 
v_cvt_f16_f32 v11, v15 // (op 6020) 
v_cvt_f16_f32 v12, v16 // (op 6020) 
v_cvt_f16_f32 v13, v17 // (op 6020) 


v_pack_b32_f16 v4, v10, v11 // (op 6022) 
v_pack_b32_f16 v5, v12, v13 // (op 6022) 

v_add_u32 v3, 160, v9 // (op 6022) 

buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 6022) Store value



v_accvgpr_read v10, a84 // (op 6032) 
v_accvgpr_read v11, a85 // (op 6032) 
v_accvgpr_read v12, a86 // (op 6032) 
v_accvgpr_read v13, a87 // (op 6032) 
v_mul_f32 v14, v7, v10 // (op 6032) 
v_mul_f32 v15, v7, v11 // (op 6032) 
v_mul_f32 v16, v7, v12 // (op 6032) 
v_mul_f32 v17, v7, v13 // (op 6032) 

v_cvt_f16_f32 v10, v14 // (op 6034) 
v_cvt_f16_f32 v11, v15 // (op 6034) 
v_cvt_f16_f32 v12, v16 // (op 6034) 
v_cvt_f16_f32 v13, v17 // (op 6034) 


v_pack_b32_f16 v4, v10, v11 // (op 6024) 
v_pack_b32_f16 v5, v12, v13 // (op 6024) 

s_lshl_b32 s0, s4, 5 // (op 6024) 
s_add_u32 s0, s0, 160 // (op 6024) 
v_add_u32 v3, v9, s0 // (op 6024) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 6024) Store value



v_accvgpr_read v10, a88 // (op 6038) 
v_accvgpr_read v11, a89 // (op 6038) 
v_accvgpr_read v12, a90 // (op 6038) 
v_accvgpr_read v13, a91 // (op 6038) 
v_mul_f32 v14, v7, v10 // (op 6038) 
v_mul_f32 v15, v7, v11 // (op 6038) 
v_mul_f32 v16, v7, v12 // (op 6038) 
v_mul_f32 v17, v7, v13 // (op 6038) 

v_cvt_f16_f32 v10, v14 // (op 6040) 
v_cvt_f16_f32 v11, v15 // (op 6040) 
v_cvt_f16_f32 v12, v16 // (op 6040) 
v_cvt_f16_f32 v13, v17 // (op 6040) 


v_pack_b32_f16 v4, v10, v11 // (op 6026) 
v_pack_b32_f16 v5, v12, v13 // (op 6026) 

s_lshl_b32 s0, s4, 6 // (op 6026) 
s_add_u32 s0, s0, 160 // (op 6026) 
v_add_u32 v3, v9, s0 // (op 6026) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 6026) Store value



v_accvgpr_read v10, a92 // (op 6044) 
v_accvgpr_read v11, a93 // (op 6044) 
v_accvgpr_read v12, a94 // (op 6044) 
v_accvgpr_read v13, a95 // (op 6044) 
v_mul_f32 v14, v7, v10 // (op 6044) 
v_mul_f32 v15, v7, v11 // (op 6044) 
v_mul_f32 v16, v7, v12 // (op 6044) 
v_mul_f32 v17, v7, v13 // (op 6044) 

v_cvt_f16_f32 v10, v14 // (op 6046) 
v_cvt_f16_f32 v11, v15 // (op 6046) 
v_cvt_f16_f32 v12, v16 // (op 6046) 
v_cvt_f16_f32 v13, v17 // (op 6046) 


v_pack_b32_f16 v4, v10, v11 // (op 6028) 
v_pack_b32_f16 v5, v12, v13 // (op 6028) 

s_lshl_b32 s0, s4, 5 // (op 6028) 
s_mul_i32 s1, 3, s0 // (op 6028) 
s_add_u32 s1, s1, 160 // (op 6028) 

v_add_u32 v3, v9, s1 // (op 6028) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 6028) Store value



v_accvgpr_read v10, a96 // (op 6094) 
v_accvgpr_read v11, a97 // (op 6094) 
v_accvgpr_read v12, a98 // (op 6094) 
v_accvgpr_read v13, a99 // (op 6094) 
v_mul_f32 v14, v7, v10 // (op 6094) 
v_mul_f32 v15, v7, v11 // (op 6094) 
v_mul_f32 v16, v7, v12 // (op 6094) 
v_mul_f32 v17, v7, v13 // (op 6094) 

v_cvt_f16_f32 v10, v14 // (op 6096) 
v_cvt_f16_f32 v11, v15 // (op 6096) 
v_cvt_f16_f32 v12, v16 // (op 6096) 
v_cvt_f16_f32 v13, v17 // (op 6096) 


v_pack_b32_f16 v4, v10, v11 // (op 6098) 
v_pack_b32_f16 v5, v12, v13 // (op 6098) 

v_add_u32 v3, 192, v9 // (op 6098) 

buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 6098) Store value



v_accvgpr_read v10, a100 // (op 6108) 
v_accvgpr_read v11, a101 // (op 6108) 
v_accvgpr_read v12, a102 // (op 6108) 
v_accvgpr_read v13, a103 // (op 6108) 
v_mul_f32 v14, v7, v10 // (op 6108) 
v_mul_f32 v15, v7, v11 // (op 6108) 
v_mul_f32 v16, v7, v12 // (op 6108) 
v_mul_f32 v17, v7, v13 // (op 6108) 

v_cvt_f16_f32 v10, v14 // (op 6110) 
v_cvt_f16_f32 v11, v15 // (op 6110) 
v_cvt_f16_f32 v12, v16 // (op 6110) 
v_cvt_f16_f32 v13, v17 // (op 6110) 


v_pack_b32_f16 v4, v10, v11 // (op 6100) 
v_pack_b32_f16 v5, v12, v13 // (op 6100) 

s_lshl_b32 s0, s4, 5 // (op 6100) 
s_add_u32 s0, s0, 192 // (op 6100) 
v_add_u32 v3, v9, s0 // (op 6100) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 6100) Store value



v_accvgpr_read v10, a104 // (op 6114) 
v_accvgpr_read v11, a105 // (op 6114) 
v_accvgpr_read v12, a106 // (op 6114) 
v_accvgpr_read v13, a107 // (op 6114) 
v_mul_f32 v14, v7, v10 // (op 6114) 
v_mul_f32 v15, v7, v11 // (op 6114) 
v_mul_f32 v16, v7, v12 // (op 6114) 
v_mul_f32 v17, v7, v13 // (op 6114) 

v_cvt_f16_f32 v10, v14 // (op 6116) 
v_cvt_f16_f32 v11, v15 // (op 6116) 
v_cvt_f16_f32 v12, v16 // (op 6116) 
v_cvt_f16_f32 v13, v17 // (op 6116) 


v_pack_b32_f16 v4, v10, v11 // (op 6102) 
v_pack_b32_f16 v5, v12, v13 // (op 6102) 

s_lshl_b32 s0, s4, 6 // (op 6102) 
s_add_u32 s0, s0, 192 // (op 6102) 
v_add_u32 v3, v9, s0 // (op 6102) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 6102) Store value



v_accvgpr_read v10, a108 // (op 6120) 
v_accvgpr_read v11, a109 // (op 6120) 
v_accvgpr_read v12, a110 // (op 6120) 
v_accvgpr_read v13, a111 // (op 6120) 
v_mul_f32 v14, v7, v10 // (op 6120) 
v_mul_f32 v15, v7, v11 // (op 6120) 
v_mul_f32 v16, v7, v12 // (op 6120) 
v_mul_f32 v17, v7, v13 // (op 6120) 

v_cvt_f16_f32 v10, v14 // (op 6122) 
v_cvt_f16_f32 v11, v15 // (op 6122) 
v_cvt_f16_f32 v12, v16 // (op 6122) 
v_cvt_f16_f32 v13, v17 // (op 6122) 


v_pack_b32_f16 v4, v10, v11 // (op 6104) 
v_pack_b32_f16 v5, v12, v13 // (op 6104) 

s_lshl_b32 s0, s4, 5 // (op 6104) 
s_mul_i32 s1, 3, s0 // (op 6104) 
s_add_u32 s1, s1, 192 // (op 6104) 

v_add_u32 v3, v9, s1 // (op 6104) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 6104) Store value



v_accvgpr_read v10, a112 // (op 6170) 
v_accvgpr_read v11, a113 // (op 6170) 
v_accvgpr_read v12, a114 // (op 6170) 
v_accvgpr_read v13, a115 // (op 6170) 
v_mul_f32 v14, v7, v10 // (op 6170) 
v_mul_f32 v15, v7, v11 // (op 6170) 
v_mul_f32 v16, v7, v12 // (op 6170) 
v_mul_f32 v17, v7, v13 // (op 6170) 

v_cvt_f16_f32 v10, v14 // (op 6172) 
v_cvt_f16_f32 v11, v15 // (op 6172) 
v_cvt_f16_f32 v12, v16 // (op 6172) 
v_cvt_f16_f32 v13, v17 // (op 6172) 


v_pack_b32_f16 v4, v10, v11 // (op 6174) 
v_pack_b32_f16 v5, v12, v13 // (op 6174) 

v_add_u32 v3, 224, v9 // (op 6174) 

buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 6174) Store value



v_accvgpr_read v10, a116 // (op 6184) 
v_accvgpr_read v11, a117 // (op 6184) 
v_accvgpr_read v12, a118 // (op 6184) 
v_accvgpr_read v13, a119 // (op 6184) 
v_mul_f32 v14, v7, v10 // (op 6184) 
v_mul_f32 v15, v7, v11 // (op 6184) 
v_mul_f32 v16, v7, v12 // (op 6184) 
v_mul_f32 v17, v7, v13 // (op 6184) 

v_cvt_f16_f32 v10, v14 // (op 6186) 
v_cvt_f16_f32 v11, v15 // (op 6186) 
v_cvt_f16_f32 v12, v16 // (op 6186) 
v_cvt_f16_f32 v13, v17 // (op 6186) 


v_pack_b32_f16 v4, v10, v11 // (op 6176) 
v_pack_b32_f16 v5, v12, v13 // (op 6176) 

s_lshl_b32 s0, s4, 5 // (op 6176) 
s_add_u32 s0, s0, 224 // (op 6176) 
v_add_u32 v3, v9, s0 // (op 6176) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 6176) Store value



v_accvgpr_read v10, a120 // (op 6190) 
v_accvgpr_read v11, a121 // (op 6190) 
v_accvgpr_read v12, a122 // (op 6190) 
v_accvgpr_read v13, a123 // (op 6190) 
v_mul_f32 v14, v7, v10 // (op 6190) 
v_mul_f32 v15, v7, v11 // (op 6190) 
v_mul_f32 v16, v7, v12 // (op 6190) 
v_mul_f32 v17, v7, v13 // (op 6190) 

v_cvt_f16_f32 v10, v14 // (op 6192) 
v_cvt_f16_f32 v11, v15 // (op 6192) 
v_cvt_f16_f32 v12, v16 // (op 6192) 
v_cvt_f16_f32 v13, v17 // (op 6192) 


v_pack_b32_f16 v4, v10, v11 // (op 6178) 
v_pack_b32_f16 v5, v12, v13 // (op 6178) 

s_lshl_b32 s0, s4, 6 // (op 6178) 
s_add_u32 s0, s0, 224 // (op 6178) 
v_add_u32 v3, v9, s0 // (op 6178) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 6178) Store value



v_accvgpr_read v10, a124 // (op 6196) 
v_accvgpr_read v11, a125 // (op 6196) 
v_accvgpr_read v12, a126 // (op 6196) 
v_accvgpr_read v13, a127 // (op 6196) 
v_mul_f32 v14, v7, v10 // (op 6196) 
v_mul_f32 v15, v7, v11 // (op 6196) 
v_mul_f32 v16, v7, v12 // (op 6196) 
v_mul_f32 v17, v7, v13 // (op 6196) 

v_cvt_f16_f32 v10, v14 // (op 6198) 
v_cvt_f16_f32 v11, v15 // (op 6198) 
v_cvt_f16_f32 v12, v16 // (op 6198) 
v_cvt_f16_f32 v13, v17 // (op 6198) 


v_pack_b32_f16 v4, v10, v11 // (op 6180) 
v_pack_b32_f16 v5, v12, v13 // (op 6180) 

s_lshl_b32 s0, s4, 5 // (op 6180) 
s_mul_i32 s1, 3, s0 // (op 6180) 
s_add_u32 s1, s1, 224 // (op 6180) 

v_add_u32 v3, v9, s1 // (op 6180) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 6180) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ConditionalBottom_0_beta_116 // (op 116) Condition: Done, jump to RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ConditionalBottom_0_beta_116
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ConditionalFalse_0_beta_116:


v_add_u32 v3, 224, v8 // (op 6972) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6972) Load value

v_accvgpr_read v10, a112 // (op 6959) 
v_accvgpr_read v11, a113 // (op 6959) 
v_accvgpr_read v12, a114 // (op 6959) 
v_accvgpr_read v13, a115 // (op 6959) 
v_mul_f32 v14, v7, v10 // (op 6959) 
v_mul_f32 v15, v7, v11 // (op 6959) 
v_mul_f32 v16, v7, v12 // (op 6959) 
v_mul_f32 v17, v7, v13 // (op 6959) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6971) 
v_lshrrev_b32 v11, 16, v4 // (op 6971) 
v_cvt_f32_f16 v11, v11 // (op 6971) 
v_mul_f32 v18, v0, v10 // (op 6971) 
v_mul_f32 v19, v0, v11 // (op 6971) 
v_cvt_f32_f16 v10, v5 // (op 6971) 
v_lshrrev_b32 v11, 16, v5 // (op 6971) 
v_cvt_f32_f16 v11, v11 // (op 6971) 
v_mul_f32 v20, v0, v10 // (op 6971) 
v_mul_f32 v21, v0, v11 // (op 6971) 


v_add_f32 v10, v18, v14 // (op 6960) 
v_add_f32 v11, v19, v15 // (op 6960) 
v_add_f32 v12, v20, v16 // (op 6960) 
v_add_f32 v13, v21, v17 // (op 6960) 


v_cvt_f16_f32 v14, v10 // (op 6961) 
v_cvt_f16_f32 v15, v11 // (op 6961) 
v_cvt_f16_f32 v16, v12 // (op 6961) 
v_cvt_f16_f32 v17, v13 // (op 6961) 


v_pack_b32_f16 v10, v14, v15 // (op 6963) 
v_pack_b32_f16 v11, v16, v17 // (op 6963) 

v_add_u32 v3, 224, v9 // (op 6963) 

buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6963) Store value





s_lshl_b32 s0, s2, 5 // (op 6977) 
s_add_u32 s0, s0, 224 // (op 6977) 
v_add_u32 v3, v8, s0 // (op 6977) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6977) Load value

v_accvgpr_read v10, a116 // (op 6989) 
v_accvgpr_read v11, a117 // (op 6989) 
v_accvgpr_read v12, a118 // (op 6989) 
v_accvgpr_read v13, a119 // (op 6989) 
v_mul_f32 v14, v7, v10 // (op 6989) 
v_mul_f32 v15, v7, v11 // (op 6989) 
v_mul_f32 v16, v7, v12 // (op 6989) 
v_mul_f32 v17, v7, v13 // (op 6989) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6974) 
v_lshrrev_b32 v11, 16, v4 // (op 6974) 
v_cvt_f32_f16 v11, v11 // (op 6974) 
v_mul_f32 v18, v0, v10 // (op 6974) 
v_mul_f32 v19, v0, v11 // (op 6974) 
v_cvt_f32_f16 v10, v5 // (op 6974) 
v_lshrrev_b32 v11, 16, v5 // (op 6974) 
v_cvt_f32_f16 v11, v11 // (op 6974) 
v_mul_f32 v20, v0, v10 // (op 6974) 
v_mul_f32 v21, v0, v11 // (op 6974) 


v_add_f32 v10, v18, v14 // (op 6975) 
v_add_f32 v11, v19, v15 // (op 6975) 
v_add_f32 v12, v20, v16 // (op 6975) 
v_add_f32 v13, v21, v17 // (op 6975) 


v_cvt_f16_f32 v14, v10 // (op 6976) 
v_cvt_f16_f32 v15, v11 // (op 6976) 
v_cvt_f16_f32 v16, v12 // (op 6976) 
v_cvt_f16_f32 v17, v13 // (op 6976) 


v_pack_b32_f16 v10, v14, v15 // (op 6965) 
v_pack_b32_f16 v11, v16, v17 // (op 6965) 

s_lshl_b32 s0, s4, 5 // (op 6965) 
s_add_u32 s0, s0, 224 // (op 6965) 
v_add_u32 v3, v9, s0 // (op 6965) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6965) Store value





s_lshl_b32 s0, s2, 6 // (op 6982) 
s_add_u32 s0, s0, 224 // (op 6982) 
v_add_u32 v3, v8, s0 // (op 6982) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6982) Load value

v_accvgpr_read v10, a120 // (op 6991) 
v_accvgpr_read v11, a121 // (op 6991) 
v_accvgpr_read v12, a122 // (op 6991) 
v_accvgpr_read v13, a123 // (op 6991) 
v_mul_f32 v14, v7, v10 // (op 6991) 
v_mul_f32 v15, v7, v11 // (op 6991) 
v_mul_f32 v16, v7, v12 // (op 6991) 
v_mul_f32 v17, v7, v13 // (op 6991) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6979) 
v_lshrrev_b32 v11, 16, v4 // (op 6979) 
v_cvt_f32_f16 v11, v11 // (op 6979) 
v_mul_f32 v18, v0, v10 // (op 6979) 
v_mul_f32 v19, v0, v11 // (op 6979) 
v_cvt_f32_f16 v10, v5 // (op 6979) 
v_lshrrev_b32 v11, 16, v5 // (op 6979) 
v_cvt_f32_f16 v11, v11 // (op 6979) 
v_mul_f32 v20, v0, v10 // (op 6979) 
v_mul_f32 v21, v0, v11 // (op 6979) 


v_add_f32 v10, v18, v14 // (op 6980) 
v_add_f32 v11, v19, v15 // (op 6980) 
v_add_f32 v12, v20, v16 // (op 6980) 
v_add_f32 v13, v21, v17 // (op 6980) 


v_cvt_f16_f32 v14, v10 // (op 6981) 
v_cvt_f16_f32 v15, v11 // (op 6981) 
v_cvt_f16_f32 v16, v12 // (op 6981) 
v_cvt_f16_f32 v17, v13 // (op 6981) 


v_pack_b32_f16 v10, v14, v15 // (op 6967) 
v_pack_b32_f16 v11, v16, v17 // (op 6967) 

s_lshl_b32 s0, s4, 6 // (op 6967) 
s_add_u32 s0, s0, 224 // (op 6967) 
v_add_u32 v3, v9, s0 // (op 6967) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6967) Store value





s_lshl_b32 s0, s2, 5 // (op 6987) 
s_mul_i32 s1, 3, s0 // (op 6987) 
s_add_u32 s1, s1, 224 // (op 6987) 

v_add_u32 v3, v8, s1 // (op 6987) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6987) Load value

v_accvgpr_read v10, a124 // (op 6993) 
v_accvgpr_read v11, a125 // (op 6993) 
v_accvgpr_read v12, a126 // (op 6993) 
v_accvgpr_read v13, a127 // (op 6993) 
v_mul_f32 v14, v7, v10 // (op 6993) 
v_mul_f32 v15, v7, v11 // (op 6993) 
v_mul_f32 v16, v7, v12 // (op 6993) 
v_mul_f32 v17, v7, v13 // (op 6993) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6984) 
v_lshrrev_b32 v11, 16, v4 // (op 6984) 
v_cvt_f32_f16 v11, v11 // (op 6984) 
v_mul_f32 v18, v0, v10 // (op 6984) 
v_mul_f32 v19, v0, v11 // (op 6984) 
v_cvt_f32_f16 v10, v5 // (op 6984) 
v_lshrrev_b32 v11, 16, v5 // (op 6984) 
v_cvt_f32_f16 v11, v11 // (op 6984) 
v_mul_f32 v20, v0, v10 // (op 6984) 
v_mul_f32 v21, v0, v11 // (op 6984) 


v_add_f32 v10, v18, v14 // (op 6985) 
v_add_f32 v11, v19, v15 // (op 6985) 
v_add_f32 v12, v20, v16 // (op 6985) 
v_add_f32 v13, v21, v17 // (op 6985) 


v_cvt_f16_f32 v14, v10 // (op 6986) 
v_cvt_f16_f32 v15, v11 // (op 6986) 
v_cvt_f16_f32 v16, v12 // (op 6986) 
v_cvt_f16_f32 v17, v13 // (op 6986) 


v_pack_b32_f16 v10, v14, v15 // (op 6969) 
v_pack_b32_f16 v11, v16, v17 // (op 6969) 

s_lshl_b32 s0, s4, 5 // (op 6969) 
s_mul_i32 s1, 3, s0 // (op 6969) 
s_add_u32 s1, s1, 224 // (op 6969) 

v_add_u32 v3, v9, s1 // (op 6969) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6969) Store value





v_add_u32 v3, 160, v8 // (op 6798) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6798) Load value

v_accvgpr_read v10, a80 // (op 6785) 
v_accvgpr_read v11, a81 // (op 6785) 
v_accvgpr_read v12, a82 // (op 6785) 
v_accvgpr_read v13, a83 // (op 6785) 
v_mul_f32 v14, v7, v10 // (op 6785) 
v_mul_f32 v15, v7, v11 // (op 6785) 
v_mul_f32 v16, v7, v12 // (op 6785) 
v_mul_f32 v17, v7, v13 // (op 6785) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6797) 
v_lshrrev_b32 v11, 16, v4 // (op 6797) 
v_cvt_f32_f16 v11, v11 // (op 6797) 
v_mul_f32 v18, v0, v10 // (op 6797) 
v_mul_f32 v19, v0, v11 // (op 6797) 
v_cvt_f32_f16 v10, v5 // (op 6797) 
v_lshrrev_b32 v11, 16, v5 // (op 6797) 
v_cvt_f32_f16 v11, v11 // (op 6797) 
v_mul_f32 v20, v0, v10 // (op 6797) 
v_mul_f32 v21, v0, v11 // (op 6797) 


v_add_f32 v10, v18, v14 // (op 6786) 
v_add_f32 v11, v19, v15 // (op 6786) 
v_add_f32 v12, v20, v16 // (op 6786) 
v_add_f32 v13, v21, v17 // (op 6786) 


v_cvt_f16_f32 v14, v10 // (op 6787) 
v_cvt_f16_f32 v15, v11 // (op 6787) 
v_cvt_f16_f32 v16, v12 // (op 6787) 
v_cvt_f16_f32 v17, v13 // (op 6787) 


v_pack_b32_f16 v10, v14, v15 // (op 6789) 
v_pack_b32_f16 v11, v16, v17 // (op 6789) 

v_add_u32 v3, 160, v9 // (op 6789) 

buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6789) Store value





s_lshl_b32 s0, s2, 5 // (op 6803) 
s_add_u32 s0, s0, 160 // (op 6803) 
v_add_u32 v3, v8, s0 // (op 6803) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6803) Load value

v_accvgpr_read v10, a84 // (op 6815) 
v_accvgpr_read v11, a85 // (op 6815) 
v_accvgpr_read v12, a86 // (op 6815) 
v_accvgpr_read v13, a87 // (op 6815) 
v_mul_f32 v14, v7, v10 // (op 6815) 
v_mul_f32 v15, v7, v11 // (op 6815) 
v_mul_f32 v16, v7, v12 // (op 6815) 
v_mul_f32 v17, v7, v13 // (op 6815) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6800) 
v_lshrrev_b32 v11, 16, v4 // (op 6800) 
v_cvt_f32_f16 v11, v11 // (op 6800) 
v_mul_f32 v18, v0, v10 // (op 6800) 
v_mul_f32 v19, v0, v11 // (op 6800) 
v_cvt_f32_f16 v10, v5 // (op 6800) 
v_lshrrev_b32 v11, 16, v5 // (op 6800) 
v_cvt_f32_f16 v11, v11 // (op 6800) 
v_mul_f32 v20, v0, v10 // (op 6800) 
v_mul_f32 v21, v0, v11 // (op 6800) 


v_add_f32 v10, v18, v14 // (op 6801) 
v_add_f32 v11, v19, v15 // (op 6801) 
v_add_f32 v12, v20, v16 // (op 6801) 
v_add_f32 v13, v21, v17 // (op 6801) 


v_cvt_f16_f32 v14, v10 // (op 6802) 
v_cvt_f16_f32 v15, v11 // (op 6802) 
v_cvt_f16_f32 v16, v12 // (op 6802) 
v_cvt_f16_f32 v17, v13 // (op 6802) 


v_pack_b32_f16 v10, v14, v15 // (op 6791) 
v_pack_b32_f16 v11, v16, v17 // (op 6791) 

s_lshl_b32 s0, s4, 5 // (op 6791) 
s_add_u32 s0, s0, 160 // (op 6791) 
v_add_u32 v3, v9, s0 // (op 6791) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6791) Store value





s_lshl_b32 s0, s2, 6 // (op 6808) 
s_add_u32 s0, s0, 160 // (op 6808) 
v_add_u32 v3, v8, s0 // (op 6808) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6808) Load value

v_accvgpr_read v10, a88 // (op 6817) 
v_accvgpr_read v11, a89 // (op 6817) 
v_accvgpr_read v12, a90 // (op 6817) 
v_accvgpr_read v13, a91 // (op 6817) 
v_mul_f32 v14, v7, v10 // (op 6817) 
v_mul_f32 v15, v7, v11 // (op 6817) 
v_mul_f32 v16, v7, v12 // (op 6817) 
v_mul_f32 v17, v7, v13 // (op 6817) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6805) 
v_lshrrev_b32 v11, 16, v4 // (op 6805) 
v_cvt_f32_f16 v11, v11 // (op 6805) 
v_mul_f32 v18, v0, v10 // (op 6805) 
v_mul_f32 v19, v0, v11 // (op 6805) 
v_cvt_f32_f16 v10, v5 // (op 6805) 
v_lshrrev_b32 v11, 16, v5 // (op 6805) 
v_cvt_f32_f16 v11, v11 // (op 6805) 
v_mul_f32 v20, v0, v10 // (op 6805) 
v_mul_f32 v21, v0, v11 // (op 6805) 


v_add_f32 v10, v18, v14 // (op 6806) 
v_add_f32 v11, v19, v15 // (op 6806) 
v_add_f32 v12, v20, v16 // (op 6806) 
v_add_f32 v13, v21, v17 // (op 6806) 


v_cvt_f16_f32 v14, v10 // (op 6807) 
v_cvt_f16_f32 v15, v11 // (op 6807) 
v_cvt_f16_f32 v16, v12 // (op 6807) 
v_cvt_f16_f32 v17, v13 // (op 6807) 


v_pack_b32_f16 v10, v14, v15 // (op 6793) 
v_pack_b32_f16 v11, v16, v17 // (op 6793) 

s_lshl_b32 s0, s4, 6 // (op 6793) 
s_add_u32 s0, s0, 160 // (op 6793) 
v_add_u32 v3, v9, s0 // (op 6793) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6793) Store value





s_lshl_b32 s0, s2, 5 // (op 6813) 
s_mul_i32 s1, 3, s0 // (op 6813) 
s_add_u32 s1, s1, 160 // (op 6813) 

v_add_u32 v3, v8, s1 // (op 6813) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6813) Load value

v_accvgpr_read v10, a92 // (op 6819) 
v_accvgpr_read v11, a93 // (op 6819) 
v_accvgpr_read v12, a94 // (op 6819) 
v_accvgpr_read v13, a95 // (op 6819) 
v_mul_f32 v14, v7, v10 // (op 6819) 
v_mul_f32 v15, v7, v11 // (op 6819) 
v_mul_f32 v16, v7, v12 // (op 6819) 
v_mul_f32 v17, v7, v13 // (op 6819) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6810) 
v_lshrrev_b32 v11, 16, v4 // (op 6810) 
v_cvt_f32_f16 v11, v11 // (op 6810) 
v_mul_f32 v18, v0, v10 // (op 6810) 
v_mul_f32 v19, v0, v11 // (op 6810) 
v_cvt_f32_f16 v10, v5 // (op 6810) 
v_lshrrev_b32 v11, 16, v5 // (op 6810) 
v_cvt_f32_f16 v11, v11 // (op 6810) 
v_mul_f32 v20, v0, v10 // (op 6810) 
v_mul_f32 v21, v0, v11 // (op 6810) 


v_add_f32 v10, v18, v14 // (op 6811) 
v_add_f32 v11, v19, v15 // (op 6811) 
v_add_f32 v12, v20, v16 // (op 6811) 
v_add_f32 v13, v21, v17 // (op 6811) 


v_cvt_f16_f32 v14, v10 // (op 6812) 
v_cvt_f16_f32 v15, v11 // (op 6812) 
v_cvt_f16_f32 v16, v12 // (op 6812) 
v_cvt_f16_f32 v17, v13 // (op 6812) 


v_pack_b32_f16 v10, v14, v15 // (op 6795) 
v_pack_b32_f16 v11, v16, v17 // (op 6795) 

s_lshl_b32 s0, s4, 5 // (op 6795) 
s_mul_i32 s1, 3, s0 // (op 6795) 
s_add_u32 s1, s1, 160 // (op 6795) 

v_add_u32 v3, v9, s1 // (op 6795) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6795) Store value





v_add_u32 v3, 128, v8 // (op 6711) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6711) Load value

v_accvgpr_read v10, a64 // (op 6698) 
v_accvgpr_read v11, a65 // (op 6698) 
v_accvgpr_read v12, a66 // (op 6698) 
v_accvgpr_read v13, a67 // (op 6698) 
v_mul_f32 v14, v7, v10 // (op 6698) 
v_mul_f32 v15, v7, v11 // (op 6698) 
v_mul_f32 v16, v7, v12 // (op 6698) 
v_mul_f32 v17, v7, v13 // (op 6698) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6710) 
v_lshrrev_b32 v11, 16, v4 // (op 6710) 
v_cvt_f32_f16 v11, v11 // (op 6710) 
v_mul_f32 v18, v0, v10 // (op 6710) 
v_mul_f32 v19, v0, v11 // (op 6710) 
v_cvt_f32_f16 v10, v5 // (op 6710) 
v_lshrrev_b32 v11, 16, v5 // (op 6710) 
v_cvt_f32_f16 v11, v11 // (op 6710) 
v_mul_f32 v20, v0, v10 // (op 6710) 
v_mul_f32 v21, v0, v11 // (op 6710) 


v_add_f32 v10, v18, v14 // (op 6699) 
v_add_f32 v11, v19, v15 // (op 6699) 
v_add_f32 v12, v20, v16 // (op 6699) 
v_add_f32 v13, v21, v17 // (op 6699) 


v_cvt_f16_f32 v14, v10 // (op 6700) 
v_cvt_f16_f32 v15, v11 // (op 6700) 
v_cvt_f16_f32 v16, v12 // (op 6700) 
v_cvt_f16_f32 v17, v13 // (op 6700) 


v_pack_b32_f16 v10, v14, v15 // (op 6702) 
v_pack_b32_f16 v11, v16, v17 // (op 6702) 

v_add_u32 v3, 128, v9 // (op 6702) 

buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6702) Store value





s_lshl_b32 s0, s2, 5 // (op 6716) 
s_add_u32 s0, s0, 128 // (op 6716) 
v_add_u32 v3, v8, s0 // (op 6716) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6716) Load value

v_accvgpr_read v10, a68 // (op 6728) 
v_accvgpr_read v11, a69 // (op 6728) 
v_accvgpr_read v12, a70 // (op 6728) 
v_accvgpr_read v13, a71 // (op 6728) 
v_mul_f32 v14, v7, v10 // (op 6728) 
v_mul_f32 v15, v7, v11 // (op 6728) 
v_mul_f32 v16, v7, v12 // (op 6728) 
v_mul_f32 v17, v7, v13 // (op 6728) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6713) 
v_lshrrev_b32 v11, 16, v4 // (op 6713) 
v_cvt_f32_f16 v11, v11 // (op 6713) 
v_mul_f32 v18, v0, v10 // (op 6713) 
v_mul_f32 v19, v0, v11 // (op 6713) 
v_cvt_f32_f16 v10, v5 // (op 6713) 
v_lshrrev_b32 v11, 16, v5 // (op 6713) 
v_cvt_f32_f16 v11, v11 // (op 6713) 
v_mul_f32 v20, v0, v10 // (op 6713) 
v_mul_f32 v21, v0, v11 // (op 6713) 


v_add_f32 v10, v18, v14 // (op 6714) 
v_add_f32 v11, v19, v15 // (op 6714) 
v_add_f32 v12, v20, v16 // (op 6714) 
v_add_f32 v13, v21, v17 // (op 6714) 


v_cvt_f16_f32 v14, v10 // (op 6715) 
v_cvt_f16_f32 v15, v11 // (op 6715) 
v_cvt_f16_f32 v16, v12 // (op 6715) 
v_cvt_f16_f32 v17, v13 // (op 6715) 


v_pack_b32_f16 v10, v14, v15 // (op 6704) 
v_pack_b32_f16 v11, v16, v17 // (op 6704) 

s_lshl_b32 s0, s4, 5 // (op 6704) 
s_add_u32 s0, s0, 128 // (op 6704) 
v_add_u32 v3, v9, s0 // (op 6704) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6704) Store value





s_lshl_b32 s0, s2, 6 // (op 6721) 
s_add_u32 s0, s0, 128 // (op 6721) 
v_add_u32 v3, v8, s0 // (op 6721) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6721) Load value

v_accvgpr_read v10, a72 // (op 6730) 
v_accvgpr_read v11, a73 // (op 6730) 
v_accvgpr_read v12, a74 // (op 6730) 
v_accvgpr_read v13, a75 // (op 6730) 
v_mul_f32 v14, v7, v10 // (op 6730) 
v_mul_f32 v15, v7, v11 // (op 6730) 
v_mul_f32 v16, v7, v12 // (op 6730) 
v_mul_f32 v17, v7, v13 // (op 6730) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6718) 
v_lshrrev_b32 v11, 16, v4 // (op 6718) 
v_cvt_f32_f16 v11, v11 // (op 6718) 
v_mul_f32 v18, v0, v10 // (op 6718) 
v_mul_f32 v19, v0, v11 // (op 6718) 
v_cvt_f32_f16 v10, v5 // (op 6718) 
v_lshrrev_b32 v11, 16, v5 // (op 6718) 
v_cvt_f32_f16 v11, v11 // (op 6718) 
v_mul_f32 v20, v0, v10 // (op 6718) 
v_mul_f32 v21, v0, v11 // (op 6718) 


v_add_f32 v10, v18, v14 // (op 6719) 
v_add_f32 v11, v19, v15 // (op 6719) 
v_add_f32 v12, v20, v16 // (op 6719) 
v_add_f32 v13, v21, v17 // (op 6719) 


v_cvt_f16_f32 v14, v10 // (op 6720) 
v_cvt_f16_f32 v15, v11 // (op 6720) 
v_cvt_f16_f32 v16, v12 // (op 6720) 
v_cvt_f16_f32 v17, v13 // (op 6720) 


v_pack_b32_f16 v10, v14, v15 // (op 6706) 
v_pack_b32_f16 v11, v16, v17 // (op 6706) 

s_lshl_b32 s0, s4, 6 // (op 6706) 
s_add_u32 s0, s0, 128 // (op 6706) 
v_add_u32 v3, v9, s0 // (op 6706) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6706) Store value





s_lshl_b32 s0, s2, 5 // (op 6726) 
s_mul_i32 s1, 3, s0 // (op 6726) 
s_add_u32 s1, s1, 128 // (op 6726) 

v_add_u32 v3, v8, s1 // (op 6726) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6726) Load value

v_accvgpr_read v10, a76 // (op 6732) 
v_accvgpr_read v11, a77 // (op 6732) 
v_accvgpr_read v12, a78 // (op 6732) 
v_accvgpr_read v13, a79 // (op 6732) 
v_mul_f32 v14, v7, v10 // (op 6732) 
v_mul_f32 v15, v7, v11 // (op 6732) 
v_mul_f32 v16, v7, v12 // (op 6732) 
v_mul_f32 v17, v7, v13 // (op 6732) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6723) 
v_lshrrev_b32 v11, 16, v4 // (op 6723) 
v_cvt_f32_f16 v11, v11 // (op 6723) 
v_mul_f32 v18, v0, v10 // (op 6723) 
v_mul_f32 v19, v0, v11 // (op 6723) 
v_cvt_f32_f16 v10, v5 // (op 6723) 
v_lshrrev_b32 v11, 16, v5 // (op 6723) 
v_cvt_f32_f16 v11, v11 // (op 6723) 
v_mul_f32 v20, v0, v10 // (op 6723) 
v_mul_f32 v21, v0, v11 // (op 6723) 


v_add_f32 v10, v18, v14 // (op 6724) 
v_add_f32 v11, v19, v15 // (op 6724) 
v_add_f32 v12, v20, v16 // (op 6724) 
v_add_f32 v13, v21, v17 // (op 6724) 


v_cvt_f16_f32 v14, v10 // (op 6725) 
v_cvt_f16_f32 v15, v11 // (op 6725) 
v_cvt_f16_f32 v16, v12 // (op 6725) 
v_cvt_f16_f32 v17, v13 // (op 6725) 


v_pack_b32_f16 v10, v14, v15 // (op 6708) 
v_pack_b32_f16 v11, v16, v17 // (op 6708) 

s_lshl_b32 s0, s4, 5 // (op 6708) 
s_mul_i32 s1, 3, s0 // (op 6708) 
s_add_u32 s1, s1, 128 // (op 6708) 

v_add_u32 v3, v9, s1 // (op 6708) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6708) Store value





v_add_u32 v3, 96, v8 // (op 6624) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6624) Load value

v_accvgpr_read v10, a48 // (op 6611) 
v_accvgpr_read v11, a49 // (op 6611) 
v_accvgpr_read v12, a50 // (op 6611) 
v_accvgpr_read v13, a51 // (op 6611) 
v_mul_f32 v14, v7, v10 // (op 6611) 
v_mul_f32 v15, v7, v11 // (op 6611) 
v_mul_f32 v16, v7, v12 // (op 6611) 
v_mul_f32 v17, v7, v13 // (op 6611) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6623) 
v_lshrrev_b32 v11, 16, v4 // (op 6623) 
v_cvt_f32_f16 v11, v11 // (op 6623) 
v_mul_f32 v18, v0, v10 // (op 6623) 
v_mul_f32 v19, v0, v11 // (op 6623) 
v_cvt_f32_f16 v10, v5 // (op 6623) 
v_lshrrev_b32 v11, 16, v5 // (op 6623) 
v_cvt_f32_f16 v11, v11 // (op 6623) 
v_mul_f32 v20, v0, v10 // (op 6623) 
v_mul_f32 v21, v0, v11 // (op 6623) 


v_add_f32 v10, v18, v14 // (op 6612) 
v_add_f32 v11, v19, v15 // (op 6612) 
v_add_f32 v12, v20, v16 // (op 6612) 
v_add_f32 v13, v21, v17 // (op 6612) 


v_cvt_f16_f32 v14, v10 // (op 6613) 
v_cvt_f16_f32 v15, v11 // (op 6613) 
v_cvt_f16_f32 v16, v12 // (op 6613) 
v_cvt_f16_f32 v17, v13 // (op 6613) 


v_pack_b32_f16 v10, v14, v15 // (op 6615) 
v_pack_b32_f16 v11, v16, v17 // (op 6615) 

v_add_u32 v3, 96, v9 // (op 6615) 

buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6615) Store value





s_lshl_b32 s0, s2, 5 // (op 6629) 
s_add_u32 s0, s0, 96 // (op 6629) 
v_add_u32 v3, v8, s0 // (op 6629) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6629) Load value

v_accvgpr_read v10, a52 // (op 6641) 
v_accvgpr_read v11, a53 // (op 6641) 
v_accvgpr_read v12, a54 // (op 6641) 
v_accvgpr_read v13, a55 // (op 6641) 
v_mul_f32 v14, v7, v10 // (op 6641) 
v_mul_f32 v15, v7, v11 // (op 6641) 
v_mul_f32 v16, v7, v12 // (op 6641) 
v_mul_f32 v17, v7, v13 // (op 6641) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6626) 
v_lshrrev_b32 v11, 16, v4 // (op 6626) 
v_cvt_f32_f16 v11, v11 // (op 6626) 
v_mul_f32 v18, v0, v10 // (op 6626) 
v_mul_f32 v19, v0, v11 // (op 6626) 
v_cvt_f32_f16 v10, v5 // (op 6626) 
v_lshrrev_b32 v11, 16, v5 // (op 6626) 
v_cvt_f32_f16 v11, v11 // (op 6626) 
v_mul_f32 v20, v0, v10 // (op 6626) 
v_mul_f32 v21, v0, v11 // (op 6626) 


v_add_f32 v10, v18, v14 // (op 6627) 
v_add_f32 v11, v19, v15 // (op 6627) 
v_add_f32 v12, v20, v16 // (op 6627) 
v_add_f32 v13, v21, v17 // (op 6627) 


v_cvt_f16_f32 v14, v10 // (op 6628) 
v_cvt_f16_f32 v15, v11 // (op 6628) 
v_cvt_f16_f32 v16, v12 // (op 6628) 
v_cvt_f16_f32 v17, v13 // (op 6628) 


v_pack_b32_f16 v10, v14, v15 // (op 6617) 
v_pack_b32_f16 v11, v16, v17 // (op 6617) 

s_lshl_b32 s0, s4, 5 // (op 6617) 
s_add_u32 s0, s0, 96 // (op 6617) 
v_add_u32 v3, v9, s0 // (op 6617) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6617) Store value





s_lshl_b32 s0, s2, 6 // (op 6634) 
s_add_u32 s0, s0, 96 // (op 6634) 
v_add_u32 v3, v8, s0 // (op 6634) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6634) Load value

v_accvgpr_read v10, a56 // (op 6643) 
v_accvgpr_read v11, a57 // (op 6643) 
v_accvgpr_read v12, a58 // (op 6643) 
v_accvgpr_read v13, a59 // (op 6643) 
v_mul_f32 v14, v7, v10 // (op 6643) 
v_mul_f32 v15, v7, v11 // (op 6643) 
v_mul_f32 v16, v7, v12 // (op 6643) 
v_mul_f32 v17, v7, v13 // (op 6643) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6631) 
v_lshrrev_b32 v11, 16, v4 // (op 6631) 
v_cvt_f32_f16 v11, v11 // (op 6631) 
v_mul_f32 v18, v0, v10 // (op 6631) 
v_mul_f32 v19, v0, v11 // (op 6631) 
v_cvt_f32_f16 v10, v5 // (op 6631) 
v_lshrrev_b32 v11, 16, v5 // (op 6631) 
v_cvt_f32_f16 v11, v11 // (op 6631) 
v_mul_f32 v20, v0, v10 // (op 6631) 
v_mul_f32 v21, v0, v11 // (op 6631) 


v_add_f32 v10, v18, v14 // (op 6632) 
v_add_f32 v11, v19, v15 // (op 6632) 
v_add_f32 v12, v20, v16 // (op 6632) 
v_add_f32 v13, v21, v17 // (op 6632) 


v_cvt_f16_f32 v14, v10 // (op 6633) 
v_cvt_f16_f32 v15, v11 // (op 6633) 
v_cvt_f16_f32 v16, v12 // (op 6633) 
v_cvt_f16_f32 v17, v13 // (op 6633) 


v_pack_b32_f16 v10, v14, v15 // (op 6619) 
v_pack_b32_f16 v11, v16, v17 // (op 6619) 

s_lshl_b32 s0, s4, 6 // (op 6619) 
s_add_u32 s0, s0, 96 // (op 6619) 
v_add_u32 v3, v9, s0 // (op 6619) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6619) Store value





s_lshl_b32 s0, s2, 5 // (op 6639) 
s_mul_i32 s1, 3, s0 // (op 6639) 
s_add_u32 s1, s1, 96 // (op 6639) 

v_add_u32 v3, v8, s1 // (op 6639) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6639) Load value

v_accvgpr_read v10, a60 // (op 6645) 
v_accvgpr_read v11, a61 // (op 6645) 
v_accvgpr_read v12, a62 // (op 6645) 
v_accvgpr_read v13, a63 // (op 6645) 
v_mul_f32 v14, v7, v10 // (op 6645) 
v_mul_f32 v15, v7, v11 // (op 6645) 
v_mul_f32 v16, v7, v12 // (op 6645) 
v_mul_f32 v17, v7, v13 // (op 6645) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6636) 
v_lshrrev_b32 v11, 16, v4 // (op 6636) 
v_cvt_f32_f16 v11, v11 // (op 6636) 
v_mul_f32 v18, v0, v10 // (op 6636) 
v_mul_f32 v19, v0, v11 // (op 6636) 
v_cvt_f32_f16 v10, v5 // (op 6636) 
v_lshrrev_b32 v11, 16, v5 // (op 6636) 
v_cvt_f32_f16 v11, v11 // (op 6636) 
v_mul_f32 v20, v0, v10 // (op 6636) 
v_mul_f32 v21, v0, v11 // (op 6636) 


v_add_f32 v10, v18, v14 // (op 6637) 
v_add_f32 v11, v19, v15 // (op 6637) 
v_add_f32 v12, v20, v16 // (op 6637) 
v_add_f32 v13, v21, v17 // (op 6637) 


v_cvt_f16_f32 v14, v10 // (op 6638) 
v_cvt_f16_f32 v15, v11 // (op 6638) 
v_cvt_f16_f32 v16, v12 // (op 6638) 
v_cvt_f16_f32 v17, v13 // (op 6638) 


v_pack_b32_f16 v10, v14, v15 // (op 6621) 
v_pack_b32_f16 v11, v16, v17 // (op 6621) 

s_lshl_b32 s0, s4, 5 // (op 6621) 
s_mul_i32 s1, 3, s0 // (op 6621) 
s_add_u32 s1, s1, 96 // (op 6621) 

v_add_u32 v3, v9, s1 // (op 6621) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6621) Store value





v_add_u32 v3, 64, v8 // (op 6537) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6537) Load value

v_accvgpr_read v10, a32 // (op 6524) 
v_accvgpr_read v11, a33 // (op 6524) 
v_accvgpr_read v12, a34 // (op 6524) 
v_accvgpr_read v13, a35 // (op 6524) 
v_mul_f32 v14, v7, v10 // (op 6524) 
v_mul_f32 v15, v7, v11 // (op 6524) 
v_mul_f32 v16, v7, v12 // (op 6524) 
v_mul_f32 v17, v7, v13 // (op 6524) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6536) 
v_lshrrev_b32 v11, 16, v4 // (op 6536) 
v_cvt_f32_f16 v11, v11 // (op 6536) 
v_mul_f32 v18, v0, v10 // (op 6536) 
v_mul_f32 v19, v0, v11 // (op 6536) 
v_cvt_f32_f16 v10, v5 // (op 6536) 
v_lshrrev_b32 v11, 16, v5 // (op 6536) 
v_cvt_f32_f16 v11, v11 // (op 6536) 
v_mul_f32 v20, v0, v10 // (op 6536) 
v_mul_f32 v21, v0, v11 // (op 6536) 


v_add_f32 v10, v18, v14 // (op 6525) 
v_add_f32 v11, v19, v15 // (op 6525) 
v_add_f32 v12, v20, v16 // (op 6525) 
v_add_f32 v13, v21, v17 // (op 6525) 


v_cvt_f16_f32 v14, v10 // (op 6526) 
v_cvt_f16_f32 v15, v11 // (op 6526) 
v_cvt_f16_f32 v16, v12 // (op 6526) 
v_cvt_f16_f32 v17, v13 // (op 6526) 


v_pack_b32_f16 v10, v14, v15 // (op 6528) 
v_pack_b32_f16 v11, v16, v17 // (op 6528) 

v_add_u32 v3, 64, v9 // (op 6528) 

buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6528) Store value





s_lshl_b32 s0, s2, 5 // (op 6542) 
s_add_u32 s0, s0, 64 // (op 6542) 
v_add_u32 v3, v8, s0 // (op 6542) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6542) Load value

v_accvgpr_read v10, a36 // (op 6554) 
v_accvgpr_read v11, a37 // (op 6554) 
v_accvgpr_read v12, a38 // (op 6554) 
v_accvgpr_read v13, a39 // (op 6554) 
v_mul_f32 v14, v7, v10 // (op 6554) 
v_mul_f32 v15, v7, v11 // (op 6554) 
v_mul_f32 v16, v7, v12 // (op 6554) 
v_mul_f32 v17, v7, v13 // (op 6554) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6539) 
v_lshrrev_b32 v11, 16, v4 // (op 6539) 
v_cvt_f32_f16 v11, v11 // (op 6539) 
v_mul_f32 v18, v0, v10 // (op 6539) 
v_mul_f32 v19, v0, v11 // (op 6539) 
v_cvt_f32_f16 v10, v5 // (op 6539) 
v_lshrrev_b32 v11, 16, v5 // (op 6539) 
v_cvt_f32_f16 v11, v11 // (op 6539) 
v_mul_f32 v20, v0, v10 // (op 6539) 
v_mul_f32 v21, v0, v11 // (op 6539) 


v_add_f32 v10, v18, v14 // (op 6540) 
v_add_f32 v11, v19, v15 // (op 6540) 
v_add_f32 v12, v20, v16 // (op 6540) 
v_add_f32 v13, v21, v17 // (op 6540) 


v_cvt_f16_f32 v14, v10 // (op 6541) 
v_cvt_f16_f32 v15, v11 // (op 6541) 
v_cvt_f16_f32 v16, v12 // (op 6541) 
v_cvt_f16_f32 v17, v13 // (op 6541) 


v_pack_b32_f16 v10, v14, v15 // (op 6530) 
v_pack_b32_f16 v11, v16, v17 // (op 6530) 

s_lshl_b32 s0, s4, 5 // (op 6530) 
s_add_u32 s0, s0, 64 // (op 6530) 
v_add_u32 v3, v9, s0 // (op 6530) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6530) Store value





s_lshl_b32 s0, s2, 6 // (op 6547) 
s_add_u32 s0, s0, 64 // (op 6547) 
v_add_u32 v3, v8, s0 // (op 6547) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6547) Load value

v_accvgpr_read v10, a40 // (op 6556) 
v_accvgpr_read v11, a41 // (op 6556) 
v_accvgpr_read v12, a42 // (op 6556) 
v_accvgpr_read v13, a43 // (op 6556) 
v_mul_f32 v14, v7, v10 // (op 6556) 
v_mul_f32 v15, v7, v11 // (op 6556) 
v_mul_f32 v16, v7, v12 // (op 6556) 
v_mul_f32 v17, v7, v13 // (op 6556) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6544) 
v_lshrrev_b32 v11, 16, v4 // (op 6544) 
v_cvt_f32_f16 v11, v11 // (op 6544) 
v_mul_f32 v18, v0, v10 // (op 6544) 
v_mul_f32 v19, v0, v11 // (op 6544) 
v_cvt_f32_f16 v10, v5 // (op 6544) 
v_lshrrev_b32 v11, 16, v5 // (op 6544) 
v_cvt_f32_f16 v11, v11 // (op 6544) 
v_mul_f32 v20, v0, v10 // (op 6544) 
v_mul_f32 v21, v0, v11 // (op 6544) 


v_add_f32 v10, v18, v14 // (op 6545) 
v_add_f32 v11, v19, v15 // (op 6545) 
v_add_f32 v12, v20, v16 // (op 6545) 
v_add_f32 v13, v21, v17 // (op 6545) 


v_cvt_f16_f32 v14, v10 // (op 6546) 
v_cvt_f16_f32 v15, v11 // (op 6546) 
v_cvt_f16_f32 v16, v12 // (op 6546) 
v_cvt_f16_f32 v17, v13 // (op 6546) 


v_pack_b32_f16 v10, v14, v15 // (op 6532) 
v_pack_b32_f16 v11, v16, v17 // (op 6532) 

s_lshl_b32 s0, s4, 6 // (op 6532) 
s_add_u32 s0, s0, 64 // (op 6532) 
v_add_u32 v3, v9, s0 // (op 6532) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6532) Store value





s_lshl_b32 s0, s2, 5 // (op 6552) 
s_mul_i32 s1, 3, s0 // (op 6552) 
s_add_u32 s1, s1, 64 // (op 6552) 

v_add_u32 v3, v8, s1 // (op 6552) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6552) Load value

v_accvgpr_read v10, a44 // (op 6558) 
v_accvgpr_read v11, a45 // (op 6558) 
v_accvgpr_read v12, a46 // (op 6558) 
v_accvgpr_read v13, a47 // (op 6558) 
v_mul_f32 v14, v7, v10 // (op 6558) 
v_mul_f32 v15, v7, v11 // (op 6558) 
v_mul_f32 v16, v7, v12 // (op 6558) 
v_mul_f32 v17, v7, v13 // (op 6558) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6549) 
v_lshrrev_b32 v11, 16, v4 // (op 6549) 
v_cvt_f32_f16 v11, v11 // (op 6549) 
v_mul_f32 v18, v0, v10 // (op 6549) 
v_mul_f32 v19, v0, v11 // (op 6549) 
v_cvt_f32_f16 v10, v5 // (op 6549) 
v_lshrrev_b32 v11, 16, v5 // (op 6549) 
v_cvt_f32_f16 v11, v11 // (op 6549) 
v_mul_f32 v20, v0, v10 // (op 6549) 
v_mul_f32 v21, v0, v11 // (op 6549) 


v_add_f32 v10, v18, v14 // (op 6550) 
v_add_f32 v11, v19, v15 // (op 6550) 
v_add_f32 v12, v20, v16 // (op 6550) 
v_add_f32 v13, v21, v17 // (op 6550) 


v_cvt_f16_f32 v14, v10 // (op 6551) 
v_cvt_f16_f32 v15, v11 // (op 6551) 
v_cvt_f16_f32 v16, v12 // (op 6551) 
v_cvt_f16_f32 v17, v13 // (op 6551) 


v_pack_b32_f16 v10, v14, v15 // (op 6534) 
v_pack_b32_f16 v11, v16, v17 // (op 6534) 

s_lshl_b32 s0, s4, 5 // (op 6534) 
s_mul_i32 s1, 3, s0 // (op 6534) 
s_add_u32 s1, s1, 64 // (op 6534) 

v_add_u32 v3, v9, s1 // (op 6534) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6534) Store value





v_add_u32 v3, 32, v8 // (op 6450) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6450) Load value

v_accvgpr_read v10, a16 // (op 6437) 
v_accvgpr_read v11, a17 // (op 6437) 
v_accvgpr_read v12, a18 // (op 6437) 
v_accvgpr_read v13, a19 // (op 6437) 
v_mul_f32 v14, v7, v10 // (op 6437) 
v_mul_f32 v15, v7, v11 // (op 6437) 
v_mul_f32 v16, v7, v12 // (op 6437) 
v_mul_f32 v17, v7, v13 // (op 6437) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6449) 
v_lshrrev_b32 v11, 16, v4 // (op 6449) 
v_cvt_f32_f16 v11, v11 // (op 6449) 
v_mul_f32 v18, v0, v10 // (op 6449) 
v_mul_f32 v19, v0, v11 // (op 6449) 
v_cvt_f32_f16 v10, v5 // (op 6449) 
v_lshrrev_b32 v11, 16, v5 // (op 6449) 
v_cvt_f32_f16 v11, v11 // (op 6449) 
v_mul_f32 v20, v0, v10 // (op 6449) 
v_mul_f32 v21, v0, v11 // (op 6449) 


v_add_f32 v10, v18, v14 // (op 6438) 
v_add_f32 v11, v19, v15 // (op 6438) 
v_add_f32 v12, v20, v16 // (op 6438) 
v_add_f32 v13, v21, v17 // (op 6438) 


v_cvt_f16_f32 v14, v10 // (op 6439) 
v_cvt_f16_f32 v15, v11 // (op 6439) 
v_cvt_f16_f32 v16, v12 // (op 6439) 
v_cvt_f16_f32 v17, v13 // (op 6439) 


v_pack_b32_f16 v10, v14, v15 // (op 6441) 
v_pack_b32_f16 v11, v16, v17 // (op 6441) 

v_add_u32 v3, 32, v9 // (op 6441) 

buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6441) Store value





s_lshl_b32 s0, s2, 5 // (op 6455) 
s_add_u32 s0, s0, 32 // (op 6455) 
v_add_u32 v3, v8, s0 // (op 6455) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6455) Load value

v_accvgpr_read v10, a20 // (op 6467) 
v_accvgpr_read v11, a21 // (op 6467) 
v_accvgpr_read v12, a22 // (op 6467) 
v_accvgpr_read v13, a23 // (op 6467) 
v_mul_f32 v14, v7, v10 // (op 6467) 
v_mul_f32 v15, v7, v11 // (op 6467) 
v_mul_f32 v16, v7, v12 // (op 6467) 
v_mul_f32 v17, v7, v13 // (op 6467) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6452) 
v_lshrrev_b32 v11, 16, v4 // (op 6452) 
v_cvt_f32_f16 v11, v11 // (op 6452) 
v_mul_f32 v18, v0, v10 // (op 6452) 
v_mul_f32 v19, v0, v11 // (op 6452) 
v_cvt_f32_f16 v10, v5 // (op 6452) 
v_lshrrev_b32 v11, 16, v5 // (op 6452) 
v_cvt_f32_f16 v11, v11 // (op 6452) 
v_mul_f32 v20, v0, v10 // (op 6452) 
v_mul_f32 v21, v0, v11 // (op 6452) 


v_add_f32 v10, v18, v14 // (op 6453) 
v_add_f32 v11, v19, v15 // (op 6453) 
v_add_f32 v12, v20, v16 // (op 6453) 
v_add_f32 v13, v21, v17 // (op 6453) 


v_cvt_f16_f32 v14, v10 // (op 6454) 
v_cvt_f16_f32 v15, v11 // (op 6454) 
v_cvt_f16_f32 v16, v12 // (op 6454) 
v_cvt_f16_f32 v17, v13 // (op 6454) 


v_pack_b32_f16 v10, v14, v15 // (op 6443) 
v_pack_b32_f16 v11, v16, v17 // (op 6443) 

s_lshl_b32 s0, s4, 5 // (op 6443) 
s_add_u32 s0, s0, 32 // (op 6443) 
v_add_u32 v3, v9, s0 // (op 6443) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6443) Store value





s_lshl_b32 s0, s2, 6 // (op 6460) 
s_add_u32 s0, s0, 32 // (op 6460) 
v_add_u32 v3, v8, s0 // (op 6460) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6460) Load value

v_accvgpr_read v10, a24 // (op 6469) 
v_accvgpr_read v11, a25 // (op 6469) 
v_accvgpr_read v12, a26 // (op 6469) 
v_accvgpr_read v13, a27 // (op 6469) 
v_mul_f32 v14, v7, v10 // (op 6469) 
v_mul_f32 v15, v7, v11 // (op 6469) 
v_mul_f32 v16, v7, v12 // (op 6469) 
v_mul_f32 v17, v7, v13 // (op 6469) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6457) 
v_lshrrev_b32 v11, 16, v4 // (op 6457) 
v_cvt_f32_f16 v11, v11 // (op 6457) 
v_mul_f32 v18, v0, v10 // (op 6457) 
v_mul_f32 v19, v0, v11 // (op 6457) 
v_cvt_f32_f16 v10, v5 // (op 6457) 
v_lshrrev_b32 v11, 16, v5 // (op 6457) 
v_cvt_f32_f16 v11, v11 // (op 6457) 
v_mul_f32 v20, v0, v10 // (op 6457) 
v_mul_f32 v21, v0, v11 // (op 6457) 


v_add_f32 v10, v18, v14 // (op 6458) 
v_add_f32 v11, v19, v15 // (op 6458) 
v_add_f32 v12, v20, v16 // (op 6458) 
v_add_f32 v13, v21, v17 // (op 6458) 


v_cvt_f16_f32 v14, v10 // (op 6459) 
v_cvt_f16_f32 v15, v11 // (op 6459) 
v_cvt_f16_f32 v16, v12 // (op 6459) 
v_cvt_f16_f32 v17, v13 // (op 6459) 


v_pack_b32_f16 v10, v14, v15 // (op 6445) 
v_pack_b32_f16 v11, v16, v17 // (op 6445) 

s_lshl_b32 s0, s4, 6 // (op 6445) 
s_add_u32 s0, s0, 32 // (op 6445) 
v_add_u32 v3, v9, s0 // (op 6445) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6445) Store value





s_lshl_b32 s0, s2, 5 // (op 6465) 
s_mul_i32 s1, 3, s0 // (op 6465) 
s_add_u32 s1, s1, 32 // (op 6465) 

v_add_u32 v3, v8, s1 // (op 6465) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6465) Load value

v_accvgpr_read v10, a28 // (op 6471) 
v_accvgpr_read v11, a29 // (op 6471) 
v_accvgpr_read v12, a30 // (op 6471) 
v_accvgpr_read v13, a31 // (op 6471) 
v_mul_f32 v14, v7, v10 // (op 6471) 
v_mul_f32 v15, v7, v11 // (op 6471) 
v_mul_f32 v16, v7, v12 // (op 6471) 
v_mul_f32 v17, v7, v13 // (op 6471) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6462) 
v_lshrrev_b32 v11, 16, v4 // (op 6462) 
v_cvt_f32_f16 v11, v11 // (op 6462) 
v_mul_f32 v18, v0, v10 // (op 6462) 
v_mul_f32 v19, v0, v11 // (op 6462) 
v_cvt_f32_f16 v10, v5 // (op 6462) 
v_lshrrev_b32 v11, 16, v5 // (op 6462) 
v_cvt_f32_f16 v11, v11 // (op 6462) 
v_mul_f32 v20, v0, v10 // (op 6462) 
v_mul_f32 v21, v0, v11 // (op 6462) 


v_add_f32 v10, v18, v14 // (op 6463) 
v_add_f32 v11, v19, v15 // (op 6463) 
v_add_f32 v12, v20, v16 // (op 6463) 
v_add_f32 v13, v21, v17 // (op 6463) 


v_cvt_f16_f32 v14, v10 // (op 6464) 
v_cvt_f16_f32 v15, v11 // (op 6464) 
v_cvt_f16_f32 v16, v12 // (op 6464) 
v_cvt_f16_f32 v17, v13 // (op 6464) 


v_pack_b32_f16 v10, v14, v15 // (op 6447) 
v_pack_b32_f16 v11, v16, v17 // (op 6447) 

s_lshl_b32 s0, s4, 5 // (op 6447) 
s_mul_i32 s1, 3, s0 // (op 6447) 
s_add_u32 s1, s1, 32 // (op 6447) 

v_add_u32 v3, v9, s1 // (op 6447) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6447) Store value





v_add_u32 v3, 192, v8 // (op 6885) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6885) Load value

v_accvgpr_read v10, a96 // (op 6872) 
v_accvgpr_read v11, a97 // (op 6872) 
v_accvgpr_read v12, a98 // (op 6872) 
v_accvgpr_read v13, a99 // (op 6872) 
v_mul_f32 v14, v7, v10 // (op 6872) 
v_mul_f32 v15, v7, v11 // (op 6872) 
v_mul_f32 v16, v7, v12 // (op 6872) 
v_mul_f32 v17, v7, v13 // (op 6872) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6884) 
v_lshrrev_b32 v11, 16, v4 // (op 6884) 
v_cvt_f32_f16 v11, v11 // (op 6884) 
v_mul_f32 v18, v0, v10 // (op 6884) 
v_mul_f32 v19, v0, v11 // (op 6884) 
v_cvt_f32_f16 v10, v5 // (op 6884) 
v_lshrrev_b32 v11, 16, v5 // (op 6884) 
v_cvt_f32_f16 v11, v11 // (op 6884) 
v_mul_f32 v20, v0, v10 // (op 6884) 
v_mul_f32 v21, v0, v11 // (op 6884) 


v_add_f32 v10, v18, v14 // (op 6873) 
v_add_f32 v11, v19, v15 // (op 6873) 
v_add_f32 v12, v20, v16 // (op 6873) 
v_add_f32 v13, v21, v17 // (op 6873) 


v_cvt_f16_f32 v14, v10 // (op 6874) 
v_cvt_f16_f32 v15, v11 // (op 6874) 
v_cvt_f16_f32 v16, v12 // (op 6874) 
v_cvt_f16_f32 v17, v13 // (op 6874) 


v_pack_b32_f16 v10, v14, v15 // (op 6876) 
v_pack_b32_f16 v11, v16, v17 // (op 6876) 

v_add_u32 v3, 192, v9 // (op 6876) 

buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6876) Store value





s_lshl_b32 s0, s2, 5 // (op 6890) 
s_add_u32 s0, s0, 192 // (op 6890) 
v_add_u32 v3, v8, s0 // (op 6890) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6890) Load value

v_accvgpr_read v10, a100 // (op 6902) 
v_accvgpr_read v11, a101 // (op 6902) 
v_accvgpr_read v12, a102 // (op 6902) 
v_accvgpr_read v13, a103 // (op 6902) 
v_mul_f32 v14, v7, v10 // (op 6902) 
v_mul_f32 v15, v7, v11 // (op 6902) 
v_mul_f32 v16, v7, v12 // (op 6902) 
v_mul_f32 v17, v7, v13 // (op 6902) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6887) 
v_lshrrev_b32 v11, 16, v4 // (op 6887) 
v_cvt_f32_f16 v11, v11 // (op 6887) 
v_mul_f32 v18, v0, v10 // (op 6887) 
v_mul_f32 v19, v0, v11 // (op 6887) 
v_cvt_f32_f16 v10, v5 // (op 6887) 
v_lshrrev_b32 v11, 16, v5 // (op 6887) 
v_cvt_f32_f16 v11, v11 // (op 6887) 
v_mul_f32 v20, v0, v10 // (op 6887) 
v_mul_f32 v21, v0, v11 // (op 6887) 


v_add_f32 v10, v18, v14 // (op 6888) 
v_add_f32 v11, v19, v15 // (op 6888) 
v_add_f32 v12, v20, v16 // (op 6888) 
v_add_f32 v13, v21, v17 // (op 6888) 


v_cvt_f16_f32 v14, v10 // (op 6889) 
v_cvt_f16_f32 v15, v11 // (op 6889) 
v_cvt_f16_f32 v16, v12 // (op 6889) 
v_cvt_f16_f32 v17, v13 // (op 6889) 


v_pack_b32_f16 v10, v14, v15 // (op 6878) 
v_pack_b32_f16 v11, v16, v17 // (op 6878) 

s_lshl_b32 s0, s4, 5 // (op 6878) 
s_add_u32 s0, s0, 192 // (op 6878) 
v_add_u32 v3, v9, s0 // (op 6878) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6878) Store value





s_lshl_b32 s0, s2, 6 // (op 6895) 
s_add_u32 s0, s0, 192 // (op 6895) 
v_add_u32 v3, v8, s0 // (op 6895) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6895) Load value

v_accvgpr_read v10, a104 // (op 6904) 
v_accvgpr_read v11, a105 // (op 6904) 
v_accvgpr_read v12, a106 // (op 6904) 
v_accvgpr_read v13, a107 // (op 6904) 
v_mul_f32 v14, v7, v10 // (op 6904) 
v_mul_f32 v15, v7, v11 // (op 6904) 
v_mul_f32 v16, v7, v12 // (op 6904) 
v_mul_f32 v17, v7, v13 // (op 6904) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6892) 
v_lshrrev_b32 v11, 16, v4 // (op 6892) 
v_cvt_f32_f16 v11, v11 // (op 6892) 
v_mul_f32 v18, v0, v10 // (op 6892) 
v_mul_f32 v19, v0, v11 // (op 6892) 
v_cvt_f32_f16 v10, v5 // (op 6892) 
v_lshrrev_b32 v11, 16, v5 // (op 6892) 
v_cvt_f32_f16 v11, v11 // (op 6892) 
v_mul_f32 v20, v0, v10 // (op 6892) 
v_mul_f32 v21, v0, v11 // (op 6892) 


v_add_f32 v10, v18, v14 // (op 6893) 
v_add_f32 v11, v19, v15 // (op 6893) 
v_add_f32 v12, v20, v16 // (op 6893) 
v_add_f32 v13, v21, v17 // (op 6893) 


v_cvt_f16_f32 v14, v10 // (op 6894) 
v_cvt_f16_f32 v15, v11 // (op 6894) 
v_cvt_f16_f32 v16, v12 // (op 6894) 
v_cvt_f16_f32 v17, v13 // (op 6894) 


v_pack_b32_f16 v10, v14, v15 // (op 6880) 
v_pack_b32_f16 v11, v16, v17 // (op 6880) 

s_lshl_b32 s0, s4, 6 // (op 6880) 
s_add_u32 s0, s0, 192 // (op 6880) 
v_add_u32 v3, v9, s0 // (op 6880) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6880) Store value





s_lshl_b32 s0, s2, 5 // (op 6900) 
s_mul_i32 s1, 3, s0 // (op 6900) 
s_add_u32 s1, s1, 192 // (op 6900) 

v_add_u32 v3, v8, s1 // (op 6900) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6900) Load value

v_accvgpr_read v10, a108 // (op 6906) 
v_accvgpr_read v11, a109 // (op 6906) 
v_accvgpr_read v12, a110 // (op 6906) 
v_accvgpr_read v13, a111 // (op 6906) 
v_mul_f32 v14, v7, v10 // (op 6906) 
v_mul_f32 v15, v7, v11 // (op 6906) 
v_mul_f32 v16, v7, v12 // (op 6906) 
v_mul_f32 v17, v7, v13 // (op 6906) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6897) 
v_lshrrev_b32 v11, 16, v4 // (op 6897) 
v_cvt_f32_f16 v11, v11 // (op 6897) 
v_mul_f32 v18, v0, v10 // (op 6897) 
v_mul_f32 v19, v0, v11 // (op 6897) 
v_cvt_f32_f16 v10, v5 // (op 6897) 
v_lshrrev_b32 v11, 16, v5 // (op 6897) 
v_cvt_f32_f16 v11, v11 // (op 6897) 
v_mul_f32 v20, v0, v10 // (op 6897) 
v_mul_f32 v21, v0, v11 // (op 6897) 


v_add_f32 v10, v18, v14 // (op 6898) 
v_add_f32 v11, v19, v15 // (op 6898) 
v_add_f32 v12, v20, v16 // (op 6898) 
v_add_f32 v13, v21, v17 // (op 6898) 


v_cvt_f16_f32 v14, v10 // (op 6899) 
v_cvt_f16_f32 v15, v11 // (op 6899) 
v_cvt_f16_f32 v16, v12 // (op 6899) 
v_cvt_f16_f32 v17, v13 // (op 6899) 


v_pack_b32_f16 v10, v14, v15 // (op 6882) 
v_pack_b32_f16 v11, v16, v17 // (op 6882) 

s_lshl_b32 s0, s4, 5 // (op 6882) 
s_mul_i32 s1, 3, s0 // (op 6882) 
s_add_u32 s1, s1, 192 // (op 6882) 

v_add_u32 v3, v9, s1 // (op 6882) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6882) Store value





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





s_lshl_b32 s0, s2, 5 // (op 6358) 
v_add_u32 v3, v8, s0 // (op 6358) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6358) Load value

v_accvgpr_read v10, a4 // (op 6354) 
v_accvgpr_read v11, a5 // (op 6354) 
v_accvgpr_read v12, a6 // (op 6354) 
v_accvgpr_read v13, a7 // (op 6354) 
v_mul_f32 v14, v7, v10 // (op 6354) 
v_mul_f32 v15, v7, v11 // (op 6354) 
v_mul_f32 v16, v7, v12 // (op 6354) 
v_mul_f32 v17, v7, v13 // (op 6354) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6359) 
v_lshrrev_b32 v11, 16, v4 // (op 6359) 
v_cvt_f32_f16 v11, v11 // (op 6359) 
v_mul_f32 v18, v0, v10 // (op 6359) 
v_mul_f32 v19, v0, v11 // (op 6359) 
v_cvt_f32_f16 v10, v5 // (op 6359) 
v_lshrrev_b32 v11, 16, v5 // (op 6359) 
v_cvt_f32_f16 v11, v11 // (op 6359) 
v_mul_f32 v20, v0, v10 // (op 6359) 
v_mul_f32 v21, v0, v11 // (op 6359) 


v_add_f32 v10, v18, v14 // (op 6355) 
v_add_f32 v11, v19, v15 // (op 6355) 
v_add_f32 v12, v20, v16 // (op 6355) 
v_add_f32 v13, v21, v17 // (op 6355) 


v_cvt_f16_f32 v14, v10 // (op 6356) 
v_cvt_f16_f32 v15, v11 // (op 6356) 
v_cvt_f16_f32 v16, v12 // (op 6356) 
v_cvt_f16_f32 v17, v13 // (op 6356) 


v_pack_b32_f16 v10, v14, v15 // (op 6357) 
v_pack_b32_f16 v11, v16, v17 // (op 6357) 

s_lshl_b32 s0, s4, 5 // (op 6357) 
v_add_u32 v3, v9, s0 // (op 6357) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6357) Store value





s_lshl_b32 s0, s2, 6 // (op 6372) 
v_add_u32 v3, v8, s0 // (op 6372) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6372) Load value

v_accvgpr_read v10, a8 // (op 6368) 
v_accvgpr_read v11, a9 // (op 6368) 
v_accvgpr_read v12, a10 // (op 6368) 
v_accvgpr_read v13, a11 // (op 6368) 
v_mul_f32 v14, v7, v10 // (op 6368) 
v_mul_f32 v15, v7, v11 // (op 6368) 
v_mul_f32 v16, v7, v12 // (op 6368) 
v_mul_f32 v17, v7, v13 // (op 6368) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6373) 
v_lshrrev_b32 v11, 16, v4 // (op 6373) 
v_cvt_f32_f16 v11, v11 // (op 6373) 
v_mul_f32 v18, v0, v10 // (op 6373) 
v_mul_f32 v19, v0, v11 // (op 6373) 
v_cvt_f32_f16 v10, v5 // (op 6373) 
v_lshrrev_b32 v11, 16, v5 // (op 6373) 
v_cvt_f32_f16 v11, v11 // (op 6373) 
v_mul_f32 v20, v0, v10 // (op 6373) 
v_mul_f32 v21, v0, v11 // (op 6373) 


v_add_f32 v10, v18, v14 // (op 6369) 
v_add_f32 v11, v19, v15 // (op 6369) 
v_add_f32 v12, v20, v16 // (op 6369) 
v_add_f32 v13, v21, v17 // (op 6369) 


v_cvt_f16_f32 v14, v10 // (op 6370) 
v_cvt_f16_f32 v15, v11 // (op 6370) 
v_cvt_f16_f32 v16, v12 // (op 6370) 
v_cvt_f16_f32 v17, v13 // (op 6370) 


v_pack_b32_f16 v10, v14, v15 // (op 6371) 
v_pack_b32_f16 v11, v16, v17 // (op 6371) 

s_lshl_b32 s0, s4, 6 // (op 6371) 
v_add_u32 v3, v9, s0 // (op 6371) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6371) Store value





s_lshl_b32 s0, s2, 5 // (op 6386) 
s_mul_i32 s1, 3, s0 // (op 6386) 

v_add_u32 v3, v8, s1 // (op 6386) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6386) Load value

v_accvgpr_read v10, a12 // (op 6382) 
v_accvgpr_read v11, a13 // (op 6382) 
v_accvgpr_read v12, a14 // (op 6382) 
v_accvgpr_read v13, a15 // (op 6382) 
v_mul_f32 v14, v7, v10 // (op 6382) 
v_mul_f32 v15, v7, v11 // (op 6382) 
v_mul_f32 v16, v7, v12 // (op 6382) 
v_mul_f32 v17, v7, v13 // (op 6382) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6387) 
v_lshrrev_b32 v11, 16, v4 // (op 6387) 
v_cvt_f32_f16 v11, v11 // (op 6387) 
v_mul_f32 v18, v0, v10 // (op 6387) 
v_mul_f32 v19, v0, v11 // (op 6387) 
v_cvt_f32_f16 v10, v5 // (op 6387) 
v_lshrrev_b32 v11, 16, v5 // (op 6387) 
v_cvt_f32_f16 v11, v11 // (op 6387) 
v_mul_f32 v20, v0, v10 // (op 6387) 
v_mul_f32 v21, v0, v11 // (op 6387) 


v_add_f32 v10, v18, v14 // (op 6383) 
v_add_f32 v11, v19, v15 // (op 6383) 
v_add_f32 v12, v20, v16 // (op 6383) 
v_add_f32 v13, v21, v17 // (op 6383) 


v_cvt_f16_f32 v14, v10 // (op 6384) 
v_cvt_f16_f32 v15, v11 // (op 6384) 
v_cvt_f16_f32 v16, v12 // (op 6384) 
v_cvt_f16_f32 v17, v13 // (op 6384) 


v_pack_b32_f16 v10, v14, v15 // (op 6385) 
v_pack_b32_f16 v11, v16, v17 // (op 6385) 

s_lshl_b32 s0, s4, 5 // (op 6385) 
s_mul_i32 s1, 3, s0 // (op 6385) 

v_add_u32 v3, v9, s1 // (op 6385) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6385) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__ConditionalBottom_0_beta_116:








































s_endpgm  // End of RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM_
.LRR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__end:
.size RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM_, .LRR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM__end-RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 296
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 73728 // lds bytes
  .amdhsa_accum_offset 168
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
    DataType: F6
    DataTypeA: F6
    DataTypeB: F6
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
    macrotile: [256, 128, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [8, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 73728
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 72
    .vgpr_count: 166
    .agpr_count: 128
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {is-null: true}, {is-null: true}]
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
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 28
        .expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}}
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
        .expression: {type: MagicMultiple, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_24, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 160, size: 4}, rhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}}}
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
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: FP6, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP6, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP6, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP6, pointerType: PointerGlobal}
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
        .expression: {type: MagicShiftAndSign, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_24, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 160, size: 4}, rhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_27
        .size: 4
        .offset: 156
        .expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: Subtract_24
        .size: 4
        .offset: 160
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
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

