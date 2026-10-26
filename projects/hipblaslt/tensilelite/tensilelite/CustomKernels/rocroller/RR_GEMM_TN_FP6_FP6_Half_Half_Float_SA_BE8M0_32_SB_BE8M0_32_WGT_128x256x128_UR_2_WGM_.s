// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT



.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM_,@function
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 13676) 
s_mul_hi_i32 s1, s16, s10 // (op 13817) 
v_lshrrev_b32 v0, 7, v1 // (op 13738) 
v_and_b32 v3, 15, v1 // (op 13738) 
v_lshl_add_u32 v4, v0, 6, v3 // (op 13738) 


v_and_b32 v0, 63, v1 // (op 13738) 
v_lshrrev_b32 v3, 4, v0 // (op 13738) 

v_lshlrev_b32 v0, 5, v3 // (op 13738) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 13738) 


v_mul_lo_u32 v0, v3, 6 // (op 13738) 

v_lshrrev_b32 v3, 3, v0 // (op 13738) 

v_lshrrev_b32 v0, 6, v1 // (op 13796) 
v_and_b32 v4, 1, v0 // (op 13796) 

v_and_b32 v0, 15, v1 // (op 13796) 
v_lshl_add_u32 v5, v4, 7, v0 // (op 13796) 


v_and_b32 v0, 63, v1 // (op 13796) 
v_lshrrev_b32 v4, 4, v0 // (op 13796) 

v_lshlrev_b32 v0, 5, v4 // (op 13796) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 13796) 


v_mul_lo_u32 v0, v4, 6 // (op 13796) 

v_lshrrev_b32 v4, 3, v0 // (op 13796) 

v_lshrrev_b32 v0, 7, v1 // (op 13848) 
v_and_b32 v5, 15, v1 // (op 13848) 
v_lshl_add_u32 v6, v0, 6, v5 // (op 13848) 


v_and_b32 v0, 63, v1 // (op 13848) 
v_lshrrev_b32 v5, 4, v0 // (op 13848) 

v_lshlrev_b32 v0, 5, v5 // (op 13848) 

v_lshl_add_u32 v5, v6, 7, v0 // (op 13848) 


v_mul_lo_u32 v0, v5, 6 // (op 13848) 

v_lshrrev_b32 v5, 3, v0 // (op 13848) 

v_lshrrev_b32 v0, 6, v1 // (op 13877) 
v_and_b32 v6, 1, v0 // (op 13877) 

v_and_b32 v0, 15, v1 // (op 13877) 
v_lshl_add_u32 v7, v6, 7, v0 // (op 13877) 


v_and_b32 v0, 63, v1 // (op 13877) 
v_lshrrev_b32 v6, 4, v0 // (op 13877) 

v_lshlrev_b32 v0, 5, v6 // (op 13877) 

v_lshl_add_u32 v6, v7, 7, v0 // (op 13877) 


v_mul_lo_u32 v0, v6, 6 // (op 13877) 

v_lshrrev_b32 v6, 3, v0 // (op 13877) 

v_lshrrev_b32 v0, 7, v1 // (op 13676) 
s_add_i32 s18, s0, s16 // (op 13676) 

v_and_b32 v7, 63, v1 // (op 13676) 
s_ashr_i32 s0, s18, 31 // (op 13676) 
v_lshrrev_b32 v8, 4, v7 // (op 13676) 
s_cmp_eq_i32 s10, 0 // (op 13676) 
s_mov_b32 s19, scc // (op 13676) 
v_lshlrev_b32 v9, 2, v8 // (op 13676) 

s_cmp_eq_i32 s19, 0 // (op 13676) 
s_cselect_b32 s49, 0, -1 // (op 13676) 

v_lshl_add_u32 v8, v0, 6, v9 // (op 13676) 


s_and_b32 s19, s13, 31 // (op 13676) 
v_lshrrev_b32 v0, 2, v7 // (op 13676) 

v_and_b32 v7, 3, v0 // (op 13676) 

v_lshrrev_b32 v0, 7, v1 // (op 13817) 
v_and_b32 v9, 63, v1 // (op 13817) 
s_lshl_b32 s50, 1, s19 // (op 13676) 
v_and_b32 v10, 3, v1 // (op 13676) 
s_add_i32 s51, s50, s49 // (op 13676) 


s_and_b32 s49, s0, s51 // (op 13676) 


s_add_i32 s0, s1, s16 // (op 13817) 

s_ashr_i32 s1, s0, 31 // (op 13817) 
v_lshl_add_u32 v11, v7, 2, v10 // (op 13676) 


s_add_i32 s50, s18, s49 // (op 13676) 


v_lshrrev_b32 v7, 6, v1 // (op 13676) 
s_ashr_i32 s18, s50, s19 // (op 13676) 


v_and_b32 v10, 1, v7 // (op 13676) 

s_lshl_b32 s19, s13, 24 // (op 13676) 
v_lshl_add_u32 v7, v10, 7, v11 // (op 13676) 


s_ashr_i32 s49, s19, 31 // (op 13676) 

s_xor_b32 s19, s18, s49 // (op 13676) 

s_sub_i32 s18, s19, s49 // (op 13676) 


s_mul_hi_i32 s19, s18, s14 // (op 13676) 
s_add_i32 s49, s19, s18 // (op 13676) 

s_ashr_i32 s19, s49, 31 // (op 13676) 
s_cmp_eq_i32 s14, 0 // (op 13676) 
s_mov_b32 s50, scc // (op 13676) 
s_cmp_eq_i32 s50, 0 // (op 13676) 
s_cselect_b32 s51, 0, -1 // (op 13676) 

s_and_b32 s50, s15, 31 // (op 13676) 
v_lshrrev_b32 v10, 4, v9 // (op 13817) 
s_cmp_eq_i32 s10, 0 // (op 13817) 
s_mov_b32 s52, scc // (op 13817) 
v_lshlrev_b32 v11, 2, v10 // (op 13817) 

s_cmp_eq_i32 s52, 0 // (op 13817) 
s_cselect_b32 s53, 0, -1 // (op 13817) 

s_lshl_b32 s52, 1, s50 // (op 13676) 
v_lshl_add_u32 v10, v0, 6, v11 // (op 13817) 


v_lshrrev_b32 v0, 2, v9 // (op 13817) 

s_add_i32 s54, s52, s51 // (op 13676) 


s_and_b32 s51, s19, s54 // (op 13676) 


s_add_i32 s19, s49, s51 // (op 13676) 


s_ashr_i32 s49, s19, s50 // (op 13676) 


s_lshl_b32 s19, s15, 24 // (op 13676) 
s_ashr_i32 s50, s19, 31 // (op 13676) 

s_xor_b32 s19, s49, s50 // (op 13676) 

s_sub_i32 s49, s19, s50 // (op 13676) 


s_mul_i32 s19, s18, s11 // (op 13676) 
s_sub_u32 s50, s16, s19 // (op 13676) 

s_mul_i32 s19, s49, s11 // (op 13676) 
s_add_u32 s19, s19, s50 // (op 13676) 

s_mul_hi_i32 s50, s16, s9 // (op 13676) 
s_add_i32 s51, s50, s16 // (op 13676) 

s_ashr_i32 s50, s51, 31 // (op 13676) 
s_cmp_eq_i32 s9, 0 // (op 13676) 
s_mov_b32 s52, scc // (op 13676) 
s_cmp_eq_i32 s52, 0 // (op 13676) 
s_cselect_b32 s54, 0, -1 // (op 13676) 

s_and_b32 s52, s13, 31 // (op 13817) 
v_and_b32 v9, 3, v0 // (op 13817) 

s_lshl_b32 s55, 1, s52 // (op 13817) 
v_and_b32 v0, 3, v1 // (op 13817) 
s_add_i32 s56, s55, s53 // (op 13817) 


s_and_b32 s53, s1, s56 // (op 13817) 


v_lshl_add_u32 v11, v9, 2, v0 // (op 13817) 


s_add_i32 s1, s0, s53 // (op 13817) 


v_lshrrev_b32 v0, 6, v1 // (op 13817) 
s_ashr_i32 s0, s1, s52 // (op 13817) 


v_and_b32 v9, 1, v0 // (op 13817) 

s_lshl_b32 s1, s13, 24 // (op 13817) 
v_lshl_add_u32 v0, v9, 7, v11 // (op 13817) 


s_ashr_i32 s52, s1, 31 // (op 13817) 

s_xor_b32 s1, s0, s52 // (op 13817) 

s_sub_i32 s0, s1, s52 // (op 13817) 


s_mul_hi_i32 s1, s0, s14 // (op 13817) 
s_add_i32 s52, s1, s0 // (op 13817) 

s_ashr_i32 s1, s52, 31 // (op 13817) 
s_cmp_eq_i32 s14, 0 // (op 13817) 
s_mov_b32 s53, scc // (op 13817) 
s_cmp_eq_i32 s53, 0 // (op 13817) 
s_cselect_b32 s55, 0, -1 // (op 13817) 

s_and_b32 s53, s15, 31 // (op 13817) 
s_lshl_b32 s56, 1, s53 // (op 13817) 
s_add_i32 s57, s56, s55 // (op 13817) 


s_and_b32 s55, s1, s57 // (op 13817) 


s_add_i32 s1, s52, s55 // (op 13817) 


s_ashr_i32 s52, s1, s53 // (op 13817) 


s_lshl_b32 s1, s15, 24 // (op 13817) 
s_ashr_i32 s53, s1, 31 // (op 13817) 

s_xor_b32 s1, s52, s53 // (op 13817) 

s_sub_i32 s52, s1, s53 // (op 13817) 


s_mul_i32 s1, s0, s11 // (op 13817) 
s_sub_u32 s53, s16, s1 // (op 13817) 

s_mul_i32 s1, s52, s11 // (op 13817) 
s_add_u32 s1, s1, s53 // (op 13817) 

s_mul_hi_i32 s53, s16, s9 // (op 13817) 
s_add_i32 s55, s53, s16 // (op 13817) 

s_ashr_i32 s53, s55, 31 // (op 13817) 
s_cmp_eq_i32 s9, 0 // (op 13817) 
s_mov_b32 s56, scc // (op 13817) 
s_cmp_eq_i32 s56, 0 // (op 13817) 
s_cselect_b32 s57, 0, -1 // (op 13817) 


s_waitcnt lgkmcnt(0)
s_and_b32 s56, s45, 31 // (op 13676) 
s_lshl_b32 s58, 1, s56 // (op 13676) 
s_add_i32 s59, s58, s54 // (op 13676) 


s_and_b32 s54, s50, s59 // (op 13676) 


s_add_i32 s50, s51, s54 // (op 13676) 


s_ashr_i32 s51, s50, s56 // (op 13676) 


s_lshl_b32 s50, s45, 24 // (op 13676) 
s_ashr_i32 s54, s50, 31 // (op 13676) 

s_xor_b32 s50, s51, s54 // (op 13676) 

s_sub_i32 s51, s50, s54 // (op 13676) 


s_and_b32 s50, s45, 31 // (op 13817) 
s_lshl_b32 s54, 1, s50 // (op 13817) 
s_add_i32 s56, s54, s57 // (op 13817) 


s_and_b32 s54, s53, s56 // (op 13817) 


s_add_i32 s53, s55, s54 // (op 13817) 


s_ashr_i32 s54, s53, s50 // (op 13817) 


s_lshl_b32 s50, s45, 24 // (op 13817) 
s_ashr_i32 s53, s50, 31 // (op 13817) 

s_xor_b32 s50, s54, s53 // (op 13817) 

s_sub_i32 s54, s50, s53 // (op 13817) 


s_add_u32 s56, s6, 256 // (op 13676) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 13676) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 13676) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 13676) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 8 // (op 13676) 

s_mul_i32 s50, s11, s56 // (op 13676) 
s_mul_i32 s53, s8, s50 // (op 13676) 

s_mul_i32 s50, s51, s53 // (op 13676) 

s_sub_u32 s53, s16, s50 // (op 13676) 

s_mov_b32 s50, s53 // (op 13676) 
s_mul_hi_i32 s55, s50, s12 // (op 13676) 
s_add_i32 s58, s55, s50 // (op 13676) 


s_ashr_i32 s50, s58, 31 // (op 13676) 
s_cmp_eq_i32 s12, 0 // (op 13676) 
s_mov_b32 s55, scc // (op 13676) 
s_cmp_eq_i32 s55, 0 // (op 13676) 
s_cselect_b32 s59, 0, -1 // (op 13676) 

s_and_b32 s55, s44, 31 // (op 13676) 
s_lshl_b32 s60, 1, s55 // (op 13676) 
s_add_i32 s61, s60, s59 // (op 13676) 


s_and_b32 s59, s50, s61 // (op 13676) 


s_add_i32 s50, s58, s59 // (op 13676) 


s_ashr_i32 s58, s50, s55 // (op 13676) 


s_lshl_b32 s50, s44, 24 // (op 13676) 
s_ashr_i32 s55, s50, 31 // (op 13676) 

s_xor_b32 s50, s58, s55 // (op 13676) 

s_sub_i32 s58, s50, s55 // (op 13676) 


s_mul_i32 s50, s46, s11 // (op 13676) 
s_add_u32 s60, s20, 128 // (op 13676) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 13676) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 13676) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 13676) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 7 // (op 13676) 

s_sub_i32 s55, s60, s50 // (op 13676) 

s_mul_i32 s59, s58, s55 // (op 13676) 

s_sub_u32 s55, s53, s59 // (op 13676) 


s_add_u32 s53, s55, s50 // (op 13676) 


s_cmp_eq_i32 s51, 0 // (op 13676) 
s_mov_b32 s50, scc // (op 13676) 

s_cmp_eq_i32 s50, 0 // (op 13676) 
s_cselect_b32 s51, s53, s19 // (op 13676) 


s_mul_i32 s19, s49, s56 // (op 13676) 


v_lshl_add_u32 v9, s51, 7, v8 // (op 13676) 


s_sub_i32 s49, s18, s19 // (op 13676) 


s_cmp_eq_i32 s50, 0 // (op 13676) 
s_cselect_b32 s51, s58, s49 // (op 13676) 



v_lshl_add_u32 v8, s51, 8, v7 // (op 13676) 


v_mul_lo_u32 v7, v8, s2 // (op 13676) 

v_add_lshl_u32 v8, v9, v7, 1 // (op 13676) 


s_add_u32 s18, s6, 256 // (op 13817) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 13817) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 13817) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 13817) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 8 // (op 13817) 

s_mul_i32 s53, s11, s18 // (op 13817) 
s_mul_i32 s49, s8, s53 // (op 13817) 

s_mul_i32 s53, s54, s49 // (op 13817) 

s_sub_u32 s49, s16, s53 // (op 13817) 

s_mov_b32 s53, s49 // (op 13817) 
s_mul_hi_i32 s50, s53, s12 // (op 13817) 
s_add_i32 s51, s50, s53 // (op 13817) 


s_ashr_i32 s50, s51, 31 // (op 13817) 
s_cmp_eq_i32 s12, 0 // (op 13817) 
s_mov_b32 s53, scc // (op 13817) 
s_cmp_eq_i32 s53, 0 // (op 13817) 
s_cselect_b32 s55, 0, -1 // (op 13817) 

s_and_b32 s53, s44, 31 // (op 13817) 
s_lshl_b32 s56, 1, s53 // (op 13817) 
s_add_i32 s57, s56, s55 // (op 13817) 


s_and_b32 s55, s50, s57 // (op 13817) 


s_add_i32 s50, s51, s55 // (op 13817) 


s_ashr_i32 s51, s50, s53 // (op 13817) 


s_lshl_b32 s50, s44, 24 // (op 13817) 
s_ashr_i32 s53, s50, 31 // (op 13817) 

s_xor_b32 s50, s51, s53 // (op 13817) 

s_sub_i32 s51, s50, s53 // (op 13817) 


s_mul_i32 s50, s46, s11 // (op 13817) 
s_add_u32 s56, s20, 128 // (op 13817) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 13817) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 13817) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 13817) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 7 // (op 13817) 

s_sub_i32 s53, s56, s50 // (op 13817) 

s_mul_i32 s55, s51, s53 // (op 13817) 

s_sub_u32 s53, s49, s55 // (op 13817) 


s_add_u32 s49, s53, s50 // (op 13817) 


s_cmp_eq_i32 s54, 0 // (op 13817) 
s_mov_b32 s50, scc // (op 13817) 

s_cmp_eq_i32 s50, 0 // (op 13817) 
s_cselect_b32 s53, s49, s1 // (op 13817) 


s_mul_i32 s1, s52, s18 // (op 13817) 


v_lshl_add_u32 v7, s53, 7, v10 // (op 13817) 


s_sub_i32 s49, s0, s1 // (op 13817) 


s_cmp_eq_i32 s50, 0 // (op 13817) 
s_cselect_b32 s0, s51, s49 // (op 13817) 



v_lshl_add_u32 v9, s0, 8, v0 // (op 13817) 


v_mul_lo_u32 v0, v9, s4 // (op 13817) 

v_add_lshl_u32 v9, v7, v0, 1 // (op 13817) 


s_mul_i32 s1, s2, s7 // (op 13678) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 13678) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 13678) least significant
s_mul_hi_u32 s19, s2, s6 // (op 13678) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13678) most significant: sum
s_add_u32 s1, s1, s19 // (op 13678) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 13678) 

s_mul_i32 s1, s4, s7 // (op 13819) most significant: low of low * high
s_mul_i32 s49, s5, s6 // (op 13819) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 13819) least significant
s_mul_hi_u32 s50, s4, s6 // (op 13819) most significant: high of low * low
s_add_u32 s1, s1, s49 // (op 13819) most significant: sum
s_add_u32 s1, s1, s50 // (op 13819) most significant: sum


s_lshl_b64 s[50:51], s[0:1], 1 // (op 13819) 

s_mov_b32 s52, s22 // (op 13678) reg expression
s_mov_b32 s53, s23 // (op 13678) reg expression
s_mov_b32 s54, s18 // (op 13678) reg expression
s_mov_b32 s55, 131072 // (op 13678) call()

s_mov_b32 s56, s24 // (op 13819) reg expression

s_mov_b32 s57, s25 // (op 13819) reg expression
s_mov_b32 s58, s50 // (op 13819) reg expression
s_mov_b32 s59, 131072 // (op 13819) call()


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
v_accvgpr_write a16, 0.00000 // (op 689) call()
v_accvgpr_write a17, 0.00000 // (op 689) call()
v_accvgpr_write a18, 0.00000 // (op 689) call()
v_accvgpr_write a19, 0.00000 // (op 689) call()
v_accvgpr_write a20, 0.00000 // (op 825) call()
v_accvgpr_write a21, 0.00000 // (op 825) call()
v_accvgpr_write a22, 0.00000 // (op 825) call()
v_accvgpr_write a23, 0.00000 // (op 825) call()
v_accvgpr_write a24, 0.00000 // (op 961) call()
v_accvgpr_write a25, 0.00000 // (op 961) call()
v_accvgpr_write a26, 0.00000 // (op 961) call()
v_accvgpr_write a27, 0.00000 // (op 961) call()
v_accvgpr_write a28, 0.00000 // (op 1097) call()
v_accvgpr_write a29, 0.00000 // (op 1097) call()
v_accvgpr_write a30, 0.00000 // (op 1097) call()
v_accvgpr_write a31, 0.00000 // (op 1097) call()
v_accvgpr_write a32, 0.00000 // (op 1409) call()
v_accvgpr_write a33, 0.00000 // (op 1409) call()
v_accvgpr_write a34, 0.00000 // (op 1409) call()
v_accvgpr_write a35, 0.00000 // (op 1409) call()
v_accvgpr_write a36, 0.00000 // (op 1930) call()
v_accvgpr_write a37, 0.00000 // (op 1930) call()
v_accvgpr_write a38, 0.00000 // (op 1930) call()
v_accvgpr_write a39, 0.00000 // (op 1930) call()
v_accvgpr_write a40, 0.00000 // (op 1931) call()
v_accvgpr_write a41, 0.00000 // (op 1931) call()
v_accvgpr_write a42, 0.00000 // (op 1931) call()
v_accvgpr_write a43, 0.00000 // (op 1931) call()
v_accvgpr_write a44, 0.00000 // (op 1932) call()
v_accvgpr_write a45, 0.00000 // (op 1932) call()
v_accvgpr_write a46, 0.00000 // (op 1932) call()
v_accvgpr_write a47, 0.00000 // (op 1932) call()
v_accvgpr_write a48, 0.00000 // (op 1933) call()
v_accvgpr_write a49, 0.00000 // (op 1933) call()
v_accvgpr_write a50, 0.00000 // (op 1933) call()
v_accvgpr_write a51, 0.00000 // (op 1933) call()
v_accvgpr_write a52, 0.00000 // (op 1934) call()
v_accvgpr_write a53, 0.00000 // (op 1934) call()
v_accvgpr_write a54, 0.00000 // (op 1934) call()
v_accvgpr_write a55, 0.00000 // (op 1934) call()
v_accvgpr_write a56, 0.00000 // (op 1935) call()
v_accvgpr_write a57, 0.00000 // (op 1935) call()
v_accvgpr_write a58, 0.00000 // (op 1935) call()
v_accvgpr_write a59, 0.00000 // (op 1935) call()
v_accvgpr_write a60, 0.00000 // (op 1936) call()
v_accvgpr_write a61, 0.00000 // (op 1936) call()
v_accvgpr_write a62, 0.00000 // (op 1936) call()
v_accvgpr_write a63, 0.00000 // (op 1936) call()
v_accvgpr_write a64, 0.00000 // (op 2627) call()
v_accvgpr_write a65, 0.00000 // (op 2627) call()
v_accvgpr_write a66, 0.00000 // (op 2627) call()
v_accvgpr_write a67, 0.00000 // (op 2627) call()
v_accvgpr_write a68, 0.00000 // (op 3148) call()
v_accvgpr_write a69, 0.00000 // (op 3148) call()
v_accvgpr_write a70, 0.00000 // (op 3148) call()
v_accvgpr_write a71, 0.00000 // (op 3148) call()
v_accvgpr_write a72, 0.00000 // (op 3149) call()
v_accvgpr_write a73, 0.00000 // (op 3149) call()
v_accvgpr_write a74, 0.00000 // (op 3149) call()
v_accvgpr_write a75, 0.00000 // (op 3149) call()
v_accvgpr_write a76, 0.00000 // (op 3150) call()
v_accvgpr_write a77, 0.00000 // (op 3150) call()
v_accvgpr_write a78, 0.00000 // (op 3150) call()
v_accvgpr_write a79, 0.00000 // (op 3150) call()
v_accvgpr_write a80, 0.00000 // (op 3151) call()
v_accvgpr_write a81, 0.00000 // (op 3151) call()
v_accvgpr_write a82, 0.00000 // (op 3151) call()
v_accvgpr_write a83, 0.00000 // (op 3151) call()
v_accvgpr_write a84, 0.00000 // (op 3152) call()
v_accvgpr_write a85, 0.00000 // (op 3152) call()
v_accvgpr_write a86, 0.00000 // (op 3152) call()
v_accvgpr_write a87, 0.00000 // (op 3152) call()
v_accvgpr_write a88, 0.00000 // (op 3153) call()
v_accvgpr_write a89, 0.00000 // (op 3153) call()
v_accvgpr_write a90, 0.00000 // (op 3153) call()
v_accvgpr_write a91, 0.00000 // (op 3153) call()
v_accvgpr_write a92, 0.00000 // (op 3154) call()
v_accvgpr_write a93, 0.00000 // (op 3154) call()
v_accvgpr_write a94, 0.00000 // (op 3154) call()
v_accvgpr_write a95, 0.00000 // (op 3154) call()
v_accvgpr_write a96, 0.00000 // (op 3845) call()
v_accvgpr_write a97, 0.00000 // (op 3845) call()
v_accvgpr_write a98, 0.00000 // (op 3845) call()
v_accvgpr_write a99, 0.00000 // (op 3845) call()
v_accvgpr_write a100, 0.00000 // (op 4366) call()
v_accvgpr_write a101, 0.00000 // (op 4366) call()
v_accvgpr_write a102, 0.00000 // (op 4366) call()
v_accvgpr_write a103, 0.00000 // (op 4366) call()
v_accvgpr_write a104, 0.00000 // (op 4367) call()
v_accvgpr_write a105, 0.00000 // (op 4367) call()
v_accvgpr_write a106, 0.00000 // (op 4367) call()
v_accvgpr_write a107, 0.00000 // (op 4367) call()
v_accvgpr_write a108, 0.00000 // (op 4368) call()
v_accvgpr_write a109, 0.00000 // (op 4368) call()
v_accvgpr_write a110, 0.00000 // (op 4368) call()
v_accvgpr_write a111, 0.00000 // (op 4368) call()
v_accvgpr_write a112, 0.00000 // (op 4369) call()
v_accvgpr_write a113, 0.00000 // (op 4369) call()
v_accvgpr_write a114, 0.00000 // (op 4369) call()
v_accvgpr_write a115, 0.00000 // (op 4369) call()
v_accvgpr_write a116, 0.00000 // (op 4370) call()
v_accvgpr_write a117, 0.00000 // (op 4370) call()
v_accvgpr_write a118, 0.00000 // (op 4370) call()
v_accvgpr_write a119, 0.00000 // (op 4370) call()
v_accvgpr_write a120, 0.00000 // (op 4371) call()
v_accvgpr_write a121, 0.00000 // (op 4371) call()
v_accvgpr_write a122, 0.00000 // (op 4371) call()
v_accvgpr_write a123, 0.00000 // (op 4371) call()
v_accvgpr_write a124, 0.00000 // (op 4372) call()
v_accvgpr_write a125, 0.00000 // (op 4372) call()
v_accvgpr_write a126, 0.00000 // (op 4372) call()
v_accvgpr_write a127, 0.00000 // (op 4372) call()
s_mul_hi_i32 s0, s16, s10 // (op 13566) 
v_lshrrev_b32 v10, 1, v1 // (op 13566) 
s_add_i32 s1, s0, s16 // (op 13566) 

v_and_b32 v11, -2, v1 // (op 13566) 
s_ashr_i32 s0, s1, 31 // (op 13566) 
v_sub_u32 v12, v1, v11 // (op 13566) 

s_cmp_eq_i32 s10, 0 // (op 13566) 
s_mov_b32 s18, scc // (op 13566) 
v_lshlrev_b32 v11, 6, v12 // (op 13566) 

s_cmp_eq_i32 s18, 0 // (op 13566) 
s_cselect_b32 s19, 0, -1 // (op 13566) 

v_mov_b32 v13, 0 // (op 13566) convert
v_mov_b32 v12, v11 // (op 13566) convert

s_and_b32 s18, s13, 31 // (op 13566) 
s_lshl_b32 s22, 1, s18 // (op 13566) 
s_add_i32 s23, s22, s19 // (op 13566) 


s_and_b32 s19, s0, s23 // (op 13566) 


s_add_i32 s0, s1, s19 // (op 13566) 


s_ashr_i32 s1, s0, s18 // (op 13566) 


s_lshl_b32 s0, s13, 24 // (op 13566) 
s_ashr_i32 s18, s0, 31 // (op 13566) 

s_xor_b32 s0, s1, s18 // (op 13566) 

s_sub_i32 s1, s0, s18 // (op 13566) 


s_mul_hi_i32 s0, s1, s14 // (op 13566) 
s_add_i32 s18, s0, s1 // (op 13566) 

s_ashr_i32 s0, s18, 31 // (op 13566) 
s_cmp_eq_i32 s14, 0 // (op 13566) 
s_mov_b32 s19, scc // (op 13566) 
s_cmp_eq_i32 s19, 0 // (op 13566) 
s_cselect_b32 s22, 0, -1 // (op 13566) 

s_and_b32 s19, s15, 31 // (op 13566) 
s_lshl_b32 s23, 1, s19 // (op 13566) 
s_add_i32 s24, s23, s22 // (op 13566) 


s_and_b32 s25, s0, s24 // (op 13566) 


s_add_i32 s0, s18, s25 // (op 13566) 


s_ashr_i32 s18, s0, s19 // (op 13566) 


s_lshl_b32 s0, s15, 24 // (op 13566) 
s_ashr_i32 s19, s0, 31 // (op 13566) 

s_xor_b32 s0, s18, s19 // (op 13566) 

s_sub_i32 s18, s0, s19 // (op 13566) 


s_mul_i32 s0, s1, s11 // (op 13566) 

s_sub_u32 s1, s16, s0 // (op 13566) 

s_mul_i32 s0, s18, s11 // (op 13566) 
s_add_u32 s0, s0, s1 // (op 13566) 


s_mul_hi_i32 s1, s16, s9 // (op 13566) 
s_add_i32 s18, s1, s16 // (op 13566) 

s_ashr_i32 s1, s18, 31 // (op 13566) 
s_cmp_eq_i32 s9, 0 // (op 13566) 
s_mov_b32 s19, scc // (op 13566) 
s_cmp_eq_i32 s19, 0 // (op 13566) 
s_cselect_b32 s22, 0, -1 // (op 13566) 

s_and_b32 s19, s45, 31 // (op 13566) 
s_lshl_b32 s23, 1, s19 // (op 13566) 
s_add_i32 s24, s23, s22 // (op 13566) 


s_and_b32 s25, s1, s24 // (op 13566) 


s_add_i32 s1, s18, s25 // (op 13566) 


s_ashr_i32 s18, s1, s19 // (op 13566) 


s_lshl_b32 s1, s45, 24 // (op 13566) 
s_ashr_i32 s19, s1, 31 // (op 13566) 

s_xor_b32 s1, s18, s19 // (op 13566) 

s_sub_i32 s18, s1, s19 // (op 13566) 


s_add_u32 s22, s6, 256 // (op 13566) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13566) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13566) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13566) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 13566) 

s_mul_i32 s1, s11, s22 // (op 13566) 

s_mul_i32 s19, s8, s1 // (op 13566) 

s_mul_i32 s1, s18, s19 // (op 13566) 

s_sub_u32 s19, s16, s1 // (op 13566) 

s_mov_b32 s1, s19 // (op 13566) 
s_mul_hi_i32 s22, s1, s12 // (op 13566) 
s_add_i32 s23, s22, s1 // (op 13566) 


s_ashr_i32 s1, s23, 31 // (op 13566) 
s_cmp_eq_i32 s12, 0 // (op 13566) 
s_mov_b32 s22, scc // (op 13566) 
s_cmp_eq_i32 s22, 0 // (op 13566) 
s_cselect_b32 s24, 0, -1 // (op 13566) 

s_and_b32 s22, s44, 31 // (op 13566) 
s_lshl_b32 s25, 1, s22 // (op 13566) 
s_add_i32 s47, s25, s24 // (op 13566) 


s_and_b32 s24, s1, s47 // (op 13566) 


s_add_i32 s1, s23, s24 // (op 13566) 


s_ashr_i32 s23, s1, s22 // (op 13566) 


s_lshl_b32 s1, s44, 24 // (op 13566) 
s_ashr_i32 s22, s1, 31 // (op 13566) 

s_xor_b32 s1, s23, s22 // (op 13566) 

s_sub_i32 s23, s1, s22 // (op 13566) 


s_mul_i32 s1, s46, s11 // (op 13566) 
s_add_u32 s24, s20, 128 // (op 13566) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 13566) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 13566) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 13566) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 7 // (op 13566) 

s_sub_i32 s22, s24, s1 // (op 13566) 

s_mul_i32 s24, s23, s22 // (op 13566) 


s_sub_u32 s25, s19, s24 // (op 13566) 


s_add_u32 s19, s25, s1 // (op 13566) 


s_cmp_eq_i32 s18, 0 // (op 13566) 
s_mov_b32 s1, scc // (op 13566) 

s_cmp_eq_i32 s1, 0 // (op 13566) 
s_cselect_b32 s18, s19, s0 // (op 13566) 



v_lshl_add_u32 v11, s18, 7, v10 // (op 13566) 


v_mul_lo_u32 v15, v11, s27 // (op 13566) most significant: low of low * high
v_mul_hi_u32 v10, v11, s26 // (op 13566) most significant: high of low * low
v_mul_lo_u32 v14, v11, s26 // (op 13566) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 13566) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 13566) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 13566) most significant half



v_mul_lo_u32 v11, v15, 6 // (op 13566) most significant: low of high * low
v_mul_hi_u32 v12, v14, 6 // (op 13566) most significant: high of low * low
v_mul_lo_u32 v10, v14, 6 // (op 13566) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 13566) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 13566) 

v_mov_b32 v10, v12 // (op 13566) convert

s_mul_i32 s1, s21, 6 // (op 13568) most significant: low of high * low
s_mul_i32 s0, s20, 6 // (op 13568) least significant
s_mul_hi_u32 s18, s20, 6 // (op 13568) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 13568) most significant: sum

s_mul_i32 s19, s26, s1 // (op 13568) most significant: low of low * high
s_mul_i32 s22, s27, s0 // (op 13568) most significant: low of high * low
s_mul_i32 s18, s26, s0 // (op 13568) least significant
s_mul_hi_u32 s23, s26, s0 // (op 13568) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 13568) most significant: sum
s_add_u32 s19, s19, s23 // (op 13568) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 13568) 

s_mov_b32 s48, s28 // (op 13568) reg expression
s_mov_b32 s49, s29 // (op 13568) reg expression
s_mov_b32 s50, s0 // (op 13568) reg expression
s_mov_b32 s51, 131072 // (op 13568) call()

s_mul_hi_i32 s47, s16, s10 // (op 13585) 
v_lshrrev_b32 v11, 1, v1 // (op 13585) 
s_add_i32 s0, s47, s16 // (op 13585) 

v_and_b32 v12, -2, v1 // (op 13585) 
s_ashr_i32 s1, s0, 31 // (op 13585) 
v_sub_u32 v13, v1, v12 // (op 13585) 

s_cmp_eq_i32 s10, 0 // (op 13585) 
s_mov_b32 s47, scc // (op 13585) 
v_lshlrev_b32 v12, 6, v13 // (op 13585) 

s_cmp_eq_i32 s47, 0 // (op 13585) 
s_cselect_b32 s18, 0, -1 // (op 13585) 

v_mov_b32 v15, 0 // (op 13585) convert
v_mov_b32 v14, v12 // (op 13585) convert

s_and_b32 s19, s13, 31 // (op 13585) 
s_lshl_b32 s47, 1, s19 // (op 13585) 
s_add_i32 s22, s47, s18 // (op 13585) 


s_and_b32 s18, s1, s22 // (op 13585) 


s_add_i32 s1, s0, s18 // (op 13585) 


s_ashr_i32 s0, s1, s19 // (op 13585) 


s_lshl_b32 s1, s13, 24 // (op 13585) 
s_ashr_i32 s47, s1, 31 // (op 13585) 

s_xor_b32 s1, s0, s47 // (op 13585) 

s_sub_i32 s0, s1, s47 // (op 13585) 


s_mul_hi_i32 s1, s0, s14 // (op 13585) 
s_add_i32 s47, s1, s0 // (op 13585) 

s_ashr_i32 s1, s47, 31 // (op 13585) 
s_cmp_eq_i32 s14, 0 // (op 13585) 
s_mov_b32 s18, scc // (op 13585) 
s_cmp_eq_i32 s18, 0 // (op 13585) 
s_cselect_b32 s19, 0, -1 // (op 13585) 

s_and_b32 s18, s15, 31 // (op 13585) 
s_lshl_b32 s22, 1, s18 // (op 13585) 
s_add_i32 s23, s22, s19 // (op 13585) 


s_and_b32 s19, s1, s23 // (op 13585) 


s_add_i32 s1, s47, s19 // (op 13585) 


s_ashr_i32 s19, s1, s18 // (op 13585) 


s_lshl_b32 s1, s15, 24 // (op 13585) 
s_ashr_i32 s18, s1, 31 // (op 13585) 

s_xor_b32 s1, s19, s18 // (op 13585) 

s_sub_i32 s19, s1, s18 // (op 13585) 


s_mul_i32 s1, s0, s11 // (op 13585) 

s_sub_u32 s0, s16, s1 // (op 13585) 

s_mul_i32 s1, s19, s11 // (op 13585) 
s_add_u32 s1, s1, s0 // (op 13585) 


s_mul_hi_i32 s0, s16, s9 // (op 13585) 
s_add_i32 s47, s0, s16 // (op 13585) 

s_ashr_i32 s0, s47, 31 // (op 13585) 
s_cmp_eq_i32 s9, 0 // (op 13585) 
s_mov_b32 s18, scc // (op 13585) 
s_cmp_eq_i32 s18, 0 // (op 13585) 
s_cselect_b32 s19, 0, -1 // (op 13585) 

s_and_b32 s18, s45, 31 // (op 13585) 
s_lshl_b32 s22, 1, s18 // (op 13585) 
s_add_i32 s23, s22, s19 // (op 13585) 


s_and_b32 s19, s0, s23 // (op 13585) 


s_add_i32 s0, s47, s19 // (op 13585) 


s_ashr_i32 s19, s0, s18 // (op 13585) 


s_lshl_b32 s0, s45, 24 // (op 13585) 
s_ashr_i32 s18, s0, 31 // (op 13585) 

s_xor_b32 s0, s19, s18 // (op 13585) 

s_sub_i32 s19, s0, s18 // (op 13585) 


s_add_u32 s22, s6, 256 // (op 13585) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13585) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13585) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13585) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 13585) 

s_mul_i32 s0, s11, s22 // (op 13585) 

s_mul_i32 s18, s8, s0 // (op 13585) 

s_mul_i32 s0, s19, s18 // (op 13585) 

s_sub_u32 s18, s16, s0 // (op 13585) 

s_mov_b32 s0, s18 // (op 13585) 
s_mul_hi_i32 s47, s0, s12 // (op 13585) 
s_add_i32 s22, s47, s0 // (op 13585) 


s_ashr_i32 s0, s22, 31 // (op 13585) 
s_cmp_eq_i32 s12, 0 // (op 13585) 
s_mov_b32 s47, scc // (op 13585) 
s_cmp_eq_i32 s47, 0 // (op 13585) 
s_cselect_b32 s23, 0, -1 // (op 13585) 

s_and_b32 s47, s44, 31 // (op 13585) 
s_lshl_b32 s24, 1, s47 // (op 13585) 
s_add_i32 s25, s24, s23 // (op 13585) 


s_and_b32 s23, s0, s25 // (op 13585) 


s_add_i32 s0, s22, s23 // (op 13585) 


s_ashr_i32 s22, s0, s47 // (op 13585) 


s_lshl_b32 s0, s44, 24 // (op 13585) 
s_ashr_i32 s47, s0, 31 // (op 13585) 

s_xor_b32 s0, s22, s47 // (op 13585) 

s_sub_i32 s22, s0, s47 // (op 13585) 


s_mul_i32 s0, s46, s11 // (op 13585) 
s_add_u32 s24, s20, 128 // (op 13585) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 13585) End of Int64 add, unlocking SCC
s_sub_u32 s60, s24, 1 // (op 13585) least significant half; sets scc
s_subb_u32 s61, s25, 0 // (op 13585) most significant half; uses scc

s_ashr_i64 s[24:25], s[60:61], 7 // (op 13585) 

s_sub_i32 s23, s24, s0 // (op 13585) 

s_mul_i32 s47, s22, s23 // (op 13585) 


s_sub_u32 s22, s18, s47 // (op 13585) 


s_add_u32 s18, s22, s0 // (op 13585) 


s_cmp_eq_i32 s19, 0 // (op 13585) 
s_mov_b32 s0, scc // (op 13585) 

s_cmp_eq_i32 s0, 0 // (op 13585) 
s_cselect_b32 s19, s18, s1 // (op 13585) 



v_lshl_add_u32 v12, s19, 7, v11 // (op 13585) 


v_mul_lo_u32 v17, v12, s27 // (op 13585) most significant: low of low * high
v_mul_hi_u32 v11, v12, s26 // (op 13585) most significant: high of low * low
v_mul_lo_u32 v16, v12, s26 // (op 13585) least significant: low of low * low
v_add_u32 v17, v17, v11 // (op 13585) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 13585) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 13585) most significant half



v_mul_lo_u32 v15, v17, 6 // (op 13585) most significant: low of high * low
v_mul_hi_u32 v11, v16, 6 // (op 13585) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 13585) least significant: low of low * low
v_add_u32 v15, v15, v11 // (op 13585) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 13585) 

v_mov_b32 v11, v12 // (op 13585) convert

s_mul_i32 s1, s21, 6 // (op 13587) most significant: low of high * low
s_mul_i32 s0, s20, 6 // (op 13587) least significant
s_mul_hi_u32 s47, s20, 6 // (op 13587) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13587) most significant: sum

s_mul_i32 s19, s26, s1 // (op 13587) most significant: low of low * high
s_mul_i32 s47, s27, s0 // (op 13587) most significant: low of high * low
s_mul_i32 s18, s26, s0 // (op 13587) least significant
s_mul_hi_u32 s22, s26, s0 // (op 13587) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 13587) most significant: sum
s_add_u32 s19, s19, s22 // (op 13587) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 13587) 

s_mov_b32 s48, s28 // (op 13587) reg expression
s_mov_b32 s49, s29 // (op 13587) reg expression
s_mov_b32 s50, s0 // (op 13587) reg expression
s_mov_b32 s51, 131072 // (op 13587) call()

s_mul_hi_i32 s47, s16, s10 // (op 13630) 
v_and_b32 v12, 4294967294, v1 // (op 13630) 
s_add_i32 s0, s47, s16 // (op 13630) 

v_and_b32 v13, -2, v1 // (op 13630) 
s_ashr_i32 s1, s0, 31 // (op 13630) 
v_sub_u32 v14, v1, v13 // (op 13630) 

s_cmp_eq_i32 s10, 0 // (op 13630) 
s_mov_b32 s47, scc // (op 13630) 
v_lshlrev_b32 v13, 6, v14 // (op 13630) 

s_cmp_eq_i32 s47, 0 // (op 13630) 
s_cselect_b32 s18, 0, -1 // (op 13630) 

v_mov_b32 v15, 0 // (op 13630) convert
v_mov_b32 v14, v13 // (op 13630) convert

s_and_b32 s19, s13, 31 // (op 13630) 
s_lshl_b32 s47, 1, s19 // (op 13630) 
s_add_i32 s22, s47, s18 // (op 13630) 


s_and_b32 s18, s1, s22 // (op 13630) 


s_add_i32 s1, s0, s18 // (op 13630) 


s_ashr_i32 s0, s1, s19 // (op 13630) 


s_lshl_b32 s1, s13, 24 // (op 13630) 
s_ashr_i32 s47, s1, 31 // (op 13630) 

s_xor_b32 s1, s0, s47 // (op 13630) 

s_sub_i32 s0, s1, s47 // (op 13630) 


s_mul_hi_i32 s1, s0, s14 // (op 13630) 
s_add_i32 s47, s1, s0 // (op 13630) 

s_ashr_i32 s1, s47, 31 // (op 13630) 
s_cmp_eq_i32 s14, 0 // (op 13630) 
s_mov_b32 s18, scc // (op 13630) 
s_cmp_eq_i32 s18, 0 // (op 13630) 
s_cselect_b32 s19, 0, -1 // (op 13630) 

s_and_b32 s18, s15, 31 // (op 13630) 
s_lshl_b32 s22, 1, s18 // (op 13630) 
s_add_i32 s23, s22, s19 // (op 13630) 


s_and_b32 s19, s1, s23 // (op 13630) 


s_add_i32 s1, s47, s19 // (op 13630) 


s_ashr_i32 s19, s1, s18 // (op 13630) 


s_lshl_b32 s1, s15, 24 // (op 13630) 
s_ashr_i32 s18, s1, 31 // (op 13630) 

s_xor_b32 s1, s19, s18 // (op 13630) 

s_sub_i32 s19, s1, s18 // (op 13630) 


s_add_u32 s22, s6, 256 // (op 13630) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13630) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13630) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13630) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 13630) 

s_mul_i32 s1, s19, s22 // (op 13630) 

s_sub_i32 s47, s0, s1 // (op 13630) 


s_mul_hi_i32 s0, s16, s9 // (op 13630) 
s_add_i32 s1, s0, s16 // (op 13630) 

s_ashr_i32 s0, s1, 31 // (op 13630) 
s_cmp_eq_i32 s9, 0 // (op 13630) 
s_mov_b32 s18, scc // (op 13630) 
s_cmp_eq_i32 s18, 0 // (op 13630) 
s_cselect_b32 s19, 0, -1 // (op 13630) 

s_and_b32 s18, s45, 31 // (op 13630) 
s_lshl_b32 s24, 1, s18 // (op 13630) 
s_add_i32 s25, s24, s19 // (op 13630) 


s_and_b32 s19, s0, s25 // (op 13630) 


s_add_i32 s0, s1, s19 // (op 13630) 


s_ashr_i32 s1, s0, s18 // (op 13630) 


s_lshl_b32 s0, s45, 24 // (op 13630) 
s_ashr_i32 s18, s0, 31 // (op 13630) 

s_xor_b32 s0, s1, s18 // (op 13630) 

s_sub_i32 s1, s0, s18 // (op 13630) 


s_mul_i32 s0, s11, s22 // (op 13630) 

s_mul_i32 s18, s8, s0 // (op 13630) 

s_mul_i32 s0, s1, s18 // (op 13630) 

s_sub_u32 s18, s16, s0 // (op 13630) 

s_mul_hi_i32 s0, s18, s12 // (op 13630) 
s_add_i32 s19, s0, s18 // (op 13630) 


s_ashr_i32 s0, s19, 31 // (op 13630) 
s_cmp_eq_i32 s12, 0 // (op 13630) 
s_mov_b32 s18, scc // (op 13630) 
s_cmp_eq_i32 s18, 0 // (op 13630) 
s_cselect_b32 s22, 0, -1 // (op 13630) 

s_and_b32 s18, s44, 31 // (op 13630) 
s_lshl_b32 s23, 1, s18 // (op 13630) 
s_add_i32 s24, s23, s22 // (op 13630) 


s_and_b32 s25, s0, s24 // (op 13630) 


s_add_i32 s0, s19, s25 // (op 13630) 


s_ashr_i32 s19, s0, s18 // (op 13630) 


s_lshl_b32 s0, s44, 24 // (op 13630) 
s_ashr_i32 s18, s0, 31 // (op 13630) 

s_xor_b32 s0, s19, s18 // (op 13630) 

s_sub_i32 s19, s0, s18 // (op 13630) 


s_cmp_eq_i32 s1, 0 // (op 13630) 
s_mov_b32 s0, scc // (op 13630) 

s_cmp_eq_i32 s0, 0 // (op 13630) 
s_cselect_b32 s1, s19, s47 // (op 13630) 



v_lshl_add_u32 v13, s1, 8, v12 // (op 13630) 


v_mul_lo_u32 v17, v13, s31 // (op 13630) most significant: low of low * high
v_mul_hi_u32 v12, v13, s30 // (op 13630) most significant: high of low * low
v_mul_lo_u32 v16, v13, s30 // (op 13630) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 13630) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 13630) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 13630) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 13630) most significant: low of high * low
v_mul_hi_u32 v16, v12, 6 // (op 13630) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 13630) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 13630) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 13630) 

v_mov_b32 v14, v12 // (op 13630) convert

s_mul_i32 s1, s7, 6 // (op 13632) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 13632) least significant
s_mul_hi_u32 s47, s6, 6 // (op 13632) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13632) most significant: sum

s_mul_i32 s19, s30, s1 // (op 13632) most significant: low of low * high
s_mul_i32 s47, s31, s0 // (op 13632) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 13632) least significant
s_mul_hi_u32 s22, s30, s0 // (op 13632) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 13632) most significant: sum
s_add_u32 s19, s19, s22 // (op 13632) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 13632) 

s_mov_b32 s60, s32 // (op 13632) reg expression
s_mov_b32 s61, s33 // (op 13632) reg expression
s_mov_b32 s62, s0 // (op 13632) reg expression
s_mov_b32 s63, 131072 // (op 13632) call()

s_mul_hi_i32 s47, s16, s10 // (op 13649) 
v_and_b32 v12, 4294967294, v1 // (op 13649) 
s_add_i32 s0, s47, s16 // (op 13649) 

v_and_b32 v13, -2, v1 // (op 13649) 
s_ashr_i32 s1, s0, 31 // (op 13649) 
v_sub_u32 v15, v1, v13 // (op 13649) 

s_cmp_eq_i32 s10, 0 // (op 13649) 
s_mov_b32 s47, scc // (op 13649) 
v_lshlrev_b32 v13, 6, v15 // (op 13649) 

s_cmp_eq_i32 s47, 0 // (op 13649) 
s_cselect_b32 s18, 0, -1 // (op 13649) 

v_mov_b32 v17, 0 // (op 13649) convert
v_mov_b32 v16, v13 // (op 13649) convert

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


s_add_u32 s22, s6, 256 // (op 13649) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13649) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13649) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13649) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 13649) 

s_mul_i32 s1, s19, s22 // (op 13649) 

s_sub_i32 s47, s0, s1 // (op 13649) 


s_mul_hi_i32 s0, s16, s9 // (op 13649) 
s_add_i32 s1, s0, s16 // (op 13649) 

s_ashr_i32 s0, s1, 31 // (op 13649) 
s_cmp_eq_i32 s9, 0 // (op 13649) 
s_mov_b32 s18, scc // (op 13649) 
s_cmp_eq_i32 s18, 0 // (op 13649) 
s_cselect_b32 s19, 0, -1 // (op 13649) 

s_and_b32 s18, s45, 31 // (op 13649) 
s_lshl_b32 s24, 1, s18 // (op 13649) 
s_add_i32 s25, s24, s19 // (op 13649) 


s_and_b32 s19, s0, s25 // (op 13649) 


s_add_i32 s0, s1, s19 // (op 13649) 


s_ashr_i32 s1, s0, s18 // (op 13649) 


s_lshl_b32 s0, s45, 24 // (op 13649) 
s_ashr_i32 s18, s0, 31 // (op 13649) 

s_xor_b32 s0, s1, s18 // (op 13649) 

s_sub_i32 s1, s0, s18 // (op 13649) 


s_mul_i32 s0, s11, s22 // (op 13649) 

s_mul_i32 s18, s8, s0 // (op 13649) 

s_mul_i32 s0, s1, s18 // (op 13649) 

s_sub_u32 s18, s16, s0 // (op 13649) 

s_mul_hi_i32 s0, s18, s12 // (op 13649) 
s_add_i32 s19, s0, s18 // (op 13649) 


s_ashr_i32 s0, s19, 31 // (op 13649) 
s_cmp_eq_i32 s12, 0 // (op 13649) 
s_mov_b32 s18, scc // (op 13649) 
s_cmp_eq_i32 s18, 0 // (op 13649) 
s_cselect_b32 s22, 0, -1 // (op 13649) 

s_and_b32 s18, s44, 31 // (op 13649) 
s_lshl_b32 s23, 1, s18 // (op 13649) 
s_add_i32 s24, s23, s22 // (op 13649) 


s_and_b32 s25, s0, s24 // (op 13649) 


s_add_i32 s0, s19, s25 // (op 13649) 


s_ashr_i32 s19, s0, s18 // (op 13649) 


s_lshl_b32 s0, s44, 24 // (op 13649) 
s_ashr_i32 s18, s0, 31 // (op 13649) 

s_xor_b32 s0, s19, s18 // (op 13649) 

s_sub_i32 s19, s0, s18 // (op 13649) 


s_cmp_eq_i32 s1, 0 // (op 13649) 
s_mov_b32 s0, scc // (op 13649) 

s_cmp_eq_i32 s0, 0 // (op 13649) 
s_cselect_b32 s1, s19, s47 // (op 13649) 



v_lshl_add_u32 v13, s1, 8, v12 // (op 13649) 


v_mul_lo_u32 v19, v13, s31 // (op 13649) most significant: low of low * high
v_mul_hi_u32 v12, v13, s30 // (op 13649) most significant: high of low * low
v_mul_lo_u32 v18, v13, s30 // (op 13649) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 13649) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v18 // (op 13649) least significant half
v_addc_co_u32 v13, s[0:1], v17, v19, s[0:1] // (op 13649) most significant half



v_mul_lo_u32 v17, v13, 6 // (op 13649) most significant: low of high * low
v_mul_hi_u32 v15, v12, 6 // (op 13649) most significant: high of low * low
v_mul_lo_u32 v16, v12, 6 // (op 13649) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 13649) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 13649) 

v_mov_b32 v15, v12 // (op 13649) convert

s_mul_i32 s1, s7, 6 // (op 13651) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 13651) least significant
s_mul_hi_u32 s47, s6, 6 // (op 13651) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13651) most significant: sum

s_mul_i32 s19, s30, s1 // (op 13651) most significant: low of low * high
s_mul_i32 s47, s31, s0 // (op 13651) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 13651) least significant
s_mul_hi_u32 s22, s30, s0 // (op 13651) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 13651) most significant: sum
s_add_u32 s19, s19, s22 // (op 13651) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 13651) 

s_mov_b32 s60, s32 // (op 13651) reg expression
s_mov_b32 s61, s33 // (op 13651) reg expression
s_mov_b32 s62, s0 // (op 13651) reg expression
s_mov_b32 s63, 131072 // (op 13651) call()

s_mul_hi_i32 s47, s16, s10 // (op 13899) 
v_and_b32 v12, 63, v1 // (op 13899) 
s_add_i32 s0, s47, s16 // (op 13899) 

v_lshrrev_b32 v13, 4, v12 // (op 13899) 
s_ashr_i32 s1, s0, 31 // (op 13899) 
v_and_b32 v16, 3, v13 // (op 13899) 

s_cmp_eq_i32 s10, 0 // (op 13899) 
s_mov_b32 s47, scc // (op 13899) 
v_and_b32 v13, 15, v1 // (op 13899) 
s_cmp_eq_i32 s47, 0 // (op 13899) 
s_cselect_b32 s18, 0, -1 // (op 13899) 

v_lshl_add_u32 v17, v16, 4, v13 // (op 13899) 


s_and_b32 s19, s13, 31 // (op 13899) 
v_lshrrev_b32 v13, 7, v1 // (op 13899) 
s_lshl_b32 s47, 1, s19 // (op 13899) 
v_lshl_add_u32 v16, v13, 6, v17 // (op 13899) 


s_add_i32 s22, s47, s18 // (op 13899) 


v_lshrrev_b32 v13, 6, v12 // (op 13899) 

s_and_b32 s18, s1, s22 // (op 13899) 


v_lshlrev_b32 v12, 2, v13 // (op 13899) 

s_add_i32 s1, s0, s18 // (op 13899) 


s_ashr_i32 s0, s1, s19 // (op 13899) 


s_lshl_b32 s1, s13, 24 // (op 13899) 
s_ashr_i32 s47, s1, 31 // (op 13899) 

s_xor_b32 s1, s0, s47 // (op 13899) 

s_sub_i32 s0, s1, s47 // (op 13899) 


s_mul_hi_i32 s1, s0, s14 // (op 13899) 
s_add_i32 s47, s1, s0 // (op 13899) 

s_ashr_i32 s1, s47, 31 // (op 13899) 
s_cmp_eq_i32 s14, 0 // (op 13899) 
s_mov_b32 s18, scc // (op 13899) 
s_cmp_eq_i32 s18, 0 // (op 13899) 
s_cselect_b32 s19, 0, -1 // (op 13899) 

s_and_b32 s18, s15, 31 // (op 13899) 
s_lshl_b32 s22, 1, s18 // (op 13899) 
s_add_i32 s23, s22, s19 // (op 13899) 


s_and_b32 s19, s1, s23 // (op 13899) 


s_add_i32 s1, s47, s19 // (op 13899) 


s_ashr_i32 s19, s1, s18 // (op 13899) 


s_lshl_b32 s1, s15, 24 // (op 13899) 
s_ashr_i32 s18, s1, 31 // (op 13899) 

s_xor_b32 s1, s19, s18 // (op 13899) 

s_sub_i32 s19, s1, s18 // (op 13899) 


s_mul_i32 s1, s0, s11 // (op 13899) 

s_sub_u32 s0, s16, s1 // (op 13899) 

s_mul_i32 s1, s19, s11 // (op 13899) 
s_add_u32 s1, s1, s0 // (op 13899) 


s_mul_hi_i32 s0, s16, s9 // (op 13899) 
s_add_i32 s47, s0, s16 // (op 13899) 

s_ashr_i32 s0, s47, 31 // (op 13899) 
s_cmp_eq_i32 s9, 0 // (op 13899) 
s_mov_b32 s18, scc // (op 13899) 
s_cmp_eq_i32 s18, 0 // (op 13899) 
s_cselect_b32 s19, 0, -1 // (op 13899) 

s_and_b32 s18, s45, 31 // (op 13899) 
s_lshl_b32 s22, 1, s18 // (op 13899) 
s_add_i32 s23, s22, s19 // (op 13899) 


s_and_b32 s19, s0, s23 // (op 13899) 


s_add_i32 s0, s47, s19 // (op 13899) 


s_ashr_i32 s19, s0, s18 // (op 13899) 


s_lshl_b32 s0, s45, 24 // (op 13899) 
s_ashr_i32 s18, s0, 31 // (op 13899) 

s_xor_b32 s0, s19, s18 // (op 13899) 

s_sub_i32 s19, s0, s18 // (op 13899) 


s_add_u32 s22, s6, 256 // (op 13899) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13899) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13899) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13899) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 13899) 

s_mul_i32 s0, s11, s22 // (op 13899) 

s_mul_i32 s18, s8, s0 // (op 13899) 

s_mul_i32 s0, s19, s18 // (op 13899) 

s_sub_u32 s18, s16, s0 // (op 13899) 

s_mov_b32 s0, s18 // (op 13899) 
s_mul_hi_i32 s47, s0, s12 // (op 13899) 
s_add_i32 s22, s47, s0 // (op 13899) 


s_ashr_i32 s0, s22, 31 // (op 13899) 
s_cmp_eq_i32 s12, 0 // (op 13899) 
s_mov_b32 s47, scc // (op 13899) 
s_cmp_eq_i32 s47, 0 // (op 13899) 
s_cselect_b32 s23, 0, -1 // (op 13899) 

s_and_b32 s47, s44, 31 // (op 13899) 
s_lshl_b32 s24, 1, s47 // (op 13899) 
s_add_i32 s25, s24, s23 // (op 13899) 


s_and_b32 s23, s0, s25 // (op 13899) 


s_add_i32 s0, s22, s23 // (op 13899) 


s_ashr_i32 s22, s0, s47 // (op 13899) 


s_lshl_b32 s0, s44, 24 // (op 13899) 
s_ashr_i32 s47, s0, 31 // (op 13899) 

s_xor_b32 s0, s22, s47 // (op 13899) 

s_sub_i32 s22, s0, s47 // (op 13899) 


s_mul_i32 s0, s46, s11 // (op 13899) 
s_add_u32 s24, s20, 128 // (op 13899) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 13899) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 13899) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 13899) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 7 // (op 13899) 

s_sub_i32 s23, s24, s0 // (op 13899) 

s_mul_i32 s47, s22, s23 // (op 13899) 


s_sub_u32 s22, s18, s47 // (op 13899) 


s_add_u32 s18, s22, s0 // (op 13899) 


s_cmp_eq_i32 s19, 0 // (op 13899) 
s_mov_b32 s0, scc // (op 13899) 

s_cmp_eq_i32 s0, 0 // (op 13899) 
s_cselect_b32 s19, s18, s1 // (op 13899) 



v_lshl_add_u32 v13, s19, 7, v16 // (op 13899) 


v_mul_lo_u32 v16, v13, s34 // (op 13899) 
v_add_u32 v16, v16, v12 // (op 13899) 


s_mul_i32 s1, s34, s21 // (op 13901) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 13901) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 13901) least significant
s_mul_hi_u32 s18, s34, s20 // (op 13901) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13901) most significant: sum
s_add_u32 s1, s1, s18 // (op 13901) most significant: sum


s_mov_b32 s64, s36 // (op 13901) reg expression
s_mov_b32 s65, s37 // (op 13901) reg expression
s_mov_b32 s66, s0 // (op 13901) reg expression
s_mov_b32 s67, 131072 // (op 13901) call()

s_mul_hi_i32 s47, s16, s10 // (op 13920) 
v_and_b32 v12, 63, v1 // (op 13920) 
s_add_i32 s0, s47, s16 // (op 13920) 

v_lshrrev_b32 v13, 4, v12 // (op 13920) 
s_ashr_i32 s1, s0, 31 // (op 13920) 
v_and_b32 v17, 3, v13 // (op 13920) 

s_cmp_eq_i32 s10, 0 // (op 13920) 
s_mov_b32 s47, scc // (op 13920) 
v_and_b32 v13, 15, v1 // (op 13920) 
s_cmp_eq_i32 s47, 0 // (op 13920) 
s_cselect_b32 s18, 0, -1 // (op 13920) 

v_lshl_add_u32 v18, v17, 4, v13 // (op 13920) 


s_and_b32 s19, s13, 31 // (op 13920) 
v_lshrrev_b32 v13, 7, v1 // (op 13920) 
s_lshl_b32 s47, 1, s19 // (op 13920) 
v_lshl_add_u32 v17, v13, 6, v18 // (op 13920) 


s_add_i32 s22, s47, s18 // (op 13920) 


v_lshrrev_b32 v13, 6, v12 // (op 13920) 

s_and_b32 s18, s1, s22 // (op 13920) 


v_lshlrev_b32 v12, 2, v13 // (op 13920) 

s_add_i32 s1, s0, s18 // (op 13920) 


s_ashr_i32 s0, s1, s19 // (op 13920) 


s_lshl_b32 s1, s13, 24 // (op 13920) 
s_ashr_i32 s47, s1, 31 // (op 13920) 

s_xor_b32 s1, s0, s47 // (op 13920) 

s_sub_i32 s0, s1, s47 // (op 13920) 


s_mul_hi_i32 s1, s0, s14 // (op 13920) 
s_add_i32 s47, s1, s0 // (op 13920) 

s_ashr_i32 s1, s47, 31 // (op 13920) 
s_cmp_eq_i32 s14, 0 // (op 13920) 
s_mov_b32 s18, scc // (op 13920) 
s_cmp_eq_i32 s18, 0 // (op 13920) 
s_cselect_b32 s19, 0, -1 // (op 13920) 

s_and_b32 s18, s15, 31 // (op 13920) 
s_lshl_b32 s22, 1, s18 // (op 13920) 
s_add_i32 s23, s22, s19 // (op 13920) 


s_and_b32 s19, s1, s23 // (op 13920) 


s_add_i32 s1, s47, s19 // (op 13920) 


s_ashr_i32 s19, s1, s18 // (op 13920) 


s_lshl_b32 s1, s15, 24 // (op 13920) 
s_ashr_i32 s18, s1, 31 // (op 13920) 

s_xor_b32 s1, s19, s18 // (op 13920) 

s_sub_i32 s19, s1, s18 // (op 13920) 


s_mul_i32 s1, s0, s11 // (op 13920) 

s_sub_u32 s0, s16, s1 // (op 13920) 

s_mul_i32 s1, s19, s11 // (op 13920) 
s_add_u32 s1, s1, s0 // (op 13920) 


s_mul_hi_i32 s0, s16, s9 // (op 13920) 
s_add_i32 s47, s0, s16 // (op 13920) 

s_ashr_i32 s0, s47, 31 // (op 13920) 
s_cmp_eq_i32 s9, 0 // (op 13920) 
s_mov_b32 s18, scc // (op 13920) 
s_cmp_eq_i32 s18, 0 // (op 13920) 
s_cselect_b32 s19, 0, -1 // (op 13920) 

s_and_b32 s18, s45, 31 // (op 13920) 
s_lshl_b32 s22, 1, s18 // (op 13920) 
s_add_i32 s23, s22, s19 // (op 13920) 


s_and_b32 s19, s0, s23 // (op 13920) 


s_add_i32 s0, s47, s19 // (op 13920) 


s_ashr_i32 s19, s0, s18 // (op 13920) 


s_lshl_b32 s0, s45, 24 // (op 13920) 
s_ashr_i32 s18, s0, 31 // (op 13920) 

s_xor_b32 s0, s19, s18 // (op 13920) 

s_sub_i32 s19, s0, s18 // (op 13920) 


s_add_u32 s22, s6, 256 // (op 13920) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13920) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13920) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13920) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 13920) 

s_mul_i32 s0, s11, s22 // (op 13920) 

s_mul_i32 s18, s8, s0 // (op 13920) 

s_mul_i32 s0, s19, s18 // (op 13920) 

s_sub_u32 s18, s16, s0 // (op 13920) 

s_mov_b32 s0, s18 // (op 13920) 
s_mul_hi_i32 s47, s0, s12 // (op 13920) 
s_add_i32 s22, s47, s0 // (op 13920) 


s_ashr_i32 s0, s22, 31 // (op 13920) 
s_cmp_eq_i32 s12, 0 // (op 13920) 
s_mov_b32 s47, scc // (op 13920) 
s_cmp_eq_i32 s47, 0 // (op 13920) 
s_cselect_b32 s23, 0, -1 // (op 13920) 

s_and_b32 s47, s44, 31 // (op 13920) 
s_lshl_b32 s24, 1, s47 // (op 13920) 
s_add_i32 s25, s24, s23 // (op 13920) 


s_and_b32 s23, s0, s25 // (op 13920) 


s_add_i32 s0, s22, s23 // (op 13920) 


s_ashr_i32 s22, s0, s47 // (op 13920) 


s_lshl_b32 s0, s44, 24 // (op 13920) 
s_ashr_i32 s47, s0, 31 // (op 13920) 

s_xor_b32 s0, s22, s47 // (op 13920) 

s_sub_i32 s22, s0, s47 // (op 13920) 


s_mul_i32 s0, s46, s11 // (op 13920) 
s_add_u32 s24, s20, 128 // (op 13920) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 13920) End of Int64 add, unlocking SCC
s_sub_u32 s68, s24, 1 // (op 13920) least significant half; sets scc
s_subb_u32 s69, s25, 0 // (op 13920) most significant half; uses scc

s_ashr_i64 s[24:25], s[68:69], 7 // (op 13920) 

s_sub_i32 s23, s24, s0 // (op 13920) 

s_mul_i32 s47, s22, s23 // (op 13920) 


s_sub_u32 s22, s18, s47 // (op 13920) 


s_add_u32 s18, s22, s0 // (op 13920) 


s_cmp_eq_i32 s19, 0 // (op 13920) 
s_mov_b32 s0, scc // (op 13920) 

s_cmp_eq_i32 s0, 0 // (op 13920) 
s_cselect_b32 s19, s18, s1 // (op 13920) 



v_lshl_add_u32 v13, s19, 7, v17 // (op 13920) 


v_mul_lo_u32 v17, v13, s34 // (op 13920) 
v_add_u32 v17, v17, v12 // (op 13920) 


s_mul_i32 s1, s34, s21 // (op 13922) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 13922) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 13922) least significant
s_mul_hi_u32 s18, s34, s20 // (op 13922) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13922) most significant: sum
s_add_u32 s1, s1, s18 // (op 13922) most significant: sum


s_mov_b32 s64, s36 // (op 13922) reg expression
s_mov_b32 s65, s37 // (op 13922) reg expression
s_mov_b32 s66, s0 // (op 13922) reg expression
s_mov_b32 s67, 131072 // (op 13922) call()

s_mul_hi_i32 s47, s16, s10 // (op 13951) 
v_and_b32 v12, 63, v1 // (op 13951) 
s_add_i32 s0, s47, s16 // (op 13951) 

v_lshrrev_b32 v13, 4, v12 // (op 13951) 
s_ashr_i32 s1, s0, 31 // (op 13951) 
v_and_b32 v18, 3, v13 // (op 13951) 

s_cmp_eq_i32 s10, 0 // (op 13951) 
s_mov_b32 s47, scc // (op 13951) 
v_and_b32 v13, 15, v1 // (op 13951) 
s_cmp_eq_i32 s47, 0 // (op 13951) 
s_cselect_b32 s18, 0, -1 // (op 13951) 

v_lshl_add_u32 v19, v18, 4, v13 // (op 13951) 


s_and_b32 s19, s13, 31 // (op 13951) 
v_lshrrev_b32 v13, 6, v1 // (op 13951) 
s_lshl_b32 s47, 1, s19 // (op 13951) 
v_and_b32 v18, 1, v13 // (op 13951) 

s_add_i32 s22, s47, s18 // (op 13951) 


v_lshl_add_u32 v13, v18, 7, v19 // (op 13951) 


s_and_b32 s18, s1, s22 // (op 13951) 


v_lshrrev_b32 v18, 6, v12 // (op 13951) 

s_add_i32 s1, s0, s18 // (op 13951) 


s_ashr_i32 s0, s1, s19 // (op 13951) 


s_lshl_b32 s1, s13, 24 // (op 13951) 
s_ashr_i32 s47, s1, 31 // (op 13951) 

s_xor_b32 s1, s0, s47 // (op 13951) 

s_sub_i32 s0, s1, s47 // (op 13951) 


s_mul_hi_i32 s1, s0, s14 // (op 13951) 
s_add_i32 s47, s1, s0 // (op 13951) 

s_ashr_i32 s1, s47, 31 // (op 13951) 
s_cmp_eq_i32 s14, 0 // (op 13951) 
s_mov_b32 s18, scc // (op 13951) 
s_cmp_eq_i32 s18, 0 // (op 13951) 
s_cselect_b32 s19, 0, -1 // (op 13951) 

s_and_b32 s18, s15, 31 // (op 13951) 
s_lshl_b32 s22, 1, s18 // (op 13951) 
s_add_i32 s23, s22, s19 // (op 13951) 


s_and_b32 s19, s1, s23 // (op 13951) 


s_add_i32 s1, s47, s19 // (op 13951) 


s_ashr_i32 s19, s1, s18 // (op 13951) 


s_lshl_b32 s1, s15, 24 // (op 13951) 
s_ashr_i32 s18, s1, 31 // (op 13951) 

s_xor_b32 s1, s19, s18 // (op 13951) 

s_sub_i32 s19, s1, s18 // (op 13951) 


s_add_u32 s22, s6, 256 // (op 13951) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13951) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13951) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13951) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 13951) 

s_mul_i32 s1, s19, s22 // (op 13951) 

s_sub_i32 s47, s0, s1 // (op 13951) 


s_mul_hi_i32 s0, s16, s9 // (op 13951) 
s_add_i32 s1, s0, s16 // (op 13951) 

s_ashr_i32 s0, s1, 31 // (op 13951) 
s_cmp_eq_i32 s9, 0 // (op 13951) 
s_mov_b32 s18, scc // (op 13951) 
s_cmp_eq_i32 s18, 0 // (op 13951) 
s_cselect_b32 s19, 0, -1 // (op 13951) 

s_and_b32 s18, s45, 31 // (op 13951) 
s_lshl_b32 s24, 1, s18 // (op 13951) 
s_add_i32 s25, s24, s19 // (op 13951) 


s_and_b32 s19, s0, s25 // (op 13951) 


s_add_i32 s0, s1, s19 // (op 13951) 


s_ashr_i32 s1, s0, s18 // (op 13951) 


s_lshl_b32 s0, s45, 24 // (op 13951) 
s_ashr_i32 s18, s0, 31 // (op 13951) 

s_xor_b32 s0, s1, s18 // (op 13951) 

s_sub_i32 s1, s0, s18 // (op 13951) 


s_mul_i32 s0, s11, s22 // (op 13951) 

s_mul_i32 s18, s8, s0 // (op 13951) 

s_mul_i32 s0, s1, s18 // (op 13951) 

s_sub_u32 s18, s16, s0 // (op 13951) 

s_mul_hi_i32 s0, s18, s12 // (op 13951) 
s_add_i32 s19, s0, s18 // (op 13951) 


s_ashr_i32 s0, s19, 31 // (op 13951) 
s_cmp_eq_i32 s12, 0 // (op 13951) 
s_mov_b32 s18, scc // (op 13951) 
s_cmp_eq_i32 s18, 0 // (op 13951) 
s_cselect_b32 s22, 0, -1 // (op 13951) 

s_and_b32 s18, s44, 31 // (op 13951) 
s_lshl_b32 s23, 1, s18 // (op 13951) 
s_add_i32 s24, s23, s22 // (op 13951) 


s_and_b32 s25, s0, s24 // (op 13951) 


s_add_i32 s0, s19, s25 // (op 13951) 


s_ashr_i32 s19, s0, s18 // (op 13951) 


s_lshl_b32 s0, s44, 24 // (op 13951) 
s_ashr_i32 s18, s0, 31 // (op 13951) 

s_xor_b32 s0, s19, s18 // (op 13951) 

s_sub_i32 s19, s0, s18 // (op 13951) 


s_cmp_eq_i32 s1, 0 // (op 13951) 
s_mov_b32 s0, scc // (op 13951) 

s_cmp_eq_i32 s0, 0 // (op 13951) 
s_cselect_b32 s1, s19, s47 // (op 13951) 



v_lshl_add_u32 v12, s1, 8, v13 // (op 13951) 


v_mul_lo_u32 v13, v12, s38 // (op 13951) 

v_lshl_add_u32 v12, v18, 2, v13 // (op 13951) 


s_mul_i32 s1, s38, s7 // (op 13953) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 13953) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 13953) least significant
s_mul_hi_u32 s18, s38, s6 // (op 13953) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13953) most significant: sum
s_add_u32 s1, s1, s18 // (op 13953) most significant: sum


s_mov_b32 s68, s40 // (op 13953) reg expression
s_mov_b32 s69, s41 // (op 13953) reg expression
s_mov_b32 s70, s0 // (op 13953) reg expression
s_mov_b32 s71, 131072 // (op 13953) call()

s_mul_hi_i32 s47, s16, s10 // (op 13972) 
v_and_b32 v13, 63, v1 // (op 13972) 
s_add_i32 s0, s47, s16 // (op 13972) 

v_lshrrev_b32 v18, 4, v13 // (op 13972) 
s_ashr_i32 s1, s0, 31 // (op 13972) 
v_and_b32 v19, 3, v18 // (op 13972) 

s_cmp_eq_i32 s10, 0 // (op 13972) 
s_mov_b32 s47, scc // (op 13972) 
v_and_b32 v18, 15, v1 // (op 13972) 
s_cmp_eq_i32 s47, 0 // (op 13972) 
s_cselect_b32 s18, 0, -1 // (op 13972) 

v_lshl_add_u32 v20, v19, 4, v18 // (op 13972) 


s_and_b32 s19, s13, 31 // (op 13972) 
v_lshrrev_b32 v18, 6, v1 // (op 13972) 
s_lshl_b32 s47, 1, s19 // (op 13972) 
v_and_b32 v19, 1, v18 // (op 13972) 

s_add_i32 s22, s47, s18 // (op 13972) 


v_lshl_add_u32 v18, v19, 7, v20 // (op 13972) 


s_and_b32 s18, s1, s22 // (op 13972) 


v_lshrrev_b32 v19, 6, v13 // (op 13972) 

s_add_i32 s1, s0, s18 // (op 13972) 


s_ashr_i32 s0, s1, s19 // (op 13972) 


s_lshl_b32 s1, s13, 24 // (op 13972) 
s_ashr_i32 s47, s1, 31 // (op 13972) 

s_xor_b32 s1, s0, s47 // (op 13972) 

s_sub_i32 s0, s1, s47 // (op 13972) 


s_mul_hi_i32 s1, s0, s14 // (op 13972) 
s_add_i32 s47, s1, s0 // (op 13972) 

s_ashr_i32 s1, s47, 31 // (op 13972) 
s_cmp_eq_i32 s14, 0 // (op 13972) 
s_mov_b32 s18, scc // (op 13972) 
s_cmp_eq_i32 s18, 0 // (op 13972) 
s_cselect_b32 s19, 0, -1 // (op 13972) 

s_and_b32 s18, s15, 31 // (op 13972) 
s_lshl_b32 s22, 1, s18 // (op 13972) 
s_add_i32 s23, s22, s19 // (op 13972) 


s_and_b32 s19, s1, s23 // (op 13972) 


s_add_i32 s1, s47, s19 // (op 13972) 


s_ashr_i32 s19, s1, s18 // (op 13972) 


s_lshl_b32 s1, s15, 24 // (op 13972) 
s_ashr_i32 s18, s1, 31 // (op 13972) 

s_xor_b32 s1, s19, s18 // (op 13972) 

s_sub_i32 s19, s1, s18 // (op 13972) 


s_add_u32 s22, s6, 256 // (op 13972) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13972) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13972) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13972) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 13972) 

s_mul_i32 s1, s19, s22 // (op 13972) 

s_sub_i32 s47, s0, s1 // (op 13972) 


s_mul_hi_i32 s0, s16, s9 // (op 13972) 
s_add_i32 s1, s0, s16 // (op 13972) 

s_ashr_i32 s0, s1, 31 // (op 13972) 
s_cmp_eq_i32 s9, 0 // (op 13972) 
s_mov_b32 s18, scc // (op 13972) 
s_cmp_eq_i32 s18, 0 // (op 13972) 
s_cselect_b32 s19, 0, -1 // (op 13972) 

s_and_b32 s18, s45, 31 // (op 13972) 
s_lshl_b32 s24, 1, s18 // (op 13972) 
s_add_i32 s25, s24, s19 // (op 13972) 


s_and_b32 s19, s0, s25 // (op 13972) 


s_add_i32 s0, s1, s19 // (op 13972) 


s_ashr_i32 s1, s0, s18 // (op 13972) 


s_lshl_b32 s0, s45, 24 // (op 13972) 
s_ashr_i32 s18, s0, 31 // (op 13972) 

s_xor_b32 s0, s1, s18 // (op 13972) 

s_sub_i32 s1, s0, s18 // (op 13972) 


s_mul_i32 s0, s11, s22 // (op 13972) 

s_mul_i32 s18, s8, s0 // (op 13972) 

s_mul_i32 s0, s1, s18 // (op 13972) 

s_sub_u32 s18, s16, s0 // (op 13972) 

s_mul_hi_i32 s0, s18, s12 // (op 13972) 
s_add_i32 s19, s0, s18 // (op 13972) 


s_ashr_i32 s0, s19, 31 // (op 13972) 
s_cmp_eq_i32 s12, 0 // (op 13972) 
s_mov_b32 s18, scc // (op 13972) 
s_cmp_eq_i32 s18, 0 // (op 13972) 
s_cselect_b32 s22, 0, -1 // (op 13972) 

s_and_b32 s18, s44, 31 // (op 13972) 
s_lshl_b32 s23, 1, s18 // (op 13972) 
s_add_i32 s24, s23, s22 // (op 13972) 


s_and_b32 s25, s0, s24 // (op 13972) 


s_add_i32 s0, s19, s25 // (op 13972) 


s_ashr_i32 s19, s0, s18 // (op 13972) 


s_lshl_b32 s0, s44, 24 // (op 13972) 
s_ashr_i32 s18, s0, 31 // (op 13972) 

s_xor_b32 s0, s19, s18 // (op 13972) 

s_sub_i32 s19, s0, s18 // (op 13972) 


s_cmp_eq_i32 s1, 0 // (op 13972) 
s_mov_b32 s0, scc // (op 13972) 

s_cmp_eq_i32 s0, 0 // (op 13972) 
s_cselect_b32 s1, s19, s47 // (op 13972) 



v_lshl_add_u32 v13, s1, 8, v18 // (op 13972) 


v_mul_lo_u32 v18, v13, s38 // (op 13972) 

v_lshl_add_u32 v13, v19, 2, v18 // (op 13972) 


s_mul_i32 s1, s38, s7 // (op 13974) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 13974) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 13974) least significant
s_mul_hi_u32 s18, s38, s6 // (op 13974) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13974) most significant: sum
s_add_u32 s1, s1, s18 // (op 13974) most significant: sum


s_mov_b32 s68, s40 // (op 13974) reg expression
s_mov_b32 s69, s41 // (op 13974) reg expression
s_mov_b32 s70, s0 // (op 13974) reg expression
s_mov_b32 s71, 131072 // (op 13974) call()


s_mul_i32 s1, s31, 6 // (op 12920) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 12920) least significant
s_mul_hi_u32 s47, s30, 6 // (op 12920) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 12920) most significant: sum

v_lshrrev_b64 v[18:19], 3, s[0:1] // (op 12920) 


v_mov_b32 v20, v14 // (op 12920) 
buffer_load_dwordx4 v[22:25], v20, s[60:63], 0 offen // (op 12920) Load value
buffer_load_dwordx2 v[26:27], v20, s[60:63], 0 offen offset:16 // (op 12920) Load value
buffer_load_dwordx4 v[28:31], v20, s[60:63], 0 offen offset:24 // (op 12920) Load value
buffer_load_dwordx2 v[32:33], v20, s[60:63], 0 offen offset:40 // (op 12920) Load value
v_add_u32 v20, v20, v18 // (op 12920) 
buffer_load_dwordx4 v[34:37], v20, s[60:63], 0 offen // (op 12920) Load value
buffer_load_dwordx2 v[38:39], v20, s[60:63], 0 offen offset:16 // (op 12920) Load value
buffer_load_dwordx4 v[40:43], v20, s[60:63], 0 offen offset:24 // (op 12920) Load value
buffer_load_dwordx2 v[44:45], v20, s[60:63], 0 offen offset:40 // (op 12920) Load value





buffer_load_dwordx4 v[46:49], v10, s[48:51], 0 offen // (op 12923) Load value
buffer_load_dwordx2 v[50:51], v10, s[48:51], 0 offen offset:16 // (op 12923) Load value
buffer_load_dwordx4 v[52:55], v10, s[48:51], 0 offen offset:24 // (op 12923) Load value
buffer_load_dwordx2 v[56:57], v10, s[48:51], 0 offen offset:40 // (op 12923) Load value


v_mov_b32 v18, v12 // (op 76) reg expression

buffer_load_dwordx2 v[20:21], v18, s[68:71], 0 offen // (op 76) Load value



s_lshl_b32 s47, s38, 6 // (op 726) 
v_add_u32 v18, v12, s47 // (op 726) 


buffer_load_dwordx2 v[58:59], v18, s[68:71], 0 offen // (op 726) Load value




v_mov_b32 v12, v16 // (op 72) reg expression

buffer_load_dwordx2 v[18:19], v12, s[64:67], 0 offen // (op 72) Load value



s_mul_i32 s1, s31, 6 // (op 12926) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 12926) least significant
s_mul_hi_u32 s47, s30, 6 // (op 12926) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 12926) most significant: sum

v_lshrrev_b64 v[60:61], 3, s[0:1] // (op 12926) 


v_mov_b32 v12, v14 // (op 12926) 
buffer_load_dwordx4 v[62:65], v12, s[60:63], 0 offen offset:96 // (op 12926) Load value
buffer_load_dwordx2 v[66:67], v12, s[60:63], 0 offen offset:112 // (op 12926) Load value
buffer_load_dwordx4 v[68:71], v12, s[60:63], 0 offen offset:120 // (op 12926) Load value
buffer_load_dwordx2 v[72:73], v12, s[60:63], 0 offen offset:136 // (op 12926) Load value
v_add_u32 v12, v12, v60 // (op 12926) 
buffer_load_dwordx4 v[74:77], v12, s[60:63], 0 offen offset:96 // (op 12926) Load value
buffer_load_dwordx2 v[78:79], v12, s[60:63], 0 offen offset:112 // (op 12926) Load value
buffer_load_dwordx4 v[80:83], v12, s[60:63], 0 offen offset:120 // (op 12926) Load value
buffer_load_dwordx2 v[84:85], v12, s[60:63], 0 offen offset:136 // (op 12926) Load value






buffer_load_dwordx4 v[86:89], v10, s[48:51], 0 offen offset:96 // (op 12929) Load value
buffer_load_dwordx2 v[90:91], v10, s[48:51], 0 offen offset:112 // (op 12929) Load value
buffer_load_dwordx4 v[92:95], v10, s[48:51], 0 offen offset:120 // (op 12929) Load value
buffer_load_dwordx2 v[96:97], v10, s[48:51], 0 offen offset:136 // (op 12929) Load value

v_and_b32 v10, -2, v1 // (op 13779) 
v_sub_u32 v12, v1, v10 // (op 13779) 

v_lshlrev_b32 v10, 6, v12 // (op 13779) 

v_lshrrev_b32 v12, 1, v1 // (op 13779) 
v_lshl_add_u32 v14, v12, 8, v10 // (op 13779) 


v_mul_lo_u32 v10, v14, 6 // (op 13779) 

v_lshrrev_b32 v12, 3, v10 // (op 13779) 




s_waitcnt vmcnt(26)
ds_write_b128 v12, v[22:25] // (op 12931) Store local data 
s_waitcnt vmcnt(25)
ds_write_b64 v12, v[26:27] offset:16 // (op 12931) Store local data 
s_waitcnt vmcnt(24)
ds_write_b128 v12, v[28:31] offset:24 // (op 12931) Store local data 
s_waitcnt vmcnt(23)
ds_write_b64 v12, v[32:33] offset:40 // (op 12931) Store local data 
s_waitcnt vmcnt(22)
ds_write_b128 v12, v[34:37] offset:96 // (op 12931) Store local data 
s_waitcnt vmcnt(21)
ds_write_b64 v12, v[38:39] offset:112 // (op 12931) Store local data 
s_waitcnt vmcnt(20)
ds_write_b128 v12, v[40:43] offset:120 // (op 12931) Store local data 
s_waitcnt vmcnt(19)
ds_write_b64 v12, v[44:45] offset:136 // (op 12931) Store local data 

v_and_b32 v10, -2, v1 // (op 13721) 
v_sub_u32 v12, v1, v10 // (op 13721) 

v_lshlrev_b32 v10, 6, v12 // (op 13721) 

v_lshrrev_b32 v12, 1, v1 // (op 13721) 
v_lshl_add_u32 v14, v12, 7, v10 // (op 13721) 


v_mul_lo_u32 v10, v14, 6 // (op 13721) 

v_lshrrev_b32 v12, 3, v10 // (op 13721) 




s_waitcnt vmcnt(18)
ds_write_b128 v12, v[46:49] offset:24576 // (op 12932) Store local data 
s_waitcnt vmcnt(17)
ds_write_b64 v12, v[50:51] offset:24592 // (op 12932) Store local data 
s_waitcnt vmcnt(16)
ds_write_b128 v12, v[52:55] offset:24600 // (op 12932) Store local data 
s_waitcnt vmcnt(15)
ds_write_b64 v12, v[56:57] offset:24616 // (op 12932) Store local data 

v_and_b32 v10, -2, v1 // (op 13705) 
v_sub_u32 v12, v1, v10 // (op 13705) 

v_lshlrev_b32 v10, 6, v12 // (op 13705) 

v_lshrrev_b32 v12, 1, v1 // (op 13705) 
v_lshl_add_u32 v14, v12, 7, v10 // (op 13705) 


v_mul_lo_u32 v10, v14, 6 // (op 13705) 

v_lshrrev_b32 v12, 3, v10 // (op 13705) 

v_and_b32 v10, -2, v1 // (op 13763) 
v_sub_u32 v14, v1, v10 // (op 13763) 

v_lshlrev_b32 v10, 6, v14 // (op 13763) 

v_lshrrev_b32 v14, 1, v1 // (op 13763) 
v_lshl_add_u32 v16, v14, 8, v10 // (op 13763) 


v_mul_lo_u32 v10, v16, 6 // (op 13763) 

v_lshrrev_b32 v14, 3, v10 // (op 13763) 

v_and_b32 v10, -2, v1 // (op 13831) 
v_sub_u32 v16, v1, v10 // (op 13831) 

v_lshlrev_b32 v10, 6, v16 // (op 13831) 

v_lshrrev_b32 v16, 1, v1 // (op 13831) 
v_lshl_add_u32 v60, v16, 7, v10 // (op 13831) 


v_mul_lo_u32 v10, v60, 6 // (op 13831) 

v_lshrrev_b32 v16, 3, v10 // (op 13831) 

v_and_b32 v10, -2, v1 // (op 13860) 
v_sub_u32 v60, v1, v10 // (op 13860) 

v_lshlrev_b32 v10, 6, v60 // (op 13860) 

v_lshrrev_b32 v60, 1, v1 // (op 13860) 
v_lshl_add_u32 v61, v60, 8, v10 // (op 13860) 


v_mul_lo_u32 v10, v61, 6 // (op 13860) 

v_lshrrev_b32 v60, 3, v10 // (op 13860) 

s_mov_b32 s47, 0 // (op 4036) call()
s_ashr_i64 s[0:1], s[42:43], 7 // (op 4035) 
s_mov_b32 s18, s0 // (op 4035) convert

s_and_b32 s19, s18, 4294967294 // (op 4035) 

s_cmp_lt_u32 s47, s19 // (op 4035) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__ForLoopBottom_KLoop_4035 // (op 4035) Condition: Top (jump to RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__ForLoopBottom_KLoop_4035 if false)
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__ForLoopTop_KLoop_4035:
s_add_u32 s47, s47, 2 // (op 4037) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 14172) 



ds_read_b128 v[98:101], v3 offset:24576 // (op 62) Load local data 
ds_read_b64 v[102:103], v3 offset:24592 // (op 62) Load local data 
s_barrier  // (op 14185) 



ds_read_b128 v[104:107], v4 // (op 67) Load local data 
ds_read_b64 v[108:109], v4 offset:16 // (op 67) Load local data 



ds_read_b128 v[110:113], v4 offset:1536 // (op 312) Load local data 
ds_read_b64 v[114:115], v4 offset:1552 // (op 312) Load local data 
s_waitcnt vmcnt(14)
v_mov_b32 v116, v20 // (op 13435) reg expression



v_mov_b32 v117, v21 // (op 13435) reg expression
s_waitcnt vmcnt(12)
v_mov_b32 v118, v18 // (op 13462) reg expression
v_mov_b32 v119, v19 // (op 13462) reg expression
ds_read_b128 v[120:123], v4 offset:3072 // (op 448) Load local data 
ds_read_b64 v[124:125], v4 offset:3088 // (op 448) Load local data 
v_bfe_u32 v126, v118, 0, 8 // (op 12241) 



v_bfe_u32 v127, v118, 8, 8 // (op 12241) 
v_bfe_u32 v128, v118, 16, 8 // (op 12241) 
v_bfe_u32 v129, v118, 24, 8 // (op 12241) 
v_permlane16_swap_b32 v126, v127 // (op 12241) 
ds_read_b128 v[130:133], v4 offset:4608 // (op 584) Load local data 
v_permlane16_swap_b32 v128, v129 // (op 12241) 
s_nop 1
v_permlane32_swap_b32 v126, v128 // (op 12241) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v127, v129 // (op 12241) 
ds_read_b64 v[134:135], v4 offset:4624 // (op 584) Load local data 



v_bfe_u32 v136, v116, 0, 8 // (op 12336) 
v_bfe_u32 v137, v116, 8, 8 // (op 12336) 
v_bfe_u32 v138, v116, 16, 8 // (op 12336) 
v_bfe_u32 v139, v116, 24, 8 // (op 12336) 
v_permlane16_swap_b32 v136, v137 // (op 12336) 
ds_read_b128 v[140:143], v4 offset:6144 // (op 720) Load local data 
ds_read_b64 v[144:145], v4 offset:6160 // (op 720) Load local data 
v_permlane16_swap_b32 v138, v139 // (op 12336) 



s_nop 1
v_permlane32_swap_b32 v136, v138 // (op 12336) Wait state hazard: VALU Write Readlane Hazard

v_permlane32_swap_b32 v137, v139 // (op 12336) 
s_mul_i32 s1, s31, 6 // (op 10) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 10) least significant
s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[98:103], v[104:109], a[0:3], v126, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 78) 
s_mul_hi_u32 s18, s30, 6 // (op 10) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 10) most significant: sum

s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[98:103], v[110:115], a[4:7], v126, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 315) 
v_lshrrev_b64 v[146:147], 3, s[0:1] // (op 10) 


v_mov_b32 v10, v15 // (op 10) 
v_mov_b32 v61, v10 // (op 10) 
buffer_load_dwordx4 v[22:25], v61, s[60:63], 0 offen offset:192 // (op 10) Load value
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[98:103], v[120:125], a[8:11], v126, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 451) 
buffer_load_dwordx2 v[26:27], v61, s[60:63], 0 offen offset:208 // (op 10) Load value
buffer_load_dwordx4 v[28:31], v61, s[60:63], 0 offen offset:216 // (op 10) Load value
v_mov_b32 v148, v58 // (op 13448) reg expression
v_mov_b32 v149, v59 // (op 13448) reg expression
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[98:103], v[130:135], a[12:15], v126, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 587) 
v_bfe_u32 v150, v148, 0, 8 // (op 12368) 
v_bfe_u32 v151, v148, 8, 8 // (op 12368) 
v_bfe_u32 v152, v148, 16, 8 // (op 12368) 
v_bfe_u32 v153, v148, 24, 8 // (op 12368) 
v_permlane16_swap_b32 v150, v151 // (op 12368) 
ds_read_b128 v[154:157], v4 offset:7680 // (op 856) Load local data 
ds_read_b64 v[158:159], v4 offset:7696 // (op 856) Load local data 
v_permlane16_swap_b32 v152, v153 // (op 12368) 



ds_read_b128 v[160:163], v4 offset:9216 // (op 992) Load local data 
s_nop 0
v_permlane32_swap_b32 v150, v152 // (op 12368) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v151, v153 // (op 12368) 
ds_read_b64 v[164:165], v4 offset:9232 // (op 992) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[98:103], v[140:145], a[16:19], v126, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 723) 
ds_read_b128 v[166:169], v4 offset:10752 // (op 1128) Load local data 
ds_read_b64 v[170:171], v4 offset:10768 // (op 1128) Load local data 
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[98:103], v[154:159], a[20:23], v126, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 859) 



s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[98:103], v[160:165], a[24:27], v126, v152 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 995) 
ds_read_b128 v[172:175], v3 offset:26112 // (op 1629) Load local data 
ds_read_b64 v[176:177], v3 offset:26128 // (op 1629) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[98:103], v[166:171], a[28:31], v126, v153 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1131) 




ds_read_b128 v[98:101], v3 offset:27648 // (op 2847) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[172:177], v[104:109], a[32:35], v127, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1899) 
ds_read_b64 v[102:103], v3 offset:27664 // (op 2847) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[172:177], v[110:115], a[36:39], v127, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1861) 
ds_read_b128 v[178:181], v3 offset:29184 // (op 4065) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[172:177], v[120:125], a[40:43], v127, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1823) 
ds_read_b64 v[182:183], v3 offset:29200 // (op 4065) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[172:177], v[130:135], a[44:47], v127, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1785) 
buffer_load_dwordx2 v[32:33], v61, s[60:63], 0 offen offset:232 // (op 10) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[172:177], v[140:145], a[48:51], v127, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1747) 
v_add_u32 v61, v61, v146 // (op 10) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[172:177], v[154:159], a[52:55], v127, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1709) 
buffer_load_dwordx4 v[34:37], v61, s[60:63], 0 offen offset:192 // (op 10) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[172:177], v[160:165], a[56:59], v127, v152 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1671) 
buffer_load_dwordx2 v[38:39], v61, s[60:63], 0 offen offset:208 // (op 10) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[172:177], v[166:171], a[60:63], v127, v153 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1633) 
buffer_load_dwordx4 v[40:43], v61, s[60:63], 0 offen offset:216 // (op 10) Load value

buffer_load_dwordx2 v[44:45], v61, s[60:63], 0 offen offset:232 // (op 10) Load value




v_mov_b32 v61, v11 // (op 4) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[98:103], v[104:109], a[64:67], v128, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 3117) 
buffer_load_dwordx4 v[46:49], v61, s[48:51], 0 offen offset:192 // (op 4) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[98:103], v[110:115], a[68:71], v128, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 3079) 
buffer_load_dwordx2 v[50:51], v61, s[48:51], 0 offen offset:208 // (op 4) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[98:103], v[120:125], a[72:75], v128, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 3041) 
buffer_load_dwordx4 v[52:55], v61, s[48:51], 0 offen offset:216 // (op 4) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[98:103], v[130:135], a[76:79], v128, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 3003) 
buffer_load_dwordx2 v[56:57], v61, s[48:51], 0 offen offset:232 // (op 4) Load value


v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[98:103], v[140:145], a[80:83], v128, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2965) 
s_barrier  // (op 14208) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[98:103], v[154:159], a[84:87], v128, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2927) 
v_mov_b32 v146, v13 // (op 13443) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[98:103], v[160:165], a[88:91], v128, v152 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2889) 
v_add_u32 v147, 8, v146 // (op 13443) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[98:103], v[166:171], a[92:95], v128, v153 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2851) 
s_waitcnt vmcnt(23)
ds_write_b128 v60, v[62:65] offset:36864 // (op 222) Store local data 

s_waitcnt vmcnt(22)
ds_write_b64 v60, v[66:67] offset:36880 // (op 222) Store local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[178:183], v[104:109], a[96:99], v129, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 4335) 
s_waitcnt vmcnt(21)
ds_write_b128 v60, v[68:71] offset:36888 // (op 222) Store local data 

s_waitcnt vmcnt(20)
ds_write_b64 v60, v[72:73] offset:36904 // (op 222) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[178:183], v[110:115], a[100:103], v129, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 4297) 
s_waitcnt vmcnt(19)
ds_write_b128 v60, v[74:77] offset:36960 // (op 222) Store local data 

s_waitcnt vmcnt(18)
ds_write_b64 v60, v[78:79] offset:36976 // (op 222) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[178:183], v[120:125], a[104:107], v129, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 4259) 
s_waitcnt vmcnt(17)
ds_write_b128 v60, v[80:83] offset:36984 // (op 222) Store local data 

s_waitcnt vmcnt(16)
ds_write_b64 v60, v[84:85] offset:37000 // (op 222) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[178:183], v[130:135], a[108:111], v129, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 4221) 


s_barrier  // (op 14202) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[178:183], v[140:145], a[112:115], v129, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 4183) 
s_waitcnt vmcnt(15)
ds_write_b128 v16, v[86:89] offset:61440 // (op 212) Store local data 

s_waitcnt vmcnt(14)
ds_write_b64 v16, v[90:91] offset:61456 // (op 212) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[178:183], v[154:159], a[116:119], v129, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 4145) 
s_waitcnt vmcnt(13)
ds_write_b128 v16, v[92:95] offset:61464 // (op 212) Store local data 

s_waitcnt vmcnt(12)
ds_write_b64 v16, v[96:97] offset:61480 // (op 212) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[178:183], v[160:165], a[120:123], v129, v152 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 4107) 
buffer_load_dwordx2 v[20:21], v147, s[68:71], 0 offen // (op 13443) Load value




v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[178:183], v[166:171], a[124:127], v129, v153 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 4069) 
s_lshl_b32 s0, s38, 6 // (op 13457) 




s_add_u32 s0, s0, 8 // (op 13457) 
v_add_u32 v147, v146, s0 // (op 13457) 


buffer_load_dwordx2 v[58:59], v147, s[68:71], 0 offen // (op 13457) Load value

s_waitcnt lgkmcnt(0)
s_barrier  // (op 14199) 






v_add_u32 v13, 8, v13 // (op 13962) 
v_mov_b32 v98, v17 // (op 13470) 
v_add_u32 v99, 8, v98 // (op 13470) 

buffer_load_dwordx2 v[18:19], v99, s[64:67], 0 offen // (op 13470) Load value


v_add_u32 v17, 8, v17 // (op 13910) 
ds_read_b128 v[98:101], v5 offset:61440 // (op 214) Load local data 

ds_read_b64 v[102:103], v5 offset:61456 // (op 214) Load local data 
s_mul_i32 s1, s31, 6 // (op 223) most significant: low of high * low
s_barrier  // (op 14205) 
s_mul_i32 s0, s30, 6 // (op 223) least significant



s_mul_hi_u32 s18, s30, 6 // (op 223) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 223) most significant: sum

v_lshrrev_b64 v[104:105], 3, s[0:1] // (op 223) 


v_mov_b32 v106, v10 // (op 223) 
ds_read_b128 v[110:113], v6 offset:36864 // (op 216) Load local data 
ds_read_b64 v[114:115], v6 offset:36880 // (op 216) Load local data 
buffer_load_dwordx4 v[62:65], v106, s[60:63], 0 offen offset:288 // (op 223) Load value



v_bfe_u32 v120, v119, 0, 8 // (op 12256) 
v_bfe_u32 v121, v119, 8, 8 // (op 12256) 
v_bfe_u32 v122, v119, 16, 8 // (op 12256) 
v_bfe_u32 v123, v119, 24, 8 // (op 12256) 
v_permlane16_swap_b32 v120, v121 // (op 12256) 
ds_read_b128 v[124:127], v6 offset:38400 // (op 324) Load local data 
ds_read_b64 v[128:129], v6 offset:38416 // (op 324) Load local data 
buffer_load_dwordx2 v[66:67], v106, s[60:63], 0 offen offset:304 // (op 223) Load value



v_permlane16_swap_b32 v122, v123 // (op 12256) 
ds_read_b128 v[130:133], v6 offset:39936 // (op 460) Load local data 
s_nop 0
v_permlane32_swap_b32 v120, v122 // (op 12256) Wait state hazard: VALU Write Readlane Hazard
buffer_load_dwordx4 v[68:71], v106, s[60:63], 0 offen offset:312 // (op 223) Load value
v_permlane32_swap_b32 v121, v123 // (op 12256) 
ds_read_b64 v[134:135], v6 offset:39952 // (op 460) Load local data 




v_bfe_u32 v136, v117, 0, 8 // (op 12351) 
v_bfe_u32 v137, v117, 8, 8 // (op 12351) 
v_bfe_u32 v138, v117, 16, 8 // (op 12351) 
v_bfe_u32 v139, v117, 24, 8 // (op 12351) 
v_permlane16_swap_b32 v136, v137 // (op 12351) 
ds_read_b128 v[140:143], v6 offset:41472 // (op 596) Load local data 
ds_read_b64 v[144:145], v6 offset:41488 // (op 596) Load local data 
v_permlane16_swap_b32 v138, v139 // (op 12351) 



s_nop 1
v_permlane32_swap_b32 v136, v138 // (op 12351) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v137, v139 // (op 12351) 
ds_read_b128 v[150:153], v6 offset:43008 // (op 732) Load local data 

ds_read_b64 v[154:155], v6 offset:43024 // (op 732) Load local data 



s_waitcnt lgkmcnt(8)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[98:103], v[110:115], a[0:3], v120, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 221) 
ds_read_b128 v[156:159], v6 offset:44544 // (op 868) Load local data 
s_waitcnt lgkmcnt(7)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[98:103], v[124:129], a[4:7], v120, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 316) 
v_bfe_u32 v116, v149, 0, 8 // (op 12383) 
v_bfe_u32 v117, v149, 8, 8 // (op 12383) 
v_bfe_u32 v118, v149, 16, 8 // (op 12383) 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[98:103], v[130:135], a[8:11], v120, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 452) 
v_bfe_u32 v119, v149, 24, 8 // (op 12383) 
v_permlane16_swap_b32 v116, v117 // (op 12383) 
s_nop 0
v_permlane16_swap_b32 v118, v119 // (op 12383) Wait state hazard: VALU Write Readlane Hazard
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[98:103], v[140:145], a[12:15], v120, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 588) 
s_nop 0
v_permlane32_swap_b32 v116, v118 // (op 12383) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v117, v119 // (op 12383) 
ds_read_b64 v[160:161], v6 offset:44560 // (op 868) Load local data 




s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[98:103], v[150:155], a[16:19], v120, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 724) 
ds_read_b128 v[162:165], v6 offset:46080 // (op 1004) Load local data 
ds_read_b64 v[166:167], v6 offset:46096 // (op 1004) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[98:103], v[156:161], a[20:23], v120, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 860) 



ds_read_b128 v[168:171], v6 offset:47616 // (op 1140) Load local data 
ds_read_b64 v[172:173], v6 offset:47632 // (op 1140) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[98:103], v[162:167], a[24:27], v120, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 996) 
buffer_load_dwordx2 v[72:73], v106, s[60:63], 0 offen offset:328 // (op 223) Load value



s_barrier  // (op 14259) 
v_add_u32 v106, v106, v104 // (op 223) 
ds_read_b128 v[174:177], v5 offset:62976 // (op 1641) Load local data 
ds_read_b64 v[178:179], v5 offset:62992 // (op 1641) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[98:103], v[168:173], a[28:31], v120, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1132) 




ds_read_b128 v[98:101], v5 offset:64512 // (op 2859) Load local data 
buffer_load_dwordx4 v[74:77], v106, s[60:63], 0 offen offset:288 // (op 223) Load value
buffer_load_dwordx2 v[78:79], v106, s[60:63], 0 offen offset:304 // (op 223) Load value
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[174:179], v[110:115], a[32:35], v121, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1900) 
ds_read_b64 v[102:103], v5 offset:64528 // (op 2859) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[174:179], v[130:135], a[40:43], v121, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1824) 
v_add_u32 v107, 66048, v5 // (op 4077) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[174:179], v[140:145], a[44:47], v121, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1786) 
ds_read_b128 v[180:183], v107 // (op 4077) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[174:179], v[124:129], a[36:39], v121, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1862) 
ds_read_b64 v[184:185], v107 offset:16 // (op 4077) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[174:179], v[150:155], a[48:51], v121, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1748) 
s_barrier  // (op 14303) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[174:179], v[162:167], a[56:59], v121, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1672) 
buffer_load_dwordx4 v[80:83], v106, s[60:63], 0 offen offset:312 // (op 223) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[174:179], v[168:173], a[60:63], v121, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1634) 
buffer_load_dwordx2 v[84:85], v106, s[60:63], 0 offen offset:328 // (op 223) Load value



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[174:179], v[156:161], a[52:55], v121, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1710) 




v_add_u32 v15, 192, v15 // (op 13641) 
buffer_load_dwordx4 v[86:89], v61, s[48:51], 0 offen offset:288 // (op 211) Load value
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[98:103], v[110:115], a[64:67], v122, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 3118) 
buffer_load_dwordx2 v[90:91], v61, s[48:51], 0 offen offset:304 // (op 211) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[98:103], v[130:135], a[72:75], v122, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 3042) 
buffer_load_dwordx4 v[92:95], v61, s[48:51], 0 offen offset:312 // (op 211) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[98:103], v[140:145], a[76:79], v122, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 3004) 
buffer_load_dwordx2 v[96:97], v61, s[48:51], 0 offen offset:328 // (op 211) Load value

v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[98:103], v[124:129], a[68:71], v122, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 3080) 
s_barrier  // (op 14188) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[98:103], v[150:155], a[80:83], v122, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2966) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[98:103], v[162:167], a[88:91], v122, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2890) 
v_add_u32 v11, 192, v11 // (op 13577) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[98:103], v[168:173], a[92:95], v122, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2852) 
s_waitcnt vmcnt(26)
ds_write_b128 v14, v[22:25] // (op 13) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[98:103], v[156:161], a[84:87], v122, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2928) 
s_waitcnt vmcnt(25)
ds_write_b64 v14, v[26:27] offset:16 // (op 13) Store local data 

s_waitcnt vmcnt(24)
ds_write_b128 v14, v[28:31] offset:24 // (op 13) Store local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[180:185], v[110:115], a[96:99], v123, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 4336) 
s_waitcnt vmcnt(23)
ds_write_b64 v14, v[32:33] offset:40 // (op 13) Store local data 

s_waitcnt vmcnt(22)
ds_write_b128 v14, v[34:37] offset:96 // (op 13) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[180:185], v[130:135], a[104:107], v123, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 4260) 
s_waitcnt vmcnt(21)
ds_write_b64 v14, v[38:39] offset:112 // (op 13) Store local data 

s_waitcnt vmcnt(20)
ds_write_b128 v14, v[40:43] offset:120 // (op 13) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[180:185], v[140:145], a[108:111], v123, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 4222) 
s_waitcnt vmcnt(19)
ds_write_b64 v14, v[44:45] offset:136 // (op 13) Store local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[180:185], v[124:129], a[100:103], v123, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 4298) 
s_barrier  // (op 14175) 





v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[180:185], v[150:155], a[112:115], v123, v116 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 4184) 
s_waitcnt vmcnt(18)
ds_write_b128 v12, v[46:49] offset:24576 // (op 6) Store local data 

s_waitcnt vmcnt(17)
ds_write_b64 v12, v[50:51] offset:24592 // (op 6) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[180:185], v[162:167], a[120:123], v123, v118 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 4108) 
s_waitcnt vmcnt(16)
ds_write_b128 v12, v[52:55] offset:24600 // (op 6) Store local data 

s_waitcnt vmcnt(15)
ds_write_b64 v12, v[56:57] offset:24616 // (op 6) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[180:185], v[168:173], a[124:127], v123, v119 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 4070) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[180:185], v[156:161], a[116:119], v123, v117 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 4146) 




s_ashr_i64 s[0:1], s[42:43], 7 // (op 4035) 
s_mov_b32 s18, s0 // (op 4035) convert

s_and_b32 s19, s18, 4294967294 // (op 4035) 

s_cmp_lt_u32 s47, s19 // (op 4035) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__ForLoopTop_KLoop_4035 // (op 4035) Condition: Bottom (jump to RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__ForLoopTop_KLoop_4035 if true)
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__ForLoopBottom_KLoop_4035:













s_mul_hi_i32 s47, s16, s10 // (op 13509) 
v_lshrrev_b32 v10, 1, v1 // (op 13509) 
s_add_i32 s0, s47, s16 // (op 13509) 

v_and_b32 v11, -2, v1 // (op 13509) 
s_ashr_i32 s1, s0, 31 // (op 13509) 
v_sub_u32 v12, v1, v11 // (op 13509) 

s_cmp_eq_i32 s10, 0 // (op 13509) 
s_mov_b32 s47, scc // (op 13509) 
v_lshlrev_b32 v11, 6, v12 // (op 13509) 

s_cmp_eq_i32 s47, 0 // (op 13509) 
s_cselect_b32 s18, 0, -1 // (op 13509) 

s_and_b32 s19, s13, 31 // (op 13509) 
s_lshl_b32 s47, 1, s19 // (op 13509) 
s_add_i32 s22, s47, s18 // (op 13509) 


s_and_b32 s18, s1, s22 // (op 13509) 


s_add_i32 s1, s0, s18 // (op 13509) 


s_ashr_i32 s0, s1, s19 // (op 13509) 


s_lshl_b32 s1, s13, 24 // (op 13509) 
s_ashr_i32 s47, s1, 31 // (op 13509) 

s_xor_b32 s1, s0, s47 // (op 13509) 

s_sub_i32 s0, s1, s47 // (op 13509) 


s_mul_hi_i32 s1, s0, s14 // (op 13509) 
s_add_i32 s47, s1, s0 // (op 13509) 

s_ashr_i32 s1, s47, 31 // (op 13509) 
s_cmp_eq_i32 s14, 0 // (op 13509) 
s_mov_b32 s18, scc // (op 13509) 
s_cmp_eq_i32 s18, 0 // (op 13509) 
s_cselect_b32 s19, 0, -1 // (op 13509) 

s_and_b32 s18, s15, 31 // (op 13509) 
s_lshl_b32 s22, 1, s18 // (op 13509) 
s_add_i32 s23, s22, s19 // (op 13509) 


s_and_b32 s19, s1, s23 // (op 13509) 


s_add_i32 s1, s47, s19 // (op 13509) 


s_ashr_i32 s19, s1, s18 // (op 13509) 


s_lshl_b32 s1, s15, 24 // (op 13509) 
s_ashr_i32 s18, s1, 31 // (op 13509) 

s_xor_b32 s1, s19, s18 // (op 13509) 

s_sub_i32 s19, s1, s18 // (op 13509) 


s_mul_i32 s1, s0, s11 // (op 13509) 

s_sub_u32 s0, s16, s1 // (op 13509) 

s_mul_i32 s1, s19, s11 // (op 13509) 
s_add_u32 s1, s1, s0 // (op 13509) 


s_mul_hi_i32 s0, s16, s9 // (op 13509) 
s_add_i32 s47, s0, s16 // (op 13509) 

s_ashr_i32 s0, s47, 31 // (op 13509) 
s_cmp_eq_i32 s9, 0 // (op 13509) 
s_mov_b32 s18, scc // (op 13509) 
s_cmp_eq_i32 s18, 0 // (op 13509) 
s_cselect_b32 s19, 0, -1 // (op 13509) 

s_and_b32 s18, s45, 31 // (op 13509) 
s_lshl_b32 s22, 1, s18 // (op 13509) 
s_add_i32 s23, s22, s19 // (op 13509) 


s_and_b32 s19, s0, s23 // (op 13509) 


s_add_i32 s0, s47, s19 // (op 13509) 


s_ashr_i32 s19, s0, s18 // (op 13509) 


s_lshl_b32 s0, s45, 24 // (op 13509) 
s_ashr_i32 s18, s0, 31 // (op 13509) 

s_xor_b32 s0, s19, s18 // (op 13509) 

s_sub_i32 s19, s0, s18 // (op 13509) 


s_add_u32 s22, s6, 256 // (op 13509) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13509) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13509) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13509) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 13509) 

s_mul_i32 s0, s11, s22 // (op 13509) 

s_mul_i32 s18, s8, s0 // (op 13509) 

s_mul_i32 s0, s19, s18 // (op 13509) 

s_sub_u32 s18, s16, s0 // (op 13509) 

s_mov_b32 s0, s18 // (op 13509) 
s_mul_hi_i32 s47, s0, s12 // (op 13509) 
s_add_i32 s22, s47, s0 // (op 13509) 


s_ashr_i32 s0, s22, 31 // (op 13509) 
s_cmp_eq_i32 s12, 0 // (op 13509) 
s_mov_b32 s47, scc // (op 13509) 
s_cmp_eq_i32 s47, 0 // (op 13509) 
s_cselect_b32 s23, 0, -1 // (op 13509) 

s_and_b32 s47, s44, 31 // (op 13509) 
s_lshl_b32 s24, 1, s47 // (op 13509) 
s_add_i32 s25, s24, s23 // (op 13509) 


s_and_b32 s23, s0, s25 // (op 13509) 


s_add_i32 s0, s22, s23 // (op 13509) 


s_ashr_i32 s22, s0, s47 // (op 13509) 


s_lshl_b32 s0, s44, 24 // (op 13509) 
s_ashr_i32 s47, s0, 31 // (op 13509) 

s_xor_b32 s0, s22, s47 // (op 13509) 

s_sub_i32 s22, s0, s47 // (op 13509) 


s_mul_i32 s0, s46, s11 // (op 13509) 
s_add_u32 s24, s20, 128 // (op 13509) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 13509) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 13509) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 13509) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 7 // (op 13509) 

s_sub_i32 s23, s24, s0 // (op 13509) 

s_mul_i32 s47, s22, s23 // (op 13509) 


s_sub_u32 s22, s18, s47 // (op 13509) 


s_add_u32 s18, s22, s0 // (op 13509) 


s_cmp_eq_i32 s19, 0 // (op 13509) 
s_mov_b32 s0, scc // (op 13509) 

s_cmp_eq_i32 s0, 0 // (op 13509) 
s_cselect_b32 s19, s18, s1 // (op 13509) 



v_lshl_add_u32 v12, s19, 7, v10 // (op 13509) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 13509) 
s_lshr_b32 s47, s0, 1 // (op 13509) 

v_lshl_add_u32 v10, s47, 8, v11 // (op 13509) 


v_mov_b32 v61, 0 // (op 13509) convert
v_mov_b32 v60, v10 // (op 13509) convert

v_mul_lo_u32 v11, v12, s27 // (op 13509) most significant: low of low * high
v_mul_hi_u32 v13, v12, s26 // (op 13509) most significant: high of low * low
v_mul_lo_u32 v10, v12, s26 // (op 13509) least significant: low of low * low
v_add_u32 v11, v11, v13 // (op 13509) most significant: sum

v_add_co_u32 v10, s[0:1], v10, v60 // (op 13509) least significant half
v_addc_co_u32 v11, s[0:1], v11, v61, s[0:1] // (op 13509) most significant half



v_mul_lo_u32 v61, v11, 6 // (op 13509) most significant: low of high * low
v_mul_hi_u32 v12, v10, 6 // (op 13509) most significant: high of low * low
v_mul_lo_u32 v60, v10, 6 // (op 13509) least significant: low of low * low
v_add_u32 v61, v61, v12 // (op 13509) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[60:61] // (op 13509) 

v_mov_b32 v12, v10 // (op 13509) convert

s_mul_i32 s1, s21, 6 // (op 13511) most significant: low of high * low
s_mul_i32 s0, s20, 6 // (op 13511) least significant
s_mul_hi_u32 s47, s20, 6 // (op 13511) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13511) most significant: sum

s_mul_i32 s19, s26, s1 // (op 13511) most significant: low of low * high
s_mul_i32 s47, s27, s0 // (op 13511) most significant: low of high * low
s_mul_i32 s18, s26, s0 // (op 13511) least significant
s_mul_hi_u32 s22, s26, s0 // (op 13511) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 13511) most significant: sum
s_add_u32 s19, s19, s22 // (op 13511) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 13511) 

s_mov_b32 s48, s28 // (op 13511) reg expression
s_mov_b32 s49, s29 // (op 13511) reg expression
s_mov_b32 s50, s0 // (op 13511) reg expression
s_mov_b32 s51, 131072 // (op 13511) call()

s_mul_hi_i32 s47, s16, s10 // (op 13607) 
v_and_b32 v10, 4294967294, v1 // (op 13607) 
s_add_i32 s0, s47, s16 // (op 13607) 

v_and_b32 v11, -2, v1 // (op 13607) 
s_ashr_i32 s1, s0, 31 // (op 13607) 
v_sub_u32 v13, v1, v11 // (op 13607) 

s_cmp_eq_i32 s10, 0 // (op 13607) 
s_mov_b32 s47, scc // (op 13607) 
v_lshlrev_b32 v11, 6, v13 // (op 13607) 

s_cmp_eq_i32 s47, 0 // (op 13607) 
s_cselect_b32 s18, 0, -1 // (op 13607) 

s_and_b32 s19, s13, 31 // (op 13607) 
s_lshl_b32 s47, 1, s19 // (op 13607) 
s_add_i32 s22, s47, s18 // (op 13607) 


s_and_b32 s18, s1, s22 // (op 13607) 


s_add_i32 s1, s0, s18 // (op 13607) 


s_ashr_i32 s0, s1, s19 // (op 13607) 


s_lshl_b32 s1, s13, 24 // (op 13607) 
s_ashr_i32 s47, s1, 31 // (op 13607) 

s_xor_b32 s1, s0, s47 // (op 13607) 

s_sub_i32 s0, s1, s47 // (op 13607) 


s_mul_hi_i32 s1, s0, s14 // (op 13607) 
s_add_i32 s47, s1, s0 // (op 13607) 

s_ashr_i32 s1, s47, 31 // (op 13607) 
s_cmp_eq_i32 s14, 0 // (op 13607) 
s_mov_b32 s18, scc // (op 13607) 
s_cmp_eq_i32 s18, 0 // (op 13607) 
s_cselect_b32 s19, 0, -1 // (op 13607) 

s_and_b32 s18, s15, 31 // (op 13607) 
s_lshl_b32 s22, 1, s18 // (op 13607) 
s_add_i32 s23, s22, s19 // (op 13607) 


s_and_b32 s19, s1, s23 // (op 13607) 


s_add_i32 s1, s47, s19 // (op 13607) 


s_ashr_i32 s19, s1, s18 // (op 13607) 


s_lshl_b32 s1, s15, 24 // (op 13607) 
s_ashr_i32 s18, s1, 31 // (op 13607) 

s_xor_b32 s1, s19, s18 // (op 13607) 

s_sub_i32 s19, s1, s18 // (op 13607) 


s_add_u32 s22, s6, 256 // (op 13607) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13607) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13607) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13607) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 13607) 

s_mul_i32 s1, s19, s22 // (op 13607) 

s_sub_i32 s47, s0, s1 // (op 13607) 


s_mul_hi_i32 s0, s16, s9 // (op 13607) 
s_add_i32 s1, s0, s16 // (op 13607) 

s_ashr_i32 s0, s1, 31 // (op 13607) 
s_cmp_eq_i32 s9, 0 // (op 13607) 
s_mov_b32 s18, scc // (op 13607) 
s_cmp_eq_i32 s18, 0 // (op 13607) 
s_cselect_b32 s19, 0, -1 // (op 13607) 

s_and_b32 s18, s45, 31 // (op 13607) 
s_lshl_b32 s24, 1, s18 // (op 13607) 
s_add_i32 s25, s24, s19 // (op 13607) 


s_and_b32 s19, s0, s25 // (op 13607) 


s_add_i32 s0, s1, s19 // (op 13607) 


s_ashr_i32 s1, s0, s18 // (op 13607) 


s_lshl_b32 s0, s45, 24 // (op 13607) 
s_ashr_i32 s18, s0, 31 // (op 13607) 

s_xor_b32 s0, s1, s18 // (op 13607) 

s_sub_i32 s1, s0, s18 // (op 13607) 


s_mul_i32 s0, s11, s22 // (op 13607) 

s_mul_i32 s18, s8, s0 // (op 13607) 

s_mul_i32 s0, s1, s18 // (op 13607) 

s_sub_u32 s18, s16, s0 // (op 13607) 

s_mul_hi_i32 s0, s18, s12 // (op 13607) 
s_add_i32 s19, s0, s18 // (op 13607) 


s_ashr_i32 s0, s19, 31 // (op 13607) 
s_cmp_eq_i32 s12, 0 // (op 13607) 
s_mov_b32 s18, scc // (op 13607) 
s_cmp_eq_i32 s18, 0 // (op 13607) 
s_cselect_b32 s22, 0, -1 // (op 13607) 

s_and_b32 s18, s44, 31 // (op 13607) 
s_lshl_b32 s23, 1, s18 // (op 13607) 
s_add_i32 s24, s23, s22 // (op 13607) 


s_and_b32 s25, s0, s24 // (op 13607) 


s_add_i32 s0, s19, s25 // (op 13607) 


s_ashr_i32 s19, s0, s18 // (op 13607) 


s_lshl_b32 s0, s44, 24 // (op 13607) 
s_ashr_i32 s18, s0, 31 // (op 13607) 

s_xor_b32 s0, s19, s18 // (op 13607) 

s_sub_i32 s19, s0, s18 // (op 13607) 


s_cmp_eq_i32 s1, 0 // (op 13607) 
s_mov_b32 s0, scc // (op 13607) 

s_cmp_eq_i32 s0, 0 // (op 13607) 
s_cselect_b32 s1, s19, s47 // (op 13607) 



v_lshl_add_u32 v13, s1, 8, v10 // (op 13607) 


v_mul_lo_u32 v61, v13, s31 // (op 13607) most significant: low of low * high
v_mul_hi_u32 v10, v13, s30 // (op 13607) most significant: high of low * low
v_mul_lo_u32 v60, v13, s30 // (op 13607) least significant: low of low * low
v_add_u32 v61, v61, v10 // (op 13607) most significant: sum


s_ashr_i64 s[0:1], s[42:43], 7 // (op 13607) 
s_lshr_b32 s47, s0, 1 // (op 13607) 

v_lshl_add_u32 v10, s47, 8, v11 // (op 13607) 


v_mov_b32 v17, 0 // (op 13607) convert
v_mov_b32 v16, v10 // (op 13607) convert

v_add_co_u32 v10, s[0:1], v16, v60 // (op 13607) least significant half
v_addc_co_u32 v11, s[0:1], v17, v61, s[0:1] // (op 13607) most significant half



v_mul_lo_u32 v61, v11, 6 // (op 13607) most significant: low of high * low
v_mul_hi_u32 v13, v10, 6 // (op 13607) most significant: high of low * low
v_mul_lo_u32 v60, v10, 6 // (op 13607) least significant: low of low * low
v_add_u32 v61, v61, v13 // (op 13607) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[60:61] // (op 13607) 

v_mov_b32 v13, v10 // (op 13607) convert

s_mul_i32 s1, s7, 6 // (op 13609) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 13609) least significant
s_mul_hi_u32 s47, s6, 6 // (op 13609) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 13609) most significant: sum

s_mul_i32 s19, s30, s1 // (op 13609) most significant: low of low * high
s_mul_i32 s47, s31, s0 // (op 13609) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 13609) least significant
s_mul_hi_u32 s22, s30, s0 // (op 13609) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 13609) most significant: sum
s_add_u32 s19, s19, s22 // (op 13609) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 13609) 

s_mov_b32 s60, s32 // (op 13609) reg expression
s_mov_b32 s61, s33 // (op 13609) reg expression
s_mov_b32 s62, s0 // (op 13609) reg expression
s_mov_b32 s63, 131072 // (op 13609) call()

v_and_b32 v10, -2, v1 // (op 13690) 
v_sub_u32 v11, v1, v10 // (op 13690) 

v_lshlrev_b32 v10, 6, v11 // (op 13690) 

v_lshrrev_b32 v11, 1, v1 // (op 13690) 
v_lshl_add_u32 v14, v11, 7, v10 // (op 13690) 


v_mul_lo_u32 v10, v14, 6 // (op 13690) 

v_lshrrev_b32 v11, 3, v10 // (op 13690) 

v_and_b32 v10, -2, v1 // (op 13750) 
v_sub_u32 v14, v1, v10 // (op 13750) 

v_lshlrev_b32 v10, 6, v14 // (op 13750) 

v_lshrrev_b32 v14, 1, v1 // (op 13750) 
v_lshl_add_u32 v15, v14, 8, v10 // (op 13750) 


v_mul_lo_u32 v10, v15, 6 // (op 13750) 

v_lshrrev_b32 v14, 3, v10 // (op 13750) 

s_mul_hi_i32 s47, s16, s10 // (op 13999) 
v_and_b32 v10, 63, v1 // (op 13999) 
s_add_i32 s0, s47, s16 // (op 13999) 

v_lshrrev_b32 v15, 4, v10 // (op 13999) 
s_ashr_i32 s1, s0, 31 // (op 13999) 
v_and_b32 v16, 3, v15 // (op 13999) 

s_cmp_eq_i32 s10, 0 // (op 13999) 
s_mov_b32 s47, scc // (op 13999) 
v_and_b32 v15, 15, v1 // (op 13999) 
s_cmp_eq_i32 s47, 0 // (op 13999) 
s_cselect_b32 s18, 0, -1 // (op 13999) 

v_lshl_add_u32 v17, v16, 4, v15 // (op 13999) 


s_and_b32 s19, s13, 31 // (op 13999) 
v_lshrrev_b32 v15, 7, v1 // (op 13999) 
s_lshl_b32 s47, 1, s19 // (op 13999) 
v_lshl_add_u32 v16, v15, 6, v17 // (op 13999) 


s_add_i32 s22, s47, s18 // (op 13999) 


v_lshrrev_b32 v15, 6, v10 // (op 13999) 

s_and_b32 s18, s1, s22 // (op 13999) 


v_lshlrev_b32 v10, 2, v15 // (op 13999) 

s_add_i32 s1, s0, s18 // (op 13999) 


s_ashr_i32 s0, s1, s19 // (op 13999) 


s_lshl_b32 s1, s13, 24 // (op 13999) 
s_ashr_i32 s47, s1, 31 // (op 13999) 

s_xor_b32 s1, s0, s47 // (op 13999) 

s_sub_i32 s0, s1, s47 // (op 13999) 


s_mul_hi_i32 s1, s0, s14 // (op 13999) 
s_add_i32 s47, s1, s0 // (op 13999) 

s_ashr_i32 s1, s47, 31 // (op 13999) 
s_cmp_eq_i32 s14, 0 // (op 13999) 
s_mov_b32 s18, scc // (op 13999) 
s_cmp_eq_i32 s18, 0 // (op 13999) 
s_cselect_b32 s19, 0, -1 // (op 13999) 

s_and_b32 s18, s15, 31 // (op 13999) 
s_lshl_b32 s22, 1, s18 // (op 13999) 
s_add_i32 s23, s22, s19 // (op 13999) 


s_and_b32 s19, s1, s23 // (op 13999) 


s_add_i32 s1, s47, s19 // (op 13999) 


s_ashr_i32 s19, s1, s18 // (op 13999) 


s_lshl_b32 s1, s15, 24 // (op 13999) 
s_ashr_i32 s18, s1, 31 // (op 13999) 

s_xor_b32 s1, s19, s18 // (op 13999) 

s_sub_i32 s19, s1, s18 // (op 13999) 


s_mul_i32 s1, s0, s11 // (op 13999) 

s_sub_u32 s0, s16, s1 // (op 13999) 

s_mul_i32 s1, s19, s11 // (op 13999) 
s_add_u32 s1, s1, s0 // (op 13999) 


s_mul_hi_i32 s0, s16, s9 // (op 13999) 
s_add_i32 s47, s0, s16 // (op 13999) 

s_ashr_i32 s0, s47, 31 // (op 13999) 
s_cmp_eq_i32 s9, 0 // (op 13999) 
s_mov_b32 s18, scc // (op 13999) 
s_cmp_eq_i32 s18, 0 // (op 13999) 
s_cselect_b32 s19, 0, -1 // (op 13999) 

s_and_b32 s18, s45, 31 // (op 13999) 
s_lshl_b32 s22, 1, s18 // (op 13999) 
s_add_i32 s23, s22, s19 // (op 13999) 


s_and_b32 s19, s0, s23 // (op 13999) 


s_add_i32 s0, s47, s19 // (op 13999) 


s_ashr_i32 s19, s0, s18 // (op 13999) 


s_lshl_b32 s0, s45, 24 // (op 13999) 
s_ashr_i32 s18, s0, 31 // (op 13999) 

s_xor_b32 s0, s19, s18 // (op 13999) 

s_sub_i32 s19, s0, s18 // (op 13999) 


s_add_u32 s22, s6, 256 // (op 13999) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 13999) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 13999) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 13999) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 13999) 

s_mul_i32 s0, s11, s22 // (op 13999) 

s_mul_i32 s18, s8, s0 // (op 13999) 

s_mul_i32 s0, s19, s18 // (op 13999) 

s_sub_u32 s18, s16, s0 // (op 13999) 

s_mov_b32 s0, s18 // (op 13999) 
s_mul_hi_i32 s47, s0, s12 // (op 13999) 
s_add_i32 s22, s47, s0 // (op 13999) 


s_ashr_i32 s0, s22, 31 // (op 13999) 
s_cmp_eq_i32 s12, 0 // (op 13999) 
s_mov_b32 s47, scc // (op 13999) 
s_cmp_eq_i32 s47, 0 // (op 13999) 
s_cselect_b32 s23, 0, -1 // (op 13999) 

s_and_b32 s47, s44, 31 // (op 13999) 
s_lshl_b32 s24, 1, s47 // (op 13999) 
s_add_i32 s25, s24, s23 // (op 13999) 


s_and_b32 s23, s0, s25 // (op 13999) 


s_add_i32 s0, s22, s23 // (op 13999) 


s_ashr_i32 s22, s0, s47 // (op 13999) 


s_lshl_b32 s0, s44, 24 // (op 13999) 
s_ashr_i32 s47, s0, 31 // (op 13999) 

s_xor_b32 s0, s22, s47 // (op 13999) 

s_sub_i32 s22, s0, s47 // (op 13999) 


s_mul_i32 s0, s46, s11 // (op 13999) 
s_add_u32 s24, s20, 128 // (op 13999) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 13999) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 13999) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 13999) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 7 // (op 13999) 

s_sub_i32 s23, s24, s0 // (op 13999) 

s_mul_i32 s47, s22, s23 // (op 13999) 


s_sub_u32 s22, s18, s47 // (op 13999) 


s_add_u32 s18, s22, s0 // (op 13999) 


s_cmp_eq_i32 s19, 0 // (op 13999) 
s_mov_b32 s0, scc // (op 13999) 

s_cmp_eq_i32 s0, 0 // (op 13999) 
s_cselect_b32 s19, s18, s1 // (op 13999) 



v_lshl_add_u32 v15, s19, 7, v16 // (op 13999) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 13999) 
s_lshr_b32 s47, s0, 1 // (op 13999) 

v_lshl_add_u32 v16, s47, 3, v10 // (op 13999) 


v_mul_lo_u32 v10, v15, s34 // (op 13999) 
v_add_u32 v10, v10, v16 // (op 13999) 


s_mul_i32 s1, s34, s21 // (op 14001) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 14001) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 14001) least significant
s_mul_hi_u32 s18, s34, s20 // (op 14001) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 14001) most significant: sum
s_add_u32 s1, s1, s18 // (op 14001) most significant: sum


s_mov_b32 s64, s36 // (op 14001) reg expression
s_mov_b32 s65, s37 // (op 14001) reg expression
s_mov_b32 s66, s0 // (op 14001) reg expression
s_mov_b32 s67, 131072 // (op 14001) call()

s_mul_hi_i32 s47, s16, s10 // (op 14026) 
v_and_b32 v15, 63, v1 // (op 14026) 
s_add_i32 s0, s47, s16 // (op 14026) 

v_lshrrev_b32 v16, 4, v15 // (op 14026) 
s_ashr_i32 s1, s0, 31 // (op 14026) 
v_and_b32 v17, 3, v16 // (op 14026) 

s_cmp_eq_i32 s10, 0 // (op 14026) 
s_mov_b32 s47, scc // (op 14026) 
v_and_b32 v16, 15, v1 // (op 14026) 
s_cmp_eq_i32 s47, 0 // (op 14026) 
s_cselect_b32 s18, 0, -1 // (op 14026) 

v_lshl_add_u32 v22, v17, 4, v16 // (op 14026) 


s_and_b32 s19, s13, 31 // (op 14026) 
v_lshrrev_b32 v16, 6, v1 // (op 14026) 
s_lshl_b32 s47, 1, s19 // (op 14026) 
v_and_b32 v17, 1, v16 // (op 14026) 

s_add_i32 s22, s47, s18 // (op 14026) 


v_lshl_add_u32 v16, v17, 7, v22 // (op 14026) 


s_and_b32 s18, s1, s22 // (op 14026) 


v_lshrrev_b32 v17, 6, v15 // (op 14026) 

s_add_i32 s1, s0, s18 // (op 14026) 


v_lshlrev_b32 v15, 2, v17 // (op 14026) 

s_ashr_i32 s0, s1, s19 // (op 14026) 


s_lshl_b32 s1, s13, 24 // (op 14026) 
s_ashr_i32 s47, s1, 31 // (op 14026) 

s_xor_b32 s1, s0, s47 // (op 14026) 

s_sub_i32 s0, s1, s47 // (op 14026) 


s_mul_hi_i32 s1, s0, s14 // (op 14026) 
s_add_i32 s47, s1, s0 // (op 14026) 

s_ashr_i32 s1, s47, 31 // (op 14026) 
s_cmp_eq_i32 s14, 0 // (op 14026) 
s_mov_b32 s18, scc // (op 14026) 
s_cmp_eq_i32 s18, 0 // (op 14026) 
s_cselect_b32 s19, 0, -1 // (op 14026) 

s_and_b32 s18, s15, 31 // (op 14026) 
s_lshl_b32 s22, 1, s18 // (op 14026) 
s_add_i32 s23, s22, s19 // (op 14026) 


s_and_b32 s19, s1, s23 // (op 14026) 


s_add_i32 s1, s47, s19 // (op 14026) 


s_ashr_i32 s19, s1, s18 // (op 14026) 


s_lshl_b32 s1, s15, 24 // (op 14026) 
s_ashr_i32 s18, s1, 31 // (op 14026) 

s_xor_b32 s1, s19, s18 // (op 14026) 

s_sub_i32 s19, s1, s18 // (op 14026) 


s_add_u32 s22, s6, 256 // (op 14026) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 14026) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 14026) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 14026) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 14026) 

s_mul_i32 s1, s19, s22 // (op 14026) 

s_sub_i32 s47, s0, s1 // (op 14026) 


s_mul_hi_i32 s0, s16, s9 // (op 14026) 
s_add_i32 s1, s0, s16 // (op 14026) 

s_ashr_i32 s0, s1, 31 // (op 14026) 
s_cmp_eq_i32 s9, 0 // (op 14026) 
s_mov_b32 s18, scc // (op 14026) 
s_cmp_eq_i32 s18, 0 // (op 14026) 
s_cselect_b32 s19, 0, -1 // (op 14026) 

s_and_b32 s18, s45, 31 // (op 14026) 
s_lshl_b32 s24, 1, s18 // (op 14026) 
s_add_i32 s25, s24, s19 // (op 14026) 


s_and_b32 s19, s0, s25 // (op 14026) 


s_add_i32 s0, s1, s19 // (op 14026) 


s_ashr_i32 s1, s0, s18 // (op 14026) 


s_lshl_b32 s0, s45, 24 // (op 14026) 
s_ashr_i32 s18, s0, 31 // (op 14026) 

s_xor_b32 s0, s1, s18 // (op 14026) 

s_sub_i32 s1, s0, s18 // (op 14026) 


s_mul_i32 s0, s11, s22 // (op 14026) 

s_mul_i32 s18, s8, s0 // (op 14026) 

s_mul_i32 s0, s1, s18 // (op 14026) 

s_sub_u32 s18, s16, s0 // (op 14026) 

s_mul_hi_i32 s0, s18, s12 // (op 14026) 
s_add_i32 s19, s0, s18 // (op 14026) 


s_ashr_i32 s0, s19, 31 // (op 14026) 
s_cmp_eq_i32 s12, 0 // (op 14026) 
s_mov_b32 s18, scc // (op 14026) 
s_cmp_eq_i32 s18, 0 // (op 14026) 
s_cselect_b32 s22, 0, -1 // (op 14026) 

s_and_b32 s18, s44, 31 // (op 14026) 
s_lshl_b32 s23, 1, s18 // (op 14026) 
s_add_i32 s24, s23, s22 // (op 14026) 


s_and_b32 s25, s0, s24 // (op 14026) 


s_add_i32 s0, s19, s25 // (op 14026) 


s_ashr_i32 s19, s0, s18 // (op 14026) 


s_lshl_b32 s0, s44, 24 // (op 14026) 
s_ashr_i32 s18, s0, 31 // (op 14026) 

s_xor_b32 s0, s19, s18 // (op 14026) 

s_sub_i32 s19, s0, s18 // (op 14026) 


s_cmp_eq_i32 s1, 0 // (op 14026) 
s_mov_b32 s0, scc // (op 14026) 

s_cmp_eq_i32 s0, 0 // (op 14026) 
s_cselect_b32 s1, s19, s47 // (op 14026) 



v_lshl_add_u32 v17, s1, 8, v16 // (op 14026) 


v_mul_lo_u32 v16, v17, s38 // (op 14026) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 14026) 
s_lshr_b32 s47, s0, 1 // (op 14026) 

v_lshl_add_u32 v17, s47, 3, v15 // (op 14026) 


v_add_u32 v15, v17, v16 // (op 14026) 


s_mul_i32 s1, s38, s7 // (op 14028) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 14028) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 14028) least significant
s_mul_hi_u32 s18, s38, s6 // (op 14028) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 14028) most significant: sum
s_add_u32 s1, s1, s18 // (op 14028) most significant: sum


s_mov_b32 s68, s40 // (op 14028) reg expression
s_mov_b32 s69, s41 // (op 14028) reg expression
s_mov_b32 s70, s0 // (op 14028) reg expression
s_mov_b32 s71, 131072 // (op 14028) call()

s_ashr_i64 s[0:1], s[42:43], 7 // (op 4014) 
s_and_b32 s47, s0, 4294967294 // (op 4014) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 4013) 
s_mov_b32 s18, s0 // (op 4013) convert

s_cmp_lt_u32 s47, s18 // (op 4013) 






s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_4013 // (op 4013) Condition: Top (jump to RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_4013 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__ForLoopTop_KLoopTail_4013:


s_add_u32 s47, s47, 1 // (op 4015) 
v_mov_b32 v16, v12 // (op 161) 

buffer_load_dwordx4 v[18:21], v16, s[48:51], 0 offen // (op 161) Load value
buffer_load_dwordx2 v[22:23], v16, s[48:51], 0 offen offset:16 // (op 161) Load value
buffer_load_dwordx4 v[24:27], v16, s[48:51], 0 offen offset:24 // (op 161) Load value
buffer_load_dwordx2 v[28:29], v16, s[48:51], 0 offen offset:40 // (op 161) Load value

v_add_u32 v12, 96, v12 // (op 13499) 

s_barrier  // (op 14178) 
s_mul_i32 s1, s31, 6 // (op 173) most significant: low of high * low



s_mul_i32 s0, s30, 6 // (op 173) least significant
s_mul_hi_u32 s18, s30, 6 // (op 173) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 173) most significant: sum

v_lshrrev_b64 v[16:17], 3, s[0:1] // (op 173) 


v_mov_b32 v30, v13 // (op 173) 
v_mov_b32 v31, v30 // (op 173) 
buffer_load_dwordx4 v[32:35], v31, s[60:63], 0 offen // (op 173) Load value
buffer_load_dwordx2 v[36:37], v31, s[60:63], 0 offen offset:16 // (op 173) Load value
s_waitcnt vmcnt(5)
ds_write_b128 v11, v[18:21] offset:24576 // (op 162) Store local data 
s_waitcnt vmcnt(4)
ds_write_b64 v11, v[22:23] offset:24592 // (op 162) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v11, v[24:27] offset:24600 // (op 162) Store local data 
s_waitcnt vmcnt(2)
ds_write_b64 v11, v[28:29] offset:24616 // (op 162) Store local data 
buffer_load_dwordx4 v[38:41], v31, s[60:63], 0 offen offset:24 // (op 173) Load value

buffer_load_dwordx2 v[42:43], v31, s[60:63], 0 offen offset:40 // (op 173) Load value
v_add_u32 v31, v31, v16 // (op 173) 
buffer_load_dwordx4 v[44:47], v31, s[60:63], 0 offen // (op 173) Load value
buffer_load_dwordx2 v[48:49], v31, s[60:63], 0 offen offset:16 // (op 173) Load value
buffer_load_dwordx4 v[50:53], v31, s[60:63], 0 offen offset:24 // (op 173) Load value
buffer_load_dwordx2 v[54:55], v31, s[60:63], 0 offen offset:40 // (op 173) Load value



v_add_u32 v13, 96, v13 // (op 13599) 
s_barrier  // (op 14191) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 14181) 






ds_read_b128 v[16:19], v3 offset:24576 // (op 164) Load local data 
ds_read_b64 v[20:21], v3 offset:24592 // (op 164) Load local data 
s_waitcnt vmcnt(7)
ds_write_b128 v14, v[32:35] // (op 172) Store local data 
s_waitcnt vmcnt(6)
ds_write_b64 v14, v[36:37] offset:16 // (op 172) Store local data 
s_waitcnt vmcnt(5)
ds_write_b128 v14, v[38:41] offset:24 // (op 172) Store local data 
s_waitcnt vmcnt(4)
ds_write_b64 v14, v[42:43] offset:40 // (op 172) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v14, v[44:47] offset:96 // (op 172) Store local data 
s_waitcnt vmcnt(2)
ds_write_b64 v14, v[48:49] offset:112 // (op 172) Store local data 
s_waitcnt vmcnt(1)
ds_write_b128 v14, v[50:53] offset:120 // (op 172) Store local data 
s_waitcnt vmcnt(0)
ds_write_b64 v14, v[54:55] offset:136 // (op 172) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 14195) 



ds_read_b128 v[22:25], v4 // (op 166) Load local data 
ds_read_b64 v[26:27], v4 offset:16 // (op 166) Load local data 


v_mov_b32 v28, v10 // (op 168) 
v_mov_b32 v29, v28 // (op 168) reg expression

buffer_load_dwordx2 v[30:31], v29, s[64:67], 0 offen // (op 168) Load value


v_add_u32 v10, 4, v10 // (op 13989) 


v_mov_b32 v28, v15 // (op 170) 
v_mov_b32 v29, v28 // (op 170) reg expression

buffer_load_dwordx2 v[32:33], v29, s[68:71], 0 offen // (op 170) Load value

s_waitcnt vmcnt(1)
v_bfe_u32 v34, v30, 0, 8 // (op 12464) 



v_bfe_u32 v35, v30, 8, 8 // (op 12464) 
v_bfe_u32 v36, v30, 16, 8 // (op 12464) 
v_bfe_u32 v37, v30, 24, 8 // (op 12464) 
v_permlane16_swap_b32 v34, v35 // (op 12464) 
ds_read_b128 v[38:41], v4 offset:1536 // (op 295) Load local data 
v_permlane16_swap_b32 v36, v37 // (op 12464) 
s_nop 1
v_permlane32_swap_b32 v34, v36 // (op 12464) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v35, v37 // (op 12464) 
ds_read_b64 v[42:43], v4 offset:1552 // (op 295) Load local data 




ds_read_b128 v[44:47], v4 offset:3072 // (op 431) Load local data 
ds_read_b64 v[48:49], v4 offset:3088 // (op 431) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v50, v32, 0, 8 // (op 12510) 



v_bfe_u32 v51, v32, 8, 8 // (op 12510) 
v_bfe_u32 v52, v32, 16, 8 // (op 12510) 
v_bfe_u32 v53, v32, 24, 8 // (op 12510) 
v_permlane16_swap_b32 v50, v51 // (op 12510) 
s_nop 0
v_permlane16_swap_b32 v52, v53 // (op 12510) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v50, v52 // (op 12510) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v51, v53 // (op 12510) 
ds_read_b128 v[54:57], v4 offset:4608 // (op 567) Load local data 

ds_read_b64 v[58:59], v4 offset:4624 // (op 567) Load local data 



s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[16:21], v[22:27], a[0:3], v34, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 171) 
ds_read_b128 v[60:63], v4 offset:6144 // (op 703) Load local data 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[16:21], v[38:43], a[4:7], v34, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 298) 
ds_read_b64 v[64:65], v4 offset:6160 // (op 703) Load local data 


s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[16:21], v[44:49], a[8:11], v34, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 434) 
s_lshl_b32 s0, s38, 6 // (op 708) 
v_add_u32 v29, v28, s0 // (op 708) 


buffer_load_dwordx2 v[30:31], v29, s[68:71], 0 offen // (op 708) Load value

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[16:21], v[54:59], a[12:15], v34, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 570) 




v_add_u32 v15, 4, v15 // (op 14016) 
ds_read_b128 v[66:69], v4 offset:7680 // (op 839) Load local data 
ds_read_b64 v[70:71], v4 offset:7696 // (op 839) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v72, v30, 0, 8 // (op 12524) 



v_bfe_u32 v73, v30, 8, 8 // (op 12524) 
v_bfe_u32 v74, v30, 16, 8 // (op 12524) 
v_bfe_u32 v75, v30, 24, 8 // (op 12524) 
v_permlane16_swap_b32 v72, v73 // (op 12524) 
s_nop 0
v_permlane16_swap_b32 v74, v75 // (op 12524) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v72, v74 // (op 12524) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v73, v75 // (op 12524) 
ds_read_b128 v[76:79], v4 offset:9216 // (op 975) Load local data 

ds_read_b64 v[80:81], v4 offset:9232 // (op 975) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[16:21], v[60:65], a[16:19], v34, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 706) 
ds_read_b128 v[28:31], v4 offset:10752 // (op 1111) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[16:21], v[66:71], a[20:23], v34, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 842) 
ds_read_b64 v[32:33], v4 offset:10768 // (op 1111) Load local data 



s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[16:21], v[76:81], a[24:27], v34, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 978) 
ds_read_b128 v[82:85], v3 offset:26112 // (op 1421) Load local data 
ds_read_b64 v[86:87], v3 offset:26128 // (op 1421) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[16:21], v[28:33], a[28:31], v34, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1114) 




ds_read_b128 v[16:19], v3 offset:27648 // (op 2639) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[82:87], v[22:27], a[32:35], v35, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1425) 
ds_read_b64 v[20:21], v3 offset:27664 // (op 2639) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[82:87], v[44:49], a[40:43], v35, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1479) 
ds_read_b128 v[88:91], v3 offset:29184 // (op 3857) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[82:87], v[54:59], a[44:47], v35, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1506) 
ds_read_b64 v[92:93], v3 offset:29200 // (op 3857) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[82:87], v[38:43], a[36:39], v35, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1452) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[82:87], v[60:65], a[48:51], v35, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1533) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[82:87], v[76:81], a[56:59], v35, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1587) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[82:87], v[28:33], a[60:63], v35, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1614) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[82:87], v[66:71], a[52:55], v35, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1560) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[16:21], v[22:27], a[64:67], v36, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2643) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[16:21], v[44:49], a[72:75], v36, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2697) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[16:21], v[54:59], a[76:79], v36, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2724) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[16:21], v[38:43], a[68:71], v36, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2670) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[16:21], v[60:65], a[80:83], v36, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2751) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[16:21], v[76:81], a[88:91], v36, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2805) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[16:21], v[28:33], a[92:95], v36, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2832) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[16:21], v[66:71], a[84:87], v36, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2778) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[88:93], v[22:27], a[96:99], v37, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 3861) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[88:93], v[44:49], a[104:107], v37, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 3915) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[88:93], v[54:59], a[108:111], v37, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 3942) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[88:93], v[38:43], a[100:103], v37, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 3888) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[88:93], v[60:65], a[112:115], v37, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 3969) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[88:93], v[76:81], a[120:123], v37, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 4023) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[88:93], v[28:33], a[124:127], v37, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 4050) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[88:93], v[66:71], a[116:119], v37, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 3996) 




s_ashr_i64 s[0:1], s[42:43], 7 // (op 4013) 
s_mov_b32 s18, s0 // (op 4013) convert

s_cmp_lt_u32 s47, s18 // (op 4013) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__ForLoopTop_KLoopTail_4013 // (op 4013) Condition: Bottom (jump to RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__ForLoopTop_KLoopTail_4013 if true)
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_4013:




















s_barrier  // (op 14287) 

















v_cmp_eq_f32 vcc, 0.00000, v0 // (op 116) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__ConditionalFalse_0_beta_116 // (op 116) Condition: False, jump to RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__ConditionalFalse_0_beta_116
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



v_accvgpr_read v10, a16 // (op 5692) 
v_accvgpr_read v11, a17 // (op 5692) 
v_accvgpr_read v12, a18 // (op 5692) 
v_accvgpr_read v13, a19 // (op 5692) 
v_mul_f32 v14, v7, v10 // (op 5692) 
v_mul_f32 v15, v7, v11 // (op 5692) 
v_mul_f32 v16, v7, v12 // (op 5692) 
v_mul_f32 v17, v7, v13 // (op 5692) 

v_cvt_f16_f32 v10, v14 // (op 5694) 
v_cvt_f16_f32 v11, v15 // (op 5694) 
v_cvt_f16_f32 v12, v16 // (op 5694) 
v_cvt_f16_f32 v13, v17 // (op 5694) 


v_pack_b32_f16 v4, v10, v11 // (op 5695) 
v_pack_b32_f16 v5, v12, v13 // (op 5695) 

s_lshl_b32 s0, s4, 7 // (op 5695) 
v_add_u32 v3, v9, s0 // (op 5695) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5695) Store value



v_accvgpr_read v10, a20 // (op 5706) 
v_accvgpr_read v11, a21 // (op 5706) 
v_accvgpr_read v12, a22 // (op 5706) 
v_accvgpr_read v13, a23 // (op 5706) 
v_mul_f32 v14, v7, v10 // (op 5706) 
v_mul_f32 v15, v7, v11 // (op 5706) 
v_mul_f32 v16, v7, v12 // (op 5706) 
v_mul_f32 v17, v7, v13 // (op 5706) 

v_cvt_f16_f32 v10, v14 // (op 5708) 
v_cvt_f16_f32 v11, v15 // (op 5708) 
v_cvt_f16_f32 v12, v16 // (op 5708) 
v_cvt_f16_f32 v13, v17 // (op 5708) 


v_pack_b32_f16 v4, v10, v11 // (op 5709) 
v_pack_b32_f16 v5, v12, v13 // (op 5709) 

s_lshl_b32 s0, s4, 5 // (op 5709) 
s_mul_i32 s1, 5, s0 // (op 5709) 

v_add_u32 v3, v9, s1 // (op 5709) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5709) Store value



v_accvgpr_read v10, a24 // (op 5720) 
v_accvgpr_read v11, a25 // (op 5720) 
v_accvgpr_read v12, a26 // (op 5720) 
v_accvgpr_read v13, a27 // (op 5720) 
v_mul_f32 v14, v7, v10 // (op 5720) 
v_mul_f32 v15, v7, v11 // (op 5720) 
v_mul_f32 v16, v7, v12 // (op 5720) 
v_mul_f32 v17, v7, v13 // (op 5720) 

v_cvt_f16_f32 v10, v14 // (op 5722) 
v_cvt_f16_f32 v11, v15 // (op 5722) 
v_cvt_f16_f32 v12, v16 // (op 5722) 
v_cvt_f16_f32 v13, v17 // (op 5722) 


v_pack_b32_f16 v4, v10, v11 // (op 5723) 
v_pack_b32_f16 v5, v12, v13 // (op 5723) 

s_lshl_b32 s0, s4, 5 // (op 5723) 
s_mul_i32 s1, 6, s0 // (op 5723) 

v_add_u32 v3, v9, s1 // (op 5723) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5723) Store value



v_accvgpr_read v10, a28 // (op 5734) 
v_accvgpr_read v11, a29 // (op 5734) 
v_accvgpr_read v12, a30 // (op 5734) 
v_accvgpr_read v13, a31 // (op 5734) 
v_mul_f32 v14, v7, v10 // (op 5734) 
v_mul_f32 v15, v7, v11 // (op 5734) 
v_mul_f32 v16, v7, v12 // (op 5734) 
v_mul_f32 v17, v7, v13 // (op 5734) 

v_cvt_f16_f32 v10, v14 // (op 5736) 
v_cvt_f16_f32 v11, v15 // (op 5736) 
v_cvt_f16_f32 v12, v16 // (op 5736) 
v_cvt_f16_f32 v13, v17 // (op 5736) 


v_pack_b32_f16 v4, v10, v11 // (op 5737) 
v_pack_b32_f16 v5, v12, v13 // (op 5737) 

s_lshl_b32 s0, s4, 5 // (op 5737) 
s_mul_i32 s1, 7, s0 // (op 5737) 

v_add_u32 v3, v9, s1 // (op 5737) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5737) Store value



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

v_add_u32 v3, 32, v9 // (op 5794) 

buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5794) Store value



v_accvgpr_read v10, a36 // (op 5812) 
v_accvgpr_read v11, a37 // (op 5812) 
v_accvgpr_read v12, a38 // (op 5812) 
v_accvgpr_read v13, a39 // (op 5812) 
v_mul_f32 v14, v7, v10 // (op 5812) 
v_mul_f32 v15, v7, v11 // (op 5812) 
v_mul_f32 v16, v7, v12 // (op 5812) 
v_mul_f32 v17, v7, v13 // (op 5812) 

v_cvt_f16_f32 v10, v14 // (op 5814) 
v_cvt_f16_f32 v11, v15 // (op 5814) 
v_cvt_f16_f32 v12, v16 // (op 5814) 
v_cvt_f16_f32 v13, v17 // (op 5814) 


v_pack_b32_f16 v4, v10, v11 // (op 5796) 
v_pack_b32_f16 v5, v12, v13 // (op 5796) 

s_lshl_b32 s0, s4, 5 // (op 5796) 
s_add_u32 s0, s0, 32 // (op 5796) 
v_add_u32 v3, v9, s0 // (op 5796) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5796) Store value



v_accvgpr_read v10, a40 // (op 5818) 
v_accvgpr_read v11, a41 // (op 5818) 
v_accvgpr_read v12, a42 // (op 5818) 
v_accvgpr_read v13, a43 // (op 5818) 
v_mul_f32 v14, v7, v10 // (op 5818) 
v_mul_f32 v15, v7, v11 // (op 5818) 
v_mul_f32 v16, v7, v12 // (op 5818) 
v_mul_f32 v17, v7, v13 // (op 5818) 

v_cvt_f16_f32 v10, v14 // (op 5820) 
v_cvt_f16_f32 v11, v15 // (op 5820) 
v_cvt_f16_f32 v12, v16 // (op 5820) 
v_cvt_f16_f32 v13, v17 // (op 5820) 


v_pack_b32_f16 v4, v10, v11 // (op 5798) 
v_pack_b32_f16 v5, v12, v13 // (op 5798) 

s_lshl_b32 s0, s4, 6 // (op 5798) 
s_add_u32 s0, s0, 32 // (op 5798) 
v_add_u32 v3, v9, s0 // (op 5798) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5798) Store value



v_accvgpr_read v10, a44 // (op 5824) 
v_accvgpr_read v11, a45 // (op 5824) 
v_accvgpr_read v12, a46 // (op 5824) 
v_accvgpr_read v13, a47 // (op 5824) 
v_mul_f32 v14, v7, v10 // (op 5824) 
v_mul_f32 v15, v7, v11 // (op 5824) 
v_mul_f32 v16, v7, v12 // (op 5824) 
v_mul_f32 v17, v7, v13 // (op 5824) 

v_cvt_f16_f32 v10, v14 // (op 5826) 
v_cvt_f16_f32 v11, v15 // (op 5826) 
v_cvt_f16_f32 v12, v16 // (op 5826) 
v_cvt_f16_f32 v13, v17 // (op 5826) 


v_pack_b32_f16 v4, v10, v11 // (op 5800) 
v_pack_b32_f16 v5, v12, v13 // (op 5800) 

s_lshl_b32 s0, s4, 5 // (op 5800) 
s_mul_i32 s1, 3, s0 // (op 5800) 
s_add_u32 s1, s1, 32 // (op 5800) 

v_add_u32 v3, v9, s1 // (op 5800) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5800) Store value



v_accvgpr_read v10, a48 // (op 5830) 
v_accvgpr_read v11, a49 // (op 5830) 
v_accvgpr_read v12, a50 // (op 5830) 
v_accvgpr_read v13, a51 // (op 5830) 
v_mul_f32 v14, v7, v10 // (op 5830) 
v_mul_f32 v15, v7, v11 // (op 5830) 
v_mul_f32 v16, v7, v12 // (op 5830) 
v_mul_f32 v17, v7, v13 // (op 5830) 

v_cvt_f16_f32 v10, v14 // (op 5832) 
v_cvt_f16_f32 v11, v15 // (op 5832) 
v_cvt_f16_f32 v12, v16 // (op 5832) 
v_cvt_f16_f32 v13, v17 // (op 5832) 


v_pack_b32_f16 v4, v10, v11 // (op 5802) 
v_pack_b32_f16 v5, v12, v13 // (op 5802) 

s_lshl_b32 s0, s4, 7 // (op 5802) 
s_add_u32 s0, s0, 32 // (op 5802) 
v_add_u32 v3, v9, s0 // (op 5802) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5802) Store value



v_accvgpr_read v10, a52 // (op 5836) 
v_accvgpr_read v11, a53 // (op 5836) 
v_accvgpr_read v12, a54 // (op 5836) 
v_accvgpr_read v13, a55 // (op 5836) 
v_mul_f32 v14, v7, v10 // (op 5836) 
v_mul_f32 v15, v7, v11 // (op 5836) 
v_mul_f32 v16, v7, v12 // (op 5836) 
v_mul_f32 v17, v7, v13 // (op 5836) 

v_cvt_f16_f32 v10, v14 // (op 5838) 
v_cvt_f16_f32 v11, v15 // (op 5838) 
v_cvt_f16_f32 v12, v16 // (op 5838) 
v_cvt_f16_f32 v13, v17 // (op 5838) 


v_pack_b32_f16 v4, v10, v11 // (op 5804) 
v_pack_b32_f16 v5, v12, v13 // (op 5804) 

s_lshl_b32 s0, s4, 5 // (op 5804) 
s_mul_i32 s1, 5, s0 // (op 5804) 
s_add_u32 s1, s1, 32 // (op 5804) 

v_add_u32 v3, v9, s1 // (op 5804) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5804) Store value



v_accvgpr_read v10, a56 // (op 5842) 
v_accvgpr_read v11, a57 // (op 5842) 
v_accvgpr_read v12, a58 // (op 5842) 
v_accvgpr_read v13, a59 // (op 5842) 
v_mul_f32 v14, v7, v10 // (op 5842) 
v_mul_f32 v15, v7, v11 // (op 5842) 
v_mul_f32 v16, v7, v12 // (op 5842) 
v_mul_f32 v17, v7, v13 // (op 5842) 

v_cvt_f16_f32 v10, v14 // (op 5844) 
v_cvt_f16_f32 v11, v15 // (op 5844) 
v_cvt_f16_f32 v12, v16 // (op 5844) 
v_cvt_f16_f32 v13, v17 // (op 5844) 


v_pack_b32_f16 v4, v10, v11 // (op 5806) 
v_pack_b32_f16 v5, v12, v13 // (op 5806) 

s_lshl_b32 s0, s4, 5 // (op 5806) 
s_mul_i32 s1, 6, s0 // (op 5806) 
s_add_u32 s1, s1, 32 // (op 5806) 

v_add_u32 v3, v9, s1 // (op 5806) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5806) Store value



v_accvgpr_read v10, a60 // (op 5848) 
v_accvgpr_read v11, a61 // (op 5848) 
v_accvgpr_read v12, a62 // (op 5848) 
v_accvgpr_read v13, a63 // (op 5848) 
v_mul_f32 v14, v7, v10 // (op 5848) 
v_mul_f32 v15, v7, v11 // (op 5848) 
v_mul_f32 v16, v7, v12 // (op 5848) 
v_mul_f32 v17, v7, v13 // (op 5848) 

v_cvt_f16_f32 v10, v14 // (op 5850) 
v_cvt_f16_f32 v11, v15 // (op 5850) 
v_cvt_f16_f32 v12, v16 // (op 5850) 
v_cvt_f16_f32 v13, v17 // (op 5850) 


v_pack_b32_f16 v4, v10, v11 // (op 5808) 
v_pack_b32_f16 v5, v12, v13 // (op 5808) 

s_lshl_b32 s0, s4, 5 // (op 5808) 
s_mul_i32 s1, 7, s0 // (op 5808) 
s_add_u32 s1, s1, 32 // (op 5808) 

v_add_u32 v3, v9, s1 // (op 5808) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5808) Store value



v_accvgpr_read v10, a64 // (op 5938) 
v_accvgpr_read v11, a65 // (op 5938) 
v_accvgpr_read v12, a66 // (op 5938) 
v_accvgpr_read v13, a67 // (op 5938) 
v_mul_f32 v14, v7, v10 // (op 5938) 
v_mul_f32 v15, v7, v11 // (op 5938) 
v_mul_f32 v16, v7, v12 // (op 5938) 
v_mul_f32 v17, v7, v13 // (op 5938) 

v_cvt_f16_f32 v10, v14 // (op 5940) 
v_cvt_f16_f32 v11, v15 // (op 5940) 
v_cvt_f16_f32 v12, v16 // (op 5940) 
v_cvt_f16_f32 v13, v17 // (op 5940) 


v_pack_b32_f16 v4, v10, v11 // (op 5942) 
v_pack_b32_f16 v5, v12, v13 // (op 5942) 

v_add_u32 v3, 64, v9 // (op 5942) 

buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5942) Store value



v_accvgpr_read v10, a68 // (op 5960) 
v_accvgpr_read v11, a69 // (op 5960) 
v_accvgpr_read v12, a70 // (op 5960) 
v_accvgpr_read v13, a71 // (op 5960) 
v_mul_f32 v14, v7, v10 // (op 5960) 
v_mul_f32 v15, v7, v11 // (op 5960) 
v_mul_f32 v16, v7, v12 // (op 5960) 
v_mul_f32 v17, v7, v13 // (op 5960) 

v_cvt_f16_f32 v10, v14 // (op 5962) 
v_cvt_f16_f32 v11, v15 // (op 5962) 
v_cvt_f16_f32 v12, v16 // (op 5962) 
v_cvt_f16_f32 v13, v17 // (op 5962) 


v_pack_b32_f16 v4, v10, v11 // (op 5944) 
v_pack_b32_f16 v5, v12, v13 // (op 5944) 

s_lshl_b32 s0, s4, 5 // (op 5944) 
s_add_u32 s0, s0, 64 // (op 5944) 
v_add_u32 v3, v9, s0 // (op 5944) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5944) Store value



v_accvgpr_read v10, a72 // (op 5966) 
v_accvgpr_read v11, a73 // (op 5966) 
v_accvgpr_read v12, a74 // (op 5966) 
v_accvgpr_read v13, a75 // (op 5966) 
v_mul_f32 v14, v7, v10 // (op 5966) 
v_mul_f32 v15, v7, v11 // (op 5966) 
v_mul_f32 v16, v7, v12 // (op 5966) 
v_mul_f32 v17, v7, v13 // (op 5966) 

v_cvt_f16_f32 v10, v14 // (op 5968) 
v_cvt_f16_f32 v11, v15 // (op 5968) 
v_cvt_f16_f32 v12, v16 // (op 5968) 
v_cvt_f16_f32 v13, v17 // (op 5968) 


v_pack_b32_f16 v4, v10, v11 // (op 5946) 
v_pack_b32_f16 v5, v12, v13 // (op 5946) 

s_lshl_b32 s0, s4, 6 // (op 5946) 
s_add_u32 s0, s0, 64 // (op 5946) 
v_add_u32 v3, v9, s0 // (op 5946) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5946) Store value



v_accvgpr_read v10, a76 // (op 5972) 
v_accvgpr_read v11, a77 // (op 5972) 
v_accvgpr_read v12, a78 // (op 5972) 
v_accvgpr_read v13, a79 // (op 5972) 
v_mul_f32 v14, v7, v10 // (op 5972) 
v_mul_f32 v15, v7, v11 // (op 5972) 
v_mul_f32 v16, v7, v12 // (op 5972) 
v_mul_f32 v17, v7, v13 // (op 5972) 

v_cvt_f16_f32 v10, v14 // (op 5974) 
v_cvt_f16_f32 v11, v15 // (op 5974) 
v_cvt_f16_f32 v12, v16 // (op 5974) 
v_cvt_f16_f32 v13, v17 // (op 5974) 


v_pack_b32_f16 v4, v10, v11 // (op 5948) 
v_pack_b32_f16 v5, v12, v13 // (op 5948) 

s_lshl_b32 s0, s4, 5 // (op 5948) 
s_mul_i32 s1, 3, s0 // (op 5948) 
s_add_u32 s1, s1, 64 // (op 5948) 

v_add_u32 v3, v9, s1 // (op 5948) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5948) Store value



v_accvgpr_read v10, a80 // (op 5978) 
v_accvgpr_read v11, a81 // (op 5978) 
v_accvgpr_read v12, a82 // (op 5978) 
v_accvgpr_read v13, a83 // (op 5978) 
v_mul_f32 v14, v7, v10 // (op 5978) 
v_mul_f32 v15, v7, v11 // (op 5978) 
v_mul_f32 v16, v7, v12 // (op 5978) 
v_mul_f32 v17, v7, v13 // (op 5978) 

v_cvt_f16_f32 v10, v14 // (op 5980) 
v_cvt_f16_f32 v11, v15 // (op 5980) 
v_cvt_f16_f32 v12, v16 // (op 5980) 
v_cvt_f16_f32 v13, v17 // (op 5980) 


v_pack_b32_f16 v4, v10, v11 // (op 5950) 
v_pack_b32_f16 v5, v12, v13 // (op 5950) 

s_lshl_b32 s0, s4, 7 // (op 5950) 
s_add_u32 s0, s0, 64 // (op 5950) 
v_add_u32 v3, v9, s0 // (op 5950) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5950) Store value



v_accvgpr_read v10, a84 // (op 5984) 
v_accvgpr_read v11, a85 // (op 5984) 
v_accvgpr_read v12, a86 // (op 5984) 
v_accvgpr_read v13, a87 // (op 5984) 
v_mul_f32 v14, v7, v10 // (op 5984) 
v_mul_f32 v15, v7, v11 // (op 5984) 
v_mul_f32 v16, v7, v12 // (op 5984) 
v_mul_f32 v17, v7, v13 // (op 5984) 

v_cvt_f16_f32 v10, v14 // (op 5986) 
v_cvt_f16_f32 v11, v15 // (op 5986) 
v_cvt_f16_f32 v12, v16 // (op 5986) 
v_cvt_f16_f32 v13, v17 // (op 5986) 


v_pack_b32_f16 v4, v10, v11 // (op 5952) 
v_pack_b32_f16 v5, v12, v13 // (op 5952) 

s_lshl_b32 s0, s4, 5 // (op 5952) 
s_mul_i32 s1, 5, s0 // (op 5952) 
s_add_u32 s1, s1, 64 // (op 5952) 

v_add_u32 v3, v9, s1 // (op 5952) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5952) Store value



v_accvgpr_read v10, a88 // (op 5990) 
v_accvgpr_read v11, a89 // (op 5990) 
v_accvgpr_read v12, a90 // (op 5990) 
v_accvgpr_read v13, a91 // (op 5990) 
v_mul_f32 v14, v7, v10 // (op 5990) 
v_mul_f32 v15, v7, v11 // (op 5990) 
v_mul_f32 v16, v7, v12 // (op 5990) 
v_mul_f32 v17, v7, v13 // (op 5990) 

v_cvt_f16_f32 v10, v14 // (op 5992) 
v_cvt_f16_f32 v11, v15 // (op 5992) 
v_cvt_f16_f32 v12, v16 // (op 5992) 
v_cvt_f16_f32 v13, v17 // (op 5992) 


v_pack_b32_f16 v4, v10, v11 // (op 5954) 
v_pack_b32_f16 v5, v12, v13 // (op 5954) 

s_lshl_b32 s0, s4, 5 // (op 5954) 
s_mul_i32 s1, 6, s0 // (op 5954) 
s_add_u32 s1, s1, 64 // (op 5954) 

v_add_u32 v3, v9, s1 // (op 5954) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5954) Store value



v_accvgpr_read v10, a92 // (op 5996) 
v_accvgpr_read v11, a93 // (op 5996) 
v_accvgpr_read v12, a94 // (op 5996) 
v_accvgpr_read v13, a95 // (op 5996) 
v_mul_f32 v14, v7, v10 // (op 5996) 
v_mul_f32 v15, v7, v11 // (op 5996) 
v_mul_f32 v16, v7, v12 // (op 5996) 
v_mul_f32 v17, v7, v13 // (op 5996) 

v_cvt_f16_f32 v10, v14 // (op 5998) 
v_cvt_f16_f32 v11, v15 // (op 5998) 
v_cvt_f16_f32 v12, v16 // (op 5998) 
v_cvt_f16_f32 v13, v17 // (op 5998) 


v_pack_b32_f16 v4, v10, v11 // (op 5956) 
v_pack_b32_f16 v5, v12, v13 // (op 5956) 

s_lshl_b32 s0, s4, 5 // (op 5956) 
s_mul_i32 s1, 7, s0 // (op 5956) 
s_add_u32 s1, s1, 64 // (op 5956) 

v_add_u32 v3, v9, s1 // (op 5956) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5956) Store value



v_accvgpr_read v10, a96 // (op 6086) 
v_accvgpr_read v11, a97 // (op 6086) 
v_accvgpr_read v12, a98 // (op 6086) 
v_accvgpr_read v13, a99 // (op 6086) 
v_mul_f32 v14, v7, v10 // (op 6086) 
v_mul_f32 v15, v7, v11 // (op 6086) 
v_mul_f32 v16, v7, v12 // (op 6086) 
v_mul_f32 v17, v7, v13 // (op 6086) 

v_cvt_f16_f32 v10, v14 // (op 6088) 
v_cvt_f16_f32 v11, v15 // (op 6088) 
v_cvt_f16_f32 v12, v16 // (op 6088) 
v_cvt_f16_f32 v13, v17 // (op 6088) 


v_pack_b32_f16 v4, v10, v11 // (op 6090) 
v_pack_b32_f16 v5, v12, v13 // (op 6090) 

v_add_u32 v3, 96, v9 // (op 6090) 

buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 6090) Store value



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


v_pack_b32_f16 v4, v10, v11 // (op 6092) 
v_pack_b32_f16 v5, v12, v13 // (op 6092) 

s_lshl_b32 s0, s4, 5 // (op 6092) 
s_add_u32 s0, s0, 96 // (op 6092) 
v_add_u32 v3, v9, s0 // (op 6092) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 6092) Store value



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


v_pack_b32_f16 v4, v10, v11 // (op 6094) 
v_pack_b32_f16 v5, v12, v13 // (op 6094) 

s_lshl_b32 s0, s4, 6 // (op 6094) 
s_add_u32 s0, s0, 96 // (op 6094) 
v_add_u32 v3, v9, s0 // (op 6094) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 6094) Store value



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


v_pack_b32_f16 v4, v10, v11 // (op 6096) 
v_pack_b32_f16 v5, v12, v13 // (op 6096) 

s_lshl_b32 s0, s4, 5 // (op 6096) 
s_mul_i32 s1, 3, s0 // (op 6096) 
s_add_u32 s1, s1, 96 // (op 6096) 

v_add_u32 v3, v9, s1 // (op 6096) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 6096) Store value



v_accvgpr_read v10, a112 // (op 6126) 
v_accvgpr_read v11, a113 // (op 6126) 
v_accvgpr_read v12, a114 // (op 6126) 
v_accvgpr_read v13, a115 // (op 6126) 
v_mul_f32 v14, v7, v10 // (op 6126) 
v_mul_f32 v15, v7, v11 // (op 6126) 
v_mul_f32 v16, v7, v12 // (op 6126) 
v_mul_f32 v17, v7, v13 // (op 6126) 

v_cvt_f16_f32 v10, v14 // (op 6128) 
v_cvt_f16_f32 v11, v15 // (op 6128) 
v_cvt_f16_f32 v12, v16 // (op 6128) 
v_cvt_f16_f32 v13, v17 // (op 6128) 


v_pack_b32_f16 v4, v10, v11 // (op 6098) 
v_pack_b32_f16 v5, v12, v13 // (op 6098) 

s_lshl_b32 s0, s4, 7 // (op 6098) 
s_add_u32 s0, s0, 96 // (op 6098) 
v_add_u32 v3, v9, s0 // (op 6098) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 6098) Store value



v_accvgpr_read v10, a116 // (op 6132) 
v_accvgpr_read v11, a117 // (op 6132) 
v_accvgpr_read v12, a118 // (op 6132) 
v_accvgpr_read v13, a119 // (op 6132) 
v_mul_f32 v14, v7, v10 // (op 6132) 
v_mul_f32 v15, v7, v11 // (op 6132) 
v_mul_f32 v16, v7, v12 // (op 6132) 
v_mul_f32 v17, v7, v13 // (op 6132) 

v_cvt_f16_f32 v10, v14 // (op 6134) 
v_cvt_f16_f32 v11, v15 // (op 6134) 
v_cvt_f16_f32 v12, v16 // (op 6134) 
v_cvt_f16_f32 v13, v17 // (op 6134) 


v_pack_b32_f16 v4, v10, v11 // (op 6100) 
v_pack_b32_f16 v5, v12, v13 // (op 6100) 

s_lshl_b32 s0, s4, 5 // (op 6100) 
s_mul_i32 s1, 5, s0 // (op 6100) 
s_add_u32 s1, s1, 96 // (op 6100) 

v_add_u32 v3, v9, s1 // (op 6100) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 6100) Store value



v_accvgpr_read v10, a120 // (op 6138) 
v_accvgpr_read v11, a121 // (op 6138) 
v_accvgpr_read v12, a122 // (op 6138) 
v_accvgpr_read v13, a123 // (op 6138) 
v_mul_f32 v14, v7, v10 // (op 6138) 
v_mul_f32 v15, v7, v11 // (op 6138) 
v_mul_f32 v16, v7, v12 // (op 6138) 
v_mul_f32 v17, v7, v13 // (op 6138) 

v_cvt_f16_f32 v10, v14 // (op 6140) 
v_cvt_f16_f32 v11, v15 // (op 6140) 
v_cvt_f16_f32 v12, v16 // (op 6140) 
v_cvt_f16_f32 v13, v17 // (op 6140) 


v_pack_b32_f16 v4, v10, v11 // (op 6102) 
v_pack_b32_f16 v5, v12, v13 // (op 6102) 

s_lshl_b32 s0, s4, 5 // (op 6102) 
s_mul_i32 s1, 6, s0 // (op 6102) 
s_add_u32 s1, s1, 96 // (op 6102) 

v_add_u32 v3, v9, s1 // (op 6102) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 6102) Store value



v_accvgpr_read v10, a124 // (op 6144) 
v_accvgpr_read v11, a125 // (op 6144) 
v_accvgpr_read v12, a126 // (op 6144) 
v_accvgpr_read v13, a127 // (op 6144) 
v_mul_f32 v14, v7, v10 // (op 6144) 
v_mul_f32 v15, v7, v11 // (op 6144) 
v_mul_f32 v16, v7, v12 // (op 6144) 
v_mul_f32 v17, v7, v13 // (op 6144) 

v_cvt_f16_f32 v10, v14 // (op 6146) 
v_cvt_f16_f32 v11, v15 // (op 6146) 
v_cvt_f16_f32 v12, v16 // (op 6146) 
v_cvt_f16_f32 v13, v17 // (op 6146) 


v_pack_b32_f16 v4, v10, v11 // (op 6104) 
v_pack_b32_f16 v5, v12, v13 // (op 6104) 

s_lshl_b32 s0, s4, 5 // (op 6104) 
s_mul_i32 s1, 7, s0 // (op 6104) 
s_add_u32 s1, s1, 96 // (op 6104) 

v_add_u32 v3, v9, s1 // (op 6104) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 6104) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__ConditionalBottom_0_beta_116 // (op 116) Condition: Done, jump to RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__ConditionalBottom_0_beta_116
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__ConditionalFalse_0_beta_116:


v_add_u32 v3, 96, v8 // (op 6876) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6876) Load value

v_accvgpr_read v10, a96 // (op 6855) 
v_accvgpr_read v11, a97 // (op 6855) 
v_accvgpr_read v12, a98 // (op 6855) 
v_accvgpr_read v13, a99 // (op 6855) 
v_mul_f32 v14, v7, v10 // (op 6855) 
v_mul_f32 v15, v7, v11 // (op 6855) 
v_mul_f32 v16, v7, v12 // (op 6855) 
v_mul_f32 v17, v7, v13 // (op 6855) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6875) 
v_lshrrev_b32 v11, 16, v4 // (op 6875) 
v_cvt_f32_f16 v11, v11 // (op 6875) 
v_mul_f32 v18, v0, v10 // (op 6875) 
v_mul_f32 v19, v0, v11 // (op 6875) 
v_cvt_f32_f16 v10, v5 // (op 6875) 
v_lshrrev_b32 v11, 16, v5 // (op 6875) 
v_cvt_f32_f16 v11, v11 // (op 6875) 
v_mul_f32 v20, v0, v10 // (op 6875) 
v_mul_f32 v21, v0, v11 // (op 6875) 


v_add_f32 v10, v18, v14 // (op 6856) 
v_add_f32 v11, v19, v15 // (op 6856) 
v_add_f32 v12, v20, v16 // (op 6856) 
v_add_f32 v13, v21, v17 // (op 6856) 


v_cvt_f16_f32 v14, v10 // (op 6857) 
v_cvt_f16_f32 v15, v11 // (op 6857) 
v_cvt_f16_f32 v16, v12 // (op 6857) 
v_cvt_f16_f32 v17, v13 // (op 6857) 


v_pack_b32_f16 v10, v14, v15 // (op 6859) 
v_pack_b32_f16 v11, v16, v17 // (op 6859) 

v_add_u32 v3, 96, v9 // (op 6859) 

buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6859) Store value





s_lshl_b32 s0, s2, 5 // (op 6881) 
s_add_u32 s0, s0, 96 // (op 6881) 
v_add_u32 v3, v8, s0 // (op 6881) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6881) Load value

v_accvgpr_read v10, a100 // (op 6913) 
v_accvgpr_read v11, a101 // (op 6913) 
v_accvgpr_read v12, a102 // (op 6913) 
v_accvgpr_read v13, a103 // (op 6913) 
v_mul_f32 v14, v7, v10 // (op 6913) 
v_mul_f32 v15, v7, v11 // (op 6913) 
v_mul_f32 v16, v7, v12 // (op 6913) 
v_mul_f32 v17, v7, v13 // (op 6913) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6878) 
v_lshrrev_b32 v11, 16, v4 // (op 6878) 
v_cvt_f32_f16 v11, v11 // (op 6878) 
v_mul_f32 v18, v0, v10 // (op 6878) 
v_mul_f32 v19, v0, v11 // (op 6878) 
v_cvt_f32_f16 v10, v5 // (op 6878) 
v_lshrrev_b32 v11, 16, v5 // (op 6878) 
v_cvt_f32_f16 v11, v11 // (op 6878) 
v_mul_f32 v20, v0, v10 // (op 6878) 
v_mul_f32 v21, v0, v11 // (op 6878) 


v_add_f32 v10, v18, v14 // (op 6879) 
v_add_f32 v11, v19, v15 // (op 6879) 
v_add_f32 v12, v20, v16 // (op 6879) 
v_add_f32 v13, v21, v17 // (op 6879) 


v_cvt_f16_f32 v14, v10 // (op 6880) 
v_cvt_f16_f32 v15, v11 // (op 6880) 
v_cvt_f16_f32 v16, v12 // (op 6880) 
v_cvt_f16_f32 v17, v13 // (op 6880) 


v_pack_b32_f16 v10, v14, v15 // (op 6861) 
v_pack_b32_f16 v11, v16, v17 // (op 6861) 

s_lshl_b32 s0, s4, 5 // (op 6861) 
s_add_u32 s0, s0, 96 // (op 6861) 
v_add_u32 v3, v9, s0 // (op 6861) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6861) Store value





s_lshl_b32 s0, s2, 6 // (op 6886) 
s_add_u32 s0, s0, 96 // (op 6886) 
v_add_u32 v3, v8, s0 // (op 6886) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6886) Load value

v_accvgpr_read v10, a104 // (op 6915) 
v_accvgpr_read v11, a105 // (op 6915) 
v_accvgpr_read v12, a106 // (op 6915) 
v_accvgpr_read v13, a107 // (op 6915) 
v_mul_f32 v14, v7, v10 // (op 6915) 
v_mul_f32 v15, v7, v11 // (op 6915) 
v_mul_f32 v16, v7, v12 // (op 6915) 
v_mul_f32 v17, v7, v13 // (op 6915) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6883) 
v_lshrrev_b32 v11, 16, v4 // (op 6883) 
v_cvt_f32_f16 v11, v11 // (op 6883) 
v_mul_f32 v18, v0, v10 // (op 6883) 
v_mul_f32 v19, v0, v11 // (op 6883) 
v_cvt_f32_f16 v10, v5 // (op 6883) 
v_lshrrev_b32 v11, 16, v5 // (op 6883) 
v_cvt_f32_f16 v11, v11 // (op 6883) 
v_mul_f32 v20, v0, v10 // (op 6883) 
v_mul_f32 v21, v0, v11 // (op 6883) 


v_add_f32 v10, v18, v14 // (op 6884) 
v_add_f32 v11, v19, v15 // (op 6884) 
v_add_f32 v12, v20, v16 // (op 6884) 
v_add_f32 v13, v21, v17 // (op 6884) 


v_cvt_f16_f32 v14, v10 // (op 6885) 
v_cvt_f16_f32 v15, v11 // (op 6885) 
v_cvt_f16_f32 v16, v12 // (op 6885) 
v_cvt_f16_f32 v17, v13 // (op 6885) 


v_pack_b32_f16 v10, v14, v15 // (op 6863) 
v_pack_b32_f16 v11, v16, v17 // (op 6863) 

s_lshl_b32 s0, s4, 6 // (op 6863) 
s_add_u32 s0, s0, 96 // (op 6863) 
v_add_u32 v3, v9, s0 // (op 6863) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6863) Store value





s_lshl_b32 s0, s2, 5 // (op 6891) 
s_mul_i32 s1, 3, s0 // (op 6891) 
s_add_u32 s1, s1, 96 // (op 6891) 

v_add_u32 v3, v8, s1 // (op 6891) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6891) Load value

v_accvgpr_read v10, a108 // (op 6917) 
v_accvgpr_read v11, a109 // (op 6917) 
v_accvgpr_read v12, a110 // (op 6917) 
v_accvgpr_read v13, a111 // (op 6917) 
v_mul_f32 v14, v7, v10 // (op 6917) 
v_mul_f32 v15, v7, v11 // (op 6917) 
v_mul_f32 v16, v7, v12 // (op 6917) 
v_mul_f32 v17, v7, v13 // (op 6917) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6888) 
v_lshrrev_b32 v11, 16, v4 // (op 6888) 
v_cvt_f32_f16 v11, v11 // (op 6888) 
v_mul_f32 v18, v0, v10 // (op 6888) 
v_mul_f32 v19, v0, v11 // (op 6888) 
v_cvt_f32_f16 v10, v5 // (op 6888) 
v_lshrrev_b32 v11, 16, v5 // (op 6888) 
v_cvt_f32_f16 v11, v11 // (op 6888) 
v_mul_f32 v20, v0, v10 // (op 6888) 
v_mul_f32 v21, v0, v11 // (op 6888) 


v_add_f32 v10, v18, v14 // (op 6889) 
v_add_f32 v11, v19, v15 // (op 6889) 
v_add_f32 v12, v20, v16 // (op 6889) 
v_add_f32 v13, v21, v17 // (op 6889) 


v_cvt_f16_f32 v14, v10 // (op 6890) 
v_cvt_f16_f32 v15, v11 // (op 6890) 
v_cvt_f16_f32 v16, v12 // (op 6890) 
v_cvt_f16_f32 v17, v13 // (op 6890) 


v_pack_b32_f16 v10, v14, v15 // (op 6865) 
v_pack_b32_f16 v11, v16, v17 // (op 6865) 

s_lshl_b32 s0, s4, 5 // (op 6865) 
s_mul_i32 s1, 3, s0 // (op 6865) 
s_add_u32 s1, s1, 96 // (op 6865) 

v_add_u32 v3, v9, s1 // (op 6865) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6865) Store value





s_lshl_b32 s0, s2, 7 // (op 6896) 
s_add_u32 s0, s0, 96 // (op 6896) 
v_add_u32 v3, v8, s0 // (op 6896) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6896) Load value

v_accvgpr_read v10, a112 // (op 6919) 
v_accvgpr_read v11, a113 // (op 6919) 
v_accvgpr_read v12, a114 // (op 6919) 
v_accvgpr_read v13, a115 // (op 6919) 
v_mul_f32 v14, v7, v10 // (op 6919) 
v_mul_f32 v15, v7, v11 // (op 6919) 
v_mul_f32 v16, v7, v12 // (op 6919) 
v_mul_f32 v17, v7, v13 // (op 6919) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6893) 
v_lshrrev_b32 v11, 16, v4 // (op 6893) 
v_cvt_f32_f16 v11, v11 // (op 6893) 
v_mul_f32 v18, v0, v10 // (op 6893) 
v_mul_f32 v19, v0, v11 // (op 6893) 
v_cvt_f32_f16 v10, v5 // (op 6893) 
v_lshrrev_b32 v11, 16, v5 // (op 6893) 
v_cvt_f32_f16 v11, v11 // (op 6893) 
v_mul_f32 v20, v0, v10 // (op 6893) 
v_mul_f32 v21, v0, v11 // (op 6893) 


v_add_f32 v10, v18, v14 // (op 6894) 
v_add_f32 v11, v19, v15 // (op 6894) 
v_add_f32 v12, v20, v16 // (op 6894) 
v_add_f32 v13, v21, v17 // (op 6894) 


v_cvt_f16_f32 v14, v10 // (op 6895) 
v_cvt_f16_f32 v15, v11 // (op 6895) 
v_cvt_f16_f32 v16, v12 // (op 6895) 
v_cvt_f16_f32 v17, v13 // (op 6895) 


v_pack_b32_f16 v10, v14, v15 // (op 6867) 
v_pack_b32_f16 v11, v16, v17 // (op 6867) 

s_lshl_b32 s0, s4, 7 // (op 6867) 
s_add_u32 s0, s0, 96 // (op 6867) 
v_add_u32 v3, v9, s0 // (op 6867) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6867) Store value





s_lshl_b32 s0, s2, 5 // (op 6901) 
s_mul_i32 s1, 5, s0 // (op 6901) 
s_add_u32 s1, s1, 96 // (op 6901) 

v_add_u32 v3, v8, s1 // (op 6901) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6901) Load value

v_accvgpr_read v10, a116 // (op 6921) 
v_accvgpr_read v11, a117 // (op 6921) 
v_accvgpr_read v12, a118 // (op 6921) 
v_accvgpr_read v13, a119 // (op 6921) 
v_mul_f32 v14, v7, v10 // (op 6921) 
v_mul_f32 v15, v7, v11 // (op 6921) 
v_mul_f32 v16, v7, v12 // (op 6921) 
v_mul_f32 v17, v7, v13 // (op 6921) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6898) 
v_lshrrev_b32 v11, 16, v4 // (op 6898) 
v_cvt_f32_f16 v11, v11 // (op 6898) 
v_mul_f32 v18, v0, v10 // (op 6898) 
v_mul_f32 v19, v0, v11 // (op 6898) 
v_cvt_f32_f16 v10, v5 // (op 6898) 
v_lshrrev_b32 v11, 16, v5 // (op 6898) 
v_cvt_f32_f16 v11, v11 // (op 6898) 
v_mul_f32 v20, v0, v10 // (op 6898) 
v_mul_f32 v21, v0, v11 // (op 6898) 


v_add_f32 v10, v18, v14 // (op 6899) 
v_add_f32 v11, v19, v15 // (op 6899) 
v_add_f32 v12, v20, v16 // (op 6899) 
v_add_f32 v13, v21, v17 // (op 6899) 


v_cvt_f16_f32 v14, v10 // (op 6900) 
v_cvt_f16_f32 v15, v11 // (op 6900) 
v_cvt_f16_f32 v16, v12 // (op 6900) 
v_cvt_f16_f32 v17, v13 // (op 6900) 


v_pack_b32_f16 v10, v14, v15 // (op 6869) 
v_pack_b32_f16 v11, v16, v17 // (op 6869) 

s_lshl_b32 s0, s4, 5 // (op 6869) 
s_mul_i32 s1, 5, s0 // (op 6869) 
s_add_u32 s1, s1, 96 // (op 6869) 

v_add_u32 v3, v9, s1 // (op 6869) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6869) Store value





s_lshl_b32 s0, s2, 5 // (op 6906) 
s_mul_i32 s1, 6, s0 // (op 6906) 
s_add_u32 s1, s1, 96 // (op 6906) 

v_add_u32 v3, v8, s1 // (op 6906) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6906) Load value

v_accvgpr_read v10, a120 // (op 6923) 
v_accvgpr_read v11, a121 // (op 6923) 
v_accvgpr_read v12, a122 // (op 6923) 
v_accvgpr_read v13, a123 // (op 6923) 
v_mul_f32 v14, v7, v10 // (op 6923) 
v_mul_f32 v15, v7, v11 // (op 6923) 
v_mul_f32 v16, v7, v12 // (op 6923) 
v_mul_f32 v17, v7, v13 // (op 6923) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6903) 
v_lshrrev_b32 v11, 16, v4 // (op 6903) 
v_cvt_f32_f16 v11, v11 // (op 6903) 
v_mul_f32 v18, v0, v10 // (op 6903) 
v_mul_f32 v19, v0, v11 // (op 6903) 
v_cvt_f32_f16 v10, v5 // (op 6903) 
v_lshrrev_b32 v11, 16, v5 // (op 6903) 
v_cvt_f32_f16 v11, v11 // (op 6903) 
v_mul_f32 v20, v0, v10 // (op 6903) 
v_mul_f32 v21, v0, v11 // (op 6903) 


v_add_f32 v10, v18, v14 // (op 6904) 
v_add_f32 v11, v19, v15 // (op 6904) 
v_add_f32 v12, v20, v16 // (op 6904) 
v_add_f32 v13, v21, v17 // (op 6904) 


v_cvt_f16_f32 v14, v10 // (op 6905) 
v_cvt_f16_f32 v15, v11 // (op 6905) 
v_cvt_f16_f32 v16, v12 // (op 6905) 
v_cvt_f16_f32 v17, v13 // (op 6905) 


v_pack_b32_f16 v10, v14, v15 // (op 6871) 
v_pack_b32_f16 v11, v16, v17 // (op 6871) 

s_lshl_b32 s0, s4, 5 // (op 6871) 
s_mul_i32 s1, 6, s0 // (op 6871) 
s_add_u32 s1, s1, 96 // (op 6871) 

v_add_u32 v3, v9, s1 // (op 6871) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6871) Store value





s_lshl_b32 s0, s2, 5 // (op 6911) 
s_mul_i32 s1, 7, s0 // (op 6911) 
s_add_u32 s1, s1, 96 // (op 6911) 

v_add_u32 v3, v8, s1 // (op 6911) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6911) Load value

v_accvgpr_read v10, a124 // (op 6925) 
v_accvgpr_read v11, a125 // (op 6925) 
v_accvgpr_read v12, a126 // (op 6925) 
v_accvgpr_read v13, a127 // (op 6925) 
v_mul_f32 v14, v7, v10 // (op 6925) 
v_mul_f32 v15, v7, v11 // (op 6925) 
v_mul_f32 v16, v7, v12 // (op 6925) 
v_mul_f32 v17, v7, v13 // (op 6925) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6908) 
v_lshrrev_b32 v11, 16, v4 // (op 6908) 
v_cvt_f32_f16 v11, v11 // (op 6908) 
v_mul_f32 v18, v0, v10 // (op 6908) 
v_mul_f32 v19, v0, v11 // (op 6908) 
v_cvt_f32_f16 v10, v5 // (op 6908) 
v_lshrrev_b32 v11, 16, v5 // (op 6908) 
v_cvt_f32_f16 v11, v11 // (op 6908) 
v_mul_f32 v20, v0, v10 // (op 6908) 
v_mul_f32 v21, v0, v11 // (op 6908) 


v_add_f32 v10, v18, v14 // (op 6909) 
v_add_f32 v11, v19, v15 // (op 6909) 
v_add_f32 v12, v20, v16 // (op 6909) 
v_add_f32 v13, v21, v17 // (op 6909) 


v_cvt_f16_f32 v14, v10 // (op 6910) 
v_cvt_f16_f32 v15, v11 // (op 6910) 
v_cvt_f16_f32 v16, v12 // (op 6910) 
v_cvt_f16_f32 v17, v13 // (op 6910) 


v_pack_b32_f16 v10, v14, v15 // (op 6873) 
v_pack_b32_f16 v11, v16, v17 // (op 6873) 

s_lshl_b32 s0, s4, 5 // (op 6873) 
s_mul_i32 s1, 7, s0 // (op 6873) 
s_add_u32 s1, s1, 96 // (op 6873) 

v_add_u32 v3, v9, s1 // (op 6873) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6873) Store value





v_add_u32 v3, 64, v8 // (op 6705) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6705) Load value

v_accvgpr_read v10, a64 // (op 6684) 
v_accvgpr_read v11, a65 // (op 6684) 
v_accvgpr_read v12, a66 // (op 6684) 
v_accvgpr_read v13, a67 // (op 6684) 
v_mul_f32 v14, v7, v10 // (op 6684) 
v_mul_f32 v15, v7, v11 // (op 6684) 
v_mul_f32 v16, v7, v12 // (op 6684) 
v_mul_f32 v17, v7, v13 // (op 6684) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6704) 
v_lshrrev_b32 v11, 16, v4 // (op 6704) 
v_cvt_f32_f16 v11, v11 // (op 6704) 
v_mul_f32 v18, v0, v10 // (op 6704) 
v_mul_f32 v19, v0, v11 // (op 6704) 
v_cvt_f32_f16 v10, v5 // (op 6704) 
v_lshrrev_b32 v11, 16, v5 // (op 6704) 
v_cvt_f32_f16 v11, v11 // (op 6704) 
v_mul_f32 v20, v0, v10 // (op 6704) 
v_mul_f32 v21, v0, v11 // (op 6704) 


v_add_f32 v10, v18, v14 // (op 6685) 
v_add_f32 v11, v19, v15 // (op 6685) 
v_add_f32 v12, v20, v16 // (op 6685) 
v_add_f32 v13, v21, v17 // (op 6685) 


v_cvt_f16_f32 v14, v10 // (op 6686) 
v_cvt_f16_f32 v15, v11 // (op 6686) 
v_cvt_f16_f32 v16, v12 // (op 6686) 
v_cvt_f16_f32 v17, v13 // (op 6686) 


v_pack_b32_f16 v10, v14, v15 // (op 6688) 
v_pack_b32_f16 v11, v16, v17 // (op 6688) 

v_add_u32 v3, 64, v9 // (op 6688) 

buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6688) Store value





s_lshl_b32 s0, s2, 5 // (op 6710) 
s_add_u32 s0, s0, 64 // (op 6710) 
v_add_u32 v3, v8, s0 // (op 6710) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6710) Load value

v_accvgpr_read v10, a68 // (op 6742) 
v_accvgpr_read v11, a69 // (op 6742) 
v_accvgpr_read v12, a70 // (op 6742) 
v_accvgpr_read v13, a71 // (op 6742) 
v_mul_f32 v14, v7, v10 // (op 6742) 
v_mul_f32 v15, v7, v11 // (op 6742) 
v_mul_f32 v16, v7, v12 // (op 6742) 
v_mul_f32 v17, v7, v13 // (op 6742) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6707) 
v_lshrrev_b32 v11, 16, v4 // (op 6707) 
v_cvt_f32_f16 v11, v11 // (op 6707) 
v_mul_f32 v18, v0, v10 // (op 6707) 
v_mul_f32 v19, v0, v11 // (op 6707) 
v_cvt_f32_f16 v10, v5 // (op 6707) 
v_lshrrev_b32 v11, 16, v5 // (op 6707) 
v_cvt_f32_f16 v11, v11 // (op 6707) 
v_mul_f32 v20, v0, v10 // (op 6707) 
v_mul_f32 v21, v0, v11 // (op 6707) 


v_add_f32 v10, v18, v14 // (op 6708) 
v_add_f32 v11, v19, v15 // (op 6708) 
v_add_f32 v12, v20, v16 // (op 6708) 
v_add_f32 v13, v21, v17 // (op 6708) 


v_cvt_f16_f32 v14, v10 // (op 6709) 
v_cvt_f16_f32 v15, v11 // (op 6709) 
v_cvt_f16_f32 v16, v12 // (op 6709) 
v_cvt_f16_f32 v17, v13 // (op 6709) 


v_pack_b32_f16 v10, v14, v15 // (op 6690) 
v_pack_b32_f16 v11, v16, v17 // (op 6690) 

s_lshl_b32 s0, s4, 5 // (op 6690) 
s_add_u32 s0, s0, 64 // (op 6690) 
v_add_u32 v3, v9, s0 // (op 6690) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6690) Store value





s_lshl_b32 s0, s2, 6 // (op 6715) 
s_add_u32 s0, s0, 64 // (op 6715) 
v_add_u32 v3, v8, s0 // (op 6715) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6715) Load value

v_accvgpr_read v10, a72 // (op 6744) 
v_accvgpr_read v11, a73 // (op 6744) 
v_accvgpr_read v12, a74 // (op 6744) 
v_accvgpr_read v13, a75 // (op 6744) 
v_mul_f32 v14, v7, v10 // (op 6744) 
v_mul_f32 v15, v7, v11 // (op 6744) 
v_mul_f32 v16, v7, v12 // (op 6744) 
v_mul_f32 v17, v7, v13 // (op 6744) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6712) 
v_lshrrev_b32 v11, 16, v4 // (op 6712) 
v_cvt_f32_f16 v11, v11 // (op 6712) 
v_mul_f32 v18, v0, v10 // (op 6712) 
v_mul_f32 v19, v0, v11 // (op 6712) 
v_cvt_f32_f16 v10, v5 // (op 6712) 
v_lshrrev_b32 v11, 16, v5 // (op 6712) 
v_cvt_f32_f16 v11, v11 // (op 6712) 
v_mul_f32 v20, v0, v10 // (op 6712) 
v_mul_f32 v21, v0, v11 // (op 6712) 


v_add_f32 v10, v18, v14 // (op 6713) 
v_add_f32 v11, v19, v15 // (op 6713) 
v_add_f32 v12, v20, v16 // (op 6713) 
v_add_f32 v13, v21, v17 // (op 6713) 


v_cvt_f16_f32 v14, v10 // (op 6714) 
v_cvt_f16_f32 v15, v11 // (op 6714) 
v_cvt_f16_f32 v16, v12 // (op 6714) 
v_cvt_f16_f32 v17, v13 // (op 6714) 


v_pack_b32_f16 v10, v14, v15 // (op 6692) 
v_pack_b32_f16 v11, v16, v17 // (op 6692) 

s_lshl_b32 s0, s4, 6 // (op 6692) 
s_add_u32 s0, s0, 64 // (op 6692) 
v_add_u32 v3, v9, s0 // (op 6692) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6692) Store value





s_lshl_b32 s0, s2, 5 // (op 6720) 
s_mul_i32 s1, 3, s0 // (op 6720) 
s_add_u32 s1, s1, 64 // (op 6720) 

v_add_u32 v3, v8, s1 // (op 6720) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6720) Load value

v_accvgpr_read v10, a76 // (op 6746) 
v_accvgpr_read v11, a77 // (op 6746) 
v_accvgpr_read v12, a78 // (op 6746) 
v_accvgpr_read v13, a79 // (op 6746) 
v_mul_f32 v14, v7, v10 // (op 6746) 
v_mul_f32 v15, v7, v11 // (op 6746) 
v_mul_f32 v16, v7, v12 // (op 6746) 
v_mul_f32 v17, v7, v13 // (op 6746) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6717) 
v_lshrrev_b32 v11, 16, v4 // (op 6717) 
v_cvt_f32_f16 v11, v11 // (op 6717) 
v_mul_f32 v18, v0, v10 // (op 6717) 
v_mul_f32 v19, v0, v11 // (op 6717) 
v_cvt_f32_f16 v10, v5 // (op 6717) 
v_lshrrev_b32 v11, 16, v5 // (op 6717) 
v_cvt_f32_f16 v11, v11 // (op 6717) 
v_mul_f32 v20, v0, v10 // (op 6717) 
v_mul_f32 v21, v0, v11 // (op 6717) 


v_add_f32 v10, v18, v14 // (op 6718) 
v_add_f32 v11, v19, v15 // (op 6718) 
v_add_f32 v12, v20, v16 // (op 6718) 
v_add_f32 v13, v21, v17 // (op 6718) 


v_cvt_f16_f32 v14, v10 // (op 6719) 
v_cvt_f16_f32 v15, v11 // (op 6719) 
v_cvt_f16_f32 v16, v12 // (op 6719) 
v_cvt_f16_f32 v17, v13 // (op 6719) 


v_pack_b32_f16 v10, v14, v15 // (op 6694) 
v_pack_b32_f16 v11, v16, v17 // (op 6694) 

s_lshl_b32 s0, s4, 5 // (op 6694) 
s_mul_i32 s1, 3, s0 // (op 6694) 
s_add_u32 s1, s1, 64 // (op 6694) 

v_add_u32 v3, v9, s1 // (op 6694) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6694) Store value





s_lshl_b32 s0, s2, 7 // (op 6725) 
s_add_u32 s0, s0, 64 // (op 6725) 
v_add_u32 v3, v8, s0 // (op 6725) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6725) Load value

v_accvgpr_read v10, a80 // (op 6748) 
v_accvgpr_read v11, a81 // (op 6748) 
v_accvgpr_read v12, a82 // (op 6748) 
v_accvgpr_read v13, a83 // (op 6748) 
v_mul_f32 v14, v7, v10 // (op 6748) 
v_mul_f32 v15, v7, v11 // (op 6748) 
v_mul_f32 v16, v7, v12 // (op 6748) 
v_mul_f32 v17, v7, v13 // (op 6748) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6722) 
v_lshrrev_b32 v11, 16, v4 // (op 6722) 
v_cvt_f32_f16 v11, v11 // (op 6722) 
v_mul_f32 v18, v0, v10 // (op 6722) 
v_mul_f32 v19, v0, v11 // (op 6722) 
v_cvt_f32_f16 v10, v5 // (op 6722) 
v_lshrrev_b32 v11, 16, v5 // (op 6722) 
v_cvt_f32_f16 v11, v11 // (op 6722) 
v_mul_f32 v20, v0, v10 // (op 6722) 
v_mul_f32 v21, v0, v11 // (op 6722) 


v_add_f32 v10, v18, v14 // (op 6723) 
v_add_f32 v11, v19, v15 // (op 6723) 
v_add_f32 v12, v20, v16 // (op 6723) 
v_add_f32 v13, v21, v17 // (op 6723) 


v_cvt_f16_f32 v14, v10 // (op 6724) 
v_cvt_f16_f32 v15, v11 // (op 6724) 
v_cvt_f16_f32 v16, v12 // (op 6724) 
v_cvt_f16_f32 v17, v13 // (op 6724) 


v_pack_b32_f16 v10, v14, v15 // (op 6696) 
v_pack_b32_f16 v11, v16, v17 // (op 6696) 

s_lshl_b32 s0, s4, 7 // (op 6696) 
s_add_u32 s0, s0, 64 // (op 6696) 
v_add_u32 v3, v9, s0 // (op 6696) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6696) Store value





s_lshl_b32 s0, s2, 5 // (op 6730) 
s_mul_i32 s1, 5, s0 // (op 6730) 
s_add_u32 s1, s1, 64 // (op 6730) 

v_add_u32 v3, v8, s1 // (op 6730) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6730) Load value

v_accvgpr_read v10, a84 // (op 6750) 
v_accvgpr_read v11, a85 // (op 6750) 
v_accvgpr_read v12, a86 // (op 6750) 
v_accvgpr_read v13, a87 // (op 6750) 
v_mul_f32 v14, v7, v10 // (op 6750) 
v_mul_f32 v15, v7, v11 // (op 6750) 
v_mul_f32 v16, v7, v12 // (op 6750) 
v_mul_f32 v17, v7, v13 // (op 6750) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6727) 
v_lshrrev_b32 v11, 16, v4 // (op 6727) 
v_cvt_f32_f16 v11, v11 // (op 6727) 
v_mul_f32 v18, v0, v10 // (op 6727) 
v_mul_f32 v19, v0, v11 // (op 6727) 
v_cvt_f32_f16 v10, v5 // (op 6727) 
v_lshrrev_b32 v11, 16, v5 // (op 6727) 
v_cvt_f32_f16 v11, v11 // (op 6727) 
v_mul_f32 v20, v0, v10 // (op 6727) 
v_mul_f32 v21, v0, v11 // (op 6727) 


v_add_f32 v10, v18, v14 // (op 6728) 
v_add_f32 v11, v19, v15 // (op 6728) 
v_add_f32 v12, v20, v16 // (op 6728) 
v_add_f32 v13, v21, v17 // (op 6728) 


v_cvt_f16_f32 v14, v10 // (op 6729) 
v_cvt_f16_f32 v15, v11 // (op 6729) 
v_cvt_f16_f32 v16, v12 // (op 6729) 
v_cvt_f16_f32 v17, v13 // (op 6729) 


v_pack_b32_f16 v10, v14, v15 // (op 6698) 
v_pack_b32_f16 v11, v16, v17 // (op 6698) 

s_lshl_b32 s0, s4, 5 // (op 6698) 
s_mul_i32 s1, 5, s0 // (op 6698) 
s_add_u32 s1, s1, 64 // (op 6698) 

v_add_u32 v3, v9, s1 // (op 6698) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6698) Store value





s_lshl_b32 s0, s2, 5 // (op 6735) 
s_mul_i32 s1, 6, s0 // (op 6735) 
s_add_u32 s1, s1, 64 // (op 6735) 

v_add_u32 v3, v8, s1 // (op 6735) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6735) Load value

v_accvgpr_read v10, a88 // (op 6752) 
v_accvgpr_read v11, a89 // (op 6752) 
v_accvgpr_read v12, a90 // (op 6752) 
v_accvgpr_read v13, a91 // (op 6752) 
v_mul_f32 v14, v7, v10 // (op 6752) 
v_mul_f32 v15, v7, v11 // (op 6752) 
v_mul_f32 v16, v7, v12 // (op 6752) 
v_mul_f32 v17, v7, v13 // (op 6752) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6732) 
v_lshrrev_b32 v11, 16, v4 // (op 6732) 
v_cvt_f32_f16 v11, v11 // (op 6732) 
v_mul_f32 v18, v0, v10 // (op 6732) 
v_mul_f32 v19, v0, v11 // (op 6732) 
v_cvt_f32_f16 v10, v5 // (op 6732) 
v_lshrrev_b32 v11, 16, v5 // (op 6732) 
v_cvt_f32_f16 v11, v11 // (op 6732) 
v_mul_f32 v20, v0, v10 // (op 6732) 
v_mul_f32 v21, v0, v11 // (op 6732) 


v_add_f32 v10, v18, v14 // (op 6733) 
v_add_f32 v11, v19, v15 // (op 6733) 
v_add_f32 v12, v20, v16 // (op 6733) 
v_add_f32 v13, v21, v17 // (op 6733) 


v_cvt_f16_f32 v14, v10 // (op 6734) 
v_cvt_f16_f32 v15, v11 // (op 6734) 
v_cvt_f16_f32 v16, v12 // (op 6734) 
v_cvt_f16_f32 v17, v13 // (op 6734) 


v_pack_b32_f16 v10, v14, v15 // (op 6700) 
v_pack_b32_f16 v11, v16, v17 // (op 6700) 

s_lshl_b32 s0, s4, 5 // (op 6700) 
s_mul_i32 s1, 6, s0 // (op 6700) 
s_add_u32 s1, s1, 64 // (op 6700) 

v_add_u32 v3, v9, s1 // (op 6700) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6700) Store value





s_lshl_b32 s0, s2, 5 // (op 6740) 
s_mul_i32 s1, 7, s0 // (op 6740) 
s_add_u32 s1, s1, 64 // (op 6740) 

v_add_u32 v3, v8, s1 // (op 6740) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6740) Load value

v_accvgpr_read v10, a92 // (op 6754) 
v_accvgpr_read v11, a93 // (op 6754) 
v_accvgpr_read v12, a94 // (op 6754) 
v_accvgpr_read v13, a95 // (op 6754) 
v_mul_f32 v14, v7, v10 // (op 6754) 
v_mul_f32 v15, v7, v11 // (op 6754) 
v_mul_f32 v16, v7, v12 // (op 6754) 
v_mul_f32 v17, v7, v13 // (op 6754) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6737) 
v_lshrrev_b32 v11, 16, v4 // (op 6737) 
v_cvt_f32_f16 v11, v11 // (op 6737) 
v_mul_f32 v18, v0, v10 // (op 6737) 
v_mul_f32 v19, v0, v11 // (op 6737) 
v_cvt_f32_f16 v10, v5 // (op 6737) 
v_lshrrev_b32 v11, 16, v5 // (op 6737) 
v_cvt_f32_f16 v11, v11 // (op 6737) 
v_mul_f32 v20, v0, v10 // (op 6737) 
v_mul_f32 v21, v0, v11 // (op 6737) 


v_add_f32 v10, v18, v14 // (op 6738) 
v_add_f32 v11, v19, v15 // (op 6738) 
v_add_f32 v12, v20, v16 // (op 6738) 
v_add_f32 v13, v21, v17 // (op 6738) 


v_cvt_f16_f32 v14, v10 // (op 6739) 
v_cvt_f16_f32 v15, v11 // (op 6739) 
v_cvt_f16_f32 v16, v12 // (op 6739) 
v_cvt_f16_f32 v17, v13 // (op 6739) 


v_pack_b32_f16 v10, v14, v15 // (op 6702) 
v_pack_b32_f16 v11, v16, v17 // (op 6702) 

s_lshl_b32 s0, s4, 5 // (op 6702) 
s_mul_i32 s1, 7, s0 // (op 6702) 
s_add_u32 s1, s1, 64 // (op 6702) 

v_add_u32 v3, v9, s1 // (op 6702) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6702) Store value





v_add_u32 v3, 32, v8 // (op 6534) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6534) Load value

v_accvgpr_read v10, a32 // (op 6513) 
v_accvgpr_read v11, a33 // (op 6513) 
v_accvgpr_read v12, a34 // (op 6513) 
v_accvgpr_read v13, a35 // (op 6513) 
v_mul_f32 v14, v7, v10 // (op 6513) 
v_mul_f32 v15, v7, v11 // (op 6513) 
v_mul_f32 v16, v7, v12 // (op 6513) 
v_mul_f32 v17, v7, v13 // (op 6513) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6533) 
v_lshrrev_b32 v11, 16, v4 // (op 6533) 
v_cvt_f32_f16 v11, v11 // (op 6533) 
v_mul_f32 v18, v0, v10 // (op 6533) 
v_mul_f32 v19, v0, v11 // (op 6533) 
v_cvt_f32_f16 v10, v5 // (op 6533) 
v_lshrrev_b32 v11, 16, v5 // (op 6533) 
v_cvt_f32_f16 v11, v11 // (op 6533) 
v_mul_f32 v20, v0, v10 // (op 6533) 
v_mul_f32 v21, v0, v11 // (op 6533) 


v_add_f32 v10, v18, v14 // (op 6514) 
v_add_f32 v11, v19, v15 // (op 6514) 
v_add_f32 v12, v20, v16 // (op 6514) 
v_add_f32 v13, v21, v17 // (op 6514) 


v_cvt_f16_f32 v14, v10 // (op 6515) 
v_cvt_f16_f32 v15, v11 // (op 6515) 
v_cvt_f16_f32 v16, v12 // (op 6515) 
v_cvt_f16_f32 v17, v13 // (op 6515) 


v_pack_b32_f16 v10, v14, v15 // (op 6517) 
v_pack_b32_f16 v11, v16, v17 // (op 6517) 

v_add_u32 v3, 32, v9 // (op 6517) 

buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6517) Store value





s_lshl_b32 s0, s2, 5 // (op 6539) 
s_add_u32 s0, s0, 32 // (op 6539) 
v_add_u32 v3, v8, s0 // (op 6539) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6539) Load value

v_accvgpr_read v10, a36 // (op 6571) 
v_accvgpr_read v11, a37 // (op 6571) 
v_accvgpr_read v12, a38 // (op 6571) 
v_accvgpr_read v13, a39 // (op 6571) 
v_mul_f32 v14, v7, v10 // (op 6571) 
v_mul_f32 v15, v7, v11 // (op 6571) 
v_mul_f32 v16, v7, v12 // (op 6571) 
v_mul_f32 v17, v7, v13 // (op 6571) 

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


v_add_f32 v10, v18, v14 // (op 6537) 
v_add_f32 v11, v19, v15 // (op 6537) 
v_add_f32 v12, v20, v16 // (op 6537) 
v_add_f32 v13, v21, v17 // (op 6537) 


v_cvt_f16_f32 v14, v10 // (op 6538) 
v_cvt_f16_f32 v15, v11 // (op 6538) 
v_cvt_f16_f32 v16, v12 // (op 6538) 
v_cvt_f16_f32 v17, v13 // (op 6538) 


v_pack_b32_f16 v10, v14, v15 // (op 6519) 
v_pack_b32_f16 v11, v16, v17 // (op 6519) 

s_lshl_b32 s0, s4, 5 // (op 6519) 
s_add_u32 s0, s0, 32 // (op 6519) 
v_add_u32 v3, v9, s0 // (op 6519) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6519) Store value





s_lshl_b32 s0, s2, 6 // (op 6544) 
s_add_u32 s0, s0, 32 // (op 6544) 
v_add_u32 v3, v8, s0 // (op 6544) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6544) Load value

v_accvgpr_read v10, a40 // (op 6573) 
v_accvgpr_read v11, a41 // (op 6573) 
v_accvgpr_read v12, a42 // (op 6573) 
v_accvgpr_read v13, a43 // (op 6573) 
v_mul_f32 v14, v7, v10 // (op 6573) 
v_mul_f32 v15, v7, v11 // (op 6573) 
v_mul_f32 v16, v7, v12 // (op 6573) 
v_mul_f32 v17, v7, v13 // (op 6573) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6541) 
v_lshrrev_b32 v11, 16, v4 // (op 6541) 
v_cvt_f32_f16 v11, v11 // (op 6541) 
v_mul_f32 v18, v0, v10 // (op 6541) 
v_mul_f32 v19, v0, v11 // (op 6541) 
v_cvt_f32_f16 v10, v5 // (op 6541) 
v_lshrrev_b32 v11, 16, v5 // (op 6541) 
v_cvt_f32_f16 v11, v11 // (op 6541) 
v_mul_f32 v20, v0, v10 // (op 6541) 
v_mul_f32 v21, v0, v11 // (op 6541) 


v_add_f32 v10, v18, v14 // (op 6542) 
v_add_f32 v11, v19, v15 // (op 6542) 
v_add_f32 v12, v20, v16 // (op 6542) 
v_add_f32 v13, v21, v17 // (op 6542) 


v_cvt_f16_f32 v14, v10 // (op 6543) 
v_cvt_f16_f32 v15, v11 // (op 6543) 
v_cvt_f16_f32 v16, v12 // (op 6543) 
v_cvt_f16_f32 v17, v13 // (op 6543) 


v_pack_b32_f16 v10, v14, v15 // (op 6521) 
v_pack_b32_f16 v11, v16, v17 // (op 6521) 

s_lshl_b32 s0, s4, 6 // (op 6521) 
s_add_u32 s0, s0, 32 // (op 6521) 
v_add_u32 v3, v9, s0 // (op 6521) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6521) Store value





s_lshl_b32 s0, s2, 5 // (op 6549) 
s_mul_i32 s1, 3, s0 // (op 6549) 
s_add_u32 s1, s1, 32 // (op 6549) 

v_add_u32 v3, v8, s1 // (op 6549) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6549) Load value

v_accvgpr_read v10, a44 // (op 6575) 
v_accvgpr_read v11, a45 // (op 6575) 
v_accvgpr_read v12, a46 // (op 6575) 
v_accvgpr_read v13, a47 // (op 6575) 
v_mul_f32 v14, v7, v10 // (op 6575) 
v_mul_f32 v15, v7, v11 // (op 6575) 
v_mul_f32 v16, v7, v12 // (op 6575) 
v_mul_f32 v17, v7, v13 // (op 6575) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6546) 
v_lshrrev_b32 v11, 16, v4 // (op 6546) 
v_cvt_f32_f16 v11, v11 // (op 6546) 
v_mul_f32 v18, v0, v10 // (op 6546) 
v_mul_f32 v19, v0, v11 // (op 6546) 
v_cvt_f32_f16 v10, v5 // (op 6546) 
v_lshrrev_b32 v11, 16, v5 // (op 6546) 
v_cvt_f32_f16 v11, v11 // (op 6546) 
v_mul_f32 v20, v0, v10 // (op 6546) 
v_mul_f32 v21, v0, v11 // (op 6546) 


v_add_f32 v10, v18, v14 // (op 6547) 
v_add_f32 v11, v19, v15 // (op 6547) 
v_add_f32 v12, v20, v16 // (op 6547) 
v_add_f32 v13, v21, v17 // (op 6547) 


v_cvt_f16_f32 v14, v10 // (op 6548) 
v_cvt_f16_f32 v15, v11 // (op 6548) 
v_cvt_f16_f32 v16, v12 // (op 6548) 
v_cvt_f16_f32 v17, v13 // (op 6548) 


v_pack_b32_f16 v10, v14, v15 // (op 6523) 
v_pack_b32_f16 v11, v16, v17 // (op 6523) 

s_lshl_b32 s0, s4, 5 // (op 6523) 
s_mul_i32 s1, 3, s0 // (op 6523) 
s_add_u32 s1, s1, 32 // (op 6523) 

v_add_u32 v3, v9, s1 // (op 6523) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6523) Store value





s_lshl_b32 s0, s2, 7 // (op 6554) 
s_add_u32 s0, s0, 32 // (op 6554) 
v_add_u32 v3, v8, s0 // (op 6554) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6554) Load value

v_accvgpr_read v10, a48 // (op 6577) 
v_accvgpr_read v11, a49 // (op 6577) 
v_accvgpr_read v12, a50 // (op 6577) 
v_accvgpr_read v13, a51 // (op 6577) 
v_mul_f32 v14, v7, v10 // (op 6577) 
v_mul_f32 v15, v7, v11 // (op 6577) 
v_mul_f32 v16, v7, v12 // (op 6577) 
v_mul_f32 v17, v7, v13 // (op 6577) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6551) 
v_lshrrev_b32 v11, 16, v4 // (op 6551) 
v_cvt_f32_f16 v11, v11 // (op 6551) 
v_mul_f32 v18, v0, v10 // (op 6551) 
v_mul_f32 v19, v0, v11 // (op 6551) 
v_cvt_f32_f16 v10, v5 // (op 6551) 
v_lshrrev_b32 v11, 16, v5 // (op 6551) 
v_cvt_f32_f16 v11, v11 // (op 6551) 
v_mul_f32 v20, v0, v10 // (op 6551) 
v_mul_f32 v21, v0, v11 // (op 6551) 


v_add_f32 v10, v18, v14 // (op 6552) 
v_add_f32 v11, v19, v15 // (op 6552) 
v_add_f32 v12, v20, v16 // (op 6552) 
v_add_f32 v13, v21, v17 // (op 6552) 


v_cvt_f16_f32 v14, v10 // (op 6553) 
v_cvt_f16_f32 v15, v11 // (op 6553) 
v_cvt_f16_f32 v16, v12 // (op 6553) 
v_cvt_f16_f32 v17, v13 // (op 6553) 


v_pack_b32_f16 v10, v14, v15 // (op 6525) 
v_pack_b32_f16 v11, v16, v17 // (op 6525) 

s_lshl_b32 s0, s4, 7 // (op 6525) 
s_add_u32 s0, s0, 32 // (op 6525) 
v_add_u32 v3, v9, s0 // (op 6525) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6525) Store value





s_lshl_b32 s0, s2, 5 // (op 6559) 
s_mul_i32 s1, 5, s0 // (op 6559) 
s_add_u32 s1, s1, 32 // (op 6559) 

v_add_u32 v3, v8, s1 // (op 6559) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6559) Load value

v_accvgpr_read v10, a52 // (op 6579) 
v_accvgpr_read v11, a53 // (op 6579) 
v_accvgpr_read v12, a54 // (op 6579) 
v_accvgpr_read v13, a55 // (op 6579) 
v_mul_f32 v14, v7, v10 // (op 6579) 
v_mul_f32 v15, v7, v11 // (op 6579) 
v_mul_f32 v16, v7, v12 // (op 6579) 
v_mul_f32 v17, v7, v13 // (op 6579) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6556) 
v_lshrrev_b32 v11, 16, v4 // (op 6556) 
v_cvt_f32_f16 v11, v11 // (op 6556) 
v_mul_f32 v18, v0, v10 // (op 6556) 
v_mul_f32 v19, v0, v11 // (op 6556) 
v_cvt_f32_f16 v10, v5 // (op 6556) 
v_lshrrev_b32 v11, 16, v5 // (op 6556) 
v_cvt_f32_f16 v11, v11 // (op 6556) 
v_mul_f32 v20, v0, v10 // (op 6556) 
v_mul_f32 v21, v0, v11 // (op 6556) 


v_add_f32 v10, v18, v14 // (op 6557) 
v_add_f32 v11, v19, v15 // (op 6557) 
v_add_f32 v12, v20, v16 // (op 6557) 
v_add_f32 v13, v21, v17 // (op 6557) 


v_cvt_f16_f32 v14, v10 // (op 6558) 
v_cvt_f16_f32 v15, v11 // (op 6558) 
v_cvt_f16_f32 v16, v12 // (op 6558) 
v_cvt_f16_f32 v17, v13 // (op 6558) 


v_pack_b32_f16 v10, v14, v15 // (op 6527) 
v_pack_b32_f16 v11, v16, v17 // (op 6527) 

s_lshl_b32 s0, s4, 5 // (op 6527) 
s_mul_i32 s1, 5, s0 // (op 6527) 
s_add_u32 s1, s1, 32 // (op 6527) 

v_add_u32 v3, v9, s1 // (op 6527) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6527) Store value





s_lshl_b32 s0, s2, 5 // (op 6564) 
s_mul_i32 s1, 6, s0 // (op 6564) 
s_add_u32 s1, s1, 32 // (op 6564) 

v_add_u32 v3, v8, s1 // (op 6564) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6564) Load value

v_accvgpr_read v10, a56 // (op 6581) 
v_accvgpr_read v11, a57 // (op 6581) 
v_accvgpr_read v12, a58 // (op 6581) 
v_accvgpr_read v13, a59 // (op 6581) 
v_mul_f32 v14, v7, v10 // (op 6581) 
v_mul_f32 v15, v7, v11 // (op 6581) 
v_mul_f32 v16, v7, v12 // (op 6581) 
v_mul_f32 v17, v7, v13 // (op 6581) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6561) 
v_lshrrev_b32 v11, 16, v4 // (op 6561) 
v_cvt_f32_f16 v11, v11 // (op 6561) 
v_mul_f32 v18, v0, v10 // (op 6561) 
v_mul_f32 v19, v0, v11 // (op 6561) 
v_cvt_f32_f16 v10, v5 // (op 6561) 
v_lshrrev_b32 v11, 16, v5 // (op 6561) 
v_cvt_f32_f16 v11, v11 // (op 6561) 
v_mul_f32 v20, v0, v10 // (op 6561) 
v_mul_f32 v21, v0, v11 // (op 6561) 


v_add_f32 v10, v18, v14 // (op 6562) 
v_add_f32 v11, v19, v15 // (op 6562) 
v_add_f32 v12, v20, v16 // (op 6562) 
v_add_f32 v13, v21, v17 // (op 6562) 


v_cvt_f16_f32 v14, v10 // (op 6563) 
v_cvt_f16_f32 v15, v11 // (op 6563) 
v_cvt_f16_f32 v16, v12 // (op 6563) 
v_cvt_f16_f32 v17, v13 // (op 6563) 


v_pack_b32_f16 v10, v14, v15 // (op 6529) 
v_pack_b32_f16 v11, v16, v17 // (op 6529) 

s_lshl_b32 s0, s4, 5 // (op 6529) 
s_mul_i32 s1, 6, s0 // (op 6529) 
s_add_u32 s1, s1, 32 // (op 6529) 

v_add_u32 v3, v9, s1 // (op 6529) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6529) Store value





s_lshl_b32 s0, s2, 5 // (op 6569) 
s_mul_i32 s1, 7, s0 // (op 6569) 
s_add_u32 s1, s1, 32 // (op 6569) 

v_add_u32 v3, v8, s1 // (op 6569) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6569) Load value

v_accvgpr_read v10, a60 // (op 6583) 
v_accvgpr_read v11, a61 // (op 6583) 
v_accvgpr_read v12, a62 // (op 6583) 
v_accvgpr_read v13, a63 // (op 6583) 
v_mul_f32 v14, v7, v10 // (op 6583) 
v_mul_f32 v15, v7, v11 // (op 6583) 
v_mul_f32 v16, v7, v12 // (op 6583) 
v_mul_f32 v17, v7, v13 // (op 6583) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6566) 
v_lshrrev_b32 v11, 16, v4 // (op 6566) 
v_cvt_f32_f16 v11, v11 // (op 6566) 
v_mul_f32 v18, v0, v10 // (op 6566) 
v_mul_f32 v19, v0, v11 // (op 6566) 
v_cvt_f32_f16 v10, v5 // (op 6566) 
v_lshrrev_b32 v11, 16, v5 // (op 6566) 
v_cvt_f32_f16 v11, v11 // (op 6566) 
v_mul_f32 v20, v0, v10 // (op 6566) 
v_mul_f32 v21, v0, v11 // (op 6566) 


v_add_f32 v10, v18, v14 // (op 6567) 
v_add_f32 v11, v19, v15 // (op 6567) 
v_add_f32 v12, v20, v16 // (op 6567) 
v_add_f32 v13, v21, v17 // (op 6567) 


v_cvt_f16_f32 v14, v10 // (op 6568) 
v_cvt_f16_f32 v15, v11 // (op 6568) 
v_cvt_f16_f32 v16, v12 // (op 6568) 
v_cvt_f16_f32 v17, v13 // (op 6568) 


v_pack_b32_f16 v10, v14, v15 // (op 6531) 
v_pack_b32_f16 v11, v16, v17 // (op 6531) 

s_lshl_b32 s0, s4, 5 // (op 6531) 
s_mul_i32 s1, 7, s0 // (op 6531) 
s_add_u32 s1, s1, 32 // (op 6531) 

v_add_u32 v3, v9, s1 // (op 6531) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6531) Store value





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





s_lshl_b32 s0, s2, 5 // (op 6338) 
v_add_u32 v3, v8, s0 // (op 6338) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6338) Load value

v_accvgpr_read v10, a4 // (op 6334) 
v_accvgpr_read v11, a5 // (op 6334) 
v_accvgpr_read v12, a6 // (op 6334) 
v_accvgpr_read v13, a7 // (op 6334) 
v_mul_f32 v14, v7, v10 // (op 6334) 
v_mul_f32 v15, v7, v11 // (op 6334) 
v_mul_f32 v16, v7, v12 // (op 6334) 
v_mul_f32 v17, v7, v13 // (op 6334) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6339) 
v_lshrrev_b32 v11, 16, v4 // (op 6339) 
v_cvt_f32_f16 v11, v11 // (op 6339) 
v_mul_f32 v18, v0, v10 // (op 6339) 
v_mul_f32 v19, v0, v11 // (op 6339) 
v_cvt_f32_f16 v10, v5 // (op 6339) 
v_lshrrev_b32 v11, 16, v5 // (op 6339) 
v_cvt_f32_f16 v11, v11 // (op 6339) 
v_mul_f32 v20, v0, v10 // (op 6339) 
v_mul_f32 v21, v0, v11 // (op 6339) 


v_add_f32 v10, v18, v14 // (op 6335) 
v_add_f32 v11, v19, v15 // (op 6335) 
v_add_f32 v12, v20, v16 // (op 6335) 
v_add_f32 v13, v21, v17 // (op 6335) 


v_cvt_f16_f32 v14, v10 // (op 6336) 
v_cvt_f16_f32 v15, v11 // (op 6336) 
v_cvt_f16_f32 v16, v12 // (op 6336) 
v_cvt_f16_f32 v17, v13 // (op 6336) 


v_pack_b32_f16 v10, v14, v15 // (op 6337) 
v_pack_b32_f16 v11, v16, v17 // (op 6337) 

s_lshl_b32 s0, s4, 5 // (op 6337) 
v_add_u32 v3, v9, s0 // (op 6337) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6337) Store value





s_lshl_b32 s0, s2, 6 // (op 6352) 
v_add_u32 v3, v8, s0 // (op 6352) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6352) Load value

v_accvgpr_read v10, a8 // (op 6348) 
v_accvgpr_read v11, a9 // (op 6348) 
v_accvgpr_read v12, a10 // (op 6348) 
v_accvgpr_read v13, a11 // (op 6348) 
v_mul_f32 v14, v7, v10 // (op 6348) 
v_mul_f32 v15, v7, v11 // (op 6348) 
v_mul_f32 v16, v7, v12 // (op 6348) 
v_mul_f32 v17, v7, v13 // (op 6348) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6353) 
v_lshrrev_b32 v11, 16, v4 // (op 6353) 
v_cvt_f32_f16 v11, v11 // (op 6353) 
v_mul_f32 v18, v0, v10 // (op 6353) 
v_mul_f32 v19, v0, v11 // (op 6353) 
v_cvt_f32_f16 v10, v5 // (op 6353) 
v_lshrrev_b32 v11, 16, v5 // (op 6353) 
v_cvt_f32_f16 v11, v11 // (op 6353) 
v_mul_f32 v20, v0, v10 // (op 6353) 
v_mul_f32 v21, v0, v11 // (op 6353) 


v_add_f32 v10, v18, v14 // (op 6349) 
v_add_f32 v11, v19, v15 // (op 6349) 
v_add_f32 v12, v20, v16 // (op 6349) 
v_add_f32 v13, v21, v17 // (op 6349) 


v_cvt_f16_f32 v14, v10 // (op 6350) 
v_cvt_f16_f32 v15, v11 // (op 6350) 
v_cvt_f16_f32 v16, v12 // (op 6350) 
v_cvt_f16_f32 v17, v13 // (op 6350) 


v_pack_b32_f16 v10, v14, v15 // (op 6351) 
v_pack_b32_f16 v11, v16, v17 // (op 6351) 

s_lshl_b32 s0, s4, 6 // (op 6351) 
v_add_u32 v3, v9, s0 // (op 6351) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6351) Store value





s_lshl_b32 s0, s2, 5 // (op 6366) 
s_mul_i32 s1, 3, s0 // (op 6366) 

v_add_u32 v3, v8, s1 // (op 6366) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6366) Load value

v_accvgpr_read v10, a12 // (op 6362) 
v_accvgpr_read v11, a13 // (op 6362) 
v_accvgpr_read v12, a14 // (op 6362) 
v_accvgpr_read v13, a15 // (op 6362) 
v_mul_f32 v14, v7, v10 // (op 6362) 
v_mul_f32 v15, v7, v11 // (op 6362) 
v_mul_f32 v16, v7, v12 // (op 6362) 
v_mul_f32 v17, v7, v13 // (op 6362) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6367) 
v_lshrrev_b32 v11, 16, v4 // (op 6367) 
v_cvt_f32_f16 v11, v11 // (op 6367) 
v_mul_f32 v18, v0, v10 // (op 6367) 
v_mul_f32 v19, v0, v11 // (op 6367) 
v_cvt_f32_f16 v10, v5 // (op 6367) 
v_lshrrev_b32 v11, 16, v5 // (op 6367) 
v_cvt_f32_f16 v11, v11 // (op 6367) 
v_mul_f32 v20, v0, v10 // (op 6367) 
v_mul_f32 v21, v0, v11 // (op 6367) 


v_add_f32 v10, v18, v14 // (op 6363) 
v_add_f32 v11, v19, v15 // (op 6363) 
v_add_f32 v12, v20, v16 // (op 6363) 
v_add_f32 v13, v21, v17 // (op 6363) 


v_cvt_f16_f32 v14, v10 // (op 6364) 
v_cvt_f16_f32 v15, v11 // (op 6364) 
v_cvt_f16_f32 v16, v12 // (op 6364) 
v_cvt_f16_f32 v17, v13 // (op 6364) 


v_pack_b32_f16 v10, v14, v15 // (op 6365) 
v_pack_b32_f16 v11, v16, v17 // (op 6365) 

s_lshl_b32 s0, s4, 5 // (op 6365) 
s_mul_i32 s1, 3, s0 // (op 6365) 

v_add_u32 v3, v9, s1 // (op 6365) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6365) Store value





s_lshl_b32 s0, s2, 7 // (op 6380) 
v_add_u32 v3, v8, s0 // (op 6380) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6380) Load value

v_accvgpr_read v10, a16 // (op 6376) 
v_accvgpr_read v11, a17 // (op 6376) 
v_accvgpr_read v12, a18 // (op 6376) 
v_accvgpr_read v13, a19 // (op 6376) 
v_mul_f32 v14, v7, v10 // (op 6376) 
v_mul_f32 v15, v7, v11 // (op 6376) 
v_mul_f32 v16, v7, v12 // (op 6376) 
v_mul_f32 v17, v7, v13 // (op 6376) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6381) 
v_lshrrev_b32 v11, 16, v4 // (op 6381) 
v_cvt_f32_f16 v11, v11 // (op 6381) 
v_mul_f32 v18, v0, v10 // (op 6381) 
v_mul_f32 v19, v0, v11 // (op 6381) 
v_cvt_f32_f16 v10, v5 // (op 6381) 
v_lshrrev_b32 v11, 16, v5 // (op 6381) 
v_cvt_f32_f16 v11, v11 // (op 6381) 
v_mul_f32 v20, v0, v10 // (op 6381) 
v_mul_f32 v21, v0, v11 // (op 6381) 


v_add_f32 v10, v18, v14 // (op 6377) 
v_add_f32 v11, v19, v15 // (op 6377) 
v_add_f32 v12, v20, v16 // (op 6377) 
v_add_f32 v13, v21, v17 // (op 6377) 


v_cvt_f16_f32 v14, v10 // (op 6378) 
v_cvt_f16_f32 v15, v11 // (op 6378) 
v_cvt_f16_f32 v16, v12 // (op 6378) 
v_cvt_f16_f32 v17, v13 // (op 6378) 


v_pack_b32_f16 v10, v14, v15 // (op 6379) 
v_pack_b32_f16 v11, v16, v17 // (op 6379) 

s_lshl_b32 s0, s4, 7 // (op 6379) 
v_add_u32 v3, v9, s0 // (op 6379) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6379) Store value





s_lshl_b32 s0, s2, 5 // (op 6394) 
s_mul_i32 s1, 5, s0 // (op 6394) 

v_add_u32 v3, v8, s1 // (op 6394) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6394) Load value

v_accvgpr_read v10, a20 // (op 6390) 
v_accvgpr_read v11, a21 // (op 6390) 
v_accvgpr_read v12, a22 // (op 6390) 
v_accvgpr_read v13, a23 // (op 6390) 
v_mul_f32 v14, v7, v10 // (op 6390) 
v_mul_f32 v15, v7, v11 // (op 6390) 
v_mul_f32 v16, v7, v12 // (op 6390) 
v_mul_f32 v17, v7, v13 // (op 6390) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6395) 
v_lshrrev_b32 v11, 16, v4 // (op 6395) 
v_cvt_f32_f16 v11, v11 // (op 6395) 
v_mul_f32 v18, v0, v10 // (op 6395) 
v_mul_f32 v19, v0, v11 // (op 6395) 
v_cvt_f32_f16 v10, v5 // (op 6395) 
v_lshrrev_b32 v11, 16, v5 // (op 6395) 
v_cvt_f32_f16 v11, v11 // (op 6395) 
v_mul_f32 v20, v0, v10 // (op 6395) 
v_mul_f32 v21, v0, v11 // (op 6395) 


v_add_f32 v10, v18, v14 // (op 6391) 
v_add_f32 v11, v19, v15 // (op 6391) 
v_add_f32 v12, v20, v16 // (op 6391) 
v_add_f32 v13, v21, v17 // (op 6391) 


v_cvt_f16_f32 v14, v10 // (op 6392) 
v_cvt_f16_f32 v15, v11 // (op 6392) 
v_cvt_f16_f32 v16, v12 // (op 6392) 
v_cvt_f16_f32 v17, v13 // (op 6392) 


v_pack_b32_f16 v10, v14, v15 // (op 6393) 
v_pack_b32_f16 v11, v16, v17 // (op 6393) 

s_lshl_b32 s0, s4, 5 // (op 6393) 
s_mul_i32 s1, 5, s0 // (op 6393) 

v_add_u32 v3, v9, s1 // (op 6393) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6393) Store value





s_lshl_b32 s0, s2, 5 // (op 6408) 
s_mul_i32 s1, 6, s0 // (op 6408) 

v_add_u32 v3, v8, s1 // (op 6408) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6408) Load value

v_accvgpr_read v10, a24 // (op 6404) 
v_accvgpr_read v11, a25 // (op 6404) 
v_accvgpr_read v12, a26 // (op 6404) 
v_accvgpr_read v13, a27 // (op 6404) 
v_mul_f32 v14, v7, v10 // (op 6404) 
v_mul_f32 v15, v7, v11 // (op 6404) 
v_mul_f32 v16, v7, v12 // (op 6404) 
v_mul_f32 v17, v7, v13 // (op 6404) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6409) 
v_lshrrev_b32 v11, 16, v4 // (op 6409) 
v_cvt_f32_f16 v11, v11 // (op 6409) 
v_mul_f32 v18, v0, v10 // (op 6409) 
v_mul_f32 v19, v0, v11 // (op 6409) 
v_cvt_f32_f16 v10, v5 // (op 6409) 
v_lshrrev_b32 v11, 16, v5 // (op 6409) 
v_cvt_f32_f16 v11, v11 // (op 6409) 
v_mul_f32 v20, v0, v10 // (op 6409) 
v_mul_f32 v21, v0, v11 // (op 6409) 


v_add_f32 v10, v18, v14 // (op 6405) 
v_add_f32 v11, v19, v15 // (op 6405) 
v_add_f32 v12, v20, v16 // (op 6405) 
v_add_f32 v13, v21, v17 // (op 6405) 


v_cvt_f16_f32 v14, v10 // (op 6406) 
v_cvt_f16_f32 v15, v11 // (op 6406) 
v_cvt_f16_f32 v16, v12 // (op 6406) 
v_cvt_f16_f32 v17, v13 // (op 6406) 


v_pack_b32_f16 v10, v14, v15 // (op 6407) 
v_pack_b32_f16 v11, v16, v17 // (op 6407) 

s_lshl_b32 s0, s4, 5 // (op 6407) 
s_mul_i32 s1, 6, s0 // (op 6407) 

v_add_u32 v3, v9, s1 // (op 6407) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6407) Store value





s_lshl_b32 s0, s2, 5 // (op 6422) 
s_mul_i32 s1, 7, s0 // (op 6422) 

v_add_u32 v3, v8, s1 // (op 6422) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 6422) Load value

v_accvgpr_read v10, a28 // (op 6418) 
v_accvgpr_read v11, a29 // (op 6418) 
v_accvgpr_read v12, a30 // (op 6418) 
v_accvgpr_read v13, a31 // (op 6418) 
v_mul_f32 v14, v7, v10 // (op 6418) 
v_mul_f32 v15, v7, v11 // (op 6418) 
v_mul_f32 v16, v7, v12 // (op 6418) 
v_mul_f32 v17, v7, v13 // (op 6418) 

s_waitcnt vmcnt(0)
v_cvt_f32_f16 v10, v4 // (op 6423) 
v_lshrrev_b32 v11, 16, v4 // (op 6423) 
v_cvt_f32_f16 v11, v11 // (op 6423) 
v_mul_f32 v18, v0, v10 // (op 6423) 
v_mul_f32 v19, v0, v11 // (op 6423) 
v_cvt_f32_f16 v10, v5 // (op 6423) 
v_lshrrev_b32 v11, 16, v5 // (op 6423) 
v_cvt_f32_f16 v11, v11 // (op 6423) 
v_mul_f32 v20, v0, v10 // (op 6423) 
v_mul_f32 v21, v0, v11 // (op 6423) 


v_add_f32 v10, v18, v14 // (op 6419) 
v_add_f32 v11, v19, v15 // (op 6419) 
v_add_f32 v12, v20, v16 // (op 6419) 
v_add_f32 v13, v21, v17 // (op 6419) 


v_cvt_f16_f32 v14, v10 // (op 6420) 
v_cvt_f16_f32 v15, v11 // (op 6420) 
v_cvt_f16_f32 v16, v12 // (op 6420) 
v_cvt_f16_f32 v17, v13 // (op 6420) 


v_pack_b32_f16 v10, v14, v15 // (op 6421) 
v_pack_b32_f16 v11, v16, v17 // (op 6421) 

s_lshl_b32 s0, s4, 5 // (op 6421) 
s_mul_i32 s1, 7, s0 // (op 6421) 

v_add_u32 v3, v9, s1 // (op 6421) 


buffer_store_dwordx2 v[10:11], v3, s[56:59], 0 offen // (op 6421) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__ConditionalBottom_0_beta_116:








































s_endpgm  // End of RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM_
.LRR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__end:
.size RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM_, .LRR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM__end-RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 316
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 73728 // lds bytes
  .amdhsa_accum_offset 188
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
    macrotile: [128, 256, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [4, 8]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x256x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 73728
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 72
    .vgpr_count: 186
    .agpr_count: 128
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {is-null: true}, {is-null: true}]
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
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 28
        .expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}}
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
        .expression: {type: MagicMultiple, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 52
        .expression: {type: MagicShiftAndSign, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}
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
        .expression: {type: MagicShiftAndSign, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_24, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 160, size: 4}, rhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_27
        .size: 4
        .offset: 156
        .expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 7}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}}
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

