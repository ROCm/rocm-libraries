// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT



.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_,@function
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 2664) 
s_mul_hi_i32 s1, s16, s10 // (op 2906) 
v_and_b32 v0, 63, v1 // (op 2721) 
v_lshrrev_b32 v3, 4, v0 // (op 2721) 

v_and_b32 v0, 1, v3 // (op 2721) 

v_and_b32 v3, 15, v1 // (op 2721) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 2721) 


v_lshrrev_b32 v0, 7, v1 // (op 2721) 
v_lshl_add_u32 v3, v0, 5, v4 // (op 2721) 


v_and_b32 v0, 32, v1 // (op 2721) 
v_lshl_add_u32 v4, v3, 7, v0 // (op 2721) 


v_mul_lo_u32 v0, v4, 6 // (op 2721) 

v_lshrrev_b32 v3, 3, v0 // (op 2721) 

v_lshrrev_b32 v0, 7, v1 // (op 2832) 
v_and_b32 v4, 31, v1 // (op 2832) 
v_lshl_add_u32 v5, v0, 5, v4 // (op 2832) 


v_and_b32 v0, 63, v1 // (op 2832) 
v_lshrrev_b32 v4, 5, v0 // (op 2832) 

v_lshl_add_u32 v0, v5, 2, v4 // (op 2832) 


v_and_b32 v4, 63, v1 // (op 2932) 
v_lshrrev_b32 v5, 4, v4 // (op 2932) 

v_and_b32 v4, 1, v5 // (op 2932) 

v_and_b32 v5, 15, v1 // (op 2932) 
v_lshl_add_u32 v6, v4, 4, v5 // (op 2932) 


v_lshrrev_b32 v4, 7, v1 // (op 2932) 
v_lshl_add_u32 v5, v4, 5, v6 // (op 2932) 


v_and_b32 v4, 32, v1 // (op 2932) 
v_lshl_add_u32 v6, v5, 7, v4 // (op 2932) 


v_mul_lo_u32 v4, v6, 6 // (op 2932) 

v_lshrrev_b32 v5, 3, v4 // (op 2932) 

v_lshrrev_b32 v4, 7, v1 // (op 2985) 
v_and_b32 v6, 31, v1 // (op 2985) 
v_lshl_add_u32 v7, v4, 5, v6 // (op 2985) 


v_and_b32 v4, 63, v1 // (op 2985) 
v_lshrrev_b32 v6, 5, v4 // (op 2985) 

v_lshl_add_u32 v4, v7, 2, v6 // (op 2985) 


v_lshrrev_b32 v6, 7, v1 // (op 2664) 
s_add_i32 s18, s0, s16 // (op 2664) 

v_and_b32 v7, 63, v1 // (op 2664) 
s_ashr_i32 s0, s18, 31 // (op 2664) 
v_lshrrev_b32 v8, 5, v7 // (op 2664) 
s_cmp_eq_i32 s10, 0 // (op 2664) 
s_mov_b32 s19, scc // (op 2664) 
v_lshlrev_b32 v9, 2, v8 // (op 2664) 

s_cmp_eq_i32 s19, 0 // (op 2664) 
s_cselect_b32 s49, 0, -1 // (op 2664) 

v_lshl_add_u32 v8, v6, 5, v9 // (op 2664) 


s_and_b32 s19, s13, 31 // (op 2664) 
v_lshrrev_b32 v6, 2, v7 // (op 2664) 

v_and_b32 v7, 7, v6 // (op 2664) 

v_and_b32 v6, 63, v1 // (op 2779) 
v_lshrrev_b32 v9, 4, v6 // (op 2779) 

v_and_b32 v6, 1, v9 // (op 2779) 

v_and_b32 v9, 15, v1 // (op 2779) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 2779) 


v_lshrrev_b32 v6, 6, v1 // (op 2779) 
v_and_b32 v9, 1, v6 // (op 2779) 

v_lshl_add_u32 v6, v9, 5, v10 // (op 2779) 


v_and_b32 v9, 32, v1 // (op 2779) 
v_lshl_add_u32 v10, v6, 7, v9 // (op 2779) 


v_mul_lo_u32 v6, v10, 6 // (op 2779) 

v_lshrrev_b32 v9, 3, v6 // (op 2779) 

v_lshrrev_b32 v6, 6, v1 // (op 2890) 
v_and_b32 v10, 1, v6 // (op 2890) 

v_and_b32 v6, 31, v1 // (op 2890) 
v_lshl_add_u32 v11, v10, 5, v6 // (op 2890) 


v_and_b32 v6, 63, v1 // (op 2890) 
v_lshrrev_b32 v10, 5, v6 // (op 2890) 

v_lshl_add_u32 v6, v11, 2, v10 // (op 2890) 


v_lshrrev_b32 v10, 7, v1 // (op 2906) 
v_and_b32 v11, 63, v1 // (op 2906) 
v_and_b32 v12, 63, v1 // (op 2961) 
v_lshrrev_b32 v13, 4, v12 // (op 2961) 

v_and_b32 v12, 1, v13 // (op 2961) 

v_and_b32 v13, 15, v1 // (op 2961) 
v_lshl_add_u32 v14, v12, 4, v13 // (op 2961) 


v_lshrrev_b32 v12, 6, v1 // (op 2961) 
v_and_b32 v13, 1, v12 // (op 2961) 

v_lshl_add_u32 v12, v13, 5, v14 // (op 2961) 


v_and_b32 v13, 32, v1 // (op 2961) 
v_lshl_add_u32 v14, v12, 7, v13 // (op 2961) 


v_mul_lo_u32 v12, v14, 6 // (op 2961) 

v_lshrrev_b32 v13, 3, v12 // (op 2961) 

v_lshrrev_b32 v12, 6, v1 // (op 3014) 
v_and_b32 v14, 1, v12 // (op 3014) 

v_and_b32 v12, 31, v1 // (op 3014) 
v_lshl_add_u32 v15, v14, 5, v12 // (op 3014) 


v_and_b32 v12, 63, v1 // (op 3014) 
v_lshrrev_b32 v14, 5, v12 // (op 3014) 

v_lshl_add_u32 v12, v15, 2, v14 // (op 3014) 


s_lshl_b32 s50, 1, s19 // (op 2664) 
v_and_b32 v14, 3, v1 // (op 2664) 
s_add_i32 s51, s50, s49 // (op 2664) 


s_and_b32 s49, s0, s51 // (op 2664) 


s_add_i32 s0, s1, s16 // (op 2906) 

s_ashr_i32 s1, s0, 31 // (op 2906) 
v_lshl_add_u32 v15, v7, 2, v14 // (op 2664) 


s_add_i32 s50, s18, s49 // (op 2664) 


v_lshrrev_b32 v7, 6, v1 // (op 2664) 
s_ashr_i32 s18, s50, s19 // (op 2664) 


v_and_b32 v14, 1, v7 // (op 2664) 

s_lshl_b32 s19, s13, 24 // (op 2664) 
v_lshl_add_u32 v7, v14, 5, v15 // (op 2664) 


s_ashr_i32 s49, s19, 31 // (op 2664) 

s_xor_b32 s19, s18, s49 // (op 2664) 

s_sub_i32 s18, s19, s49 // (op 2664) 


s_mul_hi_i32 s19, s18, s14 // (op 2664) 
s_add_i32 s49, s19, s18 // (op 2664) 

s_ashr_i32 s19, s49, 31 // (op 2664) 
s_cmp_eq_i32 s14, 0 // (op 2664) 
s_mov_b32 s50, scc // (op 2664) 
s_cmp_eq_i32 s50, 0 // (op 2664) 
s_cselect_b32 s51, 0, -1 // (op 2664) 

s_and_b32 s50, s15, 31 // (op 2664) 
v_lshrrev_b32 v14, 5, v11 // (op 2906) 
s_cmp_eq_i32 s10, 0 // (op 2906) 
s_mov_b32 s52, scc // (op 2906) 
v_lshlrev_b32 v15, 2, v14 // (op 2906) 

s_cmp_eq_i32 s52, 0 // (op 2906) 
s_cselect_b32 s53, 0, -1 // (op 2906) 

s_lshl_b32 s52, 1, s50 // (op 2664) 
v_lshl_add_u32 v14, v10, 5, v15 // (op 2906) 


v_lshrrev_b32 v10, 2, v11 // (op 2906) 

s_add_i32 s54, s52, s51 // (op 2664) 


s_and_b32 s51, s19, s54 // (op 2664) 


s_add_i32 s19, s49, s51 // (op 2664) 


s_ashr_i32 s49, s19, s50 // (op 2664) 


s_lshl_b32 s19, s15, 24 // (op 2664) 
s_ashr_i32 s50, s19, 31 // (op 2664) 

s_xor_b32 s19, s49, s50 // (op 2664) 

s_sub_i32 s49, s19, s50 // (op 2664) 


s_mul_i32 s19, s18, s11 // (op 2664) 
s_sub_u32 s50, s16, s19 // (op 2664) 

s_mul_i32 s19, s49, s11 // (op 2664) 
s_add_u32 s19, s19, s50 // (op 2664) 

s_mul_hi_i32 s50, s16, s9 // (op 2664) 
s_add_i32 s51, s50, s16 // (op 2664) 

s_ashr_i32 s50, s51, 31 // (op 2664) 
s_cmp_eq_i32 s9, 0 // (op 2664) 
s_mov_b32 s52, scc // (op 2664) 
s_cmp_eq_i32 s52, 0 // (op 2664) 
s_cselect_b32 s54, 0, -1 // (op 2664) 

s_and_b32 s52, s13, 31 // (op 2906) 
v_and_b32 v11, 7, v10 // (op 2906) 

s_lshl_b32 s55, 1, s52 // (op 2906) 
v_and_b32 v10, 3, v1 // (op 2906) 
s_add_i32 s56, s55, s53 // (op 2906) 


s_and_b32 s53, s1, s56 // (op 2906) 


v_lshl_add_u32 v15, v11, 2, v10 // (op 2906) 


s_add_i32 s1, s0, s53 // (op 2906) 


v_lshrrev_b32 v10, 6, v1 // (op 2906) 
s_ashr_i32 s0, s1, s52 // (op 2906) 


v_and_b32 v11, 1, v10 // (op 2906) 

s_lshl_b32 s1, s13, 24 // (op 2906) 
v_lshl_add_u32 v10, v11, 5, v15 // (op 2906) 


s_ashr_i32 s52, s1, 31 // (op 2906) 

s_xor_b32 s1, s0, s52 // (op 2906) 

s_sub_i32 s0, s1, s52 // (op 2906) 


s_mul_hi_i32 s1, s0, s14 // (op 2906) 
s_add_i32 s52, s1, s0 // (op 2906) 

s_ashr_i32 s1, s52, 31 // (op 2906) 
s_cmp_eq_i32 s14, 0 // (op 2906) 
s_mov_b32 s53, scc // (op 2906) 
s_cmp_eq_i32 s53, 0 // (op 2906) 
s_cselect_b32 s55, 0, -1 // (op 2906) 

s_and_b32 s53, s15, 31 // (op 2906) 
s_lshl_b32 s56, 1, s53 // (op 2906) 
s_add_i32 s57, s56, s55 // (op 2906) 


s_and_b32 s55, s1, s57 // (op 2906) 


s_add_i32 s1, s52, s55 // (op 2906) 


s_ashr_i32 s52, s1, s53 // (op 2906) 


s_lshl_b32 s1, s15, 24 // (op 2906) 
s_ashr_i32 s53, s1, 31 // (op 2906) 

s_xor_b32 s1, s52, s53 // (op 2906) 

s_sub_i32 s52, s1, s53 // (op 2906) 


s_mul_i32 s1, s0, s11 // (op 2906) 
s_sub_u32 s53, s16, s1 // (op 2906) 

s_mul_i32 s1, s52, s11 // (op 2906) 
s_add_u32 s1, s1, s53 // (op 2906) 

s_mul_hi_i32 s53, s16, s9 // (op 2906) 
s_add_i32 s55, s53, s16 // (op 2906) 

s_ashr_i32 s53, s55, 31 // (op 2906) 
s_cmp_eq_i32 s9, 0 // (op 2906) 
s_mov_b32 s56, scc // (op 2906) 
s_cmp_eq_i32 s56, 0 // (op 2906) 
s_cselect_b32 s57, 0, -1 // (op 2906) 


s_waitcnt lgkmcnt(0)
s_and_b32 s56, s45, 31 // (op 2664) 
s_lshl_b32 s58, 1, s56 // (op 2664) 
s_add_i32 s59, s58, s54 // (op 2664) 


s_and_b32 s54, s50, s59 // (op 2664) 


s_add_i32 s50, s51, s54 // (op 2664) 


s_ashr_i32 s51, s50, s56 // (op 2664) 


s_lshl_b32 s50, s45, 24 // (op 2664) 
s_ashr_i32 s54, s50, 31 // (op 2664) 

s_xor_b32 s50, s51, s54 // (op 2664) 

s_sub_i32 s51, s50, s54 // (op 2664) 


s_and_b32 s50, s45, 31 // (op 2906) 
s_lshl_b32 s54, 1, s50 // (op 2906) 
s_add_i32 s56, s54, s57 // (op 2906) 


s_and_b32 s54, s53, s56 // (op 2906) 


s_add_i32 s53, s55, s54 // (op 2906) 


s_ashr_i32 s54, s53, s50 // (op 2906) 


s_lshl_b32 s50, s45, 24 // (op 2906) 
s_ashr_i32 s53, s50, 31 // (op 2906) 

s_xor_b32 s50, s54, s53 // (op 2906) 

s_sub_i32 s54, s50, s53 // (op 2906) 


s_add_u32 s56, s6, 256 // (op 2664) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 2664) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 2664) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 2664) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 8 // (op 2664) 

s_mul_i32 s50, s11, s56 // (op 2664) 
s_mul_i32 s53, s8, s50 // (op 2664) 

s_mul_i32 s50, s51, s53 // (op 2664) 

s_sub_u32 s53, s16, s50 // (op 2664) 

s_mov_b32 s50, s53 // (op 2664) 
s_mul_hi_i32 s55, s50, s12 // (op 2664) 
s_add_i32 s58, s55, s50 // (op 2664) 


s_ashr_i32 s50, s58, 31 // (op 2664) 
s_cmp_eq_i32 s12, 0 // (op 2664) 
s_mov_b32 s55, scc // (op 2664) 
s_cmp_eq_i32 s55, 0 // (op 2664) 
s_cselect_b32 s59, 0, -1 // (op 2664) 

s_and_b32 s55, s44, 31 // (op 2664) 
s_lshl_b32 s60, 1, s55 // (op 2664) 
s_add_i32 s61, s60, s59 // (op 2664) 


s_and_b32 s59, s50, s61 // (op 2664) 


s_add_i32 s50, s58, s59 // (op 2664) 


s_ashr_i32 s58, s50, s55 // (op 2664) 


s_lshl_b32 s50, s44, 24 // (op 2664) 
s_ashr_i32 s55, s50, 31 // (op 2664) 

s_xor_b32 s50, s58, s55 // (op 2664) 

s_sub_i32 s58, s50, s55 // (op 2664) 


s_mul_i32 s50, s46, s11 // (op 2664) 
s_add_u32 s60, s20, 64 // (op 2664) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 2664) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 2664) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 2664) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 6 // (op 2664) 

s_sub_i32 s55, s60, s50 // (op 2664) 

s_mul_i32 s59, s58, s55 // (op 2664) 

s_sub_u32 s55, s53, s59 // (op 2664) 


s_add_u32 s53, s55, s50 // (op 2664) 


s_cmp_eq_i32 s51, 0 // (op 2664) 
s_mov_b32 s50, scc // (op 2664) 

s_cmp_eq_i32 s50, 0 // (op 2664) 
s_cselect_b32 s51, s53, s19 // (op 2664) 


s_mul_i32 s19, s49, s56 // (op 2664) 


v_lshl_add_u32 v11, s51, 6, v8 // (op 2664) 


s_sub_i32 s49, s18, s19 // (op 2664) 


s_cmp_eq_i32 s50, 0 // (op 2664) 
s_cselect_b32 s51, s58, s49 // (op 2664) 



v_lshl_add_u32 v8, s51, 8, v7 // (op 2664) 


v_mul_lo_u32 v7, v8, s2 // (op 2664) 

v_add_lshl_u32 v8, v11, v7, 1 // (op 2664) 


s_add_u32 s18, s6, 256 // (op 2906) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 2906) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 2906) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 2906) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 8 // (op 2906) 

s_mul_i32 s53, s11, s18 // (op 2906) 
s_mul_i32 s49, s8, s53 // (op 2906) 

s_mul_i32 s53, s54, s49 // (op 2906) 

s_sub_u32 s49, s16, s53 // (op 2906) 

s_mov_b32 s53, s49 // (op 2906) 
s_mul_hi_i32 s50, s53, s12 // (op 2906) 
s_add_i32 s51, s50, s53 // (op 2906) 


s_ashr_i32 s50, s51, 31 // (op 2906) 
s_cmp_eq_i32 s12, 0 // (op 2906) 
s_mov_b32 s53, scc // (op 2906) 
s_cmp_eq_i32 s53, 0 // (op 2906) 
s_cselect_b32 s55, 0, -1 // (op 2906) 

s_and_b32 s53, s44, 31 // (op 2906) 
s_lshl_b32 s56, 1, s53 // (op 2906) 
s_add_i32 s57, s56, s55 // (op 2906) 


s_and_b32 s55, s50, s57 // (op 2906) 


s_add_i32 s50, s51, s55 // (op 2906) 


s_ashr_i32 s51, s50, s53 // (op 2906) 


s_lshl_b32 s50, s44, 24 // (op 2906) 
s_ashr_i32 s53, s50, 31 // (op 2906) 

s_xor_b32 s50, s51, s53 // (op 2906) 

s_sub_i32 s51, s50, s53 // (op 2906) 


s_mul_i32 s50, s46, s11 // (op 2906) 
s_add_u32 s56, s20, 64 // (op 2906) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 2906) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 2906) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 2906) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 2906) 

s_sub_i32 s53, s56, s50 // (op 2906) 

s_mul_i32 s55, s51, s53 // (op 2906) 

s_sub_u32 s53, s49, s55 // (op 2906) 


s_add_u32 s49, s53, s50 // (op 2906) 


s_cmp_eq_i32 s54, 0 // (op 2906) 
s_mov_b32 s50, scc // (op 2906) 

s_cmp_eq_i32 s50, 0 // (op 2906) 
s_cselect_b32 s53, s49, s1 // (op 2906) 


s_mul_i32 s1, s52, s18 // (op 2906) 


v_lshl_add_u32 v7, s53, 6, v14 // (op 2906) 


s_sub_i32 s49, s0, s1 // (op 2906) 


s_cmp_eq_i32 s50, 0 // (op 2906) 
s_cselect_b32 s0, s51, s49 // (op 2906) 



v_lshl_add_u32 v11, s0, 8, v10 // (op 2906) 


v_mul_lo_u32 v10, v11, s4 // (op 2906) 

v_add_lshl_u32 v11, v7, v10, 1 // (op 2906) 


s_mul_i32 s1, s2, s7 // (op 2666) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 2666) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 2666) least significant
s_mul_hi_u32 s19, s2, s6 // (op 2666) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2666) most significant: sum
s_add_u32 s1, s1, s19 // (op 2666) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 2666) 

s_mul_i32 s1, s4, s7 // (op 2908) most significant: low of low * high
s_mul_i32 s49, s5, s6 // (op 2908) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 2908) least significant
s_mul_hi_u32 s50, s4, s6 // (op 2908) most significant: high of low * low
s_add_u32 s1, s1, s49 // (op 2908) most significant: sum
s_add_u32 s1, s1, s50 // (op 2908) most significant: sum


s_lshl_b64 s[50:51], s[0:1], 1 // (op 2908) 

s_mov_b32 s52, s22 // (op 2666) reg expression
s_mov_b32 s53, s23 // (op 2666) reg expression
s_mov_b32 s54, s18 // (op 2666) reg expression
s_mov_b32 s55, 131072 // (op 2666) call()

s_mov_b32 s56, s24 // (op 2908) reg expression

s_mov_b32 s57, s25 // (op 2908) reg expression
s_mov_b32 s58, s50 // (op 2908) reg expression
s_mov_b32 s59, 131072 // (op 2908) call()


v_mov_b32 v7, s47 // (op 2) Move value
v_mov_b32 v10, s48 // (op 43) Move value


v_accvgpr_write a0, 0.00000 // (op 65) call()
v_accvgpr_write a1, 0.00000 // (op 65) call()
v_accvgpr_write a2, 0.00000 // (op 65) call()
v_accvgpr_write a3, 0.00000 // (op 65) call()
v_accvgpr_write a4, 0.00000 // (op 65) call()
v_accvgpr_write a5, 0.00000 // (op 65) call()
v_accvgpr_write a6, 0.00000 // (op 65) call()
v_accvgpr_write a7, 0.00000 // (op 65) call()
v_accvgpr_write a8, 0.00000 // (op 65) call()
v_accvgpr_write a9, 0.00000 // (op 65) call()
v_accvgpr_write a10, 0.00000 // (op 65) call()
v_accvgpr_write a11, 0.00000 // (op 65) call()
v_accvgpr_write a12, 0.00000 // (op 65) call()
v_accvgpr_write a13, 0.00000 // (op 65) call()
v_accvgpr_write a14, 0.00000 // (op 65) call()
v_accvgpr_write a15, 0.00000 // (op 65) call()
v_accvgpr_write a16, 0.00000 // (op 402) call()
v_accvgpr_write a17, 0.00000 // (op 402) call()
v_accvgpr_write a18, 0.00000 // (op 402) call()
v_accvgpr_write a19, 0.00000 // (op 402) call()
v_accvgpr_write a20, 0.00000 // (op 402) call()
v_accvgpr_write a21, 0.00000 // (op 402) call()
v_accvgpr_write a22, 0.00000 // (op 402) call()
v_accvgpr_write a23, 0.00000 // (op 402) call()
v_accvgpr_write a24, 0.00000 // (op 402) call()
v_accvgpr_write a25, 0.00000 // (op 402) call()
v_accvgpr_write a26, 0.00000 // (op 402) call()
v_accvgpr_write a27, 0.00000 // (op 402) call()
v_accvgpr_write a28, 0.00000 // (op 402) call()
v_accvgpr_write a29, 0.00000 // (op 402) call()
v_accvgpr_write a30, 0.00000 // (op 402) call()
v_accvgpr_write a31, 0.00000 // (op 402) call()
v_accvgpr_write a32, 0.00000 // (op 636) call()
v_accvgpr_write a33, 0.00000 // (op 636) call()
v_accvgpr_write a34, 0.00000 // (op 636) call()
v_accvgpr_write a35, 0.00000 // (op 636) call()
v_accvgpr_write a36, 0.00000 // (op 636) call()
v_accvgpr_write a37, 0.00000 // (op 636) call()
v_accvgpr_write a38, 0.00000 // (op 636) call()
v_accvgpr_write a39, 0.00000 // (op 636) call()
v_accvgpr_write a40, 0.00000 // (op 636) call()
v_accvgpr_write a41, 0.00000 // (op 636) call()
v_accvgpr_write a42, 0.00000 // (op 636) call()
v_accvgpr_write a43, 0.00000 // (op 636) call()
v_accvgpr_write a44, 0.00000 // (op 636) call()
v_accvgpr_write a45, 0.00000 // (op 636) call()
v_accvgpr_write a46, 0.00000 // (op 636) call()
v_accvgpr_write a47, 0.00000 // (op 636) call()
v_accvgpr_write a48, 0.00000 // (op 870) call()
v_accvgpr_write a49, 0.00000 // (op 870) call()
v_accvgpr_write a50, 0.00000 // (op 870) call()
v_accvgpr_write a51, 0.00000 // (op 870) call()
v_accvgpr_write a52, 0.00000 // (op 870) call()
v_accvgpr_write a53, 0.00000 // (op 870) call()
v_accvgpr_write a54, 0.00000 // (op 870) call()
v_accvgpr_write a55, 0.00000 // (op 870) call()
v_accvgpr_write a56, 0.00000 // (op 870) call()
v_accvgpr_write a57, 0.00000 // (op 870) call()
v_accvgpr_write a58, 0.00000 // (op 870) call()
v_accvgpr_write a59, 0.00000 // (op 870) call()
v_accvgpr_write a60, 0.00000 // (op 870) call()
v_accvgpr_write a61, 0.00000 // (op 870) call()
v_accvgpr_write a62, 0.00000 // (op 870) call()
v_accvgpr_write a63, 0.00000 // (op 870) call()
s_mul_hi_i32 s0, s16, s10 // (op 2431) 
v_lshrrev_b32 v14, 2, v1 // (op 2431) 
s_add_i32 s1, s0, s16 // (op 2431) 

v_and_b32 v15, -4, v1 // (op 2431) 
s_ashr_i32 s0, s1, 31 // (op 2431) 
v_sub_u32 v16, v1, v15 // (op 2431) 

s_cmp_eq_i32 s10, 0 // (op 2431) 
s_mov_b32 s18, scc // (op 2431) 
v_lshlrev_b32 v15, 5, v16 // (op 2431) 

s_cmp_eq_i32 s18, 0 // (op 2431) 
s_cselect_b32 s19, 0, -1 // (op 2431) 

v_mov_b32 v17, 0 // (op 2431) convert
v_mov_b32 v16, v15 // (op 2431) convert

s_and_b32 s18, s13, 31 // (op 2431) 
s_lshl_b32 s22, 1, s18 // (op 2431) 
s_add_i32 s23, s22, s19 // (op 2431) 


s_and_b32 s19, s0, s23 // (op 2431) 


s_add_i32 s0, s1, s19 // (op 2431) 


s_ashr_i32 s1, s0, s18 // (op 2431) 


s_lshl_b32 s0, s13, 24 // (op 2431) 
s_ashr_i32 s18, s0, 31 // (op 2431) 

s_xor_b32 s0, s1, s18 // (op 2431) 

s_sub_i32 s1, s0, s18 // (op 2431) 


s_mul_hi_i32 s0, s1, s14 // (op 2431) 
s_add_i32 s18, s0, s1 // (op 2431) 

s_ashr_i32 s0, s18, 31 // (op 2431) 
s_cmp_eq_i32 s14, 0 // (op 2431) 
s_mov_b32 s19, scc // (op 2431) 
s_cmp_eq_i32 s19, 0 // (op 2431) 
s_cselect_b32 s22, 0, -1 // (op 2431) 

s_and_b32 s19, s15, 31 // (op 2431) 
s_lshl_b32 s23, 1, s19 // (op 2431) 
s_add_i32 s24, s23, s22 // (op 2431) 


s_and_b32 s25, s0, s24 // (op 2431) 


s_add_i32 s0, s18, s25 // (op 2431) 


s_ashr_i32 s18, s0, s19 // (op 2431) 


s_lshl_b32 s0, s15, 24 // (op 2431) 
s_ashr_i32 s19, s0, 31 // (op 2431) 

s_xor_b32 s0, s18, s19 // (op 2431) 

s_sub_i32 s18, s0, s19 // (op 2431) 


s_mul_i32 s0, s1, s11 // (op 2431) 

s_sub_u32 s1, s16, s0 // (op 2431) 

s_mul_i32 s0, s18, s11 // (op 2431) 
s_add_u32 s0, s0, s1 // (op 2431) 


s_mul_hi_i32 s1, s16, s9 // (op 2431) 
s_add_i32 s18, s1, s16 // (op 2431) 

s_ashr_i32 s1, s18, 31 // (op 2431) 
s_cmp_eq_i32 s9, 0 // (op 2431) 
s_mov_b32 s19, scc // (op 2431) 
s_cmp_eq_i32 s19, 0 // (op 2431) 
s_cselect_b32 s22, 0, -1 // (op 2431) 

s_and_b32 s19, s45, 31 // (op 2431) 
s_lshl_b32 s23, 1, s19 // (op 2431) 
s_add_i32 s24, s23, s22 // (op 2431) 


s_and_b32 s25, s1, s24 // (op 2431) 


s_add_i32 s1, s18, s25 // (op 2431) 


s_ashr_i32 s18, s1, s19 // (op 2431) 


s_lshl_b32 s1, s45, 24 // (op 2431) 
s_ashr_i32 s19, s1, 31 // (op 2431) 

s_xor_b32 s1, s18, s19 // (op 2431) 

s_sub_i32 s18, s1, s19 // (op 2431) 


s_add_u32 s22, s6, 256 // (op 2431) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2431) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2431) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2431) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 2431) 

s_mul_i32 s1, s11, s22 // (op 2431) 

s_mul_i32 s19, s8, s1 // (op 2431) 

s_mul_i32 s1, s18, s19 // (op 2431) 

s_sub_u32 s19, s16, s1 // (op 2431) 

s_mov_b32 s1, s19 // (op 2431) 
s_mul_hi_i32 s22, s1, s12 // (op 2431) 
s_add_i32 s23, s22, s1 // (op 2431) 


s_ashr_i32 s1, s23, 31 // (op 2431) 
s_cmp_eq_i32 s12, 0 // (op 2431) 
s_mov_b32 s22, scc // (op 2431) 
s_cmp_eq_i32 s22, 0 // (op 2431) 
s_cselect_b32 s24, 0, -1 // (op 2431) 

s_and_b32 s22, s44, 31 // (op 2431) 
s_lshl_b32 s25, 1, s22 // (op 2431) 
s_add_i32 s47, s25, s24 // (op 2431) 


s_and_b32 s24, s1, s47 // (op 2431) 


s_add_i32 s1, s23, s24 // (op 2431) 


s_ashr_i32 s23, s1, s22 // (op 2431) 


s_lshl_b32 s1, s44, 24 // (op 2431) 
s_ashr_i32 s22, s1, 31 // (op 2431) 

s_xor_b32 s1, s23, s22 // (op 2431) 

s_sub_i32 s23, s1, s22 // (op 2431) 


s_mul_i32 s1, s46, s11 // (op 2431) 
s_add_u32 s24, s20, 64 // (op 2431) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2431) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 2431) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 2431) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 2431) 

s_sub_i32 s22, s24, s1 // (op 2431) 

s_mul_i32 s24, s23, s22 // (op 2431) 


s_sub_u32 s25, s19, s24 // (op 2431) 


s_add_u32 s19, s25, s1 // (op 2431) 


s_cmp_eq_i32 s18, 0 // (op 2431) 
s_mov_b32 s1, scc // (op 2431) 

s_cmp_eq_i32 s1, 0 // (op 2431) 
s_cselect_b32 s18, s19, s0 // (op 2431) 



v_lshl_add_u32 v15, s18, 6, v14 // (op 2431) 


v_mul_lo_u32 v19, v15, s27 // (op 2431) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 2431) most significant: high of low * low
v_mul_lo_u32 v18, v15, s26 // (op 2431) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 2431) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 2431) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 2431) most significant half



v_mul_lo_u32 v15, v19, 6 // (op 2431) most significant: low of high * low
v_mul_hi_u32 v16, v18, 6 // (op 2431) most significant: high of low * low
v_mul_lo_u32 v14, v18, 6 // (op 2431) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 2431) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 2431) 

v_mov_b32 v14, v16 // (op 2431) convert

s_mul_i32 s1, s21, 6 // (op 2433) most significant: low of high * low
s_mul_i32 s0, s20, 6 // (op 2433) least significant
s_mul_hi_u32 s18, s20, 6 // (op 2433) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2433) most significant: sum

s_mul_i32 s19, s26, s1 // (op 2433) most significant: low of low * high
s_mul_i32 s22, s27, s0 // (op 2433) most significant: low of high * low
s_mul_i32 s18, s26, s0 // (op 2433) least significant
s_mul_hi_u32 s23, s26, s0 // (op 2433) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 2433) most significant: sum
s_add_u32 s19, s19, s23 // (op 2433) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 2433) 

s_mov_b32 s48, s28 // (op 2433) reg expression
s_mov_b32 s49, s29 // (op 2433) reg expression
s_mov_b32 s50, s0 // (op 2433) reg expression
s_mov_b32 s51, 131072 // (op 2433) call()

s_mul_hi_i32 s47, s16, s10 // (op 2450) 
v_lshrrev_b32 v15, 2, v1 // (op 2450) 
s_add_i32 s0, s47, s16 // (op 2450) 

v_and_b32 v16, -4, v1 // (op 2450) 
s_ashr_i32 s1, s0, 31 // (op 2450) 
v_sub_u32 v17, v1, v16 // (op 2450) 

s_cmp_eq_i32 s10, 0 // (op 2450) 
s_mov_b32 s47, scc // (op 2450) 
v_lshlrev_b32 v16, 5, v17 // (op 2450) 

s_cmp_eq_i32 s47, 0 // (op 2450) 
s_cselect_b32 s18, 0, -1 // (op 2450) 

v_mov_b32 v19, 0 // (op 2450) convert
v_mov_b32 v18, v16 // (op 2450) convert

s_and_b32 s19, s13, 31 // (op 2450) 
s_lshl_b32 s47, 1, s19 // (op 2450) 
s_add_i32 s22, s47, s18 // (op 2450) 


s_and_b32 s18, s1, s22 // (op 2450) 


s_add_i32 s1, s0, s18 // (op 2450) 


s_ashr_i32 s0, s1, s19 // (op 2450) 


s_lshl_b32 s1, s13, 24 // (op 2450) 
s_ashr_i32 s47, s1, 31 // (op 2450) 

s_xor_b32 s1, s0, s47 // (op 2450) 

s_sub_i32 s0, s1, s47 // (op 2450) 


s_mul_hi_i32 s1, s0, s14 // (op 2450) 
s_add_i32 s47, s1, s0 // (op 2450) 

s_ashr_i32 s1, s47, 31 // (op 2450) 
s_cmp_eq_i32 s14, 0 // (op 2450) 
s_mov_b32 s18, scc // (op 2450) 
s_cmp_eq_i32 s18, 0 // (op 2450) 
s_cselect_b32 s19, 0, -1 // (op 2450) 

s_and_b32 s18, s15, 31 // (op 2450) 
s_lshl_b32 s22, 1, s18 // (op 2450) 
s_add_i32 s23, s22, s19 // (op 2450) 


s_and_b32 s19, s1, s23 // (op 2450) 


s_add_i32 s1, s47, s19 // (op 2450) 


s_ashr_i32 s19, s1, s18 // (op 2450) 


s_lshl_b32 s1, s15, 24 // (op 2450) 
s_ashr_i32 s18, s1, 31 // (op 2450) 

s_xor_b32 s1, s19, s18 // (op 2450) 

s_sub_i32 s19, s1, s18 // (op 2450) 


s_mul_i32 s1, s0, s11 // (op 2450) 

s_sub_u32 s0, s16, s1 // (op 2450) 

s_mul_i32 s1, s19, s11 // (op 2450) 
s_add_u32 s1, s1, s0 // (op 2450) 


s_mul_hi_i32 s0, s16, s9 // (op 2450) 
s_add_i32 s47, s0, s16 // (op 2450) 

s_ashr_i32 s0, s47, 31 // (op 2450) 
s_cmp_eq_i32 s9, 0 // (op 2450) 
s_mov_b32 s18, scc // (op 2450) 
s_cmp_eq_i32 s18, 0 // (op 2450) 
s_cselect_b32 s19, 0, -1 // (op 2450) 

s_and_b32 s18, s45, 31 // (op 2450) 
s_lshl_b32 s22, 1, s18 // (op 2450) 
s_add_i32 s23, s22, s19 // (op 2450) 


s_and_b32 s19, s0, s23 // (op 2450) 


s_add_i32 s0, s47, s19 // (op 2450) 


s_ashr_i32 s19, s0, s18 // (op 2450) 


s_lshl_b32 s0, s45, 24 // (op 2450) 
s_ashr_i32 s18, s0, 31 // (op 2450) 

s_xor_b32 s0, s19, s18 // (op 2450) 

s_sub_i32 s19, s0, s18 // (op 2450) 


s_add_u32 s22, s6, 256 // (op 2450) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2450) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2450) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2450) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 2450) 

s_mul_i32 s0, s11, s22 // (op 2450) 

s_mul_i32 s18, s8, s0 // (op 2450) 

s_mul_i32 s0, s19, s18 // (op 2450) 

s_sub_u32 s18, s16, s0 // (op 2450) 

s_mov_b32 s0, s18 // (op 2450) 
s_mul_hi_i32 s47, s0, s12 // (op 2450) 
s_add_i32 s22, s47, s0 // (op 2450) 


s_ashr_i32 s0, s22, 31 // (op 2450) 
s_cmp_eq_i32 s12, 0 // (op 2450) 
s_mov_b32 s47, scc // (op 2450) 
s_cmp_eq_i32 s47, 0 // (op 2450) 
s_cselect_b32 s23, 0, -1 // (op 2450) 

s_and_b32 s47, s44, 31 // (op 2450) 
s_lshl_b32 s24, 1, s47 // (op 2450) 
s_add_i32 s25, s24, s23 // (op 2450) 


s_and_b32 s23, s0, s25 // (op 2450) 


s_add_i32 s0, s22, s23 // (op 2450) 


s_ashr_i32 s22, s0, s47 // (op 2450) 


s_lshl_b32 s0, s44, 24 // (op 2450) 
s_ashr_i32 s47, s0, 31 // (op 2450) 

s_xor_b32 s0, s22, s47 // (op 2450) 

s_sub_i32 s22, s0, s47 // (op 2450) 


s_mul_i32 s0, s46, s11 // (op 2450) 
s_add_u32 s24, s20, 64 // (op 2450) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2450) End of Int64 add, unlocking SCC
s_sub_u32 s60, s24, 1 // (op 2450) least significant half; sets scc
s_subb_u32 s61, s25, 0 // (op 2450) most significant half; uses scc

s_ashr_i64 s[24:25], s[60:61], 6 // (op 2450) 

s_sub_i32 s23, s24, s0 // (op 2450) 

s_mul_i32 s47, s22, s23 // (op 2450) 


s_sub_u32 s22, s18, s47 // (op 2450) 


s_add_u32 s18, s22, s0 // (op 2450) 


s_cmp_eq_i32 s19, 0 // (op 2450) 
s_mov_b32 s0, scc // (op 2450) 

s_cmp_eq_i32 s0, 0 // (op 2450) 
s_cselect_b32 s19, s18, s1 // (op 2450) 



v_lshl_add_u32 v16, s19, 6, v15 // (op 2450) 


v_mul_lo_u32 v21, v16, s27 // (op 2450) most significant: low of low * high
v_mul_hi_u32 v15, v16, s26 // (op 2450) most significant: high of low * low
v_mul_lo_u32 v20, v16, s26 // (op 2450) least significant: low of low * low
v_add_u32 v21, v21, v15 // (op 2450) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v18 // (op 2450) least significant half
v_addc_co_u32 v21, s[0:1], v21, v19, s[0:1] // (op 2450) most significant half



v_mul_lo_u32 v19, v21, 6 // (op 2450) most significant: low of high * low
v_mul_hi_u32 v15, v20, 6 // (op 2450) most significant: high of low * low
v_mul_lo_u32 v18, v20, 6 // (op 2450) least significant: low of low * low
v_add_u32 v19, v19, v15 // (op 2450) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 2450) 

v_mov_b32 v15, v16 // (op 2450) convert

s_mul_i32 s1, s21, 6 // (op 2452) most significant: low of high * low
s_mul_i32 s0, s20, 6 // (op 2452) least significant
s_mul_hi_u32 s47, s20, 6 // (op 2452) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2452) most significant: sum

s_mul_i32 s19, s26, s1 // (op 2452) most significant: low of low * high
s_mul_i32 s47, s27, s0 // (op 2452) most significant: low of high * low
s_mul_i32 s18, s26, s0 // (op 2452) least significant
s_mul_hi_u32 s22, s26, s0 // (op 2452) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 2452) most significant: sum
s_add_u32 s19, s19, s22 // (op 2452) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 2452) 

s_mov_b32 s48, s28 // (op 2452) reg expression
s_mov_b32 s49, s29 // (op 2452) reg expression
s_mov_b32 s50, s0 // (op 2452) reg expression
s_mov_b32 s51, 131072 // (op 2452) call()

s_mul_hi_i32 s47, s16, s10 // (op 2495) 
v_and_b32 v16, 4294967294, v1 // (op 2495) 
s_add_i32 s0, s47, s16 // (op 2495) 

v_and_b32 v17, -2, v1 // (op 2495) 
s_ashr_i32 s1, s0, 31 // (op 2495) 
v_sub_u32 v18, v1, v17 // (op 2495) 

s_cmp_eq_i32 s10, 0 // (op 2495) 
s_mov_b32 s47, scc // (op 2495) 
v_lshlrev_b32 v17, 6, v18 // (op 2495) 

s_cmp_eq_i32 s47, 0 // (op 2495) 
s_cselect_b32 s18, 0, -1 // (op 2495) 

v_mov_b32 v19, 0 // (op 2495) convert
v_mov_b32 v18, v17 // (op 2495) convert

s_and_b32 s19, s13, 31 // (op 2495) 
s_lshl_b32 s47, 1, s19 // (op 2495) 
s_add_i32 s22, s47, s18 // (op 2495) 


s_and_b32 s18, s1, s22 // (op 2495) 


s_add_i32 s1, s0, s18 // (op 2495) 


s_ashr_i32 s0, s1, s19 // (op 2495) 


s_lshl_b32 s1, s13, 24 // (op 2495) 
s_ashr_i32 s47, s1, 31 // (op 2495) 

s_xor_b32 s1, s0, s47 // (op 2495) 

s_sub_i32 s0, s1, s47 // (op 2495) 


s_mul_hi_i32 s1, s0, s14 // (op 2495) 
s_add_i32 s47, s1, s0 // (op 2495) 

s_ashr_i32 s1, s47, 31 // (op 2495) 
s_cmp_eq_i32 s14, 0 // (op 2495) 
s_mov_b32 s18, scc // (op 2495) 
s_cmp_eq_i32 s18, 0 // (op 2495) 
s_cselect_b32 s19, 0, -1 // (op 2495) 

s_and_b32 s18, s15, 31 // (op 2495) 
s_lshl_b32 s22, 1, s18 // (op 2495) 
s_add_i32 s23, s22, s19 // (op 2495) 


s_and_b32 s19, s1, s23 // (op 2495) 


s_add_i32 s1, s47, s19 // (op 2495) 


s_ashr_i32 s19, s1, s18 // (op 2495) 


s_lshl_b32 s1, s15, 24 // (op 2495) 
s_ashr_i32 s18, s1, 31 // (op 2495) 

s_xor_b32 s1, s19, s18 // (op 2495) 

s_sub_i32 s19, s1, s18 // (op 2495) 


s_add_u32 s22, s6, 256 // (op 2495) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2495) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2495) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2495) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 2495) 

s_mul_i32 s1, s19, s22 // (op 2495) 

s_sub_i32 s47, s0, s1 // (op 2495) 


s_mul_hi_i32 s0, s16, s9 // (op 2495) 
s_add_i32 s1, s0, s16 // (op 2495) 

s_ashr_i32 s0, s1, 31 // (op 2495) 
s_cmp_eq_i32 s9, 0 // (op 2495) 
s_mov_b32 s18, scc // (op 2495) 
s_cmp_eq_i32 s18, 0 // (op 2495) 
s_cselect_b32 s19, 0, -1 // (op 2495) 

s_and_b32 s18, s45, 31 // (op 2495) 
s_lshl_b32 s24, 1, s18 // (op 2495) 
s_add_i32 s25, s24, s19 // (op 2495) 


s_and_b32 s19, s0, s25 // (op 2495) 


s_add_i32 s0, s1, s19 // (op 2495) 


s_ashr_i32 s1, s0, s18 // (op 2495) 


s_lshl_b32 s0, s45, 24 // (op 2495) 
s_ashr_i32 s18, s0, 31 // (op 2495) 

s_xor_b32 s0, s1, s18 // (op 2495) 

s_sub_i32 s1, s0, s18 // (op 2495) 


s_mul_i32 s0, s11, s22 // (op 2495) 

s_mul_i32 s18, s8, s0 // (op 2495) 

s_mul_i32 s0, s1, s18 // (op 2495) 

s_sub_u32 s18, s16, s0 // (op 2495) 

s_mul_hi_i32 s0, s18, s12 // (op 2495) 
s_add_i32 s19, s0, s18 // (op 2495) 


s_ashr_i32 s0, s19, 31 // (op 2495) 
s_cmp_eq_i32 s12, 0 // (op 2495) 
s_mov_b32 s18, scc // (op 2495) 
s_cmp_eq_i32 s18, 0 // (op 2495) 
s_cselect_b32 s22, 0, -1 // (op 2495) 

s_and_b32 s18, s44, 31 // (op 2495) 
s_lshl_b32 s23, 1, s18 // (op 2495) 
s_add_i32 s24, s23, s22 // (op 2495) 


s_and_b32 s25, s0, s24 // (op 2495) 


s_add_i32 s0, s19, s25 // (op 2495) 


s_ashr_i32 s19, s0, s18 // (op 2495) 


s_lshl_b32 s0, s44, 24 // (op 2495) 
s_ashr_i32 s18, s0, 31 // (op 2495) 

s_xor_b32 s0, s19, s18 // (op 2495) 

s_sub_i32 s19, s0, s18 // (op 2495) 


s_cmp_eq_i32 s1, 0 // (op 2495) 
s_mov_b32 s0, scc // (op 2495) 

s_cmp_eq_i32 s0, 0 // (op 2495) 
s_cselect_b32 s1, s19, s47 // (op 2495) 



v_lshl_add_u32 v17, s1, 8, v16 // (op 2495) 


v_mul_lo_u32 v21, v17, s31 // (op 2495) most significant: low of low * high
v_mul_hi_u32 v16, v17, s30 // (op 2495) most significant: high of low * low
v_mul_lo_u32 v20, v17, s30 // (op 2495) least significant: low of low * low
v_add_u32 v21, v21, v16 // (op 2495) most significant: sum


v_add_co_u32 v16, s[0:1], v18, v20 // (op 2495) least significant half
v_addc_co_u32 v17, s[0:1], v19, v21, s[0:1] // (op 2495) most significant half



v_mul_lo_u32 v19, v17, 6 // (op 2495) most significant: low of high * low
v_mul_hi_u32 v20, v16, 6 // (op 2495) most significant: high of low * low
v_mul_lo_u32 v18, v16, 6 // (op 2495) least significant: low of low * low
v_add_u32 v19, v19, v20 // (op 2495) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 2495) 

v_mov_b32 v18, v16 // (op 2495) convert

s_mul_i32 s1, s7, 6 // (op 2497) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 2497) least significant
s_mul_hi_u32 s47, s6, 6 // (op 2497) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2497) most significant: sum

s_mul_i32 s19, s30, s1 // (op 2497) most significant: low of low * high
s_mul_i32 s47, s31, s0 // (op 2497) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 2497) least significant
s_mul_hi_u32 s22, s30, s0 // (op 2497) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 2497) most significant: sum
s_add_u32 s19, s19, s22 // (op 2497) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 2497) 

s_mov_b32 s60, s32 // (op 2497) reg expression
s_mov_b32 s61, s33 // (op 2497) reg expression
s_mov_b32 s62, s0 // (op 2497) reg expression
s_mov_b32 s63, 131072 // (op 2497) call()

s_mul_hi_i32 s47, s16, s10 // (op 2514) 
v_and_b32 v16, 4294967294, v1 // (op 2514) 
s_add_i32 s0, s47, s16 // (op 2514) 

v_and_b32 v17, -2, v1 // (op 2514) 
s_ashr_i32 s1, s0, 31 // (op 2514) 
v_sub_u32 v19, v1, v17 // (op 2514) 

s_cmp_eq_i32 s10, 0 // (op 2514) 
s_mov_b32 s47, scc // (op 2514) 
v_lshlrev_b32 v17, 6, v19 // (op 2514) 

s_cmp_eq_i32 s47, 0 // (op 2514) 
s_cselect_b32 s18, 0, -1 // (op 2514) 

v_mov_b32 v21, 0 // (op 2514) convert
v_mov_b32 v20, v17 // (op 2514) convert

s_and_b32 s19, s13, 31 // (op 2514) 
s_lshl_b32 s47, 1, s19 // (op 2514) 
s_add_i32 s22, s47, s18 // (op 2514) 


s_and_b32 s18, s1, s22 // (op 2514) 


s_add_i32 s1, s0, s18 // (op 2514) 


s_ashr_i32 s0, s1, s19 // (op 2514) 


s_lshl_b32 s1, s13, 24 // (op 2514) 
s_ashr_i32 s47, s1, 31 // (op 2514) 

s_xor_b32 s1, s0, s47 // (op 2514) 

s_sub_i32 s0, s1, s47 // (op 2514) 


s_mul_hi_i32 s1, s0, s14 // (op 2514) 
s_add_i32 s47, s1, s0 // (op 2514) 

s_ashr_i32 s1, s47, 31 // (op 2514) 
s_cmp_eq_i32 s14, 0 // (op 2514) 
s_mov_b32 s18, scc // (op 2514) 
s_cmp_eq_i32 s18, 0 // (op 2514) 
s_cselect_b32 s19, 0, -1 // (op 2514) 

s_and_b32 s18, s15, 31 // (op 2514) 
s_lshl_b32 s22, 1, s18 // (op 2514) 
s_add_i32 s23, s22, s19 // (op 2514) 


s_and_b32 s19, s1, s23 // (op 2514) 


s_add_i32 s1, s47, s19 // (op 2514) 


s_ashr_i32 s19, s1, s18 // (op 2514) 


s_lshl_b32 s1, s15, 24 // (op 2514) 
s_ashr_i32 s18, s1, 31 // (op 2514) 

s_xor_b32 s1, s19, s18 // (op 2514) 

s_sub_i32 s19, s1, s18 // (op 2514) 


s_add_u32 s22, s6, 256 // (op 2514) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2514) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2514) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2514) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 2514) 

s_mul_i32 s1, s19, s22 // (op 2514) 

s_sub_i32 s47, s0, s1 // (op 2514) 


s_mul_hi_i32 s0, s16, s9 // (op 2514) 
s_add_i32 s1, s0, s16 // (op 2514) 

s_ashr_i32 s0, s1, 31 // (op 2514) 
s_cmp_eq_i32 s9, 0 // (op 2514) 
s_mov_b32 s18, scc // (op 2514) 
s_cmp_eq_i32 s18, 0 // (op 2514) 
s_cselect_b32 s19, 0, -1 // (op 2514) 

s_and_b32 s18, s45, 31 // (op 2514) 
s_lshl_b32 s24, 1, s18 // (op 2514) 
s_add_i32 s25, s24, s19 // (op 2514) 


s_and_b32 s19, s0, s25 // (op 2514) 


s_add_i32 s0, s1, s19 // (op 2514) 


s_ashr_i32 s1, s0, s18 // (op 2514) 


s_lshl_b32 s0, s45, 24 // (op 2514) 
s_ashr_i32 s18, s0, 31 // (op 2514) 

s_xor_b32 s0, s1, s18 // (op 2514) 

s_sub_i32 s1, s0, s18 // (op 2514) 


s_mul_i32 s0, s11, s22 // (op 2514) 

s_mul_i32 s18, s8, s0 // (op 2514) 

s_mul_i32 s0, s1, s18 // (op 2514) 

s_sub_u32 s18, s16, s0 // (op 2514) 

s_mul_hi_i32 s0, s18, s12 // (op 2514) 
s_add_i32 s19, s0, s18 // (op 2514) 


s_ashr_i32 s0, s19, 31 // (op 2514) 
s_cmp_eq_i32 s12, 0 // (op 2514) 
s_mov_b32 s18, scc // (op 2514) 
s_cmp_eq_i32 s18, 0 // (op 2514) 
s_cselect_b32 s22, 0, -1 // (op 2514) 

s_and_b32 s18, s44, 31 // (op 2514) 
s_lshl_b32 s23, 1, s18 // (op 2514) 
s_add_i32 s24, s23, s22 // (op 2514) 


s_and_b32 s25, s0, s24 // (op 2514) 


s_add_i32 s0, s19, s25 // (op 2514) 


s_ashr_i32 s19, s0, s18 // (op 2514) 


s_lshl_b32 s0, s44, 24 // (op 2514) 
s_ashr_i32 s18, s0, 31 // (op 2514) 

s_xor_b32 s0, s19, s18 // (op 2514) 

s_sub_i32 s19, s0, s18 // (op 2514) 


s_cmp_eq_i32 s1, 0 // (op 2514) 
s_mov_b32 s0, scc // (op 2514) 

s_cmp_eq_i32 s0, 0 // (op 2514) 
s_cselect_b32 s1, s19, s47 // (op 2514) 



v_lshl_add_u32 v17, s1, 8, v16 // (op 2514) 


v_mul_lo_u32 v23, v17, s31 // (op 2514) most significant: low of low * high
v_mul_hi_u32 v16, v17, s30 // (op 2514) most significant: high of low * low
v_mul_lo_u32 v22, v17, s30 // (op 2514) least significant: low of low * low
v_add_u32 v23, v23, v16 // (op 2514) most significant: sum


v_add_co_u32 v16, s[0:1], v20, v22 // (op 2514) least significant half
v_addc_co_u32 v17, s[0:1], v21, v23, s[0:1] // (op 2514) most significant half



v_mul_lo_u32 v21, v17, 6 // (op 2514) most significant: low of high * low
v_mul_hi_u32 v19, v16, 6 // (op 2514) most significant: high of low * low
v_mul_lo_u32 v20, v16, 6 // (op 2514) least significant: low of low * low
v_add_u32 v21, v21, v19 // (op 2514) most significant: sum


v_lshrrev_b64 v[16:17], 3, v[20:21] // (op 2514) 

v_mov_b32 v19, v16 // (op 2514) convert

s_mul_i32 s1, s7, 6 // (op 2516) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 2516) least significant
s_mul_hi_u32 s47, s6, 6 // (op 2516) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2516) most significant: sum

s_mul_i32 s19, s30, s1 // (op 2516) most significant: low of low * high
s_mul_i32 s47, s31, s0 // (op 2516) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 2516) least significant
s_mul_hi_u32 s22, s30, s0 // (op 2516) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 2516) most significant: sum
s_add_u32 s19, s19, s22 // (op 2516) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 2516) 

s_mov_b32 s60, s32 // (op 2516) reg expression
s_mov_b32 s61, s33 // (op 2516) reg expression
s_mov_b32 s62, s0 // (op 2516) reg expression
s_mov_b32 s63, 131072 // (op 2516) call()

s_mul_hi_i32 s47, s16, s10 // (op 2559) 
v_lshrrev_b32 v16, 2, v1 // (op 2559) 
s_add_i32 s0, s47, s16 // (op 2559) 

v_and_b32 v17, -4, v1 // (op 2559) 
s_ashr_i32 s1, s0, 31 // (op 2559) 
v_sub_u32 v20, v1, v17 // (op 2559) 

s_cmp_eq_i32 s10, 0 // (op 2559) 
s_mov_b32 s47, scc // (op 2559) 
s_cmp_eq_i32 s47, 0 // (op 2559) 
s_cselect_b32 s18, 0, -1 // (op 2559) 

s_and_b32 s19, s13, 31 // (op 2559) 
s_lshl_b32 s47, 1, s19 // (op 2559) 
s_add_i32 s22, s47, s18 // (op 2559) 


s_and_b32 s18, s1, s22 // (op 2559) 


s_add_i32 s1, s0, s18 // (op 2559) 


s_ashr_i32 s0, s1, s19 // (op 2559) 


s_lshl_b32 s1, s13, 24 // (op 2559) 
s_ashr_i32 s47, s1, 31 // (op 2559) 

s_xor_b32 s1, s0, s47 // (op 2559) 

s_sub_i32 s0, s1, s47 // (op 2559) 


s_mul_hi_i32 s1, s0, s14 // (op 2559) 
s_add_i32 s47, s1, s0 // (op 2559) 

s_ashr_i32 s1, s47, 31 // (op 2559) 
s_cmp_eq_i32 s14, 0 // (op 2559) 
s_mov_b32 s18, scc // (op 2559) 
s_cmp_eq_i32 s18, 0 // (op 2559) 
s_cselect_b32 s19, 0, -1 // (op 2559) 

s_and_b32 s18, s15, 31 // (op 2559) 
s_lshl_b32 s22, 1, s18 // (op 2559) 
s_add_i32 s23, s22, s19 // (op 2559) 


s_and_b32 s19, s1, s23 // (op 2559) 


s_add_i32 s1, s47, s19 // (op 2559) 


s_ashr_i32 s19, s1, s18 // (op 2559) 


s_lshl_b32 s1, s15, 24 // (op 2559) 
s_ashr_i32 s18, s1, 31 // (op 2559) 

s_xor_b32 s1, s19, s18 // (op 2559) 

s_sub_i32 s19, s1, s18 // (op 2559) 


s_mul_i32 s1, s0, s11 // (op 2559) 

s_sub_u32 s0, s16, s1 // (op 2559) 

s_mul_i32 s1, s19, s11 // (op 2559) 
s_add_u32 s1, s1, s0 // (op 2559) 


s_mul_hi_i32 s0, s16, s9 // (op 2559) 
s_add_i32 s47, s0, s16 // (op 2559) 

s_ashr_i32 s0, s47, 31 // (op 2559) 
s_cmp_eq_i32 s9, 0 // (op 2559) 
s_mov_b32 s18, scc // (op 2559) 
s_cmp_eq_i32 s18, 0 // (op 2559) 
s_cselect_b32 s19, 0, -1 // (op 2559) 

s_and_b32 s18, s45, 31 // (op 2559) 
s_lshl_b32 s22, 1, s18 // (op 2559) 
s_add_i32 s23, s22, s19 // (op 2559) 


s_and_b32 s19, s0, s23 // (op 2559) 


s_add_i32 s0, s47, s19 // (op 2559) 


s_ashr_i32 s19, s0, s18 // (op 2559) 


s_lshl_b32 s0, s45, 24 // (op 2559) 
s_ashr_i32 s18, s0, 31 // (op 2559) 

s_xor_b32 s0, s19, s18 // (op 2559) 

s_sub_i32 s19, s0, s18 // (op 2559) 


s_add_u32 s22, s6, 256 // (op 2559) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2559) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2559) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2559) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 2559) 

s_mul_i32 s0, s11, s22 // (op 2559) 

s_mul_i32 s18, s8, s0 // (op 2559) 

s_mul_i32 s0, s19, s18 // (op 2559) 

s_sub_u32 s18, s16, s0 // (op 2559) 

s_mov_b32 s0, s18 // (op 2559) 
s_mul_hi_i32 s47, s0, s12 // (op 2559) 
s_add_i32 s22, s47, s0 // (op 2559) 


s_ashr_i32 s0, s22, 31 // (op 2559) 
s_cmp_eq_i32 s12, 0 // (op 2559) 
s_mov_b32 s47, scc // (op 2559) 
s_cmp_eq_i32 s47, 0 // (op 2559) 
s_cselect_b32 s23, 0, -1 // (op 2559) 

s_and_b32 s47, s44, 31 // (op 2559) 
s_lshl_b32 s24, 1, s47 // (op 2559) 
s_add_i32 s25, s24, s23 // (op 2559) 


s_and_b32 s23, s0, s25 // (op 2559) 


s_add_i32 s0, s22, s23 // (op 2559) 


s_ashr_i32 s22, s0, s47 // (op 2559) 


s_lshl_b32 s0, s44, 24 // (op 2559) 
s_ashr_i32 s47, s0, 31 // (op 2559) 

s_xor_b32 s0, s22, s47 // (op 2559) 

s_sub_i32 s22, s0, s47 // (op 2559) 


s_mul_i32 s0, s46, s11 // (op 2559) 
s_add_u32 s24, s20, 64 // (op 2559) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2559) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 2559) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 2559) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 6 // (op 2559) 

s_sub_i32 s23, s24, s0 // (op 2559) 

s_mul_i32 s47, s22, s23 // (op 2559) 


s_sub_u32 s22, s18, s47 // (op 2559) 


s_add_u32 s18, s22, s0 // (op 2559) 


s_cmp_eq_i32 s19, 0 // (op 2559) 
s_mov_b32 s0, scc // (op 2559) 

s_cmp_eq_i32 s0, 0 // (op 2559) 
s_cselect_b32 s19, s18, s1 // (op 2559) 



v_lshl_add_u32 v17, s19, 6, v16 // (op 2559) 


v_mul_lo_u32 v16, v17, s34 // (op 2559) 
v_add_u32 v16, v16, v20 // (op 2559) 


s_mul_i32 s1, s34, s21 // (op 2561) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 2561) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 2561) least significant
s_mul_hi_u32 s18, s34, s20 // (op 2561) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2561) most significant: sum
s_add_u32 s1, s1, s18 // (op 2561) most significant: sum


s_mov_b32 s64, s36 // (op 2561) reg expression
s_mov_b32 s65, s37 // (op 2561) reg expression
s_mov_b32 s66, s0 // (op 2561) reg expression
s_mov_b32 s67, 131072 // (op 2561) call()

s_mul_hi_i32 s47, s16, s10 // (op 2578) 
v_lshrrev_b32 v17, 2, v1 // (op 2578) 
s_add_i32 s0, s47, s16 // (op 2578) 

v_and_b32 v20, -4, v1 // (op 2578) 
s_ashr_i32 s1, s0, 31 // (op 2578) 
v_sub_u32 v21, v1, v20 // (op 2578) 

s_cmp_eq_i32 s10, 0 // (op 2578) 
s_mov_b32 s47, scc // (op 2578) 
s_cmp_eq_i32 s47, 0 // (op 2578) 
s_cselect_b32 s18, 0, -1 // (op 2578) 

s_and_b32 s19, s13, 31 // (op 2578) 
s_lshl_b32 s47, 1, s19 // (op 2578) 
s_add_i32 s22, s47, s18 // (op 2578) 


s_and_b32 s18, s1, s22 // (op 2578) 


s_add_i32 s1, s0, s18 // (op 2578) 


s_ashr_i32 s0, s1, s19 // (op 2578) 


s_lshl_b32 s1, s13, 24 // (op 2578) 
s_ashr_i32 s47, s1, 31 // (op 2578) 

s_xor_b32 s1, s0, s47 // (op 2578) 

s_sub_i32 s0, s1, s47 // (op 2578) 


s_mul_hi_i32 s1, s0, s14 // (op 2578) 
s_add_i32 s47, s1, s0 // (op 2578) 

s_ashr_i32 s1, s47, 31 // (op 2578) 
s_cmp_eq_i32 s14, 0 // (op 2578) 
s_mov_b32 s18, scc // (op 2578) 
s_cmp_eq_i32 s18, 0 // (op 2578) 
s_cselect_b32 s19, 0, -1 // (op 2578) 

s_and_b32 s18, s15, 31 // (op 2578) 
s_lshl_b32 s22, 1, s18 // (op 2578) 
s_add_i32 s23, s22, s19 // (op 2578) 


s_and_b32 s19, s1, s23 // (op 2578) 


s_add_i32 s1, s47, s19 // (op 2578) 


s_ashr_i32 s19, s1, s18 // (op 2578) 


s_lshl_b32 s1, s15, 24 // (op 2578) 
s_ashr_i32 s18, s1, 31 // (op 2578) 

s_xor_b32 s1, s19, s18 // (op 2578) 

s_sub_i32 s19, s1, s18 // (op 2578) 


s_mul_i32 s1, s0, s11 // (op 2578) 

s_sub_u32 s0, s16, s1 // (op 2578) 

s_mul_i32 s1, s19, s11 // (op 2578) 
s_add_u32 s1, s1, s0 // (op 2578) 


s_mul_hi_i32 s0, s16, s9 // (op 2578) 
s_add_i32 s47, s0, s16 // (op 2578) 

s_ashr_i32 s0, s47, 31 // (op 2578) 
s_cmp_eq_i32 s9, 0 // (op 2578) 
s_mov_b32 s18, scc // (op 2578) 
s_cmp_eq_i32 s18, 0 // (op 2578) 
s_cselect_b32 s19, 0, -1 // (op 2578) 

s_and_b32 s18, s45, 31 // (op 2578) 
s_lshl_b32 s22, 1, s18 // (op 2578) 
s_add_i32 s23, s22, s19 // (op 2578) 


s_and_b32 s19, s0, s23 // (op 2578) 


s_add_i32 s0, s47, s19 // (op 2578) 


s_ashr_i32 s19, s0, s18 // (op 2578) 


s_lshl_b32 s0, s45, 24 // (op 2578) 
s_ashr_i32 s18, s0, 31 // (op 2578) 

s_xor_b32 s0, s19, s18 // (op 2578) 

s_sub_i32 s19, s0, s18 // (op 2578) 


s_add_u32 s22, s6, 256 // (op 2578) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2578) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2578) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2578) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 2578) 

s_mul_i32 s0, s11, s22 // (op 2578) 

s_mul_i32 s18, s8, s0 // (op 2578) 

s_mul_i32 s0, s19, s18 // (op 2578) 

s_sub_u32 s18, s16, s0 // (op 2578) 

s_mov_b32 s0, s18 // (op 2578) 
s_mul_hi_i32 s47, s0, s12 // (op 2578) 
s_add_i32 s22, s47, s0 // (op 2578) 


s_ashr_i32 s0, s22, 31 // (op 2578) 
s_cmp_eq_i32 s12, 0 // (op 2578) 
s_mov_b32 s47, scc // (op 2578) 
s_cmp_eq_i32 s47, 0 // (op 2578) 
s_cselect_b32 s23, 0, -1 // (op 2578) 

s_and_b32 s47, s44, 31 // (op 2578) 
s_lshl_b32 s24, 1, s47 // (op 2578) 
s_add_i32 s25, s24, s23 // (op 2578) 


s_and_b32 s23, s0, s25 // (op 2578) 


s_add_i32 s0, s22, s23 // (op 2578) 


s_ashr_i32 s22, s0, s47 // (op 2578) 


s_lshl_b32 s0, s44, 24 // (op 2578) 
s_ashr_i32 s47, s0, 31 // (op 2578) 

s_xor_b32 s0, s22, s47 // (op 2578) 

s_sub_i32 s22, s0, s47 // (op 2578) 


s_mul_i32 s0, s46, s11 // (op 2578) 
s_add_u32 s24, s20, 64 // (op 2578) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2578) End of Int64 add, unlocking SCC
s_sub_u32 s68, s24, 1 // (op 2578) least significant half; sets scc
s_subb_u32 s69, s25, 0 // (op 2578) most significant half; uses scc

s_ashr_i64 s[24:25], s[68:69], 6 // (op 2578) 

s_sub_i32 s23, s24, s0 // (op 2578) 

s_mul_i32 s47, s22, s23 // (op 2578) 


s_sub_u32 s22, s18, s47 // (op 2578) 


s_add_u32 s18, s22, s0 // (op 2578) 


s_cmp_eq_i32 s19, 0 // (op 2578) 
s_mov_b32 s0, scc // (op 2578) 

s_cmp_eq_i32 s0, 0 // (op 2578) 
s_cselect_b32 s19, s18, s1 // (op 2578) 



v_lshl_add_u32 v20, s19, 6, v17 // (op 2578) 


v_mul_lo_u32 v17, v20, s34 // (op 2578) 
v_add_u32 v17, v17, v21 // (op 2578) 


s_mul_i32 s1, s34, s21 // (op 2580) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 2580) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 2580) least significant
s_mul_hi_u32 s18, s34, s20 // (op 2580) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2580) most significant: sum
s_add_u32 s1, s1, s18 // (op 2580) most significant: sum


s_mov_b32 s64, s36 // (op 2580) reg expression
s_mov_b32 s65, s37 // (op 2580) reg expression
s_mov_b32 s66, s0 // (op 2580) reg expression
s_mov_b32 s67, 131072 // (op 2580) call()

s_mul_hi_i32 s47, s16, s10 // (op 2623) 
s_add_i32 s0, s47, s16 // (op 2623) 

s_ashr_i32 s1, s0, 31 // (op 2623) 
s_cmp_eq_i32 s10, 0 // (op 2623) 
s_mov_b32 s47, scc // (op 2623) 
s_cmp_eq_i32 s47, 0 // (op 2623) 
s_cselect_b32 s18, 0, -1 // (op 2623) 

s_and_b32 s19, s13, 31 // (op 2623) 
s_lshl_b32 s47, 1, s19 // (op 2623) 
s_add_i32 s22, s47, s18 // (op 2623) 


s_and_b32 s18, s1, s22 // (op 2623) 


s_add_i32 s1, s0, s18 // (op 2623) 


s_ashr_i32 s0, s1, s19 // (op 2623) 


s_lshl_b32 s1, s13, 24 // (op 2623) 
s_ashr_i32 s47, s1, 31 // (op 2623) 

s_xor_b32 s1, s0, s47 // (op 2623) 

s_sub_i32 s0, s1, s47 // (op 2623) 


s_mul_hi_i32 s1, s0, s14 // (op 2623) 
s_add_i32 s47, s1, s0 // (op 2623) 

s_ashr_i32 s1, s47, 31 // (op 2623) 
s_cmp_eq_i32 s14, 0 // (op 2623) 
s_mov_b32 s18, scc // (op 2623) 
s_cmp_eq_i32 s18, 0 // (op 2623) 
s_cselect_b32 s19, 0, -1 // (op 2623) 

s_and_b32 s18, s15, 31 // (op 2623) 
s_lshl_b32 s22, 1, s18 // (op 2623) 
s_add_i32 s23, s22, s19 // (op 2623) 


s_and_b32 s19, s1, s23 // (op 2623) 


s_add_i32 s1, s47, s19 // (op 2623) 


s_ashr_i32 s19, s1, s18 // (op 2623) 


s_lshl_b32 s1, s15, 24 // (op 2623) 
s_ashr_i32 s18, s1, 31 // (op 2623) 

s_xor_b32 s1, s19, s18 // (op 2623) 

s_sub_i32 s19, s1, s18 // (op 2623) 


s_add_u32 s22, s6, 256 // (op 2623) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2623) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2623) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2623) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 2623) 

s_mul_i32 s1, s19, s22 // (op 2623) 

s_sub_i32 s47, s0, s1 // (op 2623) 


s_mul_hi_i32 s0, s16, s9 // (op 2623) 
s_add_i32 s1, s0, s16 // (op 2623) 

s_ashr_i32 s0, s1, 31 // (op 2623) 
s_cmp_eq_i32 s9, 0 // (op 2623) 
s_mov_b32 s18, scc // (op 2623) 
s_cmp_eq_i32 s18, 0 // (op 2623) 
s_cselect_b32 s19, 0, -1 // (op 2623) 

s_and_b32 s18, s45, 31 // (op 2623) 
s_lshl_b32 s24, 1, s18 // (op 2623) 
s_add_i32 s25, s24, s19 // (op 2623) 


s_and_b32 s19, s0, s25 // (op 2623) 


s_add_i32 s0, s1, s19 // (op 2623) 


s_ashr_i32 s1, s0, s18 // (op 2623) 


s_lshl_b32 s0, s45, 24 // (op 2623) 
s_ashr_i32 s18, s0, 31 // (op 2623) 

s_xor_b32 s0, s1, s18 // (op 2623) 

s_sub_i32 s1, s0, s18 // (op 2623) 


s_mul_i32 s0, s11, s22 // (op 2623) 

s_mul_i32 s18, s8, s0 // (op 2623) 

s_mul_i32 s0, s1, s18 // (op 2623) 

s_sub_u32 s18, s16, s0 // (op 2623) 

s_mul_hi_i32 s0, s18, s12 // (op 2623) 
s_add_i32 s19, s0, s18 // (op 2623) 


s_ashr_i32 s0, s19, 31 // (op 2623) 
s_cmp_eq_i32 s12, 0 // (op 2623) 
s_mov_b32 s18, scc // (op 2623) 
s_cmp_eq_i32 s18, 0 // (op 2623) 
s_cselect_b32 s22, 0, -1 // (op 2623) 

s_and_b32 s18, s44, 31 // (op 2623) 
s_lshl_b32 s23, 1, s18 // (op 2623) 
s_add_i32 s24, s23, s22 // (op 2623) 


s_and_b32 s25, s0, s24 // (op 2623) 


s_add_i32 s0, s19, s25 // (op 2623) 


s_ashr_i32 s19, s0, s18 // (op 2623) 


s_lshl_b32 s0, s44, 24 // (op 2623) 
s_ashr_i32 s18, s0, 31 // (op 2623) 

s_xor_b32 s0, s19, s18 // (op 2623) 

s_sub_i32 s19, s0, s18 // (op 2623) 


s_cmp_eq_i32 s1, 0 // (op 2623) 
s_mov_b32 s0, scc // (op 2623) 

s_cmp_eq_i32 s0, 0 // (op 2623) 
s_cselect_b32 s1, s19, s47 // (op 2623) 



v_lshl_add_u32 v20, s1, 8, v1 // (op 2623) 

v_mul_lo_u32 v21, v20, s38 // (op 2623) 

s_mul_i32 s1, s38, s7 // (op 2625) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 2625) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 2625) least significant
s_mul_hi_u32 s18, s38, s6 // (op 2625) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2625) most significant: sum
s_add_u32 s1, s1, s18 // (op 2625) most significant: sum


s_mov_b32 s68, s40 // (op 2625) reg expression
s_mov_b32 s69, s41 // (op 2625) reg expression
s_mov_b32 s70, s0 // (op 2625) reg expression
s_mov_b32 s71, 131072 // (op 2625) call()

s_mul_hi_i32 s47, s16, s10 // (op 2642) 
s_add_i32 s0, s47, s16 // (op 2642) 

s_ashr_i32 s1, s0, 31 // (op 2642) 
s_cmp_eq_i32 s10, 0 // (op 2642) 
s_mov_b32 s47, scc // (op 2642) 
s_cmp_eq_i32 s47, 0 // (op 2642) 
s_cselect_b32 s18, 0, -1 // (op 2642) 

s_and_b32 s19, s13, 31 // (op 2642) 
s_lshl_b32 s47, 1, s19 // (op 2642) 
s_add_i32 s22, s47, s18 // (op 2642) 


s_and_b32 s18, s1, s22 // (op 2642) 


s_add_i32 s1, s0, s18 // (op 2642) 


s_ashr_i32 s0, s1, s19 // (op 2642) 


s_lshl_b32 s1, s13, 24 // (op 2642) 
s_ashr_i32 s47, s1, 31 // (op 2642) 

s_xor_b32 s1, s0, s47 // (op 2642) 

s_sub_i32 s0, s1, s47 // (op 2642) 


s_mul_hi_i32 s1, s0, s14 // (op 2642) 
s_add_i32 s47, s1, s0 // (op 2642) 

s_ashr_i32 s1, s47, 31 // (op 2642) 
s_cmp_eq_i32 s14, 0 // (op 2642) 
s_mov_b32 s18, scc // (op 2642) 
s_cmp_eq_i32 s18, 0 // (op 2642) 
s_cselect_b32 s19, 0, -1 // (op 2642) 

s_and_b32 s18, s15, 31 // (op 2642) 
s_lshl_b32 s22, 1, s18 // (op 2642) 
s_add_i32 s23, s22, s19 // (op 2642) 


s_and_b32 s19, s1, s23 // (op 2642) 


s_add_i32 s1, s47, s19 // (op 2642) 


s_ashr_i32 s19, s1, s18 // (op 2642) 


s_lshl_b32 s1, s15, 24 // (op 2642) 
s_ashr_i32 s18, s1, 31 // (op 2642) 

s_xor_b32 s1, s19, s18 // (op 2642) 

s_sub_i32 s19, s1, s18 // (op 2642) 


s_add_u32 s22, s6, 256 // (op 2642) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2642) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2642) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2642) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 2642) 

s_mul_i32 s1, s19, s22 // (op 2642) 

s_sub_i32 s47, s0, s1 // (op 2642) 


s_mul_hi_i32 s0, s16, s9 // (op 2642) 
s_add_i32 s1, s0, s16 // (op 2642) 

s_ashr_i32 s0, s1, 31 // (op 2642) 
s_cmp_eq_i32 s9, 0 // (op 2642) 
s_mov_b32 s18, scc // (op 2642) 
s_cmp_eq_i32 s18, 0 // (op 2642) 
s_cselect_b32 s19, 0, -1 // (op 2642) 

s_and_b32 s18, s45, 31 // (op 2642) 
s_lshl_b32 s24, 1, s18 // (op 2642) 
s_add_i32 s25, s24, s19 // (op 2642) 


s_and_b32 s19, s0, s25 // (op 2642) 


s_add_i32 s0, s1, s19 // (op 2642) 


s_ashr_i32 s1, s0, s18 // (op 2642) 


s_lshl_b32 s0, s45, 24 // (op 2642) 
s_ashr_i32 s18, s0, 31 // (op 2642) 

s_xor_b32 s0, s1, s18 // (op 2642) 

s_sub_i32 s1, s0, s18 // (op 2642) 


s_mul_i32 s0, s11, s22 // (op 2642) 

s_mul_i32 s18, s8, s0 // (op 2642) 

s_mul_i32 s0, s1, s18 // (op 2642) 

s_sub_u32 s18, s16, s0 // (op 2642) 

s_mul_hi_i32 s0, s18, s12 // (op 2642) 
s_add_i32 s19, s0, s18 // (op 2642) 


s_ashr_i32 s0, s19, 31 // (op 2642) 
s_cmp_eq_i32 s12, 0 // (op 2642) 
s_mov_b32 s18, scc // (op 2642) 
s_cmp_eq_i32 s18, 0 // (op 2642) 
s_cselect_b32 s22, 0, -1 // (op 2642) 

s_and_b32 s18, s44, 31 // (op 2642) 
s_lshl_b32 s23, 1, s18 // (op 2642) 
s_add_i32 s24, s23, s22 // (op 2642) 


s_and_b32 s25, s0, s24 // (op 2642) 


s_add_i32 s0, s19, s25 // (op 2642) 


s_ashr_i32 s19, s0, s18 // (op 2642) 


s_lshl_b32 s0, s44, 24 // (op 2642) 
s_ashr_i32 s18, s0, 31 // (op 2642) 

s_xor_b32 s0, s19, s18 // (op 2642) 

s_sub_i32 s19, s0, s18 // (op 2642) 


s_cmp_eq_i32 s1, 0 // (op 2642) 
s_mov_b32 s0, scc // (op 2642) 

s_cmp_eq_i32 s0, 0 // (op 2642) 
s_cselect_b32 s1, s19, s47 // (op 2642) 



v_lshl_add_u32 v20, s1, 8, v1 // (op 2642) 

v_mul_lo_u32 v22, v20, s38 // (op 2642) 

s_mul_i32 s1, s38, s7 // (op 2644) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 2644) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 2644) least significant
s_mul_hi_u32 s18, s38, s6 // (op 2644) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2644) most significant: sum
s_add_u32 s1, s1, s18 // (op 2644) most significant: sum


s_mov_b32 s68, s40 // (op 2644) reg expression
s_mov_b32 s69, s41 // (op 2644) reg expression
s_mov_b32 s70, s0 // (op 2644) reg expression
s_mov_b32 s71, 131072 // (op 2644) call()


s_mul_i32 s1, s31, 6 // (op 2169) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 2169) least significant
s_mul_hi_u32 s47, s30, 6 // (op 2169) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2169) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 2169) 


v_mov_b32 v20, v18 // (op 2169) 
buffer_load_dwordx4 v[26:29], v20, s[60:63], 0 offen // (op 2169) Load value
buffer_load_dwordx2 v[30:31], v20, s[60:63], 0 offen offset:16 // (op 2169) Load value
buffer_load_dwordx4 v[32:35], v20, s[60:63], 0 offen offset:24 // (op 2169) Load value
buffer_load_dwordx2 v[36:37], v20, s[60:63], 0 offen offset:40 // (op 2169) Load value
v_add_u32 v20, v20, v24 // (op 2169) 
buffer_load_dwordx4 v[38:41], v20, s[60:63], 0 offen // (op 2169) Load value
buffer_load_dwordx2 v[42:43], v20, s[60:63], 0 offen offset:16 // (op 2169) Load value
buffer_load_dwordx4 v[44:47], v20, s[60:63], 0 offen offset:24 // (op 2169) Load value
buffer_load_dwordx2 v[48:49], v20, s[60:63], 0 offen offset:40 // (op 2169) Load value





buffer_load_dword v20, v21, s[68:71], 0 offen // (op 2172) Load value



buffer_load_dwordx4 v[50:53], v14, s[48:51], 0 offen // (op 2175) Load value
buffer_load_dwordx2 v[54:55], v14, s[48:51], 0 offen offset:16 // (op 2175) Load value



buffer_load_ubyte v23, v16, s[64:67], 0 offen offset: 0 // (op 2178) Load value

s_mul_i32 s1, s31, 6 // (op 2181) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 2181) least significant
s_mul_hi_u32 s47, s30, 6 // (op 2181) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2181) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 2181) 


v_mov_b32 v56, v18 // (op 2181) 
buffer_load_dwordx4 v[58:61], v56, s[60:63], 0 offen offset:96 // (op 2181) Load value
buffer_load_dwordx2 v[62:63], v56, s[60:63], 0 offen offset:112 // (op 2181) Load value
buffer_load_dwordx4 v[64:67], v56, s[60:63], 0 offen offset:120 // (op 2181) Load value
buffer_load_dwordx2 v[68:69], v56, s[60:63], 0 offen offset:136 // (op 2181) Load value
v_add_u32 v56, v56, v24 // (op 2181) 
buffer_load_dwordx4 v[70:73], v56, s[60:63], 0 offen offset:96 // (op 2181) Load value
buffer_load_dwordx2 v[74:75], v56, s[60:63], 0 offen offset:112 // (op 2181) Load value
buffer_load_dwordx4 v[76:79], v56, s[60:63], 0 offen offset:120 // (op 2181) Load value
buffer_load_dwordx2 v[80:81], v56, s[60:63], 0 offen offset:136 // (op 2181) Load value






buffer_load_dword v18, v21, s[68:71], 0 offen offset:4 // (op 2184) Load value




buffer_load_dwordx4 v[82:85], v14, s[48:51], 0 offen offset:96 // (op 2187) Load value
buffer_load_dwordx2 v[86:87], v14, s[48:51], 0 offen offset:112 // (op 2187) Load value




buffer_load_ubyte v14, v16, s[64:67], 0 offen offset:4 // (op 2190) Load value

v_and_b32 v16, -2, v1 // (op 2762) 
v_sub_u32 v21, v1, v16 // (op 2762) 

v_lshlrev_b32 v16, 6, v21 // (op 2762) 

v_lshrrev_b32 v21, 1, v1 // (op 2762) 
v_lshl_add_u32 v24, v21, 8, v16 // (op 2762) 


v_mul_lo_u32 v16, v24, 6 // (op 2762) 

v_lshrrev_b32 v21, 3, v16 // (op 2762) 




s_waitcnt vmcnt(23)
ds_write_b128 v21, v[26:29] // (op 2192) Store local data 
s_waitcnt vmcnt(22)
ds_write_b64 v21, v[30:31] offset:16 // (op 2192) Store local data 
s_waitcnt vmcnt(21)
ds_write_b128 v21, v[32:35] offset:24 // (op 2192) Store local data 
s_waitcnt vmcnt(20)
ds_write_b64 v21, v[36:37] offset:40 // (op 2192) Store local data 
s_waitcnt vmcnt(19)
ds_write_b128 v21, v[38:41] offset:96 // (op 2192) Store local data 
s_waitcnt vmcnt(18)
ds_write_b64 v21, v[42:43] offset:112 // (op 2192) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v21, v[44:47] offset:120 // (op 2192) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v21, v[48:49] offset:136 // (op 2192) Store local data 

v_lshlrev_b32 v16, 2, v1 // (op 2873) 



s_waitcnt vmcnt(15)
ds_write_b32 v16, v20 offset:24576 // (op 2193) Store local data 

v_and_b32 v16, -4, v1 // (op 2709) 
v_sub_u32 v21, v1, v16 // (op 2709) 

v_lshlrev_b32 v16, 5, v21 // (op 2709) 

v_lshrrev_b32 v21, 2, v1 // (op 2709) 
v_lshl_add_u32 v24, v21, 7, v16 // (op 2709) 


v_mul_lo_u32 v16, v24, 6 // (op 2709) 

v_lshrrev_b32 v21, 3, v16 // (op 2709) 




s_waitcnt vmcnt(14)
ds_write_b128 v21, v[50:53] offset:25600 // (op 2194) Store local data 
s_waitcnt vmcnt(13)
ds_write_b64 v21, v[54:55] offset:25616 // (op 2194) Store local data 

v_and_b32 v16, -4, v1 // (op 2820) 
v_sub_u32 v21, v1, v16 // (op 2820) 

v_and_b32 v16, 4294967292, v1 // (op 2820) 
v_add_u32 v24, v16, v21 // (op 2820) 





s_waitcnt vmcnt(12)
ds_write_b8 v24, v23 offset:31744 // (op 2195) Store local data 

s_waitcnt lgkmcnt(4)
s_barrier  // (op 3096) 



ds_read_b128 v[88:91], v9 // (op 2213) Load local data 
ds_read_b64 v[92:93], v9 offset:16 // (op 2213) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 3137) 



ds_read_u8 v16, v6 offset:24576 // (op 2236) Load local data 



ds_read_b128 v[94:97], v9 offset:6144 // (op 2218) Load local data 
ds_read_b64 v[98:99], v9 offset:6160 // (op 2218) Load local data 



ds_read_u8 v21, v6 offset:24832 // (op 2241) Load local data 



ds_read_b128 v[100:103], v9 offset:12288 // (op 2223) Load local data 
ds_read_b64 v[104:105], v9 offset:12304 // (op 2223) Load local data 



ds_read_u8 v24, v6 offset:25088 // (op 2246) Load local data 



ds_read_b128 v[106:109], v9 offset:18432 // (op 2228) Load local data 
ds_read_b64 v[110:111], v9 offset:18448 // (op 2228) Load local data 



ds_read_u8 v25, v6 offset:25344 // (op 2251) Load local data 
s_waitcnt lgkmcnt(13)
s_barrier  // (op 3077) 



ds_read_b128 v[112:115], v3 offset:25600 // (op 2209) Load local data 
ds_read_b64 v[116:117], v3 offset:25616 // (op 2209) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 3117) 



ds_read_u8 v56, v0 offset:31744 // (op 2232) Load local data 
v_and_b32 v57, -4, v1 // (op 2680) 
v_sub_u32 v118, v1, v57 // (op 2680) 

v_lshlrev_b32 v57, 5, v118 // (op 2680) 

v_lshrrev_b32 v118, 2, v1 // (op 2680) 
v_lshl_add_u32 v119, v118, 7, v57 // (op 2680) 


v_mul_lo_u32 v57, v119, 6 // (op 2680) 

v_lshrrev_b32 v118, 3, v57 // (op 2680) 

v_and_b32 v57, -2, v1 // (op 2733) 
v_sub_u32 v119, v1, v57 // (op 2733) 

v_lshlrev_b32 v57, 6, v119 // (op 2733) 

v_lshrrev_b32 v119, 1, v1 // (op 2733) 
v_lshl_add_u32 v120, v119, 8, v57 // (op 2733) 


v_mul_lo_u32 v57, v120, 6 // (op 2733) 

v_lshrrev_b32 v119, 3, v57 // (op 2733) 

v_and_b32 v57, -4, v1 // (op 2791) 
v_sub_u32 v120, v1, v57 // (op 2791) 

v_and_b32 v57, 4294967292, v1 // (op 2791) 
v_add_u32 v121, v57, v120 // (op 2791) 


v_lshlrev_b32 v57, 2, v1 // (op 2844) 
v_and_b32 v120, -4, v1 // (op 2920) 
v_sub_u32 v122, v1, v120 // (op 2920) 

v_lshlrev_b32 v120, 5, v122 // (op 2920) 

v_lshrrev_b32 v122, 2, v1 // (op 2920) 
v_lshl_add_u32 v123, v122, 7, v120 // (op 2920) 


v_mul_lo_u32 v120, v123, 6 // (op 2920) 

v_lshrrev_b32 v122, 3, v120 // (op 2920) 

v_and_b32 v120, -2, v1 // (op 2944) 
v_sub_u32 v123, v1, v120 // (op 2944) 

v_lshlrev_b32 v120, 6, v123 // (op 2944) 

v_lshrrev_b32 v123, 1, v1 // (op 2944) 
v_lshl_add_u32 v124, v123, 8, v120 // (op 2944) 


v_mul_lo_u32 v120, v124, 6 // (op 2944) 

v_lshrrev_b32 v123, 3, v120 // (op 2944) 

v_and_b32 v120, -4, v1 // (op 2973) 
v_sub_u32 v124, v1, v120 // (op 2973) 

v_and_b32 v120, 4294967292, v1 // (op 2973) 
v_add_u32 v125, v120, v124 // (op 2973) 


v_lshlrev_b32 v120, 2, v1 // (op 2997) 
s_mov_b32 s47, 0 // (op 872) call()
s_ashr_i64 s[0:1], s[42:43], 7 // (op 871) 
s_mov_b32 s18, s0 // (op 871) convert

s_and_b32 s19, s18, 4294967294 // (op 871) 

s_cmp_lt_u32 s47, s19 // (op 871) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopBottom_KLoop_871 // (op 871) Condition: Top (jump to RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopBottom_KLoop_871 if false)
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopTop_KLoop_871:

s_add_u32 s47, s47, 2 // (op 873) 
s_mul_i32 s1, s31, 6 // (op 10) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 10) least significant
s_mul_hi_u32 s18, s30, 6 // (op 10) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 10) most significant: sum

v_lshrrev_b64 v[126:127], 3, s[0:1] // (op 10) 


v_mov_b32 v124, v19 // (op 10) 
v_mov_b32 v128, v124 // (op 10) 
buffer_load_dwordx4 v[26:29], v128, s[60:63], 0 offen offset:192 // (op 10) Load value
buffer_load_dwordx2 v[30:31], v128, s[60:63], 0 offen offset:208 // (op 10) Load value
buffer_load_dwordx4 v[32:35], v128, s[60:63], 0 offen offset:216 // (op 10) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[112:117], v[88:93], a[0:15], v56, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 90) 
buffer_load_dwordx2 v[36:37], v128, s[60:63], 0 offen offset:232 // (op 10) Load value
s_barrier  // (op 3086) 
v_add_u32 v128, v128, v126 // (op 10) 



buffer_load_dwordx4 v[38:41], v128, s[60:63], 0 offen offset:192 // (op 10) Load value
buffer_load_dwordx2 v[42:43], v128, s[60:63], 0 offen offset:208 // (op 10) Load value
ds_read_b128 v[130:133], v3 offset:25648 // (op 93) Load local data 
ds_read_b64 v[134:135], v3 offset:25664 // (op 93) Load local data 
buffer_load_dwordx4 v[44:47], v128, s[60:63], 0 offen offset:216 // (op 10) Load value
s_barrier  // (op 3106) 
buffer_load_dwordx2 v[48:49], v128, s[60:63], 0 offen offset:232 // (op 10) Load value







ds_read_b128 v[88:91], v9 offset:48 // (op 97) Load local data 
ds_read_b64 v[92:93], v9 offset:64 // (op 97) Load local data 
v_mov_b32 v126, v22 // (op 25) 

s_barrier  // (op 3127) 
buffer_load_dword v20, v126, s[68:71], 0 offen offset:8 // (op 25) Load value





ds_read_u8 v127, v0 offset:31746 // (op 101) Load local data 
v_mov_b32 v128, v15 // (op 4) 

s_barrier  // (op 3147) 
buffer_load_dwordx4 v[50:53], v128, s[48:51], 0 offen offset:192 // (op 4) Load value



ds_read_u8 v16, v6 offset:24578 // (op 105) Load local data 
buffer_load_dwordx2 v[54:55], v128, s[48:51], 0 offen offset:208 // (op 4) Load value


v_mov_b32 v129, v17 // (op 17) 

buffer_load_ubyte v23, v129, s[64:67], 0 offen offset:8 // (op 17) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[130:135], v[88:93], a[0:15], v127, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 107) 
s_barrier  // (op 3167) 



v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[112:117], v[94:99], a[16:31], v56, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 451) 
s_waitcnt vmcnt(23)
ds_write_b128 v123, v[58:61] offset:32000 // (op 306) Store local data 



ds_read_b128 v[88:91], v9 offset:6192 // (op 455) Load local data 
ds_read_b64 v[92:93], v9 offset:6208 // (op 455) Load local data 
s_waitcnt vmcnt(22)
ds_write_b64 v123, v[62:63] offset:32016 // (op 306) Store local data 



ds_read_u8 v16, v6 offset:24834 // (op 459) Load local data 
s_waitcnt vmcnt(21)
ds_write_b128 v123, v[64:67] offset:32024 // (op 306) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[130:135], v[88:93], a[16:31], v127, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 460) 
s_waitcnt vmcnt(20)
ds_write_b64 v123, v[68:69] offset:32040 // (op 306) Store local data 
s_waitcnt vmcnt(19)
ds_write_b128 v123, v[70:73] offset:32096 // (op 306) Store local data 
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[112:117], v[100:105], a[32:47], v56, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 685) 
s_waitcnt vmcnt(18)
ds_write_b64 v123, v[74:75] offset:32112 // (op 306) Store local data 



ds_read_b128 v[88:91], v9 offset:12336 // (op 689) Load local data 
ds_read_b64 v[92:93], v9 offset:12352 // (op 689) Load local data 
s_waitcnt vmcnt(17)
ds_write_b128 v123, v[76:79] offset:32120 // (op 306) Store local data 



ds_read_u8 v16, v6 offset:25090 // (op 693) Load local data 
s_waitcnt vmcnt(16)
ds_write_b64 v123, v[80:81] offset:32136 // (op 306) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[130:135], v[88:93], a[32:47], v127, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 694) 
s_barrier  // (op 3179) 



v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[112:117], v[106:111], a[48:63], v56, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 919) 
s_waitcnt vmcnt(15)
ds_write_b32 v120, v18 offset:56576 // (op 308) Store local data 



ds_read_b128 v[106:109], v9 offset:18480 // (op 923) Load local data 
s_barrier  // (op 3160) 
ds_read_b64 v[110:111], v9 offset:18496 // (op 923) Load local data 






ds_read_u8 v25, v6 offset:25346 // (op 927) Load local data 
s_waitcnt vmcnt(14)
ds_write_b128 v122, v[82:85] offset:57600 // (op 287) Store local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[130:135], v[106:111], a[48:63], v127, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 928) 
s_waitcnt vmcnt(13)
ds_write_b64 v122, v[86:87] offset:57616 // (op 287) Store local data 


s_barrier  // (op 3173) 



s_waitcnt vmcnt(12)
ds_write_b8 v125, v14 offset:63744 // (op 307) Store local data 
s_barrier  // (op 3163) 



ds_read_b128 v[88:91], v13 offset:32000 // (op 291) Load local data 
ds_read_b64 v[92:93], v13 offset:32016 // (op 291) Load local data 
s_barrier  // (op 3176) 



ds_read_u8 v16, v12 offset:56576 // (op 295) Load local data 



ds_read_b128 v[94:97], v13 offset:38144 // (op 474) Load local data 
ds_read_b64 v[98:99], v13 offset:38160 // (op 474) Load local data 



ds_read_u8 v21, v12 offset:56832 // (op 478) Load local data 



ds_read_b128 v[100:103], v13 offset:44288 // (op 708) Load local data 
ds_read_b64 v[104:105], v13 offset:44304 // (op 708) Load local data 



ds_read_u8 v24, v12 offset:57088 // (op 712) Load local data 



ds_read_b128 v[130:133], v13 offset:50432 // (op 942) Load local data 
ds_read_b64 v[134:135], v13 offset:50448 // (op 942) Load local data 



ds_read_u8 v127, v12 offset:57344 // (op 946) Load local data 
s_waitcnt lgkmcnt(13)
s_barrier  // (op 3157) 



ds_read_b128 v[112:115], v5 offset:57600 // (op 289) Load local data 
ds_read_b64 v[116:117], v5 offset:57616 // (op 289) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 3170) 



ds_read_u8 v56, v4 offset:63744 // (op 293) Load local data 

s_mul_i32 s1, s31, 6 // (op 309) most significant: low of high * low
s_mul_i32 s0, s30, 6 // (op 309) least significant
s_mul_hi_u32 s18, s30, 6 // (op 309) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 309) most significant: sum

v_lshrrev_b64 v[136:137], 3, s[0:1] // (op 309) 


v_mov_b32 v138, v124 // (op 309) 
buffer_load_dwordx4 v[58:61], v138, s[60:63], 0 offen offset:288 // (op 309) Load value
buffer_load_dwordx2 v[62:63], v138, s[60:63], 0 offen offset:304 // (op 309) Load value
buffer_load_dwordx4 v[64:67], v138, s[60:63], 0 offen offset:312 // (op 309) Load value
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[112:117], v[88:93], a[0:15], v56, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 296) 
buffer_load_dwordx2 v[68:69], v138, s[60:63], 0 offen offset:328 // (op 309) Load value
v_add_u32 v138, v138, v136 // (op 309) 



v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[112:117], v[94:99], a[16:31], v56, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 461) 
buffer_load_dwordx4 v[70:73], v138, s[60:63], 0 offen offset:288 // (op 309) Load value
buffer_load_dwordx2 v[74:75], v138, s[60:63], 0 offen offset:304 // (op 309) Load value
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[112:117], v[100:105], a[32:47], v56, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 695) 
ds_read_b128 v[140:143], v5 offset:57648 // (op 298) Load local data 
ds_read_b64 v[144:145], v5 offset:57664 // (op 298) Load local data 
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[112:117], v[130:135], a[48:63], v56, v127 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 929) 





s_barrier  // (op 3228) 
ds_read_b128 v[130:133], v13 offset:32048 // (op 300) Load local data 
ds_read_b64 v[134:135], v13 offset:32064 // (op 300) Load local data 
buffer_load_dwordx4 v[76:79], v138, s[60:63], 0 offen offset:312 // (op 309) Load value



ds_read_u8 v127, v4 offset:63746 // (op 302) Load local data 
buffer_load_dwordx2 v[80:81], v138, s[60:63], 0 offen offset:328 // (op 309) Load value









s_barrier  // (op 3232) 
buffer_load_dword v18, v126, s[68:71], 0 offen offset:12 // (op 311) Load value
v_add_u32 v19, 192, v19 // (op 2506) 

ds_read_u8 v124, v12 offset:56578 // (op 304) Load local data 






v_add_u32 v22, 8, v22 // (op 2634) 
ds_read_b128 v[146:149], v13 offset:38192 // (op 465) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[140:145], v[130:135], a[0:15], v127, v124 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 305) 
ds_read_b64 v[150:151], v13 offset:38208 // (op 465) Load local data 





ds_read_u8 v124, v12 offset:56834 // (op 469) Load local data 
buffer_load_dwordx4 v[82:85], v128, s[48:51], 0 offen offset:288 // (op 286) Load value



ds_read_b128 v[130:133], v13 offset:44336 // (op 699) Load local data 
ds_read_b64 v[134:135], v13 offset:44352 // (op 699) Load local data 
buffer_load_dwordx2 v[86:87], v128, s[48:51], 0 offen offset:304 // (op 286) Load value




ds_read_u8 v126, v12 offset:57090 // (op 703) Load local data 






buffer_load_ubyte v14, v129, s[64:67], 0 offen offset:12 // (op 310) Load value
v_add_u32 v15, 192, v15 // (op 2442) 

s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[140:145], v[146:151], a[16:31], v127, v124 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 470) 


s_barrier  // (op 3099) 



v_add_u32 v17, 8, v17 // (op 2570) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[140:145], v[130:135], a[32:47], v127, v126 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 704) 
ds_read_b128 v[146:149], v13 offset:50480 // (op 933) Load local data 


ds_read_b64 v[150:151], v13 offset:50496 // (op 933) Load local data 
s_waitcnt vmcnt(23)
ds_write_b128 v119, v[26:29] // (op 13) Store local data 
s_waitcnt vmcnt(22)
ds_write_b64 v119, v[30:31] offset:16 // (op 13) Store local data 



s_barrier  // (op 3288) 
ds_read_u8 v124, v12 offset:57346 // (op 937) Load local data 
s_waitcnt vmcnt(21)
ds_write_b128 v119, v[32:35] offset:24 // (op 13) Store local data 
s_waitcnt vmcnt(20)
ds_write_b64 v119, v[36:37] offset:40 // (op 13) Store local data 
s_barrier  // (op 3292) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[140:145], v[146:151], a[48:63], v127, v124 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 938) 
s_waitcnt vmcnt(19)
ds_write_b128 v119, v[38:41] offset:96 // (op 13) Store local data 




s_waitcnt vmcnt(18)
ds_write_b64 v119, v[42:43] offset:112 // (op 13) Store local data 
s_waitcnt vmcnt(17)
ds_write_b128 v119, v[44:47] offset:120 // (op 13) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v119, v[48:49] offset:136 // (op 13) Store local data 
s_barrier  // (op 3140) 



s_waitcnt vmcnt(15)
ds_write_b32 v57, v20 offset:24576 // (op 30) Store local data 
s_barrier  // (op 3080) 



s_waitcnt vmcnt(14)
ds_write_b128 v118, v[50:53] offset:25600 // (op 6) Store local data 
s_waitcnt vmcnt(13)
ds_write_b64 v118, v[54:55] offset:25616 // (op 6) Store local data 
s_barrier  // (op 3120) 



s_waitcnt vmcnt(12)
ds_write_b8 v121, v23 offset:31744 // (op 21) Store local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 3109) 



ds_read_b128 v[88:91], v9 // (op 77) Load local data 
ds_read_b64 v[92:93], v9 offset:16 // (op 77) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 3150) 



ds_read_u8 v16, v6 offset:24576 // (op 87) Load local data 



ds_read_b128 v[94:97], v9 offset:6144 // (op 446) Load local data 
ds_read_b64 v[98:99], v9 offset:6160 // (op 446) Load local data 



ds_read_u8 v21, v6 offset:24832 // (op 450) Load local data 



ds_read_b128 v[100:103], v9 offset:12288 // (op 680) Load local data 
ds_read_b64 v[104:105], v9 offset:12304 // (op 680) Load local data 



ds_read_u8 v24, v6 offset:25088 // (op 684) Load local data 



ds_read_b128 v[106:109], v9 offset:18432 // (op 914) Load local data 
ds_read_b64 v[110:111], v9 offset:18448 // (op 914) Load local data 



ds_read_u8 v25, v6 offset:25344 // (op 918) Load local data 
s_waitcnt lgkmcnt(13)
s_barrier  // (op 3089) 



ds_read_b128 v[112:115], v3 offset:25600 // (op 72) Load local data 
ds_read_b64 v[116:117], v3 offset:25616 // (op 72) Load local data 
s_waitcnt lgkmcnt(14)
s_barrier  // (op 3130) 



ds_read_u8 v56, v0 offset:31744 // (op 82) Load local data 
s_ashr_i64 s[0:1], s[42:43], 7 // (op 871) 
s_mov_b32 s18, s0 // (op 871) convert

s_and_b32 s19, s18, 4294967294 // (op 871) 

s_cmp_lt_u32 s47, s19 // (op 871) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopTop_KLoop_871 // (op 871) Condition: Bottom (jump to RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopTop_KLoop_871 if true)
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopBottom_KLoop_871:

















s_mul_hi_i32 s47, s16, s10 // (op 2402) 
v_lshrrev_b32 v15, 2, v1 // (op 2402) 
s_add_i32 s0, s47, s16 // (op 2402) 

v_and_b32 v17, -4, v1 // (op 2402) 
s_ashr_i32 s1, s0, 31 // (op 2402) 
v_sub_u32 v57, v1, v17 // (op 2402) 

s_cmp_eq_i32 s10, 0 // (op 2402) 
s_mov_b32 s47, scc // (op 2402) 
v_lshlrev_b32 v17, 5, v57 // (op 2402) 

s_cmp_eq_i32 s47, 0 // (op 2402) 
s_cselect_b32 s18, 0, -1 // (op 2402) 

s_and_b32 s19, s13, 31 // (op 2402) 
s_lshl_b32 s47, 1, s19 // (op 2402) 
s_add_i32 s22, s47, s18 // (op 2402) 


s_and_b32 s18, s1, s22 // (op 2402) 


s_add_i32 s1, s0, s18 // (op 2402) 


s_ashr_i32 s0, s1, s19 // (op 2402) 


s_lshl_b32 s1, s13, 24 // (op 2402) 
s_ashr_i32 s47, s1, 31 // (op 2402) 

s_xor_b32 s1, s0, s47 // (op 2402) 

s_sub_i32 s0, s1, s47 // (op 2402) 


s_mul_hi_i32 s1, s0, s14 // (op 2402) 
s_add_i32 s47, s1, s0 // (op 2402) 

s_ashr_i32 s1, s47, 31 // (op 2402) 
s_cmp_eq_i32 s14, 0 // (op 2402) 
s_mov_b32 s18, scc // (op 2402) 
s_cmp_eq_i32 s18, 0 // (op 2402) 
s_cselect_b32 s19, 0, -1 // (op 2402) 

s_and_b32 s18, s15, 31 // (op 2402) 
s_lshl_b32 s22, 1, s18 // (op 2402) 
s_add_i32 s23, s22, s19 // (op 2402) 


s_and_b32 s19, s1, s23 // (op 2402) 


s_add_i32 s1, s47, s19 // (op 2402) 


s_ashr_i32 s19, s1, s18 // (op 2402) 


s_lshl_b32 s1, s15, 24 // (op 2402) 
s_ashr_i32 s18, s1, 31 // (op 2402) 

s_xor_b32 s1, s19, s18 // (op 2402) 

s_sub_i32 s19, s1, s18 // (op 2402) 


s_mul_i32 s1, s0, s11 // (op 2402) 

s_sub_u32 s0, s16, s1 // (op 2402) 

s_mul_i32 s1, s19, s11 // (op 2402) 
s_add_u32 s1, s1, s0 // (op 2402) 


s_mul_hi_i32 s0, s16, s9 // (op 2402) 
s_add_i32 s47, s0, s16 // (op 2402) 

s_ashr_i32 s0, s47, 31 // (op 2402) 
s_cmp_eq_i32 s9, 0 // (op 2402) 
s_mov_b32 s18, scc // (op 2402) 
s_cmp_eq_i32 s18, 0 // (op 2402) 
s_cselect_b32 s19, 0, -1 // (op 2402) 

s_and_b32 s18, s45, 31 // (op 2402) 
s_lshl_b32 s22, 1, s18 // (op 2402) 
s_add_i32 s23, s22, s19 // (op 2402) 


s_and_b32 s19, s0, s23 // (op 2402) 


s_add_i32 s0, s47, s19 // (op 2402) 


s_ashr_i32 s19, s0, s18 // (op 2402) 


s_lshl_b32 s0, s45, 24 // (op 2402) 
s_ashr_i32 s18, s0, 31 // (op 2402) 

s_xor_b32 s0, s19, s18 // (op 2402) 

s_sub_i32 s19, s0, s18 // (op 2402) 


s_add_u32 s22, s6, 256 // (op 2402) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2402) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2402) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2402) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 2402) 

s_mul_i32 s0, s11, s22 // (op 2402) 

s_mul_i32 s18, s8, s0 // (op 2402) 

s_mul_i32 s0, s19, s18 // (op 2402) 

s_sub_u32 s18, s16, s0 // (op 2402) 

s_mov_b32 s0, s18 // (op 2402) 
s_mul_hi_i32 s47, s0, s12 // (op 2402) 
s_add_i32 s22, s47, s0 // (op 2402) 


s_ashr_i32 s0, s22, 31 // (op 2402) 
s_cmp_eq_i32 s12, 0 // (op 2402) 
s_mov_b32 s47, scc // (op 2402) 
s_cmp_eq_i32 s47, 0 // (op 2402) 
s_cselect_b32 s23, 0, -1 // (op 2402) 

s_and_b32 s47, s44, 31 // (op 2402) 
s_lshl_b32 s24, 1, s47 // (op 2402) 
s_add_i32 s25, s24, s23 // (op 2402) 


s_and_b32 s23, s0, s25 // (op 2402) 


s_add_i32 s0, s22, s23 // (op 2402) 


s_ashr_i32 s22, s0, s47 // (op 2402) 


s_lshl_b32 s0, s44, 24 // (op 2402) 
s_ashr_i32 s47, s0, 31 // (op 2402) 

s_xor_b32 s0, s22, s47 // (op 2402) 

s_sub_i32 s22, s0, s47 // (op 2402) 


s_mul_i32 s0, s46, s11 // (op 2402) 
s_add_u32 s24, s20, 64 // (op 2402) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2402) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 2402) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 2402) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 6 // (op 2402) 

s_sub_i32 s23, s24, s0 // (op 2402) 

s_mul_i32 s47, s22, s23 // (op 2402) 


s_sub_u32 s22, s18, s47 // (op 2402) 


s_add_u32 s18, s22, s0 // (op 2402) 


s_cmp_eq_i32 s19, 0 // (op 2402) 
s_mov_b32 s0, scc // (op 2402) 

s_cmp_eq_i32 s0, 0 // (op 2402) 
s_cselect_b32 s19, s18, s1 // (op 2402) 



v_lshl_add_u32 v57, s19, 6, v15 // (op 2402) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 2402) 
s_lshr_b32 s47, s0, 1 // (op 2402) 

v_lshl_add_u32 v15, s47, 8, v17 // (op 2402) 


v_mov_b32 v23, 0 // (op 2402) convert
v_mov_b32 v22, v15 // (op 2402) convert

v_mul_lo_u32 v27, v57, s27 // (op 2402) most significant: low of low * high
v_mul_hi_u32 v15, v57, s26 // (op 2402) most significant: high of low * low
v_mul_lo_u32 v26, v57, s26 // (op 2402) least significant: low of low * low
v_add_u32 v27, v27, v15 // (op 2402) most significant: sum

v_add_co_u32 v26, s[0:1], v26, v22 // (op 2402) least significant half
v_addc_co_u32 v27, s[0:1], v27, v23, s[0:1] // (op 2402) most significant half



v_mul_lo_u32 v23, v27, 6 // (op 2402) most significant: low of high * low
v_mul_hi_u32 v15, v26, 6 // (op 2402) most significant: high of low * low
v_mul_lo_u32 v22, v26, 6 // (op 2402) least significant: low of low * low
v_add_u32 v23, v23, v15 // (op 2402) most significant: sum


v_lshrrev_b64 v[26:27], 3, v[22:23] // (op 2402) 

v_mov_b32 v15, v26 // (op 2402) convert

s_mul_i32 s1, s21, 6 // (op 2404) most significant: low of high * low
s_mul_i32 s0, s20, 6 // (op 2404) least significant
s_mul_hi_u32 s47, s20, 6 // (op 2404) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2404) most significant: sum

s_mul_i32 s19, s26, s1 // (op 2404) most significant: low of low * high
s_mul_i32 s47, s27, s0 // (op 2404) most significant: low of high * low
s_mul_i32 s18, s26, s0 // (op 2404) least significant
s_mul_hi_u32 s22, s26, s0 // (op 2404) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 2404) most significant: sum
s_add_u32 s19, s19, s22 // (op 2404) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 2404) 

s_mov_b32 s48, s28 // (op 2404) reg expression
s_mov_b32 s49, s29 // (op 2404) reg expression
s_mov_b32 s50, s0 // (op 2404) reg expression
s_mov_b32 s51, 131072 // (op 2404) call()

s_mul_hi_i32 s47, s16, s10 // (op 2472) 
v_and_b32 v17, 4294967294, v1 // (op 2472) 
s_add_i32 s0, s47, s16 // (op 2472) 

v_and_b32 v57, -2, v1 // (op 2472) 
s_ashr_i32 s1, s0, 31 // (op 2472) 
v_sub_u32 v19, v1, v57 // (op 2472) 

s_cmp_eq_i32 s10, 0 // (op 2472) 
s_mov_b32 s47, scc // (op 2472) 
v_lshlrev_b32 v20, 6, v19 // (op 2472) 

s_cmp_eq_i32 s47, 0 // (op 2472) 
s_cselect_b32 s18, 0, -1 // (op 2472) 

s_and_b32 s19, s13, 31 // (op 2472) 
s_lshl_b32 s47, 1, s19 // (op 2472) 
s_add_i32 s22, s47, s18 // (op 2472) 


s_and_b32 s18, s1, s22 // (op 2472) 


s_add_i32 s1, s0, s18 // (op 2472) 


s_ashr_i32 s0, s1, s19 // (op 2472) 


s_lshl_b32 s1, s13, 24 // (op 2472) 
s_ashr_i32 s47, s1, 31 // (op 2472) 

s_xor_b32 s1, s0, s47 // (op 2472) 

s_sub_i32 s0, s1, s47 // (op 2472) 


s_mul_hi_i32 s1, s0, s14 // (op 2472) 
s_add_i32 s47, s1, s0 // (op 2472) 

s_ashr_i32 s1, s47, 31 // (op 2472) 
s_cmp_eq_i32 s14, 0 // (op 2472) 
s_mov_b32 s18, scc // (op 2472) 
s_cmp_eq_i32 s18, 0 // (op 2472) 
s_cselect_b32 s19, 0, -1 // (op 2472) 

s_and_b32 s18, s15, 31 // (op 2472) 
s_lshl_b32 s22, 1, s18 // (op 2472) 
s_add_i32 s23, s22, s19 // (op 2472) 


s_and_b32 s19, s1, s23 // (op 2472) 


s_add_i32 s1, s47, s19 // (op 2472) 


s_ashr_i32 s19, s1, s18 // (op 2472) 


s_lshl_b32 s1, s15, 24 // (op 2472) 
s_ashr_i32 s18, s1, 31 // (op 2472) 

s_xor_b32 s1, s19, s18 // (op 2472) 

s_sub_i32 s19, s1, s18 // (op 2472) 


s_add_u32 s22, s6, 256 // (op 2472) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2472) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2472) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2472) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 2472) 

s_mul_i32 s1, s19, s22 // (op 2472) 

s_sub_i32 s47, s0, s1 // (op 2472) 


s_mul_hi_i32 s0, s16, s9 // (op 2472) 
s_add_i32 s1, s0, s16 // (op 2472) 

s_ashr_i32 s0, s1, 31 // (op 2472) 
s_cmp_eq_i32 s9, 0 // (op 2472) 
s_mov_b32 s18, scc // (op 2472) 
s_cmp_eq_i32 s18, 0 // (op 2472) 
s_cselect_b32 s19, 0, -1 // (op 2472) 

s_and_b32 s18, s45, 31 // (op 2472) 
s_lshl_b32 s24, 1, s18 // (op 2472) 
s_add_i32 s25, s24, s19 // (op 2472) 


s_and_b32 s19, s0, s25 // (op 2472) 


s_add_i32 s0, s1, s19 // (op 2472) 


s_ashr_i32 s1, s0, s18 // (op 2472) 


s_lshl_b32 s0, s45, 24 // (op 2472) 
s_ashr_i32 s18, s0, 31 // (op 2472) 

s_xor_b32 s0, s1, s18 // (op 2472) 

s_sub_i32 s1, s0, s18 // (op 2472) 


s_mul_i32 s0, s11, s22 // (op 2472) 

s_mul_i32 s18, s8, s0 // (op 2472) 

s_mul_i32 s0, s1, s18 // (op 2472) 

s_sub_u32 s18, s16, s0 // (op 2472) 

s_mul_hi_i32 s0, s18, s12 // (op 2472) 
s_add_i32 s19, s0, s18 // (op 2472) 


s_ashr_i32 s0, s19, 31 // (op 2472) 
s_cmp_eq_i32 s12, 0 // (op 2472) 
s_mov_b32 s18, scc // (op 2472) 
s_cmp_eq_i32 s18, 0 // (op 2472) 
s_cselect_b32 s22, 0, -1 // (op 2472) 

s_and_b32 s18, s44, 31 // (op 2472) 
s_lshl_b32 s23, 1, s18 // (op 2472) 
s_add_i32 s24, s23, s22 // (op 2472) 


s_and_b32 s25, s0, s24 // (op 2472) 


s_add_i32 s0, s19, s25 // (op 2472) 


s_ashr_i32 s19, s0, s18 // (op 2472) 


s_lshl_b32 s0, s44, 24 // (op 2472) 
s_ashr_i32 s18, s0, 31 // (op 2472) 

s_xor_b32 s0, s19, s18 // (op 2472) 

s_sub_i32 s19, s0, s18 // (op 2472) 


s_cmp_eq_i32 s1, 0 // (op 2472) 
s_mov_b32 s0, scc // (op 2472) 

s_cmp_eq_i32 s0, 0 // (op 2472) 
s_cselect_b32 s1, s19, s47 // (op 2472) 



v_lshl_add_u32 v19, s1, 8, v17 // (op 2472) 


v_mul_lo_u32 v23, v19, s31 // (op 2472) most significant: low of low * high
v_mul_hi_u32 v17, v19, s30 // (op 2472) most significant: high of low * low
v_mul_lo_u32 v22, v19, s30 // (op 2472) least significant: low of low * low
v_add_u32 v23, v23, v17 // (op 2472) most significant: sum


s_ashr_i64 s[0:1], s[42:43], 7 // (op 2472) 
s_lshr_b32 s47, s0, 1 // (op 2472) 

v_lshl_add_u32 v17, s47, 8, v20 // (op 2472) 


v_mov_b32 v27, 0 // (op 2472) convert
v_mov_b32 v26, v17 // (op 2472) convert

v_add_co_u32 v28, s[0:1], v26, v22 // (op 2472) least significant half
v_addc_co_u32 v29, s[0:1], v27, v23, s[0:1] // (op 2472) most significant half



v_mul_lo_u32 v23, v29, 6 // (op 2472) most significant: low of high * low
v_mul_hi_u32 v17, v28, 6 // (op 2472) most significant: high of low * low
v_mul_lo_u32 v22, v28, 6 // (op 2472) least significant: low of low * low
v_add_u32 v23, v23, v17 // (op 2472) most significant: sum


v_lshrrev_b64 v[26:27], 3, v[22:23] // (op 2472) 

v_mov_b32 v17, v26 // (op 2472) convert

s_mul_i32 s1, s7, 6 // (op 2474) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 2474) least significant
s_mul_hi_u32 s47, s6, 6 // (op 2474) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2474) most significant: sum

s_mul_i32 s19, s30, s1 // (op 2474) most significant: low of low * high
s_mul_i32 s47, s31, s0 // (op 2474) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 2474) least significant
s_mul_hi_u32 s22, s30, s0 // (op 2474) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 2474) most significant: sum
s_add_u32 s19, s19, s22 // (op 2474) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 2474) 

s_mov_b32 s60, s32 // (op 2474) reg expression
s_mov_b32 s61, s33 // (op 2474) reg expression
s_mov_b32 s62, s0 // (op 2474) reg expression
s_mov_b32 s63, 131072 // (op 2474) call()

s_mul_hi_i32 s47, s16, s10 // (op 2536) 
v_lshrrev_b32 v57, 2, v1 // (op 2536) 
s_add_i32 s0, s47, s16 // (op 2536) 

v_and_b32 v19, -4, v1 // (op 2536) 
s_ashr_i32 s1, s0, 31 // (op 2536) 
v_sub_u32 v20, v1, v19 // (op 2536) 

s_cmp_eq_i32 s10, 0 // (op 2536) 
s_mov_b32 s47, scc // (op 2536) 
s_cmp_eq_i32 s47, 0 // (op 2536) 
s_cselect_b32 s18, 0, -1 // (op 2536) 

s_and_b32 s19, s13, 31 // (op 2536) 
s_lshl_b32 s47, 1, s19 // (op 2536) 
s_add_i32 s22, s47, s18 // (op 2536) 


s_and_b32 s18, s1, s22 // (op 2536) 


s_add_i32 s1, s0, s18 // (op 2536) 


s_ashr_i32 s0, s1, s19 // (op 2536) 


s_lshl_b32 s1, s13, 24 // (op 2536) 
s_ashr_i32 s47, s1, 31 // (op 2536) 

s_xor_b32 s1, s0, s47 // (op 2536) 

s_sub_i32 s0, s1, s47 // (op 2536) 


s_mul_hi_i32 s1, s0, s14 // (op 2536) 
s_add_i32 s47, s1, s0 // (op 2536) 

s_ashr_i32 s1, s47, 31 // (op 2536) 
s_cmp_eq_i32 s14, 0 // (op 2536) 
s_mov_b32 s18, scc // (op 2536) 
s_cmp_eq_i32 s18, 0 // (op 2536) 
s_cselect_b32 s19, 0, -1 // (op 2536) 

s_and_b32 s18, s15, 31 // (op 2536) 
s_lshl_b32 s22, 1, s18 // (op 2536) 
s_add_i32 s23, s22, s19 // (op 2536) 


s_and_b32 s19, s1, s23 // (op 2536) 


s_add_i32 s1, s47, s19 // (op 2536) 


s_ashr_i32 s19, s1, s18 // (op 2536) 


s_lshl_b32 s1, s15, 24 // (op 2536) 
s_ashr_i32 s18, s1, 31 // (op 2536) 

s_xor_b32 s1, s19, s18 // (op 2536) 

s_sub_i32 s19, s1, s18 // (op 2536) 


s_mul_i32 s1, s0, s11 // (op 2536) 

s_sub_u32 s0, s16, s1 // (op 2536) 

s_mul_i32 s1, s19, s11 // (op 2536) 
s_add_u32 s1, s1, s0 // (op 2536) 


s_mul_hi_i32 s0, s16, s9 // (op 2536) 
s_add_i32 s47, s0, s16 // (op 2536) 

s_ashr_i32 s0, s47, 31 // (op 2536) 
s_cmp_eq_i32 s9, 0 // (op 2536) 
s_mov_b32 s18, scc // (op 2536) 
s_cmp_eq_i32 s18, 0 // (op 2536) 
s_cselect_b32 s19, 0, -1 // (op 2536) 

s_and_b32 s18, s45, 31 // (op 2536) 
s_lshl_b32 s22, 1, s18 // (op 2536) 
s_add_i32 s23, s22, s19 // (op 2536) 


s_and_b32 s19, s0, s23 // (op 2536) 


s_add_i32 s0, s47, s19 // (op 2536) 


s_ashr_i32 s19, s0, s18 // (op 2536) 


s_lshl_b32 s0, s45, 24 // (op 2536) 
s_ashr_i32 s18, s0, 31 // (op 2536) 

s_xor_b32 s0, s19, s18 // (op 2536) 

s_sub_i32 s19, s0, s18 // (op 2536) 


s_add_u32 s22, s6, 256 // (op 2536) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2536) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2536) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2536) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 2536) 

s_mul_i32 s0, s11, s22 // (op 2536) 

s_mul_i32 s18, s8, s0 // (op 2536) 

s_mul_i32 s0, s19, s18 // (op 2536) 

s_sub_u32 s18, s16, s0 // (op 2536) 

s_mov_b32 s0, s18 // (op 2536) 
s_mul_hi_i32 s47, s0, s12 // (op 2536) 
s_add_i32 s22, s47, s0 // (op 2536) 


s_ashr_i32 s0, s22, 31 // (op 2536) 
s_cmp_eq_i32 s12, 0 // (op 2536) 
s_mov_b32 s47, scc // (op 2536) 
s_cmp_eq_i32 s47, 0 // (op 2536) 
s_cselect_b32 s23, 0, -1 // (op 2536) 

s_and_b32 s47, s44, 31 // (op 2536) 
s_lshl_b32 s24, 1, s47 // (op 2536) 
s_add_i32 s25, s24, s23 // (op 2536) 


s_and_b32 s23, s0, s25 // (op 2536) 


s_add_i32 s0, s22, s23 // (op 2536) 


s_ashr_i32 s22, s0, s47 // (op 2536) 


s_lshl_b32 s0, s44, 24 // (op 2536) 
s_ashr_i32 s47, s0, 31 // (op 2536) 

s_xor_b32 s0, s22, s47 // (op 2536) 

s_sub_i32 s22, s0, s47 // (op 2536) 


s_mul_i32 s0, s46, s11 // (op 2536) 
s_add_u32 s24, s20, 64 // (op 2536) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2536) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 2536) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 2536) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 6 // (op 2536) 

s_sub_i32 s23, s24, s0 // (op 2536) 

s_mul_i32 s47, s22, s23 // (op 2536) 


s_sub_u32 s22, s18, s47 // (op 2536) 


s_add_u32 s18, s22, s0 // (op 2536) 


s_cmp_eq_i32 s19, 0 // (op 2536) 
s_mov_b32 s0, scc // (op 2536) 

s_cmp_eq_i32 s0, 0 // (op 2536) 
s_cselect_b32 s19, s18, s1 // (op 2536) 



v_lshl_add_u32 v19, s19, 6, v57 // (op 2536) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 2536) 
s_lshr_b32 s47, s0, 1 // (op 2536) 

v_lshl_add_u32 v57, s47, 3, v20 // (op 2536) 


v_mul_lo_u32 v20, v19, s34 // (op 2536) 
v_add_u32 v20, v20, v57 // (op 2536) 


s_mul_i32 s1, s34, s21 // (op 2538) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 2538) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 2538) least significant
s_mul_hi_u32 s18, s34, s20 // (op 2538) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2538) most significant: sum
s_add_u32 s1, s1, s18 // (op 2538) most significant: sum


s_mov_b32 s64, s36 // (op 2538) reg expression
s_mov_b32 s65, s37 // (op 2538) reg expression
s_mov_b32 s66, s0 // (op 2538) reg expression
s_mov_b32 s67, 131072 // (op 2538) call()

s_mul_hi_i32 s47, s16, s10 // (op 2600) 
s_add_i32 s0, s47, s16 // (op 2600) 

s_ashr_i32 s1, s0, 31 // (op 2600) 
s_cmp_eq_i32 s10, 0 // (op 2600) 
s_mov_b32 s47, scc // (op 2600) 
s_cmp_eq_i32 s47, 0 // (op 2600) 
s_cselect_b32 s18, 0, -1 // (op 2600) 

s_and_b32 s19, s13, 31 // (op 2600) 
s_lshl_b32 s47, 1, s19 // (op 2600) 
s_add_i32 s22, s47, s18 // (op 2600) 


s_and_b32 s18, s1, s22 // (op 2600) 


s_add_i32 s1, s0, s18 // (op 2600) 


s_ashr_i32 s0, s1, s19 // (op 2600) 


s_lshl_b32 s1, s13, 24 // (op 2600) 
s_ashr_i32 s47, s1, 31 // (op 2600) 

s_xor_b32 s1, s0, s47 // (op 2600) 

s_sub_i32 s0, s1, s47 // (op 2600) 


s_mul_hi_i32 s1, s0, s14 // (op 2600) 
s_add_i32 s47, s1, s0 // (op 2600) 

s_ashr_i32 s1, s47, 31 // (op 2600) 
s_cmp_eq_i32 s14, 0 // (op 2600) 
s_mov_b32 s18, scc // (op 2600) 
s_cmp_eq_i32 s18, 0 // (op 2600) 
s_cselect_b32 s19, 0, -1 // (op 2600) 

s_and_b32 s18, s15, 31 // (op 2600) 
s_lshl_b32 s22, 1, s18 // (op 2600) 
s_add_i32 s23, s22, s19 // (op 2600) 


s_and_b32 s19, s1, s23 // (op 2600) 


s_add_i32 s1, s47, s19 // (op 2600) 


s_ashr_i32 s19, s1, s18 // (op 2600) 


s_lshl_b32 s1, s15, 24 // (op 2600) 
s_ashr_i32 s18, s1, 31 // (op 2600) 

s_xor_b32 s1, s19, s18 // (op 2600) 

s_sub_i32 s19, s1, s18 // (op 2600) 


s_add_u32 s22, s6, 256 // (op 2600) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2600) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2600) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2600) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 2600) 

s_mul_i32 s1, s19, s22 // (op 2600) 

s_sub_i32 s47, s0, s1 // (op 2600) 


s_mul_hi_i32 s0, s16, s9 // (op 2600) 
s_add_i32 s1, s0, s16 // (op 2600) 

s_ashr_i32 s0, s1, 31 // (op 2600) 
s_cmp_eq_i32 s9, 0 // (op 2600) 
s_mov_b32 s18, scc // (op 2600) 
s_cmp_eq_i32 s18, 0 // (op 2600) 
s_cselect_b32 s19, 0, -1 // (op 2600) 

s_and_b32 s18, s45, 31 // (op 2600) 
s_lshl_b32 s24, 1, s18 // (op 2600) 
s_add_i32 s25, s24, s19 // (op 2600) 


s_and_b32 s19, s0, s25 // (op 2600) 


s_add_i32 s0, s1, s19 // (op 2600) 


s_ashr_i32 s1, s0, s18 // (op 2600) 


s_lshl_b32 s0, s45, 24 // (op 2600) 
s_ashr_i32 s18, s0, 31 // (op 2600) 

s_xor_b32 s0, s1, s18 // (op 2600) 

s_sub_i32 s1, s0, s18 // (op 2600) 


s_mul_i32 s0, s11, s22 // (op 2600) 

s_mul_i32 s18, s8, s0 // (op 2600) 

s_mul_i32 s0, s1, s18 // (op 2600) 

s_sub_u32 s18, s16, s0 // (op 2600) 

s_mul_hi_i32 s0, s18, s12 // (op 2600) 
s_add_i32 s19, s0, s18 // (op 2600) 


s_ashr_i32 s0, s19, 31 // (op 2600) 
s_cmp_eq_i32 s12, 0 // (op 2600) 
s_mov_b32 s18, scc // (op 2600) 
s_cmp_eq_i32 s18, 0 // (op 2600) 
s_cselect_b32 s22, 0, -1 // (op 2600) 

s_and_b32 s18, s44, 31 // (op 2600) 
s_lshl_b32 s23, 1, s18 // (op 2600) 
s_add_i32 s24, s23, s22 // (op 2600) 


s_and_b32 s25, s0, s24 // (op 2600) 


s_add_i32 s0, s19, s25 // (op 2600) 


s_ashr_i32 s19, s0, s18 // (op 2600) 


s_lshl_b32 s0, s44, 24 // (op 2600) 
s_ashr_i32 s18, s0, 31 // (op 2600) 

s_xor_b32 s0, s19, s18 // (op 2600) 

s_sub_i32 s19, s0, s18 // (op 2600) 


s_cmp_eq_i32 s1, 0 // (op 2600) 
s_mov_b32 s0, scc // (op 2600) 

s_cmp_eq_i32 s0, 0 // (op 2600) 
s_cselect_b32 s1, s19, s47 // (op 2600) 



v_lshl_add_u32 v19, s1, 8, v1 // (op 2600) 

v_mul_lo_u32 v57, v19, s38 // (op 2600) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 2600) 
s_lshr_b32 s47, s0, 1 // (op 2600) 

v_lshl_add_u32 v19, s47, 3, v57 // (op 2600) 


s_mul_i32 s1, s38, s7 // (op 2602) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 2602) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 2602) least significant
s_mul_hi_u32 s18, s38, s6 // (op 2602) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2602) most significant: sum
s_add_u32 s1, s1, s18 // (op 2602) most significant: sum


s_mov_b32 s68, s40 // (op 2602) reg expression
s_mov_b32 s69, s41 // (op 2602) reg expression
s_mov_b32 s70, s0 // (op 2602) reg expression
s_mov_b32 s71, 131072 // (op 2602) call()

v_and_b32 v57, -4, v1 // (op 2693) 
v_sub_u32 v22, v1, v57 // (op 2693) 

v_lshlrev_b32 v23, 5, v22 // (op 2693) 

v_lshrrev_b32 v22, 2, v1 // (op 2693) 
v_lshl_add_u32 v57, v22, 7, v23 // (op 2693) 


v_mul_lo_u32 v22, v57, 6 // (op 2693) 

v_lshrrev_b32 v23, 3, v22 // (op 2693) 

v_and_b32 v22, -2, v1 // (op 2746) 
v_sub_u32 v57, v1, v22 // (op 2746) 

v_lshlrev_b32 v22, 6, v57 // (op 2746) 

v_lshrrev_b32 v57, 1, v1 // (op 2746) 
v_lshl_add_u32 v26, v57, 8, v22 // (op 2746) 


v_mul_lo_u32 v22, v26, 6 // (op 2746) 

v_lshrrev_b32 v57, 3, v22 // (op 2746) 

v_and_b32 v22, -4, v1 // (op 2804) 
v_sub_u32 v26, v1, v22 // (op 2804) 

v_and_b32 v22, 4294967292, v1 // (op 2804) 
v_add_u32 v27, v22, v26 // (op 2804) 


v_lshlrev_b32 v22, 2, v1 // (op 2857) 
s_ashr_i64 s[0:1], s[42:43], 7 // (op 877) 
s_and_b32 s47, s0, 4294967294 // (op 877) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 876) 
s_mov_b32 s18, s0 // (op 876) convert

s_cmp_lt_u32 s47, s18 // (op 876) 















s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_876 // (op 876) Condition: Top (jump to RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_876 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopTop_KLoopTail_876:


s_add_u32 s47, s47, 1 // (op 878) 
v_mov_b32 v14, v15 // (op 202) 

buffer_load_dwordx4 v[28:31], v14, s[48:51], 0 offen // (op 202) Load value
buffer_load_dwordx2 v[32:33], v14, s[48:51], 0 offen offset:16 // (op 202) Load value

v_add_u32 v15, 96, v15 // (op 2392) 

s_barrier  // (op 3083) 
s_mul_i32 s1, s31, 6 // (op 225) most significant: low of high * low



s_mul_i32 s0, s30, 6 // (op 225) least significant
s_mul_hi_u32 s18, s30, 6 // (op 225) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 225) most significant: sum

v_lshrrev_b64 v[24:25], 3, s[0:1] // (op 225) 


v_mov_b32 v14, v17 // (op 225) 
v_mov_b32 v16, v14 // (op 225) 
buffer_load_dwordx4 v[58:61], v16, s[60:63], 0 offen // (op 225) Load value
s_waitcnt vmcnt(2)
ds_write_b128 v23, v[28:31] offset:25600 // (op 203) Store local data 
buffer_load_dwordx2 v[62:63], v16, s[60:63], 0 offen offset:16 // (op 225) Load value
buffer_load_dwordx4 v[64:67], v16, s[60:63], 0 offen offset:24 // (op 225) Load value
buffer_load_dwordx2 v[68:69], v16, s[60:63], 0 offen offset:40 // (op 225) Load value
v_add_u32 v16, v16, v24 // (op 225) 
s_waitcnt vmcnt(4)
ds_write_b64 v23, v[32:33] offset:25616 // (op 203) Store local data 
buffer_load_dwordx4 v[70:73], v16, s[60:63], 0 offen // (op 225) Load value

buffer_load_dwordx2 v[74:75], v16, s[60:63], 0 offen offset:16 // (op 225) Load value
buffer_load_dwordx4 v[76:79], v16, s[60:63], 0 offen offset:24 // (op 225) Load value
buffer_load_dwordx2 v[80:81], v16, s[60:63], 0 offen offset:40 // (op 225) Load value



v_add_u32 v17, 96, v17 // (op 2464) 


s_barrier  // (op 3102) 
v_mov_b32 v14, v20 // (op 226) 




buffer_load_ubyte v16, v14, s[64:67], 0 offen offset: 0 // (op 226) Load value

v_add_u32 v20, 4, v20 // (op 2528) 
s_waitcnt vmcnt(8)
ds_write_b128 v57, v[58:61] // (op 222) Store local data 


v_mov_b32 v14, v19 // (op 227) 

buffer_load_dword v18, v14, s[68:71], 0 offen // (op 227) Load value

v_add_u32 v19, 4, v19 // (op 2592) 
s_waitcnt vmcnt(8)
ds_write_b64 v57, v[62:63] offset:16 // (op 222) Store local data 
s_waitcnt lgkmcnt(2)
s_barrier  // (op 3092) 
s_waitcnt vmcnt(7)
ds_write_b128 v57, v[64:67] offset:24 // (op 222) Store local data 



ds_read_b128 v[28:31], v3 offset:25600 // (op 205) Load local data 
ds_read_b64 v[32:33], v3 offset:25616 // (op 205) Load local data 
s_waitcnt vmcnt(6)
ds_write_b64 v57, v[68:69] offset:40 // (op 222) Store local data 
s_waitcnt vmcnt(5)
ds_write_b128 v57, v[70:73] offset:96 // (op 222) Store local data 
s_waitcnt vmcnt(4)
ds_write_b64 v57, v[74:75] offset:112 // (op 222) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v57, v[76:79] offset:120 // (op 222) Store local data 
s_waitcnt vmcnt(2)
ds_write_b64 v57, v[80:81] offset:136 // (op 222) Store local data 

s_barrier  // (op 3123) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3113) 






ds_read_b128 v[34:37], v9 // (op 207) Load local data 
ds_read_b64 v[38:39], v9 offset:16 // (op 207) Load local data 
s_waitcnt vmcnt(1)
ds_write_b8 v27, v16 offset:31744 // (op 223) Store local data 

s_barrier  // (op 3143) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 3133) 






ds_read_u8 v14, v0 offset:31744 // (op 209) Load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v22, v18 offset:24576 // (op 224) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 3153) 



ds_read_u8 v16, v6 offset:24576 // (op 211) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[28:33], v[34:39], a[0:15], v14, v16 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 212) 





ds_read_b128 v[34:37], v3 offset:25648 // (op 214) Load local data 
ds_read_b64 v[38:39], v3 offset:25664 // (op 214) Load local data 



ds_read_b128 v[40:43], v9 offset:48 // (op 216) Load local data 
ds_read_b64 v[44:45], v9 offset:64 // (op 216) Load local data 



ds_read_u8 v16, v0 offset:31746 // (op 218) Load local data 



ds_read_u8 v18, v6 offset:24578 // (op 220) Load local data 



ds_read_b128 v[46:49], v9 offset:6144 // (op 416) Load local data 
ds_read_b64 v[50:51], v9 offset:6160 // (op 416) Load local data 



ds_read_u8 v21, v6 offset:24832 // (op 420) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[28:33], v[46:51], a[16:31], v14, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 421) 





ds_read_b128 v[46:49], v9 offset:6192 // (op 425) Load local data 
ds_read_b64 v[50:51], v9 offset:6208 // (op 425) Load local data 



ds_read_u8 v21, v6 offset:24834 // (op 429) Load local data 



ds_read_b128 v[58:61], v9 offset:12288 // (op 650) Load local data 
ds_read_b64 v[62:63], v9 offset:12304 // (op 650) Load local data 



ds_read_u8 v24, v6 offset:25088 // (op 654) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[28:33], v[58:63], a[32:47], v14, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 655) 





ds_read_b128 v[58:61], v9 offset:12336 // (op 659) Load local data 
ds_read_b64 v[62:63], v9 offset:12352 // (op 659) Load local data 



ds_read_u8 v24, v6 offset:25090 // (op 663) Load local data 



ds_read_b128 v[64:67], v9 offset:18432 // (op 884) Load local data 
ds_read_b64 v[68:69], v9 offset:18448 // (op 884) Load local data 



ds_read_u8 v25, v6 offset:25344 // (op 888) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[28:33], v[64:69], a[48:63], v14, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 889) 







v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[34:39], v[40:45], a[0:15], v16, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 221) 
ds_read_b128 v[28:31], v9 offset:18480 // (op 893) Load local data 


ds_read_b64 v[32:33], v9 offset:18496 // (op 893) Load local data 



v_mfma_scale_f32_32x32x64_f8f6f4 a[16:31], v[34:39], v[46:51], a[16:31], v16, v21 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 430) 
ds_read_u8 v14, v6 offset:25346 // (op 897) Load local data 


v_mfma_scale_f32_32x32x64_f8f6f4 a[32:47], v[34:39], v[58:63], a[32:47], v16, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 664) 


s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[48:63], v[34:39], v[28:33], a[48:63], v16, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 898) 




s_ashr_i64 s[0:1], s[42:43], 7 // (op 876) 
s_mov_b32 s18, s0 // (op 876) convert

s_cmp_lt_u32 s47, s18 // (op 876) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopTop_KLoopTail_876 // (op 876) Condition: Bottom (jump to RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopTop_KLoopTail_876 if true)
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_876:






















s_barrier  // (op 3274) 





















v_cmp_eq_f32 vcc, 0.00000, v7 // (op 157) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ConditionalFalse_0_beta_157 // (op 157) Condition: False, jump to RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ConditionalFalse_0_beta_157
v_accvgpr_read v12, a0 // (op 50) 
v_accvgpr_read v13, a1 // (op 50) 
v_accvgpr_read v14, a2 // (op 50) 
v_accvgpr_read v15, a3 // (op 50) 
v_accvgpr_read v16, a4 // (op 50) 
v_accvgpr_read v17, a5 // (op 50) 
v_accvgpr_read v18, a6 // (op 50) 
v_accvgpr_read v19, a7 // (op 50) 
v_accvgpr_read v20, a8 // (op 50) 
v_accvgpr_read v21, a9 // (op 50) 
v_accvgpr_read v22, a10 // (op 50) 
v_accvgpr_read v23, a11 // (op 50) 
v_accvgpr_read v24, a12 // (op 50) 
v_accvgpr_read v25, a13 // (op 50) 
v_accvgpr_read v26, a14 // (op 50) 
v_accvgpr_read v27, a15 // (op 50) 
v_mul_f32 v28, v10, v12 // (op 50) 
v_mul_f32 v29, v10, v13 // (op 50) 
v_mul_f32 v30, v10, v14 // (op 50) 
v_mul_f32 v31, v10, v15 // (op 50) 
v_mul_f32 v32, v10, v16 // (op 50) 
v_mul_f32 v33, v10, v17 // (op 50) 
v_mul_f32 v34, v10, v18 // (op 50) 
v_mul_f32 v35, v10, v19 // (op 50) 
v_mul_f32 v36, v10, v20 // (op 50) 
v_mul_f32 v37, v10, v21 // (op 50) 
v_mul_f32 v38, v10, v22 // (op 50) 
v_mul_f32 v39, v10, v23 // (op 50) 
v_mul_f32 v40, v10, v24 // (op 50) 
v_mul_f32 v41, v10, v25 // (op 50) 
v_mul_f32 v42, v10, v26 // (op 50) 
v_mul_f32 v43, v10, v27 // (op 50) 

v_cvt_f16_f32 v12, v28 // (op 56) 
v_cvt_f16_f32 v13, v29 // (op 56) 
v_cvt_f16_f32 v14, v30 // (op 56) 
v_cvt_f16_f32 v15, v31 // (op 56) 
v_cvt_f16_f32 v16, v32 // (op 56) 
v_cvt_f16_f32 v17, v33 // (op 56) 
v_cvt_f16_f32 v18, v34 // (op 56) 
v_cvt_f16_f32 v19, v35 // (op 56) 
v_cvt_f16_f32 v20, v36 // (op 56) 
v_cvt_f16_f32 v21, v37 // (op 56) 
v_cvt_f16_f32 v22, v38 // (op 56) 
v_cvt_f16_f32 v23, v39 // (op 56) 
v_cvt_f16_f32 v24, v40 // (op 56) 
v_cvt_f16_f32 v25, v41 // (op 56) 
v_cvt_f16_f32 v26, v42 // (op 56) 
v_cvt_f16_f32 v27, v43 // (op 56) 


v_pack_b32_f16 v28, v12, v13 // (op 58) 
v_pack_b32_f16 v29, v14, v15 // (op 58) 
v_pack_b32_f16 v30, v16, v17 // (op 58) 
v_pack_b32_f16 v31, v18, v19 // (op 58) 
v_pack_b32_f16 v32, v20, v21 // (op 58) 
v_pack_b32_f16 v33, v22, v23 // (op 58) 
v_pack_b32_f16 v34, v24, v25 // (op 58) 
v_pack_b32_f16 v35, v26, v27 // (op 58) 

v_mov_b32 v0, v11 // (op 58) reg expression

buffer_store_dwordx2 v[28:29], v0, s[56:59], 0 offen // (op 58) Store value
buffer_store_dwordx2 v[30:31], v0, s[56:59], 0 offen offset:16 // (op 58) Store value
buffer_store_dwordx2 v[32:33], v0, s[56:59], 0 offen offset:32 // (op 58) Store value
buffer_store_dwordx2 v[34:35], v0, s[56:59], 0 offen offset:48 // (op 58) Store value



v_accvgpr_read v12, a16 // (op 1262) 
v_accvgpr_read v13, a17 // (op 1262) 
v_accvgpr_read v14, a18 // (op 1262) 
v_accvgpr_read v15, a19 // (op 1262) 
v_accvgpr_read v16, a20 // (op 1262) 
v_accvgpr_read v17, a21 // (op 1262) 
v_accvgpr_read v18, a22 // (op 1262) 
v_accvgpr_read v19, a23 // (op 1262) 
v_accvgpr_read v20, a24 // (op 1262) 
v_accvgpr_read v21, a25 // (op 1262) 
v_accvgpr_read v22, a26 // (op 1262) 
v_accvgpr_read v23, a27 // (op 1262) 
v_accvgpr_read v24, a28 // (op 1262) 
v_accvgpr_read v25, a29 // (op 1262) 
v_accvgpr_read v26, a30 // (op 1262) 
v_accvgpr_read v27, a31 // (op 1262) 
v_mul_f32 v28, v10, v12 // (op 1262) 
v_mul_f32 v29, v10, v13 // (op 1262) 
v_mul_f32 v30, v10, v14 // (op 1262) 
v_mul_f32 v31, v10, v15 // (op 1262) 
v_mul_f32 v32, v10, v16 // (op 1262) 
v_mul_f32 v33, v10, v17 // (op 1262) 
v_mul_f32 v34, v10, v18 // (op 1262) 
v_mul_f32 v35, v10, v19 // (op 1262) 
v_mul_f32 v36, v10, v20 // (op 1262) 
v_mul_f32 v37, v10, v21 // (op 1262) 
v_mul_f32 v38, v10, v22 // (op 1262) 
v_mul_f32 v39, v10, v23 // (op 1262) 
v_mul_f32 v40, v10, v24 // (op 1262) 
v_mul_f32 v41, v10, v25 // (op 1262) 
v_mul_f32 v42, v10, v26 // (op 1262) 
v_mul_f32 v43, v10, v27 // (op 1262) 

v_cvt_f16_f32 v12, v28 // (op 1264) 
v_cvt_f16_f32 v13, v29 // (op 1264) 
v_cvt_f16_f32 v14, v30 // (op 1264) 
v_cvt_f16_f32 v15, v31 // (op 1264) 
v_cvt_f16_f32 v16, v32 // (op 1264) 
v_cvt_f16_f32 v17, v33 // (op 1264) 
v_cvt_f16_f32 v18, v34 // (op 1264) 
v_cvt_f16_f32 v19, v35 // (op 1264) 
v_cvt_f16_f32 v20, v36 // (op 1264) 
v_cvt_f16_f32 v21, v37 // (op 1264) 
v_cvt_f16_f32 v22, v38 // (op 1264) 
v_cvt_f16_f32 v23, v39 // (op 1264) 
v_cvt_f16_f32 v24, v40 // (op 1264) 
v_cvt_f16_f32 v25, v41 // (op 1264) 
v_cvt_f16_f32 v26, v42 // (op 1264) 
v_cvt_f16_f32 v27, v43 // (op 1264) 


v_pack_b32_f16 v28, v12, v13 // (op 1265) 
v_pack_b32_f16 v29, v14, v15 // (op 1265) 
v_pack_b32_f16 v30, v16, v17 // (op 1265) 
v_pack_b32_f16 v31, v18, v19 // (op 1265) 
v_pack_b32_f16 v32, v20, v21 // (op 1265) 
v_pack_b32_f16 v33, v22, v23 // (op 1265) 
v_pack_b32_f16 v34, v24, v25 // (op 1265) 
v_pack_b32_f16 v35, v26, v27 // (op 1265) 

s_lshl_b32 s0, s4, 7 // (op 1265) 
v_add_u32 v0, v11, s0 // (op 1265) 


buffer_store_dwordx2 v[28:29], v0, s[56:59], 0 offen // (op 1265) Store value
buffer_store_dwordx2 v[30:31], v0, s[56:59], 0 offen offset:16 // (op 1265) Store value
buffer_store_dwordx2 v[32:33], v0, s[56:59], 0 offen offset:32 // (op 1265) Store value
buffer_store_dwordx2 v[34:35], v0, s[56:59], 0 offen offset:48 // (op 1265) Store value



v_accvgpr_read v12, a32 // (op 1276) 
v_accvgpr_read v13, a33 // (op 1276) 
v_accvgpr_read v14, a34 // (op 1276) 
v_accvgpr_read v15, a35 // (op 1276) 
v_accvgpr_read v16, a36 // (op 1276) 
v_accvgpr_read v17, a37 // (op 1276) 
v_accvgpr_read v18, a38 // (op 1276) 
v_accvgpr_read v19, a39 // (op 1276) 
v_accvgpr_read v20, a40 // (op 1276) 
v_accvgpr_read v21, a41 // (op 1276) 
v_accvgpr_read v22, a42 // (op 1276) 
v_accvgpr_read v23, a43 // (op 1276) 
v_accvgpr_read v24, a44 // (op 1276) 
v_accvgpr_read v25, a45 // (op 1276) 
v_accvgpr_read v26, a46 // (op 1276) 
v_accvgpr_read v27, a47 // (op 1276) 
v_mul_f32 v28, v10, v12 // (op 1276) 
v_mul_f32 v29, v10, v13 // (op 1276) 
v_mul_f32 v30, v10, v14 // (op 1276) 
v_mul_f32 v31, v10, v15 // (op 1276) 
v_mul_f32 v32, v10, v16 // (op 1276) 
v_mul_f32 v33, v10, v17 // (op 1276) 
v_mul_f32 v34, v10, v18 // (op 1276) 
v_mul_f32 v35, v10, v19 // (op 1276) 
v_mul_f32 v36, v10, v20 // (op 1276) 
v_mul_f32 v37, v10, v21 // (op 1276) 
v_mul_f32 v38, v10, v22 // (op 1276) 
v_mul_f32 v39, v10, v23 // (op 1276) 
v_mul_f32 v40, v10, v24 // (op 1276) 
v_mul_f32 v41, v10, v25 // (op 1276) 
v_mul_f32 v42, v10, v26 // (op 1276) 
v_mul_f32 v43, v10, v27 // (op 1276) 

v_cvt_f16_f32 v12, v28 // (op 1278) 
v_cvt_f16_f32 v13, v29 // (op 1278) 
v_cvt_f16_f32 v14, v30 // (op 1278) 
v_cvt_f16_f32 v15, v31 // (op 1278) 
v_cvt_f16_f32 v16, v32 // (op 1278) 
v_cvt_f16_f32 v17, v33 // (op 1278) 
v_cvt_f16_f32 v18, v34 // (op 1278) 
v_cvt_f16_f32 v19, v35 // (op 1278) 
v_cvt_f16_f32 v20, v36 // (op 1278) 
v_cvt_f16_f32 v21, v37 // (op 1278) 
v_cvt_f16_f32 v22, v38 // (op 1278) 
v_cvt_f16_f32 v23, v39 // (op 1278) 
v_cvt_f16_f32 v24, v40 // (op 1278) 
v_cvt_f16_f32 v25, v41 // (op 1278) 
v_cvt_f16_f32 v26, v42 // (op 1278) 
v_cvt_f16_f32 v27, v43 // (op 1278) 


v_pack_b32_f16 v28, v12, v13 // (op 1279) 
v_pack_b32_f16 v29, v14, v15 // (op 1279) 
v_pack_b32_f16 v30, v16, v17 // (op 1279) 
v_pack_b32_f16 v31, v18, v19 // (op 1279) 
v_pack_b32_f16 v32, v20, v21 // (op 1279) 
v_pack_b32_f16 v33, v22, v23 // (op 1279) 
v_pack_b32_f16 v34, v24, v25 // (op 1279) 
v_pack_b32_f16 v35, v26, v27 // (op 1279) 

s_lshl_b32 s0, s4, 8 // (op 1279) 
v_add_u32 v0, v11, s0 // (op 1279) 


buffer_store_dwordx2 v[28:29], v0, s[56:59], 0 offen // (op 1279) Store value
buffer_store_dwordx2 v[30:31], v0, s[56:59], 0 offen offset:16 // (op 1279) Store value
buffer_store_dwordx2 v[32:33], v0, s[56:59], 0 offen offset:32 // (op 1279) Store value
buffer_store_dwordx2 v[34:35], v0, s[56:59], 0 offen offset:48 // (op 1279) Store value



v_accvgpr_read v12, a48 // (op 1290) 
v_accvgpr_read v13, a49 // (op 1290) 
v_accvgpr_read v14, a50 // (op 1290) 
v_accvgpr_read v15, a51 // (op 1290) 
v_accvgpr_read v16, a52 // (op 1290) 
v_accvgpr_read v17, a53 // (op 1290) 
v_accvgpr_read v18, a54 // (op 1290) 
v_accvgpr_read v19, a55 // (op 1290) 
v_accvgpr_read v20, a56 // (op 1290) 
v_accvgpr_read v21, a57 // (op 1290) 
v_accvgpr_read v22, a58 // (op 1290) 
v_accvgpr_read v23, a59 // (op 1290) 
v_accvgpr_read v24, a60 // (op 1290) 
v_accvgpr_read v25, a61 // (op 1290) 
v_accvgpr_read v26, a62 // (op 1290) 
v_accvgpr_read v27, a63 // (op 1290) 
v_mul_f32 v28, v10, v12 // (op 1290) 
v_mul_f32 v29, v10, v13 // (op 1290) 
v_mul_f32 v30, v10, v14 // (op 1290) 
v_mul_f32 v31, v10, v15 // (op 1290) 
v_mul_f32 v32, v10, v16 // (op 1290) 
v_mul_f32 v33, v10, v17 // (op 1290) 
v_mul_f32 v34, v10, v18 // (op 1290) 
v_mul_f32 v35, v10, v19 // (op 1290) 
v_mul_f32 v36, v10, v20 // (op 1290) 
v_mul_f32 v37, v10, v21 // (op 1290) 
v_mul_f32 v38, v10, v22 // (op 1290) 
v_mul_f32 v39, v10, v23 // (op 1290) 
v_mul_f32 v40, v10, v24 // (op 1290) 
v_mul_f32 v41, v10, v25 // (op 1290) 
v_mul_f32 v42, v10, v26 // (op 1290) 
v_mul_f32 v43, v10, v27 // (op 1290) 

v_cvt_f16_f32 v12, v28 // (op 1292) 
v_cvt_f16_f32 v13, v29 // (op 1292) 
v_cvt_f16_f32 v14, v30 // (op 1292) 
v_cvt_f16_f32 v15, v31 // (op 1292) 
v_cvt_f16_f32 v16, v32 // (op 1292) 
v_cvt_f16_f32 v17, v33 // (op 1292) 
v_cvt_f16_f32 v18, v34 // (op 1292) 
v_cvt_f16_f32 v19, v35 // (op 1292) 
v_cvt_f16_f32 v20, v36 // (op 1292) 
v_cvt_f16_f32 v21, v37 // (op 1292) 
v_cvt_f16_f32 v22, v38 // (op 1292) 
v_cvt_f16_f32 v23, v39 // (op 1292) 
v_cvt_f16_f32 v24, v40 // (op 1292) 
v_cvt_f16_f32 v25, v41 // (op 1292) 
v_cvt_f16_f32 v26, v42 // (op 1292) 
v_cvt_f16_f32 v27, v43 // (op 1292) 


v_pack_b32_f16 v28, v12, v13 // (op 1293) 
v_pack_b32_f16 v29, v14, v15 // (op 1293) 
v_pack_b32_f16 v30, v16, v17 // (op 1293) 
v_pack_b32_f16 v31, v18, v19 // (op 1293) 
v_pack_b32_f16 v32, v20, v21 // (op 1293) 
v_pack_b32_f16 v33, v22, v23 // (op 1293) 
v_pack_b32_f16 v34, v24, v25 // (op 1293) 
v_pack_b32_f16 v35, v26, v27 // (op 1293) 

s_lshl_b32 s0, s4, 7 // (op 1293) 
s_mul_i32 s1, 3, s0 // (op 1293) 

v_add_u32 v0, v11, s1 // (op 1293) 


buffer_store_dwordx2 v[28:29], v0, s[56:59], 0 offen // (op 1293) Store value
buffer_store_dwordx2 v[30:31], v0, s[56:59], 0 offen offset:16 // (op 1293) Store value
buffer_store_dwordx2 v[32:33], v0, s[56:59], 0 offen offset:32 // (op 1293) Store value
buffer_store_dwordx2 v[34:35], v0, s[56:59], 0 offen offset:48 // (op 1293) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ConditionalBottom_0_beta_157 // (op 157) Condition: Done, jump to RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ConditionalBottom_0_beta_157
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ConditionalFalse_0_beta_157:


v_mov_b32 v0, v8 // (op 170) reg expression

buffer_load_dwordx2 v[12:13], v0, s[52:55], 0 offen // (op 170) Load value
buffer_load_dwordx2 v[14:15], v0, s[52:55], 0 offen offset:16 // (op 170) Load value
buffer_load_dwordx2 v[16:17], v0, s[52:55], 0 offen offset:32 // (op 170) Load value
v_accvgpr_read v20, a0 // (op 166) 
v_accvgpr_read v21, a1 // (op 166) 
v_accvgpr_read v22, a2 // (op 166) 
v_accvgpr_read v23, a3 // (op 166) 
v_accvgpr_read v24, a4 // (op 166) 
v_accvgpr_read v25, a5 // (op 166) 
v_accvgpr_read v26, a6 // (op 166) 
v_accvgpr_read v27, a7 // (op 166) 
v_accvgpr_read v28, a8 // (op 166) 
v_accvgpr_read v29, a9 // (op 166) 
v_accvgpr_read v30, a10 // (op 166) 
v_accvgpr_read v31, a11 // (op 166) 
v_accvgpr_read v32, a12 // (op 166) 
v_accvgpr_read v33, a13 // (op 166) 
v_accvgpr_read v34, a14 // (op 166) 
v_accvgpr_read v35, a15 // (op 166) 
v_mul_f32 v36, v10, v20 // (op 166) 
v_mul_f32 v37, v10, v21 // (op 166) 
v_mul_f32 v38, v10, v22 // (op 166) 
v_mul_f32 v39, v10, v23 // (op 166) 
v_mul_f32 v40, v10, v24 // (op 166) 
v_mul_f32 v41, v10, v25 // (op 166) 
v_mul_f32 v42, v10, v26 // (op 166) 
v_mul_f32 v43, v10, v27 // (op 166) 
v_mul_f32 v44, v10, v28 // (op 166) 
v_mul_f32 v45, v10, v29 // (op 166) 
v_mul_f32 v46, v10, v30 // (op 166) 
v_mul_f32 v47, v10, v31 // (op 166) 
v_mul_f32 v48, v10, v32 // (op 166) 
v_mul_f32 v49, v10, v33 // (op 166) 
v_mul_f32 v50, v10, v34 // (op 166) 
v_mul_f32 v51, v10, v35 // (op 166) 

buffer_load_dwordx2 v[18:19], v0, s[52:55], 0 offen offset:48 // (op 170) Load value

s_waitcnt vmcnt(3)
v_cvt_f32_f16 v20, v12 // (op 171) 
v_lshrrev_b32 v21, 16, v12 // (op 171) 
v_cvt_f32_f16 v21, v21 // (op 171) 
v_mul_f32 v52, v7, v20 // (op 171) 
v_mul_f32 v53, v7, v21 // (op 171) 
v_cvt_f32_f16 v20, v13 // (op 171) 
v_lshrrev_b32 v21, 16, v13 // (op 171) 
v_cvt_f32_f16 v21, v21 // (op 171) 
v_mul_f32 v54, v7, v20 // (op 171) 
v_mul_f32 v55, v7, v21 // (op 171) 
s_waitcnt vmcnt(2)
v_cvt_f32_f16 v20, v14 // (op 171) 
v_lshrrev_b32 v21, 16, v14 // (op 171) 
v_cvt_f32_f16 v21, v21 // (op 171) 
v_mul_f32 v56, v7, v20 // (op 171) 
v_mul_f32 v57, v7, v21 // (op 171) 
v_cvt_f32_f16 v20, v15 // (op 171) 
v_lshrrev_b32 v21, 16, v15 // (op 171) 
v_cvt_f32_f16 v21, v21 // (op 171) 
v_mul_f32 v58, v7, v20 // (op 171) 
v_mul_f32 v59, v7, v21 // (op 171) 
s_waitcnt vmcnt(1)
v_cvt_f32_f16 v20, v16 // (op 171) 
v_lshrrev_b32 v21, 16, v16 // (op 171) 
v_cvt_f32_f16 v21, v21 // (op 171) 
v_mul_f32 v60, v7, v20 // (op 171) 
v_mul_f32 v61, v7, v21 // (op 171) 
v_cvt_f32_f16 v20, v17 // (op 171) 
v_lshrrev_b32 v21, 16, v17 // (op 171) 
v_cvt_f32_f16 v21, v21 // (op 171) 
v_mul_f32 v62, v7, v20 // (op 171) 
v_mul_f32 v63, v7, v21 // (op 171) 
s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v18 // (op 171) 
v_lshrrev_b32 v21, 16, v18 // (op 171) 
v_cvt_f32_f16 v21, v21 // (op 171) 
v_mul_f32 v64, v7, v20 // (op 171) 
v_mul_f32 v65, v7, v21 // (op 171) 
v_cvt_f32_f16 v20, v19 // (op 171) 
v_lshrrev_b32 v21, 16, v19 // (op 171) 
v_cvt_f32_f16 v21, v21 // (op 171) 
v_mul_f32 v66, v7, v20 // (op 171) 
v_mul_f32 v67, v7, v21 // (op 171) 


v_add_f32 v12, v52, v36 // (op 167) 
v_add_f32 v13, v53, v37 // (op 167) 
v_add_f32 v14, v54, v38 // (op 167) 
v_add_f32 v15, v55, v39 // (op 167) 
v_add_f32 v16, v56, v40 // (op 167) 
v_add_f32 v17, v57, v41 // (op 167) 
v_add_f32 v18, v58, v42 // (op 167) 
v_add_f32 v19, v59, v43 // (op 167) 
v_add_f32 v20, v60, v44 // (op 167) 
v_add_f32 v21, v61, v45 // (op 167) 
v_add_f32 v22, v62, v46 // (op 167) 
v_add_f32 v23, v63, v47 // (op 167) 
v_add_f32 v24, v64, v48 // (op 167) 
v_add_f32 v25, v65, v49 // (op 167) 
v_add_f32 v26, v66, v50 // (op 167) 
v_add_f32 v27, v67, v51 // (op 167) 


v_cvt_f16_f32 v28, v12 // (op 168) 
v_cvt_f16_f32 v29, v13 // (op 168) 
v_cvt_f16_f32 v30, v14 // (op 168) 
v_cvt_f16_f32 v31, v15 // (op 168) 
v_cvt_f16_f32 v32, v16 // (op 168) 
v_cvt_f16_f32 v33, v17 // (op 168) 
v_cvt_f16_f32 v34, v18 // (op 168) 
v_cvt_f16_f32 v35, v19 // (op 168) 
v_cvt_f16_f32 v36, v20 // (op 168) 
v_cvt_f16_f32 v37, v21 // (op 168) 
v_cvt_f16_f32 v38, v22 // (op 168) 
v_cvt_f16_f32 v39, v23 // (op 168) 
v_cvt_f16_f32 v40, v24 // (op 168) 
v_cvt_f16_f32 v41, v25 // (op 168) 
v_cvt_f16_f32 v42, v26 // (op 168) 
v_cvt_f16_f32 v43, v27 // (op 168) 


v_pack_b32_f16 v12, v28, v29 // (op 169) 
v_pack_b32_f16 v13, v30, v31 // (op 169) 
v_pack_b32_f16 v14, v32, v33 // (op 169) 
v_pack_b32_f16 v15, v34, v35 // (op 169) 
v_pack_b32_f16 v16, v36, v37 // (op 169) 
v_pack_b32_f16 v17, v38, v39 // (op 169) 
v_pack_b32_f16 v18, v40, v41 // (op 169) 
v_pack_b32_f16 v19, v42, v43 // (op 169) 

v_mov_b32 v0, v11 // (op 169) reg expression

buffer_store_dwordx2 v[12:13], v0, s[56:59], 0 offen // (op 169) Store value
buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen offset:16 // (op 169) Store value
buffer_store_dwordx2 v[16:17], v0, s[56:59], 0 offen offset:32 // (op 169) Store value
buffer_store_dwordx2 v[18:19], v0, s[56:59], 0 offen offset:48 // (op 169) Store value





s_lshl_b32 s0, s2, 7 // (op 1327) 
v_add_u32 v0, v8, s0 // (op 1327) 


buffer_load_dwordx2 v[12:13], v0, s[52:55], 0 offen // (op 1327) Load value
buffer_load_dwordx2 v[14:15], v0, s[52:55], 0 offen offset:16 // (op 1327) Load value
v_accvgpr_read v20, a16 // (op 1323) 
v_accvgpr_read v21, a17 // (op 1323) 
v_accvgpr_read v22, a18 // (op 1323) 
v_accvgpr_read v23, a19 // (op 1323) 
v_accvgpr_read v24, a20 // (op 1323) 
v_accvgpr_read v25, a21 // (op 1323) 
v_accvgpr_read v26, a22 // (op 1323) 
v_accvgpr_read v27, a23 // (op 1323) 
v_accvgpr_read v28, a24 // (op 1323) 
v_accvgpr_read v29, a25 // (op 1323) 
v_accvgpr_read v30, a26 // (op 1323) 
v_accvgpr_read v31, a27 // (op 1323) 
v_accvgpr_read v32, a28 // (op 1323) 
v_accvgpr_read v33, a29 // (op 1323) 
v_accvgpr_read v34, a30 // (op 1323) 
v_accvgpr_read v35, a31 // (op 1323) 
v_mul_f32 v36, v10, v20 // (op 1323) 
v_mul_f32 v37, v10, v21 // (op 1323) 
v_mul_f32 v38, v10, v22 // (op 1323) 
v_mul_f32 v39, v10, v23 // (op 1323) 
v_mul_f32 v40, v10, v24 // (op 1323) 
v_mul_f32 v41, v10, v25 // (op 1323) 
v_mul_f32 v42, v10, v26 // (op 1323) 
v_mul_f32 v43, v10, v27 // (op 1323) 
v_mul_f32 v44, v10, v28 // (op 1323) 
v_mul_f32 v45, v10, v29 // (op 1323) 
v_mul_f32 v46, v10, v30 // (op 1323) 
v_mul_f32 v47, v10, v31 // (op 1323) 
v_mul_f32 v48, v10, v32 // (op 1323) 
v_mul_f32 v49, v10, v33 // (op 1323) 
v_mul_f32 v50, v10, v34 // (op 1323) 
v_mul_f32 v51, v10, v35 // (op 1323) 

buffer_load_dwordx2 v[16:17], v0, s[52:55], 0 offen offset:32 // (op 1327) Load value
buffer_load_dwordx2 v[18:19], v0, s[52:55], 0 offen offset:48 // (op 1327) Load value

s_waitcnt vmcnt(3)
v_cvt_f32_f16 v20, v12 // (op 1328) 
v_lshrrev_b32 v21, 16, v12 // (op 1328) 
v_cvt_f32_f16 v21, v21 // (op 1328) 
v_mul_f32 v52, v7, v20 // (op 1328) 
v_mul_f32 v53, v7, v21 // (op 1328) 
v_cvt_f32_f16 v20, v13 // (op 1328) 
v_lshrrev_b32 v21, 16, v13 // (op 1328) 
v_cvt_f32_f16 v21, v21 // (op 1328) 
v_mul_f32 v54, v7, v20 // (op 1328) 
v_mul_f32 v55, v7, v21 // (op 1328) 
s_waitcnt vmcnt(2)
v_cvt_f32_f16 v20, v14 // (op 1328) 
v_lshrrev_b32 v21, 16, v14 // (op 1328) 
v_cvt_f32_f16 v21, v21 // (op 1328) 
v_mul_f32 v56, v7, v20 // (op 1328) 
v_mul_f32 v57, v7, v21 // (op 1328) 
v_cvt_f32_f16 v20, v15 // (op 1328) 
v_lshrrev_b32 v21, 16, v15 // (op 1328) 
v_cvt_f32_f16 v21, v21 // (op 1328) 
v_mul_f32 v58, v7, v20 // (op 1328) 
v_mul_f32 v59, v7, v21 // (op 1328) 
s_waitcnt vmcnt(1)
v_cvt_f32_f16 v20, v16 // (op 1328) 
v_lshrrev_b32 v21, 16, v16 // (op 1328) 
v_cvt_f32_f16 v21, v21 // (op 1328) 
v_mul_f32 v60, v7, v20 // (op 1328) 
v_mul_f32 v61, v7, v21 // (op 1328) 
v_cvt_f32_f16 v20, v17 // (op 1328) 
v_lshrrev_b32 v21, 16, v17 // (op 1328) 
v_cvt_f32_f16 v21, v21 // (op 1328) 
v_mul_f32 v62, v7, v20 // (op 1328) 
v_mul_f32 v63, v7, v21 // (op 1328) 
s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v18 // (op 1328) 
v_lshrrev_b32 v21, 16, v18 // (op 1328) 
v_cvt_f32_f16 v21, v21 // (op 1328) 
v_mul_f32 v64, v7, v20 // (op 1328) 
v_mul_f32 v65, v7, v21 // (op 1328) 
v_cvt_f32_f16 v20, v19 // (op 1328) 
v_lshrrev_b32 v21, 16, v19 // (op 1328) 
v_cvt_f32_f16 v21, v21 // (op 1328) 
v_mul_f32 v66, v7, v20 // (op 1328) 
v_mul_f32 v67, v7, v21 // (op 1328) 


v_add_f32 v12, v52, v36 // (op 1324) 
v_add_f32 v13, v53, v37 // (op 1324) 
v_add_f32 v14, v54, v38 // (op 1324) 
v_add_f32 v15, v55, v39 // (op 1324) 
v_add_f32 v16, v56, v40 // (op 1324) 
v_add_f32 v17, v57, v41 // (op 1324) 
v_add_f32 v18, v58, v42 // (op 1324) 
v_add_f32 v19, v59, v43 // (op 1324) 
v_add_f32 v20, v60, v44 // (op 1324) 
v_add_f32 v21, v61, v45 // (op 1324) 
v_add_f32 v22, v62, v46 // (op 1324) 
v_add_f32 v23, v63, v47 // (op 1324) 
v_add_f32 v24, v64, v48 // (op 1324) 
v_add_f32 v25, v65, v49 // (op 1324) 
v_add_f32 v26, v66, v50 // (op 1324) 
v_add_f32 v27, v67, v51 // (op 1324) 


v_cvt_f16_f32 v28, v12 // (op 1325) 
v_cvt_f16_f32 v29, v13 // (op 1325) 
v_cvt_f16_f32 v30, v14 // (op 1325) 
v_cvt_f16_f32 v31, v15 // (op 1325) 
v_cvt_f16_f32 v32, v16 // (op 1325) 
v_cvt_f16_f32 v33, v17 // (op 1325) 
v_cvt_f16_f32 v34, v18 // (op 1325) 
v_cvt_f16_f32 v35, v19 // (op 1325) 
v_cvt_f16_f32 v36, v20 // (op 1325) 
v_cvt_f16_f32 v37, v21 // (op 1325) 
v_cvt_f16_f32 v38, v22 // (op 1325) 
v_cvt_f16_f32 v39, v23 // (op 1325) 
v_cvt_f16_f32 v40, v24 // (op 1325) 
v_cvt_f16_f32 v41, v25 // (op 1325) 
v_cvt_f16_f32 v42, v26 // (op 1325) 
v_cvt_f16_f32 v43, v27 // (op 1325) 


v_pack_b32_f16 v12, v28, v29 // (op 1326) 
v_pack_b32_f16 v13, v30, v31 // (op 1326) 
v_pack_b32_f16 v14, v32, v33 // (op 1326) 
v_pack_b32_f16 v15, v34, v35 // (op 1326) 
v_pack_b32_f16 v16, v36, v37 // (op 1326) 
v_pack_b32_f16 v17, v38, v39 // (op 1326) 
v_pack_b32_f16 v18, v40, v41 // (op 1326) 
v_pack_b32_f16 v19, v42, v43 // (op 1326) 

s_lshl_b32 s0, s4, 7 // (op 1326) 
v_add_u32 v0, v11, s0 // (op 1326) 


buffer_store_dwordx2 v[12:13], v0, s[56:59], 0 offen // (op 1326) Store value
buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen offset:16 // (op 1326) Store value
buffer_store_dwordx2 v[16:17], v0, s[56:59], 0 offen offset:32 // (op 1326) Store value
buffer_store_dwordx2 v[18:19], v0, s[56:59], 0 offen offset:48 // (op 1326) Store value





s_lshl_b32 s0, s2, 8 // (op 1341) 
v_add_u32 v0, v8, s0 // (op 1341) 


buffer_load_dwordx2 v[12:13], v0, s[52:55], 0 offen // (op 1341) Load value
buffer_load_dwordx2 v[14:15], v0, s[52:55], 0 offen offset:16 // (op 1341) Load value
v_accvgpr_read v20, a32 // (op 1337) 
v_accvgpr_read v21, a33 // (op 1337) 
v_accvgpr_read v22, a34 // (op 1337) 
v_accvgpr_read v23, a35 // (op 1337) 
v_accvgpr_read v24, a36 // (op 1337) 
v_accvgpr_read v25, a37 // (op 1337) 
v_accvgpr_read v26, a38 // (op 1337) 
v_accvgpr_read v27, a39 // (op 1337) 
v_accvgpr_read v28, a40 // (op 1337) 
v_accvgpr_read v29, a41 // (op 1337) 
v_accvgpr_read v30, a42 // (op 1337) 
v_accvgpr_read v31, a43 // (op 1337) 
v_accvgpr_read v32, a44 // (op 1337) 
v_accvgpr_read v33, a45 // (op 1337) 
v_accvgpr_read v34, a46 // (op 1337) 
v_accvgpr_read v35, a47 // (op 1337) 
v_mul_f32 v36, v10, v20 // (op 1337) 
v_mul_f32 v37, v10, v21 // (op 1337) 
v_mul_f32 v38, v10, v22 // (op 1337) 
v_mul_f32 v39, v10, v23 // (op 1337) 
v_mul_f32 v40, v10, v24 // (op 1337) 
v_mul_f32 v41, v10, v25 // (op 1337) 
v_mul_f32 v42, v10, v26 // (op 1337) 
v_mul_f32 v43, v10, v27 // (op 1337) 
v_mul_f32 v44, v10, v28 // (op 1337) 
v_mul_f32 v45, v10, v29 // (op 1337) 
v_mul_f32 v46, v10, v30 // (op 1337) 
v_mul_f32 v47, v10, v31 // (op 1337) 
v_mul_f32 v48, v10, v32 // (op 1337) 
v_mul_f32 v49, v10, v33 // (op 1337) 
v_mul_f32 v50, v10, v34 // (op 1337) 
v_mul_f32 v51, v10, v35 // (op 1337) 

buffer_load_dwordx2 v[16:17], v0, s[52:55], 0 offen offset:32 // (op 1341) Load value
buffer_load_dwordx2 v[18:19], v0, s[52:55], 0 offen offset:48 // (op 1341) Load value

s_waitcnt vmcnt(3)
v_cvt_f32_f16 v20, v12 // (op 1342) 
v_lshrrev_b32 v21, 16, v12 // (op 1342) 
v_cvt_f32_f16 v21, v21 // (op 1342) 
v_mul_f32 v52, v7, v20 // (op 1342) 
v_mul_f32 v53, v7, v21 // (op 1342) 
v_cvt_f32_f16 v20, v13 // (op 1342) 
v_lshrrev_b32 v21, 16, v13 // (op 1342) 
v_cvt_f32_f16 v21, v21 // (op 1342) 
v_mul_f32 v54, v7, v20 // (op 1342) 
v_mul_f32 v55, v7, v21 // (op 1342) 
s_waitcnt vmcnt(2)
v_cvt_f32_f16 v20, v14 // (op 1342) 
v_lshrrev_b32 v21, 16, v14 // (op 1342) 
v_cvt_f32_f16 v21, v21 // (op 1342) 
v_mul_f32 v56, v7, v20 // (op 1342) 
v_mul_f32 v57, v7, v21 // (op 1342) 
v_cvt_f32_f16 v20, v15 // (op 1342) 
v_lshrrev_b32 v21, 16, v15 // (op 1342) 
v_cvt_f32_f16 v21, v21 // (op 1342) 
v_mul_f32 v58, v7, v20 // (op 1342) 
v_mul_f32 v59, v7, v21 // (op 1342) 
s_waitcnt vmcnt(1)
v_cvt_f32_f16 v20, v16 // (op 1342) 
v_lshrrev_b32 v21, 16, v16 // (op 1342) 
v_cvt_f32_f16 v21, v21 // (op 1342) 
v_mul_f32 v60, v7, v20 // (op 1342) 
v_mul_f32 v61, v7, v21 // (op 1342) 
v_cvt_f32_f16 v20, v17 // (op 1342) 
v_lshrrev_b32 v21, 16, v17 // (op 1342) 
v_cvt_f32_f16 v21, v21 // (op 1342) 
v_mul_f32 v62, v7, v20 // (op 1342) 
v_mul_f32 v63, v7, v21 // (op 1342) 
s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v18 // (op 1342) 
v_lshrrev_b32 v21, 16, v18 // (op 1342) 
v_cvt_f32_f16 v21, v21 // (op 1342) 
v_mul_f32 v64, v7, v20 // (op 1342) 
v_mul_f32 v65, v7, v21 // (op 1342) 
v_cvt_f32_f16 v20, v19 // (op 1342) 
v_lshrrev_b32 v21, 16, v19 // (op 1342) 
v_cvt_f32_f16 v21, v21 // (op 1342) 
v_mul_f32 v66, v7, v20 // (op 1342) 
v_mul_f32 v67, v7, v21 // (op 1342) 


v_add_f32 v12, v52, v36 // (op 1338) 
v_add_f32 v13, v53, v37 // (op 1338) 
v_add_f32 v14, v54, v38 // (op 1338) 
v_add_f32 v15, v55, v39 // (op 1338) 
v_add_f32 v16, v56, v40 // (op 1338) 
v_add_f32 v17, v57, v41 // (op 1338) 
v_add_f32 v18, v58, v42 // (op 1338) 
v_add_f32 v19, v59, v43 // (op 1338) 
v_add_f32 v20, v60, v44 // (op 1338) 
v_add_f32 v21, v61, v45 // (op 1338) 
v_add_f32 v22, v62, v46 // (op 1338) 
v_add_f32 v23, v63, v47 // (op 1338) 
v_add_f32 v24, v64, v48 // (op 1338) 
v_add_f32 v25, v65, v49 // (op 1338) 
v_add_f32 v26, v66, v50 // (op 1338) 
v_add_f32 v27, v67, v51 // (op 1338) 


v_cvt_f16_f32 v28, v12 // (op 1339) 
v_cvt_f16_f32 v29, v13 // (op 1339) 
v_cvt_f16_f32 v30, v14 // (op 1339) 
v_cvt_f16_f32 v31, v15 // (op 1339) 
v_cvt_f16_f32 v32, v16 // (op 1339) 
v_cvt_f16_f32 v33, v17 // (op 1339) 
v_cvt_f16_f32 v34, v18 // (op 1339) 
v_cvt_f16_f32 v35, v19 // (op 1339) 
v_cvt_f16_f32 v36, v20 // (op 1339) 
v_cvt_f16_f32 v37, v21 // (op 1339) 
v_cvt_f16_f32 v38, v22 // (op 1339) 
v_cvt_f16_f32 v39, v23 // (op 1339) 
v_cvt_f16_f32 v40, v24 // (op 1339) 
v_cvt_f16_f32 v41, v25 // (op 1339) 
v_cvt_f16_f32 v42, v26 // (op 1339) 
v_cvt_f16_f32 v43, v27 // (op 1339) 


v_pack_b32_f16 v12, v28, v29 // (op 1340) 
v_pack_b32_f16 v13, v30, v31 // (op 1340) 
v_pack_b32_f16 v14, v32, v33 // (op 1340) 
v_pack_b32_f16 v15, v34, v35 // (op 1340) 
v_pack_b32_f16 v16, v36, v37 // (op 1340) 
v_pack_b32_f16 v17, v38, v39 // (op 1340) 
v_pack_b32_f16 v18, v40, v41 // (op 1340) 
v_pack_b32_f16 v19, v42, v43 // (op 1340) 

s_lshl_b32 s0, s4, 8 // (op 1340) 
v_add_u32 v0, v11, s0 // (op 1340) 


buffer_store_dwordx2 v[12:13], v0, s[56:59], 0 offen // (op 1340) Store value
buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen offset:16 // (op 1340) Store value
buffer_store_dwordx2 v[16:17], v0, s[56:59], 0 offen offset:32 // (op 1340) Store value
buffer_store_dwordx2 v[18:19], v0, s[56:59], 0 offen offset:48 // (op 1340) Store value





s_lshl_b32 s0, s2, 7 // (op 1355) 
s_mul_i32 s1, 3, s0 // (op 1355) 

v_add_u32 v0, v8, s1 // (op 1355) 


buffer_load_dwordx2 v[12:13], v0, s[52:55], 0 offen // (op 1355) Load value
buffer_load_dwordx2 v[14:15], v0, s[52:55], 0 offen offset:16 // (op 1355) Load value
v_accvgpr_read v20, a48 // (op 1351) 
v_accvgpr_read v21, a49 // (op 1351) 
v_accvgpr_read v22, a50 // (op 1351) 
v_accvgpr_read v23, a51 // (op 1351) 
v_accvgpr_read v24, a52 // (op 1351) 
v_accvgpr_read v25, a53 // (op 1351) 
v_accvgpr_read v26, a54 // (op 1351) 
v_accvgpr_read v27, a55 // (op 1351) 
v_accvgpr_read v28, a56 // (op 1351) 
v_accvgpr_read v29, a57 // (op 1351) 
v_accvgpr_read v30, a58 // (op 1351) 
v_accvgpr_read v31, a59 // (op 1351) 
v_accvgpr_read v32, a60 // (op 1351) 
v_accvgpr_read v33, a61 // (op 1351) 
v_accvgpr_read v34, a62 // (op 1351) 
v_accvgpr_read v35, a63 // (op 1351) 
v_mul_f32 v36, v10, v20 // (op 1351) 
v_mul_f32 v37, v10, v21 // (op 1351) 
v_mul_f32 v38, v10, v22 // (op 1351) 
v_mul_f32 v39, v10, v23 // (op 1351) 
v_mul_f32 v40, v10, v24 // (op 1351) 
v_mul_f32 v41, v10, v25 // (op 1351) 
v_mul_f32 v42, v10, v26 // (op 1351) 
v_mul_f32 v43, v10, v27 // (op 1351) 
v_mul_f32 v44, v10, v28 // (op 1351) 
v_mul_f32 v45, v10, v29 // (op 1351) 
v_mul_f32 v46, v10, v30 // (op 1351) 
v_mul_f32 v47, v10, v31 // (op 1351) 
v_mul_f32 v48, v10, v32 // (op 1351) 
v_mul_f32 v49, v10, v33 // (op 1351) 
v_mul_f32 v50, v10, v34 // (op 1351) 
v_mul_f32 v51, v10, v35 // (op 1351) 

buffer_load_dwordx2 v[16:17], v0, s[52:55], 0 offen offset:32 // (op 1355) Load value
buffer_load_dwordx2 v[18:19], v0, s[52:55], 0 offen offset:48 // (op 1355) Load value

s_waitcnt vmcnt(3)
v_cvt_f32_f16 v20, v12 // (op 1356) 
v_lshrrev_b32 v21, 16, v12 // (op 1356) 
v_cvt_f32_f16 v21, v21 // (op 1356) 
v_mul_f32 v52, v7, v20 // (op 1356) 
v_mul_f32 v53, v7, v21 // (op 1356) 
v_cvt_f32_f16 v20, v13 // (op 1356) 
v_lshrrev_b32 v21, 16, v13 // (op 1356) 
v_cvt_f32_f16 v21, v21 // (op 1356) 
v_mul_f32 v54, v7, v20 // (op 1356) 
v_mul_f32 v55, v7, v21 // (op 1356) 
s_waitcnt vmcnt(2)
v_cvt_f32_f16 v20, v14 // (op 1356) 
v_lshrrev_b32 v21, 16, v14 // (op 1356) 
v_cvt_f32_f16 v21, v21 // (op 1356) 
v_mul_f32 v56, v7, v20 // (op 1356) 
v_mul_f32 v57, v7, v21 // (op 1356) 
v_cvt_f32_f16 v20, v15 // (op 1356) 
v_lshrrev_b32 v21, 16, v15 // (op 1356) 
v_cvt_f32_f16 v21, v21 // (op 1356) 
v_mul_f32 v58, v7, v20 // (op 1356) 
v_mul_f32 v59, v7, v21 // (op 1356) 
s_waitcnt vmcnt(1)
v_cvt_f32_f16 v20, v16 // (op 1356) 
v_lshrrev_b32 v21, 16, v16 // (op 1356) 
v_cvt_f32_f16 v21, v21 // (op 1356) 
v_mul_f32 v60, v7, v20 // (op 1356) 
v_mul_f32 v61, v7, v21 // (op 1356) 
v_cvt_f32_f16 v20, v17 // (op 1356) 
v_lshrrev_b32 v21, 16, v17 // (op 1356) 
v_cvt_f32_f16 v21, v21 // (op 1356) 
v_mul_f32 v62, v7, v20 // (op 1356) 
v_mul_f32 v63, v7, v21 // (op 1356) 
s_waitcnt vmcnt(0)
v_cvt_f32_f16 v20, v18 // (op 1356) 
v_lshrrev_b32 v21, 16, v18 // (op 1356) 
v_cvt_f32_f16 v21, v21 // (op 1356) 
v_mul_f32 v64, v7, v20 // (op 1356) 
v_mul_f32 v65, v7, v21 // (op 1356) 
v_cvt_f32_f16 v20, v19 // (op 1356) 
v_lshrrev_b32 v21, 16, v19 // (op 1356) 
v_cvt_f32_f16 v21, v21 // (op 1356) 
v_mul_f32 v66, v7, v20 // (op 1356) 
v_mul_f32 v67, v7, v21 // (op 1356) 


v_add_f32 v12, v52, v36 // (op 1352) 
v_add_f32 v13, v53, v37 // (op 1352) 
v_add_f32 v14, v54, v38 // (op 1352) 
v_add_f32 v15, v55, v39 // (op 1352) 
v_add_f32 v16, v56, v40 // (op 1352) 
v_add_f32 v17, v57, v41 // (op 1352) 
v_add_f32 v18, v58, v42 // (op 1352) 
v_add_f32 v19, v59, v43 // (op 1352) 
v_add_f32 v20, v60, v44 // (op 1352) 
v_add_f32 v21, v61, v45 // (op 1352) 
v_add_f32 v22, v62, v46 // (op 1352) 
v_add_f32 v23, v63, v47 // (op 1352) 
v_add_f32 v24, v64, v48 // (op 1352) 
v_add_f32 v25, v65, v49 // (op 1352) 
v_add_f32 v26, v66, v50 // (op 1352) 
v_add_f32 v27, v67, v51 // (op 1352) 


v_cvt_f16_f32 v28, v12 // (op 1353) 
v_cvt_f16_f32 v29, v13 // (op 1353) 
v_cvt_f16_f32 v30, v14 // (op 1353) 
v_cvt_f16_f32 v31, v15 // (op 1353) 
v_cvt_f16_f32 v32, v16 // (op 1353) 
v_cvt_f16_f32 v33, v17 // (op 1353) 
v_cvt_f16_f32 v34, v18 // (op 1353) 
v_cvt_f16_f32 v35, v19 // (op 1353) 
v_cvt_f16_f32 v36, v20 // (op 1353) 
v_cvt_f16_f32 v37, v21 // (op 1353) 
v_cvt_f16_f32 v38, v22 // (op 1353) 
v_cvt_f16_f32 v39, v23 // (op 1353) 
v_cvt_f16_f32 v40, v24 // (op 1353) 
v_cvt_f16_f32 v41, v25 // (op 1353) 
v_cvt_f16_f32 v42, v26 // (op 1353) 
v_cvt_f16_f32 v43, v27 // (op 1353) 


v_pack_b32_f16 v12, v28, v29 // (op 1354) 
v_pack_b32_f16 v13, v30, v31 // (op 1354) 
v_pack_b32_f16 v14, v32, v33 // (op 1354) 
v_pack_b32_f16 v15, v34, v35 // (op 1354) 
v_pack_b32_f16 v16, v36, v37 // (op 1354) 
v_pack_b32_f16 v17, v38, v39 // (op 1354) 
v_pack_b32_f16 v18, v40, v41 // (op 1354) 
v_pack_b32_f16 v19, v42, v43 // (op 1354) 

s_lshl_b32 s0, s4, 7 // (op 1354) 
s_mul_i32 s1, 3, s0 // (op 1354) 

v_add_u32 v0, v11, s1 // (op 1354) 


buffer_store_dwordx2 v[12:13], v0, s[56:59], 0 offen // (op 1354) Store value
buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen offset:16 // (op 1354) Store value
buffer_store_dwordx2 v[16:17], v0, s[56:59], 0 offen offset:32 // (op 1354) Store value
buffer_store_dwordx2 v[18:19], v0, s[56:59], 0 offen offset:48 // (op 1354) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ConditionalBottom_0_beta_157:












s_endpgm  // End of RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_
.LRR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__end:
.size RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_, .LRR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__end-RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 216
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 64000 // lds bytes
  .amdhsa_accum_offset 152
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
    macrotile: [64, 256, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
  MatrixInstruction: [32, 32, 64, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [1, 4]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 64000
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 74
    .vgpr_count: 152
    .agpr_count: 64
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {is-null: true}, {is-null: true}]
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
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 28
        .expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}}
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
        .expression: {type: MagicShiftAndSign, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_24, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 160, size: 4}, rhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_27
        .size: 4
        .offset: 156
        .expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}}
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

