// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM_,@function
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 2405) 
s_mul_hi_i32 s1, s16, s10 // (op 2660) 
v_lshrrev_b32 v0, 7, v1 // (op 2466) 
v_and_b32 v3, 15, v1 // (op 2466) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 2466) 


v_and_b32 v0, 48, v1 // (op 2466) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 2466) 


v_lshrrev_b32 v0, 6, v1 // (op 2523) 
v_and_b32 v4, 1, v0 // (op 2523) 

v_and_b32 v0, 15, v1 // (op 2523) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 2523) 


v_and_b32 v0, 48, v1 // (op 2523) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 2523) 


v_lshrrev_b32 v0, 7, v1 // (op 2581) 
v_and_b32 v5, 15, v1 // (op 2581) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 2581) 


v_and_b32 v0, 63, v1 // (op 2581) 
v_lshrrev_b32 v5, 4, v0 // (op 2581) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 2581) 


v_lshrrev_b32 v5, 6, v1 // (op 2639) 
v_and_b32 v6, 1, v5 // (op 2639) 

v_and_b32 v5, 15, v1 // (op 2639) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 2639) 


v_and_b32 v5, 63, v1 // (op 2639) 
v_lshrrev_b32 v6, 4, v5 // (op 2639) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 2639) 


v_lshrrev_b32 v6, 7, v1 // (op 2706) 
v_and_b32 v7, 15, v1 // (op 2706) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 2706) 


v_and_b32 v6, 48, v1 // (op 2706) 
v_lshl_add_u32 v7, v8, 7, v6 // (op 2706) 


v_lshrrev_b32 v6, 6, v1 // (op 2750) 
v_and_b32 v8, 1, v6 // (op 2750) 

v_and_b32 v6, 15, v1 // (op 2750) 
v_lshl_add_u32 v9, v8, 4, v6 // (op 2750) 


v_and_b32 v6, 48, v1 // (op 2750) 
v_lshl_add_u32 v8, v9, 7, v6 // (op 2750) 


v_lshrrev_b32 v6, 7, v1 // (op 2779) 
v_and_b32 v9, 15, v1 // (op 2779) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 2779) 


v_and_b32 v6, 63, v1 // (op 2779) 
v_lshrrev_b32 v9, 4, v6 // (op 2779) 

v_lshl_add_u32 v6, v10, 2, v9 // (op 2779) 


v_lshrrev_b32 v9, 6, v1 // (op 2808) 
v_and_b32 v10, 1, v9 // (op 2808) 

v_and_b32 v9, 15, v1 // (op 2808) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 2808) 


v_and_b32 v9, 63, v1 // (op 2808) 
v_lshrrev_b32 v10, 4, v9 // (op 2808) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 2808) 


v_lshrrev_b32 v10, 7, v1 // (op 2405) 
s_add_i32 s18, s0, s16 // (op 2405) 

v_and_b32 v11, 63, v1 // (op 2405) 
s_ashr_i32 s0, s18, 31 // (op 2405) 
v_lshrrev_b32 v12, 4, v11 // (op 2405) 
s_cmp_eq_i32 s10, 0 // (op 2405) 
s_mov_b32 s19, scc // (op 2405) 
v_lshlrev_b32 v13, 2, v12 // (op 2405) 

s_cmp_eq_i32 s19, 0 // (op 2405) 
s_cselect_b32 s49, 0, -1 // (op 2405) 

v_lshl_add_u32 v12, v10, 4, v13 // (op 2405) 


s_and_b32 s19, s13, 31 // (op 2405) 
v_lshrrev_b32 v10, 2, v11 // (op 2405) 

v_and_b32 v11, 3, v10 // (op 2405) 

v_lshrrev_b32 v10, 7, v1 // (op 2660) 
v_and_b32 v13, 63, v1 // (op 2660) 
s_lshl_b32 s50, 1, s19 // (op 2405) 
v_and_b32 v14, 3, v1 // (op 2405) 
s_add_i32 s51, s50, s49 // (op 2405) 


s_and_b32 s49, s0, s51 // (op 2405) 


s_add_i32 s0, s1, s16 // (op 2660) 

s_ashr_i32 s1, s0, 31 // (op 2660) 
v_lshl_add_u32 v15, v11, 2, v14 // (op 2405) 


s_add_i32 s50, s18, s49 // (op 2405) 


v_lshrrev_b32 v11, 6, v1 // (op 2405) 
s_ashr_i32 s18, s50, s19 // (op 2405) 


v_and_b32 v14, 1, v11 // (op 2405) 

s_lshl_b32 s19, s13, 24 // (op 2405) 
v_lshl_add_u32 v11, v14, 4, v15 // (op 2405) 


s_ashr_i32 s49, s19, 31 // (op 2405) 

s_xor_b32 s19, s18, s49 // (op 2405) 

s_sub_i32 s18, s19, s49 // (op 2405) 


s_mul_hi_i32 s19, s18, s14 // (op 2405) 
s_add_i32 s49, s19, s18 // (op 2405) 

s_ashr_i32 s19, s49, 31 // (op 2405) 
s_cmp_eq_i32 s14, 0 // (op 2405) 
s_mov_b32 s50, scc // (op 2405) 
s_cmp_eq_i32 s50, 0 // (op 2405) 
s_cselect_b32 s51, 0, -1 // (op 2405) 

s_and_b32 s50, s15, 31 // (op 2405) 
v_lshrrev_b32 v14, 4, v13 // (op 2660) 
s_cmp_eq_i32 s10, 0 // (op 2660) 
s_mov_b32 s52, scc // (op 2660) 
v_lshlrev_b32 v15, 2, v14 // (op 2660) 

s_cmp_eq_i32 s52, 0 // (op 2660) 
s_cselect_b32 s53, 0, -1 // (op 2660) 

s_lshl_b32 s52, 1, s50 // (op 2405) 
v_lshl_add_u32 v14, v10, 4, v15 // (op 2660) 


v_lshrrev_b32 v10, 2, v13 // (op 2660) 

s_add_i32 s54, s52, s51 // (op 2405) 


s_and_b32 s51, s19, s54 // (op 2405) 


s_add_i32 s19, s49, s51 // (op 2405) 


s_ashr_i32 s49, s19, s50 // (op 2405) 


s_lshl_b32 s19, s15, 24 // (op 2405) 
s_ashr_i32 s50, s19, 31 // (op 2405) 

s_xor_b32 s19, s49, s50 // (op 2405) 

s_sub_i32 s49, s19, s50 // (op 2405) 


s_mul_i32 s19, s18, s11 // (op 2405) 
s_sub_u32 s50, s16, s19 // (op 2405) 

s_mul_i32 s19, s49, s11 // (op 2405) 
s_add_u32 s19, s19, s50 // (op 2405) 

s_mul_hi_i32 s50, s16, s9 // (op 2405) 
s_add_i32 s51, s50, s16 // (op 2405) 

s_ashr_i32 s50, s51, 31 // (op 2405) 
s_cmp_eq_i32 s9, 0 // (op 2405) 
s_mov_b32 s52, scc // (op 2405) 
s_cmp_eq_i32 s52, 0 // (op 2405) 
s_cselect_b32 s54, 0, -1 // (op 2405) 

s_and_b32 s52, s13, 31 // (op 2660) 
v_and_b32 v13, 3, v10 // (op 2660) 

s_lshl_b32 s55, 1, s52 // (op 2660) 
v_and_b32 v10, 3, v1 // (op 2660) 
s_add_i32 s56, s55, s53 // (op 2660) 


s_and_b32 s53, s1, s56 // (op 2660) 


v_lshl_add_u32 v15, v13, 2, v10 // (op 2660) 


s_add_i32 s1, s0, s53 // (op 2660) 


v_lshrrev_b32 v10, 6, v1 // (op 2660) 
s_ashr_i32 s0, s1, s52 // (op 2660) 


v_and_b32 v13, 1, v10 // (op 2660) 

s_lshl_b32 s1, s13, 24 // (op 2660) 
v_lshl_add_u32 v10, v13, 4, v15 // (op 2660) 


s_ashr_i32 s52, s1, 31 // (op 2660) 

s_xor_b32 s1, s0, s52 // (op 2660) 

s_sub_i32 s0, s1, s52 // (op 2660) 


s_mul_hi_i32 s1, s0, s14 // (op 2660) 
s_add_i32 s52, s1, s0 // (op 2660) 

s_ashr_i32 s1, s52, 31 // (op 2660) 
s_cmp_eq_i32 s14, 0 // (op 2660) 
s_mov_b32 s53, scc // (op 2660) 
s_cmp_eq_i32 s53, 0 // (op 2660) 
s_cselect_b32 s55, 0, -1 // (op 2660) 

s_and_b32 s53, s15, 31 // (op 2660) 
s_lshl_b32 s56, 1, s53 // (op 2660) 
s_add_i32 s57, s56, s55 // (op 2660) 


s_and_b32 s55, s1, s57 // (op 2660) 


s_add_i32 s1, s52, s55 // (op 2660) 


s_ashr_i32 s52, s1, s53 // (op 2660) 


s_lshl_b32 s1, s15, 24 // (op 2660) 
s_ashr_i32 s53, s1, 31 // (op 2660) 

s_xor_b32 s1, s52, s53 // (op 2660) 

s_sub_i32 s52, s1, s53 // (op 2660) 


s_mul_i32 s1, s0, s11 // (op 2660) 
s_sub_u32 s53, s16, s1 // (op 2660) 

s_mul_i32 s1, s52, s11 // (op 2660) 
s_add_u32 s1, s1, s53 // (op 2660) 

s_mul_hi_i32 s53, s16, s9 // (op 2660) 
s_add_i32 s55, s53, s16 // (op 2660) 

s_ashr_i32 s53, s55, 31 // (op 2660) 
s_cmp_eq_i32 s9, 0 // (op 2660) 
s_mov_b32 s56, scc // (op 2660) 
s_cmp_eq_i32 s56, 0 // (op 2660) 
s_cselect_b32 s57, 0, -1 // (op 2660) 


s_waitcnt lgkmcnt(0)
s_and_b32 s56, s45, 31 // (op 2405) 
s_lshl_b32 s58, 1, s56 // (op 2405) 
s_add_i32 s59, s58, s54 // (op 2405) 


s_and_b32 s54, s50, s59 // (op 2405) 


s_add_i32 s50, s51, s54 // (op 2405) 


s_ashr_i32 s51, s50, s56 // (op 2405) 


s_lshl_b32 s50, s45, 24 // (op 2405) 
s_ashr_i32 s54, s50, 31 // (op 2405) 

s_xor_b32 s50, s51, s54 // (op 2405) 

s_sub_i32 s51, s50, s54 // (op 2405) 


s_and_b32 s50, s45, 31 // (op 2660) 
s_lshl_b32 s54, 1, s50 // (op 2660) 
s_add_i32 s56, s54, s57 // (op 2660) 


s_and_b32 s54, s53, s56 // (op 2660) 


s_add_i32 s53, s55, s54 // (op 2660) 


s_ashr_i32 s54, s53, s50 // (op 2660) 


s_lshl_b32 s50, s45, 24 // (op 2660) 
s_ashr_i32 s53, s50, 31 // (op 2660) 

s_xor_b32 s50, s54, s53 // (op 2660) 

s_sub_i32 s54, s50, s53 // (op 2660) 


s_add_u32 s56, s6, 64 // (op 2405) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 2405) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 2405) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 2405) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 2405) 

s_mul_i32 s50, s11, s56 // (op 2405) 
s_mul_i32 s53, s8, s50 // (op 2405) 

s_mul_i32 s50, s51, s53 // (op 2405) 

s_sub_u32 s53, s16, s50 // (op 2405) 

s_mov_b32 s50, s53 // (op 2405) 
s_mul_hi_i32 s55, s50, s12 // (op 2405) 
s_add_i32 s58, s55, s50 // (op 2405) 


s_ashr_i32 s50, s58, 31 // (op 2405) 
s_cmp_eq_i32 s12, 0 // (op 2405) 
s_mov_b32 s55, scc // (op 2405) 
s_cmp_eq_i32 s55, 0 // (op 2405) 
s_cselect_b32 s59, 0, -1 // (op 2405) 

s_and_b32 s55, s44, 31 // (op 2405) 
s_lshl_b32 s60, 1, s55 // (op 2405) 
s_add_i32 s61, s60, s59 // (op 2405) 


s_and_b32 s59, s50, s61 // (op 2405) 


s_add_i32 s50, s58, s59 // (op 2405) 


s_ashr_i32 s58, s50, s55 // (op 2405) 


s_lshl_b32 s50, s44, 24 // (op 2405) 
s_ashr_i32 s55, s50, 31 // (op 2405) 

s_xor_b32 s50, s58, s55 // (op 2405) 

s_sub_i32 s58, s50, s55 // (op 2405) 


s_mul_i32 s50, s46, s11 // (op 2405) 
s_add_u32 s60, s20, 64 // (op 2405) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 2405) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 2405) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 2405) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 6 // (op 2405) 

s_sub_i32 s55, s60, s50 // (op 2405) 

s_mul_i32 s59, s58, s55 // (op 2405) 

s_sub_u32 s55, s53, s59 // (op 2405) 


s_add_u32 s53, s55, s50 // (op 2405) 


s_cmp_eq_i32 s51, 0 // (op 2405) 
s_mov_b32 s50, scc // (op 2405) 

s_cmp_eq_i32 s50, 0 // (op 2405) 
s_cselect_b32 s51, s53, s19 // (op 2405) 


s_mul_i32 s19, s49, s56 // (op 2405) 


v_lshl_add_u32 v13, s51, 6, v12 // (op 2405) 


s_sub_i32 s49, s18, s19 // (op 2405) 


s_cmp_eq_i32 s50, 0 // (op 2405) 
s_cselect_b32 s51, s58, s49 // (op 2405) 



v_lshl_add_u32 v12, s51, 6, v11 // (op 2405) 


v_mul_lo_u32 v11, v12, s2 // (op 2405) 

v_add_lshl_u32 v12, v13, v11, 2 // (op 2405) 


s_add_u32 s18, s6, 64 // (op 2660) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 2660) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 2660) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 2660) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 6 // (op 2660) 

s_mul_i32 s53, s11, s18 // (op 2660) 
s_mul_i32 s49, s8, s53 // (op 2660) 

s_mul_i32 s53, s54, s49 // (op 2660) 

s_sub_u32 s49, s16, s53 // (op 2660) 

s_mov_b32 s53, s49 // (op 2660) 
s_mul_hi_i32 s50, s53, s12 // (op 2660) 
s_add_i32 s51, s50, s53 // (op 2660) 


s_ashr_i32 s50, s51, 31 // (op 2660) 
s_cmp_eq_i32 s12, 0 // (op 2660) 
s_mov_b32 s53, scc // (op 2660) 
s_cmp_eq_i32 s53, 0 // (op 2660) 
s_cselect_b32 s55, 0, -1 // (op 2660) 

s_and_b32 s53, s44, 31 // (op 2660) 
s_lshl_b32 s56, 1, s53 // (op 2660) 
s_add_i32 s57, s56, s55 // (op 2660) 


s_and_b32 s55, s50, s57 // (op 2660) 


s_add_i32 s50, s51, s55 // (op 2660) 


s_ashr_i32 s51, s50, s53 // (op 2660) 


s_lshl_b32 s50, s44, 24 // (op 2660) 
s_ashr_i32 s53, s50, 31 // (op 2660) 

s_xor_b32 s50, s51, s53 // (op 2660) 

s_sub_i32 s51, s50, s53 // (op 2660) 


s_mul_i32 s50, s46, s11 // (op 2660) 
s_add_u32 s56, s20, 64 // (op 2660) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 2660) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 2660) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 2660) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 2660) 

s_sub_i32 s53, s56, s50 // (op 2660) 

s_mul_i32 s55, s51, s53 // (op 2660) 

s_sub_u32 s53, s49, s55 // (op 2660) 


s_add_u32 s49, s53, s50 // (op 2660) 


s_cmp_eq_i32 s54, 0 // (op 2660) 
s_mov_b32 s50, scc // (op 2660) 

s_cmp_eq_i32 s50, 0 // (op 2660) 
s_cselect_b32 s53, s49, s1 // (op 2660) 


s_mul_i32 s1, s52, s18 // (op 2660) 


v_lshl_add_u32 v11, s53, 6, v14 // (op 2660) 


s_sub_i32 s49, s0, s1 // (op 2660) 


s_cmp_eq_i32 s50, 0 // (op 2660) 
s_cselect_b32 s0, s51, s49 // (op 2660) 



v_lshl_add_u32 v13, s0, 6, v10 // (op 2660) 


v_mul_lo_u32 v10, v13, s4 // (op 2660) 

v_add_lshl_u32 v13, v11, v10, 2 // (op 2660) 


s_mul_i32 s1, s2, s7 // (op 2407) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 2407) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 2407) least significant
s_mul_hi_u32 s19, s2, s6 // (op 2407) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2407) most significant: sum
s_add_u32 s1, s1, s19 // (op 2407) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 2407) 

s_mul_i32 s1, s4, s7 // (op 2662) most significant: low of low * high
s_mul_i32 s49, s5, s6 // (op 2662) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 2662) least significant
s_mul_hi_u32 s50, s4, s6 // (op 2662) most significant: high of low * low
s_add_u32 s1, s1, s49 // (op 2662) most significant: sum
s_add_u32 s1, s1, s50 // (op 2662) most significant: sum


s_lshl_b64 s[50:51], s[0:1], 2 // (op 2662) 

s_mov_b32 s52, s22 // (op 2407) reg expression
s_mov_b32 s53, s23 // (op 2407) reg expression
s_mov_b32 s54, s18 // (op 2407) reg expression
s_mov_b32 s55, 131072 // (op 2407) call()

s_mov_b32 s56, s24 // (op 2662) reg expression

s_mov_b32 s57, s25 // (op 2662) reg expression
s_mov_b32 s58, s50 // (op 2662) reg expression
s_mov_b32 s59, 131072 // (op 2662) call()


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
v_accvgpr_write a8, 0.00000 // (op 556) call()
v_accvgpr_write a9, 0.00000 // (op 556) call()
v_accvgpr_write a10, 0.00000 // (op 556) call()
v_accvgpr_write a11, 0.00000 // (op 556) call()
v_accvgpr_write a12, 0.00000 // (op 711) call()
v_accvgpr_write a13, 0.00000 // (op 711) call()
v_accvgpr_write a14, 0.00000 // (op 711) call()
v_accvgpr_write a15, 0.00000 // (op 711) call()
s_mul_hi_i32 s0, s16, s10 // (op 2167) 
v_lshrrev_b32 v14, 3, v1 // (op 2167) 
s_add_i32 s1, s0, s16 // (op 2167) 

v_and_b32 v15, -8, v1 // (op 2167) 
s_ashr_i32 s0, s1, 31 // (op 2167) 
v_sub_u32 v16, v1, v15 // (op 2167) 

s_cmp_eq_i32 s10, 0 // (op 2167) 
s_mov_b32 s18, scc // (op 2167) 
v_lshlrev_b32 v15, 4, v16 // (op 2167) 

s_cmp_eq_i32 s18, 0 // (op 2167) 
s_cselect_b32 s19, 0, -1 // (op 2167) 

v_mov_b32 v17, 0 // (op 2167) convert
v_mov_b32 v16, v15 // (op 2167) convert

s_and_b32 s18, s13, 31 // (op 2167) 
s_lshl_b32 s22, 1, s18 // (op 2167) 
s_add_i32 s23, s22, s19 // (op 2167) 


s_and_b32 s19, s0, s23 // (op 2167) 


s_add_i32 s0, s1, s19 // (op 2167) 


s_ashr_i32 s1, s0, s18 // (op 2167) 


s_lshl_b32 s0, s13, 24 // (op 2167) 
s_ashr_i32 s18, s0, 31 // (op 2167) 

s_xor_b32 s0, s1, s18 // (op 2167) 

s_sub_i32 s1, s0, s18 // (op 2167) 


s_mul_hi_i32 s0, s1, s14 // (op 2167) 
s_add_i32 s18, s0, s1 // (op 2167) 

s_ashr_i32 s0, s18, 31 // (op 2167) 
s_cmp_eq_i32 s14, 0 // (op 2167) 
s_mov_b32 s19, scc // (op 2167) 
s_cmp_eq_i32 s19, 0 // (op 2167) 
s_cselect_b32 s22, 0, -1 // (op 2167) 

s_and_b32 s19, s15, 31 // (op 2167) 
s_lshl_b32 s23, 1, s19 // (op 2167) 
s_add_i32 s24, s23, s22 // (op 2167) 


s_and_b32 s25, s0, s24 // (op 2167) 


s_add_i32 s0, s18, s25 // (op 2167) 


s_ashr_i32 s18, s0, s19 // (op 2167) 


s_lshl_b32 s0, s15, 24 // (op 2167) 
s_ashr_i32 s19, s0, 31 // (op 2167) 

s_xor_b32 s0, s18, s19 // (op 2167) 

s_sub_i32 s18, s0, s19 // (op 2167) 


s_mul_i32 s0, s1, s11 // (op 2167) 

s_sub_u32 s1, s16, s0 // (op 2167) 

s_mul_i32 s0, s18, s11 // (op 2167) 
s_add_u32 s0, s0, s1 // (op 2167) 


s_mul_hi_i32 s1, s16, s9 // (op 2167) 
s_add_i32 s18, s1, s16 // (op 2167) 

s_ashr_i32 s1, s18, 31 // (op 2167) 
s_cmp_eq_i32 s9, 0 // (op 2167) 
s_mov_b32 s19, scc // (op 2167) 
s_cmp_eq_i32 s19, 0 // (op 2167) 
s_cselect_b32 s22, 0, -1 // (op 2167) 

s_and_b32 s19, s45, 31 // (op 2167) 
s_lshl_b32 s23, 1, s19 // (op 2167) 
s_add_i32 s24, s23, s22 // (op 2167) 


s_and_b32 s25, s1, s24 // (op 2167) 


s_add_i32 s1, s18, s25 // (op 2167) 


s_ashr_i32 s18, s1, s19 // (op 2167) 


s_lshl_b32 s1, s45, 24 // (op 2167) 
s_ashr_i32 s19, s1, 31 // (op 2167) 

s_xor_b32 s1, s18, s19 // (op 2167) 

s_sub_i32 s18, s1, s19 // (op 2167) 


s_add_u32 s22, s6, 64 // (op 2167) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2167) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2167) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2167) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 2167) 

s_mul_i32 s1, s11, s22 // (op 2167) 

s_mul_i32 s19, s8, s1 // (op 2167) 

s_mul_i32 s1, s18, s19 // (op 2167) 

s_sub_u32 s19, s16, s1 // (op 2167) 

s_mov_b32 s1, s19 // (op 2167) 
s_mul_hi_i32 s22, s1, s12 // (op 2167) 
s_add_i32 s23, s22, s1 // (op 2167) 


s_ashr_i32 s1, s23, 31 // (op 2167) 
s_cmp_eq_i32 s12, 0 // (op 2167) 
s_mov_b32 s22, scc // (op 2167) 
s_cmp_eq_i32 s22, 0 // (op 2167) 
s_cselect_b32 s24, 0, -1 // (op 2167) 

s_and_b32 s22, s44, 31 // (op 2167) 
s_lshl_b32 s25, 1, s22 // (op 2167) 
s_add_i32 s47, s25, s24 // (op 2167) 


s_and_b32 s24, s1, s47 // (op 2167) 


s_add_i32 s1, s23, s24 // (op 2167) 


s_ashr_i32 s23, s1, s22 // (op 2167) 


s_lshl_b32 s1, s44, 24 // (op 2167) 
s_ashr_i32 s22, s1, 31 // (op 2167) 

s_xor_b32 s1, s23, s22 // (op 2167) 

s_sub_i32 s23, s1, s22 // (op 2167) 


s_mul_i32 s1, s46, s11 // (op 2167) 
s_add_u32 s24, s20, 64 // (op 2167) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2167) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 2167) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 2167) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 2167) 

s_sub_i32 s22, s24, s1 // (op 2167) 

s_mul_i32 s24, s23, s22 // (op 2167) 


s_sub_u32 s25, s19, s24 // (op 2167) 


s_add_u32 s19, s25, s1 // (op 2167) 


s_cmp_eq_i32 s18, 0 // (op 2167) 
s_mov_b32 s1, scc // (op 2167) 

s_cmp_eq_i32 s1, 0 // (op 2167) 
s_cselect_b32 s18, s19, s0 // (op 2167) 



v_lshl_add_u32 v15, s18, 6, v14 // (op 2167) 


v_mul_lo_u32 v19, v15, s27 // (op 2167) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 2167) most significant: high of low * low
v_mul_lo_u32 v18, v15, s26 // (op 2167) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 2167) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 2167) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 2167) most significant half



s_mul_i32 s1, s26, s21 // (op 2169) most significant: low of low * high
s_mul_i32 s18, s27, s20 // (op 2169) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 2169) least significant
s_mul_hi_u32 s19, s26, s20 // (op 2169) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 2169) most significant: sum
s_add_u32 s1, s1, s19 // (op 2169) most significant: sum


s_mov_b32 s48, s28 // (op 2169) reg expression
s_mov_b32 s49, s29 // (op 2169) reg expression
s_mov_b32 s50, s0 // (op 2169) reg expression
s_mov_b32 s51, 131072 // (op 2169) call()

s_mul_hi_i32 s47, s16, s10 // (op 2186) 
v_lshrrev_b32 v14, 3, v1 // (op 2186) 
s_add_i32 s0, s47, s16 // (op 2186) 

v_and_b32 v15, -8, v1 // (op 2186) 
s_ashr_i32 s1, s0, 31 // (op 2186) 
v_sub_u32 v16, v1, v15 // (op 2186) 

s_cmp_eq_i32 s10, 0 // (op 2186) 
s_mov_b32 s47, scc // (op 2186) 
v_lshlrev_b32 v15, 4, v16 // (op 2186) 

s_cmp_eq_i32 s47, 0 // (op 2186) 
s_cselect_b32 s18, 0, -1 // (op 2186) 

v_mov_b32 v17, 0 // (op 2186) convert
v_mov_b32 v16, v15 // (op 2186) convert

s_and_b32 s19, s13, 31 // (op 2186) 
s_lshl_b32 s47, 1, s19 // (op 2186) 
s_add_i32 s22, s47, s18 // (op 2186) 


s_and_b32 s18, s1, s22 // (op 2186) 


s_add_i32 s1, s0, s18 // (op 2186) 


s_ashr_i32 s0, s1, s19 // (op 2186) 


s_lshl_b32 s1, s13, 24 // (op 2186) 
s_ashr_i32 s47, s1, 31 // (op 2186) 

s_xor_b32 s1, s0, s47 // (op 2186) 

s_sub_i32 s0, s1, s47 // (op 2186) 


s_mul_hi_i32 s1, s0, s14 // (op 2186) 
s_add_i32 s47, s1, s0 // (op 2186) 

s_ashr_i32 s1, s47, 31 // (op 2186) 
s_cmp_eq_i32 s14, 0 // (op 2186) 
s_mov_b32 s18, scc // (op 2186) 
s_cmp_eq_i32 s18, 0 // (op 2186) 
s_cselect_b32 s19, 0, -1 // (op 2186) 

s_and_b32 s18, s15, 31 // (op 2186) 
s_lshl_b32 s22, 1, s18 // (op 2186) 
s_add_i32 s23, s22, s19 // (op 2186) 


s_and_b32 s19, s1, s23 // (op 2186) 


s_add_i32 s1, s47, s19 // (op 2186) 


s_ashr_i32 s19, s1, s18 // (op 2186) 


s_lshl_b32 s1, s15, 24 // (op 2186) 
s_ashr_i32 s18, s1, 31 // (op 2186) 

s_xor_b32 s1, s19, s18 // (op 2186) 

s_sub_i32 s19, s1, s18 // (op 2186) 


s_mul_i32 s1, s0, s11 // (op 2186) 

s_sub_u32 s0, s16, s1 // (op 2186) 

s_mul_i32 s1, s19, s11 // (op 2186) 
s_add_u32 s1, s1, s0 // (op 2186) 


s_mul_hi_i32 s0, s16, s9 // (op 2186) 
s_add_i32 s47, s0, s16 // (op 2186) 

s_ashr_i32 s0, s47, 31 // (op 2186) 
s_cmp_eq_i32 s9, 0 // (op 2186) 
s_mov_b32 s18, scc // (op 2186) 
s_cmp_eq_i32 s18, 0 // (op 2186) 
s_cselect_b32 s19, 0, -1 // (op 2186) 

s_and_b32 s18, s45, 31 // (op 2186) 
s_lshl_b32 s22, 1, s18 // (op 2186) 
s_add_i32 s23, s22, s19 // (op 2186) 


s_and_b32 s19, s0, s23 // (op 2186) 


s_add_i32 s0, s47, s19 // (op 2186) 


s_ashr_i32 s19, s0, s18 // (op 2186) 


s_lshl_b32 s0, s45, 24 // (op 2186) 
s_ashr_i32 s18, s0, 31 // (op 2186) 

s_xor_b32 s0, s19, s18 // (op 2186) 

s_sub_i32 s19, s0, s18 // (op 2186) 


s_add_u32 s22, s6, 64 // (op 2186) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2186) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2186) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2186) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 2186) 

s_mul_i32 s0, s11, s22 // (op 2186) 

s_mul_i32 s18, s8, s0 // (op 2186) 

s_mul_i32 s0, s19, s18 // (op 2186) 

s_sub_u32 s18, s16, s0 // (op 2186) 

s_mov_b32 s0, s18 // (op 2186) 
s_mul_hi_i32 s47, s0, s12 // (op 2186) 
s_add_i32 s22, s47, s0 // (op 2186) 


s_ashr_i32 s0, s22, 31 // (op 2186) 
s_cmp_eq_i32 s12, 0 // (op 2186) 
s_mov_b32 s47, scc // (op 2186) 
s_cmp_eq_i32 s47, 0 // (op 2186) 
s_cselect_b32 s23, 0, -1 // (op 2186) 

s_and_b32 s47, s44, 31 // (op 2186) 
s_lshl_b32 s24, 1, s47 // (op 2186) 
s_add_i32 s25, s24, s23 // (op 2186) 


s_and_b32 s23, s0, s25 // (op 2186) 


s_add_i32 s0, s22, s23 // (op 2186) 


s_ashr_i32 s22, s0, s47 // (op 2186) 


s_lshl_b32 s0, s44, 24 // (op 2186) 
s_ashr_i32 s47, s0, 31 // (op 2186) 

s_xor_b32 s0, s22, s47 // (op 2186) 

s_sub_i32 s22, s0, s47 // (op 2186) 


s_mul_i32 s0, s46, s11 // (op 2186) 
s_add_u32 s24, s20, 64 // (op 2186) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2186) End of Int64 add, unlocking SCC
s_sub_u32 s60, s24, 1 // (op 2186) least significant half; sets scc
s_subb_u32 s61, s25, 0 // (op 2186) most significant half; uses scc

s_ashr_i64 s[24:25], s[60:61], 6 // (op 2186) 

s_sub_i32 s23, s24, s0 // (op 2186) 

s_mul_i32 s47, s22, s23 // (op 2186) 


s_sub_u32 s22, s18, s47 // (op 2186) 


s_add_u32 s18, s22, s0 // (op 2186) 


s_cmp_eq_i32 s19, 0 // (op 2186) 
s_mov_b32 s0, scc // (op 2186) 

s_cmp_eq_i32 s0, 0 // (op 2186) 
s_cselect_b32 s19, s18, s1 // (op 2186) 



v_lshl_add_u32 v15, s19, 6, v14 // (op 2186) 


v_mul_lo_u32 v21, v15, s27 // (op 2186) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 2186) most significant: high of low * low
v_mul_lo_u32 v20, v15, s26 // (op 2186) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 2186) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v16 // (op 2186) least significant half
v_addc_co_u32 v21, s[0:1], v21, v17, s[0:1] // (op 2186) most significant half



s_mul_i32 s1, s26, s21 // (op 2188) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 2188) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 2188) least significant
s_mul_hi_u32 s18, s26, s20 // (op 2188) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2188) most significant: sum
s_add_u32 s1, s1, s18 // (op 2188) most significant: sum


s_mov_b32 s48, s28 // (op 2188) reg expression
s_mov_b32 s49, s29 // (op 2188) reg expression
s_mov_b32 s50, s0 // (op 2188) reg expression
s_mov_b32 s51, 131072 // (op 2188) call()

s_mul_hi_i32 s47, s16, s10 // (op 2231) 
v_lshrrev_b32 v14, 3, v1 // (op 2231) 
s_add_i32 s0, s47, s16 // (op 2231) 

v_and_b32 v15, -8, v1 // (op 2231) 
s_ashr_i32 s1, s0, 31 // (op 2231) 
v_sub_u32 v16, v1, v15 // (op 2231) 

s_cmp_eq_i32 s10, 0 // (op 2231) 
s_mov_b32 s47, scc // (op 2231) 
v_lshlrev_b32 v15, 4, v16 // (op 2231) 

s_cmp_eq_i32 s47, 0 // (op 2231) 
s_cselect_b32 s18, 0, -1 // (op 2231) 

v_mov_b32 v17, 0 // (op 2231) convert
v_mov_b32 v16, v15 // (op 2231) convert

s_and_b32 s19, s13, 31 // (op 2231) 
s_lshl_b32 s47, 1, s19 // (op 2231) 
s_add_i32 s22, s47, s18 // (op 2231) 


s_and_b32 s18, s1, s22 // (op 2231) 


s_add_i32 s1, s0, s18 // (op 2231) 


s_ashr_i32 s0, s1, s19 // (op 2231) 


s_lshl_b32 s1, s13, 24 // (op 2231) 
s_ashr_i32 s47, s1, 31 // (op 2231) 

s_xor_b32 s1, s0, s47 // (op 2231) 

s_sub_i32 s0, s1, s47 // (op 2231) 


s_mul_hi_i32 s1, s0, s14 // (op 2231) 
s_add_i32 s47, s1, s0 // (op 2231) 

s_ashr_i32 s1, s47, 31 // (op 2231) 
s_cmp_eq_i32 s14, 0 // (op 2231) 
s_mov_b32 s18, scc // (op 2231) 
s_cmp_eq_i32 s18, 0 // (op 2231) 
s_cselect_b32 s19, 0, -1 // (op 2231) 

s_and_b32 s18, s15, 31 // (op 2231) 
s_lshl_b32 s22, 1, s18 // (op 2231) 
s_add_i32 s23, s22, s19 // (op 2231) 


s_and_b32 s19, s1, s23 // (op 2231) 


s_add_i32 s1, s47, s19 // (op 2231) 


s_ashr_i32 s19, s1, s18 // (op 2231) 


s_lshl_b32 s1, s15, 24 // (op 2231) 
s_ashr_i32 s18, s1, 31 // (op 2231) 

s_xor_b32 s1, s19, s18 // (op 2231) 

s_sub_i32 s19, s1, s18 // (op 2231) 


s_add_u32 s22, s6, 64 // (op 2231) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2231) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2231) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2231) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 2231) 

s_mul_i32 s1, s19, s22 // (op 2231) 

s_sub_i32 s47, s0, s1 // (op 2231) 


s_mul_hi_i32 s0, s16, s9 // (op 2231) 
s_add_i32 s1, s0, s16 // (op 2231) 

s_ashr_i32 s0, s1, 31 // (op 2231) 
s_cmp_eq_i32 s9, 0 // (op 2231) 
s_mov_b32 s18, scc // (op 2231) 
s_cmp_eq_i32 s18, 0 // (op 2231) 
s_cselect_b32 s19, 0, -1 // (op 2231) 

s_and_b32 s18, s45, 31 // (op 2231) 
s_lshl_b32 s24, 1, s18 // (op 2231) 
s_add_i32 s25, s24, s19 // (op 2231) 


s_and_b32 s19, s0, s25 // (op 2231) 


s_add_i32 s0, s1, s19 // (op 2231) 


s_ashr_i32 s1, s0, s18 // (op 2231) 


s_lshl_b32 s0, s45, 24 // (op 2231) 
s_ashr_i32 s18, s0, 31 // (op 2231) 

s_xor_b32 s0, s1, s18 // (op 2231) 

s_sub_i32 s1, s0, s18 // (op 2231) 


s_mul_i32 s0, s11, s22 // (op 2231) 

s_mul_i32 s18, s8, s0 // (op 2231) 

s_mul_i32 s0, s1, s18 // (op 2231) 

s_sub_u32 s18, s16, s0 // (op 2231) 

s_mul_hi_i32 s0, s18, s12 // (op 2231) 
s_add_i32 s19, s0, s18 // (op 2231) 


s_ashr_i32 s0, s19, 31 // (op 2231) 
s_cmp_eq_i32 s12, 0 // (op 2231) 
s_mov_b32 s18, scc // (op 2231) 
s_cmp_eq_i32 s18, 0 // (op 2231) 
s_cselect_b32 s22, 0, -1 // (op 2231) 

s_and_b32 s18, s44, 31 // (op 2231) 
s_lshl_b32 s23, 1, s18 // (op 2231) 
s_add_i32 s24, s23, s22 // (op 2231) 


s_and_b32 s25, s0, s24 // (op 2231) 


s_add_i32 s0, s19, s25 // (op 2231) 


s_ashr_i32 s19, s0, s18 // (op 2231) 


s_lshl_b32 s0, s44, 24 // (op 2231) 
s_ashr_i32 s18, s0, 31 // (op 2231) 

s_xor_b32 s0, s19, s18 // (op 2231) 

s_sub_i32 s19, s0, s18 // (op 2231) 


s_cmp_eq_i32 s1, 0 // (op 2231) 
s_mov_b32 s0, scc // (op 2231) 

s_cmp_eq_i32 s0, 0 // (op 2231) 
s_cselect_b32 s1, s19, s47 // (op 2231) 



v_lshl_add_u32 v15, s1, 6, v14 // (op 2231) 


v_mul_lo_u32 v23, v15, s31 // (op 2231) most significant: low of low * high
v_mul_hi_u32 v14, v15, s30 // (op 2231) most significant: high of low * low
v_mul_lo_u32 v22, v15, s30 // (op 2231) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 2231) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v22 // (op 2231) least significant half
v_addc_co_u32 v15, s[0:1], v17, v23, s[0:1] // (op 2231) most significant half



s_mul_i32 s1, s30, s7 // (op 2233) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 2233) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 2233) least significant
s_mul_hi_u32 s18, s30, s6 // (op 2233) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2233) most significant: sum
s_add_u32 s1, s1, s18 // (op 2233) most significant: sum


s_mov_b32 s60, s32 // (op 2233) reg expression
s_mov_b32 s61, s33 // (op 2233) reg expression
s_mov_b32 s62, s0 // (op 2233) reg expression
s_mov_b32 s63, 131072 // (op 2233) call()

s_mul_hi_i32 s47, s16, s10 // (op 2250) 
v_lshrrev_b32 v16, 3, v1 // (op 2250) 
s_add_i32 s0, s47, s16 // (op 2250) 

v_and_b32 v17, -8, v1 // (op 2250) 
s_ashr_i32 s1, s0, 31 // (op 2250) 
v_sub_u32 v22, v1, v17 // (op 2250) 

s_cmp_eq_i32 s10, 0 // (op 2250) 
s_mov_b32 s47, scc // (op 2250) 
v_lshlrev_b32 v17, 4, v22 // (op 2250) 

s_cmp_eq_i32 s47, 0 // (op 2250) 
s_cselect_b32 s18, 0, -1 // (op 2250) 

v_mov_b32 v23, 0 // (op 2250) convert
v_mov_b32 v22, v17 // (op 2250) convert

s_and_b32 s19, s13, 31 // (op 2250) 
s_lshl_b32 s47, 1, s19 // (op 2250) 
s_add_i32 s22, s47, s18 // (op 2250) 


s_and_b32 s18, s1, s22 // (op 2250) 


s_add_i32 s1, s0, s18 // (op 2250) 


s_ashr_i32 s0, s1, s19 // (op 2250) 


s_lshl_b32 s1, s13, 24 // (op 2250) 
s_ashr_i32 s47, s1, 31 // (op 2250) 

s_xor_b32 s1, s0, s47 // (op 2250) 

s_sub_i32 s0, s1, s47 // (op 2250) 


s_mul_hi_i32 s1, s0, s14 // (op 2250) 
s_add_i32 s47, s1, s0 // (op 2250) 

s_ashr_i32 s1, s47, 31 // (op 2250) 
s_cmp_eq_i32 s14, 0 // (op 2250) 
s_mov_b32 s18, scc // (op 2250) 
s_cmp_eq_i32 s18, 0 // (op 2250) 
s_cselect_b32 s19, 0, -1 // (op 2250) 

s_and_b32 s18, s15, 31 // (op 2250) 
s_lshl_b32 s22, 1, s18 // (op 2250) 
s_add_i32 s23, s22, s19 // (op 2250) 


s_and_b32 s19, s1, s23 // (op 2250) 


s_add_i32 s1, s47, s19 // (op 2250) 


s_ashr_i32 s19, s1, s18 // (op 2250) 


s_lshl_b32 s1, s15, 24 // (op 2250) 
s_ashr_i32 s18, s1, 31 // (op 2250) 

s_xor_b32 s1, s19, s18 // (op 2250) 

s_sub_i32 s19, s1, s18 // (op 2250) 


s_add_u32 s22, s6, 64 // (op 2250) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2250) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2250) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2250) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 2250) 

s_mul_i32 s1, s19, s22 // (op 2250) 

s_sub_i32 s47, s0, s1 // (op 2250) 


s_mul_hi_i32 s0, s16, s9 // (op 2250) 
s_add_i32 s1, s0, s16 // (op 2250) 

s_ashr_i32 s0, s1, 31 // (op 2250) 
s_cmp_eq_i32 s9, 0 // (op 2250) 
s_mov_b32 s18, scc // (op 2250) 
s_cmp_eq_i32 s18, 0 // (op 2250) 
s_cselect_b32 s19, 0, -1 // (op 2250) 

s_and_b32 s18, s45, 31 // (op 2250) 
s_lshl_b32 s24, 1, s18 // (op 2250) 
s_add_i32 s25, s24, s19 // (op 2250) 


s_and_b32 s19, s0, s25 // (op 2250) 


s_add_i32 s0, s1, s19 // (op 2250) 


s_ashr_i32 s1, s0, s18 // (op 2250) 


s_lshl_b32 s0, s45, 24 // (op 2250) 
s_ashr_i32 s18, s0, 31 // (op 2250) 

s_xor_b32 s0, s1, s18 // (op 2250) 

s_sub_i32 s1, s0, s18 // (op 2250) 


s_mul_i32 s0, s11, s22 // (op 2250) 

s_mul_i32 s18, s8, s0 // (op 2250) 

s_mul_i32 s0, s1, s18 // (op 2250) 

s_sub_u32 s18, s16, s0 // (op 2250) 

s_mul_hi_i32 s0, s18, s12 // (op 2250) 
s_add_i32 s19, s0, s18 // (op 2250) 


s_ashr_i32 s0, s19, 31 // (op 2250) 
s_cmp_eq_i32 s12, 0 // (op 2250) 
s_mov_b32 s18, scc // (op 2250) 
s_cmp_eq_i32 s18, 0 // (op 2250) 
s_cselect_b32 s22, 0, -1 // (op 2250) 

s_and_b32 s18, s44, 31 // (op 2250) 
s_lshl_b32 s23, 1, s18 // (op 2250) 
s_add_i32 s24, s23, s22 // (op 2250) 


s_and_b32 s25, s0, s24 // (op 2250) 


s_add_i32 s0, s19, s25 // (op 2250) 


s_ashr_i32 s19, s0, s18 // (op 2250) 


s_lshl_b32 s0, s44, 24 // (op 2250) 
s_ashr_i32 s18, s0, 31 // (op 2250) 

s_xor_b32 s0, s19, s18 // (op 2250) 

s_sub_i32 s19, s0, s18 // (op 2250) 


s_cmp_eq_i32 s1, 0 // (op 2250) 
s_mov_b32 s0, scc // (op 2250) 

s_cmp_eq_i32 s0, 0 // (op 2250) 
s_cselect_b32 s1, s19, s47 // (op 2250) 



v_lshl_add_u32 v17, s1, 6, v16 // (op 2250) 


v_mul_lo_u32 v25, v17, s31 // (op 2250) most significant: low of low * high
v_mul_hi_u32 v16, v17, s30 // (op 2250) most significant: high of low * low
v_mul_lo_u32 v24, v17, s30 // (op 2250) least significant: low of low * low
v_add_u32 v25, v25, v16 // (op 2250) most significant: sum


v_add_co_u32 v16, s[0:1], v22, v24 // (op 2250) least significant half
v_addc_co_u32 v17, s[0:1], v23, v25, s[0:1] // (op 2250) most significant half



s_mul_i32 s1, s30, s7 // (op 2252) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 2252) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 2252) least significant
s_mul_hi_u32 s18, s30, s6 // (op 2252) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2252) most significant: sum
s_add_u32 s1, s1, s18 // (op 2252) most significant: sum


s_mov_b32 s60, s32 // (op 2252) reg expression
s_mov_b32 s61, s33 // (op 2252) reg expression
s_mov_b32 s62, s0 // (op 2252) reg expression
s_mov_b32 s63, 131072 // (op 2252) call()

s_mul_hi_i32 s47, s16, s10 // (op 2295) 
v_lshrrev_b32 v22, 2, v1 // (op 2295) 
s_add_i32 s0, s47, s16 // (op 2295) 

v_and_b32 v23, -4, v1 // (op 2295) 
s_ashr_i32 s1, s0, 31 // (op 2295) 
v_sub_u32 v24, v1, v23 // (op 2295) 

s_cmp_eq_i32 s10, 0 // (op 2295) 
s_mov_b32 s47, scc // (op 2295) 
s_cmp_eq_i32 s47, 0 // (op 2295) 
s_cselect_b32 s18, 0, -1 // (op 2295) 

s_and_b32 s19, s13, 31 // (op 2295) 
s_lshl_b32 s47, 1, s19 // (op 2295) 
s_add_i32 s22, s47, s18 // (op 2295) 


s_and_b32 s18, s1, s22 // (op 2295) 


s_add_i32 s1, s0, s18 // (op 2295) 


s_ashr_i32 s0, s1, s19 // (op 2295) 


s_lshl_b32 s1, s13, 24 // (op 2295) 
s_ashr_i32 s47, s1, 31 // (op 2295) 

s_xor_b32 s1, s0, s47 // (op 2295) 

s_sub_i32 s0, s1, s47 // (op 2295) 


s_mul_hi_i32 s1, s0, s14 // (op 2295) 
s_add_i32 s47, s1, s0 // (op 2295) 

s_ashr_i32 s1, s47, 31 // (op 2295) 
s_cmp_eq_i32 s14, 0 // (op 2295) 
s_mov_b32 s18, scc // (op 2295) 
s_cmp_eq_i32 s18, 0 // (op 2295) 
s_cselect_b32 s19, 0, -1 // (op 2295) 

s_and_b32 s18, s15, 31 // (op 2295) 
s_lshl_b32 s22, 1, s18 // (op 2295) 
s_add_i32 s23, s22, s19 // (op 2295) 


s_and_b32 s19, s1, s23 // (op 2295) 


s_add_i32 s1, s47, s19 // (op 2295) 


s_ashr_i32 s19, s1, s18 // (op 2295) 


s_lshl_b32 s1, s15, 24 // (op 2295) 
s_ashr_i32 s18, s1, 31 // (op 2295) 

s_xor_b32 s1, s19, s18 // (op 2295) 

s_sub_i32 s19, s1, s18 // (op 2295) 


s_mul_i32 s1, s0, s11 // (op 2295) 

s_sub_u32 s0, s16, s1 // (op 2295) 

s_mul_i32 s1, s19, s11 // (op 2295) 
s_add_u32 s1, s1, s0 // (op 2295) 


s_mul_hi_i32 s0, s16, s9 // (op 2295) 
s_add_i32 s47, s0, s16 // (op 2295) 

s_ashr_i32 s0, s47, 31 // (op 2295) 
s_cmp_eq_i32 s9, 0 // (op 2295) 
s_mov_b32 s18, scc // (op 2295) 
s_cmp_eq_i32 s18, 0 // (op 2295) 
s_cselect_b32 s19, 0, -1 // (op 2295) 

s_and_b32 s18, s45, 31 // (op 2295) 
s_lshl_b32 s22, 1, s18 // (op 2295) 
s_add_i32 s23, s22, s19 // (op 2295) 


s_and_b32 s19, s0, s23 // (op 2295) 


s_add_i32 s0, s47, s19 // (op 2295) 


s_ashr_i32 s19, s0, s18 // (op 2295) 


s_lshl_b32 s0, s45, 24 // (op 2295) 
s_ashr_i32 s18, s0, 31 // (op 2295) 

s_xor_b32 s0, s19, s18 // (op 2295) 

s_sub_i32 s19, s0, s18 // (op 2295) 


s_add_u32 s22, s6, 64 // (op 2295) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2295) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2295) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2295) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 2295) 

s_mul_i32 s0, s11, s22 // (op 2295) 

s_mul_i32 s18, s8, s0 // (op 2295) 

s_mul_i32 s0, s19, s18 // (op 2295) 

s_sub_u32 s18, s16, s0 // (op 2295) 

s_mov_b32 s0, s18 // (op 2295) 
s_mul_hi_i32 s47, s0, s12 // (op 2295) 
s_add_i32 s22, s47, s0 // (op 2295) 


s_ashr_i32 s0, s22, 31 // (op 2295) 
s_cmp_eq_i32 s12, 0 // (op 2295) 
s_mov_b32 s47, scc // (op 2295) 
s_cmp_eq_i32 s47, 0 // (op 2295) 
s_cselect_b32 s23, 0, -1 // (op 2295) 

s_and_b32 s47, s44, 31 // (op 2295) 
s_lshl_b32 s24, 1, s47 // (op 2295) 
s_add_i32 s25, s24, s23 // (op 2295) 


s_and_b32 s23, s0, s25 // (op 2295) 


s_add_i32 s0, s22, s23 // (op 2295) 


s_ashr_i32 s22, s0, s47 // (op 2295) 


s_lshl_b32 s0, s44, 24 // (op 2295) 
s_ashr_i32 s47, s0, 31 // (op 2295) 

s_xor_b32 s0, s22, s47 // (op 2295) 

s_sub_i32 s22, s0, s47 // (op 2295) 


s_mul_i32 s0, s46, s11 // (op 2295) 
s_add_u32 s24, s20, 64 // (op 2295) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2295) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 2295) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 2295) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 6 // (op 2295) 

s_sub_i32 s23, s24, s0 // (op 2295) 

s_mul_i32 s47, s22, s23 // (op 2295) 


s_sub_u32 s22, s18, s47 // (op 2295) 


s_add_u32 s18, s22, s0 // (op 2295) 


s_cmp_eq_i32 s19, 0 // (op 2295) 
s_mov_b32 s0, scc // (op 2295) 

s_cmp_eq_i32 s0, 0 // (op 2295) 
s_cselect_b32 s19, s18, s1 // (op 2295) 



v_lshl_add_u32 v23, s19, 6, v22 // (op 2295) 


v_mul_lo_u32 v22, v23, s34 // (op 2295) 
v_add_u32 v22, v22, v24 // (op 2295) 


s_mul_i32 s1, s34, s21 // (op 2297) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 2297) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 2297) least significant
s_mul_hi_u32 s18, s34, s20 // (op 2297) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2297) most significant: sum
s_add_u32 s1, s1, s18 // (op 2297) most significant: sum


s_mov_b32 s64, s36 // (op 2297) reg expression
s_mov_b32 s65, s37 // (op 2297) reg expression
s_mov_b32 s66, s0 // (op 2297) reg expression
s_mov_b32 s67, 131072 // (op 2297) call()

s_mul_hi_i32 s47, s16, s10 // (op 2314) 
v_lshrrev_b32 v23, 2, v1 // (op 2314) 
s_add_i32 s0, s47, s16 // (op 2314) 

v_and_b32 v24, -4, v1 // (op 2314) 
s_ashr_i32 s1, s0, 31 // (op 2314) 
v_sub_u32 v25, v1, v24 // (op 2314) 

s_cmp_eq_i32 s10, 0 // (op 2314) 
s_mov_b32 s47, scc // (op 2314) 
s_cmp_eq_i32 s47, 0 // (op 2314) 
s_cselect_b32 s18, 0, -1 // (op 2314) 

s_and_b32 s19, s13, 31 // (op 2314) 
s_lshl_b32 s47, 1, s19 // (op 2314) 
s_add_i32 s22, s47, s18 // (op 2314) 


s_and_b32 s18, s1, s22 // (op 2314) 


s_add_i32 s1, s0, s18 // (op 2314) 


s_ashr_i32 s0, s1, s19 // (op 2314) 


s_lshl_b32 s1, s13, 24 // (op 2314) 
s_ashr_i32 s47, s1, 31 // (op 2314) 

s_xor_b32 s1, s0, s47 // (op 2314) 

s_sub_i32 s0, s1, s47 // (op 2314) 


s_mul_hi_i32 s1, s0, s14 // (op 2314) 
s_add_i32 s47, s1, s0 // (op 2314) 

s_ashr_i32 s1, s47, 31 // (op 2314) 
s_cmp_eq_i32 s14, 0 // (op 2314) 
s_mov_b32 s18, scc // (op 2314) 
s_cmp_eq_i32 s18, 0 // (op 2314) 
s_cselect_b32 s19, 0, -1 // (op 2314) 

s_and_b32 s18, s15, 31 // (op 2314) 
s_lshl_b32 s22, 1, s18 // (op 2314) 
s_add_i32 s23, s22, s19 // (op 2314) 


s_and_b32 s19, s1, s23 // (op 2314) 


s_add_i32 s1, s47, s19 // (op 2314) 


s_ashr_i32 s19, s1, s18 // (op 2314) 


s_lshl_b32 s1, s15, 24 // (op 2314) 
s_ashr_i32 s18, s1, 31 // (op 2314) 

s_xor_b32 s1, s19, s18 // (op 2314) 

s_sub_i32 s19, s1, s18 // (op 2314) 


s_mul_i32 s1, s0, s11 // (op 2314) 

s_sub_u32 s0, s16, s1 // (op 2314) 

s_mul_i32 s1, s19, s11 // (op 2314) 
s_add_u32 s1, s1, s0 // (op 2314) 


s_mul_hi_i32 s0, s16, s9 // (op 2314) 
s_add_i32 s47, s0, s16 // (op 2314) 

s_ashr_i32 s0, s47, 31 // (op 2314) 
s_cmp_eq_i32 s9, 0 // (op 2314) 
s_mov_b32 s18, scc // (op 2314) 
s_cmp_eq_i32 s18, 0 // (op 2314) 
s_cselect_b32 s19, 0, -1 // (op 2314) 

s_and_b32 s18, s45, 31 // (op 2314) 
s_lshl_b32 s22, 1, s18 // (op 2314) 
s_add_i32 s23, s22, s19 // (op 2314) 


s_and_b32 s19, s0, s23 // (op 2314) 


s_add_i32 s0, s47, s19 // (op 2314) 


s_ashr_i32 s19, s0, s18 // (op 2314) 


s_lshl_b32 s0, s45, 24 // (op 2314) 
s_ashr_i32 s18, s0, 31 // (op 2314) 

s_xor_b32 s0, s19, s18 // (op 2314) 

s_sub_i32 s19, s0, s18 // (op 2314) 


s_add_u32 s22, s6, 64 // (op 2314) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2314) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2314) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2314) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 2314) 

s_mul_i32 s0, s11, s22 // (op 2314) 

s_mul_i32 s18, s8, s0 // (op 2314) 

s_mul_i32 s0, s19, s18 // (op 2314) 

s_sub_u32 s18, s16, s0 // (op 2314) 

s_mov_b32 s0, s18 // (op 2314) 
s_mul_hi_i32 s47, s0, s12 // (op 2314) 
s_add_i32 s22, s47, s0 // (op 2314) 


s_ashr_i32 s0, s22, 31 // (op 2314) 
s_cmp_eq_i32 s12, 0 // (op 2314) 
s_mov_b32 s47, scc // (op 2314) 
s_cmp_eq_i32 s47, 0 // (op 2314) 
s_cselect_b32 s23, 0, -1 // (op 2314) 

s_and_b32 s47, s44, 31 // (op 2314) 
s_lshl_b32 s24, 1, s47 // (op 2314) 
s_add_i32 s25, s24, s23 // (op 2314) 


s_and_b32 s23, s0, s25 // (op 2314) 


s_add_i32 s0, s22, s23 // (op 2314) 


s_ashr_i32 s22, s0, s47 // (op 2314) 


s_lshl_b32 s0, s44, 24 // (op 2314) 
s_ashr_i32 s47, s0, 31 // (op 2314) 

s_xor_b32 s0, s22, s47 // (op 2314) 

s_sub_i32 s22, s0, s47 // (op 2314) 


s_mul_i32 s0, s46, s11 // (op 2314) 
s_add_u32 s24, s20, 64 // (op 2314) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2314) End of Int64 add, unlocking SCC
s_sub_u32 s68, s24, 1 // (op 2314) least significant half; sets scc
s_subb_u32 s69, s25, 0 // (op 2314) most significant half; uses scc

s_ashr_i64 s[24:25], s[68:69], 6 // (op 2314) 

s_sub_i32 s23, s24, s0 // (op 2314) 

s_mul_i32 s47, s22, s23 // (op 2314) 


s_sub_u32 s22, s18, s47 // (op 2314) 


s_add_u32 s18, s22, s0 // (op 2314) 


s_cmp_eq_i32 s19, 0 // (op 2314) 
s_mov_b32 s0, scc // (op 2314) 

s_cmp_eq_i32 s0, 0 // (op 2314) 
s_cselect_b32 s19, s18, s1 // (op 2314) 



v_lshl_add_u32 v24, s19, 6, v23 // (op 2314) 


v_mul_lo_u32 v23, v24, s34 // (op 2314) 
v_add_u32 v23, v23, v25 // (op 2314) 


s_mul_i32 s1, s34, s21 // (op 2316) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 2316) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 2316) least significant
s_mul_hi_u32 s18, s34, s20 // (op 2316) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2316) most significant: sum
s_add_u32 s1, s1, s18 // (op 2316) most significant: sum


s_mov_b32 s64, s36 // (op 2316) reg expression
s_mov_b32 s65, s37 // (op 2316) reg expression
s_mov_b32 s66, s0 // (op 2316) reg expression
s_mov_b32 s67, 131072 // (op 2316) call()

s_mul_hi_i32 s47, s16, s10 // (op 2359) 
v_lshrrev_b32 v24, 2, v1 // (op 2359) 
s_add_i32 s0, s47, s16 // (op 2359) 

v_and_b32 v25, -4, v1 // (op 2359) 
s_ashr_i32 s1, s0, 31 // (op 2359) 
v_sub_u32 v26, v1, v25 // (op 2359) 

s_cmp_eq_i32 s10, 0 // (op 2359) 
s_mov_b32 s47, scc // (op 2359) 
s_cmp_eq_i32 s47, 0 // (op 2359) 
s_cselect_b32 s18, 0, -1 // (op 2359) 

s_and_b32 s19, s13, 31 // (op 2359) 
s_lshl_b32 s47, 1, s19 // (op 2359) 
s_add_i32 s22, s47, s18 // (op 2359) 


s_and_b32 s18, s1, s22 // (op 2359) 


s_add_i32 s1, s0, s18 // (op 2359) 


s_ashr_i32 s0, s1, s19 // (op 2359) 


s_lshl_b32 s1, s13, 24 // (op 2359) 
s_ashr_i32 s47, s1, 31 // (op 2359) 

s_xor_b32 s1, s0, s47 // (op 2359) 

s_sub_i32 s0, s1, s47 // (op 2359) 


s_mul_hi_i32 s1, s0, s14 // (op 2359) 
s_add_i32 s47, s1, s0 // (op 2359) 

s_ashr_i32 s1, s47, 31 // (op 2359) 
s_cmp_eq_i32 s14, 0 // (op 2359) 
s_mov_b32 s18, scc // (op 2359) 
s_cmp_eq_i32 s18, 0 // (op 2359) 
s_cselect_b32 s19, 0, -1 // (op 2359) 

s_and_b32 s18, s15, 31 // (op 2359) 
s_lshl_b32 s22, 1, s18 // (op 2359) 
s_add_i32 s23, s22, s19 // (op 2359) 


s_and_b32 s19, s1, s23 // (op 2359) 


s_add_i32 s1, s47, s19 // (op 2359) 


s_ashr_i32 s19, s1, s18 // (op 2359) 


s_lshl_b32 s1, s15, 24 // (op 2359) 
s_ashr_i32 s18, s1, 31 // (op 2359) 

s_xor_b32 s1, s19, s18 // (op 2359) 

s_sub_i32 s19, s1, s18 // (op 2359) 


s_add_u32 s22, s6, 64 // (op 2359) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2359) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2359) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2359) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 2359) 

s_mul_i32 s1, s19, s22 // (op 2359) 

s_sub_i32 s47, s0, s1 // (op 2359) 


s_mul_hi_i32 s0, s16, s9 // (op 2359) 
s_add_i32 s1, s0, s16 // (op 2359) 

s_ashr_i32 s0, s1, 31 // (op 2359) 
s_cmp_eq_i32 s9, 0 // (op 2359) 
s_mov_b32 s18, scc // (op 2359) 
s_cmp_eq_i32 s18, 0 // (op 2359) 
s_cselect_b32 s19, 0, -1 // (op 2359) 

s_and_b32 s18, s45, 31 // (op 2359) 
s_lshl_b32 s24, 1, s18 // (op 2359) 
s_add_i32 s25, s24, s19 // (op 2359) 


s_and_b32 s19, s0, s25 // (op 2359) 


s_add_i32 s0, s1, s19 // (op 2359) 


s_ashr_i32 s1, s0, s18 // (op 2359) 


s_lshl_b32 s0, s45, 24 // (op 2359) 
s_ashr_i32 s18, s0, 31 // (op 2359) 

s_xor_b32 s0, s1, s18 // (op 2359) 

s_sub_i32 s1, s0, s18 // (op 2359) 


s_mul_i32 s0, s11, s22 // (op 2359) 

s_mul_i32 s18, s8, s0 // (op 2359) 

s_mul_i32 s0, s1, s18 // (op 2359) 

s_sub_u32 s18, s16, s0 // (op 2359) 

s_mul_hi_i32 s0, s18, s12 // (op 2359) 
s_add_i32 s19, s0, s18 // (op 2359) 


s_ashr_i32 s0, s19, 31 // (op 2359) 
s_cmp_eq_i32 s12, 0 // (op 2359) 
s_mov_b32 s18, scc // (op 2359) 
s_cmp_eq_i32 s18, 0 // (op 2359) 
s_cselect_b32 s22, 0, -1 // (op 2359) 

s_and_b32 s18, s44, 31 // (op 2359) 
s_lshl_b32 s23, 1, s18 // (op 2359) 
s_add_i32 s24, s23, s22 // (op 2359) 


s_and_b32 s25, s0, s24 // (op 2359) 


s_add_i32 s0, s19, s25 // (op 2359) 


s_ashr_i32 s19, s0, s18 // (op 2359) 


s_lshl_b32 s0, s44, 24 // (op 2359) 
s_ashr_i32 s18, s0, 31 // (op 2359) 

s_xor_b32 s0, s19, s18 // (op 2359) 

s_sub_i32 s19, s0, s18 // (op 2359) 


s_cmp_eq_i32 s1, 0 // (op 2359) 
s_mov_b32 s0, scc // (op 2359) 

s_cmp_eq_i32 s0, 0 // (op 2359) 
s_cselect_b32 s1, s19, s47 // (op 2359) 



v_lshl_add_u32 v25, s1, 6, v24 // (op 2359) 


v_mul_lo_u32 v24, v25, s38 // (op 2359) 

v_add_u32 v25, v26, v24 // (op 2359) 


s_mul_i32 s1, s38, s7 // (op 2361) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 2361) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 2361) least significant
s_mul_hi_u32 s18, s38, s6 // (op 2361) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2361) most significant: sum
s_add_u32 s1, s1, s18 // (op 2361) most significant: sum


s_mov_b32 s68, s40 // (op 2361) reg expression
s_mov_b32 s69, s41 // (op 2361) reg expression
s_mov_b32 s70, s0 // (op 2361) reg expression
s_mov_b32 s71, 131072 // (op 2361) call()

s_mul_hi_i32 s47, s16, s10 // (op 2378) 
v_lshrrev_b32 v24, 2, v1 // (op 2378) 
s_add_i32 s0, s47, s16 // (op 2378) 

v_and_b32 v26, -4, v1 // (op 2378) 
s_ashr_i32 s1, s0, 31 // (op 2378) 
v_sub_u32 v27, v1, v26 // (op 2378) 

s_cmp_eq_i32 s10, 0 // (op 2378) 
s_mov_b32 s47, scc // (op 2378) 
s_cmp_eq_i32 s47, 0 // (op 2378) 
s_cselect_b32 s18, 0, -1 // (op 2378) 

s_and_b32 s19, s13, 31 // (op 2378) 
s_lshl_b32 s47, 1, s19 // (op 2378) 
s_add_i32 s22, s47, s18 // (op 2378) 


s_and_b32 s18, s1, s22 // (op 2378) 


s_add_i32 s1, s0, s18 // (op 2378) 


s_ashr_i32 s0, s1, s19 // (op 2378) 


s_lshl_b32 s1, s13, 24 // (op 2378) 
s_ashr_i32 s47, s1, 31 // (op 2378) 

s_xor_b32 s1, s0, s47 // (op 2378) 

s_sub_i32 s0, s1, s47 // (op 2378) 


s_mul_hi_i32 s1, s0, s14 // (op 2378) 
s_add_i32 s47, s1, s0 // (op 2378) 

s_ashr_i32 s1, s47, 31 // (op 2378) 
s_cmp_eq_i32 s14, 0 // (op 2378) 
s_mov_b32 s18, scc // (op 2378) 
s_cmp_eq_i32 s18, 0 // (op 2378) 
s_cselect_b32 s19, 0, -1 // (op 2378) 

s_and_b32 s18, s15, 31 // (op 2378) 
s_lshl_b32 s22, 1, s18 // (op 2378) 
s_add_i32 s23, s22, s19 // (op 2378) 


s_and_b32 s19, s1, s23 // (op 2378) 


s_add_i32 s1, s47, s19 // (op 2378) 


s_ashr_i32 s19, s1, s18 // (op 2378) 


s_lshl_b32 s1, s15, 24 // (op 2378) 
s_ashr_i32 s18, s1, 31 // (op 2378) 

s_xor_b32 s1, s19, s18 // (op 2378) 

s_sub_i32 s19, s1, s18 // (op 2378) 


s_add_u32 s22, s6, 64 // (op 2378) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2378) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2378) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2378) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 2378) 

s_mul_i32 s1, s19, s22 // (op 2378) 

s_sub_i32 s47, s0, s1 // (op 2378) 


s_mul_hi_i32 s0, s16, s9 // (op 2378) 
s_add_i32 s1, s0, s16 // (op 2378) 

s_ashr_i32 s0, s1, 31 // (op 2378) 
s_cmp_eq_i32 s9, 0 // (op 2378) 
s_mov_b32 s18, scc // (op 2378) 
s_cmp_eq_i32 s18, 0 // (op 2378) 
s_cselect_b32 s19, 0, -1 // (op 2378) 

s_and_b32 s18, s45, 31 // (op 2378) 
s_lshl_b32 s24, 1, s18 // (op 2378) 
s_add_i32 s25, s24, s19 // (op 2378) 


s_and_b32 s19, s0, s25 // (op 2378) 


s_add_i32 s0, s1, s19 // (op 2378) 


s_ashr_i32 s1, s0, s18 // (op 2378) 


s_lshl_b32 s0, s45, 24 // (op 2378) 
s_ashr_i32 s18, s0, 31 // (op 2378) 

s_xor_b32 s0, s1, s18 // (op 2378) 

s_sub_i32 s1, s0, s18 // (op 2378) 


s_mul_i32 s0, s11, s22 // (op 2378) 

s_mul_i32 s18, s8, s0 // (op 2378) 

s_mul_i32 s0, s1, s18 // (op 2378) 

s_sub_u32 s18, s16, s0 // (op 2378) 

s_mul_hi_i32 s0, s18, s12 // (op 2378) 
s_add_i32 s19, s0, s18 // (op 2378) 


s_ashr_i32 s0, s19, 31 // (op 2378) 
s_cmp_eq_i32 s12, 0 // (op 2378) 
s_mov_b32 s18, scc // (op 2378) 
s_cmp_eq_i32 s18, 0 // (op 2378) 
s_cselect_b32 s22, 0, -1 // (op 2378) 

s_and_b32 s18, s44, 31 // (op 2378) 
s_lshl_b32 s23, 1, s18 // (op 2378) 
s_add_i32 s24, s23, s22 // (op 2378) 


s_and_b32 s25, s0, s24 // (op 2378) 


s_add_i32 s0, s19, s25 // (op 2378) 


s_ashr_i32 s19, s0, s18 // (op 2378) 


s_lshl_b32 s0, s44, 24 // (op 2378) 
s_ashr_i32 s18, s0, 31 // (op 2378) 

s_xor_b32 s0, s19, s18 // (op 2378) 

s_sub_i32 s19, s0, s18 // (op 2378) 


s_cmp_eq_i32 s1, 0 // (op 2378) 
s_mov_b32 s0, scc // (op 2378) 

s_cmp_eq_i32 s0, 0 // (op 2378) 
s_cselect_b32 s1, s19, s47 // (op 2378) 



v_lshl_add_u32 v26, s1, 6, v24 // (op 2378) 


v_mul_lo_u32 v24, v26, s38 // (op 2378) 

v_add_u32 v26, v27, v24 // (op 2378) 


s_mul_i32 s1, s38, s7 // (op 2380) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 2380) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 2380) least significant
s_mul_hi_u32 s18, s38, s6 // (op 2380) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2380) most significant: sum
s_add_u32 s1, s1, s18 // (op 2380) most significant: sum


s_mov_b32 s68, s40 // (op 2380) reg expression
s_mov_b32 s69, s41 // (op 2380) reg expression
s_mov_b32 s70, s0 // (op 2380) reg expression
s_mov_b32 s71, 131072 // (op 2380) call()

v_and_b32 v24, -8, v1 // (op 2506) 
v_sub_u32 v27, v1, v24 // (op 2506) 

v_lshlrev_b32 v24, 4, v27 // (op 2506) 

v_lshrrev_b32 v27, 3, v1 // (op 2506) 
v_lshl_add_u32 v28, v27, 7, v24 // (op 2506) 


s_nop 0
v_readfirstlane_b32 s47, v28 // (op 2506) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 5 // (op 2113) 
v_mov_b64 v[28:29], s[0:1] // (op 2113) convert


v_mov_b64 v[30:31], v[14:15] // (op 2113) 
s_mov_b32 m0, s47 // (op 2113) reg expression
buffer_load_dwordx4 v30, s[60:63], 0 offen offset: 0 lds // (op 2113) Load value direct to lds
v_add_co_u32 v30, s[0:1], v30, v28 // (op 2113) least significant half
v_addc_co_u32 v31, s[0:1], 0, v31, s[0:1] // (op 2113) most significant half

s_add_u32 m0, m0, 4096 // (op 2113) 
buffer_load_dwordx4 v30, s[60:63], 0 offen offset: 0 lds // (op 2113) Load value direct to lds






buffer_load_ubyte v24, v25, s[68:71], 0 offen offset: 0 // (op 1943) Load value
v_and_b32 v27, -8, v1 // (op 2449) 
v_sub_u32 v28, v1, v27 // (op 2449) 

v_lshlrev_b32 v27, 4, v28 // (op 2449) 

v_lshrrev_b32 v28, 3, v1 // (op 2449) 
v_lshl_add_u32 v29, v28, 7, v27 // (op 2449) 


s_nop 0
v_readfirstlane_b32 s47, v29 // (op 2449) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 5 // (op 2118) 
v_mov_b64 v[28:29], s[0:1] // (op 2118) convert


v_mov_b64 v[30:31], v[18:19] // (op 2118) 
s_add_u32 s0, 8192, s47 // (op 2118) 
s_mov_b32 m0, s0 // (op 2118) reg expression
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 2118) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 2118) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 2118) most significant half

s_add_u32 m0, m0, 4096 // (op 2118) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 2118) Load value direct to lds







buffer_load_ubyte v27, v22, s[64:67], 0 offen offset: 0 // (op 1949) Load value
v_and_b32 v28, -8, v1 // (op 2733) 
v_sub_u32 v29, v1, v28 // (op 2733) 

v_lshlrev_b32 v28, 4, v29 // (op 2733) 

v_lshrrev_b32 v29, 3, v1 // (op 2733) 
v_lshl_add_u32 v30, v29, 7, v28 // (op 2733) 


s_nop 0
v_readfirstlane_b32 s47, v30 // (op 2733) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 5 // (op 2123) 
v_mov_b64 v[28:29], s[0:1] // (op 2123) convert


v_mov_b64 v[30:31], v[14:15] // (op 2123) 
s_add_u32 s0, 16384, s47 // (op 2123) 
s_mov_b32 m0, s0 // (op 2123) reg expression
s_mov_b32 s1, 128 // (op 2123) call()
buffer_load_dwordx4 v30, s[60:63], s1 offen offset: 0 lds // (op 2123) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 2123) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 2123) most significant half

s_add_u32 m0, m0, 4096 // (op 2123) 
buffer_load_dwordx4 v30, s[60:63], s1 offen offset: 0 lds // (op 2123) Load value direct to lds









buffer_load_ubyte v14, v25, s[68:71], 0 offen offset:4 // (op 1955) Load value

v_and_b32 v15, -8, v1 // (op 2689) 
v_sub_u32 v25, v1, v15 // (op 2689) 

v_lshlrev_b32 v15, 4, v25 // (op 2689) 

v_lshrrev_b32 v25, 3, v1 // (op 2689) 
v_lshl_add_u32 v28, v25, 7, v15 // (op 2689) 


s_nop 0
v_readfirstlane_b32 s47, v28 // (op 2689) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 5 // (op 2125) 
v_mov_b64 v[28:29], s[0:1] // (op 2125) convert


v_mov_b64 v[30:31], v[18:19] // (op 2125) 
s_add_u32 s0, 24576, s47 // (op 2125) 
s_mov_b32 m0, s0 // (op 2125) reg expression
s_mov_b32 s1, 128 // (op 2125) call()
buffer_load_dwordx4 v30, s[48:51], s1 offen offset: 0 lds // (op 2125) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 2125) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 2125) most significant half

s_add_u32 m0, m0, 4096 // (op 2125) 
buffer_load_dwordx4 v30, s[48:51], s1 offen offset: 0 lds // (op 2125) Load value direct to lds









buffer_load_ubyte v15, v22, s[64:67], 0 offen offset:4 // (op 1961) Load value

v_and_b32 v22, -4, v1 // (op 2622) 
v_sub_u32 v25, v1, v22 // (op 2622) 

v_and_b32 v22, 4294967292, v1 // (op 2622) 
v_add_u32 v18, v22, v25 // (op 2622) 





s_waitcnt vmcnt(9)
ds_write_b8 v18, v24 offset:32768 // (op 1964) Store local data 

v_and_b32 v22, -4, v1 // (op 2564) 
v_sub_u32 v25, v1, v22 // (op 2564) 

v_and_b32 v22, 4294967292, v1 // (op 2564) 
v_add_u32 v18, v22, v25 // (op 2564) 





s_waitcnt vmcnt(6)
ds_write_b8 v18, v27 offset:33024 // (op 1966) Store local data 

s_barrier  // (op 2889) 



ds_read_b128 v[28:31], v4 // (op 1991) Load local data 
ds_read_b128 v[32:35], v4 offset:64 // (op 1991) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 2924) 



ds_read_u8 v22, v5 offset:32768 // (op 2011) Load local data 



ds_read_b128 v[36:39], v4 offset:4096 // (op 1996) Load local data 
ds_read_b128 v[40:43], v4 offset:4160 // (op 1996) Load local data 



ds_read_u8 v25, v5 offset:32896 // (op 2016) Load local data 
s_barrier  // (op 2873) 



ds_read_b128 v[44:47], v3 offset:8192 // (op 1981) Load local data 
ds_read_b128 v[48:51], v3 offset:8256 // (op 1981) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 2906) 



ds_read_u8 v18, v0 offset:33024 // (op 2001) Load local data 



ds_read_b128 v[52:55], v3 offset:12288 // (op 1986) Load local data 
ds_read_b128 v[56:59], v3 offset:12352 // (op 1986) Load local data 



ds_read_u8 v19, v0 offset:33152 // (op 2006) Load local data 
v_and_b32 v60, -8, v1 // (op 2421) 
v_sub_u32 v61, v1, v60 // (op 2421) 

v_lshlrev_b32 v60, 4, v61 // (op 2421) 

v_lshrrev_b32 v61, 3, v1 // (op 2421) 
v_lshl_add_u32 v62, v61, 7, v60 // (op 2421) 


s_nop 0
v_readfirstlane_b32 s47, v62 // (op 2421) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v60, -8, v1 // (op 2478) 
v_sub_u32 v61, v1, v60 // (op 2478) 

v_lshlrev_b32 v60, 4, v61 // (op 2478) 

v_lshrrev_b32 v61, 3, v1 // (op 2478) 
v_lshl_add_u32 v62, v61, 7, v60 // (op 2478) 


s_nop 0
v_readfirstlane_b32 s0, v62 // (op 2478) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v60, -4, v1 // (op 2535) 
v_sub_u32 v61, v1, v60 // (op 2535) 

v_and_b32 v60, 4294967292, v1 // (op 2535) 
v_add_u32 v62, v60, v61 // (op 2535) 


v_and_b32 v60, -4, v1 // (op 2593) 
v_sub_u32 v61, v1, v60 // (op 2593) 

v_and_b32 v60, 4294967292, v1 // (op 2593) 
v_add_u32 v63, v60, v61 // (op 2593) 


v_and_b32 v60, -8, v1 // (op 2674) 
v_sub_u32 v61, v1, v60 // (op 2674) 

v_lshlrev_b32 v60, 4, v61 // (op 2674) 

v_lshrrev_b32 v61, 3, v1 // (op 2674) 
v_lshl_add_u32 v64, v61, 7, v60 // (op 2674) 


s_nop 0
v_readfirstlane_b32 s1, v64 // (op 2674) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v60, -8, v1 // (op 2718) 
v_sub_u32 v61, v1, v60 // (op 2718) 

v_lshlrev_b32 v60, 4, v61 // (op 2718) 

v_lshrrev_b32 v61, 3, v1 // (op 2718) 
v_lshl_add_u32 v64, v61, 7, v60 // (op 2718) 


s_nop 0
v_readfirstlane_b32 s18, v64 // (op 2718) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v60, -4, v1 // (op 2762) 
v_sub_u32 v61, v1, v60 // (op 2762) 

v_and_b32 v60, 4294967292, v1 // (op 2762) 
v_add_u32 v64, v60, v61 // (op 2762) 


v_and_b32 v60, -4, v1 // (op 2791) 
v_sub_u32 v61, v1, v60 // (op 2791) 

v_and_b32 v60, 4294967292, v1 // (op 2791) 
v_add_u32 v65, v60, v61 // (op 2791) 


s_mov_b32 s19, 0 // (op 589) call()
s_ashr_i64 s[22:23], s[42:43], 7 // (op 588) 
s_mov_b32 s24, s22 // (op 588) convert

s_and_b32 s25, s24, 4294967294 // (op 588) 

s_cmp_lt_u32 s19, s25 // (op 588) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ForLoopBottom_KLoop_588 // (op 588) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ForLoopBottom_KLoop_588 if false)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ForLoopTop_KLoop_588:
s_add_u32 s19, s19, 2 // (op 590) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:51], v[28:35], a[0:3], v18, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 88) 
s_barrier  // (op 2892) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[44:51], v[36:43], a[4:7], v18, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 376) 
s_lshl_b64 s[22:23], s[30:31], 5 // (op 2085) 
v_mov_b64 v[60:61], s[22:23] // (op 2085) convert


v_mov_b64 v[66:67], v[16:17] // (op 2085) 
v_mov_b64 v[68:69], v[66:67] // (op 2085) 
s_mov_b32 m0, s0 // (op 2085) reg expression
s_mov_b32 s22, 256 // (op 2085) call()
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[52:59], v[28:35], a[8:11], v19, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 673) 
buffer_load_dwordx4 v68, s[60:63], s22 offen offset: 0 lds // (op 2085) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[52:59], v[36:43], a[12:15], v19, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 627) 
v_add_co_u32 v68, s[24:25], v68, v60 // (op 2085) least significant half
v_addc_co_u32 v69, s[24:25], 0, v69, s[24:25] // (op 2085) most significant half

s_add_u32 m0, m0, 4096 // (op 2085) 
buffer_load_dwordx4 v68, s[60:63], s22 offen offset: 0 lds // (op 2085) Load value direct to lds





v_mov_b32 v60, v26 // (op 25) 

buffer_load_ubyte v24, v60, s[68:71], 0 offen offset:8 // (op 25) Load value
s_barrier  // (op 2876) 

s_lshl_b64 s[22:23], s[26:27], 5 // (op 2080) 
v_mov_b64 v[68:69], s[22:23] // (op 2080) convert


v_mov_b64 v[70:71], v[20:21] // (op 2080) 
v_mov_b64 v[72:73], v[70:71] // (op 2080) 
s_add_u32 s22, 8192, s47 // (op 2080) 
s_mov_b32 m0, s22 // (op 2080) reg expression
s_mov_b32 s23, 256 // (op 2080) call()
buffer_load_dwordx4 v72, s[48:51], s23 offen offset: 0 lds // (op 2080) Load value direct to lds
v_add_co_u32 v72, s[24:25], v72, v68 // (op 2080) least significant half
v_addc_co_u32 v73, s[24:25], 0, v73, s[24:25] // (op 2080) most significant half

s_add_u32 m0, m0, 4096 // (op 2080) 
buffer_load_dwordx4 v72, s[48:51], s23 offen offset: 0 lds // (op 2080) Load value direct to lds






v_mov_b32 v61, v23 // (op 17) 

buffer_load_ubyte v27, v61, s[64:67], 0 offen offset:8 // (op 17) Load value
s_barrier  // (op 2964) 



s_waitcnt vmcnt(9)
ds_write_b8 v65, v14 offset:33280 // (op 255) Store local data 
s_barrier  // (op 2958) 



s_waitcnt vmcnt(6)
ds_write_b8 v64, v15 offset:33536 // (op 254) Store local data 
s_barrier  // (op 2947) 



ds_read_b128 v[72:75], v8 offset:16384 // (op 247) Load local data 
ds_read_b128 v[76:79], v8 offset:16448 // (op 247) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 2961) 



ds_read_u8 v68, v9 offset:33280 // (op 251) Load local data 



ds_read_b128 v[80:83], v8 offset:20480 // (op 381) Load local data 
ds_read_b128 v[84:87], v8 offset:20544 // (op 381) Load local data 



ds_read_u8 v69, v9 offset:33408 // (op 385) Load local data 
s_barrier  // (op 2941) 
s_barrier  // (op 3005) 



ds_read_b128 v[44:47], v7 offset:24576 // (op 245) Load local data 
ds_read_b128 v[48:51], v7 offset:24640 // (op 245) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 2954) 



ds_read_u8 v18, v6 offset:33536 // (op 249) Load local data 



ds_read_b128 v[52:55], v7 offset:28672 // (op 630) Load local data 
ds_read_b128 v[56:59], v7 offset:28736 // (op 630) Load local data 



ds_read_u8 v19, v6 offset:33664 // (op 633) Load local data 
s_barrier  // (op 3027) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[44:51], v[72:79], a[0:3], v18, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 252) 
s_barrier  // (op 2951) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[44:51], v[80:87], a[4:7], v18, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 377) 
s_lshl_b64 s[22:23], s[30:31], 5 // (op 2108) 
v_mov_b64 v[88:89], s[22:23] // (op 2108) convert


v_mov_b64 v[90:91], v[66:67] // (op 2108) 
s_add_u32 s22, 16384, s18 // (op 2108) 
s_mov_b32 m0, s22 // (op 2108) reg expression
s_mov_b32 s23, 384 // (op 2108) call()
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[52:59], v[72:79], a[8:11], v19, v68 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 674) 
buffer_load_dwordx4 v90, s[60:63], s23 offen offset: 0 lds // (op 2108) Load value direct to lds


v_add_co_u32 v90, s[24:25], v90, v88 // (op 2108) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[52:59], v[80:87], a[12:15], v19, v69 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 628) 
v_addc_co_u32 v91, s[24:25], 0, v91, s[24:25] // (op 2108) most significant half



s_add_u32 m0, m0, 4096 // (op 2108) 
buffer_load_dwordx4 v90, s[60:63], s23 offen offset: 0 lds // (op 2108) Load value direct to lds








v_add_co_u32 v16, vcc, 256, v16 // (op 2242) least significant half
buffer_load_ubyte v14, v60, s[68:71], 0 offen offset:12 // (op 258) Load value
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 2242) most significant half

s_barrier  // (op 2944) 
v_add_u32 v26, 8, v26 // (op 2370) 

s_lshl_b64 s[22:23], s[26:27], 5 // (op 2103) 
v_mov_b64 v[66:67], s[22:23] // (op 2103) convert


v_mov_b64 v[68:69], v[70:71] // (op 2103) 
s_add_u32 s22, 24576, s1 // (op 2103) 
s_mov_b32 m0, s22 // (op 2103) reg expression
s_mov_b32 s23, 384 // (op 2103) call()
buffer_load_dwordx4 v68, s[48:51], s23 offen offset: 0 lds // (op 2103) Load value direct to lds
v_add_co_u32 v68, s[24:25], v68, v66 // (op 2103) least significant half
v_addc_co_u32 v69, s[24:25], 0, v69, s[24:25] // (op 2103) most significant half

s_add_u32 m0, m0, 4096 // (op 2103) 
buffer_load_dwordx4 v68, s[48:51], s23 offen offset: 0 lds // (op 2103) Load value direct to lds








v_add_co_u32 v20, vcc, 256, v20 // (op 2178) least significant half
buffer_load_ubyte v15, v61, s[64:67], 0 offen offset:12 // (op 257) Load value
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 2178) most significant half

s_barrier  // (op 2927) 
v_add_u32 v23, 8, v23 // (op 2306) 



s_waitcnt vmcnt(9)
ds_write_b8 v63, v24 offset:32768 // (op 30) Store local data 
s_barrier  // (op 2909) 



s_waitcnt vmcnt(6)
ds_write_b8 v62, v27 offset:33024 // (op 21) Store local data 
s_barrier  // (op 2898) 



ds_read_b128 v[28:31], v4 // (op 75) Load local data 
ds_read_b128 v[32:35], v4 offset:64 // (op 75) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 2934) 



ds_read_u8 v22, v5 offset:32768 // (op 85) Load local data 



ds_read_b128 v[36:39], v4 offset:4096 // (op 371) Load local data 
ds_read_b128 v[40:43], v4 offset:4160 // (op 371) Load local data 



ds_read_u8 v25, v5 offset:32896 // (op 375) Load local data 
s_barrier  // (op 2882) 



ds_read_b128 v[44:47], v3 offset:8192 // (op 70) Load local data 
ds_read_b128 v[48:51], v3 offset:8256 // (op 70) Load local data 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 2916) 



ds_read_u8 v18, v0 offset:33024 // (op 80) Load local data 



ds_read_b128 v[52:55], v3 offset:12288 // (op 622) Load local data 
ds_read_b128 v[56:59], v3 offset:12352 // (op 622) Load local data 



ds_read_u8 v19, v0 offset:33152 // (op 625) Load local data 
s_ashr_i64 s[22:23], s[42:43], 7 // (op 588) 
s_mov_b32 s24, s22 // (op 588) convert

s_and_b32 s25, s24, 4294967294 // (op 588) 

s_cmp_lt_u32 s19, s25 // (op 588) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ForLoopTop_KLoop_588 // (op 588) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ForLoopTop_KLoop_588 if true)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ForLoopBottom_KLoop_588:















s_barrier  // (op 3011) 
s_mul_hi_i32 s47, s16, s10 // (op 2138) 
v_lshrrev_b32 v16, 3, v1 // (op 2138) 
s_add_i32 s0, s47, s16 // (op 2138) 

v_and_b32 v17, -8, v1 // (op 2138) 
s_ashr_i32 s1, s0, 31 // (op 2138) 
v_sub_u32 v20, v1, v17 // (op 2138) 

s_cmp_eq_i32 s10, 0 // (op 2138) 
s_mov_b32 s47, scc // (op 2138) 
v_lshlrev_b32 v17, 4, v20 // (op 2138) 

s_cmp_eq_i32 s47, 0 // (op 2138) 
s_cselect_b32 s18, 0, -1 // (op 2138) 

s_and_b32 s19, s13, 31 // (op 2138) 
s_lshl_b32 s47, 1, s19 // (op 2138) 
s_add_i32 s22, s47, s18 // (op 2138) 


s_and_b32 s18, s1, s22 // (op 2138) 


s_add_i32 s1, s0, s18 // (op 2138) 


s_ashr_i32 s0, s1, s19 // (op 2138) 


s_lshl_b32 s1, s13, 24 // (op 2138) 
s_ashr_i32 s47, s1, 31 // (op 2138) 

s_xor_b32 s1, s0, s47 // (op 2138) 

s_sub_i32 s0, s1, s47 // (op 2138) 


s_mul_hi_i32 s1, s0, s14 // (op 2138) 
s_add_i32 s47, s1, s0 // (op 2138) 

s_ashr_i32 s1, s47, 31 // (op 2138) 
s_cmp_eq_i32 s14, 0 // (op 2138) 
s_mov_b32 s18, scc // (op 2138) 
s_cmp_eq_i32 s18, 0 // (op 2138) 
s_cselect_b32 s19, 0, -1 // (op 2138) 

s_and_b32 s18, s15, 31 // (op 2138) 
s_lshl_b32 s22, 1, s18 // (op 2138) 
s_add_i32 s23, s22, s19 // (op 2138) 


s_and_b32 s19, s1, s23 // (op 2138) 


s_add_i32 s1, s47, s19 // (op 2138) 


s_ashr_i32 s19, s1, s18 // (op 2138) 


s_lshl_b32 s1, s15, 24 // (op 2138) 
s_ashr_i32 s18, s1, 31 // (op 2138) 

s_xor_b32 s1, s19, s18 // (op 2138) 

s_sub_i32 s19, s1, s18 // (op 2138) 


s_mul_i32 s1, s0, s11 // (op 2138) 

s_sub_u32 s0, s16, s1 // (op 2138) 

s_mul_i32 s1, s19, s11 // (op 2138) 
s_add_u32 s1, s1, s0 // (op 2138) 


s_mul_hi_i32 s0, s16, s9 // (op 2138) 
s_add_i32 s47, s0, s16 // (op 2138) 

s_ashr_i32 s0, s47, 31 // (op 2138) 
s_cmp_eq_i32 s9, 0 // (op 2138) 
s_mov_b32 s18, scc // (op 2138) 
s_cmp_eq_i32 s18, 0 // (op 2138) 
s_cselect_b32 s19, 0, -1 // (op 2138) 

s_and_b32 s18, s45, 31 // (op 2138) 
s_lshl_b32 s22, 1, s18 // (op 2138) 
s_add_i32 s23, s22, s19 // (op 2138) 


s_and_b32 s19, s0, s23 // (op 2138) 


s_add_i32 s0, s47, s19 // (op 2138) 


s_ashr_i32 s19, s0, s18 // (op 2138) 


s_lshl_b32 s0, s45, 24 // (op 2138) 
s_ashr_i32 s18, s0, 31 // (op 2138) 

s_xor_b32 s0, s19, s18 // (op 2138) 

s_sub_i32 s19, s0, s18 // (op 2138) 


s_add_u32 s22, s6, 64 // (op 2138) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2138) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2138) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2138) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 2138) 

s_mul_i32 s0, s11, s22 // (op 2138) 

s_mul_i32 s18, s8, s0 // (op 2138) 

s_mul_i32 s0, s19, s18 // (op 2138) 

s_sub_u32 s18, s16, s0 // (op 2138) 

s_mov_b32 s0, s18 // (op 2138) 
s_mul_hi_i32 s47, s0, s12 // (op 2138) 
s_add_i32 s22, s47, s0 // (op 2138) 


s_ashr_i32 s0, s22, 31 // (op 2138) 
s_cmp_eq_i32 s12, 0 // (op 2138) 
s_mov_b32 s47, scc // (op 2138) 
s_cmp_eq_i32 s47, 0 // (op 2138) 
s_cselect_b32 s23, 0, -1 // (op 2138) 

s_and_b32 s47, s44, 31 // (op 2138) 
s_lshl_b32 s24, 1, s47 // (op 2138) 
s_add_i32 s25, s24, s23 // (op 2138) 


s_and_b32 s23, s0, s25 // (op 2138) 


s_add_i32 s0, s22, s23 // (op 2138) 


s_ashr_i32 s22, s0, s47 // (op 2138) 


s_lshl_b32 s0, s44, 24 // (op 2138) 
s_ashr_i32 s47, s0, 31 // (op 2138) 

s_xor_b32 s0, s22, s47 // (op 2138) 

s_sub_i32 s22, s0, s47 // (op 2138) 


s_mul_i32 s0, s46, s11 // (op 2138) 
s_add_u32 s24, s20, 64 // (op 2138) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2138) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 2138) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 2138) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 6 // (op 2138) 

s_sub_i32 s23, s24, s0 // (op 2138) 

s_mul_i32 s47, s22, s23 // (op 2138) 


s_sub_u32 s22, s18, s47 // (op 2138) 


s_add_u32 s18, s22, s0 // (op 2138) 


s_cmp_eq_i32 s19, 0 // (op 2138) 
s_mov_b32 s0, scc // (op 2138) 

s_cmp_eq_i32 s0, 0 // (op 2138) 
s_cselect_b32 s19, s18, s1 // (op 2138) 



v_lshl_add_u32 v20, s19, 6, v16 // (op 2138) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 2138) 
s_lshr_b32 s47, s0, 1 // (op 2138) 

v_lshl_add_u32 v16, s47, 8, v17 // (op 2138) 


v_mov_b32 v27, 0 // (op 2138) convert
v_mov_b32 v26, v16 // (op 2138) convert

v_mul_lo_u32 v17, v20, s27 // (op 2138) most significant: low of low * high
v_mul_hi_u32 v21, v20, s26 // (op 2138) most significant: high of low * low
v_mul_lo_u32 v16, v20, s26 // (op 2138) least significant: low of low * low
v_add_u32 v17, v17, v21 // (op 2138) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v26 // (op 2138) least significant half
v_addc_co_u32 v17, s[0:1], v17, v27, s[0:1] // (op 2138) most significant half



s_mul_i32 s1, s26, s21 // (op 2140) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 2140) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 2140) least significant
s_mul_hi_u32 s18, s26, s20 // (op 2140) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2140) most significant: sum
s_add_u32 s1, s1, s18 // (op 2140) most significant: sum


s_mov_b32 s48, s28 // (op 2140) reg expression
s_mov_b32 s49, s29 // (op 2140) reg expression
s_mov_b32 s50, s0 // (op 2140) reg expression
s_mov_b32 s51, 131072 // (op 2140) call()

s_mul_hi_i32 s47, s16, s10 // (op 2208) 
v_lshrrev_b32 v20, 3, v1 // (op 2208) 
s_add_i32 s0, s47, s16 // (op 2208) 

v_and_b32 v21, -8, v1 // (op 2208) 
s_ashr_i32 s1, s0, 31 // (op 2208) 
v_sub_u32 v23, v1, v21 // (op 2208) 

s_cmp_eq_i32 s10, 0 // (op 2208) 
s_mov_b32 s47, scc // (op 2208) 
v_lshlrev_b32 v21, 4, v23 // (op 2208) 

s_cmp_eq_i32 s47, 0 // (op 2208) 
s_cselect_b32 s18, 0, -1 // (op 2208) 

s_and_b32 s19, s13, 31 // (op 2208) 
s_lshl_b32 s47, 1, s19 // (op 2208) 
s_add_i32 s22, s47, s18 // (op 2208) 


s_and_b32 s18, s1, s22 // (op 2208) 


s_add_i32 s1, s0, s18 // (op 2208) 


s_ashr_i32 s0, s1, s19 // (op 2208) 


s_lshl_b32 s1, s13, 24 // (op 2208) 
s_ashr_i32 s47, s1, 31 // (op 2208) 

s_xor_b32 s1, s0, s47 // (op 2208) 

s_sub_i32 s0, s1, s47 // (op 2208) 


s_mul_hi_i32 s1, s0, s14 // (op 2208) 
s_add_i32 s47, s1, s0 // (op 2208) 

s_ashr_i32 s1, s47, 31 // (op 2208) 
s_cmp_eq_i32 s14, 0 // (op 2208) 
s_mov_b32 s18, scc // (op 2208) 
s_cmp_eq_i32 s18, 0 // (op 2208) 
s_cselect_b32 s19, 0, -1 // (op 2208) 

s_and_b32 s18, s15, 31 // (op 2208) 
s_lshl_b32 s22, 1, s18 // (op 2208) 
s_add_i32 s23, s22, s19 // (op 2208) 


s_and_b32 s19, s1, s23 // (op 2208) 


s_add_i32 s1, s47, s19 // (op 2208) 


s_ashr_i32 s19, s1, s18 // (op 2208) 


s_lshl_b32 s1, s15, 24 // (op 2208) 
s_ashr_i32 s18, s1, 31 // (op 2208) 

s_xor_b32 s1, s19, s18 // (op 2208) 

s_sub_i32 s19, s1, s18 // (op 2208) 


s_add_u32 s22, s6, 64 // (op 2208) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2208) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2208) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2208) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 2208) 

s_mul_i32 s1, s19, s22 // (op 2208) 

s_sub_i32 s47, s0, s1 // (op 2208) 


s_mul_hi_i32 s0, s16, s9 // (op 2208) 
s_add_i32 s1, s0, s16 // (op 2208) 

s_ashr_i32 s0, s1, 31 // (op 2208) 
s_cmp_eq_i32 s9, 0 // (op 2208) 
s_mov_b32 s18, scc // (op 2208) 
s_cmp_eq_i32 s18, 0 // (op 2208) 
s_cselect_b32 s19, 0, -1 // (op 2208) 

s_and_b32 s18, s45, 31 // (op 2208) 
s_lshl_b32 s24, 1, s18 // (op 2208) 
s_add_i32 s25, s24, s19 // (op 2208) 


s_and_b32 s19, s0, s25 // (op 2208) 


s_add_i32 s0, s1, s19 // (op 2208) 


s_ashr_i32 s1, s0, s18 // (op 2208) 


s_lshl_b32 s0, s45, 24 // (op 2208) 
s_ashr_i32 s18, s0, 31 // (op 2208) 

s_xor_b32 s0, s1, s18 // (op 2208) 

s_sub_i32 s1, s0, s18 // (op 2208) 


s_mul_i32 s0, s11, s22 // (op 2208) 

s_mul_i32 s18, s8, s0 // (op 2208) 

s_mul_i32 s0, s1, s18 // (op 2208) 

s_sub_u32 s18, s16, s0 // (op 2208) 

s_mul_hi_i32 s0, s18, s12 // (op 2208) 
s_add_i32 s19, s0, s18 // (op 2208) 


s_ashr_i32 s0, s19, 31 // (op 2208) 
s_cmp_eq_i32 s12, 0 // (op 2208) 
s_mov_b32 s18, scc // (op 2208) 
s_cmp_eq_i32 s18, 0 // (op 2208) 
s_cselect_b32 s22, 0, -1 // (op 2208) 

s_and_b32 s18, s44, 31 // (op 2208) 
s_lshl_b32 s23, 1, s18 // (op 2208) 
s_add_i32 s24, s23, s22 // (op 2208) 


s_and_b32 s25, s0, s24 // (op 2208) 


s_add_i32 s0, s19, s25 // (op 2208) 


s_ashr_i32 s19, s0, s18 // (op 2208) 


s_lshl_b32 s0, s44, 24 // (op 2208) 
s_ashr_i32 s18, s0, 31 // (op 2208) 

s_xor_b32 s0, s19, s18 // (op 2208) 

s_sub_i32 s19, s0, s18 // (op 2208) 


s_cmp_eq_i32 s1, 0 // (op 2208) 
s_mov_b32 s0, scc // (op 2208) 

s_cmp_eq_i32 s0, 0 // (op 2208) 
s_cselect_b32 s1, s19, s47 // (op 2208) 



v_lshl_add_u32 v23, s1, 6, v20 // (op 2208) 


v_mul_lo_u32 v27, v23, s31 // (op 2208) most significant: low of low * high
v_mul_hi_u32 v20, v23, s30 // (op 2208) most significant: high of low * low
v_mul_lo_u32 v26, v23, s30 // (op 2208) least significant: low of low * low
v_add_u32 v27, v27, v20 // (op 2208) most significant: sum


s_ashr_i64 s[0:1], s[42:43], 7 // (op 2208) 
s_lshr_b32 s47, s0, 1 // (op 2208) 

v_lshl_add_u32 v20, s47, 8, v21 // (op 2208) 


v_mov_b32 v61, 0 // (op 2208) convert
v_mov_b32 v60, v20 // (op 2208) convert

v_add_co_u32 v20, s[0:1], v60, v26 // (op 2208) least significant half
v_addc_co_u32 v21, s[0:1], v61, v27, s[0:1] // (op 2208) most significant half



s_mul_i32 s1, s30, s7 // (op 2210) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 2210) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 2210) least significant
s_mul_hi_u32 s18, s30, s6 // (op 2210) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2210) most significant: sum
s_add_u32 s1, s1, s18 // (op 2210) most significant: sum


s_mov_b32 s60, s32 // (op 2210) reg expression
s_mov_b32 s61, s33 // (op 2210) reg expression
s_mov_b32 s62, s0 // (op 2210) reg expression
s_mov_b32 s63, 131072 // (op 2210) call()

s_mul_hi_i32 s47, s16, s10 // (op 2272) 
v_lshrrev_b32 v23, 2, v1 // (op 2272) 
s_add_i32 s0, s47, s16 // (op 2272) 

v_and_b32 v24, -4, v1 // (op 2272) 
s_ashr_i32 s1, s0, 31 // (op 2272) 
v_sub_u32 v26, v1, v24 // (op 2272) 

s_cmp_eq_i32 s10, 0 // (op 2272) 
s_mov_b32 s47, scc // (op 2272) 
s_cmp_eq_i32 s47, 0 // (op 2272) 
s_cselect_b32 s18, 0, -1 // (op 2272) 

s_and_b32 s19, s13, 31 // (op 2272) 
s_lshl_b32 s47, 1, s19 // (op 2272) 
s_add_i32 s22, s47, s18 // (op 2272) 


s_and_b32 s18, s1, s22 // (op 2272) 


s_add_i32 s1, s0, s18 // (op 2272) 


s_ashr_i32 s0, s1, s19 // (op 2272) 


s_lshl_b32 s1, s13, 24 // (op 2272) 
s_ashr_i32 s47, s1, 31 // (op 2272) 

s_xor_b32 s1, s0, s47 // (op 2272) 

s_sub_i32 s0, s1, s47 // (op 2272) 


s_mul_hi_i32 s1, s0, s14 // (op 2272) 
s_add_i32 s47, s1, s0 // (op 2272) 

s_ashr_i32 s1, s47, 31 // (op 2272) 
s_cmp_eq_i32 s14, 0 // (op 2272) 
s_mov_b32 s18, scc // (op 2272) 
s_cmp_eq_i32 s18, 0 // (op 2272) 
s_cselect_b32 s19, 0, -1 // (op 2272) 

s_and_b32 s18, s15, 31 // (op 2272) 
s_lshl_b32 s22, 1, s18 // (op 2272) 
s_add_i32 s23, s22, s19 // (op 2272) 


s_and_b32 s19, s1, s23 // (op 2272) 


s_add_i32 s1, s47, s19 // (op 2272) 


s_ashr_i32 s19, s1, s18 // (op 2272) 


s_lshl_b32 s1, s15, 24 // (op 2272) 
s_ashr_i32 s18, s1, 31 // (op 2272) 

s_xor_b32 s1, s19, s18 // (op 2272) 

s_sub_i32 s19, s1, s18 // (op 2272) 


s_mul_i32 s1, s0, s11 // (op 2272) 

s_sub_u32 s0, s16, s1 // (op 2272) 

s_mul_i32 s1, s19, s11 // (op 2272) 
s_add_u32 s1, s1, s0 // (op 2272) 


s_mul_hi_i32 s0, s16, s9 // (op 2272) 
s_add_i32 s47, s0, s16 // (op 2272) 

s_ashr_i32 s0, s47, 31 // (op 2272) 
s_cmp_eq_i32 s9, 0 // (op 2272) 
s_mov_b32 s18, scc // (op 2272) 
s_cmp_eq_i32 s18, 0 // (op 2272) 
s_cselect_b32 s19, 0, -1 // (op 2272) 

s_and_b32 s18, s45, 31 // (op 2272) 
s_lshl_b32 s22, 1, s18 // (op 2272) 
s_add_i32 s23, s22, s19 // (op 2272) 


s_and_b32 s19, s0, s23 // (op 2272) 


s_add_i32 s0, s47, s19 // (op 2272) 


s_ashr_i32 s19, s0, s18 // (op 2272) 


s_lshl_b32 s0, s45, 24 // (op 2272) 
s_ashr_i32 s18, s0, 31 // (op 2272) 

s_xor_b32 s0, s19, s18 // (op 2272) 

s_sub_i32 s19, s0, s18 // (op 2272) 


s_add_u32 s22, s6, 64 // (op 2272) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2272) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2272) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2272) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 2272) 

s_mul_i32 s0, s11, s22 // (op 2272) 

s_mul_i32 s18, s8, s0 // (op 2272) 

s_mul_i32 s0, s19, s18 // (op 2272) 

s_sub_u32 s18, s16, s0 // (op 2272) 

s_mov_b32 s0, s18 // (op 2272) 
s_mul_hi_i32 s47, s0, s12 // (op 2272) 
s_add_i32 s22, s47, s0 // (op 2272) 


s_ashr_i32 s0, s22, 31 // (op 2272) 
s_cmp_eq_i32 s12, 0 // (op 2272) 
s_mov_b32 s47, scc // (op 2272) 
s_cmp_eq_i32 s47, 0 // (op 2272) 
s_cselect_b32 s23, 0, -1 // (op 2272) 

s_and_b32 s47, s44, 31 // (op 2272) 
s_lshl_b32 s24, 1, s47 // (op 2272) 
s_add_i32 s25, s24, s23 // (op 2272) 


s_and_b32 s23, s0, s25 // (op 2272) 


s_add_i32 s0, s22, s23 // (op 2272) 


s_ashr_i32 s22, s0, s47 // (op 2272) 


s_lshl_b32 s0, s44, 24 // (op 2272) 
s_ashr_i32 s47, s0, 31 // (op 2272) 

s_xor_b32 s0, s22, s47 // (op 2272) 

s_sub_i32 s22, s0, s47 // (op 2272) 


s_mul_i32 s0, s46, s11 // (op 2272) 
s_add_u32 s24, s20, 64 // (op 2272) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 2272) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 2272) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 2272) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 6 // (op 2272) 

s_sub_i32 s23, s24, s0 // (op 2272) 

s_mul_i32 s47, s22, s23 // (op 2272) 


s_sub_u32 s22, s18, s47 // (op 2272) 


s_add_u32 s18, s22, s0 // (op 2272) 


s_cmp_eq_i32 s19, 0 // (op 2272) 
s_mov_b32 s0, scc // (op 2272) 

s_cmp_eq_i32 s0, 0 // (op 2272) 
s_cselect_b32 s19, s18, s1 // (op 2272) 



v_lshl_add_u32 v24, s19, 6, v23 // (op 2272) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 2272) 
s_lshr_b32 s47, s0, 1 // (op 2272) 

v_lshl_add_u32 v23, s47, 3, v26 // (op 2272) 


v_mul_lo_u32 v26, v24, s34 // (op 2272) 
v_add_u32 v26, v26, v23 // (op 2272) 


s_mul_i32 s1, s34, s21 // (op 2274) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 2274) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 2274) least significant
s_mul_hi_u32 s18, s34, s20 // (op 2274) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2274) most significant: sum
s_add_u32 s1, s1, s18 // (op 2274) most significant: sum


s_mov_b32 s64, s36 // (op 2274) reg expression
s_mov_b32 s65, s37 // (op 2274) reg expression
s_mov_b32 s66, s0 // (op 2274) reg expression
s_mov_b32 s67, 131072 // (op 2274) call()

s_mul_hi_i32 s47, s16, s10 // (op 2336) 
v_lshrrev_b32 v27, 2, v1 // (op 2336) 
s_add_i32 s0, s47, s16 // (op 2336) 

v_and_b32 v23, -4, v1 // (op 2336) 
s_ashr_i32 s1, s0, 31 // (op 2336) 
v_sub_u32 v24, v1, v23 // (op 2336) 

s_cmp_eq_i32 s10, 0 // (op 2336) 
s_mov_b32 s47, scc // (op 2336) 
s_cmp_eq_i32 s47, 0 // (op 2336) 
s_cselect_b32 s18, 0, -1 // (op 2336) 

s_and_b32 s19, s13, 31 // (op 2336) 
s_lshl_b32 s47, 1, s19 // (op 2336) 
s_add_i32 s22, s47, s18 // (op 2336) 


s_and_b32 s18, s1, s22 // (op 2336) 


s_add_i32 s1, s0, s18 // (op 2336) 


s_ashr_i32 s0, s1, s19 // (op 2336) 


s_lshl_b32 s1, s13, 24 // (op 2336) 
s_ashr_i32 s47, s1, 31 // (op 2336) 

s_xor_b32 s1, s0, s47 // (op 2336) 

s_sub_i32 s0, s1, s47 // (op 2336) 


s_mul_hi_i32 s1, s0, s14 // (op 2336) 
s_add_i32 s47, s1, s0 // (op 2336) 

s_ashr_i32 s1, s47, 31 // (op 2336) 
s_cmp_eq_i32 s14, 0 // (op 2336) 
s_mov_b32 s18, scc // (op 2336) 
s_cmp_eq_i32 s18, 0 // (op 2336) 
s_cselect_b32 s19, 0, -1 // (op 2336) 

s_and_b32 s18, s15, 31 // (op 2336) 
s_lshl_b32 s22, 1, s18 // (op 2336) 
s_add_i32 s23, s22, s19 // (op 2336) 


s_and_b32 s19, s1, s23 // (op 2336) 


s_add_i32 s1, s47, s19 // (op 2336) 


s_ashr_i32 s19, s1, s18 // (op 2336) 


s_lshl_b32 s1, s15, 24 // (op 2336) 
s_ashr_i32 s18, s1, 31 // (op 2336) 

s_xor_b32 s1, s19, s18 // (op 2336) 

s_sub_i32 s19, s1, s18 // (op 2336) 


s_add_u32 s22, s6, 64 // (op 2336) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 2336) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 2336) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 2336) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 6 // (op 2336) 

s_mul_i32 s1, s19, s22 // (op 2336) 

s_sub_i32 s47, s0, s1 // (op 2336) 


s_mul_hi_i32 s0, s16, s9 // (op 2336) 
s_add_i32 s1, s0, s16 // (op 2336) 

s_ashr_i32 s0, s1, 31 // (op 2336) 
s_cmp_eq_i32 s9, 0 // (op 2336) 
s_mov_b32 s18, scc // (op 2336) 
s_cmp_eq_i32 s18, 0 // (op 2336) 
s_cselect_b32 s19, 0, -1 // (op 2336) 

s_and_b32 s18, s45, 31 // (op 2336) 
s_lshl_b32 s24, 1, s18 // (op 2336) 
s_add_i32 s25, s24, s19 // (op 2336) 


s_and_b32 s19, s0, s25 // (op 2336) 


s_add_i32 s0, s1, s19 // (op 2336) 


s_ashr_i32 s1, s0, s18 // (op 2336) 


s_lshl_b32 s0, s45, 24 // (op 2336) 
s_ashr_i32 s18, s0, 31 // (op 2336) 

s_xor_b32 s0, s1, s18 // (op 2336) 

s_sub_i32 s1, s0, s18 // (op 2336) 


s_mul_i32 s0, s11, s22 // (op 2336) 

s_mul_i32 s18, s8, s0 // (op 2336) 

s_mul_i32 s0, s1, s18 // (op 2336) 

s_sub_u32 s18, s16, s0 // (op 2336) 

s_mul_hi_i32 s0, s18, s12 // (op 2336) 
s_add_i32 s19, s0, s18 // (op 2336) 


s_ashr_i32 s0, s19, 31 // (op 2336) 
s_cmp_eq_i32 s12, 0 // (op 2336) 
s_mov_b32 s18, scc // (op 2336) 
s_cmp_eq_i32 s18, 0 // (op 2336) 
s_cselect_b32 s22, 0, -1 // (op 2336) 

s_and_b32 s18, s44, 31 // (op 2336) 
s_lshl_b32 s23, 1, s18 // (op 2336) 
s_add_i32 s24, s23, s22 // (op 2336) 


s_and_b32 s25, s0, s24 // (op 2336) 


s_add_i32 s0, s19, s25 // (op 2336) 


s_ashr_i32 s19, s0, s18 // (op 2336) 


s_lshl_b32 s0, s44, 24 // (op 2336) 
s_ashr_i32 s18, s0, 31 // (op 2336) 

s_xor_b32 s0, s19, s18 // (op 2336) 

s_sub_i32 s19, s0, s18 // (op 2336) 


s_cmp_eq_i32 s1, 0 // (op 2336) 
s_mov_b32 s0, scc // (op 2336) 

s_cmp_eq_i32 s0, 0 // (op 2336) 
s_cselect_b32 s1, s19, s47 // (op 2336) 



v_lshl_add_u32 v23, s1, 6, v27 // (op 2336) 


v_mul_lo_u32 v27, v23, s38 // (op 2336) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 2336) 
s_lshr_b32 s47, s0, 1 // (op 2336) 

v_lshl_add_u32 v23, s47, 3, v24 // (op 2336) 


v_add_u32 v24, v23, v27 // (op 2336) 


s_mul_i32 s1, s38, s7 // (op 2338) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 2338) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 2338) least significant
s_mul_hi_u32 s18, s38, s6 // (op 2338) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 2338) most significant: sum
s_add_u32 s1, s1, s18 // (op 2338) most significant: sum


s_mov_b32 s68, s40 // (op 2338) reg expression
s_mov_b32 s69, s41 // (op 2338) reg expression
s_mov_b32 s70, s0 // (op 2338) reg expression
s_mov_b32 s71, 131072 // (op 2338) call()

v_and_b32 v23, -8, v1 // (op 2434) 
v_sub_u32 v27, v1, v23 // (op 2434) 

v_lshlrev_b32 v23, 4, v27 // (op 2434) 

v_lshrrev_b32 v27, 3, v1 // (op 2434) 
v_lshl_add_u32 v60, v27, 7, v23 // (op 2434) 


s_nop 0
v_readfirstlane_b32 s47, v60 // (op 2434) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v23, -8, v1 // (op 2491) 
v_sub_u32 v27, v1, v23 // (op 2491) 

v_lshlrev_b32 v23, 4, v27 // (op 2491) 

v_lshrrev_b32 v27, 3, v1 // (op 2491) 
v_lshl_add_u32 v60, v27, 7, v23 // (op 2491) 


s_nop 0
v_readfirstlane_b32 s0, v60 // (op 2491) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v23, -4, v1 // (op 2548) 
v_sub_u32 v27, v1, v23 // (op 2548) 

v_and_b32 v23, 4294967292, v1 // (op 2548) 
v_add_u32 v60, v23, v27 // (op 2548) 


v_and_b32 v23, -4, v1 // (op 2606) 
v_sub_u32 v27, v1, v23 // (op 2606) 

v_and_b32 v23, 4294967292, v1 // (op 2606) 
v_add_u32 v61, v23, v27 // (op 2606) 


s_ashr_i64 s[18:19], s[42:43], 7 // (op 594) 
s_and_b32 s1, s18, 4294967294 // (op 594) 

s_ashr_i64 s[18:19], s[42:43], 7 // (op 593) 
s_mov_b32 s22, s18 // (op 593) convert

s_cmp_lt_u32 s1, s22 // (op 593) 











s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_593 // (op 593) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_593 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ForLoopTop_KLoopTail_593:
s_barrier  // (op 2879) 
s_add_u32 s1, s1, 1 // (op 595) 

s_lshl_b64 s[18:19], s[26:27], 5 // (op 2090) 
v_mov_b64 v[14:15], s[18:19] // (op 2090) convert


v_mov_b64 v[18:19], v[16:17] // (op 2090) 
v_mov_b64 v[22:23], v[18:19] // (op 2090) 
s_add_u32 s18, 8192, s47 // (op 2090) 
s_mov_b32 m0, s18 // (op 2090) reg expression
buffer_load_dwordx4 v22, s[48:51], 0 offen offset: 0 lds // (op 2090) Load value direct to lds
v_add_co_u32 v22, s[22:23], v22, v14 // (op 2090) least significant half
v_addc_co_u32 v23, s[22:23], 0, v23, s[22:23] // (op 2090) most significant half

s_add_u32 m0, m0, 4096 // (op 2090) 
buffer_load_dwordx4 v22, s[48:51], 0 offen offset: 0 lds // (op 2090) Load value direct to lds




v_add_co_u32 v16, vcc, 128, v16 // (op 2128) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 2128) most significant half
s_barrier  // (op 2895) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 2096) 
v_mov_b64 v[14:15], s[18:19] // (op 2096) convert


v_mov_b64 v[18:19], v[20:21] // (op 2096) 
v_mov_b64 v[22:23], v[18:19] // (op 2096) 
s_mov_b32 m0, s0 // (op 2096) reg expression
buffer_load_dwordx4 v22, s[60:63], 0 offen offset: 0 lds // (op 2096) Load value direct to lds
v_add_co_u32 v22, s[18:19], v22, v14 // (op 2096) least significant half
v_addc_co_u32 v23, s[18:19], 0, v23, s[18:19] // (op 2096) most significant half

s_add_u32 m0, m0, 4096 // (op 2096) 
buffer_load_dwordx4 v22, s[60:63], 0 offen offset: 0 lds // (op 2096) Load value direct to lds



v_add_co_u32 v20, vcc, 128, v20 // (op 2200) least significant half
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 2200) most significant half


v_mov_b32 v25, v26 // (op 191) 

buffer_load_ubyte v14, v25, s[64:67], 0 offen offset: 0 // (op 191) Load value

v_add_u32 v26, 4, v26 // (op 2264) 


s_barrier  // (op 2912) 
v_mov_b32 v15, v24 // (op 192) 




buffer_load_ubyte v25, v15, s[68:71], 0 offen offset: 0 // (op 192) Load value

v_add_u32 v24, 4, v24 // (op 2328) 
s_waitcnt vmcnt(1)
ds_write_b8 v60, v14 offset:33024 // (op 188) Store local data 

s_barrier  // (op 2885) 



s_barrier  // (op 2930) 
ds_read_b128 v[52:55], v3 offset:8192 // (op 179) Load local data 



ds_read_b128 v[56:59], v3 offset:8256 // (op 179) Load local data 
s_waitcnt vmcnt(0)
ds_write_b8 v61, v25 offset:32768 // (op 189) Store local data 

s_barrier  // (op 2902) 



ds_read_b128 v[44:47], v4 // (op 181) Load local data 
ds_read_b128 v[48:51], v4 offset:64 // (op 181) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 2920) 



ds_read_u8 v25, v0 offset:33024 // (op 183) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 2937) 



ds_read_u8 v14, v5 offset:32768 // (op 185) Load local data 



ds_read_b128 v[36:39], v4 offset:4096 // (op 350) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[52:59], v[44:51], a[0:3], v25, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 186) 
ds_read_b128 v[40:43], v4 offset:4160 // (op 350) Load local data 



ds_read_u8 v15, v5 offset:32896 // (op 354) Load local data 



ds_read_b128 v[28:31], v3 offset:12288 // (op 568) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[52:59], v[36:43], a[4:7], v25, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 355) 
ds_read_b128 v[32:35], v3 offset:12352 // (op 568) Load local data 





ds_read_u8 v25, v0 offset:33152 // (op 571) Load local data 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[28:35], v[44:51], a[8:11], v25, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 573) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[28:35], v[36:43], a[12:15], v25, v15 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 604) 




s_ashr_i64 s[18:19], s[42:43], 7 // (op 593) 
s_mov_b32 s22, s18 // (op 593) convert

s_cmp_lt_u32 s1, s22 // (op 593) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ForLoopTop_KLoopTail_593 // (op 593) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ForLoopTop_KLoopTail_593 if true)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ForLoopBottom_KLoopTail_593:






















s_barrier  // (op 3015) 





















v_cmp_eq_f32 vcc, 0.00000, v10 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ConditionalFalse_0_beta_133
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


v_accvgpr_read v6, a4 // (op 1017) 
v_accvgpr_read v7, a5 // (op 1017) 
v_accvgpr_read v8, a6 // (op 1017) 
v_accvgpr_read v9, a7 // (op 1017) 
v_mul_f32 v14, v11, v6 // (op 1017) 
v_mul_f32 v15, v11, v7 // (op 1017) 
v_mul_f32 v16, v11, v8 // (op 1017) 
v_mul_f32 v17, v11, v9 // (op 1017) 



s_lshl_b32 s0, s4, 7 // (op 1019) 
v_add_u32 v0, v13, s0 // (op 1019) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 1019) Store value


v_accvgpr_read v6, a8 // (op 1041) 
v_accvgpr_read v7, a9 // (op 1041) 
v_accvgpr_read v8, a10 // (op 1041) 
v_accvgpr_read v9, a11 // (op 1041) 
v_mul_f32 v14, v11, v6 // (op 1041) 
v_mul_f32 v15, v11, v7 // (op 1041) 
v_mul_f32 v16, v11, v8 // (op 1041) 
v_mul_f32 v17, v11, v9 // (op 1041) 



v_add_u32 v0, 128, v13 // (op 1044) 

buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 1044) Store value


v_accvgpr_read v6, a12 // (op 1050) 
v_accvgpr_read v7, a13 // (op 1050) 
v_accvgpr_read v8, a14 // (op 1050) 
v_accvgpr_read v9, a15 // (op 1050) 
v_mul_f32 v14, v11, v6 // (op 1050) 
v_mul_f32 v15, v11, v7 // (op 1050) 
v_mul_f32 v16, v11, v8 // (op 1050) 
v_mul_f32 v17, v11, v9 // (op 1050) 



s_lshl_b32 s0, s4, 7 // (op 1046) 
s_add_u32 s0, s0, 128 // (op 1046) 
v_add_u32 v0, v13, s0 // (op 1046) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 1046) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ConditionalBottom_0_beta_133
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ConditionalFalse_0_beta_133:


v_add_u32 v0, 128, v12 // (op 1130) 

buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 1130) Load value

v_accvgpr_read v14, a8 // (op 1122) 
v_accvgpr_read v15, a9 // (op 1122) 
v_accvgpr_read v16, a10 // (op 1122) 
v_accvgpr_read v17, a11 // (op 1122) 
v_mul_f32 v18, v11, v14 // (op 1122) 
v_mul_f32 v19, v11, v15 // (op 1122) 
v_mul_f32 v20, v11, v16 // (op 1122) 
v_mul_f32 v21, v11, v17 // (op 1122) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 1129) 
v_mul_f32 v15, v10, v7 // (op 1129) 
v_mul_f32 v16, v10, v8 // (op 1129) 
v_mul_f32 v17, v10, v9 // (op 1129) 

v_add_f32 v6, v14, v18 // (op 1123) 
v_add_f32 v7, v15, v19 // (op 1123) 
v_add_f32 v8, v16, v20 // (op 1123) 
v_add_f32 v9, v17, v21 // (op 1123) 




v_add_u32 v0, 128, v13 // (op 1125) 

buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 1125) Store value




s_lshl_b32 s0, s2, 7 // (op 1134) 
s_add_u32 s0, s0, 128 // (op 1134) 
v_add_u32 v0, v12, s0 // (op 1134) 


buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 1134) Load value

v_accvgpr_read v14, a12 // (op 1136) 
v_accvgpr_read v15, a13 // (op 1136) 
v_accvgpr_read v16, a14 // (op 1136) 
v_accvgpr_read v17, a15 // (op 1136) 
v_mul_f32 v18, v11, v14 // (op 1136) 
v_mul_f32 v19, v11, v15 // (op 1136) 
v_mul_f32 v20, v11, v16 // (op 1136) 
v_mul_f32 v21, v11, v17 // (op 1136) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 1132) 
v_mul_f32 v15, v10, v7 // (op 1132) 
v_mul_f32 v16, v10, v8 // (op 1132) 
v_mul_f32 v17, v10, v9 // (op 1132) 

v_add_f32 v6, v14, v18 // (op 1133) 
v_add_f32 v7, v15, v19 // (op 1133) 
v_add_f32 v8, v16, v20 // (op 1133) 
v_add_f32 v9, v17, v21 // (op 1133) 




s_lshl_b32 s0, s4, 7 // (op 1127) 
s_add_u32 s0, s0, 128 // (op 1127) 
v_add_u32 v0, v13, s0 // (op 1127) 


buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 1127) Store value




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




s_lshl_b32 s0, s2, 7 // (op 1092) 
v_add_u32 v0, v12, s0 // (op 1092) 


buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 1092) Load value

v_accvgpr_read v14, a4 // (op 1089) 
v_accvgpr_read v15, a5 // (op 1089) 
v_accvgpr_read v16, a6 // (op 1089) 
v_accvgpr_read v17, a7 // (op 1089) 
v_mul_f32 v18, v11, v14 // (op 1089) 
v_mul_f32 v19, v11, v15 // (op 1089) 
v_mul_f32 v20, v11, v16 // (op 1089) 
v_mul_f32 v21, v11, v17 // (op 1089) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 1093) 
v_mul_f32 v15, v10, v7 // (op 1093) 
v_mul_f32 v16, v10, v8 // (op 1093) 
v_mul_f32 v17, v10, v9 // (op 1093) 

v_add_f32 v6, v14, v18 // (op 1090) 
v_add_f32 v7, v15, v19 // (op 1090) 
v_add_f32 v8, v16, v20 // (op 1090) 
v_add_f32 v9, v17, v21 // (op 1090) 




s_lshl_b32 s0, s4, 7 // (op 1091) 
v_add_u32 v0, v13, s0 // (op 1091) 


buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 1091) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__ConditionalBottom_0_beta_133:












s_endpgm  // End of RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM_
.LRR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__end:
.size RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM_, .LRR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM__end-RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 108
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 33792 // lds bytes
  .amdhsa_accum_offset 92
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
    macrotile: [64, 64, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [2, 2]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x64x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 33792
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 74
    .vgpr_count: 92
    .agpr_count: 16
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {is-null: true}, {is-null: true}]
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
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 28
        .expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}}
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
        .expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}}}}
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

