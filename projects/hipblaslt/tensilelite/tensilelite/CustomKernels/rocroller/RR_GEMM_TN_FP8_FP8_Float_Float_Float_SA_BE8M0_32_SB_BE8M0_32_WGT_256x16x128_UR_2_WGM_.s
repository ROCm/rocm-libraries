// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM_,@function
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 3805) 
s_mul_hi_i32 s1, s16, s10 // (op 3992) 
v_and_b32 v0, 15, v1 // (op 3918) 
v_and_b32 v3, 48, v1 // (op 3918) 
v_lshl_add_u32 v4, v0, 7, v3 // (op 3918) 


v_and_b32 v0, 15, v1 // (op 4077) 
v_and_b32 v3, 48, v1 // (op 4077) 
v_lshl_add_u32 v5, v0, 7, v3 // (op 4077) 


v_lshrrev_b32 v0, 6, v1 // (op 3805) 
s_add_i32 s18, s0, s16 // (op 3805) 

v_and_b32 v3, 63, v1 // (op 3805) 
s_ashr_i32 s0, s18, 31 // (op 3805) 
v_lshrrev_b32 v6, 4, v3 // (op 3805) 
s_cmp_eq_i32 s10, 0 // (op 3805) 
s_mov_b32 s19, scc // (op 3805) 
v_lshlrev_b32 v7, 2, v6 // (op 3805) 

s_cmp_eq_i32 s19, 0 // (op 3805) 
s_cselect_b32 s49, 0, -1 // (op 3805) 

v_lshl_add_u32 v6, v0, 4, v7 // (op 3805) 


s_and_b32 s19, s13, 31 // (op 3805) 
v_lshrrev_b32 v0, 2, v3 // (op 3805) 

v_and_b32 v3, 3, v0 // (op 3805) 

v_lshrrev_b32 v0, 6, v1 // (op 3866) 
v_and_b32 v7, 15, v1 // (op 3866) 
v_lshl_add_u32 v8, v0, 4, v7 // (op 3866) 


v_and_b32 v0, 48, v1 // (op 3866) 
v_lshl_add_u32 v7, v8, 7, v0 // (op 3866) 


v_lshrrev_b32 v0, 6, v1 // (op 3976) 
v_and_b32 v8, 15, v1 // (op 3976) 
v_lshl_add_u32 v9, v0, 4, v8 // (op 3976) 


v_and_b32 v0, 63, v1 // (op 3976) 
v_lshrrev_b32 v8, 4, v0 // (op 3976) 

v_lshl_add_u32 v0, v9, 2, v8 // (op 3976) 


v_lshrrev_b32 v8, 6, v1 // (op 3992) 
v_and_b32 v9, 63, v1 // (op 3992) 
v_lshrrev_b32 v10, 6, v1 // (op 4038) 
v_and_b32 v11, 15, v1 // (op 4038) 
v_lshl_add_u32 v12, v10, 4, v11 // (op 4038) 


v_and_b32 v10, 48, v1 // (op 4038) 
v_lshl_add_u32 v11, v12, 7, v10 // (op 4038) 


v_lshrrev_b32 v10, 6, v1 // (op 4106) 
v_and_b32 v12, 15, v1 // (op 4106) 
v_lshl_add_u32 v13, v10, 4, v12 // (op 4106) 


v_and_b32 v10, 63, v1 // (op 4106) 
v_lshrrev_b32 v12, 4, v10 // (op 4106) 

v_lshl_add_u32 v10, v13, 2, v12 // (op 4106) 


s_lshl_b32 s50, 1, s19 // (op 3805) 
v_and_b32 v12, 3, v1 // (op 3805) 
s_add_i32 s51, s50, s49 // (op 3805) 


s_and_b32 s49, s0, s51 // (op 3805) 


s_add_i32 s0, s1, s16 // (op 3992) 

s_ashr_i32 s1, s0, 31 // (op 3992) 
v_lshl_add_u32 v13, v3, 2, v12 // (op 3805) 


s_add_i32 s50, s18, s49 // (op 3805) 


s_ashr_i32 s18, s50, s19 // (op 3805) 


s_lshl_b32 s19, s13, 24 // (op 3805) 
s_ashr_i32 s49, s19, 31 // (op 3805) 

s_xor_b32 s19, s18, s49 // (op 3805) 

s_sub_i32 s18, s19, s49 // (op 3805) 


s_mul_hi_i32 s19, s18, s14 // (op 3805) 
s_add_i32 s49, s19, s18 // (op 3805) 

s_ashr_i32 s19, s49, 31 // (op 3805) 
s_cmp_eq_i32 s14, 0 // (op 3805) 
s_mov_b32 s50, scc // (op 3805) 
s_cmp_eq_i32 s50, 0 // (op 3805) 
s_cselect_b32 s51, 0, -1 // (op 3805) 

s_and_b32 s50, s15, 31 // (op 3805) 
v_lshrrev_b32 v3, 4, v9 // (op 3992) 
s_cmp_eq_i32 s10, 0 // (op 3992) 
s_mov_b32 s52, scc // (op 3992) 
v_lshlrev_b32 v12, 2, v3 // (op 3992) 

s_cmp_eq_i32 s52, 0 // (op 3992) 
s_cselect_b32 s53, 0, -1 // (op 3992) 

s_lshl_b32 s52, 1, s50 // (op 3805) 
v_lshl_add_u32 v3, v8, 4, v12 // (op 3992) 


v_lshrrev_b32 v8, 2, v9 // (op 3992) 

s_add_i32 s54, s52, s51 // (op 3805) 


s_and_b32 s51, s19, s54 // (op 3805) 


s_add_i32 s19, s49, s51 // (op 3805) 


s_ashr_i32 s49, s19, s50 // (op 3805) 


s_lshl_b32 s19, s15, 24 // (op 3805) 
s_ashr_i32 s50, s19, 31 // (op 3805) 

s_xor_b32 s19, s49, s50 // (op 3805) 

s_sub_i32 s49, s19, s50 // (op 3805) 


s_mul_i32 s19, s18, s11 // (op 3805) 
s_sub_u32 s50, s16, s19 // (op 3805) 

s_mul_i32 s19, s49, s11 // (op 3805) 
s_add_u32 s19, s19, s50 // (op 3805) 

s_mul_hi_i32 s50, s16, s9 // (op 3805) 
s_add_i32 s51, s50, s16 // (op 3805) 

s_ashr_i32 s50, s51, 31 // (op 3805) 
s_cmp_eq_i32 s9, 0 // (op 3805) 
s_mov_b32 s52, scc // (op 3805) 
s_cmp_eq_i32 s52, 0 // (op 3805) 
s_cselect_b32 s54, 0, -1 // (op 3805) 

s_and_b32 s52, s13, 31 // (op 3992) 
v_and_b32 v9, 3, v8 // (op 3992) 

s_lshl_b32 s55, 1, s52 // (op 3992) 
v_and_b32 v8, 3, v1 // (op 3992) 
s_add_i32 s56, s55, s53 // (op 3992) 


s_and_b32 s53, s1, s56 // (op 3992) 


v_lshl_add_u32 v12, v9, 2, v8 // (op 3992) 


s_add_i32 s1, s0, s53 // (op 3992) 


s_ashr_i32 s0, s1, s52 // (op 3992) 


s_lshl_b32 s1, s13, 24 // (op 3992) 
s_ashr_i32 s52, s1, 31 // (op 3992) 

s_xor_b32 s1, s0, s52 // (op 3992) 

s_sub_i32 s0, s1, s52 // (op 3992) 


s_mul_hi_i32 s1, s0, s14 // (op 3992) 
s_add_i32 s52, s1, s0 // (op 3992) 

s_ashr_i32 s1, s52, 31 // (op 3992) 
s_cmp_eq_i32 s14, 0 // (op 3992) 
s_mov_b32 s53, scc // (op 3992) 
s_cmp_eq_i32 s53, 0 // (op 3992) 
s_cselect_b32 s55, 0, -1 // (op 3992) 

s_and_b32 s53, s15, 31 // (op 3992) 
s_lshl_b32 s56, 1, s53 // (op 3992) 
s_add_i32 s57, s56, s55 // (op 3992) 


s_and_b32 s55, s1, s57 // (op 3992) 


s_add_i32 s1, s52, s55 // (op 3992) 


s_ashr_i32 s52, s1, s53 // (op 3992) 


s_lshl_b32 s1, s15, 24 // (op 3992) 
s_ashr_i32 s53, s1, 31 // (op 3992) 

s_xor_b32 s1, s52, s53 // (op 3992) 

s_sub_i32 s52, s1, s53 // (op 3992) 


s_mul_i32 s1, s0, s11 // (op 3992) 
s_sub_u32 s53, s16, s1 // (op 3992) 

s_mul_i32 s1, s52, s11 // (op 3992) 
s_add_u32 s1, s1, s53 // (op 3992) 

s_mul_hi_i32 s53, s16, s9 // (op 3992) 
s_add_i32 s55, s53, s16 // (op 3992) 

s_ashr_i32 s53, s55, 31 // (op 3992) 
s_cmp_eq_i32 s9, 0 // (op 3992) 
s_mov_b32 s56, scc // (op 3992) 
s_cmp_eq_i32 s56, 0 // (op 3992) 
s_cselect_b32 s57, 0, -1 // (op 3992) 


s_waitcnt lgkmcnt(0)
s_and_b32 s56, s45, 31 // (op 3805) 
s_lshl_b32 s58, 1, s56 // (op 3805) 
s_add_i32 s59, s58, s54 // (op 3805) 


s_and_b32 s54, s50, s59 // (op 3805) 


s_add_i32 s50, s51, s54 // (op 3805) 


s_ashr_i32 s51, s50, s56 // (op 3805) 


s_lshl_b32 s50, s45, 24 // (op 3805) 
s_ashr_i32 s54, s50, 31 // (op 3805) 

s_xor_b32 s50, s51, s54 // (op 3805) 

s_sub_i32 s51, s50, s54 // (op 3805) 


s_and_b32 s50, s45, 31 // (op 3992) 
s_lshl_b32 s54, 1, s50 // (op 3992) 
s_add_i32 s56, s54, s57 // (op 3992) 


s_and_b32 s54, s53, s56 // (op 3992) 


s_add_i32 s53, s55, s54 // (op 3992) 


s_ashr_i32 s54, s53, s50 // (op 3992) 


s_lshl_b32 s50, s45, 24 // (op 3992) 
s_ashr_i32 s53, s50, 31 // (op 3992) 

s_xor_b32 s50, s54, s53 // (op 3992) 

s_sub_i32 s54, s50, s53 // (op 3992) 


s_add_u32 s56, s4, 16 // (op 3805) Start of Int64 add, locking SCC
s_addc_u32 s57, s5, 0 // (op 3805) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 3805) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 3805) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 4 // (op 3805) 

s_mul_i32 s50, s11, s56 // (op 3805) 
s_mul_i32 s53, s8, s50 // (op 3805) 

s_mul_i32 s50, s51, s53 // (op 3805) 

s_sub_u32 s53, s16, s50 // (op 3805) 

s_mov_b32 s50, s53 // (op 3805) 
s_mul_hi_i32 s55, s50, s12 // (op 3805) 
s_add_i32 s58, s55, s50 // (op 3805) 


s_ashr_i32 s50, s58, 31 // (op 3805) 
s_cmp_eq_i32 s12, 0 // (op 3805) 
s_mov_b32 s55, scc // (op 3805) 
s_cmp_eq_i32 s55, 0 // (op 3805) 
s_cselect_b32 s59, 0, -1 // (op 3805) 

s_and_b32 s55, s44, 31 // (op 3805) 
s_lshl_b32 s60, 1, s55 // (op 3805) 
s_add_i32 s61, s60, s59 // (op 3805) 


s_and_b32 s59, s50, s61 // (op 3805) 


s_add_i32 s50, s58, s59 // (op 3805) 


s_ashr_i32 s58, s50, s55 // (op 3805) 


s_lshl_b32 s50, s44, 24 // (op 3805) 
s_ashr_i32 s55, s50, 31 // (op 3805) 

s_xor_b32 s50, s58, s55 // (op 3805) 

s_sub_i32 s58, s50, s55 // (op 3805) 


s_mul_i32 s50, s46, s11 // (op 3805) 
s_add_u32 s60, s6, 256 // (op 3805) Start of Int64 add, locking SCC
s_addc_u32 s61, s7, 0 // (op 3805) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 3805) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 3805) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 8 // (op 3805) 

s_sub_i32 s55, s60, s50 // (op 3805) 

s_mul_i32 s59, s58, s55 // (op 3805) 

s_sub_u32 s55, s53, s59 // (op 3805) 


s_add_u32 s53, s55, s50 // (op 3805) 


s_cmp_eq_i32 s51, 0 // (op 3805) 
s_mov_b32 s50, scc // (op 3805) 

s_cmp_eq_i32 s50, 0 // (op 3805) 
s_cselect_b32 s51, s53, s19 // (op 3805) 


s_mul_i32 s19, s49, s56 // (op 3805) 


v_lshl_add_u32 v8, s51, 8, v6 // (op 3805) 


s_sub_i32 s49, s18, s19 // (op 3805) 


s_cmp_eq_i32 s50, 0 // (op 3805) 
s_cselect_b32 s51, s58, s49 // (op 3805) 



v_lshl_add_u32 v6, s51, 4, v13 // (op 3805) 


v_mul_lo_u32 v9, v6, s2 // (op 3805) 

v_add_lshl_u32 v6, v8, v9, 2 // (op 3805) 


s_add_u32 s18, s4, 16 // (op 3992) Start of Int64 add, locking SCC
s_addc_u32 s19, s5, 0 // (op 3992) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 3992) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 3992) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 4 // (op 3992) 

s_mul_i32 s53, s11, s18 // (op 3992) 
s_mul_i32 s49, s8, s53 // (op 3992) 

s_mul_i32 s53, s54, s49 // (op 3992) 

s_sub_u32 s49, s16, s53 // (op 3992) 

s_mov_b32 s53, s49 // (op 3992) 
s_mul_hi_i32 s50, s53, s12 // (op 3992) 
s_add_i32 s51, s50, s53 // (op 3992) 


s_ashr_i32 s50, s51, 31 // (op 3992) 
s_cmp_eq_i32 s12, 0 // (op 3992) 
s_mov_b32 s53, scc // (op 3992) 
s_cmp_eq_i32 s53, 0 // (op 3992) 
s_cselect_b32 s55, 0, -1 // (op 3992) 

s_and_b32 s53, s44, 31 // (op 3992) 
s_lshl_b32 s56, 1, s53 // (op 3992) 
s_add_i32 s57, s56, s55 // (op 3992) 


s_and_b32 s55, s50, s57 // (op 3992) 


s_add_i32 s50, s51, s55 // (op 3992) 


s_ashr_i32 s51, s50, s53 // (op 3992) 


s_lshl_b32 s50, s44, 24 // (op 3992) 
s_ashr_i32 s53, s50, 31 // (op 3992) 

s_xor_b32 s50, s51, s53 // (op 3992) 

s_sub_i32 s51, s50, s53 // (op 3992) 


s_mul_i32 s50, s46, s11 // (op 3992) 
s_add_u32 s56, s6, 256 // (op 3992) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 3992) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 3992) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 3992) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 8 // (op 3992) 

s_sub_i32 s53, s56, s50 // (op 3992) 

s_mul_i32 s55, s51, s53 // (op 3992) 

s_sub_u32 s53, s49, s55 // (op 3992) 


s_add_u32 s49, s53, s50 // (op 3992) 


s_cmp_eq_i32 s54, 0 // (op 3992) 
s_mov_b32 s50, scc // (op 3992) 

s_cmp_eq_i32 s50, 0 // (op 3992) 
s_cselect_b32 s53, s49, s1 // (op 3992) 


s_mul_i32 s1, s52, s18 // (op 3992) 


v_lshl_add_u32 v8, s53, 8, v3 // (op 3992) 


s_sub_i32 s49, s0, s1 // (op 3992) 


s_cmp_eq_i32 s50, 0 // (op 3992) 
s_cselect_b32 s0, s51, s49 // (op 3992) 



v_lshl_add_u32 v3, s0, 4, v12 // (op 3992) 


v_mul_lo_u32 v9, v3, s20 // (op 3992) 

v_add_lshl_u32 v3, v8, v9, 2 // (op 3992) 


s_mul_i32 s1, s2, s5 // (op 3807) most significant: low of low * high
s_mul_i32 s18, s3, s4 // (op 3807) most significant: low of high * low
s_mul_i32 s0, s2, s4 // (op 3807) least significant
s_mul_hi_u32 s19, s2, s4 // (op 3807) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3807) most significant: sum
s_add_u32 s1, s1, s19 // (op 3807) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 3807) 

s_mul_i32 s1, s20, s5 // (op 3994) most significant: low of low * high
s_mul_i32 s49, s21, s4 // (op 3994) most significant: low of high * low
s_mul_i32 s0, s20, s4 // (op 3994) least significant
s_mul_hi_u32 s50, s20, s4 // (op 3994) most significant: high of low * low
s_add_u32 s1, s1, s49 // (op 3994) most significant: sum
s_add_u32 s1, s1, s50 // (op 3994) most significant: sum


s_lshl_b64 s[50:51], s[0:1], 2 // (op 3994) 

s_mov_b32 s52, s22 // (op 3807) reg expression
s_mov_b32 s53, s23 // (op 3807) reg expression
s_mov_b32 s54, s18 // (op 3807) reg expression
s_mov_b32 s55, 131072 // (op 3807) call()

s_mov_b32 s56, s24 // (op 3994) reg expression

s_mov_b32 s57, s25 // (op 3994) reg expression
s_mov_b32 s58, s50 // (op 3994) reg expression
s_mov_b32 s59, 131072 // (op 3994) call()


v_mov_b32 v8, s47 // (op 2) Move value
v_mov_b32 v9, s48 // (op 39) Move value


v_accvgpr_write a0, 0.00000 // (op 59) call()
v_accvgpr_write a1, 0.00000 // (op 59) call()
v_accvgpr_write a2, 0.00000 // (op 59) call()
v_accvgpr_write a3, 0.00000 // (op 59) call()
v_accvgpr_write a4, 0.00000 // (op 309) call()
v_accvgpr_write a5, 0.00000 // (op 309) call()
v_accvgpr_write a6, 0.00000 // (op 309) call()
v_accvgpr_write a7, 0.00000 // (op 309) call()
v_accvgpr_write a8, 0.00000 // (op 471) call()
v_accvgpr_write a9, 0.00000 // (op 471) call()
v_accvgpr_write a10, 0.00000 // (op 471) call()
v_accvgpr_write a11, 0.00000 // (op 471) call()
v_accvgpr_write a12, 0.00000 // (op 633) call()
v_accvgpr_write a13, 0.00000 // (op 633) call()
v_accvgpr_write a14, 0.00000 // (op 633) call()
v_accvgpr_write a15, 0.00000 // (op 633) call()
v_accvgpr_write a16, 0.00000 // (op 795) call()
v_accvgpr_write a17, 0.00000 // (op 795) call()
v_accvgpr_write a18, 0.00000 // (op 795) call()
v_accvgpr_write a19, 0.00000 // (op 795) call()
v_accvgpr_write a20, 0.00000 // (op 957) call()
v_accvgpr_write a21, 0.00000 // (op 957) call()
v_accvgpr_write a22, 0.00000 // (op 957) call()
v_accvgpr_write a23, 0.00000 // (op 957) call()
v_accvgpr_write a24, 0.00000 // (op 1119) call()
v_accvgpr_write a25, 0.00000 // (op 1119) call()
v_accvgpr_write a26, 0.00000 // (op 1119) call()
v_accvgpr_write a27, 0.00000 // (op 1119) call()
v_accvgpr_write a28, 0.00000 // (op 1281) call()
v_accvgpr_write a29, 0.00000 // (op 1281) call()
v_accvgpr_write a30, 0.00000 // (op 1281) call()
v_accvgpr_write a31, 0.00000 // (op 1281) call()
s_mul_hi_i32 s0, s16, s10 // (op 3572) 
v_lshrrev_b32 v12, 3, v1 // (op 3572) 
s_add_i32 s1, s0, s16 // (op 3572) 

v_and_b32 v13, -8, v1 // (op 3572) 
s_ashr_i32 s0, s1, 31 // (op 3572) 
v_sub_u32 v14, v1, v13 // (op 3572) 

s_cmp_eq_i32 s10, 0 // (op 3572) 
s_mov_b32 s18, scc // (op 3572) 
v_lshlrev_b32 v13, 4, v14 // (op 3572) 

s_cmp_eq_i32 s18, 0 // (op 3572) 
s_cselect_b32 s19, 0, -1 // (op 3572) 

v_mov_b32 v15, 0 // (op 3572) convert
v_mov_b32 v14, v13 // (op 3572) convert

s_and_b32 s18, s13, 31 // (op 3572) 
s_lshl_b32 s22, 1, s18 // (op 3572) 
s_add_i32 s23, s22, s19 // (op 3572) 


s_and_b32 s19, s0, s23 // (op 3572) 


s_add_i32 s0, s1, s19 // (op 3572) 


s_ashr_i32 s1, s0, s18 // (op 3572) 


s_lshl_b32 s0, s13, 24 // (op 3572) 
s_ashr_i32 s18, s0, 31 // (op 3572) 

s_xor_b32 s0, s1, s18 // (op 3572) 

s_sub_i32 s1, s0, s18 // (op 3572) 


s_mul_hi_i32 s0, s1, s14 // (op 3572) 
s_add_i32 s18, s0, s1 // (op 3572) 

s_ashr_i32 s0, s18, 31 // (op 3572) 
s_cmp_eq_i32 s14, 0 // (op 3572) 
s_mov_b32 s19, scc // (op 3572) 
s_cmp_eq_i32 s19, 0 // (op 3572) 
s_cselect_b32 s22, 0, -1 // (op 3572) 

s_and_b32 s19, s15, 31 // (op 3572) 
s_lshl_b32 s23, 1, s19 // (op 3572) 
s_add_i32 s24, s23, s22 // (op 3572) 


s_and_b32 s25, s0, s24 // (op 3572) 


s_add_i32 s0, s18, s25 // (op 3572) 


s_ashr_i32 s18, s0, s19 // (op 3572) 


s_lshl_b32 s0, s15, 24 // (op 3572) 
s_ashr_i32 s19, s0, 31 // (op 3572) 

s_xor_b32 s0, s18, s19 // (op 3572) 

s_sub_i32 s18, s0, s19 // (op 3572) 


s_mul_i32 s0, s1, s11 // (op 3572) 

s_sub_u32 s1, s16, s0 // (op 3572) 

s_mul_i32 s0, s18, s11 // (op 3572) 
s_add_u32 s0, s0, s1 // (op 3572) 


s_mul_hi_i32 s1, s16, s9 // (op 3572) 
s_add_i32 s18, s1, s16 // (op 3572) 

s_ashr_i32 s1, s18, 31 // (op 3572) 
s_cmp_eq_i32 s9, 0 // (op 3572) 
s_mov_b32 s19, scc // (op 3572) 
s_cmp_eq_i32 s19, 0 // (op 3572) 
s_cselect_b32 s22, 0, -1 // (op 3572) 

s_and_b32 s19, s45, 31 // (op 3572) 
s_lshl_b32 s23, 1, s19 // (op 3572) 
s_add_i32 s24, s23, s22 // (op 3572) 


s_and_b32 s25, s1, s24 // (op 3572) 


s_add_i32 s1, s18, s25 // (op 3572) 


s_ashr_i32 s18, s1, s19 // (op 3572) 


s_lshl_b32 s1, s45, 24 // (op 3572) 
s_ashr_i32 s19, s1, 31 // (op 3572) 

s_xor_b32 s1, s18, s19 // (op 3572) 

s_sub_i32 s18, s1, s19 // (op 3572) 


s_add_u32 s22, s4, 16 // (op 3572) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 3572) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 3572) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 3572) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 3572) 

s_mul_i32 s1, s11, s22 // (op 3572) 

s_mul_i32 s19, s8, s1 // (op 3572) 

s_mul_i32 s1, s18, s19 // (op 3572) 

s_sub_u32 s19, s16, s1 // (op 3572) 

s_mov_b32 s1, s19 // (op 3572) 
s_mul_hi_i32 s22, s1, s12 // (op 3572) 
s_add_i32 s23, s22, s1 // (op 3572) 


s_ashr_i32 s1, s23, 31 // (op 3572) 
s_cmp_eq_i32 s12, 0 // (op 3572) 
s_mov_b32 s22, scc // (op 3572) 
s_cmp_eq_i32 s22, 0 // (op 3572) 
s_cselect_b32 s24, 0, -1 // (op 3572) 

s_and_b32 s22, s44, 31 // (op 3572) 
s_lshl_b32 s25, 1, s22 // (op 3572) 
s_add_i32 s47, s25, s24 // (op 3572) 


s_and_b32 s24, s1, s47 // (op 3572) 


s_add_i32 s1, s23, s24 // (op 3572) 


s_ashr_i32 s23, s1, s22 // (op 3572) 


s_lshl_b32 s1, s44, 24 // (op 3572) 
s_ashr_i32 s22, s1, 31 // (op 3572) 

s_xor_b32 s1, s23, s22 // (op 3572) 

s_sub_i32 s23, s1, s22 // (op 3572) 


s_mul_i32 s1, s46, s11 // (op 3572) 
s_add_u32 s24, s6, 256 // (op 3572) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 3572) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 3572) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 3572) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 8 // (op 3572) 

s_sub_i32 s22, s24, s1 // (op 3572) 

s_mul_i32 s24, s23, s22 // (op 3572) 


s_sub_u32 s25, s19, s24 // (op 3572) 


s_add_u32 s19, s25, s1 // (op 3572) 


s_cmp_eq_i32 s18, 0 // (op 3572) 
s_mov_b32 s1, scc // (op 3572) 

s_cmp_eq_i32 s1, 0 // (op 3572) 
s_cselect_b32 s18, s19, s0 // (op 3572) 



v_lshl_add_u32 v13, s18, 8, v12 // (op 3572) 


v_mul_lo_u32 v17, v13, s27 // (op 3572) most significant: low of low * high
v_mul_hi_u32 v12, v13, s26 // (op 3572) most significant: high of low * low
v_mul_lo_u32 v16, v13, s26 // (op 3572) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 3572) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 3572) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 3572) most significant half



s_mul_i32 s1, s26, s7 // (op 3574) most significant: low of low * high
s_mul_i32 s18, s27, s6 // (op 3574) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 3574) least significant
s_mul_hi_u32 s19, s26, s6 // (op 3574) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 3574) most significant: sum
s_add_u32 s1, s1, s19 // (op 3574) most significant: sum


s_mov_b32 s48, s28 // (op 3574) reg expression
s_mov_b32 s49, s29 // (op 3574) reg expression
s_mov_b32 s50, s0 // (op 3574) reg expression
s_mov_b32 s51, 131072 // (op 3574) call()

s_mul_hi_i32 s47, s16, s10 // (op 3591) 
v_lshrrev_b32 v12, 3, v1 // (op 3591) 
s_add_i32 s0, s47, s16 // (op 3591) 

v_and_b32 v13, -8, v1 // (op 3591) 
s_ashr_i32 s1, s0, 31 // (op 3591) 
v_sub_u32 v14, v1, v13 // (op 3591) 

s_cmp_eq_i32 s10, 0 // (op 3591) 
s_mov_b32 s47, scc // (op 3591) 
v_lshlrev_b32 v13, 4, v14 // (op 3591) 

s_cmp_eq_i32 s47, 0 // (op 3591) 
s_cselect_b32 s18, 0, -1 // (op 3591) 

v_mov_b32 v15, 0 // (op 3591) convert
v_mov_b32 v14, v13 // (op 3591) convert

s_and_b32 s19, s13, 31 // (op 3591) 
s_lshl_b32 s47, 1, s19 // (op 3591) 
s_add_i32 s22, s47, s18 // (op 3591) 


s_and_b32 s18, s1, s22 // (op 3591) 


s_add_i32 s1, s0, s18 // (op 3591) 


s_ashr_i32 s0, s1, s19 // (op 3591) 


s_lshl_b32 s1, s13, 24 // (op 3591) 
s_ashr_i32 s47, s1, 31 // (op 3591) 

s_xor_b32 s1, s0, s47 // (op 3591) 

s_sub_i32 s0, s1, s47 // (op 3591) 


s_mul_hi_i32 s1, s0, s14 // (op 3591) 
s_add_i32 s47, s1, s0 // (op 3591) 

s_ashr_i32 s1, s47, 31 // (op 3591) 
s_cmp_eq_i32 s14, 0 // (op 3591) 
s_mov_b32 s18, scc // (op 3591) 
s_cmp_eq_i32 s18, 0 // (op 3591) 
s_cselect_b32 s19, 0, -1 // (op 3591) 

s_and_b32 s18, s15, 31 // (op 3591) 
s_lshl_b32 s22, 1, s18 // (op 3591) 
s_add_i32 s23, s22, s19 // (op 3591) 


s_and_b32 s19, s1, s23 // (op 3591) 


s_add_i32 s1, s47, s19 // (op 3591) 


s_ashr_i32 s19, s1, s18 // (op 3591) 


s_lshl_b32 s1, s15, 24 // (op 3591) 
s_ashr_i32 s18, s1, 31 // (op 3591) 

s_xor_b32 s1, s19, s18 // (op 3591) 

s_sub_i32 s19, s1, s18 // (op 3591) 


s_mul_i32 s1, s0, s11 // (op 3591) 

s_sub_u32 s0, s16, s1 // (op 3591) 

s_mul_i32 s1, s19, s11 // (op 3591) 
s_add_u32 s1, s1, s0 // (op 3591) 


s_mul_hi_i32 s0, s16, s9 // (op 3591) 
s_add_i32 s47, s0, s16 // (op 3591) 

s_ashr_i32 s0, s47, 31 // (op 3591) 
s_cmp_eq_i32 s9, 0 // (op 3591) 
s_mov_b32 s18, scc // (op 3591) 
s_cmp_eq_i32 s18, 0 // (op 3591) 
s_cselect_b32 s19, 0, -1 // (op 3591) 

s_and_b32 s18, s45, 31 // (op 3591) 
s_lshl_b32 s22, 1, s18 // (op 3591) 
s_add_i32 s23, s22, s19 // (op 3591) 


s_and_b32 s19, s0, s23 // (op 3591) 


s_add_i32 s0, s47, s19 // (op 3591) 


s_ashr_i32 s19, s0, s18 // (op 3591) 


s_lshl_b32 s0, s45, 24 // (op 3591) 
s_ashr_i32 s18, s0, 31 // (op 3591) 

s_xor_b32 s0, s19, s18 // (op 3591) 

s_sub_i32 s19, s0, s18 // (op 3591) 


s_add_u32 s22, s4, 16 // (op 3591) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 3591) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 3591) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 3591) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 3591) 

s_mul_i32 s0, s11, s22 // (op 3591) 

s_mul_i32 s18, s8, s0 // (op 3591) 

s_mul_i32 s0, s19, s18 // (op 3591) 

s_sub_u32 s18, s16, s0 // (op 3591) 

s_mov_b32 s0, s18 // (op 3591) 
s_mul_hi_i32 s47, s0, s12 // (op 3591) 
s_add_i32 s22, s47, s0 // (op 3591) 


s_ashr_i32 s0, s22, 31 // (op 3591) 
s_cmp_eq_i32 s12, 0 // (op 3591) 
s_mov_b32 s47, scc // (op 3591) 
s_cmp_eq_i32 s47, 0 // (op 3591) 
s_cselect_b32 s23, 0, -1 // (op 3591) 

s_and_b32 s47, s44, 31 // (op 3591) 
s_lshl_b32 s24, 1, s47 // (op 3591) 
s_add_i32 s25, s24, s23 // (op 3591) 


s_and_b32 s23, s0, s25 // (op 3591) 


s_add_i32 s0, s22, s23 // (op 3591) 


s_ashr_i32 s22, s0, s47 // (op 3591) 


s_lshl_b32 s0, s44, 24 // (op 3591) 
s_ashr_i32 s47, s0, 31 // (op 3591) 

s_xor_b32 s0, s22, s47 // (op 3591) 

s_sub_i32 s22, s0, s47 // (op 3591) 


s_mul_i32 s0, s46, s11 // (op 3591) 
s_add_u32 s24, s6, 256 // (op 3591) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 3591) End of Int64 add, unlocking SCC
s_sub_u32 s60, s24, 1 // (op 3591) least significant half; sets scc
s_subb_u32 s61, s25, 0 // (op 3591) most significant half; uses scc

s_ashr_i64 s[24:25], s[60:61], 8 // (op 3591) 

s_sub_i32 s23, s24, s0 // (op 3591) 

s_mul_i32 s47, s22, s23 // (op 3591) 


s_sub_u32 s22, s18, s47 // (op 3591) 


s_add_u32 s18, s22, s0 // (op 3591) 


s_cmp_eq_i32 s19, 0 // (op 3591) 
s_mov_b32 s0, scc // (op 3591) 

s_cmp_eq_i32 s0, 0 // (op 3591) 
s_cselect_b32 s19, s18, s1 // (op 3591) 



v_lshl_add_u32 v13, s19, 8, v12 // (op 3591) 


v_mul_lo_u32 v19, v13, s27 // (op 3591) most significant: low of low * high
v_mul_hi_u32 v12, v13, s26 // (op 3591) most significant: high of low * low
v_mul_lo_u32 v18, v13, s26 // (op 3591) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 3591) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v14 // (op 3591) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 3591) most significant half



s_mul_i32 s1, s26, s7 // (op 3593) most significant: low of low * high
s_mul_i32 s47, s27, s6 // (op 3593) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 3593) least significant
s_mul_hi_u32 s18, s26, s6 // (op 3593) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 3593) most significant: sum
s_add_u32 s1, s1, s18 // (op 3593) most significant: sum


s_mov_b32 s48, s28 // (op 3593) reg expression
s_mov_b32 s49, s29 // (op 3593) reg expression
s_mov_b32 s50, s0 // (op 3593) reg expression
s_mov_b32 s51, 131072 // (op 3593) call()

s_mul_hi_i32 s47, s16, s10 // (op 3636) 
v_lshrrev_b32 v12, 3, v1 // (op 3636) 
s_add_i32 s0, s47, s16 // (op 3636) 

v_and_b32 v13, -8, v1 // (op 3636) 
s_ashr_i32 s1, s0, 31 // (op 3636) 
v_sub_u32 v14, v1, v13 // (op 3636) 

s_cmp_eq_i32 s10, 0 // (op 3636) 
s_mov_b32 s47, scc // (op 3636) 
v_lshlrev_b32 v13, 4, v14 // (op 3636) 

s_cmp_eq_i32 s47, 0 // (op 3636) 
s_cselect_b32 s18, 0, -1 // (op 3636) 

v_mov_b32 v15, 0 // (op 3636) convert
v_mov_b32 v14, v13 // (op 3636) convert

s_and_b32 s19, s13, 31 // (op 3636) 
s_lshl_b32 s47, 1, s19 // (op 3636) 
s_add_i32 s22, s47, s18 // (op 3636) 


s_and_b32 s18, s1, s22 // (op 3636) 


s_add_i32 s1, s0, s18 // (op 3636) 


s_ashr_i32 s0, s1, s19 // (op 3636) 


s_lshl_b32 s1, s13, 24 // (op 3636) 
s_ashr_i32 s47, s1, 31 // (op 3636) 

s_xor_b32 s1, s0, s47 // (op 3636) 

s_sub_i32 s0, s1, s47 // (op 3636) 


s_mul_hi_i32 s1, s0, s14 // (op 3636) 
s_add_i32 s47, s1, s0 // (op 3636) 

s_ashr_i32 s1, s47, 31 // (op 3636) 
s_cmp_eq_i32 s14, 0 // (op 3636) 
s_mov_b32 s18, scc // (op 3636) 
s_cmp_eq_i32 s18, 0 // (op 3636) 
s_cselect_b32 s19, 0, -1 // (op 3636) 

s_and_b32 s18, s15, 31 // (op 3636) 
s_lshl_b32 s22, 1, s18 // (op 3636) 
s_add_i32 s23, s22, s19 // (op 3636) 


s_and_b32 s19, s1, s23 // (op 3636) 


s_add_i32 s1, s47, s19 // (op 3636) 


s_ashr_i32 s19, s1, s18 // (op 3636) 


s_lshl_b32 s1, s15, 24 // (op 3636) 
s_ashr_i32 s18, s1, 31 // (op 3636) 

s_xor_b32 s1, s19, s18 // (op 3636) 

s_sub_i32 s19, s1, s18 // (op 3636) 


s_add_u32 s22, s4, 16 // (op 3636) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 3636) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 3636) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 3636) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 3636) 

s_mul_i32 s1, s19, s22 // (op 3636) 

s_sub_i32 s47, s0, s1 // (op 3636) 


s_mul_hi_i32 s0, s16, s9 // (op 3636) 
s_add_i32 s1, s0, s16 // (op 3636) 

s_ashr_i32 s0, s1, 31 // (op 3636) 
s_cmp_eq_i32 s9, 0 // (op 3636) 
s_mov_b32 s18, scc // (op 3636) 
s_cmp_eq_i32 s18, 0 // (op 3636) 
s_cselect_b32 s19, 0, -1 // (op 3636) 

s_and_b32 s18, s45, 31 // (op 3636) 
s_lshl_b32 s24, 1, s18 // (op 3636) 
s_add_i32 s25, s24, s19 // (op 3636) 


s_and_b32 s19, s0, s25 // (op 3636) 


s_add_i32 s0, s1, s19 // (op 3636) 


s_ashr_i32 s1, s0, s18 // (op 3636) 


s_lshl_b32 s0, s45, 24 // (op 3636) 
s_ashr_i32 s18, s0, 31 // (op 3636) 

s_xor_b32 s0, s1, s18 // (op 3636) 

s_sub_i32 s1, s0, s18 // (op 3636) 


s_mul_i32 s0, s11, s22 // (op 3636) 

s_mul_i32 s18, s8, s0 // (op 3636) 

s_mul_i32 s0, s1, s18 // (op 3636) 

s_sub_u32 s18, s16, s0 // (op 3636) 

s_mul_hi_i32 s0, s18, s12 // (op 3636) 
s_add_i32 s19, s0, s18 // (op 3636) 


s_ashr_i32 s0, s19, 31 // (op 3636) 
s_cmp_eq_i32 s12, 0 // (op 3636) 
s_mov_b32 s18, scc // (op 3636) 
s_cmp_eq_i32 s18, 0 // (op 3636) 
s_cselect_b32 s22, 0, -1 // (op 3636) 

s_and_b32 s18, s44, 31 // (op 3636) 
s_lshl_b32 s23, 1, s18 // (op 3636) 
s_add_i32 s24, s23, s22 // (op 3636) 


s_and_b32 s25, s0, s24 // (op 3636) 


s_add_i32 s0, s19, s25 // (op 3636) 


s_ashr_i32 s19, s0, s18 // (op 3636) 


s_lshl_b32 s0, s44, 24 // (op 3636) 
s_ashr_i32 s18, s0, 31 // (op 3636) 

s_xor_b32 s0, s19, s18 // (op 3636) 

s_sub_i32 s19, s0, s18 // (op 3636) 


s_cmp_eq_i32 s1, 0 // (op 3636) 
s_mov_b32 s0, scc // (op 3636) 

s_cmp_eq_i32 s0, 0 // (op 3636) 
s_cselect_b32 s1, s19, s47 // (op 3636) 



v_lshl_add_u32 v13, s1, 4, v12 // (op 3636) 


v_mul_lo_u32 v21, v13, s31 // (op 3636) most significant: low of low * high
v_mul_hi_u32 v12, v13, s30 // (op 3636) most significant: high of low * low
v_mul_lo_u32 v20, v13, s30 // (op 3636) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 3636) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v20 // (op 3636) least significant half
v_addc_co_u32 v13, s[0:1], v15, v21, s[0:1] // (op 3636) most significant half



s_mul_i32 s1, s30, s5 // (op 3638) most significant: low of low * high
s_mul_i32 s47, s31, s4 // (op 3638) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 3638) least significant
s_mul_hi_u32 s18, s30, s4 // (op 3638) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 3638) most significant: sum
s_add_u32 s1, s1, s18 // (op 3638) most significant: sum


s_mov_b32 s60, s32 // (op 3638) reg expression
s_mov_b32 s61, s33 // (op 3638) reg expression
s_mov_b32 s62, s0 // (op 3638) reg expression
s_mov_b32 s63, 131072 // (op 3638) call()

s_mul_hi_i32 s47, s16, s10 // (op 3655) 
v_lshrrev_b32 v14, 3, v1 // (op 3655) 
s_add_i32 s0, s47, s16 // (op 3655) 

v_and_b32 v15, -8, v1 // (op 3655) 
s_ashr_i32 s1, s0, 31 // (op 3655) 
v_sub_u32 v20, v1, v15 // (op 3655) 

s_cmp_eq_i32 s10, 0 // (op 3655) 
s_mov_b32 s47, scc // (op 3655) 
v_lshlrev_b32 v15, 4, v20 // (op 3655) 

s_cmp_eq_i32 s47, 0 // (op 3655) 
s_cselect_b32 s18, 0, -1 // (op 3655) 

v_mov_b32 v21, 0 // (op 3655) convert
v_mov_b32 v20, v15 // (op 3655) convert

s_and_b32 s19, s13, 31 // (op 3655) 
s_lshl_b32 s47, 1, s19 // (op 3655) 
s_add_i32 s22, s47, s18 // (op 3655) 


s_and_b32 s18, s1, s22 // (op 3655) 


s_add_i32 s1, s0, s18 // (op 3655) 


s_ashr_i32 s0, s1, s19 // (op 3655) 


s_lshl_b32 s1, s13, 24 // (op 3655) 
s_ashr_i32 s47, s1, 31 // (op 3655) 

s_xor_b32 s1, s0, s47 // (op 3655) 

s_sub_i32 s0, s1, s47 // (op 3655) 


s_mul_hi_i32 s1, s0, s14 // (op 3655) 
s_add_i32 s47, s1, s0 // (op 3655) 

s_ashr_i32 s1, s47, 31 // (op 3655) 
s_cmp_eq_i32 s14, 0 // (op 3655) 
s_mov_b32 s18, scc // (op 3655) 
s_cmp_eq_i32 s18, 0 // (op 3655) 
s_cselect_b32 s19, 0, -1 // (op 3655) 

s_and_b32 s18, s15, 31 // (op 3655) 
s_lshl_b32 s22, 1, s18 // (op 3655) 
s_add_i32 s23, s22, s19 // (op 3655) 


s_and_b32 s19, s1, s23 // (op 3655) 


s_add_i32 s1, s47, s19 // (op 3655) 


s_ashr_i32 s19, s1, s18 // (op 3655) 


s_lshl_b32 s1, s15, 24 // (op 3655) 
s_ashr_i32 s18, s1, 31 // (op 3655) 

s_xor_b32 s1, s19, s18 // (op 3655) 

s_sub_i32 s19, s1, s18 // (op 3655) 


s_add_u32 s22, s4, 16 // (op 3655) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 3655) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 3655) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 3655) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 3655) 

s_mul_i32 s1, s19, s22 // (op 3655) 

s_sub_i32 s47, s0, s1 // (op 3655) 


s_mul_hi_i32 s0, s16, s9 // (op 3655) 
s_add_i32 s1, s0, s16 // (op 3655) 

s_ashr_i32 s0, s1, 31 // (op 3655) 
s_cmp_eq_i32 s9, 0 // (op 3655) 
s_mov_b32 s18, scc // (op 3655) 
s_cmp_eq_i32 s18, 0 // (op 3655) 
s_cselect_b32 s19, 0, -1 // (op 3655) 

s_and_b32 s18, s45, 31 // (op 3655) 
s_lshl_b32 s24, 1, s18 // (op 3655) 
s_add_i32 s25, s24, s19 // (op 3655) 


s_and_b32 s19, s0, s25 // (op 3655) 


s_add_i32 s0, s1, s19 // (op 3655) 


s_ashr_i32 s1, s0, s18 // (op 3655) 


s_lshl_b32 s0, s45, 24 // (op 3655) 
s_ashr_i32 s18, s0, 31 // (op 3655) 

s_xor_b32 s0, s1, s18 // (op 3655) 

s_sub_i32 s1, s0, s18 // (op 3655) 


s_mul_i32 s0, s11, s22 // (op 3655) 

s_mul_i32 s18, s8, s0 // (op 3655) 

s_mul_i32 s0, s1, s18 // (op 3655) 

s_sub_u32 s18, s16, s0 // (op 3655) 

s_mul_hi_i32 s0, s18, s12 // (op 3655) 
s_add_i32 s19, s0, s18 // (op 3655) 


s_ashr_i32 s0, s19, 31 // (op 3655) 
s_cmp_eq_i32 s12, 0 // (op 3655) 
s_mov_b32 s18, scc // (op 3655) 
s_cmp_eq_i32 s18, 0 // (op 3655) 
s_cselect_b32 s22, 0, -1 // (op 3655) 

s_and_b32 s18, s44, 31 // (op 3655) 
s_lshl_b32 s23, 1, s18 // (op 3655) 
s_add_i32 s24, s23, s22 // (op 3655) 


s_and_b32 s25, s0, s24 // (op 3655) 


s_add_i32 s0, s19, s25 // (op 3655) 


s_ashr_i32 s19, s0, s18 // (op 3655) 


s_lshl_b32 s0, s44, 24 // (op 3655) 
s_ashr_i32 s18, s0, 31 // (op 3655) 

s_xor_b32 s0, s19, s18 // (op 3655) 

s_sub_i32 s19, s0, s18 // (op 3655) 


s_cmp_eq_i32 s1, 0 // (op 3655) 
s_mov_b32 s0, scc // (op 3655) 

s_cmp_eq_i32 s0, 0 // (op 3655) 
s_cselect_b32 s1, s19, s47 // (op 3655) 



v_lshl_add_u32 v15, s1, 4, v14 // (op 3655) 


v_mul_lo_u32 v23, v15, s31 // (op 3655) most significant: low of low * high
v_mul_hi_u32 v14, v15, s30 // (op 3655) most significant: high of low * low
v_mul_lo_u32 v22, v15, s30 // (op 3655) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 3655) most significant: sum


v_add_co_u32 v14, s[0:1], v20, v22 // (op 3655) least significant half
v_addc_co_u32 v15, s[0:1], v21, v23, s[0:1] // (op 3655) most significant half



s_mul_i32 s1, s30, s5 // (op 3657) most significant: low of low * high
s_mul_i32 s47, s31, s4 // (op 3657) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 3657) least significant
s_mul_hi_u32 s18, s30, s4 // (op 3657) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 3657) most significant: sum
s_add_u32 s1, s1, s18 // (op 3657) most significant: sum


s_mov_b32 s60, s32 // (op 3657) reg expression
s_mov_b32 s61, s33 // (op 3657) reg expression
s_mov_b32 s62, s0 // (op 3657) reg expression
s_mov_b32 s63, 131072 // (op 3657) call()

s_mul_hi_i32 s47, s16, s10 // (op 3700) 
v_lshlrev_b32 v20, 1, v1 // (op 3700) 
s_add_i32 s0, s47, s16 // (op 3700) 

s_ashr_i32 s1, s0, 31 // (op 3700) 
s_cmp_eq_i32 s10, 0 // (op 3700) 
s_mov_b32 s47, scc // (op 3700) 
s_cmp_eq_i32 s47, 0 // (op 3700) 
s_cselect_b32 s18, 0, -1 // (op 3700) 

s_and_b32 s19, s13, 31 // (op 3700) 
s_lshl_b32 s47, 1, s19 // (op 3700) 
s_add_i32 s22, s47, s18 // (op 3700) 


s_and_b32 s18, s1, s22 // (op 3700) 


s_add_i32 s1, s0, s18 // (op 3700) 


s_ashr_i32 s0, s1, s19 // (op 3700) 


s_lshl_b32 s1, s13, 24 // (op 3700) 
s_ashr_i32 s47, s1, 31 // (op 3700) 

s_xor_b32 s1, s0, s47 // (op 3700) 

s_sub_i32 s0, s1, s47 // (op 3700) 


s_mul_hi_i32 s1, s0, s14 // (op 3700) 
s_add_i32 s47, s1, s0 // (op 3700) 

s_ashr_i32 s1, s47, 31 // (op 3700) 
s_cmp_eq_i32 s14, 0 // (op 3700) 
s_mov_b32 s18, scc // (op 3700) 
s_cmp_eq_i32 s18, 0 // (op 3700) 
s_cselect_b32 s19, 0, -1 // (op 3700) 

s_and_b32 s18, s15, 31 // (op 3700) 
s_lshl_b32 s22, 1, s18 // (op 3700) 
s_add_i32 s23, s22, s19 // (op 3700) 


s_and_b32 s19, s1, s23 // (op 3700) 


s_add_i32 s1, s47, s19 // (op 3700) 


s_ashr_i32 s19, s1, s18 // (op 3700) 


s_lshl_b32 s1, s15, 24 // (op 3700) 
s_ashr_i32 s18, s1, 31 // (op 3700) 

s_xor_b32 s1, s19, s18 // (op 3700) 

s_sub_i32 s19, s1, s18 // (op 3700) 


s_mul_i32 s1, s0, s11 // (op 3700) 

s_sub_u32 s0, s16, s1 // (op 3700) 

s_mul_i32 s1, s19, s11 // (op 3700) 
s_add_u32 s1, s1, s0 // (op 3700) 


s_mul_hi_i32 s0, s16, s9 // (op 3700) 
s_add_i32 s47, s0, s16 // (op 3700) 

s_ashr_i32 s0, s47, 31 // (op 3700) 
s_cmp_eq_i32 s9, 0 // (op 3700) 
s_mov_b32 s18, scc // (op 3700) 
s_cmp_eq_i32 s18, 0 // (op 3700) 
s_cselect_b32 s19, 0, -1 // (op 3700) 

s_and_b32 s18, s45, 31 // (op 3700) 
s_lshl_b32 s22, 1, s18 // (op 3700) 
s_add_i32 s23, s22, s19 // (op 3700) 


s_and_b32 s19, s0, s23 // (op 3700) 


s_add_i32 s0, s47, s19 // (op 3700) 


s_ashr_i32 s19, s0, s18 // (op 3700) 


s_lshl_b32 s0, s45, 24 // (op 3700) 
s_ashr_i32 s18, s0, 31 // (op 3700) 

s_xor_b32 s0, s19, s18 // (op 3700) 

s_sub_i32 s19, s0, s18 // (op 3700) 


s_add_u32 s22, s4, 16 // (op 3700) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 3700) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 3700) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 3700) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 3700) 

s_mul_i32 s0, s11, s22 // (op 3700) 

s_mul_i32 s18, s8, s0 // (op 3700) 

s_mul_i32 s0, s19, s18 // (op 3700) 

s_sub_u32 s18, s16, s0 // (op 3700) 

s_mov_b32 s0, s18 // (op 3700) 
s_mul_hi_i32 s47, s0, s12 // (op 3700) 
s_add_i32 s22, s47, s0 // (op 3700) 


s_ashr_i32 s0, s22, 31 // (op 3700) 
s_cmp_eq_i32 s12, 0 // (op 3700) 
s_mov_b32 s47, scc // (op 3700) 
s_cmp_eq_i32 s47, 0 // (op 3700) 
s_cselect_b32 s23, 0, -1 // (op 3700) 

s_and_b32 s47, s44, 31 // (op 3700) 
s_lshl_b32 s24, 1, s47 // (op 3700) 
s_add_i32 s25, s24, s23 // (op 3700) 


s_and_b32 s23, s0, s25 // (op 3700) 


s_add_i32 s0, s22, s23 // (op 3700) 


s_ashr_i32 s22, s0, s47 // (op 3700) 


s_lshl_b32 s0, s44, 24 // (op 3700) 
s_ashr_i32 s47, s0, 31 // (op 3700) 

s_xor_b32 s0, s22, s47 // (op 3700) 

s_sub_i32 s22, s0, s47 // (op 3700) 


s_mul_i32 s0, s46, s11 // (op 3700) 
s_add_u32 s24, s6, 256 // (op 3700) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 3700) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 3700) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 3700) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 8 // (op 3700) 

s_sub_i32 s23, s24, s0 // (op 3700) 

s_mul_i32 s47, s22, s23 // (op 3700) 


s_sub_u32 s22, s18, s47 // (op 3700) 


s_add_u32 s18, s22, s0 // (op 3700) 


s_cmp_eq_i32 s19, 0 // (op 3700) 
s_mov_b32 s0, scc // (op 3700) 

s_cmp_eq_i32 s0, 0 // (op 3700) 
s_cselect_b32 s19, s18, s1 // (op 3700) 



v_lshl_add_u32 v21, s19, 8, v20 // (op 3700) 


v_mul_lo_u32 v20, v21, s34 // (op 3700) 

s_mul_i32 s1, s34, s7 // (op 3702) most significant: low of low * high
s_mul_i32 s47, s35, s6 // (op 3702) most significant: low of high * low
s_mul_i32 s0, s34, s6 // (op 3702) least significant
s_mul_hi_u32 s18, s34, s6 // (op 3702) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 3702) most significant: sum
s_add_u32 s1, s1, s18 // (op 3702) most significant: sum


s_mov_b32 s64, s36 // (op 3702) reg expression
s_mov_b32 s65, s37 // (op 3702) reg expression
s_mov_b32 s66, s0 // (op 3702) reg expression
s_mov_b32 s67, 131072 // (op 3702) call()

s_mul_hi_i32 s47, s16, s10 // (op 3719) 
v_lshlrev_b32 v21, 1, v1 // (op 3719) 
s_add_i32 s0, s47, s16 // (op 3719) 

s_ashr_i32 s1, s0, 31 // (op 3719) 
s_cmp_eq_i32 s10, 0 // (op 3719) 
s_mov_b32 s47, scc // (op 3719) 
s_cmp_eq_i32 s47, 0 // (op 3719) 
s_cselect_b32 s18, 0, -1 // (op 3719) 

s_and_b32 s19, s13, 31 // (op 3719) 
s_lshl_b32 s47, 1, s19 // (op 3719) 
s_add_i32 s22, s47, s18 // (op 3719) 


s_and_b32 s18, s1, s22 // (op 3719) 


s_add_i32 s1, s0, s18 // (op 3719) 


s_ashr_i32 s0, s1, s19 // (op 3719) 


s_lshl_b32 s1, s13, 24 // (op 3719) 
s_ashr_i32 s47, s1, 31 // (op 3719) 

s_xor_b32 s1, s0, s47 // (op 3719) 

s_sub_i32 s0, s1, s47 // (op 3719) 


s_mul_hi_i32 s1, s0, s14 // (op 3719) 
s_add_i32 s47, s1, s0 // (op 3719) 

s_ashr_i32 s1, s47, 31 // (op 3719) 
s_cmp_eq_i32 s14, 0 // (op 3719) 
s_mov_b32 s18, scc // (op 3719) 
s_cmp_eq_i32 s18, 0 // (op 3719) 
s_cselect_b32 s19, 0, -1 // (op 3719) 

s_and_b32 s18, s15, 31 // (op 3719) 
s_lshl_b32 s22, 1, s18 // (op 3719) 
s_add_i32 s23, s22, s19 // (op 3719) 


s_and_b32 s19, s1, s23 // (op 3719) 


s_add_i32 s1, s47, s19 // (op 3719) 


s_ashr_i32 s19, s1, s18 // (op 3719) 


s_lshl_b32 s1, s15, 24 // (op 3719) 
s_ashr_i32 s18, s1, 31 // (op 3719) 

s_xor_b32 s1, s19, s18 // (op 3719) 

s_sub_i32 s19, s1, s18 // (op 3719) 


s_mul_i32 s1, s0, s11 // (op 3719) 

s_sub_u32 s0, s16, s1 // (op 3719) 

s_mul_i32 s1, s19, s11 // (op 3719) 
s_add_u32 s1, s1, s0 // (op 3719) 


s_mul_hi_i32 s0, s16, s9 // (op 3719) 
s_add_i32 s47, s0, s16 // (op 3719) 

s_ashr_i32 s0, s47, 31 // (op 3719) 
s_cmp_eq_i32 s9, 0 // (op 3719) 
s_mov_b32 s18, scc // (op 3719) 
s_cmp_eq_i32 s18, 0 // (op 3719) 
s_cselect_b32 s19, 0, -1 // (op 3719) 

s_and_b32 s18, s45, 31 // (op 3719) 
s_lshl_b32 s22, 1, s18 // (op 3719) 
s_add_i32 s23, s22, s19 // (op 3719) 


s_and_b32 s19, s0, s23 // (op 3719) 


s_add_i32 s0, s47, s19 // (op 3719) 


s_ashr_i32 s19, s0, s18 // (op 3719) 


s_lshl_b32 s0, s45, 24 // (op 3719) 
s_ashr_i32 s18, s0, 31 // (op 3719) 

s_xor_b32 s0, s19, s18 // (op 3719) 

s_sub_i32 s19, s0, s18 // (op 3719) 


s_add_u32 s22, s4, 16 // (op 3719) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 3719) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 3719) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 3719) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 3719) 

s_mul_i32 s0, s11, s22 // (op 3719) 

s_mul_i32 s18, s8, s0 // (op 3719) 

s_mul_i32 s0, s19, s18 // (op 3719) 

s_sub_u32 s18, s16, s0 // (op 3719) 

s_mov_b32 s0, s18 // (op 3719) 
s_mul_hi_i32 s47, s0, s12 // (op 3719) 
s_add_i32 s22, s47, s0 // (op 3719) 


s_ashr_i32 s0, s22, 31 // (op 3719) 
s_cmp_eq_i32 s12, 0 // (op 3719) 
s_mov_b32 s47, scc // (op 3719) 
s_cmp_eq_i32 s47, 0 // (op 3719) 
s_cselect_b32 s23, 0, -1 // (op 3719) 

s_and_b32 s47, s44, 31 // (op 3719) 
s_lshl_b32 s24, 1, s47 // (op 3719) 
s_add_i32 s25, s24, s23 // (op 3719) 


s_and_b32 s23, s0, s25 // (op 3719) 


s_add_i32 s0, s22, s23 // (op 3719) 


s_ashr_i32 s22, s0, s47 // (op 3719) 


s_lshl_b32 s0, s44, 24 // (op 3719) 
s_ashr_i32 s47, s0, 31 // (op 3719) 

s_xor_b32 s0, s22, s47 // (op 3719) 

s_sub_i32 s22, s0, s47 // (op 3719) 


s_mul_i32 s0, s46, s11 // (op 3719) 
s_add_u32 s24, s6, 256 // (op 3719) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 3719) End of Int64 add, unlocking SCC
s_sub_u32 s68, s24, 1 // (op 3719) least significant half; sets scc
s_subb_u32 s69, s25, 0 // (op 3719) most significant half; uses scc

s_ashr_i64 s[24:25], s[68:69], 8 // (op 3719) 

s_sub_i32 s23, s24, s0 // (op 3719) 

s_mul_i32 s47, s22, s23 // (op 3719) 


s_sub_u32 s22, s18, s47 // (op 3719) 


s_add_u32 s18, s22, s0 // (op 3719) 


s_cmp_eq_i32 s19, 0 // (op 3719) 
s_mov_b32 s0, scc // (op 3719) 

s_cmp_eq_i32 s0, 0 // (op 3719) 
s_cselect_b32 s19, s18, s1 // (op 3719) 



v_lshl_add_u32 v22, s19, 8, v21 // (op 3719) 


v_mul_lo_u32 v21, v22, s34 // (op 3719) 

s_mul_i32 s1, s34, s7 // (op 3721) most significant: low of low * high
s_mul_i32 s47, s35, s6 // (op 3721) most significant: low of high * low
s_mul_i32 s0, s34, s6 // (op 3721) least significant
s_mul_hi_u32 s18, s34, s6 // (op 3721) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 3721) most significant: sum
s_add_u32 s1, s1, s18 // (op 3721) most significant: sum


s_mov_b32 s64, s36 // (op 3721) reg expression
s_mov_b32 s65, s37 // (op 3721) reg expression
s_mov_b32 s66, s0 // (op 3721) reg expression
s_mov_b32 s67, 131072 // (op 3721) call()

s_mul_hi_i32 s47, s16, s10 // (op 3764) 
v_and_b32 v22, 15, v1 // (op 3764) 
s_add_i32 s0, s47, s16 // (op 3764) 

v_and_b32 v23, 63, v1 // (op 3764) 
s_ashr_i32 s1, s0, 31 // (op 3764) 
v_lshrrev_b32 v24, 4, v23 // (op 3764) 

s_cmp_eq_i32 s10, 0 // (op 3764) 
s_mov_b32 s47, scc // (op 3764) 
s_cmp_eq_i32 s47, 0 // (op 3764) 
s_cselect_b32 s18, 0, -1 // (op 3764) 

s_and_b32 s19, s13, 31 // (op 3764) 
s_lshl_b32 s47, 1, s19 // (op 3764) 
s_add_i32 s22, s47, s18 // (op 3764) 


s_and_b32 s18, s1, s22 // (op 3764) 


s_add_i32 s1, s0, s18 // (op 3764) 


s_ashr_i32 s0, s1, s19 // (op 3764) 


s_lshl_b32 s1, s13, 24 // (op 3764) 
s_ashr_i32 s47, s1, 31 // (op 3764) 

s_xor_b32 s1, s0, s47 // (op 3764) 

s_sub_i32 s0, s1, s47 // (op 3764) 


s_mul_hi_i32 s1, s0, s14 // (op 3764) 
s_add_i32 s47, s1, s0 // (op 3764) 

s_ashr_i32 s1, s47, 31 // (op 3764) 
s_cmp_eq_i32 s14, 0 // (op 3764) 
s_mov_b32 s18, scc // (op 3764) 
s_cmp_eq_i32 s18, 0 // (op 3764) 
s_cselect_b32 s19, 0, -1 // (op 3764) 

s_and_b32 s18, s15, 31 // (op 3764) 
s_lshl_b32 s22, 1, s18 // (op 3764) 
s_add_i32 s23, s22, s19 // (op 3764) 


s_and_b32 s19, s1, s23 // (op 3764) 


s_add_i32 s1, s47, s19 // (op 3764) 


s_ashr_i32 s19, s1, s18 // (op 3764) 


s_lshl_b32 s1, s15, 24 // (op 3764) 
s_ashr_i32 s18, s1, 31 // (op 3764) 

s_xor_b32 s1, s19, s18 // (op 3764) 

s_sub_i32 s19, s1, s18 // (op 3764) 


s_add_u32 s22, s4, 16 // (op 3764) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 3764) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 3764) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 3764) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 3764) 

s_mul_i32 s1, s19, s22 // (op 3764) 

s_sub_i32 s47, s0, s1 // (op 3764) 


s_mul_hi_i32 s0, s16, s9 // (op 3764) 
s_add_i32 s1, s0, s16 // (op 3764) 

s_ashr_i32 s0, s1, 31 // (op 3764) 
s_cmp_eq_i32 s9, 0 // (op 3764) 
s_mov_b32 s18, scc // (op 3764) 
s_cmp_eq_i32 s18, 0 // (op 3764) 
s_cselect_b32 s19, 0, -1 // (op 3764) 

s_and_b32 s18, s45, 31 // (op 3764) 
s_lshl_b32 s24, 1, s18 // (op 3764) 
s_add_i32 s25, s24, s19 // (op 3764) 


s_and_b32 s19, s0, s25 // (op 3764) 


s_add_i32 s0, s1, s19 // (op 3764) 


s_ashr_i32 s1, s0, s18 // (op 3764) 


s_lshl_b32 s0, s45, 24 // (op 3764) 
s_ashr_i32 s18, s0, 31 // (op 3764) 

s_xor_b32 s0, s1, s18 // (op 3764) 

s_sub_i32 s1, s0, s18 // (op 3764) 


s_mul_i32 s0, s11, s22 // (op 3764) 

s_mul_i32 s18, s8, s0 // (op 3764) 

s_mul_i32 s0, s1, s18 // (op 3764) 

s_sub_u32 s18, s16, s0 // (op 3764) 

s_mul_hi_i32 s0, s18, s12 // (op 3764) 
s_add_i32 s19, s0, s18 // (op 3764) 


s_ashr_i32 s0, s19, 31 // (op 3764) 
s_cmp_eq_i32 s12, 0 // (op 3764) 
s_mov_b32 s18, scc // (op 3764) 
s_cmp_eq_i32 s18, 0 // (op 3764) 
s_cselect_b32 s22, 0, -1 // (op 3764) 

s_and_b32 s18, s44, 31 // (op 3764) 
s_lshl_b32 s23, 1, s18 // (op 3764) 
s_add_i32 s24, s23, s22 // (op 3764) 


s_and_b32 s25, s0, s24 // (op 3764) 


s_add_i32 s0, s19, s25 // (op 3764) 


s_ashr_i32 s19, s0, s18 // (op 3764) 


s_lshl_b32 s0, s44, 24 // (op 3764) 
s_ashr_i32 s18, s0, 31 // (op 3764) 

s_xor_b32 s0, s19, s18 // (op 3764) 

s_sub_i32 s19, s0, s18 // (op 3764) 


s_cmp_eq_i32 s1, 0 // (op 3764) 
s_mov_b32 s0, scc // (op 3764) 

s_cmp_eq_i32 s0, 0 // (op 3764) 
s_cselect_b32 s1, s19, s47 // (op 3764) 



v_lshl_add_u32 v23, s1, 4, v22 // (op 3764) 


v_mul_lo_u32 v22, v23, s38 // (op 3764) 

v_add_u32 v23, v24, v22 // (op 3764) 


s_mul_i32 s1, s38, s5 // (op 3766) most significant: low of low * high
s_mul_i32 s47, s39, s4 // (op 3766) most significant: low of high * low
s_mul_i32 s0, s38, s4 // (op 3766) least significant
s_mul_hi_u32 s18, s38, s4 // (op 3766) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 3766) most significant: sum
s_add_u32 s1, s1, s18 // (op 3766) most significant: sum


s_mov_b32 s68, s40 // (op 3766) reg expression
s_mov_b32 s69, s41 // (op 3766) reg expression
s_mov_b32 s70, s0 // (op 3766) reg expression
s_mov_b32 s71, 131072 // (op 3766) call()

s_mul_hi_i32 s47, s16, s10 // (op 3783) 
v_and_b32 v22, 15, v1 // (op 3783) 
s_add_i32 s0, s47, s16 // (op 3783) 

v_and_b32 v24, 63, v1 // (op 3783) 
s_ashr_i32 s1, s0, 31 // (op 3783) 
v_lshrrev_b32 v25, 4, v24 // (op 3783) 

s_cmp_eq_i32 s10, 0 // (op 3783) 
s_mov_b32 s47, scc // (op 3783) 
s_cmp_eq_i32 s47, 0 // (op 3783) 
s_cselect_b32 s18, 0, -1 // (op 3783) 

s_and_b32 s19, s13, 31 // (op 3783) 
s_lshl_b32 s47, 1, s19 // (op 3783) 
s_add_i32 s22, s47, s18 // (op 3783) 


s_and_b32 s18, s1, s22 // (op 3783) 


s_add_i32 s1, s0, s18 // (op 3783) 


s_ashr_i32 s0, s1, s19 // (op 3783) 


s_lshl_b32 s1, s13, 24 // (op 3783) 
s_ashr_i32 s47, s1, 31 // (op 3783) 

s_xor_b32 s1, s0, s47 // (op 3783) 

s_sub_i32 s0, s1, s47 // (op 3783) 


s_mul_hi_i32 s1, s0, s14 // (op 3783) 
s_add_i32 s47, s1, s0 // (op 3783) 

s_ashr_i32 s1, s47, 31 // (op 3783) 
s_cmp_eq_i32 s14, 0 // (op 3783) 
s_mov_b32 s18, scc // (op 3783) 
s_cmp_eq_i32 s18, 0 // (op 3783) 
s_cselect_b32 s19, 0, -1 // (op 3783) 

s_and_b32 s18, s15, 31 // (op 3783) 
s_lshl_b32 s22, 1, s18 // (op 3783) 
s_add_i32 s23, s22, s19 // (op 3783) 


s_and_b32 s19, s1, s23 // (op 3783) 


s_add_i32 s1, s47, s19 // (op 3783) 


s_ashr_i32 s19, s1, s18 // (op 3783) 


s_lshl_b32 s1, s15, 24 // (op 3783) 
s_ashr_i32 s18, s1, 31 // (op 3783) 

s_xor_b32 s1, s19, s18 // (op 3783) 

s_sub_i32 s19, s1, s18 // (op 3783) 


s_add_u32 s22, s4, 16 // (op 3783) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 3783) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 3783) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 3783) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 3783) 

s_mul_i32 s1, s19, s22 // (op 3783) 

s_sub_i32 s47, s0, s1 // (op 3783) 


s_mul_hi_i32 s0, s16, s9 // (op 3783) 
s_add_i32 s1, s0, s16 // (op 3783) 

s_ashr_i32 s0, s1, 31 // (op 3783) 
s_cmp_eq_i32 s9, 0 // (op 3783) 
s_mov_b32 s18, scc // (op 3783) 
s_cmp_eq_i32 s18, 0 // (op 3783) 
s_cselect_b32 s19, 0, -1 // (op 3783) 

s_and_b32 s18, s45, 31 // (op 3783) 
s_lshl_b32 s24, 1, s18 // (op 3783) 
s_add_i32 s25, s24, s19 // (op 3783) 


s_and_b32 s19, s0, s25 // (op 3783) 


s_add_i32 s0, s1, s19 // (op 3783) 


s_ashr_i32 s1, s0, s18 // (op 3783) 


s_lshl_b32 s0, s45, 24 // (op 3783) 
s_ashr_i32 s18, s0, 31 // (op 3783) 

s_xor_b32 s0, s1, s18 // (op 3783) 

s_sub_i32 s1, s0, s18 // (op 3783) 


s_mul_i32 s0, s11, s22 // (op 3783) 

s_mul_i32 s18, s8, s0 // (op 3783) 

s_mul_i32 s0, s1, s18 // (op 3783) 

s_sub_u32 s18, s16, s0 // (op 3783) 

s_mul_hi_i32 s0, s18, s12 // (op 3783) 
s_add_i32 s19, s0, s18 // (op 3783) 


s_ashr_i32 s0, s19, 31 // (op 3783) 
s_cmp_eq_i32 s12, 0 // (op 3783) 
s_mov_b32 s18, scc // (op 3783) 
s_cmp_eq_i32 s18, 0 // (op 3783) 
s_cselect_b32 s22, 0, -1 // (op 3783) 

s_and_b32 s18, s44, 31 // (op 3783) 
s_lshl_b32 s23, 1, s18 // (op 3783) 
s_add_i32 s24, s23, s22 // (op 3783) 


s_and_b32 s25, s0, s24 // (op 3783) 


s_add_i32 s0, s19, s25 // (op 3783) 


s_ashr_i32 s19, s0, s18 // (op 3783) 


s_lshl_b32 s0, s44, 24 // (op 3783) 
s_ashr_i32 s18, s0, 31 // (op 3783) 

s_xor_b32 s0, s19, s18 // (op 3783) 

s_sub_i32 s19, s0, s18 // (op 3783) 


s_cmp_eq_i32 s1, 0 // (op 3783) 
s_mov_b32 s0, scc // (op 3783) 

s_cmp_eq_i32 s0, 0 // (op 3783) 
s_cselect_b32 s1, s19, s47 // (op 3783) 



v_lshl_add_u32 v24, s1, 4, v22 // (op 3783) 


v_mul_lo_u32 v22, v24, s38 // (op 3783) 

v_add_u32 v24, v25, v22 // (op 3783) 


s_mul_i32 s1, s38, s5 // (op 3785) most significant: low of low * high
s_mul_i32 s47, s39, s4 // (op 3785) most significant: low of high * low
s_mul_i32 s0, s38, s4 // (op 3785) least significant
s_mul_hi_u32 s18, s38, s4 // (op 3785) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 3785) most significant: sum
s_add_u32 s1, s1, s18 // (op 3785) most significant: sum


s_mov_b32 s68, s40 // (op 3785) reg expression
s_mov_b32 s69, s41 // (op 3785) reg expression
s_mov_b32 s70, s0 // (op 3785) reg expression
s_mov_b32 s71, 131072 // (op 3785) call()




buffer_load_ubyte v22, v23, s[68:71], 0 offen offset: 0 // (op 3252) Load value
v_and_b32 v25, -8, v1 // (op 3906) 
v_sub_u32 v26, v1, v25 // (op 3906) 

v_lshlrev_b32 v25, 4, v26 // (op 3906) 

v_lshrrev_b32 v26, 3, v1 // (op 3906) 
v_lshl_add_u32 v27, v26, 7, v25 // (op 3906) 


s_nop 0
v_readfirstlane_b32 s47, v27 // (op 3906) Wait state hazard: VALU Write Readlane Hazard




s_nop 1
s_mov_b32 m0, s47 // (op 3514) reg expression
buffer_load_dwordx4 v12, s[60:63], 0 offen offset: 0 lds // (op 3514) Load value direct to lds

v_and_b32 v25, -8, v1 // (op 3849) 
v_sub_u32 v26, v1, v25 // (op 3849) 

v_lshlrev_b32 v25, 4, v26 // (op 3849) 

v_lshrrev_b32 v26, 3, v1 // (op 3849) 
v_lshl_add_u32 v27, v26, 7, v25 // (op 3849) 


s_nop 0
v_readfirstlane_b32 s47, v27 // (op 3849) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 4 // (op 3519) 
v_mov_b64 v[26:27], s[0:1] // (op 3519) convert


v_mov_b64 v[28:29], v[16:17] // (op 3519) 
s_add_u32 s0, 2048, s47 // (op 3519) 
s_mov_b32 m0, s0 // (op 3519) reg expression
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3519) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 3519) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 3519) most significant half

s_add_u32 m0, m0, 2048 // (op 3519) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3519) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 3519) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 3519) most significant half

s_add_u32 m0, m0, 2048 // (op 3519) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3519) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 3519) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 3519) most significant half

s_add_u32 m0, m0, 2048 // (op 3519) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3519) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 3519) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 3519) most significant half

s_add_u32 m0, m0, 2048 // (op 3519) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3519) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 3519) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 3519) most significant half

s_add_u32 m0, m0, 2048 // (op 3519) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3519) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 3519) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 3519) most significant half

s_add_u32 m0, m0, 2048 // (op 3519) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3519) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 3519) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 3519) most significant half

s_add_u32 m0, m0, 2048 // (op 3519) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3519) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 3519) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 3519) most significant half

s_add_u32 m0, m0, 2048 // (op 3519) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3519) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 3519) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 3519) most significant half

s_add_u32 m0, m0, 2048 // (op 3519) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3519) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 3519) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 3519) most significant half

s_add_u32 m0, m0, 2048 // (op 3519) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3519) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 3519) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 3519) most significant half

s_add_u32 m0, m0, 2048 // (op 3519) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3519) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 3519) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 3519) most significant half

s_add_u32 m0, m0, 2048 // (op 3519) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3519) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 3519) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 3519) most significant half

s_add_u32 m0, m0, 2048 // (op 3519) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3519) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 3519) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 3519) most significant half

s_add_u32 m0, m0, 2048 // (op 3519) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3519) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 3519) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 3519) most significant half

s_add_u32 m0, m0, 2048 // (op 3519) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 3519) Load value direct to lds





v_mov_b64 v[26:27], s[34:35] // (op 3262) convert

v_mov_b32 v25, v20 // (op 3262) 
buffer_load_dword v28, v25, s[64:67], 0 offen // (op 3262) Load value
v_add_u32 v25, v25, v26 // (op 3262) 
buffer_load_dword v29, v25, s[64:67], 0 offen // (op 3262) Load value





buffer_load_ubyte v25, v23, s[68:71], 0 offen offset:4 // (op 3266) Load value

v_and_b32 v23, -8, v1 // (op 4065) 
v_sub_u32 v26, v1, v23 // (op 4065) 

v_lshlrev_b32 v23, 4, v26 // (op 4065) 

v_lshrrev_b32 v26, 3, v1 // (op 4065) 
v_lshl_add_u32 v27, v26, 7, v23 // (op 4065) 


s_nop 0
v_readfirstlane_b32 s47, v27 // (op 4065) Wait state hazard: VALU Write Readlane Hazard



s_nop 1
s_add_u32 s0, 34816, s47 // (op 3524) Wait state hazard: VALU Write SGPR/VCC Hazard

s_mov_b32 m0, s0 // (op 3524) reg expression
s_mov_b32 s1, 128 // (op 3524) call()
buffer_load_dwordx4 v12, s[60:63], s1 offen offset: 0 lds // (op 3524) Load value direct to lds




v_and_b32 v23, -8, v1 // (op 4021) 
v_sub_u32 v12, v1, v23 // (op 4021) 

v_lshlrev_b32 v13, 4, v12 // (op 4021) 

v_lshrrev_b32 v12, 3, v1 // (op 4021) 
v_lshl_add_u32 v23, v12, 7, v13 // (op 4021) 


s_nop 0
v_readfirstlane_b32 s47, v23 // (op 4021) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 4 // (op 3526) 
v_mov_b64 v[12:13], s[0:1] // (op 3526) convert


v_mov_b64 v[26:27], v[16:17] // (op 3526) 
s_add_u32 s0, 36864, s47 // (op 3526) 
s_mov_b32 m0, s0 // (op 3526) reg expression
s_mov_b32 s1, 128 // (op 3526) call()
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 3526) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 3526) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 3526) most significant half

s_add_u32 m0, m0, 2048 // (op 3526) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 3526) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 3526) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 3526) most significant half

s_add_u32 m0, m0, 2048 // (op 3526) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 3526) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 3526) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 3526) most significant half

s_add_u32 m0, m0, 2048 // (op 3526) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 3526) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 3526) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 3526) most significant half

s_add_u32 m0, m0, 2048 // (op 3526) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 3526) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 3526) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 3526) most significant half

s_add_u32 m0, m0, 2048 // (op 3526) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 3526) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 3526) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 3526) most significant half

s_add_u32 m0, m0, 2048 // (op 3526) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 3526) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 3526) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 3526) most significant half

s_add_u32 m0, m0, 2048 // (op 3526) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 3526) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 3526) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 3526) most significant half

s_add_u32 m0, m0, 2048 // (op 3526) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 3526) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 3526) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 3526) most significant half

s_add_u32 m0, m0, 2048 // (op 3526) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 3526) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 3526) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 3526) most significant half

s_add_u32 m0, m0, 2048 // (op 3526) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 3526) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 3526) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 3526) most significant half

s_add_u32 m0, m0, 2048 // (op 3526) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 3526) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 3526) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 3526) most significant half

s_add_u32 m0, m0, 2048 // (op 3526) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 3526) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 3526) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 3526) most significant half

s_add_u32 m0, m0, 2048 // (op 3526) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 3526) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 3526) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 3526) most significant half

s_add_u32 m0, m0, 2048 // (op 3526) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 3526) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v12 // (op 3526) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 3526) most significant half

s_add_u32 m0, m0, 2048 // (op 3526) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 3526) Load value direct to lds







v_mov_b64 v[12:13], s[34:35] // (op 3276) convert

v_mov_b32 v23, v20 // (op 3276) 
buffer_load_dword v16, v23, s[64:67], 0 offen offset:4 // (op 3276) Load value
v_add_u32 v23, v23, v12 // (op 3276) 
buffer_load_dword v17, v23, s[64:67], 0 offen offset:4 // (op 3276) Load value



v_lshlrev_b32 v20, 3, v1 // (op 3959) 



v_add_u32 v23, 69632, v20 // (op 3280) 
s_waitcnt vmcnt(21)
ds_write_b32 v23, v28 // (op 3280) Store local data 

v_add_u32 v23, 69636, v20 // (op 3280) 
s_waitcnt vmcnt(20)
ds_write_b32 v23, v29 // (op 3280) Store local data 


s_barrier  // (op 4181) 



ds_read_b128 v[30:33], v4 // (op 3333) Load local data 
ds_read_b128 v[34:37], v4 offset:64 // (op 3333) Load local data 
s_barrier  // (op 4165) 



ds_read_b128 v[38:41], v7 offset:2048 // (op 3294) Load local data 
ds_read_b128 v[42:45], v7 offset:2112 // (op 3294) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 4197) 



v_add_u32 v20, 69632, v0 // (op 3337) 
ds_read_u8 v23, v20 // (op 3337) Load local data 




ds_read_b128 v[46:49], v7 offset:6144 // (op 3299) Load local data 
ds_read_b128 v[50:53], v7 offset:6208 // (op 3299) Load local data 



v_add_u32 v20, 69760, v0 // (op 3342) 
ds_read_u8 v12, v20 // (op 3342) Load local data 




ds_read_b128 v[54:57], v7 offset:10240 // (op 3304) Load local data 
ds_read_b128 v[58:61], v7 offset:10304 // (op 3304) Load local data 



v_add_u32 v13, 69888, v0 // (op 3347) 
ds_read_u8 v20, v13 // (op 3347) Load local data 




ds_read_b128 v[62:65], v7 offset:14336 // (op 3309) Load local data 
ds_read_b128 v[66:69], v7 offset:14400 // (op 3309) Load local data 



v_add_u32 v13, 70016, v0 // (op 3352) 
ds_read_u8 v26, v13 // (op 3352) Load local data 




ds_read_b128 v[70:73], v7 offset:18432 // (op 3314) Load local data 
ds_read_b128 v[74:77], v7 offset:18496 // (op 3314) Load local data 



v_add_u32 v13, 70144, v0 // (op 3357) 
ds_read_u8 v27, v13 // (op 3357) Load local data 




ds_read_b128 v[78:81], v7 offset:22528 // (op 3319) Load local data 
ds_read_b128 v[82:85], v7 offset:22592 // (op 3319) Load local data 



v_add_u32 v13, 70272, v0 // (op 3362) 
ds_read_u8 v86, v13 // (op 3362) Load local data 




ds_read_b128 v[88:91], v7 offset:26624 // (op 3324) Load local data 
ds_read_b128 v[92:95], v7 offset:26688 // (op 3324) Load local data 



v_add_u32 v13, 70400, v0 // (op 3367) 
ds_read_u8 v87, v13 // (op 3367) Load local data 




ds_read_b128 v[96:99], v7 offset:30720 // (op 3329) Load local data 
ds_read_b128 v[100:103], v7 offset:30784 // (op 3329) Load local data 



v_add_u32 v13, 70528, v0 // (op 3372) 
ds_read_u8 v104, v13 // (op 3372) Load local data 

v_and_b32 v13, -8, v1 // (op 3821) 
v_sub_u32 v105, v1, v13 // (op 3821) 

v_lshlrev_b32 v13, 4, v105 // (op 3821) 

v_lshrrev_b32 v105, 3, v1 // (op 3821) 
v_lshl_add_u32 v106, v105, 7, v13 // (op 3821) 


s_nop 0
v_readfirstlane_b32 s47, v106 // (op 3821) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v13, -8, v1 // (op 3878) 
v_sub_u32 v105, v1, v13 // (op 3878) 

v_lshlrev_b32 v13, 4, v105 // (op 3878) 

v_lshrrev_b32 v105, 3, v1 // (op 3878) 
v_lshl_add_u32 v106, v105, 7, v13 // (op 3878) 


s_nop 0
v_readfirstlane_b32 s0, v106 // (op 3878) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v13, 3, v1 // (op 3930) 
v_and_b32 v105, -8, v1 // (op 4006) 
v_sub_u32 v106, v1, v105 // (op 4006) 

v_lshlrev_b32 v105, 4, v106 // (op 4006) 

v_lshrrev_b32 v106, 3, v1 // (op 4006) 
v_lshl_add_u32 v107, v106, 7, v105 // (op 4006) 


s_nop 0
v_readfirstlane_b32 s1, v107 // (op 4006) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v105, -8, v1 // (op 4050) 
v_sub_u32 v106, v1, v105 // (op 4050) 

v_lshlrev_b32 v105, 4, v106 // (op 4050) 

v_lshrrev_b32 v106, 3, v1 // (op 4050) 
v_lshl_add_u32 v107, v106, 7, v105 // (op 4050) 


s_nop 0
v_readfirstlane_b32 s18, v107 // (op 4050) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v105, 3, v1 // (op 4089) 
s_mov_b32 s19, 0 // (op 1283) call()
s_ashr_i64 s[22:23], s[42:43], 7 // (op 1282) 
s_mov_b32 s24, s22 // (op 1282) convert

s_and_b32 s25, s24, 4294967294 // (op 1282) 

s_cmp_lt_u32 s19, s25 // (op 1282) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ForLoopBottom_KLoop_1282 // (op 1282) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ForLoopBottom_KLoop_1282 if false)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ForLoopTop_KLoop_1282:
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:45], v[30:37], a[0:3], v23, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 82) 
s_add_u32 s19, s19, 2 // (op 1284) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[46:53], v[30:37], a[4:7], v12, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 346) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[54:61], v[30:37], a[8:11], v20, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 508) 
s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[62:69], v[30:37], a[12:15], v26, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 670) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[70:77], v[30:37], a[16:19], v27, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 832) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:85], v[30:37], a[20:23], v86, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 994) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[88:95], v[30:37], a[24:27], v87, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1156) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[96:103], v[30:37], a[28:31], v104, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1318) 


v_mov_b32 v106, v24 // (op 80) 

buffer_load_ubyte v22, v106, s[68:71], 0 offen offset:8 // (op 80) Load value
s_barrier  // (op 4184) 


v_mov_b64 v[108:109], v[14:15] // (op 3486) 

s_mov_b32 m0, s0 // (op 3486) reg expression
s_mov_b32 s22, 256 // (op 3486) call()
buffer_load_dwordx4 v108, s[60:63], s22 offen offset: 0 lds // (op 3486) Load value direct to lds

s_barrier  // (op 4168) 

s_lshl_b64 s[22:23], s[26:27], 4 // (op 3481) 
v_mov_b64 v[110:111], s[22:23] // (op 3481) convert


v_mov_b64 v[112:113], v[18:19] // (op 3481) 
v_mov_b64 v[114:115], v[112:113] // (op 3481) 
s_add_u32 s22, 2048, s47 // (op 3481) 
s_mov_b32 m0, s22 // (op 3481) reg expression
s_mov_b32 s23, 256 // (op 3481) call()
buffer_load_dwordx4 v114, s[48:51], s23 offen offset: 0 lds // (op 3481) Load value direct to lds
v_add_co_u32 v114, s[24:25], v114, v110 // (op 3481) least significant half
v_addc_co_u32 v115, s[24:25], 0, v115, s[24:25] // (op 3481) most significant half

s_add_u32 m0, m0, 2048 // (op 3481) 
buffer_load_dwordx4 v114, s[48:51], s23 offen offset: 0 lds // (op 3481) Load value direct to lds
v_add_co_u32 v114, s[24:25], v114, v110 // (op 3481) least significant half
v_addc_co_u32 v115, s[24:25], 0, v115, s[24:25] // (op 3481) most significant half

s_add_u32 m0, m0, 2048 // (op 3481) 
buffer_load_dwordx4 v114, s[48:51], s23 offen offset: 0 lds // (op 3481) Load value direct to lds
v_add_co_u32 v114, s[24:25], v114, v110 // (op 3481) least significant half
v_addc_co_u32 v115, s[24:25], 0, v115, s[24:25] // (op 3481) most significant half

s_add_u32 m0, m0, 2048 // (op 3481) 
buffer_load_dwordx4 v114, s[48:51], s23 offen offset: 0 lds // (op 3481) Load value direct to lds
v_add_co_u32 v114, s[24:25], v114, v110 // (op 3481) least significant half
v_addc_co_u32 v115, s[24:25], 0, v115, s[24:25] // (op 3481) most significant half

s_add_u32 m0, m0, 2048 // (op 3481) 
buffer_load_dwordx4 v114, s[48:51], s23 offen offset: 0 lds // (op 3481) Load value direct to lds
v_add_co_u32 v114, s[24:25], v114, v110 // (op 3481) least significant half
v_addc_co_u32 v115, s[24:25], 0, v115, s[24:25] // (op 3481) most significant half

s_add_u32 m0, m0, 2048 // (op 3481) 
buffer_load_dwordx4 v114, s[48:51], s23 offen offset: 0 lds // (op 3481) Load value direct to lds
v_add_co_u32 v114, s[24:25], v114, v110 // (op 3481) least significant half
v_addc_co_u32 v115, s[24:25], 0, v115, s[24:25] // (op 3481) most significant half

s_add_u32 m0, m0, 2048 // (op 3481) 
buffer_load_dwordx4 v114, s[48:51], s23 offen offset: 0 lds // (op 3481) Load value direct to lds
v_add_co_u32 v114, s[24:25], v114, v110 // (op 3481) least significant half
v_addc_co_u32 v115, s[24:25], 0, v115, s[24:25] // (op 3481) most significant half

s_add_u32 m0, m0, 2048 // (op 3481) 
buffer_load_dwordx4 v114, s[48:51], s23 offen offset: 0 lds // (op 3481) Load value direct to lds
v_add_co_u32 v114, s[24:25], v114, v110 // (op 3481) least significant half
v_addc_co_u32 v115, s[24:25], 0, v115, s[24:25] // (op 3481) most significant half

s_add_u32 m0, m0, 2048 // (op 3481) 
buffer_load_dwordx4 v114, s[48:51], s23 offen offset: 0 lds // (op 3481) Load value direct to lds
v_add_co_u32 v114, s[24:25], v114, v110 // (op 3481) least significant half
v_addc_co_u32 v115, s[24:25], 0, v115, s[24:25] // (op 3481) most significant half

s_add_u32 m0, m0, 2048 // (op 3481) 
buffer_load_dwordx4 v114, s[48:51], s23 offen offset: 0 lds // (op 3481) Load value direct to lds
v_add_co_u32 v114, s[24:25], v114, v110 // (op 3481) least significant half
v_addc_co_u32 v115, s[24:25], 0, v115, s[24:25] // (op 3481) most significant half

s_add_u32 m0, m0, 2048 // (op 3481) 
buffer_load_dwordx4 v114, s[48:51], s23 offen offset: 0 lds // (op 3481) Load value direct to lds
v_add_co_u32 v114, s[24:25], v114, v110 // (op 3481) least significant half
v_addc_co_u32 v115, s[24:25], 0, v115, s[24:25] // (op 3481) most significant half

s_add_u32 m0, m0, 2048 // (op 3481) 
buffer_load_dwordx4 v114, s[48:51], s23 offen offset: 0 lds // (op 3481) Load value direct to lds
v_add_co_u32 v114, s[24:25], v114, v110 // (op 3481) least significant half
v_addc_co_u32 v115, s[24:25], 0, v115, s[24:25] // (op 3481) most significant half

s_add_u32 m0, m0, 2048 // (op 3481) 
buffer_load_dwordx4 v114, s[48:51], s23 offen offset: 0 lds // (op 3481) Load value direct to lds
v_add_co_u32 v114, s[24:25], v114, v110 // (op 3481) least significant half
v_addc_co_u32 v115, s[24:25], 0, v115, s[24:25] // (op 3481) most significant half

s_add_u32 m0, m0, 2048 // (op 3481) 
buffer_load_dwordx4 v114, s[48:51], s23 offen offset: 0 lds // (op 3481) Load value direct to lds
v_add_co_u32 v114, s[24:25], v114, v110 // (op 3481) least significant half
v_addc_co_u32 v115, s[24:25], 0, v115, s[24:25] // (op 3481) most significant half

s_add_u32 m0, m0, 2048 // (op 3481) 
buffer_load_dwordx4 v114, s[48:51], s23 offen offset: 0 lds // (op 3481) Load value direct to lds
v_add_co_u32 v114, s[24:25], v114, v110 // (op 3481) least significant half
v_addc_co_u32 v115, s[24:25], 0, v115, s[24:25] // (op 3481) most significant half

s_add_u32 m0, m0, 2048 // (op 3481) 
buffer_load_dwordx4 v114, s[48:51], s23 offen offset: 0 lds // (op 3481) Load value direct to lds





v_mov_b64 v[110:111], s[34:35] // (op 17) convert

v_mov_b32 v107, v21 // (op 17) 
v_mov_b32 v114, v107 // (op 17) 
buffer_load_dword v28, v114, s[64:67], 0 offen offset:8 // (op 17) Load value
v_add_u32 v114, v114, v110 // (op 17) 
buffer_load_dword v29, v114, s[64:67], 0 offen offset:8 // (op 17) Load value


s_barrier  // (op 4229) 



v_add_u32 v110, 70656, v105 // (op 236) 
s_waitcnt vmcnt(21)
ds_write_b32 v110, v16 // (op 236) Store local data 

v_add_u32 v110, 70660, v105 // (op 236) 
s_waitcnt vmcnt(20)
ds_write_b32 v110, v17 // (op 236) Store local data 

s_barrier  // (op 4220) 



ds_read_b128 v[30:33], v5 offset:34816 // (op 229) Load local data 
ds_read_b128 v[34:37], v5 offset:34880 // (op 229) Load local data 
s_barrier  // (op 4214) 



ds_read_b128 v[38:41], v11 offset:36864 // (op 227) Load local data 
ds_read_b128 v[42:45], v11 offset:36928 // (op 227) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 4226) 



v_add_u32 v110, 70656, v10 // (op 231) 
ds_read_u8 v23, v110 // (op 231) Load local data 




ds_read_b128 v[46:49], v11 offset:40960 // (op 353) Load local data 
ds_read_b128 v[50:53], v11 offset:41024 // (op 353) Load local data 



v_add_u32 v110, 70784, v10 // (op 357) 
ds_read_u8 v12, v110 // (op 357) Load local data 




ds_read_b128 v[54:57], v11 offset:45056 // (op 515) Load local data 
ds_read_b128 v[58:61], v11 offset:45120 // (op 515) Load local data 



v_add_u32 v110, 70912, v10 // (op 519) 
ds_read_u8 v20, v110 // (op 519) Load local data 




ds_read_b128 v[62:65], v11 offset:49152 // (op 677) Load local data 
ds_read_b128 v[66:69], v11 offset:49216 // (op 677) Load local data 



v_add_u32 v110, 71040, v10 // (op 681) 
ds_read_u8 v26, v110 // (op 681) Load local data 




ds_read_b128 v[70:73], v11 offset:53248 // (op 839) Load local data 
ds_read_b128 v[74:77], v11 offset:53312 // (op 839) Load local data 



v_add_u32 v110, 71168, v10 // (op 843) 
ds_read_u8 v27, v110 // (op 843) Load local data 




ds_read_b128 v[78:81], v11 offset:57344 // (op 1001) Load local data 
ds_read_b128 v[82:85], v11 offset:57408 // (op 1001) Load local data 



v_add_u32 v110, 71296, v10 // (op 1005) 
ds_read_u8 v86, v110 // (op 1005) Load local data 




ds_read_b128 v[88:91], v11 offset:61440 // (op 1163) Load local data 
ds_read_b128 v[92:95], v11 offset:61504 // (op 1163) Load local data 



v_add_u32 v110, 71424, v10 // (op 1167) 
ds_read_u8 v87, v110 // (op 1167) Load local data 




v_add_u32 v110, 65536, v11 // (op 1325) 
ds_read_b128 v[96:99], v110 // (op 1325) Load local data 

v_add_u32 v110, 65600, v11 // (op 1325) 
ds_read_b128 v[100:103], v110 // (op 1325) Load local data 




v_add_u32 v110, 71552, v10 // (op 1329) 
ds_read_u8 v104, v110 // (op 1329) Load local data 

s_barrier  // (op 4304) 
s_waitcnt lgkmcnt(15)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[38:45], v[30:37], a[0:3], v23, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 234) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[46:53], v[30:37], a[4:7], v12, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 347) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[54:61], v[30:37], a[8:11], v20, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 509) 
s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[62:69], v[30:37], a[12:15], v26, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 671) 
s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[70:77], v[30:37], a[16:19], v27, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 833) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:85], v[30:37], a[20:23], v86, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 995) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[88:95], v[30:37], a[24:27], v87, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1157) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[96:103], v[30:37], a[28:31], v104, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1319) 



buffer_load_ubyte v25, v106, s[68:71], 0 offen offset:12 // (op 233) Load value

s_barrier  // (op 4223) 
v_add_u32 v24, 8, v24 // (op 3775) 


s_add_u32 s22, 34816, s18 // (op 3509) 

s_mov_b32 m0, s22 // (op 3509) reg expression
s_mov_b32 s23, 384 // (op 3509) call()
buffer_load_dwordx4 v108, s[60:63], s23 offen offset: 0 lds // (op 3509) Load value direct to lds



s_barrier  // (op 4217) 
v_add_co_u32 v14, vcc, 256, v14 // (op 3647) least significant half

v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 3647) most significant half
s_lshl_b64 s[22:23], s[26:27], 4 // (op 3504) 
v_mov_b64 v[108:109], s[22:23] // (op 3504) convert


v_mov_b64 v[110:111], v[112:113] // (op 3504) 
s_add_u32 s22, 36864, s1 // (op 3504) 
s_mov_b32 m0, s22 // (op 3504) reg expression
s_mov_b32 s23, 384 // (op 3504) call()
buffer_load_dwordx4 v110, s[48:51], s23 offen offset: 0 lds // (op 3504) Load value direct to lds
v_add_co_u32 v110, s[24:25], v110, v108 // (op 3504) least significant half
v_addc_co_u32 v111, s[24:25], 0, v111, s[24:25] // (op 3504) most significant half

s_add_u32 m0, m0, 2048 // (op 3504) 
buffer_load_dwordx4 v110, s[48:51], s23 offen offset: 0 lds // (op 3504) Load value direct to lds
v_add_co_u32 v110, s[24:25], v110, v108 // (op 3504) least significant half
v_addc_co_u32 v111, s[24:25], 0, v111, s[24:25] // (op 3504) most significant half

s_add_u32 m0, m0, 2048 // (op 3504) 
buffer_load_dwordx4 v110, s[48:51], s23 offen offset: 0 lds // (op 3504) Load value direct to lds
v_add_co_u32 v110, s[24:25], v110, v108 // (op 3504) least significant half
v_addc_co_u32 v111, s[24:25], 0, v111, s[24:25] // (op 3504) most significant half

s_add_u32 m0, m0, 2048 // (op 3504) 
buffer_load_dwordx4 v110, s[48:51], s23 offen offset: 0 lds // (op 3504) Load value direct to lds
v_add_co_u32 v110, s[24:25], v110, v108 // (op 3504) least significant half
v_addc_co_u32 v111, s[24:25], 0, v111, s[24:25] // (op 3504) most significant half

s_add_u32 m0, m0, 2048 // (op 3504) 
buffer_load_dwordx4 v110, s[48:51], s23 offen offset: 0 lds // (op 3504) Load value direct to lds
v_add_co_u32 v110, s[24:25], v110, v108 // (op 3504) least significant half
v_addc_co_u32 v111, s[24:25], 0, v111, s[24:25] // (op 3504) most significant half

s_add_u32 m0, m0, 2048 // (op 3504) 
buffer_load_dwordx4 v110, s[48:51], s23 offen offset: 0 lds // (op 3504) Load value direct to lds
v_add_co_u32 v110, s[24:25], v110, v108 // (op 3504) least significant half
v_addc_co_u32 v111, s[24:25], 0, v111, s[24:25] // (op 3504) most significant half

s_add_u32 m0, m0, 2048 // (op 3504) 
buffer_load_dwordx4 v110, s[48:51], s23 offen offset: 0 lds // (op 3504) Load value direct to lds
v_add_co_u32 v110, s[24:25], v110, v108 // (op 3504) least significant half
v_addc_co_u32 v111, s[24:25], 0, v111, s[24:25] // (op 3504) most significant half

s_add_u32 m0, m0, 2048 // (op 3504) 
buffer_load_dwordx4 v110, s[48:51], s23 offen offset: 0 lds // (op 3504) Load value direct to lds
v_add_co_u32 v110, s[24:25], v110, v108 // (op 3504) least significant half
v_addc_co_u32 v111, s[24:25], 0, v111, s[24:25] // (op 3504) most significant half

s_add_u32 m0, m0, 2048 // (op 3504) 
buffer_load_dwordx4 v110, s[48:51], s23 offen offset: 0 lds // (op 3504) Load value direct to lds
v_add_co_u32 v110, s[24:25], v110, v108 // (op 3504) least significant half
v_addc_co_u32 v111, s[24:25], 0, v111, s[24:25] // (op 3504) most significant half

s_add_u32 m0, m0, 2048 // (op 3504) 
buffer_load_dwordx4 v110, s[48:51], s23 offen offset: 0 lds // (op 3504) Load value direct to lds
v_add_co_u32 v110, s[24:25], v110, v108 // (op 3504) least significant half
v_addc_co_u32 v111, s[24:25], 0, v111, s[24:25] // (op 3504) most significant half

s_add_u32 m0, m0, 2048 // (op 3504) 
buffer_load_dwordx4 v110, s[48:51], s23 offen offset: 0 lds // (op 3504) Load value direct to lds
v_add_co_u32 v110, s[24:25], v110, v108 // (op 3504) least significant half
v_addc_co_u32 v111, s[24:25], 0, v111, s[24:25] // (op 3504) most significant half

s_add_u32 m0, m0, 2048 // (op 3504) 
buffer_load_dwordx4 v110, s[48:51], s23 offen offset: 0 lds // (op 3504) Load value direct to lds
v_add_co_u32 v110, s[24:25], v110, v108 // (op 3504) least significant half
v_addc_co_u32 v111, s[24:25], 0, v111, s[24:25] // (op 3504) most significant half

s_add_u32 m0, m0, 2048 // (op 3504) 
buffer_load_dwordx4 v110, s[48:51], s23 offen offset: 0 lds // (op 3504) Load value direct to lds
v_add_co_u32 v110, s[24:25], v110, v108 // (op 3504) least significant half
v_addc_co_u32 v111, s[24:25], 0, v111, s[24:25] // (op 3504) most significant half

s_add_u32 m0, m0, 2048 // (op 3504) 
buffer_load_dwordx4 v110, s[48:51], s23 offen offset: 0 lds // (op 3504) Load value direct to lds
v_add_co_u32 v110, s[24:25], v110, v108 // (op 3504) least significant half
v_addc_co_u32 v111, s[24:25], 0, v111, s[24:25] // (op 3504) most significant half

s_add_u32 m0, m0, 2048 // (op 3504) 
buffer_load_dwordx4 v110, s[48:51], s23 offen offset: 0 lds // (op 3504) Load value direct to lds
v_add_co_u32 v110, s[24:25], v110, v108 // (op 3504) least significant half
v_addc_co_u32 v111, s[24:25], 0, v111, s[24:25] // (op 3504) most significant half

s_add_u32 m0, m0, 2048 // (op 3504) 
buffer_load_dwordx4 v110, s[48:51], s23 offen offset: 0 lds // (op 3504) Load value direct to lds






v_add_co_u32 v18, vcc, 256, v18 // (op 3583) least significant half
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 3583) most significant half
v_mov_b64 v[108:109], s[34:35] // (op 238) convert

v_mov_b32 v106, v107 // (op 238) 
buffer_load_dword v16, v106, s[64:67], 0 offen offset:12 // (op 238) Load value
v_add_u32 v106, v106, v108 // (op 238) 
buffer_load_dword v17, v106, s[64:67], 0 offen offset:12 // (op 238) Load value



v_add_u32 v21, 8, v21 // (op 3711) 
s_barrier  // (op 4200) 



v_add_u32 v106, 69632, v13 // (op 21) 
s_waitcnt vmcnt(21)
ds_write_b32 v106, v28 // (op 21) Store local data 

v_add_u32 v106, 69636, v13 // (op 21) 
s_waitcnt vmcnt(20)
ds_write_b32 v106, v29 // (op 21) Store local data 

s_barrier  // (op 4190) 



ds_read_b128 v[30:33], v4 // (op 70) Load local data 
ds_read_b128 v[34:37], v4 offset:64 // (op 70) Load local data 
s_barrier  // (op 4174) 



ds_read_b128 v[38:41], v7 offset:2048 // (op 65) Load local data 
ds_read_b128 v[42:45], v7 offset:2112 // (op 65) Load local data 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 4207) 



v_add_u32 v106, 69632, v0 // (op 75) 
ds_read_u8 v23, v106 // (op 75) Load local data 




ds_read_b128 v[46:49], v7 offset:6144 // (op 340) Load local data 
ds_read_b128 v[50:53], v7 offset:6208 // (op 340) Load local data 



v_add_u32 v106, 69760, v0 // (op 344) 
ds_read_u8 v12, v106 // (op 344) Load local data 




ds_read_b128 v[54:57], v7 offset:10240 // (op 502) Load local data 
ds_read_b128 v[58:61], v7 offset:10304 // (op 502) Load local data 



v_add_u32 v106, 69888, v0 // (op 506) 
ds_read_u8 v20, v106 // (op 506) Load local data 




ds_read_b128 v[62:65], v7 offset:14336 // (op 664) Load local data 
ds_read_b128 v[66:69], v7 offset:14400 // (op 664) Load local data 



v_add_u32 v106, 70016, v0 // (op 668) 
ds_read_u8 v26, v106 // (op 668) Load local data 




ds_read_b128 v[70:73], v7 offset:18432 // (op 826) Load local data 
ds_read_b128 v[74:77], v7 offset:18496 // (op 826) Load local data 



v_add_u32 v106, 70144, v0 // (op 830) 
ds_read_u8 v27, v106 // (op 830) Load local data 




ds_read_b128 v[78:81], v7 offset:22528 // (op 988) Load local data 
ds_read_b128 v[82:85], v7 offset:22592 // (op 988) Load local data 



v_add_u32 v106, 70272, v0 // (op 992) 
ds_read_u8 v86, v106 // (op 992) Load local data 




ds_read_b128 v[88:91], v7 offset:26624 // (op 1150) Load local data 
ds_read_b128 v[92:95], v7 offset:26688 // (op 1150) Load local data 



v_add_u32 v106, 70400, v0 // (op 1154) 
ds_read_u8 v87, v106 // (op 1154) Load local data 




ds_read_b128 v[96:99], v7 offset:30720 // (op 1312) Load local data 
ds_read_b128 v[100:103], v7 offset:30784 // (op 1312) Load local data 



v_add_u32 v106, 70528, v0 // (op 1316) 
ds_read_u8 v104, v106 // (op 1316) Load local data 

s_ashr_i64 s[22:23], s[42:43], 7 // (op 1282) 
s_mov_b32 s24, s22 // (op 1282) convert

s_and_b32 s25, s24, 4294967294 // (op 1282) 

s_cmp_lt_u32 s19, s25 // (op 1282) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ForLoopTop_KLoop_1282 // (op 1282) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ForLoopTop_KLoop_1282 if true)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ForLoopBottom_KLoop_1282:




















s_barrier  // (op 4288) 
s_mul_hi_i32 s47, s16, s10 // (op 3539) 
v_lshrrev_b32 v12, 3, v1 // (op 3539) 
s_add_i32 s0, s47, s16 // (op 3539) 

v_and_b32 v13, -8, v1 // (op 3539) 
s_ashr_i32 s1, s0, 31 // (op 3539) 
v_sub_u32 v14, v1, v13 // (op 3539) 

s_cmp_eq_i32 s10, 0 // (op 3539) 
s_mov_b32 s47, scc // (op 3539) 
v_lshlrev_b32 v13, 4, v14 // (op 3539) 

s_cmp_eq_i32 s47, 0 // (op 3539) 
s_cselect_b32 s18, 0, -1 // (op 3539) 

s_and_b32 s19, s13, 31 // (op 3539) 
s_lshl_b32 s47, 1, s19 // (op 3539) 
s_add_i32 s22, s47, s18 // (op 3539) 


s_and_b32 s18, s1, s22 // (op 3539) 


s_add_i32 s1, s0, s18 // (op 3539) 


s_ashr_i32 s0, s1, s19 // (op 3539) 


s_lshl_b32 s1, s13, 24 // (op 3539) 
s_ashr_i32 s47, s1, 31 // (op 3539) 

s_xor_b32 s1, s0, s47 // (op 3539) 

s_sub_i32 s0, s1, s47 // (op 3539) 


s_mul_hi_i32 s1, s0, s14 // (op 3539) 
s_add_i32 s47, s1, s0 // (op 3539) 

s_ashr_i32 s1, s47, 31 // (op 3539) 
s_cmp_eq_i32 s14, 0 // (op 3539) 
s_mov_b32 s18, scc // (op 3539) 
s_cmp_eq_i32 s18, 0 // (op 3539) 
s_cselect_b32 s19, 0, -1 // (op 3539) 

s_and_b32 s18, s15, 31 // (op 3539) 
s_lshl_b32 s22, 1, s18 // (op 3539) 
s_add_i32 s23, s22, s19 // (op 3539) 


s_and_b32 s19, s1, s23 // (op 3539) 


s_add_i32 s1, s47, s19 // (op 3539) 


s_ashr_i32 s19, s1, s18 // (op 3539) 


s_lshl_b32 s1, s15, 24 // (op 3539) 
s_ashr_i32 s18, s1, 31 // (op 3539) 

s_xor_b32 s1, s19, s18 // (op 3539) 

s_sub_i32 s19, s1, s18 // (op 3539) 


s_mul_i32 s1, s0, s11 // (op 3539) 

s_sub_u32 s0, s16, s1 // (op 3539) 

s_mul_i32 s1, s19, s11 // (op 3539) 
s_add_u32 s1, s1, s0 // (op 3539) 


s_mul_hi_i32 s0, s16, s9 // (op 3539) 
s_add_i32 s47, s0, s16 // (op 3539) 

s_ashr_i32 s0, s47, 31 // (op 3539) 
s_cmp_eq_i32 s9, 0 // (op 3539) 
s_mov_b32 s18, scc // (op 3539) 
s_cmp_eq_i32 s18, 0 // (op 3539) 
s_cselect_b32 s19, 0, -1 // (op 3539) 

s_and_b32 s18, s45, 31 // (op 3539) 
s_lshl_b32 s22, 1, s18 // (op 3539) 
s_add_i32 s23, s22, s19 // (op 3539) 


s_and_b32 s19, s0, s23 // (op 3539) 


s_add_i32 s0, s47, s19 // (op 3539) 


s_ashr_i32 s19, s0, s18 // (op 3539) 


s_lshl_b32 s0, s45, 24 // (op 3539) 
s_ashr_i32 s18, s0, 31 // (op 3539) 

s_xor_b32 s0, s19, s18 // (op 3539) 

s_sub_i32 s19, s0, s18 // (op 3539) 


s_add_u32 s22, s4, 16 // (op 3539) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 3539) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 3539) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 3539) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 3539) 

s_mul_i32 s0, s11, s22 // (op 3539) 

s_mul_i32 s18, s8, s0 // (op 3539) 

s_mul_i32 s0, s19, s18 // (op 3539) 

s_sub_u32 s18, s16, s0 // (op 3539) 

s_mov_b32 s0, s18 // (op 3539) 
s_mul_hi_i32 s47, s0, s12 // (op 3539) 
s_add_i32 s22, s47, s0 // (op 3539) 


s_ashr_i32 s0, s22, 31 // (op 3539) 
s_cmp_eq_i32 s12, 0 // (op 3539) 
s_mov_b32 s47, scc // (op 3539) 
s_cmp_eq_i32 s47, 0 // (op 3539) 
s_cselect_b32 s23, 0, -1 // (op 3539) 

s_and_b32 s47, s44, 31 // (op 3539) 
s_lshl_b32 s24, 1, s47 // (op 3539) 
s_add_i32 s25, s24, s23 // (op 3539) 


s_and_b32 s23, s0, s25 // (op 3539) 


s_add_i32 s0, s22, s23 // (op 3539) 


s_ashr_i32 s22, s0, s47 // (op 3539) 


s_lshl_b32 s0, s44, 24 // (op 3539) 
s_ashr_i32 s47, s0, 31 // (op 3539) 

s_xor_b32 s0, s22, s47 // (op 3539) 

s_sub_i32 s22, s0, s47 // (op 3539) 


s_mul_i32 s0, s46, s11 // (op 3539) 
s_add_u32 s24, s6, 256 // (op 3539) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 3539) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 3539) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 3539) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 8 // (op 3539) 

s_sub_i32 s23, s24, s0 // (op 3539) 

s_mul_i32 s47, s22, s23 // (op 3539) 


s_sub_u32 s22, s18, s47 // (op 3539) 


s_add_u32 s18, s22, s0 // (op 3539) 


s_cmp_eq_i32 s19, 0 // (op 3539) 
s_mov_b32 s0, scc // (op 3539) 

s_cmp_eq_i32 s0, 0 // (op 3539) 
s_cselect_b32 s19, s18, s1 // (op 3539) 



v_lshl_add_u32 v14, s19, 8, v12 // (op 3539) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 3539) 
s_lshr_b32 s47, s0, 1 // (op 3539) 

v_lshl_add_u32 v12, s47, 8, v13 // (op 3539) 


v_mov_b32 v19, 0 // (op 3539) convert
v_mov_b32 v18, v12 // (op 3539) convert

v_mul_lo_u32 v13, v14, s27 // (op 3539) most significant: low of low * high
v_mul_hi_u32 v15, v14, s26 // (op 3539) most significant: high of low * low
v_mul_lo_u32 v12, v14, s26 // (op 3539) least significant: low of low * low
v_add_u32 v13, v13, v15 // (op 3539) most significant: sum

v_add_co_u32 v12, s[0:1], v12, v18 // (op 3539) least significant half
v_addc_co_u32 v13, s[0:1], v13, v19, s[0:1] // (op 3539) most significant half



s_mul_i32 s1, s26, s7 // (op 3541) most significant: low of low * high
s_mul_i32 s47, s27, s6 // (op 3541) most significant: low of high * low
s_mul_i32 s0, s26, s6 // (op 3541) least significant
s_mul_hi_u32 s18, s26, s6 // (op 3541) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 3541) most significant: sum
s_add_u32 s1, s1, s18 // (op 3541) most significant: sum


s_mov_b32 s48, s28 // (op 3541) reg expression
s_mov_b32 s49, s29 // (op 3541) reg expression
s_mov_b32 s50, s0 // (op 3541) reg expression
s_mov_b32 s51, 131072 // (op 3541) call()

s_mul_hi_i32 s47, s16, s10 // (op 3613) 
v_lshrrev_b32 v14, 3, v1 // (op 3613) 
s_add_i32 s0, s47, s16 // (op 3613) 

v_and_b32 v15, -8, v1 // (op 3613) 
s_ashr_i32 s1, s0, 31 // (op 3613) 
v_sub_u32 v18, v1, v15 // (op 3613) 

s_cmp_eq_i32 s10, 0 // (op 3613) 
s_mov_b32 s47, scc // (op 3613) 
v_lshlrev_b32 v15, 4, v18 // (op 3613) 

s_cmp_eq_i32 s47, 0 // (op 3613) 
s_cselect_b32 s18, 0, -1 // (op 3613) 

s_and_b32 s19, s13, 31 // (op 3613) 
s_lshl_b32 s47, 1, s19 // (op 3613) 
s_add_i32 s22, s47, s18 // (op 3613) 


s_and_b32 s18, s1, s22 // (op 3613) 


s_add_i32 s1, s0, s18 // (op 3613) 


s_ashr_i32 s0, s1, s19 // (op 3613) 


s_lshl_b32 s1, s13, 24 // (op 3613) 
s_ashr_i32 s47, s1, 31 // (op 3613) 

s_xor_b32 s1, s0, s47 // (op 3613) 

s_sub_i32 s0, s1, s47 // (op 3613) 


s_mul_hi_i32 s1, s0, s14 // (op 3613) 
s_add_i32 s47, s1, s0 // (op 3613) 

s_ashr_i32 s1, s47, 31 // (op 3613) 
s_cmp_eq_i32 s14, 0 // (op 3613) 
s_mov_b32 s18, scc // (op 3613) 
s_cmp_eq_i32 s18, 0 // (op 3613) 
s_cselect_b32 s19, 0, -1 // (op 3613) 

s_and_b32 s18, s15, 31 // (op 3613) 
s_lshl_b32 s22, 1, s18 // (op 3613) 
s_add_i32 s23, s22, s19 // (op 3613) 


s_and_b32 s19, s1, s23 // (op 3613) 


s_add_i32 s1, s47, s19 // (op 3613) 


s_ashr_i32 s19, s1, s18 // (op 3613) 


s_lshl_b32 s1, s15, 24 // (op 3613) 
s_ashr_i32 s18, s1, 31 // (op 3613) 

s_xor_b32 s1, s19, s18 // (op 3613) 

s_sub_i32 s19, s1, s18 // (op 3613) 


s_add_u32 s22, s4, 16 // (op 3613) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 3613) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 3613) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 3613) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 3613) 

s_mul_i32 s1, s19, s22 // (op 3613) 

s_sub_i32 s47, s0, s1 // (op 3613) 


s_mul_hi_i32 s0, s16, s9 // (op 3613) 
s_add_i32 s1, s0, s16 // (op 3613) 

s_ashr_i32 s0, s1, 31 // (op 3613) 
s_cmp_eq_i32 s9, 0 // (op 3613) 
s_mov_b32 s18, scc // (op 3613) 
s_cmp_eq_i32 s18, 0 // (op 3613) 
s_cselect_b32 s19, 0, -1 // (op 3613) 

s_and_b32 s18, s45, 31 // (op 3613) 
s_lshl_b32 s24, 1, s18 // (op 3613) 
s_add_i32 s25, s24, s19 // (op 3613) 


s_and_b32 s19, s0, s25 // (op 3613) 


s_add_i32 s0, s1, s19 // (op 3613) 


s_ashr_i32 s1, s0, s18 // (op 3613) 


s_lshl_b32 s0, s45, 24 // (op 3613) 
s_ashr_i32 s18, s0, 31 // (op 3613) 

s_xor_b32 s0, s1, s18 // (op 3613) 

s_sub_i32 s1, s0, s18 // (op 3613) 


s_mul_i32 s0, s11, s22 // (op 3613) 

s_mul_i32 s18, s8, s0 // (op 3613) 

s_mul_i32 s0, s1, s18 // (op 3613) 

s_sub_u32 s18, s16, s0 // (op 3613) 

s_mul_hi_i32 s0, s18, s12 // (op 3613) 
s_add_i32 s19, s0, s18 // (op 3613) 


s_ashr_i32 s0, s19, 31 // (op 3613) 
s_cmp_eq_i32 s12, 0 // (op 3613) 
s_mov_b32 s18, scc // (op 3613) 
s_cmp_eq_i32 s18, 0 // (op 3613) 
s_cselect_b32 s22, 0, -1 // (op 3613) 

s_and_b32 s18, s44, 31 // (op 3613) 
s_lshl_b32 s23, 1, s18 // (op 3613) 
s_add_i32 s24, s23, s22 // (op 3613) 


s_and_b32 s25, s0, s24 // (op 3613) 


s_add_i32 s0, s19, s25 // (op 3613) 


s_ashr_i32 s19, s0, s18 // (op 3613) 


s_lshl_b32 s0, s44, 24 // (op 3613) 
s_ashr_i32 s18, s0, 31 // (op 3613) 

s_xor_b32 s0, s19, s18 // (op 3613) 

s_sub_i32 s19, s0, s18 // (op 3613) 


s_cmp_eq_i32 s1, 0 // (op 3613) 
s_mov_b32 s0, scc // (op 3613) 

s_cmp_eq_i32 s0, 0 // (op 3613) 
s_cselect_b32 s1, s19, s47 // (op 3613) 



v_lshl_add_u32 v18, s1, 4, v14 // (op 3613) 


v_mul_lo_u32 v29, v18, s31 // (op 3613) most significant: low of low * high
v_mul_hi_u32 v14, v18, s30 // (op 3613) most significant: high of low * low
v_mul_lo_u32 v28, v18, s30 // (op 3613) least significant: low of low * low
v_add_u32 v29, v29, v14 // (op 3613) most significant: sum


s_ashr_i64 s[0:1], s[42:43], 7 // (op 3613) 
s_lshr_b32 s47, s0, 1 // (op 3613) 

v_lshl_add_u32 v14, s47, 8, v15 // (op 3613) 


v_mov_b32 v19, 0 // (op 3613) convert
v_mov_b32 v18, v14 // (op 3613) convert

v_add_co_u32 v14, s[0:1], v18, v28 // (op 3613) least significant half
v_addc_co_u32 v15, s[0:1], v19, v29, s[0:1] // (op 3613) most significant half



s_mul_i32 s1, s30, s5 // (op 3615) most significant: low of low * high
s_mul_i32 s47, s31, s4 // (op 3615) most significant: low of high * low
s_mul_i32 s0, s30, s4 // (op 3615) least significant
s_mul_hi_u32 s18, s30, s4 // (op 3615) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 3615) most significant: sum
s_add_u32 s1, s1, s18 // (op 3615) most significant: sum


s_mov_b32 s60, s32 // (op 3615) reg expression
s_mov_b32 s61, s33 // (op 3615) reg expression
s_mov_b32 s62, s0 // (op 3615) reg expression
s_mov_b32 s63, 131072 // (op 3615) call()

s_mul_hi_i32 s47, s16, s10 // (op 3677) 
v_lshlrev_b32 v18, 1, v1 // (op 3677) 
s_add_i32 s0, s47, s16 // (op 3677) 

s_ashr_i32 s1, s0, 31 // (op 3677) 
s_cmp_eq_i32 s10, 0 // (op 3677) 
s_mov_b32 s47, scc // (op 3677) 
s_cmp_eq_i32 s47, 0 // (op 3677) 
s_cselect_b32 s18, 0, -1 // (op 3677) 

s_and_b32 s19, s13, 31 // (op 3677) 
s_lshl_b32 s47, 1, s19 // (op 3677) 
s_add_i32 s22, s47, s18 // (op 3677) 


s_and_b32 s18, s1, s22 // (op 3677) 


s_add_i32 s1, s0, s18 // (op 3677) 


s_ashr_i32 s0, s1, s19 // (op 3677) 


s_lshl_b32 s1, s13, 24 // (op 3677) 
s_ashr_i32 s47, s1, 31 // (op 3677) 

s_xor_b32 s1, s0, s47 // (op 3677) 

s_sub_i32 s0, s1, s47 // (op 3677) 


s_mul_hi_i32 s1, s0, s14 // (op 3677) 
s_add_i32 s47, s1, s0 // (op 3677) 

s_ashr_i32 s1, s47, 31 // (op 3677) 
s_cmp_eq_i32 s14, 0 // (op 3677) 
s_mov_b32 s18, scc // (op 3677) 
s_cmp_eq_i32 s18, 0 // (op 3677) 
s_cselect_b32 s19, 0, -1 // (op 3677) 

s_and_b32 s18, s15, 31 // (op 3677) 
s_lshl_b32 s22, 1, s18 // (op 3677) 
s_add_i32 s23, s22, s19 // (op 3677) 


s_and_b32 s19, s1, s23 // (op 3677) 


s_add_i32 s1, s47, s19 // (op 3677) 


s_ashr_i32 s19, s1, s18 // (op 3677) 


s_lshl_b32 s1, s15, 24 // (op 3677) 
s_ashr_i32 s18, s1, 31 // (op 3677) 

s_xor_b32 s1, s19, s18 // (op 3677) 

s_sub_i32 s19, s1, s18 // (op 3677) 


s_mul_i32 s1, s0, s11 // (op 3677) 

s_sub_u32 s0, s16, s1 // (op 3677) 

s_mul_i32 s1, s19, s11 // (op 3677) 
s_add_u32 s1, s1, s0 // (op 3677) 


s_mul_hi_i32 s0, s16, s9 // (op 3677) 
s_add_i32 s47, s0, s16 // (op 3677) 

s_ashr_i32 s0, s47, 31 // (op 3677) 
s_cmp_eq_i32 s9, 0 // (op 3677) 
s_mov_b32 s18, scc // (op 3677) 
s_cmp_eq_i32 s18, 0 // (op 3677) 
s_cselect_b32 s19, 0, -1 // (op 3677) 

s_and_b32 s18, s45, 31 // (op 3677) 
s_lshl_b32 s22, 1, s18 // (op 3677) 
s_add_i32 s23, s22, s19 // (op 3677) 


s_and_b32 s19, s0, s23 // (op 3677) 


s_add_i32 s0, s47, s19 // (op 3677) 


s_ashr_i32 s19, s0, s18 // (op 3677) 


s_lshl_b32 s0, s45, 24 // (op 3677) 
s_ashr_i32 s18, s0, 31 // (op 3677) 

s_xor_b32 s0, s19, s18 // (op 3677) 

s_sub_i32 s19, s0, s18 // (op 3677) 


s_add_u32 s22, s4, 16 // (op 3677) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 3677) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 3677) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 3677) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 3677) 

s_mul_i32 s0, s11, s22 // (op 3677) 

s_mul_i32 s18, s8, s0 // (op 3677) 

s_mul_i32 s0, s19, s18 // (op 3677) 

s_sub_u32 s18, s16, s0 // (op 3677) 

s_mov_b32 s0, s18 // (op 3677) 
s_mul_hi_i32 s47, s0, s12 // (op 3677) 
s_add_i32 s22, s47, s0 // (op 3677) 


s_ashr_i32 s0, s22, 31 // (op 3677) 
s_cmp_eq_i32 s12, 0 // (op 3677) 
s_mov_b32 s47, scc // (op 3677) 
s_cmp_eq_i32 s47, 0 // (op 3677) 
s_cselect_b32 s23, 0, -1 // (op 3677) 

s_and_b32 s47, s44, 31 // (op 3677) 
s_lshl_b32 s24, 1, s47 // (op 3677) 
s_add_i32 s25, s24, s23 // (op 3677) 


s_and_b32 s23, s0, s25 // (op 3677) 


s_add_i32 s0, s22, s23 // (op 3677) 


s_ashr_i32 s22, s0, s47 // (op 3677) 


s_lshl_b32 s0, s44, 24 // (op 3677) 
s_ashr_i32 s47, s0, 31 // (op 3677) 

s_xor_b32 s0, s22, s47 // (op 3677) 

s_sub_i32 s22, s0, s47 // (op 3677) 


s_mul_i32 s0, s46, s11 // (op 3677) 
s_add_u32 s24, s6, 256 // (op 3677) Start of Int64 add, locking SCC
s_addc_u32 s25, s7, 0 // (op 3677) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 3677) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 3677) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 8 // (op 3677) 

s_sub_i32 s23, s24, s0 // (op 3677) 

s_mul_i32 s47, s22, s23 // (op 3677) 


s_sub_u32 s22, s18, s47 // (op 3677) 


s_add_u32 s18, s22, s0 // (op 3677) 


s_cmp_eq_i32 s19, 0 // (op 3677) 
s_mov_b32 s0, scc // (op 3677) 

s_cmp_eq_i32 s0, 0 // (op 3677) 
s_cselect_b32 s19, s18, s1 // (op 3677) 



v_lshl_add_u32 v19, s19, 8, v18 // (op 3677) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 3677) 
s_lshr_b32 s47, s0, 1 // (op 3677) 

s_lshl_b32 s0, s47, 3 // (op 3677) 

v_mul_lo_u32 v18, v19, s34 // (op 3677) 
v_add_u32 v18, v18, s0 // (op 3677) 


s_mul_i32 s1, s34, s7 // (op 3679) most significant: low of low * high
s_mul_i32 s47, s35, s6 // (op 3679) most significant: low of high * low
s_mul_i32 s0, s34, s6 // (op 3679) least significant
s_mul_hi_u32 s18, s34, s6 // (op 3679) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 3679) most significant: sum
s_add_u32 s1, s1, s18 // (op 3679) most significant: sum


s_mov_b32 s64, s36 // (op 3679) reg expression
s_mov_b32 s65, s37 // (op 3679) reg expression
s_mov_b32 s66, s0 // (op 3679) reg expression
s_mov_b32 s67, 131072 // (op 3679) call()

s_mul_hi_i32 s47, s16, s10 // (op 3741) 
v_and_b32 v19, 15, v1 // (op 3741) 
s_add_i32 s0, s47, s16 // (op 3741) 

v_and_b32 v20, 63, v1 // (op 3741) 
s_ashr_i32 s1, s0, 31 // (op 3741) 
v_lshrrev_b32 v21, 4, v20 // (op 3741) 

s_cmp_eq_i32 s10, 0 // (op 3741) 
s_mov_b32 s47, scc // (op 3741) 
s_cmp_eq_i32 s47, 0 // (op 3741) 
s_cselect_b32 s18, 0, -1 // (op 3741) 

s_and_b32 s19, s13, 31 // (op 3741) 
s_lshl_b32 s47, 1, s19 // (op 3741) 
s_add_i32 s22, s47, s18 // (op 3741) 


s_and_b32 s18, s1, s22 // (op 3741) 


s_add_i32 s1, s0, s18 // (op 3741) 


s_ashr_i32 s0, s1, s19 // (op 3741) 


s_lshl_b32 s1, s13, 24 // (op 3741) 
s_ashr_i32 s47, s1, 31 // (op 3741) 

s_xor_b32 s1, s0, s47 // (op 3741) 

s_sub_i32 s0, s1, s47 // (op 3741) 


s_mul_hi_i32 s1, s0, s14 // (op 3741) 
s_add_i32 s47, s1, s0 // (op 3741) 

s_ashr_i32 s1, s47, 31 // (op 3741) 
s_cmp_eq_i32 s14, 0 // (op 3741) 
s_mov_b32 s18, scc // (op 3741) 
s_cmp_eq_i32 s18, 0 // (op 3741) 
s_cselect_b32 s19, 0, -1 // (op 3741) 

s_and_b32 s18, s15, 31 // (op 3741) 
s_lshl_b32 s22, 1, s18 // (op 3741) 
s_add_i32 s23, s22, s19 // (op 3741) 


s_and_b32 s19, s1, s23 // (op 3741) 


s_add_i32 s1, s47, s19 // (op 3741) 


s_ashr_i32 s19, s1, s18 // (op 3741) 


s_lshl_b32 s1, s15, 24 // (op 3741) 
s_ashr_i32 s18, s1, 31 // (op 3741) 

s_xor_b32 s1, s19, s18 // (op 3741) 

s_sub_i32 s19, s1, s18 // (op 3741) 


s_add_u32 s22, s4, 16 // (op 3741) Start of Int64 add, locking SCC
s_addc_u32 s23, s5, 0 // (op 3741) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 3741) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 3741) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 4 // (op 3741) 

s_mul_i32 s1, s19, s22 // (op 3741) 

s_sub_i32 s47, s0, s1 // (op 3741) 


s_mul_hi_i32 s0, s16, s9 // (op 3741) 
s_add_i32 s1, s0, s16 // (op 3741) 

s_ashr_i32 s0, s1, 31 // (op 3741) 
s_cmp_eq_i32 s9, 0 // (op 3741) 
s_mov_b32 s18, scc // (op 3741) 
s_cmp_eq_i32 s18, 0 // (op 3741) 
s_cselect_b32 s19, 0, -1 // (op 3741) 

s_and_b32 s18, s45, 31 // (op 3741) 
s_lshl_b32 s24, 1, s18 // (op 3741) 
s_add_i32 s25, s24, s19 // (op 3741) 


s_and_b32 s19, s0, s25 // (op 3741) 


s_add_i32 s0, s1, s19 // (op 3741) 


s_ashr_i32 s1, s0, s18 // (op 3741) 


s_lshl_b32 s0, s45, 24 // (op 3741) 
s_ashr_i32 s18, s0, 31 // (op 3741) 

s_xor_b32 s0, s1, s18 // (op 3741) 

s_sub_i32 s1, s0, s18 // (op 3741) 


s_mul_i32 s0, s11, s22 // (op 3741) 

s_mul_i32 s18, s8, s0 // (op 3741) 

s_mul_i32 s0, s1, s18 // (op 3741) 

s_sub_u32 s18, s16, s0 // (op 3741) 

s_mul_hi_i32 s0, s18, s12 // (op 3741) 
s_add_i32 s19, s0, s18 // (op 3741) 


s_ashr_i32 s0, s19, 31 // (op 3741) 
s_cmp_eq_i32 s12, 0 // (op 3741) 
s_mov_b32 s18, scc // (op 3741) 
s_cmp_eq_i32 s18, 0 // (op 3741) 
s_cselect_b32 s22, 0, -1 // (op 3741) 

s_and_b32 s18, s44, 31 // (op 3741) 
s_lshl_b32 s23, 1, s18 // (op 3741) 
s_add_i32 s24, s23, s22 // (op 3741) 


s_and_b32 s25, s0, s24 // (op 3741) 


s_add_i32 s0, s19, s25 // (op 3741) 


s_ashr_i32 s19, s0, s18 // (op 3741) 


s_lshl_b32 s0, s44, 24 // (op 3741) 
s_ashr_i32 s18, s0, 31 // (op 3741) 

s_xor_b32 s0, s19, s18 // (op 3741) 

s_sub_i32 s19, s0, s18 // (op 3741) 


s_cmp_eq_i32 s1, 0 // (op 3741) 
s_mov_b32 s0, scc // (op 3741) 

s_cmp_eq_i32 s0, 0 // (op 3741) 
s_cselect_b32 s1, s19, s47 // (op 3741) 



v_lshl_add_u32 v20, s1, 4, v19 // (op 3741) 


v_mul_lo_u32 v19, v20, s38 // (op 3741) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 3741) 
s_lshr_b32 s47, s0, 1 // (op 3741) 

v_lshl_add_u32 v20, s47, 3, v21 // (op 3741) 


v_add_u32 v21, v20, v19 // (op 3741) 


s_mul_i32 s1, s38, s5 // (op 3743) most significant: low of low * high
s_mul_i32 s47, s39, s4 // (op 3743) most significant: low of high * low
s_mul_i32 s0, s38, s4 // (op 3743) least significant
s_mul_hi_u32 s18, s38, s4 // (op 3743) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 3743) most significant: sum
s_add_u32 s1, s1, s18 // (op 3743) most significant: sum


s_mov_b32 s68, s40 // (op 3743) reg expression
s_mov_b32 s69, s41 // (op 3743) reg expression
s_mov_b32 s70, s0 // (op 3743) reg expression
s_mov_b32 s71, 131072 // (op 3743) call()

v_and_b32 v19, -8, v1 // (op 3834) 
v_sub_u32 v20, v1, v19 // (op 3834) 

v_lshlrev_b32 v19, 4, v20 // (op 3834) 

v_lshrrev_b32 v20, 3, v1 // (op 3834) 
v_lshl_add_u32 v22, v20, 7, v19 // (op 3834) 


s_nop 0
v_readfirstlane_b32 s47, v22 // (op 3834) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v19, -8, v1 // (op 3891) 
v_sub_u32 v20, v1, v19 // (op 3891) 

v_lshlrev_b32 v19, 4, v20 // (op 3891) 

v_lshrrev_b32 v20, 3, v1 // (op 3891) 
v_lshl_add_u32 v22, v20, 7, v19 // (op 3891) 


s_nop 0
v_readfirstlane_b32 s0, v22 // (op 3891) Wait state hazard: VALU Write Readlane Hazard

v_lshlrev_b32 v19, 3, v1 // (op 3943) 
s_ashr_i64 s[18:19], s[42:43], 7 // (op 1288) 
s_and_b32 s1, s18, 4294967294 // (op 1288) 

s_ashr_i64 s[18:19], s[42:43], 7 // (op 1287) 
s_mov_b32 s22, s18 // (op 1287) convert

s_cmp_lt_u32 s1, s22 // (op 1287) 













s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_1287 // (op 1287) Condition: Top (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_1287 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ForLoopTop_KLoopTail_1287:
s_barrier  // (op 4171) 
s_add_u32 s1, s1, 1 // (op 1289) 

s_lshl_b64 s[18:19], s[26:27], 4 // (op 3491) 
v_mov_b64 v[16:17], s[18:19] // (op 3491) convert


v_mov_b64 v[22:23], v[12:13] // (op 3491) 
v_mov_b64 v[24:25], v[22:23] // (op 3491) 
s_add_u32 s18, 2048, s47 // (op 3491) 
s_mov_b32 m0, s18 // (op 3491) reg expression
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3491) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v16 // (op 3491) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 3491) most significant half

s_add_u32 m0, m0, 2048 // (op 3491) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3491) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v16 // (op 3491) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 3491) most significant half

s_add_u32 m0, m0, 2048 // (op 3491) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3491) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v16 // (op 3491) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 3491) most significant half

s_add_u32 m0, m0, 2048 // (op 3491) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3491) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v16 // (op 3491) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 3491) most significant half

s_add_u32 m0, m0, 2048 // (op 3491) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3491) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v16 // (op 3491) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 3491) most significant half

s_add_u32 m0, m0, 2048 // (op 3491) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3491) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v16 // (op 3491) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 3491) most significant half

s_add_u32 m0, m0, 2048 // (op 3491) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3491) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v16 // (op 3491) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 3491) most significant half

s_add_u32 m0, m0, 2048 // (op 3491) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3491) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v16 // (op 3491) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 3491) most significant half

s_add_u32 m0, m0, 2048 // (op 3491) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3491) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v16 // (op 3491) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 3491) most significant half

s_add_u32 m0, m0, 2048 // (op 3491) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3491) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v16 // (op 3491) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 3491) most significant half

s_add_u32 m0, m0, 2048 // (op 3491) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3491) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v16 // (op 3491) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 3491) most significant half

s_add_u32 m0, m0, 2048 // (op 3491) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3491) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v16 // (op 3491) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 3491) most significant half

s_add_u32 m0, m0, 2048 // (op 3491) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3491) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v16 // (op 3491) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 3491) most significant half

s_add_u32 m0, m0, 2048 // (op 3491) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3491) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v16 // (op 3491) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 3491) most significant half

s_add_u32 m0, m0, 2048 // (op 3491) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3491) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v16 // (op 3491) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 3491) most significant half

s_add_u32 m0, m0, 2048 // (op 3491) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 3491) Load value direct to lds




v_add_co_u32 v12, vcc, 128, v12 // (op 3529) least significant half
v_addc_co_u32 v13, vcc, 0, v13, vcc // (op 3529) most significant half
s_barrier  // (op 4187) 


v_mov_b64 v[16:17], v[14:15] // (op 3497) 

s_mov_b32 m0, s0 // (op 3497) reg expression
buffer_load_dwordx4 v16, s[60:63], 0 offen offset: 0 lds // (op 3497) Load value direct to lds

v_add_co_u32 v14, vcc, 128, v14 // (op 3605) least significant half
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 3605) most significant half

v_mov_b64 v[16:17], s[34:35] // (op 180) convert

v_mov_b32 v20, v18 // (op 180) 
v_mov_b32 v22, v20 // (op 180) 
buffer_load_dword v24, v22, s[64:67], 0 offen // (op 180) Load value
v_add_u32 v22, v22, v16 // (op 180) 
buffer_load_dword v25, v22, s[64:67], 0 offen // (op 180) Load value



v_add_u32 v18, 4, v18 // (op 3669) 
s_barrier  // (op 4203) 
s_waitcnt vmcnt(3)
s_barrier  // (op 4177) 






v_add_u32 v20, 69632, v19 // (op 178) 
ds_read_b128 v[26:29], v7 offset:2048 // (op 169) Load local data 
ds_read_b128 v[30:33], v7 offset:2112 // (op 169) Load local data 
s_waitcnt vmcnt(1)
ds_write_b32 v20, v24 // (op 178) Store local data 

s_barrier  // (op 4193) 
v_add_u32 v20, 69636, v19 // (op 178) 



ds_read_b128 v[34:37], v4 // (op 171) Load local data 
ds_read_b128 v[38:41], v4 offset:64 // (op 171) Load local data 
s_waitcnt vmcnt(0)
ds_write_b32 v20, v25 // (op 178) Store local data 


s_waitcnt lgkmcnt(0)
s_barrier  // (op 4210) 



v_add_u32 v20, 69632, v0 // (op 173) 
ds_read_u8 v16, v20 // (op 173) Load local data 



v_mov_b32 v17, v21 // (op 175) 

buffer_load_ubyte v20, v17, s[68:71], 0 offen offset: 0 // (op 175) Load value

v_add_u32 v21, 4, v21 // (op 3733) 



ds_read_b128 v[42:45], v7 offset:6144 // (op 321) Load local data 
ds_read_b128 v[46:49], v7 offset:6208 // (op 321) Load local data 
s_waitcnt vmcnt(0) lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:33], v[34:41], a[0:3], v16, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 176) 





v_add_u32 v16, 69760, v0 // (op 325) 
ds_read_u8 v17, v16 // (op 325) Load local data 




ds_read_b128 v[22:25], v7 offset:10240 // (op 483) Load local data 
ds_read_b128 v[26:29], v7 offset:10304 // (op 483) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[42:49], v[34:41], a[4:7], v17, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 327) 





v_add_u32 v16, 69888, v0 // (op 487) 
ds_read_u8 v17, v16 // (op 487) Load local data 




ds_read_b128 v[42:45], v7 offset:14336 // (op 645) Load local data 
ds_read_b128 v[46:49], v7 offset:14400 // (op 645) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[22:29], v[34:41], a[8:11], v17, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 489) 





v_add_u32 v16, 70016, v0 // (op 649) 
ds_read_u8 v17, v16 // (op 649) Load local data 




ds_read_b128 v[22:25], v7 offset:18432 // (op 807) Load local data 
ds_read_b128 v[26:29], v7 offset:18496 // (op 807) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[42:49], v[34:41], a[12:15], v17, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 651) 





v_add_u32 v16, 70144, v0 // (op 811) 
ds_read_u8 v17, v16 // (op 811) Load local data 




ds_read_b128 v[42:45], v7 offset:22528 // (op 969) Load local data 
ds_read_b128 v[46:49], v7 offset:22592 // (op 969) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[22:29], v[34:41], a[16:19], v17, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 813) 





v_add_u32 v16, 70272, v0 // (op 973) 
ds_read_u8 v17, v16 // (op 973) Load local data 




ds_read_b128 v[22:25], v7 offset:26624 // (op 1131) Load local data 
ds_read_b128 v[26:29], v7 offset:26688 // (op 1131) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[42:49], v[34:41], a[20:23], v17, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 975) 





v_add_u32 v16, 70400, v0 // (op 1135) 
ds_read_u8 v17, v16 // (op 1135) Load local data 




ds_read_b128 v[42:45], v7 offset:30720 // (op 1293) Load local data 
ds_read_b128 v[46:49], v7 offset:30784 // (op 1293) Load local data 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[22:29], v[34:41], a[24:27], v17, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1137) 





v_add_u32 v16, 70528, v0 // (op 1297) 
ds_read_u8 v17, v16 // (op 1297) Load local data 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[42:49], v[34:41], a[28:31], v17, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b000 blgp:0b000 // (op 1299) 




s_ashr_i64 s[18:19], s[42:43], 7 // (op 1287) 
s_mov_b32 s22, s18 // (op 1287) convert

s_cmp_lt_u32 s1, s22 // (op 1287) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ForLoopTop_KLoopTail_1287 // (op 1287) Condition: Bottom (jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ForLoopTop_KLoopTail_1287 if true)
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ForLoopBottom_KLoopTail_1287:





















s_barrier  // (op 4292) 



















v_cmp_eq_f32 vcc, 0.00000, v8 // (op 123) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ConditionalFalse_0_beta_123 // (op 123) Condition: False, jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ConditionalFalse_0_beta_123
v_accvgpr_read v10, a0 // (op 46) 
v_accvgpr_read v11, a1 // (op 46) 
v_accvgpr_read v12, a2 // (op 46) 
v_accvgpr_read v13, a3 // (op 46) 
v_mul_f32 v14, v9, v10 // (op 46) 
v_mul_f32 v15, v9, v11 // (op 46) 
v_mul_f32 v16, v9, v12 // (op 46) 
v_mul_f32 v17, v9, v13 // (op 46) 




buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen // (op 52) Store value

v_accvgpr_read v10, a4 // (op 1617) 
v_accvgpr_read v11, a5 // (op 1617) 
v_accvgpr_read v12, a6 // (op 1617) 
v_accvgpr_read v13, a7 // (op 1617) 
v_mul_f32 v14, v9, v10 // (op 1617) 
v_mul_f32 v15, v9, v11 // (op 1617) 
v_mul_f32 v16, v9, v12 // (op 1617) 
v_mul_f32 v17, v9, v13 // (op 1617) 




buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen offset:128 // (op 1619) Store value

v_accvgpr_read v10, a8 // (op 1635) 
v_accvgpr_read v11, a9 // (op 1635) 
v_accvgpr_read v12, a10 // (op 1635) 
v_accvgpr_read v13, a11 // (op 1635) 
v_mul_f32 v14, v9, v10 // (op 1635) 
v_mul_f32 v15, v9, v11 // (op 1635) 
v_mul_f32 v16, v9, v12 // (op 1635) 
v_mul_f32 v17, v9, v13 // (op 1635) 




buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen offset:256 // (op 1637) Store value

v_accvgpr_read v10, a12 // (op 1653) 
v_accvgpr_read v11, a13 // (op 1653) 
v_accvgpr_read v12, a14 // (op 1653) 
v_accvgpr_read v13, a15 // (op 1653) 
v_mul_f32 v14, v9, v10 // (op 1653) 
v_mul_f32 v15, v9, v11 // (op 1653) 
v_mul_f32 v16, v9, v12 // (op 1653) 
v_mul_f32 v17, v9, v13 // (op 1653) 




buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen offset:384 // (op 1655) Store value

v_accvgpr_read v10, a16 // (op 1671) 
v_accvgpr_read v11, a17 // (op 1671) 
v_accvgpr_read v12, a18 // (op 1671) 
v_accvgpr_read v13, a19 // (op 1671) 
v_mul_f32 v14, v9, v10 // (op 1671) 
v_mul_f32 v15, v9, v11 // (op 1671) 
v_mul_f32 v16, v9, v12 // (op 1671) 
v_mul_f32 v17, v9, v13 // (op 1671) 




buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen offset:512 // (op 1673) Store value

v_accvgpr_read v10, a20 // (op 1689) 
v_accvgpr_read v11, a21 // (op 1689) 
v_accvgpr_read v12, a22 // (op 1689) 
v_accvgpr_read v13, a23 // (op 1689) 
v_mul_f32 v14, v9, v10 // (op 1689) 
v_mul_f32 v15, v9, v11 // (op 1689) 
v_mul_f32 v16, v9, v12 // (op 1689) 
v_mul_f32 v17, v9, v13 // (op 1689) 




buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen offset:640 // (op 1691) Store value

v_accvgpr_read v10, a24 // (op 1707) 
v_accvgpr_read v11, a25 // (op 1707) 
v_accvgpr_read v12, a26 // (op 1707) 
v_accvgpr_read v13, a27 // (op 1707) 
v_mul_f32 v14, v9, v10 // (op 1707) 
v_mul_f32 v15, v9, v11 // (op 1707) 
v_mul_f32 v16, v9, v12 // (op 1707) 
v_mul_f32 v17, v9, v13 // (op 1707) 




buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen offset:768 // (op 1709) Store value

v_accvgpr_read v10, a28 // (op 1725) 
v_accvgpr_read v11, a29 // (op 1725) 
v_accvgpr_read v12, a30 // (op 1725) 
v_accvgpr_read v13, a31 // (op 1725) 
v_mul_f32 v14, v9, v10 // (op 1725) 
v_mul_f32 v15, v9, v11 // (op 1725) 
v_mul_f32 v16, v9, v12 // (op 1725) 
v_mul_f32 v17, v9, v13 // (op 1725) 




buffer_store_dwordx4 v[14:17], v3, s[56:59], 0 offen offset:896 // (op 1727) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ConditionalBottom_0_beta_123 // (op 123) Condition: Done, jump to RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ConditionalBottom_0_beta_123
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ConditionalFalse_0_beta_123:



buffer_load_dwordx4 v[10:13], v6, s[52:55], 0 offen offset:896 // (op 1893) Load value
v_accvgpr_read v14, a28 // (op 1890) 
v_accvgpr_read v15, a29 // (op 1890) 
v_accvgpr_read v16, a30 // (op 1890) 
v_accvgpr_read v17, a31 // (op 1890) 
v_mul_f32 v18, v9, v14 // (op 1890) 
v_mul_f32 v19, v9, v15 // (op 1890) 
v_mul_f32 v20, v9, v16 // (op 1890) 
v_mul_f32 v21, v9, v17 // (op 1890) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v8, v10 // (op 1894) 
v_mul_f32 v15, v8, v11 // (op 1894) 
v_mul_f32 v16, v8, v12 // (op 1894) 
v_mul_f32 v17, v8, v13 // (op 1894) 

v_add_f32 v10, v14, v18 // (op 1891) 
v_add_f32 v11, v15, v19 // (op 1891) 
v_add_f32 v12, v16, v20 // (op 1891) 
v_add_f32 v13, v17, v21 // (op 1891) 





buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen offset:896 // (op 1892) Store value




s_nop 1
buffer_load_dwordx4 v[10:13], v6, s[52:55], 0 offen offset:768 // (op 1875) Load value
v_accvgpr_read v14, a24 // (op 1872) 
v_accvgpr_read v15, a25 // (op 1872) 
v_accvgpr_read v16, a26 // (op 1872) 
v_accvgpr_read v17, a27 // (op 1872) 
v_mul_f32 v18, v9, v14 // (op 1872) 
v_mul_f32 v19, v9, v15 // (op 1872) 
v_mul_f32 v20, v9, v16 // (op 1872) 
v_mul_f32 v21, v9, v17 // (op 1872) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v8, v10 // (op 1876) 
v_mul_f32 v15, v8, v11 // (op 1876) 
v_mul_f32 v16, v8, v12 // (op 1876) 
v_mul_f32 v17, v8, v13 // (op 1876) 

v_add_f32 v10, v14, v18 // (op 1873) 
v_add_f32 v11, v15, v19 // (op 1873) 
v_add_f32 v12, v16, v20 // (op 1873) 
v_add_f32 v13, v17, v21 // (op 1873) 





buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen offset:768 // (op 1874) Store value




s_nop 1
buffer_load_dwordx4 v[10:13], v6, s[52:55], 0 offen offset:640 // (op 1857) Load value
v_accvgpr_read v14, a20 // (op 1854) 
v_accvgpr_read v15, a21 // (op 1854) 
v_accvgpr_read v16, a22 // (op 1854) 
v_accvgpr_read v17, a23 // (op 1854) 
v_mul_f32 v18, v9, v14 // (op 1854) 
v_mul_f32 v19, v9, v15 // (op 1854) 
v_mul_f32 v20, v9, v16 // (op 1854) 
v_mul_f32 v21, v9, v17 // (op 1854) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v8, v10 // (op 1858) 
v_mul_f32 v15, v8, v11 // (op 1858) 
v_mul_f32 v16, v8, v12 // (op 1858) 
v_mul_f32 v17, v8, v13 // (op 1858) 

v_add_f32 v10, v14, v18 // (op 1855) 
v_add_f32 v11, v15, v19 // (op 1855) 
v_add_f32 v12, v16, v20 // (op 1855) 
v_add_f32 v13, v17, v21 // (op 1855) 





buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen offset:640 // (op 1856) Store value




s_nop 1
buffer_load_dwordx4 v[10:13], v6, s[52:55], 0 offen offset:512 // (op 1839) Load value
v_accvgpr_read v14, a16 // (op 1836) 
v_accvgpr_read v15, a17 // (op 1836) 
v_accvgpr_read v16, a18 // (op 1836) 
v_accvgpr_read v17, a19 // (op 1836) 
v_mul_f32 v18, v9, v14 // (op 1836) 
v_mul_f32 v19, v9, v15 // (op 1836) 
v_mul_f32 v20, v9, v16 // (op 1836) 
v_mul_f32 v21, v9, v17 // (op 1836) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v8, v10 // (op 1840) 
v_mul_f32 v15, v8, v11 // (op 1840) 
v_mul_f32 v16, v8, v12 // (op 1840) 
v_mul_f32 v17, v8, v13 // (op 1840) 

v_add_f32 v10, v14, v18 // (op 1837) 
v_add_f32 v11, v15, v19 // (op 1837) 
v_add_f32 v12, v16, v20 // (op 1837) 
v_add_f32 v13, v17, v21 // (op 1837) 





buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen offset:512 // (op 1838) Store value




s_nop 1
buffer_load_dwordx4 v[10:13], v6, s[52:55], 0 offen offset:384 // (op 1821) Load value
v_accvgpr_read v14, a12 // (op 1818) 
v_accvgpr_read v15, a13 // (op 1818) 
v_accvgpr_read v16, a14 // (op 1818) 
v_accvgpr_read v17, a15 // (op 1818) 
v_mul_f32 v18, v9, v14 // (op 1818) 
v_mul_f32 v19, v9, v15 // (op 1818) 
v_mul_f32 v20, v9, v16 // (op 1818) 
v_mul_f32 v21, v9, v17 // (op 1818) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v8, v10 // (op 1822) 
v_mul_f32 v15, v8, v11 // (op 1822) 
v_mul_f32 v16, v8, v12 // (op 1822) 
v_mul_f32 v17, v8, v13 // (op 1822) 

v_add_f32 v10, v14, v18 // (op 1819) 
v_add_f32 v11, v15, v19 // (op 1819) 
v_add_f32 v12, v16, v20 // (op 1819) 
v_add_f32 v13, v17, v21 // (op 1819) 





buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen offset:384 // (op 1820) Store value




s_nop 1
buffer_load_dwordx4 v[10:13], v6, s[52:55], 0 offen offset:256 // (op 1803) Load value
v_accvgpr_read v14, a8 // (op 1800) 
v_accvgpr_read v15, a9 // (op 1800) 
v_accvgpr_read v16, a10 // (op 1800) 
v_accvgpr_read v17, a11 // (op 1800) 
v_mul_f32 v18, v9, v14 // (op 1800) 
v_mul_f32 v19, v9, v15 // (op 1800) 
v_mul_f32 v20, v9, v16 // (op 1800) 
v_mul_f32 v21, v9, v17 // (op 1800) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v8, v10 // (op 1804) 
v_mul_f32 v15, v8, v11 // (op 1804) 
v_mul_f32 v16, v8, v12 // (op 1804) 
v_mul_f32 v17, v8, v13 // (op 1804) 

v_add_f32 v10, v14, v18 // (op 1801) 
v_add_f32 v11, v15, v19 // (op 1801) 
v_add_f32 v12, v16, v20 // (op 1801) 
v_add_f32 v13, v17, v21 // (op 1801) 





buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen offset:256 // (op 1802) Store value




s_nop 1
buffer_load_dwordx4 v[10:13], v6, s[52:55], 0 offen offset:128 // (op 1785) Load value
v_accvgpr_read v14, a4 // (op 1782) 
v_accvgpr_read v15, a5 // (op 1782) 
v_accvgpr_read v16, a6 // (op 1782) 
v_accvgpr_read v17, a7 // (op 1782) 
v_mul_f32 v18, v9, v14 // (op 1782) 
v_mul_f32 v19, v9, v15 // (op 1782) 
v_mul_f32 v20, v9, v16 // (op 1782) 
v_mul_f32 v21, v9, v17 // (op 1782) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v8, v10 // (op 1786) 
v_mul_f32 v15, v8, v11 // (op 1786) 
v_mul_f32 v16, v8, v12 // (op 1786) 
v_mul_f32 v17, v8, v13 // (op 1786) 

v_add_f32 v10, v14, v18 // (op 1783) 
v_add_f32 v11, v15, v19 // (op 1783) 
v_add_f32 v12, v16, v20 // (op 1783) 
v_add_f32 v13, v17, v21 // (op 1783) 





buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen offset:128 // (op 1784) Store value




s_nop 1
buffer_load_dwordx4 v[10:13], v6, s[52:55], 0 offen // (op 135) Load value
v_accvgpr_read v14, a0 // (op 132) 
v_accvgpr_read v15, a1 // (op 132) 
v_accvgpr_read v16, a2 // (op 132) 
v_accvgpr_read v17, a3 // (op 132) 
v_mul_f32 v18, v9, v14 // (op 132) 
v_mul_f32 v19, v9, v15 // (op 132) 
v_mul_f32 v20, v9, v16 // (op 132) 
v_mul_f32 v21, v9, v17 // (op 132) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v8, v10 // (op 136) 
v_mul_f32 v15, v8, v11 // (op 136) 
v_mul_f32 v16, v8, v12 // (op 136) 
v_mul_f32 v17, v8, v13 // (op 136) 

v_add_f32 v10, v14, v18 // (op 133) 
v_add_f32 v11, v15, v19 // (op 133) 
v_add_f32 v12, v16, v20 // (op 133) 
v_add_f32 v13, v17, v21 // (op 133) 





buffer_store_dwordx4 v[10:13], v3, s[56:59], 0 offen // (op 134) Store value

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__ConditionalBottom_0_beta_123:
















s_endpgm  // End of RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM_
.LRR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__end:
.size RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM_, .LRR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM__end-RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 148
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 71680 // lds bytes
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
    macrotile: [256, 16, 128]
    threads: [128, 1, 1]
    grid: [TilesXY, One, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [8, 1]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_FP8_FP8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x16x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 71680
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 74
    .vgpr_count: 116
    .agpr_count: 32
    .max_flat_workgroup_size: 128
    .workgroup_size: [128, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 128}}, {is-null: true}, {is-null: true}]
    .dynamic_sharedmemory_bytes: {type: LiteralValue.UInt32, dataType: UInt32, value: 0}
    .args:
      - .name: Tensor_10_stride_1_10
        .size: 8
        .offset: 0
        .expression: {type: CommandArgument, size: 8, offset: 192, name: Tensor_10_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_size_1_4
        .size: 8
        .offset: 8
        .expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_0_size_0_1
        .size: 8
        .offset: 16
        .expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Subtract_21
        .size: 4
        .offset: 24
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 28
        .expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}}
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
        .expression: {type: MagicMultiple, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_29
        .size: 4
        .offset: 52
        .expression: {type: MagicShiftAndSign, arg: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_21_stride_1_16
        .size: 8
        .offset: 56
        .expression: {type: CommandArgument, size: 8, offset: 240, name: Tensor_21_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
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
        .expression: {type: MagicShiftAndSign, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_24, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 160, size: 4}, rhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_27
        .size: 4
        .offset: 156
        .expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 16}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 4}}, dataType: Int32}}}}
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

