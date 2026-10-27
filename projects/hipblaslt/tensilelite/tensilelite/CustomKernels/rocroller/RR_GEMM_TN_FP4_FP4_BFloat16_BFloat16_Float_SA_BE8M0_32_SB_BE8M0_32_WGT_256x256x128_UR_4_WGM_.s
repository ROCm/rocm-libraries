// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT



.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WGM_
.p2align 8
.type RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WGM_,@function
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 41472) 
s_mul_hi_i32 s1, s16, s10 // (op 41611) 
v_and_b32 v0, 4294967168, v1 // (op 41533) 
v_and_b32 v3, 15, v1 // (op 41533) 
v_add_lshl_u32 v4, v0, v3, 7 // (op 41533) 


v_and_b32 v0, 63, v1 // (op 41533) 
v_lshrrev_b32 v3, 4, v0 // (op 41533) 

v_lshlrev_b32 v0, 5, v3 // (op 41533) 

v_add_lshl_u32 v3, v4, v0, 2 // (op 41533) 


v_lshrrev_b32 v0, 3, v3 // (op 41533) 

v_lshrrev_b32 v3, 6, v1 // (op 41590) 
v_and_b32 v4, 1, v3 // (op 41590) 

v_and_b32 v3, 15, v1 // (op 41590) 
v_lshl_add_u32 v5, v4, 7, v3 // (op 41590) 


v_and_b32 v3, 63, v1 // (op 41590) 
v_lshrrev_b32 v4, 4, v3 // (op 41590) 

v_lshlrev_b32 v3, 5, v4 // (op 41590) 

v_lshl_add_u32 v4, v5, 7, v3 // (op 41590) 


v_lshlrev_b32 v3, 2, v4 // (op 41590) 

v_lshrrev_b32 v4, 3, v3 // (op 41590) 

v_and_b32 v3, 4294967168, v1 // (op 41657) 
v_and_b32 v5, 15, v1 // (op 41657) 
v_add_lshl_u32 v6, v3, v5, 7 // (op 41657) 


v_and_b32 v3, 63, v1 // (op 41657) 
v_lshrrev_b32 v5, 4, v3 // (op 41657) 

v_lshlrev_b32 v3, 5, v5 // (op 41657) 

v_add_lshl_u32 v5, v6, v3, 2 // (op 41657) 


v_lshrrev_b32 v3, 3, v5 // (op 41657) 

v_lshrrev_b32 v5, 6, v1 // (op 41701) 
v_and_b32 v6, 1, v5 // (op 41701) 

v_and_b32 v5, 15, v1 // (op 41701) 
v_lshl_add_u32 v7, v6, 7, v5 // (op 41701) 


v_and_b32 v5, 63, v1 // (op 41701) 
v_lshrrev_b32 v6, 4, v5 // (op 41701) 

v_lshlrev_b32 v5, 5, v6 // (op 41701) 

v_lshl_add_u32 v6, v7, 7, v5 // (op 41701) 


v_lshlrev_b32 v5, 2, v6 // (op 41701) 

v_lshrrev_b32 v6, 3, v5 // (op 41701) 

v_and_b32 v5, 4294967168, v1 // (op 41745) 
v_and_b32 v7, 15, v1 // (op 41745) 
v_add_lshl_u32 v8, v5, v7, 7 // (op 41745) 


v_and_b32 v5, 63, v1 // (op 41745) 
v_lshrrev_b32 v7, 4, v5 // (op 41745) 

v_lshlrev_b32 v5, 5, v7 // (op 41745) 

v_add_lshl_u32 v7, v8, v5, 2 // (op 41745) 


v_lshrrev_b32 v5, 3, v7 // (op 41745) 

v_lshrrev_b32 v7, 6, v1 // (op 41789) 
v_and_b32 v8, 1, v7 // (op 41789) 

v_and_b32 v7, 15, v1 // (op 41789) 
v_lshl_add_u32 v9, v8, 7, v7 // (op 41789) 


v_and_b32 v7, 63, v1 // (op 41789) 
v_lshrrev_b32 v8, 4, v7 // (op 41789) 

v_lshlrev_b32 v7, 5, v8 // (op 41789) 

v_lshl_add_u32 v8, v9, 7, v7 // (op 41789) 


v_lshlrev_b32 v7, 2, v8 // (op 41789) 

v_lshrrev_b32 v8, 3, v7 // (op 41789) 

v_and_b32 v7, 4294967168, v1 // (op 41833) 
v_and_b32 v9, 15, v1 // (op 41833) 
v_add_lshl_u32 v10, v7, v9, 7 // (op 41833) 


v_and_b32 v7, 63, v1 // (op 41833) 
v_lshrrev_b32 v9, 4, v7 // (op 41833) 

v_lshlrev_b32 v7, 5, v9 // (op 41833) 

v_add_lshl_u32 v9, v10, v7, 2 // (op 41833) 


v_lshrrev_b32 v7, 3, v9 // (op 41833) 

v_lshrrev_b32 v9, 6, v1 // (op 41877) 
v_and_b32 v10, 1, v9 // (op 41877) 

v_and_b32 v9, 15, v1 // (op 41877) 
v_lshl_add_u32 v11, v10, 7, v9 // (op 41877) 


v_and_b32 v9, 63, v1 // (op 41877) 
v_lshrrev_b32 v10, 4, v9 // (op 41877) 

v_lshlrev_b32 v9, 5, v10 // (op 41877) 

v_lshl_add_u32 v10, v11, 7, v9 // (op 41877) 


v_lshlrev_b32 v9, 2, v10 // (op 41877) 

v_lshrrev_b32 v10, 3, v9 // (op 41877) 

v_and_b32 v9, 4294967168, v1 // (op 41472) 
s_add_i32 s18, s0, s16 // (op 41472) 

v_and_b32 v11, 63, v1 // (op 41472) 
s_ashr_i32 s0, s18, 31 // (op 41472) 
v_lshrrev_b32 v12, 4, v11 // (op 41472) 
s_cmp_eq_i32 s10, 0 // (op 41472) 
s_mov_b32 s19, scc // (op 41472) 
v_lshlrev_b32 v13, 2, v12 // (op 41472) 

s_cmp_eq_i32 s19, 0 // (op 41472) 
s_cselect_b32 s49, 0, -1 // (op 41472) 

v_add_u32 v12, v9, v13 // (op 41472) 


s_and_b32 s19, s13, 31 // (op 41472) 
v_lshrrev_b32 v9, 2, v11 // (op 41472) 

v_and_b32 v11, 3, v9 // (op 41472) 

v_and_b32 v9, 4294967168, v1 // (op 41611) 
v_and_b32 v13, 63, v1 // (op 41611) 
s_lshl_b32 s50, 1, s19 // (op 41472) 
v_and_b32 v14, 3, v1 // (op 41472) 
s_add_i32 s51, s50, s49 // (op 41472) 


s_and_b32 s49, s0, s51 // (op 41472) 


s_add_i32 s0, s1, s16 // (op 41611) 

s_ashr_i32 s1, s0, 31 // (op 41611) 
v_lshl_add_u32 v15, v11, 2, v14 // (op 41472) 


s_add_i32 s50, s18, s49 // (op 41472) 


v_lshrrev_b32 v11, 6, v1 // (op 41472) 
s_ashr_i32 s18, s50, s19 // (op 41472) 


v_and_b32 v14, 1, v11 // (op 41472) 

s_lshl_b32 s19, s13, 24 // (op 41472) 
v_lshl_add_u32 v11, v14, 7, v15 // (op 41472) 


s_ashr_i32 s49, s19, 31 // (op 41472) 

s_xor_b32 s19, s18, s49 // (op 41472) 

s_sub_i32 s18, s19, s49 // (op 41472) 


s_mul_hi_i32 s19, s18, s14 // (op 41472) 
s_add_i32 s49, s19, s18 // (op 41472) 

s_ashr_i32 s19, s49, 31 // (op 41472) 
s_cmp_eq_i32 s14, 0 // (op 41472) 
s_mov_b32 s50, scc // (op 41472) 
s_cmp_eq_i32 s50, 0 // (op 41472) 
s_cselect_b32 s51, 0, -1 // (op 41472) 

s_and_b32 s50, s15, 31 // (op 41472) 
v_lshrrev_b32 v14, 4, v13 // (op 41611) 
s_cmp_eq_i32 s10, 0 // (op 41611) 
s_mov_b32 s52, scc // (op 41611) 
v_lshlrev_b32 v15, 2, v14 // (op 41611) 

s_cmp_eq_i32 s52, 0 // (op 41611) 
s_cselect_b32 s53, 0, -1 // (op 41611) 

s_lshl_b32 s52, 1, s50 // (op 41472) 
v_add_u32 v14, v9, v15 // (op 41611) 


v_lshrrev_b32 v9, 2, v13 // (op 41611) 

s_add_i32 s54, s52, s51 // (op 41472) 


s_and_b32 s51, s19, s54 // (op 41472) 


s_add_i32 s19, s49, s51 // (op 41472) 


s_ashr_i32 s49, s19, s50 // (op 41472) 


s_lshl_b32 s19, s15, 24 // (op 41472) 
s_ashr_i32 s50, s19, 31 // (op 41472) 

s_xor_b32 s19, s49, s50 // (op 41472) 

s_sub_i32 s49, s19, s50 // (op 41472) 


s_mul_i32 s19, s18, s11 // (op 41472) 
s_sub_u32 s50, s16, s19 // (op 41472) 

s_mul_i32 s19, s49, s11 // (op 41472) 
s_add_u32 s19, s19, s50 // (op 41472) 

s_mul_hi_i32 s50, s16, s9 // (op 41472) 
s_add_i32 s51, s50, s16 // (op 41472) 

s_ashr_i32 s50, s51, 31 // (op 41472) 
s_cmp_eq_i32 s9, 0 // (op 41472) 
s_mov_b32 s52, scc // (op 41472) 
s_cmp_eq_i32 s52, 0 // (op 41472) 
s_cselect_b32 s54, 0, -1 // (op 41472) 

s_and_b32 s52, s13, 31 // (op 41611) 
v_and_b32 v13, 3, v9 // (op 41611) 

s_lshl_b32 s55, 1, s52 // (op 41611) 
v_and_b32 v9, 3, v1 // (op 41611) 
s_add_i32 s56, s55, s53 // (op 41611) 


s_and_b32 s53, s1, s56 // (op 41611) 


v_lshl_add_u32 v15, v13, 2, v9 // (op 41611) 


s_add_i32 s1, s0, s53 // (op 41611) 


v_lshrrev_b32 v9, 6, v1 // (op 41611) 
s_ashr_i32 s0, s1, s52 // (op 41611) 


v_and_b32 v13, 1, v9 // (op 41611) 

s_lshl_b32 s1, s13, 24 // (op 41611) 
v_lshl_add_u32 v9, v13, 7, v15 // (op 41611) 


s_ashr_i32 s52, s1, 31 // (op 41611) 

s_xor_b32 s1, s0, s52 // (op 41611) 

s_sub_i32 s0, s1, s52 // (op 41611) 


s_mul_hi_i32 s1, s0, s14 // (op 41611) 
s_add_i32 s52, s1, s0 // (op 41611) 

s_ashr_i32 s1, s52, 31 // (op 41611) 
s_cmp_eq_i32 s14, 0 // (op 41611) 
s_mov_b32 s53, scc // (op 41611) 
s_cmp_eq_i32 s53, 0 // (op 41611) 
s_cselect_b32 s55, 0, -1 // (op 41611) 

s_and_b32 s53, s15, 31 // (op 41611) 
s_lshl_b32 s56, 1, s53 // (op 41611) 
s_add_i32 s57, s56, s55 // (op 41611) 


s_and_b32 s55, s1, s57 // (op 41611) 


s_add_i32 s1, s52, s55 // (op 41611) 


s_ashr_i32 s52, s1, s53 // (op 41611) 


s_lshl_b32 s1, s15, 24 // (op 41611) 
s_ashr_i32 s53, s1, 31 // (op 41611) 

s_xor_b32 s1, s52, s53 // (op 41611) 

s_sub_i32 s52, s1, s53 // (op 41611) 


s_mul_i32 s1, s0, s11 // (op 41611) 
s_sub_u32 s53, s16, s1 // (op 41611) 

s_mul_i32 s1, s52, s11 // (op 41611) 
s_add_u32 s1, s1, s53 // (op 41611) 

s_mul_hi_i32 s53, s16, s9 // (op 41611) 
s_add_i32 s55, s53, s16 // (op 41611) 

s_ashr_i32 s53, s55, 31 // (op 41611) 
s_cmp_eq_i32 s9, 0 // (op 41611) 
s_mov_b32 s56, scc // (op 41611) 
s_cmp_eq_i32 s56, 0 // (op 41611) 
s_cselect_b32 s57, 0, -1 // (op 41611) 


s_waitcnt lgkmcnt(0)
s_and_b32 s56, s45, 31 // (op 41472) 
s_lshl_b32 s58, 1, s56 // (op 41472) 
s_add_i32 s59, s58, s54 // (op 41472) 


s_and_b32 s54, s50, s59 // (op 41472) 


s_add_i32 s50, s51, s54 // (op 41472) 


s_ashr_i32 s51, s50, s56 // (op 41472) 


s_lshl_b32 s50, s45, 24 // (op 41472) 
s_ashr_i32 s54, s50, 31 // (op 41472) 

s_xor_b32 s50, s51, s54 // (op 41472) 

s_sub_i32 s51, s50, s54 // (op 41472) 


s_and_b32 s50, s45, 31 // (op 41611) 
s_lshl_b32 s54, 1, s50 // (op 41611) 
s_add_i32 s56, s54, s57 // (op 41611) 


s_and_b32 s54, s53, s56 // (op 41611) 


s_add_i32 s53, s55, s54 // (op 41611) 


s_ashr_i32 s54, s53, s50 // (op 41611) 


s_lshl_b32 s50, s45, 24 // (op 41611) 
s_ashr_i32 s53, s50, 31 // (op 41611) 

s_xor_b32 s50, s54, s53 // (op 41611) 

s_sub_i32 s54, s50, s53 // (op 41611) 


s_add_u32 s56, s6, 256 // (op 41472) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 41472) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 41472) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 41472) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 8 // (op 41472) 

s_mul_i32 s50, s11, s56 // (op 41472) 
s_mul_i32 s53, s8, s50 // (op 41472) 

s_mul_i32 s50, s51, s53 // (op 41472) 

s_sub_u32 s53, s16, s50 // (op 41472) 

s_mov_b32 s50, s53 // (op 41472) 
s_mul_hi_i32 s55, s50, s12 // (op 41472) 
s_add_i32 s58, s55, s50 // (op 41472) 


s_ashr_i32 s50, s58, 31 // (op 41472) 
s_cmp_eq_i32 s12, 0 // (op 41472) 
s_mov_b32 s55, scc // (op 41472) 
s_cmp_eq_i32 s55, 0 // (op 41472) 
s_cselect_b32 s59, 0, -1 // (op 41472) 

s_and_b32 s55, s44, 31 // (op 41472) 
s_lshl_b32 s60, 1, s55 // (op 41472) 
s_add_i32 s61, s60, s59 // (op 41472) 


s_and_b32 s59, s50, s61 // (op 41472) 


s_add_i32 s50, s58, s59 // (op 41472) 


s_ashr_i32 s58, s50, s55 // (op 41472) 


s_lshl_b32 s50, s44, 24 // (op 41472) 
s_ashr_i32 s55, s50, 31 // (op 41472) 

s_xor_b32 s50, s58, s55 // (op 41472) 

s_sub_i32 s58, s50, s55 // (op 41472) 


s_mul_i32 s50, s46, s11 // (op 41472) 
s_add_u32 s60, s20, 256 // (op 41472) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 41472) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 41472) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 41472) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 8 // (op 41472) 

s_sub_i32 s55, s60, s50 // (op 41472) 

s_mul_i32 s59, s58, s55 // (op 41472) 

s_sub_u32 s55, s53, s59 // (op 41472) 


s_add_u32 s53, s55, s50 // (op 41472) 


s_cmp_eq_i32 s51, 0 // (op 41472) 
s_mov_b32 s50, scc // (op 41472) 

s_cmp_eq_i32 s50, 0 // (op 41472) 
s_cselect_b32 s51, s53, s19 // (op 41472) 


s_mul_i32 s19, s49, s56 // (op 41472) 


v_lshl_add_u32 v13, s51, 8, v12 // (op 41472) 


s_sub_i32 s49, s18, s19 // (op 41472) 


s_cmp_eq_i32 s50, 0 // (op 41472) 
s_cselect_b32 s51, s58, s49 // (op 41472) 



v_lshl_add_u32 v12, s51, 8, v11 // (op 41472) 


v_mul_lo_u32 v11, v12, s2 // (op 41472) 

v_add_lshl_u32 v12, v13, v11, 1 // (op 41472) 


s_add_u32 s18, s6, 256 // (op 41611) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 41611) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 41611) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 41611) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 8 // (op 41611) 

s_mul_i32 s53, s11, s18 // (op 41611) 
s_mul_i32 s49, s8, s53 // (op 41611) 

s_mul_i32 s53, s54, s49 // (op 41611) 

s_sub_u32 s49, s16, s53 // (op 41611) 

s_mov_b32 s53, s49 // (op 41611) 
s_mul_hi_i32 s50, s53, s12 // (op 41611) 
s_add_i32 s51, s50, s53 // (op 41611) 


s_ashr_i32 s50, s51, 31 // (op 41611) 
s_cmp_eq_i32 s12, 0 // (op 41611) 
s_mov_b32 s53, scc // (op 41611) 
s_cmp_eq_i32 s53, 0 // (op 41611) 
s_cselect_b32 s55, 0, -1 // (op 41611) 

s_and_b32 s53, s44, 31 // (op 41611) 
s_lshl_b32 s56, 1, s53 // (op 41611) 
s_add_i32 s57, s56, s55 // (op 41611) 


s_and_b32 s55, s50, s57 // (op 41611) 


s_add_i32 s50, s51, s55 // (op 41611) 


s_ashr_i32 s51, s50, s53 // (op 41611) 


s_lshl_b32 s50, s44, 24 // (op 41611) 
s_ashr_i32 s53, s50, 31 // (op 41611) 

s_xor_b32 s50, s51, s53 // (op 41611) 

s_sub_i32 s51, s50, s53 // (op 41611) 


s_mul_i32 s50, s46, s11 // (op 41611) 
s_add_u32 s56, s20, 256 // (op 41611) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 41611) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 41611) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 41611) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 8 // (op 41611) 

s_sub_i32 s53, s56, s50 // (op 41611) 

s_mul_i32 s55, s51, s53 // (op 41611) 

s_sub_u32 s53, s49, s55 // (op 41611) 


s_add_u32 s49, s53, s50 // (op 41611) 


s_cmp_eq_i32 s54, 0 // (op 41611) 
s_mov_b32 s50, scc // (op 41611) 

s_cmp_eq_i32 s50, 0 // (op 41611) 
s_cselect_b32 s53, s49, s1 // (op 41611) 


s_mul_i32 s1, s52, s18 // (op 41611) 


v_lshl_add_u32 v11, s53, 8, v14 // (op 41611) 


s_sub_i32 s49, s0, s1 // (op 41611) 


s_cmp_eq_i32 s50, 0 // (op 41611) 
s_cselect_b32 s0, s51, s49 // (op 41611) 



v_lshl_add_u32 v13, s0, 8, v9 // (op 41611) 


v_mul_lo_u32 v9, v13, s4 // (op 41611) 

v_add_lshl_u32 v13, v11, v9, 1 // (op 41611) 


s_mul_i32 s1, s2, s7 // (op 41474) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 41474) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 41474) least significant
s_mul_hi_u32 s19, s2, s6 // (op 41474) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 41474) most significant: sum
s_add_u32 s1, s1, s19 // (op 41474) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 41474) 

s_mul_i32 s1, s4, s7 // (op 41613) most significant: low of low * high
s_mul_i32 s49, s5, s6 // (op 41613) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 41613) least significant
s_mul_hi_u32 s50, s4, s6 // (op 41613) most significant: high of low * low
s_add_u32 s1, s1, s49 // (op 41613) most significant: sum
s_add_u32 s1, s1, s50 // (op 41613) most significant: sum


s_lshl_b64 s[50:51], s[0:1], 1 // (op 41613) 

s_mov_b32 s52, s22 // (op 41474) reg expression
s_mov_b32 s53, s23 // (op 41474) reg expression
s_mov_b32 s54, s18 // (op 41474) reg expression
s_mov_b32 s55, 131072 // (op 41474) call()

s_mov_b32 s56, s24 // (op 41613) reg expression

s_mov_b32 s57, s25 // (op 41613) reg expression
s_mov_b32 s58, s50 // (op 41613) reg expression
s_mov_b32 s59, 131072 // (op 41613) call()


v_mov_b32 v9, s47 // (op 2) Move value
v_mov_b32 v11, s48 // (op 35) Move value


v_accvgpr_write a0, 0.00000 // (op 57) call()
v_accvgpr_write a1, 0.00000 // (op 57) call()
v_accvgpr_write a2, 0.00000 // (op 57) call()
v_accvgpr_write a3, 0.00000 // (op 57) call()
v_accvgpr_write a4, 0.00000 // (op 387) call()
v_accvgpr_write a5, 0.00000 // (op 387) call()
v_accvgpr_write a6, 0.00000 // (op 387) call()
v_accvgpr_write a7, 0.00000 // (op 387) call()
v_accvgpr_write a8, 0.00000 // (op 613) call()
v_accvgpr_write a9, 0.00000 // (op 613) call()
v_accvgpr_write a10, 0.00000 // (op 613) call()
v_accvgpr_write a11, 0.00000 // (op 613) call()
v_accvgpr_write a12, 0.00000 // (op 839) call()
v_accvgpr_write a13, 0.00000 // (op 839) call()
v_accvgpr_write a14, 0.00000 // (op 839) call()
v_accvgpr_write a15, 0.00000 // (op 839) call()
v_accvgpr_write a16, 0.00000 // (op 1065) call()
v_accvgpr_write a17, 0.00000 // (op 1065) call()
v_accvgpr_write a18, 0.00000 // (op 1065) call()
v_accvgpr_write a19, 0.00000 // (op 1065) call()
v_accvgpr_write a20, 0.00000 // (op 1291) call()
v_accvgpr_write a21, 0.00000 // (op 1291) call()
v_accvgpr_write a22, 0.00000 // (op 1291) call()
v_accvgpr_write a23, 0.00000 // (op 1291) call()
v_accvgpr_write a24, 0.00000 // (op 1517) call()
v_accvgpr_write a25, 0.00000 // (op 1517) call()
v_accvgpr_write a26, 0.00000 // (op 1517) call()
v_accvgpr_write a27, 0.00000 // (op 1517) call()
v_accvgpr_write a28, 0.00000 // (op 1743) call()
v_accvgpr_write a29, 0.00000 // (op 1743) call()
v_accvgpr_write a30, 0.00000 // (op 1743) call()
v_accvgpr_write a31, 0.00000 // (op 1743) call()
v_accvgpr_write a32, 0.00000 // (op 2241) call()
v_accvgpr_write a33, 0.00000 // (op 2241) call()
v_accvgpr_write a34, 0.00000 // (op 2241) call()
v_accvgpr_write a35, 0.00000 // (op 2241) call()
v_accvgpr_write a36, 0.00000 // (op 3066) call()
v_accvgpr_write a37, 0.00000 // (op 3066) call()
v_accvgpr_write a38, 0.00000 // (op 3066) call()
v_accvgpr_write a39, 0.00000 // (op 3066) call()
v_accvgpr_write a40, 0.00000 // (op 3067) call()
v_accvgpr_write a41, 0.00000 // (op 3067) call()
v_accvgpr_write a42, 0.00000 // (op 3067) call()
v_accvgpr_write a43, 0.00000 // (op 3067) call()
v_accvgpr_write a44, 0.00000 // (op 3068) call()
v_accvgpr_write a45, 0.00000 // (op 3068) call()
v_accvgpr_write a46, 0.00000 // (op 3068) call()
v_accvgpr_write a47, 0.00000 // (op 3068) call()
v_accvgpr_write a48, 0.00000 // (op 3069) call()
v_accvgpr_write a49, 0.00000 // (op 3069) call()
v_accvgpr_write a50, 0.00000 // (op 3069) call()
v_accvgpr_write a51, 0.00000 // (op 3069) call()
v_accvgpr_write a52, 0.00000 // (op 3070) call()
v_accvgpr_write a53, 0.00000 // (op 3070) call()
v_accvgpr_write a54, 0.00000 // (op 3070) call()
v_accvgpr_write a55, 0.00000 // (op 3070) call()
v_accvgpr_write a56, 0.00000 // (op 3071) call()
v_accvgpr_write a57, 0.00000 // (op 3071) call()
v_accvgpr_write a58, 0.00000 // (op 3071) call()
v_accvgpr_write a59, 0.00000 // (op 3071) call()
v_accvgpr_write a60, 0.00000 // (op 3072) call()
v_accvgpr_write a61, 0.00000 // (op 3072) call()
v_accvgpr_write a62, 0.00000 // (op 3072) call()
v_accvgpr_write a63, 0.00000 // (op 3072) call()
v_accvgpr_write a64, 0.00000 // (op 4243) call()
v_accvgpr_write a65, 0.00000 // (op 4243) call()
v_accvgpr_write a66, 0.00000 // (op 4243) call()
v_accvgpr_write a67, 0.00000 // (op 4243) call()
v_accvgpr_write a68, 0.00000 // (op 5068) call()
v_accvgpr_write a69, 0.00000 // (op 5068) call()
v_accvgpr_write a70, 0.00000 // (op 5068) call()
v_accvgpr_write a71, 0.00000 // (op 5068) call()
v_accvgpr_write a72, 0.00000 // (op 5069) call()
v_accvgpr_write a73, 0.00000 // (op 5069) call()
v_accvgpr_write a74, 0.00000 // (op 5069) call()
v_accvgpr_write a75, 0.00000 // (op 5069) call()
v_accvgpr_write a76, 0.00000 // (op 5070) call()
v_accvgpr_write a77, 0.00000 // (op 5070) call()
v_accvgpr_write a78, 0.00000 // (op 5070) call()
v_accvgpr_write a79, 0.00000 // (op 5070) call()
v_accvgpr_write a80, 0.00000 // (op 5071) call()
v_accvgpr_write a81, 0.00000 // (op 5071) call()
v_accvgpr_write a82, 0.00000 // (op 5071) call()
v_accvgpr_write a83, 0.00000 // (op 5071) call()
v_accvgpr_write a84, 0.00000 // (op 5072) call()
v_accvgpr_write a85, 0.00000 // (op 5072) call()
v_accvgpr_write a86, 0.00000 // (op 5072) call()
v_accvgpr_write a87, 0.00000 // (op 5072) call()
v_accvgpr_write a88, 0.00000 // (op 5073) call()
v_accvgpr_write a89, 0.00000 // (op 5073) call()
v_accvgpr_write a90, 0.00000 // (op 5073) call()
v_accvgpr_write a91, 0.00000 // (op 5073) call()
v_accvgpr_write a92, 0.00000 // (op 5074) call()
v_accvgpr_write a93, 0.00000 // (op 5074) call()
v_accvgpr_write a94, 0.00000 // (op 5074) call()
v_accvgpr_write a95, 0.00000 // (op 5074) call()
v_accvgpr_write a96, 0.00000 // (op 6245) call()
v_accvgpr_write a97, 0.00000 // (op 6245) call()
v_accvgpr_write a98, 0.00000 // (op 6245) call()
v_accvgpr_write a99, 0.00000 // (op 6245) call()
v_accvgpr_write a100, 0.00000 // (op 7070) call()
v_accvgpr_write a101, 0.00000 // (op 7070) call()
v_accvgpr_write a102, 0.00000 // (op 7070) call()
v_accvgpr_write a103, 0.00000 // (op 7070) call()
v_accvgpr_write a104, 0.00000 // (op 7071) call()
v_accvgpr_write a105, 0.00000 // (op 7071) call()
v_accvgpr_write a106, 0.00000 // (op 7071) call()
v_accvgpr_write a107, 0.00000 // (op 7071) call()
v_accvgpr_write a108, 0.00000 // (op 7072) call()
v_accvgpr_write a109, 0.00000 // (op 7072) call()
v_accvgpr_write a110, 0.00000 // (op 7072) call()
v_accvgpr_write a111, 0.00000 // (op 7072) call()
v_accvgpr_write a112, 0.00000 // (op 7073) call()
v_accvgpr_write a113, 0.00000 // (op 7073) call()
v_accvgpr_write a114, 0.00000 // (op 7073) call()
v_accvgpr_write a115, 0.00000 // (op 7073) call()
v_accvgpr_write a116, 0.00000 // (op 7074) call()
v_accvgpr_write a117, 0.00000 // (op 7074) call()
v_accvgpr_write a118, 0.00000 // (op 7074) call()
v_accvgpr_write a119, 0.00000 // (op 7074) call()
v_accvgpr_write a120, 0.00000 // (op 7075) call()
v_accvgpr_write a121, 0.00000 // (op 7075) call()
v_accvgpr_write a122, 0.00000 // (op 7075) call()
v_accvgpr_write a123, 0.00000 // (op 7075) call()
v_accvgpr_write a124, 0.00000 // (op 7076) call()
v_accvgpr_write a125, 0.00000 // (op 7076) call()
v_accvgpr_write a126, 0.00000 // (op 7076) call()
v_accvgpr_write a127, 0.00000 // (op 7076) call()
v_accvgpr_write a128, 0.00000 // (op 8247) call()
v_accvgpr_write a129, 0.00000 // (op 8247) call()
v_accvgpr_write a130, 0.00000 // (op 8247) call()
v_accvgpr_write a131, 0.00000 // (op 8247) call()
v_accvgpr_write a132, 0.00000 // (op 9072) call()
v_accvgpr_write a133, 0.00000 // (op 9072) call()
v_accvgpr_write a134, 0.00000 // (op 9072) call()
v_accvgpr_write a135, 0.00000 // (op 9072) call()
v_accvgpr_write a136, 0.00000 // (op 9073) call()
v_accvgpr_write a137, 0.00000 // (op 9073) call()
v_accvgpr_write a138, 0.00000 // (op 9073) call()
v_accvgpr_write a139, 0.00000 // (op 9073) call()
v_accvgpr_write a140, 0.00000 // (op 9074) call()
v_accvgpr_write a141, 0.00000 // (op 9074) call()
v_accvgpr_write a142, 0.00000 // (op 9074) call()
v_accvgpr_write a143, 0.00000 // (op 9074) call()
v_accvgpr_write a144, 0.00000 // (op 9075) call()
v_accvgpr_write a145, 0.00000 // (op 9075) call()
v_accvgpr_write a146, 0.00000 // (op 9075) call()
v_accvgpr_write a147, 0.00000 // (op 9075) call()
v_accvgpr_write a148, 0.00000 // (op 9076) call()
v_accvgpr_write a149, 0.00000 // (op 9076) call()
v_accvgpr_write a150, 0.00000 // (op 9076) call()
v_accvgpr_write a151, 0.00000 // (op 9076) call()
v_accvgpr_write a152, 0.00000 // (op 9077) call()
v_accvgpr_write a153, 0.00000 // (op 9077) call()
v_accvgpr_write a154, 0.00000 // (op 9077) call()
v_accvgpr_write a155, 0.00000 // (op 9077) call()
v_accvgpr_write a156, 0.00000 // (op 9078) call()
v_accvgpr_write a157, 0.00000 // (op 9078) call()
v_accvgpr_write a158, 0.00000 // (op 9078) call()
v_accvgpr_write a159, 0.00000 // (op 9078) call()
v_accvgpr_write a160, 0.00000 // (op 10249) call()
v_accvgpr_write a161, 0.00000 // (op 10249) call()
v_accvgpr_write a162, 0.00000 // (op 10249) call()
v_accvgpr_write a163, 0.00000 // (op 10249) call()
v_accvgpr_write a164, 0.00000 // (op 11074) call()
v_accvgpr_write a165, 0.00000 // (op 11074) call()
v_accvgpr_write a166, 0.00000 // (op 11074) call()
v_accvgpr_write a167, 0.00000 // (op 11074) call()
v_accvgpr_write a168, 0.00000 // (op 11075) call()
v_accvgpr_write a169, 0.00000 // (op 11075) call()
v_accvgpr_write a170, 0.00000 // (op 11075) call()
v_accvgpr_write a171, 0.00000 // (op 11075) call()
v_accvgpr_write a172, 0.00000 // (op 11076) call()
v_accvgpr_write a173, 0.00000 // (op 11076) call()
v_accvgpr_write a174, 0.00000 // (op 11076) call()
v_accvgpr_write a175, 0.00000 // (op 11076) call()
v_accvgpr_write a176, 0.00000 // (op 11077) call()
v_accvgpr_write a177, 0.00000 // (op 11077) call()
v_accvgpr_write a178, 0.00000 // (op 11077) call()
v_accvgpr_write a179, 0.00000 // (op 11077) call()
v_accvgpr_write a180, 0.00000 // (op 11078) call()
v_accvgpr_write a181, 0.00000 // (op 11078) call()
v_accvgpr_write a182, 0.00000 // (op 11078) call()
v_accvgpr_write a183, 0.00000 // (op 11078) call()
v_accvgpr_write a184, 0.00000 // (op 11079) call()
v_accvgpr_write a185, 0.00000 // (op 11079) call()
v_accvgpr_write a186, 0.00000 // (op 11079) call()
v_accvgpr_write a187, 0.00000 // (op 11079) call()
v_accvgpr_write a188, 0.00000 // (op 11080) call()
v_accvgpr_write a189, 0.00000 // (op 11080) call()
v_accvgpr_write a190, 0.00000 // (op 11080) call()
v_accvgpr_write a191, 0.00000 // (op 11080) call()
v_accvgpr_write a192, 0.00000 // (op 12251) call()
v_accvgpr_write a193, 0.00000 // (op 12251) call()
v_accvgpr_write a194, 0.00000 // (op 12251) call()
v_accvgpr_write a195, 0.00000 // (op 12251) call()
v_accvgpr_write a196, 0.00000 // (op 13076) call()
v_accvgpr_write a197, 0.00000 // (op 13076) call()
v_accvgpr_write a198, 0.00000 // (op 13076) call()
v_accvgpr_write a199, 0.00000 // (op 13076) call()
v_accvgpr_write a200, 0.00000 // (op 13077) call()
v_accvgpr_write a201, 0.00000 // (op 13077) call()
v_accvgpr_write a202, 0.00000 // (op 13077) call()
v_accvgpr_write a203, 0.00000 // (op 13077) call()
v_accvgpr_write a204, 0.00000 // (op 13078) call()
v_accvgpr_write a205, 0.00000 // (op 13078) call()
v_accvgpr_write a206, 0.00000 // (op 13078) call()
v_accvgpr_write a207, 0.00000 // (op 13078) call()
v_accvgpr_write a208, 0.00000 // (op 13079) call()
v_accvgpr_write a209, 0.00000 // (op 13079) call()
v_accvgpr_write a210, 0.00000 // (op 13079) call()
v_accvgpr_write a211, 0.00000 // (op 13079) call()
v_accvgpr_write a212, 0.00000 // (op 13080) call()
v_accvgpr_write a213, 0.00000 // (op 13080) call()
v_accvgpr_write a214, 0.00000 // (op 13080) call()
v_accvgpr_write a215, 0.00000 // (op 13080) call()
v_accvgpr_write a216, 0.00000 // (op 13081) call()
v_accvgpr_write a217, 0.00000 // (op 13081) call()
v_accvgpr_write a218, 0.00000 // (op 13081) call()
v_accvgpr_write a219, 0.00000 // (op 13081) call()
v_accvgpr_write a220, 0.00000 // (op 13082) call()
v_accvgpr_write a221, 0.00000 // (op 13082) call()
v_accvgpr_write a222, 0.00000 // (op 13082) call()
v_accvgpr_write a223, 0.00000 // (op 13082) call()
v_accvgpr_write a224, 0.00000 // (op 14253) call()
v_accvgpr_write a225, 0.00000 // (op 14253) call()
v_accvgpr_write a226, 0.00000 // (op 14253) call()
v_accvgpr_write a227, 0.00000 // (op 14253) call()
v_accvgpr_write a228, 0.00000 // (op 15078) call()
v_accvgpr_write a229, 0.00000 // (op 15078) call()
v_accvgpr_write a230, 0.00000 // (op 15078) call()
v_accvgpr_write a231, 0.00000 // (op 15078) call()
v_accvgpr_write a232, 0.00000 // (op 15079) call()
v_accvgpr_write a233, 0.00000 // (op 15079) call()
v_accvgpr_write a234, 0.00000 // (op 15079) call()
v_accvgpr_write a235, 0.00000 // (op 15079) call()
v_accvgpr_write a236, 0.00000 // (op 15080) call()
v_accvgpr_write a237, 0.00000 // (op 15080) call()
v_accvgpr_write a238, 0.00000 // (op 15080) call()
v_accvgpr_write a239, 0.00000 // (op 15080) call()
v_accvgpr_write a240, 0.00000 // (op 15081) call()
v_accvgpr_write a241, 0.00000 // (op 15081) call()
v_accvgpr_write a242, 0.00000 // (op 15081) call()
v_accvgpr_write a243, 0.00000 // (op 15081) call()
v_accvgpr_write a244, 0.00000 // (op 15082) call()
v_accvgpr_write a245, 0.00000 // (op 15082) call()
v_accvgpr_write a246, 0.00000 // (op 15082) call()
v_accvgpr_write a247, 0.00000 // (op 15082) call()
v_accvgpr_write a248, 0.00000 // (op 15083) call()
v_accvgpr_write a249, 0.00000 // (op 15083) call()
v_accvgpr_write a250, 0.00000 // (op 15083) call()
v_accvgpr_write a251, 0.00000 // (op 15083) call()
v_accvgpr_write a252, 0.00000 // (op 15084) call()
v_accvgpr_write a253, 0.00000 // (op 15084) call()
v_accvgpr_write a254, 0.00000 // (op 15084) call()
v_accvgpr_write a255, 0.00000 // (op 15084) call()
s_mul_hi_i32 s0, s16, s10 // (op 41362) 
v_lshrrev_b32 v14, 2, v1 // (op 41362) 
s_add_i32 s1, s0, s16 // (op 41362) 

v_and_b32 v15, -4, v1 // (op 41362) 
s_ashr_i32 s0, s1, 31 // (op 41362) 
v_sub_u32 v16, v1, v15 // (op 41362) 

s_cmp_eq_i32 s10, 0 // (op 41362) 
s_mov_b32 s18, scc // (op 41362) 
v_lshlrev_b32 v15, 5, v16 // (op 41362) 

s_cmp_eq_i32 s18, 0 // (op 41362) 
s_cselect_b32 s19, 0, -1 // (op 41362) 

v_mov_b32 v17, 0 // (op 41362) convert
v_mov_b32 v16, v15 // (op 41362) convert

s_and_b32 s18, s13, 31 // (op 41362) 
s_lshl_b32 s22, 1, s18 // (op 41362) 
s_add_i32 s23, s22, s19 // (op 41362) 


s_and_b32 s19, s0, s23 // (op 41362) 


s_add_i32 s0, s1, s19 // (op 41362) 


s_ashr_i32 s1, s0, s18 // (op 41362) 


s_lshl_b32 s0, s13, 24 // (op 41362) 
s_ashr_i32 s18, s0, 31 // (op 41362) 

s_xor_b32 s0, s1, s18 // (op 41362) 

s_sub_i32 s1, s0, s18 // (op 41362) 


s_mul_hi_i32 s0, s1, s14 // (op 41362) 
s_add_i32 s18, s0, s1 // (op 41362) 

s_ashr_i32 s0, s18, 31 // (op 41362) 
s_cmp_eq_i32 s14, 0 // (op 41362) 
s_mov_b32 s19, scc // (op 41362) 
s_cmp_eq_i32 s19, 0 // (op 41362) 
s_cselect_b32 s22, 0, -1 // (op 41362) 

s_and_b32 s19, s15, 31 // (op 41362) 
s_lshl_b32 s23, 1, s19 // (op 41362) 
s_add_i32 s24, s23, s22 // (op 41362) 


s_and_b32 s25, s0, s24 // (op 41362) 


s_add_i32 s0, s18, s25 // (op 41362) 


s_ashr_i32 s18, s0, s19 // (op 41362) 


s_lshl_b32 s0, s15, 24 // (op 41362) 
s_ashr_i32 s19, s0, 31 // (op 41362) 

s_xor_b32 s0, s18, s19 // (op 41362) 

s_sub_i32 s18, s0, s19 // (op 41362) 


s_mul_i32 s0, s1, s11 // (op 41362) 

s_sub_u32 s1, s16, s0 // (op 41362) 

s_mul_i32 s0, s18, s11 // (op 41362) 
s_add_u32 s0, s0, s1 // (op 41362) 


s_mul_hi_i32 s1, s16, s9 // (op 41362) 
s_add_i32 s18, s1, s16 // (op 41362) 

s_ashr_i32 s1, s18, 31 // (op 41362) 
s_cmp_eq_i32 s9, 0 // (op 41362) 
s_mov_b32 s19, scc // (op 41362) 
s_cmp_eq_i32 s19, 0 // (op 41362) 
s_cselect_b32 s22, 0, -1 // (op 41362) 

s_and_b32 s19, s45, 31 // (op 41362) 
s_lshl_b32 s23, 1, s19 // (op 41362) 
s_add_i32 s24, s23, s22 // (op 41362) 


s_and_b32 s25, s1, s24 // (op 41362) 


s_add_i32 s1, s18, s25 // (op 41362) 


s_ashr_i32 s18, s1, s19 // (op 41362) 


s_lshl_b32 s1, s45, 24 // (op 41362) 
s_ashr_i32 s19, s1, 31 // (op 41362) 

s_xor_b32 s1, s18, s19 // (op 41362) 

s_sub_i32 s18, s1, s19 // (op 41362) 


s_add_u32 s22, s6, 256 // (op 41362) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 41362) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 41362) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 41362) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 41362) 

s_mul_i32 s1, s11, s22 // (op 41362) 

s_mul_i32 s19, s8, s1 // (op 41362) 

s_mul_i32 s1, s18, s19 // (op 41362) 

s_sub_u32 s19, s16, s1 // (op 41362) 

s_mov_b32 s1, s19 // (op 41362) 
s_mul_hi_i32 s22, s1, s12 // (op 41362) 
s_add_i32 s23, s22, s1 // (op 41362) 


s_ashr_i32 s1, s23, 31 // (op 41362) 
s_cmp_eq_i32 s12, 0 // (op 41362) 
s_mov_b32 s22, scc // (op 41362) 
s_cmp_eq_i32 s22, 0 // (op 41362) 
s_cselect_b32 s24, 0, -1 // (op 41362) 

s_and_b32 s22, s44, 31 // (op 41362) 
s_lshl_b32 s25, 1, s22 // (op 41362) 
s_add_i32 s47, s25, s24 // (op 41362) 


s_and_b32 s24, s1, s47 // (op 41362) 


s_add_i32 s1, s23, s24 // (op 41362) 


s_ashr_i32 s23, s1, s22 // (op 41362) 


s_lshl_b32 s1, s44, 24 // (op 41362) 
s_ashr_i32 s22, s1, 31 // (op 41362) 

s_xor_b32 s1, s23, s22 // (op 41362) 

s_sub_i32 s23, s1, s22 // (op 41362) 


s_mul_i32 s1, s46, s11 // (op 41362) 
s_add_u32 s24, s20, 256 // (op 41362) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 41362) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 41362) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 41362) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 8 // (op 41362) 

s_sub_i32 s22, s24, s1 // (op 41362) 

s_mul_i32 s24, s23, s22 // (op 41362) 


s_sub_u32 s25, s19, s24 // (op 41362) 


s_add_u32 s19, s25, s1 // (op 41362) 


s_cmp_eq_i32 s18, 0 // (op 41362) 
s_mov_b32 s1, scc // (op 41362) 

s_cmp_eq_i32 s1, 0 // (op 41362) 
s_cselect_b32 s18, s19, s0 // (op 41362) 



v_lshl_add_u32 v15, s18, 8, v14 // (op 41362) 


v_mul_lo_u32 v19, v15, s27 // (op 41362) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 41362) most significant: high of low * low
v_mul_lo_u32 v18, v15, s26 // (op 41362) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 41362) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 41362) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 41362) most significant half



v_lshlrev_b64 v[14:15], 2, v[18:19] // (op 41362) 

v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 41362) 

s_mul_i32 s1, s26, s21 // (op 41364) most significant: low of low * high
s_mul_i32 s18, s27, s20 // (op 41364) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 41364) least significant
s_mul_hi_u32 s19, s26, s20 // (op 41364) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 41364) most significant: sum
s_add_u32 s1, s1, s19 // (op 41364) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 41364) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 41364) 

s_mov_b32 s48, s28 // (op 41364) reg expression
s_mov_b32 s49, s29 // (op 41364) reg expression
s_mov_b32 s50, s0 // (op 41364) reg expression
s_mov_b32 s51, 131072 // (op 41364) call()

s_mul_hi_i32 s47, s16, s10 // (op 41381) 
v_lshrrev_b32 v14, 2, v1 // (op 41381) 
s_add_i32 s0, s47, s16 // (op 41381) 

v_and_b32 v15, -4, v1 // (op 41381) 
s_ashr_i32 s1, s0, 31 // (op 41381) 
v_sub_u32 v18, v1, v15 // (op 41381) 

s_cmp_eq_i32 s10, 0 // (op 41381) 
s_mov_b32 s47, scc // (op 41381) 
v_lshlrev_b32 v15, 5, v18 // (op 41381) 

s_cmp_eq_i32 s47, 0 // (op 41381) 
s_cselect_b32 s18, 0, -1 // (op 41381) 

v_mov_b32 v19, 0 // (op 41381) convert
v_mov_b32 v18, v15 // (op 41381) convert

s_and_b32 s19, s13, 31 // (op 41381) 
s_lshl_b32 s47, 1, s19 // (op 41381) 
s_add_i32 s22, s47, s18 // (op 41381) 


s_and_b32 s18, s1, s22 // (op 41381) 


s_add_i32 s1, s0, s18 // (op 41381) 


s_ashr_i32 s0, s1, s19 // (op 41381) 


s_lshl_b32 s1, s13, 24 // (op 41381) 
s_ashr_i32 s47, s1, 31 // (op 41381) 

s_xor_b32 s1, s0, s47 // (op 41381) 

s_sub_i32 s0, s1, s47 // (op 41381) 


s_mul_hi_i32 s1, s0, s14 // (op 41381) 
s_add_i32 s47, s1, s0 // (op 41381) 

s_ashr_i32 s1, s47, 31 // (op 41381) 
s_cmp_eq_i32 s14, 0 // (op 41381) 
s_mov_b32 s18, scc // (op 41381) 
s_cmp_eq_i32 s18, 0 // (op 41381) 
s_cselect_b32 s19, 0, -1 // (op 41381) 

s_and_b32 s18, s15, 31 // (op 41381) 
s_lshl_b32 s22, 1, s18 // (op 41381) 
s_add_i32 s23, s22, s19 // (op 41381) 


s_and_b32 s19, s1, s23 // (op 41381) 


s_add_i32 s1, s47, s19 // (op 41381) 


s_ashr_i32 s19, s1, s18 // (op 41381) 


s_lshl_b32 s1, s15, 24 // (op 41381) 
s_ashr_i32 s18, s1, 31 // (op 41381) 

s_xor_b32 s1, s19, s18 // (op 41381) 

s_sub_i32 s19, s1, s18 // (op 41381) 


s_mul_i32 s1, s0, s11 // (op 41381) 

s_sub_u32 s0, s16, s1 // (op 41381) 

s_mul_i32 s1, s19, s11 // (op 41381) 
s_add_u32 s1, s1, s0 // (op 41381) 


s_mul_hi_i32 s0, s16, s9 // (op 41381) 
s_add_i32 s47, s0, s16 // (op 41381) 

s_ashr_i32 s0, s47, 31 // (op 41381) 
s_cmp_eq_i32 s9, 0 // (op 41381) 
s_mov_b32 s18, scc // (op 41381) 
s_cmp_eq_i32 s18, 0 // (op 41381) 
s_cselect_b32 s19, 0, -1 // (op 41381) 

s_and_b32 s18, s45, 31 // (op 41381) 
s_lshl_b32 s22, 1, s18 // (op 41381) 
s_add_i32 s23, s22, s19 // (op 41381) 


s_and_b32 s19, s0, s23 // (op 41381) 


s_add_i32 s0, s47, s19 // (op 41381) 


s_ashr_i32 s19, s0, s18 // (op 41381) 


s_lshl_b32 s0, s45, 24 // (op 41381) 
s_ashr_i32 s18, s0, 31 // (op 41381) 

s_xor_b32 s0, s19, s18 // (op 41381) 

s_sub_i32 s19, s0, s18 // (op 41381) 


s_add_u32 s22, s6, 256 // (op 41381) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 41381) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 41381) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 41381) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 41381) 

s_mul_i32 s0, s11, s22 // (op 41381) 

s_mul_i32 s18, s8, s0 // (op 41381) 

s_mul_i32 s0, s19, s18 // (op 41381) 

s_sub_u32 s18, s16, s0 // (op 41381) 

s_mov_b32 s0, s18 // (op 41381) 
s_mul_hi_i32 s47, s0, s12 // (op 41381) 
s_add_i32 s22, s47, s0 // (op 41381) 


s_ashr_i32 s0, s22, 31 // (op 41381) 
s_cmp_eq_i32 s12, 0 // (op 41381) 
s_mov_b32 s47, scc // (op 41381) 
s_cmp_eq_i32 s47, 0 // (op 41381) 
s_cselect_b32 s23, 0, -1 // (op 41381) 

s_and_b32 s47, s44, 31 // (op 41381) 
s_lshl_b32 s24, 1, s47 // (op 41381) 
s_add_i32 s25, s24, s23 // (op 41381) 


s_and_b32 s23, s0, s25 // (op 41381) 


s_add_i32 s0, s22, s23 // (op 41381) 


s_ashr_i32 s22, s0, s47 // (op 41381) 


s_lshl_b32 s0, s44, 24 // (op 41381) 
s_ashr_i32 s47, s0, 31 // (op 41381) 

s_xor_b32 s0, s22, s47 // (op 41381) 

s_sub_i32 s22, s0, s47 // (op 41381) 


s_mul_i32 s0, s46, s11 // (op 41381) 
s_add_u32 s24, s20, 256 // (op 41381) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 41381) End of Int64 add, unlocking SCC
s_sub_u32 s60, s24, 1 // (op 41381) least significant half; sets scc
s_subb_u32 s61, s25, 0 // (op 41381) most significant half; uses scc

s_ashr_i64 s[24:25], s[60:61], 8 // (op 41381) 

s_sub_i32 s23, s24, s0 // (op 41381) 

s_mul_i32 s47, s22, s23 // (op 41381) 


s_sub_u32 s22, s18, s47 // (op 41381) 


s_add_u32 s18, s22, s0 // (op 41381) 


s_cmp_eq_i32 s19, 0 // (op 41381) 
s_mov_b32 s0, scc // (op 41381) 

s_cmp_eq_i32 s0, 0 // (op 41381) 
s_cselect_b32 s19, s18, s1 // (op 41381) 



v_lshl_add_u32 v15, s19, 8, v14 // (op 41381) 


v_mul_lo_u32 v21, v15, s27 // (op 41381) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 41381) most significant: high of low * low
v_mul_lo_u32 v20, v15, s26 // (op 41381) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 41381) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v18 // (op 41381) least significant half
v_addc_co_u32 v21, s[0:1], v21, v19, s[0:1] // (op 41381) most significant half



v_lshlrev_b64 v[14:15], 2, v[20:21] // (op 41381) 

v_lshrrev_b64 v[18:19], 3, v[14:15] // (op 41381) 

s_mul_i32 s1, s26, s21 // (op 41383) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 41383) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 41383) least significant
s_mul_hi_u32 s18, s26, s20 // (op 41383) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 41383) most significant: sum
s_add_u32 s1, s1, s18 // (op 41383) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 41383) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 41383) 

s_mov_b32 s48, s28 // (op 41383) reg expression
s_mov_b32 s49, s29 // (op 41383) reg expression
s_mov_b32 s50, s0 // (op 41383) reg expression
s_mov_b32 s51, 131072 // (op 41383) call()

s_mul_hi_i32 s47, s16, s10 // (op 41426) 
v_lshrrev_b32 v14, 2, v1 // (op 41426) 
s_add_i32 s0, s47, s16 // (op 41426) 

v_and_b32 v15, -4, v1 // (op 41426) 
s_ashr_i32 s1, s0, 31 // (op 41426) 
v_sub_u32 v20, v1, v15 // (op 41426) 

s_cmp_eq_i32 s10, 0 // (op 41426) 
s_mov_b32 s47, scc // (op 41426) 
v_lshlrev_b32 v15, 5, v20 // (op 41426) 

s_cmp_eq_i32 s47, 0 // (op 41426) 
s_cselect_b32 s18, 0, -1 // (op 41426) 

v_mov_b32 v21, 0 // (op 41426) convert
v_mov_b32 v20, v15 // (op 41426) convert

s_and_b32 s19, s13, 31 // (op 41426) 
s_lshl_b32 s47, 1, s19 // (op 41426) 
s_add_i32 s22, s47, s18 // (op 41426) 


s_and_b32 s18, s1, s22 // (op 41426) 


s_add_i32 s1, s0, s18 // (op 41426) 


s_ashr_i32 s0, s1, s19 // (op 41426) 


s_lshl_b32 s1, s13, 24 // (op 41426) 
s_ashr_i32 s47, s1, 31 // (op 41426) 

s_xor_b32 s1, s0, s47 // (op 41426) 

s_sub_i32 s0, s1, s47 // (op 41426) 


s_mul_hi_i32 s1, s0, s14 // (op 41426) 
s_add_i32 s47, s1, s0 // (op 41426) 

s_ashr_i32 s1, s47, 31 // (op 41426) 
s_cmp_eq_i32 s14, 0 // (op 41426) 
s_mov_b32 s18, scc // (op 41426) 
s_cmp_eq_i32 s18, 0 // (op 41426) 
s_cselect_b32 s19, 0, -1 // (op 41426) 

s_and_b32 s18, s15, 31 // (op 41426) 
s_lshl_b32 s22, 1, s18 // (op 41426) 
s_add_i32 s23, s22, s19 // (op 41426) 


s_and_b32 s19, s1, s23 // (op 41426) 


s_add_i32 s1, s47, s19 // (op 41426) 


s_ashr_i32 s19, s1, s18 // (op 41426) 


s_lshl_b32 s1, s15, 24 // (op 41426) 
s_ashr_i32 s18, s1, 31 // (op 41426) 

s_xor_b32 s1, s19, s18 // (op 41426) 

s_sub_i32 s19, s1, s18 // (op 41426) 


s_add_u32 s22, s6, 256 // (op 41426) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 41426) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 41426) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 41426) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 41426) 

s_mul_i32 s1, s19, s22 // (op 41426) 

s_sub_i32 s47, s0, s1 // (op 41426) 


s_mul_hi_i32 s0, s16, s9 // (op 41426) 
s_add_i32 s1, s0, s16 // (op 41426) 

s_ashr_i32 s0, s1, 31 // (op 41426) 
s_cmp_eq_i32 s9, 0 // (op 41426) 
s_mov_b32 s18, scc // (op 41426) 
s_cmp_eq_i32 s18, 0 // (op 41426) 
s_cselect_b32 s19, 0, -1 // (op 41426) 

s_and_b32 s18, s45, 31 // (op 41426) 
s_lshl_b32 s24, 1, s18 // (op 41426) 
s_add_i32 s25, s24, s19 // (op 41426) 


s_and_b32 s19, s0, s25 // (op 41426) 


s_add_i32 s0, s1, s19 // (op 41426) 


s_ashr_i32 s1, s0, s18 // (op 41426) 


s_lshl_b32 s0, s45, 24 // (op 41426) 
s_ashr_i32 s18, s0, 31 // (op 41426) 

s_xor_b32 s0, s1, s18 // (op 41426) 

s_sub_i32 s1, s0, s18 // (op 41426) 


s_mul_i32 s0, s11, s22 // (op 41426) 

s_mul_i32 s18, s8, s0 // (op 41426) 

s_mul_i32 s0, s1, s18 // (op 41426) 

s_sub_u32 s18, s16, s0 // (op 41426) 

s_mul_hi_i32 s0, s18, s12 // (op 41426) 
s_add_i32 s19, s0, s18 // (op 41426) 


s_ashr_i32 s0, s19, 31 // (op 41426) 
s_cmp_eq_i32 s12, 0 // (op 41426) 
s_mov_b32 s18, scc // (op 41426) 
s_cmp_eq_i32 s18, 0 // (op 41426) 
s_cselect_b32 s22, 0, -1 // (op 41426) 

s_and_b32 s18, s44, 31 // (op 41426) 
s_lshl_b32 s23, 1, s18 // (op 41426) 
s_add_i32 s24, s23, s22 // (op 41426) 


s_and_b32 s25, s0, s24 // (op 41426) 


s_add_i32 s0, s19, s25 // (op 41426) 


s_ashr_i32 s19, s0, s18 // (op 41426) 


s_lshl_b32 s0, s44, 24 // (op 41426) 
s_ashr_i32 s18, s0, 31 // (op 41426) 

s_xor_b32 s0, s19, s18 // (op 41426) 

s_sub_i32 s19, s0, s18 // (op 41426) 


s_cmp_eq_i32 s1, 0 // (op 41426) 
s_mov_b32 s0, scc // (op 41426) 

s_cmp_eq_i32 s0, 0 // (op 41426) 
s_cselect_b32 s1, s19, s47 // (op 41426) 



v_lshl_add_u32 v15, s1, 8, v14 // (op 41426) 


v_mul_lo_u32 v23, v15, s31 // (op 41426) most significant: low of low * high
v_mul_hi_u32 v14, v15, s30 // (op 41426) most significant: high of low * low
v_mul_lo_u32 v22, v15, s30 // (op 41426) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 41426) most significant: sum


v_add_co_u32 v14, s[0:1], v20, v22 // (op 41426) least significant half
v_addc_co_u32 v15, s[0:1], v21, v23, s[0:1] // (op 41426) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 41426) 


v_lshrrev_b64 v[20:21], 3, v[14:15] // (op 41426) 

s_mul_i32 s1, s30, s7 // (op 41428) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 41428) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 41428) least significant
s_mul_hi_u32 s18, s30, s6 // (op 41428) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 41428) most significant: sum
s_add_u32 s1, s1, s18 // (op 41428) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 41428) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 41428) 

s_mov_b32 s60, s32 // (op 41428) reg expression
s_mov_b32 s61, s33 // (op 41428) reg expression
s_mov_b32 s62, s0 // (op 41428) reg expression
s_mov_b32 s63, 131072 // (op 41428) call()

s_mul_hi_i32 s47, s16, s10 // (op 41445) 
v_lshrrev_b32 v14, 2, v1 // (op 41445) 
s_add_i32 s0, s47, s16 // (op 41445) 

v_and_b32 v15, -4, v1 // (op 41445) 
s_ashr_i32 s1, s0, 31 // (op 41445) 
v_sub_u32 v22, v1, v15 // (op 41445) 

s_cmp_eq_i32 s10, 0 // (op 41445) 
s_mov_b32 s47, scc // (op 41445) 
v_lshlrev_b32 v15, 5, v22 // (op 41445) 

s_cmp_eq_i32 s47, 0 // (op 41445) 
s_cselect_b32 s18, 0, -1 // (op 41445) 

v_mov_b32 v23, 0 // (op 41445) convert
v_mov_b32 v22, v15 // (op 41445) convert

s_and_b32 s19, s13, 31 // (op 41445) 
s_lshl_b32 s47, 1, s19 // (op 41445) 
s_add_i32 s22, s47, s18 // (op 41445) 


s_and_b32 s18, s1, s22 // (op 41445) 


s_add_i32 s1, s0, s18 // (op 41445) 


s_ashr_i32 s0, s1, s19 // (op 41445) 


s_lshl_b32 s1, s13, 24 // (op 41445) 
s_ashr_i32 s47, s1, 31 // (op 41445) 

s_xor_b32 s1, s0, s47 // (op 41445) 

s_sub_i32 s0, s1, s47 // (op 41445) 


s_mul_hi_i32 s1, s0, s14 // (op 41445) 
s_add_i32 s47, s1, s0 // (op 41445) 

s_ashr_i32 s1, s47, 31 // (op 41445) 
s_cmp_eq_i32 s14, 0 // (op 41445) 
s_mov_b32 s18, scc // (op 41445) 
s_cmp_eq_i32 s18, 0 // (op 41445) 
s_cselect_b32 s19, 0, -1 // (op 41445) 

s_and_b32 s18, s15, 31 // (op 41445) 
s_lshl_b32 s22, 1, s18 // (op 41445) 
s_add_i32 s23, s22, s19 // (op 41445) 


s_and_b32 s19, s1, s23 // (op 41445) 


s_add_i32 s1, s47, s19 // (op 41445) 


s_ashr_i32 s19, s1, s18 // (op 41445) 


s_lshl_b32 s1, s15, 24 // (op 41445) 
s_ashr_i32 s18, s1, 31 // (op 41445) 

s_xor_b32 s1, s19, s18 // (op 41445) 

s_sub_i32 s19, s1, s18 // (op 41445) 


s_add_u32 s22, s6, 256 // (op 41445) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 41445) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 41445) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 41445) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 41445) 

s_mul_i32 s1, s19, s22 // (op 41445) 

s_sub_i32 s47, s0, s1 // (op 41445) 


s_mul_hi_i32 s0, s16, s9 // (op 41445) 
s_add_i32 s1, s0, s16 // (op 41445) 

s_ashr_i32 s0, s1, 31 // (op 41445) 
s_cmp_eq_i32 s9, 0 // (op 41445) 
s_mov_b32 s18, scc // (op 41445) 
s_cmp_eq_i32 s18, 0 // (op 41445) 
s_cselect_b32 s19, 0, -1 // (op 41445) 

s_and_b32 s18, s45, 31 // (op 41445) 
s_lshl_b32 s24, 1, s18 // (op 41445) 
s_add_i32 s25, s24, s19 // (op 41445) 


s_and_b32 s19, s0, s25 // (op 41445) 


s_add_i32 s0, s1, s19 // (op 41445) 


s_ashr_i32 s1, s0, s18 // (op 41445) 


s_lshl_b32 s0, s45, 24 // (op 41445) 
s_ashr_i32 s18, s0, 31 // (op 41445) 

s_xor_b32 s0, s1, s18 // (op 41445) 

s_sub_i32 s1, s0, s18 // (op 41445) 


s_mul_i32 s0, s11, s22 // (op 41445) 

s_mul_i32 s18, s8, s0 // (op 41445) 

s_mul_i32 s0, s1, s18 // (op 41445) 

s_sub_u32 s18, s16, s0 // (op 41445) 

s_mul_hi_i32 s0, s18, s12 // (op 41445) 
s_add_i32 s19, s0, s18 // (op 41445) 


s_ashr_i32 s0, s19, 31 // (op 41445) 
s_cmp_eq_i32 s12, 0 // (op 41445) 
s_mov_b32 s18, scc // (op 41445) 
s_cmp_eq_i32 s18, 0 // (op 41445) 
s_cselect_b32 s22, 0, -1 // (op 41445) 

s_and_b32 s18, s44, 31 // (op 41445) 
s_lshl_b32 s23, 1, s18 // (op 41445) 
s_add_i32 s24, s23, s22 // (op 41445) 


s_and_b32 s25, s0, s24 // (op 41445) 


s_add_i32 s0, s19, s25 // (op 41445) 


s_ashr_i32 s19, s0, s18 // (op 41445) 


s_lshl_b32 s0, s44, 24 // (op 41445) 
s_ashr_i32 s18, s0, 31 // (op 41445) 

s_xor_b32 s0, s19, s18 // (op 41445) 

s_sub_i32 s19, s0, s18 // (op 41445) 


s_cmp_eq_i32 s1, 0 // (op 41445) 
s_mov_b32 s0, scc // (op 41445) 

s_cmp_eq_i32 s0, 0 // (op 41445) 
s_cselect_b32 s1, s19, s47 // (op 41445) 



v_lshl_add_u32 v15, s1, 8, v14 // (op 41445) 


v_mul_lo_u32 v25, v15, s31 // (op 41445) most significant: low of low * high
v_mul_hi_u32 v14, v15, s30 // (op 41445) most significant: high of low * low
v_mul_lo_u32 v24, v15, s30 // (op 41445) least significant: low of low * low
v_add_u32 v25, v25, v14 // (op 41445) most significant: sum


v_add_co_u32 v14, s[0:1], v22, v24 // (op 41445) least significant half
v_addc_co_u32 v15, s[0:1], v23, v25, s[0:1] // (op 41445) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 41445) 


v_lshrrev_b64 v[22:23], 3, v[14:15] // (op 41445) 

s_mul_i32 s1, s30, s7 // (op 41447) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 41447) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 41447) least significant
s_mul_hi_u32 s18, s30, s6 // (op 41447) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 41447) most significant: sum
s_add_u32 s1, s1, s18 // (op 41447) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 41447) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 41447) 

s_mov_b32 s60, s32 // (op 41447) reg expression
s_mov_b32 s61, s33 // (op 41447) reg expression
s_mov_b32 s62, s0 // (op 41447) reg expression
s_mov_b32 s63, 131072 // (op 41447) call()

s_mul_hi_i32 s47, s16, s10 // (op 41899) 
v_and_b32 v14, 63, v1 // (op 41899) 
s_add_i32 s0, s47, s16 // (op 41899) 

v_lshrrev_b32 v15, 4, v14 // (op 41899) 
s_ashr_i32 s1, s0, 31 // (op 41899) 
v_and_b32 v24, 3, v15 // (op 41899) 

s_cmp_eq_i32 s10, 0 // (op 41899) 
s_mov_b32 s47, scc // (op 41899) 
v_and_b32 v15, 15, v1 // (op 41899) 
s_cmp_eq_i32 s47, 0 // (op 41899) 
s_cselect_b32 s18, 0, -1 // (op 41899) 

v_lshl_add_u32 v25, v24, 4, v15 // (op 41899) 


s_and_b32 s19, s13, 31 // (op 41899) 
v_and_b32 v15, 4294967168, v1 // (op 41899) 
s_lshl_b32 s47, 1, s19 // (op 41899) 
v_add_u32 v24, v15, v25 // (op 41899) 


s_add_i32 s22, s47, s18 // (op 41899) 


v_lshrrev_b32 v15, 6, v14 // (op 41899) 

s_and_b32 s18, s1, s22 // (op 41899) 


v_lshlrev_b32 v14, 2, v15 // (op 41899) 

s_add_i32 s1, s0, s18 // (op 41899) 


s_ashr_i32 s0, s1, s19 // (op 41899) 


s_lshl_b32 s1, s13, 24 // (op 41899) 
s_ashr_i32 s47, s1, 31 // (op 41899) 

s_xor_b32 s1, s0, s47 // (op 41899) 

s_sub_i32 s0, s1, s47 // (op 41899) 


s_mul_hi_i32 s1, s0, s14 // (op 41899) 
s_add_i32 s47, s1, s0 // (op 41899) 

s_ashr_i32 s1, s47, 31 // (op 41899) 
s_cmp_eq_i32 s14, 0 // (op 41899) 
s_mov_b32 s18, scc // (op 41899) 
s_cmp_eq_i32 s18, 0 // (op 41899) 
s_cselect_b32 s19, 0, -1 // (op 41899) 

s_and_b32 s18, s15, 31 // (op 41899) 
s_lshl_b32 s22, 1, s18 // (op 41899) 
s_add_i32 s23, s22, s19 // (op 41899) 


s_and_b32 s19, s1, s23 // (op 41899) 


s_add_i32 s1, s47, s19 // (op 41899) 


s_ashr_i32 s19, s1, s18 // (op 41899) 


s_lshl_b32 s1, s15, 24 // (op 41899) 
s_ashr_i32 s18, s1, 31 // (op 41899) 

s_xor_b32 s1, s19, s18 // (op 41899) 

s_sub_i32 s19, s1, s18 // (op 41899) 


s_mul_i32 s1, s0, s11 // (op 41899) 

s_sub_u32 s0, s16, s1 // (op 41899) 

s_mul_i32 s1, s19, s11 // (op 41899) 
s_add_u32 s1, s1, s0 // (op 41899) 


s_mul_hi_i32 s0, s16, s9 // (op 41899) 
s_add_i32 s47, s0, s16 // (op 41899) 

s_ashr_i32 s0, s47, 31 // (op 41899) 
s_cmp_eq_i32 s9, 0 // (op 41899) 
s_mov_b32 s18, scc // (op 41899) 
s_cmp_eq_i32 s18, 0 // (op 41899) 
s_cselect_b32 s19, 0, -1 // (op 41899) 

s_and_b32 s18, s45, 31 // (op 41899) 
s_lshl_b32 s22, 1, s18 // (op 41899) 
s_add_i32 s23, s22, s19 // (op 41899) 


s_and_b32 s19, s0, s23 // (op 41899) 


s_add_i32 s0, s47, s19 // (op 41899) 


s_ashr_i32 s19, s0, s18 // (op 41899) 


s_lshl_b32 s0, s45, 24 // (op 41899) 
s_ashr_i32 s18, s0, 31 // (op 41899) 

s_xor_b32 s0, s19, s18 // (op 41899) 

s_sub_i32 s19, s0, s18 // (op 41899) 


s_add_u32 s22, s6, 256 // (op 41899) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 41899) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 41899) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 41899) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 41899) 

s_mul_i32 s0, s11, s22 // (op 41899) 

s_mul_i32 s18, s8, s0 // (op 41899) 

s_mul_i32 s0, s19, s18 // (op 41899) 

s_sub_u32 s18, s16, s0 // (op 41899) 

s_mov_b32 s0, s18 // (op 41899) 
s_mul_hi_i32 s47, s0, s12 // (op 41899) 
s_add_i32 s22, s47, s0 // (op 41899) 


s_ashr_i32 s0, s22, 31 // (op 41899) 
s_cmp_eq_i32 s12, 0 // (op 41899) 
s_mov_b32 s47, scc // (op 41899) 
s_cmp_eq_i32 s47, 0 // (op 41899) 
s_cselect_b32 s23, 0, -1 // (op 41899) 

s_and_b32 s47, s44, 31 // (op 41899) 
s_lshl_b32 s24, 1, s47 // (op 41899) 
s_add_i32 s25, s24, s23 // (op 41899) 


s_and_b32 s23, s0, s25 // (op 41899) 


s_add_i32 s0, s22, s23 // (op 41899) 


s_ashr_i32 s22, s0, s47 // (op 41899) 


s_lshl_b32 s0, s44, 24 // (op 41899) 
s_ashr_i32 s47, s0, 31 // (op 41899) 

s_xor_b32 s0, s22, s47 // (op 41899) 

s_sub_i32 s22, s0, s47 // (op 41899) 


s_mul_i32 s0, s46, s11 // (op 41899) 
s_add_u32 s24, s20, 256 // (op 41899) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 41899) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 41899) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 41899) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 8 // (op 41899) 

s_sub_i32 s23, s24, s0 // (op 41899) 

s_mul_i32 s47, s22, s23 // (op 41899) 


s_sub_u32 s22, s18, s47 // (op 41899) 


s_add_u32 s18, s22, s0 // (op 41899) 


s_cmp_eq_i32 s19, 0 // (op 41899) 
s_mov_b32 s0, scc // (op 41899) 

s_cmp_eq_i32 s0, 0 // (op 41899) 
s_cselect_b32 s19, s18, s1 // (op 41899) 



v_lshl_add_u32 v15, s19, 8, v24 // (op 41899) 


v_mul_lo_u32 v24, v15, s34 // (op 41899) 
v_add_u32 v24, v24, v14 // (op 41899) 


s_mul_i32 s1, s34, s21 // (op 41901) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 41901) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 41901) least significant
s_mul_hi_u32 s18, s34, s20 // (op 41901) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 41901) most significant: sum
s_add_u32 s1, s1, s18 // (op 41901) most significant: sum


s_mov_b32 s64, s36 // (op 41901) reg expression
s_mov_b32 s65, s37 // (op 41901) reg expression
s_mov_b32 s66, s0 // (op 41901) reg expression
s_mov_b32 s67, 131072 // (op 41901) call()

s_mul_hi_i32 s47, s16, s10 // (op 41920) 
v_and_b32 v14, 63, v1 // (op 41920) 
s_add_i32 s0, s47, s16 // (op 41920) 

v_lshrrev_b32 v15, 4, v14 // (op 41920) 
s_ashr_i32 s1, s0, 31 // (op 41920) 
v_and_b32 v25, 3, v15 // (op 41920) 

s_cmp_eq_i32 s10, 0 // (op 41920) 
s_mov_b32 s47, scc // (op 41920) 
v_and_b32 v15, 15, v1 // (op 41920) 
s_cmp_eq_i32 s47, 0 // (op 41920) 
s_cselect_b32 s18, 0, -1 // (op 41920) 

v_lshl_add_u32 v26, v25, 4, v15 // (op 41920) 


s_and_b32 s19, s13, 31 // (op 41920) 
v_and_b32 v15, 4294967168, v1 // (op 41920) 
s_lshl_b32 s47, 1, s19 // (op 41920) 
v_add_u32 v25, v15, v26 // (op 41920) 


s_add_i32 s22, s47, s18 // (op 41920) 


v_lshrrev_b32 v15, 6, v14 // (op 41920) 

s_and_b32 s18, s1, s22 // (op 41920) 


v_lshlrev_b32 v14, 2, v15 // (op 41920) 

s_add_i32 s1, s0, s18 // (op 41920) 


s_ashr_i32 s0, s1, s19 // (op 41920) 


s_lshl_b32 s1, s13, 24 // (op 41920) 
s_ashr_i32 s47, s1, 31 // (op 41920) 

s_xor_b32 s1, s0, s47 // (op 41920) 

s_sub_i32 s0, s1, s47 // (op 41920) 


s_mul_hi_i32 s1, s0, s14 // (op 41920) 
s_add_i32 s47, s1, s0 // (op 41920) 

s_ashr_i32 s1, s47, 31 // (op 41920) 
s_cmp_eq_i32 s14, 0 // (op 41920) 
s_mov_b32 s18, scc // (op 41920) 
s_cmp_eq_i32 s18, 0 // (op 41920) 
s_cselect_b32 s19, 0, -1 // (op 41920) 

s_and_b32 s18, s15, 31 // (op 41920) 
s_lshl_b32 s22, 1, s18 // (op 41920) 
s_add_i32 s23, s22, s19 // (op 41920) 


s_and_b32 s19, s1, s23 // (op 41920) 


s_add_i32 s1, s47, s19 // (op 41920) 


s_ashr_i32 s19, s1, s18 // (op 41920) 


s_lshl_b32 s1, s15, 24 // (op 41920) 
s_ashr_i32 s18, s1, 31 // (op 41920) 

s_xor_b32 s1, s19, s18 // (op 41920) 

s_sub_i32 s19, s1, s18 // (op 41920) 


s_mul_i32 s1, s0, s11 // (op 41920) 

s_sub_u32 s0, s16, s1 // (op 41920) 

s_mul_i32 s1, s19, s11 // (op 41920) 
s_add_u32 s1, s1, s0 // (op 41920) 


s_mul_hi_i32 s0, s16, s9 // (op 41920) 
s_add_i32 s47, s0, s16 // (op 41920) 

s_ashr_i32 s0, s47, 31 // (op 41920) 
s_cmp_eq_i32 s9, 0 // (op 41920) 
s_mov_b32 s18, scc // (op 41920) 
s_cmp_eq_i32 s18, 0 // (op 41920) 
s_cselect_b32 s19, 0, -1 // (op 41920) 

s_and_b32 s18, s45, 31 // (op 41920) 
s_lshl_b32 s22, 1, s18 // (op 41920) 
s_add_i32 s23, s22, s19 // (op 41920) 


s_and_b32 s19, s0, s23 // (op 41920) 


s_add_i32 s0, s47, s19 // (op 41920) 


s_ashr_i32 s19, s0, s18 // (op 41920) 


s_lshl_b32 s0, s45, 24 // (op 41920) 
s_ashr_i32 s18, s0, 31 // (op 41920) 

s_xor_b32 s0, s19, s18 // (op 41920) 

s_sub_i32 s19, s0, s18 // (op 41920) 


s_add_u32 s22, s6, 256 // (op 41920) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 41920) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 41920) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 41920) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 41920) 

s_mul_i32 s0, s11, s22 // (op 41920) 

s_mul_i32 s18, s8, s0 // (op 41920) 

s_mul_i32 s0, s19, s18 // (op 41920) 

s_sub_u32 s18, s16, s0 // (op 41920) 

s_mov_b32 s0, s18 // (op 41920) 
s_mul_hi_i32 s47, s0, s12 // (op 41920) 
s_add_i32 s22, s47, s0 // (op 41920) 


s_ashr_i32 s0, s22, 31 // (op 41920) 
s_cmp_eq_i32 s12, 0 // (op 41920) 
s_mov_b32 s47, scc // (op 41920) 
s_cmp_eq_i32 s47, 0 // (op 41920) 
s_cselect_b32 s23, 0, -1 // (op 41920) 

s_and_b32 s47, s44, 31 // (op 41920) 
s_lshl_b32 s24, 1, s47 // (op 41920) 
s_add_i32 s25, s24, s23 // (op 41920) 


s_and_b32 s23, s0, s25 // (op 41920) 


s_add_i32 s0, s22, s23 // (op 41920) 


s_ashr_i32 s22, s0, s47 // (op 41920) 


s_lshl_b32 s0, s44, 24 // (op 41920) 
s_ashr_i32 s47, s0, 31 // (op 41920) 

s_xor_b32 s0, s22, s47 // (op 41920) 

s_sub_i32 s22, s0, s47 // (op 41920) 


s_mul_i32 s0, s46, s11 // (op 41920) 
s_add_u32 s24, s20, 256 // (op 41920) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 41920) End of Int64 add, unlocking SCC
s_sub_u32 s68, s24, 1 // (op 41920) least significant half; sets scc
s_subb_u32 s69, s25, 0 // (op 41920) most significant half; uses scc

s_ashr_i64 s[24:25], s[68:69], 8 // (op 41920) 

s_sub_i32 s23, s24, s0 // (op 41920) 

s_mul_i32 s47, s22, s23 // (op 41920) 


s_sub_u32 s22, s18, s47 // (op 41920) 


s_add_u32 s18, s22, s0 // (op 41920) 


s_cmp_eq_i32 s19, 0 // (op 41920) 
s_mov_b32 s0, scc // (op 41920) 

s_cmp_eq_i32 s0, 0 // (op 41920) 
s_cselect_b32 s19, s18, s1 // (op 41920) 



v_lshl_add_u32 v15, s19, 8, v25 // (op 41920) 


v_mul_lo_u32 v25, v15, s34 // (op 41920) 
v_add_u32 v25, v25, v14 // (op 41920) 


s_mul_i32 s1, s34, s21 // (op 41922) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 41922) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 41922) least significant
s_mul_hi_u32 s18, s34, s20 // (op 41922) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 41922) most significant: sum
s_add_u32 s1, s1, s18 // (op 41922) most significant: sum


s_mov_b32 s64, s36 // (op 41922) reg expression
s_mov_b32 s65, s37 // (op 41922) reg expression
s_mov_b32 s66, s0 // (op 41922) reg expression
s_mov_b32 s67, 131072 // (op 41922) call()

s_mul_hi_i32 s47, s16, s10 // (op 41951) 
v_and_b32 v14, 63, v1 // (op 41951) 
s_add_i32 s0, s47, s16 // (op 41951) 

v_lshrrev_b32 v15, 4, v14 // (op 41951) 
s_ashr_i32 s1, s0, 31 // (op 41951) 
v_and_b32 v26, 3, v15 // (op 41951) 

s_cmp_eq_i32 s10, 0 // (op 41951) 
s_mov_b32 s47, scc // (op 41951) 
v_and_b32 v15, 15, v1 // (op 41951) 
s_cmp_eq_i32 s47, 0 // (op 41951) 
s_cselect_b32 s18, 0, -1 // (op 41951) 

v_lshl_add_u32 v27, v26, 4, v15 // (op 41951) 


s_and_b32 s19, s13, 31 // (op 41951) 
v_lshrrev_b32 v15, 6, v1 // (op 41951) 
s_lshl_b32 s47, 1, s19 // (op 41951) 
v_and_b32 v26, 1, v15 // (op 41951) 

s_add_i32 s22, s47, s18 // (op 41951) 


v_lshl_add_u32 v15, v26, 7, v27 // (op 41951) 


s_and_b32 s18, s1, s22 // (op 41951) 


v_lshrrev_b32 v26, 6, v14 // (op 41951) 

s_add_i32 s1, s0, s18 // (op 41951) 


s_ashr_i32 s0, s1, s19 // (op 41951) 


s_lshl_b32 s1, s13, 24 // (op 41951) 
s_ashr_i32 s47, s1, 31 // (op 41951) 

s_xor_b32 s1, s0, s47 // (op 41951) 

s_sub_i32 s0, s1, s47 // (op 41951) 


s_mul_hi_i32 s1, s0, s14 // (op 41951) 
s_add_i32 s47, s1, s0 // (op 41951) 

s_ashr_i32 s1, s47, 31 // (op 41951) 
s_cmp_eq_i32 s14, 0 // (op 41951) 
s_mov_b32 s18, scc // (op 41951) 
s_cmp_eq_i32 s18, 0 // (op 41951) 
s_cselect_b32 s19, 0, -1 // (op 41951) 

s_and_b32 s18, s15, 31 // (op 41951) 
s_lshl_b32 s22, 1, s18 // (op 41951) 
s_add_i32 s23, s22, s19 // (op 41951) 


s_and_b32 s19, s1, s23 // (op 41951) 


s_add_i32 s1, s47, s19 // (op 41951) 


s_ashr_i32 s19, s1, s18 // (op 41951) 


s_lshl_b32 s1, s15, 24 // (op 41951) 
s_ashr_i32 s18, s1, 31 // (op 41951) 

s_xor_b32 s1, s19, s18 // (op 41951) 

s_sub_i32 s19, s1, s18 // (op 41951) 


s_add_u32 s22, s6, 256 // (op 41951) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 41951) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 41951) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 41951) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 41951) 

s_mul_i32 s1, s19, s22 // (op 41951) 

s_sub_i32 s47, s0, s1 // (op 41951) 


s_mul_hi_i32 s0, s16, s9 // (op 41951) 
s_add_i32 s1, s0, s16 // (op 41951) 

s_ashr_i32 s0, s1, 31 // (op 41951) 
s_cmp_eq_i32 s9, 0 // (op 41951) 
s_mov_b32 s18, scc // (op 41951) 
s_cmp_eq_i32 s18, 0 // (op 41951) 
s_cselect_b32 s19, 0, -1 // (op 41951) 

s_and_b32 s18, s45, 31 // (op 41951) 
s_lshl_b32 s24, 1, s18 // (op 41951) 
s_add_i32 s25, s24, s19 // (op 41951) 


s_and_b32 s19, s0, s25 // (op 41951) 


s_add_i32 s0, s1, s19 // (op 41951) 


s_ashr_i32 s1, s0, s18 // (op 41951) 


s_lshl_b32 s0, s45, 24 // (op 41951) 
s_ashr_i32 s18, s0, 31 // (op 41951) 

s_xor_b32 s0, s1, s18 // (op 41951) 

s_sub_i32 s1, s0, s18 // (op 41951) 


s_mul_i32 s0, s11, s22 // (op 41951) 

s_mul_i32 s18, s8, s0 // (op 41951) 

s_mul_i32 s0, s1, s18 // (op 41951) 

s_sub_u32 s18, s16, s0 // (op 41951) 

s_mul_hi_i32 s0, s18, s12 // (op 41951) 
s_add_i32 s19, s0, s18 // (op 41951) 


s_ashr_i32 s0, s19, 31 // (op 41951) 
s_cmp_eq_i32 s12, 0 // (op 41951) 
s_mov_b32 s18, scc // (op 41951) 
s_cmp_eq_i32 s18, 0 // (op 41951) 
s_cselect_b32 s22, 0, -1 // (op 41951) 

s_and_b32 s18, s44, 31 // (op 41951) 
s_lshl_b32 s23, 1, s18 // (op 41951) 
s_add_i32 s24, s23, s22 // (op 41951) 


s_and_b32 s25, s0, s24 // (op 41951) 


s_add_i32 s0, s19, s25 // (op 41951) 


s_ashr_i32 s19, s0, s18 // (op 41951) 


s_lshl_b32 s0, s44, 24 // (op 41951) 
s_ashr_i32 s18, s0, 31 // (op 41951) 

s_xor_b32 s0, s19, s18 // (op 41951) 

s_sub_i32 s19, s0, s18 // (op 41951) 


s_cmp_eq_i32 s1, 0 // (op 41951) 
s_mov_b32 s0, scc // (op 41951) 

s_cmp_eq_i32 s0, 0 // (op 41951) 
s_cselect_b32 s1, s19, s47 // (op 41951) 



v_lshl_add_u32 v14, s1, 8, v15 // (op 41951) 


v_mul_lo_u32 v15, v14, s38 // (op 41951) 

v_lshl_add_u32 v14, v26, 2, v15 // (op 41951) 


s_mul_i32 s1, s38, s7 // (op 41953) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 41953) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 41953) least significant
s_mul_hi_u32 s18, s38, s6 // (op 41953) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 41953) most significant: sum
s_add_u32 s1, s1, s18 // (op 41953) most significant: sum


s_mov_b32 s68, s40 // (op 41953) reg expression
s_mov_b32 s69, s41 // (op 41953) reg expression
s_mov_b32 s70, s0 // (op 41953) reg expression
s_mov_b32 s71, 131072 // (op 41953) call()

s_mul_hi_i32 s47, s16, s10 // (op 41972) 
v_and_b32 v15, 63, v1 // (op 41972) 
s_add_i32 s0, s47, s16 // (op 41972) 

v_lshrrev_b32 v26, 4, v15 // (op 41972) 
s_ashr_i32 s1, s0, 31 // (op 41972) 
v_and_b32 v27, 3, v26 // (op 41972) 

s_cmp_eq_i32 s10, 0 // (op 41972) 
s_mov_b32 s47, scc // (op 41972) 
v_and_b32 v26, 15, v1 // (op 41972) 
s_cmp_eq_i32 s47, 0 // (op 41972) 
s_cselect_b32 s18, 0, -1 // (op 41972) 

v_lshl_add_u32 v28, v27, 4, v26 // (op 41972) 


s_and_b32 s19, s13, 31 // (op 41972) 
v_lshrrev_b32 v26, 6, v1 // (op 41972) 
s_lshl_b32 s47, 1, s19 // (op 41972) 
v_and_b32 v27, 1, v26 // (op 41972) 

s_add_i32 s22, s47, s18 // (op 41972) 


v_lshl_add_u32 v26, v27, 7, v28 // (op 41972) 


s_and_b32 s18, s1, s22 // (op 41972) 


v_lshrrev_b32 v27, 6, v15 // (op 41972) 

s_add_i32 s1, s0, s18 // (op 41972) 


s_ashr_i32 s0, s1, s19 // (op 41972) 


s_lshl_b32 s1, s13, 24 // (op 41972) 
s_ashr_i32 s47, s1, 31 // (op 41972) 

s_xor_b32 s1, s0, s47 // (op 41972) 

s_sub_i32 s0, s1, s47 // (op 41972) 


s_mul_hi_i32 s1, s0, s14 // (op 41972) 
s_add_i32 s47, s1, s0 // (op 41972) 

s_ashr_i32 s1, s47, 31 // (op 41972) 
s_cmp_eq_i32 s14, 0 // (op 41972) 
s_mov_b32 s18, scc // (op 41972) 
s_cmp_eq_i32 s18, 0 // (op 41972) 
s_cselect_b32 s19, 0, -1 // (op 41972) 

s_and_b32 s18, s15, 31 // (op 41972) 
s_lshl_b32 s22, 1, s18 // (op 41972) 
s_add_i32 s23, s22, s19 // (op 41972) 


s_and_b32 s19, s1, s23 // (op 41972) 


s_add_i32 s1, s47, s19 // (op 41972) 


s_ashr_i32 s19, s1, s18 // (op 41972) 


s_lshl_b32 s1, s15, 24 // (op 41972) 
s_ashr_i32 s18, s1, 31 // (op 41972) 

s_xor_b32 s1, s19, s18 // (op 41972) 

s_sub_i32 s19, s1, s18 // (op 41972) 


s_add_u32 s22, s6, 256 // (op 41972) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 41972) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 41972) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 41972) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 41972) 

s_mul_i32 s1, s19, s22 // (op 41972) 

s_sub_i32 s47, s0, s1 // (op 41972) 


s_mul_hi_i32 s0, s16, s9 // (op 41972) 
s_add_i32 s1, s0, s16 // (op 41972) 

s_ashr_i32 s0, s1, 31 // (op 41972) 
s_cmp_eq_i32 s9, 0 // (op 41972) 
s_mov_b32 s18, scc // (op 41972) 
s_cmp_eq_i32 s18, 0 // (op 41972) 
s_cselect_b32 s19, 0, -1 // (op 41972) 

s_and_b32 s18, s45, 31 // (op 41972) 
s_lshl_b32 s24, 1, s18 // (op 41972) 
s_add_i32 s25, s24, s19 // (op 41972) 


s_and_b32 s19, s0, s25 // (op 41972) 


s_add_i32 s0, s1, s19 // (op 41972) 


s_ashr_i32 s1, s0, s18 // (op 41972) 


s_lshl_b32 s0, s45, 24 // (op 41972) 
s_ashr_i32 s18, s0, 31 // (op 41972) 

s_xor_b32 s0, s1, s18 // (op 41972) 

s_sub_i32 s1, s0, s18 // (op 41972) 


s_mul_i32 s0, s11, s22 // (op 41972) 

s_mul_i32 s18, s8, s0 // (op 41972) 

s_mul_i32 s0, s1, s18 // (op 41972) 

s_sub_u32 s18, s16, s0 // (op 41972) 

s_mul_hi_i32 s0, s18, s12 // (op 41972) 
s_add_i32 s19, s0, s18 // (op 41972) 


s_ashr_i32 s0, s19, 31 // (op 41972) 
s_cmp_eq_i32 s12, 0 // (op 41972) 
s_mov_b32 s18, scc // (op 41972) 
s_cmp_eq_i32 s18, 0 // (op 41972) 
s_cselect_b32 s22, 0, -1 // (op 41972) 

s_and_b32 s18, s44, 31 // (op 41972) 
s_lshl_b32 s23, 1, s18 // (op 41972) 
s_add_i32 s24, s23, s22 // (op 41972) 


s_and_b32 s25, s0, s24 // (op 41972) 


s_add_i32 s0, s19, s25 // (op 41972) 


s_ashr_i32 s19, s0, s18 // (op 41972) 


s_lshl_b32 s0, s44, 24 // (op 41972) 
s_ashr_i32 s18, s0, 31 // (op 41972) 

s_xor_b32 s0, s19, s18 // (op 41972) 

s_sub_i32 s19, s0, s18 // (op 41972) 


s_cmp_eq_i32 s1, 0 // (op 41972) 
s_mov_b32 s0, scc // (op 41972) 

s_cmp_eq_i32 s0, 0 // (op 41972) 
s_cselect_b32 s1, s19, s47 // (op 41972) 



v_lshl_add_u32 v15, s1, 8, v26 // (op 41972) 


v_mul_lo_u32 v26, v15, s38 // (op 41972) 

v_lshl_add_u32 v15, v27, 2, v26 // (op 41972) 


s_mul_i32 s1, s38, s7 // (op 41974) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 41974) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 41974) least significant
s_mul_hi_u32 s18, s38, s6 // (op 41974) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 41974) most significant: sum
s_add_u32 s1, s1, s18 // (op 41974) most significant: sum


s_mov_b32 s68, s40 // (op 41974) reg expression
s_mov_b32 s69, s41 // (op 41974) reg expression
s_mov_b32 s70, s0 // (op 41974) reg expression
s_mov_b32 s71, 131072 // (op 41974) call()

v_and_b32 v26, -4, v1 // (op 41573) 
v_sub_u32 v27, v1, v26 // (op 41573) 

v_lshlrev_b32 v26, 5, v27 // (op 41573) 

v_lshrrev_b32 v27, 2, v1 // (op 41573) 
v_lshl_add_u32 v28, v27, 7, v26 // (op 41573) 


v_lshlrev_b32 v26, 2, v28 // (op 41573) 

v_lshrrev_b32 v27, 3, v26 // (op 41573) 

s_nop 0
v_readfirstlane_b32 s47, v27 // (op 41573) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 6 // (op 41240) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 41240) 

v_lshrrev_b64 v[26:27], 3, s[18:19] // (op 41240) 


v_mov_b64 v[28:29], v[20:21] // (op 41240) 
s_mov_b32 m0, s47 // (op 41240) reg expression
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 41240) Load value direct to lds
v_add_co_u32 v28, s[0:1], v28, v26 // (op 41240) least significant half
v_addc_co_u32 v29, s[0:1], 0, v29, s[0:1] // (op 41240) most significant half

s_add_u32 m0, m0, 4096 // (op 41240) 
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 41240) Load value direct to lds
v_add_co_u32 v28, s[0:1], v28, v26 // (op 41240) least significant half
v_addc_co_u32 v29, s[0:1], 0, v29, s[0:1] // (op 41240) most significant half

s_add_u32 m0, m0, 4096 // (op 41240) 
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 41240) Load value direct to lds
v_add_co_u32 v28, s[0:1], v28, v26 // (op 41240) least significant half
v_addc_co_u32 v29, s[0:1], 0, v29, s[0:1] // (op 41240) most significant half

s_add_u32 m0, m0, 4096 // (op 41240) 
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 41240) Load value direct to lds



v_and_b32 v26, -4, v1 // (op 41516) 
v_sub_u32 v27, v1, v26 // (op 41516) 

v_lshlrev_b32 v26, 5, v27 // (op 41516) 

v_lshrrev_b32 v27, 2, v1 // (op 41516) 
v_lshl_add_u32 v28, v27, 7, v26 // (op 41516) 


v_lshlrev_b32 v26, 2, v28 // (op 41516) 

v_lshrrev_b32 v27, 3, v26 // (op 41516) 

s_nop 0
v_readfirstlane_b32 s47, v27 // (op 41516) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 6 // (op 41245) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 41245) 

v_lshrrev_b64 v[26:27], 3, s[18:19] // (op 41245) 


v_mov_b64 v[28:29], v[16:17] // (op 41245) 
s_add_u32 s0, 16384, s47 // (op 41245) 
s_mov_b32 m0, s0 // (op 41245) reg expression
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 41245) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 41245) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 41245) most significant half

s_add_u32 m0, m0, 4096 // (op 41245) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 41245) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 41245) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 41245) most significant half

s_add_u32 m0, m0, 4096 // (op 41245) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 41245) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 41245) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 41245) most significant half

s_add_u32 m0, m0, 4096 // (op 41245) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 41245) Load value direct to lds






v_mov_b32 v26, v14 // (op 76) reg expression

buffer_load_dwordx4 v[28:31], v26, s[68:71], 0 offen // (op 76) Load value



s_lshl_b32 s47, s38, 6 // (op 1104) 
v_add_u32 v26, v14, s47 // (op 1104) 


buffer_load_dwordx4 v[32:35], v26, s[68:71], 0 offen // (op 1104) Load value




s_lshl_b32 s47, s34, 6 // (op 8533) 
v_add_u32 v14, v24, s47 // (op 8533) 


buffer_load_dwordx4 v[36:39], v14, s[64:67], 0 offen // (op 8533) Load value



v_mov_b32 v14, v24 // (op 72) reg expression

buffer_load_dwordx4 v[40:43], v14, s[64:67], 0 offen // (op 72) Load value


v_and_b32 v14, -4, v1 // (op 41684) 
v_sub_u32 v24, v1, v14 // (op 41684) 

v_lshlrev_b32 v14, 5, v24 // (op 41684) 

v_lshrrev_b32 v24, 2, v1 // (op 41684) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 41684) 


v_lshlrev_b32 v14, 2, v26 // (op 41684) 

v_lshrrev_b32 v24, 3, v14 // (op 41684) 

s_nop 0
v_readfirstlane_b32 s47, v24 // (op 41684) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 6 // (op 41250) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 41250) 

v_lshrrev_b64 v[26:27], 3, s[18:19] // (op 41250) 


v_mov_b64 v[44:45], v[20:21] // (op 41250) 
s_add_u32 s0, 32768, s47 // (op 41250) 
s_mov_b32 m0, s0 // (op 41250) reg expression
buffer_load_dwordx4 v44, s[60:63], 64 offen offset: 0 lds // (op 41250) Load value direct to lds
v_add_co_u32 v44, s[18:19], v44, v26 // (op 41250) least significant half
v_addc_co_u32 v45, s[18:19], 0, v45, s[18:19] // (op 41250) most significant half

s_add_u32 m0, m0, 4096 // (op 41250) 
buffer_load_dwordx4 v44, s[60:63], 64 offen offset: 0 lds // (op 41250) Load value direct to lds
v_add_co_u32 v44, s[18:19], v44, v26 // (op 41250) least significant half
v_addc_co_u32 v45, s[18:19], 0, v45, s[18:19] // (op 41250) most significant half

s_add_u32 m0, m0, 4096 // (op 41250) 
buffer_load_dwordx4 v44, s[60:63], 64 offen offset: 0 lds // (op 41250) Load value direct to lds
v_add_co_u32 v44, s[18:19], v44, v26 // (op 41250) least significant half
v_addc_co_u32 v45, s[18:19], 0, v45, s[18:19] // (op 41250) most significant half

s_add_u32 m0, m0, 4096 // (op 41250) 
buffer_load_dwordx4 v44, s[60:63], 64 offen offset: 0 lds // (op 41250) Load value direct to lds




v_and_b32 v14, -4, v1 // (op 41640) 
v_sub_u32 v24, v1, v14 // (op 41640) 

v_lshlrev_b32 v14, 5, v24 // (op 41640) 

v_lshrrev_b32 v24, 2, v1 // (op 41640) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 41640) 


v_lshlrev_b32 v14, 2, v26 // (op 41640) 

v_lshrrev_b32 v24, 3, v14 // (op 41640) 

s_nop 0
v_readfirstlane_b32 s47, v24 // (op 41640) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 6 // (op 41252) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 41252) 

v_lshrrev_b64 v[26:27], 3, s[18:19] // (op 41252) 


v_mov_b64 v[44:45], v[16:17] // (op 41252) 
s_add_u32 s0, 49152, s47 // (op 41252) 
s_mov_b32 m0, s0 // (op 41252) reg expression
buffer_load_dwordx4 v44, s[48:51], 64 offen offset: 0 lds // (op 41252) Load value direct to lds
v_add_co_u32 v44, s[18:19], v44, v26 // (op 41252) least significant half
v_addc_co_u32 v45, s[18:19], 0, v45, s[18:19] // (op 41252) most significant half

s_add_u32 m0, m0, 4096 // (op 41252) 
buffer_load_dwordx4 v44, s[48:51], 64 offen offset: 0 lds // (op 41252) Load value direct to lds
v_add_co_u32 v44, s[18:19], v44, v26 // (op 41252) least significant half
v_addc_co_u32 v45, s[18:19], 0, v45, s[18:19] // (op 41252) most significant half

s_add_u32 m0, m0, 4096 // (op 41252) 
buffer_load_dwordx4 v44, s[48:51], 64 offen offset: 0 lds // (op 41252) Load value direct to lds
v_add_co_u32 v44, s[18:19], v44, v26 // (op 41252) least significant half
v_addc_co_u32 v45, s[18:19], 0, v45, s[18:19] // (op 41252) most significant half

s_add_u32 m0, m0, 4096 // (op 41252) 
buffer_load_dwordx4 v44, s[48:51], 64 offen offset: 0 lds // (op 41252) Load value direct to lds




v_and_b32 v14, -4, v1 // (op 41772) 
v_sub_u32 v24, v1, v14 // (op 41772) 

v_lshlrev_b32 v14, 5, v24 // (op 41772) 

v_lshrrev_b32 v24, 2, v1 // (op 41772) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 41772) 


v_lshlrev_b32 v14, 2, v26 // (op 41772) 

v_lshrrev_b32 v24, 3, v14 // (op 41772) 

s_nop 0
v_readfirstlane_b32 s47, v24 // (op 41772) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 6 // (op 41254) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 41254) 

v_lshrrev_b64 v[26:27], 3, s[18:19] // (op 41254) 


v_mov_b64 v[44:45], v[20:21] // (op 41254) 
s_add_u32 s0, 65536, s47 // (op 41254) 
s_mov_b32 m0, s0 // (op 41254) reg expression
s_mov_b32 s1, 128 // (op 41254) call()
buffer_load_dwordx4 v44, s[60:63], s1 offen offset: 0 lds // (op 41254) Load value direct to lds
v_add_co_u32 v44, s[18:19], v44, v26 // (op 41254) least significant half
v_addc_co_u32 v45, s[18:19], 0, v45, s[18:19] // (op 41254) most significant half

s_add_u32 m0, m0, 4096 // (op 41254) 
buffer_load_dwordx4 v44, s[60:63], s1 offen offset: 0 lds // (op 41254) Load value direct to lds
v_add_co_u32 v44, s[18:19], v44, v26 // (op 41254) least significant half
v_addc_co_u32 v45, s[18:19], 0, v45, s[18:19] // (op 41254) most significant half

s_add_u32 m0, m0, 4096 // (op 41254) 
buffer_load_dwordx4 v44, s[60:63], s1 offen offset: 0 lds // (op 41254) Load value direct to lds
v_add_co_u32 v44, s[18:19], v44, v26 // (op 41254) least significant half
v_addc_co_u32 v45, s[18:19], 0, v45, s[18:19] // (op 41254) most significant half

s_add_u32 m0, m0, 4096 // (op 41254) 
buffer_load_dwordx4 v44, s[60:63], s1 offen offset: 0 lds // (op 41254) Load value direct to lds





v_and_b32 v14, -4, v1 // (op 41728) 
v_sub_u32 v24, v1, v14 // (op 41728) 

v_lshlrev_b32 v14, 5, v24 // (op 41728) 

v_lshrrev_b32 v24, 2, v1 // (op 41728) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 41728) 


v_lshlrev_b32 v14, 2, v26 // (op 41728) 

v_lshrrev_b32 v24, 3, v14 // (op 41728) 

s_nop 0
v_readfirstlane_b32 s47, v24 // (op 41728) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 6 // (op 41256) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 41256) 

v_lshrrev_b64 v[26:27], 3, s[18:19] // (op 41256) 


v_mov_b64 v[44:45], v[16:17] // (op 41256) 
s_add_u32 s0, 81920, s47 // (op 41256) 
s_mov_b32 m0, s0 // (op 41256) reg expression
s_mov_b32 s1, 128 // (op 41256) call()
buffer_load_dwordx4 v44, s[48:51], s1 offen offset: 0 lds // (op 41256) Load value direct to lds
v_add_co_u32 v44, s[18:19], v44, v26 // (op 41256) least significant half
v_addc_co_u32 v45, s[18:19], 0, v45, s[18:19] // (op 41256) most significant half

s_add_u32 m0, m0, 4096 // (op 41256) 
buffer_load_dwordx4 v44, s[48:51], s1 offen offset: 0 lds // (op 41256) Load value direct to lds
v_add_co_u32 v44, s[18:19], v44, v26 // (op 41256) least significant half
v_addc_co_u32 v45, s[18:19], 0, v45, s[18:19] // (op 41256) most significant half

s_add_u32 m0, m0, 4096 // (op 41256) 
buffer_load_dwordx4 v44, s[48:51], s1 offen offset: 0 lds // (op 41256) Load value direct to lds
v_add_co_u32 v44, s[18:19], v44, v26 // (op 41256) least significant half
v_addc_co_u32 v45, s[18:19], 0, v45, s[18:19] // (op 41256) most significant half

s_add_u32 m0, m0, 4096 // (op 41256) 
buffer_load_dwordx4 v44, s[48:51], s1 offen offset: 0 lds // (op 41256) Load value direct to lds





v_and_b32 v14, -4, v1 // (op 41860) 
v_sub_u32 v24, v1, v14 // (op 41860) 

v_lshlrev_b32 v14, 5, v24 // (op 41860) 

v_lshrrev_b32 v24, 2, v1 // (op 41860) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 41860) 


v_lshlrev_b32 v14, 2, v26 // (op 41860) 

v_lshrrev_b32 v24, 3, v14 // (op 41860) 

s_nop 0
v_readfirstlane_b32 s47, v24 // (op 41860) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 6 // (op 41258) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 41258) 

v_lshrrev_b64 v[26:27], 3, s[18:19] // (op 41258) 


v_mov_b64 v[44:45], v[20:21] // (op 41258) 
s_add_u32 s0, 98304, s47 // (op 41258) 
s_mov_b32 m0, s0 // (op 41258) reg expression
s_mov_b32 s1, 192 // (op 41258) call()
buffer_load_dwordx4 v44, s[60:63], s1 offen offset: 0 lds // (op 41258) Load value direct to lds
v_add_co_u32 v44, s[18:19], v44, v26 // (op 41258) least significant half
v_addc_co_u32 v45, s[18:19], 0, v45, s[18:19] // (op 41258) most significant half

s_add_u32 m0, m0, 4096 // (op 41258) 
buffer_load_dwordx4 v44, s[60:63], s1 offen offset: 0 lds // (op 41258) Load value direct to lds
v_add_co_u32 v44, s[18:19], v44, v26 // (op 41258) least significant half
v_addc_co_u32 v45, s[18:19], 0, v45, s[18:19] // (op 41258) most significant half

s_add_u32 m0, m0, 4096 // (op 41258) 
buffer_load_dwordx4 v44, s[60:63], s1 offen offset: 0 lds // (op 41258) Load value direct to lds
v_add_co_u32 v44, s[18:19], v44, v26 // (op 41258) least significant half
v_addc_co_u32 v45, s[18:19], 0, v45, s[18:19] // (op 41258) most significant half

s_add_u32 m0, m0, 4096 // (op 41258) 
buffer_load_dwordx4 v44, s[60:63], s1 offen offset: 0 lds // (op 41258) Load value direct to lds






v_and_b32 v14, -4, v1 // (op 41816) 
v_sub_u32 v24, v1, v14 // (op 41816) 

v_lshlrev_b32 v14, 5, v24 // (op 41816) 

v_lshrrev_b32 v24, 2, v1 // (op 41816) 
v_lshl_add_u32 v20, v24, 7, v14 // (op 41816) 


v_lshlrev_b32 v14, 2, v20 // (op 41816) 

v_lshrrev_b32 v24, 3, v14 // (op 41816) 

s_nop 0
v_readfirstlane_b32 s47, v24 // (op 41816) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 6 // (op 41260) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 41260) 

v_lshrrev_b64 v[20:21], 3, s[18:19] // (op 41260) 


v_mov_b64 v[26:27], v[16:17] // (op 41260) 
s_add_u32 s0, 114688, s47 // (op 41260) 
s_mov_b32 m0, s0 // (op 41260) reg expression
s_mov_b32 s1, 192 // (op 41260) call()
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 41260) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v20 // (op 41260) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 41260) most significant half

s_add_u32 m0, m0, 4096 // (op 41260) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 41260) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v20 // (op 41260) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 41260) most significant half

s_add_u32 m0, m0, 4096 // (op 41260) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 41260) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v20 // (op 41260) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 41260) most significant half

s_add_u32 m0, m0, 4096 // (op 41260) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 41260) Load value direct to lds






s_waitcnt vmcnt(32)
s_barrier  // (op 42702) 



ds_read_b128 v[44:47], v4 // (op 40101) Load local data 



ds_read_b128 v[48:51], v4 offset:1024 // (op 40106) Load local data 



ds_read_b128 v[52:55], v4 offset:2048 // (op 40111) Load local data 



ds_read_b128 v[56:59], v4 offset:3072 // (op 40116) Load local data 



ds_read_b128 v[60:63], v4 offset:4096 // (op 40121) Load local data 



ds_read_b128 v[64:67], v4 offset:5120 // (op 40126) Load local data 



ds_read_b128 v[68:71], v4 offset:6144 // (op 40131) Load local data 



ds_read_b128 v[72:75], v4 offset:7168 // (op 40136) Load local data 
s_waitcnt vmcnt(28)
s_barrier  // (op 42685) 



ds_read_b128 v[76:79], v0 offset:16384 // (op 40061) Load local data 



ds_read_b128 v[80:83], v0 offset:17408 // (op 40066) Load local data 



ds_read_b128 v[84:87], v0 offset:18432 // (op 40071) Load local data 



ds_read_b128 v[88:91], v0 offset:19456 // (op 40076) Load local data 



ds_read_b128 v[92:95], v0 offset:20480 // (op 40081) Load local data 



ds_read_b128 v[96:99], v0 offset:21504 // (op 40086) Load local data 



ds_read_b128 v[100:103], v0 offset:22528 // (op 40091) Load local data 



ds_read_b128 v[104:107], v0 offset:23552 // (op 40096) Load local data 
v_and_b32 v14, -4, v1 // (op 41488) 
v_sub_u32 v24, v1, v14 // (op 41488) 

v_lshlrev_b32 v14, 5, v24 // (op 41488) 

v_lshrrev_b32 v24, 2, v1 // (op 41488) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 41488) 


v_lshlrev_b32 v14, 2, v16 // (op 41488) 

v_lshrrev_b32 v24, 3, v14 // (op 41488) 

s_nop 0
v_readfirstlane_b32 s47, v24 // (op 41488) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 41545) 
v_sub_u32 v24, v1, v14 // (op 41545) 

v_lshlrev_b32 v14, 5, v24 // (op 41545) 

v_lshrrev_b32 v24, 2, v1 // (op 41545) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 41545) 


v_lshlrev_b32 v14, 2, v16 // (op 41545) 

v_lshrrev_b32 v24, 3, v14 // (op 41545) 

s_nop 0
v_readfirstlane_b32 s0, v24 // (op 41545) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 41625) 
v_sub_u32 v24, v1, v14 // (op 41625) 

v_lshlrev_b32 v14, 5, v24 // (op 41625) 

v_lshrrev_b32 v24, 2, v1 // (op 41625) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 41625) 


v_lshlrev_b32 v14, 2, v16 // (op 41625) 

v_lshrrev_b32 v24, 3, v14 // (op 41625) 

s_nop 0
v_readfirstlane_b32 s1, v24 // (op 41625) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 41669) 
v_sub_u32 v24, v1, v14 // (op 41669) 

v_lshlrev_b32 v14, 5, v24 // (op 41669) 

v_lshrrev_b32 v24, 2, v1 // (op 41669) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 41669) 


v_lshlrev_b32 v14, 2, v16 // (op 41669) 

v_lshrrev_b32 v24, 3, v14 // (op 41669) 

s_nop 0
v_readfirstlane_b32 s18, v24 // (op 41669) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 41713) 
v_sub_u32 v24, v1, v14 // (op 41713) 

v_lshlrev_b32 v14, 5, v24 // (op 41713) 

v_lshrrev_b32 v24, 2, v1 // (op 41713) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 41713) 


v_lshlrev_b32 v14, 2, v16 // (op 41713) 

v_lshrrev_b32 v24, 3, v14 // (op 41713) 

s_nop 0
v_readfirstlane_b32 s19, v24 // (op 41713) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 41757) 
v_sub_u32 v24, v1, v14 // (op 41757) 

v_lshlrev_b32 v14, 5, v24 // (op 41757) 

v_lshrrev_b32 v24, 2, v1 // (op 41757) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 41757) 


v_lshlrev_b32 v14, 2, v16 // (op 41757) 

v_lshrrev_b32 v24, 3, v14 // (op 41757) 

s_nop 0
v_readfirstlane_b32 s22, v24 // (op 41757) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 41801) 
v_sub_u32 v24, v1, v14 // (op 41801) 

v_lshlrev_b32 v14, 5, v24 // (op 41801) 

v_lshrrev_b32 v24, 2, v1 // (op 41801) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 41801) 


v_lshlrev_b32 v14, 2, v16 // (op 41801) 

v_lshrrev_b32 v24, 3, v14 // (op 41801) 

s_nop 0
v_readfirstlane_b32 s23, v24 // (op 41801) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 41845) 
v_sub_u32 v24, v1, v14 // (op 41845) 

v_lshlrev_b32 v14, 5, v24 // (op 41845) 

v_lshrrev_b32 v24, 2, v1 // (op 41845) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 41845) 


v_lshlrev_b32 v14, 2, v16 // (op 41845) 

v_lshrrev_b32 v24, 3, v14 // (op 41845) 

s_nop 0
v_readfirstlane_b32 s24, v24 // (op 41845) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s25, 0 // (op 12280) call()
s_ashr_i64 s[72:73], s[42:43], 7 // (op 12279) 
s_mov_b32 s74, s72 // (op 12279) convert

s_and_b32 s72, s74, 4294967292 // (op 12279) 

s_cmp_lt_u32 s25, s72 // (op 12279) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WGM__ForLoopBottom_KLoop_12279 // (op 12279) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WGM__ForLoopBottom_KLoop_12279 if false)
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WGM__ForLoopTop_KLoop_12279:
s_add_u32 s25, s25, 4 // (op 12281) 
s_waitcnt vmcnt(27)
v_mov_b32 v108, v28 // (op 41098) reg expression
v_mov_b32 v109, v29 // (op 41098) reg expression
v_mov_b32 v110, v30 // (op 41098) reg expression
v_mov_b32 v111, v31 // (op 41098) reg expression
s_waitcnt lgkmcnt(8)
s_barrier  // (op 42705) 
s_waitcnt vmcnt(26)
v_mov_b32 v112, v32 // (op 41111) reg expression

v_mov_b32 v113, v33 // (op 41111) reg expression
v_mov_b32 v114, v34 // (op 41111) reg expression
v_mov_b32 v115, v35 // (op 41111) reg expression
s_lshl_b64 s[72:73], s[30:31], 6 // (op 41193) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 41193) 

v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 41193) 


v_mov_b64 v[20:21], v[22:23] // (op 41193) 
v_mov_b64 v[26:27], v[20:21] // (op 41193) 
s_mov_b32 m0, s0 // (op 41193) reg expression
s_mov_b32 s72, 256 // (op 41193) call()
buffer_load_dwordx4 v26, s[60:63], s72 offen offset: 0 lds // (op 41193) Load value direct to lds
v_add_co_u32 v26, s[74:75], v26, v16 // (op 41193) least significant half
v_addc_co_u32 v27, s[74:75], 0, v27, s[74:75] // (op 41193) most significant half

s_add_u32 m0, m0, 4096 // (op 41193) 
buffer_load_dwordx4 v26, s[60:63], s72 offen offset: 0 lds // (op 41193) Load value direct to lds
v_add_co_u32 v26, s[74:75], v26, v16 // (op 41193) least significant half
v_addc_co_u32 v27, s[74:75], 0, v27, s[74:75] // (op 41193) most significant half

s_add_u32 m0, m0, 4096 // (op 41193) 
buffer_load_dwordx4 v26, s[60:63], s72 offen offset: 0 lds // (op 41193) Load value direct to lds
v_add_co_u32 v26, s[74:75], v26, v16 // (op 41193) least significant half
v_addc_co_u32 v27, s[74:75], 0, v27, s[74:75] // (op 41193) most significant half

s_add_u32 m0, m0, 4096 // (op 41193) 
s_waitcnt vmcnt(28)
v_mov_b32 v116, v36 // (op 41124) reg expression
v_mov_b32 v117, v37 // (op 41124) reg expression
v_mov_b32 v118, v38 // (op 41124) reg expression
v_mov_b32 v119, v39 // (op 41124) reg expression
s_waitcnt vmcnt(27)
v_mov_b32 v120, v40 // (op 41137) reg expression
v_mov_b32 v121, v41 // (op 41137) reg expression
v_mov_b32 v122, v42 // (op 41137) reg expression
v_mov_b32 v123, v43 // (op 41137) reg expression
buffer_load_dwordx4 v26, s[60:63], s72 offen offset: 0 lds // (op 41193) Load value direct to lds



v_bfe_u32 v124, v120, 0, 8 // (op 37684) 
v_bfe_u32 v125, v120, 8, 8 // (op 37684) 
v_bfe_u32 v126, v120, 16, 8 // (op 37684) 
v_bfe_u32 v127, v120, 24, 8 // (op 37684) 
v_permlane16_swap_b32 v124, v125 // (op 37684) 
s_nop 0
v_permlane16_swap_b32 v126, v127 // (op 37684) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v124, v126 // (op 37684) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v125, v127 // (op 37684) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 42688) 

s_lshl_b64 s[72:73], s[26:27], 6 // (op 41188) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 41188) 

v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 41188) 


v_mov_b64 v[26:27], v[18:19] // (op 41188) 
v_mov_b64 v[128:129], v[26:27] // (op 41188) 
s_add_u32 s72, 16384, s47 // (op 41188) 
s_mov_b32 m0, s72 // (op 41188) reg expression
s_mov_b32 s73, 256 // (op 41188) call()
buffer_load_dwordx4 v128, s[48:51], s73 offen offset: 0 lds // (op 41188) Load value direct to lds
v_add_co_u32 v128, s[74:75], v128, v16 // (op 41188) least significant half
v_addc_co_u32 v129, s[74:75], 0, v129, s[74:75] // (op 41188) most significant half

s_add_u32 m0, m0, 4096 // (op 41188) 
buffer_load_dwordx4 v128, s[48:51], s73 offen offset: 0 lds // (op 41188) Load value direct to lds
v_add_co_u32 v128, s[74:75], v128, v16 // (op 41188) least significant half
v_addc_co_u32 v129, s[74:75], 0, v129, s[74:75] // (op 41188) most significant half

s_add_u32 m0, m0, 4096 // (op 41188) 
v_bfe_u32 v130, v108, 0, 8 // (op 38073) 
v_bfe_u32 v131, v108, 8, 8 // (op 38073) 
v_bfe_u32 v132, v108, 16, 8 // (op 38073) 
v_bfe_u32 v133, v108, 24, 8 // (op 38073) 
v_permlane16_swap_b32 v130, v131 // (op 38073) 
s_nop 0
v_permlane16_swap_b32 v132, v133 // (op 38073) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v130, v132 // (op 38073) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v131, v133 // (op 38073) 
buffer_load_dwordx4 v128, s[48:51], s73 offen offset: 0 lds // (op 41188) Load value direct to lds
v_add_co_u32 v128, s[74:75], v128, v16 // (op 41188) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[76:79], v[44:47], a[0:3], v124, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 78) 
v_addc_co_u32 v129, s[74:75], 0, v129, s[74:75] // (op 41188) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[76:79], v[48:51], a[4:7], v124, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 421) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[76:79], v[52:55], a[8:11], v124, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 647) 
s_add_u32 m0, m0, 4096 // (op 41188) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[76:79], v[56:59], a[12:15], v124, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 873) 
v_bfe_u32 v134, v112, 0, 8 // (op 38141) 
v_bfe_u32 v135, v112, 8, 8 // (op 38141) 
v_bfe_u32 v136, v112, 16, 8 // (op 38141) 
v_bfe_u32 v137, v112, 24, 8 // (op 38141) 
buffer_load_dwordx4 v128, s[48:51], s73 offen offset: 0 lds // (op 41188) Load value direct to lds




v_permlane16_swap_b32 v134, v135 // (op 38141) 


v_permlane16_swap_b32 v136, v137 // (op 38141) 
v_mov_b32 v14, v15 // (op 41106) 
v_add_u32 v24, 16, v14 // (op 41106) 

buffer_load_dwordx4 v[28:31], v24, s[68:71], 0 offen // (op 41106) Load value

v_permlane32_swap_b32 v134, v136 // (op 38141) 


v_permlane32_swap_b32 v135, v137 // (op 38141) 
s_lshl_b32 s72, s38, 6 // (op 41119) 
s_add_u32 s72, s72, 16 // (op 41119) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[76:79], v[60:63], a[16:19], v124, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1099) 
v_add_u32 v24, v14, s72 // (op 41119) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[76:79], v[64:67], a[20:23], v124, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1325) 
v_bfe_u32 v138, v116, 0, 8 // (op 37749) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[76:79], v[68:71], a[24:27], v124, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1551) 
v_bfe_u32 v139, v116, 8, 8 // (op 37749) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[76:79], v[72:75], a[28:31], v124, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1777) 
v_bfe_u32 v140, v116, 16, 8 // (op 37749) 
v_bfe_u32 v141, v116, 24, 8 // (op 37749) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[80:83], v[44:47], a[32:35], v125, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2997) 
v_permlane16_swap_b32 v138, v139 // (op 37749) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[80:83], v[48:51], a[36:39], v125, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2921) 
v_permlane16_swap_b32 v140, v141 // (op 37749) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[80:83], v[52:55], a[40:43], v125, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2845) 
s_nop 0
v_permlane32_swap_b32 v138, v140 // (op 37749) Wait state hazard: VALU Write Readlane Hazard
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[80:83], v[56:59], a[44:47], v125, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2769) 
v_permlane32_swap_b32 v139, v141 // (op 37749) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[80:83], v[60:63], a[48:51], v125, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2693) 
s_waitcnt vmcnt(29)
s_barrier  // (op 42725) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[80:83], v[64:67], a[52:55], v125, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2617) 
ds_read_b128 v[142:145], v6 offset:32768 // (op 216) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[80:83], v[68:71], a[56:59], v125, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2541) 
ds_read_b128 v[146:149], v6 offset:33792 // (op 432) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[80:83], v[72:75], a[60:63], v125, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2465) 
ds_read_b128 v[150:153], v6 offset:34816 // (op 658) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[84:87], v[44:47], a[64:67], v126, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4999) 
ds_read_b128 v[154:157], v6 offset:35840 // (op 884) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[84:87], v[48:51], a[68:71], v126, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4923) 
ds_read_b128 v[158:161], v6 offset:36864 // (op 1110) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[84:87], v[52:55], a[72:75], v126, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4847) 
ds_read_b128 v[162:165], v6 offset:37888 // (op 1336) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[84:87], v[56:59], a[76:79], v126, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4771) 
ds_read_b128 v[166:169], v6 offset:38912 // (op 1562) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[84:87], v[60:63], a[80:83], v126, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4695) 
ds_read_b128 v[80:83], v6 offset:39936 // (op 1788) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[84:87], v[64:67], a[84:87], v126, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4619) 
buffer_load_dwordx4 v[32:35], v24, s[68:71], 0 offen // (op 41119) Load value


v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[84:87], v[68:71], a[88:91], v126, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4543) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[84:87], v[72:75], a[92:95], v126, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4467) 
v_add_u32 v15, 16, v15 // (op 41962) 
v_mov_b32 v14, v25 // (op 41132) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[88:91], v[44:47], a[96:99], v127, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7001) 
s_lshl_b32 s72, s34, 6 // (op 41132) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[88:91], v[48:51], a[100:103], v127, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6925) 
s_add_u32 s73, 16, s72 // (op 41132) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[88:91], v[52:55], a[104:107], v127, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6849) 
v_add_u32 v24, v14, s73 // (op 41132) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[88:91], v[56:59], a[108:111], v127, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6773) 
buffer_load_dwordx4 v[36:39], v24, s[64:67], 0 offen // (op 41132) Load value



v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[88:91], v[60:63], a[112:115], v127, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6697) 
v_add_u32 v24, 16, v14 // (op 41145) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[88:91], v[64:67], a[116:119], v127, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6621) 
buffer_load_dwordx4 v[40:43], v24, s[64:67], 0 offen // (op 41145) Load value


v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[88:91], v[68:71], a[120:123], v127, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6545) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[88:91], v[72:75], a[124:127], v127, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6469) 
v_add_u32 v25, 16, v25 // (op 41910) 

s_waitcnt vmcnt(28)
s_barrier  // (op 42718) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[92:95], v[44:47], a[128:131], v138, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9003) 
ds_read_b128 v[76:79], v3 offset:49152 // (op 214) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[92:95], v[48:51], a[132:135], v138, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8927) 
ds_read_b128 v[84:87], v3 offset:50176 // (op 2475) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[92:95], v[52:55], a[136:139], v138, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8851) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[92:95], v[56:59], a[140:143], v138, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8775) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[92:95], v[60:63], a[144:147], v138, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8699) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[92:95], v[64:67], a[148:151], v138, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8623) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[92:95], v[68:71], a[152:155], v138, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8547) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[92:95], v[72:75], a[156:159], v138, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8471) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[96:99], v[44:47], a[160:163], v139, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11005) 
ds_read_b128 v[88:91], v3 offset:51200 // (op 4477) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[96:99], v[48:51], a[164:167], v139, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10929) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[96:99], v[52:55], a[168:171], v139, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10853) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[96:99], v[56:59], a[172:175], v139, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10777) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[96:99], v[60:63], a[176:179], v139, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10701) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[96:99], v[64:67], a[180:183], v139, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10625) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[96:99], v[68:71], a[184:187], v139, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10549) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[96:99], v[72:75], a[188:191], v139, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10473) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[100:103], v[44:47], a[192:195], v140, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13007) 
ds_read_b128 v[92:95], v3 offset:52224 // (op 6479) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[100:103], v[48:51], a[196:199], v140, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12931) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[100:103], v[52:55], a[200:203], v140, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12855) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[100:103], v[56:59], a[204:207], v140, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12779) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[100:103], v[60:63], a[208:211], v140, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12703) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[100:103], v[64:67], a[212:215], v140, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12627) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[100:103], v[68:71], a[216:219], v140, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12551) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[100:103], v[72:75], a[220:223], v140, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12475) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[104:107], v[44:47], a[224:227], v141, v130 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 15009) 
ds_read_b128 v[96:99], v3 offset:53248 // (op 8481) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[104:107], v[48:51], a[228:231], v141, v131 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14933) 
ds_read_b128 v[100:103], v3 offset:54272 // (op 10483) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[104:107], v[52:55], a[232:235], v141, v132 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14857) 
ds_read_b128 v[124:127], v3 offset:55296 // (op 12485) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[104:107], v[56:59], a[236:239], v141, v133 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14781) 
ds_read_b128 v[170:173], v3 offset:56320 // (op 14487) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[104:107], v[60:63], a[240:243], v141, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14705) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[104:107], v[64:67], a[244:247], v141, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14629) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[104:107], v[68:71], a[248:251], v141, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14553) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[104:107], v[72:75], a[252:255], v141, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14477) 


v_bfe_u32 v128, v121, 0, 8 // (op 37699) 
v_bfe_u32 v129, v121, 8, 8 // (op 37699) 
v_bfe_u32 v130, v121, 16, 8 // (op 37699) 
v_bfe_u32 v131, v121, 24, 8 // (op 37699) 
v_permlane16_swap_b32 v128, v129 // (op 37699) 
s_nop 0
v_permlane16_swap_b32 v130, v131 // (op 37699) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v128, v130 // (op 37699) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v129, v131 // (op 37699) 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 42729) 

s_lshl_b64 s[72:73], s[30:31], 6 // (op 41215) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 41215) 

v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 41215) 


v_mov_b64 v[132:133], v[20:21] // (op 41215) 
s_add_u32 s72, 32768, s18 // (op 41215) 
s_mov_b32 m0, s72 // (op 41215) reg expression
s_mov_b32 s73, 320 // (op 41215) call()
v_bfe_u32 v134, v109, 0, 8 // (op 38088) 
v_bfe_u32 v135, v109, 8, 8 // (op 38088) 
v_bfe_u32 v136, v109, 16, 8 // (op 38088) 
v_bfe_u32 v137, v109, 24, 8 // (op 38088) 
v_permlane16_swap_b32 v134, v135 // (op 38088) 
s_nop 0
v_permlane16_swap_b32 v136, v137 // (op 38088) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v134, v136 // (op 38088) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v135, v137 // (op 38088) 
buffer_load_dwordx4 v132, s[60:63], s73 offen offset: 0 lds // (op 41215) Load value direct to lds
v_add_co_u32 v132, s[74:75], v132, v16 // (op 41215) least significant half
s_waitcnt lgkmcnt(7)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[76:79], v[142:145], a[0:3], v128, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 221) 
v_addc_co_u32 v133, s[74:75], 0, v133, s[74:75] // (op 41215) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[76:79], v[146:149], a[4:7], v128, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 422) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[76:79], v[150:153], a[8:11], v128, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 648) 
s_add_u32 m0, m0, 4096 // (op 41215) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[76:79], v[154:157], a[12:15], v128, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 874) 
v_bfe_u32 v138, v113, 0, 8 // (op 38156) 
v_bfe_u32 v139, v113, 8, 8 // (op 38156) 
v_bfe_u32 v140, v113, 16, 8 // (op 38156) 
v_bfe_u32 v141, v113, 24, 8 // (op 38156) 
v_permlane16_swap_b32 v138, v139 // (op 38156) 
s_nop 0
v_permlane16_swap_b32 v140, v141 // (op 38156) Wait state hazard: VALU Write Readlane Hazard
buffer_load_dwordx4 v132, s[60:63], s73 offen offset: 0 lds // (op 41215) Load value direct to lds
v_add_co_u32 v132, s[74:75], v132, v16 // (op 41215) least significant half
v_addc_co_u32 v133, s[74:75], 0, v133, s[74:75] // (op 41215) most significant half

s_add_u32 m0, m0, 4096 // (op 41215) 
v_permlane32_swap_b32 v138, v140 // (op 38156) 
v_permlane32_swap_b32 v139, v141 // (op 38156) 
buffer_load_dwordx4 v132, s[60:63], s73 offen offset: 0 lds // (op 41215) Load value direct to lds
v_add_co_u32 v132, s[74:75], v132, v16 // (op 41215) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[76:79], v[158:161], a[16:19], v128, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1100) 
v_addc_co_u32 v133, s[74:75], 0, v133, s[74:75] // (op 41215) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[76:79], v[162:165], a[20:23], v128, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1326) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[76:79], v[166:169], a[24:27], v128, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1552) 
s_add_u32 m0, m0, 4096 // (op 41215) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[76:79], v[80:83], a[28:31], v128, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1778) 
v_bfe_u32 v174, v117, 0, 8 // (op 37764) 
v_bfe_u32 v175, v117, 8, 8 // (op 37764) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[84:87], v[142:145], a[32:35], v129, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2998) 
v_bfe_u32 v176, v117, 16, 8 // (op 37764) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[84:87], v[146:149], a[36:39], v129, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2922) 
v_bfe_u32 v177, v117, 24, 8 // (op 37764) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[84:87], v[150:153], a[40:43], v129, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2846) 
v_permlane16_swap_b32 v174, v175 // (op 37764) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[84:87], v[154:157], a[44:47], v129, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2770) 
v_permlane16_swap_b32 v176, v177 // (op 37764) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[84:87], v[158:161], a[48:51], v129, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2694) 
s_nop 0
v_permlane32_swap_b32 v174, v176 // (op 37764) Wait state hazard: VALU Write Readlane Hazard
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[84:87], v[162:165], a[52:55], v129, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2618) 
v_permlane32_swap_b32 v175, v177 // (op 37764) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[84:87], v[166:169], a[56:59], v129, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2542) 
s_waitcnt vmcnt(27)
s_barrier  // (op 42740) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[84:87], v[80:83], a[60:63], v129, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2466) 
v_add_u32 v14, 65536, v8 // (op 244) 
ds_read_b128 v[44:47], v14 // (op 244) Load local data 




s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[88:91], v[142:145], a[64:67], v130, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5000) 
v_add_u32 v14, 66560, v8 // (op 442) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[88:91], v[146:149], a[68:71], v130, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4924) 
ds_read_b128 v[48:51], v14 // (op 442) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[88:91], v[150:153], a[72:75], v130, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4848) 
v_add_u32 v14, 67584, v8 // (op 668) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[88:91], v[154:157], a[76:79], v130, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4772) 
ds_read_b128 v[52:55], v14 // (op 668) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[88:91], v[158:161], a[80:83], v130, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4696) 
v_add_u32 v14, 68608, v8 // (op 894) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[88:91], v[162:165], a[84:87], v130, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4620) 
ds_read_b128 v[56:59], v14 // (op 894) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[88:91], v[166:169], a[88:91], v130, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4544) 
v_add_u32 v14, 69632, v8 // (op 1120) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[88:91], v[80:83], a[92:95], v130, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4468) 
ds_read_b128 v[60:63], v14 // (op 1120) Load local data 




s_waitcnt lgkmcnt(9)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[92:95], v[142:145], a[96:99], v131, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7002) 
v_add_u32 v14, 70656, v8 // (op 1346) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[92:95], v[146:149], a[100:103], v131, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6926) 
ds_read_b128 v[64:67], v14 // (op 1346) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[92:95], v[150:153], a[104:107], v131, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6850) 
v_add_u32 v14, 71680, v8 // (op 1572) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[92:95], v[154:157], a[108:111], v131, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6774) 
ds_read_b128 v[68:71], v14 // (op 1572) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[92:95], v[158:161], a[112:115], v131, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6698) 
v_add_u32 v14, 72704, v8 // (op 1798) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[92:95], v[162:165], a[116:119], v131, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6622) 
ds_read_b128 v[72:75], v14 // (op 1798) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[92:95], v[166:169], a[120:123], v131, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6546) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[92:95], v[80:83], a[124:127], v131, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6470) 
s_waitcnt vmcnt(23)
s_barrier  // (op 42733) 




s_waitcnt lgkmcnt(11)
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[96:99], v[142:145], a[128:131], v174, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9004) 
v_add_u32 v14, 81920, v5 // (op 242) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[96:99], v[146:149], a[132:135], v174, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8928) 
ds_read_b128 v[76:79], v14 // (op 242) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[96:99], v[150:153], a[136:139], v174, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8852) 
v_add_u32 v14, 82944, v5 // (op 2485) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[96:99], v[154:157], a[140:143], v174, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8776) 
ds_read_b128 v[84:87], v14 // (op 2485) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[96:99], v[158:161], a[144:147], v174, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8700) 
buffer_load_dwordx4 v132, s[60:63], s73 offen offset: 0 lds // (op 41215) Load value direct to lds




v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[96:99], v[162:165], a[148:151], v174, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8624) 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 42722) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[96:99], v[166:169], a[152:155], v174, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8548) 
s_lshl_b64 s[72:73], s[26:27], 6 // (op 41210) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[96:99], v[80:83], a[156:159], v174, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8472) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 41210) 

v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 41210) 





v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[100:103], v[142:145], a[160:163], v175, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11006) 
v_add_u32 v14, 83968, v5 // (op 4487) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[100:103], v[146:149], a[164:167], v175, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10930) 
ds_read_b128 v[88:91], v14 // (op 4487) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[100:103], v[150:153], a[168:171], v175, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10854) 
v_mov_b64 v[128:129], v[26:27] // (op 41210) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[100:103], v[154:157], a[172:175], v175, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10778) 
s_add_u32 s72, 49152, s1 // (op 41210) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[100:103], v[158:161], a[176:179], v175, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10702) 
s_mov_b32 m0, s72 // (op 41210) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[100:103], v[162:165], a[180:183], v175, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10626) 
s_mov_b32 s73, 320 // (op 41210) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[100:103], v[166:169], a[184:187], v175, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10550) 
buffer_load_dwordx4 v128, s[48:51], s73 offen offset: 0 lds // (op 41210) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[100:103], v[80:83], a[188:191], v175, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10474) 
v_add_co_u32 v128, s[74:75], v128, v16 // (op 41210) least significant half
v_addc_co_u32 v129, s[74:75], 0, v129, s[74:75] // (op 41210) most significant half




v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[124:127], v[142:145], a[192:195], v176, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13008) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[124:127], v[146:149], a[196:199], v176, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12932) 
s_add_u32 m0, m0, 4096 // (op 41210) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[124:127], v[150:153], a[200:203], v176, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12856) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[124:127], v[154:157], a[204:207], v176, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12780) 
buffer_load_dwordx4 v128, s[48:51], s73 offen offset: 0 lds // (op 41210) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[124:127], v[158:161], a[208:211], v176, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12704) 
v_add_u32 v14, 84992, v5 // (op 6489) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[124:127], v[162:165], a[212:215], v176, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12628) 
ds_read_b128 v[92:95], v14 // (op 6489) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[124:127], v[166:169], a[216:219], v176, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12552) 
v_add_co_u32 v128, s[74:75], v128, v16 // (op 41210) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[124:127], v[80:83], a[220:223], v176, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12476) 
v_addc_co_u32 v129, s[74:75], 0, v129, s[74:75] // (op 41210) most significant half





v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[170:173], v[142:145], a[224:227], v177, v134 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 15010) 
s_add_u32 m0, m0, 4096 // (op 41210) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[170:173], v[146:149], a[228:231], v177, v135 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14934) 
v_add_u32 v14, 86016, v5 // (op 8491) 

ds_read_b128 v[124:127], v14 // (op 8491) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[170:173], v[150:153], a[232:235], v177, v136 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14858) 
v_add_u32 v14, 87040, v5 // (op 10493) 

ds_read_b128 v[130:133], v14 // (op 10493) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[170:173], v[154:157], a[236:239], v177, v137 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14782) 
v_add_u32 v14, 88064, v5 // (op 12495) 


ds_read_b128 v[134:137], v14 // (op 12495) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[170:173], v[158:161], a[240:243], v177, v138 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14706) 
v_add_u32 v14, 89088, v5 // (op 14497) 

ds_read_b128 v[142:145], v14 // (op 14497) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[170:173], v[162:165], a[244:247], v177, v139 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14630) 
buffer_load_dwordx4 v128, s[48:51], s73 offen offset: 0 lds // (op 41210) Load value direct to lds

v_add_co_u32 v128, s[74:75], v128, v16 // (op 41210) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[170:173], v[166:169], a[248:251], v177, v140 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14554) 
v_addc_co_u32 v129, s[74:75], 0, v129, s[74:75] // (op 41210) most significant half


v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[170:173], v[80:83], a[252:255], v177, v141 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14478) 
s_add_u32 m0, m0, 4096 // (op 41210) 



buffer_load_dwordx4 v128, s[48:51], s73 offen offset: 0 lds // (op 41210) Load value direct to lds




v_bfe_u32 v138, v122, 0, 8 // (op 37716) 
v_bfe_u32 v139, v122, 8, 8 // (op 37716) 
v_bfe_u32 v140, v122, 16, 8 // (op 37716) 
v_bfe_u32 v141, v122, 24, 8 // (op 37716) 
v_permlane16_swap_b32 v138, v139 // (op 37716) 
s_nop 0
v_permlane16_swap_b32 v140, v141 // (op 37716) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v138, v140 // (op 37716) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v139, v141 // (op 37716) 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 42743) 

s_lshl_b64 s[72:73], s[30:31], 6 // (op 41225) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 41225) 

v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 41225) 


v_mov_b64 v[128:129], v[20:21] // (op 41225) 
s_add_u32 s72, 65536, s22 // (op 41225) 
s_mov_b32 m0, s72 // (op 41225) reg expression
s_mov_b32 s73, 384 // (op 41225) call()
v_bfe_u32 v146, v110, 0, 8 // (op 38106) 
v_bfe_u32 v147, v110, 8, 8 // (op 38106) 
v_bfe_u32 v148, v110, 16, 8 // (op 38106) 
v_bfe_u32 v149, v110, 24, 8 // (op 38106) 
buffer_load_dwordx4 v128, s[60:63], s73 offen offset: 0 lds // (op 41225) Load value direct to lds
v_permlane16_swap_b32 v146, v147 // (op 38106) 
v_permlane16_swap_b32 v148, v149 // (op 38106) 
v_add_co_u32 v128, s[74:75], v128, v16 // (op 41225) least significant half
v_addc_co_u32 v129, s[74:75], 0, v129, s[74:75] // (op 41225) most significant half

s_add_u32 m0, m0, 4096 // (op 41225) 
v_permlane32_swap_b32 v146, v148 // (op 38106) 
v_permlane32_swap_b32 v147, v149 // (op 38106) 
buffer_load_dwordx4 v128, s[60:63], s73 offen offset: 0 lds // (op 41225) Load value direct to lds
v_add_co_u32 v128, s[74:75], v128, v16 // (op 41225) least significant half
s_waitcnt lgkmcnt(7)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[76:79], v[44:47], a[0:3], v138, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 249) 
v_addc_co_u32 v129, s[74:75], 0, v129, s[74:75] // (op 41225) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[76:79], v[48:51], a[4:7], v138, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 423) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[76:79], v[52:55], a[8:11], v138, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 649) 
s_add_u32 m0, m0, 4096 // (op 41225) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[76:79], v[56:59], a[12:15], v138, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 875) 
v_bfe_u32 v150, v114, 0, 8 // (op 38174) 
v_bfe_u32 v151, v114, 8, 8 // (op 38174) 
v_bfe_u32 v152, v114, 16, 8 // (op 38174) 
v_bfe_u32 v153, v114, 24, 8 // (op 38174) 
v_permlane16_swap_b32 v150, v151 // (op 38174) 
s_nop 0
v_permlane16_swap_b32 v152, v153 // (op 38174) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v150, v152 // (op 38174) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v151, v153 // (op 38174) 
buffer_load_dwordx4 v128, s[60:63], s73 offen offset: 0 lds // (op 41225) Load value direct to lds
v_add_co_u32 v128, s[74:75], v128, v16 // (op 41225) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[76:79], v[60:63], a[16:19], v138, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1101) 
v_addc_co_u32 v129, s[74:75], 0, v129, s[74:75] // (op 41225) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[76:79], v[64:67], a[20:23], v138, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1327) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[76:79], v[68:71], a[24:27], v138, v152 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1553) 
s_add_u32 m0, m0, 4096 // (op 41225) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[76:79], v[72:75], a[28:31], v138, v153 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1779) 
v_bfe_u32 v154, v118, 0, 8 // (op 37782) 
v_bfe_u32 v155, v118, 8, 8 // (op 37782) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[84:87], v[44:47], a[32:35], v139, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2999) 
v_bfe_u32 v156, v118, 16, 8 // (op 37782) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[84:87], v[48:51], a[36:39], v139, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2923) 
buffer_load_dwordx4 v128, s[60:63], s73 offen offset: 0 lds // (op 41225) Load value direct to lds




v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[84:87], v[52:55], a[40:43], v139, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2847) 
v_bfe_u32 v157, v118, 24, 8 // (op 37782) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[84:87], v[56:59], a[44:47], v139, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2771) 
v_permlane16_swap_b32 v154, v155 // (op 37782) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[84:87], v[60:63], a[48:51], v139, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2695) 
v_permlane16_swap_b32 v156, v157 // (op 37782) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[84:87], v[64:67], a[52:55], v139, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2619) 
s_nop 0
v_permlane32_swap_b32 v154, v156 // (op 37782) Wait state hazard: VALU Write Readlane Hazard
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[84:87], v[68:71], a[56:59], v139, v152 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2543) 
v_permlane32_swap_b32 v155, v157 // (op 37782) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[84:87], v[72:75], a[60:63], v139, v153 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2467) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 42737) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[88:91], v[44:47], a[64:67], v140, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5001) 
s_lshl_b64 s[72:73], s[26:27], 6 // (op 41220) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[88:91], v[48:51], a[68:71], v140, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4925) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 41220) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[88:91], v[52:55], a[72:75], v140, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4849) 
v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 41220) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[88:91], v[56:59], a[76:79], v140, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4773) 
v_mov_b64 v[128:129], v[26:27] // (op 41220) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[88:91], v[60:63], a[80:83], v140, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4697) 
s_add_u32 s72, 81920, s19 // (op 41220) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[88:91], v[64:67], a[84:87], v140, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4621) 
s_mov_b32 m0, s72 // (op 41220) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[88:91], v[68:71], a[88:91], v140, v152 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4545) 
s_mov_b32 s73, 384 // (op 41220) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[88:91], v[72:75], a[92:95], v140, v153 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4469) 
s_waitcnt vmcnt(28)
s_barrier  // (op 42753) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[92:95], v[44:47], a[96:99], v141, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7003) 
v_add_u32 v14, 98304, v10 // (op 272) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[92:95], v[48:51], a[100:103], v141, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6927) 
ds_read_b128 v[96:99], v14 // (op 272) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[92:95], v[52:55], a[104:107], v141, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6851) 
v_add_u32 v14, 99328, v10 // (op 452) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[92:95], v[56:59], a[108:111], v141, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6775) 
ds_read_b128 v[100:103], v14 // (op 452) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[92:95], v[60:63], a[112:115], v141, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6699) 
v_add_u32 v14, 100352, v10 // (op 678) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[92:95], v[64:67], a[116:119], v141, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6623) 
ds_read_b128 v[104:107], v14 // (op 678) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[92:95], v[68:71], a[120:123], v141, v152 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6547) 
v_add_u32 v14, 101376, v10 // (op 904) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[92:95], v[72:75], a[124:127], v141, v153 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6471) 
ds_read_b128 v[158:161], v14 // (op 904) Load local data 





v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[124:127], v[44:47], a[128:131], v154, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9005) 
v_add_u32 v14, 102400, v10 // (op 1130) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[124:127], v[48:51], a[132:135], v154, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8929) 
ds_read_b128 v[138:141], v14 // (op 1130) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[124:127], v[52:55], a[136:139], v154, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8853) 
v_add_u32 v14, 103424, v10 // (op 1356) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[124:127], v[56:59], a[140:143], v154, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8777) 
ds_read_b128 v[162:165], v14 // (op 1356) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[124:127], v[60:63], a[144:147], v154, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8701) 
v_add_u32 v14, 104448, v10 // (op 1582) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[124:127], v[64:67], a[148:151], v154, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8625) 
ds_read_b128 v[166:169], v14 // (op 1582) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[124:127], v[68:71], a[152:155], v154, v152 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8549) 
v_add_u32 v14, 105472, v10 // (op 1808) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[124:127], v[72:75], a[156:159], v154, v153 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8473) 
ds_read_b128 v[170:173], v14 // (op 1808) Load local data 


v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[130:133], v[44:47], a[160:163], v155, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11007) 
s_waitcnt vmcnt(24)
s_barrier  // (op 42746) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[130:133], v[48:51], a[164:167], v155, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10931) 
v_add_u32 v14, 114688, v7 // (op 270) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[130:133], v[52:55], a[168:171], v155, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10855) 
ds_read_b128 v[76:79], v14 // (op 270) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[130:133], v[56:59], a[172:175], v155, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10779) 
v_add_u32 v14, 115712, v7 // (op 2495) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[130:133], v[60:63], a[176:179], v155, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10703) 
ds_read_b128 v[80:83], v14 // (op 2495) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[130:133], v[64:67], a[180:183], v155, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10627) 
v_add_u32 v14, 116736, v7 // (op 4497) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[130:133], v[68:71], a[184:187], v155, v152 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10551) 
ds_read_b128 v[84:87], v14 // (op 4497) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[130:133], v[72:75], a[188:191], v155, v153 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10475) 
buffer_load_dwordx4 v128, s[48:51], s73 offen offset: 0 lds // (op 41220) Load value direct to lds

v_add_co_u32 v128, s[74:75], v128, v16 // (op 41220) least significant half



v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[134:137], v[44:47], a[192:195], v156, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13009) 
v_addc_co_u32 v129, s[74:75], 0, v129, s[74:75] // (op 41220) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[134:137], v[48:51], a[196:199], v156, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12933) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[134:137], v[52:55], a[200:203], v156, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12857) 
s_add_u32 m0, m0, 4096 // (op 41220) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[134:137], v[56:59], a[204:207], v156, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12781) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[134:137], v[60:63], a[208:211], v156, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12705) 
v_add_u32 v14, 117760, v7 // (op 6499) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[134:137], v[64:67], a[212:215], v156, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12629) 
ds_read_b128 v[88:91], v14 // (op 6499) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[134:137], v[68:71], a[216:219], v156, v152 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12553) 
buffer_load_dwordx4 v128, s[48:51], s73 offen offset: 0 lds // (op 41220) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[134:137], v[72:75], a[220:223], v156, v153 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12477) 
v_add_co_u32 v128, s[74:75], v128, v16 // (op 41220) least significant half

v_addc_co_u32 v129, s[74:75], 0, v129, s[74:75] // (op 41220) most significant half




v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[142:145], v[44:47], a[224:227], v157, v146 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 15011) 
s_add_u32 m0, m0, 4096 // (op 41220) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[142:145], v[48:51], a[228:231], v157, v147 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14935) 
v_add_u32 v14, 118784, v7 // (op 8501) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[142:145], v[52:55], a[232:235], v157, v148 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14859) 
ds_read_b128 v[92:95], v14 // (op 8501) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[142:145], v[56:59], a[236:239], v157, v149 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14783) 
v_add_u32 v14, 119808, v7 // (op 10503) 

ds_read_b128 v[124:127], v14 // (op 10503) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[142:145], v[60:63], a[240:243], v157, v150 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14707) 
v_add_u32 v14, 120832, v7 // (op 12505) 
buffer_load_dwordx4 v128, s[48:51], s73 offen offset: 0 lds // (op 41220) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[142:145], v[64:67], a[244:247], v157, v151 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14631) 
v_add_co_u32 v128, s[74:75], v128, v16 // (op 41220) least significant half
v_addc_co_u32 v129, s[74:75], 0, v129, s[74:75] // (op 41220) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[142:145], v[68:71], a[248:251], v157, v152 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14555) 
s_add_u32 m0, m0, 4096 // (op 41220) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[142:145], v[72:75], a[252:255], v157, v153 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14479) 



ds_read_b128 v[130:133], v14 // (op 12505) Load local data 

buffer_load_dwordx4 v128, s[48:51], s73 offen offset: 0 lds // (op 41220) Load value direct to lds







v_add_u32 v14, 121856, v7 // (op 14507) 
ds_read_b128 v[134:137], v14 // (op 14507) Load local data 

v_bfe_u32 v142, v123, 0, 8 // (op 37733) 
v_bfe_u32 v143, v123, 8, 8 // (op 37733) 
v_bfe_u32 v144, v123, 16, 8 // (op 37733) 
v_bfe_u32 v145, v123, 24, 8 // (op 37733) 
v_permlane16_swap_b32 v142, v143 // (op 37733) 
s_nop 0
v_permlane16_swap_b32 v144, v145 // (op 37733) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v142, v144 // (op 37733) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v143, v145 // (op 37733) 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 42756) 


s_lshl_b64 s[72:73], s[30:31], 6 // (op 41235) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 41235) 

v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 41235) 


v_mov_b64 v[128:129], v[20:21] // (op 41235) 
s_add_u32 s72, 98304, s24 // (op 41235) 
s_mov_b32 m0, s72 // (op 41235) reg expression
s_mov_b32 s73, 448 // (op 41235) call()
v_bfe_u32 v120, v111, 0, 8 // (op 38124) 
v_bfe_u32 v121, v111, 8, 8 // (op 38124) 
v_bfe_u32 v122, v111, 16, 8 // (op 38124) 
v_bfe_u32 v123, v111, 24, 8 // (op 38124) 
v_permlane16_swap_b32 v120, v121 // (op 38124) 
s_nop 0
v_permlane16_swap_b32 v122, v123 // (op 38124) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v120, v122 // (op 38124) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v121, v123 // (op 38124) 
buffer_load_dwordx4 v128, s[60:63], s73 offen offset: 0 lds // (op 41235) Load value direct to lds

v_add_co_u32 v128, s[74:75], v128, v16 // (op 41235) least significant half
s_waitcnt lgkmcnt(7)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[76:79], v[96:99], a[0:3], v142, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 277) 
v_addc_co_u32 v129, s[74:75], 0, v129, s[74:75] // (op 41235) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[76:79], v[100:103], a[4:7], v142, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 424) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[76:79], v[104:107], a[8:11], v142, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 650) 
s_add_u32 m0, m0, 4096 // (op 41235) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[76:79], v[158:161], a[12:15], v142, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 876) 
v_bfe_u32 v108, v115, 0, 8 // (op 38192) 
v_bfe_u32 v109, v115, 8, 8 // (op 38192) 
v_bfe_u32 v110, v115, 16, 8 // (op 38192) 
v_bfe_u32 v111, v115, 24, 8 // (op 38192) 
v_permlane16_swap_b32 v108, v109 // (op 38192) 
s_nop 0
v_permlane16_swap_b32 v110, v111 // (op 38192) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v108, v110 // (op 38192) Wait state hazard: VALU Write Readlane Hazard
buffer_load_dwordx4 v128, s[60:63], s73 offen offset: 0 lds // (op 41235) Load value direct to lds
v_permlane32_swap_b32 v109, v111 // (op 38192) 
v_add_co_u32 v128, s[74:75], v128, v16 // (op 41235) least significant half

v_addc_co_u32 v129, s[74:75], 0, v129, s[74:75] // (op 41235) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[76:79], v[138:141], a[16:19], v142, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1102) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[76:79], v[162:165], a[20:23], v142, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1328) 
s_add_u32 m0, m0, 4096 // (op 41235) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[76:79], v[166:169], a[24:27], v142, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1554) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[76:79], v[170:173], a[28:31], v142, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1780) 
v_bfe_u32 v112, v119, 0, 8 // (op 37800) 
v_bfe_u32 v113, v119, 8, 8 // (op 37800) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[80:83], v[96:99], a[32:35], v143, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3000) 
v_bfe_u32 v114, v119, 16, 8 // (op 37800) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[80:83], v[100:103], a[36:39], v143, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2924) 
v_bfe_u32 v115, v119, 24, 8 // (op 37800) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[80:83], v[104:107], a[40:43], v143, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2848) 
v_permlane16_swap_b32 v112, v113 // (op 37800) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[80:83], v[158:161], a[44:47], v143, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2772) 
v_permlane16_swap_b32 v114, v115 // (op 37800) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[80:83], v[138:141], a[48:51], v143, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2696) 
s_nop 0
v_permlane32_swap_b32 v112, v114 // (op 37800) Wait state hazard: VALU Write Readlane Hazard
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[80:83], v[162:165], a[52:55], v143, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2620) 
v_permlane32_swap_b32 v113, v115 // (op 37800) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[80:83], v[166:169], a[56:59], v143, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2544) 
s_waitcnt vmcnt(26)
s_barrier  // (op 42711) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[80:83], v[170:173], a[60:63], v143, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2468) 
ds_read_b128 v[44:47], v4 // (op 67) Load local data 



ds_read_b128 v[48:51], v4 offset:1024 // (op 418) Load local data 
s_waitcnt lgkmcnt(7)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[84:87], v[96:99], a[64:67], v144, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5002) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[84:87], v[100:103], a[68:71], v144, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4926) 
ds_read_b128 v[52:55], v4 offset:2048 // (op 644) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[84:87], v[104:107], a[72:75], v144, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4850) 
ds_read_b128 v[56:59], v4 offset:3072 // (op 870) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[84:87], v[158:161], a[76:79], v144, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4774) 
ds_read_b128 v[60:63], v4 offset:4096 // (op 1096) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[84:87], v[138:141], a[80:83], v144, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4698) 
ds_read_b128 v[64:67], v4 offset:5120 // (op 1322) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[84:87], v[162:165], a[84:87], v144, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4622) 
ds_read_b128 v[68:71], v4 offset:6144 // (op 1548) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[84:87], v[166:169], a[88:91], v144, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4546) 
ds_read_b128 v[72:75], v4 offset:7168 // (op 1774) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[84:87], v[170:173], a[92:95], v144, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4470) 
buffer_load_dwordx4 v128, s[60:63], s73 offen offset: 0 lds // (op 41235) Load value direct to lds
v_add_co_u32 v128, s[74:75], v128, v16 // (op 41235) least significant half
s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[88:91], v[96:99], a[96:99], v145, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7004) 
v_addc_co_u32 v129, s[74:75], 0, v129, s[74:75] // (op 41235) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[88:91], v[100:103], a[100:103], v145, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6928) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[88:91], v[104:107], a[104:107], v145, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6852) 
s_add_u32 m0, m0, 4096 // (op 41235) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[88:91], v[158:161], a[108:111], v145, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6776) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[88:91], v[138:141], a[112:115], v145, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6700) 
s_waitcnt vmcnt(23)
s_barrier  // (op 42694) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[88:91], v[162:165], a[116:119], v145, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6624) 
ds_read_b128 v[76:79], v0 offset:16384 // (op 62) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[88:91], v[166:169], a[120:123], v145, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6548) 
buffer_load_dwordx4 v128, s[60:63], s73 offen offset: 0 lds // (op 41235) Load value direct to lds





v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[88:91], v[170:173], a[124:127], v145, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6472) 

v_add_co_u32 v22, vcc, 256, v22 // (op 41437) least significant half



s_waitcnt lgkmcnt(12)
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[92:95], v[96:99], a[128:131], v112, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 9006) 
ds_read_b128 v[80:83], v0 offset:17408 // (op 2461) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[92:95], v[100:103], a[132:135], v112, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8930) 
v_addc_co_u32 v23, vcc, 0, v23, vcc // (op 41437) most significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[92:95], v[104:107], a[136:139], v112, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8854) 
s_waitcnt lgkmcnt(10)
s_barrier  // (op 42750) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[92:95], v[158:161], a[140:143], v112, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8778) 
s_lshl_b64 s[72:73], s[26:27], 6 // (op 41230) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[92:95], v[138:141], a[144:147], v112, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8702) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 41230) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[92:95], v[162:165], a[148:151], v112, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8626) 
v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 41230) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[92:95], v[166:169], a[152:155], v112, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8550) 
v_mov_b64 v[20:21], v[26:27] // (op 41230) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[92:95], v[170:173], a[156:159], v112, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8474) 
s_add_u32 s72, 114688, s23 // (op 41230) 
s_mov_b32 m0, s72 // (op 41230) reg expression



v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[124:127], v[96:99], a[160:163], v113, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 11008) 
ds_read_b128 v[84:87], v0 offset:18432 // (op 4463) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[124:127], v[100:103], a[164:167], v113, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10932) 
s_mov_b32 s73, 448 // (op 41230) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[124:127], v[104:107], a[168:171], v113, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10856) 
buffer_load_dwordx4 v20, s[48:51], s73 offen offset: 0 lds // (op 41230) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[124:127], v[158:161], a[172:175], v113, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10780) 
v_add_co_u32 v20, s[74:75], v20, v16 // (op 41230) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[124:127], v[138:141], a[176:179], v113, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10704) 
v_addc_co_u32 v21, s[74:75], 0, v21, s[74:75] // (op 41230) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[124:127], v[162:165], a[180:183], v113, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10628) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[124:127], v[166:169], a[184:187], v113, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10552) 
s_add_u32 m0, m0, 4096 // (op 41230) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[124:127], v[170:173], a[188:191], v113, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10476) 

buffer_load_dwordx4 v20, s[48:51], s73 offen offset: 0 lds // (op 41230) Load value direct to lds



v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[130:133], v[96:99], a[192:195], v114, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 13010) 
ds_read_b128 v[88:91], v0 offset:19456 // (op 6465) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[130:133], v[100:103], a[196:199], v114, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12934) 
v_add_co_u32 v20, s[74:75], v20, v16 // (op 41230) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[130:133], v[104:107], a[200:203], v114, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12858) 
v_addc_co_u32 v21, s[74:75], 0, v21, s[74:75] // (op 41230) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[130:133], v[158:161], a[204:207], v114, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12782) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[130:133], v[138:141], a[208:211], v114, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12706) 
s_add_u32 m0, m0, 4096 // (op 41230) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[130:133], v[162:165], a[212:215], v114, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12630) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[130:133], v[166:169], a[216:219], v114, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12554) 
buffer_load_dwordx4 v20, s[48:51], s73 offen offset: 0 lds // (op 41230) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[130:133], v[170:173], a[220:223], v114, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12478) 
v_add_co_u32 v20, s[74:75], v20, v16 // (op 41230) least significant half

v_addc_co_u32 v21, s[74:75], 0, v21, s[74:75] // (op 41230) most significant half




v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[134:137], v[96:99], a[224:227], v115, v120 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 15012) 
s_add_u32 m0, m0, 4096 // (op 41230) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[134:137], v[100:103], a[228:231], v115, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14936) 
ds_read_b128 v[92:95], v0 offset:20480 // (op 8467) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[134:137], v[104:107], a[232:235], v115, v122 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14860) 
ds_read_b128 v[96:99], v0 offset:21504 // (op 10469) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[134:137], v[158:161], a[236:239], v115, v123 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14784) 
ds_read_b128 v[100:103], v0 offset:22528 // (op 12471) Load local data 





v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[134:137], v[138:141], a[240:243], v115, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14708) 
buffer_load_dwordx4 v20, s[48:51], s73 offen offset: 0 lds // (op 41230) Load value direct to lds






v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[134:137], v[162:165], a[244:247], v115, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14632) 
v_add_co_u32 v18, vcc, 256, v18 // (op 41373) least significant half

v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 41373) most significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[134:137], v[166:169], a[248:251], v115, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14556) 
ds_read_b128 v[104:107], v0 offset:23552 // (op 14473) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[134:137], v[170:173], a[252:255], v115, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14480) 




s_ashr_i64 s[72:73], s[42:43], 7 // (op 12279) 
s_mov_b32 s74, s72 // (op 12279) convert

s_and_b32 s72, s74, 4294967292 // (op 12279) 

s_cmp_lt_u32 s25, s72 // (op 12279) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WGM__ForLoopTop_KLoop_12279 // (op 12279) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WGM__ForLoopTop_KLoop_12279 if true)
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WGM__ForLoopBottom_KLoop_12279:
















s_barrier  // (op 42849) 
s_mul_hi_i32 s47, s16, s10 // (op 41273) 
v_lshrrev_b32 v14, 2, v1 // (op 41273) 
s_add_i32 s0, s47, s16 // (op 41273) 

v_and_b32 v15, -4, v1 // (op 41273) 
s_ashr_i32 s1, s0, 31 // (op 41273) 
v_sub_u32 v16, v1, v15 // (op 41273) 

s_cmp_eq_i32 s10, 0 // (op 41273) 
s_mov_b32 s47, scc // (op 41273) 
v_lshlrev_b32 v15, 5, v16 // (op 41273) 

s_cmp_eq_i32 s47, 0 // (op 41273) 
s_cselect_b32 s18, 0, -1 // (op 41273) 

s_and_b32 s19, s13, 31 // (op 41273) 
s_lshl_b32 s47, 1, s19 // (op 41273) 
s_add_i32 s22, s47, s18 // (op 41273) 


s_and_b32 s18, s1, s22 // (op 41273) 


s_add_i32 s1, s0, s18 // (op 41273) 


s_ashr_i32 s0, s1, s19 // (op 41273) 


s_lshl_b32 s1, s13, 24 // (op 41273) 
s_ashr_i32 s47, s1, 31 // (op 41273) 

s_xor_b32 s1, s0, s47 // (op 41273) 

s_sub_i32 s0, s1, s47 // (op 41273) 


s_mul_hi_i32 s1, s0, s14 // (op 41273) 
s_add_i32 s47, s1, s0 // (op 41273) 

s_ashr_i32 s1, s47, 31 // (op 41273) 
s_cmp_eq_i32 s14, 0 // (op 41273) 
s_mov_b32 s18, scc // (op 41273) 
s_cmp_eq_i32 s18, 0 // (op 41273) 
s_cselect_b32 s19, 0, -1 // (op 41273) 

s_and_b32 s18, s15, 31 // (op 41273) 
s_lshl_b32 s22, 1, s18 // (op 41273) 
s_add_i32 s23, s22, s19 // (op 41273) 


s_and_b32 s19, s1, s23 // (op 41273) 


s_add_i32 s1, s47, s19 // (op 41273) 


s_ashr_i32 s19, s1, s18 // (op 41273) 


s_lshl_b32 s1, s15, 24 // (op 41273) 
s_ashr_i32 s18, s1, 31 // (op 41273) 

s_xor_b32 s1, s19, s18 // (op 41273) 

s_sub_i32 s19, s1, s18 // (op 41273) 


s_mul_i32 s1, s0, s11 // (op 41273) 

s_sub_u32 s0, s16, s1 // (op 41273) 

s_mul_i32 s1, s19, s11 // (op 41273) 
s_add_u32 s1, s1, s0 // (op 41273) 


s_mul_hi_i32 s0, s16, s9 // (op 41273) 
s_add_i32 s47, s0, s16 // (op 41273) 

s_ashr_i32 s0, s47, 31 // (op 41273) 
s_cmp_eq_i32 s9, 0 // (op 41273) 
s_mov_b32 s18, scc // (op 41273) 
s_cmp_eq_i32 s18, 0 // (op 41273) 
s_cselect_b32 s19, 0, -1 // (op 41273) 

s_and_b32 s18, s45, 31 // (op 41273) 
s_lshl_b32 s22, 1, s18 // (op 41273) 
s_add_i32 s23, s22, s19 // (op 41273) 


s_and_b32 s19, s0, s23 // (op 41273) 


s_add_i32 s0, s47, s19 // (op 41273) 


s_ashr_i32 s19, s0, s18 // (op 41273) 


s_lshl_b32 s0, s45, 24 // (op 41273) 
s_ashr_i32 s18, s0, 31 // (op 41273) 

s_xor_b32 s0, s19, s18 // (op 41273) 

s_sub_i32 s19, s0, s18 // (op 41273) 


s_add_u32 s22, s6, 256 // (op 41273) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 41273) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 41273) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 41273) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 41273) 

s_mul_i32 s0, s11, s22 // (op 41273) 

s_mul_i32 s18, s8, s0 // (op 41273) 

s_mul_i32 s0, s19, s18 // (op 41273) 

s_sub_u32 s18, s16, s0 // (op 41273) 

s_mov_b32 s0, s18 // (op 41273) 
s_mul_hi_i32 s47, s0, s12 // (op 41273) 
s_add_i32 s22, s47, s0 // (op 41273) 


s_ashr_i32 s0, s22, 31 // (op 41273) 
s_cmp_eq_i32 s12, 0 // (op 41273) 
s_mov_b32 s47, scc // (op 41273) 
s_cmp_eq_i32 s47, 0 // (op 41273) 
s_cselect_b32 s23, 0, -1 // (op 41273) 

s_and_b32 s47, s44, 31 // (op 41273) 
s_lshl_b32 s24, 1, s47 // (op 41273) 
s_add_i32 s25, s24, s23 // (op 41273) 


s_and_b32 s23, s0, s25 // (op 41273) 


s_add_i32 s0, s22, s23 // (op 41273) 


s_ashr_i32 s22, s0, s47 // (op 41273) 


s_lshl_b32 s0, s44, 24 // (op 41273) 
s_ashr_i32 s47, s0, 31 // (op 41273) 

s_xor_b32 s0, s22, s47 // (op 41273) 

s_sub_i32 s22, s0, s47 // (op 41273) 


s_mul_i32 s0, s46, s11 // (op 41273) 
s_add_u32 s24, s20, 256 // (op 41273) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 41273) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 41273) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 41273) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 8 // (op 41273) 

s_sub_i32 s23, s24, s0 // (op 41273) 

s_mul_i32 s47, s22, s23 // (op 41273) 


s_sub_u32 s22, s18, s47 // (op 41273) 


s_add_u32 s18, s22, s0 // (op 41273) 


s_cmp_eq_i32 s19, 0 // (op 41273) 
s_mov_b32 s0, scc // (op 41273) 

s_cmp_eq_i32 s0, 0 // (op 41273) 
s_cselect_b32 s19, s18, s1 // (op 41273) 



v_lshl_add_u32 v16, s19, 8, v14 // (op 41273) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 41273) 
s_lshr_b32 s47, s0, 2 // (op 41273) 

v_lshl_add_u32 v14, s47, 9, v15 // (op 41273) 


v_mov_b32 v27, 0 // (op 41273) convert
v_mov_b32 v26, v14 // (op 41273) convert

v_mul_lo_u32 v15, v16, s27 // (op 41273) most significant: low of low * high
v_mul_hi_u32 v17, v16, s26 // (op 41273) most significant: high of low * low
v_mul_lo_u32 v14, v16, s26 // (op 41273) least significant: low of low * low
v_add_u32 v15, v15, v17 // (op 41273) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v26 // (op 41273) least significant half
v_addc_co_u32 v15, s[0:1], v15, v27, s[0:1] // (op 41273) most significant half



v_lshlrev_b64 v[16:17], 2, v[14:15] // (op 41273) 

v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 41273) 

s_mul_i32 s1, s26, s21 // (op 41275) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 41275) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 41275) least significant
s_mul_hi_u32 s18, s26, s20 // (op 41275) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 41275) most significant: sum
s_add_u32 s1, s1, s18 // (op 41275) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 41275) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 41275) 

s_mov_b32 s48, s28 // (op 41275) reg expression
s_mov_b32 s49, s29 // (op 41275) reg expression
s_mov_b32 s50, s0 // (op 41275) reg expression
s_mov_b32 s51, 131072 // (op 41275) call()

s_mul_hi_i32 s47, s16, s10 // (op 41403) 
v_lshrrev_b32 v16, 2, v1 // (op 41403) 
s_add_i32 s0, s47, s16 // (op 41403) 

v_and_b32 v17, -4, v1 // (op 41403) 
s_ashr_i32 s1, s0, 31 // (op 41403) 
v_sub_u32 v18, v1, v17 // (op 41403) 

s_cmp_eq_i32 s10, 0 // (op 41403) 
s_mov_b32 s47, scc // (op 41403) 
v_lshlrev_b32 v17, 5, v18 // (op 41403) 

s_cmp_eq_i32 s47, 0 // (op 41403) 
s_cselect_b32 s18, 0, -1 // (op 41403) 

s_and_b32 s19, s13, 31 // (op 41403) 
s_lshl_b32 s47, 1, s19 // (op 41403) 
s_add_i32 s22, s47, s18 // (op 41403) 


s_and_b32 s18, s1, s22 // (op 41403) 


s_add_i32 s1, s0, s18 // (op 41403) 


s_ashr_i32 s0, s1, s19 // (op 41403) 


s_lshl_b32 s1, s13, 24 // (op 41403) 
s_ashr_i32 s47, s1, 31 // (op 41403) 

s_xor_b32 s1, s0, s47 // (op 41403) 

s_sub_i32 s0, s1, s47 // (op 41403) 


s_mul_hi_i32 s1, s0, s14 // (op 41403) 
s_add_i32 s47, s1, s0 // (op 41403) 

s_ashr_i32 s1, s47, 31 // (op 41403) 
s_cmp_eq_i32 s14, 0 // (op 41403) 
s_mov_b32 s18, scc // (op 41403) 
s_cmp_eq_i32 s18, 0 // (op 41403) 
s_cselect_b32 s19, 0, -1 // (op 41403) 

s_and_b32 s18, s15, 31 // (op 41403) 
s_lshl_b32 s22, 1, s18 // (op 41403) 
s_add_i32 s23, s22, s19 // (op 41403) 


s_and_b32 s19, s1, s23 // (op 41403) 


s_add_i32 s1, s47, s19 // (op 41403) 


s_ashr_i32 s19, s1, s18 // (op 41403) 


s_lshl_b32 s1, s15, 24 // (op 41403) 
s_ashr_i32 s18, s1, 31 // (op 41403) 

s_xor_b32 s1, s19, s18 // (op 41403) 

s_sub_i32 s19, s1, s18 // (op 41403) 


s_add_u32 s22, s6, 256 // (op 41403) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 41403) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 41403) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 41403) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 41403) 

s_mul_i32 s1, s19, s22 // (op 41403) 

s_sub_i32 s47, s0, s1 // (op 41403) 


s_mul_hi_i32 s0, s16, s9 // (op 41403) 
s_add_i32 s1, s0, s16 // (op 41403) 

s_ashr_i32 s0, s1, 31 // (op 41403) 
s_cmp_eq_i32 s9, 0 // (op 41403) 
s_mov_b32 s18, scc // (op 41403) 
s_cmp_eq_i32 s18, 0 // (op 41403) 
s_cselect_b32 s19, 0, -1 // (op 41403) 

s_and_b32 s18, s45, 31 // (op 41403) 
s_lshl_b32 s24, 1, s18 // (op 41403) 
s_add_i32 s25, s24, s19 // (op 41403) 


s_and_b32 s19, s0, s25 // (op 41403) 


s_add_i32 s0, s1, s19 // (op 41403) 


s_ashr_i32 s1, s0, s18 // (op 41403) 


s_lshl_b32 s0, s45, 24 // (op 41403) 
s_ashr_i32 s18, s0, 31 // (op 41403) 

s_xor_b32 s0, s1, s18 // (op 41403) 

s_sub_i32 s1, s0, s18 // (op 41403) 


s_mul_i32 s0, s11, s22 // (op 41403) 

s_mul_i32 s18, s8, s0 // (op 41403) 

s_mul_i32 s0, s1, s18 // (op 41403) 

s_sub_u32 s18, s16, s0 // (op 41403) 

s_mul_hi_i32 s0, s18, s12 // (op 41403) 
s_add_i32 s19, s0, s18 // (op 41403) 


s_ashr_i32 s0, s19, 31 // (op 41403) 
s_cmp_eq_i32 s12, 0 // (op 41403) 
s_mov_b32 s18, scc // (op 41403) 
s_cmp_eq_i32 s18, 0 // (op 41403) 
s_cselect_b32 s22, 0, -1 // (op 41403) 

s_and_b32 s18, s44, 31 // (op 41403) 
s_lshl_b32 s23, 1, s18 // (op 41403) 
s_add_i32 s24, s23, s22 // (op 41403) 


s_and_b32 s25, s0, s24 // (op 41403) 


s_add_i32 s0, s19, s25 // (op 41403) 


s_ashr_i32 s19, s0, s18 // (op 41403) 


s_lshl_b32 s0, s44, 24 // (op 41403) 
s_ashr_i32 s18, s0, 31 // (op 41403) 

s_xor_b32 s0, s19, s18 // (op 41403) 

s_sub_i32 s19, s0, s18 // (op 41403) 


s_cmp_eq_i32 s1, 0 // (op 41403) 
s_mov_b32 s0, scc // (op 41403) 

s_cmp_eq_i32 s0, 0 // (op 41403) 
s_cselect_b32 s1, s19, s47 // (op 41403) 



v_lshl_add_u32 v18, s1, 8, v16 // (op 41403) 


v_mul_lo_u32 v27, v18, s31 // (op 41403) most significant: low of low * high
v_mul_hi_u32 v16, v18, s30 // (op 41403) most significant: high of low * low
v_mul_lo_u32 v26, v18, s30 // (op 41403) least significant: low of low * low
v_add_u32 v27, v27, v16 // (op 41403) most significant: sum


s_ashr_i64 s[0:1], s[42:43], 7 // (op 41403) 
s_lshr_b32 s47, s0, 2 // (op 41403) 

v_lshl_add_u32 v16, s47, 9, v17 // (op 41403) 


v_mov_b32 v25, 0 // (op 41403) convert
v_mov_b32 v24, v16 // (op 41403) convert

v_add_co_u32 v16, s[0:1], v24, v26 // (op 41403) least significant half
v_addc_co_u32 v17, s[0:1], v25, v27, s[0:1] // (op 41403) most significant half

v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 41403) 


v_lshrrev_b64 v[18:19], 3, v[16:17] // (op 41403) 

s_mul_i32 s1, s30, s7 // (op 41405) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 41405) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 41405) least significant
s_mul_hi_u32 s18, s30, s6 // (op 41405) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 41405) most significant: sum
s_add_u32 s1, s1, s18 // (op 41405) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 41405) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 41405) 

s_mov_b32 s60, s32 // (op 41405) reg expression
s_mov_b32 s61, s33 // (op 41405) reg expression
s_mov_b32 s62, s0 // (op 41405) reg expression
s_mov_b32 s63, 131072 // (op 41405) call()

v_and_b32 v16, -4, v1 // (op 41501) 
v_sub_u32 v17, v1, v16 // (op 41501) 

v_lshlrev_b32 v16, 5, v17 // (op 41501) 

v_lshrrev_b32 v17, 2, v1 // (op 41501) 
v_lshl_add_u32 v20, v17, 7, v16 // (op 41501) 


v_lshlrev_b32 v16, 2, v20 // (op 41501) 

v_lshrrev_b32 v17, 3, v16 // (op 41501) 

s_nop 0
v_readfirstlane_b32 s47, v17 // (op 41501) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -4, v1 // (op 41558) 
v_sub_u32 v17, v1, v16 // (op 41558) 

v_lshlrev_b32 v16, 5, v17 // (op 41558) 

v_lshrrev_b32 v17, 2, v1 // (op 41558) 
v_lshl_add_u32 v20, v17, 7, v16 // (op 41558) 


v_lshlrev_b32 v16, 2, v20 // (op 41558) 

v_lshrrev_b32 v17, 3, v16 // (op 41558) 

s_nop 0
v_readfirstlane_b32 s0, v17 // (op 41558) Wait state hazard: VALU Write Readlane Hazard

s_mul_hi_i32 s1, s16, s10 // (op 41999) 
v_and_b32 v16, 63, v1 // (op 41999) 
s_add_i32 s18, s1, s16 // (op 41999) 

v_lshrrev_b32 v17, 4, v16 // (op 41999) 
s_ashr_i32 s1, s18, 31 // (op 41999) 
v_and_b32 v20, 3, v17 // (op 41999) 

s_cmp_eq_i32 s10, 0 // (op 41999) 
s_mov_b32 s19, scc // (op 41999) 
v_and_b32 v17, 15, v1 // (op 41999) 
s_cmp_eq_i32 s19, 0 // (op 41999) 
s_cselect_b32 s22, 0, -1 // (op 41999) 

v_lshl_add_u32 v21, v20, 4, v17 // (op 41999) 


s_and_b32 s19, s13, 31 // (op 41999) 
v_and_b32 v17, 4294967168, v1 // (op 41999) 
s_lshl_b32 s23, 1, s19 // (op 41999) 
v_add_u32 v20, v17, v21 // (op 41999) 


s_add_i32 s24, s23, s22 // (op 41999) 


v_lshrrev_b32 v17, 6, v16 // (op 41999) 

s_and_b32 s25, s1, s24 // (op 41999) 


v_lshlrev_b32 v16, 2, v17 // (op 41999) 

s_add_i32 s1, s18, s25 // (op 41999) 


s_ashr_i32 s18, s1, s19 // (op 41999) 


s_lshl_b32 s1, s13, 24 // (op 41999) 
s_ashr_i32 s19, s1, 31 // (op 41999) 

s_xor_b32 s1, s18, s19 // (op 41999) 

s_sub_i32 s18, s1, s19 // (op 41999) 


s_mul_hi_i32 s1, s18, s14 // (op 41999) 
s_add_i32 s19, s1, s18 // (op 41999) 

s_ashr_i32 s1, s19, 31 // (op 41999) 
s_cmp_eq_i32 s14, 0 // (op 41999) 
s_mov_b32 s22, scc // (op 41999) 
s_cmp_eq_i32 s22, 0 // (op 41999) 
s_cselect_b32 s23, 0, -1 // (op 41999) 

s_and_b32 s22, s15, 31 // (op 41999) 
s_lshl_b32 s24, 1, s22 // (op 41999) 
s_add_i32 s25, s24, s23 // (op 41999) 


s_and_b32 s23, s1, s25 // (op 41999) 


s_add_i32 s1, s19, s23 // (op 41999) 


s_ashr_i32 s19, s1, s22 // (op 41999) 


s_lshl_b32 s1, s15, 24 // (op 41999) 
s_ashr_i32 s22, s1, 31 // (op 41999) 

s_xor_b32 s1, s19, s22 // (op 41999) 

s_sub_i32 s19, s1, s22 // (op 41999) 


s_mul_i32 s1, s18, s11 // (op 41999) 

s_sub_u32 s18, s16, s1 // (op 41999) 

s_mul_i32 s1, s19, s11 // (op 41999) 
s_add_u32 s1, s1, s18 // (op 41999) 


s_mul_hi_i32 s18, s16, s9 // (op 41999) 
s_add_i32 s19, s18, s16 // (op 41999) 

s_ashr_i32 s18, s19, 31 // (op 41999) 
s_cmp_eq_i32 s9, 0 // (op 41999) 
s_mov_b32 s22, scc // (op 41999) 
s_cmp_eq_i32 s22, 0 // (op 41999) 
s_cselect_b32 s23, 0, -1 // (op 41999) 

s_and_b32 s22, s45, 31 // (op 41999) 
s_lshl_b32 s24, 1, s22 // (op 41999) 
s_add_i32 s25, s24, s23 // (op 41999) 


s_and_b32 s23, s18, s25 // (op 41999) 


s_add_i32 s18, s19, s23 // (op 41999) 


s_ashr_i32 s19, s18, s22 // (op 41999) 


s_lshl_b32 s18, s45, 24 // (op 41999) 
s_ashr_i32 s22, s18, 31 // (op 41999) 

s_xor_b32 s18, s19, s22 // (op 41999) 

s_sub_i32 s19, s18, s22 // (op 41999) 


s_add_u32 s22, s6, 256 // (op 41999) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 41999) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 41999) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 41999) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 41999) 

s_mul_i32 s18, s11, s22 // (op 41999) 

s_mul_i32 s22, s8, s18 // (op 41999) 

s_mul_i32 s18, s19, s22 // (op 41999) 

s_sub_u32 s22, s16, s18 // (op 41999) 

s_mov_b32 s18, s22 // (op 41999) 
s_mul_hi_i32 s23, s18, s12 // (op 41999) 
s_add_i32 s24, s23, s18 // (op 41999) 


s_ashr_i32 s18, s24, 31 // (op 41999) 
s_cmp_eq_i32 s12, 0 // (op 41999) 
s_mov_b32 s23, scc // (op 41999) 
s_cmp_eq_i32 s23, 0 // (op 41999) 
s_cselect_b32 s25, 0, -1 // (op 41999) 

s_and_b32 s23, s44, 31 // (op 41999) 
s_lshl_b32 s64, 1, s23 // (op 41999) 
s_add_i32 s65, s64, s25 // (op 41999) 


s_and_b32 s25, s18, s65 // (op 41999) 


s_add_i32 s18, s24, s25 // (op 41999) 


s_ashr_i32 s24, s18, s23 // (op 41999) 


s_lshl_b32 s18, s44, 24 // (op 41999) 
s_ashr_i32 s23, s18, 31 // (op 41999) 

s_xor_b32 s18, s24, s23 // (op 41999) 

s_sub_i32 s24, s18, s23 // (op 41999) 


s_mul_i32 s18, s46, s11 // (op 41999) 
s_add_u32 s64, s20, 256 // (op 41999) Start of Int64 add, locking SCC
s_addc_u32 s65, s21, 0 // (op 41999) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 41999) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 41999) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 8 // (op 41999) 

s_sub_i32 s23, s64, s18 // (op 41999) 

s_mul_i32 s25, s24, s23 // (op 41999) 


s_sub_u32 s23, s22, s25 // (op 41999) 


s_add_u32 s22, s23, s18 // (op 41999) 


s_cmp_eq_i32 s19, 0 // (op 41999) 
s_mov_b32 s18, scc // (op 41999) 

s_cmp_eq_i32 s18, 0 // (op 41999) 
s_cselect_b32 s19, s22, s1 // (op 41999) 



v_lshl_add_u32 v17, s19, 8, v20 // (op 41999) 


s_ashr_i64 s[18:19], s[42:43], 7 // (op 41999) 
s_lshr_b32 s1, s18, 2 // (op 41999) 

v_lshl_add_u32 v20, s1, 4, v16 // (op 41999) 


v_mul_lo_u32 v16, v17, s34 // (op 41999) 
v_add_u32 v16, v16, v20 // (op 41999) 


s_mul_i32 s19, s34, s21 // (op 42001) most significant: low of low * high
s_mul_i32 s1, s35, s20 // (op 42001) most significant: low of high * low
s_mul_i32 s18, s34, s20 // (op 42001) least significant
s_mul_hi_u32 s22, s34, s20 // (op 42001) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 42001) most significant: sum
s_add_u32 s19, s19, s22 // (op 42001) most significant: sum


s_mov_b32 s64, s36 // (op 42001) reg expression
s_mov_b32 s65, s37 // (op 42001) reg expression
s_mov_b32 s66, s18 // (op 42001) reg expression
s_mov_b32 s67, 131072 // (op 42001) call()

s_mul_hi_i32 s1, s16, s10 // (op 42026) 
v_and_b32 v17, 63, v1 // (op 42026) 
s_add_i32 s18, s1, s16 // (op 42026) 

v_lshrrev_b32 v20, 4, v17 // (op 42026) 
s_ashr_i32 s1, s18, 31 // (op 42026) 
v_and_b32 v21, 3, v20 // (op 42026) 

s_cmp_eq_i32 s10, 0 // (op 42026) 
s_mov_b32 s19, scc // (op 42026) 
v_and_b32 v20, 15, v1 // (op 42026) 
s_cmp_eq_i32 s19, 0 // (op 42026) 
s_cselect_b32 s22, 0, -1 // (op 42026) 

v_lshl_add_u32 v22, v21, 4, v20 // (op 42026) 


s_and_b32 s19, s13, 31 // (op 42026) 
v_lshrrev_b32 v20, 6, v1 // (op 42026) 
s_lshl_b32 s23, 1, s19 // (op 42026) 
v_and_b32 v21, 1, v20 // (op 42026) 

s_add_i32 s24, s23, s22 // (op 42026) 


v_lshl_add_u32 v20, v21, 7, v22 // (op 42026) 


s_and_b32 s25, s1, s24 // (op 42026) 


v_lshrrev_b32 v21, 6, v17 // (op 42026) 

s_add_i32 s1, s18, s25 // (op 42026) 


v_lshlrev_b32 v17, 2, v21 // (op 42026) 

s_ashr_i32 s18, s1, s19 // (op 42026) 


s_lshl_b32 s1, s13, 24 // (op 42026) 
s_ashr_i32 s19, s1, 31 // (op 42026) 

s_xor_b32 s1, s18, s19 // (op 42026) 

s_sub_i32 s18, s1, s19 // (op 42026) 


s_mul_hi_i32 s1, s18, s14 // (op 42026) 
s_add_i32 s19, s1, s18 // (op 42026) 

s_ashr_i32 s1, s19, 31 // (op 42026) 
s_cmp_eq_i32 s14, 0 // (op 42026) 
s_mov_b32 s22, scc // (op 42026) 
s_cmp_eq_i32 s22, 0 // (op 42026) 
s_cselect_b32 s23, 0, -1 // (op 42026) 

s_and_b32 s22, s15, 31 // (op 42026) 
s_lshl_b32 s24, 1, s22 // (op 42026) 
s_add_i32 s25, s24, s23 // (op 42026) 


s_and_b32 s23, s1, s25 // (op 42026) 


s_add_i32 s1, s19, s23 // (op 42026) 


s_ashr_i32 s19, s1, s22 // (op 42026) 


s_lshl_b32 s1, s15, 24 // (op 42026) 
s_ashr_i32 s22, s1, 31 // (op 42026) 

s_xor_b32 s1, s19, s22 // (op 42026) 

s_sub_i32 s19, s1, s22 // (op 42026) 


s_add_u32 s22, s6, 256 // (op 42026) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 42026) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 42026) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 42026) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 42026) 

s_mul_i32 s1, s19, s22 // (op 42026) 

s_sub_i32 s19, s18, s1 // (op 42026) 


s_mul_hi_i32 s1, s16, s9 // (op 42026) 
s_add_i32 s18, s1, s16 // (op 42026) 

s_ashr_i32 s1, s18, 31 // (op 42026) 
s_cmp_eq_i32 s9, 0 // (op 42026) 
s_mov_b32 s24, scc // (op 42026) 
s_cmp_eq_i32 s24, 0 // (op 42026) 
s_cselect_b32 s25, 0, -1 // (op 42026) 

s_and_b32 s24, s45, 31 // (op 42026) 
s_lshl_b32 s68, 1, s24 // (op 42026) 
s_add_i32 s69, s68, s25 // (op 42026) 


s_and_b32 s25, s1, s69 // (op 42026) 


s_add_i32 s1, s18, s25 // (op 42026) 


s_ashr_i32 s18, s1, s24 // (op 42026) 


s_lshl_b32 s1, s45, 24 // (op 42026) 
s_ashr_i32 s24, s1, 31 // (op 42026) 

s_xor_b32 s1, s18, s24 // (op 42026) 

s_sub_i32 s18, s1, s24 // (op 42026) 


s_mul_i32 s1, s11, s22 // (op 42026) 

s_mul_i32 s22, s8, s1 // (op 42026) 

s_mul_i32 s1, s18, s22 // (op 42026) 

s_sub_u32 s22, s16, s1 // (op 42026) 

s_mul_hi_i32 s1, s22, s12 // (op 42026) 
s_add_i32 s23, s1, s22 // (op 42026) 


s_ashr_i32 s1, s23, 31 // (op 42026) 
s_cmp_eq_i32 s12, 0 // (op 42026) 
s_mov_b32 s22, scc // (op 42026) 
s_cmp_eq_i32 s22, 0 // (op 42026) 
s_cselect_b32 s24, 0, -1 // (op 42026) 

s_and_b32 s22, s44, 31 // (op 42026) 
s_lshl_b32 s25, 1, s22 // (op 42026) 
s_add_i32 s68, s25, s24 // (op 42026) 


s_and_b32 s24, s1, s68 // (op 42026) 


s_add_i32 s1, s23, s24 // (op 42026) 


s_ashr_i32 s23, s1, s22 // (op 42026) 


s_lshl_b32 s1, s44, 24 // (op 42026) 
s_ashr_i32 s22, s1, 31 // (op 42026) 

s_xor_b32 s1, s23, s22 // (op 42026) 

s_sub_i32 s23, s1, s22 // (op 42026) 


s_cmp_eq_i32 s18, 0 // (op 42026) 
s_mov_b32 s1, scc // (op 42026) 

s_cmp_eq_i32 s1, 0 // (op 42026) 
s_cselect_b32 s18, s23, s19 // (op 42026) 



v_lshl_add_u32 v21, s18, 8, v20 // (op 42026) 


v_mul_lo_u32 v20, v21, s38 // (op 42026) 

s_ashr_i64 s[18:19], s[42:43], 7 // (op 42026) 
s_lshr_b32 s1, s18, 2 // (op 42026) 

v_lshl_add_u32 v21, s1, 4, v17 // (op 42026) 


v_add_u32 v17, v21, v20 // (op 42026) 


s_mul_i32 s19, s38, s7 // (op 42028) most significant: low of low * high
s_mul_i32 s1, s39, s6 // (op 42028) most significant: low of high * low
s_mul_i32 s18, s38, s6 // (op 42028) least significant
s_mul_hi_u32 s22, s38, s6 // (op 42028) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 42028) most significant: sum
s_add_u32 s19, s19, s22 // (op 42028) most significant: sum


s_mov_b32 s68, s40 // (op 42028) reg expression
s_mov_b32 s69, s41 // (op 42028) reg expression
s_mov_b32 s70, s18 // (op 42028) reg expression
s_mov_b32 s71, 131072 // (op 42028) call()

s_ashr_i64 s[18:19], s[42:43], 7 // (op 14449) 
s_and_b32 s1, s18, 4294967292 // (op 14449) 

s_ashr_i64 s[18:19], s[42:43], 7 // (op 14448) 
s_mov_b32 s22, s18 // (op 14448) convert

s_cmp_lt_u32 s1, s22 // (op 14448) 




















s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WGM__ForLoopBottom_KLoopTail_14448 // (op 14448) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WGM__ForLoopBottom_KLoopTail_14448 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WGM__ForLoopTop_KLoopTail_14448:
s_barrier  // (op 42691) 
s_add_u32 s1, s1, 1 // (op 14450) 

s_lshl_b64 s[18:19], s[26:27], 6 // (op 41198) 
s_lshl_b64 s[22:23], s[18:19], 2 // (op 41198) 

v_lshrrev_b64 v[20:21], 3, s[22:23] // (op 41198) 


v_mov_b64 v[22:23], v[14:15] // (op 41198) 
v_mov_b64 v[24:25], v[22:23] // (op 41198) 
s_add_u32 s18, 16384, s47 // (op 41198) 
s_mov_b32 m0, s18 // (op 41198) reg expression
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 41198) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v20 // (op 41198) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 41198) most significant half

s_add_u32 m0, m0, 4096 // (op 41198) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 41198) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v20 // (op 41198) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 41198) most significant half

s_add_u32 m0, m0, 4096 // (op 41198) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 41198) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v20 // (op 41198) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 41198) most significant half

s_add_u32 m0, m0, 4096 // (op 41198) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 41198) Load value direct to lds




v_add_co_u32 v14, s[18:19], 64, v14 // (op 41263) least significant half
v_addc_co_u32 v15, s[18:19], 0, v15, s[18:19] // (op 41263) most significant half

s_barrier  // (op 42708) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 41204) 
s_lshl_b64 s[22:23], s[18:19], 2 // (op 41204) 

v_lshrrev_b64 v[20:21], 3, s[22:23] // (op 41204) 


v_mov_b64 v[22:23], v[18:19] // (op 41204) 
v_mov_b64 v[24:25], v[22:23] // (op 41204) 
s_mov_b32 m0, s0 // (op 41204) reg expression
buffer_load_dwordx4 v24, s[60:63], 0 offen offset: 0 lds // (op 41204) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v20 // (op 41204) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 41204) most significant half

s_add_u32 m0, m0, 4096 // (op 41204) 
buffer_load_dwordx4 v24, s[60:63], 0 offen offset: 0 lds // (op 41204) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v20 // (op 41204) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 41204) most significant half

s_add_u32 m0, m0, 4096 // (op 41204) 
buffer_load_dwordx4 v24, s[60:63], 0 offen offset: 0 lds // (op 41204) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v20 // (op 41204) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 41204) most significant half

s_add_u32 m0, m0, 4096 // (op 41204) 
buffer_load_dwordx4 v24, s[60:63], 0 offen offset: 0 lds // (op 41204) Load value direct to lds



v_add_co_u32 v18, s[18:19], 64, v18 // (op 41395) least significant half
v_addc_co_u32 v19, s[18:19], 0, v19, s[18:19] // (op 41395) most significant half

s_waitcnt vmcnt(4)
s_barrier  // (op 42698) 



ds_read_b128 v[20:23], v0 offset:16384 // (op 164) Load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 42714) 



ds_read_b128 v[24:27], v4 // (op 166) Load local data 


v_mov_b32 v28, v16 // (op 168) 
v_mov_b32 v29, v28 // (op 168) reg expression

buffer_load_dwordx4 v[30:33], v29, s[64:67], 0 offen // (op 168) Load value



v_mov_b32 v29, v17 // (op 170) 
v_mov_b32 v34, v29 // (op 170) reg expression

buffer_load_dwordx4 v[36:39], v34, s[68:71], 0 offen // (op 170) Load value

s_waitcnt vmcnt(1)
v_bfe_u32 v40, v30, 0, 8 // (op 38465) 



v_bfe_u32 v41, v30, 8, 8 // (op 38465) 
v_bfe_u32 v42, v30, 16, 8 // (op 38465) 
v_bfe_u32 v43, v30, 24, 8 // (op 38465) 
v_permlane16_swap_b32 v40, v41 // (op 38465) 
ds_read_b128 v[44:47], v4 offset:1024 // (op 401) Load local data 
v_permlane16_swap_b32 v42, v43 // (op 38465) 



s_nop 1
v_permlane32_swap_b32 v40, v42 // (op 38465) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v41, v43 // (op 38465) 
ds_read_b128 v[48:51], v4 offset:2048 // (op 627) Load local data 




ds_read_b128 v[30:33], v4 offset:3072 // (op 853) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v52, v36, 0, 8 // (op 38557) 



v_bfe_u32 v53, v36, 8, 8 // (op 38557) 
v_bfe_u32 v54, v36, 16, 8 // (op 38557) 
v_bfe_u32 v55, v36, 24, 8 // (op 38557) 
v_permlane16_swap_b32 v52, v53 // (op 38557) 
s_nop 0
v_permlane16_swap_b32 v54, v55 // (op 38557) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v52, v54 // (op 38557) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v53, v55 // (op 38557) 
ds_read_b128 v[56:59], v4 offset:4096 // (op 1079) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:23], v[24:27], a[0:3], v40, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 171) 
s_lshl_b32 s18, s38, 6 // (op 1084) 
v_add_u32 v34, v29, s18 // (op 1084) 


buffer_load_dwordx4 v[36:39], v34, s[68:71], 0 offen // (op 1084) Load value

s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:23], v[44:47], a[4:7], v40, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 404) 




v_add_u32 v17, 4, v17 // (op 42016) 
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:23], v[48:51], a[8:11], v40, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 630) 
ds_read_b128 v[60:63], v4 offset:5120 // (op 1305) Load local data 



ds_read_b128 v[64:67], v4 offset:6144 // (op 1531) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:23], v[30:33], a[12:15], v40, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 856) 



ds_read_b128 v[68:71], v4 offset:7168 // (op 1757) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v72, v36, 0, 8 // (op 38571) 



v_bfe_u32 v73, v36, 8, 8 // (op 38571) 
v_bfe_u32 v74, v36, 16, 8 // (op 38571) 
v_bfe_u32 v75, v36, 24, 8 // (op 38571) 
v_permlane16_swap_b32 v72, v73 // (op 38571) 
s_nop 0
v_permlane16_swap_b32 v74, v75 // (op 38571) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v72, v74 // (op 38571) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v73, v75 // (op 38571) 
ds_read_b128 v[76:79], v0 offset:17408 // (op 2253) Load local data 




s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[20:23], v[56:59], a[16:19], v40, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1082) 
ds_read_b128 v[34:37], v0 offset:18432 // (op 4255) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[20:23], v[60:63], a[20:23], v40, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1308) 
ds_read_b128 v[80:83], v0 offset:19456 // (op 6257) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[20:23], v[64:67], a[24:27], v40, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1534) 
ds_read_b128 v[84:87], v0 offset:20480 // (op 8259) Load local data 


s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[20:23], v[68:71], a[28:31], v40, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1760) 
s_lshl_b32 s18, s34, 6 // (op 8268) 

v_add_u32 v29, v28, s18 // (op 8268) 


buffer_load_dwordx4 v[20:23], v29, s[64:67], 0 offen // (op 8268) Load value

s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[76:79], v[24:27], a[32:35], v41, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2257) 




v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[76:79], v[48:51], a[40:43], v41, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2311) 
v_add_u32 v16, 4, v16 // (op 41989) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[76:79], v[30:33], a[44:47], v41, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2338) 
ds_read_b128 v[88:91], v0 offset:21504 // (op 10261) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[76:79], v[44:47], a[36:39], v41, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2284) 
ds_read_b128 v[92:95], v0 offset:22528 // (op 12263) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[76:79], v[56:59], a[48:51], v41, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2365) 
s_waitcnt vmcnt(0)
v_bfe_u32 v96, v20, 0, 8 // (op 38479) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[76:79], v[64:67], a[56:59], v41, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2419) 
v_bfe_u32 v97, v20, 8, 8 // (op 38479) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[76:79], v[68:71], a[60:63], v41, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2446) 
v_bfe_u32 v98, v20, 16, 8 // (op 38479) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[76:79], v[60:63], a[52:55], v41, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2392) 
v_bfe_u32 v99, v20, 24, 8 // (op 38479) 

v_permlane16_swap_b32 v96, v97 // (op 38479) 
s_nop 0
v_permlane16_swap_b32 v98, v99 // (op 38479) Wait state hazard: VALU Write Readlane Hazard
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[34:37], v[24:27], a[64:67], v42, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4259) 
s_nop 0
v_permlane32_swap_b32 v96, v98 // (op 38479) Wait state hazard: VALU Write Readlane Hazard
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[34:37], v[48:51], a[72:75], v42, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4313) 
v_permlane32_swap_b32 v97, v99 // (op 38479) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[34:37], v[30:33], a[76:79], v42, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4340) 
ds_read_b128 v[20:23], v0 offset:23552 // (op 14265) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[34:37], v[44:47], a[68:71], v42, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4286) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[34:37], v[56:59], a[80:83], v42, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4367) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[34:37], v[64:67], a[88:91], v42, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4421) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[34:37], v[68:71], a[92:95], v42, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4448) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[34:37], v[60:63], a[84:87], v42, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4394) 

s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[80:83], v[24:27], a[96:99], v43, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6261) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[80:83], v[48:51], a[104:107], v43, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6315) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[80:83], v[30:33], a[108:111], v43, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6342) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[80:83], v[44:47], a[100:103], v43, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6288) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[80:83], v[56:59], a[112:115], v43, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6369) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[80:83], v[64:67], a[120:123], v43, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6423) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[80:83], v[68:71], a[124:127], v43, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6450) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[80:83], v[60:63], a[116:119], v43, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6396) 


s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[84:87], v[44:47], a[132:135], v96, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8290) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[84:87], v[60:63], a[148:151], v96, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8398) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[84:87], v[24:27], a[128:131], v96, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8263) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[84:87], v[48:51], a[136:139], v96, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8317) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[84:87], v[30:33], a[140:143], v96, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8344) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[84:87], v[56:59], a[144:147], v96, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8371) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[84:87], v[64:67], a[152:155], v96, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8425) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[84:87], v[68:71], a[156:159], v96, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 8452) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[88:91], v[44:47], a[164:167], v97, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10292) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[88:91], v[60:63], a[180:183], v97, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10400) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[88:91], v[24:27], a[160:163], v97, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10265) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[88:91], v[48:51], a[168:171], v97, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10319) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[88:91], v[30:33], a[172:175], v97, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10346) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[88:91], v[56:59], a[176:179], v97, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10373) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[88:91], v[64:67], a[184:187], v97, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10427) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[88:91], v[68:71], a[188:191], v97, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 10454) 

s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[92:95], v[44:47], a[196:199], v98, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12294) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[92:95], v[60:63], a[212:215], v98, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12402) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[92:95], v[24:27], a[192:195], v98, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12267) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[92:95], v[48:51], a[200:203], v98, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12321) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[92:95], v[30:33], a[204:207], v98, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12348) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[92:95], v[56:59], a[208:211], v98, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12375) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[92:95], v[64:67], a[216:219], v98, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12429) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[92:95], v[68:71], a[220:223], v98, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 12456) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[20:23], v[44:47], a[228:231], v99, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14296) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[20:23], v[60:63], a[244:247], v99, v73 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14404) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[20:23], v[24:27], a[224:227], v99, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14269) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[20:23], v[48:51], a[232:235], v99, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14323) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[20:23], v[30:33], a[236:239], v99, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14350) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[20:23], v[56:59], a[240:243], v99, v72 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14377) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[20:23], v[64:67], a[248:251], v99, v74 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14431) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[20:23], v[68:71], a[252:255], v99, v75 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 14458) 




s_ashr_i64 s[18:19], s[42:43], 7 // (op 14448) 
s_mov_b32 s22, s18 // (op 14448) convert

s_cmp_lt_u32 s1, s22 // (op 14448) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WGM__ForLoopTop_KLoopTail_14448 // (op 14448) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WGM__ForLoopTop_KLoopTail_14448 if true)
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WGM__ForLoopBottom_KLoopTail_14448:




















s_barrier  // (op 42877) 





















v_cmp_eq_f32 vcc, 0.00000, v9 // (op 116) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WGM__ConditionalFalse_0_beta_116 // (op 116) Condition: False, jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WGM__ConditionalFalse_0_beta_116
v_accvgpr_read v4, a0 // (op 42) 
v_accvgpr_read v5, a1 // (op 42) 
v_accvgpr_read v6, a2 // (op 42) 
v_accvgpr_read v7, a3 // (op 42) 
v_mul_f32 v14, v11, v4 // (op 42) 
v_mul_f32 v15, v11, v5 // (op 42) 
v_mul_f32 v16, v11, v6 // (op 42) 
v_mul_f32 v17, v11, v7 // (op 42) 

v_lshrrev_b32 v4, 16, v14 // (op 48) 
v_lshrrev_b32 v5, 16, v15 // (op 48) 
v_lshrrev_b32 v6, 16, v16 // (op 48) 
v_lshrrev_b32 v7, 16, v17 // (op 48) 


v_pack_b32_f16 v14, v4, v5 // (op 50) 
v_pack_b32_f16 v15, v6, v7 // (op 50) 

v_mov_b32 v0, v13 // (op 50) reg expression

buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 50) Store value



v_accvgpr_read v4, a4 // (op 18202) 
v_accvgpr_read v5, a5 // (op 18202) 
v_accvgpr_read v6, a6 // (op 18202) 
v_accvgpr_read v7, a7 // (op 18202) 
v_mul_f32 v14, v11, v4 // (op 18202) 
v_mul_f32 v15, v11, v5 // (op 18202) 
v_mul_f32 v16, v11, v6 // (op 18202) 
v_mul_f32 v17, v11, v7 // (op 18202) 

v_lshrrev_b32 v4, 16, v14 // (op 18204) 
v_lshrrev_b32 v5, 16, v15 // (op 18204) 
v_lshrrev_b32 v6, 16, v16 // (op 18204) 
v_lshrrev_b32 v7, 16, v17 // (op 18204) 


v_pack_b32_f16 v14, v4, v5 // (op 18205) 
v_pack_b32_f16 v15, v6, v7 // (op 18205) 

s_lshl_b32 s0, s4, 5 // (op 18205) 
v_add_u32 v0, v13, s0 // (op 18205) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18205) Store value



v_accvgpr_read v4, a8 // (op 18216) 
v_accvgpr_read v5, a9 // (op 18216) 
v_accvgpr_read v6, a10 // (op 18216) 
v_accvgpr_read v7, a11 // (op 18216) 
v_mul_f32 v14, v11, v4 // (op 18216) 
v_mul_f32 v15, v11, v5 // (op 18216) 
v_mul_f32 v16, v11, v6 // (op 18216) 
v_mul_f32 v17, v11, v7 // (op 18216) 

v_lshrrev_b32 v4, 16, v14 // (op 18218) 
v_lshrrev_b32 v5, 16, v15 // (op 18218) 
v_lshrrev_b32 v6, 16, v16 // (op 18218) 
v_lshrrev_b32 v7, 16, v17 // (op 18218) 


v_pack_b32_f16 v14, v4, v5 // (op 18219) 
v_pack_b32_f16 v15, v6, v7 // (op 18219) 

s_lshl_b32 s0, s4, 6 // (op 18219) 
v_add_u32 v0, v13, s0 // (op 18219) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18219) Store value



v_accvgpr_read v4, a12 // (op 18230) 
v_accvgpr_read v5, a13 // (op 18230) 
v_accvgpr_read v6, a14 // (op 18230) 
v_accvgpr_read v7, a15 // (op 18230) 
v_mul_f32 v14, v11, v4 // (op 18230) 
v_mul_f32 v15, v11, v5 // (op 18230) 
v_mul_f32 v16, v11, v6 // (op 18230) 
v_mul_f32 v17, v11, v7 // (op 18230) 

v_lshrrev_b32 v4, 16, v14 // (op 18232) 
v_lshrrev_b32 v5, 16, v15 // (op 18232) 
v_lshrrev_b32 v6, 16, v16 // (op 18232) 
v_lshrrev_b32 v7, 16, v17 // (op 18232) 


v_pack_b32_f16 v14, v4, v5 // (op 18233) 
v_pack_b32_f16 v15, v6, v7 // (op 18233) 

s_lshl_b32 s0, s4, 5 // (op 18233) 
s_mul_i32 s1, 3, s0 // (op 18233) 

v_add_u32 v0, v13, s1 // (op 18233) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18233) Store value



v_accvgpr_read v4, a16 // (op 18244) 
v_accvgpr_read v5, a17 // (op 18244) 
v_accvgpr_read v6, a18 // (op 18244) 
v_accvgpr_read v7, a19 // (op 18244) 
v_mul_f32 v14, v11, v4 // (op 18244) 
v_mul_f32 v15, v11, v5 // (op 18244) 
v_mul_f32 v16, v11, v6 // (op 18244) 
v_mul_f32 v17, v11, v7 // (op 18244) 

v_lshrrev_b32 v4, 16, v14 // (op 18246) 
v_lshrrev_b32 v5, 16, v15 // (op 18246) 
v_lshrrev_b32 v6, 16, v16 // (op 18246) 
v_lshrrev_b32 v7, 16, v17 // (op 18246) 


v_pack_b32_f16 v14, v4, v5 // (op 18247) 
v_pack_b32_f16 v15, v6, v7 // (op 18247) 

s_lshl_b32 s0, s4, 7 // (op 18247) 
v_add_u32 v0, v13, s0 // (op 18247) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18247) Store value



v_accvgpr_read v4, a20 // (op 18258) 
v_accvgpr_read v5, a21 // (op 18258) 
v_accvgpr_read v6, a22 // (op 18258) 
v_accvgpr_read v7, a23 // (op 18258) 
v_mul_f32 v14, v11, v4 // (op 18258) 
v_mul_f32 v15, v11, v5 // (op 18258) 
v_mul_f32 v16, v11, v6 // (op 18258) 
v_mul_f32 v17, v11, v7 // (op 18258) 

v_lshrrev_b32 v4, 16, v14 // (op 18260) 
v_lshrrev_b32 v5, 16, v15 // (op 18260) 
v_lshrrev_b32 v6, 16, v16 // (op 18260) 
v_lshrrev_b32 v7, 16, v17 // (op 18260) 


v_pack_b32_f16 v14, v4, v5 // (op 18261) 
v_pack_b32_f16 v15, v6, v7 // (op 18261) 

s_lshl_b32 s0, s4, 5 // (op 18261) 
s_mul_i32 s1, 5, s0 // (op 18261) 

v_add_u32 v0, v13, s1 // (op 18261) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18261) Store value



v_accvgpr_read v4, a24 // (op 18272) 
v_accvgpr_read v5, a25 // (op 18272) 
v_accvgpr_read v6, a26 // (op 18272) 
v_accvgpr_read v7, a27 // (op 18272) 
v_mul_f32 v14, v11, v4 // (op 18272) 
v_mul_f32 v15, v11, v5 // (op 18272) 
v_mul_f32 v16, v11, v6 // (op 18272) 
v_mul_f32 v17, v11, v7 // (op 18272) 

v_lshrrev_b32 v4, 16, v14 // (op 18274) 
v_lshrrev_b32 v5, 16, v15 // (op 18274) 
v_lshrrev_b32 v6, 16, v16 // (op 18274) 
v_lshrrev_b32 v7, 16, v17 // (op 18274) 


v_pack_b32_f16 v14, v4, v5 // (op 18275) 
v_pack_b32_f16 v15, v6, v7 // (op 18275) 

s_lshl_b32 s0, s4, 5 // (op 18275) 
s_mul_i32 s1, 6, s0 // (op 18275) 

v_add_u32 v0, v13, s1 // (op 18275) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18275) Store value



v_accvgpr_read v4, a28 // (op 18286) 
v_accvgpr_read v5, a29 // (op 18286) 
v_accvgpr_read v6, a30 // (op 18286) 
v_accvgpr_read v7, a31 // (op 18286) 
v_mul_f32 v14, v11, v4 // (op 18286) 
v_mul_f32 v15, v11, v5 // (op 18286) 
v_mul_f32 v16, v11, v6 // (op 18286) 
v_mul_f32 v17, v11, v7 // (op 18286) 

v_lshrrev_b32 v4, 16, v14 // (op 18288) 
v_lshrrev_b32 v5, 16, v15 // (op 18288) 
v_lshrrev_b32 v6, 16, v16 // (op 18288) 
v_lshrrev_b32 v7, 16, v17 // (op 18288) 


v_pack_b32_f16 v14, v4, v5 // (op 18289) 
v_pack_b32_f16 v15, v6, v7 // (op 18289) 

s_lshl_b32 s0, s4, 5 // (op 18289) 
s_mul_i32 s1, 7, s0 // (op 18289) 

v_add_u32 v0, v13, s1 // (op 18289) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18289) Store value



v_accvgpr_read v4, a32 // (op 18342) 
v_accvgpr_read v5, a33 // (op 18342) 
v_accvgpr_read v6, a34 // (op 18342) 
v_accvgpr_read v7, a35 // (op 18342) 
v_mul_f32 v14, v11, v4 // (op 18342) 
v_mul_f32 v15, v11, v5 // (op 18342) 
v_mul_f32 v16, v11, v6 // (op 18342) 
v_mul_f32 v17, v11, v7 // (op 18342) 

v_lshrrev_b32 v4, 16, v14 // (op 18344) 
v_lshrrev_b32 v5, 16, v15 // (op 18344) 
v_lshrrev_b32 v6, 16, v16 // (op 18344) 
v_lshrrev_b32 v7, 16, v17 // (op 18344) 


v_pack_b32_f16 v14, v4, v5 // (op 18346) 
v_pack_b32_f16 v15, v6, v7 // (op 18346) 

v_add_u32 v0, 32, v13 // (op 18346) 

buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18346) Store value



v_accvgpr_read v4, a36 // (op 18364) 
v_accvgpr_read v5, a37 // (op 18364) 
v_accvgpr_read v6, a38 // (op 18364) 
v_accvgpr_read v7, a39 // (op 18364) 
v_mul_f32 v14, v11, v4 // (op 18364) 
v_mul_f32 v15, v11, v5 // (op 18364) 
v_mul_f32 v16, v11, v6 // (op 18364) 
v_mul_f32 v17, v11, v7 // (op 18364) 

v_lshrrev_b32 v4, 16, v14 // (op 18366) 
v_lshrrev_b32 v5, 16, v15 // (op 18366) 
v_lshrrev_b32 v6, 16, v16 // (op 18366) 
v_lshrrev_b32 v7, 16, v17 // (op 18366) 


v_pack_b32_f16 v14, v4, v5 // (op 18348) 
v_pack_b32_f16 v15, v6, v7 // (op 18348) 

s_lshl_b32 s0, s4, 5 // (op 18348) 
s_add_u32 s0, s0, 32 // (op 18348) 
v_add_u32 v0, v13, s0 // (op 18348) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18348) Store value



v_accvgpr_read v4, a40 // (op 18370) 
v_accvgpr_read v5, a41 // (op 18370) 
v_accvgpr_read v6, a42 // (op 18370) 
v_accvgpr_read v7, a43 // (op 18370) 
v_mul_f32 v14, v11, v4 // (op 18370) 
v_mul_f32 v15, v11, v5 // (op 18370) 
v_mul_f32 v16, v11, v6 // (op 18370) 
v_mul_f32 v17, v11, v7 // (op 18370) 

v_lshrrev_b32 v4, 16, v14 // (op 18372) 
v_lshrrev_b32 v5, 16, v15 // (op 18372) 
v_lshrrev_b32 v6, 16, v16 // (op 18372) 
v_lshrrev_b32 v7, 16, v17 // (op 18372) 


v_pack_b32_f16 v14, v4, v5 // (op 18350) 
v_pack_b32_f16 v15, v6, v7 // (op 18350) 

s_lshl_b32 s0, s4, 6 // (op 18350) 
s_add_u32 s0, s0, 32 // (op 18350) 
v_add_u32 v0, v13, s0 // (op 18350) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18350) Store value



v_accvgpr_read v4, a44 // (op 18376) 
v_accvgpr_read v5, a45 // (op 18376) 
v_accvgpr_read v6, a46 // (op 18376) 
v_accvgpr_read v7, a47 // (op 18376) 
v_mul_f32 v14, v11, v4 // (op 18376) 
v_mul_f32 v15, v11, v5 // (op 18376) 
v_mul_f32 v16, v11, v6 // (op 18376) 
v_mul_f32 v17, v11, v7 // (op 18376) 

v_lshrrev_b32 v4, 16, v14 // (op 18378) 
v_lshrrev_b32 v5, 16, v15 // (op 18378) 
v_lshrrev_b32 v6, 16, v16 // (op 18378) 
v_lshrrev_b32 v7, 16, v17 // (op 18378) 


v_pack_b32_f16 v14, v4, v5 // (op 18352) 
v_pack_b32_f16 v15, v6, v7 // (op 18352) 

s_lshl_b32 s0, s4, 5 // (op 18352) 
s_mul_i32 s1, 3, s0 // (op 18352) 
s_add_u32 s1, s1, 32 // (op 18352) 

v_add_u32 v0, v13, s1 // (op 18352) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18352) Store value



v_accvgpr_read v4, a48 // (op 18382) 
v_accvgpr_read v5, a49 // (op 18382) 
v_accvgpr_read v6, a50 // (op 18382) 
v_accvgpr_read v7, a51 // (op 18382) 
v_mul_f32 v14, v11, v4 // (op 18382) 
v_mul_f32 v15, v11, v5 // (op 18382) 
v_mul_f32 v16, v11, v6 // (op 18382) 
v_mul_f32 v17, v11, v7 // (op 18382) 

v_lshrrev_b32 v4, 16, v14 // (op 18384) 
v_lshrrev_b32 v5, 16, v15 // (op 18384) 
v_lshrrev_b32 v6, 16, v16 // (op 18384) 
v_lshrrev_b32 v7, 16, v17 // (op 18384) 


v_pack_b32_f16 v14, v4, v5 // (op 18354) 
v_pack_b32_f16 v15, v6, v7 // (op 18354) 

s_lshl_b32 s0, s4, 7 // (op 18354) 
s_add_u32 s0, s0, 32 // (op 18354) 
v_add_u32 v0, v13, s0 // (op 18354) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18354) Store value



v_accvgpr_read v4, a52 // (op 18388) 
v_accvgpr_read v5, a53 // (op 18388) 
v_accvgpr_read v6, a54 // (op 18388) 
v_accvgpr_read v7, a55 // (op 18388) 
v_mul_f32 v14, v11, v4 // (op 18388) 
v_mul_f32 v15, v11, v5 // (op 18388) 
v_mul_f32 v16, v11, v6 // (op 18388) 
v_mul_f32 v17, v11, v7 // (op 18388) 

v_lshrrev_b32 v4, 16, v14 // (op 18390) 
v_lshrrev_b32 v5, 16, v15 // (op 18390) 
v_lshrrev_b32 v6, 16, v16 // (op 18390) 
v_lshrrev_b32 v7, 16, v17 // (op 18390) 


v_pack_b32_f16 v14, v4, v5 // (op 18356) 
v_pack_b32_f16 v15, v6, v7 // (op 18356) 

s_lshl_b32 s0, s4, 5 // (op 18356) 
s_mul_i32 s1, 5, s0 // (op 18356) 
s_add_u32 s1, s1, 32 // (op 18356) 

v_add_u32 v0, v13, s1 // (op 18356) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18356) Store value



v_accvgpr_read v4, a56 // (op 18394) 
v_accvgpr_read v5, a57 // (op 18394) 
v_accvgpr_read v6, a58 // (op 18394) 
v_accvgpr_read v7, a59 // (op 18394) 
v_mul_f32 v14, v11, v4 // (op 18394) 
v_mul_f32 v15, v11, v5 // (op 18394) 
v_mul_f32 v16, v11, v6 // (op 18394) 
v_mul_f32 v17, v11, v7 // (op 18394) 

v_lshrrev_b32 v4, 16, v14 // (op 18396) 
v_lshrrev_b32 v5, 16, v15 // (op 18396) 
v_lshrrev_b32 v6, 16, v16 // (op 18396) 
v_lshrrev_b32 v7, 16, v17 // (op 18396) 


v_pack_b32_f16 v14, v4, v5 // (op 18358) 
v_pack_b32_f16 v15, v6, v7 // (op 18358) 

s_lshl_b32 s0, s4, 5 // (op 18358) 
s_mul_i32 s1, 6, s0 // (op 18358) 
s_add_u32 s1, s1, 32 // (op 18358) 

v_add_u32 v0, v13, s1 // (op 18358) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18358) Store value



v_accvgpr_read v4, a60 // (op 18400) 
v_accvgpr_read v5, a61 // (op 18400) 
v_accvgpr_read v6, a62 // (op 18400) 
v_accvgpr_read v7, a63 // (op 18400) 
v_mul_f32 v14, v11, v4 // (op 18400) 
v_mul_f32 v15, v11, v5 // (op 18400) 
v_mul_f32 v16, v11, v6 // (op 18400) 
v_mul_f32 v17, v11, v7 // (op 18400) 

v_lshrrev_b32 v4, 16, v14 // (op 18402) 
v_lshrrev_b32 v5, 16, v15 // (op 18402) 
v_lshrrev_b32 v6, 16, v16 // (op 18402) 
v_lshrrev_b32 v7, 16, v17 // (op 18402) 


v_pack_b32_f16 v14, v4, v5 // (op 18360) 
v_pack_b32_f16 v15, v6, v7 // (op 18360) 

s_lshl_b32 s0, s4, 5 // (op 18360) 
s_mul_i32 s1, 7, s0 // (op 18360) 
s_add_u32 s1, s1, 32 // (op 18360) 

v_add_u32 v0, v13, s1 // (op 18360) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18360) Store value



v_accvgpr_read v4, a64 // (op 18490) 
v_accvgpr_read v5, a65 // (op 18490) 
v_accvgpr_read v6, a66 // (op 18490) 
v_accvgpr_read v7, a67 // (op 18490) 
v_mul_f32 v14, v11, v4 // (op 18490) 
v_mul_f32 v15, v11, v5 // (op 18490) 
v_mul_f32 v16, v11, v6 // (op 18490) 
v_mul_f32 v17, v11, v7 // (op 18490) 

v_lshrrev_b32 v4, 16, v14 // (op 18492) 
v_lshrrev_b32 v5, 16, v15 // (op 18492) 
v_lshrrev_b32 v6, 16, v16 // (op 18492) 
v_lshrrev_b32 v7, 16, v17 // (op 18492) 


v_pack_b32_f16 v14, v4, v5 // (op 18494) 
v_pack_b32_f16 v15, v6, v7 // (op 18494) 

v_add_u32 v0, 64, v13 // (op 18494) 

buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18494) Store value



v_accvgpr_read v4, a68 // (op 18512) 
v_accvgpr_read v5, a69 // (op 18512) 
v_accvgpr_read v6, a70 // (op 18512) 
v_accvgpr_read v7, a71 // (op 18512) 
v_mul_f32 v14, v11, v4 // (op 18512) 
v_mul_f32 v15, v11, v5 // (op 18512) 
v_mul_f32 v16, v11, v6 // (op 18512) 
v_mul_f32 v17, v11, v7 // (op 18512) 

v_lshrrev_b32 v4, 16, v14 // (op 18514) 
v_lshrrev_b32 v5, 16, v15 // (op 18514) 
v_lshrrev_b32 v6, 16, v16 // (op 18514) 
v_lshrrev_b32 v7, 16, v17 // (op 18514) 


v_pack_b32_f16 v14, v4, v5 // (op 18496) 
v_pack_b32_f16 v15, v6, v7 // (op 18496) 

s_lshl_b32 s0, s4, 5 // (op 18496) 
s_add_u32 s0, s0, 64 // (op 18496) 
v_add_u32 v0, v13, s0 // (op 18496) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18496) Store value



v_accvgpr_read v4, a72 // (op 18518) 
v_accvgpr_read v5, a73 // (op 18518) 
v_accvgpr_read v6, a74 // (op 18518) 
v_accvgpr_read v7, a75 // (op 18518) 
v_mul_f32 v14, v11, v4 // (op 18518) 
v_mul_f32 v15, v11, v5 // (op 18518) 
v_mul_f32 v16, v11, v6 // (op 18518) 
v_mul_f32 v17, v11, v7 // (op 18518) 

v_lshrrev_b32 v4, 16, v14 // (op 18520) 
v_lshrrev_b32 v5, 16, v15 // (op 18520) 
v_lshrrev_b32 v6, 16, v16 // (op 18520) 
v_lshrrev_b32 v7, 16, v17 // (op 18520) 


v_pack_b32_f16 v14, v4, v5 // (op 18498) 
v_pack_b32_f16 v15, v6, v7 // (op 18498) 

s_lshl_b32 s0, s4, 6 // (op 18498) 
s_add_u32 s0, s0, 64 // (op 18498) 
v_add_u32 v0, v13, s0 // (op 18498) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18498) Store value



v_accvgpr_read v4, a76 // (op 18524) 
v_accvgpr_read v5, a77 // (op 18524) 
v_accvgpr_read v6, a78 // (op 18524) 
v_accvgpr_read v7, a79 // (op 18524) 
v_mul_f32 v14, v11, v4 // (op 18524) 
v_mul_f32 v15, v11, v5 // (op 18524) 
v_mul_f32 v16, v11, v6 // (op 18524) 
v_mul_f32 v17, v11, v7 // (op 18524) 

v_lshrrev_b32 v4, 16, v14 // (op 18526) 
v_lshrrev_b32 v5, 16, v15 // (op 18526) 
v_lshrrev_b32 v6, 16, v16 // (op 18526) 
v_lshrrev_b32 v7, 16, v17 // (op 18526) 


v_pack_b32_f16 v14, v4, v5 // (op 18500) 
v_pack_b32_f16 v15, v6, v7 // (op 18500) 

s_lshl_b32 s0, s4, 5 // (op 18500) 
s_mul_i32 s1, 3, s0 // (op 18500) 
s_add_u32 s1, s1, 64 // (op 18500) 

v_add_u32 v0, v13, s1 // (op 18500) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18500) Store value



v_accvgpr_read v4, a80 // (op 18530) 
v_accvgpr_read v5, a81 // (op 18530) 
v_accvgpr_read v6, a82 // (op 18530) 
v_accvgpr_read v7, a83 // (op 18530) 
v_mul_f32 v14, v11, v4 // (op 18530) 
v_mul_f32 v15, v11, v5 // (op 18530) 
v_mul_f32 v16, v11, v6 // (op 18530) 
v_mul_f32 v17, v11, v7 // (op 18530) 

v_lshrrev_b32 v4, 16, v14 // (op 18532) 
v_lshrrev_b32 v5, 16, v15 // (op 18532) 
v_lshrrev_b32 v6, 16, v16 // (op 18532) 
v_lshrrev_b32 v7, 16, v17 // (op 18532) 


v_pack_b32_f16 v14, v4, v5 // (op 18502) 
v_pack_b32_f16 v15, v6, v7 // (op 18502) 

s_lshl_b32 s0, s4, 7 // (op 18502) 
s_add_u32 s0, s0, 64 // (op 18502) 
v_add_u32 v0, v13, s0 // (op 18502) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18502) Store value



v_accvgpr_read v4, a84 // (op 18536) 
v_accvgpr_read v5, a85 // (op 18536) 
v_accvgpr_read v6, a86 // (op 18536) 
v_accvgpr_read v7, a87 // (op 18536) 
v_mul_f32 v14, v11, v4 // (op 18536) 
v_mul_f32 v15, v11, v5 // (op 18536) 
v_mul_f32 v16, v11, v6 // (op 18536) 
v_mul_f32 v17, v11, v7 // (op 18536) 

v_lshrrev_b32 v4, 16, v14 // (op 18538) 
v_lshrrev_b32 v5, 16, v15 // (op 18538) 
v_lshrrev_b32 v6, 16, v16 // (op 18538) 
v_lshrrev_b32 v7, 16, v17 // (op 18538) 


v_pack_b32_f16 v14, v4, v5 // (op 18504) 
v_pack_b32_f16 v15, v6, v7 // (op 18504) 

s_lshl_b32 s0, s4, 5 // (op 18504) 
s_mul_i32 s1, 5, s0 // (op 18504) 
s_add_u32 s1, s1, 64 // (op 18504) 

v_add_u32 v0, v13, s1 // (op 18504) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18504) Store value



v_accvgpr_read v4, a88 // (op 18542) 
v_accvgpr_read v5, a89 // (op 18542) 
v_accvgpr_read v6, a90 // (op 18542) 
v_accvgpr_read v7, a91 // (op 18542) 
v_mul_f32 v14, v11, v4 // (op 18542) 
v_mul_f32 v15, v11, v5 // (op 18542) 
v_mul_f32 v16, v11, v6 // (op 18542) 
v_mul_f32 v17, v11, v7 // (op 18542) 

v_lshrrev_b32 v4, 16, v14 // (op 18544) 
v_lshrrev_b32 v5, 16, v15 // (op 18544) 
v_lshrrev_b32 v6, 16, v16 // (op 18544) 
v_lshrrev_b32 v7, 16, v17 // (op 18544) 


v_pack_b32_f16 v14, v4, v5 // (op 18506) 
v_pack_b32_f16 v15, v6, v7 // (op 18506) 

s_lshl_b32 s0, s4, 5 // (op 18506) 
s_mul_i32 s1, 6, s0 // (op 18506) 
s_add_u32 s1, s1, 64 // (op 18506) 

v_add_u32 v0, v13, s1 // (op 18506) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18506) Store value



v_accvgpr_read v4, a92 // (op 18548) 
v_accvgpr_read v5, a93 // (op 18548) 
v_accvgpr_read v6, a94 // (op 18548) 
v_accvgpr_read v7, a95 // (op 18548) 
v_mul_f32 v14, v11, v4 // (op 18548) 
v_mul_f32 v15, v11, v5 // (op 18548) 
v_mul_f32 v16, v11, v6 // (op 18548) 
v_mul_f32 v17, v11, v7 // (op 18548) 

v_lshrrev_b32 v4, 16, v14 // (op 18550) 
v_lshrrev_b32 v5, 16, v15 // (op 18550) 
v_lshrrev_b32 v6, 16, v16 // (op 18550) 
v_lshrrev_b32 v7, 16, v17 // (op 18550) 


v_pack_b32_f16 v14, v4, v5 // (op 18508) 
v_pack_b32_f16 v15, v6, v7 // (op 18508) 

s_lshl_b32 s0, s4, 5 // (op 18508) 
s_mul_i32 s1, 7, s0 // (op 18508) 
s_add_u32 s1, s1, 64 // (op 18508) 

v_add_u32 v0, v13, s1 // (op 18508) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18508) Store value



v_accvgpr_read v4, a96 // (op 18638) 
v_accvgpr_read v5, a97 // (op 18638) 
v_accvgpr_read v6, a98 // (op 18638) 
v_accvgpr_read v7, a99 // (op 18638) 
v_mul_f32 v14, v11, v4 // (op 18638) 
v_mul_f32 v15, v11, v5 // (op 18638) 
v_mul_f32 v16, v11, v6 // (op 18638) 
v_mul_f32 v17, v11, v7 // (op 18638) 

v_lshrrev_b32 v4, 16, v14 // (op 18640) 
v_lshrrev_b32 v5, 16, v15 // (op 18640) 
v_lshrrev_b32 v6, 16, v16 // (op 18640) 
v_lshrrev_b32 v7, 16, v17 // (op 18640) 


v_pack_b32_f16 v14, v4, v5 // (op 18642) 
v_pack_b32_f16 v15, v6, v7 // (op 18642) 

v_add_u32 v0, 96, v13 // (op 18642) 

buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18642) Store value



v_accvgpr_read v4, a100 // (op 18660) 
v_accvgpr_read v5, a101 // (op 18660) 
v_accvgpr_read v6, a102 // (op 18660) 
v_accvgpr_read v7, a103 // (op 18660) 
v_mul_f32 v14, v11, v4 // (op 18660) 
v_mul_f32 v15, v11, v5 // (op 18660) 
v_mul_f32 v16, v11, v6 // (op 18660) 
v_mul_f32 v17, v11, v7 // (op 18660) 

v_lshrrev_b32 v4, 16, v14 // (op 18662) 
v_lshrrev_b32 v5, 16, v15 // (op 18662) 
v_lshrrev_b32 v6, 16, v16 // (op 18662) 
v_lshrrev_b32 v7, 16, v17 // (op 18662) 


v_pack_b32_f16 v14, v4, v5 // (op 18644) 
v_pack_b32_f16 v15, v6, v7 // (op 18644) 

s_lshl_b32 s0, s4, 5 // (op 18644) 
s_add_u32 s0, s0, 96 // (op 18644) 
v_add_u32 v0, v13, s0 // (op 18644) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18644) Store value



v_accvgpr_read v4, a104 // (op 18666) 
v_accvgpr_read v5, a105 // (op 18666) 
v_accvgpr_read v6, a106 // (op 18666) 
v_accvgpr_read v7, a107 // (op 18666) 
v_mul_f32 v14, v11, v4 // (op 18666) 
v_mul_f32 v15, v11, v5 // (op 18666) 
v_mul_f32 v16, v11, v6 // (op 18666) 
v_mul_f32 v17, v11, v7 // (op 18666) 

v_lshrrev_b32 v4, 16, v14 // (op 18668) 
v_lshrrev_b32 v5, 16, v15 // (op 18668) 
v_lshrrev_b32 v6, 16, v16 // (op 18668) 
v_lshrrev_b32 v7, 16, v17 // (op 18668) 


v_pack_b32_f16 v14, v4, v5 // (op 18646) 
v_pack_b32_f16 v15, v6, v7 // (op 18646) 

s_lshl_b32 s0, s4, 6 // (op 18646) 
s_add_u32 s0, s0, 96 // (op 18646) 
v_add_u32 v0, v13, s0 // (op 18646) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18646) Store value



v_accvgpr_read v4, a108 // (op 18672) 
v_accvgpr_read v5, a109 // (op 18672) 
v_accvgpr_read v6, a110 // (op 18672) 
v_accvgpr_read v7, a111 // (op 18672) 
v_mul_f32 v14, v11, v4 // (op 18672) 
v_mul_f32 v15, v11, v5 // (op 18672) 
v_mul_f32 v16, v11, v6 // (op 18672) 
v_mul_f32 v17, v11, v7 // (op 18672) 

v_lshrrev_b32 v4, 16, v14 // (op 18674) 
v_lshrrev_b32 v5, 16, v15 // (op 18674) 
v_lshrrev_b32 v6, 16, v16 // (op 18674) 
v_lshrrev_b32 v7, 16, v17 // (op 18674) 


v_pack_b32_f16 v14, v4, v5 // (op 18648) 
v_pack_b32_f16 v15, v6, v7 // (op 18648) 

s_lshl_b32 s0, s4, 5 // (op 18648) 
s_mul_i32 s1, 3, s0 // (op 18648) 
s_add_u32 s1, s1, 96 // (op 18648) 

v_add_u32 v0, v13, s1 // (op 18648) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18648) Store value



v_accvgpr_read v4, a112 // (op 18678) 
v_accvgpr_read v5, a113 // (op 18678) 
v_accvgpr_read v6, a114 // (op 18678) 
v_accvgpr_read v7, a115 // (op 18678) 
v_mul_f32 v14, v11, v4 // (op 18678) 
v_mul_f32 v15, v11, v5 // (op 18678) 
v_mul_f32 v16, v11, v6 // (op 18678) 
v_mul_f32 v17, v11, v7 // (op 18678) 

v_lshrrev_b32 v4, 16, v14 // (op 18680) 
v_lshrrev_b32 v5, 16, v15 // (op 18680) 
v_lshrrev_b32 v6, 16, v16 // (op 18680) 
v_lshrrev_b32 v7, 16, v17 // (op 18680) 


v_pack_b32_f16 v14, v4, v5 // (op 18650) 
v_pack_b32_f16 v15, v6, v7 // (op 18650) 

s_lshl_b32 s0, s4, 7 // (op 18650) 
s_add_u32 s0, s0, 96 // (op 18650) 
v_add_u32 v0, v13, s0 // (op 18650) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18650) Store value



v_accvgpr_read v4, a116 // (op 18684) 
v_accvgpr_read v5, a117 // (op 18684) 
v_accvgpr_read v6, a118 // (op 18684) 
v_accvgpr_read v7, a119 // (op 18684) 
v_mul_f32 v14, v11, v4 // (op 18684) 
v_mul_f32 v15, v11, v5 // (op 18684) 
v_mul_f32 v16, v11, v6 // (op 18684) 
v_mul_f32 v17, v11, v7 // (op 18684) 

v_lshrrev_b32 v4, 16, v14 // (op 18686) 
v_lshrrev_b32 v5, 16, v15 // (op 18686) 
v_lshrrev_b32 v6, 16, v16 // (op 18686) 
v_lshrrev_b32 v7, 16, v17 // (op 18686) 


v_pack_b32_f16 v14, v4, v5 // (op 18652) 
v_pack_b32_f16 v15, v6, v7 // (op 18652) 

s_lshl_b32 s0, s4, 5 // (op 18652) 
s_mul_i32 s1, 5, s0 // (op 18652) 
s_add_u32 s1, s1, 96 // (op 18652) 

v_add_u32 v0, v13, s1 // (op 18652) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18652) Store value



v_accvgpr_read v4, a120 // (op 18690) 
v_accvgpr_read v5, a121 // (op 18690) 
v_accvgpr_read v6, a122 // (op 18690) 
v_accvgpr_read v7, a123 // (op 18690) 
v_mul_f32 v14, v11, v4 // (op 18690) 
v_mul_f32 v15, v11, v5 // (op 18690) 
v_mul_f32 v16, v11, v6 // (op 18690) 
v_mul_f32 v17, v11, v7 // (op 18690) 

v_lshrrev_b32 v4, 16, v14 // (op 18692) 
v_lshrrev_b32 v5, 16, v15 // (op 18692) 
v_lshrrev_b32 v6, 16, v16 // (op 18692) 
v_lshrrev_b32 v7, 16, v17 // (op 18692) 


v_pack_b32_f16 v14, v4, v5 // (op 18654) 
v_pack_b32_f16 v15, v6, v7 // (op 18654) 

s_lshl_b32 s0, s4, 5 // (op 18654) 
s_mul_i32 s1, 6, s0 // (op 18654) 
s_add_u32 s1, s1, 96 // (op 18654) 

v_add_u32 v0, v13, s1 // (op 18654) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18654) Store value



v_accvgpr_read v4, a124 // (op 18696) 
v_accvgpr_read v5, a125 // (op 18696) 
v_accvgpr_read v6, a126 // (op 18696) 
v_accvgpr_read v7, a127 // (op 18696) 
v_mul_f32 v14, v11, v4 // (op 18696) 
v_mul_f32 v15, v11, v5 // (op 18696) 
v_mul_f32 v16, v11, v6 // (op 18696) 
v_mul_f32 v17, v11, v7 // (op 18696) 

v_lshrrev_b32 v4, 16, v14 // (op 18698) 
v_lshrrev_b32 v5, 16, v15 // (op 18698) 
v_lshrrev_b32 v6, 16, v16 // (op 18698) 
v_lshrrev_b32 v7, 16, v17 // (op 18698) 


v_pack_b32_f16 v14, v4, v5 // (op 18656) 
v_pack_b32_f16 v15, v6, v7 // (op 18656) 

s_lshl_b32 s0, s4, 5 // (op 18656) 
s_mul_i32 s1, 7, s0 // (op 18656) 
s_add_u32 s1, s1, 96 // (op 18656) 

v_add_u32 v0, v13, s1 // (op 18656) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18656) Store value



v_accvgpr_read v4, a128 // (op 18786) 
v_accvgpr_read v5, a129 // (op 18786) 
v_accvgpr_read v6, a130 // (op 18786) 
v_accvgpr_read v7, a131 // (op 18786) 
v_mul_f32 v14, v11, v4 // (op 18786) 
v_mul_f32 v15, v11, v5 // (op 18786) 
v_mul_f32 v16, v11, v6 // (op 18786) 
v_mul_f32 v17, v11, v7 // (op 18786) 

v_lshrrev_b32 v4, 16, v14 // (op 18788) 
v_lshrrev_b32 v5, 16, v15 // (op 18788) 
v_lshrrev_b32 v6, 16, v16 // (op 18788) 
v_lshrrev_b32 v7, 16, v17 // (op 18788) 


v_pack_b32_f16 v14, v4, v5 // (op 18790) 
v_pack_b32_f16 v15, v6, v7 // (op 18790) 

v_add_u32 v0, 128, v13 // (op 18790) 

buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18790) Store value



v_accvgpr_read v4, a132 // (op 18808) 
v_accvgpr_read v5, a133 // (op 18808) 
v_accvgpr_read v6, a134 // (op 18808) 
v_accvgpr_read v7, a135 // (op 18808) 
v_mul_f32 v14, v11, v4 // (op 18808) 
v_mul_f32 v15, v11, v5 // (op 18808) 
v_mul_f32 v16, v11, v6 // (op 18808) 
v_mul_f32 v17, v11, v7 // (op 18808) 

v_lshrrev_b32 v4, 16, v14 // (op 18810) 
v_lshrrev_b32 v5, 16, v15 // (op 18810) 
v_lshrrev_b32 v6, 16, v16 // (op 18810) 
v_lshrrev_b32 v7, 16, v17 // (op 18810) 


v_pack_b32_f16 v14, v4, v5 // (op 18792) 
v_pack_b32_f16 v15, v6, v7 // (op 18792) 

s_lshl_b32 s0, s4, 5 // (op 18792) 
s_add_u32 s0, s0, 128 // (op 18792) 
v_add_u32 v0, v13, s0 // (op 18792) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18792) Store value



v_accvgpr_read v4, a136 // (op 18814) 
v_accvgpr_read v5, a137 // (op 18814) 
v_accvgpr_read v6, a138 // (op 18814) 
v_accvgpr_read v7, a139 // (op 18814) 
v_mul_f32 v14, v11, v4 // (op 18814) 
v_mul_f32 v15, v11, v5 // (op 18814) 
v_mul_f32 v16, v11, v6 // (op 18814) 
v_mul_f32 v17, v11, v7 // (op 18814) 

v_lshrrev_b32 v4, 16, v14 // (op 18816) 
v_lshrrev_b32 v5, 16, v15 // (op 18816) 
v_lshrrev_b32 v6, 16, v16 // (op 18816) 
v_lshrrev_b32 v7, 16, v17 // (op 18816) 


v_pack_b32_f16 v14, v4, v5 // (op 18794) 
v_pack_b32_f16 v15, v6, v7 // (op 18794) 

s_lshl_b32 s0, s4, 6 // (op 18794) 
s_add_u32 s0, s0, 128 // (op 18794) 
v_add_u32 v0, v13, s0 // (op 18794) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18794) Store value



v_accvgpr_read v4, a140 // (op 18820) 
v_accvgpr_read v5, a141 // (op 18820) 
v_accvgpr_read v6, a142 // (op 18820) 
v_accvgpr_read v7, a143 // (op 18820) 
v_mul_f32 v14, v11, v4 // (op 18820) 
v_mul_f32 v15, v11, v5 // (op 18820) 
v_mul_f32 v16, v11, v6 // (op 18820) 
v_mul_f32 v17, v11, v7 // (op 18820) 

v_lshrrev_b32 v4, 16, v14 // (op 18822) 
v_lshrrev_b32 v5, 16, v15 // (op 18822) 
v_lshrrev_b32 v6, 16, v16 // (op 18822) 
v_lshrrev_b32 v7, 16, v17 // (op 18822) 


v_pack_b32_f16 v14, v4, v5 // (op 18796) 
v_pack_b32_f16 v15, v6, v7 // (op 18796) 

s_lshl_b32 s0, s4, 5 // (op 18796) 
s_mul_i32 s1, 3, s0 // (op 18796) 
s_add_u32 s1, s1, 128 // (op 18796) 

v_add_u32 v0, v13, s1 // (op 18796) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18796) Store value



v_accvgpr_read v4, a144 // (op 18826) 
v_accvgpr_read v5, a145 // (op 18826) 
v_accvgpr_read v6, a146 // (op 18826) 
v_accvgpr_read v7, a147 // (op 18826) 
v_mul_f32 v14, v11, v4 // (op 18826) 
v_mul_f32 v15, v11, v5 // (op 18826) 
v_mul_f32 v16, v11, v6 // (op 18826) 
v_mul_f32 v17, v11, v7 // (op 18826) 

v_lshrrev_b32 v4, 16, v14 // (op 18828) 
v_lshrrev_b32 v5, 16, v15 // (op 18828) 
v_lshrrev_b32 v6, 16, v16 // (op 18828) 
v_lshrrev_b32 v7, 16, v17 // (op 18828) 


v_pack_b32_f16 v14, v4, v5 // (op 18798) 
v_pack_b32_f16 v15, v6, v7 // (op 18798) 

s_lshl_b32 s0, s4, 7 // (op 18798) 
s_add_u32 s0, s0, 128 // (op 18798) 
v_add_u32 v0, v13, s0 // (op 18798) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18798) Store value



v_accvgpr_read v4, a148 // (op 18832) 
v_accvgpr_read v5, a149 // (op 18832) 
v_accvgpr_read v6, a150 // (op 18832) 
v_accvgpr_read v7, a151 // (op 18832) 
v_mul_f32 v14, v11, v4 // (op 18832) 
v_mul_f32 v15, v11, v5 // (op 18832) 
v_mul_f32 v16, v11, v6 // (op 18832) 
v_mul_f32 v17, v11, v7 // (op 18832) 

v_lshrrev_b32 v4, 16, v14 // (op 18834) 
v_lshrrev_b32 v5, 16, v15 // (op 18834) 
v_lshrrev_b32 v6, 16, v16 // (op 18834) 
v_lshrrev_b32 v7, 16, v17 // (op 18834) 


v_pack_b32_f16 v14, v4, v5 // (op 18800) 
v_pack_b32_f16 v15, v6, v7 // (op 18800) 

s_lshl_b32 s0, s4, 5 // (op 18800) 
s_mul_i32 s1, 5, s0 // (op 18800) 
s_add_u32 s1, s1, 128 // (op 18800) 

v_add_u32 v0, v13, s1 // (op 18800) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18800) Store value



v_accvgpr_read v4, a152 // (op 18838) 
v_accvgpr_read v5, a153 // (op 18838) 
v_accvgpr_read v6, a154 // (op 18838) 
v_accvgpr_read v7, a155 // (op 18838) 
v_mul_f32 v14, v11, v4 // (op 18838) 
v_mul_f32 v15, v11, v5 // (op 18838) 
v_mul_f32 v16, v11, v6 // (op 18838) 
v_mul_f32 v17, v11, v7 // (op 18838) 

v_lshrrev_b32 v4, 16, v14 // (op 18840) 
v_lshrrev_b32 v5, 16, v15 // (op 18840) 
v_lshrrev_b32 v6, 16, v16 // (op 18840) 
v_lshrrev_b32 v7, 16, v17 // (op 18840) 


v_pack_b32_f16 v14, v4, v5 // (op 18802) 
v_pack_b32_f16 v15, v6, v7 // (op 18802) 

s_lshl_b32 s0, s4, 5 // (op 18802) 
s_mul_i32 s1, 6, s0 // (op 18802) 
s_add_u32 s1, s1, 128 // (op 18802) 

v_add_u32 v0, v13, s1 // (op 18802) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18802) Store value



v_accvgpr_read v4, a156 // (op 18844) 
v_accvgpr_read v5, a157 // (op 18844) 
v_accvgpr_read v6, a158 // (op 18844) 
v_accvgpr_read v7, a159 // (op 18844) 
v_mul_f32 v14, v11, v4 // (op 18844) 
v_mul_f32 v15, v11, v5 // (op 18844) 
v_mul_f32 v16, v11, v6 // (op 18844) 
v_mul_f32 v17, v11, v7 // (op 18844) 

v_lshrrev_b32 v4, 16, v14 // (op 18846) 
v_lshrrev_b32 v5, 16, v15 // (op 18846) 
v_lshrrev_b32 v6, 16, v16 // (op 18846) 
v_lshrrev_b32 v7, 16, v17 // (op 18846) 


v_pack_b32_f16 v14, v4, v5 // (op 18804) 
v_pack_b32_f16 v15, v6, v7 // (op 18804) 

s_lshl_b32 s0, s4, 5 // (op 18804) 
s_mul_i32 s1, 7, s0 // (op 18804) 
s_add_u32 s1, s1, 128 // (op 18804) 

v_add_u32 v0, v13, s1 // (op 18804) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18804) Store value



v_accvgpr_read v4, a160 // (op 18934) 
v_accvgpr_read v5, a161 // (op 18934) 
v_accvgpr_read v6, a162 // (op 18934) 
v_accvgpr_read v7, a163 // (op 18934) 
v_mul_f32 v14, v11, v4 // (op 18934) 
v_mul_f32 v15, v11, v5 // (op 18934) 
v_mul_f32 v16, v11, v6 // (op 18934) 
v_mul_f32 v17, v11, v7 // (op 18934) 

v_lshrrev_b32 v4, 16, v14 // (op 18936) 
v_lshrrev_b32 v5, 16, v15 // (op 18936) 
v_lshrrev_b32 v6, 16, v16 // (op 18936) 
v_lshrrev_b32 v7, 16, v17 // (op 18936) 


v_pack_b32_f16 v14, v4, v5 // (op 18938) 
v_pack_b32_f16 v15, v6, v7 // (op 18938) 

v_add_u32 v0, 160, v13 // (op 18938) 

buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18938) Store value



v_accvgpr_read v4, a164 // (op 18956) 
v_accvgpr_read v5, a165 // (op 18956) 
v_accvgpr_read v6, a166 // (op 18956) 
v_accvgpr_read v7, a167 // (op 18956) 
v_mul_f32 v14, v11, v4 // (op 18956) 
v_mul_f32 v15, v11, v5 // (op 18956) 
v_mul_f32 v16, v11, v6 // (op 18956) 
v_mul_f32 v17, v11, v7 // (op 18956) 

v_lshrrev_b32 v4, 16, v14 // (op 18958) 
v_lshrrev_b32 v5, 16, v15 // (op 18958) 
v_lshrrev_b32 v6, 16, v16 // (op 18958) 
v_lshrrev_b32 v7, 16, v17 // (op 18958) 


v_pack_b32_f16 v14, v4, v5 // (op 18940) 
v_pack_b32_f16 v15, v6, v7 // (op 18940) 

s_lshl_b32 s0, s4, 5 // (op 18940) 
s_add_u32 s0, s0, 160 // (op 18940) 
v_add_u32 v0, v13, s0 // (op 18940) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18940) Store value



v_accvgpr_read v4, a168 // (op 18962) 
v_accvgpr_read v5, a169 // (op 18962) 
v_accvgpr_read v6, a170 // (op 18962) 
v_accvgpr_read v7, a171 // (op 18962) 
v_mul_f32 v14, v11, v4 // (op 18962) 
v_mul_f32 v15, v11, v5 // (op 18962) 
v_mul_f32 v16, v11, v6 // (op 18962) 
v_mul_f32 v17, v11, v7 // (op 18962) 

v_lshrrev_b32 v4, 16, v14 // (op 18964) 
v_lshrrev_b32 v5, 16, v15 // (op 18964) 
v_lshrrev_b32 v6, 16, v16 // (op 18964) 
v_lshrrev_b32 v7, 16, v17 // (op 18964) 


v_pack_b32_f16 v14, v4, v5 // (op 18942) 
v_pack_b32_f16 v15, v6, v7 // (op 18942) 

s_lshl_b32 s0, s4, 6 // (op 18942) 
s_add_u32 s0, s0, 160 // (op 18942) 
v_add_u32 v0, v13, s0 // (op 18942) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18942) Store value



v_accvgpr_read v4, a172 // (op 18968) 
v_accvgpr_read v5, a173 // (op 18968) 
v_accvgpr_read v6, a174 // (op 18968) 
v_accvgpr_read v7, a175 // (op 18968) 
v_mul_f32 v14, v11, v4 // (op 18968) 
v_mul_f32 v15, v11, v5 // (op 18968) 
v_mul_f32 v16, v11, v6 // (op 18968) 
v_mul_f32 v17, v11, v7 // (op 18968) 

v_lshrrev_b32 v4, 16, v14 // (op 18970) 
v_lshrrev_b32 v5, 16, v15 // (op 18970) 
v_lshrrev_b32 v6, 16, v16 // (op 18970) 
v_lshrrev_b32 v7, 16, v17 // (op 18970) 


v_pack_b32_f16 v14, v4, v5 // (op 18944) 
v_pack_b32_f16 v15, v6, v7 // (op 18944) 

s_lshl_b32 s0, s4, 5 // (op 18944) 
s_mul_i32 s1, 3, s0 // (op 18944) 
s_add_u32 s1, s1, 160 // (op 18944) 

v_add_u32 v0, v13, s1 // (op 18944) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18944) Store value



v_accvgpr_read v4, a176 // (op 18974) 
v_accvgpr_read v5, a177 // (op 18974) 
v_accvgpr_read v6, a178 // (op 18974) 
v_accvgpr_read v7, a179 // (op 18974) 
v_mul_f32 v14, v11, v4 // (op 18974) 
v_mul_f32 v15, v11, v5 // (op 18974) 
v_mul_f32 v16, v11, v6 // (op 18974) 
v_mul_f32 v17, v11, v7 // (op 18974) 

v_lshrrev_b32 v4, 16, v14 // (op 18976) 
v_lshrrev_b32 v5, 16, v15 // (op 18976) 
v_lshrrev_b32 v6, 16, v16 // (op 18976) 
v_lshrrev_b32 v7, 16, v17 // (op 18976) 


v_pack_b32_f16 v14, v4, v5 // (op 18946) 
v_pack_b32_f16 v15, v6, v7 // (op 18946) 

s_lshl_b32 s0, s4, 7 // (op 18946) 
s_add_u32 s0, s0, 160 // (op 18946) 
v_add_u32 v0, v13, s0 // (op 18946) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18946) Store value



v_accvgpr_read v4, a180 // (op 18980) 
v_accvgpr_read v5, a181 // (op 18980) 
v_accvgpr_read v6, a182 // (op 18980) 
v_accvgpr_read v7, a183 // (op 18980) 
v_mul_f32 v14, v11, v4 // (op 18980) 
v_mul_f32 v15, v11, v5 // (op 18980) 
v_mul_f32 v16, v11, v6 // (op 18980) 
v_mul_f32 v17, v11, v7 // (op 18980) 

v_lshrrev_b32 v4, 16, v14 // (op 18982) 
v_lshrrev_b32 v5, 16, v15 // (op 18982) 
v_lshrrev_b32 v6, 16, v16 // (op 18982) 
v_lshrrev_b32 v7, 16, v17 // (op 18982) 


v_pack_b32_f16 v14, v4, v5 // (op 18948) 
v_pack_b32_f16 v15, v6, v7 // (op 18948) 

s_lshl_b32 s0, s4, 5 // (op 18948) 
s_mul_i32 s1, 5, s0 // (op 18948) 
s_add_u32 s1, s1, 160 // (op 18948) 

v_add_u32 v0, v13, s1 // (op 18948) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18948) Store value



v_accvgpr_read v4, a184 // (op 18986) 
v_accvgpr_read v5, a185 // (op 18986) 
v_accvgpr_read v6, a186 // (op 18986) 
v_accvgpr_read v7, a187 // (op 18986) 
v_mul_f32 v14, v11, v4 // (op 18986) 
v_mul_f32 v15, v11, v5 // (op 18986) 
v_mul_f32 v16, v11, v6 // (op 18986) 
v_mul_f32 v17, v11, v7 // (op 18986) 

v_lshrrev_b32 v4, 16, v14 // (op 18988) 
v_lshrrev_b32 v5, 16, v15 // (op 18988) 
v_lshrrev_b32 v6, 16, v16 // (op 18988) 
v_lshrrev_b32 v7, 16, v17 // (op 18988) 


v_pack_b32_f16 v14, v4, v5 // (op 18950) 
v_pack_b32_f16 v15, v6, v7 // (op 18950) 

s_lshl_b32 s0, s4, 5 // (op 18950) 
s_mul_i32 s1, 6, s0 // (op 18950) 
s_add_u32 s1, s1, 160 // (op 18950) 

v_add_u32 v0, v13, s1 // (op 18950) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18950) Store value



v_accvgpr_read v4, a188 // (op 18992) 
v_accvgpr_read v5, a189 // (op 18992) 
v_accvgpr_read v6, a190 // (op 18992) 
v_accvgpr_read v7, a191 // (op 18992) 
v_mul_f32 v14, v11, v4 // (op 18992) 
v_mul_f32 v15, v11, v5 // (op 18992) 
v_mul_f32 v16, v11, v6 // (op 18992) 
v_mul_f32 v17, v11, v7 // (op 18992) 

v_lshrrev_b32 v4, 16, v14 // (op 18994) 
v_lshrrev_b32 v5, 16, v15 // (op 18994) 
v_lshrrev_b32 v6, 16, v16 // (op 18994) 
v_lshrrev_b32 v7, 16, v17 // (op 18994) 


v_pack_b32_f16 v14, v4, v5 // (op 18952) 
v_pack_b32_f16 v15, v6, v7 // (op 18952) 

s_lshl_b32 s0, s4, 5 // (op 18952) 
s_mul_i32 s1, 7, s0 // (op 18952) 
s_add_u32 s1, s1, 160 // (op 18952) 

v_add_u32 v0, v13, s1 // (op 18952) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 18952) Store value



v_accvgpr_read v4, a192 // (op 19082) 
v_accvgpr_read v5, a193 // (op 19082) 
v_accvgpr_read v6, a194 // (op 19082) 
v_accvgpr_read v7, a195 // (op 19082) 
v_mul_f32 v14, v11, v4 // (op 19082) 
v_mul_f32 v15, v11, v5 // (op 19082) 
v_mul_f32 v16, v11, v6 // (op 19082) 
v_mul_f32 v17, v11, v7 // (op 19082) 

v_lshrrev_b32 v4, 16, v14 // (op 19084) 
v_lshrrev_b32 v5, 16, v15 // (op 19084) 
v_lshrrev_b32 v6, 16, v16 // (op 19084) 
v_lshrrev_b32 v7, 16, v17 // (op 19084) 


v_pack_b32_f16 v14, v4, v5 // (op 19086) 
v_pack_b32_f16 v15, v6, v7 // (op 19086) 

v_add_u32 v0, 192, v13 // (op 19086) 

buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 19086) Store value



v_accvgpr_read v4, a196 // (op 19104) 
v_accvgpr_read v5, a197 // (op 19104) 
v_accvgpr_read v6, a198 // (op 19104) 
v_accvgpr_read v7, a199 // (op 19104) 
v_mul_f32 v14, v11, v4 // (op 19104) 
v_mul_f32 v15, v11, v5 // (op 19104) 
v_mul_f32 v16, v11, v6 // (op 19104) 
v_mul_f32 v17, v11, v7 // (op 19104) 

v_lshrrev_b32 v4, 16, v14 // (op 19106) 
v_lshrrev_b32 v5, 16, v15 // (op 19106) 
v_lshrrev_b32 v6, 16, v16 // (op 19106) 
v_lshrrev_b32 v7, 16, v17 // (op 19106) 


v_pack_b32_f16 v14, v4, v5 // (op 19088) 
v_pack_b32_f16 v15, v6, v7 // (op 19088) 

s_lshl_b32 s0, s4, 5 // (op 19088) 
s_add_u32 s0, s0, 192 // (op 19088) 
v_add_u32 v0, v13, s0 // (op 19088) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 19088) Store value



v_accvgpr_read v4, a200 // (op 19110) 
v_accvgpr_read v5, a201 // (op 19110) 
v_accvgpr_read v6, a202 // (op 19110) 
v_accvgpr_read v7, a203 // (op 19110) 
v_mul_f32 v14, v11, v4 // (op 19110) 
v_mul_f32 v15, v11, v5 // (op 19110) 
v_mul_f32 v16, v11, v6 // (op 19110) 
v_mul_f32 v17, v11, v7 // (op 19110) 

v_lshrrev_b32 v4, 16, v14 // (op 19112) 
v_lshrrev_b32 v5, 16, v15 // (op 19112) 
v_lshrrev_b32 v6, 16, v16 // (op 19112) 
v_lshrrev_b32 v7, 16, v17 // (op 19112) 


v_pack_b32_f16 v14, v4, v5 // (op 19090) 
v_pack_b32_f16 v15, v6, v7 // (op 19090) 

s_lshl_b32 s0, s4, 6 // (op 19090) 
s_add_u32 s0, s0, 192 // (op 19090) 
v_add_u32 v0, v13, s0 // (op 19090) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 19090) Store value



v_accvgpr_read v4, a204 // (op 19116) 
v_accvgpr_read v5, a205 // (op 19116) 
v_accvgpr_read v6, a206 // (op 19116) 
v_accvgpr_read v7, a207 // (op 19116) 
v_mul_f32 v14, v11, v4 // (op 19116) 
v_mul_f32 v15, v11, v5 // (op 19116) 
v_mul_f32 v16, v11, v6 // (op 19116) 
v_mul_f32 v17, v11, v7 // (op 19116) 

v_lshrrev_b32 v4, 16, v14 // (op 19118) 
v_lshrrev_b32 v5, 16, v15 // (op 19118) 
v_lshrrev_b32 v6, 16, v16 // (op 19118) 
v_lshrrev_b32 v7, 16, v17 // (op 19118) 


v_pack_b32_f16 v14, v4, v5 // (op 19092) 
v_pack_b32_f16 v15, v6, v7 // (op 19092) 

s_lshl_b32 s0, s4, 5 // (op 19092) 
s_mul_i32 s1, 3, s0 // (op 19092) 
s_add_u32 s1, s1, 192 // (op 19092) 

v_add_u32 v0, v13, s1 // (op 19092) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 19092) Store value



v_accvgpr_read v4, a208 // (op 19122) 
v_accvgpr_read v5, a209 // (op 19122) 
v_accvgpr_read v6, a210 // (op 19122) 
v_accvgpr_read v7, a211 // (op 19122) 
v_mul_f32 v14, v11, v4 // (op 19122) 
v_mul_f32 v15, v11, v5 // (op 19122) 
v_mul_f32 v16, v11, v6 // (op 19122) 
v_mul_f32 v17, v11, v7 // (op 19122) 

v_lshrrev_b32 v4, 16, v14 // (op 19124) 
v_lshrrev_b32 v5, 16, v15 // (op 19124) 
v_lshrrev_b32 v6, 16, v16 // (op 19124) 
v_lshrrev_b32 v7, 16, v17 // (op 19124) 


v_pack_b32_f16 v14, v4, v5 // (op 19094) 
v_pack_b32_f16 v15, v6, v7 // (op 19094) 

s_lshl_b32 s0, s4, 7 // (op 19094) 
s_add_u32 s0, s0, 192 // (op 19094) 
v_add_u32 v0, v13, s0 // (op 19094) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 19094) Store value



v_accvgpr_read v4, a212 // (op 19128) 
v_accvgpr_read v5, a213 // (op 19128) 
v_accvgpr_read v6, a214 // (op 19128) 
v_accvgpr_read v7, a215 // (op 19128) 
v_mul_f32 v14, v11, v4 // (op 19128) 
v_mul_f32 v15, v11, v5 // (op 19128) 
v_mul_f32 v16, v11, v6 // (op 19128) 
v_mul_f32 v17, v11, v7 // (op 19128) 

v_lshrrev_b32 v4, 16, v14 // (op 19130) 
v_lshrrev_b32 v5, 16, v15 // (op 19130) 
v_lshrrev_b32 v6, 16, v16 // (op 19130) 
v_lshrrev_b32 v7, 16, v17 // (op 19130) 


v_pack_b32_f16 v14, v4, v5 // (op 19096) 
v_pack_b32_f16 v15, v6, v7 // (op 19096) 

s_lshl_b32 s0, s4, 5 // (op 19096) 
s_mul_i32 s1, 5, s0 // (op 19096) 
s_add_u32 s1, s1, 192 // (op 19096) 

v_add_u32 v0, v13, s1 // (op 19096) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 19096) Store value



v_accvgpr_read v4, a216 // (op 19134) 
v_accvgpr_read v5, a217 // (op 19134) 
v_accvgpr_read v6, a218 // (op 19134) 
v_accvgpr_read v7, a219 // (op 19134) 
v_mul_f32 v14, v11, v4 // (op 19134) 
v_mul_f32 v15, v11, v5 // (op 19134) 
v_mul_f32 v16, v11, v6 // (op 19134) 
v_mul_f32 v17, v11, v7 // (op 19134) 

v_lshrrev_b32 v4, 16, v14 // (op 19136) 
v_lshrrev_b32 v5, 16, v15 // (op 19136) 
v_lshrrev_b32 v6, 16, v16 // (op 19136) 
v_lshrrev_b32 v7, 16, v17 // (op 19136) 


v_pack_b32_f16 v14, v4, v5 // (op 19098) 
v_pack_b32_f16 v15, v6, v7 // (op 19098) 

s_lshl_b32 s0, s4, 5 // (op 19098) 
s_mul_i32 s1, 6, s0 // (op 19098) 
s_add_u32 s1, s1, 192 // (op 19098) 

v_add_u32 v0, v13, s1 // (op 19098) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 19098) Store value



v_accvgpr_read v4, a220 // (op 19140) 
v_accvgpr_read v5, a221 // (op 19140) 
v_accvgpr_read v6, a222 // (op 19140) 
v_accvgpr_read v7, a223 // (op 19140) 
v_mul_f32 v14, v11, v4 // (op 19140) 
v_mul_f32 v15, v11, v5 // (op 19140) 
v_mul_f32 v16, v11, v6 // (op 19140) 
v_mul_f32 v17, v11, v7 // (op 19140) 

v_lshrrev_b32 v4, 16, v14 // (op 19142) 
v_lshrrev_b32 v5, 16, v15 // (op 19142) 
v_lshrrev_b32 v6, 16, v16 // (op 19142) 
v_lshrrev_b32 v7, 16, v17 // (op 19142) 


v_pack_b32_f16 v14, v4, v5 // (op 19100) 
v_pack_b32_f16 v15, v6, v7 // (op 19100) 

s_lshl_b32 s0, s4, 5 // (op 19100) 
s_mul_i32 s1, 7, s0 // (op 19100) 
s_add_u32 s1, s1, 192 // (op 19100) 

v_add_u32 v0, v13, s1 // (op 19100) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 19100) Store value



v_accvgpr_read v4, a224 // (op 19230) 
v_accvgpr_read v5, a225 // (op 19230) 
v_accvgpr_read v6, a226 // (op 19230) 
v_accvgpr_read v7, a227 // (op 19230) 
v_mul_f32 v14, v11, v4 // (op 19230) 
v_mul_f32 v15, v11, v5 // (op 19230) 
v_mul_f32 v16, v11, v6 // (op 19230) 
v_mul_f32 v17, v11, v7 // (op 19230) 

v_lshrrev_b32 v4, 16, v14 // (op 19232) 
v_lshrrev_b32 v5, 16, v15 // (op 19232) 
v_lshrrev_b32 v6, 16, v16 // (op 19232) 
v_lshrrev_b32 v7, 16, v17 // (op 19232) 


v_pack_b32_f16 v14, v4, v5 // (op 19234) 
v_pack_b32_f16 v15, v6, v7 // (op 19234) 

v_add_u32 v0, 224, v13 // (op 19234) 

buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 19234) Store value



v_accvgpr_read v4, a228 // (op 19252) 
v_accvgpr_read v5, a229 // (op 19252) 
v_accvgpr_read v6, a230 // (op 19252) 
v_accvgpr_read v7, a231 // (op 19252) 
v_mul_f32 v14, v11, v4 // (op 19252) 
v_mul_f32 v15, v11, v5 // (op 19252) 
v_mul_f32 v16, v11, v6 // (op 19252) 
v_mul_f32 v17, v11, v7 // (op 19252) 

v_lshrrev_b32 v4, 16, v14 // (op 19254) 
v_lshrrev_b32 v5, 16, v15 // (op 19254) 
v_lshrrev_b32 v6, 16, v16 // (op 19254) 
v_lshrrev_b32 v7, 16, v17 // (op 19254) 


v_pack_b32_f16 v14, v4, v5 // (op 19236) 
v_pack_b32_f16 v15, v6, v7 // (op 19236) 

s_lshl_b32 s0, s4, 5 // (op 19236) 
s_add_u32 s0, s0, 224 // (op 19236) 
v_add_u32 v0, v13, s0 // (op 19236) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 19236) Store value



v_accvgpr_read v4, a232 // (op 19258) 
v_accvgpr_read v5, a233 // (op 19258) 
v_accvgpr_read v6, a234 // (op 19258) 
v_accvgpr_read v7, a235 // (op 19258) 
v_mul_f32 v14, v11, v4 // (op 19258) 
v_mul_f32 v15, v11, v5 // (op 19258) 
v_mul_f32 v16, v11, v6 // (op 19258) 
v_mul_f32 v17, v11, v7 // (op 19258) 

v_lshrrev_b32 v4, 16, v14 // (op 19260) 
v_lshrrev_b32 v5, 16, v15 // (op 19260) 
v_lshrrev_b32 v6, 16, v16 // (op 19260) 
v_lshrrev_b32 v7, 16, v17 // (op 19260) 


v_pack_b32_f16 v14, v4, v5 // (op 19238) 
v_pack_b32_f16 v15, v6, v7 // (op 19238) 

s_lshl_b32 s0, s4, 6 // (op 19238) 
s_add_u32 s0, s0, 224 // (op 19238) 
v_add_u32 v0, v13, s0 // (op 19238) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 19238) Store value



v_accvgpr_read v4, a236 // (op 19264) 
v_accvgpr_read v5, a237 // (op 19264) 
v_accvgpr_read v6, a238 // (op 19264) 
v_accvgpr_read v7, a239 // (op 19264) 
v_mul_f32 v14, v11, v4 // (op 19264) 
v_mul_f32 v15, v11, v5 // (op 19264) 
v_mul_f32 v16, v11, v6 // (op 19264) 
v_mul_f32 v17, v11, v7 // (op 19264) 

v_lshrrev_b32 v4, 16, v14 // (op 19266) 
v_lshrrev_b32 v5, 16, v15 // (op 19266) 
v_lshrrev_b32 v6, 16, v16 // (op 19266) 
v_lshrrev_b32 v7, 16, v17 // (op 19266) 


v_pack_b32_f16 v14, v4, v5 // (op 19240) 
v_pack_b32_f16 v15, v6, v7 // (op 19240) 

s_lshl_b32 s0, s4, 5 // (op 19240) 
s_mul_i32 s1, 3, s0 // (op 19240) 
s_add_u32 s1, s1, 224 // (op 19240) 

v_add_u32 v0, v13, s1 // (op 19240) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 19240) Store value



v_accvgpr_read v4, a240 // (op 19270) 
v_accvgpr_read v5, a241 // (op 19270) 
v_accvgpr_read v6, a242 // (op 19270) 
v_accvgpr_read v7, a243 // (op 19270) 
v_mul_f32 v14, v11, v4 // (op 19270) 
v_mul_f32 v15, v11, v5 // (op 19270) 
v_mul_f32 v16, v11, v6 // (op 19270) 
v_mul_f32 v17, v11, v7 // (op 19270) 

v_lshrrev_b32 v4, 16, v14 // (op 19272) 
v_lshrrev_b32 v5, 16, v15 // (op 19272) 
v_lshrrev_b32 v6, 16, v16 // (op 19272) 
v_lshrrev_b32 v7, 16, v17 // (op 19272) 


v_pack_b32_f16 v14, v4, v5 // (op 19242) 
v_pack_b32_f16 v15, v6, v7 // (op 19242) 

s_lshl_b32 s0, s4, 7 // (op 19242) 
s_add_u32 s0, s0, 224 // (op 19242) 
v_add_u32 v0, v13, s0 // (op 19242) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 19242) Store value



v_accvgpr_read v4, a244 // (op 19276) 
v_accvgpr_read v5, a245 // (op 19276) 
v_accvgpr_read v6, a246 // (op 19276) 
v_accvgpr_read v7, a247 // (op 19276) 
v_mul_f32 v14, v11, v4 // (op 19276) 
v_mul_f32 v15, v11, v5 // (op 19276) 
v_mul_f32 v16, v11, v6 // (op 19276) 
v_mul_f32 v17, v11, v7 // (op 19276) 

v_lshrrev_b32 v4, 16, v14 // (op 19278) 
v_lshrrev_b32 v5, 16, v15 // (op 19278) 
v_lshrrev_b32 v6, 16, v16 // (op 19278) 
v_lshrrev_b32 v7, 16, v17 // (op 19278) 


v_pack_b32_f16 v14, v4, v5 // (op 19244) 
v_pack_b32_f16 v15, v6, v7 // (op 19244) 

s_lshl_b32 s0, s4, 5 // (op 19244) 
s_mul_i32 s1, 5, s0 // (op 19244) 
s_add_u32 s1, s1, 224 // (op 19244) 

v_add_u32 v0, v13, s1 // (op 19244) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 19244) Store value



v_accvgpr_read v4, a248 // (op 19282) 
v_accvgpr_read v5, a249 // (op 19282) 
v_accvgpr_read v6, a250 // (op 19282) 
v_accvgpr_read v7, a251 // (op 19282) 
v_mul_f32 v14, v11, v4 // (op 19282) 
v_mul_f32 v15, v11, v5 // (op 19282) 
v_mul_f32 v16, v11, v6 // (op 19282) 
v_mul_f32 v17, v11, v7 // (op 19282) 

v_lshrrev_b32 v4, 16, v14 // (op 19284) 
v_lshrrev_b32 v5, 16, v15 // (op 19284) 
v_lshrrev_b32 v6, 16, v16 // (op 19284) 
v_lshrrev_b32 v7, 16, v17 // (op 19284) 


v_pack_b32_f16 v14, v4, v5 // (op 19246) 
v_pack_b32_f16 v15, v6, v7 // (op 19246) 

s_lshl_b32 s0, s4, 5 // (op 19246) 
s_mul_i32 s1, 6, s0 // (op 19246) 
s_add_u32 s1, s1, 224 // (op 19246) 

v_add_u32 v0, v13, s1 // (op 19246) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 19246) Store value



v_accvgpr_read v4, a252 // (op 19288) 
v_accvgpr_read v5, a253 // (op 19288) 
v_accvgpr_read v6, a254 // (op 19288) 
v_accvgpr_read v7, a255 // (op 19288) 
v_mul_f32 v14, v11, v4 // (op 19288) 
v_mul_f32 v15, v11, v5 // (op 19288) 
v_mul_f32 v16, v11, v6 // (op 19288) 
v_mul_f32 v17, v11, v7 // (op 19288) 

v_lshrrev_b32 v4, 16, v14 // (op 19290) 
v_lshrrev_b32 v5, 16, v15 // (op 19290) 
v_lshrrev_b32 v6, 16, v16 // (op 19290) 
v_lshrrev_b32 v7, 16, v17 // (op 19290) 


v_pack_b32_f16 v14, v4, v5 // (op 19248) 
v_pack_b32_f16 v15, v6, v7 // (op 19248) 

s_lshl_b32 s0, s4, 5 // (op 19248) 
s_mul_i32 s1, 7, s0 // (op 19248) 
s_add_u32 s1, s1, 224 // (op 19248) 

v_add_u32 v0, v13, s1 // (op 19248) 


buffer_store_dwordx2 v[14:15], v0, s[56:59], 0 offen // (op 19248) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WGM__ConditionalBottom_0_beta_116 // (op 116) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WGM__ConditionalBottom_0_beta_116
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WGM__ConditionalFalse_0_beta_116:


v_add_u32 v0, 224, v12 // (op 20808) 

buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20808) Load value

v_accvgpr_read v14, a224 // (op 20787) 
v_accvgpr_read v15, a225 // (op 20787) 
v_accvgpr_read v16, a226 // (op 20787) 
v_accvgpr_read v17, a227 // (op 20787) 
v_mul_f32 v18, v11, v14 // (op 20787) 
v_mul_f32 v19, v11, v15 // (op 20787) 
v_mul_f32 v20, v11, v16 // (op 20787) 
v_mul_f32 v21, v11, v17 // (op 20787) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20807) 
v_lshrrev_b32 v7, 16, v4 // (op 20807) 
v_lshlrev_b32 v6, 16, v6 // (op 20807) 
v_lshlrev_b32 v7, 16, v7 // (op 20807) 
v_mul_f32 v14, v9, v6 // (op 20807) 
v_mul_f32 v15, v9, v7 // (op 20807) 
v_and_b32 v6, 65535, v5 // (op 20807) 
v_lshrrev_b32 v7, 16, v5 // (op 20807) 
v_lshlrev_b32 v6, 16, v6 // (op 20807) 
v_lshlrev_b32 v7, 16, v7 // (op 20807) 
v_mul_f32 v16, v9, v6 // (op 20807) 
v_mul_f32 v17, v9, v7 // (op 20807) 


v_add_f32 v4, v14, v18 // (op 20788) 
v_add_f32 v5, v15, v19 // (op 20788) 
v_add_f32 v6, v16, v20 // (op 20788) 
v_add_f32 v7, v17, v21 // (op 20788) 


v_lshrrev_b32 v14, 16, v4 // (op 20789) 
v_lshrrev_b32 v15, 16, v5 // (op 20789) 
v_lshrrev_b32 v16, 16, v6 // (op 20789) 
v_lshrrev_b32 v17, 16, v7 // (op 20789) 


v_pack_b32_f16 v4, v14, v15 // (op 20791) 
v_pack_b32_f16 v5, v16, v17 // (op 20791) 

v_add_u32 v0, 224, v13 // (op 20791) 

buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20791) Store value





s_lshl_b32 s0, s2, 5 // (op 20813) 
s_add_u32 s0, s0, 224 // (op 20813) 
v_add_u32 v0, v12, s0 // (op 20813) 


buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20813) Load value

v_accvgpr_read v14, a228 // (op 20845) 
v_accvgpr_read v15, a229 // (op 20845) 
v_accvgpr_read v16, a230 // (op 20845) 
v_accvgpr_read v17, a231 // (op 20845) 
v_mul_f32 v18, v11, v14 // (op 20845) 
v_mul_f32 v19, v11, v15 // (op 20845) 
v_mul_f32 v20, v11, v16 // (op 20845) 
v_mul_f32 v21, v11, v17 // (op 20845) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20810) 
v_lshrrev_b32 v7, 16, v4 // (op 20810) 
v_lshlrev_b32 v6, 16, v6 // (op 20810) 
v_lshlrev_b32 v7, 16, v7 // (op 20810) 
v_mul_f32 v14, v9, v6 // (op 20810) 
v_mul_f32 v15, v9, v7 // (op 20810) 
v_and_b32 v6, 65535, v5 // (op 20810) 
v_lshrrev_b32 v7, 16, v5 // (op 20810) 
v_lshlrev_b32 v6, 16, v6 // (op 20810) 
v_lshlrev_b32 v7, 16, v7 // (op 20810) 
v_mul_f32 v16, v9, v6 // (op 20810) 
v_mul_f32 v17, v9, v7 // (op 20810) 


v_add_f32 v4, v14, v18 // (op 20811) 
v_add_f32 v5, v15, v19 // (op 20811) 
v_add_f32 v6, v16, v20 // (op 20811) 
v_add_f32 v7, v17, v21 // (op 20811) 


v_lshrrev_b32 v14, 16, v4 // (op 20812) 
v_lshrrev_b32 v15, 16, v5 // (op 20812) 
v_lshrrev_b32 v16, 16, v6 // (op 20812) 
v_lshrrev_b32 v17, 16, v7 // (op 20812) 


v_pack_b32_f16 v4, v14, v15 // (op 20793) 
v_pack_b32_f16 v5, v16, v17 // (op 20793) 

s_lshl_b32 s0, s4, 5 // (op 20793) 
s_add_u32 s0, s0, 224 // (op 20793) 
v_add_u32 v0, v13, s0 // (op 20793) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20793) Store value





s_lshl_b32 s0, s2, 6 // (op 20818) 
s_add_u32 s0, s0, 224 // (op 20818) 
v_add_u32 v0, v12, s0 // (op 20818) 


buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20818) Load value

v_accvgpr_read v14, a232 // (op 20847) 
v_accvgpr_read v15, a233 // (op 20847) 
v_accvgpr_read v16, a234 // (op 20847) 
v_accvgpr_read v17, a235 // (op 20847) 
v_mul_f32 v18, v11, v14 // (op 20847) 
v_mul_f32 v19, v11, v15 // (op 20847) 
v_mul_f32 v20, v11, v16 // (op 20847) 
v_mul_f32 v21, v11, v17 // (op 20847) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20815) 
v_lshrrev_b32 v7, 16, v4 // (op 20815) 
v_lshlrev_b32 v6, 16, v6 // (op 20815) 
v_lshlrev_b32 v7, 16, v7 // (op 20815) 
v_mul_f32 v14, v9, v6 // (op 20815) 
v_mul_f32 v15, v9, v7 // (op 20815) 
v_and_b32 v6, 65535, v5 // (op 20815) 
v_lshrrev_b32 v7, 16, v5 // (op 20815) 
v_lshlrev_b32 v6, 16, v6 // (op 20815) 
v_lshlrev_b32 v7, 16, v7 // (op 20815) 
v_mul_f32 v16, v9, v6 // (op 20815) 
v_mul_f32 v17, v9, v7 // (op 20815) 


v_add_f32 v4, v14, v18 // (op 20816) 
v_add_f32 v5, v15, v19 // (op 20816) 
v_add_f32 v6, v16, v20 // (op 20816) 
v_add_f32 v7, v17, v21 // (op 20816) 


v_lshrrev_b32 v14, 16, v4 // (op 20817) 
v_lshrrev_b32 v15, 16, v5 // (op 20817) 
v_lshrrev_b32 v16, 16, v6 // (op 20817) 
v_lshrrev_b32 v17, 16, v7 // (op 20817) 


v_pack_b32_f16 v4, v14, v15 // (op 20795) 
v_pack_b32_f16 v5, v16, v17 // (op 20795) 

s_lshl_b32 s0, s4, 6 // (op 20795) 
s_add_u32 s0, s0, 224 // (op 20795) 
v_add_u32 v0, v13, s0 // (op 20795) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20795) Store value





s_lshl_b32 s0, s2, 5 // (op 20823) 
s_mul_i32 s1, 3, s0 // (op 20823) 
s_add_u32 s1, s1, 224 // (op 20823) 

v_add_u32 v0, v12, s1 // (op 20823) 


buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20823) Load value

v_accvgpr_read v14, a236 // (op 20849) 
v_accvgpr_read v15, a237 // (op 20849) 
v_accvgpr_read v16, a238 // (op 20849) 
v_accvgpr_read v17, a239 // (op 20849) 
v_mul_f32 v18, v11, v14 // (op 20849) 
v_mul_f32 v19, v11, v15 // (op 20849) 
v_mul_f32 v20, v11, v16 // (op 20849) 
v_mul_f32 v21, v11, v17 // (op 20849) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20820) 
v_lshrrev_b32 v7, 16, v4 // (op 20820) 
v_lshlrev_b32 v6, 16, v6 // (op 20820) 
v_lshlrev_b32 v7, 16, v7 // (op 20820) 
v_mul_f32 v14, v9, v6 // (op 20820) 
v_mul_f32 v15, v9, v7 // (op 20820) 
v_and_b32 v6, 65535, v5 // (op 20820) 
v_lshrrev_b32 v7, 16, v5 // (op 20820) 
v_lshlrev_b32 v6, 16, v6 // (op 20820) 
v_lshlrev_b32 v7, 16, v7 // (op 20820) 
v_mul_f32 v16, v9, v6 // (op 20820) 
v_mul_f32 v17, v9, v7 // (op 20820) 


v_add_f32 v4, v14, v18 // (op 20821) 
v_add_f32 v5, v15, v19 // (op 20821) 
v_add_f32 v6, v16, v20 // (op 20821) 
v_add_f32 v7, v17, v21 // (op 20821) 


v_lshrrev_b32 v14, 16, v4 // (op 20822) 
v_lshrrev_b32 v15, 16, v5 // (op 20822) 
v_lshrrev_b32 v16, 16, v6 // (op 20822) 
v_lshrrev_b32 v17, 16, v7 // (op 20822) 


v_pack_b32_f16 v4, v14, v15 // (op 20797) 
v_pack_b32_f16 v5, v16, v17 // (op 20797) 

s_lshl_b32 s0, s4, 5 // (op 20797) 
s_mul_i32 s1, 3, s0 // (op 20797) 
s_add_u32 s1, s1, 224 // (op 20797) 

v_add_u32 v0, v13, s1 // (op 20797) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20797) Store value





s_lshl_b32 s0, s2, 7 // (op 20828) 
s_add_u32 s0, s0, 224 // (op 20828) 
v_add_u32 v0, v12, s0 // (op 20828) 


buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20828) Load value

v_accvgpr_read v14, a240 // (op 20851) 
v_accvgpr_read v15, a241 // (op 20851) 
v_accvgpr_read v16, a242 // (op 20851) 
v_accvgpr_read v17, a243 // (op 20851) 
v_mul_f32 v18, v11, v14 // (op 20851) 
v_mul_f32 v19, v11, v15 // (op 20851) 
v_mul_f32 v20, v11, v16 // (op 20851) 
v_mul_f32 v21, v11, v17 // (op 20851) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20825) 
v_lshrrev_b32 v7, 16, v4 // (op 20825) 
v_lshlrev_b32 v6, 16, v6 // (op 20825) 
v_lshlrev_b32 v7, 16, v7 // (op 20825) 
v_mul_f32 v14, v9, v6 // (op 20825) 
v_mul_f32 v15, v9, v7 // (op 20825) 
v_and_b32 v6, 65535, v5 // (op 20825) 
v_lshrrev_b32 v7, 16, v5 // (op 20825) 
v_lshlrev_b32 v6, 16, v6 // (op 20825) 
v_lshlrev_b32 v7, 16, v7 // (op 20825) 
v_mul_f32 v16, v9, v6 // (op 20825) 
v_mul_f32 v17, v9, v7 // (op 20825) 


v_add_f32 v4, v14, v18 // (op 20826) 
v_add_f32 v5, v15, v19 // (op 20826) 
v_add_f32 v6, v16, v20 // (op 20826) 
v_add_f32 v7, v17, v21 // (op 20826) 


v_lshrrev_b32 v14, 16, v4 // (op 20827) 
v_lshrrev_b32 v15, 16, v5 // (op 20827) 
v_lshrrev_b32 v16, 16, v6 // (op 20827) 
v_lshrrev_b32 v17, 16, v7 // (op 20827) 


v_pack_b32_f16 v4, v14, v15 // (op 20799) 
v_pack_b32_f16 v5, v16, v17 // (op 20799) 

s_lshl_b32 s0, s4, 7 // (op 20799) 
s_add_u32 s0, s0, 224 // (op 20799) 
v_add_u32 v0, v13, s0 // (op 20799) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20799) Store value





s_lshl_b32 s0, s2, 5 // (op 20833) 
s_mul_i32 s1, 5, s0 // (op 20833) 
s_add_u32 s1, s1, 224 // (op 20833) 

v_add_u32 v0, v12, s1 // (op 20833) 


buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20833) Load value

v_accvgpr_read v14, a244 // (op 20853) 
v_accvgpr_read v15, a245 // (op 20853) 
v_accvgpr_read v16, a246 // (op 20853) 
v_accvgpr_read v17, a247 // (op 20853) 
v_mul_f32 v18, v11, v14 // (op 20853) 
v_mul_f32 v19, v11, v15 // (op 20853) 
v_mul_f32 v20, v11, v16 // (op 20853) 
v_mul_f32 v21, v11, v17 // (op 20853) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20830) 
v_lshrrev_b32 v7, 16, v4 // (op 20830) 
v_lshlrev_b32 v6, 16, v6 // (op 20830) 
v_lshlrev_b32 v7, 16, v7 // (op 20830) 
v_mul_f32 v14, v9, v6 // (op 20830) 
v_mul_f32 v15, v9, v7 // (op 20830) 
v_and_b32 v6, 65535, v5 // (op 20830) 
v_lshrrev_b32 v7, 16, v5 // (op 20830) 
v_lshlrev_b32 v6, 16, v6 // (op 20830) 
v_lshlrev_b32 v7, 16, v7 // (op 20830) 
v_mul_f32 v16, v9, v6 // (op 20830) 
v_mul_f32 v17, v9, v7 // (op 20830) 


v_add_f32 v4, v14, v18 // (op 20831) 
v_add_f32 v5, v15, v19 // (op 20831) 
v_add_f32 v6, v16, v20 // (op 20831) 
v_add_f32 v7, v17, v21 // (op 20831) 


v_lshrrev_b32 v14, 16, v4 // (op 20832) 
v_lshrrev_b32 v15, 16, v5 // (op 20832) 
v_lshrrev_b32 v16, 16, v6 // (op 20832) 
v_lshrrev_b32 v17, 16, v7 // (op 20832) 


v_pack_b32_f16 v4, v14, v15 // (op 20801) 
v_pack_b32_f16 v5, v16, v17 // (op 20801) 

s_lshl_b32 s0, s4, 5 // (op 20801) 
s_mul_i32 s1, 5, s0 // (op 20801) 
s_add_u32 s1, s1, 224 // (op 20801) 

v_add_u32 v0, v13, s1 // (op 20801) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20801) Store value





s_lshl_b32 s0, s2, 5 // (op 20838) 
s_mul_i32 s1, 6, s0 // (op 20838) 
s_add_u32 s1, s1, 224 // (op 20838) 

v_add_u32 v0, v12, s1 // (op 20838) 


buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20838) Load value

v_accvgpr_read v14, a248 // (op 20855) 
v_accvgpr_read v15, a249 // (op 20855) 
v_accvgpr_read v16, a250 // (op 20855) 
v_accvgpr_read v17, a251 // (op 20855) 
v_mul_f32 v18, v11, v14 // (op 20855) 
v_mul_f32 v19, v11, v15 // (op 20855) 
v_mul_f32 v20, v11, v16 // (op 20855) 
v_mul_f32 v21, v11, v17 // (op 20855) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20835) 
v_lshrrev_b32 v7, 16, v4 // (op 20835) 
v_lshlrev_b32 v6, 16, v6 // (op 20835) 
v_lshlrev_b32 v7, 16, v7 // (op 20835) 
v_mul_f32 v14, v9, v6 // (op 20835) 
v_mul_f32 v15, v9, v7 // (op 20835) 
v_and_b32 v6, 65535, v5 // (op 20835) 
v_lshrrev_b32 v7, 16, v5 // (op 20835) 
v_lshlrev_b32 v6, 16, v6 // (op 20835) 
v_lshlrev_b32 v7, 16, v7 // (op 20835) 
v_mul_f32 v16, v9, v6 // (op 20835) 
v_mul_f32 v17, v9, v7 // (op 20835) 


v_add_f32 v4, v14, v18 // (op 20836) 
v_add_f32 v5, v15, v19 // (op 20836) 
v_add_f32 v6, v16, v20 // (op 20836) 
v_add_f32 v7, v17, v21 // (op 20836) 


v_lshrrev_b32 v14, 16, v4 // (op 20837) 
v_lshrrev_b32 v15, 16, v5 // (op 20837) 
v_lshrrev_b32 v16, 16, v6 // (op 20837) 
v_lshrrev_b32 v17, 16, v7 // (op 20837) 


v_pack_b32_f16 v4, v14, v15 // (op 20803) 
v_pack_b32_f16 v5, v16, v17 // (op 20803) 

s_lshl_b32 s0, s4, 5 // (op 20803) 
s_mul_i32 s1, 6, s0 // (op 20803) 
s_add_u32 s1, s1, 224 // (op 20803) 

v_add_u32 v0, v13, s1 // (op 20803) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20803) Store value





s_lshl_b32 s0, s2, 5 // (op 20843) 
s_mul_i32 s1, 7, s0 // (op 20843) 
s_add_u32 s1, s1, 224 // (op 20843) 

v_add_u32 v0, v12, s1 // (op 20843) 


buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20843) Load value

v_accvgpr_read v14, a252 // (op 20857) 
v_accvgpr_read v15, a253 // (op 20857) 
v_accvgpr_read v16, a254 // (op 20857) 
v_accvgpr_read v17, a255 // (op 20857) 
v_mul_f32 v18, v11, v14 // (op 20857) 
v_mul_f32 v19, v11, v15 // (op 20857) 
v_mul_f32 v20, v11, v16 // (op 20857) 
v_mul_f32 v21, v11, v17 // (op 20857) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20840) 
v_lshrrev_b32 v7, 16, v4 // (op 20840) 
v_lshlrev_b32 v6, 16, v6 // (op 20840) 
v_lshlrev_b32 v7, 16, v7 // (op 20840) 
v_mul_f32 v14, v9, v6 // (op 20840) 
v_mul_f32 v15, v9, v7 // (op 20840) 
v_and_b32 v6, 65535, v5 // (op 20840) 
v_lshrrev_b32 v7, 16, v5 // (op 20840) 
v_lshlrev_b32 v6, 16, v6 // (op 20840) 
v_lshlrev_b32 v7, 16, v7 // (op 20840) 
v_mul_f32 v16, v9, v6 // (op 20840) 
v_mul_f32 v17, v9, v7 // (op 20840) 


v_add_f32 v4, v14, v18 // (op 20841) 
v_add_f32 v5, v15, v19 // (op 20841) 
v_add_f32 v6, v16, v20 // (op 20841) 
v_add_f32 v7, v17, v21 // (op 20841) 


v_lshrrev_b32 v14, 16, v4 // (op 20842) 
v_lshrrev_b32 v15, 16, v5 // (op 20842) 
v_lshrrev_b32 v16, 16, v6 // (op 20842) 
v_lshrrev_b32 v17, 16, v7 // (op 20842) 


v_pack_b32_f16 v4, v14, v15 // (op 20805) 
v_pack_b32_f16 v5, v16, v17 // (op 20805) 

s_lshl_b32 s0, s4, 5 // (op 20805) 
s_mul_i32 s1, 7, s0 // (op 20805) 
s_add_u32 s1, s1, 224 // (op 20805) 

v_add_u32 v0, v13, s1 // (op 20805) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20805) Store value





v_add_u32 v0, 192, v12 // (op 20637) 

buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20637) Load value

v_accvgpr_read v14, a192 // (op 20616) 
v_accvgpr_read v15, a193 // (op 20616) 
v_accvgpr_read v16, a194 // (op 20616) 
v_accvgpr_read v17, a195 // (op 20616) 
v_mul_f32 v18, v11, v14 // (op 20616) 
v_mul_f32 v19, v11, v15 // (op 20616) 
v_mul_f32 v20, v11, v16 // (op 20616) 
v_mul_f32 v21, v11, v17 // (op 20616) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20636) 
v_lshrrev_b32 v7, 16, v4 // (op 20636) 
v_lshlrev_b32 v6, 16, v6 // (op 20636) 
v_lshlrev_b32 v7, 16, v7 // (op 20636) 
v_mul_f32 v14, v9, v6 // (op 20636) 
v_mul_f32 v15, v9, v7 // (op 20636) 
v_and_b32 v6, 65535, v5 // (op 20636) 
v_lshrrev_b32 v7, 16, v5 // (op 20636) 
v_lshlrev_b32 v6, 16, v6 // (op 20636) 
v_lshlrev_b32 v7, 16, v7 // (op 20636) 
v_mul_f32 v16, v9, v6 // (op 20636) 
v_mul_f32 v17, v9, v7 // (op 20636) 


v_add_f32 v4, v14, v18 // (op 20617) 
v_add_f32 v5, v15, v19 // (op 20617) 
v_add_f32 v6, v16, v20 // (op 20617) 
v_add_f32 v7, v17, v21 // (op 20617) 


v_lshrrev_b32 v14, 16, v4 // (op 20618) 
v_lshrrev_b32 v15, 16, v5 // (op 20618) 
v_lshrrev_b32 v16, 16, v6 // (op 20618) 
v_lshrrev_b32 v17, 16, v7 // (op 20618) 


v_pack_b32_f16 v4, v14, v15 // (op 20620) 
v_pack_b32_f16 v5, v16, v17 // (op 20620) 

v_add_u32 v0, 192, v13 // (op 20620) 

buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20620) Store value





s_lshl_b32 s0, s2, 5 // (op 20642) 
s_add_u32 s0, s0, 192 // (op 20642) 
v_add_u32 v0, v12, s0 // (op 20642) 


buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20642) Load value

v_accvgpr_read v14, a196 // (op 20674) 
v_accvgpr_read v15, a197 // (op 20674) 
v_accvgpr_read v16, a198 // (op 20674) 
v_accvgpr_read v17, a199 // (op 20674) 
v_mul_f32 v18, v11, v14 // (op 20674) 
v_mul_f32 v19, v11, v15 // (op 20674) 
v_mul_f32 v20, v11, v16 // (op 20674) 
v_mul_f32 v21, v11, v17 // (op 20674) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20639) 
v_lshrrev_b32 v7, 16, v4 // (op 20639) 
v_lshlrev_b32 v6, 16, v6 // (op 20639) 
v_lshlrev_b32 v7, 16, v7 // (op 20639) 
v_mul_f32 v14, v9, v6 // (op 20639) 
v_mul_f32 v15, v9, v7 // (op 20639) 
v_and_b32 v6, 65535, v5 // (op 20639) 
v_lshrrev_b32 v7, 16, v5 // (op 20639) 
v_lshlrev_b32 v6, 16, v6 // (op 20639) 
v_lshlrev_b32 v7, 16, v7 // (op 20639) 
v_mul_f32 v16, v9, v6 // (op 20639) 
v_mul_f32 v17, v9, v7 // (op 20639) 


v_add_f32 v4, v14, v18 // (op 20640) 
v_add_f32 v5, v15, v19 // (op 20640) 
v_add_f32 v6, v16, v20 // (op 20640) 
v_add_f32 v7, v17, v21 // (op 20640) 


v_lshrrev_b32 v14, 16, v4 // (op 20641) 
v_lshrrev_b32 v15, 16, v5 // (op 20641) 
v_lshrrev_b32 v16, 16, v6 // (op 20641) 
v_lshrrev_b32 v17, 16, v7 // (op 20641) 


v_pack_b32_f16 v4, v14, v15 // (op 20622) 
v_pack_b32_f16 v5, v16, v17 // (op 20622) 

s_lshl_b32 s0, s4, 5 // (op 20622) 
s_add_u32 s0, s0, 192 // (op 20622) 
v_add_u32 v0, v13, s0 // (op 20622) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20622) Store value





s_lshl_b32 s0, s2, 6 // (op 20647) 
s_add_u32 s0, s0, 192 // (op 20647) 
v_add_u32 v0, v12, s0 // (op 20647) 


buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20647) Load value

v_accvgpr_read v14, a200 // (op 20676) 
v_accvgpr_read v15, a201 // (op 20676) 
v_accvgpr_read v16, a202 // (op 20676) 
v_accvgpr_read v17, a203 // (op 20676) 
v_mul_f32 v18, v11, v14 // (op 20676) 
v_mul_f32 v19, v11, v15 // (op 20676) 
v_mul_f32 v20, v11, v16 // (op 20676) 
v_mul_f32 v21, v11, v17 // (op 20676) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20644) 
v_lshrrev_b32 v7, 16, v4 // (op 20644) 
v_lshlrev_b32 v6, 16, v6 // (op 20644) 
v_lshlrev_b32 v7, 16, v7 // (op 20644) 
v_mul_f32 v14, v9, v6 // (op 20644) 
v_mul_f32 v15, v9, v7 // (op 20644) 
v_and_b32 v6, 65535, v5 // (op 20644) 
v_lshrrev_b32 v7, 16, v5 // (op 20644) 
v_lshlrev_b32 v6, 16, v6 // (op 20644) 
v_lshlrev_b32 v7, 16, v7 // (op 20644) 
v_mul_f32 v16, v9, v6 // (op 20644) 
v_mul_f32 v17, v9, v7 // (op 20644) 


v_add_f32 v4, v14, v18 // (op 20645) 
v_add_f32 v5, v15, v19 // (op 20645) 
v_add_f32 v6, v16, v20 // (op 20645) 
v_add_f32 v7, v17, v21 // (op 20645) 


v_lshrrev_b32 v14, 16, v4 // (op 20646) 
v_lshrrev_b32 v15, 16, v5 // (op 20646) 
v_lshrrev_b32 v16, 16, v6 // (op 20646) 
v_lshrrev_b32 v17, 16, v7 // (op 20646) 


v_pack_b32_f16 v4, v14, v15 // (op 20624) 
v_pack_b32_f16 v5, v16, v17 // (op 20624) 

s_lshl_b32 s0, s4, 6 // (op 20624) 
s_add_u32 s0, s0, 192 // (op 20624) 
v_add_u32 v0, v13, s0 // (op 20624) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20624) Store value





s_lshl_b32 s0, s2, 5 // (op 20652) 
s_mul_i32 s1, 3, s0 // (op 20652) 
s_add_u32 s1, s1, 192 // (op 20652) 

v_add_u32 v0, v12, s1 // (op 20652) 


buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20652) Load value

v_accvgpr_read v14, a204 // (op 20678) 
v_accvgpr_read v15, a205 // (op 20678) 
v_accvgpr_read v16, a206 // (op 20678) 
v_accvgpr_read v17, a207 // (op 20678) 
v_mul_f32 v18, v11, v14 // (op 20678) 
v_mul_f32 v19, v11, v15 // (op 20678) 
v_mul_f32 v20, v11, v16 // (op 20678) 
v_mul_f32 v21, v11, v17 // (op 20678) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20649) 
v_lshrrev_b32 v7, 16, v4 // (op 20649) 
v_lshlrev_b32 v6, 16, v6 // (op 20649) 
v_lshlrev_b32 v7, 16, v7 // (op 20649) 
v_mul_f32 v14, v9, v6 // (op 20649) 
v_mul_f32 v15, v9, v7 // (op 20649) 
v_and_b32 v6, 65535, v5 // (op 20649) 
v_lshrrev_b32 v7, 16, v5 // (op 20649) 
v_lshlrev_b32 v6, 16, v6 // (op 20649) 
v_lshlrev_b32 v7, 16, v7 // (op 20649) 
v_mul_f32 v16, v9, v6 // (op 20649) 
v_mul_f32 v17, v9, v7 // (op 20649) 


v_add_f32 v4, v14, v18 // (op 20650) 
v_add_f32 v5, v15, v19 // (op 20650) 
v_add_f32 v6, v16, v20 // (op 20650) 
v_add_f32 v7, v17, v21 // (op 20650) 


v_lshrrev_b32 v14, 16, v4 // (op 20651) 
v_lshrrev_b32 v15, 16, v5 // (op 20651) 
v_lshrrev_b32 v16, 16, v6 // (op 20651) 
v_lshrrev_b32 v17, 16, v7 // (op 20651) 


v_pack_b32_f16 v4, v14, v15 // (op 20626) 
v_pack_b32_f16 v5, v16, v17 // (op 20626) 

s_lshl_b32 s0, s4, 5 // (op 20626) 
s_mul_i32 s1, 3, s0 // (op 20626) 
s_add_u32 s1, s1, 192 // (op 20626) 

v_add_u32 v0, v13, s1 // (op 20626) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20626) Store value





s_lshl_b32 s0, s2, 7 // (op 20657) 
s_add_u32 s0, s0, 192 // (op 20657) 
v_add_u32 v0, v12, s0 // (op 20657) 


buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20657) Load value

v_accvgpr_read v14, a208 // (op 20680) 
v_accvgpr_read v15, a209 // (op 20680) 
v_accvgpr_read v16, a210 // (op 20680) 
v_accvgpr_read v17, a211 // (op 20680) 
v_mul_f32 v18, v11, v14 // (op 20680) 
v_mul_f32 v19, v11, v15 // (op 20680) 
v_mul_f32 v20, v11, v16 // (op 20680) 
v_mul_f32 v21, v11, v17 // (op 20680) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20654) 
v_lshrrev_b32 v7, 16, v4 // (op 20654) 
v_lshlrev_b32 v6, 16, v6 // (op 20654) 
v_lshlrev_b32 v7, 16, v7 // (op 20654) 
v_mul_f32 v14, v9, v6 // (op 20654) 
v_mul_f32 v15, v9, v7 // (op 20654) 
v_and_b32 v6, 65535, v5 // (op 20654) 
v_lshrrev_b32 v7, 16, v5 // (op 20654) 
v_lshlrev_b32 v6, 16, v6 // (op 20654) 
v_lshlrev_b32 v7, 16, v7 // (op 20654) 
v_mul_f32 v16, v9, v6 // (op 20654) 
v_mul_f32 v17, v9, v7 // (op 20654) 


v_add_f32 v4, v14, v18 // (op 20655) 
v_add_f32 v5, v15, v19 // (op 20655) 
v_add_f32 v6, v16, v20 // (op 20655) 
v_add_f32 v7, v17, v21 // (op 20655) 


v_lshrrev_b32 v14, 16, v4 // (op 20656) 
v_lshrrev_b32 v15, 16, v5 // (op 20656) 
v_lshrrev_b32 v16, 16, v6 // (op 20656) 
v_lshrrev_b32 v17, 16, v7 // (op 20656) 


v_pack_b32_f16 v4, v14, v15 // (op 20628) 
v_pack_b32_f16 v5, v16, v17 // (op 20628) 

s_lshl_b32 s0, s4, 7 // (op 20628) 
s_add_u32 s0, s0, 192 // (op 20628) 
v_add_u32 v0, v13, s0 // (op 20628) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20628) Store value





s_lshl_b32 s0, s2, 5 // (op 20662) 
s_mul_i32 s1, 5, s0 // (op 20662) 
s_add_u32 s1, s1, 192 // (op 20662) 

v_add_u32 v0, v12, s1 // (op 20662) 


buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20662) Load value

v_accvgpr_read v14, a212 // (op 20682) 
v_accvgpr_read v15, a213 // (op 20682) 
v_accvgpr_read v16, a214 // (op 20682) 
v_accvgpr_read v17, a215 // (op 20682) 
v_mul_f32 v18, v11, v14 // (op 20682) 
v_mul_f32 v19, v11, v15 // (op 20682) 
v_mul_f32 v20, v11, v16 // (op 20682) 
v_mul_f32 v21, v11, v17 // (op 20682) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20659) 
v_lshrrev_b32 v7, 16, v4 // (op 20659) 
v_lshlrev_b32 v6, 16, v6 // (op 20659) 
v_lshlrev_b32 v7, 16, v7 // (op 20659) 
v_mul_f32 v14, v9, v6 // (op 20659) 
v_mul_f32 v15, v9, v7 // (op 20659) 
v_and_b32 v6, 65535, v5 // (op 20659) 
v_lshrrev_b32 v7, 16, v5 // (op 20659) 
v_lshlrev_b32 v6, 16, v6 // (op 20659) 
v_lshlrev_b32 v7, 16, v7 // (op 20659) 
v_mul_f32 v16, v9, v6 // (op 20659) 
v_mul_f32 v17, v9, v7 // (op 20659) 


v_add_f32 v4, v14, v18 // (op 20660) 
v_add_f32 v5, v15, v19 // (op 20660) 
v_add_f32 v6, v16, v20 // (op 20660) 
v_add_f32 v7, v17, v21 // (op 20660) 


v_lshrrev_b32 v14, 16, v4 // (op 20661) 
v_lshrrev_b32 v15, 16, v5 // (op 20661) 
v_lshrrev_b32 v16, 16, v6 // (op 20661) 
v_lshrrev_b32 v17, 16, v7 // (op 20661) 


v_pack_b32_f16 v4, v14, v15 // (op 20630) 
v_pack_b32_f16 v5, v16, v17 // (op 20630) 

s_lshl_b32 s0, s4, 5 // (op 20630) 
s_mul_i32 s1, 5, s0 // (op 20630) 
s_add_u32 s1, s1, 192 // (op 20630) 

v_add_u32 v0, v13, s1 // (op 20630) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20630) Store value





s_lshl_b32 s0, s2, 5 // (op 20667) 
s_mul_i32 s1, 6, s0 // (op 20667) 
s_add_u32 s1, s1, 192 // (op 20667) 

v_add_u32 v0, v12, s1 // (op 20667) 


buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20667) Load value

v_accvgpr_read v14, a216 // (op 20684) 
v_accvgpr_read v15, a217 // (op 20684) 
v_accvgpr_read v16, a218 // (op 20684) 
v_accvgpr_read v17, a219 // (op 20684) 
v_mul_f32 v18, v11, v14 // (op 20684) 
v_mul_f32 v19, v11, v15 // (op 20684) 
v_mul_f32 v20, v11, v16 // (op 20684) 
v_mul_f32 v21, v11, v17 // (op 20684) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20664) 
v_lshrrev_b32 v7, 16, v4 // (op 20664) 
v_lshlrev_b32 v6, 16, v6 // (op 20664) 
v_lshlrev_b32 v7, 16, v7 // (op 20664) 
v_mul_f32 v14, v9, v6 // (op 20664) 
v_mul_f32 v15, v9, v7 // (op 20664) 
v_and_b32 v6, 65535, v5 // (op 20664) 
v_lshrrev_b32 v7, 16, v5 // (op 20664) 
v_lshlrev_b32 v6, 16, v6 // (op 20664) 
v_lshlrev_b32 v7, 16, v7 // (op 20664) 
v_mul_f32 v16, v9, v6 // (op 20664) 
v_mul_f32 v17, v9, v7 // (op 20664) 


v_add_f32 v4, v14, v18 // (op 20665) 
v_add_f32 v5, v15, v19 // (op 20665) 
v_add_f32 v6, v16, v20 // (op 20665) 
v_add_f32 v7, v17, v21 // (op 20665) 


v_lshrrev_b32 v14, 16, v4 // (op 20666) 
v_lshrrev_b32 v15, 16, v5 // (op 20666) 
v_lshrrev_b32 v16, 16, v6 // (op 20666) 
v_lshrrev_b32 v17, 16, v7 // (op 20666) 


v_pack_b32_f16 v4, v14, v15 // (op 20632) 
v_pack_b32_f16 v5, v16, v17 // (op 20632) 

s_lshl_b32 s0, s4, 5 // (op 20632) 
s_mul_i32 s1, 6, s0 // (op 20632) 
s_add_u32 s1, s1, 192 // (op 20632) 

v_add_u32 v0, v13, s1 // (op 20632) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20632) Store value





s_lshl_b32 s0, s2, 5 // (op 20672) 
s_mul_i32 s1, 7, s0 // (op 20672) 
s_add_u32 s1, s1, 192 // (op 20672) 

v_add_u32 v0, v12, s1 // (op 20672) 


buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20672) Load value

v_accvgpr_read v14, a220 // (op 20686) 
v_accvgpr_read v15, a221 // (op 20686) 
v_accvgpr_read v16, a222 // (op 20686) 
v_accvgpr_read v17, a223 // (op 20686) 
v_mul_f32 v18, v11, v14 // (op 20686) 
v_mul_f32 v19, v11, v15 // (op 20686) 
v_mul_f32 v20, v11, v16 // (op 20686) 
v_mul_f32 v21, v11, v17 // (op 20686) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20669) 
v_lshrrev_b32 v7, 16, v4 // (op 20669) 
v_lshlrev_b32 v6, 16, v6 // (op 20669) 
v_lshlrev_b32 v7, 16, v7 // (op 20669) 
v_mul_f32 v14, v9, v6 // (op 20669) 
v_mul_f32 v15, v9, v7 // (op 20669) 
v_and_b32 v6, 65535, v5 // (op 20669) 
v_lshrrev_b32 v7, 16, v5 // (op 20669) 
v_lshlrev_b32 v6, 16, v6 // (op 20669) 
v_lshlrev_b32 v7, 16, v7 // (op 20669) 
v_mul_f32 v16, v9, v6 // (op 20669) 
v_mul_f32 v17, v9, v7 // (op 20669) 


v_add_f32 v4, v14, v18 // (op 20670) 
v_add_f32 v5, v15, v19 // (op 20670) 
v_add_f32 v6, v16, v20 // (op 20670) 
v_add_f32 v7, v17, v21 // (op 20670) 


v_lshrrev_b32 v14, 16, v4 // (op 20671) 
v_lshrrev_b32 v15, 16, v5 // (op 20671) 
v_lshrrev_b32 v16, 16, v6 // (op 20671) 
v_lshrrev_b32 v17, 16, v7 // (op 20671) 


v_pack_b32_f16 v4, v14, v15 // (op 20634) 
v_pack_b32_f16 v5, v16, v17 // (op 20634) 

s_lshl_b32 s0, s4, 5 // (op 20634) 
s_mul_i32 s1, 7, s0 // (op 20634) 
s_add_u32 s1, s1, 192 // (op 20634) 

v_add_u32 v0, v13, s1 // (op 20634) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20634) Store value





v_add_u32 v0, 160, v12 // (op 20466) 

buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20466) Load value

v_accvgpr_read v14, a160 // (op 20445) 
v_accvgpr_read v15, a161 // (op 20445) 
v_accvgpr_read v16, a162 // (op 20445) 
v_accvgpr_read v17, a163 // (op 20445) 
v_mul_f32 v18, v11, v14 // (op 20445) 
v_mul_f32 v19, v11, v15 // (op 20445) 
v_mul_f32 v20, v11, v16 // (op 20445) 
v_mul_f32 v21, v11, v17 // (op 20445) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20465) 
v_lshrrev_b32 v7, 16, v4 // (op 20465) 
v_lshlrev_b32 v6, 16, v6 // (op 20465) 
v_lshlrev_b32 v7, 16, v7 // (op 20465) 
v_mul_f32 v14, v9, v6 // (op 20465) 
v_mul_f32 v15, v9, v7 // (op 20465) 
v_and_b32 v6, 65535, v5 // (op 20465) 
v_lshrrev_b32 v7, 16, v5 // (op 20465) 
v_lshlrev_b32 v6, 16, v6 // (op 20465) 
v_lshlrev_b32 v7, 16, v7 // (op 20465) 
v_mul_f32 v16, v9, v6 // (op 20465) 
v_mul_f32 v17, v9, v7 // (op 20465) 


v_add_f32 v4, v14, v18 // (op 20446) 
v_add_f32 v5, v15, v19 // (op 20446) 
v_add_f32 v6, v16, v20 // (op 20446) 
v_add_f32 v7, v17, v21 // (op 20446) 


v_lshrrev_b32 v14, 16, v4 // (op 20447) 
v_lshrrev_b32 v15, 16, v5 // (op 20447) 
v_lshrrev_b32 v16, 16, v6 // (op 20447) 
v_lshrrev_b32 v17, 16, v7 // (op 20447) 


v_pack_b32_f16 v4, v14, v15 // (op 20449) 
v_pack_b32_f16 v5, v16, v17 // (op 20449) 

v_add_u32 v0, 160, v13 // (op 20449) 

buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20449) Store value





s_lshl_b32 s0, s2, 5 // (op 20471) 
s_add_u32 s0, s0, 160 // (op 20471) 
v_add_u32 v0, v12, s0 // (op 20471) 


buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20471) Load value

v_accvgpr_read v14, a164 // (op 20503) 
v_accvgpr_read v15, a165 // (op 20503) 
v_accvgpr_read v16, a166 // (op 20503) 
v_accvgpr_read v17, a167 // (op 20503) 
v_mul_f32 v18, v11, v14 // (op 20503) 
v_mul_f32 v19, v11, v15 // (op 20503) 
v_mul_f32 v20, v11, v16 // (op 20503) 
v_mul_f32 v21, v11, v17 // (op 20503) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20468) 
v_lshrrev_b32 v7, 16, v4 // (op 20468) 
v_lshlrev_b32 v6, 16, v6 // (op 20468) 
v_lshlrev_b32 v7, 16, v7 // (op 20468) 
v_mul_f32 v14, v9, v6 // (op 20468) 
v_mul_f32 v15, v9, v7 // (op 20468) 
v_and_b32 v6, 65535, v5 // (op 20468) 
v_lshrrev_b32 v7, 16, v5 // (op 20468) 
v_lshlrev_b32 v6, 16, v6 // (op 20468) 
v_lshlrev_b32 v7, 16, v7 // (op 20468) 
v_mul_f32 v16, v9, v6 // (op 20468) 
v_mul_f32 v17, v9, v7 // (op 20468) 


v_add_f32 v4, v14, v18 // (op 20469) 
v_add_f32 v5, v15, v19 // (op 20469) 
v_add_f32 v6, v16, v20 // (op 20469) 
v_add_f32 v7, v17, v21 // (op 20469) 


v_lshrrev_b32 v14, 16, v4 // (op 20470) 
v_lshrrev_b32 v15, 16, v5 // (op 20470) 
v_lshrrev_b32 v16, 16, v6 // (op 20470) 
v_lshrrev_b32 v17, 16, v7 // (op 20470) 


v_pack_b32_f16 v4, v14, v15 // (op 20451) 
v_pack_b32_f16 v5, v16, v17 // (op 20451) 

s_lshl_b32 s0, s4, 5 // (op 20451) 
s_add_u32 s0, s0, 160 // (op 20451) 
v_add_u32 v0, v13, s0 // (op 20451) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20451) Store value





s_lshl_b32 s0, s2, 6 // (op 20476) 
s_add_u32 s0, s0, 160 // (op 20476) 
v_add_u32 v0, v12, s0 // (op 20476) 


buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20476) Load value

v_accvgpr_read v14, a168 // (op 20505) 
v_accvgpr_read v15, a169 // (op 20505) 
v_accvgpr_read v16, a170 // (op 20505) 
v_accvgpr_read v17, a171 // (op 20505) 
v_mul_f32 v18, v11, v14 // (op 20505) 
v_mul_f32 v19, v11, v15 // (op 20505) 
v_mul_f32 v20, v11, v16 // (op 20505) 
v_mul_f32 v21, v11, v17 // (op 20505) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20473) 
v_lshrrev_b32 v7, 16, v4 // (op 20473) 
v_lshlrev_b32 v6, 16, v6 // (op 20473) 
v_lshlrev_b32 v7, 16, v7 // (op 20473) 
v_mul_f32 v14, v9, v6 // (op 20473) 
v_mul_f32 v15, v9, v7 // (op 20473) 
v_and_b32 v6, 65535, v5 // (op 20473) 
v_lshrrev_b32 v7, 16, v5 // (op 20473) 
v_lshlrev_b32 v6, 16, v6 // (op 20473) 
v_lshlrev_b32 v7, 16, v7 // (op 20473) 
v_mul_f32 v16, v9, v6 // (op 20473) 
v_mul_f32 v17, v9, v7 // (op 20473) 


v_add_f32 v4, v14, v18 // (op 20474) 
v_add_f32 v5, v15, v19 // (op 20474) 
v_add_f32 v6, v16, v20 // (op 20474) 
v_add_f32 v7, v17, v21 // (op 20474) 


v_lshrrev_b32 v14, 16, v4 // (op 20475) 
v_lshrrev_b32 v15, 16, v5 // (op 20475) 
v_lshrrev_b32 v16, 16, v6 // (op 20475) 
v_lshrrev_b32 v17, 16, v7 // (op 20475) 


v_pack_b32_f16 v4, v14, v15 // (op 20453) 
v_pack_b32_f16 v5, v16, v17 // (op 20453) 

s_lshl_b32 s0, s4, 6 // (op 20453) 
s_add_u32 s0, s0, 160 // (op 20453) 
v_add_u32 v0, v13, s0 // (op 20453) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20453) Store value





s_lshl_b32 s0, s2, 5 // (op 20481) 
s_mul_i32 s1, 3, s0 // (op 20481) 
s_add_u32 s1, s1, 160 // (op 20481) 

v_add_u32 v0, v12, s1 // (op 20481) 


buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20481) Load value

v_accvgpr_read v14, a172 // (op 20507) 
v_accvgpr_read v15, a173 // (op 20507) 
v_accvgpr_read v16, a174 // (op 20507) 
v_accvgpr_read v17, a175 // (op 20507) 
v_mul_f32 v18, v11, v14 // (op 20507) 
v_mul_f32 v19, v11, v15 // (op 20507) 
v_mul_f32 v20, v11, v16 // (op 20507) 
v_mul_f32 v21, v11, v17 // (op 20507) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20478) 
v_lshrrev_b32 v7, 16, v4 // (op 20478) 
v_lshlrev_b32 v6, 16, v6 // (op 20478) 
v_lshlrev_b32 v7, 16, v7 // (op 20478) 
v_mul_f32 v14, v9, v6 // (op 20478) 
v_mul_f32 v15, v9, v7 // (op 20478) 
v_and_b32 v6, 65535, v5 // (op 20478) 
v_lshrrev_b32 v7, 16, v5 // (op 20478) 
v_lshlrev_b32 v6, 16, v6 // (op 20478) 
v_lshlrev_b32 v7, 16, v7 // (op 20478) 
v_mul_f32 v16, v9, v6 // (op 20478) 
v_mul_f32 v17, v9, v7 // (op 20478) 


v_add_f32 v4, v14, v18 // (op 20479) 
v_add_f32 v5, v15, v19 // (op 20479) 
v_add_f32 v6, v16, v20 // (op 20479) 
v_add_f32 v7, v17, v21 // (op 20479) 


v_lshrrev_b32 v14, 16, v4 // (op 20480) 
v_lshrrev_b32 v15, 16, v5 // (op 20480) 
v_lshrrev_b32 v16, 16, v6 // (op 20480) 
v_lshrrev_b32 v17, 16, v7 // (op 20480) 


v_pack_b32_f16 v4, v14, v15 // (op 20455) 
v_pack_b32_f16 v5, v16, v17 // (op 20455) 

s_lshl_b32 s0, s4, 5 // (op 20455) 
s_mul_i32 s1, 3, s0 // (op 20455) 
s_add_u32 s1, s1, 160 // (op 20455) 

v_add_u32 v0, v13, s1 // (op 20455) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20455) Store value





s_lshl_b32 s0, s2, 7 // (op 20486) 
s_add_u32 s0, s0, 160 // (op 20486) 
v_add_u32 v0, v12, s0 // (op 20486) 


buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20486) Load value

v_accvgpr_read v14, a176 // (op 20509) 
v_accvgpr_read v15, a177 // (op 20509) 
v_accvgpr_read v16, a178 // (op 20509) 
v_accvgpr_read v17, a179 // (op 20509) 
v_mul_f32 v18, v11, v14 // (op 20509) 
v_mul_f32 v19, v11, v15 // (op 20509) 
v_mul_f32 v20, v11, v16 // (op 20509) 
v_mul_f32 v21, v11, v17 // (op 20509) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20483) 
v_lshrrev_b32 v7, 16, v4 // (op 20483) 
v_lshlrev_b32 v6, 16, v6 // (op 20483) 
v_lshlrev_b32 v7, 16, v7 // (op 20483) 
v_mul_f32 v14, v9, v6 // (op 20483) 
v_mul_f32 v15, v9, v7 // (op 20483) 
v_and_b32 v6, 65535, v5 // (op 20483) 
v_lshrrev_b32 v7, 16, v5 // (op 20483) 
v_lshlrev_b32 v6, 16, v6 // (op 20483) 
v_lshlrev_b32 v7, 16, v7 // (op 20483) 
v_mul_f32 v16, v9, v6 // (op 20483) 
v_mul_f32 v17, v9, v7 // (op 20483) 


v_add_f32 v4, v14, v18 // (op 20484) 
v_add_f32 v5, v15, v19 // (op 20484) 
v_add_f32 v6, v16, v20 // (op 20484) 
v_add_f32 v7, v17, v21 // (op 20484) 


v_lshrrev_b32 v14, 16, v4 // (op 20485) 
v_lshrrev_b32 v15, 16, v5 // (op 20485) 
v_lshrrev_b32 v16, 16, v6 // (op 20485) 
v_lshrrev_b32 v17, 16, v7 // (op 20485) 


v_pack_b32_f16 v4, v14, v15 // (op 20457) 
v_pack_b32_f16 v5, v16, v17 // (op 20457) 

s_lshl_b32 s0, s4, 7 // (op 20457) 
s_add_u32 s0, s0, 160 // (op 20457) 
v_add_u32 v0, v13, s0 // (op 20457) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20457) Store value





s_lshl_b32 s0, s2, 5 // (op 20491) 
s_mul_i32 s1, 5, s0 // (op 20491) 
s_add_u32 s1, s1, 160 // (op 20491) 

v_add_u32 v0, v12, s1 // (op 20491) 


buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20491) Load value

v_accvgpr_read v14, a180 // (op 20511) 
v_accvgpr_read v15, a181 // (op 20511) 
v_accvgpr_read v16, a182 // (op 20511) 
v_accvgpr_read v17, a183 // (op 20511) 
v_mul_f32 v18, v11, v14 // (op 20511) 
v_mul_f32 v19, v11, v15 // (op 20511) 
v_mul_f32 v20, v11, v16 // (op 20511) 
v_mul_f32 v21, v11, v17 // (op 20511) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20488) 
v_lshrrev_b32 v7, 16, v4 // (op 20488) 
v_lshlrev_b32 v6, 16, v6 // (op 20488) 
v_lshlrev_b32 v7, 16, v7 // (op 20488) 
v_mul_f32 v14, v9, v6 // (op 20488) 
v_mul_f32 v15, v9, v7 // (op 20488) 
v_and_b32 v6, 65535, v5 // (op 20488) 
v_lshrrev_b32 v7, 16, v5 // (op 20488) 
v_lshlrev_b32 v6, 16, v6 // (op 20488) 
v_lshlrev_b32 v7, 16, v7 // (op 20488) 
v_mul_f32 v16, v9, v6 // (op 20488) 
v_mul_f32 v17, v9, v7 // (op 20488) 


v_add_f32 v4, v14, v18 // (op 20489) 
v_add_f32 v5, v15, v19 // (op 20489) 
v_add_f32 v6, v16, v20 // (op 20489) 
v_add_f32 v7, v17, v21 // (op 20489) 


v_lshrrev_b32 v14, 16, v4 // (op 20490) 
v_lshrrev_b32 v15, 16, v5 // (op 20490) 
v_lshrrev_b32 v16, 16, v6 // (op 20490) 
v_lshrrev_b32 v17, 16, v7 // (op 20490) 


v_pack_b32_f16 v4, v14, v15 // (op 20459) 
v_pack_b32_f16 v5, v16, v17 // (op 20459) 

s_lshl_b32 s0, s4, 5 // (op 20459) 
s_mul_i32 s1, 5, s0 // (op 20459) 
s_add_u32 s1, s1, 160 // (op 20459) 

v_add_u32 v0, v13, s1 // (op 20459) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20459) Store value





s_lshl_b32 s0, s2, 5 // (op 20496) 
s_mul_i32 s1, 6, s0 // (op 20496) 
s_add_u32 s1, s1, 160 // (op 20496) 

v_add_u32 v0, v12, s1 // (op 20496) 


buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20496) Load value

v_accvgpr_read v14, a184 // (op 20513) 
v_accvgpr_read v15, a185 // (op 20513) 
v_accvgpr_read v16, a186 // (op 20513) 
v_accvgpr_read v17, a187 // (op 20513) 
v_mul_f32 v18, v11, v14 // (op 20513) 
v_mul_f32 v19, v11, v15 // (op 20513) 
v_mul_f32 v20, v11, v16 // (op 20513) 
v_mul_f32 v21, v11, v17 // (op 20513) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20493) 
v_lshrrev_b32 v7, 16, v4 // (op 20493) 
v_lshlrev_b32 v6, 16, v6 // (op 20493) 
v_lshlrev_b32 v7, 16, v7 // (op 20493) 
v_mul_f32 v14, v9, v6 // (op 20493) 
v_mul_f32 v15, v9, v7 // (op 20493) 
v_and_b32 v6, 65535, v5 // (op 20493) 
v_lshrrev_b32 v7, 16, v5 // (op 20493) 
v_lshlrev_b32 v6, 16, v6 // (op 20493) 
v_lshlrev_b32 v7, 16, v7 // (op 20493) 
v_mul_f32 v16, v9, v6 // (op 20493) 
v_mul_f32 v17, v9, v7 // (op 20493) 


v_add_f32 v4, v14, v18 // (op 20494) 
v_add_f32 v5, v15, v19 // (op 20494) 
v_add_f32 v6, v16, v20 // (op 20494) 
v_add_f32 v7, v17, v21 // (op 20494) 


v_lshrrev_b32 v14, 16, v4 // (op 20495) 
v_lshrrev_b32 v15, 16, v5 // (op 20495) 
v_lshrrev_b32 v16, 16, v6 // (op 20495) 
v_lshrrev_b32 v17, 16, v7 // (op 20495) 


v_pack_b32_f16 v4, v14, v15 // (op 20461) 
v_pack_b32_f16 v5, v16, v17 // (op 20461) 

s_lshl_b32 s0, s4, 5 // (op 20461) 
s_mul_i32 s1, 6, s0 // (op 20461) 
s_add_u32 s1, s1, 160 // (op 20461) 

v_add_u32 v0, v13, s1 // (op 20461) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20461) Store value





s_lshl_b32 s0, s2, 5 // (op 20501) 
s_mul_i32 s1, 7, s0 // (op 20501) 
s_add_u32 s1, s1, 160 // (op 20501) 

v_add_u32 v0, v12, s1 // (op 20501) 


buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20501) Load value

v_accvgpr_read v14, a188 // (op 20515) 
v_accvgpr_read v15, a189 // (op 20515) 
v_accvgpr_read v16, a190 // (op 20515) 
v_accvgpr_read v17, a191 // (op 20515) 
v_mul_f32 v18, v11, v14 // (op 20515) 
v_mul_f32 v19, v11, v15 // (op 20515) 
v_mul_f32 v20, v11, v16 // (op 20515) 
v_mul_f32 v21, v11, v17 // (op 20515) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20498) 
v_lshrrev_b32 v7, 16, v4 // (op 20498) 
v_lshlrev_b32 v6, 16, v6 // (op 20498) 
v_lshlrev_b32 v7, 16, v7 // (op 20498) 
v_mul_f32 v14, v9, v6 // (op 20498) 
v_mul_f32 v15, v9, v7 // (op 20498) 
v_and_b32 v6, 65535, v5 // (op 20498) 
v_lshrrev_b32 v7, 16, v5 // (op 20498) 
v_lshlrev_b32 v6, 16, v6 // (op 20498) 
v_lshlrev_b32 v7, 16, v7 // (op 20498) 
v_mul_f32 v16, v9, v6 // (op 20498) 
v_mul_f32 v17, v9, v7 // (op 20498) 


v_add_f32 v4, v14, v18 // (op 20499) 
v_add_f32 v5, v15, v19 // (op 20499) 
v_add_f32 v6, v16, v20 // (op 20499) 
v_add_f32 v7, v17, v21 // (op 20499) 


v_lshrrev_b32 v14, 16, v4 // (op 20500) 
v_lshrrev_b32 v15, 16, v5 // (op 20500) 
v_lshrrev_b32 v16, 16, v6 // (op 20500) 
v_lshrrev_b32 v17, 16, v7 // (op 20500) 


v_pack_b32_f16 v4, v14, v15 // (op 20463) 
v_pack_b32_f16 v5, v16, v17 // (op 20463) 

s_lshl_b32 s0, s4, 5 // (op 20463) 
s_mul_i32 s1, 7, s0 // (op 20463) 
s_add_u32 s1, s1, 160 // (op 20463) 

v_add_u32 v0, v13, s1 // (op 20463) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20463) Store value





v_add_u32 v0, 128, v12 // (op 20295) 

buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20295) Load value

v_accvgpr_read v14, a128 // (op 20274) 
v_accvgpr_read v15, a129 // (op 20274) 
v_accvgpr_read v16, a130 // (op 20274) 
v_accvgpr_read v17, a131 // (op 20274) 
v_mul_f32 v18, v11, v14 // (op 20274) 
v_mul_f32 v19, v11, v15 // (op 20274) 
v_mul_f32 v20, v11, v16 // (op 20274) 
v_mul_f32 v21, v11, v17 // (op 20274) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20294) 
v_lshrrev_b32 v7, 16, v4 // (op 20294) 
v_lshlrev_b32 v6, 16, v6 // (op 20294) 
v_lshlrev_b32 v7, 16, v7 // (op 20294) 
v_mul_f32 v14, v9, v6 // (op 20294) 
v_mul_f32 v15, v9, v7 // (op 20294) 
v_and_b32 v6, 65535, v5 // (op 20294) 
v_lshrrev_b32 v7, 16, v5 // (op 20294) 
v_lshlrev_b32 v6, 16, v6 // (op 20294) 
v_lshlrev_b32 v7, 16, v7 // (op 20294) 
v_mul_f32 v16, v9, v6 // (op 20294) 
v_mul_f32 v17, v9, v7 // (op 20294) 


v_add_f32 v4, v14, v18 // (op 20275) 
v_add_f32 v5, v15, v19 // (op 20275) 
v_add_f32 v6, v16, v20 // (op 20275) 
v_add_f32 v7, v17, v21 // (op 20275) 


v_lshrrev_b32 v14, 16, v4 // (op 20276) 
v_lshrrev_b32 v15, 16, v5 // (op 20276) 
v_lshrrev_b32 v16, 16, v6 // (op 20276) 
v_lshrrev_b32 v17, 16, v7 // (op 20276) 


v_pack_b32_f16 v4, v14, v15 // (op 20278) 
v_pack_b32_f16 v5, v16, v17 // (op 20278) 

v_add_u32 v0, 128, v13 // (op 20278) 

buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20278) Store value





s_lshl_b32 s0, s2, 5 // (op 20300) 
s_add_u32 s0, s0, 128 // (op 20300) 
v_add_u32 v0, v12, s0 // (op 20300) 


buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20300) Load value

v_accvgpr_read v14, a132 // (op 20332) 
v_accvgpr_read v15, a133 // (op 20332) 
v_accvgpr_read v16, a134 // (op 20332) 
v_accvgpr_read v17, a135 // (op 20332) 
v_mul_f32 v18, v11, v14 // (op 20332) 
v_mul_f32 v19, v11, v15 // (op 20332) 
v_mul_f32 v20, v11, v16 // (op 20332) 
v_mul_f32 v21, v11, v17 // (op 20332) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20297) 
v_lshrrev_b32 v7, 16, v4 // (op 20297) 
v_lshlrev_b32 v6, 16, v6 // (op 20297) 
v_lshlrev_b32 v7, 16, v7 // (op 20297) 
v_mul_f32 v14, v9, v6 // (op 20297) 
v_mul_f32 v15, v9, v7 // (op 20297) 
v_and_b32 v6, 65535, v5 // (op 20297) 
v_lshrrev_b32 v7, 16, v5 // (op 20297) 
v_lshlrev_b32 v6, 16, v6 // (op 20297) 
v_lshlrev_b32 v7, 16, v7 // (op 20297) 
v_mul_f32 v16, v9, v6 // (op 20297) 
v_mul_f32 v17, v9, v7 // (op 20297) 


v_add_f32 v4, v14, v18 // (op 20298) 
v_add_f32 v5, v15, v19 // (op 20298) 
v_add_f32 v6, v16, v20 // (op 20298) 
v_add_f32 v7, v17, v21 // (op 20298) 


v_lshrrev_b32 v14, 16, v4 // (op 20299) 
v_lshrrev_b32 v15, 16, v5 // (op 20299) 
v_lshrrev_b32 v16, 16, v6 // (op 20299) 
v_lshrrev_b32 v17, 16, v7 // (op 20299) 


v_pack_b32_f16 v4, v14, v15 // (op 20280) 
v_pack_b32_f16 v5, v16, v17 // (op 20280) 

s_lshl_b32 s0, s4, 5 // (op 20280) 
s_add_u32 s0, s0, 128 // (op 20280) 
v_add_u32 v0, v13, s0 // (op 20280) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20280) Store value





s_lshl_b32 s0, s2, 6 // (op 20305) 
s_add_u32 s0, s0, 128 // (op 20305) 
v_add_u32 v0, v12, s0 // (op 20305) 


buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20305) Load value

v_accvgpr_read v14, a136 // (op 20334) 
v_accvgpr_read v15, a137 // (op 20334) 
v_accvgpr_read v16, a138 // (op 20334) 
v_accvgpr_read v17, a139 // (op 20334) 
v_mul_f32 v18, v11, v14 // (op 20334) 
v_mul_f32 v19, v11, v15 // (op 20334) 
v_mul_f32 v20, v11, v16 // (op 20334) 
v_mul_f32 v21, v11, v17 // (op 20334) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20302) 
v_lshrrev_b32 v7, 16, v4 // (op 20302) 
v_lshlrev_b32 v6, 16, v6 // (op 20302) 
v_lshlrev_b32 v7, 16, v7 // (op 20302) 
v_mul_f32 v14, v9, v6 // (op 20302) 
v_mul_f32 v15, v9, v7 // (op 20302) 
v_and_b32 v6, 65535, v5 // (op 20302) 
v_lshrrev_b32 v7, 16, v5 // (op 20302) 
v_lshlrev_b32 v6, 16, v6 // (op 20302) 
v_lshlrev_b32 v7, 16, v7 // (op 20302) 
v_mul_f32 v16, v9, v6 // (op 20302) 
v_mul_f32 v17, v9, v7 // (op 20302) 


v_add_f32 v4, v14, v18 // (op 20303) 
v_add_f32 v5, v15, v19 // (op 20303) 
v_add_f32 v6, v16, v20 // (op 20303) 
v_add_f32 v7, v17, v21 // (op 20303) 


v_lshrrev_b32 v14, 16, v4 // (op 20304) 
v_lshrrev_b32 v15, 16, v5 // (op 20304) 
v_lshrrev_b32 v16, 16, v6 // (op 20304) 
v_lshrrev_b32 v17, 16, v7 // (op 20304) 


v_pack_b32_f16 v4, v14, v15 // (op 20282) 
v_pack_b32_f16 v5, v16, v17 // (op 20282) 

s_lshl_b32 s0, s4, 6 // (op 20282) 
s_add_u32 s0, s0, 128 // (op 20282) 
v_add_u32 v0, v13, s0 // (op 20282) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20282) Store value





s_lshl_b32 s0, s2, 5 // (op 20310) 
s_mul_i32 s1, 3, s0 // (op 20310) 
s_add_u32 s1, s1, 128 // (op 20310) 

v_add_u32 v0, v12, s1 // (op 20310) 


buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20310) Load value

v_accvgpr_read v14, a140 // (op 20336) 
v_accvgpr_read v15, a141 // (op 20336) 
v_accvgpr_read v16, a142 // (op 20336) 
v_accvgpr_read v17, a143 // (op 20336) 
v_mul_f32 v18, v11, v14 // (op 20336) 
v_mul_f32 v19, v11, v15 // (op 20336) 
v_mul_f32 v20, v11, v16 // (op 20336) 
v_mul_f32 v21, v11, v17 // (op 20336) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20307) 
v_lshrrev_b32 v7, 16, v4 // (op 20307) 
v_lshlrev_b32 v6, 16, v6 // (op 20307) 
v_lshlrev_b32 v7, 16, v7 // (op 20307) 
v_mul_f32 v14, v9, v6 // (op 20307) 
v_mul_f32 v15, v9, v7 // (op 20307) 
v_and_b32 v6, 65535, v5 // (op 20307) 
v_lshrrev_b32 v7, 16, v5 // (op 20307) 
v_lshlrev_b32 v6, 16, v6 // (op 20307) 
v_lshlrev_b32 v7, 16, v7 // (op 20307) 
v_mul_f32 v16, v9, v6 // (op 20307) 
v_mul_f32 v17, v9, v7 // (op 20307) 


v_add_f32 v4, v14, v18 // (op 20308) 
v_add_f32 v5, v15, v19 // (op 20308) 
v_add_f32 v6, v16, v20 // (op 20308) 
v_add_f32 v7, v17, v21 // (op 20308) 


v_lshrrev_b32 v14, 16, v4 // (op 20309) 
v_lshrrev_b32 v15, 16, v5 // (op 20309) 
v_lshrrev_b32 v16, 16, v6 // (op 20309) 
v_lshrrev_b32 v17, 16, v7 // (op 20309) 


v_pack_b32_f16 v4, v14, v15 // (op 20284) 
v_pack_b32_f16 v5, v16, v17 // (op 20284) 

s_lshl_b32 s0, s4, 5 // (op 20284) 
s_mul_i32 s1, 3, s0 // (op 20284) 
s_add_u32 s1, s1, 128 // (op 20284) 

v_add_u32 v0, v13, s1 // (op 20284) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20284) Store value





s_lshl_b32 s0, s2, 7 // (op 20315) 
s_add_u32 s0, s0, 128 // (op 20315) 
v_add_u32 v0, v12, s0 // (op 20315) 


buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20315) Load value

v_accvgpr_read v14, a144 // (op 20338) 
v_accvgpr_read v15, a145 // (op 20338) 
v_accvgpr_read v16, a146 // (op 20338) 
v_accvgpr_read v17, a147 // (op 20338) 
v_mul_f32 v18, v11, v14 // (op 20338) 
v_mul_f32 v19, v11, v15 // (op 20338) 
v_mul_f32 v20, v11, v16 // (op 20338) 
v_mul_f32 v21, v11, v17 // (op 20338) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20312) 
v_lshrrev_b32 v7, 16, v4 // (op 20312) 
v_lshlrev_b32 v6, 16, v6 // (op 20312) 
v_lshlrev_b32 v7, 16, v7 // (op 20312) 
v_mul_f32 v14, v9, v6 // (op 20312) 
v_mul_f32 v15, v9, v7 // (op 20312) 
v_and_b32 v6, 65535, v5 // (op 20312) 
v_lshrrev_b32 v7, 16, v5 // (op 20312) 
v_lshlrev_b32 v6, 16, v6 // (op 20312) 
v_lshlrev_b32 v7, 16, v7 // (op 20312) 
v_mul_f32 v16, v9, v6 // (op 20312) 
v_mul_f32 v17, v9, v7 // (op 20312) 


v_add_f32 v4, v14, v18 // (op 20313) 
v_add_f32 v5, v15, v19 // (op 20313) 
v_add_f32 v6, v16, v20 // (op 20313) 
v_add_f32 v7, v17, v21 // (op 20313) 


v_lshrrev_b32 v14, 16, v4 // (op 20314) 
v_lshrrev_b32 v15, 16, v5 // (op 20314) 
v_lshrrev_b32 v16, 16, v6 // (op 20314) 
v_lshrrev_b32 v17, 16, v7 // (op 20314) 


v_pack_b32_f16 v4, v14, v15 // (op 20286) 
v_pack_b32_f16 v5, v16, v17 // (op 20286) 

s_lshl_b32 s0, s4, 7 // (op 20286) 
s_add_u32 s0, s0, 128 // (op 20286) 
v_add_u32 v0, v13, s0 // (op 20286) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20286) Store value





s_lshl_b32 s0, s2, 5 // (op 20320) 
s_mul_i32 s1, 5, s0 // (op 20320) 
s_add_u32 s1, s1, 128 // (op 20320) 

v_add_u32 v0, v12, s1 // (op 20320) 


buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20320) Load value

v_accvgpr_read v14, a148 // (op 20340) 
v_accvgpr_read v15, a149 // (op 20340) 
v_accvgpr_read v16, a150 // (op 20340) 
v_accvgpr_read v17, a151 // (op 20340) 
v_mul_f32 v18, v11, v14 // (op 20340) 
v_mul_f32 v19, v11, v15 // (op 20340) 
v_mul_f32 v20, v11, v16 // (op 20340) 
v_mul_f32 v21, v11, v17 // (op 20340) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20317) 
v_lshrrev_b32 v7, 16, v4 // (op 20317) 
v_lshlrev_b32 v6, 16, v6 // (op 20317) 
v_lshlrev_b32 v7, 16, v7 // (op 20317) 
v_mul_f32 v14, v9, v6 // (op 20317) 
v_mul_f32 v15, v9, v7 // (op 20317) 
v_and_b32 v6, 65535, v5 // (op 20317) 
v_lshrrev_b32 v7, 16, v5 // (op 20317) 
v_lshlrev_b32 v6, 16, v6 // (op 20317) 
v_lshlrev_b32 v7, 16, v7 // (op 20317) 
v_mul_f32 v16, v9, v6 // (op 20317) 
v_mul_f32 v17, v9, v7 // (op 20317) 


v_add_f32 v4, v14, v18 // (op 20318) 
v_add_f32 v5, v15, v19 // (op 20318) 
v_add_f32 v6, v16, v20 // (op 20318) 
v_add_f32 v7, v17, v21 // (op 20318) 


v_lshrrev_b32 v14, 16, v4 // (op 20319) 
v_lshrrev_b32 v15, 16, v5 // (op 20319) 
v_lshrrev_b32 v16, 16, v6 // (op 20319) 
v_lshrrev_b32 v17, 16, v7 // (op 20319) 


v_pack_b32_f16 v4, v14, v15 // (op 20288) 
v_pack_b32_f16 v5, v16, v17 // (op 20288) 

s_lshl_b32 s0, s4, 5 // (op 20288) 
s_mul_i32 s1, 5, s0 // (op 20288) 
s_add_u32 s1, s1, 128 // (op 20288) 

v_add_u32 v0, v13, s1 // (op 20288) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20288) Store value





s_lshl_b32 s0, s2, 5 // (op 20325) 
s_mul_i32 s1, 6, s0 // (op 20325) 
s_add_u32 s1, s1, 128 // (op 20325) 

v_add_u32 v0, v12, s1 // (op 20325) 


buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20325) Load value

v_accvgpr_read v14, a152 // (op 20342) 
v_accvgpr_read v15, a153 // (op 20342) 
v_accvgpr_read v16, a154 // (op 20342) 
v_accvgpr_read v17, a155 // (op 20342) 
v_mul_f32 v18, v11, v14 // (op 20342) 
v_mul_f32 v19, v11, v15 // (op 20342) 
v_mul_f32 v20, v11, v16 // (op 20342) 
v_mul_f32 v21, v11, v17 // (op 20342) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20322) 
v_lshrrev_b32 v7, 16, v4 // (op 20322) 
v_lshlrev_b32 v6, 16, v6 // (op 20322) 
v_lshlrev_b32 v7, 16, v7 // (op 20322) 
v_mul_f32 v14, v9, v6 // (op 20322) 
v_mul_f32 v15, v9, v7 // (op 20322) 
v_and_b32 v6, 65535, v5 // (op 20322) 
v_lshrrev_b32 v7, 16, v5 // (op 20322) 
v_lshlrev_b32 v6, 16, v6 // (op 20322) 
v_lshlrev_b32 v7, 16, v7 // (op 20322) 
v_mul_f32 v16, v9, v6 // (op 20322) 
v_mul_f32 v17, v9, v7 // (op 20322) 


v_add_f32 v4, v14, v18 // (op 20323) 
v_add_f32 v5, v15, v19 // (op 20323) 
v_add_f32 v6, v16, v20 // (op 20323) 
v_add_f32 v7, v17, v21 // (op 20323) 


v_lshrrev_b32 v14, 16, v4 // (op 20324) 
v_lshrrev_b32 v15, 16, v5 // (op 20324) 
v_lshrrev_b32 v16, 16, v6 // (op 20324) 
v_lshrrev_b32 v17, 16, v7 // (op 20324) 


v_pack_b32_f16 v4, v14, v15 // (op 20290) 
v_pack_b32_f16 v5, v16, v17 // (op 20290) 

s_lshl_b32 s0, s4, 5 // (op 20290) 
s_mul_i32 s1, 6, s0 // (op 20290) 
s_add_u32 s1, s1, 128 // (op 20290) 

v_add_u32 v0, v13, s1 // (op 20290) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20290) Store value





s_lshl_b32 s0, s2, 5 // (op 20330) 
s_mul_i32 s1, 7, s0 // (op 20330) 
s_add_u32 s1, s1, 128 // (op 20330) 

v_add_u32 v0, v12, s1 // (op 20330) 


buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20330) Load value

v_accvgpr_read v14, a156 // (op 20344) 
v_accvgpr_read v15, a157 // (op 20344) 
v_accvgpr_read v16, a158 // (op 20344) 
v_accvgpr_read v17, a159 // (op 20344) 
v_mul_f32 v18, v11, v14 // (op 20344) 
v_mul_f32 v19, v11, v15 // (op 20344) 
v_mul_f32 v20, v11, v16 // (op 20344) 
v_mul_f32 v21, v11, v17 // (op 20344) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20327) 
v_lshrrev_b32 v7, 16, v4 // (op 20327) 
v_lshlrev_b32 v6, 16, v6 // (op 20327) 
v_lshlrev_b32 v7, 16, v7 // (op 20327) 
v_mul_f32 v14, v9, v6 // (op 20327) 
v_mul_f32 v15, v9, v7 // (op 20327) 
v_and_b32 v6, 65535, v5 // (op 20327) 
v_lshrrev_b32 v7, 16, v5 // (op 20327) 
v_lshlrev_b32 v6, 16, v6 // (op 20327) 
v_lshlrev_b32 v7, 16, v7 // (op 20327) 
v_mul_f32 v16, v9, v6 // (op 20327) 
v_mul_f32 v17, v9, v7 // (op 20327) 


v_add_f32 v4, v14, v18 // (op 20328) 
v_add_f32 v5, v15, v19 // (op 20328) 
v_add_f32 v6, v16, v20 // (op 20328) 
v_add_f32 v7, v17, v21 // (op 20328) 


v_lshrrev_b32 v14, 16, v4 // (op 20329) 
v_lshrrev_b32 v15, 16, v5 // (op 20329) 
v_lshrrev_b32 v16, 16, v6 // (op 20329) 
v_lshrrev_b32 v17, 16, v7 // (op 20329) 


v_pack_b32_f16 v4, v14, v15 // (op 20292) 
v_pack_b32_f16 v5, v16, v17 // (op 20292) 

s_lshl_b32 s0, s4, 5 // (op 20292) 
s_mul_i32 s1, 7, s0 // (op 20292) 
s_add_u32 s1, s1, 128 // (op 20292) 

v_add_u32 v0, v13, s1 // (op 20292) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20292) Store value





v_add_u32 v0, 96, v12 // (op 20124) 

buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20124) Load value

v_accvgpr_read v14, a96 // (op 20103) 
v_accvgpr_read v15, a97 // (op 20103) 
v_accvgpr_read v16, a98 // (op 20103) 
v_accvgpr_read v17, a99 // (op 20103) 
v_mul_f32 v18, v11, v14 // (op 20103) 
v_mul_f32 v19, v11, v15 // (op 20103) 
v_mul_f32 v20, v11, v16 // (op 20103) 
v_mul_f32 v21, v11, v17 // (op 20103) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20123) 
v_lshrrev_b32 v7, 16, v4 // (op 20123) 
v_lshlrev_b32 v6, 16, v6 // (op 20123) 
v_lshlrev_b32 v7, 16, v7 // (op 20123) 
v_mul_f32 v14, v9, v6 // (op 20123) 
v_mul_f32 v15, v9, v7 // (op 20123) 
v_and_b32 v6, 65535, v5 // (op 20123) 
v_lshrrev_b32 v7, 16, v5 // (op 20123) 
v_lshlrev_b32 v6, 16, v6 // (op 20123) 
v_lshlrev_b32 v7, 16, v7 // (op 20123) 
v_mul_f32 v16, v9, v6 // (op 20123) 
v_mul_f32 v17, v9, v7 // (op 20123) 


v_add_f32 v4, v14, v18 // (op 20104) 
v_add_f32 v5, v15, v19 // (op 20104) 
v_add_f32 v6, v16, v20 // (op 20104) 
v_add_f32 v7, v17, v21 // (op 20104) 


v_lshrrev_b32 v14, 16, v4 // (op 20105) 
v_lshrrev_b32 v15, 16, v5 // (op 20105) 
v_lshrrev_b32 v16, 16, v6 // (op 20105) 
v_lshrrev_b32 v17, 16, v7 // (op 20105) 


v_pack_b32_f16 v4, v14, v15 // (op 20107) 
v_pack_b32_f16 v5, v16, v17 // (op 20107) 

v_add_u32 v0, 96, v13 // (op 20107) 

buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20107) Store value





s_lshl_b32 s0, s2, 5 // (op 20129) 
s_add_u32 s0, s0, 96 // (op 20129) 
v_add_u32 v0, v12, s0 // (op 20129) 


buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20129) Load value

v_accvgpr_read v14, a100 // (op 20161) 
v_accvgpr_read v15, a101 // (op 20161) 
v_accvgpr_read v16, a102 // (op 20161) 
v_accvgpr_read v17, a103 // (op 20161) 
v_mul_f32 v18, v11, v14 // (op 20161) 
v_mul_f32 v19, v11, v15 // (op 20161) 
v_mul_f32 v20, v11, v16 // (op 20161) 
v_mul_f32 v21, v11, v17 // (op 20161) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20126) 
v_lshrrev_b32 v7, 16, v4 // (op 20126) 
v_lshlrev_b32 v6, 16, v6 // (op 20126) 
v_lshlrev_b32 v7, 16, v7 // (op 20126) 
v_mul_f32 v14, v9, v6 // (op 20126) 
v_mul_f32 v15, v9, v7 // (op 20126) 
v_and_b32 v6, 65535, v5 // (op 20126) 
v_lshrrev_b32 v7, 16, v5 // (op 20126) 
v_lshlrev_b32 v6, 16, v6 // (op 20126) 
v_lshlrev_b32 v7, 16, v7 // (op 20126) 
v_mul_f32 v16, v9, v6 // (op 20126) 
v_mul_f32 v17, v9, v7 // (op 20126) 


v_add_f32 v4, v14, v18 // (op 20127) 
v_add_f32 v5, v15, v19 // (op 20127) 
v_add_f32 v6, v16, v20 // (op 20127) 
v_add_f32 v7, v17, v21 // (op 20127) 


v_lshrrev_b32 v14, 16, v4 // (op 20128) 
v_lshrrev_b32 v15, 16, v5 // (op 20128) 
v_lshrrev_b32 v16, 16, v6 // (op 20128) 
v_lshrrev_b32 v17, 16, v7 // (op 20128) 


v_pack_b32_f16 v4, v14, v15 // (op 20109) 
v_pack_b32_f16 v5, v16, v17 // (op 20109) 

s_lshl_b32 s0, s4, 5 // (op 20109) 
s_add_u32 s0, s0, 96 // (op 20109) 
v_add_u32 v0, v13, s0 // (op 20109) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20109) Store value





s_lshl_b32 s0, s2, 6 // (op 20134) 
s_add_u32 s0, s0, 96 // (op 20134) 
v_add_u32 v0, v12, s0 // (op 20134) 


buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20134) Load value

v_accvgpr_read v14, a104 // (op 20163) 
v_accvgpr_read v15, a105 // (op 20163) 
v_accvgpr_read v16, a106 // (op 20163) 
v_accvgpr_read v17, a107 // (op 20163) 
v_mul_f32 v18, v11, v14 // (op 20163) 
v_mul_f32 v19, v11, v15 // (op 20163) 
v_mul_f32 v20, v11, v16 // (op 20163) 
v_mul_f32 v21, v11, v17 // (op 20163) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20131) 
v_lshrrev_b32 v7, 16, v4 // (op 20131) 
v_lshlrev_b32 v6, 16, v6 // (op 20131) 
v_lshlrev_b32 v7, 16, v7 // (op 20131) 
v_mul_f32 v14, v9, v6 // (op 20131) 
v_mul_f32 v15, v9, v7 // (op 20131) 
v_and_b32 v6, 65535, v5 // (op 20131) 
v_lshrrev_b32 v7, 16, v5 // (op 20131) 
v_lshlrev_b32 v6, 16, v6 // (op 20131) 
v_lshlrev_b32 v7, 16, v7 // (op 20131) 
v_mul_f32 v16, v9, v6 // (op 20131) 
v_mul_f32 v17, v9, v7 // (op 20131) 


v_add_f32 v4, v14, v18 // (op 20132) 
v_add_f32 v5, v15, v19 // (op 20132) 
v_add_f32 v6, v16, v20 // (op 20132) 
v_add_f32 v7, v17, v21 // (op 20132) 


v_lshrrev_b32 v14, 16, v4 // (op 20133) 
v_lshrrev_b32 v15, 16, v5 // (op 20133) 
v_lshrrev_b32 v16, 16, v6 // (op 20133) 
v_lshrrev_b32 v17, 16, v7 // (op 20133) 


v_pack_b32_f16 v4, v14, v15 // (op 20111) 
v_pack_b32_f16 v5, v16, v17 // (op 20111) 

s_lshl_b32 s0, s4, 6 // (op 20111) 
s_add_u32 s0, s0, 96 // (op 20111) 
v_add_u32 v0, v13, s0 // (op 20111) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20111) Store value





s_lshl_b32 s0, s2, 5 // (op 20139) 
s_mul_i32 s1, 3, s0 // (op 20139) 
s_add_u32 s1, s1, 96 // (op 20139) 

v_add_u32 v0, v12, s1 // (op 20139) 


v_accvgpr_read v4, a108 // (op 20165) 
v_accvgpr_read v5, a109 // (op 20165) 
v_accvgpr_read v6, a110 // (op 20165) 
v_accvgpr_read v7, a111 // (op 20165) 
v_mul_f32 v14, v11, v4 // (op 20165) 
v_mul_f32 v15, v11, v5 // (op 20165) 
v_mul_f32 v16, v11, v6 // (op 20165) 
v_mul_f32 v17, v11, v7 // (op 20165) 

buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20139) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20136) 
v_lshrrev_b32 v7, 16, v4 // (op 20136) 
v_lshlrev_b32 v6, 16, v6 // (op 20136) 
v_lshlrev_b32 v7, 16, v7 // (op 20136) 
v_mul_f32 v18, v9, v6 // (op 20136) 
v_mul_f32 v19, v9, v7 // (op 20136) 
v_and_b32 v6, 65535, v5 // (op 20136) 
v_lshrrev_b32 v7, 16, v5 // (op 20136) 
v_lshlrev_b32 v6, 16, v6 // (op 20136) 
v_lshlrev_b32 v7, 16, v7 // (op 20136) 
v_mul_f32 v20, v9, v6 // (op 20136) 
v_mul_f32 v21, v9, v7 // (op 20136) 


v_add_f32 v4, v18, v14 // (op 20137) 
v_add_f32 v5, v19, v15 // (op 20137) 
v_add_f32 v6, v20, v16 // (op 20137) 
v_add_f32 v7, v21, v17 // (op 20137) 


v_lshrrev_b32 v14, 16, v4 // (op 20138) 
v_lshrrev_b32 v15, 16, v5 // (op 20138) 
v_lshrrev_b32 v16, 16, v6 // (op 20138) 
v_lshrrev_b32 v17, 16, v7 // (op 20138) 


v_pack_b32_f16 v4, v14, v15 // (op 20113) 
v_pack_b32_f16 v5, v16, v17 // (op 20113) 

s_lshl_b32 s0, s4, 5 // (op 20113) 
s_mul_i32 s1, 3, s0 // (op 20113) 
s_add_u32 s1, s1, 96 // (op 20113) 

v_add_u32 v0, v13, s1 // (op 20113) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20113) Store value





s_lshl_b32 s0, s2, 7 // (op 20144) 
s_add_u32 s0, s0, 96 // (op 20144) 
v_add_u32 v0, v12, s0 // (op 20144) 


v_accvgpr_read v4, a112 // (op 20167) 
v_accvgpr_read v5, a113 // (op 20167) 
v_accvgpr_read v6, a114 // (op 20167) 
v_accvgpr_read v7, a115 // (op 20167) 
v_mul_f32 v14, v11, v4 // (op 20167) 
v_mul_f32 v15, v11, v5 // (op 20167) 
v_mul_f32 v16, v11, v6 // (op 20167) 
v_mul_f32 v17, v11, v7 // (op 20167) 

buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20144) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20141) 
v_lshrrev_b32 v7, 16, v4 // (op 20141) 
v_lshlrev_b32 v6, 16, v6 // (op 20141) 
v_lshlrev_b32 v7, 16, v7 // (op 20141) 
v_mul_f32 v18, v9, v6 // (op 20141) 
v_mul_f32 v19, v9, v7 // (op 20141) 
v_and_b32 v6, 65535, v5 // (op 20141) 
v_lshrrev_b32 v7, 16, v5 // (op 20141) 
v_lshlrev_b32 v6, 16, v6 // (op 20141) 
v_lshlrev_b32 v7, 16, v7 // (op 20141) 
v_mul_f32 v20, v9, v6 // (op 20141) 
v_mul_f32 v21, v9, v7 // (op 20141) 


v_add_f32 v4, v18, v14 // (op 20142) 
v_add_f32 v5, v19, v15 // (op 20142) 
v_add_f32 v6, v20, v16 // (op 20142) 
v_add_f32 v7, v21, v17 // (op 20142) 


v_lshrrev_b32 v14, 16, v4 // (op 20143) 
v_lshrrev_b32 v15, 16, v5 // (op 20143) 
v_lshrrev_b32 v16, 16, v6 // (op 20143) 
v_lshrrev_b32 v17, 16, v7 // (op 20143) 


v_pack_b32_f16 v4, v14, v15 // (op 20115) 
v_pack_b32_f16 v5, v16, v17 // (op 20115) 

s_lshl_b32 s0, s4, 7 // (op 20115) 
s_add_u32 s0, s0, 96 // (op 20115) 
v_add_u32 v0, v13, s0 // (op 20115) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20115) Store value





s_lshl_b32 s0, s2, 5 // (op 20149) 
s_mul_i32 s1, 5, s0 // (op 20149) 
s_add_u32 s1, s1, 96 // (op 20149) 

v_add_u32 v0, v12, s1 // (op 20149) 


v_accvgpr_read v4, a116 // (op 20169) 
v_accvgpr_read v5, a117 // (op 20169) 
v_accvgpr_read v6, a118 // (op 20169) 
v_accvgpr_read v7, a119 // (op 20169) 
v_mul_f32 v14, v11, v4 // (op 20169) 
v_mul_f32 v15, v11, v5 // (op 20169) 
v_mul_f32 v16, v11, v6 // (op 20169) 
v_mul_f32 v17, v11, v7 // (op 20169) 

buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20149) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20146) 
v_lshrrev_b32 v7, 16, v4 // (op 20146) 
v_lshlrev_b32 v6, 16, v6 // (op 20146) 
v_lshlrev_b32 v7, 16, v7 // (op 20146) 
v_mul_f32 v18, v9, v6 // (op 20146) 
v_mul_f32 v19, v9, v7 // (op 20146) 
v_and_b32 v6, 65535, v5 // (op 20146) 
v_lshrrev_b32 v7, 16, v5 // (op 20146) 
v_lshlrev_b32 v6, 16, v6 // (op 20146) 
v_lshlrev_b32 v7, 16, v7 // (op 20146) 
v_mul_f32 v20, v9, v6 // (op 20146) 
v_mul_f32 v21, v9, v7 // (op 20146) 


v_add_f32 v4, v18, v14 // (op 20147) 
v_add_f32 v5, v19, v15 // (op 20147) 
v_add_f32 v6, v20, v16 // (op 20147) 
v_add_f32 v7, v21, v17 // (op 20147) 


v_lshrrev_b32 v14, 16, v4 // (op 20148) 
v_lshrrev_b32 v15, 16, v5 // (op 20148) 
v_lshrrev_b32 v16, 16, v6 // (op 20148) 
v_lshrrev_b32 v17, 16, v7 // (op 20148) 


v_pack_b32_f16 v4, v14, v15 // (op 20117) 
v_pack_b32_f16 v5, v16, v17 // (op 20117) 

s_lshl_b32 s0, s4, 5 // (op 20117) 
s_mul_i32 s1, 5, s0 // (op 20117) 
s_add_u32 s1, s1, 96 // (op 20117) 

v_add_u32 v0, v13, s1 // (op 20117) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20117) Store value





s_lshl_b32 s0, s2, 5 // (op 20154) 
s_mul_i32 s1, 6, s0 // (op 20154) 
s_add_u32 s1, s1, 96 // (op 20154) 

v_add_u32 v0, v12, s1 // (op 20154) 


v_accvgpr_read v4, a120 // (op 20171) 
v_accvgpr_read v5, a121 // (op 20171) 
v_accvgpr_read v6, a122 // (op 20171) 
v_accvgpr_read v7, a123 // (op 20171) 
v_mul_f32 v14, v11, v4 // (op 20171) 
v_mul_f32 v15, v11, v5 // (op 20171) 
v_mul_f32 v16, v11, v6 // (op 20171) 
v_mul_f32 v17, v11, v7 // (op 20171) 

buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20154) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20151) 
v_lshrrev_b32 v7, 16, v4 // (op 20151) 
v_lshlrev_b32 v6, 16, v6 // (op 20151) 
v_lshlrev_b32 v7, 16, v7 // (op 20151) 
v_mul_f32 v18, v9, v6 // (op 20151) 
v_mul_f32 v19, v9, v7 // (op 20151) 
v_and_b32 v6, 65535, v5 // (op 20151) 
v_lshrrev_b32 v7, 16, v5 // (op 20151) 
v_lshlrev_b32 v6, 16, v6 // (op 20151) 
v_lshlrev_b32 v7, 16, v7 // (op 20151) 
v_mul_f32 v20, v9, v6 // (op 20151) 
v_mul_f32 v21, v9, v7 // (op 20151) 


v_add_f32 v4, v18, v14 // (op 20152) 
v_add_f32 v5, v19, v15 // (op 20152) 
v_add_f32 v6, v20, v16 // (op 20152) 
v_add_f32 v7, v21, v17 // (op 20152) 


v_lshrrev_b32 v14, 16, v4 // (op 20153) 
v_lshrrev_b32 v15, 16, v5 // (op 20153) 
v_lshrrev_b32 v16, 16, v6 // (op 20153) 
v_lshrrev_b32 v17, 16, v7 // (op 20153) 


v_pack_b32_f16 v4, v14, v15 // (op 20119) 
v_pack_b32_f16 v5, v16, v17 // (op 20119) 

s_lshl_b32 s0, s4, 5 // (op 20119) 
s_mul_i32 s1, 6, s0 // (op 20119) 
s_add_u32 s1, s1, 96 // (op 20119) 

v_add_u32 v0, v13, s1 // (op 20119) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20119) Store value





s_lshl_b32 s0, s2, 5 // (op 20159) 
s_mul_i32 s1, 7, s0 // (op 20159) 
s_add_u32 s1, s1, 96 // (op 20159) 

v_add_u32 v0, v12, s1 // (op 20159) 


v_accvgpr_read v4, a124 // (op 20173) 
v_accvgpr_read v5, a125 // (op 20173) 
v_accvgpr_read v6, a126 // (op 20173) 
v_accvgpr_read v7, a127 // (op 20173) 
v_mul_f32 v14, v11, v4 // (op 20173) 
v_mul_f32 v15, v11, v5 // (op 20173) 
v_mul_f32 v16, v11, v6 // (op 20173) 
v_mul_f32 v17, v11, v7 // (op 20173) 

buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 20159) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 20156) 
v_lshrrev_b32 v7, 16, v4 // (op 20156) 
v_lshlrev_b32 v6, 16, v6 // (op 20156) 
v_lshlrev_b32 v7, 16, v7 // (op 20156) 
v_mul_f32 v18, v9, v6 // (op 20156) 
v_mul_f32 v19, v9, v7 // (op 20156) 
v_and_b32 v6, 65535, v5 // (op 20156) 
v_lshrrev_b32 v7, 16, v5 // (op 20156) 
v_lshlrev_b32 v6, 16, v6 // (op 20156) 
v_lshlrev_b32 v7, 16, v7 // (op 20156) 
v_mul_f32 v20, v9, v6 // (op 20156) 
v_mul_f32 v21, v9, v7 // (op 20156) 


v_add_f32 v4, v18, v14 // (op 20157) 
v_add_f32 v5, v19, v15 // (op 20157) 
v_add_f32 v6, v20, v16 // (op 20157) 
v_add_f32 v7, v21, v17 // (op 20157) 


v_lshrrev_b32 v14, 16, v4 // (op 20158) 
v_lshrrev_b32 v15, 16, v5 // (op 20158) 
v_lshrrev_b32 v16, 16, v6 // (op 20158) 
v_lshrrev_b32 v17, 16, v7 // (op 20158) 


v_pack_b32_f16 v4, v14, v15 // (op 20121) 
v_pack_b32_f16 v5, v16, v17 // (op 20121) 

s_lshl_b32 s0, s4, 5 // (op 20121) 
s_mul_i32 s1, 7, s0 // (op 20121) 
s_add_u32 s1, s1, 96 // (op 20121) 

v_add_u32 v0, v13, s1 // (op 20121) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 20121) Store value





v_add_u32 v0, 64, v12 // (op 19953) 

v_accvgpr_read v4, a64 // (op 19932) 
v_accvgpr_read v5, a65 // (op 19932) 
v_accvgpr_read v6, a66 // (op 19932) 
v_accvgpr_read v7, a67 // (op 19932) 
v_mul_f32 v14, v11, v4 // (op 19932) 
v_mul_f32 v15, v11, v5 // (op 19932) 
v_mul_f32 v16, v11, v6 // (op 19932) 
v_mul_f32 v17, v11, v7 // (op 19932) 

buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 19953) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 19952) 
v_lshrrev_b32 v7, 16, v4 // (op 19952) 
v_lshlrev_b32 v6, 16, v6 // (op 19952) 
v_lshlrev_b32 v7, 16, v7 // (op 19952) 
v_mul_f32 v18, v9, v6 // (op 19952) 
v_mul_f32 v19, v9, v7 // (op 19952) 
v_and_b32 v6, 65535, v5 // (op 19952) 
v_lshrrev_b32 v7, 16, v5 // (op 19952) 
v_lshlrev_b32 v6, 16, v6 // (op 19952) 
v_lshlrev_b32 v7, 16, v7 // (op 19952) 
v_mul_f32 v20, v9, v6 // (op 19952) 
v_mul_f32 v21, v9, v7 // (op 19952) 


v_add_f32 v4, v18, v14 // (op 19933) 
v_add_f32 v5, v19, v15 // (op 19933) 
v_add_f32 v6, v20, v16 // (op 19933) 
v_add_f32 v7, v21, v17 // (op 19933) 


v_lshrrev_b32 v14, 16, v4 // (op 19934) 
v_lshrrev_b32 v15, 16, v5 // (op 19934) 
v_lshrrev_b32 v16, 16, v6 // (op 19934) 
v_lshrrev_b32 v17, 16, v7 // (op 19934) 


v_pack_b32_f16 v4, v14, v15 // (op 19936) 
v_pack_b32_f16 v5, v16, v17 // (op 19936) 

v_add_u32 v0, 64, v13 // (op 19936) 

buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 19936) Store value





s_lshl_b32 s0, s2, 5 // (op 19958) 
s_add_u32 s0, s0, 64 // (op 19958) 
v_add_u32 v0, v12, s0 // (op 19958) 


v_accvgpr_read v4, a68 // (op 19990) 
v_accvgpr_read v5, a69 // (op 19990) 
v_accvgpr_read v6, a70 // (op 19990) 
v_accvgpr_read v7, a71 // (op 19990) 
v_mul_f32 v14, v11, v4 // (op 19990) 
v_mul_f32 v15, v11, v5 // (op 19990) 
v_mul_f32 v16, v11, v6 // (op 19990) 
v_mul_f32 v17, v11, v7 // (op 19990) 

buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 19958) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 19955) 
v_lshrrev_b32 v7, 16, v4 // (op 19955) 
v_lshlrev_b32 v6, 16, v6 // (op 19955) 
v_lshlrev_b32 v7, 16, v7 // (op 19955) 
v_mul_f32 v18, v9, v6 // (op 19955) 
v_mul_f32 v19, v9, v7 // (op 19955) 
v_and_b32 v6, 65535, v5 // (op 19955) 
v_lshrrev_b32 v7, 16, v5 // (op 19955) 
v_lshlrev_b32 v6, 16, v6 // (op 19955) 
v_lshlrev_b32 v7, 16, v7 // (op 19955) 
v_mul_f32 v20, v9, v6 // (op 19955) 
v_mul_f32 v21, v9, v7 // (op 19955) 


v_add_f32 v4, v18, v14 // (op 19956) 
v_add_f32 v5, v19, v15 // (op 19956) 
v_add_f32 v6, v20, v16 // (op 19956) 
v_add_f32 v7, v21, v17 // (op 19956) 


v_lshrrev_b32 v14, 16, v4 // (op 19957) 
v_lshrrev_b32 v15, 16, v5 // (op 19957) 
v_lshrrev_b32 v16, 16, v6 // (op 19957) 
v_lshrrev_b32 v17, 16, v7 // (op 19957) 


v_pack_b32_f16 v4, v14, v15 // (op 19938) 
v_pack_b32_f16 v5, v16, v17 // (op 19938) 

s_lshl_b32 s0, s4, 5 // (op 19938) 
s_add_u32 s0, s0, 64 // (op 19938) 
v_add_u32 v0, v13, s0 // (op 19938) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 19938) Store value





s_lshl_b32 s0, s2, 6 // (op 19963) 
s_add_u32 s0, s0, 64 // (op 19963) 
v_add_u32 v0, v12, s0 // (op 19963) 


v_accvgpr_read v4, a72 // (op 19992) 
v_accvgpr_read v5, a73 // (op 19992) 
v_accvgpr_read v6, a74 // (op 19992) 
v_accvgpr_read v7, a75 // (op 19992) 
v_mul_f32 v14, v11, v4 // (op 19992) 
v_mul_f32 v15, v11, v5 // (op 19992) 
v_mul_f32 v16, v11, v6 // (op 19992) 
v_mul_f32 v17, v11, v7 // (op 19992) 

buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 19963) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 19960) 
v_lshrrev_b32 v7, 16, v4 // (op 19960) 
v_lshlrev_b32 v6, 16, v6 // (op 19960) 
v_lshlrev_b32 v7, 16, v7 // (op 19960) 
v_mul_f32 v18, v9, v6 // (op 19960) 
v_mul_f32 v19, v9, v7 // (op 19960) 
v_and_b32 v6, 65535, v5 // (op 19960) 
v_lshrrev_b32 v7, 16, v5 // (op 19960) 
v_lshlrev_b32 v6, 16, v6 // (op 19960) 
v_lshlrev_b32 v7, 16, v7 // (op 19960) 
v_mul_f32 v20, v9, v6 // (op 19960) 
v_mul_f32 v21, v9, v7 // (op 19960) 


v_add_f32 v4, v18, v14 // (op 19961) 
v_add_f32 v5, v19, v15 // (op 19961) 
v_add_f32 v6, v20, v16 // (op 19961) 
v_add_f32 v7, v21, v17 // (op 19961) 


v_lshrrev_b32 v14, 16, v4 // (op 19962) 
v_lshrrev_b32 v15, 16, v5 // (op 19962) 
v_lshrrev_b32 v16, 16, v6 // (op 19962) 
v_lshrrev_b32 v17, 16, v7 // (op 19962) 


v_pack_b32_f16 v4, v14, v15 // (op 19940) 
v_pack_b32_f16 v5, v16, v17 // (op 19940) 

s_lshl_b32 s0, s4, 6 // (op 19940) 
s_add_u32 s0, s0, 64 // (op 19940) 
v_add_u32 v0, v13, s0 // (op 19940) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 19940) Store value





s_lshl_b32 s0, s2, 5 // (op 19968) 
s_mul_i32 s1, 3, s0 // (op 19968) 
s_add_u32 s1, s1, 64 // (op 19968) 

v_add_u32 v0, v12, s1 // (op 19968) 


v_accvgpr_read v4, a76 // (op 19994) 
v_accvgpr_read v5, a77 // (op 19994) 
v_accvgpr_read v6, a78 // (op 19994) 
v_accvgpr_read v7, a79 // (op 19994) 
v_mul_f32 v14, v11, v4 // (op 19994) 
v_mul_f32 v15, v11, v5 // (op 19994) 
v_mul_f32 v16, v11, v6 // (op 19994) 
v_mul_f32 v17, v11, v7 // (op 19994) 

buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 19968) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 19965) 
v_lshrrev_b32 v7, 16, v4 // (op 19965) 
v_lshlrev_b32 v6, 16, v6 // (op 19965) 
v_lshlrev_b32 v7, 16, v7 // (op 19965) 
v_mul_f32 v18, v9, v6 // (op 19965) 
v_mul_f32 v19, v9, v7 // (op 19965) 
v_and_b32 v6, 65535, v5 // (op 19965) 
v_lshrrev_b32 v7, 16, v5 // (op 19965) 
v_lshlrev_b32 v6, 16, v6 // (op 19965) 
v_lshlrev_b32 v7, 16, v7 // (op 19965) 
v_mul_f32 v20, v9, v6 // (op 19965) 
v_mul_f32 v21, v9, v7 // (op 19965) 


v_add_f32 v4, v18, v14 // (op 19966) 
v_add_f32 v5, v19, v15 // (op 19966) 
v_add_f32 v6, v20, v16 // (op 19966) 
v_add_f32 v7, v21, v17 // (op 19966) 


v_lshrrev_b32 v14, 16, v4 // (op 19967) 
v_lshrrev_b32 v15, 16, v5 // (op 19967) 
v_lshrrev_b32 v16, 16, v6 // (op 19967) 
v_lshrrev_b32 v17, 16, v7 // (op 19967) 


v_pack_b32_f16 v4, v14, v15 // (op 19942) 
v_pack_b32_f16 v5, v16, v17 // (op 19942) 

s_lshl_b32 s0, s4, 5 // (op 19942) 
s_mul_i32 s1, 3, s0 // (op 19942) 
s_add_u32 s1, s1, 64 // (op 19942) 

v_add_u32 v0, v13, s1 // (op 19942) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 19942) Store value





s_lshl_b32 s0, s2, 7 // (op 19973) 
s_add_u32 s0, s0, 64 // (op 19973) 
v_add_u32 v0, v12, s0 // (op 19973) 


v_accvgpr_read v4, a80 // (op 19996) 
v_accvgpr_read v5, a81 // (op 19996) 
v_accvgpr_read v6, a82 // (op 19996) 
v_accvgpr_read v7, a83 // (op 19996) 
v_mul_f32 v14, v11, v4 // (op 19996) 
v_mul_f32 v15, v11, v5 // (op 19996) 
v_mul_f32 v16, v11, v6 // (op 19996) 
v_mul_f32 v17, v11, v7 // (op 19996) 

buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 19973) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 19970) 
v_lshrrev_b32 v7, 16, v4 // (op 19970) 
v_lshlrev_b32 v6, 16, v6 // (op 19970) 
v_lshlrev_b32 v7, 16, v7 // (op 19970) 
v_mul_f32 v18, v9, v6 // (op 19970) 
v_mul_f32 v19, v9, v7 // (op 19970) 
v_and_b32 v6, 65535, v5 // (op 19970) 
v_lshrrev_b32 v7, 16, v5 // (op 19970) 
v_lshlrev_b32 v6, 16, v6 // (op 19970) 
v_lshlrev_b32 v7, 16, v7 // (op 19970) 
v_mul_f32 v20, v9, v6 // (op 19970) 
v_mul_f32 v21, v9, v7 // (op 19970) 


v_add_f32 v4, v18, v14 // (op 19971) 
v_add_f32 v5, v19, v15 // (op 19971) 
v_add_f32 v6, v20, v16 // (op 19971) 
v_add_f32 v7, v21, v17 // (op 19971) 


v_lshrrev_b32 v14, 16, v4 // (op 19972) 
v_lshrrev_b32 v15, 16, v5 // (op 19972) 
v_lshrrev_b32 v16, 16, v6 // (op 19972) 
v_lshrrev_b32 v17, 16, v7 // (op 19972) 


v_pack_b32_f16 v4, v14, v15 // (op 19944) 
v_pack_b32_f16 v5, v16, v17 // (op 19944) 

s_lshl_b32 s0, s4, 7 // (op 19944) 
s_add_u32 s0, s0, 64 // (op 19944) 
v_add_u32 v0, v13, s0 // (op 19944) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 19944) Store value





s_lshl_b32 s0, s2, 5 // (op 19978) 
s_mul_i32 s1, 5, s0 // (op 19978) 
s_add_u32 s1, s1, 64 // (op 19978) 

v_add_u32 v0, v12, s1 // (op 19978) 


v_accvgpr_read v4, a84 // (op 19998) 
v_accvgpr_read v5, a85 // (op 19998) 
v_accvgpr_read v6, a86 // (op 19998) 
v_accvgpr_read v7, a87 // (op 19998) 
v_mul_f32 v14, v11, v4 // (op 19998) 
v_mul_f32 v15, v11, v5 // (op 19998) 
v_mul_f32 v16, v11, v6 // (op 19998) 
v_mul_f32 v17, v11, v7 // (op 19998) 

buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 19978) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 19975) 
v_lshrrev_b32 v7, 16, v4 // (op 19975) 
v_lshlrev_b32 v6, 16, v6 // (op 19975) 
v_lshlrev_b32 v7, 16, v7 // (op 19975) 
v_mul_f32 v18, v9, v6 // (op 19975) 
v_mul_f32 v19, v9, v7 // (op 19975) 
v_and_b32 v6, 65535, v5 // (op 19975) 
v_lshrrev_b32 v7, 16, v5 // (op 19975) 
v_lshlrev_b32 v6, 16, v6 // (op 19975) 
v_lshlrev_b32 v7, 16, v7 // (op 19975) 
v_mul_f32 v20, v9, v6 // (op 19975) 
v_mul_f32 v21, v9, v7 // (op 19975) 


v_add_f32 v4, v18, v14 // (op 19976) 
v_add_f32 v5, v19, v15 // (op 19976) 
v_add_f32 v6, v20, v16 // (op 19976) 
v_add_f32 v7, v21, v17 // (op 19976) 


v_lshrrev_b32 v14, 16, v4 // (op 19977) 
v_lshrrev_b32 v15, 16, v5 // (op 19977) 
v_lshrrev_b32 v16, 16, v6 // (op 19977) 
v_lshrrev_b32 v17, 16, v7 // (op 19977) 


v_pack_b32_f16 v4, v14, v15 // (op 19946) 
v_pack_b32_f16 v5, v16, v17 // (op 19946) 

s_lshl_b32 s0, s4, 5 // (op 19946) 
s_mul_i32 s1, 5, s0 // (op 19946) 
s_add_u32 s1, s1, 64 // (op 19946) 

v_add_u32 v0, v13, s1 // (op 19946) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 19946) Store value





s_lshl_b32 s0, s2, 5 // (op 19983) 
s_mul_i32 s1, 6, s0 // (op 19983) 
s_add_u32 s1, s1, 64 // (op 19983) 

v_add_u32 v0, v12, s1 // (op 19983) 


v_accvgpr_read v4, a88 // (op 20000) 
v_accvgpr_read v5, a89 // (op 20000) 
v_accvgpr_read v6, a90 // (op 20000) 
v_accvgpr_read v7, a91 // (op 20000) 
v_mul_f32 v14, v11, v4 // (op 20000) 
v_mul_f32 v15, v11, v5 // (op 20000) 
v_mul_f32 v16, v11, v6 // (op 20000) 
v_mul_f32 v17, v11, v7 // (op 20000) 

buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 19983) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 19980) 
v_lshrrev_b32 v7, 16, v4 // (op 19980) 
v_lshlrev_b32 v6, 16, v6 // (op 19980) 
v_lshlrev_b32 v7, 16, v7 // (op 19980) 
v_mul_f32 v18, v9, v6 // (op 19980) 
v_mul_f32 v19, v9, v7 // (op 19980) 
v_and_b32 v6, 65535, v5 // (op 19980) 
v_lshrrev_b32 v7, 16, v5 // (op 19980) 
v_lshlrev_b32 v6, 16, v6 // (op 19980) 
v_lshlrev_b32 v7, 16, v7 // (op 19980) 
v_mul_f32 v20, v9, v6 // (op 19980) 
v_mul_f32 v21, v9, v7 // (op 19980) 


v_add_f32 v4, v18, v14 // (op 19981) 
v_add_f32 v5, v19, v15 // (op 19981) 
v_add_f32 v6, v20, v16 // (op 19981) 
v_add_f32 v7, v21, v17 // (op 19981) 


v_lshrrev_b32 v14, 16, v4 // (op 19982) 
v_lshrrev_b32 v15, 16, v5 // (op 19982) 
v_lshrrev_b32 v16, 16, v6 // (op 19982) 
v_lshrrev_b32 v17, 16, v7 // (op 19982) 


v_pack_b32_f16 v4, v14, v15 // (op 19948) 
v_pack_b32_f16 v5, v16, v17 // (op 19948) 

s_lshl_b32 s0, s4, 5 // (op 19948) 
s_mul_i32 s1, 6, s0 // (op 19948) 
s_add_u32 s1, s1, 64 // (op 19948) 

v_add_u32 v0, v13, s1 // (op 19948) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 19948) Store value





s_lshl_b32 s0, s2, 5 // (op 19988) 
s_mul_i32 s1, 7, s0 // (op 19988) 
s_add_u32 s1, s1, 64 // (op 19988) 

v_add_u32 v0, v12, s1 // (op 19988) 


v_accvgpr_read v4, a92 // (op 20002) 
v_accvgpr_read v5, a93 // (op 20002) 
v_accvgpr_read v6, a94 // (op 20002) 
v_accvgpr_read v7, a95 // (op 20002) 
v_mul_f32 v14, v11, v4 // (op 20002) 
v_mul_f32 v15, v11, v5 // (op 20002) 
v_mul_f32 v16, v11, v6 // (op 20002) 
v_mul_f32 v17, v11, v7 // (op 20002) 

buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 19988) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 19985) 
v_lshrrev_b32 v7, 16, v4 // (op 19985) 
v_lshlrev_b32 v6, 16, v6 // (op 19985) 
v_lshlrev_b32 v7, 16, v7 // (op 19985) 
v_mul_f32 v18, v9, v6 // (op 19985) 
v_mul_f32 v19, v9, v7 // (op 19985) 
v_and_b32 v6, 65535, v5 // (op 19985) 
v_lshrrev_b32 v7, 16, v5 // (op 19985) 
v_lshlrev_b32 v6, 16, v6 // (op 19985) 
v_lshlrev_b32 v7, 16, v7 // (op 19985) 
v_mul_f32 v20, v9, v6 // (op 19985) 
v_mul_f32 v21, v9, v7 // (op 19985) 


v_add_f32 v4, v18, v14 // (op 19986) 
v_add_f32 v5, v19, v15 // (op 19986) 
v_add_f32 v6, v20, v16 // (op 19986) 
v_add_f32 v7, v21, v17 // (op 19986) 


v_lshrrev_b32 v14, 16, v4 // (op 19987) 
v_lshrrev_b32 v15, 16, v5 // (op 19987) 
v_lshrrev_b32 v16, 16, v6 // (op 19987) 
v_lshrrev_b32 v17, 16, v7 // (op 19987) 


v_pack_b32_f16 v4, v14, v15 // (op 19950) 
v_pack_b32_f16 v5, v16, v17 // (op 19950) 

s_lshl_b32 s0, s4, 5 // (op 19950) 
s_mul_i32 s1, 7, s0 // (op 19950) 
s_add_u32 s1, s1, 64 // (op 19950) 

v_add_u32 v0, v13, s1 // (op 19950) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 19950) Store value





v_add_u32 v0, 32, v12 // (op 19782) 

v_accvgpr_read v4, a32 // (op 19761) 
v_accvgpr_read v5, a33 // (op 19761) 
v_accvgpr_read v6, a34 // (op 19761) 
v_accvgpr_read v7, a35 // (op 19761) 
v_mul_f32 v14, v11, v4 // (op 19761) 
v_mul_f32 v15, v11, v5 // (op 19761) 
v_mul_f32 v16, v11, v6 // (op 19761) 
v_mul_f32 v17, v11, v7 // (op 19761) 

buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 19782) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 19781) 
v_lshrrev_b32 v7, 16, v4 // (op 19781) 
v_lshlrev_b32 v6, 16, v6 // (op 19781) 
v_lshlrev_b32 v7, 16, v7 // (op 19781) 
v_mul_f32 v18, v9, v6 // (op 19781) 
v_mul_f32 v19, v9, v7 // (op 19781) 
v_and_b32 v6, 65535, v5 // (op 19781) 
v_lshrrev_b32 v7, 16, v5 // (op 19781) 
v_lshlrev_b32 v6, 16, v6 // (op 19781) 
v_lshlrev_b32 v7, 16, v7 // (op 19781) 
v_mul_f32 v20, v9, v6 // (op 19781) 
v_mul_f32 v21, v9, v7 // (op 19781) 


v_add_f32 v4, v18, v14 // (op 19762) 
v_add_f32 v5, v19, v15 // (op 19762) 
v_add_f32 v6, v20, v16 // (op 19762) 
v_add_f32 v7, v21, v17 // (op 19762) 


v_lshrrev_b32 v14, 16, v4 // (op 19763) 
v_lshrrev_b32 v15, 16, v5 // (op 19763) 
v_lshrrev_b32 v16, 16, v6 // (op 19763) 
v_lshrrev_b32 v17, 16, v7 // (op 19763) 


v_pack_b32_f16 v4, v14, v15 // (op 19765) 
v_pack_b32_f16 v5, v16, v17 // (op 19765) 

v_add_u32 v0, 32, v13 // (op 19765) 

buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 19765) Store value





s_lshl_b32 s0, s2, 5 // (op 19787) 
s_add_u32 s0, s0, 32 // (op 19787) 
v_add_u32 v0, v12, s0 // (op 19787) 


v_accvgpr_read v4, a36 // (op 19819) 
v_accvgpr_read v5, a37 // (op 19819) 
v_accvgpr_read v6, a38 // (op 19819) 
v_accvgpr_read v7, a39 // (op 19819) 
v_mul_f32 v14, v11, v4 // (op 19819) 
v_mul_f32 v15, v11, v5 // (op 19819) 
v_mul_f32 v16, v11, v6 // (op 19819) 
v_mul_f32 v17, v11, v7 // (op 19819) 

buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 19787) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 19784) 
v_lshrrev_b32 v7, 16, v4 // (op 19784) 
v_lshlrev_b32 v6, 16, v6 // (op 19784) 
v_lshlrev_b32 v7, 16, v7 // (op 19784) 
v_mul_f32 v18, v9, v6 // (op 19784) 
v_mul_f32 v19, v9, v7 // (op 19784) 
v_and_b32 v6, 65535, v5 // (op 19784) 
v_lshrrev_b32 v7, 16, v5 // (op 19784) 
v_lshlrev_b32 v6, 16, v6 // (op 19784) 
v_lshlrev_b32 v7, 16, v7 // (op 19784) 
v_mul_f32 v20, v9, v6 // (op 19784) 
v_mul_f32 v21, v9, v7 // (op 19784) 


v_add_f32 v4, v18, v14 // (op 19785) 
v_add_f32 v5, v19, v15 // (op 19785) 
v_add_f32 v6, v20, v16 // (op 19785) 
v_add_f32 v7, v21, v17 // (op 19785) 


v_lshrrev_b32 v14, 16, v4 // (op 19786) 
v_lshrrev_b32 v15, 16, v5 // (op 19786) 
v_lshrrev_b32 v16, 16, v6 // (op 19786) 
v_lshrrev_b32 v17, 16, v7 // (op 19786) 


v_pack_b32_f16 v4, v14, v15 // (op 19767) 
v_pack_b32_f16 v5, v16, v17 // (op 19767) 

s_lshl_b32 s0, s4, 5 // (op 19767) 
s_add_u32 s0, s0, 32 // (op 19767) 
v_add_u32 v0, v13, s0 // (op 19767) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 19767) Store value





s_lshl_b32 s0, s2, 6 // (op 19792) 
s_add_u32 s0, s0, 32 // (op 19792) 
v_add_u32 v0, v12, s0 // (op 19792) 


v_accvgpr_read v4, a40 // (op 19821) 
v_accvgpr_read v5, a41 // (op 19821) 
v_accvgpr_read v6, a42 // (op 19821) 
v_accvgpr_read v7, a43 // (op 19821) 
v_mul_f32 v14, v11, v4 // (op 19821) 
v_mul_f32 v15, v11, v5 // (op 19821) 
v_mul_f32 v16, v11, v6 // (op 19821) 
v_mul_f32 v17, v11, v7 // (op 19821) 

buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 19792) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 19789) 
v_lshrrev_b32 v7, 16, v4 // (op 19789) 
v_lshlrev_b32 v6, 16, v6 // (op 19789) 
v_lshlrev_b32 v7, 16, v7 // (op 19789) 
v_mul_f32 v18, v9, v6 // (op 19789) 
v_mul_f32 v19, v9, v7 // (op 19789) 
v_and_b32 v6, 65535, v5 // (op 19789) 
v_lshrrev_b32 v7, 16, v5 // (op 19789) 
v_lshlrev_b32 v6, 16, v6 // (op 19789) 
v_lshlrev_b32 v7, 16, v7 // (op 19789) 
v_mul_f32 v20, v9, v6 // (op 19789) 
v_mul_f32 v21, v9, v7 // (op 19789) 


v_add_f32 v4, v18, v14 // (op 19790) 
v_add_f32 v5, v19, v15 // (op 19790) 
v_add_f32 v6, v20, v16 // (op 19790) 
v_add_f32 v7, v21, v17 // (op 19790) 


v_lshrrev_b32 v14, 16, v4 // (op 19791) 
v_lshrrev_b32 v15, 16, v5 // (op 19791) 
v_lshrrev_b32 v16, 16, v6 // (op 19791) 
v_lshrrev_b32 v17, 16, v7 // (op 19791) 


v_pack_b32_f16 v4, v14, v15 // (op 19769) 
v_pack_b32_f16 v5, v16, v17 // (op 19769) 

s_lshl_b32 s0, s4, 6 // (op 19769) 
s_add_u32 s0, s0, 32 // (op 19769) 
v_add_u32 v0, v13, s0 // (op 19769) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 19769) Store value





s_lshl_b32 s0, s2, 5 // (op 19797) 
s_mul_i32 s1, 3, s0 // (op 19797) 
s_add_u32 s1, s1, 32 // (op 19797) 

v_add_u32 v0, v12, s1 // (op 19797) 


v_accvgpr_read v4, a44 // (op 19823) 
v_accvgpr_read v5, a45 // (op 19823) 
v_accvgpr_read v6, a46 // (op 19823) 
v_accvgpr_read v7, a47 // (op 19823) 
v_mul_f32 v14, v11, v4 // (op 19823) 
v_mul_f32 v15, v11, v5 // (op 19823) 
v_mul_f32 v16, v11, v6 // (op 19823) 
v_mul_f32 v17, v11, v7 // (op 19823) 

buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 19797) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 19794) 
v_lshrrev_b32 v7, 16, v4 // (op 19794) 
v_lshlrev_b32 v6, 16, v6 // (op 19794) 
v_lshlrev_b32 v7, 16, v7 // (op 19794) 
v_mul_f32 v18, v9, v6 // (op 19794) 
v_mul_f32 v19, v9, v7 // (op 19794) 
v_and_b32 v6, 65535, v5 // (op 19794) 
v_lshrrev_b32 v7, 16, v5 // (op 19794) 
v_lshlrev_b32 v6, 16, v6 // (op 19794) 
v_lshlrev_b32 v7, 16, v7 // (op 19794) 
v_mul_f32 v20, v9, v6 // (op 19794) 
v_mul_f32 v21, v9, v7 // (op 19794) 


v_add_f32 v4, v18, v14 // (op 19795) 
v_add_f32 v5, v19, v15 // (op 19795) 
v_add_f32 v6, v20, v16 // (op 19795) 
v_add_f32 v7, v21, v17 // (op 19795) 


v_lshrrev_b32 v14, 16, v4 // (op 19796) 
v_lshrrev_b32 v15, 16, v5 // (op 19796) 
v_lshrrev_b32 v16, 16, v6 // (op 19796) 
v_lshrrev_b32 v17, 16, v7 // (op 19796) 


v_pack_b32_f16 v4, v14, v15 // (op 19771) 
v_pack_b32_f16 v5, v16, v17 // (op 19771) 

s_lshl_b32 s0, s4, 5 // (op 19771) 
s_mul_i32 s1, 3, s0 // (op 19771) 
s_add_u32 s1, s1, 32 // (op 19771) 

v_add_u32 v0, v13, s1 // (op 19771) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 19771) Store value





s_lshl_b32 s0, s2, 7 // (op 19802) 
s_add_u32 s0, s0, 32 // (op 19802) 
v_add_u32 v0, v12, s0 // (op 19802) 


v_accvgpr_read v4, a48 // (op 19825) 
v_accvgpr_read v5, a49 // (op 19825) 
v_accvgpr_read v6, a50 // (op 19825) 
v_accvgpr_read v7, a51 // (op 19825) 
v_mul_f32 v14, v11, v4 // (op 19825) 
v_mul_f32 v15, v11, v5 // (op 19825) 
v_mul_f32 v16, v11, v6 // (op 19825) 
v_mul_f32 v17, v11, v7 // (op 19825) 

buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 19802) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 19799) 
v_lshrrev_b32 v7, 16, v4 // (op 19799) 
v_lshlrev_b32 v6, 16, v6 // (op 19799) 
v_lshlrev_b32 v7, 16, v7 // (op 19799) 
v_mul_f32 v18, v9, v6 // (op 19799) 
v_mul_f32 v19, v9, v7 // (op 19799) 
v_and_b32 v6, 65535, v5 // (op 19799) 
v_lshrrev_b32 v7, 16, v5 // (op 19799) 
v_lshlrev_b32 v6, 16, v6 // (op 19799) 
v_lshlrev_b32 v7, 16, v7 // (op 19799) 
v_mul_f32 v20, v9, v6 // (op 19799) 
v_mul_f32 v21, v9, v7 // (op 19799) 


v_add_f32 v4, v18, v14 // (op 19800) 
v_add_f32 v5, v19, v15 // (op 19800) 
v_add_f32 v6, v20, v16 // (op 19800) 
v_add_f32 v7, v21, v17 // (op 19800) 


v_lshrrev_b32 v14, 16, v4 // (op 19801) 
v_lshrrev_b32 v15, 16, v5 // (op 19801) 
v_lshrrev_b32 v16, 16, v6 // (op 19801) 
v_lshrrev_b32 v17, 16, v7 // (op 19801) 


v_pack_b32_f16 v4, v14, v15 // (op 19773) 
v_pack_b32_f16 v5, v16, v17 // (op 19773) 

s_lshl_b32 s0, s4, 7 // (op 19773) 
s_add_u32 s0, s0, 32 // (op 19773) 
v_add_u32 v0, v13, s0 // (op 19773) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 19773) Store value





s_lshl_b32 s0, s2, 5 // (op 19807) 
s_mul_i32 s1, 5, s0 // (op 19807) 
s_add_u32 s1, s1, 32 // (op 19807) 

v_add_u32 v0, v12, s1 // (op 19807) 


v_accvgpr_read v4, a52 // (op 19827) 
v_accvgpr_read v5, a53 // (op 19827) 
v_accvgpr_read v6, a54 // (op 19827) 
v_accvgpr_read v7, a55 // (op 19827) 
v_mul_f32 v14, v11, v4 // (op 19827) 
v_mul_f32 v15, v11, v5 // (op 19827) 
v_mul_f32 v16, v11, v6 // (op 19827) 
v_mul_f32 v17, v11, v7 // (op 19827) 

buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 19807) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 19804) 
v_lshrrev_b32 v7, 16, v4 // (op 19804) 
v_lshlrev_b32 v6, 16, v6 // (op 19804) 
v_lshlrev_b32 v7, 16, v7 // (op 19804) 
v_mul_f32 v18, v9, v6 // (op 19804) 
v_mul_f32 v19, v9, v7 // (op 19804) 
v_and_b32 v6, 65535, v5 // (op 19804) 
v_lshrrev_b32 v7, 16, v5 // (op 19804) 
v_lshlrev_b32 v6, 16, v6 // (op 19804) 
v_lshlrev_b32 v7, 16, v7 // (op 19804) 
v_mul_f32 v20, v9, v6 // (op 19804) 
v_mul_f32 v21, v9, v7 // (op 19804) 


v_add_f32 v4, v18, v14 // (op 19805) 
v_add_f32 v5, v19, v15 // (op 19805) 
v_add_f32 v6, v20, v16 // (op 19805) 
v_add_f32 v7, v21, v17 // (op 19805) 


v_lshrrev_b32 v14, 16, v4 // (op 19806) 
v_lshrrev_b32 v15, 16, v5 // (op 19806) 
v_lshrrev_b32 v16, 16, v6 // (op 19806) 
v_lshrrev_b32 v17, 16, v7 // (op 19806) 


v_pack_b32_f16 v4, v14, v15 // (op 19775) 
v_pack_b32_f16 v5, v16, v17 // (op 19775) 

s_lshl_b32 s0, s4, 5 // (op 19775) 
s_mul_i32 s1, 5, s0 // (op 19775) 
s_add_u32 s1, s1, 32 // (op 19775) 

v_add_u32 v0, v13, s1 // (op 19775) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 19775) Store value





s_lshl_b32 s0, s2, 5 // (op 19812) 
s_mul_i32 s1, 6, s0 // (op 19812) 
s_add_u32 s1, s1, 32 // (op 19812) 

v_add_u32 v0, v12, s1 // (op 19812) 


v_accvgpr_read v4, a56 // (op 19829) 
v_accvgpr_read v5, a57 // (op 19829) 
v_accvgpr_read v6, a58 // (op 19829) 
v_accvgpr_read v7, a59 // (op 19829) 
v_mul_f32 v14, v11, v4 // (op 19829) 
v_mul_f32 v15, v11, v5 // (op 19829) 
v_mul_f32 v16, v11, v6 // (op 19829) 
v_mul_f32 v17, v11, v7 // (op 19829) 

buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 19812) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 19809) 
v_lshrrev_b32 v7, 16, v4 // (op 19809) 
v_lshlrev_b32 v6, 16, v6 // (op 19809) 
v_lshlrev_b32 v7, 16, v7 // (op 19809) 
v_mul_f32 v18, v9, v6 // (op 19809) 
v_mul_f32 v19, v9, v7 // (op 19809) 
v_and_b32 v6, 65535, v5 // (op 19809) 
v_lshrrev_b32 v7, 16, v5 // (op 19809) 
v_lshlrev_b32 v6, 16, v6 // (op 19809) 
v_lshlrev_b32 v7, 16, v7 // (op 19809) 
v_mul_f32 v20, v9, v6 // (op 19809) 
v_mul_f32 v21, v9, v7 // (op 19809) 


v_add_f32 v4, v18, v14 // (op 19810) 
v_add_f32 v5, v19, v15 // (op 19810) 
v_add_f32 v6, v20, v16 // (op 19810) 
v_add_f32 v7, v21, v17 // (op 19810) 


v_lshrrev_b32 v14, 16, v4 // (op 19811) 
v_lshrrev_b32 v15, 16, v5 // (op 19811) 
v_lshrrev_b32 v16, 16, v6 // (op 19811) 
v_lshrrev_b32 v17, 16, v7 // (op 19811) 


v_pack_b32_f16 v4, v14, v15 // (op 19777) 
v_pack_b32_f16 v5, v16, v17 // (op 19777) 

s_lshl_b32 s0, s4, 5 // (op 19777) 
s_mul_i32 s1, 6, s0 // (op 19777) 
s_add_u32 s1, s1, 32 // (op 19777) 

v_add_u32 v0, v13, s1 // (op 19777) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 19777) Store value





s_lshl_b32 s0, s2, 5 // (op 19817) 
s_mul_i32 s1, 7, s0 // (op 19817) 
s_add_u32 s1, s1, 32 // (op 19817) 

v_add_u32 v0, v12, s1 // (op 19817) 


v_accvgpr_read v4, a60 // (op 19831) 
v_accvgpr_read v5, a61 // (op 19831) 
v_accvgpr_read v6, a62 // (op 19831) 
v_accvgpr_read v7, a63 // (op 19831) 
v_mul_f32 v14, v11, v4 // (op 19831) 
v_mul_f32 v15, v11, v5 // (op 19831) 
v_mul_f32 v16, v11, v6 // (op 19831) 
v_mul_f32 v17, v11, v7 // (op 19831) 

buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 19817) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 19814) 
v_lshrrev_b32 v7, 16, v4 // (op 19814) 
v_lshlrev_b32 v6, 16, v6 // (op 19814) 
v_lshlrev_b32 v7, 16, v7 // (op 19814) 
v_mul_f32 v18, v9, v6 // (op 19814) 
v_mul_f32 v19, v9, v7 // (op 19814) 
v_and_b32 v6, 65535, v5 // (op 19814) 
v_lshrrev_b32 v7, 16, v5 // (op 19814) 
v_lshlrev_b32 v6, 16, v6 // (op 19814) 
v_lshlrev_b32 v7, 16, v7 // (op 19814) 
v_mul_f32 v20, v9, v6 // (op 19814) 
v_mul_f32 v21, v9, v7 // (op 19814) 


v_add_f32 v4, v18, v14 // (op 19815) 
v_add_f32 v5, v19, v15 // (op 19815) 
v_add_f32 v6, v20, v16 // (op 19815) 
v_add_f32 v7, v21, v17 // (op 19815) 


v_lshrrev_b32 v14, 16, v4 // (op 19816) 
v_lshrrev_b32 v15, 16, v5 // (op 19816) 
v_lshrrev_b32 v16, 16, v6 // (op 19816) 
v_lshrrev_b32 v17, 16, v7 // (op 19816) 


v_pack_b32_f16 v4, v14, v15 // (op 19779) 
v_pack_b32_f16 v5, v16, v17 // (op 19779) 

s_lshl_b32 s0, s4, 5 // (op 19779) 
s_mul_i32 s1, 7, s0 // (op 19779) 
s_add_u32 s1, s1, 32 // (op 19779) 

v_add_u32 v0, v13, s1 // (op 19779) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 19779) Store value





v_mov_b32 v0, v12 // (op 129) reg expression

v_accvgpr_read v4, a0 // (op 125) 
v_accvgpr_read v5, a1 // (op 125) 
v_accvgpr_read v6, a2 // (op 125) 
v_accvgpr_read v7, a3 // (op 125) 
v_mul_f32 v14, v11, v4 // (op 125) 
v_mul_f32 v15, v11, v5 // (op 125) 
v_mul_f32 v16, v11, v6 // (op 125) 
v_mul_f32 v17, v11, v7 // (op 125) 

buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 129) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 130) 
v_lshrrev_b32 v7, 16, v4 // (op 130) 
v_lshlrev_b32 v6, 16, v6 // (op 130) 
v_lshlrev_b32 v7, 16, v7 // (op 130) 
v_mul_f32 v18, v9, v6 // (op 130) 
v_mul_f32 v19, v9, v7 // (op 130) 
v_and_b32 v6, 65535, v5 // (op 130) 
v_lshrrev_b32 v7, 16, v5 // (op 130) 
v_lshlrev_b32 v6, 16, v6 // (op 130) 
v_lshlrev_b32 v7, 16, v7 // (op 130) 
v_mul_f32 v20, v9, v6 // (op 130) 
v_mul_f32 v21, v9, v7 // (op 130) 


v_add_f32 v4, v18, v14 // (op 126) 
v_add_f32 v5, v19, v15 // (op 126) 
v_add_f32 v6, v20, v16 // (op 126) 
v_add_f32 v7, v21, v17 // (op 126) 


v_lshrrev_b32 v14, 16, v4 // (op 127) 
v_lshrrev_b32 v15, 16, v5 // (op 127) 
v_lshrrev_b32 v16, 16, v6 // (op 127) 
v_lshrrev_b32 v17, 16, v7 // (op 127) 


v_pack_b32_f16 v4, v14, v15 // (op 128) 
v_pack_b32_f16 v5, v16, v17 // (op 128) 

v_mov_b32 v0, v13 // (op 128) reg expression

buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 128) Store value





s_lshl_b32 s0, s2, 5 // (op 19586) 
v_add_u32 v0, v12, s0 // (op 19586) 


v_accvgpr_read v4, a4 // (op 19582) 
v_accvgpr_read v5, a5 // (op 19582) 
v_accvgpr_read v6, a6 // (op 19582) 
v_accvgpr_read v7, a7 // (op 19582) 
v_mul_f32 v14, v11, v4 // (op 19582) 
v_mul_f32 v15, v11, v5 // (op 19582) 
v_mul_f32 v16, v11, v6 // (op 19582) 
v_mul_f32 v17, v11, v7 // (op 19582) 

buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 19586) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 19587) 
v_lshrrev_b32 v7, 16, v4 // (op 19587) 
v_lshlrev_b32 v6, 16, v6 // (op 19587) 
v_lshlrev_b32 v7, 16, v7 // (op 19587) 
v_mul_f32 v18, v9, v6 // (op 19587) 
v_mul_f32 v19, v9, v7 // (op 19587) 
v_and_b32 v6, 65535, v5 // (op 19587) 
v_lshrrev_b32 v7, 16, v5 // (op 19587) 
v_lshlrev_b32 v6, 16, v6 // (op 19587) 
v_lshlrev_b32 v7, 16, v7 // (op 19587) 
v_mul_f32 v20, v9, v6 // (op 19587) 
v_mul_f32 v21, v9, v7 // (op 19587) 


v_add_f32 v4, v18, v14 // (op 19583) 
v_add_f32 v5, v19, v15 // (op 19583) 
v_add_f32 v6, v20, v16 // (op 19583) 
v_add_f32 v7, v21, v17 // (op 19583) 


v_lshrrev_b32 v14, 16, v4 // (op 19584) 
v_lshrrev_b32 v15, 16, v5 // (op 19584) 
v_lshrrev_b32 v16, 16, v6 // (op 19584) 
v_lshrrev_b32 v17, 16, v7 // (op 19584) 


v_pack_b32_f16 v4, v14, v15 // (op 19585) 
v_pack_b32_f16 v5, v16, v17 // (op 19585) 

s_lshl_b32 s0, s4, 5 // (op 19585) 
v_add_u32 v0, v13, s0 // (op 19585) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 19585) Store value





s_lshl_b32 s0, s2, 6 // (op 19600) 
v_add_u32 v0, v12, s0 // (op 19600) 


v_accvgpr_read v4, a8 // (op 19596) 
v_accvgpr_read v5, a9 // (op 19596) 
v_accvgpr_read v6, a10 // (op 19596) 
v_accvgpr_read v7, a11 // (op 19596) 
v_mul_f32 v14, v11, v4 // (op 19596) 
v_mul_f32 v15, v11, v5 // (op 19596) 
v_mul_f32 v16, v11, v6 // (op 19596) 
v_mul_f32 v17, v11, v7 // (op 19596) 

buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 19600) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 19601) 
v_lshrrev_b32 v7, 16, v4 // (op 19601) 
v_lshlrev_b32 v6, 16, v6 // (op 19601) 
v_lshlrev_b32 v7, 16, v7 // (op 19601) 
v_mul_f32 v18, v9, v6 // (op 19601) 
v_mul_f32 v19, v9, v7 // (op 19601) 
v_and_b32 v6, 65535, v5 // (op 19601) 
v_lshrrev_b32 v7, 16, v5 // (op 19601) 
v_lshlrev_b32 v6, 16, v6 // (op 19601) 
v_lshlrev_b32 v7, 16, v7 // (op 19601) 
v_mul_f32 v20, v9, v6 // (op 19601) 
v_mul_f32 v21, v9, v7 // (op 19601) 


v_add_f32 v4, v18, v14 // (op 19597) 
v_add_f32 v5, v19, v15 // (op 19597) 
v_add_f32 v6, v20, v16 // (op 19597) 
v_add_f32 v7, v21, v17 // (op 19597) 


v_lshrrev_b32 v14, 16, v4 // (op 19598) 
v_lshrrev_b32 v15, 16, v5 // (op 19598) 
v_lshrrev_b32 v16, 16, v6 // (op 19598) 
v_lshrrev_b32 v17, 16, v7 // (op 19598) 


v_pack_b32_f16 v4, v14, v15 // (op 19599) 
v_pack_b32_f16 v5, v16, v17 // (op 19599) 

s_lshl_b32 s0, s4, 6 // (op 19599) 
v_add_u32 v0, v13, s0 // (op 19599) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 19599) Store value





s_lshl_b32 s0, s2, 5 // (op 19614) 
s_mul_i32 s1, 3, s0 // (op 19614) 

v_add_u32 v0, v12, s1 // (op 19614) 


v_accvgpr_read v4, a12 // (op 19610) 
v_accvgpr_read v5, a13 // (op 19610) 
v_accvgpr_read v6, a14 // (op 19610) 
v_accvgpr_read v7, a15 // (op 19610) 
v_mul_f32 v14, v11, v4 // (op 19610) 
v_mul_f32 v15, v11, v5 // (op 19610) 
v_mul_f32 v16, v11, v6 // (op 19610) 
v_mul_f32 v17, v11, v7 // (op 19610) 

buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 19614) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 19615) 
v_lshrrev_b32 v7, 16, v4 // (op 19615) 
v_lshlrev_b32 v6, 16, v6 // (op 19615) 
v_lshlrev_b32 v7, 16, v7 // (op 19615) 
v_mul_f32 v18, v9, v6 // (op 19615) 
v_mul_f32 v19, v9, v7 // (op 19615) 
v_and_b32 v6, 65535, v5 // (op 19615) 
v_lshrrev_b32 v7, 16, v5 // (op 19615) 
v_lshlrev_b32 v6, 16, v6 // (op 19615) 
v_lshlrev_b32 v7, 16, v7 // (op 19615) 
v_mul_f32 v20, v9, v6 // (op 19615) 
v_mul_f32 v21, v9, v7 // (op 19615) 


v_add_f32 v4, v18, v14 // (op 19611) 
v_add_f32 v5, v19, v15 // (op 19611) 
v_add_f32 v6, v20, v16 // (op 19611) 
v_add_f32 v7, v21, v17 // (op 19611) 


v_lshrrev_b32 v14, 16, v4 // (op 19612) 
v_lshrrev_b32 v15, 16, v5 // (op 19612) 
v_lshrrev_b32 v16, 16, v6 // (op 19612) 
v_lshrrev_b32 v17, 16, v7 // (op 19612) 


v_pack_b32_f16 v4, v14, v15 // (op 19613) 
v_pack_b32_f16 v5, v16, v17 // (op 19613) 

s_lshl_b32 s0, s4, 5 // (op 19613) 
s_mul_i32 s1, 3, s0 // (op 19613) 

v_add_u32 v0, v13, s1 // (op 19613) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 19613) Store value





s_lshl_b32 s0, s2, 7 // (op 19628) 
v_add_u32 v0, v12, s0 // (op 19628) 


v_accvgpr_read v4, a16 // (op 19624) 
v_accvgpr_read v5, a17 // (op 19624) 
v_accvgpr_read v6, a18 // (op 19624) 
v_accvgpr_read v7, a19 // (op 19624) 
v_mul_f32 v14, v11, v4 // (op 19624) 
v_mul_f32 v15, v11, v5 // (op 19624) 
v_mul_f32 v16, v11, v6 // (op 19624) 
v_mul_f32 v17, v11, v7 // (op 19624) 

buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 19628) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 19629) 
v_lshrrev_b32 v7, 16, v4 // (op 19629) 
v_lshlrev_b32 v6, 16, v6 // (op 19629) 
v_lshlrev_b32 v7, 16, v7 // (op 19629) 
v_mul_f32 v18, v9, v6 // (op 19629) 
v_mul_f32 v19, v9, v7 // (op 19629) 
v_and_b32 v6, 65535, v5 // (op 19629) 
v_lshrrev_b32 v7, 16, v5 // (op 19629) 
v_lshlrev_b32 v6, 16, v6 // (op 19629) 
v_lshlrev_b32 v7, 16, v7 // (op 19629) 
v_mul_f32 v20, v9, v6 // (op 19629) 
v_mul_f32 v21, v9, v7 // (op 19629) 


v_add_f32 v4, v18, v14 // (op 19625) 
v_add_f32 v5, v19, v15 // (op 19625) 
v_add_f32 v6, v20, v16 // (op 19625) 
v_add_f32 v7, v21, v17 // (op 19625) 


v_lshrrev_b32 v14, 16, v4 // (op 19626) 
v_lshrrev_b32 v15, 16, v5 // (op 19626) 
v_lshrrev_b32 v16, 16, v6 // (op 19626) 
v_lshrrev_b32 v17, 16, v7 // (op 19626) 


v_pack_b32_f16 v4, v14, v15 // (op 19627) 
v_pack_b32_f16 v5, v16, v17 // (op 19627) 

s_lshl_b32 s0, s4, 7 // (op 19627) 
v_add_u32 v0, v13, s0 // (op 19627) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 19627) Store value





s_lshl_b32 s0, s2, 5 // (op 19642) 
s_mul_i32 s1, 5, s0 // (op 19642) 

v_add_u32 v0, v12, s1 // (op 19642) 


v_accvgpr_read v4, a20 // (op 19638) 
v_accvgpr_read v5, a21 // (op 19638) 
v_accvgpr_read v6, a22 // (op 19638) 
v_accvgpr_read v7, a23 // (op 19638) 
v_mul_f32 v14, v11, v4 // (op 19638) 
v_mul_f32 v15, v11, v5 // (op 19638) 
v_mul_f32 v16, v11, v6 // (op 19638) 
v_mul_f32 v17, v11, v7 // (op 19638) 

buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 19642) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 19643) 
v_lshrrev_b32 v7, 16, v4 // (op 19643) 
v_lshlrev_b32 v6, 16, v6 // (op 19643) 
v_lshlrev_b32 v7, 16, v7 // (op 19643) 
v_mul_f32 v18, v9, v6 // (op 19643) 
v_mul_f32 v19, v9, v7 // (op 19643) 
v_and_b32 v6, 65535, v5 // (op 19643) 
v_lshrrev_b32 v7, 16, v5 // (op 19643) 
v_lshlrev_b32 v6, 16, v6 // (op 19643) 
v_lshlrev_b32 v7, 16, v7 // (op 19643) 
v_mul_f32 v20, v9, v6 // (op 19643) 
v_mul_f32 v21, v9, v7 // (op 19643) 


v_add_f32 v4, v18, v14 // (op 19639) 
v_add_f32 v5, v19, v15 // (op 19639) 
v_add_f32 v6, v20, v16 // (op 19639) 
v_add_f32 v7, v21, v17 // (op 19639) 


v_lshrrev_b32 v14, 16, v4 // (op 19640) 
v_lshrrev_b32 v15, 16, v5 // (op 19640) 
v_lshrrev_b32 v16, 16, v6 // (op 19640) 
v_lshrrev_b32 v17, 16, v7 // (op 19640) 


v_pack_b32_f16 v4, v14, v15 // (op 19641) 
v_pack_b32_f16 v5, v16, v17 // (op 19641) 

s_lshl_b32 s0, s4, 5 // (op 19641) 
s_mul_i32 s1, 5, s0 // (op 19641) 

v_add_u32 v0, v13, s1 // (op 19641) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 19641) Store value





s_lshl_b32 s0, s2, 5 // (op 19656) 
s_mul_i32 s1, 6, s0 // (op 19656) 

v_add_u32 v0, v12, s1 // (op 19656) 


v_accvgpr_read v4, a24 // (op 19652) 
v_accvgpr_read v5, a25 // (op 19652) 
v_accvgpr_read v6, a26 // (op 19652) 
v_accvgpr_read v7, a27 // (op 19652) 
v_mul_f32 v14, v11, v4 // (op 19652) 
v_mul_f32 v15, v11, v5 // (op 19652) 
v_mul_f32 v16, v11, v6 // (op 19652) 
v_mul_f32 v17, v11, v7 // (op 19652) 

buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 19656) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 19657) 
v_lshrrev_b32 v7, 16, v4 // (op 19657) 
v_lshlrev_b32 v6, 16, v6 // (op 19657) 
v_lshlrev_b32 v7, 16, v7 // (op 19657) 
v_mul_f32 v18, v9, v6 // (op 19657) 
v_mul_f32 v19, v9, v7 // (op 19657) 
v_and_b32 v6, 65535, v5 // (op 19657) 
v_lshrrev_b32 v7, 16, v5 // (op 19657) 
v_lshlrev_b32 v6, 16, v6 // (op 19657) 
v_lshlrev_b32 v7, 16, v7 // (op 19657) 
v_mul_f32 v20, v9, v6 // (op 19657) 
v_mul_f32 v21, v9, v7 // (op 19657) 


v_add_f32 v4, v18, v14 // (op 19653) 
v_add_f32 v5, v19, v15 // (op 19653) 
v_add_f32 v6, v20, v16 // (op 19653) 
v_add_f32 v7, v21, v17 // (op 19653) 


v_lshrrev_b32 v14, 16, v4 // (op 19654) 
v_lshrrev_b32 v15, 16, v5 // (op 19654) 
v_lshrrev_b32 v16, 16, v6 // (op 19654) 
v_lshrrev_b32 v17, 16, v7 // (op 19654) 


v_pack_b32_f16 v4, v14, v15 // (op 19655) 
v_pack_b32_f16 v5, v16, v17 // (op 19655) 

s_lshl_b32 s0, s4, 5 // (op 19655) 
s_mul_i32 s1, 6, s0 // (op 19655) 

v_add_u32 v0, v13, s1 // (op 19655) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 19655) Store value





s_lshl_b32 s0, s2, 5 // (op 19670) 
s_mul_i32 s1, 7, s0 // (op 19670) 

v_add_u32 v0, v12, s1 // (op 19670) 


v_accvgpr_read v4, a28 // (op 19666) 
v_accvgpr_read v5, a29 // (op 19666) 
v_accvgpr_read v6, a30 // (op 19666) 
v_accvgpr_read v7, a31 // (op 19666) 
v_mul_f32 v14, v11, v4 // (op 19666) 
v_mul_f32 v15, v11, v5 // (op 19666) 
v_mul_f32 v16, v11, v6 // (op 19666) 
v_mul_f32 v17, v11, v7 // (op 19666) 

buffer_load_dwordx2 v[4:5], v0, s[52:55], 0 offen // (op 19670) Load value

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 19671) 
v_lshrrev_b32 v7, 16, v4 // (op 19671) 
v_lshlrev_b32 v6, 16, v6 // (op 19671) 
v_lshlrev_b32 v7, 16, v7 // (op 19671) 
v_mul_f32 v18, v9, v6 // (op 19671) 
v_mul_f32 v19, v9, v7 // (op 19671) 
v_and_b32 v6, 65535, v5 // (op 19671) 
v_lshrrev_b32 v7, 16, v5 // (op 19671) 
v_lshlrev_b32 v6, 16, v6 // (op 19671) 
v_lshlrev_b32 v7, 16, v7 // (op 19671) 
v_mul_f32 v20, v9, v6 // (op 19671) 
v_mul_f32 v21, v9, v7 // (op 19671) 


v_add_f32 v4, v18, v14 // (op 19667) 
v_add_f32 v5, v19, v15 // (op 19667) 
v_add_f32 v6, v20, v16 // (op 19667) 
v_add_f32 v7, v21, v17 // (op 19667) 


v_lshrrev_b32 v14, 16, v4 // (op 19668) 
v_lshrrev_b32 v15, 16, v5 // (op 19668) 
v_lshrrev_b32 v16, 16, v6 // (op 19668) 
v_lshrrev_b32 v17, 16, v7 // (op 19668) 


v_pack_b32_f16 v4, v14, v15 // (op 19669) 
v_pack_b32_f16 v5, v16, v17 // (op 19669) 

s_lshl_b32 s0, s4, 5 // (op 19669) 
s_mul_i32 s1, 7, s0 // (op 19669) 

v_add_u32 v0, v13, s1 // (op 19669) 


buffer_store_dwordx2 v[4:5], v0, s[56:59], 0 offen // (op 19669) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WGM__ConditionalBottom_0_beta_116:








































































s_endpgm  // End of RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WGM_
.LRR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WGM__end:
.size RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WGM_, .LRR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WGM__end-RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WGM_
  .amdhsa_next_free_vgpr 436
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 131072 // lds bytes
  .amdhsa_accum_offset 180
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
    DataType: F4
    DataTypeA: F4
    DataTypeB: F4
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
    macrotile: [256, 256, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [8, 8]
  StaggerU: 0
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WGM_
    .symbol: RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x256x128_UR_4_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 131072
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 178
    .agpr_count: 256
    .max_flat_workgroup_size: 256
    .workgroup_size: [256, 1, 1]
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {is-null: true}, {is-null: true}]
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
        .expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}
        .variableType: {dataType: Int32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicMultiple_26
        .size: 4
        .offset: 28
        .expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}}
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
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: FP4, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP4, pointerType: PointerGlobal}
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
        .expression: {type: MagicShiftAndSign, arg: {type: Subtract, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_24, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_22, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 152, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_23, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}, offset: 156, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 160, size: 4}, rhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}}}}
        .variableType: {dataType: UInt32, pointerType: Value}
        .value_kind: by_value
      - .name: MagicShiftAndSign_27
        .size: 4
        .offset: 156
        .expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: Subtract_21, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: Subtract, lhs: {type: BitwiseXor, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: BitwiseAnd, lhs: {type: ArithmeticShiftR, lhs: {type: Add, lhs: {type: MultiplyHigh, lhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}, rhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}}, rhs: {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_1, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 8, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}, rhs: {type: Add, lhs: {type: ShiftL, lhs: {type: LiteralValue.Int32, dataType: Int32, value: 1}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: Conditional, lhs: {type: Equal, lhs: {type: Kernel Argument, name: MagicMultiple_19, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicMultiple, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 140, size: 4}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}, r1hs: {type: LiteralValue.Int32, dataType: Int32, value: -1}, r2hs: {type: LiteralValue.Int32, dataType: Int32, value: 0}}}}}, rhs: {type: BitwiseAnd, lhs: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, rhs: {type: ArithmeticShiftR, lhs: {type: ShiftL, lhs: {type: Convert, arg: {type: Kernel Argument, name: MagicShiftAndSign_20, variableType: {dataType: UInt32, pointerType: Value}, dataDirection: read_only, expression: {type: MagicShiftAndSign, arg: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}, offset: 144, size: 4}, dataType: Int32}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 24}}, rhs: {type: LiteralValue.Int32, dataType: Int32, value: 31}}}, offset: 148, size: 4}, rhs: {type: Multiply, lhs: {type: Kernel Argument, name: WGM_18, variableType: {dataType: Int32, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 4, offset: 248, name: WGM, variableType: {dataType: Int32, pointerType: Value}, direction: read_only}, offset: 136, size: 4}, rhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_4, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 32, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 8}}, dataType: Int32}}}}
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

