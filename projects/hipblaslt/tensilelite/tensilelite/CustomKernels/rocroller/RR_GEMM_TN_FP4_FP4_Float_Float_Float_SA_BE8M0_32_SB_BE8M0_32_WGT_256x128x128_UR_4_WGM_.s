// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_WGM_
.p2align 8
.type RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_WGM_,@function
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 20642) 
s_mul_hi_i32 s1, s16, s10 // (op 20781) 
v_and_b32 v0, 4294967168, v1 // (op 20703) 
v_and_b32 v3, 15, v1 // (op 20703) 
v_add_lshl_u32 v4, v0, v3, 7 // (op 20703) 


v_and_b32 v0, 63, v1 // (op 20703) 
v_lshrrev_b32 v3, 4, v0 // (op 20703) 

v_lshlrev_b32 v0, 5, v3 // (op 20703) 

v_add_lshl_u32 v3, v4, v0, 2 // (op 20703) 


v_lshrrev_b32 v0, 3, v3 // (op 20703) 

v_lshrrev_b32 v3, 6, v1 // (op 20760) 
v_and_b32 v4, 1, v3 // (op 20760) 

v_and_b32 v3, 15, v1 // (op 20760) 
v_lshl_add_u32 v5, v4, 6, v3 // (op 20760) 


v_and_b32 v3, 63, v1 // (op 20760) 
v_lshrrev_b32 v4, 4, v3 // (op 20760) 

v_lshlrev_b32 v3, 5, v4 // (op 20760) 

v_lshl_add_u32 v4, v5, 7, v3 // (op 20760) 


v_lshlrev_b32 v3, 2, v4 // (op 20760) 

v_lshrrev_b32 v4, 3, v3 // (op 20760) 

v_and_b32 v3, 4294967168, v1 // (op 20827) 
v_and_b32 v5, 15, v1 // (op 20827) 
v_add_lshl_u32 v6, v3, v5, 7 // (op 20827) 


v_and_b32 v3, 63, v1 // (op 20827) 
v_lshrrev_b32 v5, 4, v3 // (op 20827) 

v_lshlrev_b32 v3, 5, v5 // (op 20827) 

v_add_lshl_u32 v5, v6, v3, 2 // (op 20827) 


v_lshrrev_b32 v3, 3, v5 // (op 20827) 

v_lshrrev_b32 v5, 6, v1 // (op 20871) 
v_and_b32 v6, 1, v5 // (op 20871) 

v_and_b32 v5, 15, v1 // (op 20871) 
v_lshl_add_u32 v7, v6, 6, v5 // (op 20871) 


v_and_b32 v5, 63, v1 // (op 20871) 
v_lshrrev_b32 v6, 4, v5 // (op 20871) 

v_lshlrev_b32 v5, 5, v6 // (op 20871) 

v_lshl_add_u32 v6, v7, 7, v5 // (op 20871) 


v_lshlrev_b32 v5, 2, v6 // (op 20871) 

v_lshrrev_b32 v6, 3, v5 // (op 20871) 

v_and_b32 v5, 4294967168, v1 // (op 20915) 
v_and_b32 v7, 15, v1 // (op 20915) 
v_add_lshl_u32 v8, v5, v7, 7 // (op 20915) 


v_and_b32 v5, 63, v1 // (op 20915) 
v_lshrrev_b32 v7, 4, v5 // (op 20915) 

v_lshlrev_b32 v5, 5, v7 // (op 20915) 

v_add_lshl_u32 v7, v8, v5, 2 // (op 20915) 


v_lshrrev_b32 v5, 3, v7 // (op 20915) 

v_lshrrev_b32 v7, 6, v1 // (op 20959) 
v_and_b32 v8, 1, v7 // (op 20959) 

v_and_b32 v7, 15, v1 // (op 20959) 
v_lshl_add_u32 v9, v8, 6, v7 // (op 20959) 


v_and_b32 v7, 63, v1 // (op 20959) 
v_lshrrev_b32 v8, 4, v7 // (op 20959) 

v_lshlrev_b32 v7, 5, v8 // (op 20959) 

v_lshl_add_u32 v8, v9, 7, v7 // (op 20959) 


v_lshlrev_b32 v7, 2, v8 // (op 20959) 

v_lshrrev_b32 v8, 3, v7 // (op 20959) 

v_and_b32 v7, 4294967168, v1 // (op 21003) 
v_and_b32 v9, 15, v1 // (op 21003) 
v_add_lshl_u32 v10, v7, v9, 7 // (op 21003) 


v_and_b32 v7, 63, v1 // (op 21003) 
v_lshrrev_b32 v9, 4, v7 // (op 21003) 

v_lshlrev_b32 v7, 5, v9 // (op 21003) 

v_add_lshl_u32 v9, v10, v7, 2 // (op 21003) 


v_lshrrev_b32 v7, 3, v9 // (op 21003) 

v_lshrrev_b32 v9, 6, v1 // (op 21047) 
v_and_b32 v10, 1, v9 // (op 21047) 

v_and_b32 v9, 15, v1 // (op 21047) 
v_lshl_add_u32 v11, v10, 6, v9 // (op 21047) 


v_and_b32 v9, 63, v1 // (op 21047) 
v_lshrrev_b32 v10, 4, v9 // (op 21047) 

v_lshlrev_b32 v9, 5, v10 // (op 21047) 

v_lshl_add_u32 v10, v11, 7, v9 // (op 21047) 


v_lshlrev_b32 v9, 2, v10 // (op 21047) 

v_lshrrev_b32 v10, 3, v9 // (op 21047) 

v_and_b32 v9, 4294967168, v1 // (op 20642) 
s_add_i32 s18, s0, s16 // (op 20642) 

v_and_b32 v11, 63, v1 // (op 20642) 
s_ashr_i32 s0, s18, 31 // (op 20642) 
v_lshrrev_b32 v12, 4, v11 // (op 20642) 
s_cmp_eq_i32 s10, 0 // (op 20642) 
s_mov_b32 s19, scc // (op 20642) 
v_lshlrev_b32 v13, 2, v12 // (op 20642) 

s_cmp_eq_i32 s19, 0 // (op 20642) 
s_cselect_b32 s49, 0, -1 // (op 20642) 

v_add_u32 v12, v9, v13 // (op 20642) 


s_and_b32 s19, s13, 31 // (op 20642) 
v_lshrrev_b32 v9, 2, v11 // (op 20642) 

v_and_b32 v11, 3, v9 // (op 20642) 

v_and_b32 v9, 4294967168, v1 // (op 20781) 
v_and_b32 v13, 63, v1 // (op 20781) 
s_lshl_b32 s50, 1, s19 // (op 20642) 
v_and_b32 v14, 3, v1 // (op 20642) 
s_add_i32 s51, s50, s49 // (op 20642) 


s_and_b32 s49, s0, s51 // (op 20642) 


s_add_i32 s0, s1, s16 // (op 20781) 

s_ashr_i32 s1, s0, 31 // (op 20781) 
v_lshl_add_u32 v15, v11, 2, v14 // (op 20642) 


s_add_i32 s50, s18, s49 // (op 20642) 


v_lshrrev_b32 v11, 6, v1 // (op 20642) 
s_ashr_i32 s18, s50, s19 // (op 20642) 


v_and_b32 v14, 1, v11 // (op 20642) 

s_lshl_b32 s19, s13, 24 // (op 20642) 
v_lshl_add_u32 v11, v14, 6, v15 // (op 20642) 


s_ashr_i32 s49, s19, 31 // (op 20642) 

s_xor_b32 s19, s18, s49 // (op 20642) 

s_sub_i32 s18, s19, s49 // (op 20642) 


s_mul_hi_i32 s19, s18, s14 // (op 20642) 
s_add_i32 s49, s19, s18 // (op 20642) 

s_ashr_i32 s19, s49, 31 // (op 20642) 
s_cmp_eq_i32 s14, 0 // (op 20642) 
s_mov_b32 s50, scc // (op 20642) 
s_cmp_eq_i32 s50, 0 // (op 20642) 
s_cselect_b32 s51, 0, -1 // (op 20642) 

s_and_b32 s50, s15, 31 // (op 20642) 
v_lshrrev_b32 v14, 4, v13 // (op 20781) 
s_cmp_eq_i32 s10, 0 // (op 20781) 
s_mov_b32 s52, scc // (op 20781) 
v_lshlrev_b32 v15, 2, v14 // (op 20781) 

s_cmp_eq_i32 s52, 0 // (op 20781) 
s_cselect_b32 s53, 0, -1 // (op 20781) 

s_lshl_b32 s52, 1, s50 // (op 20642) 
v_add_u32 v14, v9, v15 // (op 20781) 


v_lshrrev_b32 v9, 2, v13 // (op 20781) 

s_add_i32 s54, s52, s51 // (op 20642) 


s_and_b32 s51, s19, s54 // (op 20642) 


s_add_i32 s19, s49, s51 // (op 20642) 


s_ashr_i32 s49, s19, s50 // (op 20642) 


s_lshl_b32 s19, s15, 24 // (op 20642) 
s_ashr_i32 s50, s19, 31 // (op 20642) 

s_xor_b32 s19, s49, s50 // (op 20642) 

s_sub_i32 s49, s19, s50 // (op 20642) 


s_mul_i32 s19, s18, s11 // (op 20642) 
s_sub_u32 s50, s16, s19 // (op 20642) 

s_mul_i32 s19, s49, s11 // (op 20642) 
s_add_u32 s19, s19, s50 // (op 20642) 

s_mul_hi_i32 s50, s16, s9 // (op 20642) 
s_add_i32 s51, s50, s16 // (op 20642) 

s_ashr_i32 s50, s51, 31 // (op 20642) 
s_cmp_eq_i32 s9, 0 // (op 20642) 
s_mov_b32 s52, scc // (op 20642) 
s_cmp_eq_i32 s52, 0 // (op 20642) 
s_cselect_b32 s54, 0, -1 // (op 20642) 

s_and_b32 s52, s13, 31 // (op 20781) 
v_and_b32 v13, 3, v9 // (op 20781) 

s_lshl_b32 s55, 1, s52 // (op 20781) 
v_and_b32 v9, 3, v1 // (op 20781) 
s_add_i32 s56, s55, s53 // (op 20781) 


s_and_b32 s53, s1, s56 // (op 20781) 


v_lshl_add_u32 v15, v13, 2, v9 // (op 20781) 


s_add_i32 s1, s0, s53 // (op 20781) 


v_lshrrev_b32 v9, 6, v1 // (op 20781) 
s_ashr_i32 s0, s1, s52 // (op 20781) 


v_and_b32 v13, 1, v9 // (op 20781) 

s_lshl_b32 s1, s13, 24 // (op 20781) 
v_lshl_add_u32 v9, v13, 6, v15 // (op 20781) 


s_ashr_i32 s52, s1, 31 // (op 20781) 

s_xor_b32 s1, s0, s52 // (op 20781) 

s_sub_i32 s0, s1, s52 // (op 20781) 


s_mul_hi_i32 s1, s0, s14 // (op 20781) 
s_add_i32 s52, s1, s0 // (op 20781) 

s_ashr_i32 s1, s52, 31 // (op 20781) 
s_cmp_eq_i32 s14, 0 // (op 20781) 
s_mov_b32 s53, scc // (op 20781) 
s_cmp_eq_i32 s53, 0 // (op 20781) 
s_cselect_b32 s55, 0, -1 // (op 20781) 

s_and_b32 s53, s15, 31 // (op 20781) 
s_lshl_b32 s56, 1, s53 // (op 20781) 
s_add_i32 s57, s56, s55 // (op 20781) 


s_and_b32 s55, s1, s57 // (op 20781) 


s_add_i32 s1, s52, s55 // (op 20781) 


s_ashr_i32 s52, s1, s53 // (op 20781) 


s_lshl_b32 s1, s15, 24 // (op 20781) 
s_ashr_i32 s53, s1, 31 // (op 20781) 

s_xor_b32 s1, s52, s53 // (op 20781) 

s_sub_i32 s52, s1, s53 // (op 20781) 


s_mul_i32 s1, s0, s11 // (op 20781) 
s_sub_u32 s53, s16, s1 // (op 20781) 

s_mul_i32 s1, s52, s11 // (op 20781) 
s_add_u32 s1, s1, s53 // (op 20781) 

s_mul_hi_i32 s53, s16, s9 // (op 20781) 
s_add_i32 s55, s53, s16 // (op 20781) 

s_ashr_i32 s53, s55, 31 // (op 20781) 
s_cmp_eq_i32 s9, 0 // (op 20781) 
s_mov_b32 s56, scc // (op 20781) 
s_cmp_eq_i32 s56, 0 // (op 20781) 
s_cselect_b32 s57, 0, -1 // (op 20781) 


s_waitcnt lgkmcnt(0)
s_and_b32 s56, s45, 31 // (op 20642) 
s_lshl_b32 s58, 1, s56 // (op 20642) 
s_add_i32 s59, s58, s54 // (op 20642) 


s_and_b32 s54, s50, s59 // (op 20642) 


s_add_i32 s50, s51, s54 // (op 20642) 


s_ashr_i32 s51, s50, s56 // (op 20642) 


s_lshl_b32 s50, s45, 24 // (op 20642) 
s_ashr_i32 s54, s50, 31 // (op 20642) 

s_xor_b32 s50, s51, s54 // (op 20642) 

s_sub_i32 s51, s50, s54 // (op 20642) 


s_and_b32 s50, s45, 31 // (op 20781) 
s_lshl_b32 s54, 1, s50 // (op 20781) 
s_add_i32 s56, s54, s57 // (op 20781) 


s_and_b32 s54, s53, s56 // (op 20781) 


s_add_i32 s53, s55, s54 // (op 20781) 


s_ashr_i32 s54, s53, s50 // (op 20781) 


s_lshl_b32 s50, s45, 24 // (op 20781) 
s_ashr_i32 s53, s50, 31 // (op 20781) 

s_xor_b32 s50, s54, s53 // (op 20781) 

s_sub_i32 s54, s50, s53 // (op 20781) 


s_add_u32 s56, s6, 128 // (op 20642) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 20642) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 20642) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 20642) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 7 // (op 20642) 

s_mul_i32 s50, s11, s56 // (op 20642) 
s_mul_i32 s53, s8, s50 // (op 20642) 

s_mul_i32 s50, s51, s53 // (op 20642) 

s_sub_u32 s53, s16, s50 // (op 20642) 

s_mov_b32 s50, s53 // (op 20642) 
s_mul_hi_i32 s55, s50, s12 // (op 20642) 
s_add_i32 s58, s55, s50 // (op 20642) 


s_ashr_i32 s50, s58, 31 // (op 20642) 
s_cmp_eq_i32 s12, 0 // (op 20642) 
s_mov_b32 s55, scc // (op 20642) 
s_cmp_eq_i32 s55, 0 // (op 20642) 
s_cselect_b32 s59, 0, -1 // (op 20642) 

s_and_b32 s55, s44, 31 // (op 20642) 
s_lshl_b32 s60, 1, s55 // (op 20642) 
s_add_i32 s61, s60, s59 // (op 20642) 


s_and_b32 s59, s50, s61 // (op 20642) 


s_add_i32 s50, s58, s59 // (op 20642) 


s_ashr_i32 s58, s50, s55 // (op 20642) 


s_lshl_b32 s50, s44, 24 // (op 20642) 
s_ashr_i32 s55, s50, 31 // (op 20642) 

s_xor_b32 s50, s58, s55 // (op 20642) 

s_sub_i32 s58, s50, s55 // (op 20642) 


s_mul_i32 s50, s46, s11 // (op 20642) 
s_add_u32 s60, s20, 256 // (op 20642) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 20642) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 20642) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 20642) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 8 // (op 20642) 

s_sub_i32 s55, s60, s50 // (op 20642) 

s_mul_i32 s59, s58, s55 // (op 20642) 

s_sub_u32 s55, s53, s59 // (op 20642) 


s_add_u32 s53, s55, s50 // (op 20642) 


s_cmp_eq_i32 s51, 0 // (op 20642) 
s_mov_b32 s50, scc // (op 20642) 

s_cmp_eq_i32 s50, 0 // (op 20642) 
s_cselect_b32 s51, s53, s19 // (op 20642) 


s_mul_i32 s19, s49, s56 // (op 20642) 


v_lshl_add_u32 v13, s51, 8, v12 // (op 20642) 


s_sub_i32 s49, s18, s19 // (op 20642) 


s_cmp_eq_i32 s50, 0 // (op 20642) 
s_cselect_b32 s51, s58, s49 // (op 20642) 



v_lshl_add_u32 v12, s51, 7, v11 // (op 20642) 


v_mul_lo_u32 v11, v12, s2 // (op 20642) 

v_add_lshl_u32 v12, v13, v11, 2 // (op 20642) 


s_add_u32 s18, s6, 128 // (op 20781) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 20781) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 20781) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 20781) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 7 // (op 20781) 

s_mul_i32 s53, s11, s18 // (op 20781) 
s_mul_i32 s49, s8, s53 // (op 20781) 

s_mul_i32 s53, s54, s49 // (op 20781) 

s_sub_u32 s49, s16, s53 // (op 20781) 

s_mov_b32 s53, s49 // (op 20781) 
s_mul_hi_i32 s50, s53, s12 // (op 20781) 
s_add_i32 s51, s50, s53 // (op 20781) 


s_ashr_i32 s50, s51, 31 // (op 20781) 
s_cmp_eq_i32 s12, 0 // (op 20781) 
s_mov_b32 s53, scc // (op 20781) 
s_cmp_eq_i32 s53, 0 // (op 20781) 
s_cselect_b32 s55, 0, -1 // (op 20781) 

s_and_b32 s53, s44, 31 // (op 20781) 
s_lshl_b32 s56, 1, s53 // (op 20781) 
s_add_i32 s57, s56, s55 // (op 20781) 


s_and_b32 s55, s50, s57 // (op 20781) 


s_add_i32 s50, s51, s55 // (op 20781) 


s_ashr_i32 s51, s50, s53 // (op 20781) 


s_lshl_b32 s50, s44, 24 // (op 20781) 
s_ashr_i32 s53, s50, 31 // (op 20781) 

s_xor_b32 s50, s51, s53 // (op 20781) 

s_sub_i32 s51, s50, s53 // (op 20781) 


s_mul_i32 s50, s46, s11 // (op 20781) 
s_add_u32 s56, s20, 256 // (op 20781) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 20781) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 20781) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 20781) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 8 // (op 20781) 

s_sub_i32 s53, s56, s50 // (op 20781) 

s_mul_i32 s55, s51, s53 // (op 20781) 

s_sub_u32 s53, s49, s55 // (op 20781) 


s_add_u32 s49, s53, s50 // (op 20781) 


s_cmp_eq_i32 s54, 0 // (op 20781) 
s_mov_b32 s50, scc // (op 20781) 

s_cmp_eq_i32 s50, 0 // (op 20781) 
s_cselect_b32 s53, s49, s1 // (op 20781) 


s_mul_i32 s1, s52, s18 // (op 20781) 


v_lshl_add_u32 v11, s53, 8, v14 // (op 20781) 


s_sub_i32 s49, s0, s1 // (op 20781) 


s_cmp_eq_i32 s50, 0 // (op 20781) 
s_cselect_b32 s0, s51, s49 // (op 20781) 



v_lshl_add_u32 v13, s0, 7, v9 // (op 20781) 


v_mul_lo_u32 v9, v13, s4 // (op 20781) 

v_add_lshl_u32 v13, v11, v9, 2 // (op 20781) 


s_mul_i32 s1, s2, s7 // (op 20644) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 20644) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 20644) least significant
s_mul_hi_u32 s19, s2, s6 // (op 20644) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 20644) most significant: sum
s_add_u32 s1, s1, s19 // (op 20644) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 20644) 

s_mul_i32 s1, s4, s7 // (op 20783) most significant: low of low * high
s_mul_i32 s49, s5, s6 // (op 20783) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 20783) least significant
s_mul_hi_u32 s50, s4, s6 // (op 20783) most significant: high of low * low
s_add_u32 s1, s1, s49 // (op 20783) most significant: sum
s_add_u32 s1, s1, s50 // (op 20783) most significant: sum


s_lshl_b64 s[50:51], s[0:1], 2 // (op 20783) 

s_mov_b32 s52, s22 // (op 20644) reg expression
s_mov_b32 s53, s23 // (op 20644) reg expression
s_mov_b32 s54, s18 // (op 20644) reg expression
s_mov_b32 s55, 131072 // (op 20644) call()

s_mov_b32 s56, s24 // (op 20783) reg expression

s_mov_b32 s57, s25 // (op 20783) reg expression
s_mov_b32 s58, s50 // (op 20783) reg expression
s_mov_b32 s59, 131072 // (op 20783) call()


v_mov_b32 v9, s47 // (op 2) Move value
v_mov_b32 v11, s48 // (op 35) Move value


v_accvgpr_write a0, 0.00000 // (op 55) call()
v_accvgpr_write a1, 0.00000 // (op 55) call()
v_accvgpr_write a2, 0.00000 // (op 55) call()
v_accvgpr_write a3, 0.00000 // (op 55) call()
v_accvgpr_write a4, 0.00000 // (op 383) call()
v_accvgpr_write a5, 0.00000 // (op 383) call()
v_accvgpr_write a6, 0.00000 // (op 383) call()
v_accvgpr_write a7, 0.00000 // (op 383) call()
v_accvgpr_write a8, 0.00000 // (op 609) call()
v_accvgpr_write a9, 0.00000 // (op 609) call()
v_accvgpr_write a10, 0.00000 // (op 609) call()
v_accvgpr_write a11, 0.00000 // (op 609) call()
v_accvgpr_write a12, 0.00000 // (op 835) call()
v_accvgpr_write a13, 0.00000 // (op 835) call()
v_accvgpr_write a14, 0.00000 // (op 835) call()
v_accvgpr_write a15, 0.00000 // (op 835) call()
v_accvgpr_write a16, 0.00000 // (op 1197) call()
v_accvgpr_write a17, 0.00000 // (op 1197) call()
v_accvgpr_write a18, 0.00000 // (op 1197) call()
v_accvgpr_write a19, 0.00000 // (op 1197) call()
v_accvgpr_write a20, 0.00000 // (op 1610) call()
v_accvgpr_write a21, 0.00000 // (op 1610) call()
v_accvgpr_write a22, 0.00000 // (op 1610) call()
v_accvgpr_write a23, 0.00000 // (op 1610) call()
v_accvgpr_write a24, 0.00000 // (op 1611) call()
v_accvgpr_write a25, 0.00000 // (op 1611) call()
v_accvgpr_write a26, 0.00000 // (op 1611) call()
v_accvgpr_write a27, 0.00000 // (op 1611) call()
v_accvgpr_write a28, 0.00000 // (op 1612) call()
v_accvgpr_write a29, 0.00000 // (op 1612) call()
v_accvgpr_write a30, 0.00000 // (op 1612) call()
v_accvgpr_write a31, 0.00000 // (op 1612) call()
v_accvgpr_write a32, 0.00000 // (op 2199) call()
v_accvgpr_write a33, 0.00000 // (op 2199) call()
v_accvgpr_write a34, 0.00000 // (op 2199) call()
v_accvgpr_write a35, 0.00000 // (op 2199) call()
v_accvgpr_write a36, 0.00000 // (op 2612) call()
v_accvgpr_write a37, 0.00000 // (op 2612) call()
v_accvgpr_write a38, 0.00000 // (op 2612) call()
v_accvgpr_write a39, 0.00000 // (op 2612) call()
v_accvgpr_write a40, 0.00000 // (op 2613) call()
v_accvgpr_write a41, 0.00000 // (op 2613) call()
v_accvgpr_write a42, 0.00000 // (op 2613) call()
v_accvgpr_write a43, 0.00000 // (op 2613) call()
v_accvgpr_write a44, 0.00000 // (op 2614) call()
v_accvgpr_write a45, 0.00000 // (op 2614) call()
v_accvgpr_write a46, 0.00000 // (op 2614) call()
v_accvgpr_write a47, 0.00000 // (op 2614) call()
v_accvgpr_write a48, 0.00000 // (op 3201) call()
v_accvgpr_write a49, 0.00000 // (op 3201) call()
v_accvgpr_write a50, 0.00000 // (op 3201) call()
v_accvgpr_write a51, 0.00000 // (op 3201) call()
v_accvgpr_write a52, 0.00000 // (op 3614) call()
v_accvgpr_write a53, 0.00000 // (op 3614) call()
v_accvgpr_write a54, 0.00000 // (op 3614) call()
v_accvgpr_write a55, 0.00000 // (op 3614) call()
v_accvgpr_write a56, 0.00000 // (op 3615) call()
v_accvgpr_write a57, 0.00000 // (op 3615) call()
v_accvgpr_write a58, 0.00000 // (op 3615) call()
v_accvgpr_write a59, 0.00000 // (op 3615) call()
v_accvgpr_write a60, 0.00000 // (op 3616) call()
v_accvgpr_write a61, 0.00000 // (op 3616) call()
v_accvgpr_write a62, 0.00000 // (op 3616) call()
v_accvgpr_write a63, 0.00000 // (op 3616) call()
v_accvgpr_write a64, 0.00000 // (op 4203) call()
v_accvgpr_write a65, 0.00000 // (op 4203) call()
v_accvgpr_write a66, 0.00000 // (op 4203) call()
v_accvgpr_write a67, 0.00000 // (op 4203) call()
v_accvgpr_write a68, 0.00000 // (op 4616) call()
v_accvgpr_write a69, 0.00000 // (op 4616) call()
v_accvgpr_write a70, 0.00000 // (op 4616) call()
v_accvgpr_write a71, 0.00000 // (op 4616) call()
v_accvgpr_write a72, 0.00000 // (op 4617) call()
v_accvgpr_write a73, 0.00000 // (op 4617) call()
v_accvgpr_write a74, 0.00000 // (op 4617) call()
v_accvgpr_write a75, 0.00000 // (op 4617) call()
v_accvgpr_write a76, 0.00000 // (op 4618) call()
v_accvgpr_write a77, 0.00000 // (op 4618) call()
v_accvgpr_write a78, 0.00000 // (op 4618) call()
v_accvgpr_write a79, 0.00000 // (op 4618) call()
v_accvgpr_write a80, 0.00000 // (op 5205) call()
v_accvgpr_write a81, 0.00000 // (op 5205) call()
v_accvgpr_write a82, 0.00000 // (op 5205) call()
v_accvgpr_write a83, 0.00000 // (op 5205) call()
v_accvgpr_write a84, 0.00000 // (op 5618) call()
v_accvgpr_write a85, 0.00000 // (op 5618) call()
v_accvgpr_write a86, 0.00000 // (op 5618) call()
v_accvgpr_write a87, 0.00000 // (op 5618) call()
v_accvgpr_write a88, 0.00000 // (op 5619) call()
v_accvgpr_write a89, 0.00000 // (op 5619) call()
v_accvgpr_write a90, 0.00000 // (op 5619) call()
v_accvgpr_write a91, 0.00000 // (op 5619) call()
v_accvgpr_write a92, 0.00000 // (op 5620) call()
v_accvgpr_write a93, 0.00000 // (op 5620) call()
v_accvgpr_write a94, 0.00000 // (op 5620) call()
v_accvgpr_write a95, 0.00000 // (op 5620) call()
v_accvgpr_write a96, 0.00000 // (op 6207) call()
v_accvgpr_write a97, 0.00000 // (op 6207) call()
v_accvgpr_write a98, 0.00000 // (op 6207) call()
v_accvgpr_write a99, 0.00000 // (op 6207) call()
v_accvgpr_write a100, 0.00000 // (op 6620) call()
v_accvgpr_write a101, 0.00000 // (op 6620) call()
v_accvgpr_write a102, 0.00000 // (op 6620) call()
v_accvgpr_write a103, 0.00000 // (op 6620) call()
v_accvgpr_write a104, 0.00000 // (op 6621) call()
v_accvgpr_write a105, 0.00000 // (op 6621) call()
v_accvgpr_write a106, 0.00000 // (op 6621) call()
v_accvgpr_write a107, 0.00000 // (op 6621) call()
v_accvgpr_write a108, 0.00000 // (op 6622) call()
v_accvgpr_write a109, 0.00000 // (op 6622) call()
v_accvgpr_write a110, 0.00000 // (op 6622) call()
v_accvgpr_write a111, 0.00000 // (op 6622) call()
v_accvgpr_write a112, 0.00000 // (op 7209) call()
v_accvgpr_write a113, 0.00000 // (op 7209) call()
v_accvgpr_write a114, 0.00000 // (op 7209) call()
v_accvgpr_write a115, 0.00000 // (op 7209) call()
v_accvgpr_write a116, 0.00000 // (op 7622) call()
v_accvgpr_write a117, 0.00000 // (op 7622) call()
v_accvgpr_write a118, 0.00000 // (op 7622) call()
v_accvgpr_write a119, 0.00000 // (op 7622) call()
v_accvgpr_write a120, 0.00000 // (op 7623) call()
v_accvgpr_write a121, 0.00000 // (op 7623) call()
v_accvgpr_write a122, 0.00000 // (op 7623) call()
v_accvgpr_write a123, 0.00000 // (op 7623) call()
v_accvgpr_write a124, 0.00000 // (op 7624) call()
v_accvgpr_write a125, 0.00000 // (op 7624) call()
v_accvgpr_write a126, 0.00000 // (op 7624) call()
v_accvgpr_write a127, 0.00000 // (op 7624) call()
s_mul_hi_i32 s0, s16, s10 // (op 20532) 
v_lshrrev_b32 v14, 2, v1 // (op 20532) 
s_add_i32 s1, s0, s16 // (op 20532) 

v_and_b32 v15, -4, v1 // (op 20532) 
s_ashr_i32 s0, s1, 31 // (op 20532) 
v_sub_u32 v16, v1, v15 // (op 20532) 

s_cmp_eq_i32 s10, 0 // (op 20532) 
s_mov_b32 s18, scc // (op 20532) 
v_lshlrev_b32 v15, 5, v16 // (op 20532) 

s_cmp_eq_i32 s18, 0 // (op 20532) 
s_cselect_b32 s19, 0, -1 // (op 20532) 

v_mov_b32 v17, 0 // (op 20532) convert
v_mov_b32 v16, v15 // (op 20532) convert

s_and_b32 s18, s13, 31 // (op 20532) 
s_lshl_b32 s22, 1, s18 // (op 20532) 
s_add_i32 s23, s22, s19 // (op 20532) 


s_and_b32 s19, s0, s23 // (op 20532) 


s_add_i32 s0, s1, s19 // (op 20532) 


s_ashr_i32 s1, s0, s18 // (op 20532) 


s_lshl_b32 s0, s13, 24 // (op 20532) 
s_ashr_i32 s18, s0, 31 // (op 20532) 

s_xor_b32 s0, s1, s18 // (op 20532) 

s_sub_i32 s1, s0, s18 // (op 20532) 


s_mul_hi_i32 s0, s1, s14 // (op 20532) 
s_add_i32 s18, s0, s1 // (op 20532) 

s_ashr_i32 s0, s18, 31 // (op 20532) 
s_cmp_eq_i32 s14, 0 // (op 20532) 
s_mov_b32 s19, scc // (op 20532) 
s_cmp_eq_i32 s19, 0 // (op 20532) 
s_cselect_b32 s22, 0, -1 // (op 20532) 

s_and_b32 s19, s15, 31 // (op 20532) 
s_lshl_b32 s23, 1, s19 // (op 20532) 
s_add_i32 s24, s23, s22 // (op 20532) 


s_and_b32 s25, s0, s24 // (op 20532) 


s_add_i32 s0, s18, s25 // (op 20532) 


s_ashr_i32 s18, s0, s19 // (op 20532) 


s_lshl_b32 s0, s15, 24 // (op 20532) 
s_ashr_i32 s19, s0, 31 // (op 20532) 

s_xor_b32 s0, s18, s19 // (op 20532) 

s_sub_i32 s18, s0, s19 // (op 20532) 


s_mul_i32 s0, s1, s11 // (op 20532) 

s_sub_u32 s1, s16, s0 // (op 20532) 

s_mul_i32 s0, s18, s11 // (op 20532) 
s_add_u32 s0, s0, s1 // (op 20532) 


s_mul_hi_i32 s1, s16, s9 // (op 20532) 
s_add_i32 s18, s1, s16 // (op 20532) 

s_ashr_i32 s1, s18, 31 // (op 20532) 
s_cmp_eq_i32 s9, 0 // (op 20532) 
s_mov_b32 s19, scc // (op 20532) 
s_cmp_eq_i32 s19, 0 // (op 20532) 
s_cselect_b32 s22, 0, -1 // (op 20532) 

s_and_b32 s19, s45, 31 // (op 20532) 
s_lshl_b32 s23, 1, s19 // (op 20532) 
s_add_i32 s24, s23, s22 // (op 20532) 


s_and_b32 s25, s1, s24 // (op 20532) 


s_add_i32 s1, s18, s25 // (op 20532) 


s_ashr_i32 s18, s1, s19 // (op 20532) 


s_lshl_b32 s1, s45, 24 // (op 20532) 
s_ashr_i32 s19, s1, 31 // (op 20532) 

s_xor_b32 s1, s18, s19 // (op 20532) 

s_sub_i32 s18, s1, s19 // (op 20532) 


s_add_u32 s22, s6, 128 // (op 20532) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 20532) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 20532) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 20532) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 20532) 

s_mul_i32 s1, s11, s22 // (op 20532) 

s_mul_i32 s19, s8, s1 // (op 20532) 

s_mul_i32 s1, s18, s19 // (op 20532) 

s_sub_u32 s19, s16, s1 // (op 20532) 

s_mov_b32 s1, s19 // (op 20532) 
s_mul_hi_i32 s22, s1, s12 // (op 20532) 
s_add_i32 s23, s22, s1 // (op 20532) 


s_ashr_i32 s1, s23, 31 // (op 20532) 
s_cmp_eq_i32 s12, 0 // (op 20532) 
s_mov_b32 s22, scc // (op 20532) 
s_cmp_eq_i32 s22, 0 // (op 20532) 
s_cselect_b32 s24, 0, -1 // (op 20532) 

s_and_b32 s22, s44, 31 // (op 20532) 
s_lshl_b32 s25, 1, s22 // (op 20532) 
s_add_i32 s47, s25, s24 // (op 20532) 


s_and_b32 s24, s1, s47 // (op 20532) 


s_add_i32 s1, s23, s24 // (op 20532) 


s_ashr_i32 s23, s1, s22 // (op 20532) 


s_lshl_b32 s1, s44, 24 // (op 20532) 
s_ashr_i32 s22, s1, 31 // (op 20532) 

s_xor_b32 s1, s23, s22 // (op 20532) 

s_sub_i32 s23, s1, s22 // (op 20532) 


s_mul_i32 s1, s46, s11 // (op 20532) 
s_add_u32 s24, s20, 256 // (op 20532) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 20532) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 20532) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 20532) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 8 // (op 20532) 

s_sub_i32 s22, s24, s1 // (op 20532) 

s_mul_i32 s24, s23, s22 // (op 20532) 


s_sub_u32 s25, s19, s24 // (op 20532) 


s_add_u32 s19, s25, s1 // (op 20532) 


s_cmp_eq_i32 s18, 0 // (op 20532) 
s_mov_b32 s1, scc // (op 20532) 

s_cmp_eq_i32 s1, 0 // (op 20532) 
s_cselect_b32 s18, s19, s0 // (op 20532) 



v_lshl_add_u32 v15, s18, 8, v14 // (op 20532) 


v_mul_lo_u32 v19, v15, s27 // (op 20532) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 20532) most significant: high of low * low
v_mul_lo_u32 v18, v15, s26 // (op 20532) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 20532) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 20532) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 20532) most significant half



v_lshlrev_b64 v[14:15], 2, v[18:19] // (op 20532) 

v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 20532) 

s_mul_i32 s1, s26, s21 // (op 20534) most significant: low of low * high
s_mul_i32 s18, s27, s20 // (op 20534) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 20534) least significant
s_mul_hi_u32 s19, s26, s20 // (op 20534) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 20534) most significant: sum
s_add_u32 s1, s1, s19 // (op 20534) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 20534) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 20534) 

s_mov_b32 s48, s28 // (op 20534) reg expression
s_mov_b32 s49, s29 // (op 20534) reg expression
s_mov_b32 s50, s0 // (op 20534) reg expression
s_mov_b32 s51, 131072 // (op 20534) call()

s_mul_hi_i32 s47, s16, s10 // (op 20551) 
v_lshrrev_b32 v14, 2, v1 // (op 20551) 
s_add_i32 s0, s47, s16 // (op 20551) 

v_and_b32 v15, -4, v1 // (op 20551) 
s_ashr_i32 s1, s0, 31 // (op 20551) 
v_sub_u32 v18, v1, v15 // (op 20551) 

s_cmp_eq_i32 s10, 0 // (op 20551) 
s_mov_b32 s47, scc // (op 20551) 
v_lshlrev_b32 v15, 5, v18 // (op 20551) 

s_cmp_eq_i32 s47, 0 // (op 20551) 
s_cselect_b32 s18, 0, -1 // (op 20551) 

v_mov_b32 v19, 0 // (op 20551) convert
v_mov_b32 v18, v15 // (op 20551) convert

s_and_b32 s19, s13, 31 // (op 20551) 
s_lshl_b32 s47, 1, s19 // (op 20551) 
s_add_i32 s22, s47, s18 // (op 20551) 


s_and_b32 s18, s1, s22 // (op 20551) 


s_add_i32 s1, s0, s18 // (op 20551) 


s_ashr_i32 s0, s1, s19 // (op 20551) 


s_lshl_b32 s1, s13, 24 // (op 20551) 
s_ashr_i32 s47, s1, 31 // (op 20551) 

s_xor_b32 s1, s0, s47 // (op 20551) 

s_sub_i32 s0, s1, s47 // (op 20551) 


s_mul_hi_i32 s1, s0, s14 // (op 20551) 
s_add_i32 s47, s1, s0 // (op 20551) 

s_ashr_i32 s1, s47, 31 // (op 20551) 
s_cmp_eq_i32 s14, 0 // (op 20551) 
s_mov_b32 s18, scc // (op 20551) 
s_cmp_eq_i32 s18, 0 // (op 20551) 
s_cselect_b32 s19, 0, -1 // (op 20551) 

s_and_b32 s18, s15, 31 // (op 20551) 
s_lshl_b32 s22, 1, s18 // (op 20551) 
s_add_i32 s23, s22, s19 // (op 20551) 


s_and_b32 s19, s1, s23 // (op 20551) 


s_add_i32 s1, s47, s19 // (op 20551) 


s_ashr_i32 s19, s1, s18 // (op 20551) 


s_lshl_b32 s1, s15, 24 // (op 20551) 
s_ashr_i32 s18, s1, 31 // (op 20551) 

s_xor_b32 s1, s19, s18 // (op 20551) 

s_sub_i32 s19, s1, s18 // (op 20551) 


s_mul_i32 s1, s0, s11 // (op 20551) 

s_sub_u32 s0, s16, s1 // (op 20551) 

s_mul_i32 s1, s19, s11 // (op 20551) 
s_add_u32 s1, s1, s0 // (op 20551) 


s_mul_hi_i32 s0, s16, s9 // (op 20551) 
s_add_i32 s47, s0, s16 // (op 20551) 

s_ashr_i32 s0, s47, 31 // (op 20551) 
s_cmp_eq_i32 s9, 0 // (op 20551) 
s_mov_b32 s18, scc // (op 20551) 
s_cmp_eq_i32 s18, 0 // (op 20551) 
s_cselect_b32 s19, 0, -1 // (op 20551) 

s_and_b32 s18, s45, 31 // (op 20551) 
s_lshl_b32 s22, 1, s18 // (op 20551) 
s_add_i32 s23, s22, s19 // (op 20551) 


s_and_b32 s19, s0, s23 // (op 20551) 


s_add_i32 s0, s47, s19 // (op 20551) 


s_ashr_i32 s19, s0, s18 // (op 20551) 


s_lshl_b32 s0, s45, 24 // (op 20551) 
s_ashr_i32 s18, s0, 31 // (op 20551) 

s_xor_b32 s0, s19, s18 // (op 20551) 

s_sub_i32 s19, s0, s18 // (op 20551) 


s_add_u32 s22, s6, 128 // (op 20551) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 20551) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 20551) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 20551) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 20551) 

s_mul_i32 s0, s11, s22 // (op 20551) 

s_mul_i32 s18, s8, s0 // (op 20551) 

s_mul_i32 s0, s19, s18 // (op 20551) 

s_sub_u32 s18, s16, s0 // (op 20551) 

s_mov_b32 s0, s18 // (op 20551) 
s_mul_hi_i32 s47, s0, s12 // (op 20551) 
s_add_i32 s22, s47, s0 // (op 20551) 


s_ashr_i32 s0, s22, 31 // (op 20551) 
s_cmp_eq_i32 s12, 0 // (op 20551) 
s_mov_b32 s47, scc // (op 20551) 
s_cmp_eq_i32 s47, 0 // (op 20551) 
s_cselect_b32 s23, 0, -1 // (op 20551) 

s_and_b32 s47, s44, 31 // (op 20551) 
s_lshl_b32 s24, 1, s47 // (op 20551) 
s_add_i32 s25, s24, s23 // (op 20551) 


s_and_b32 s23, s0, s25 // (op 20551) 


s_add_i32 s0, s22, s23 // (op 20551) 


s_ashr_i32 s22, s0, s47 // (op 20551) 


s_lshl_b32 s0, s44, 24 // (op 20551) 
s_ashr_i32 s47, s0, 31 // (op 20551) 

s_xor_b32 s0, s22, s47 // (op 20551) 

s_sub_i32 s22, s0, s47 // (op 20551) 


s_mul_i32 s0, s46, s11 // (op 20551) 
s_add_u32 s24, s20, 256 // (op 20551) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 20551) End of Int64 add, unlocking SCC
s_sub_u32 s60, s24, 1 // (op 20551) least significant half; sets scc
s_subb_u32 s61, s25, 0 // (op 20551) most significant half; uses scc

s_ashr_i64 s[24:25], s[60:61], 8 // (op 20551) 

s_sub_i32 s23, s24, s0 // (op 20551) 

s_mul_i32 s47, s22, s23 // (op 20551) 


s_sub_u32 s22, s18, s47 // (op 20551) 


s_add_u32 s18, s22, s0 // (op 20551) 


s_cmp_eq_i32 s19, 0 // (op 20551) 
s_mov_b32 s0, scc // (op 20551) 

s_cmp_eq_i32 s0, 0 // (op 20551) 
s_cselect_b32 s19, s18, s1 // (op 20551) 



v_lshl_add_u32 v15, s19, 8, v14 // (op 20551) 


v_mul_lo_u32 v21, v15, s27 // (op 20551) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 20551) most significant: high of low * low
v_mul_lo_u32 v20, v15, s26 // (op 20551) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 20551) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v18 // (op 20551) least significant half
v_addc_co_u32 v21, s[0:1], v21, v19, s[0:1] // (op 20551) most significant half



v_lshlrev_b64 v[14:15], 2, v[20:21] // (op 20551) 

v_lshrrev_b64 v[18:19], 3, v[14:15] // (op 20551) 

s_mul_i32 s1, s26, s21 // (op 20553) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 20553) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 20553) least significant
s_mul_hi_u32 s18, s26, s20 // (op 20553) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 20553) most significant: sum
s_add_u32 s1, s1, s18 // (op 20553) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 20553) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 20553) 

s_mov_b32 s48, s28 // (op 20553) reg expression
s_mov_b32 s49, s29 // (op 20553) reg expression
s_mov_b32 s50, s0 // (op 20553) reg expression
s_mov_b32 s51, 131072 // (op 20553) call()

s_mul_hi_i32 s47, s16, s10 // (op 20596) 
v_lshrrev_b32 v14, 2, v1 // (op 20596) 
s_add_i32 s0, s47, s16 // (op 20596) 

v_and_b32 v15, -4, v1 // (op 20596) 
s_ashr_i32 s1, s0, 31 // (op 20596) 
v_sub_u32 v20, v1, v15 // (op 20596) 

s_cmp_eq_i32 s10, 0 // (op 20596) 
s_mov_b32 s47, scc // (op 20596) 
v_lshlrev_b32 v15, 5, v20 // (op 20596) 

s_cmp_eq_i32 s47, 0 // (op 20596) 
s_cselect_b32 s18, 0, -1 // (op 20596) 

v_mov_b32 v21, 0 // (op 20596) convert
v_mov_b32 v20, v15 // (op 20596) convert

s_and_b32 s19, s13, 31 // (op 20596) 
s_lshl_b32 s47, 1, s19 // (op 20596) 
s_add_i32 s22, s47, s18 // (op 20596) 


s_and_b32 s18, s1, s22 // (op 20596) 


s_add_i32 s1, s0, s18 // (op 20596) 


s_ashr_i32 s0, s1, s19 // (op 20596) 


s_lshl_b32 s1, s13, 24 // (op 20596) 
s_ashr_i32 s47, s1, 31 // (op 20596) 

s_xor_b32 s1, s0, s47 // (op 20596) 

s_sub_i32 s0, s1, s47 // (op 20596) 


s_mul_hi_i32 s1, s0, s14 // (op 20596) 
s_add_i32 s47, s1, s0 // (op 20596) 

s_ashr_i32 s1, s47, 31 // (op 20596) 
s_cmp_eq_i32 s14, 0 // (op 20596) 
s_mov_b32 s18, scc // (op 20596) 
s_cmp_eq_i32 s18, 0 // (op 20596) 
s_cselect_b32 s19, 0, -1 // (op 20596) 

s_and_b32 s18, s15, 31 // (op 20596) 
s_lshl_b32 s22, 1, s18 // (op 20596) 
s_add_i32 s23, s22, s19 // (op 20596) 


s_and_b32 s19, s1, s23 // (op 20596) 


s_add_i32 s1, s47, s19 // (op 20596) 


s_ashr_i32 s19, s1, s18 // (op 20596) 


s_lshl_b32 s1, s15, 24 // (op 20596) 
s_ashr_i32 s18, s1, 31 // (op 20596) 

s_xor_b32 s1, s19, s18 // (op 20596) 

s_sub_i32 s19, s1, s18 // (op 20596) 


s_add_u32 s22, s6, 128 // (op 20596) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 20596) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 20596) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 20596) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 20596) 

s_mul_i32 s1, s19, s22 // (op 20596) 

s_sub_i32 s47, s0, s1 // (op 20596) 


s_mul_hi_i32 s0, s16, s9 // (op 20596) 
s_add_i32 s1, s0, s16 // (op 20596) 

s_ashr_i32 s0, s1, 31 // (op 20596) 
s_cmp_eq_i32 s9, 0 // (op 20596) 
s_mov_b32 s18, scc // (op 20596) 
s_cmp_eq_i32 s18, 0 // (op 20596) 
s_cselect_b32 s19, 0, -1 // (op 20596) 

s_and_b32 s18, s45, 31 // (op 20596) 
s_lshl_b32 s24, 1, s18 // (op 20596) 
s_add_i32 s25, s24, s19 // (op 20596) 


s_and_b32 s19, s0, s25 // (op 20596) 


s_add_i32 s0, s1, s19 // (op 20596) 


s_ashr_i32 s1, s0, s18 // (op 20596) 


s_lshl_b32 s0, s45, 24 // (op 20596) 
s_ashr_i32 s18, s0, 31 // (op 20596) 

s_xor_b32 s0, s1, s18 // (op 20596) 

s_sub_i32 s1, s0, s18 // (op 20596) 


s_mul_i32 s0, s11, s22 // (op 20596) 

s_mul_i32 s18, s8, s0 // (op 20596) 

s_mul_i32 s0, s1, s18 // (op 20596) 

s_sub_u32 s18, s16, s0 // (op 20596) 

s_mul_hi_i32 s0, s18, s12 // (op 20596) 
s_add_i32 s19, s0, s18 // (op 20596) 


s_ashr_i32 s0, s19, 31 // (op 20596) 
s_cmp_eq_i32 s12, 0 // (op 20596) 
s_mov_b32 s18, scc // (op 20596) 
s_cmp_eq_i32 s18, 0 // (op 20596) 
s_cselect_b32 s22, 0, -1 // (op 20596) 

s_and_b32 s18, s44, 31 // (op 20596) 
s_lshl_b32 s23, 1, s18 // (op 20596) 
s_add_i32 s24, s23, s22 // (op 20596) 


s_and_b32 s25, s0, s24 // (op 20596) 


s_add_i32 s0, s19, s25 // (op 20596) 


s_ashr_i32 s19, s0, s18 // (op 20596) 


s_lshl_b32 s0, s44, 24 // (op 20596) 
s_ashr_i32 s18, s0, 31 // (op 20596) 

s_xor_b32 s0, s19, s18 // (op 20596) 

s_sub_i32 s19, s0, s18 // (op 20596) 


s_cmp_eq_i32 s1, 0 // (op 20596) 
s_mov_b32 s0, scc // (op 20596) 

s_cmp_eq_i32 s0, 0 // (op 20596) 
s_cselect_b32 s1, s19, s47 // (op 20596) 



v_lshl_add_u32 v15, s1, 7, v14 // (op 20596) 


v_mul_lo_u32 v23, v15, s31 // (op 20596) most significant: low of low * high
v_mul_hi_u32 v14, v15, s30 // (op 20596) most significant: high of low * low
v_mul_lo_u32 v22, v15, s30 // (op 20596) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 20596) most significant: sum


v_add_co_u32 v14, s[0:1], v20, v22 // (op 20596) least significant half
v_addc_co_u32 v15, s[0:1], v21, v23, s[0:1] // (op 20596) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 20596) 


v_lshrrev_b64 v[20:21], 3, v[14:15] // (op 20596) 

s_mul_i32 s1, s30, s7 // (op 20598) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 20598) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 20598) least significant
s_mul_hi_u32 s18, s30, s6 // (op 20598) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 20598) most significant: sum
s_add_u32 s1, s1, s18 // (op 20598) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 20598) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 20598) 

s_mov_b32 s60, s32 // (op 20598) reg expression
s_mov_b32 s61, s33 // (op 20598) reg expression
s_mov_b32 s62, s0 // (op 20598) reg expression
s_mov_b32 s63, 131072 // (op 20598) call()

s_mul_hi_i32 s47, s16, s10 // (op 20615) 
v_lshrrev_b32 v14, 2, v1 // (op 20615) 
s_add_i32 s0, s47, s16 // (op 20615) 

v_and_b32 v15, -4, v1 // (op 20615) 
s_ashr_i32 s1, s0, 31 // (op 20615) 
v_sub_u32 v22, v1, v15 // (op 20615) 

s_cmp_eq_i32 s10, 0 // (op 20615) 
s_mov_b32 s47, scc // (op 20615) 
v_lshlrev_b32 v15, 5, v22 // (op 20615) 

s_cmp_eq_i32 s47, 0 // (op 20615) 
s_cselect_b32 s18, 0, -1 // (op 20615) 

v_mov_b32 v23, 0 // (op 20615) convert
v_mov_b32 v22, v15 // (op 20615) convert

s_and_b32 s19, s13, 31 // (op 20615) 
s_lshl_b32 s47, 1, s19 // (op 20615) 
s_add_i32 s22, s47, s18 // (op 20615) 


s_and_b32 s18, s1, s22 // (op 20615) 


s_add_i32 s1, s0, s18 // (op 20615) 


s_ashr_i32 s0, s1, s19 // (op 20615) 


s_lshl_b32 s1, s13, 24 // (op 20615) 
s_ashr_i32 s47, s1, 31 // (op 20615) 

s_xor_b32 s1, s0, s47 // (op 20615) 

s_sub_i32 s0, s1, s47 // (op 20615) 


s_mul_hi_i32 s1, s0, s14 // (op 20615) 
s_add_i32 s47, s1, s0 // (op 20615) 

s_ashr_i32 s1, s47, 31 // (op 20615) 
s_cmp_eq_i32 s14, 0 // (op 20615) 
s_mov_b32 s18, scc // (op 20615) 
s_cmp_eq_i32 s18, 0 // (op 20615) 
s_cselect_b32 s19, 0, -1 // (op 20615) 

s_and_b32 s18, s15, 31 // (op 20615) 
s_lshl_b32 s22, 1, s18 // (op 20615) 
s_add_i32 s23, s22, s19 // (op 20615) 


s_and_b32 s19, s1, s23 // (op 20615) 


s_add_i32 s1, s47, s19 // (op 20615) 


s_ashr_i32 s19, s1, s18 // (op 20615) 


s_lshl_b32 s1, s15, 24 // (op 20615) 
s_ashr_i32 s18, s1, 31 // (op 20615) 

s_xor_b32 s1, s19, s18 // (op 20615) 

s_sub_i32 s19, s1, s18 // (op 20615) 


s_add_u32 s22, s6, 128 // (op 20615) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 20615) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 20615) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 20615) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 20615) 

s_mul_i32 s1, s19, s22 // (op 20615) 

s_sub_i32 s47, s0, s1 // (op 20615) 


s_mul_hi_i32 s0, s16, s9 // (op 20615) 
s_add_i32 s1, s0, s16 // (op 20615) 

s_ashr_i32 s0, s1, 31 // (op 20615) 
s_cmp_eq_i32 s9, 0 // (op 20615) 
s_mov_b32 s18, scc // (op 20615) 
s_cmp_eq_i32 s18, 0 // (op 20615) 
s_cselect_b32 s19, 0, -1 // (op 20615) 

s_and_b32 s18, s45, 31 // (op 20615) 
s_lshl_b32 s24, 1, s18 // (op 20615) 
s_add_i32 s25, s24, s19 // (op 20615) 


s_and_b32 s19, s0, s25 // (op 20615) 


s_add_i32 s0, s1, s19 // (op 20615) 


s_ashr_i32 s1, s0, s18 // (op 20615) 


s_lshl_b32 s0, s45, 24 // (op 20615) 
s_ashr_i32 s18, s0, 31 // (op 20615) 

s_xor_b32 s0, s1, s18 // (op 20615) 

s_sub_i32 s1, s0, s18 // (op 20615) 


s_mul_i32 s0, s11, s22 // (op 20615) 

s_mul_i32 s18, s8, s0 // (op 20615) 

s_mul_i32 s0, s1, s18 // (op 20615) 

s_sub_u32 s18, s16, s0 // (op 20615) 

s_mul_hi_i32 s0, s18, s12 // (op 20615) 
s_add_i32 s19, s0, s18 // (op 20615) 


s_ashr_i32 s0, s19, 31 // (op 20615) 
s_cmp_eq_i32 s12, 0 // (op 20615) 
s_mov_b32 s18, scc // (op 20615) 
s_cmp_eq_i32 s18, 0 // (op 20615) 
s_cselect_b32 s22, 0, -1 // (op 20615) 

s_and_b32 s18, s44, 31 // (op 20615) 
s_lshl_b32 s23, 1, s18 // (op 20615) 
s_add_i32 s24, s23, s22 // (op 20615) 


s_and_b32 s25, s0, s24 // (op 20615) 


s_add_i32 s0, s19, s25 // (op 20615) 


s_ashr_i32 s19, s0, s18 // (op 20615) 


s_lshl_b32 s0, s44, 24 // (op 20615) 
s_ashr_i32 s18, s0, 31 // (op 20615) 

s_xor_b32 s0, s19, s18 // (op 20615) 

s_sub_i32 s19, s0, s18 // (op 20615) 


s_cmp_eq_i32 s1, 0 // (op 20615) 
s_mov_b32 s0, scc // (op 20615) 

s_cmp_eq_i32 s0, 0 // (op 20615) 
s_cselect_b32 s1, s19, s47 // (op 20615) 



v_lshl_add_u32 v15, s1, 7, v14 // (op 20615) 


v_mul_lo_u32 v25, v15, s31 // (op 20615) most significant: low of low * high
v_mul_hi_u32 v14, v15, s30 // (op 20615) most significant: high of low * low
v_mul_lo_u32 v24, v15, s30 // (op 20615) least significant: low of low * low
v_add_u32 v25, v25, v14 // (op 20615) most significant: sum


v_add_co_u32 v14, s[0:1], v22, v24 // (op 20615) least significant half
v_addc_co_u32 v15, s[0:1], v23, v25, s[0:1] // (op 20615) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 20615) 


v_lshrrev_b64 v[22:23], 3, v[14:15] // (op 20615) 

s_mul_i32 s1, s30, s7 // (op 20617) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 20617) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 20617) least significant
s_mul_hi_u32 s18, s30, s6 // (op 20617) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 20617) most significant: sum
s_add_u32 s1, s1, s18 // (op 20617) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 20617) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 20617) 

s_mov_b32 s60, s32 // (op 20617) reg expression
s_mov_b32 s61, s33 // (op 20617) reg expression
s_mov_b32 s62, s0 // (op 20617) reg expression
s_mov_b32 s63, 131072 // (op 20617) call()

s_mul_hi_i32 s47, s16, s10 // (op 21069) 
v_and_b32 v14, 63, v1 // (op 21069) 
s_add_i32 s0, s47, s16 // (op 21069) 

v_lshrrev_b32 v15, 4, v14 // (op 21069) 
s_ashr_i32 s1, s0, 31 // (op 21069) 
v_and_b32 v24, 3, v15 // (op 21069) 

s_cmp_eq_i32 s10, 0 // (op 21069) 
s_mov_b32 s47, scc // (op 21069) 
v_and_b32 v15, 15, v1 // (op 21069) 
s_cmp_eq_i32 s47, 0 // (op 21069) 
s_cselect_b32 s18, 0, -1 // (op 21069) 

v_lshl_add_u32 v25, v24, 4, v15 // (op 21069) 


s_and_b32 s19, s13, 31 // (op 21069) 
v_and_b32 v15, 4294967168, v1 // (op 21069) 
s_lshl_b32 s47, 1, s19 // (op 21069) 
v_add_u32 v24, v15, v25 // (op 21069) 


s_add_i32 s22, s47, s18 // (op 21069) 


v_lshrrev_b32 v15, 6, v14 // (op 21069) 

s_and_b32 s18, s1, s22 // (op 21069) 


v_lshlrev_b32 v14, 2, v15 // (op 21069) 

s_add_i32 s1, s0, s18 // (op 21069) 


s_ashr_i32 s0, s1, s19 // (op 21069) 


s_lshl_b32 s1, s13, 24 // (op 21069) 
s_ashr_i32 s47, s1, 31 // (op 21069) 

s_xor_b32 s1, s0, s47 // (op 21069) 

s_sub_i32 s0, s1, s47 // (op 21069) 


s_mul_hi_i32 s1, s0, s14 // (op 21069) 
s_add_i32 s47, s1, s0 // (op 21069) 

s_ashr_i32 s1, s47, 31 // (op 21069) 
s_cmp_eq_i32 s14, 0 // (op 21069) 
s_mov_b32 s18, scc // (op 21069) 
s_cmp_eq_i32 s18, 0 // (op 21069) 
s_cselect_b32 s19, 0, -1 // (op 21069) 

s_and_b32 s18, s15, 31 // (op 21069) 
s_lshl_b32 s22, 1, s18 // (op 21069) 
s_add_i32 s23, s22, s19 // (op 21069) 


s_and_b32 s19, s1, s23 // (op 21069) 


s_add_i32 s1, s47, s19 // (op 21069) 


s_ashr_i32 s19, s1, s18 // (op 21069) 


s_lshl_b32 s1, s15, 24 // (op 21069) 
s_ashr_i32 s18, s1, 31 // (op 21069) 

s_xor_b32 s1, s19, s18 // (op 21069) 

s_sub_i32 s19, s1, s18 // (op 21069) 


s_mul_i32 s1, s0, s11 // (op 21069) 

s_sub_u32 s0, s16, s1 // (op 21069) 

s_mul_i32 s1, s19, s11 // (op 21069) 
s_add_u32 s1, s1, s0 // (op 21069) 


s_mul_hi_i32 s0, s16, s9 // (op 21069) 
s_add_i32 s47, s0, s16 // (op 21069) 

s_ashr_i32 s0, s47, 31 // (op 21069) 
s_cmp_eq_i32 s9, 0 // (op 21069) 
s_mov_b32 s18, scc // (op 21069) 
s_cmp_eq_i32 s18, 0 // (op 21069) 
s_cselect_b32 s19, 0, -1 // (op 21069) 

s_and_b32 s18, s45, 31 // (op 21069) 
s_lshl_b32 s22, 1, s18 // (op 21069) 
s_add_i32 s23, s22, s19 // (op 21069) 


s_and_b32 s19, s0, s23 // (op 21069) 


s_add_i32 s0, s47, s19 // (op 21069) 


s_ashr_i32 s19, s0, s18 // (op 21069) 


s_lshl_b32 s0, s45, 24 // (op 21069) 
s_ashr_i32 s18, s0, 31 // (op 21069) 

s_xor_b32 s0, s19, s18 // (op 21069) 

s_sub_i32 s19, s0, s18 // (op 21069) 


s_add_u32 s22, s6, 128 // (op 21069) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 21069) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 21069) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 21069) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 21069) 

s_mul_i32 s0, s11, s22 // (op 21069) 

s_mul_i32 s18, s8, s0 // (op 21069) 

s_mul_i32 s0, s19, s18 // (op 21069) 

s_sub_u32 s18, s16, s0 // (op 21069) 

s_mov_b32 s0, s18 // (op 21069) 
s_mul_hi_i32 s47, s0, s12 // (op 21069) 
s_add_i32 s22, s47, s0 // (op 21069) 


s_ashr_i32 s0, s22, 31 // (op 21069) 
s_cmp_eq_i32 s12, 0 // (op 21069) 
s_mov_b32 s47, scc // (op 21069) 
s_cmp_eq_i32 s47, 0 // (op 21069) 
s_cselect_b32 s23, 0, -1 // (op 21069) 

s_and_b32 s47, s44, 31 // (op 21069) 
s_lshl_b32 s24, 1, s47 // (op 21069) 
s_add_i32 s25, s24, s23 // (op 21069) 


s_and_b32 s23, s0, s25 // (op 21069) 


s_add_i32 s0, s22, s23 // (op 21069) 


s_ashr_i32 s22, s0, s47 // (op 21069) 


s_lshl_b32 s0, s44, 24 // (op 21069) 
s_ashr_i32 s47, s0, 31 // (op 21069) 

s_xor_b32 s0, s22, s47 // (op 21069) 

s_sub_i32 s22, s0, s47 // (op 21069) 


s_mul_i32 s0, s46, s11 // (op 21069) 
s_add_u32 s24, s20, 256 // (op 21069) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 21069) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 21069) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 21069) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 8 // (op 21069) 

s_sub_i32 s23, s24, s0 // (op 21069) 

s_mul_i32 s47, s22, s23 // (op 21069) 


s_sub_u32 s22, s18, s47 // (op 21069) 


s_add_u32 s18, s22, s0 // (op 21069) 


s_cmp_eq_i32 s19, 0 // (op 21069) 
s_mov_b32 s0, scc // (op 21069) 

s_cmp_eq_i32 s0, 0 // (op 21069) 
s_cselect_b32 s19, s18, s1 // (op 21069) 



v_lshl_add_u32 v15, s19, 8, v24 // (op 21069) 


v_mul_lo_u32 v24, v15, s34 // (op 21069) 
v_add_u32 v24, v24, v14 // (op 21069) 


s_mul_i32 s1, s34, s21 // (op 21071) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 21071) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 21071) least significant
s_mul_hi_u32 s18, s34, s20 // (op 21071) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 21071) most significant: sum
s_add_u32 s1, s1, s18 // (op 21071) most significant: sum


s_mov_b32 s64, s36 // (op 21071) reg expression
s_mov_b32 s65, s37 // (op 21071) reg expression
s_mov_b32 s66, s0 // (op 21071) reg expression
s_mov_b32 s67, 131072 // (op 21071) call()

s_mul_hi_i32 s47, s16, s10 // (op 21090) 
v_and_b32 v14, 63, v1 // (op 21090) 
s_add_i32 s0, s47, s16 // (op 21090) 

v_lshrrev_b32 v15, 4, v14 // (op 21090) 
s_ashr_i32 s1, s0, 31 // (op 21090) 
v_and_b32 v25, 3, v15 // (op 21090) 

s_cmp_eq_i32 s10, 0 // (op 21090) 
s_mov_b32 s47, scc // (op 21090) 
v_and_b32 v15, 15, v1 // (op 21090) 
s_cmp_eq_i32 s47, 0 // (op 21090) 
s_cselect_b32 s18, 0, -1 // (op 21090) 

v_lshl_add_u32 v26, v25, 4, v15 // (op 21090) 


s_and_b32 s19, s13, 31 // (op 21090) 
v_and_b32 v15, 4294967168, v1 // (op 21090) 
s_lshl_b32 s47, 1, s19 // (op 21090) 
v_add_u32 v25, v15, v26 // (op 21090) 


s_add_i32 s22, s47, s18 // (op 21090) 


v_lshrrev_b32 v15, 6, v14 // (op 21090) 

s_and_b32 s18, s1, s22 // (op 21090) 


v_lshlrev_b32 v14, 2, v15 // (op 21090) 

s_add_i32 s1, s0, s18 // (op 21090) 


s_ashr_i32 s0, s1, s19 // (op 21090) 


s_lshl_b32 s1, s13, 24 // (op 21090) 
s_ashr_i32 s47, s1, 31 // (op 21090) 

s_xor_b32 s1, s0, s47 // (op 21090) 

s_sub_i32 s0, s1, s47 // (op 21090) 


s_mul_hi_i32 s1, s0, s14 // (op 21090) 
s_add_i32 s47, s1, s0 // (op 21090) 

s_ashr_i32 s1, s47, 31 // (op 21090) 
s_cmp_eq_i32 s14, 0 // (op 21090) 
s_mov_b32 s18, scc // (op 21090) 
s_cmp_eq_i32 s18, 0 // (op 21090) 
s_cselect_b32 s19, 0, -1 // (op 21090) 

s_and_b32 s18, s15, 31 // (op 21090) 
s_lshl_b32 s22, 1, s18 // (op 21090) 
s_add_i32 s23, s22, s19 // (op 21090) 


s_and_b32 s19, s1, s23 // (op 21090) 


s_add_i32 s1, s47, s19 // (op 21090) 


s_ashr_i32 s19, s1, s18 // (op 21090) 


s_lshl_b32 s1, s15, 24 // (op 21090) 
s_ashr_i32 s18, s1, 31 // (op 21090) 

s_xor_b32 s1, s19, s18 // (op 21090) 

s_sub_i32 s19, s1, s18 // (op 21090) 


s_mul_i32 s1, s0, s11 // (op 21090) 

s_sub_u32 s0, s16, s1 // (op 21090) 

s_mul_i32 s1, s19, s11 // (op 21090) 
s_add_u32 s1, s1, s0 // (op 21090) 


s_mul_hi_i32 s0, s16, s9 // (op 21090) 
s_add_i32 s47, s0, s16 // (op 21090) 

s_ashr_i32 s0, s47, 31 // (op 21090) 
s_cmp_eq_i32 s9, 0 // (op 21090) 
s_mov_b32 s18, scc // (op 21090) 
s_cmp_eq_i32 s18, 0 // (op 21090) 
s_cselect_b32 s19, 0, -1 // (op 21090) 

s_and_b32 s18, s45, 31 // (op 21090) 
s_lshl_b32 s22, 1, s18 // (op 21090) 
s_add_i32 s23, s22, s19 // (op 21090) 


s_and_b32 s19, s0, s23 // (op 21090) 


s_add_i32 s0, s47, s19 // (op 21090) 


s_ashr_i32 s19, s0, s18 // (op 21090) 


s_lshl_b32 s0, s45, 24 // (op 21090) 
s_ashr_i32 s18, s0, 31 // (op 21090) 

s_xor_b32 s0, s19, s18 // (op 21090) 

s_sub_i32 s19, s0, s18 // (op 21090) 


s_add_u32 s22, s6, 128 // (op 21090) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 21090) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 21090) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 21090) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 21090) 

s_mul_i32 s0, s11, s22 // (op 21090) 

s_mul_i32 s18, s8, s0 // (op 21090) 

s_mul_i32 s0, s19, s18 // (op 21090) 

s_sub_u32 s18, s16, s0 // (op 21090) 

s_mov_b32 s0, s18 // (op 21090) 
s_mul_hi_i32 s47, s0, s12 // (op 21090) 
s_add_i32 s22, s47, s0 // (op 21090) 


s_ashr_i32 s0, s22, 31 // (op 21090) 
s_cmp_eq_i32 s12, 0 // (op 21090) 
s_mov_b32 s47, scc // (op 21090) 
s_cmp_eq_i32 s47, 0 // (op 21090) 
s_cselect_b32 s23, 0, -1 // (op 21090) 

s_and_b32 s47, s44, 31 // (op 21090) 
s_lshl_b32 s24, 1, s47 // (op 21090) 
s_add_i32 s25, s24, s23 // (op 21090) 


s_and_b32 s23, s0, s25 // (op 21090) 


s_add_i32 s0, s22, s23 // (op 21090) 


s_ashr_i32 s22, s0, s47 // (op 21090) 


s_lshl_b32 s0, s44, 24 // (op 21090) 
s_ashr_i32 s47, s0, 31 // (op 21090) 

s_xor_b32 s0, s22, s47 // (op 21090) 

s_sub_i32 s22, s0, s47 // (op 21090) 


s_mul_i32 s0, s46, s11 // (op 21090) 
s_add_u32 s24, s20, 256 // (op 21090) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 21090) End of Int64 add, unlocking SCC
s_sub_u32 s68, s24, 1 // (op 21090) least significant half; sets scc
s_subb_u32 s69, s25, 0 // (op 21090) most significant half; uses scc

s_ashr_i64 s[24:25], s[68:69], 8 // (op 21090) 

s_sub_i32 s23, s24, s0 // (op 21090) 

s_mul_i32 s47, s22, s23 // (op 21090) 


s_sub_u32 s22, s18, s47 // (op 21090) 


s_add_u32 s18, s22, s0 // (op 21090) 


s_cmp_eq_i32 s19, 0 // (op 21090) 
s_mov_b32 s0, scc // (op 21090) 

s_cmp_eq_i32 s0, 0 // (op 21090) 
s_cselect_b32 s19, s18, s1 // (op 21090) 



v_lshl_add_u32 v15, s19, 8, v25 // (op 21090) 


v_mul_lo_u32 v25, v15, s34 // (op 21090) 
v_add_u32 v25, v25, v14 // (op 21090) 


s_mul_i32 s1, s34, s21 // (op 21092) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 21092) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 21092) least significant
s_mul_hi_u32 s18, s34, s20 // (op 21092) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 21092) most significant: sum
s_add_u32 s1, s1, s18 // (op 21092) most significant: sum


s_mov_b32 s64, s36 // (op 21092) reg expression
s_mov_b32 s65, s37 // (op 21092) reg expression
s_mov_b32 s66, s0 // (op 21092) reg expression
s_mov_b32 s67, 131072 // (op 21092) call()

s_mul_hi_i32 s47, s16, s10 // (op 21121) 
v_and_b32 v14, 63, v1 // (op 21121) 
s_add_i32 s0, s47, s16 // (op 21121) 

v_lshrrev_b32 v15, 4, v14 // (op 21121) 
s_ashr_i32 s1, s0, 31 // (op 21121) 
v_and_b32 v26, 3, v15 // (op 21121) 

s_cmp_eq_i32 s10, 0 // (op 21121) 
s_mov_b32 s47, scc // (op 21121) 
v_and_b32 v15, 15, v1 // (op 21121) 
s_cmp_eq_i32 s47, 0 // (op 21121) 
s_cselect_b32 s18, 0, -1 // (op 21121) 

v_lshl_add_u32 v27, v26, 4, v15 // (op 21121) 


s_and_b32 s19, s13, 31 // (op 21121) 
v_lshrrev_b32 v15, 6, v1 // (op 21121) 
s_lshl_b32 s47, 1, s19 // (op 21121) 
v_and_b32 v26, 1, v15 // (op 21121) 

s_add_i32 s22, s47, s18 // (op 21121) 


v_lshl_add_u32 v15, v26, 6, v27 // (op 21121) 


s_and_b32 s18, s1, s22 // (op 21121) 


v_lshrrev_b32 v26, 6, v14 // (op 21121) 

s_add_i32 s1, s0, s18 // (op 21121) 


s_ashr_i32 s0, s1, s19 // (op 21121) 


s_lshl_b32 s1, s13, 24 // (op 21121) 
s_ashr_i32 s47, s1, 31 // (op 21121) 

s_xor_b32 s1, s0, s47 // (op 21121) 

s_sub_i32 s0, s1, s47 // (op 21121) 


s_mul_hi_i32 s1, s0, s14 // (op 21121) 
s_add_i32 s47, s1, s0 // (op 21121) 

s_ashr_i32 s1, s47, 31 // (op 21121) 
s_cmp_eq_i32 s14, 0 // (op 21121) 
s_mov_b32 s18, scc // (op 21121) 
s_cmp_eq_i32 s18, 0 // (op 21121) 
s_cselect_b32 s19, 0, -1 // (op 21121) 

s_and_b32 s18, s15, 31 // (op 21121) 
s_lshl_b32 s22, 1, s18 // (op 21121) 
s_add_i32 s23, s22, s19 // (op 21121) 


s_and_b32 s19, s1, s23 // (op 21121) 


s_add_i32 s1, s47, s19 // (op 21121) 


s_ashr_i32 s19, s1, s18 // (op 21121) 


s_lshl_b32 s1, s15, 24 // (op 21121) 
s_ashr_i32 s18, s1, 31 // (op 21121) 

s_xor_b32 s1, s19, s18 // (op 21121) 

s_sub_i32 s19, s1, s18 // (op 21121) 


s_add_u32 s22, s6, 128 // (op 21121) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 21121) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 21121) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 21121) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 21121) 

s_mul_i32 s1, s19, s22 // (op 21121) 

s_sub_i32 s47, s0, s1 // (op 21121) 


s_mul_hi_i32 s0, s16, s9 // (op 21121) 
s_add_i32 s1, s0, s16 // (op 21121) 

s_ashr_i32 s0, s1, 31 // (op 21121) 
s_cmp_eq_i32 s9, 0 // (op 21121) 
s_mov_b32 s18, scc // (op 21121) 
s_cmp_eq_i32 s18, 0 // (op 21121) 
s_cselect_b32 s19, 0, -1 // (op 21121) 

s_and_b32 s18, s45, 31 // (op 21121) 
s_lshl_b32 s24, 1, s18 // (op 21121) 
s_add_i32 s25, s24, s19 // (op 21121) 


s_and_b32 s19, s0, s25 // (op 21121) 


s_add_i32 s0, s1, s19 // (op 21121) 


s_ashr_i32 s1, s0, s18 // (op 21121) 


s_lshl_b32 s0, s45, 24 // (op 21121) 
s_ashr_i32 s18, s0, 31 // (op 21121) 

s_xor_b32 s0, s1, s18 // (op 21121) 

s_sub_i32 s1, s0, s18 // (op 21121) 


s_mul_i32 s0, s11, s22 // (op 21121) 

s_mul_i32 s18, s8, s0 // (op 21121) 

s_mul_i32 s0, s1, s18 // (op 21121) 

s_sub_u32 s18, s16, s0 // (op 21121) 

s_mul_hi_i32 s0, s18, s12 // (op 21121) 
s_add_i32 s19, s0, s18 // (op 21121) 


s_ashr_i32 s0, s19, 31 // (op 21121) 
s_cmp_eq_i32 s12, 0 // (op 21121) 
s_mov_b32 s18, scc // (op 21121) 
s_cmp_eq_i32 s18, 0 // (op 21121) 
s_cselect_b32 s22, 0, -1 // (op 21121) 

s_and_b32 s18, s44, 31 // (op 21121) 
s_lshl_b32 s23, 1, s18 // (op 21121) 
s_add_i32 s24, s23, s22 // (op 21121) 


s_and_b32 s25, s0, s24 // (op 21121) 


s_add_i32 s0, s19, s25 // (op 21121) 


s_ashr_i32 s19, s0, s18 // (op 21121) 


s_lshl_b32 s0, s44, 24 // (op 21121) 
s_ashr_i32 s18, s0, 31 // (op 21121) 

s_xor_b32 s0, s19, s18 // (op 21121) 

s_sub_i32 s19, s0, s18 // (op 21121) 


s_cmp_eq_i32 s1, 0 // (op 21121) 
s_mov_b32 s0, scc // (op 21121) 

s_cmp_eq_i32 s0, 0 // (op 21121) 
s_cselect_b32 s1, s19, s47 // (op 21121) 



v_lshl_add_u32 v14, s1, 7, v15 // (op 21121) 


v_mul_lo_u32 v15, v14, s38 // (op 21121) 

v_lshl_add_u32 v14, v26, 2, v15 // (op 21121) 


s_mul_i32 s1, s38, s7 // (op 21123) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 21123) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 21123) least significant
s_mul_hi_u32 s18, s38, s6 // (op 21123) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 21123) most significant: sum
s_add_u32 s1, s1, s18 // (op 21123) most significant: sum


s_mov_b32 s68, s40 // (op 21123) reg expression
s_mov_b32 s69, s41 // (op 21123) reg expression
s_mov_b32 s70, s0 // (op 21123) reg expression
s_mov_b32 s71, 131072 // (op 21123) call()

s_mul_hi_i32 s47, s16, s10 // (op 21142) 
v_and_b32 v15, 63, v1 // (op 21142) 
s_add_i32 s0, s47, s16 // (op 21142) 

v_lshrrev_b32 v26, 4, v15 // (op 21142) 
s_ashr_i32 s1, s0, 31 // (op 21142) 
v_and_b32 v27, 3, v26 // (op 21142) 

s_cmp_eq_i32 s10, 0 // (op 21142) 
s_mov_b32 s47, scc // (op 21142) 
v_and_b32 v26, 15, v1 // (op 21142) 
s_cmp_eq_i32 s47, 0 // (op 21142) 
s_cselect_b32 s18, 0, -1 // (op 21142) 

v_lshl_add_u32 v28, v27, 4, v26 // (op 21142) 


s_and_b32 s19, s13, 31 // (op 21142) 
v_lshrrev_b32 v26, 6, v1 // (op 21142) 
s_lshl_b32 s47, 1, s19 // (op 21142) 
v_and_b32 v27, 1, v26 // (op 21142) 

s_add_i32 s22, s47, s18 // (op 21142) 


v_lshl_add_u32 v26, v27, 6, v28 // (op 21142) 


s_and_b32 s18, s1, s22 // (op 21142) 


v_lshrrev_b32 v27, 6, v15 // (op 21142) 

s_add_i32 s1, s0, s18 // (op 21142) 


s_ashr_i32 s0, s1, s19 // (op 21142) 


s_lshl_b32 s1, s13, 24 // (op 21142) 
s_ashr_i32 s47, s1, 31 // (op 21142) 

s_xor_b32 s1, s0, s47 // (op 21142) 

s_sub_i32 s0, s1, s47 // (op 21142) 


s_mul_hi_i32 s1, s0, s14 // (op 21142) 
s_add_i32 s47, s1, s0 // (op 21142) 

s_ashr_i32 s1, s47, 31 // (op 21142) 
s_cmp_eq_i32 s14, 0 // (op 21142) 
s_mov_b32 s18, scc // (op 21142) 
s_cmp_eq_i32 s18, 0 // (op 21142) 
s_cselect_b32 s19, 0, -1 // (op 21142) 

s_and_b32 s18, s15, 31 // (op 21142) 
s_lshl_b32 s22, 1, s18 // (op 21142) 
s_add_i32 s23, s22, s19 // (op 21142) 


s_and_b32 s19, s1, s23 // (op 21142) 


s_add_i32 s1, s47, s19 // (op 21142) 


s_ashr_i32 s19, s1, s18 // (op 21142) 


s_lshl_b32 s1, s15, 24 // (op 21142) 
s_ashr_i32 s18, s1, 31 // (op 21142) 

s_xor_b32 s1, s19, s18 // (op 21142) 

s_sub_i32 s19, s1, s18 // (op 21142) 


s_add_u32 s22, s6, 128 // (op 21142) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 21142) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 21142) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 21142) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 21142) 

s_mul_i32 s1, s19, s22 // (op 21142) 

s_sub_i32 s47, s0, s1 // (op 21142) 


s_mul_hi_i32 s0, s16, s9 // (op 21142) 
s_add_i32 s1, s0, s16 // (op 21142) 

s_ashr_i32 s0, s1, 31 // (op 21142) 
s_cmp_eq_i32 s9, 0 // (op 21142) 
s_mov_b32 s18, scc // (op 21142) 
s_cmp_eq_i32 s18, 0 // (op 21142) 
s_cselect_b32 s19, 0, -1 // (op 21142) 

s_and_b32 s18, s45, 31 // (op 21142) 
s_lshl_b32 s24, 1, s18 // (op 21142) 
s_add_i32 s25, s24, s19 // (op 21142) 


s_and_b32 s19, s0, s25 // (op 21142) 


s_add_i32 s0, s1, s19 // (op 21142) 


s_ashr_i32 s1, s0, s18 // (op 21142) 


s_lshl_b32 s0, s45, 24 // (op 21142) 
s_ashr_i32 s18, s0, 31 // (op 21142) 

s_xor_b32 s0, s1, s18 // (op 21142) 

s_sub_i32 s1, s0, s18 // (op 21142) 


s_mul_i32 s0, s11, s22 // (op 21142) 

s_mul_i32 s18, s8, s0 // (op 21142) 

s_mul_i32 s0, s1, s18 // (op 21142) 

s_sub_u32 s18, s16, s0 // (op 21142) 

s_mul_hi_i32 s0, s18, s12 // (op 21142) 
s_add_i32 s19, s0, s18 // (op 21142) 


s_ashr_i32 s0, s19, 31 // (op 21142) 
s_cmp_eq_i32 s12, 0 // (op 21142) 
s_mov_b32 s18, scc // (op 21142) 
s_cmp_eq_i32 s18, 0 // (op 21142) 
s_cselect_b32 s22, 0, -1 // (op 21142) 

s_and_b32 s18, s44, 31 // (op 21142) 
s_lshl_b32 s23, 1, s18 // (op 21142) 
s_add_i32 s24, s23, s22 // (op 21142) 


s_and_b32 s25, s0, s24 // (op 21142) 


s_add_i32 s0, s19, s25 // (op 21142) 


s_ashr_i32 s19, s0, s18 // (op 21142) 


s_lshl_b32 s0, s44, 24 // (op 21142) 
s_ashr_i32 s18, s0, 31 // (op 21142) 

s_xor_b32 s0, s19, s18 // (op 21142) 

s_sub_i32 s19, s0, s18 // (op 21142) 


s_cmp_eq_i32 s1, 0 // (op 21142) 
s_mov_b32 s0, scc // (op 21142) 

s_cmp_eq_i32 s0, 0 // (op 21142) 
s_cselect_b32 s1, s19, s47 // (op 21142) 



v_lshl_add_u32 v15, s1, 7, v26 // (op 21142) 


v_mul_lo_u32 v26, v15, s38 // (op 21142) 

v_lshl_add_u32 v15, v27, 2, v26 // (op 21142) 


s_mul_i32 s1, s38, s7 // (op 21144) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 21144) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 21144) least significant
s_mul_hi_u32 s18, s38, s6 // (op 21144) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 21144) most significant: sum
s_add_u32 s1, s1, s18 // (op 21144) most significant: sum


s_mov_b32 s68, s40 // (op 21144) reg expression
s_mov_b32 s69, s41 // (op 21144) reg expression
s_mov_b32 s70, s0 // (op 21144) reg expression
s_mov_b32 s71, 131072 // (op 21144) call()

v_and_b32 v26, -4, v1 // (op 20743) 
v_sub_u32 v27, v1, v26 // (op 20743) 

v_lshlrev_b32 v26, 5, v27 // (op 20743) 

v_lshrrev_b32 v27, 2, v1 // (op 20743) 
v_lshl_add_u32 v28, v27, 7, v26 // (op 20743) 


v_lshlrev_b32 v26, 2, v28 // (op 20743) 

v_lshrrev_b32 v27, 3, v26 // (op 20743) 

s_nop 0
v_readfirstlane_b32 s47, v27 // (op 20743) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 6 // (op 20442) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 20442) 

v_lshrrev_b64 v[26:27], 3, s[18:19] // (op 20442) 


v_mov_b64 v[28:29], v[20:21] // (op 20442) 
s_mov_b32 m0, s47 // (op 20442) reg expression
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 20442) Load value direct to lds
v_add_co_u32 v28, s[0:1], v28, v26 // (op 20442) least significant half
v_addc_co_u32 v29, s[0:1], 0, v29, s[0:1] // (op 20442) most significant half

s_add_u32 m0, m0, 4096 // (op 20442) 
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 20442) Load value direct to lds



v_and_b32 v26, -4, v1 // (op 20686) 
v_sub_u32 v27, v1, v26 // (op 20686) 

v_lshlrev_b32 v26, 5, v27 // (op 20686) 

v_lshrrev_b32 v27, 2, v1 // (op 20686) 
v_lshl_add_u32 v28, v27, 7, v26 // (op 20686) 


v_lshlrev_b32 v26, 2, v28 // (op 20686) 

v_lshrrev_b32 v27, 3, v26 // (op 20686) 

s_nop 0
v_readfirstlane_b32 s47, v27 // (op 20686) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 6 // (op 20447) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 20447) 

v_lshrrev_b64 v[26:27], 3, s[18:19] // (op 20447) 


v_mov_b64 v[28:29], v[16:17] // (op 20447) 
s_add_u32 s0, 8192, s47 // (op 20447) 
s_mov_b32 m0, s0 // (op 20447) reg expression
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 20447) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 20447) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 20447) most significant half

s_add_u32 m0, m0, 4096 // (op 20447) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 20447) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 20447) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 20447) most significant half

s_add_u32 m0, m0, 4096 // (op 20447) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 20447) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 20447) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 20447) most significant half

s_add_u32 m0, m0, 4096 // (op 20447) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 20447) Load value direct to lds






v_mov_b32 v26, v14 // (op 74) reg expression

buffer_load_dwordx4 v[28:31], v26, s[68:71], 0 offen // (op 74) Load value




s_lshl_b32 s47, s34, 6 // (op 4381) 
v_add_u32 v14, v24, s47 // (op 4381) 


buffer_load_dwordx4 v[32:35], v14, s[64:67], 0 offen // (op 4381) Load value



v_mov_b32 v14, v24 // (op 70) reg expression

buffer_load_dwordx4 v[36:39], v14, s[64:67], 0 offen // (op 70) Load value


v_and_b32 v14, -4, v1 // (op 20854) 
v_sub_u32 v24, v1, v14 // (op 20854) 

v_lshlrev_b32 v14, 5, v24 // (op 20854) 

v_lshrrev_b32 v24, 2, v1 // (op 20854) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 20854) 


v_lshlrev_b32 v14, 2, v26 // (op 20854) 

v_lshrrev_b32 v24, 3, v14 // (op 20854) 

s_nop 0
v_readfirstlane_b32 s47, v24 // (op 20854) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 6 // (op 20452) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 20452) 

v_lshrrev_b64 v[26:27], 3, s[18:19] // (op 20452) 


v_mov_b64 v[40:41], v[20:21] // (op 20452) 
s_add_u32 s0, 24576, s47 // (op 20452) 
s_mov_b32 m0, s0 // (op 20452) reg expression
buffer_load_dwordx4 v40, s[60:63], 64 offen offset: 0 lds // (op 20452) Load value direct to lds
v_add_co_u32 v40, s[18:19], v40, v26 // (op 20452) least significant half
v_addc_co_u32 v41, s[18:19], 0, v41, s[18:19] // (op 20452) most significant half

s_add_u32 m0, m0, 4096 // (op 20452) 
buffer_load_dwordx4 v40, s[60:63], 64 offen offset: 0 lds // (op 20452) Load value direct to lds




v_and_b32 v14, -4, v1 // (op 20810) 
v_sub_u32 v24, v1, v14 // (op 20810) 

v_lshlrev_b32 v14, 5, v24 // (op 20810) 

v_lshrrev_b32 v24, 2, v1 // (op 20810) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 20810) 


v_lshlrev_b32 v14, 2, v26 // (op 20810) 

v_lshrrev_b32 v24, 3, v14 // (op 20810) 

s_nop 0
v_readfirstlane_b32 s47, v24 // (op 20810) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 6 // (op 20454) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 20454) 

v_lshrrev_b64 v[26:27], 3, s[18:19] // (op 20454) 


v_mov_b64 v[40:41], v[16:17] // (op 20454) 
s_add_u32 s0, 32768, s47 // (op 20454) 
s_mov_b32 m0, s0 // (op 20454) reg expression
buffer_load_dwordx4 v40, s[48:51], 64 offen offset: 0 lds // (op 20454) Load value direct to lds
v_add_co_u32 v40, s[18:19], v40, v26 // (op 20454) least significant half
v_addc_co_u32 v41, s[18:19], 0, v41, s[18:19] // (op 20454) most significant half

s_add_u32 m0, m0, 4096 // (op 20454) 
buffer_load_dwordx4 v40, s[48:51], 64 offen offset: 0 lds // (op 20454) Load value direct to lds
v_add_co_u32 v40, s[18:19], v40, v26 // (op 20454) least significant half
v_addc_co_u32 v41, s[18:19], 0, v41, s[18:19] // (op 20454) most significant half

s_add_u32 m0, m0, 4096 // (op 20454) 
buffer_load_dwordx4 v40, s[48:51], 64 offen offset: 0 lds // (op 20454) Load value direct to lds
v_add_co_u32 v40, s[18:19], v40, v26 // (op 20454) least significant half
v_addc_co_u32 v41, s[18:19], 0, v41, s[18:19] // (op 20454) most significant half

s_add_u32 m0, m0, 4096 // (op 20454) 
buffer_load_dwordx4 v40, s[48:51], 64 offen offset: 0 lds // (op 20454) Load value direct to lds




v_and_b32 v14, -4, v1 // (op 20942) 
v_sub_u32 v24, v1, v14 // (op 20942) 

v_lshlrev_b32 v14, 5, v24 // (op 20942) 

v_lshrrev_b32 v24, 2, v1 // (op 20942) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 20942) 


v_lshlrev_b32 v14, 2, v26 // (op 20942) 

v_lshrrev_b32 v24, 3, v14 // (op 20942) 

s_nop 0
v_readfirstlane_b32 s47, v24 // (op 20942) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 6 // (op 20456) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 20456) 

v_lshrrev_b64 v[26:27], 3, s[18:19] // (op 20456) 


v_mov_b64 v[40:41], v[20:21] // (op 20456) 
s_add_u32 s0, 49152, s47 // (op 20456) 
s_mov_b32 m0, s0 // (op 20456) reg expression
s_mov_b32 s1, 128 // (op 20456) call()
buffer_load_dwordx4 v40, s[60:63], s1 offen offset: 0 lds // (op 20456) Load value direct to lds
v_add_co_u32 v40, s[18:19], v40, v26 // (op 20456) least significant half
v_addc_co_u32 v41, s[18:19], 0, v41, s[18:19] // (op 20456) most significant half

s_add_u32 m0, m0, 4096 // (op 20456) 
buffer_load_dwordx4 v40, s[60:63], s1 offen offset: 0 lds // (op 20456) Load value direct to lds





v_and_b32 v14, -4, v1 // (op 20898) 
v_sub_u32 v24, v1, v14 // (op 20898) 

v_lshlrev_b32 v14, 5, v24 // (op 20898) 

v_lshrrev_b32 v24, 2, v1 // (op 20898) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 20898) 


v_lshlrev_b32 v14, 2, v26 // (op 20898) 

v_lshrrev_b32 v24, 3, v14 // (op 20898) 

s_nop 0
v_readfirstlane_b32 s47, v24 // (op 20898) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 6 // (op 20458) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 20458) 

v_lshrrev_b64 v[26:27], 3, s[18:19] // (op 20458) 


v_mov_b64 v[40:41], v[16:17] // (op 20458) 
s_add_u32 s0, 57344, s47 // (op 20458) 
s_mov_b32 m0, s0 // (op 20458) reg expression
s_mov_b32 s1, 128 // (op 20458) call()
buffer_load_dwordx4 v40, s[48:51], s1 offen offset: 0 lds // (op 20458) Load value direct to lds
v_add_co_u32 v40, s[18:19], v40, v26 // (op 20458) least significant half
v_addc_co_u32 v41, s[18:19], 0, v41, s[18:19] // (op 20458) most significant half

s_add_u32 m0, m0, 4096 // (op 20458) 
buffer_load_dwordx4 v40, s[48:51], s1 offen offset: 0 lds // (op 20458) Load value direct to lds
v_add_co_u32 v40, s[18:19], v40, v26 // (op 20458) least significant half
v_addc_co_u32 v41, s[18:19], 0, v41, s[18:19] // (op 20458) most significant half

s_add_u32 m0, m0, 4096 // (op 20458) 
buffer_load_dwordx4 v40, s[48:51], s1 offen offset: 0 lds // (op 20458) Load value direct to lds
v_add_co_u32 v40, s[18:19], v40, v26 // (op 20458) least significant half
v_addc_co_u32 v41, s[18:19], 0, v41, s[18:19] // (op 20458) most significant half

s_add_u32 m0, m0, 4096 // (op 20458) 
buffer_load_dwordx4 v40, s[48:51], s1 offen offset: 0 lds // (op 20458) Load value direct to lds





v_and_b32 v14, -4, v1 // (op 21030) 
v_sub_u32 v24, v1, v14 // (op 21030) 

v_lshlrev_b32 v14, 5, v24 // (op 21030) 

v_lshrrev_b32 v24, 2, v1 // (op 21030) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 21030) 


v_lshlrev_b32 v14, 2, v26 // (op 21030) 

v_lshrrev_b32 v24, 3, v14 // (op 21030) 

s_nop 0
v_readfirstlane_b32 s47, v24 // (op 21030) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 6 // (op 20460) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 20460) 

v_lshrrev_b64 v[26:27], 3, s[18:19] // (op 20460) 


v_mov_b64 v[40:41], v[20:21] // (op 20460) 
s_add_u32 s0, 73728, s47 // (op 20460) 
s_mov_b32 m0, s0 // (op 20460) reg expression
s_mov_b32 s1, 192 // (op 20460) call()
buffer_load_dwordx4 v40, s[60:63], s1 offen offset: 0 lds // (op 20460) Load value direct to lds
v_add_co_u32 v40, s[18:19], v40, v26 // (op 20460) least significant half
v_addc_co_u32 v41, s[18:19], 0, v41, s[18:19] // (op 20460) most significant half

s_add_u32 m0, m0, 4096 // (op 20460) 
buffer_load_dwordx4 v40, s[60:63], s1 offen offset: 0 lds // (op 20460) Load value direct to lds






v_and_b32 v14, -4, v1 // (op 20986) 
v_sub_u32 v24, v1, v14 // (op 20986) 

v_lshlrev_b32 v14, 5, v24 // (op 20986) 

v_lshrrev_b32 v24, 2, v1 // (op 20986) 
v_lshl_add_u32 v20, v24, 7, v14 // (op 20986) 


v_lshlrev_b32 v14, 2, v20 // (op 20986) 

v_lshrrev_b32 v24, 3, v14 // (op 20986) 

s_nop 0
v_readfirstlane_b32 s47, v24 // (op 20986) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 6 // (op 20462) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 20462) 

v_lshrrev_b64 v[20:21], 3, s[18:19] // (op 20462) 


v_mov_b64 v[26:27], v[16:17] // (op 20462) 
s_add_u32 s0, 81920, s47 // (op 20462) 
s_mov_b32 m0, s0 // (op 20462) reg expression
s_mov_b32 s1, 192 // (op 20462) call()
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 20462) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v20 // (op 20462) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 20462) most significant half

s_add_u32 m0, m0, 4096 // (op 20462) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 20462) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v20 // (op 20462) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 20462) most significant half

s_add_u32 m0, m0, 4096 // (op 20462) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 20462) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v20 // (op 20462) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 20462) most significant half

s_add_u32 m0, m0, 4096 // (op 20462) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 20462) Load value direct to lds






s_waitcnt vmcnt(25)
s_barrier  // (op 21587) 



ds_read_b128 v[40:43], v4 // (op 19770) Load local data 



ds_read_b128 v[44:47], v4 offset:1024 // (op 19775) Load local data 



ds_read_b128 v[48:51], v4 offset:2048 // (op 19780) Load local data 



ds_read_b128 v[52:55], v4 offset:3072 // (op 19785) Load local data 
s_waitcnt vmcnt(21)
s_barrier  // (op 21571) 



ds_read_b128 v[56:59], v0 offset:8192 // (op 19730) Load local data 



ds_read_b128 v[60:63], v0 offset:9216 // (op 19735) Load local data 



ds_read_b128 v[64:67], v0 offset:10240 // (op 19740) Load local data 



ds_read_b128 v[68:71], v0 offset:11264 // (op 19745) Load local data 



ds_read_b128 v[72:75], v0 offset:12288 // (op 19750) Load local data 



ds_read_b128 v[76:79], v0 offset:13312 // (op 19755) Load local data 



ds_read_b128 v[80:83], v0 offset:14336 // (op 19760) Load local data 



ds_read_b128 v[84:87], v0 offset:15360 // (op 19765) Load local data 
v_and_b32 v14, -4, v1 // (op 20658) 
v_sub_u32 v24, v1, v14 // (op 20658) 

v_lshlrev_b32 v14, 5, v24 // (op 20658) 

v_lshrrev_b32 v24, 2, v1 // (op 20658) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 20658) 


v_lshlrev_b32 v14, 2, v16 // (op 20658) 

v_lshrrev_b32 v24, 3, v14 // (op 20658) 

s_nop 0
v_readfirstlane_b32 s47, v24 // (op 20658) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 20715) 
v_sub_u32 v24, v1, v14 // (op 20715) 

v_lshlrev_b32 v14, 5, v24 // (op 20715) 

v_lshrrev_b32 v24, 2, v1 // (op 20715) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 20715) 


v_lshlrev_b32 v14, 2, v16 // (op 20715) 

v_lshrrev_b32 v24, 3, v14 // (op 20715) 

s_nop 0
v_readfirstlane_b32 s0, v24 // (op 20715) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 20795) 
v_sub_u32 v24, v1, v14 // (op 20795) 

v_lshlrev_b32 v14, 5, v24 // (op 20795) 

v_lshrrev_b32 v24, 2, v1 // (op 20795) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 20795) 


v_lshlrev_b32 v14, 2, v16 // (op 20795) 

v_lshrrev_b32 v24, 3, v14 // (op 20795) 

s_nop 0
v_readfirstlane_b32 s1, v24 // (op 20795) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 20839) 
v_sub_u32 v24, v1, v14 // (op 20839) 

v_lshlrev_b32 v14, 5, v24 // (op 20839) 

v_lshrrev_b32 v24, 2, v1 // (op 20839) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 20839) 


v_lshlrev_b32 v14, 2, v16 // (op 20839) 

v_lshrrev_b32 v24, 3, v14 // (op 20839) 

s_nop 0
v_readfirstlane_b32 s18, v24 // (op 20839) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 20883) 
v_sub_u32 v24, v1, v14 // (op 20883) 

v_lshlrev_b32 v14, 5, v24 // (op 20883) 

v_lshrrev_b32 v24, 2, v1 // (op 20883) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 20883) 


v_lshlrev_b32 v14, 2, v16 // (op 20883) 

v_lshrrev_b32 v24, 3, v14 // (op 20883) 

s_nop 0
v_readfirstlane_b32 s19, v24 // (op 20883) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 20927) 
v_sub_u32 v24, v1, v14 // (op 20927) 

v_lshlrev_b32 v14, 5, v24 // (op 20927) 

v_lshrrev_b32 v24, 2, v1 // (op 20927) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 20927) 


v_lshlrev_b32 v14, 2, v16 // (op 20927) 

v_lshrrev_b32 v24, 3, v14 // (op 20927) 

s_nop 0
v_readfirstlane_b32 s22, v24 // (op 20927) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 20971) 
v_sub_u32 v24, v1, v14 // (op 20971) 

v_lshlrev_b32 v14, 5, v24 // (op 20971) 

v_lshrrev_b32 v24, 2, v1 // (op 20971) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 20971) 


v_lshlrev_b32 v14, 2, v16 // (op 20971) 

v_lshrrev_b32 v24, 3, v14 // (op 20971) 

s_nop 0
v_readfirstlane_b32 s23, v24 // (op 20971) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 21015) 
v_sub_u32 v24, v1, v14 // (op 21015) 

v_lshlrev_b32 v14, 5, v24 // (op 21015) 

v_lshrrev_b32 v24, 2, v1 // (op 21015) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 21015) 


v_lshlrev_b32 v14, 2, v16 // (op 21015) 

v_lshrrev_b32 v24, 3, v14 // (op 21015) 

s_nop 0
v_readfirstlane_b32 s24, v24 // (op 21015) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s25, 0 // (op 6290) call()
s_ashr_i64 s[72:73], s[42:43], 7 // (op 6289) 
s_mov_b32 s74, s72 // (op 6289) convert

s_and_b32 s72, s74, 4294967292 // (op 6289) 

s_cmp_lt_u32 s25, s72 // (op 6289) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_WGM__ForLoopBottom_KLoop_6289 // (op 6289) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_WGM__ForLoopBottom_KLoop_6289 if false)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_WGM__ForLoopTop_KLoop_6289:
s_add_u32 s25, s25, 4 // (op 6291) 
s_waitcnt vmcnt(20)
v_mov_b32 v88, v28 // (op 20321) reg expression
v_mov_b32 v89, v29 // (op 20321) reg expression
v_mov_b32 v90, v30 // (op 20321) reg expression
v_mov_b32 v91, v31 // (op 20321) reg expression
s_waitcnt lgkmcnt(8)
s_barrier  // (op 21590) 
s_waitcnt vmcnt(19)
v_mov_b32 v92, v32 // (op 20334) reg expression

v_mov_b32 v93, v33 // (op 20334) reg expression
v_mov_b32 v94, v34 // (op 20334) reg expression
v_mov_b32 v95, v35 // (op 20334) reg expression
s_lshl_b64 s[72:73], s[30:31], 6 // (op 20395) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 20395) 

v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 20395) 


v_mov_b64 v[20:21], v[22:23] // (op 20395) 
v_mov_b64 v[26:27], v[20:21] // (op 20395) 
s_mov_b32 m0, s0 // (op 20395) reg expression
s_mov_b32 s72, 256 // (op 20395) call()
s_waitcnt vmcnt(18)
v_mov_b32 v96, v36 // (op 20347) reg expression
v_mov_b32 v97, v37 // (op 20347) reg expression
v_mov_b32 v98, v38 // (op 20347) reg expression
v_mov_b32 v99, v39 // (op 20347) reg expression
buffer_load_dwordx4 v26, s[60:63], s72 offen offset: 0 lds // (op 20395) Load value direct to lds
v_add_co_u32 v26, s[74:75], v26, v16 // (op 20395) least significant half
v_addc_co_u32 v27, s[74:75], 0, v27, s[74:75] // (op 20395) most significant half

s_add_u32 m0, m0, 4096 // (op 20395) 
buffer_load_dwordx4 v26, s[60:63], s72 offen offset: 0 lds // (op 20395) Load value direct to lds



v_bfe_u32 v100, v96, 0, 8 // (op 18568) 
v_bfe_u32 v101, v96, 8, 8 // (op 18568) 
v_bfe_u32 v102, v96, 16, 8 // (op 18568) 
v_bfe_u32 v103, v96, 24, 8 // (op 18568) 
v_permlane16_swap_b32 v100, v101 // (op 18568) 
s_nop 0
v_permlane16_swap_b32 v102, v103 // (op 18568) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v100, v102 // (op 18568) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v101, v103 // (op 18568) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 21574) 

s_lshl_b64 s[72:73], s[26:27], 6 // (op 20390) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 20390) 

v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 20390) 


v_mov_b64 v[26:27], v[18:19] // (op 20390) 
v_mov_b64 v[104:105], v[26:27] // (op 20390) 
s_add_u32 s72, 8192, s47 // (op 20390) 
s_mov_b32 m0, s72 // (op 20390) reg expression
s_mov_b32 s73, 256 // (op 20390) call()
buffer_load_dwordx4 v104, s[48:51], s73 offen offset: 0 lds // (op 20390) Load value direct to lds
v_add_co_u32 v104, s[74:75], v104, v16 // (op 20390) least significant half
v_addc_co_u32 v105, s[74:75], 0, v105, s[74:75] // (op 20390) most significant half

s_add_u32 m0, m0, 4096 // (op 20390) 
v_bfe_u32 v106, v88, 0, 8 // (op 18829) 
v_bfe_u32 v107, v88, 8, 8 // (op 18829) 
v_bfe_u32 v108, v88, 16, 8 // (op 18829) 
v_bfe_u32 v109, v88, 24, 8 // (op 18829) 
v_permlane16_swap_b32 v106, v107 // (op 18829) 
s_nop 0
v_permlane16_swap_b32 v108, v109 // (op 18829) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v106, v108 // (op 18829) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v107, v109 // (op 18829) 
buffer_load_dwordx4 v104, s[48:51], s73 offen offset: 0 lds // (op 20390) Load value direct to lds
v_add_co_u32 v104, s[74:75], v104, v16 // (op 20390) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:59], v[40:43], a[0:3], v100, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 76) 
v_addc_co_u32 v105, s[74:75], 0, v105, s[74:75] // (op 20390) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:59], v[44:47], a[4:7], v100, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 417) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[56:59], v[48:51], a[8:11], v100, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 643) 
s_add_u32 m0, m0, 4096 // (op 20390) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[56:59], v[52:55], a[12:15], v100, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 869) 
buffer_load_dwordx4 v104, s[48:51], s73 offen offset: 0 lds // (op 20390) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[60:63], v[40:43], a[16:19], v101, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1541) 
v_add_co_u32 v104, s[74:75], v104, v16 // (op 20390) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[60:63], v[44:47], a[20:23], v101, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1465) 
v_addc_co_u32 v105, s[74:75], 0, v105, s[74:75] // (op 20390) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[60:63], v[48:51], a[24:27], v101, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1389) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[60:63], v[52:55], a[28:31], v101, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1313) 
s_add_u32 m0, m0, 4096 // (op 20390) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[64:67], v[40:43], a[32:35], v102, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2543) 
buffer_load_dwordx4 v104, s[48:51], s73 offen offset: 0 lds // (op 20390) Load value direct to lds






v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[64:67], v[44:47], a[36:39], v102, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2467) 
v_mov_b32 v14, v15 // (op 20329) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[64:67], v[48:51], a[40:43], v102, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2391) 
v_add_u32 v24, 16, v14 // (op 20329) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[64:67], v[52:55], a[44:47], v102, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2315) 
v_bfe_u32 v110, v92, 0, 8 // (op 18633) 
v_bfe_u32 v111, v92, 8, 8 // (op 18633) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[68:71], v[40:43], a[48:51], v103, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3545) 
v_bfe_u32 v112, v92, 16, 8 // (op 18633) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[68:71], v[44:47], a[52:55], v103, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3469) 
v_bfe_u32 v113, v92, 24, 8 // (op 18633) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[68:71], v[48:51], a[56:59], v103, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3393) 
v_permlane16_swap_b32 v110, v111 // (op 18633) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[68:71], v[52:55], a[60:63], v103, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3317) 
v_permlane16_swap_b32 v112, v113 // (op 18633) 

s_nop 1
v_permlane32_swap_b32 v110, v112 // (op 18633) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v111, v113 // (op 18633) 
s_waitcnt vmcnt(22)
s_barrier  // (op 21609) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[72:75], v[40:43], a[64:67], v110, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4547) 
ds_read_b128 v[100:103], v6 offset:24576 // (op 212) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[72:75], v[44:47], a[68:71], v110, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4471) 
ds_read_b128 v[114:117], v6 offset:25600 // (op 428) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[72:75], v[48:51], a[72:75], v110, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4395) 
ds_read_b128 v[118:121], v6 offset:26624 // (op 654) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[72:75], v[52:55], a[76:79], v110, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4319) 
ds_read_b128 v[122:125], v6 offset:27648 // (op 880) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[76:79], v[40:43], a[80:83], v111, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5549) 
s_waitcnt vmcnt(18)
s_barrier  // (op 21603) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[76:79], v[44:47], a[84:87], v111, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5473) 
ds_read_b128 v[56:59], v3 offset:32768 // (op 210) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[76:79], v[48:51], a[88:91], v111, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5397) 
ds_read_b128 v[60:63], v3 offset:33792 // (op 1323) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[76:79], v[52:55], a[92:95], v111, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5321) 
ds_read_b128 v[64:67], v3 offset:34816 // (op 2325) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[80:83], v[40:43], a[96:99], v112, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6551) 
ds_read_b128 v[68:71], v3 offset:35840 // (op 3327) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[80:83], v[44:47], a[100:103], v112, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6475) 
ds_read_b128 v[72:75], v3 offset:36864 // (op 4329) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[80:83], v[48:51], a[104:107], v112, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6399) 
ds_read_b128 v[76:79], v3 offset:37888 // (op 5331) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[80:83], v[52:55], a[108:111], v112, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6323) 
buffer_load_dwordx4 v[28:31], v24, s[68:71], 0 offen // (op 20329) Load value


v_add_u32 v15, 16, v15 // (op 21132) 





v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[84:87], v[40:43], a[112:115], v113, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7553) 
ds_read_b128 v[80:83], v3 offset:38912 // (op 6333) Load local data 
v_mov_b32 v14, v25 // (op 20342) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[84:87], v[44:47], a[116:119], v113, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7477) 
s_lshl_b32 s72, s34, 6 // (op 20342) 
s_add_u32 s73, 16, s72 // (op 20342) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[84:87], v[48:51], a[120:123], v113, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7401) 
v_add_u32 v24, v14, s73 // (op 20342) 


ds_read_b128 v[126:129], v3 offset:39936 // (op 7335) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[84:87], v[52:55], a[124:127], v113, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7325) 
buffer_load_dwordx4 v[32:35], v24, s[64:67], 0 offen // (op 20342) Load value





v_add_u32 v24, 16, v14 // (op 20355) 

v_bfe_u32 v104, v97, 0, 8 // (op 18583) 
v_bfe_u32 v105, v97, 8, 8 // (op 18583) 
v_bfe_u32 v106, v97, 16, 8 // (op 18583) 
v_bfe_u32 v107, v97, 24, 8 // (op 18583) 
v_permlane16_swap_b32 v104, v105 // (op 18583) 
s_nop 0
v_permlane16_swap_b32 v106, v107 // (op 18583) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v104, v106 // (op 18583) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v105, v107 // (op 18583) 
buffer_load_dwordx4 v[36:39], v24, s[64:67], 0 offen // (op 20355) Load value


v_add_u32 v25, 16, v25 // (op 21080) 
v_bfe_u32 v108, v89, 0, 8 // (op 18844) 
v_bfe_u32 v109, v89, 8, 8 // (op 18844) 
v_bfe_u32 v110, v89, 16, 8 // (op 18844) 
v_bfe_u32 v111, v89, 24, 8 // (op 18844) 
v_permlane16_swap_b32 v108, v109 // (op 18844) 
s_nop 0
v_permlane16_swap_b32 v110, v111 // (op 18844) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v108, v110 // (op 18844) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v109, v111 // (op 18844) 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 21613) 

s_waitcnt lgkmcnt(7)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:59], v[100:103], a[0:3], v104, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 217) 
s_lshl_b64 s[72:73], s[30:31], 6 // (op 20417) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:59], v[114:117], a[4:7], v104, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 418) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 20417) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[56:59], v[118:121], a[8:11], v104, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 644) 
v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 20417) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[56:59], v[122:125], a[12:15], v104, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 870) 
v_mov_b64 v[112:113], v[20:21] // (op 20417) 
s_add_u32 s72, 24576, s18 // (op 20417) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[60:63], v[100:103], a[16:19], v105, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1542) 
s_mov_b32 m0, s72 // (op 20417) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[60:63], v[114:117], a[20:23], v105, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1466) 
s_mov_b32 s73, 320 // (op 20417) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[60:63], v[118:121], a[24:27], v105, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1390) 
v_bfe_u32 v130, v93, 0, 8 // (op 18648) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[60:63], v[122:125], a[28:31], v105, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1314) 
v_bfe_u32 v131, v93, 8, 8 // (op 18648) 
v_bfe_u32 v132, v93, 16, 8 // (op 18648) 
v_bfe_u32 v133, v93, 24, 8 // (op 18648) 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[64:67], v[100:103], a[32:35], v106, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2544) 
v_permlane16_swap_b32 v130, v131 // (op 18648) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[64:67], v[114:117], a[36:39], v106, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2468) 
v_permlane16_swap_b32 v132, v133 // (op 18648) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[64:67], v[118:121], a[40:43], v106, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2392) 
s_nop 0
v_permlane32_swap_b32 v130, v132 // (op 18648) Wait state hazard: VALU Write Readlane Hazard
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[64:67], v[122:125], a[44:47], v106, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2316) 
v_permlane32_swap_b32 v131, v133 // (op 18648) 
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[68:71], v[100:103], a[48:51], v107, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3546) 
s_waitcnt vmcnt(19)
s_barrier  // (op 21623) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[68:71], v[114:117], a[52:55], v107, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3470) 
ds_read_b128 v[40:43], v8 offset:49152 // (op 240) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[68:71], v[118:121], a[56:59], v107, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3394) 
ds_read_b128 v[44:47], v8 offset:50176 // (op 438) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[68:71], v[122:125], a[60:63], v107, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3318) 
ds_read_b128 v[48:51], v8 offset:51200 // (op 664) Load local data 




s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[72:75], v[100:103], a[64:67], v130, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4548) 
ds_read_b128 v[52:55], v8 offset:52224 // (op 890) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[72:75], v[114:117], a[68:71], v130, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4472) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[72:75], v[118:121], a[72:75], v130, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4396) 
s_waitcnt vmcnt(15)
s_barrier  // (op 21617) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[72:75], v[122:125], a[76:79], v130, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4320) 
ds_read_b128 v[56:59], v5 offset:57344 // (op 238) Load local data 



s_waitcnt lgkmcnt(7)
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[76:79], v[100:103], a[80:83], v131, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5550) 
ds_read_b128 v[60:63], v5 offset:58368 // (op 1333) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[76:79], v[114:117], a[84:87], v131, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5474) 
ds_read_b128 v[64:67], v5 offset:59392 // (op 2335) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[76:79], v[118:121], a[88:91], v131, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5398) 
ds_read_b128 v[68:71], v5 offset:60416 // (op 3337) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[76:79], v[122:125], a[92:95], v131, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5322) 
ds_read_b128 v[72:75], v5 offset:61440 // (op 4339) Load local data 
buffer_load_dwordx4 v112, s[60:63], s73 offen offset: 0 lds // (op 20417) Load value direct to lds



s_waitcnt lgkmcnt(10)
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[80:83], v[100:103], a[96:99], v132, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6552) 
ds_read_b128 v[76:79], v5 offset:62464 // (op 5341) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[80:83], v[114:117], a[100:103], v132, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6476) 
v_add_co_u32 v112, s[74:75], v112, v16 // (op 20417) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[80:83], v[118:121], a[104:107], v132, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6400) 
v_addc_co_u32 v113, s[74:75], 0, v113, s[74:75] // (op 20417) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[80:83], v[122:125], a[108:111], v132, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6324) 
s_add_u32 m0, m0, 4096 // (op 20417) 



s_waitcnt lgkmcnt(10)
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[126:129], v[100:103], a[112:115], v133, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7554) 

ds_read_b128 v[80:83], v5 offset:63488 // (op 6343) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[126:129], v[114:117], a[116:119], v133, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7478) 
ds_read_b128 v[84:87], v5 offset:64512 // (op 7345) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[126:129], v[118:121], a[120:123], v133, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7402) 
buffer_load_dwordx4 v112, s[60:63], s73 offen offset: 0 lds // (op 20417) Load value direct to lds





s_barrier  // (op 21606) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[126:129], v[122:125], a[124:127], v133, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7326) 





s_lshl_b64 s[72:73], s[26:27], 6 // (op 20412) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 20412) 

v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 20412) 


v_mov_b64 v[100:101], v[26:27] // (op 20412) 
s_add_u32 s72, 32768, s1 // (op 20412) 
s_mov_b32 m0, s72 // (op 20412) reg expression
s_mov_b32 s73, 320 // (op 20412) call()
buffer_load_dwordx4 v100, s[48:51], s73 offen offset: 0 lds // (op 20412) Load value direct to lds
v_add_co_u32 v100, s[74:75], v100, v16 // (op 20412) least significant half
v_addc_co_u32 v101, s[74:75], 0, v101, s[74:75] // (op 20412) most significant half

s_add_u32 m0, m0, 4096 // (op 20412) 
buffer_load_dwordx4 v100, s[48:51], s73 offen offset: 0 lds // (op 20412) Load value direct to lds
v_add_co_u32 v100, s[74:75], v100, v16 // (op 20412) least significant half
v_addc_co_u32 v101, s[74:75], 0, v101, s[74:75] // (op 20412) most significant half

s_add_u32 m0, m0, 4096 // (op 20412) 
buffer_load_dwordx4 v100, s[48:51], s73 offen offset: 0 lds // (op 20412) Load value direct to lds
v_add_co_u32 v100, s[74:75], v100, v16 // (op 20412) least significant half
v_addc_co_u32 v101, s[74:75], 0, v101, s[74:75] // (op 20412) most significant half

s_add_u32 m0, m0, 4096 // (op 20412) 
buffer_load_dwordx4 v100, s[48:51], s73 offen offset: 0 lds // (op 20412) Load value direct to lds




v_bfe_u32 v100, v98, 0, 8 // (op 18600) 
v_bfe_u32 v101, v98, 8, 8 // (op 18600) 
v_bfe_u32 v102, v98, 16, 8 // (op 18600) 
v_bfe_u32 v103, v98, 24, 8 // (op 18600) 
v_permlane16_swap_b32 v100, v101 // (op 18600) 
s_nop 0
v_permlane16_swap_b32 v102, v103 // (op 18600) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v100, v102 // (op 18600) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v101, v103 // (op 18600) 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 21626) 

s_lshl_b64 s[72:73], s[30:31], 6 // (op 20427) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 20427) 

v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 20427) 


v_mov_b64 v[104:105], v[20:21] // (op 20427) 
s_add_u32 s72, 49152, s22 // (op 20427) 
s_mov_b32 m0, s72 // (op 20427) reg expression
s_mov_b32 s73, 384 // (op 20427) call()
v_bfe_u32 v106, v90, 0, 8 // (op 18862) 
v_bfe_u32 v107, v90, 8, 8 // (op 18862) 
v_bfe_u32 v108, v90, 16, 8 // (op 18862) 
v_bfe_u32 v109, v90, 24, 8 // (op 18862) 
v_permlane16_swap_b32 v106, v107 // (op 18862) 
s_nop 0
v_permlane16_swap_b32 v108, v109 // (op 18862) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v106, v108 // (op 18862) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v107, v109 // (op 18862) 
buffer_load_dwordx4 v104, s[60:63], s73 offen offset: 0 lds // (op 20427) Load value direct to lds
v_add_co_u32 v104, s[74:75], v104, v16 // (op 20427) least significant half
s_waitcnt lgkmcnt(7)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:59], v[40:43], a[0:3], v100, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 245) 
v_addc_co_u32 v105, s[74:75], 0, v105, s[74:75] // (op 20427) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:59], v[44:47], a[4:7], v100, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 419) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[56:59], v[48:51], a[8:11], v100, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 645) 
s_add_u32 m0, m0, 4096 // (op 20427) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[56:59], v[52:55], a[12:15], v100, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 871) 
v_bfe_u32 v110, v94, 0, 8 // (op 18666) 
v_bfe_u32 v111, v94, 8, 8 // (op 18666) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[60:63], v[40:43], a[16:19], v101, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1543) 
v_bfe_u32 v112, v94, 16, 8 // (op 18666) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[60:63], v[44:47], a[20:23], v101, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1467) 
v_bfe_u32 v113, v94, 24, 8 // (op 18666) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[60:63], v[48:51], a[24:27], v101, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1391) 
buffer_load_dwordx4 v104, s[60:63], s73 offen offset: 0 lds // (op 20427) Load value direct to lds




v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[60:63], v[52:55], a[28:31], v101, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1315) 
v_permlane16_swap_b32 v110, v111 // (op 18666) 
v_permlane16_swap_b32 v112, v113 // (op 18666) 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[64:67], v[40:43], a[32:35], v102, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2545) 
s_nop 0
v_permlane32_swap_b32 v110, v112 // (op 18666) Wait state hazard: VALU Write Readlane Hazard
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[64:67], v[44:47], a[36:39], v102, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2469) 
v_permlane32_swap_b32 v111, v113 // (op 18666) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[64:67], v[48:51], a[40:43], v102, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2393) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 21620) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[64:67], v[52:55], a[44:47], v102, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2317) 
s_lshl_b64 s[72:73], s[26:27], 6 // (op 20422) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 20422) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[68:71], v[40:43], a[48:51], v103, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3547) 
v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 20422) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[68:71], v[44:47], a[52:55], v103, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3471) 
v_mov_b64 v[104:105], v[26:27] // (op 20422) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[68:71], v[48:51], a[56:59], v103, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3395) 
s_add_u32 s72, 57344, s19 // (op 20422) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[68:71], v[52:55], a[60:63], v103, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3319) 
s_mov_b32 m0, s72 // (op 20422) reg expression

s_mov_b32 s73, 384 // (op 20422) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[72:75], v[40:43], a[64:67], v110, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4549) 
buffer_load_dwordx4 v104, s[48:51], s73 offen offset: 0 lds // (op 20422) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[72:75], v[44:47], a[68:71], v110, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4473) 
v_add_co_u32 v104, s[74:75], v104, v16 // (op 20422) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[72:75], v[48:51], a[72:75], v110, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4397) 
v_addc_co_u32 v105, s[74:75], 0, v105, s[74:75] // (op 20422) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[72:75], v[52:55], a[76:79], v110, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4321) 
s_add_u32 m0, m0, 4096 // (op 20422) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[76:79], v[40:43], a[80:83], v111, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5551) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[76:79], v[44:47], a[84:87], v111, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5475) 
s_waitcnt vmcnt(22)
s_barrier  // (op 21635) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[76:79], v[48:51], a[88:91], v111, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5399) 
v_add_u32 v14, 73728, v10 // (op 268) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[76:79], v[52:55], a[92:95], v111, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5323) 
ds_read_b128 v[100:103], v14 // (op 268) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[80:83], v[40:43], a[96:99], v112, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6553) 
v_add_u32 v14, 74752, v10 // (op 448) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[80:83], v[44:47], a[100:103], v112, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6477) 
ds_read_b128 v[114:117], v14 // (op 448) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[80:83], v[48:51], a[104:107], v112, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6401) 
v_add_u32 v14, 75776, v10 // (op 674) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[80:83], v[52:55], a[108:111], v112, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6325) 
ds_read_b128 v[118:121], v14 // (op 674) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[84:87], v[40:43], a[112:115], v113, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7555) 
v_add_u32 v14, 76800, v10 // (op 900) 
ds_read_b128 v[122:125], v14 // (op 900) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[84:87], v[44:47], a[116:119], v113, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7479) 
s_waitcnt vmcnt(18)
s_barrier  // (op 21629) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[84:87], v[48:51], a[120:123], v113, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7403) 
v_add_u32 v14, 81920, v7 // (op 266) 
ds_read_b128 v[56:59], v14 // (op 266) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[84:87], v[52:55], a[124:127], v113, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7327) 
v_add_u32 v14, 82944, v7 // (op 1343) 


ds_read_b128 v[60:63], v14 // (op 1343) Load local data 

buffer_load_dwordx4 v104, s[48:51], s73 offen offset: 0 lds // (op 20422) Load value direct to lds



v_add_u32 v14, 83968, v7 // (op 2345) 
ds_read_b128 v[64:67], v14 // (op 2345) Load local data 

v_add_co_u32 v104, s[74:75], v104, v16 // (op 20422) least significant half



v_addc_co_u32 v105, s[74:75], 0, v105, s[74:75] // (op 20422) most significant half

s_add_u32 m0, m0, 4096 // (op 20422) 
v_add_u32 v14, 84992, v7 // (op 3347) 
ds_read_b128 v[68:71], v14 // (op 3347) Load local data 

buffer_load_dwordx4 v104, s[48:51], s73 offen offset: 0 lds // (op 20422) Load value direct to lds



v_add_u32 v14, 86016, v7 // (op 4349) 
ds_read_b128 v[72:75], v14 // (op 4349) Load local data 

v_add_co_u32 v104, s[74:75], v104, v16 // (op 20422) least significant half



v_addc_co_u32 v105, s[74:75], 0, v105, s[74:75] // (op 20422) most significant half

s_add_u32 m0, m0, 4096 // (op 20422) 
v_add_u32 v14, 87040, v7 // (op 5351) 
buffer_load_dwordx4 v104, s[48:51], s73 offen offset: 0 lds // (op 20422) Load value direct to lds




ds_read_b128 v[76:79], v14 // (op 5351) Load local data 




v_add_u32 v14, 88064, v7 // (op 6353) 
ds_read_b128 v[80:83], v14 // (op 6353) Load local data 




v_add_u32 v14, 89088, v7 // (op 7355) 
ds_read_b128 v[104:107], v14 // (op 7355) Load local data 

v_bfe_u32 v108, v99, 0, 8 // (op 18617) 
v_bfe_u32 v109, v99, 8, 8 // (op 18617) 
v_bfe_u32 v110, v99, 16, 8 // (op 18617) 
v_bfe_u32 v111, v99, 24, 8 // (op 18617) 
v_permlane16_swap_b32 v108, v109 // (op 18617) 
s_nop 0
v_permlane16_swap_b32 v110, v111 // (op 18617) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v108, v110 // (op 18617) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v109, v111 // (op 18617) 
s_waitcnt lgkmcnt(8)
s_barrier  // (op 21638) 


s_lshl_b64 s[72:73], s[30:31], 6 // (op 20437) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 20437) 

v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 20437) 


v_mov_b64 v[112:113], v[20:21] // (op 20437) 
s_add_u32 s72, 73728, s24 // (op 20437) 
s_mov_b32 m0, s72 // (op 20437) reg expression
s_mov_b32 s73, 448 // (op 20437) call()
v_bfe_u32 v96, v91, 0, 8 // (op 18880) 
v_bfe_u32 v97, v91, 8, 8 // (op 18880) 
v_bfe_u32 v98, v91, 16, 8 // (op 18880) 
v_bfe_u32 v99, v91, 24, 8 // (op 18880) 
v_permlane16_swap_b32 v96, v97 // (op 18880) 
s_nop 0
v_permlane16_swap_b32 v98, v99 // (op 18880) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v96, v98 // (op 18880) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v97, v99 // (op 18880) 
buffer_load_dwordx4 v112, s[60:63], s73 offen offset: 0 lds // (op 20437) Load value direct to lds

v_add_co_u32 v112, s[74:75], v112, v16 // (op 20437) least significant half
s_waitcnt lgkmcnt(7)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:59], v[100:103], a[0:3], v108, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 273) 
v_addc_co_u32 v113, s[74:75], 0, v113, s[74:75] // (op 20437) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:59], v[114:117], a[4:7], v108, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 420) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[56:59], v[118:121], a[8:11], v108, v98 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 646) 
s_add_u32 m0, m0, 4096 // (op 20437) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[56:59], v[122:125], a[12:15], v108, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 872) 
v_bfe_u32 v88, v95, 0, 8 // (op 18684) 
v_bfe_u32 v89, v95, 8, 8 // (op 18684) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[60:63], v[100:103], a[16:19], v109, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1544) 
v_bfe_u32 v90, v95, 16, 8 // (op 18684) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[60:63], v[114:117], a[20:23], v109, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1468) 
v_bfe_u32 v91, v95, 24, 8 // (op 18684) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[60:63], v[118:121], a[24:27], v109, v98 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1392) 
buffer_load_dwordx4 v112, s[60:63], s73 offen offset: 0 lds // (op 20437) Load value direct to lds





v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[60:63], v[122:125], a[28:31], v109, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1316) 
v_permlane16_swap_b32 v88, v89 // (op 18684) 
v_permlane16_swap_b32 v90, v91 // (op 18684) 
v_add_co_u32 v22, vcc, 256, v22 // (op 20607) least significant half
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[64:67], v[100:103], a[32:35], v110, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2546) 
v_permlane32_swap_b32 v88, v90 // (op 18684) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[64:67], v[114:117], a[36:39], v110, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2470) 
v_permlane32_swap_b32 v89, v91 // (op 18684) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[64:67], v[118:121], a[40:43], v110, v98 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2394) 
v_addc_co_u32 v23, vcc, 0, v23, vcc // (op 20607) most significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[64:67], v[122:125], a[44:47], v110, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2318) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 21632) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[68:71], v[100:103], a[48:51], v111, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3548) 
s_lshl_b64 s[72:73], s[26:27], 6 // (op 20432) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[68:71], v[114:117], a[52:55], v111, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3472) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 20432) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[68:71], v[118:121], a[56:59], v111, v98 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3396) 
v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 20432) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[68:71], v[122:125], a[60:63], v111, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3320) 
v_mov_b64 v[20:21], v[26:27] // (op 20432) 

s_add_u32 s72, 81920, s23 // (op 20432) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[72:75], v[100:103], a[64:67], v88, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4550) 
s_mov_b32 m0, s72 // (op 20432) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[72:75], v[114:117], a[68:71], v88, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4474) 
s_mov_b32 s73, 448 // (op 20432) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[72:75], v[118:121], a[72:75], v88, v98 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4398) 
buffer_load_dwordx4 v20, s[48:51], s73 offen offset: 0 lds // (op 20432) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[72:75], v[122:125], a[76:79], v88, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4322) 
v_add_co_u32 v20, s[74:75], v20, v16 // (op 20432) least significant half
v_addc_co_u32 v21, s[74:75], 0, v21, s[74:75] // (op 20432) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[76:79], v[100:103], a[80:83], v89, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5552) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[76:79], v[114:117], a[84:87], v89, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5476) 
s_add_u32 m0, m0, 4096 // (op 20432) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[76:79], v[118:121], a[88:91], v89, v98 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5400) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[76:79], v[122:125], a[92:95], v89, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5324) 
s_waitcnt vmcnt(22)
s_barrier  // (op 21596) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[80:83], v[100:103], a[96:99], v90, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6554) 
ds_read_b128 v[40:43], v4 // (op 65) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[80:83], v[114:117], a[100:103], v90, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6478) 
ds_read_b128 v[44:47], v4 offset:1024 // (op 414) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[80:83], v[118:121], a[104:107], v90, v98 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6402) 
ds_read_b128 v[48:51], v4 offset:2048 // (op 640) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[80:83], v[122:125], a[108:111], v90, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6326) 
ds_read_b128 v[52:55], v4 offset:3072 // (op 866) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[104:107], v[100:103], a[112:115], v91, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7556) 
s_waitcnt vmcnt(18)
s_barrier  // (op 21580) 




v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[104:107], v[114:117], a[116:119], v91, v97 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7480) 
ds_read_b128 v[56:59], v0 offset:8192 // (op 60) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[104:107], v[118:121], a[120:123], v91, v98 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7404) 
ds_read_b128 v[60:63], v0 offset:9216 // (op 1309) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[104:107], v[122:125], a[124:127], v91, v99 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7328) 
ds_read_b128 v[64:67], v0 offset:10240 // (op 2311) Load local data 







ds_read_b128 v[68:71], v0 offset:11264 // (op 3313) Load local data 
buffer_load_dwordx4 v20, s[48:51], s73 offen offset: 0 lds // (op 20432) Load value direct to lds



ds_read_b128 v[72:75], v0 offset:12288 // (op 4315) Load local data 
v_add_co_u32 v20, s[74:75], v20, v16 // (op 20432) least significant half



ds_read_b128 v[76:79], v0 offset:13312 // (op 5317) Load local data 
v_addc_co_u32 v21, s[74:75], 0, v21, s[74:75] // (op 20432) most significant half




s_add_u32 m0, m0, 4096 // (op 20432) 
ds_read_b128 v[80:83], v0 offset:14336 // (op 6319) Load local data 
buffer_load_dwordx4 v20, s[48:51], s73 offen offset: 0 lds // (op 20432) Load value direct to lds



v_add_co_u32 v20, s[74:75], v20, v16 // (op 20432) least significant half
v_addc_co_u32 v21, s[74:75], 0, v21, s[74:75] // (op 20432) most significant half

s_add_u32 m0, m0, 4096 // (op 20432) 
buffer_load_dwordx4 v20, s[48:51], s73 offen offset: 0 lds // (op 20432) Load value direct to lds




ds_read_b128 v[84:87], v0 offset:15360 // (op 7321) Load local data 

v_add_co_u32 v18, vcc, 256, v18 // (op 20543) least significant half
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 20543) most significant half
s_ashr_i64 s[72:73], s[42:43], 7 // (op 6289) 
s_mov_b32 s74, s72 // (op 6289) convert

s_and_b32 s72, s74, 4294967292 // (op 6289) 

s_cmp_lt_u32 s25, s72 // (op 6289) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_WGM__ForLoopTop_KLoop_6289 // (op 6289) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_WGM__ForLoopTop_KLoop_6289 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_WGM__ForLoopBottom_KLoop_6289:















s_barrier  // (op 21729) 
s_mul_hi_i32 s47, s16, s10 // (op 20475) 
v_lshrrev_b32 v14, 2, v1 // (op 20475) 
s_add_i32 s0, s47, s16 // (op 20475) 

v_and_b32 v15, -4, v1 // (op 20475) 
s_ashr_i32 s1, s0, 31 // (op 20475) 
v_sub_u32 v16, v1, v15 // (op 20475) 

s_cmp_eq_i32 s10, 0 // (op 20475) 
s_mov_b32 s47, scc // (op 20475) 
v_lshlrev_b32 v15, 5, v16 // (op 20475) 

s_cmp_eq_i32 s47, 0 // (op 20475) 
s_cselect_b32 s18, 0, -1 // (op 20475) 

s_and_b32 s19, s13, 31 // (op 20475) 
s_lshl_b32 s47, 1, s19 // (op 20475) 
s_add_i32 s22, s47, s18 // (op 20475) 


s_and_b32 s18, s1, s22 // (op 20475) 


s_add_i32 s1, s0, s18 // (op 20475) 


s_ashr_i32 s0, s1, s19 // (op 20475) 


s_lshl_b32 s1, s13, 24 // (op 20475) 
s_ashr_i32 s47, s1, 31 // (op 20475) 

s_xor_b32 s1, s0, s47 // (op 20475) 

s_sub_i32 s0, s1, s47 // (op 20475) 


s_mul_hi_i32 s1, s0, s14 // (op 20475) 
s_add_i32 s47, s1, s0 // (op 20475) 

s_ashr_i32 s1, s47, 31 // (op 20475) 
s_cmp_eq_i32 s14, 0 // (op 20475) 
s_mov_b32 s18, scc // (op 20475) 
s_cmp_eq_i32 s18, 0 // (op 20475) 
s_cselect_b32 s19, 0, -1 // (op 20475) 

s_and_b32 s18, s15, 31 // (op 20475) 
s_lshl_b32 s22, 1, s18 // (op 20475) 
s_add_i32 s23, s22, s19 // (op 20475) 


s_and_b32 s19, s1, s23 // (op 20475) 


s_add_i32 s1, s47, s19 // (op 20475) 


s_ashr_i32 s19, s1, s18 // (op 20475) 


s_lshl_b32 s1, s15, 24 // (op 20475) 
s_ashr_i32 s18, s1, 31 // (op 20475) 

s_xor_b32 s1, s19, s18 // (op 20475) 

s_sub_i32 s19, s1, s18 // (op 20475) 


s_mul_i32 s1, s0, s11 // (op 20475) 

s_sub_u32 s0, s16, s1 // (op 20475) 

s_mul_i32 s1, s19, s11 // (op 20475) 
s_add_u32 s1, s1, s0 // (op 20475) 


s_mul_hi_i32 s0, s16, s9 // (op 20475) 
s_add_i32 s47, s0, s16 // (op 20475) 

s_ashr_i32 s0, s47, 31 // (op 20475) 
s_cmp_eq_i32 s9, 0 // (op 20475) 
s_mov_b32 s18, scc // (op 20475) 
s_cmp_eq_i32 s18, 0 // (op 20475) 
s_cselect_b32 s19, 0, -1 // (op 20475) 

s_and_b32 s18, s45, 31 // (op 20475) 
s_lshl_b32 s22, 1, s18 // (op 20475) 
s_add_i32 s23, s22, s19 // (op 20475) 


s_and_b32 s19, s0, s23 // (op 20475) 


s_add_i32 s0, s47, s19 // (op 20475) 


s_ashr_i32 s19, s0, s18 // (op 20475) 


s_lshl_b32 s0, s45, 24 // (op 20475) 
s_ashr_i32 s18, s0, 31 // (op 20475) 

s_xor_b32 s0, s19, s18 // (op 20475) 

s_sub_i32 s19, s0, s18 // (op 20475) 


s_add_u32 s22, s6, 128 // (op 20475) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 20475) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 20475) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 20475) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 20475) 

s_mul_i32 s0, s11, s22 // (op 20475) 

s_mul_i32 s18, s8, s0 // (op 20475) 

s_mul_i32 s0, s19, s18 // (op 20475) 

s_sub_u32 s18, s16, s0 // (op 20475) 

s_mov_b32 s0, s18 // (op 20475) 
s_mul_hi_i32 s47, s0, s12 // (op 20475) 
s_add_i32 s22, s47, s0 // (op 20475) 


s_ashr_i32 s0, s22, 31 // (op 20475) 
s_cmp_eq_i32 s12, 0 // (op 20475) 
s_mov_b32 s47, scc // (op 20475) 
s_cmp_eq_i32 s47, 0 // (op 20475) 
s_cselect_b32 s23, 0, -1 // (op 20475) 

s_and_b32 s47, s44, 31 // (op 20475) 
s_lshl_b32 s24, 1, s47 // (op 20475) 
s_add_i32 s25, s24, s23 // (op 20475) 


s_and_b32 s23, s0, s25 // (op 20475) 


s_add_i32 s0, s22, s23 // (op 20475) 


s_ashr_i32 s22, s0, s47 // (op 20475) 


s_lshl_b32 s0, s44, 24 // (op 20475) 
s_ashr_i32 s47, s0, 31 // (op 20475) 

s_xor_b32 s0, s22, s47 // (op 20475) 

s_sub_i32 s22, s0, s47 // (op 20475) 


s_mul_i32 s0, s46, s11 // (op 20475) 
s_add_u32 s24, s20, 256 // (op 20475) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 20475) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 20475) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 20475) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 8 // (op 20475) 

s_sub_i32 s23, s24, s0 // (op 20475) 

s_mul_i32 s47, s22, s23 // (op 20475) 


s_sub_u32 s22, s18, s47 // (op 20475) 


s_add_u32 s18, s22, s0 // (op 20475) 


s_cmp_eq_i32 s19, 0 // (op 20475) 
s_mov_b32 s0, scc // (op 20475) 

s_cmp_eq_i32 s0, 0 // (op 20475) 
s_cselect_b32 s19, s18, s1 // (op 20475) 



v_lshl_add_u32 v16, s19, 8, v14 // (op 20475) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 20475) 
s_lshr_b32 s47, s0, 2 // (op 20475) 

v_lshl_add_u32 v14, s47, 9, v15 // (op 20475) 


v_mov_b32 v27, 0 // (op 20475) convert
v_mov_b32 v26, v14 // (op 20475) convert

v_mul_lo_u32 v15, v16, s27 // (op 20475) most significant: low of low * high
v_mul_hi_u32 v17, v16, s26 // (op 20475) most significant: high of low * low
v_mul_lo_u32 v14, v16, s26 // (op 20475) least significant: low of low * low
v_add_u32 v15, v15, v17 // (op 20475) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v26 // (op 20475) least significant half
v_addc_co_u32 v15, s[0:1], v15, v27, s[0:1] // (op 20475) most significant half



v_lshlrev_b64 v[16:17], 2, v[14:15] // (op 20475) 

v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 20475) 

s_mul_i32 s1, s26, s21 // (op 20477) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 20477) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 20477) least significant
s_mul_hi_u32 s18, s26, s20 // (op 20477) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 20477) most significant: sum
s_add_u32 s1, s1, s18 // (op 20477) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 20477) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 20477) 

s_mov_b32 s48, s28 // (op 20477) reg expression
s_mov_b32 s49, s29 // (op 20477) reg expression
s_mov_b32 s50, s0 // (op 20477) reg expression
s_mov_b32 s51, 131072 // (op 20477) call()

s_mul_hi_i32 s47, s16, s10 // (op 20573) 
v_lshrrev_b32 v16, 2, v1 // (op 20573) 
s_add_i32 s0, s47, s16 // (op 20573) 

v_and_b32 v17, -4, v1 // (op 20573) 
s_ashr_i32 s1, s0, 31 // (op 20573) 
v_sub_u32 v18, v1, v17 // (op 20573) 

s_cmp_eq_i32 s10, 0 // (op 20573) 
s_mov_b32 s47, scc // (op 20573) 
v_lshlrev_b32 v17, 5, v18 // (op 20573) 

s_cmp_eq_i32 s47, 0 // (op 20573) 
s_cselect_b32 s18, 0, -1 // (op 20573) 

s_and_b32 s19, s13, 31 // (op 20573) 
s_lshl_b32 s47, 1, s19 // (op 20573) 
s_add_i32 s22, s47, s18 // (op 20573) 


s_and_b32 s18, s1, s22 // (op 20573) 


s_add_i32 s1, s0, s18 // (op 20573) 


s_ashr_i32 s0, s1, s19 // (op 20573) 


s_lshl_b32 s1, s13, 24 // (op 20573) 
s_ashr_i32 s47, s1, 31 // (op 20573) 

s_xor_b32 s1, s0, s47 // (op 20573) 

s_sub_i32 s0, s1, s47 // (op 20573) 


s_mul_hi_i32 s1, s0, s14 // (op 20573) 
s_add_i32 s47, s1, s0 // (op 20573) 

s_ashr_i32 s1, s47, 31 // (op 20573) 
s_cmp_eq_i32 s14, 0 // (op 20573) 
s_mov_b32 s18, scc // (op 20573) 
s_cmp_eq_i32 s18, 0 // (op 20573) 
s_cselect_b32 s19, 0, -1 // (op 20573) 

s_and_b32 s18, s15, 31 // (op 20573) 
s_lshl_b32 s22, 1, s18 // (op 20573) 
s_add_i32 s23, s22, s19 // (op 20573) 


s_and_b32 s19, s1, s23 // (op 20573) 


s_add_i32 s1, s47, s19 // (op 20573) 


s_ashr_i32 s19, s1, s18 // (op 20573) 


s_lshl_b32 s1, s15, 24 // (op 20573) 
s_ashr_i32 s18, s1, 31 // (op 20573) 

s_xor_b32 s1, s19, s18 // (op 20573) 

s_sub_i32 s19, s1, s18 // (op 20573) 


s_add_u32 s22, s6, 128 // (op 20573) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 20573) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 20573) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 20573) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 20573) 

s_mul_i32 s1, s19, s22 // (op 20573) 

s_sub_i32 s47, s0, s1 // (op 20573) 


s_mul_hi_i32 s0, s16, s9 // (op 20573) 
s_add_i32 s1, s0, s16 // (op 20573) 

s_ashr_i32 s0, s1, 31 // (op 20573) 
s_cmp_eq_i32 s9, 0 // (op 20573) 
s_mov_b32 s18, scc // (op 20573) 
s_cmp_eq_i32 s18, 0 // (op 20573) 
s_cselect_b32 s19, 0, -1 // (op 20573) 

s_and_b32 s18, s45, 31 // (op 20573) 
s_lshl_b32 s24, 1, s18 // (op 20573) 
s_add_i32 s25, s24, s19 // (op 20573) 


s_and_b32 s19, s0, s25 // (op 20573) 


s_add_i32 s0, s1, s19 // (op 20573) 


s_ashr_i32 s1, s0, s18 // (op 20573) 


s_lshl_b32 s0, s45, 24 // (op 20573) 
s_ashr_i32 s18, s0, 31 // (op 20573) 

s_xor_b32 s0, s1, s18 // (op 20573) 

s_sub_i32 s1, s0, s18 // (op 20573) 


s_mul_i32 s0, s11, s22 // (op 20573) 

s_mul_i32 s18, s8, s0 // (op 20573) 

s_mul_i32 s0, s1, s18 // (op 20573) 

s_sub_u32 s18, s16, s0 // (op 20573) 

s_mul_hi_i32 s0, s18, s12 // (op 20573) 
s_add_i32 s19, s0, s18 // (op 20573) 


s_ashr_i32 s0, s19, 31 // (op 20573) 
s_cmp_eq_i32 s12, 0 // (op 20573) 
s_mov_b32 s18, scc // (op 20573) 
s_cmp_eq_i32 s18, 0 // (op 20573) 
s_cselect_b32 s22, 0, -1 // (op 20573) 

s_and_b32 s18, s44, 31 // (op 20573) 
s_lshl_b32 s23, 1, s18 // (op 20573) 
s_add_i32 s24, s23, s22 // (op 20573) 


s_and_b32 s25, s0, s24 // (op 20573) 


s_add_i32 s0, s19, s25 // (op 20573) 


s_ashr_i32 s19, s0, s18 // (op 20573) 


s_lshl_b32 s0, s44, 24 // (op 20573) 
s_ashr_i32 s18, s0, 31 // (op 20573) 

s_xor_b32 s0, s19, s18 // (op 20573) 

s_sub_i32 s19, s0, s18 // (op 20573) 


s_cmp_eq_i32 s1, 0 // (op 20573) 
s_mov_b32 s0, scc // (op 20573) 

s_cmp_eq_i32 s0, 0 // (op 20573) 
s_cselect_b32 s1, s19, s47 // (op 20573) 



v_lshl_add_u32 v18, s1, 7, v16 // (op 20573) 


v_mul_lo_u32 v27, v18, s31 // (op 20573) most significant: low of low * high
v_mul_hi_u32 v16, v18, s30 // (op 20573) most significant: high of low * low
v_mul_lo_u32 v26, v18, s30 // (op 20573) least significant: low of low * low
v_add_u32 v27, v27, v16 // (op 20573) most significant: sum


s_ashr_i64 s[0:1], s[42:43], 7 // (op 20573) 
s_lshr_b32 s47, s0, 2 // (op 20573) 

v_lshl_add_u32 v16, s47, 9, v17 // (op 20573) 


v_mov_b32 v25, 0 // (op 20573) convert
v_mov_b32 v24, v16 // (op 20573) convert

v_add_co_u32 v16, s[0:1], v24, v26 // (op 20573) least significant half
v_addc_co_u32 v17, s[0:1], v25, v27, s[0:1] // (op 20573) most significant half

v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 20573) 


v_lshrrev_b64 v[18:19], 3, v[16:17] // (op 20573) 

s_mul_i32 s1, s30, s7 // (op 20575) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 20575) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 20575) least significant
s_mul_hi_u32 s18, s30, s6 // (op 20575) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 20575) most significant: sum
s_add_u32 s1, s1, s18 // (op 20575) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 20575) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 20575) 

s_mov_b32 s60, s32 // (op 20575) reg expression
s_mov_b32 s61, s33 // (op 20575) reg expression
s_mov_b32 s62, s0 // (op 20575) reg expression
s_mov_b32 s63, 131072 // (op 20575) call()

v_and_b32 v16, -4, v1 // (op 20671) 
v_sub_u32 v17, v1, v16 // (op 20671) 

v_lshlrev_b32 v16, 5, v17 // (op 20671) 

v_lshrrev_b32 v17, 2, v1 // (op 20671) 
v_lshl_add_u32 v20, v17, 7, v16 // (op 20671) 


v_lshlrev_b32 v16, 2, v20 // (op 20671) 

v_lshrrev_b32 v17, 3, v16 // (op 20671) 

s_nop 0
v_readfirstlane_b32 s47, v17 // (op 20671) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -4, v1 // (op 20728) 
v_sub_u32 v17, v1, v16 // (op 20728) 

v_lshlrev_b32 v16, 5, v17 // (op 20728) 

v_lshrrev_b32 v17, 2, v1 // (op 20728) 
v_lshl_add_u32 v20, v17, 7, v16 // (op 20728) 


v_lshlrev_b32 v16, 2, v20 // (op 20728) 

v_lshrrev_b32 v17, 3, v16 // (op 20728) 

s_nop 0
v_readfirstlane_b32 s0, v17 // (op 20728) Wait state hazard: VALU Write Readlane Hazard

s_mul_hi_i32 s1, s16, s10 // (op 21169) 
v_and_b32 v16, 63, v1 // (op 21169) 
s_add_i32 s18, s1, s16 // (op 21169) 

v_lshrrev_b32 v17, 4, v16 // (op 21169) 
s_ashr_i32 s1, s18, 31 // (op 21169) 
v_and_b32 v20, 3, v17 // (op 21169) 

s_cmp_eq_i32 s10, 0 // (op 21169) 
s_mov_b32 s19, scc // (op 21169) 
v_and_b32 v17, 15, v1 // (op 21169) 
s_cmp_eq_i32 s19, 0 // (op 21169) 
s_cselect_b32 s22, 0, -1 // (op 21169) 

v_lshl_add_u32 v21, v20, 4, v17 // (op 21169) 


s_and_b32 s19, s13, 31 // (op 21169) 
v_and_b32 v17, 4294967168, v1 // (op 21169) 
s_lshl_b32 s23, 1, s19 // (op 21169) 
v_add_u32 v20, v17, v21 // (op 21169) 


s_add_i32 s24, s23, s22 // (op 21169) 


v_lshrrev_b32 v17, 6, v16 // (op 21169) 

s_and_b32 s25, s1, s24 // (op 21169) 


v_lshlrev_b32 v16, 2, v17 // (op 21169) 

s_add_i32 s1, s18, s25 // (op 21169) 


s_ashr_i32 s18, s1, s19 // (op 21169) 


s_lshl_b32 s1, s13, 24 // (op 21169) 
s_ashr_i32 s19, s1, 31 // (op 21169) 

s_xor_b32 s1, s18, s19 // (op 21169) 

s_sub_i32 s18, s1, s19 // (op 21169) 


s_mul_hi_i32 s1, s18, s14 // (op 21169) 
s_add_i32 s19, s1, s18 // (op 21169) 

s_ashr_i32 s1, s19, 31 // (op 21169) 
s_cmp_eq_i32 s14, 0 // (op 21169) 
s_mov_b32 s22, scc // (op 21169) 
s_cmp_eq_i32 s22, 0 // (op 21169) 
s_cselect_b32 s23, 0, -1 // (op 21169) 

s_and_b32 s22, s15, 31 // (op 21169) 
s_lshl_b32 s24, 1, s22 // (op 21169) 
s_add_i32 s25, s24, s23 // (op 21169) 


s_and_b32 s23, s1, s25 // (op 21169) 


s_add_i32 s1, s19, s23 // (op 21169) 


s_ashr_i32 s19, s1, s22 // (op 21169) 


s_lshl_b32 s1, s15, 24 // (op 21169) 
s_ashr_i32 s22, s1, 31 // (op 21169) 

s_xor_b32 s1, s19, s22 // (op 21169) 

s_sub_i32 s19, s1, s22 // (op 21169) 


s_mul_i32 s1, s18, s11 // (op 21169) 

s_sub_u32 s18, s16, s1 // (op 21169) 

s_mul_i32 s1, s19, s11 // (op 21169) 
s_add_u32 s1, s1, s18 // (op 21169) 


s_mul_hi_i32 s18, s16, s9 // (op 21169) 
s_add_i32 s19, s18, s16 // (op 21169) 

s_ashr_i32 s18, s19, 31 // (op 21169) 
s_cmp_eq_i32 s9, 0 // (op 21169) 
s_mov_b32 s22, scc // (op 21169) 
s_cmp_eq_i32 s22, 0 // (op 21169) 
s_cselect_b32 s23, 0, -1 // (op 21169) 

s_and_b32 s22, s45, 31 // (op 21169) 
s_lshl_b32 s24, 1, s22 // (op 21169) 
s_add_i32 s25, s24, s23 // (op 21169) 


s_and_b32 s23, s18, s25 // (op 21169) 


s_add_i32 s18, s19, s23 // (op 21169) 


s_ashr_i32 s19, s18, s22 // (op 21169) 


s_lshl_b32 s18, s45, 24 // (op 21169) 
s_ashr_i32 s22, s18, 31 // (op 21169) 

s_xor_b32 s18, s19, s22 // (op 21169) 

s_sub_i32 s19, s18, s22 // (op 21169) 


s_add_u32 s22, s6, 128 // (op 21169) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 21169) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 21169) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 21169) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 21169) 

s_mul_i32 s18, s11, s22 // (op 21169) 

s_mul_i32 s22, s8, s18 // (op 21169) 

s_mul_i32 s18, s19, s22 // (op 21169) 

s_sub_u32 s22, s16, s18 // (op 21169) 

s_mov_b32 s18, s22 // (op 21169) 
s_mul_hi_i32 s23, s18, s12 // (op 21169) 
s_add_i32 s24, s23, s18 // (op 21169) 


s_ashr_i32 s18, s24, 31 // (op 21169) 
s_cmp_eq_i32 s12, 0 // (op 21169) 
s_mov_b32 s23, scc // (op 21169) 
s_cmp_eq_i32 s23, 0 // (op 21169) 
s_cselect_b32 s25, 0, -1 // (op 21169) 

s_and_b32 s23, s44, 31 // (op 21169) 
s_lshl_b32 s64, 1, s23 // (op 21169) 
s_add_i32 s65, s64, s25 // (op 21169) 


s_and_b32 s25, s18, s65 // (op 21169) 


s_add_i32 s18, s24, s25 // (op 21169) 


s_ashr_i32 s24, s18, s23 // (op 21169) 


s_lshl_b32 s18, s44, 24 // (op 21169) 
s_ashr_i32 s23, s18, 31 // (op 21169) 

s_xor_b32 s18, s24, s23 // (op 21169) 

s_sub_i32 s24, s18, s23 // (op 21169) 


s_mul_i32 s18, s46, s11 // (op 21169) 
s_add_u32 s64, s20, 256 // (op 21169) Start of Int64 add, locking SCC
s_addc_u32 s65, s21, 0 // (op 21169) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 21169) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 21169) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 8 // (op 21169) 

s_sub_i32 s23, s64, s18 // (op 21169) 

s_mul_i32 s25, s24, s23 // (op 21169) 


s_sub_u32 s23, s22, s25 // (op 21169) 


s_add_u32 s22, s23, s18 // (op 21169) 


s_cmp_eq_i32 s19, 0 // (op 21169) 
s_mov_b32 s18, scc // (op 21169) 

s_cmp_eq_i32 s18, 0 // (op 21169) 
s_cselect_b32 s19, s22, s1 // (op 21169) 



v_lshl_add_u32 v17, s19, 8, v20 // (op 21169) 


s_ashr_i64 s[18:19], s[42:43], 7 // (op 21169) 
s_lshr_b32 s1, s18, 2 // (op 21169) 

v_lshl_add_u32 v20, s1, 4, v16 // (op 21169) 


v_mul_lo_u32 v16, v17, s34 // (op 21169) 
v_add_u32 v16, v16, v20 // (op 21169) 


s_mul_i32 s19, s34, s21 // (op 21171) most significant: low of low * high
s_mul_i32 s1, s35, s20 // (op 21171) most significant: low of high * low
s_mul_i32 s18, s34, s20 // (op 21171) least significant
s_mul_hi_u32 s22, s34, s20 // (op 21171) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 21171) most significant: sum
s_add_u32 s19, s19, s22 // (op 21171) most significant: sum


s_mov_b32 s64, s36 // (op 21171) reg expression
s_mov_b32 s65, s37 // (op 21171) reg expression
s_mov_b32 s66, s18 // (op 21171) reg expression
s_mov_b32 s67, 131072 // (op 21171) call()

s_mul_hi_i32 s1, s16, s10 // (op 21196) 
v_and_b32 v17, 63, v1 // (op 21196) 
s_add_i32 s18, s1, s16 // (op 21196) 

v_lshrrev_b32 v20, 4, v17 // (op 21196) 
s_ashr_i32 s1, s18, 31 // (op 21196) 
v_and_b32 v21, 3, v20 // (op 21196) 

s_cmp_eq_i32 s10, 0 // (op 21196) 
s_mov_b32 s19, scc // (op 21196) 
v_and_b32 v20, 15, v1 // (op 21196) 
s_cmp_eq_i32 s19, 0 // (op 21196) 
s_cselect_b32 s22, 0, -1 // (op 21196) 

v_lshl_add_u32 v22, v21, 4, v20 // (op 21196) 


s_and_b32 s19, s13, 31 // (op 21196) 
v_lshrrev_b32 v20, 6, v1 // (op 21196) 
s_lshl_b32 s23, 1, s19 // (op 21196) 
v_and_b32 v21, 1, v20 // (op 21196) 

s_add_i32 s24, s23, s22 // (op 21196) 


v_lshl_add_u32 v20, v21, 6, v22 // (op 21196) 


s_and_b32 s25, s1, s24 // (op 21196) 


v_lshrrev_b32 v21, 6, v17 // (op 21196) 

s_add_i32 s1, s18, s25 // (op 21196) 


v_lshlrev_b32 v17, 2, v21 // (op 21196) 

s_ashr_i32 s18, s1, s19 // (op 21196) 


s_lshl_b32 s1, s13, 24 // (op 21196) 
s_ashr_i32 s19, s1, 31 // (op 21196) 

s_xor_b32 s1, s18, s19 // (op 21196) 

s_sub_i32 s18, s1, s19 // (op 21196) 


s_mul_hi_i32 s1, s18, s14 // (op 21196) 
s_add_i32 s19, s1, s18 // (op 21196) 

s_ashr_i32 s1, s19, 31 // (op 21196) 
s_cmp_eq_i32 s14, 0 // (op 21196) 
s_mov_b32 s22, scc // (op 21196) 
s_cmp_eq_i32 s22, 0 // (op 21196) 
s_cselect_b32 s23, 0, -1 // (op 21196) 

s_and_b32 s22, s15, 31 // (op 21196) 
s_lshl_b32 s24, 1, s22 // (op 21196) 
s_add_i32 s25, s24, s23 // (op 21196) 


s_and_b32 s23, s1, s25 // (op 21196) 


s_add_i32 s1, s19, s23 // (op 21196) 


s_ashr_i32 s19, s1, s22 // (op 21196) 


s_lshl_b32 s1, s15, 24 // (op 21196) 
s_ashr_i32 s22, s1, 31 // (op 21196) 

s_xor_b32 s1, s19, s22 // (op 21196) 

s_sub_i32 s19, s1, s22 // (op 21196) 


s_add_u32 s22, s6, 128 // (op 21196) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 21196) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 21196) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 21196) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 21196) 

s_mul_i32 s1, s19, s22 // (op 21196) 

s_sub_i32 s19, s18, s1 // (op 21196) 


s_mul_hi_i32 s1, s16, s9 // (op 21196) 
s_add_i32 s18, s1, s16 // (op 21196) 

s_ashr_i32 s1, s18, 31 // (op 21196) 
s_cmp_eq_i32 s9, 0 // (op 21196) 
s_mov_b32 s24, scc // (op 21196) 
s_cmp_eq_i32 s24, 0 // (op 21196) 
s_cselect_b32 s25, 0, -1 // (op 21196) 

s_and_b32 s24, s45, 31 // (op 21196) 
s_lshl_b32 s68, 1, s24 // (op 21196) 
s_add_i32 s69, s68, s25 // (op 21196) 


s_and_b32 s25, s1, s69 // (op 21196) 


s_add_i32 s1, s18, s25 // (op 21196) 


s_ashr_i32 s18, s1, s24 // (op 21196) 


s_lshl_b32 s1, s45, 24 // (op 21196) 
s_ashr_i32 s24, s1, 31 // (op 21196) 

s_xor_b32 s1, s18, s24 // (op 21196) 

s_sub_i32 s18, s1, s24 // (op 21196) 


s_mul_i32 s1, s11, s22 // (op 21196) 

s_mul_i32 s22, s8, s1 // (op 21196) 

s_mul_i32 s1, s18, s22 // (op 21196) 

s_sub_u32 s22, s16, s1 // (op 21196) 

s_mul_hi_i32 s1, s22, s12 // (op 21196) 
s_add_i32 s23, s1, s22 // (op 21196) 


s_ashr_i32 s1, s23, 31 // (op 21196) 
s_cmp_eq_i32 s12, 0 // (op 21196) 
s_mov_b32 s22, scc // (op 21196) 
s_cmp_eq_i32 s22, 0 // (op 21196) 
s_cselect_b32 s24, 0, -1 // (op 21196) 

s_and_b32 s22, s44, 31 // (op 21196) 
s_lshl_b32 s25, 1, s22 // (op 21196) 
s_add_i32 s68, s25, s24 // (op 21196) 


s_and_b32 s24, s1, s68 // (op 21196) 


s_add_i32 s1, s23, s24 // (op 21196) 


s_ashr_i32 s23, s1, s22 // (op 21196) 


s_lshl_b32 s1, s44, 24 // (op 21196) 
s_ashr_i32 s22, s1, 31 // (op 21196) 

s_xor_b32 s1, s23, s22 // (op 21196) 

s_sub_i32 s23, s1, s22 // (op 21196) 


s_cmp_eq_i32 s18, 0 // (op 21196) 
s_mov_b32 s1, scc // (op 21196) 

s_cmp_eq_i32 s1, 0 // (op 21196) 
s_cselect_b32 s18, s23, s19 // (op 21196) 



v_lshl_add_u32 v21, s18, 7, v20 // (op 21196) 


v_mul_lo_u32 v20, v21, s38 // (op 21196) 

s_ashr_i64 s[18:19], s[42:43], 7 // (op 21196) 
s_lshr_b32 s1, s18, 2 // (op 21196) 

v_lshl_add_u32 v21, s1, 4, v17 // (op 21196) 


v_add_u32 v17, v21, v20 // (op 21196) 


s_mul_i32 s19, s38, s7 // (op 21198) most significant: low of low * high
s_mul_i32 s1, s39, s6 // (op 21198) most significant: low of high * low
s_mul_i32 s18, s38, s6 // (op 21198) least significant
s_mul_hi_u32 s22, s38, s6 // (op 21198) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 21198) most significant: sum
s_add_u32 s19, s19, s22 // (op 21198) most significant: sum


s_mov_b32 s68, s40 // (op 21198) reg expression
s_mov_b32 s69, s41 // (op 21198) reg expression
s_mov_b32 s70, s18 // (op 21198) reg expression
s_mov_b32 s71, 131072 // (op 21198) call()

s_ashr_i64 s[18:19], s[42:43], 7 // (op 7297) 
s_and_b32 s1, s18, 4294967292 // (op 7297) 

s_ashr_i64 s[18:19], s[42:43], 7 // (op 7296) 
s_mov_b32 s22, s18 // (op 7296) convert

s_cmp_lt_u32 s1, s22 // (op 7296) 
















s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_WGM__ForLoopBottom_KLoopTail_7296 // (op 7296) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_WGM__ForLoopBottom_KLoopTail_7296 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_WGM__ForLoopTop_KLoopTail_7296:
s_barrier  // (op 21577) 
s_add_u32 s1, s1, 1 // (op 7298) 

s_lshl_b64 s[18:19], s[26:27], 6 // (op 20400) 
s_lshl_b64 s[22:23], s[18:19], 2 // (op 20400) 

v_lshrrev_b64 v[20:21], 3, s[22:23] // (op 20400) 


v_mov_b64 v[22:23], v[14:15] // (op 20400) 
v_mov_b64 v[24:25], v[22:23] // (op 20400) 
s_add_u32 s18, 8192, s47 // (op 20400) 
s_mov_b32 m0, s18 // (op 20400) reg expression
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 20400) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v20 // (op 20400) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 20400) most significant half

s_add_u32 m0, m0, 4096 // (op 20400) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 20400) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v20 // (op 20400) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 20400) most significant half

s_add_u32 m0, m0, 4096 // (op 20400) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 20400) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v20 // (op 20400) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 20400) most significant half

s_add_u32 m0, m0, 4096 // (op 20400) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 20400) Load value direct to lds




v_add_co_u32 v14, s[18:19], 64, v14 // (op 20465) least significant half
v_addc_co_u32 v15, s[18:19], 0, v15, s[18:19] // (op 20465) most significant half

s_barrier  // (op 21593) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 20406) 
s_lshl_b64 s[22:23], s[18:19], 2 // (op 20406) 

v_lshrrev_b64 v[20:21], 3, s[22:23] // (op 20406) 


v_mov_b64 v[22:23], v[18:19] // (op 20406) 
v_mov_b64 v[24:25], v[22:23] // (op 20406) 
s_mov_b32 m0, s0 // (op 20406) reg expression
buffer_load_dwordx4 v24, s[60:63], 0 offen offset: 0 lds // (op 20406) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v20 // (op 20406) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 20406) most significant half

s_add_u32 m0, m0, 4096 // (op 20406) 
buffer_load_dwordx4 v24, s[60:63], 0 offen offset: 0 lds // (op 20406) Load value direct to lds



v_add_co_u32 v18, s[18:19], 64, v18 // (op 20565) least significant half
v_addc_co_u32 v19, s[18:19], 0, v19, s[18:19] // (op 20565) most significant half

s_waitcnt vmcnt(2)
s_barrier  // (op 21583) 



ds_read_b128 v[20:23], v0 offset:8192 // (op 160) Load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 21599) 



ds_read_b128 v[24:27], v4 // (op 162) Load local data 


v_mov_b32 v28, v16 // (op 164) 
v_mov_b32 v29, v28 // (op 164) reg expression

buffer_load_dwordx4 v[30:33], v29, s[64:67], 0 offen // (op 164) Load value



v_mov_b32 v29, v17 // (op 166) 
v_mov_b32 v34, v29 // (op 166) reg expression

buffer_load_dwordx4 v[36:39], v34, s[68:71], 0 offen // (op 166) Load value


v_add_u32 v17, 4, v17 // (op 21186) 
s_waitcnt vmcnt(1)
v_bfe_u32 v40, v30, 0, 8 // (op 19025) 



v_bfe_u32 v41, v30, 8, 8 // (op 19025) 
v_bfe_u32 v42, v30, 16, 8 // (op 19025) 
v_bfe_u32 v43, v30, 24, 8 // (op 19025) 
v_permlane16_swap_b32 v40, v41 // (op 19025) 
ds_read_b128 v[44:47], v4 offset:1024 // (op 397) Load local data 
v_permlane16_swap_b32 v42, v43 // (op 19025) 



s_nop 1
v_permlane32_swap_b32 v40, v42 // (op 19025) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v41, v43 // (op 19025) 
ds_read_b128 v[48:51], v4 offset:2048 // (op 623) Load local data 




ds_read_b128 v[32:35], v4 offset:3072 // (op 849) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v52, v36, 0, 8 // (op 19085) 



v_bfe_u32 v53, v36, 8, 8 // (op 19085) 
v_bfe_u32 v54, v36, 16, 8 // (op 19085) 
v_bfe_u32 v55, v36, 24, 8 // (op 19085) 
v_permlane16_swap_b32 v52, v53 // (op 19085) 
s_nop 0
v_permlane16_swap_b32 v54, v55 // (op 19085) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v52, v54 // (op 19085) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v53, v55 // (op 19085) 
ds_read_b128 v[56:59], v0 offset:9216 // (op 1209) Load local data 




s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:23], v[24:27], a[0:3], v40, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 167) 
ds_read_b128 v[36:39], v0 offset:10240 // (op 2211) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:23], v[44:47], a[4:7], v40, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 400) 
ds_read_b128 v[60:63], v0 offset:11264 // (op 3213) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:23], v[48:51], a[8:11], v40, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 626) 
ds_read_b128 v[64:67], v0 offset:12288 // (op 4215) Load local data 


s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:23], v[32:35], a[12:15], v40, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 852) 
s_lshl_b32 s18, s34, 6 // (op 4224) 

v_add_u32 v20, v28, s18 // (op 4224) 


buffer_load_dwordx4 v[68:71], v20, s[64:67], 0 offen // (op 4224) Load value

s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[56:59], v[24:27], a[16:19], v41, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1213) 




v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[56:59], v[48:51], a[24:27], v41, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1267) 
v_add_u32 v16, 4, v16 // (op 21159) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[56:59], v[32:35], a[28:31], v41, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1294) 
s_waitcnt vmcnt(0)
v_bfe_u32 v20, v68, 0, 8 // (op 19039) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[56:59], v[44:47], a[20:23], v41, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1240) 
v_bfe_u32 v21, v68, 8, 8 // (op 19039) 

v_bfe_u32 v22, v68, 16, 8 // (op 19039) 
v_bfe_u32 v23, v68, 24, 8 // (op 19039) 
v_permlane16_swap_b32 v20, v21 // (op 19039) 
s_nop 0
v_permlane16_swap_b32 v22, v23 // (op 19039) Wait state hazard: VALU Write Readlane Hazard
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[36:39], v[24:27], a[32:35], v42, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2215) 
ds_read_b128 v[28:31], v0 offset:13312 // (op 5217) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[36:39], v[48:51], a[40:43], v42, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2269) 
v_permlane32_swap_b32 v20, v22 // (op 19039) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[36:39], v[32:35], a[44:47], v42, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2296) 
v_permlane32_swap_b32 v21, v23 // (op 19039) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[36:39], v[44:47], a[36:39], v42, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2242) 
ds_read_b128 v[56:59], v0 offset:14336 // (op 6219) Load local data 




ds_read_b128 v[36:39], v0 offset:15360 // (op 7221) Load local data 
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[60:63], v[24:27], a[48:51], v43, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3217) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[60:63], v[48:51], a[56:59], v43, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3271) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[60:63], v[32:35], a[60:63], v43, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3298) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[60:63], v[44:47], a[52:55], v43, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3244) 


s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[64:67], v[44:47], a[68:71], v20, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4246) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[64:67], v[24:27], a[64:67], v20, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4219) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[64:67], v[48:51], a[72:75], v20, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4273) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[64:67], v[32:35], a[76:79], v20, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 4300) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[28:31], v[44:47], a[84:87], v21, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5248) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[28:31], v[24:27], a[80:83], v21, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5221) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[28:31], v[48:51], a[88:91], v21, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5275) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[28:31], v[32:35], a[92:95], v21, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 5302) 

s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[56:59], v[44:47], a[100:103], v22, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6250) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[56:59], v[24:27], a[96:99], v22, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6223) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[56:59], v[48:51], a[104:107], v22, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6277) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[56:59], v[32:35], a[108:111], v22, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 6304) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[36:39], v[44:47], a[116:119], v23, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7252) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[36:39], v[24:27], a[112:115], v23, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7225) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[36:39], v[48:51], a[120:123], v23, v54 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7279) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[36:39], v[32:35], a[124:127], v23, v55 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 7306) 




s_ashr_i64 s[18:19], s[42:43], 7 // (op 7296) 
s_mov_b32 s22, s18 // (op 7296) convert

s_cmp_lt_u32 s1, s22 // (op 7296) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_WGM__ForLoopTop_KLoopTail_7296 // (op 7296) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_WGM__ForLoopTop_KLoopTail_7296 if true)
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_WGM__ForLoopBottom_KLoopTail_7296:




















s_barrier  // (op 21749) 





















v_cmp_eq_f32 vcc, 0.00000, v9 // (op 114) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_WGM__ConditionalFalse_0_beta_114 // (op 114) Condition: False, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_WGM__ConditionalFalse_0_beta_114
v_accvgpr_read v4, a0 // (op 42) 
v_accvgpr_read v5, a1 // (op 42) 
v_accvgpr_read v6, a2 // (op 42) 
v_accvgpr_read v7, a3 // (op 42) 
v_mul_f32 v14, v11, v4 // (op 42) 
v_mul_f32 v15, v11, v5 // (op 42) 
v_mul_f32 v16, v11, v6 // (op 42) 
v_mul_f32 v17, v11, v7 // (op 42) 



v_mov_b32 v0, v13 // (op 48) reg expression

buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 48) Store value


v_accvgpr_read v4, a4 // (op 9198) 
v_accvgpr_read v5, a5 // (op 9198) 
v_accvgpr_read v6, a6 // (op 9198) 
v_accvgpr_read v7, a7 // (op 9198) 
v_mul_f32 v14, v11, v4 // (op 9198) 
v_mul_f32 v15, v11, v5 // (op 9198) 
v_mul_f32 v16, v11, v6 // (op 9198) 
v_mul_f32 v17, v11, v7 // (op 9198) 



s_lshl_b32 s0, s4, 6 // (op 9200) 
v_add_u32 v0, v13, s0 // (op 9200) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 9200) Store value


v_accvgpr_read v4, a8 // (op 9210) 
v_accvgpr_read v5, a9 // (op 9210) 
v_accvgpr_read v6, a10 // (op 9210) 
v_accvgpr_read v7, a11 // (op 9210) 
v_mul_f32 v14, v11, v4 // (op 9210) 
v_mul_f32 v15, v11, v5 // (op 9210) 
v_mul_f32 v16, v11, v6 // (op 9210) 
v_mul_f32 v17, v11, v7 // (op 9210) 



s_lshl_b32 s0, s4, 7 // (op 9212) 
v_add_u32 v0, v13, s0 // (op 9212) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 9212) Store value


v_accvgpr_read v4, a12 // (op 9222) 
v_accvgpr_read v5, a13 // (op 9222) 
v_accvgpr_read v6, a14 // (op 9222) 
v_accvgpr_read v7, a15 // (op 9222) 
v_mul_f32 v14, v11, v4 // (op 9222) 
v_mul_f32 v15, v11, v5 // (op 9222) 
v_mul_f32 v16, v11, v6 // (op 9222) 
v_mul_f32 v17, v11, v7 // (op 9222) 



s_lshl_b32 s0, s4, 6 // (op 9224) 
s_mul_i32 s1, 3, s0 // (op 9224) 

v_add_u32 v0, v13, s1 // (op 9224) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 9224) Store value


v_accvgpr_read v4, a16 // (op 9256) 
v_accvgpr_read v5, a17 // (op 9256) 
v_accvgpr_read v6, a18 // (op 9256) 
v_accvgpr_read v7, a19 // (op 9256) 
v_mul_f32 v14, v11, v4 // (op 9256) 
v_mul_f32 v15, v11, v5 // (op 9256) 
v_mul_f32 v16, v11, v6 // (op 9256) 
v_mul_f32 v17, v11, v7 // (op 9256) 



v_add_u32 v0, 64, v13 // (op 9259) 

buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 9259) Store value


v_accvgpr_read v4, a20 // (op 9269) 
v_accvgpr_read v5, a21 // (op 9269) 
v_accvgpr_read v6, a22 // (op 9269) 
v_accvgpr_read v7, a23 // (op 9269) 
v_mul_f32 v14, v11, v4 // (op 9269) 
v_mul_f32 v15, v11, v5 // (op 9269) 
v_mul_f32 v16, v11, v6 // (op 9269) 
v_mul_f32 v17, v11, v7 // (op 9269) 



s_lshl_b32 s0, s4, 6 // (op 9261) 
s_add_u32 s0, s0, 64 // (op 9261) 
v_add_u32 v0, v13, s0 // (op 9261) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 9261) Store value


v_accvgpr_read v4, a24 // (op 9274) 
v_accvgpr_read v5, a25 // (op 9274) 
v_accvgpr_read v6, a26 // (op 9274) 
v_accvgpr_read v7, a27 // (op 9274) 
v_mul_f32 v14, v11, v4 // (op 9274) 
v_mul_f32 v15, v11, v5 // (op 9274) 
v_mul_f32 v16, v11, v6 // (op 9274) 
v_mul_f32 v17, v11, v7 // (op 9274) 



s_lshl_b32 s0, s4, 7 // (op 9263) 
s_add_u32 s0, s0, 64 // (op 9263) 
v_add_u32 v0, v13, s0 // (op 9263) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 9263) Store value


v_accvgpr_read v4, a28 // (op 9279) 
v_accvgpr_read v5, a29 // (op 9279) 
v_accvgpr_read v6, a30 // (op 9279) 
v_accvgpr_read v7, a31 // (op 9279) 
v_mul_f32 v14, v11, v4 // (op 9279) 
v_mul_f32 v15, v11, v5 // (op 9279) 
v_mul_f32 v16, v11, v6 // (op 9279) 
v_mul_f32 v17, v11, v7 // (op 9279) 



s_lshl_b32 s0, s4, 6 // (op 9265) 
s_mul_i32 s1, 3, s0 // (op 9265) 
s_add_u32 s1, s1, 64 // (op 9265) 

v_add_u32 v0, v13, s1 // (op 9265) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 9265) Store value


v_accvgpr_read v4, a32 // (op 9324) 
v_accvgpr_read v5, a33 // (op 9324) 
v_accvgpr_read v6, a34 // (op 9324) 
v_accvgpr_read v7, a35 // (op 9324) 
v_mul_f32 v14, v11, v4 // (op 9324) 
v_mul_f32 v15, v11, v5 // (op 9324) 
v_mul_f32 v16, v11, v6 // (op 9324) 
v_mul_f32 v17, v11, v7 // (op 9324) 



v_add_u32 v0, 128, v13 // (op 9327) 

buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 9327) Store value


v_accvgpr_read v4, a36 // (op 9337) 
v_accvgpr_read v5, a37 // (op 9337) 
v_accvgpr_read v6, a38 // (op 9337) 
v_accvgpr_read v7, a39 // (op 9337) 
v_mul_f32 v14, v11, v4 // (op 9337) 
v_mul_f32 v15, v11, v5 // (op 9337) 
v_mul_f32 v16, v11, v6 // (op 9337) 
v_mul_f32 v17, v11, v7 // (op 9337) 



s_lshl_b32 s0, s4, 6 // (op 9329) 
s_add_u32 s0, s0, 128 // (op 9329) 
v_add_u32 v0, v13, s0 // (op 9329) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 9329) Store value


v_accvgpr_read v4, a40 // (op 9342) 
v_accvgpr_read v5, a41 // (op 9342) 
v_accvgpr_read v6, a42 // (op 9342) 
v_accvgpr_read v7, a43 // (op 9342) 
v_mul_f32 v14, v11, v4 // (op 9342) 
v_mul_f32 v15, v11, v5 // (op 9342) 
v_mul_f32 v16, v11, v6 // (op 9342) 
v_mul_f32 v17, v11, v7 // (op 9342) 



s_lshl_b32 s0, s4, 7 // (op 9331) 
s_add_u32 s0, s0, 128 // (op 9331) 
v_add_u32 v0, v13, s0 // (op 9331) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 9331) Store value


v_accvgpr_read v4, a44 // (op 9347) 
v_accvgpr_read v5, a45 // (op 9347) 
v_accvgpr_read v6, a46 // (op 9347) 
v_accvgpr_read v7, a47 // (op 9347) 
v_mul_f32 v14, v11, v4 // (op 9347) 
v_mul_f32 v15, v11, v5 // (op 9347) 
v_mul_f32 v16, v11, v6 // (op 9347) 
v_mul_f32 v17, v11, v7 // (op 9347) 



s_lshl_b32 s0, s4, 6 // (op 9333) 
s_mul_i32 s1, 3, s0 // (op 9333) 
s_add_u32 s1, s1, 128 // (op 9333) 

v_add_u32 v0, v13, s1 // (op 9333) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 9333) Store value


v_accvgpr_read v4, a48 // (op 9392) 
v_accvgpr_read v5, a49 // (op 9392) 
v_accvgpr_read v6, a50 // (op 9392) 
v_accvgpr_read v7, a51 // (op 9392) 
v_mul_f32 v14, v11, v4 // (op 9392) 
v_mul_f32 v15, v11, v5 // (op 9392) 
v_mul_f32 v16, v11, v6 // (op 9392) 
v_mul_f32 v17, v11, v7 // (op 9392) 



v_add_u32 v0, 192, v13 // (op 9395) 

buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 9395) Store value


v_accvgpr_read v4, a52 // (op 9405) 
v_accvgpr_read v5, a53 // (op 9405) 
v_accvgpr_read v6, a54 // (op 9405) 
v_accvgpr_read v7, a55 // (op 9405) 
v_mul_f32 v14, v11, v4 // (op 9405) 
v_mul_f32 v15, v11, v5 // (op 9405) 
v_mul_f32 v16, v11, v6 // (op 9405) 
v_mul_f32 v17, v11, v7 // (op 9405) 



s_lshl_b32 s0, s4, 6 // (op 9397) 
s_add_u32 s0, s0, 192 // (op 9397) 
v_add_u32 v0, v13, s0 // (op 9397) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 9397) Store value


v_accvgpr_read v4, a56 // (op 9410) 
v_accvgpr_read v5, a57 // (op 9410) 
v_accvgpr_read v6, a58 // (op 9410) 
v_accvgpr_read v7, a59 // (op 9410) 
v_mul_f32 v14, v11, v4 // (op 9410) 
v_mul_f32 v15, v11, v5 // (op 9410) 
v_mul_f32 v16, v11, v6 // (op 9410) 
v_mul_f32 v17, v11, v7 // (op 9410) 



s_lshl_b32 s0, s4, 7 // (op 9399) 
s_add_u32 s0, s0, 192 // (op 9399) 
v_add_u32 v0, v13, s0 // (op 9399) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 9399) Store value


v_accvgpr_read v4, a60 // (op 9415) 
v_accvgpr_read v5, a61 // (op 9415) 
v_accvgpr_read v6, a62 // (op 9415) 
v_accvgpr_read v7, a63 // (op 9415) 
v_mul_f32 v14, v11, v4 // (op 9415) 
v_mul_f32 v15, v11, v5 // (op 9415) 
v_mul_f32 v16, v11, v6 // (op 9415) 
v_mul_f32 v17, v11, v7 // (op 9415) 



s_lshl_b32 s0, s4, 6 // (op 9401) 
s_mul_i32 s1, 3, s0 // (op 9401) 
s_add_u32 s1, s1, 192 // (op 9401) 

v_add_u32 v0, v13, s1 // (op 9401) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 9401) Store value


v_accvgpr_read v4, a64 // (op 9460) 
v_accvgpr_read v5, a65 // (op 9460) 
v_accvgpr_read v6, a66 // (op 9460) 
v_accvgpr_read v7, a67 // (op 9460) 
v_mul_f32 v14, v11, v4 // (op 9460) 
v_mul_f32 v15, v11, v5 // (op 9460) 
v_mul_f32 v16, v11, v6 // (op 9460) 
v_mul_f32 v17, v11, v7 // (op 9460) 



v_add_u32 v0, 256, v13 // (op 9463) 

buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 9463) Store value


v_accvgpr_read v4, a68 // (op 9473) 
v_accvgpr_read v5, a69 // (op 9473) 
v_accvgpr_read v6, a70 // (op 9473) 
v_accvgpr_read v7, a71 // (op 9473) 
v_mul_f32 v14, v11, v4 // (op 9473) 
v_mul_f32 v15, v11, v5 // (op 9473) 
v_mul_f32 v16, v11, v6 // (op 9473) 
v_mul_f32 v17, v11, v7 // (op 9473) 



s_lshl_b32 s0, s4, 6 // (op 9465) 
s_add_u32 s0, s0, 256 // (op 9465) 
v_add_u32 v0, v13, s0 // (op 9465) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 9465) Store value


v_accvgpr_read v4, a72 // (op 9478) 
v_accvgpr_read v5, a73 // (op 9478) 
v_accvgpr_read v6, a74 // (op 9478) 
v_accvgpr_read v7, a75 // (op 9478) 
v_mul_f32 v14, v11, v4 // (op 9478) 
v_mul_f32 v15, v11, v5 // (op 9478) 
v_mul_f32 v16, v11, v6 // (op 9478) 
v_mul_f32 v17, v11, v7 // (op 9478) 



s_lshl_b32 s0, s4, 7 // (op 9467) 
s_add_u32 s0, s0, 256 // (op 9467) 
v_add_u32 v0, v13, s0 // (op 9467) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 9467) Store value


v_accvgpr_read v4, a76 // (op 9483) 
v_accvgpr_read v5, a77 // (op 9483) 
v_accvgpr_read v6, a78 // (op 9483) 
v_accvgpr_read v7, a79 // (op 9483) 
v_mul_f32 v14, v11, v4 // (op 9483) 
v_mul_f32 v15, v11, v5 // (op 9483) 
v_mul_f32 v16, v11, v6 // (op 9483) 
v_mul_f32 v17, v11, v7 // (op 9483) 



s_lshl_b32 s0, s4, 6 // (op 9469) 
s_mul_i32 s1, 3, s0 // (op 9469) 
s_add_u32 s1, s1, 256 // (op 9469) 

v_add_u32 v0, v13, s1 // (op 9469) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 9469) Store value


v_accvgpr_read v4, a80 // (op 9528) 
v_accvgpr_read v5, a81 // (op 9528) 
v_accvgpr_read v6, a82 // (op 9528) 
v_accvgpr_read v7, a83 // (op 9528) 
v_mul_f32 v14, v11, v4 // (op 9528) 
v_mul_f32 v15, v11, v5 // (op 9528) 
v_mul_f32 v16, v11, v6 // (op 9528) 
v_mul_f32 v17, v11, v7 // (op 9528) 



v_add_u32 v0, 320, v13 // (op 9531) 

buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 9531) Store value


v_accvgpr_read v4, a84 // (op 9541) 
v_accvgpr_read v5, a85 // (op 9541) 
v_accvgpr_read v6, a86 // (op 9541) 
v_accvgpr_read v7, a87 // (op 9541) 
v_mul_f32 v14, v11, v4 // (op 9541) 
v_mul_f32 v15, v11, v5 // (op 9541) 
v_mul_f32 v16, v11, v6 // (op 9541) 
v_mul_f32 v17, v11, v7 // (op 9541) 



s_lshl_b32 s0, s4, 6 // (op 9533) 
s_add_u32 s0, s0, 320 // (op 9533) 
v_add_u32 v0, v13, s0 // (op 9533) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 9533) Store value


v_accvgpr_read v4, a88 // (op 9546) 
v_accvgpr_read v5, a89 // (op 9546) 
v_accvgpr_read v6, a90 // (op 9546) 
v_accvgpr_read v7, a91 // (op 9546) 
v_mul_f32 v14, v11, v4 // (op 9546) 
v_mul_f32 v15, v11, v5 // (op 9546) 
v_mul_f32 v16, v11, v6 // (op 9546) 
v_mul_f32 v17, v11, v7 // (op 9546) 



s_lshl_b32 s0, s4, 7 // (op 9535) 
s_add_u32 s0, s0, 320 // (op 9535) 
v_add_u32 v0, v13, s0 // (op 9535) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 9535) Store value


v_accvgpr_read v4, a92 // (op 9551) 
v_accvgpr_read v5, a93 // (op 9551) 
v_accvgpr_read v6, a94 // (op 9551) 
v_accvgpr_read v7, a95 // (op 9551) 
v_mul_f32 v14, v11, v4 // (op 9551) 
v_mul_f32 v15, v11, v5 // (op 9551) 
v_mul_f32 v16, v11, v6 // (op 9551) 
v_mul_f32 v17, v11, v7 // (op 9551) 



s_lshl_b32 s0, s4, 6 // (op 9537) 
s_mul_i32 s1, 3, s0 // (op 9537) 
s_add_u32 s1, s1, 320 // (op 9537) 

v_add_u32 v0, v13, s1 // (op 9537) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 9537) Store value


v_accvgpr_read v4, a96 // (op 9596) 
v_accvgpr_read v5, a97 // (op 9596) 
v_accvgpr_read v6, a98 // (op 9596) 
v_accvgpr_read v7, a99 // (op 9596) 
v_mul_f32 v14, v11, v4 // (op 9596) 
v_mul_f32 v15, v11, v5 // (op 9596) 
v_mul_f32 v16, v11, v6 // (op 9596) 
v_mul_f32 v17, v11, v7 // (op 9596) 



v_add_u32 v0, 384, v13 // (op 9599) 

buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 9599) Store value


v_accvgpr_read v4, a100 // (op 9609) 
v_accvgpr_read v5, a101 // (op 9609) 
v_accvgpr_read v6, a102 // (op 9609) 
v_accvgpr_read v7, a103 // (op 9609) 
v_mul_f32 v14, v11, v4 // (op 9609) 
v_mul_f32 v15, v11, v5 // (op 9609) 
v_mul_f32 v16, v11, v6 // (op 9609) 
v_mul_f32 v17, v11, v7 // (op 9609) 



s_lshl_b32 s0, s4, 6 // (op 9601) 
s_add_u32 s0, s0, 384 // (op 9601) 
v_add_u32 v0, v13, s0 // (op 9601) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 9601) Store value


v_accvgpr_read v4, a104 // (op 9614) 
v_accvgpr_read v5, a105 // (op 9614) 
v_accvgpr_read v6, a106 // (op 9614) 
v_accvgpr_read v7, a107 // (op 9614) 
v_mul_f32 v14, v11, v4 // (op 9614) 
v_mul_f32 v15, v11, v5 // (op 9614) 
v_mul_f32 v16, v11, v6 // (op 9614) 
v_mul_f32 v17, v11, v7 // (op 9614) 



s_lshl_b32 s0, s4, 7 // (op 9603) 
s_add_u32 s0, s0, 384 // (op 9603) 
v_add_u32 v0, v13, s0 // (op 9603) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 9603) Store value


v_accvgpr_read v4, a108 // (op 9619) 
v_accvgpr_read v5, a109 // (op 9619) 
v_accvgpr_read v6, a110 // (op 9619) 
v_accvgpr_read v7, a111 // (op 9619) 
v_mul_f32 v14, v11, v4 // (op 9619) 
v_mul_f32 v15, v11, v5 // (op 9619) 
v_mul_f32 v16, v11, v6 // (op 9619) 
v_mul_f32 v17, v11, v7 // (op 9619) 



s_lshl_b32 s0, s4, 6 // (op 9605) 
s_mul_i32 s1, 3, s0 // (op 9605) 
s_add_u32 s1, s1, 384 // (op 9605) 

v_add_u32 v0, v13, s1 // (op 9605) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 9605) Store value


v_accvgpr_read v4, a112 // (op 9664) 
v_accvgpr_read v5, a113 // (op 9664) 
v_accvgpr_read v6, a114 // (op 9664) 
v_accvgpr_read v7, a115 // (op 9664) 
v_mul_f32 v14, v11, v4 // (op 9664) 
v_mul_f32 v15, v11, v5 // (op 9664) 
v_mul_f32 v16, v11, v6 // (op 9664) 
v_mul_f32 v17, v11, v7 // (op 9664) 



v_add_u32 v0, 448, v13 // (op 9667) 

buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 9667) Store value


v_accvgpr_read v4, a116 // (op 9677) 
v_accvgpr_read v5, a117 // (op 9677) 
v_accvgpr_read v6, a118 // (op 9677) 
v_accvgpr_read v7, a119 // (op 9677) 
v_mul_f32 v14, v11, v4 // (op 9677) 
v_mul_f32 v15, v11, v5 // (op 9677) 
v_mul_f32 v16, v11, v6 // (op 9677) 
v_mul_f32 v17, v11, v7 // (op 9677) 



s_lshl_b32 s0, s4, 6 // (op 9669) 
s_add_u32 s0, s0, 448 // (op 9669) 
v_add_u32 v0, v13, s0 // (op 9669) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 9669) Store value


v_accvgpr_read v4, a120 // (op 9682) 
v_accvgpr_read v5, a121 // (op 9682) 
v_accvgpr_read v6, a122 // (op 9682) 
v_accvgpr_read v7, a123 // (op 9682) 
v_mul_f32 v14, v11, v4 // (op 9682) 
v_mul_f32 v15, v11, v5 // (op 9682) 
v_mul_f32 v16, v11, v6 // (op 9682) 
v_mul_f32 v17, v11, v7 // (op 9682) 



s_lshl_b32 s0, s4, 7 // (op 9671) 
s_add_u32 s0, s0, 448 // (op 9671) 
v_add_u32 v0, v13, s0 // (op 9671) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 9671) Store value


v_accvgpr_read v4, a124 // (op 9687) 
v_accvgpr_read v5, a125 // (op 9687) 
v_accvgpr_read v6, a126 // (op 9687) 
v_accvgpr_read v7, a127 // (op 9687) 
v_mul_f32 v14, v11, v4 // (op 9687) 
v_mul_f32 v15, v11, v5 // (op 9687) 
v_mul_f32 v16, v11, v6 // (op 9687) 
v_mul_f32 v17, v11, v7 // (op 9687) 



s_lshl_b32 s0, s4, 6 // (op 9673) 
s_mul_i32 s1, 3, s0 // (op 9673) 
s_add_u32 s1, s1, 448 // (op 9673) 

v_add_u32 v0, v13, s1 // (op 9673) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 9673) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_WGM__ConditionalBottom_0_beta_114 // (op 114) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_WGM__ConditionalBottom_0_beta_114
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_WGM__ConditionalFalse_0_beta_114:


v_add_u32 v0, 448, v12 // (op 10403) 

buffer_load_dwordx4 v[4:7], v0, s[52:55], 0 offen // (op 10403) Load value

v_accvgpr_read v14, a112 // (op 10391) 
v_accvgpr_read v15, a113 // (op 10391) 
v_accvgpr_read v16, a114 // (op 10391) 
v_accvgpr_read v17, a115 // (op 10391) 
v_mul_f32 v18, v11, v14 // (op 10391) 
v_mul_f32 v19, v11, v15 // (op 10391) 
v_mul_f32 v20, v11, v16 // (op 10391) 
v_mul_f32 v21, v11, v17 // (op 10391) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 10402) 
v_mul_f32 v15, v9, v5 // (op 10402) 
v_mul_f32 v16, v9, v6 // (op 10402) 
v_mul_f32 v17, v9, v7 // (op 10402) 

v_add_f32 v4, v14, v18 // (op 10392) 
v_add_f32 v5, v15, v19 // (op 10392) 
v_add_f32 v6, v16, v20 // (op 10392) 
v_add_f32 v7, v17, v21 // (op 10392) 




v_add_u32 v0, 448, v13 // (op 10394) 

buffer_store_dwordx4 v[4:7], v0, s[56:59], 0 offen // (op 10394) Store value




s_lshl_b32 s0, s2, 6 // (op 10407) 
s_add_u32 s0, s0, 448 // (op 10407) 
v_add_u32 v0, v12, s0 // (op 10407) 


buffer_load_dwordx4 v[4:7], v0, s[52:55], 0 offen // (op 10407) Load value

v_accvgpr_read v14, a116 // (op 10417) 
v_accvgpr_read v15, a117 // (op 10417) 
v_accvgpr_read v16, a118 // (op 10417) 
v_accvgpr_read v17, a119 // (op 10417) 
v_mul_f32 v18, v11, v14 // (op 10417) 
v_mul_f32 v19, v11, v15 // (op 10417) 
v_mul_f32 v20, v11, v16 // (op 10417) 
v_mul_f32 v21, v11, v17 // (op 10417) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 10405) 
v_mul_f32 v15, v9, v5 // (op 10405) 
v_mul_f32 v16, v9, v6 // (op 10405) 
v_mul_f32 v17, v9, v7 // (op 10405) 

v_add_f32 v4, v14, v18 // (op 10406) 
v_add_f32 v5, v15, v19 // (op 10406) 
v_add_f32 v6, v16, v20 // (op 10406) 
v_add_f32 v7, v17, v21 // (op 10406) 




s_lshl_b32 s0, s4, 6 // (op 10396) 
s_add_u32 s0, s0, 448 // (op 10396) 
v_add_u32 v0, v13, s0 // (op 10396) 


buffer_store_dwordx4 v[4:7], v0, s[56:59], 0 offen // (op 10396) Store value




s_lshl_b32 s0, s2, 7 // (op 10411) 
s_add_u32 s0, s0, 448 // (op 10411) 
v_add_u32 v0, v12, s0 // (op 10411) 


buffer_load_dwordx4 v[4:7], v0, s[52:55], 0 offen // (op 10411) Load value

v_accvgpr_read v14, a120 // (op 10419) 
v_accvgpr_read v15, a121 // (op 10419) 
v_accvgpr_read v16, a122 // (op 10419) 
v_accvgpr_read v17, a123 // (op 10419) 
v_mul_f32 v18, v11, v14 // (op 10419) 
v_mul_f32 v19, v11, v15 // (op 10419) 
v_mul_f32 v20, v11, v16 // (op 10419) 
v_mul_f32 v21, v11, v17 // (op 10419) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 10409) 
v_mul_f32 v15, v9, v5 // (op 10409) 
v_mul_f32 v16, v9, v6 // (op 10409) 
v_mul_f32 v17, v9, v7 // (op 10409) 

v_add_f32 v4, v14, v18 // (op 10410) 
v_add_f32 v5, v15, v19 // (op 10410) 
v_add_f32 v6, v16, v20 // (op 10410) 
v_add_f32 v7, v17, v21 // (op 10410) 




s_lshl_b32 s0, s4, 7 // (op 10398) 
s_add_u32 s0, s0, 448 // (op 10398) 
v_add_u32 v0, v13, s0 // (op 10398) 


buffer_store_dwordx4 v[4:7], v0, s[56:59], 0 offen // (op 10398) Store value




s_lshl_b32 s0, s2, 6 // (op 10415) 
s_mul_i32 s1, 3, s0 // (op 10415) 
s_add_u32 s1, s1, 448 // (op 10415) 

v_add_u32 v0, v12, s1 // (op 10415) 


buffer_load_dwordx4 v[4:7], v0, s[52:55], 0 offen // (op 10415) Load value

v_accvgpr_read v14, a124 // (op 10421) 
v_accvgpr_read v15, a125 // (op 10421) 
v_accvgpr_read v16, a126 // (op 10421) 
v_accvgpr_read v17, a127 // (op 10421) 
v_mul_f32 v18, v11, v14 // (op 10421) 
v_mul_f32 v19, v11, v15 // (op 10421) 
v_mul_f32 v20, v11, v16 // (op 10421) 
v_mul_f32 v21, v11, v17 // (op 10421) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 10413) 
v_mul_f32 v15, v9, v5 // (op 10413) 
v_mul_f32 v16, v9, v6 // (op 10413) 
v_mul_f32 v17, v9, v7 // (op 10413) 

v_add_f32 v4, v14, v18 // (op 10414) 
v_add_f32 v5, v15, v19 // (op 10414) 
v_add_f32 v6, v16, v20 // (op 10414) 
v_add_f32 v7, v17, v21 // (op 10414) 




s_lshl_b32 s0, s4, 6 // (op 10400) 
s_mul_i32 s1, 3, s0 // (op 10400) 
s_add_u32 s1, s1, 448 // (op 10400) 

v_add_u32 v0, v13, s1 // (op 10400) 


buffer_store_dwordx4 v[4:7], v0, s[56:59], 0 offen // (op 10400) Store value




v_add_u32 v0, 384, v12 // (op 10324) 

s_nop 0
buffer_load_dwordx4 v[4:7], v0, s[52:55], 0 offen // (op 10324) Load value

v_accvgpr_read v14, a96 // (op 10312) 
v_accvgpr_read v15, a97 // (op 10312) 
v_accvgpr_read v16, a98 // (op 10312) 
v_accvgpr_read v17, a99 // (op 10312) 
v_mul_f32 v18, v11, v14 // (op 10312) 
v_mul_f32 v19, v11, v15 // (op 10312) 
v_mul_f32 v20, v11, v16 // (op 10312) 
v_mul_f32 v21, v11, v17 // (op 10312) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 10323) 
v_mul_f32 v15, v9, v5 // (op 10323) 
v_mul_f32 v16, v9, v6 // (op 10323) 
v_mul_f32 v17, v9, v7 // (op 10323) 

v_add_f32 v4, v14, v18 // (op 10313) 
v_add_f32 v5, v15, v19 // (op 10313) 
v_add_f32 v6, v16, v20 // (op 10313) 
v_add_f32 v7, v17, v21 // (op 10313) 




v_add_u32 v0, 384, v13 // (op 10315) 

buffer_store_dwordx4 v[4:7], v0, s[56:59], 0 offen // (op 10315) Store value




s_lshl_b32 s0, s2, 6 // (op 10328) 
s_add_u32 s0, s0, 384 // (op 10328) 
v_add_u32 v0, v12, s0 // (op 10328) 


buffer_load_dwordx4 v[4:7], v0, s[52:55], 0 offen // (op 10328) Load value

v_accvgpr_read v14, a100 // (op 10338) 
v_accvgpr_read v15, a101 // (op 10338) 
v_accvgpr_read v16, a102 // (op 10338) 
v_accvgpr_read v17, a103 // (op 10338) 
v_mul_f32 v18, v11, v14 // (op 10338) 
v_mul_f32 v19, v11, v15 // (op 10338) 
v_mul_f32 v20, v11, v16 // (op 10338) 
v_mul_f32 v21, v11, v17 // (op 10338) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 10326) 
v_mul_f32 v15, v9, v5 // (op 10326) 
v_mul_f32 v16, v9, v6 // (op 10326) 
v_mul_f32 v17, v9, v7 // (op 10326) 

v_add_f32 v4, v14, v18 // (op 10327) 
v_add_f32 v5, v15, v19 // (op 10327) 
v_add_f32 v6, v16, v20 // (op 10327) 
v_add_f32 v7, v17, v21 // (op 10327) 




s_lshl_b32 s0, s4, 6 // (op 10317) 
s_add_u32 s0, s0, 384 // (op 10317) 
v_add_u32 v0, v13, s0 // (op 10317) 


buffer_store_dwordx4 v[4:7], v0, s[56:59], 0 offen // (op 10317) Store value




s_lshl_b32 s0, s2, 7 // (op 10332) 
s_add_u32 s0, s0, 384 // (op 10332) 
v_add_u32 v0, v12, s0 // (op 10332) 


buffer_load_dwordx4 v[4:7], v0, s[52:55], 0 offen // (op 10332) Load value

v_accvgpr_read v14, a104 // (op 10340) 
v_accvgpr_read v15, a105 // (op 10340) 
v_accvgpr_read v16, a106 // (op 10340) 
v_accvgpr_read v17, a107 // (op 10340) 
v_mul_f32 v18, v11, v14 // (op 10340) 
v_mul_f32 v19, v11, v15 // (op 10340) 
v_mul_f32 v20, v11, v16 // (op 10340) 
v_mul_f32 v21, v11, v17 // (op 10340) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 10330) 
v_mul_f32 v15, v9, v5 // (op 10330) 
v_mul_f32 v16, v9, v6 // (op 10330) 
v_mul_f32 v17, v9, v7 // (op 10330) 

v_add_f32 v4, v14, v18 // (op 10331) 
v_add_f32 v5, v15, v19 // (op 10331) 
v_add_f32 v6, v16, v20 // (op 10331) 
v_add_f32 v7, v17, v21 // (op 10331) 




s_lshl_b32 s0, s4, 7 // (op 10319) 
s_add_u32 s0, s0, 384 // (op 10319) 
v_add_u32 v0, v13, s0 // (op 10319) 


buffer_store_dwordx4 v[4:7], v0, s[56:59], 0 offen // (op 10319) Store value




s_lshl_b32 s0, s2, 6 // (op 10336) 
s_mul_i32 s1, 3, s0 // (op 10336) 
s_add_u32 s1, s1, 384 // (op 10336) 

v_add_u32 v0, v12, s1 // (op 10336) 


buffer_load_dwordx4 v[4:7], v0, s[52:55], 0 offen // (op 10336) Load value

v_accvgpr_read v14, a108 // (op 10342) 
v_accvgpr_read v15, a109 // (op 10342) 
v_accvgpr_read v16, a110 // (op 10342) 
v_accvgpr_read v17, a111 // (op 10342) 
v_mul_f32 v18, v11, v14 // (op 10342) 
v_mul_f32 v19, v11, v15 // (op 10342) 
v_mul_f32 v20, v11, v16 // (op 10342) 
v_mul_f32 v21, v11, v17 // (op 10342) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 10334) 
v_mul_f32 v15, v9, v5 // (op 10334) 
v_mul_f32 v16, v9, v6 // (op 10334) 
v_mul_f32 v17, v9, v7 // (op 10334) 

v_add_f32 v4, v14, v18 // (op 10335) 
v_add_f32 v5, v15, v19 // (op 10335) 
v_add_f32 v6, v16, v20 // (op 10335) 
v_add_f32 v7, v17, v21 // (op 10335) 




s_lshl_b32 s0, s4, 6 // (op 10321) 
s_mul_i32 s1, 3, s0 // (op 10321) 
s_add_u32 s1, s1, 384 // (op 10321) 

v_add_u32 v0, v13, s1 // (op 10321) 


buffer_store_dwordx4 v[4:7], v0, s[56:59], 0 offen // (op 10321) Store value




v_add_u32 v0, 320, v12 // (op 10245) 

s_nop 0
buffer_load_dwordx4 v[4:7], v0, s[52:55], 0 offen // (op 10245) Load value

v_accvgpr_read v14, a80 // (op 10233) 
v_accvgpr_read v15, a81 // (op 10233) 
v_accvgpr_read v16, a82 // (op 10233) 
v_accvgpr_read v17, a83 // (op 10233) 
v_mul_f32 v18, v11, v14 // (op 10233) 
v_mul_f32 v19, v11, v15 // (op 10233) 
v_mul_f32 v20, v11, v16 // (op 10233) 
v_mul_f32 v21, v11, v17 // (op 10233) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 10244) 
v_mul_f32 v15, v9, v5 // (op 10244) 
v_mul_f32 v16, v9, v6 // (op 10244) 
v_mul_f32 v17, v9, v7 // (op 10244) 

v_add_f32 v4, v14, v18 // (op 10234) 
v_add_f32 v5, v15, v19 // (op 10234) 
v_add_f32 v6, v16, v20 // (op 10234) 
v_add_f32 v7, v17, v21 // (op 10234) 




v_add_u32 v0, 320, v13 // (op 10236) 

buffer_store_dwordx4 v[4:7], v0, s[56:59], 0 offen // (op 10236) Store value




s_lshl_b32 s0, s2, 6 // (op 10249) 
s_add_u32 s0, s0, 320 // (op 10249) 
v_add_u32 v0, v12, s0 // (op 10249) 


buffer_load_dwordx4 v[4:7], v0, s[52:55], 0 offen // (op 10249) Load value

v_accvgpr_read v14, a84 // (op 10259) 
v_accvgpr_read v15, a85 // (op 10259) 
v_accvgpr_read v16, a86 // (op 10259) 
v_accvgpr_read v17, a87 // (op 10259) 
v_mul_f32 v18, v11, v14 // (op 10259) 
v_mul_f32 v19, v11, v15 // (op 10259) 
v_mul_f32 v20, v11, v16 // (op 10259) 
v_mul_f32 v21, v11, v17 // (op 10259) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 10247) 
v_mul_f32 v15, v9, v5 // (op 10247) 
v_mul_f32 v16, v9, v6 // (op 10247) 
v_mul_f32 v17, v9, v7 // (op 10247) 

v_add_f32 v4, v14, v18 // (op 10248) 
v_add_f32 v5, v15, v19 // (op 10248) 
v_add_f32 v6, v16, v20 // (op 10248) 
v_add_f32 v7, v17, v21 // (op 10248) 




s_lshl_b32 s0, s4, 6 // (op 10238) 
s_add_u32 s0, s0, 320 // (op 10238) 
v_add_u32 v0, v13, s0 // (op 10238) 


buffer_store_dwordx4 v[4:7], v0, s[56:59], 0 offen // (op 10238) Store value




s_lshl_b32 s0, s2, 7 // (op 10253) 
s_add_u32 s0, s0, 320 // (op 10253) 
v_add_u32 v0, v12, s0 // (op 10253) 


buffer_load_dwordx4 v[4:7], v0, s[52:55], 0 offen // (op 10253) Load value

v_accvgpr_read v14, a88 // (op 10261) 
v_accvgpr_read v15, a89 // (op 10261) 
v_accvgpr_read v16, a90 // (op 10261) 
v_accvgpr_read v17, a91 // (op 10261) 
v_mul_f32 v18, v11, v14 // (op 10261) 
v_mul_f32 v19, v11, v15 // (op 10261) 
v_mul_f32 v20, v11, v16 // (op 10261) 
v_mul_f32 v21, v11, v17 // (op 10261) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 10251) 
v_mul_f32 v15, v9, v5 // (op 10251) 
v_mul_f32 v16, v9, v6 // (op 10251) 
v_mul_f32 v17, v9, v7 // (op 10251) 

v_add_f32 v4, v14, v18 // (op 10252) 
v_add_f32 v5, v15, v19 // (op 10252) 
v_add_f32 v6, v16, v20 // (op 10252) 
v_add_f32 v7, v17, v21 // (op 10252) 




s_lshl_b32 s0, s4, 7 // (op 10240) 
s_add_u32 s0, s0, 320 // (op 10240) 
v_add_u32 v0, v13, s0 // (op 10240) 


buffer_store_dwordx4 v[4:7], v0, s[56:59], 0 offen // (op 10240) Store value




s_lshl_b32 s0, s2, 6 // (op 10257) 
s_mul_i32 s1, 3, s0 // (op 10257) 
s_add_u32 s1, s1, 320 // (op 10257) 

v_add_u32 v0, v12, s1 // (op 10257) 


buffer_load_dwordx4 v[4:7], v0, s[52:55], 0 offen // (op 10257) Load value

v_accvgpr_read v14, a92 // (op 10263) 
v_accvgpr_read v15, a93 // (op 10263) 
v_accvgpr_read v16, a94 // (op 10263) 
v_accvgpr_read v17, a95 // (op 10263) 
v_mul_f32 v18, v11, v14 // (op 10263) 
v_mul_f32 v19, v11, v15 // (op 10263) 
v_mul_f32 v20, v11, v16 // (op 10263) 
v_mul_f32 v21, v11, v17 // (op 10263) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 10255) 
v_mul_f32 v15, v9, v5 // (op 10255) 
v_mul_f32 v16, v9, v6 // (op 10255) 
v_mul_f32 v17, v9, v7 // (op 10255) 

v_add_f32 v4, v14, v18 // (op 10256) 
v_add_f32 v5, v15, v19 // (op 10256) 
v_add_f32 v6, v16, v20 // (op 10256) 
v_add_f32 v7, v17, v21 // (op 10256) 




s_lshl_b32 s0, s4, 6 // (op 10242) 
s_mul_i32 s1, 3, s0 // (op 10242) 
s_add_u32 s1, s1, 320 // (op 10242) 

v_add_u32 v0, v13, s1 // (op 10242) 


buffer_store_dwordx4 v[4:7], v0, s[56:59], 0 offen // (op 10242) Store value




v_add_u32 v0, 256, v12 // (op 10166) 

s_nop 0
buffer_load_dwordx4 v[4:7], v0, s[52:55], 0 offen // (op 10166) Load value

v_accvgpr_read v14, a64 // (op 10154) 
v_accvgpr_read v15, a65 // (op 10154) 
v_accvgpr_read v16, a66 // (op 10154) 
v_accvgpr_read v17, a67 // (op 10154) 
v_mul_f32 v18, v11, v14 // (op 10154) 
v_mul_f32 v19, v11, v15 // (op 10154) 
v_mul_f32 v20, v11, v16 // (op 10154) 
v_mul_f32 v21, v11, v17 // (op 10154) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 10165) 
v_mul_f32 v15, v9, v5 // (op 10165) 
v_mul_f32 v16, v9, v6 // (op 10165) 
v_mul_f32 v17, v9, v7 // (op 10165) 

v_add_f32 v4, v14, v18 // (op 10155) 
v_add_f32 v5, v15, v19 // (op 10155) 
v_add_f32 v6, v16, v20 // (op 10155) 
v_add_f32 v7, v17, v21 // (op 10155) 




v_add_u32 v0, 256, v13 // (op 10157) 

buffer_store_dwordx4 v[4:7], v0, s[56:59], 0 offen // (op 10157) Store value




s_lshl_b32 s0, s2, 6 // (op 10170) 
s_add_u32 s0, s0, 256 // (op 10170) 
v_add_u32 v0, v12, s0 // (op 10170) 


buffer_load_dwordx4 v[4:7], v0, s[52:55], 0 offen // (op 10170) Load value

v_accvgpr_read v14, a68 // (op 10180) 
v_accvgpr_read v15, a69 // (op 10180) 
v_accvgpr_read v16, a70 // (op 10180) 
v_accvgpr_read v17, a71 // (op 10180) 
v_mul_f32 v18, v11, v14 // (op 10180) 
v_mul_f32 v19, v11, v15 // (op 10180) 
v_mul_f32 v20, v11, v16 // (op 10180) 
v_mul_f32 v21, v11, v17 // (op 10180) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 10168) 
v_mul_f32 v15, v9, v5 // (op 10168) 
v_mul_f32 v16, v9, v6 // (op 10168) 
v_mul_f32 v17, v9, v7 // (op 10168) 

v_add_f32 v4, v14, v18 // (op 10169) 
v_add_f32 v5, v15, v19 // (op 10169) 
v_add_f32 v6, v16, v20 // (op 10169) 
v_add_f32 v7, v17, v21 // (op 10169) 




s_lshl_b32 s0, s4, 6 // (op 10159) 
s_add_u32 s0, s0, 256 // (op 10159) 
v_add_u32 v0, v13, s0 // (op 10159) 


buffer_store_dwordx4 v[4:7], v0, s[56:59], 0 offen // (op 10159) Store value




s_lshl_b32 s0, s2, 7 // (op 10174) 
s_add_u32 s0, s0, 256 // (op 10174) 
v_add_u32 v0, v12, s0 // (op 10174) 


buffer_load_dwordx4 v[4:7], v0, s[52:55], 0 offen // (op 10174) Load value

v_accvgpr_read v14, a72 // (op 10182) 
v_accvgpr_read v15, a73 // (op 10182) 
v_accvgpr_read v16, a74 // (op 10182) 
v_accvgpr_read v17, a75 // (op 10182) 
v_mul_f32 v18, v11, v14 // (op 10182) 
v_mul_f32 v19, v11, v15 // (op 10182) 
v_mul_f32 v20, v11, v16 // (op 10182) 
v_mul_f32 v21, v11, v17 // (op 10182) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 10172) 
v_mul_f32 v15, v9, v5 // (op 10172) 
v_mul_f32 v16, v9, v6 // (op 10172) 
v_mul_f32 v17, v9, v7 // (op 10172) 

v_add_f32 v4, v14, v18 // (op 10173) 
v_add_f32 v5, v15, v19 // (op 10173) 
v_add_f32 v6, v16, v20 // (op 10173) 
v_add_f32 v7, v17, v21 // (op 10173) 




s_lshl_b32 s0, s4, 7 // (op 10161) 
s_add_u32 s0, s0, 256 // (op 10161) 
v_add_u32 v0, v13, s0 // (op 10161) 


buffer_store_dwordx4 v[4:7], v0, s[56:59], 0 offen // (op 10161) Store value




s_lshl_b32 s0, s2, 6 // (op 10178) 
s_mul_i32 s1, 3, s0 // (op 10178) 
s_add_u32 s1, s1, 256 // (op 10178) 

v_add_u32 v0, v12, s1 // (op 10178) 


buffer_load_dwordx4 v[4:7], v0, s[52:55], 0 offen // (op 10178) Load value

v_accvgpr_read v14, a76 // (op 10184) 
v_accvgpr_read v15, a77 // (op 10184) 
v_accvgpr_read v16, a78 // (op 10184) 
v_accvgpr_read v17, a79 // (op 10184) 
v_mul_f32 v18, v11, v14 // (op 10184) 
v_mul_f32 v19, v11, v15 // (op 10184) 
v_mul_f32 v20, v11, v16 // (op 10184) 
v_mul_f32 v21, v11, v17 // (op 10184) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 10176) 
v_mul_f32 v15, v9, v5 // (op 10176) 
v_mul_f32 v16, v9, v6 // (op 10176) 
v_mul_f32 v17, v9, v7 // (op 10176) 

v_add_f32 v4, v14, v18 // (op 10177) 
v_add_f32 v5, v15, v19 // (op 10177) 
v_add_f32 v6, v16, v20 // (op 10177) 
v_add_f32 v7, v17, v21 // (op 10177) 




s_lshl_b32 s0, s4, 6 // (op 10163) 
s_mul_i32 s1, 3, s0 // (op 10163) 
s_add_u32 s1, s1, 256 // (op 10163) 

v_add_u32 v0, v13, s1 // (op 10163) 


buffer_store_dwordx4 v[4:7], v0, s[56:59], 0 offen // (op 10163) Store value




v_add_u32 v0, 192, v12 // (op 10087) 

s_nop 0
buffer_load_dwordx4 v[4:7], v0, s[52:55], 0 offen // (op 10087) Load value

v_accvgpr_read v14, a48 // (op 10075) 
v_accvgpr_read v15, a49 // (op 10075) 
v_accvgpr_read v16, a50 // (op 10075) 
v_accvgpr_read v17, a51 // (op 10075) 
v_mul_f32 v18, v11, v14 // (op 10075) 
v_mul_f32 v19, v11, v15 // (op 10075) 
v_mul_f32 v20, v11, v16 // (op 10075) 
v_mul_f32 v21, v11, v17 // (op 10075) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 10086) 
v_mul_f32 v15, v9, v5 // (op 10086) 
v_mul_f32 v16, v9, v6 // (op 10086) 
v_mul_f32 v17, v9, v7 // (op 10086) 

v_add_f32 v4, v14, v18 // (op 10076) 
v_add_f32 v5, v15, v19 // (op 10076) 
v_add_f32 v6, v16, v20 // (op 10076) 
v_add_f32 v7, v17, v21 // (op 10076) 




v_add_u32 v0, 192, v13 // (op 10078) 

buffer_store_dwordx4 v[4:7], v0, s[56:59], 0 offen // (op 10078) Store value




s_lshl_b32 s0, s2, 6 // (op 10091) 
s_add_u32 s0, s0, 192 // (op 10091) 
v_add_u32 v0, v12, s0 // (op 10091) 


buffer_load_dwordx4 v[4:7], v0, s[52:55], 0 offen // (op 10091) Load value

v_accvgpr_read v14, a52 // (op 10101) 
v_accvgpr_read v15, a53 // (op 10101) 
v_accvgpr_read v16, a54 // (op 10101) 
v_accvgpr_read v17, a55 // (op 10101) 
v_mul_f32 v18, v11, v14 // (op 10101) 
v_mul_f32 v19, v11, v15 // (op 10101) 
v_mul_f32 v20, v11, v16 // (op 10101) 
v_mul_f32 v21, v11, v17 // (op 10101) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 10089) 
v_mul_f32 v15, v9, v5 // (op 10089) 
v_mul_f32 v16, v9, v6 // (op 10089) 
v_mul_f32 v17, v9, v7 // (op 10089) 

v_add_f32 v4, v14, v18 // (op 10090) 
v_add_f32 v5, v15, v19 // (op 10090) 
v_add_f32 v6, v16, v20 // (op 10090) 
v_add_f32 v7, v17, v21 // (op 10090) 




s_lshl_b32 s0, s4, 6 // (op 10080) 
s_add_u32 s0, s0, 192 // (op 10080) 
v_add_u32 v0, v13, s0 // (op 10080) 


buffer_store_dwordx4 v[4:7], v0, s[56:59], 0 offen // (op 10080) Store value




s_lshl_b32 s0, s2, 7 // (op 10095) 
s_add_u32 s0, s0, 192 // (op 10095) 
v_add_u32 v0, v12, s0 // (op 10095) 


buffer_load_dwordx4 v[4:7], v0, s[52:55], 0 offen // (op 10095) Load value

v_accvgpr_read v14, a56 // (op 10103) 
v_accvgpr_read v15, a57 // (op 10103) 
v_accvgpr_read v16, a58 // (op 10103) 
v_accvgpr_read v17, a59 // (op 10103) 
v_mul_f32 v18, v11, v14 // (op 10103) 
v_mul_f32 v19, v11, v15 // (op 10103) 
v_mul_f32 v20, v11, v16 // (op 10103) 
v_mul_f32 v21, v11, v17 // (op 10103) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 10093) 
v_mul_f32 v15, v9, v5 // (op 10093) 
v_mul_f32 v16, v9, v6 // (op 10093) 
v_mul_f32 v17, v9, v7 // (op 10093) 

v_add_f32 v4, v14, v18 // (op 10094) 
v_add_f32 v5, v15, v19 // (op 10094) 
v_add_f32 v6, v16, v20 // (op 10094) 
v_add_f32 v7, v17, v21 // (op 10094) 




s_lshl_b32 s0, s4, 7 // (op 10082) 
s_add_u32 s0, s0, 192 // (op 10082) 
v_add_u32 v0, v13, s0 // (op 10082) 


buffer_store_dwordx4 v[4:7], v0, s[56:59], 0 offen // (op 10082) Store value




s_lshl_b32 s0, s2, 6 // (op 10099) 
s_mul_i32 s1, 3, s0 // (op 10099) 
s_add_u32 s1, s1, 192 // (op 10099) 

v_add_u32 v0, v12, s1 // (op 10099) 


buffer_load_dwordx4 v[4:7], v0, s[52:55], 0 offen // (op 10099) Load value

v_accvgpr_read v14, a60 // (op 10105) 
v_accvgpr_read v15, a61 // (op 10105) 
v_accvgpr_read v16, a62 // (op 10105) 
v_accvgpr_read v17, a63 // (op 10105) 
v_mul_f32 v18, v11, v14 // (op 10105) 
v_mul_f32 v19, v11, v15 // (op 10105) 
v_mul_f32 v20, v11, v16 // (op 10105) 
v_mul_f32 v21, v11, v17 // (op 10105) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 10097) 
v_mul_f32 v15, v9, v5 // (op 10097) 
v_mul_f32 v16, v9, v6 // (op 10097) 
v_mul_f32 v17, v9, v7 // (op 10097) 

v_add_f32 v4, v14, v18 // (op 10098) 
v_add_f32 v5, v15, v19 // (op 10098) 
v_add_f32 v6, v16, v20 // (op 10098) 
v_add_f32 v7, v17, v21 // (op 10098) 




s_lshl_b32 s0, s4, 6 // (op 10084) 
s_mul_i32 s1, 3, s0 // (op 10084) 
s_add_u32 s1, s1, 192 // (op 10084) 

v_add_u32 v0, v13, s1 // (op 10084) 


buffer_store_dwordx4 v[4:7], v0, s[56:59], 0 offen // (op 10084) Store value




v_add_u32 v0, 128, v12 // (op 10008) 

s_nop 0
buffer_load_dwordx4 v[4:7], v0, s[52:55], 0 offen // (op 10008) Load value

v_accvgpr_read v14, a32 // (op 9996) 
v_accvgpr_read v15, a33 // (op 9996) 
v_accvgpr_read v16, a34 // (op 9996) 
v_accvgpr_read v17, a35 // (op 9996) 
v_mul_f32 v18, v11, v14 // (op 9996) 
v_mul_f32 v19, v11, v15 // (op 9996) 
v_mul_f32 v20, v11, v16 // (op 9996) 
v_mul_f32 v21, v11, v17 // (op 9996) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 10007) 
v_mul_f32 v15, v9, v5 // (op 10007) 
v_mul_f32 v16, v9, v6 // (op 10007) 
v_mul_f32 v17, v9, v7 // (op 10007) 

v_add_f32 v4, v14, v18 // (op 9997) 
v_add_f32 v5, v15, v19 // (op 9997) 
v_add_f32 v6, v16, v20 // (op 9997) 
v_add_f32 v7, v17, v21 // (op 9997) 




v_add_u32 v0, 128, v13 // (op 9999) 

buffer_store_dwordx4 v[4:7], v0, s[56:59], 0 offen // (op 9999) Store value




s_lshl_b32 s0, s2, 6 // (op 10012) 
s_add_u32 s0, s0, 128 // (op 10012) 
v_add_u32 v0, v12, s0 // (op 10012) 


buffer_load_dwordx4 v[4:7], v0, s[52:55], 0 offen // (op 10012) Load value

v_accvgpr_read v14, a36 // (op 10022) 
v_accvgpr_read v15, a37 // (op 10022) 
v_accvgpr_read v16, a38 // (op 10022) 
v_accvgpr_read v17, a39 // (op 10022) 
v_mul_f32 v18, v11, v14 // (op 10022) 
v_mul_f32 v19, v11, v15 // (op 10022) 
v_mul_f32 v20, v11, v16 // (op 10022) 
v_mul_f32 v21, v11, v17 // (op 10022) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 10010) 
v_mul_f32 v15, v9, v5 // (op 10010) 
v_mul_f32 v16, v9, v6 // (op 10010) 
v_mul_f32 v17, v9, v7 // (op 10010) 

v_add_f32 v4, v14, v18 // (op 10011) 
v_add_f32 v5, v15, v19 // (op 10011) 
v_add_f32 v6, v16, v20 // (op 10011) 
v_add_f32 v7, v17, v21 // (op 10011) 




s_lshl_b32 s0, s4, 6 // (op 10001) 
s_add_u32 s0, s0, 128 // (op 10001) 
v_add_u32 v0, v13, s0 // (op 10001) 


buffer_store_dwordx4 v[4:7], v0, s[56:59], 0 offen // (op 10001) Store value




s_lshl_b32 s0, s2, 7 // (op 10016) 
s_add_u32 s0, s0, 128 // (op 10016) 
v_add_u32 v0, v12, s0 // (op 10016) 


buffer_load_dwordx4 v[4:7], v0, s[52:55], 0 offen // (op 10016) Load value

v_accvgpr_read v14, a40 // (op 10024) 
v_accvgpr_read v15, a41 // (op 10024) 
v_accvgpr_read v16, a42 // (op 10024) 
v_accvgpr_read v17, a43 // (op 10024) 
v_mul_f32 v18, v11, v14 // (op 10024) 
v_mul_f32 v19, v11, v15 // (op 10024) 
v_mul_f32 v20, v11, v16 // (op 10024) 
v_mul_f32 v21, v11, v17 // (op 10024) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 10014) 
v_mul_f32 v15, v9, v5 // (op 10014) 
v_mul_f32 v16, v9, v6 // (op 10014) 
v_mul_f32 v17, v9, v7 // (op 10014) 

v_add_f32 v4, v14, v18 // (op 10015) 
v_add_f32 v5, v15, v19 // (op 10015) 
v_add_f32 v6, v16, v20 // (op 10015) 
v_add_f32 v7, v17, v21 // (op 10015) 




s_lshl_b32 s0, s4, 7 // (op 10003) 
s_add_u32 s0, s0, 128 // (op 10003) 
v_add_u32 v0, v13, s0 // (op 10003) 


buffer_store_dwordx4 v[4:7], v0, s[56:59], 0 offen // (op 10003) Store value




s_lshl_b32 s0, s2, 6 // (op 10020) 
s_mul_i32 s1, 3, s0 // (op 10020) 
s_add_u32 s1, s1, 128 // (op 10020) 

v_add_u32 v0, v12, s1 // (op 10020) 


buffer_load_dwordx4 v[4:7], v0, s[52:55], 0 offen // (op 10020) Load value

v_accvgpr_read v14, a44 // (op 10026) 
v_accvgpr_read v15, a45 // (op 10026) 
v_accvgpr_read v16, a46 // (op 10026) 
v_accvgpr_read v17, a47 // (op 10026) 
v_mul_f32 v18, v11, v14 // (op 10026) 
v_mul_f32 v19, v11, v15 // (op 10026) 
v_mul_f32 v20, v11, v16 // (op 10026) 
v_mul_f32 v21, v11, v17 // (op 10026) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 10018) 
v_mul_f32 v15, v9, v5 // (op 10018) 
v_mul_f32 v16, v9, v6 // (op 10018) 
v_mul_f32 v17, v9, v7 // (op 10018) 

v_add_f32 v4, v14, v18 // (op 10019) 
v_add_f32 v5, v15, v19 // (op 10019) 
v_add_f32 v6, v16, v20 // (op 10019) 
v_add_f32 v7, v17, v21 // (op 10019) 




s_lshl_b32 s0, s4, 6 // (op 10005) 
s_mul_i32 s1, 3, s0 // (op 10005) 
s_add_u32 s1, s1, 128 // (op 10005) 

v_add_u32 v0, v13, s1 // (op 10005) 


buffer_store_dwordx4 v[4:7], v0, s[56:59], 0 offen // (op 10005) Store value




v_add_u32 v0, 64, v12 // (op 9929) 

s_nop 0
buffer_load_dwordx4 v[4:7], v0, s[52:55], 0 offen // (op 9929) Load value

v_accvgpr_read v14, a16 // (op 9917) 
v_accvgpr_read v15, a17 // (op 9917) 
v_accvgpr_read v16, a18 // (op 9917) 
v_accvgpr_read v17, a19 // (op 9917) 
v_mul_f32 v18, v11, v14 // (op 9917) 
v_mul_f32 v19, v11, v15 // (op 9917) 
v_mul_f32 v20, v11, v16 // (op 9917) 
v_mul_f32 v21, v11, v17 // (op 9917) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 9928) 
v_mul_f32 v15, v9, v5 // (op 9928) 
v_mul_f32 v16, v9, v6 // (op 9928) 
v_mul_f32 v17, v9, v7 // (op 9928) 

v_add_f32 v4, v14, v18 // (op 9918) 
v_add_f32 v5, v15, v19 // (op 9918) 
v_add_f32 v6, v16, v20 // (op 9918) 
v_add_f32 v7, v17, v21 // (op 9918) 




v_add_u32 v0, 64, v13 // (op 9920) 

buffer_store_dwordx4 v[4:7], v0, s[56:59], 0 offen // (op 9920) Store value




s_lshl_b32 s0, s2, 6 // (op 9933) 
s_add_u32 s0, s0, 64 // (op 9933) 
v_add_u32 v0, v12, s0 // (op 9933) 


buffer_load_dwordx4 v[4:7], v0, s[52:55], 0 offen // (op 9933) Load value

v_accvgpr_read v14, a20 // (op 9943) 
v_accvgpr_read v15, a21 // (op 9943) 
v_accvgpr_read v16, a22 // (op 9943) 
v_accvgpr_read v17, a23 // (op 9943) 
v_mul_f32 v18, v11, v14 // (op 9943) 
v_mul_f32 v19, v11, v15 // (op 9943) 
v_mul_f32 v20, v11, v16 // (op 9943) 
v_mul_f32 v21, v11, v17 // (op 9943) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 9931) 
v_mul_f32 v15, v9, v5 // (op 9931) 
v_mul_f32 v16, v9, v6 // (op 9931) 
v_mul_f32 v17, v9, v7 // (op 9931) 

v_add_f32 v4, v14, v18 // (op 9932) 
v_add_f32 v5, v15, v19 // (op 9932) 
v_add_f32 v6, v16, v20 // (op 9932) 
v_add_f32 v7, v17, v21 // (op 9932) 




s_lshl_b32 s0, s4, 6 // (op 9922) 
s_add_u32 s0, s0, 64 // (op 9922) 
v_add_u32 v0, v13, s0 // (op 9922) 


buffer_store_dwordx4 v[4:7], v0, s[56:59], 0 offen // (op 9922) Store value




s_lshl_b32 s0, s2, 7 // (op 9937) 
s_add_u32 s0, s0, 64 // (op 9937) 
v_add_u32 v0, v12, s0 // (op 9937) 


buffer_load_dwordx4 v[4:7], v0, s[52:55], 0 offen // (op 9937) Load value

v_accvgpr_read v14, a24 // (op 9945) 
v_accvgpr_read v15, a25 // (op 9945) 
v_accvgpr_read v16, a26 // (op 9945) 
v_accvgpr_read v17, a27 // (op 9945) 
v_mul_f32 v18, v11, v14 // (op 9945) 
v_mul_f32 v19, v11, v15 // (op 9945) 
v_mul_f32 v20, v11, v16 // (op 9945) 
v_mul_f32 v21, v11, v17 // (op 9945) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 9935) 
v_mul_f32 v15, v9, v5 // (op 9935) 
v_mul_f32 v16, v9, v6 // (op 9935) 
v_mul_f32 v17, v9, v7 // (op 9935) 

v_add_f32 v4, v14, v18 // (op 9936) 
v_add_f32 v5, v15, v19 // (op 9936) 
v_add_f32 v6, v16, v20 // (op 9936) 
v_add_f32 v7, v17, v21 // (op 9936) 




s_lshl_b32 s0, s4, 7 // (op 9924) 
s_add_u32 s0, s0, 64 // (op 9924) 
v_add_u32 v0, v13, s0 // (op 9924) 


buffer_store_dwordx4 v[4:7], v0, s[56:59], 0 offen // (op 9924) Store value




s_lshl_b32 s0, s2, 6 // (op 9941) 
s_mul_i32 s1, 3, s0 // (op 9941) 
s_add_u32 s1, s1, 64 // (op 9941) 

v_add_u32 v0, v12, s1 // (op 9941) 


buffer_load_dwordx4 v[4:7], v0, s[52:55], 0 offen // (op 9941) Load value

v_accvgpr_read v14, a28 // (op 9947) 
v_accvgpr_read v15, a29 // (op 9947) 
v_accvgpr_read v16, a30 // (op 9947) 
v_accvgpr_read v17, a31 // (op 9947) 
v_mul_f32 v18, v11, v14 // (op 9947) 
v_mul_f32 v19, v11, v15 // (op 9947) 
v_mul_f32 v20, v11, v16 // (op 9947) 
v_mul_f32 v21, v11, v17 // (op 9947) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 9939) 
v_mul_f32 v15, v9, v5 // (op 9939) 
v_mul_f32 v16, v9, v6 // (op 9939) 
v_mul_f32 v17, v9, v7 // (op 9939) 

v_add_f32 v4, v14, v18 // (op 9940) 
v_add_f32 v5, v15, v19 // (op 9940) 
v_add_f32 v6, v16, v20 // (op 9940) 
v_add_f32 v7, v17, v21 // (op 9940) 




s_lshl_b32 s0, s4, 6 // (op 9926) 
s_mul_i32 s1, 3, s0 // (op 9926) 
s_add_u32 s1, s1, 64 // (op 9926) 

v_add_u32 v0, v13, s1 // (op 9926) 


buffer_store_dwordx4 v[4:7], v0, s[56:59], 0 offen // (op 9926) Store value




v_mov_b32 v0, v12 // (op 126) reg expression

s_nop 0
buffer_load_dwordx4 v[4:7], v0, s[52:55], 0 offen // (op 126) Load value

v_accvgpr_read v14, a0 // (op 123) 
v_accvgpr_read v15, a1 // (op 123) 
v_accvgpr_read v16, a2 // (op 123) 
v_accvgpr_read v17, a3 // (op 123) 
v_mul_f32 v18, v11, v14 // (op 123) 
v_mul_f32 v19, v11, v15 // (op 123) 
v_mul_f32 v20, v11, v16 // (op 123) 
v_mul_f32 v21, v11, v17 // (op 123) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 127) 
v_mul_f32 v15, v9, v5 // (op 127) 
v_mul_f32 v16, v9, v6 // (op 127) 
v_mul_f32 v17, v9, v7 // (op 127) 

v_add_f32 v4, v14, v18 // (op 124) 
v_add_f32 v5, v15, v19 // (op 124) 
v_add_f32 v6, v16, v20 // (op 124) 
v_add_f32 v7, v17, v21 // (op 124) 




v_mov_b32 v0, v13 // (op 125) reg expression

buffer_store_dwordx4 v[4:7], v0, s[56:59], 0 offen // (op 125) Store value




s_lshl_b32 s0, s2, 6 // (op 9843) 
v_add_u32 v0, v12, s0 // (op 9843) 


buffer_load_dwordx4 v[4:7], v0, s[52:55], 0 offen // (op 9843) Load value

v_accvgpr_read v14, a4 // (op 9840) 
v_accvgpr_read v15, a5 // (op 9840) 
v_accvgpr_read v16, a6 // (op 9840) 
v_accvgpr_read v17, a7 // (op 9840) 
v_mul_f32 v18, v11, v14 // (op 9840) 
v_mul_f32 v19, v11, v15 // (op 9840) 
v_mul_f32 v20, v11, v16 // (op 9840) 
v_mul_f32 v21, v11, v17 // (op 9840) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 9844) 
v_mul_f32 v15, v9, v5 // (op 9844) 
v_mul_f32 v16, v9, v6 // (op 9844) 
v_mul_f32 v17, v9, v7 // (op 9844) 

v_add_f32 v4, v14, v18 // (op 9841) 
v_add_f32 v5, v15, v19 // (op 9841) 
v_add_f32 v6, v16, v20 // (op 9841) 
v_add_f32 v7, v17, v21 // (op 9841) 




s_lshl_b32 s0, s4, 6 // (op 9842) 
v_add_u32 v0, v13, s0 // (op 9842) 


buffer_store_dwordx4 v[4:7], v0, s[56:59], 0 offen // (op 9842) Store value




s_lshl_b32 s0, s2, 7 // (op 9855) 
v_add_u32 v0, v12, s0 // (op 9855) 


buffer_load_dwordx4 v[4:7], v0, s[52:55], 0 offen // (op 9855) Load value

v_accvgpr_read v14, a8 // (op 9852) 
v_accvgpr_read v15, a9 // (op 9852) 
v_accvgpr_read v16, a10 // (op 9852) 
v_accvgpr_read v17, a11 // (op 9852) 
v_mul_f32 v18, v11, v14 // (op 9852) 
v_mul_f32 v19, v11, v15 // (op 9852) 
v_mul_f32 v20, v11, v16 // (op 9852) 
v_mul_f32 v21, v11, v17 // (op 9852) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 9856) 
v_mul_f32 v15, v9, v5 // (op 9856) 
v_mul_f32 v16, v9, v6 // (op 9856) 
v_mul_f32 v17, v9, v7 // (op 9856) 

v_add_f32 v4, v14, v18 // (op 9853) 
v_add_f32 v5, v15, v19 // (op 9853) 
v_add_f32 v6, v16, v20 // (op 9853) 
v_add_f32 v7, v17, v21 // (op 9853) 




s_lshl_b32 s0, s4, 7 // (op 9854) 
v_add_u32 v0, v13, s0 // (op 9854) 


buffer_store_dwordx4 v[4:7], v0, s[56:59], 0 offen // (op 9854) Store value




s_lshl_b32 s0, s2, 6 // (op 9867) 
s_mul_i32 s1, 3, s0 // (op 9867) 

v_add_u32 v0, v12, s1 // (op 9867) 


buffer_load_dwordx4 v[4:7], v0, s[52:55], 0 offen // (op 9867) Load value

v_accvgpr_read v14, a12 // (op 9864) 
v_accvgpr_read v15, a13 // (op 9864) 
v_accvgpr_read v16, a14 // (op 9864) 
v_accvgpr_read v17, a15 // (op 9864) 
v_mul_f32 v18, v11, v14 // (op 9864) 
v_mul_f32 v19, v11, v15 // (op 9864) 
v_mul_f32 v20, v11, v16 // (op 9864) 
v_mul_f32 v21, v11, v17 // (op 9864) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v9, v4 // (op 9868) 
v_mul_f32 v15, v9, v5 // (op 9868) 
v_mul_f32 v16, v9, v6 // (op 9868) 
v_mul_f32 v17, v9, v7 // (op 9868) 

v_add_f32 v4, v14, v18 // (op 9865) 
v_add_f32 v5, v15, v19 // (op 9865) 
v_add_f32 v6, v16, v20 // (op 9865) 
v_add_f32 v7, v17, v21 // (op 9865) 




s_lshl_b32 s0, s4, 6 // (op 9866) 
s_mul_i32 s1, 3, s0 // (op 9866) 

v_add_u32 v0, v13, s1 // (op 9866) 


buffer_store_dwordx4 v[4:7], v0, s[56:59], 0 offen // (op 9866) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_WGM__ConditionalBottom_0_beta_114:








































s_endpgm  // End of RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_WGM_
.LRR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_WGM__end:
.size RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_WGM_, .LRR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_WGM__end-RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_WGM_
  .amdhsa_next_free_vgpr 264
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 98304 // lds bytes
  .amdhsa_accum_offset 136
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
  - .name: RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_WGM_
    .symbol: RR_GEMM_TN_FP4_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_256x128x128_UR_4_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 98304
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 134
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

