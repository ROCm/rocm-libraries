// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_,@function
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 8757) 
s_mul_hi_i32 s1, s16, s10 // (op 9012) 
v_lshrrev_b32 v0, 7, v1 // (op 8818) 
v_and_b32 v3, 15, v1 // (op 8818) 
v_lshl_add_u32 v4, v0, 4, v3 // (op 8818) 


v_and_b32 v0, 48, v1 // (op 8818) 
v_lshl_add_u32 v3, v4, 7, v0 // (op 8818) 


v_lshrrev_b32 v0, 6, v1 // (op 8875) 
v_and_b32 v4, 1, v0 // (op 8875) 

v_and_b32 v0, 15, v1 // (op 8875) 
v_lshl_add_u32 v5, v4, 4, v0 // (op 8875) 


v_and_b32 v0, 48, v1 // (op 8875) 
v_lshl_add_u32 v4, v5, 7, v0 // (op 8875) 


v_lshrrev_b32 v0, 7, v1 // (op 8933) 
v_and_b32 v5, 15, v1 // (op 8933) 
v_lshl_add_u32 v6, v0, 4, v5 // (op 8933) 


v_and_b32 v0, 63, v1 // (op 8933) 
v_lshrrev_b32 v5, 4, v0 // (op 8933) 

v_lshl_add_u32 v0, v6, 2, v5 // (op 8933) 


v_lshrrev_b32 v5, 6, v1 // (op 8991) 
v_and_b32 v6, 1, v5 // (op 8991) 

v_and_b32 v5, 15, v1 // (op 8991) 
v_lshl_add_u32 v7, v6, 4, v5 // (op 8991) 


v_and_b32 v5, 63, v1 // (op 8991) 
v_lshrrev_b32 v6, 4, v5 // (op 8991) 

v_lshl_add_u32 v5, v7, 2, v6 // (op 8991) 


v_lshrrev_b32 v6, 7, v1 // (op 9058) 
v_and_b32 v7, 15, v1 // (op 9058) 
v_lshl_add_u32 v8, v6, 4, v7 // (op 9058) 


v_and_b32 v6, 48, v1 // (op 9058) 
v_lshl_add_u32 v7, v8, 7, v6 // (op 9058) 


v_lshrrev_b32 v6, 6, v1 // (op 9102) 
v_and_b32 v8, 1, v6 // (op 9102) 

v_and_b32 v6, 15, v1 // (op 9102) 
v_lshl_add_u32 v9, v8, 4, v6 // (op 9102) 


v_and_b32 v6, 48, v1 // (op 9102) 
v_lshl_add_u32 v8, v9, 7, v6 // (op 9102) 


v_lshrrev_b32 v6, 7, v1 // (op 9131) 
v_and_b32 v9, 15, v1 // (op 9131) 
v_lshl_add_u32 v10, v6, 4, v9 // (op 9131) 


v_and_b32 v6, 63, v1 // (op 9131) 
v_lshrrev_b32 v9, 4, v6 // (op 9131) 

v_lshl_add_u32 v6, v10, 2, v9 // (op 9131) 


v_lshrrev_b32 v9, 6, v1 // (op 9160) 
v_and_b32 v10, 1, v9 // (op 9160) 

v_and_b32 v9, 15, v1 // (op 9160) 
v_lshl_add_u32 v11, v10, 4, v9 // (op 9160) 


v_and_b32 v9, 63, v1 // (op 9160) 
v_lshrrev_b32 v10, 4, v9 // (op 9160) 

v_lshl_add_u32 v9, v11, 2, v10 // (op 9160) 


v_lshrrev_b32 v10, 7, v1 // (op 8757) 
s_add_i32 s18, s0, s16 // (op 8757) 

v_and_b32 v11, 63, v1 // (op 8757) 
s_ashr_i32 s0, s18, 31 // (op 8757) 
v_lshrrev_b32 v12, 4, v11 // (op 8757) 
s_cmp_eq_i32 s10, 0 // (op 8757) 
s_mov_b32 s19, scc // (op 8757) 
v_lshlrev_b32 v13, 2, v12 // (op 8757) 

s_cmp_eq_i32 s19, 0 // (op 8757) 
s_cselect_b32 s49, 0, -1 // (op 8757) 

v_lshl_add_u32 v12, v10, 4, v13 // (op 8757) 


s_and_b32 s19, s13, 31 // (op 8757) 
v_lshrrev_b32 v10, 2, v11 // (op 8757) 

v_and_b32 v11, 3, v10 // (op 8757) 

v_lshrrev_b32 v10, 7, v1 // (op 9012) 
v_and_b32 v13, 63, v1 // (op 9012) 
s_lshl_b32 s50, 1, s19 // (op 8757) 
v_and_b32 v14, 3, v1 // (op 8757) 
s_add_i32 s51, s50, s49 // (op 8757) 


s_and_b32 s49, s0, s51 // (op 8757) 


s_add_i32 s0, s1, s16 // (op 9012) 

s_ashr_i32 s1, s0, 31 // (op 9012) 
v_lshl_add_u32 v15, v11, 2, v14 // (op 8757) 


s_add_i32 s50, s18, s49 // (op 8757) 


v_lshrrev_b32 v11, 6, v1 // (op 8757) 
s_ashr_i32 s18, s50, s19 // (op 8757) 


v_and_b32 v14, 1, v11 // (op 8757) 

s_lshl_b32 s19, s13, 24 // (op 8757) 
v_lshl_add_u32 v11, v14, 4, v15 // (op 8757) 


s_ashr_i32 s49, s19, 31 // (op 8757) 

s_xor_b32 s19, s18, s49 // (op 8757) 

s_sub_i32 s18, s19, s49 // (op 8757) 


s_mul_hi_i32 s19, s18, s14 // (op 8757) 
s_add_i32 s49, s19, s18 // (op 8757) 

s_ashr_i32 s19, s49, 31 // (op 8757) 
s_cmp_eq_i32 s14, 0 // (op 8757) 
s_mov_b32 s50, scc // (op 8757) 
s_cmp_eq_i32 s50, 0 // (op 8757) 
s_cselect_b32 s51, 0, -1 // (op 8757) 

s_and_b32 s50, s15, 31 // (op 8757) 
v_lshrrev_b32 v14, 4, v13 // (op 9012) 
s_cmp_eq_i32 s10, 0 // (op 9012) 
s_mov_b32 s52, scc // (op 9012) 
v_lshlrev_b32 v15, 2, v14 // (op 9012) 

s_cmp_eq_i32 s52, 0 // (op 9012) 
s_cselect_b32 s53, 0, -1 // (op 9012) 

s_lshl_b32 s52, 1, s50 // (op 8757) 
v_lshl_add_u32 v14, v10, 4, v15 // (op 9012) 


v_lshrrev_b32 v10, 2, v13 // (op 9012) 

s_add_i32 s54, s52, s51 // (op 8757) 


s_and_b32 s51, s19, s54 // (op 8757) 


s_add_i32 s19, s49, s51 // (op 8757) 


s_ashr_i32 s49, s19, s50 // (op 8757) 


s_lshl_b32 s19, s15, 24 // (op 8757) 
s_ashr_i32 s50, s19, 31 // (op 8757) 

s_xor_b32 s19, s49, s50 // (op 8757) 

s_sub_i32 s49, s19, s50 // (op 8757) 


s_mul_i32 s19, s18, s11 // (op 8757) 
s_sub_u32 s50, s16, s19 // (op 8757) 

s_mul_i32 s19, s49, s11 // (op 8757) 
s_add_u32 s19, s19, s50 // (op 8757) 

s_mul_hi_i32 s50, s16, s9 // (op 8757) 
s_add_i32 s51, s50, s16 // (op 8757) 

s_ashr_i32 s50, s51, 31 // (op 8757) 
s_cmp_eq_i32 s9, 0 // (op 8757) 
s_mov_b32 s52, scc // (op 8757) 
s_cmp_eq_i32 s52, 0 // (op 8757) 
s_cselect_b32 s54, 0, -1 // (op 8757) 

s_and_b32 s52, s13, 31 // (op 9012) 
v_and_b32 v13, 3, v10 // (op 9012) 

s_lshl_b32 s55, 1, s52 // (op 9012) 
v_and_b32 v10, 3, v1 // (op 9012) 
s_add_i32 s56, s55, s53 // (op 9012) 


s_and_b32 s53, s1, s56 // (op 9012) 


v_lshl_add_u32 v15, v13, 2, v10 // (op 9012) 


s_add_i32 s1, s0, s53 // (op 9012) 


v_lshrrev_b32 v10, 6, v1 // (op 9012) 
s_ashr_i32 s0, s1, s52 // (op 9012) 


v_and_b32 v13, 1, v10 // (op 9012) 

s_lshl_b32 s1, s13, 24 // (op 9012) 
v_lshl_add_u32 v10, v13, 4, v15 // (op 9012) 


s_ashr_i32 s52, s1, 31 // (op 9012) 

s_xor_b32 s1, s0, s52 // (op 9012) 

s_sub_i32 s0, s1, s52 // (op 9012) 


s_mul_hi_i32 s1, s0, s14 // (op 9012) 
s_add_i32 s52, s1, s0 // (op 9012) 

s_ashr_i32 s1, s52, 31 // (op 9012) 
s_cmp_eq_i32 s14, 0 // (op 9012) 
s_mov_b32 s53, scc // (op 9012) 
s_cmp_eq_i32 s53, 0 // (op 9012) 
s_cselect_b32 s55, 0, -1 // (op 9012) 

s_and_b32 s53, s15, 31 // (op 9012) 
s_lshl_b32 s56, 1, s53 // (op 9012) 
s_add_i32 s57, s56, s55 // (op 9012) 


s_and_b32 s55, s1, s57 // (op 9012) 


s_add_i32 s1, s52, s55 // (op 9012) 


s_ashr_i32 s52, s1, s53 // (op 9012) 


s_lshl_b32 s1, s15, 24 // (op 9012) 
s_ashr_i32 s53, s1, 31 // (op 9012) 

s_xor_b32 s1, s52, s53 // (op 9012) 

s_sub_i32 s52, s1, s53 // (op 9012) 


s_mul_i32 s1, s0, s11 // (op 9012) 
s_sub_u32 s53, s16, s1 // (op 9012) 

s_mul_i32 s1, s52, s11 // (op 9012) 
s_add_u32 s1, s1, s53 // (op 9012) 

s_mul_hi_i32 s53, s16, s9 // (op 9012) 
s_add_i32 s55, s53, s16 // (op 9012) 

s_ashr_i32 s53, s55, 31 // (op 9012) 
s_cmp_eq_i32 s9, 0 // (op 9012) 
s_mov_b32 s56, scc // (op 9012) 
s_cmp_eq_i32 s56, 0 // (op 9012) 
s_cselect_b32 s57, 0, -1 // (op 9012) 


s_waitcnt lgkmcnt(0)
s_and_b32 s56, s45, 31 // (op 8757) 
s_lshl_b32 s58, 1, s56 // (op 8757) 
s_add_i32 s59, s58, s54 // (op 8757) 


s_and_b32 s54, s50, s59 // (op 8757) 


s_add_i32 s50, s51, s54 // (op 8757) 


s_ashr_i32 s51, s50, s56 // (op 8757) 


s_lshl_b32 s50, s45, 24 // (op 8757) 
s_ashr_i32 s54, s50, 31 // (op 8757) 

s_xor_b32 s50, s51, s54 // (op 8757) 

s_sub_i32 s51, s50, s54 // (op 8757) 


s_and_b32 s50, s45, 31 // (op 9012) 
s_lshl_b32 s54, 1, s50 // (op 9012) 
s_add_i32 s56, s54, s57 // (op 9012) 


s_and_b32 s54, s53, s56 // (op 9012) 


s_add_i32 s53, s55, s54 // (op 9012) 


s_ashr_i32 s54, s53, s50 // (op 9012) 


s_lshl_b32 s50, s45, 24 // (op 9012) 
s_ashr_i32 s53, s50, 31 // (op 9012) 

s_xor_b32 s50, s54, s53 // (op 9012) 

s_sub_i32 s54, s50, s53 // (op 9012) 


s_add_u32 s56, s6, 256 // (op 8757) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 8757) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 8757) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 8757) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 8 // (op 8757) 

s_mul_i32 s50, s11, s56 // (op 8757) 
s_mul_i32 s53, s8, s50 // (op 8757) 

s_mul_i32 s50, s51, s53 // (op 8757) 

s_sub_u32 s53, s16, s50 // (op 8757) 

s_mov_b32 s50, s53 // (op 8757) 
s_mul_hi_i32 s55, s50, s12 // (op 8757) 
s_add_i32 s58, s55, s50 // (op 8757) 


s_ashr_i32 s50, s58, 31 // (op 8757) 
s_cmp_eq_i32 s12, 0 // (op 8757) 
s_mov_b32 s55, scc // (op 8757) 
s_cmp_eq_i32 s55, 0 // (op 8757) 
s_cselect_b32 s59, 0, -1 // (op 8757) 

s_and_b32 s55, s44, 31 // (op 8757) 
s_lshl_b32 s60, 1, s55 // (op 8757) 
s_add_i32 s61, s60, s59 // (op 8757) 


s_and_b32 s59, s50, s61 // (op 8757) 


s_add_i32 s50, s58, s59 // (op 8757) 


s_ashr_i32 s58, s50, s55 // (op 8757) 


s_lshl_b32 s50, s44, 24 // (op 8757) 
s_ashr_i32 s55, s50, 31 // (op 8757) 

s_xor_b32 s50, s58, s55 // (op 8757) 

s_sub_i32 s58, s50, s55 // (op 8757) 


s_mul_i32 s50, s46, s11 // (op 8757) 
s_add_u32 s60, s20, 64 // (op 8757) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 8757) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 8757) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 8757) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 6 // (op 8757) 

s_sub_i32 s55, s60, s50 // (op 8757) 

s_mul_i32 s59, s58, s55 // (op 8757) 

s_sub_u32 s55, s53, s59 // (op 8757) 


s_add_u32 s53, s55, s50 // (op 8757) 


s_cmp_eq_i32 s51, 0 // (op 8757) 
s_mov_b32 s50, scc // (op 8757) 

s_cmp_eq_i32 s50, 0 // (op 8757) 
s_cselect_b32 s51, s53, s19 // (op 8757) 


s_mul_i32 s19, s49, s56 // (op 8757) 


v_lshl_add_u32 v13, s51, 6, v12 // (op 8757) 


s_sub_i32 s49, s18, s19 // (op 8757) 


s_cmp_eq_i32 s50, 0 // (op 8757) 
s_cselect_b32 s51, s58, s49 // (op 8757) 



v_lshl_add_u32 v12, s51, 8, v11 // (op 8757) 


v_mul_lo_u32 v11, v12, s2 // (op 8757) 

v_add_lshl_u32 v12, v13, v11, 2 // (op 8757) 


s_add_u32 s18, s6, 256 // (op 9012) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 9012) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 9012) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 9012) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 8 // (op 9012) 

s_mul_i32 s53, s11, s18 // (op 9012) 
s_mul_i32 s49, s8, s53 // (op 9012) 

s_mul_i32 s53, s54, s49 // (op 9012) 

s_sub_u32 s49, s16, s53 // (op 9012) 

s_mov_b32 s53, s49 // (op 9012) 
s_mul_hi_i32 s50, s53, s12 // (op 9012) 
s_add_i32 s51, s50, s53 // (op 9012) 


s_ashr_i32 s50, s51, 31 // (op 9012) 
s_cmp_eq_i32 s12, 0 // (op 9012) 
s_mov_b32 s53, scc // (op 9012) 
s_cmp_eq_i32 s53, 0 // (op 9012) 
s_cselect_b32 s55, 0, -1 // (op 9012) 

s_and_b32 s53, s44, 31 // (op 9012) 
s_lshl_b32 s56, 1, s53 // (op 9012) 
s_add_i32 s57, s56, s55 // (op 9012) 


s_and_b32 s55, s50, s57 // (op 9012) 


s_add_i32 s50, s51, s55 // (op 9012) 


s_ashr_i32 s51, s50, s53 // (op 9012) 


s_lshl_b32 s50, s44, 24 // (op 9012) 
s_ashr_i32 s53, s50, 31 // (op 9012) 

s_xor_b32 s50, s51, s53 // (op 9012) 

s_sub_i32 s51, s50, s53 // (op 9012) 


s_mul_i32 s50, s46, s11 // (op 9012) 
s_add_u32 s56, s20, 64 // (op 9012) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 9012) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 9012) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 9012) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 6 // (op 9012) 

s_sub_i32 s53, s56, s50 // (op 9012) 

s_mul_i32 s55, s51, s53 // (op 9012) 

s_sub_u32 s53, s49, s55 // (op 9012) 


s_add_u32 s49, s53, s50 // (op 9012) 


s_cmp_eq_i32 s54, 0 // (op 9012) 
s_mov_b32 s50, scc // (op 9012) 

s_cmp_eq_i32 s50, 0 // (op 9012) 
s_cselect_b32 s53, s49, s1 // (op 9012) 


s_mul_i32 s1, s52, s18 // (op 9012) 


v_lshl_add_u32 v11, s53, 6, v14 // (op 9012) 


s_sub_i32 s49, s0, s1 // (op 9012) 


s_cmp_eq_i32 s50, 0 // (op 9012) 
s_cselect_b32 s0, s51, s49 // (op 9012) 



v_lshl_add_u32 v13, s0, 8, v10 // (op 9012) 


v_mul_lo_u32 v10, v13, s4 // (op 9012) 

v_add_lshl_u32 v13, v11, v10, 2 // (op 9012) 


s_mul_i32 s1, s2, s7 // (op 8759) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 8759) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 8759) least significant
s_mul_hi_u32 s19, s2, s6 // (op 8759) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8759) most significant: sum
s_add_u32 s1, s1, s19 // (op 8759) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 8759) 

s_mul_i32 s1, s4, s7 // (op 9014) most significant: low of low * high
s_mul_i32 s49, s5, s6 // (op 9014) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 9014) least significant
s_mul_hi_u32 s50, s4, s6 // (op 9014) most significant: high of low * low
s_add_u32 s1, s1, s49 // (op 9014) most significant: sum
s_add_u32 s1, s1, s50 // (op 9014) most significant: sum


s_lshl_b64 s[50:51], s[0:1], 2 // (op 9014) 

s_mov_b32 s52, s22 // (op 8759) reg expression
s_mov_b32 s53, s23 // (op 8759) reg expression
s_mov_b32 s54, s18 // (op 8759) reg expression
s_mov_b32 s55, 131072 // (op 8759) call()

s_mov_b32 s56, s24 // (op 9014) reg expression

s_mov_b32 s57, s25 // (op 9014) reg expression
s_mov_b32 s58, s50 // (op 9014) reg expression
s_mov_b32 s59, 131072 // (op 9014) call()


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
v_accvgpr_write a16, 0.00000 // (op 864) call()
v_accvgpr_write a17, 0.00000 // (op 864) call()
v_accvgpr_write a18, 0.00000 // (op 864) call()
v_accvgpr_write a19, 0.00000 // (op 864) call()
v_accvgpr_write a20, 0.00000 // (op 1040) call()
v_accvgpr_write a21, 0.00000 // (op 1040) call()
v_accvgpr_write a22, 0.00000 // (op 1040) call()
v_accvgpr_write a23, 0.00000 // (op 1040) call()
v_accvgpr_write a24, 0.00000 // (op 1216) call()
v_accvgpr_write a25, 0.00000 // (op 1216) call()
v_accvgpr_write a26, 0.00000 // (op 1216) call()
v_accvgpr_write a27, 0.00000 // (op 1216) call()
v_accvgpr_write a28, 0.00000 // (op 1392) call()
v_accvgpr_write a29, 0.00000 // (op 1392) call()
v_accvgpr_write a30, 0.00000 // (op 1392) call()
v_accvgpr_write a31, 0.00000 // (op 1392) call()
v_accvgpr_write a32, 0.00000 // (op 1744) call()
v_accvgpr_write a33, 0.00000 // (op 1744) call()
v_accvgpr_write a34, 0.00000 // (op 1744) call()
v_accvgpr_write a35, 0.00000 // (op 1744) call()
v_accvgpr_write a36, 0.00000 // (op 2361) call()
v_accvgpr_write a37, 0.00000 // (op 2361) call()
v_accvgpr_write a38, 0.00000 // (op 2361) call()
v_accvgpr_write a39, 0.00000 // (op 2361) call()
v_accvgpr_write a40, 0.00000 // (op 2362) call()
v_accvgpr_write a41, 0.00000 // (op 2362) call()
v_accvgpr_write a42, 0.00000 // (op 2362) call()
v_accvgpr_write a43, 0.00000 // (op 2362) call()
v_accvgpr_write a44, 0.00000 // (op 2363) call()
v_accvgpr_write a45, 0.00000 // (op 2363) call()
v_accvgpr_write a46, 0.00000 // (op 2363) call()
v_accvgpr_write a47, 0.00000 // (op 2363) call()
v_accvgpr_write a48, 0.00000 // (op 2364) call()
v_accvgpr_write a49, 0.00000 // (op 2364) call()
v_accvgpr_write a50, 0.00000 // (op 2364) call()
v_accvgpr_write a51, 0.00000 // (op 2364) call()
v_accvgpr_write a52, 0.00000 // (op 2365) call()
v_accvgpr_write a53, 0.00000 // (op 2365) call()
v_accvgpr_write a54, 0.00000 // (op 2365) call()
v_accvgpr_write a55, 0.00000 // (op 2365) call()
v_accvgpr_write a56, 0.00000 // (op 2366) call()
v_accvgpr_write a57, 0.00000 // (op 2366) call()
v_accvgpr_write a58, 0.00000 // (op 2366) call()
v_accvgpr_write a59, 0.00000 // (op 2366) call()
v_accvgpr_write a60, 0.00000 // (op 2367) call()
v_accvgpr_write a61, 0.00000 // (op 2367) call()
v_accvgpr_write a62, 0.00000 // (op 2367) call()
v_accvgpr_write a63, 0.00000 // (op 2367) call()
s_mul_hi_i32 s0, s16, s10 // (op 8519) 
v_lshrrev_b32 v14, 3, v1 // (op 8519) 
s_add_i32 s1, s0, s16 // (op 8519) 

v_and_b32 v15, -8, v1 // (op 8519) 
s_ashr_i32 s0, s1, 31 // (op 8519) 
v_sub_u32 v16, v1, v15 // (op 8519) 

s_cmp_eq_i32 s10, 0 // (op 8519) 
s_mov_b32 s18, scc // (op 8519) 
v_lshlrev_b32 v15, 4, v16 // (op 8519) 

s_cmp_eq_i32 s18, 0 // (op 8519) 
s_cselect_b32 s19, 0, -1 // (op 8519) 

v_mov_b32 v17, 0 // (op 8519) convert
v_mov_b32 v16, v15 // (op 8519) convert

s_and_b32 s18, s13, 31 // (op 8519) 
s_lshl_b32 s22, 1, s18 // (op 8519) 
s_add_i32 s23, s22, s19 // (op 8519) 


s_and_b32 s19, s0, s23 // (op 8519) 


s_add_i32 s0, s1, s19 // (op 8519) 


s_ashr_i32 s1, s0, s18 // (op 8519) 


s_lshl_b32 s0, s13, 24 // (op 8519) 
s_ashr_i32 s18, s0, 31 // (op 8519) 

s_xor_b32 s0, s1, s18 // (op 8519) 

s_sub_i32 s1, s0, s18 // (op 8519) 


s_mul_hi_i32 s0, s1, s14 // (op 8519) 
s_add_i32 s18, s0, s1 // (op 8519) 

s_ashr_i32 s0, s18, 31 // (op 8519) 
s_cmp_eq_i32 s14, 0 // (op 8519) 
s_mov_b32 s19, scc // (op 8519) 
s_cmp_eq_i32 s19, 0 // (op 8519) 
s_cselect_b32 s22, 0, -1 // (op 8519) 

s_and_b32 s19, s15, 31 // (op 8519) 
s_lshl_b32 s23, 1, s19 // (op 8519) 
s_add_i32 s24, s23, s22 // (op 8519) 


s_and_b32 s25, s0, s24 // (op 8519) 


s_add_i32 s0, s18, s25 // (op 8519) 


s_ashr_i32 s18, s0, s19 // (op 8519) 


s_lshl_b32 s0, s15, 24 // (op 8519) 
s_ashr_i32 s19, s0, 31 // (op 8519) 

s_xor_b32 s0, s18, s19 // (op 8519) 

s_sub_i32 s18, s0, s19 // (op 8519) 


s_mul_i32 s0, s1, s11 // (op 8519) 

s_sub_u32 s1, s16, s0 // (op 8519) 

s_mul_i32 s0, s18, s11 // (op 8519) 
s_add_u32 s0, s0, s1 // (op 8519) 


s_mul_hi_i32 s1, s16, s9 // (op 8519) 
s_add_i32 s18, s1, s16 // (op 8519) 

s_ashr_i32 s1, s18, 31 // (op 8519) 
s_cmp_eq_i32 s9, 0 // (op 8519) 
s_mov_b32 s19, scc // (op 8519) 
s_cmp_eq_i32 s19, 0 // (op 8519) 
s_cselect_b32 s22, 0, -1 // (op 8519) 

s_and_b32 s19, s45, 31 // (op 8519) 
s_lshl_b32 s23, 1, s19 // (op 8519) 
s_add_i32 s24, s23, s22 // (op 8519) 


s_and_b32 s25, s1, s24 // (op 8519) 


s_add_i32 s1, s18, s25 // (op 8519) 


s_ashr_i32 s18, s1, s19 // (op 8519) 


s_lshl_b32 s1, s45, 24 // (op 8519) 
s_ashr_i32 s19, s1, 31 // (op 8519) 

s_xor_b32 s1, s18, s19 // (op 8519) 

s_sub_i32 s18, s1, s19 // (op 8519) 


s_add_u32 s22, s6, 256 // (op 8519) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8519) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 8519) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 8519) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 8519) 

s_mul_i32 s1, s11, s22 // (op 8519) 

s_mul_i32 s19, s8, s1 // (op 8519) 

s_mul_i32 s1, s18, s19 // (op 8519) 

s_sub_u32 s19, s16, s1 // (op 8519) 

s_mov_b32 s1, s19 // (op 8519) 
s_mul_hi_i32 s22, s1, s12 // (op 8519) 
s_add_i32 s23, s22, s1 // (op 8519) 


s_ashr_i32 s1, s23, 31 // (op 8519) 
s_cmp_eq_i32 s12, 0 // (op 8519) 
s_mov_b32 s22, scc // (op 8519) 
s_cmp_eq_i32 s22, 0 // (op 8519) 
s_cselect_b32 s24, 0, -1 // (op 8519) 

s_and_b32 s22, s44, 31 // (op 8519) 
s_lshl_b32 s25, 1, s22 // (op 8519) 
s_add_i32 s47, s25, s24 // (op 8519) 


s_and_b32 s24, s1, s47 // (op 8519) 


s_add_i32 s1, s23, s24 // (op 8519) 


s_ashr_i32 s23, s1, s22 // (op 8519) 


s_lshl_b32 s1, s44, 24 // (op 8519) 
s_ashr_i32 s22, s1, 31 // (op 8519) 

s_xor_b32 s1, s23, s22 // (op 8519) 

s_sub_i32 s23, s1, s22 // (op 8519) 


s_mul_i32 s1, s46, s11 // (op 8519) 
s_add_u32 s24, s20, 64 // (op 8519) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 8519) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 8519) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 8519) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 6 // (op 8519) 

s_sub_i32 s22, s24, s1 // (op 8519) 

s_mul_i32 s24, s23, s22 // (op 8519) 


s_sub_u32 s25, s19, s24 // (op 8519) 


s_add_u32 s19, s25, s1 // (op 8519) 


s_cmp_eq_i32 s18, 0 // (op 8519) 
s_mov_b32 s1, scc // (op 8519) 

s_cmp_eq_i32 s1, 0 // (op 8519) 
s_cselect_b32 s18, s19, s0 // (op 8519) 



v_lshl_add_u32 v15, s18, 6, v14 // (op 8519) 


v_mul_lo_u32 v19, v15, s27 // (op 8519) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 8519) most significant: high of low * low
v_mul_lo_u32 v18, v15, s26 // (op 8519) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 8519) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 8519) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 8519) most significant half



s_mul_i32 s1, s26, s21 // (op 8521) most significant: low of low * high
s_mul_i32 s18, s27, s20 // (op 8521) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 8521) least significant
s_mul_hi_u32 s19, s26, s20 // (op 8521) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 8521) most significant: sum
s_add_u32 s1, s1, s19 // (op 8521) most significant: sum


s_mov_b32 s48, s28 // (op 8521) reg expression
s_mov_b32 s49, s29 // (op 8521) reg expression
s_mov_b32 s50, s0 // (op 8521) reg expression
s_mov_b32 s51, 131072 // (op 8521) call()

s_mul_hi_i32 s47, s16, s10 // (op 8538) 
v_lshrrev_b32 v14, 3, v1 // (op 8538) 
s_add_i32 s0, s47, s16 // (op 8538) 

v_and_b32 v15, -8, v1 // (op 8538) 
s_ashr_i32 s1, s0, 31 // (op 8538) 
v_sub_u32 v16, v1, v15 // (op 8538) 

s_cmp_eq_i32 s10, 0 // (op 8538) 
s_mov_b32 s47, scc // (op 8538) 
v_lshlrev_b32 v15, 4, v16 // (op 8538) 

s_cmp_eq_i32 s47, 0 // (op 8538) 
s_cselect_b32 s18, 0, -1 // (op 8538) 

v_mov_b32 v17, 0 // (op 8538) convert
v_mov_b32 v16, v15 // (op 8538) convert

s_and_b32 s19, s13, 31 // (op 8538) 
s_lshl_b32 s47, 1, s19 // (op 8538) 
s_add_i32 s22, s47, s18 // (op 8538) 


s_and_b32 s18, s1, s22 // (op 8538) 


s_add_i32 s1, s0, s18 // (op 8538) 


s_ashr_i32 s0, s1, s19 // (op 8538) 


s_lshl_b32 s1, s13, 24 // (op 8538) 
s_ashr_i32 s47, s1, 31 // (op 8538) 

s_xor_b32 s1, s0, s47 // (op 8538) 

s_sub_i32 s0, s1, s47 // (op 8538) 


s_mul_hi_i32 s1, s0, s14 // (op 8538) 
s_add_i32 s47, s1, s0 // (op 8538) 

s_ashr_i32 s1, s47, 31 // (op 8538) 
s_cmp_eq_i32 s14, 0 // (op 8538) 
s_mov_b32 s18, scc // (op 8538) 
s_cmp_eq_i32 s18, 0 // (op 8538) 
s_cselect_b32 s19, 0, -1 // (op 8538) 

s_and_b32 s18, s15, 31 // (op 8538) 
s_lshl_b32 s22, 1, s18 // (op 8538) 
s_add_i32 s23, s22, s19 // (op 8538) 


s_and_b32 s19, s1, s23 // (op 8538) 


s_add_i32 s1, s47, s19 // (op 8538) 


s_ashr_i32 s19, s1, s18 // (op 8538) 


s_lshl_b32 s1, s15, 24 // (op 8538) 
s_ashr_i32 s18, s1, 31 // (op 8538) 

s_xor_b32 s1, s19, s18 // (op 8538) 

s_sub_i32 s19, s1, s18 // (op 8538) 


s_mul_i32 s1, s0, s11 // (op 8538) 

s_sub_u32 s0, s16, s1 // (op 8538) 

s_mul_i32 s1, s19, s11 // (op 8538) 
s_add_u32 s1, s1, s0 // (op 8538) 


s_mul_hi_i32 s0, s16, s9 // (op 8538) 
s_add_i32 s47, s0, s16 // (op 8538) 

s_ashr_i32 s0, s47, 31 // (op 8538) 
s_cmp_eq_i32 s9, 0 // (op 8538) 
s_mov_b32 s18, scc // (op 8538) 
s_cmp_eq_i32 s18, 0 // (op 8538) 
s_cselect_b32 s19, 0, -1 // (op 8538) 

s_and_b32 s18, s45, 31 // (op 8538) 
s_lshl_b32 s22, 1, s18 // (op 8538) 
s_add_i32 s23, s22, s19 // (op 8538) 


s_and_b32 s19, s0, s23 // (op 8538) 


s_add_i32 s0, s47, s19 // (op 8538) 


s_ashr_i32 s19, s0, s18 // (op 8538) 


s_lshl_b32 s0, s45, 24 // (op 8538) 
s_ashr_i32 s18, s0, 31 // (op 8538) 

s_xor_b32 s0, s19, s18 // (op 8538) 

s_sub_i32 s19, s0, s18 // (op 8538) 


s_add_u32 s22, s6, 256 // (op 8538) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8538) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 8538) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 8538) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 8538) 

s_mul_i32 s0, s11, s22 // (op 8538) 

s_mul_i32 s18, s8, s0 // (op 8538) 

s_mul_i32 s0, s19, s18 // (op 8538) 

s_sub_u32 s18, s16, s0 // (op 8538) 

s_mov_b32 s0, s18 // (op 8538) 
s_mul_hi_i32 s47, s0, s12 // (op 8538) 
s_add_i32 s22, s47, s0 // (op 8538) 


s_ashr_i32 s0, s22, 31 // (op 8538) 
s_cmp_eq_i32 s12, 0 // (op 8538) 
s_mov_b32 s47, scc // (op 8538) 
s_cmp_eq_i32 s47, 0 // (op 8538) 
s_cselect_b32 s23, 0, -1 // (op 8538) 

s_and_b32 s47, s44, 31 // (op 8538) 
s_lshl_b32 s24, 1, s47 // (op 8538) 
s_add_i32 s25, s24, s23 // (op 8538) 


s_and_b32 s23, s0, s25 // (op 8538) 


s_add_i32 s0, s22, s23 // (op 8538) 


s_ashr_i32 s22, s0, s47 // (op 8538) 


s_lshl_b32 s0, s44, 24 // (op 8538) 
s_ashr_i32 s47, s0, 31 // (op 8538) 

s_xor_b32 s0, s22, s47 // (op 8538) 

s_sub_i32 s22, s0, s47 // (op 8538) 


s_mul_i32 s0, s46, s11 // (op 8538) 
s_add_u32 s24, s20, 64 // (op 8538) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 8538) End of Int64 add, unlocking SCC
s_sub_u32 s60, s24, 1 // (op 8538) least significant half; sets scc
s_subb_u32 s61, s25, 0 // (op 8538) most significant half; uses scc

s_ashr_i64 s[24:25], s[60:61], 6 // (op 8538) 

s_sub_i32 s23, s24, s0 // (op 8538) 

s_mul_i32 s47, s22, s23 // (op 8538) 


s_sub_u32 s22, s18, s47 // (op 8538) 


s_add_u32 s18, s22, s0 // (op 8538) 


s_cmp_eq_i32 s19, 0 // (op 8538) 
s_mov_b32 s0, scc // (op 8538) 

s_cmp_eq_i32 s0, 0 // (op 8538) 
s_cselect_b32 s19, s18, s1 // (op 8538) 



v_lshl_add_u32 v15, s19, 6, v14 // (op 8538) 


v_mul_lo_u32 v21, v15, s27 // (op 8538) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 8538) most significant: high of low * low
v_mul_lo_u32 v20, v15, s26 // (op 8538) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 8538) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v16 // (op 8538) least significant half
v_addc_co_u32 v21, s[0:1], v21, v17, s[0:1] // (op 8538) most significant half



s_mul_i32 s1, s26, s21 // (op 8540) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 8540) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 8540) least significant
s_mul_hi_u32 s18, s26, s20 // (op 8540) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 8540) most significant: sum
s_add_u32 s1, s1, s18 // (op 8540) most significant: sum


s_mov_b32 s48, s28 // (op 8540) reg expression
s_mov_b32 s49, s29 // (op 8540) reg expression
s_mov_b32 s50, s0 // (op 8540) reg expression
s_mov_b32 s51, 131072 // (op 8540) call()

s_mul_hi_i32 s47, s16, s10 // (op 8583) 
v_lshrrev_b32 v14, 3, v1 // (op 8583) 
s_add_i32 s0, s47, s16 // (op 8583) 

v_and_b32 v15, -8, v1 // (op 8583) 
s_ashr_i32 s1, s0, 31 // (op 8583) 
v_sub_u32 v16, v1, v15 // (op 8583) 

s_cmp_eq_i32 s10, 0 // (op 8583) 
s_mov_b32 s47, scc // (op 8583) 
v_lshlrev_b32 v15, 4, v16 // (op 8583) 

s_cmp_eq_i32 s47, 0 // (op 8583) 
s_cselect_b32 s18, 0, -1 // (op 8583) 

v_mov_b32 v17, 0 // (op 8583) convert
v_mov_b32 v16, v15 // (op 8583) convert

s_and_b32 s19, s13, 31 // (op 8583) 
s_lshl_b32 s47, 1, s19 // (op 8583) 
s_add_i32 s22, s47, s18 // (op 8583) 


s_and_b32 s18, s1, s22 // (op 8583) 


s_add_i32 s1, s0, s18 // (op 8583) 


s_ashr_i32 s0, s1, s19 // (op 8583) 


s_lshl_b32 s1, s13, 24 // (op 8583) 
s_ashr_i32 s47, s1, 31 // (op 8583) 

s_xor_b32 s1, s0, s47 // (op 8583) 

s_sub_i32 s0, s1, s47 // (op 8583) 


s_mul_hi_i32 s1, s0, s14 // (op 8583) 
s_add_i32 s47, s1, s0 // (op 8583) 

s_ashr_i32 s1, s47, 31 // (op 8583) 
s_cmp_eq_i32 s14, 0 // (op 8583) 
s_mov_b32 s18, scc // (op 8583) 
s_cmp_eq_i32 s18, 0 // (op 8583) 
s_cselect_b32 s19, 0, -1 // (op 8583) 

s_and_b32 s18, s15, 31 // (op 8583) 
s_lshl_b32 s22, 1, s18 // (op 8583) 
s_add_i32 s23, s22, s19 // (op 8583) 


s_and_b32 s19, s1, s23 // (op 8583) 


s_add_i32 s1, s47, s19 // (op 8583) 


s_ashr_i32 s19, s1, s18 // (op 8583) 


s_lshl_b32 s1, s15, 24 // (op 8583) 
s_ashr_i32 s18, s1, 31 // (op 8583) 

s_xor_b32 s1, s19, s18 // (op 8583) 

s_sub_i32 s19, s1, s18 // (op 8583) 


s_add_u32 s22, s6, 256 // (op 8583) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8583) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 8583) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 8583) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 8583) 

s_mul_i32 s1, s19, s22 // (op 8583) 

s_sub_i32 s47, s0, s1 // (op 8583) 


s_mul_hi_i32 s0, s16, s9 // (op 8583) 
s_add_i32 s1, s0, s16 // (op 8583) 

s_ashr_i32 s0, s1, 31 // (op 8583) 
s_cmp_eq_i32 s9, 0 // (op 8583) 
s_mov_b32 s18, scc // (op 8583) 
s_cmp_eq_i32 s18, 0 // (op 8583) 
s_cselect_b32 s19, 0, -1 // (op 8583) 

s_and_b32 s18, s45, 31 // (op 8583) 
s_lshl_b32 s24, 1, s18 // (op 8583) 
s_add_i32 s25, s24, s19 // (op 8583) 


s_and_b32 s19, s0, s25 // (op 8583) 


s_add_i32 s0, s1, s19 // (op 8583) 


s_ashr_i32 s1, s0, s18 // (op 8583) 


s_lshl_b32 s0, s45, 24 // (op 8583) 
s_ashr_i32 s18, s0, 31 // (op 8583) 

s_xor_b32 s0, s1, s18 // (op 8583) 

s_sub_i32 s1, s0, s18 // (op 8583) 


s_mul_i32 s0, s11, s22 // (op 8583) 

s_mul_i32 s18, s8, s0 // (op 8583) 

s_mul_i32 s0, s1, s18 // (op 8583) 

s_sub_u32 s18, s16, s0 // (op 8583) 

s_mul_hi_i32 s0, s18, s12 // (op 8583) 
s_add_i32 s19, s0, s18 // (op 8583) 


s_ashr_i32 s0, s19, 31 // (op 8583) 
s_cmp_eq_i32 s12, 0 // (op 8583) 
s_mov_b32 s18, scc // (op 8583) 
s_cmp_eq_i32 s18, 0 // (op 8583) 
s_cselect_b32 s22, 0, -1 // (op 8583) 

s_and_b32 s18, s44, 31 // (op 8583) 
s_lshl_b32 s23, 1, s18 // (op 8583) 
s_add_i32 s24, s23, s22 // (op 8583) 


s_and_b32 s25, s0, s24 // (op 8583) 


s_add_i32 s0, s19, s25 // (op 8583) 


s_ashr_i32 s19, s0, s18 // (op 8583) 


s_lshl_b32 s0, s44, 24 // (op 8583) 
s_ashr_i32 s18, s0, 31 // (op 8583) 

s_xor_b32 s0, s19, s18 // (op 8583) 

s_sub_i32 s19, s0, s18 // (op 8583) 


s_cmp_eq_i32 s1, 0 // (op 8583) 
s_mov_b32 s0, scc // (op 8583) 

s_cmp_eq_i32 s0, 0 // (op 8583) 
s_cselect_b32 s1, s19, s47 // (op 8583) 



v_lshl_add_u32 v15, s1, 8, v14 // (op 8583) 


v_mul_lo_u32 v23, v15, s31 // (op 8583) most significant: low of low * high
v_mul_hi_u32 v14, v15, s30 // (op 8583) most significant: high of low * low
v_mul_lo_u32 v22, v15, s30 // (op 8583) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 8583) most significant: sum


v_add_co_u32 v14, s[0:1], v16, v22 // (op 8583) least significant half
v_addc_co_u32 v15, s[0:1], v17, v23, s[0:1] // (op 8583) most significant half



s_mul_i32 s1, s30, s7 // (op 8585) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 8585) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 8585) least significant
s_mul_hi_u32 s18, s30, s6 // (op 8585) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 8585) most significant: sum
s_add_u32 s1, s1, s18 // (op 8585) most significant: sum


s_mov_b32 s60, s32 // (op 8585) reg expression
s_mov_b32 s61, s33 // (op 8585) reg expression
s_mov_b32 s62, s0 // (op 8585) reg expression
s_mov_b32 s63, 131072 // (op 8585) call()

s_mul_hi_i32 s47, s16, s10 // (op 8602) 
v_lshrrev_b32 v16, 3, v1 // (op 8602) 
s_add_i32 s0, s47, s16 // (op 8602) 

v_and_b32 v17, -8, v1 // (op 8602) 
s_ashr_i32 s1, s0, 31 // (op 8602) 
v_sub_u32 v22, v1, v17 // (op 8602) 

s_cmp_eq_i32 s10, 0 // (op 8602) 
s_mov_b32 s47, scc // (op 8602) 
v_lshlrev_b32 v17, 4, v22 // (op 8602) 

s_cmp_eq_i32 s47, 0 // (op 8602) 
s_cselect_b32 s18, 0, -1 // (op 8602) 

v_mov_b32 v23, 0 // (op 8602) convert
v_mov_b32 v22, v17 // (op 8602) convert

s_and_b32 s19, s13, 31 // (op 8602) 
s_lshl_b32 s47, 1, s19 // (op 8602) 
s_add_i32 s22, s47, s18 // (op 8602) 


s_and_b32 s18, s1, s22 // (op 8602) 


s_add_i32 s1, s0, s18 // (op 8602) 


s_ashr_i32 s0, s1, s19 // (op 8602) 


s_lshl_b32 s1, s13, 24 // (op 8602) 
s_ashr_i32 s47, s1, 31 // (op 8602) 

s_xor_b32 s1, s0, s47 // (op 8602) 

s_sub_i32 s0, s1, s47 // (op 8602) 


s_mul_hi_i32 s1, s0, s14 // (op 8602) 
s_add_i32 s47, s1, s0 // (op 8602) 

s_ashr_i32 s1, s47, 31 // (op 8602) 
s_cmp_eq_i32 s14, 0 // (op 8602) 
s_mov_b32 s18, scc // (op 8602) 
s_cmp_eq_i32 s18, 0 // (op 8602) 
s_cselect_b32 s19, 0, -1 // (op 8602) 

s_and_b32 s18, s15, 31 // (op 8602) 
s_lshl_b32 s22, 1, s18 // (op 8602) 
s_add_i32 s23, s22, s19 // (op 8602) 


s_and_b32 s19, s1, s23 // (op 8602) 


s_add_i32 s1, s47, s19 // (op 8602) 


s_ashr_i32 s19, s1, s18 // (op 8602) 


s_lshl_b32 s1, s15, 24 // (op 8602) 
s_ashr_i32 s18, s1, 31 // (op 8602) 

s_xor_b32 s1, s19, s18 // (op 8602) 

s_sub_i32 s19, s1, s18 // (op 8602) 


s_add_u32 s22, s6, 256 // (op 8602) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8602) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 8602) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 8602) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 8602) 

s_mul_i32 s1, s19, s22 // (op 8602) 

s_sub_i32 s47, s0, s1 // (op 8602) 


s_mul_hi_i32 s0, s16, s9 // (op 8602) 
s_add_i32 s1, s0, s16 // (op 8602) 

s_ashr_i32 s0, s1, 31 // (op 8602) 
s_cmp_eq_i32 s9, 0 // (op 8602) 
s_mov_b32 s18, scc // (op 8602) 
s_cmp_eq_i32 s18, 0 // (op 8602) 
s_cselect_b32 s19, 0, -1 // (op 8602) 

s_and_b32 s18, s45, 31 // (op 8602) 
s_lshl_b32 s24, 1, s18 // (op 8602) 
s_add_i32 s25, s24, s19 // (op 8602) 


s_and_b32 s19, s0, s25 // (op 8602) 


s_add_i32 s0, s1, s19 // (op 8602) 


s_ashr_i32 s1, s0, s18 // (op 8602) 


s_lshl_b32 s0, s45, 24 // (op 8602) 
s_ashr_i32 s18, s0, 31 // (op 8602) 

s_xor_b32 s0, s1, s18 // (op 8602) 

s_sub_i32 s1, s0, s18 // (op 8602) 


s_mul_i32 s0, s11, s22 // (op 8602) 

s_mul_i32 s18, s8, s0 // (op 8602) 

s_mul_i32 s0, s1, s18 // (op 8602) 

s_sub_u32 s18, s16, s0 // (op 8602) 

s_mul_hi_i32 s0, s18, s12 // (op 8602) 
s_add_i32 s19, s0, s18 // (op 8602) 


s_ashr_i32 s0, s19, 31 // (op 8602) 
s_cmp_eq_i32 s12, 0 // (op 8602) 
s_mov_b32 s18, scc // (op 8602) 
s_cmp_eq_i32 s18, 0 // (op 8602) 
s_cselect_b32 s22, 0, -1 // (op 8602) 

s_and_b32 s18, s44, 31 // (op 8602) 
s_lshl_b32 s23, 1, s18 // (op 8602) 
s_add_i32 s24, s23, s22 // (op 8602) 


s_and_b32 s25, s0, s24 // (op 8602) 


s_add_i32 s0, s19, s25 // (op 8602) 


s_ashr_i32 s19, s0, s18 // (op 8602) 


s_lshl_b32 s0, s44, 24 // (op 8602) 
s_ashr_i32 s18, s0, 31 // (op 8602) 

s_xor_b32 s0, s19, s18 // (op 8602) 

s_sub_i32 s19, s0, s18 // (op 8602) 


s_cmp_eq_i32 s1, 0 // (op 8602) 
s_mov_b32 s0, scc // (op 8602) 

s_cmp_eq_i32 s0, 0 // (op 8602) 
s_cselect_b32 s1, s19, s47 // (op 8602) 



v_lshl_add_u32 v17, s1, 8, v16 // (op 8602) 


v_mul_lo_u32 v25, v17, s31 // (op 8602) most significant: low of low * high
v_mul_hi_u32 v16, v17, s30 // (op 8602) most significant: high of low * low
v_mul_lo_u32 v24, v17, s30 // (op 8602) least significant: low of low * low
v_add_u32 v25, v25, v16 // (op 8602) most significant: sum


v_add_co_u32 v16, s[0:1], v22, v24 // (op 8602) least significant half
v_addc_co_u32 v17, s[0:1], v23, v25, s[0:1] // (op 8602) most significant half



s_mul_i32 s1, s30, s7 // (op 8604) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 8604) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 8604) least significant
s_mul_hi_u32 s18, s30, s6 // (op 8604) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 8604) most significant: sum
s_add_u32 s1, s1, s18 // (op 8604) most significant: sum


s_mov_b32 s60, s32 // (op 8604) reg expression
s_mov_b32 s61, s33 // (op 8604) reg expression
s_mov_b32 s62, s0 // (op 8604) reg expression
s_mov_b32 s63, 131072 // (op 8604) call()

s_mul_hi_i32 s47, s16, s10 // (op 8647) 
v_lshrrev_b32 v22, 2, v1 // (op 8647) 
s_add_i32 s0, s47, s16 // (op 8647) 

v_and_b32 v23, -4, v1 // (op 8647) 
s_ashr_i32 s1, s0, 31 // (op 8647) 
v_sub_u32 v24, v1, v23 // (op 8647) 

s_cmp_eq_i32 s10, 0 // (op 8647) 
s_mov_b32 s47, scc // (op 8647) 
s_cmp_eq_i32 s47, 0 // (op 8647) 
s_cselect_b32 s18, 0, -1 // (op 8647) 

s_and_b32 s19, s13, 31 // (op 8647) 
s_lshl_b32 s47, 1, s19 // (op 8647) 
s_add_i32 s22, s47, s18 // (op 8647) 


s_and_b32 s18, s1, s22 // (op 8647) 


s_add_i32 s1, s0, s18 // (op 8647) 


s_ashr_i32 s0, s1, s19 // (op 8647) 


s_lshl_b32 s1, s13, 24 // (op 8647) 
s_ashr_i32 s47, s1, 31 // (op 8647) 

s_xor_b32 s1, s0, s47 // (op 8647) 

s_sub_i32 s0, s1, s47 // (op 8647) 


s_mul_hi_i32 s1, s0, s14 // (op 8647) 
s_add_i32 s47, s1, s0 // (op 8647) 

s_ashr_i32 s1, s47, 31 // (op 8647) 
s_cmp_eq_i32 s14, 0 // (op 8647) 
s_mov_b32 s18, scc // (op 8647) 
s_cmp_eq_i32 s18, 0 // (op 8647) 
s_cselect_b32 s19, 0, -1 // (op 8647) 

s_and_b32 s18, s15, 31 // (op 8647) 
s_lshl_b32 s22, 1, s18 // (op 8647) 
s_add_i32 s23, s22, s19 // (op 8647) 


s_and_b32 s19, s1, s23 // (op 8647) 


s_add_i32 s1, s47, s19 // (op 8647) 


s_ashr_i32 s19, s1, s18 // (op 8647) 


s_lshl_b32 s1, s15, 24 // (op 8647) 
s_ashr_i32 s18, s1, 31 // (op 8647) 

s_xor_b32 s1, s19, s18 // (op 8647) 

s_sub_i32 s19, s1, s18 // (op 8647) 


s_mul_i32 s1, s0, s11 // (op 8647) 

s_sub_u32 s0, s16, s1 // (op 8647) 

s_mul_i32 s1, s19, s11 // (op 8647) 
s_add_u32 s1, s1, s0 // (op 8647) 


s_mul_hi_i32 s0, s16, s9 // (op 8647) 
s_add_i32 s47, s0, s16 // (op 8647) 

s_ashr_i32 s0, s47, 31 // (op 8647) 
s_cmp_eq_i32 s9, 0 // (op 8647) 
s_mov_b32 s18, scc // (op 8647) 
s_cmp_eq_i32 s18, 0 // (op 8647) 
s_cselect_b32 s19, 0, -1 // (op 8647) 

s_and_b32 s18, s45, 31 // (op 8647) 
s_lshl_b32 s22, 1, s18 // (op 8647) 
s_add_i32 s23, s22, s19 // (op 8647) 


s_and_b32 s19, s0, s23 // (op 8647) 


s_add_i32 s0, s47, s19 // (op 8647) 


s_ashr_i32 s19, s0, s18 // (op 8647) 


s_lshl_b32 s0, s45, 24 // (op 8647) 
s_ashr_i32 s18, s0, 31 // (op 8647) 

s_xor_b32 s0, s19, s18 // (op 8647) 

s_sub_i32 s19, s0, s18 // (op 8647) 


s_add_u32 s22, s6, 256 // (op 8647) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8647) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 8647) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 8647) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 8647) 

s_mul_i32 s0, s11, s22 // (op 8647) 

s_mul_i32 s18, s8, s0 // (op 8647) 

s_mul_i32 s0, s19, s18 // (op 8647) 

s_sub_u32 s18, s16, s0 // (op 8647) 

s_mov_b32 s0, s18 // (op 8647) 
s_mul_hi_i32 s47, s0, s12 // (op 8647) 
s_add_i32 s22, s47, s0 // (op 8647) 


s_ashr_i32 s0, s22, 31 // (op 8647) 
s_cmp_eq_i32 s12, 0 // (op 8647) 
s_mov_b32 s47, scc // (op 8647) 
s_cmp_eq_i32 s47, 0 // (op 8647) 
s_cselect_b32 s23, 0, -1 // (op 8647) 

s_and_b32 s47, s44, 31 // (op 8647) 
s_lshl_b32 s24, 1, s47 // (op 8647) 
s_add_i32 s25, s24, s23 // (op 8647) 


s_and_b32 s23, s0, s25 // (op 8647) 


s_add_i32 s0, s22, s23 // (op 8647) 


s_ashr_i32 s22, s0, s47 // (op 8647) 


s_lshl_b32 s0, s44, 24 // (op 8647) 
s_ashr_i32 s47, s0, 31 // (op 8647) 

s_xor_b32 s0, s22, s47 // (op 8647) 

s_sub_i32 s22, s0, s47 // (op 8647) 


s_mul_i32 s0, s46, s11 // (op 8647) 
s_add_u32 s24, s20, 64 // (op 8647) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 8647) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 8647) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 8647) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 6 // (op 8647) 

s_sub_i32 s23, s24, s0 // (op 8647) 

s_mul_i32 s47, s22, s23 // (op 8647) 


s_sub_u32 s22, s18, s47 // (op 8647) 


s_add_u32 s18, s22, s0 // (op 8647) 


s_cmp_eq_i32 s19, 0 // (op 8647) 
s_mov_b32 s0, scc // (op 8647) 

s_cmp_eq_i32 s0, 0 // (op 8647) 
s_cselect_b32 s19, s18, s1 // (op 8647) 



v_lshl_add_u32 v23, s19, 6, v22 // (op 8647) 


v_mul_lo_u32 v22, v23, s34 // (op 8647) 
v_add_u32 v22, v22, v24 // (op 8647) 


s_mul_i32 s1, s34, s21 // (op 8649) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 8649) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 8649) least significant
s_mul_hi_u32 s18, s34, s20 // (op 8649) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 8649) most significant: sum
s_add_u32 s1, s1, s18 // (op 8649) most significant: sum


s_mov_b32 s64, s36 // (op 8649) reg expression
s_mov_b32 s65, s37 // (op 8649) reg expression
s_mov_b32 s66, s0 // (op 8649) reg expression
s_mov_b32 s67, 131072 // (op 8649) call()

s_mul_hi_i32 s47, s16, s10 // (op 8666) 
v_lshrrev_b32 v23, 2, v1 // (op 8666) 
s_add_i32 s0, s47, s16 // (op 8666) 

v_and_b32 v24, -4, v1 // (op 8666) 
s_ashr_i32 s1, s0, 31 // (op 8666) 
v_sub_u32 v25, v1, v24 // (op 8666) 

s_cmp_eq_i32 s10, 0 // (op 8666) 
s_mov_b32 s47, scc // (op 8666) 
s_cmp_eq_i32 s47, 0 // (op 8666) 
s_cselect_b32 s18, 0, -1 // (op 8666) 

s_and_b32 s19, s13, 31 // (op 8666) 
s_lshl_b32 s47, 1, s19 // (op 8666) 
s_add_i32 s22, s47, s18 // (op 8666) 


s_and_b32 s18, s1, s22 // (op 8666) 


s_add_i32 s1, s0, s18 // (op 8666) 


s_ashr_i32 s0, s1, s19 // (op 8666) 


s_lshl_b32 s1, s13, 24 // (op 8666) 
s_ashr_i32 s47, s1, 31 // (op 8666) 

s_xor_b32 s1, s0, s47 // (op 8666) 

s_sub_i32 s0, s1, s47 // (op 8666) 


s_mul_hi_i32 s1, s0, s14 // (op 8666) 
s_add_i32 s47, s1, s0 // (op 8666) 

s_ashr_i32 s1, s47, 31 // (op 8666) 
s_cmp_eq_i32 s14, 0 // (op 8666) 
s_mov_b32 s18, scc // (op 8666) 
s_cmp_eq_i32 s18, 0 // (op 8666) 
s_cselect_b32 s19, 0, -1 // (op 8666) 

s_and_b32 s18, s15, 31 // (op 8666) 
s_lshl_b32 s22, 1, s18 // (op 8666) 
s_add_i32 s23, s22, s19 // (op 8666) 


s_and_b32 s19, s1, s23 // (op 8666) 


s_add_i32 s1, s47, s19 // (op 8666) 


s_ashr_i32 s19, s1, s18 // (op 8666) 


s_lshl_b32 s1, s15, 24 // (op 8666) 
s_ashr_i32 s18, s1, 31 // (op 8666) 

s_xor_b32 s1, s19, s18 // (op 8666) 

s_sub_i32 s19, s1, s18 // (op 8666) 


s_mul_i32 s1, s0, s11 // (op 8666) 

s_sub_u32 s0, s16, s1 // (op 8666) 

s_mul_i32 s1, s19, s11 // (op 8666) 
s_add_u32 s1, s1, s0 // (op 8666) 


s_mul_hi_i32 s0, s16, s9 // (op 8666) 
s_add_i32 s47, s0, s16 // (op 8666) 

s_ashr_i32 s0, s47, 31 // (op 8666) 
s_cmp_eq_i32 s9, 0 // (op 8666) 
s_mov_b32 s18, scc // (op 8666) 
s_cmp_eq_i32 s18, 0 // (op 8666) 
s_cselect_b32 s19, 0, -1 // (op 8666) 

s_and_b32 s18, s45, 31 // (op 8666) 
s_lshl_b32 s22, 1, s18 // (op 8666) 
s_add_i32 s23, s22, s19 // (op 8666) 


s_and_b32 s19, s0, s23 // (op 8666) 


s_add_i32 s0, s47, s19 // (op 8666) 


s_ashr_i32 s19, s0, s18 // (op 8666) 


s_lshl_b32 s0, s45, 24 // (op 8666) 
s_ashr_i32 s18, s0, 31 // (op 8666) 

s_xor_b32 s0, s19, s18 // (op 8666) 

s_sub_i32 s19, s0, s18 // (op 8666) 


s_add_u32 s22, s6, 256 // (op 8666) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8666) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 8666) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 8666) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 8666) 

s_mul_i32 s0, s11, s22 // (op 8666) 

s_mul_i32 s18, s8, s0 // (op 8666) 

s_mul_i32 s0, s19, s18 // (op 8666) 

s_sub_u32 s18, s16, s0 // (op 8666) 

s_mov_b32 s0, s18 // (op 8666) 
s_mul_hi_i32 s47, s0, s12 // (op 8666) 
s_add_i32 s22, s47, s0 // (op 8666) 


s_ashr_i32 s0, s22, 31 // (op 8666) 
s_cmp_eq_i32 s12, 0 // (op 8666) 
s_mov_b32 s47, scc // (op 8666) 
s_cmp_eq_i32 s47, 0 // (op 8666) 
s_cselect_b32 s23, 0, -1 // (op 8666) 

s_and_b32 s47, s44, 31 // (op 8666) 
s_lshl_b32 s24, 1, s47 // (op 8666) 
s_add_i32 s25, s24, s23 // (op 8666) 


s_and_b32 s23, s0, s25 // (op 8666) 


s_add_i32 s0, s22, s23 // (op 8666) 


s_ashr_i32 s22, s0, s47 // (op 8666) 


s_lshl_b32 s0, s44, 24 // (op 8666) 
s_ashr_i32 s47, s0, 31 // (op 8666) 

s_xor_b32 s0, s22, s47 // (op 8666) 

s_sub_i32 s22, s0, s47 // (op 8666) 


s_mul_i32 s0, s46, s11 // (op 8666) 
s_add_u32 s24, s20, 64 // (op 8666) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 8666) End of Int64 add, unlocking SCC
s_sub_u32 s68, s24, 1 // (op 8666) least significant half; sets scc
s_subb_u32 s69, s25, 0 // (op 8666) most significant half; uses scc

s_ashr_i64 s[24:25], s[68:69], 6 // (op 8666) 

s_sub_i32 s23, s24, s0 // (op 8666) 

s_mul_i32 s47, s22, s23 // (op 8666) 


s_sub_u32 s22, s18, s47 // (op 8666) 


s_add_u32 s18, s22, s0 // (op 8666) 


s_cmp_eq_i32 s19, 0 // (op 8666) 
s_mov_b32 s0, scc // (op 8666) 

s_cmp_eq_i32 s0, 0 // (op 8666) 
s_cselect_b32 s19, s18, s1 // (op 8666) 



v_lshl_add_u32 v24, s19, 6, v23 // (op 8666) 


v_mul_lo_u32 v23, v24, s34 // (op 8666) 
v_add_u32 v23, v23, v25 // (op 8666) 


s_mul_i32 s1, s34, s21 // (op 8668) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 8668) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 8668) least significant
s_mul_hi_u32 s18, s34, s20 // (op 8668) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 8668) most significant: sum
s_add_u32 s1, s1, s18 // (op 8668) most significant: sum


s_mov_b32 s64, s36 // (op 8668) reg expression
s_mov_b32 s65, s37 // (op 8668) reg expression
s_mov_b32 s66, s0 // (op 8668) reg expression
s_mov_b32 s67, 131072 // (op 8668) call()

s_mul_hi_i32 s47, s16, s10 // (op 8711) 
s_add_i32 s0, s47, s16 // (op 8711) 

s_ashr_i32 s1, s0, 31 // (op 8711) 
s_cmp_eq_i32 s10, 0 // (op 8711) 
s_mov_b32 s47, scc // (op 8711) 
s_cmp_eq_i32 s47, 0 // (op 8711) 
s_cselect_b32 s18, 0, -1 // (op 8711) 

s_and_b32 s19, s13, 31 // (op 8711) 
s_lshl_b32 s47, 1, s19 // (op 8711) 
s_add_i32 s22, s47, s18 // (op 8711) 


s_and_b32 s18, s1, s22 // (op 8711) 


s_add_i32 s1, s0, s18 // (op 8711) 


s_ashr_i32 s0, s1, s19 // (op 8711) 


s_lshl_b32 s1, s13, 24 // (op 8711) 
s_ashr_i32 s47, s1, 31 // (op 8711) 

s_xor_b32 s1, s0, s47 // (op 8711) 

s_sub_i32 s0, s1, s47 // (op 8711) 


s_mul_hi_i32 s1, s0, s14 // (op 8711) 
s_add_i32 s47, s1, s0 // (op 8711) 

s_ashr_i32 s1, s47, 31 // (op 8711) 
s_cmp_eq_i32 s14, 0 // (op 8711) 
s_mov_b32 s18, scc // (op 8711) 
s_cmp_eq_i32 s18, 0 // (op 8711) 
s_cselect_b32 s19, 0, -1 // (op 8711) 

s_and_b32 s18, s15, 31 // (op 8711) 
s_lshl_b32 s22, 1, s18 // (op 8711) 
s_add_i32 s23, s22, s19 // (op 8711) 


s_and_b32 s19, s1, s23 // (op 8711) 


s_add_i32 s1, s47, s19 // (op 8711) 


s_ashr_i32 s19, s1, s18 // (op 8711) 


s_lshl_b32 s1, s15, 24 // (op 8711) 
s_ashr_i32 s18, s1, 31 // (op 8711) 

s_xor_b32 s1, s19, s18 // (op 8711) 

s_sub_i32 s19, s1, s18 // (op 8711) 


s_add_u32 s22, s6, 256 // (op 8711) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8711) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 8711) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 8711) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 8711) 

s_mul_i32 s1, s19, s22 // (op 8711) 

s_sub_i32 s47, s0, s1 // (op 8711) 


s_mul_hi_i32 s0, s16, s9 // (op 8711) 
s_add_i32 s1, s0, s16 // (op 8711) 

s_ashr_i32 s0, s1, 31 // (op 8711) 
s_cmp_eq_i32 s9, 0 // (op 8711) 
s_mov_b32 s18, scc // (op 8711) 
s_cmp_eq_i32 s18, 0 // (op 8711) 
s_cselect_b32 s19, 0, -1 // (op 8711) 

s_and_b32 s18, s45, 31 // (op 8711) 
s_lshl_b32 s24, 1, s18 // (op 8711) 
s_add_i32 s25, s24, s19 // (op 8711) 


s_and_b32 s19, s0, s25 // (op 8711) 


s_add_i32 s0, s1, s19 // (op 8711) 


s_ashr_i32 s1, s0, s18 // (op 8711) 


s_lshl_b32 s0, s45, 24 // (op 8711) 
s_ashr_i32 s18, s0, 31 // (op 8711) 

s_xor_b32 s0, s1, s18 // (op 8711) 

s_sub_i32 s1, s0, s18 // (op 8711) 


s_mul_i32 s0, s11, s22 // (op 8711) 

s_mul_i32 s18, s8, s0 // (op 8711) 

s_mul_i32 s0, s1, s18 // (op 8711) 

s_sub_u32 s18, s16, s0 // (op 8711) 

s_mul_hi_i32 s0, s18, s12 // (op 8711) 
s_add_i32 s19, s0, s18 // (op 8711) 


s_ashr_i32 s0, s19, 31 // (op 8711) 
s_cmp_eq_i32 s12, 0 // (op 8711) 
s_mov_b32 s18, scc // (op 8711) 
s_cmp_eq_i32 s18, 0 // (op 8711) 
s_cselect_b32 s22, 0, -1 // (op 8711) 

s_and_b32 s18, s44, 31 // (op 8711) 
s_lshl_b32 s23, 1, s18 // (op 8711) 
s_add_i32 s24, s23, s22 // (op 8711) 


s_and_b32 s25, s0, s24 // (op 8711) 


s_add_i32 s0, s19, s25 // (op 8711) 


s_ashr_i32 s19, s0, s18 // (op 8711) 


s_lshl_b32 s0, s44, 24 // (op 8711) 
s_ashr_i32 s18, s0, 31 // (op 8711) 

s_xor_b32 s0, s19, s18 // (op 8711) 

s_sub_i32 s19, s0, s18 // (op 8711) 


s_cmp_eq_i32 s1, 0 // (op 8711) 
s_mov_b32 s0, scc // (op 8711) 

s_cmp_eq_i32 s0, 0 // (op 8711) 
s_cselect_b32 s1, s19, s47 // (op 8711) 



v_lshl_add_u32 v24, s1, 8, v1 // (op 8711) 

v_mul_lo_u32 v25, v24, s38 // (op 8711) 

s_mul_i32 s1, s38, s7 // (op 8713) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 8713) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 8713) least significant
s_mul_hi_u32 s18, s38, s6 // (op 8713) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 8713) most significant: sum
s_add_u32 s1, s1, s18 // (op 8713) most significant: sum


s_mov_b32 s68, s40 // (op 8713) reg expression
s_mov_b32 s69, s41 // (op 8713) reg expression
s_mov_b32 s70, s0 // (op 8713) reg expression
s_mov_b32 s71, 131072 // (op 8713) call()

s_mul_hi_i32 s47, s16, s10 // (op 8730) 
s_add_i32 s0, s47, s16 // (op 8730) 

s_ashr_i32 s1, s0, 31 // (op 8730) 
s_cmp_eq_i32 s10, 0 // (op 8730) 
s_mov_b32 s47, scc // (op 8730) 
s_cmp_eq_i32 s47, 0 // (op 8730) 
s_cselect_b32 s18, 0, -1 // (op 8730) 

s_and_b32 s19, s13, 31 // (op 8730) 
s_lshl_b32 s47, 1, s19 // (op 8730) 
s_add_i32 s22, s47, s18 // (op 8730) 


s_and_b32 s18, s1, s22 // (op 8730) 


s_add_i32 s1, s0, s18 // (op 8730) 


s_ashr_i32 s0, s1, s19 // (op 8730) 


s_lshl_b32 s1, s13, 24 // (op 8730) 
s_ashr_i32 s47, s1, 31 // (op 8730) 

s_xor_b32 s1, s0, s47 // (op 8730) 

s_sub_i32 s0, s1, s47 // (op 8730) 


s_mul_hi_i32 s1, s0, s14 // (op 8730) 
s_add_i32 s47, s1, s0 // (op 8730) 

s_ashr_i32 s1, s47, 31 // (op 8730) 
s_cmp_eq_i32 s14, 0 // (op 8730) 
s_mov_b32 s18, scc // (op 8730) 
s_cmp_eq_i32 s18, 0 // (op 8730) 
s_cselect_b32 s19, 0, -1 // (op 8730) 

s_and_b32 s18, s15, 31 // (op 8730) 
s_lshl_b32 s22, 1, s18 // (op 8730) 
s_add_i32 s23, s22, s19 // (op 8730) 


s_and_b32 s19, s1, s23 // (op 8730) 


s_add_i32 s1, s47, s19 // (op 8730) 


s_ashr_i32 s19, s1, s18 // (op 8730) 


s_lshl_b32 s1, s15, 24 // (op 8730) 
s_ashr_i32 s18, s1, 31 // (op 8730) 

s_xor_b32 s1, s19, s18 // (op 8730) 

s_sub_i32 s19, s1, s18 // (op 8730) 


s_add_u32 s22, s6, 256 // (op 8730) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8730) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 8730) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 8730) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 8730) 

s_mul_i32 s1, s19, s22 // (op 8730) 

s_sub_i32 s47, s0, s1 // (op 8730) 


s_mul_hi_i32 s0, s16, s9 // (op 8730) 
s_add_i32 s1, s0, s16 // (op 8730) 

s_ashr_i32 s0, s1, 31 // (op 8730) 
s_cmp_eq_i32 s9, 0 // (op 8730) 
s_mov_b32 s18, scc // (op 8730) 
s_cmp_eq_i32 s18, 0 // (op 8730) 
s_cselect_b32 s19, 0, -1 // (op 8730) 

s_and_b32 s18, s45, 31 // (op 8730) 
s_lshl_b32 s24, 1, s18 // (op 8730) 
s_add_i32 s25, s24, s19 // (op 8730) 


s_and_b32 s19, s0, s25 // (op 8730) 


s_add_i32 s0, s1, s19 // (op 8730) 


s_ashr_i32 s1, s0, s18 // (op 8730) 


s_lshl_b32 s0, s45, 24 // (op 8730) 
s_ashr_i32 s18, s0, 31 // (op 8730) 

s_xor_b32 s0, s1, s18 // (op 8730) 

s_sub_i32 s1, s0, s18 // (op 8730) 


s_mul_i32 s0, s11, s22 // (op 8730) 

s_mul_i32 s18, s8, s0 // (op 8730) 

s_mul_i32 s0, s1, s18 // (op 8730) 

s_sub_u32 s18, s16, s0 // (op 8730) 

s_mul_hi_i32 s0, s18, s12 // (op 8730) 
s_add_i32 s19, s0, s18 // (op 8730) 


s_ashr_i32 s0, s19, 31 // (op 8730) 
s_cmp_eq_i32 s12, 0 // (op 8730) 
s_mov_b32 s18, scc // (op 8730) 
s_cmp_eq_i32 s18, 0 // (op 8730) 
s_cselect_b32 s22, 0, -1 // (op 8730) 

s_and_b32 s18, s44, 31 // (op 8730) 
s_lshl_b32 s23, 1, s18 // (op 8730) 
s_add_i32 s24, s23, s22 // (op 8730) 


s_and_b32 s25, s0, s24 // (op 8730) 


s_add_i32 s0, s19, s25 // (op 8730) 


s_ashr_i32 s19, s0, s18 // (op 8730) 


s_lshl_b32 s0, s44, 24 // (op 8730) 
s_ashr_i32 s18, s0, 31 // (op 8730) 

s_xor_b32 s0, s19, s18 // (op 8730) 

s_sub_i32 s19, s0, s18 // (op 8730) 


s_cmp_eq_i32 s1, 0 // (op 8730) 
s_mov_b32 s0, scc // (op 8730) 

s_cmp_eq_i32 s0, 0 // (op 8730) 
s_cselect_b32 s1, s19, s47 // (op 8730) 



v_lshl_add_u32 v24, s1, 8, v1 // (op 8730) 

v_mul_lo_u32 v26, v24, s38 // (op 8730) 

s_mul_i32 s1, s38, s7 // (op 8732) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 8732) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 8732) least significant
s_mul_hi_u32 s18, s38, s6 // (op 8732) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 8732) most significant: sum
s_add_u32 s1, s1, s18 // (op 8732) most significant: sum


s_mov_b32 s68, s40 // (op 8732) reg expression
s_mov_b32 s69, s41 // (op 8732) reg expression
s_mov_b32 s70, s0 // (op 8732) reg expression
s_mov_b32 s71, 131072 // (op 8732) call()

v_and_b32 v24, -8, v1 // (op 8858) 
v_sub_u32 v27, v1, v24 // (op 8858) 

v_lshlrev_b32 v24, 4, v27 // (op 8858) 

v_lshrrev_b32 v27, 3, v1 // (op 8858) 
v_lshl_add_u32 v28, v27, 7, v24 // (op 8858) 


s_nop 0
v_readfirstlane_b32 s47, v28 // (op 8858) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 5 // (op 8453) 
v_mov_b64 v[28:29], s[0:1] // (op 8453) convert


v_mov_b64 v[30:31], v[14:15] // (op 8453) 
s_mov_b32 m0, s47 // (op 8453) reg expression
buffer_load_dwordx4 v30, s[60:63], 0 offen offset: 0 lds // (op 8453) Load value direct to lds
v_add_co_u32 v30, s[0:1], v30, v28 // (op 8453) least significant half
v_addc_co_u32 v31, s[0:1], 0, v31, s[0:1] // (op 8453) most significant half

s_add_u32 m0, m0, 4096 // (op 8453) 
buffer_load_dwordx4 v30, s[60:63], 0 offen offset: 0 lds // (op 8453) Load value direct to lds
v_add_co_u32 v30, s[0:1], v30, v28 // (op 8453) least significant half
v_addc_co_u32 v31, s[0:1], 0, v31, s[0:1] // (op 8453) most significant half

s_add_u32 m0, m0, 4096 // (op 8453) 
buffer_load_dwordx4 v30, s[60:63], 0 offen offset: 0 lds // (op 8453) Load value direct to lds
v_add_co_u32 v30, s[0:1], v30, v28 // (op 8453) least significant half
v_addc_co_u32 v31, s[0:1], 0, v31, s[0:1] // (op 8453) most significant half

s_add_u32 m0, m0, 4096 // (op 8453) 
buffer_load_dwordx4 v30, s[60:63], 0 offen offset: 0 lds // (op 8453) Load value direct to lds
v_add_co_u32 v30, s[0:1], v30, v28 // (op 8453) least significant half
v_addc_co_u32 v31, s[0:1], 0, v31, s[0:1] // (op 8453) most significant half

s_add_u32 m0, m0, 4096 // (op 8453) 
buffer_load_dwordx4 v30, s[60:63], 0 offen offset: 0 lds // (op 8453) Load value direct to lds
v_add_co_u32 v30, s[0:1], v30, v28 // (op 8453) least significant half
v_addc_co_u32 v31, s[0:1], 0, v31, s[0:1] // (op 8453) most significant half

s_add_u32 m0, m0, 4096 // (op 8453) 
buffer_load_dwordx4 v30, s[60:63], 0 offen offset: 0 lds // (op 8453) Load value direct to lds
v_add_co_u32 v30, s[0:1], v30, v28 // (op 8453) least significant half
v_addc_co_u32 v31, s[0:1], 0, v31, s[0:1] // (op 8453) most significant half

s_add_u32 m0, m0, 4096 // (op 8453) 
buffer_load_dwordx4 v30, s[60:63], 0 offen offset: 0 lds // (op 8453) Load value direct to lds
v_add_co_u32 v30, s[0:1], v30, v28 // (op 8453) least significant half
v_addc_co_u32 v31, s[0:1], 0, v31, s[0:1] // (op 8453) most significant half

s_add_u32 m0, m0, 4096 // (op 8453) 
buffer_load_dwordx4 v30, s[60:63], 0 offen offset: 0 lds // (op 8453) Load value direct to lds






buffer_load_dword v24, v25, s[68:71], 0 offen // (op 8139) Load value
v_and_b32 v27, -8, v1 // (op 8801) 
v_sub_u32 v28, v1, v27 // (op 8801) 

v_lshlrev_b32 v27, 4, v28 // (op 8801) 

v_lshrrev_b32 v28, 3, v1 // (op 8801) 
v_lshl_add_u32 v29, v28, 7, v27 // (op 8801) 


s_nop 0
v_readfirstlane_b32 s47, v29 // (op 8801) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 5 // (op 8458) 
v_mov_b64 v[28:29], s[0:1] // (op 8458) convert


v_mov_b64 v[30:31], v[18:19] // (op 8458) 
s_add_u32 s0, 32768, s47 // (op 8458) 
s_mov_b32 m0, s0 // (op 8458) reg expression
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 8458) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 8458) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 8458) most significant half

s_add_u32 m0, m0, 4096 // (op 8458) 
buffer_load_dwordx4 v30, s[48:51], 0 offen offset: 0 lds // (op 8458) Load value direct to lds







buffer_load_ubyte v27, v22, s[64:67], 0 offen offset: 0 // (op 8145) Load value
v_and_b32 v28, -8, v1 // (op 9085) 
v_sub_u32 v29, v1, v28 // (op 9085) 

v_lshlrev_b32 v28, 4, v29 // (op 9085) 

v_lshrrev_b32 v29, 3, v1 // (op 9085) 
v_lshl_add_u32 v30, v29, 7, v28 // (op 9085) 


s_nop 0
v_readfirstlane_b32 s47, v30 // (op 9085) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 5 // (op 8463) 
v_mov_b64 v[28:29], s[0:1] // (op 8463) convert


v_mov_b64 v[30:31], v[14:15] // (op 8463) 
s_add_u32 s0, 40960, s47 // (op 8463) 
s_mov_b32 m0, s0 // (op 8463) reg expression
s_mov_b32 s1, 128 // (op 8463) call()
buffer_load_dwordx4 v30, s[60:63], s1 offen offset: 0 lds // (op 8463) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 8463) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 8463) most significant half

s_add_u32 m0, m0, 4096 // (op 8463) 
buffer_load_dwordx4 v30, s[60:63], s1 offen offset: 0 lds // (op 8463) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 8463) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 8463) most significant half

s_add_u32 m0, m0, 4096 // (op 8463) 
buffer_load_dwordx4 v30, s[60:63], s1 offen offset: 0 lds // (op 8463) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 8463) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 8463) most significant half

s_add_u32 m0, m0, 4096 // (op 8463) 
buffer_load_dwordx4 v30, s[60:63], s1 offen offset: 0 lds // (op 8463) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 8463) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 8463) most significant half

s_add_u32 m0, m0, 4096 // (op 8463) 
buffer_load_dwordx4 v30, s[60:63], s1 offen offset: 0 lds // (op 8463) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 8463) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 8463) most significant half

s_add_u32 m0, m0, 4096 // (op 8463) 
buffer_load_dwordx4 v30, s[60:63], s1 offen offset: 0 lds // (op 8463) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 8463) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 8463) most significant half

s_add_u32 m0, m0, 4096 // (op 8463) 
buffer_load_dwordx4 v30, s[60:63], s1 offen offset: 0 lds // (op 8463) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 8463) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 8463) most significant half

s_add_u32 m0, m0, 4096 // (op 8463) 
buffer_load_dwordx4 v30, s[60:63], s1 offen offset: 0 lds // (op 8463) Load value direct to lds









buffer_load_dword v14, v25, s[68:71], 0 offen offset:4 // (op 8151) Load value

v_and_b32 v15, -8, v1 // (op 9041) 
v_sub_u32 v25, v1, v15 // (op 9041) 

v_lshlrev_b32 v15, 4, v25 // (op 9041) 

v_lshrrev_b32 v25, 3, v1 // (op 9041) 
v_lshl_add_u32 v28, v25, 7, v15 // (op 9041) 


s_nop 0
v_readfirstlane_b32 s47, v28 // (op 9041) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 5 // (op 8465) 
v_mov_b64 v[28:29], s[0:1] // (op 8465) convert


v_mov_b64 v[30:31], v[18:19] // (op 8465) 
s_add_u32 s0, 73728, s47 // (op 8465) 
s_mov_b32 m0, s0 // (op 8465) reg expression
s_mov_b32 s1, 128 // (op 8465) call()
buffer_load_dwordx4 v30, s[48:51], s1 offen offset: 0 lds // (op 8465) Load value direct to lds
v_add_co_u32 v30, s[18:19], v30, v28 // (op 8465) least significant half
v_addc_co_u32 v31, s[18:19], 0, v31, s[18:19] // (op 8465) most significant half

s_add_u32 m0, m0, 4096 // (op 8465) 
buffer_load_dwordx4 v30, s[48:51], s1 offen offset: 0 lds // (op 8465) Load value direct to lds









buffer_load_ubyte v15, v22, s[64:67], 0 offen offset:4 // (op 8157) Load value

v_lshlrev_b32 v22, 2, v1 // (op 8974) 



v_add_u32 v25, 81920, v22 // (op 8160) 
s_waitcnt vmcnt(15)
ds_write_b32 v25, v24 // (op 8160) Store local data 


v_and_b32 v22, -4, v1 // (op 8916) 
v_sub_u32 v25, v1, v22 // (op 8916) 

v_and_b32 v22, 4294967292, v1 // (op 8916) 
v_add_u32 v18, v22, v25 // (op 8916) 





v_add_u32 v19, 82944, v18 // (op 8162) 
s_waitcnt vmcnt(12)
ds_write_b8 v19, v27 // (op 8162) Store local data 


s_barrier  // (op 9301) 



ds_read_b128 v[28:31], v4 // (op 8187) Load local data 
ds_read_b128 v[32:35], v4 offset:64 // (op 8187) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 9336) 



v_add_u32 v22, 81920, v5 // (op 8237) 
ds_read_u8 v25, v22 // (op 8237) Load local data 




ds_read_b128 v[36:39], v4 offset:4096 // (op 8192) Load local data 
ds_read_b128 v[40:43], v4 offset:4160 // (op 8192) Load local data 



v_add_u32 v22, 82048, v5 // (op 8242) 
ds_read_u8 v18, v22 // (op 8242) Load local data 




ds_read_b128 v[44:47], v4 offset:8192 // (op 8197) Load local data 
ds_read_b128 v[48:51], v4 offset:8256 // (op 8197) Load local data 



v_add_u32 v19, 82176, v5 // (op 8247) 
ds_read_u8 v22, v19 // (op 8247) Load local data 




ds_read_b128 v[52:55], v4 offset:12288 // (op 8202) Load local data 
ds_read_b128 v[56:59], v4 offset:12352 // (op 8202) Load local data 



v_add_u32 v19, 82304, v5 // (op 8252) 
ds_read_u8 v60, v19 // (op 8252) Load local data 




ds_read_b128 v[62:65], v4 offset:16384 // (op 8207) Load local data 
ds_read_b128 v[66:69], v4 offset:16448 // (op 8207) Load local data 



v_add_u32 v19, 82432, v5 // (op 8257) 
ds_read_u8 v61, v19 // (op 8257) Load local data 




ds_read_b128 v[70:73], v4 offset:20480 // (op 8212) Load local data 
ds_read_b128 v[74:77], v4 offset:20544 // (op 8212) Load local data 



v_add_u32 v19, 82560, v5 // (op 8262) 
ds_read_u8 v78, v19 // (op 8262) Load local data 




ds_read_b128 v[80:83], v4 offset:24576 // (op 8217) Load local data 
ds_read_b128 v[84:87], v4 offset:24640 // (op 8217) Load local data 



v_add_u32 v19, 82688, v5 // (op 8267) 
ds_read_u8 v79, v19 // (op 8267) Load local data 




ds_read_b128 v[88:91], v4 offset:28672 // (op 8222) Load local data 
ds_read_b128 v[92:95], v4 offset:28736 // (op 8222) Load local data 



v_add_u32 v19, 82816, v5 // (op 8272) 
ds_read_u8 v96, v19 // (op 8272) Load local data 

s_barrier  // (op 9285) 



ds_read_b128 v[98:101], v3 offset:32768 // (op 8177) Load local data 
ds_read_b128 v[102:105], v3 offset:32832 // (op 8177) Load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 9318) 



v_add_u32 v19, 82944, v0 // (op 8227) 
ds_read_u8 v97, v19 // (op 8227) Load local data 




ds_read_b128 v[106:109], v3 offset:36864 // (op 8182) Load local data 
ds_read_b128 v[110:113], v3 offset:36928 // (op 8182) Load local data 



v_add_u32 v19, 83072, v0 // (op 8232) 
ds_read_u8 v114, v19 // (op 8232) Load local data 

v_and_b32 v19, -8, v1 // (op 8773) 
v_sub_u32 v115, v1, v19 // (op 8773) 

v_lshlrev_b32 v19, 4, v115 // (op 8773) 

v_lshrrev_b32 v115, 3, v1 // (op 8773) 
v_lshl_add_u32 v116, v115, 7, v19 // (op 8773) 


s_nop 0
v_readfirstlane_b32 s47, v116 // (op 8773) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v19, -8, v1 // (op 8830) 
v_sub_u32 v115, v1, v19 // (op 8830) 

v_lshlrev_b32 v19, 4, v115 // (op 8830) 

v_lshrrev_b32 v115, 3, v1 // (op 8830) 
v_lshl_add_u32 v116, v115, 7, v19 // (op 8830) 


s_nop 0
v_readfirstlane_b32 s0, v116 // (op 8830) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v19, -4, v1 // (op 8887) 
v_sub_u32 v115, v1, v19 // (op 8887) 

v_and_b32 v19, 4294967292, v1 // (op 8887) 
v_add_u32 v116, v19, v115 // (op 8887) 


v_lshlrev_b32 v19, 2, v1 // (op 8945) 
v_and_b32 v115, -8, v1 // (op 9026) 
v_sub_u32 v117, v1, v115 // (op 9026) 

v_lshlrev_b32 v115, 4, v117 // (op 9026) 

v_lshrrev_b32 v117, 3, v1 // (op 9026) 
v_lshl_add_u32 v118, v117, 7, v115 // (op 9026) 


s_nop 0
v_readfirstlane_b32 s1, v118 // (op 9026) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v115, -8, v1 // (op 9070) 
v_sub_u32 v117, v1, v115 // (op 9070) 

v_lshlrev_b32 v115, 4, v117 // (op 9070) 

v_lshrrev_b32 v117, 3, v1 // (op 9070) 
v_lshl_add_u32 v118, v117, 7, v115 // (op 9070) 


s_nop 0
v_readfirstlane_b32 s18, v118 // (op 9070) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v115, -4, v1 // (op 9114) 
v_sub_u32 v117, v1, v115 // (op 9114) 

v_and_b32 v115, 4294967292, v1 // (op 9114) 
v_add_u32 v118, v115, v117 // (op 9114) 


v_lshlrev_b32 v115, 2, v1 // (op 9143) 
s_mov_b32 s19, 0 // (op 1394) call()
s_ashr_i64 s[22:23], s[42:43], 7 // (op 1393) 
s_mov_b32 s24, s22 // (op 1393) convert

s_and_b32 s25, s24, 4294967294 // (op 1393) 

s_cmp_lt_u32 s19, s25 // (op 1393) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopBottom_KLoop_1393 // (op 1393) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopBottom_KLoop_1393 if false)
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopTop_KLoop_1393:
s_add_u32 s19, s19, 2 // (op 1395) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[98:105], v[28:35], a[0:3], v97, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 88) 
s_barrier  // (op 9304) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[98:105], v[36:43], a[4:7], v97, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 376) 
s_lshl_b64 s[22:23], s[30:31], 5 // (op 8425) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[98:105], v[44:51], a[8:11], v97, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 552) 
v_mov_b64 v[120:121], s[22:23] // (op 8425) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[98:105], v[52:59], a[12:15], v97, v60 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 728) 
v_mov_b64 v[122:123], v[16:17] // (op 8425) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[98:105], v[62:69], a[16:19], v97, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 904) 
v_mov_b64 v[124:125], v[122:123] // (op 8425) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[98:105], v[70:77], a[20:23], v97, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1080) 
s_mov_b32 m0, s0 // (op 8425) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[98:105], v[80:87], a[24:27], v97, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1256) 
s_mov_b32 s22, 256 // (op 8425) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[98:105], v[88:95], a[28:31], v97, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1432) 
buffer_load_dwordx4 v124, s[60:63], s22 offen offset: 0 lds // (op 8425) Load value direct to lds
v_add_co_u32 v124, s[24:25], v124, v120 // (op 8425) least significant half
v_addc_co_u32 v125, s[24:25], 0, v125, s[24:25] // (op 8425) most significant half

s_add_u32 m0, m0, 4096 // (op 8425) 
buffer_load_dwordx4 v124, s[60:63], s22 offen offset: 0 lds // (op 8425) Load value direct to lds
v_add_co_u32 v124, s[24:25], v124, v120 // (op 8425) least significant half
v_addc_co_u32 v125, s[24:25], 0, v125, s[24:25] // (op 8425) most significant half

s_add_u32 m0, m0, 4096 // (op 8425) 
buffer_load_dwordx4 v124, s[60:63], s22 offen offset: 0 lds // (op 8425) Load value direct to lds
v_add_co_u32 v124, s[24:25], v124, v120 // (op 8425) least significant half
v_addc_co_u32 v125, s[24:25], 0, v125, s[24:25] // (op 8425) most significant half

s_add_u32 m0, m0, 4096 // (op 8425) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[106:113], v[28:35], a[32:35], v114, v25 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2323) 
buffer_load_dwordx4 v124, s[60:63], s22 offen offset: 0 lds // (op 8425) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[106:113], v[36:43], a[36:39], v114, v18 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2277) 
v_add_co_u32 v124, s[24:25], v124, v120 // (op 8425) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[106:113], v[44:51], a[40:43], v114, v22 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2231) 
v_addc_co_u32 v125, s[24:25], 0, v125, s[24:25] // (op 8425) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[106:113], v[52:59], a[44:47], v114, v60 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2185) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[106:113], v[62:69], a[48:51], v114, v61 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2139) 
s_add_u32 m0, m0, 4096 // (op 8425) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[106:113], v[70:77], a[52:55], v114, v78 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2093) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[106:113], v[80:87], a[56:59], v114, v79 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2047) 
buffer_load_dwordx4 v124, s[60:63], s22 offen offset: 0 lds // (op 8425) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[106:113], v[88:95], a[60:63], v114, v96 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2001) 
v_add_co_u32 v124, s[24:25], v124, v120 // (op 8425) least significant half
v_addc_co_u32 v125, s[24:25], 0, v125, s[24:25] // (op 8425) most significant half

s_add_u32 m0, m0, 4096 // (op 8425) 
buffer_load_dwordx4 v124, s[60:63], s22 offen offset: 0 lds // (op 8425) Load value direct to lds
v_add_co_u32 v124, s[24:25], v124, v120 // (op 8425) least significant half
v_addc_co_u32 v125, s[24:25], 0, v125, s[24:25] // (op 8425) most significant half

s_add_u32 m0, m0, 4096 // (op 8425) 
buffer_load_dwordx4 v124, s[60:63], s22 offen offset: 0 lds // (op 8425) Load value direct to lds
v_add_co_u32 v124, s[24:25], v124, v120 // (op 8425) least significant half
v_addc_co_u32 v125, s[24:25], 0, v125, s[24:25] // (op 8425) most significant half

s_add_u32 m0, m0, 4096 // (op 8425) 
buffer_load_dwordx4 v124, s[60:63], s22 offen offset: 0 lds // (op 8425) Load value direct to lds





v_mov_b32 v117, v26 // (op 25) 

buffer_load_dword v24, v117, s[68:71], 0 offen offset:8 // (op 25) Load value
s_barrier  // (op 9288) 

s_lshl_b64 s[22:23], s[26:27], 5 // (op 8420) 
v_mov_b64 v[120:121], s[22:23] // (op 8420) convert


v_mov_b64 v[124:125], v[20:21] // (op 8420) 
v_mov_b64 v[126:127], v[124:125] // (op 8420) 
s_add_u32 s22, 32768, s47 // (op 8420) 
s_mov_b32 m0, s22 // (op 8420) reg expression
s_mov_b32 s23, 256 // (op 8420) call()
buffer_load_dwordx4 v126, s[48:51], s23 offen offset: 0 lds // (op 8420) Load value direct to lds
v_add_co_u32 v126, s[24:25], v126, v120 // (op 8420) least significant half
v_addc_co_u32 v127, s[24:25], 0, v127, s[24:25] // (op 8420) most significant half

s_add_u32 m0, m0, 4096 // (op 8420) 
buffer_load_dwordx4 v126, s[48:51], s23 offen offset: 0 lds // (op 8420) Load value direct to lds






v_mov_b32 v119, v23 // (op 17) 

buffer_load_ubyte v27, v119, s[64:67], 0 offen offset:8 // (op 17) Load value
s_barrier  // (op 9376) 



v_add_u32 v120, 83200, v115 // (op 255) 
s_waitcnt vmcnt(15)
ds_write_b32 v120, v14 // (op 255) Store local data 

s_barrier  // (op 9370) 



v_add_u32 v120, 84224, v118 // (op 254) 
s_waitcnt vmcnt(12)
ds_write_b8 v120, v15 // (op 254) Store local data 

s_barrier  // (op 9359) 



ds_read_b128 v[126:129], v8 offset:40960 // (op 247) Load local data 
ds_read_b128 v[130:133], v8 offset:41024 // (op 247) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 9373) 



v_add_u32 v120, 83200, v9 // (op 251) 
ds_read_u8 v121, v120 // (op 251) Load local data 




ds_read_b128 v[134:137], v8 offset:45056 // (op 381) Load local data 
ds_read_b128 v[138:141], v8 offset:45120 // (op 381) Load local data 



v_add_u32 v120, 83328, v9 // (op 385) 
ds_read_u8 v142, v120 // (op 385) Load local data 




ds_read_b128 v[144:147], v8 offset:49152 // (op 557) Load local data 
ds_read_b128 v[148:151], v8 offset:49216 // (op 557) Load local data 



v_add_u32 v120, 83456, v9 // (op 561) 
ds_read_u8 v143, v120 // (op 561) Load local data 




ds_read_b128 v[152:155], v8 offset:53248 // (op 733) Load local data 
ds_read_b128 v[156:159], v8 offset:53312 // (op 733) Load local data 



v_add_u32 v120, 83584, v9 // (op 737) 
ds_read_u8 v160, v120 // (op 737) Load local data 




ds_read_b128 v[162:165], v8 offset:57344 // (op 909) Load local data 
ds_read_b128 v[166:169], v8 offset:57408 // (op 909) Load local data 



v_add_u32 v120, 83712, v9 // (op 913) 
ds_read_u8 v161, v120 // (op 913) Load local data 




ds_read_b128 v[170:173], v8 offset:61440 // (op 1085) Load local data 
ds_read_b128 v[174:177], v8 offset:61504 // (op 1085) Load local data 



v_add_u32 v120, 83840, v9 // (op 1089) 
ds_read_u8 v178, v120 // (op 1089) Load local data 




v_add_u32 v120, 65536, v8 // (op 1261) 
ds_read_b128 v[180:183], v120 // (op 1261) Load local data 

v_add_u32 v120, 65600, v8 // (op 1261) 
ds_read_b128 v[184:187], v120 // (op 1261) Load local data 




v_add_u32 v120, 83968, v9 // (op 1265) 
ds_read_u8 v179, v120 // (op 1265) Load local data 




v_add_u32 v120, 69632, v8 // (op 1437) 
ds_read_b128 v[188:191], v120 // (op 1437) Load local data 

v_add_u32 v120, 69696, v8 // (op 1437) 
ds_read_b128 v[192:195], v120 // (op 1437) Load local data 




v_add_u32 v120, 84096, v9 // (op 1441) 
ds_read_u8 v196, v120 // (op 1441) Load local data 

s_barrier  // (op 9353) 
s_barrier  // (op 9421) 



v_add_u32 v120, 73728, v7 // (op 245) 
ds_read_b128 v[98:101], v120 // (op 245) Load local data 

v_add_u32 v120, 73792, v7 // (op 245) 
ds_read_b128 v[102:105], v120 // (op 245) Load local data 

s_waitcnt lgkmcnt(15)
s_barrier  // (op 9366) 



v_add_u32 v120, 84224, v6 // (op 249) 
ds_read_u8 v97, v120 // (op 249) Load local data 




v_add_u32 v120, 77824, v7 // (op 2004) 
ds_read_b128 v[106:109], v120 // (op 2004) Load local data 

v_add_u32 v120, 77888, v7 // (op 2004) 
ds_read_b128 v[110:113], v120 // (op 2004) Load local data 




v_add_u32 v120, 84352, v6 // (op 2007) 
ds_read_u8 v114, v120 // (op 2007) Load local data 

s_barrier  // (op 9451) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[98:105], v[126:133], a[0:3], v97, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 252) 
s_barrier  // (op 9363) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[98:105], v[134:141], a[4:7], v97, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 377) 
s_lshl_b64 s[22:23], s[30:31], 5 // (op 8448) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[98:105], v[144:151], a[8:11], v97, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 553) 
v_mov_b64 v[198:199], s[22:23] // (op 8448) convert


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[98:105], v[152:159], a[12:15], v97, v160 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 729) 
v_mov_b64 v[200:201], v[122:123] // (op 8448) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[98:105], v[162:169], a[16:19], v97, v161 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 905) 
s_add_u32 s22, 40960, s18 // (op 8448) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[98:105], v[170:177], a[20:23], v97, v178 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1081) 
s_mov_b32 m0, s22 // (op 8448) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[98:105], v[180:187], a[24:27], v97, v179 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1257) 
s_mov_b32 s23, 384 // (op 8448) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[98:105], v[188:195], a[28:31], v97, v196 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1433) 
buffer_load_dwordx4 v200, s[60:63], s23 offen offset: 0 lds // (op 8448) Load value direct to lds
v_add_co_u32 v200, s[24:25], v200, v198 // (op 8448) least significant half
v_addc_co_u32 v201, s[24:25], 0, v201, s[24:25] // (op 8448) most significant half

s_add_u32 m0, m0, 4096 // (op 8448) 
buffer_load_dwordx4 v200, s[60:63], s23 offen offset: 0 lds // (op 8448) Load value direct to lds
v_add_co_u32 v200, s[24:25], v200, v198 // (op 8448) least significant half
v_addc_co_u32 v201, s[24:25], 0, v201, s[24:25] // (op 8448) most significant half

s_add_u32 m0, m0, 4096 // (op 8448) 
buffer_load_dwordx4 v200, s[60:63], s23 offen offset: 0 lds // (op 8448) Load value direct to lds
v_add_co_u32 v200, s[24:25], v200, v198 // (op 8448) least significant half
v_addc_co_u32 v201, s[24:25], 0, v201, s[24:25] // (op 8448) most significant half

s_add_u32 m0, m0, 4096 // (op 8448) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[106:113], v[126:133], a[32:35], v114, v121 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2324) 
buffer_load_dwordx4 v200, s[60:63], s23 offen offset: 0 lds // (op 8448) Load value direct to lds


v_add_co_u32 v200, s[24:25], v200, v198 // (op 8448) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[106:113], v[134:141], a[36:39], v114, v142 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2278) 
v_addc_co_u32 v201, s[24:25], 0, v201, s[24:25] // (op 8448) most significant half



v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[106:113], v[144:151], a[40:43], v114, v143 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2232) 
s_add_u32 m0, m0, 4096 // (op 8448) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[106:113], v[152:159], a[44:47], v114, v160 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2186) 
buffer_load_dwordx4 v200, s[60:63], s23 offen offset: 0 lds // (op 8448) Load value direct to lds


v_add_co_u32 v200, s[24:25], v200, v198 // (op 8448) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[106:113], v[162:169], a[48:51], v114, v161 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2140) 
v_addc_co_u32 v201, s[24:25], 0, v201, s[24:25] // (op 8448) most significant half



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[106:113], v[170:177], a[52:55], v114, v178 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2094) 
s_add_u32 m0, m0, 4096 // (op 8448) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[106:113], v[180:187], a[56:59], v114, v179 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2048) 
buffer_load_dwordx4 v200, s[60:63], s23 offen offset: 0 lds // (op 8448) Load value direct to lds


v_add_co_u32 v200, s[24:25], v200, v198 // (op 8448) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[106:113], v[188:195], a[60:63], v114, v196 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 2002) 
v_addc_co_u32 v201, s[24:25], 0, v201, s[24:25] // (op 8448) most significant half



s_add_u32 m0, m0, 4096 // (op 8448) 
buffer_load_dwordx4 v200, s[60:63], s23 offen offset: 0 lds // (op 8448) Load value direct to lds
v_add_co_u32 v200, s[24:25], v200, v198 // (op 8448) least significant half
v_addc_co_u32 v201, s[24:25], 0, v201, s[24:25] // (op 8448) most significant half

s_add_u32 m0, m0, 4096 // (op 8448) 
buffer_load_dwordx4 v200, s[60:63], s23 offen offset: 0 lds // (op 8448) Load value direct to lds








v_add_co_u32 v16, vcc, 256, v16 // (op 8594) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 8594) most significant half
buffer_load_dword v14, v117, s[68:71], 0 offen offset:12 // (op 258) Load value

s_barrier  // (op 9356) 
v_add_u32 v26, 8, v26 // (op 8722) 

s_lshl_b64 s[22:23], s[26:27], 5 // (op 8443) 
v_mov_b64 v[120:121], s[22:23] // (op 8443) convert


v_mov_b64 v[122:123], v[124:125] // (op 8443) 
s_add_u32 s22, 73728, s1 // (op 8443) 
s_mov_b32 m0, s22 // (op 8443) reg expression
s_mov_b32 s23, 384 // (op 8443) call()
buffer_load_dwordx4 v122, s[48:51], s23 offen offset: 0 lds // (op 8443) Load value direct to lds
v_add_co_u32 v122, s[24:25], v122, v120 // (op 8443) least significant half
v_addc_co_u32 v123, s[24:25], 0, v123, s[24:25] // (op 8443) most significant half

s_add_u32 m0, m0, 4096 // (op 8443) 
buffer_load_dwordx4 v122, s[48:51], s23 offen offset: 0 lds // (op 8443) Load value direct to lds








v_add_co_u32 v20, vcc, 256, v20 // (op 8530) least significant half
v_addc_co_u32 v21, vcc, 0, v21, vcc // (op 8530) most significant half
buffer_load_ubyte v15, v119, s[64:67], 0 offen offset:12 // (op 257) Load value

s_barrier  // (op 9339) 
v_add_u32 v23, 8, v23 // (op 8658) 



v_add_u32 v117, 81920, v19 // (op 30) 
s_waitcnt vmcnt(15)
ds_write_b32 v117, v24 // (op 30) Store local data 

s_barrier  // (op 9321) 



v_add_u32 v117, 82944, v116 // (op 21) 
s_waitcnt vmcnt(12)
ds_write_b8 v117, v27 // (op 21) Store local data 

s_barrier  // (op 9310) 



ds_read_b128 v[28:31], v4 // (op 75) Load local data 
ds_read_b128 v[32:35], v4 offset:64 // (op 75) Load local data 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 9346) 



v_add_u32 v117, 81920, v5 // (op 85) 
ds_read_u8 v25, v117 // (op 85) Load local data 




ds_read_b128 v[36:39], v4 offset:4096 // (op 371) Load local data 
ds_read_b128 v[40:43], v4 offset:4160 // (op 371) Load local data 



v_add_u32 v117, 82048, v5 // (op 375) 
ds_read_u8 v18, v117 // (op 375) Load local data 




ds_read_b128 v[44:47], v4 offset:8192 // (op 547) Load local data 
ds_read_b128 v[48:51], v4 offset:8256 // (op 547) Load local data 



v_add_u32 v117, 82176, v5 // (op 551) 
ds_read_u8 v22, v117 // (op 551) Load local data 




ds_read_b128 v[52:55], v4 offset:12288 // (op 723) Load local data 
ds_read_b128 v[56:59], v4 offset:12352 // (op 723) Load local data 



v_add_u32 v117, 82304, v5 // (op 727) 
ds_read_u8 v60, v117 // (op 727) Load local data 




ds_read_b128 v[62:65], v4 offset:16384 // (op 899) Load local data 
ds_read_b128 v[66:69], v4 offset:16448 // (op 899) Load local data 



v_add_u32 v117, 82432, v5 // (op 903) 
ds_read_u8 v61, v117 // (op 903) Load local data 




ds_read_b128 v[70:73], v4 offset:20480 // (op 1075) Load local data 
ds_read_b128 v[74:77], v4 offset:20544 // (op 1075) Load local data 



v_add_u32 v117, 82560, v5 // (op 1079) 
ds_read_u8 v78, v117 // (op 1079) Load local data 




ds_read_b128 v[80:83], v4 offset:24576 // (op 1251) Load local data 
ds_read_b128 v[84:87], v4 offset:24640 // (op 1251) Load local data 



v_add_u32 v117, 82688, v5 // (op 1255) 
ds_read_u8 v79, v117 // (op 1255) Load local data 




ds_read_b128 v[88:91], v4 offset:28672 // (op 1427) Load local data 
ds_read_b128 v[92:95], v4 offset:28736 // (op 1427) Load local data 



v_add_u32 v117, 82816, v5 // (op 1431) 
ds_read_u8 v96, v117 // (op 1431) Load local data 

s_barrier  // (op 9294) 



ds_read_b128 v[98:101], v3 offset:32768 // (op 70) Load local data 
ds_read_b128 v[102:105], v3 offset:32832 // (op 70) Load local data 
s_waitcnt lgkmcnt(15)
s_barrier  // (op 9328) 



v_add_u32 v117, 82944, v0 // (op 80) 
ds_read_u8 v97, v117 // (op 80) Load local data 




ds_read_b128 v[106:109], v3 offset:36864 // (op 1996) Load local data 
ds_read_b128 v[110:113], v3 offset:36928 // (op 1996) Load local data 



v_add_u32 v117, 83072, v0 // (op 1999) 
ds_read_u8 v114, v117 // (op 1999) Load local data 

s_ashr_i64 s[22:23], s[42:43], 7 // (op 1393) 
s_mov_b32 s24, s22 // (op 1393) convert

s_and_b32 s25, s24, 4294967294 // (op 1393) 

s_cmp_lt_u32 s19, s25 // (op 1393) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopTop_KLoop_1393 // (op 1393) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopTop_KLoop_1393 if true)
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopBottom_KLoop_1393:

























s_barrier  // (op 9411) 
s_mul_hi_i32 s47, s16, s10 // (op 8478) 
v_lshrrev_b32 v16, 3, v1 // (op 8478) 
s_add_i32 s0, s47, s16 // (op 8478) 

v_and_b32 v17, -8, v1 // (op 8478) 
s_ashr_i32 s1, s0, 31 // (op 8478) 
v_sub_u32 v18, v1, v17 // (op 8478) 

s_cmp_eq_i32 s10, 0 // (op 8478) 
s_mov_b32 s47, scc // (op 8478) 
v_lshlrev_b32 v17, 4, v18 // (op 8478) 

s_cmp_eq_i32 s47, 0 // (op 8478) 
s_cselect_b32 s18, 0, -1 // (op 8478) 

s_and_b32 s19, s13, 31 // (op 8478) 
s_lshl_b32 s47, 1, s19 // (op 8478) 
s_add_i32 s22, s47, s18 // (op 8478) 


s_and_b32 s18, s1, s22 // (op 8478) 


s_add_i32 s1, s0, s18 // (op 8478) 


s_ashr_i32 s0, s1, s19 // (op 8478) 


s_lshl_b32 s1, s13, 24 // (op 8478) 
s_ashr_i32 s47, s1, 31 // (op 8478) 

s_xor_b32 s1, s0, s47 // (op 8478) 

s_sub_i32 s0, s1, s47 // (op 8478) 


s_mul_hi_i32 s1, s0, s14 // (op 8478) 
s_add_i32 s47, s1, s0 // (op 8478) 

s_ashr_i32 s1, s47, 31 // (op 8478) 
s_cmp_eq_i32 s14, 0 // (op 8478) 
s_mov_b32 s18, scc // (op 8478) 
s_cmp_eq_i32 s18, 0 // (op 8478) 
s_cselect_b32 s19, 0, -1 // (op 8478) 

s_and_b32 s18, s15, 31 // (op 8478) 
s_lshl_b32 s22, 1, s18 // (op 8478) 
s_add_i32 s23, s22, s19 // (op 8478) 


s_and_b32 s19, s1, s23 // (op 8478) 


s_add_i32 s1, s47, s19 // (op 8478) 


s_ashr_i32 s19, s1, s18 // (op 8478) 


s_lshl_b32 s1, s15, 24 // (op 8478) 
s_ashr_i32 s18, s1, 31 // (op 8478) 

s_xor_b32 s1, s19, s18 // (op 8478) 

s_sub_i32 s19, s1, s18 // (op 8478) 


s_mul_i32 s1, s0, s11 // (op 8478) 

s_sub_u32 s0, s16, s1 // (op 8478) 

s_mul_i32 s1, s19, s11 // (op 8478) 
s_add_u32 s1, s1, s0 // (op 8478) 


s_mul_hi_i32 s0, s16, s9 // (op 8478) 
s_add_i32 s47, s0, s16 // (op 8478) 

s_ashr_i32 s0, s47, 31 // (op 8478) 
s_cmp_eq_i32 s9, 0 // (op 8478) 
s_mov_b32 s18, scc // (op 8478) 
s_cmp_eq_i32 s18, 0 // (op 8478) 
s_cselect_b32 s19, 0, -1 // (op 8478) 

s_and_b32 s18, s45, 31 // (op 8478) 
s_lshl_b32 s22, 1, s18 // (op 8478) 
s_add_i32 s23, s22, s19 // (op 8478) 


s_and_b32 s19, s0, s23 // (op 8478) 


s_add_i32 s0, s47, s19 // (op 8478) 


s_ashr_i32 s19, s0, s18 // (op 8478) 


s_lshl_b32 s0, s45, 24 // (op 8478) 
s_ashr_i32 s18, s0, 31 // (op 8478) 

s_xor_b32 s0, s19, s18 // (op 8478) 

s_sub_i32 s19, s0, s18 // (op 8478) 


s_add_u32 s22, s6, 256 // (op 8478) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8478) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 8478) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 8478) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 8478) 

s_mul_i32 s0, s11, s22 // (op 8478) 

s_mul_i32 s18, s8, s0 // (op 8478) 

s_mul_i32 s0, s19, s18 // (op 8478) 

s_sub_u32 s18, s16, s0 // (op 8478) 

s_mov_b32 s0, s18 // (op 8478) 
s_mul_hi_i32 s47, s0, s12 // (op 8478) 
s_add_i32 s22, s47, s0 // (op 8478) 


s_ashr_i32 s0, s22, 31 // (op 8478) 
s_cmp_eq_i32 s12, 0 // (op 8478) 
s_mov_b32 s47, scc // (op 8478) 
s_cmp_eq_i32 s47, 0 // (op 8478) 
s_cselect_b32 s23, 0, -1 // (op 8478) 

s_and_b32 s47, s44, 31 // (op 8478) 
s_lshl_b32 s24, 1, s47 // (op 8478) 
s_add_i32 s25, s24, s23 // (op 8478) 


s_and_b32 s23, s0, s25 // (op 8478) 


s_add_i32 s0, s22, s23 // (op 8478) 


s_ashr_i32 s22, s0, s47 // (op 8478) 


s_lshl_b32 s0, s44, 24 // (op 8478) 
s_ashr_i32 s47, s0, 31 // (op 8478) 

s_xor_b32 s0, s22, s47 // (op 8478) 

s_sub_i32 s22, s0, s47 // (op 8478) 


s_mul_i32 s0, s46, s11 // (op 8478) 
s_add_u32 s24, s20, 64 // (op 8478) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 8478) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 8478) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 8478) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 6 // (op 8478) 

s_sub_i32 s23, s24, s0 // (op 8478) 

s_mul_i32 s47, s22, s23 // (op 8478) 


s_sub_u32 s22, s18, s47 // (op 8478) 


s_add_u32 s18, s22, s0 // (op 8478) 


s_cmp_eq_i32 s19, 0 // (op 8478) 
s_mov_b32 s0, scc // (op 8478) 

s_cmp_eq_i32 s0, 0 // (op 8478) 
s_cselect_b32 s19, s18, s1 // (op 8478) 



v_lshl_add_u32 v18, s19, 6, v16 // (op 8478) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 8478) 
s_lshr_b32 s47, s0, 1 // (op 8478) 

v_lshl_add_u32 v16, s47, 8, v17 // (op 8478) 


v_mov_b32 v69, 0 // (op 8478) convert
v_mov_b32 v68, v16 // (op 8478) convert

v_mul_lo_u32 v17, v18, s27 // (op 8478) most significant: low of low * high
v_mul_hi_u32 v19, v18, s26 // (op 8478) most significant: high of low * low
v_mul_lo_u32 v16, v18, s26 // (op 8478) least significant: low of low * low
v_add_u32 v17, v17, v19 // (op 8478) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v68 // (op 8478) least significant half
v_addc_co_u32 v17, s[0:1], v17, v69, s[0:1] // (op 8478) most significant half



s_mul_i32 s1, s26, s21 // (op 8480) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 8480) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 8480) least significant
s_mul_hi_u32 s18, s26, s20 // (op 8480) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 8480) most significant: sum
s_add_u32 s1, s1, s18 // (op 8480) most significant: sum


s_mov_b32 s48, s28 // (op 8480) reg expression
s_mov_b32 s49, s29 // (op 8480) reg expression
s_mov_b32 s50, s0 // (op 8480) reg expression
s_mov_b32 s51, 131072 // (op 8480) call()

s_mul_hi_i32 s47, s16, s10 // (op 8560) 
v_lshrrev_b32 v18, 3, v1 // (op 8560) 
s_add_i32 s0, s47, s16 // (op 8560) 

v_and_b32 v19, -8, v1 // (op 8560) 
s_ashr_i32 s1, s0, 31 // (op 8560) 
v_sub_u32 v20, v1, v19 // (op 8560) 

s_cmp_eq_i32 s10, 0 // (op 8560) 
s_mov_b32 s47, scc // (op 8560) 
v_lshlrev_b32 v19, 4, v20 // (op 8560) 

s_cmp_eq_i32 s47, 0 // (op 8560) 
s_cselect_b32 s18, 0, -1 // (op 8560) 

s_and_b32 s19, s13, 31 // (op 8560) 
s_lshl_b32 s47, 1, s19 // (op 8560) 
s_add_i32 s22, s47, s18 // (op 8560) 


s_and_b32 s18, s1, s22 // (op 8560) 


s_add_i32 s1, s0, s18 // (op 8560) 


s_ashr_i32 s0, s1, s19 // (op 8560) 


s_lshl_b32 s1, s13, 24 // (op 8560) 
s_ashr_i32 s47, s1, 31 // (op 8560) 

s_xor_b32 s1, s0, s47 // (op 8560) 

s_sub_i32 s0, s1, s47 // (op 8560) 


s_mul_hi_i32 s1, s0, s14 // (op 8560) 
s_add_i32 s47, s1, s0 // (op 8560) 

s_ashr_i32 s1, s47, 31 // (op 8560) 
s_cmp_eq_i32 s14, 0 // (op 8560) 
s_mov_b32 s18, scc // (op 8560) 
s_cmp_eq_i32 s18, 0 // (op 8560) 
s_cselect_b32 s19, 0, -1 // (op 8560) 

s_and_b32 s18, s15, 31 // (op 8560) 
s_lshl_b32 s22, 1, s18 // (op 8560) 
s_add_i32 s23, s22, s19 // (op 8560) 


s_and_b32 s19, s1, s23 // (op 8560) 


s_add_i32 s1, s47, s19 // (op 8560) 


s_ashr_i32 s19, s1, s18 // (op 8560) 


s_lshl_b32 s1, s15, 24 // (op 8560) 
s_ashr_i32 s18, s1, 31 // (op 8560) 

s_xor_b32 s1, s19, s18 // (op 8560) 

s_sub_i32 s19, s1, s18 // (op 8560) 


s_add_u32 s22, s6, 256 // (op 8560) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8560) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 8560) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 8560) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 8560) 

s_mul_i32 s1, s19, s22 // (op 8560) 

s_sub_i32 s47, s0, s1 // (op 8560) 


s_mul_hi_i32 s0, s16, s9 // (op 8560) 
s_add_i32 s1, s0, s16 // (op 8560) 

s_ashr_i32 s0, s1, 31 // (op 8560) 
s_cmp_eq_i32 s9, 0 // (op 8560) 
s_mov_b32 s18, scc // (op 8560) 
s_cmp_eq_i32 s18, 0 // (op 8560) 
s_cselect_b32 s19, 0, -1 // (op 8560) 

s_and_b32 s18, s45, 31 // (op 8560) 
s_lshl_b32 s24, 1, s18 // (op 8560) 
s_add_i32 s25, s24, s19 // (op 8560) 


s_and_b32 s19, s0, s25 // (op 8560) 


s_add_i32 s0, s1, s19 // (op 8560) 


s_ashr_i32 s1, s0, s18 // (op 8560) 


s_lshl_b32 s0, s45, 24 // (op 8560) 
s_ashr_i32 s18, s0, 31 // (op 8560) 

s_xor_b32 s0, s1, s18 // (op 8560) 

s_sub_i32 s1, s0, s18 // (op 8560) 


s_mul_i32 s0, s11, s22 // (op 8560) 

s_mul_i32 s18, s8, s0 // (op 8560) 

s_mul_i32 s0, s1, s18 // (op 8560) 

s_sub_u32 s18, s16, s0 // (op 8560) 

s_mul_hi_i32 s0, s18, s12 // (op 8560) 
s_add_i32 s19, s0, s18 // (op 8560) 


s_ashr_i32 s0, s19, 31 // (op 8560) 
s_cmp_eq_i32 s12, 0 // (op 8560) 
s_mov_b32 s18, scc // (op 8560) 
s_cmp_eq_i32 s18, 0 // (op 8560) 
s_cselect_b32 s22, 0, -1 // (op 8560) 

s_and_b32 s18, s44, 31 // (op 8560) 
s_lshl_b32 s23, 1, s18 // (op 8560) 
s_add_i32 s24, s23, s22 // (op 8560) 


s_and_b32 s25, s0, s24 // (op 8560) 


s_add_i32 s0, s19, s25 // (op 8560) 


s_ashr_i32 s19, s0, s18 // (op 8560) 


s_lshl_b32 s0, s44, 24 // (op 8560) 
s_ashr_i32 s18, s0, 31 // (op 8560) 

s_xor_b32 s0, s19, s18 // (op 8560) 

s_sub_i32 s19, s0, s18 // (op 8560) 


s_cmp_eq_i32 s1, 0 // (op 8560) 
s_mov_b32 s0, scc // (op 8560) 

s_cmp_eq_i32 s0, 0 // (op 8560) 
s_cselect_b32 s1, s19, s47 // (op 8560) 



v_lshl_add_u32 v20, s1, 8, v18 // (op 8560) 


v_mul_lo_u32 v69, v20, s31 // (op 8560) most significant: low of low * high
v_mul_hi_u32 v18, v20, s30 // (op 8560) most significant: high of low * low
v_mul_lo_u32 v68, v20, s30 // (op 8560) least significant: low of low * low
v_add_u32 v69, v69, v18 // (op 8560) most significant: sum


s_ashr_i64 s[0:1], s[42:43], 7 // (op 8560) 
s_lshr_b32 s47, s0, 1 // (op 8560) 

v_lshl_add_u32 v18, s47, 8, v19 // (op 8560) 


v_mov_b32 v67, 0 // (op 8560) convert
v_mov_b32 v66, v18 // (op 8560) convert

v_add_co_u32 v18, s[0:1], v66, v68 // (op 8560) least significant half
v_addc_co_u32 v19, s[0:1], v67, v69, s[0:1] // (op 8560) most significant half



s_mul_i32 s1, s30, s7 // (op 8562) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 8562) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 8562) least significant
s_mul_hi_u32 s18, s30, s6 // (op 8562) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 8562) most significant: sum
s_add_u32 s1, s1, s18 // (op 8562) most significant: sum


s_mov_b32 s60, s32 // (op 8562) reg expression
s_mov_b32 s61, s33 // (op 8562) reg expression
s_mov_b32 s62, s0 // (op 8562) reg expression
s_mov_b32 s63, 131072 // (op 8562) call()

s_mul_hi_i32 s47, s16, s10 // (op 8624) 
v_lshrrev_b32 v20, 2, v1 // (op 8624) 
s_add_i32 s0, s47, s16 // (op 8624) 

v_and_b32 v21, -4, v1 // (op 8624) 
s_ashr_i32 s1, s0, 31 // (op 8624) 
v_sub_u32 v22, v1, v21 // (op 8624) 

s_cmp_eq_i32 s10, 0 // (op 8624) 
s_mov_b32 s47, scc // (op 8624) 
s_cmp_eq_i32 s47, 0 // (op 8624) 
s_cselect_b32 s18, 0, -1 // (op 8624) 

s_and_b32 s19, s13, 31 // (op 8624) 
s_lshl_b32 s47, 1, s19 // (op 8624) 
s_add_i32 s22, s47, s18 // (op 8624) 


s_and_b32 s18, s1, s22 // (op 8624) 


s_add_i32 s1, s0, s18 // (op 8624) 


s_ashr_i32 s0, s1, s19 // (op 8624) 


s_lshl_b32 s1, s13, 24 // (op 8624) 
s_ashr_i32 s47, s1, 31 // (op 8624) 

s_xor_b32 s1, s0, s47 // (op 8624) 

s_sub_i32 s0, s1, s47 // (op 8624) 


s_mul_hi_i32 s1, s0, s14 // (op 8624) 
s_add_i32 s47, s1, s0 // (op 8624) 

s_ashr_i32 s1, s47, 31 // (op 8624) 
s_cmp_eq_i32 s14, 0 // (op 8624) 
s_mov_b32 s18, scc // (op 8624) 
s_cmp_eq_i32 s18, 0 // (op 8624) 
s_cselect_b32 s19, 0, -1 // (op 8624) 

s_and_b32 s18, s15, 31 // (op 8624) 
s_lshl_b32 s22, 1, s18 // (op 8624) 
s_add_i32 s23, s22, s19 // (op 8624) 


s_and_b32 s19, s1, s23 // (op 8624) 


s_add_i32 s1, s47, s19 // (op 8624) 


s_ashr_i32 s19, s1, s18 // (op 8624) 


s_lshl_b32 s1, s15, 24 // (op 8624) 
s_ashr_i32 s18, s1, 31 // (op 8624) 

s_xor_b32 s1, s19, s18 // (op 8624) 

s_sub_i32 s19, s1, s18 // (op 8624) 


s_mul_i32 s1, s0, s11 // (op 8624) 

s_sub_u32 s0, s16, s1 // (op 8624) 

s_mul_i32 s1, s19, s11 // (op 8624) 
s_add_u32 s1, s1, s0 // (op 8624) 


s_mul_hi_i32 s0, s16, s9 // (op 8624) 
s_add_i32 s47, s0, s16 // (op 8624) 

s_ashr_i32 s0, s47, 31 // (op 8624) 
s_cmp_eq_i32 s9, 0 // (op 8624) 
s_mov_b32 s18, scc // (op 8624) 
s_cmp_eq_i32 s18, 0 // (op 8624) 
s_cselect_b32 s19, 0, -1 // (op 8624) 

s_and_b32 s18, s45, 31 // (op 8624) 
s_lshl_b32 s22, 1, s18 // (op 8624) 
s_add_i32 s23, s22, s19 // (op 8624) 


s_and_b32 s19, s0, s23 // (op 8624) 


s_add_i32 s0, s47, s19 // (op 8624) 


s_ashr_i32 s19, s0, s18 // (op 8624) 


s_lshl_b32 s0, s45, 24 // (op 8624) 
s_ashr_i32 s18, s0, 31 // (op 8624) 

s_xor_b32 s0, s19, s18 // (op 8624) 

s_sub_i32 s19, s0, s18 // (op 8624) 


s_add_u32 s22, s6, 256 // (op 8624) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8624) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 8624) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 8624) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 8624) 

s_mul_i32 s0, s11, s22 // (op 8624) 

s_mul_i32 s18, s8, s0 // (op 8624) 

s_mul_i32 s0, s19, s18 // (op 8624) 

s_sub_u32 s18, s16, s0 // (op 8624) 

s_mov_b32 s0, s18 // (op 8624) 
s_mul_hi_i32 s47, s0, s12 // (op 8624) 
s_add_i32 s22, s47, s0 // (op 8624) 


s_ashr_i32 s0, s22, 31 // (op 8624) 
s_cmp_eq_i32 s12, 0 // (op 8624) 
s_mov_b32 s47, scc // (op 8624) 
s_cmp_eq_i32 s47, 0 // (op 8624) 
s_cselect_b32 s23, 0, -1 // (op 8624) 

s_and_b32 s47, s44, 31 // (op 8624) 
s_lshl_b32 s24, 1, s47 // (op 8624) 
s_add_i32 s25, s24, s23 // (op 8624) 


s_and_b32 s23, s0, s25 // (op 8624) 


s_add_i32 s0, s22, s23 // (op 8624) 


s_ashr_i32 s22, s0, s47 // (op 8624) 


s_lshl_b32 s0, s44, 24 // (op 8624) 
s_ashr_i32 s47, s0, 31 // (op 8624) 

s_xor_b32 s0, s22, s47 // (op 8624) 

s_sub_i32 s22, s0, s47 // (op 8624) 


s_mul_i32 s0, s46, s11 // (op 8624) 
s_add_u32 s24, s20, 64 // (op 8624) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 8624) End of Int64 add, unlocking SCC
s_sub_u32 s72, s24, 1 // (op 8624) least significant half; sets scc
s_subb_u32 s73, s25, 0 // (op 8624) most significant half; uses scc

s_ashr_i64 s[24:25], s[72:73], 6 // (op 8624) 

s_sub_i32 s23, s24, s0 // (op 8624) 

s_mul_i32 s47, s22, s23 // (op 8624) 


s_sub_u32 s22, s18, s47 // (op 8624) 


s_add_u32 s18, s22, s0 // (op 8624) 


s_cmp_eq_i32 s19, 0 // (op 8624) 
s_mov_b32 s0, scc // (op 8624) 

s_cmp_eq_i32 s0, 0 // (op 8624) 
s_cselect_b32 s19, s18, s1 // (op 8624) 



v_lshl_add_u32 v21, s19, 6, v20 // (op 8624) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 8624) 
s_lshr_b32 s47, s0, 1 // (op 8624) 

v_lshl_add_u32 v20, s47, 3, v22 // (op 8624) 


v_mul_lo_u32 v22, v21, s34 // (op 8624) 
v_add_u32 v22, v22, v20 // (op 8624) 


s_mul_i32 s1, s34, s21 // (op 8626) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 8626) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 8626) least significant
s_mul_hi_u32 s18, s34, s20 // (op 8626) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 8626) most significant: sum
s_add_u32 s1, s1, s18 // (op 8626) most significant: sum


s_mov_b32 s64, s36 // (op 8626) reg expression
s_mov_b32 s65, s37 // (op 8626) reg expression
s_mov_b32 s66, s0 // (op 8626) reg expression
s_mov_b32 s67, 131072 // (op 8626) call()

s_mul_hi_i32 s47, s16, s10 // (op 8688) 
s_add_i32 s0, s47, s16 // (op 8688) 

s_ashr_i32 s1, s0, 31 // (op 8688) 
s_cmp_eq_i32 s10, 0 // (op 8688) 
s_mov_b32 s47, scc // (op 8688) 
s_cmp_eq_i32 s47, 0 // (op 8688) 
s_cselect_b32 s18, 0, -1 // (op 8688) 

s_and_b32 s19, s13, 31 // (op 8688) 
s_lshl_b32 s47, 1, s19 // (op 8688) 
s_add_i32 s22, s47, s18 // (op 8688) 


s_and_b32 s18, s1, s22 // (op 8688) 


s_add_i32 s1, s0, s18 // (op 8688) 


s_ashr_i32 s0, s1, s19 // (op 8688) 


s_lshl_b32 s1, s13, 24 // (op 8688) 
s_ashr_i32 s47, s1, 31 // (op 8688) 

s_xor_b32 s1, s0, s47 // (op 8688) 

s_sub_i32 s0, s1, s47 // (op 8688) 


s_mul_hi_i32 s1, s0, s14 // (op 8688) 
s_add_i32 s47, s1, s0 // (op 8688) 

s_ashr_i32 s1, s47, 31 // (op 8688) 
s_cmp_eq_i32 s14, 0 // (op 8688) 
s_mov_b32 s18, scc // (op 8688) 
s_cmp_eq_i32 s18, 0 // (op 8688) 
s_cselect_b32 s19, 0, -1 // (op 8688) 

s_and_b32 s18, s15, 31 // (op 8688) 
s_lshl_b32 s22, 1, s18 // (op 8688) 
s_add_i32 s23, s22, s19 // (op 8688) 


s_and_b32 s19, s1, s23 // (op 8688) 


s_add_i32 s1, s47, s19 // (op 8688) 


s_ashr_i32 s19, s1, s18 // (op 8688) 


s_lshl_b32 s1, s15, 24 // (op 8688) 
s_ashr_i32 s18, s1, 31 // (op 8688) 

s_xor_b32 s1, s19, s18 // (op 8688) 

s_sub_i32 s19, s1, s18 // (op 8688) 


s_add_u32 s22, s6, 256 // (op 8688) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 8688) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 8688) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 8688) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 8 // (op 8688) 

s_mul_i32 s1, s19, s22 // (op 8688) 

s_sub_i32 s47, s0, s1 // (op 8688) 


s_mul_hi_i32 s0, s16, s9 // (op 8688) 
s_add_i32 s1, s0, s16 // (op 8688) 

s_ashr_i32 s0, s1, 31 // (op 8688) 
s_cmp_eq_i32 s9, 0 // (op 8688) 
s_mov_b32 s18, scc // (op 8688) 
s_cmp_eq_i32 s18, 0 // (op 8688) 
s_cselect_b32 s19, 0, -1 // (op 8688) 

s_and_b32 s18, s45, 31 // (op 8688) 
s_lshl_b32 s24, 1, s18 // (op 8688) 
s_add_i32 s25, s24, s19 // (op 8688) 


s_and_b32 s19, s0, s25 // (op 8688) 


s_add_i32 s0, s1, s19 // (op 8688) 


s_ashr_i32 s1, s0, s18 // (op 8688) 


s_lshl_b32 s0, s45, 24 // (op 8688) 
s_ashr_i32 s18, s0, 31 // (op 8688) 

s_xor_b32 s0, s1, s18 // (op 8688) 

s_sub_i32 s1, s0, s18 // (op 8688) 


s_mul_i32 s0, s11, s22 // (op 8688) 

s_mul_i32 s18, s8, s0 // (op 8688) 

s_mul_i32 s0, s1, s18 // (op 8688) 

s_sub_u32 s18, s16, s0 // (op 8688) 

s_mul_hi_i32 s0, s18, s12 // (op 8688) 
s_add_i32 s19, s0, s18 // (op 8688) 


s_ashr_i32 s0, s19, 31 // (op 8688) 
s_cmp_eq_i32 s12, 0 // (op 8688) 
s_mov_b32 s18, scc // (op 8688) 
s_cmp_eq_i32 s18, 0 // (op 8688) 
s_cselect_b32 s22, 0, -1 // (op 8688) 

s_and_b32 s18, s44, 31 // (op 8688) 
s_lshl_b32 s23, 1, s18 // (op 8688) 
s_add_i32 s24, s23, s22 // (op 8688) 


s_and_b32 s25, s0, s24 // (op 8688) 


s_add_i32 s0, s19, s25 // (op 8688) 


s_ashr_i32 s19, s0, s18 // (op 8688) 


s_lshl_b32 s0, s44, 24 // (op 8688) 
s_ashr_i32 s18, s0, 31 // (op 8688) 

s_xor_b32 s0, s19, s18 // (op 8688) 

s_sub_i32 s19, s0, s18 // (op 8688) 


s_cmp_eq_i32 s1, 0 // (op 8688) 
s_mov_b32 s0, scc // (op 8688) 

s_cmp_eq_i32 s0, 0 // (op 8688) 
s_cselect_b32 s1, s19, s47 // (op 8688) 



v_lshl_add_u32 v20, s1, 8, v1 // (op 8688) 

v_mul_lo_u32 v21, v20, s38 // (op 8688) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 8688) 
s_lshr_b32 s47, s0, 1 // (op 8688) 

v_lshl_add_u32 v20, s47, 3, v21 // (op 8688) 


s_mul_i32 s1, s38, s7 // (op 8690) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 8690) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 8690) least significant
s_mul_hi_u32 s18, s38, s6 // (op 8690) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 8690) most significant: sum
s_add_u32 s1, s1, s18 // (op 8690) most significant: sum


s_mov_b32 s68, s40 // (op 8690) reg expression
s_mov_b32 s69, s41 // (op 8690) reg expression
s_mov_b32 s70, s0 // (op 8690) reg expression
s_mov_b32 s71, 131072 // (op 8690) call()

v_and_b32 v21, -8, v1 // (op 8786) 
v_sub_u32 v23, v1, v21 // (op 8786) 

v_lshlrev_b32 v21, 4, v23 // (op 8786) 

v_lshrrev_b32 v23, 3, v1 // (op 8786) 
v_lshl_add_u32 v24, v23, 7, v21 // (op 8786) 


s_nop 0
v_readfirstlane_b32 s47, v24 // (op 8786) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -8, v1 // (op 8843) 
v_sub_u32 v23, v1, v21 // (op 8843) 

v_lshlrev_b32 v21, 4, v23 // (op 8843) 

v_lshrrev_b32 v23, 3, v1 // (op 8843) 
v_lshl_add_u32 v24, v23, 7, v21 // (op 8843) 


s_nop 0
v_readfirstlane_b32 s0, v24 // (op 8843) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v21, -4, v1 // (op 8900) 
v_sub_u32 v23, v1, v21 // (op 8900) 

v_and_b32 v21, 4294967292, v1 // (op 8900) 
v_add_u32 v24, v21, v23 // (op 8900) 


v_lshlrev_b32 v21, 2, v1 // (op 8958) 
s_ashr_i64 s[18:19], s[42:43], 7 // (op 1968) 
s_and_b32 s1, s18, 4294967294 // (op 1968) 

s_ashr_i64 s[18:19], s[42:43], 7 // (op 1967) 
s_mov_b32 s22, s18 // (op 1967) convert

s_cmp_lt_u32 s1, s22 // (op 1967) 













s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_1967 // (op 1967) Condition: Top (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_1967 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopTop_KLoopTail_1967:
s_barrier  // (op 9291) 
s_add_u32 s1, s1, 1 // (op 1969) 

s_lshl_b64 s[18:19], s[26:27], 5 // (op 8430) 
v_mov_b64 v[14:15], s[18:19] // (op 8430) convert


v_mov_b64 v[26:27], v[16:17] // (op 8430) 
v_mov_b64 v[28:29], v[26:27] // (op 8430) 
s_add_u32 s18, 32768, s47 // (op 8430) 
s_mov_b32 m0, s18 // (op 8430) reg expression
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 8430) Load value direct to lds
v_add_co_u32 v28, s[22:23], v28, v14 // (op 8430) least significant half
v_addc_co_u32 v29, s[22:23], 0, v29, s[22:23] // (op 8430) most significant half

s_add_u32 m0, m0, 4096 // (op 8430) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 8430) Load value direct to lds




v_add_co_u32 v16, vcc, 128, v16 // (op 8468) least significant half
v_addc_co_u32 v17, vcc, 0, v17, vcc // (op 8468) most significant half
s_barrier  // (op 9307) 

s_lshl_b64 s[18:19], s[30:31], 5 // (op 8436) 
v_mov_b64 v[14:15], s[18:19] // (op 8436) convert


v_mov_b64 v[26:27], v[18:19] // (op 8436) 
v_mov_b64 v[28:29], v[26:27] // (op 8436) 
s_mov_b32 m0, s0 // (op 8436) reg expression
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 8436) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v14 // (op 8436) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 8436) most significant half

s_add_u32 m0, m0, 4096 // (op 8436) 
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 8436) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v14 // (op 8436) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 8436) most significant half

s_add_u32 m0, m0, 4096 // (op 8436) 
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 8436) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v14 // (op 8436) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 8436) most significant half

s_add_u32 m0, m0, 4096 // (op 8436) 
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 8436) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v14 // (op 8436) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 8436) most significant half

s_add_u32 m0, m0, 4096 // (op 8436) 
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 8436) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v14 // (op 8436) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 8436) most significant half

s_add_u32 m0, m0, 4096 // (op 8436) 
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 8436) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v14 // (op 8436) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 8436) most significant half

s_add_u32 m0, m0, 4096 // (op 8436) 
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 8436) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v14 // (op 8436) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 8436) most significant half

s_add_u32 m0, m0, 4096 // (op 8436) 
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 8436) Load value direct to lds



v_add_co_u32 v18, vcc, 128, v18 // (op 8552) least significant half
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 8552) most significant half


v_mov_b32 v23, v22 // (op 191) 

buffer_load_ubyte v14, v23, s[64:67], 0 offen offset: 0 // (op 191) Load value

v_add_u32 v22, 4, v22 // (op 8616) 


s_barrier  // (op 9324) 
v_mov_b32 v15, v20 // (op 192) 




v_add_u32 v23, 82944, v24 // (op 188) 
buffer_load_dword v25, v15, s[68:71], 0 offen // (op 192) Load value

v_add_u32 v20, 4, v20 // (op 8680) 
s_waitcnt vmcnt(1)
ds_write_b8 v23, v14 // (op 188) Store local data 


s_barrier  // (op 9297) 



s_barrier  // (op 9342) 
ds_read_b128 v[26:29], v3 offset:32768 // (op 179) Load local data 



ds_read_b128 v[30:33], v3 offset:32832 // (op 179) Load local data 
v_add_u32 v23, 81920, v21 // (op 189) 
s_barrier  // (op 9314) 
s_waitcnt vmcnt(0)
ds_write_b32 v23, v25 // (op 189) Store local data 





ds_read_b128 v[34:37], v4 // (op 181) Load local data 
ds_read_b128 v[38:41], v4 offset:64 // (op 181) Load local data 
s_waitcnt lgkmcnt(5)
s_barrier  // (op 9332) 



v_add_u32 v23, 82944, v0 // (op 183) 
ds_read_u8 v25, v23 // (op 183) Load local data 

s_waitcnt lgkmcnt(3)
s_barrier  // (op 9349) 



v_add_u32 v23, 81920, v5 // (op 185) 
ds_read_u8 v14, v23 // (op 185) Load local data 




ds_read_b128 v[42:45], v4 offset:4096 // (op 350) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:33], v[34:41], a[0:3], v25, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 186) 
ds_read_b128 v[46:49], v4 offset:4160 // (op 350) Load local data 



v_add_u32 v15, 82048, v5 // (op 354) 
ds_read_u8 v23, v15 // (op 354) Load local data 




ds_read_b128 v[50:53], v4 offset:8192 // (op 526) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[26:33], v[42:49], a[4:7], v25, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 355) 
ds_read_b128 v[54:57], v4 offset:8256 // (op 526) Load local data 



v_add_u32 v15, 82176, v5 // (op 530) 
ds_read_u8 v58, v15 // (op 530) Load local data 




ds_read_b128 v[60:63], v4 offset:12288 // (op 702) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[26:33], v[50:57], a[8:11], v25, v58 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 531) 
ds_read_b128 v[64:67], v4 offset:12352 // (op 702) Load local data 



v_add_u32 v15, 82304, v5 // (op 706) 
ds_read_u8 v59, v15 // (op 706) Load local data 




ds_read_b128 v[68:71], v4 offset:16384 // (op 878) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[26:33], v[60:67], a[12:15], v25, v59 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 707) 
ds_read_b128 v[72:75], v4 offset:16448 // (op 878) Load local data 



v_add_u32 v15, 82432, v5 // (op 882) 
ds_read_u8 v76, v15 // (op 882) Load local data 




ds_read_b128 v[78:81], v4 offset:20480 // (op 1054) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[26:33], v[68:75], a[16:19], v25, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 883) 
ds_read_b128 v[82:85], v4 offset:20544 // (op 1054) Load local data 



v_add_u32 v15, 82560, v5 // (op 1058) 
ds_read_u8 v77, v15 // (op 1058) Load local data 




ds_read_b128 v[86:89], v4 offset:24576 // (op 1230) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[26:33], v[78:85], a[20:23], v25, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1059) 
ds_read_b128 v[90:93], v4 offset:24640 // (op 1230) Load local data 



v_add_u32 v15, 82688, v5 // (op 1234) 
ds_read_u8 v94, v15 // (op 1234) Load local data 




ds_read_b128 v[96:99], v4 offset:28672 // (op 1406) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[26:33], v[86:93], a[24:27], v25, v94 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1235) 
ds_read_b128 v[100:103], v4 offset:28736 // (op 1406) Load local data 



v_add_u32 v15, 82816, v5 // (op 1410) 
ds_read_u8 v95, v15 // (op 1410) Load local data 




ds_read_b128 v[104:107], v3 offset:36864 // (op 1756) Load local data 
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[26:33], v[96:103], a[28:31], v25, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1411) 
ds_read_b128 v[108:111], v3 offset:36928 // (op 1756) Load local data 





v_add_u32 v15, 83072, v0 // (op 1759) 
ds_read_u8 v25, v15 // (op 1759) Load local data 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[104:111], v[34:41], a[32:35], v25, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1761) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[104:111], v[42:49], a[36:39], v25, v23 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1792) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[104:111], v[50:57], a[40:43], v25, v58 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1823) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[104:111], v[60:67], a[44:47], v25, v59 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1854) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[104:111], v[68:75], a[48:51], v25, v76 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1885) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[104:111], v[78:85], a[52:55], v25, v77 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1916) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[104:111], v[86:93], a[56:59], v25, v94 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1947) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[104:111], v[96:103], a[60:63], v25, v95 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b001 // (op 1978) 




s_ashr_i64 s[18:19], s[42:43], 7 // (op 1967) 
s_mov_b32 s22, s18 // (op 1967) convert

s_cmp_lt_u32 s1, s22 // (op 1967) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopTop_KLoopTail_1967 // (op 1967) Condition: Bottom (jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopTop_KLoopTail_1967 if true)
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ForLoopBottom_KLoopTail_1967:






















s_barrier  // (op 9439) 





















v_cmp_eq_f32 vcc, 0.00000, v10 // (op 133) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ConditionalFalse_0_beta_133
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


v_accvgpr_read v6, a4 // (op 3573) 
v_accvgpr_read v7, a5 // (op 3573) 
v_accvgpr_read v8, a6 // (op 3573) 
v_accvgpr_read v9, a7 // (op 3573) 
v_mul_f32 v14, v11, v6 // (op 3573) 
v_mul_f32 v15, v11, v7 // (op 3573) 
v_mul_f32 v16, v11, v8 // (op 3573) 
v_mul_f32 v17, v11, v9 // (op 3573) 



s_lshl_b32 s0, s4, 7 // (op 3575) 
v_add_u32 v0, v13, s0 // (op 3575) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 3575) Store value


v_accvgpr_read v6, a8 // (op 3585) 
v_accvgpr_read v7, a9 // (op 3585) 
v_accvgpr_read v8, a10 // (op 3585) 
v_accvgpr_read v9, a11 // (op 3585) 
v_mul_f32 v14, v11, v6 // (op 3585) 
v_mul_f32 v15, v11, v7 // (op 3585) 
v_mul_f32 v16, v11, v8 // (op 3585) 
v_mul_f32 v17, v11, v9 // (op 3585) 



s_lshl_b32 s0, s4, 8 // (op 3587) 
v_add_u32 v0, v13, s0 // (op 3587) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 3587) Store value


v_accvgpr_read v6, a12 // (op 3597) 
v_accvgpr_read v7, a13 // (op 3597) 
v_accvgpr_read v8, a14 // (op 3597) 
v_accvgpr_read v9, a15 // (op 3597) 
v_mul_f32 v14, v11, v6 // (op 3597) 
v_mul_f32 v15, v11, v7 // (op 3597) 
v_mul_f32 v16, v11, v8 // (op 3597) 
v_mul_f32 v17, v11, v9 // (op 3597) 



s_lshl_b32 s0, s4, 7 // (op 3599) 
s_mul_i32 s1, 3, s0 // (op 3599) 

v_add_u32 v0, v13, s1 // (op 3599) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 3599) Store value


v_accvgpr_read v6, a16 // (op 3609) 
v_accvgpr_read v7, a17 // (op 3609) 
v_accvgpr_read v8, a18 // (op 3609) 
v_accvgpr_read v9, a19 // (op 3609) 
v_mul_f32 v14, v11, v6 // (op 3609) 
v_mul_f32 v15, v11, v7 // (op 3609) 
v_mul_f32 v16, v11, v8 // (op 3609) 
v_mul_f32 v17, v11, v9 // (op 3609) 



s_lshl_b32 s0, s4, 9 // (op 3611) 
v_add_u32 v0, v13, s0 // (op 3611) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 3611) Store value


v_accvgpr_read v6, a20 // (op 3621) 
v_accvgpr_read v7, a21 // (op 3621) 
v_accvgpr_read v8, a22 // (op 3621) 
v_accvgpr_read v9, a23 // (op 3621) 
v_mul_f32 v14, v11, v6 // (op 3621) 
v_mul_f32 v15, v11, v7 // (op 3621) 
v_mul_f32 v16, v11, v8 // (op 3621) 
v_mul_f32 v17, v11, v9 // (op 3621) 



s_lshl_b32 s0, s4, 7 // (op 3623) 
s_mul_i32 s1, 5, s0 // (op 3623) 

v_add_u32 v0, v13, s1 // (op 3623) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 3623) Store value


v_accvgpr_read v6, a24 // (op 3633) 
v_accvgpr_read v7, a25 // (op 3633) 
v_accvgpr_read v8, a26 // (op 3633) 
v_accvgpr_read v9, a27 // (op 3633) 
v_mul_f32 v14, v11, v6 // (op 3633) 
v_mul_f32 v15, v11, v7 // (op 3633) 
v_mul_f32 v16, v11, v8 // (op 3633) 
v_mul_f32 v17, v11, v9 // (op 3633) 



s_lshl_b32 s0, s4, 7 // (op 3635) 
s_mul_i32 s1, 6, s0 // (op 3635) 

v_add_u32 v0, v13, s1 // (op 3635) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 3635) Store value


v_accvgpr_read v6, a28 // (op 3645) 
v_accvgpr_read v7, a29 // (op 3645) 
v_accvgpr_read v8, a30 // (op 3645) 
v_accvgpr_read v9, a31 // (op 3645) 
v_mul_f32 v14, v11, v6 // (op 3645) 
v_mul_f32 v15, v11, v7 // (op 3645) 
v_mul_f32 v16, v11, v8 // (op 3645) 
v_mul_f32 v17, v11, v9 // (op 3645) 



s_lshl_b32 s0, s4, 7 // (op 3647) 
s_mul_i32 s1, 7, s0 // (op 3647) 

v_add_u32 v0, v13, s1 // (op 3647) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 3647) Store value


v_accvgpr_read v6, a32 // (op 3699) 
v_accvgpr_read v7, a33 // (op 3699) 
v_accvgpr_read v8, a34 // (op 3699) 
v_accvgpr_read v9, a35 // (op 3699) 
v_mul_f32 v14, v11, v6 // (op 3699) 
v_mul_f32 v15, v11, v7 // (op 3699) 
v_mul_f32 v16, v11, v8 // (op 3699) 
v_mul_f32 v17, v11, v9 // (op 3699) 



v_add_u32 v0, 128, v13 // (op 3702) 

buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 3702) Store value


v_accvgpr_read v6, a36 // (op 3720) 
v_accvgpr_read v7, a37 // (op 3720) 
v_accvgpr_read v8, a38 // (op 3720) 
v_accvgpr_read v9, a39 // (op 3720) 
v_mul_f32 v14, v11, v6 // (op 3720) 
v_mul_f32 v15, v11, v7 // (op 3720) 
v_mul_f32 v16, v11, v8 // (op 3720) 
v_mul_f32 v17, v11, v9 // (op 3720) 



s_lshl_b32 s0, s4, 7 // (op 3704) 
s_add_u32 s0, s0, 128 // (op 3704) 
v_add_u32 v0, v13, s0 // (op 3704) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 3704) Store value


v_accvgpr_read v6, a40 // (op 3725) 
v_accvgpr_read v7, a41 // (op 3725) 
v_accvgpr_read v8, a42 // (op 3725) 
v_accvgpr_read v9, a43 // (op 3725) 
v_mul_f32 v14, v11, v6 // (op 3725) 
v_mul_f32 v15, v11, v7 // (op 3725) 
v_mul_f32 v16, v11, v8 // (op 3725) 
v_mul_f32 v17, v11, v9 // (op 3725) 



s_lshl_b32 s0, s4, 8 // (op 3706) 
s_add_u32 s0, s0, 128 // (op 3706) 
v_add_u32 v0, v13, s0 // (op 3706) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 3706) Store value


v_accvgpr_read v6, a44 // (op 3730) 
v_accvgpr_read v7, a45 // (op 3730) 
v_accvgpr_read v8, a46 // (op 3730) 
v_accvgpr_read v9, a47 // (op 3730) 
v_mul_f32 v14, v11, v6 // (op 3730) 
v_mul_f32 v15, v11, v7 // (op 3730) 
v_mul_f32 v16, v11, v8 // (op 3730) 
v_mul_f32 v17, v11, v9 // (op 3730) 



s_lshl_b32 s0, s4, 7 // (op 3708) 
s_mul_i32 s1, 3, s0 // (op 3708) 
s_add_u32 s1, s1, 128 // (op 3708) 

v_add_u32 v0, v13, s1 // (op 3708) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 3708) Store value


v_accvgpr_read v6, a48 // (op 3735) 
v_accvgpr_read v7, a49 // (op 3735) 
v_accvgpr_read v8, a50 // (op 3735) 
v_accvgpr_read v9, a51 // (op 3735) 
v_mul_f32 v14, v11, v6 // (op 3735) 
v_mul_f32 v15, v11, v7 // (op 3735) 
v_mul_f32 v16, v11, v8 // (op 3735) 
v_mul_f32 v17, v11, v9 // (op 3735) 



s_lshl_b32 s0, s4, 9 // (op 3710) 
s_add_u32 s0, s0, 128 // (op 3710) 
v_add_u32 v0, v13, s0 // (op 3710) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 3710) Store value


v_accvgpr_read v6, a52 // (op 3740) 
v_accvgpr_read v7, a53 // (op 3740) 
v_accvgpr_read v8, a54 // (op 3740) 
v_accvgpr_read v9, a55 // (op 3740) 
v_mul_f32 v14, v11, v6 // (op 3740) 
v_mul_f32 v15, v11, v7 // (op 3740) 
v_mul_f32 v16, v11, v8 // (op 3740) 
v_mul_f32 v17, v11, v9 // (op 3740) 



s_lshl_b32 s0, s4, 7 // (op 3712) 
s_mul_i32 s1, 5, s0 // (op 3712) 
s_add_u32 s1, s1, 128 // (op 3712) 

v_add_u32 v0, v13, s1 // (op 3712) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 3712) Store value


v_accvgpr_read v6, a56 // (op 3745) 
v_accvgpr_read v7, a57 // (op 3745) 
v_accvgpr_read v8, a58 // (op 3745) 
v_accvgpr_read v9, a59 // (op 3745) 
v_mul_f32 v14, v11, v6 // (op 3745) 
v_mul_f32 v15, v11, v7 // (op 3745) 
v_mul_f32 v16, v11, v8 // (op 3745) 
v_mul_f32 v17, v11, v9 // (op 3745) 



s_lshl_b32 s0, s4, 7 // (op 3714) 
s_mul_i32 s1, 6, s0 // (op 3714) 
s_add_u32 s1, s1, 128 // (op 3714) 

v_add_u32 v0, v13, s1 // (op 3714) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 3714) Store value


v_accvgpr_read v6, a60 // (op 3750) 
v_accvgpr_read v7, a61 // (op 3750) 
v_accvgpr_read v8, a62 // (op 3750) 
v_accvgpr_read v9, a63 // (op 3750) 
v_mul_f32 v14, v11, v6 // (op 3750) 
v_mul_f32 v15, v11, v7 // (op 3750) 
v_mul_f32 v16, v11, v8 // (op 3750) 
v_mul_f32 v17, v11, v9 // (op 3750) 



s_lshl_b32 s0, s4, 7 // (op 3716) 
s_mul_i32 s1, 7, s0 // (op 3716) 
s_add_u32 s1, s1, 128 // (op 3716) 

v_add_u32 v0, v13, s1 // (op 3716) 


buffer_store_dwordx4 v[14:17], v0, s[56:59], 0 offen // (op 3716) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ConditionalBottom_0_beta_133
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ConditionalFalse_0_beta_133:


v_add_u32 v0, 128, v12 // (op 4064) 

buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 4064) Load value

v_accvgpr_read v14, a32 // (op 4044) 
v_accvgpr_read v15, a33 // (op 4044) 
v_accvgpr_read v16, a34 // (op 4044) 
v_accvgpr_read v17, a35 // (op 4044) 
v_mul_f32 v18, v11, v14 // (op 4044) 
v_mul_f32 v19, v11, v15 // (op 4044) 
v_mul_f32 v20, v11, v16 // (op 4044) 
v_mul_f32 v21, v11, v17 // (op 4044) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4063) 
v_mul_f32 v15, v10, v7 // (op 4063) 
v_mul_f32 v16, v10, v8 // (op 4063) 
v_mul_f32 v17, v10, v9 // (op 4063) 

v_add_f32 v6, v14, v18 // (op 4045) 
v_add_f32 v7, v15, v19 // (op 4045) 
v_add_f32 v8, v16, v20 // (op 4045) 
v_add_f32 v9, v17, v21 // (op 4045) 




v_add_u32 v0, 128, v13 // (op 4047) 

buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 4047) Store value




s_lshl_b32 s0, s2, 7 // (op 4068) 
s_add_u32 s0, s0, 128 // (op 4068) 
v_add_u32 v0, v12, s0 // (op 4068) 


buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 4068) Load value

v_accvgpr_read v14, a36 // (op 4094) 
v_accvgpr_read v15, a37 // (op 4094) 
v_accvgpr_read v16, a38 // (op 4094) 
v_accvgpr_read v17, a39 // (op 4094) 
v_mul_f32 v18, v11, v14 // (op 4094) 
v_mul_f32 v19, v11, v15 // (op 4094) 
v_mul_f32 v20, v11, v16 // (op 4094) 
v_mul_f32 v21, v11, v17 // (op 4094) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4066) 
v_mul_f32 v15, v10, v7 // (op 4066) 
v_mul_f32 v16, v10, v8 // (op 4066) 
v_mul_f32 v17, v10, v9 // (op 4066) 

v_add_f32 v6, v14, v18 // (op 4067) 
v_add_f32 v7, v15, v19 // (op 4067) 
v_add_f32 v8, v16, v20 // (op 4067) 
v_add_f32 v9, v17, v21 // (op 4067) 




s_lshl_b32 s0, s4, 7 // (op 4049) 
s_add_u32 s0, s0, 128 // (op 4049) 
v_add_u32 v0, v13, s0 // (op 4049) 


buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 4049) Store value




s_lshl_b32 s0, s2, 8 // (op 4072) 
s_add_u32 s0, s0, 128 // (op 4072) 
v_add_u32 v0, v12, s0 // (op 4072) 


buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 4072) Load value

v_accvgpr_read v14, a40 // (op 4096) 
v_accvgpr_read v15, a41 // (op 4096) 
v_accvgpr_read v16, a42 // (op 4096) 
v_accvgpr_read v17, a43 // (op 4096) 
v_mul_f32 v18, v11, v14 // (op 4096) 
v_mul_f32 v19, v11, v15 // (op 4096) 
v_mul_f32 v20, v11, v16 // (op 4096) 
v_mul_f32 v21, v11, v17 // (op 4096) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4070) 
v_mul_f32 v15, v10, v7 // (op 4070) 
v_mul_f32 v16, v10, v8 // (op 4070) 
v_mul_f32 v17, v10, v9 // (op 4070) 

v_add_f32 v6, v14, v18 // (op 4071) 
v_add_f32 v7, v15, v19 // (op 4071) 
v_add_f32 v8, v16, v20 // (op 4071) 
v_add_f32 v9, v17, v21 // (op 4071) 




s_lshl_b32 s0, s4, 8 // (op 4051) 
s_add_u32 s0, s0, 128 // (op 4051) 
v_add_u32 v0, v13, s0 // (op 4051) 


buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 4051) Store value




s_lshl_b32 s0, s2, 7 // (op 4076) 
s_mul_i32 s1, 3, s0 // (op 4076) 
s_add_u32 s1, s1, 128 // (op 4076) 

v_add_u32 v0, v12, s1 // (op 4076) 


buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 4076) Load value

v_accvgpr_read v14, a44 // (op 4098) 
v_accvgpr_read v15, a45 // (op 4098) 
v_accvgpr_read v16, a46 // (op 4098) 
v_accvgpr_read v17, a47 // (op 4098) 
v_mul_f32 v18, v11, v14 // (op 4098) 
v_mul_f32 v19, v11, v15 // (op 4098) 
v_mul_f32 v20, v11, v16 // (op 4098) 
v_mul_f32 v21, v11, v17 // (op 4098) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4074) 
v_mul_f32 v15, v10, v7 // (op 4074) 
v_mul_f32 v16, v10, v8 // (op 4074) 
v_mul_f32 v17, v10, v9 // (op 4074) 

v_add_f32 v6, v14, v18 // (op 4075) 
v_add_f32 v7, v15, v19 // (op 4075) 
v_add_f32 v8, v16, v20 // (op 4075) 
v_add_f32 v9, v17, v21 // (op 4075) 




s_lshl_b32 s0, s4, 7 // (op 4053) 
s_mul_i32 s1, 3, s0 // (op 4053) 
s_add_u32 s1, s1, 128 // (op 4053) 

v_add_u32 v0, v13, s1 // (op 4053) 


buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 4053) Store value




s_lshl_b32 s0, s2, 9 // (op 4080) 
s_add_u32 s0, s0, 128 // (op 4080) 
v_add_u32 v0, v12, s0 // (op 4080) 


buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 4080) Load value

v_accvgpr_read v14, a48 // (op 4100) 
v_accvgpr_read v15, a49 // (op 4100) 
v_accvgpr_read v16, a50 // (op 4100) 
v_accvgpr_read v17, a51 // (op 4100) 
v_mul_f32 v18, v11, v14 // (op 4100) 
v_mul_f32 v19, v11, v15 // (op 4100) 
v_mul_f32 v20, v11, v16 // (op 4100) 
v_mul_f32 v21, v11, v17 // (op 4100) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4078) 
v_mul_f32 v15, v10, v7 // (op 4078) 
v_mul_f32 v16, v10, v8 // (op 4078) 
v_mul_f32 v17, v10, v9 // (op 4078) 

v_add_f32 v6, v14, v18 // (op 4079) 
v_add_f32 v7, v15, v19 // (op 4079) 
v_add_f32 v8, v16, v20 // (op 4079) 
v_add_f32 v9, v17, v21 // (op 4079) 




s_lshl_b32 s0, s4, 9 // (op 4055) 
s_add_u32 s0, s0, 128 // (op 4055) 
v_add_u32 v0, v13, s0 // (op 4055) 


buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 4055) Store value




s_lshl_b32 s0, s2, 7 // (op 4084) 
s_mul_i32 s1, 5, s0 // (op 4084) 
s_add_u32 s1, s1, 128 // (op 4084) 

v_add_u32 v0, v12, s1 // (op 4084) 


buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 4084) Load value

v_accvgpr_read v14, a52 // (op 4102) 
v_accvgpr_read v15, a53 // (op 4102) 
v_accvgpr_read v16, a54 // (op 4102) 
v_accvgpr_read v17, a55 // (op 4102) 
v_mul_f32 v18, v11, v14 // (op 4102) 
v_mul_f32 v19, v11, v15 // (op 4102) 
v_mul_f32 v20, v11, v16 // (op 4102) 
v_mul_f32 v21, v11, v17 // (op 4102) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4082) 
v_mul_f32 v15, v10, v7 // (op 4082) 
v_mul_f32 v16, v10, v8 // (op 4082) 
v_mul_f32 v17, v10, v9 // (op 4082) 

v_add_f32 v6, v14, v18 // (op 4083) 
v_add_f32 v7, v15, v19 // (op 4083) 
v_add_f32 v8, v16, v20 // (op 4083) 
v_add_f32 v9, v17, v21 // (op 4083) 




s_lshl_b32 s0, s4, 7 // (op 4057) 
s_mul_i32 s1, 5, s0 // (op 4057) 
s_add_u32 s1, s1, 128 // (op 4057) 

v_add_u32 v0, v13, s1 // (op 4057) 


buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 4057) Store value




s_lshl_b32 s0, s2, 7 // (op 4088) 
s_mul_i32 s1, 6, s0 // (op 4088) 
s_add_u32 s1, s1, 128 // (op 4088) 

v_add_u32 v0, v12, s1 // (op 4088) 


buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 4088) Load value

v_accvgpr_read v14, a56 // (op 4104) 
v_accvgpr_read v15, a57 // (op 4104) 
v_accvgpr_read v16, a58 // (op 4104) 
v_accvgpr_read v17, a59 // (op 4104) 
v_mul_f32 v18, v11, v14 // (op 4104) 
v_mul_f32 v19, v11, v15 // (op 4104) 
v_mul_f32 v20, v11, v16 // (op 4104) 
v_mul_f32 v21, v11, v17 // (op 4104) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4086) 
v_mul_f32 v15, v10, v7 // (op 4086) 
v_mul_f32 v16, v10, v8 // (op 4086) 
v_mul_f32 v17, v10, v9 // (op 4086) 

v_add_f32 v6, v14, v18 // (op 4087) 
v_add_f32 v7, v15, v19 // (op 4087) 
v_add_f32 v8, v16, v20 // (op 4087) 
v_add_f32 v9, v17, v21 // (op 4087) 




s_lshl_b32 s0, s4, 7 // (op 4059) 
s_mul_i32 s1, 6, s0 // (op 4059) 
s_add_u32 s1, s1, 128 // (op 4059) 

v_add_u32 v0, v13, s1 // (op 4059) 


buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 4059) Store value




s_lshl_b32 s0, s2, 7 // (op 4092) 
s_mul_i32 s1, 7, s0 // (op 4092) 
s_add_u32 s1, s1, 128 // (op 4092) 

v_add_u32 v0, v12, s1 // (op 4092) 


buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 4092) Load value

v_accvgpr_read v14, a60 // (op 4106) 
v_accvgpr_read v15, a61 // (op 4106) 
v_accvgpr_read v16, a62 // (op 4106) 
v_accvgpr_read v17, a63 // (op 4106) 
v_mul_f32 v18, v11, v14 // (op 4106) 
v_mul_f32 v19, v11, v15 // (op 4106) 
v_mul_f32 v20, v11, v16 // (op 4106) 
v_mul_f32 v21, v11, v17 // (op 4106) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 4090) 
v_mul_f32 v15, v10, v7 // (op 4090) 
v_mul_f32 v16, v10, v8 // (op 4090) 
v_mul_f32 v17, v10, v9 // (op 4090) 

v_add_f32 v6, v14, v18 // (op 4091) 
v_add_f32 v7, v15, v19 // (op 4091) 
v_add_f32 v8, v16, v20 // (op 4091) 
v_add_f32 v9, v17, v21 // (op 4091) 




s_lshl_b32 s0, s4, 7 // (op 4061) 
s_mul_i32 s1, 7, s0 // (op 4061) 
s_add_u32 s1, s1, 128 // (op 4061) 

v_add_u32 v0, v13, s1 // (op 4061) 


buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 4061) Store value




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




s_lshl_b32 s0, s2, 7 // (op 3882) 
v_add_u32 v0, v12, s0 // (op 3882) 


buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 3882) Load value

v_accvgpr_read v14, a4 // (op 3879) 
v_accvgpr_read v15, a5 // (op 3879) 
v_accvgpr_read v16, a6 // (op 3879) 
v_accvgpr_read v17, a7 // (op 3879) 
v_mul_f32 v18, v11, v14 // (op 3879) 
v_mul_f32 v19, v11, v15 // (op 3879) 
v_mul_f32 v20, v11, v16 // (op 3879) 
v_mul_f32 v21, v11, v17 // (op 3879) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3883) 
v_mul_f32 v15, v10, v7 // (op 3883) 
v_mul_f32 v16, v10, v8 // (op 3883) 
v_mul_f32 v17, v10, v9 // (op 3883) 

v_add_f32 v6, v14, v18 // (op 3880) 
v_add_f32 v7, v15, v19 // (op 3880) 
v_add_f32 v8, v16, v20 // (op 3880) 
v_add_f32 v9, v17, v21 // (op 3880) 




s_lshl_b32 s0, s4, 7 // (op 3881) 
v_add_u32 v0, v13, s0 // (op 3881) 


buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 3881) Store value




s_lshl_b32 s0, s2, 8 // (op 3894) 
v_add_u32 v0, v12, s0 // (op 3894) 


buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 3894) Load value

v_accvgpr_read v14, a8 // (op 3891) 
v_accvgpr_read v15, a9 // (op 3891) 
v_accvgpr_read v16, a10 // (op 3891) 
v_accvgpr_read v17, a11 // (op 3891) 
v_mul_f32 v18, v11, v14 // (op 3891) 
v_mul_f32 v19, v11, v15 // (op 3891) 
v_mul_f32 v20, v11, v16 // (op 3891) 
v_mul_f32 v21, v11, v17 // (op 3891) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3895) 
v_mul_f32 v15, v10, v7 // (op 3895) 
v_mul_f32 v16, v10, v8 // (op 3895) 
v_mul_f32 v17, v10, v9 // (op 3895) 

v_add_f32 v6, v14, v18 // (op 3892) 
v_add_f32 v7, v15, v19 // (op 3892) 
v_add_f32 v8, v16, v20 // (op 3892) 
v_add_f32 v9, v17, v21 // (op 3892) 




s_lshl_b32 s0, s4, 8 // (op 3893) 
v_add_u32 v0, v13, s0 // (op 3893) 


buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 3893) Store value




s_lshl_b32 s0, s2, 7 // (op 3906) 
s_mul_i32 s1, 3, s0 // (op 3906) 

v_add_u32 v0, v12, s1 // (op 3906) 


buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 3906) Load value

v_accvgpr_read v14, a12 // (op 3903) 
v_accvgpr_read v15, a13 // (op 3903) 
v_accvgpr_read v16, a14 // (op 3903) 
v_accvgpr_read v17, a15 // (op 3903) 
v_mul_f32 v18, v11, v14 // (op 3903) 
v_mul_f32 v19, v11, v15 // (op 3903) 
v_mul_f32 v20, v11, v16 // (op 3903) 
v_mul_f32 v21, v11, v17 // (op 3903) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3907) 
v_mul_f32 v15, v10, v7 // (op 3907) 
v_mul_f32 v16, v10, v8 // (op 3907) 
v_mul_f32 v17, v10, v9 // (op 3907) 

v_add_f32 v6, v14, v18 // (op 3904) 
v_add_f32 v7, v15, v19 // (op 3904) 
v_add_f32 v8, v16, v20 // (op 3904) 
v_add_f32 v9, v17, v21 // (op 3904) 




s_lshl_b32 s0, s4, 7 // (op 3905) 
s_mul_i32 s1, 3, s0 // (op 3905) 

v_add_u32 v0, v13, s1 // (op 3905) 


buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 3905) Store value




s_lshl_b32 s0, s2, 9 // (op 3918) 
v_add_u32 v0, v12, s0 // (op 3918) 


buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 3918) Load value

v_accvgpr_read v14, a16 // (op 3915) 
v_accvgpr_read v15, a17 // (op 3915) 
v_accvgpr_read v16, a18 // (op 3915) 
v_accvgpr_read v17, a19 // (op 3915) 
v_mul_f32 v18, v11, v14 // (op 3915) 
v_mul_f32 v19, v11, v15 // (op 3915) 
v_mul_f32 v20, v11, v16 // (op 3915) 
v_mul_f32 v21, v11, v17 // (op 3915) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3919) 
v_mul_f32 v15, v10, v7 // (op 3919) 
v_mul_f32 v16, v10, v8 // (op 3919) 
v_mul_f32 v17, v10, v9 // (op 3919) 

v_add_f32 v6, v14, v18 // (op 3916) 
v_add_f32 v7, v15, v19 // (op 3916) 
v_add_f32 v8, v16, v20 // (op 3916) 
v_add_f32 v9, v17, v21 // (op 3916) 




s_lshl_b32 s0, s4, 9 // (op 3917) 
v_add_u32 v0, v13, s0 // (op 3917) 


buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 3917) Store value




s_lshl_b32 s0, s2, 7 // (op 3930) 
s_mul_i32 s1, 5, s0 // (op 3930) 

v_add_u32 v0, v12, s1 // (op 3930) 


buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 3930) Load value

v_accvgpr_read v14, a20 // (op 3927) 
v_accvgpr_read v15, a21 // (op 3927) 
v_accvgpr_read v16, a22 // (op 3927) 
v_accvgpr_read v17, a23 // (op 3927) 
v_mul_f32 v18, v11, v14 // (op 3927) 
v_mul_f32 v19, v11, v15 // (op 3927) 
v_mul_f32 v20, v11, v16 // (op 3927) 
v_mul_f32 v21, v11, v17 // (op 3927) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3931) 
v_mul_f32 v15, v10, v7 // (op 3931) 
v_mul_f32 v16, v10, v8 // (op 3931) 
v_mul_f32 v17, v10, v9 // (op 3931) 

v_add_f32 v6, v14, v18 // (op 3928) 
v_add_f32 v7, v15, v19 // (op 3928) 
v_add_f32 v8, v16, v20 // (op 3928) 
v_add_f32 v9, v17, v21 // (op 3928) 




s_lshl_b32 s0, s4, 7 // (op 3929) 
s_mul_i32 s1, 5, s0 // (op 3929) 

v_add_u32 v0, v13, s1 // (op 3929) 


buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 3929) Store value




s_lshl_b32 s0, s2, 7 // (op 3942) 
s_mul_i32 s1, 6, s0 // (op 3942) 

v_add_u32 v0, v12, s1 // (op 3942) 


buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 3942) Load value

v_accvgpr_read v14, a24 // (op 3939) 
v_accvgpr_read v15, a25 // (op 3939) 
v_accvgpr_read v16, a26 // (op 3939) 
v_accvgpr_read v17, a27 // (op 3939) 
v_mul_f32 v18, v11, v14 // (op 3939) 
v_mul_f32 v19, v11, v15 // (op 3939) 
v_mul_f32 v20, v11, v16 // (op 3939) 
v_mul_f32 v21, v11, v17 // (op 3939) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3943) 
v_mul_f32 v15, v10, v7 // (op 3943) 
v_mul_f32 v16, v10, v8 // (op 3943) 
v_mul_f32 v17, v10, v9 // (op 3943) 

v_add_f32 v6, v14, v18 // (op 3940) 
v_add_f32 v7, v15, v19 // (op 3940) 
v_add_f32 v8, v16, v20 // (op 3940) 
v_add_f32 v9, v17, v21 // (op 3940) 




s_lshl_b32 s0, s4, 7 // (op 3941) 
s_mul_i32 s1, 6, s0 // (op 3941) 

v_add_u32 v0, v13, s1 // (op 3941) 


buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 3941) Store value




s_lshl_b32 s0, s2, 7 // (op 3954) 
s_mul_i32 s1, 7, s0 // (op 3954) 

v_add_u32 v0, v12, s1 // (op 3954) 


buffer_load_dwordx4 v[6:9], v0, s[52:55], 0 offen // (op 3954) Load value

v_accvgpr_read v14, a28 // (op 3951) 
v_accvgpr_read v15, a29 // (op 3951) 
v_accvgpr_read v16, a30 // (op 3951) 
v_accvgpr_read v17, a31 // (op 3951) 
v_mul_f32 v18, v11, v14 // (op 3951) 
v_mul_f32 v19, v11, v15 // (op 3951) 
v_mul_f32 v20, v11, v16 // (op 3951) 
v_mul_f32 v21, v11, v17 // (op 3951) 

s_waitcnt vmcnt(0)
v_mul_f32 v14, v10, v6 // (op 3955) 
v_mul_f32 v15, v10, v7 // (op 3955) 
v_mul_f32 v16, v10, v8 // (op 3955) 
v_mul_f32 v17, v10, v9 // (op 3955) 

v_add_f32 v6, v14, v18 // (op 3952) 
v_add_f32 v7, v15, v19 // (op 3952) 
v_add_f32 v8, v16, v20 // (op 3952) 
v_add_f32 v9, v17, v21 // (op 3952) 




s_lshl_b32 s0, s4, 7 // (op 3953) 
s_mul_i32 s1, 7, s0 // (op 3953) 

v_add_u32 v0, v13, s1 // (op 3953) 


buffer_store_dwordx4 v[6:9], v0, s[56:59], 0 offen // (op 3953) Store value


s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__ConditionalBottom_0_beta_133:
























s_endpgm  // End of RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_
.LRR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__end:
.size RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_, .LRR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM__end-RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 268
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 84480 // lds bytes
  .amdhsa_accum_offset 204
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
    DataType: B8
    DataTypeA: B8
    DataTypeB: B8
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
    macrotile: [64, 256, 128]
    threads: [256, 1, 1]
    grid: [TilesXY, One, Batch]
  MatrixInstruction: [16, 16, 128, 1]
  StaggerU: 0
  EnableMatrixInstruction: True
  MIWaveTile: [2, 8]
  WavefrontSize: 64
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_BF8_BF8_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_64x256x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 84480
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 74
    .vgpr_count: 202
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
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: BF8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF8, pointerType: PointerGlobal}
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: BF8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: BF8, pointerType: PointerGlobal}
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

