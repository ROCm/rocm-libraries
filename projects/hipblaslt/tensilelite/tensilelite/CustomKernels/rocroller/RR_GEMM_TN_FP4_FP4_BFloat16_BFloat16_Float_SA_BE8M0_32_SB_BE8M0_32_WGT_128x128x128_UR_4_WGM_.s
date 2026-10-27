// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM_
.p2align 8
.type RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM_,@function
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 10486) 
s_mul_hi_i32 s1, s16, s10 // (op 10625) 
v_lshrrev_b32 v0, 7, v1 // (op 10547) 
v_and_b32 v3, 15, v1 // (op 10547) 
v_lshl_add_u32 v4, v0, 6, v3 // (op 10547) 


v_and_b32 v0, 63, v1 // (op 10547) 
v_lshrrev_b32 v3, 4, v0 // (op 10547) 

v_lshlrev_b32 v0, 5, v3 // (op 10547) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 10547) 


v_lshlrev_b32 v0, 2, v3 // (op 10547) 

v_lshrrev_b32 v3, 3, v0 // (op 10547) 

v_lshrrev_b32 v0, 6, v1 // (op 10604) 
v_and_b32 v4, 1, v0 // (op 10604) 

v_and_b32 v0, 15, v1 // (op 10604) 
v_lshl_add_u32 v5, v4, 6, v0 // (op 10604) 


v_and_b32 v0, 63, v1 // (op 10604) 
v_lshrrev_b32 v4, 4, v0 // (op 10604) 

v_lshlrev_b32 v0, 5, v4 // (op 10604) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 10604) 


v_lshlrev_b32 v0, 2, v4 // (op 10604) 

v_lshrrev_b32 v4, 3, v0 // (op 10604) 

v_lshrrev_b32 v0, 7, v1 // (op 10671) 
v_and_b32 v5, 15, v1 // (op 10671) 
v_lshl_add_u32 v6, v0, 6, v5 // (op 10671) 


v_and_b32 v0, 63, v1 // (op 10671) 
v_lshrrev_b32 v5, 4, v0 // (op 10671) 

v_lshlrev_b32 v0, 5, v5 // (op 10671) 

v_lshl_add_u32 v5, v6, 7, v0 // (op 10671) 


v_lshlrev_b32 v0, 2, v5 // (op 10671) 

v_lshrrev_b32 v5, 3, v0 // (op 10671) 

v_lshrrev_b32 v0, 6, v1 // (op 10715) 
v_and_b32 v6, 1, v0 // (op 10715) 

v_and_b32 v0, 15, v1 // (op 10715) 
v_lshl_add_u32 v7, v6, 6, v0 // (op 10715) 


v_and_b32 v0, 63, v1 // (op 10715) 
v_lshrrev_b32 v6, 4, v0 // (op 10715) 

v_lshlrev_b32 v0, 5, v6 // (op 10715) 

v_lshl_add_u32 v6, v7, 7, v0 // (op 10715) 


v_lshlrev_b32 v0, 2, v6 // (op 10715) 

v_lshrrev_b32 v6, 3, v0 // (op 10715) 

v_lshrrev_b32 v0, 7, v1 // (op 10759) 
v_and_b32 v7, 15, v1 // (op 10759) 
v_lshl_add_u32 v8, v0, 6, v7 // (op 10759) 


v_and_b32 v0, 63, v1 // (op 10759) 
v_lshrrev_b32 v7, 4, v0 // (op 10759) 

v_lshlrev_b32 v0, 5, v7 // (op 10759) 

v_lshl_add_u32 v7, v8, 7, v0 // (op 10759) 


v_lshlrev_b32 v0, 2, v7 // (op 10759) 

v_lshrrev_b32 v7, 3, v0 // (op 10759) 

v_lshrrev_b32 v0, 6, v1 // (op 10803) 
v_and_b32 v8, 1, v0 // (op 10803) 

v_and_b32 v0, 15, v1 // (op 10803) 
v_lshl_add_u32 v9, v8, 6, v0 // (op 10803) 


v_and_b32 v0, 63, v1 // (op 10803) 
v_lshrrev_b32 v8, 4, v0 // (op 10803) 

v_lshlrev_b32 v0, 5, v8 // (op 10803) 

v_lshl_add_u32 v8, v9, 7, v0 // (op 10803) 


v_lshlrev_b32 v0, 2, v8 // (op 10803) 

v_lshrrev_b32 v8, 3, v0 // (op 10803) 

v_lshrrev_b32 v0, 7, v1 // (op 10847) 
v_and_b32 v9, 15, v1 // (op 10847) 
v_lshl_add_u32 v10, v0, 6, v9 // (op 10847) 


v_and_b32 v0, 63, v1 // (op 10847) 
v_lshrrev_b32 v9, 4, v0 // (op 10847) 

v_lshlrev_b32 v0, 5, v9 // (op 10847) 

v_lshl_add_u32 v9, v10, 7, v0 // (op 10847) 


v_lshlrev_b32 v0, 2, v9 // (op 10847) 

v_lshrrev_b32 v9, 3, v0 // (op 10847) 

v_lshrrev_b32 v0, 6, v1 // (op 10891) 
v_and_b32 v10, 1, v0 // (op 10891) 

v_and_b32 v0, 15, v1 // (op 10891) 
v_lshl_add_u32 v11, v10, 6, v0 // (op 10891) 


v_and_b32 v0, 63, v1 // (op 10891) 
v_lshrrev_b32 v10, 4, v0 // (op 10891) 

v_lshlrev_b32 v0, 5, v10 // (op 10891) 

v_lshl_add_u32 v10, v11, 7, v0 // (op 10891) 


v_lshlrev_b32 v0, 2, v10 // (op 10891) 

v_lshrrev_b32 v10, 3, v0 // (op 10891) 

v_lshrrev_b32 v0, 7, v1 // (op 10486) 
s_add_i32 s18, s0, s16 // (op 10486) 

v_and_b32 v11, 63, v1 // (op 10486) 
s_ashr_i32 s0, s18, 31 // (op 10486) 
v_lshrrev_b32 v12, 4, v11 // (op 10486) 
s_cmp_eq_i32 s10, 0 // (op 10486) 
s_mov_b32 s19, scc // (op 10486) 
v_lshlrev_b32 v13, 2, v12 // (op 10486) 

s_cmp_eq_i32 s19, 0 // (op 10486) 
s_cselect_b32 s49, 0, -1 // (op 10486) 

v_lshl_add_u32 v12, v0, 6, v13 // (op 10486) 


s_and_b32 s19, s13, 31 // (op 10486) 
v_lshrrev_b32 v0, 2, v11 // (op 10486) 

v_and_b32 v11, 3, v0 // (op 10486) 

v_lshrrev_b32 v0, 7, v1 // (op 10625) 
v_and_b32 v13, 63, v1 // (op 10625) 
s_lshl_b32 s50, 1, s19 // (op 10486) 
v_and_b32 v14, 3, v1 // (op 10486) 
s_add_i32 s51, s50, s49 // (op 10486) 


s_and_b32 s49, s0, s51 // (op 10486) 


s_add_i32 s0, s1, s16 // (op 10625) 

s_ashr_i32 s1, s0, 31 // (op 10625) 
v_lshl_add_u32 v15, v11, 2, v14 // (op 10486) 


s_add_i32 s50, s18, s49 // (op 10486) 


v_lshrrev_b32 v11, 6, v1 // (op 10486) 
s_ashr_i32 s18, s50, s19 // (op 10486) 


v_and_b32 v14, 1, v11 // (op 10486) 

s_lshl_b32 s19, s13, 24 // (op 10486) 
v_lshl_add_u32 v11, v14, 6, v15 // (op 10486) 


s_ashr_i32 s49, s19, 31 // (op 10486) 

s_xor_b32 s19, s18, s49 // (op 10486) 

s_sub_i32 s18, s19, s49 // (op 10486) 


s_mul_hi_i32 s19, s18, s14 // (op 10486) 
s_add_i32 s49, s19, s18 // (op 10486) 

s_ashr_i32 s19, s49, 31 // (op 10486) 
s_cmp_eq_i32 s14, 0 // (op 10486) 
s_mov_b32 s50, scc // (op 10486) 
s_cmp_eq_i32 s50, 0 // (op 10486) 
s_cselect_b32 s51, 0, -1 // (op 10486) 

s_and_b32 s50, s15, 31 // (op 10486) 
v_lshrrev_b32 v14, 4, v13 // (op 10625) 
s_cmp_eq_i32 s10, 0 // (op 10625) 
s_mov_b32 s52, scc // (op 10625) 
v_lshlrev_b32 v15, 2, v14 // (op 10625) 

s_cmp_eq_i32 s52, 0 // (op 10625) 
s_cselect_b32 s53, 0, -1 // (op 10625) 

s_lshl_b32 s52, 1, s50 // (op 10486) 
v_lshl_add_u32 v14, v0, 6, v15 // (op 10625) 


v_lshrrev_b32 v0, 2, v13 // (op 10625) 

s_add_i32 s54, s52, s51 // (op 10486) 


s_and_b32 s51, s19, s54 // (op 10486) 


s_add_i32 s19, s49, s51 // (op 10486) 


s_ashr_i32 s49, s19, s50 // (op 10486) 


s_lshl_b32 s19, s15, 24 // (op 10486) 
s_ashr_i32 s50, s19, 31 // (op 10486) 

s_xor_b32 s19, s49, s50 // (op 10486) 

s_sub_i32 s49, s19, s50 // (op 10486) 


s_mul_i32 s19, s18, s11 // (op 10486) 
s_sub_u32 s50, s16, s19 // (op 10486) 

s_mul_i32 s19, s49, s11 // (op 10486) 
s_add_u32 s19, s19, s50 // (op 10486) 

s_mul_hi_i32 s50, s16, s9 // (op 10486) 
s_add_i32 s51, s50, s16 // (op 10486) 

s_ashr_i32 s50, s51, 31 // (op 10486) 
s_cmp_eq_i32 s9, 0 // (op 10486) 
s_mov_b32 s52, scc // (op 10486) 
s_cmp_eq_i32 s52, 0 // (op 10486) 
s_cselect_b32 s54, 0, -1 // (op 10486) 

s_and_b32 s52, s13, 31 // (op 10625) 
v_and_b32 v13, 3, v0 // (op 10625) 

s_lshl_b32 s55, 1, s52 // (op 10625) 
v_and_b32 v0, 3, v1 // (op 10625) 
s_add_i32 s56, s55, s53 // (op 10625) 


s_and_b32 s53, s1, s56 // (op 10625) 


v_lshl_add_u32 v15, v13, 2, v0 // (op 10625) 


s_add_i32 s1, s0, s53 // (op 10625) 


v_lshrrev_b32 v0, 6, v1 // (op 10625) 
s_ashr_i32 s0, s1, s52 // (op 10625) 


v_and_b32 v13, 1, v0 // (op 10625) 

s_lshl_b32 s1, s13, 24 // (op 10625) 
v_lshl_add_u32 v0, v13, 6, v15 // (op 10625) 


s_ashr_i32 s52, s1, 31 // (op 10625) 

s_xor_b32 s1, s0, s52 // (op 10625) 

s_sub_i32 s0, s1, s52 // (op 10625) 


s_mul_hi_i32 s1, s0, s14 // (op 10625) 
s_add_i32 s52, s1, s0 // (op 10625) 

s_ashr_i32 s1, s52, 31 // (op 10625) 
s_cmp_eq_i32 s14, 0 // (op 10625) 
s_mov_b32 s53, scc // (op 10625) 
s_cmp_eq_i32 s53, 0 // (op 10625) 
s_cselect_b32 s55, 0, -1 // (op 10625) 

s_and_b32 s53, s15, 31 // (op 10625) 
s_lshl_b32 s56, 1, s53 // (op 10625) 
s_add_i32 s57, s56, s55 // (op 10625) 


s_and_b32 s55, s1, s57 // (op 10625) 


s_add_i32 s1, s52, s55 // (op 10625) 


s_ashr_i32 s52, s1, s53 // (op 10625) 


s_lshl_b32 s1, s15, 24 // (op 10625) 
s_ashr_i32 s53, s1, 31 // (op 10625) 

s_xor_b32 s1, s52, s53 // (op 10625) 

s_sub_i32 s52, s1, s53 // (op 10625) 


s_mul_i32 s1, s0, s11 // (op 10625) 
s_sub_u32 s53, s16, s1 // (op 10625) 

s_mul_i32 s1, s52, s11 // (op 10625) 
s_add_u32 s1, s1, s53 // (op 10625) 

s_mul_hi_i32 s53, s16, s9 // (op 10625) 
s_add_i32 s55, s53, s16 // (op 10625) 

s_ashr_i32 s53, s55, 31 // (op 10625) 
s_cmp_eq_i32 s9, 0 // (op 10625) 
s_mov_b32 s56, scc // (op 10625) 
s_cmp_eq_i32 s56, 0 // (op 10625) 
s_cselect_b32 s57, 0, -1 // (op 10625) 


s_waitcnt lgkmcnt(0)
s_and_b32 s56, s45, 31 // (op 10486) 
s_lshl_b32 s58, 1, s56 // (op 10486) 
s_add_i32 s59, s58, s54 // (op 10486) 


s_and_b32 s54, s50, s59 // (op 10486) 


s_add_i32 s50, s51, s54 // (op 10486) 


s_ashr_i32 s51, s50, s56 // (op 10486) 


s_lshl_b32 s50, s45, 24 // (op 10486) 
s_ashr_i32 s54, s50, 31 // (op 10486) 

s_xor_b32 s50, s51, s54 // (op 10486) 

s_sub_i32 s51, s50, s54 // (op 10486) 


s_and_b32 s50, s45, 31 // (op 10625) 
s_lshl_b32 s54, 1, s50 // (op 10625) 
s_add_i32 s56, s54, s57 // (op 10625) 


s_and_b32 s54, s53, s56 // (op 10625) 


s_add_i32 s53, s55, s54 // (op 10625) 


s_ashr_i32 s54, s53, s50 // (op 10625) 


s_lshl_b32 s50, s45, 24 // (op 10625) 
s_ashr_i32 s53, s50, 31 // (op 10625) 

s_xor_b32 s50, s54, s53 // (op 10625) 

s_sub_i32 s54, s50, s53 // (op 10625) 


s_add_u32 s56, s6, 128 // (op 10486) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 10486) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 10486) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 10486) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 7 // (op 10486) 

s_mul_i32 s50, s11, s56 // (op 10486) 
s_mul_i32 s53, s8, s50 // (op 10486) 

s_mul_i32 s50, s51, s53 // (op 10486) 

s_sub_u32 s53, s16, s50 // (op 10486) 

s_mov_b32 s50, s53 // (op 10486) 
s_mul_hi_i32 s55, s50, s12 // (op 10486) 
s_add_i32 s58, s55, s50 // (op 10486) 


s_ashr_i32 s50, s58, 31 // (op 10486) 
s_cmp_eq_i32 s12, 0 // (op 10486) 
s_mov_b32 s55, scc // (op 10486) 
s_cmp_eq_i32 s55, 0 // (op 10486) 
s_cselect_b32 s59, 0, -1 // (op 10486) 

s_and_b32 s55, s44, 31 // (op 10486) 
s_lshl_b32 s60, 1, s55 // (op 10486) 
s_add_i32 s61, s60, s59 // (op 10486) 


s_and_b32 s59, s50, s61 // (op 10486) 


s_add_i32 s50, s58, s59 // (op 10486) 


s_ashr_i32 s58, s50, s55 // (op 10486) 


s_lshl_b32 s50, s44, 24 // (op 10486) 
s_ashr_i32 s55, s50, 31 // (op 10486) 

s_xor_b32 s50, s58, s55 // (op 10486) 

s_sub_i32 s58, s50, s55 // (op 10486) 


s_mul_i32 s50, s46, s11 // (op 10486) 
s_add_u32 s60, s20, 128 // (op 10486) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 10486) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 10486) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 10486) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 7 // (op 10486) 

s_sub_i32 s55, s60, s50 // (op 10486) 

s_mul_i32 s59, s58, s55 // (op 10486) 

s_sub_u32 s55, s53, s59 // (op 10486) 


s_add_u32 s53, s55, s50 // (op 10486) 


s_cmp_eq_i32 s51, 0 // (op 10486) 
s_mov_b32 s50, scc // (op 10486) 

s_cmp_eq_i32 s50, 0 // (op 10486) 
s_cselect_b32 s51, s53, s19 // (op 10486) 


s_mul_i32 s19, s49, s56 // (op 10486) 


v_lshl_add_u32 v13, s51, 7, v12 // (op 10486) 


s_sub_i32 s49, s18, s19 // (op 10486) 


s_cmp_eq_i32 s50, 0 // (op 10486) 
s_cselect_b32 s51, s58, s49 // (op 10486) 



v_lshl_add_u32 v12, s51, 7, v11 // (op 10486) 


v_mul_lo_u32 v11, v12, s2 // (op 10486) 

v_add_lshl_u32 v12, v13, v11, 1 // (op 10486) 


s_add_u32 s18, s6, 128 // (op 10625) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 10625) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 10625) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 10625) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 7 // (op 10625) 

s_mul_i32 s53, s11, s18 // (op 10625) 
s_mul_i32 s49, s8, s53 // (op 10625) 

s_mul_i32 s53, s54, s49 // (op 10625) 

s_sub_u32 s49, s16, s53 // (op 10625) 

s_mov_b32 s53, s49 // (op 10625) 
s_mul_hi_i32 s50, s53, s12 // (op 10625) 
s_add_i32 s51, s50, s53 // (op 10625) 


s_ashr_i32 s50, s51, 31 // (op 10625) 
s_cmp_eq_i32 s12, 0 // (op 10625) 
s_mov_b32 s53, scc // (op 10625) 
s_cmp_eq_i32 s53, 0 // (op 10625) 
s_cselect_b32 s55, 0, -1 // (op 10625) 

s_and_b32 s53, s44, 31 // (op 10625) 
s_lshl_b32 s56, 1, s53 // (op 10625) 
s_add_i32 s57, s56, s55 // (op 10625) 


s_and_b32 s55, s50, s57 // (op 10625) 


s_add_i32 s50, s51, s55 // (op 10625) 


s_ashr_i32 s51, s50, s53 // (op 10625) 


s_lshl_b32 s50, s44, 24 // (op 10625) 
s_ashr_i32 s53, s50, 31 // (op 10625) 

s_xor_b32 s50, s51, s53 // (op 10625) 

s_sub_i32 s51, s50, s53 // (op 10625) 


s_mul_i32 s50, s46, s11 // (op 10625) 
s_add_u32 s56, s20, 128 // (op 10625) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 10625) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 10625) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 10625) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 7 // (op 10625) 

s_sub_i32 s53, s56, s50 // (op 10625) 

s_mul_i32 s55, s51, s53 // (op 10625) 

s_sub_u32 s53, s49, s55 // (op 10625) 


s_add_u32 s49, s53, s50 // (op 10625) 


s_cmp_eq_i32 s54, 0 // (op 10625) 
s_mov_b32 s50, scc // (op 10625) 

s_cmp_eq_i32 s50, 0 // (op 10625) 
s_cselect_b32 s53, s49, s1 // (op 10625) 


s_mul_i32 s1, s52, s18 // (op 10625) 


v_lshl_add_u32 v11, s53, 7, v14 // (op 10625) 


s_sub_i32 s49, s0, s1 // (op 10625) 


s_cmp_eq_i32 s50, 0 // (op 10625) 
s_cselect_b32 s0, s51, s49 // (op 10625) 



v_lshl_add_u32 v13, s0, 7, v0 // (op 10625) 


v_mul_lo_u32 v0, v13, s4 // (op 10625) 

v_add_lshl_u32 v13, v11, v0, 1 // (op 10625) 


s_mul_i32 s1, s2, s7 // (op 10488) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 10488) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 10488) least significant
s_mul_hi_u32 s19, s2, s6 // (op 10488) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 10488) most significant: sum
s_add_u32 s1, s1, s19 // (op 10488) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 10488) 

s_mul_i32 s1, s4, s7 // (op 10627) most significant: low of low * high
s_mul_i32 s49, s5, s6 // (op 10627) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 10627) least significant
s_mul_hi_u32 s50, s4, s6 // (op 10627) most significant: high of low * low
s_add_u32 s1, s1, s49 // (op 10627) most significant: sum
s_add_u32 s1, s1, s50 // (op 10627) most significant: sum


s_lshl_b64 s[50:51], s[0:1], 1 // (op 10627) 

s_mov_b32 s52, s22 // (op 10488) reg expression
s_mov_b32 s53, s23 // (op 10488) reg expression
s_mov_b32 s54, s18 // (op 10488) reg expression
s_mov_b32 s55, 131072 // (op 10488) call()

s_mov_b32 s56, s24 // (op 10627) reg expression

s_mov_b32 s57, s25 // (op 10627) reg expression
s_mov_b32 s58, s50 // (op 10627) reg expression
s_mov_b32 s59, 131072 // (op 10627) call()


v_mov_b32 v0, s47 // (op 2) Move value
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
v_accvgpr_write a16, 0.00000 // (op 1201) call()
v_accvgpr_write a17, 0.00000 // (op 1201) call()
v_accvgpr_write a18, 0.00000 // (op 1201) call()
v_accvgpr_write a19, 0.00000 // (op 1201) call()
v_accvgpr_write a20, 0.00000 // (op 1614) call()
v_accvgpr_write a21, 0.00000 // (op 1614) call()
v_accvgpr_write a22, 0.00000 // (op 1614) call()
v_accvgpr_write a23, 0.00000 // (op 1614) call()
v_accvgpr_write a24, 0.00000 // (op 1615) call()
v_accvgpr_write a25, 0.00000 // (op 1615) call()
v_accvgpr_write a26, 0.00000 // (op 1615) call()
v_accvgpr_write a27, 0.00000 // (op 1615) call()
v_accvgpr_write a28, 0.00000 // (op 1616) call()
v_accvgpr_write a29, 0.00000 // (op 1616) call()
v_accvgpr_write a30, 0.00000 // (op 1616) call()
v_accvgpr_write a31, 0.00000 // (op 1616) call()
v_accvgpr_write a32, 0.00000 // (op 2203) call()
v_accvgpr_write a33, 0.00000 // (op 2203) call()
v_accvgpr_write a34, 0.00000 // (op 2203) call()
v_accvgpr_write a35, 0.00000 // (op 2203) call()
v_accvgpr_write a36, 0.00000 // (op 2616) call()
v_accvgpr_write a37, 0.00000 // (op 2616) call()
v_accvgpr_write a38, 0.00000 // (op 2616) call()
v_accvgpr_write a39, 0.00000 // (op 2616) call()
v_accvgpr_write a40, 0.00000 // (op 2617) call()
v_accvgpr_write a41, 0.00000 // (op 2617) call()
v_accvgpr_write a42, 0.00000 // (op 2617) call()
v_accvgpr_write a43, 0.00000 // (op 2617) call()
v_accvgpr_write a44, 0.00000 // (op 2618) call()
v_accvgpr_write a45, 0.00000 // (op 2618) call()
v_accvgpr_write a46, 0.00000 // (op 2618) call()
v_accvgpr_write a47, 0.00000 // (op 2618) call()
v_accvgpr_write a48, 0.00000 // (op 3205) call()
v_accvgpr_write a49, 0.00000 // (op 3205) call()
v_accvgpr_write a50, 0.00000 // (op 3205) call()
v_accvgpr_write a51, 0.00000 // (op 3205) call()
v_accvgpr_write a52, 0.00000 // (op 3618) call()
v_accvgpr_write a53, 0.00000 // (op 3618) call()
v_accvgpr_write a54, 0.00000 // (op 3618) call()
v_accvgpr_write a55, 0.00000 // (op 3618) call()
v_accvgpr_write a56, 0.00000 // (op 3619) call()
v_accvgpr_write a57, 0.00000 // (op 3619) call()
v_accvgpr_write a58, 0.00000 // (op 3619) call()
v_accvgpr_write a59, 0.00000 // (op 3619) call()
v_accvgpr_write a60, 0.00000 // (op 3620) call()
v_accvgpr_write a61, 0.00000 // (op 3620) call()
v_accvgpr_write a62, 0.00000 // (op 3620) call()
v_accvgpr_write a63, 0.00000 // (op 3620) call()
s_mul_hi_i32 s0, s16, s10 // (op 10376) 
v_lshrrev_b32 v14, 2, v1 // (op 10376) 
s_add_i32 s1, s0, s16 // (op 10376) 

v_and_b32 v15, -4, v1 // (op 10376) 
s_ashr_i32 s0, s1, 31 // (op 10376) 
v_sub_u32 v16, v1, v15 // (op 10376) 

s_cmp_eq_i32 s10, 0 // (op 10376) 
s_mov_b32 s18, scc // (op 10376) 
v_lshlrev_b32 v15, 5, v16 // (op 10376) 

s_cmp_eq_i32 s18, 0 // (op 10376) 
s_cselect_b32 s19, 0, -1 // (op 10376) 

v_mov_b32 v17, 0 // (op 10376) convert
v_mov_b32 v16, v15 // (op 10376) convert

s_and_b32 s18, s13, 31 // (op 10376) 
s_lshl_b32 s22, 1, s18 // (op 10376) 
s_add_i32 s23, s22, s19 // (op 10376) 


s_and_b32 s19, s0, s23 // (op 10376) 


s_add_i32 s0, s1, s19 // (op 10376) 


s_ashr_i32 s1, s0, s18 // (op 10376) 


s_lshl_b32 s0, s13, 24 // (op 10376) 
s_ashr_i32 s18, s0, 31 // (op 10376) 

s_xor_b32 s0, s1, s18 // (op 10376) 

s_sub_i32 s1, s0, s18 // (op 10376) 


s_mul_hi_i32 s0, s1, s14 // (op 10376) 
s_add_i32 s18, s0, s1 // (op 10376) 

s_ashr_i32 s0, s18, 31 // (op 10376) 
s_cmp_eq_i32 s14, 0 // (op 10376) 
s_mov_b32 s19, scc // (op 10376) 
s_cmp_eq_i32 s19, 0 // (op 10376) 
s_cselect_b32 s22, 0, -1 // (op 10376) 

s_and_b32 s19, s15, 31 // (op 10376) 
s_lshl_b32 s23, 1, s19 // (op 10376) 
s_add_i32 s24, s23, s22 // (op 10376) 


s_and_b32 s25, s0, s24 // (op 10376) 


s_add_i32 s0, s18, s25 // (op 10376) 


s_ashr_i32 s18, s0, s19 // (op 10376) 


s_lshl_b32 s0, s15, 24 // (op 10376) 
s_ashr_i32 s19, s0, 31 // (op 10376) 

s_xor_b32 s0, s18, s19 // (op 10376) 

s_sub_i32 s18, s0, s19 // (op 10376) 


s_mul_i32 s0, s1, s11 // (op 10376) 

s_sub_u32 s1, s16, s0 // (op 10376) 

s_mul_i32 s0, s18, s11 // (op 10376) 
s_add_u32 s0, s0, s1 // (op 10376) 


s_mul_hi_i32 s1, s16, s9 // (op 10376) 
s_add_i32 s18, s1, s16 // (op 10376) 

s_ashr_i32 s1, s18, 31 // (op 10376) 
s_cmp_eq_i32 s9, 0 // (op 10376) 
s_mov_b32 s19, scc // (op 10376) 
s_cmp_eq_i32 s19, 0 // (op 10376) 
s_cselect_b32 s22, 0, -1 // (op 10376) 

s_and_b32 s19, s45, 31 // (op 10376) 
s_lshl_b32 s23, 1, s19 // (op 10376) 
s_add_i32 s24, s23, s22 // (op 10376) 


s_and_b32 s25, s1, s24 // (op 10376) 


s_add_i32 s1, s18, s25 // (op 10376) 


s_ashr_i32 s18, s1, s19 // (op 10376) 


s_lshl_b32 s1, s45, 24 // (op 10376) 
s_ashr_i32 s19, s1, 31 // (op 10376) 

s_xor_b32 s1, s18, s19 // (op 10376) 

s_sub_i32 s18, s1, s19 // (op 10376) 


s_add_u32 s22, s6, 128 // (op 10376) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 10376) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 10376) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 10376) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 10376) 

s_mul_i32 s1, s11, s22 // (op 10376) 

s_mul_i32 s19, s8, s1 // (op 10376) 

s_mul_i32 s1, s18, s19 // (op 10376) 

s_sub_u32 s19, s16, s1 // (op 10376) 

s_mov_b32 s1, s19 // (op 10376) 
s_mul_hi_i32 s22, s1, s12 // (op 10376) 
s_add_i32 s23, s22, s1 // (op 10376) 


s_ashr_i32 s1, s23, 31 // (op 10376) 
s_cmp_eq_i32 s12, 0 // (op 10376) 
s_mov_b32 s22, scc // (op 10376) 
s_cmp_eq_i32 s22, 0 // (op 10376) 
s_cselect_b32 s24, 0, -1 // (op 10376) 

s_and_b32 s22, s44, 31 // (op 10376) 
s_lshl_b32 s25, 1, s22 // (op 10376) 
s_add_i32 s47, s25, s24 // (op 10376) 


s_and_b32 s24, s1, s47 // (op 10376) 


s_add_i32 s1, s23, s24 // (op 10376) 


s_ashr_i32 s23, s1, s22 // (op 10376) 


s_lshl_b32 s1, s44, 24 // (op 10376) 
s_ashr_i32 s22, s1, 31 // (op 10376) 

s_xor_b32 s1, s23, s22 // (op 10376) 

s_sub_i32 s23, s1, s22 // (op 10376) 


s_mul_i32 s1, s46, s11 // (op 10376) 
s_add_u32 s24, s20, 128 // (op 10376) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 10376) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 10376) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 10376) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 7 // (op 10376) 

s_sub_i32 s22, s24, s1 // (op 10376) 

s_mul_i32 s24, s23, s22 // (op 10376) 


s_sub_u32 s25, s19, s24 // (op 10376) 


s_add_u32 s19, s25, s1 // (op 10376) 


s_cmp_eq_i32 s18, 0 // (op 10376) 
s_mov_b32 s1, scc // (op 10376) 

s_cmp_eq_i32 s1, 0 // (op 10376) 
s_cselect_b32 s18, s19, s0 // (op 10376) 



v_lshl_add_u32 v15, s18, 7, v14 // (op 10376) 


v_mul_lo_u32 v19, v15, s27 // (op 10376) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 10376) most significant: high of low * low
v_mul_lo_u32 v18, v15, s26 // (op 10376) least significant: low of low * low
v_add_u32 v19, v19, v14 // (op 10376) most significant: sum

v_add_co_u32 v18, s[0:1], v18, v16 // (op 10376) least significant half
v_addc_co_u32 v19, s[0:1], v19, v17, s[0:1] // (op 10376) most significant half



v_lshlrev_b64 v[14:15], 2, v[18:19] // (op 10376) 

v_lshrrev_b64 v[16:17], 3, v[14:15] // (op 10376) 

s_mul_i32 s1, s26, s21 // (op 10378) most significant: low of low * high
s_mul_i32 s18, s27, s20 // (op 10378) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 10378) least significant
s_mul_hi_u32 s19, s26, s20 // (op 10378) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 10378) most significant: sum
s_add_u32 s1, s1, s19 // (op 10378) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 10378) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 10378) 

s_mov_b32 s48, s28 // (op 10378) reg expression
s_mov_b32 s49, s29 // (op 10378) reg expression
s_mov_b32 s50, s0 // (op 10378) reg expression
s_mov_b32 s51, 131072 // (op 10378) call()

s_mul_hi_i32 s47, s16, s10 // (op 10395) 
v_lshrrev_b32 v14, 2, v1 // (op 10395) 
s_add_i32 s0, s47, s16 // (op 10395) 

v_and_b32 v15, -4, v1 // (op 10395) 
s_ashr_i32 s1, s0, 31 // (op 10395) 
v_sub_u32 v18, v1, v15 // (op 10395) 

s_cmp_eq_i32 s10, 0 // (op 10395) 
s_mov_b32 s47, scc // (op 10395) 
v_lshlrev_b32 v15, 5, v18 // (op 10395) 

s_cmp_eq_i32 s47, 0 // (op 10395) 
s_cselect_b32 s18, 0, -1 // (op 10395) 

v_mov_b32 v19, 0 // (op 10395) convert
v_mov_b32 v18, v15 // (op 10395) convert

s_and_b32 s19, s13, 31 // (op 10395) 
s_lshl_b32 s47, 1, s19 // (op 10395) 
s_add_i32 s22, s47, s18 // (op 10395) 


s_and_b32 s18, s1, s22 // (op 10395) 


s_add_i32 s1, s0, s18 // (op 10395) 


s_ashr_i32 s0, s1, s19 // (op 10395) 


s_lshl_b32 s1, s13, 24 // (op 10395) 
s_ashr_i32 s47, s1, 31 // (op 10395) 

s_xor_b32 s1, s0, s47 // (op 10395) 

s_sub_i32 s0, s1, s47 // (op 10395) 


s_mul_hi_i32 s1, s0, s14 // (op 10395) 
s_add_i32 s47, s1, s0 // (op 10395) 

s_ashr_i32 s1, s47, 31 // (op 10395) 
s_cmp_eq_i32 s14, 0 // (op 10395) 
s_mov_b32 s18, scc // (op 10395) 
s_cmp_eq_i32 s18, 0 // (op 10395) 
s_cselect_b32 s19, 0, -1 // (op 10395) 

s_and_b32 s18, s15, 31 // (op 10395) 
s_lshl_b32 s22, 1, s18 // (op 10395) 
s_add_i32 s23, s22, s19 // (op 10395) 


s_and_b32 s19, s1, s23 // (op 10395) 


s_add_i32 s1, s47, s19 // (op 10395) 


s_ashr_i32 s19, s1, s18 // (op 10395) 


s_lshl_b32 s1, s15, 24 // (op 10395) 
s_ashr_i32 s18, s1, 31 // (op 10395) 

s_xor_b32 s1, s19, s18 // (op 10395) 

s_sub_i32 s19, s1, s18 // (op 10395) 


s_mul_i32 s1, s0, s11 // (op 10395) 

s_sub_u32 s0, s16, s1 // (op 10395) 

s_mul_i32 s1, s19, s11 // (op 10395) 
s_add_u32 s1, s1, s0 // (op 10395) 


s_mul_hi_i32 s0, s16, s9 // (op 10395) 
s_add_i32 s47, s0, s16 // (op 10395) 

s_ashr_i32 s0, s47, 31 // (op 10395) 
s_cmp_eq_i32 s9, 0 // (op 10395) 
s_mov_b32 s18, scc // (op 10395) 
s_cmp_eq_i32 s18, 0 // (op 10395) 
s_cselect_b32 s19, 0, -1 // (op 10395) 

s_and_b32 s18, s45, 31 // (op 10395) 
s_lshl_b32 s22, 1, s18 // (op 10395) 
s_add_i32 s23, s22, s19 // (op 10395) 


s_and_b32 s19, s0, s23 // (op 10395) 


s_add_i32 s0, s47, s19 // (op 10395) 


s_ashr_i32 s19, s0, s18 // (op 10395) 


s_lshl_b32 s0, s45, 24 // (op 10395) 
s_ashr_i32 s18, s0, 31 // (op 10395) 

s_xor_b32 s0, s19, s18 // (op 10395) 

s_sub_i32 s19, s0, s18 // (op 10395) 


s_add_u32 s22, s6, 128 // (op 10395) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 10395) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 10395) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 10395) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 10395) 

s_mul_i32 s0, s11, s22 // (op 10395) 

s_mul_i32 s18, s8, s0 // (op 10395) 

s_mul_i32 s0, s19, s18 // (op 10395) 

s_sub_u32 s18, s16, s0 // (op 10395) 

s_mov_b32 s0, s18 // (op 10395) 
s_mul_hi_i32 s47, s0, s12 // (op 10395) 
s_add_i32 s22, s47, s0 // (op 10395) 


s_ashr_i32 s0, s22, 31 // (op 10395) 
s_cmp_eq_i32 s12, 0 // (op 10395) 
s_mov_b32 s47, scc // (op 10395) 
s_cmp_eq_i32 s47, 0 // (op 10395) 
s_cselect_b32 s23, 0, -1 // (op 10395) 

s_and_b32 s47, s44, 31 // (op 10395) 
s_lshl_b32 s24, 1, s47 // (op 10395) 
s_add_i32 s25, s24, s23 // (op 10395) 


s_and_b32 s23, s0, s25 // (op 10395) 


s_add_i32 s0, s22, s23 // (op 10395) 


s_ashr_i32 s22, s0, s47 // (op 10395) 


s_lshl_b32 s0, s44, 24 // (op 10395) 
s_ashr_i32 s47, s0, 31 // (op 10395) 

s_xor_b32 s0, s22, s47 // (op 10395) 

s_sub_i32 s22, s0, s47 // (op 10395) 


s_mul_i32 s0, s46, s11 // (op 10395) 
s_add_u32 s24, s20, 128 // (op 10395) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 10395) End of Int64 add, unlocking SCC
s_sub_u32 s60, s24, 1 // (op 10395) least significant half; sets scc
s_subb_u32 s61, s25, 0 // (op 10395) most significant half; uses scc

s_ashr_i64 s[24:25], s[60:61], 7 // (op 10395) 

s_sub_i32 s23, s24, s0 // (op 10395) 

s_mul_i32 s47, s22, s23 // (op 10395) 


s_sub_u32 s22, s18, s47 // (op 10395) 


s_add_u32 s18, s22, s0 // (op 10395) 


s_cmp_eq_i32 s19, 0 // (op 10395) 
s_mov_b32 s0, scc // (op 10395) 

s_cmp_eq_i32 s0, 0 // (op 10395) 
s_cselect_b32 s19, s18, s1 // (op 10395) 



v_lshl_add_u32 v15, s19, 7, v14 // (op 10395) 


v_mul_lo_u32 v21, v15, s27 // (op 10395) most significant: low of low * high
v_mul_hi_u32 v14, v15, s26 // (op 10395) most significant: high of low * low
v_mul_lo_u32 v20, v15, s26 // (op 10395) least significant: low of low * low
v_add_u32 v21, v21, v14 // (op 10395) most significant: sum

v_add_co_u32 v20, s[0:1], v20, v18 // (op 10395) least significant half
v_addc_co_u32 v21, s[0:1], v21, v19, s[0:1] // (op 10395) most significant half



v_lshlrev_b64 v[14:15], 2, v[20:21] // (op 10395) 

v_lshrrev_b64 v[18:19], 3, v[14:15] // (op 10395) 

s_mul_i32 s1, s26, s21 // (op 10397) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 10397) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 10397) least significant
s_mul_hi_u32 s18, s26, s20 // (op 10397) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 10397) most significant: sum
s_add_u32 s1, s1, s18 // (op 10397) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 10397) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 10397) 

s_mov_b32 s48, s28 // (op 10397) reg expression
s_mov_b32 s49, s29 // (op 10397) reg expression
s_mov_b32 s50, s0 // (op 10397) reg expression
s_mov_b32 s51, 131072 // (op 10397) call()

s_mul_hi_i32 s47, s16, s10 // (op 10440) 
v_lshrrev_b32 v14, 2, v1 // (op 10440) 
s_add_i32 s0, s47, s16 // (op 10440) 

v_and_b32 v15, -4, v1 // (op 10440) 
s_ashr_i32 s1, s0, 31 // (op 10440) 
v_sub_u32 v20, v1, v15 // (op 10440) 

s_cmp_eq_i32 s10, 0 // (op 10440) 
s_mov_b32 s47, scc // (op 10440) 
v_lshlrev_b32 v15, 5, v20 // (op 10440) 

s_cmp_eq_i32 s47, 0 // (op 10440) 
s_cselect_b32 s18, 0, -1 // (op 10440) 

v_mov_b32 v21, 0 // (op 10440) convert
v_mov_b32 v20, v15 // (op 10440) convert

s_and_b32 s19, s13, 31 // (op 10440) 
s_lshl_b32 s47, 1, s19 // (op 10440) 
s_add_i32 s22, s47, s18 // (op 10440) 


s_and_b32 s18, s1, s22 // (op 10440) 


s_add_i32 s1, s0, s18 // (op 10440) 


s_ashr_i32 s0, s1, s19 // (op 10440) 


s_lshl_b32 s1, s13, 24 // (op 10440) 
s_ashr_i32 s47, s1, 31 // (op 10440) 

s_xor_b32 s1, s0, s47 // (op 10440) 

s_sub_i32 s0, s1, s47 // (op 10440) 


s_mul_hi_i32 s1, s0, s14 // (op 10440) 
s_add_i32 s47, s1, s0 // (op 10440) 

s_ashr_i32 s1, s47, 31 // (op 10440) 
s_cmp_eq_i32 s14, 0 // (op 10440) 
s_mov_b32 s18, scc // (op 10440) 
s_cmp_eq_i32 s18, 0 // (op 10440) 
s_cselect_b32 s19, 0, -1 // (op 10440) 

s_and_b32 s18, s15, 31 // (op 10440) 
s_lshl_b32 s22, 1, s18 // (op 10440) 
s_add_i32 s23, s22, s19 // (op 10440) 


s_and_b32 s19, s1, s23 // (op 10440) 


s_add_i32 s1, s47, s19 // (op 10440) 


s_ashr_i32 s19, s1, s18 // (op 10440) 


s_lshl_b32 s1, s15, 24 // (op 10440) 
s_ashr_i32 s18, s1, 31 // (op 10440) 

s_xor_b32 s1, s19, s18 // (op 10440) 

s_sub_i32 s19, s1, s18 // (op 10440) 


s_add_u32 s22, s6, 128 // (op 10440) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 10440) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 10440) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 10440) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 10440) 

s_mul_i32 s1, s19, s22 // (op 10440) 

s_sub_i32 s47, s0, s1 // (op 10440) 


s_mul_hi_i32 s0, s16, s9 // (op 10440) 
s_add_i32 s1, s0, s16 // (op 10440) 

s_ashr_i32 s0, s1, 31 // (op 10440) 
s_cmp_eq_i32 s9, 0 // (op 10440) 
s_mov_b32 s18, scc // (op 10440) 
s_cmp_eq_i32 s18, 0 // (op 10440) 
s_cselect_b32 s19, 0, -1 // (op 10440) 

s_and_b32 s18, s45, 31 // (op 10440) 
s_lshl_b32 s24, 1, s18 // (op 10440) 
s_add_i32 s25, s24, s19 // (op 10440) 


s_and_b32 s19, s0, s25 // (op 10440) 


s_add_i32 s0, s1, s19 // (op 10440) 


s_ashr_i32 s1, s0, s18 // (op 10440) 


s_lshl_b32 s0, s45, 24 // (op 10440) 
s_ashr_i32 s18, s0, 31 // (op 10440) 

s_xor_b32 s0, s1, s18 // (op 10440) 

s_sub_i32 s1, s0, s18 // (op 10440) 


s_mul_i32 s0, s11, s22 // (op 10440) 

s_mul_i32 s18, s8, s0 // (op 10440) 

s_mul_i32 s0, s1, s18 // (op 10440) 

s_sub_u32 s18, s16, s0 // (op 10440) 

s_mul_hi_i32 s0, s18, s12 // (op 10440) 
s_add_i32 s19, s0, s18 // (op 10440) 


s_ashr_i32 s0, s19, 31 // (op 10440) 
s_cmp_eq_i32 s12, 0 // (op 10440) 
s_mov_b32 s18, scc // (op 10440) 
s_cmp_eq_i32 s18, 0 // (op 10440) 
s_cselect_b32 s22, 0, -1 // (op 10440) 

s_and_b32 s18, s44, 31 // (op 10440) 
s_lshl_b32 s23, 1, s18 // (op 10440) 
s_add_i32 s24, s23, s22 // (op 10440) 


s_and_b32 s25, s0, s24 // (op 10440) 


s_add_i32 s0, s19, s25 // (op 10440) 


s_ashr_i32 s19, s0, s18 // (op 10440) 


s_lshl_b32 s0, s44, 24 // (op 10440) 
s_ashr_i32 s18, s0, 31 // (op 10440) 

s_xor_b32 s0, s19, s18 // (op 10440) 

s_sub_i32 s19, s0, s18 // (op 10440) 


s_cmp_eq_i32 s1, 0 // (op 10440) 
s_mov_b32 s0, scc // (op 10440) 

s_cmp_eq_i32 s0, 0 // (op 10440) 
s_cselect_b32 s1, s19, s47 // (op 10440) 



v_lshl_add_u32 v15, s1, 7, v14 // (op 10440) 


v_mul_lo_u32 v23, v15, s31 // (op 10440) most significant: low of low * high
v_mul_hi_u32 v14, v15, s30 // (op 10440) most significant: high of low * low
v_mul_lo_u32 v22, v15, s30 // (op 10440) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 10440) most significant: sum


v_add_co_u32 v14, s[0:1], v20, v22 // (op 10440) least significant half
v_addc_co_u32 v15, s[0:1], v21, v23, s[0:1] // (op 10440) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 10440) 


v_lshrrev_b64 v[20:21], 3, v[14:15] // (op 10440) 

s_mul_i32 s1, s30, s7 // (op 10442) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 10442) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 10442) least significant
s_mul_hi_u32 s18, s30, s6 // (op 10442) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 10442) most significant: sum
s_add_u32 s1, s1, s18 // (op 10442) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 10442) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 10442) 

s_mov_b32 s60, s32 // (op 10442) reg expression
s_mov_b32 s61, s33 // (op 10442) reg expression
s_mov_b32 s62, s0 // (op 10442) reg expression
s_mov_b32 s63, 131072 // (op 10442) call()

s_mul_hi_i32 s47, s16, s10 // (op 10459) 
v_lshrrev_b32 v14, 2, v1 // (op 10459) 
s_add_i32 s0, s47, s16 // (op 10459) 

v_and_b32 v15, -4, v1 // (op 10459) 
s_ashr_i32 s1, s0, 31 // (op 10459) 
v_sub_u32 v22, v1, v15 // (op 10459) 

s_cmp_eq_i32 s10, 0 // (op 10459) 
s_mov_b32 s47, scc // (op 10459) 
v_lshlrev_b32 v15, 5, v22 // (op 10459) 

s_cmp_eq_i32 s47, 0 // (op 10459) 
s_cselect_b32 s18, 0, -1 // (op 10459) 

v_mov_b32 v23, 0 // (op 10459) convert
v_mov_b32 v22, v15 // (op 10459) convert

s_and_b32 s19, s13, 31 // (op 10459) 
s_lshl_b32 s47, 1, s19 // (op 10459) 
s_add_i32 s22, s47, s18 // (op 10459) 


s_and_b32 s18, s1, s22 // (op 10459) 


s_add_i32 s1, s0, s18 // (op 10459) 


s_ashr_i32 s0, s1, s19 // (op 10459) 


s_lshl_b32 s1, s13, 24 // (op 10459) 
s_ashr_i32 s47, s1, 31 // (op 10459) 

s_xor_b32 s1, s0, s47 // (op 10459) 

s_sub_i32 s0, s1, s47 // (op 10459) 


s_mul_hi_i32 s1, s0, s14 // (op 10459) 
s_add_i32 s47, s1, s0 // (op 10459) 

s_ashr_i32 s1, s47, 31 // (op 10459) 
s_cmp_eq_i32 s14, 0 // (op 10459) 
s_mov_b32 s18, scc // (op 10459) 
s_cmp_eq_i32 s18, 0 // (op 10459) 
s_cselect_b32 s19, 0, -1 // (op 10459) 

s_and_b32 s18, s15, 31 // (op 10459) 
s_lshl_b32 s22, 1, s18 // (op 10459) 
s_add_i32 s23, s22, s19 // (op 10459) 


s_and_b32 s19, s1, s23 // (op 10459) 


s_add_i32 s1, s47, s19 // (op 10459) 


s_ashr_i32 s19, s1, s18 // (op 10459) 


s_lshl_b32 s1, s15, 24 // (op 10459) 
s_ashr_i32 s18, s1, 31 // (op 10459) 

s_xor_b32 s1, s19, s18 // (op 10459) 

s_sub_i32 s19, s1, s18 // (op 10459) 


s_add_u32 s22, s6, 128 // (op 10459) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 10459) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 10459) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 10459) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 10459) 

s_mul_i32 s1, s19, s22 // (op 10459) 

s_sub_i32 s47, s0, s1 // (op 10459) 


s_mul_hi_i32 s0, s16, s9 // (op 10459) 
s_add_i32 s1, s0, s16 // (op 10459) 

s_ashr_i32 s0, s1, 31 // (op 10459) 
s_cmp_eq_i32 s9, 0 // (op 10459) 
s_mov_b32 s18, scc // (op 10459) 
s_cmp_eq_i32 s18, 0 // (op 10459) 
s_cselect_b32 s19, 0, -1 // (op 10459) 

s_and_b32 s18, s45, 31 // (op 10459) 
s_lshl_b32 s24, 1, s18 // (op 10459) 
s_add_i32 s25, s24, s19 // (op 10459) 


s_and_b32 s19, s0, s25 // (op 10459) 


s_add_i32 s0, s1, s19 // (op 10459) 


s_ashr_i32 s1, s0, s18 // (op 10459) 


s_lshl_b32 s0, s45, 24 // (op 10459) 
s_ashr_i32 s18, s0, 31 // (op 10459) 

s_xor_b32 s0, s1, s18 // (op 10459) 

s_sub_i32 s1, s0, s18 // (op 10459) 


s_mul_i32 s0, s11, s22 // (op 10459) 

s_mul_i32 s18, s8, s0 // (op 10459) 

s_mul_i32 s0, s1, s18 // (op 10459) 

s_sub_u32 s18, s16, s0 // (op 10459) 

s_mul_hi_i32 s0, s18, s12 // (op 10459) 
s_add_i32 s19, s0, s18 // (op 10459) 


s_ashr_i32 s0, s19, 31 // (op 10459) 
s_cmp_eq_i32 s12, 0 // (op 10459) 
s_mov_b32 s18, scc // (op 10459) 
s_cmp_eq_i32 s18, 0 // (op 10459) 
s_cselect_b32 s22, 0, -1 // (op 10459) 

s_and_b32 s18, s44, 31 // (op 10459) 
s_lshl_b32 s23, 1, s18 // (op 10459) 
s_add_i32 s24, s23, s22 // (op 10459) 


s_and_b32 s25, s0, s24 // (op 10459) 


s_add_i32 s0, s19, s25 // (op 10459) 


s_ashr_i32 s19, s0, s18 // (op 10459) 


s_lshl_b32 s0, s44, 24 // (op 10459) 
s_ashr_i32 s18, s0, 31 // (op 10459) 

s_xor_b32 s0, s19, s18 // (op 10459) 

s_sub_i32 s19, s0, s18 // (op 10459) 


s_cmp_eq_i32 s1, 0 // (op 10459) 
s_mov_b32 s0, scc // (op 10459) 

s_cmp_eq_i32 s0, 0 // (op 10459) 
s_cselect_b32 s1, s19, s47 // (op 10459) 



v_lshl_add_u32 v15, s1, 7, v14 // (op 10459) 


v_mul_lo_u32 v25, v15, s31 // (op 10459) most significant: low of low * high
v_mul_hi_u32 v14, v15, s30 // (op 10459) most significant: high of low * low
v_mul_lo_u32 v24, v15, s30 // (op 10459) least significant: low of low * low
v_add_u32 v25, v25, v14 // (op 10459) most significant: sum


v_add_co_u32 v14, s[0:1], v22, v24 // (op 10459) least significant half
v_addc_co_u32 v15, s[0:1], v23, v25, s[0:1] // (op 10459) most significant half

v_lshlrev_b64 v[14:15], 2, v[14:15] // (op 10459) 


v_lshrrev_b64 v[22:23], 3, v[14:15] // (op 10459) 

s_mul_i32 s1, s30, s7 // (op 10461) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 10461) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 10461) least significant
s_mul_hi_u32 s18, s30, s6 // (op 10461) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 10461) most significant: sum
s_add_u32 s1, s1, s18 // (op 10461) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 10461) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 10461) 

s_mov_b32 s60, s32 // (op 10461) reg expression
s_mov_b32 s61, s33 // (op 10461) reg expression
s_mov_b32 s62, s0 // (op 10461) reg expression
s_mov_b32 s63, 131072 // (op 10461) call()

s_mul_hi_i32 s47, s16, s10 // (op 10913) 
v_and_b32 v14, 63, v1 // (op 10913) 
s_add_i32 s0, s47, s16 // (op 10913) 

v_lshrrev_b32 v15, 4, v14 // (op 10913) 
s_ashr_i32 s1, s0, 31 // (op 10913) 
v_and_b32 v24, 3, v15 // (op 10913) 

s_cmp_eq_i32 s10, 0 // (op 10913) 
s_mov_b32 s47, scc // (op 10913) 
v_and_b32 v15, 15, v1 // (op 10913) 
s_cmp_eq_i32 s47, 0 // (op 10913) 
s_cselect_b32 s18, 0, -1 // (op 10913) 

v_lshl_add_u32 v25, v24, 4, v15 // (op 10913) 


s_and_b32 s19, s13, 31 // (op 10913) 
v_lshrrev_b32 v15, 7, v1 // (op 10913) 
s_lshl_b32 s47, 1, s19 // (op 10913) 
v_lshl_add_u32 v24, v15, 6, v25 // (op 10913) 


s_add_i32 s22, s47, s18 // (op 10913) 


v_lshrrev_b32 v15, 6, v14 // (op 10913) 

s_and_b32 s18, s1, s22 // (op 10913) 


v_lshlrev_b32 v14, 2, v15 // (op 10913) 

s_add_i32 s1, s0, s18 // (op 10913) 


s_ashr_i32 s0, s1, s19 // (op 10913) 


s_lshl_b32 s1, s13, 24 // (op 10913) 
s_ashr_i32 s47, s1, 31 // (op 10913) 

s_xor_b32 s1, s0, s47 // (op 10913) 

s_sub_i32 s0, s1, s47 // (op 10913) 


s_mul_hi_i32 s1, s0, s14 // (op 10913) 
s_add_i32 s47, s1, s0 // (op 10913) 

s_ashr_i32 s1, s47, 31 // (op 10913) 
s_cmp_eq_i32 s14, 0 // (op 10913) 
s_mov_b32 s18, scc // (op 10913) 
s_cmp_eq_i32 s18, 0 // (op 10913) 
s_cselect_b32 s19, 0, -1 // (op 10913) 

s_and_b32 s18, s15, 31 // (op 10913) 
s_lshl_b32 s22, 1, s18 // (op 10913) 
s_add_i32 s23, s22, s19 // (op 10913) 


s_and_b32 s19, s1, s23 // (op 10913) 


s_add_i32 s1, s47, s19 // (op 10913) 


s_ashr_i32 s19, s1, s18 // (op 10913) 


s_lshl_b32 s1, s15, 24 // (op 10913) 
s_ashr_i32 s18, s1, 31 // (op 10913) 

s_xor_b32 s1, s19, s18 // (op 10913) 

s_sub_i32 s19, s1, s18 // (op 10913) 


s_mul_i32 s1, s0, s11 // (op 10913) 

s_sub_u32 s0, s16, s1 // (op 10913) 

s_mul_i32 s1, s19, s11 // (op 10913) 
s_add_u32 s1, s1, s0 // (op 10913) 


s_mul_hi_i32 s0, s16, s9 // (op 10913) 
s_add_i32 s47, s0, s16 // (op 10913) 

s_ashr_i32 s0, s47, 31 // (op 10913) 
s_cmp_eq_i32 s9, 0 // (op 10913) 
s_mov_b32 s18, scc // (op 10913) 
s_cmp_eq_i32 s18, 0 // (op 10913) 
s_cselect_b32 s19, 0, -1 // (op 10913) 

s_and_b32 s18, s45, 31 // (op 10913) 
s_lshl_b32 s22, 1, s18 // (op 10913) 
s_add_i32 s23, s22, s19 // (op 10913) 


s_and_b32 s19, s0, s23 // (op 10913) 


s_add_i32 s0, s47, s19 // (op 10913) 


s_ashr_i32 s19, s0, s18 // (op 10913) 


s_lshl_b32 s0, s45, 24 // (op 10913) 
s_ashr_i32 s18, s0, 31 // (op 10913) 

s_xor_b32 s0, s19, s18 // (op 10913) 

s_sub_i32 s19, s0, s18 // (op 10913) 


s_add_u32 s22, s6, 128 // (op 10913) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 10913) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 10913) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 10913) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 10913) 

s_mul_i32 s0, s11, s22 // (op 10913) 

s_mul_i32 s18, s8, s0 // (op 10913) 

s_mul_i32 s0, s19, s18 // (op 10913) 

s_sub_u32 s18, s16, s0 // (op 10913) 

s_mov_b32 s0, s18 // (op 10913) 
s_mul_hi_i32 s47, s0, s12 // (op 10913) 
s_add_i32 s22, s47, s0 // (op 10913) 


s_ashr_i32 s0, s22, 31 // (op 10913) 
s_cmp_eq_i32 s12, 0 // (op 10913) 
s_mov_b32 s47, scc // (op 10913) 
s_cmp_eq_i32 s47, 0 // (op 10913) 
s_cselect_b32 s23, 0, -1 // (op 10913) 

s_and_b32 s47, s44, 31 // (op 10913) 
s_lshl_b32 s24, 1, s47 // (op 10913) 
s_add_i32 s25, s24, s23 // (op 10913) 


s_and_b32 s23, s0, s25 // (op 10913) 


s_add_i32 s0, s22, s23 // (op 10913) 


s_ashr_i32 s22, s0, s47 // (op 10913) 


s_lshl_b32 s0, s44, 24 // (op 10913) 
s_ashr_i32 s47, s0, 31 // (op 10913) 

s_xor_b32 s0, s22, s47 // (op 10913) 

s_sub_i32 s22, s0, s47 // (op 10913) 


s_mul_i32 s0, s46, s11 // (op 10913) 
s_add_u32 s24, s20, 128 // (op 10913) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 10913) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 10913) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 10913) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 7 // (op 10913) 

s_sub_i32 s23, s24, s0 // (op 10913) 

s_mul_i32 s47, s22, s23 // (op 10913) 


s_sub_u32 s22, s18, s47 // (op 10913) 


s_add_u32 s18, s22, s0 // (op 10913) 


s_cmp_eq_i32 s19, 0 // (op 10913) 
s_mov_b32 s0, scc // (op 10913) 

s_cmp_eq_i32 s0, 0 // (op 10913) 
s_cselect_b32 s19, s18, s1 // (op 10913) 



v_lshl_add_u32 v15, s19, 7, v24 // (op 10913) 


v_mul_lo_u32 v24, v15, s34 // (op 10913) 
v_add_u32 v24, v24, v14 // (op 10913) 


s_mul_i32 s1, s34, s21 // (op 10915) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 10915) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 10915) least significant
s_mul_hi_u32 s18, s34, s20 // (op 10915) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 10915) most significant: sum
s_add_u32 s1, s1, s18 // (op 10915) most significant: sum


s_mov_b32 s64, s36 // (op 10915) reg expression
s_mov_b32 s65, s37 // (op 10915) reg expression
s_mov_b32 s66, s0 // (op 10915) reg expression
s_mov_b32 s67, 131072 // (op 10915) call()

s_mul_hi_i32 s47, s16, s10 // (op 10934) 
v_and_b32 v14, 63, v1 // (op 10934) 
s_add_i32 s0, s47, s16 // (op 10934) 

v_lshrrev_b32 v15, 4, v14 // (op 10934) 
s_ashr_i32 s1, s0, 31 // (op 10934) 
v_and_b32 v25, 3, v15 // (op 10934) 

s_cmp_eq_i32 s10, 0 // (op 10934) 
s_mov_b32 s47, scc // (op 10934) 
v_and_b32 v15, 15, v1 // (op 10934) 
s_cmp_eq_i32 s47, 0 // (op 10934) 
s_cselect_b32 s18, 0, -1 // (op 10934) 

v_lshl_add_u32 v26, v25, 4, v15 // (op 10934) 


s_and_b32 s19, s13, 31 // (op 10934) 
v_lshrrev_b32 v15, 7, v1 // (op 10934) 
s_lshl_b32 s47, 1, s19 // (op 10934) 
v_lshl_add_u32 v25, v15, 6, v26 // (op 10934) 


s_add_i32 s22, s47, s18 // (op 10934) 


v_lshrrev_b32 v15, 6, v14 // (op 10934) 

s_and_b32 s18, s1, s22 // (op 10934) 


v_lshlrev_b32 v14, 2, v15 // (op 10934) 

s_add_i32 s1, s0, s18 // (op 10934) 


s_ashr_i32 s0, s1, s19 // (op 10934) 


s_lshl_b32 s1, s13, 24 // (op 10934) 
s_ashr_i32 s47, s1, 31 // (op 10934) 

s_xor_b32 s1, s0, s47 // (op 10934) 

s_sub_i32 s0, s1, s47 // (op 10934) 


s_mul_hi_i32 s1, s0, s14 // (op 10934) 
s_add_i32 s47, s1, s0 // (op 10934) 

s_ashr_i32 s1, s47, 31 // (op 10934) 
s_cmp_eq_i32 s14, 0 // (op 10934) 
s_mov_b32 s18, scc // (op 10934) 
s_cmp_eq_i32 s18, 0 // (op 10934) 
s_cselect_b32 s19, 0, -1 // (op 10934) 

s_and_b32 s18, s15, 31 // (op 10934) 
s_lshl_b32 s22, 1, s18 // (op 10934) 
s_add_i32 s23, s22, s19 // (op 10934) 


s_and_b32 s19, s1, s23 // (op 10934) 


s_add_i32 s1, s47, s19 // (op 10934) 


s_ashr_i32 s19, s1, s18 // (op 10934) 


s_lshl_b32 s1, s15, 24 // (op 10934) 
s_ashr_i32 s18, s1, 31 // (op 10934) 

s_xor_b32 s1, s19, s18 // (op 10934) 

s_sub_i32 s19, s1, s18 // (op 10934) 


s_mul_i32 s1, s0, s11 // (op 10934) 

s_sub_u32 s0, s16, s1 // (op 10934) 

s_mul_i32 s1, s19, s11 // (op 10934) 
s_add_u32 s1, s1, s0 // (op 10934) 


s_mul_hi_i32 s0, s16, s9 // (op 10934) 
s_add_i32 s47, s0, s16 // (op 10934) 

s_ashr_i32 s0, s47, 31 // (op 10934) 
s_cmp_eq_i32 s9, 0 // (op 10934) 
s_mov_b32 s18, scc // (op 10934) 
s_cmp_eq_i32 s18, 0 // (op 10934) 
s_cselect_b32 s19, 0, -1 // (op 10934) 

s_and_b32 s18, s45, 31 // (op 10934) 
s_lshl_b32 s22, 1, s18 // (op 10934) 
s_add_i32 s23, s22, s19 // (op 10934) 


s_and_b32 s19, s0, s23 // (op 10934) 


s_add_i32 s0, s47, s19 // (op 10934) 


s_ashr_i32 s19, s0, s18 // (op 10934) 


s_lshl_b32 s0, s45, 24 // (op 10934) 
s_ashr_i32 s18, s0, 31 // (op 10934) 

s_xor_b32 s0, s19, s18 // (op 10934) 

s_sub_i32 s19, s0, s18 // (op 10934) 


s_add_u32 s22, s6, 128 // (op 10934) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 10934) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 10934) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 10934) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 10934) 

s_mul_i32 s0, s11, s22 // (op 10934) 

s_mul_i32 s18, s8, s0 // (op 10934) 

s_mul_i32 s0, s19, s18 // (op 10934) 

s_sub_u32 s18, s16, s0 // (op 10934) 

s_mov_b32 s0, s18 // (op 10934) 
s_mul_hi_i32 s47, s0, s12 // (op 10934) 
s_add_i32 s22, s47, s0 // (op 10934) 


s_ashr_i32 s0, s22, 31 // (op 10934) 
s_cmp_eq_i32 s12, 0 // (op 10934) 
s_mov_b32 s47, scc // (op 10934) 
s_cmp_eq_i32 s47, 0 // (op 10934) 
s_cselect_b32 s23, 0, -1 // (op 10934) 

s_and_b32 s47, s44, 31 // (op 10934) 
s_lshl_b32 s24, 1, s47 // (op 10934) 
s_add_i32 s25, s24, s23 // (op 10934) 


s_and_b32 s23, s0, s25 // (op 10934) 


s_add_i32 s0, s22, s23 // (op 10934) 


s_ashr_i32 s22, s0, s47 // (op 10934) 


s_lshl_b32 s0, s44, 24 // (op 10934) 
s_ashr_i32 s47, s0, 31 // (op 10934) 

s_xor_b32 s0, s22, s47 // (op 10934) 

s_sub_i32 s22, s0, s47 // (op 10934) 


s_mul_i32 s0, s46, s11 // (op 10934) 
s_add_u32 s24, s20, 128 // (op 10934) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 10934) End of Int64 add, unlocking SCC
s_sub_u32 s68, s24, 1 // (op 10934) least significant half; sets scc
s_subb_u32 s69, s25, 0 // (op 10934) most significant half; uses scc

s_ashr_i64 s[24:25], s[68:69], 7 // (op 10934) 

s_sub_i32 s23, s24, s0 // (op 10934) 

s_mul_i32 s47, s22, s23 // (op 10934) 


s_sub_u32 s22, s18, s47 // (op 10934) 


s_add_u32 s18, s22, s0 // (op 10934) 


s_cmp_eq_i32 s19, 0 // (op 10934) 
s_mov_b32 s0, scc // (op 10934) 

s_cmp_eq_i32 s0, 0 // (op 10934) 
s_cselect_b32 s19, s18, s1 // (op 10934) 



v_lshl_add_u32 v15, s19, 7, v25 // (op 10934) 


v_mul_lo_u32 v25, v15, s34 // (op 10934) 
v_add_u32 v25, v25, v14 // (op 10934) 


s_mul_i32 s1, s34, s21 // (op 10936) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 10936) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 10936) least significant
s_mul_hi_u32 s18, s34, s20 // (op 10936) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 10936) most significant: sum
s_add_u32 s1, s1, s18 // (op 10936) most significant: sum


s_mov_b32 s64, s36 // (op 10936) reg expression
s_mov_b32 s65, s37 // (op 10936) reg expression
s_mov_b32 s66, s0 // (op 10936) reg expression
s_mov_b32 s67, 131072 // (op 10936) call()

s_mul_hi_i32 s47, s16, s10 // (op 10965) 
v_and_b32 v14, 63, v1 // (op 10965) 
s_add_i32 s0, s47, s16 // (op 10965) 

v_lshrrev_b32 v15, 4, v14 // (op 10965) 
s_ashr_i32 s1, s0, 31 // (op 10965) 
v_and_b32 v26, 3, v15 // (op 10965) 

s_cmp_eq_i32 s10, 0 // (op 10965) 
s_mov_b32 s47, scc // (op 10965) 
v_and_b32 v15, 15, v1 // (op 10965) 
s_cmp_eq_i32 s47, 0 // (op 10965) 
s_cselect_b32 s18, 0, -1 // (op 10965) 

v_lshl_add_u32 v27, v26, 4, v15 // (op 10965) 


s_and_b32 s19, s13, 31 // (op 10965) 
v_lshrrev_b32 v15, 6, v1 // (op 10965) 
s_lshl_b32 s47, 1, s19 // (op 10965) 
v_and_b32 v26, 1, v15 // (op 10965) 

s_add_i32 s22, s47, s18 // (op 10965) 


v_lshl_add_u32 v15, v26, 6, v27 // (op 10965) 


s_and_b32 s18, s1, s22 // (op 10965) 


v_lshrrev_b32 v26, 6, v14 // (op 10965) 

s_add_i32 s1, s0, s18 // (op 10965) 


s_ashr_i32 s0, s1, s19 // (op 10965) 


s_lshl_b32 s1, s13, 24 // (op 10965) 
s_ashr_i32 s47, s1, 31 // (op 10965) 

s_xor_b32 s1, s0, s47 // (op 10965) 

s_sub_i32 s0, s1, s47 // (op 10965) 


s_mul_hi_i32 s1, s0, s14 // (op 10965) 
s_add_i32 s47, s1, s0 // (op 10965) 

s_ashr_i32 s1, s47, 31 // (op 10965) 
s_cmp_eq_i32 s14, 0 // (op 10965) 
s_mov_b32 s18, scc // (op 10965) 
s_cmp_eq_i32 s18, 0 // (op 10965) 
s_cselect_b32 s19, 0, -1 // (op 10965) 

s_and_b32 s18, s15, 31 // (op 10965) 
s_lshl_b32 s22, 1, s18 // (op 10965) 
s_add_i32 s23, s22, s19 // (op 10965) 


s_and_b32 s19, s1, s23 // (op 10965) 


s_add_i32 s1, s47, s19 // (op 10965) 


s_ashr_i32 s19, s1, s18 // (op 10965) 


s_lshl_b32 s1, s15, 24 // (op 10965) 
s_ashr_i32 s18, s1, 31 // (op 10965) 

s_xor_b32 s1, s19, s18 // (op 10965) 

s_sub_i32 s19, s1, s18 // (op 10965) 


s_add_u32 s22, s6, 128 // (op 10965) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 10965) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 10965) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 10965) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 10965) 

s_mul_i32 s1, s19, s22 // (op 10965) 

s_sub_i32 s47, s0, s1 // (op 10965) 


s_mul_hi_i32 s0, s16, s9 // (op 10965) 
s_add_i32 s1, s0, s16 // (op 10965) 

s_ashr_i32 s0, s1, 31 // (op 10965) 
s_cmp_eq_i32 s9, 0 // (op 10965) 
s_mov_b32 s18, scc // (op 10965) 
s_cmp_eq_i32 s18, 0 // (op 10965) 
s_cselect_b32 s19, 0, -1 // (op 10965) 

s_and_b32 s18, s45, 31 // (op 10965) 
s_lshl_b32 s24, 1, s18 // (op 10965) 
s_add_i32 s25, s24, s19 // (op 10965) 


s_and_b32 s19, s0, s25 // (op 10965) 


s_add_i32 s0, s1, s19 // (op 10965) 


s_ashr_i32 s1, s0, s18 // (op 10965) 


s_lshl_b32 s0, s45, 24 // (op 10965) 
s_ashr_i32 s18, s0, 31 // (op 10965) 

s_xor_b32 s0, s1, s18 // (op 10965) 

s_sub_i32 s1, s0, s18 // (op 10965) 


s_mul_i32 s0, s11, s22 // (op 10965) 

s_mul_i32 s18, s8, s0 // (op 10965) 

s_mul_i32 s0, s1, s18 // (op 10965) 

s_sub_u32 s18, s16, s0 // (op 10965) 

s_mul_hi_i32 s0, s18, s12 // (op 10965) 
s_add_i32 s19, s0, s18 // (op 10965) 


s_ashr_i32 s0, s19, 31 // (op 10965) 
s_cmp_eq_i32 s12, 0 // (op 10965) 
s_mov_b32 s18, scc // (op 10965) 
s_cmp_eq_i32 s18, 0 // (op 10965) 
s_cselect_b32 s22, 0, -1 // (op 10965) 

s_and_b32 s18, s44, 31 // (op 10965) 
s_lshl_b32 s23, 1, s18 // (op 10965) 
s_add_i32 s24, s23, s22 // (op 10965) 


s_and_b32 s25, s0, s24 // (op 10965) 


s_add_i32 s0, s19, s25 // (op 10965) 


s_ashr_i32 s19, s0, s18 // (op 10965) 


s_lshl_b32 s0, s44, 24 // (op 10965) 
s_ashr_i32 s18, s0, 31 // (op 10965) 

s_xor_b32 s0, s19, s18 // (op 10965) 

s_sub_i32 s19, s0, s18 // (op 10965) 


s_cmp_eq_i32 s1, 0 // (op 10965) 
s_mov_b32 s0, scc // (op 10965) 

s_cmp_eq_i32 s0, 0 // (op 10965) 
s_cselect_b32 s1, s19, s47 // (op 10965) 



v_lshl_add_u32 v14, s1, 7, v15 // (op 10965) 


v_mul_lo_u32 v15, v14, s38 // (op 10965) 

v_lshl_add_u32 v14, v26, 2, v15 // (op 10965) 


s_mul_i32 s1, s38, s7 // (op 10967) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 10967) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 10967) least significant
s_mul_hi_u32 s18, s38, s6 // (op 10967) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 10967) most significant: sum
s_add_u32 s1, s1, s18 // (op 10967) most significant: sum


s_mov_b32 s68, s40 // (op 10967) reg expression
s_mov_b32 s69, s41 // (op 10967) reg expression
s_mov_b32 s70, s0 // (op 10967) reg expression
s_mov_b32 s71, 131072 // (op 10967) call()

s_mul_hi_i32 s47, s16, s10 // (op 10986) 
v_and_b32 v15, 63, v1 // (op 10986) 
s_add_i32 s0, s47, s16 // (op 10986) 

v_lshrrev_b32 v26, 4, v15 // (op 10986) 
s_ashr_i32 s1, s0, 31 // (op 10986) 
v_and_b32 v27, 3, v26 // (op 10986) 

s_cmp_eq_i32 s10, 0 // (op 10986) 
s_mov_b32 s47, scc // (op 10986) 
v_and_b32 v26, 15, v1 // (op 10986) 
s_cmp_eq_i32 s47, 0 // (op 10986) 
s_cselect_b32 s18, 0, -1 // (op 10986) 

v_lshl_add_u32 v28, v27, 4, v26 // (op 10986) 


s_and_b32 s19, s13, 31 // (op 10986) 
v_lshrrev_b32 v26, 6, v1 // (op 10986) 
s_lshl_b32 s47, 1, s19 // (op 10986) 
v_and_b32 v27, 1, v26 // (op 10986) 

s_add_i32 s22, s47, s18 // (op 10986) 


v_lshl_add_u32 v26, v27, 6, v28 // (op 10986) 


s_and_b32 s18, s1, s22 // (op 10986) 


v_lshrrev_b32 v27, 6, v15 // (op 10986) 

s_add_i32 s1, s0, s18 // (op 10986) 


s_ashr_i32 s0, s1, s19 // (op 10986) 


s_lshl_b32 s1, s13, 24 // (op 10986) 
s_ashr_i32 s47, s1, 31 // (op 10986) 

s_xor_b32 s1, s0, s47 // (op 10986) 

s_sub_i32 s0, s1, s47 // (op 10986) 


s_mul_hi_i32 s1, s0, s14 // (op 10986) 
s_add_i32 s47, s1, s0 // (op 10986) 

s_ashr_i32 s1, s47, 31 // (op 10986) 
s_cmp_eq_i32 s14, 0 // (op 10986) 
s_mov_b32 s18, scc // (op 10986) 
s_cmp_eq_i32 s18, 0 // (op 10986) 
s_cselect_b32 s19, 0, -1 // (op 10986) 

s_and_b32 s18, s15, 31 // (op 10986) 
s_lshl_b32 s22, 1, s18 // (op 10986) 
s_add_i32 s23, s22, s19 // (op 10986) 


s_and_b32 s19, s1, s23 // (op 10986) 


s_add_i32 s1, s47, s19 // (op 10986) 


s_ashr_i32 s19, s1, s18 // (op 10986) 


s_lshl_b32 s1, s15, 24 // (op 10986) 
s_ashr_i32 s18, s1, 31 // (op 10986) 

s_xor_b32 s1, s19, s18 // (op 10986) 

s_sub_i32 s19, s1, s18 // (op 10986) 


s_add_u32 s22, s6, 128 // (op 10986) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 10986) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 10986) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 10986) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 10986) 

s_mul_i32 s1, s19, s22 // (op 10986) 

s_sub_i32 s47, s0, s1 // (op 10986) 


s_mul_hi_i32 s0, s16, s9 // (op 10986) 
s_add_i32 s1, s0, s16 // (op 10986) 

s_ashr_i32 s0, s1, 31 // (op 10986) 
s_cmp_eq_i32 s9, 0 // (op 10986) 
s_mov_b32 s18, scc // (op 10986) 
s_cmp_eq_i32 s18, 0 // (op 10986) 
s_cselect_b32 s19, 0, -1 // (op 10986) 

s_and_b32 s18, s45, 31 // (op 10986) 
s_lshl_b32 s24, 1, s18 // (op 10986) 
s_add_i32 s25, s24, s19 // (op 10986) 


s_and_b32 s19, s0, s25 // (op 10986) 


s_add_i32 s0, s1, s19 // (op 10986) 


s_ashr_i32 s1, s0, s18 // (op 10986) 


s_lshl_b32 s0, s45, 24 // (op 10986) 
s_ashr_i32 s18, s0, 31 // (op 10986) 

s_xor_b32 s0, s1, s18 // (op 10986) 

s_sub_i32 s1, s0, s18 // (op 10986) 


s_mul_i32 s0, s11, s22 // (op 10986) 

s_mul_i32 s18, s8, s0 // (op 10986) 

s_mul_i32 s0, s1, s18 // (op 10986) 

s_sub_u32 s18, s16, s0 // (op 10986) 

s_mul_hi_i32 s0, s18, s12 // (op 10986) 
s_add_i32 s19, s0, s18 // (op 10986) 


s_ashr_i32 s0, s19, 31 // (op 10986) 
s_cmp_eq_i32 s12, 0 // (op 10986) 
s_mov_b32 s18, scc // (op 10986) 
s_cmp_eq_i32 s18, 0 // (op 10986) 
s_cselect_b32 s22, 0, -1 // (op 10986) 

s_and_b32 s18, s44, 31 // (op 10986) 
s_lshl_b32 s23, 1, s18 // (op 10986) 
s_add_i32 s24, s23, s22 // (op 10986) 


s_and_b32 s25, s0, s24 // (op 10986) 


s_add_i32 s0, s19, s25 // (op 10986) 


s_ashr_i32 s19, s0, s18 // (op 10986) 


s_lshl_b32 s0, s44, 24 // (op 10986) 
s_ashr_i32 s18, s0, 31 // (op 10986) 

s_xor_b32 s0, s19, s18 // (op 10986) 

s_sub_i32 s19, s0, s18 // (op 10986) 


s_cmp_eq_i32 s1, 0 // (op 10986) 
s_mov_b32 s0, scc // (op 10986) 

s_cmp_eq_i32 s0, 0 // (op 10986) 
s_cselect_b32 s1, s19, s47 // (op 10986) 



v_lshl_add_u32 v15, s1, 7, v26 // (op 10986) 


v_mul_lo_u32 v26, v15, s38 // (op 10986) 

v_lshl_add_u32 v15, v27, 2, v26 // (op 10986) 


s_mul_i32 s1, s38, s7 // (op 10988) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 10988) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 10988) least significant
s_mul_hi_u32 s18, s38, s6 // (op 10988) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 10988) most significant: sum
s_add_u32 s1, s1, s18 // (op 10988) most significant: sum


s_mov_b32 s68, s40 // (op 10988) reg expression
s_mov_b32 s69, s41 // (op 10988) reg expression
s_mov_b32 s70, s0 // (op 10988) reg expression
s_mov_b32 s71, 131072 // (op 10988) call()

v_and_b32 v26, -4, v1 // (op 10587) 
v_sub_u32 v27, v1, v26 // (op 10587) 

v_lshlrev_b32 v26, 5, v27 // (op 10587) 

v_lshrrev_b32 v27, 2, v1 // (op 10587) 
v_lshl_add_u32 v28, v27, 7, v26 // (op 10587) 


v_lshlrev_b32 v26, 2, v28 // (op 10587) 

v_lshrrev_b32 v27, 3, v26 // (op 10587) 

s_nop 0
v_readfirstlane_b32 s47, v27 // (op 10587) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 6 // (op 10302) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 10302) 

v_lshrrev_b64 v[26:27], 3, s[18:19] // (op 10302) 


v_mov_b64 v[28:29], v[20:21] // (op 10302) 
s_mov_b32 m0, s47 // (op 10302) reg expression
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 10302) Load value direct to lds
v_add_co_u32 v28, s[0:1], v28, v26 // (op 10302) least significant half
v_addc_co_u32 v29, s[0:1], 0, v29, s[0:1] // (op 10302) most significant half

s_add_u32 m0, m0, 4096 // (op 10302) 
buffer_load_dwordx4 v28, s[60:63], 0 offen offset: 0 lds // (op 10302) Load value direct to lds



v_and_b32 v26, -4, v1 // (op 10530) 
v_sub_u32 v27, v1, v26 // (op 10530) 

v_lshlrev_b32 v26, 5, v27 // (op 10530) 

v_lshrrev_b32 v27, 2, v1 // (op 10530) 
v_lshl_add_u32 v28, v27, 7, v26 // (op 10530) 


v_lshlrev_b32 v26, 2, v28 // (op 10530) 

v_lshrrev_b32 v27, 3, v26 // (op 10530) 

s_nop 0
v_readfirstlane_b32 s47, v27 // (op 10530) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 6 // (op 10307) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 10307) 

v_lshrrev_b64 v[26:27], 3, s[18:19] // (op 10307) 


v_mov_b64 v[28:29], v[16:17] // (op 10307) 
s_add_u32 s0, 8192, s47 // (op 10307) 
s_mov_b32 m0, s0 // (op 10307) reg expression
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 10307) Load value direct to lds
v_add_co_u32 v28, s[18:19], v28, v26 // (op 10307) least significant half
v_addc_co_u32 v29, s[18:19], 0, v29, s[18:19] // (op 10307) most significant half

s_add_u32 m0, m0, 4096 // (op 10307) 
buffer_load_dwordx4 v28, s[48:51], 0 offen offset: 0 lds // (op 10307) Load value direct to lds






v_mov_b32 v26, v14 // (op 76) reg expression

buffer_load_dwordx4 v[28:31], v26, s[68:71], 0 offen // (op 76) Load value




v_mov_b32 v14, v24 // (op 72) reg expression

buffer_load_dwordx4 v[32:35], v14, s[64:67], 0 offen // (op 72) Load value


v_and_b32 v14, -4, v1 // (op 10698) 
v_sub_u32 v24, v1, v14 // (op 10698) 

v_lshlrev_b32 v14, 5, v24 // (op 10698) 

v_lshrrev_b32 v24, 2, v1 // (op 10698) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 10698) 


v_lshlrev_b32 v14, 2, v26 // (op 10698) 

v_lshrrev_b32 v24, 3, v14 // (op 10698) 

s_nop 0
v_readfirstlane_b32 s47, v24 // (op 10698) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 6 // (op 10312) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 10312) 

v_lshrrev_b64 v[26:27], 3, s[18:19] // (op 10312) 


v_mov_b64 v[36:37], v[20:21] // (op 10312) 
s_add_u32 s0, 16384, s47 // (op 10312) 
s_mov_b32 m0, s0 // (op 10312) reg expression
buffer_load_dwordx4 v36, s[60:63], 64 offen offset: 0 lds // (op 10312) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v26 // (op 10312) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 10312) most significant half

s_add_u32 m0, m0, 4096 // (op 10312) 
buffer_load_dwordx4 v36, s[60:63], 64 offen offset: 0 lds // (op 10312) Load value direct to lds




v_and_b32 v14, -4, v1 // (op 10654) 
v_sub_u32 v24, v1, v14 // (op 10654) 

v_lshlrev_b32 v14, 5, v24 // (op 10654) 

v_lshrrev_b32 v24, 2, v1 // (op 10654) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 10654) 


v_lshlrev_b32 v14, 2, v26 // (op 10654) 

v_lshrrev_b32 v24, 3, v14 // (op 10654) 

s_nop 0
v_readfirstlane_b32 s47, v24 // (op 10654) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 6 // (op 10314) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 10314) 

v_lshrrev_b64 v[26:27], 3, s[18:19] // (op 10314) 


v_mov_b64 v[36:37], v[16:17] // (op 10314) 
s_add_u32 s0, 24576, s47 // (op 10314) 
s_mov_b32 m0, s0 // (op 10314) reg expression
buffer_load_dwordx4 v36, s[48:51], 64 offen offset: 0 lds // (op 10314) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v26 // (op 10314) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 10314) most significant half

s_add_u32 m0, m0, 4096 // (op 10314) 
buffer_load_dwordx4 v36, s[48:51], 64 offen offset: 0 lds // (op 10314) Load value direct to lds




v_and_b32 v14, -4, v1 // (op 10786) 
v_sub_u32 v24, v1, v14 // (op 10786) 

v_lshlrev_b32 v14, 5, v24 // (op 10786) 

v_lshrrev_b32 v24, 2, v1 // (op 10786) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 10786) 


v_lshlrev_b32 v14, 2, v26 // (op 10786) 

v_lshrrev_b32 v24, 3, v14 // (op 10786) 

s_nop 0
v_readfirstlane_b32 s47, v24 // (op 10786) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 6 // (op 10316) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 10316) 

v_lshrrev_b64 v[26:27], 3, s[18:19] // (op 10316) 


v_mov_b64 v[36:37], v[20:21] // (op 10316) 
s_add_u32 s0, 32768, s47 // (op 10316) 
s_mov_b32 m0, s0 // (op 10316) reg expression
s_mov_b32 s1, 128 // (op 10316) call()
buffer_load_dwordx4 v36, s[60:63], s1 offen offset: 0 lds // (op 10316) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v26 // (op 10316) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 10316) most significant half

s_add_u32 m0, m0, 4096 // (op 10316) 
buffer_load_dwordx4 v36, s[60:63], s1 offen offset: 0 lds // (op 10316) Load value direct to lds





v_and_b32 v14, -4, v1 // (op 10742) 
v_sub_u32 v24, v1, v14 // (op 10742) 

v_lshlrev_b32 v14, 5, v24 // (op 10742) 

v_lshrrev_b32 v24, 2, v1 // (op 10742) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 10742) 


v_lshlrev_b32 v14, 2, v26 // (op 10742) 

v_lshrrev_b32 v24, 3, v14 // (op 10742) 

s_nop 0
v_readfirstlane_b32 s47, v24 // (op 10742) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 6 // (op 10318) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 10318) 

v_lshrrev_b64 v[26:27], 3, s[18:19] // (op 10318) 


v_mov_b64 v[36:37], v[16:17] // (op 10318) 
s_add_u32 s0, 40960, s47 // (op 10318) 
s_mov_b32 m0, s0 // (op 10318) reg expression
s_mov_b32 s1, 128 // (op 10318) call()
buffer_load_dwordx4 v36, s[48:51], s1 offen offset: 0 lds // (op 10318) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v26 // (op 10318) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 10318) most significant half

s_add_u32 m0, m0, 4096 // (op 10318) 
buffer_load_dwordx4 v36, s[48:51], s1 offen offset: 0 lds // (op 10318) Load value direct to lds





v_and_b32 v14, -4, v1 // (op 10874) 
v_sub_u32 v24, v1, v14 // (op 10874) 

v_lshlrev_b32 v14, 5, v24 // (op 10874) 

v_lshrrev_b32 v24, 2, v1 // (op 10874) 
v_lshl_add_u32 v26, v24, 7, v14 // (op 10874) 


v_lshlrev_b32 v14, 2, v26 // (op 10874) 

v_lshrrev_b32 v24, 3, v14 // (op 10874) 

s_nop 0
v_readfirstlane_b32 s47, v24 // (op 10874) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[30:31], 6 // (op 10320) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 10320) 

v_lshrrev_b64 v[26:27], 3, s[18:19] // (op 10320) 


v_mov_b64 v[36:37], v[20:21] // (op 10320) 
s_add_u32 s0, 49152, s47 // (op 10320) 
s_mov_b32 m0, s0 // (op 10320) reg expression
s_mov_b32 s1, 192 // (op 10320) call()
buffer_load_dwordx4 v36, s[60:63], s1 offen offset: 0 lds // (op 10320) Load value direct to lds
v_add_co_u32 v36, s[18:19], v36, v26 // (op 10320) least significant half
v_addc_co_u32 v37, s[18:19], 0, v37, s[18:19] // (op 10320) most significant half

s_add_u32 m0, m0, 4096 // (op 10320) 
buffer_load_dwordx4 v36, s[60:63], s1 offen offset: 0 lds // (op 10320) Load value direct to lds






v_and_b32 v14, -4, v1 // (op 10830) 
v_sub_u32 v24, v1, v14 // (op 10830) 

v_lshlrev_b32 v14, 5, v24 // (op 10830) 

v_lshrrev_b32 v24, 2, v1 // (op 10830) 
v_lshl_add_u32 v20, v24, 7, v14 // (op 10830) 


v_lshlrev_b32 v14, 2, v20 // (op 10830) 

v_lshrrev_b32 v24, 3, v14 // (op 10830) 

s_nop 0
v_readfirstlane_b32 s47, v24 // (op 10830) Wait state hazard: VALU Write Readlane Hazard


s_lshl_b64 s[0:1], s[26:27], 6 // (op 10322) 
s_lshl_b64 s[18:19], s[0:1], 2 // (op 10322) 

v_lshrrev_b64 v[20:21], 3, s[18:19] // (op 10322) 


v_mov_b64 v[26:27], v[16:17] // (op 10322) 
s_add_u32 s0, 57344, s47 // (op 10322) 
s_mov_b32 m0, s0 // (op 10322) reg expression
s_mov_b32 s1, 192 // (op 10322) call()
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 10322) Load value direct to lds
v_add_co_u32 v26, s[18:19], v26, v20 // (op 10322) least significant half
v_addc_co_u32 v27, s[18:19], 0, v27, s[18:19] // (op 10322) most significant half

s_add_u32 m0, m0, 4096 // (op 10322) 
buffer_load_dwordx4 v26, s[48:51], s1 offen offset: 0 lds // (op 10322) Load value direct to lds






s_waitcnt vmcnt(16)
s_barrier  // (op 11275) 



ds_read_b128 v[36:39], v4 // (op 9884) Load local data 



ds_read_b128 v[40:43], v4 offset:1024 // (op 9889) Load local data 



ds_read_b128 v[44:47], v4 offset:2048 // (op 9894) Load local data 



ds_read_b128 v[48:51], v4 offset:3072 // (op 9899) Load local data 
s_waitcnt vmcnt(14)
s_barrier  // (op 11258) 



ds_read_b128 v[52:55], v3 offset:8192 // (op 9864) Load local data 



ds_read_b128 v[56:59], v3 offset:9216 // (op 9869) Load local data 



ds_read_b128 v[60:63], v3 offset:10240 // (op 9874) Load local data 



ds_read_b128 v[64:67], v3 offset:11264 // (op 9879) Load local data 
v_and_b32 v14, -4, v1 // (op 10502) 
v_sub_u32 v24, v1, v14 // (op 10502) 

v_lshlrev_b32 v14, 5, v24 // (op 10502) 

v_lshrrev_b32 v24, 2, v1 // (op 10502) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 10502) 


v_lshlrev_b32 v14, 2, v16 // (op 10502) 

v_lshrrev_b32 v24, 3, v14 // (op 10502) 

s_nop 0
v_readfirstlane_b32 s47, v24 // (op 10502) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 10559) 
v_sub_u32 v24, v1, v14 // (op 10559) 

v_lshlrev_b32 v14, 5, v24 // (op 10559) 

v_lshrrev_b32 v24, 2, v1 // (op 10559) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 10559) 


v_lshlrev_b32 v14, 2, v16 // (op 10559) 

v_lshrrev_b32 v24, 3, v14 // (op 10559) 

s_nop 0
v_readfirstlane_b32 s0, v24 // (op 10559) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 10639) 
v_sub_u32 v24, v1, v14 // (op 10639) 

v_lshlrev_b32 v14, 5, v24 // (op 10639) 

v_lshrrev_b32 v24, 2, v1 // (op 10639) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 10639) 


v_lshlrev_b32 v14, 2, v16 // (op 10639) 

v_lshrrev_b32 v24, 3, v14 // (op 10639) 

s_nop 0
v_readfirstlane_b32 s1, v24 // (op 10639) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 10683) 
v_sub_u32 v24, v1, v14 // (op 10683) 

v_lshlrev_b32 v14, 5, v24 // (op 10683) 

v_lshrrev_b32 v24, 2, v1 // (op 10683) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 10683) 


v_lshlrev_b32 v14, 2, v16 // (op 10683) 

v_lshrrev_b32 v24, 3, v14 // (op 10683) 

s_nop 0
v_readfirstlane_b32 s18, v24 // (op 10683) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 10727) 
v_sub_u32 v24, v1, v14 // (op 10727) 

v_lshlrev_b32 v14, 5, v24 // (op 10727) 

v_lshrrev_b32 v24, 2, v1 // (op 10727) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 10727) 


v_lshlrev_b32 v14, 2, v16 // (op 10727) 

v_lshrrev_b32 v24, 3, v14 // (op 10727) 

s_nop 0
v_readfirstlane_b32 s19, v24 // (op 10727) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 10771) 
v_sub_u32 v24, v1, v14 // (op 10771) 

v_lshlrev_b32 v14, 5, v24 // (op 10771) 

v_lshrrev_b32 v24, 2, v1 // (op 10771) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 10771) 


v_lshlrev_b32 v14, 2, v16 // (op 10771) 

v_lshrrev_b32 v24, 3, v14 // (op 10771) 

s_nop 0
v_readfirstlane_b32 s22, v24 // (op 10771) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 10815) 
v_sub_u32 v24, v1, v14 // (op 10815) 

v_lshlrev_b32 v14, 5, v24 // (op 10815) 

v_lshrrev_b32 v24, 2, v1 // (op 10815) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 10815) 


v_lshlrev_b32 v14, 2, v16 // (op 10815) 

v_lshrrev_b32 v24, 3, v14 // (op 10815) 

s_nop 0
v_readfirstlane_b32 s23, v24 // (op 10815) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v14, -4, v1 // (op 10859) 
v_sub_u32 v24, v1, v14 // (op 10859) 

v_lshlrev_b32 v14, 5, v24 // (op 10859) 

v_lshrrev_b32 v24, 2, v1 // (op 10859) 
v_lshl_add_u32 v16, v24, 7, v14 // (op 10859) 


v_lshlrev_b32 v14, 2, v16 // (op 10859) 

v_lshrrev_b32 v24, 3, v14 // (op 10859) 

s_nop 0
v_readfirstlane_b32 s24, v24 // (op 10859) Wait state hazard: VALU Write Readlane Hazard

s_mov_b32 s25, 0 // (op 3288) call()
s_ashr_i64 s[72:73], s[42:43], 7 // (op 3287) 
s_mov_b32 s74, s72 // (op 3287) convert

s_and_b32 s72, s74, 4294967292 // (op 3287) 

s_cmp_lt_u32 s25, s72 // (op 3287) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopBottom_KLoop_3287 // (op 3287) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopBottom_KLoop_3287 if false)
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopTop_KLoop_3287:
s_add_u32 s25, s25, 4 // (op 3289) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 11278) 
s_waitcnt vmcnt(13)
v_mov_b32 v68, v28 // (op 10202) reg expression

v_mov_b32 v69, v29 // (op 10202) reg expression
v_mov_b32 v70, v30 // (op 10202) reg expression
v_mov_b32 v71, v31 // (op 10202) reg expression
s_lshl_b64 s[72:73], s[30:31], 6 // (op 10255) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 10255) 

v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 10255) 


v_mov_b64 v[20:21], v[22:23] // (op 10255) 
v_mov_b64 v[26:27], v[20:21] // (op 10255) 
s_mov_b32 m0, s0 // (op 10255) reg expression
s_mov_b32 s72, 256 // (op 10255) call()
s_waitcnt vmcnt(12)
v_mov_b32 v72, v32 // (op 10215) reg expression
v_mov_b32 v73, v33 // (op 10215) reg expression
v_mov_b32 v74, v34 // (op 10215) reg expression
v_mov_b32 v75, v35 // (op 10215) reg expression
buffer_load_dwordx4 v26, s[60:63], s72 offen offset: 0 lds // (op 10255) Load value direct to lds
v_add_co_u32 v26, s[74:75], v26, v16 // (op 10255) least significant half
v_addc_co_u32 v27, s[74:75], 0, v27, s[74:75] // (op 10255) most significant half

s_add_u32 m0, m0, 4096 // (op 10255) 
v_bfe_u32 v76, v72, 0, 8 // (op 9217) 
v_bfe_u32 v77, v72, 8, 8 // (op 9217) 
v_bfe_u32 v78, v72, 16, 8 // (op 9217) 
v_bfe_u32 v79, v72, 24, 8 // (op 9217) 
v_permlane16_swap_b32 v76, v77 // (op 9217) 
s_nop 0
v_permlane16_swap_b32 v78, v79 // (op 9217) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v76, v78 // (op 9217) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v77, v79 // (op 9217) 
buffer_load_dwordx4 v26, s[60:63], s72 offen offset: 0 lds // (op 10255) Load value direct to lds



v_bfe_u32 v80, v68, 0, 8 // (op 9346) 
v_bfe_u32 v81, v68, 8, 8 // (op 9346) 
v_bfe_u32 v82, v68, 16, 8 // (op 9346) 
v_bfe_u32 v83, v68, 24, 8 // (op 9346) 
v_permlane16_swap_b32 v80, v81 // (op 9346) 
s_nop 0
v_permlane16_swap_b32 v82, v83 // (op 9346) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v80, v82 // (op 9346) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v81, v83 // (op 9346) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 11261) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[52:55], v[36:39], a[0:3], v76, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 78) 
s_lshl_b64 s[72:73], s[26:27], 6 // (op 10250) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[52:55], v[40:43], a[4:7], v76, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 421) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 10250) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[52:55], v[44:47], a[8:11], v76, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 647) 
v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 10250) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[52:55], v[48:51], a[12:15], v76, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 873) 
v_mov_b64 v[26:27], v[18:19] // (op 10250) 
v_mov_b64 v[84:85], v[26:27] // (op 10250) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[56:59], v[36:39], a[16:19], v77, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1545) 
s_add_u32 s72, 8192, s47 // (op 10250) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[56:59], v[40:43], a[20:23], v77, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1469) 
s_mov_b32 m0, s72 // (op 10250) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[56:59], v[44:47], a[24:27], v77, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1393) 
s_mov_b32 s73, 256 // (op 10250) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[56:59], v[48:51], a[28:31], v77, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1317) 
buffer_load_dwordx4 v84, s[48:51], s73 offen offset: 0 lds // (op 10250) Load value direct to lds
v_add_co_u32 v84, s[74:75], v84, v16 // (op 10250) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[60:63], v[36:39], a[32:35], v78, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2547) 
v_addc_co_u32 v85, s[74:75], 0, v85, s[74:75] // (op 10250) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[60:63], v[40:43], a[36:39], v78, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2471) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[60:63], v[44:47], a[40:43], v78, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2395) 
s_add_u32 m0, m0, 4096 // (op 10250) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[60:63], v[48:51], a[44:47], v78, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2319) 
s_waitcnt vmcnt(13)
s_barrier  // (op 11298) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[64:67], v[36:39], a[48:51], v79, v80 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3549) 
ds_read_b128 v[86:89], v6 offset:16384 // (op 216) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[64:67], v[40:43], a[52:55], v79, v81 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3473) 
ds_read_b128 v[90:93], v6 offset:17408 // (op 432) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[64:67], v[44:47], a[56:59], v79, v82 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3397) 
ds_read_b128 v[94:97], v6 offset:18432 // (op 658) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[64:67], v[48:51], a[60:63], v79, v83 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3321) 
ds_read_b128 v[98:101], v6 offset:19456 // (op 884) Load local data 


buffer_load_dwordx4 v84, s[48:51], s73 offen offset: 0 lds // (op 10250) Load value direct to lds






v_mov_b32 v14, v15 // (op 10210) 
v_add_u32 v24, 16, v14 // (op 10210) 

buffer_load_dwordx4 v[28:31], v24, s[68:71], 0 offen // (op 10210) Load value


v_add_u32 v15, 16, v15 // (op 10976) 
s_waitcnt vmcnt(13)
s_barrier  // (op 11291) 





ds_read_b128 v[52:55], v5 offset:24576 // (op 214) Load local data 
v_mov_b32 v14, v25 // (op 10223) 



ds_read_b128 v[56:59], v5 offset:25600 // (op 1327) Load local data 
v_add_u32 v24, 16, v14 // (op 10223) 




ds_read_b128 v[60:63], v5 offset:26624 // (op 2329) Load local data 
buffer_load_dwordx4 v[32:35], v24, s[64:67], 0 offen // (op 10223) Load value





v_add_u32 v25, 16, v25 // (op 10924) 
ds_read_b128 v[76:79], v5 offset:27648 // (op 3331) Load local data 
v_bfe_u32 v80, v73, 0, 8 // (op 9232) 
v_bfe_u32 v81, v73, 8, 8 // (op 9232) 
v_bfe_u32 v82, v73, 16, 8 // (op 9232) 
v_bfe_u32 v83, v73, 24, 8 // (op 9232) 
v_permlane16_swap_b32 v80, v81 // (op 9232) 
s_nop 0
v_permlane16_swap_b32 v82, v83 // (op 9232) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v80, v82 // (op 9232) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v81, v83 // (op 9232) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 11302) 

s_lshl_b64 s[72:73], s[30:31], 6 // (op 10277) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 10277) 

v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 10277) 


v_mov_b64 v[84:85], v[20:21] // (op 10277) 
s_add_u32 s72, 16384, s18 // (op 10277) 
s_mov_b32 m0, s72 // (op 10277) reg expression
s_mov_b32 s73, 320 // (op 10277) call()
v_bfe_u32 v102, v69, 0, 8 // (op 9361) 
v_bfe_u32 v103, v69, 8, 8 // (op 9361) 
v_bfe_u32 v104, v69, 16, 8 // (op 9361) 
v_bfe_u32 v105, v69, 24, 8 // (op 9361) 
v_permlane16_swap_b32 v102, v103 // (op 9361) 
s_nop 0
v_permlane16_swap_b32 v104, v105 // (op 9361) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v102, v104 // (op 9361) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v103, v105 // (op 9361) 
buffer_load_dwordx4 v84, s[60:63], s73 offen offset: 0 lds // (op 10277) Load value direct to lds
v_add_co_u32 v84, s[74:75], v84, v16 // (op 10277) least significant half
v_addc_co_u32 v85, s[74:75], 0, v85, s[74:75] // (op 10277) most significant half

s_add_u32 m0, m0, 4096 // (op 10277) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[52:55], v[86:89], a[0:3], v80, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 221) 
buffer_load_dwordx4 v84, s[60:63], s73 offen offset: 0 lds // (op 10277) Load value direct to lds




v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[52:55], v[90:93], a[4:7], v80, v103 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 422) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 11295) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[52:55], v[94:97], a[8:11], v80, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 648) 
s_lshl_b64 s[72:73], s[26:27], 6 // (op 10272) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[52:55], v[98:101], a[12:15], v80, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 874) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 10272) 

v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 10272) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[56:59], v[86:89], a[16:19], v81, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1546) 
v_mov_b64 v[84:85], v[26:27] // (op 10272) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[56:59], v[90:93], a[20:23], v81, v103 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1470) 
s_add_u32 s72, 24576, s1 // (op 10272) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[56:59], v[94:97], a[24:27], v81, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1394) 
s_mov_b32 m0, s72 // (op 10272) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[56:59], v[98:101], a[28:31], v81, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1318) 
s_mov_b32 s73, 320 // (op 10272) call()
buffer_load_dwordx4 v84, s[48:51], s73 offen offset: 0 lds // (op 10272) Load value direct to lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[60:63], v[86:89], a[32:35], v82, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2548) 
v_add_co_u32 v84, s[74:75], v84, v16 // (op 10272) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[60:63], v[90:93], a[36:39], v82, v103 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2472) 
v_addc_co_u32 v85, s[74:75], 0, v85, s[74:75] // (op 10272) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[60:63], v[94:97], a[40:43], v82, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2396) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[60:63], v[98:101], a[44:47], v82, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2320) 
s_add_u32 m0, m0, 4096 // (op 10272) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[76:79], v[86:89], a[48:51], v83, v102 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3550) 
s_waitcnt vmcnt(15)
s_barrier  // (op 11313) 




v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[76:79], v[90:93], a[52:55], v83, v103 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3474) 
ds_read_b128 v[36:39], v8 offset:32768 // (op 244) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[76:79], v[94:97], a[56:59], v83, v104 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3398) 
ds_read_b128 v[40:43], v8 offset:33792 // (op 442) Load local data 




v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[76:79], v[98:101], a[60:63], v83, v105 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3322) 
ds_read_b128 v[44:47], v8 offset:34816 // (op 668) Load local data 







ds_read_b128 v[48:51], v8 offset:35840 // (op 894) Load local data 
buffer_load_dwordx4 v84, s[48:51], s73 offen offset: 0 lds // (op 10272) Load value direct to lds




s_waitcnt vmcnt(14)
s_barrier  // (op 11306) 



ds_read_b128 v[52:55], v7 offset:40960 // (op 242) Load local data 



ds_read_b128 v[56:59], v7 offset:41984 // (op 1337) Load local data 



ds_read_b128 v[76:79], v7 offset:43008 // (op 2339) Load local data 



ds_read_b128 v[80:83], v7 offset:44032 // (op 3341) Load local data 
v_bfe_u32 v84, v74, 0, 8 // (op 9249) 
v_bfe_u32 v85, v74, 8, 8 // (op 9249) 
v_bfe_u32 v86, v74, 16, 8 // (op 9249) 
v_bfe_u32 v87, v74, 24, 8 // (op 9249) 
v_permlane16_swap_b32 v84, v85 // (op 9249) 
s_nop 0
v_permlane16_swap_b32 v86, v87 // (op 9249) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v84, v86 // (op 9249) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v85, v87 // (op 9249) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 11316) 

s_lshl_b64 s[72:73], s[30:31], 6 // (op 10287) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 10287) 

v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 10287) 


v_mov_b64 v[88:89], v[20:21] // (op 10287) 
s_add_u32 s72, 32768, s22 // (op 10287) 
s_mov_b32 m0, s72 // (op 10287) reg expression
s_mov_b32 s73, 384 // (op 10287) call()
buffer_load_dwordx4 v88, s[60:63], s73 offen offset: 0 lds // (op 10287) Load value direct to lds
v_add_co_u32 v88, s[74:75], v88, v16 // (op 10287) least significant half
v_addc_co_u32 v89, s[74:75], 0, v89, s[74:75] // (op 10287) most significant half

s_add_u32 m0, m0, 4096 // (op 10287) 
buffer_load_dwordx4 v88, s[60:63], s73 offen offset: 0 lds // (op 10287) Load value direct to lds




v_bfe_u32 v88, v70, 0, 8 // (op 9379) 
v_bfe_u32 v89, v70, 8, 8 // (op 9379) 
v_bfe_u32 v90, v70, 16, 8 // (op 9379) 
v_bfe_u32 v91, v70, 24, 8 // (op 9379) 
v_permlane16_swap_b32 v88, v89 // (op 9379) 
s_nop 0
v_permlane16_swap_b32 v90, v91 // (op 9379) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v88, v90 // (op 9379) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v89, v91 // (op 9379) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 11310) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[52:55], v[36:39], a[0:3], v84, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 249) 
s_lshl_b64 s[72:73], s[26:27], 6 // (op 10282) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[52:55], v[40:43], a[4:7], v84, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 423) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 10282) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[52:55], v[44:47], a[8:11], v84, v90 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 649) 
v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 10282) 


v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[52:55], v[48:51], a[12:15], v84, v91 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 875) 
v_mov_b64 v[92:93], v[26:27] // (op 10282) 
s_add_u32 s72, 40960, s19 // (op 10282) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[56:59], v[36:39], a[16:19], v85, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1547) 
s_mov_b32 m0, s72 // (op 10282) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[56:59], v[40:43], a[20:23], v85, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1471) 
s_mov_b32 s73, 384 // (op 10282) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[56:59], v[44:47], a[24:27], v85, v90 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1395) 
s_waitcnt vmcnt(14)
s_barrier  // (op 11326) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[56:59], v[48:51], a[28:31], v85, v91 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1319) 
ds_read_b128 v[60:63], v10 offset:49152 // (op 272) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[76:79], v[36:39], a[32:35], v86, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2549) 
ds_read_b128 v[64:67], v10 offset:50176 // (op 452) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[76:79], v[40:43], a[36:39], v86, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2473) 
ds_read_b128 v[94:97], v10 offset:51200 // (op 678) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[76:79], v[44:47], a[40:43], v86, v90 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2397) 
ds_read_b128 v[98:101], v10 offset:52224 // (op 904) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[76:79], v[48:51], a[44:47], v86, v91 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2321) 

s_waitcnt vmcnt(12)
s_barrier  // (op 11319) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[80:83], v[36:39], a[48:51], v87, v88 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3551) 
ds_read_b128 v[52:55], v9 offset:57344 // (op 270) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[80:83], v[40:43], a[52:55], v87, v89 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3475) 
ds_read_b128 v[56:59], v9 offset:58368 // (op 1347) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[80:83], v[44:47], a[56:59], v87, v90 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3399) 
ds_read_b128 v[76:79], v9 offset:59392 // (op 2349) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[80:83], v[48:51], a[60:63], v87, v91 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3323) 
ds_read_b128 v[102:105], v9 offset:60416 // (op 3351) Load local data 



buffer_load_dwordx4 v92, s[48:51], s73 offen offset: 0 lds // (op 10282) Load value direct to lds
v_add_co_u32 v92, s[74:75], v92, v16 // (op 10282) least significant half
v_addc_co_u32 v93, s[74:75], 0, v93, s[74:75] // (op 10282) most significant half

s_add_u32 m0, m0, 4096 // (op 10282) 
buffer_load_dwordx4 v92, s[48:51], s73 offen offset: 0 lds // (op 10282) Load value direct to lds




v_bfe_u32 v80, v75, 0, 8 // (op 9266) 
v_bfe_u32 v81, v75, 8, 8 // (op 9266) 
v_bfe_u32 v82, v75, 16, 8 // (op 9266) 
v_bfe_u32 v83, v75, 24, 8 // (op 9266) 
v_permlane16_swap_b32 v80, v81 // (op 9266) 
s_nop 0
v_permlane16_swap_b32 v82, v83 // (op 9266) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v80, v82 // (op 9266) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v81, v83 // (op 9266) 
s_waitcnt lgkmcnt(4)
s_barrier  // (op 11329) 


s_lshl_b64 s[72:73], s[30:31], 6 // (op 10297) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 10297) 

v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 10297) 


v_mov_b64 v[72:73], v[20:21] // (op 10297) 
s_add_u32 s72, 49152, s24 // (op 10297) 
s_mov_b32 m0, s72 // (op 10297) reg expression
s_mov_b32 s73, 448 // (op 10297) call()
buffer_load_dwordx4 v72, s[60:63], s73 offen offset: 0 lds // (op 10297) Load value direct to lds
v_add_co_u32 v72, s[74:75], v72, v16 // (op 10297) least significant half
v_addc_co_u32 v73, s[74:75], 0, v73, s[74:75] // (op 10297) most significant half

s_add_u32 m0, m0, 4096 // (op 10297) 
v_bfe_u32 v84, v71, 0, 8 // (op 9397) 
v_bfe_u32 v85, v71, 8, 8 // (op 9397) 
v_bfe_u32 v86, v71, 16, 8 // (op 9397) 
v_bfe_u32 v87, v71, 24, 8 // (op 9397) 
v_permlane16_swap_b32 v84, v85 // (op 9397) 
s_nop 0
v_permlane16_swap_b32 v86, v87 // (op 9397) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v84, v86 // (op 9397) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v85, v87 // (op 9397) 
buffer_load_dwordx4 v72, s[60:63], s73 offen offset: 0 lds // (op 10297) Load value direct to lds






v_add_co_u32 v22, vcc, 256, v22 // (op 10451) least significant half
v_addc_co_u32 v23, vcc, 0, v23, vcc // (op 10451) most significant half
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[52:55], v[60:63], a[0:3], v80, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 277) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 11323) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[52:55], v[64:67], a[4:7], v80, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 424) 
s_lshl_b64 s[72:73], s[26:27], 6 // (op 10292) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[52:55], v[94:97], a[8:11], v80, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 650) 
s_lshl_b64 s[74:75], s[72:73], 2 // (op 10292) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[52:55], v[98:101], a[12:15], v80, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 876) 
v_lshrrev_b64 v[16:17], 3, s[74:75] // (op 10292) 


v_mov_b64 v[20:21], v[26:27] // (op 10292) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[56:59], v[60:63], a[16:19], v81, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1548) 
s_add_u32 s72, 57344, s23 // (op 10292) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[56:59], v[64:67], a[20:23], v81, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1472) 
s_mov_b32 m0, s72 // (op 10292) reg expression
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[56:59], v[94:97], a[24:27], v81, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1396) 
s_mov_b32 s73, 448 // (op 10292) call()
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[56:59], v[98:101], a[28:31], v81, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1320) 
s_waitcnt vmcnt(14)
s_barrier  // (op 11284) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[76:79], v[60:63], a[32:35], v82, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2550) 
ds_read_b128 v[36:39], v4 // (op 67) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[76:79], v[64:67], a[36:39], v82, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2474) 
ds_read_b128 v[40:43], v4 offset:1024 // (op 418) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[76:79], v[94:97], a[40:43], v82, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2398) 
ds_read_b128 v[44:47], v4 offset:2048 // (op 644) Load local data 



v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[76:79], v[98:101], a[44:47], v82, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2322) 
buffer_load_dwordx4 v20, s[48:51], s73 offen offset: 0 lds // (op 10292) Load value direct to lds

v_add_co_u32 v20, s[74:75], v20, v16 // (op 10292) least significant half
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[102:105], v[60:63], a[48:51], v83, v84 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3552) 
v_addc_co_u32 v21, s[74:75], 0, v21, s[74:75] // (op 10292) most significant half

v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[102:105], v[64:67], a[52:55], v83, v85 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3476) 
s_add_u32 m0, m0, 4096 // (op 10292) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[102:105], v[94:97], a[56:59], v83, v86 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3400) 
ds_read_b128 v[48:51], v4 offset:3072 // (op 870) Load local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[102:105], v[98:101], a[60:63], v83, v87 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3324) 
buffer_load_dwordx4 v20, s[48:51], s73 offen offset: 0 lds // (op 10292) Load value direct to lds









s_waitcnt vmcnt(14)
s_barrier  // (op 11267) 



ds_read_b128 v[52:55], v3 offset:8192 // (op 62) Load local data 
v_add_co_u32 v18, vcc, 256, v18 // (op 10387) least significant half



v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 10387) most significant half
ds_read_b128 v[56:59], v3 offset:9216 // (op 1313) Load local data 



ds_read_b128 v[60:63], v3 offset:10240 // (op 2315) Load local data 



ds_read_b128 v[64:67], v3 offset:11264 // (op 3317) Load local data 
s_ashr_i64 s[72:73], s[42:43], 7 // (op 3287) 
s_mov_b32 s74, s72 // (op 3287) convert

s_and_b32 s72, s74, 4294967292 // (op 3287) 

s_cmp_lt_u32 s25, s72 // (op 3287) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopTop_KLoop_3287 // (op 3287) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopTop_KLoop_3287 if true)
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopBottom_KLoop_3287:















s_barrier  // (op 11402) 
s_mul_hi_i32 s47, s16, s10 // (op 10335) 
v_lshrrev_b32 v14, 2, v1 // (op 10335) 
s_add_i32 s0, s47, s16 // (op 10335) 

v_and_b32 v15, -4, v1 // (op 10335) 
s_ashr_i32 s1, s0, 31 // (op 10335) 
v_sub_u32 v16, v1, v15 // (op 10335) 

s_cmp_eq_i32 s10, 0 // (op 10335) 
s_mov_b32 s47, scc // (op 10335) 
v_lshlrev_b32 v15, 5, v16 // (op 10335) 

s_cmp_eq_i32 s47, 0 // (op 10335) 
s_cselect_b32 s18, 0, -1 // (op 10335) 

s_and_b32 s19, s13, 31 // (op 10335) 
s_lshl_b32 s47, 1, s19 // (op 10335) 
s_add_i32 s22, s47, s18 // (op 10335) 


s_and_b32 s18, s1, s22 // (op 10335) 


s_add_i32 s1, s0, s18 // (op 10335) 


s_ashr_i32 s0, s1, s19 // (op 10335) 


s_lshl_b32 s1, s13, 24 // (op 10335) 
s_ashr_i32 s47, s1, 31 // (op 10335) 

s_xor_b32 s1, s0, s47 // (op 10335) 

s_sub_i32 s0, s1, s47 // (op 10335) 


s_mul_hi_i32 s1, s0, s14 // (op 10335) 
s_add_i32 s47, s1, s0 // (op 10335) 

s_ashr_i32 s1, s47, 31 // (op 10335) 
s_cmp_eq_i32 s14, 0 // (op 10335) 
s_mov_b32 s18, scc // (op 10335) 
s_cmp_eq_i32 s18, 0 // (op 10335) 
s_cselect_b32 s19, 0, -1 // (op 10335) 

s_and_b32 s18, s15, 31 // (op 10335) 
s_lshl_b32 s22, 1, s18 // (op 10335) 
s_add_i32 s23, s22, s19 // (op 10335) 


s_and_b32 s19, s1, s23 // (op 10335) 


s_add_i32 s1, s47, s19 // (op 10335) 


s_ashr_i32 s19, s1, s18 // (op 10335) 


s_lshl_b32 s1, s15, 24 // (op 10335) 
s_ashr_i32 s18, s1, 31 // (op 10335) 

s_xor_b32 s1, s19, s18 // (op 10335) 

s_sub_i32 s19, s1, s18 // (op 10335) 


s_mul_i32 s1, s0, s11 // (op 10335) 

s_sub_u32 s0, s16, s1 // (op 10335) 

s_mul_i32 s1, s19, s11 // (op 10335) 
s_add_u32 s1, s1, s0 // (op 10335) 


s_mul_hi_i32 s0, s16, s9 // (op 10335) 
s_add_i32 s47, s0, s16 // (op 10335) 

s_ashr_i32 s0, s47, 31 // (op 10335) 
s_cmp_eq_i32 s9, 0 // (op 10335) 
s_mov_b32 s18, scc // (op 10335) 
s_cmp_eq_i32 s18, 0 // (op 10335) 
s_cselect_b32 s19, 0, -1 // (op 10335) 

s_and_b32 s18, s45, 31 // (op 10335) 
s_lshl_b32 s22, 1, s18 // (op 10335) 
s_add_i32 s23, s22, s19 // (op 10335) 


s_and_b32 s19, s0, s23 // (op 10335) 


s_add_i32 s0, s47, s19 // (op 10335) 


s_ashr_i32 s19, s0, s18 // (op 10335) 


s_lshl_b32 s0, s45, 24 // (op 10335) 
s_ashr_i32 s18, s0, 31 // (op 10335) 

s_xor_b32 s0, s19, s18 // (op 10335) 

s_sub_i32 s19, s0, s18 // (op 10335) 


s_add_u32 s22, s6, 128 // (op 10335) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 10335) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 10335) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 10335) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 10335) 

s_mul_i32 s0, s11, s22 // (op 10335) 

s_mul_i32 s18, s8, s0 // (op 10335) 

s_mul_i32 s0, s19, s18 // (op 10335) 

s_sub_u32 s18, s16, s0 // (op 10335) 

s_mov_b32 s0, s18 // (op 10335) 
s_mul_hi_i32 s47, s0, s12 // (op 10335) 
s_add_i32 s22, s47, s0 // (op 10335) 


s_ashr_i32 s0, s22, 31 // (op 10335) 
s_cmp_eq_i32 s12, 0 // (op 10335) 
s_mov_b32 s47, scc // (op 10335) 
s_cmp_eq_i32 s47, 0 // (op 10335) 
s_cselect_b32 s23, 0, -1 // (op 10335) 

s_and_b32 s47, s44, 31 // (op 10335) 
s_lshl_b32 s24, 1, s47 // (op 10335) 
s_add_i32 s25, s24, s23 // (op 10335) 


s_and_b32 s23, s0, s25 // (op 10335) 


s_add_i32 s0, s22, s23 // (op 10335) 


s_ashr_i32 s22, s0, s47 // (op 10335) 


s_lshl_b32 s0, s44, 24 // (op 10335) 
s_ashr_i32 s47, s0, 31 // (op 10335) 

s_xor_b32 s0, s22, s47 // (op 10335) 

s_sub_i32 s22, s0, s47 // (op 10335) 


s_mul_i32 s0, s46, s11 // (op 10335) 
s_add_u32 s24, s20, 128 // (op 10335) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 10335) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 10335) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 10335) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 7 // (op 10335) 

s_sub_i32 s23, s24, s0 // (op 10335) 

s_mul_i32 s47, s22, s23 // (op 10335) 


s_sub_u32 s22, s18, s47 // (op 10335) 


s_add_u32 s18, s22, s0 // (op 10335) 


s_cmp_eq_i32 s19, 0 // (op 10335) 
s_mov_b32 s0, scc // (op 10335) 

s_cmp_eq_i32 s0, 0 // (op 10335) 
s_cselect_b32 s19, s18, s1 // (op 10335) 



v_lshl_add_u32 v16, s19, 7, v14 // (op 10335) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 10335) 
s_lshr_b32 s47, s0, 2 // (op 10335) 

v_lshl_add_u32 v14, s47, 9, v15 // (op 10335) 


v_mov_b32 v27, 0 // (op 10335) convert
v_mov_b32 v26, v14 // (op 10335) convert

v_mul_lo_u32 v15, v16, s27 // (op 10335) most significant: low of low * high
v_mul_hi_u32 v17, v16, s26 // (op 10335) most significant: high of low * low
v_mul_lo_u32 v14, v16, s26 // (op 10335) least significant: low of low * low
v_add_u32 v15, v15, v17 // (op 10335) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v26 // (op 10335) least significant half
v_addc_co_u32 v15, s[0:1], v15, v27, s[0:1] // (op 10335) most significant half



v_lshlrev_b64 v[16:17], 2, v[14:15] // (op 10335) 

v_lshrrev_b64 v[14:15], 3, v[16:17] // (op 10335) 

s_mul_i32 s1, s26, s21 // (op 10337) most significant: low of low * high
s_mul_i32 s47, s27, s20 // (op 10337) most significant: low of high * low
s_mul_i32 s0, s26, s20 // (op 10337) least significant
s_mul_hi_u32 s18, s26, s20 // (op 10337) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 10337) most significant: sum
s_add_u32 s1, s1, s18 // (op 10337) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 10337) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 10337) 

s_mov_b32 s48, s28 // (op 10337) reg expression
s_mov_b32 s49, s29 // (op 10337) reg expression
s_mov_b32 s50, s0 // (op 10337) reg expression
s_mov_b32 s51, 131072 // (op 10337) call()

s_mul_hi_i32 s47, s16, s10 // (op 10417) 
v_lshrrev_b32 v16, 2, v1 // (op 10417) 
s_add_i32 s0, s47, s16 // (op 10417) 

v_and_b32 v17, -4, v1 // (op 10417) 
s_ashr_i32 s1, s0, 31 // (op 10417) 
v_sub_u32 v18, v1, v17 // (op 10417) 

s_cmp_eq_i32 s10, 0 // (op 10417) 
s_mov_b32 s47, scc // (op 10417) 
v_lshlrev_b32 v17, 5, v18 // (op 10417) 

s_cmp_eq_i32 s47, 0 // (op 10417) 
s_cselect_b32 s18, 0, -1 // (op 10417) 

s_and_b32 s19, s13, 31 // (op 10417) 
s_lshl_b32 s47, 1, s19 // (op 10417) 
s_add_i32 s22, s47, s18 // (op 10417) 


s_and_b32 s18, s1, s22 // (op 10417) 


s_add_i32 s1, s0, s18 // (op 10417) 


s_ashr_i32 s0, s1, s19 // (op 10417) 


s_lshl_b32 s1, s13, 24 // (op 10417) 
s_ashr_i32 s47, s1, 31 // (op 10417) 

s_xor_b32 s1, s0, s47 // (op 10417) 

s_sub_i32 s0, s1, s47 // (op 10417) 


s_mul_hi_i32 s1, s0, s14 // (op 10417) 
s_add_i32 s47, s1, s0 // (op 10417) 

s_ashr_i32 s1, s47, 31 // (op 10417) 
s_cmp_eq_i32 s14, 0 // (op 10417) 
s_mov_b32 s18, scc // (op 10417) 
s_cmp_eq_i32 s18, 0 // (op 10417) 
s_cselect_b32 s19, 0, -1 // (op 10417) 

s_and_b32 s18, s15, 31 // (op 10417) 
s_lshl_b32 s22, 1, s18 // (op 10417) 
s_add_i32 s23, s22, s19 // (op 10417) 


s_and_b32 s19, s1, s23 // (op 10417) 


s_add_i32 s1, s47, s19 // (op 10417) 


s_ashr_i32 s19, s1, s18 // (op 10417) 


s_lshl_b32 s1, s15, 24 // (op 10417) 
s_ashr_i32 s18, s1, 31 // (op 10417) 

s_xor_b32 s1, s19, s18 // (op 10417) 

s_sub_i32 s19, s1, s18 // (op 10417) 


s_add_u32 s22, s6, 128 // (op 10417) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 10417) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 10417) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 10417) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 10417) 

s_mul_i32 s1, s19, s22 // (op 10417) 

s_sub_i32 s47, s0, s1 // (op 10417) 


s_mul_hi_i32 s0, s16, s9 // (op 10417) 
s_add_i32 s1, s0, s16 // (op 10417) 

s_ashr_i32 s0, s1, 31 // (op 10417) 
s_cmp_eq_i32 s9, 0 // (op 10417) 
s_mov_b32 s18, scc // (op 10417) 
s_cmp_eq_i32 s18, 0 // (op 10417) 
s_cselect_b32 s19, 0, -1 // (op 10417) 

s_and_b32 s18, s45, 31 // (op 10417) 
s_lshl_b32 s24, 1, s18 // (op 10417) 
s_add_i32 s25, s24, s19 // (op 10417) 


s_and_b32 s19, s0, s25 // (op 10417) 


s_add_i32 s0, s1, s19 // (op 10417) 


s_ashr_i32 s1, s0, s18 // (op 10417) 


s_lshl_b32 s0, s45, 24 // (op 10417) 
s_ashr_i32 s18, s0, 31 // (op 10417) 

s_xor_b32 s0, s1, s18 // (op 10417) 

s_sub_i32 s1, s0, s18 // (op 10417) 


s_mul_i32 s0, s11, s22 // (op 10417) 

s_mul_i32 s18, s8, s0 // (op 10417) 

s_mul_i32 s0, s1, s18 // (op 10417) 

s_sub_u32 s18, s16, s0 // (op 10417) 

s_mul_hi_i32 s0, s18, s12 // (op 10417) 
s_add_i32 s19, s0, s18 // (op 10417) 


s_ashr_i32 s0, s19, 31 // (op 10417) 
s_cmp_eq_i32 s12, 0 // (op 10417) 
s_mov_b32 s18, scc // (op 10417) 
s_cmp_eq_i32 s18, 0 // (op 10417) 
s_cselect_b32 s22, 0, -1 // (op 10417) 

s_and_b32 s18, s44, 31 // (op 10417) 
s_lshl_b32 s23, 1, s18 // (op 10417) 
s_add_i32 s24, s23, s22 // (op 10417) 


s_and_b32 s25, s0, s24 // (op 10417) 


s_add_i32 s0, s19, s25 // (op 10417) 


s_ashr_i32 s19, s0, s18 // (op 10417) 


s_lshl_b32 s0, s44, 24 // (op 10417) 
s_ashr_i32 s18, s0, 31 // (op 10417) 

s_xor_b32 s0, s19, s18 // (op 10417) 

s_sub_i32 s19, s0, s18 // (op 10417) 


s_cmp_eq_i32 s1, 0 // (op 10417) 
s_mov_b32 s0, scc // (op 10417) 

s_cmp_eq_i32 s0, 0 // (op 10417) 
s_cselect_b32 s1, s19, s47 // (op 10417) 



v_lshl_add_u32 v18, s1, 7, v16 // (op 10417) 


v_mul_lo_u32 v27, v18, s31 // (op 10417) most significant: low of low * high
v_mul_hi_u32 v16, v18, s30 // (op 10417) most significant: high of low * low
v_mul_lo_u32 v26, v18, s30 // (op 10417) least significant: low of low * low
v_add_u32 v27, v27, v16 // (op 10417) most significant: sum


s_ashr_i64 s[0:1], s[42:43], 7 // (op 10417) 
s_lshr_b32 s47, s0, 2 // (op 10417) 

v_lshl_add_u32 v16, s47, 9, v17 // (op 10417) 


v_mov_b32 v25, 0 // (op 10417) convert
v_mov_b32 v24, v16 // (op 10417) convert

v_add_co_u32 v16, s[0:1], v24, v26 // (op 10417) least significant half
v_addc_co_u32 v17, s[0:1], v25, v27, s[0:1] // (op 10417) most significant half

v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 10417) 


v_lshrrev_b64 v[18:19], 3, v[16:17] // (op 10417) 

s_mul_i32 s1, s30, s7 // (op 10419) most significant: low of low * high
s_mul_i32 s47, s31, s6 // (op 10419) most significant: low of high * low
s_mul_i32 s0, s30, s6 // (op 10419) least significant
s_mul_hi_u32 s18, s30, s6 // (op 10419) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 10419) most significant: sum
s_add_u32 s1, s1, s18 // (op 10419) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 2 // (op 10419) 

s_ashr_i64 s[0:1], s[18:19], 3 // (op 10419) 

s_mov_b32 s60, s32 // (op 10419) reg expression
s_mov_b32 s61, s33 // (op 10419) reg expression
s_mov_b32 s62, s0 // (op 10419) reg expression
s_mov_b32 s63, 131072 // (op 10419) call()

v_and_b32 v16, -4, v1 // (op 10515) 
v_sub_u32 v17, v1, v16 // (op 10515) 

v_lshlrev_b32 v16, 5, v17 // (op 10515) 

v_lshrrev_b32 v17, 2, v1 // (op 10515) 
v_lshl_add_u32 v20, v17, 7, v16 // (op 10515) 


v_lshlrev_b32 v16, 2, v20 // (op 10515) 

v_lshrrev_b32 v17, 3, v16 // (op 10515) 

s_nop 0
v_readfirstlane_b32 s47, v17 // (op 10515) Wait state hazard: VALU Write Readlane Hazard

v_and_b32 v16, -4, v1 // (op 10572) 
v_sub_u32 v17, v1, v16 // (op 10572) 

v_lshlrev_b32 v16, 5, v17 // (op 10572) 

v_lshrrev_b32 v17, 2, v1 // (op 10572) 
v_lshl_add_u32 v20, v17, 7, v16 // (op 10572) 


v_lshlrev_b32 v16, 2, v20 // (op 10572) 

v_lshrrev_b32 v17, 3, v16 // (op 10572) 

s_nop 0
v_readfirstlane_b32 s0, v17 // (op 10572) Wait state hazard: VALU Write Readlane Hazard

s_mul_hi_i32 s1, s16, s10 // (op 11013) 
v_and_b32 v16, 63, v1 // (op 11013) 
s_add_i32 s18, s1, s16 // (op 11013) 

v_lshrrev_b32 v17, 4, v16 // (op 11013) 
s_ashr_i32 s1, s18, 31 // (op 11013) 
v_and_b32 v20, 3, v17 // (op 11013) 

s_cmp_eq_i32 s10, 0 // (op 11013) 
s_mov_b32 s19, scc // (op 11013) 
v_and_b32 v17, 15, v1 // (op 11013) 
s_cmp_eq_i32 s19, 0 // (op 11013) 
s_cselect_b32 s22, 0, -1 // (op 11013) 

v_lshl_add_u32 v21, v20, 4, v17 // (op 11013) 


s_and_b32 s19, s13, 31 // (op 11013) 
v_lshrrev_b32 v17, 7, v1 // (op 11013) 
s_lshl_b32 s23, 1, s19 // (op 11013) 
v_lshl_add_u32 v20, v17, 6, v21 // (op 11013) 


s_add_i32 s24, s23, s22 // (op 11013) 


v_lshrrev_b32 v17, 6, v16 // (op 11013) 

s_and_b32 s25, s1, s24 // (op 11013) 


v_lshlrev_b32 v16, 2, v17 // (op 11013) 

s_add_i32 s1, s18, s25 // (op 11013) 


s_ashr_i32 s18, s1, s19 // (op 11013) 


s_lshl_b32 s1, s13, 24 // (op 11013) 
s_ashr_i32 s19, s1, 31 // (op 11013) 

s_xor_b32 s1, s18, s19 // (op 11013) 

s_sub_i32 s18, s1, s19 // (op 11013) 


s_mul_hi_i32 s1, s18, s14 // (op 11013) 
s_add_i32 s19, s1, s18 // (op 11013) 

s_ashr_i32 s1, s19, 31 // (op 11013) 
s_cmp_eq_i32 s14, 0 // (op 11013) 
s_mov_b32 s22, scc // (op 11013) 
s_cmp_eq_i32 s22, 0 // (op 11013) 
s_cselect_b32 s23, 0, -1 // (op 11013) 

s_and_b32 s22, s15, 31 // (op 11013) 
s_lshl_b32 s24, 1, s22 // (op 11013) 
s_add_i32 s25, s24, s23 // (op 11013) 


s_and_b32 s23, s1, s25 // (op 11013) 


s_add_i32 s1, s19, s23 // (op 11013) 


s_ashr_i32 s19, s1, s22 // (op 11013) 


s_lshl_b32 s1, s15, 24 // (op 11013) 
s_ashr_i32 s22, s1, 31 // (op 11013) 

s_xor_b32 s1, s19, s22 // (op 11013) 

s_sub_i32 s19, s1, s22 // (op 11013) 


s_mul_i32 s1, s18, s11 // (op 11013) 

s_sub_u32 s18, s16, s1 // (op 11013) 

s_mul_i32 s1, s19, s11 // (op 11013) 
s_add_u32 s1, s1, s18 // (op 11013) 


s_mul_hi_i32 s18, s16, s9 // (op 11013) 
s_add_i32 s19, s18, s16 // (op 11013) 

s_ashr_i32 s18, s19, 31 // (op 11013) 
s_cmp_eq_i32 s9, 0 // (op 11013) 
s_mov_b32 s22, scc // (op 11013) 
s_cmp_eq_i32 s22, 0 // (op 11013) 
s_cselect_b32 s23, 0, -1 // (op 11013) 

s_and_b32 s22, s45, 31 // (op 11013) 
s_lshl_b32 s24, 1, s22 // (op 11013) 
s_add_i32 s25, s24, s23 // (op 11013) 


s_and_b32 s23, s18, s25 // (op 11013) 


s_add_i32 s18, s19, s23 // (op 11013) 


s_ashr_i32 s19, s18, s22 // (op 11013) 


s_lshl_b32 s18, s45, 24 // (op 11013) 
s_ashr_i32 s22, s18, 31 // (op 11013) 

s_xor_b32 s18, s19, s22 // (op 11013) 

s_sub_i32 s19, s18, s22 // (op 11013) 


s_add_u32 s22, s6, 128 // (op 11013) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 11013) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 11013) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 11013) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 11013) 

s_mul_i32 s18, s11, s22 // (op 11013) 

s_mul_i32 s22, s8, s18 // (op 11013) 

s_mul_i32 s18, s19, s22 // (op 11013) 

s_sub_u32 s22, s16, s18 // (op 11013) 

s_mov_b32 s18, s22 // (op 11013) 
s_mul_hi_i32 s23, s18, s12 // (op 11013) 
s_add_i32 s24, s23, s18 // (op 11013) 


s_ashr_i32 s18, s24, 31 // (op 11013) 
s_cmp_eq_i32 s12, 0 // (op 11013) 
s_mov_b32 s23, scc // (op 11013) 
s_cmp_eq_i32 s23, 0 // (op 11013) 
s_cselect_b32 s25, 0, -1 // (op 11013) 

s_and_b32 s23, s44, 31 // (op 11013) 
s_lshl_b32 s64, 1, s23 // (op 11013) 
s_add_i32 s65, s64, s25 // (op 11013) 


s_and_b32 s25, s18, s65 // (op 11013) 


s_add_i32 s18, s24, s25 // (op 11013) 


s_ashr_i32 s24, s18, s23 // (op 11013) 


s_lshl_b32 s18, s44, 24 // (op 11013) 
s_ashr_i32 s23, s18, 31 // (op 11013) 

s_xor_b32 s18, s24, s23 // (op 11013) 

s_sub_i32 s24, s18, s23 // (op 11013) 


s_mul_i32 s18, s46, s11 // (op 11013) 
s_add_u32 s64, s20, 128 // (op 11013) Start of Int64 add, locking SCC
s_addc_u32 s65, s21, 0 // (op 11013) End of Int64 add, unlocking SCC
s_sub_u32 s66, s64, 1 // (op 11013) least significant half; sets scc
s_subb_u32 s67, s65, 0 // (op 11013) most significant half; uses scc

s_ashr_i64 s[64:65], s[66:67], 7 // (op 11013) 

s_sub_i32 s23, s64, s18 // (op 11013) 

s_mul_i32 s25, s24, s23 // (op 11013) 


s_sub_u32 s23, s22, s25 // (op 11013) 


s_add_u32 s22, s23, s18 // (op 11013) 


s_cmp_eq_i32 s19, 0 // (op 11013) 
s_mov_b32 s18, scc // (op 11013) 

s_cmp_eq_i32 s18, 0 // (op 11013) 
s_cselect_b32 s19, s22, s1 // (op 11013) 



v_lshl_add_u32 v17, s19, 7, v20 // (op 11013) 


s_ashr_i64 s[18:19], s[42:43], 7 // (op 11013) 
s_lshr_b32 s1, s18, 2 // (op 11013) 

v_lshl_add_u32 v20, s1, 4, v16 // (op 11013) 


v_mul_lo_u32 v16, v17, s34 // (op 11013) 
v_add_u32 v16, v16, v20 // (op 11013) 


s_mul_i32 s19, s34, s21 // (op 11015) most significant: low of low * high
s_mul_i32 s1, s35, s20 // (op 11015) most significant: low of high * low
s_mul_i32 s18, s34, s20 // (op 11015) least significant
s_mul_hi_u32 s22, s34, s20 // (op 11015) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 11015) most significant: sum
s_add_u32 s19, s19, s22 // (op 11015) most significant: sum


s_mov_b32 s64, s36 // (op 11015) reg expression
s_mov_b32 s65, s37 // (op 11015) reg expression
s_mov_b32 s66, s18 // (op 11015) reg expression
s_mov_b32 s67, 131072 // (op 11015) call()

s_mul_hi_i32 s1, s16, s10 // (op 11040) 
v_and_b32 v17, 63, v1 // (op 11040) 
s_add_i32 s18, s1, s16 // (op 11040) 

v_lshrrev_b32 v20, 4, v17 // (op 11040) 
s_ashr_i32 s1, s18, 31 // (op 11040) 
v_and_b32 v21, 3, v20 // (op 11040) 

s_cmp_eq_i32 s10, 0 // (op 11040) 
s_mov_b32 s19, scc // (op 11040) 
v_and_b32 v20, 15, v1 // (op 11040) 
s_cmp_eq_i32 s19, 0 // (op 11040) 
s_cselect_b32 s22, 0, -1 // (op 11040) 

v_lshl_add_u32 v22, v21, 4, v20 // (op 11040) 


s_and_b32 s19, s13, 31 // (op 11040) 
v_lshrrev_b32 v20, 6, v1 // (op 11040) 
s_lshl_b32 s23, 1, s19 // (op 11040) 
v_and_b32 v21, 1, v20 // (op 11040) 

s_add_i32 s24, s23, s22 // (op 11040) 


v_lshl_add_u32 v20, v21, 6, v22 // (op 11040) 


s_and_b32 s25, s1, s24 // (op 11040) 


v_lshrrev_b32 v21, 6, v17 // (op 11040) 

s_add_i32 s1, s18, s25 // (op 11040) 


v_lshlrev_b32 v17, 2, v21 // (op 11040) 

s_ashr_i32 s18, s1, s19 // (op 11040) 


s_lshl_b32 s1, s13, 24 // (op 11040) 
s_ashr_i32 s19, s1, 31 // (op 11040) 

s_xor_b32 s1, s18, s19 // (op 11040) 

s_sub_i32 s18, s1, s19 // (op 11040) 


s_mul_hi_i32 s1, s18, s14 // (op 11040) 
s_add_i32 s19, s1, s18 // (op 11040) 

s_ashr_i32 s1, s19, 31 // (op 11040) 
s_cmp_eq_i32 s14, 0 // (op 11040) 
s_mov_b32 s22, scc // (op 11040) 
s_cmp_eq_i32 s22, 0 // (op 11040) 
s_cselect_b32 s23, 0, -1 // (op 11040) 

s_and_b32 s22, s15, 31 // (op 11040) 
s_lshl_b32 s24, 1, s22 // (op 11040) 
s_add_i32 s25, s24, s23 // (op 11040) 


s_and_b32 s23, s1, s25 // (op 11040) 


s_add_i32 s1, s19, s23 // (op 11040) 


s_ashr_i32 s19, s1, s22 // (op 11040) 


s_lshl_b32 s1, s15, 24 // (op 11040) 
s_ashr_i32 s22, s1, 31 // (op 11040) 

s_xor_b32 s1, s19, s22 // (op 11040) 

s_sub_i32 s19, s1, s22 // (op 11040) 


s_add_u32 s22, s6, 128 // (op 11040) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 11040) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 11040) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 11040) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 11040) 

s_mul_i32 s1, s19, s22 // (op 11040) 

s_sub_i32 s19, s18, s1 // (op 11040) 


s_mul_hi_i32 s1, s16, s9 // (op 11040) 
s_add_i32 s18, s1, s16 // (op 11040) 

s_ashr_i32 s1, s18, 31 // (op 11040) 
s_cmp_eq_i32 s9, 0 // (op 11040) 
s_mov_b32 s24, scc // (op 11040) 
s_cmp_eq_i32 s24, 0 // (op 11040) 
s_cselect_b32 s25, 0, -1 // (op 11040) 

s_and_b32 s24, s45, 31 // (op 11040) 
s_lshl_b32 s68, 1, s24 // (op 11040) 
s_add_i32 s69, s68, s25 // (op 11040) 


s_and_b32 s25, s1, s69 // (op 11040) 


s_add_i32 s1, s18, s25 // (op 11040) 


s_ashr_i32 s18, s1, s24 // (op 11040) 


s_lshl_b32 s1, s45, 24 // (op 11040) 
s_ashr_i32 s24, s1, 31 // (op 11040) 

s_xor_b32 s1, s18, s24 // (op 11040) 

s_sub_i32 s18, s1, s24 // (op 11040) 


s_mul_i32 s1, s11, s22 // (op 11040) 

s_mul_i32 s22, s8, s1 // (op 11040) 

s_mul_i32 s1, s18, s22 // (op 11040) 

s_sub_u32 s22, s16, s1 // (op 11040) 

s_mul_hi_i32 s1, s22, s12 // (op 11040) 
s_add_i32 s23, s1, s22 // (op 11040) 


s_ashr_i32 s1, s23, 31 // (op 11040) 
s_cmp_eq_i32 s12, 0 // (op 11040) 
s_mov_b32 s22, scc // (op 11040) 
s_cmp_eq_i32 s22, 0 // (op 11040) 
s_cselect_b32 s24, 0, -1 // (op 11040) 

s_and_b32 s22, s44, 31 // (op 11040) 
s_lshl_b32 s25, 1, s22 // (op 11040) 
s_add_i32 s68, s25, s24 // (op 11040) 


s_and_b32 s24, s1, s68 // (op 11040) 


s_add_i32 s1, s23, s24 // (op 11040) 


s_ashr_i32 s23, s1, s22 // (op 11040) 


s_lshl_b32 s1, s44, 24 // (op 11040) 
s_ashr_i32 s22, s1, 31 // (op 11040) 

s_xor_b32 s1, s23, s22 // (op 11040) 

s_sub_i32 s23, s1, s22 // (op 11040) 


s_cmp_eq_i32 s18, 0 // (op 11040) 
s_mov_b32 s1, scc // (op 11040) 

s_cmp_eq_i32 s1, 0 // (op 11040) 
s_cselect_b32 s18, s23, s19 // (op 11040) 



v_lshl_add_u32 v21, s18, 7, v20 // (op 11040) 


v_mul_lo_u32 v20, v21, s38 // (op 11040) 

s_ashr_i64 s[18:19], s[42:43], 7 // (op 11040) 
s_lshr_b32 s1, s18, 2 // (op 11040) 

v_lshl_add_u32 v21, s1, 4, v17 // (op 11040) 


v_add_u32 v17, v21, v20 // (op 11040) 


s_mul_i32 s19, s38, s7 // (op 11042) most significant: low of low * high
s_mul_i32 s1, s39, s6 // (op 11042) most significant: low of high * low
s_mul_i32 s18, s38, s6 // (op 11042) least significant
s_mul_hi_u32 s22, s38, s6 // (op 11042) most significant: high of low * low
s_add_u32 s19, s19, s1 // (op 11042) most significant: sum
s_add_u32 s19, s19, s22 // (op 11042) most significant: sum


s_mov_b32 s68, s40 // (op 11042) reg expression
s_mov_b32 s69, s41 // (op 11042) reg expression
s_mov_b32 s70, s18 // (op 11042) reg expression
s_mov_b32 s71, 131072 // (op 11042) call()

s_ashr_i64 s[18:19], s[42:43], 7 // (op 3293) 
s_and_b32 s1, s18, 4294967292 // (op 3293) 

s_ashr_i64 s[18:19], s[42:43], 7 // (op 3292) 
s_mov_b32 s22, s18 // (op 3292) convert

s_cmp_lt_u32 s1, s22 // (op 3292) 











s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopBottom_KLoopTail_3292 // (op 3292) Condition: Top (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopBottom_KLoopTail_3292 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopTop_KLoopTail_3292:
s_barrier  // (op 11264) 
s_add_u32 s1, s1, 1 // (op 3294) 

s_lshl_b64 s[18:19], s[26:27], 6 // (op 10260) 
s_lshl_b64 s[22:23], s[18:19], 2 // (op 10260) 

v_lshrrev_b64 v[20:21], 3, s[22:23] // (op 10260) 


v_mov_b64 v[22:23], v[14:15] // (op 10260) 
v_mov_b64 v[24:25], v[22:23] // (op 10260) 
s_add_u32 s18, 8192, s47 // (op 10260) 
s_mov_b32 m0, s18 // (op 10260) reg expression
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 10260) Load value direct to lds
v_add_co_u32 v24, s[22:23], v24, v20 // (op 10260) least significant half
v_addc_co_u32 v25, s[22:23], 0, v25, s[22:23] // (op 10260) most significant half

s_add_u32 m0, m0, 4096 // (op 10260) 
buffer_load_dwordx4 v24, s[48:51], 0 offen offset: 0 lds // (op 10260) Load value direct to lds




v_add_co_u32 v14, s[18:19], 64, v14 // (op 10325) least significant half
v_addc_co_u32 v15, s[18:19], 0, v15, s[18:19] // (op 10325) most significant half

s_barrier  // (op 11281) 

s_lshl_b64 s[18:19], s[30:31], 6 // (op 10266) 
s_lshl_b64 s[22:23], s[18:19], 2 // (op 10266) 

v_lshrrev_b64 v[20:21], 3, s[22:23] // (op 10266) 


v_mov_b64 v[22:23], v[18:19] // (op 10266) 
v_mov_b64 v[24:25], v[22:23] // (op 10266) 
s_mov_b32 m0, s0 // (op 10266) reg expression
buffer_load_dwordx4 v24, s[60:63], 0 offen offset: 0 lds // (op 10266) Load value direct to lds
v_add_co_u32 v24, s[18:19], v24, v20 // (op 10266) least significant half
v_addc_co_u32 v25, s[18:19], 0, v25, s[18:19] // (op 10266) most significant half

s_add_u32 m0, m0, 4096 // (op 10266) 
buffer_load_dwordx4 v24, s[60:63], 0 offen offset: 0 lds // (op 10266) Load value direct to lds



v_add_co_u32 v18, s[18:19], 64, v18 // (op 10409) least significant half
v_addc_co_u32 v19, s[18:19], 0, v19, s[18:19] // (op 10409) most significant half

s_waitcnt vmcnt(2)
s_barrier  // (op 11271) 



ds_read_b128 v[20:23], v3 offset:8192 // (op 164) Load local data 
s_waitcnt vmcnt(0)
s_barrier  // (op 11287) 



ds_read_b128 v[24:27], v4 // (op 166) Load local data 


v_mov_b32 v28, v16 // (op 168) 
v_mov_b32 v29, v28 // (op 168) reg expression

buffer_load_dwordx4 v[30:33], v29, s[64:67], 0 offen // (op 168) Load value


v_add_u32 v16, 4, v16 // (op 11003) 


v_mov_b32 v28, v17 // (op 170) 
v_mov_b32 v29, v28 // (op 170) reg expression

buffer_load_dwordx4 v[34:37], v29, s[68:71], 0 offen // (op 170) Load value


v_add_u32 v17, 4, v17 // (op 11030) 
s_waitcnt vmcnt(1)
v_bfe_u32 v38, v30, 0, 8 // (op 9478) 



v_bfe_u32 v39, v30, 8, 8 // (op 9478) 
v_bfe_u32 v40, v30, 16, 8 // (op 9478) 
v_bfe_u32 v41, v30, 24, 8 // (op 9478) 
v_permlane16_swap_b32 v38, v39 // (op 9478) 
ds_read_b128 v[42:45], v4 offset:1024 // (op 401) Load local data 
v_permlane16_swap_b32 v40, v41 // (op 9478) 



s_nop 1
v_permlane32_swap_b32 v38, v40 // (op 9478) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v39, v41 // (op 9478) 
ds_read_b128 v[46:49], v4 offset:2048 // (op 627) Load local data 




ds_read_b128 v[28:31], v4 offset:3072 // (op 853) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v50, v34, 0, 8 // (op 9508) 



v_bfe_u32 v51, v34, 8, 8 // (op 9508) 
v_bfe_u32 v52, v34, 16, 8 // (op 9508) 
v_bfe_u32 v53, v34, 24, 8 // (op 9508) 
v_permlane16_swap_b32 v50, v51 // (op 9508) 
s_nop 0
v_permlane16_swap_b32 v52, v53 // (op 9508) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v50, v52 // (op 9508) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v51, v53 // (op 9508) 
ds_read_b128 v[54:57], v3 offset:9216 // (op 1213) Load local data 




s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[20:23], v[24:27], a[0:3], v38, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 171) 
ds_read_b128 v[32:35], v3 offset:10240 // (op 2215) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[20:23], v[42:45], a[4:7], v38, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 404) 
ds_read_b128 v[58:61], v3 offset:11264 // (op 3217) Load local data 
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[20:23], v[46:49], a[8:11], v38, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 630) 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[20:23], v[28:31], a[12:15], v38, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 856) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[54:57], v[24:27], a[16:19], v39, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1217) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[54:57], v[46:49], a[24:27], v39, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1271) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[54:57], v[28:31], a[28:31], v39, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1298) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[54:57], v[42:45], a[20:23], v39, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 1244) 

s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[32:35], v[24:27], a[32:35], v40, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2219) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[32:35], v[46:49], a[40:43], v40, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2273) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[32:35], v[28:31], a[44:47], v40, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2300) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[32:35], v[42:45], a[36:39], v40, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 2246) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[58:61], v[24:27], a[48:51], v41, v50 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3221) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[58:61], v[46:49], a[56:59], v41, v52 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3275) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[58:61], v[28:31], a[60:63], v41, v53 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3302) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[58:61], v[42:45], a[52:55], v41, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b100 blgp:0b100 // (op 3248) 




s_ashr_i64 s[18:19], s[42:43], 7 // (op 3292) 
s_mov_b32 s22, s18 // (op 3292) convert

s_cmp_lt_u32 s1, s22 // (op 3292) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopTop_KLoopTail_3292 // (op 3292) Condition: Bottom (jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopTop_KLoopTail_3292 if true)
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ForLoopBottom_KLoopTail_3292:




















s_barrier  // (op 11406) 





















v_cmp_eq_f32 vcc, 0.00000, v0 // (op 116) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ConditionalFalse_0_beta_116 // (op 116) Condition: False, jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ConditionalFalse_0_beta_116
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


v_pack_b32_f16 v8, v4, v5 // (op 50) 
v_pack_b32_f16 v9, v6, v7 // (op 50) 

v_mov_b32 v3, v13 // (op 50) reg expression

buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 50) Store value



v_accvgpr_read v4, a4 // (op 4698) 
v_accvgpr_read v5, a5 // (op 4698) 
v_accvgpr_read v6, a6 // (op 4698) 
v_accvgpr_read v7, a7 // (op 4698) 
v_mul_f32 v14, v11, v4 // (op 4698) 
v_mul_f32 v15, v11, v5 // (op 4698) 
v_mul_f32 v16, v11, v6 // (op 4698) 
v_mul_f32 v17, v11, v7 // (op 4698) 

v_lshrrev_b32 v4, 16, v14 // (op 4700) 
v_lshrrev_b32 v5, 16, v15 // (op 4700) 
v_lshrrev_b32 v6, 16, v16 // (op 4700) 
v_lshrrev_b32 v7, 16, v17 // (op 4700) 


v_pack_b32_f16 v8, v4, v5 // (op 4701) 
v_pack_b32_f16 v9, v6, v7 // (op 4701) 

s_lshl_b32 s0, s4, 5 // (op 4701) 
v_add_u32 v3, v13, s0 // (op 4701) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 4701) Store value



v_accvgpr_read v4, a8 // (op 4712) 
v_accvgpr_read v5, a9 // (op 4712) 
v_accvgpr_read v6, a10 // (op 4712) 
v_accvgpr_read v7, a11 // (op 4712) 
v_mul_f32 v14, v11, v4 // (op 4712) 
v_mul_f32 v15, v11, v5 // (op 4712) 
v_mul_f32 v16, v11, v6 // (op 4712) 
v_mul_f32 v17, v11, v7 // (op 4712) 

v_lshrrev_b32 v4, 16, v14 // (op 4714) 
v_lshrrev_b32 v5, 16, v15 // (op 4714) 
v_lshrrev_b32 v6, 16, v16 // (op 4714) 
v_lshrrev_b32 v7, 16, v17 // (op 4714) 


v_pack_b32_f16 v8, v4, v5 // (op 4715) 
v_pack_b32_f16 v9, v6, v7 // (op 4715) 

s_lshl_b32 s0, s4, 6 // (op 4715) 
v_add_u32 v3, v13, s0 // (op 4715) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 4715) Store value



v_accvgpr_read v4, a12 // (op 4726) 
v_accvgpr_read v5, a13 // (op 4726) 
v_accvgpr_read v6, a14 // (op 4726) 
v_accvgpr_read v7, a15 // (op 4726) 
v_mul_f32 v14, v11, v4 // (op 4726) 
v_mul_f32 v15, v11, v5 // (op 4726) 
v_mul_f32 v16, v11, v6 // (op 4726) 
v_mul_f32 v17, v11, v7 // (op 4726) 

v_lshrrev_b32 v4, 16, v14 // (op 4728) 
v_lshrrev_b32 v5, 16, v15 // (op 4728) 
v_lshrrev_b32 v6, 16, v16 // (op 4728) 
v_lshrrev_b32 v7, 16, v17 // (op 4728) 


v_pack_b32_f16 v8, v4, v5 // (op 4729) 
v_pack_b32_f16 v9, v6, v7 // (op 4729) 

s_lshl_b32 s0, s4, 5 // (op 4729) 
s_mul_i32 s1, 3, s0 // (op 4729) 

v_add_u32 v3, v13, s1 // (op 4729) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 4729) Store value



v_accvgpr_read v4, a16 // (op 4762) 
v_accvgpr_read v5, a17 // (op 4762) 
v_accvgpr_read v6, a18 // (op 4762) 
v_accvgpr_read v7, a19 // (op 4762) 
v_mul_f32 v14, v11, v4 // (op 4762) 
v_mul_f32 v15, v11, v5 // (op 4762) 
v_mul_f32 v16, v11, v6 // (op 4762) 
v_mul_f32 v17, v11, v7 // (op 4762) 

v_lshrrev_b32 v4, 16, v14 // (op 4764) 
v_lshrrev_b32 v5, 16, v15 // (op 4764) 
v_lshrrev_b32 v6, 16, v16 // (op 4764) 
v_lshrrev_b32 v7, 16, v17 // (op 4764) 


v_pack_b32_f16 v8, v4, v5 // (op 4766) 
v_pack_b32_f16 v9, v6, v7 // (op 4766) 

v_add_u32 v3, 32, v13 // (op 4766) 

buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 4766) Store value



v_accvgpr_read v4, a20 // (op 4776) 
v_accvgpr_read v5, a21 // (op 4776) 
v_accvgpr_read v6, a22 // (op 4776) 
v_accvgpr_read v7, a23 // (op 4776) 
v_mul_f32 v14, v11, v4 // (op 4776) 
v_mul_f32 v15, v11, v5 // (op 4776) 
v_mul_f32 v16, v11, v6 // (op 4776) 
v_mul_f32 v17, v11, v7 // (op 4776) 

v_lshrrev_b32 v4, 16, v14 // (op 4778) 
v_lshrrev_b32 v5, 16, v15 // (op 4778) 
v_lshrrev_b32 v6, 16, v16 // (op 4778) 
v_lshrrev_b32 v7, 16, v17 // (op 4778) 


v_pack_b32_f16 v8, v4, v5 // (op 4768) 
v_pack_b32_f16 v9, v6, v7 // (op 4768) 

s_lshl_b32 s0, s4, 5 // (op 4768) 
s_add_u32 s0, s0, 32 // (op 4768) 
v_add_u32 v3, v13, s0 // (op 4768) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 4768) Store value



v_accvgpr_read v4, a24 // (op 4782) 
v_accvgpr_read v5, a25 // (op 4782) 
v_accvgpr_read v6, a26 // (op 4782) 
v_accvgpr_read v7, a27 // (op 4782) 
v_mul_f32 v14, v11, v4 // (op 4782) 
v_mul_f32 v15, v11, v5 // (op 4782) 
v_mul_f32 v16, v11, v6 // (op 4782) 
v_mul_f32 v17, v11, v7 // (op 4782) 

v_lshrrev_b32 v4, 16, v14 // (op 4784) 
v_lshrrev_b32 v5, 16, v15 // (op 4784) 
v_lshrrev_b32 v6, 16, v16 // (op 4784) 
v_lshrrev_b32 v7, 16, v17 // (op 4784) 


v_pack_b32_f16 v8, v4, v5 // (op 4770) 
v_pack_b32_f16 v9, v6, v7 // (op 4770) 

s_lshl_b32 s0, s4, 6 // (op 4770) 
s_add_u32 s0, s0, 32 // (op 4770) 
v_add_u32 v3, v13, s0 // (op 4770) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 4770) Store value



v_accvgpr_read v4, a28 // (op 4788) 
v_accvgpr_read v5, a29 // (op 4788) 
v_accvgpr_read v6, a30 // (op 4788) 
v_accvgpr_read v7, a31 // (op 4788) 
v_mul_f32 v14, v11, v4 // (op 4788) 
v_mul_f32 v15, v11, v5 // (op 4788) 
v_mul_f32 v16, v11, v6 // (op 4788) 
v_mul_f32 v17, v11, v7 // (op 4788) 

v_lshrrev_b32 v4, 16, v14 // (op 4790) 
v_lshrrev_b32 v5, 16, v15 // (op 4790) 
v_lshrrev_b32 v6, 16, v16 // (op 4790) 
v_lshrrev_b32 v7, 16, v17 // (op 4790) 


v_pack_b32_f16 v8, v4, v5 // (op 4772) 
v_pack_b32_f16 v9, v6, v7 // (op 4772) 

s_lshl_b32 s0, s4, 5 // (op 4772) 
s_mul_i32 s1, 3, s0 // (op 4772) 
s_add_u32 s1, s1, 32 // (op 4772) 

v_add_u32 v3, v13, s1 // (op 4772) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 4772) Store value



v_accvgpr_read v4, a32 // (op 4838) 
v_accvgpr_read v5, a33 // (op 4838) 
v_accvgpr_read v6, a34 // (op 4838) 
v_accvgpr_read v7, a35 // (op 4838) 
v_mul_f32 v14, v11, v4 // (op 4838) 
v_mul_f32 v15, v11, v5 // (op 4838) 
v_mul_f32 v16, v11, v6 // (op 4838) 
v_mul_f32 v17, v11, v7 // (op 4838) 

v_lshrrev_b32 v4, 16, v14 // (op 4840) 
v_lshrrev_b32 v5, 16, v15 // (op 4840) 
v_lshrrev_b32 v6, 16, v16 // (op 4840) 
v_lshrrev_b32 v7, 16, v17 // (op 4840) 


v_pack_b32_f16 v8, v4, v5 // (op 4842) 
v_pack_b32_f16 v9, v6, v7 // (op 4842) 

v_add_u32 v3, 64, v13 // (op 4842) 

buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 4842) Store value



v_accvgpr_read v4, a36 // (op 4852) 
v_accvgpr_read v5, a37 // (op 4852) 
v_accvgpr_read v6, a38 // (op 4852) 
v_accvgpr_read v7, a39 // (op 4852) 
v_mul_f32 v14, v11, v4 // (op 4852) 
v_mul_f32 v15, v11, v5 // (op 4852) 
v_mul_f32 v16, v11, v6 // (op 4852) 
v_mul_f32 v17, v11, v7 // (op 4852) 

v_lshrrev_b32 v4, 16, v14 // (op 4854) 
v_lshrrev_b32 v5, 16, v15 // (op 4854) 
v_lshrrev_b32 v6, 16, v16 // (op 4854) 
v_lshrrev_b32 v7, 16, v17 // (op 4854) 


v_pack_b32_f16 v8, v4, v5 // (op 4844) 
v_pack_b32_f16 v9, v6, v7 // (op 4844) 

s_lshl_b32 s0, s4, 5 // (op 4844) 
s_add_u32 s0, s0, 64 // (op 4844) 
v_add_u32 v3, v13, s0 // (op 4844) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 4844) Store value



v_accvgpr_read v4, a40 // (op 4858) 
v_accvgpr_read v5, a41 // (op 4858) 
v_accvgpr_read v6, a42 // (op 4858) 
v_accvgpr_read v7, a43 // (op 4858) 
v_mul_f32 v14, v11, v4 // (op 4858) 
v_mul_f32 v15, v11, v5 // (op 4858) 
v_mul_f32 v16, v11, v6 // (op 4858) 
v_mul_f32 v17, v11, v7 // (op 4858) 

v_lshrrev_b32 v4, 16, v14 // (op 4860) 
v_lshrrev_b32 v5, 16, v15 // (op 4860) 
v_lshrrev_b32 v6, 16, v16 // (op 4860) 
v_lshrrev_b32 v7, 16, v17 // (op 4860) 


v_pack_b32_f16 v8, v4, v5 // (op 4846) 
v_pack_b32_f16 v9, v6, v7 // (op 4846) 

s_lshl_b32 s0, s4, 6 // (op 4846) 
s_add_u32 s0, s0, 64 // (op 4846) 
v_add_u32 v3, v13, s0 // (op 4846) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 4846) Store value



v_accvgpr_read v4, a44 // (op 4864) 
v_accvgpr_read v5, a45 // (op 4864) 
v_accvgpr_read v6, a46 // (op 4864) 
v_accvgpr_read v7, a47 // (op 4864) 
v_mul_f32 v14, v11, v4 // (op 4864) 
v_mul_f32 v15, v11, v5 // (op 4864) 
v_mul_f32 v16, v11, v6 // (op 4864) 
v_mul_f32 v17, v11, v7 // (op 4864) 

v_lshrrev_b32 v4, 16, v14 // (op 4866) 
v_lshrrev_b32 v5, 16, v15 // (op 4866) 
v_lshrrev_b32 v6, 16, v16 // (op 4866) 
v_lshrrev_b32 v7, 16, v17 // (op 4866) 


v_pack_b32_f16 v8, v4, v5 // (op 4848) 
v_pack_b32_f16 v9, v6, v7 // (op 4848) 

s_lshl_b32 s0, s4, 5 // (op 4848) 
s_mul_i32 s1, 3, s0 // (op 4848) 
s_add_u32 s1, s1, 64 // (op 4848) 

v_add_u32 v3, v13, s1 // (op 4848) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 4848) Store value



v_accvgpr_read v4, a48 // (op 4914) 
v_accvgpr_read v5, a49 // (op 4914) 
v_accvgpr_read v6, a50 // (op 4914) 
v_accvgpr_read v7, a51 // (op 4914) 
v_mul_f32 v14, v11, v4 // (op 4914) 
v_mul_f32 v15, v11, v5 // (op 4914) 
v_mul_f32 v16, v11, v6 // (op 4914) 
v_mul_f32 v17, v11, v7 // (op 4914) 

v_lshrrev_b32 v4, 16, v14 // (op 4916) 
v_lshrrev_b32 v5, 16, v15 // (op 4916) 
v_lshrrev_b32 v6, 16, v16 // (op 4916) 
v_lshrrev_b32 v7, 16, v17 // (op 4916) 


v_pack_b32_f16 v8, v4, v5 // (op 4918) 
v_pack_b32_f16 v9, v6, v7 // (op 4918) 

v_add_u32 v3, 96, v13 // (op 4918) 

buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 4918) Store value



v_accvgpr_read v4, a52 // (op 4928) 
v_accvgpr_read v5, a53 // (op 4928) 
v_accvgpr_read v6, a54 // (op 4928) 
v_accvgpr_read v7, a55 // (op 4928) 
v_mul_f32 v14, v11, v4 // (op 4928) 
v_mul_f32 v15, v11, v5 // (op 4928) 
v_mul_f32 v16, v11, v6 // (op 4928) 
v_mul_f32 v17, v11, v7 // (op 4928) 

v_lshrrev_b32 v4, 16, v14 // (op 4930) 
v_lshrrev_b32 v5, 16, v15 // (op 4930) 
v_lshrrev_b32 v6, 16, v16 // (op 4930) 
v_lshrrev_b32 v7, 16, v17 // (op 4930) 


v_pack_b32_f16 v8, v4, v5 // (op 4920) 
v_pack_b32_f16 v9, v6, v7 // (op 4920) 

s_lshl_b32 s0, s4, 5 // (op 4920) 
s_add_u32 s0, s0, 96 // (op 4920) 
v_add_u32 v3, v13, s0 // (op 4920) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 4920) Store value



v_accvgpr_read v4, a56 // (op 4934) 
v_accvgpr_read v5, a57 // (op 4934) 
v_accvgpr_read v6, a58 // (op 4934) 
v_accvgpr_read v7, a59 // (op 4934) 
v_mul_f32 v14, v11, v4 // (op 4934) 
v_mul_f32 v15, v11, v5 // (op 4934) 
v_mul_f32 v16, v11, v6 // (op 4934) 
v_mul_f32 v17, v11, v7 // (op 4934) 

v_lshrrev_b32 v4, 16, v14 // (op 4936) 
v_lshrrev_b32 v5, 16, v15 // (op 4936) 
v_lshrrev_b32 v6, 16, v16 // (op 4936) 
v_lshrrev_b32 v7, 16, v17 // (op 4936) 


v_pack_b32_f16 v8, v4, v5 // (op 4922) 
v_pack_b32_f16 v9, v6, v7 // (op 4922) 

s_lshl_b32 s0, s4, 6 // (op 4922) 
s_add_u32 s0, s0, 96 // (op 4922) 
v_add_u32 v3, v13, s0 // (op 4922) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 4922) Store value



v_accvgpr_read v4, a60 // (op 4940) 
v_accvgpr_read v5, a61 // (op 4940) 
v_accvgpr_read v6, a62 // (op 4940) 
v_accvgpr_read v7, a63 // (op 4940) 
v_mul_f32 v14, v11, v4 // (op 4940) 
v_mul_f32 v15, v11, v5 // (op 4940) 
v_mul_f32 v16, v11, v6 // (op 4940) 
v_mul_f32 v17, v11, v7 // (op 4940) 

v_lshrrev_b32 v4, 16, v14 // (op 4942) 
v_lshrrev_b32 v5, 16, v15 // (op 4942) 
v_lshrrev_b32 v6, 16, v16 // (op 4942) 
v_lshrrev_b32 v7, 16, v17 // (op 4942) 


v_pack_b32_f16 v8, v4, v5 // (op 4924) 
v_pack_b32_f16 v9, v6, v7 // (op 4924) 

s_lshl_b32 s0, s4, 5 // (op 4924) 
s_mul_i32 s1, 3, s0 // (op 4924) 
s_add_u32 s1, s1, 96 // (op 4924) 

v_add_u32 v3, v13, s1 // (op 4924) 


buffer_store_dwordx2 v[8:9], v3, s[56:59], 0 offen // (op 4924) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_branch RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ConditionalBottom_0_beta_116 // (op 116) Condition: Done, jump to RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ConditionalBottom_0_beta_116
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ConditionalFalse_0_beta_116:


v_add_u32 v3, 64, v12 // (op 5225) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 5225) Load value

v_accvgpr_read v6, a32 // (op 5212) 
v_accvgpr_read v7, a33 // (op 5212) 
v_accvgpr_read v8, a34 // (op 5212) 
v_accvgpr_read v9, a35 // (op 5212) 
v_mul_f32 v14, v11, v6 // (op 5212) 
v_mul_f32 v15, v11, v7 // (op 5212) 
v_mul_f32 v16, v11, v8 // (op 5212) 
v_mul_f32 v17, v11, v9 // (op 5212) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 5224) 
v_lshrrev_b32 v7, 16, v4 // (op 5224) 
v_lshlrev_b32 v6, 16, v6 // (op 5224) 
v_lshlrev_b32 v7, 16, v7 // (op 5224) 
v_mul_f32 v18, v0, v6 // (op 5224) 
v_mul_f32 v19, v0, v7 // (op 5224) 
v_and_b32 v6, 65535, v5 // (op 5224) 
v_lshrrev_b32 v7, 16, v5 // (op 5224) 
v_lshlrev_b32 v6, 16, v6 // (op 5224) 
v_lshlrev_b32 v7, 16, v7 // (op 5224) 
v_mul_f32 v20, v0, v6 // (op 5224) 
v_mul_f32 v21, v0, v7 // (op 5224) 


v_add_f32 v4, v18, v14 // (op 5213) 
v_add_f32 v5, v19, v15 // (op 5213) 
v_add_f32 v6, v20, v16 // (op 5213) 
v_add_f32 v7, v21, v17 // (op 5213) 


v_lshrrev_b32 v14, 16, v4 // (op 5214) 
v_lshrrev_b32 v15, 16, v5 // (op 5214) 
v_lshrrev_b32 v16, 16, v6 // (op 5214) 
v_lshrrev_b32 v17, 16, v7 // (op 5214) 


v_pack_b32_f16 v4, v14, v15 // (op 5216) 
v_pack_b32_f16 v5, v16, v17 // (op 5216) 

v_add_u32 v3, 64, v13 // (op 5216) 

buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5216) Store value





s_lshl_b32 s0, s2, 5 // (op 5230) 
s_add_u32 s0, s0, 64 // (op 5230) 
v_add_u32 v3, v12, s0 // (op 5230) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 5230) Load value

v_accvgpr_read v6, a36 // (op 5242) 
v_accvgpr_read v7, a37 // (op 5242) 
v_accvgpr_read v8, a38 // (op 5242) 
v_accvgpr_read v9, a39 // (op 5242) 
v_mul_f32 v14, v11, v6 // (op 5242) 
v_mul_f32 v15, v11, v7 // (op 5242) 
v_mul_f32 v16, v11, v8 // (op 5242) 
v_mul_f32 v17, v11, v9 // (op 5242) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 5227) 
v_lshrrev_b32 v7, 16, v4 // (op 5227) 
v_lshlrev_b32 v6, 16, v6 // (op 5227) 
v_lshlrev_b32 v7, 16, v7 // (op 5227) 
v_mul_f32 v18, v0, v6 // (op 5227) 
v_mul_f32 v19, v0, v7 // (op 5227) 
v_and_b32 v6, 65535, v5 // (op 5227) 
v_lshrrev_b32 v7, 16, v5 // (op 5227) 
v_lshlrev_b32 v6, 16, v6 // (op 5227) 
v_lshlrev_b32 v7, 16, v7 // (op 5227) 
v_mul_f32 v20, v0, v6 // (op 5227) 
v_mul_f32 v21, v0, v7 // (op 5227) 


v_add_f32 v4, v18, v14 // (op 5228) 
v_add_f32 v5, v19, v15 // (op 5228) 
v_add_f32 v6, v20, v16 // (op 5228) 
v_add_f32 v7, v21, v17 // (op 5228) 


v_lshrrev_b32 v14, 16, v4 // (op 5229) 
v_lshrrev_b32 v15, 16, v5 // (op 5229) 
v_lshrrev_b32 v16, 16, v6 // (op 5229) 
v_lshrrev_b32 v17, 16, v7 // (op 5229) 


v_pack_b32_f16 v4, v14, v15 // (op 5218) 
v_pack_b32_f16 v5, v16, v17 // (op 5218) 

s_lshl_b32 s0, s4, 5 // (op 5218) 
s_add_u32 s0, s0, 64 // (op 5218) 
v_add_u32 v3, v13, s0 // (op 5218) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5218) Store value





s_lshl_b32 s0, s2, 6 // (op 5235) 
s_add_u32 s0, s0, 64 // (op 5235) 
v_add_u32 v3, v12, s0 // (op 5235) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 5235) Load value

v_accvgpr_read v6, a40 // (op 5244) 
v_accvgpr_read v7, a41 // (op 5244) 
v_accvgpr_read v8, a42 // (op 5244) 
v_accvgpr_read v9, a43 // (op 5244) 
v_mul_f32 v14, v11, v6 // (op 5244) 
v_mul_f32 v15, v11, v7 // (op 5244) 
v_mul_f32 v16, v11, v8 // (op 5244) 
v_mul_f32 v17, v11, v9 // (op 5244) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 5232) 
v_lshrrev_b32 v7, 16, v4 // (op 5232) 
v_lshlrev_b32 v6, 16, v6 // (op 5232) 
v_lshlrev_b32 v7, 16, v7 // (op 5232) 
v_mul_f32 v18, v0, v6 // (op 5232) 
v_mul_f32 v19, v0, v7 // (op 5232) 
v_and_b32 v6, 65535, v5 // (op 5232) 
v_lshrrev_b32 v7, 16, v5 // (op 5232) 
v_lshlrev_b32 v6, 16, v6 // (op 5232) 
v_lshlrev_b32 v7, 16, v7 // (op 5232) 
v_mul_f32 v20, v0, v6 // (op 5232) 
v_mul_f32 v21, v0, v7 // (op 5232) 


v_add_f32 v4, v18, v14 // (op 5233) 
v_add_f32 v5, v19, v15 // (op 5233) 
v_add_f32 v6, v20, v16 // (op 5233) 
v_add_f32 v7, v21, v17 // (op 5233) 


v_lshrrev_b32 v14, 16, v4 // (op 5234) 
v_lshrrev_b32 v15, 16, v5 // (op 5234) 
v_lshrrev_b32 v16, 16, v6 // (op 5234) 
v_lshrrev_b32 v17, 16, v7 // (op 5234) 


v_pack_b32_f16 v4, v14, v15 // (op 5220) 
v_pack_b32_f16 v5, v16, v17 // (op 5220) 

s_lshl_b32 s0, s4, 6 // (op 5220) 
s_add_u32 s0, s0, 64 // (op 5220) 
v_add_u32 v3, v13, s0 // (op 5220) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5220) Store value





s_lshl_b32 s0, s2, 5 // (op 5240) 
s_mul_i32 s1, 3, s0 // (op 5240) 
s_add_u32 s1, s1, 64 // (op 5240) 

v_add_u32 v3, v12, s1 // (op 5240) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 5240) Load value

v_accvgpr_read v6, a44 // (op 5246) 
v_accvgpr_read v7, a45 // (op 5246) 
v_accvgpr_read v8, a46 // (op 5246) 
v_accvgpr_read v9, a47 // (op 5246) 
v_mul_f32 v14, v11, v6 // (op 5246) 
v_mul_f32 v15, v11, v7 // (op 5246) 
v_mul_f32 v16, v11, v8 // (op 5246) 
v_mul_f32 v17, v11, v9 // (op 5246) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 5237) 
v_lshrrev_b32 v7, 16, v4 // (op 5237) 
v_lshlrev_b32 v6, 16, v6 // (op 5237) 
v_lshlrev_b32 v7, 16, v7 // (op 5237) 
v_mul_f32 v18, v0, v6 // (op 5237) 
v_mul_f32 v19, v0, v7 // (op 5237) 
v_and_b32 v6, 65535, v5 // (op 5237) 
v_lshrrev_b32 v7, 16, v5 // (op 5237) 
v_lshlrev_b32 v6, 16, v6 // (op 5237) 
v_lshlrev_b32 v7, 16, v7 // (op 5237) 
v_mul_f32 v20, v0, v6 // (op 5237) 
v_mul_f32 v21, v0, v7 // (op 5237) 


v_add_f32 v4, v18, v14 // (op 5238) 
v_add_f32 v5, v19, v15 // (op 5238) 
v_add_f32 v6, v20, v16 // (op 5238) 
v_add_f32 v7, v21, v17 // (op 5238) 


v_lshrrev_b32 v14, 16, v4 // (op 5239) 
v_lshrrev_b32 v15, 16, v5 // (op 5239) 
v_lshrrev_b32 v16, 16, v6 // (op 5239) 
v_lshrrev_b32 v17, 16, v7 // (op 5239) 


v_pack_b32_f16 v4, v14, v15 // (op 5222) 
v_pack_b32_f16 v5, v16, v17 // (op 5222) 

s_lshl_b32 s0, s4, 5 // (op 5222) 
s_mul_i32 s1, 3, s0 // (op 5222) 
s_add_u32 s1, s1, 64 // (op 5222) 

v_add_u32 v3, v13, s1 // (op 5222) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5222) Store value





v_add_u32 v3, 32, v12 // (op 5138) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 5138) Load value

v_accvgpr_read v6, a16 // (op 5125) 
v_accvgpr_read v7, a17 // (op 5125) 
v_accvgpr_read v8, a18 // (op 5125) 
v_accvgpr_read v9, a19 // (op 5125) 
v_mul_f32 v14, v11, v6 // (op 5125) 
v_mul_f32 v15, v11, v7 // (op 5125) 
v_mul_f32 v16, v11, v8 // (op 5125) 
v_mul_f32 v17, v11, v9 // (op 5125) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 5137) 
v_lshrrev_b32 v7, 16, v4 // (op 5137) 
v_lshlrev_b32 v6, 16, v6 // (op 5137) 
v_lshlrev_b32 v7, 16, v7 // (op 5137) 
v_mul_f32 v18, v0, v6 // (op 5137) 
v_mul_f32 v19, v0, v7 // (op 5137) 
v_and_b32 v6, 65535, v5 // (op 5137) 
v_lshrrev_b32 v7, 16, v5 // (op 5137) 
v_lshlrev_b32 v6, 16, v6 // (op 5137) 
v_lshlrev_b32 v7, 16, v7 // (op 5137) 
v_mul_f32 v20, v0, v6 // (op 5137) 
v_mul_f32 v21, v0, v7 // (op 5137) 


v_add_f32 v4, v18, v14 // (op 5126) 
v_add_f32 v5, v19, v15 // (op 5126) 
v_add_f32 v6, v20, v16 // (op 5126) 
v_add_f32 v7, v21, v17 // (op 5126) 


v_lshrrev_b32 v14, 16, v4 // (op 5127) 
v_lshrrev_b32 v15, 16, v5 // (op 5127) 
v_lshrrev_b32 v16, 16, v6 // (op 5127) 
v_lshrrev_b32 v17, 16, v7 // (op 5127) 


v_pack_b32_f16 v4, v14, v15 // (op 5129) 
v_pack_b32_f16 v5, v16, v17 // (op 5129) 

v_add_u32 v3, 32, v13 // (op 5129) 

buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5129) Store value





s_lshl_b32 s0, s2, 5 // (op 5143) 
s_add_u32 s0, s0, 32 // (op 5143) 
v_add_u32 v3, v12, s0 // (op 5143) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 5143) Load value

v_accvgpr_read v6, a20 // (op 5155) 
v_accvgpr_read v7, a21 // (op 5155) 
v_accvgpr_read v8, a22 // (op 5155) 
v_accvgpr_read v9, a23 // (op 5155) 
v_mul_f32 v14, v11, v6 // (op 5155) 
v_mul_f32 v15, v11, v7 // (op 5155) 
v_mul_f32 v16, v11, v8 // (op 5155) 
v_mul_f32 v17, v11, v9 // (op 5155) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 5140) 
v_lshrrev_b32 v7, 16, v4 // (op 5140) 
v_lshlrev_b32 v6, 16, v6 // (op 5140) 
v_lshlrev_b32 v7, 16, v7 // (op 5140) 
v_mul_f32 v18, v0, v6 // (op 5140) 
v_mul_f32 v19, v0, v7 // (op 5140) 
v_and_b32 v6, 65535, v5 // (op 5140) 
v_lshrrev_b32 v7, 16, v5 // (op 5140) 
v_lshlrev_b32 v6, 16, v6 // (op 5140) 
v_lshlrev_b32 v7, 16, v7 // (op 5140) 
v_mul_f32 v20, v0, v6 // (op 5140) 
v_mul_f32 v21, v0, v7 // (op 5140) 


v_add_f32 v4, v18, v14 // (op 5141) 
v_add_f32 v5, v19, v15 // (op 5141) 
v_add_f32 v6, v20, v16 // (op 5141) 
v_add_f32 v7, v21, v17 // (op 5141) 


v_lshrrev_b32 v14, 16, v4 // (op 5142) 
v_lshrrev_b32 v15, 16, v5 // (op 5142) 
v_lshrrev_b32 v16, 16, v6 // (op 5142) 
v_lshrrev_b32 v17, 16, v7 // (op 5142) 


v_pack_b32_f16 v4, v14, v15 // (op 5131) 
v_pack_b32_f16 v5, v16, v17 // (op 5131) 

s_lshl_b32 s0, s4, 5 // (op 5131) 
s_add_u32 s0, s0, 32 // (op 5131) 
v_add_u32 v3, v13, s0 // (op 5131) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5131) Store value





s_lshl_b32 s0, s2, 6 // (op 5148) 
s_add_u32 s0, s0, 32 // (op 5148) 
v_add_u32 v3, v12, s0 // (op 5148) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 5148) Load value

v_accvgpr_read v6, a24 // (op 5157) 
v_accvgpr_read v7, a25 // (op 5157) 
v_accvgpr_read v8, a26 // (op 5157) 
v_accvgpr_read v9, a27 // (op 5157) 
v_mul_f32 v14, v11, v6 // (op 5157) 
v_mul_f32 v15, v11, v7 // (op 5157) 
v_mul_f32 v16, v11, v8 // (op 5157) 
v_mul_f32 v17, v11, v9 // (op 5157) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 5145) 
v_lshrrev_b32 v7, 16, v4 // (op 5145) 
v_lshlrev_b32 v6, 16, v6 // (op 5145) 
v_lshlrev_b32 v7, 16, v7 // (op 5145) 
v_mul_f32 v18, v0, v6 // (op 5145) 
v_mul_f32 v19, v0, v7 // (op 5145) 
v_and_b32 v6, 65535, v5 // (op 5145) 
v_lshrrev_b32 v7, 16, v5 // (op 5145) 
v_lshlrev_b32 v6, 16, v6 // (op 5145) 
v_lshlrev_b32 v7, 16, v7 // (op 5145) 
v_mul_f32 v20, v0, v6 // (op 5145) 
v_mul_f32 v21, v0, v7 // (op 5145) 


v_add_f32 v4, v18, v14 // (op 5146) 
v_add_f32 v5, v19, v15 // (op 5146) 
v_add_f32 v6, v20, v16 // (op 5146) 
v_add_f32 v7, v21, v17 // (op 5146) 


v_lshrrev_b32 v14, 16, v4 // (op 5147) 
v_lshrrev_b32 v15, 16, v5 // (op 5147) 
v_lshrrev_b32 v16, 16, v6 // (op 5147) 
v_lshrrev_b32 v17, 16, v7 // (op 5147) 


v_pack_b32_f16 v4, v14, v15 // (op 5133) 
v_pack_b32_f16 v5, v16, v17 // (op 5133) 

s_lshl_b32 s0, s4, 6 // (op 5133) 
s_add_u32 s0, s0, 32 // (op 5133) 
v_add_u32 v3, v13, s0 // (op 5133) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5133) Store value





s_lshl_b32 s0, s2, 5 // (op 5153) 
s_mul_i32 s1, 3, s0 // (op 5153) 
s_add_u32 s1, s1, 32 // (op 5153) 

v_add_u32 v3, v12, s1 // (op 5153) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 5153) Load value

v_accvgpr_read v6, a28 // (op 5159) 
v_accvgpr_read v7, a29 // (op 5159) 
v_accvgpr_read v8, a30 // (op 5159) 
v_accvgpr_read v9, a31 // (op 5159) 
v_mul_f32 v14, v11, v6 // (op 5159) 
v_mul_f32 v15, v11, v7 // (op 5159) 
v_mul_f32 v16, v11, v8 // (op 5159) 
v_mul_f32 v17, v11, v9 // (op 5159) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 5150) 
v_lshrrev_b32 v7, 16, v4 // (op 5150) 
v_lshlrev_b32 v6, 16, v6 // (op 5150) 
v_lshlrev_b32 v7, 16, v7 // (op 5150) 
v_mul_f32 v18, v0, v6 // (op 5150) 
v_mul_f32 v19, v0, v7 // (op 5150) 
v_and_b32 v6, 65535, v5 // (op 5150) 
v_lshrrev_b32 v7, 16, v5 // (op 5150) 
v_lshlrev_b32 v6, 16, v6 // (op 5150) 
v_lshlrev_b32 v7, 16, v7 // (op 5150) 
v_mul_f32 v20, v0, v6 // (op 5150) 
v_mul_f32 v21, v0, v7 // (op 5150) 


v_add_f32 v4, v18, v14 // (op 5151) 
v_add_f32 v5, v19, v15 // (op 5151) 
v_add_f32 v6, v20, v16 // (op 5151) 
v_add_f32 v7, v21, v17 // (op 5151) 


v_lshrrev_b32 v14, 16, v4 // (op 5152) 
v_lshrrev_b32 v15, 16, v5 // (op 5152) 
v_lshrrev_b32 v16, 16, v6 // (op 5152) 
v_lshrrev_b32 v17, 16, v7 // (op 5152) 


v_pack_b32_f16 v4, v14, v15 // (op 5135) 
v_pack_b32_f16 v5, v16, v17 // (op 5135) 

s_lshl_b32 s0, s4, 5 // (op 5135) 
s_mul_i32 s1, 3, s0 // (op 5135) 
s_add_u32 s1, s1, 32 // (op 5135) 

v_add_u32 v3, v13, s1 // (op 5135) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5135) Store value





v_add_u32 v3, 96, v12 // (op 5312) 

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 5312) Load value

v_accvgpr_read v6, a48 // (op 5299) 
v_accvgpr_read v7, a49 // (op 5299) 
v_accvgpr_read v8, a50 // (op 5299) 
v_accvgpr_read v9, a51 // (op 5299) 
v_mul_f32 v14, v11, v6 // (op 5299) 
v_mul_f32 v15, v11, v7 // (op 5299) 
v_mul_f32 v16, v11, v8 // (op 5299) 
v_mul_f32 v17, v11, v9 // (op 5299) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 5311) 
v_lshrrev_b32 v7, 16, v4 // (op 5311) 
v_lshlrev_b32 v6, 16, v6 // (op 5311) 
v_lshlrev_b32 v7, 16, v7 // (op 5311) 
v_mul_f32 v18, v0, v6 // (op 5311) 
v_mul_f32 v19, v0, v7 // (op 5311) 
v_and_b32 v6, 65535, v5 // (op 5311) 
v_lshrrev_b32 v7, 16, v5 // (op 5311) 
v_lshlrev_b32 v6, 16, v6 // (op 5311) 
v_lshlrev_b32 v7, 16, v7 // (op 5311) 
v_mul_f32 v20, v0, v6 // (op 5311) 
v_mul_f32 v21, v0, v7 // (op 5311) 


v_add_f32 v4, v18, v14 // (op 5300) 
v_add_f32 v5, v19, v15 // (op 5300) 
v_add_f32 v6, v20, v16 // (op 5300) 
v_add_f32 v7, v21, v17 // (op 5300) 


v_lshrrev_b32 v14, 16, v4 // (op 5301) 
v_lshrrev_b32 v15, 16, v5 // (op 5301) 
v_lshrrev_b32 v16, 16, v6 // (op 5301) 
v_lshrrev_b32 v17, 16, v7 // (op 5301) 


v_pack_b32_f16 v4, v14, v15 // (op 5303) 
v_pack_b32_f16 v5, v16, v17 // (op 5303) 

v_add_u32 v3, 96, v13 // (op 5303) 

buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5303) Store value





s_lshl_b32 s0, s2, 5 // (op 5317) 
s_add_u32 s0, s0, 96 // (op 5317) 
v_add_u32 v3, v12, s0 // (op 5317) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 5317) Load value

v_accvgpr_read v6, a52 // (op 5329) 
v_accvgpr_read v7, a53 // (op 5329) 
v_accvgpr_read v8, a54 // (op 5329) 
v_accvgpr_read v9, a55 // (op 5329) 
v_mul_f32 v14, v11, v6 // (op 5329) 
v_mul_f32 v15, v11, v7 // (op 5329) 
v_mul_f32 v16, v11, v8 // (op 5329) 
v_mul_f32 v17, v11, v9 // (op 5329) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 5314) 
v_lshrrev_b32 v7, 16, v4 // (op 5314) 
v_lshlrev_b32 v6, 16, v6 // (op 5314) 
v_lshlrev_b32 v7, 16, v7 // (op 5314) 
v_mul_f32 v18, v0, v6 // (op 5314) 
v_mul_f32 v19, v0, v7 // (op 5314) 
v_and_b32 v6, 65535, v5 // (op 5314) 
v_lshrrev_b32 v7, 16, v5 // (op 5314) 
v_lshlrev_b32 v6, 16, v6 // (op 5314) 
v_lshlrev_b32 v7, 16, v7 // (op 5314) 
v_mul_f32 v20, v0, v6 // (op 5314) 
v_mul_f32 v21, v0, v7 // (op 5314) 


v_add_f32 v4, v18, v14 // (op 5315) 
v_add_f32 v5, v19, v15 // (op 5315) 
v_add_f32 v6, v20, v16 // (op 5315) 
v_add_f32 v7, v21, v17 // (op 5315) 


v_lshrrev_b32 v14, 16, v4 // (op 5316) 
v_lshrrev_b32 v15, 16, v5 // (op 5316) 
v_lshrrev_b32 v16, 16, v6 // (op 5316) 
v_lshrrev_b32 v17, 16, v7 // (op 5316) 


v_pack_b32_f16 v4, v14, v15 // (op 5305) 
v_pack_b32_f16 v5, v16, v17 // (op 5305) 

s_lshl_b32 s0, s4, 5 // (op 5305) 
s_add_u32 s0, s0, 96 // (op 5305) 
v_add_u32 v3, v13, s0 // (op 5305) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5305) Store value





s_lshl_b32 s0, s2, 6 // (op 5322) 
s_add_u32 s0, s0, 96 // (op 5322) 
v_add_u32 v3, v12, s0 // (op 5322) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 5322) Load value

v_accvgpr_read v6, a56 // (op 5331) 
v_accvgpr_read v7, a57 // (op 5331) 
v_accvgpr_read v8, a58 // (op 5331) 
v_accvgpr_read v9, a59 // (op 5331) 
v_mul_f32 v14, v11, v6 // (op 5331) 
v_mul_f32 v15, v11, v7 // (op 5331) 
v_mul_f32 v16, v11, v8 // (op 5331) 
v_mul_f32 v17, v11, v9 // (op 5331) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 5319) 
v_lshrrev_b32 v7, 16, v4 // (op 5319) 
v_lshlrev_b32 v6, 16, v6 // (op 5319) 
v_lshlrev_b32 v7, 16, v7 // (op 5319) 
v_mul_f32 v18, v0, v6 // (op 5319) 
v_mul_f32 v19, v0, v7 // (op 5319) 
v_and_b32 v6, 65535, v5 // (op 5319) 
v_lshrrev_b32 v7, 16, v5 // (op 5319) 
v_lshlrev_b32 v6, 16, v6 // (op 5319) 
v_lshlrev_b32 v7, 16, v7 // (op 5319) 
v_mul_f32 v20, v0, v6 // (op 5319) 
v_mul_f32 v21, v0, v7 // (op 5319) 


v_add_f32 v4, v18, v14 // (op 5320) 
v_add_f32 v5, v19, v15 // (op 5320) 
v_add_f32 v6, v20, v16 // (op 5320) 
v_add_f32 v7, v21, v17 // (op 5320) 


v_lshrrev_b32 v14, 16, v4 // (op 5321) 
v_lshrrev_b32 v15, 16, v5 // (op 5321) 
v_lshrrev_b32 v16, 16, v6 // (op 5321) 
v_lshrrev_b32 v17, 16, v7 // (op 5321) 


v_pack_b32_f16 v4, v14, v15 // (op 5307) 
v_pack_b32_f16 v5, v16, v17 // (op 5307) 

s_lshl_b32 s0, s4, 6 // (op 5307) 
s_add_u32 s0, s0, 96 // (op 5307) 
v_add_u32 v3, v13, s0 // (op 5307) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5307) Store value





s_lshl_b32 s0, s2, 5 // (op 5327) 
s_mul_i32 s1, 3, s0 // (op 5327) 
s_add_u32 s1, s1, 96 // (op 5327) 

v_add_u32 v3, v12, s1 // (op 5327) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 5327) Load value

v_accvgpr_read v6, a60 // (op 5333) 
v_accvgpr_read v7, a61 // (op 5333) 
v_accvgpr_read v8, a62 // (op 5333) 
v_accvgpr_read v9, a63 // (op 5333) 
v_mul_f32 v14, v11, v6 // (op 5333) 
v_mul_f32 v15, v11, v7 // (op 5333) 
v_mul_f32 v16, v11, v8 // (op 5333) 
v_mul_f32 v17, v11, v9 // (op 5333) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 5324) 
v_lshrrev_b32 v7, 16, v4 // (op 5324) 
v_lshlrev_b32 v6, 16, v6 // (op 5324) 
v_lshlrev_b32 v7, 16, v7 // (op 5324) 
v_mul_f32 v18, v0, v6 // (op 5324) 
v_mul_f32 v19, v0, v7 // (op 5324) 
v_and_b32 v6, 65535, v5 // (op 5324) 
v_lshrrev_b32 v7, 16, v5 // (op 5324) 
v_lshlrev_b32 v6, 16, v6 // (op 5324) 
v_lshlrev_b32 v7, 16, v7 // (op 5324) 
v_mul_f32 v20, v0, v6 // (op 5324) 
v_mul_f32 v21, v0, v7 // (op 5324) 


v_add_f32 v4, v18, v14 // (op 5325) 
v_add_f32 v5, v19, v15 // (op 5325) 
v_add_f32 v6, v20, v16 // (op 5325) 
v_add_f32 v7, v21, v17 // (op 5325) 


v_lshrrev_b32 v14, 16, v4 // (op 5326) 
v_lshrrev_b32 v15, 16, v5 // (op 5326) 
v_lshrrev_b32 v16, 16, v6 // (op 5326) 
v_lshrrev_b32 v17, 16, v7 // (op 5326) 


v_pack_b32_f16 v4, v14, v15 // (op 5309) 
v_pack_b32_f16 v5, v16, v17 // (op 5309) 

s_lshl_b32 s0, s4, 5 // (op 5309) 
s_mul_i32 s1, 3, s0 // (op 5309) 
s_add_u32 s1, s1, 96 // (op 5309) 

v_add_u32 v3, v13, s1 // (op 5309) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5309) Store value





v_mov_b32 v3, v12 // (op 129) reg expression

buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 129) Load value

v_accvgpr_read v6, a0 // (op 125) 
v_accvgpr_read v7, a1 // (op 125) 
v_accvgpr_read v8, a2 // (op 125) 
v_accvgpr_read v9, a3 // (op 125) 
v_mul_f32 v14, v11, v6 // (op 125) 
v_mul_f32 v15, v11, v7 // (op 125) 
v_mul_f32 v16, v11, v8 // (op 125) 
v_mul_f32 v17, v11, v9 // (op 125) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 130) 
v_lshrrev_b32 v7, 16, v4 // (op 130) 
v_lshlrev_b32 v6, 16, v6 // (op 130) 
v_lshlrev_b32 v7, 16, v7 // (op 130) 
v_mul_f32 v18, v0, v6 // (op 130) 
v_mul_f32 v19, v0, v7 // (op 130) 
v_and_b32 v6, 65535, v5 // (op 130) 
v_lshrrev_b32 v7, 16, v5 // (op 130) 
v_lshlrev_b32 v6, 16, v6 // (op 130) 
v_lshlrev_b32 v7, 16, v7 // (op 130) 
v_mul_f32 v20, v0, v6 // (op 130) 
v_mul_f32 v21, v0, v7 // (op 130) 


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

v_mov_b32 v3, v13 // (op 128) reg expression

buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 128) Store value





s_lshl_b32 s0, s2, 5 // (op 5046) 
v_add_u32 v3, v12, s0 // (op 5046) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 5046) Load value

v_accvgpr_read v6, a4 // (op 5042) 
v_accvgpr_read v7, a5 // (op 5042) 
v_accvgpr_read v8, a6 // (op 5042) 
v_accvgpr_read v9, a7 // (op 5042) 
v_mul_f32 v14, v11, v6 // (op 5042) 
v_mul_f32 v15, v11, v7 // (op 5042) 
v_mul_f32 v16, v11, v8 // (op 5042) 
v_mul_f32 v17, v11, v9 // (op 5042) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 5047) 
v_lshrrev_b32 v7, 16, v4 // (op 5047) 
v_lshlrev_b32 v6, 16, v6 // (op 5047) 
v_lshlrev_b32 v7, 16, v7 // (op 5047) 
v_mul_f32 v18, v0, v6 // (op 5047) 
v_mul_f32 v19, v0, v7 // (op 5047) 
v_and_b32 v6, 65535, v5 // (op 5047) 
v_lshrrev_b32 v7, 16, v5 // (op 5047) 
v_lshlrev_b32 v6, 16, v6 // (op 5047) 
v_lshlrev_b32 v7, 16, v7 // (op 5047) 
v_mul_f32 v20, v0, v6 // (op 5047) 
v_mul_f32 v21, v0, v7 // (op 5047) 


v_add_f32 v4, v18, v14 // (op 5043) 
v_add_f32 v5, v19, v15 // (op 5043) 
v_add_f32 v6, v20, v16 // (op 5043) 
v_add_f32 v7, v21, v17 // (op 5043) 


v_lshrrev_b32 v14, 16, v4 // (op 5044) 
v_lshrrev_b32 v15, 16, v5 // (op 5044) 
v_lshrrev_b32 v16, 16, v6 // (op 5044) 
v_lshrrev_b32 v17, 16, v7 // (op 5044) 


v_pack_b32_f16 v4, v14, v15 // (op 5045) 
v_pack_b32_f16 v5, v16, v17 // (op 5045) 

s_lshl_b32 s0, s4, 5 // (op 5045) 
v_add_u32 v3, v13, s0 // (op 5045) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5045) Store value





s_lshl_b32 s0, s2, 6 // (op 5060) 
v_add_u32 v3, v12, s0 // (op 5060) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 5060) Load value

v_accvgpr_read v6, a8 // (op 5056) 
v_accvgpr_read v7, a9 // (op 5056) 
v_accvgpr_read v8, a10 // (op 5056) 
v_accvgpr_read v9, a11 // (op 5056) 
v_mul_f32 v14, v11, v6 // (op 5056) 
v_mul_f32 v15, v11, v7 // (op 5056) 
v_mul_f32 v16, v11, v8 // (op 5056) 
v_mul_f32 v17, v11, v9 // (op 5056) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 5061) 
v_lshrrev_b32 v7, 16, v4 // (op 5061) 
v_lshlrev_b32 v6, 16, v6 // (op 5061) 
v_lshlrev_b32 v7, 16, v7 // (op 5061) 
v_mul_f32 v18, v0, v6 // (op 5061) 
v_mul_f32 v19, v0, v7 // (op 5061) 
v_and_b32 v6, 65535, v5 // (op 5061) 
v_lshrrev_b32 v7, 16, v5 // (op 5061) 
v_lshlrev_b32 v6, 16, v6 // (op 5061) 
v_lshlrev_b32 v7, 16, v7 // (op 5061) 
v_mul_f32 v20, v0, v6 // (op 5061) 
v_mul_f32 v21, v0, v7 // (op 5061) 


v_add_f32 v4, v18, v14 // (op 5057) 
v_add_f32 v5, v19, v15 // (op 5057) 
v_add_f32 v6, v20, v16 // (op 5057) 
v_add_f32 v7, v21, v17 // (op 5057) 


v_lshrrev_b32 v14, 16, v4 // (op 5058) 
v_lshrrev_b32 v15, 16, v5 // (op 5058) 
v_lshrrev_b32 v16, 16, v6 // (op 5058) 
v_lshrrev_b32 v17, 16, v7 // (op 5058) 


v_pack_b32_f16 v4, v14, v15 // (op 5059) 
v_pack_b32_f16 v5, v16, v17 // (op 5059) 

s_lshl_b32 s0, s4, 6 // (op 5059) 
v_add_u32 v3, v13, s0 // (op 5059) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5059) Store value





s_lshl_b32 s0, s2, 5 // (op 5074) 
s_mul_i32 s1, 3, s0 // (op 5074) 

v_add_u32 v3, v12, s1 // (op 5074) 


buffer_load_dwordx2 v[4:5], v3, s[52:55], 0 offen // (op 5074) Load value

v_accvgpr_read v6, a12 // (op 5070) 
v_accvgpr_read v7, a13 // (op 5070) 
v_accvgpr_read v8, a14 // (op 5070) 
v_accvgpr_read v9, a15 // (op 5070) 
v_mul_f32 v14, v11, v6 // (op 5070) 
v_mul_f32 v15, v11, v7 // (op 5070) 
v_mul_f32 v16, v11, v8 // (op 5070) 
v_mul_f32 v17, v11, v9 // (op 5070) 

s_waitcnt vmcnt(0)
v_and_b32 v6, 65535, v4 // (op 5075) 
v_lshrrev_b32 v7, 16, v4 // (op 5075) 
v_lshlrev_b32 v6, 16, v6 // (op 5075) 
v_lshlrev_b32 v7, 16, v7 // (op 5075) 
v_mul_f32 v18, v0, v6 // (op 5075) 
v_mul_f32 v19, v0, v7 // (op 5075) 
v_and_b32 v6, 65535, v5 // (op 5075) 
v_lshrrev_b32 v7, 16, v5 // (op 5075) 
v_lshlrev_b32 v6, 16, v6 // (op 5075) 
v_lshlrev_b32 v7, 16, v7 // (op 5075) 
v_mul_f32 v20, v0, v6 // (op 5075) 
v_mul_f32 v21, v0, v7 // (op 5075) 


v_add_f32 v4, v18, v14 // (op 5071) 
v_add_f32 v5, v19, v15 // (op 5071) 
v_add_f32 v6, v20, v16 // (op 5071) 
v_add_f32 v7, v21, v17 // (op 5071) 


v_lshrrev_b32 v14, 16, v4 // (op 5072) 
v_lshrrev_b32 v15, 16, v5 // (op 5072) 
v_lshrrev_b32 v16, 16, v6 // (op 5072) 
v_lshrrev_b32 v17, 16, v7 // (op 5072) 


v_pack_b32_f16 v4, v14, v15 // (op 5073) 
v_pack_b32_f16 v5, v16, v17 // (op 5073) 

s_lshl_b32 s0, s4, 5 // (op 5073) 
s_mul_i32 s1, 3, s0 // (op 5073) 

v_add_u32 v3, v13, s1 // (op 5073) 


buffer_store_dwordx2 v[4:5], v3, s[56:59], 0 offen // (op 5073) Store value



s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__ConditionalBottom_0_beta_116:
























s_endpgm  // End of RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM_
.LRR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__end:
.size RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM_, .LRR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM__end-RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM_
  .amdhsa_next_free_vgpr 172
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 65536 // lds bytes
  .amdhsa_accum_offset 108
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
  - .name: RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM_
    .symbol: RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_4_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 65536
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 76
    .vgpr_count: 106
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

