// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT



.amdgcn_target "amdgcn-amd-amdhsa--gfx950:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM_
.p2align 8
.type RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM_,@function
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM_:
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
s_nop 0
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__exec_begin:


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


s_mul_hi_i32 s0, s16, s10 // (op 6944) 
s_mul_hi_i32 s1, s16, s10 // (op 7085) 
v_lshrrev_b32 v0, 7, v1 // (op 7006) 
v_and_b32 v3, 15, v1 // (op 7006) 
v_lshl_add_u32 v4, v0, 6, v3 // (op 7006) 


v_and_b32 v0, 63, v1 // (op 7006) 
v_lshrrev_b32 v3, 4, v0 // (op 7006) 

v_lshlrev_b32 v0, 5, v3 // (op 7006) 

v_lshl_add_u32 v3, v4, 7, v0 // (op 7006) 


v_mul_lo_u32 v0, v3, 6 // (op 7006) 

v_lshrrev_b32 v3, 3, v0 // (op 7006) 

v_lshrrev_b32 v0, 6, v1 // (op 7064) 
v_and_b32 v4, 1, v0 // (op 7064) 

v_and_b32 v0, 15, v1 // (op 7064) 
v_lshl_add_u32 v5, v4, 6, v0 // (op 7064) 


v_and_b32 v0, 63, v1 // (op 7064) 
v_lshrrev_b32 v4, 4, v0 // (op 7064) 

v_lshlrev_b32 v0, 5, v4 // (op 7064) 

v_lshl_add_u32 v4, v5, 7, v0 // (op 7064) 


v_mul_lo_u32 v0, v4, 6 // (op 7064) 

v_lshrrev_b32 v4, 3, v0 // (op 7064) 

v_lshrrev_b32 v0, 7, v1 // (op 7116) 
v_and_b32 v5, 15, v1 // (op 7116) 
v_lshl_add_u32 v6, v0, 6, v5 // (op 7116) 


v_and_b32 v0, 63, v1 // (op 7116) 
v_lshrrev_b32 v5, 4, v0 // (op 7116) 

v_lshlrev_b32 v0, 5, v5 // (op 7116) 

v_lshl_add_u32 v5, v6, 7, v0 // (op 7116) 


v_mul_lo_u32 v0, v5, 6 // (op 7116) 

v_lshrrev_b32 v5, 3, v0 // (op 7116) 

v_lshrrev_b32 v0, 6, v1 // (op 7145) 
v_and_b32 v6, 1, v0 // (op 7145) 

v_and_b32 v0, 15, v1 // (op 7145) 
v_lshl_add_u32 v7, v6, 6, v0 // (op 7145) 


v_and_b32 v0, 63, v1 // (op 7145) 
v_lshrrev_b32 v6, 4, v0 // (op 7145) 

v_lshlrev_b32 v0, 5, v6 // (op 7145) 

v_lshl_add_u32 v6, v7, 7, v0 // (op 7145) 


v_mul_lo_u32 v0, v6, 6 // (op 7145) 

v_lshrrev_b32 v6, 3, v0 // (op 7145) 

v_lshrrev_b32 v0, 7, v1 // (op 6944) 
s_add_i32 s18, s0, s16 // (op 6944) 

v_and_b32 v7, 63, v1 // (op 6944) 
s_ashr_i32 s0, s18, 31 // (op 6944) 
v_lshrrev_b32 v8, 4, v7 // (op 6944) 
s_cmp_eq_i32 s10, 0 // (op 6944) 
s_mov_b32 s19, scc // (op 6944) 
v_lshlrev_b32 v9, 2, v8 // (op 6944) 

s_cmp_eq_i32 s19, 0 // (op 6944) 
s_cselect_b32 s49, 0, -1 // (op 6944) 

v_lshl_add_u32 v8, v0, 6, v9 // (op 6944) 


s_and_b32 s19, s13, 31 // (op 6944) 
v_lshrrev_b32 v0, 2, v7 // (op 6944) 

v_and_b32 v7, 3, v0 // (op 6944) 

v_lshrrev_b32 v0, 7, v1 // (op 7085) 
v_and_b32 v9, 63, v1 // (op 7085) 
s_lshl_b32 s50, 1, s19 // (op 6944) 
v_and_b32 v10, 3, v1 // (op 6944) 
s_add_i32 s51, s50, s49 // (op 6944) 


s_and_b32 s49, s0, s51 // (op 6944) 


s_add_i32 s0, s1, s16 // (op 7085) 

s_ashr_i32 s1, s0, 31 // (op 7085) 
v_lshl_add_u32 v11, v7, 2, v10 // (op 6944) 


s_add_i32 s50, s18, s49 // (op 6944) 


v_lshrrev_b32 v7, 6, v1 // (op 6944) 
s_ashr_i32 s18, s50, s19 // (op 6944) 


v_and_b32 v10, 1, v7 // (op 6944) 

s_lshl_b32 s19, s13, 24 // (op 6944) 
v_lshl_add_u32 v7, v10, 6, v11 // (op 6944) 


s_ashr_i32 s49, s19, 31 // (op 6944) 

s_xor_b32 s19, s18, s49 // (op 6944) 

s_sub_i32 s18, s19, s49 // (op 6944) 


s_mul_hi_i32 s19, s18, s14 // (op 6944) 
s_add_i32 s49, s19, s18 // (op 6944) 

s_ashr_i32 s19, s49, 31 // (op 6944) 
s_cmp_eq_i32 s14, 0 // (op 6944) 
s_mov_b32 s50, scc // (op 6944) 
s_cmp_eq_i32 s50, 0 // (op 6944) 
s_cselect_b32 s51, 0, -1 // (op 6944) 

s_and_b32 s50, s15, 31 // (op 6944) 
v_lshrrev_b32 v10, 4, v9 // (op 7085) 
s_cmp_eq_i32 s10, 0 // (op 7085) 
s_mov_b32 s52, scc // (op 7085) 
v_lshlrev_b32 v11, 2, v10 // (op 7085) 

s_cmp_eq_i32 s52, 0 // (op 7085) 
s_cselect_b32 s53, 0, -1 // (op 7085) 

s_lshl_b32 s52, 1, s50 // (op 6944) 
v_lshl_add_u32 v10, v0, 6, v11 // (op 7085) 


v_lshrrev_b32 v0, 2, v9 // (op 7085) 

s_add_i32 s54, s52, s51 // (op 6944) 


s_and_b32 s51, s19, s54 // (op 6944) 


s_add_i32 s19, s49, s51 // (op 6944) 


s_ashr_i32 s49, s19, s50 // (op 6944) 


s_lshl_b32 s19, s15, 24 // (op 6944) 
s_ashr_i32 s50, s19, 31 // (op 6944) 

s_xor_b32 s19, s49, s50 // (op 6944) 

s_sub_i32 s49, s19, s50 // (op 6944) 


s_mul_i32 s19, s18, s11 // (op 6944) 
s_sub_u32 s50, s16, s19 // (op 6944) 

s_mul_i32 s19, s49, s11 // (op 6944) 
s_add_u32 s19, s19, s50 // (op 6944) 

s_mul_hi_i32 s50, s16, s9 // (op 6944) 
s_add_i32 s51, s50, s16 // (op 6944) 

s_ashr_i32 s50, s51, 31 // (op 6944) 
s_cmp_eq_i32 s9, 0 // (op 6944) 
s_mov_b32 s52, scc // (op 6944) 
s_cmp_eq_i32 s52, 0 // (op 6944) 
s_cselect_b32 s54, 0, -1 // (op 6944) 

s_and_b32 s52, s13, 31 // (op 7085) 
v_and_b32 v9, 3, v0 // (op 7085) 

s_lshl_b32 s55, 1, s52 // (op 7085) 
v_and_b32 v0, 3, v1 // (op 7085) 
s_add_i32 s56, s55, s53 // (op 7085) 


s_and_b32 s53, s1, s56 // (op 7085) 


v_lshl_add_u32 v11, v9, 2, v0 // (op 7085) 


s_add_i32 s1, s0, s53 // (op 7085) 


v_lshrrev_b32 v0, 6, v1 // (op 7085) 
s_ashr_i32 s0, s1, s52 // (op 7085) 


v_and_b32 v9, 1, v0 // (op 7085) 

s_lshl_b32 s1, s13, 24 // (op 7085) 
v_lshl_add_u32 v0, v9, 6, v11 // (op 7085) 


s_ashr_i32 s52, s1, 31 // (op 7085) 

s_xor_b32 s1, s0, s52 // (op 7085) 

s_sub_i32 s0, s1, s52 // (op 7085) 


s_mul_hi_i32 s1, s0, s14 // (op 7085) 
s_add_i32 s52, s1, s0 // (op 7085) 

s_ashr_i32 s1, s52, 31 // (op 7085) 
s_cmp_eq_i32 s14, 0 // (op 7085) 
s_mov_b32 s53, scc // (op 7085) 
s_cmp_eq_i32 s53, 0 // (op 7085) 
s_cselect_b32 s55, 0, -1 // (op 7085) 

s_and_b32 s53, s15, 31 // (op 7085) 
s_lshl_b32 s56, 1, s53 // (op 7085) 
s_add_i32 s57, s56, s55 // (op 7085) 


s_and_b32 s55, s1, s57 // (op 7085) 


s_add_i32 s1, s52, s55 // (op 7085) 


s_ashr_i32 s52, s1, s53 // (op 7085) 


s_lshl_b32 s1, s15, 24 // (op 7085) 
s_ashr_i32 s53, s1, 31 // (op 7085) 

s_xor_b32 s1, s52, s53 // (op 7085) 

s_sub_i32 s52, s1, s53 // (op 7085) 


s_mul_i32 s1, s0, s11 // (op 7085) 
s_sub_u32 s53, s16, s1 // (op 7085) 

s_mul_i32 s1, s52, s11 // (op 7085) 
s_add_u32 s1, s1, s53 // (op 7085) 

s_mul_hi_i32 s53, s16, s9 // (op 7085) 
s_add_i32 s55, s53, s16 // (op 7085) 

s_ashr_i32 s53, s55, 31 // (op 7085) 
s_cmp_eq_i32 s9, 0 // (op 7085) 
s_mov_b32 s56, scc // (op 7085) 
s_cmp_eq_i32 s56, 0 // (op 7085) 
s_cselect_b32 s57, 0, -1 // (op 7085) 


s_waitcnt lgkmcnt(0)
s_and_b32 s56, s45, 31 // (op 6944) 
s_lshl_b32 s58, 1, s56 // (op 6944) 
s_add_i32 s59, s58, s54 // (op 6944) 


s_and_b32 s54, s50, s59 // (op 6944) 


s_add_i32 s50, s51, s54 // (op 6944) 


s_ashr_i32 s51, s50, s56 // (op 6944) 


s_lshl_b32 s50, s45, 24 // (op 6944) 
s_ashr_i32 s54, s50, 31 // (op 6944) 

s_xor_b32 s50, s51, s54 // (op 6944) 

s_sub_i32 s51, s50, s54 // (op 6944) 


s_and_b32 s50, s45, 31 // (op 7085) 
s_lshl_b32 s54, 1, s50 // (op 7085) 
s_add_i32 s56, s54, s57 // (op 7085) 


s_and_b32 s54, s53, s56 // (op 7085) 


s_add_i32 s53, s55, s54 // (op 7085) 


s_ashr_i32 s54, s53, s50 // (op 7085) 


s_lshl_b32 s50, s45, 24 // (op 7085) 
s_ashr_i32 s53, s50, 31 // (op 7085) 

s_xor_b32 s50, s54, s53 // (op 7085) 

s_sub_i32 s54, s50, s53 // (op 7085) 


s_add_u32 s56, s6, 128 // (op 6944) Start of Int64 add, locking SCC
s_addc_u32 s57, s7, 0 // (op 6944) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 6944) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 6944) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 7 // (op 6944) 

s_mul_i32 s50, s11, s56 // (op 6944) 
s_mul_i32 s53, s8, s50 // (op 6944) 

s_mul_i32 s50, s51, s53 // (op 6944) 

s_sub_u32 s53, s16, s50 // (op 6944) 

s_mov_b32 s50, s53 // (op 6944) 
s_mul_hi_i32 s55, s50, s12 // (op 6944) 
s_add_i32 s58, s55, s50 // (op 6944) 


s_ashr_i32 s50, s58, 31 // (op 6944) 
s_cmp_eq_i32 s12, 0 // (op 6944) 
s_mov_b32 s55, scc // (op 6944) 
s_cmp_eq_i32 s55, 0 // (op 6944) 
s_cselect_b32 s59, 0, -1 // (op 6944) 

s_and_b32 s55, s44, 31 // (op 6944) 
s_lshl_b32 s60, 1, s55 // (op 6944) 
s_add_i32 s61, s60, s59 // (op 6944) 


s_and_b32 s59, s50, s61 // (op 6944) 


s_add_i32 s50, s58, s59 // (op 6944) 


s_ashr_i32 s58, s50, s55 // (op 6944) 


s_lshl_b32 s50, s44, 24 // (op 6944) 
s_ashr_i32 s55, s50, 31 // (op 6944) 

s_xor_b32 s50, s58, s55 // (op 6944) 

s_sub_i32 s58, s50, s55 // (op 6944) 


s_mul_i32 s50, s46, s11 // (op 6944) 
s_add_u32 s60, s20, 128 // (op 6944) Start of Int64 add, locking SCC
s_addc_u32 s61, s21, 0 // (op 6944) End of Int64 add, unlocking SCC
s_sub_u32 s62, s60, 1 // (op 6944) least significant half; sets scc
s_subb_u32 s63, s61, 0 // (op 6944) most significant half; uses scc

s_ashr_i64 s[60:61], s[62:63], 7 // (op 6944) 

s_sub_i32 s55, s60, s50 // (op 6944) 

s_mul_i32 s59, s58, s55 // (op 6944) 

s_sub_u32 s55, s53, s59 // (op 6944) 


s_add_u32 s53, s55, s50 // (op 6944) 


s_cmp_eq_i32 s51, 0 // (op 6944) 
s_mov_b32 s50, scc // (op 6944) 

s_cmp_eq_i32 s50, 0 // (op 6944) 
s_cselect_b32 s51, s53, s19 // (op 6944) 


s_mul_i32 s19, s49, s56 // (op 6944) 


v_lshl_add_u32 v9, s51, 7, v8 // (op 6944) 


s_sub_i32 s49, s18, s19 // (op 6944) 


s_cmp_eq_i32 s50, 0 // (op 6944) 
s_cselect_b32 s51, s58, s49 // (op 6944) 



v_lshl_add_u32 v8, s51, 7, v7 // (op 6944) 


v_mul_lo_u32 v7, v8, s2 // (op 6944) 

v_add_lshl_u32 v8, v9, v7, 1 // (op 6944) 


s_add_u32 s18, s6, 128 // (op 7085) Start of Int64 add, locking SCC
s_addc_u32 s19, s7, 0 // (op 7085) End of Int64 add, unlocking SCC
s_sub_u32 s50, s18, 1 // (op 7085) least significant half; sets scc
s_subb_u32 s51, s19, 0 // (op 7085) most significant half; uses scc

s_ashr_i64 s[18:19], s[50:51], 7 // (op 7085) 

s_mul_i32 s53, s11, s18 // (op 7085) 
s_mul_i32 s49, s8, s53 // (op 7085) 

s_mul_i32 s53, s54, s49 // (op 7085) 

s_sub_u32 s49, s16, s53 // (op 7085) 

s_mov_b32 s53, s49 // (op 7085) 
s_mul_hi_i32 s50, s53, s12 // (op 7085) 
s_add_i32 s51, s50, s53 // (op 7085) 


s_ashr_i32 s50, s51, 31 // (op 7085) 
s_cmp_eq_i32 s12, 0 // (op 7085) 
s_mov_b32 s53, scc // (op 7085) 
s_cmp_eq_i32 s53, 0 // (op 7085) 
s_cselect_b32 s55, 0, -1 // (op 7085) 

s_and_b32 s53, s44, 31 // (op 7085) 
s_lshl_b32 s56, 1, s53 // (op 7085) 
s_add_i32 s57, s56, s55 // (op 7085) 


s_and_b32 s55, s50, s57 // (op 7085) 


s_add_i32 s50, s51, s55 // (op 7085) 


s_ashr_i32 s51, s50, s53 // (op 7085) 


s_lshl_b32 s50, s44, 24 // (op 7085) 
s_ashr_i32 s53, s50, 31 // (op 7085) 

s_xor_b32 s50, s51, s53 // (op 7085) 

s_sub_i32 s51, s50, s53 // (op 7085) 


s_mul_i32 s50, s46, s11 // (op 7085) 
s_add_u32 s56, s20, 128 // (op 7085) Start of Int64 add, locking SCC
s_addc_u32 s57, s21, 0 // (op 7085) End of Int64 add, unlocking SCC
s_sub_u32 s58, s56, 1 // (op 7085) least significant half; sets scc
s_subb_u32 s59, s57, 0 // (op 7085) most significant half; uses scc

s_ashr_i64 s[56:57], s[58:59], 7 // (op 7085) 

s_sub_i32 s53, s56, s50 // (op 7085) 

s_mul_i32 s55, s51, s53 // (op 7085) 

s_sub_u32 s53, s49, s55 // (op 7085) 


s_add_u32 s49, s53, s50 // (op 7085) 


s_cmp_eq_i32 s54, 0 // (op 7085) 
s_mov_b32 s50, scc // (op 7085) 

s_cmp_eq_i32 s50, 0 // (op 7085) 
s_cselect_b32 s53, s49, s1 // (op 7085) 


s_mul_i32 s1, s52, s18 // (op 7085) 


v_lshl_add_u32 v7, s53, 7, v10 // (op 7085) 


s_sub_i32 s49, s0, s1 // (op 7085) 


s_cmp_eq_i32 s50, 0 // (op 7085) 
s_cselect_b32 s0, s51, s49 // (op 7085) 



v_lshl_add_u32 v9, s0, 7, v0 // (op 7085) 


v_mul_lo_u32 v0, v9, s4 // (op 7085) 

v_add_lshl_u32 v9, v7, v0, 1 // (op 7085) 


s_mul_i32 s1, s2, s7 // (op 6946) most significant: low of low * high
s_mul_i32 s18, s3, s6 // (op 6946) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 6946) least significant
s_mul_hi_u32 s19, s2, s6 // (op 6946) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 6946) most significant: sum
s_add_u32 s1, s1, s19 // (op 6946) most significant: sum


s_lshl_b64 s[18:19], s[0:1], 1 // (op 6946) 

s_mul_i32 s1, s4, s7 // (op 7087) most significant: low of low * high
s_mul_i32 s49, s5, s6 // (op 7087) most significant: low of high * low
s_mul_i32 s0, s4, s6 // (op 7087) least significant
s_mul_hi_u32 s50, s4, s6 // (op 7087) most significant: high of low * low
s_add_u32 s1, s1, s49 // (op 7087) most significant: sum
s_add_u32 s1, s1, s50 // (op 7087) most significant: sum


s_lshl_b64 s[50:51], s[0:1], 1 // (op 7087) 

s_mov_b32 s52, s22 // (op 6946) reg expression
s_mov_b32 s53, s23 // (op 6946) reg expression
s_mov_b32 s54, s18 // (op 6946) reg expression
s_mov_b32 s55, 131072 // (op 6946) call()

s_mov_b32 s56, s24 // (op 7087) reg expression

s_mov_b32 s57, s25 // (op 7087) reg expression
s_mov_b32 s58, s50 // (op 7087) reg expression
s_mov_b32 s59, 131072 // (op 7087) call()


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
s_mul_hi_i32 s0, s16, s10 // (op 6834) 
v_lshrrev_b32 v10, 1, v1 // (op 6834) 
s_add_i32 s1, s0, s16 // (op 6834) 

v_and_b32 v11, -2, v1 // (op 6834) 
s_ashr_i32 s0, s1, 31 // (op 6834) 
v_sub_u32 v12, v1, v11 // (op 6834) 

s_cmp_eq_i32 s10, 0 // (op 6834) 
s_mov_b32 s18, scc // (op 6834) 
v_lshlrev_b32 v11, 6, v12 // (op 6834) 

s_cmp_eq_i32 s18, 0 // (op 6834) 
s_cselect_b32 s19, 0, -1 // (op 6834) 

v_mov_b32 v13, 0 // (op 6834) convert
v_mov_b32 v12, v11 // (op 6834) convert

s_and_b32 s18, s13, 31 // (op 6834) 
s_lshl_b32 s22, 1, s18 // (op 6834) 
s_add_i32 s23, s22, s19 // (op 6834) 


s_and_b32 s19, s0, s23 // (op 6834) 


s_add_i32 s0, s1, s19 // (op 6834) 


s_ashr_i32 s1, s0, s18 // (op 6834) 


s_lshl_b32 s0, s13, 24 // (op 6834) 
s_ashr_i32 s18, s0, 31 // (op 6834) 

s_xor_b32 s0, s1, s18 // (op 6834) 

s_sub_i32 s1, s0, s18 // (op 6834) 


s_mul_hi_i32 s0, s1, s14 // (op 6834) 
s_add_i32 s18, s0, s1 // (op 6834) 

s_ashr_i32 s0, s18, 31 // (op 6834) 
s_cmp_eq_i32 s14, 0 // (op 6834) 
s_mov_b32 s19, scc // (op 6834) 
s_cmp_eq_i32 s19, 0 // (op 6834) 
s_cselect_b32 s22, 0, -1 // (op 6834) 

s_and_b32 s19, s15, 31 // (op 6834) 
s_lshl_b32 s23, 1, s19 // (op 6834) 
s_add_i32 s24, s23, s22 // (op 6834) 


s_and_b32 s25, s0, s24 // (op 6834) 


s_add_i32 s0, s18, s25 // (op 6834) 


s_ashr_i32 s18, s0, s19 // (op 6834) 


s_lshl_b32 s0, s15, 24 // (op 6834) 
s_ashr_i32 s19, s0, 31 // (op 6834) 

s_xor_b32 s0, s18, s19 // (op 6834) 

s_sub_i32 s18, s0, s19 // (op 6834) 


s_mul_i32 s0, s1, s11 // (op 6834) 

s_sub_u32 s1, s16, s0 // (op 6834) 

s_mul_i32 s0, s18, s11 // (op 6834) 
s_add_u32 s0, s0, s1 // (op 6834) 


s_mul_hi_i32 s1, s16, s9 // (op 6834) 
s_add_i32 s18, s1, s16 // (op 6834) 

s_ashr_i32 s1, s18, 31 // (op 6834) 
s_cmp_eq_i32 s9, 0 // (op 6834) 
s_mov_b32 s19, scc // (op 6834) 
s_cmp_eq_i32 s19, 0 // (op 6834) 
s_cselect_b32 s22, 0, -1 // (op 6834) 

s_and_b32 s19, s45, 31 // (op 6834) 
s_lshl_b32 s23, 1, s19 // (op 6834) 
s_add_i32 s24, s23, s22 // (op 6834) 


s_and_b32 s25, s1, s24 // (op 6834) 


s_add_i32 s1, s18, s25 // (op 6834) 


s_ashr_i32 s18, s1, s19 // (op 6834) 


s_lshl_b32 s1, s45, 24 // (op 6834) 
s_ashr_i32 s19, s1, 31 // (op 6834) 

s_xor_b32 s1, s18, s19 // (op 6834) 

s_sub_i32 s18, s1, s19 // (op 6834) 


s_add_u32 s22, s6, 128 // (op 6834) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 6834) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 6834) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 6834) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 6834) 

s_mul_i32 s1, s11, s22 // (op 6834) 

s_mul_i32 s19, s8, s1 // (op 6834) 

s_mul_i32 s1, s18, s19 // (op 6834) 

s_sub_u32 s19, s16, s1 // (op 6834) 

s_mov_b32 s1, s19 // (op 6834) 
s_mul_hi_i32 s22, s1, s12 // (op 6834) 
s_add_i32 s23, s22, s1 // (op 6834) 


s_ashr_i32 s1, s23, 31 // (op 6834) 
s_cmp_eq_i32 s12, 0 // (op 6834) 
s_mov_b32 s22, scc // (op 6834) 
s_cmp_eq_i32 s22, 0 // (op 6834) 
s_cselect_b32 s24, 0, -1 // (op 6834) 

s_and_b32 s22, s44, 31 // (op 6834) 
s_lshl_b32 s25, 1, s22 // (op 6834) 
s_add_i32 s47, s25, s24 // (op 6834) 


s_and_b32 s24, s1, s47 // (op 6834) 


s_add_i32 s1, s23, s24 // (op 6834) 


s_ashr_i32 s23, s1, s22 // (op 6834) 


s_lshl_b32 s1, s44, 24 // (op 6834) 
s_ashr_i32 s22, s1, 31 // (op 6834) 

s_xor_b32 s1, s23, s22 // (op 6834) 

s_sub_i32 s23, s1, s22 // (op 6834) 


s_mul_i32 s1, s46, s11 // (op 6834) 
s_add_u32 s24, s20, 128 // (op 6834) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 6834) End of Int64 add, unlocking SCC
s_sub_u32 s50, s24, 1 // (op 6834) least significant half; sets scc
s_subb_u32 s51, s25, 0 // (op 6834) most significant half; uses scc

s_ashr_i64 s[24:25], s[50:51], 7 // (op 6834) 

s_sub_i32 s22, s24, s1 // (op 6834) 

s_mul_i32 s24, s23, s22 // (op 6834) 


s_sub_u32 s25, s19, s24 // (op 6834) 


s_add_u32 s19, s25, s1 // (op 6834) 


s_cmp_eq_i32 s18, 0 // (op 6834) 
s_mov_b32 s1, scc // (op 6834) 

s_cmp_eq_i32 s1, 0 // (op 6834) 
s_cselect_b32 s18, s19, s0 // (op 6834) 



v_lshl_add_u32 v11, s18, 7, v10 // (op 6834) 


v_mul_lo_u32 v15, v11, s27 // (op 6834) most significant: low of low * high
v_mul_hi_u32 v10, v11, s26 // (op 6834) most significant: high of low * low
v_mul_lo_u32 v14, v11, s26 // (op 6834) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 6834) most significant: sum

v_add_co_u32 v14, s[0:1], v14, v12 // (op 6834) least significant half
v_addc_co_u32 v15, s[0:1], v15, v13, s[0:1] // (op 6834) most significant half



v_mul_lo_u32 v11, v15, 6 // (op 6834) most significant: low of high * low
v_mul_hi_u32 v12, v14, 6 // (op 6834) most significant: high of low * low
v_mul_lo_u32 v10, v14, 6 // (op 6834) least significant: low of low * low
v_add_u32 v11, v11, v12 // (op 6834) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[10:11] // (op 6834) 

v_mov_b32 v10, v12 // (op 6834) convert

s_mul_i32 s1, s21, 6 // (op 6836) most significant: low of high * low
s_mul_i32 s0, s20, 6 // (op 6836) least significant
s_mul_hi_u32 s18, s20, 6 // (op 6836) most significant: high of low * low
s_add_u32 s1, s1, s18 // (op 6836) most significant: sum

s_mul_i32 s19, s26, s1 // (op 6836) most significant: low of low * high
s_mul_i32 s22, s27, s0 // (op 6836) most significant: low of high * low
s_mul_i32 s18, s26, s0 // (op 6836) least significant
s_mul_hi_u32 s23, s26, s0 // (op 6836) most significant: high of low * low
s_add_u32 s19, s19, s22 // (op 6836) most significant: sum
s_add_u32 s19, s19, s23 // (op 6836) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 6836) 

s_mov_b32 s48, s28 // (op 6836) reg expression
s_mov_b32 s49, s29 // (op 6836) reg expression
s_mov_b32 s50, s0 // (op 6836) reg expression
s_mov_b32 s51, 131072 // (op 6836) call()

s_mul_hi_i32 s47, s16, s10 // (op 6853) 
v_lshrrev_b32 v11, 1, v1 // (op 6853) 
s_add_i32 s0, s47, s16 // (op 6853) 

v_and_b32 v12, -2, v1 // (op 6853) 
s_ashr_i32 s1, s0, 31 // (op 6853) 
v_sub_u32 v13, v1, v12 // (op 6853) 

s_cmp_eq_i32 s10, 0 // (op 6853) 
s_mov_b32 s47, scc // (op 6853) 
v_lshlrev_b32 v12, 6, v13 // (op 6853) 

s_cmp_eq_i32 s47, 0 // (op 6853) 
s_cselect_b32 s18, 0, -1 // (op 6853) 

v_mov_b32 v15, 0 // (op 6853) convert
v_mov_b32 v14, v12 // (op 6853) convert

s_and_b32 s19, s13, 31 // (op 6853) 
s_lshl_b32 s47, 1, s19 // (op 6853) 
s_add_i32 s22, s47, s18 // (op 6853) 


s_and_b32 s18, s1, s22 // (op 6853) 


s_add_i32 s1, s0, s18 // (op 6853) 


s_ashr_i32 s0, s1, s19 // (op 6853) 


s_lshl_b32 s1, s13, 24 // (op 6853) 
s_ashr_i32 s47, s1, 31 // (op 6853) 

s_xor_b32 s1, s0, s47 // (op 6853) 

s_sub_i32 s0, s1, s47 // (op 6853) 


s_mul_hi_i32 s1, s0, s14 // (op 6853) 
s_add_i32 s47, s1, s0 // (op 6853) 

s_ashr_i32 s1, s47, 31 // (op 6853) 
s_cmp_eq_i32 s14, 0 // (op 6853) 
s_mov_b32 s18, scc // (op 6853) 
s_cmp_eq_i32 s18, 0 // (op 6853) 
s_cselect_b32 s19, 0, -1 // (op 6853) 

s_and_b32 s18, s15, 31 // (op 6853) 
s_lshl_b32 s22, 1, s18 // (op 6853) 
s_add_i32 s23, s22, s19 // (op 6853) 


s_and_b32 s19, s1, s23 // (op 6853) 


s_add_i32 s1, s47, s19 // (op 6853) 


s_ashr_i32 s19, s1, s18 // (op 6853) 


s_lshl_b32 s1, s15, 24 // (op 6853) 
s_ashr_i32 s18, s1, 31 // (op 6853) 

s_xor_b32 s1, s19, s18 // (op 6853) 

s_sub_i32 s19, s1, s18 // (op 6853) 


s_mul_i32 s1, s0, s11 // (op 6853) 

s_sub_u32 s0, s16, s1 // (op 6853) 

s_mul_i32 s1, s19, s11 // (op 6853) 
s_add_u32 s1, s1, s0 // (op 6853) 


s_mul_hi_i32 s0, s16, s9 // (op 6853) 
s_add_i32 s47, s0, s16 // (op 6853) 

s_ashr_i32 s0, s47, 31 // (op 6853) 
s_cmp_eq_i32 s9, 0 // (op 6853) 
s_mov_b32 s18, scc // (op 6853) 
s_cmp_eq_i32 s18, 0 // (op 6853) 
s_cselect_b32 s19, 0, -1 // (op 6853) 

s_and_b32 s18, s45, 31 // (op 6853) 
s_lshl_b32 s22, 1, s18 // (op 6853) 
s_add_i32 s23, s22, s19 // (op 6853) 


s_and_b32 s19, s0, s23 // (op 6853) 


s_add_i32 s0, s47, s19 // (op 6853) 


s_ashr_i32 s19, s0, s18 // (op 6853) 


s_lshl_b32 s0, s45, 24 // (op 6853) 
s_ashr_i32 s18, s0, 31 // (op 6853) 

s_xor_b32 s0, s19, s18 // (op 6853) 

s_sub_i32 s19, s0, s18 // (op 6853) 


s_add_u32 s22, s6, 128 // (op 6853) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 6853) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 6853) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 6853) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 6853) 

s_mul_i32 s0, s11, s22 // (op 6853) 

s_mul_i32 s18, s8, s0 // (op 6853) 

s_mul_i32 s0, s19, s18 // (op 6853) 

s_sub_u32 s18, s16, s0 // (op 6853) 

s_mov_b32 s0, s18 // (op 6853) 
s_mul_hi_i32 s47, s0, s12 // (op 6853) 
s_add_i32 s22, s47, s0 // (op 6853) 


s_ashr_i32 s0, s22, 31 // (op 6853) 
s_cmp_eq_i32 s12, 0 // (op 6853) 
s_mov_b32 s47, scc // (op 6853) 
s_cmp_eq_i32 s47, 0 // (op 6853) 
s_cselect_b32 s23, 0, -1 // (op 6853) 

s_and_b32 s47, s44, 31 // (op 6853) 
s_lshl_b32 s24, 1, s47 // (op 6853) 
s_add_i32 s25, s24, s23 // (op 6853) 


s_and_b32 s23, s0, s25 // (op 6853) 


s_add_i32 s0, s22, s23 // (op 6853) 


s_ashr_i32 s22, s0, s47 // (op 6853) 


s_lshl_b32 s0, s44, 24 // (op 6853) 
s_ashr_i32 s47, s0, 31 // (op 6853) 

s_xor_b32 s0, s22, s47 // (op 6853) 

s_sub_i32 s22, s0, s47 // (op 6853) 


s_mul_i32 s0, s46, s11 // (op 6853) 
s_add_u32 s24, s20, 128 // (op 6853) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 6853) End of Int64 add, unlocking SCC
s_sub_u32 s60, s24, 1 // (op 6853) least significant half; sets scc
s_subb_u32 s61, s25, 0 // (op 6853) most significant half; uses scc

s_ashr_i64 s[24:25], s[60:61], 7 // (op 6853) 

s_sub_i32 s23, s24, s0 // (op 6853) 

s_mul_i32 s47, s22, s23 // (op 6853) 


s_sub_u32 s22, s18, s47 // (op 6853) 


s_add_u32 s18, s22, s0 // (op 6853) 


s_cmp_eq_i32 s19, 0 // (op 6853) 
s_mov_b32 s0, scc // (op 6853) 

s_cmp_eq_i32 s0, 0 // (op 6853) 
s_cselect_b32 s19, s18, s1 // (op 6853) 



v_lshl_add_u32 v12, s19, 7, v11 // (op 6853) 


v_mul_lo_u32 v17, v12, s27 // (op 6853) most significant: low of low * high
v_mul_hi_u32 v11, v12, s26 // (op 6853) most significant: high of low * low
v_mul_lo_u32 v16, v12, s26 // (op 6853) least significant: low of low * low
v_add_u32 v17, v17, v11 // (op 6853) most significant: sum

v_add_co_u32 v16, s[0:1], v16, v14 // (op 6853) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 6853) most significant half



v_mul_lo_u32 v15, v17, 6 // (op 6853) most significant: low of high * low
v_mul_hi_u32 v11, v16, 6 // (op 6853) most significant: high of low * low
v_mul_lo_u32 v14, v16, 6 // (op 6853) least significant: low of low * low
v_add_u32 v15, v15, v11 // (op 6853) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 6853) 

v_mov_b32 v11, v12 // (op 6853) convert

s_mul_i32 s1, s21, 6 // (op 6855) most significant: low of high * low
s_mul_i32 s0, s20, 6 // (op 6855) least significant
s_mul_hi_u32 s47, s20, 6 // (op 6855) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 6855) most significant: sum

s_mul_i32 s19, s26, s1 // (op 6855) most significant: low of low * high
s_mul_i32 s47, s27, s0 // (op 6855) most significant: low of high * low
s_mul_i32 s18, s26, s0 // (op 6855) least significant
s_mul_hi_u32 s22, s26, s0 // (op 6855) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 6855) most significant: sum
s_add_u32 s19, s19, s22 // (op 6855) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 6855) 

s_mov_b32 s48, s28 // (op 6855) reg expression
s_mov_b32 s49, s29 // (op 6855) reg expression
s_mov_b32 s50, s0 // (op 6855) reg expression
s_mov_b32 s51, 131072 // (op 6855) call()

s_mul_hi_i32 s47, s16, s10 // (op 6898) 
v_lshrrev_b32 v12, 1, v1 // (op 6898) 
s_add_i32 s0, s47, s16 // (op 6898) 

v_and_b32 v13, -2, v1 // (op 6898) 
s_ashr_i32 s1, s0, 31 // (op 6898) 
v_sub_u32 v14, v1, v13 // (op 6898) 

s_cmp_eq_i32 s10, 0 // (op 6898) 
s_mov_b32 s47, scc // (op 6898) 
v_lshlrev_b32 v13, 6, v14 // (op 6898) 

s_cmp_eq_i32 s47, 0 // (op 6898) 
s_cselect_b32 s18, 0, -1 // (op 6898) 

v_mov_b32 v15, 0 // (op 6898) convert
v_mov_b32 v14, v13 // (op 6898) convert

s_and_b32 s19, s13, 31 // (op 6898) 
s_lshl_b32 s47, 1, s19 // (op 6898) 
s_add_i32 s22, s47, s18 // (op 6898) 


s_and_b32 s18, s1, s22 // (op 6898) 


s_add_i32 s1, s0, s18 // (op 6898) 


s_ashr_i32 s0, s1, s19 // (op 6898) 


s_lshl_b32 s1, s13, 24 // (op 6898) 
s_ashr_i32 s47, s1, 31 // (op 6898) 

s_xor_b32 s1, s0, s47 // (op 6898) 

s_sub_i32 s0, s1, s47 // (op 6898) 


s_mul_hi_i32 s1, s0, s14 // (op 6898) 
s_add_i32 s47, s1, s0 // (op 6898) 

s_ashr_i32 s1, s47, 31 // (op 6898) 
s_cmp_eq_i32 s14, 0 // (op 6898) 
s_mov_b32 s18, scc // (op 6898) 
s_cmp_eq_i32 s18, 0 // (op 6898) 
s_cselect_b32 s19, 0, -1 // (op 6898) 

s_and_b32 s18, s15, 31 // (op 6898) 
s_lshl_b32 s22, 1, s18 // (op 6898) 
s_add_i32 s23, s22, s19 // (op 6898) 


s_and_b32 s19, s1, s23 // (op 6898) 


s_add_i32 s1, s47, s19 // (op 6898) 


s_ashr_i32 s19, s1, s18 // (op 6898) 


s_lshl_b32 s1, s15, 24 // (op 6898) 
s_ashr_i32 s18, s1, 31 // (op 6898) 

s_xor_b32 s1, s19, s18 // (op 6898) 

s_sub_i32 s19, s1, s18 // (op 6898) 


s_add_u32 s22, s6, 128 // (op 6898) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 6898) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 6898) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 6898) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 6898) 

s_mul_i32 s1, s19, s22 // (op 6898) 

s_sub_i32 s47, s0, s1 // (op 6898) 


s_mul_hi_i32 s0, s16, s9 // (op 6898) 
s_add_i32 s1, s0, s16 // (op 6898) 

s_ashr_i32 s0, s1, 31 // (op 6898) 
s_cmp_eq_i32 s9, 0 // (op 6898) 
s_mov_b32 s18, scc // (op 6898) 
s_cmp_eq_i32 s18, 0 // (op 6898) 
s_cselect_b32 s19, 0, -1 // (op 6898) 

s_and_b32 s18, s45, 31 // (op 6898) 
s_lshl_b32 s24, 1, s18 // (op 6898) 
s_add_i32 s25, s24, s19 // (op 6898) 


s_and_b32 s19, s0, s25 // (op 6898) 


s_add_i32 s0, s1, s19 // (op 6898) 


s_ashr_i32 s1, s0, s18 // (op 6898) 


s_lshl_b32 s0, s45, 24 // (op 6898) 
s_ashr_i32 s18, s0, 31 // (op 6898) 

s_xor_b32 s0, s1, s18 // (op 6898) 

s_sub_i32 s1, s0, s18 // (op 6898) 


s_mul_i32 s0, s11, s22 // (op 6898) 

s_mul_i32 s18, s8, s0 // (op 6898) 

s_mul_i32 s0, s1, s18 // (op 6898) 

s_sub_u32 s18, s16, s0 // (op 6898) 

s_mul_hi_i32 s0, s18, s12 // (op 6898) 
s_add_i32 s19, s0, s18 // (op 6898) 


s_ashr_i32 s0, s19, 31 // (op 6898) 
s_cmp_eq_i32 s12, 0 // (op 6898) 
s_mov_b32 s18, scc // (op 6898) 
s_cmp_eq_i32 s18, 0 // (op 6898) 
s_cselect_b32 s22, 0, -1 // (op 6898) 

s_and_b32 s18, s44, 31 // (op 6898) 
s_lshl_b32 s23, 1, s18 // (op 6898) 
s_add_i32 s24, s23, s22 // (op 6898) 


s_and_b32 s25, s0, s24 // (op 6898) 


s_add_i32 s0, s19, s25 // (op 6898) 


s_ashr_i32 s19, s0, s18 // (op 6898) 


s_lshl_b32 s0, s44, 24 // (op 6898) 
s_ashr_i32 s18, s0, 31 // (op 6898) 

s_xor_b32 s0, s19, s18 // (op 6898) 

s_sub_i32 s19, s0, s18 // (op 6898) 


s_cmp_eq_i32 s1, 0 // (op 6898) 
s_mov_b32 s0, scc // (op 6898) 

s_cmp_eq_i32 s0, 0 // (op 6898) 
s_cselect_b32 s1, s19, s47 // (op 6898) 



v_lshl_add_u32 v13, s1, 7, v12 // (op 6898) 


v_mul_lo_u32 v17, v13, s31 // (op 6898) most significant: low of low * high
v_mul_hi_u32 v12, v13, s30 // (op 6898) most significant: high of low * low
v_mul_lo_u32 v16, v13, s30 // (op 6898) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 6898) most significant: sum


v_add_co_u32 v12, s[0:1], v14, v16 // (op 6898) least significant half
v_addc_co_u32 v13, s[0:1], v15, v17, s[0:1] // (op 6898) most significant half



v_mul_lo_u32 v15, v13, 6 // (op 6898) most significant: low of high * low
v_mul_hi_u32 v16, v12, 6 // (op 6898) most significant: high of low * low
v_mul_lo_u32 v14, v12, 6 // (op 6898) least significant: low of low * low
v_add_u32 v15, v15, v16 // (op 6898) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[14:15] // (op 6898) 

v_mov_b32 v14, v12 // (op 6898) convert

s_mul_i32 s1, s7, 6 // (op 6900) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 6900) least significant
s_mul_hi_u32 s47, s6, 6 // (op 6900) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 6900) most significant: sum

s_mul_i32 s19, s30, s1 // (op 6900) most significant: low of low * high
s_mul_i32 s47, s31, s0 // (op 6900) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 6900) least significant
s_mul_hi_u32 s22, s30, s0 // (op 6900) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 6900) most significant: sum
s_add_u32 s19, s19, s22 // (op 6900) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 6900) 

s_mov_b32 s60, s32 // (op 6900) reg expression
s_mov_b32 s61, s33 // (op 6900) reg expression
s_mov_b32 s62, s0 // (op 6900) reg expression
s_mov_b32 s63, 131072 // (op 6900) call()

s_mul_hi_i32 s47, s16, s10 // (op 6917) 
v_lshrrev_b32 v12, 1, v1 // (op 6917) 
s_add_i32 s0, s47, s16 // (op 6917) 

v_and_b32 v13, -2, v1 // (op 6917) 
s_ashr_i32 s1, s0, 31 // (op 6917) 
v_sub_u32 v15, v1, v13 // (op 6917) 

s_cmp_eq_i32 s10, 0 // (op 6917) 
s_mov_b32 s47, scc // (op 6917) 
v_lshlrev_b32 v13, 6, v15 // (op 6917) 

s_cmp_eq_i32 s47, 0 // (op 6917) 
s_cselect_b32 s18, 0, -1 // (op 6917) 

v_mov_b32 v17, 0 // (op 6917) convert
v_mov_b32 v16, v13 // (op 6917) convert

s_and_b32 s19, s13, 31 // (op 6917) 
s_lshl_b32 s47, 1, s19 // (op 6917) 
s_add_i32 s22, s47, s18 // (op 6917) 


s_and_b32 s18, s1, s22 // (op 6917) 


s_add_i32 s1, s0, s18 // (op 6917) 


s_ashr_i32 s0, s1, s19 // (op 6917) 


s_lshl_b32 s1, s13, 24 // (op 6917) 
s_ashr_i32 s47, s1, 31 // (op 6917) 

s_xor_b32 s1, s0, s47 // (op 6917) 

s_sub_i32 s0, s1, s47 // (op 6917) 


s_mul_hi_i32 s1, s0, s14 // (op 6917) 
s_add_i32 s47, s1, s0 // (op 6917) 

s_ashr_i32 s1, s47, 31 // (op 6917) 
s_cmp_eq_i32 s14, 0 // (op 6917) 
s_mov_b32 s18, scc // (op 6917) 
s_cmp_eq_i32 s18, 0 // (op 6917) 
s_cselect_b32 s19, 0, -1 // (op 6917) 

s_and_b32 s18, s15, 31 // (op 6917) 
s_lshl_b32 s22, 1, s18 // (op 6917) 
s_add_i32 s23, s22, s19 // (op 6917) 


s_and_b32 s19, s1, s23 // (op 6917) 


s_add_i32 s1, s47, s19 // (op 6917) 


s_ashr_i32 s19, s1, s18 // (op 6917) 


s_lshl_b32 s1, s15, 24 // (op 6917) 
s_ashr_i32 s18, s1, 31 // (op 6917) 

s_xor_b32 s1, s19, s18 // (op 6917) 

s_sub_i32 s19, s1, s18 // (op 6917) 


s_add_u32 s22, s6, 128 // (op 6917) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 6917) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 6917) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 6917) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 6917) 

s_mul_i32 s1, s19, s22 // (op 6917) 

s_sub_i32 s47, s0, s1 // (op 6917) 


s_mul_hi_i32 s0, s16, s9 // (op 6917) 
s_add_i32 s1, s0, s16 // (op 6917) 

s_ashr_i32 s0, s1, 31 // (op 6917) 
s_cmp_eq_i32 s9, 0 // (op 6917) 
s_mov_b32 s18, scc // (op 6917) 
s_cmp_eq_i32 s18, 0 // (op 6917) 
s_cselect_b32 s19, 0, -1 // (op 6917) 

s_and_b32 s18, s45, 31 // (op 6917) 
s_lshl_b32 s24, 1, s18 // (op 6917) 
s_add_i32 s25, s24, s19 // (op 6917) 


s_and_b32 s19, s0, s25 // (op 6917) 


s_add_i32 s0, s1, s19 // (op 6917) 


s_ashr_i32 s1, s0, s18 // (op 6917) 


s_lshl_b32 s0, s45, 24 // (op 6917) 
s_ashr_i32 s18, s0, 31 // (op 6917) 

s_xor_b32 s0, s1, s18 // (op 6917) 

s_sub_i32 s1, s0, s18 // (op 6917) 


s_mul_i32 s0, s11, s22 // (op 6917) 

s_mul_i32 s18, s8, s0 // (op 6917) 

s_mul_i32 s0, s1, s18 // (op 6917) 

s_sub_u32 s18, s16, s0 // (op 6917) 

s_mul_hi_i32 s0, s18, s12 // (op 6917) 
s_add_i32 s19, s0, s18 // (op 6917) 


s_ashr_i32 s0, s19, 31 // (op 6917) 
s_cmp_eq_i32 s12, 0 // (op 6917) 
s_mov_b32 s18, scc // (op 6917) 
s_cmp_eq_i32 s18, 0 // (op 6917) 
s_cselect_b32 s22, 0, -1 // (op 6917) 

s_and_b32 s18, s44, 31 // (op 6917) 
s_lshl_b32 s23, 1, s18 // (op 6917) 
s_add_i32 s24, s23, s22 // (op 6917) 


s_and_b32 s25, s0, s24 // (op 6917) 


s_add_i32 s0, s19, s25 // (op 6917) 


s_ashr_i32 s19, s0, s18 // (op 6917) 


s_lshl_b32 s0, s44, 24 // (op 6917) 
s_ashr_i32 s18, s0, 31 // (op 6917) 

s_xor_b32 s0, s19, s18 // (op 6917) 

s_sub_i32 s19, s0, s18 // (op 6917) 


s_cmp_eq_i32 s1, 0 // (op 6917) 
s_mov_b32 s0, scc // (op 6917) 

s_cmp_eq_i32 s0, 0 // (op 6917) 
s_cselect_b32 s1, s19, s47 // (op 6917) 



v_lshl_add_u32 v13, s1, 7, v12 // (op 6917) 


v_mul_lo_u32 v19, v13, s31 // (op 6917) most significant: low of low * high
v_mul_hi_u32 v12, v13, s30 // (op 6917) most significant: high of low * low
v_mul_lo_u32 v18, v13, s30 // (op 6917) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 6917) most significant: sum


v_add_co_u32 v12, s[0:1], v16, v18 // (op 6917) least significant half
v_addc_co_u32 v13, s[0:1], v17, v19, s[0:1] // (op 6917) most significant half



v_mul_lo_u32 v17, v13, 6 // (op 6917) most significant: low of high * low
v_mul_hi_u32 v15, v12, 6 // (op 6917) most significant: high of low * low
v_mul_lo_u32 v16, v12, 6 // (op 6917) least significant: low of low * low
v_add_u32 v17, v17, v15 // (op 6917) most significant: sum


v_lshrrev_b64 v[12:13], 3, v[16:17] // (op 6917) 

v_mov_b32 v15, v12 // (op 6917) convert

s_mul_i32 s1, s7, 6 // (op 6919) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 6919) least significant
s_mul_hi_u32 s47, s6, 6 // (op 6919) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 6919) most significant: sum

s_mul_i32 s19, s30, s1 // (op 6919) most significant: low of low * high
s_mul_i32 s47, s31, s0 // (op 6919) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 6919) least significant
s_mul_hi_u32 s22, s30, s0 // (op 6919) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 6919) most significant: sum
s_add_u32 s19, s19, s22 // (op 6919) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 6919) 

s_mov_b32 s60, s32 // (op 6919) reg expression
s_mov_b32 s61, s33 // (op 6919) reg expression
s_mov_b32 s62, s0 // (op 6919) reg expression
s_mov_b32 s63, 131072 // (op 6919) call()

s_mul_hi_i32 s47, s16, s10 // (op 7167) 
v_and_b32 v12, 63, v1 // (op 7167) 
s_add_i32 s0, s47, s16 // (op 7167) 

v_lshrrev_b32 v13, 4, v12 // (op 7167) 
s_ashr_i32 s1, s0, 31 // (op 7167) 
v_and_b32 v16, 3, v13 // (op 7167) 

s_cmp_eq_i32 s10, 0 // (op 7167) 
s_mov_b32 s47, scc // (op 7167) 
v_and_b32 v13, 15, v1 // (op 7167) 
s_cmp_eq_i32 s47, 0 // (op 7167) 
s_cselect_b32 s18, 0, -1 // (op 7167) 

v_lshl_add_u32 v17, v16, 4, v13 // (op 7167) 


s_and_b32 s19, s13, 31 // (op 7167) 
v_lshrrev_b32 v13, 7, v1 // (op 7167) 
s_lshl_b32 s47, 1, s19 // (op 7167) 
v_lshl_add_u32 v16, v13, 6, v17 // (op 7167) 


s_add_i32 s22, s47, s18 // (op 7167) 


v_lshrrev_b32 v13, 6, v12 // (op 7167) 

s_and_b32 s18, s1, s22 // (op 7167) 


v_lshlrev_b32 v12, 2, v13 // (op 7167) 

s_add_i32 s1, s0, s18 // (op 7167) 


s_ashr_i32 s0, s1, s19 // (op 7167) 


s_lshl_b32 s1, s13, 24 // (op 7167) 
s_ashr_i32 s47, s1, 31 // (op 7167) 

s_xor_b32 s1, s0, s47 // (op 7167) 

s_sub_i32 s0, s1, s47 // (op 7167) 


s_mul_hi_i32 s1, s0, s14 // (op 7167) 
s_add_i32 s47, s1, s0 // (op 7167) 

s_ashr_i32 s1, s47, 31 // (op 7167) 
s_cmp_eq_i32 s14, 0 // (op 7167) 
s_mov_b32 s18, scc // (op 7167) 
s_cmp_eq_i32 s18, 0 // (op 7167) 
s_cselect_b32 s19, 0, -1 // (op 7167) 

s_and_b32 s18, s15, 31 // (op 7167) 
s_lshl_b32 s22, 1, s18 // (op 7167) 
s_add_i32 s23, s22, s19 // (op 7167) 


s_and_b32 s19, s1, s23 // (op 7167) 


s_add_i32 s1, s47, s19 // (op 7167) 


s_ashr_i32 s19, s1, s18 // (op 7167) 


s_lshl_b32 s1, s15, 24 // (op 7167) 
s_ashr_i32 s18, s1, 31 // (op 7167) 

s_xor_b32 s1, s19, s18 // (op 7167) 

s_sub_i32 s19, s1, s18 // (op 7167) 


s_mul_i32 s1, s0, s11 // (op 7167) 

s_sub_u32 s0, s16, s1 // (op 7167) 

s_mul_i32 s1, s19, s11 // (op 7167) 
s_add_u32 s1, s1, s0 // (op 7167) 


s_mul_hi_i32 s0, s16, s9 // (op 7167) 
s_add_i32 s47, s0, s16 // (op 7167) 

s_ashr_i32 s0, s47, 31 // (op 7167) 
s_cmp_eq_i32 s9, 0 // (op 7167) 
s_mov_b32 s18, scc // (op 7167) 
s_cmp_eq_i32 s18, 0 // (op 7167) 
s_cselect_b32 s19, 0, -1 // (op 7167) 

s_and_b32 s18, s45, 31 // (op 7167) 
s_lshl_b32 s22, 1, s18 // (op 7167) 
s_add_i32 s23, s22, s19 // (op 7167) 


s_and_b32 s19, s0, s23 // (op 7167) 


s_add_i32 s0, s47, s19 // (op 7167) 


s_ashr_i32 s19, s0, s18 // (op 7167) 


s_lshl_b32 s0, s45, 24 // (op 7167) 
s_ashr_i32 s18, s0, 31 // (op 7167) 

s_xor_b32 s0, s19, s18 // (op 7167) 

s_sub_i32 s19, s0, s18 // (op 7167) 


s_add_u32 s22, s6, 128 // (op 7167) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 7167) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 7167) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 7167) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 7167) 

s_mul_i32 s0, s11, s22 // (op 7167) 

s_mul_i32 s18, s8, s0 // (op 7167) 

s_mul_i32 s0, s19, s18 // (op 7167) 

s_sub_u32 s18, s16, s0 // (op 7167) 

s_mov_b32 s0, s18 // (op 7167) 
s_mul_hi_i32 s47, s0, s12 // (op 7167) 
s_add_i32 s22, s47, s0 // (op 7167) 


s_ashr_i32 s0, s22, 31 // (op 7167) 
s_cmp_eq_i32 s12, 0 // (op 7167) 
s_mov_b32 s47, scc // (op 7167) 
s_cmp_eq_i32 s47, 0 // (op 7167) 
s_cselect_b32 s23, 0, -1 // (op 7167) 

s_and_b32 s47, s44, 31 // (op 7167) 
s_lshl_b32 s24, 1, s47 // (op 7167) 
s_add_i32 s25, s24, s23 // (op 7167) 


s_and_b32 s23, s0, s25 // (op 7167) 


s_add_i32 s0, s22, s23 // (op 7167) 


s_ashr_i32 s22, s0, s47 // (op 7167) 


s_lshl_b32 s0, s44, 24 // (op 7167) 
s_ashr_i32 s47, s0, 31 // (op 7167) 

s_xor_b32 s0, s22, s47 // (op 7167) 

s_sub_i32 s22, s0, s47 // (op 7167) 


s_mul_i32 s0, s46, s11 // (op 7167) 
s_add_u32 s24, s20, 128 // (op 7167) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 7167) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 7167) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 7167) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 7 // (op 7167) 

s_sub_i32 s23, s24, s0 // (op 7167) 

s_mul_i32 s47, s22, s23 // (op 7167) 


s_sub_u32 s22, s18, s47 // (op 7167) 


s_add_u32 s18, s22, s0 // (op 7167) 


s_cmp_eq_i32 s19, 0 // (op 7167) 
s_mov_b32 s0, scc // (op 7167) 

s_cmp_eq_i32 s0, 0 // (op 7167) 
s_cselect_b32 s19, s18, s1 // (op 7167) 



v_lshl_add_u32 v13, s19, 7, v16 // (op 7167) 


v_mul_lo_u32 v16, v13, s34 // (op 7167) 
v_add_u32 v16, v16, v12 // (op 7167) 


s_mul_i32 s1, s34, s21 // (op 7169) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 7169) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 7169) least significant
s_mul_hi_u32 s18, s34, s20 // (op 7169) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 7169) most significant: sum
s_add_u32 s1, s1, s18 // (op 7169) most significant: sum


s_mov_b32 s64, s36 // (op 7169) reg expression
s_mov_b32 s65, s37 // (op 7169) reg expression
s_mov_b32 s66, s0 // (op 7169) reg expression
s_mov_b32 s67, 131072 // (op 7169) call()

s_mul_hi_i32 s47, s16, s10 // (op 7188) 
v_and_b32 v12, 63, v1 // (op 7188) 
s_add_i32 s0, s47, s16 // (op 7188) 

v_lshrrev_b32 v13, 4, v12 // (op 7188) 
s_ashr_i32 s1, s0, 31 // (op 7188) 
v_and_b32 v17, 3, v13 // (op 7188) 

s_cmp_eq_i32 s10, 0 // (op 7188) 
s_mov_b32 s47, scc // (op 7188) 
v_and_b32 v13, 15, v1 // (op 7188) 
s_cmp_eq_i32 s47, 0 // (op 7188) 
s_cselect_b32 s18, 0, -1 // (op 7188) 

v_lshl_add_u32 v18, v17, 4, v13 // (op 7188) 


s_and_b32 s19, s13, 31 // (op 7188) 
v_lshrrev_b32 v13, 7, v1 // (op 7188) 
s_lshl_b32 s47, 1, s19 // (op 7188) 
v_lshl_add_u32 v17, v13, 6, v18 // (op 7188) 


s_add_i32 s22, s47, s18 // (op 7188) 


v_lshrrev_b32 v13, 6, v12 // (op 7188) 

s_and_b32 s18, s1, s22 // (op 7188) 


v_lshlrev_b32 v12, 2, v13 // (op 7188) 

s_add_i32 s1, s0, s18 // (op 7188) 


s_ashr_i32 s0, s1, s19 // (op 7188) 


s_lshl_b32 s1, s13, 24 // (op 7188) 
s_ashr_i32 s47, s1, 31 // (op 7188) 

s_xor_b32 s1, s0, s47 // (op 7188) 

s_sub_i32 s0, s1, s47 // (op 7188) 


s_mul_hi_i32 s1, s0, s14 // (op 7188) 
s_add_i32 s47, s1, s0 // (op 7188) 

s_ashr_i32 s1, s47, 31 // (op 7188) 
s_cmp_eq_i32 s14, 0 // (op 7188) 
s_mov_b32 s18, scc // (op 7188) 
s_cmp_eq_i32 s18, 0 // (op 7188) 
s_cselect_b32 s19, 0, -1 // (op 7188) 

s_and_b32 s18, s15, 31 // (op 7188) 
s_lshl_b32 s22, 1, s18 // (op 7188) 
s_add_i32 s23, s22, s19 // (op 7188) 


s_and_b32 s19, s1, s23 // (op 7188) 


s_add_i32 s1, s47, s19 // (op 7188) 


s_ashr_i32 s19, s1, s18 // (op 7188) 


s_lshl_b32 s1, s15, 24 // (op 7188) 
s_ashr_i32 s18, s1, 31 // (op 7188) 

s_xor_b32 s1, s19, s18 // (op 7188) 

s_sub_i32 s19, s1, s18 // (op 7188) 


s_mul_i32 s1, s0, s11 // (op 7188) 

s_sub_u32 s0, s16, s1 // (op 7188) 

s_mul_i32 s1, s19, s11 // (op 7188) 
s_add_u32 s1, s1, s0 // (op 7188) 


s_mul_hi_i32 s0, s16, s9 // (op 7188) 
s_add_i32 s47, s0, s16 // (op 7188) 

s_ashr_i32 s0, s47, 31 // (op 7188) 
s_cmp_eq_i32 s9, 0 // (op 7188) 
s_mov_b32 s18, scc // (op 7188) 
s_cmp_eq_i32 s18, 0 // (op 7188) 
s_cselect_b32 s19, 0, -1 // (op 7188) 

s_and_b32 s18, s45, 31 // (op 7188) 
s_lshl_b32 s22, 1, s18 // (op 7188) 
s_add_i32 s23, s22, s19 // (op 7188) 


s_and_b32 s19, s0, s23 // (op 7188) 


s_add_i32 s0, s47, s19 // (op 7188) 


s_ashr_i32 s19, s0, s18 // (op 7188) 


s_lshl_b32 s0, s45, 24 // (op 7188) 
s_ashr_i32 s18, s0, 31 // (op 7188) 

s_xor_b32 s0, s19, s18 // (op 7188) 

s_sub_i32 s19, s0, s18 // (op 7188) 


s_add_u32 s22, s6, 128 // (op 7188) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 7188) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 7188) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 7188) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 7188) 

s_mul_i32 s0, s11, s22 // (op 7188) 

s_mul_i32 s18, s8, s0 // (op 7188) 

s_mul_i32 s0, s19, s18 // (op 7188) 

s_sub_u32 s18, s16, s0 // (op 7188) 

s_mov_b32 s0, s18 // (op 7188) 
s_mul_hi_i32 s47, s0, s12 // (op 7188) 
s_add_i32 s22, s47, s0 // (op 7188) 


s_ashr_i32 s0, s22, 31 // (op 7188) 
s_cmp_eq_i32 s12, 0 // (op 7188) 
s_mov_b32 s47, scc // (op 7188) 
s_cmp_eq_i32 s47, 0 // (op 7188) 
s_cselect_b32 s23, 0, -1 // (op 7188) 

s_and_b32 s47, s44, 31 // (op 7188) 
s_lshl_b32 s24, 1, s47 // (op 7188) 
s_add_i32 s25, s24, s23 // (op 7188) 


s_and_b32 s23, s0, s25 // (op 7188) 


s_add_i32 s0, s22, s23 // (op 7188) 


s_ashr_i32 s22, s0, s47 // (op 7188) 


s_lshl_b32 s0, s44, 24 // (op 7188) 
s_ashr_i32 s47, s0, 31 // (op 7188) 

s_xor_b32 s0, s22, s47 // (op 7188) 

s_sub_i32 s22, s0, s47 // (op 7188) 


s_mul_i32 s0, s46, s11 // (op 7188) 
s_add_u32 s24, s20, 128 // (op 7188) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 7188) End of Int64 add, unlocking SCC
s_sub_u32 s68, s24, 1 // (op 7188) least significant half; sets scc
s_subb_u32 s69, s25, 0 // (op 7188) most significant half; uses scc

s_ashr_i64 s[24:25], s[68:69], 7 // (op 7188) 

s_sub_i32 s23, s24, s0 // (op 7188) 

s_mul_i32 s47, s22, s23 // (op 7188) 


s_sub_u32 s22, s18, s47 // (op 7188) 


s_add_u32 s18, s22, s0 // (op 7188) 


s_cmp_eq_i32 s19, 0 // (op 7188) 
s_mov_b32 s0, scc // (op 7188) 

s_cmp_eq_i32 s0, 0 // (op 7188) 
s_cselect_b32 s19, s18, s1 // (op 7188) 



v_lshl_add_u32 v13, s19, 7, v17 // (op 7188) 


v_mul_lo_u32 v17, v13, s34 // (op 7188) 
v_add_u32 v17, v17, v12 // (op 7188) 


s_mul_i32 s1, s34, s21 // (op 7190) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 7190) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 7190) least significant
s_mul_hi_u32 s18, s34, s20 // (op 7190) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 7190) most significant: sum
s_add_u32 s1, s1, s18 // (op 7190) most significant: sum


s_mov_b32 s64, s36 // (op 7190) reg expression
s_mov_b32 s65, s37 // (op 7190) reg expression
s_mov_b32 s66, s0 // (op 7190) reg expression
s_mov_b32 s67, 131072 // (op 7190) call()

s_mul_hi_i32 s47, s16, s10 // (op 7219) 
v_and_b32 v12, 63, v1 // (op 7219) 
s_add_i32 s0, s47, s16 // (op 7219) 

v_lshrrev_b32 v13, 4, v12 // (op 7219) 
s_ashr_i32 s1, s0, 31 // (op 7219) 
v_and_b32 v18, 3, v13 // (op 7219) 

s_cmp_eq_i32 s10, 0 // (op 7219) 
s_mov_b32 s47, scc // (op 7219) 
v_and_b32 v13, 15, v1 // (op 7219) 
s_cmp_eq_i32 s47, 0 // (op 7219) 
s_cselect_b32 s18, 0, -1 // (op 7219) 

v_lshl_add_u32 v19, v18, 4, v13 // (op 7219) 


s_and_b32 s19, s13, 31 // (op 7219) 
v_lshrrev_b32 v13, 6, v1 // (op 7219) 
s_lshl_b32 s47, 1, s19 // (op 7219) 
v_and_b32 v18, 1, v13 // (op 7219) 

s_add_i32 s22, s47, s18 // (op 7219) 


v_lshl_add_u32 v13, v18, 6, v19 // (op 7219) 


s_and_b32 s18, s1, s22 // (op 7219) 


v_lshrrev_b32 v18, 6, v12 // (op 7219) 

s_add_i32 s1, s0, s18 // (op 7219) 


s_ashr_i32 s0, s1, s19 // (op 7219) 


s_lshl_b32 s1, s13, 24 // (op 7219) 
s_ashr_i32 s47, s1, 31 // (op 7219) 

s_xor_b32 s1, s0, s47 // (op 7219) 

s_sub_i32 s0, s1, s47 // (op 7219) 


s_mul_hi_i32 s1, s0, s14 // (op 7219) 
s_add_i32 s47, s1, s0 // (op 7219) 

s_ashr_i32 s1, s47, 31 // (op 7219) 
s_cmp_eq_i32 s14, 0 // (op 7219) 
s_mov_b32 s18, scc // (op 7219) 
s_cmp_eq_i32 s18, 0 // (op 7219) 
s_cselect_b32 s19, 0, -1 // (op 7219) 

s_and_b32 s18, s15, 31 // (op 7219) 
s_lshl_b32 s22, 1, s18 // (op 7219) 
s_add_i32 s23, s22, s19 // (op 7219) 


s_and_b32 s19, s1, s23 // (op 7219) 


s_add_i32 s1, s47, s19 // (op 7219) 


s_ashr_i32 s19, s1, s18 // (op 7219) 


s_lshl_b32 s1, s15, 24 // (op 7219) 
s_ashr_i32 s18, s1, 31 // (op 7219) 

s_xor_b32 s1, s19, s18 // (op 7219) 

s_sub_i32 s19, s1, s18 // (op 7219) 


s_add_u32 s22, s6, 128 // (op 7219) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 7219) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 7219) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 7219) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 7219) 

s_mul_i32 s1, s19, s22 // (op 7219) 

s_sub_i32 s47, s0, s1 // (op 7219) 


s_mul_hi_i32 s0, s16, s9 // (op 7219) 
s_add_i32 s1, s0, s16 // (op 7219) 

s_ashr_i32 s0, s1, 31 // (op 7219) 
s_cmp_eq_i32 s9, 0 // (op 7219) 
s_mov_b32 s18, scc // (op 7219) 
s_cmp_eq_i32 s18, 0 // (op 7219) 
s_cselect_b32 s19, 0, -1 // (op 7219) 

s_and_b32 s18, s45, 31 // (op 7219) 
s_lshl_b32 s24, 1, s18 // (op 7219) 
s_add_i32 s25, s24, s19 // (op 7219) 


s_and_b32 s19, s0, s25 // (op 7219) 


s_add_i32 s0, s1, s19 // (op 7219) 


s_ashr_i32 s1, s0, s18 // (op 7219) 


s_lshl_b32 s0, s45, 24 // (op 7219) 
s_ashr_i32 s18, s0, 31 // (op 7219) 

s_xor_b32 s0, s1, s18 // (op 7219) 

s_sub_i32 s1, s0, s18 // (op 7219) 


s_mul_i32 s0, s11, s22 // (op 7219) 

s_mul_i32 s18, s8, s0 // (op 7219) 

s_mul_i32 s0, s1, s18 // (op 7219) 

s_sub_u32 s18, s16, s0 // (op 7219) 

s_mul_hi_i32 s0, s18, s12 // (op 7219) 
s_add_i32 s19, s0, s18 // (op 7219) 


s_ashr_i32 s0, s19, 31 // (op 7219) 
s_cmp_eq_i32 s12, 0 // (op 7219) 
s_mov_b32 s18, scc // (op 7219) 
s_cmp_eq_i32 s18, 0 // (op 7219) 
s_cselect_b32 s22, 0, -1 // (op 7219) 

s_and_b32 s18, s44, 31 // (op 7219) 
s_lshl_b32 s23, 1, s18 // (op 7219) 
s_add_i32 s24, s23, s22 // (op 7219) 


s_and_b32 s25, s0, s24 // (op 7219) 


s_add_i32 s0, s19, s25 // (op 7219) 


s_ashr_i32 s19, s0, s18 // (op 7219) 


s_lshl_b32 s0, s44, 24 // (op 7219) 
s_ashr_i32 s18, s0, 31 // (op 7219) 

s_xor_b32 s0, s19, s18 // (op 7219) 

s_sub_i32 s19, s0, s18 // (op 7219) 


s_cmp_eq_i32 s1, 0 // (op 7219) 
s_mov_b32 s0, scc // (op 7219) 

s_cmp_eq_i32 s0, 0 // (op 7219) 
s_cselect_b32 s1, s19, s47 // (op 7219) 



v_lshl_add_u32 v12, s1, 7, v13 // (op 7219) 


v_mul_lo_u32 v13, v12, s38 // (op 7219) 

v_lshl_add_u32 v12, v18, 2, v13 // (op 7219) 


s_mul_i32 s1, s38, s7 // (op 7221) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 7221) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 7221) least significant
s_mul_hi_u32 s18, s38, s6 // (op 7221) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 7221) most significant: sum
s_add_u32 s1, s1, s18 // (op 7221) most significant: sum


s_mov_b32 s68, s40 // (op 7221) reg expression
s_mov_b32 s69, s41 // (op 7221) reg expression
s_mov_b32 s70, s0 // (op 7221) reg expression
s_mov_b32 s71, 131072 // (op 7221) call()

s_mul_hi_i32 s47, s16, s10 // (op 7240) 
v_and_b32 v13, 63, v1 // (op 7240) 
s_add_i32 s0, s47, s16 // (op 7240) 

v_lshrrev_b32 v18, 4, v13 // (op 7240) 
s_ashr_i32 s1, s0, 31 // (op 7240) 
v_and_b32 v19, 3, v18 // (op 7240) 

s_cmp_eq_i32 s10, 0 // (op 7240) 
s_mov_b32 s47, scc // (op 7240) 
v_and_b32 v18, 15, v1 // (op 7240) 
s_cmp_eq_i32 s47, 0 // (op 7240) 
s_cselect_b32 s18, 0, -1 // (op 7240) 

v_lshl_add_u32 v20, v19, 4, v18 // (op 7240) 


s_and_b32 s19, s13, 31 // (op 7240) 
v_lshrrev_b32 v18, 6, v1 // (op 7240) 
s_lshl_b32 s47, 1, s19 // (op 7240) 
v_and_b32 v19, 1, v18 // (op 7240) 

s_add_i32 s22, s47, s18 // (op 7240) 


v_lshl_add_u32 v18, v19, 6, v20 // (op 7240) 


s_and_b32 s18, s1, s22 // (op 7240) 


v_lshrrev_b32 v19, 6, v13 // (op 7240) 

s_add_i32 s1, s0, s18 // (op 7240) 


s_ashr_i32 s0, s1, s19 // (op 7240) 


s_lshl_b32 s1, s13, 24 // (op 7240) 
s_ashr_i32 s47, s1, 31 // (op 7240) 

s_xor_b32 s1, s0, s47 // (op 7240) 

s_sub_i32 s0, s1, s47 // (op 7240) 


s_mul_hi_i32 s1, s0, s14 // (op 7240) 
s_add_i32 s47, s1, s0 // (op 7240) 

s_ashr_i32 s1, s47, 31 // (op 7240) 
s_cmp_eq_i32 s14, 0 // (op 7240) 
s_mov_b32 s18, scc // (op 7240) 
s_cmp_eq_i32 s18, 0 // (op 7240) 
s_cselect_b32 s19, 0, -1 // (op 7240) 

s_and_b32 s18, s15, 31 // (op 7240) 
s_lshl_b32 s22, 1, s18 // (op 7240) 
s_add_i32 s23, s22, s19 // (op 7240) 


s_and_b32 s19, s1, s23 // (op 7240) 


s_add_i32 s1, s47, s19 // (op 7240) 


s_ashr_i32 s19, s1, s18 // (op 7240) 


s_lshl_b32 s1, s15, 24 // (op 7240) 
s_ashr_i32 s18, s1, 31 // (op 7240) 

s_xor_b32 s1, s19, s18 // (op 7240) 

s_sub_i32 s19, s1, s18 // (op 7240) 


s_add_u32 s22, s6, 128 // (op 7240) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 7240) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 7240) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 7240) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 7240) 

s_mul_i32 s1, s19, s22 // (op 7240) 

s_sub_i32 s47, s0, s1 // (op 7240) 


s_mul_hi_i32 s0, s16, s9 // (op 7240) 
s_add_i32 s1, s0, s16 // (op 7240) 

s_ashr_i32 s0, s1, 31 // (op 7240) 
s_cmp_eq_i32 s9, 0 // (op 7240) 
s_mov_b32 s18, scc // (op 7240) 
s_cmp_eq_i32 s18, 0 // (op 7240) 
s_cselect_b32 s19, 0, -1 // (op 7240) 

s_and_b32 s18, s45, 31 // (op 7240) 
s_lshl_b32 s24, 1, s18 // (op 7240) 
s_add_i32 s25, s24, s19 // (op 7240) 


s_and_b32 s19, s0, s25 // (op 7240) 


s_add_i32 s0, s1, s19 // (op 7240) 


s_ashr_i32 s1, s0, s18 // (op 7240) 


s_lshl_b32 s0, s45, 24 // (op 7240) 
s_ashr_i32 s18, s0, 31 // (op 7240) 

s_xor_b32 s0, s1, s18 // (op 7240) 

s_sub_i32 s1, s0, s18 // (op 7240) 


s_mul_i32 s0, s11, s22 // (op 7240) 

s_mul_i32 s18, s8, s0 // (op 7240) 

s_mul_i32 s0, s1, s18 // (op 7240) 

s_sub_u32 s18, s16, s0 // (op 7240) 

s_mul_hi_i32 s0, s18, s12 // (op 7240) 
s_add_i32 s19, s0, s18 // (op 7240) 


s_ashr_i32 s0, s19, 31 // (op 7240) 
s_cmp_eq_i32 s12, 0 // (op 7240) 
s_mov_b32 s18, scc // (op 7240) 
s_cmp_eq_i32 s18, 0 // (op 7240) 
s_cselect_b32 s22, 0, -1 // (op 7240) 

s_and_b32 s18, s44, 31 // (op 7240) 
s_lshl_b32 s23, 1, s18 // (op 7240) 
s_add_i32 s24, s23, s22 // (op 7240) 


s_and_b32 s25, s0, s24 // (op 7240) 


s_add_i32 s0, s19, s25 // (op 7240) 


s_ashr_i32 s19, s0, s18 // (op 7240) 


s_lshl_b32 s0, s44, 24 // (op 7240) 
s_ashr_i32 s18, s0, 31 // (op 7240) 

s_xor_b32 s0, s19, s18 // (op 7240) 

s_sub_i32 s19, s0, s18 // (op 7240) 


s_cmp_eq_i32 s1, 0 // (op 7240) 
s_mov_b32 s0, scc // (op 7240) 

s_cmp_eq_i32 s0, 0 // (op 7240) 
s_cselect_b32 s1, s19, s47 // (op 7240) 



v_lshl_add_u32 v13, s1, 7, v18 // (op 7240) 


v_mul_lo_u32 v18, v13, s38 // (op 7240) 

v_lshl_add_u32 v13, v19, 2, v18 // (op 7240) 


s_mul_i32 s1, s38, s7 // (op 7242) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 7242) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 7242) least significant
s_mul_hi_u32 s18, s38, s6 // (op 7242) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 7242) most significant: sum
s_add_u32 s1, s1, s18 // (op 7242) most significant: sum


s_mov_b32 s68, s40 // (op 7242) reg expression
s_mov_b32 s69, s41 // (op 7242) reg expression
s_mov_b32 s70, s0 // (op 7242) reg expression
s_mov_b32 s71, 131072 // (op 7242) call()




buffer_load_dwordx4 v[18:21], v14, s[60:63], 0 offen // (op 6473) Load value
buffer_load_dwordx2 v[22:23], v14, s[60:63], 0 offen offset:16 // (op 6473) Load value
buffer_load_dwordx4 v[24:27], v14, s[60:63], 0 offen offset:24 // (op 6473) Load value
buffer_load_dwordx2 v[28:29], v14, s[60:63], 0 offen offset:40 // (op 6473) Load value



buffer_load_dwordx4 v[30:33], v10, s[48:51], 0 offen // (op 6476) Load value
buffer_load_dwordx2 v[34:35], v10, s[48:51], 0 offen offset:16 // (op 6476) Load value
buffer_load_dwordx4 v[36:39], v10, s[48:51], 0 offen offset:24 // (op 6476) Load value
buffer_load_dwordx2 v[40:41], v10, s[48:51], 0 offen offset:40 // (op 6476) Load value


v_mov_b32 v42, v12 // (op 76) reg expression

buffer_load_dwordx2 v[44:45], v42, s[68:71], 0 offen // (op 76) Load value




v_mov_b32 v12, v16 // (op 72) reg expression

buffer_load_dwordx2 v[42:43], v12, s[64:67], 0 offen // (op 72) Load value





buffer_load_dwordx4 v[46:49], v14, s[60:63], 0 offen offset:96 // (op 6479) Load value
buffer_load_dwordx2 v[50:51], v14, s[60:63], 0 offen offset:112 // (op 6479) Load value
buffer_load_dwordx4 v[52:55], v14, s[60:63], 0 offen offset:120 // (op 6479) Load value
buffer_load_dwordx2 v[56:57], v14, s[60:63], 0 offen offset:136 // (op 6479) Load value




buffer_load_dwordx4 v[58:61], v10, s[48:51], 0 offen offset:96 // (op 6482) Load value
buffer_load_dwordx2 v[62:63], v10, s[48:51], 0 offen offset:112 // (op 6482) Load value
buffer_load_dwordx4 v[64:67], v10, s[48:51], 0 offen offset:120 // (op 6482) Load value
buffer_load_dwordx2 v[68:69], v10, s[48:51], 0 offen offset:136 // (op 6482) Load value

v_and_b32 v10, -2, v1 // (op 7047) 
v_sub_u32 v12, v1, v10 // (op 7047) 

v_lshlrev_b32 v10, 6, v12 // (op 7047) 

v_lshrrev_b32 v12, 1, v1 // (op 7047) 
v_lshl_add_u32 v14, v12, 7, v10 // (op 7047) 


v_mul_lo_u32 v10, v14, 6 // (op 7047) 

v_lshrrev_b32 v12, 3, v10 // (op 7047) 




s_waitcnt vmcnt(17)
ds_write_b128 v12, v[18:21] // (op 6484) Store local data 
s_waitcnt vmcnt(16)
ds_write_b64 v12, v[22:23] offset:16 // (op 6484) Store local data 
s_waitcnt vmcnt(15)
ds_write_b128 v12, v[24:27] offset:24 // (op 6484) Store local data 
s_waitcnt vmcnt(14)
ds_write_b64 v12, v[28:29] offset:40 // (op 6484) Store local data 

v_and_b32 v10, -2, v1 // (op 6989) 
v_sub_u32 v12, v1, v10 // (op 6989) 

v_lshlrev_b32 v10, 6, v12 // (op 6989) 

v_lshrrev_b32 v12, 1, v1 // (op 6989) 
v_lshl_add_u32 v14, v12, 7, v10 // (op 6989) 


v_mul_lo_u32 v10, v14, 6 // (op 6989) 

v_lshrrev_b32 v12, 3, v10 // (op 6989) 




s_waitcnt vmcnt(13)
ds_write_b128 v12, v[30:33] offset:12288 // (op 6485) Store local data 
s_waitcnt vmcnt(12)
ds_write_b64 v12, v[34:35] offset:12304 // (op 6485) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v12, v[36:39] offset:12312 // (op 6485) Store local data 
s_waitcnt vmcnt(10)
ds_write_b64 v12, v[40:41] offset:12328 // (op 6485) Store local data 

v_and_b32 v10, -2, v1 // (op 6973) 
v_sub_u32 v12, v1, v10 // (op 6973) 

v_lshlrev_b32 v10, 6, v12 // (op 6973) 

v_lshrrev_b32 v12, 1, v1 // (op 6973) 
v_lshl_add_u32 v14, v12, 7, v10 // (op 6973) 


v_mul_lo_u32 v10, v14, 6 // (op 6973) 

v_lshrrev_b32 v12, 3, v10 // (op 6973) 

v_and_b32 v10, -2, v1 // (op 7031) 
v_sub_u32 v14, v1, v10 // (op 7031) 

v_lshlrev_b32 v10, 6, v14 // (op 7031) 

v_lshrrev_b32 v14, 1, v1 // (op 7031) 
v_lshl_add_u32 v16, v14, 7, v10 // (op 7031) 


v_mul_lo_u32 v10, v16, 6 // (op 7031) 

v_lshrrev_b32 v14, 3, v10 // (op 7031) 

v_and_b32 v10, -2, v1 // (op 7099) 
v_sub_u32 v16, v1, v10 // (op 7099) 

v_lshlrev_b32 v10, 6, v16 // (op 7099) 

v_lshrrev_b32 v16, 1, v1 // (op 7099) 
v_lshl_add_u32 v70, v16, 7, v10 // (op 7099) 


v_mul_lo_u32 v10, v70, 6 // (op 7099) 

v_lshrrev_b32 v16, 3, v10 // (op 7099) 

v_and_b32 v10, -2, v1 // (op 7128) 
v_sub_u32 v70, v1, v10 // (op 7128) 

v_lshlrev_b32 v10, 6, v70 // (op 7128) 

v_lshrrev_b32 v70, 1, v1 // (op 7128) 
v_lshl_add_u32 v71, v70, 7, v10 // (op 7128) 


v_mul_lo_u32 v10, v71, 6 // (op 7128) 

v_lshrrev_b32 v70, 3, v10 // (op 7128) 

s_mov_b32 s47, 0 // (op 2080) call()
s_ashr_i64 s[0:1], s[42:43], 7 // (op 2079) 
s_mov_b32 s18, s0 // (op 2079) convert

s_and_b32 s19, s18, 4294967294 // (op 2079) 

s_cmp_lt_u32 s47, s19 // (op 2079) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ForLoopBottom_KLoop_2079 // (op 2079) Condition: Top (jump to RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ForLoopBottom_KLoop_2079 if false)
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ForLoopTop_KLoop_2079:
s_add_u32 s47, s47, 2 // (op 2081) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 7384) 



ds_read_b128 v[72:75], v3 offset:12288 // (op 62) Load local data 
ds_read_b64 v[76:77], v3 offset:12304 // (op 62) Load local data 
s_barrier  // (op 7397) 



ds_read_b128 v[78:81], v4 // (op 67) Load local data 
ds_read_b64 v[82:83], v4 offset:16 // (op 67) Load local data 



ds_read_b128 v[84:87], v4 offset:1536 // (op 312) Load local data 


v_mov_b32 v10, v15 // (op 10) 

ds_read_b64 v[88:89], v4 offset:1552 // (op 312) Load local data 
s_waitcnt vmcnt(9)
v_mov_b32 v90, v44 // (op 6739) reg expression



v_mov_b32 v91, v45 // (op 6739) reg expression
s_waitcnt vmcnt(8)
v_mov_b32 v92, v42 // (op 6752) reg expression
v_mov_b32 v93, v43 // (op 6752) reg expression
ds_read_b128 v[94:97], v4 offset:3072 // (op 448) Load local data 
ds_read_b64 v[98:99], v4 offset:3088 // (op 448) Load local data 
v_bfe_u32 v100, v92, 0, 8 // (op 6114) 



v_bfe_u32 v101, v92, 8, 8 // (op 6114) 
v_bfe_u32 v102, v92, 16, 8 // (op 6114) 
v_bfe_u32 v103, v92, 24, 8 // (op 6114) 
v_permlane16_swap_b32 v100, v101 // (op 6114) 
ds_read_b128 v[104:107], v4 offset:4608 // (op 584) Load local data 
v_permlane16_swap_b32 v102, v103 // (op 6114) 
s_nop 1
v_permlane32_swap_b32 v100, v102 // (op 6114) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v101, v103 // (op 6114) 
buffer_load_dwordx4 v[18:21], v10, s[60:63], 0 offen offset:192 // (op 10) Load value
buffer_load_dwordx2 v[22:23], v10, s[60:63], 0 offen offset:208 // (op 10) Load value
v_bfe_u32 v110, v90, 0, 8 // (op 6177) 
buffer_load_dwordx4 v[24:27], v10, s[60:63], 0 offen offset:216 // (op 10) Load value
v_bfe_u32 v111, v90, 8, 8 // (op 6177) 
v_bfe_u32 v112, v90, 16, 8 // (op 6177) 
v_bfe_u32 v113, v90, 24, 8 // (op 6177) 
v_permlane16_swap_b32 v110, v111 // (op 6177) 
s_nop 0
v_permlane16_swap_b32 v112, v113 // (op 6177) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v110, v112 // (op 6177) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v111, v113 // (op 6177) 
ds_read_b64 v[108:109], v4 offset:4624 // (op 584) Load local data 



s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[72:77], v[78:83], a[0:3], v100, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 78) 
ds_read_b128 v[114:117], v3 offset:13824 // (op 889) Load local data 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:77], v[84:89], a[4:7], v100, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 315) 
ds_read_b64 v[118:119], v3 offset:13840 // (op 889) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[72:77], v[94:99], a[8:11], v100, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 451) 
ds_read_b128 v[120:123], v3 offset:15360 // (op 1499) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[72:77], v[104:109], a[12:15], v100, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 587) 
ds_read_b64 v[124:125], v3 offset:15376 // (op 1499) Load local data 




s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[114:119], v[84:89], a[20:23], v101, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 969) 
ds_read_b128 v[72:75], v3 offset:16896 // (op 2109) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[114:119], v[94:99], a[24:27], v101, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 931) 
ds_read_b64 v[76:77], v3 offset:16912 // (op 2109) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[114:119], v[104:109], a[28:31], v101, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 893) 
buffer_load_dwordx2 v[28:29], v10, s[60:63], 0 offen offset:232 // (op 10) Load value


v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[114:119], v[78:83], a[16:19], v101, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1007) 
v_mov_b32 v71, v11 // (op 4) 


buffer_load_dwordx4 v[30:33], v71, s[48:51], 0 offen offset:192 // (op 4) Load value
buffer_load_dwordx2 v[34:35], v71, s[48:51], 0 offen offset:208 // (op 4) Load value
s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[120:125], v[78:83], a[32:35], v102, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1617) 
buffer_load_dwordx4 v[36:39], v71, s[48:51], 0 offen offset:216 // (op 4) Load value
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[120:125], v[84:89], a[36:39], v102, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1579) 
buffer_load_dwordx2 v[40:41], v71, s[48:51], 0 offen offset:232 // (op 4) Load value


s_barrier  // (op 7420) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[120:125], v[94:99], a[40:43], v102, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1541) 



v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[120:125], v[104:109], a[44:47], v102, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1503) 
v_mov_b32 v114, v13 // (op 6747) 

v_add_u32 v115, 8, v114 // (op 6747) 

buffer_load_dwordx2 v[44:45], v115, s[68:71], 0 offen // (op 6747) Load value


v_add_u32 v13, 8, v13 // (op 7230) 
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[72:77], v[78:83], a[48:51], v103, v110 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2227) 



v_mov_b32 v78, v17 // (op 6760) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[72:77], v[84:89], a[52:55], v103, v111 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2189) 
v_add_u32 v79, 8, v78 // (op 6760) 


s_waitcnt vmcnt(16)
ds_write_b128 v70, v[46:49] offset:24576 // (op 222) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[72:77], v[94:99], a[56:59], v103, v112 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2151) 
s_waitcnt vmcnt(15)
ds_write_b64 v70, v[50:51] offset:24592 // (op 222) Store local data 

s_waitcnt vmcnt(14)
ds_write_b128 v70, v[52:55] offset:24600 // (op 222) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[72:77], v[104:109], a[60:63], v103, v113 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2113) 
s_waitcnt vmcnt(13)
ds_write_b64 v70, v[56:57] offset:24616 // (op 222) Store local data 




buffer_load_dwordx2 v[42:43], v79, s[64:67], 0 offen // (op 6760) Load value


s_barrier  // (op 7414) 
v_add_u32 v17, 8, v17 // (op 7178) 



s_waitcnt vmcnt(13)
ds_write_b128 v16, v[58:61] offset:36864 // (op 212) Store local data 
s_waitcnt vmcnt(12)
ds_write_b64 v16, v[62:63] offset:36880 // (op 212) Store local data 
s_waitcnt vmcnt(11)
ds_write_b128 v16, v[64:67] offset:36888 // (op 212) Store local data 
s_waitcnt vmcnt(10)
ds_write_b64 v16, v[68:69] offset:36904 // (op 212) Store local data 



buffer_load_dwordx4 v[46:49], v10, s[60:63], 0 offen offset:288 // (op 223) Load value
buffer_load_dwordx2 v[50:51], v10, s[60:63], 0 offen offset:304 // (op 223) Load value
s_waitcnt lgkmcnt(0)
s_barrier  // (op 7411) 
buffer_load_dwordx4 v[52:55], v10, s[60:63], 0 offen offset:312 // (op 223) Load value



ds_read_b128 v[72:75], v5 offset:36864 // (op 214) Load local data 
ds_read_b64 v[76:77], v5 offset:36880 // (op 214) Load local data 
buffer_load_dwordx2 v[56:57], v10, s[60:63], 0 offen offset:328 // (op 223) Load value

s_barrier  // (op 7417) 






buffer_load_dwordx4 v[58:61], v71, s[48:51], 0 offen offset:288 // (op 211) Load value
v_add_u32 v15, 192, v15 // (op 6909) 
ds_read_b128 v[78:81], v6 offset:24576 // (op 216) Load local data 
ds_read_b64 v[82:83], v6 offset:24592 // (op 216) Load local data 
buffer_load_dwordx2 v[62:63], v71, s[48:51], 0 offen offset:304 // (op 211) Load value



v_bfe_u32 v84, v93, 0, 8 // (op 6129) 
v_bfe_u32 v85, v93, 8, 8 // (op 6129) 
v_bfe_u32 v86, v93, 16, 8 // (op 6129) 
v_bfe_u32 v87, v93, 24, 8 // (op 6129) 
v_permlane16_swap_b32 v84, v85 // (op 6129) 
ds_read_b128 v[94:97], v6 offset:26112 // (op 324) Load local data 
ds_read_b64 v[98:99], v6 offset:26128 // (op 324) Load local data 
v_permlane16_swap_b32 v86, v87 // (op 6129) 



ds_read_b128 v[100:103], v6 offset:27648 // (op 460) Load local data 
s_nop 0
v_permlane32_swap_b32 v84, v86 // (op 6129) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v85, v87 // (op 6129) 
ds_read_b64 v[104:105], v6 offset:27664 // (op 460) Load local data 




buffer_load_dwordx4 v[64:67], v71, s[48:51], 0 offen offset:312 // (op 211) Load value
buffer_load_dwordx2 v[68:69], v71, s[48:51], 0 offen offset:328 // (op 211) Load value
v_bfe_u32 v106, v91, 0, 8 // (op 6192) 

v_bfe_u32 v107, v91, 8, 8 // (op 6192) 
s_barrier  // (op 7400) 
v_bfe_u32 v108, v91, 16, 8 // (op 6192) 



v_bfe_u32 v109, v91, 24, 8 // (op 6192) 
v_permlane16_swap_b32 v106, v107 // (op 6192) 
v_add_u32 v11, 192, v11 // (op 6845) 
v_permlane16_swap_b32 v108, v109 // (op 6192) 
ds_read_b128 v[110:113], v6 offset:29184 // (op 596) Load local data 
ds_read_b64 v[114:115], v6 offset:29200 // (op 596) Load local data 
v_permlane32_swap_b32 v106, v108 // (op 6192) 
v_permlane32_swap_b32 v107, v109 // (op 6192) 




s_barrier  // (op 7479) 
s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[72:77], v[78:83], a[0:3], v84, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 221) 
ds_read_b128 v[88:91], v5 offset:38400 // (op 901) Load local data 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[72:77], v[94:99], a[4:7], v84, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 316) 
ds_read_b64 v[92:93], v5 offset:38416 // (op 901) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[72:77], v[100:105], a[8:11], v84, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 452) 
ds_read_b128 v[116:119], v5 offset:39936 // (op 1511) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[72:77], v[110:115], a[12:15], v84, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 588) 
ds_read_b64 v[120:121], v5 offset:39952 // (op 1511) Load local data 




s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[88:93], v[78:83], a[16:19], v85, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1008) 
ds_read_b128 v[72:75], v5 offset:41472 // (op 2121) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[88:93], v[100:105], a[24:27], v85, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 932) 
ds_read_b64 v[76:77], v5 offset:41488 // (op 2121) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[88:93], v[110:115], a[28:31], v85, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 894) 
s_barrier  // (op 7509) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[88:93], v[94:99], a[20:23], v85, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 970) 
s_waitcnt vmcnt(17)
ds_write_b128 v14, v[18:21] // (op 13) Store local data 

s_waitcnt vmcnt(16)
ds_write_b64 v14, v[22:23] offset:16 // (op 13) Store local data 
s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[116:121], v[78:83], a[32:35], v86, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1618) 
s_waitcnt vmcnt(15)
ds_write_b128 v14, v[24:27] offset:24 // (op 13) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[116:121], v[100:105], a[40:43], v86, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1542) 
s_waitcnt vmcnt(14)
ds_write_b64 v14, v[28:29] offset:40 // (op 13) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[116:121], v[110:115], a[44:47], v86, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1504) 
s_barrier  // (op 7387) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[116:121], v[94:99], a[36:39], v86, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1580) 




s_waitcnt vmcnt(13)
ds_write_b128 v12, v[30:33] offset:12288 // (op 6) Store local data 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[72:77], v[78:83], a[48:51], v87, v106 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2228) 
s_waitcnt vmcnt(12)
ds_write_b64 v12, v[34:35] offset:12304 // (op 6) Store local data 

s_waitcnt vmcnt(11)
ds_write_b128 v12, v[36:39] offset:12312 // (op 6) Store local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[72:77], v[100:105], a[56:59], v87, v108 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2152) 
s_waitcnt vmcnt(10)
ds_write_b64 v12, v[40:41] offset:12328 // (op 6) Store local data 

v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[72:77], v[110:115], a[60:63], v87, v109 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2114) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[72:77], v[94:99], a[52:55], v87, v107 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2190) 




s_ashr_i64 s[0:1], s[42:43], 7 // (op 2079) 
s_mov_b32 s18, s0 // (op 2079) convert

s_and_b32 s19, s18, 4294967294 // (op 2079) 

s_cmp_lt_u32 s47, s19 // (op 2079) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ForLoopTop_KLoop_2079 // (op 2079) Condition: Bottom (jump to RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ForLoopTop_KLoop_2079 if true)
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ForLoopBottom_KLoop_2079:













s_mul_hi_i32 s47, s16, s10 // (op 6793) 
v_lshrrev_b32 v10, 1, v1 // (op 6793) 
s_add_i32 s0, s47, s16 // (op 6793) 

v_and_b32 v11, -2, v1 // (op 6793) 
s_ashr_i32 s1, s0, 31 // (op 6793) 
v_sub_u32 v12, v1, v11 // (op 6793) 

s_cmp_eq_i32 s10, 0 // (op 6793) 
s_mov_b32 s47, scc // (op 6793) 
v_lshlrev_b32 v11, 6, v12 // (op 6793) 

s_cmp_eq_i32 s47, 0 // (op 6793) 
s_cselect_b32 s18, 0, -1 // (op 6793) 

s_and_b32 s19, s13, 31 // (op 6793) 
s_lshl_b32 s47, 1, s19 // (op 6793) 
s_add_i32 s22, s47, s18 // (op 6793) 


s_and_b32 s18, s1, s22 // (op 6793) 


s_add_i32 s1, s0, s18 // (op 6793) 


s_ashr_i32 s0, s1, s19 // (op 6793) 


s_lshl_b32 s1, s13, 24 // (op 6793) 
s_ashr_i32 s47, s1, 31 // (op 6793) 

s_xor_b32 s1, s0, s47 // (op 6793) 

s_sub_i32 s0, s1, s47 // (op 6793) 


s_mul_hi_i32 s1, s0, s14 // (op 6793) 
s_add_i32 s47, s1, s0 // (op 6793) 

s_ashr_i32 s1, s47, 31 // (op 6793) 
s_cmp_eq_i32 s14, 0 // (op 6793) 
s_mov_b32 s18, scc // (op 6793) 
s_cmp_eq_i32 s18, 0 // (op 6793) 
s_cselect_b32 s19, 0, -1 // (op 6793) 

s_and_b32 s18, s15, 31 // (op 6793) 
s_lshl_b32 s22, 1, s18 // (op 6793) 
s_add_i32 s23, s22, s19 // (op 6793) 


s_and_b32 s19, s1, s23 // (op 6793) 


s_add_i32 s1, s47, s19 // (op 6793) 


s_ashr_i32 s19, s1, s18 // (op 6793) 


s_lshl_b32 s1, s15, 24 // (op 6793) 
s_ashr_i32 s18, s1, 31 // (op 6793) 

s_xor_b32 s1, s19, s18 // (op 6793) 

s_sub_i32 s19, s1, s18 // (op 6793) 


s_mul_i32 s1, s0, s11 // (op 6793) 

s_sub_u32 s0, s16, s1 // (op 6793) 

s_mul_i32 s1, s19, s11 // (op 6793) 
s_add_u32 s1, s1, s0 // (op 6793) 


s_mul_hi_i32 s0, s16, s9 // (op 6793) 
s_add_i32 s47, s0, s16 // (op 6793) 

s_ashr_i32 s0, s47, 31 // (op 6793) 
s_cmp_eq_i32 s9, 0 // (op 6793) 
s_mov_b32 s18, scc // (op 6793) 
s_cmp_eq_i32 s18, 0 // (op 6793) 
s_cselect_b32 s19, 0, -1 // (op 6793) 

s_and_b32 s18, s45, 31 // (op 6793) 
s_lshl_b32 s22, 1, s18 // (op 6793) 
s_add_i32 s23, s22, s19 // (op 6793) 


s_and_b32 s19, s0, s23 // (op 6793) 


s_add_i32 s0, s47, s19 // (op 6793) 


s_ashr_i32 s19, s0, s18 // (op 6793) 


s_lshl_b32 s0, s45, 24 // (op 6793) 
s_ashr_i32 s18, s0, 31 // (op 6793) 

s_xor_b32 s0, s19, s18 // (op 6793) 

s_sub_i32 s19, s0, s18 // (op 6793) 


s_add_u32 s22, s6, 128 // (op 6793) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 6793) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 6793) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 6793) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 6793) 

s_mul_i32 s0, s11, s22 // (op 6793) 

s_mul_i32 s18, s8, s0 // (op 6793) 

s_mul_i32 s0, s19, s18 // (op 6793) 

s_sub_u32 s18, s16, s0 // (op 6793) 

s_mov_b32 s0, s18 // (op 6793) 
s_mul_hi_i32 s47, s0, s12 // (op 6793) 
s_add_i32 s22, s47, s0 // (op 6793) 


s_ashr_i32 s0, s22, 31 // (op 6793) 
s_cmp_eq_i32 s12, 0 // (op 6793) 
s_mov_b32 s47, scc // (op 6793) 
s_cmp_eq_i32 s47, 0 // (op 6793) 
s_cselect_b32 s23, 0, -1 // (op 6793) 

s_and_b32 s47, s44, 31 // (op 6793) 
s_lshl_b32 s24, 1, s47 // (op 6793) 
s_add_i32 s25, s24, s23 // (op 6793) 


s_and_b32 s23, s0, s25 // (op 6793) 


s_add_i32 s0, s22, s23 // (op 6793) 


s_ashr_i32 s22, s0, s47 // (op 6793) 


s_lshl_b32 s0, s44, 24 // (op 6793) 
s_ashr_i32 s47, s0, 31 // (op 6793) 

s_xor_b32 s0, s22, s47 // (op 6793) 

s_sub_i32 s22, s0, s47 // (op 6793) 


s_mul_i32 s0, s46, s11 // (op 6793) 
s_add_u32 s24, s20, 128 // (op 6793) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 6793) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 6793) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 6793) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 7 // (op 6793) 

s_sub_i32 s23, s24, s0 // (op 6793) 

s_mul_i32 s47, s22, s23 // (op 6793) 


s_sub_u32 s22, s18, s47 // (op 6793) 


s_add_u32 s18, s22, s0 // (op 6793) 


s_cmp_eq_i32 s19, 0 // (op 6793) 
s_mov_b32 s0, scc // (op 6793) 

s_cmp_eq_i32 s0, 0 // (op 6793) 
s_cselect_b32 s19, s18, s1 // (op 6793) 



v_lshl_add_u32 v12, s19, 7, v10 // (op 6793) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 6793) 
s_lshr_b32 s47, s0, 1 // (op 6793) 

v_lshl_add_u32 v10, s47, 8, v11 // (op 6793) 


v_mov_b32 v41, 0 // (op 6793) convert
v_mov_b32 v40, v10 // (op 6793) convert

v_mul_lo_u32 v11, v12, s27 // (op 6793) most significant: low of low * high
v_mul_hi_u32 v13, v12, s26 // (op 6793) most significant: high of low * low
v_mul_lo_u32 v10, v12, s26 // (op 6793) least significant: low of low * low
v_add_u32 v11, v11, v13 // (op 6793) most significant: sum

v_add_co_u32 v10, s[0:1], v10, v40 // (op 6793) least significant half
v_addc_co_u32 v11, s[0:1], v11, v41, s[0:1] // (op 6793) most significant half



v_mul_lo_u32 v13, v11, 6 // (op 6793) most significant: low of high * low
v_mul_hi_u32 v14, v10, 6 // (op 6793) most significant: high of low * low
v_mul_lo_u32 v12, v10, 6 // (op 6793) least significant: low of low * low
v_add_u32 v13, v13, v14 // (op 6793) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[12:13] // (op 6793) 

v_mov_b32 v12, v10 // (op 6793) convert

s_mul_i32 s1, s21, 6 // (op 6795) most significant: low of high * low
s_mul_i32 s0, s20, 6 // (op 6795) least significant
s_mul_hi_u32 s47, s20, 6 // (op 6795) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 6795) most significant: sum

s_mul_i32 s19, s26, s1 // (op 6795) most significant: low of low * high
s_mul_i32 s47, s27, s0 // (op 6795) most significant: low of high * low
s_mul_i32 s18, s26, s0 // (op 6795) least significant
s_mul_hi_u32 s22, s26, s0 // (op 6795) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 6795) most significant: sum
s_add_u32 s19, s19, s22 // (op 6795) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 6795) 

s_mov_b32 s48, s28 // (op 6795) reg expression
s_mov_b32 s49, s29 // (op 6795) reg expression
s_mov_b32 s50, s0 // (op 6795) reg expression
s_mov_b32 s51, 131072 // (op 6795) call()

s_mul_hi_i32 s47, s16, s10 // (op 6875) 
v_lshrrev_b32 v10, 1, v1 // (op 6875) 
s_add_i32 s0, s47, s16 // (op 6875) 

v_and_b32 v11, -2, v1 // (op 6875) 
s_ashr_i32 s1, s0, 31 // (op 6875) 
v_sub_u32 v13, v1, v11 // (op 6875) 

s_cmp_eq_i32 s10, 0 // (op 6875) 
s_mov_b32 s47, scc // (op 6875) 
v_lshlrev_b32 v11, 6, v13 // (op 6875) 

s_cmp_eq_i32 s47, 0 // (op 6875) 
s_cselect_b32 s18, 0, -1 // (op 6875) 

s_and_b32 s19, s13, 31 // (op 6875) 
s_lshl_b32 s47, 1, s19 // (op 6875) 
s_add_i32 s22, s47, s18 // (op 6875) 


s_and_b32 s18, s1, s22 // (op 6875) 


s_add_i32 s1, s0, s18 // (op 6875) 


s_ashr_i32 s0, s1, s19 // (op 6875) 


s_lshl_b32 s1, s13, 24 // (op 6875) 
s_ashr_i32 s47, s1, 31 // (op 6875) 

s_xor_b32 s1, s0, s47 // (op 6875) 

s_sub_i32 s0, s1, s47 // (op 6875) 


s_mul_hi_i32 s1, s0, s14 // (op 6875) 
s_add_i32 s47, s1, s0 // (op 6875) 

s_ashr_i32 s1, s47, 31 // (op 6875) 
s_cmp_eq_i32 s14, 0 // (op 6875) 
s_mov_b32 s18, scc // (op 6875) 
s_cmp_eq_i32 s18, 0 // (op 6875) 
s_cselect_b32 s19, 0, -1 // (op 6875) 

s_and_b32 s18, s15, 31 // (op 6875) 
s_lshl_b32 s22, 1, s18 // (op 6875) 
s_add_i32 s23, s22, s19 // (op 6875) 


s_and_b32 s19, s1, s23 // (op 6875) 


s_add_i32 s1, s47, s19 // (op 6875) 


s_ashr_i32 s19, s1, s18 // (op 6875) 


s_lshl_b32 s1, s15, 24 // (op 6875) 
s_ashr_i32 s18, s1, 31 // (op 6875) 

s_xor_b32 s1, s19, s18 // (op 6875) 

s_sub_i32 s19, s1, s18 // (op 6875) 


s_add_u32 s22, s6, 128 // (op 6875) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 6875) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 6875) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 6875) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 6875) 

s_mul_i32 s1, s19, s22 // (op 6875) 

s_sub_i32 s47, s0, s1 // (op 6875) 


s_mul_hi_i32 s0, s16, s9 // (op 6875) 
s_add_i32 s1, s0, s16 // (op 6875) 

s_ashr_i32 s0, s1, 31 // (op 6875) 
s_cmp_eq_i32 s9, 0 // (op 6875) 
s_mov_b32 s18, scc // (op 6875) 
s_cmp_eq_i32 s18, 0 // (op 6875) 
s_cselect_b32 s19, 0, -1 // (op 6875) 

s_and_b32 s18, s45, 31 // (op 6875) 
s_lshl_b32 s24, 1, s18 // (op 6875) 
s_add_i32 s25, s24, s19 // (op 6875) 


s_and_b32 s19, s0, s25 // (op 6875) 


s_add_i32 s0, s1, s19 // (op 6875) 


s_ashr_i32 s1, s0, s18 // (op 6875) 


s_lshl_b32 s0, s45, 24 // (op 6875) 
s_ashr_i32 s18, s0, 31 // (op 6875) 

s_xor_b32 s0, s1, s18 // (op 6875) 

s_sub_i32 s1, s0, s18 // (op 6875) 


s_mul_i32 s0, s11, s22 // (op 6875) 

s_mul_i32 s18, s8, s0 // (op 6875) 

s_mul_i32 s0, s1, s18 // (op 6875) 

s_sub_u32 s18, s16, s0 // (op 6875) 

s_mul_hi_i32 s0, s18, s12 // (op 6875) 
s_add_i32 s19, s0, s18 // (op 6875) 


s_ashr_i32 s0, s19, 31 // (op 6875) 
s_cmp_eq_i32 s12, 0 // (op 6875) 
s_mov_b32 s18, scc // (op 6875) 
s_cmp_eq_i32 s18, 0 // (op 6875) 
s_cselect_b32 s22, 0, -1 // (op 6875) 

s_and_b32 s18, s44, 31 // (op 6875) 
s_lshl_b32 s23, 1, s18 // (op 6875) 
s_add_i32 s24, s23, s22 // (op 6875) 


s_and_b32 s25, s0, s24 // (op 6875) 


s_add_i32 s0, s19, s25 // (op 6875) 


s_ashr_i32 s19, s0, s18 // (op 6875) 


s_lshl_b32 s0, s44, 24 // (op 6875) 
s_ashr_i32 s18, s0, 31 // (op 6875) 

s_xor_b32 s0, s19, s18 // (op 6875) 

s_sub_i32 s19, s0, s18 // (op 6875) 


s_cmp_eq_i32 s1, 0 // (op 6875) 
s_mov_b32 s0, scc // (op 6875) 

s_cmp_eq_i32 s0, 0 // (op 6875) 
s_cselect_b32 s1, s19, s47 // (op 6875) 



v_lshl_add_u32 v13, s1, 7, v10 // (op 6875) 


v_mul_lo_u32 v15, v13, s31 // (op 6875) most significant: low of low * high
v_mul_hi_u32 v10, v13, s30 // (op 6875) most significant: high of low * low
v_mul_lo_u32 v14, v13, s30 // (op 6875) least significant: low of low * low
v_add_u32 v15, v15, v10 // (op 6875) most significant: sum


s_ashr_i64 s[0:1], s[42:43], 7 // (op 6875) 
s_lshr_b32 s47, s0, 1 // (op 6875) 

v_lshl_add_u32 v10, s47, 8, v11 // (op 6875) 


v_mov_b32 v17, 0 // (op 6875) convert
v_mov_b32 v16, v10 // (op 6875) convert

v_add_co_u32 v10, s[0:1], v16, v14 // (op 6875) least significant half
v_addc_co_u32 v11, s[0:1], v17, v15, s[0:1] // (op 6875) most significant half



v_mul_lo_u32 v41, v11, 6 // (op 6875) most significant: low of high * low
v_mul_hi_u32 v13, v10, 6 // (op 6875) most significant: high of low * low
v_mul_lo_u32 v40, v10, 6 // (op 6875) least significant: low of low * low
v_add_u32 v41, v41, v13 // (op 6875) most significant: sum


v_lshrrev_b64 v[10:11], 3, v[40:41] // (op 6875) 

v_mov_b32 v13, v10 // (op 6875) convert

s_mul_i32 s1, s7, 6 // (op 6877) most significant: low of high * low
s_mul_i32 s0, s6, 6 // (op 6877) least significant
s_mul_hi_u32 s47, s6, 6 // (op 6877) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 6877) most significant: sum

s_mul_i32 s19, s30, s1 // (op 6877) most significant: low of low * high
s_mul_i32 s47, s31, s0 // (op 6877) most significant: low of high * low
s_mul_i32 s18, s30, s0 // (op 6877) least significant
s_mul_hi_u32 s22, s30, s0 // (op 6877) most significant: high of low * low
s_add_u32 s19, s19, s47 // (op 6877) most significant: sum
s_add_u32 s19, s19, s22 // (op 6877) most significant: sum



s_ashr_i64 s[0:1], s[18:19], 3 // (op 6877) 

s_mov_b32 s60, s32 // (op 6877) reg expression
s_mov_b32 s61, s33 // (op 6877) reg expression
s_mov_b32 s62, s0 // (op 6877) reg expression
s_mov_b32 s63, 131072 // (op 6877) call()

v_and_b32 v10, -2, v1 // (op 6958) 
v_sub_u32 v11, v1, v10 // (op 6958) 

v_lshlrev_b32 v10, 6, v11 // (op 6958) 

v_lshrrev_b32 v11, 1, v1 // (op 6958) 
v_lshl_add_u32 v14, v11, 7, v10 // (op 6958) 


v_mul_lo_u32 v10, v14, 6 // (op 6958) 

v_lshrrev_b32 v11, 3, v10 // (op 6958) 

v_and_b32 v10, -2, v1 // (op 7018) 
v_sub_u32 v14, v1, v10 // (op 7018) 

v_lshlrev_b32 v10, 6, v14 // (op 7018) 

v_lshrrev_b32 v14, 1, v1 // (op 7018) 
v_lshl_add_u32 v15, v14, 7, v10 // (op 7018) 


v_mul_lo_u32 v10, v15, 6 // (op 7018) 

v_lshrrev_b32 v14, 3, v10 // (op 7018) 

s_mul_hi_i32 s47, s16, s10 // (op 7267) 
v_and_b32 v10, 63, v1 // (op 7267) 
s_add_i32 s0, s47, s16 // (op 7267) 

v_lshrrev_b32 v15, 4, v10 // (op 7267) 
s_ashr_i32 s1, s0, 31 // (op 7267) 
v_and_b32 v16, 3, v15 // (op 7267) 

s_cmp_eq_i32 s10, 0 // (op 7267) 
s_mov_b32 s47, scc // (op 7267) 
v_and_b32 v15, 15, v1 // (op 7267) 
s_cmp_eq_i32 s47, 0 // (op 7267) 
s_cselect_b32 s18, 0, -1 // (op 7267) 

v_lshl_add_u32 v17, v16, 4, v15 // (op 7267) 


s_and_b32 s19, s13, 31 // (op 7267) 
v_lshrrev_b32 v15, 7, v1 // (op 7267) 
s_lshl_b32 s47, 1, s19 // (op 7267) 
v_lshl_add_u32 v16, v15, 6, v17 // (op 7267) 


s_add_i32 s22, s47, s18 // (op 7267) 


v_lshrrev_b32 v15, 6, v10 // (op 7267) 

s_and_b32 s18, s1, s22 // (op 7267) 


v_lshlrev_b32 v10, 2, v15 // (op 7267) 

s_add_i32 s1, s0, s18 // (op 7267) 


s_ashr_i32 s0, s1, s19 // (op 7267) 


s_lshl_b32 s1, s13, 24 // (op 7267) 
s_ashr_i32 s47, s1, 31 // (op 7267) 

s_xor_b32 s1, s0, s47 // (op 7267) 

s_sub_i32 s0, s1, s47 // (op 7267) 


s_mul_hi_i32 s1, s0, s14 // (op 7267) 
s_add_i32 s47, s1, s0 // (op 7267) 

s_ashr_i32 s1, s47, 31 // (op 7267) 
s_cmp_eq_i32 s14, 0 // (op 7267) 
s_mov_b32 s18, scc // (op 7267) 
s_cmp_eq_i32 s18, 0 // (op 7267) 
s_cselect_b32 s19, 0, -1 // (op 7267) 

s_and_b32 s18, s15, 31 // (op 7267) 
s_lshl_b32 s22, 1, s18 // (op 7267) 
s_add_i32 s23, s22, s19 // (op 7267) 


s_and_b32 s19, s1, s23 // (op 7267) 


s_add_i32 s1, s47, s19 // (op 7267) 


s_ashr_i32 s19, s1, s18 // (op 7267) 


s_lshl_b32 s1, s15, 24 // (op 7267) 
s_ashr_i32 s18, s1, 31 // (op 7267) 

s_xor_b32 s1, s19, s18 // (op 7267) 

s_sub_i32 s19, s1, s18 // (op 7267) 


s_mul_i32 s1, s0, s11 // (op 7267) 

s_sub_u32 s0, s16, s1 // (op 7267) 

s_mul_i32 s1, s19, s11 // (op 7267) 
s_add_u32 s1, s1, s0 // (op 7267) 


s_mul_hi_i32 s0, s16, s9 // (op 7267) 
s_add_i32 s47, s0, s16 // (op 7267) 

s_ashr_i32 s0, s47, 31 // (op 7267) 
s_cmp_eq_i32 s9, 0 // (op 7267) 
s_mov_b32 s18, scc // (op 7267) 
s_cmp_eq_i32 s18, 0 // (op 7267) 
s_cselect_b32 s19, 0, -1 // (op 7267) 

s_and_b32 s18, s45, 31 // (op 7267) 
s_lshl_b32 s22, 1, s18 // (op 7267) 
s_add_i32 s23, s22, s19 // (op 7267) 


s_and_b32 s19, s0, s23 // (op 7267) 


s_add_i32 s0, s47, s19 // (op 7267) 


s_ashr_i32 s19, s0, s18 // (op 7267) 


s_lshl_b32 s0, s45, 24 // (op 7267) 
s_ashr_i32 s18, s0, 31 // (op 7267) 

s_xor_b32 s0, s19, s18 // (op 7267) 

s_sub_i32 s19, s0, s18 // (op 7267) 


s_add_u32 s22, s6, 128 // (op 7267) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 7267) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 7267) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 7267) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 7267) 

s_mul_i32 s0, s11, s22 // (op 7267) 

s_mul_i32 s18, s8, s0 // (op 7267) 

s_mul_i32 s0, s19, s18 // (op 7267) 

s_sub_u32 s18, s16, s0 // (op 7267) 

s_mov_b32 s0, s18 // (op 7267) 
s_mul_hi_i32 s47, s0, s12 // (op 7267) 
s_add_i32 s22, s47, s0 // (op 7267) 


s_ashr_i32 s0, s22, 31 // (op 7267) 
s_cmp_eq_i32 s12, 0 // (op 7267) 
s_mov_b32 s47, scc // (op 7267) 
s_cmp_eq_i32 s47, 0 // (op 7267) 
s_cselect_b32 s23, 0, -1 // (op 7267) 

s_and_b32 s47, s44, 31 // (op 7267) 
s_lshl_b32 s24, 1, s47 // (op 7267) 
s_add_i32 s25, s24, s23 // (op 7267) 


s_and_b32 s23, s0, s25 // (op 7267) 


s_add_i32 s0, s22, s23 // (op 7267) 


s_ashr_i32 s22, s0, s47 // (op 7267) 


s_lshl_b32 s0, s44, 24 // (op 7267) 
s_ashr_i32 s47, s0, 31 // (op 7267) 

s_xor_b32 s0, s22, s47 // (op 7267) 

s_sub_i32 s22, s0, s47 // (op 7267) 


s_mul_i32 s0, s46, s11 // (op 7267) 
s_add_u32 s24, s20, 128 // (op 7267) Start of Int64 add, locking SCC
s_addc_u32 s25, s21, 0 // (op 7267) End of Int64 add, unlocking SCC
s_sub_u32 s64, s24, 1 // (op 7267) least significant half; sets scc
s_subb_u32 s65, s25, 0 // (op 7267) most significant half; uses scc

s_ashr_i64 s[24:25], s[64:65], 7 // (op 7267) 

s_sub_i32 s23, s24, s0 // (op 7267) 

s_mul_i32 s47, s22, s23 // (op 7267) 


s_sub_u32 s22, s18, s47 // (op 7267) 


s_add_u32 s18, s22, s0 // (op 7267) 


s_cmp_eq_i32 s19, 0 // (op 7267) 
s_mov_b32 s0, scc // (op 7267) 

s_cmp_eq_i32 s0, 0 // (op 7267) 
s_cselect_b32 s19, s18, s1 // (op 7267) 



v_lshl_add_u32 v15, s19, 7, v16 // (op 7267) 


s_ashr_i64 s[0:1], s[42:43], 7 // (op 7267) 
s_lshr_b32 s47, s0, 1 // (op 7267) 

v_lshl_add_u32 v16, s47, 3, v10 // (op 7267) 


v_mul_lo_u32 v10, v15, s34 // (op 7267) 
v_add_u32 v10, v10, v16 // (op 7267) 


s_mul_i32 s1, s34, s21 // (op 7269) most significant: low of low * high
s_mul_i32 s47, s35, s20 // (op 7269) most significant: low of high * low
s_mul_i32 s0, s34, s20 // (op 7269) least significant
s_mul_hi_u32 s18, s34, s20 // (op 7269) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 7269) most significant: sum
s_add_u32 s1, s1, s18 // (op 7269) most significant: sum


s_mov_b32 s64, s36 // (op 7269) reg expression
s_mov_b32 s65, s37 // (op 7269) reg expression
s_mov_b32 s66, s0 // (op 7269) reg expression
s_mov_b32 s67, 131072 // (op 7269) call()

s_mul_hi_i32 s47, s16, s10 // (op 7294) 
v_and_b32 v15, 63, v1 // (op 7294) 
s_add_i32 s0, s47, s16 // (op 7294) 

v_lshrrev_b32 v16, 4, v15 // (op 7294) 
s_ashr_i32 s1, s0, 31 // (op 7294) 
v_and_b32 v17, 3, v16 // (op 7294) 

s_cmp_eq_i32 s10, 0 // (op 7294) 
s_mov_b32 s47, scc // (op 7294) 
v_and_b32 v16, 15, v1 // (op 7294) 
s_cmp_eq_i32 s47, 0 // (op 7294) 
s_cselect_b32 s18, 0, -1 // (op 7294) 

v_lshl_add_u32 v18, v17, 4, v16 // (op 7294) 


s_and_b32 s19, s13, 31 // (op 7294) 
v_lshrrev_b32 v16, 6, v1 // (op 7294) 
s_lshl_b32 s47, 1, s19 // (op 7294) 
v_and_b32 v17, 1, v16 // (op 7294) 

s_add_i32 s22, s47, s18 // (op 7294) 


v_lshl_add_u32 v16, v17, 6, v18 // (op 7294) 


s_and_b32 s18, s1, s22 // (op 7294) 


v_lshrrev_b32 v17, 6, v15 // (op 7294) 

s_add_i32 s1, s0, s18 // (op 7294) 


v_lshlrev_b32 v15, 2, v17 // (op 7294) 

s_ashr_i32 s0, s1, s19 // (op 7294) 


s_lshl_b32 s1, s13, 24 // (op 7294) 
s_ashr_i32 s47, s1, 31 // (op 7294) 

s_xor_b32 s1, s0, s47 // (op 7294) 

s_sub_i32 s0, s1, s47 // (op 7294) 


s_mul_hi_i32 s1, s0, s14 // (op 7294) 
s_add_i32 s47, s1, s0 // (op 7294) 

s_ashr_i32 s1, s47, 31 // (op 7294) 
s_cmp_eq_i32 s14, 0 // (op 7294) 
s_mov_b32 s18, scc // (op 7294) 
s_cmp_eq_i32 s18, 0 // (op 7294) 
s_cselect_b32 s19, 0, -1 // (op 7294) 

s_and_b32 s18, s15, 31 // (op 7294) 
s_lshl_b32 s22, 1, s18 // (op 7294) 
s_add_i32 s23, s22, s19 // (op 7294) 


s_and_b32 s19, s1, s23 // (op 7294) 


s_add_i32 s1, s47, s19 // (op 7294) 


s_ashr_i32 s19, s1, s18 // (op 7294) 


s_lshl_b32 s1, s15, 24 // (op 7294) 
s_ashr_i32 s18, s1, 31 // (op 7294) 

s_xor_b32 s1, s19, s18 // (op 7294) 

s_sub_i32 s19, s1, s18 // (op 7294) 


s_add_u32 s22, s6, 128 // (op 7294) Start of Int64 add, locking SCC
s_addc_u32 s23, s7, 0 // (op 7294) End of Int64 add, unlocking SCC
s_sub_u32 s24, s22, 1 // (op 7294) least significant half; sets scc
s_subb_u32 s25, s23, 0 // (op 7294) most significant half; uses scc

s_ashr_i64 s[22:23], s[24:25], 7 // (op 7294) 

s_mul_i32 s1, s19, s22 // (op 7294) 

s_sub_i32 s47, s0, s1 // (op 7294) 


s_mul_hi_i32 s0, s16, s9 // (op 7294) 
s_add_i32 s1, s0, s16 // (op 7294) 

s_ashr_i32 s0, s1, 31 // (op 7294) 
s_cmp_eq_i32 s9, 0 // (op 7294) 
s_mov_b32 s18, scc // (op 7294) 
s_cmp_eq_i32 s18, 0 // (op 7294) 
s_cselect_b32 s19, 0, -1 // (op 7294) 

s_and_b32 s18, s45, 31 // (op 7294) 
s_lshl_b32 s24, 1, s18 // (op 7294) 
s_add_i32 s25, s24, s19 // (op 7294) 


s_and_b32 s19, s0, s25 // (op 7294) 


s_add_i32 s0, s1, s19 // (op 7294) 


s_ashr_i32 s1, s0, s18 // (op 7294) 


s_lshl_b32 s0, s45, 24 // (op 7294) 
s_ashr_i32 s18, s0, 31 // (op 7294) 

s_xor_b32 s0, s1, s18 // (op 7294) 

s_sub_i32 s1, s0, s18 // (op 7294) 


s_mul_i32 s0, s11, s22 // (op 7294) 

s_mul_i32 s18, s8, s0 // (op 7294) 

s_mul_i32 s0, s1, s18 // (op 7294) 

s_sub_u32 s18, s16, s0 // (op 7294) 

s_mul_hi_i32 s0, s18, s12 // (op 7294) 
s_add_i32 s19, s0, s18 // (op 7294) 


s_ashr_i32 s0, s19, 31 // (op 7294) 
s_cmp_eq_i32 s12, 0 // (op 7294) 
s_mov_b32 s18, scc // (op 7294) 
s_cmp_eq_i32 s18, 0 // (op 7294) 
s_cselect_b32 s22, 0, -1 // (op 7294) 

s_and_b32 s18, s44, 31 // (op 7294) 
s_lshl_b32 s23, 1, s18 // (op 7294) 
s_add_i32 s24, s23, s22 // (op 7294) 


s_and_b32 s25, s0, s24 // (op 7294) 


s_add_i32 s0, s19, s25 // (op 7294) 


s_ashr_i32 s19, s0, s18 // (op 7294) 


s_lshl_b32 s0, s44, 24 // (op 7294) 
s_ashr_i32 s18, s0, 31 // (op 7294) 

s_xor_b32 s0, s19, s18 // (op 7294) 

s_sub_i32 s19, s0, s18 // (op 7294) 


s_cmp_eq_i32 s1, 0 // (op 7294) 
s_mov_b32 s0, scc // (op 7294) 

s_cmp_eq_i32 s0, 0 // (op 7294) 
s_cselect_b32 s1, s19, s47 // (op 7294) 



v_lshl_add_u32 v17, s1, 7, v16 // (op 7294) 


v_mul_lo_u32 v16, v17, s38 // (op 7294) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 7294) 
s_lshr_b32 s47, s0, 1 // (op 7294) 

v_lshl_add_u32 v17, s47, 3, v15 // (op 7294) 


v_add_u32 v15, v17, v16 // (op 7294) 


s_mul_i32 s1, s38, s7 // (op 7296) most significant: low of low * high
s_mul_i32 s47, s39, s6 // (op 7296) most significant: low of high * low
s_mul_i32 s0, s38, s6 // (op 7296) least significant
s_mul_hi_u32 s18, s38, s6 // (op 7296) most significant: high of low * low
s_add_u32 s1, s1, s47 // (op 7296) most significant: sum
s_add_u32 s1, s1, s18 // (op 7296) most significant: sum


s_mov_b32 s68, s40 // (op 7296) reg expression
s_mov_b32 s69, s41 // (op 7296) reg expression
s_mov_b32 s70, s0 // (op 7296) reg expression
s_mov_b32 s71, 131072 // (op 7296) call()

s_ashr_i64 s[0:1], s[42:43], 7 // (op 157) 
s_and_b32 s47, s0, 4294967294 // (op 157) 

s_ashr_i64 s[0:1], s[42:43], 7 // (op 156) 
s_mov_b32 s18, s0 // (op 156) convert

s_cmp_lt_u32 s47, s18 // (op 156) 





s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc0 RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ForLoopBottom_KLoopTail_156 // (op 156) Condition: Top (jump to RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ForLoopBottom_KLoopTail_156 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ForLoopTop_KLoopTail_156:


s_add_u32 s47, s47, 1 // (op 158) 
v_mov_b32 v16, v12 // (op 161) 

buffer_load_dwordx4 v[18:21], v16, s[48:51], 0 offen // (op 161) Load value
buffer_load_dwordx2 v[22:23], v16, s[48:51], 0 offen offset:16 // (op 161) Load value
buffer_load_dwordx4 v[24:27], v16, s[48:51], 0 offen offset:24 // (op 161) Load value
buffer_load_dwordx2 v[28:29], v16, s[48:51], 0 offen offset:40 // (op 161) Load value

v_add_u32 v12, 96, v12 // (op 6783) 


s_barrier  // (op 7390) 
v_mov_b32 v16, v13 // (op 173) 




buffer_load_dwordx4 v[30:33], v16, s[60:63], 0 offen // (op 173) Load value
buffer_load_dwordx2 v[34:35], v16, s[60:63], 0 offen offset:16 // (op 173) Load value
s_waitcnt vmcnt(5)
ds_write_b128 v11, v[18:21] offset:12288 // (op 162) Store local data 
s_waitcnt vmcnt(4)
ds_write_b64 v11, v[22:23] offset:12304 // (op 162) Store local data 
s_waitcnt vmcnt(3)
ds_write_b128 v11, v[24:27] offset:12312 // (op 162) Store local data 
s_waitcnt vmcnt(2)
ds_write_b64 v11, v[28:29] offset:12328 // (op 162) Store local data 
buffer_load_dwordx4 v[36:39], v16, s[60:63], 0 offen offset:24 // (op 173) Load value

buffer_load_dwordx2 v[40:41], v16, s[60:63], 0 offen offset:40 // (op 173) Load value

v_add_u32 v13, 96, v13 // (op 6867) 
s_barrier  // (op 7403) 
s_waitcnt lgkmcnt(0)
s_barrier  // (op 7393) 






ds_read_b128 v[16:19], v3 offset:12288 // (op 164) Load local data 
ds_read_b64 v[20:21], v3 offset:12304 // (op 164) Load local data 
s_waitcnt vmcnt(3)
ds_write_b128 v14, v[30:33] // (op 172) Store local data 
s_waitcnt vmcnt(2)
ds_write_b64 v14, v[34:35] offset:16 // (op 172) Store local data 
s_waitcnt vmcnt(1)
ds_write_b128 v14, v[36:39] offset:24 // (op 172) Store local data 
s_waitcnt vmcnt(0)
ds_write_b64 v14, v[40:41] offset:40 // (op 172) Store local data 

s_waitcnt lgkmcnt(0)
s_barrier  // (op 7407) 



ds_read_b128 v[22:25], v4 // (op 166) Load local data 
ds_read_b64 v[26:27], v4 offset:16 // (op 166) Load local data 


v_mov_b32 v28, v10 // (op 168) 
v_mov_b32 v29, v28 // (op 168) reg expression

buffer_load_dwordx2 v[30:31], v29, s[64:67], 0 offen // (op 168) Load value


v_add_u32 v10, 4, v10 // (op 7257) 


v_mov_b32 v28, v15 // (op 170) 
v_mov_b32 v29, v28 // (op 170) reg expression

buffer_load_dwordx2 v[32:33], v29, s[68:71], 0 offen // (op 170) Load value


v_add_u32 v15, 4, v15 // (op 7284) 
s_waitcnt vmcnt(1)
v_bfe_u32 v34, v30, 0, 8 // (op 6241) 



v_bfe_u32 v35, v30, 8, 8 // (op 6241) 
v_bfe_u32 v36, v30, 16, 8 // (op 6241) 
v_bfe_u32 v37, v30, 24, 8 // (op 6241) 
v_permlane16_swap_b32 v34, v35 // (op 6241) 
ds_read_b128 v[38:41], v4 offset:1536 // (op 295) Load local data 
v_permlane16_swap_b32 v36, v37 // (op 6241) 
s_nop 1
v_permlane32_swap_b32 v34, v36 // (op 6241) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v35, v37 // (op 6241) 
ds_read_b64 v[42:43], v4 offset:1552 // (op 295) Load local data 




ds_read_b128 v[44:47], v4 offset:3072 // (op 431) Load local data 
ds_read_b64 v[48:49], v4 offset:3088 // (op 431) Load local data 
s_waitcnt vmcnt(0)
v_bfe_u32 v28, v32, 0, 8 // (op 6271) 



v_bfe_u32 v29, v32, 8, 8 // (op 6271) 
v_bfe_u32 v30, v32, 16, 8 // (op 6271) 
v_bfe_u32 v31, v32, 24, 8 // (op 6271) 
v_permlane16_swap_b32 v28, v29 // (op 6271) 
s_nop 0
v_permlane16_swap_b32 v30, v31 // (op 6271) Wait state hazard: VALU Write Readlane Hazard
s_nop 1
v_permlane32_swap_b32 v28, v30 // (op 6271) Wait state hazard: VALU Write Readlane Hazard
v_permlane32_swap_b32 v29, v31 // (op 6271) 
ds_read_b128 v[50:53], v4 offset:4608 // (op 567) Load local data 

ds_read_b64 v[54:55], v4 offset:4624 // (op 567) Load local data 



s_waitcnt lgkmcnt(6)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[16:21], v[22:27], a[0:3], v34, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 171) 
ds_read_b128 v[56:59], v3 offset:13824 // (op 789) Load local data 
s_waitcnt lgkmcnt(5)
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[16:21], v[38:43], a[4:7], v34, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 298) 
ds_read_b64 v[60:61], v3 offset:13840 // (op 789) Load local data 



s_waitcnt lgkmcnt(4)
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[16:21], v[44:49], a[8:11], v34, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 434) 
ds_read_b128 v[62:65], v3 offset:15360 // (op 1399) Load local data 
s_waitcnt lgkmcnt(3)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[16:21], v[50:55], a[12:15], v34, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 570) 
ds_read_b64 v[66:67], v3 offset:15376 // (op 1399) Load local data 




s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[56:61], v[22:27], a[16:19], v35, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 793) 
ds_read_b128 v[16:19], v3 offset:16896 // (op 2009) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[56:61], v[44:49], a[24:27], v35, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 847) 
ds_read_b64 v[20:21], v3 offset:16912 // (op 2009) Load local data 
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[56:61], v[50:55], a[28:31], v35, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 874) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[56:61], v[38:43], a[20:23], v35, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 820) 

s_waitcnt lgkmcnt(2)
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[62:67], v[22:27], a[32:35], v36, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1403) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[62:67], v[44:49], a[40:43], v36, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1457) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[62:67], v[50:55], a[44:47], v36, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1484) 
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[62:67], v[38:43], a[36:39], v36, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 1430) 

s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[16:21], v[22:27], a[48:51], v37, v28 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2013) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[16:21], v[44:49], a[56:59], v37, v30 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2067) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[16:21], v[50:55], a[60:63], v37, v31 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2094) 

v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[16:21], v[38:43], a[52:55], v37, v29 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b010 blgp:0b010 // (op 2040) 




s_ashr_i64 s[0:1], s[42:43], 7 // (op 156) 
s_mov_b32 s18, s0 // (op 156) convert

s_cmp_lt_u32 s47, s18 // (op 156) 

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_scc1 RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ForLoopTop_KLoopTail_156 // (op 156) Condition: Bottom (jump to RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ForLoopTop_KLoopTail_156 if true)
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ForLoopBottom_KLoopTail_156:




















s_barrier  // (op 7461) 

















v_cmp_eq_f32 vcc, 0.00000, v0 // (op 116) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into conditional label!
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
s_cbranch_vccz RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ConditionalFalse_0_beta_116 // (op 116) Condition: False, jump to RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ConditionalFalse_0_beta_116
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
s_branch RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ConditionalBottom_0_beta_116 // (op 116) Condition: Done, jump to RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ConditionalBottom_0_beta_116
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ConditionalFalse_0_beta_116:


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
RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__ConditionalBottom_0_beta_116:
























s_endpgm  // End of RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM_
.LRR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__end:
.size RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM_, .LRR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM__end-RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM_
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM_
  .amdhsa_next_free_vgpr 192
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 49152 // lds bytes
  .amdhsa_accum_offset 128
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
  - .name: RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM_
    .symbol: RR_GEMM_TN_FP6_FP6_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_128x128x128_UR_2_WGM_.kd
    .kernarg_segment_size: 172
    .group_segment_fixed_size: 49152
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 72
    .vgpr_count: 126
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

